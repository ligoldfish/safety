"""Annotate records with teacher first-generated-token top-1 probability.

Used by Phase 1 curation: records whose teacher (Qwen3.5-9B) emits a
diffuse / uncertain next-token distribution at the first generation
position carry a noisy hidden state, and including them in the SVD
contrast pool drags the safety direction toward style rather than
intent. Records flagged with low confidence are filtered out by
``src/data/curation.py`` before SVD.

This module loads the teacher once, runs a single forward pass per
record (batched), and attaches ``metadata.teacher_first_gen_top1_prob``
in-place. It returns the same records — no filtering, no shuffling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.data.template_qwen import render_qwen_generation_prompt

_LOGGER = logging.getLogger(__name__)


def _build_messages(record: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    messages = list(record.get("messages") or [])
    if messages and any(str(m.get("role", "")).lower() == "user" for m in messages):
        return [m for m in messages if str(m.get("role", "")).lower() != "assistant"]
    user_text = str(record.get("user_text") or "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


@torch.no_grad()
def _batch_top1_probs(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[List[Dict[str, str]]],
    max_length: int,
) -> List[float]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    rendered = [render_qwen_generation_prompt(tokenizer=tokenizer, messages=m) for m in prompts]
    # Left-truncate so the trailing "<|im_start|>assistant\n" suffix — the
    # position whose next-token distribution we want to measure — is never
    # cut off when a prompt exceeds ``max_length``.
    _prev_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.truncation_side = _prev_truncation_side
    # Warn if any row hit ``max_length`` (i.e. was truncated). We treat any
    # row whose attended length equals ``max_length`` as a truncation hit;
    # this is a conservative over-estimate when a prompt is exactly the
    # limit, which is acceptable for an advisory log.
    try:
        _attended = encoded["attention_mask"].sum(dim=1)
        _truncated = int((_attended >= int(max_length)).sum().item())
        if _truncated > 0:
            _LOGGER.warning(
                "teacher_confidence: %d/%d prompts hit max_length=%d (left-truncated)",
                _truncated,
                int(_attended.shape[0]),
                int(max_length),
            )
    except Exception:  # noqa: BLE001 - logging must never break the forward pass
        pass
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(
        **encoded,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs.logits  # [B, T, V]
    attention_mask = encoded["attention_mask"]
    # Last non-padding position per row.
    reversed_mask = torch.flip(attention_mask.to(dtype=torch.long), dims=[1])
    from_end = torch.argmax(reversed_mask, dim=1)
    last_positions = attention_mask.size(1) - 1 - from_end
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    selected_logits = logits[batch_indices, last_positions, :]
    probs = torch.softmax(selected_logits.float(), dim=-1)
    top1 = probs.max(dim=-1).values.detach().cpu().tolist()
    return [float(value) for value in top1]


def annotate_teacher_confidence(
    records: Sequence[Dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 1024,
    system_prompt: str = "You are a helpful assistant.",
    progress_log_every: int = 200,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Stream over records, attach ``metadata.teacher_first_gen_top1_prob``.

    Returns the same records (mutated in place + returned for chaining).
    Records that already carry the field are skipped to keep re-runs
    idempotent.
    """

    out: List[Dict[str, Any]] = list(records)
    pending_indices: List[int] = []
    pending_messages: List[List[Dict[str, str]]] = []

    def flush() -> None:
        if not pending_indices:
            return
        probs = _batch_top1_probs(
            model=model,
            tokenizer=tokenizer,
            prompts=pending_messages,
            max_length=max_length,
        )
        for idx, prob in zip(pending_indices, probs):
            meta = out[idx].get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                out[idx]["metadata"] = meta
            meta["teacher_first_gen_top1_prob"] = float(prob)
        pending_indices.clear()
        pending_messages.clear()

    processed = 0
    for idx, record in enumerate(out):
        meta = record.get("metadata")
        if isinstance(meta, dict) and "teacher_first_gen_top1_prob" in meta:
            continue
        pending_indices.append(idx)
        pending_messages.append(_build_messages(record, system_prompt))
        if len(pending_indices) >= int(batch_size):
            flush()
            processed += int(batch_size)
            if progress_callback is not None and processed % int(progress_log_every) == 0:
                progress_callback(processed, len(out))
    flush()
    if progress_callback is not None:
        progress_callback(len(out), len(out))
    return out


def load_teacher_for_confidence(
    model_path: str | Path,
    *,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    runtime_backend: str = "",
    runtime_device: str = "",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "",
) -> tuple[Any, Any]:
    """Thin wrapper around ``src.models.hf_loader.load_hf_model``.

    Returns (tokenizer, model). Caller is responsible for freeing the
    model after annotation (``del model; torch.cuda.empty_cache()`` or
    equivalent NPU/PPU call).
    """

    from src.models.hf_loader import load_hf_model

    tokenizer, model, _meta = load_hf_model(
        str(model_path),
        device_map=device_map,
        torch_dtype=torch_dtype,
        runtime_backend=runtime_backend,
        runtime_device=runtime_device,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    return tokenizer, model
