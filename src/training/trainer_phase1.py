from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.template_qwen import (
    render_qwen_generation_prompt,
    render_qwen_supervised_text,
    strip_qwen_thinking_content,
)
from src.phase_b.hidden_states import load_hidden_state_split
from src.training.eval_utils import (
    judge_harmful_response_safety,
    looks_like_refusal,
    mean,
)
from src.training.losses import cosine_layer_alignment_loss
from src.utils.io import ensure_dir, read_jsonl, write_json


def load_student_target_map(target_dir: str | Path) -> tuple[Dict[str, Dict[int, torch.Tensor]], List[int]]:
    """Load per-pair recomposed safety targets emitted by 08_recompose.

    Returns ``(target_map, pair_keys)`` where ``target_map[sample_id][pair_idx]``
    is the K-th recomposed target tensor for that sample and ``pair_keys`` is
    the sorted list of pair indices [0, 1, ..., K-1]. Per the user's option-a
    setting in 08, each pair_idx is one independent alignment term; multiple
    pair_idx values may share the same student_layer (cross-scale collision).
    The pair_idx -> (teacher_layer, student_layer) resolution lives in the
    pairing file from 04 and the per-shard ``pair_index_to_pair`` metadata
    written by 08; load it via ``load_pair_to_student_layer`` separately.
    """

    target_path = Path(target_dir)
    part_paths = sorted(target_path.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No student target shards found under: {target_path}")

    target_map: Dict[str, Dict[int, torch.Tensor]] = {}
    pair_keys: set[int] = set()
    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu", weights_only=True)
        if "student_safe_target_by_pair" not in payload:
            raise KeyError(
                f"Target shard {part_path} lacks 'student_safe_target_by_pair'. "
                "Re-run 08_recompose_student_targets.py to produce per-pair "
                "targets (option-a layout)."
            )
        sample_ids = [str(sample_id) for sample_id in payload["sample_ids"]]
        for pair_text, tensor in payload["student_safe_target_by_pair"].items():
            pair_idx = int(pair_text)
            pair_keys.add(pair_idx)
            for row_idx, sample_id in enumerate(sample_ids):
                target_map.setdefault(sample_id, {})[pair_idx] = tensor[row_idx].to(dtype=torch.float32)
    return target_map, sorted(pair_keys)


def load_pair_to_student_layer(pairing_path: str | Path) -> Dict[int, int]:
    """Read 04's pairing file and return ``{pair_idx: student_layer}``."""

    payload = json.loads(Path(pairing_path).read_text(encoding="utf-8"))
    pairs = list(payload["pairs"])
    return {idx: int(pair["student_layer"]) for idx, pair in enumerate(pairs)}


def load_student_anchor_map(
    hidden_dir: str | Path,
    *,
    layer_ids: Sequence[int],
) -> Dict[str, Dict[int, torch.Tensor]]:
    hidden_split = load_hidden_state_split(hidden_dir, selected_layers=[int(layer_idx) for layer_idx in layer_ids])
    anchor_map: Dict[str, Dict[int, torch.Tensor]] = {}
    for row_idx, sample_id in enumerate(hidden_split.sample_ids):
        sample_layers: Dict[int, torch.Tensor] = {}
        for layer_idx in layer_ids:
            sample_layers[int(layer_idx)] = hidden_split.layer_tensors[int(layer_idx)][row_idx].to(dtype=torch.float32)
        anchor_map[str(sample_id)] = sample_layers
    return anchor_map


def summarize_target_map(target_map: Dict[str, Dict[int, torch.Tensor]]) -> Dict[str, Any]:
    layer_norms: Dict[int, List[float]] = {}
    for layer_targets in target_map.values():
        for layer_idx, tensor in layer_targets.items():
            layer_norms.setdefault(int(layer_idx), []).append(float(tensor.to(dtype=torch.float32).norm().item()))

    layer_stats: Dict[str, Dict[str, float]] = {}
    for layer_idx, norms in sorted(layer_norms.items()):
        norm_tensor = torch.tensor(norms, dtype=torch.float32)
        layer_stats[str(layer_idx)] = {
            "count": int(norm_tensor.numel()),
            "mean_l2_norm": float(norm_tensor.mean().item()),
            "std_l2_norm": float(norm_tensor.std(unbiased=False).item()) if norm_tensor.numel() > 1 else 0.0,
        }

    return {
        "num_samples": int(len(target_map)),
        "num_layers": int(len(layer_stats)),
        "layer_stats": layer_stats,
    }


def build_random_target_map(
    target_map: Dict[str, Dict[int, torch.Tensor]],
    *,
    seed: int,
    match_l2_norm: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Dict[int, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    randomized: Dict[str, Dict[int, torch.Tensor]] = {}

    for sample_id in sorted(target_map.keys()):
        randomized[sample_id] = {}
        for layer_idx in sorted(target_map[sample_id].keys()):
            target = target_map[sample_id][layer_idx].detach().cpu().to(dtype=torch.float32)
            random_vec = torch.randn(target.shape, generator=generator, dtype=torch.float32)
            random_vec = random_vec / random_vec.norm().clamp_min(eps)
            if match_l2_norm:
                target_norm = target.norm()
                if float(target_norm.item()) <= eps:
                    random_vec = torch.zeros_like(target)
                else:
                    random_vec = random_vec * target_norm
            randomized[sample_id][int(layer_idx)] = random_vec
    return randomized


class SemAlignDataset(Dataset):
    """Joins training records with student layer targets.

    Direction A (decoupled train_set vs alignment_set): records whose
    ``id`` is missing from ``target_map`` are still KEPT for SFT (L_out)
    supervision; they receive a zero placeholder target and a
    ``has_layer_target=False`` flag so the layer-alignment loss
    (L_layer) contributes zero weight for those rows. This implements
    the spec "L_layer driven by curated subspace (~5k), L_out trained
    on full data (~20k)" without silently dropping rows.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        target_map: Dict[str, Dict[int, torch.Tensor]],
        *,
        anchor_map: Dict[str, Dict[int, torch.Tensor]] | None = None,
        filter_harmful_targets: bool = False,
    ) -> None:
        self.anchor_map = anchor_map or {}
        self.filtered_harmful_target_count = 0
        self.missing_harmless_anchor_count = 0
        self.missing_layer_target_count = 0
        kept_records: List[Dict[str, Any]] = []
        has_layer_target_flags: List[bool] = []
        for record in records:
            sample_id = str(record["id"])
            has_target = sample_id in target_map
            if anchor_map is not None and str(record.get("label", "")) == "harmless" and sample_id not in anchor_map:
                self.missing_harmless_anchor_count += 1
                continue
            if filter_harmful_targets and str(record.get("label", "")) == "harmful":
                target_text = str(record.get("target_response") or record.get("rejected_response") or "")
                if not target_text.strip():
                    self.filtered_harmful_target_count += 1
                    continue
                judgment = judge_harmful_response_safety(target_text)
                if not looks_like_refusal(target_text) and not judgment.is_safe:
                    self.filtered_harmful_target_count += 1
                    continue
            if not has_target:
                self.missing_layer_target_count += 1
            kept_records.append(record)
            has_layer_target_flags.append(has_target)
        self.records = kept_records
        self.has_layer_target_flags = has_layer_target_flags
        self.target_map = target_map
        if not self.records:
            raise ValueError("No records remain after joining against student targets.")
        if anchor_map is not None and self.missing_harmless_anchor_count > 0:
            raise ValueError(
                "Missing harmless base anchors for "
                f"{self.missing_harmless_anchor_count} records after joining targets."
            )
        # Build a zero-tensor placeholder that matches any real target's
        # per-pair shape. Direction A: records without a target_map entry
        # still need a tensor of the right shape so the collator can stack;
        # the zero weight in _layer_sample_weights makes L_layer ignore them.
        self._placeholder_targets: Dict[int, torch.Tensor] = {}
        any_target = next(iter(target_map.values())) if target_map else None
        if any_target is not None:
            for pair_idx, tensor in any_target.items():
                self._placeholder_targets[int(pair_idx)] = torch.zeros_like(tensor)
        if self.missing_layer_target_count > 0 and not self._placeholder_targets:
            raise ValueError(
                "target_map is empty but records are missing targets; cannot "
                "construct placeholder tensors for SFT-only rows."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = str(record["id"])
        has_target = self.has_layer_target_flags[idx]
        targets = self.target_map[sample_id] if has_target else self._placeholder_targets
        return {
            "record": record,
            "targets": targets,
            "anchors": self.anchor_map.get(sample_id),
            "has_layer_target": has_target,
        }


@dataclass
class BatchPayload:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_last_positions: torch.Tensor
    layer_targets: Dict[int, torch.Tensor]
    layer_anchors: Dict[int, torch.Tensor] | None
    sample_ids: List[str]
    labels_text: List[str]
    messages: List[Sequence[Dict[str, str]]]
    # Direction A: bool tensor [B], False = SFT-only row (no L_layer
    # contribution because the record's id is not in the alignment-set
    # target map). True = full L_layer + L_out supervision.
    has_layer_target: torch.Tensor


class SemAlignCollator:
    """Tokenize supervised batches and mark the first-generated-token position.

    ``prompt_last_positions`` is the index of the last token of the generation-
    prompt rendering (``apply_chat_template(..., add_generation_prompt=True)``),
    which is the same position used by ``01_extract_hidden_states.py`` to cache
    teacher/student "first generated token" hidden states. The supervision mask
    sets labels to ``-100`` on the prompt span so only the assistant response
    contributes to ``L_out``.

    ``layer_ids`` is the list of pair indices (one per row of the pairing file
    from 04). ``pair_to_student_layer`` resolves each pair_idx back to its real
    student layer; the collator uses it when stacking per-pair anchors keyed by
    the actual student layer in ``anchor_dict``.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        layer_ids: Sequence[int],
        pair_to_student_layer: Dict[int, int],
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.layer_ids = [int(layer_idx) for layer_idx in layer_ids]
        self.pair_to_student_layer = {int(k): int(v) for k, v in pair_to_student_layer.items()}
        missing = [pair_idx for pair_idx in self.layer_ids if pair_idx not in self.pair_to_student_layer]
        if missing:
            raise KeyError(
                f"pair_to_student_layer is missing entries for pair_idx values: {missing}"
            )

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> BatchPayload:
        records = [item["record"] for item in batch]
        target_dicts = [item["targets"] for item in batch]
        anchor_dicts = [item.get("anchors") for item in batch]
        prompt_texts = [
            render_qwen_generation_prompt(self.tokenizer, record["messages"])
            for record in records
        ]
        full_texts = [
            render_qwen_supervised_text(
                self.tokenizer,
                record["messages"],
                str(record.get("target_response") or record.get("accept_response") or ""),
            )
            for record in records
        ]

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        encoded_full = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded_prompt = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        self.tokenizer.padding_side = previous_padding_side

        prompt_lengths = encoded_prompt["attention_mask"].sum(dim=1).to(dtype=torch.long)
        prompt_last_positions = torch.clamp(prompt_lengths - 1, min=0)

        labels = encoded_full["input_ids"].clone()
        for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[row_idx, :prompt_len] = -100

        layer_targets = {
            pair_idx: torch.stack([target_dict[pair_idx] for target_dict in target_dicts], dim=0)
            for pair_idx in self.layer_ids
        }
        layer_anchors = None
        if any(anchor_dict is not None for anchor_dict in anchor_dicts):
            layer_anchors = {}
            for pair_idx in self.layer_ids:
                student_layer = self.pair_to_student_layer[pair_idx]
                layer_anchors[pair_idx] = torch.stack(
                    [
                        (
                            anchor_dict[student_layer]
                            if anchor_dict is not None and student_layer in anchor_dict
                            else target_dicts[row_idx][pair_idx]
                        )
                        for row_idx, anchor_dict in enumerate(anchor_dicts)
                    ],
                    dim=0,
                )
        has_layer_target = torch.tensor(
            [bool(item.get("has_layer_target", True)) for item in batch],
            dtype=torch.bool,
        )
        return BatchPayload(
            input_ids=encoded_full["input_ids"],
            attention_mask=encoded_full["attention_mask"],
            labels=labels,
            prompt_last_positions=prompt_last_positions,
            layer_targets=layer_targets,
            layer_anchors=layer_anchors,
            sample_ids=[str(record["id"]) for record in records],
            labels_text=[str(record["label"]) for record in records],
            messages=[record["messages"] for record in records],
            has_layer_target=has_layer_target,
        )


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn: Any,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def _capture_layer_outputs(
    model: nn.Module,
    *,
    layer_ids: Sequence[int],
    prompt_last_positions: torch.Tensor,
    cache: Dict[int, torch.Tensor],
    pair_to_student_layer: Dict[int, int],
):
    """Register one forward hook per pair_idx. The hook target is the actual
    student transformer block ``pair_to_student_layer[pair_idx]``; multiple
    pair_idx values that share the same student layer end up registering
    multiple hooks on that same block, each writing the same hidden tensor to
    its own ``cache[pair_idx]`` slot. This gives every pair its own predicted
    tensor and keeps the K-term alignment loss correct under collisions."""

    hooks = []
    for pair_idx in layer_ids:
        student_layer = int(pair_to_student_layer[int(pair_idx)])
        layer = model.model.layers[student_layer]

        def hook(_module, _inputs, output, current_pair_idx=int(pair_idx)):
            hidden = output[0] if isinstance(output, tuple) else output
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            selected = hidden[batch_indices, prompt_last_positions.to(hidden.device), :]
            cache[current_pair_idx] = selected
            return output

        hooks.append(layer.register_forward_hook(hook))
    return hooks


def _build_layer_targets_for_policy(
    batch: BatchPayload,
    *,
    device: torch.device,
    layer_ids: Sequence[int],
    layer_loss_policy: str,
) -> Dict[int, torch.Tensor]:
    policy = str(layer_loss_policy).strip().lower()
    target_by_layer = {
        int(layer_idx): tensor.to(device)
        for layer_idx, tensor in batch.layer_targets.items()
    }
    if policy != "harmless_anchor":
        return target_by_layer

    if batch.layer_anchors is None:
        raise ValueError("layer_loss_policy='harmless_anchor' requires layer_anchors in the batch.")
    harmless_mask = torch.tensor(
        [str(label) == "harmless" for label in batch.labels_text],
        device=device,
        dtype=torch.bool,
    )
    if not bool(harmless_mask.any().detach().cpu().item()):
        return target_by_layer

    mixed_targets: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_ids:
        layer_idx = int(layer_idx)
        if layer_idx not in batch.layer_anchors:
            raise KeyError(f"Missing harmless anchor tensor for layer {layer_idx}.")
        semantic_target = target_by_layer[layer_idx]
        anchor_target = batch.layer_anchors[layer_idx].to(device=device, dtype=semantic_target.dtype)
        mixed = semantic_target.clone()
        mixed[harmless_mask] = anchor_target[harmless_mask]
        mixed_targets[layer_idx] = mixed
    return mixed_targets


def forward_semalign_batch(
    model: nn.Module,
    batch: BatchPayload,
    *,
    device: torch.device,
    layer_ids: Sequence[int],
    pair_to_student_layer: Dict[int, int],
    layer_loss_weight: float,
    sft_loss_weight: float = 1.0,
    layer_loss_policy: str = "all",
    harmful_layer_weight: float = 1.0,
    harmless_layer_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    inputs = {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
        "labels": batch.labels.to(device),
    }
    prompt_last_positions = batch.prompt_last_positions.to(device)
    cache: Dict[int, torch.Tensor] = {}
    hooks = _capture_layer_outputs(
        model,
        layer_ids=layer_ids,
        prompt_last_positions=prompt_last_positions,
        cache=cache,
        pair_to_student_layer=pair_to_student_layer,
    )
    try:
        outputs = model(**inputs, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()

    predicted_by_layer = {int(layer_idx): cache[int(layer_idx)] for layer_idx in layer_ids}
    target_by_layer = _build_layer_targets_for_policy(
        batch,
        device=device,
        layer_ids=layer_ids,
        layer_loss_policy=layer_loss_policy,
    )
    sample_weights = _layer_sample_weights(
        batch.labels_text,
        device=device,
        layer_loss_policy=layer_loss_policy,
        harmful_layer_weight=harmful_layer_weight,
        harmless_layer_weight=harmless_layer_weight,
        has_layer_target=batch.has_layer_target,
    )
    loss_layer, cosine_by_layer = cosine_layer_alignment_loss(
        predicted_by_layer,
        target_by_layer,
        sample_weights=sample_weights,
    )
    loss_out = outputs.loss
    loss_total = (sft_loss_weight * loss_out) + (layer_loss_weight * loss_layer)
    active_layer_weight_sum = (
        float(batch.input_ids.size(0))
        if sample_weights is None
        else float(sample_weights.detach().sum().cpu().item())
    )
    metrics = {
        "loss_total": float(loss_total.detach().cpu().item()),
        "loss_out": float(loss_out.detach().cpu().item()),
        "loss_layer": float(loss_layer.detach().cpu().item()),
        "layer_target_cosine_mean": mean(cosine_by_layer.values()),
        "active_layer_weight_sum": active_layer_weight_sum,
    }
    return loss_total, metrics


def _layer_sample_weights(
    labels_text: Sequence[str],
    *,
    device: torch.device,
    layer_loss_policy: str,
    harmful_layer_weight: float,
    harmless_layer_weight: float,
    has_layer_target: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute per-row L_layer weights.

    Direction A: rows with ``has_layer_target=False`` are forced to
    weight 0.0 in BOTH harmful and harmless branches, so SFT-only rows
    do not contribute to the layer alignment loss. When the policy is
    ``'all'`` but at least one row lacks a target, we still return an
    explicit weight tensor (rather than ``None``) so the loss masks
    those rows out.
    """

    policy = str(layer_loss_policy).strip().lower()
    if policy not in {"all", "harmful_only", "label_weighted", "harmless_anchor"}:
        raise ValueError(
            f"Unsupported layer_loss_policy: {layer_loss_policy}. "
            "Expected 'all', 'harmful_only', 'label_weighted', or 'harmless_anchor'."
        )
    has_target_list = (
        [bool(flag) for flag in has_layer_target.detach().cpu().tolist()]
        if has_layer_target is not None
        else [True] * len(labels_text)
    )
    any_missing = not all(has_target_list)
    if policy == "all" and not any_missing:
        return None
    weights = []
    for label, has_target in zip(labels_text, has_target_list):
        if not has_target:
            weights.append(0.0)
            continue
        if policy == "all":
            weights.append(1.0)
        elif str(label) == "harmful":
            weights.append(float(harmful_layer_weight))
        elif str(label) == "harmless":
            weights.append(0.0 if policy == "harmful_only" else float(harmless_layer_weight))
        else:
            weights.append(0.0)
    return torch.tensor(weights, device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate_layer_alignment(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    layer_ids: Sequence[int],
    pair_to_student_layer: Dict[int, int],
    layer_loss_policy: str = "all",
    harmful_layer_weight: float = 1.0,
    harmless_layer_weight: float = 1.0,
) -> float:
    cosine_scores: List[float] = []
    model.eval()
    for batch in dataloader:
        inputs = {
            "input_ids": batch.input_ids.to(device),
            "attention_mask": batch.attention_mask.to(device),
        }
        prompt_last_positions = batch.prompt_last_positions.to(device)
        cache: Dict[int, torch.Tensor] = {}
        hooks = _capture_layer_outputs(
            model,
            layer_ids=layer_ids,
            prompt_last_positions=prompt_last_positions,
            cache=cache,
            pair_to_student_layer=pair_to_student_layer,
        )
        try:
            model(**inputs, use_cache=False, return_dict=True)
        finally:
            for hook in hooks:
                hook.remove()

        target_by_layer = _build_layer_targets_for_policy(
            batch,
            device=device,
            layer_ids=layer_ids,
            layer_loss_policy=layer_loss_policy,
        )
        sample_weights = _layer_sample_weights(
            batch.labels_text,
            device=device,
            layer_loss_policy=layer_loss_policy,
            harmful_layer_weight=harmful_layer_weight,
            harmless_layer_weight=harmless_layer_weight,
            has_layer_target=batch.has_layer_target,
        )
        for layer_idx in layer_ids:
            predicted = cache[int(layer_idx)]
            target = target_by_layer[int(layer_idx)]
            cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
            if sample_weights is not None:
                keep_mask = sample_weights > 0
                if bool(keep_mask.any().detach().cpu().item()):
                    cosine = cosine[keep_mask]
                else:
                    continue
            cosine_scores.extend(cosine.detach().cpu().tolist())
    return mean(cosine_scores)


@torch.no_grad()
def evaluate_generation_refusal_metrics(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
    initial_max_new_tokens: int = 0,
) -> Dict[str, Any]:
    """PAN safety metrics with refusal/safe-non-refusal/unsafe split.

    Returns the original Phase-G binary fields (refusal_rate, unsafe_output_rate
    = 1 - refusal, over_refusal_rate) for backward compatibility, plus a
    ``judge_harmful_response_safety``-derived breakdown of the non-refusal
    bucket into ``safe_non_refusal_rate`` (boundary warning + redirect /
    concise boundary / preventive support) and ``strict_unsafe_rate``
    (actionable harmful guidance + ambiguous non-refusal). The strict rate
    is the metric to report when comparing PAN training against an untuned
    baseline whose harmful outputs frequently take a "this is sensitive,
    instead consider..." shape that the binary refusal regex would label
    unsafe but a 3-class judge would label safe.

    Reasoning preamble handling: Qwen3.5-9B and similar models emit
    plain-text "Here's a thinking process..." blocks even with
    chat_template_enable_thinking=False. ``strip_qwen_thinking_content``
    extracts the post-final-response section; when that returns empty we
    interpret it as "preamble truncated before final response was emitted"
    and, if ``initial_max_new_tokens`` is set below ``max_new_tokens``,
    re-generate the affected sample with the full ``max_new_tokens`` budget
    so the final response can appear.
    """

    harmful_total = 0
    harmless_total = 0
    harmful_refusals = 0
    harmful_unsafes = 0
    harmful_safe_non_refusal = 0
    harmful_strict_unsafe = 0
    harmless_refusals = 0
    num_preamble_retries = 0
    num_preamble_unresolved = 0
    generations: List[Dict[str, Any]] = []

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()

    full_max_new_tokens = int(max_new_tokens)
    requested_initial = int(initial_max_new_tokens)
    if requested_initial > 0:
        first_pass_max_new_tokens = max(1, min(requested_initial, full_max_new_tokens))
    else:
        first_pass_max_new_tokens = full_max_new_tokens
    can_retry = first_pass_max_new_tokens < full_max_new_tokens

    eos_ids: List[int] = []
    if isinstance(tokenizer.eos_token_id, (list, tuple)):
        eos_ids = [int(eid) for eid in tokenizer.eos_token_id if eid is not None]
    elif tokenizer.eos_token_id is not None:
        eos_ids = [int(tokenizer.eos_token_id)]

    def _generate_batch(prompts: Sequence[str], gen_max_new_tokens: int) -> tuple[list[str], list[int]]:
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        # PPU/NPU eager mode: no XLA graph step required.
        prompt_width = int(encoded["input_ids"].size(1))
        generated_only = generated[:, prompt_width:]
        decoded = [
            tokenizer.decode(generated_only[row_idx], skip_special_tokens=True)
            for row_idx in range(generated_only.size(0))
        ]
        # Real per-sample stop count: index of first EOS token (or full window
        # if model never emitted EOS within gen_max_new_tokens). The previous
        # implementation always reported gen_max_new_tokens for every row,
        # which masked legitimate EOS-emission diagnostics.
        used: list[int] = []
        for row_idx in range(generated_only.size(0)):
            row = generated_only[row_idx].tolist()
            stop_idx = -1
            if eos_ids:
                for token_pos, token_id in enumerate(row):
                    if token_id in eos_ids:
                        stop_idx = token_pos + 1
                        break
            used.append(int(stop_idx if stop_idx > 0 else len(row)))
        return decoded, used

    effective_batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(records), effective_batch_size):
        batch_records = list(records[batch_start: batch_start + effective_batch_size])
        prompt_texts = [
            render_qwen_generation_prompt(tokenizer, record["messages"])
            for record in batch_records
        ]
        decoded, used_max_new_tokens = _generate_batch(prompt_texts, first_pass_max_new_tokens)

        retry_indices: List[int] = []
        if can_retry:
            for row_idx, text in enumerate(decoded):
                cleaned = strip_qwen_thinking_content(text, require_final_response=True)
                # Empty cleaned + non-empty raw = preamble present but final
                # response section truncated. Worth retrying with full budget.
                if not cleaned and text.strip():
                    retry_indices.append(row_idx)

        if retry_indices:
            retry_prompts = [prompt_texts[i] for i in retry_indices]
            retry_decoded, retry_used = _generate_batch(retry_prompts, full_max_new_tokens)
            for retry_pos, sample_idx in enumerate(retry_indices):
                decoded[sample_idx] = retry_decoded[retry_pos]
                used_max_new_tokens[sample_idx] = retry_used[retry_pos]
                num_preamble_retries += 1

        for row_idx, record in enumerate(batch_records):
            text = decoded[row_idx]
            cleaned = strip_qwen_thinking_content(text, require_final_response=True)
            preamble_truncated = bool(text.strip()) and not cleaned
            if preamble_truncated:
                num_preamble_unresolved += 1
            final_text = cleaned if cleaned else text
            is_refusal = looks_like_refusal(final_text)
            judgment = judge_harmful_response_safety(final_text)
            label = str(record["label"])
            if label == "harmful":
                harmful_total += 1
                harmful_refusals += int(is_refusal)
                harmful_unsafes += int(not is_refusal)
                if not is_refusal:
                    if judgment.is_safe:
                        harmful_safe_non_refusal += 1
                    else:
                        harmful_strict_unsafe += 1
            elif label == "harmless":
                harmless_total += 1
                harmless_refusals += int(is_refusal)
            generations.append(
                {
                    "id": str(record["id"]),
                    "label": label,
                    "generated_text": text,
                    "final_text": final_text,
                    "is_refusal": bool(is_refusal),
                    "judge_reason": judgment.reason,
                    "judge_is_safe": bool(judgment.is_safe),
                    "preamble_unresolved": preamble_truncated,
                    "used_max_new_tokens": int(used_max_new_tokens[row_idx]),
                }
            )
    tokenizer.padding_side = previous_padding_side

    return {
        "harmful_refusal_rate": 0.0 if harmful_total == 0 else harmful_refusals / harmful_total,
        "harmful_unsafe_output_rate": 0.0 if harmful_total == 0 else harmful_unsafes / harmful_total,
        "harmful_safe_non_refusal_rate": 0.0 if harmful_total == 0 else harmful_safe_non_refusal / harmful_total,
        "harmful_strict_unsafe_rate": 0.0 if harmful_total == 0 else harmful_strict_unsafe / harmful_total,
        "harmless_over_refusal_rate": 0.0 if harmless_total == 0 else harmless_refusals / harmless_total,
        "num_harmful": harmful_total,
        "num_harmless": harmless_total,
        "num_preamble_retries": num_preamble_retries,
        "num_preamble_unresolved": num_preamble_unresolved,
        "generations": generations,
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: Dict[str, Any],
    save_mode: str = "trainable",
    save_optimizer: bool = True,
) -> None:
    """Persist a training checkpoint.

    ``save_mode='trainable'`` (default) writes ``trainable_state_dict`` --
    the historical LoRA-delta-only payload. ``save_mode='full'`` writes
    the complete ``model_state_dict`` for full fine-tuning runs. The
    eval / merge dispatch in ``src/baselines/eval.py`` and
    ``scripts/16_merge_lora_for_opencompass.py`` discriminates on the
    manifest's ``mode`` field, not on the payload key, so both layouts
    interoperate.

    ``save_optimizer=False`` drops the AdamW state, which can otherwise
    inflate full-fine-tune checkpoints by ~6 GB on a 0.8B model.
    """

    target = Path(checkpoint_path)
    ensure_dir(target.parent)
    if save_mode == "full":
        state = {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
        }
        payload_key = "model_state_dict"
    elif save_mode == "trainable":
        state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        payload_key = "trainable_state_dict"
    else:
        raise ValueError(
            f"Unsupported save_mode={save_mode!r}; expected 'trainable' or 'full'."
        )
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        payload_key: state,
        "save_mode": save_mode,
        "extra": extra,
    }
    if save_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, target)


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def write_train_metric(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_val_metrics(path: str | Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
