from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from src.data.template_qwen import render_qwen_generation_prompt


def build_chat_batch(
    tokenizer: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    max_length: int,
    device: torch.device | str | None = None,
) -> tuple[Dict[str, torch.Tensor], List[str]]:
    prompt_texts = [
        render_qwen_generation_prompt(tokenizer=tokenizer, messages=messages)
        for messages in messages_batch
    ]
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded, prompt_texts


def _last_non_padding_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, T]")
    reversed_mask = torch.flip(attention_mask.to(dtype=torch.long), dims=[1])
    from_end = torch.argmax(reversed_mask, dim=1)
    return attention_mask.size(1) - 1 - from_end


def extract_last_position_hidden(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    skip_embedding_layer: bool = True,
) -> List[torch.Tensor]:
    if not hidden_states:
        raise ValueError("hidden_states must be non-empty")

    selected_states = list(hidden_states[1:] if skip_embedding_layer else hidden_states)
    last_positions = _last_non_padding_positions(attention_mask)
    extracted: List[torch.Tensor] = []
    for layer_hidden in selected_states:
        batch_indices = torch.arange(layer_hidden.size(0), device=layer_hidden.device)
        gathered = layer_hidden[batch_indices, last_positions.to(layer_hidden.device), :]
        extracted.append(gathered.detach())
    return extracted


@torch.no_grad()
def gather_first_generated_token_representations(
    model: Any,
    tokenizer: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    max_length: int,
) -> tuple[List[torch.Tensor], List[str], torch.Tensor]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    encoded, prompt_texts = build_chat_batch(
        tokenizer=tokenizer,
        messages_batch=messages_batch,
        max_length=max_length,
        device=device,
    )
    outputs = model(
        **encoded,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)
    if runtime_backend == "tpu" and xla_model is not None:
        xla_model.mark_step()
    layer_hiddens = extract_last_position_hidden(
        hidden_states=outputs.hidden_states,
        attention_mask=encoded["attention_mask"],
        skip_embedding_layer=True,
    )
    last_positions = _last_non_padding_positions(encoded["attention_mask"]).detach().cpu()
    layer_hiddens = [tensor.detach().cpu() for tensor in layer_hiddens]
    return layer_hiddens, prompt_texts, last_positions
