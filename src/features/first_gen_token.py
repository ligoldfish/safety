from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from src.data.template_qwen import render_qwen_generation_prompt
from src.ablations.strategies.representation import extract_position_hidden


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


def extract_position_hiddens(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    mode: str = "last_prompt",
    generated_mask: torch.Tensor | None = None,
    skip_embedding_layer: bool = True,
) -> List[torch.Tensor]:
    if not hidden_states:
        raise ValueError("hidden_states must be non-empty")
    selected_states = list(hidden_states[1:] if skip_embedding_layer else hidden_states)
    return [
        extract_position_hidden(
            layer_hidden,
            attention_mask,
            mode=mode,
            generated_mask=generated_mask,
        ).detach()
        for layer_hidden in selected_states
    ]


def generated_token_mask(
    prompt_mask: torch.Tensor,
    sequences: torch.Tensor,
    *,
    generated_count: int,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build prompt+generation masks using HF generate's padded input boundary."""
    if prompt_mask.ndim != 2 or sequences.ndim != 2 or prompt_mask.size(0) != sequences.size(0):
        raise ValueError("prompt_mask and sequences must have compatible [batch, sequence] shapes")
    input_width = prompt_mask.size(1)
    if sequences.size(1) < input_width:
        raise ValueError("generated sequences cannot be shorter than the padded model input")
    actual_generated = sequences.size(1) - input_width
    if type(generated_count) is not int or generated_count <= 0:
        raise ValueError("generated_count must be a positive integer")
    if actual_generated <= 0:
        raise ValueError("generation returned no tokens")
    keep = min(actual_generated, generated_count)
    full_mask = torch.cat(
        [
            prompt_mask,
            torch.ones(
                (prompt_mask.size(0), actual_generated),
                dtype=prompt_mask.dtype,
                device=prompt_mask.device,
            ),
        ],
        dim=1,
    )
    generated_mask = torch.zeros_like(full_mask)
    generated_mask[:, input_width : input_width + keep] = 1
    generated_tokens = sequences[:, input_width : input_width + keep]
    for row in range(generated_tokens.size(0)):
        row_tokens = generated_tokens[row]
        stop = keep
        if eos_token_id is not None:
            eos_positions = (row_tokens == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                stop = int(eos_positions[0].item()) + 1
        elif pad_token_id is not None:
            pad_positions = (row_tokens == int(pad_token_id)).nonzero(as_tuple=False)
            if pad_positions.numel() > 0:
                stop = int(pad_positions[0].item())
        generated_mask[row, input_width + stop : input_width + keep] = 0
        full_mask[row, input_width + stop :] = 0
    return full_mask, generated_mask


@torch.no_grad()
def gather_first_generated_token_representations(
    model: Any,
    tokenizer: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    max_length: int,
) -> tuple[List[torch.Tensor], List[str], torch.Tensor]:
    """Gather each layer's hidden state at the first-assistant-token prediction position.

    The prompt is rendered with ``apply_chat_template(..., add_generation_prompt=True)``
    so the final non-padding token corresponds to the position whose next-token
    distribution is the first generated token. This matches the original proposal's
    "first generated token hidden state" semantics (Pan-style).
    """

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
    # PPU/NPU eager mode: no XLA graph step required.
    layer_hiddens = extract_last_position_hidden(
        hidden_states=outputs.hidden_states,
        attention_mask=encoded["attention_mask"],
        skip_embedding_layer=True,
    )
    last_positions = _last_non_padding_positions(encoded["attention_mask"]).detach().cpu()
    layer_hiddens = [tensor.detach().cpu() for tensor in layer_hiddens]
    return layer_hiddens, prompt_texts, last_positions


@torch.no_grad()
def gather_position_representations(
    model: Any,
    tokenizer: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    max_length: int,
    *,
    mode: str = "last_prompt",
    generated_tokens: int = 4,
) -> tuple[List[torch.Tensor], List[str], torch.Tensor]:
    """Gather a configured prompt or generated-token representation.

    Generated-token modes use deterministic greedy generation and then perform a
    second forward pass on prompt+generated tokens so they are not silently
    approximated by the prompt's next-token prediction position.
    """

    if mode == "last_prompt":
        return gather_first_generated_token_representations(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            max_length=max_length,
        )
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
    prompt_mask = encoded["attention_mask"]
    if mode == "mean_prompt":
        outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
        layers = extract_position_hiddens(outputs.hidden_states, prompt_mask, mode=mode)
        lengths = prompt_mask.sum(dim=1).detach().cpu()
        return [tensor.cpu() for tensor in layers], prompt_texts, lengths - 1
    if mode not in {"first_generated", "first_4_generated_mean"}:
        raise ValueError(f"unsupported representation mode: {mode}")
    count = 1 if mode == "first_generated" else generated_tokens
    if type(count) is not int or count <= 0:
        raise ValueError("generated_tokens must be a positive integer")
    generated = model.generate(
        **encoded,
        max_new_tokens=count,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    full_mask, generated_mask = generated_token_mask(
        prompt_mask,
        generated,
        generated_count=count,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    outputs = model(
        input_ids=generated,
        attention_mask=full_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    layers = extract_position_hiddens(
        outputs.hidden_states,
        full_mask,
        mode=mode,
        generated_mask=generated_mask,
    )
    positions = generated_mask.to(torch.long).argmax(dim=1).detach().cpu()
    return [tensor.cpu() for tensor in layers], prompt_texts, positions


# Backward-compatible alias retained so older callers (e.g., previously-generated
# shards described as "final_response_prefix") keep importing without breakage.
# New code must use gather_first_generated_token_representations.
gather_final_response_prefix_representations = gather_first_generated_token_representations
