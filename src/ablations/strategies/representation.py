from __future__ import annotations

from typing import Literal

import torch


RepresentationMode = Literal[
    "last_prompt", "mean_prompt", "first_generated", "first_4_generated_mean"
]


def _validate(hidden_states: torch.Tensor, prompt_mask: torch.Tensor) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if prompt_mask.ndim != 2 or tuple(prompt_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError("prompt_mask shape must match hidden_states [batch, sequence]")
    if bool((prompt_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every sample must contain at least one prompt token")


def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (hidden_states * weights).sum(dim=1) / denominator


def extract_position_hidden(
    hidden_states: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    mode: RepresentationMode | str,
    generated_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract one representation per sample without including padding tokens."""

    _validate(hidden_states, prompt_mask)
    normalized = str(mode).strip().lower()
    if normalized == "last_prompt":
        positions = torch.arange(prompt_mask.size(1), device=prompt_mask.device).expand_as(prompt_mask)
        positions = positions.masked_fill(prompt_mask == 0, -1).max(dim=1).values
        rows = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[rows, positions.to(hidden_states.device), :]
    if normalized == "mean_prompt":
        return _masked_mean(hidden_states, prompt_mask)
    if normalized not in {"first_generated", "first_4_generated_mean"}:
        raise ValueError(f"unsupported representation mode: {mode}")
    if generated_mask is None:
        raise ValueError(f"{normalized} requires generated_mask")
    if generated_mask.ndim != 2 or tuple(generated_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError("generated_mask shape must match hidden_states [batch, sequence]")
    counts = generated_mask.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise ValueError("one or more samples have no generated tokens")
    positions = torch.arange(generated_mask.size(1), device=generated_mask.device).expand_as(generated_mask)
    if normalized == "first_generated":
        first = positions.masked_fill(generated_mask == 0, generated_mask.size(1)).min(dim=1).values
        rows = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[rows, first.to(hidden_states.device), :]
    ranks = generated_mask.to(torch.long).cumsum(dim=1)
    first_four_mask = (generated_mask != 0) & (ranks <= 4)
    return _masked_mean(hidden_states, first_four_mask)
