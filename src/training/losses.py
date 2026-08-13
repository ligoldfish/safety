from __future__ import annotations

from typing import Dict

import torch
from src.ablations.strategies.losses import layer_alignment_loss


def cosine_layer_alignment_loss(
    predicted_by_layer: Dict[int, torch.Tensor],
    target_by_layer: Dict[int, torch.Tensor],
    *,
    sample_weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, Dict[int, float]]:
    return layer_alignment_loss(
        predicted_by_layer,
        target_by_layer,
        kind="cosine",
        sample_weights=sample_weights,
        eps=eps,
    )
