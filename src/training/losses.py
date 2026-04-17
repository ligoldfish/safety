from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def cosine_layer_alignment_loss(
    predicted_by_layer: Dict[int, torch.Tensor],
    target_by_layer: Dict[int, torch.Tensor],
) -> tuple[torch.Tensor, Dict[int, float]]:
    if not predicted_by_layer:
        raise ValueError("predicted_by_layer must be non-empty.")

    losses = []
    cosine_means: Dict[int, float] = {}
    for layer_idx, predicted in sorted(predicted_by_layer.items()):
        if layer_idx not in target_by_layer:
            raise KeyError(f"Missing target tensor for layer {layer_idx}.")
        target = target_by_layer[layer_idx].to(device=predicted.device, dtype=predicted.dtype)
        cosine = F.cosine_similarity(predicted, target, dim=-1)
        losses.append(1.0 - cosine.mean())
        cosine_means[layer_idx] = float(cosine.mean().detach().cpu().item())

    loss = torch.stack(losses).mean()
    return loss, cosine_means
