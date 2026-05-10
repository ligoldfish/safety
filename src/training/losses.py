from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def cosine_layer_alignment_loss(
    predicted_by_layer: Dict[int, torch.Tensor],
    target_by_layer: Dict[int, torch.Tensor],
    *,
    sample_weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, Dict[int, float]]:
    if not predicted_by_layer:
        raise ValueError("predicted_by_layer must be non-empty.")

    losses = []
    cosine_means: Dict[int, float] = {}
    normalized_weights: torch.Tensor | None = None
    all_weights_zero = False
    if sample_weights is not None:
        if sample_weights.ndim != 1:
            raise ValueError("sample_weights must have shape [B].")
        normalized_weights = sample_weights.to(dtype=torch.float32)
        weight_sum = normalized_weights.sum()
        if float(weight_sum.item()) <= eps:
            all_weights_zero = True
        else:
            normalized_weights = normalized_weights / weight_sum

    for layer_idx, predicted in sorted(predicted_by_layer.items()):
        if layer_idx not in target_by_layer:
            raise KeyError(f"Missing target tensor for layer {layer_idx}.")
        target = target_by_layer[layer_idx].to(device=predicted.device, dtype=predicted.dtype)
        cosine = F.cosine_similarity(predicted, target, dim=-1)
        if all_weights_zero:
            layer_cosine = cosine.mean()
            losses.append(predicted.sum() * 0.0)
            cosine_means[layer_idx] = float(layer_cosine.detach().cpu().item())
            continue
        if normalized_weights is None:
            layer_cosine = cosine.mean()
        else:
            if normalized_weights.numel() != cosine.numel():
                raise ValueError(
                    f"sample_weights length {normalized_weights.numel()} does not match "
                    f"batch size {cosine.numel()} for layer {layer_idx}."
                )
            weights = normalized_weights.to(device=cosine.device, dtype=cosine.dtype)
            layer_cosine = (cosine * weights).sum()
        losses.append(1.0 - layer_cosine)
        cosine_means[layer_idx] = float(layer_cosine.detach().cpu().item())

    loss = torch.stack(losses).mean()
    return loss, cosine_means
