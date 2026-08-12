from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def supervision_weights(
    labels: Sequence[str],
    *,
    mode: str,
    harmful_weight: float = 1.0,
    harmless_weight: float = 1.0,
    has_target: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor | None:
    normalized = str(mode).strip().lower()
    if normalized not in {"all", "harmful_only", "label_weighted", "harmless_anchor"}:
        raise ValueError(f"unsupported supervision mode: {mode}")
    flags = [True] * len(labels) if has_target is None else [bool(v) for v in has_target.cpu().tolist()]
    if len(flags) != len(labels):
        raise ValueError("has_target length must match labels")
    if normalized == "all" and all(flags):
        return None
    values: list[float] = []
    for label, available in zip(labels, flags):
        if not available:
            values.append(0.0)
        elif normalized == "all":
            values.append(1.0)
        elif str(label) == "harmful":
            values.append(float(harmful_weight))
        elif str(label) == "harmless":
            values.append(0.0 if normalized == "harmful_only" else float(harmless_weight))
        else:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if weights is None:
        return values.mean()
    if weights.ndim != 1 or weights.numel() != values.numel():
        raise ValueError("sample_weights must have shape [batch]")
    weights = weights.to(device=values.device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("sample_weights must be finite and non-negative")
    total = weights.sum()
    if float(total.item()) <= 1e-12:
        return reference.sum() * 0.0
    return (values.to(torch.float32) * (weights / total)).sum()


def layer_alignment_loss(
    predicted_by_layer: Mapping[int, torch.Tensor],
    target_by_layer: Mapping[int, torch.Tensor],
    *,
    kind: str = "cosine",
    sample_weights: torch.Tensor | None = None,
    margin: float = 0.2,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[int, float]]:
    if not predicted_by_layer:
        raise ValueError("predicted_by_layer must be non-empty")
    normalized = str(kind).strip().lower()
    if normalized not in {"cosine", "normalized_mse", "raw_mse", "margin_contrastive"}:
        raise ValueError(f"unsupported layer alignment loss: {kind}")
    losses: list[torch.Tensor] = []
    similarities: dict[int, float] = {}
    for layer, predicted in sorted(predicted_by_layer.items()):
        if layer not in target_by_layer:
            raise KeyError(f"missing target tensor for layer {layer}")
        target = target_by_layer[layer].to(device=predicted.device, dtype=predicted.dtype)
        if target.shape != predicted.shape or predicted.ndim != 2:
            raise ValueError("predicted and target tensors must share [batch, hidden] shape")
        cosine = F.cosine_similarity(predicted.to(torch.float32), target.to(torch.float32), dim=-1, eps=eps)
        if sample_weights is None or float(sample_weights.sum().item()) <= eps:
            metric_cosine = cosine.mean()
        else:
            metric_weights = sample_weights.to(device=cosine.device, dtype=torch.float32)
            metric_cosine = (cosine * (metric_weights / metric_weights.sum())).sum()
        similarities[int(layer)] = float(metric_cosine.detach().cpu().item())
        if normalized == "cosine":
            per_sample = 1.0 - cosine
        elif normalized == "normalized_mse":
            per_sample = (
                F.normalize(predicted.to(torch.float32), dim=-1, eps=eps)
                - F.normalize(target.to(torch.float32), dim=-1, eps=eps)
            ).pow(2).mean(dim=-1)
        elif normalized == "raw_mse":
            per_sample = (predicted.to(torch.float32) - target.to(torch.float32)).pow(2).mean(dim=-1)
        else:
            if predicted.size(0) < 2:
                per_sample = predicted.to(torch.float32).sum(dim=-1) * 0.0
            else:
                negative = target.to(torch.float32).roll(shifts=1, dims=0)
                negative_cosine = F.cosine_similarity(predicted.to(torch.float32), negative, dim=-1, eps=eps)
                per_sample = F.relu(float(margin) - cosine + negative_cosine)
        losses.append(_weighted_mean(per_sample, sample_weights, predicted))
    return torch.stack(losses).mean(), similarities
