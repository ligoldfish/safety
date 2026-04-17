from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LayerScoreResult:
    layer_idx: int
    harmful_count: int
    harmless_count: int
    harmful_mean: torch.Tensor
    harmless_mean: torch.Tensor
    mean_diff: torch.Tensor
    mean_diff_norm: float
    linear_probe_acc: float
    final_score: float


def _binary_targets(labels: Iterable[str], *, positive_label: str) -> torch.Tensor:
    return torch.tensor(
        [1 if str(label) == positive_label else 0 for label in labels],
        dtype=torch.float32,
    )


def _standardize(
    train_hidden: torch.Tensor,
    val_hidden: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_hidden.mean(dim=0, keepdim=True)
    std = train_hidden.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (train_hidden - mean) / std, (val_hidden - mean) / std


def fit_linear_probe_accuracy(
    train_hidden: torch.Tensor,
    train_labels: List[str],
    val_hidden: torch.Tensor,
    val_labels: List[str],
    *,
    positive_label: str = "harmful",
    max_iter: int = 100,
    weight_decay: float = 1e-4,
) -> float:
    train_targets = _binary_targets(train_labels, positive_label=positive_label)
    val_targets = _binary_targets(val_labels, positive_label=positive_label)

    if train_hidden.ndim != 2 or val_hidden.ndim != 2:
        raise ValueError("Hidden states for linear probe must have shape [N, d].")
    if train_hidden.size(0) != train_targets.numel():
        raise ValueError("train_hidden and train_labels size mismatch.")
    if val_hidden.size(0) != val_targets.numel():
        raise ValueError("val_hidden and val_labels size mismatch.")

    train_x, val_x = _standardize(train_hidden.to(dtype=torch.float32), val_hidden.to(dtype=torch.float32))
    probe = nn.Linear(train_x.size(1), 1)
    optimizer = torch.optim.LBFGS(
        probe.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = probe(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_targets)
        if weight_decay > 0:
            loss = loss + (0.5 * weight_decay * torch.sum(probe.weight.pow(2)))
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        logits = probe(val_x).squeeze(-1)
        predictions = (logits >= 0).to(dtype=torch.float32)
        accuracy = torch.mean((predictions == val_targets).to(dtype=torch.float32))
    return float(accuracy.item())


def score_teacher_layer(
    *,
    layer_idx: int,
    train_hidden: torch.Tensor,
    train_labels: List[str],
    val_hidden: torch.Tensor,
    val_labels: List[str],
    harmful_label: str = "harmful",
    harmless_label: str = "harmless",
    probe_max_iter: int = 100,
    probe_weight_decay: float = 1e-4,
) -> LayerScoreResult:
    harmful_mask = torch.tensor([str(label) == harmful_label for label in train_labels], dtype=torch.bool)
    harmless_mask = torch.tensor([str(label) == harmless_label for label in train_labels], dtype=torch.bool)
    if harmful_mask.sum().item() == 0 or harmless_mask.sum().item() == 0:
        raise ValueError(f"Layer {layer_idx} needs both harmful and harmless samples.")

    harmful_hidden = train_hidden[harmful_mask]
    harmless_hidden = train_hidden[harmless_mask]
    harmful_mean = harmful_hidden.mean(dim=0)
    harmless_mean = harmless_hidden.mean(dim=0)
    mean_diff = harmful_mean - harmless_mean
    mean_diff_norm = float(torch.linalg.norm(mean_diff).item())
    linear_probe_acc = fit_linear_probe_accuracy(
        train_hidden=train_hidden,
        train_labels=train_labels,
        val_hidden=val_hidden,
        val_labels=val_labels,
        positive_label=harmful_label,
        max_iter=probe_max_iter,
        weight_decay=probe_weight_decay,
    )
    final_score = mean_diff_norm + linear_probe_acc

    return LayerScoreResult(
        layer_idx=layer_idx,
        harmful_count=int(harmful_mask.sum().item()),
        harmless_count=int(harmless_mask.sum().item()),
        harmful_mean=harmful_mean,
        harmless_mean=harmless_mean,
        mean_diff=mean_diff,
        mean_diff_norm=mean_diff_norm,
        linear_probe_acc=linear_probe_acc,
        final_score=final_score,
    )


def top_k_layers(results: List[LayerScoreResult], top_k: int) -> List[LayerScoreResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    return sorted(
        results,
        key=lambda item: (-item.final_score, item.layer_idx),
    )[:top_k]
