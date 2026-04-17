from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


@dataclass
class LayerSubspaceModel:
    layer_idx: int
    basis: torch.Tensor
    target_center: torch.Tensor
    reference_center: torch.Tensor
    target_label: str
    reference_label: str


@dataclass
class ThresholdResult:
    threshold: float
    metric_name: str
    metric_value: float
    metrics: Dict[str, float]


def _normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / (torch.linalg.norm(vector) + eps)


def _binary_targets(labels: Iterable[str], target_label: str, reference_label: str) -> torch.Tensor:
    mapped: List[int] = []
    for label in labels:
        if label == target_label:
            mapped.append(1)
        elif label == reference_label:
            mapped.append(0)
        else:
            raise ValueError(f"Unexpected label {label!r}; expected {target_label!r} or {reference_label!r}.")
    return torch.tensor(mapped, dtype=torch.long)


def _build_basis(
    target_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_center = target_hidden.mean(dim=0)
    reference_center = reference_hidden.mean(dim=0)
    direction = _normalize(target_center - reference_center)
    if rank <= 1:
        return direction.unsqueeze(0), target_center, reference_center

    centered = torch.cat(
        [
            target_hidden - target_center,
            reference_hidden - reference_center,
        ],
        dim=0,
    )
    if centered.size(0) < 2:
        return direction.unsqueeze(0), target_center, reference_center

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    extra = vh[: max(0, rank - 1)]
    stacked = torch.cat([direction.unsqueeze(0), extra], dim=0)
    q, _ = torch.linalg.qr(stacked.T, mode="reduced")
    basis = q[:, : min(rank, q.size(1))].T.contiguous()
    return basis, target_center, reference_center


def fit_layer_subspace(
    layer_idx: int,
    hidden_states: torch.Tensor,
    labels: List[str],
    *,
    rank: int,
    target_label: str,
    reference_label: str,
) -> LayerSubspaceModel:
    targets = _binary_targets(labels, target_label, reference_label)
    target_hidden = hidden_states[targets == 1]
    reference_hidden = hidden_states[targets == 0]
    if target_hidden.numel() == 0 or reference_hidden.numel() == 0:
        raise ValueError(f"Layer {layer_idx} needs both target and reference samples to fit the subspace.")

    basis, target_center, reference_center = _build_basis(target_hidden, reference_hidden, rank)
    return LayerSubspaceModel(
        layer_idx=layer_idx,
        basis=basis,
        target_center=target_center,
        reference_center=reference_center,
        target_label=target_label,
        reference_label=reference_label,
    )


def score_with_subspace(model: LayerSubspaceModel, hidden_states: torch.Tensor) -> torch.Tensor:
    sample_proj = hidden_states @ model.basis.T
    target_proj = model.target_center @ model.basis.T
    reference_proj = model.reference_center @ model.basis.T
    target_distance = torch.sum((sample_proj - target_proj) ** 2, dim=1)
    reference_distance = torch.sum((sample_proj - reference_proj) ** 2, dim=1)
    return reference_distance - target_distance


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predictions = predictions.to(dtype=torch.long)
    targets = targets.to(dtype=torch.long)
    tp = int(torch.sum((predictions == 1) & (targets == 1)).item())
    tn = int(torch.sum((predictions == 0) & (targets == 0)).item())
    fp = int(torch.sum((predictions == 1) & (targets == 0)).item())
    fn = int(torch.sum((predictions == 0) & (targets == 1)).item())
    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    recall_pos = tp / max(1, tp + fn)
    recall_neg = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 0.0 if precision + recall_pos == 0.0 else 2 * precision * recall_pos / (precision + recall_pos)
    balanced_accuracy = 0.5 * (recall_pos + recall_neg)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall_pos,
        "specificity": recall_neg,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def find_best_threshold(
    scores: torch.Tensor,
    labels: List[str],
    *,
    target_label: str,
    reference_label: str,
    metric_name: str,
) -> ThresholdResult:
    targets = _binary_targets(labels, target_label, reference_label)
    unique_scores = torch.unique(scores).sort().values
    if unique_scores.numel() == 1:
        candidates = unique_scores
    else:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2
        candidates = torch.cat(
            [
                unique_scores[:1] - 1e-6,
                mids,
                unique_scores[-1:] + 1e-6,
            ]
        )

    best_metrics: Dict[str, float] | None = None
    best_threshold = float(candidates[0].item())
    best_value = float("-inf")

    for threshold in candidates:
        predictions = (scores >= threshold).to(dtype=torch.long)
        metrics = _compute_metrics(predictions, targets)
        metric_value = float(metrics[metric_name])
        if metric_value > best_value:
            best_value = metric_value
            best_threshold = float(threshold.item())
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Failed to select a threshold.")

    return ThresholdResult(
        threshold=best_threshold,
        metric_name=metric_name,
        metric_value=best_value,
        metrics=best_metrics,
    )


def evaluate_layer_model(
    model: LayerSubspaceModel,
    hidden_states: torch.Tensor,
    labels: List[str],
    *,
    threshold: float,
) -> Dict[str, float]:
    scores = score_with_subspace(model, hidden_states)
    targets = _binary_targets(labels, model.target_label, model.reference_label)
    predictions = (scores >= threshold).to(dtype=torch.long)
    metrics = _compute_metrics(predictions, targets)
    metrics["score_mean"] = float(scores.mean().item())
    metrics["score_std"] = float(scores.std(unbiased=False).item())
    metrics["threshold"] = float(threshold)
    return metrics


def select_best_layer(
    layer_results: List[Dict[str, float | int | str]],
    *,
    metric_name: str,
) -> Dict[str, float | int | str]:
    if not layer_results:
        raise ValueError("layer_results must be non-empty.")
    return sorted(
        layer_results,
        key=lambda row: (-float(row[metric_name]), int(row["layer_idx"])),
    )[0]
