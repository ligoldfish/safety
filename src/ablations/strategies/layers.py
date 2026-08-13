from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LayerCandidate:
    layer_idx: int
    effect_size: float
    probe_accuracy: float


def _validate(candidates: Sequence[LayerCandidate], k: int) -> tuple[LayerCandidate, ...]:
    normalized = tuple(candidates)
    if not normalized:
        raise ValueError("layer candidates must be non-empty")
    if type(k) is not int or not 1 <= k <= len(normalized):
        raise ValueError(f"expected 1 <= k <= {len(normalized)}")
    indices = [item.layer_idx for item in normalized]
    if len(indices) != len(set(indices)):
        raise ValueError("layer candidate indices must be unique")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("layer indices must be non-negative integers")
    return normalized


def _even_indices(sorted_indices: tuple[int, ...], k: int) -> tuple[int, ...]:
    if k == 1:
        return (sorted_indices[-1],)
    final = len(sorted_indices) - 1
    positions = [round(offset * final / (k - 1)) for offset in range(k)]
    result = tuple(sorted_indices[position] for position in positions)
    if len(set(result)) != k:
        raise ValueError("evenly-spaced selection could not produce k unique layers")
    return result


def select_layers(
    candidates: Sequence[LayerCandidate],
    *,
    k: int,
    mode: str,
    seed: int | None = None,
) -> tuple[int, ...]:
    normalized = _validate(candidates, k)
    strategy = str(mode).strip().lower()
    if strategy in {"effect_only", "probe_only", "effect_probe_sum"}:
        if strategy == "effect_only":
            score = lambda item: item.effect_size
        elif strategy == "probe_only":
            score = lambda item: item.probe_accuracy
        else:
            score = lambda item: item.effect_size + item.probe_accuracy
        ranked = sorted(normalized, key=lambda item: (-score(item), item.layer_idx))
        return tuple(item.layer_idx for item in ranked[:k])
    indices = tuple(sorted(item.layer_idx for item in normalized))
    if strategy == "last_k":
        return indices[-k:]
    if strategy == "evenly_spaced":
        return _even_indices(indices, k)
    if strategy == "random_k":
        if type(seed) is not int:
            raise ValueError("random_k requires an integer seed")
        return tuple(sorted(random.Random(seed).sample(indices, k)))
    raise ValueError(f"unsupported layer selection mode: {mode}")
