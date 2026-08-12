from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Sequence


def _validate_paired(left: Sequence, right: Sequence) -> None:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must be non-empty and equal length")


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    *,
    draws: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    _validate_paired(left, right)
    if tuple(map(str, left_ids)) != tuple(map(str, right_ids)) or len(left_ids) != len(left):
        raise ValueError("paired bootstrap requires identical sample IDs in identical order")
    if type(draws) is not int or draws <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("draws must be positive and confidence must be in (0,1)")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    estimates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(draws)
    )
    alpha = (1.0 - confidence) / 2.0
    lo_index = max(0, min(draws - 1, int(math.floor(alpha * draws))))
    hi_index = max(0, min(draws - 1, int(math.ceil((1.0 - alpha) * draws)) - 1))
    return {
        "n": len(differences),
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "mean_difference": sum(differences) / len(differences),
        "ci_low": estimates[lo_index],
        "ci_high": estimates[hi_index],
    }


def mcnemar_exact(left: Sequence[object], right: Sequence[object]) -> dict:
    _validate_paired(left, right)
    b = sum(bool(a) and not bool(c) for a, c in zip(left, right))
    c = sum(not bool(a) and bool(c) for a, c in zip(left, right))
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "discordant": n, "p_value": p_value}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("p-values must be in [0,1]")
    ordered = sorted(enumerate(map(float, p_values)), key=lambda item: item[1])
    result = [0.0] * len(ordered)
    running = 0.0
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        result[original_index] = running
    return result


def cohen_kappa(left: Sequence[Hashable], right: Sequence[Hashable]) -> float:
    _validate_paired(left, right)
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] / n * right_counts[label] / n for label in set(left_counts) | set(right_counts))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
