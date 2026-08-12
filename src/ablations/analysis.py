from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence


def pan_bucket(record: Mapping) -> str:
    if str(record.get("label", "")).lower() == "harmless":
        return "benign"
    if str(record.get("method", "")).strip():
        return "jailbreak"
    return "harmful_other"


def summarize_corpus_matrix(
    rows: Sequence[Mapping], *, corpora: Sequence[str], suites: Sequence[str]
) -> dict:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["train_corpus"]), str(row["test_suite"]))].append(float(row["score"]))
    matrix: dict[str, dict[str, dict]] = {}
    for corpus in corpora:
        matrix[str(corpus)] = {}
        for suite in suites:
            values = grouped.get((str(corpus), str(suite)), [])
            matrix[str(corpus)][str(suite)] = {
                "n": len(values),
                "mean": None if not values else sum(values) / len(values),
            }
    return matrix


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(map(float, values)), key=lambda item: item[1])
    ranks = [0.0] * len(ordered)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average = (position + 1 + end) / 2.0
        for index, _ in ordered[position:end]:
            ranks[index] = average
        position = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    x, y = _average_ranks(left), _average_ranks(right)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    covariance = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y))
    return None if denominator == 0.0 else covariance / denominator
