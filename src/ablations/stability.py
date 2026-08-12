from __future__ import annotations

from typing import Iterable

import torch


def _orthonormalize(basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2 or basis.size(0) == 0 or basis.size(1) == 0:
        raise ValueError("basis must have non-empty shape [hidden, rank]")
    q, _ = torch.linalg.qr(basis.to(dtype=torch.float64), mode="reduced")
    return q


def principal_angles(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    q_left = _orthonormalize(left)
    q_right = _orthonormalize(right)
    singular_values = torch.linalg.svdvals(q_left.T @ q_right).clamp(0.0, 1.0)
    return torch.arccos(singular_values).to(dtype=torch.float32)


def projection_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    q_left = _orthonormalize(left)
    q_right = _orthonormalize(right)
    common_rank = min(q_left.size(1), q_right.size(1))
    overlap = torch.linalg.matrix_norm(q_left.T @ q_right, ord="fro").pow(2) / max(common_rank, 1)
    return float(overlap.clamp(0.0, 1.0).item())


def layer_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = {int(value) for value in left}
    right_set = {int(value) for value in right}
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def bootstrap_indices(sample_count: int, *, draws: int = 20, seed: int = 42) -> tuple[torch.Tensor, ...]:
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(draws) is not int or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return tuple(torch.randint(sample_count, (sample_count,), generator=generator) for _ in range(draws))
