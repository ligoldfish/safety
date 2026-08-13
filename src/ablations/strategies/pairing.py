from __future__ import annotations

import random
from typing import Mapping, Sequence

import torch


def linear_cka_matrix(
    teacher_by_layer: Mapping[int, torch.Tensor],
    student_by_layer: Mapping[int, torch.Tensor],
    *,
    teacher_key_layers: Sequence[int],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return linear CKA for each key-teacher/all-student layer pair.

    The covariance formulation avoids materialising an ``N x N`` Gram matrix.
    All layers must describe the same ordered samples; callers are responsible
    for checking stable sample IDs before invoking this function.
    """

    if not teacher_key_layers or not student_by_layer:
        raise ValueError("CKA requires teacher key layers and student layers")
    rows: list[torch.Tensor] = []
    expected_samples: int | None = None
    for teacher_layer in teacher_key_layers:
        if int(teacher_layer) not in teacher_by_layer:
            raise KeyError(f"missing teacher hidden states for layer {teacher_layer}")
        teacher = teacher_by_layer[int(teacher_layer)].to(torch.float64)
        if teacher.ndim != 2 or teacher.size(0) < 2:
            raise ValueError("CKA hidden states must have shape [samples>=2, hidden]")
        expected_samples = int(teacher.size(0)) if expected_samples is None else expected_samples
        if int(teacher.size(0)) != expected_samples:
            raise ValueError("all CKA layers must have the same sample count")
        teacher = teacher - teacher.mean(dim=0, keepdim=True)
        teacher_norm = torch.linalg.matrix_norm(teacher.T @ teacher, ord="fro")
        similarities: list[torch.Tensor] = []
        for student_layer in sorted(student_by_layer):
            student = student_by_layer[student_layer].to(torch.float64)
            if student.ndim != 2 or int(student.size(0)) != expected_samples:
                raise ValueError("teacher and student CKA layers must share ordered samples")
            student = student - student.mean(dim=0, keepdim=True)
            cross = torch.linalg.matrix_norm(teacher.T @ student, ord="fro").square()
            denominator = teacher_norm * torch.linalg.matrix_norm(student.T @ student, ord="fro")
            similarities.append(torch.where(denominator > eps, cross / denominator, cross * 0.0))
        rows.append(torch.stack(similarities))
    return torch.stack(rows).to(torch.float32)


def pair_layers(
    teacher_key_layers: Sequence[int],
    *,
    teacher_layers: int,
    student_layers: int,
    mode: str,
    cka: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[int, ...]:
    keys = tuple(int(layer) for layer in teacher_key_layers)
    if type(teacher_layers) is not int or type(student_layers) is not int or teacher_layers <= 0 or student_layers <= 0:
        raise ValueError("teacher_layers and student_layers must be positive integers")
    if not keys or len(keys) != len(set(keys)) or any(layer < 0 or layer >= teacher_layers for layer in keys):
        raise ValueError("teacher key layers must be unique and in range")
    normalized = str(mode).strip().lower()
    if normalized == "relative_depth":
        return tuple(max(0, min(student_layers - 1, ((layer + 1) * student_layers // teacher_layers) - 1)) for layer in keys)
    if normalized == "same_index_clamped":
        return tuple(min(layer, student_layers - 1) for layer in keys)
    if normalized == "cka_nearest":
        if cka is None or cka.ndim != 2 or tuple(cka.shape) != (len(keys), student_layers):
            raise ValueError("cka_nearest requires cka shape [key_teacher_layers, student_layers]")
        return tuple(int(index) for index in cka.argmax(dim=1).tolist())
    if normalized == "random_permutation":
        if type(seed) is not int:
            raise ValueError("random_permutation requires an integer seed")
        if len(keys) > student_layers:
            raise ValueError("random_permutation requires at least as many student layers as key layers")
        return tuple(random.Random(seed).sample(tuple(range(student_layers)), len(keys)))
    raise ValueError(f"unsupported pairing mode: {mode}")
