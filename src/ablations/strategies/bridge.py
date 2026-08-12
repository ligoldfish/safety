from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenBridgeResult:
    teacher_to_student: Mapping[int, int]
    coverage: float
    conflicts: int
    unmatched_teacher_ids: tuple[int, ...]
    eligible_teacher_tokens: int


def _paired_matrices(teacher: torch.Tensor, student: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher.ndim != 2 or student.ndim != 2:
        raise ValueError("teacher and student alignment tensors must have shape [samples, hidden]")
    if teacher.size(0) != student.size(0) or teacher.size(0) == 0:
        raise ValueError("teacher and student alignment tensors need the same non-zero sample count")
    return teacher.to(dtype=torch.float64), student.to(dtype=torch.float64)


def fit_ridge(teacher: torch.Tensor, student: torch.Tensor, *, alpha: float = 1e-4) -> torch.Tensor:
    x, y = _paired_matrices(teacher, student)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
        raise ValueError("ridge alpha must be non-negative")
    identity = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
    mapping = torch.linalg.solve(x.T @ x + float(alpha) * identity, x.T @ y)
    return mapping.to(dtype=torch.float32)


def fit_orthogonal_procrustes(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    x, y = _paired_matrices(teacher, student)
    if x.size(1) < y.size(1):
        raise ValueError(
            "orthogonal Procrustes requires teacher hidden size >= student hidden size"
        )
    if x.size(1) == y.size(1):
        reduced = x
        reducer = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
    else:
        _, _, vh_x = torch.linalg.svd(x, full_matrices=False)
        reducer = vh_x[: y.size(1)].T
        reduced = x @ reducer
    u, _, vh = torch.linalg.svd(reduced.T @ y, full_matrices=False)
    return (reducer @ u @ vh).to(dtype=torch.float32)


def apply_linear_bridge(hidden: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 2 or mapping.ndim != 2 or hidden.size(1) != mapping.size(0):
        raise ValueError("hidden and mapping shapes are incompatible")
    return hidden.to(dtype=torch.float32) @ mapping.to(device=hidden.device, dtype=torch.float32)


def _reverse_vocab(vocab: Mapping[str, int]) -> dict[int, list[str]]:
    reverse: dict[int, list[str]] = {}
    for token, raw_id in vocab.items():
        token_id = int(raw_id)
        reverse.setdefault(token_id, []).append(str(token))
    return reverse


def match_token_strings(
    teacher_vocab: Mapping[str, int],
    student_vocab: Mapping[str, int],
    *,
    teacher_special_ids: set[int] | frozenset[int] = frozenset(),
    student_special_ids: set[int] | frozenset[int] = frozenset(),
) -> TokenBridgeResult:
    teacher_reverse = _reverse_vocab(teacher_vocab)
    student_reverse = _reverse_vocab(student_vocab)
    student_by_token: dict[str, int] = {}
    ambiguous_student_tokens: set[str] = set()
    for token_id, tokens in student_reverse.items():
        if token_id in student_special_ids:
            continue
        for token in tokens:
            if token in student_by_token and student_by_token[token] != token_id:
                ambiguous_student_tokens.add(token)
            else:
                student_by_token[token] = token_id

    mapping: dict[int, int] = {}
    unmatched: list[int] = []
    eligible = 0
    for token_id, tokens in sorted(teacher_reverse.items()):
        if token_id in teacher_special_ids:
            continue
        eligible += 1
        candidates = {
            student_by_token[token]
            for token in tokens
            if token in student_by_token and token not in ambiguous_student_tokens
        }
        if len(candidates) == 1:
            mapping[token_id] = next(iter(candidates))
        else:
            unmatched.append(token_id)
    conflicts = len(mapping) - len(set(mapping.values()))
    coverage = len(mapping) / eligible if eligible else 0.0
    return TokenBridgeResult(mapping, coverage, conflicts, tuple(unmatched), eligible)


def match_embedding_nearest(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
    *,
    min_cosine: float = 0.0,
    teacher_special_ids: set[int] | frozenset[int] = frozenset(),
    student_special_ids: set[int] | frozenset[int] = frozenset(),
) -> TokenBridgeResult:
    if teacher_embeddings.ndim != 2 or student_embeddings.ndim != 2:
        raise ValueError("embedding matrices must have shape [vocab, hidden]")
    if teacher_embeddings.size(1) != student_embeddings.size(1):
        raise ValueError("embedding-nearest requires equal embedding dimensions")
    teacher = F.normalize(teacher_embeddings.to(dtype=torch.float32), dim=1)
    student = F.normalize(student_embeddings.to(dtype=torch.float32), dim=1)
    valid_student = [idx for idx in range(student.size(0)) if idx not in student_special_ids]
    if not valid_student:
        raise ValueError("no eligible student tokens for embedding matching")
    student_index = torch.tensor(valid_student, dtype=torch.long, device=student.device)
    mapping: dict[int, int] = {}
    unmatched: list[int] = []
    eligible = 0
    for teacher_id in range(teacher.size(0)):
        if teacher_id in teacher_special_ids:
            continue
        eligible += 1
        similarities = teacher[teacher_id] @ student.index_select(0, student_index).T
        value, position = similarities.max(dim=0)
        if float(value.item()) < float(min_cosine):
            unmatched.append(teacher_id)
            continue
        mapping[teacher_id] = valid_student[int(position.item())]
    conflicts = len(mapping) - len(set(mapping.values()))
    coverage = len(mapping) / eligible if eligible else 0.0
    return TokenBridgeResult(mapping, coverage, conflicts, tuple(unmatched), eligible)


def validate_bridge_mode(mode: str, *, tokenizer_shared: bool) -> str:
    normalized = str(mode).strip().lower()
    allowed = {"vocabulary", "token_string", "embedding_nearest", "ridge", "orthogonal_procrustes"}
    if normalized not in allowed:
        raise ValueError(f"unsupported bridge mode: {mode}")
    if normalized == "vocabulary" and not tokenizer_shared:
        raise ValueError("vocabulary token-id bridge is forbidden for cross-tokenizer model pairs")
    return normalized


def vocabularies_identical(teacher_vocab: Mapping[str, int], student_vocab: Mapping[str, int]) -> bool:
    return {str(k): int(v) for k, v in teacher_vocab.items()} == {
        str(k): int(v) for k, v in student_vocab.items()
    }


def remap_sparse_coefficients(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    teacher_to_student: Mapping[int, int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if top_indices.ndim != 2 or top_values.ndim != 2 or top_indices.shape != top_values.shape:
        raise ValueError("top_indices and top_values must have the same [batch, terms] shape")
    mapped_indices = torch.zeros_like(top_indices, dtype=torch.long)
    mapped_values = top_values.clone()
    flat_source = top_indices.reshape(-1).tolist()
    flat_target = mapped_indices.reshape(-1)
    flat_values = mapped_values.reshape(-1)
    matched = 0
    for position, raw_id in enumerate(flat_source):
        teacher_id = int(raw_id)
        if teacher_id in teacher_to_student:
            flat_target[position] = int(teacher_to_student[teacher_id])
            matched += 1
        else:
            flat_values[position] = 0
    total = len(flat_source)
    return mapped_indices, mapped_values, {
        "total_terms": total,
        "matched_terms": matched,
        "unmatched_terms": total - matched,
    }


def hidden_bridge_targets(
    safe_components_by_teacher_layer: Mapping[int, torch.Tensor],
    pairs: Sequence[Mapping[str, int]],
    mappings_by_pair: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    targets: dict[int, torch.Tensor] = {}
    for raw_pair in pairs:
        pair_idx = int(raw_pair["pair_idx"])
        teacher_layer = int(raw_pair["teacher_layer"])
        if teacher_layer not in safe_components_by_teacher_layer:
            raise KeyError(f"missing safe component for teacher layer {teacher_layer}")
        if pair_idx not in mappings_by_pair:
            raise KeyError(f"missing hidden bridge mapping for pair {pair_idx}")
        targets[pair_idx] = apply_linear_bridge(
            safe_components_by_teacher_layer[teacher_layer], mappings_by_pair[pair_idx]
        )
    return targets
