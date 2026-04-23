from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class LayerPair:
    teacher_layer: int
    student_layer: int
    teacher_relative_depth: float


def map_teacher_to_student_layer(
    teacher_layer: int,
    *,
    teacher_num_layers: int,
    student_num_layers: int,
) -> int:
    """Map a teacher layer index (0-based) to a student layer index (0-based).

    The 方案详述 spec states (1-based):

        l_S = floor(l_T / L_T * L_S)   with l_S clamped to >= 1

    All hidden-state shards produced by ``01_extract_hidden_states.py`` are stored
    with ``skip_embedding_layer=True`` so index 0 corresponds to the first
    transformer-block output (0-based). We therefore translate the teacher layer
    to 1-based, apply the spec formula, clamp to [1, L_S], then translate back
    to 0-based. This ensures every valid teacher index maps to a valid student
    index in ``[0, L_S - 1]`` (student layer 0 is reachable when the ratio is
    small enough).
    """

    if teacher_num_layers <= 0 or student_num_layers <= 0:
        raise ValueError("teacher_num_layers and student_num_layers must be positive.")
    if teacher_layer < 0 or teacher_layer >= teacher_num_layers:
        raise ValueError(
            f"teacher_layer={teacher_layer} out of range for teacher_num_layers={teacher_num_layers}"
        )

    teacher_layer_one_based = int(teacher_layer) + 1
    mapped_one_based = math.floor(
        teacher_layer_one_based / teacher_num_layers * student_num_layers
    )
    mapped_one_based = max(1, min(student_num_layers, mapped_one_based))
    return int(mapped_one_based - 1)


def build_layer_pairs(
    teacher_layers: List[int],
    *,
    teacher_num_layers: int,
    student_num_layers: int,
) -> List[LayerPair]:
    pairs: List[LayerPair] = []
    for teacher_layer in teacher_layers:
        pairs.append(
            LayerPair(
                teacher_layer=int(teacher_layer),
                student_layer=map_teacher_to_student_layer(
                    int(teacher_layer),
                    teacher_num_layers=teacher_num_layers,
                    student_num_layers=student_num_layers,
                ),
                teacher_relative_depth=float((int(teacher_layer) + 1) / teacher_num_layers),
            )
        )
    return pairs
