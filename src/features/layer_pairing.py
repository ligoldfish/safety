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
    if teacher_num_layers <= 0 or student_num_layers <= 0:
        raise ValueError("teacher_num_layers and student_num_layers must be positive.")
    mapped = math.floor((teacher_layer / teacher_num_layers) * student_num_layers)
    mapped = max(1, mapped)
    mapped = min(student_num_layers - 1, mapped)
    return int(mapped)


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
                teacher_relative_depth=float(teacher_layer / teacher_num_layers),
            )
        )
    return pairs
