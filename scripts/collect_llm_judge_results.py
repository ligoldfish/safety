from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pairs import DEFAULT_PAIR, PAIRS

FORMAL_PAIRS = (
    "qwen35_9b_to_08b",
    "llama31_8b_to_1b",
    "qwen3_8b_to_06b",
    "qwen3_8b_to_4b",
    "qwen3_4b_to_06b",
)
DATASETS = (
    "pan", "safety_tuned_llamas", "coconot",
    "wildguardmix", "wildjailbreak", "c5",
)
METHODS = ("ours", "ours_sft1", "sft", "distill", "nosft")


@dataclass(frozen=True)
class ResultSpec:
    pair_id: str
    dataset: str
    method: str
    epoch: str
    result_dir: Path


def _epochs(dataset: str, method: str) -> tuple[str, ...]:
    if method == "nosft":
        return ("single",)
    if dataset == "safety_tuned_llamas" and method in {"sft", "distill"}:
        return ("epoch_006",)
    return ("epoch_002", "epoch_003")


def _method_root(outputs_root: Path, pair_id: str, dataset: str, method: str) -> Path:
    pair = PAIRS[pair_id]
    student_tag = pair["student"]["tag"]
    dataset_suffix = "" if dataset == "pan" else f"_{dataset}"
    if method in {"ours", "ours_sft1"}:
        training = "training_sft1" if method == "ours_sft1" else "training"
        if dataset == "pan":
            return outputs_root / f"{pair_id}_phase1_npu" / training / "eval_suite"
        pair_suffix = "" if pair_id == DEFAULT_PAIR else f"_{pair_id}"
        return (
            outputs_root / f"safety_full_{dataset}_npu{pair_suffix}"
            / "phase1" / training / "eval_suite"
        )
    if method == "sft":
        return outputs_root / "baselines" / f"sft_{student_tag}{dataset_suffix}_npu" / "eval_suite"
    if method == "distill":
        return outputs_root / "baselines" / f"distill_{pair_id}{dataset_suffix}_npu" / "eval_suite"
    if dataset == "pan":
        return outputs_root / "baselines" / f"no_sft_{student_tag}_npu"
    legacy_prefix = "" if pair_id == DEFAULT_PAIR else f"{student_tag}_"
    return outputs_root / "baselines" / f"{legacy_prefix}eval_{dataset}_npu"


def iter_result_specs(
    outputs_root: Path,
    pair_ids: Sequence[str] = FORMAL_PAIRS,
) -> list[ResultSpec]:
    specs = []
    for pair_id in pair_ids:
        for dataset in DATASETS:
            for method in METHODS:
                root = _method_root(outputs_root, pair_id, dataset, method)
                for epoch in _epochs(dataset, method):
                    result_dir = root if epoch == "single" else root / epoch
                    specs.append(ResultSpec(pair_id, dataset, method, epoch, result_dir))
    return specs
