from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.semantic_recompose import recompose_from_sparse_coeffs
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


SPLIT_DIR_MAP = {
    "alignment": ("teacher_top256_coeffs_alignment", "student_safe_targets_alignment"),
    "analysis_val": ("teacher_top256_coeffs_val", "student_safe_targets_val"),
    "pan_test": ("teacher_top256_coeffs_pan_test", "student_safe_targets_pan_test"),
    "sanity_test": ("teacher_top256_coeffs_sanity_test", "student_safe_targets_sanity_test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompose student safe targets from teacher semantic coefficients."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=sorted(SPLIT_DIR_MAP.keys()),
        help="Teacher semantic-coefficient split to process.",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        help="On-disk dtype for recomposed student targets.",
    )
    parser.add_argument(
        "--allow-multi-teacher-mean",
        action="store_true",
        help=(
            "Opt-in: when multiple teacher key layers map to the same student "
            "layer via the pairing table, element-wise average their recomposed "
            "targets. Default (strict): fail fast with a clear error message "
            "instead of silently mixing teacher layers."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_multi_teacher_reduction(
    teacher_to_student: Dict[int, int],
    *,
    allow_multi_teacher_mean: bool,
    pairing_path: Path | str = "<pairing>",
) -> tuple[str, Dict[str, List[int]]]:
    """Return the reduction strategy and (if any) multi-teacher collision groups.

    Strict (default): raise ValueError if any student layer has more than one
    teacher layer mapped to it. Opt-in: return ``"mean"`` and the collision
    groups for manifest/log recording.
    """

    student_to_teachers: Dict[int, List[int]] = defaultdict(list)
    for teacher_layer, student_layer in teacher_to_student.items():
        student_to_teachers[int(student_layer)].append(int(teacher_layer))
    multi_layer_groups = {
        str(student_layer): sorted(teachers)
        for student_layer, teachers in student_to_teachers.items()
        if len(teachers) > 1
    }
    reduction_strategy = "mean" if allow_multi_teacher_mean else "strict_single"
    if multi_layer_groups and not allow_multi_teacher_mean:
        raise ValueError(
            "Multi-teacher-to-one-student layer collision detected in pairing "
            f"file {pairing_path}: {multi_layer_groups}. Default recomposition "
            "is strict (one teacher per student layer). Resolve the pairing or "
            "re-run with --allow-multi-teacher-mean to explicitly average the "
            "recomposed targets (the chosen strategy will be recorded in the "
            "manifest as multi_to_one_reduction='mean')."
        )
    return reduction_strategy, multi_layer_groups


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)
    logger, log_path = setup_stage_logger("08_recompose_student_targets", Path(cfg.extraction.output_root) / "logs")

    if not hasattr(torch, args.storage_dtype):
        raise ValueError(f"Unsupported storage dtype: {args.storage_dtype}")
    storage_dtype = getattr(torch, args.storage_dtype)

    input_dir_name, output_dir_name = SPLIT_DIR_MAP[args.split]
    input_dir = Path(cfg.extraction.output_root) / "semantic_coeffs" / input_dir_name
    output_dir = ensure_dir(Path(cfg.extraction.output_root) / "student_targets" / output_dir_name)
    student_basis_path = Path(cfg.extraction.output_root) / "semantic_bases" / "student_semantic_basis.pt"
    pairing_path = Path(cfg.extraction.output_root) / "layer_pairing" / "teacher_student_layer_pairs.json"

    student_basis_payload = torch.load(student_basis_path, map_location="cpu", weights_only=True)
    student_basis = student_basis_payload["basis"]
    pairing_payload = _load_json(pairing_path)
    teacher_to_student = {
        int(item["teacher_layer"]): int(item["student_layer"])
        for item in pairing_payload["pairs"]
    }

    # Detect multi-teacher-to-one-student collisions up front. The 方案详述 fixes
    # the pairing rule as "relative depth, no interpolation" and does not
    # prescribe how to combine multiple teachers pointing at the same student
    # layer. Silent averaging would quietly change the supervision signal, so
    # by default we refuse to run and require an explicit opt-in.
    reduction_strategy, multi_layer_groups = resolve_multi_teacher_reduction(
        teacher_to_student,
        allow_multi_teacher_mean=bool(args.allow_multi_teacher_mean),
        pairing_path=pairing_path,
    )

    part_paths = sorted(input_dir.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No semantic coefficient shards found under: {input_dir}")
    log_kv(
        logger,
        "student_recompose_setup",
        config_path=str(Path(args.config).resolve()),
        split=args.split,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        student_basis_path=str(student_basis_path),
        pairing_path=str(pairing_path),
        num_pairs=len(teacher_to_student),
        num_parts=len(part_paths),
        storage_dtype=args.storage_dtype,
        multi_to_one_reduction=reduction_strategy,
        allow_multi_teacher_mean=bool(args.allow_multi_teacher_mean),
        multi_teacher_to_one_student_groups=multi_layer_groups,
        log_path=str(log_path),
    )

    manifest_files: List[str] = []
    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu", weights_only=True)
        accumulators: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for teacher_layer_text, coeff_payload in payload["semantic_coeffs_by_layer"].items():
            teacher_layer = int(teacher_layer_text)
            if teacher_layer not in teacher_to_student:
                continue
            student_layer = teacher_to_student[teacher_layer]
            recomposed = recompose_from_sparse_coeffs(
                student_basis,
                coeff_payload["top_indices"],
                coeff_payload["top_values"],
            )
            accumulators[student_layer].append(recomposed)

        student_safe_target_by_layer: Dict[str, torch.Tensor] = {}
        # Reduction rule:
        #   * strict_single (default): each student layer has exactly one
        #     teacher; the collision check above already rejected any other
        #     configuration, so ``tensors`` always has length 1 and we write
        #     the single recomposed target through unchanged.
        #   * mean (opt-in via --allow-multi-teacher-mean): element-wise
        #     average of all teacher-layer recompositions mapped to the same
        #     student layer. Strategy is echoed in the manifest.
        for student_layer, tensors in sorted(accumulators.items()):
            if len(tensors) == 1:
                combined = tensors[0]
            else:
                combined = torch.stack(tensors, dim=0).mean(dim=0)
            student_safe_target_by_layer[str(student_layer)] = combined.to(dtype=storage_dtype)

        output_payload = {
            "sample_ids": payload["sample_ids"],
            "labels": payload["labels"],
            "student_safe_target_by_layer": student_safe_target_by_layer,
        }
        target_path = output_dir / part_path.name
        torch.save(output_payload, target_path)
        manifest_files.append(str(target_path))
        log_kv(
            logger,
            "student_recompose_part_complete",
            split=args.split,
            part_path=str(part_path),
            output_path=str(target_path),
            sample_count=len(payload["sample_ids"]),
            student_layers=sorted(int(layer_idx) for layer_idx in student_safe_target_by_layer.keys()),
        )

    student_to_teachers: Dict[int, List[int]] = defaultdict(list)
    for teacher_layer, student_layer in teacher_to_student.items():
        student_to_teachers[int(student_layer)].append(int(teacher_layer))
    if reduction_strategy == "mean":
        reduction_doc = (
            "student_safe_target_by_layer[l_S] = mean over teacher layers l_T "
            "paired to l_S of (student_basis @ sparse_a_{l_T}). Explicit opt-in "
            "via --allow-multi-teacher-mean."
        )
    else:
        reduction_doc = (
            "student_safe_target_by_layer[l_S] = student_basis @ sparse_a_{l_T} "
            "for the unique teacher layer l_T paired to l_S (strict 1:1 rule)."
        )
    write_json(
        output_dir / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "split": args.split,
            "semantic_coeff_dir": str(input_dir),
            "student_basis_path": str(student_basis_path),
            "pairing_path": str(pairing_path),
            "teacher_to_student": teacher_to_student,
            "student_to_teachers": {
                str(student_layer): sorted(teachers)
                for student_layer, teachers in student_to_teachers.items()
            },
            "multi_teacher_to_one_student_groups": multi_layer_groups,
            "multi_to_one_reduction": reduction_strategy,
            "allow_multi_teacher_mean": bool(args.allow_multi_teacher_mean),
            "storage_dtype": args.storage_dtype,
            "target_semantics": (
                f"{reduction_doc} sparse_a_{{l_T}} is the top-256 teacher "
                "semantic coefficient vector from step 07. Lives in student "
                "hidden space; supervises cosine alignment against student "
                "layer output at the generation-prompt last-token position."
            ),
            "files": manifest_files,
        },
    )
    log_kv(logger, "student_recompose_complete", split=args.split, output_dir=str(output_dir), files=manifest_files)

    print(
        json.dumps(
            {
                "split": args.split,
                "output_dir": str(output_dir),
                "num_pairs": len(teacher_to_student),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
