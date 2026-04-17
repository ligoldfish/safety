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
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        for student_layer, tensors in sorted(accumulators.items()):
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

    write_json(
        output_dir / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "split": args.split,
            "semantic_coeff_dir": str(input_dir),
            "student_basis_path": str(student_basis_path),
            "pairing_path": str(pairing_path),
            "teacher_to_student": teacher_to_student,
            "storage_dtype": args.storage_dtype,
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
