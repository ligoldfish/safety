from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.semantic_decompose import topk_semantic_coefficients
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


SPLIT_DIR_MAP = {
    "alignment": ("teacher_safe_component_alignment", "teacher_top256_coeffs_alignment"),
    "analysis_val": ("teacher_safe_component_val", "teacher_top256_coeffs_val"),
    "pan_test": ("teacher_safe_component_pan_test", "teacher_top256_coeffs_pan_test"),
    "sanity_test": ("teacher_safe_component_sanity_test", "teacher_top256_coeffs_sanity_test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose teacher safe components into top-k semantic coefficients."
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
        help="Safe-component split to process.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="How many semantic coefficients to keep per sample per layer.",
    )
    parser.add_argument(
        "--vocab-chunk-size",
        type=int,
        default=4096,
        help="Vocabulary chunk size for streaming top-k search.",
    )
    return parser.parse_args()


def _load_basis(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)
    logger, log_path = setup_stage_logger("07_decompose_teacher_semantics", Path(cfg.extraction.output_root) / "logs")

    input_dir_name, output_dir_name = SPLIT_DIR_MAP[args.split]
    input_dir = Path(cfg.extraction.output_root) / "safe_projection" / input_dir_name
    output_dir = ensure_dir(Path(cfg.extraction.output_root) / "semantic_coeffs" / output_dir_name)
    basis_path = Path(cfg.extraction.output_root) / "semantic_bases" / "teacher_semantic_basis.pt"
    basis_payload = _load_basis(basis_path)
    basis = basis_payload["basis"]

    part_paths = sorted(input_dir.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No projected safe-component shards found under: {input_dir}")
    log_kv(
        logger,
        "semantic_decompose_setup",
        config_path=str(Path(args.config).resolve()),
        split=args.split,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        teacher_basis_path=str(basis_path),
        top_k=int(args.top_k),
        vocab_chunk_size=int(args.vocab_chunk_size),
        num_parts=len(part_paths),
        log_path=str(log_path),
    )

    manifest_files: List[str] = []
    key_layers: List[int] | None = None
    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu", weights_only=True)
        semantic_coeffs_by_layer: Dict[str, Dict[str, torch.Tensor]] = {}
        part_key_layers = sorted(int(layer_idx) for layer_idx in payload["safe_component_by_layer"].keys())
        key_layers = part_key_layers

        for layer_idx in part_key_layers:
            hidden_safe = payload["safe_component_by_layer"][str(layer_idx)].to(dtype=torch.float32)
            top_indices, top_values = topk_semantic_coefficients(
                hidden_safe,
                basis,
                top_k=args.top_k,
                vocab_chunk_size=args.vocab_chunk_size,
            )
            semantic_coeffs_by_layer[str(layer_idx)] = {
                "top_indices": top_indices,
                "top_values": top_values.to(dtype=torch.float32),
            }

        output_payload = {
            "sample_ids": payload["sample_ids"],
            "labels": payload["labels"],
            "semantic_coeffs_by_layer": semantic_coeffs_by_layer,
        }
        target_path = output_dir / part_path.name
        torch.save(output_payload, target_path)
        manifest_files.append(str(target_path))
        log_kv(
            logger,
            "semantic_decompose_part_complete",
            split=args.split,
            part_path=str(part_path),
            output_path=str(target_path),
            sample_count=len(payload["sample_ids"]),
            key_layers=part_key_layers,
        )

    write_json(
        output_dir / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "split": args.split,
            "safe_component_dir": str(input_dir),
            "teacher_basis_path": str(basis_path),
            "top_k": args.top_k,
            "vocab_chunk_size": args.vocab_chunk_size,
            "key_layers": [] if key_layers is None else key_layers,
            "files": manifest_files,
        },
    )
    log_kv(logger, "semantic_decompose_complete", split=args.split, output_dir=str(output_dir), files=manifest_files)

    print(
        json.dumps(
            {
                "split": args.split,
                "output_dir": str(output_dir),
                "top_k": args.top_k,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
