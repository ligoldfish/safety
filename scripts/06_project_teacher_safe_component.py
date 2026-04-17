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

from src.features.projection import project_coeff, project_to_subspace, residual_norm_ratio
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


SPLIT_DIR_MAP = {
    "alignment": ("teacher_alignment", "teacher_safe_component_alignment"),
    "analysis_val": ("teacher_analysis_val", "teacher_safe_component_val"),
    "pan_test": ("teacher_pan_test", "teacher_safe_component_pan_test"),
    "sanity_test": ("teacher_sanity_test", "teacher_safe_component_sanity_test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project teacher hidden states into the selected safety subspaces."
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
        help="Teacher split to project.",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        help="On-disk dtype for projected safe components.",
    )
    return parser.parse_args()


def _load_key_layers(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(layer_idx) for layer_idx in payload["key_layers"]]


def _load_subspace_payload(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)
    logger, log_path = setup_stage_logger("06_project_teacher_safe_component", Path(cfg.extraction.output_root) / "logs")

    hidden_dir_name, output_dir_name = SPLIT_DIR_MAP[args.split]
    hidden_dir = Path(cfg.extraction.output_root) / "hidden_states" / hidden_dir_name
    output_dir = ensure_dir(Path(cfg.extraction.output_root) / "safe_projection" / output_dir_name)

    key_layers_path = Path(cfg.extraction.output_root) / "layer_analysis" / "teacher_key_layers.json"
    safe_subspace_dir = Path(cfg.extraction.output_root) / "safe_subspaces"
    key_layers = _load_key_layers(key_layers_path)
    if not key_layers:
        raise ValueError(f"No key layers found in: {key_layers_path}")

    subspaces = {
        layer_idx: _load_subspace_payload(safe_subspace_dir / f"teacher_safe_subspace_layer_{layer_idx:02d}.pt")
        for layer_idx in key_layers
    }
    if not hasattr(torch, args.storage_dtype):
        raise ValueError(f"Unsupported storage dtype: {args.storage_dtype}")
    storage_dtype = getattr(torch, args.storage_dtype)

    part_paths = sorted(hidden_dir.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No hidden-state shards found under: {hidden_dir}")
    log_kv(
        logger,
        "safe_projection_setup",
        config_path=str(Path(args.config).resolve()),
        split=args.split,
        hidden_dir=str(hidden_dir),
        output_dir=str(output_dir),
        key_layers=key_layers,
        num_parts=len(part_paths),
        storage_dtype=args.storage_dtype,
        log_path=str(log_path),
    )

    manifest_files: List[str] = []
    orthogonality_checks: Dict[str, float] = {}
    energy_checks: Dict[str, float] = {}
    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu", weights_only=True)
        safe_component_by_layer: Dict[str, torch.Tensor] = {}
        safe_coeff_by_layer: Dict[str, torch.Tensor] = {}
        residual_ratio_by_layer: Dict[str, torch.Tensor] = {}
        residual_ratio_means: Dict[str, float] = {}

        for layer_idx in key_layers:
            basis = subspaces[layer_idx]["basis"].to(dtype=torch.float32)
            hidden = payload["hidden_by_layer"][str(layer_idx)].to(dtype=torch.float32)
            coeff = project_coeff(hidden, basis)
            hidden_safe = project_to_subspace(hidden, basis)
            residual = hidden - hidden_safe
            residual_ratio = residual_norm_ratio(hidden, hidden_safe)

            safe_component_by_layer[str(layer_idx)] = hidden_safe.to(dtype=storage_dtype)
            safe_coeff_by_layer[str(layer_idx)] = coeff.to(dtype=torch.float32)
            residual_ratio_by_layer[str(layer_idx)] = residual_ratio.to(dtype=torch.float32)
            residual_ratio_means[str(layer_idx)] = float(residual_ratio.mean().item())

            orth_error = basis.T @ residual.T
            orthogonality_checks[str(layer_idx)] = float(orth_error.abs().mean().item())
            energy_error = (
                hidden.pow(2).sum(dim=1)
                - hidden_safe.pow(2).sum(dim=1)
                - residual.pow(2).sum(dim=1)
            ).abs()
            energy_checks[str(layer_idx)] = float(energy_error.mean().item())

        output_payload = {
            "model_name": payload["model_name"],
            "sample_ids": payload["sample_ids"],
            "labels": payload["labels"],
            "safe_component_by_layer": safe_component_by_layer,
            "safe_coeff_by_layer": safe_coeff_by_layer,
            "residual_norm_ratio_by_layer": residual_ratio_by_layer,
        }
        target_path = output_dir / part_path.name
        torch.save(output_payload, target_path)
        manifest_files.append(str(target_path))
        log_kv(
            logger,
            "safe_projection_part_complete",
            split=args.split,
            part_path=str(part_path),
            output_path=str(target_path),
            sample_count=len(payload["sample_ids"]),
            mean_residual_norm_ratio_by_layer=residual_ratio_means,
        )

    write_json(
        output_dir / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "split": args.split,
            "teacher_hidden_dir": str(hidden_dir),
            "key_layers_path": str(key_layers_path),
            "safe_subspace_dir": str(safe_subspace_dir),
            "key_layers": key_layers,
            "storage_dtype": args.storage_dtype,
            "orthogonality_check_mean_abs": orthogonality_checks,
            "energy_decomposition_mean_abs": energy_checks,
            "files": manifest_files,
        },
    )
    log_kv(logger, "safe_projection_complete", split=args.split, output_dir=str(output_dir), files=manifest_files)

    print(
        json.dumps(
            {
                "split": args.split,
                "output_dir": str(output_dir),
                "key_layers": key_layers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
