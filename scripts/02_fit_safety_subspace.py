from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_b.hidden_states import load_hidden_state_split
from src.phase_b.subspace import (
    evaluate_layer_model,
    find_best_threshold,
    fit_layer_subspace,
    score_with_subspace,
    select_best_layer,
)
from src.utils.config import load_phaseb_config
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a stage-B safety subspace from phase-A hidden states and validate it."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phaseB_cpu.yaml",
        help="Path to the phase-B YAML config.",
    )
    return parser.parse_args()


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_phaseb_config(args.config)
    set_global_seed(cfg.seed)

    hidden_root = Path(cfg.inputs.hidden_root)
    output_root = ensure_dir(cfg.output.output_root)

    train_split = load_hidden_state_split(
        hidden_root / cfg.inputs.train_dir,
        max_samples_per_label=cfg.limits.train_max_samples_per_label,
        selected_layers=cfg.method.selected_layers or None,
    )
    val_split = load_hidden_state_split(
        hidden_root / cfg.inputs.val_dir,
        max_samples_per_label=cfg.limits.val_max_samples_per_label,
        selected_layers=train_split.available_layers,
    )

    test_split = None
    if cfg.inputs.test_dir:
        test_split = load_hidden_state_split(
            hidden_root / cfg.inputs.test_dir,
            max_samples_per_label=cfg.limits.test_max_samples_per_label,
            selected_layers=train_split.available_layers,
        )

    per_layer_rows: List[Dict[str, object]] = []
    serialized_models: Dict[int, Dict[str, torch.Tensor | str | int]] = {}

    for layer_idx in train_split.available_layers:
        layer_model = fit_layer_subspace(
            layer_idx=layer_idx,
            hidden_states=train_split.layer_tensors[layer_idx],
            labels=train_split.labels,
            rank=cfg.method.rank,
            target_label=cfg.method.target_label,
            reference_label=cfg.method.reference_label,
        )
        val_scores = score_with_subspace(layer_model, val_split.layer_tensors[layer_idx])
        threshold_result = find_best_threshold(
            val_scores,
            val_split.labels,
            target_label=cfg.method.target_label,
            reference_label=cfg.method.reference_label,
            metric_name=cfg.method.selection_metric,
        )

        row: Dict[str, object] = {
            "layer_idx": layer_idx,
            "rank": int(layer_model.basis.size(0)),
            "selection_metric": cfg.method.selection_metric,
            "threshold": threshold_result.threshold,
            f"val_{cfg.method.selection_metric}": threshold_result.metric_value,
        }
        for key, value in threshold_result.metrics.items():
            row[f"val_{key}"] = value

        if test_split is not None:
            test_metrics = evaluate_layer_model(
                layer_model,
                test_split.layer_tensors[layer_idx],
                test_split.labels,
                threshold=threshold_result.threshold,
            )
            for key, value in test_metrics.items():
                row[f"test_{key}"] = value

        per_layer_rows.append(row)
        serialized_models[layer_idx] = {
            "layer_idx": layer_idx,
            "basis": layer_model.basis.cpu(),
            "target_center": layer_model.target_center.cpu(),
            "reference_center": layer_model.reference_center.cpu(),
            "target_label": layer_model.target_label,
            "reference_label": layer_model.reference_label,
        }

    metric_key = f"val_{cfg.method.selection_metric}"
    best_layer_row = select_best_layer(per_layer_rows, metric_name=metric_key)
    best_layer_idx = int(best_layer_row["layer_idx"])

    artifact_path = output_root / "subspace_artifacts.pt"
    torch.save(
        {
            "config_path": str(Path(args.config).resolve()),
            "train_split": train_split.split_dir,
            "val_split": val_split.split_dir,
            "test_split": "" if test_split is None else test_split.split_dir,
            "rank": cfg.method.rank,
            "target_label": cfg.method.target_label,
            "reference_label": cfg.method.reference_label,
            "selection_metric": cfg.method.selection_metric,
            "best_layer_idx": best_layer_idx,
            "best_threshold": float(best_layer_row["threshold"]),
            "models": serialized_models,
        },
        artifact_path,
    )

    _write_csv(per_layer_rows, output_root / "layer_metrics.csv")
    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "train_split_dir": train_split.split_dir,
            "train_label_counts": train_split.label_counts(),
            "val_split_dir": val_split.split_dir,
            "val_label_counts": val_split.label_counts(),
            "test_split_dir": "" if test_split is None else test_split.split_dir,
            "test_label_counts": {} if test_split is None else test_split.label_counts(),
            "rank": cfg.method.rank,
            "target_label": cfg.method.target_label,
            "reference_label": cfg.method.reference_label,
            "selection_metric": cfg.method.selection_metric,
            "best_layer": best_layer_row,
            "artifact_path": str(artifact_path),
            "num_layers_evaluated": len(per_layer_rows),
        },
    )

    print(json.dumps(best_layer_row, ensure_ascii=False, indent=2))
    print(artifact_path)


if __name__ == "__main__":
    main()
