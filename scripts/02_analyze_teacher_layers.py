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

from src.features.layer_scoring import LayerScoreResult, score_teacher_layer, top_k_layers
from src.phase_b.hidden_states import load_hidden_state_split
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze teacher layers and select top-k safety key layers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="teacher_alignment",
        help="Teacher hidden-state split directory under output_root/hidden_states.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="teacher_analysis_val",
        help="Teacher validation hidden-state split directory under output_root/hidden_states.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many teacher key layers to keep.",
    )
    parser.add_argument(
        "--train-max-samples-per-label",
        type=int,
        default=0,
        help="Optional cap per label on the teacher alignment split.",
    )
    parser.add_argument(
        "--val-max-samples-per-label",
        type=int,
        default=0,
        help="Optional cap per label on the teacher validation split.",
    )
    parser.add_argument(
        "--probe-max-iter",
        type=int,
        default=100,
        help="LBFGS max iterations for the simple linear probe.",
    )
    parser.add_argument(
        "--probe-weight-decay",
        type=float,
        default=1e-4,
        help="L2 penalty for the simple linear probe.",
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


def _serialize_layer_result(result: LayerScoreResult) -> Dict[str, object]:
    return {
        "layer_idx": result.layer_idx,
        "harmful_count": result.harmful_count,
        "harmless_count": result.harmless_count,
        "mean_diff_norm": result.mean_diff_norm,
        "linear_probe_acc": result.linear_probe_acc,
        "final_score": result.final_score,
    }


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)

    hidden_root = Path(cfg.extraction.output_root) / "hidden_states"
    output_root = ensure_dir(Path(cfg.extraction.output_root) / "layer_analysis")
    logger, log_path = setup_stage_logger("02_analyze_teacher_layers", Path(cfg.extraction.output_root) / "logs")
    train_split = load_hidden_state_split(
        hidden_root / args.train_dir,
        max_samples_per_label=args.train_max_samples_per_label,
    )
    val_split = load_hidden_state_split(
        hidden_root / args.val_dir,
        max_samples_per_label=args.val_max_samples_per_label,
        selected_layers=train_split.available_layers,
    )
    mean_diff_root = ensure_dir(output_root / "teacher_mean_diff")
    log_kv(
        logger,
        "layer_analysis_setup",
        config_path=str(Path(args.config).resolve()),
        train_split_dir=train_split.split_dir,
        val_split_dir=val_split.split_dir,
        train_label_counts=train_split.label_counts(),
        val_label_counts=val_split.label_counts(),
        top_k=int(args.top_k),
        available_layers=train_split.available_layers,
        log_path=str(log_path),
    )

    results: List[LayerScoreResult] = []
    for layer_idx in train_split.available_layers:
        result = score_teacher_layer(
            layer_idx=layer_idx,
            train_hidden=train_split.layer_tensors[layer_idx],
            train_labels=train_split.labels,
            val_hidden=val_split.layer_tensors[layer_idx],
            val_labels=val_split.labels,
            probe_max_iter=args.probe_max_iter,
            probe_weight_decay=args.probe_weight_decay,
        )
        results.append(result)
        torch.save(
            {
                "layer_idx": result.layer_idx,
                "harmful_count": result.harmful_count,
                "harmless_count": result.harmless_count,
                "harmful_mean": result.harmful_mean,
                "harmless_mean": result.harmless_mean,
                "mean_diff": result.mean_diff,
                "mean_diff_norm": result.mean_diff_norm,
            },
            mean_diff_root / f"teacher_mean_diff_layer_{layer_idx:02d}.pt",
        )

    sorted_results = sorted(results, key=lambda item: (-item.final_score, item.layer_idx))
    key_layer_results = top_k_layers(sorted_results, args.top_k)
    key_layers = [result.layer_idx for result in key_layer_results]
    csv_rows = [_serialize_layer_result(result) for result in sorted_results]

    _write_csv(csv_rows, output_root / "teacher_layer_scores.csv")
    write_json(
        output_root / "teacher_key_layers.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "teacher_model": cfg.teacher.name,
            "teacher_num_layers": len(train_split.available_layers),
            "train_split_dir": train_split.split_dir,
            "val_split_dir": val_split.split_dir,
            "top_k": args.top_k,
            "key_layers": key_layers,
            "layer_rows": csv_rows,
        },
    )
    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "teacher_model": cfg.teacher.name,
            "teacher_num_layers": len(train_split.available_layers),
            "train_split_dir": train_split.split_dir,
            "train_label_counts": train_split.label_counts(),
            "val_split_dir": val_split.split_dir,
            "val_label_counts": val_split.label_counts(),
            "top_k": args.top_k,
            "selected_key_layers": key_layers,
            "teacher_layer_scores_csv": str(output_root / "teacher_layer_scores.csv"),
            "teacher_key_layers_json": str(output_root / "teacher_key_layers.json"),
        },
    )
    log_kv(
        logger,
        "layer_analysis_complete",
        key_layers=key_layers,
        top_rows=csv_rows[: min(5, len(csv_rows))],
        output_root=str(output_root),
    )

    print(json.dumps({"key_layers": key_layers, "num_layers": len(sorted_results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
