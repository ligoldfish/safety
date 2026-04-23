from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble phase-1 summary tables from the generated artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--training-dir-name",
        type=str,
        default="training",
        help="Training subdirectory name under the phase-1 output root.",
    )
    parser.add_argument(
        "--sanity-dir-name",
        type=str,
        default="sanity_eval",
        help="Sanity-eval subdirectory name under the phase-1 output root.",
    )
    parser.add_argument(
        "--tables-dir-name",
        type=str,
        default="tables",
        help="Output tables subdirectory name under the phase-1 output root.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    output_root = Path(cfg.extraction.output_root)
    tables_dir = ensure_dir(output_root / args.tables_dir_name)
    logger, log_path = setup_stage_logger("11_make_tables", output_root / "logs")

    layer_scores_path = output_root / "layer_analysis" / "teacher_layer_scores.csv"
    key_layers_path = output_root / "layer_analysis" / "teacher_key_layers.json"
    layer_pairing_path = output_root / "layer_pairing" / "teacher_student_layer_pairs.json"
    training_manifest_path = output_root / args.training_dir_name / "manifest.json"
    training_val_metrics_path = output_root / args.training_dir_name / "val_metrics.json"
    sanity_manifest_path = output_root / args.sanity_dir_name / "manifest.json"
    sanity_comparison_path = output_root / args.sanity_dir_name / "comparison.csv"

    layer_scores = _read_csv(layer_scores_path)
    key_layers_payload = _read_json(key_layers_path)
    key_layers = [int(layer_idx) for layer_idx in key_layers_payload["key_layers"]]
    key_layer_rows = [
        row for row in layer_scores
        if int(row["layer_idx"]) in key_layers
    ]
    key_layer_rows = sorted(
        key_layer_rows,
        key=lambda row: key_layers.index(int(row["layer_idx"])),
    )
    _write_csv(
        tables_dir / "table_key_layers.csv",
        key_layer_rows,
        fieldnames=list(key_layer_rows[0].keys()) if key_layer_rows else list(layer_scores[0].keys()),
    )

    pairing_payload = _read_json(layer_pairing_path)
    layer_pair_rows = [
        {
            "teacher_layer": int(item["teacher_layer"]),
            "student_layer": int(item["student_layer"]),
            "teacher_relative_depth": float(item["teacher_relative_depth"]),
        }
        for item in pairing_payload["pairs"]
    ]
    _write_csv(
        tables_dir / "table_layer_pairs.csv",
        layer_pair_rows,
        fieldnames=["teacher_layer", "student_layer", "teacher_relative_depth"],
    )

    val_metrics_payload = _read_json(training_val_metrics_path)
    training_rows: List[Dict[str, Any]] = []
    for epoch_name, metrics in sorted(
        val_metrics_payload.items(),
        key=lambda item: int(item[0].split("_")[-1]),
    ):
        epoch_idx = int(epoch_name.split("_")[-1])
        # Phase-G sanity evaluator is refusal-only; no safe/unsafe sub-split is
        # persisted. harmful_unsafe_output_rate = 1 - harmful_refusal_rate by
        # construction in trainer_phase1.evaluate_generation_refusal_metrics.
        training_rows.append(
            {
                "epoch": epoch_idx,
                "harmful_refusal_rate": metrics["harmful_refusal_rate"],
                "harmful_unsafe_output_rate": metrics["harmful_unsafe_output_rate"],
                "harmless_over_refusal_rate": metrics["harmless_over_refusal_rate"],
                "layer_target_cosine_mean": metrics["layer_target_cosine_mean"],
                "num_harmful": metrics["num_harmful"],
                "num_harmless": metrics["num_harmless"],
            }
        )
    _write_csv(
        tables_dir / "table_training_val.csv",
        training_rows,
        fieldnames=[
            "epoch",
            "harmful_refusal_rate",
            "harmful_unsafe_output_rate",
            "harmless_over_refusal_rate",
            "layer_target_cosine_mean",
            "num_harmful",
            "num_harmless",
        ],
    )

    sanity_rows = _read_csv(sanity_comparison_path)
    _write_csv(
        tables_dir / "table_sanity_comparison.csv",
        sanity_rows,
        fieldnames=["metric", "baseline", "semalign", "delta"],
    )

    training_manifest = _read_json(training_manifest_path)
    sanity_manifest = _read_json(sanity_manifest_path)
    overview = {
        "config_path": str(Path(args.config).resolve()),
        "teacher_model": cfg.teacher.name,
        "student_model": cfg.student.name,
        "key_layers": key_layers,
        "layer_pairs": layer_pair_rows,
        "training": {
            "paired_student_layers": training_manifest["paired_student_layers"],
            "train_num_samples": training_manifest["train_num_samples"],
            "val_num_samples": training_manifest["val_num_samples"],
            "epochs": training_manifest["epochs"],
            "trainable_parameters": training_manifest["trainable_parameters"],
        },
        "sanity_eval": {
            "num_records": sanity_manifest["num_records"],
            "label_counts": sanity_manifest["label_counts"],
            "comparison_csv": str(sanity_comparison_path),
        },
        "tables": {
            "key_layers_csv": str((tables_dir / "table_key_layers.csv").resolve()),
            "layer_pairs_csv": str((tables_dir / "table_layer_pairs.csv").resolve()),
            "training_val_csv": str((tables_dir / "table_training_val.csv").resolve()),
            "sanity_comparison_csv": str((tables_dir / "table_sanity_comparison.csv").resolve()),
        },
    }
    write_json(tables_dir / "phase1_overview.json", overview)
    log_kv(
        logger,
        "make_tables_complete",
        config_path=str(Path(args.config).resolve()),
        tables_dir=str(tables_dir),
        training_dir_name=args.training_dir_name,
        sanity_dir_name=args.sanity_dir_name,
        key_layers=key_layers,
        training_manifest_path=str(training_manifest_path),
        sanity_manifest_path=str(sanity_manifest_path),
        log_path=str(log_path),
    )

    print(
        json.dumps(
            {
                "tables_dir": str(tables_dir),
                "key_layers": key_layers,
                "num_training_epochs": len(training_rows),
                "num_sanity_metrics": len(sanity_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
