from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hf_loader import load_hf_model
from src.phase_b.subspace import LayerSubspaceModel, evaluate_layer_model, score_with_subspace
from src.phase_c.intervention import (
    build_intervention_spec,
    load_intervention_artifact,
    run_intervened_last_token_hidden,
)
from src.utils.config import load_phasec_config
from src.utils.io import chunked, ensure_dir, read_jsonl, write_json
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage-C safety intervention using the phase-B subspace artifact."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phaseC_cpu.yaml",
        help="Path to the phase-C YAML config.",
    )
    return parser.parse_args()


def _limit_records(records: List[Dict[str, object]], max_samples_per_label: int) -> List[Dict[str, object]]:
    if max_samples_per_label <= 0:
        return records

    kept: List[Dict[str, object]] = []
    counts: Counter[str] = Counter()
    for record in records:
        label = str(record["label"])
        if counts[label] >= max_samples_per_label:
            continue
        counts[label] += 1
        kept.append(record)
    return kept


def _layer_model_from_artifact(artifact) -> LayerSubspaceModel:
    return LayerSubspaceModel(
        layer_idx=artifact.best_layer_idx,
        basis=artifact.basis,
        target_center=artifact.target_center,
        reference_center=artifact.reference_center,
        target_label=artifact.target_label,
        reference_label=artifact.reference_label,
    )


def _collect_hidden_states(
    *,
    model,
    tokenizer,
    records: Sequence[Dict[str, object]],
    alpha: float,
    spec,
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    hidden_batches: List[torch.Tensor] = []
    labels: List[str] = []
    sample_ids: List[str] = []

    for batch in chunked(records, batch_size):
        messages_batch = [record["messages"] for record in batch]
        hidden = run_intervened_last_token_hidden(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            spec=spec,
            alpha=alpha,
            max_length=max_length,
        )
        hidden_batches.append(hidden)
        labels.extend(str(record["label"]) for record in batch)
        sample_ids.extend(str(record["id"]) for record in batch)

    return torch.cat(hidden_batches, dim=0), labels, sample_ids


def _evaluate_split(
    *,
    split_name: str,
    model,
    tokenizer,
    records: Sequence[Dict[str, object]],
    alpha: float,
    spec,
    layer_model: LayerSubspaceModel,
    max_length: int,
    batch_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    hidden_states, labels, sample_ids = _collect_hidden_states(
        model=model,
        tokenizer=tokenizer,
        records=records,
        alpha=alpha,
        spec=spec,
        max_length=max_length,
        batch_size=batch_size,
    )
    metrics = evaluate_layer_model(
        layer_model,
        hidden_states,
        labels,
        threshold=spec.threshold,
    )
    scores = score_with_subspace(layer_model, hidden_states)

    sample_rows: List[Dict[str, object]] = []
    label_scores: Dict[str, List[float]] = {layer_model.target_label: [], layer_model.reference_label: []}
    for sample_id, label, score in zip(sample_ids, labels, scores.tolist()):
        label_scores[str(label)].append(float(score))
        sample_rows.append(
            {
                "split": split_name,
                "alpha": alpha,
                "sample_id": sample_id,
                "label": label,
                "score": float(score),
                "prediction": layer_model.target_label if float(score) >= spec.threshold else layer_model.reference_label,
            }
        )

    row: Dict[str, object] = {
        "split": split_name,
        "alpha": alpha,
        "threshold": spec.threshold,
        "num_samples": len(labels),
        "target_label": layer_model.target_label,
        "reference_label": layer_model.reference_label,
        "target_score_mean": float(sum(label_scores[layer_model.target_label]) / max(1, len(label_scores[layer_model.target_label]))),
        "reference_score_mean": float(sum(label_scores[layer_model.reference_label]) / max(1, len(label_scores[layer_model.reference_label]))),
    }
    for key, value in metrics.items():
        row[key] = value
    return row, sample_rows


def _select_best_alpha(rows: Sequence[Dict[str, object]], metric_name: str) -> Dict[str, object]:
    if not rows:
        raise ValueError("rows must be non-empty when selecting alpha.")
    return sorted(
        rows,
        key=lambda row: (-float(row[metric_name]), abs(float(row["alpha"])), float(row["alpha"])),
    )[0]


def _write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
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
    cfg = load_phasec_config(args.config)
    set_global_seed(cfg.seed)

    artifact = load_intervention_artifact(cfg.inputs.artifact_path)
    spec = build_intervention_spec(artifact)
    layer_model = _layer_model_from_artifact(artifact)

    val_records = _limit_records(read_jsonl(cfg.inputs.val_split), cfg.limits.val_max_samples_per_label)
    test_records = _limit_records(read_jsonl(cfg.inputs.test_split), cfg.limits.test_max_samples_per_label)
    if not val_records or not test_records:
        raise ValueError("Phase-C validation and test splits must both be non-empty after filtering.")

    tokenizer, model, _ = load_hf_model(
        model_path=cfg.model.path,
        device_map=cfg.model.device_map,
        torch_dtype=cfg.model.torch_dtype,
        runtime_backend=cfg.model.runtime_backend,
        runtime_device=cfg.model.runtime_device,
        trust_remote_code=cfg.model.trust_remote_code,
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
    )

    val_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    for alpha in cfg.method.alphas:
        val_row, val_samples = _evaluate_split(
            split_name="val",
            model=model,
            tokenizer=tokenizer,
            records=val_records,
            alpha=float(alpha),
            spec=spec,
            layer_model=layer_model,
            max_length=cfg.method.max_length,
            batch_size=cfg.method.batch_size,
        )
        test_row, test_samples = _evaluate_split(
            split_name="test",
            model=model,
            tokenizer=tokenizer,
            records=test_records,
            alpha=float(alpha),
            spec=spec,
            layer_model=layer_model,
            max_length=cfg.method.max_length,
            batch_size=cfg.method.batch_size,
        )
        val_rows.append(val_row)
        test_rows.append(test_row)
        sample_rows.extend(val_samples)
        sample_rows.extend(test_samples)

    best_val_row = _select_best_alpha(val_rows, cfg.method.selection_metric)
    chosen_alpha = float(best_val_row["alpha"])
    chosen_test_row = next(row for row in test_rows if float(row["alpha"]) == chosen_alpha)
    baseline_test_row = next(row for row in test_rows if float(row["alpha"]) == 0.0)

    output_root = ensure_dir(cfg.output.output_root)
    _write_csv(val_rows, output_root / "alpha_sweep_val.csv")
    _write_csv(test_rows, output_root / "alpha_sweep_test.csv")
    _write_csv(sample_rows, output_root / "sample_scores.csv")

    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "artifact_path": artifact.artifact_path,
            "best_layer_idx_from_phase_b": artifact.best_layer_idx,
            "threshold_from_phase_b": artifact.best_threshold,
            "target_label": artifact.target_label,
            "reference_label": artifact.reference_label,
            "val_split_path": cfg.inputs.val_split,
            "test_split_path": cfg.inputs.test_split,
            "val_num_samples": len(val_records),
            "test_num_samples": len(test_records),
            "alphas": [float(alpha) for alpha in cfg.method.alphas],
            "selection_metric": cfg.method.selection_metric,
            "selected_alpha": chosen_alpha,
            "selected_val_row": best_val_row,
            "selected_test_row": chosen_test_row,
            "baseline_test_row": baseline_test_row,
        },
    )

    print(json.dumps(best_val_row, ensure_ascii=False, indent=2))
    print(json.dumps(chosen_test_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
