from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .analysis import pan_bucket, spearman_correlation, summarize_corpus_matrix
from .data_audit import audit_train_eval_splits
from .manual_audit import build_blind_packet, import_double_annotations
from .statistics import cohen_kappa, paired_bootstrap


class HandlerBlocked(RuntimeError):
    """Raised when a real analysis input is absent or incomplete."""


@dataclass(frozen=True)
class HandlerContract:
    required_inputs: tuple[str, ...]
    description: str


_INPUTS = {
    "provenance_matrix": ("model_registry", "dataset_registry"),
    "matched_controls": ("phase1_data", "phasef_data"),
    "judge_agreement_audit": ("wildguard_model", "human_annotations"),
    "seed_and_paired_bootstrap": ("aligned_sample_predictions",),
    "validation_iso_hr": ("validation_predictions", "test_predictions"),
    "wjb_failure_analysis": ("wildjailbreak_data", "common_test"),
    "global_default_budget": ("search_ledger",),
    "cross_corpus_matrix": ("trained_checkpoints", "common_test"),
    "target_control": ("semantic_targets",),
    "subspace_control": ("teacher_hidden_states",),
    "bridge_control": ("alignment_hidden_states",),
    "layer_selection_control": ("layer_scores",),
    "layer_score_control": ("layer_scores",),
    "pairing_control": ("teacher_student_alignment_hidden_states",),
    "representation_position_control": ("phase1_data",),
    "semantic_top_m_sweep": ("semantic_bases",),
    "semantic_selection_control": ("semantic_bases",),
    "layer_loss_weight_sweep": ("phasef_data",),
    "supervision_policy_control": ("phasef_data",),
    "layer_loss_kind_control": ("phasef_data",),
    "lora_capacity_control": ("phasef_data",),
    "subspace_hyperparameter_sweep": ("teacher_hidden_states",),
    "data_efficiency_sweep": ("phase1_data", "phasef_data"),
    "curation_control": ("phase1_data",),
    "pan_subgroup_analysis": ("pan_predictions", "pan_metadata"),
    "general_capability_suite": ("trained_checkpoints", "benchmark_assets"),
    "representation_behavior_analysis": ("pre_post_hidden_states", "aligned_predictions"),
    "subspace_bootstrap": ("alignment_hidden_states",),
    "causal_intervention": ("subspace_artifact", "intervention_data"),
    "teacher_quality_control": ("teacher_checkpoints",),
    "cross_tokenizer_bridge": ("cross_family_models", "tokenizer_metadata"),
    "decoding_robustness": ("trained_checkpoints", "evaluation_data"),
    "efficiency_profile": ("phase_runtime_logs",),
    "ethics_data_audit": ("dataset_registry", "split_manifests"),
}


def handler_contracts() -> dict[str, HandlerContract]:
    return {
        name: HandlerContract(
            required_inputs=inputs,
            description=f"Validate declared inputs and compute the real {name} result contract.",
        )
        for name, inputs in _INPUTS.items()
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise HandlerBlocked(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandlerBlocked(f"cannot parse required input: {path}") from exc
    if not rows:
        raise HandlerBlocked(f"required input is empty: {path}")
    return rows


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandlerBlocked(f"cannot parse required input: {path}") from exc
    return value


def _resolve_inputs(handler: str, spec: Mapping) -> dict[str, Path]:
    contract = handler_contracts()[handler]
    raw = spec.get("inputs") or {}
    if not isinstance(raw, Mapping):
        raise HandlerBlocked("cell inputs must be a mapping")
    result = {}
    for name in contract.required_inputs:
        value = str(raw.get(name, "")).strip()
        if not value:
            raise HandlerBlocked(f"missing required input declaration: {name}")
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise HandlerBlocked(f"missing required input {name}: {path}")
        result[name] = path
    return result


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _provenance(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    rows = _read_jsonl(inputs["model_registry"])
    required = {"cell_id", "model_hash", "dataset_hash", "config_hash", "checkpoint_hash", "commit"}
    audited = []
    for row in rows:
        missing = sorted(key for key in required if not str(row.get(key, "")).strip())
        audited.append({**row, "traceable": not missing, "missing_fields": missing})
    covered = sum(bool(row["traceable"]) for row in audited)
    return {
        "provenance_matrix.jsonl": audited,
        "coverage_summary.json": {
            "cells": len(audited),
            "traceable_cells": covered,
            "coverage_rate": covered / len(audited),
            "missing_cells": [row["cell_id"] for row in audited if not row["traceable"]],
        },
    }


def _bootstrap(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    rows = _read_jsonl(inputs["aligned_sample_predictions"])
    required = {"sample_id", "left", "right"}
    if any(not required <= set(row) for row in rows):
        raise HandlerBlocked("aligned predictions require sample_id, left, and right")
    draws = int(axes.get("bootstrap_draws", 10000))
    seed = int(axes.get("seed", 42))
    result = paired_bootstrap(
        [float(row["left"]) for row in rows],
        [float(row["right"]) for row in rows],
        [str(row["sample_id"]) for row in rows],
        [str(row["sample_id"]) for row in rows],
        draws=draws,
        seed=seed,
    )
    return {
        "paired_bootstrap.json": result,
        "seed_summary.json": {"seed": seed, "n": len(rows), "mean_difference": result["mean_difference"]},
    }


def _corpus(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    path = inputs["common_test"]
    rows = _read_jsonl(path if path.is_file() else path / "scores.jsonl")
    corpora = sorted({str(row["train_corpus"]) for row in rows})
    suites = sorted({str(row["test_suite"]) for row in rows})
    return {"cross_corpus_matrix.json": summarize_corpus_matrix(rows, corpora=corpora, suites=suites)}


def _iso_hr(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    if str(axes.get("selection_split", "validation")) != "validation":
        raise HandlerBlocked("ISO-HR selection must use validation only")
    validation = _read_jsonl(inputs["validation_predictions"])
    test = {str(row["checkpoint"]): row for row in _read_jsonl(inputs["test_predictions"])}
    target = next((float(row["hr"]) for row in validation if row.get("method") == "ours"), None)
    if target is None:
        raise HandlerBlocked("validation predictions lack the ours target HR")
    selected = {}
    for method in sorted({str(row["method"]) for row in validation}):
        candidates = [row for row in validation if str(row["method"]) == method]
        winner = min(candidates, key=lambda row: (abs(float(row["hr"]) - target), str(row["checkpoint"])))
        checkpoint = str(winner["checkpoint"])
        if checkpoint not in test:
            raise HandlerBlocked(f"selected checkpoint lacks test result: {checkpoint}")
        selected[method] = {"checkpoint": checkpoint, "validation_hr": float(winner["hr"]), "test": test[checkpoint]}
    return {"iso_hr_comparison.json": {"selection_split": "validation", "target_hr": target, "methods": selected}}


def _pan_subgroups(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    rows = _read_jsonl(inputs["pan_predictions"])
    groups: dict[str, list[float]] = {}
    for row in rows:
        group = str(row.get("attack_family") or pan_bucket(row))
        groups.setdefault(group, []).append(float(row["unsafe"]))
    return {"pan_subgroups.json": {name: {"n": len(values), "asr": sum(values) / len(values)} for name, values in sorted(groups.items())}}


def _representation(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    rows = _read_jsonl(inputs["aligned_predictions"])
    pre = [float(row["cosine_pre"]) for row in rows]
    post = [float(row["cosine_post"]) for row in rows]
    behavior = [float(row["behavior_delta"]) for row in rows]
    delta = [right - left for left, right in zip(pre, post)]
    return {"representation_behavior.json": {"n": len(rows), "mean_cosine_delta": sum(delta) / len(delta), "spearman": spearman_correlation(delta, behavior)}}


def _efficiency(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    rows = _read_jsonl(inputs["phase_runtime_logs"])
    required = {"phase", "wall_seconds", "peak_memory_bytes", "disk_delta_bytes", "device_hours"}
    if any(not required <= set(row) for row in rows):
        raise HandlerBlocked("phase runtime log is missing required efficiency fields")
    return {"efficiency_profile.json": {"phases": rows, "wall_seconds": sum(float(row["wall_seconds"]) for row in rows), "device_hours": sum(float(row["device_hours"]) for row in rows), "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in rows)}}


def _ethics(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    registry = _read_json(inputs["dataset_registry"])
    manifests = _read_json(inputs["split_manifests"])
    datasets = registry.get("datasets", registry) if isinstance(registry, dict) else None
    if not isinstance(datasets, Mapping) or not datasets:
        raise HandlerBlocked("dataset registry must contain dataset metadata")
    missing_license = [name for name, item in datasets.items() if not str(item.get("license", "")).strip()]
    return {"ethics_data_audit.json": {"dataset_count": len(datasets), "license_coverage": (len(datasets) - len(missing_license)) / len(datasets), "missing_license": missing_license, "split_manifests": manifests}}


_IMPLEMENTATIONS: dict[str, Callable[[Mapping[str, Path], Mapping], dict[str, object]]] = {
    "provenance_matrix": _provenance,
    "seed_and_paired_bootstrap": _bootstrap,
    "validation_iso_hr": _iso_hr,
    "cross_corpus_matrix": _corpus,
    "pan_subgroup_analysis": _pan_subgroups,
    "representation_behavior_analysis": _representation,
    "efficiency_profile": _efficiency,
    "ethics_data_audit": _ethics,
}


def execute_handler(
    handler: str,
    spec: Mapping,
    *,
    output_dir: str | Path,
    required_artifacts: Sequence[str],
) -> None:
    if handler not in handler_contracts():
        raise ValueError(f"unknown ablation handler: {handler}")
    inputs = _resolve_inputs(handler, spec)
    implementation = _IMPLEMENTATIONS.get(handler)
    if implementation is None:
        raise HandlerBlocked(
            f"{handler} is executed by the training/evaluation backend, not the analysis worker"
        )
    results = implementation(inputs, dict(spec.get("axes") or {}))
    missing = sorted(set(required_artifacts) - set(results))
    if missing:
        raise HandlerBlocked(f"handler did not compute required artifacts: {missing}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in required_artifacts:
        value = results[name]
        if name.endswith(".jsonl"):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise HandlerBlocked(f"handler result for {name} must be rows")
            _write_jsonl(target / name, value)
        else:
            _write_json(target / name, value)
