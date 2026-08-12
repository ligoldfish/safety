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
from .stability import layer_jaccard, principal_angles, projection_overlap


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
    "causal_intervention": ("subspace_artifact", "intervention_data", "intervention_model"),
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
    rows = _read_jsonl(inputs["model_registry"])
    selectors = {
        key: str(axes[key])
        for key in ("pair", "dataset", "method")
        if key in axes
    }
    if selectors:
        rows = [
            row
            for row in rows
            if all(str(row.get(key, "")) == value for key, value in selectors.items())
        ]
        if len(rows) != 1:
            raise HandlerBlocked(
                f"main-table provenance cell must match exactly one registry row: {selectors}"
            )
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
    path = inputs["common_test"]
    rows = _read_jsonl(path if path.is_file() else path / "scores.jsonl")
    requested_suite = str(axes.get("test_suite", ""))
    if requested_suite:
        rows = [row for row in rows if str(row.get("test_suite", "")) == requested_suite]
    if not rows:
        raise HandlerBlocked(f"no common-test rows for suite: {requested_suite}")
    corpora = sorted({str(row["train_corpus"]) for row in rows})
    suites = sorted({str(row["test_suite"]) for row in rows})
    return {"cross_corpus_matrix.json": {"train_corpora": corpora, "test_suites": suites, "matrix": summarize_corpus_matrix(rows, corpora=corpora, suites=suites)}}


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
    rows = _read_jsonl(inputs["pan_predictions"])
    grouping = str(axes.get("grouping", "attack_family"))
    if grouping not in {"attack_family", "benign_length", "benign_topic"}:
        raise HandlerBlocked(f"unsupported PAN subgroup axis: {grouping}")
    groups: dict[str, list[float]] = {}
    for row in rows:
        group = str(row.get(grouping) or (pan_bucket(row) if grouping == "attack_family" else "unknown"))
        groups.setdefault(group, []).append(float(row["unsafe"]))
    return {"pan_subgroups.json": {name: {"n": len(values), "asr": sum(values) / len(values)} for name, values in sorted(groups.items())}}


def _representation(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    rows = _read_jsonl(inputs["aligned_predictions"])
    label = str(axes.get("label", ""))
    if label:
        rows = [row for row in rows if str(row.get("label", "")) == label]
    if not rows:
        raise HandlerBlocked(f"no aligned representation rows for label: {label}")
    pre = [float(row["cosine_pre"]) for row in rows]
    post = [float(row["cosine_post"]) for row in rows]
    behavior = [float(row["behavior_delta"]) for row in rows]
    delta = [right - left for left, right in zip(pre, post)]
    return {"representation_behavior.json": {"label": label or "all", "n": len(rows), "mean_cosine_delta": sum(delta) / len(delta), "spearman": spearman_correlation(delta, behavior)}}


def _efficiency(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    rows = _read_jsonl(inputs["phase_runtime_logs"])
    required = {"phase", "wall_seconds", "peak_memory_bytes", "disk_delta_bytes", "device_hours"}
    if any(not required <= set(row) for row in rows):
        raise HandlerBlocked("phase runtime log is missing required efficiency fields")
    phase = str(axes.get("phase", ""))
    if phase:
        rows = [row for row in rows if str(row["phase"]) == phase]
    if not rows:
        raise HandlerBlocked(f"no efficiency records for phase: {phase}")
    return {"efficiency_profile.json": {"phase": phase or "all", "phases": rows, "wall_seconds": sum(float(row["wall_seconds"]) for row in rows), "device_hours": sum(float(row["device_hours"]) for row in rows), "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in rows)}}


def _ethics(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    del axes
    registry = _read_json(inputs["dataset_registry"])
    manifests = _read_json(inputs["split_manifests"])
    datasets = registry.get("datasets", registry) if isinstance(registry, dict) else None
    if not isinstance(datasets, Mapping) or not datasets:
        raise HandlerBlocked("dataset registry must contain dataset metadata")
    missing_license = [name for name, item in datasets.items() if not str(item.get("license", "")).strip()]
    return {"ethics_data_audit.json": {"dataset_count": len(datasets), "license_coverage": (len(datasets) - len(missing_license)) / len(datasets), "missing_license": missing_license, "split_manifests": manifests}}


def _manual_agreement(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    root = inputs["human_annotations"]
    if not root.is_dir():
        raise HandlerBlocked("human_annotations must be a directory containing the blind audit files")
    predictions = _read_jsonl(root / "judge_predictions.jsonl")
    key = _read_json(root / "blind_key.json")
    if not isinstance(key, Mapping) or not key:
        raise HandlerBlocked("blind annotation key must be a non-empty object")
    try:
        annotated = import_double_annotations(
            key,
            _read_jsonl(root / "rater_a.jsonl"),
            _read_jsonl(root / "rater_b.jsonl"),
            allowed_labels={"safe", "unsafe"},
        )
    except ValueError as exc:
        raise HandlerBlocked(str(exc)) from exc
    stratum = str(axes.get("stratum", ""))
    if stratum:
        annotated = [row for row in annotated if str(row.get("stratum", "")) == stratum]
        predictions = [row for row in predictions if str(row.get("stratum", "")) == stratum]
    if not annotated or not predictions:
        raise HandlerBlocked(f"manual audit has no rows for stratum: {stratum}")
    sample_to_wildguard = {
        str(row.get("sample_id", "")): str(row.get("wildguard_label", ""))
        for row in predictions
    }
    if set(sample_to_wildguard) != {str(row["sample_id"]) for row in annotated}:
        raise HandlerBlocked("WildGuard and human judgments must cover identical sample IDs")
    left = [str(row["rater_a"]) for row in annotated]
    right = [str(row["rater_b"]) for row in annotated]
    wildguard = [sample_to_wildguard[str(row["sample_id"])] for row in annotated]
    consensus = [a if a == b else "disagreement" for a, b in zip(left, right)]
    comparable = [(w, h) for w, h in zip(wildguard, consensus) if h != "disagreement"]
    return {
        "judge_predictions.jsonl": predictions,
        "manual_audit_summary.json": {
            "stratum": stratum or "all",
            "n": len(annotated),
            "human_human_kappa": cohen_kappa(left, right),
            "human_human_agreement": sum(a == b for a, b in zip(left, right)) / len(left),
            "wildguard_human_agreement": (
                sum(a == b for a, b in comparable) / len(comparable) if comparable else None
            ),
            "wildguard_model_path": str(inputs["wildguard_model"]),
        },
    }


def _bootstrap_subspace(inputs: Mapping[str, Path], axes: Mapping) -> dict[str, object]:
    try:
        import torch

        from src.ablations.strategies.layers import LayerCandidate, select_layers
        from src.features.subspace import build_teacher_safe_subspace
        from src.phase_b.hidden_states import load_hidden_state_split
    except ImportError as exc:  # pragma: no cover - environment preflight owns this
        raise HandlerBlocked("subspace bootstrap requires the project torch environment") from exc
    split = load_hidden_state_split(inputs["alignment_hidden_states"])
    harmful_mask = torch.tensor([label == "harmful" for label in split.labels], dtype=torch.bool)
    harmless_mask = torch.tensor([label == "harmless" for label in split.labels], dtype=torch.bool)
    if not bool(harmful_mask.any()) or not bool(harmless_mask.any()):
        raise HandlerBlocked("subspace bootstrap requires both harmful and harmless samples")
    draw = int(axes.get("draw", 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42 + draw)
    layer_rows = []
    baseline_candidates = []
    bootstrap_candidates = []
    baseline_subspaces = {}
    bootstrap_subspaces = {}
    for layer_idx in split.available_layers:
        hidden = split.layer_tensors[layer_idx]
        harmful = hidden[harmful_mask]
        harmless = hidden[harmless_mask]
        baseline = build_teacher_safe_subspace(
            layer_idx=layer_idx,
            harmful_hidden=harmful,
            harmless_hidden=harmless,
            k=min(16, hidden.size(1)),
        )
        sampled_harmful = harmful.index_select(
            0, torch.randint(harmful.size(0), (harmful.size(0),), generator=generator)
        )
        sampled_harmless = harmless.index_select(
            0, torch.randint(harmless.size(0), (harmless.size(0),), generator=generator)
        )
        resampled = build_teacher_safe_subspace(
            layer_idx=layer_idx,
            harmful_hidden=sampled_harmful,
            harmless_hidden=sampled_harmless,
            k=baseline.k,
        )
        def effect(result, harmful_values, harmless_values):
            centered = torch.cat(
                [
                    harmful_values - result.harmful_mean,
                    harmless_values - result.harmless_mean,
                ],
                dim=0,
            )
            within = centered.norm(dim=1).mean().clamp_min(1e-6)
            return float((result.mean_diff.norm() / within).item())

        baseline_candidates.append(
            LayerCandidate(layer_idx, effect(baseline, harmful, harmless), 0.0)
        )
        bootstrap_candidates.append(
            LayerCandidate(
                layer_idx,
                effect(resampled, sampled_harmful, sampled_harmless),
                0.0,
            )
        )
        baseline_subspaces[layer_idx] = baseline
        bootstrap_subspaces[layer_idx] = resampled
    selection_k = min(5, len(split.available_layers))
    baseline_layers = select_layers(
        baseline_candidates, k=selection_k, mode="effect_only"
    )
    bootstrap_layers = select_layers(
        bootstrap_candidates, k=selection_k, mode="effect_only"
    )
    for layer_idx in sorted(set(baseline_layers) & set(bootstrap_layers)):
        baseline = baseline_subspaces[layer_idx]
        resampled = bootstrap_subspaces[layer_idx]
        angles = principal_angles(baseline.basis, resampled.basis)
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "rank": baseline.k,
                "principal_angles_radians": [float(value) for value in angles.tolist()],
                "mean_principal_angle_radians": float(angles.mean().item()),
                "projection_overlap": projection_overlap(baseline.basis, resampled.basis),
            }
        )
    return {
        "bootstrap_stability.json": {
            "draw": draw,
            "seed": 42 + draw,
            "sample_count": split.num_samples,
            "representation_mode": split.representation_mode,
            "selection_mode": "effect_only",
            "selection_k": selection_k,
            "baseline_key_layers": list(baseline_layers),
            "bootstrap_key_layers": list(bootstrap_layers),
            "layer_jaccard": layer_jaccard(baseline_layers, bootstrap_layers),
            "layers": layer_rows,
        }
    }


_IMPLEMENTATIONS: dict[str, Callable[[Mapping[str, Path], Mapping], dict[str, object]]] = {
    "provenance_matrix": _provenance,
    "seed_and_paired_bootstrap": _bootstrap,
    "validation_iso_hr": _iso_hr,
    "cross_corpus_matrix": _corpus,
    "pan_subgroup_analysis": _pan_subgroups,
    "representation_behavior_analysis": _representation,
    "efficiency_profile": _efficiency,
    "ethics_data_audit": _ethics,
    "judge_agreement_audit": _manual_agreement,
    "subspace_bootstrap": _bootstrap_subspace,
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
