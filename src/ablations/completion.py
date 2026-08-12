from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import sha256_file


class CompletionError(RuntimeError):
    """Raised when a backend run lacks evidence required by its contract."""


def _read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionError(f"missing or invalid {label}: {path}") from exc
    if not isinstance(value, dict) or not value:
        raise CompletionError(f"empty or invalid {label}: {path}")
    return value


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _latest_pan_results(phase1_root: Path) -> tuple[Path, dict]:
    candidates = sorted(
        (phase1_root / "training" / "eval_suite").glob("epoch_*/pan_results.json")
    )
    if not candidates:
        raise CompletionError("real backend did not produce per-sample evaluation predictions")
    path = candidates[-1]
    payload = _read_object(path, "PAN prediction result")
    rows = payload.get("generations")
    if payload.get("status") != "ok" or not isinstance(rows, list) or not rows:
        raise CompletionError("real backend prediction result is not complete")
    if any(not isinstance(row, Mapping) for row in rows):
        raise CompletionError("real backend prediction rows must be objects")
    return path, payload


def _source_record(path: Path, payload: Mapping) -> dict:
    return {
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "payload": dict(payload),
    }


def collect_training_contract(
    output_dir: str | Path,
    required_artifacts: Sequence[str],
    phase1_root: str | Path,
    *,
    cell_spec: Mapping,
) -> None:
    """Derive completion artifacts exclusively from a successful real backend.

    Small JSON manifests are normalized into the stable ablation schema. Large
    binary bridge artifacts are copied byte-for-byte. Missing evidence always
    fails closed; this function never creates placeholder model results.
    """

    target = Path(output_dir)
    phase1 = Path(phase1_root)
    target.mkdir(parents=True, exist_ok=True)
    training_path = phase1 / "training" / "manifest.json"
    training = _read_object(training_path, "training manifest")
    axes = dict(cell_spec.get("axes") or {})
    experiment_id = str(cell_spec.get("experiment_id", ""))
    common = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "axes": axes,
        "training_manifest_sha256": sha256_file(training_path),
    }

    json_sources = {
        "subspace_manifest.json": phase1 / "safe_subspaces" / "manifest.json",
        "layer_selection.json": phase1 / "layer_analysis" / "teacher_key_layers.json",
        "pairing_manifest.json": phase1 / "layer_pairing" / "teacher_student_layer_pairs.json",
        "position_manifest.json": phase1 / "hidden_states" / "teacher_alignment" / "manifest.json",
        "semantic_manifest.json": phase1 / "semantic_coeffs_teacher_alignment" / "manifest.json",
        "bridge_audit.json": phase1 / "semantic_bases" / "vocab_index_map.json",
    }
    predictions: tuple[Path, dict] | None = None

    for name in required_artifacts:
        destination = target / name
        if name == "eval_predictions.jsonl":
            predictions = predictions or _latest_pan_results(phase1)
            source_path, result = predictions
            normalized = []
            for index, row in enumerate(result["generations"]):
                item = dict(row)
                sample_id = item.get("sample_id", item.get("id"))
                if sample_id is None or not str(sample_id).strip():
                    raise CompletionError(f"prediction row {index} lacks a stable sample_id")
                item["sample_id"] = str(sample_id)
                item["source_result"] = str(source_path.resolve())
                normalized.append(item)
            _write_jsonl(destination, normalized)
            continue
        if name == "bridge_artifact.pt":
            source = phase1 / "semantic_bases" / "bridge_artifact.pt"
            if not source.is_file() or source.stat().st_size == 0:
                raise CompletionError(f"missing real bridge artifact: {source}")
            shutil.copyfile(source, destination)
            continue
        if name == "search_ledger.jsonl":
            _write_jsonl(
                destination,
                [{**common, "budget": {key: training.get(key) for key in ("epochs_completed", "train_num_samples", "trainable_parameters")}}],
            )
            continue

        payload: dict
        if name in json_sources:
            source = json_sources[name]
            payload = {**common, **_source_record(source, _read_object(source, name))}
        elif name == "run_manifest.json" or name == "training_manifest.json":
            payload = {**common, **_source_record(training_path, training)}
        elif name == "parameter_budget.json" or name == "budget_summary.json":
            required_budget = ("trainable_parameters", "total_parameters", "epochs_completed", "train_num_samples")
            if any(key not in training for key in required_budget):
                raise CompletionError("training manifest lacks exact parameter/training budget fields")
            payload = {**common, **{key: training[key] for key in required_budget}}
        elif name == "permutation_manifest.json":
            manifests = training.get("target_permutation_manifests")
            if not isinstance(manifests, Mapping) or not manifests:
                raise CompletionError("training manifest lacks target permutation evidence")
            validated = {}
            for split, raw_path in manifests.items():
                path = Path(str(raw_path))
                validated[str(split)] = _source_record(path, _read_object(path, "permutation manifest"))
            payload = {**common, "splits": validated}
        elif name == "sampling_manifest.json":
            if "train_num_samples" not in training:
                raise CompletionError("training manifest lacks actual sample count")
            payload = {**common, "train_num_samples": training["train_num_samples"], "requested": axes}
        elif name == "curation_manifest.json":
            extraction_path = phase1 / "hidden_states" / "teacher_alignment" / "manifest.json"
            payload = {**common, **_source_record(extraction_path, _read_object(extraction_path, "curated extraction manifest"))}
        elif name == "failure_analysis.json":
            predictions = predictions or _latest_pan_results(phase1)
            payload = {**common, "evaluation": _source_record(predictions[0], predictions[1]), "training": training}
        elif name == "teacher_quality.json":
            predictions = predictions or _latest_pan_results(phase1)
            payload = {**common, "teacher_variant": axes.get("teacher"), "student_evaluation": _source_record(predictions[0], predictions[1])}
        else:
            raise CompletionError(f"no real completion collector is registered for {name}")
        _write_json(destination, payload)
