from __future__ import annotations

import json
import importlib.util
import math
import shutil
import sys
from collections import Counter
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


def _read_jsonl(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CompletionError(f"{label} row {line_number} must be an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionError(f"missing or invalid {label}: {path}") from exc
    if not rows:
        raise CompletionError(f"empty {label}: {path}")
    return rows


def _validated_search_ledger(cell_spec: Mapping) -> tuple[Path, list[dict], dict]:
    axes = dict(cell_spec.get("axes") or {})
    inputs = dict(cell_spec.get("inputs") or {})
    source_text = str(inputs.get("search_ledger", "")).strip()
    if not source_text:
        raise CompletionError("P0-07 requires a declared real search_ledger input")
    source = Path(source_text)
    dataset = str(axes.get("dataset", ""))
    config = str(axes.get("config", ""))
    rows = [
        row
        for row in _read_jsonl(source, "search ledger")
        if str(row.get("dataset", "")) == dataset
        and str(row.get("config", "")) == config
    ]
    if not rows:
        raise CompletionError(f"search ledger has no rows for dataset={dataset}, config={config}")
    required = {
        "trial_id",
        "dataset",
        "config",
        "method",
        "selection_split",
        "selected",
        "validation_metric",
    }
    if any(not required <= set(row) for row in rows):
        raise CompletionError("search ledger lacks required trial provenance fields")
    if any(str(row["selection_split"]) != "validation" for row in rows):
        raise CompletionError("search ledger contains non-validation model selection")
    counts: dict[str, int] = {}
    selected: dict[str, str] = {}
    for row in rows:
        method = str(row["method"])
        counts[method] = counts.get(method, 0) + 1
        if bool(row["selected"]):
            if method in selected:
                raise CompletionError(f"method {method} has multiple selected search trials")
            selected[method] = str(row["trial_id"])
    if len(counts) < 2 or len(set(counts.values())) != 1:
        raise CompletionError(f"search budgets must be equal across methods: {counts}")
    if config == "validation_selected" and set(selected) != set(counts):
        raise CompletionError("validation_selected requires exactly one selected trial per method")
    return source, rows, {
        "search_count": len(rows),
        "search_count_by_method": dict(sorted(counts.items())),
        "selected_trial_ids": dict(sorted(selected.items())),
        "selection_split": "validation",
    }


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


def _finite_rate(payload: Mapping, name: str, label: str) -> float:
    value = payload.get(name)
    if type(value) not in {int, float}:
        raise CompletionError(f"{label} lacks numeric {name}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CompletionError(f"{label} has invalid {name}: {value}")
    return number


def _validated_pan_result(path: Path, label: str) -> dict:
    payload = _read_object(path, label)
    rows = payload.get("generations")
    if payload.get("status") != "ok" or not isinstance(rows, list) or not rows:
        raise CompletionError(f"{label} lacks successful real generations")
    for name in (
        "harmful_refusal_rate",
        "harmful_unsafe_output_rate",
        "harmless_over_refusal_rate",
    ):
        _finite_rate(payload, name, label)
    return payload


def _validated_wildguard_result(path: Path, label: str) -> dict:
    payload = _read_object(path, label)
    if payload.get("judge") != "wildguard":
        raise CompletionError(f"{label} was not produced by WildGuard")
    if int(payload.get("num_unmatched_ids", -1)) != 0:
        raise CompletionError(f"{label} has unmatched sample IDs")
    if _finite_rate(payload, "judge_parse_rate", label) <= 0.0:
        raise CompletionError(f"{label} has zero parsed WildGuard verdicts")
    for name in ("llm_judge_asr", "llm_judge_refusal_rate", "llm_judge_over_refusal"):
        _finite_rate(payload, name, label)
    return payload


def _refusal_classifier():
    module_path = Path(__file__).resolve().parents[1] / "training" / "eval_utils.py"
    name = "_ablation_completion_eval_utils"
    if name in sys.modules:
        return sys.modules[name].looks_like_refusal
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise CompletionError(f"cannot load refusal metric implementation: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.looks_like_refusal


def _target_text(row: Mapping) -> str:
    for key in ("target_response", "rejected_response", "response", "target"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    for message in reversed(list(row.get("messages") or [])):
        if isinstance(message, Mapping) and str(message.get("role", "")).lower() == "assistant":
            value = str(message.get("content", "")).strip()
            if value:
                return value
    return ""


def _distribution(rows: Sequence[Mapping], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        value = str(metadata.get(key, row.get(key, ""))).strip() or "_missing"
        counts[value] += 1
    return dict(sorted(counts.items()))


def _failure_boundary_contract(phase1: Path, axes: Mapping) -> dict:
    boundary = phase1 / "training" / "failure_boundary"
    manifest_path = boundary / "evaluation_manifest.json"
    manifest = _read_object(manifest_path, "P0-06 evaluation manifest")
    if manifest.get("experiment_id") != "P0-06":
        raise CompletionError("P0-06 evaluation manifest has the wrong experiment")
    for key in ("pair", "config", "curation", "method"):
        if str(manifest.get(key, "")) != str(axes.get(key, "")):
            raise CompletionError(f"P0-06 evaluation manifest mismatches axis {key}")

    split_audit = manifest.get("split_audit")
    if not isinstance(split_audit, Mapping) or set(split_audit) != {
        "train_target_overlap",
        "train_common_overlap",
        "common_target_overlap",
    }:
        raise CompletionError("P0-06 evaluation manifest lacks a complete split audit")
    if any(type(value) is not int or value != 0 for value in split_audit.values()):
        raise CompletionError(f"P0-06 split audit contains overlap: {split_audit}")

    def declared_path(name: str, label: str) -> Path:
        value = str(manifest.get(name, "")).strip()
        if not value:
            raise CompletionError(f"P0-06 evaluation manifest lacks {name}")
        path = Path(value)
        if not path.is_file() or path.stat().st_size <= 0:
            raise CompletionError(f"missing {label}: {path}")
        return path

    training_path = declared_path("training_jsonl", "WildJailbreak training data")
    curation_path = declared_path("curation_summary", "WildJailbreak curation summary")
    target_path = declared_path("target_result", "WildJailbreak target evaluation")
    common_path = declared_path("common_result", "common-test evaluation")
    target_judge_path = declared_path("target_judge", "WildJailbreak WildGuard evaluation")
    common_judge_path = declared_path("common_judge", "common-test WildGuard evaluation")
    checkpoint_path = declared_path("adapter_checkpoint", "P0-06 adapter checkpoint")
    expected_checkpoint_hash = str(manifest.get("adapter_checkpoint_sha256", "")).strip()
    if not expected_checkpoint_hash or sha256_file(checkpoint_path) != expected_checkpoint_hash:
        raise CompletionError("P0-06 adapter checkpoint hash changed after evaluation")
    dataset_hashes = manifest.get("dataset_sha256")
    if not isinstance(dataset_hashes, Mapping):
        raise CompletionError("P0-06 evaluation manifest lacks dataset hashes")
    training_rows = _read_jsonl(training_path, "WildJailbreak training data")
    for name, path in (
        ("training", training_path),
        ("target_test", declared_path("target_test_jsonl", "WildJailbreak test data")),
        ("common_test", declared_path("common_test_jsonl", "common-test data")),
    ):
        if str(dataset_hashes.get(name, "")) != sha256_file(path):
            raise CompletionError(f"P0-06 {name} dataset hash changed after evaluation")
    curation = _read_object(curation_path, "WildJailbreak curation summary")
    if str(curation.get("baseline", "")) != "wildjailbreak":
        raise CompletionError("curation summary is not for WildJailbreak")
    if str(curation.get("mode", "")) != str(axes.get("curation", "")):
        raise CompletionError("curation summary does not match the cell curation axis")
    target = _validated_pan_result(target_path, "WildJailbreak target evaluation")
    common = _validated_pan_result(common_path, "common-test evaluation")
    target_judge = _validated_wildguard_result(
        target_judge_path, "WildJailbreak WildGuard evaluation"
    )
    common_judge = _validated_wildguard_result(
        common_judge_path, "common-test WildGuard evaluation"
    )

    harmful = [
        row for row in training_rows if str(row.get("label", "")).strip().lower() == "harmful"
    ]
    if not harmful:
        raise CompletionError("WildJailbreak training data contains no harmful targets")
    targets = [_target_text(row) for row in harmful]
    if any(not text for text in targets):
        raise CompletionError("WildJailbreak harmful row lacks a training target")
    looks_like_refusal = _refusal_classifier()
    lengths = [len(text.split()) for text in targets]
    target_refusal_rate = sum(bool(looks_like_refusal(text)) for text in targets) / len(targets)
    normalized_targets = {" ".join(text.lower().split()) for text in targets}
    template_diversity = len(normalized_targets) / len(targets)
    target_sources = _distribution(harmful, "target_source")
    data_types = _distribution(training_rows, "data_type")
    return {
        "schema_version": 1,
        "experiment_id": "P0-06",
        "axes": dict(axes),
        "metrics": {
            "target_refusal_rate": target_refusal_rate,
            "template_diversity": template_diversity,
            "common_test_asr": _finite_rate(common_judge, "llm_judge_asr", "common-test WildGuard evaluation"),
        },
        "data_audit": {
            "training_rows": len(training_rows),
            "harmful_targets": len(harmful),
            "unique_harmful_targets": len(normalized_targets),
            "template_pool_count": target_sources.get("template_pool", 0),
            "target_source_distribution": target_sources,
            "data_type_distribution": data_types,
            "target_length_words": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
            "curation": _source_record(curation_path, curation),
            "training_data_path": str(training_path.resolve()),
            "training_data_sha256": sha256_file(training_path),
        },
        "target_evaluation": {
            "keyword": _source_record(target_path, target),
            "wildguard": _source_record(target_judge_path, target_judge),
        },
        "common_evaluation": {
            "keyword": _source_record(common_path, common),
            "wildguard": _source_record(common_judge_path, common_judge),
        },
        "evaluation_manifest": _source_record(manifest_path, manifest),
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
    search_contract: tuple[Path, list[dict], dict] | None = None

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
            if experiment_id != "P0-07":
                raise CompletionError("search_ledger.jsonl is only valid for P0-07")
            search_contract = search_contract or _validated_search_ledger(cell_spec)
            _write_jsonl(destination, search_contract[1])
            continue

        payload: dict
        if name in json_sources:
            source = json_sources[name]
            payload = {**common, **_source_record(source, _read_object(source, name))}
        elif name == "run_manifest.json" or name == "training_manifest.json":
            payload = {**common, **_source_record(training_path, training)}
        elif name == "budget_summary.json" and experiment_id == "P0-07":
            search_contract = search_contract or _validated_search_ledger(cell_spec)
            source, _, search_summary = search_contract
            required_budget = ("trainable_parameters", "total_parameters", "epochs_completed", "train_num_samples")
            if any(key not in training for key in required_budget):
                raise CompletionError("training manifest lacks exact parameter/training budget fields")
            payload = {
                **common,
                **search_summary,
                "search_ledger_source": str(source.resolve()),
                "search_ledger_sha256": sha256_file(source),
                "training_budget": {key: training[key] for key in required_budget},
            }
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
            payload = {**common, **_failure_boundary_contract(phase1, axes), "training": training}
        elif name == "teacher_quality.json":
            predictions = predictions or _latest_pan_results(phase1)
            payload = {**common, "teacher_variant": axes.get("teacher"), "student_evaluation": _source_record(predictions[0], predictions[1])}
        else:
            raise CompletionError(f"no real completion collector is registered for {name}")
        _write_json(destination, payload)
