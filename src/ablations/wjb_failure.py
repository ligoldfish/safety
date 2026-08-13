from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from src.pairs import apply_tokens

from .artifacts import sha256_file
from .data_audit import prompt_sha256
from .preflight import inspect_model_directory


class FailureBoundaryError(RuntimeError):
    """Raised when a P0-06 boundary evaluation cannot be staged safely."""


@dataclass(frozen=True)
class FailureEvaluationPlan:
    commands: tuple[tuple[str, ...], ...]
    common_config: Path
    target_result: Path
    common_result: Path
    manifest: Path


def _nonempty_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FailureBoundaryError(f"missing non-empty {label}: {resolved}")
    return resolved


def _latest_target_result(phase1: Path) -> Path:
    candidates = sorted(
        (phase1 / "training" / "eval_suite").glob("epoch_*/pan_results.json")
    )
    if not candidates:
        raise FailureBoundaryError("P0-06 lacks a real WildJailbreak evaluation result")
    return _nonempty_file(candidates[-1], "WildJailbreak evaluation result")


def _latest_checkpoint(training: Path) -> Path:
    candidates = sorted((training / "checkpoints").glob("epoch_*.pt"))
    if not candidates:
        candidates = sorted(training.glob("epoch_*.pt"))
    if not candidates:
        raise FailureBoundaryError(f"P0-06 training lacks an epoch checkpoint: {training}")
    return _nonempty_file(candidates[-1], "adapter checkpoint")


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FailureBoundaryError(f"{label} row {line_number} must be an object")
                if not str(value.get("id", "")).strip():
                    raise FailureBoundaryError(f"{label} row {line_number} lacks an id")
                if not str(value.get("label", "")).strip():
                    raise FailureBoundaryError(f"{label} row {line_number} lacks a label")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FailureBoundaryError(f"cannot parse {label}: {path}") from exc
    if not rows:
        raise FailureBoundaryError(f"{label} is empty: {path}")
    if any(not str(row.get("prompt") or row.get("user_text") or row.get("messages") or "").strip() for row in rows):
        raise FailureBoundaryError(f"{label} contains a row without a prompt")
    return rows


def _split_audit(training: Path, target: Path, common: Path) -> dict[str, int]:
    groups = {
        "train": {prompt_sha256(row) for row in _read_jsonl(training, "WildJailbreak training JSONL")},
        "target": {prompt_sha256(row) for row in _read_jsonl(target, "WildJailbreak test JSONL")},
        "common": {prompt_sha256(row) for row in _read_jsonl(common, "common-test JSONL")},
    }
    audit = {
        "train_target_overlap": len(groups["train"] & groups["target"]),
        "train_common_overlap": len(groups["train"] & groups["common"]),
        "common_target_overlap": len(groups["common"] & groups["target"]),
    }
    if any(audit.values()):
        raise FailureBoundaryError(f"P0-06 split overlap detected: {audit}")
    return audit


def prepare_failure_evaluations(
    spec: Mapping,
    *,
    phase1_root: str | Path,
    phasef_config: str | Path,
    project_root: str | Path,
    python_executable: str,
    device: str,
    device_id: int,
    target_test_jsonl: str | Path,
    training_jsonl: str | Path | None = None,
    curation_summary: str | Path | None = None,
) -> FailureEvaluationPlan:
    """Stage the independent common-test evaluation required by P0-06.

    The in-domain WJB generation already exists after the normal one-click run.
    This plan evaluates the exact same final adapter on ``common_safety.jsonl``
    and applies the same WildGuard checkpoint to both generation files.
    """

    if str(spec.get("experiment_id", "")) != "P0-06":
        raise FailureBoundaryError("failure evaluation is only valid for P0-06")
    axes = dict(spec.get("axes") or {})
    pair = str(axes.get("pair", "")).strip()
    if not pair:
        raise FailureBoundaryError("P0-06 requires a model-pair axis")
    inputs = spec.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise FailureBoundaryError("P0-06 inputs must be a mapping")
    common_root = Path(str(inputs.get("common_test", ""))).expanduser().resolve()
    common_jsonl = _nonempty_file(common_root / "common_safety.jsonl", "common_safety.jsonl")
    wildguard = Path(str(inputs.get("wildguard_model", ""))).expanduser().resolve()
    model_report = inspect_model_directory(wildguard)
    if model_report.status != "READY":
        codes = ", ".join(issue.code for issue in model_report.issues)
        raise FailureBoundaryError(f"incomplete WildGuard model: {wildguard} ({codes})")

    project = Path(project_root).resolve()
    phase1 = Path(phase1_root).resolve()
    training = phase1 / "training"
    target_result = _latest_target_result(phase1)
    target_test = _nonempty_file(target_test_jsonl, "WildJailbreak test JSONL")
    train_manifest = _nonempty_file(training / "manifest.json", "training manifest")
    checkpoint = _latest_checkpoint(training)
    phasef_path = _nonempty_file(phasef_config, "staged PhaseF config")
    phasef = yaml.safe_load(phasef_path.read_text(encoding="utf-8")) or {}
    if not isinstance(phasef, dict) or not isinstance(phasef.get("model"), dict):
        raise FailureBoundaryError(f"invalid staged PhaseF model config: {phasef_path}")

    template_name = apply_tokens(
        f"baseline_eval_qwen35_08b_wildjailbreak_{device}.yaml",
        pair,
    )
    template = _nonempty_file(project / "configs" / template_name, "pair evaluation config")
    config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise FailureBoundaryError(f"invalid pair evaluation config: {template}")
    config["model"] = dict(phasef["model"])
    datasets = config.setdefault("datasets", {})
    for task in datasets.values():
        if isinstance(task, dict):
            task["enabled"] = False
    pan = datasets.setdefault("pan", {})
    pan.update(
        path=str(common_jsonl),
        enabled=True,
        placeholder_ok=False,
        shuffle=False,
        max_samples=0,
    )
    boundary = training / "failure_boundary"
    common_output = boundary / "common_eval"
    config.setdefault("output", {})["output_root"] = str(common_output)
    common_config = boundary / "common_eval.yaml"
    common_config.parent.mkdir(parents=True, exist_ok=True)
    common_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    common_result = common_output / "pan_results.json"
    target_judge = boundary / "target_judge_results.json"
    common_judge = boundary / "common_judge_results.json"
    runtime_device = str(config["model"].get("runtime_device") or "")
    runtime_backend = str(config["model"].get("runtime_backend") or device)
    if not runtime_device:
        runtime_device = "cpu" if device == "cpu" else f"{device}:{device_id}"

    evaluation = (
        python_executable,
        str(project / "scripts" / "12_eval_baseline_suite.py"),
        "--config",
        str(common_config),
        "--output-dir",
        str(common_output),
        "--adapter-manifest",
        str(train_manifest),
        "--adapter-checkpoint",
        str(checkpoint),
    )

    def judge(result: Path, test: Path, output: Path) -> tuple[str, ...]:
        return (
            python_executable,
            str(project / "scripts" / "22_judge_generations.py"),
            "--pan-results",
            str(result),
            "--test-jsonl",
            str(test),
            "--judge-model",
            str(wildguard),
            "--runtime-backend",
            runtime_backend,
            "--runtime-device",
            runtime_device,
            "--out",
            str(output),
        )

    if training_jsonl is None or curation_summary is None:
        raise FailureBoundaryError(
            "P0-06 requires the exact WildJailbreak training JSONL and curation summary"
        )
    training_path = _nonempty_file(training_jsonl, "WildJailbreak training JSONL")
    curation_path = _nonempty_file(curation_summary, "curation summary")
    split_audit = _split_audit(training_path, target_test, common_jsonl)
    manifest_path = boundary / "evaluation_manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": "P0-06",
        **{key: axes.get(key) for key in ("pair", "config", "curation", "method")},
        "training_jsonl": str(training_path),
        "curation_summary": str(curation_path),
        "target_test_jsonl": str(target_test),
        "common_test_jsonl": str(common_jsonl),
        "target_result": str(target_result),
        "target_judge": str(target_judge),
        "common_result": str(common_result),
        "common_judge": str(common_judge),
        "adapter_manifest": str(train_manifest),
        "adapter_checkpoint": str(checkpoint),
        "adapter_checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": {
            "training": sha256_file(training_path),
            "target_test": sha256_file(target_test),
            "common_test": sha256_file(common_jsonl),
        },
        "split_audit": split_audit,
        "common_config": str(common_config),
    }
    _write_json(manifest_path, manifest)
    return FailureEvaluationPlan(
        (evaluation, judge(target_result, target_test, target_judge), judge(common_result, common_jsonl, common_judge)),
        common_config.resolve(),
        target_result,
        common_result.resolve(),
        manifest_path.resolve(),
    )
