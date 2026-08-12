from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


class EvaluationError(RuntimeError):
    """Raised when an evaluation cell cannot be staged or collected safely."""


@dataclass(frozen=True)
class EvaluationPlan:
    argv: tuple[str, ...]
    artifact_name: str
    config_path: Path | None = None


_OPENCOMPASS_DATASETS = {
    "mmlu": "mmlu_gen",
    "gsm8k": "gsm8k_gen",
    "ifeval": "IFEval_gen",
    "humaneval": "humaneval_gen",
    "mbpp": "mbpp_gen",
}


def _path(inputs: Mapping, name: str, *, directory: bool | None = None) -> Path:
    value = str(inputs.get(name, "")).strip()
    if not value:
        raise EvaluationError(f"missing required evaluation input: {name}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise EvaluationError(f"missing required evaluation input {name}: {path}")
    if directory is True and not path.is_dir():
        raise EvaluationError(f"evaluation input {name} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise EvaluationError(f"evaluation input {name} must be a file: {path}")
    return path


def _write_yaml(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _checkpoint_files(root: Path) -> tuple[Path | None, Path | None]:
    manifest = root / "manifest.json"
    checkpoints = sorted(root.glob("epoch_*.pt"))
    if not checkpoints:
        checkpoints = sorted((root / "checkpoints").glob("epoch_*.pt"))
    return (manifest if manifest.is_file() else None, checkpoints[-1] if checkpoints else None)


def prepare_evaluation(
    handler: str,
    spec: Mapping,
    *,
    output_dir: str | Path,
    project_root: str | Path,
    python_executable: str,
    device: str,
    device_id: int,
) -> EvaluationPlan:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = Path(project_root).resolve()
    axes = dict(spec.get("axes") or {})
    inputs = spec.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise EvaluationError("evaluation inputs must be a mapping")

    if handler == "decoding_robustness":
        trained = _path(inputs, "trained_checkpoints", directory=True)
        evaluation_data = _path(inputs, "evaluation_data", directory=False)
        template = project / "configs" / "baseline_eval_qwen35_08b_npu.yaml"
        if not template.is_file():
            raise EvaluationError(f"missing canonical evaluation config: {template}")
        payload = yaml.safe_load(template.read_text(encoding="utf-8"))
        for task in payload.get("datasets", {}).values():
            if isinstance(task, dict):
                task["enabled"] = False
        pan = payload.setdefault("datasets", {}).setdefault("pan", {})
        pan.update(
            path=str(evaluation_data),
            enabled=True,
            placeholder_ok=False,
            shuffle=False,
            temperature=float(axes["temperature"]),
            top_p=float(axes["top_p"]),
            max_new_tokens=int(axes["max_new_tokens"]),
            initial_max_new_tokens=0,
        )
        payload["output"]["output_root"] = str(output / "raw")
        if (trained / "config.json").is_file():
            payload["model"]["path"] = str(trained)
            manifest = checkpoint = None
        else:
            manifest, checkpoint = _checkpoint_files(trained)
            if manifest is None or checkpoint is None:
                raise EvaluationError("trained_checkpoints lacks a merged model or manifest+epoch checkpoint")
        config_path = output / "eval_config.yaml"
        _write_yaml(config_path, payload)
        argv = [
            python_executable,
            str(project / "scripts" / "12_eval_baseline_suite.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output / "raw"),
        ]
        if checkpoint is not None and manifest is not None:
            argv.extend(["--adapter-manifest", str(manifest), "--adapter-checkpoint", str(checkpoint)])
        return EvaluationPlan(tuple(argv), "decoding_robustness.json", config_path)

    if handler == "general_capability_suite":
        model = _path(inputs, "trained_checkpoints", directory=True)
        opencompass = _path(inputs, "benchmark_assets", directory=True)
        benchmark = str(axes.get("benchmark", "")).lower()
        if benchmark not in _OPENCOMPASS_DATASETS:
            raise EvaluationError(f"unsupported general-capability benchmark: {benchmark}")
        argv = (
            python_executable,
            str(project / "scripts" / "17_eval_opencompass.py"),
            "--merged-model-dir",
            str(model),
            "--opencompass-dir",
            str(opencompass),
            "--work-dir",
            str(output / "raw"),
            "--datasets",
            _OPENCOMPASS_DATASETS[benchmark],
            "--device",
            str(device),
            "--num-gpus",
            "0" if str(device) == "cpu" else "1",
        )
        return EvaluationPlan(argv, "general_capability.json")

    if handler == "causal_intervention":
        artifact = _path(inputs, "subspace_artifact", directory=False)
        data = _path(inputs, "intervention_data", directory=True)
        model = _path(inputs, "intervention_model", directory=True)
        val = data / "val.jsonl"
        test = data / "test.jsonl"
        if not val.is_file() or not test.is_file():
            raise EvaluationError("intervention_data must contain val.jsonl and test.jsonl")
        strength = float(axes["strength"])
        sign = int(axes["sign"])
        if sign not in {-1, 1} or strength <= 0:
            raise EvaluationError("intervention sign must be +/-1 and strength must be positive")
        payload = {
            "seed": 42,
            "model": {
                "name": model.name,
                "path": str(model),
                "device_map": "auto",
                "torch_dtype": "auto",
                "runtime_backend": str(device),
                "runtime_device": str(device_id),
                "local_files_only": True,
            },
            "inputs": {"artifact_path": str(artifact), "val_split": str(val), "test_split": str(test)},
            "method": {
                "alphas": [sign * strength],
                "selection_metric": "balanced_accuracy",
                "max_length": 4096,
                "batch_size": 1,
                "layer_mode": str(axes["layers"]),
                "random_seed": 42,
            },
            "limits": {},
            "output": {"output_root": str(output / "raw")},
        }
        config_path = output / "phaseC.yaml"
        _write_yaml(config_path, payload)
        return EvaluationPlan(
            (python_executable, str(project / "scripts" / "03_run_safety_intervention.py"), "--config", str(config_path)),
            "intervention_results.json",
            config_path,
        )

    raise EvaluationError(f"unknown evaluation handler: {handler}")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"missing or invalid evaluation result: {path}") from exc
    if not isinstance(value, dict) or not value:
        raise EvaluationError(f"empty evaluation result: {path}")
    return value


def collect_evaluation_result(handler: str, spec: Mapping, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    raw = output / "raw"
    axes = dict(spec.get("axes") or {})
    if handler == "decoding_robustness":
        result = _read_json(raw / "pan_results.json")
        if result.get("status") != "ok" or not result.get("generations"):
            raise EvaluationError("decoding evaluation lacks successful per-sample generations")
        payload = {"schema_version": 1, "decode": axes, "result": result}
        destination = output / "decoding_robustness.json"
    elif handler == "causal_intervention":
        manifest = _read_json(raw / "manifest.json")
        sample_path = raw / "sample_scores.csv"
        if not sample_path.is_file() or not list(csv.DictReader(sample_path.open(encoding="utf-8"))):
            raise EvaluationError("causal evaluation lacks per-sample intervention scores")
        payload = {"schema_version": 1, "intervention": axes, "manifest": manifest, "sample_scores_csv": str(sample_path.resolve())}
        destination = output / "intervention_results.json"
    elif handler == "general_capability_suite":
        candidates = sorted(raw.glob("**/summary_*.csv"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise EvaluationError("OpenCompass did not produce a summary CSV")
        rows = list(csv.DictReader(candidates[-1].open(encoding="utf-8")))
        if not rows:
            raise EvaluationError("OpenCompass summary is empty")
        payload = {"schema_version": 1, "benchmark": axes.get("benchmark"), "summary_csv": str(candidates[-1].resolve()), "rows": rows}
        destination = output / "general_capability.json"
    else:
        raise EvaluationError(f"unknown evaluation handler: {handler}")
    destination.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return destination
