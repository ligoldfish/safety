from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .artifacts import sha256_file
from .fairness import (
    FAIRNESS_METHODS,
    GLOBAL_HYPERPARAMETERS,
    HISTORICAL_HYPERPARAMETERS,
    VALIDATION_SELECTED_DATASETS,
    validate_search_ledger_rows,
    verify_search_ledger_evidence,
)


class FairnessSearchError(RuntimeError):
    """Raised when a validation-only fairness search is incomplete or inconsistent."""


_CANDIDATES = ("global", "historical_override")
_SELECTION_METRIC = "wildguard_refusal_minus_over_refusal"


@dataclass(frozen=True)
class FairnessSearchTrial:
    trial_id: str
    dataset: str
    method: str
    candidate: str
    hyperparameters: dict[str, int | float]
    output_dir: Path

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _method_hyperparameters(
    values: Mapping[str, int | float], method: str
) -> dict[str, int | float]:
    result = dict(values)
    if method == "sft1":
        result["layer_loss_weight"] = 0.0
    return result


def build_fairness_search_trials(output_root: str | Path) -> tuple[FairnessSearchTrial, ...]:
    root = Path(output_root)
    trials = []
    for dataset in sorted(VALIDATION_SELECTED_DATASETS):
        for method in FAIRNESS_METHODS:
            for candidate in _CANDIDATES:
                source = (
                    GLOBAL_HYPERPARAMETERS
                    if candidate == "global"
                    else HISTORICAL_HYPERPARAMETERS[dataset]
                )
                trial_id = f"{dataset}-{method}-{candidate}"
                trials.append(
                    FairnessSearchTrial(
                        trial_id=trial_id,
                        dataset=dataset,
                        method=method,
                        candidate=candidate,
                        hyperparameters=_method_hyperparameters(source, method),
                        output_dir=root / dataset / method / candidate,
                    )
                )
    return tuple(trials)


def compile_fairness_search_command(
    trial: FairnessSearchTrial,
    *,
    project_root: str | Path,
    python_executable: str,
    device: str,
    device_id: int,
    pair: str = "qwen35_9b_to_08b",
) -> tuple[str, ...]:
    values = trial.hyperparameters
    phasef = {
        "target.mode": "random_same_norm" if trial.method == "random" else "semantic",
        "optim.layer_loss_weight": values["layer_loss_weight"],
        "optim.epochs": values["epochs"],
    }
    extras = {
        "analyze": ["--top-k", str(values["top_k"])],
        "subspace": [
            "--energy-threshold",
            str(values["energy_threshold"]),
            "--rank-cap",
            str(values["rank_cap"]),
        ],
    }
    cell_spec = {
        "experiment_id": "P0-07-search",
        "axes": {
            "dataset": trial.dataset,
            "method": trial.method,
            "candidate": trial.candidate,
        },
        "hyperparameters": values,
        "selection_split": "validation",
    }
    return (
        str(python_executable),
        str(Path(project_root) / "scripts" / "30_run_ablation_cell.py"),
        f"--cell-id={trial.trial_id}",
        "--experiment-id=P0-07-search",
        "--cell-spec=" + json.dumps(cell_spec, sort_keys=True, separators=(",", ":")),
        f"--output-dir={trial.output_dir}",
        "--required-artifacts=[]",
        f"--pair={pair}",
        f"--dataset={trial.dataset}",
        f"--method={trial.method}",
        f"--device={device}",
        f"--device-id={int(device_id)}",
        "--phase1-updates={}",
        "--phasef-updates=" + json.dumps(phasef, sort_keys=True, separators=(",", ":")),
        "--phase1-stage-extras=" + json.dumps(extras, sort_keys=True, separators=(",", ":")),
        "--disable-dataset-overrides",
        "--skip-test-eval",
    )


def compile_fairness_judge_command(
    trial: FairnessSearchTrial,
    *,
    project_root: str | Path,
    python_executable: str,
    judge_model: str | Path,
    device: str,
    device_id: int,
) -> tuple[str, ...]:
    training_root = trial.output_dir / "pipeline" / "phase1" / "training"
    manifest = _read_object(training_root / "manifest.json", "fairness training manifest")
    val_split = str(manifest.get("val_split", "")).strip()
    if not val_split:
        raise FairnessSearchError(f"training manifest lacks val_split: {trial.trial_id}")
    epoch = int(trial.hyperparameters["epochs"])
    generations = training_root / "logs" / "val_generations" / f"epoch_{epoch:03d}.json"
    output = generations.with_name(f"epoch_{epoch:03d}.wildguard.json")
    return (
        str(python_executable),
        str(Path(project_root) / "scripts" / "22_judge_generations.py"),
        "--pan-results", str(generations),
        "--test-jsonl", val_split,
        "--judge-model", str(judge_model),
        "--runtime-backend", device,
        "--runtime-device", f"{device}:{int(device_id)}",
        "--out", str(output),
    )


def _read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FairnessSearchError(f"missing or invalid {label}: {path}") from exc
    if not isinstance(value, dict) or not value:
        raise FairnessSearchError(f"empty or invalid {label}: {path}")
    return value


def _same_number(actual: object, expected: object) -> bool:
    return (
        type(actual) in {int, float}
        and math.isfinite(float(actual))
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    )


def _evidence(trial: FairnessSearchTrial) -> dict:
    phase1 = trial.output_dir / "pipeline" / "phase1"
    values = trial.hyperparameters
    layer_path = phase1 / "layer_analysis" / "teacher_key_layers.json"
    subspace_path = phase1 / "safe_subspaces" / "manifest.json"
    training_path = phase1 / "training" / "manifest.json"
    epoch = int(values["epochs"])
    generations_path = phase1 / "training" / "logs" / "val_generations" / f"epoch_{epoch:03d}.json"
    validation_path = generations_path.with_name(f"epoch_{epoch:03d}.wildguard.json")
    layer = _read_object(layer_path, "fairness layer manifest")
    subspace = _read_object(subspace_path, "fairness subspace manifest")
    training = _read_object(training_path, "fairness training manifest")
    validation = _read_object(validation_path, "fairness validation metrics")
    val_split = str(training.get("val_split", "")).strip()
    train_split = str(training.get("train_split", "")).strip()
    if not val_split or not Path(val_split).is_file() or not train_split or not Path(train_split).is_file():
        raise FairnessSearchError(f"training manifest has no readable train/val split: {trial.trial_id}")
    expected_mode = "random_same_norm" if trial.method == "random" else "semantic"
    actual = {
        "top_k": layer.get("top_k"),
        "energy_threshold": subspace.get("energy_threshold"),
        "rank_cap": subspace.get("rank_cap"),
        "layer_loss_weight": training.get("layer_loss_weight"),
        "epochs": training.get("epochs"),
        "target_mode": training.get("target_mode"),
    }
    if (
        actual["top_k"] != values["top_k"]
        or actual["rank_cap"] != values["rank_cap"]
        or actual["epochs"] != values["epochs"]
        or actual["target_mode"] != expected_mode
        or not _same_number(actual["energy_threshold"], values["energy_threshold"])
        or not _same_number(actual["layer_loss_weight"], values["layer_loss_weight"])
    ):
        raise FairnessSearchError(
            f"search backend configuration differs for {trial.trial_id}: "
            f"expected={values}, actual={actual}"
        )
    if training.get("epochs_completed") != epoch:
        raise FairnessSearchError(f"search backend did not complete epoch {epoch}: {trial.trial_id}")
    if validation.get("judge") != "wildguard" or validation.get("num_unmatched_ids") != 0:
        raise FairnessSearchError(f"incomplete WildGuard validation for {trial.trial_id}")
    parse_rate = validation.get("judge_parse_rate")
    if not _same_number(parse_rate, 1.0):
        raise FairnessSearchError(f"WildGuard must parse every validation item for {trial.trial_id}")
    hr = validation.get("llm_judge_refusal_rate")
    over_refusal = validation.get("llm_judge_over_refusal")
    if not _same_number(hr, hr) or not 0.0 <= float(hr) <= 1.0:
        raise FairnessSearchError(f"invalid validation harmful refusal for {trial.trial_id}")
    if not _same_number(over_refusal, over_refusal) or not 0.0 <= float(over_refusal) <= 1.0:
        raise FairnessSearchError(f"invalid validation over-refusal for {trial.trial_id}")
    sources = {
        "judge": validation_path,
        "generations": generations_path,
        "train_split": Path(train_split),
        "validation_split": Path(val_split),
        "training": training_path,
        "layer": layer_path,
        "subspace": subspace_path,
    }
    return {
        "validation_metric": float(hr) - float(over_refusal),
        "validation_harmful_refusal": float(hr),
        "validation_over_refusal": float(over_refusal),
        "validation_epoch": epoch,
        "selection_metric": _SELECTION_METRIC,
        "evidence": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
    }


def _validate_trial_set(trials: Sequence[FairnessSearchTrial]) -> None:
    keys = [(trial.dataset, trial.method, trial.candidate) for trial in trials]
    expected = {
        (dataset, method, candidate)
        for dataset in VALIDATION_SELECTED_DATASETS
        for method in FAIRNESS_METHODS
        for candidate in _CANDIDATES
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise FairnessSearchError("fairness search requires the exact 12 equal-budget trials")
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise FairnessSearchError("fairness search trial IDs must be unique")


def _global_rows() -> list[dict]:
    datasets = ("pan", "safety_tuned_llamas", "coconot", "c5", "wildjailbreak", "wildguardmix")
    return [
        {
            "trial_id": f"{dataset}-global-{method}",
            "dataset": dataset,
            "config": "global",
            "method": method,
            "selection_split": "validation",
            "selected": False,
            "validation_metric": 0.0,
            "hyperparameters": _method_hyperparameters(GLOBAL_HYPERPARAMETERS, method),
        }
        for dataset in datasets
        for method in FAIRNESS_METHODS
    ]


def collect_fairness_search_ledger(
    trials: Iterable[FairnessSearchTrial], output: str | Path
) -> tuple[dict, ...]:
    ordered = tuple(trials)
    _validate_trial_set(ordered)
    evidence = {trial.trial_id: _evidence(trial) for trial in ordered}
    winners = {}
    for dataset in VALIDATION_SELECTED_DATASETS:
        for method in FAIRNESS_METHODS:
            candidates = [
                trial
                for trial in ordered
                if trial.dataset == dataset and trial.method == method
            ]
            # Candidate order is global then historical, so exact ties retain
            # the preregistered global default rather than rewarding tuning.
            winners[(dataset, method)] = max(
                candidates,
                key=lambda trial: evidence[trial.trial_id]["validation_metric"],
            ).trial_id
    rows = _global_rows()
    for trial in ordered:
        rows.append(
            {
                "trial_id": trial.trial_id,
                "dataset": trial.dataset,
                "config": "validation_selected",
                "method": trial.method,
                "candidate": trial.candidate,
                "selection_split": "validation",
                "selected": winners[(trial.dataset, trial.method)] == trial.trial_id,
                "hyperparameters": dict(trial.hyperparameters),
                **evidence[trial.trial_id],
            }
        )
    normalized = validate_search_ledger_rows(rows)
    verify_search_ledger_evidence(normalized)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return normalized
