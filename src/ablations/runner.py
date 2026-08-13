from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .artifacts import canonical_hash, sha256_file
from .efficiency import StageProfiler
from .ledger import ExperimentLedger, LedgerError, RunState
from .schema import ExecutionKind, ExperimentCatalog, ExperimentCell
from .preflight import requirements_from_manifest, run_preflight


class RunnerError(RuntimeError):
    """Raised when a cell cannot be executed or validated safely."""


@dataclass(frozen=True)
class RunnerContext:
    project_root: Path
    state_root: Path
    python_executable: str = sys.executable
    device: str = "npu"
    device_id: int = 0
    num_devices: int = 1
    asset_manifest: Path | None = None


@dataclass(frozen=True)
class CommandSpec:
    stage: str
    argv: tuple[str, ...]
    completion_artifacts: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "argv": list(self.argv),
            "completion_artifacts": list(self.completion_artifacts),
        }


_TRAIN_HANDLERS = {
    "matched_controls",
    "wjb_failure_analysis",
    "global_default_budget",
    "target_control",
    "subspace_control",
    "bridge_control",
    "layer_selection_control",
    "layer_score_control",
    "pairing_control",
    "representation_position_control",
    "semantic_top_m_sweep",
    "semantic_selection_control",
    "layer_loss_weight_sweep",
    "supervision_policy_control",
    "layer_loss_kind_control",
    "lora_capacity_control",
    "subspace_hyperparameter_sweep",
    "data_efficiency_sweep",
    "curation_control",
    "teacher_quality_control",
    "cross_tokenizer_bridge",
}
_EVAL_HANDLERS = {
    "cross_corpus_matrix",
    "general_capability_suite",
    "causal_intervention",
    "decoding_robustness",
}
_ANALYSIS_HANDLERS = {
    "provenance_matrix",
    "seed_and_paired_bootstrap",
    "validation_iso_hr",
    "pan_subgroup_analysis",
    "representation_behavior_analysis",
    "subspace_bootstrap",
    "efficiency_profile",
    "ethics_data_audit",
}
_MANUAL_HANDLERS = {"judge_agreement_audit"}


def executable_handlers() -> set[str]:
    return _TRAIN_HANDLERS | _EVAL_HANDLERS | _ANALYSIS_HANDLERS | _MANUAL_HANDLERS


def _axis(cell: ExperimentCell, name: str, default=None):
    return cell.axes.get(name, cell.overrides.get(name, default))


def _merged_axes(cell: ExperimentCell) -> dict:
    merged = dict(cell.overrides)
    merged.update(cell.axes)
    return merged


def _phase_extras(cell: ExperimentCell) -> dict[str, list[str]]:
    axes = _merged_axes(cell)
    extras: dict[str, list[str]] = {}

    def add(stage: str, *tokens: object) -> None:
        extras.setdefault(stage, []).extend(str(token) for token in tokens)

    experiment = cell.experiment_id
    if experiment == "P1-08":
        add("extract", "--representation-mode", axes["mode"])
    if experiment in {"P1-05", "P1-06"}:
        add("analyze", "--selection-mode", axes["mode"])
        if "draw" in axes:
            add("analyze", "--draw-seed", axes["draw"])
    if experiment in {"P1-02", "P1-03"}:
        mode = axes.get("subspace_mode", axes.get("subspace.mode", "random_orthogonal"))
        add("subspace", "--subspace-mode", mode)
        if "draw" in axes:
            add("subspace", "--draw-seed", axes["draw"])
    if experiment == "P1-15":
        add("subspace", "--energy-threshold", axes["energy_threshold"])
        add("subspace", "--rank-cap", axes["rank_cap"])
    if experiment in {"P1-04", "P2-04"}:
        bridge_mode = axes.get("bridge_mode", axes.get("bridge.mode"))
        add("bridge", "--bridge-mode", bridge_mode)
        add("recompose", "--bridge-mode", bridge_mode)
    if experiment == "P1-07":
        add("pairing", "--pairing-mode", axes["mode"])
    if experiment == "P1-09":
        add("decompose", "--top-k", axes["top_m"])
    if experiment == "P1-10":
        add("decompose", "--selection-mode", axes["selection_mode"])
        if not bool(axes["token_filter"]):
            add("decompose", "--disable-token-filter")
    if experiment == "P1-16" and axes["phase"] == "phase1" and axes["samples_per_label"] != "all":
        add("extract_alignment", "--max-samples-per-label", axes["samples_per_label"])
    if experiment == "P1-17" and not bool(axes["balance"]):
        add("subspace", "--no-balance-labels")
    if experiment == "P2-01":
        add("analyze", "--draw-seed", axes["draw"])
        add("subspace", "--subspace-mode", "learned", "--draw-seed", axes["draw"])
    return extras


def phasef_updates_for_cell(cell: ExperimentCell) -> dict[str, object]:
    axes = _merged_axes(cell)
    updates: dict[str, object] = {}
    experiment = cell.experiment_id
    if experiment in {"P0-02", "P0-06"}:
        method = str(axes["method"])
        if "seed" in axes:
            updates["seed"] = int(axes["seed"])
        updates["target.mode"] = "random_same_norm" if method == "random" else "semantic"
        updates["optim.layer_loss_weight"] = 0.0 if method == "sft1" else 0.25
    elif experiment == "P1-01":
        updates["target.mode"] = axes["target_mode"]
    elif experiment == "P1-11":
        updates["optim.layer_loss_weight"] = float(axes["layer_loss_weight"])
    elif experiment == "P1-12":
        mode = str(axes["mode"])
        if mode.startswith("label_weighted_"):
            updates["target.layer_loss_policy"] = "label_weighted"
            harmless = {"label_weighted_1_05": 0.5, "label_weighted_1_1": 1.0}[mode]
            updates["target.harmful_layer_weight"] = 1.0
            updates["target.harmless_layer_weight"] = harmless
        else:
            updates["target.layer_loss_policy"] = mode
    elif experiment == "P1-13":
        updates["target.loss_kind"] = axes["loss_kind"]
    elif experiment == "P1-14":
        updates["lora.rank"] = int(axes["rank"])
        updates["lora.placement"] = axes["placement"]
    elif experiment == "P1-16" and axes["phase"] == "phasef":
        updates["inputs.max_samples_per_label"] = axes["samples_per_label"]
    elif experiment == "P1-08":
        updates["target.representation_mode"] = axes["mode"]
    return updates


def _phase1_updates(cell: ExperimentCell) -> dict[str, object]:
    axes = _merged_axes(cell)
    updates: dict[str, object] = {}
    if cell.experiment_id == "P0-06":
        updates["dataset.curation_mode"] = axes["curation"]
    elif cell.experiment_id == "P1-17":
        updates["dataset.curation_mode"] = axes["curation"]
    return updates


def _train_command(
    definition,
    cell: ExperimentCell,
    context: RunnerContext,
) -> CommandSpec:
    axes = _merged_axes(cell)
    assets = _load_asset_manifest(context.asset_manifest)
    inputs: dict[str, str] = {}
    for requirement in definition.requires:
        if requirement not in assets:
            continue
        raw = assets[requirement]
        path_text = str(raw.get("path", "")) if isinstance(raw, Mapping) else str(raw)
        path = Path(path_text).expanduser()
        if not path.is_absolute() and context.asset_manifest is not None:
            path = context.asset_manifest.parent / path
        inputs[requirement] = str(path.resolve())
    pair = str(axes.get("pair", "qwen35_9b_to_08b"))
    teacher_variant = ""
    if cell.experiment_id == "P2-03":
        teacher = str(axes["teacher"])
        pair = "qwen3_8b_to_06b" if teacher == "qwen3_8b" else "qwen3_4b_to_06b"
        if teacher in {"same_size_base", "safety_tuned"}:
            teacher_variant = teacher
    elif cell.experiment_id == "P2-04":
        pair = "qwen3_8b_to_llama32_1b"
    default_dataset = "wildjailbreak" if cell.experiment_id == "P0-06" else "pan"
    dataset = str(axes.get("dataset", default_dataset))
    method = str(axes.get("method", "ours"))
    if cell.experiment_id == "P0-02" and method not in {"ours", "random", "sft1"}:
        raise RunnerError(f"unsupported matched-control method: {method}")
    argv = [
        context.python_executable,
        str(context.project_root / "scripts" / "30_run_ablation_cell.py"),
        "--cell-id",
        cell.cell_id,
        "--experiment-id",
        cell.experiment_id,
        "--cell-spec=" + json.dumps(
            {"experiment_id": cell.experiment_id, "axes": axes, "inputs": inputs},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "--pair",
        pair,
        "--dataset",
        dataset,
        "--method",
        method,
        "--device",
        context.device,
        "--device-id",
        str(context.device_id),
        "--output-dir",
        str(context.state_root / cell.cell_id / "artifacts"),
        "--phase1-updates=" + json.dumps(_phase1_updates(cell), ensure_ascii=False, sort_keys=True),
        "--phasef-updates=" + json.dumps(phasef_updates_for_cell(cell), ensure_ascii=False, sort_keys=True),
        "--phase1-stage-extras=" + json.dumps(_phase_extras(cell), ensure_ascii=False, sort_keys=True),
        "--required-artifacts=" + json.dumps(list(definition.completion_artifacts)),
    ]
    if teacher_variant:
        argv.extend(["--teacher-variant", teacher_variant])
    if cell.experiment_id in {"P0-06", "P0-07"} and axes.get("config") == "global":
        argv.append("--disable-dataset-overrides")
    return CommandSpec("train", tuple(argv), tuple(definition.completion_artifacts))


def _load_asset_manifest(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid asset manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RunnerError("asset manifest must be a JSON object")
    return payload


def _worker_command(definition, cell: ExperimentCell, context: RunnerContext) -> CommandSpec:
    assets = _load_asset_manifest(context.asset_manifest)
    inputs = {}
    for requirement in definition.requires:
        if requirement in assets:
            raw = assets[requirement]
            inputs[requirement] = str(raw.get("path", "")) if isinstance(raw, Mapping) else str(raw)
    worker_flag = (
        "--evaluation-handler"
        if definition.handler in {"general_capability_suite", "causal_intervention", "decoding_robustness"}
        else "--analysis-handler"
    )
    argv = (
        context.python_executable,
        str(context.project_root / "scripts" / "30_run_ablation_cell.py"),
        worker_flag,
        definition.handler,
        "--cell-id",
        cell.cell_id,
        "--cell-spec=" + json.dumps(
            {
                "experiment_id": cell.experiment_id,
                "axes": dict(cell.axes),
                "overrides": dict(cell.overrides),
                "inputs": inputs,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "--output-dir",
        str(context.state_root / cell.cell_id / "artifacts"),
        "--device",
        context.device,
        "--device-id",
        str(context.device_id),
        "--required-artifacts=" + json.dumps(list(definition.completion_artifacts)),
    )
    return CommandSpec(definition.execution_kind.value, argv, tuple(definition.completion_artifacts))


def compile_cell_commands(
    catalog: ExperimentCatalog,
    cell: ExperimentCell,
    context: RunnerContext,
) -> tuple[CommandSpec, ...]:
    if cell.experiment_id not in catalog.experiments:
        raise RunnerError(f"unknown experiment: {cell.experiment_id}")
    definition = catalog.experiments[cell.experiment_id]
    if definition.handler not in executable_handlers():
        raise RunnerError(f"handler is not executable: {definition.handler}")
    if definition.execution_kind is ExecutionKind.TRAIN:
        return (_train_command(definition, cell, context),)
    return (_worker_command(definition, cell, context),)


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_completion(cell_dir: Path, artifacts: Sequence[str]) -> str:
    artifact_root = cell_dir / "artifacts"
    missing = [name for name in artifacts if not (artifact_root / name).is_file()]
    if missing:
        raise RunnerError(f"missing completion artifact(s): {missing}")
    for name in artifacts:
        path = artifact_root / name
        if path.stat().st_size <= 0:
            raise RunnerError(f"completion artifact is empty: {name}")
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RunnerError(f"invalid JSON completion artifact: {name}") from exc
            if not isinstance(payload, (dict, list)):
                raise RunnerError(f"JSON completion artifact must be an object or list: {name}")
        elif path.suffix == ".jsonl":
            rows = 0
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise RunnerError(
                                f"JSONL row must be an object: {name}:{line_number}"
                            )
                        rows += 1
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RunnerError(f"invalid JSONL completion artifact: {name}") from exc
            if rows == 0:
                raise RunnerError(f"completion artifact is empty: {name}")
    hashes = {name: sha256_file(artifact_root / name) for name in artifacts}
    _atomic_json(
        cell_dir / "completion.json",
        {"schema_version": 1, "artifacts": hashes},
    )
    return canonical_hash(hashes)


Executor = Callable[..., subprocess.CompletedProcess]


class AblationRunner:
    def __init__(
        self,
        catalog: ExperimentCatalog,
        context: RunnerContext,
        *,
        executor: Executor | None = None,
        environment: Mapping[str, str] | None = None,
        enforce_preflight: bool | None = None,
    ) -> None:
        self.catalog = catalog
        self.context = context
        self.executor = executor or self._execute
        self.environment = dict(os.environ if environment is None else environment)
        self.enforce_preflight = (executor is None) if enforce_preflight is None else bool(enforce_preflight)

    @staticmethod
    def _execute(command: list[str], *, cwd: Path, env: Mapping[str, str]):
        return subprocess.run(command, cwd=str(cwd), env=dict(env), check=False)

    @staticmethod
    def select_cell(cells: Iterable[ExperimentCell], cell_id: str) -> ExperimentCell:
        matches = [cell for cell in cells if cell.cell_id == str(cell_id)]
        if len(matches) != 1:
            raise RunnerError(f"unknown cell: {cell_id}")
        return matches[0]

    def _ledger(self, cell: ExperimentCell, *, dry_run: bool = False) -> ExperimentLedger:
        fingerprint = canonical_hash(
            {
                "cell": asdict(cell),
                "definition": asdict(self.catalog.experiments[cell.experiment_id]),
                "asset_manifest": (
                    None
                    if self.context.asset_manifest is None
                    else canonical_hash(_load_asset_manifest(self.context.asset_manifest))
                ),
            }
        )
        ledger = ExperimentLedger(
            self.context.state_root,
            cell.cell_id,
            config_hash=fingerprint,
        )
        ledger.initialize(dry_run=dry_run)
        return ledger

    def status(self, cell: ExperimentCell) -> dict:
        return self._ledger(cell).read()

    def run_cell(self, cell: ExperimentCell, *, dry_run: bool = False) -> dict:
        commands = compile_cell_commands(self.catalog, cell, self.context)
        ledger = self._ledger(cell, dry_run=dry_run)
        with ledger.acquire_lock():
            current = ledger.read()
            state = RunState(current["state"])
            if state is RunState.COMPLETED:
                return current
            if current.get("dry_run") and not dry_run:
                raise RunnerError("dry-run state cannot be reused for a real execution")
            if state in {RunState.PLANNED, RunState.BLOCKED, RunState.FAILED}:
                ledger.transition(RunState.READY)
            if dry_run:
                _atomic_json(
                    ledger.cell_dir / "commands.json",
                    {"schema_version": 1, "dry_run": True, "commands": [item.to_dict() for item in commands]},
                )
                return ledger.read()
            if self.enforce_preflight:
                if self.context.asset_manifest is None:
                    return ledger.transition(
                        RunState.BLOCKED,
                        reason="asset manifest is required for real execution",
                    )
                manifest = _load_asset_manifest(self.context.asset_manifest)
                requirements, missing = requirements_from_manifest(
                    self.catalog.experiments[cell.experiment_id].requires,
                    manifest,
                    cell_id=cell.cell_id,
                    base_dir=self.context.asset_manifest.parent,
                )
                if missing:
                    return ledger.transition(
                        RunState.BLOCKED,
                        reason=f"asset manifest is missing required keys: {list(missing)}",
                    )
                report = run_preflight(requirements, environment=self.environment)
                if report.status != "READY":
                    return ledger.transition(
                        RunState.BLOCKED,
                        reason=json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True),
                    )
            ledger.transition(RunState.RUNNING)
            efficiency = []
            try:
                for command in commands:
                    profiler = StageProfiler(
                        command.stage,
                        output_root=ledger.cell_dir,
                        cell_id=cell.cell_id,
                        device_count=self.context.num_devices,
                    )
                    profiler.start()
                    result = self.executor(
                        list(command.argv),
                        cwd=self.context.project_root,
                        env=self.environment,
                    )
                    efficiency.append(profiler.finish(exit_code=int(result.returncode)))
                    if int(result.returncode) != 0:
                        raise RunnerError(
                            f"cell {cell.cell_id} stage {command.stage} failed with exit code {result.returncode}"
                        )
                artifact_hash = validate_completion(
                    ledger.cell_dir,
                    self.catalog.experiments[cell.experiment_id].completion_artifacts,
                )
                _atomic_json(
                    ledger.cell_dir / "efficiency.json",
                    {"schema_version": 1, "stages": [record.to_dict() for record in efficiency]},
                )
                return ledger.transition(RunState.COMPLETED, artifact_hash=artifact_hash)
            except (KeyboardInterrupt, BaseException) as exc:
                # SystemExit and KeyboardInterrupt must also leave a recoverable
                # FAILED ledger. Re-raise the original signal after persistence.
                ledger.transition(RunState.FAILED, reason=f"{type(exc).__name__}: {exc}")
                raise
