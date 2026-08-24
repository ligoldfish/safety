from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.catalog import load_catalog
from src.ablations.planner import validate_plan
from src.ablations.schema import ExecutionKind, ExperimentCatalog, ExperimentPlan


EXECUTION_WAVES = (
    "core-train",
    "wjb",
    "fairness",
    "evaluate",
    "analyze",
    "manual",
)


@dataclass(frozen=True)
class JobConfig:
    wave: str
    model_root: Path
    data_root: Path
    output_root: Path
    asset_manifest: Path
    shard_index: int
    shard_count: int
    max_cells: int
    device: str
    device_id: int
    num_devices: int
    dry_run: bool = False


def select_wave_cells(
    catalog: ExperimentCatalog,
    plan: ExperimentPlan,
    wave: str,
) -> ExperimentPlan:
    if wave not in EXECUTION_WAVES:
        raise ValueError(f"unknown execution wave: {wave}")

    def selected(cell) -> bool:
        kind = catalog.experiments[cell.experiment_id].execution_kind
        if wave == "core-train":
            return kind is ExecutionKind.TRAIN and cell.experiment_id not in {"P0-06", "P0-07"}
        if wave == "wjb":
            return cell.experiment_id == "P0-06"
        if wave == "fairness":
            return cell.experiment_id == "P0-07"
        return kind.value == wave

    return validate_plan(
        ExperimentPlan(plan.schema_version, tuple(cell for cell in plan.cells if selected(cell)))
    )


def validate_persistent_root(path: Path, label: str) -> Path:
    source = Path(path).expanduser()
    resolved = source.resolve()
    normalized = source.as_posix().lower().rstrip("/") + "/"
    resolved_normalized = resolved.as_posix().lower().rstrip("/") + "/"
    forbidden = ("/tmp/", "/cache/", "/home/work/user-job-dir/app/")
    if any(
        normalized.startswith(prefix) or resolved_normalized.startswith(prefix)
        for prefix in forbidden
    ):
        raise ValueError(f"{label} root must be persistent, not {resolved}")
    return resolved


def _plan_args(config: JobConfig, plan_path: Path, *, project_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "scripts" / "30_ablation.py"),
        "plan",
        "--scope",
        "full",
        "--output-root",
        str(config.output_root / "cell-outputs"),
        "--output",
        str(plan_path),
    ]
    if config.wave == "core-train":
        command.extend(
            [
                "--execution-kind",
                "train",
                "--exclude-experiment-id",
                "P0-06",
                "--exclude-experiment-id",
                "P0-07",
            ]
        )
    elif config.wave == "wjb":
        command.extend(["--experiment-id", "P0-06"])
    elif config.wave == "fairness":
        command.extend(["--experiment-id", "P0-07"])
    else:
        command.extend(["--execution-kind", config.wave])
    return command


def build_commands(config: JobConfig, *, project_root: Path = PROJECT_ROOT) -> tuple[tuple[str, ...], ...]:
    project_root = Path(project_root).resolve()
    wave_root = config.output_root / "jobs" / config.wave
    shard_root = wave_root / f"shard-{config.shard_index:05d}-of-{config.shard_count:05d}"
    plan_path = shard_root / "plan.jsonl"
    preflight_path = shard_root / "preflight.json"
    state_root = wave_root / ("dry-run-state" if config.dry_run else "run-state")
    common = [sys.executable, str(project_root / "scripts" / "30_ablation.py")]
    commands: list[Sequence[str]] = [
        _plan_args(config, plan_path, project_root=project_root),
        [
            *common,
            "preflight",
            "--plan",
            str(plan_path),
            "--asset-manifest",
            str(config.asset_manifest),
            "--output",
            str(preflight_path),
            "--shard-index",
            str(config.shard_index),
            "--shard-count",
            str(config.shard_count),
            "--max-cells",
            str(config.max_cells),
            "--device",
            config.device,
        ],
        [
            *common,
            "run",
            "--plan",
            str(plan_path),
            "--shard-index",
            str(config.shard_index),
            "--shard-count",
            str(config.shard_count),
            "--max-cells",
            str(config.max_cells),
            "--state-root",
            str(state_root),
            "--asset-manifest",
            str(config.asset_manifest),
            "--device",
            config.device,
            "--device-id",
            str(config.device_id),
            "--num-devices",
            str(config.num_devices),
            *(["--dry-run"] if config.dry_run else []),
        ],
        [
            *common,
            "status",
            "--plan",
            str(plan_path),
            "--state-root",
            str(state_root),
            "--output",
            str(shard_root / ("dry-run-status.json" if config.dry_run else "status.json")),
        ],
    ]
    return tuple(tuple(str(token) for token in command) for command in commands)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded ICLR ablation wave on ModelMate.")
    parser.add_argument("--wave", choices=EXECUTION_WAVES, required=True)
    parser.add_argument("--model-root", default="/opt/dpcvol/models/safetytransfer")
    parser.add_argument("--data-root", default="/opt/dpcvol/datasets/safetytransfer")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--asset-manifest",
        default=str(PROJECT_ROOT / "configs" / "ablations" / "assets.modelmate.template.json"),
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--max-cells", type=int, default=1)
    parser.add_argument("--device", choices=["npu", "ppu", "cuda", "cpu"], default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _job_config(args: argparse.Namespace) -> JobConfig:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.max_cells <= 0 or args.num_devices <= 0:
        raise ValueError("max-cells and num-devices must be positive")
    return JobConfig(
        wave=args.wave,
        model_root=validate_persistent_root(Path(args.model_root), "model"),
        data_root=validate_persistent_root(Path(args.data_root), "data"),
        output_root=validate_persistent_root(Path(args.output_root), "output"),
        asset_manifest=Path(args.asset_manifest).expanduser().resolve(),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_cells=args.max_cells,
        device=args.device,
        device_id=args.device_id,
        num_devices=args.num_devices,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None, *, environment: Mapping[str, str] | None = None) -> int:
    config = _job_config(_parse_args(argv))
    if not config.asset_manifest.is_file():
        raise FileNotFoundError(f"asset manifest is missing: {config.asset_manifest}")
    config.output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ if environment is None else environment)
    env.update(
        SAFETY_MODEL_ROOT=str(config.model_root),
        SAFETY_DATA_ROOT=str(config.data_root),
        SAFETY_OUTPUT_ROOT=str(config.output_root),
        HF_HOME=str(config.data_root / "_hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SAFETY_REQUIRE_PREPARED_DATA="1",
    )
    for command in build_commands(config):
        print(json.dumps({"command": list(command)}, ensure_ascii=False), flush=True)
        completed = subprocess.run(list(command), cwd=str(PROJECT_ROOT), env=env, check=False)
        if completed.returncode:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
