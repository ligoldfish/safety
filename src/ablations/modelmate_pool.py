from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .planner import validate_plan
from .schema import ExperimentCatalog, ExperimentCell, ExperimentPlan


@dataclass(frozen=True)
class RoundSpec:
    name: str
    experiment_ids: tuple[str, ...]
    expected_cells: int
    state_group: str
    default_device: str
    cell_limit: int | None = None
    prerequisites: tuple[str, ...] = ()


def _ids(prefix: str, start: int, stop: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{index:02d}" for index in range(start, stop + 1))


ROUND_SPECS: Mapping[str, RoundSpec] = {
    "p0-smoke": RoundSpec(
        "p0-smoke", ("P0-02",), 8, "p0-core", "npu", cell_limit=8
    ),
    "p0-core": RoundSpec(
        "p0-core", ("P0-02",), 54, "p0-core", "npu", prerequisites=("p0-smoke",)
    ),
    "p0-wjb": RoundSpec(
        "p0-wjb", ("P0-06",), 90, "p0-wjb", "npu", prerequisites=("p0-core",)
    ),
    "p0-fairness": RoundSpec(
        "p0-fairness",
        ("P0-07",),
        24,
        "p0-fairness",
        "npu",
        prerequisites=("p0-wjb",),
    ),
    "p0-evaluate": RoundSpec(
        "p0-evaluate",
        ("P0-08",),
        2,
        "p0-evaluate",
        "npu",
        prerequisites=("p0-fairness",),
    ),
    "p0-analyze": RoundSpec(
        "p0-analyze",
        ("P0-01", "P0-04", "P0-05"),
        154,
        "p0-analyze",
        "cpu",
        prerequisites=("p0-evaluate",),
    ),
    "p0-manual": RoundSpec(
        "p0-manual",
        ("P0-03",),
        3,
        "p0-manual",
        "cpu",
        prerequisites=("p0-analyze",),
    ),
    "p1-mechanism": RoundSpec(
        "p1-mechanism",
        _ids("P1", 1, 15),
        99,
        "p1-mechanism",
        "npu",
        prerequisites=("p0-manual",),
    ),
    "p1-data": RoundSpec(
        "p1-data",
        ("P1-16", "P1-17"),
        16,
        "p1-data",
        "npu",
        prerequisites=("p1-mechanism",),
    ),
    "p1-evaluate": RoundSpec(
        "p1-evaluate",
        ("P1-19",),
        5,
        "p1-evaluate",
        "npu",
        prerequisites=("p1-data",),
    ),
    "p1-analyze": RoundSpec(
        "p1-analyze",
        ("P1-18", "P1-20"),
        5,
        "p1-analyze",
        "cpu",
        prerequisites=("p1-evaluate",),
    ),
    "p2-generalization": RoundSpec(
        "p2-generalization",
        ("P2-03", "P2-04"),
        6,
        "p2-generalization",
        "npu",
        prerequisites=("p1-analyze",),
    ),
    "p2-evaluate": RoundSpec(
        "p2-evaluate",
        ("P2-02", "P2-05"),
        24,
        "p2-evaluate",
        "npu",
        prerequisites=("p2-generalization",),
    ),
    "p2-analyze": RoundSpec(
        "p2-analyze",
        ("P2-01", "P2-06", "P2-07"),
        27,
        "p2-analyze",
        "cpu",
        prerequisites=("p2-evaluate",),
    ),
}


# Smoke is intentionally excluded because it is a reusable subset of p0-core.
# These rounds must cover the complete catalog exactly once.
FINAL_ROUND_ORDER: tuple[str, ...] = (
    "p0-core",
    "p0-wjb",
    "p0-fairness",
    "p0-evaluate",
    "p0-analyze",
    "p0-manual",
    "p1-mechanism",
    "p1-data",
    "p1-evaluate",
    "p1-analyze",
    "p2-generalization",
    "p2-evaluate",
    "p2-analyze",
)


@dataclass(frozen=True)
class PoolLayout:
    cell_count: int
    shard_count: int
    max_cells_per_shard: int
    device_count: int


@dataclass(frozen=True)
class ShardResult:
    shard_index: int
    device_id: int
    returncode: int
    duration_seconds: float
    error: str = ""


ShardWorker = Callable[[int, int], int]


def select_round_cells(
    catalog: ExperimentCatalog,
    complete_plan: ExperimentPlan,
    spec: RoundSpec,
) -> tuple[ExperimentCell, ...]:
    unknown = sorted(set(spec.experiment_ids).difference(catalog.experiments))
    if unknown:
        raise ValueError(f"round {spec.name} references unknown experiments: {unknown}")
    selected = tuple(
        sorted(
            (
                cell
                for cell in complete_plan.cells
                if cell.experiment_id in set(spec.experiment_ids)
            ),
            key=lambda cell: cell.cell_id,
        )
    )
    if spec.cell_limit is not None:
        selected = selected[: spec.cell_limit]
    if len(selected) != spec.expected_cells:
        raise ValueError(
            f"round {spec.name} expected {spec.expected_cells} cells, got {len(selected)}; "
            "catalog drift requires an explicit round-plan review"
        )
    return selected


def build_round_plan(
    catalog: ExperimentCatalog,
    complete_plan: ExperimentPlan,
    spec: RoundSpec,
) -> ExperimentPlan:
    return validate_plan(
        ExperimentPlan(
            schema_version=complete_plan.schema_version,
            cells=select_round_cells(catalog, complete_plan, spec),
        )
    )


def derive_pool_layout(
    *,
    cell_count: int,
    requested_shards: int,
    requested_devices: int,
) -> PoolLayout:
    if cell_count <= 0:
        raise ValueError("cell_count must be positive")
    if requested_shards <= 0:
        raise ValueError("requested_shards must be positive")
    if requested_devices <= 0:
        raise ValueError("requested_devices must be positive")
    shard_count = min(cell_count, requested_shards)
    device_count = min(shard_count, requested_devices)
    return PoolLayout(
        cell_count=cell_count,
        shard_count=shard_count,
        max_cells_per_shard=math.ceil(cell_count / shard_count),
        device_count=device_count,
    )


def build_shard_command(
    *,
    python_executable: str,
    project_root: Path,
    plan_path: Path,
    state_root: Path,
    asset_manifest: Path,
    layout: PoolLayout,
    shard_index: int,
    device: str,
    device_id: int,
    dry_run: bool,
) -> tuple[str, ...]:
    if not 0 <= shard_index < layout.shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if device not in {"npu", "ppu", "cuda", "cpu"}:
        raise ValueError(f"unsupported device: {device}")
    command = [
        str(python_executable),
        str(Path(project_root) / "scripts" / "30_ablation.py"),
        "run",
        "--plan",
        str(plan_path),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(layout.shard_count),
        "--max-cells",
        str(layout.max_cells_per_shard),
        "--state-root",
        str(state_root),
        "--asset-manifest",
        str(asset_manifest),
        "--device",
        device,
        "--device-id",
        str(device_id),
        # Each cell deliberately retains the validated single-device recipe.
        "--num-devices",
        "1",
    ]
    if dry_run:
        command.append("--dry-run")
    return tuple(command)


def run_shard_pool(
    *,
    shard_count: int,
    device_ids: Sequence[int],
    worker: ShardWorker,
    stagger_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ShardResult, ...]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    normalized_devices = tuple(int(device_id) for device_id in device_ids)
    if not normalized_devices:
        raise ValueError("at least one device_id is required")
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError("device_ids must be unique")
    if any(device_id < 0 for device_id in normalized_devices):
        raise ValueError("device_ids must be non-negative")
    if stagger_seconds < 0:
        raise ValueError("stagger_seconds cannot be negative")

    pending: queue.Queue[int] = queue.Queue()
    for shard_index in range(shard_count):
        pending.put(shard_index)

    stop = threading.Event()
    result_lock = threading.Lock()
    results: list[ShardResult] = []

    def device_loop(device_id: int) -> None:
        while not stop.is_set():
            try:
                shard_index = pending.get_nowait()
            except queue.Empty:
                return
            started = time.monotonic()
            returncode = 1
            error = ""
            try:
                returncode = int(worker(shard_index, device_id))
            except Exception as exc:  # worker errors must become a durable pool failure
                error = f"{type(exc).__name__}: {exc}"
            finally:
                duration = time.monotonic() - started
                pending.task_done()
            with result_lock:
                results.append(
                    ShardResult(
                        shard_index=shard_index,
                        device_id=device_id,
                        returncode=returncode,
                        duration_seconds=duration,
                        error=error,
                    )
                )
            if returncode != 0 or error:
                stop.set()
                return

    threads: list[threading.Thread] = []
    for device_id in normalized_devices[: min(len(normalized_devices), shard_count)]:
        if stop.is_set():
            break
        thread = threading.Thread(
            target=device_loop,
            args=(device_id,),
            name=f"modelmate-device-{device_id}",
            daemon=False,
        )
        thread.start()
        threads.append(thread)
        if stagger_seconds and len(threads) < min(len(normalized_devices), shard_count):
            sleep(stagger_seconds)

    for thread in threads:
        thread.join()
    return tuple(sorted(results, key=lambda result: result.shard_index))
