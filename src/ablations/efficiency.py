from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return int(root.stat().st_size)
    total = 0
    for current, _, names in os.walk(root, followlinks=False):
        for name in names:
            file = Path(current) / name
            if not file.is_symlink():
                try:
                    total += int(file.stat().st_size)
                except FileNotFoundError:
                    pass
    return total


@dataclass(frozen=True)
class StageEfficiency:
    schema_version: int
    cell_id: str
    stage: str
    wall_seconds: float
    peak_memory_bytes: int
    disk_delta_bytes: int
    device_count: int
    device_hours: float
    exit_code: int

    def to_dict(self) -> dict:
        return asdict(self)


class StageProfiler:
    def __init__(
        self,
        stage: str,
        *,
        output_root: str | Path,
        cell_id: str = "",
        device_count: int = 1,
    ) -> None:
        if not str(stage).strip():
            raise ValueError("stage must be non-empty")
        if type(device_count) is not int or device_count < 0:
            raise ValueError("device_count must be a non-negative integer")
        self.stage = str(stage)
        self.output_root = Path(output_root)
        self.cell_id = str(cell_id)
        self.device_count = device_count
        self._started_at: float | None = None
        self._disk_before = 0
        self._peak_memory = 0

    def start(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("profiler already started")
        self._disk_before = directory_size(self.output_root)
        self._started_at = time.perf_counter()

    def update_peak_memory(self, peak_bytes: int) -> None:
        if type(peak_bytes) is not int or peak_bytes < 0:
            raise ValueError("peak memory must be a non-negative integer")
        self._peak_memory = max(self._peak_memory, peak_bytes)

    def finish(self, *, exit_code: int) -> StageEfficiency:
        if self._started_at is None:
            raise RuntimeError("profiler is not running")
        wall_seconds = max(0.0, time.perf_counter() - self._started_at)
        self._started_at = None
        disk_delta = directory_size(self.output_root) - self._disk_before
        return StageEfficiency(
            schema_version=1,
            cell_id=self.cell_id,
            stage=self.stage,
            wall_seconds=wall_seconds,
            peak_memory_bytes=self._peak_memory,
            disk_delta_bytes=disk_delta,
            device_count=self.device_count,
            device_hours=wall_seconds * self.device_count / 3600.0,
            exit_code=int(exit_code),
        )

    def profile_noop(self) -> StageEfficiency:
        self.start()
        return self.finish(exit_code=0)


def summarize_efficiency(records: Iterable[StageEfficiency]) -> dict:
    items = tuple(records)
    if not items:
        raise ValueError("at least one efficiency record is required")
    cell_ids = {item.cell_id for item in items}
    if len(cell_ids) != 1:
        raise ValueError("efficiency records must belong to the same cell")
    return {
        "schema_version": 1,
        "cell_id": items[0].cell_id,
        "wall_seconds": sum(item.wall_seconds for item in items),
        "peak_memory_bytes": max(item.peak_memory_bytes for item in items),
        "disk_delta_bytes": sum(item.disk_delta_bytes for item in items),
        "device_hours": sum(item.device_hours for item in items),
        "stages": [item.to_dict() for item in items],
    }
