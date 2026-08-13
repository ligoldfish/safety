from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_SCRIPT_PHASES = {
    "01_extract_hidden_states.py": "extract",
    "02_analyze_teacher_layers.py": "subspace",
    "03_build_teacher_safe_subspace.py": "subspace",
    "04_pair_layers.py": "semantic_basis",
    "05_build_semantic_bases.py": "semantic_basis",
    "06_project_teacher_safe_component.py": "decompose",
    "07_decompose_teacher_semantics.py": "decompose",
    "08_recompose_student_targets.py": "recompose",
    "09_train_student_semalign.py": "train",
}


def phase_for_script(script_name: str) -> str | None:
    return _SCRIPT_PHASES.get(Path(script_name).name)


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
    peak_memory_bytes: int | None
    disk_delta_bytes: int
    device_count: int
    device_hours: float
    exit_code: int
    script: str = ""
    memory_measurement: str = "manual"

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
        "peak_memory_bytes": max(
            (item.peak_memory_bytes for item in items if item.peak_memory_bytes is not None),
            default=None,
        ),
        "disk_delta_bytes": sum(item.disk_delta_bytes for item in items),
        "device_hours": sum(item.device_hours for item in items),
        "stages": [item.to_dict() for item in items],
    }


def _linux_process_tree_rss(pid: int) -> int | None:
    pending = [int(pid)]
    seen: set[int] = set()
    total = 0
    measured = False
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    measured = True
                    break
            children = Path(f"/proc/{current}/task/{current}/children")
            if children.is_file():
                pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            continue
    return total if measured else None


def _windows_process_rss(pid: int) -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, int(pid))
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return None
            return int(counters.WorkingSetSize)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except (AttributeError, OSError, ValueError):
        return None


def _process_rss(pid: int) -> tuple[int | None, str]:
    if os.name == "nt":
        return _windows_process_rss(pid), "process_rss"
    return _linux_process_tree_rss(pid), "process_tree_rss"


def run_profiled_subprocess(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    stage: str,
    script: str,
    output_root: str | Path,
    cell_id: str,
    device_count: int,
    poll_seconds: float = 0.05,
) -> tuple[int, StageEfficiency]:
    if not command:
        raise ValueError("profiled command must be non-empty")
    root = Path(output_root)
    disk_before = directory_size(root)
    started = time.perf_counter()
    process = subprocess.Popen(list(command), cwd=str(cwd), env=dict(env))
    peak: int | None = None
    measurement = "unavailable"
    while True:
        rss, source = _process_rss(process.pid)
        if rss is not None:
            peak = rss if peak is None else max(peak, rss)
            measurement = source
        returncode = process.poll()
        if returncode is not None:
            break
        time.sleep(max(0.001, float(poll_seconds)))
    wall_seconds = max(0.0, time.perf_counter() - started)
    record = StageEfficiency(
        schema_version=1,
        cell_id=str(cell_id),
        stage=str(stage),
        wall_seconds=wall_seconds,
        peak_memory_bytes=peak,
        disk_delta_bytes=directory_size(root) - disk_before,
        device_count=int(device_count),
        device_hours=wall_seconds * int(device_count) / 3600.0,
        exit_code=int(returncode),
        script=Path(script).name,
        memory_measurement=measurement,
    )
    return int(returncode), record


def append_efficiency_record(path: str | Path, record: StageEfficiency) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(destination, flags, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("partial efficiency log write")
    finally:
        os.close(descriptor)
