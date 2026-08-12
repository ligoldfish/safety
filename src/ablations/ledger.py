from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class LedgerError(RuntimeError):
    """Raised when a run ledger would become ambiguous or unsafe."""


class RunState(str, Enum):
    PLANNED = "PLANNED"
    BLOCKED = "BLOCKED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


_TRANSITIONS = {
    RunState.PLANNED: {RunState.BLOCKED, RunState.READY},
    RunState.BLOCKED: {RunState.BLOCKED, RunState.READY},
    RunState.READY: {RunState.RUNNING, RunState.BLOCKED},
    RunState.RUNNING: {RunState.COMPLETED, RunState.FAILED},
    RunState.FAILED: {RunState.READY, RunState.BLOCKED},
    RunState.COMPLETED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class ExperimentLedger:
    root: Path
    cell_id: str
    config_hash: str

    def __init__(self, root: str | Path, cell_id: str, *, config_hash: str) -> None:
        if not str(cell_id).strip():
            raise ValueError("cell_id must be non-empty")
        if not str(config_hash).strip():
            raise ValueError("config_hash must be non-empty")
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "cell_id", str(cell_id))
        object.__setattr__(self, "config_hash", str(config_hash))

    @property
    def cell_dir(self) -> Path:
        return self.root / self.cell_id

    @property
    def status_path(self) -> Path:
        return self.cell_dir / "status.json"

    @property
    def lock_path(self) -> Path:
        return self.cell_dir / ".writer.lock"

    def read(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            raise LedgerError(f"ledger is not initialized: {self.cell_id}")
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerError(f"invalid ledger for {self.cell_id}") from exc
        if not isinstance(payload, dict):
            raise LedgerError(f"invalid ledger for {self.cell_id}")
        return payload

    def initialize(self, *, dry_run: bool = False) -> dict[str, Any]:
        if self.status_path.exists():
            current = self.read()
            if current.get("config_hash") != self.config_hash:
                raise LedgerError(
                    f"configuration fingerprint mismatch for {self.cell_id}; "
                    "use a new cell/output directory"
                )
            return current
        payload = {
            "schema_version": 1,
            "cell_id": self.cell_id,
            "config_hash": self.config_hash,
            "state": RunState.PLANNED.value,
            "dry_run": bool(dry_run),
            "created_at": _now(),
            "updated_at": _now(),
        }
        _atomic_json(self.status_path, payload)
        return payload

    def transition(
        self,
        target: RunState,
        *,
        artifact_hash: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not self.lock_path.is_file():
            raise LedgerError(f"writer lock is required to modify ledger: {self.cell_id}")
        if not isinstance(target, RunState):
            target = RunState(str(target))
        current = self.read()
        source = RunState(current["state"])
        if target not in _TRANSITIONS[source]:
            raise LedgerError(f"illegal state transition: {source.value} -> {target.value}")
        if target is RunState.COMPLETED:
            if current.get("dry_run"):
                raise LedgerError("dry-run ledger cannot be marked COMPLETED")
            if not artifact_hash:
                raise LedgerError("COMPLETED requires a non-empty artifact_hash")
        updated = dict(current)
        updated["state"] = target.value
        updated["updated_at"] = _now()
        if artifact_hash is not None:
            updated["artifact_hash"] = str(artifact_hash)
        if reason is not None:
            updated["reason"] = str(reason)
        _atomic_json(self.status_path, updated)
        return updated

    @contextmanager
    def acquire_lock(self) -> Iterator[None]:
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "created_at": _now()},
            sort_keys=True,
        ).encode("utf-8")
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise LedgerError(f"cell is already locked by another writer: {self.cell_id}") from exc
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(metadata)
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
