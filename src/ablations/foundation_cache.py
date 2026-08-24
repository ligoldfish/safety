from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class FoundationCacheError(RuntimeError):
    """Raised when a declared reusable Phase-1 foundation is incomplete."""


def required_foundation_artifacts(
    phase1_root: str | Path, *, validation_only: bool
) -> tuple[Path, ...]:
    root = Path(phase1_root)
    target_splits = ("alignment", "analysis_val") if validation_only else (
        "alignment",
        "analysis_val",
        "pan_test",
        "sanity_test",
    )
    return (
        root / "safe_subspaces" / "manifest.json",
        root / "layer_pairing" / "teacher_student_layer_pairs.json",
        root / "semantic_bases" / "manifest.json",
        root / "semantic_bases" / "bridge_artifact.pt",
        *(
            root / "student_targets" / f"student_safe_targets_{split}" / "manifest.json"
            for split in target_splits
        ),
    )


def foundation_is_ready(
    phase1_root: str | Path, cache_key: str, *, validation_only: bool
) -> bool:
    root = Path(phase1_root)
    marker = root / ".foundation-ready.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
        return False
    if bool(payload.get("validation_only")) != bool(validation_only):
        return False
    declared = payload.get("artifacts")
    if not isinstance(declared, dict):
        return False
    for path in required_foundation_artifacts(root, validation_only=validation_only):
        relative = path.relative_to(root).as_posix()
        expected_size = declared.get(relative)
        if type(expected_size) is not int or expected_size <= 0:
            return False
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
    return True


def mark_foundation_ready(
    phase1_root: str | Path, cache_key: str, *, validation_only: bool
) -> None:
    root = Path(phase1_root)
    artifacts = required_foundation_artifacts(root, validation_only=validation_only)
    missing = [str(path) for path in artifacts if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FoundationCacheError(f"Phase-1 foundation is incomplete: {missing}")
    payload = {
        "schema_version": 1,
        "cache_key": cache_key,
        "validation_only": bool(validation_only),
        "artifacts": {
            path.relative_to(root).as_posix(): path.stat().st_size for path in artifacts
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".foundation-ready.json"
    temporary = root / f".foundation-ready.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


@contextmanager
def foundation_lock(phase1_root: str | Path) -> Iterator[None]:
    """Serialize one cache producer while allowing later cells to reuse it."""

    root = Path(phase1_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".foundation.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
