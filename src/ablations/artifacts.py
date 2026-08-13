from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


PHASE1_REQUIRED_FIELDS = frozenset(
    {
        "teacher",
        "student",
        "teacher_tokenizer",
        "student_tokenizer",
        "dataset",
        "seed",
        "representation",
        "layer_selection",
        "subspace",
        "bridge",
        "pairing",
        "target",
        "commit",
        "schema_version",
    }
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported value in canonical artifact payload: {type(value).__name__}")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def phase1_artifact_key(fields: Mapping[str, Any]) -> str:
    missing = PHASE1_REQUIRED_FIELDS - set(fields)
    if missing:
        raise ValueError(f"missing phase1 artifact fields: {sorted(missing)}")
    unexpected = set(fields) - PHASE1_REQUIRED_FIELDS
    if unexpected:
        raise ValueError(f"unexpected phase1 artifact fields: {sorted(unexpected)}")
    return f"phase1-v1-{canonical_hash(fields)}"
