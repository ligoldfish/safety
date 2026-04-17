from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence, TypeVar


T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[dict[str, Any]]:
    target = Path(path)
    rows: List[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunked(items: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]
