from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def prompt_text(record: Mapping) -> str:
    if record.get("user_text"):
        return str(record["user_text"])
    if record.get("prompt"):
        return str(record["prompt"])
    messages = record.get("messages") or []
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if str(message.get("role", "")).lower() == "user"
    )


def canonical_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def prompt_sha256(record: Mapping) -> str:
    return hashlib.sha256(canonical_prompt(prompt_text(record)).encode("utf-8")).hexdigest()


def _drop_record(record: Mapping, *, reason: str) -> dict:
    return {
        "id": str(record.get("id", "")),
        "label": str(record.get("label", "")),
        "source": str(record.get("source") or record.get("source_dataset") or record.get("dataset") or ""),
        "prompt_sha256": prompt_sha256(record),
        "drop_reason": reason,
    }


def exclude_protected_prompts(
    candidates: Sequence[Mapping],
    protected: Sequence[Mapping],
    *,
    reason: str,
) -> tuple[list[dict], list[dict]]:
    protected_hashes = {prompt_sha256(record) for record in protected}
    kept: list[dict] = []
    dropped: list[dict] = []
    for record in candidates:
        if prompt_sha256(record) in protected_hashes:
            dropped.append(_drop_record(record, reason=reason))
        else:
            kept.append(dict(record))
    return kept, dropped


def deduplicate_prompts(
    records: Sequence[Mapping],
    *,
    reason: str = "duplicate_prompt",
) -> tuple[list[dict], list[dict]]:
    """Keep the first record for each canonical prompt, preserving order."""

    seen: set[str] = set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for record in records:
        prompt_hash = prompt_sha256(record)
        if prompt_hash in seen:
            dropped.append(_drop_record(record, reason=reason))
            continue
        seen.add(prompt_hash)
        kept.append(dict(record))
    return kept, dropped


def stratified_holdout(
    records: Sequence[Mapping], *, fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("holdout fraction must be between zero and one")
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get("label", "")), []).append(dict(record))
    train: list[dict] = []
    evaluation: list[dict] = []
    for label, group in sorted(groups.items()):
        by_prompt: dict[str, list[dict]] = {}
        for item in group:
            by_prompt.setdefault(prompt_sha256(item), []).append(item)
        prompt_groups = [
            sorted(items, key=lambda item: str(item.get("id", "")))
            for _, items in sorted(by_prompt.items())
        ]
        random.Random(f"{int(seed)}:{label}").shuffle(prompt_groups)
        holdout_size = max(1, int(round(len(group) * float(fraction))))
        if len(prompt_groups) < 2 or holdout_size >= len(group):
            raise ValueError(f"holdout consumes entire label group: {label}")
        selected_group_count = 0
        selected_record_count = 0
        for items in prompt_groups[:-1]:
            if selected_record_count >= holdout_size:
                break
            selected_group_count += 1
            selected_record_count += len(items)
        evaluation.extend(item for items in prompt_groups[:selected_group_count] for item in items)
        train.extend(item for items in prompt_groups[selected_group_count:] for item in items)
    key = lambda item: (str(item.get("label", "")), str(item.get("id", "")))
    return sorted(train, key=key), sorted(evaluation, key=key)


def _duplicate_count(records: Iterable[Mapping]) -> int:
    counts = Counter(prompt_sha256(record) for record in records)
    return sum(count - 1 for count in counts.values() if count > 1)


def audit_train_eval_splits(train: Sequence[Mapping], evaluation: Sequence[Mapping]) -> dict:
    train_hashes = {prompt_sha256(record) for record in train}
    eval_hashes = {prompt_sha256(record) for record in evaluation}
    overlap = sorted(train_hashes & eval_hashes)
    return {
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_duplicate_count": _duplicate_count(train),
        "eval_duplicate_count": _duplicate_count(evaluation),
        "overlap_count": len(overlap),
        "overlap_prompt_sha256": overlap,
    }


def write_data_audit(
    path: str | Path,
    *,
    dataset: str,
    train: Sequence[Mapping],
    evaluation: Sequence[Mapping],
    drops: Sequence[Mapping],
    license_name: str,
    intended_use: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "license": license_name,
        "intended_use": intended_use,
        **audit_train_eval_splits(train, evaluation),
        "drops": [dict(item) for item in drops],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target
