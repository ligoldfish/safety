from __future__ import annotations

import hashlib
import random
from typing import Mapping, Sequence


def build_blind_packet(
    rows: Sequence[Mapping], *, per_stratum: int, seed: int
) -> tuple[list[dict], dict[str, dict]]:
    if type(per_stratum) is not int or per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    groups: dict[str, list[Mapping]] = {}
    for row in rows:
        groups.setdefault(str(row["stratum"]), []).append(row)
    selected: list[Mapping] = []
    for stratum, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: str(row["sample_id"]))
        random.Random(f"{seed}:{stratum}").shuffle(ordered)
        if len(ordered) < per_stratum:
            raise ValueError(f"insufficient rows for stratum {stratum}")
        selected.extend(ordered[:per_stratum])
    packet: list[dict] = []
    key: dict[str, dict] = {}
    for row in sorted(selected, key=lambda item: (str(item["stratum"]), str(item["sample_id"]))):
        blind_id = hashlib.sha256(f"{seed}:{row['sample_id']}".encode("utf-8")).hexdigest()[:16]
        packet.append({"blind_id": blind_id, "stratum": str(row["stratum"]), "prompt": str(row["prompt"]), "response": str(row["response"]), "annotation": ""})
        key[blind_id] = {"sample_id": str(row["sample_id"]), "method": str(row["method"]), "stratum": str(row["stratum"])}
    return packet, key


def _annotation_map(rows: Sequence[Mapping], allowed_labels: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        blind_id = str(row.get("blind_id", ""))
        label = str(row.get("annotation", "")).strip()
        if not blind_id or blind_id in result:
            raise ValueError("annotation blind IDs must be non-empty and unique")
        if label not in allowed_labels:
            raise ValueError(f"unknown annotation label: {label}")
        result[blind_id] = label
    return result


def import_double_annotations(
    key: Mapping[str, Mapping],
    rater_a: Sequence[Mapping],
    rater_b: Sequence[Mapping],
    *,
    allowed_labels: set[str],
) -> list[dict]:
    left = _annotation_map(rater_a, allowed_labels)
    right = _annotation_map(rater_b, allowed_labels)
    expected = set(map(str, key))
    if set(left) != expected or set(right) != expected:
        raise ValueError("both raters must annotate exactly every blind ID")
    return [
        {"blind_id": blind_id, **dict(key[blind_id]), "rater_a": left[blind_id], "rater_b": right[blind_id]}
        for blind_id in sorted(expected)
    ]
