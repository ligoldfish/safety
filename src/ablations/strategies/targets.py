from __future__ import annotations

import hashlib
import random
from typing import Mapping

import torch


TargetMap = Mapping[str, Mapping[int, torch.Tensor]]


def _seed_for_group(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _derangement(items: tuple[str, ...], *, seed: int) -> tuple[str, ...]:
    if len(items) < 2:
        raise ValueError("permutation control requires at least two source samples per group")
    shuffled = list(items)
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(items, shuffled)):
            return tuple(shuffled)
    return items[1:] + items[:1]


def permute_target_map(
    target_map: TargetMap,
    labels: Mapping[str, str],
    *,
    mode: str,
    seed: int,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, str]]:
    sample_ids = tuple(sorted(str(item) for item in target_map))
    normalized_labels = {str(item): str(label) for item, label in labels.items()}
    if set(sample_ids) != set(normalized_labels):
        raise ValueError("target_map and labels must cover exactly the same stable sample IDs")
    normalized = str(mode).strip().lower()
    manifest: dict[str, str] = {}
    if normalized == "within_label_permutation":
        groups: dict[str, list[str]] = {}
        for sample_id in sample_ids:
            groups.setdefault(normalized_labels[sample_id], []).append(sample_id)
        for label, group in sorted(groups.items()):
            ordered = tuple(sorted(group))
            sources = _derangement(ordered, seed=_seed_for_group(seed, label))
            manifest.update(zip(ordered, sources))
    elif normalized == "cross_label_permutation":
        groups: dict[str, tuple[str, ...]] = {}
        for label in sorted({normalized_labels[item] for item in sample_ids}):
            groups[label] = tuple(item for item in sample_ids if normalized_labels[item] == label)
        if len(groups) != 2:
            raise ValueError("cross_label_permutation requires exactly two labels")
        left_label, right_label = tuple(groups)
        left, right = groups[left_label], groups[right_label]
        if len(left) != len(right):
            raise ValueError("cross_label_permutation requires equal label counts")
        right_sources = list(right)
        left_sources = list(left)
        random.Random(_seed_for_group(seed, "cross-right")).shuffle(right_sources)
        random.Random(_seed_for_group(seed, "cross-left")).shuffle(left_sources)
        manifest.update(zip(left, right_sources))
        manifest.update(zip(right, left_sources))
    else:
        raise ValueError(f"unsupported target permutation mode: {mode}")
    result: dict[str, dict[int, torch.Tensor]] = {}
    for destination, source in manifest.items():
        result[destination] = {
            int(layer): tensor.detach().clone()
            for layer, tensor in target_map[source].items()
        }
    return result, manifest
