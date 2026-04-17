from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class HiddenStateSplit:
    split_dir: str
    layer_tensors: Dict[int, torch.Tensor]
    labels: List[str]
    sample_ids: List[str]

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    @property
    def available_layers(self) -> List[int]:
        return sorted(self.layer_tensors.keys())

    def label_counts(self) -> Dict[str, int]:
        return dict(Counter(self.labels))


def load_hidden_state_split(
    split_dir: str | Path,
    *,
    max_samples_per_label: int = 0,
    selected_layers: List[int] | None = None,
) -> HiddenStateSplit:
    split_path = Path(split_dir)
    part_paths = sorted(split_path.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No hidden-state shards found under: {split_path}")

    kept_labels: List[str] = []
    kept_sample_ids: List[str] = []
    counts: Counter[str] = Counter()
    layer_buffers: Dict[int, List[torch.Tensor]] = {}

    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu")
        layer_keys = sorted(int(key) for key in payload["hidden_by_layer"].keys())
        chosen_layers = selected_layers or layer_keys
        keep_indices: List[int] = []

        for idx, label in enumerate(payload["labels"]):
            label_text = str(label)
            if max_samples_per_label > 0 and counts[label_text] >= max_samples_per_label:
                continue
            counts[label_text] += 1
            keep_indices.append(idx)
            kept_labels.append(label_text)
            kept_sample_ids.append(str(payload["sample_ids"][idx]))

        if not keep_indices:
            continue

        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        for layer_idx in chosen_layers:
            layer_tensor = payload["hidden_by_layer"][str(layer_idx)].index_select(0, index_tensor)
            layer_buffers.setdefault(layer_idx, []).append(layer_tensor.to(dtype=torch.float32))

    if not kept_labels:
        raise ValueError(f"No samples left after filtering hidden states from: {split_path}")

    return HiddenStateSplit(
        split_dir=str(split_path),
        layer_tensors={
            layer_idx: torch.cat(buffers, dim=0)
            for layer_idx, buffers in sorted(layer_buffers.items())
        },
        labels=kept_labels,
        sample_ids=kept_sample_ids,
    )
