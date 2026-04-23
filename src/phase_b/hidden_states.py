from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

CANONICAL_FEATURE_TYPE = "first_generated_token_hidden_state"
LEGACY_FEATURE_TYPE = "final_response_prefix_hidden_state"

# Default path only accepts the canonical first-generated-token tag. An empty
# ``feature_type`` is tolerated as well for shards written by pre-manifest
# versions of 01_extract_hidden_states.py; the on-disk layout is unchanged
# (last-position hidden state of the generation-prompt rendering).
STRICT_SUPPORTED_FEATURE_TYPES = frozenset({"", CANONICAL_FEATURE_TYPE})

# Legacy tag produced by the now-removed thinking-era extractor. The stored
# vectors come from ``render_qwen_final_response_prefix`` (assistant role
# opened with empty content rather than the canonical generation prompt), so
# the position is *close to* but NOT identical with the first-generated-token
# position. Mixing the two into one analysis silently biases the safety
# subspace, so reads of legacy shards must be explicitly opted into.
LEGACY_SUPPORTED_FEATURE_TYPES = frozenset(
    STRICT_SUPPORTED_FEATURE_TYPES | {LEGACY_FEATURE_TYPE}
)


@dataclass
class HiddenStateSplit:
    split_dir: str
    layer_tensors: Dict[int, torch.Tensor]
    labels: List[str]
    sample_ids: List[str]
    # ``legacy_final_response_prefix`` is True only when any shard in this
    # split carried the legacy feature_type tag. Consumers that care about
    # reproducibility (manifests, analysis reports) should surface this flag.
    legacy_final_response_prefix: bool = False

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
    allow_legacy_final_response_prefix: bool = False,
) -> HiddenStateSplit:
    """Load hidden-state shards produced by ``scripts/01_extract_hidden_states.py``.

    Default behaviour is strict: only shards tagged as
    ``first_generated_token_hidden_state`` (or with an empty tag, for very
    early shards) are accepted. Legacy thinking-era shards tagged
    ``final_response_prefix_hidden_state`` are rejected unless the caller
    explicitly passes ``allow_legacy_final_response_prefix=True``. When the
    legacy path is taken, a warning is emitted and the returned
    ``HiddenStateSplit.legacy_final_response_prefix`` flag is set to ``True``
    so the caller can record the deviation in its manifest/log.
    """

    split_path = Path(split_dir)
    part_paths = sorted(split_path.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No hidden-state shards found under: {split_path}")

    supported = (
        LEGACY_SUPPORTED_FEATURE_TYPES
        if allow_legacy_final_response_prefix
        else STRICT_SUPPORTED_FEATURE_TYPES
    )

    kept_labels: List[str] = []
    kept_sample_ids: List[str] = []
    counts: Counter[str] = Counter()
    layer_buffers: Dict[int, List[torch.Tensor]] = {}
    saw_legacy = False

    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu", weights_only=True)
        feature_type = str(payload.get("feature_type", ""))
        if feature_type not in supported:
            if feature_type == LEGACY_FEATURE_TYPE:
                raise ValueError(
                    f"Legacy hidden-state shard detected in {part_path}: "
                    f"feature_type='{LEGACY_FEATURE_TYPE}'. Default pipeline "
                    f"requires '{CANONICAL_FEATURE_TYPE}'. Regenerate via "
                    "scripts/01_extract_hidden_states.py, or explicitly opt in "
                    "by passing allow_legacy_final_response_prefix=True."
                )
            raise ValueError(
                f"Unsupported hidden-state feature type in {part_path}: "
                f"'{feature_type}'. Expected '{CANONICAL_FEATURE_TYPE}'."
            )
        if feature_type == LEGACY_FEATURE_TYPE:
            saw_legacy = True
            warnings.warn(
                f"Loading LEGACY final_response_prefix hidden-state shard {part_path}. "
                "Position semantics differ from the canonical first_generated_token "
                "pipeline; analysis output will carry legacy_final_response_prefix=True.",
                stacklevel=2,
            )
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
        legacy_final_response_prefix=saw_legacy,
    )
