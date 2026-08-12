from __future__ import annotations

from typing import Iterable


def resolve_lora_layers(
    selected_layers: Iterable[int],
    *,
    placement: str,
    num_layers: int,
) -> list[int]:
    selected = sorted({int(layer) for layer in selected_layers})
    if not selected:
        raise ValueError("selected_layers must be non-empty")
    normalized = str(placement).strip().lower()
    if normalized == "selected":
        return selected
    if normalized == "all_layers_parameter_matched":
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        return list(range(num_layers))
    raise ValueError(f"unsupported LoRA placement: {placement}")
