from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from src.features.first_gen_token import build_chat_batch


@dataclass
class InterventionArtifact:
    artifact_path: str
    best_layer_idx: int
    best_threshold: float
    rank: int
    target_label: str
    reference_label: str
    basis: torch.Tensor
    target_center: torch.Tensor
    reference_center: torch.Tensor


@dataclass
class InterventionSpec:
    layer_idx: int
    threshold: float
    target_label: str
    reference_label: str
    basis: torch.Tensor
    target_center: torch.Tensor
    reference_center: torch.Tensor


def load_intervention_artifact(path: str | Path) -> InterventionArtifact:
    artifact_path = Path(path)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    best_layer_idx = int(payload["best_layer_idx"])
    best_model = payload["models"][best_layer_idx]
    return InterventionArtifact(
        artifact_path=str(artifact_path),
        best_layer_idx=best_layer_idx,
        best_threshold=float(payload["best_threshold"]),
        rank=int(payload["rank"]),
        target_label=str(payload["target_label"]),
        reference_label=str(payload["reference_label"]),
        basis=best_model["basis"].to(dtype=torch.float32),
        target_center=best_model["target_center"].to(dtype=torch.float32),
        reference_center=best_model["reference_center"].to(dtype=torch.float32),
    )


def build_intervention_spec(artifact: InterventionArtifact) -> InterventionSpec:
    return InterventionSpec(
        layer_idx=artifact.best_layer_idx,
        threshold=artifact.best_threshold,
        target_label=artifact.target_label,
        reference_label=artifact.reference_label,
        basis=artifact.basis,
        target_center=artifact.target_center,
        reference_center=artifact.reference_center,
    )


def _last_non_padding_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    reversed_mask = torch.flip(attention_mask.to(dtype=torch.long), dims=[1])
    from_end = torch.argmax(reversed_mask, dim=1)
    return attention_mask.size(1) - 1 - from_end


def _make_last_token_affine_projection_hook(
    *,
    basis: torch.Tensor,
    target_center: torch.Tensor,
    last_positions: torch.Tensor,
    alpha: float,
    cache: Dict[str, torch.Tensor],
):
    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            hidden_states = output[0]
            tail = output[1:]
            modified = _apply_projection(hidden_states, basis, target_center, last_positions, alpha)
            cache["selected_hidden"] = _gather_selected_hidden(modified, last_positions)
            return (modified,) + tail
        modified = _apply_projection(output, basis, target_center, last_positions, alpha)
        cache["selected_hidden"] = _gather_selected_hidden(modified, last_positions)
        return modified

    return hook


def _apply_projection(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
    target_center: torch.Tensor,
    last_positions: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha == 0.0:
        return hidden_states

    layer_device = hidden_states.device
    layer_dtype = hidden_states.dtype
    basis_local = basis.to(device=layer_device, dtype=layer_dtype)
    center_local = target_center.to(device=layer_device, dtype=layer_dtype)
    token_positions = last_positions.to(device=layer_device)
    batch_indices = torch.arange(hidden_states.size(0), device=layer_device)

    selected = hidden_states[batch_indices, token_positions, :]
    delta = selected - center_local
    coeff = delta @ basis_local.T
    projected = coeff @ basis_local
    updated = selected - (alpha * projected)

    modified = hidden_states.clone()
    modified[batch_indices, token_positions, :] = updated
    return modified


def _gather_selected_hidden(hidden_states: torch.Tensor, last_positions: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_positions.to(hidden_states.device), :].detach().cpu().to(dtype=torch.float32)


@torch.no_grad()
def run_intervened_last_token_hidden(
    *,
    model: Any,
    tokenizer: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    spec: InterventionSpec,
    alpha: float,
    max_length: int,
) -> torch.Tensor:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    encoded, _ = build_chat_batch(
        tokenizer=tokenizer,
        messages_batch=messages_batch,
        max_length=max_length,
        device=device,
    )
    last_positions = _last_non_padding_positions(encoded["attention_mask"])
    layer = model.model.layers[spec.layer_idx]
    cache: Dict[str, torch.Tensor] = {}
    hook = layer.register_forward_hook(
        _make_last_token_affine_projection_hook(
            basis=spec.basis,
            target_center=spec.target_center,
            last_positions=last_positions,
            alpha=alpha,
            cache=cache,
        )
    )
    try:
        outputs = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        hook.remove()

    if "selected_hidden" in cache:
        return cache["selected_hidden"]

    layer_hidden = outputs.hidden_states[spec.layer_idx + 1]
    return _gather_selected_hidden(layer_hidden, last_positions)
