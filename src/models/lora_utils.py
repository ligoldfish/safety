from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        output_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LoRA.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        parameter_device = self.base_layer.weight.device

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base_layer.in_features,
                dtype=torch.float32,
                device=parameter_device,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                dtype=torch.float32,
                device=parameter_device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if output_mask is not None:
            if tuple(output_mask.shape) != (base_layer.out_features,):
                raise ValueError("output_mask must have shape [out_features].")
            self.register_buffer(
                "output_mask",
                output_mask.to(device=parameter_device, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.output_mask = None

        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_input = self.dropout(x).to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        update = (lora_input @ self.lora_A.T) @ self.lora_B.T
        if self.output_mask is not None:
            update = update * self.output_mask.to(device=update.device, dtype=update.dtype)
        return base_out + update.to(device=base_out.device, dtype=base_out.dtype) * self.scaling


@dataclass
class LoRAInjectionResult:
    replaced_module_names: List[str]
    layer_indices: List[int]


def _resolve_module(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _module_exists(root: nn.Module, path: str) -> bool:
    try:
        parent, attr_name = _resolve_module(root, path)
        getattr(parent, attr_name)
        return True
    except (AttributeError, IndexError):
        return False


def resolve_target_suffixes(
    model: nn.Module,
    *,
    layer_indices: Iterable[int],
    requested_suffixes: Iterable[str],
) -> List[str]:
    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty.")
    probe_layer = layer_indices[0]
    resolved: List[str] = []
    for suffix in requested_suffixes:
        direct_path = f"model.layers.{probe_layer}.{suffix}"
        if _module_exists(model, direct_path):
            resolved.append(suffix)
            continue

        if suffix == "self_attn.v_proj":
            fallback = "linear_attn.in_proj_qkv"
        elif suffix == "self_attn.o_proj":
            fallback = "linear_attn.out_proj"
        else:
            fallback = suffix

        fallback_path = f"model.layers.{probe_layer}.{fallback}"
        if _module_exists(model, fallback_path):
            resolved.append(fallback)
            continue

        raise AttributeError(
            f"Could not resolve requested LoRA target '{suffix}' on layer {probe_layer}."
        )
    return resolved


def inject_lora_modules(
    model: nn.Module,
    *,
    layer_indices: Iterable[int],
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRAInjectionResult:
    replaced: List[str] = []
    layer_indices = [int(x) for x in layer_indices]
    for layer_idx in layer_indices:
        for requested_suffix in target_suffixes:
            resolved_suffix = resolve_target_suffixes(
                model,
                layer_indices=[layer_idx],
                requested_suffixes=[requested_suffix],
            )[0]
            module_name = f"model.layers.{int(layer_idx)}.{resolved_suffix}"
            parent, attr_name = _resolve_module(model, module_name)
            module = getattr(parent, attr_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {module_name}, found {type(module).__name__}.")
            output_mask = None
            if requested_suffix == "self_attn.v_proj" and resolved_suffix == "linear_attn.in_proj_qkv":
                linear_attn = model.model.layers[int(layer_idx)].linear_attn
                value_dim = int(linear_attn.value_dim)
                output_mask = torch.zeros(module.out_features, dtype=torch.float32)
                output_mask[-value_dim:] = 1.0
            setattr(
                parent,
                attr_name,
                LoRALinear(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    output_mask=output_mask,
                ),
            )
            replaced.append(module_name)
    return LoRAInjectionResult(replaced_module_names=replaced, layer_indices=layer_indices)


def inject_lora_modules_by_names(
    model: nn.Module,
    *,
    module_names: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRAInjectionResult:
    replaced: List[str] = []
    touched_layers: List[int] = []
    for module_name in module_names:
        parent, attr_name = _resolve_module(model, module_name)
        module = getattr(parent, attr_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {module_name}, found {type(module).__name__}.")

        output_mask = None
        if module_name.endswith("linear_attn.in_proj_qkv"):
            parts = module_name.split(".")
            try:
                layer_idx = int(parts[2])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Could not parse layer index from module name: {module_name}") from exc
            linear_attn = model.model.layers[layer_idx].linear_attn
            value_dim = int(linear_attn.value_dim)
            output_mask = torch.zeros(module.out_features, dtype=torch.float32)
            output_mask[-value_dim:] = 1.0
            touched_layers.append(layer_idx)
        else:
            parts = module_name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                touched_layers.append(int(parts[2]))

        setattr(
            parent,
            attr_name,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                output_mask=output_mask,
            ),
        )
        replaced.append(module_name)
    return LoRAInjectionResult(
        replaced_module_names=replaced,
        layer_indices=sorted(set(touched_layers)),
    )


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def trainable_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return trainable, total
