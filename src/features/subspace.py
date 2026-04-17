from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SafeSubspaceResult:
    layer_idx: int
    k: int
    basis: torch.Tensor
    singular_values: torch.Tensor
    mean_diff: torch.Tensor
    explained_ratio_topk: torch.Tensor
    harmful_count: int
    harmless_count: int


def build_teacher_safe_subspace(
    *,
    layer_idx: int,
    harmful_hidden: torch.Tensor,
    harmless_hidden: torch.Tensor,
    k: int = 8,
) -> SafeSubspaceResult:
    if harmful_hidden.ndim != 2 or harmless_hidden.ndim != 2:
        raise ValueError("harmful_hidden and harmless_hidden must have shape [N, d].")
    if harmful_hidden.size(1) != harmless_hidden.size(1):
        raise ValueError("harmful_hidden and harmless_hidden must share the same hidden size.")
    if harmful_hidden.size(0) == 0 or harmless_hidden.size(0) == 0:
        raise ValueError(f"Layer {layer_idx} needs both harmful and harmless samples.")

    harmful_hidden = harmful_hidden.to(dtype=torch.float32)
    harmless_hidden = harmless_hidden.to(dtype=torch.float32)
    harmless_mean = harmless_hidden.mean(dim=0)
    harmful_mean = harmful_hidden.mean(dim=0)
    mean_diff = harmful_mean - harmless_mean
    delta = harmful_hidden - harmless_mean

    _, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = max(1, min(int(k), int(vh.size(0))))
    basis = vh[:rank].T.contiguous()
    energy = singular_values.pow(2)
    total_energy = energy.sum().clamp_min(1e-12)
    explained_ratio_topk = energy[:rank] / total_energy

    return SafeSubspaceResult(
        layer_idx=layer_idx,
        k=rank,
        basis=basis,
        singular_values=singular_values,
        mean_diff=mean_diff,
        explained_ratio_topk=explained_ratio_topk,
        harmful_count=int(harmful_hidden.size(0)),
        harmless_count=int(harmless_hidden.size(0)),
    )
