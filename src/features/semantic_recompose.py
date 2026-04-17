from __future__ import annotations

import torch


@torch.no_grad()
def recompose_from_sparse_coeffs(
    basis: torch.Tensor,
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError("basis must have shape [V, d].")
    if top_indices.ndim != 2 or top_values.ndim != 2:
        raise ValueError("top_indices and top_values must have shape [B, K].")
    if top_indices.shape != top_values.shape:
        raise ValueError("top_indices and top_values must have the same shape.")

    gathered = basis.index_select(0, top_indices.reshape(-1).to(dtype=torch.long))
    gathered = gathered.view(top_indices.size(0), top_indices.size(1), basis.size(1)).to(dtype=torch.float32)
    weights = top_values.to(dtype=torch.float32).unsqueeze(-1)
    return torch.sum(gathered * weights, dim=1)
