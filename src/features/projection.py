from __future__ import annotations

import torch


def project_coeff(H: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    if H.ndim != 2 or U.ndim != 2:
        raise ValueError("H and U must both be rank-2 tensors.")
    return H @ U


def project_to_subspace(H: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    coeff = project_coeff(H, U)
    return coeff @ U.T


def residual_norm_ratio(H: torch.Tensor, H_safe: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if H.shape != H_safe.shape:
        raise ValueError("H and H_safe must have the same shape.")
    return torch.linalg.norm(H_safe, dim=-1) / (torch.linalg.norm(H, dim=-1) + eps)
