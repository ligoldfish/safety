from __future__ import annotations

import torch


def _validate_basis(basis: torch.Tensor) -> tuple[int, int]:
    if basis.ndim != 2:
        raise ValueError("subspace basis must have shape [hidden, rank]")
    hidden, rank = basis.shape
    if hidden <= 0 or rank <= 0 or rank > hidden:
        raise ValueError("subspace basis rank must satisfy 1 <= rank <= hidden")
    return int(hidden), int(rank)


def build_control_subspace(
    learned_basis: torch.Tensor,
    *,
    mode: str,
    seed: int,
) -> torch.Tensor:
    hidden, rank = _validate_basis(learned_basis)
    normalized = str(mode).strip().lower()
    if normalized == "learned":
        return learned_basis
    if normalized == "random_orthogonal":
        if type(seed) is not int:
            raise ValueError("random_orthogonal requires an integer seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        sample = torch.randn(hidden, rank, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(sample, mode="reduced")
        signs = torch.where(torch.diag(r) < 0, -torch.ones(rank, dtype=q.dtype), torch.ones(rank, dtype=q.dtype))
        return (q * signs).to(device=learned_basis.device, dtype=learned_basis.dtype)
    if normalized == "none":
        return learned_basis.new_empty((hidden, 0))
    raise ValueError(f"unsupported subspace mode: {mode}")


def project_with_mode(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [samples, hidden]")
    normalized = str(mode).strip().lower()
    if normalized == "none":
        return hidden_states
    _validate_basis(basis)
    if hidden_states.size(1) != basis.size(0):
        raise ValueError("hidden_states and basis hidden dimensions must match")
    if normalized not in {"learned", "random_orthogonal"}:
        raise ValueError(f"unsupported subspace mode: {mode}")
    basis = basis.to(device=hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states @ basis @ basis.T
