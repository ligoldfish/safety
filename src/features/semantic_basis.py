from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SemanticBasisResult:
    vocab_size: int
    hidden_size: int
    basis: torch.Tensor
    token_ids: torch.Tensor
    normalized: bool
    gram_condition_number: float


def _resolve_storage_dtype(storage_dtype: str) -> torch.dtype:
    if not hasattr(torch, storage_dtype):
        raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
    return getattr(torch, storage_dtype)


@torch.no_grad()
def build_semantic_basis_from_lm_head(
    lm_head_weight: torch.Tensor,
    *,
    chunk_size: int = 4096,
    storage_dtype: str = "float16",
    eps: float = 1e-12,
) -> SemanticBasisResult:
    if lm_head_weight.ndim != 2:
        raise ValueError("lm_head_weight must have shape [vocab_size, hidden_size].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    weight = lm_head_weight.detach().cpu()
    vocab_size, hidden_size = weight.shape
    storage = _resolve_storage_dtype(storage_dtype)

    gram = torch.zeros((hidden_size, hidden_size), dtype=torch.float32)
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        chunk = weight[start:end].to(dtype=torch.float32)
        gram += chunk.T @ chunk

    inv_gram = torch.linalg.pinv(gram)
    singular_values = torch.linalg.svdvals(gram)
    cond = float((singular_values[0] / singular_values[-1].clamp_min(eps)).item())

    basis = torch.empty((vocab_size, hidden_size), dtype=storage)
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        chunk = weight[start:end].to(dtype=torch.float32)
        basis_chunk = chunk @ inv_gram
        basis_chunk = basis_chunk / basis_chunk.norm(dim=1, keepdim=True).clamp_min(eps)
        basis[start:end] = basis_chunk.to(dtype=storage)

    return SemanticBasisResult(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        basis=basis,
        token_ids=torch.arange(vocab_size, dtype=torch.long),
        normalized=True,
        gram_condition_number=cond,
    )
