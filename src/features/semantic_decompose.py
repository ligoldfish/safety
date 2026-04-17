from __future__ import annotations

import torch


@torch.no_grad()
def topk_semantic_coefficients(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
    *,
    top_k: int = 256,
    vocab_chunk_size: int = 4096,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [B, d].")
    if basis.ndim != 2:
        raise ValueError("basis must have shape [V, d].")
    if hidden_states.size(1) != basis.size(1):
        raise ValueError("hidden_states and basis must share the same hidden size.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive.")

    hidden = hidden_states.to(dtype=torch.float32)
    hidden = hidden / hidden.norm(dim=1, keepdim=True).clamp_min(eps)
    batch_size = hidden.size(0)
    vocab_size = basis.size(0)

    best_values = torch.empty((batch_size, 0), dtype=torch.float32)
    best_indices = torch.empty((batch_size, 0), dtype=torch.long)

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(vocab_size, start + vocab_chunk_size)
        basis_chunk = basis[start:end].to(dtype=torch.float32)
        scores = hidden @ basis_chunk.T
        chunk_indices = torch.arange(start, end, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        if best_values.numel() == 0:
            merged_values = scores
            merged_indices = chunk_indices
        else:
            merged_values = torch.cat([best_values, scores], dim=1)
            merged_indices = torch.cat([best_indices, chunk_indices], dim=1)

        keep_k = min(top_k, merged_values.size(1))
        top_positions = torch.topk(merged_values.abs(), k=keep_k, dim=1).indices
        best_values = merged_values.gather(1, top_positions)
        best_indices = merged_indices.gather(1, top_positions)

    order = torch.argsort(best_values.abs(), dim=1, descending=True)
    best_values = best_values.gather(1, order)
    best_indices = best_indices.gather(1, order)
    return best_indices.contiguous(), best_values.contiguous()
