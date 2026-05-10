from __future__ import annotations

import torch


@torch.no_grad()
def topk_semantic_coefficients(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
    *,
    top_k: int = 256,
    vocab_chunk_size: int = 4096,
    selection_mode: str = "abs",
    valid_token_mask: torch.Tensor | None = None,
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
    mode = str(selection_mode).strip().lower()
    if mode not in {"abs", "positive", "negative"}:
        raise ValueError("selection_mode must be one of: abs, positive, negative.")

    hidden = hidden_states.to(dtype=torch.float32)
    hidden = hidden / hidden.norm(dim=1, keepdim=True).clamp_min(eps)
    batch_size = hidden.size(0)
    vocab_size = basis.size(0)
    if valid_token_mask is not None:
        if valid_token_mask.ndim != 1 or valid_token_mask.numel() != vocab_size:
            raise ValueError("valid_token_mask must have shape [V].")
        valid_token_mask = valid_token_mask.to(dtype=torch.bool, device="cpu")
        valid_count = int(valid_token_mask.sum().item())
        if valid_count <= 0:
            raise ValueError("valid_token_mask excludes every token.")
        top_k = min(int(top_k), valid_count)

    best_values = torch.empty((batch_size, 0), dtype=torch.float32)
    best_indices = torch.empty((batch_size, 0), dtype=torch.long)
    best_priorities = torch.empty((batch_size, 0), dtype=torch.float32)

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(vocab_size, start + vocab_chunk_size)
        basis_chunk = basis[start:end].to(dtype=torch.float32)
        scores = hidden @ basis_chunk.T
        if mode == "abs":
            priorities = scores.abs()
        elif mode == "positive":
            priorities = scores.masked_fill(scores <= 0, float("-inf"))
        else:
            priorities = (-scores).masked_fill(scores >= 0, float("-inf"))
        if valid_token_mask is not None:
            chunk_mask = valid_token_mask[start:end].unsqueeze(0).expand(batch_size, -1)
            priorities = priorities.masked_fill(~chunk_mask, float("-inf"))
        chunk_indices = torch.arange(start, end, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        if best_values.numel() == 0:
            merged_values = scores
            merged_indices = chunk_indices
            merged_priorities = priorities
        else:
            merged_values = torch.cat([best_values, scores], dim=1)
            merged_indices = torch.cat([best_indices, chunk_indices], dim=1)
            merged_priorities = torch.cat([best_priorities, priorities], dim=1)

        keep_k = min(top_k, merged_values.size(1))
        top_positions = torch.topk(merged_priorities, k=keep_k, dim=1).indices
        best_values = merged_values.gather(1, top_positions)
        best_indices = merged_indices.gather(1, top_positions)
        best_priorities = merged_priorities.gather(1, top_positions)

    order = torch.argsort(best_priorities, dim=1, descending=True)
    best_values = best_values.gather(1, order)
    best_indices = best_indices.gather(1, order)
    return best_indices.contiguous(), best_values.contiguous()
