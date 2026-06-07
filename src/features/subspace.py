from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SafeSubspaceResult:
    layer_idx: int
    k: int
    basis: torch.Tensor
    singular_values: torch.Tensor
    mean_diff: torch.Tensor  # harmful_mean - harmless_mean (per 方案详述 r_l)
    explained_ratio_topk: torch.Tensor
    harmful_count: int
    harmless_count: int
    harmful_mean: torch.Tensor
    harmless_mean: torch.Tensor


def build_teacher_safe_subspace(
    *,
    layer_idx: int,
    harmful_hidden: torch.Tensor,
    harmless_hidden: torch.Tensor,
    k: int = 16,
    energy_threshold: float | None = None,
    rank_cap: int = 64,
    normalize_hidden: bool = False,
    eps: float = 1e-12,
) -> SafeSubspaceResult:
    """Build the teacher safe subspace for a single key layer.

    Semantics (方案详述 §4.1-§4.3):

    - ``harmless_mean = mean(h^{harmless}_l)``
    - ``harmful_mean  = mean(h^{harmful}_l)``
    - ``mean_diff = harmful_mean - harmless_mean``  (the "r_l" direction)
    - ``Delta_l = harmful_hidden - harmless_mean``  (each row is h^{harmful}_i - mu^{harmless}_l)
    - ``Delta_l = U diag(sigma) V^T`` via thin SVD.
    - ``basis`` holds the top-k right singular vectors of ``Delta_l`` (columns
      of ``V``), so ``basis.T @ basis = I_k`` (orthonormal). It defines the
      *delta safety subspace*, i.e., the subspace of harmful-vs-harmless
      differences — it is NOT "the pure mean_diff direction".

    Rank selection:

    - If ``energy_threshold`` (tau) is ``None``, the rank is the fixed ``k``
      (clamped to the available rank).
    - If ``energy_threshold`` is set, the rank is a per-layer *effective rank*:
      the smallest ``r`` whose cumulative singular-value energy reaches ``tau``
      (``min{ r : cumsum(sigma^2)[:r] / sum(sigma^2) >= tau }``), clamped to
      ``[1, rank_cap]``. This is a PAN-inspired energy-threshold criterion;
      note PAN defines effective rank on ``SVD(W - I)`` (the fitted residual
      map), whereas here it is applied to the contrast matrix ``Delta_l`` — so
      it is "PAN-inspired", not PAN §4 verbatim. Early/low-rank layers shrink
      to 1-2 dims automatically; mid-late layers grow toward ``rank_cap``.

    We additionally persist ``harmful_mean`` and ``harmless_mean`` so that
    downstream scripts (``06`` projection, ``07`` semantic decomposition) can
    be audited against the subspace origin without ambiguity.
    """
    if harmful_hidden.ndim != 2 or harmless_hidden.ndim != 2:
        raise ValueError("harmful_hidden and harmless_hidden must have shape [N, d].")
    if harmful_hidden.size(1) != harmless_hidden.size(1):
        raise ValueError("harmful_hidden and harmless_hidden must share the same hidden size.")
    if harmful_hidden.size(0) == 0 or harmless_hidden.size(0) == 0:
        raise ValueError(f"Layer {layer_idx} needs both harmful and harmless samples.")

    harmful_hidden = harmful_hidden.to(dtype=torch.float32)
    harmless_hidden = harmless_hidden.to(dtype=torch.float32)
    if normalize_hidden:
        harmful_hidden = harmful_hidden / harmful_hidden.norm(dim=1, keepdim=True).clamp_min(eps)
        harmless_hidden = harmless_hidden / harmless_hidden.norm(dim=1, keepdim=True).clamp_min(eps)
    harmless_mean = harmless_hidden.mean(dim=0)
    harmful_mean = harmful_hidden.mean(dim=0)
    mean_diff = harmful_mean - harmless_mean
    delta = harmful_hidden - harmless_mean

    _, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
    max_rank = int(vh.size(0))
    energy = singular_values.pow(2)
    total_energy = energy.sum().clamp_min(1e-12)
    if energy_threshold is not None:
        # Per-layer effective rank: smallest r whose cumulative energy reaches tau.
        cum_ratio = torch.cumsum(energy, dim=0) / total_energy
        reached = int((cum_ratio < float(energy_threshold)).sum().item()) + 1
        cap = max(1, min(int(rank_cap), max_rank))
        rank = max(1, min(reached, cap))
    else:
        rank = max(1, min(int(k), max_rank))
    basis = vh[:rank].T.contiguous()
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
        harmful_mean=harmful_mean,
        harmless_mean=harmless_mean,
    )
