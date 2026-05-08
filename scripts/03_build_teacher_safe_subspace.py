from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.subspace import build_teacher_safe_subspace
from src.phase_b.hidden_states import load_hidden_state_split
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build teacher safe subspaces for the selected key layers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="teacher_alignment",
        help="Teacher hidden-state split directory under output_root/hidden_states.",
    )
    parser.add_argument(
        "--key-layers-path",
        type=str,
        default="",
        help="Optional explicit path to teacher_key_layers.json.",
    )
    parser.add_argument(
        "--train-max-samples-per-label",
        type=int,
        default=0,
        help="Optional cap per label on the teacher alignment split.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Subspace rank k. The proposal fixes this to 8 for stage C.",
    )
    parser.add_argument(
        "--balance-polarities",
        dest="balance_polarities",
        action="store_true",
        default=True,
        help=(
            "(default ON) Downsample the larger polarity to match the smaller "
            "before the SVD that builds the safe subspace. Prevents corpus "
            "imbalance (e.g. BeaverTails ~1.36:1) from biasing the subspace "
            "toward the refusal direction."
        ),
    )
    parser.add_argument(
        "--no-balance-polarities",
        dest="balance_polarities",
        action="store_false",
        help="Disable pre-SVD polarity balancing (keeps the historical imbalance).",
    )
    return parser.parse_args()


def _load_key_layers(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key_layers = [int(layer_idx) for layer_idx in payload["key_layers"]]
    if not key_layers:
        raise ValueError(f"No key layers found in: {path}")
    return key_layers


def _balance_polarities(
    harmful_mask: torch.Tensor,
    harmless_mask: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Downsample the larger polarity to match the smaller. Pure on input masks.

    Uses a dedicated ``torch.Generator`` so the global RNG is unaffected.
    Returns (new_harmful_mask, new_harmless_mask, info_dict).
    """

    n_harmful = int(harmful_mask.sum().item())
    n_harmless = int(harmless_mask.sum().item())
    if n_harmful == 0 or n_harmless == 0:
        return harmful_mask, harmless_mask, {"applied": False, "reason": "empty_polarity"}
    keep = min(n_harmful, n_harmless)
    generator = torch.Generator().manual_seed(int(seed))
    new_harmful = harmful_mask
    new_harmless = harmless_mask
    if n_harmful > keep:
        idx = torch.nonzero(harmful_mask, as_tuple=False).flatten()
        perm = torch.randperm(idx.numel(), generator=generator)[: idx.numel() - keep]
        new_harmful = harmful_mask.clone()
        new_harmful[idx[perm]] = False
    if n_harmless > keep:
        idx = torch.nonzero(harmless_mask, as_tuple=False).flatten()
        perm = torch.randperm(idx.numel(), generator=generator)[: idx.numel() - keep]
        new_harmless = harmless_mask.clone()
        new_harmless[idx[perm]] = False
    return new_harmful, new_harmless, {
        "applied": True,
        "kept_per_polarity": keep,
        "harmful_dropped": n_harmful - keep,
        "harmless_dropped": n_harmless - keep,
    }


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)

    output_root = ensure_dir(Path(cfg.extraction.output_root) / "safe_subspaces")
    logger, log_path = setup_stage_logger("03_build_teacher_safe_subspace", Path(cfg.extraction.output_root) / "logs")
    hidden_root = Path(cfg.extraction.output_root) / "hidden_states"
    key_layers_path = (
        Path(args.key_layers_path).resolve()
        if args.key_layers_path
        else Path(cfg.extraction.output_root) / "layer_analysis" / "teacher_key_layers.json"
    )
    key_layers = _load_key_layers(key_layers_path)

    train_split = load_hidden_state_split(
        hidden_root / args.train_dir,
        max_samples_per_label=args.train_max_samples_per_label,
        selected_layers=key_layers,
    )
    harmful_mask = torch.tensor([label == "harmful" for label in train_split.labels], dtype=torch.bool)
    harmless_mask = torch.tensor([label == "harmless" for label in train_split.labels], dtype=torch.bool)
    log_kv(
        logger,
        "safe_subspace_setup",
        config_path=str(Path(args.config).resolve()),
        train_split_dir=train_split.split_dir,
        train_label_counts=train_split.label_counts(),
        key_layers=key_layers,
        requested_rank=int(args.rank),
        balance_polarities=bool(args.balance_polarities),
        log_path=str(log_path),
    )

    if args.balance_polarities:
        harmful_mask, harmless_mask, balance_info = _balance_polarities(
            harmful_mask, harmless_mask, seed=cfg.seed,
        )
        log_kv(logger, "subspace_balance_applied", **balance_info)

    n_harmful = int(harmful_mask.sum().item())
    n_harmless = int(harmless_mask.sum().item())
    if n_harmless > 0:
        ratio = n_harmful / n_harmless
        if not (0.77 <= ratio <= 1.3):
            log_kv(
                logger,
                "subspace_label_imbalance_warning",
                harmful_count=n_harmful,
                harmless_count=n_harmless,
                ratio_harmful_over_harmless=ratio,
                note=(
                    "Heavy harmful/harmless imbalance biases the safe subspace "
                    "toward the majority polarity. For BeaverTails this often "
                    "indicates is_safe labels RESPONSE safety, not PROMPT "
                    "harmlessness; consider builder dedup_prompts=True or "
                    "prompt-level filtering."
                ),
            )

    generated_files: List[str] = []
    for layer_idx in key_layers:
        result = build_teacher_safe_subspace(
            layer_idx=layer_idx,
            harmful_hidden=train_split.layer_tensors[layer_idx][harmful_mask],
            harmless_hidden=train_split.layer_tensors[layer_idx][harmless_mask],
            k=args.rank,
        )
        layer_path = output_root / f"teacher_safe_subspace_layer_{layer_idx:02d}.pt"
        torch.save(
            {
                "layer_idx": result.layer_idx,
                "k": result.k,
                # basis: [d, k]. Orthonormal columns spanning the "delta safety
                # subspace" (top-k right singular vectors of Delta_l =
                # harmful_hidden - harmless_mean). It is NOT a scalar multiple
                # of mean_diff; it is the subspace of harmful-vs-harmless
                # differences.
                "basis": result.basis,
                "singular_values": result.singular_values,
                # mean_diff: d-vector. harmful_mean - harmless_mean (the r_l
                # direction from 方案详述 §3.2). Useful as a sanity-check
                # target direction; not itself the training target.
                "mean_diff": result.mean_diff,
                "harmful_mean": result.harmful_mean,
                "harmless_mean": result.harmless_mean,
                "explained_ratio_top8": result.explained_ratio_topk,
                "harmful_count": result.harmful_count,
                "harmless_count": result.harmless_count,
                "subspace_definition": (
                    "basis = top-{k} right singular vectors of "
                    "Delta_l where Delta_l[i] = h_harmful[i] - mean(h_harmless)"
                ).format(k=result.k),
            },
            layer_path,
        )
        generated_files.append(str(layer_path))
        log_kv(
            logger,
            "safe_subspace_saved",
            layer_idx=int(layer_idx),
            actual_rank=int(result.k),
            basis_shape=list(result.basis.shape),
            harmful_count=int(result.harmful_count),
            harmless_count=int(result.harmless_count),
            output_path=str(layer_path),
        )

    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "teacher_model": cfg.teacher.name,
            "train_split_dir": train_split.split_dir,
            "train_label_counts": train_split.label_counts(),
            "key_layers_path": str(key_layers_path),
            "key_layers": key_layers,
            "rank": args.rank,
            "files": generated_files,
        },
    )
    log_kv(logger, "safe_subspace_complete", output_root=str(output_root), files=generated_files)

    print(json.dumps({"key_layers": key_layers, "rank": args.rank}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
