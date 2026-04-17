from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Phase1ModelConfig, load_phase1_config
from src.utils.io import chunked, ensure_dir, read_jsonl
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


SPLIT_TO_FILENAME = {
    "alignment": "alignment_set.jsonl",
    "analysis_val": "analysis_val_set.jsonl",
    "pan_test": "pan_test_set.jsonl",
    "sanity_test": "sanity_test_set.jsonl",
    "pan_train": "pan_train_set.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first-generated-token hidden states for Qwen phase-A splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_7b_to_15b_phase1.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=sorted(SPLIT_TO_FILENAME.keys()),
        help="Prepared dataset split to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["teacher", "student"],
        help="Which model config to use.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for smoke testing. Zero means full split.",
    )
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=0,
        help="Optional cap per label for balanced smoke testing. Zero means no per-label cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files even when skip_existing is enabled.",
    )
    return parser.parse_args()


def _select_model_config(cfg, model_key: str) -> Phase1ModelConfig:
    return cfg.teacher if model_key == "teacher" else cfg.student


def _storage_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported storage dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def _load_split_records(processed_dir: Path, split: str, max_samples: int) -> List[Dict[str, Any]]:
    filename = SPLIT_TO_FILENAME[split]
    records = read_jsonl(processed_dir / filename)
    if max_samples > 0:
        records = records[:max_samples]
    return records


def _apply_per_label_limit(
    records: List[Dict[str, Any]],
    max_samples_per_label: int,
) -> List[Dict[str, Any]]:
    if max_samples_per_label <= 0:
        return records

    kept: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        label = str(record["label"])
        if counts[label] >= max_samples_per_label:
            continue
        counts[label] += 1
        kept.append(record)
    return kept


def _save_shard(
    shard_records: List[Dict[str, Any]],
    shard_idx: int,
    shard_dir: Path,
    model_cfg: Phase1ModelConfig,
    gather_fn: Any,
    tokenizer: Any,
    model: Any,
    max_length: int,
    batch_size: int,
    storage_dtype: torch.dtype,
) -> None:
    hidden_by_layer: Dict[str, List[torch.Tensor]] = {}
    sample_ids: List[str] = []
    labels: List[str] = []
    source_datasets: List[str] = []
    prompt_token_counts: List[int] = []

    for batch in chunked(shard_records, batch_size):
        messages_batch = [record["messages"] for record in batch]
        layer_hiddens, _, last_positions = gather_fn(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            max_length=max_length,
        )
        if not hidden_by_layer:
            hidden_by_layer = {str(layer_idx): [] for layer_idx in range(len(layer_hiddens))}

        for layer_idx, tensor in enumerate(layer_hiddens):
            hidden_by_layer[str(layer_idx)].append(tensor.to(dtype=storage_dtype))

        sample_ids.extend(record["id"] for record in batch)
        labels.extend(record["label"] for record in batch)
        source_datasets.extend(record["source_dataset"] for record in batch)
        prompt_token_counts.extend((last_positions + 1).tolist())

    payload = {
        "model_name": model_cfg.name,
        "model_path": model_cfg.path,
        "feature_type": "first_generated_token_hidden_state",
        "layer_indices": list(range(len(hidden_by_layer))),
        "sample_ids": sample_ids,
        "labels": labels,
        "source_datasets": source_datasets,
        "prompt_token_counts": prompt_token_counts,
        "hidden_by_layer": {
            layer_idx: torch.cat(tensors, dim=0)
            for layer_idx, tensors in hidden_by_layer.items()
        },
    }
    torch.save(payload, shard_dir / f"part_{shard_idx:03d}.pt")


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for hidden-state extraction. Install torch in the active Python environment first."
        )

    from src.features.first_gen_token import gather_first_generated_token_representations
    from src.models.hf_loader import load_hf_model

    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)
    logger, log_path = setup_stage_logger("01_extract_hidden_states", Path(cfg.extraction.output_root) / "logs")

    model_cfg = _select_model_config(cfg, args.model)
    processed_dir = Path(cfg.dataset.processed_dir)
    records = _load_split_records(processed_dir, args.split, args.max_samples)
    records = _apply_per_label_limit(records, args.max_samples_per_label)
    if not records:
        raise ValueError(f"No records found for split={args.split}. Run 00_prepare_data.py first.")
    log_kv(
        logger,
        "extract_setup",
        config_path=str(Path(args.config).resolve()),
        split=args.split,
        model=args.model,
        model_name=model_cfg.name,
        runtime_backend=model_cfg.runtime_backend,
        runtime_device=model_cfg.runtime_device,
        num_records=len(records),
        label_counts=dict(Counter(str(record["label"]) for record in records)),
        shard_size=int(cfg.extraction.shard_size),
        batch_size=int(cfg.extraction.batch_size),
        max_samples=int(args.max_samples),
        max_samples_per_label=int(args.max_samples_per_label),
        log_path=str(log_path),
    )

    tokenizer, model, model_meta = load_hf_model(
        model_path=model_cfg.path,
        device_map=model_cfg.device_map,
        torch_dtype=model_cfg.torch_dtype,
        runtime_backend=model_cfg.runtime_backend,
        runtime_device=model_cfg.runtime_device,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=model_cfg.local_files_only,
        attn_implementation=model_cfg.attn_implementation,
    )

    shard_dir = ensure_dir(
        Path(cfg.extraction.output_root)
        / "hidden_states"
        / f"{args.model}_{args.split}"
    )
    storage_dtype = _storage_dtype(cfg.extraction.storage_dtype)
    for shard_idx, shard_records in enumerate(chunked(records, cfg.extraction.shard_size)):
        shard_path = shard_dir / f"part_{shard_idx:03d}.pt"
        if shard_path.exists() and cfg.extraction.skip_existing and not args.overwrite:
            log_kv(logger, "shard_skipped", shard_idx=shard_idx, shard_path=str(shard_path))
            continue
        _save_shard(
            shard_records=list(shard_records),
            shard_idx=shard_idx,
            shard_dir=shard_dir,
            model_cfg=model_cfg,
            gather_fn=gather_first_generated_token_representations,
            tokenizer=tokenizer,
            model=model,
            max_length=cfg.extraction.max_length,
            batch_size=cfg.extraction.batch_size,
            storage_dtype=storage_dtype,
        )
        log_kv(
            logger,
            "shard_saved",
            shard_idx=shard_idx,
            shard_path=str(shard_path),
            num_records=len(shard_records),
        )

    summary = {
        "model_name": model_cfg.name,
        "model_path": model_cfg.path,
        "split": args.split,
        "num_samples": len(records),
        "num_layers": model_meta["num_layers"],
        "hidden_size": model_meta["hidden_size"],
        "storage_dtype": cfg.extraction.storage_dtype,
        "shard_size": cfg.extraction.shard_size,
        "batch_size": cfg.extraction.batch_size,
        "max_samples_per_label": args.max_samples_per_label,
    }
    summary_path = shard_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_kv(logger, "extract_complete", manifest_path=str(summary_path), **summary)
    print(summary_path)


if __name__ == "__main__":
    main()
