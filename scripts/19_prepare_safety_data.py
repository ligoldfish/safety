"""Materialize safety SFT JSONL training files from upstream corpora.

This is the only entry point that touches HuggingFace ``load_dataset`` for
safety SFT training. The new safety baselines (``tulu3_safety``,
``safety_tuned_llamas``, ``beavertails``) all point their YAML
``data.train_split`` at the JSONL produced by this script, and the
training script ``13_train_pan_sft.py`` keeps reading from disk via
``load_records``.

Usage:
    python scripts/19_prepare_safety_data.py \
        --config configs/baseline_sft_qwen35_08b_tulu3_safety.yaml
    python scripts/19_prepare_safety_data.py \
        --config configs/baseline_sft_qwen35_08b_safety_tuned_llamas.yaml --force-rebuild

The dataset to build is inferred from ``data.safety_dataset.name`` in the
referenced config. Pass ``--force-rebuild`` to re-materialize even when the
target JSONL already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.config import (
    BaselineDistillConfig,
    BaselineSFTConfig,
    SupervisedDataConfig,
    load_distill_config,
    load_sft_config,
)
from src.data.safety_datasets import (
    SafetyDatasetSpec,
    materialize_safety_train_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a safety SFT/distill JSONL training file from upstream "
            "corpora. Auto-detects whether the config is an SFT or distill recipe."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a baseline SFT or distill YAML whose data.safety_dataset "
        "block names the dataset to build.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the JSONL even when it already exists.",
    )
    return parser.parse_args()


def _load_data_block(config_path: Path) -> tuple[SupervisedDataConfig, str]:
    """Try SFT first; fall back to distill. Both share SupervisedDataConfig."""

    last_exc: Exception | None = None
    for loader, kind in ((load_sft_config, "sft"), (load_distill_config, "distill")):
        try:
            cfg = loader(config_path)
        except Exception as exc:  # config schema mismatch — try the next loader
            last_exc = exc
            continue
        if isinstance(cfg, (BaselineSFTConfig, BaselineDistillConfig)):
            return cfg.data, kind
    raise ValueError(
        f"Could not load {config_path} as either SFT or distill config. "
        f"Last loader error: {last_exc!r}"
    )


def _build_spec(config_path: Path, force_rebuild: bool) -> tuple[SafetyDatasetSpec, str]:
    data, kind = _load_data_block(config_path)
    safety_cfg = data.safety_dataset
    if not safety_cfg.name:
        raise ValueError(
            f"{config_path} does not declare data.safety_dataset.name; "
            "cannot determine which safety dataset to build."
        )
    spec = SafetyDatasetSpec(
        name=safety_cfg.name,
        output_path=data.train_split,
        force_rebuild=force_rebuild or safety_cfg.force_rebuild,
        cache_dir=safety_cfg.cache_dir or None,
        source_name=safety_cfg.source_name or None,
        split=safety_cfg.split or None,
        sources=list(safety_cfg.sources) or None,
        repo_or_data_path=safety_cfg.repo_or_data_path or None,
        file_name=safety_cfg.file_name or None,
        refusal_template=safety_cfg.refusal_template or None,
        system_prompt=safety_cfg.system_prompt or None,
        include_harmless_contrast=bool(safety_cfg.include_harmless_contrast),
        harmless_file_name=safety_cfg.harmless_file_name or "alpaca_small.json",
        dedup_prompts=bool(safety_cfg.dedup_prompts),
    )
    return spec, kind


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    spec, kind = _build_spec(config_path, args.force_rebuild)
    output_path = materialize_safety_train_dataset(spec)
    summary = {
        "config_path": str(config_path),
        "config_kind": kind,
        "dataset_name": spec.name,
        "output_path": str(output_path),
        "force_rebuild": spec.force_rebuild,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
