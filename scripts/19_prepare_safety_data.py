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

from src.baselines.config import load_sft_config
from src.data.safety_datasets import (
    SafetyDatasetSpec,
    materialize_safety_train_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a safety SFT JSONL training file from upstream corpora."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a baseline SFT YAML whose data.safety_dataset block "
        "names the dataset to build.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the JSONL even when it already exists.",
    )
    return parser.parse_args()


def _build_spec(config_path: Path, force_rebuild: bool) -> SafetyDatasetSpec:
    cfg = load_sft_config(config_path)
    safety_cfg = cfg.data.safety_dataset
    if not safety_cfg.name:
        raise ValueError(
            f"{config_path} does not declare data.safety_dataset.name; "
            "cannot determine which safety dataset to build."
        )
    return SafetyDatasetSpec(
        name=safety_cfg.name,
        output_path=cfg.data.train_split,
        force_rebuild=force_rebuild or safety_cfg.force_rebuild,
        cache_dir=safety_cfg.cache_dir or None,
        source_name=safety_cfg.source_name or None,
        split=safety_cfg.split or None,
        sources=list(safety_cfg.sources) or None,
        repo_or_data_path=safety_cfg.repo_or_data_path or None,
        file_name=safety_cfg.file_name or None,
        refusal_template=safety_cfg.refusal_template or None,
        system_prompt=safety_cfg.system_prompt or None,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    spec = _build_spec(config_path, args.force_rebuild)
    output_path = materialize_safety_train_dataset(spec)
    summary = {
        "config_path": str(config_path),
        "dataset_name": spec.name,
        "output_path": str(output_path),
        "force_rebuild": spec.force_rebuild,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
