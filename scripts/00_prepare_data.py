from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_io import prepare_phase1_datasets
from src.utils.config import load_phase1_config
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Pan-style Qwen phase-A datasets.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_7b_to_15b_phase1.yaml",
        help="Path to the phase-A YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)
    logger, log_path = setup_stage_logger("00_prepare_data", Path(cfg.dataset.metadata_dir) / "logs")
    log_kv(
        logger,
        "prepare_data_setup",
        config_path=str(Path(args.config).resolve()),
        seed=int(cfg.seed),
        pan_repo_dir=cfg.dataset.pan_repo_dir,
        raw_dir=cfg.dataset.raw_dir,
        processed_dir=cfg.dataset.processed_dir,
        metadata_dir=cfg.dataset.metadata_dir,
        pan_train_size=int(cfg.dataset.pan_train_size),
        alignment_size=int(cfg.dataset.alignment_size),
        analysis_val_size=int(cfg.dataset.analysis_val_size),
        sanity_test_size_per_label=int(cfg.dataset.sanity_test_size_per_label),
        log_path=str(log_path),
    )
    outputs = prepare_phase1_datasets(
        pan_repo_dir=cfg.dataset.pan_repo_dir,
        raw_dir=cfg.dataset.raw_dir,
        processed_dir=cfg.dataset.processed_dir,
        metadata_dir=cfg.dataset.metadata_dir,
        exposure_size=cfg.dataset.exposure_size,
        pan_test_size_per_type=cfg.dataset.pan_test_size_per_type,
        pan_train_size=cfg.dataset.pan_train_size,
        alignment_size=cfg.dataset.alignment_size,
        analysis_val_size=cfg.dataset.analysis_val_size,
        sanity_test_size_per_label=cfg.dataset.sanity_test_size_per_label,
        max_prompt_chars=cfg.dataset.max_prompt_chars,
        seed=cfg.seed,
        system_prompt=cfg.dataset.system_prompt,
        include_system_prompt=cfg.dataset.include_system_prompt,
    )
    log_kv(logger, "prepare_data_complete", **outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
