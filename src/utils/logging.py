from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from .io import ensure_dir


def setup_stage_logger(stage_name: str, log_dir: str | Path) -> tuple[logging.Logger, Path]:
    target_dir = ensure_dir(log_dir)
    log_path = target_dir / f"{stage_name}.log"
    logger_name = f"safety_pipeline.{stage_name}.{log_path.resolve()}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger, log_path

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def log_kv(logger: logging.Logger, event: str, **payload: Any) -> None:
    if payload:
        logger.info("%s | %s", event, json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    logger.info("%s", event)
