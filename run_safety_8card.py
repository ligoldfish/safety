#!/usr/bin/env python3
"""Stable ModelMate startup shim for the allowlisted 8-card ablation pool."""

import os
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
BOOT = APP_ROOT / "boot_safety_8card.sh"

if not BOOT.is_file():
    raise SystemExit(f"missing startup script: {BOOT}")

os.execv(
    "/usr/bin/env",
    ["env", "bash", str(BOOT), *sys.argv[1:]],
)
