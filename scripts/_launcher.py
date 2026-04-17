from __future__ import annotations

import runpy
import sys
from pathlib import Path


def launch_neighbor(script_name: str, default_config: str) -> None:
    script_path = Path(__file__).resolve().with_name(script_name)
    forwarded_args = list(sys.argv[1:])
    if "--config" not in forwarded_args:
        forwarded_args = ["--config", default_config] + forwarded_args

    previous_argv = sys.argv
    sys.argv = [str(script_path)] + forwarded_args
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = previous_argv
