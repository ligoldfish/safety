#!/usr/bin/env python3
"""Run the complete ablation campaign in six user-visible ModelMate waves."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.modelmate_pool import CAMPAIGN_WAVES, ROUND_SPECS


DEFAULT_MODEL_ROOT = Path("/opt/dpcvol/models/safetytransfer")
DEFAULT_DATA_ROOT = Path("/opt/dpcvol/datasets/safetytransfer")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "ablation-outputs" / "iclr-886760f"
DEFAULT_ASSET_MANIFEST = PROJECT_ROOT / "configs" / "ablations" / "assets.modelmate.template.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one of six fail-closed ModelMate campaign waves."
    )
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--wave", choices=tuple(CAMPAIGN_WAVES), default="canary")
    choice.add_argument(
        "--round",
        choices=tuple(ROUND_SPECS),
        help="Backward-compatible single internal round; prefer --wave.",
    )
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--asset-manifest", default=str(DEFAULT_ASSET_MANIFEST))
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--device-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--logical-shards", type=int, default=16)
    parser.add_argument("--launch-stagger-seconds", type=float, default=15.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-device-check", action="store_true")
    parser.add_argument("--checkpoint_url", "--checkpoint-url", default="", help=argparse.SUPPRESS)
    parser.add_argument("--data_url", "--data-url", default="", help=argparse.SUPPRESS)
    return parser


def _common_pool_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--model-root",
        str(args.model_root),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--asset-manifest",
        str(args.asset_manifest),
        "--devices",
        str(args.devices),
        "--device-ids",
        str(args.device_ids),
        "--logical-shards",
        str(args.logical_shards),
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
    ]
    if args.preflight_only:
        values.append("--preflight-only")
    if args.dry_run:
        values.append("--dry-run")
    if args.skip_device_check:
        values.append("--skip-device-check")
    return values


def build_campaign_commands(args: argparse.Namespace) -> tuple[tuple[str, ...], ...]:
    pool = PROJECT_ROOT / "scripts" / "35_modelmate_8card_pool.py"
    gate = PROJECT_ROOT / "scripts" / "36_modelmate_ablation_final_gate.py"
    if args.round:
        return (
            (
                sys.executable,
                str(pool),
                "--round",
                str(args.round),
                *_common_pool_args(args),
            ),
        )
    wave = CAMPAIGN_WAVES[args.wave]
    commands = [
        (
            sys.executable,
            str(pool),
            "--round",
            round_name,
            *_common_pool_args(args),
        )
        for round_name in wave.rounds
    ]
    if wave.final_gate:
        commands.append(
            (
                sys.executable,
                str(gate),
                "--output-root",
                str(args.output_root),
            )
        )
    return tuple(commands)


def run_campaign(
    args: argparse.Namespace,
    *,
    runner=subprocess.run,
) -> tuple[int, dict[str, object]]:
    wave_name = f"round-{args.round}" if args.round else str(args.wave)
    commands = build_campaign_commands(args)
    results: list[dict[str, object]] = []
    exit_code = 0
    for index, command in enumerate(commands):
        label = (
            Path(command[1]).stem
            if "--round" not in command
            else command[command.index("--round") + 1]
        )
        started = _utc_now()
        print(
            json.dumps(
                {
                    "event": "campaign_stage_start",
                    "wave": wave_name,
                    "stage_index": index,
                    "stage": label,
                    "created_at": started,
                    "command": list(command),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        completed = runner(list(command), cwd=str(PROJECT_ROOT), check=False)
        returncode = int(completed.returncode)
        results.append(
            {
                "stage_index": index,
                "stage": label,
                "started_at": started,
                "finished_at": _utc_now(),
                "returncode": returncode,
            }
        )
        if returncode:
            exit_code = returncode
            break
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "wave": wave_name,
        "status": "READY" if exit_code == 0 and len(results) == len(commands) else "FAILED",
        "preflight_only": bool(args.preflight_only),
        "dry_run": bool(args.dry_run),
        "planned_stage_count": len(commands),
        "executed_stage_count": len(results),
        "results": results,
    }
    return exit_code, payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.devices <= 0 or args.logical_shards <= 0:
        raise ValueError("--devices and --logical-shards must be positive")
    if args.launch_stagger_seconds < 0:
        raise ValueError("--launch-stagger-seconds cannot be negative")
    if args.skip_device_check and not args.dry_run:
        raise ValueError("--skip-device-check is allowed only with --dry-run")
    exit_code, payload = run_campaign(args)
    wave_name = f"round-{args.round}" if args.round else str(args.wave)
    summary = Path(args.output_root).expanduser().resolve() / "jobs" / f"campaign-{wave_name}" / "wave-summary.json"
    _atomic_json(summary, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3)
