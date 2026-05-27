"""Single-cell ablation orchestrator for the cross-scale safety alignment sweep.

Edits phaseF / phase1 yaml fields and/or injects extra args into the launcher's
02_analyze_teacher_layers / 03_build_teacher_safe_subspace invocations, runs
``safety-full`` for one (axis, baseline) pair, parses ``final_summary.json``,
appends a row to ``sweep_results.csv``, then restores all touched files.

Usage:
  python scripts/run_param_sweep.py \\
    --axis B1 --baseline beavertails --device npu --device-id 0 \\
    --phasef-set '{"optim.layer_loss_weight": 0.1}'

All edits are in-place + restored in a ``finally`` block so a Ctrl-C mid-run
leaves the tree clean.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_YAML = {
    ("npu", "phase1"): PROJECT_ROOT / "configs" / "qwen35_9b_to_08b_phase1_npu.yaml",
    ("npu", "phasef"): PROJECT_ROOT / "configs" / "qwen35_9b_to_08b_phaseF_npu.yaml",
    ("ppu", "phase1"): PROJECT_ROOT / "configs" / "qwen35_9b_to_08b_phase1_ppu.yaml",
    ("ppu", "phasef"): PROJECT_ROOT / "configs" / "qwen35_9b_to_08b_phaseF_ppu.yaml",
}

LAUNCHER = PROJECT_ROOT / "scripts" / "15_run_oneclick.py"
SWEEP_CSV = PROJECT_ROOT / "sweep_results.csv"

ANALYZE_ANCHOR = 'analyze_args = ["--config", str(phase1_config)]'
SUBSPACE_ANCHOR = 'subspace_args = ["--config", str(phase1_config)]'


def set_dotted(data: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def patch_yaml(path: Path, updates: dict) -> Optional[str]:
    if not updates:
        return None
    orig_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(orig_text)
    for dotted, value in updates.items():
        set_dotted(data, dotted, value)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return orig_text


def patch_launcher_phase1_args(
    analyze_extras: list[str], subspace_extras: list[str]
) -> Optional[str]:
    if not analyze_extras and not subspace_extras:
        return None
    orig_text = LAUNCHER.read_text(encoding="utf-8")
    new_text = orig_text
    if analyze_extras:
        inj = ", ".join(repr(x) for x in analyze_extras)
        replaced = new_text.replace(
            ANALYZE_ANCHOR,
            f'analyze_args = ["--config", str(phase1_config), {inj}]',
            1,
        )
        if replaced == new_text:
            raise RuntimeError(
                f"Could not find anchor for analyze_args in {LAUNCHER}"
            )
        new_text = replaced
    if subspace_extras:
        inj = ", ".join(repr(x) for x in subspace_extras)
        replaced = new_text.replace(
            SUBSPACE_ANCHOR,
            f'subspace_args = ["--config", str(phase1_config), {inj}]',
            1,
        )
        if replaced == new_text:
            raise RuntimeError(
                f"Could not find anchor for subspace_args in {LAUNCHER}"
            )
        new_text = replaced
    LAUNCHER.write_text(new_text, encoding="utf-8")
    return orig_text


def restore(path: Path, original_text: Optional[str]) -> None:
    if original_text is not None:
        path.write_text(original_text, encoding="utf-8")


def find_summary(baseline: str, device: str) -> Optional[Path]:
    if baseline == "pan":
        # PAN routes to _run_full_pipeline which writes under
        # outputs/qwen35_9b_to_08b_phase1_<device>/training/eval_suite/
        base = PROJECT_ROOT / "outputs" / f"qwen35_9b_to_08b_phase1_{device}"
        pattern = "training/eval_suite/epoch_*/final_summary.json"
    else:
        # Safety baselines route to _run_safety_full which writes under
        # outputs/safety_full_<baseline>_<device>/phase1/training/eval_suite/
        base = PROJECT_ROOT / "outputs" / f"safety_full_{baseline}_{device}"
        pattern = "phase1/training/eval_suite/epoch_*/final_summary.json"
    candidates = sorted(base.glob(pattern), key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def parse_hr_or(summary_path: Optional[Path]) -> tuple[Optional[float], Optional[float]]:
    if not summary_path or not summary_path.exists():
        return None, None
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    def walk(node: Any) -> Optional[tuple[float, float]]:
        if isinstance(node, dict):
            hr_key = next(
                (k for k in node if "harmful_refusal_rate" in k.lower()), None
            )
            or_key = next(
                (k for k in node if "harmless_over_refusal_rate" in k.lower()), None
            )
            if hr_key and or_key:
                try:
                    return float(node[hr_key]), float(node[or_key])
                except (TypeError, ValueError):
                    pass
            for v in node.values():
                got = walk(v)
                if got is not None:
                    return got
        elif isinstance(node, list):
            for item in node:
                got = walk(item)
                if got is not None:
                    return got
        return None

    hit = walk(data)
    return hit if hit else (None, None)


def safety_f1(hr: Optional[float], or_: Optional[float]) -> Optional[float]:
    if hr is None or or_ is None:
        return None
    helpful = 100.0 - or_
    denom = hr + helpful
    return 0.0 if denom == 0 else 2.0 * hr * helpful / denom


def append_csv_row(row: dict) -> None:
    new_file = not SWEEP_CSV.exists()
    with SWEEP_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description="Single-cell sweep ablation runner")
    p.add_argument("--axis", required=True, help="Axis label e.g. B1, A1, D1, G1, C1, E1, F1")
    p.add_argument(
        "--baseline",
        choices=["pan", "beavertails", "safety_tuned_llamas"],
        required=True,
    )
    p.add_argument("--device", choices=["npu", "ppu"], default="npu")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument(
        "--phasef-set",
        default="{}",
        help='JSON dict of dotted phaseF yaml fields to set, e.g. \'{"optim.layer_loss_weight": 0.1}\'',
    )
    p.add_argument(
        "--phase1-set",
        default="{}",
        help='JSON dict of dotted phase1 yaml fields',
    )
    p.add_argument(
        "--analyze-extra",
        default="[]",
        help='JSON list of extra CLI tokens for 02_analyze_teacher_layers, e.g. \'["--top-k","5"]\'',
    )
    p.add_argument(
        "--subspace-extra",
        default="[]",
        help='JSON list of extra CLI tokens for 03_build_teacher_safe_subspace, e.g. \'["--rank","8"]\' or \'["--no-balance-labels"]\'',
    )
    p.add_argument("--enable-opencompass", action="store_true")
    p.add_argument("--opencompass-dir", default="")
    # Default OFF: safety data JSONL is unchanged across ablation cells, so
    # re-downloading + re-tokenizing on every cell wastes ~10-30 min/cell.
    # Pass --force-rebuild explicitly on the first cell of a fresh checkout
    # or whenever source data / builder changes.
    p.add_argument("--force-rebuild", action="store_true", default=False)
    p.add_argument("--no-force-rebuild", action="store_false", dest="force_rebuild")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    phasef_yaml = DEVICE_YAML[(args.device, "phasef")]
    phase1_yaml = DEVICE_YAML[(args.device, "phase1")]
    phasef_updates = json.loads(args.phasef_set)
    phase1_updates = json.loads(args.phase1_set)
    analyze_extras = json.loads(args.analyze_extra) if args.analyze_extra else []
    subspace_extras = json.loads(args.subspace_extra) if args.subspace_extra else []
    if not isinstance(analyze_extras, list) or not all(isinstance(x, str) for x in analyze_extras):
        raise ValueError("--analyze-extra must be a JSON list of strings")
    if not isinstance(subspace_extras, list) or not all(isinstance(x, str) for x in subspace_extras):
        raise ValueError("--subspace-extra must be a JSON list of strings")

    print(f"[sweep] axis={args.axis} baseline={args.baseline} device={args.device}:{args.device_id}")
    print(f"[sweep] phasef_set={phasef_updates}")
    print(f"[sweep] phase1_set={phase1_updates}")
    print(f"[sweep] analyze_extra={analyze_extras}")
    print(f"[sweep] subspace_extra={subspace_extras}")

    orig_phasef = patch_yaml(phasef_yaml, phasef_updates)
    orig_phase1 = patch_yaml(phase1_yaml, phase1_updates)
    orig_launcher = patch_launcher_phase1_args(analyze_extras, subspace_extras)

    exit_code = -1
    elapsed = 0
    try:
        # Use `full --baseline X` (universal dispatch) instead of `safety-full`.
        # `safety-full --baseline pan` is rejected by argparse (pan not in SAFETY_SFT_BASELINES);
        # `full --baseline pan` -> _run_full_pipeline, `full --baseline <safety>` -> _run_safety_full.
        cmd: list[str] = [
            sys.executable,
            str(LAUNCHER),
            "full",
            "--baseline",
            args.baseline,
            "--device",
            args.device,
            "--device-id",
            str(args.device_id),
        ]
        if args.force_rebuild:
            cmd.append("--force-rebuild")
        if args.enable_opencompass:
            cmd.append("--enable-opencompass")
            if args.opencompass_dir:
                cmd.extend(["--opencompass-dir", args.opencompass_dir])
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"[sweep] launch: {' '.join(cmd)}")
        start = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = int(time.time() - start)
        exit_code = result.returncode
    finally:
        restore(phasef_yaml, orig_phasef)
        restore(phase1_yaml, orig_phase1)
        restore(LAUNCHER, orig_launcher)

    summary_path = find_summary(args.baseline, args.device)
    hr, or_ = parse_hr_or(summary_path)
    f1 = safety_f1(hr, or_)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "axis": args.axis,
        "baseline": args.baseline,
        "device": args.device,
        "device_id": args.device_id,
        "phasef_updates": json.dumps(phasef_updates, ensure_ascii=False),
        "phase1_updates": json.dumps(phase1_updates, ensure_ascii=False),
        "analyze_extra": json.dumps(analyze_extras),
        "subspace_extra": json.dumps(subspace_extras),
        "exit_code": exit_code,
        "elapsed_sec": elapsed,
        "HR": hr,
        "OR": or_,
        "F1": f1,
        "summary_path": str(summary_path) if summary_path else "",
    }
    append_csv_row(row)
    print(f"[sweep] done axis={args.axis} baseline={args.baseline} HR={hr} OR={or_} F1={f1}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
