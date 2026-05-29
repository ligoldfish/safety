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
import shutil
import subprocess
import sys
import time
import uuid
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
SWEEP_CONFIG_DIR = PROJECT_ROOT / "configs" / "_sweep"


def set_dotted(data: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def stage_isolated_yamls(
    cell_id: str,
    device: str,
    phasef_updates: dict,
    phase1_updates: dict,
    phasef_output_root_override: Optional[Path] = None,
    phase1_output_root_override: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """Copy phaseF + phase1 base yamls to a cell-unique dir, apply patches.

    Returns (phasef_copy_path, phase1_copy_path, cell_dir). Caller passes the
    two paths to the launcher via --phasef-config / --phase1-config so two
    concurrent sweep cells do not race on the shared base yaml.
    """
    uuid_str = uuid.uuid4().hex[:8]
    cell_dir = SWEEP_CONFIG_DIR / f"{cell_id}_{uuid_str}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    phasef_src = DEVICE_YAML[(device, "phasef")]
    phase1_src = DEVICE_YAML[(device, "phase1")]
    phasef_copy = cell_dir / "phaseF.yaml"
    phase1_copy = cell_dir / "phase1.yaml"

    phasef_data = yaml.safe_load(phasef_src.read_text(encoding="utf-8"))
    phase1_data = yaml.safe_load(phase1_src.read_text(encoding="utf-8"))

    for dotted, value in (phasef_updates or {}).items():
        set_dotted(phasef_data, dotted, value)
    for dotted, value in (phase1_updates or {}).items():
        set_dotted(phase1_data, dotted, value)

    if phasef_output_root_override is not None:
        set_dotted(phasef_data, "output.output_root", str(phasef_output_root_override))
    if phase1_output_root_override is not None:
        set_dotted(phase1_data, "extraction.output_root", str(phase1_output_root_override))

    phasef_copy.write_text(
        yaml.safe_dump(phasef_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    phase1_copy.write_text(
        yaml.safe_dump(phase1_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return phasef_copy, phase1_copy, cell_dir


def cleanup_isolated_yamls(cell_dir: Path) -> None:
    if cell_dir.exists():
        shutil.rmtree(cell_dir, ignore_errors=True)


def archive_cell_outputs(
    axis: str,
    baseline: str,
    device: str,
    summary_path: Optional[Path],
) -> Optional[Path]:
    """Copy per-cell eval artifacts to outputs/sweep/<axis>_<baseline>/.

    Preserves eval_suite/ (small, ~MB), manifest.json, and training.log per
    ablation cell, so reruns of a different cell on the same baseline do not
    overwrite prior results. Checkpoints are NOT copied (GB-sized; cell rerun
    cost ~3-6h is acceptable if checkpoint inspection is needed later).
    """
    if not summary_path or not summary_path.exists():
        return None
    eval_suite_root = summary_path.parents[1]  # .../eval_suite/
    training_root = eval_suite_root.parent  # .../training/

    archive_dir = PROJECT_ROOT / "outputs" / "sweep" / f"{axis}_{baseline}_{device}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    target_eval_suite = archive_dir / "eval_suite"
    if target_eval_suite.exists():
        shutil.rmtree(target_eval_suite)
    shutil.copytree(eval_suite_root, target_eval_suite)

    for fname in ("manifest.json",):
        src = training_root / fname
        if src.exists():
            shutil.copy2(src, archive_dir / fname)

    training_log = training_root / "logs" / "training.log"
    if training_log.exists():
        shutil.copy2(training_log, archive_dir / "training.log")

    return archive_dir


def find_summary(baseline: str, device: str, cell_id: str = "") -> Optional[Path]:
    """Locate per-cell final_summary.json.

    With cell_id (parallel sweep mode):
      PAN:    outputs/sweep_runs/<cell_id>/phase1/training/eval_suite/epoch_*/final_summary.json
      Safety: outputs/safety_full_<baseline>_<device>_<cell_id>/phase1/training/eval_suite/epoch_*/final_summary.json

    Without cell_id (legacy / direct invocation):
      PAN:    outputs/qwen35_9b_to_08b_phase1_<device>/training/eval_suite/epoch_*/final_summary.json
      Safety: outputs/safety_full_<baseline>_<device>/phase1/training/eval_suite/epoch_*/final_summary.json
    """
    candidates_roots: list[tuple[Path, str]] = []
    if cell_id:
        if baseline == "pan":
            candidates_roots.append(
                (PROJECT_ROOT / "outputs" / "sweep_runs" / cell_id,
                 "phase1/training/eval_suite/epoch_*/final_summary.json")
            )
        else:
            candidates_roots.append(
                (PROJECT_ROOT / "outputs" / f"safety_full_{baseline}_{device}_{cell_id}",
                 "phase1/training/eval_suite/epoch_*/final_summary.json")
            )
    # Always also check the legacy (non-cell) paths as fallback so old runs
    # remain discoverable.
    if baseline == "pan":
        candidates_roots.append(
            (PROJECT_ROOT / "outputs" / f"qwen35_9b_to_08b_phase1_{device}",
             "training/eval_suite/epoch_*/final_summary.json")
        )
    else:
        candidates_roots.append(
            (PROJECT_ROOT / "outputs" / f"safety_full_{baseline}_{device}",
             "phase1/training/eval_suite/epoch_*/final_summary.json")
        )

    for base, pattern in candidates_roots:
        candidates = sorted(base.glob(pattern), key=lambda p: p.parent.name)
        if candidates:
            return candidates[-1]
    return None


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
        # Accept any baseline registered in the launcher; explicit choices kept
        # loose so new datasets (wildjailbreak/wildguardmix/hh_rlhf/...) work
        # without touching this script.
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

    phasef_updates = json.loads(args.phasef_set)
    phase1_updates = json.loads(args.phase1_set)
    analyze_extras = json.loads(args.analyze_extra) if args.analyze_extra else []
    subspace_extras = json.loads(args.subspace_extra) if args.subspace_extra else []
    if not isinstance(analyze_extras, list) or not all(isinstance(x, str) for x in analyze_extras):
        raise ValueError("--analyze-extra must be a JSON list of strings")
    if not isinstance(subspace_extras, list) or not all(isinstance(x, str) for x in subspace_extras):
        raise ValueError("--subspace-extra must be a JSON list of strings")

    cell_id = f"{args.axis}_{args.baseline}_{args.device}_{args.device_id}"

    print(f"[sweep] axis={args.axis} baseline={args.baseline} device={args.device}:{args.device_id}")
    print(f"[sweep] cell_id={cell_id}")
    print(f"[sweep] phasef_set={phasef_updates}")
    print(f"[sweep] phase1_set={phase1_updates}")
    print(f"[sweep] analyze_extra={analyze_extras}")
    print(f"[sweep] subspace_extra={subspace_extras}")

    # Per-cell isolated output dirs.
    # PAN flow reads output paths directly from the yaml copies.
    # Safety flow ignores yaml output_root (overridden in _make_safety_full_overrides)
    # and uses --cell-id suffix on outputs/safety_full_<baseline>_<device>_<cell_id>/ instead.
    pan_cell_output_root = (
        PROJECT_ROOT / "outputs" / "sweep_runs" / cell_id
    ).resolve()
    pan_phase1_output_root = pan_cell_output_root / "phase1"
    pan_phasef_output_root = pan_phase1_output_root / "training"

    phasef_copy, phase1_copy, cell_dir = stage_isolated_yamls(
        cell_id=cell_id,
        device=args.device,
        phasef_updates=phasef_updates,
        phase1_updates=phase1_updates,
        # PAN flow uses these paths directly. Safety flow overrides them anyway.
        phasef_output_root_override=pan_phasef_output_root,
        phase1_output_root_override=pan_phase1_output_root,
    )

    exit_code = -1
    elapsed = 0
    try:
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
            "--phasef-config",
            str(phasef_copy),
            "--phase1-config",
            str(phase1_copy),
            "--cell-id",
            cell_id,
        ]
        # Pass as JSON strings so argparse does not mis-parse "--"-prefixed
        # tokens (e.g. --top-k 1, --no-balance-labels) as new flags.
        cmd.append(f"--phase1-analyze-extra={json.dumps(analyze_extras)}")
        cmd.append(f"--phase1-subspace-extra={json.dumps(subspace_extras)}")
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
        cleanup_isolated_yamls(cell_dir)

    summary_path = find_summary(args.baseline, args.device, cell_id=cell_id)
    hr, or_ = parse_hr_or(summary_path)
    f1 = safety_f1(hr, or_)

    # Archive eval_suite + manifest + training log to outputs/sweep/<axis>_<baseline>_<device>/
    # for stable per-cell browsability (safe to delete the live training/ dir after).
    archive_dir = archive_cell_outputs(args.axis, args.baseline, args.device, summary_path)

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
        "archive_dir": str(archive_dir) if archive_dir else "",
    }
    append_csv_row(row)
    print(f"[sweep] done axis={args.axis} baseline={args.baseline} HR={hr} OR={or_} F1={f1}")
    if archive_dir:
        print(f"[sweep] archived to {archive_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
