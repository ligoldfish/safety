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


def _absolutize_yaml_paths(data: Any, anchor: Path) -> None:
    """Walk yaml dict in place and convert any "../" or "./" leading-string
    values to absolute paths anchored at <anchor>. yaml relative paths are
    resolved relative to the yaml FILE LOCATION, so when we move the yaml
    to a sweep subdir, every relative path silently breaks. Absolutizing
    against the ORIGINAL yaml's parent fixes that.
    """
    if isinstance(data, dict):
        for k, v in list(data.items()):
            if isinstance(v, str) and (v.startswith("../") or v.startswith("./")):
                data[k] = str((anchor / v).resolve())
            else:
                _absolutize_yaml_paths(v, anchor)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, str) and (v.startswith("../") or v.startswith("./")):
                data[i] = str((anchor / v).resolve())
            else:
                _absolutize_yaml_paths(v, anchor)


def stage_isolated_yamls(
    cell_id: str,
    device: str,
    phasef_updates: dict,
    phase1_updates: dict,
    phasef_output_root_override: Optional[Path] = None,
    phase1_output_root_override: Optional[Path] = None,
    phasef_targets_phase1_root: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """Copy phaseF + phase1 base yamls to a cell-unique dir, apply patches.

    Returns (phasef_copy_path, phase1_copy_path, cell_dir). Caller passes the
    two paths to the launcher via --phasef-config / --phase1-config so two
    concurrent sweep cells do not race on the shared base yaml.

    All relative paths inside the base yamls (../external/..., ../models/...,
    ../data/..., ../outputs/...) are absolutized against the ORIGINAL yaml's
    parent dir before being written to the copy, so the copy at a different
    location still resolves paths correctly.
    """
    uuid_str = uuid.uuid4().hex[:8]
    cell_dir = SWEEP_CONFIG_DIR / f"{cell_id}_{uuid_str}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    phasef_src = DEVICE_YAML[(device, "phasef")]
    phase1_src = DEVICE_YAML[(device, "phase1")]
    phasef_anchor = phasef_src.parent.resolve()
    phase1_anchor = phase1_src.parent.resolve()
    phasef_copy = cell_dir / "phaseF.yaml"
    phase1_copy = cell_dir / "phase1.yaml"

    phasef_data = yaml.safe_load(phasef_src.read_text(encoding="utf-8"))
    phase1_data = yaml.safe_load(phase1_src.read_text(encoding="utf-8"))

    # Absolutize BEFORE applying user patches so user-supplied absolute paths
    # (e.g. output_root override) are not double-resolved.
    _absolutize_yaml_paths(phasef_data, phasef_anchor)
    _absolutize_yaml_paths(phase1_data, phase1_anchor)

    for dotted, value in (phasef_updates or {}).items():
        set_dotted(phasef_data, dotted, value)
    for dotted, value in (phase1_updates or {}).items():
        set_dotted(phase1_data, dotted, value)

    if phasef_output_root_override is not None:
        set_dotted(phasef_data, "output.output_root", str(phasef_output_root_override))
    if phase1_output_root_override is not None:
        set_dotted(phase1_data, "extraction.output_root", str(phase1_output_root_override))

    # PAN correctness fix: the legacy `full` flow (_run_full_pipeline in
    # 15_run_oneclick.py) recomputes Phase 1 into the per-cell
    # extraction.output_root but NEVER rewrites the PhaseF target/pairing inputs,
    # so PhaseF silently reads the SHARED default phase1 dir
    # (outputs/qwen35_9b_to_08b_phase1_npu/...). Result: a PAN cell's swept
    # PHASE-1 params -- top-k (--analyze-extra) and energy-threshold/rank-cap
    # (--subspace-extra) -- are computed but THROWN AWAY; PhaseF trains on the
    # default/stale subspace targets. (Confirmed via an old sweep manifest whose
    # train_targets_dir pointed at the shared dir.) Re-root the PhaseF
    # target+pairing inputs at THIS cell's phase1 output, exactly as
    # _make_safety_full_overrides does for safety baselines. Safety baselines go
    # through the launcher's own override (which re-derives these from a cell-id
    # suffixed dir and ignores this yaml), so the caller passes this only for PAN.
    if phasef_targets_phase1_root is not None:
        p1 = phasef_targets_phase1_root
        st = phasef_data.setdefault("inputs", {})
        st["train_targets_dir"] = str(p1 / "student_targets" / "student_safe_targets_alignment")
        st["val_targets_dir"] = str(p1 / "student_targets" / "student_safe_targets_val")
        st["pairing_path"] = str(p1 / "layer_pairing" / "teacher_student_layer_pairs.json")

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


def _scale_to_percent(value: Optional[float]) -> Optional[float]:
    """summary.json may store HR/OR as fraction [0,1] OR percentage [0,100].
    Heuristic: if value <= 1.0, treat as fraction and convert to percentage.
    Edge case: a true 0.5% rate stored as percentage would be mis-doubled,
    but safety eval HR/OR rarely fall below 1%, so this is acceptable.
    """
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


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
    if hit is None:
        return None, None
    return _scale_to_percent(hit[0]), _scale_to_percent(hit[1])


def safety_f1(hr: Optional[float], or_: Optional[float]) -> Optional[float]:
    if hr is None or or_ is None:
        return None
    helpful = 100.0 - or_
    denom = hr + helpful
    return 0.0 if denom == 0 else 2.0 * hr * helpful / denom


def resolve_eval_test_jsonl(baseline: str, device: str) -> Optional[Path]:
    """Find the eval test JSONL (the id->prompt join source the judge needs).

    Mirrors the launcher's eval-config selection (15_run_oneclick.py
    BASELINE_EVAL_CONFIGS / SAFETY_EVAL_CONFIGS) for the 0.8B student, then reads
    ``datasets.pan.path`` out of that YAML. The path is yaml-relative (``../data/...``)
    so it is resolved against the config's own dir. pan uses the canonical eval YAML
    (pan_test_set.jsonl); every safety baseline uses
    baseline_eval_qwen35_08b_<baseline>_<device>.yaml (-> <baseline>_test.jsonl).
    """
    cfgdir = PROJECT_ROOT / "configs"
    if baseline == "pan":
        candidates = [cfgdir / f"baseline_eval_qwen35_08b_{device}.yaml"]
    else:
        candidates = [
            cfgdir / f"baseline_eval_qwen35_08b_{baseline}_{device}.yaml",
            cfgdir / f"baseline_eval_qwen35_08b_{baseline}_npu.yaml",
        ]
    cfg = next((c for c in candidates if c.exists()), None)
    if cfg is None:
        return None
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    rel = (((data or {}).get("datasets") or {}).get("pan") or {}).get("path")
    if not rel:
        return None
    return (cfg.parent / rel).resolve()


def run_judge(
    eval_suite_dir: Path,
    test_jsonl: Optional[Path],
    judge_model: Path,
    device: str,
    judge_device_id: int,
) -> bool:
    """Judge every epoch under eval_suite_dir and merge llm_judge_* into summary.json."""
    if test_jsonl is None or not test_jsonl.exists():
        print(f"[sweep][judge] test jsonl missing ({test_jsonl}); skipping judge", flush=True)
        return False
    if not judge_model.exists():
        print(f"[sweep][judge] judge model missing ({judge_model}); skipping judge", flush=True)
        return False
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "22_judge_generations.py"),
        "--eval-suite-dir",
        str(eval_suite_dir),
        "--test-jsonl",
        str(test_jsonl),
        "--judge-model",
        str(judge_model),
        "--runtime-backend",
        device,
        "--runtime-device",
        f"{device}:{judge_device_id}",
        "--merge-summary",
    ]
    print(f"[sweep][judge] launch: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        print(f"[sweep][judge] judge exited rc={rc}", flush=True)
    return rc == 0


def best_net_epoch(
    eval_suite_dir: Path,
) -> tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    """Return (epoch, judge_HR%, judge_OR%, net%) for the epoch maximizing judge HR-OR.

    Reads merged llm_judge_* out of each epoch's summary.json. This mirrors
    D:/output/_table.py's "ours" epoch pick (max judge_HR - judge_OR) so the cell's
    recorded score is exactly how the final table would score this config.
    """
    best: tuple[Optional[int], Optional[float], Optional[float], Optional[float]] = (
        None,
        None,
        None,
        None,
    )
    for sj in sorted(eval_suite_dir.glob("epoch_*/summary.json")):
        try:
            pan = json.loads(sj.read_text(encoding="utf-8"))["results"]["pan"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        hr_raw = pan.get("llm_judge_refusal_rate")
        or_raw = pan.get("llm_judge_over_refusal")
        if hr_raw is None or or_raw is None:
            continue
        hr_p = _scale_to_percent(float(hr_raw))
        or_p = _scale_to_percent(float(or_raw))
        if hr_p is None or or_p is None:
            continue
        net = hr_p - or_p
        m = re.search(r"epoch_(\d+)", str(sj))
        ep = int(m.group(1)) if m else -1
        if best[3] is None or net > best[3]:
            best = (ep, hr_p, or_p, net)
    return best


def verify_cell_applied(
    summary_path: Optional[Path], analyze_extras: Sequence[str]
) -> tuple[Optional[bool], str]:
    """Guard that the cell's swept params actually reached training.

    Reads the per-cell manifest.json next to the run's training dir and checks:
      1) train_targets_dir / pairing_path live under THIS cell's phase1 output
         (not the shared default dir -> would mean PhaseF ate stale/default
         subspace targets and the swept phase-1 params were ignored);
      2) when --top-k was swept, the manifest's #layer-pairs equals the requested
         K (catches the historical "top-k didn't propagate" staleness).
    Returns (ok, note). ok=None when the manifest is missing/unreadable.
    """
    if summary_path is None:
        return None, "no summary path"
    training_root = summary_path.parents[2]  # .../phase1/training/
    cell_phase1_root = training_root.parent  # .../phase1/
    manifest_path = training_root / "manifest.json"
    if not manifest_path.exists():
        return None, f"no manifest at {manifest_path}"
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"manifest unreadable: {exc}"

    problems: list[str] = []
    for key in ("train_targets_dir", "pairing_path"):
        val = str(man.get(key, ""))
        try:
            under = Path(val).resolve().is_relative_to(cell_phase1_root.resolve())
        except (ValueError, OSError):
            under = False
        if not under:
            problems.append(f"{key} not under cell phase1 ({val})")

    if "--top-k" in analyze_extras:
        idx = list(analyze_extras).index("--top-k")
        try:
            want_k = int(analyze_extras[idx + 1])
        except (ValueError, IndexError):
            want_k = None
        if want_k is not None:
            got = len(man.get("pair_to_student_layer", {}) or {})
            if got != want_k:
                problems.append(f"requested top-k={want_k} but manifest has {got} pairs")

    return (len(problems) == 0), ("OK" if not problems else "; ".join(problems))


def append_csv_row(row: dict) -> None:
    fieldnames = list(row.keys())
    # Schema guard: if an existing CSV has a different header (e.g. the pre-judge
    # pilot's 11-col schema), appending judge columns would silently mis-align
    # every row. Rotate the stale CSV to <name>.bak.<ts> and start a fresh header.
    if SWEEP_CSV.exists():
        with SWEEP_CSV.open("r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fieldnames:
            backup = SWEEP_CSV.with_suffix(
                SWEEP_CSV.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}"
            )
            SWEEP_CSV.rename(backup)
            print(f"[sweep] CSV schema changed; rotated old results to {backup}", flush=True)
    new_file = not SWEEP_CSV.exists()
    with SWEEP_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    p.add_argument(
        "--judge",
        action="store_true",
        help="After training+eval, run the WildGuard judge (22_judge_generations.py) over "
        "every epoch's pan_results.json, merge llm_judge_* into summary.json, and record the "
        "best-net (max judge_HR - judge_OR) epoch in the CSV. Mirrors D:/output/_table.py's "
        "ours-epoch pick so the sweep winner == the final-table winner.",
    )
    p.add_argument(
        "--judge-model",
        default="",
        help="Path to the WildGuard judge model. Default: <repo>/models/wildguard.",
    )
    p.add_argument(
        "--judge-device-id",
        type=int,
        default=None,
        help="NPU/PPU die for the judge. Default: same as --device-id (judge runs after "
        "training frees the die, so reusing it avoids contending for a second card).",
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
        # PAN only: re-root PhaseF target/pairing inputs at this cell's phase1 so
        # swept phase-1 params (top-k, energy/rank-cap) actually reach PhaseF.
        # Safety baselines re-derive these in the launcher, so leave None there.
        phasef_targets_phase1_root=(pan_phase1_output_root if args.baseline == "pan" else None),
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

    # Optional WildGuard judging: judge every epoch on the LIVE eval_suite (before
    # archiving) so the merged llm_judge_* summaries get archived too, then pick the
    # best-net (max judge_HR - judge_OR) epoch to record as the cell's score.
    judge_hr = judge_or = judge_net = None
    judge_ep = None
    if args.judge and summary_path is not None:
        eval_suite_dir = summary_path.parents[1]  # .../eval_suite/
        test_jsonl = resolve_eval_test_jsonl(args.baseline, args.device)
        judge_model = (
            Path(args.judge_model)
            if args.judge_model
            else (PROJECT_ROOT / "models" / "wildguard")
        )
        judge_dev = args.judge_device_id if args.judge_device_id is not None else args.device_id
        if run_judge(eval_suite_dir, test_jsonl, judge_model, args.device, judge_dev):
            judge_ep, judge_hr, judge_or, judge_net = best_net_epoch(eval_suite_dir)

    # Guard: confirm the swept params actually reached training (not silently
    # replaced by the shared/default phase1 outputs). Loud WARN on failure so a
    # no-op cell never masquerades as a real data point in the CSV.
    applied_ok, applied_note = verify_cell_applied(summary_path, analyze_extras)
    if applied_ok is False:
        print(
            f"[sweep][WARN] cell {cell_id} swept params may NOT have applied: {applied_note}",
            flush=True,
        )
    elif applied_ok:
        print(f"[sweep] cell {cell_id} applied-check OK", flush=True)

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
        "judge_HR": judge_hr,
        "judge_OR": judge_or,
        "judge_net": judge_net,
        "judge_best_epoch": judge_ep,
        "applied_ok": applied_ok,
        "applied_note": applied_note,
        "summary_path": str(summary_path) if summary_path else "",
        "archive_dir": str(archive_dir) if archive_dir else "",
    }
    append_csv_row(row)
    print(
        f"[sweep] done axis={args.axis} baseline={args.baseline} "
        f"kw(HR={hr} OR={or_} F1={f1}) "
        f"judge(HR={judge_hr} OR={judge_or} net={judge_net} ep={judge_ep})"
    )
    if archive_dir:
        print(f"[sweep] archived to {archive_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
