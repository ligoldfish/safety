"""iso-HR comparison: match each baseline to ours' final-epoch HR, then compare OR.

For each method (`--method NAME:TRAIN_DIR`) we read per-checkpoint VAL HR from
``val_metrics.json`` (epoch_* + sub-epoch step_*). The target is the method named
``ours`` at its last epoch. For each baseline we select the checkpoint whose VAL HR is
closest to the target (val HR only -- never OR, never the test set), then report that
checkpoint's TEST HR/OR. The ``matched`` flag is |TEST HR − ours' TEST HR| ≤ ε.

This isolates the over-refusal (OR) contribution: at a matched harmful-refusal operating
point, is ours' OR lower? Selection is by HR-proximity ONLY -- not cherry-picked on OR.

Example:
    python scripts/23_iso_hr_compare.py --dataset beavertails --epsilon 0.5 \
        --eval-config configs/baseline_eval_qwen35_08b_beavertails_category_npu.yaml \
        --method ours:../outputs/safety_full_beavertails_category_npu/phase1/training \
        --method sft:../outputs/baselines/sft_qwen35_08b_beavertails_category_npu \
        --method distill:../outputs/baselines/distill_qwen35_9b_to_08b_beavertails_category_npu \
        --run-eval --out iso_hr_beavertails.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.iso_hr import last_epoch_key, read_test_hr_or, read_val_hr_or, select_iso_hr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME:TRAIN_DIR",
        help="Repeatable. One method must be named 'ours' (the target).",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.5, help="Match tolerance in percentage points.")
    parser.add_argument("--eval-config", type=str, default="", help="Eval yaml; needed for --run-eval.")
    parser.add_argument("--run-eval", action="store_true", help="Run 12_eval on a selected checkpoint lacking a test summary.")
    parser.add_argument("--out", type=str, default="", help="CSV output path.")
    parser.add_argument("--selection-split", choices=["validation"], default="validation")
    return parser.parse_args()


def _split_method(entry: str) -> Tuple[str, Path]:
    name, _, path = entry.partition(":")
    if not name or not path:
        raise SystemExit(f"--method must be NAME:TRAIN_DIR, got {entry!r}")
    return name.strip(), Path(path.strip())


def _run_eval_checkpoint(train_dir: Path, ckpt_key: str, eval_config: str) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "12_eval_baseline_suite.py"),
        "--config", eval_config,
        "--adapter-manifest", str(train_dir / "manifest.json"),
        "--adapter-checkpoint", str(train_dir / "checkpoints" / f"{ckpt_key}.pt"),
        "--output-dir", str(train_dir / "eval_suite" / ckpt_key),
    ]
    print(f"[iso-hr] running eval for {ckpt_key}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    eps_frac = float(args.epsilon) / 100.0
    methods = dict(_split_method(m) for m in args.method)
    if "ours" not in methods:
        raise SystemExit("One --method must be named 'ours' (the target operating point).")

    val: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        name: read_val_hr_or(train_dir) for name, train_dir in methods.items()
    }

    ours_key = last_epoch_key(val["ours"])
    if ours_key is None:
        raise SystemExit("ours has no epoch_* val HR; cannot define the target.")
    target_hr = val["ours"][ours_key]["val_hr"]
    if target_hr is None:
        raise SystemExit("ours last-epoch val HR is missing.")
    ours_test_hr, ours_test_or = read_test_hr_or(methods["ours"], ours_key)

    rows: List[Dict[str, object]] = []
    needs_eval: List[str] = []
    needs_more_epochs: List[str] = []

    # ours row (the target).
    rows.append(
        {
            "dataset": args.dataset, "method": "ours", "ckpt": ours_key,
            "val_hr_pct": _pct(target_hr), "test_hr_pct": _pct(ours_test_hr),
            "delta_hr_pp": 0.0, "test_or_pct": _pct(ours_test_or), "matched": True,
        }
    )

    for name, train_dir in methods.items():
        if name == "ours":
            continue
        candidates = [{"ckpt": k, "val_hr": v["val_hr"]} for k, v in val[name].items()]
        sel = select_iso_hr(candidates, target_hr, eps_frac, selection_split=args.selection_split)
        if sel is None:
            needs_more_epochs.append(f"{name} (no val HR)")
            continue
        ckpt_key = sel["ckpt"]
        test_hr, test_or = read_test_hr_or(train_dir, ckpt_key)
        if test_hr is None and args.run_eval and args.eval_config:
            _run_eval_checkpoint(train_dir, ckpt_key, args.eval_config)
            test_hr, test_or = read_test_hr_or(train_dir, ckpt_key)
        if test_hr is None:
            needs_eval.append(f"{name}:{ckpt_key} -> {train_dir}/checkpoints/{ckpt_key}.pt")
        delta_test = None if (test_hr is None or ours_test_hr is None) else (test_hr - ours_test_hr)
        matched = delta_test is not None and abs(delta_test) <= eps_frac + 1e-9
        if not matched and test_hr is not None:
            needs_more_epochs.append(f"{name} (|ΔHR_test|={_pct(delta_test):.2f}pp)")
        rows.append(
            {
                "dataset": args.dataset, "method": name, "ckpt": ckpt_key,
                "val_hr_pct": _pct(sel["val_hr"]), "test_hr_pct": _pct(test_hr),
                "delta_hr_pp": None if delta_test is None else _pct(delta_test),
                "test_or_pct": _pct(test_or), "matched": matched,
            }
        )

    _print_table(rows, ours_test_or, args.epsilon, needs_eval, needs_more_epochs)
    if args.out:
        _write_csv(Path(args.out), rows)


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value) * 100.0, 4)


def _print_table(rows, ours_test_or, epsilon, needs_eval, needs_more_epochs) -> None:
    print(f"\n=== iso-HR comparison (ε={epsilon}pp) ===")
    print(f"{'method':<10} {'ckpt':<16} {'val_HR%':>8} {'test_HR%':>9} {'ΔHR_pp':>7} {'test_OR%':>9} {'matched':>8}")
    for r in rows:
        print(
            f"{str(r['method']):<10} {str(r['ckpt']):<16} "
            f"{_fmt(r['val_hr_pct']):>8} {_fmt(r['test_hr_pct']):>9} "
            f"{_fmt(r['delta_hr_pp']):>7} {_fmt(r['test_or_pct']):>9} {str(r['matched']):>8}"
        )
    matched_or = [(r['method'], r['test_or_pct']) for r in rows if r['matched'] and r['test_or_pct'] is not None]
    if matched_or and ours_test_or is not None:
        lowest = min(matched_or, key=lambda x: x[1])
        verdict = "YES" if lowest[0] == "ours" else f"NO (lowest = {lowest[0]})"
        print(f"\n[verdict] at iso-HR (±{epsilon}pp): ours OR lowest? {verdict}")
    if needs_eval:
        print("\n[NEEDS_EVAL] selected checkpoints lacking a test summary (rerun with --run-eval, or eval manually):")
        for item in needs_eval:
            print(f"  - {item}")
    if needs_more_epochs:
        print("\n[NEEDS_MORE_EPOCHS] baselines not matched within ε (raise optim.epochs / match_band, rerun):")
        for item in needs_more_epochs:
            print(f"  - {item}")


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["dataset", "method", "ckpt", "val_hr_pct", "test_hr_pct", "delta_hr_pp", "test_or_pct", "matched"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[iso-hr] wrote {path}")


if __name__ == "__main__":
    main()
