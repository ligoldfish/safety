"""Select the best training epoch by harmful-refusal rate (HR), tie-break by OR.

Baselines (13_train_pan_sft / 14_train_pan_distill) save every ``epoch_00N.pt``
and ``12_eval_baseline_suite`` evaluates each, writing
``eval_suite/epoch_00N/summary.json`` with ``results.pan`` HR/OR (keyword and,
if judged, ``llm_judge_*``). The deployed/reported checkpoint is otherwise the
fixed last epoch; for the STL baselines (and any run where more epochs were
added to let HR catch up to ours) we instead pick the epoch with the highest
harmful-refusal rate, breaking ties by lower over-refusal.

This is a deploy/report-time SELECTION (post-eval); it does NOT change training.
State the rule ("best epoch by HR") in the paper for the affected baselines.

Usage:
    python scripts/select_best_epoch_by_hr.py <output_dir> [<output_dir> ...] \\
        [--metric judge|keyword|auto] [--write]

``auto`` (default) uses judge HR/OR when present, else keyword.
``--write`` writes ``best_epoch_by_hr.json`` into each output dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_pan(summary_path: Path) -> Optional[Dict]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    pan = data.get("results", {}).get("pan")
    return pan if isinstance(pan, dict) else None


def _hr_or(pan: Dict, metric: str) -> Optional[Tuple[float, float]]:
    """Return (HR, OR) as fractions for the requested metric, or None."""
    judged = "llm_judge_refusal_rate" in pan and "llm_judge_over_refusal" in pan
    if metric == "judge" or (metric == "auto" and judged):
        if not judged:
            return None
        return float(pan["llm_judge_refusal_rate"]), float(pan["llm_judge_over_refusal"])
    # keyword
    if "harmful_refusal_rate" not in pan or "harmless_over_refusal_rate" not in pan:
        return None
    return float(pan["harmful_refusal_rate"]), float(pan["harmless_over_refusal_rate"])


def select_best_epoch(output_dir: Path, metric: str = "auto") -> Optional[Dict]:
    """Scan eval_suite/epoch_*/summary.json; return the best-HR epoch info."""
    eval_root = output_dir / "eval_suite"
    candidates: List[Dict] = []
    for epoch_dir in sorted(eval_root.glob("epoch_*")):
        try:
            epoch = int(epoch_dir.name.split("_")[-1])
        except ValueError:
            continue
        pan = _read_pan(epoch_dir / "summary.json")
        if pan is None:
            continue
        hr_or = _hr_or(pan, metric)
        if hr_or is None:
            continue
        hr, orr = hr_or
        candidates.append({"epoch": epoch, "hr": hr, "or": orr, "metric": metric})
    if not candidates:
        return None
    # max HR, tie-break lower OR, then earlier epoch.
    best = max(candidates, key=lambda c: (c["hr"], -c["or"], -c["epoch"]))
    return {
        "output_dir": str(output_dir),
        "metric": metric,
        "best_epoch": best["epoch"],
        "best_hr": best["hr"],
        "best_or": best["or"],
        "per_epoch": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", help="Model output dir(s) containing eval_suite/.")
    parser.add_argument("--metric", choices=["judge", "keyword", "auto"], default="auto")
    parser.add_argument("--write", action="store_true", help="Write best_epoch_by_hr.json into each dir.")
    args = parser.parse_args()

    for raw in args.output_dirs:
        out_dir = Path(raw)
        result = select_best_epoch(out_dir, metric=args.metric)
        if result is None:
            print(f"{out_dir}: no eval_suite epochs with HR/OR found")
            continue
        print(
            f"{out_dir}: best_epoch={result['best_epoch']:03d} "
            f"HR={result['best_hr'] * 100:.1f} OR={result['best_or'] * 100:.1f} "
            f"(metric={result['metric']})"
        )
        if args.write:
            (out_dir / "best_epoch_by_hr.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
