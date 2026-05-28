"""Inspect sweep_results.csv. Three modes: all / winners / status.

Usage:
  python scripts/sweep_summary.py --csv sweep_results.csv --mode all
  python scripts/sweep_summary.py --csv sweep_results.csv --mode winners
  python scripts/sweep_summary.py --csv sweep_results.csv --mode status --total-cells 33

Modes:
  all       : print all rows grouped by axis, show HR/OR/F1 per baseline.
  winners   : for each axis, pick value with best mean F1 across baselines.
              Tie-break: lower mean OR.
  status    : pending vs completed counts. Needs --total-cells N.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional


def parse_axis_prefix(axis_id: str) -> str:
    """Return the alphabetic prefix part of an axis label, e.g. 'B' from 'B1'."""
    for i, ch in enumerate(axis_id):
        if ch.isdigit() or ch == "_":
            return axis_id[:i] if i > 0 else axis_id
    return axis_id


def fmt_pct(v) -> str:
    if v is None or v == "" or v == "None":
        return "  -  "
    try:
        return f"{float(v):5.2f}"
    except (TypeError, ValueError):
        return "  -  "


def load_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in ("HR", "OR", "F1"):
                v = row.get(k)
                if v in ("", "None", None):
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = None
            rows.append(row)
    return rows


def mode_all(rows: list[dict]) -> None:
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r)

    print(f"{'axis':6} {'baseline':22} {'HR':>6} {'OR':>6} {'F1':>6} {'elapsed':>8} {'config':60}")
    print("-" * 120)
    for axis in sorted(by_axis):
        for r in by_axis[axis]:
            hr = fmt_pct(r["HR"])
            or_ = fmt_pct(r["OR"])
            f1 = fmt_pct(r["F1"])
            cfg_parts = []
            for k in ("phasef_updates", "phase1_updates", "analyze_extra", "subspace_extra"):
                v = r.get(k, "")
                if v and v not in ("{}", "[]"):
                    cfg_parts.append(f"{k.split('_')[0]}={v}")
            cfg = "; ".join(cfg_parts)[:60]
            print(f"{axis:6} {r['baseline']:22} {hr:>6} {or_:>6} {f1:>6} {r.get('elapsed_sec',''):>8} {cfg}")
        print()


def mode_winners(rows: list[dict]) -> None:
    # Group: axis_id -> baseline -> row (latest if duplicates)
    by_cell: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["F1"] is None:
            continue
        key = (r["axis"], r["baseline"])
        # Keep latest timestamp
        if key not in by_cell or r["timestamp"] > by_cell[key]["timestamp"]:
            by_cell[key] = r

    # Mean F1 per axis_id across baselines that have data
    cell_stats: dict[str, dict] = {}
    for (axis, _bl), r in by_cell.items():
        stat = cell_stats.setdefault(axis, {"f1s": [], "ors": [], "hrs": [], "rows": []})
        stat["f1s"].append(r["F1"])
        stat["ors"].append(r["OR"] if r["OR"] is not None else 100.0)
        stat["hrs"].append(r["HR"] if r["HR"] is not None else 0.0)
        stat["rows"].append(r)

    # Per-axis-prefix: choose cell with highest mean F1, tiebreak by lower mean OR
    prefix_winners: dict[str, dict] = {}
    for axis_id, stat in cell_stats.items():
        prefix = parse_axis_prefix(axis_id)
        candidate = {
            "axis_id": axis_id,
            "mean_F1": mean(stat["f1s"]),
            "mean_OR": mean(stat["ors"]),
            "mean_HR": mean(stat["hrs"]),
            "n_baselines": len(stat["f1s"]),
            "rows": stat["rows"],
        }
        existing = prefix_winners.get(prefix)
        if existing is None:
            prefix_winners[prefix] = candidate
            continue
        if candidate["mean_F1"] > existing["mean_F1"] + 1e-9:
            prefix_winners[prefix] = candidate
        elif abs(candidate["mean_F1"] - existing["mean_F1"]) <= 1e-9 and candidate["mean_OR"] < existing["mean_OR"]:
            prefix_winners[prefix] = candidate

    print(f"{'axis':6} {'winner':6} {'n':>3} {'mean_HR':>8} {'mean_OR':>8} {'mean_F1':>8}  config")
    print("-" * 100)
    for prefix in sorted(prefix_winners):
        w = prefix_winners[prefix]
        cfg_row = w["rows"][0]
        cfg_parts = []
        for k in ("phasef_updates", "phase1_updates", "analyze_extra", "subspace_extra"):
            v = cfg_row.get(k, "")
            if v and v not in ("{}", "[]"):
                cfg_parts.append(f"{k.split('_')[0]}={v}")
        cfg = "; ".join(cfg_parts)
        print(
            f"{prefix:6} {w['axis_id']:6} {w['n_baselines']:>3} "
            f"{w['mean_HR']:>8.2f} {w['mean_OR']:>8.2f} {w['mean_F1']:>8.2f}  {cfg}"
        )

    # Suggested Round 2 stack
    print()
    print("=== Suggested Round 2 combo configs (stack top per-axis-prefix winners) ===")
    phasef_merged: dict = {}
    analyze_extra: list[str] = []
    subspace_extra: list[str] = []
    for prefix in ("A", "B", "D"):  # pain-point-1 axes
        w = prefix_winners.get(prefix)
        if not w:
            continue
        cfg_row = w["rows"][0]
        try:
            d = json.loads(cfg_row.get("phasef_updates", "{}") or "{}")
            if isinstance(d, dict):
                phasef_merged.update(d)
        except json.JSONDecodeError:
            pass
    for prefix in ("G",):
        w = prefix_winners.get(prefix)
        if not w:
            continue
        cfg_row = w["rows"][0]
        try:
            analyze_extra = json.loads(cfg_row.get("analyze_extra", "[]") or "[]")
        except json.JSONDecodeError:
            pass
    for prefix in ("C", "F"):
        w = prefix_winners.get(prefix)
        if not w:
            continue
        cfg_row = w["rows"][0]
        try:
            subspace_extra = json.loads(cfg_row.get("subspace_extra", "[]") or "[]")
        except json.JSONDecodeError:
            pass

    if phasef_merged or analyze_extra or subspace_extra:
        print()
        print("  Round 2 R2_STACK_BEST suggestion:")
        if phasef_merged:
            print(f"    --phasef-set={json.dumps(phasef_merged, ensure_ascii=False)}")
        if analyze_extra:
            print(f"    --analyze-extra={json.dumps(analyze_extra, ensure_ascii=False)}")
        if subspace_extra:
            print(f"    --subspace-extra={json.dumps(subspace_extra, ensure_ascii=False)}")


def mode_status(rows: list[dict], total: Optional[int]) -> None:
    completed = sum(1 for r in rows if r["F1"] is not None)
    failed = sum(1 for r in rows if r["exit_code"] not in ("0", 0, "", None))
    seen_keys = {(r["axis"], r["baseline"]) for r in rows if r["F1"] is not None}
    print(f"Total rows in CSV:    {len(rows)}")
    print(f"Successful (F1 set):  {completed}")
    print(f"Non-zero exit code:   {failed}")
    print(f"Unique (axis,baseline) succeeded: {len(seen_keys)}")
    if total is not None:
        print(f"Pending vs total: {total - len(seen_keys)} / {total}")
    print()
    by_prefix: dict[str, int] = defaultdict(int)
    for axis, _bl in seen_keys:
        by_prefix[parse_axis_prefix(axis)] += 1
    print("Per axis-prefix completed cells:")
    for prefix in sorted(by_prefix):
        print(f"  {prefix}: {by_prefix[prefix]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="sweep_results.csv")
    p.add_argument("--mode", choices=["all", "winners", "status"], required=True)
    p.add_argument("--total-cells", type=int, default=None)
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = load_rows(csv_path)
    if args.mode == "all":
        mode_all(rows)
    elif args.mode == "winners":
        mode_winners(rows)
    elif args.mode == "status":
        mode_status(rows, args.total_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
