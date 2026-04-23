"""Merge PAN baseline-eval summary with OpenCompass general-capability results.

The one-click launcher routes PAN safety evaluation through
``scripts/12_eval_baseline_suite.py`` (emitting ``summary.json``) and routes
general-capability datasets (MMLU / GSM8K / HumanEval / MBPP ...) through
``scripts/17_eval_opencompass.py`` (OpenCompass). This helper consolidates the
two sources into a single ``final_summary.json`` so downstream reporting does
not have to know about either layout.

Design notes
------------
OpenCompass writes its rollups as time-stamped CSV tables under
``<work_dir>/summary/summary_<timestamp>.csv`` (older builds) or
``<work_dir>/<timestamp>/summary/summary_<timestamp>.csv`` (newer builds). We
pick the latest such file, best-effort parse it as a table of
``(dataset, metric, value)`` rows, and surface both a structured dict and a
pointer to the raw file. The rest of the eval pipeline remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge PAN eval summary.json with OpenCompass work_dir output into a "
            "single final_summary.json."
        )
    )
    parser.add_argument(
        "--pan-summary",
        type=str,
        required=True,
        help="Path to summary.json produced by scripts/12_eval_baseline_suite.py.",
    )
    parser.add_argument(
        "--opencompass-work-dir",
        type=str,
        default="",
        help=(
            "Path to the OpenCompass --work-dir used by scripts/17_eval_opencompass.py. "
            "Leave blank to record PAN-only results."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the merged final_summary.json.",
    )
    return parser.parse_args()


def _latest_opencompass_summary_csv(work_dir: Path) -> Optional[Path]:
    if not work_dir.exists():
        return None
    candidates: List[Path] = []
    for pattern in ("summary/summary_*.csv", "*/summary/summary_*.csv"):
        candidates.extend(work_dir.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_opencompass_summary_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse OpenCompass summary CSV into {dataset -> {metric -> value}}.

    OpenCompass CSVs usually have columns like:
        dataset, version, metric, mode, <model-name>
    The value column name depends on the evaluated model. We take any column
    that is not in the known header set as a candidate value column; the first
    such column wins.
    """

    known_header = {"dataset", "version", "metric", "mode", "category"}
    parsed: Dict[str, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        value_cols = [col for col in (reader.fieldnames or []) if col.lower() not in known_header]
        if not value_cols:
            return {}
        value_col = value_cols[0]
        for row in reader:
            dataset = str(row.get("dataset", "") or "").strip()
            metric = str(row.get("metric", "") or "").strip() or "score"
            raw_value = (row.get(value_col) or "").strip()
            if not dataset or not raw_value:
                continue
            try:
                value: Any = float(raw_value)
            except (TypeError, ValueError):
                value = raw_value
            parsed.setdefault(dataset, {})[metric] = value
    return parsed


def _load_pan_summary(pan_summary_path: Path) -> Dict[str, Any]:
    if not pan_summary_path.exists():
        raise FileNotFoundError(f"PAN summary not found: {pan_summary_path}")
    return json.loads(pan_summary_path.read_text(encoding="utf-8"))


# PAN safety eval is the single source of truth we keep from
# scripts/12_eval_baseline_suite.py. Any other dataset that the local suite
# happens to have run (mmlu / gsm8k / humaneval / mbpp, whether "disabled" or
# real numbers) is intentionally stripped from final_summary.json so general-
# capability numbers can ONLY come from OpenCompass. This prevents the local
# thinking-era evaluator from silently contributing numbers to the final
# merged report when a user flips a dataset.enabled flag in the eval YAML.
_PAN_TASK_KEY = "pan"


def _filter_pan_summary(pan_summary: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    filtered = dict(pan_summary)
    dropped_tasks: List[str] = []
    results = filtered.get("results")
    if isinstance(results, dict):
        kept = {}
        for task_name, task_result in results.items():
            if task_name == _PAN_TASK_KEY:
                kept[task_name] = task_result
            else:
                dropped_tasks.append(str(task_name))
        filtered["results"] = kept
    debug_outputs = filtered.get("debug_outputs")
    if isinstance(debug_outputs, dict):
        filtered["debug_outputs"] = {
            task_name: payload
            for task_name, payload in debug_outputs.items()
            if task_name == _PAN_TASK_KEY
        }
    return filtered, dropped_tasks


def build_merged_summary(
    *,
    pan_summary_path: Path,
    opencompass_work_dir: Optional[Path],
) -> Dict[str, Any]:
    pan_summary = _load_pan_summary(pan_summary_path)
    filtered_pan_summary, dropped_local_general_tasks = _filter_pan_summary(pan_summary)

    merged: Dict[str, Any] = {
        "pan_summary_path": str(pan_summary_path.resolve()),
        "pan_summary": filtered_pan_summary,
        "dropped_local_general_tasks": dropped_local_general_tasks,
        "opencompass": {
            "enabled": False,
            "work_dir": None,
            "summary_csv": None,
            "results": {},
        },
    }

    if opencompass_work_dir is None:
        return merged

    summary_csv = _latest_opencompass_summary_csv(opencompass_work_dir)
    merged["opencompass"] = {
        "enabled": True,
        "work_dir": str(opencompass_work_dir.resolve()),
        "summary_csv": str(summary_csv.resolve()) if summary_csv is not None else None,
        "results": _parse_opencompass_summary_csv(summary_csv) if summary_csv is not None else {},
    }
    return merged


def main() -> None:
    args = parse_args()
    pan_summary_path = Path(args.pan_summary).resolve()
    opencompass_work_dir = (
        Path(args.opencompass_work_dir).resolve()
        if args.opencompass_work_dir
        else None
    )
    merged = build_merged_summary(
        pan_summary_path=pan_summary_path,
        opencompass_work_dir=opencompass_work_dir,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
