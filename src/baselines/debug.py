from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import ensure_dir, write_json, write_jsonl


DEBUG_TASKS = {"mmlu", "gsm8k", "humaneval", "mbpp"}


def _is_error_prediction(task_name: str, prediction: Dict[str, Any]) -> bool:
    task_key = str(task_name).strip().lower()
    if task_key in {"mmlu", "gsm8k"}:
        return not bool(prediction.get("correct", False))
    if task_key in {"humaneval", "mbpp"}:
        return not bool(prediction.get("passed", False))
    return False


def collect_error_predictions(
    task_name: str,
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    task_key = str(task_name).strip().lower()
    if task_key not in DEBUG_TASKS:
        return []
    return [prediction for prediction in predictions if _is_error_prediction(task_key, prediction)]


def export_error_predictions(
    output_root: str | Path,
    task_name: str,
    task_result: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not isinstance(task_result, dict) or task_result.get("status") != "ok":
        return None
    predictions = task_result.get("predictions")
    if not isinstance(predictions, list):
        return None

    task_key = str(task_name).strip().lower()
    error_rows = collect_error_predictions(task_key, predictions)
    debug_dir = ensure_dir(Path(output_root) / "debug")
    debug_path = debug_dir / f"{task_key}_error_samples.jsonl"
    write_jsonl(debug_path, error_rows)

    summary = {
        "task": task_key,
        "num_predictions": len(predictions),
        "num_error_samples": len(error_rows),
        "path": str(debug_path),
    }
    write_json(debug_dir / f"{task_key}_error_samples_summary.json", summary)
    return summary
