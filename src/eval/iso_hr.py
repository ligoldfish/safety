"""Pure helpers for the iso-HR (matched harmful-refusal) comparison.

Selection picks, per baseline, the checkpoint whose VALIDATION HR is closest to a
target (ours' final-epoch val HR) -- using val HR ONLY, never OR, so we neither
cherry-pick on over-refusal nor select checkpoints on the test set. The matched
flag is then computed on the TEST HR vs ours' test HR by the caller.

torch-free: reads JSON only, so it is unit-testable anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def select_iso_hr(
    checkpoints: List[Dict[str, Any]], target_hr: float, epsilon: float, *, selection_split: str = "validation"
) -> Optional[Dict[str, Any]]:
    """Pick the checkpoint with val HR closest to ``target_hr``.

    ``checkpoints``: list of dicts each carrying at least ``ckpt`` and ``val_hr``.
    Selection uses ``val_hr`` ONLY (OR is never consulted). Returns the chosen dict
    augmented with ``delta_hr`` (= val_hr − target_hr) and ``matched`` (|delta| ≤ ε),
    or None if no checkpoint has a usable val HR. ``target_hr``/``epsilon`` are in the
    same units as ``val_hr`` (fractions in [0,1]).
    """

    if str(selection_split).strip().lower() not in {"validation", "val"}:
        raise ValueError("ISO-HR checkpoint selection must use validation only")
    usable = [c for c in checkpoints if _to_float(c.get("val_hr")) is not None]
    if not usable:
        return None
    best = min(usable, key=lambda c: abs(float(c["val_hr"]) - float(target_hr)))
    delta = float(best["val_hr"]) - float(target_hr)
    chosen = dict(best)
    chosen["delta_hr"] = delta
    chosen["matched"] = abs(delta) <= float(epsilon) + 1e-9  # FP-robust boundary
    return chosen


def read_val_hr_or(train_dir: str | Path) -> Dict[str, Dict[str, Optional[float]]]:
    """``val_metrics.json`` → ``{ckpt_key: {"val_hr": .., "val_or": ..}}`` (epoch_* + step_*)."""

    path = Path(train_dir) / "val_metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        out[str(key)] = {
            "val_hr": _to_float(metrics.get("harmful_refusal_rate")),
            "val_or": _to_float(metrics.get("harmless_over_refusal_rate")),
        }
    return out


def read_test_hr_or(
    train_dir: str | Path, ckpt_key: str
) -> Tuple[Optional[float], Optional[float]]:
    """``eval_suite/<ckpt_key>/summary.json`` → (test_hr, test_or), or (None, None)."""

    path = Path(train_dir) / "eval_suite" / ckpt_key / "summary.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    pan = (data.get("results") or {}).get("pan") or {}
    return _to_float(pan.get("harmful_refusal_rate")), _to_float(
        pan.get("harmless_over_refusal_rate")
    )


def last_epoch_key(val_hr_or: Dict[str, Any]) -> Optional[str]:
    """The highest-numbered ``epoch_<N>`` key (= ours' final epoch)."""

    epochs: List[Tuple[int, str]] = []
    for key in val_hr_or:
        if str(key).startswith("epoch_"):
            try:
                epochs.append((int(str(key).split("_", 1)[1]), str(key)))
            except (ValueError, IndexError):
                continue
    return max(epochs)[1] if epochs else None
