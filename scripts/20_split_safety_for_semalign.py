"""Split a materialized safety SFT JSONL into PAN-style splits for SemAlign.

The 00→11 SemAlign pipeline (extract hidden states, analyze layers, build
safe subspace, decompose semantics, recompose student targets, train) is
keyed off five JSONL files inside ``processed_dir``:

    alignment_set.jsonl       — Phase 1 training contrast set (harmful + harmless)
    analysis_val_set.jsonl    — Phase 1 validation contrast set
    sanity_test_set.jsonl     — Sanity-check held-out set
    pan_train_set.jsonl       — Hidden-state extraction source (= alignment_set)
    pan_test_set.jsonl        — Final transfer test set

For SemAlign to compute a meaningful "safe" direction in step 03 the train
set MUST contain BOTH ``harmful`` and ``harmless`` labels. Safety SFT
corpora differ:

* ``beavertails`` already carries an ``is_safe`` flag per record, so we
  map ``is_safe=True → harmless`` and ``is_safe=False → harmful``.
* ``tulu3_safety`` and ``safety_tuned_llamas`` are pure-harmful sets, so
  we have to inject harmless contrast samples from the existing PAN
  ``alignment_set.jsonl`` before splitting.

The transfer test set ``pan_test_set.jsonl`` is preserved as-is because
the goal is to measure whether safety-data-trained students still refuse
PAN-style attacks.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a materialized safety SFT JSONL into PAN-style splits "
            "consumable by the 00→11 SemAlign pipeline."
        )
    )
    parser.add_argument(
        "--safety-jsonl",
        required=True,
        help="Path to the materialized safety SFT JSONL written by "
        "scripts/19_prepare_safety_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the five PAN-style JSONL files into.",
    )
    parser.add_argument(
        "--pan-processed-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory holding the canonical PAN splits, used to source "
        "harmless contrast (when needed) and to copy pan_test_set.jsonl.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of the combined contrast set used as analysis_val.",
    )
    parser.add_argument(
        "--sanity-fraction",
        type=float,
        default=0.05,
        help="Fraction of the combined contrast set used as sanity_test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2042,
        help="Shuffle seed for reproducible splits.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt expected by Phase 1 record schema.",
    )
    return parser.parse_args()


def _normalize_record(
    raw: Dict[str, Any],
    *,
    label: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Coerce a record into the PAN training schema expected by Phase 1.

    Required fields: ``id``, ``label``, ``messages`` (with system + user),
    ``target_response`` (assistant target). PAN records also carry
    ``user_text``, ``accept_response`` and ``rejected_response`` but those
    are not consumed by 01-08; we still populate them so the merged set
    matches the format on disk.
    """

    messages = list(raw.get("messages") or [])
    if not messages or str(messages[0].get("role", "")).lower() != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages
    user_text = ""
    for message in messages:
        if str(message.get("role", "")).lower() == "user":
            user_text = str(message.get("content", ""))
            break
    target_response = str(raw.get("target_response") or "").strip()
    if not target_response:
        # Last-ditch: lift the assistant message if present.
        for message in reversed(messages):
            if str(message.get("role", "")).lower() == "assistant":
                target_response = str(message.get("content", ""))
                break
    accept_response = str(raw.get("accept_response") or "")
    rejected_response = str(raw.get("rejected_response") or target_response)
    return {
        "id": str(raw.get("id") or ""),
        "label": label,
        "user_text": user_text,
        "messages": [m for m in messages if str(m.get("role", "")).lower() != "assistant"],
        "target_response": target_response,
        "accept_response": accept_response,
        "rejected_response": rejected_response,
        "source_dataset": str(raw.get("dataset") or raw.get("source") or ""),
    }


def _label_safety_record(record: Dict[str, Any]) -> str:
    """BeaverTails carries an explicit ``is_safe`` flag; everything else
    is treated as harmful (jailbreak / safety SFT prompts)."""

    if "is_safe" in record:
        return "harmless" if bool(record["is_safe"]) else "harmful"
    return "harmful"


def _load_pan_harmless_records(
    pan_processed_dir: Path,
    *,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    alignment_path = pan_processed_dir / "alignment_set.jsonl"
    if not alignment_path.exists():
        raise FileNotFoundError(
            "Need PAN alignment_set.jsonl as the harmless contrast source for "
            "safety baselines that lack a harmless half. "
            f"Expected at {alignment_path}; run scripts/00_prepare_data.py "
            "against a PAN config first to materialize it."
        )
    rows = read_jsonl(alignment_path)
    harmless = [
        _normalize_record(row, label="harmless", system_prompt=system_prompt)
        for row in rows
        if str(row.get("label", "")).lower() == "harmless"
    ]
    if not harmless:
        raise RuntimeError(
            f"PAN alignment_set.jsonl at {alignment_path} contains no "
            "harmless rows. Cannot inject contrast for safety-only corpora."
        )
    return harmless


def _split(
    records: List[Dict[str, Any]],
    *,
    val_fraction: float,
    sanity_fraction: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(records)
    val_size = max(1, int(round(n * val_fraction))) if n > 0 else 0
    sanity_size = max(1, int(round(n * sanity_fraction))) if n > 0 else 0
    if val_size + sanity_size >= n:
        raise ValueError(
            f"Combined val+sanity sizes ({val_size}+{sanity_size}) exceed "
            f"total record count {n}; lower --val-fraction / --sanity-fraction."
        )
    val_split = records[:val_size]
    sanity_split = records[val_size : val_size + sanity_size]
    train_split = records[val_size + sanity_size :]
    return train_split, val_split, sanity_split


def main() -> None:
    args = parse_args()
    safety_path = Path(args.safety_jsonl).resolve()
    if not safety_path.exists():
        raise FileNotFoundError(f"Safety JSONL not found: {safety_path}")
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    pan_processed_dir = Path(args.pan_processed_dir).resolve()

    raw_records = read_jsonl(safety_path)
    safety_records = [
        _normalize_record(
            row,
            label=_label_safety_record(row),
            system_prompt=args.system_prompt,
        )
        for row in raw_records
    ]

    has_harmless = any(record["label"] == "harmless" for record in safety_records)
    has_harmful = any(record["label"] == "harmful" for record in safety_records)

    if not has_harmful:
        raise RuntimeError(
            "Safety JSONL contains no harmful-labeled records; cannot build "
            "a SemAlign contrast set."
        )

    if not has_harmless:
        injected = _load_pan_harmless_records(
            pan_processed_dir,
            system_prompt=args.system_prompt,
        )
        safety_records.extend(injected)

    rng = random.Random(int(args.seed))
    rng.shuffle(safety_records)

    train_split, val_split, sanity_split = _split(
        safety_records,
        val_fraction=float(args.val_fraction),
        sanity_fraction=float(args.sanity_fraction),
    )

    write_jsonl(output_dir / "alignment_set.jsonl", train_split)
    write_jsonl(output_dir / "analysis_val_set.jsonl", val_split)
    write_jsonl(output_dir / "sanity_test_set.jsonl", sanity_split)
    # pan_train_set is the hidden-state extraction source; SemAlign uses the
    # same record set as alignment_set for that step.
    write_jsonl(output_dir / "pan_train_set.jsonl", train_split)

    pan_test_src = pan_processed_dir / "pan_test_set.jsonl"
    pan_test_dst = output_dir / "pan_test_set.jsonl"
    if pan_test_src.exists():
        shutil.copyfile(pan_test_src, pan_test_dst)
    else:
        # Fall back to the sanity split so downstream scripts still find a
        # readable file. Transfer-test numbers will then be on safety data,
        # which is recorded explicitly in the summary below.
        write_jsonl(pan_test_dst, sanity_split)

    label_counts = {"harmful": 0, "harmless": 0}
    for record in train_split:
        label_counts[record["label"]] = label_counts.get(record["label"], 0) + 1

    summary = {
        "safety_jsonl": str(safety_path),
        "output_dir": str(output_dir),
        "pan_processed_dir": str(pan_processed_dir),
        "pan_test_copied": pan_test_src.exists(),
        "total_records": len(safety_records),
        "train_size": len(train_split),
        "val_size": len(val_split),
        "sanity_size": len(sanity_split),
        "train_label_counts": label_counts,
        "harmless_injected_from_pan": not has_harmless,
        "seed": int(args.seed),
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
