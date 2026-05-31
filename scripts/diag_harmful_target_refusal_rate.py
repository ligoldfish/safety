#!/usr/bin/env python
"""Diagnostic: what fraction of HARMFUL SFT targets are actually refusals?

Runs the SAME judge the eval uses (src.training.eval_utils.judge_harmful_response_safety
/ looks_like_refusal) over the `target_response` of every harmful record in a
prepared jsonl. This answers whether the harmful targets teach crisp refusals or
"safe-but-evasive" non-answers (ambiguous_non_refusal), which is what drags down
the data-dependent methods (sft / ours SFT component) on the refusal-strict eval.

Pure regex judge -> NO torch / NPU needed. Run on the training box (or anywhere
the jsonl is reachable):

    cd /root/safety
    python scripts/diag_harmful_target_refusal_rate.py \
        --jsonl data/processed/safety/hh_rlhf_20k_train.jsonl

Point --jsonl at whatever split you want to audit, e.g. the semalign alignment
set the "ours" run actually trained on:

    python scripts/diag_harmful_target_refusal_rate.py \
        --jsonl data/processed/safety_full_hh_rlhf/alignment_set.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the production judge directly by file path. Importing via the package
# (src.training) would trigger src/training/__init__.py -> torch, but eval_utils
# itself is pure stdlib, so the audit runs with or without torch installed.
_EVAL_UTILS = PROJECT_ROOT / "src" / "training" / "eval_utils.py"
_spec = importlib.util.spec_from_file_location("_eval_utils_standalone", _EVAL_UTILS)
_eu = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eu  # register so @dataclass can resolve annotations
_spec.loader.exec_module(_eu)
judge_harmful_response_safety = _eu.judge_harmful_response_safety
looks_like_refusal = _eu.looks_like_refusal


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _target_text(record: dict) -> str:
    for key in ("target_response", "rejected_response", "response", "target"):
        value = record.get(key)
        if value:
            return str(value)
    # Fall back to the last assistant message if present.
    messages = record.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def _target_source(record: dict) -> str:
    meta = record.get("metadata") or {}
    return str(meta.get("target_source") or record.get("source") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Prepared split jsonl to audit.")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--harmful-label", default="harmful")
    parser.add_argument("--harmless-label", default="harmless")
    parser.add_argument("--max", type=int, default=0, help="Cap records scanned (0 = all).")
    parser.add_argument("--show", type=int, default=4, help="Example targets to print per reason bucket.")
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"jsonl not found: {path}")

    harmful_reasons: Counter[str] = Counter()
    harmful_by_source: Counter[str] = Counter()
    refusal_by_source: Counter[str] = Counter()
    harmless_refusal_leak = 0
    n_harmful = 0
    n_harmless = 0
    n_refusal = 0
    n_safe = 0
    examples: dict[str, list[str]] = {}

    for idx, record in enumerate(_iter_jsonl(path)):
        if args.max and idx >= args.max:
            break
        label = str(record.get(args.label_field, ""))
        target = _target_text(record).strip()

        if label == args.harmful_label:
            n_harmful += 1
            source = _target_source(record)
            harmful_by_source[source] += 1
            judgment = judge_harmful_response_safety(target)
            harmful_reasons[judgment.reason] += 1
            n_refusal += int(judgment.is_refusal)
            n_safe += int(judgment.is_safe)
            if judgment.is_refusal:
                refusal_by_source[source] += 1
            bucket = examples.setdefault(judgment.reason, [])
            if len(bucket) < args.show:
                bucket.append(target[:200].replace("\n", " "))
        elif label == args.harmless_label:
            n_harmless += 1
            if looks_like_refusal(target):
                harmless_refusal_leak += 1

    def pct(num: int, den: int) -> str:
        return f"{(100.0 * num / den):.1f}%" if den else "n/a"

    print(f"=== Harmful-target refusal audit: {path} ===")
    print(f"harmful records      : {n_harmful}")
    print(f"harmless records     : {n_harmless}")
    print()
    print(f"harmful is_refusal   : {n_refusal}  ({pct(n_refusal, n_harmful)})   <- want HIGH")
    print(f"harmful is_safe      : {n_safe}  ({pct(n_safe, n_harmful)})")
    print(f"harmful NOT-refusal  : {n_harmful - n_refusal}  ({pct(n_harmful - n_refusal, n_harmful)})")
    print()
    print("harmful judge_reason breakdown (matches eval judge):")
    for reason, count in harmful_reasons.most_common():
        print(f"  {reason:24s} {count:7d}  ({pct(count, n_harmful)})")
    print()
    print("harmful refusal-rate by target_source:")
    for source, total in harmful_by_source.most_common():
        print(f"  {source:34s} refusal {refusal_by_source[source]:6d} / {total:6d}  ({pct(refusal_by_source[source], total)})")
    print()
    print(f"harmless side refusal-leak (over-refusal targets): {harmless_refusal_leak}  ({pct(harmless_refusal_leak, n_harmless)})")
    print()
    for reason in ("explicit_refusal", "ambiguous_non_refusal", "actionable_guidance"):
        if examples.get(reason):
            print(f"--- sample targets [{reason}] ---")
            for text in examples[reason]:
                print(f"  {text}")
            print()


if __name__ == "__main__":
    main()
