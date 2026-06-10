"""Curate the Phase 1 contrast subset for SemAlign SVD.

Reads ``<processed_dir>/train_set.jsonl`` (full safety SFT data) and
writes ``<processed_dir>/alignment_set.jsonl`` (curated harmful/harmless
subset, ≤ 5k rows) along with ``curation_summary.json``.

Pipeline (round-2: teacher-confidence annotation + filter removed):
    1. Pre-filter (mode=strict, per-baseline; e.g. WJB/WGM keep vanilla only).
    2. Universal length filter (prompt token count ∈ [length_min, length_max]).
    3. Stratified sample to n_per_side per label (the balance step).

The teacher-confidence filter (>=0.7) was dropped: its magic threshold +
"missing -> keep" behaviour was inconsistent with the length filter, and the
annotation forward pass it required is no longer run. Curation is now native
pre-filter + length hygiene + balance only.

For ``mode=off``, the script copies ``train_set.jsonl`` → ``alignment_set.jsonl``
byte-identically and skips the teacher forward entirely.

Usage:
    python scripts/19b_curate_phase1_subset.py \
        --baseline wildjailbreak \
        --processed-dir data/processed/safety_full_wildjailbreak \
        --mode auto \
        --teacher-path models/Qwen3.5-9B \
        --runtime-backend npu --runtime-device npu:0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.curation import (
    VALID_MODES,
    curate_phase1_subset,
    resolve_curation_mode,
)
from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce a curated harmful/harmless Phase 1 contrast subset "
            "(alignment_set.jsonl) from the full SFT train_set.jsonl."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline key (wildjailbreak, wildguardmix, pan, ...). Controls "
        "the per-baseline pre-filter and default mode.",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Directory containing train_set.jsonl; alignment_set.jsonl + "
        "curation_summary.json will be written here.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", *VALID_MODES),
        help="Curation mode. 'auto' (default) looks up DEFAULT_MODE per baseline.",
    )
    parser.add_argument("--n-per-side", type=int, default=2500)
    parser.add_argument("--length-min", type=int, default=5)
    parser.add_argument("--length-max", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument(
        "--teacher-path",
        default="",
        help="HF path to the teacher (e.g. models/Qwen3.5-9B). Required unless "
        "--skip-teacher-confidence is set or --mode=off.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--runtime-backend", default="", help="cuda|npu|ppu|cpu (optional)")
    parser.add_argument("--runtime-device", default="")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--skip-teacher-confidence",
        action="store_true",
        help="Skip teacher forward; rely on whatever confidence is already "
        "stored in metadata.teacher_first_gen_top1_prob. Records missing "
        "the field are treated as confidence 1.0.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Overwrite alignment_set.jsonl even when it already exists.",
    )
    parser.add_argument(
        "--train-filename",
        default="train_set.jsonl",
        help="Filename for the full Phase F training data inside --processed-dir.",
    )
    parser.add_argument(
        "--alignment-filename",
        default="alignment_set.jsonl",
        help="Filename to write the curated subset to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir).resolve()
    train_path = processed_dir / args.train_filename
    alignment_path = processed_dir / args.alignment_filename
    summary_path = processed_dir / "curation_summary.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train_set not found at {train_path}; run "
            "scripts/20_split_safety_for_semalign.py (or "
            "scripts/00_prepare_data.py for PAN) first."
        )

    mode = resolve_curation_mode(args.baseline, args.mode)
    print(f"[19b] baseline={args.baseline} mode={mode}")

    if alignment_path.exists() and not args.force_rebuild:
        # Guard against the "20_split wrote alignment_set (== train_set) but
        # a previous 19b crashed before overwriting it" race. The presence of
        # alignment_set.jsonl alone is NOT proof of a completed curation —
        # require a matching curation_summary.json as well.
        if not summary_path.exists():
            raise RuntimeError(
                f"[19b] {alignment_path} exists but {summary_path} is missing. "
                "This usually means a previous 19b run crashed before writing "
                "the summary (or 20_split wrote alignment_set as a copy of "
                "train_set and 19b never completed). Re-run with --force-rebuild "
                "to regenerate the curated subset."
            )
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"[19b] Failed to parse existing {summary_path} ({exc!r}). "
                "Re-run with --force-rebuild to regenerate."
            ) from exc
        existing_baseline = existing_summary.get("baseline")
        existing_mode = existing_summary.get("mode")
        if existing_baseline != args.baseline or existing_mode != mode:
            raise RuntimeError(
                f"[19b] Existing {summary_path} was produced for "
                f"baseline={existing_baseline!r} mode={existing_mode!r}, but "
                f"this run requested baseline={args.baseline!r} mode={mode!r}. "
                "Re-run with --force-rebuild to regenerate the curated subset "
                "for the requested configuration."
            )
        print(
            f"[19b] {alignment_path} already exists and curation_summary.json "
            f"matches (baseline={args.baseline}, mode={mode}); skipping (use "
            "--force-rebuild to overwrite)."
        )
        return

    if mode == "off":
        shutil.copyfile(train_path, alignment_path)
        summary = {
            "baseline": args.baseline,
            "mode": "off",
            "input_count": sum(1 for _ in train_path.open("r", encoding="utf-8")),
            "output_count": None,
            "note": "off-mode: alignment_set is a byte-identical copy of train_set.",
        }
        summary["output_count"] = summary["input_count"]
        write_json(summary_path, summary)
        print(f"[19b] off mode: copied {summary['input_count']} records to {alignment_path}")
        return

    print(f"[19b] Loading {train_path}")
    records = read_jsonl(train_path)
    print(f"[19b] Loaded {len(records)} records")

    tokenizer_for_length: Any = None
    if mode != "off":
        # Round-2: teacher-confidence annotation removed with the conf filter.
        # The tokenizer is still loaded for the length filter only.
        if args.teacher_path and not args.skip_teacher_confidence:
            from transformers import AutoTokenizer

            try:
                tokenizer_for_length = AutoTokenizer.from_pretrained(
                    args.teacher_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_fast=False,
                )
            except Exception as exc:
                print(f"[19b] Tokenizer load failed ({exc!r}); falling back to whitespace length.")
                tokenizer_for_length = None

    curated, summary = curate_phase1_subset(
        records,
        baseline_name=args.baseline,
        mode=mode,
        tokenizer=tokenizer_for_length,
        n_per_side=int(args.n_per_side),
        length_min=int(args.length_min),
        length_max=int(args.length_max),
        seed=int(args.seed),
    )

    # Validation: ensure the curated subset is usable for Phase 1 SVD.
    # An empty or one-sided alignment_set will cause downstream SVD to choke
    # silently — fail loudly here instead with the full summary so the user
    # can see exactly which filter dropped the records.
    final_counts = summary.get("stratified_final_counts", {}) or {}
    harmful_n = int(final_counts.get("harmful", 0) or 0)
    harmless_n = int(final_counts.get("harmless", 0) or 0)
    total_n = len(curated)
    min_total = 50
    warn_threshold = 200

    if total_n < min_total or harmful_n == 0 or harmless_n == 0:
        raise RuntimeError(
            "[19b] Curation produced an unusable Phase 1 subset "
            f"(total={total_n}, harmful={harmful_n}, harmless={harmless_n}; "
            f"require total>={min_total} and both sides>0). "
            "Phase 1 SVD requires balanced harmful/harmless pairs. "
            "Inspect the curation_summary below to see which filter dropped records:\n"
            f"{json.dumps(summary, ensure_ascii=False, indent=2)}"
        )

    if total_n < warn_threshold:
        print(
            f"[19b][WARN] Curated subset is small (total={total_n}, "
            f"harmful={harmful_n}, harmless={harmless_n}); "
            f"Phase 1 SVD may be unstable below {warn_threshold} samples.",
            file=sys.stderr,
        )

    write_jsonl(alignment_path, curated)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[19b] Wrote curated set: {alignment_path} ({len(curated)} records)")


if __name__ == "__main__":
    main()
