"""Materialize the C5 dual-pole safety dataset (PAN-format) from HuggingFace.

Harmful pole : declare-lab/HarmfulQA  (bare `question` + native refusal mined from
               `blue_conversations`; ~80% of questions ship a genuine refusal).
Harmless pole: HuggingFaceH4/no_robots (instruction + native human assistant reply).

Outputs two PAN-schema JSONLs (train + test) plus a `.summary.json`:
    {id, pan_split, source_dataset, label, user_text, target_response,
     accept_response, rejected_response, messages, category, source}

The bare question is the prompt (vanilla single-turn) so the SVD intent axis stays
clean; the multi-turn CoU is used only to mine the refusal target. HarmfulQA is a
benchmark -> train-only; never report it as our eval.

Usage:
    python scripts/build_c5_dataset.py --output-dir data/processed/c5 \
        --n-per-pole 1500 --refusal-fallback drop
    # refusal-fallback: "drop" = pure-native refusals only (recommended, ~1.5k);
    #                   "template" = pad no-refusal rows with pick_refusal_template.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.c5_dataset import build_c5_dataset
from src.utils.io import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed/c5")
    parser.add_argument("--harmful-hf-id", default="declare-lab/HarmfulQA")
    parser.add_argument("--benign-hf-id", default="HuggingFaceH4/no_robots")
    parser.add_argument("--benign-split", default="train_sft")
    parser.add_argument("--n-per-pole", type=int, default=0, help="0 = min(usable harmful, usable benign)")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--refusal-fallback",
        choices=["drop", "template"],
        default="drop",
        help="how to handle HarmfulQA rows whose blue conversation has no native refusal",
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset  # type: ignore[import-not-found]

    cache = args.cache_dir or None
    harmful = list(load_dataset(args.harmful_hf_id, split="train", cache_dir=cache))
    try:
        benign = list(load_dataset(args.benign_hf_id, split=args.benign_split, cache_dir=cache))
    except (ValueError, KeyError):
        benign = list(load_dataset(args.benign_hf_id, split="train", cache_dir=cache))

    result = build_c5_dataset(
        harmful,
        benign,
        n_per_pole=int(args.n_per_pole),
        test_fraction=float(args.test_fraction),
        refusal_fallback=args.refusal_fallback,
        seed=int(args.seed),
    )

    out_dir = ensure_dir(args.output_dir)
    train_path = out_dir / "c5_train_set.jsonl"
    test_path = out_dir / "c5_test_set.jsonl"
    write_jsonl(train_path, result["train"])
    write_jsonl(test_path, result["test"])
    (out_dir / "c5_dataset.summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    print("C5 dataset written:")
    print(f"  train -> {train_path}  ({result['summary']['n_train']} rows)")
    print(f"  test  -> {test_path}  ({result['summary']['n_test']} rows)")
    print("  summary:", json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
