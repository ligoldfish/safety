"""Build per-baseline harmful/harmless test JSONLs.

Each safety baseline trains on its own corpus, so its evaluation set must
also live inside the same dataset family to avoid cross-corpus leakage.
This script materializes one ``data/processed/eval/<baseline>_test.jsonl``
per baseline, in the same schema as PAN's ``pan_test_set.jsonl``:

    {"id": str, "label": "harmful" | "harmless",
     "messages": [{"role": "system"/"user", "content": str}, ...]}

So that ``scripts/12_eval_baseline_suite.py`` can consume the JSONL via
``cfg.datasets.pan.path`` without any further code changes.

Per-baseline mappings:

* ``pan``                  -> existing ``pan_test_set.jsonl`` (no-op).
* ``beavertails``          -> 30k_train held-out (~10%) excluding the
                              ids written into the training JSONL.
                              ``is_safe=True -> harmless``,
                              ``is_safe=False -> harmful``. Prompts are
                              de-duplicated.
* ``tulu3_safety``         -> WildGuardTest (``prompt_harm_label``)
                              + WildJailbreak eval (``data_type``).
                              Native binary on both. CoCoNot contrast is
                              passed via launcher ``--safety-eval-datasets``.
* ``safety_tuned_llamas``  -> STL held-out (~10%) labelled harmful, plus
                              alpaca_small held-out (~10%) labelled
                              harmless. Both come from the same upstream
                              ``vinid/safety-tuned-llamas`` clone.

Usage:
    python scripts/21_build_baseline_eval_jsonls.py --baseline all
    python scripts/21_build_baseline_eval_jsonls.py --baseline tulu3_safety --force-rebuild
    python scripts/21_build_baseline_eval_jsonls.py --baseline wildjailbreak --max-eval-samples 2000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.safety_datasets import (
    DEFAULT_TULU3_SAFETY_SOURCES,
    build_beavertails_category_records,
    build_beavertails_records,
    build_hh_rlhf_records,
    build_safety_tuned_llamas_records,
    build_wildguardmix_records,
    build_wildjailbreak_records,
)
from src.data.safety_eval_datasets import (
    SafetyEvalExample,
    load_wildguard_test,
    load_wildjailbreak_eval,
)
from src.data.template_qwen import DEFAULT_SYSTEM_PROMPT, build_qwen_messages
from src.utils.io import ensure_dir, read_jsonl, write_jsonl


SUPPORTED_BASELINES = (
    "pan",
    "beavertails",
    "tulu3_safety",
    "safety_tuned_llamas",
    "wildjailbreak",
    "wildguardmix",
    "hh_rlhf",
    "beavertails_category",
)
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-baseline harmful/harmless test JSONLs."
    )
    parser.add_argument(
        "--baseline",
        choices=("all", *SUPPORTED_BASELINES),
        default="all",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the test JSONLs even when they already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for held-out splits (BeaverTails / STL).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.1,
        help="Fraction of records to hold out as test for in-domain baselines.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional HF datasets cache dir.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Cap for new in-domain safety tests (0 = full official test).",
    )
    parser.add_argument(
        "--max-eval-samples-per-label",
        type=int,
        default=0,
        help="Per-label cap for new in-domain safety tests (0 = infer from max-eval-samples).",
    )
    parser.add_argument(
        "--eval-full",
        action="store_true",
        help="Use full available test split for new baselines. This is already the default.",
    )
    parser.add_argument(
        "--stl-repo-path",
        default=str(PROJECT_ROOT / "external" / "safety-tuned-llamas"),
        help="Path to the cloned vinid/safety-tuned-llamas repo (or a directory "
        "containing safety_only_data_Instructions.json + alpaca_small.json).",
    )
    return parser.parse_args()


def _eval_jsonl_path(baseline: str) -> Path:
    return EVAL_DIR / f"{baseline}_test.jsonl"


def _stamp_pan() -> Path:
    """PAN test set is the canonical reference; no-op build."""

    src = PROJECT_ROOT / "data" / "processed" / "pan_test_set.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"PAN test set missing at {src}; run scripts/00_prepare_data.py first."
        )
    out = _eval_jsonl_path("pan")
    if out.resolve() == src.resolve():
        return src
    ensure_dir(out.parent)
    shutil.copyfile(src, out)
    return out


def _build_beavertails_test(
    *,
    seed: int,
    holdout_fraction: float,
    cache_dir: Optional[str],
) -> Path:
    out = _eval_jsonl_path("beavertails")
    train_jsonl = PROJECT_ROOT / "data" / "processed" / "safety" / "beavertails_30k_train.jsonl"
    train_ids: set[str] = set()
    if train_jsonl.exists():
        train_ids = {row.get("id", "") for row in read_jsonl(train_jsonl)}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_jsonl = Path(tmpdir) / "bt_full.jsonl"
        all_records = build_beavertails_records(
            output_path=tmp_jsonl,
            split="30k_train",
            cache_dir=cache_dir,
            dedup_prompts=True,
            label_strategy="is_safe",
        )

    held_out = [r for r in all_records if r["id"] not in train_ids]
    if not held_out:
        # Fallback: training jsonl unavailable -- use full deduped set, but
        # still hold out a fraction so the eval set is distinct.
        held_out = list(all_records)
    rng = random.Random(seed)
    rng.shuffle(held_out)
    n = max(1, int(round(len(held_out) * holdout_fraction)))
    sliced = held_out[:n]

    eval_records: List[Dict[str, Any]] = []
    for r in sliced:
        label = str(r.get("label", "")).lower()
        if label not in {"harmful", "harmless"}:
            # Builder under category_any always emits harmful/harmless;
            # fall back to is_safe flip just in case the strategy was
            # toggled via env override.
            label = "harmless" if bool(r.get("is_safe", False)) else "harmful"
        eval_records.append(
            {
                "id": f"bt_test_{r['id']}",
                "label": label,
                "messages": list(r["messages"]),
                "source": r.get("source"),
                "is_safe": bool(r.get("is_safe", False)),
                "category": r.get("category"),
            }
        )
    write_jsonl(out, eval_records)
    return out


def _classify_wildguard_label(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "harmful"
    if text in {"unharmful", "benign", "harmless", "safe"}:
        return "harmless"
    if "benign" in text or "harmless" in text or "unharm" in text:
        return "harmless"
    return "harmful"


def _classify_wildjailbreak_label(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "harmful"
    # benign and adversarial_benign both signal that the prompt itself is
    # benign (the adversarial framing is style, not content).
    if "benign" in text:
        return "harmless"
    return "harmful"


def _examples_to_records(
    examples: List[SafetyEvalExample],
    *,
    id_prefix: str,
    classifier,
) -> List[Dict[str, Any]]:
    seen_prompts: set[str] = set()
    out: List[Dict[str, Any]] = []
    for ex in examples:
        prompt = (ex.prompt or "").strip()
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        label = classifier(ex.label)
        out.append(
            {
                "id": f"{id_prefix}_{ex.sample_id}",
                "label": label,
                "messages": build_qwen_messages(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT),
                "source": ex.source,
                "category": ex.category,
                "raw_label": ex.label,
            }
        )
    return out


def _build_tulu3_test(*, cache_dir: Optional[str]) -> Path:
    out = _eval_jsonl_path("tulu3_safety")
    wgt_examples = load_wildguard_test(cache_dir=cache_dir)
    wjb_examples = load_wildjailbreak_eval(cache_dir=cache_dir)
    records = _examples_to_records(
        wgt_examples,
        id_prefix="wgt",
        classifier=_classify_wildguard_label,
    )
    records.extend(
        _examples_to_records(
            wjb_examples,
            id_prefix="wjb",
            classifier=_classify_wildjailbreak_label,
        )
    )
    if not records:
        raise RuntimeError(
            "tulu3_safety eval jsonl is empty -- check that WildGuardTest "
            "and WildJailbreak eval are accessible."
        )
    write_jsonl(out, records)
    return out


def _build_safety_tuned_llamas_test(
    *,
    repo_path: Path,
    seed: int,
    holdout_fraction: float,
) -> Path:
    out = _eval_jsonl_path("safety_tuned_llamas")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_jsonl = Path(tmpdir) / "stl_full.jsonl"
        records = build_safety_tuned_llamas_records(
            output_path=tmp_jsonl,
            repo_or_data_path=repo_path,
            include_harmless_contrast=True,
        )

    harmful = [r for r in records if r.get("dataset") == "safety_tuned_llamas"]
    harmless = [r for r in records if r.get("dataset") == "safety_tuned_llamas_harmless"]
    if not harmful:
        raise RuntimeError("STL harmful records missing -- builder produced zero.")
    if not harmless:
        raise RuntimeError(
            "STL harmless contrast missing -- ensure alpaca_small.json is in "
            f"{repo_path}/ or {repo_path}/data/."
        )

    rng_h = random.Random(seed)
    rng_h.shuffle(harmful)
    rng_b = random.Random(seed + 1)
    rng_b.shuffle(harmless)
    n_harmful = max(1, int(round(len(harmful) * holdout_fraction)))
    n_harmless = max(1, int(round(len(harmless) * holdout_fraction)))

    eval_records: List[Dict[str, Any]] = []
    for r in harmful[:n_harmful]:
        eval_records.append(
            {
                "id": f"stl_test_{r['id']}",
                "label": "harmful",
                "messages": list(r["messages"]),
                "source": r.get("source"),
            }
        )
    for r in harmless[:n_harmless]:
        eval_records.append(
            {
                "id": f"stl_test_{r['id']}",
                "label": "harmless",
                "messages": list(r["messages"]),
                "source": r.get("source"),
            }
        )
    write_jsonl(out, eval_records)
    return out


def _build_new_dataset_test(
    baseline: str,
    *,
    seed: int,
    cache_dir: Optional[str],
    eval_subset_mode: bool,
    max_eval_samples: int,
    max_eval_samples_per_label: int,
) -> Path:
    out = _eval_jsonl_path(baseline)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_jsonl = Path(tmpdir) / f"{baseline}_train.jsonl"
        common = {
            "eval_output_path": out,
            "eval_subset_mode": bool(eval_subset_mode),
            "max_eval_samples": int(max_eval_samples),
            "max_eval_samples_per_label": int(max_eval_samples_per_label),
            "seed": int(seed),
            "cache_dir": cache_dir,
        }
        if baseline == "wildjailbreak":
            build_wildjailbreak_records(
                output_path=tmp_jsonl,
                train_subset_mode=True,
                max_train_samples=20000,
                max_train_samples_per_label=10000,
                **common,
            )
        elif baseline == "wildguardmix":
            build_wildguardmix_records(
                output_path=tmp_jsonl,
                train_subset_mode=True,
                max_train_samples=20000,
                max_train_samples_per_label=10000,
                **common,
            )
        elif baseline == "hh_rlhf":
            build_hh_rlhf_records(
                output_path=tmp_jsonl,
                train_subset_mode=True,
                max_train_samples=20000,
                max_train_samples_per_label=10000,
                **common,
            )
        elif baseline == "beavertails_category":
            build_beavertails_category_records(
                output_path=tmp_jsonl,
                train_subset_mode=False,
                max_train_samples=0,
                max_train_samples_per_label=0,
                **common,
            )
        else:  # pragma: no cover -- guarded by caller
            raise ValueError(f"Unsupported new dataset baseline: {baseline}")
    return out


def main() -> None:
    args = parse_args()
    targets = list(SUPPORTED_BASELINES) if args.baseline == "all" else [args.baseline]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"force_rebuild": args.force_rebuild, "outputs": {}}
    cache_dir = args.cache_dir or None

    for baseline in targets:
        out_path = _eval_jsonl_path(baseline)
        if out_path.exists() and not args.force_rebuild:
            summary["outputs"][baseline] = {
                "path": str(out_path.resolve()),
                "status": "skipped (exists; pass --force-rebuild to overwrite)",
            }
            continue
        if baseline == "pan":
            written = _stamp_pan()
        elif baseline == "beavertails":
            written = _build_beavertails_test(
                seed=int(args.seed),
                holdout_fraction=float(args.holdout_fraction),
                cache_dir=cache_dir,
            )
        elif baseline == "tulu3_safety":
            written = _build_tulu3_test(cache_dir=cache_dir)
        elif baseline == "safety_tuned_llamas":
            written = _build_safety_tuned_llamas_test(
                repo_path=Path(args.stl_repo_path).expanduser().resolve(),
                seed=int(args.seed),
                holdout_fraction=float(args.holdout_fraction),
            )
        elif baseline in {
            "wildjailbreak",
            "wildguardmix",
            "hh_rlhf",
            "beavertails_category",
        }:
            written = _build_new_dataset_test(
                baseline,
                seed=int(args.seed),
                cache_dir=cache_dir,
                eval_subset_mode=not bool(args.eval_full) and int(args.max_eval_samples) != 0,
                max_eval_samples=int(args.max_eval_samples),
                max_eval_samples_per_label=int(args.max_eval_samples_per_label),
            )
        else:  # pragma: no cover -- guarded by argparse choices
            raise ValueError(f"Unknown baseline: {baseline}")

        rows = read_jsonl(written)
        labels = [str(row.get("label", "")) for row in rows]
        out_entry: Dict[str, Any] = {
            "path": str(written.resolve()),
            "n_records": len(rows),
            "n_harmful": sum(1 for label in labels if label == "harmful"),
            "n_harmless": sum(1 for label in labels if label == "harmless"),
        }
        if baseline == "tulu3_safety":
            buckets: Dict[str, Dict[str, int]] = {}
            for row in rows:
                bucket = str(row.get("source", "")) or "unknown"
                label = str(row.get("label", ""))
                buckets.setdefault(bucket, {"harmful": 0, "harmless": 0})
                if label in buckets[bucket]:
                    buckets[bucket][label] += 1
            out_entry["n_by_source"] = buckets
        summary["outputs"][baseline] = out_entry

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
