"""Post-hoc WildGuard judging of saved eval generations.

Reads an already-produced ``pan_results.json`` (which stores per-generation
``id``/``label``/``final_text``/``is_refusal`` but NOT the prompt), reconstructs the
``(prompt, response)`` pair by joining on ``id`` against the dataset's eval test JSONL,
runs the WildGuard judge once over everything, and writes ``judge_results.json`` with
ASR / over-refusal / refusal-rate + keyword-vs-WildGuard agreement. Optionally merges the
scalar judge metrics into the run's ``summary.json`` under ``results.pan``.

Fully read-only over existing artifacts -- no change to generation/training code. WildGuard
is loaded ONCE and applied uniformly across every dataset (the same-measuring-stick setup
mirrored from Tulu-3 / allenai/safety-eval).

Example (single result file):
    python scripts/22_judge_generations.py \
        --pan-results ../outputs/baselines/sft_qwen35_08b_coconot_npu/eval_suite/epoch_003/pan_results.json \
        --test-jsonl  ../data/processed/eval/coconot_test.jsonl \
        --judge-model ../models/wildguard \
        --runtime-backend npu --runtime-device npu:0 \
        --merge-summary

Example (whole eval_suite, all epochs):
    python scripts/22_judge_generations.py \
        --eval-suite-dir ../outputs/baselines/sft_qwen35_08b_coconot_npu/eval_suite \
        --test-jsonl ../data/processed/eval/coconot_test.jsonl \
        --judge-model ../models/wildguard --runtime-backend npu --runtime-device npu:0 --merge-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.llm_judge import WildGuardJudge, aggregate_judge_metrics
from src.utils.io import read_jsonl, write_json
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pan-results", type=str, help="A single pan_results.json to judge.")
    src.add_argument(
        "--eval-suite-dir",
        type=str,
        help="An eval_suite/ dir; judges every epoch_*/pan_results.json under it.",
    )
    parser.add_argument(
        "--test-jsonl",
        type=str,
        required=True,
        help="The dataset's eval test JSONL (id -> prompt join source).",
    )
    parser.add_argument("--judge-model", type=str, required=True, help="Path to allenai/wildguard.")
    parser.add_argument("--runtime-backend", type=str, default="")
    parser.add_argument("--runtime-device", type=str, default="")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output judge_results.json path (default: alongside each pan_results.json).",
    )
    parser.add_argument(
        "--merge-summary",
        action="store_true",
        help="Also merge llm_judge_* scalars into the sibling summary.json results.pan.",
    )
    return parser.parse_args()


def _first_user_text(record: Dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if str(message.get("role", "")).lower() == "user":
            return str(message.get("content", "")).strip()
    return ""


def _load_id_to_prompt(test_jsonl: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for record in read_jsonl(test_jsonl):
        rid = str(record.get("id", ""))
        prompt = _first_user_text(record)
        if rid and prompt:
            mapping[rid] = prompt
    return mapping


def _judge_one(
    pan_results: Path,
    id_to_prompt: Dict[str, str],
    judge: WildGuardJudge,
    out_path: Optional[Path],
    merge_summary: bool,
) -> Dict[str, Any]:
    data = json.loads(pan_results.read_text(encoding="utf-8"))
    generations: List[Dict[str, Any]] = data.get("generations") or []

    pairs = []
    items = []
    unmatched: List[str] = []
    for gen in generations:
        rid = str(gen.get("id", ""))
        prompt = id_to_prompt.get(rid)
        if prompt is None:
            unmatched.append(rid)
            continue
        response = str(gen.get("final_text") or gen.get("generated_text") or "")
        pairs.append((prompt, response))
        items.append(
            {
                "id": rid,
                "label": str(gen.get("label", "")),
                "keyword_is_refusal": gen.get("is_refusal"),
            }
        )

    verdicts = judge.score(pairs) if pairs else []
    for item, verdict in zip(items, verdicts):
        item["verdict"] = verdict

    metrics = aggregate_judge_metrics(items)
    metrics["judge_model"] = str(judge.model_path)
    metrics["num_unmatched_ids"] = len(unmatched)
    metrics["num_generations"] = len(generations)

    payload = {
        **metrics,
        "pan_results": str(pan_results),
        "test_jsonl_prompt_count": len(id_to_prompt),
        "unmatched_ids_sample": unmatched[:20],
        "generations": [
            {
                "id": item["id"],
                "label": item["label"],
                "keyword_is_refusal": item.get("keyword_is_refusal"),
                "wildguard_prompt_harmful": item["verdict"].prompt_harmful,
                "wildguard_response_refusal": item["verdict"].response_refusal,
                "wildguard_response_harmful": item["verdict"].response_harmful,
            }
            for item in items
        ],
    }
    target = out_path or pan_results.with_name("judge_results.json")
    write_json(target, payload)

    if merge_summary:
        summary_path = pan_results.with_name("summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            results = summary.setdefault("results", {})
            pan = results.setdefault("pan", {})
            for key in (
                "llm_judge_asr",
                "llm_judge_over_refusal",
                "llm_judge_refusal_rate",
                "judge_keyword_agreement",
                "judge_cohen_kappa",
                "judge_parse_rate",
                "judge_num_harmful_scored",
                "judge_num_harmless_scored",
            ):
                pan[key] = metrics[key]
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print(
        f"[judge] {pan_results.name}: ASR={metrics['llm_judge_asr']:.4f} "
        f"over_refusal={metrics['llm_judge_over_refusal']:.4f} "
        f"refusal_rate={metrics['llm_judge_refusal_rate']:.4f} "
        f"kw_agreement={metrics['judge_keyword_agreement']} "
        f"kappa={metrics['judge_cohen_kappa']} "
        f"parse_rate={metrics['judge_parse_rate']:.3f} unmatched={len(unmatched)} -> {target}",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    test_jsonl = Path(args.test_jsonl)
    if not test_jsonl.exists():
        raise SystemExit(f"--test-jsonl not found: {test_jsonl}")
    id_to_prompt = _load_id_to_prompt(test_jsonl)
    if not id_to_prompt:
        raise SystemExit(f"No id->prompt pairs parsed from {test_jsonl}")

    if args.eval_suite_dir:
        suite = Path(args.eval_suite_dir)
        targets = sorted(suite.glob("epoch_*/pan_results.json"))
        if not targets:
            raise SystemExit(f"No epoch_*/pan_results.json under {suite}")
    else:
        targets = [Path(args.pan_results)]
        if not targets[0].exists():
            raise SystemExit(f"--pan-results not found: {targets[0]}")

    judge = WildGuardJudge(
        model_path=args.judge_model,
        runtime_backend=args.runtime_backend,
        runtime_device=args.runtime_device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    judge.load()

    out_override = Path(args.out) if args.out else None
    for pan_results in targets:
        _judge_one(pan_results, id_to_prompt, judge, out_override, args.merge_summary)


if __name__ == "__main__":
    main()
