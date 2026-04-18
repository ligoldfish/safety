from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import (
    AdapterConfig,
    evaluate_code_generation,
    evaluate_gsm8k,
    evaluate_mcq,
    evaluate_pan,
    export_error_predictions,
    load_code_examples,
    load_eval_config,
    load_gsm8k_examples,
    load_mcq_examples,
    load_model_for_evaluation,
)
from src.baselines.eval import filter_records, placeholder_result
from src.training import load_records
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PAN + MMLU + GSM8K + HumanEval + MBPP benchmark evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_eval_qwen35_1b.yaml",
        help="Path to the baseline evaluation YAML config.",
    )
    parser.add_argument(
        "--adapter-manifest",
        type=str,
        default="",
        help="Optional LoRA manifest path overriding the config adapter section.",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=str,
        default="",
        help="Optional LoRA checkpoint path overriding the config adapter section.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory overriding the config output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    set_global_seed(cfg.seed)

    adapter = AdapterConfig(
        manifest_path=args.adapter_manifest or cfg.adapter.manifest_path,
        checkpoint_path=args.adapter_checkpoint or cfg.adapter.checkpoint_path,
    )
    output_root = ensure_dir(args.output_dir or cfg.output.output_root)
    logger, log_path = setup_stage_logger("12_eval_baseline_suite", output_root / "logs")
    log_kv(
        logger,
        "baseline_eval_setup",
        config_path=str(Path(args.config).resolve()),
        output_root=str(output_root),
        model_name=cfg.model.name,
        adapter_manifest=adapter.manifest_path,
        adapter_checkpoint=adapter.checkpoint_path,
        log_path=str(log_path),
    )

    tokenizer, model = load_model_for_evaluation(cfg.model, adapter)
    results: dict[str, dict] = {}
    debug_outputs: dict[str, dict] = {}

    pan_cfg = cfg.datasets.pan
    try:
        if not pan_cfg.enabled:
            results["pan"] = {"status": "disabled", "reason": "PAN evaluation disabled by config"}
        elif not pan_cfg.path or not Path(pan_cfg.path).exists():
            if pan_cfg.placeholder_ok:
                results["pan"] = placeholder_result(f"PAN path not found: {pan_cfg.path}")
            else:
                raise FileNotFoundError(f"PAN path not found: {pan_cfg.path}")
        else:
            pan_records = load_records(pan_cfg.path)
            if pan_cfg.shuffle:
                random.Random(cfg.seed).shuffle(pan_records)
            pan_records = filter_records(pan_records, pan_cfg.max_samples)
            results["pan"] = evaluate_pan(
                model,
                tokenizer,
                pan_records,
                max_length=cfg.runtime.max_length,
                max_new_tokens=pan_cfg.max_new_tokens,
            )
    except Exception as exc:
        if pan_cfg.placeholder_ok:
            results["pan"] = placeholder_result(str(exc))
        else:
            raise
    write_json(output_root / "pan_results.json", results["pan"])

    mmlu_cfg = cfg.datasets.mmlu
    try:
        if not mmlu_cfg.enabled:
            results["mmlu"] = {"status": "disabled", "reason": "MMLU evaluation disabled by config"}
        elif not mmlu_cfg.path or not Path(mmlu_cfg.path).exists():
            if mmlu_cfg.placeholder_ok:
                results["mmlu"] = placeholder_result(f"MMLU path not found: {mmlu_cfg.path}")
            else:
                raise FileNotFoundError(f"MMLU path not found: {mmlu_cfg.path}")
        else:
            mmlu_examples = load_mcq_examples(
                mmlu_cfg.path,
                split=mmlu_cfg.split,
                max_samples=mmlu_cfg.max_samples,
                shuffle=mmlu_cfg.shuffle,
            )
            results["mmlu"] = evaluate_mcq(
                model,
                tokenizer,
                mmlu_examples,
                max_length=cfg.runtime.max_length,
                max_new_tokens=mmlu_cfg.max_new_tokens,
            )
    except Exception as exc:
        if mmlu_cfg.placeholder_ok:
            results["mmlu"] = placeholder_result(str(exc))
        else:
            raise
    write_json(output_root / "mmlu_results.json", results["mmlu"])

    gsm8k_cfg = cfg.datasets.gsm8k
    try:
        if not gsm8k_cfg.enabled:
            results["gsm8k"] = {"status": "disabled", "reason": "GSM8K evaluation disabled by config"}
        elif not gsm8k_cfg.path or not Path(gsm8k_cfg.path).exists():
            if gsm8k_cfg.placeholder_ok:
                results["gsm8k"] = placeholder_result(f"GSM8K path not found: {gsm8k_cfg.path}")
            else:
                raise FileNotFoundError(f"GSM8K path not found: {gsm8k_cfg.path}")
        else:
            gsm8k_examples = load_gsm8k_examples(
                gsm8k_cfg.path,
                split=gsm8k_cfg.split,
                max_samples=gsm8k_cfg.max_samples,
            )
            results["gsm8k"] = evaluate_gsm8k(
                model,
                tokenizer,
                gsm8k_examples,
                max_length=cfg.runtime.max_length,
                max_new_tokens=gsm8k_cfg.max_new_tokens,
            )
    except Exception as exc:
        if gsm8k_cfg.placeholder_ok:
            results["gsm8k"] = placeholder_result(str(exc))
        else:
            raise
    write_json(output_root / "gsm8k_results.json", results["gsm8k"])

    humaneval_cfg = cfg.datasets.humaneval
    try:
        if not humaneval_cfg.enabled:
            results["humaneval"] = {"status": "disabled", "reason": "HumanEval evaluation disabled by config"}
        elif not humaneval_cfg.path or not Path(humaneval_cfg.path).exists():
            if humaneval_cfg.placeholder_ok:
                results["humaneval"] = placeholder_result(f"HumanEval path not found: {humaneval_cfg.path}")
            else:
                raise FileNotFoundError(f"HumanEval path not found: {humaneval_cfg.path}")
        else:
            humaneval_examples = load_code_examples(
                "humaneval",
                humaneval_cfg.path,
                split=humaneval_cfg.split,
                max_samples=humaneval_cfg.max_samples,
            )
            results["humaneval"] = evaluate_code_generation(
                model,
                tokenizer,
                humaneval_examples,
                dataset_name="humaneval",
                max_length=cfg.runtime.max_length,
                max_new_tokens=humaneval_cfg.max_new_tokens,
                exec_timeout_seconds=humaneval_cfg.exec_timeout_seconds,
            )
    except Exception as exc:
        if humaneval_cfg.placeholder_ok:
            results["humaneval"] = placeholder_result(str(exc))
        else:
            raise
    write_json(output_root / "humaneval_results.json", results["humaneval"])

    mbpp_cfg = cfg.datasets.mbpp
    try:
        if not mbpp_cfg.enabled:
            results["mbpp"] = {"status": "disabled", "reason": "MBPP evaluation disabled by config"}
        elif not mbpp_cfg.path or not Path(mbpp_cfg.path).exists():
            if mbpp_cfg.placeholder_ok:
                results["mbpp"] = placeholder_result(f"MBPP path not found: {mbpp_cfg.path}")
            else:
                raise FileNotFoundError(f"MBPP path not found: {mbpp_cfg.path}")
        else:
            mbpp_examples = load_code_examples(
                "mbpp",
                mbpp_cfg.path,
                split=mbpp_cfg.split,
                max_samples=mbpp_cfg.max_samples,
            )
            results["mbpp"] = evaluate_code_generation(
                model,
                tokenizer,
                mbpp_examples,
                dataset_name="mbpp",
                max_length=cfg.runtime.max_length,
                max_new_tokens=mbpp_cfg.max_new_tokens,
                exec_timeout_seconds=mbpp_cfg.exec_timeout_seconds,
            )
    except Exception as exc:
        if mbpp_cfg.placeholder_ok:
            results["mbpp"] = placeholder_result(str(exc))
        else:
            raise
    write_json(output_root / "mbpp_results.json", results["mbpp"])

    for task_name in ("mmlu", "gsm8k", "humaneval", "mbpp"):
        debug_summary = export_error_predictions(
            output_root,
            task_name,
            results.get(task_name, {}),
        )
        if debug_summary is not None:
            debug_outputs[task_name] = debug_summary

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "model_name": cfg.model.name,
        "model_path": cfg.model.path,
        "adapter_manifest": adapter.manifest_path,
        "adapter_checkpoint": adapter.checkpoint_path,
        "results": {
            task_name: {
                key: value
                for key, value in task_result.items()
                if key not in {"predictions", "generations"}
            }
            for task_name, task_result in results.items()
        },
        "debug_outputs": debug_outputs,
        "log_path": str(log_path),
    }
    write_json(output_root / "summary.json", summary)
    log_kv(logger, "baseline_eval_complete", summary=summary["results"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
