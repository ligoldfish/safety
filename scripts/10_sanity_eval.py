from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import inject_lora_modules_by_names
from src.models.hf_loader import load_hf_model
from src.training import (
    SemAlignCollator,
    SemAlignDataset,
    build_dataloader,
    evaluate_generation_refusal_metrics,
    evaluate_layer_alignment,
    load_records,
    load_student_target_map,
)
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the phase-G sanity evaluation for baseline and semalign-trained student models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path. Defaults to the latest checkpoint under outputs/.../training/checkpoints.",
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        default="",
        help="Optional explicit training directory. Defaults to outputs/.../training.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="sanity_eval",
        help="Where to write sanity-eval outputs under the phase-1 output root.",
    )
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=200,
        help="How many harmful / harmless sanity samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for layer-cosine evaluation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Generation length for sanity responses.",
    )
    return parser.parse_args()


def _select_records_with_targets(
    records: Sequence[Dict[str, Any]],
    target_map: Dict[str, Dict[int, torch.Tensor]],
    max_samples_per_label: int,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        sample_id = str(record["id"])
        label = str(record["label"])
        if sample_id not in target_map:
            continue
        if max_samples_per_label > 0 and counts[label] >= max_samples_per_label:
            continue
        counts[label] += 1
        kept.append(record)
    return kept


def _latest_checkpoint_path(training_dir: Path) -> Path:
    checkpoint_dir = training_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_dir}")
    return candidates[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trained_model(
    *,
    model_cfg: Any,
    checkpoint_path: Path,
    training_manifest: Dict[str, Any],
) -> tuple[Any, Any]:
    tokenizer, model, _ = load_hf_model(
        model_path=model_cfg.path,
        device_map=model_cfg.device_map,
        torch_dtype=model_cfg.torch_dtype or "float32",
        chat_template_enable_thinking=model_cfg.chat_template_enable_thinking,
        runtime_backend=model_cfg.runtime_backend,
        runtime_device=model_cfg.runtime_device,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=model_cfg.local_files_only,
        attn_implementation=model_cfg.attn_implementation,
    )
    inject_lora_modules_by_names(
        model,
        module_names=training_manifest["lora_modules"],
        rank=int(training_manifest["lora_rank"]),
        alpha=float(training_manifest["lora_alpha"]),
        dropout=float(training_manifest["lora_dropout"]),
    )
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint_payload["trainable_state_dict"]
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")
    missing_lora = [key for key in load_result.missing_keys if ".lora_" in key]
    if missing_lora:
        raise ValueError(f"Missing LoRA weights while loading checkpoint: {missing_lora}")
    model.eval()
    return tokenizer, model


def _load_baseline_model(model_cfg: Any) -> tuple[Any, Any]:
    tokenizer, model, _ = load_hf_model(
        model_path=model_cfg.path,
        device_map=model_cfg.device_map,
        torch_dtype=model_cfg.torch_dtype or "float32",
        chat_template_enable_thinking=model_cfg.chat_template_enable_thinking,
        runtime_backend=model_cfg.runtime_backend,
        runtime_device=model_cfg.runtime_device,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=model_cfg.local_files_only,
        attn_implementation=model_cfg.attn_implementation,
    )
    model.eval()
    return tokenizer, model


def _evaluate_variant(
    *,
    variant_name: str,
    tokenizer: Any,
    model: Any,
    records: Sequence[Dict[str, Any]],
    target_map: Dict[str, Dict[int, torch.Tensor]],
    layer_ids: Sequence[int],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    dataset = SemAlignDataset(records, target_map)
    collator = SemAlignCollator(tokenizer, max_length=max_length, layer_ids=layer_ids)
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    layer_target_cosine_mean = evaluate_layer_alignment(
        model,
        dataloader,
        device=device,
        layer_ids=layer_ids,
    )
    generation_metrics = evaluate_generation_refusal_metrics(
        model,
        tokenizer,
        dataset.records,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    return {
        "variant": variant_name,
        "num_samples": len(dataset),
        "num_harmful": generation_metrics["num_harmful"],
        "num_harmless": generation_metrics["num_harmless"],
        "harmful_refusal_rate": generation_metrics["harmful_refusal_rate"],
        "harmful_unsafe_output_rate": generation_metrics["harmful_unsafe_output_rate"],
        "harmless_over_refusal_rate": generation_metrics["harmless_over_refusal_rate"],
        "layer_target_cosine_mean": layer_target_cosine_mean,
        "generations": generation_metrics["generations"],
    }


def _write_comparison_csv(path: Path, baseline: Dict[str, Any], semalign: Dict[str, Any]) -> None:
    metric_names = [
        "harmful_refusal_rate",
        "harmful_unsafe_output_rate",
        "harmless_over_refusal_rate",
        "layer_target_cosine_mean",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "semalign", "delta"])
        writer.writeheader()
        for metric_name in metric_names:
            baseline_value = float(baseline[metric_name])
            semalign_value = float(semalign[metric_name])
            writer.writerow(
                {
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "semalign": semalign_value,
                    "delta": semalign_value - baseline_value,
                }
            )


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)

    output_root = Path(cfg.extraction.output_root)
    training_dir = Path(args.training_dir).resolve() if args.training_dir else (output_root / "training")
    sanity_output_dir = ensure_dir(output_root / args.output_dir_name)
    logger, log_path = setup_stage_logger("10_sanity_eval", sanity_output_dir / "logs")
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else _latest_checkpoint_path(training_dir)
    training_manifest = _load_json(training_dir / "manifest.json")
    log_kv(
        logger,
        "sanity_eval_setup",
        config_path=str(Path(args.config).resolve()),
        training_dir=str(training_dir),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(sanity_output_dir),
        target_mode=str(training_manifest.get("target_mode", "semantic")),
        log_path=str(log_path),
    )

    sanity_target_dir = output_root / "student_targets" / "student_safe_targets_sanity_test"
    fallback_target_dir = output_root / "student_targets" / "student_safe_targets_pan_test"
    if sanity_target_dir.exists():
        target_dir = sanity_target_dir
        records_path = Path(cfg.dataset.processed_dir) / "sanity_test_set.jsonl"
    else:
        target_dir = fallback_target_dir
        records_path = Path(cfg.dataset.processed_dir) / "pan_test_set.jsonl"
    target_map, layer_ids = load_student_target_map(target_dir)
    records = load_records(records_path)
    records = _select_records_with_targets(records, target_map, args.max_samples_per_label)
    if not records:
        raise ValueError(
            "No sanity records matched the available pan_test student targets. "
            "Generate teacher/student pan_test artifacts first."
        )
    log_kv(
        logger,
        "sanity_eval_records",
        target_dir=str(target_dir),
        records_path=str(records_path),
        num_records=len(records),
        label_counts=dict(Counter(str(record["label"]) for record in records)),
        layer_ids=list(layer_ids),
        max_samples_per_label=args.max_samples_per_label,
    )

    baseline_tokenizer, baseline_model = _load_baseline_model(cfg.student)
    baseline_results = _evaluate_variant(
        variant_name="baseline",
        tokenizer=baseline_tokenizer,
        model=baseline_model,
        records=records,
        target_map=target_map,
        layer_ids=layer_ids,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    log_kv(
        logger,
        "variant_complete",
        variant="baseline",
        harmful_refusal_rate=baseline_results["harmful_refusal_rate"],
        harmful_unsafe_output_rate=baseline_results["harmful_unsafe_output_rate"],
        harmless_over_refusal_rate=baseline_results["harmless_over_refusal_rate"],
        layer_target_cosine_mean=baseline_results["layer_target_cosine_mean"],
    )

    trained_tokenizer, trained_model = _load_trained_model(
        model_cfg=cfg.student,
        checkpoint_path=checkpoint_path,
        training_manifest=training_manifest,
    )
    semalign_results = _evaluate_variant(
        variant_name="semalign",
        tokenizer=trained_tokenizer,
        model=trained_model,
        records=records,
        target_map=target_map,
        layer_ids=layer_ids,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    log_kv(
        logger,
        "variant_complete",
        variant="semalign",
        harmful_refusal_rate=semalign_results["harmful_refusal_rate"],
        harmful_unsafe_output_rate=semalign_results["harmful_unsafe_output_rate"],
        harmless_over_refusal_rate=semalign_results["harmless_over_refusal_rate"],
        layer_target_cosine_mean=semalign_results["layer_target_cosine_mean"],
    )

    write_json(sanity_output_dir / "baseline_results.json", baseline_results)
    write_json(sanity_output_dir / "semalign_results.json", semalign_results)
    _write_comparison_csv(sanity_output_dir / "comparison.csv", baseline_results, semalign_results)
    write_json(
        sanity_output_dir / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "checkpoint_path": str(checkpoint_path),
            "training_dir": str(training_dir),
            "training_manifest_path": str((training_dir / "manifest.json").resolve()),
            "target_dir": str(target_dir.resolve()),
            "num_records": len(records),
            "label_counts": dict(Counter(str(record["label"]) for record in records)),
            "layer_ids": list(layer_ids),
            "max_samples_per_label": args.max_samples_per_label,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "target_mode": str(training_manifest.get("target_mode", "semantic")),
            "log_path": str(log_path),
        },
    )

    log_kv(
        logger,
        "sanity_eval_complete",
        output_dir=str(sanity_output_dir),
        baseline_harmful_refusal_rate=baseline_results["harmful_refusal_rate"],
        semalign_harmful_refusal_rate=semalign_results["harmful_refusal_rate"],
        baseline_semantic_cosine=baseline_results["layer_target_cosine_mean"],
        semalign_semantic_cosine=semalign_results["layer_target_cosine_mean"],
    )
    print(
        json.dumps(
            {
                "baseline": {
                    "harmful_refusal_rate": baseline_results["harmful_refusal_rate"],
                    "harmful_unsafe_output_rate": baseline_results["harmful_unsafe_output_rate"],
                    "harmless_over_refusal_rate": baseline_results["harmless_over_refusal_rate"],
                    "layer_target_cosine_mean": baseline_results["layer_target_cosine_mean"],
                },
                "semalign": {
                    "harmful_refusal_rate": semalign_results["harmful_refusal_rate"],
                    "harmful_unsafe_output_rate": semalign_results["harmful_unsafe_output_rate"],
                    "harmless_over_refusal_rate": semalign_results["harmless_over_refusal_rate"],
                    "layer_target_cosine_mean": semalign_results["layer_target_cosine_mean"],
                },
                "sanity_output_dir": str(sanity_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
