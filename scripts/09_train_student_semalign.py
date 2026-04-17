from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import count_trainable_parameters, freeze_non_lora_parameters, inject_lora_modules
from src.models.hf_loader import load_hf_model
from src.training import (
    SemAlignCollator,
    SemAlignDataset,
    build_random_target_map,
    build_dataloader,
    evaluate_generation_refusal_metrics,
    evaluate_layer_alignment,
    forward_semalign_batch,
    load_records,
    load_student_target_map,
    save_checkpoint,
    summarize_target_map,
    write_train_metric,
    write_val_metrics,
)
from src.utils.config import load_phasef_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the student model with semalign layer supervision and LoRA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phaseF_cpu.yaml",
        help="Path to the phase-F YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phasef_config(args.config)
    set_global_seed(cfg.seed)

    output_root = ensure_dir(cfg.output.output_root)
    logger, log_path = setup_stage_logger("09_train_student_semalign", output_root / "logs")
    target_mode = str(cfg.target.mode).strip().lower()
    if target_mode not in {"semantic", "random_same_norm"}:
        raise ValueError(
            f"Unsupported target mode: {cfg.target.mode}. Expected 'semantic' or 'random_same_norm'."
        )

    semantic_train_target_map, train_layer_ids = load_student_target_map(cfg.inputs.train_targets_dir)
    semantic_val_target_map, val_layer_ids = load_student_target_map(cfg.inputs.val_targets_dir)
    if train_layer_ids != val_layer_ids:
        raise ValueError("Train and validation student target layers do not match.")
    layer_ids = train_layer_ids
    pairing_payload = json.loads(Path(cfg.inputs.pairing_path).read_text(encoding="utf-8"))
    paired_student_layers = sorted(int(item["student_layer"]) for item in pairing_payload["pairs"])
    if layer_ids != paired_student_layers:
        raise ValueError(
            f"Student target layers {layer_ids} do not match pairing file layers {paired_student_layers}."
        )

    if target_mode == "semantic":
        train_target_map = semantic_train_target_map
        val_target_map = semantic_val_target_map
    else:
        train_target_map = build_random_target_map(
            semantic_train_target_map,
            seed=int(cfg.target.random_seed),
            match_l2_norm=bool(cfg.target.match_l2_norm),
        )
        val_target_map = build_random_target_map(
            semantic_val_target_map,
            seed=int(cfg.target.random_seed) + 1,
            match_l2_norm=bool(cfg.target.match_l2_norm),
        )

    train_records = load_records(cfg.inputs.train_split)
    val_records = load_records(cfg.inputs.val_split)
    train_dataset = SemAlignDataset(train_records, train_target_map)
    val_dataset = SemAlignDataset(val_records, val_target_map)
    semantic_reference_val_dataset = (
        None
        if target_mode == "semantic"
        else SemAlignDataset(val_records, semantic_val_target_map)
    )

    tokenizer, model, _ = load_hf_model(
        model_path=cfg.model.path,
        device_map=cfg.model.device_map,
        torch_dtype=cfg.model.torch_dtype,
        runtime_backend=cfg.model.runtime_backend,
        runtime_device=cfg.model.runtime_device,
        trust_remote_code=cfg.model.trust_remote_code,
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.train()

    injection = inject_lora_modules(
        model,
        layer_indices=layer_ids,
        target_suffixes=cfg.lora.target_modules,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )
    freeze_non_lora_parameters(model)
    trainable_params, total_params = count_trainable_parameters(model)

    collator = SemAlignCollator(tokenizer, max_length=cfg.optim.max_length, layer_ids=layer_ids)
    micro_batch_size = int(cfg.optim.micro_batch_size or cfg.optim.batch_size)
    micro_batch_size = max(1, min(micro_batch_size, int(cfg.optim.batch_size)))
    gradient_accumulation_steps = max(1, math.ceil(cfg.optim.batch_size / micro_batch_size))
    train_loader = build_dataloader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    semantic_reference_val_loader = None
    if semantic_reference_val_dataset is not None:
        semantic_reference_val_loader = build_dataloader(
            semantic_reference_val_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
    )

    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    logs_dir = ensure_dir(output_root / "logs")
    train_metrics_path = output_root / "train_metrics.jsonl"
    val_metrics_path = output_root / "val_metrics.json"
    generations_dir = ensure_dir(logs_dir / "val_generations")
    if train_metrics_path.exists():
        train_metrics_path.unlink()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)

    log_kv(
        logger,
        "training_setup",
        config_path=str(Path(args.config).resolve()),
        target_mode=target_mode,
        random_seed=int(cfg.target.random_seed),
        match_l2_norm=bool(cfg.target.match_l2_norm),
        layer_ids=layer_ids,
        paired_student_layers=paired_student_layers,
        train_num_samples=len(train_dataset),
        val_num_samples=len(val_dataset),
        effective_batch_size=int(cfg.optim.batch_size),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        runtime_backend=runtime_backend,
        runtime_device=str(getattr(model, "_codex_runtime_device", "")),
        trainable_parameters=trainable_params,
        total_parameters=total_params,
        lora_modules=injection.replaced_module_names,
        train_target_summary=summarize_target_map(train_target_map),
        val_target_summary=summarize_target_map(val_target_map),
        log_path=str(log_path),
    )

    val_metrics = {}
    global_step = 0
    for epoch in range(1, cfg.optim.epochs + 1):
        log_kv(logger, "epoch_start", epoch=epoch, total_epochs=int(cfg.optim.epochs))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_batches = 0
        accumulation_examples = 0
        accumulation_sums = {
            "loss_total": 0.0,
            "loss_out": 0.0,
            "loss_layer": 0.0,
            "layer_target_cosine_mean": 0.0,
        }
        for batch_idx, batch in enumerate(train_loader, start=1):
            loss_total, metrics = forward_semalign_batch(
                model,
                batch,
                device=device,
                layer_ids=layer_ids,
                layer_loss_weight=cfg.optim.layer_loss_weight,
            )
            microbatch_size = int(batch.input_ids.size(0))
            (loss_total / gradient_accumulation_steps).backward()
            accumulation_batches += 1
            accumulation_examples += microbatch_size
            for metric_name in accumulation_sums:
                accumulation_sums[metric_name] += metrics[metric_name] * microbatch_size

            should_step = (
                accumulation_batches >= gradient_accumulation_steps
                or batch_idx == len(train_loader)
            )
            if should_step:
                if runtime_backend == "tpu":
                    if xla_model is None:
                        raise RuntimeError("TPU backend requested but torch_xla runtime is unavailable on the model.")
                    xla_model.optimizer_step(optimizer, barrier=True)
                    xla_model.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                averaged_metrics = {
                    metric_name: accumulation_sums[metric_name] / max(accumulation_examples, 1)
                    for metric_name in accumulation_sums
                }
                accumulation_batches = 0
                accumulation_examples = 0
                for metric_name in accumulation_sums:
                    accumulation_sums[metric_name] = 0.0

                if global_step % cfg.optim.log_every_steps == 0:
                    write_train_metric(
                        train_metrics_path,
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "effective_batch_size": cfg.optim.batch_size,
                            "micro_batch_size": micro_batch_size,
                            "gradient_accumulation_steps": gradient_accumulation_steps,
                            "loss_total": averaged_metrics["loss_total"],
                            "loss_out": averaged_metrics["loss_out"],
                            "loss_layer": averaged_metrics["loss_layer"],
                            "layer_target_cosine_mean": averaged_metrics["layer_target_cosine_mean"],
                            "lr": cfg.optim.learning_rate,
                        },
                    )

        layer_target_cosine_mean = evaluate_layer_alignment(
            model,
            val_loader,
            device=device,
            layer_ids=layer_ids,
        )
        semantic_target_cosine_mean = layer_target_cosine_mean
        if semantic_reference_val_loader is not None:
            semantic_target_cosine_mean = evaluate_layer_alignment(
                model,
                semantic_reference_val_loader,
                device=device,
                layer_ids=layer_ids,
            )
        generation_metrics = evaluate_generation_refusal_metrics(
            model,
            tokenizer,
            val_dataset.records,
            device=device,
            max_length=cfg.optim.max_length,
            max_new_tokens=cfg.optim.max_new_tokens,
        )
        epoch_metrics = {
            "harmful_refusal_rate": generation_metrics["harmful_refusal_rate"],
            "harmful_unsafe_output_rate": generation_metrics["harmful_unsafe_output_rate"],
            "harmless_over_refusal_rate": generation_metrics["harmless_over_refusal_rate"],
            "layer_target_cosine_mean": layer_target_cosine_mean,
            "active_target_cosine_mean": layer_target_cosine_mean,
            "semantic_target_cosine_mean": semantic_target_cosine_mean,
            "num_harmful": generation_metrics["num_harmful"],
            "num_harmless": generation_metrics["num_harmless"],
        }
        val_metrics[f"epoch_{epoch}"] = epoch_metrics
        write_val_metrics(val_metrics_path, val_metrics)
        write_json(generations_dir / f"epoch_{epoch:03d}.json", generation_metrics)
        log_kv(
            logger,
            "epoch_complete",
            epoch=epoch,
            harmful_refusal_rate=epoch_metrics["harmful_refusal_rate"],
            harmful_unsafe_output_rate=epoch_metrics["harmful_unsafe_output_rate"],
            harmless_over_refusal_rate=epoch_metrics["harmless_over_refusal_rate"],
            active_target_cosine_mean=epoch_metrics["active_target_cosine_mean"],
            semantic_target_cosine_mean=epoch_metrics["semantic_target_cosine_mean"],
            num_harmful=epoch_metrics["num_harmful"],
            num_harmless=epoch_metrics["num_harmless"],
        )

        save_checkpoint(
            checkpoints_dir / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            extra={
                "config_path": str(Path(args.config).resolve()),
                "layer_ids": layer_ids,
                "epoch_metrics": epoch_metrics,
            },
        )

    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "student_model": cfg.model.name,
            "train_split": cfg.inputs.train_split,
            "val_split": cfg.inputs.val_split,
            "train_targets_dir": cfg.inputs.train_targets_dir,
            "val_targets_dir": cfg.inputs.val_targets_dir,
            "pairing_path": cfg.inputs.pairing_path,
            "paired_student_layers": paired_student_layers,
            "lora_modules": injection.replaced_module_names,
            "lora_rank": cfg.lora.rank,
            "lora_alpha": cfg.lora.alpha,
            "lora_dropout": cfg.lora.dropout,
            "epochs": cfg.optim.epochs,
            "batch_size": cfg.optim.batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": cfg.optim.learning_rate,
            "layer_loss_weight": cfg.optim.layer_loss_weight,
            "train_num_samples": len(train_dataset),
            "val_num_samples": len(val_dataset),
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "target_mode": target_mode,
            "target_random_seed": int(cfg.target.random_seed),
            "target_match_l2_norm": bool(cfg.target.match_l2_norm),
            "train_target_summary": summarize_target_map(train_target_map),
            "val_target_summary": summarize_target_map(val_target_map),
            "train_metrics_path": str(train_metrics_path),
            "val_metrics_path": str(val_metrics_path),
            "checkpoints_dir": str(checkpoints_dir),
            "log_path": str(log_path),
        },
    )

    log_kv(
        logger,
        "training_complete",
        output_root=str(output_root),
        checkpoints_dir=str(checkpoints_dir),
        val_metrics_path=str(val_metrics_path),
        train_metrics_path=str(train_metrics_path),
    )
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
