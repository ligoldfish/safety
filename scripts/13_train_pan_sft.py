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

from src.baselines import (
    PanSupervisedDataset,
    SupervisedCollator,
    build_supervised_dataloader,
    forward_supervised_batch,
    load_sft_config,
)
from src.models import count_trainable_parameters, freeze_non_lora_parameters, inject_lora_modules
from src.models.hf_loader import load_hf_model
from src.training import (
    evaluate_generation_refusal_metrics,
    load_records,
    save_checkpoint,
    write_train_metric,
    write_val_metrics,
)
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PAN SFT baseline with LoRA.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_sft_qwen35_08b.yaml",
        help="Path to the PAN SFT YAML config.",
    )
    return parser.parse_args()


def _resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _evaluate_val_loss(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            loss, _ = forward_supervised_batch(model, batch, device=device)
            if runtime_backend == "tpu" and xla_model is not None:
                xla_model.mark_step()
            batch_size = int(batch.input_ids.size(0))
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size
    return 0.0 if total_examples == 0 else total_loss / total_examples


def main() -> None:
    args = parse_args()
    cfg = load_sft_config(args.config)
    set_global_seed(cfg.seed)

    output_root = ensure_dir(cfg.output.output_root)
    logger, log_path = setup_stage_logger("13_train_pan_sft", output_root / "logs")

    train_records = load_records(cfg.data.train_split)
    val_records = load_records(cfg.data.val_split)
    train_dataset = PanSupervisedDataset(train_records)
    val_dataset = PanSupervisedDataset(val_records)

    tokenizer, model, meta = load_hf_model(
        model_path=cfg.model.path,
        device_map=cfg.model.device_map,
        torch_dtype=cfg.model.torch_dtype,
        chat_template_enable_thinking=cfg.model.chat_template_enable_thinking,
        runtime_backend=cfg.model.runtime_backend,
        runtime_device=cfg.model.runtime_device,
        trust_remote_code=cfg.model.trust_remote_code,
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.train()

    layer_indices = list(range(int(meta["num_layers"])))
    injection = inject_lora_modules(
        model,
        layer_indices=layer_indices,
        target_suffixes=cfg.lora.target_modules,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )
    freeze_non_lora_parameters(model)
    trainable_params, total_params = count_trainable_parameters(model)

    collator = SupervisedCollator(tokenizer, max_length=cfg.optim.max_length)
    micro_batch_size = int(cfg.optim.micro_batch_size or cfg.optim.batch_size)
    micro_batch_size = max(1, min(micro_batch_size, int(cfg.optim.batch_size)))
    gradient_accumulation_steps = max(1, math.ceil(cfg.optim.batch_size / micro_batch_size))
    train_loader = build_supervised_dataloader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = build_supervised_dataloader(
        val_dataset,
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

    device = _resolve_device(model)
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)

    log_kv(
        logger,
        "training_setup",
        config_path=str(Path(args.config).resolve()),
        train_num_samples=len(train_dataset),
        val_num_samples=len(val_dataset),
        runtime_backend=runtime_backend,
        runtime_device=str(getattr(model, "_codex_runtime_device", "")),
        effective_batch_size=int(cfg.optim.batch_size),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        trainable_parameters=trainable_params,
        total_parameters=total_params,
        lora_modules=injection.replaced_module_names,
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
        accumulation_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            loss_total, metrics = forward_supervised_batch(model, batch, device=device)
            microbatch_size = int(batch.input_ids.size(0))
            (loss_total / gradient_accumulation_steps).backward()
            accumulation_batches += 1
            accumulation_examples += microbatch_size
            accumulation_loss += metrics["loss_total"] * microbatch_size

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
                averaged_loss = accumulation_loss / max(accumulation_examples, 1)
                accumulation_batches = 0
                accumulation_examples = 0
                accumulation_loss = 0.0

                if global_step % cfg.optim.log_every_steps == 0:
                    write_train_metric(
                        train_metrics_path,
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss_total": averaged_loss,
                            "learning_rate": cfg.optim.learning_rate,
                            "effective_batch_size": cfg.optim.batch_size,
                            "micro_batch_size": micro_batch_size,
                            "gradient_accumulation_steps": gradient_accumulation_steps,
                        },
                    )

        val_loss = _evaluate_val_loss(model, val_loader, device=device)
        generation_metrics = evaluate_generation_refusal_metrics(
            model,
            tokenizer,
            val_dataset.records,
            device=device,
            max_length=cfg.optim.max_length,
            max_new_tokens=cfg.optim.max_new_tokens,
            batch_size=micro_batch_size,
        )
        epoch_metrics = {
            "val_loss": val_loss,
            "harmful_refusal_rate": generation_metrics["harmful_refusal_rate"],
            "harmful_unsafe_output_rate": generation_metrics["harmful_unsafe_output_rate"],
            "harmful_safe_non_refusal_rate": generation_metrics["harmful_safe_non_refusal_rate"],
            "harmful_strict_unsafe_rate": generation_metrics["harmful_strict_unsafe_rate"],
            "harmless_over_refusal_rate": generation_metrics["harmless_over_refusal_rate"],
            "num_harmful": generation_metrics["num_harmful"],
            "num_harmless": generation_metrics["num_harmless"],
            "num_preamble_retries": generation_metrics["num_preamble_retries"],
            "num_preamble_unresolved": generation_metrics["num_preamble_unresolved"],
        }
        val_metrics[f"epoch_{epoch}"] = epoch_metrics
        write_val_metrics(val_metrics_path, val_metrics)
        write_json(generations_dir / f"epoch_{epoch:03d}.json", generation_metrics)
        log_kv(logger, "epoch_complete", epoch=epoch, **epoch_metrics)

        save_checkpoint(
            checkpoints_dir / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            extra={
                "config_path": str(Path(args.config).resolve()),
                "epoch_metrics": epoch_metrics,
            },
        )

    manifest = {
        "config_path": str(Path(args.config).resolve()),
        "mode": "sft",
        "model_name": cfg.model.name,
        "model_path": cfg.model.path,
        "train_split": cfg.data.train_split,
        "val_split": cfg.data.val_split,
        "test_split": cfg.data.test_split,
        "lora_modules": injection.replaced_module_names,
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "epochs": cfg.optim.epochs,
        "batch_size": cfg.optim.batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": cfg.optim.learning_rate,
        "train_num_samples": len(train_dataset),
        "val_num_samples": len(val_dataset),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "train_metrics_path": str(train_metrics_path),
        "val_metrics_path": str(val_metrics_path),
        "checkpoints_dir": str(checkpoints_dir),
        "log_path": str(log_path),
    }
    write_json(output_root / "manifest.json", manifest)
    log_kv(logger, "training_complete", output_root=str(output_root), manifest=manifest)
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
