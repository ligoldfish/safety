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
    forward_distill_batch,
    load_distill_config,
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
    parser = argparse.ArgumentParser(description="Train a PAN distillation baseline with LoRA.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_distill_qwen35_9b_to_1b.yaml",
        help="Path to the PAN distillation YAML config.",
    )
    return parser.parse_args()


def _resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _evaluate_val_distill_loss(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    dataloader,
    *,
    student_device: torch.device,
    teacher_device: torch.device,
    temperature: float,
    hard_loss_weight: float,
    soft_loss_weight: float,
) -> dict[str, float]:
    student_model.eval()
    teacher_model.eval()
    runtime_backend = str(getattr(student_model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(student_model, "_codex_xla_model", None)
    total_examples = 0
    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    with torch.no_grad():
        for batch in dataloader:
            loss_total, metrics = forward_distill_batch(
                student_model,
                teacher_model,
                batch,
                student_device=student_device,
                teacher_device=teacher_device,
                temperature=temperature,
                hard_loss_weight=hard_loss_weight,
                soft_loss_weight=soft_loss_weight,
            )
            if runtime_backend == "tpu" and xla_model is not None:
                xla_model.mark_step()
            batch_size = int(batch.input_ids.size(0))
            total_examples += batch_size
            total_loss += float(loss_total.detach().cpu().item()) * batch_size
            total_hard += metrics["loss_hard"] * batch_size
            total_soft += metrics["loss_soft"] * batch_size
    if total_examples == 0:
        return {"val_loss": 0.0, "val_hard_loss": 0.0, "val_soft_loss": 0.0}
    return {
        "val_loss": total_loss / total_examples,
        "val_hard_loss": total_hard / total_examples,
        "val_soft_loss": total_soft / total_examples,
    }


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)
    set_global_seed(cfg.seed)

    output_root = ensure_dir(cfg.output.output_root)
    logger, log_path = setup_stage_logger("14_train_pan_distill", output_root / "logs")

    train_records = load_records(cfg.data.train_split)
    val_records = load_records(cfg.data.val_split)
    train_dataset = PanSupervisedDataset(train_records)
    val_dataset = PanSupervisedDataset(val_records)

    _, teacher_model, _ = load_hf_model(
        model_path=cfg.teacher.path,
        device_map=cfg.teacher.device_map,
        torch_dtype=cfg.teacher.torch_dtype,
        chat_template_enable_thinking=cfg.teacher.chat_template_enable_thinking,
        runtime_backend=cfg.teacher.runtime_backend,
        runtime_device=cfg.teacher.runtime_device,
        trust_remote_code=cfg.teacher.trust_remote_code,
        local_files_only=cfg.teacher.local_files_only,
        attn_implementation=cfg.teacher.attn_implementation,
    )
    teacher_model.eval()

    tokenizer, student_model, meta = load_hf_model(
        model_path=cfg.student.path,
        device_map=cfg.student.device_map,
        torch_dtype=cfg.student.torch_dtype,
        chat_template_enable_thinking=cfg.student.chat_template_enable_thinking,
        runtime_backend=cfg.student.runtime_backend,
        runtime_device=cfg.student.runtime_device,
        trust_remote_code=cfg.student.trust_remote_code,
        local_files_only=cfg.student.local_files_only,
        attn_implementation=cfg.student.attn_implementation,
    )
    student_model.train()

    layer_indices = list(range(int(meta["num_layers"])))
    injection = inject_lora_modules(
        student_model,
        layer_indices=layer_indices,
        target_suffixes=cfg.lora.target_modules,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )
    freeze_non_lora_parameters(student_model)
    trainable_params, total_params = count_trainable_parameters(student_model)

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
        [parameter for parameter in student_model.parameters() if parameter.requires_grad],
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

    student_device = _resolve_device(student_model)
    teacher_device = _resolve_device(teacher_model)
    student_runtime_backend = str(getattr(student_model, "_codex_runtime_backend", "")).lower()
    student_xla_model = getattr(student_model, "_codex_xla_model", None)

    log_kv(
        logger,
        "training_setup",
        config_path=str(Path(args.config).resolve()),
        train_num_samples=len(train_dataset),
        val_num_samples=len(val_dataset),
        teacher_model=cfg.teacher.name,
        student_model=cfg.student.name,
        student_runtime_backend=student_runtime_backend,
        student_runtime_device=str(getattr(student_model, "_codex_runtime_device", "")),
        teacher_runtime_device=str(getattr(teacher_model, "_codex_runtime_device", "")),
        effective_batch_size=int(cfg.optim.batch_size),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        temperature=cfg.distill.temperature,
        hard_loss_weight=cfg.distill.hard_loss_weight,
        soft_loss_weight=cfg.distill.soft_loss_weight,
        trainable_parameters=trainable_params,
        total_parameters=total_params,
        lora_modules=injection.replaced_module_names,
        log_path=str(log_path),
    )

    val_metrics = {}
    global_step = 0
    for epoch in range(1, cfg.optim.epochs + 1):
        log_kv(logger, "epoch_start", epoch=epoch, total_epochs=int(cfg.optim.epochs))
        student_model.train()
        teacher_model.eval()
        optimizer.zero_grad(set_to_none=True)
        accumulation_batches = 0
        accumulation_examples = 0
        accumulation_total = 0.0
        accumulation_hard = 0.0
        accumulation_soft = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            loss_total, metrics = forward_distill_batch(
                student_model,
                teacher_model,
                batch,
                student_device=student_device,
                teacher_device=teacher_device,
                temperature=cfg.distill.temperature,
                hard_loss_weight=cfg.distill.hard_loss_weight,
                soft_loss_weight=cfg.distill.soft_loss_weight,
            )
            microbatch_size = int(batch.input_ids.size(0))
            (loss_total / gradient_accumulation_steps).backward()
            accumulation_batches += 1
            accumulation_examples += microbatch_size
            accumulation_total += metrics["loss_total"] * microbatch_size
            accumulation_hard += metrics["loss_hard"] * microbatch_size
            accumulation_soft += metrics["loss_soft"] * microbatch_size

            should_step = (
                accumulation_batches >= gradient_accumulation_steps
                or batch_idx == len(train_loader)
            )
            if should_step:
                if student_runtime_backend == "tpu":
                    if student_xla_model is None:
                        raise RuntimeError("TPU backend requested but torch_xla runtime is unavailable on the student model.")
                    student_xla_model.optimizer_step(optimizer, barrier=True)
                    student_xla_model.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                averaged_total = accumulation_total / max(accumulation_examples, 1)
                averaged_hard = accumulation_hard / max(accumulation_examples, 1)
                averaged_soft = accumulation_soft / max(accumulation_examples, 1)
                accumulation_batches = 0
                accumulation_examples = 0
                accumulation_total = 0.0
                accumulation_hard = 0.0
                accumulation_soft = 0.0

                if global_step % cfg.optim.log_every_steps == 0:
                    write_train_metric(
                        train_metrics_path,
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss_total": averaged_total,
                            "loss_hard": averaged_hard,
                            "loss_soft": averaged_soft,
                            "learning_rate": cfg.optim.learning_rate,
                            "effective_batch_size": cfg.optim.batch_size,
                            "micro_batch_size": micro_batch_size,
                            "gradient_accumulation_steps": gradient_accumulation_steps,
                        },
                    )

        val_loss_metrics = _evaluate_val_distill_loss(
            student_model,
            teacher_model,
            val_loader,
            student_device=student_device,
            teacher_device=teacher_device,
            temperature=cfg.distill.temperature,
            hard_loss_weight=cfg.distill.hard_loss_weight,
            soft_loss_weight=cfg.distill.soft_loss_weight,
        )
        generation_metrics = evaluate_generation_refusal_metrics(
            student_model,
            tokenizer,
            val_dataset.records,
            device=student_device,
            max_length=cfg.optim.max_length,
            max_new_tokens=cfg.optim.max_new_tokens,
        )
        epoch_metrics = {
            **val_loss_metrics,
            "harmful_refusal_rate": generation_metrics["harmful_refusal_rate"],
            "harmful_safe_response_rate": generation_metrics["harmful_safe_response_rate"],
            "harmful_unsafe_output_rate": generation_metrics["harmful_unsafe_output_rate"],
            "harmless_over_refusal_rate": generation_metrics["harmless_over_refusal_rate"],
            "num_harmful": generation_metrics["num_harmful"],
            "num_harmless": generation_metrics["num_harmless"],
        }
        val_metrics[f"epoch_{epoch}"] = epoch_metrics
        write_val_metrics(val_metrics_path, val_metrics)
        write_json(generations_dir / f"epoch_{epoch:03d}.json", generation_metrics)
        log_kv(logger, "epoch_complete", epoch=epoch, **epoch_metrics)

        save_checkpoint(
            checkpoints_dir / f"epoch_{epoch:03d}.pt",
            model=student_model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            extra={
                "config_path": str(Path(args.config).resolve()),
                "epoch_metrics": epoch_metrics,
                "teacher_model": cfg.teacher.name,
                "student_model": cfg.student.name,
            },
        )

    manifest = {
        "config_path": str(Path(args.config).resolve()),
        "mode": "distill",
        "teacher_model_name": cfg.teacher.name,
        "teacher_model_path": cfg.teacher.path,
        "student_model_name": cfg.student.name,
        "student_model_path": cfg.student.path,
        "train_split": cfg.data.train_split,
        "val_split": cfg.data.val_split,
        "test_split": cfg.data.test_split,
        "lora_modules": injection.replaced_module_names,
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "temperature": cfg.distill.temperature,
        "hard_loss_weight": cfg.distill.hard_loss_weight,
        "soft_loss_weight": cfg.distill.soft_loss_weight,
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
