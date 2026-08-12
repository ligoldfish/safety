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
    load_pair_to_student_layer,
    load_records,
    load_student_anchor_map,
    load_student_target_map,
    save_checkpoint,
    summarize_target_map,
    write_train_metric,
    write_val_metrics,
)
from src.ablations.strategies.targets import permute_target_map
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
    if target_mode not in {"semantic", "random_same_norm", "within_label_permutation", "cross_label_permutation"}:
        raise ValueError(
            f"Unsupported target mode: {cfg.target.mode}."
        )
    layer_loss_policy = str(cfg.target.layer_loss_policy).strip().lower()
    if layer_loss_policy not in {"all", "harmful_only", "label_weighted", "harmless_anchor"}:
        raise ValueError(
            f"Unsupported target.layer_loss_policy: {cfg.target.layer_loss_policy}. "
            "Expected 'all', 'harmful_only', 'label_weighted', or 'harmless_anchor'."
        )
    layer_loss_kind = str(cfg.target.loss_kind).strip().lower()
    if layer_loss_kind not in {"cosine", "normalized_mse", "raw_mse", "margin_contrastive"}:
        raise ValueError(
            f"Unsupported target.loss_kind: {cfg.target.loss_kind}. Expected cosine, "
            "normalized_mse, raw_mse, or margin_contrastive."
        )
    contrastive_margin = float(cfg.target.contrastive_margin)
    if not math.isfinite(contrastive_margin) or contrastive_margin < 0.0:
        raise ValueError("target.contrastive_margin must be finite and non-negative.")

    semantic_train_target_map, train_pair_keys = load_student_target_map(cfg.inputs.train_targets_dir)
    semantic_val_target_map, val_pair_keys = load_student_target_map(cfg.inputs.val_targets_dir)
    if train_pair_keys != val_pair_keys:
        raise ValueError("Train and validation student target pair indices do not match.")
    layer_ids = train_pair_keys
    pair_to_student_layer = load_pair_to_student_layer(cfg.inputs.pairing_path)
    expected_pair_keys = sorted(pair_to_student_layer.keys())
    if layer_ids != expected_pair_keys:
        raise ValueError(
            f"Student target pair indices {layer_ids} do not match pairing file indices {expected_pair_keys}."
        )
    unique_student_layers = sorted(set(pair_to_student_layer.values()))
    train_anchor_map = None
    val_anchor_map = None
    if layer_loss_policy == "harmless_anchor":
        if not cfg.inputs.train_anchor_dir or not cfg.inputs.val_anchor_dir:
            raise ValueError(
                "target.layer_loss_policy='harmless_anchor' requires inputs.train_anchor_dir "
                "and inputs.val_anchor_dir."
            )
        train_anchor_map = load_student_anchor_map(cfg.inputs.train_anchor_dir, layer_ids=unique_student_layers)
        val_anchor_map = load_student_anchor_map(cfg.inputs.val_anchor_dir, layer_ids=unique_student_layers)

    if target_mode == "random_same_norm":
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
    else:
        train_target_map = semantic_train_target_map
        val_target_map = semantic_val_target_map

    train_split_path = Path(cfg.inputs.train_split)
    if not train_split_path.exists():
        raise FileNotFoundError(
            f"train_set.jsonl missing at {cfg.inputs.train_split}. "
            "Likely a stale data/processed dir from before the train_set/alignment_set decoupling. "
            "Re-run scripts/00_prepare_data.py (PAN) or scripts/15_run_oneclick.py safety-full "
            "(safety baselines) with --force-rebuild to materialize train_set.jsonl. "
            "If you want to skip materialization for a quick test, you can: "
            "cp <dir>/alignment_set.jsonl <dir>/train_set.jsonl (not recommended for real runs)."
        )
    train_records = load_records(cfg.inputs.train_split)
    val_records = load_records(cfg.inputs.val_split)
    permutation_manifests: dict[str, dict[str, str]] = {}
    if target_mode in {"within_label_permutation", "cross_label_permutation"}:
        def labels_for_targets(records, targets, *, split_name: str) -> dict[str, str]:
            labels_by_id = {str(record["id"]): str(record.get("label", "")) for record in records}
            missing = sorted(set(targets) - set(labels_by_id))
            if missing:
                raise ValueError(
                    f"{split_name} target IDs are absent from the split: {missing[:5]}"
                )
            return {sample_id: labels_by_id[sample_id] for sample_id in targets}

        train_target_map, train_permutation = permute_target_map(
            semantic_train_target_map,
            labels_for_targets(train_records, semantic_train_target_map, split_name="train"),
            mode=target_mode,
            seed=int(cfg.target.random_seed),
        )
        val_target_map, val_permutation = permute_target_map(
            semantic_val_target_map,
            labels_for_targets(val_records, semantic_val_target_map, split_name="validation"),
            mode=target_mode,
            seed=int(cfg.target.random_seed) + 1,
        )
        permutation_manifests = {"train": train_permutation, "validation": val_permutation}
        for split_name, mapping in permutation_manifests.items():
            permutation_seed = int(cfg.target.random_seed) + (1 if split_name == "validation" else 0)
            write_json(
                output_root / f"target_permutation_{split_name}.json",
                {"mode": target_mode, "seed": permutation_seed, "mapping": mapping},
            )
    train_dataset = SemAlignDataset(
        train_records,
        train_target_map,
        anchor_map=train_anchor_map,
        filter_harmful_targets=bool(cfg.target.filter_harmful_targets),
    )
    val_dataset = SemAlignDataset(
        val_records,
        val_target_map,
        anchor_map=val_anchor_map,
    )
    semantic_reference_val_dataset = (
        None
        if target_mode == "semantic"
        else SemAlignDataset(val_records, semantic_val_target_map, anchor_map=val_anchor_map)
    )

    tokenizer, model, _ = load_hf_model(
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

    injection = inject_lora_modules(
        model,
        layer_indices=unique_student_layers,  # paired student layers (e.g. [16,18,19]); NOT pair indices
        target_suffixes=cfg.lora.target_modules,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )
    # Guard: LoRA must be injected on the physical student layers that L_layer supervises.
    # pair indices (layer_ids) are NOT physical layers; reusing them here was a real bug.
    _supervised_layers = {int(v) for v in pair_to_student_layer.values()}
    _injected_layers = {int(x) for x in injection.layer_indices}
    if not _injected_layers >= _supervised_layers:
        raise ValueError(
            f"LoRA injected on layers {sorted(_injected_layers)} but L_layer supervises student "
            f"layers {sorted(_supervised_layers)}; injection must cover the supervised layers. "
            f"(pair indices are NOT physical layers - use unique_student_layers.)"
        )
    freeze_non_lora_parameters(model)
    trainable_params, total_params = count_trainable_parameters(model)

    collator = SemAlignCollator(
        tokenizer,
        max_length=cfg.optim.max_length,
        layer_ids=layer_ids,
        pair_to_student_layer=pair_to_student_layer,
    )
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
        betas=(0.9, 0.95),
    )
    micro_batches_total = max(1, len(train_loader) * int(cfg.optim.epochs))
    num_training_steps = max(1, micro_batches_total // gradient_accumulation_steps)
    warmup_ratio = float(getattr(cfg.optim, "warmup_ratio", 0.0) or 0.0)
    num_warmup_steps = max(0, int(warmup_ratio * num_training_steps))
    scheduler_kind = str(getattr(cfg.optim, "lr_scheduler", "constant") or "constant").strip().lower()
    if scheduler_kind == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
        )
    elif scheduler_kind == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
        )
    elif scheduler_kind == "constant":
        if num_warmup_steps > 0:
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
        else:
            scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler={scheduler_kind!r}; expected cosine|linear|constant.")
    max_grad_norm = float(getattr(cfg.optim, "max_grad_norm", 0.0) or 0.0)
    early_stopping_patience = int(getattr(cfg.optim, "early_stopping_patience", 0) or 0)

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

    log_kv(
        logger,
        "training_setup",
        config_path=str(Path(args.config).resolve()),
        target_mode=target_mode,
        random_seed=int(cfg.target.random_seed),
        match_l2_norm=bool(cfg.target.match_l2_norm),
        layer_loss_policy=layer_loss_policy,
        layer_loss_kind=layer_loss_kind,
        contrastive_margin=contrastive_margin,
        harmful_layer_weight=float(cfg.target.harmful_layer_weight),
        harmless_layer_weight=float(cfg.target.harmless_layer_weight),
        filter_harmful_targets=bool(cfg.target.filter_harmful_targets),
        train_filtered_harmful_targets=int(train_dataset.filtered_harmful_target_count),
        train_missing_layer_target_count=int(train_dataset.missing_layer_target_count),
        val_missing_layer_target_count=int(val_dataset.missing_layer_target_count),
        train_anchor_summary=None if train_anchor_map is None else summarize_target_map(train_anchor_map),
        val_anchor_summary=None if val_anchor_map is None else summarize_target_map(val_anchor_map),
        layer_ids=layer_ids,
        pair_to_student_layer=pair_to_student_layer,
        unique_student_layers=unique_student_layers,
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
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
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
            "active_layer_weight_sum": 0.0,
        }
        for batch_idx, batch in enumerate(train_loader, start=1):
            loss_total, metrics = forward_semalign_batch(
                model,
                batch,
                device=device,
                layer_ids=layer_ids,
                pair_to_student_layer=pair_to_student_layer,
                layer_loss_weight=cfg.optim.layer_loss_weight,
                sft_loss_weight=cfg.optim.sft_loss_weight,
                layer_loss_policy=layer_loss_policy,
                layer_loss_kind=layer_loss_kind,
                contrastive_margin=contrastive_margin,
                harmful_layer_weight=float(cfg.target.harmful_layer_weight),
                harmless_layer_weight=float(cfg.target.harmless_layer_weight),
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
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                # PPU and NPU both run eager mode -> standard optimizer.step().
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
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
                            "active_layer_weight_sum": averaged_metrics["active_layer_weight_sum"],
                            "lr": cfg.optim.learning_rate,
                        },
                    )

        layer_target_cosine_mean = evaluate_layer_alignment(
            model,
            val_loader,
            device=device,
            layer_ids=layer_ids,
            pair_to_student_layer=pair_to_student_layer,
            layer_loss_policy=layer_loss_policy,
            harmful_layer_weight=float(cfg.target.harmful_layer_weight),
            harmless_layer_weight=float(cfg.target.harmless_layer_weight),
        )
        semantic_target_cosine_mean = layer_target_cosine_mean
        if semantic_reference_val_loader is not None:
            semantic_target_cosine_mean = evaluate_layer_alignment(
                model,
                semantic_reference_val_loader,
                device=device,
                layer_ids=layer_ids,
                pair_to_student_layer=pair_to_student_layer,
                layer_loss_policy=layer_loss_policy,
                harmful_layer_weight=float(cfg.target.harmful_layer_weight),
                harmless_layer_weight=float(cfg.target.harmless_layer_weight),
            )
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
            "harmful_refusal_rate": generation_metrics["harmful_refusal_rate"],
            "harmful_unsafe_output_rate": generation_metrics["harmful_unsafe_output_rate"],
            "harmful_safe_non_refusal_rate": generation_metrics["harmful_safe_non_refusal_rate"],
            "harmful_strict_unsafe_rate": generation_metrics["harmful_strict_unsafe_rate"],
            "harmless_over_refusal_rate": generation_metrics["harmless_over_refusal_rate"],
            "layer_target_cosine_mean": layer_target_cosine_mean,
            "active_target_cosine_mean": layer_target_cosine_mean,
            "semantic_target_cosine_mean": semantic_target_cosine_mean,
            "num_harmful": generation_metrics["num_harmful"],
            "num_harmless": generation_metrics["num_harmless"],
            "num_preamble_retries": generation_metrics["num_preamble_retries"],
            "num_preamble_unresolved": generation_metrics["num_preamble_unresolved"],
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
            harmful_strict_unsafe_rate=epoch_metrics["harmful_strict_unsafe_rate"],
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

        # Early stopping driven by layer alignment cosine (PhaseF has no scalar
        # val_loss exposed here; use semantic target cosine which is the natural
        # PAN-paper-aligned signal). Higher cosine = better; track best by epoch.
        epoch_alignment = float(
            epoch_metrics.get("semantic_target_cosine_mean")
            or epoch_metrics.get("active_target_cosine_mean")
            or 0.0
        )
        if epoch_alignment > -best_val_loss:
            best_val_loss = -epoch_alignment
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if early_stopping_patience > 0 and epochs_since_improve >= early_stopping_patience:
            log_kv(
                logger,
                "early_stop",
                stopped_at_epoch=epoch,
                best_epoch=best_epoch,
                best_alignment=-best_val_loss,
                patience=early_stopping_patience,
            )
            break

    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "student_model": cfg.model.name,
            "train_split": cfg.inputs.train_split,
            "val_split": cfg.inputs.val_split,
            "train_targets_dir": cfg.inputs.train_targets_dir,
            "val_targets_dir": cfg.inputs.val_targets_dir,
            "train_anchor_dir": cfg.inputs.train_anchor_dir,
            "val_anchor_dir": cfg.inputs.val_anchor_dir,
            "pairing_path": cfg.inputs.pairing_path,
            "pair_to_student_layer": pair_to_student_layer,
            "unique_student_layers": unique_student_layers,
            "lora_modules": injection.replaced_module_names,
            "lora_rank": cfg.lora.rank,
            "lora_alpha": cfg.lora.alpha,
            "lora_dropout": cfg.lora.dropout,
            "epochs": cfg.optim.epochs,
            "batch_size": cfg.optim.batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": cfg.optim.learning_rate,
            "sft_loss_weight": cfg.optim.sft_loss_weight,
            "layer_loss_weight": cfg.optim.layer_loss_weight,
            "warmup_ratio": float(getattr(cfg.optim, "warmup_ratio", 0.0) or 0.0),
            "max_grad_norm": float(getattr(cfg.optim, "max_grad_norm", 0.0) or 0.0),
            "lr_scheduler": str(getattr(cfg.optim, "lr_scheduler", "constant") or "constant"),
            "early_stopping_patience": int(getattr(cfg.optim, "early_stopping_patience", 0) or 0),
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "train_num_samples": len(train_dataset),
            "val_num_samples": len(val_dataset),
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "target_mode": target_mode,
            "target_random_seed": int(cfg.target.random_seed),
            "target_match_l2_norm": bool(cfg.target.match_l2_norm),
            "target_layer_loss_policy": layer_loss_policy,
            "target_loss_kind": layer_loss_kind,
            "target_contrastive_margin": contrastive_margin,
            "target_permutation_manifests": {
                split_name: str(output_root / f"target_permutation_{split_name}.json")
                for split_name in permutation_manifests
            },
            "target_harmful_layer_weight": float(cfg.target.harmful_layer_weight),
            "target_harmless_layer_weight": float(cfg.target.harmless_layer_weight),
            "target_filter_harmful_targets": bool(cfg.target.filter_harmful_targets),
            "train_filtered_harmful_targets": int(train_dataset.filtered_harmful_target_count),
            "train_missing_layer_target_count": int(train_dataset.missing_layer_target_count),
            "val_missing_layer_target_count": int(val_dataset.missing_layer_target_count),
            "train_target_summary": summarize_target_map(train_target_map),
            "val_target_summary": summarize_target_map(val_target_map),
            "train_anchor_summary": None if train_anchor_map is None else summarize_target_map(train_anchor_map),
            "val_anchor_summary": None if val_anchor_map is None else summarize_target_map(val_anchor_map),
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
