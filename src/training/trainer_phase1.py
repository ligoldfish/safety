from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.template_qwen import (
    render_qwen_final_response_prefix,
    render_qwen_generation_prompt,
    render_qwen_supervised_text,
    strip_qwen_thinking_content,
)
from src.training.eval_utils import HarmfulResponseJudgment, judge_harmful_response_safety, looks_like_refusal, mean
from src.training.losses import cosine_layer_alignment_loss
from src.utils.io import ensure_dir, read_jsonl, write_json


def load_student_target_map(target_dir: str | Path) -> tuple[Dict[str, Dict[int, torch.Tensor]], List[int]]:
    target_path = Path(target_dir)
    part_paths = sorted(target_path.glob("part_*.pt"))
    if not part_paths:
        raise FileNotFoundError(f"No student target shards found under: {target_path}")

    target_map: Dict[str, Dict[int, torch.Tensor]] = {}
    layer_ids: set[int] = set()
    for part_path in part_paths:
        payload = torch.load(part_path, map_location="cpu", weights_only=True)
        sample_ids = [str(sample_id) for sample_id in payload["sample_ids"]]
        for layer_text, tensor in payload["student_safe_target_by_layer"].items():
            layer_idx = int(layer_text)
            layer_ids.add(layer_idx)
            for row_idx, sample_id in enumerate(sample_ids):
                target_map.setdefault(sample_id, {})[layer_idx] = tensor[row_idx].to(dtype=torch.float32)
    return target_map, sorted(layer_ids)


def summarize_target_map(target_map: Dict[str, Dict[int, torch.Tensor]]) -> Dict[str, Any]:
    layer_norms: Dict[int, List[float]] = {}
    for layer_targets in target_map.values():
        for layer_idx, tensor in layer_targets.items():
            layer_norms.setdefault(int(layer_idx), []).append(float(tensor.to(dtype=torch.float32).norm().item()))

    layer_stats: Dict[str, Dict[str, float]] = {}
    for layer_idx, norms in sorted(layer_norms.items()):
        norm_tensor = torch.tensor(norms, dtype=torch.float32)
        layer_stats[str(layer_idx)] = {
            "count": int(norm_tensor.numel()),
            "mean_l2_norm": float(norm_tensor.mean().item()),
            "std_l2_norm": float(norm_tensor.std(unbiased=False).item()) if norm_tensor.numel() > 1 else 0.0,
        }

    return {
        "num_samples": int(len(target_map)),
        "num_layers": int(len(layer_stats)),
        "layer_stats": layer_stats,
    }


def build_random_target_map(
    target_map: Dict[str, Dict[int, torch.Tensor]],
    *,
    seed: int,
    match_l2_norm: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Dict[int, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    randomized: Dict[str, Dict[int, torch.Tensor]] = {}

    for sample_id in sorted(target_map.keys()):
        randomized[sample_id] = {}
        for layer_idx in sorted(target_map[sample_id].keys()):
            target = target_map[sample_id][layer_idx].detach().cpu().to(dtype=torch.float32)
            random_vec = torch.randn(target.shape, generator=generator, dtype=torch.float32)
            random_vec = random_vec / random_vec.norm().clamp_min(eps)
            if match_l2_norm:
                target_norm = target.norm()
                if float(target_norm.item()) <= eps:
                    random_vec = torch.zeros_like(target)
                else:
                    random_vec = random_vec * target_norm
            randomized[sample_id][int(layer_idx)] = random_vec
    return randomized


class SemAlignDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], target_map: Dict[str, Dict[int, torch.Tensor]]) -> None:
        self.records = [record for record in records if str(record["id"]) in target_map]
        self.target_map = target_map
        if not self.records:
            raise ValueError("No records remain after joining against student targets.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = str(record["id"])
        return {
            "record": record,
            "targets": self.target_map[sample_id],
        }


@dataclass
class BatchPayload:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_prefix_last_positions: torch.Tensor
    layer_targets: Dict[int, torch.Tensor]
    sample_ids: List[str]
    labels_text: List[str]
    messages: List[Sequence[Dict[str, str]]]


class SemAlignCollator:
    def __init__(self, tokenizer: Any, *, max_length: int, layer_ids: Sequence[int]) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.layer_ids = [int(layer_idx) for layer_idx in layer_ids]

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> BatchPayload:
        records = [item["record"] for item in batch]
        target_dicts = [item["targets"] for item in batch]
        response_prefix_texts = [
            render_qwen_final_response_prefix(self.tokenizer, record["messages"])
            for record in records
        ]
        full_texts = [
            render_qwen_supervised_text(
                self.tokenizer,
                record["messages"],
                str(record.get("target_response") or record.get("accept_response") or ""),
            )
            for record in records
        ]

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        encoded_full = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded_prefix = self.tokenizer(
            response_prefix_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        self.tokenizer.padding_side = previous_padding_side

        response_prefix_lengths = encoded_prefix["attention_mask"].sum(dim=1).to(dtype=torch.long)
        response_prefix_last_positions = torch.clamp(response_prefix_lengths - 1, min=0)

        labels = encoded_full["input_ids"].clone()
        for row_idx, prefix_len in enumerate(response_prefix_lengths.tolist()):
            labels[row_idx, :prefix_len] = -100

        layer_targets = {
            layer_idx: torch.stack([target_dict[layer_idx] for target_dict in target_dicts], dim=0)
            for layer_idx in self.layer_ids
        }
        return BatchPayload(
            input_ids=encoded_full["input_ids"],
            attention_mask=encoded_full["attention_mask"],
            labels=labels,
            response_prefix_last_positions=response_prefix_last_positions,
            layer_targets=layer_targets,
            sample_ids=[str(record["id"]) for record in records],
            labels_text=[str(record["label"]) for record in records],
            messages=[record["messages"] for record in records],
        )


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn: Any,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def _capture_layer_outputs(
    model: nn.Module,
    *,
    layer_ids: Sequence[int],
    response_prefix_last_positions: torch.Tensor,
    cache: Dict[int, torch.Tensor],
):
    hooks = []
    for layer_idx in layer_ids:
        layer = model.model.layers[layer_idx]

        def hook(_module, _inputs, output, current_layer_idx=layer_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            selected = hidden[batch_indices, response_prefix_last_positions.to(hidden.device), :]
            cache[current_layer_idx] = selected
            return output

        hooks.append(layer.register_forward_hook(hook))
    return hooks


def forward_semalign_batch(
    model: nn.Module,
    batch: BatchPayload,
    *,
    device: torch.device,
    layer_ids: Sequence[int],
    layer_loss_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    inputs = {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
        "labels": batch.labels.to(device),
    }
    response_prefix_last_positions = batch.response_prefix_last_positions.to(device)
    cache: Dict[int, torch.Tensor] = {}
    hooks = _capture_layer_outputs(
        model,
        layer_ids=layer_ids,
        response_prefix_last_positions=response_prefix_last_positions,
        cache=cache,
    )
    try:
        outputs = model(**inputs, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()

    predicted_by_layer = {int(layer_idx): cache[int(layer_idx)] for layer_idx in layer_ids}
    target_by_layer = {
        int(layer_idx): tensor.to(device)
        for layer_idx, tensor in batch.layer_targets.items()
    }
    loss_layer, cosine_by_layer = cosine_layer_alignment_loss(predicted_by_layer, target_by_layer)
    loss_out = outputs.loss
    loss_total = loss_out + (layer_loss_weight * loss_layer)
    metrics = {
        "loss_total": float(loss_total.detach().cpu().item()),
        "loss_out": float(loss_out.detach().cpu().item()),
        "loss_layer": float(loss_layer.detach().cpu().item()),
        "layer_target_cosine_mean": mean(cosine_by_layer.values()),
    }
    return loss_total, metrics


@torch.no_grad()
def evaluate_layer_alignment(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    layer_ids: Sequence[int],
) -> float:
    cosine_scores: List[float] = []
    model.eval()
    for batch in dataloader:
        inputs = {
            "input_ids": batch.input_ids.to(device),
            "attention_mask": batch.attention_mask.to(device),
        }
        response_prefix_last_positions = batch.response_prefix_last_positions.to(device)
        cache: Dict[int, torch.Tensor] = {}
        hooks = _capture_layer_outputs(
            model,
            layer_ids=layer_ids,
            response_prefix_last_positions=response_prefix_last_positions,
            cache=cache,
        )
        try:
            model(**inputs, use_cache=False, return_dict=True)
        finally:
            for hook in hooks:
                hook.remove()

        for layer_idx in layer_ids:
            predicted = cache[int(layer_idx)]
            target = batch.layer_targets[int(layer_idx)].to(device)
            cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
            cosine_scores.extend(cosine.detach().cpu().tolist())
    return mean(cosine_scores)


@torch.no_grad()
def evaluate_generation_refusal_metrics(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    harmful_total = 0
    harmless_total = 0
    harmful_refusals = 0
    harmful_safes = 0
    harmful_unsafes = 0
    harmful_incomplete = 0
    harmless_refusals = 0
    harmless_incomplete = 0
    generations: List[Dict[str, Any]] = []

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)
    require_final_response = bool(getattr(tokenizer, "_codex_chat_template_enable_thinking", False))

    def _generate_response(prompt_text: str, *, generation_max_new_tokens: int) -> tuple[str, str]:
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=generation_max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if runtime_backend == "tpu" and xla_model is not None:
            xla_model.mark_step()
        new_tokens = generated[0, encoded["input_ids"].size(1):]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        final_text = strip_qwen_thinking_content(
            raw_text,
            require_final_response=require_final_response,
        )
        return raw_text, final_text

    for record in records:
        prompt_text = render_qwen_generation_prompt(tokenizer, record["messages"])
        raw_text, text = _generate_response(prompt_text, generation_max_new_tokens=max_new_tokens)
        used_max_new_tokens = int(max_new_tokens)
        retried_for_final_response = False
        if require_final_response and raw_text.strip() and not text:
            retry_max_new_tokens = min(max(int(max_new_tokens) * 2, int(max_new_tokens) + 512), 4096)
            if retry_max_new_tokens > int(max_new_tokens):
                retried_for_final_response = True
                used_max_new_tokens = retry_max_new_tokens
                raw_text, text = _generate_response(prompt_text, generation_max_new_tokens=retry_max_new_tokens)
        label = str(record["label"])
        is_refusal: bool | None = None
        is_safe: bool | None = None
        safety_reason: str | None = None
        incomplete_final_response = bool(require_final_response and raw_text.strip() and not text)
        if label == "harmful":
            if incomplete_final_response:
                judgment = HarmfulResponseJudgment(
                    is_refusal=False,
                    is_safe=False,
                    reason="incomplete_final_response",
                )
                harmful_incomplete += 1
            else:
                judgment = judge_harmful_response_safety(text)
            harmful_total += 1
            harmful_refusals += int(judgment.is_refusal)
            harmful_safes += int(judgment.is_safe)
            harmful_unsafes += int(not judgment.is_safe)
            is_refusal = judgment.is_refusal
            is_safe = judgment.is_safe
            safety_reason = judgment.reason
        elif label == "harmless":
            is_refusal = looks_like_refusal(text)
            harmless_total += 1
            harmless_refusals += int(is_refusal)
            harmless_incomplete += int(incomplete_final_response)
        generations.append(
            {
                "id": str(record["id"]),
                "label": label,
                "generated_text": raw_text,
                "final_text": text,
                "is_refusal": is_refusal,
                "is_safe": is_safe,
                "safety_reason": safety_reason,
                "incomplete_final_response": incomplete_final_response,
                "retried_for_final_response": retried_for_final_response,
                "used_max_new_tokens": used_max_new_tokens,
            }
        )
    tokenizer.padding_side = previous_padding_side

    return {
        "harmful_refusal_rate": 0.0 if harmful_total == 0 else harmful_refusals / harmful_total,
        "harmful_safe_response_rate": 0.0 if harmful_total == 0 else harmful_safes / harmful_total,
        "harmful_unsafe_output_rate": 0.0 if harmful_total == 0 else harmful_unsafes / harmful_total,
        "harmful_incomplete_output_rate": 0.0 if harmful_total == 0 else harmful_incomplete / harmful_total,
        "harmless_over_refusal_rate": 0.0 if harmless_total == 0 else harmless_refusals / harmless_total,
        "harmless_incomplete_output_rate": 0.0 if harmless_total == 0 else harmless_incomplete / harmless_total,
        "num_harmful": harmful_total,
        "num_harmless": harmless_total,
        "num_incomplete_final_response": harmful_incomplete + harmless_incomplete,
        "generations": generations,
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: Dict[str, Any],
) -> None:
    target = Path(checkpoint_path)
    ensure_dir(target.parent)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "extra": extra,
        },
        target,
    )


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def write_train_metric(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_val_metrics(path: str | Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
