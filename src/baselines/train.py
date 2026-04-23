from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.template_qwen import render_qwen_generation_prompt, render_qwen_supervised_text


class PanSupervisedDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("No records provided for supervised training.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


@dataclass
class SupervisedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sample_ids: List[str]
    label_texts: List[str]
    messages: List[Sequence[Dict[str, str]]]


class SupervisedCollator:
    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> SupervisedBatch:
        records = list(batch)
        prompt_texts = [
            render_qwen_generation_prompt(self.tokenizer, record["messages"])
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
        encoded_prompt = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        self.tokenizer.padding_side = previous_padding_side

        prompt_lengths = encoded_prompt["attention_mask"].sum(dim=1).to(dtype=torch.long)
        labels = encoded_full["input_ids"].clone()
        for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[row_idx, :prompt_len] = -100

        return SupervisedBatch(
            input_ids=encoded_full["input_ids"],
            attention_mask=encoded_full["attention_mask"],
            labels=labels,
            sample_ids=[str(record["id"]) for record in records],
            label_texts=[str(record.get("label", "")) for record in records],
            messages=[record["messages"] for record in records],
        )


def build_supervised_dataloader(
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


def forward_supervised_batch(
    model: Any,
    batch: SupervisedBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    inputs = {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
        "labels": batch.labels.to(device),
    }
    outputs = model(**inputs, use_cache=False, return_dict=True)
    loss = outputs.loss
    metrics = {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_ce": float(loss.detach().cpu().item()),
    }
    return loss, metrics


def forward_distill_batch(
    student_model: Any,
    teacher_model: Any,
    batch: SupervisedBatch,
    *,
    student_device: torch.device,
    teacher_device: torch.device,
    temperature: float,
    hard_loss_weight: float,
    soft_loss_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    student_inputs = {
        "input_ids": batch.input_ids.to(student_device),
        "attention_mask": batch.attention_mask.to(student_device),
        "labels": batch.labels.to(student_device),
    }
    teacher_inputs = {
        "input_ids": batch.input_ids.to(teacher_device),
        "attention_mask": batch.attention_mask.to(teacher_device),
    }

    student_outputs = student_model(**student_inputs, use_cache=False, return_dict=True)
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_inputs, use_cache=False, return_dict=True)

    shift_student_logits = student_outputs.logits[:, :-1, :]
    shift_teacher_logits = teacher_outputs.logits[:, :-1, :].to(shift_student_logits.device)
    shift_labels = student_inputs["labels"][:, 1:]
    mask = shift_labels.ne(-100)

    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    mask_f = mask.to(dtype=kl_per_token.dtype)
    valid_token_count = mask_f.sum().clamp_min(1.0)
    soft_loss = (kl_per_token * mask_f).sum() / valid_token_count
    soft_loss = soft_loss * (temperature ** 2)

    hard_loss = student_outputs.loss
    total_loss = (hard_loss_weight * hard_loss) + (soft_loss_weight * soft_loss)
    metrics = {
        "loss_total": float(total_loss.detach().cpu().item()),
        "loss_hard": float(hard_loss.detach().cpu().item()),
        "loss_soft": float(soft_loss.detach().cpu().item()),
    }
    return total_loss, metrics
