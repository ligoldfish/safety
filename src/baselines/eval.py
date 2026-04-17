from __future__ import annotations

import json
import multiprocessing as mp
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from src.baselines.config import AdapterConfig, BaselineModelConfig
from src.baselines.datasets import (
    CodeExample,
    GSM8KExample,
    extract_prediction_number,
    sanitize_code_generation,
)
from src.data.task_datasets import MCQExample, load_mcq_dataset, render_mcq_prompt
from src.data.template_qwen import build_qwen_messages, render_qwen_generation_prompt
from src.models import inject_lora_modules_by_names
from src.models.hf_loader import load_hf_model
from src.training import evaluate_generation_refusal_metrics


def load_model_for_evaluation(
    model_cfg: BaselineModelConfig,
    adapter_cfg: AdapterConfig | None = None,
) -> tuple[Any, Any]:
    tokenizer, model, _ = load_hf_model(
        model_path=model_cfg.path,
        device_map=model_cfg.device_map,
        torch_dtype=model_cfg.torch_dtype,
        runtime_backend=model_cfg.runtime_backend,
        runtime_device=model_cfg.runtime_device,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=model_cfg.local_files_only,
        attn_implementation=model_cfg.attn_implementation,
    )

    adapter_cfg = adapter_cfg or AdapterConfig()
    manifest_path = Path(adapter_cfg.manifest_path) if adapter_cfg.manifest_path else None
    checkpoint_path = Path(adapter_cfg.checkpoint_path) if adapter_cfg.checkpoint_path else None
    if not manifest_path or not checkpoint_path:
        model.eval()
        return tokenizer, model

    if not manifest_path.exists():
        raise FileNotFoundError(f"Adapter manifest not found: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {checkpoint_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inject_lora_modules_by_names(
        model,
        module_names=manifest["lora_modules"],
        rank=int(manifest["lora_rank"]),
        alpha=float(manifest["lora_alpha"]),
        dropout=float(manifest["lora_dropout"]),
    )
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(checkpoint_payload["trainable_state_dict"], strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"Unexpected adapter checkpoint keys: {load_result.unexpected_keys}")
    missing_lora = [key for key in load_result.missing_keys if ".lora_" in key]
    if missing_lora:
        raise ValueError(f"Missing LoRA weights while loading adapter: {missing_lora}")
    model.eval()
    return tokenizer, model


def _resolve_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> str:
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    tokenizer.padding_side = previous_padding_side
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if runtime_backend == "tpu" and xla_model is not None:
        xla_model.mark_step()
    new_tokens = generated[0, encoded["input_ids"].size(1) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_pan(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    max_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    metrics = evaluate_generation_refusal_metrics(
        model,
        tokenizer,
        records,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    metrics["status"] = "ok"
    metrics["num_samples"] = len(records)
    return metrics


def _continuation_nll(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    continuation_text: str,
    *,
    device: torch.device,
    max_length: int,
) -> float:
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)
    encoded_prompt = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded_full = tokenizer(
        prompt_text + continuation_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    prompt_len = int(encoded_prompt["attention_mask"].sum().item())
    input_ids = encoded_full["input_ids"].to(device)
    attention_mask = encoded_full["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    if runtime_backend == "tpu" and xla_model is not None:
        xla_model.mark_step()

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_mask = attention_mask[:, 1:].bool()
    candidate_mask = torch.zeros_like(token_mask)
    start_index = max(prompt_len - 1, 0)
    candidate_mask[:, start_index:] = True
    active_mask = token_mask & candidate_mask
    if not bool(active_mask.any()):
        return float("inf")

    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    return float((loss_per_token * active_mask.to(loss_per_token.dtype)).sum().item())


def evaluate_mcq(
    model: Any,
    tokenizer: Any,
    examples: Sequence[MCQExample],
    *,
    max_length: int,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    correct = 0
    predictions: List[Dict[str, Any]] = []
    for example in examples:
        prompt = render_mcq_prompt(example)
        prompt_text = render_qwen_generation_prompt(
            tokenizer,
            build_qwen_messages(prompt),
        )
        option_labels = [chr(ord("A") + idx) for idx in range(len(example.choices))]
        scores = {
            label: -_continuation_nll(
                model,
                tokenizer,
                prompt_text,
                f" {label}",
                device=device,
                max_length=max_length,
            )
            for label in option_labels
        }
        predicted_label = max(scores, key=scores.get)
        predicted_index = option_labels.index(predicted_label)
        is_correct = int(predicted_index == int(example.answer_index))
        correct += is_correct
        predictions.append(
            {
                "id": example.sample_id,
                "subject": example.subject,
                "prediction": predicted_label,
                "target": option_labels[int(example.answer_index)],
                "correct": bool(is_correct),
                "scores": scores,
            }
        )

    total = len(examples)
    return {
        "status": "ok",
        "num_samples": total,
        "num_correct": correct,
        "accuracy": 0.0 if total == 0 else correct / total,
        "predictions": predictions,
    }


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    examples: Sequence[GSM8KExample],
    *,
    max_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    correct = 0
    predictions: List[Dict[str, Any]] = []
    for example in examples:
        user_text = (
            "Solve the following grade-school math problem. "
            "Show your reasoning briefly and end with 'Final answer: <number>'.\n\n"
            f"Question: {example.question}"
        )
        prompt_text = render_qwen_generation_prompt(
            tokenizer,
            build_qwen_messages(user_text),
        )
        generated_text = _generate_text(
            model,
            tokenizer,
            prompt_text,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        prediction = extract_prediction_number(generated_text)
        is_correct = int(prediction == example.final_answer)
        correct += is_correct
        predictions.append(
            {
                "id": example.sample_id,
                "prediction": prediction,
                "target": example.final_answer,
                "correct": bool(is_correct),
                "generated_text": generated_text,
            }
        )

    total = len(examples)
    return {
        "status": "ok",
        "num_samples": total,
        "num_correct": correct,
        "accuracy": 0.0 if total == 0 else correct / total,
        "predictions": predictions,
    }


def _exec_program(program: str, result_queue: mp.Queue) -> None:
    namespace: Dict[str, Any] = {}
    try:
        exec(program, namespace, namespace)
    except Exception:
        result_queue.put(
            {
                "passed": False,
                "error": traceback.format_exc(),
            }
        )
        return
    result_queue.put({"passed": True, "error": ""})


def _run_code_program(program: str, timeout_seconds: int) -> tuple[bool, str]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_exec_program, args=(program, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"Timed out after {timeout_seconds} seconds"
    if result_queue.empty():
        return False, "Execution finished without a result payload"
    result = result_queue.get()
    return bool(result.get("passed", False)), str(result.get("error", ""))


def _assemble_humaneval_program(example: CodeExample, generated_text: str) -> str:
    completion = sanitize_code_generation(generated_text)
    stripped_completion = completion.lstrip()
    prompt_text = example.prompt.rstrip()
    if stripped_completion.startswith(prompt_text.lstrip()):
        candidate_code = completion
    elif example.entry_point and re.search(
        rf"def\s+{re.escape(example.entry_point)}\s*\(",
        completion,
    ):
        candidate_code = completion
    else:
        candidate_code = prompt_text + completion
    tests_blob = "\n\n".join(example.tests)
    return f"{candidate_code}\n\n{tests_blob}\n\ncheck({example.entry_point})\n"


def _assemble_mbpp_program(example: CodeExample, generated_text: str) -> str:
    candidate_code = sanitize_code_generation(generated_text)
    tests_blob = "\n".join(example.tests)
    return f"{candidate_code}\n\n{tests_blob}\n"


def evaluate_code_generation(
    model: Any,
    tokenizer: Any,
    examples: Sequence[CodeExample],
    *,
    dataset_name: str,
    max_length: int,
    max_new_tokens: int,
    exec_timeout_seconds: int,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    passed = 0
    executable = 0
    predictions: List[Dict[str, Any]] = []
    dataset_key = str(dataset_name).strip().lower()

    for example in examples:
        if dataset_key == "humaneval":
            user_text = (
                "Complete the following Python function. "
                "Return only Python code with no markdown fences.\n\n"
                f"{example.prompt}"
            )
        elif dataset_key == "mbpp":
            user_text = (
                "Write Python code that solves the following task. "
                "Return only Python code with no markdown fences.\n\n"
                f"{example.prompt}"
            )
        else:
            raise ValueError(f"Unsupported code dataset: {dataset_name}")

        prompt_text = render_qwen_generation_prompt(
            tokenizer,
            build_qwen_messages(user_text),
        )
        generated_text = _generate_text(
            model,
            tokenizer,
            prompt_text,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        if dataset_key == "humaneval":
            if not example.entry_point:
                predictions.append(
                    {
                        "task_id": example.task_id,
                        "passed": False,
                        "error": "Missing entry_point",
                        "generated_text": generated_text,
                    }
                )
                continue
            program = _assemble_humaneval_program(example, generated_text)
        else:
            program = _assemble_mbpp_program(example, generated_text)

        executable += 1
        task_passed, error_text = _run_code_program(program, exec_timeout_seconds)
        passed += int(task_passed)
        predictions.append(
            {
                "task_id": example.task_id,
                "passed": bool(task_passed),
                "error": error_text,
                "generated_text": generated_text,
                "sanitized_code": sanitize_code_generation(generated_text),
            }
        )

    return {
        "status": "ok",
        "num_samples": len(examples),
        "num_executable": executable,
        "num_passed": passed,
        "pass_at_1": 0.0 if executable == 0 else passed / executable,
        "predictions": predictions,
    }


def load_mcq_examples(
    path: str,
    *,
    split: str,
    max_samples: int,
    shuffle: bool,
) -> Sequence[MCQExample]:
    return load_mcq_dataset(
        dataset_name="mmlu",
        data_path=path,
        max_samples=max_samples,
        seed=42,
        split=split,
        shuffle=shuffle,
    )


def placeholder_result(reason: str) -> Dict[str, Any]:
    return {
        "status": "placeholder",
        "reason": reason,
    }


def filter_records(records: Sequence[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if max_samples <= 0:
        return list(records)
    return list(records)[:max_samples]
