from __future__ import annotations

import ast
import json
import math
import multiprocessing as mp
import re
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from src.baselines.config import AdapterConfig, BaselineModelConfig
from src.baselines.datasets import (
    CodeExample,
    GSM8KExample,
    extract_official_gsm8k_prediction,
    extract_official_mmlu_prediction,
    extract_multiple_choice_prediction,
    extract_prediction_number,
    normalize_numeric_answer,
    sanitize_code_generation,
)
from src.data.task_datasets import MCQExample, load_mcq_dataset
from src.data.template_qwen import (
    build_qwen_messages,
    render_qwen_generation_prompt,
    strip_qwen_thinking_content,
)
from src.models import inject_lora_modules_by_names
from src.models.hf_loader import load_hf_model
from src.training import evaluate_generation_refusal_metrics


RETRY_MAX_NEW_TOKENS_CAP = 8192


def load_model_for_evaluation(
    model_cfg: BaselineModelConfig,
    adapter_cfg: AdapterConfig | None = None,
) -> tuple[Any, Any]:
    tokenizer, model, _ = load_hf_model(
        model_path=model_cfg.path,
        device_map=model_cfg.device_map,
        torch_dtype=model_cfg.torch_dtype,
        chat_template_enable_thinking=model_cfg.chat_template_enable_thinking,
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


def _generate_text_batch_with_info(
    model: Any,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> tuple[List[str], List[int]]:
    if not prompt_texts:
        return [], []
    runtime_backend = str(getattr(model, "_codex_runtime_backend", "")).lower()
    xla_model = getattr(model, "_codex_xla_model", None)
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    tokenizer.padding_side = previous_padding_side
    prompt_seq_lens = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    if runtime_backend == "tpu" and xla_model is not None:
        xla_model.mark_step()
    prompt_width = int(encoded["input_ids"].size(1))
    texts = [
        tokenizer.decode(generated[row_idx, prompt_width:], skip_special_tokens=True)
        for row_idx in range(len(prompt_texts))
    ]
    return texts, prompt_seq_lens


def _generate_text_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> List[str]:
    texts, _ = _generate_text_batch_with_info(
        model,
        tokenizer,
        prompt_texts,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    return texts


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> str:
    return _generate_text_batch(
        model,
        tokenizer,
        [prompt_text],
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )[0]


def _resolve_first_pass_max_new_tokens(
    max_new_tokens: int,
    initial_max_new_tokens: int,
    require_final_response: bool,
) -> int:
    base = int(max_new_tokens)
    if require_final_response and int(initial_max_new_tokens) > 0:
        return max(1, min(int(initial_max_new_tokens), base))
    return base


def _resolve_retry_max_new_tokens(
    first_max_new_tokens: int,
    max_new_tokens: int,
    retry_cap: int = RETRY_MAX_NEW_TOKENS_CAP,
) -> int:
    if int(first_max_new_tokens) < int(max_new_tokens):
        return int(max_new_tokens)
    return min(
        max(int(max_new_tokens) * 2, int(max_new_tokens) + 512),
        int(retry_cap),
    )


def _generate_with_retry_for_final_response(
    model: Any,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    initial_max_new_tokens: int,
    require_final_response: bool,
    retry_cap: int = RETRY_MAX_NEW_TOKENS_CAP,
) -> List[Dict[str, Any]]:
    if not prompt_texts:
        return []

    first_max_new_tokens = _resolve_first_pass_max_new_tokens(
        max_new_tokens,
        initial_max_new_tokens,
        require_final_response,
    )

    raw_texts, prompt_seq_lens = _generate_text_batch_with_info(
        model,
        tokenizer,
        prompt_texts,
        device=device,
        max_length=max_length,
        max_new_tokens=first_max_new_tokens,
    )
    final_texts = [
        strip_qwen_thinking_content(raw, require_final_response=require_final_response)
        for raw in raw_texts
    ]
    used_max_new_tokens = [int(first_max_new_tokens)] * len(prompt_texts)
    retried_flags = [False] * len(prompt_texts)

    if require_final_response:
        retry_indexes = [
            idx
            for idx, (raw, final) in enumerate(zip(raw_texts, final_texts))
            if raw.strip() and not final
        ]
        if retry_indexes:
            retry_max_new_tokens = _resolve_retry_max_new_tokens(
                first_max_new_tokens,
                max_new_tokens,
                retry_cap=retry_cap,
            )
            if retry_max_new_tokens > first_max_new_tokens:
                retry_prompt_texts = [prompt_texts[idx] for idx in retry_indexes]
                retry_raw_texts, _ = _generate_text_batch_with_info(
                    model,
                    tokenizer,
                    retry_prompt_texts,
                    device=device,
                    max_length=max_length,
                    max_new_tokens=retry_max_new_tokens,
                )
                for local_idx, original_idx in enumerate(retry_indexes):
                    raw_texts[original_idx] = retry_raw_texts[local_idx]
                    final_texts[original_idx] = strip_qwen_thinking_content(
                        retry_raw_texts[local_idx],
                        require_final_response=require_final_response,
                    )
                    used_max_new_tokens[original_idx] = int(retry_max_new_tokens)
                    retried_flags[original_idx] = True

    results: List[Dict[str, Any]] = []
    for idx in range(len(prompt_texts)):
        raw = raw_texts[idx]
        final = final_texts[idx]
        incomplete = bool(require_final_response and raw.strip() and not final)
        results.append(
            {
                "raw_text": raw,
                "final_text": final,
                "used_max_new_tokens": int(used_max_new_tokens[idx]),
                "retried_for_final_response": bool(retried_flags[idx]),
                "incomplete_final_response": incomplete,
                "possibly_truncated_prompt": bool(prompt_seq_lens[idx] >= int(max_length)),
            }
        )
    return results


def evaluate_pan(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    max_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
    initial_max_new_tokens: int = 0,  # retained for config/YAML compatibility; ignored here
) -> Dict[str, Any]:
    device = _resolve_device(model)
    metrics = evaluate_generation_refusal_metrics(
        model,
        tokenizer,
        records,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
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


def _mmlu_final_answer_instruction() -> str:
    return (
        "Please reason step by step, then finish with a single line in this exact format:\n"
        "Final Answer: X\n"
        "where X is one of A, B, C, or D."
    )


def _gsm8k_final_answer_instruction() -> str:
    return (
        "Please reason step by step, then finish with a single line in this exact format:\n"
        "Final Answer: <number>\n"
        "Return only the numeric value on that line."
    )


def _code_final_answer_instruction() -> str:
    return (
        "After your reasoning, output the final solution as a single Python code block "
        "(```python ... ```) containing the complete implementation. Do not include "
        "example usage or tests inside the code block."
    )


def _render_official_mmlu_chat_prompt(example: MCQExample) -> str:
    option_labels = [chr(ord("A") + idx) for idx in range(len(example.choices))]
    option_lines = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(option_labels, example.choices)
    )
    return (
        "The following is a multiple-choice question. "
        "Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        f"{example.question}\n{option_lines}\n\n"
        f"{_mmlu_final_answer_instruction()}"
    )


def _build_gsm8k_prompt(example: GSM8KExample) -> str:
    return f"{example.question}\n\n{_gsm8k_final_answer_instruction()}"


def _safe_numeric_eval(text: str) -> float:
    candidate = str(text).strip().replace(",", "")
    if not candidate:
        raise ValueError("empty numeric expression")
    if not re.fullmatch(r"[0-9eE\.\+\-\(\)\/ ]+", candidate):
        raise ValueError(f"unsupported numeric expression: {text}")
    return float(eval(candidate, {"__builtins__": {}}, {}))


def _gsm8k_answers_match(target: str, prediction: str) -> bool:
    if not target or not prediction:
        return False
    try:
        return math.isclose(
            _safe_numeric_eval(target),
            _safe_numeric_eval(prediction),
            abs_tol=1e-4,
        )
    except Exception:
        return normalize_numeric_answer(target) == normalize_numeric_answer(prediction)


def _build_humaneval_official_prompt(example: CodeExample) -> str:
    prompt_text = str(example.prompt)
    signature_match = re.search(
        rf"def\s+({re.escape(example.entry_point)}\s*\(.*?\))\s*:",
        prompt_text,
        flags=re.DOTALL,
    )
    signature = signature_match.group(1).strip() if signature_match else f"{example.entry_point}(...)"
    description_match = re.search(
        r'"""(.*?)"""|\'\'\'(.*?)\'\'\'',
        prompt_text,
        flags=re.DOTALL,
    )
    description = ""
    if description_match:
        description = description_match.group(1) or description_match.group(2) or ""
        description = description.strip()
    if not description:
        description = prompt_text.strip()
    return (
        f"Write a Python function `{signature}` to solve the following problem:\n"
        f"{description}\n"
        f"{prompt_text}\n\n"
        f"{_code_final_answer_instruction()}"
    )


def _extract_humaneval_completion(
    generated_text: str,
    entry_point: str,
    *,
    require_final_response: bool,
) -> str:
    final_text = strip_qwen_thinking_content(
        generated_text,
        require_final_response=require_final_response,
    )
    if not final_text:
        return ""

    fenced_def = re.search(
        rf"```(?:python)?\s*def\s+{re.escape(entry_point)}\s*\(.*?\):\s*\n(.*?)```",
        final_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_def:
        return fenced_def.group(1)

    def_match = re.search(
        rf"def\s+{re.escape(entry_point)}\s*\(.*?\):\s*\n(.*)",
        final_text,
        flags=re.DOTALL,
    )
    if def_match:
        return def_match.group(1)

    fenced = re.search(
        r"```(?:python)?\s*(.*?)```",
        final_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        fenced_text = fenced.group(1).strip("\n")
        if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", fenced_text):
            nested_match = re.search(
                rf"def\s+{re.escape(entry_point)}\s*\(.*?\):\s*\n(.*)",
                fenced_text,
                flags=re.DOTALL,
            )
            if nested_match:
                return nested_match.group(1)
        return textwrap.indent(fenced_text, "    ")

    stripped = sanitize_code_generation(
        final_text,
        require_final_response=False,
    )
    if not stripped:
        return ""
    return textwrap.indent(stripped, "    ")


def _build_mbpp_officialish_prompt(example: CodeExample) -> str:
    header: str
    if example.entry_point:
        header = (
            f"Write a Python function `{example.entry_point}` to solve the following problem:\n"
            f"{example.prompt}"
        )
    else:
        header = (
            "Write Python code to solve the following problem. Return only Python code.\n"
            f"{example.prompt}"
        )
    return f"{header}\n\n{_code_final_answer_instruction()}"


def _extract_mbpp_code(
    generated_text: str,
    entry_point: str,
    *,
    require_final_response: bool,
) -> str:
    final_text = strip_qwen_thinking_content(
        generated_text,
        require_final_response=require_final_response,
    )
    if not final_text:
        return ""

    fenced = re.findall(
        r"```(?:python)?\s*(.*?)```",
        final_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        final_text = max(fenced, key=len).strip()

    if entry_point:
        entry_match = re.search(
            rf"(def\s+{re.escape(entry_point)}\s*\(.*)",
            final_text,
            flags=re.DOTALL,
        )
        if entry_match:
            return entry_match.group(1).strip()

    generic_def = re.search(r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*)", final_text, flags=re.DOTALL)
    if generic_def:
        return generic_def.group(1).strip()
    return sanitize_code_generation(final_text, require_final_response=False)


def evaluate_mcq(
    model: Any,
    tokenizer: Any,
    examples: Sequence[MCQExample],
    *,
    max_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
    initial_max_new_tokens: int = 0,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    correct = 0
    num_incomplete = 0
    num_truncated_prompts = 0
    predictions: List[Dict[str, Any]] = []
    use_thinking_generation = bool(getattr(tokenizer, "_codex_chat_template_enable_thinking", False))
    if use_thinking_generation:
        effective_batch_size = max(1, int(batch_size))
        for batch_start in range(0, len(examples), effective_batch_size):
            batch_examples = list(examples[batch_start: batch_start + effective_batch_size])
            prompt_payloads: List[tuple[MCQExample, List[str], Dict[str, str], str]] = []
            prompt_texts: List[str] = []
            for example in batch_examples:
                prompt = _render_official_mmlu_chat_prompt(example)
                option_labels = [chr(ord("A") + idx) for idx in range(len(example.choices))]
                choice_map = {
                    label: str(choice)
                    for label, choice in zip(option_labels, example.choices)
                }
                prompt_text = render_qwen_generation_prompt(
                    tokenizer,
                    build_qwen_messages(prompt),
                )
                prompt_payloads.append((example, option_labels, choice_map, prompt_text))
                prompt_texts.append(prompt_text)

            generation_results = _generate_with_retry_for_final_response(
                model,
                tokenizer,
                prompt_texts,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                initial_max_new_tokens=initial_max_new_tokens,
                require_final_response=True,
            )
            for row_idx, (example, option_labels, choice_map, _prompt_text) in enumerate(prompt_payloads):
                gen_info = generation_results[row_idx]
                generated_text = gen_info["raw_text"]
                final_text = gen_info["final_text"]
                incomplete = bool(gen_info["incomplete_final_response"])
                if incomplete:
                    num_incomplete += 1
                if gen_info["possibly_truncated_prompt"]:
                    num_truncated_prompts += 1
                predicted_label = (
                    ""
                    if incomplete
                    else extract_official_mmlu_prediction(
                        final_text,
                        option_labels,
                        choice_map,
                    )
                )
                predicted_index = option_labels.index(predicted_label) if predicted_label in option_labels else -1
                is_correct = int(predicted_index == int(example.answer_index))
                correct += is_correct
                predictions.append(
                    {
                        "id": example.sample_id,
                        "subject": example.subject,
                        "question": example.question,
                        "choices": list(example.choices),
                        "prediction": predicted_label,
                        "target": option_labels[int(example.answer_index)],
                        "correct": bool(is_correct),
                        "generated_text": generated_text,
                        "final_text": final_text,
                        "choice_map": choice_map,
                        "incomplete_final_response": incomplete,
                        "retried_for_final_response": bool(gen_info["retried_for_final_response"]),
                        "used_max_new_tokens": int(gen_info["used_max_new_tokens"]),
                        "possibly_truncated_prompt": bool(gen_info["possibly_truncated_prompt"]),
                    }
                )
        total = len(examples)
        effective = max(total - num_incomplete, 0)
        return {
            "status": "ok",
            "num_samples": total,
            "num_correct": correct,
            "num_incomplete": num_incomplete,
            "num_effective": effective,
            "num_possibly_truncated_prompts": num_truncated_prompts,
            "accuracy": 0.0 if total == 0 else correct / total,
            "accuracy_effective": 0.0 if effective == 0 else correct / effective,
            "incomplete_final_response_rate": 0.0 if total == 0 else num_incomplete / total,
            "predictions": predictions,
        }

    # Loglikelihood path (thinking disabled): score each option via continuation NLL.
    for example in examples:
        prompt = _render_official_mmlu_chat_prompt(example)
        option_labels = [chr(ord("A") + idx) for idx in range(len(example.choices))]

        prompt_text = render_qwen_generation_prompt(
            tokenizer,
            build_qwen_messages(prompt),
            enable_thinking=False,
        )
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
                "question": example.question,
                "choices": list(example.choices),
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
        "num_incomplete": 0,
        "num_effective": total,
        "num_possibly_truncated_prompts": 0,
        "accuracy": 0.0 if total == 0 else correct / total,
        "accuracy_effective": 0.0 if total == 0 else correct / total,
        "incomplete_final_response_rate": 0.0,
        "predictions": predictions,
    }


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    examples: Sequence[GSM8KExample],
    *,
    max_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
    initial_max_new_tokens: int = 0,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    correct = 0
    num_incomplete = 0
    num_truncated_prompts = 0
    predictions: List[Dict[str, Any]] = []
    require_final_response = bool(getattr(tokenizer, "_codex_chat_template_enable_thinking", False))
    effective_batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(examples), effective_batch_size):
        batch_examples = list(examples[batch_start: batch_start + effective_batch_size])
        prompt_texts = [
            render_qwen_generation_prompt(
                tokenizer,
                build_qwen_messages(_build_gsm8k_prompt(example)),
            )
            for example in batch_examples
        ]
        generation_results = _generate_with_retry_for_final_response(
            model,
            tokenizer,
            prompt_texts,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            initial_max_new_tokens=initial_max_new_tokens,
            require_final_response=require_final_response,
        )
        for row_idx, example in enumerate(batch_examples):
            gen_info = generation_results[row_idx]
            generated_text = gen_info["raw_text"]
            final_text = gen_info["final_text"]
            incomplete = bool(gen_info["incomplete_final_response"])
            if incomplete:
                num_incomplete += 1
            if gen_info["possibly_truncated_prompt"]:
                num_truncated_prompts += 1
            prediction = (
                "" if incomplete else extract_official_gsm8k_prediction(final_text)
            )
            is_correct = int(_gsm8k_answers_match(example.final_answer, prediction))
            correct += is_correct
            predictions.append(
                {
                    "id": example.sample_id,
                    "question": example.question,
                    "prediction": prediction,
                    "target": example.final_answer,
                    "correct": bool(is_correct),
                    "generated_text": generated_text,
                    "final_text": final_text,
                    "incomplete_final_response": incomplete,
                    "retried_for_final_response": bool(gen_info["retried_for_final_response"]),
                    "used_max_new_tokens": int(gen_info["used_max_new_tokens"]),
                    "possibly_truncated_prompt": bool(gen_info["possibly_truncated_prompt"]),
                }
            )

    total = len(examples)
    effective = max(total - num_incomplete, 0)
    return {
        "status": "ok",
        "num_samples": total,
        "num_correct": correct,
        "num_incomplete": num_incomplete,
        "num_effective": effective,
        "num_possibly_truncated_prompts": num_truncated_prompts,
        "accuracy": 0.0 if total == 0 else correct / total,
        "accuracy_effective": 0.0 if effective == 0 else correct / effective,
        "incomplete_final_response_rate": 0.0 if total == 0 else num_incomplete / total,
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


def _trim_to_code_start(text: str) -> str:
    lines = str(text).splitlines()
    code_start_patterns = [
        r"^\s*@",
        r"^\s*def\b",
        r"^\s*class\b",
        r"^\s*(?:from|import)\b",
        r"^\s*(?:if|for|while|try|with)\b",
        r"^\s*(?:return|pass|raise|yield|assert)\b",
        r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=",
    ]
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(pattern, line) for pattern in code_start_patterns):
            return "\n".join(lines[idx:]).strip()
    return str(text).strip()


def _truncate_to_longest_python_prefix(text: str) -> str:
    candidate = str(text).strip()
    if not candidate:
        return ""
    lines = candidate.splitlines()
    for end_idx in range(len(lines), 0, -1):
        prefix = "\n".join(lines[:end_idx]).rstrip()
        if not prefix.strip():
            continue
        try:
            ast.parse(prefix)
        except SyntaxError:
            continue
        return prefix
    return candidate


def _assemble_humaneval_program(example: CodeExample, completion: str) -> str:
    candidate_code = _truncate_to_longest_python_prefix(str(example.prompt) + completion)
    tests_blob = "\n\n".join(example.tests)
    return f"{candidate_code}\n\n{tests_blob}\n\ncheck({example.entry_point})\n"


def _assemble_mbpp_program(example: CodeExample, candidate_code: str) -> str:
    candidate_code = _trim_to_code_start(candidate_code)
    candidate_code = _truncate_to_longest_python_prefix(candidate_code)
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
    batch_size: int = 1,
    initial_max_new_tokens: int = 0,
) -> Dict[str, Any]:
    device = _resolve_device(model)
    passed = 0
    executable = 0
    num_incomplete = 0
    num_truncated_prompts = 0
    predictions: List[Dict[str, Any]] = []
    dataset_key = str(dataset_name).strip().lower()
    require_final_response = bool(getattr(tokenizer, "_codex_chat_template_enable_thinking", False))

    effective_batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(examples), effective_batch_size):
        batch_examples = list(examples[batch_start: batch_start + effective_batch_size])
        prompt_texts: List[str] = []
        for example in batch_examples:
            setattr(example, "requires_final_response", require_final_response)
            if dataset_key == "humaneval":
                user_text = _build_humaneval_official_prompt(example)
            elif dataset_key == "mbpp":
                user_text = _build_mbpp_officialish_prompt(example)
            else:
                raise ValueError(f"Unsupported code dataset: {dataset_name}")
            prompt_texts.append(
                render_qwen_generation_prompt(
                    tokenizer,
                    build_qwen_messages(user_text),
                )
            )

        generation_results = _generate_with_retry_for_final_response(
            model,
            tokenizer,
            prompt_texts,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            initial_max_new_tokens=initial_max_new_tokens,
            require_final_response=require_final_response,
        )
        for row_idx, example in enumerate(batch_examples):
            gen_info = generation_results[row_idx]
            generated_text = gen_info["raw_text"]
            final_text_only = gen_info["final_text"]
            incomplete = bool(gen_info["incomplete_final_response"])
            if gen_info["possibly_truncated_prompt"]:
                num_truncated_prompts += 1

            prediction_base: Dict[str, Any] = {
                "task_id": example.task_id,
                "prompt": example.prompt,
                "entry_point": example.entry_point,
                "generated_text": generated_text,
                "final_text": final_text_only,
                "incomplete_final_response": incomplete,
                "retried_for_final_response": bool(gen_info["retried_for_final_response"]),
                "used_max_new_tokens": int(gen_info["used_max_new_tokens"]),
                "possibly_truncated_prompt": bool(gen_info["possibly_truncated_prompt"]),
            }

            if dataset_key == "humaneval" and not example.entry_point:
                prediction_base.update({"passed": False, "error": "Missing entry_point"})
                predictions.append(prediction_base)
                continue

            if incomplete:
                num_incomplete += 1
                prediction_base.update(
                    {
                        "passed": False,
                        "error": "incomplete_final_response",
                        "sanitized_code": "",
                        "assembled_program": "",
                    }
                )
                predictions.append(prediction_base)
                continue

            if dataset_key == "humaneval":
                completion = _extract_humaneval_completion(
                    generated_text,
                    example.entry_point,
                    require_final_response=require_final_response,
                )
                if not completion.strip():
                    num_incomplete += 1
                    prediction_base.update(
                        {
                            "passed": False,
                            "error": "empty_completion",
                            "sanitized_code": "",
                            "assembled_program": "",
                        }
                    )
                    predictions.append(prediction_base)
                    continue
                program = _assemble_humaneval_program(example, completion)
            else:
                candidate_code = _extract_mbpp_code(
                    generated_text,
                    example.entry_point,
                    require_final_response=require_final_response,
                )
                if not candidate_code.strip():
                    num_incomplete += 1
                    prediction_base.update(
                        {
                            "passed": False,
                            "error": "empty_completion",
                            "sanitized_code": "",
                            "assembled_program": "",
                        }
                    )
                    predictions.append(prediction_base)
                    continue
                program = _assemble_mbpp_program(example, candidate_code)

            executable += 1
            task_passed, error_text = _run_code_program(program, exec_timeout_seconds)
            passed += int(task_passed)
            prediction_base.update(
                {
                    "passed": bool(task_passed),
                    "error": error_text,
                    "sanitized_code": sanitize_code_generation(
                        generated_text,
                        require_final_response=require_final_response,
                    ),
                    "assembled_program": program,
                }
            )
            predictions.append(prediction_base)

    total = len(examples)
    return {
        "status": "ok",
        "num_samples": total,
        "num_executable": executable,
        "num_passed": passed,
        "num_incomplete": num_incomplete,
        "num_possibly_truncated_prompts": num_truncated_prompts,
        "pass_at_1": 0.0 if total == 0 else passed / total,
        "pass_at_1_effective": 0.0 if executable == 0 else passed / executable,
        "incomplete_final_response_rate": 0.0 if total == 0 else num_incomplete / total,
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
