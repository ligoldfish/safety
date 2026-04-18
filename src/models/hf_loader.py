from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(torch_dtype: str):
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


def _resolve_runtime(runtime_backend: str = "", runtime_device: str = "") -> Dict[str, Any]:
    backend = str(runtime_backend or "").strip().lower()
    device_name = str(runtime_device or "").strip()
    if not backend:
        return {"backend": "", "device": None, "xla_model": None}

    if backend == "cpu":
        return {"backend": "cpu", "device": torch.device(device_name or "cpu"), "xla_model": None}
    if backend == "cuda":
        return {"backend": "cuda", "device": torch.device(device_name or "cuda:0"), "xla_model": None}
    if backend == "npu":
        import torch_npu  # noqa: F401

        device = torch.device(device_name or "npu:0")
        torch.npu.set_device(device)
        return {"backend": "npu", "device": device, "xla_model": None}
    if backend == "tpu":
        import torch_xla.core.xla_model as xm

        if not device_name or device_name == "xla":
            device = xm.xla_device()
        elif device_name.startswith("xla:"):
            ordinal = int(device_name.split(":", 1)[1])
            device = xm.xla_device(ordinal)
        else:
            raise ValueError(
                f"Unsupported TPU runtime device: {runtime_device}. Expected 'xla' or 'xla:<ordinal>'."
            )
        return {"backend": "tpu", "device": device, "xla_model": xm}
    raise ValueError(f"Unsupported runtime backend: {runtime_backend}")


def _extract_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Unsupported model architecture: cannot find transformer layers.")


def extract_model_meta(model: Any) -> Dict[str, Any]:
    layers = _extract_layers(model)
    return {
        "num_layers": int(len(layers)),
        "hidden_size": int(getattr(model.config, "hidden_size")),
        "vocab_size": int(getattr(model.config, "vocab_size")),
        "num_attention_heads": int(getattr(model.config, "num_attention_heads")),
    }


def load_hf_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    chat_template_enable_thinking: bool | None = None,
    runtime_backend: str = "",
    runtime_device: str = "",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "",
) -> Tuple[Any, Any, Dict[str, Any]]:
    resolved = Path(model_path)
    model_ref = str(resolved if resolved.exists() else model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    setattr(tokenizer, "_codex_chat_template_enable_thinking", chat_template_enable_thinking)

    runtime = _resolve_runtime(runtime_backend=runtime_backend, runtime_device=runtime_device)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "torch_dtype": _resolve_dtype(torch_dtype),
    }
    if runtime["backend"]:
        if device_map and str(device_map).lower() not in {"", "none", "cpu"}:
            model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    if runtime["device"] is not None:
        model.to(runtime["device"])
    setattr(model, "_codex_runtime_backend", runtime["backend"] or str(device_map))
    setattr(model, "_codex_runtime_device", str(runtime["device"]) if runtime["device"] is not None else str(device_map))
    setattr(model, "_codex_xla_model", runtime["xla_model"])
    setattr(model, "_codex_chat_template_enable_thinking", chat_template_enable_thinking)
    model.eval()
    return tokenizer, model, extract_model_meta(model)
