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
    if backend == "ppu":
        # PPU backend follows the torch_npu calling convention (eager mode +
        # torch.<backend>.set_device). Vendor SDKs that ship as an
        # out-of-tree extension via PyTorch's PrivateUse1 mechanism (e.g.
        # Cambricon CATCH, Ascend torch_npu) expose torch_ppu and
        # torch.ppu.set_device. If your PPU SDK uses a different module name
        # or device API, adjust the two calls below.
        import torch_ppu  # type: ignore[import-not-found]  # noqa: F401

        device = torch.device(device_name or "ppu:0")
        if hasattr(torch, "ppu") and hasattr(torch.ppu, "set_device"):
            torch.ppu.set_device(device)
        return {"backend": "ppu", "device": device, "xla_model": None}
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
    chat_template_enable_thinking: bool = False,
    runtime_backend: str = "",
    runtime_device: str = "",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "",
) -> Tuple[Any, Any, Dict[str, Any]]:
    resolved = Path(model_path)
    model_ref = str(resolved if resolved.exists() else model_path)
    thinking_enabled = bool(chat_template_enable_thinking)

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    # Qwen3 chat template terminates each turn with <|im_end|>. Base tokenizer
    # may default eos_token_id to <|endoftext|> which never appears in chat
    # generations -> model.generate() runs to max_new_tokens, training labels
    # contain no learnable EOS, OpenCompass HuggingFacewithChatTemplate cannot
    # stop. Force the chat-mode EOS when the vocab exposes <|im_end|>.
    # Reference: Qwen3 official ms-swift recipe.
    im_end_id: int | None = None
    try:
        candidate = tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(candidate, int) and candidate >= 0 and candidate != unk_id:
            im_end_id = int(candidate)
    except Exception:
        im_end_id = None
    if im_end_id is not None:
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.eos_token_id = im_end_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    setattr(tokenizer, "_codex_chat_template_enable_thinking", thinking_enabled)

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
    # Mirror chat-mode EOS into model.generation_config so model.generate()
    # and downstream save_pretrained checkpoints carry stop tokens that match
    # the rendered chat conversations. Append rather than overwrite so the
    # original eos_token_id (e.g. <|endoftext|>) still terminates legacy outputs.
    if im_end_id is not None and hasattr(model, "generation_config") and model.generation_config is not None:
        existing = model.generation_config.eos_token_id
        if existing is None:
            model.generation_config.eos_token_id = im_end_id
        elif isinstance(existing, (list, tuple)):
            if im_end_id not in existing:
                model.generation_config.eos_token_id = list(existing) + [im_end_id]
        elif isinstance(existing, int) and existing != im_end_id:
            model.generation_config.eos_token_id = [existing, im_end_id]
        if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    setattr(model, "_codex_runtime_backend", runtime["backend"] or str(device_map))
    setattr(model, "_codex_runtime_device", str(runtime["device"]) if runtime["device"] is not None else str(device_map))
    setattr(model, "_codex_xla_model", runtime["xla_model"])
    setattr(model, "_codex_chat_template_enable_thinking", thinking_enabled)
    model.eval()
    return tokenizer, model, extract_model_meta(model)
