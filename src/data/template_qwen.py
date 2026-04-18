from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_QWEN_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
_QWEN_SPECIAL_MARKER_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")


def _resolve_enable_thinking(tokenizer: Any, enable_thinking: bool | None) -> bool:
    if enable_thinking is False:
        raise ValueError("Non-thinking chat template mode is no longer supported for Qwen3.5.")
    if enable_thinking is True:
        return True
    inherited = getattr(tokenizer, "_codex_chat_template_enable_thinking", None)
    if inherited is False:
        raise ValueError("Non-thinking chat template mode is no longer supported for Qwen3.5.")
    if inherited is None:
        return True
    return bool(inherited)


def build_qwen_messages(
    user_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_system_prompt: bool = True,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if include_system_prompt and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def render_qwen_generation_prompt(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    enable_thinking: bool | None = True,
) -> str:
    resolved_enable_thinking = _resolve_enable_thinking(tokenizer, enable_thinking)
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        kwargs["enable_thinking"] = resolved_enable_thinking
        try:
            return tokenizer.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(list(messages), tokenize=False)

    text_parts: List[str] = []
    for message in messages:
        role = str(message["role"]).strip().upper()
        content = str(message["content"]).strip()
        text_parts.append(f"{role}: {content}")
    text_parts.append("ASSISTANT:")
    return "\n".join(text_parts)


def render_qwen_supervised_text(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    assistant_text: str,
    enable_thinking: bool | None = True,
) -> str:
    full_messages = list(messages) + [{"role": "assistant", "content": assistant_text}]
    resolved_enable_thinking = _resolve_enable_thinking(tokenizer, enable_thinking)
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        kwargs["enable_thinking"] = resolved_enable_thinking
        try:
            return tokenizer.apply_chat_template(full_messages, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(full_messages, tokenize=False)

    prompt = render_qwen_generation_prompt(tokenizer=tokenizer, messages=messages)
    return prompt + str(assistant_text)


def render_qwen_final_response_prefix(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    enable_thinking: bool | None = True,
) -> str:
    return render_qwen_supervised_text(
        tokenizer=tokenizer,
        messages=messages,
        assistant_text="",
        enable_thinking=enable_thinking,
    )


def strip_qwen_thinking_content(
    text: str,
    *,
    require_final_response: bool = False,
) -> str:
    raw_text = str(text or "")
    if not raw_text:
        return ""

    lowered = raw_text.lower()
    closing_tag = "</think>"
    if closing_tag in lowered:
        split_at = lowered.rfind(closing_tag)
        cleaned = raw_text[split_at + len(closing_tag) :]
    else:
        if require_final_response and "<think>" in lowered:
            return ""
        cleaned = raw_text

    cleaned = _QWEN_THINK_TAG_RE.sub("", cleaned)
    cleaned = _QWEN_SPECIAL_MARKER_RE.sub("", cleaned)
    return cleaned.strip()
