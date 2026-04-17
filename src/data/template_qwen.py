from __future__ import annotations

from typing import Any, Dict, List, Sequence


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


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
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
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
) -> str:
    full_messages = list(messages) + [{"role": "assistant", "content": assistant_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(full_messages, tokenize=False)

    prompt = render_qwen_generation_prompt(tokenizer=tokenizer, messages=messages)
    return prompt + str(assistant_text)
