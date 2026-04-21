from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_QWEN_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
_QWEN_SPECIAL_MARKER_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_QWEN_REASONING_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"thinking process:"
    r"|here(?:'s| is) a thinking process\b"
    r"|the user is asking me\b"
    r"|okay,\s*let'?s tackle this problem\b"
    r")",
    flags=re.IGNORECASE,
)
_QWEN_REASONING_SECTION_RE = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(?:\*+\s*)?(?:"
    r"final output generation"
    r"|final response"
    r"|final answer"
    r"|drafting the refusal"
    r")\s*:\s*"
)
_QWEN_REASONING_META_HEADER_RE = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(?:\*+\s*)?(?:"
    r"analyze the request"
    r"|identify"
    r"|evaluate"
    r"|determine"
    r"|drafting"
    r"|refining"
    r"|review"
    r"|final polish"
    r"|final plan"
    r"|decision"
    r"|plan"
    r"|wait\b"
    r"|self-correction"
    r")"
)
_QWEN_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s*")
_QWEN_QUOTED_FRAGMENT_RE = re.compile(r"[\"“](.+?)[\"”]")
_QWEN_RESPONSE_LINE_RE = re.compile(
    r"(?i)^(?:"
    r"i (?:cannot|can't|will not|won't|must decline|reject|am unable|cannot fulfill)"
    r"|that would be"
    r"|instead(?:,| of)?"
    r"|i can help with"
    r"|i can offer"
    r"|please consult"
    r"|focus on"
    r")\b"
)


def _resolve_enable_thinking(tokenizer: Any, enable_thinking: bool | None) -> bool:
    if enable_thinking is False:
        return False
    if enable_thinking is True:
        return True
    inherited = getattr(tokenizer, "_codex_chat_template_enable_thinking", None)
    if inherited is None:
        return False
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
    enable_thinking: bool | None = None,
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
    enable_thinking: bool | None = None,
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
    enable_thinking: bool | None = None,
) -> str:
    return render_qwen_supervised_text(
        tokenizer=tokenizer,
        messages=messages,
        assistant_text="",
        enable_thinking=enable_thinking,
    )


def _looks_like_qwen_reasoning_trace(text: str) -> bool:
    cleaned = _QWEN_SPECIAL_MARKER_RE.sub("", str(text or "")).strip()
    if not cleaned:
        return False
    if _QWEN_REASONING_PREAMBLE_RE.match(cleaned):
        return True
    lowered = cleaned.lower()
    if "final output generation" in lowered and re.search(r"(?m)^\s*\d+\.", cleaned):
        return True
    return False


def _normalize_reasoning_response_fragment(text: str) -> str:
    normalized = _QWEN_LIST_PREFIX_RE.sub("", str(text or "").strip())
    normalized = normalized.strip("*_` ")
    normalized = normalized.strip()
    if normalized.startswith(("(", "[")) and normalized.endswith((")", "]")) and len(normalized) < 120:
        return ""
    return normalized


def _join_reasoning_response_fragments(fragments: Sequence[str]) -> str:
    cleaned: List[str] = []
    for fragment in fragments:
        normalized = _normalize_reasoning_response_fragment(fragment)
        if not normalized:
            continue
        if cleaned and normalized == cleaned[-1]:
            continue
        cleaned.append(normalized)
    if not cleaned:
        return ""
    return "\n".join(cleaned).strip()


def _extract_qwen_reasoning_response(section_text: str) -> str:
    fragments: List[str] = []
    for raw_line in str(section_text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if fragments:
                break
            continue
        if _QWEN_REASONING_META_HEADER_RE.match(stripped):
            if fragments:
                break
            continue
        quoted = _QWEN_QUOTED_FRAGMENT_RE.findall(stripped)
        if quoted:
            fragments.extend(fragment.strip() for fragment in quoted if fragment.strip())
            continue
        normalized = _normalize_reasoning_response_fragment(stripped)
        if _QWEN_RESPONSE_LINE_RE.match(normalized):
            fragments.append(normalized)
            continue
        if fragments:
            break
    return _join_reasoning_response_fragments(fragments)


def _extract_final_response_from_qwen_reasoning(text: str) -> str:
    raw_text = str(text or "")
    if not _looks_like_qwen_reasoning_trace(raw_text):
        return ""

    matches = list(_QWEN_REASONING_SECTION_RE.finditer(raw_text))
    for match in reversed(matches):
        extracted = _extract_qwen_reasoning_response(raw_text[match.end() :])
        if extracted:
            first_line = extracted.splitlines()[0].strip() if extracted.splitlines() else extracted.strip()
            if _QWEN_RESPONSE_LINE_RE.match(first_line) or len(extracted.split()) >= 8:
                return extracted
    return ""


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
    cleaned = cleaned.strip()

    if require_final_response:
        extracted = _extract_final_response_from_qwen_reasoning(cleaned)
        if extracted:
            return extracted
        if _looks_like_qwen_reasoning_trace(cleaned):
            return ""

    return cleaned
