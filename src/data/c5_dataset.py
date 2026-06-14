"""C5 dual-pole safety dataset builder (PAN-format, native responses).

Harmful pole : ``declare-lab/HarmfulQA`` -- the bare ``question`` (vanilla,
  single-turn, prompt-intent harmful) paired with a NATIVE refusal extracted from
  the same row's ``blue_conversations`` (a ChatGPT safe-roleplay Chain-of-
  Utterances). The first genuine refusal turn (refusal-gated) becomes the SFT /
  L_out target, so the refusal is real, diverse and prompt-conditioned -- matching
  PAN's accept/rejected structure (PAN uses Llama-3.1-405B-generated refusals;
  here ChatGPT-generated, both native + 1:1 paired). Rows whose blue conversation
  has no genuine refusal either fall back to a deterministic template
  (``pick_refusal_template``) or are dropped, per ``refusal_fallback``.

Harmless pole: ``HuggingFaceH4/no_robots`` -- the user instruction paired with its
  NATIVE human-written assistant reply as the helpful comply target.

Both poles emit PAN-compatible records ``{id, label, user_text, messages,
  target_response, accept_response, rejected_response, ...}`` consumed identically
  by the SVD intent axis (prompt + label only) and ours / sft / distill
  (target_response). The bare ``question`` is the prompt so the SVD axis stays
  vanilla single-turn; the multi-turn CoU is used only to mine the refusal target.

CONTAMINATION: HarmfulQA / CatQA are published red-teaming benchmarks -> use
  train-only and NEVER report them as our eval (name a disjoint refusal benchmark).
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence

from src.data.safety_datasets import pick_refusal_template
from src.data.template_qwen import DEFAULT_SYSTEM_PROMPT, build_qwen_messages

# Require an explicit DECLINE (not just "I'm sorry") so weak deflections like
# "I'm sorry, I don't understand" are not mistaken for refusals.
HARMFULQA_REFUSAL_RE = re.compile(
    r"\b(i can'?t|i cannot|i can not|can'?t assist|cannot assist|can'?t help|cannot help|"
    r"won'?t (assist|help|provide|be able)|i will not|i must decline|can'?t provide|"
    r"cannot provide|cannot comply|can'?t comply|unable to (assist|help|provide|comply)|"
    r"not able to (assist|help|provide)|can'?t fulfill|cannot fulfill|can'?t support|cannot support)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _blue_gpt_turns(blue: Any) -> List[str]:
    """Flatten every assistant ("gpt") turn across all blue sub-conversations."""
    convs: List[Any] = []
    if isinstance(blue, dict):
        convs = [v for v in blue.values() if isinstance(v, list)]
    elif isinstance(blue, list):
        convs = [blue]
    turns: List[str] = []
    for conv in convs:
        for turn in conv:
            if isinstance(turn, dict) and str(turn.get("from")) == "gpt":
                turns.append(str(turn.get("value", "")))
    return turns


def extract_harmfulqa_refusal(blue: Any) -> Optional[str]:
    """First genuine refusal turn mined from a HarmfulQA blue conversation, else None."""
    for turn in _blue_gpt_turns(blue):
        if HARMFULQA_REFUSAL_RE.search(turn):
            return turn.strip()
    return None


def _record(
    *,
    rec_id: str,
    label: str,
    user_text: str,
    target_response: str,
    accept_response: str,
    rejected_response: str,
    source_dataset: str,
    source: str,
    category: str,
    system_prompt: str,
) -> Dict[str, Any]:
    return {
        "id": rec_id,
        "source_dataset": source_dataset,
        "label": label,
        "user_text": user_text,
        "target_response": target_response,
        "accept_response": accept_response,
        "rejected_response": rejected_response,
        "messages": build_qwen_messages(user_text, system_prompt=system_prompt),
        "category": category,
        "source": source,
    }


def harmfulqa_records(
    rows: Sequence[Dict[str, Any]],
    *,
    refusal_fallback: str = "template",
    seed: int = 42,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Build harmful records from HarmfulQA rows.

    For each row: prompt = ``question``; target = the first native refusal mined
    from ``blue_conversations``. When no native refusal exists, ``refusal_fallback``
    decides: "template" -> deterministic ``pick_refusal_template``; "drop" -> skip.
    Returns ``{"records": [...], "n_native": int, "n_template": int, "n_dropped": int}``.
    """
    if refusal_fallback not in {"template", "drop"}:
        raise ValueError("refusal_fallback must be 'template' or 'drop'")
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    n_native = n_template = n_dropped = 0
    for idx, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        if not question:
            n_dropped += 1
            continue
        key = _norm(question)
        if key in seen:
            n_dropped += 1
            continue
        refusal = extract_harmfulqa_refusal(row.get("blue_conversations"))
        if refusal:
            n_native += 1
        elif refusal_fallback == "template":
            refusal = pick_refusal_template(question, seed=seed)
            n_template += 1
        else:
            n_dropped += 1
            continue
        seen.add(key)
        category = "/".join(
            p for p in (str(row.get("topic") or ""), str(row.get("subtopic") or "")) if p
        )
        out.append(
            _record(
                rec_id=f"c5_harmful_{idx:06d}",
                label="harmful",
                user_text=question,
                target_response=refusal,
                accept_response="",  # we never store the compliant harmful answer
                rejected_response=refusal,
                source_dataset="harmfulqa",
                source="declare-lab/HarmfulQA:blue_conversations",
                category=category,
                system_prompt=system_prompt,
            )
        )
    return {"records": out, "n_native": n_native, "n_template": n_template, "n_dropped": n_dropped}


def _no_robots_response(row: Dict[str, Any]) -> str:
    """Native human assistant reply from a No Robots row (last assistant turn)."""
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and str(msg.get("role")) == "assistant":
                content = str(msg.get("content") or "").strip()
                if content:
                    return content
    return str(row.get("response") or "").strip()


def _no_robots_prompt(row: Dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and str(msg.get("role")) == "user":
                return str(msg.get("content") or "").strip()
    return ""


def no_robots_records(
    rows: Sequence[Dict[str, Any]],
    *,
    drop_categories: Sequence[str] = ("Chat",),
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Build harmless records from No Robots rows (native human helpful target)."""
    drop = {str(c) for c in drop_categories}
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    n_dropped = 0
    for idx, row in enumerate(rows):
        if str(row.get("category")) in drop:
            n_dropped += 1
            continue
        prompt = _no_robots_prompt(row)
        response = _no_robots_response(row)
        if not prompt or not response:
            n_dropped += 1
            continue
        key = _norm(prompt)
        if key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        out.append(
            _record(
                rec_id=f"c5_harmless_{idx:06d}",
                label="harmless",
                user_text=prompt,
                target_response=response,
                accept_response=response,
                rejected_response="",
                source_dataset="no_robots",
                source="HuggingFaceH4/no_robots",
                category=str(row.get("category") or ""),
                system_prompt=system_prompt,
            )
        )
    return {"records": out, "n_dropped": n_dropped}


def build_c5_dataset(
    harmful_rows: Sequence[Dict[str, Any]],
    benign_rows: Sequence[Dict[str, Any]],
    *,
    n_per_pole: int = 0,
    test_fraction: float = 0.1,
    refusal_fallback: str = "template",
    seed: int = 42,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Materialize the balanced C5 dataset (PAN-format) as train/test record lists.

    ``n_per_pole=0`` balances to ``min(usable_harmful, usable_benign)``. The poles
    are balanced 1:1, deterministically shuffled (``seed``) and split per pole into
    test (``test_fraction``) + train so both splits stay balanced. Returns a dict
    with ``train``, ``test`` and a ``summary`` of pole counts / refusal provenance.
    """
    harm = harmfulqa_records(
        harmful_rows, refusal_fallback=refusal_fallback, seed=seed, system_prompt=system_prompt
    )
    ben = no_robots_records(benign_rows, system_prompt=system_prompt)
    harm_recs, ben_recs = harm["records"], ben["records"]

    cap = min(len(harm_recs), len(ben_recs))
    if n_per_pole > 0:
        cap = min(cap, n_per_pole)

    rng = random.Random(seed)
    rng.shuffle(harm_recs)
    rng.shuffle(ben_recs)
    harm_recs = harm_recs[:cap]
    ben_recs = ben_recs[:cap]

    n_test = int(round(cap * test_fraction))

    def _split(recs: List[Dict[str, Any]], split_name: str) -> None:
        for rec in recs[:n_test]:
            rec["pan_split"] = "test"
        for rec in recs[n_test:]:
            rec["pan_split"] = "train"

    _split(harm_recs, "harmful")
    _split(ben_recs, "harmless")

    train = [r for r in harm_recs + ben_recs if r["pan_split"] == "train"]
    test = [r for r in harm_recs + ben_recs if r["pan_split"] == "test"]
    rng.shuffle(train)
    rng.shuffle(test)

    summary = {
        "n_per_pole": cap,
        "n_train": len(train),
        "n_test": len(test),
        "harmful_native_refusals": harm["n_native"],
        "harmful_template_refusals": harm["n_template"],
        "harmful_dropped": harm["n_dropped"],
        "benign_dropped": ben["n_dropped"],
        "refusal_fallback": refusal_fallback,
        "seed": seed,
    }
    return {"train": train, "test": test, "summary": summary}
