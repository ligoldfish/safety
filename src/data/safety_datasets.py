"""Safety SFT dataset builders.

This module is the single place where the upstream safety SFT corpora are
loaded, filtered and converted into the project's canonical training record
format (``{"id", "messages", "target_response", "label", "source", ...}``).
Training scripts must consume the materialized JSONL written by these
builders rather than calling ``load_dataset`` themselves.

Supported training corpora (see plan):

* ``tulu3_safety``        - subset of ``allenai/tulu-3-sft-mixture`` filtered
                            to the three safety-focused source slices.
* ``safety_tuned_llamas`` - 2k Alpaca-format records from the
                            ``vinid/safety-tuned-llamas`` GitHub repository
                            (``data/safety_only_data_Instructions.json``).
* ``beavertails``         - ``PKU-Alignment/BeaverTails``; unsafe responses
                            are replaced with a project-wide refusal template.
* ``wildjailbreak``       - prompt labels from ``data_type`` with 20k/2k
                            default balanced train/test subsets.
* ``wildguardmix``        - prompt labels from ``prompt_harm_label``; response
                            labels are used only for target hygiene.
* ``hh_rlhf``             - ``harmless-base`` as harmful prompts and
                            ``helpful-base`` as harmless prompts, using chosen.
* ``beavertails_category``- BeaverTails prompt-level category labels; ``is_safe``
                            only selects the training target.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data.template_qwen import DEFAULT_SYSTEM_PROMPT
from src.utils.io import ensure_dir, write_json, write_jsonl


DEFAULT_SAFETY_REFUSAL_TEMPLATE = (
    "I can't help with requests that could cause harm or enable unsafe behavior. "
    "If your goal is legitimate or safety-related, I can help with a safer "
    "alternative, prevention guidance, or high-level information."
)

# Eight paraphrased refusals used when the upstream dataset's own response
# field is unsafe (or absent) and we need to substitute a synthetic refusal.
# Indexing is deterministic per (seed, prompt) via ``pick_refusal_template``
# so the same prompt always lands on the same template across reruns.
# The first slot is the historical single-string template so any consumer
# that still references DEFAULT_SAFETY_REFUSAL_TEMPLATE keeps the original
# wording.
DEFAULT_SAFETY_REFUSAL_TEMPLATES: Tuple[str, ...] = (
    DEFAULT_SAFETY_REFUSAL_TEMPLATE,
    (
        "I'm not able to help with that. If you're trying to understand the "
        "topic for safety, education or harm prevention I can offer general "
        "background, point you to vetted resources, or talk through a safer "
        "alternative."
    ),
    (
        "Sorry, I won't help with this. Following through could enable real "
        "harm, and that's not something I'll provide. I'm happy to discuss "
        "the underlying concern through a safer angle."
    ),
    (
        "I have to decline this request. Producing it could lead to harm to "
        "others or to you, and I'd rather not contribute to that outcome. "
        "If your goal is legitimate (research, prevention, support), let me "
        "know and I'll help from that direction instead."
    ),
    (
        "I'm going to pass on this. The kind of guidance you're asking for "
        "could enable serious harm, and it's outside what I can responsibly "
        "provide. I can speak to prevention, policy, or general safety "
        "background if that helps."
    ),
    (
        "I can't go down this path. The request would push me into producing "
        "content that could be misused. If there is a legal, safety-related "
        "or educational concern behind it, share more context and I'll try "
        "to help from there."
    ),
    (
        "Helping with this would put real people at risk, so I won't. I do "
        "want to be useful though: if there is a non-harmful version of what "
        "you're trying to accomplish, describe it and I'll work that with you."
    ),
    (
        "This is a request I'll decline. I'm built to avoid output that could "
        "cause harm, even indirectly, and that applies here. I'm glad to help "
        "with related but safer questions — let me know which angle matters "
        "to you and I'll take it from there."
    ),
)


def pick_refusal_template(prompt: str, *, seed: int = 42) -> str:
    """Deterministic per-prompt selection from ``DEFAULT_SAFETY_REFUSAL_TEMPLATES``.

    Uses sha256 over (seed, prompt) so behaviour is stable across Python
    processes (the built-in ``hash`` is randomised since Python 3.3 and
    would change selection from run to run). The same prompt under the
    same seed always returns the same template, so rebuilding a dataset
    is reproducible.
    """

    digest = hashlib.sha256(
        f"{int(seed)}|{prompt}".encode("utf-8")
    ).hexdigest()
    idx = int(digest, 16) % len(DEFAULT_SAFETY_REFUSAL_TEMPLATES)
    return DEFAULT_SAFETY_REFUSAL_TEMPLATES[idx]

DEFAULT_TULU3_SAFETY_SOURCES: Tuple[str, ...] = (
    "ai2-adapt-dev/tulu_v3.9_wildguardmix",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak",
    "ai2-adapt-dev/coconot_converted",
)

# Round 2: helpful slices used as in-domain harmless contrast for the
# tulu3_safety_v2 corpus. Both come from the same ``allenai/tulu-3-sft-mixture``
# train split, identified via the ``source`` field.
DEFAULT_TULU3_HELPFUL_SOURCES: Tuple[str, ...] = (
    "ai2-adapt-dev/personahub_math_v5_regen_149960",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
)

# WildJailbreak ``data_type`` -> binary harmful/harmless mapping for the v2
# Tülu3 builder. ``adversarial_*`` framings are styling, not content; the
# benign content stays harmless.
WILDJAILBREAK_HARMFUL_DATA_TYPES: frozenset[str] = frozenset(
    {"vanilla_harmful", "adversarial_harmful"}
)
WILDJAILBREAK_HARMLESS_DATA_TYPES: frozenset[str] = frozenset(
    {"vanilla_benign", "adversarial_benign"}
)
# WildGuardMix ``prompt_harm_label`` mapping.
WILDGUARDMIX_HARMFUL_LABELS: frozenset[str] = frozenset({"harmful"})
WILDGUARDMIX_HARMLESS_LABELS: frozenset[str] = frozenset({"unharmful"})

DEFAULT_SAFETY_TUNED_LLAMAS_FILE = "safety_only_data_Instructions.json"
SAFETY_TUNED_LLAMAS_REPO_URL = "https://github.com/vinid/safety-tuned-llamas"


@dataclass
class SafetyDatasetSpec:
    """Container for per-baseline dataset construction knobs.

    The fields below intentionally mirror the YAML keys under
    ``data.safety_dataset`` in the new safety baseline configs.
    """

    name: str
    output_path: str
    force_rebuild: bool = False
    cache_dir: Optional[str] = None
    source_name: Optional[str] = None
    split: Optional[str] = None
    sources: Optional[List[str]] = None
    repo_or_data_path: Optional[str] = None
    file_name: Optional[str] = None
    refusal_template: Optional[str] = None
    system_prompt: Optional[str] = None
    # Safety-Tuned LLaMAs harmless contrast (alpaca_small.json) toggle.
    include_harmless_contrast: bool = False
    harmless_file_name: str = "alpaca_small.json"
    harmless_max_samples: Optional[int] = None
    # BeaverTails de-duplication: 30k_train ships multiple is_safe-tagged
    # responses per prompt which inflates class imbalance after binary mapping.
    dedup_prompts: bool = True
    # BeaverTails label assignment. Round 2: prompt-level via the
    # ``category`` dict (any True -> harmful) is the new default; ``is_safe``
    # is response-level and biases toward the majority refusal style.
    label_strategy: str = "category_any"
    # Tülu3 v2: helpful slices to mix in as in-domain harmless contrast.
    helpful_sources: Optional[List[str]] = None
    helpful_max_samples: Optional[int] = None
    train_subset_mode: bool = True
    max_train_samples: int = 0
    max_train_samples_per_label: int = 0
    eval_subset_mode: bool = False
    max_eval_samples: int = 0
    max_eval_samples_per_label: int = 0
    eval_output_path: Optional[str] = None
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Imported lazily so unit tests can monkeypatch ``datasets.load_dataset``
    via this module without paying the import cost up front."""

    from datasets import load_dataset  # type: ignore[import-not-found]

    return load_dataset(*args, **kwargs)


def _coerce_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_messages, Iterable):
        raise ValueError("messages payload must be an iterable of role/content dicts")
    coerced: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError(f"Each message must be a dict, got {type(item).__name__}")
        role = str(item.get("role", "")).strip()
        content = item.get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(piece.get("text", "")) if isinstance(piece, dict) else str(piece)
                for piece in content
            )
        coerced.append({"role": role, "content": str(content)})
    return coerced


def _split_prompt_messages_and_target(
    messages: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], str]:
    """Return prompt-only messages plus the final assistant target.

    ``SupervisedCollator`` appends ``target_response`` after rendering
    ``messages`` as a generation prompt, so the final assistant answer must
    not remain in ``messages``.
    """

    normalized = list(messages)
    for index in range(len(normalized) - 1, -1, -1):
        if str(normalized[index].get("role", "")).strip().lower() == "assistant":
            target = str(normalized[index].get("content", "")).strip()
            prompt_messages = list(normalized[:index])
            # Some upstream transcripts contain literal "\n\nAssistant:" text
            # inside an assistant answer. The simple turn parser can split that
            # into consecutive assistant messages; merge those back into the
            # target so prompt messages still end at the user turn.
            while prompt_messages and str(prompt_messages[-1].get("role", "")).strip().lower() == "assistant":
                previous = str(prompt_messages.pop().get("content", "")).strip()
                if previous:
                    target = f"{previous}\n\n{target}" if target else previous
            return prompt_messages, target
    return normalized, ""


def _ensure_system_prompt(
    messages: Sequence[Dict[str, str]],
    *,
    system_prompt: str,
) -> List[Dict[str, str]]:
    coerced = list(messages)
    if not coerced or str(coerced[0].get("role", "")).lower() != "system":
        return [{"role": "system", "content": system_prompt}] + coerced
    return coerced


# ---------------------------------------------------------------------------
# Tulu 3 safety subset
# ---------------------------------------------------------------------------


def build_tulu3_safety_records(
    *,
    output_path: str | Path,
    sources: Sequence[str] = DEFAULT_TULU3_SAFETY_SOURCES,
    source_name: str = "allenai/tulu-3-sft-mixture",
    split: str = "train",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Materialize the Tülu 3 safety subset to ``output_path`` (JSONL).

    The filter happens here: only rows whose ``source`` field matches one
    of ``sources`` are kept. Each row's ``messages`` are normalized to the
    project's training schema and written as JSONL records.
    """

    accepted_sources = {str(value) for value in sources}
    if not accepted_sources:
        raise ValueError("`sources` must list at least one Tulu3 source slice")

    dataset = _load_dataset(source_name, split=split, cache_dir=cache_dir)
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(dataset):
        row_source = str(row.get("source", "")).strip()
        if row_source not in accepted_sources:
            continue
        messages = _coerce_messages(row.get("messages") or [])
        if not messages:
            continue
        messages = _ensure_system_prompt(messages, system_prompt=system_prompt)
        prompt_messages, target_response = _split_prompt_messages_and_target(messages)
        if not target_response:
            continue
        record_id = str(row.get("id") or f"tulu3_safety_{index:08d}")
        records.append(
            {
                "id": record_id,
                "messages": prompt_messages,
                "target_response": target_response,
                "label": "tulu3_safety",
                "source": row_source,
                "dataset": "tulu3_safety",
            }
        )

    if not records:
        raise RuntimeError(
            "Tulu3 safety filter produced zero records. "
            f"Check that one of {sorted(accepted_sources)} exists in {source_name}."
        )

    write_jsonl(output_path, records)
    return records


# ---------------------------------------------------------------------------
# Tülu 3 safety v2 (Round 2): raw harmful/harmless labels + personahub helpful
# ---------------------------------------------------------------------------


def _classify_wildjailbreak_data_type(value: str) -> Optional[str]:
    text = (value or "").strip().lower()
    if text in WILDJAILBREAK_HARMFUL_DATA_TYPES:
        return "harmful"
    if text in WILDJAILBREAK_HARMLESS_DATA_TYPES:
        return "harmless"
    return None


def _classify_wildguardmix_label(value: str) -> Optional[str]:
    text = (value or "").strip().lower()
    if text in WILDGUARDMIX_HARMFUL_LABELS:
        return "harmful"
    if text in WILDGUARDMIX_HARMLESS_LABELS:
        return "harmless"
    return None


def _build_v2_record(
    *,
    record_id: str,
    prompt: str,
    target: str,
    label: str,
    raw_label: str,
    source: str,
    system_prompt: str,
) -> Dict[str, Any]:
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "target_response": target,
        "label": label,
        "raw_label": raw_label,
        "source": source,
        "dataset": "tulu3_safety_v2",
    }


def _normalize_raw_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_binary_record(
    *,
    record_id: str,
    prompt: str,
    target: str,
    label: str,
    raw_label: str,
    source: str,
    dataset: str,
    system_prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "target_response": target,
        "label": label,
        "raw_label": raw_label,
        "source": source,
        "dataset": dataset,
        "metadata": dict(metadata or {}),
    }


def _validate_binary_training_records(records: Sequence[Dict[str, Any]]) -> None:
    valid_labels = {"harmful", "harmless"}
    for index, record in enumerate(records):
        label = str(record.get("label", "")).lower()
        if label not in valid_labels:
            raise ValueError(f"Record {index} has non-binary label: {label!r}")
        target = str(record.get("target_response", "")).strip()
        if not target:
            raise ValueError(f"Record {index} has empty target_response")
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Record {index} has invalid messages")
        if str(messages[-1].get("role", "")).lower() == "assistant":
            raise ValueError(f"Record {index} leaks assistant target into messages")


def _label_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        label = str(record.get("label", "")).lower()
        counts[label] = counts.get(label, 0) + 1
    return counts


def _sample_balanced_by_label(
    records: Sequence[Dict[str, Any]],
    *,
    subset_mode: bool,
    max_samples: int,
    max_samples_per_label: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not subset_mode or int(max_samples) <= 0:
        return list(records)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("label", "")).lower()].append(record)
    labels = [label for label in ("harmful", "harmless") if groups.get(label)]
    if not labels:
        return []
    per_label = int(max_samples_per_label)
    if per_label <= 0:
        per_label = max(int(max_samples) // len(labels), 1)
    sampled: List[Dict[str, Any]] = []
    for offset, label in enumerate(labels):
        bucket = list(groups[label])
        random.Random(int(seed) + offset).shuffle(bucket)
        sampled.extend(bucket[:per_label])
    if len(sampled) > int(max_samples):
        rng = random.Random(int(seed) + 17)
        rng.shuffle(sampled)
        sampled = sampled[: int(max_samples)]
    return sampled


def _sample_by_metadata_key(
    records: Sequence[Dict[str, Any]],
    *,
    metadata_key: str,
    subset_mode: bool,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not subset_mode or int(max_samples) <= 0:
        return list(records)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        key = str(metadata.get(metadata_key) or record.get(metadata_key) or record.get("raw_label") or "")
        if key:
            groups[key].append(record)
    if not groups:
        return _sample_balanced_by_label(
            records,
            subset_mode=subset_mode,
            max_samples=max_samples,
            max_samples_per_label=0,
            seed=seed,
        )
    per_group = max(int(max_samples) // len(groups), 1)
    sampled: List[Dict[str, Any]] = []
    for offset, key in enumerate(sorted(groups)):
        bucket = list(groups[key])
        random.Random(int(seed) + offset).shuffle(bucket)
        sampled.extend(bucket[:per_group])
    if len(sampled) > int(max_samples):
        rng = random.Random(int(seed) + 29)
        rng.shuffle(sampled)
        sampled = sampled[: int(max_samples)]
    return sampled


def _to_eval_record(record: Dict[str, Any], *, id_prefix: str = "eval") -> Dict[str, Any]:
    return {
        "id": f"{id_prefix}_{record.get('id', '')}",
        "label": str(record.get("label", "")).lower(),
        "messages": list(record.get("messages") or []),
        "source": record.get("source"),
        "dataset": record.get("dataset"),
        "raw_label": record.get("raw_label"),
        "metadata": record.get("metadata", {}),
    }


def _write_materialization_summary(
    *,
    output_path: str | Path,
    dataset_name: str,
    train_records: Sequence[Dict[str, Any]],
    train_pool_count: int,
    eval_records: Sequence[Dict[str, Any]] = (),
    eval_output_path: str | Path | None = None,
    train_subset_mode: bool,
    eval_subset_mode: bool,
    seed: int,
    test_source: str = "",
    test_fallback_reason: str = "",
    drops: Optional[Dict[str, int]] = None,
) -> None:
    output = Path(output_path)
    summary = {
        "dataset_name": dataset_name,
        "train_output_path": str(output),
        "train_pool_count": int(train_pool_count),
        "train_written_count": len(train_records),
        "train_label_counts": _label_counts(train_records),
        "train_subset_mode": bool(train_subset_mode),
        "eval_output_path": str(eval_output_path) if eval_output_path else "",
        "eval_written_count": len(eval_records),
        "eval_label_counts": _label_counts(eval_records),
        "eval_subset_mode": bool(eval_subset_mode),
        "test_source": test_source,
        "test_fallback_used": bool(test_fallback_reason),
        "test_fallback_reason": test_fallback_reason,
        "seed": int(seed),
        "drops": dict(drops or {}),
        "full_test": not bool(eval_subset_mode),
    }
    write_json(output.with_suffix(output.suffix + ".summary.json"), summary)
    if eval_output_path:
        write_json(Path(eval_output_path).with_suffix(Path(eval_output_path).suffix + ".summary.json"), summary)


def _format_fallback_reason(exc: BaseException) -> str:
    message = str(exc).strip()
    reason = type(exc).__name__
    if message:
        reason = f"{reason}: {message}"
    return reason[:1000]


def _existing_eval_matches_request(
    *,
    eval_path: Path | None,
    eval_subset_mode: bool,
    max_eval_samples: int,
) -> bool:
    if eval_path is None:
        return True
    if not eval_path.exists():
        return False
    requested_full = (not bool(eval_subset_mode)) or int(max_eval_samples) <= 0
    if not requested_full:
        return True
    summary_path = eval_path.with_suffix(eval_path.suffix + ".summary.json")
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(summary.get("full_test")) and not bool(summary.get("eval_subset_mode"))


def _iter_dataset_rows(loaded: Any) -> Iterable[Dict[str, Any]]:
    """Iterate either a HF Dataset or a DatasetDict's first split.

    HF ``load_dataset`` returns a ``DatasetDict`` (a dict subclass keyed
    by split name) when ``split=`` is omitted. ``DatasetDict`` is itself
    iterable, but iteration yields split names (strs), not rows. A plain
    HF ``Dataset`` is not a dict subclass, so ``isinstance(..., dict)``
    cleanly separates the two cases.
    """

    if isinstance(loaded, dict):
        keys = list(loaded.keys())
        if keys:
            return loaded[keys[0]]
    return loaded


def _collect_wildguardmix_train_rows(
    *,
    source_name: str,
    config_name: str,
    cache_dir: Optional[str],
    system_prompt: str,
    refusal_template: str,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Pull harmful/harmless rows directly from upstream WildGuardMix train.

    The Tülu3 SFT mixture strips ``prompt_harm_label`` so we cannot rely
    on it; the upstream dataset still carries the field.
    """

    try:
        loaded = _load_dataset(source_name, config_name, cache_dir=cache_dir)
    except Exception as exc:  # network / gated access failure
        raise RuntimeError(
            f"Failed to load {source_name}:{config_name} for tulu3_safety_v2 "
            f"harmful side. Underlying error: {exc!r}. Ensure the dataset is "
            "accessible and the HF token has access."
        ) from exc
    rows = _iter_dataset_rows(loaded)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt = str(row.get("prompt", "") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        raw_label = str(row.get("prompt_harm_label", "")).strip()
        mapped = _classify_wildguardmix_label(raw_label)
        if mapped is None:
            continue
        response = str(row.get("response", "") or "").strip()
        response_harm = str(row.get("response_harm_label", "")).strip().lower()
        if mapped == "harmless":
            # Skip rows whose response itself is harmful even though the
            # prompt is benign — keeping them would teach unsafe completion
            # on a harmless prompt.
            if not response or response_harm == "harmful":
                continue
            target = response
        else:
            # Harmful prompt: prefer the upstream response when WildGuardMix
            # tagged it as a safe completion (``response_harm_label="unharmful"``)
            # so we ride the dataset's own natural-language refusal/redirect
            # text instead of substituting one synthetic template. Fall back
            # to the per-prompt refusal pool when the upstream response is
            # missing or itself unsafe — both cases would otherwise produce
            # bad training targets.
            if response and response_harm == "unharmful":
                target = response
            else:
                target = pick_refusal_template(prompt, seed=seed)
        out.append(
            _build_v2_record(
                record_id=str(row.get("id") or f"tulu3v2_wgm_{index:08d}"),
                prompt=prompt,
                target=target,
                label=mapped,
                raw_label=raw_label,
                source=f"{source_name}:{config_name}",
                system_prompt=system_prompt,
            )
        )
    return out


def _collect_wildjailbreak_train_rows(
    *,
    source_name: str,
    config_name: str,
    cache_dir: Optional[str],
    system_prompt: str,
    refusal_template: str,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    try:
        loaded = _load_dataset(
            source_name,
            config_name,
            delimiter="\t",
            keep_default_na=False,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {source_name}:{config_name} for tulu3_safety_v2 "
            f"harmful side. Underlying error: {exc!r}. WildJailbreak is gated; "
            "accept the license on huggingface.co and retry with HF_TOKEN set."
        ) from exc
    rows = _iter_dataset_rows(loaded)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        # Adversarial prompts dominate the train split; fall back to the
        # vanilla rendition when adversarial is empty.
        prompt = (
            str(row.get("adversarial", "") or "").strip()
            or str(row.get("vanilla", "") or "").strip()
        )
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        raw_label = str(row.get("data_type", "") or row.get("type", "")).strip()
        mapped = _classify_wildjailbreak_data_type(raw_label)
        if mapped is None:
            continue
        if mapped == "harmless":
            target = str(row.get("completion", "") or "").strip()
            if not target:
                continue
        else:
            # WildJailbreak ``completion`` for harmful (adversarial / vanilla)
            # rows is the upstream successful-jailbreak output and is unsafe
            # by construction, so we cannot ride it. Use the per-prompt
            # refusal pool so distinct prompts land on distinct refusals.
            target = pick_refusal_template(prompt, seed=seed)
        out.append(
            _build_v2_record(
                record_id=str(row.get("id") or f"tulu3v2_wjb_{index:08d}"),
                prompt=prompt,
                target=target,
                label=mapped,
                raw_label=raw_label,
                source=f"{source_name}:{config_name}",
                system_prompt=system_prompt,
            )
        )
    return out


def build_tulu3_safety_v2_records(
    *,
    output_path: str | Path,
    source_name: str = "allenai/tulu-3-sft-mixture",
    split: str = "train",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    helpful_sources: Sequence[str] = DEFAULT_TULU3_HELPFUL_SOURCES,
    helpful_max_samples: Optional[int] = None,
    seed: int = 42,
    refusal_template: str = DEFAULT_SAFETY_REFUSAL_TEMPLATE,
    wildguardmix_source: str = "allenai/wildguardmix",
    wildguardmix_config: str = "wildguardtrain",
    wildjailbreak_source: str = "allenai/wildjailbreak",
    wildjailbreak_config: str = "train",
) -> List[Dict[str, Any]]:
    """Build the Tülu3 safety v2 corpus (Round 2 contrast).

    Differs from :func:`build_tulu3_safety_records` in two ways:

    1. The harmful / harmless safety rows come from the **upstream**
       ``allenai/wildguardmix`` (config ``wildguardtrain``) and
       ``allenai/wildjailbreak`` (config ``train``) datasets directly,
       not from the Tülu3 SFT mixture. The mixture re-packages them as
       ``{id, messages, source}`` and drops ``prompt_harm_label`` /
       ``data_type``, which we need for binary supervision.
    2. Helpful in-domain harmless contrast is mixed in from
       ``allenai/tulu-3-sft-mixture`` via ``helpful_sources`` (default:
       personahub_math + open_math_2_gsm8k). This replaces the previous
       PAN-harmless cross-corpus injection that 20_split fell back to
       when every Tülu3 row was tagged ``harmful``.

    CoCoNot-converted rows are intentionally excluded from this builder;
    they are kept in :func:`load_coconot_contrast` as a separate
    over-refusal eval signal.

    ``helpful_max_samples`` caps the personahub side. ``None`` (default)
    means: top up only enough so harmless ≈ harmful after counting the
    in-domain harmless rows already pulled from WildGuardMix /
    WildJailbreak.
    """

    accepted_helpful = {str(value) for value in helpful_sources}
    if not accepted_helpful:
        raise ValueError("`helpful_sources` must list at least one helpful slice")

    safety_records: List[Dict[str, Any]] = []
    safety_records.extend(
        _collect_wildguardmix_train_rows(
            source_name=wildguardmix_source,
            config_name=wildguardmix_config,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            refusal_template=refusal_template,
            seed=seed,
        )
    )
    safety_records.extend(
        _collect_wildjailbreak_train_rows(
            source_name=wildjailbreak_source,
            config_name=wildjailbreak_config,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            refusal_template=refusal_template,
            seed=seed,
        )
    )

    if not safety_records:
        raise RuntimeError(
            "tulu3_safety_v2 harmful side produced zero records. Verify that "
            f"{wildguardmix_source}:{wildguardmix_config} and "
            f"{wildjailbreak_source}:{wildjailbreak_config} are both reachable."
        )

    helpful_pool: List[Dict[str, Any]] = []
    try:
        mixture = _load_dataset(source_name, split=split, cache_dir=cache_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {source_name}:{split} for personahub helpful "
            f"slice. Underlying error: {exc!r}."
        ) from exc

    for index, row in enumerate(mixture):
        row_source = str(row.get("source", "")).strip()
        if row_source not in accepted_helpful:
            continue
        messages = _coerce_messages(row.get("messages") or [])
        if not messages:
            continue
        messages = _ensure_system_prompt(messages, system_prompt=system_prompt)
        prompt_messages, target_response = _split_prompt_messages_and_target(messages)
        if not target_response:
            continue
        user_text = ""
        for message in prompt_messages:
            if str(message.get("role", "")).lower() == "user":
                user_text = str(message.get("content", ""))
                break
        if not user_text:
            continue
        helpful_pool.append(
            _build_v2_record(
                record_id=str(row.get("id") or f"tulu3v2_helpful_{index:08d}"),
                prompt=user_text,
                target=target_response,
                label="harmless",
                raw_label="personahub_helpful",
                source=row_source,
                system_prompt=system_prompt,
            )
        )

    n_harmful = sum(1 for record in safety_records if record["label"] == "harmful")
    n_harmless_safety = sum(
        1 for record in safety_records if record["label"] == "harmless"
    )
    if helpful_max_samples is None:
        # Default: top up helpful slice only as much as needed so the
        # harmless half (WildGuardMix unharmful + WildJailbreak benign +
        # personahub) does not overshoot the harmful half.
        helpful_cap = max(n_harmful - n_harmless_safety, 0)
    else:
        helpful_cap = max(int(helpful_max_samples), 0)

    if helpful_pool and helpful_cap > 0:
        rng = random.Random(int(seed))
        rng.shuffle(helpful_pool)
        helpful_records = helpful_pool[:helpful_cap]
    else:
        helpful_records = []

    records = safety_records + helpful_records
    write_jsonl(output_path, records)
    return records


# ---------------------------------------------------------------------------
# Standalone WildJailbreak / WildGuardMix / HH-RLHF builders
# ---------------------------------------------------------------------------


def _load_wildjailbreak_rows(
    *,
    source_name: str,
    config_name: str,
    cache_dir: Optional[str],
) -> Iterable[Dict[str, Any]]:
    loaded = _load_dataset(
        source_name,
        config_name,
        delimiter="\t",
        keep_default_na=False,
        cache_dir=cache_dir,
    )
    return _iter_dataset_rows(loaded)


def _wildjailbreak_prompt(row: Dict[str, Any]) -> str:
    return (
        str(row.get("adversarial", "") or "").strip()
        or str(row.get("vanilla", "") or "").strip()
        or str(row.get("prompt", "") or "").strip()
    )


def _safe_wildjailbreak_harmful_target(
    *,
    prompt: str,
    completion: str,
    seed: int,
) -> Tuple[str, Dict[str, Any]]:
    if not completion.strip():
        return pick_refusal_template(prompt, seed=seed), {
            "target_source": "template_pool",
            "target_filter_reason": "empty_completion",
        }
    try:
        from src.training.eval_utils import judge_harmful_response_safety
    except ModuleNotFoundError:
        # In lightweight data-builder tests torch may be absent, and importing
        # src.training runs package-level training imports before eval_utils.
        # Load the standalone heuristic file directly; it has no torch
        # dependency and preserves the repository's existing safety filter.
        import importlib.util
        import sys

        eval_utils_path = Path(__file__).resolve().parents[1] / "training" / "eval_utils.py"
        spec = importlib.util.spec_from_file_location(
            "_safety_eval_utils_standalone",
            eval_utils_path,
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        judge_harmful_response_safety = module.judge_harmful_response_safety

    judgment = judge_harmful_response_safety(completion)
    if judgment.is_safe:
        return completion, {
            "target_source": "upstream_completion",
            "target_filter_reason": judgment.reason,
            "target_is_refusal": bool(judgment.is_refusal),
        }
    return pick_refusal_template(prompt, seed=seed), {
        "target_source": "template_pool",
        "target_filter_reason": judgment.reason,
        "target_is_refusal": bool(judgment.is_refusal),
    }


def _build_wildjailbreak_pool(
    rows: Iterable[Dict[str, Any]],
    *,
    source_name: str,
    config_name: str,
    system_prompt: str,
    seed: int,
    for_eval: bool = False,
    drops: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    records: List[Dict[str, Any]] = []
    drop_counts = drops if drops is not None else {}
    for index, row in enumerate(rows):
        prompt = _wildjailbreak_prompt(dict(row))
        if not prompt:
            drop_counts["missing_prompt"] = drop_counts.get("missing_prompt", 0) + 1
            continue
        data_type = _normalize_raw_label(row.get("data_type") or row.get("type"))
        label = _classify_wildjailbreak_data_type(data_type)
        if label is None:
            drop_counts["unknown_data_type"] = drop_counts.get("unknown_data_type", 0) + 1
            continue
        dedup_key = f"{data_type}|{prompt}"
        if dedup_key in seen:
            drop_counts["duplicate_prompt"] = drop_counts.get("duplicate_prompt", 0) + 1
            continue
        seen.add(dedup_key)
        completion = str(row.get("completion", "") or row.get("response", "") or "").strip()
        metadata: Dict[str, Any] = {
            "data_type": data_type,
            "source_config": config_name,
        }
        if label == "harmless":
            if not completion and not for_eval:
                drop_counts["harmless_missing_completion"] = (
                    drop_counts.get("harmless_missing_completion", 0) + 1
                )
                continue
            target = completion or pick_refusal_template(prompt, seed=seed)
            metadata["target_source"] = "upstream_completion"
        else:
            target, target_meta = _safe_wildjailbreak_harmful_target(
                prompt=prompt,
                completion=completion,
                seed=seed,
            )
            metadata.update(target_meta)
        records.append(
            _build_binary_record(
                record_id=str(row.get("id") or f"wildjailbreak_{config_name}_{index:08d}"),
                prompt=prompt,
                target=target,
                label=label,
                raw_label=data_type,
                source=f"{source_name}:{config_name}",
                dataset="wildjailbreak",
                system_prompt=system_prompt,
                metadata=metadata,
            )
        )
    return records


def build_wildjailbreak_records(
    *,
    output_path: str | Path,
    source_name: str = "allenai/wildjailbreak",
    split: str = "train",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    train_subset_mode: bool = True,
    max_train_samples: int = 20000,
    max_train_samples_per_label: int = 10000,
    eval_subset_mode: bool = False,
    max_eval_samples: int = 0,
    max_eval_samples_per_label: int = 0,
    eval_output_path: str | Path | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    drops: Dict[str, int] = {}
    train_pool = _build_wildjailbreak_pool(
        _load_wildjailbreak_rows(
            source_name=source_name,
            config_name=split or "train",
            cache_dir=cache_dir,
        ),
        source_name=source_name,
        config_name=split or "train",
        system_prompt=system_prompt,
        seed=seed,
        drops=drops,
    )
    records = _sample_by_metadata_key(
        train_pool,
        metadata_key="data_type",
        subset_mode=train_subset_mode,
        max_samples=max_train_samples,
        seed=seed,
    )
    # If a split lacks all four data_type buckets, fall back to a binary
    # harmful/harmless cap so callers still get the requested 10k/10k shape.
    if len({r.get("raw_label") for r in records}) < 4:
        records = _sample_balanced_by_label(
            train_pool,
            subset_mode=train_subset_mode,
            max_samples=max_train_samples,
            max_samples_per_label=max_train_samples_per_label,
            seed=seed,
        )
    _validate_binary_training_records(records)
    write_jsonl(output_path, records)

    eval_records: List[Dict[str, Any]] = []
    test_source = ""
    test_fallback_reason = ""
    if eval_output_path:
        try:
            eval_pool = _build_wildjailbreak_pool(
                _load_wildjailbreak_rows(
                    source_name=source_name,
                    config_name="eval",
                    cache_dir=cache_dir,
                ),
                source_name=source_name,
                config_name="eval",
                system_prompt=system_prompt,
                seed=seed,
                for_eval=True,
                drops=drops,
            )
            if len({r["label"] for r in eval_pool}) < 2:
                raise RuntimeError("WildJailbreak eval does not contain both labels")
            test_source = f"{source_name}:eval"
        except Exception as exc:
            test_fallback_reason = _format_fallback_reason(exc)
            selected_ids = {str(record.get("id", "")) for record in records}
            eval_pool = [record for record in train_pool if str(record.get("id", "")) not in selected_ids]
            test_source = f"{source_name}:{split or 'train'} holdout"
        eval_selected = _sample_balanced_by_label(
            eval_pool,
            subset_mode=eval_subset_mode,
            max_samples=max_eval_samples,
            max_samples_per_label=max_eval_samples_per_label,
            seed=seed + 101,
        )
        eval_records = [_to_eval_record(record, id_prefix="wildjailbreak_test") for record in eval_selected]
        write_jsonl(eval_output_path, eval_records)

    _write_materialization_summary(
        output_path=output_path,
        dataset_name="wildjailbreak",
        train_records=records,
        train_pool_count=len(train_pool),
        eval_records=eval_records,
        eval_output_path=eval_output_path,
        train_subset_mode=bool(train_subset_mode) and int(max_train_samples) > 0,
        eval_subset_mode=bool(eval_subset_mode) and int(max_eval_samples) > 0,
        seed=seed,
        test_source=test_source,
        test_fallback_reason=test_fallback_reason,
        drops=drops,
    )
    return records


def _response_harm_is_safe(value: Any) -> bool:
    text = _normalize_raw_label(value)
    return text in {"unharmful", "harmless", "safe", "benign"}


def _response_is_refusal(value: Any) -> bool:
    text = _normalize_raw_label(value)
    if not text:
        return False
    if text in {"compliance", "non-refusal", "non_refusal", "nonrefusal", "answer"}:
        return False
    return "refusal" in text or text in {"refuse", "refused"}


def _response_is_explicit_compliance(value: Any) -> bool:
    text = _normalize_raw_label(value)
    return text in {"compliance", "non-refusal", "non_refusal", "nonrefusal", "answer"}


def _response_text_looks_like_refusal(text: str) -> bool:
    if not str(text or "").strip():
        return False
    try:
        from src.training.eval_utils import looks_like_refusal
    except ModuleNotFoundError:
        import importlib.util
        import sys

        eval_utils_path = Path(__file__).resolve().parents[1] / "training" / "eval_utils.py"
        spec = importlib.util.spec_from_file_location(
            "_safety_eval_utils_standalone",
            eval_utils_path,
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        looks_like_refusal = module.looks_like_refusal
    return bool(looks_like_refusal(str(text)))


def _load_wildguardmix_rows(
    *,
    source_name: str,
    config_name: str,
    cache_dir: Optional[str],
) -> Iterable[Dict[str, Any]]:
    loaded = _load_dataset(source_name, config_name, cache_dir=cache_dir)
    return _iter_dataset_rows(loaded)


def _build_wildguardmix_pool(
    rows: Iterable[Dict[str, Any]],
    *,
    source_name: str,
    config_name: str,
    system_prompt: str,
    seed: int,
    for_eval: bool = False,
    drops: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    records: List[Dict[str, Any]] = []
    drop_counts = drops if drops is not None else {}
    for index, row in enumerate(rows):
        prompt = str(row.get("prompt", "") or row.get("instruction", "") or "").strip()
        if not prompt:
            drop_counts["missing_prompt"] = drop_counts.get("missing_prompt", 0) + 1
            continue
        raw_label = _normalize_raw_label(row.get("prompt_harm_label"))
        label = _classify_wildguardmix_label(raw_label)
        if label is None:
            drop_counts["unknown_prompt_harm_label"] = (
                drop_counts.get("unknown_prompt_harm_label", 0) + 1
            )
            continue
        response = str(row.get("response", "") or row.get("completion", "") or "").strip()
        response_harm = _normalize_raw_label(row.get("response_harm_label"))
        response_refusal = _normalize_raw_label(row.get("response_refusal_label"))
        metadata = {
            "prompt_harm_label": raw_label,
            "response_harm_label": response_harm,
            "response_refusal_label": response_refusal,
            "source_config": config_name,
        }
        if not response and not for_eval:
            drop_counts["prompt_only"] = drop_counts.get("prompt_only", 0) + 1
            continue
        if label == "harmless":
            if not for_eval and not _response_harm_is_safe(response_harm):
                drop_counts["harmless_unsafe_response"] = (
                    drop_counts.get("harmless_unsafe_response", 0) + 1
                )
                continue
            if not for_eval:
                if _response_is_refusal(response_refusal):
                    drop_counts["harmless_refusal_response"] = (
                        drop_counts.get("harmless_refusal_response", 0) + 1
                    )
                    continue
                if not _response_is_explicit_compliance(response_refusal):
                    if _response_text_looks_like_refusal(response):
                        drop_counts["harmless_refusal_response_heuristic"] = (
                            drop_counts.get("harmless_refusal_response_heuristic", 0) + 1
                        )
                        continue
                    metadata["response_refusal_label_imputed"] = "non_refusal_by_text"
            if not response and not for_eval:
                drop_counts["harmless_missing_response"] = (
                    drop_counts.get("harmless_missing_response", 0) + 1
                )
                continue
            target = response or pick_refusal_template(prompt, seed=seed)
            metadata["target_source"] = "upstream_response"
        else:
            if response and _response_harm_is_safe(response_harm):
                target = response
                metadata["target_source"] = "upstream_safe_response"
            else:
                target = pick_refusal_template(prompt, seed=seed)
                metadata["target_source"] = "template_pool"
        if prompt in seen:
            drop_counts["duplicate_prompt"] = drop_counts.get("duplicate_prompt", 0) + 1
            continue
        seen.add(prompt)
        records.append(
            _build_binary_record(
                record_id=str(row.get("id") or f"wildguardmix_{config_name}_{index:08d}"),
                prompt=prompt,
                target=target,
                label=label,
                raw_label=raw_label,
                source=f"{source_name}:{config_name}",
                dataset="wildguardmix",
                system_prompt=system_prompt,
                metadata=metadata,
            )
        )
    return records


def build_wildguardmix_records(
    *,
    output_path: str | Path,
    source_name: str = "allenai/wildguardmix",
    split: str = "wildguardtrain",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    train_subset_mode: bool = True,
    max_train_samples: int = 20000,
    max_train_samples_per_label: int = 10000,
    eval_subset_mode: bool = False,
    max_eval_samples: int = 0,
    max_eval_samples_per_label: int = 0,
    eval_output_path: str | Path | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    drops: Dict[str, int] = {}
    train_pool = _build_wildguardmix_pool(
        _load_wildguardmix_rows(
            source_name=source_name,
            config_name=split or "wildguardtrain",
            cache_dir=cache_dir,
        ),
        source_name=source_name,
        config_name=split or "wildguardtrain",
        system_prompt=system_prompt,
        seed=seed,
        drops=drops,
    )
    records = _sample_balanced_by_label(
        train_pool,
        subset_mode=train_subset_mode,
        max_samples=max_train_samples,
        max_samples_per_label=max_train_samples_per_label,
        seed=seed,
    )
    _validate_binary_training_records(records)
    write_jsonl(output_path, records)

    eval_records: List[Dict[str, Any]] = []
    test_source = ""
    test_fallback_reason = ""
    if eval_output_path:
        try:
            eval_pool = _build_wildguardmix_pool(
                _load_wildguardmix_rows(
                    source_name=source_name,
                    config_name="wildguardtest",
                    cache_dir=cache_dir,
                ),
                source_name=source_name,
                config_name="wildguardtest",
                system_prompt=system_prompt,
                seed=seed,
                for_eval=True,
                drops=drops,
            )
            if len({r["label"] for r in eval_pool}) < 2:
                raise RuntimeError("WildGuardTest does not contain both labels")
            test_source = f"{source_name}:wildguardtest"
        except Exception as exc:
            test_fallback_reason = _format_fallback_reason(exc)
            selected_ids = {str(record.get("id", "")) for record in records}
            eval_pool = [record for record in train_pool if str(record.get("id", "")) not in selected_ids]
            test_source = f"{source_name}:{split or 'wildguardtrain'} holdout"
        eval_selected = _sample_balanced_by_label(
            eval_pool,
            subset_mode=eval_subset_mode,
            max_samples=max_eval_samples,
            max_samples_per_label=max_eval_samples_per_label,
            seed=seed + 101,
        )
        eval_records = [_to_eval_record(record, id_prefix="wildguardmix_test") for record in eval_selected]
        write_jsonl(eval_output_path, eval_records)

    _write_materialization_summary(
        output_path=output_path,
        dataset_name="wildguardmix",
        train_records=records,
        train_pool_count=len(train_pool),
        eval_records=eval_records,
        eval_output_path=eval_output_path,
        train_subset_mode=bool(train_subset_mode) and int(max_train_samples) > 0,
        eval_subset_mode=bool(eval_subset_mode) and int(max_eval_samples) > 0,
        seed=seed,
        test_source=test_source,
        test_fallback_reason=test_fallback_reason,
        drops=drops,
    )
    return records


HH_TURN_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):", re.IGNORECASE)


def _parse_hh_messages(
    text: str,
    *,
    system_prompt: str,
) -> Tuple[List[Dict[str, str]], str]:
    raw = str(text or "")
    matches = list(HH_TURN_RE.finditer(raw))
    if not matches:
        return [], ""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for idx, match in enumerate(matches):
        role_name = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        if not content:
            continue
        role = "user" if role_name == "human" else "assistant"
        messages.append({"role": role, "content": content})
    if len(messages) <= 1:
        return [], ""
    prompt_messages, target_response = _split_prompt_messages_and_target(messages)
    return prompt_messages, target_response


def _load_hh_rlhf_rows(
    *,
    source_name: str,
    subset_name: str,
    split: str,
    cache_dir: Optional[str],
) -> Iterable[Dict[str, Any]]:
    errors: List[Exception] = []
    attempts = [
        lambda: _load_dataset(source_name, subset_name, split=split, cache_dir=cache_dir),
        lambda: _load_dataset(source_name, data_dir=subset_name, split=split, cache_dir=cache_dir),
        lambda: _load_dataset(source_name, subset_name, cache_dir=cache_dir),
        lambda: _load_dataset(source_name, data_dir=subset_name, cache_dir=cache_dir),
    ]
    for attempt in attempts:
        try:
            return _iter_dataset_rows(attempt())
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(
        f"Failed to load {source_name}:{subset_name}:{split}. "
        f"Last error: {errors[-1]!r}"
    )


def _build_hh_rlhf_pool(
    rows: Iterable[Dict[str, Any]],
    *,
    source_name: str,
    subset_name: str,
    split: str,
    label: str,
    system_prompt: str,
    drops: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    drop_counts = drops if drops is not None else {}
    for index, row in enumerate(rows):
        chosen = str(row.get("chosen", "") or "").strip()
        if not chosen:
            drop_counts["missing_chosen"] = drop_counts.get("missing_chosen", 0) + 1
            continue
        messages, target_response = _parse_hh_messages(chosen, system_prompt=system_prompt)
        if not messages or not target_response:
            drop_counts["parse_failure"] = drop_counts.get("parse_failure", 0) + 1
            continue
        records.append(
            {
                "id": str(row.get("id") or f"hh_rlhf_{subset_name}_{split}_{index:08d}"),
                "messages": messages,
                "target_response": target_response,
                "label": label,
                "raw_label": subset_name,
                "source": f"{source_name}:{subset_name}:{split}",
                "dataset": "hh_rlhf",
                "metadata": {
                    "subset": subset_name,
                    "split": split,
                    "target_source": "chosen",
                    "rejected_discarded": "rejected" in row,
                },
            }
        )
    return records


def build_hh_rlhf_records(
    *,
    output_path: str | Path,
    source_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    train_subset_mode: bool = True,
    max_train_samples: int = 20000,
    max_train_samples_per_label: int = 10000,
    eval_subset_mode: bool = False,
    max_eval_samples: int = 0,
    max_eval_samples_per_label: int = 0,
    eval_output_path: str | Path | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    drops: Dict[str, int] = {}
    subset_specs = (("harmless-base", "harmful"), ("helpful-base", "harmless"))
    train_pool: List[Dict[str, Any]] = []
    for subset_name, label in subset_specs:
        train_pool.extend(
            _build_hh_rlhf_pool(
                _load_hh_rlhf_rows(
                    source_name=source_name,
                    subset_name=subset_name,
                    split=split or "train",
                    cache_dir=cache_dir,
                ),
                source_name=source_name,
                subset_name=subset_name,
                split=split or "train",
                label=label,
                system_prompt=system_prompt,
                drops=drops,
            )
        )
    records = _sample_balanced_by_label(
        train_pool,
        subset_mode=train_subset_mode,
        max_samples=max_train_samples,
        max_samples_per_label=max_train_samples_per_label,
        seed=seed,
    )
    _validate_binary_training_records(records)
    write_jsonl(output_path, records)

    eval_records: List[Dict[str, Any]] = []
    test_source = ""
    test_fallback_reason = ""
    if eval_output_path:
        try:
            eval_pool: List[Dict[str, Any]] = []
            for subset_name, label in subset_specs:
                eval_pool.extend(
                    _build_hh_rlhf_pool(
                        _load_hh_rlhf_rows(
                            source_name=source_name,
                            subset_name=subset_name,
                            split="test",
                            cache_dir=cache_dir,
                        ),
                        source_name=source_name,
                        subset_name=subset_name,
                        split="test",
                        label=label,
                        system_prompt=system_prompt,
                        drops=drops,
                    )
                )
            if len({r["label"] for r in eval_pool}) < 2:
                raise RuntimeError("HH-RLHF official test does not contain both labels")
            test_source = f"{source_name}:harmless-base/helpful-base:test"
        except Exception as exc:
            test_fallback_reason = _format_fallback_reason(exc)
            selected_ids = {str(record.get("id", "")) for record in records}
            eval_pool = [record for record in train_pool if str(record.get("id", "")) not in selected_ids]
            test_source = f"{source_name}:harmless-base/helpful-base:{split or 'train'} holdout"
        eval_selected = _sample_balanced_by_label(
            eval_pool,
            subset_mode=eval_subset_mode,
            max_samples=max_eval_samples,
            max_samples_per_label=max_eval_samples_per_label,
            seed=seed + 101,
        )
        eval_records = [_to_eval_record(record, id_prefix="hh_rlhf_test") for record in eval_selected]
        write_jsonl(eval_output_path, eval_records)

    _write_materialization_summary(
        output_path=output_path,
        dataset_name="hh_rlhf",
        train_records=records,
        train_pool_count=len(train_pool),
        eval_records=eval_records,
        eval_output_path=eval_output_path,
        train_subset_mode=bool(train_subset_mode) and int(max_train_samples) > 0,
        eval_subset_mode=bool(eval_subset_mode) and int(max_eval_samples) > 0,
        seed=seed,
        test_source=test_source,
        test_fallback_reason=test_fallback_reason,
        drops=drops,
    )
    return records


# ---------------------------------------------------------------------------
# Safety-Tuned LLaMAs
# ---------------------------------------------------------------------------


def _resolve_safety_tuned_llamas_file(
    repo_or_data_path: str | Path,
    file_name: str,
) -> Path:
    base = Path(repo_or_data_path).expanduser()
    candidates: List[Path] = []
    if base.is_file():
        candidates.append(base)
    else:
        # Upstream layout has fluctuated across commits: files live at the
        # repo root in old snapshots, under ``data/`` in some forks, and
        # under ``data/training/`` in the current vinid HEAD. Probe all
        # known locations before giving up.
        candidates.extend(
            [
                base / file_name,
                base / "data" / file_name,
                base / "data" / "training" / file_name,
                base / "training" / file_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate Safety-Tuned LLaMAs data file.\n"
        f"  repo_or_data_path: {base}\n"
        f"  file_name: {file_name}\n"
        f"Searched:\n  - {searched}\n"
        f"Clone the upstream repo from {SAFETY_TUNED_LLAMAS_REPO_URL} and "
        "either point `repo_or_data_path` at the repo root or directly at "
        f"the {file_name} file."
    )


def _parse_alpaca_records(
    json_path: Path,
    *,
    system_prompt: str,
    id_prefix: str,
    label: str,
    dataset: str,
) -> List[Dict[str, Any]]:
    """Parse an Alpaca-format JSON file into the project's record schema."""

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list in {json_path}, got {type(raw).__name__}."
        )
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(
                f"Row {index} of {json_path} is not a JSON object: {row!r}"
            )
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            continue
        user_text = instruction if not user_input else f"{instruction}\n\n{user_input}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        records.append(
            {
                "id": f"{id_prefix}_{index:06d}",
                "messages": messages,
                "target_response": output,
                "label": label,
                "source": str(json_path),
                "dataset": dataset,
            }
        )
    return records


def build_safety_tuned_llamas_records(
    *,
    output_path: str | Path,
    repo_or_data_path: str | Path,
    file_name: str = DEFAULT_SAFETY_TUNED_LLAMAS_FILE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_harmless_contrast: bool = False,
    harmless_file_name: str = "alpaca_small.json",
    harmless_max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Materialize the 2k Safety-Tuned LLaMAs Alpaca-format records.

    The Alpaca schema is ``{"instruction", "input", "output"}``. The user
    turn becomes ``instruction`` (with ``input`` appended after a blank
    line when present) and the assistant turn becomes ``output``.

    When ``include_harmless_contrast=True``, the upstream repo's
    ``alpaca_small.json`` (general Alpaca instructions) is also loaded as
    ``label="harmless"``. By default it is deterministically downsampled
    to the harmful count so the 20k harmless Alpaca pool does not swamp
    the ~2.5k safety-only records. Pass ``harmless_max_samples`` to set a
    different cap.

    Round 2: ``label`` is always one of ``{"harmful", "harmless"}`` so
    ``20_split_safety_for_semalign.py`` recognizes both poles in-domain
    without falling back to PAN harmless injection. Provenance lives on
    the ``dataset`` field instead.
    """

    json_path = _resolve_safety_tuned_llamas_file(repo_or_data_path, file_name)
    records = _parse_alpaca_records(
        json_path,
        system_prompt=system_prompt,
        id_prefix="safety_tuned_llamas",
        label="harmful",
        dataset="safety_tuned_llamas",
    )

    if not records:
        raise RuntimeError(
            f"No usable records were parsed from {json_path}; "
            "check that the file follows the Alpaca schema."
        )

    if include_harmless_contrast:
        harmless_path = _resolve_safety_tuned_llamas_file(
            repo_or_data_path, harmless_file_name
        )
        harmless_records = _parse_alpaca_records(
            harmless_path,
            system_prompt=system_prompt,
            id_prefix="safety_tuned_llamas_harmless",
            label="harmless",
            dataset="safety_tuned_llamas_harmless",
        )
        if not harmless_records:
            raise RuntimeError(
                f"include_harmless_contrast=True but {harmless_path} parsed "
                "into zero records; check the alpaca_small.json schema."
            )
        harmless_cap = (
            len(records)
            if harmless_max_samples is None
            else max(int(harmless_max_samples), 0)
        )
        rng = random.Random(int(seed))
        rng.shuffle(harmless_records)
        harmless_records = harmless_records[:harmless_cap]
        records.extend(harmless_records)

    write_jsonl(output_path, records)
    return records


# ---------------------------------------------------------------------------
# BeaverTails
# ---------------------------------------------------------------------------


SUPPORTED_BEAVERTAILS_TRAIN_SPLITS = frozenset({"30k_train", "330k_train"})


def _bt_label_from_category(category: Any) -> str:
    """Round 2: prompt-level harmful when any category flag is True."""

    if isinstance(category, dict) and category:
        return "harmful" if any(bool(value) for value in category.values()) else "harmless"
    # Missing / non-dict category -> conservative harmful (BT prompts without
    # categories almost never appear in the 30k_train split).
    return "harmful"


def build_beavertails_records(
    *,
    output_path: str | Path,
    split: str = "30k_train",
    refusal_template: str = DEFAULT_SAFETY_REFUSAL_TEMPLATE,
    source_name: str = "PKU-Alignment/BeaverTails",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    dedup_prompts: bool = True,
    label_strategy: str = "category_any",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Materialize a BeaverTails SFT split.

    BeaverTails 30k_train ships multiple ``is_safe``-tagged responses per
    prompt (different model generations). When ``dedup_prompts=True`` (default)
    we group rows by prompt and pick a single canonical response per group,
    preferring an upstream ``is_safe=True`` response as the refusal target.
    This matches the BeaverTails paper SFT protocol and avoids the synthetic
    8-template refusal pool except as last-resort fallback.

    Per-prompt selection rule (``dedup_prompts=True``):

    1. Determine prompt-level danger label:
       * ``label_strategy="is_safe"``: prompt is ``harmful`` if any row in the
         group has ``is_safe=False`` (i.e. at least one model failed to refuse),
         else ``harmless``.
       * ``label_strategy="category_any"``: prompt is ``harmful`` if any flag
         in the BT ``category`` dict is True, else ``harmless``. Prompts whose
         category label disagrees with the per-group ``is_safe`` distribution
         (e.g. benign category but all upstream responses unsafe) are dropped.
    2. Pick assistant target:
       * harmful prompt → prefer first ``is_safe=True`` response in the group
         (natural BT refusal text). Fallback to ``pick_refusal_template``
         (8-string pool) only when no safe response exists for that prompt.
       * harmless prompt → use first ``is_safe=True`` response in the group.

    When ``dedup_prompts=False`` we keep the legacy per-row emission for
    backward compatibility with older runs.
    """

    if split not in SUPPORTED_BEAVERTAILS_TRAIN_SPLITS:
        raise ValueError(
            f"Unsupported BeaverTails split: {split!r}. "
            f"Expected one of {sorted(SUPPORTED_BEAVERTAILS_TRAIN_SPLITS)}."
        )
    if not refusal_template.strip():
        raise ValueError("`refusal_template` must be a non-empty string")
    if label_strategy not in {"category_any", "is_safe"}:
        raise ValueError(
            f"Unknown label_strategy {label_strategy!r}; expected 'category_any' "
            "or 'is_safe'."
        )

    dataset = _load_dataset(source_name, split=split, cache_dir=cache_dir)
    records: List[Dict[str, Any]] = []

    if dedup_prompts:
        # Group all rows by prompt so we can pick the best response per prompt.
        prompt_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(dataset):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            prompt_groups[prompt].append((index, dict(row)))

        for prompt, group in prompt_groups.items():
            safe_rows = [(i, r) for i, r in group if bool(r.get("is_safe", False))]
            unsafe_rows = [(i, r) for i, r in group if not bool(r.get("is_safe", False))]

            if label_strategy == "category_any":
                category = group[0][1].get("category")
                label = _bt_label_from_category(category)
                # Drop when prompt-level harmfulness disagrees with the actual
                # distribution of upstream is_safe flags for this prompt:
                #   * harmful category but no unsafe response → not actually risky
                #   * harmless category but no safe response → suspicious labelling
                if label == "harmful" and not unsafe_rows:
                    continue
                if label == "harmless" and not safe_rows:
                    continue
            else:  # "is_safe"
                label = "harmful" if unsafe_rows else "harmless"

            if safe_rows:
                chosen_index, chosen_row = safe_rows[0]
                assistant_text = str(chosen_row.get("response", "")).strip()
                chosen_is_safe = True
                refusal_source = "upstream_safe"
            else:
                chosen_index, chosen_row = unsafe_rows[0]
                assistant_text = pick_refusal_template(prompt, seed=seed)
                chosen_is_safe = False
                refusal_source = "template_pool"

            if not assistant_text:
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            records.append(
                {
                    "id": f"beavertails_{split}_{chosen_index:08d}",
                    "messages": messages,
                    "target_response": assistant_text,
                    "original_response": str(chosen_row.get("response", "")).strip(),
                    "is_safe": chosen_is_safe,
                    "category": chosen_row.get("category"),
                    "label": label,
                    "label_strategy": label_strategy,
                    "source": source_name,
                    "split": split,
                    "dataset": "beavertails",
                    "refusal_source": refusal_source,
                    "prompt_group_size": len(group),
                    "prompt_group_safe_count": len(safe_rows),
                    "prompt_group_unsafe_count": len(unsafe_rows),
                }
            )
    else:
        # Legacy per-row path. Kept for backward compatibility; emits one
        # record per dataset row using the response that row carries.
        for index, row in enumerate(dataset):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            is_safe = bool(row.get("is_safe", False))
            original_response = str(row.get("response", "")).strip()
            category = row.get("category")
            if label_strategy == "category_any":
                label = _bt_label_from_category(category)
                response_label = "harmless" if is_safe else "harmful"
                if label != response_label:
                    continue
            else:
                label = "harmless" if is_safe else "harmful"
            if label == "harmless":
                assistant_text = original_response
            elif is_safe and original_response:
                assistant_text = original_response
            else:
                assistant_text = pick_refusal_template(prompt, seed=seed)
            if not assistant_text:
                continue
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            records.append(
                {
                    "id": f"beavertails_{split}_{index:08d}",
                    "messages": messages,
                    "target_response": assistant_text,
                    "original_response": original_response,
                    "is_safe": is_safe,
                    "category": category,
                    "label": label,
                    "label_strategy": label_strategy,
                    "source": source_name,
                    "split": split,
                    "dataset": "beavertails",
                }
            )

    if not records:
        raise RuntimeError(
            f"BeaverTails {split} produced zero usable records from {source_name}."
        )

    write_jsonl(output_path, records)
    return records


def _build_beavertails_category_pool(
    rows: Iterable[Dict[str, Any]],
    *,
    source_name: str,
    split: str,
    system_prompt: str,
    seed: int,
    dedup_prompts: bool,
    drops: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    drop_counts = drops if drops is not None else {}
    records: List[Dict[str, Any]] = []

    if dedup_prompts:
        prompt_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(rows):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                drop_counts["missing_prompt"] = drop_counts.get("missing_prompt", 0) + 1
                continue
            prompt_groups[prompt].append((index, dict(row)))

        for prompt, group in prompt_groups.items():
            category = group[0][1].get("category")
            label = _bt_label_from_category(category)
            safe_rows = [
                (i, r)
                for i, r in group
                if bool(r.get("is_safe", False)) and str(r.get("response", "")).strip()
            ]
            unsafe_rows = [(i, r) for i, r in group if not bool(r.get("is_safe", False))]
            if safe_rows:
                chosen_index, chosen_row = max(
                    safe_rows,
                    key=lambda item: len(str(item[1].get("response", "")).strip()),
                )
                assistant_text = str(chosen_row.get("response", "")).strip()
                target_source = "longest_upstream_safe"
                chosen_is_safe = True
            else:
                chosen_index, chosen_row = group[0]
                assistant_text = pick_refusal_template(prompt, seed=seed)
                target_source = "template_pool_all_unsafe"
                chosen_is_safe = False
                if not unsafe_rows:
                    drop_counts["all_missing_is_safe"] = drop_counts.get("all_missing_is_safe", 0) + 1
            records.append(
                {
                    "id": f"beavertails_category_{split}_{chosen_index:08d}",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "target_response": assistant_text,
                    "original_response": str(chosen_row.get("response", "")).strip(),
                    "is_safe": chosen_is_safe,
                    "category": category,
                    "label": label,
                    "raw_label": "category_any",
                    "label_strategy": "category_any",
                    "source": source_name,
                    "split": split,
                    "dataset": "beavertails_category",
                    "metadata": {
                        "target_source": target_source,
                        "prompt_group_size": len(group),
                        "prompt_group_safe_count": len(safe_rows),
                        "prompt_group_unsafe_count": len(unsafe_rows),
                    },
                }
            )
        return records

    for index, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            drop_counts["missing_prompt"] = drop_counts.get("missing_prompt", 0) + 1
            continue
        response = str(row.get("response", "")).strip()
        is_safe = bool(row.get("is_safe", False))
        category = row.get("category")
        label = _bt_label_from_category(category)
        target = response if is_safe and response else pick_refusal_template(prompt, seed=seed)
        records.append(
            {
                "id": f"beavertails_category_{split}_{index:08d}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "target_response": target,
                "original_response": response,
                "is_safe": is_safe,
                "category": category,
                "label": label,
                "raw_label": "category_any",
                "label_strategy": "category_any",
                "source": source_name,
                "split": split,
                "dataset": "beavertails_category",
                "metadata": {
                    "target_source": "upstream_safe" if is_safe and response else "template_pool_all_unsafe"
                },
            }
        )
    return records


def build_beavertails_category_records(
    *,
    output_path: str | Path,
    split: str = "30k_train",
    source_name: str = "PKU-Alignment/BeaverTails",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    dedup_prompts: bool = True,
    train_subset_mode: bool = False,
    max_train_samples: int = 0,
    max_train_samples_per_label: int = 0,
    eval_subset_mode: bool = False,
    max_eval_samples: int = 0,
    max_eval_samples_per_label: int = 0,
    eval_output_path: str | Path | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if split not in SUPPORTED_BEAVERTAILS_TRAIN_SPLITS:
        raise ValueError(
            f"Unsupported BeaverTails split: {split!r}. "
            f"Expected one of {sorted(SUPPORTED_BEAVERTAILS_TRAIN_SPLITS)}."
        )
    drops: Dict[str, int] = {}
    dataset = _load_dataset(source_name, split=split, cache_dir=cache_dir)
    train_pool = _build_beavertails_category_pool(
        dataset,
        source_name=source_name,
        split=split,
        system_prompt=system_prompt,
        seed=seed,
        dedup_prompts=dedup_prompts,
        drops=drops,
    )
    records = _sample_balanced_by_label(
        train_pool,
        subset_mode=train_subset_mode,
        max_samples=max_train_samples,
        max_samples_per_label=max_train_samples_per_label,
        seed=seed,
    )
    _validate_binary_training_records(records)
    write_jsonl(output_path, records)

    eval_records: List[Dict[str, Any]] = []
    test_source = ""
    test_fallback_reason = ""
    if eval_output_path:
        eval_split = "30k_test" if split.startswith("30k") else "330k_test"
        try:
            eval_dataset = _load_dataset(source_name, split=eval_split, cache_dir=cache_dir)
            eval_pool = _build_beavertails_category_pool(
                eval_dataset,
                source_name=source_name,
                split=eval_split,
                system_prompt=system_prompt,
                seed=seed,
                dedup_prompts=dedup_prompts,
                drops=drops,
            )
            if len({r["label"] for r in eval_pool}) < 2:
                raise RuntimeError("BeaverTails category test does not contain both labels")
            test_source = f"{source_name}:{eval_split}"
        except Exception as exc:
            test_fallback_reason = _format_fallback_reason(exc)
            selected_ids = {str(record.get("id", "")) for record in records}
            eval_pool = [record for record in train_pool if str(record.get("id", "")) not in selected_ids]
            test_source = f"{source_name}:{split} holdout"
        eval_selected = _sample_balanced_by_label(
            eval_pool,
            subset_mode=eval_subset_mode,
            max_samples=max_eval_samples,
            max_samples_per_label=max_eval_samples_per_label,
            seed=seed + 101,
        )
        eval_records = [_to_eval_record(record, id_prefix="beavertails_category_test") for record in eval_selected]
        write_jsonl(eval_output_path, eval_records)

    _write_materialization_summary(
        output_path=output_path,
        dataset_name="beavertails_category",
        train_records=records,
        train_pool_count=len(train_pool),
        eval_records=eval_records,
        eval_output_path=eval_output_path,
        train_subset_mode=bool(train_subset_mode) and int(max_train_samples) > 0,
        eval_subset_mode=bool(eval_subset_mode) and int(max_eval_samples) > 0,
        seed=seed,
        test_source=test_source,
        test_fallback_reason=test_fallback_reason,
        drops=drops,
    )
    return records


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


SafetyDatasetBuilder = Callable[[SafetyDatasetSpec], List[Dict[str, Any]]]


def _build_tulu3_safety(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_tulu3_safety_records(
        output_path=spec.output_path,
        sources=tuple(spec.sources or DEFAULT_TULU3_SAFETY_SOURCES),
        source_name=spec.source_name or "allenai/tulu-3-sft-mixture",
        split=spec.split or "train",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
    )


def _build_safety_tuned_llamas(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    if not spec.repo_or_data_path:
        raise ValueError(
            "safety_tuned_llamas requires `repo_or_data_path` (path to the "
            f"upstream {SAFETY_TUNED_LLAMAS_REPO_URL} clone or directly to "
            f"{DEFAULT_SAFETY_TUNED_LLAMAS_FILE})."
        )
    return build_safety_tuned_llamas_records(
        output_path=spec.output_path,
        repo_or_data_path=spec.repo_or_data_path,
        file_name=spec.file_name or DEFAULT_SAFETY_TUNED_LLAMAS_FILE,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        include_harmless_contrast=bool(spec.include_harmless_contrast),
        harmless_file_name=spec.harmless_file_name or "alpaca_small.json",
        harmless_max_samples=spec.harmless_max_samples,
        seed=int(spec.seed),
    )


def _build_beavertails(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_beavertails_records(
        output_path=spec.output_path,
        split=spec.split or "30k_train",
        refusal_template=spec.refusal_template or DEFAULT_SAFETY_REFUSAL_TEMPLATE,
        source_name=spec.source_name or "PKU-Alignment/BeaverTails",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        dedup_prompts=bool(spec.dedup_prompts),
        label_strategy=spec.label_strategy or "category_any",
        seed=int(spec.seed),
    )


def _build_wildjailbreak(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_wildjailbreak_records(
        output_path=spec.output_path,
        source_name=spec.source_name or "allenai/wildjailbreak",
        split=spec.split or "train",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        train_subset_mode=bool(spec.train_subset_mode),
        max_train_samples=int(spec.max_train_samples),
        max_train_samples_per_label=int(spec.max_train_samples_per_label),
        eval_subset_mode=bool(spec.eval_subset_mode),
        max_eval_samples=int(spec.max_eval_samples),
        max_eval_samples_per_label=int(spec.max_eval_samples_per_label),
        eval_output_path=spec.eval_output_path,
        seed=int(spec.seed),
    )


def _build_wildguardmix(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_wildguardmix_records(
        output_path=spec.output_path,
        source_name=spec.source_name or "allenai/wildguardmix",
        split=spec.split or "wildguardtrain",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        train_subset_mode=bool(spec.train_subset_mode),
        max_train_samples=int(spec.max_train_samples),
        max_train_samples_per_label=int(spec.max_train_samples_per_label),
        eval_subset_mode=bool(spec.eval_subset_mode),
        max_eval_samples=int(spec.max_eval_samples),
        max_eval_samples_per_label=int(spec.max_eval_samples_per_label),
        eval_output_path=spec.eval_output_path,
        seed=int(spec.seed),
    )


def _build_hh_rlhf(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_hh_rlhf_records(
        output_path=spec.output_path,
        source_name=spec.source_name or "Anthropic/hh-rlhf",
        split=spec.split or "train",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        train_subset_mode=bool(spec.train_subset_mode),
        max_train_samples=int(spec.max_train_samples),
        max_train_samples_per_label=int(spec.max_train_samples_per_label),
        eval_subset_mode=bool(spec.eval_subset_mode),
        max_eval_samples=int(spec.max_eval_samples),
        max_eval_samples_per_label=int(spec.max_eval_samples_per_label),
        eval_output_path=spec.eval_output_path,
        seed=int(spec.seed),
    )


def _build_beavertails_category(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_beavertails_category_records(
        output_path=spec.output_path,
        split=spec.split or "30k_train",
        source_name=spec.source_name or "PKU-Alignment/BeaverTails",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        dedup_prompts=bool(spec.dedup_prompts),
        train_subset_mode=bool(spec.train_subset_mode),
        max_train_samples=int(spec.max_train_samples),
        max_train_samples_per_label=int(spec.max_train_samples_per_label),
        eval_subset_mode=bool(spec.eval_subset_mode),
        max_eval_samples=int(spec.max_eval_samples),
        max_eval_samples_per_label=int(spec.max_eval_samples_per_label),
        eval_output_path=spec.eval_output_path,
        seed=int(spec.seed),
    )


def _build_tulu3_safety_v2(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_tulu3_safety_v2_records(
        output_path=spec.output_path,
        source_name=spec.source_name or "allenai/tulu-3-sft-mixture",
        split=spec.split or "train",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
        helpful_sources=tuple(spec.helpful_sources or DEFAULT_TULU3_HELPFUL_SOURCES),
        helpful_max_samples=spec.helpful_max_samples,
        seed=int(spec.seed),
    )


SAFETY_TRAIN_DATASETS: Dict[str, SafetyDatasetBuilder] = {
    "tulu3_safety": _build_tulu3_safety,
    "tulu3_safety_v2": _build_tulu3_safety_v2,
    "safety_tuned_llamas": _build_safety_tuned_llamas,
    "beavertails": _build_beavertails,
    "wildjailbreak": _build_wildjailbreak,
    "wildguardmix": _build_wildguardmix,
    "hh_rlhf": _build_hh_rlhf,
    "beavertails_category": _build_beavertails_category,
}


def materialize_safety_train_dataset(
    spec: SafetyDatasetSpec,
) -> Path:
    """Build (or reuse) the JSONL training file for a safety baseline.

    Returns the absolute output path. When ``spec.force_rebuild`` is False
    and the requested train/eval JSONLs already match the requested subset
    mode, the existing files are left untouched.
    """

    if spec.name not in SAFETY_TRAIN_DATASETS:
        supported = ", ".join(sorted(SAFETY_TRAIN_DATASETS))
        raise ValueError(
            f"Unknown safety dataset {spec.name!r}. Expected one of: {supported}."
        )

    output_path = Path(spec.output_path)
    eval_path = Path(spec.eval_output_path) if spec.eval_output_path else None
    if (
        not spec.force_rebuild
        and output_path.exists()
        and _existing_eval_matches_request(
            eval_path=eval_path,
            eval_subset_mode=bool(spec.eval_subset_mode),
            max_eval_samples=int(spec.max_eval_samples),
        )
    ):
        return output_path.resolve()

    ensure_dir(output_path.parent)
    builder = SAFETY_TRAIN_DATASETS[spec.name]
    builder(spec)
    return output_path.resolve()


__all__ = [
    "DEFAULT_SAFETY_REFUSAL_TEMPLATE",
    "DEFAULT_SAFETY_TUNED_LLAMAS_FILE",
    "DEFAULT_TULU3_HELPFUL_SOURCES",
    "DEFAULT_TULU3_SAFETY_SOURCES",
    "SAFETY_TUNED_LLAMAS_REPO_URL",
    "SAFETY_TRAIN_DATASETS",
    "SafetyDatasetSpec",
    "SUPPORTED_BEAVERTAILS_TRAIN_SPLITS",
    "WILDGUARDMIX_HARMFUL_LABELS",
    "WILDGUARDMIX_HARMLESS_LABELS",
    "WILDJAILBREAK_HARMFUL_DATA_TYPES",
    "WILDJAILBREAK_HARMLESS_DATA_TYPES",
    "_bt_label_from_category",
    "_parse_hh_messages",
    "build_beavertails_records",
    "build_beavertails_category_records",
    "build_hh_rlhf_records",
    "build_safety_tuned_llamas_records",
    "build_tulu3_safety_records",
    "build_tulu3_safety_v2_records",
    "build_wildguardmix_records",
    "build_wildjailbreak_records",
    "materialize_safety_train_dataset",
]
