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
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data.template_qwen import DEFAULT_SYSTEM_PROMPT
from src.utils.io import ensure_dir, write_jsonl


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
            return normalized[:index], target
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

    Unsafe rows (``is_safe == False``) get their assistant turn replaced
    with the project-wide refusal template; safe rows keep the original
    response. Both branches share the same ``messages`` schema.

    BeaverTails 30k_train ships multiple ``is_safe``-tagged responses per
    prompt. ``dedup_prompts=True`` (default) keeps only the first record
    per prompt to flatten that response-level inflation.

    ``label_strategy`` selects the binary label semantics:

    * ``"category_any"`` (default, Round 2): prompt-level — ``harmful``
      when any flag in the BT ``category`` dict is True, ``harmless``
      otherwise. To prevent target/label mismatch (e.g. a benign-category
      prompt paired with an ``is_safe=False`` unsafe response would teach
      the model to refuse benign prompts), this strategy **drops** rows
      where the prompt-level label disagrees with response-level
      ``is_safe``. The kept rows have aligned signal:
      ``(harmful, is_safe=False)`` → refusal target, and
      ``(harmless, is_safe=True)`` → keep the original safe response.
    * ``"is_safe"`` (legacy, Round 1): response-level — flips the upstream
      ``is_safe`` boolean (``True → harmless``, ``False → harmful``).
      Retained for reproducibility of older runs but biased toward refusal
      style rather than prompt content.
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
    seen_prompts: set[str] = set()
    for index, row in enumerate(dataset):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        if dedup_prompts:
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
        is_safe = bool(row.get("is_safe", False))
        original_response = str(row.get("response", "")).strip()
        category = row.get("category")
        if label_strategy == "category_any":
            label = _bt_label_from_category(category)
            response_label = "harmless" if is_safe else "harmful"
            if label != response_label:
                # BT noise: drop prompts where category-level harmfulness
                # disagrees with response-level is_safe. Keeping these
                # would teach the model the wrong association.
                continue
        else:
            label = "harmless" if is_safe else "harmful"
        if label == "harmless":
            assistant_text = original_response
        elif is_safe and original_response:
            # Upstream BT response was tagged safe for this harmful prompt — use
            # it verbatim instead of synthesising a refusal. Under the default
            # ``label_strategy="category_any"`` the alignment-mismatch drop above
            # makes this branch unreachable (category-harmful rows always carry
            # is_safe=False), but the conditional keeps the more permissive
            # ``label_strategy="is_safe"`` path correct.
            assistant_text = original_response
        else:
            assistant_text = pick_refusal_template(prompt, seed=seed)
        if not assistant_text:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        record = {
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
        records.append(record)

    if not records:
        raise RuntimeError(
            f"BeaverTails {split} produced zero usable records from {source_name}."
        )

    write_jsonl(output_path, records)
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
}


def materialize_safety_train_dataset(
    spec: SafetyDatasetSpec,
) -> Path:
    """Build (or reuse) the JSONL training file for a safety baseline.

    Returns the absolute output path. When ``spec.force_rebuild`` is False
    and the JSONL already exists, the existing file is left untouched and
    the function is effectively a no-op.
    """

    if spec.name not in SAFETY_TRAIN_DATASETS:
        supported = ", ".join(sorted(SAFETY_TRAIN_DATASETS))
        raise ValueError(
            f"Unknown safety dataset {spec.name!r}. Expected one of: {supported}."
        )

    output_path = Path(spec.output_path)
    if not spec.force_rebuild and output_path.exists():
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
    "build_beavertails_records",
    "build_safety_tuned_llamas_records",
    "build_tulu3_safety_records",
    "build_tulu3_safety_v2_records",
    "materialize_safety_train_dataset",
]
