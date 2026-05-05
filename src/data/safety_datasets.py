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

import json
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

DEFAULT_TULU3_SAFETY_SOURCES: Tuple[str, ...] = (
    "ai2-adapt-dev/tulu_v3.9_wildguardmix",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak",
    "ai2-adapt-dev/coconot_converted",
)

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
        candidates.append(base / file_name)
        candidates.append(base / "data" / file_name)
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


def build_safety_tuned_llamas_records(
    *,
    output_path: str | Path,
    repo_or_data_path: str | Path,
    file_name: str = DEFAULT_SAFETY_TUNED_LLAMAS_FILE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Materialize the 2k Safety-Tuned LLaMAs Alpaca-format records.

    The Alpaca schema is ``{"instruction", "input", "output"}``. The user
    turn becomes ``instruction`` (with ``input`` appended after a blank
    line when present) and the assistant turn becomes ``output``.
    """

    json_path = _resolve_safety_tuned_llamas_file(repo_or_data_path, file_name)
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
                "id": f"safety_tuned_llamas_{index:06d}",
                "messages": messages,
                "target_response": output,
                "label": "safety_tuned_llamas",
                "source": str(json_path),
                "dataset": "safety_tuned_llamas",
            }
        )

    if not records:
        raise RuntimeError(
            f"No usable records were parsed from {json_path}; "
            "check that the file follows the Alpaca schema."
        )

    write_jsonl(output_path, records)
    return records


# ---------------------------------------------------------------------------
# BeaverTails
# ---------------------------------------------------------------------------


SUPPORTED_BEAVERTAILS_TRAIN_SPLITS = frozenset({"30k_train", "330k_train"})


def build_beavertails_records(
    *,
    output_path: str | Path,
    split: str = "30k_train",
    refusal_template: str = DEFAULT_SAFETY_REFUSAL_TEMPLATE,
    source_name: str = "PKU-Alignment/BeaverTails",
    cache_dir: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Materialize a BeaverTails SFT split.

    Unsafe rows (``is_safe == False``) get their assistant turn replaced
    with the project-wide refusal template; safe rows keep the original
    response. Both branches share the same ``messages`` schema.
    """

    if split not in SUPPORTED_BEAVERTAILS_TRAIN_SPLITS:
        raise ValueError(
            f"Unsupported BeaverTails split: {split!r}. "
            f"Expected one of {sorted(SUPPORTED_BEAVERTAILS_TRAIN_SPLITS)}."
        )
    if not refusal_template.strip():
        raise ValueError("`refusal_template` must be a non-empty string")

    dataset = _load_dataset(source_name, split=split, cache_dir=cache_dir)
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(dataset):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        is_safe = bool(row.get("is_safe", False))
        original_response = str(row.get("response", "")).strip()
        assistant_text = original_response if is_safe else refusal_template
        if not assistant_text:
            continue
        category = row.get("category")
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
            "label": "beavertails_safe" if is_safe else "beavertails_unsafe",
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
    )


def _build_beavertails(spec: SafetyDatasetSpec) -> List[Dict[str, Any]]:
    return build_beavertails_records(
        output_path=spec.output_path,
        split=spec.split or "30k_train",
        refusal_template=spec.refusal_template or DEFAULT_SAFETY_REFUSAL_TEMPLATE,
        source_name=spec.source_name or "PKU-Alignment/BeaverTails",
        cache_dir=spec.cache_dir,
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
    )


SAFETY_TRAIN_DATASETS: Dict[str, SafetyDatasetBuilder] = {
    "tulu3_safety": _build_tulu3_safety,
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
    "DEFAULT_TULU3_SAFETY_SOURCES",
    "SAFETY_TUNED_LLAMAS_REPO_URL",
    "SAFETY_TRAIN_DATASETS",
    "SafetyDatasetSpec",
    "SUPPORTED_BEAVERTAILS_TRAIN_SPLITS",
    "build_beavertails_records",
    "build_safety_tuned_llamas_records",
    "build_tulu3_safety_records",
    "materialize_safety_train_dataset",
]
