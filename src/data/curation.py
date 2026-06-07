"""Phase 1 curation pipeline — produce a clean harmful/harmless subset.

The SemAlign Phase 1 SVD safety subspace is sensitive to noisy first-gen
token hidden states from adversarial-styled prompts (WildJailbreak's
adversarial_harmful, WildGuardMix red-team-heavy benigns). Per the Pan
2025 / Arditi 2024 / LIMA / Tülu 3 line of work, the standard remedy is
to extract directions from a small (~2-5k) high-quality contrast subset,
while leaving the full SFT corpus to Phase F.

This module decouples the two: ``curate_phase1_subset`` consumes the
records that would otherwise have gone straight into
``alignment_set.jsonl``, applies a per-baseline pre-filter plus a
universal length + teacher-confidence + stratified-sample filter, and
returns a curated subset + drop statistics. The Phase F training set is
left untouched.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

CurationMode = str  # "off" | "minimal" | "strict"

VALID_MODES = ("off", "minimal", "strict")


DEFAULT_MODE: Dict[str, CurationMode] = {
    "pan": "off",
    "safety_tuned_llamas": "off",
    "coconot": "off",
    "hh_rlhf": "off",
    "beavertails_category": "strict",
    "beavertails": "strict",
    "tulu3_safety": "off",
    "wildjailbreak": "strict",
    "wildguardmix": "strict",
}


def _wjb_pre_filter(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep_types = {"vanilla_harmful", "vanilla_benign"}
    out: List[Dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        data_type = str(metadata.get("data_type") or "").strip().lower()
        if data_type in keep_types:
            out.append(record)
    return out


def _wgm_pre_filter(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep_prompt_labels = {"harmful", "unharmful"}
    keep_refusal_labels = {"refusal", "compliance"}
    out: List[Dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        prompt_label = str(metadata.get("prompt_harm_label") or "").strip().lower()
        refusal_label = str(metadata.get("response_refusal_label") or "").strip().lower()
        if prompt_label in keep_prompt_labels and refusal_label in keep_refusal_labels:
            out.append(record)
    return out


def _bt_pre_filter(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """BeaverTails Phase1 consensus filter (Q7), using per-prompt response counts.

    The BT builder labels a prompt harmful via ``category_any`` (any category
    flag True) and forces missing/empty category -> harmful. That contaminates
    the harmful pole with benign/borderline prompts and pushes the SVD axis
    toward "sensitive topic" rather than "harmful intent".

    Here we keep only consensus rows, using the per-prompt-group response
    safety counts the BT builder writes into ``metadata`` (these survive
    20_split's metadata passthrough; the top-level ``category`` does not):

    - ``harmful`` kept iff ``prompt_group_unsafe_count > 0`` (at least one
      response was actually unsafe -> consensus-harmful). This also discards
      the ``missing-category -> forced harmful`` rows that have no unsafe
      response.
    - ``harmless`` kept iff ``prompt_group_safe_count > 0`` (the prompt has a
      genuine safe response, i.e. it is compliable -> keep, including the
      over-refusal-bait benigns PAN wants in the harmless pole).
    - rows lacking the group counts (not from the BT category builder) are
      dropped from the SVD pool since intent cannot be verified.

    Phase1-only: this runs on the curated alignment subset; ``train_set.jsonl``
    (Phase F L_out) and the eval test set are untouched.
    """
    out: List[Dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        label = str(record.get("label") or "").strip().lower()
        safe_count = metadata.get("prompt_group_safe_count")
        unsafe_count = metadata.get("prompt_group_unsafe_count")
        if safe_count is None or unsafe_count is None:
            continue
        try:
            safe_count = int(safe_count)
            unsafe_count = int(unsafe_count)
        except (TypeError, ValueError):
            continue
        if label == "harmful" and unsafe_count > 0:
            out.append(record)
        elif label == "harmless" and safe_count > 0:
            out.append(record)
    return out


PRE_FILTERS: Dict[str, Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "wildjailbreak": _wjb_pre_filter,
    "wildguardmix": _wgm_pre_filter,
    "beavertails_category": _bt_pre_filter,
    "beavertails": _bt_pre_filter,
}


def resolve_curation_mode(
    baseline_name: str,
    requested_mode: Optional[str] = None,
) -> CurationMode:
    if requested_mode:
        mode = str(requested_mode).strip().lower()
        if mode == "auto":
            return DEFAULT_MODE.get(baseline_name, "off")
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unknown curation_mode {requested_mode!r}; expected one of "
                f"{VALID_MODES} or 'auto'."
            )
        return mode
    return DEFAULT_MODE.get(baseline_name, "off")


def _prompt_token_count(record: Mapping[str, Any], tokenizer: Any = None) -> int:
    """Approximate prompt length in tokens.

    When a tokenizer is provided (real curation run), uses
    ``tokenizer.encode``. Without one (unit tests / off-mode), falls back
    to whitespace split which preserves ordering for filter logic.
    """

    user_text = str(record.get("user_text") or "")
    if not user_text:
        for message in record.get("messages") or []:
            if str(message.get("role", "")).lower() == "user":
                user_text = str(message.get("content") or "")
                break
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(user_text, add_special_tokens=False))
        except Exception:
            pass
    return len(user_text.split())


def _length_filter(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for record in records:
        n = _prompt_token_count(record, tokenizer=tokenizer)
        if min_tokens <= n <= max_tokens:
            kept.append(record)
        else:
            dropped += 1
    return kept, dropped


def _confidence_filter(
    records: Sequence[Dict[str, Any]],
    *,
    confidence_min: float,
    confidence_key: str = "teacher_first_gen_top1_prob",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Filter on the teacher confidence metadata field.

    Returns: (kept, dropped_low_conf, missing_conf). Records missing the
    field are kept (treated as 1.0 fallback) so callers that opt out of
    the teacher forward pass still get a working pipeline; missing count
    is surfaced in the summary.
    """

    kept: List[Dict[str, Any]] = []
    dropped = 0
    missing = 0
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if confidence_key not in metadata:
            missing += 1
            kept.append(record)
            continue
        try:
            conf = float(metadata[confidence_key])
        except (TypeError, ValueError):
            missing += 1
            kept.append(record)
            continue
        if conf >= confidence_min:
            kept.append(record)
        else:
            dropped += 1
    return kept, dropped, missing


def _stratified_sample(
    records: Sequence[Dict[str, Any]],
    *,
    n_per_side: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {"harmful": [], "harmless": []}
    for record in records:
        label = str(record.get("label", "")).strip().lower()
        if label in by_label:
            by_label[label].append(record)
    sampled: List[Dict[str, Any]] = []
    final_counts: Dict[str, int] = {}
    for offset, (label, bucket) in enumerate(sorted(by_label.items())):
        rng = random.Random(int(seed) + offset)
        rng.shuffle(bucket)
        take = bucket[: max(0, int(n_per_side))]
        final_counts[label] = len(take)
        sampled.extend(take)
    rng = random.Random(int(seed) + 17)
    rng.shuffle(sampled)
    return sampled, final_counts


def curate_phase1_subset(
    records: Sequence[Dict[str, Any]],
    *,
    baseline_name: str,
    mode: CurationMode,
    tokenizer: Any = None,
    n_per_side: int = 2500,
    confidence_min: float = 0.7,
    length_min: int = 5,
    length_max: int = 500,
    seed: int = 2042,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (curated_records, summary).

    Pipeline by mode:
      * off    — identity. alignment_set == train_set.
      * minimal — length filter + teacher-confidence filter + stratified sample.
      * strict — per-baseline pre-filter, then minimal pipeline.

    The teacher confidence field is consumed from ``record.metadata
    [teacher_first_gen_top1_prob]``; populating that field is the job of
    ``src/data/teacher_confidence.py`` and is opt-out (no teacher forward
    in ``off`` mode). Records missing the field are kept (treated as
    full confidence) so unit tests do not require a real teacher model.
    """

    mode = str(mode or "off").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown curation_mode {mode!r}; expected one of {VALID_MODES}.")

    summary: Dict[str, Any] = {
        "baseline": baseline_name,
        "mode": mode,
        "input_count": len(records),
        "input_label_counts": _label_counts(records),
        "pre_filter_dropped": 0,
        "length_dropped": 0,
        "confidence_dropped": 0,
        "confidence_missing": 0,
        "stratified_final_counts": {},
        "params": {
            "n_per_side": int(n_per_side),
            "confidence_min": float(confidence_min),
            "length_min": int(length_min),
            "length_max": int(length_max),
            "seed": int(seed),
        },
    }

    if mode == "off":
        summary["output_count"] = len(records)
        summary["output_label_counts"] = summary["input_label_counts"]
        return list(records), summary

    pipeline = list(records)

    if mode == "strict":
        pre = PRE_FILTERS.get(baseline_name)
        if pre is not None:
            before = len(pipeline)
            pipeline = pre(pipeline)
            summary["pre_filter_dropped"] = before - len(pipeline)

    pipeline, len_dropped = _length_filter(
        pipeline,
        tokenizer=tokenizer,
        min_tokens=int(length_min),
        max_tokens=int(length_max),
    )
    summary["length_dropped"] = len_dropped

    pipeline, conf_dropped, conf_missing = _confidence_filter(
        pipeline,
        confidence_min=float(confidence_min),
    )
    summary["confidence_dropped"] = conf_dropped
    summary["confidence_missing"] = conf_missing

    sampled, final_counts = _stratified_sample(
        pipeline,
        n_per_side=int(n_per_side),
        seed=int(seed),
    )
    summary["stratified_final_counts"] = final_counts
    summary["output_count"] = len(sampled)
    summary["output_label_counts"] = final_counts
    return sampled, summary


def _label_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        label = str(record.get("label", "")).strip().lower() or "_unlabeled"
        counts[label] = counts.get(label, 0) + 1
    return counts
