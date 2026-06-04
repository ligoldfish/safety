"""Smoke tests for src/data/curation.py.

No teacher model is loaded — tests pre-populate
``metadata.teacher_first_gen_top1_prob`` directly so the curation
pipeline is exercised end-to-end without depending on transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.curation import (
    DEFAULT_MODE,
    PRE_FILTERS,
    VALID_MODES,
    curate_phase1_subset,
    resolve_curation_mode,
)


def _record(
    rid: str,
    label: str,
    user_text: str,
    *,
    data_type: str = "",
    prompt_harm_label: str = "",
    response_refusal_label: str = "",
    teacher_conf: float | None = None,
) -> dict:
    metadata: dict = {}
    if data_type:
        metadata["data_type"] = data_type
    if prompt_harm_label:
        metadata["prompt_harm_label"] = prompt_harm_label
    if response_refusal_label:
        metadata["response_refusal_label"] = response_refusal_label
    if teacher_conf is not None:
        metadata["teacher_first_gen_top1_prob"] = float(teacher_conf)
    return {
        "id": rid,
        "label": label,
        "user_text": user_text,
        "messages": [{"role": "user", "content": user_text}],
        "metadata": metadata,
    }


def test_off_mode_is_identity():
    records = [_record(f"r{i}", "harmful" if i % 2 == 0 else "harmless", "hi there") for i in range(10)]
    curated, summary = curate_phase1_subset(records, baseline_name="pan", mode="off")
    assert curated == records
    assert summary["mode"] == "off"
    assert summary["output_count"] == 10
    assert summary["pre_filter_dropped"] == 0


def test_wjb_pre_filter_drops_adversarial():
    records = [
        _record("v1", "harmful", "hi there", data_type="vanilla_harmful", teacher_conf=0.9),
        _record("a1", "harmful", "hi there", data_type="adversarial_harmful", teacher_conf=0.9),
        _record("v2", "harmless", "hi there", data_type="vanilla_benign", teacher_conf=0.9),
        _record("a2", "harmless", "hi there", data_type="adversarial_benign", teacher_conf=0.9),
    ]
    kept = PRE_FILTERS["wildjailbreak"](records)
    assert {r["id"] for r in kept} == {"v1", "v2"}


def test_wgm_pre_filter_drops_ambiguous():
    records = [
        _record("a", "harmful", "x", prompt_harm_label="harmful", response_refusal_label="refusal", teacher_conf=0.9),
        _record("b", "harmless", "x", prompt_harm_label="unharmful", response_refusal_label="compliance", teacher_conf=0.9),
        _record("c", "harmful", "x", prompt_harm_label="vague", response_refusal_label="refusal", teacher_conf=0.9),
        _record("d", "harmless", "x", prompt_harm_label="unharmful", response_refusal_label="other", teacher_conf=0.9),
    ]
    kept = PRE_FILTERS["wildguardmix"](records)
    assert {r["id"] for r in kept} == {"a", "b"}


def test_length_filter_excludes_too_short_and_too_long():
    records = [
        _record("short", "harmful", "hi", teacher_conf=0.9),
        _record("ok", "harmful", "hi " * 30, teacher_conf=0.9),
        _record("long", "harmless", "hi " * 700, teacher_conf=0.9),
    ]
    curated, summary = curate_phase1_subset(
        records,
        baseline_name="beavertails_category",
        mode="minimal",
        length_min=5,
        length_max=500,
        n_per_side=10,
    )
    ids = {r["id"] for r in curated}
    assert "ok" in ids
    assert "short" not in ids
    assert "long" not in ids
    assert summary["length_dropped"] == 2


def test_confidence_filter_drops_below_threshold():
    records = [
        _record("hi_conf_h", "harmful", "decent prompt length here ok for tests", teacher_conf=0.95),
        _record("lo_conf_h", "harmful", "decent prompt length here ok for tests", teacher_conf=0.3),
        _record("hi_conf_b", "harmless", "decent prompt length here ok for tests", teacher_conf=0.85),
        _record("lo_conf_b", "harmless", "decent prompt length here ok for tests", teacher_conf=0.4),
    ]
    curated, summary = curate_phase1_subset(
        records,
        baseline_name="beavertails_category",
        mode="minimal",
        confidence_min=0.7,
        n_per_side=10,
    )
    ids = {r["id"] for r in curated}
    assert ids == {"hi_conf_h", "hi_conf_b"}
    assert summary["confidence_dropped"] == 2


def test_stratified_sample_caps_per_side():
    records = []
    for i in range(50):
        records.append(_record(f"h{i}", "harmful", "decent prompt content here for tests", teacher_conf=0.9))
    for i in range(50):
        records.append(_record(f"b{i}", "harmless", "decent prompt content here for tests", teacher_conf=0.9))
    curated, summary = curate_phase1_subset(
        records,
        baseline_name="beavertails_category",
        mode="minimal",
        n_per_side=10,
        seed=42,
    )
    labels = [r["label"] for r in curated]
    assert labels.count("harmful") == 10
    assert labels.count("harmless") == 10
    assert summary["stratified_final_counts"] == {"harmful": 10, "harmless": 10}


def test_strict_mode_combines_pre_filter_with_universal():
    records = [
        _record("v1", "harmful", "decent prompt content here for tests", data_type="vanilla_harmful", teacher_conf=0.9),
        _record("v2", "harmless", "decent prompt content here for tests", data_type="vanilla_benign", teacher_conf=0.9),
        _record("a1", "harmful", "decent prompt content here for tests", data_type="adversarial_harmful", teacher_conf=0.99),
        _record("lc", "harmful", "decent prompt content here for tests", data_type="vanilla_harmful", teacher_conf=0.3),
    ]
    curated, summary = curate_phase1_subset(
        records,
        baseline_name="wildjailbreak",
        mode="strict",
        n_per_side=10,
        confidence_min=0.7,
    )
    ids = {r["id"] for r in curated}
    assert ids == {"v1", "v2"}
    assert summary["pre_filter_dropped"] == 1
    assert summary["confidence_dropped"] == 1


def test_resolve_curation_mode_auto_and_explicit():
    assert resolve_curation_mode("wildjailbreak") == "strict"
    assert resolve_curation_mode("pan") == "off"
    assert resolve_curation_mode("beavertails_category") == "minimal"
    assert resolve_curation_mode("tulu3_safety") == "off"
    assert resolve_curation_mode("hh_rlhf") == "off"
    assert resolve_curation_mode("wildjailbreak", "auto") == "strict"
    assert resolve_curation_mode("wildjailbreak", "off") == "off"
    assert resolve_curation_mode("unknown_baseline") == "off"


def test_default_mode_table_matches_user_decisions():
    """Lock in the per-baseline mode defaults agreed in the plan."""
    assert DEFAULT_MODE["pan"] == "off"
    assert DEFAULT_MODE["safety_tuned_llamas"] == "off"
    assert DEFAULT_MODE["coconot"] == "off"
    assert DEFAULT_MODE["hh_rlhf"] == "off"
    assert DEFAULT_MODE["tulu3_safety"] == "off"
    assert DEFAULT_MODE["beavertails_category"] == "minimal"
    assert DEFAULT_MODE["wildjailbreak"] == "strict"
    assert DEFAULT_MODE["wildguardmix"] == "strict"


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        curate_phase1_subset([], baseline_name="pan", mode="bogus")


def test_valid_modes_constant_complete():
    assert set(VALID_MODES) == {"off", "minimal", "strict"}


def test_records_missing_confidence_are_kept_with_missing_count():
    records = [
        _record("a", "harmful", "decent prompt content here for tests", teacher_conf=0.9),
        _record("b", "harmless", "decent prompt content here for tests"),  # no teacher_conf
    ]
    curated, summary = curate_phase1_subset(
        records,
        baseline_name="beavertails_category",
        mode="minimal",
        n_per_side=10,
    )
    ids = {r["id"] for r in curated}
    assert ids == {"a", "b"}
    assert summary["confidence_missing"] == 1
    assert summary["confidence_dropped"] == 0
