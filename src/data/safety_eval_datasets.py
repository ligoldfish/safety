"""Safety evaluation dataset loaders.

These loaders return lists of plain dictionaries describing prompts to be
graded by the project's safety evaluators. They live here (and not in the
training pipeline) so that any HuggingFace ``load_dataset`` call lives in
exactly one place per dataset.

Datasets covered:

* ``WildGuardTest``               - ``allenai/wildguardmix``, config
                                    ``wildguardtest``.
* ``WildJailbreak`` eval split    - ``allenai/wildjailbreak``, config
                                    ``eval``. ``delimiter='\\t'`` and
                                    ``keep_default_na=False`` are
                                    mandatory upstream loader options.
* ``CoCoNot`` contrast            - ``allenai/coconot``, config
                                    ``contrast``, split ``test``.
* ``BeaverTails-Evaluation``      - ``PKU-Alignment/BeaverTails-Evaluation``,
                                    split ``test``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SafetyEvalExample:
    sample_id: str
    prompt: str
    label: str
    category: str = ""
    source: str = ""
    extra: Dict[str, Any] | None = None


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset  # type: ignore[import-not-found]

    return load_dataset(*args, **kwargs)


def _str_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# WildGuardTest
# ---------------------------------------------------------------------------


def load_wildguard_test(
    *,
    cache_dir: Optional[str] = None,
    source_name: str = "allenai/wildguardmix",
    config_name: str = "wildguardtest",
) -> List[SafetyEvalExample]:
    dataset = _load_dataset(source_name, config_name, cache_dir=cache_dir)
    rows = dataset if not hasattr(dataset, "keys") else dataset[next(iter(dataset.keys()))]
    examples: List[SafetyEvalExample] = []
    for index, row in enumerate(rows):
        prompt = _str_or_empty(row.get("prompt") or row.get("input"))
        if not prompt.strip():
            continue
        prompt_harm = _str_or_empty(row.get("prompt_harm_label"))
        examples.append(
            SafetyEvalExample(
                sample_id=str(row.get("id", f"wildguard_test_{index:06d}")),
                prompt=prompt,
                label=prompt_harm or "unknown",
                category=_str_or_empty(row.get("subcategory") or row.get("category")),
                source=f"{source_name}:{config_name}",
                extra={
                    "response": _str_or_empty(row.get("response")),
                    "response_refusal_label": _str_or_empty(row.get("response_refusal_label")),
                },
            )
        )
    return examples


# ---------------------------------------------------------------------------
# WildJailbreak eval
# ---------------------------------------------------------------------------


def load_wildjailbreak_eval(
    *,
    cache_dir: Optional[str] = None,
    source_name: str = "allenai/wildjailbreak",
    config_name: str = "eval",
) -> List[SafetyEvalExample]:
    dataset = _load_dataset(
        source_name,
        config_name,
        delimiter="\t",
        keep_default_na=False,
        cache_dir=cache_dir,
    )
    rows = dataset if not hasattr(dataset, "keys") else dataset[next(iter(dataset.keys()))]
    examples: List[SafetyEvalExample] = []
    for index, row in enumerate(rows):
        prompt = _str_or_empty(row.get("adversarial") or row.get("prompt"))
        if not prompt.strip():
            continue
        examples.append(
            SafetyEvalExample(
                sample_id=str(row.get("id", f"wildjailbreak_eval_{index:06d}")),
                prompt=prompt,
                label=_str_or_empty(row.get("data_type") or row.get("type") or "unknown"),
                category=_str_or_empty(row.get("tactic")),
                source=f"{source_name}:{config_name}",
                extra={
                    "vanilla": _str_or_empty(row.get("vanilla")),
                    "completion": _str_or_empty(row.get("completion")),
                },
            )
        )
    return examples


# ---------------------------------------------------------------------------
# CoCoNot contrast
# ---------------------------------------------------------------------------


def load_coconot_contrast(
    *,
    cache_dir: Optional[str] = None,
    source_name: str = "allenai/coconot",
    config_name: str = "contrast",
    split: str = "test",
) -> List[SafetyEvalExample]:
    rows = _load_dataset(source_name, config_name, split=split, cache_dir=cache_dir)
    examples: List[SafetyEvalExample] = []
    for index, row in enumerate(rows):
        prompt = _str_or_empty(row.get("prompt"))
        if not prompt.strip():
            continue
        examples.append(
            SafetyEvalExample(
                sample_id=str(row.get("id", f"coconot_contrast_{index:06d}")),
                prompt=prompt,
                label=_str_or_empty(row.get("category", "contrast")),
                category=_str_or_empty(row.get("subcategory")),
                source=f"{source_name}:{config_name}:{split}",
                extra={"response": _str_or_empty(row.get("response"))},
            )
        )
    return examples


# ---------------------------------------------------------------------------
# BeaverTails-Evaluation
# ---------------------------------------------------------------------------


def load_beavertails_evaluation(
    *,
    cache_dir: Optional[str] = None,
    source_name: str = "PKU-Alignment/BeaverTails-Evaluation",
    split: str = "test",
) -> List[SafetyEvalExample]:
    rows = _load_dataset(source_name, split=split, cache_dir=cache_dir)
    examples: List[SafetyEvalExample] = []
    for index, row in enumerate(rows):
        prompt = _str_or_empty(row.get("prompt"))
        if not prompt.strip():
            continue
        category = row.get("category")
        if isinstance(category, dict):
            primary_category = ",".join(sorted(key for key, value in category.items() if value))
        else:
            primary_category = _str_or_empty(category)
        examples.append(
            SafetyEvalExample(
                sample_id=str(row.get("id", f"beavertails_eval_{index:06d}")),
                prompt=prompt,
                label="beavertails_eval",
                category=primary_category,
                source=f"{source_name}:{split}",
                extra={"raw_category": category},
            )
        )
    return examples


SAFETY_EVAL_LOADERS = {
    "wildguard_test": load_wildguard_test,
    "wildjailbreak_eval": load_wildjailbreak_eval,
    "coconot_contrast": load_coconot_contrast,
    "beavertails_evaluation": load_beavertails_evaluation,
}


__all__ = [
    "SAFETY_EVAL_LOADERS",
    "SafetyEvalExample",
    "load_beavertails_evaluation",
    "load_coconot_contrast",
    "load_wildguard_test",
    "load_wildjailbreak_eval",
]
