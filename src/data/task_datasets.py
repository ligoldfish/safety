from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_from_disk as hf_load_from_disk
except Exception:
    hf_load_from_disk = None


MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


@dataclass
class MCQExample:
    sample_id: str
    question: str
    choices: List[str]
    answer_index: int
    subject: str = ""


def _read_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if (
        isinstance(payload, dict)
        and "data" in payload
        and isinstance(payload["data"], list)
    ):
        return payload["data"]
    raise ValueError(f"Unsupported JSON format in {path}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to read parquet files") from exc
    return pd.read_parquet(path).to_dict("records")


def _load_rows_from_path(path: Path, split: str) -> List[Dict[str, Any]]:
    if path.is_file():
        if path.suffix == ".jsonl":
            return _read_jsonl(path)
        if path.suffix == ".json":
            return _read_json(path)
        if path.suffix == ".parquet":
            return _read_parquet(path)
        raise ValueError(f"Unsupported file format: {path}")

    dataset_dict_marker = path / "dataset_dict.json"
    dataset_info_marker = path / "dataset_info.json"
    state_marker = path / "state.json"
    if dataset_dict_marker.exists() or (dataset_info_marker.exists() and state_marker.exists()):
        if hf_load_from_disk is None:
            raise RuntimeError(
                "datasets is required to read HuggingFace save_to_disk datasets"
            )
        loaded = hf_load_from_disk(str(path))
        if hasattr(loaded, "keys"):
            split_name = split if split in loaded else next(iter(loaded.keys()), None)
            if split_name is None:
                return []
            return list(loaded[split_name])
        return list(loaded)

    split_candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / f"{split}.parquet",
    ]
    for c in split_candidates:
        if c.exists():
            return _load_rows_from_path(c, split)

    rows: List[Dict[str, Any]] = []
    for p in sorted(path.rglob("*.jsonl")):
        rows.extend(_read_jsonl(p))
    if rows:
        return rows
    for p in sorted(path.rglob("*.json")):
        rows.extend(_read_json(p))
    if rows:
        return rows
    for p in sorted(path.rglob("*.parquet")):
        rows.extend(_read_parquet(p))
    return rows


def _normalize_arc_row(row: Dict[str, Any], sample_id: str) -> Optional[MCQExample]:
    question = str(row.get("question") or row.get("question_stem") or "").strip()
    if not question:
        return None

    if "choices" in row and isinstance(row["choices"], dict):
        texts = [str(x) for x in row["choices"].get("text", [])]
        labels = [str(x) for x in row["choices"].get("label", [])]
        answer_key = str(row.get("answerKey", "")).strip()
        if not texts or not labels or answer_key not in labels:
            return None
        answer_index = labels.index(answer_key)
        return MCQExample(
            sample_id=sample_id,
            question=question,
            choices=texts,
            answer_index=answer_index,
            subject=str(row.get("subject", "")),
        )

    choices = row.get("choices")
    if isinstance(choices, list):
        answer = row.get("answer")
        if isinstance(answer, int):
            answer_idx = int(answer)
        else:
            answer_idx = int(str(answer))
        if 0 <= answer_idx < len(choices):
            return MCQExample(
                sample_id=sample_id,
                question=question,
                choices=[str(c) for c in choices],
                answer_index=answer_idx,
                subject=str(row.get("subject", "")),
            )
    return None


def _normalize_mmlu_row(row: Dict[str, Any], sample_id: str) -> Optional[MCQExample]:
    question = str(row.get("question", "")).strip()
    if not question:
        return None

    choices = row.get("choices")
    answer = row.get("answer")

    if isinstance(choices, list) and answer is not None:
        answer_idx = int(answer)
        if 0 <= answer_idx < len(choices):
            return MCQExample(
                sample_id=sample_id,
                question=question,
                choices=[str(c) for c in choices],
                answer_index=answer_idx,
                subject=str(row.get("subject", "")),
            )
    return _normalize_arc_row(row, sample_id)


def _normalize_piqa_row(row: Dict[str, Any], sample_id: str) -> Optional[MCQExample]:
    question = str(row.get("question") or row.get("goal") or "").strip()
    if not question:
        return None

    choices = row.get("choices")
    if not isinstance(choices, list) and "sol1" in row and "sol2" in row:
        choices = [row["sol1"], row["sol2"]]
    if not isinstance(choices, list) or len(choices) != 2:
        return None

    if row.get("answer_index") is not None:
        ans = int(row["answer_index"])
    else:
        raw = str(row.get("answer", "")).strip().upper()
        ans = 0 if raw in {"A", "0", "1"} else 1

    if ans not in {0, 1}:
        return None
    return MCQExample(
        sample_id=sample_id,
        question=question,
        choices=[str(choices[0]), str(choices[1])],
        answer_index=ans,
        subject=str(row.get("subject", "piqa")),
    )


def _normalize_hellaswag_row(
    row: Dict[str, Any], sample_id: str
) -> Optional[MCQExample]:
    question = str(row.get("ctx") or row.get("question") or "").strip()
    endings = row.get("endings")
    if not question or not isinstance(endings, list) or len(endings) < 2:
        return None
    label = row.get("label")
    if label is None:
        return None
    ans = int(str(label).strip())
    if ans < 0 or ans >= len(endings):
        return None
    return MCQExample(
        sample_id=sample_id,
        question=question,
        choices=[str(x) for x in endings],
        answer_index=ans,
        subject=str(row.get("activity_label", "hellaswag")),
    )


def _normalize_winogrande_row(
    row: Dict[str, Any], sample_id: str
) -> Optional[MCQExample]:
    question = str(row.get("sentence") or row.get("question") or "").strip()
    o1 = row.get("option1")
    o2 = row.get("option2")
    if not question or o1 is None or o2 is None:
        return None
    ans = str(row.get("answer", "")).strip()
    if ans == "1":
        idx = 0
    elif ans == "2":
        idx = 1
    else:
        return None
    return MCQExample(
        sample_id=sample_id,
        question=question,
        choices=[str(o1), str(o2)],
        answer_index=idx,
        subject=str(row.get("subject", "winogrande")),
    )


def _sample_rows(
    rows: List[Dict[str, Any]], max_samples: int, seed: int, shuffle: bool
) -> List[Dict[str, Any]]:
    if shuffle:
        random.Random(seed).shuffle(rows)
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def load_mcq_dataset(
    dataset_name: str,
    data_path: str,
    max_samples: int,
    seed: int,
    split: str = "test",
    shuffle: bool = False,
) -> List[MCQExample]:
    name = dataset_name.lower().strip()
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rows: List[Dict[str, Any]] = []
    if name == "mmlu" and path.is_dir():
        for subject in MMLU_SUBJECTS:
            subject_path = path / subject
            if not subject_path.exists():
                continue
            rows.extend(_load_rows_from_path(subject_path, split))
        if not rows:
            rows = _load_rows_from_path(path, split)
    else:
        rows = _load_rows_from_path(path, split)

    rows = _sample_rows(rows, max_samples=max_samples, seed=seed, shuffle=shuffle)
    examples: List[MCQExample] = []
    for i, row in enumerate(rows):
        sample_id = str(row.get("id", f"{name}_{i:06d}"))
        if name == "mmlu":
            ex = _normalize_mmlu_row(row, sample_id)
        elif name == "piqa":
            ex = _normalize_piqa_row(row, sample_id)
        elif name == "hellaswag":
            ex = _normalize_hellaswag_row(row, sample_id)
        elif name in {"winogrande", "wino"}:
            ex = _normalize_winogrande_row(row, sample_id)
        else:
            ex = _normalize_arc_row(row, sample_id)
        if ex is not None:
            examples.append(ex)
    return examples


def render_mcq_prompt(example: MCQExample) -> str:
    option_labels = [chr(ord("A") + i) for i in range(len(example.choices))]
    options = "\n".join(f"{k}. {v}" for k, v in zip(option_labels, example.choices))
    return f"Question: {example.question}\n{options}\nAnswer:"
