from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GSM8KExample:
    sample_id: str
    question: str
    answer_text: str
    final_answer: str


@dataclass
class CodeExample:
    task_id: str
    prompt: str
    tests: list[str]
    entry_point: str = ""
    reference_solution: str = ""


def _read_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
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
    except Exception as exc:  # pragma: no cover
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

    split_candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / f"{split}.parquet",
    ]
    for candidate in split_candidates:
        if candidate.exists():
            return _load_rows_from_path(candidate, split)

    rows: List[Dict[str, Any]] = []
    for candidate in sorted(path.rglob("*.jsonl")):
        rows.extend(_read_jsonl(candidate))
    if rows:
        return rows
    for candidate in sorted(path.rglob("*.json")):
        rows.extend(_read_json(candidate))
    if rows:
        return rows
    for candidate in sorted(path.rglob("*.parquet")):
        rows.extend(_read_parquet(candidate))
    return rows


def normalize_numeric_answer(text: str) -> str:
    cleaned = str(text).strip().replace(",", "")
    if not cleaned:
        return ""
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]

    fraction_match = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", cleaned)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator == 0:
            return cleaned
        fraction = Fraction(numerator, denominator)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return str(float(fraction))

    try:
        decimal_value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned.lower()

    normalized = format(decimal_value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def extract_reference_number(text: str) -> str:
    answer_text = str(text)
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", answer_text)
    if not matches:
        return ""
    return normalize_numeric_answer(matches[-1])


def extract_prediction_number(text: str) -> str:
    answer_text = str(text)
    final_answer_match = re.search(
        r"final\s+answer\s*:\s*([^\n]+)",
        answer_text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        extracted = extract_reference_number(final_answer_match.group(1))
        if extracted:
            return extracted
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", answer_text)
    if not matches:
        return ""
    return normalize_numeric_answer(matches[-1])


def load_gsm8k_examples(
    data_path: str,
    *,
    split: str = "test",
    max_samples: int = 0,
) -> List[GSM8KExample]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rows = _load_rows_from_path(path, split)
    if max_samples > 0:
        rows = rows[:max_samples]

    examples: List[GSM8KExample] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("input") or row.get("prompt") or "").strip()
        answer_text = str(row.get("answer") or row.get("target") or row.get("output") or "").strip()
        final_answer = extract_reference_number(answer_text)
        if not question or not answer_text or not final_answer:
            continue
        examples.append(
            GSM8KExample(
                sample_id=str(row.get("id", f"gsm8k_{idx:06d}")),
                question=question,
                answer_text=answer_text,
                final_answer=final_answer,
            )
        )
    return examples


def _normalize_code_tests(raw_tests: Any) -> list[str]:
    if raw_tests is None:
        return []
    if isinstance(raw_tests, list):
        return [str(item) for item in raw_tests if str(item).strip()]
    if isinstance(raw_tests, str):
        stripped = raw_tests.strip()
        if not stripped:
            return []
        return [stripped]
    return [str(raw_tests)]


def load_code_examples(
    dataset_name: str,
    data_path: str,
    *,
    split: str = "test",
    max_samples: int = 0,
) -> List[CodeExample]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rows = _load_rows_from_path(path, split)
    if max_samples > 0:
        rows = rows[:max_samples]

    examples: List[CodeExample] = []
    name = str(dataset_name).strip().lower()
    for idx, row in enumerate(rows):
        task_id = str(row.get("task_id") or row.get("id") or f"{name}_{idx:06d}")
        if name == "humaneval":
            prompt = str(row.get("prompt") or "").rstrip()
            tests = _normalize_code_tests(row.get("test"))
            entry_point = str(row.get("entry_point") or "").strip()
            reference_solution = str(row.get("canonical_solution") or row.get("solution") or "")
        elif name == "mbpp":
            prompt = str(row.get("prompt") or row.get("text") or "").strip()
            tests = _normalize_code_tests(row.get("test_list") or row.get("tests"))
            entry_point = str(row.get("entry_point") or row.get("function_name") or "").strip()
            reference_solution = str(row.get("code") or row.get("reference_solution") or "")
        else:
            raise ValueError(f"Unsupported code dataset: {dataset_name}")

        if not prompt or not tests:
            continue
        examples.append(
            CodeExample(
                task_id=task_id,
                prompt=prompt,
                tests=tests,
                entry_point=entry_point,
                reference_solution=reference_solution,
            )
        )
    return examples


def sanitize_code_generation(text: str) -> str:
    code = str(text).strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        code = max(fenced, key=len).strip()
    code = re.sub(r"^python\s*", "", code, flags=re.IGNORECASE)
    return code.strip()
