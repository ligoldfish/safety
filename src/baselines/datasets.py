from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List

from src.data.template_qwen import strip_qwen_thinking_content

try:
    from datasets import load_from_disk as hf_load_from_disk
except Exception:
    hf_load_from_disk = None

try:
    from thefuzz import process as fuzz_process
except Exception:
    fuzz_process = None


HF_DATASET_METADATA_FILES = {
    "dataset_dict.json",
    "dataset_info.json",
    "state.json",
}
MBPP_STANDARD_TEST_START_TASK_ID = 11
MBPP_STANDARD_TEST_END_TASK_ID = 510
MBPP_STANDARD_TEST_SIZE = (
    MBPP_STANDARD_TEST_END_TASK_ID - MBPP_STANDARD_TEST_START_TASK_ID + 1
)
OFFICIAL_GSM8K_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?")


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

    dataset_dict_marker = path / "dataset_dict.json"
    dataset_info_marker = path / "dataset_info.json"
    state_marker = path / "state.json"
    if dataset_dict_marker.exists() or (dataset_info_marker.exists() and state_marker.exists()):
        if hf_load_from_disk is None:
            raise RuntimeError("datasets is required to read HuggingFace save_to_disk datasets")
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
    for candidate in split_candidates:
        if candidate.exists():
            return _load_rows_from_path(candidate, split)

    rows: List[Dict[str, Any]] = []
    for candidate in sorted(path.rglob("*.jsonl")):
        rows.extend(_read_jsonl(candidate))
    if rows:
        return rows
    for candidate in sorted(path.rglob("*.json")):
        if candidate.name in HF_DATASET_METADATA_FILES:
            continue
        rows.extend(_read_json(candidate))
    if rows:
        return rows
    for candidate in sorted(path.rglob("*.parquet")):
        rows.extend(_read_parquet(candidate))
    return rows


def _extract_mbpp_task_id(row: Dict[str, Any]) -> int | None:
    raw_value = row.get("task_id", row.get("id"))
    if raw_value is None:
        return None
    if isinstance(raw_value, int):
        return raw_value
    match = re.search(r"\d+", str(raw_value))
    if not match:
        return None
    return int(match.group(0))


def _resolve_mbpp_test_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    split_key_candidates = ("split", "subset", "partition", "source_split")
    split_rows = [
        row
        for row in rows
        if any(str(row.get(key, "")).strip().lower() == "test" for key in split_key_candidates)
    ]
    if len(split_rows) == MBPP_STANDARD_TEST_SIZE:
        return split_rows

    task_rows: List[tuple[int, Dict[str, Any]]] = []
    for row in rows:
        task_id = _extract_mbpp_task_id(row)
        if task_id is None:
            continue
        if MBPP_STANDARD_TEST_START_TASK_ID <= task_id <= MBPP_STANDARD_TEST_END_TASK_ID:
            task_rows.append((task_id, row))
    if len(task_rows) == MBPP_STANDARD_TEST_SIZE:
        task_rows.sort(key=lambda item: item[0])
        return [row for _, row in task_rows]

    if len(rows) == MBPP_STANDARD_TEST_SIZE:
        return rows

    raise ValueError(
        "MBPP test split should resolve to exactly 500 rows. "
        f"Loaded {len(rows)} rows instead."
    )


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


def extract_multiple_choice_prediction(
    text: str,
    option_labels: List[str],
) -> str:
    answer_text = str(text).strip()
    if not answer_text:
        return ""

    option_set = {str(label).upper() for label in option_labels}
    patterns = [
        r'"answer"\s*:\s*"([A-Z])"',
        r"'answer'\s*:\s*'([A-Z])'",
        r"\banswer\s*[:=]\s*['\"]?([A-Z])['\"]?\b",
        r"\boption\s*[:=]?\s*([A-Z])\b",
        r"^\s*([A-Z])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer_text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        candidate = match.group(1).upper()
        if candidate in option_set:
            return candidate

    cleaned = re.sub(r"[^A-Za-z]", " ", answer_text).split()
    for token in cleaned:
        candidate = token.upper()
        if candidate in option_set:
            return candidate
    return ""


def _replace_choice_text_with_labels(
    text: str,
    choice_map: Dict[str, str],
) -> str:
    processed = str(text)
    for label, choice in sorted(choice_map.items(), key=lambda item: len(str(item[1])), reverse=True):
        choice_text = re.sub(r"\s+", " ", str(choice)).strip()
        if not choice_text:
            continue
        processed = re.sub(
            re.escape(choice_text),
            f" {label} ",
            processed,
            flags=re.IGNORECASE,
        )
    return processed


def extract_official_mmlu_prediction(
    text: str,
    option_labels: List[str],
    choice_map: Dict[str, str],
) -> str:
    answer_text = str(text).strip()
    if not answer_text:
        return ""

    processed = _replace_choice_text_with_labels(answer_text, choice_map)
    label_group = "|".join(re.escape(label) for label in option_labels)
    patterns = [
        rf"answer(?:\s+is|:)?\s*\(?({label_group})\)?\b",
        rf"the\s+correct\s+answer\s+is\s*\(?({label_group})\)?\b",
        rf"choose\s*\(?({label_group})\)?\b",
        rf"option\s*\(?({label_group})\)?\b",
        rf"^\s*\(?({label_group})\)?[\s\.\):,-]*$",
        rf"\b({label_group})\b(?=[\s\.\):,-]*(?:is\s+correct|is\s+right|$))",
    ]
    for pattern in patterns:
        match = re.search(pattern, processed, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        candidate = match.group(1).upper()
        if candidate in option_labels:
            return candidate

    match = re.search(rf"\b({label_group})\b", processed, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in option_labels:
            return candidate

    choice_texts = [str(choice_map[label]) for label in option_labels if label in choice_map]
    if fuzz_process is not None and choice_texts:
        result = fuzz_process.extractOne(answer_text, choice_texts)
        if result:
            matched_choice = result[0]
            for label in option_labels:
                if str(choice_map.get(label, "")) == matched_choice:
                    return label

    best_label = ""
    best_score = 0.0
    normalized_answer = re.sub(r"\s+", " ", answer_text).strip().lower()
    for label in option_labels:
        choice_text = re.sub(r"\s+", " ", str(choice_map.get(label, ""))).strip().lower()
        if not choice_text:
            continue
        score = SequenceMatcher(a=normalized_answer, b=choice_text).ratio()
        if score > best_score:
            best_score = score
            best_label = label
    if best_score >= 0.6:
        return best_label
    return ""


def extract_official_gsm8k_prediction(text: str) -> str:
    answer_text = str(text)
    matches = OFFICIAL_GSM8K_NUMBER_RE.findall(answer_text)
    if not matches:
        return ""
    return matches[-1].replace(",", "").lstrip("+")


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
    name = str(dataset_name).strip().lower()
    if name == "mbpp" and str(split).strip().lower() == "test":
        rows = _resolve_mbpp_test_rows(rows)
    if max_samples > 0:
        rows = rows[:max_samples]

    examples: List[CodeExample] = []
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


def sanitize_code_generation(
    text: str,
    *,
    require_final_response: bool = False,
) -> str:
    code = strip_qwen_thinking_content(
        str(text),
        require_final_response=require_final_response,
    )
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        code = max(fenced, key=len).strip()
    code = re.sub(r"^python\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(
        r"^(?:sure[,!: ]*|here(?:'s| is)(?: the)? code[:：]?\s*|the code is[:：]?\s*|final code[:：]?\s*)",
        "",
        code,
        flags=re.IGNORECASE,
    )
    return code.strip()
