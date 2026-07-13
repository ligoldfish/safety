from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pairs import DEFAULT_PAIR, PAIRS

FORMAL_PAIRS = (
    "qwen35_9b_to_08b",
    "llama31_8b_to_1b",
    "qwen3_8b_to_06b",
    "qwen3_8b_to_4b",
    "qwen3_4b_to_06b",
)
DATASETS = (
    "pan", "safety_tuned_llamas", "coconot",
    "wildguardmix", "wildjailbreak", "c5",
)
METHODS = ("ours", "ours_sft1", "sft", "distill", "nosft")
CORE_METRICS = (
    "llm_judge_asr",
    "llm_judge_over_refusal",
    "llm_judge_refusal_rate",
)
OPTIONAL_METRICS = (
    "judge_keyword_agreement",
    "judge_cohen_kappa",
    "judge_parse_rate",
    "judge_num_items",
    "judge_num_parsed",
    "judge_num_harmful_scored",
    "judge_num_harmless_scored",
)
CSV_FIELDS = (
    "model_pair", "teacher_model", "student_model", "dataset", "method",
    "epoch", "status", "source_kind", *CORE_METRICS, *OPTIONAL_METRICS,
    "source_path",
)


@dataclass(frozen=True)
class ResultSpec:
    pair_id: str
    dataset: str
    method: str
    epoch: str
    result_dir: Path


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metric_error(key: str, value: object) -> str | None:
    rate_metrics = (*CORE_METRICS, "judge_keyword_agreement", "judge_parse_rate")
    if key in rate_metrics:
        if not _valid_number(value) or not math.isfinite(value) or not 0 <= value <= 1:
            return f"{key} must be finite and within [0,1]"
    elif key == "judge_cohen_kappa":
        if not _valid_number(value) or not math.isfinite(value) or not -1 <= value <= 1:
            return f"{key} must be finite and within [-1,1]"
    elif key in {
        "judge_num_items",
        "judge_num_parsed",
        "judge_num_harmful_scored",
        "judge_num_harmless_scored",
    }:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return f"{key} must be a nonnegative integer"
    return None


def _judge_evidence_error(payload: dict[str, object]) -> str | None:
    reasons = []
    num_items = payload.get("judge_num_items")
    if num_items is not None and (
        not isinstance(num_items, int) or isinstance(num_items, bool) or num_items <= 0
    ):
        reasons.append("judge_num_items must be a positive integer")
    parse_rate = payload.get("judge_parse_rate")
    if _valid_number(parse_rate) and math.isfinite(parse_rate) and parse_rate <= 0:
        reasons.append("judge_parse_rate must be positive")
    scored = (
        payload.get("judge_num_harmful_scored"),
        payload.get("judge_num_harmless_scored"),
    )
    if all(isinstance(value, int) and not isinstance(value, bool) for value in scored):
        if sum(scored) <= 0:
            reasons.append("scored item count must be positive")
    if reasons:
        return f"no usable judged items: {'; '.join(reasons)}"
    return None


def _read_candidate(path: Path, summary: bool) -> tuple[dict[str, object] | None, str, str]:
    if not path.is_file():
        return None, "missing", "file not found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, "malformed", f"cannot read file: {exc.strerror or type(exc).__name__}"
    except UnicodeError as exc:
        return None, "malformed", f"cannot decode UTF-8: {getattr(exc, 'reason', type(exc).__name__)}"
    except json.JSONDecodeError as exc:
        return (
            None,
            "malformed",
            f"cannot parse JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
        )
    if summary:
        if not isinstance(payload, dict):
            return None, "malformed", "summary payload is not an object"
        results = payload.get("results", {})
        if not isinstance(results, dict):
            return None, "malformed", "summary results is not an object"
        payload = results.get("pan", {})
        if not isinstance(payload, dict):
            return None, "malformed", "summary results.pan is not an object"
    if not isinstance(payload, dict):
        return None, "malformed", "metric payload is not an object"
    missing = [key for key in CORE_METRICS if key not in payload]
    invalid = [
        error
        for key in (*CORE_METRICS, *OPTIONAL_METRICS)
        if key in payload and payload[key] is not None
        if (error := _metric_error(key, payload[key])) is not None
    ]
    if missing:
        return None, "incomplete", f"missing metrics: {', '.join(missing)}"
    if invalid:
        return None, "malformed", f"invalid metrics: {'; '.join(invalid)}"
    if not summary and (evidence_error := _judge_evidence_error(payload)) is not None:
        return None, "incomplete", evidence_error
    return payload, "ok", ""


def load_result(spec: ResultSpec, project_root: Path) -> dict[str, object]:
    pair = PAIRS[spec.pair_id]
    row: dict[str, object] = {
        "model_pair": spec.pair_id,
        "teacher_model": pair["teacher"]["name"],
        "student_model": pair["student"]["name"],
        "dataset": spec.dataset,
        "method": spec.method,
        "epoch": spec.epoch,
        "status": "missing",
        "source_kind": "",
        **{key: "" for key in (*CORE_METRICS, *OPTIONAL_METRICS)},
        "source_path": "",
        "error": "",
        "candidate_failures": [],
    }
    candidates = (
        (spec.result_dir / "judge_results.json", False, "judge_results", "judge_results"),
        (spec.result_dir / "summary.json", True, "summary", "summary_fallback"),
    )
    failures: list[dict[str, object]] = []
    for path, summary, candidate_kind, source_kind in candidates:
        payload, status, error = _read_candidate(path, summary)
        if payload is None:
            failures.append({
                "kind": candidate_kind,
                "status": status,
                "path": portable_path(path, project_root),
                "error": error,
            })
            continue
        row["status"] = "ok"
        row["source_kind"] = source_kind
        row["source_path"] = portable_path(path, project_root)
        for key in (*CORE_METRICS, *OPTIONAL_METRICS):
            if key in payload:
                row[key] = payload[key]
        if failures:
            row["candidate_failures"] = failures
            row["error"] = "; ".join(
                f"{failure['kind']}: {failure['error']}" for failure in failures
            )
        return row

    precedence = {"missing": 0, "incomplete": 1, "malformed": 2}
    status = max((str(item["status"]) for item in failures), key=precedence.__getitem__)
    row["status"] = status
    row["candidate_failures"] = failures
    row["error"] = "; ".join(
        f"{failure['kind']}: {failure['error']}" for failure in failures
    )
    return row


def _epochs(dataset: str, method: str) -> tuple[str, ...]:
    if method == "nosft":
        return ("single",)
    if dataset == "safety_tuned_llamas" and method in {"sft", "distill"}:
        return ("epoch_006",)
    return ("epoch_002", "epoch_003")


def _method_root(outputs_root: Path, pair_id: str, dataset: str, method: str) -> Path:
    pair = PAIRS[pair_id]
    student_tag = pair["student"]["tag"]
    dataset_suffix = "" if dataset == "pan" else f"_{dataset}"
    if method in {"ours", "ours_sft1"}:
        training = "training_sft1" if method == "ours_sft1" else "training"
        if dataset == "pan":
            return outputs_root / f"{pair_id}_phase1_npu" / training / "eval_suite"
        pair_suffix = "" if pair_id == DEFAULT_PAIR else f"_{pair_id}"
        return (
            outputs_root / f"safety_full_{dataset}_npu{pair_suffix}"
            / "phase1" / training / "eval_suite"
        )
    if method == "sft":
        return outputs_root / "baselines" / f"sft_{student_tag}{dataset_suffix}_npu" / "eval_suite"
    if method == "distill":
        return outputs_root / "baselines" / f"distill_{pair_id}{dataset_suffix}_npu" / "eval_suite"
    if dataset == "pan":
        return outputs_root / "baselines" / f"no_sft_{student_tag}_npu"
    legacy_prefix = "" if pair_id == DEFAULT_PAIR else f"{student_tag}_"
    return outputs_root / "baselines" / f"{legacy_prefix}eval_{dataset}_npu"


def iter_result_specs(
    outputs_root: Path,
    pair_ids: Sequence[str] = FORMAL_PAIRS,
) -> list[ResultSpec]:
    specs = []
    for pair_id in pair_ids:
        for dataset in DATASETS:
            for method in METHODS:
                root = _method_root(outputs_root, pair_id, dataset, method)
                for epoch in _epochs(dataset, method):
                    result_dir = root if epoch == "single" else root / epoch
                    specs.append(ResultSpec(pair_id, dataset, method, epoch, result_dir))
    return specs


def portable_path(path: Path, project_root: Path | None = None) -> str:
    if project_root is not None:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def identity_and_source(row: dict[str, object]) -> dict[str, object]:
    identity = {
        key: row[key]
        for key in ("model_pair", "dataset", "method", "epoch", "source_kind", "source_path")
    }
    failures = row.get("candidate_failures", [])
    if failures:
        identity["primary_candidate"] = failures[0]
    return identity


def identity_source_and_error(row: dict[str, object]) -> dict[str, object]:
    issue = {
        key: row[key]
        for key in ("model_pair", "dataset", "method", "epoch", "status", "source_path", "error")
    }
    issue["candidates"] = row.get("candidate_failures", [])
    return issue


def collect_rows(
    project_root: Path,
    outputs_root: Path,
    pair_ids: Sequence[str] = FORMAL_PAIRS,
) -> list[dict[str, object]]:
    return [
        load_result(spec, project_root)
        for spec in iter_result_specs(outputs_root, pair_ids)
    ]


def write_reports(
    rows: Sequence[dict[str, object]], output_dir: Path, outputs_root: Path
) -> list[Path]:
    _validate_output_dir(output_dir, outputs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for pair_id in FORMAL_PAIRS:
        path = output_dir / f"llm_judge_results_{pair_id}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(row for row in rows if row["model_pair"] == pair_id)
        paths.append(path)

    audit = {
        "outputs_root": portable_path(outputs_root),
        "total_rows": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "source_kind_counts": dict(sorted(
            Counter(row["source_kind"] for row in rows if row["source_kind"]).items()
        )),
        "summary_fallback_rows": [
            identity_and_source(row)
            for row in rows
            if row["source_kind"] == "summary_fallback"
        ],
        "issues": [
            identity_source_and_error(row)
            for row in rows
            if row["status"] != "ok"
        ],
    }
    audit_path = output_dir / "llm_judge_results_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    paths.append(audit_path)
    return paths


def _resolve_root(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _is_at_or_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_output_dir(output_dir: Path, outputs_root: Path) -> None:
    if _is_at_or_within(output_dir, outputs_root):
        raise ValueError("--output-dir must be outside --outputs-root")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    outputs_root = _resolve_root(args.outputs_root, PROJECT_ROOT)
    output_dir = _resolve_root(args.output_dir, PROJECT_ROOT)
    try:
        _validate_output_dir(output_dir, outputs_root)
    except ValueError as exc:
        parser.error(str(exc))
    rows = collect_rows(PROJECT_ROOT, outputs_root)
    paths = write_reports(rows, output_dir, outputs_root)
    ok_count = sum(row["status"] == "ok" for row in rows)
    fallback_count = sum(row["source_kind"] == "summary_fallback" for row in rows)
    issue_count = len(rows) - ok_count
    print(f"total={len(rows)} ok={ok_count} fallback={fallback_count} issues={issue_count}")
    for path in paths:
        print(portable_path(path))
    return 1 if issue_count and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
