# LLM Judge Result Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Python collector that emits one uniform LLM-judge CSV per formal model pair and a complete audit JSON without ingesting hyperparameter-search runs.

**Architecture:** A single dependency-free CLI script owns the explicit experiment matrix, deterministic formal path construction, metric loading, and report writing. A focused unittest module imports the script directly and exercises it against temporary synthetic output trees, so no model, accelerator, network, or remote filesystem is needed.

**Tech Stack:** Python 3.10 standard library (`argparse`, `csv`, `dataclasses`, `json`, `pathlib`, `unittest`, `tempfile`) plus the existing `src.pairs` registry.

## Global Constraints

- Include exactly five model pairs: `qwen35_9b_to_08b`, `llama31_8b_to_1b`, `qwen3_8b_to_06b`, `qwen3_8b_to_4b`, and `qwen3_4b_to_06b`.
- Include exactly six datasets: `pan`, `safety_tuned_llamas`, `coconot`, `wildguardmix`, `wildjailbreak`, and `c5`; do not include TuluSafety, BeaverTails, or HH-RLHF.
- Include exactly five methods: `ours`, `ours_sft1`, `sft`, `distill`, and `nosft`.
- Non-STL trained methods emit both `epoch_002` and `epoch_003`.
- STL `ours` and `ours_sft1` emit epochs 2 and 3; STL `sft` and `distill` emit only epoch 6; `nosft` always emits `single`.
- Construct canonical formal paths directly; never recursively discover runs or read `outputs/sweep`, live `LW/TK/DEF/EP5` cells, or other search outputs.
- Read `judge_results.json` first and deterministically fall back to `summary.json` at `results.pan`.
- Preserve stored zero-to-one metric values; do not convert to percentages or select a best epoch.
- Never silently drop an expected row. Missing, malformed, or incomplete results remain in CSV and audit output.
- The collector must not load models, invoke a judge, run training, or modify anything under `outputs`.

---

### Task 1: Formal Matrix and Deterministic Path Resolution

**Files:**
- Create: `scripts/collect_llm_judge_results.py`
- Create: `tests/test_collect_llm_judge_results.py`

**Interfaces:**
- Produces: immutable `ResultSpec(pair_id: str, dataset: str, method: str, epoch: str, result_dir: Path)`.
- Produces: `iter_result_specs(outputs_root: Path, pair_ids: Sequence[str] = FORMAL_PAIRS) -> list[ResultSpec]`.
- Consumes: `src.pairs.PAIRS` and `src.pairs.DEFAULT_PAIR` for model names and student tags.

- [ ] **Step 1: Write failing matrix and path tests**

Create `tests/test_collect_llm_judge_results.py` with a direct script loader and these assertions:

```python
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_collector():
    path = PROJECT_ROOT / "scripts" / "collect_llm_judge_results.py"
    spec = importlib.util.spec_from_file_location("llm_judge_collector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResultSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collector = _load_collector()

    def test_each_pair_has_52_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            for pair_id in self.collector.FORMAL_PAIRS:
                specs = self.collector.iter_result_specs(root, [pair_id])
                self.assertEqual(len(specs), 52)

    def test_canonical_and_extension_paths_are_exact(self) -> None:
        root = Path("/repo/outputs")
        specs = self.collector.iter_result_specs(
            root, ["qwen35_9b_to_08b", "llama31_8b_to_1b"]
        )
        by_key = {(s.pair_id, s.dataset, s.method, s.epoch): s for s in specs}
        canonical = by_key[("qwen35_9b_to_08b", "pan", "ours", "epoch_002")]
        self.assertEqual(
            canonical.result_dir,
            root / "qwen35_9b_to_08b_phase1_npu" / "training"
            / "eval_suite" / "epoch_002",
        )
        extension = by_key[
            ("llama31_8b_to_1b", "safety_tuned_llamas", "distill", "epoch_006")
        ]
        self.assertEqual(
            extension.result_dir,
            root / "baselines"
            / "distill_llama31_8b_to_1b_safety_tuned_llamas_npu"
            / "eval_suite" / "epoch_006",
        )
        nosft = by_key[("llama31_8b_to_1b", "c5", "nosft", "single")]
        self.assertEqual(
            nosft.result_dir,
            root / "baselines" / "llama32_1b_eval_c5_npu",
        )

    def test_stl_epoch_rules_are_method_specific(self) -> None:
        specs = self.collector.iter_result_specs(Path("outputs"), ["qwen35_9b_to_08b"])
        epochs = {}
        for spec in specs:
            if spec.dataset == "safety_tuned_llamas":
                epochs.setdefault(spec.method, []).append(spec.epoch)
        self.assertEqual(epochs["ours"], ["epoch_002", "epoch_003"])
        self.assertEqual(epochs["ours_sft1"], ["epoch_002", "epoch_003"])
        self.assertEqual(epochs["sft"], ["epoch_006"])
        self.assertEqual(epochs["distill"], ["epoch_006"])
        self.assertEqual(epochs["nosft"], ["single"])

    def test_no_constructed_path_can_point_at_search_outputs(self) -> None:
        specs = self.collector.iter_result_specs(Path("outputs"))
        forbidden = ("/sweep/", "LW05", "LW10", "TK2", "DEF_", "EP5")
        for spec in specs:
            normalized = "/" + spec.result_dir.as_posix() + "/"
            self.assertFalse(any(token in normalized for token in forbidden))
```

- [ ] **Step 2: Run the test and verify it fails because the script does not exist**

Run:

```bash
python -m unittest discover -s tests -p "test_collect_llm_judge_results.py" -v
```

Expected: import fails with `FileNotFoundError` for `scripts/collect_llm_judge_results.py`.

- [ ] **Step 3: Implement the matrix and resolver**

Create the script with these constants and interfaces:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


@dataclass(frozen=True)
class ResultSpec:
    pair_id: str
    dataset: str
    method: str
    epoch: str
    result_dir: Path


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
```

At the top of the script, insert the project root into `sys.path` before importing `src.pairs`, matching existing scripts.

- [ ] **Step 4: Run the focused test and verify all path tests pass**

Run the same unittest discovery command. Expected: 4 tests pass.

- [ ] **Step 5: Commit the independently testable resolver**

```bash
git add scripts/collect_llm_judge_results.py tests/test_collect_llm_judge_results.py
git commit -m "feat: define formal llm judge result matrix"
```

---

### Task 2: Judge Metric Loading and Auditable Fallback

**Files:**
- Modify: `scripts/collect_llm_judge_results.py`
- Modify: `tests/test_collect_llm_judge_results.py`

**Interfaces:**
- Consumes: `ResultSpec` from Task 1.
- Produces: `load_result(spec: ResultSpec, project_root: Path) -> dict[str, object]` with stable CSV fields.
- Produces: statuses `ok`, `missing`, `malformed`, or `incomplete`; source kinds `judge_results`, `summary_fallback`, or blank.

- [ ] **Step 1: Add failing metric-loading tests**

Add helpers that write JSON under a temporary `ResultSpec.result_dir`, then test:

```python
import json


CORE = {
    "llm_judge_asr": 0.1,
    "llm_judge_over_refusal": 0.2,
    "llm_judge_refusal_rate": 0.7,
}


class MetricLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collector = _load_collector()

    def _spec(self, root: Path):
        return self.collector.ResultSpec(
            "qwen35_9b_to_08b", "pan", "ours", "epoch_002", root
        )

    def test_judge_results_has_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "outputs" / "run" / "epoch_002"
            result_dir.mkdir(parents=True)
            (result_dir / "judge_results.json").write_text(
                json.dumps({**CORE, "judge_parse_rate": 0.99}), encoding="utf-8"
            )
            (result_dir / "summary.json").write_text(
                json.dumps({"results": {"pan": {**CORE, "llm_judge_asr": 0.9}}}),
                encoding="utf-8",
            )
            row = self.collector.load_result(self._spec(result_dir), Path(tmp))
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["source_kind"], "judge_results")
            self.assertEqual(row["llm_judge_asr"], 0.1)

    def test_incomplete_judge_falls_back_to_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "outputs" / "run"
            result_dir.mkdir(parents=True)
            (result_dir / "judge_results.json").write_text(
                json.dumps({"llm_judge_asr": 0.1}), encoding="utf-8"
            )
            (result_dir / "summary.json").write_text(
                json.dumps({"results": {"pan": CORE}}), encoding="utf-8"
            )
            row = self.collector.load_result(self._spec(result_dir), Path(tmp))
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["source_kind"], "summary_fallback")

    def test_missing_result_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "absent"
            row = self.collector.load_result(self._spec(result_dir), Path(tmp))
            self.assertEqual(row["status"], "missing")
            self.assertEqual(row["source_kind"], "")
            self.assertEqual(row["llm_judge_asr"], "")

    def test_malformed_and_incomplete_results_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed"
            malformed.mkdir()
            (malformed / "judge_results.json").write_text("{", encoding="utf-8")
            bad = self.collector.load_result(self._spec(malformed), root)
            self.assertEqual(bad["status"], "malformed")

            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "judge_results.json").write_text(
                json.dumps({"llm_judge_asr": 0.1}), encoding="utf-8"
            )
            partial = self.collector.load_result(self._spec(incomplete), root)
            self.assertEqual(partial["status"], "incomplete")
```

- [ ] **Step 2: Run tests and verify the new tests fail with missing `load_result`**

Run focused unittest discovery. Expected: the four Task 1 tests pass and four new tests error with `AttributeError`.

- [ ] **Step 3: Implement strict payload reading and row creation**

Add constants for the core and optional metric fields and implement:

```python
CORE_METRICS = (
    "llm_judge_asr",
    "llm_judge_over_refusal",
    "llm_judge_refusal_rate",
)
OPTIONAL_METRICS = (
    "judge_keyword_agreement",
    "judge_cohen_kappa",
    "judge_parse_rate",
    "judge_num_harmful_scored",
    "judge_num_harmless_scored",
)


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_candidate(path: Path, summary: bool) -> tuple[dict[str, object] | None, str, str]:
    if not path.is_file():
        return None, "missing", f"file not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, "malformed", f"cannot read JSON: {exc}"
    if summary:
        payload = payload.get("results", {}).get("pan", {}) if isinstance(payload, dict) else {}
    if not isinstance(payload, dict):
        return None, "malformed", "metric payload is not an object"
    missing = [key for key in CORE_METRICS if key not in payload]
    invalid = [key for key in CORE_METRICS if key in payload and not _valid_number(payload[key])]
    optional_invalid = [
        key for key in OPTIONAL_METRICS
        if key in payload and payload[key] is not None and not _valid_number(payload[key])
    ]
    if missing:
        return None, "incomplete", f"missing metrics: {', '.join(missing)}"
    if invalid or optional_invalid:
        return None, "malformed", f"non-numeric metrics: {', '.join(invalid + optional_invalid)}"
    return payload, "ok", ""
```

`load_result` builds identity fields from `PAIRS[spec.pair_id]`, tries the judge candidate then summary candidate, copies scalar metrics on success, records a portable source path, and chooses a final failure status using precedence `malformed`, `incomplete`, then `missing`. Include an `error` field for audit details but omit it from CSV fields.

- [ ] **Step 4: Run focused tests and verify all eight tests pass**

Run focused unittest discovery. Expected: 8 tests pass without importing Torch.

- [ ] **Step 5: Commit metric loading**

```bash
git add scripts/collect_llm_judge_results.py tests/test_collect_llm_judge_results.py
git commit -m "feat: load llm judge metrics with audited fallback"
```

---

### Task 3: Per-Pair CSVs, Audit JSON, and CLI

**Files:**
- Modify: `scripts/collect_llm_judge_results.py`
- Modify: `tests/test_collect_llm_judge_results.py`

**Interfaces:**
- Consumes: rows returned by `load_result` in Task 2.
- Produces: `collect_rows(project_root: Path, outputs_root: Path, pair_ids: Sequence[str] = FORMAL_PAIRS) -> list[dict[str, object]]`.
- Produces: `write_reports(rows: Sequence[dict[str, object]], output_dir: Path, outputs_root: Path) -> list[Path]`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Add failing report and CLI tests**

Add tests that create one good row and one missing row, then assert stable files and exit behavior:

```python
import csv


class ReportWritingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collector = _load_collector()

    def test_writes_one_csv_per_pair_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for pair_id in self.collector.FORMAL_PAIRS:
                spec = self.collector.ResultSpec(
                    pair_id, "pan", "nosft", "single", root / pair_id
                )
                rows.append(self.collector.load_result(spec, root))
            paths = self.collector.write_reports(rows, root / "reports", root / "outputs")
            self.assertEqual(len(paths), 6)
            for pair_id in self.collector.FORMAL_PAIRS:
                csv_path = root / "reports" / f"llm_judge_results_{pair_id}.csv"
                self.assertTrue(csv_path.is_file())
                with csv_path.open(encoding="utf-8", newline="") as handle:
                    records = list(csv.DictReader(handle))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["status"], "missing")
            audit = json.loads(
                (root / "reports" / "llm_judge_results_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["status_counts"], {"missing": 5})
            self.assertEqual(len(audit["issues"]), 5)

    def test_summary_fallback_is_listed_in_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "run"
            result_dir.mkdir()
            (result_dir / "summary.json").write_text(
                json.dumps({"results": {"pan": CORE}}), encoding="utf-8"
            )
            spec = self.collector.ResultSpec(
                "qwen35_9b_to_08b", "pan", "ours", "epoch_002", result_dir
            )
            row = self.collector.load_result(spec, root)
            self.collector.write_reports([row], root / "reports", root / "outputs")
            audit = json.loads(
                (root / "reports" / "llm_judge_results_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(audit["summary_fallback_rows"]), 1)

    def test_main_exit_code_respects_allow_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ["--outputs-root", str(root / "outputs"), "--output-dir", str(root / "reports")]
            self.assertEqual(self.collector.main(common), 1)
            self.assertEqual(self.collector.main([*common, "--allow-missing"]), 0)
```

- [ ] **Step 2: Run focused tests and verify failures for missing report functions**

Run focused unittest discovery. Expected: prior tests pass; the three new tests fail with missing `write_reports` or `main`.

- [ ] **Step 3: Implement deterministic report writing and CLI**

Use this stable CSV schema:

```python
CSV_FIELDS = (
    "model_pair", "teacher_model", "student_model", "dataset", "method",
    "epoch", "status", "source_kind", *CORE_METRICS, *OPTIONAL_METRICS,
    "source_path",
)
```

Implement `collect_rows` by iterating `iter_result_specs` and calling `load_result`. Implement `write_reports` with `csv.DictWriter(extrasaction="ignore")`, always creating all five pair CSVs in `FORMAL_PAIRS` order. The deterministic audit payload must contain:

```python
audit = {
    "outputs_root": portable_path(outputs_root),
    "total_rows": len(rows),
    "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    "source_kind_counts": dict(sorted(Counter(row["source_kind"] for row in rows if row["source_kind"]).items())),
    "summary_fallback_rows": [identity_and_source(row) for row in rows if row["source_kind"] == "summary_fallback"],
    "issues": [identity_source_and_error(row) for row in rows if row["status"] != "ok"],
}
```

Implement argparse options `--outputs-root`, `--output-dir`, and `--allow-missing`. Relative roots resolve against the project root, defaults are `<project>/outputs` and `<project>`. `main` writes reports, prints total/ok/fallback/issue counts and each output path, then returns 1 only when issues exist and `--allow-missing` is false. End with `raise SystemExit(main())`.

- [ ] **Step 4: Run focused tests and verify all tests pass**

Run focused unittest discovery. Expected: 11 tests pass.

- [ ] **Step 5: Run broader regression and static checks**

Run:

```bash
python -m unittest discover -s tests -p "test_collect_llm_judge_results.py" -v
python -m py_compile scripts/collect_llm_judge_results.py tests/test_collect_llm_judge_results.py
python scripts/collect_llm_judge_results.py --outputs-root D:/output --output-dir .tmp/llm_judge_report --allow-missing
git diff --check
```

Expected:

- All collector tests pass.
- Both Python files compile.
- The local `D:/output` smoke run writes five CSVs and one audit file without reading sweep results; missing extension-pair cells appear explicitly in the audit.
- `git diff --check` reports no whitespace errors.

- [ ] **Step 6: Inspect smoke output invariants**

Confirm each pair CSV contains only the six allowed datasets, five allowed methods, requested epochs, and formal source paths. Confirm `llm_judge_results_audit.json` reports all missing/fallback rows and no source path contains `/sweep/`, `LW`, `TK`, `DEF`, or `EP5`.

- [ ] **Step 7: Commit the complete collector**

```bash
git add scripts/collect_llm_judge_results.py tests/test_collect_llm_judge_results.py
git commit -m "feat: collect formal llm judge results"
```
