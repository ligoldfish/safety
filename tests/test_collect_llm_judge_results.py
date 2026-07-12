from __future__ import annotations

import csv
import importlib.util
import json
import math
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


CORE = {
    "llm_judge_asr": 0.1,
    "llm_judge_over_refusal": 0.2,
    "llm_judge_refusal_rate": 0.7,
}


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
        expected = {
            ("qwen35_9b_to_08b", "pan", "ours", "epoch_002"):
                "qwen35_9b_to_08b_phase1_npu/training/eval_suite/epoch_002",
            ("qwen35_9b_to_08b", "pan", "ours_sft1", "epoch_003"):
                "qwen35_9b_to_08b_phase1_npu/training_sft1/eval_suite/epoch_003",
            ("qwen35_9b_to_08b", "pan", "sft", "epoch_002"):
                "baselines/sft_qwen35_08b_npu/eval_suite/epoch_002",
            ("qwen35_9b_to_08b", "pan", "distill", "epoch_003"):
                "baselines/distill_qwen35_9b_to_08b_npu/eval_suite/epoch_003",
            ("qwen35_9b_to_08b", "pan", "nosft", "single"):
                "baselines/no_sft_qwen35_08b_npu",
            ("qwen35_9b_to_08b", "c5", "ours", "epoch_002"):
                "safety_full_c5_npu/phase1/training/eval_suite/epoch_002",
            ("qwen35_9b_to_08b", "coconot", "ours_sft1", "epoch_003"):
                "safety_full_coconot_npu/phase1/training_sft1/eval_suite/epoch_003",
            ("qwen35_9b_to_08b", "wildguardmix", "sft", "epoch_002"):
                "baselines/sft_qwen35_08b_wildguardmix_npu/eval_suite/epoch_002",
            ("qwen35_9b_to_08b", "safety_tuned_llamas", "distill", "epoch_006"):
                "baselines/distill_qwen35_9b_to_08b_safety_tuned_llamas_npu/"
                "eval_suite/epoch_006",
            ("qwen35_9b_to_08b", "wildjailbreak", "nosft", "single"):
                "baselines/eval_wildjailbreak_npu",
            ("llama31_8b_to_1b", "pan", "ours", "epoch_002"):
                "llama31_8b_to_1b_phase1_npu/training/eval_suite/epoch_002",
            ("llama31_8b_to_1b", "pan", "ours_sft1", "epoch_003"):
                "llama31_8b_to_1b_phase1_npu/training_sft1/eval_suite/epoch_003",
            ("llama31_8b_to_1b", "pan", "sft", "epoch_002"):
                "baselines/sft_llama32_1b_npu/eval_suite/epoch_002",
            ("llama31_8b_to_1b", "pan", "distill", "epoch_003"):
                "baselines/distill_llama31_8b_to_1b_npu/eval_suite/epoch_003",
            ("llama31_8b_to_1b", "pan", "nosft", "single"):
                "baselines/no_sft_llama32_1b_npu",
            ("llama31_8b_to_1b", "c5", "ours", "epoch_002"):
                "safety_full_c5_npu_llama31_8b_to_1b/phase1/training/"
                "eval_suite/epoch_002",
            ("llama31_8b_to_1b", "coconot", "ours_sft1", "epoch_003"):
                "safety_full_coconot_npu_llama31_8b_to_1b/phase1/training_sft1/"
                "eval_suite/epoch_003",
            ("llama31_8b_to_1b", "wildguardmix", "sft", "epoch_002"):
                "baselines/sft_llama32_1b_wildguardmix_npu/eval_suite/epoch_002",
            ("llama31_8b_to_1b", "safety_tuned_llamas", "distill", "epoch_006"):
                "baselines/distill_llama31_8b_to_1b_safety_tuned_llamas_npu/"
                "eval_suite/epoch_006",
            ("llama31_8b_to_1b", "c5", "nosft", "single"):
                "baselines/llama32_1b_eval_c5_npu",
            ("qwen3_8b_to_06b", "pan", "nosft", "single"):
                "baselines/no_sft_qwen3_06b_npu",
            ("qwen3_4b_to_06b", "pan", "nosft", "single"):
                "baselines/no_sft_qwen3_06b_npu",
            ("qwen3_8b_to_06b", "c5", "nosft", "single"):
                "baselines/qwen3_06b_eval_c5_npu",
            ("qwen3_4b_to_06b", "c5", "nosft", "single"):
                "baselines/qwen3_06b_eval_c5_npu",
        }
        pair_ids = list(dict.fromkeys(key[0] for key in expected))
        specs = self.collector.iter_result_specs(root, pair_ids)
        by_key = {(s.pair_id, s.dataset, s.method, s.epoch): s for s in specs}
        for key, relative_path in expected.items():
            with self.subTest(key=key):
                self.assertEqual(by_key[key].result_dir, root / relative_path)

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

    def test_external_source_path_is_portable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            result_dir = root / "external_outputs" / "run"
            result_dir.mkdir(parents=True)
            source_path = result_dir / "judge_results.json"
            source_path.write_text(json.dumps(CORE), encoding="utf-8")

            row = self.collector.load_result(self._spec(result_dir), project_root)

            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["source_path"], source_path.as_posix())

    def test_rejects_nonfinite_and_out_of_domain_metrics(self) -> None:
        cases = (
            ("asr_nan", "llm_judge_asr", math.nan),
            ("asr_below", "llm_judge_asr", -0.01),
            ("asr_above", "llm_judge_asr", 1.01),
            ("over_refusal_inf", "llm_judge_over_refusal", math.inf),
            ("over_refusal_neg_inf", "llm_judge_over_refusal", -math.inf),
            ("refusal_below", "llm_judge_refusal_rate", -0.01),
            ("refusal_above", "llm_judge_refusal_rate", 1.01),
            ("agreement_nan", "judge_keyword_agreement", math.nan),
            ("agreement_below", "judge_keyword_agreement", -0.01),
            ("agreement_above", "judge_keyword_agreement", 1.01),
            ("parse_inf", "judge_parse_rate", math.inf),
            ("parse_below", "judge_parse_rate", -0.01),
            ("parse_above", "judge_parse_rate", 1.01),
            ("kappa_nan", "judge_cohen_kappa", math.nan),
            ("kappa_below", "judge_cohen_kappa", -1.01),
            ("kappa_above", "judge_cohen_kappa", 1.01),
            ("harmful_negative", "judge_num_harmful_scored", -1),
            ("harmful_fraction", "judge_num_harmful_scored", 1.5),
            ("harmful_bool", "judge_num_harmful_scored", True),
            ("harmless_negative", "judge_num_harmless_scored", -1),
            ("harmless_fraction", "judge_num_harmless_scored", 1.5),
            ("harmless_bool", "judge_num_harmless_scored", False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, metric, value in cases:
                with self.subTest(metric=metric, value=value):
                    result_dir = root / name
                    result_dir.mkdir()
                    (result_dir / "judge_results.json").write_text(
                        json.dumps({**CORE, metric: value}), encoding="utf-8"
                    )
                    row = self.collector.load_result(self._spec(result_dir), root)
                    self.assertEqual(row["status"], "malformed")
                    self.assertIn(metric, row["error"])

    def test_accepts_metric_domain_boundaries_and_integer_counts(self) -> None:
        payloads = (
            {
                "llm_judge_asr": 0,
                "llm_judge_over_refusal": 0,
                "llm_judge_refusal_rate": 0,
                "judge_keyword_agreement": 0,
                "judge_parse_rate": 0,
                "judge_cohen_kappa": -1,
                "judge_num_harmful_scored": 0,
                "judge_num_harmless_scored": 0,
            },
            {
                "llm_judge_asr": 1,
                "llm_judge_over_refusal": 1,
                "llm_judge_refusal_rate": 1,
                "judge_keyword_agreement": 1,
                "judge_parse_rate": 1,
                "judge_cohen_kappa": 1,
                "judge_num_harmful_scored": 2,
                "judge_num_harmless_scored": 3,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, payload in enumerate(payloads):
                result_dir = root / f"boundary_{index}"
                result_dir.mkdir()
                (result_dir / "judge_results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                row = self.collector.load_result(self._spec(result_dir), root)
                self.assertEqual(row["status"], "ok")

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

    def test_malformed_summary_structures_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed_payloads = (
                {"results": None},
                {"results": []},
                {"results": "bad"},
                [],
            )
            for index, payload in enumerate(malformed_payloads):
                result_dir = root / f"malformed_{index}"
                result_dir.mkdir()
                (result_dir / "summary.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                row = self.collector.load_result(self._spec(result_dir), root)
                self.assertEqual(row["status"], "malformed")
                self.assertEqual(row["source_kind"], "")


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

    def test_summary_fallback_audit_describes_primary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                ("missing", None, "file not found"),
                (
                    "malformed",
                    {**CORE, "judge_parse_rate": math.inf},
                    "invalid metrics: judge_parse_rate must be finite and within [0,1]",
                ),
                (
                    "incomplete",
                    {"llm_judge_asr": 0.1},
                    "missing metrics: llm_judge_over_refusal, llm_judge_refusal_rate",
                ),
            )
            rows = []
            expected_primary = []
            for status, primary_payload, error in cases:
                result_dir = root / f"run_{status}"
                result_dir.mkdir()
                if primary_payload is not None:
                    (result_dir / "judge_results.json").write_text(
                        json.dumps(primary_payload), encoding="utf-8"
                    )
                (result_dir / "summary.json").write_text(
                    json.dumps({"results": {"pan": CORE}}), encoding="utf-8"
                )
                spec = self.collector.ResultSpec(
                    "qwen35_9b_to_08b", "pan", "ours", "epoch_002", result_dir
                )
                row = self.collector.load_result(spec, root)
                self.assertEqual(row["source_kind"], "summary_fallback")
                rows.append(row)
                expected_primary.append({
                    "kind": "judge_results",
                    "status": status,
                    "path": f"run_{status}/judge_results.json",
                    "error": error,
                })

            self.collector.write_reports(rows, root / "reports", root / "outputs")
            audit = json.loads(
                (root / "reports" / "llm_judge_results_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["primary_candidate"] for entry in audit["summary_fallback_rows"]],
                expected_primary,
            )
            csv_path = root / "reports" / "llm_judge_results_qwen35_9b_to_08b.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                records = list(reader)
            self.assertEqual(reader.fieldnames, list(self.collector.CSV_FIELDS))
            self.assertEqual(len(records), 3)

    def test_issue_audit_has_portable_candidate_paths_and_path_neutral_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            result_dirs = (project_root / "inside", root / "external")
            rows = []
            for result_dir in result_dirs:
                spec = self.collector.ResultSpec(
                    "qwen35_9b_to_08b", "pan", "ours", "epoch_002", result_dir
                )
                rows.append(self.collector.load_result(spec, project_root))

            self.collector.write_reports(rows, root / "reports", root / "outputs")
            audit = json.loads(
                (root / "reports" / "llm_judge_results_audit.json").read_text(encoding="utf-8")
            )
            expected_paths = (
                ("inside/judge_results.json", "inside/summary.json"),
                (
                    (root / "external" / "judge_results.json").as_posix(),
                    (root / "external" / "summary.json").as_posix(),
                ),
            )
            for issue, paths in zip(audit["issues"], expected_paths):
                self.assertEqual(issue["error"], "judge_results: file not found; summary: file not found")
                self.assertEqual(
                    issue["candidates"],
                    [
                        {
                            "kind": "judge_results",
                            "status": "missing",
                            "path": paths[0],
                            "error": "file not found",
                        },
                        {
                            "kind": "summary",
                            "status": "missing",
                            "path": paths[1],
                            "error": "file not found",
                        },
                    ],
                )
                self.assertNotIn(str(root), issue["error"])

    def test_csv_header_and_row_order_match_formal_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_root = root / "outputs"
            pair_id = "qwen35_9b_to_08b"
            specs = self.collector.iter_result_specs(outputs_root, [pair_id])
            rows = self.collector.collect_rows(root, outputs_root, [pair_id])
            self.collector.write_reports(rows, root / "reports", outputs_root)

            csv_path = root / "reports" / f"llm_judge_results_{pair_id}.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                records = list(reader)
            self.assertEqual(reader.fieldnames, list(self.collector.CSV_FIELDS))
            expected_identity = [
                (spec.pair_id, spec.dataset, spec.method, spec.epoch)
                for spec in specs
            ]
            actual_identity = [
                (row["model_pair"], row["dataset"], row["method"], row["epoch"])
                for row in records
            ]
            self.assertEqual(actual_identity, expected_identity)

    def test_main_exit_code_respects_allow_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ["--outputs-root", str(root / "outputs"), "--output-dir", str(root / "reports")]
            self.assertEqual(self.collector.main(common), 1)
            self.assertEqual(self.collector.main([*common, "--allow-missing"]), 0)

    def test_main_rejects_output_dir_at_or_inside_outputs_root_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_root = root / "outputs"
            outputs_root.mkdir()
            for output_dir in (outputs_root, outputs_root / "reports"):
                with self.assertRaises(SystemExit) as raised:
                    self.collector.main([
                        "--outputs-root", str(outputs_root),
                        "--output-dir", str(output_dir),
                    ])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(list(outputs_root.glob("llm_judge_results_*")))
                self.assertFalse((outputs_root / "reports").exists())

    def test_write_reports_rejects_output_dir_at_or_inside_outputs_root_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_root = root / "outputs"
            outputs_root.mkdir()
            for output_dir in (outputs_root, outputs_root / "reports"):
                with self.assertRaises(ValueError):
                    self.collector.write_reports([], output_dir, outputs_root)
                self.assertFalse(list(outputs_root.glob("llm_judge_results_*")))
                self.assertFalse((outputs_root / "reports").exists())
