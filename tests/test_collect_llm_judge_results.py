from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import csv
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
