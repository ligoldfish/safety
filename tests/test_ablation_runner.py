from __future__ import annotations

import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.ablations.benchmarks import BenchmarkRequest, DecodeConfig, preflight_benchmark
from src.ablations.catalog import load_catalog
from src.ablations.planner import build_catalog_plan
from src.ablations.runner import (
    AblationRunner,
    RunnerContext,
    RunnerError,
    compile_cell_commands,
    executable_handlers,
    phasef_updates_for_cell,
    validate_completion,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")


def _cell(experiment_id: str, **axes):
    plan = build_catalog_plan(CATALOG, output_root="/persistent/out", scope="all")
    return next(
        cell
        for cell in plan.cells
        if cell.experiment_id == experiment_id
        and all(cell.axes.get(key) == value for key, value in axes.items())
    )


class AblationCompileTests(unittest.TestCase):
    def test_all_catalog_handlers_are_executable(self) -> None:
        self.assertEqual(
            executable_handlers(),
            {definition.handler for definition in CATALOG.experiments.values()},
        )

    def test_representation_strategy_is_forwarded_to_extraction_as_json(self) -> None:
        cell = _cell("P1-08", mode="first_4_generated_mean")
        context = RunnerContext(
            project_root=ROOT,
            state_root=Path("/persistent/state"),
            python_executable="python with spaces",
            device="npu",
            device_id=3,
        )
        command = compile_cell_commands(CATALOG, cell, context)[0]
        self.assertEqual(command.argv[0], "python with spaces")
        stage_flag = next(arg for arg in command.argv if arg.startswith("--phase1-stage-extras="))
        extras = json.loads(stage_flag.split("=", 1)[1])
        self.assertEqual(
            extras["extract"],
            ["--representation-mode", "first_4_generated_mean"],
        )
        self.assertEqual(command.argv[command.argv.index("--device-id") + 1], "3")

    def test_each_core_strategy_reaches_its_own_phase(self) -> None:
        cases = (
            ("P1-03", {"draw": 4}, "subspace", "--subspace-mode", "random_orthogonal"),
            ("P1-04", {"bridge_mode": "ridge"}, "bridge", "--bridge-mode", "ridge"),
            ("P1-07", {"mode": "cka_nearest"}, "pairing", "--pairing-mode", "cka_nearest"),
            ("P1-09", {"top_m": 512}, "decompose", "--top-k", "512"),
        )
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        for experiment_id, axes, phase, flag, expected in cases:
            with self.subTest(experiment_id=experiment_id, axes=axes):
                command = compile_cell_commands(CATALOG, _cell(experiment_id, **axes), context)[0]
                encoded = next(arg for arg in command.argv if arg.startswith("--phase1-stage-extras="))
                extras = json.loads(encoded.split("=", 1)[1])
                self.assertEqual(extras[phase][extras[phase].index(flag) + 1], expected)

    def test_every_train_cell_compiles_only_supported_phasef_fields(self) -> None:
        plan = build_catalog_plan(CATALOG, output_root="/persistent/out", scope="all")
        allowed = {
            "seed",
            "target.mode",
            "target.representation_mode",
            "target.loss_kind",
            "target.layer_loss_policy",
            "target.harmful_layer_weight",
            "target.harmless_layer_weight",
            "optim.layer_loss_weight",
            "lora.rank",
            "lora.placement",
            "inputs.max_samples_per_label",
        }
        for cell in plan.cells:
            if CATALOG.experiments[cell.experiment_id].execution_kind.value != "train":
                continue
            with self.subTest(cell=cell.cell_id, experiment=cell.experiment_id):
                self.assertLessEqual(set(phasef_updates_for_cell(cell)), allowed)

    def test_oneclick_forwards_every_stage_extra_to_the_correct_phase(self) -> None:
        path = ROOT / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("oneclick_ablation_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        calls = []
        module._run_script = lambda script, argv, **kwargs: calls.append((script, list(argv)))
        module._run_phase1_precompute(
            Path("phase one.yaml"),
            smoke=False,
            dry_run=True,
            stage_extras={
                "extract": ["--representation-mode", "mean_prompt"],
                "analyze": ["--selection-mode", "effect_only"],
                "subspace": ["--subspace-mode", "none"],
                "pairing": ["--pairing-mode", "cka_nearest"],
                "bridge": ["--bridge-mode", "ridge"],
                "decompose": ["--top-k", "32"],
                "recompose": ["--bridge-mode", "ridge"],
            },
        )
        by_script = {}
        for script, argv in calls:
            by_script.setdefault(script, []).append(argv)
        self.assertIn("--representation-mode", by_script["01_extract_hidden_states.py"][0])
        self.assertIn("effect_only", by_script["02_analyze_teacher_layers.py"][0])
        self.assertIn("none", by_script["03_build_teacher_safe_subspace.py"][0])
        self.assertIn("cka_nearest", by_script["04_pair_layers.py"][0])
        self.assertIn("ridge", by_script["05_build_semantic_bases.py"][0])
        self.assertIn("32", by_script["07_decompose_teacher_semantics.py"][0])
        self.assertIn("ridge", by_script["08_recompose_student_targets.py"][0])

    def test_benchmark_missing_assets_is_blocked_and_decode_is_shared(self) -> None:
        decode = DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=1024)
        with tempfile.TemporaryDirectory() as td:
            request = BenchmarkRequest(
                name="mmlu",
                asset_path=Path(td) / "missing.jsonl",
                decode=decode,
            )
            report = preflight_benchmark(request)
        self.assertEqual(report.status, "BLOCKED")
        self.assertEqual(report.decode, decode)


class AblationRunnerStateTests(unittest.TestCase):
    def test_completion_validation_rejects_empty_or_invalid_structured_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cell_dir = Path(td)
            artifacts = cell_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "bad.json").write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "invalid JSON"):
                validate_completion(cell_dir, ["bad.json"])
            (artifacts / "bad.json").write_text("{}\n", encoding="utf-8")
            (artifacts / "empty.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "empty"):
                validate_completion(cell_dir, ["bad.json", "empty.jsonl"])

    def test_success_failure_dry_run_and_recovery(self) -> None:
        cell = _cell("P1-11", layer_loss_weight=0.25)
        definition = CATALOG.experiments[cell.experiment_id]
        with tempfile.TemporaryDirectory() as td:
            state_root = Path(td) / "state"
            context = RunnerContext(ROOT, state_root, "python", "npu", 0)
            calls = []

            def success(command, *, cwd, env):
                calls.append((command, cwd, env))
                out = state_root / cell.cell_id / "artifacts"
                out.mkdir(parents=True, exist_ok=True)
                for name in definition.completion_artifacts:
                    (out / name).write_text("{}\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            runner = AblationRunner(CATALOG, context, executor=success)
            dry = runner.run_cell(cell, dry_run=True)
            self.assertNotEqual(dry["state"], "COMPLETED")

            # A dry-run ledger is intentionally immutable as a real run; use a
            # fresh state root to prove a real execution completes.
            context = RunnerContext(ROOT, Path(td) / "real", "python", "npu", 0)
            state_root = context.state_root
            runner = AblationRunner(CATALOG, context, executor=success)
            complete = runner.run_cell(cell)
            self.assertEqual(complete["state"], "COMPLETED")
            self.assertTrue(calls)
            self.assertIsInstance(calls[-1][0], list)

            fail_root = Path(td) / "failure"
            failed_once = {"value": True}

            def flaky(command, *, cwd, env):
                if failed_once["value"]:
                    failed_once["value"] = False
                    return subprocess.CompletedProcess(command, 9)
                out = fail_root / cell.cell_id / "artifacts"
                out.mkdir(parents=True, exist_ok=True)
                for name in definition.completion_artifacts:
                    (out / name).write_text("{}\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            recovery = AblationRunner(
                CATALOG,
                RunnerContext(ROOT, fail_root, "python", "npu", 0),
                executor=flaky,
            )
            with self.assertRaisesRegex(RunnerError, "exit code 9"):
                recovery.run_cell(cell)
            self.assertEqual(recovery.status(cell)["state"], "FAILED")
            self.assertEqual(recovery.run_cell(cell)["state"], "COMPLETED")

    def test_unknown_cell_and_missing_completion_artifacts_fail_closed(self) -> None:
        cell = _cell("P1-13", loss_kind="cosine")
        with tempfile.TemporaryDirectory() as td:
            runner = AblationRunner(
                CATALOG,
                RunnerContext(ROOT, Path(td), "python", "npu", 0),
                executor=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
            )
            with self.assertRaisesRegex(RunnerError, "completion artifact"):
                runner.run_cell(cell)
            self.assertEqual(runner.status(cell)["state"], "FAILED")
            with self.assertRaisesRegex(RunnerError, "unknown cell"):
                runner.select_cell([cell], "not-a-cell")


if __name__ == "__main__":
    unittest.main()
