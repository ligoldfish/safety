from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    if not path.is_file():
        raise AssertionError(f"missing script: {path}")
    module_name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FormalTargetTests(unittest.TestCase):
    def test_targets_cover_exact_formal_matrix_without_search_paths(self) -> None:
        module = _load_script("formal_llm_judge_targets.py")
        targets = module.iter_formal_targets(Path("outputs"))

        self.assertEqual(len(targets), 260)
        identities = {
            (target.pair_id, target.dataset, target.method, target.epoch)
            for target in targets
        }
        self.assertEqual(len(identities), 260)
        forbidden = ("/sweep/", "/sweep_runs/", "_LW05_", "_TK2_", "_DEF_")
        for target in targets:
            normalized = f"/{target.pan_results_path.as_posix()}/"
            self.assertFalse(any(token in normalized for token in forbidden))

    def test_target_paths_expose_queue_owner_and_weight_free_run_root(self) -> None:
        module = _load_script("formal_llm_judge_targets.py")
        targets = module.iter_formal_targets(
            Path("/repo/outputs"), ["qwen35_9b_to_08b"]
        )
        by_key = {
            (target.dataset, target.method, target.epoch): target for target in targets
        }

        trained = by_key[("c5", "ours", "epoch_002")]
        self.assertEqual(trained.kind, "suite")
        self.assertEqual(
            trained.owner_path,
            Path("/repo/outputs/safety_full_c5_npu/phase1/training/eval_suite"),
        )
        self.assertEqual(
            trained.run_root,
            Path("/repo/outputs/safety_full_c5_npu/phase1/training"),
        )
        self.assertEqual(
            trained.pan_results_path,
            trained.owner_path / "epoch_002" / "pan_results.json",
        )

        standalone = by_key[("c5", "nosft", "single")]
        self.assertEqual(standalone.kind, "single")
        self.assertEqual(standalone.owner_path, standalone.pan_results_path)
        self.assertEqual(
            standalone.run_root, Path("/repo/outputs/baselines/eval_c5_npu")
        )

    def test_complete_judge_rejects_zero_item_and_invalid_core_payloads(self) -> None:
        module = _load_script("formal_llm_judge_targets.py")
        core = {
            "llm_judge_asr": 0.1,
            "llm_judge_over_refusal": 0.2,
            "llm_judge_refusal_rate": 0.7,
        }
        self.assertTrue(module.judge_payload_is_complete(core))
        self.assertTrue(
            module.judge_payload_is_complete(
                {**core, "judge_num_items": 10, "judge_parse_rate": 0.9}
            )
        )
        self.assertFalse(
            module.judge_payload_is_complete(
                {
                    **core,
                    "judge_num_items": 0,
                    "judge_num_harmful_scored": 0,
                    "judge_num_harmless_scored": 0,
                    "judge_parse_rate": 0.0,
                }
            )
        )
        self.assertFalse(module.judge_payload_is_complete({**core, "llm_judge_asr": 2}))


class PackageTests(unittest.TestCase):
    def _target(self, module, outputs_root: Path):
        run_root = outputs_root / "formal_run" / "training"
        suite = run_root / "eval_suite"
        return module.FormalJudgeTarget(
            pair_id="qwen35_9b_to_08b",
            dataset="pan",
            method="ours",
            epoch="epoch_002",
            kind="suite",
            owner_path=suite,
            pan_results_path=suite / "epoch_002" / "pan_results.json",
            run_root=run_root,
        )

    def test_package_plan_preserves_formal_tree_and_excludes_model_files(self) -> None:
        targets_module = _load_script("formal_llm_judge_targets.py")
        package_module = _load_script("package_llm_judge_artifacts.py")
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            outputs_root = project_root / "outputs"
            target = self._target(targets_module, outputs_root)
            epoch = target.pan_results_path.parent
            epoch.mkdir(parents=True)
            (epoch / "pan_results.json").write_text("{}", encoding="utf-8")
            (epoch / "judge_results.json").write_text("{}", encoding="utf-8")
            (target.run_root / "manifest.json").write_text("{}", encoding="utf-8")
            checkpoint = target.run_root / "checkpoints"
            checkpoint.mkdir()
            (checkpoint / "epoch_002.pt").write_bytes(b"weights")
            (target.run_root / "model.safetensors").write_bytes(b"weights")
            sweep = outputs_root / "sweep" / "bad"
            sweep.mkdir(parents=True)
            (sweep / "judge_results.json").write_text("{}", encoding="utf-8")

            plan = package_module.build_package_plan(
                project_root,
                outputs_root,
                [target],
                include_eval_data=False,
                include_auxiliary=False,
            )
            members = {member.archive_path.as_posix() for member in plan.members}

            self.assertIn(
                "outputs/formal_run/training/eval_suite/epoch_002/pan_results.json",
                members,
            )
            self.assertIn("outputs/formal_run/training/manifest.json", members)
            self.assertFalse(any("checkpoints" in member for member in members))
            self.assertFalse(any(member.endswith(".safetensors") for member in members))
            self.assertFalse(any("/sweep/" in f"/{member}/" for member in members))
            self.assertEqual(
                [item["pan_results_path"] for item in plan.missing_expected_pan_results],
                [],
            )

    def test_archive_contains_manifest_and_sidecar_with_missing_targets(self) -> None:
        targets_module = _load_script("formal_llm_judge_targets.py")
        package_module = _load_script("package_llm_judge_artifacts.py")
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            outputs_root = project_root / "outputs"
            target = self._target(targets_module, outputs_root)
            target.run_root.mkdir(parents=True)
            (target.run_root / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
            plan = package_module.build_package_plan(
                project_root,
                outputs_root,
                [target],
                include_eval_data=False,
                include_auxiliary=False,
            )
            archive = project_root / "bundle.tar.gz"
            manifest_path = package_module.write_package(plan, archive)

            with tarfile.open(archive, "r:gz") as handle:
                names = set(handle.getnames())
                embedded = json.load(handle.extractfile("llm_judge_package_manifest.json"))
            sidecar = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertIn("outputs/formal_run/training/train_metrics.jsonl", names)
            self.assertIn("llm_judge_package_manifest.json", names)
            self.assertEqual(embedded["missing_expected_pan_results"], sidecar["missing_expected_pan_results"])
            self.assertEqual(len(sidecar["missing_expected_pan_results"]), 1)

    def test_package_keeps_and_fingerprints_both_eval_dataset_copies(self) -> None:
        targets_module = _load_script("formal_llm_judge_targets.py")
        package_module = _load_script("package_llm_judge_artifacts.py")
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            outputs_root = project_root / "outputs"
            run_root = outputs_root / "formal_wjb" / "training"
            suite = run_root / "eval_suite"
            target = targets_module.FormalJudgeTarget(
                pair_id="qwen35_9b_to_08b",
                dataset="wildjailbreak",
                method="ours",
                epoch="epoch_002",
                kind="suite",
                owner_path=suite,
                pan_results_path=suite / "epoch_002" / "pan_results.json",
                run_root=run_root,
            )
            run_root.mkdir(parents=True)
            primary = project_root / "data" / "processed" / "eval" / "wildjailbreak_test.jsonl"
            fallback = (
                project_root
                / "data"
                / "processed"
                / "processed"
                / "eval"
                / "wildjailbreak_test.jsonl"
            )
            primary.parent.mkdir(parents=True)
            fallback.parent.mkdir(parents=True)
            primary.write_text('{"id":"new"}\n', encoding="utf-8")
            fallback.write_text('{"id":"old"}\n', encoding="utf-8")

            plan = package_module.build_package_plan(
                project_root,
                outputs_root,
                [target],
                include_eval_data=True,
                include_auxiliary=False,
            )
            members = {member.archive_path.as_posix() for member in plan.members}

            self.assertIn("data/processed/eval/wildjailbreak_test.jsonl", members)
            self.assertIn(
                "data/processed/processed/eval/wildjailbreak_test.jsonl", members
            )
            self.assertEqual(len(plan.eval_data_fingerprints), 2)
            self.assertEqual(
                len({item["sha256"] for item in plan.eval_data_fingerprints}), 2
            )


class BackfillScriptTests(unittest.TestCase):
    def test_backfill_defaults_to_the_formal_manifest(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "judge_backfill.sh").read_text(encoding="utf-8")
        self.assertIn('FORMAL_ONLY="${FORMAL_ONLY:-1}"', text)
        self.assertIn("formal_llm_judge_targets.py", text)


class JudgeGenerationGuardTests(unittest.TestCase):
    def test_successful_judge_records_test_jsonl_fingerprint(self) -> None:
        module = _load_script("22_judge_generations.py")
        from src.eval.llm_judge import parse_wildguard_output

        class FakeJudge:
            model_path = Path("fake-wildguard")

            def score(self, pairs):
                return [
                    parse_wildguard_output(
                        "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
                    )
                    for _ in pairs
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pan_results = root / "pan_results.json"
            test_jsonl = root / "test.jsonl"
            output = root / "judge_results.json"
            pan_results.write_text(
                json.dumps({
                    "generations": [
                        {
                            "id": "id-1",
                            "label": "harmful",
                            "final_text": "response",
                        }
                    ]
                }),
                encoding="utf-8",
            )
            test_bytes = b'{"id":"id-1"}\n'
            test_jsonl.write_bytes(test_bytes)

            module._judge_one(
                pan_results,
                {"id-1": "prompt"},
                FakeJudge(),
                output,
                False,
                test_jsonl_path=test_jsonl,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["test_jsonl"], str(test_jsonl))
            self.assertEqual(
                payload["test_jsonl_sha256"], hashlib.sha256(test_bytes).hexdigest()
            )

    def test_unmatched_prompt_ids_fail_before_writing_or_scoring(self) -> None:
        module = _load_script("22_judge_generations.py")
        self.assertIn("allow_unmatched", inspect.signature(module._judge_one).parameters)

        class FakeJudge:
            model_path = Path("fake-wildguard")
            called = False

            def score(self, pairs):
                self.called = True
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pan_results = root / "pan_results.json"
            output = root / "judge_results.json"
            pan_results.write_text(
                json.dumps({
                    "generations": [
                        {
                            "id": "missing-id",
                            "label": "harmful",
                            "final_text": "response",
                        }
                    ]
                }),
                encoding="utf-8",
            )
            judge = FakeJudge()

            with self.assertRaisesRegex(ValueError, "unmatched"):
                module._judge_one(
                    pan_results,
                    {"different-id": "prompt"},
                    judge,
                    output,
                    False,
                    allow_unmatched=False,
                )

            self.assertFalse(judge.called)
            self.assertFalse(output.exists())

    def test_zero_parse_rate_fails_without_overwriting_output(self) -> None:
        module = _load_script("22_judge_generations.py")
        from src.eval.llm_judge import parse_wildguard_output

        class FakeJudge:
            model_path = Path("fake-wildguard")

            def score(self, pairs):
                return [parse_wildguard_output("not a WildGuard verdict")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pan_results = root / "pan_results.json"
            output = root / "judge_results.json"
            pan_results.write_text(
                json.dumps({
                    "generations": [
                        {
                            "id": "id-1",
                            "label": "harmful",
                            "final_text": "response",
                        }
                    ]
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "parsed zero"):
                module._judge_one(
                    pan_results,
                    {"id-1": "prompt"},
                    FakeJudge(),
                    output,
                    False,
                    allow_unmatched=False,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
