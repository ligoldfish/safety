from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import torch

from src.ablations.evaluation import EvaluationError, collect_evaluation_result, prepare_evaluation
from src.phase_c.intervention import load_intervention_artifact
from src.utils.config import load_phasec_config


ROOT = Path(__file__).resolve().parents[1]


class AblationEvaluationTests(unittest.TestCase):
    @staticmethod
    def _complete_model(path: Path) -> None:
        path.mkdir()
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")

    def test_decoding_robustness_stages_exact_decode_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "manifest.json").write_text("{}\n", encoding="utf-8")
            (checkpoints / "epoch_002.pt").write_bytes(b"checkpoint")
            data = root / "eval.jsonl"
            data.write_text(json.dumps({"id": "x", "label": "harmful", "messages": []}) + "\n", encoding="utf-8")
            plan = prepare_evaluation(
                "decoding_robustness",
                {"axes": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 1024}, "inputs": {"trained_checkpoints": str(checkpoints), "evaluation_data": str(data)}},
                output_dir=root / "out",
                project_root=ROOT,
                python_executable="python",
                device="npu",
                device_id=2,
            )
            staged = yaml.safe_load(plan.config_path.read_text(encoding="utf-8"))
        self.assertEqual(staged["datasets"]["pan"]["temperature"], 0.7)
        self.assertEqual(staged["datasets"]["pan"]["top_p"], 0.9)
        self.assertEqual(staged["datasets"]["pan"]["max_new_tokens"], 1024)
        self.assertIn("--adapter-checkpoint", plan.argv)

    def test_general_capability_uses_only_the_declared_opencompass_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "merged-model"
            self._complete_model(model)
            assets = root / "opencompass"
            assets.mkdir()
            (assets / "run.py").write_text("pass\n", encoding="utf-8")
            plan = prepare_evaluation(
                "general_capability_suite",
                {"axes": {"benchmark": "ifeval"}, "inputs": {"trained_checkpoints": str(model), "benchmark_assets": str(assets)}},
                output_dir=root / "out",
                project_root=ROOT,
                python_executable="python",
                device="npu",
                device_id=0,
            )
        self.assertEqual(plan.argv[plan.argv.index("--datasets") + 1], "IFEval_gen")
        self.assertEqual(plan.artifact_name, "general_capability.json")

    def test_general_capability_rejects_incomplete_hf_model_and_opencompass_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            assets = root / "opencompass"
            assets.mkdir()
            cases = (
                ("model", model, assets),
                ("OpenCompass", self._make_model(root / "complete"), assets),
            )
            for label, checkpoint, opencompass in cases:
                with self.subTest(label=label), self.assertRaisesRegex(EvaluationError, label):
                    prepare_evaluation(
                        "general_capability_suite",
                        {"axes": {"benchmark": "mmlu"}, "inputs": {"trained_checkpoints": str(checkpoint), "benchmark_assets": str(opencompass)}},
                        output_dir=root / f"out-{label}",
                        project_root=ROOT,
                        python_executable="python",
                        device="npu",
                        device_id=0,
                    )

    def _make_model(self, path: Path) -> Path:
        self._complete_model(path)
        return path

    def test_decoding_merged_model_rejects_missing_tokenizer_or_weights(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "eval.jsonl"
            data.write_text("{}\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "model"):
                prepare_evaluation(
                    "decoding_robustness",
                    {"axes": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 256}, "inputs": {"trained_checkpoints": str(model), "evaluation_data": str(data)}},
                    output_dir=root / "out",
                    project_root=ROOT,
                    python_executable="python",
                    device="npu",
                    device_id=0,
                )

    def test_causal_intervention_stages_requested_layer_sign_and_strength(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "subspace.pt"
            artifact.write_bytes(b"subspace")
            data = root / "data"
            data.mkdir()
            (data / "val.jsonl").write_text("{}\n", encoding="utf-8")
            (data / "test.jsonl").write_text("{}\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            plan = prepare_evaluation(
                "causal_intervention",
                {"axes": {"layers": "random", "sign": -1, "strength": 0.5}, "inputs": {"subspace_artifact": str(artifact), "intervention_data": str(data), "intervention_model": str(model)}},
                output_dir=root / "out",
                project_root=ROOT,
                python_executable="python",
                device="npu",
                device_id=1,
            )
            config = yaml.safe_load(plan.config_path.read_text(encoding="utf-8"))
            loaded = load_phasec_config(plan.config_path)
            self.assertEqual(config["method"]["alphas"], [-0.5])
            self.assertEqual(config["method"]["layer_mode"], "random")
            self.assertEqual(loaded.method.layer_mode, "random")

    def test_random_intervention_layer_is_reproducible_and_excludes_the_key_layer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.pt"
            models = {
                layer: {
                    "basis": torch.eye(3)[:1],
                    "target_center": torch.zeros(3),
                    "reference_center": torch.ones(3),
                }
                for layer in (1, 3, 5)
            }
            torch.save(
                {
                    "best_layer_idx": 3,
                    "best_threshold": 0.5,
                    "rank": 1,
                    "target_label": "harmful",
                    "reference_label": "harmless",
                    "models": models,
                },
                path,
            )
            left = load_intervention_artifact(path, layer_mode="random", random_seed=17)
            right = load_intervention_artifact(path, layer_mode="random", random_seed=17)
        self.assertEqual(left.best_layer_idx, right.best_layer_idx)
        self.assertIn(left.best_layer_idx, {1, 5})

    def test_missing_inputs_fail_before_any_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(EvaluationError, "missing"):
                prepare_evaluation(
                    "general_capability_suite",
                    {"axes": {"benchmark": "mmlu"}, "inputs": {"trained_checkpoints": str(Path(td) / "missing"), "benchmark_assets": td}},
                    output_dir=Path(td) / "out",
                    project_root=ROOT,
                    python_executable="python",
                    device="npu",
                    device_id=0,
                )

    def test_cross_corpus_stages_real_batch_evaluation_with_wildguard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self._make_model(root / "model")
            wildguard = self._make_model(root / "wildguard")
            common = root / "common"
            common.mkdir()
            (common / "pan_heldout.jsonl").write_text("{}\n", encoding="utf-8")
            registry = root / "checkpoints.jsonl"
            registry.write_text(
                json.dumps(
                    {
                        "checkpoint_id": "pan-ours",
                        "pair": "qwen35_9b_to_08b",
                        "train_corpus": "pan",
                        "method": "ours",
                        "kind": "merged",
                        "model_path": str(model),
                        "checkpoint_hash": "sha256:abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = prepare_evaluation(
                "cross_corpus_matrix",
                {
                    "axes": {"test_suite": "pan_heldout"},
                    "inputs": {
                        "trained_checkpoints": str(registry),
                        "common_test": str(common),
                        "wildguard_model": str(wildguard),
                    },
                },
                output_dir=root / "out",
                project_root=ROOT,
                python_executable="python",
                device="npu",
                device_id=2,
            )
            manifest = json.loads((root / "out" / "cross_corpus_run_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(plan.argv[1].endswith("31_eval_cross_corpus.py"))
        self.assertEqual(manifest["test_suite"], "pan_heldout")
        self.assertEqual(manifest["checkpoints"][0]["checkpoint_id"], "pan-ours")
        self.assertEqual(manifest["runtime_device"], "npu:2")

    def test_cross_corpus_collection_requires_real_generations_and_judge_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "out"
            run = output / "raw" / "pan-ours"
            run.mkdir(parents=True)
            (output / "cross_corpus_run_manifest.json").write_text(
                json.dumps(
                    {
                        "test_suite": "common_safety",
                        "checkpoints": [
                            {
                                "checkpoint_id": "pan-ours",
                                "pair": "qwen35_9b_to_08b",
                                "train_corpus": "pan",
                                "method": "ours",
                                "checkpoint_hash": "sha256:abc",
                                "output_dir": str(run),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run / "pan_results.json").write_text(
                json.dumps({"status": "ok", "generations": [{"id": "a"}], "harmful_refusal_rate": 0.8, "harmless_over_refusal_rate": 0.1}),
                encoding="utf-8",
            )
            (run / "judge_results.json").write_text(
                json.dumps({"judge": "wildguard", "judge_parse_rate": 1.0, "num_unmatched_ids": 0, "llm_judge_asr": 0.2, "llm_judge_refusal_rate": 0.75, "llm_judge_over_refusal": 0.12}),
                encoding="utf-8",
            )
            destination = collect_evaluation_result(
                "cross_corpus_matrix",
                {"axes": {"test_suite": "common_safety"}},
                output,
            )
            result = json.loads(destination.read_text(encoding="utf-8"))
            (run / "judge_results.json").unlink()
            with self.assertRaises(EvaluationError):
                collect_evaluation_result(
                    "cross_corpus_matrix",
                    {"axes": {"test_suite": "common_safety"}},
                    output,
                )
        self.assertEqual(result["matrix"]["qwen35_9b_to_08b"]["pan"]["ours"]["wildguard_asr"], 0.2)
        self.assertEqual(result["rows"][0]["keyword_harmful_refusal"], 0.8)

    def test_cross_corpus_worker_runs_generation_then_judge_for_merged_and_adapter(self) -> None:
        script = ROOT / "scripts" / "31_eval_cross_corpus.py"
        spec = importlib.util.spec_from_file_location("cross_corpus_worker_test", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_jsonl = root / "test.jsonl"
            test_jsonl.write_text("{}\n", encoding="utf-8")
            entries = [
                {
                    "checkpoint_id": "merged",
                    "kind": "merged",
                    "model_path": str(root / "model"),
                    "output_dir": str(root / "merged-out"),
                },
                {
                    "checkpoint_id": "adapter",
                    "kind": "adapter",
                    "base_model_path": str(root / "base"),
                    "manifest_path": str(root / "manifest.json"),
                    "checkpoint_path": str(root / "epoch.pt"),
                    "output_dir": str(root / "adapter-out"),
                },
            ]
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "test_jsonl": str(test_jsonl),
                        "wildguard_model": str(root / "wildguard"),
                        "runtime_backend": "npu",
                        "runtime_device": "npu:0",
                        "checkpoints": entries,
                    }
                ),
                encoding="utf-8",
            )
            calls = []
            with patch.object(module, "parse_args", return_value=type("Args", (), {"manifest": str(manifest)})()), patch.object(
                module.subprocess,
                "run",
                side_effect=lambda command, **kwargs: calls.append((list(command), kwargs)),
            ):
                self.assertEqual(module.main(), 0)
        self.assertEqual(len(calls), 4)
        self.assertTrue(calls[0][0][1].endswith("12_eval_baseline_suite.py"))
        self.assertTrue(calls[1][0][1].endswith("22_judge_generations.py"))
        self.assertNotIn("--adapter-checkpoint", calls[0][0])
        self.assertIn("--adapter-manifest", calls[2][0])
        self.assertIn("--adapter-checkpoint", calls[2][0])
        self.assertTrue(calls[3][0][1].endswith("22_judge_generations.py"))
        self.assertTrue(all(call[1]["check"] for call in calls))


if __name__ == "__main__":
    unittest.main()
