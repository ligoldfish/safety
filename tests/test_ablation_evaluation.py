from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
import torch

from src.ablations.evaluation import EvaluationError, prepare_evaluation
from src.phase_c.intervention import load_intervention_artifact
from src.utils.config import load_phasec_config


ROOT = Path(__file__).resolve().parents[1]


class AblationEvaluationTests(unittest.TestCase):
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
            model.mkdir()
            assets = root / "opencompass"
            (assets / "opencompass").mkdir(parents=True)
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


if __name__ == "__main__":
    unittest.main()
