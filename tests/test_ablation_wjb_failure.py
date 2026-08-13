from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.ablations.wjb_failure import FailureBoundaryError, prepare_failure_evaluations


ROOT = Path(__file__).resolve().parents[1]


def _write_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


class WildJailbreakFailureEvaluationTests(unittest.TestCase):
    def test_prepares_pair_specific_common_eval_and_two_wildguard_judgments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = root / "phase1"
            training = phase1 / "training"
            target_dir = training / "eval_suite" / "epoch_003"
            target_dir.mkdir(parents=True)
            (target_dir / "pan_results.json").write_text(
                json.dumps({"status": "ok", "generations": [{"id": "target-1"}]}) + "\n",
                encoding="utf-8",
            )
            (training / "manifest.json").write_text("{}\n", encoding="utf-8")
            (training / "epoch_003.pt").write_bytes(b"adapter")
            common = root / "common"
            common.mkdir()
            (common / "common_safety.jsonl").write_text(
                json.dumps({"id": "common-1", "label": "harmful", "prompt": "independent common prompt"}) + "\n",
                encoding="utf-8",
            )
            wildguard = root / "wildguard"
            _write_model(wildguard)
            target_jsonl = root / "wildjailbreak_test.jsonl"
            target_jsonl.write_text(
                json.dumps({"id": "target-1", "label": "harmful", "prompt": "held out WJB prompt"}) + "\n",
                encoding="utf-8",
            )
            training_jsonl = root / "wildjailbreak_train.jsonl"
            training_jsonl.write_text(
                json.dumps({"id": "train-1", "label": "harmful", "prompt": "training WJB prompt"}) + "\n",
                encoding="utf-8",
            )
            curation_summary = root / "curation_summary.json"
            curation_summary.write_text(
                json.dumps({"baseline": "wildjailbreak", "mode": "strict", "output_count": 1}) + "\n",
                encoding="utf-8",
            )
            phasef = root / "phaseF.yaml"
            phasef.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "name": "Llama-3.2-1B-Instruct",
                            "path": "/models/Llama-3.2-1B-Instruct",
                            "runtime_backend": "npu",
                            "runtime_device": "npu:2",
                            "local_files_only": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            plan = prepare_failure_evaluations(
                {
                    "experiment_id": "P0-06",
                    "axes": {"pair": "llama31_8b_to_1b"},
                    "inputs": {
                        "common_test": str(common),
                        "wildguard_model": str(wildguard),
                    },
                },
                phase1_root=phase1,
                phasef_config=phasef,
                project_root=ROOT,
                python_executable="python",
                device="npu",
                device_id=2,
                target_test_jsonl=target_jsonl,
                training_jsonl=training_jsonl,
                curation_summary=curation_summary,
            )

            config = yaml.safe_load(plan.common_config.read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["name"], "Llama-3.2-1B-Instruct")
            self.assertEqual(config["datasets"]["pan"]["path"], str((common / "common_safety.jsonl").resolve()))
            self.assertFalse(config["datasets"]["pan"]["placeholder_ok"])
            self.assertEqual(len(plan.commands), 3)
            self.assertIn("12_eval_baseline_suite.py", plan.commands[0][1])
            self.assertEqual(plan.commands[0][-4:], ("--adapter-manifest", str((training / "manifest.json").resolve()), "--adapter-checkpoint", str((training / "epoch_003.pt").resolve())))
            self.assertIn("22_judge_generations.py", plan.commands[1][1])
            self.assertIn(str(target_jsonl.resolve()), plan.commands[1])
            self.assertIn("22_judge_generations.py", plan.commands[2][1])
            self.assertIn(str((common / "common_safety.jsonl").resolve()), plan.commands[2])
            self.assertEqual(plan.target_result, (target_dir / "pan_results.json").resolve())
            manifest = json.loads(plan.manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["split_audit"],
                {
                    "common_target_overlap": 0,
                    "train_common_overlap": 0,
                    "train_target_overlap": 0,
                },
            )
            self.assertEqual(set(manifest["dataset_sha256"]), {"training", "target_test", "common_test"})

    def test_rejects_common_test_without_exact_common_safety_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "common").mkdir()
            with self.assertRaisesRegex(FailureBoundaryError, "common_safety.jsonl"):
                prepare_failure_evaluations(
                    {
                        "experiment_id": "P0-06",
                        "axes": {"pair": "qwen35_9b_to_08b"},
                        "inputs": {
                            "common_test": str(root / "common"),
                            "wildguard_model": str(root / "wildguard"),
                        },
                    },
                    phase1_root=root / "phase1",
                    phasef_config=root / "phaseF.yaml",
                    project_root=ROOT,
                    python_executable="python",
                    device="npu",
                    device_id=0,
                    target_test_jsonl=root / "target.jsonl",
                )

    def test_rejects_any_prompt_overlap_across_train_target_and_common_splits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            common = root / "common"
            common.mkdir()
            duplicate = {"id": "same", "label": "harmful", "prompt": "  Duplicate  Prompt "}
            (common / "common_safety.jsonl").write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
            training = root / "training.jsonl"
            training.write_text(
                json.dumps({**duplicate, "id": "train", "prompt": "duplicate prompt"}) + "\n",
                encoding="utf-8",
            )
            target = root / "target.jsonl"
            target.write_text(json.dumps({"id": "target", "label": "harmful", "prompt": "unique target"}) + "\n", encoding="utf-8")
            curation = root / "curation.json"
            curation.write_text(json.dumps({"baseline": "wildjailbreak", "mode": "off"}) + "\n", encoding="utf-8")
            wildguard = root / "wildguard"
            _write_model(wildguard)
            phase1 = root / "phase1"
            result = phase1 / "training" / "eval_suite" / "epoch_001" / "pan_results.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"status": "ok", "generations": [{"id": "target"}]}) + "\n", encoding="utf-8")
            (phase1 / "training" / "manifest.json").write_text("{}\n", encoding="utf-8")
            (phase1 / "training" / "epoch_001.pt").write_bytes(b"adapter")
            phasef = root / "phaseF.yaml"
            phasef.write_text(yaml.safe_dump({"model": {"name": "Qwen3.5-0.8B", "path": "/models/student"}}), encoding="utf-8")
            with self.assertRaisesRegex(FailureBoundaryError, "overlap"):
                prepare_failure_evaluations(
                    {
                        "experiment_id": "P0-06",
                        "axes": {"pair": "qwen35_9b_to_08b"},
                        "inputs": {"common_test": str(common), "wildguard_model": str(wildguard)},
                    },
                    phase1_root=phase1,
                    phasef_config=phasef,
                    project_root=ROOT,
                    python_executable="python",
                    device="npu",
                    device_id=0,
                    target_test_jsonl=target,
                    training_jsonl=training,
                    curation_summary=curation,
                )


if __name__ == "__main__":
    unittest.main()
