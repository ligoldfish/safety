from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts" / "backfill_missing_pan_results.py"
    if not path.is_file():
        raise AssertionError(f"missing script: {path}")
    spec = importlib.util.spec_from_file_location("backfill_missing_pan_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MissingPanTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script()

    def test_targets_are_exactly_the_three_known_missing_generations(self) -> None:
        root = Path("/root/safety").resolve()
        targets = self.module.build_targets(
            root,
            nosft_device=0,
            wjb_device=1,
            nosft_batch_size=16,
            wjb_batch_size=4,
        )

        self.assertEqual([target.key for target in targets], [
            "qwen35_c5_nosft",
            "qwen3_4b_wjb_sft_epoch_002",
            "qwen3_4b_wjb_sft_epoch_003",
        ])
        self.assertEqual(
            targets[0].output_dir,
            root / "outputs" / "baselines" / "eval_c5_npu",
        )
        for epoch, target in zip(("epoch_002", "epoch_003"), targets[1:]):
            training_root = (
                root / "outputs" / "baselines" / "sft_qwen3_4b_wildjailbreak_npu"
            )
            self.assertEqual(target.output_dir, training_root / "eval_suite" / epoch)
            self.assertEqual(target.manifest_path, training_root / "manifest.json")
            self.assertEqual(
                target.checkpoint_path, training_root / "checkpoints" / f"{epoch}.pt"
            )
            self.assertEqual(target.physical_device, 1)
            self.assertEqual(target.batch_size, 4)

    def test_pan_result_requires_nonempty_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pan_results.json"
            self.assertFalse(self.module.pan_results_is_usable(path))
            path.write_text(json.dumps({"generations": []}), encoding="utf-8")
            self.assertFalse(self.module.pan_results_is_usable(path))
            path.write_text(
                json.dumps({"generations": [{"id": "sample-1"}]}), encoding="utf-8"
            )
            self.assertTrue(self.module.pan_results_is_usable(path))

    def test_effective_config_uses_absolute_paths_and_logical_npu_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            output_dir = root / "outputs" / "run" / "eval_suite" / "epoch_002"
            config_dir.mkdir(parents=True)
            source = config_dir / "eval.yaml"
            source.write_text(
                yaml.safe_dump({
                    "seed": 42,
                    "model": {
                        "name": "student",
                        "path": "../models/student",
                        "runtime_backend": "npu",
                        "runtime_device": "npu:7",
                    },
                    "adapter": {"manifest_path": "", "checkpoint_path": ""},
                    "datasets": {
                        "pan": {"path": "../data/test.jsonl", "enabled": True}
                    },
                    "runtime": {"max_length": 4096, "batch_size": 16},
                    "output": {"output_root": "../outputs/base"},
                }),
                encoding="utf-8",
            )
            target = self.module.PanBackfillTarget(
                key="test",
                eval_config=source,
                output_dir=output_dir,
                physical_device=3,
                batch_size=4,
                manifest_path=root / "outputs" / "run" / "manifest.json",
                checkpoint_path=root / "outputs" / "run" / "checkpoints" / "epoch_002.pt",
            )
            destination = output_dir / "backfill_eval_config.yaml"

            self.module.write_effective_config(target, destination)
            payload = yaml.safe_load(destination.read_text(encoding="utf-8"))

            self.assertEqual(payload["model"]["runtime_device"], "npu:0")
            self.assertEqual(payload["runtime"]["batch_size"], 4)
            self.assertEqual(payload["model"]["path"], str((root / "models/student").resolve()))
            self.assertEqual(
                payload["datasets"]["pan"]["path"],
                str((root / "data/test.jsonl").resolve()),
            )
            self.assertEqual(payload["output"]["output_root"], str(output_dir.resolve()))
            self.assertEqual(
                payload["adapter"]["checkpoint_path"],
                str(target.checkpoint_path.resolve()),
            )

    def test_eval_command_explicitly_passes_full_finetune_artifacts(self) -> None:
        root = Path("/root/safety")
        target = self.module.build_targets(
            root,
            nosft_device=0,
            wjb_device=1,
            nosft_batch_size=16,
            wjb_batch_size=4,
        )[1]
        config = target.output_dir / "backfill_eval_config.yaml"

        command = self.module.build_eval_command("python", root, target, config)

        self.assertIn("--adapter-manifest", command)
        self.assertIn(str(target.manifest_path), command)
        self.assertIn("--adapter-checkpoint", command)
        self.assertIn(str(target.checkpoint_path), command)
        self.assertEqual(command[-2:], ["--output-dir", str(target.output_dir)])


if __name__ == "__main__":
    unittest.main()
