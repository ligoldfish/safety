"""Smoke tests for the full-fine-tuning training mode dispatch.

Covers:
* ``save_checkpoint(save_mode='full' | 'trainable')`` writes the right key.
* ``BaselineSFTConfig`` defaults to ``training.mode='lora'`` for legacy YAMLs.
* All current SFT + distill YAMLs declare ``training.mode='full_finetune'``
  with ``optim.learning_rate==1e-4`` and ``optim.gradient_checkpointing==True``.
* ``load_model_for_evaluation`` skips LoRA injection when the manifest says
  ``mode='sft_full_finetune'``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import (
    BaselineTrainingModeConfig,
    load_distill_config,
    load_sft_config,
)
from src.training.trainer_phase1 import save_checkpoint


class SaveCheckpointModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def test_default_trainable_mode_writes_trainable_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ck.pt"
            save_checkpoint(
                ckpt,
                model=self.model,
                optimizer=self.optimizer,
                epoch=1,
                step=10,
                extra={},
            )
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertIn("trainable_state_dict", payload)
            self.assertNotIn("model_state_dict", payload)
            self.assertEqual(payload["save_mode"], "trainable")
            self.assertIn("optimizer_state_dict", payload)

    def test_full_mode_writes_model_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ck.pt"
            save_checkpoint(
                ckpt,
                model=self.model,
                optimizer=self.optimizer,
                epoch=1,
                step=10,
                extra={},
                save_mode="full",
            )
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", payload)
            self.assertNotIn("trainable_state_dict", payload)
            self.assertEqual(payload["save_mode"], "full")
            full_keys = set(payload["model_state_dict"].keys())
            expected = set(self.model.state_dict().keys())
            self.assertEqual(full_keys, expected)

    def test_save_optimizer_false_drops_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ck.pt"
            save_checkpoint(
                ckpt,
                model=self.model,
                optimizer=self.optimizer,
                epoch=1,
                step=10,
                extra={},
                save_mode="full",
                save_optimizer=False,
            )
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer_state_dict", payload)

    def test_unknown_save_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ck.pt"
            with self.assertRaises(ValueError):
                save_checkpoint(
                    ckpt,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=1,
                    step=10,
                    extra={},
                    save_mode="bogus",
                )


class TrainingModeDataclassTests(unittest.TestCase):
    def test_default_lora_for_legacy_yaml(self) -> None:
        # Synthesize a legacy YAML on the fly (no `training` block).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "legacy.yaml"
            tmp.write_text(
                """
seed: 1
model:
  name: stub
  path: stub-path
data:
  train_split: train.jsonl
  val_split: val.jsonl
optim:
  learning_rate: 5e-5
output:
  output_root: out
""".strip(),
                encoding="utf-8",
            )
            cfg = load_sft_config(tmp)
            self.assertEqual(cfg.training.mode, "lora")
            # Defaults from BaselineOptimConfig must remain backward compat:
            self.assertFalse(cfg.optim.gradient_checkpointing)
            self.assertTrue(cfg.optim.save_optimizer_state)

    def test_full_finetune_yaml_picks_up_mode(self) -> None:
        cfg = load_sft_config(PROJECT_ROOT / "configs" / "baseline_sft_qwen35_08b_npu.yaml")
        self.assertEqual(cfg.training.mode, "full_finetune")
        # Deliberate post-round-2 SFT lr policy (catastrophic-forgetting fix):
        # clean datasets 5e-6, WGM/WJB 1e-6. pan (this config) is clean -> 5e-6.
        self.assertEqual(cfg.optim.learning_rate, 5e-6)
        self.assertTrue(cfg.optim.gradient_checkpointing)
        self.assertFalse(cfg.optim.save_optimizer_state)


# SFT lr policy: full FT for all; WGM/WJB use a lower 1e-6 (they over-refuse at
# 5e-6), every other (clean) baseline uses 5e-6. Distill: full FT @ 5e-6.
SFT_LR_OVERRIDES = {
    "baseline_sft_qwen35_08b_wildguardmix_npu.yaml": 1e-6,
    "baseline_sft_qwen35_08b_wildjailbreak_npu.yaml": 1e-6,
}


class SFTYamlSweepTests(unittest.TestCase):
    """Every SFT + distill YAML on disk must be full FT at the policy lr."""

    def test_sft_configs_full_finetune(self) -> None:
        for yaml_path in sorted((PROJECT_ROOT / "configs").glob("baseline_sft_qwen35_*.yaml")):
            with self.subTest(yaml=yaml_path.name):
                cfg = load_sft_config(yaml_path)
                self.assertEqual(cfg.training.mode, "full_finetune", msg=str(yaml_path))
                expected_lr = SFT_LR_OVERRIDES.get(yaml_path.name, 5e-6)
                self.assertEqual(cfg.optim.learning_rate, expected_lr, msg=str(yaml_path))
                self.assertTrue(cfg.optim.gradient_checkpointing, msg=str(yaml_path))

    def test_distill_configs_full_finetune(self) -> None:
        for yaml_path in sorted((PROJECT_ROOT / "configs").glob("baseline_distill_qwen35_*.yaml")):
            with self.subTest(yaml=yaml_path.name):
                cfg = load_distill_config(yaml_path)
                self.assertEqual(cfg.training.mode, "full_finetune", msg=str(yaml_path))
                self.assertEqual(cfg.optim.learning_rate, 5e-6, msg=str(yaml_path))


class LoadModelForEvaluationDispatchTests(unittest.TestCase):
    def test_full_finetune_manifest_skips_lora_injection(self) -> None:
        # Stub manifest + checkpoint + load_hf_model so we never touch a real model.
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            checkpoint_path = Path(tmpdir) / "ckpt.pt"
            manifest_path.write_text(
                json.dumps({"mode": "sft_full_finetune", "training_mode": "full_finetune"}),
                encoding="utf-8",
            )
            stub_state = {"layer.weight": torch.zeros(2, 2)}
            torch.save(
                {"model_state_dict": stub_state, "save_mode": "full"},
                checkpoint_path,
            )

            class StubModel:
                def __init__(self) -> None:
                    self.calls: list[str] = []

                def load_state_dict(self, state, strict=True):  # noqa: ARG002
                    self.calls.append("load_state_dict")
                    result = mock.Mock()
                    result.unexpected_keys = []
                    result.missing_keys = []
                    return result

                def eval(self) -> None:
                    self.calls.append("eval")

            stub_model = StubModel()
            stub_tokenizer = object()

            from src.baselines import config as cfg_module
            from src.baselines import eval as eval_module

            with mock.patch.object(
                eval_module,
                "load_hf_model",
                return_value=(stub_tokenizer, stub_model, {}),
            ), mock.patch.object(
                eval_module,
                "inject_lora_modules_by_names",
                side_effect=AssertionError("LoRA injection must NOT be called for full FT"),
            ):
                model_cfg = cfg_module.BaselineModelConfig(name="stub", path="stub")
                adapter_cfg = cfg_module.AdapterConfig(
                    manifest_path=str(manifest_path),
                    checkpoint_path=str(checkpoint_path),
                )
                tok, model = eval_module.load_model_for_evaluation(model_cfg, adapter_cfg)
            self.assertIs(tok, stub_tokenizer)
            self.assertIn("load_state_dict", stub_model.calls)
            self.assertIn("eval", stub_model.calls)


if __name__ == "__main__":
    unittest.main()
