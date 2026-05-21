"""Smoke tests for Round 2 changes.

Covers:

* PhaseF YAML LoRA capacity bump (rank=16, alpha=32, target_modules with
  ``mlp.gate_proj``).
* BeaverTails ``label_strategy="category_any"`` produces prompt-level
  harmful/harmless via the BT category dict (the legacy ``"is_safe"``
  strategy still works for back-compat).
* Tülu3 v2 builder dispatches on ``source`` field and maps raw labels
  (WildGuardMix ``prompt_harm_label``, WildJailbreak ``data_type``,
  personahub helpful → harmless).
* ``scripts/20_split_safety_for_semalign.py`` strict ``--harmless-source
  auto`` rejects safety JSONLs that lack in-domain harmless rows.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data.safety_datasets as safety_datasets
from src.data.safety_datasets import (
    _bt_label_from_category,
    build_beavertails_records,
    build_tulu3_safety_v2_records,
)


PHASEF_YAMLS = (
    "qwen35_9b_to_08b_phaseF_npu.yaml",
    "qwen35_9b_to_08b_phaseF_ppu.yaml",
    "qwen35_9b_to_08b_phaseF_npu_sft1.yaml",
    "qwen35_9b_to_08b_phaseF_npu_random.yaml",
    "qwen35_9b_to_08b_phaseF_ppu_random.yaml",
)


class PhaseFLoRACapacityTests(unittest.TestCase):
    """Round 2 bumps every phaseF YAML to rank=16, alpha=32 with the full
    mlp.* projection set so SemAlign 'ours' has enough capacity."""

    REQUIRED_TARGETS = {
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    def test_all_phasef_yamls_meet_capacity(self) -> None:
        for name in PHASEF_YAMLS:
            yaml_path = PROJECT_ROOT / "configs" / name
            with self.subTest(yaml=name):
                self.assertTrue(yaml_path.exists(), msg=str(yaml_path))
                cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                lora = cfg.get("lora", {})
                self.assertEqual(int(lora.get("rank", 0)), 16, msg=name)
                self.assertEqual(float(lora.get("alpha", 0.0)), 32.0, msg=name)
                targets = set(lora.get("target_modules", []))
                self.assertTrue(
                    self.REQUIRED_TARGETS.issubset(targets),
                    msg=f"{name} missing: {self.REQUIRED_TARGETS - targets}",
                )

    def test_npu_ppu_random_use_two_epochs(self) -> None:
        for name in (
            "qwen35_9b_to_08b_phaseF_npu.yaml",
            "qwen35_9b_to_08b_phaseF_ppu.yaml",
            "qwen35_9b_to_08b_phaseF_npu_sft1.yaml",
            "qwen35_9b_to_08b_phaseF_npu_random.yaml",
            "qwen35_9b_to_08b_phaseF_ppu_random.yaml",
        ):
            yaml_path = PROJECT_ROOT / "configs" / name
            with self.subTest(yaml=name):
                cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                self.assertEqual(int(cfg["optim"]["epochs"]), 2, msg=name)

    def test_npu_sft1_ablation_is_sft_only_and_isolated(self) -> None:
        yaml_path = PROJECT_ROOT / "configs" / "qwen35_9b_to_08b_phaseF_npu_sft1.yaml"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.assertEqual(float(cfg["optim"]["sft_loss_weight"]), 1.0)
        self.assertEqual(float(cfg["optim"]["layer_loss_weight"]), 0.0)
        self.assertTrue(str(cfg["output"]["output_root"]).endswith("/training_sft1"))


class BeaverTailsLabelStrategyTests(unittest.TestCase):
    @staticmethod
    def _row(*, prompt: str, is_safe: bool, category: dict | None) -> dict:
        return {
            "prompt": prompt,
            "response": "stub response.",
            "is_safe": is_safe,
            "category": category,
        }

    def test_bt_category_helper(self) -> None:
        self.assertEqual(_bt_label_from_category({"violence": True}), "harmful")
        self.assertEqual(
            _bt_label_from_category({"violence": False, "drugs": False}),
            "harmless",
        )
        self.assertEqual(_bt_label_from_category({}), "harmful")
        self.assertEqual(_bt_label_from_category(None), "harmful")

    def test_category_any_drops_label_target_mismatch(self) -> None:
        # Aligned: keep both. Misaligned: drop both. This guards against
        # training a refusal target on a benign prompt and vice-versa.
        rows = [
            self._row(
                prompt="aligned harmful",
                is_safe=False,
                category={"violence": True},
            ),
            self._row(
                prompt="aligned harmless",
                is_safe=True,
                category={"violence": False, "drugs": False},
            ),
            self._row(
                prompt="mismatch a (cat harmful, response safe)",
                is_safe=True,
                category={"violence": True},
            ),
            self._row(
                prompt="mismatch b (cat harmless, response unsafe)",
                is_safe=False,
                category={"violence": False, "drugs": False},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(
                safety_datasets, "_load_dataset", return_value=rows
            ):
                records = build_beavertails_records(
                    output_path=output_path,
                    label_strategy="category_any",
                )
        labels = [r["label"] for r in records]
        self.assertEqual(labels, ["harmful", "harmless"])
        # Harmful row -> refusal target; harmless row -> original safe text.
        targets_by_label = {r["label"]: r["target_response"] for r in records}
        self.assertEqual(targets_by_label["harmless"], "stub response.")
        self.assertNotEqual(targets_by_label["harmful"], "stub response.")
        for record in records:
            self.assertEqual(record["label_strategy"], "category_any")

    def test_is_safe_strategy_uses_response_flag(self) -> None:
        rows = [
            self._row(prompt="a", is_safe=True, category={"violence": True}),
            self._row(prompt="b", is_safe=False, category={"violence": False}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(
                safety_datasets, "_load_dataset", return_value=rows
            ):
                records = build_beavertails_records(
                    output_path=output_path,
                    label_strategy="is_safe",
                )
        self.assertEqual([r["label"] for r in records], ["harmless", "harmful"])

    def test_unknown_strategy_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=[]):
                with self.assertRaises(ValueError):
                    build_beavertails_records(
                        output_path=output_path,
                        label_strategy="bogus",
                    )


class Tulu3SafetyV2Tests(unittest.TestCase):
    """v2 builder pulls harmful side from upstream WildGuardMix /
    WildJailbreak directly (the Tülu3 SFT mixture strips
    prompt_harm_label / data_type) and uses the mixture only for
    personahub helpful contrast."""

    def setUp(self) -> None:
        self.wgm_rows = [
            {
                "id": "wgm-harmful",
                "prompt": "how to do bad thing",
                "response": "I cannot help with that.",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unharmful",
            },
            {
                "id": "wgm-benign",
                "prompt": "how to bake bread",
                "response": "Mix flour, water, yeast.",
                "prompt_harm_label": "unharmful",
                "response_harm_label": "unharmful",
            },
            {
                "id": "wgm-benign-noisy-response",
                "prompt": "tell me about chemistry",
                "response": "Detailed step-by-step harm.",
                "prompt_harm_label": "unharmful",
                # Response itself flagged harmful -> drop to avoid teaching
                # unsafe completion on harmless prompts.
                "response_harm_label": "harmful",
            },
            {
                "id": "wgm-bad-label",
                "prompt": "missing label",
                "response": "x",
                "prompt_harm_label": "",
            },
        ]
        self.wjb_rows = [
            {
                "id": "wjb-adv-harmful",
                "adversarial": "disguised attack",
                "vanilla": "",
                "completion": "I will not assist.",
                "data_type": "adversarial_harmful",
            },
            {
                "id": "wjb-adv-benign",
                "adversarial": "dressed-up benign",
                "vanilla": "plain benign",
                "completion": "Sure, here is a benign answer.",
                "data_type": "adversarial_benign",
            },
            {
                "id": "wjb-skip-bad-type",
                "adversarial": "weird",
                "vanilla": "",
                "completion": "y",
                "data_type": "unknown_type",
            },
        ]
        self.mixture_rows = [
            {
                "id": "personahub-1",
                "source": "ai2-adapt-dev/personahub_math_v5_regen_149960",
                "messages": [
                    {"role": "user", "content": "solve x^2 = 4"},
                    {"role": "assistant", "content": "x = ±2"},
                ],
            },
            {
                "id": "personahub-2",
                "source": "ai2-adapt-dev/personahub_math_v5_regen_149960",
                "messages": [
                    {"role": "user", "content": "compute 17 * 23"},
                    {"role": "assistant", "content": "391"},
                ],
            },
            {
                "id": "skip-other-source",
                "source": "ai2-adapt-dev/tulu_v3.9_wildguardmix",
                "messages": [
                    {"role": "user", "content": "ignored — comes from upstream now"},
                    {"role": "assistant", "content": "x"},
                ],
            },
        ]

    def _patch_loader(self):
        # _load_dataset called in order: WGM train, WJB train, mixture.
        return mock.patch.object(
            safety_datasets,
            "_load_dataset",
            side_effect=[self.wgm_rows, self.wjb_rows, self.mixture_rows],
        )

    def test_v2_builder_label_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tulu3v2.jsonl"
            with self._patch_loader():
                records = build_tulu3_safety_v2_records(
                    output_path=output_path,
                    helpful_max_samples=2,
                )
        by_id = {r["id"]: r for r in records}
        # WGM noisy response + bad label dropped.
        self.assertNotIn("wgm-benign-noisy-response", by_id)
        self.assertNotIn("wgm-bad-label", by_id)
        # WJB unknown data_type dropped.
        self.assertNotIn("wjb-skip-bad-type", by_id)
        # Mixture entries other than personahub dropped.
        self.assertNotIn("skip-other-source", by_id)
        self.assertEqual(by_id["wgm-harmful"]["label"], "harmful")
        self.assertEqual(
            by_id["wgm-harmful"]["target_response"],
            safety_datasets.DEFAULT_SAFETY_REFUSAL_TEMPLATE,
        )
        self.assertEqual(by_id["wgm-benign"]["label"], "harmless")
        self.assertEqual(by_id["wgm-benign"]["target_response"], "Mix flour, water, yeast.")
        self.assertEqual(by_id["wjb-adv-harmful"]["label"], "harmful")
        self.assertEqual(
            by_id["wjb-adv-harmful"]["target_response"],
            safety_datasets.DEFAULT_SAFETY_REFUSAL_TEMPLATE,
        )
        self.assertEqual(by_id["wjb-adv-benign"]["label"], "harmless")
        self.assertEqual(by_id["wjb-adv-benign"]["target_response"], "Sure, here is a benign answer.")
        self.assertEqual(by_id["personahub-1"]["label"], "harmless")
        self.assertEqual(by_id["personahub-1"]["raw_label"], "personahub_helpful")
        for record in records:
            self.assertEqual(record["dataset"], "tulu3_safety_v2")

    def test_v2_builder_balances_helpful_to_harmful(self) -> None:
        # WGM: 1 harmful + 1 harmless. WJB: 1 harmful + 1 harmless.
        # n_harmful = 2, n_harmless_safety = 2 -> default helpful_cap = 0.
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tulu3v2.jsonl"
            with self._patch_loader():
                records = build_tulu3_safety_v2_records(output_path=output_path)
        helpful_n = sum(
            1 for r in records if r.get("raw_label") == "personahub_helpful"
        )
        harmful_n = sum(1 for r in records if r["label"] == "harmful")
        harmless_n = sum(1 for r in records if r["label"] == "harmless")
        self.assertGreaterEqual(harmful_n, 1)
        self.assertLessEqual(helpful_n, harmful_n)
        self.assertLessEqual(harmless_n, harmful_n + 1)

    def test_v2_builder_empty_safety_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tulu3v2.jsonl"
            with mock.patch.object(
                safety_datasets,
                "_load_dataset",
                side_effect=[[], [], self.mixture_rows],
            ):
                with self.assertRaises(RuntimeError):
                    build_tulu3_safety_v2_records(output_path=output_path)

    def test_v2_builder_unwraps_datasetdict_shape(self) -> None:
        # HF returns a DatasetDict (a dict subclass) when ``split=`` is
        # not passed. Iterating a dict yields keys (strs), not rows, so
        # the builder must unwrap to the first split before iterating.
        wgm_dict = {"train": self.wgm_rows}
        wjb_dict = {"train": self.wjb_rows}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tulu3v2.jsonl"
            with mock.patch.object(
                safety_datasets,
                "_load_dataset",
                side_effect=[wgm_dict, wjb_dict, self.mixture_rows],
            ):
                records = build_tulu3_safety_v2_records(
                    output_path=output_path,
                    helpful_max_samples=2,
                )
        ids = {r["id"] for r in records}
        self.assertIn("wgm-harmful", ids)
        self.assertIn("wjb-adv-harmful", ids)


class SafetyDatasetRegistryTests(unittest.TestCase):
    def test_v2_in_registry(self) -> None:
        from src.data.safety_datasets import SAFETY_TRAIN_DATASETS

        self.assertIn("tulu3_safety_v2", SAFETY_TRAIN_DATASETS)
        self.assertIn("tulu3_safety", SAFETY_TRAIN_DATASETS)


class SplitHarmlessSourceCLITests(unittest.TestCase):
    """End-to-end smoke for ``20_split_safety_for_semalign.py`` CLI."""

    SCRIPT = PROJECT_ROOT / "scripts" / "20_split_safety_for_semalign.py"

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_auto_rejects_single_pole_jsonl(self) -> None:
        rows = [
            {
                "id": f"only-harmful-{i}",
                "label": "harmful",
                "messages": [
                    {"role": "user", "content": f"prompt {i}"},
                ],
                "target_response": "refuse",
            }
            for i in range(40)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            safety_jsonl = Path(tmpdir) / "harmful_only.jsonl"
            self._write_jsonl(safety_jsonl, rows)
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.SCRIPT),
                    "--safety-jsonl",
                    str(safety_jsonl),
                    "--output-dir",
                    str(output_dir),
                    "--pan-processed-dir",
                    str(tmpdir),
                    "--harmless-source",
                    "auto",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("harmless", (result.stderr or "").lower())

    def test_auto_succeeds_with_in_domain_harmless(self) -> None:
        rows: list[dict] = []
        for i in range(40):
            rows.append(
                {
                    "id": f"harmful-{i}",
                    "label": "harmful",
                    "messages": [{"role": "user", "content": f"bad-{i}"}],
                    "target_response": "refuse",
                }
            )
            rows.append(
                {
                    "id": f"harmless-{i}",
                    "label": "harmless",
                    "messages": [{"role": "user", "content": f"benign-{i}"}],
                    "target_response": "sure",
                }
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            safety_jsonl = Path(tmpdir) / "binary.jsonl"
            self._write_jsonl(safety_jsonl, rows)
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.SCRIPT),
                    "--safety-jsonl",
                    str(safety_jsonl),
                    "--output-dir",
                    str(output_dir),
                    "--pan-processed-dir",
                    str(tmpdir),
                    "--harmless-source",
                    "auto",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            for name in (
                "alignment_set.jsonl",
                "analysis_val_set.jsonl",
                "sanity_test_set.jsonl",
                "pan_train_set.jsonl",
            ):
                self.assertTrue(
                    (output_dir / name).exists(),
                    msg=f"{name} missing under {output_dir}",
                )


if __name__ == "__main__":
    unittest.main()
