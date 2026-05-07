"""Smoke tests for the per-baseline eval JSONLs + launcher routing.

We do NOT exercise the actual builders here -- those need network access
to allenai/wildguardmix etc. Instead we cover:

* Schema check: when a baseline JSONL exists at
  data/processed/eval/<baseline>_test.jsonl it parses cleanly with the
  required fields and binary labels.
* SAFETY_EVAL_CONFIGS in the launcher covers the 6 expected combinations.
* Per-baseline eval YAMLs point datasets.pan.path at the expected
  per-baseline jsonl path.
* WildGuardTest / WildJailbreak label classifiers in script 21 produce
  the expected binary mapping.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BaselineEvalJsonlSchemaTests(unittest.TestCase):
    EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"

    def test_pan_test_unchanged(self) -> None:
        pan_test = PROJECT_ROOT / "data" / "processed" / "pan_test_set.jsonl"
        if not pan_test.exists():
            self.skipTest(f"{pan_test} missing — run scripts/00_prepare_data.py first")
        n = sum(1 for line in pan_test.open(encoding="utf-8") if line.strip())
        self.assertEqual(n, 960, msg="PAN test should have 480 harmful + 480 harmless")

    def test_each_existing_baseline_jsonl_schema(self) -> None:
        for baseline in ("pan", "beavertails", "tulu3_safety", "safety_tuned_llamas"):
            path = self.EVAL_DIR / f"{baseline}_test.jsonl"
            if not path.exists():
                continue  # Eval JSONLs are optional artifacts -- script 21 builds them.
            with self.subTest(baseline=baseline):
                rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
                self.assertGreater(len(rows), 0, msg=f"{path} is empty")
                for row in rows[:50]:
                    self.assertIn("id", row)
                    self.assertIn("label", row)
                    self.assertIn(row["label"], {"harmful", "harmless"}, msg=row)
                    self.assertIn("messages", row)
                    self.assertGreater(len(row["messages"]), 0)
                    roles = {m.get("role") for m in row["messages"]}
                    self.assertIn("user", roles)


class LauncherSafetyEvalRoutingTests(unittest.TestCase):
    LAUNCHER = PROJECT_ROOT / "scripts" / "15_run_oneclick.py"

    def test_safety_eval_configs_covers_expected_keys(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        expected = {
            ("npu", "0.8b", "tulu3_safety"),
            ("tpu", "0.8b", "tulu3_safety"),
            ("npu", "0.8b", "beavertails"),
            ("tpu", "0.8b", "beavertails"),
            ("npu", "0.8b", "safety_tuned_llamas"),
            ("tpu", "0.8b", "safety_tuned_llamas"),
        }
        self.assertTrue(expected.issubset(set(module.SAFETY_EVAL_CONFIGS.keys())))

    def test_safety_eval_datasets_by_baseline_tulu3_has_coconot(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        self.assertEqual(module.SAFETY_EVAL_DATASETS_BY_BASELINE["tulu3_safety"], ("coconot_contrast",))
        self.assertEqual(module.SAFETY_EVAL_DATASETS_BY_BASELINE["safety_tuned_llamas"], ())
        self.assertEqual(module.SAFETY_EVAL_DATASETS_BY_BASELINE["beavertails"], ())

    def test_safety_eval_config_helper_falls_back(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        # Known key returns the safety eval YAML.
        self.assertTrue(
            module._safety_eval_config("npu", "0.8b", "tulu3_safety").endswith(
                "baseline_eval_qwen35_08b_tulu3_safety_npu.yaml"
            )
        )
        # Unknown baseline falls back to the canonical PAN eval YAML.
        self.assertTrue(
            module._safety_eval_config("npu", "0.8b", "unknown_baseline").endswith(
                "baseline_eval_qwen35_08b_npu.yaml"
            )
        )


class PerBaselineEvalYamlPointsAtCorrectJsonl(unittest.TestCase):
    def test_yaml_pan_paths(self) -> None:
        from src.baselines import load_eval_config

        for baseline in ("tulu3_safety", "beavertails", "safety_tuned_llamas"):
            for device in ("npu", "tpu"):
                yaml_path = (
                    PROJECT_ROOT / "configs" / f"baseline_eval_qwen35_08b_{baseline}_{device}.yaml"
                )
                with self.subTest(baseline=baseline, device=device):
                    self.assertTrue(yaml_path.exists(), msg=str(yaml_path))
                    cfg = load_eval_config(yaml_path)
                    self.assertTrue(
                        cfg.datasets.pan.path.endswith(f"{baseline}_test.jsonl"),
                        msg=cfg.datasets.pan.path,
                    )


class WildGuardLabelClassifierTests(unittest.TestCase):
    SCRIPT = PROJECT_ROOT / "scripts" / "21_build_baseline_eval_jsonls.py"

    def test_classifier_mappings(self) -> None:
        module = _load_module("build_baseline_eval_jsonls", self.SCRIPT)
        self.assertEqual(module._classify_wildguard_label("harmful"), "harmful")
        self.assertEqual(module._classify_wildguard_label("unharmful"), "harmless")
        self.assertEqual(module._classify_wildguard_label("benign"), "harmless")
        self.assertEqual(module._classify_wildguard_label(""), "harmful")
        self.assertEqual(module._classify_wildjailbreak_label("benign_query"), "harmless")
        self.assertEqual(module._classify_wildjailbreak_label("harmful_query"), "harmful")
        self.assertEqual(module._classify_wildjailbreak_label("adversarial_benign"), "harmless")
        self.assertEqual(module._classify_wildjailbreak_label("adversarial_harmful"), "harmful")


if __name__ == "__main__":
    unittest.main()
