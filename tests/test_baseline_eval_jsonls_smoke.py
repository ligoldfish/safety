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
            ("ppu", "0.8b", "tulu3_safety"),
            ("npu", "0.8b", "beavertails"),
            ("ppu", "0.8b", "beavertails"),
            ("npu", "0.8b", "safety_tuned_llamas"),
            ("ppu", "0.8b", "safety_tuned_llamas"),
        }
        self.assertTrue(expected.issubset(set(module.SAFETY_EVAL_CONFIGS.keys())))

    def test_safety_eval_datasets_by_baseline_tulu3_has_coconot(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        self.assertEqual(module.SAFETY_EVAL_DATASETS_BY_BASELINE["tulu3_safety"], ("coconot_contrast",))
        self.assertEqual(module.SAFETY_EVAL_DATASETS_BY_BASELINE["safety_tuned_llamas"], ())
        self.assertEqual(module.SAFETY_EVAL_DATASETS_BY_BASELINE["beavertails"], ())

    def test_safety_phase_overrides_wildjailbreak(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        ov = module.SAFETY_PHASE_OVERRIDES_BY_BASELINE["wildjailbreak"]
        self.assertEqual(ov["phasef_epochs"], 5)
        self.assertEqual(tuple(ov["analyze_extra"]), ("--top-k", "3"))
        self.assertEqual(tuple(ov["subspace_extra"]), ("--energy-threshold", "0.7", "--rank-cap", "8"))

    def test_bothpole_pipeline_config_and_policy(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        self.assertIn("npu", module.BOTHPOLE_PIPELINE_CONFIGS)
        phasef = PROJECT_ROOT / module.BOTHPOLE_PIPELINE_CONFIGS["npu"]["phasef"]
        # Same phase1 as the main ours run (reuse, no rebuild of subspace/targets).
        self.assertEqual(
            module.BOTHPOLE_PIPELINE_CONFIGS["npu"]["phase1"],
            module.FULL_PIPELINE_CONFIGS["npu"]["phase1"],
        )
        import yaml

        raw = yaml.safe_load(phasef.read_text(encoding="utf-8"))
        # Both poles supervised (vs harmful_only), and a separate output dir.
        self.assertEqual(raw["target"]["layer_loss_policy"], "label_weighted")
        self.assertTrue(str(raw["output"]["output_root"]).endswith("training_bothpole"))

    def test_safety_phase_overrides_wildguardmix(self) -> None:
        module = _load_module("oneclick_launcher", self.LAUNCHER)
        ov = module.SAFETY_PHASE_OVERRIDES_BY_BASELINE["wildguardmix"]
        # Clean contrast -> richer subspace + stronger L_layer; NO epoch override.
        self.assertNotIn("phasef_epochs", ov)
        self.assertEqual(ov["phasef_layer_loss_weight"], 0.5)
        self.assertEqual(tuple(ov["analyze_extra"]), ("--top-k", "7"))
        self.assertEqual(tuple(ov["subspace_extra"]), ("--energy-threshold", "0.9"))
        # A baseline with no override entry stays on global defaults.
        self.assertNotIn("beavertails", module.SAFETY_PHASE_OVERRIDES_BY_BASELINE)

    def test_make_safety_full_overrides_injects_wjb_epochs(self) -> None:
        import tempfile
        import yaml

        module = _load_module("oneclick_launcher", self.LAUNCHER)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            phase1_base = tdp / "phase1.yaml"
            phasef_base = tdp / "phasef.yaml"
            phase1_base.write_text("extraction: {}\ndataset: {}\n", encoding="utf-8")
            phasef_base.write_text(
                "optim:\n  epochs: 3\ninputs: {}\noutput: {}\n", encoding="utf-8"
            )

            def epochs_for(baseline: str) -> int:
                _, phasef_path = module._make_safety_full_overrides(
                    device="npu",
                    device_id=0,
                    baseline_name=baseline,
                    safety_processed_dir=tdp / "proc",
                    safety_phase1_output_root=tdp / "p1",
                    safety_phasef_output_root=tdp / "pf",
                    phasef_base_override=str(phasef_base),
                    phase1_base_override=str(phase1_base),
                )
                raw = yaml.safe_load(Path(phasef_path).read_text(encoding="utf-8"))
                return int(raw["optim"]["epochs"])

            self.assertEqual(epochs_for("wildjailbreak"), 5)
            self.assertEqual(epochs_for("wildguardmix"), 3)  # untouched -> base value

    def test_make_safety_full_overrides_injects_wgm_layer_loss_weight(self) -> None:
        import tempfile
        import yaml

        module = _load_module("oneclick_launcher", self.LAUNCHER)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            phase1_base = tdp / "phase1.yaml"
            phase1_base.write_text("extraction: {}\ndataset: {}\n", encoding="utf-8")

            def llw_for(baseline: str, base_llw: float) -> float:
                phasef_base = tdp / f"phasef_{baseline}_{base_llw}.yaml"
                phasef_base.write_text(
                    f"optim:\n  epochs: 3\n  layer_loss_weight: {base_llw}\ninputs: {{}}\noutput: {{}}\n",
                    encoding="utf-8",
                )
                _, phasef_path = module._make_safety_full_overrides(
                    device="npu", device_id=0, baseline_name=baseline,
                    safety_processed_dir=tdp / "proc",
                    safety_phase1_output_root=tdp / "p1",
                    safety_phasef_output_root=tdp / "pf",
                    phasef_base_override=str(phasef_base),
                    phase1_base_override=str(phase1_base),
                )
                raw = yaml.safe_load(Path(phasef_path).read_text(encoding="utf-8"))
                return float(raw["optim"]["layer_loss_weight"])

            # WGM (override 0.5) applied when the base L_layer weight is non-zero (ours/random)...
            self.assertEqual(llw_for("wildguardmix", 0.25), 0.5)
            # ...but the gate skips it when base is 0.0 (the sft1 ablation must stay 0).
            self.assertEqual(llw_for("wildguardmix", 0.0), 0.0)
            # A baseline without an override entry is untouched.
            self.assertEqual(llw_for("beavertails", 0.25), 0.25)

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
            for device in ("npu", "ppu"):
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
