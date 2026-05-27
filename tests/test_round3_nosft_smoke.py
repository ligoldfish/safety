"""Smoke tests for the Round 3 nosft baseline-aware extension.

Covers:

* ``--baseline pan`` (default) keeps invoking the PAN eval YAML — i.e. the
  pre-Round-3 byte-for-byte behaviour for back-compat.
* ``--baseline {tulu3_safety, beavertails, safety_tuned_llamas, ...}`` routes to
  the per-baseline eval YAML and lifts the over-refusal probe from
  ``SAFETY_EVAL_DATASETS_BY_BASELINE``.
* ``--baseline all`` loops PAN + every registered safety baseline once each.
* ``--baseline beavertails`` does NOT add ``--safety-eval-datasets`` (the
  registry maps BT/STL to an empty tuple).
* ``--model 9b --baseline beavertails`` raises ``ValueError`` because the
  9B variant has no per-baseline eval YAML.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_launcher():
    """Import scripts/15_run_oneclick.py despite its leading-digit filename."""

    path = PROJECT_ROOT / "scripts" / "15_run_oneclick.py"
    spec = importlib.util.spec_from_file_location("oneclick_launcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _NosftHarness:
    """Patch the side-effecting helpers around ``_run_baseline_nosft_one``
    so the test can observe just the ``_run_script`` invocations."""

    def __init__(self, launcher) -> None:
        self.launcher = launcher
        self.run_script_calls: list[tuple[str, list[str]]] = []

    def __enter__(self):
        def _capture_run_script(script_name, args, **kwargs):  # noqa: ANN001
            self.run_script_calls.append((script_name, list(args)))

        def _passthrough_runtime_override(yaml_path, **kwargs):  # noqa: ANN001
            return Path(yaml_path)

        def _stub_load_eval_config(_path):  # noqa: ANN001
            return SimpleNamespace(
                output=SimpleNamespace(output_root="outputs/_test_nosft"),
            )

        self._patches = [
            mock.patch.object(self.launcher, "_run_script", _capture_run_script),
            mock.patch.object(
                self.launcher,
                "_make_runtime_override_config",
                _passthrough_runtime_override,
            ),
            mock.patch.object(
                self.launcher, "load_eval_config", _stub_load_eval_config
            ),
            mock.patch.object(self.launcher, "_validate_device_request", lambda n: None),
            mock.patch.object(self.launcher, "_build_env_overrides", lambda *a, **kw: {}),
            mock.patch.object(
                self.launcher, "_should_run_opencompass", lambda *a, **kw: False
            ),
            mock.patch.object(self.launcher, "_run_final_merge", lambda **kw: None),
            mock.patch.object(
                self.launcher, "_run_opencompass_for_base_model", lambda **kw: None
            ),
        ]
        for patcher in self._patches:
            patcher.start()
        return self

    def __exit__(self, *exc) -> None:
        for patcher in self._patches:
            patcher.stop()


class NosftBaselineRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def _run(self, **overrides):
        defaults = dict(
            device="npu",
            model_size="0.8b",
            baseline_name="pan",
            device_id=0,
            num_devices=1,
            dry_run=True,
            opencompass_dir="",
            opencompass_datasets=(),
            skip_opencompass=True,
            enable_opencompass=False,
        )
        defaults.update(overrides)
        device = defaults.pop("device")
        model_size = defaults.pop("model_size")
        with _NosftHarness(self.launcher) as harness:
            self.launcher._run_baseline_nosft(device, model_size, **defaults)
        return harness.run_script_calls

    def test_default_pan_uses_pan_yaml(self) -> None:
        calls = self._run()
        self.assertEqual(len(calls), 1)
        script, args = calls[0]
        self.assertEqual(script, "12_eval_baseline_suite.py")
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_08b_npu.yaml", joined)
        self.assertNotIn("--safety-eval-datasets", args)

    def test_tulu3_routes_to_baseline_yaml_and_passes_coconot(self) -> None:
        calls = self._run(baseline_name="tulu3_safety")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_08b_tulu3_safety_npu.yaml", joined)
        self.assertIn("--safety-eval-datasets", args)
        idx = args.index("--safety-eval-datasets")
        self.assertEqual(args[idx + 1], "coconot_contrast")

    def test_beavertails_routes_without_extra_safety_eval_datasets(self) -> None:
        calls = self._run(baseline_name="beavertails")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_08b_beavertails_npu.yaml", joined)
        self.assertNotIn("--safety-eval-datasets", args)

    def test_safety_tuned_llamas_routes_without_extra(self) -> None:
        calls = self._run(baseline_name="safety_tuned_llamas")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_08b_safety_tuned_llamas_npu.yaml", joined)
        self.assertNotIn("--safety-eval-datasets", args)

    def test_all_iterates_registered_baselines(self) -> None:
        calls = self._run(baseline_name="all")
        self.assertEqual(len(calls), 1 + len(self.launcher.SAFETY_SFT_BASELINES))
        joined_per_call = [" ".join(args) for _, args in calls]
        self.assertTrue(any("baseline_eval_qwen35_08b_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(
            any("tulu3_safety_npu.yaml" in j for j in joined_per_call)
        )
        self.assertTrue(any("beavertails_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(
            any("safety_tuned_llamas_npu.yaml" in j for j in joined_per_call)
        )
        self.assertTrue(any("wildjailbreak_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(any("wildguardmix_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(any("hh_rlhf_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(any("beavertails_category_npu.yaml" in j for j in joined_per_call))

    def test_9b_with_baseline_other_than_pan_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._run(model_size="9b", baseline_name="beavertails")
        self.assertIn("0.8b", str(ctx.exception).lower())

    def test_9b_pan_still_works(self) -> None:
        # Backwards-compat: nosft --model 9b --baseline pan
        calls = self._run(model_size="9b", baseline_name="pan")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_9b_npu.yaml", joined)


if __name__ == "__main__":
    unittest.main()
