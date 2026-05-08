"""Smoke tests for the Round 3 nosft baseline-aware extension.

Covers:

* ``--baseline pan`` (default) keeps invoking the PAN eval YAML — i.e. the
  pre-Round-3 byte-for-byte behaviour for back-compat.
* ``--baseline {tulu3_safety, beavertails, safety_tuned_llamas}`` routes to
  the per-baseline eval YAML and lifts the over-refusal probe from
  ``SAFETY_EVAL_DATASETS_BY_BASELINE``.
* ``--baseline all`` loops PAN + the three safety baselines once each.
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

    def test_all_iterates_four_baselines(self) -> None:
        calls = self._run(baseline_name="all")
        self.assertEqual(len(calls), 4)
        joined_per_call = [" ".join(args) for _, args in calls]
        self.assertTrue(any("baseline_eval_qwen35_08b_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(
            any("tulu3_safety_npu.yaml" in j for j in joined_per_call)
        )
        self.assertTrue(any("beavertails_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(
            any("safety_tuned_llamas_npu.yaml" in j for j in joined_per_call)
        )

    def test_9b_pan_still_works(self) -> None:
        # Backwards-compat: nosft --model 9b --baseline pan
        calls = self._run(model_size="9b", baseline_name="pan")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_9b_npu.yaml", joined)

    def test_9b_beavertails_routes_to_9b_baseline_yaml(self) -> None:
        calls = self._run(model_size="9b", baseline_name="beavertails")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_9b_beavertails_npu.yaml", joined)
        self.assertNotIn("--safety-eval-datasets", args)

    def test_9b_safety_tuned_llamas_routes_to_9b_baseline_yaml(self) -> None:
        calls = self._run(model_size="9b", baseline_name="safety_tuned_llamas")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_9b_safety_tuned_llamas_npu.yaml", joined)

    def test_9b_tulu3_routes_to_9b_baseline_yaml_with_coconot(self) -> None:
        calls = self._run(model_size="9b", baseline_name="tulu3_safety")
        self.assertEqual(len(calls), 1)
        _, args = calls[0]
        joined = " ".join(args)
        self.assertIn("baseline_eval_qwen35_9b_tulu3_safety_npu.yaml", joined)
        self.assertIn("--safety-eval-datasets", args)
        idx = args.index("--safety-eval-datasets")
        self.assertEqual(args[idx + 1], "coconot_contrast")

    def test_9b_all_iterates_four_baselines(self) -> None:
        calls = self._run(model_size="9b", baseline_name="all")
        self.assertEqual(len(calls), 4)
        joined_per_call = [" ".join(args) for _, args in calls]
        self.assertTrue(any("baseline_eval_qwen35_9b_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(any("9b_tulu3_safety_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(any("9b_beavertails_npu.yaml" in j for j in joined_per_call))
        self.assertTrue(any("9b_safety_tuned_llamas_npu.yaml" in j for j in joined_per_call))


class NosftArgparseDefaultTests(unittest.TestCase):
    """argparse default --baseline=all (Round 4)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def test_baseline_default_is_all(self) -> None:
        argv = ["15_run_oneclick.py", "nosft", "--device", "npu", "--model", "0.8b"]
        with mock.patch.object(sys, "argv", argv):
            args = self.launcher.parse_args()
        self.assertEqual(args.baseline, "all")


class OpenCompassAutoEnableTests(unittest.TestCase):
    """Round 4: ``_should_run_opencompass`` auto-enables when opencompass-dir
    points at an existing path, even without explicit ``--enable-opencompass``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def test_auto_enable_when_dir_exists(self) -> None:
        with mock.patch.object(self.launcher.Path, "exists", return_value=True):
            self.assertTrue(
                self.launcher._should_run_opencompass(
                    "/some/oc/dir", skip_opencompass=False, enable_opencompass=False,
                )
            )

    def test_skip_overrides_auto_enable(self) -> None:
        with mock.patch.object(self.launcher.Path, "exists", return_value=True):
            self.assertFalse(
                self.launcher._should_run_opencompass(
                    "/some/oc/dir", skip_opencompass=True, enable_opencompass=False,
                )
            )

    def test_empty_dir_keeps_disabled(self) -> None:
        self.assertFalse(
            self.launcher._should_run_opencompass(
                "", skip_opencompass=False, enable_opencompass=False,
            )
        )

    def test_explicit_enable_with_missing_dir_warns(self) -> None:
        with mock.patch.object(self.launcher.Path, "exists", return_value=False):
            self.assertFalse(
                self.launcher._should_run_opencompass(
                    "/missing/oc/dir", skip_opencompass=False, enable_opencompass=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
