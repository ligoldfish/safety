from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oneclick_search_test", ROOT / "scripts" / "15_run_oneclick.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OneClickFairnessSearchTests(unittest.TestCase):
    def test_skip_test_eval_runs_training_but_no_sanity_tables_or_test_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = root / "phase1"
            phasef = phase1 / "training"
            processed = root / "processed"
            pan = root / "pan"
            scripts: list[str] = []
            script_calls: list[tuple[str, tuple[str, ...]]] = []

            with (
                patch.object(MODULE, "_validate_device_request"),
                patch.object(MODULE, "_make_runtime_override_config", side_effect=lambda path, **_: path),
                patch.object(MODULE, "_resolve", side_effect=lambda path: Path(path)),
                patch.object(MODULE, "_safety_eval_config", return_value="eval.yaml") as eval_config,
                patch.object(MODULE, "load_sft_config", return_value=SimpleNamespace(data=SimpleNamespace(train_split=str(root / "train.jsonl")))),
                patch.object(MODULE, "_build_env_overrides", return_value={}),
                patch.object(MODULE, "_resolve_safety_full_roots", return_value=(processed, pan, phase1, phasef)),
                patch.object(MODULE, "_make_safety_full_overrides", return_value=(root / "phase1.yaml", root / "phasef.yaml")),
                patch.object(MODULE, "_invoke_phase1_curation"),
                patch.object(MODULE, "_run_phase1_precompute") as precompute,
                patch.object(
                    MODULE, "_run_script",
                    side_effect=lambda name, args, **kwargs: (
                        scripts.append(name), script_calls.append((name, tuple(args)))
                    ),
                ),
                patch.object(MODULE, "_run_adapter_eval") as adapter,
            ):
                MODULE._run_safety_full(
                    "npu", baseline_name="wildjailbreak", device_id=0,
                    num_devices=1, dry_run=True, force_rebuild=False, smoke=False,
                    opencompass_dir="", opencompass_datasets=(), skip_opencompass=True,
                    enable_opencompass=False, skip_test_eval=True,
                )

        self.assertIn("09_train_student_semalign.py", scripts)
        split_call = next(call for call in script_calls if call[0] == "20_split_safety_for_semalign.py")
        self.assertIn("--validation-only", split_call[1])
        self.assertTrue(precompute.call_args.kwargs["validation_only"])
        self.assertNotIn("10_sanity_eval.py", scripts)
        self.assertNotIn("11_make_tables.py", scripts)
        eval_config.assert_not_called()
        adapter.assert_not_called()

    def test_validation_only_precompute_never_reads_pan_or_sanity_test_splits(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        with patch.object(
            MODULE, "_run_script",
            side_effect=lambda name, args, **kwargs: calls.append((name, tuple(args))),
        ):
            MODULE._run_phase1_precompute(
                Path("phase1.yaml"), smoke=False, dry_run=True,
                skip_prepare=True, validation_only=True,
            )
        flattened = " ".join(token for _, args in calls for token in args)
        self.assertIn("alignment", flattened)
        self.assertIn("analysis_val", flattened)
        self.assertNotIn("pan_test", flattened)
        self.assertNotIn("sanity_test", flattened)


if __name__ == "__main__":
    unittest.main()
