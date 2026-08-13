from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fairness_search_cli", ROOT / "scripts" / "34_fairness_search.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FairnessSearchCliTests(unittest.TestCase):
    def test_plan_writes_exactly_twelve_bounded_trials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            output = root / "plan.jsonl"
            self.assertEqual(
                MODULE.main(["plan", "--output-root", str(root / "runs"), "--out", str(output)]),
                0,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["trial_id"] for row in rows}), 12)

    def test_run_dry_run_selects_one_trial_and_never_executes(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(
            MODULE.subprocess, "run"
        ) as run:
            root = Path(td).resolve() / "runs"
            result = MODULE.main([
                "run", "--output-root", str(root),
                "--trial-id", "wildjailbreak-ours-global",
                "--judge-model", str(Path(td) / "judge"),
                "--device", "npu", "--dry-run",
            ])
        self.assertEqual(result, 0)
        run.assert_not_called()

    def test_unknown_trial_and_source_tree_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(MODULE.FairnessSearchError, "unknown"):
                MODULE.main([
                    "run", "--output-root", str(Path(td).resolve()),
                    "--trial-id", "all", "--judge-model", str(Path(td)),
                    "--device", "npu", "--dry-run",
                ])
        with self.assertRaisesRegex(MODULE.FairnessSearchError, "uploaded source"):
            MODULE.main(["plan", "--output-root", str(ROOT / "bad")])
        with tempfile.TemporaryDirectory() as td, self.assertRaisesRegex(
            MODULE.FairnessSearchError, "uploaded source"
        ):
            MODULE.main([
                "plan", "--output-root", str(Path(td).resolve()),
                "--out", str(ROOT / "bad-plan.jsonl"),
            ])
        with patch.object(MODULE.Path, "resolve", return_value=Path("/tmp/fairness")):
            with self.assertRaisesRegex(MODULE.FairnessSearchError, "persistent"):
                MODULE._absolute_root(Path("/tmp/fairness"))

    def test_real_run_rejects_incomplete_judge_before_starting_training(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(
            MODULE.subprocess, "run"
        ) as run:
            with self.assertRaisesRegex(MODULE.FairnessSearchError, "MODEL_CONFIG_MISSING"):
                MODULE.main([
                    "run", "--output-root", str(Path(td).resolve() / "runs"),
                    "--trial-id", "wildjailbreak-ours-global",
                    "--judge-model", str(Path(td)), "--device", "npu",
                ])
        run.assert_not_called()

    def test_real_run_preflight_never_inspects_test_assets(self) -> None:
        ready = SimpleNamespace(status="READY", issues=())
        model = SimpleNamespace(asset_id="training_teacher_model")
        train = SimpleNamespace(asset_id="training_safety_train")
        test = SimpleNamespace(asset_id="training_safety_eval")
        pan_test = SimpleNamespace(asset_id="training_pan_test")
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(MODULE, "inspect_model_directory", return_value=ready),
            patch.object(MODULE, "training_model_requirements", return_value=(model,)),
            patch.object(MODULE, "training_data_requirements", return_value=(train, test, pan_test)),
            patch.object(MODULE, "run_preflight", return_value=ready) as preflight,
            patch.object(MODULE.subprocess, "run", return_value=SimpleNamespace(returncode=7)),
        ):
            result = MODULE.main([
                "run", "--output-root", str(Path(td).resolve() / "runs"),
                "--trial-id", "wildjailbreak-ours-global",
                "--judge-model", str(Path(td)), "--device", "npu",
            ])
        self.assertEqual(result, 7)
        checked = preflight.call_args.args[0]
        self.assertEqual({item.asset_id for item in checked}, {"training_teacher_model", "training_safety_train"})


if __name__ == "__main__":
    unittest.main()
