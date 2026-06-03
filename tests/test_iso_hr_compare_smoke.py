"""Torch-free tests for the iso-HR selection logic.

Critical invariant: checkpoint selection uses VAL HR proximity ONLY -- it must never
prefer a far-HR/low-OR checkpoint over a near-HR/high-OR one (no cherry-picking on OR,
no selecting on the test set).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.iso_hr import last_epoch_key, read_test_hr_or, read_val_hr_or, select_iso_hr


class SelectIsoHrTests(unittest.TestCase):
    def test_picks_val_hr_nearest_ignoring_or(self) -> None:
        # 'a' has far HR but tempting low OR; 'b' is HR-nearest with worse OR.
        # Selection must choose 'b' (HR proximity only).
        checkpoints = [
            {"ckpt": "epoch_001", "val_hr": 0.60, "val_or": 0.01},
            {"ckpt": "step_000500", "val_hr": 0.79, "val_or": 0.20},
        ]
        sel = select_iso_hr(checkpoints, target_hr=0.80, epsilon=0.05)
        self.assertEqual(sel["ckpt"], "step_000500")
        self.assertAlmostEqual(sel["delta_hr"], -0.01)
        self.assertTrue(sel["matched"])

    def test_matched_boundary(self) -> None:
        near = select_iso_hr([{"ckpt": "x", "val_hr": 0.795}], target_hr=0.80, epsilon=0.005)
        self.assertTrue(near["matched"])  # |−0.005| ≤ 0.005
        far = select_iso_hr([{"ckpt": "y", "val_hr": 0.794}], target_hr=0.80, epsilon=0.005)
        self.assertFalse(far["matched"])  # |−0.006| > 0.005

    def test_skips_missing_val_hr(self) -> None:
        sel = select_iso_hr(
            [{"ckpt": "a", "val_hr": None}, {"ckpt": "b", "val_hr": 0.7}],
            target_hr=0.8, epsilon=0.05,
        )
        self.assertEqual(sel["ckpt"], "b")

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(select_iso_hr([], target_hr=0.8, epsilon=0.05))
        self.assertIsNone(select_iso_hr([{"ckpt": "a", "val_hr": None}], 0.8, 0.05))


class ReadHelperTests(unittest.TestCase):
    def test_read_val_and_last_epoch_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp)
            (train / "val_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch_1": {"harmful_refusal_rate": 0.5, "harmless_over_refusal_rate": 0.02},
                        "epoch_2": {"harmful_refusal_rate": 0.7, "harmless_over_refusal_rate": 0.05},
                        "step_000300": {"harmful_refusal_rate": 0.78, "harmless_over_refusal_rate": 0.08},
                    }
                ),
                encoding="utf-8",
            )
            val = read_val_hr_or(train)
            self.assertEqual(val["step_000300"]["val_hr"], 0.78)
            self.assertEqual(last_epoch_key(val), "epoch_2")

            ev = train / "eval_suite" / "epoch_2"
            ev.mkdir(parents=True)
            (ev / "summary.json").write_text(
                json.dumps({"results": {"pan": {"harmful_refusal_rate": 0.72, "harmless_over_refusal_rate": 0.06}}}),
                encoding="utf-8",
            )
            test_hr, test_or = read_test_hr_or(train, "epoch_2")
            self.assertAlmostEqual(test_hr, 0.72)
            self.assertAlmostEqual(test_or, 0.06)
            self.assertEqual(read_test_hr_or(train, "step_000300"), (None, None))


if __name__ == "__main__":
    unittest.main()
