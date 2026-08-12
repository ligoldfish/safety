from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ablations.handlers import HandlerBlocked, execute_handler


class AblationE2ETests(unittest.TestCase):
    def test_provenance_handler_completes_from_real_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "main_table.jsonl").write_text(
                json.dumps(
                    {
                        "cell_id": "c",
                        "model_hash": "m",
                        "dataset_hash": "d",
                        "config_hash": "x",
                        "checkpoint_hash": "k",
                        "commit": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            execute_handler(
                "provenance_matrix",
                {"experiment_id": "P0-01", "axes": {}, "inputs": {"model_registry": str(inputs / "main_table.jsonl"), "dataset_registry": str(inputs / "main_table.jsonl")}},
                output_dir=output,
                required_artifacts=["provenance_matrix.jsonl", "coverage_summary.json"],
            )
            summary = json.loads((output / "coverage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["coverage_rate"], 1.0)

    def test_missing_analysis_asset_is_blocked_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "out"
            with self.assertRaisesRegex(HandlerBlocked, "missing"):
                execute_handler(
                    "seed_and_paired_bootstrap",
                    {"experiment_id": "P0-04", "axes": {"seed": 42}, "inputs": {"aligned_sample_predictions": str(Path(td) / "missing.jsonl")}},
                    output_dir=output,
                    required_artifacts=["paired_bootstrap.json", "seed_summary.json"],
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
