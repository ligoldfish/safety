from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "30_ablation.py"


class AblationCliTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def test_catalog_and_plan_do_not_load_a_model(self) -> None:
        catalog = self._run("catalog", "--json")
        self.assertEqual(catalog.returncode, 0, catalog.stderr)
        self.assertEqual(json.loads(catalog.stdout)["experiment_count"], 35)
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.jsonl"
            result = self._run(
                "plan",
                "--scope",
                "all",
                "--output-root",
                "/persistent/outputs",
                "--output",
                str(plan_path),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["experiment_id"] for row in rows}, {f"P0-{i:02d}" for i in range(1, 9)} | {f"P1-{i:02d}" for i in range(1, 21)} | {f"P2-{i:02d}" for i in range(1, 8)})

    def test_run_refuses_unbounded_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.jsonl"
            made = self._run("plan", "--scope", "main-table", "--output", str(plan_path))
            self.assertEqual(made.returncode, 0, made.stderr)
            rejected = self._run("run", "--plan", str(plan_path), "--dry-run")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--cell-id", rejected.stderr)

    def test_plan_filters_exact_experiments_and_execution_kinds_for_job_waves(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "train-wave.jsonl"
            result = self._run(
                "plan",
                "--scope",
                "all",
                "--experiment-id",
                "P0-02",
                "--experiment-id",
                "P2-03",
                "--execution-kind",
                "train",
                "--output",
                str(plan_path),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [
                json.loads(line)
                for line in plan_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual({row["experiment_id"] for row in rows}, {"P0-02", "P2-03"})
        self.assertEqual(len(rows), 58)

    def test_plan_can_exclude_special_training_waves(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "core-train.jsonl"
            result = self._run(
                "plan",
                "--scope",
                "all",
                "--execution-kind",
                "train",
                "--exclude-experiment-id",
                "P0-06",
                "--exclude-experiment-id",
                "P0-07",
                "--output",
                str(output),
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 175)
        self.assertFalse({"P0-06", "P0-07"} & {row["experiment_id"] for row in rows})

    def test_plan_rejects_unknown_experiment_and_empty_filter_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            unknown = self._run(
                "plan", "--experiment-id", "P9-99", "--output", str(Path(td) / "x")
            )
            empty = self._run(
                "plan",
                "--experiment-id",
                "P0-03",
                "--execution-kind",
                "train",
                "--output",
                str(Path(td) / "y"),
            )
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("unknown experiment", unknown.stderr)
        self.assertNotEqual(empty.returncode, 0)
        self.assertIn("selected no cells", empty.stderr)

    def test_preflight_uses_the_same_typed_asset_manifest_as_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "plan.jsonl"
            self.assertEqual(
                self._run("plan", "--scope", "main-table", "--output", str(plan_path)).returncode,
                0,
            )
            plan_rows = [
                json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()
            ]
            model_registry = root / "model-registry.jsonl"
            model_registry.write_text(
                "".join(
                    json.dumps(
                        {
                            "cell_id": row["cell_id"],
                            **row["axes"],
                            "model_hash": "a" * 64,
                            "dataset_hash": "b" * 64,
                            "config_hash": "c" * 64,
                            "checkpoint_hash": f"{index:064x}",
                            "commit": "e" * 40,
                        }
                    )
                    + "\n"
                    for index, row in enumerate(plan_rows, 1)
                ),
                encoding="utf-8",
            )
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_registry": {"path": "model-registry.jsonl", "kind": "file"},
                    }
                ),
                encoding="utf-8",
            )
            result = self._run(
                "preflight",
                "--plan",
                str(plan_path),
                "--asset-manifest",
                str(manifest),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "READY")

    def test_preflight_reports_missing_manifest_keys_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "plan.jsonl"
            self.assertEqual(
                self._run("plan", "--scope", "main-table", "--output", str(plan_path)).returncode,
                0,
            )
            manifest = root / "assets.json"
            manifest.write_text("{}\n", encoding="utf-8")
            result = self._run(
                "preflight",
                "--plan",
                str(plan_path),
                "--asset-manifest",
                str(manifest),
            )
        self.assertEqual(result.returncode, 3)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertEqual(
            {issue["asset_id"] for issue in payload["issues"]},
            {"model_registry"},
        )

    def test_preflight_can_be_limited_to_the_exact_bounded_job_shard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "mixed.jsonl"
            made = self._run(
                "plan",
                "--experiment-id",
                "P0-04",
                "--experiment-id",
                "P1-18",
                "--output",
                str(plan_path),
            )
            self.assertEqual(made.returncode, 0, made.stderr)
            rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            ordered = sorted(rows, key=lambda row: row["cell_id"])
            target_index = next(
                index for index, row in enumerate(ordered) if row["experiment_id"] == "P0-04"
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps(
                    {
                        "aligned_sample_predictions": {
                            "path": str(predictions),
                            "kind": "file",
                        }
                    }
                ),
                encoding="utf-8",
            )
            full = self._run(
                "preflight", "--plan", str(plan_path), "--asset-manifest", str(manifest)
            )
            bounded = self._run(
                "preflight",
                "--plan",
                str(plan_path),
                "--asset-manifest",
                str(manifest),
                "--shard-index",
                str(target_index),
                "--shard-count",
                str(len(ordered)),
                "--max-cells",
                "1",
            )
        self.assertEqual(full.returncode, 3)
        self.assertEqual(bounded.returncode, 0, bounded.stderr)
        self.assertEqual(json.loads(bounded.stdout)["status"], "READY")

    def test_preflight_resolves_training_configs_for_the_requested_device(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("ablation_cli_device_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cell = {
            "cell_id": "training-cell",
            "experiment_id": "P0-02",
            "axes": {"dataset": "pan", "method": "ours", "seed": 42},
            "overrides": {},
            "output_dir": "/persistent/output/training-cell",
            "depends_on": [],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = root / "plan.jsonl"
            plan.write_text(json.dumps(cell) + "\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text("{}\n", encoding="utf-8")
            report = type(
                "Report",
                (),
                {
                    "status": "READY",
                    "issues": (),
                    "checked": (),
                    "to_dict": lambda self: {
                        "status": "READY", "issues": [], "checked_assets": []
                    },
                },
            )()
            with (
                patch.object(module, "training_model_requirements", return_value=()) as models,
                patch.object(module, "training_data_requirements", return_value=()) as data,
                patch.object(module, "run_preflight", return_value=report),
            ):
                code = module.main(
                    [
                        "preflight", "--plan", str(plan), "--asset-manifest", str(manifest),
                        "--device", "cpu",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(models.call_args.kwargs["device"], "cpu")
        self.assertEqual(data.call_args.kwargs["device"], "cpu")

    def test_bounded_shard_dry_run_is_stable_and_status_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "plan.jsonl"
            state = root / "state"
            made = self._run("plan", "--scope", "all", "--output", str(plan_path))
            self.assertEqual(made.returncode, 0, made.stderr)
            command = (
                "run",
                "--plan",
                str(plan_path),
                "--state-root",
                str(state),
                "--dry-run",
                "--shard-index",
                "1",
                "--shard-count",
                "7",
                "--max-cells",
                "3",
            )
            first = self._run(*command)
            second = self._run(*command)
            status = self._run("status", "--plan", str(plan_path), "--state-root", str(state))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        left = json.loads(first.stdout)
        right = json.loads(second.stdout)
        self.assertEqual(left["selected_cell_ids"], right["selected_cell_ids"])
        self.assertEqual(len(left["selected_cell_ids"]), 3)
        self.assertEqual(status.returncode, 0, status.stderr)
        states = {row["cell_id"]: row["state"] for row in json.loads(status.stdout)["cells"]}
        self.assertTrue(all(states[cell_id] == "READY" for cell_id in left["selected_cell_ids"]))

    def test_shard_requires_all_bounds_and_rejects_cell_id_combination(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.jsonl"
            self.assertEqual(
                self._run("plan", "--scope", "main-table", "--output", str(plan_path)).returncode,
                0,
            )
            incomplete = self._run(
                "run", "--plan", str(plan_path), "--shard-index", "0", "--shard-count", "2"
            )
            combined = self._run(
                "run",
                "--plan",
                str(plan_path),
                "--cell-id",
                "x",
                "--shard-index",
                "0",
                "--shard-count",
                "2",
                "--max-cells",
                "1",
            )
        self.assertNotEqual(incomplete.returncode, 0)
        self.assertIn("max-cells", incomplete.stderr)
        self.assertNotEqual(combined.returncode, 0)
        self.assertIn("either", combined.stderr)

    def test_real_run_blocked_by_missing_manifest_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "plan.jsonl"
            made = self._run("plan", "--scope", "main-table", "--output", str(plan_path))
            rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            result = self._run(
                "run",
                "--plan",
                str(plan_path),
                "--cell-id",
                rows[0]["cell_id"],
                "--state-root",
                str(root / "state"),
            )
        self.assertEqual(result.returncode, 3)
        self.assertEqual(json.loads(result.stdout)["state"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
