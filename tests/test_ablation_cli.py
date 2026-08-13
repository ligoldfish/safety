from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
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

    def test_preflight_uses_the_same_typed_asset_manifest_as_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "plan.jsonl"
            self.assertEqual(
                self._run("plan", "--scope", "main-table", "--output", str(plan_path)).returncode,
                0,
            )
            (root / "model-registry").mkdir()
            (root / "dataset-registry").mkdir()
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_registry": {"path": "model-registry", "kind": "directory"},
                        "dataset_registry": {"path": "dataset-registry", "kind": "directory"},
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
            {"model_registry", "dataset_registry"},
        )

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
