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


if __name__ == "__main__":
    unittest.main()
