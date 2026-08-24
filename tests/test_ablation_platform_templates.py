from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.ablations.catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")
TEMPLATE = ROOT / "configs" / "ablations" / "assets.modelmate.template.json"
RUNBOOK = ROOT / "docs" / "ABLATION_MODELMATE_RUNBOOK.md"
COVERAGE = ROOT / "docs" / "ablations" / "EXPERIMENT_COVERAGE.md"
ALL_RUNBOOK = ROOT / "docs" / "ablations" / "ICLR_ALL_ABLATIONS_RUNBOOK.md"


class AblationPlatformTemplateTests(unittest.TestCase):
    def test_asset_template_covers_every_catalog_requirement_with_one_explicit_kind(self) -> None:
        assets = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        required = {
            asset_id
            for definition in CATALOG.experiments.values()
            for asset_id in definition.requires
        }
        self.assertEqual(set(assets), required)
        for asset_id, entry in assets.items():
            with self.subTest(asset_id=asset_id):
                self.assertEqual(set(entry), {"path", "kind"})
                self.assertIn(entry["kind"], {"file", "directory", "model"})
                self.assertRegex(
                    entry["path"],
                    r"^\$\{SAFETY_(DATA|MODEL|OUTPUT)_ROOT\}/",
                )
                self.assertNotIn("/home/work/user-job-dir/app/models", entry["path"])
        self.assertEqual(assets["checkpoint_registry"]["kind"], "file")
        self.assertEqual(assets["trained_checkpoints"]["kind"], "directory")
        self.assertEqual(assets["wildguard_model"]["kind"], "model")

    def test_runbook_has_persistent_roots_bounded_jobs_and_separate_dry_run_state(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        for required in (
            "/opt/dpcvol/models/safetytransfer",
            "/opt/dpcvol/datasets/safetytransfer",
            "SAFETY_OUTPUT_ROOT",
            "--asset-manifest",
            "--shard-index",
            "--shard-count",
            "--max-cells",
            "dry-run-state",
            "run-state",
            "ASCEND_RT_VISIBLE_DEVICES",
            "checkpoint_registry",
            "search_ledger",
            "509",
            "289",
            "360",
            "140",
            "31",
            "186",
            "3",
            "P0-06",
            "P0-07",
            "preflight",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        self.assertNotIn("ln -s", text)
        self.assertNotIn("pip install --upgrade", text)

    def test_readme_and_coverage_docs_are_catalog_complete(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        coverage = COVERAGE.read_text(encoding="utf-8")
        runbook = ALL_RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("docs/ablations/ICLR_ALL_ABLATIONS_RUNBOOK.md", readme)
        self.assertIn("docs/ablations/EXPERIMENT_COVERAGE.md", readme)
        for experiment_id in CATALOG.experiments:
            self.assertIn(experiment_id, coverage)
        for token in (
            "509", "289", "360", "140", "150", "35/35", "plan", "preflight", "run", "status", "summarize"
        ):
            self.assertIn(token, coverage + runbook)


if __name__ == "__main__":
    unittest.main()
