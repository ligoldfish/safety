from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "33_modelmate_ablation_job.py"


def _load():
    spec = importlib.util.spec_from_file_location("modelmate_ablation_job_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelMateAblationJobTests(unittest.TestCase):
    def test_wave_partition_covers_every_cell_exactly_once(self) -> None:
        module = _load()
        from src.ablations.catalog import load_catalog
        from src.ablations.planner import build_catalog_plan

        catalog = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")
        complete = build_catalog_plan(catalog, output_root="/out", scope="all")
        seen: set[str] = set()
        counts = {}
        for wave in module.EXECUTION_WAVES:
            selected = module.select_wave_cells(catalog, complete, wave)
            counts[wave] = len(selected.cells)
            self.assertFalse(seen & {cell.cell_id for cell in selected.cells})
            seen.update(cell.cell_id for cell in selected.cells)
        self.assertEqual(
            counts,
            {
                "core-train": 175,
                "wjb": 90,
                "fairness": 12,
                "evaluate": 31,
                "analyze": 186,
                "manual": 3,
            },
        )
        self.assertEqual(seen, {cell.cell_id for cell in complete.cells})

    def test_commands_are_bounded_persistent_and_keep_dry_run_state_separate(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            config = module.JobConfig(
                wave="core-train",
                model_root=root / "models",
                data_root=root / "datasets",
                output_root=root / "outputs",
                asset_manifest=ROOT / "configs" / "ablations" / "assets.modelmate.template.json",
                shard_index=2,
                shard_count=175,
                max_cells=1,
                device="npu",
                device_id=0,
                num_devices=1,
                dry_run=True,
            )
            commands = module.build_commands(config, project_root=ROOT)
        rendered = [token for command in commands for token in command]
        self.assertEqual(len(commands), 4)
        self.assertIn("--max-cells", rendered)
        self.assertIn("1", rendered)
        self.assertIn("--dry-run", rendered)
        preflight = commands[1]
        self.assertIn("--device", preflight)
        self.assertEqual(preflight[preflight.index("--device") + 1], "npu")
        self.assertTrue(any("dry-run-state" in token for token in rendered))
        self.assertFalse(any("run-state" in token and "dry-run-state" not in token for token in rendered))
        self.assertNotIn("pip", rendered)
        self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", rendered)

    def test_job_rejects_ephemeral_or_snapshot_output_roots(self) -> None:
        module = _load()
        with self.assertRaisesRegex(ValueError, "persistent"):
            module.validate_persistent_root(Path("/tmp/output"), "output")
        with self.assertRaisesRegex(ValueError, "persistent"):
            module.validate_persistent_root(
                Path("/home/work/user-job-dir/app/output"), "output"
            )

    def test_command_builder_honors_the_supplied_project_snapshot(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            project = root / "immutable-source"
            config = module.JobConfig(
                wave="evaluate",
                model_root=root / "models",
                data_root=root / "datasets",
                output_root=root / "outputs",
                asset_manifest=root / "assets.json",
                shard_index=0,
                shard_count=31,
                max_cells=1,
                device="npu",
                device_id=0,
                num_devices=1,
            )
            commands = module.build_commands(config, project_root=project)
        scripts = [
            token
            for command in commands
            for token in command
            if token.endswith("30_ablation.py")
        ]
        self.assertTrue(scripts)
        self.assertEqual(set(scripts), {str(project / "scripts" / "30_ablation.py")})

    def test_parallel_shards_have_private_reports_and_shared_run_state(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            base = dict(
                wave="core-train",
                model_root=root / "models",
                data_root=root / "datasets",
                output_root=root / "outputs",
                asset_manifest=root / "assets.json",
                shard_count=175,
                max_cells=1,
                device="npu",
                device_id=0,
                num_devices=1,
            )
            left = module.build_commands(module.JobConfig(shard_index=0, **base))
            right = module.build_commands(module.JobConfig(shard_index=1, **base))
        left_plan = Path(left[0][left[0].index("--output") + 1])
        right_plan = Path(right[0][right[0].index("--output") + 1])
        left_state = Path(left[2][left[2].index("--state-root") + 1])
        right_state = Path(right[2][right[2].index("--state-root") + 1])
        self.assertNotEqual(left_plan, right_plan)
        self.assertIn("shard-00000-of-00175", left_plan.as_posix())
        self.assertIn("shard-00001-of-00175", right_plan.as_posix())
        self.assertEqual(left_state, right_state)


if __name__ == "__main__":
    unittest.main()
