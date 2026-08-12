from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from src.ablations.catalog import load_catalog
from src.ablations.planner import PlanError, build_main_table_plan, canonical_cell_id, validate_plan
from src.ablations.schema import ExperimentCell


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "ablations" / "catalog.yaml"


class AblationPlannerTests(unittest.TestCase):
    def test_main_table_is_exactly_150_unique_cells(self) -> None:
        plan = build_main_table_plan(load_catalog(CATALOG_PATH), output_root="/persistent/outputs")
        self.assertEqual(len(plan.cells), 150)
        self.assertEqual(len({cell.cell_id for cell in plan.cells}), 150)
        self.assertEqual(len({cell.output_dir for cell in plan.cells}), 150)

    def test_canonical_cell_id_does_not_depend_on_mapping_order(self) -> None:
        left = {
            "experiment_id": "P0-01",
            "axes": {"pair": "p", "dataset": "d", "method": "ours"},
            "overrides": {"seed": 42, "target": {"mode": "semantic"}},
        }
        right = {
            "overrides": {"target": {"mode": "semantic"}, "seed": 42},
            "axes": {"method": "ours", "dataset": "d", "pair": "p"},
            "experiment_id": "P0-01",
        }
        self.assertEqual(canonical_cell_id(left), canonical_cell_id(right))

    def test_duplicate_output_directory_is_rejected(self) -> None:
        plan = build_main_table_plan(load_catalog(CATALOG_PATH), output_root="/persistent/outputs")
        bad = replace(plan.cells[1], output_dir=plan.cells[0].output_dir)
        with self.assertRaisesRegex(PlanError, "duplicate output directory"):
            validate_plan(replace(plan, cells=(plan.cells[0], bad)))

    def test_dependency_cycle_is_rejected(self) -> None:
        one = ExperimentCell(
            cell_id="one",
            experiment_id="P0-01",
            axes={},
            overrides={},
            output_dir="/out/one",
            depends_on=("two",),
        )
        two = replace(one, cell_id="two", output_dir="/out/two", depends_on=("one",))
        with self.assertRaisesRegex(PlanError, "dependency cycle"):
            validate_plan((one, two))


if __name__ == "__main__":
    unittest.main()
