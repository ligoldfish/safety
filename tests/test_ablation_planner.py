from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from src.ablations.catalog import load_catalog
from src.ablations.planner import (
    PlanError,
    build_catalog_plan,
    build_main_table_plan,
    canonical_cell_id,
    validate_plan,
)
from src.ablations.schema import ExperimentCell


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "ablations" / "catalog.yaml"


class AblationPlannerTests(unittest.TestCase):
    def test_all_plan_covers_every_experiment_and_keeps_outputs_unique(self) -> None:
        catalog = load_catalog(CATALOG_PATH)
        plan = build_catalog_plan(catalog, output_root="/persistent/outputs", scope="all")
        self.assertEqual({cell.experiment_id for cell in plan.cells}, set(catalog.experiments))
        self.assertEqual(
            sum(cell.experiment_id == "P0-01" for cell in plan.cells),
            150,
        )
        self.assertEqual(len({cell.cell_id for cell in plan.cells}), len(plan.cells))
        self.assertEqual(len({cell.output_dir for cell in plan.cells}), len(plan.cells))
        p105 = [cell for cell in plan.cells if cell.experiment_id == "P1-05"]
        self.assertEqual(len(p105), 16)
        self.assertEqual(sum(cell.axes["mode"] == "random_k" for cell in p105), 10)
        p115 = [cell for cell in plan.cells if cell.experiment_id == "P1-15"]
        self.assertEqual(len(p115), 8)
        self.assertTrue(
            all(cell.axes["rank_cap"] == 32 or cell.axes["energy_threshold"] == 0.8 for cell in p115)
        )
        p006 = [cell for cell in plan.cells if cell.experiment_id == "P0-06"]
        self.assertEqual(len(p006), 90)
        self.assertEqual({cell.axes["pair"] for cell in p006}, set(catalog.formal_pairs))
        p007 = [cell for cell in plan.cells if cell.experiment_id == "P0-07"]
        self.assertEqual(len(p007), 24)
        self.assertEqual({cell.axes["method"] for cell in p007}, {"sft1", "random", "ours"})
        global_cells = [cell for cell in p007 if cell.axes["config"] == "global"]
        selected_cells = [
            cell for cell in p007 if cell.axes["config"] == "validation_selected"
        ]
        self.assertEqual(len(global_cells), 18)
        self.assertEqual({cell.axes["dataset"] for cell in global_cells}, set(catalog.formal_datasets))
        self.assertEqual(len(selected_cells), 6)
        self.assertEqual(
            {cell.axes["dataset"] for cell in selected_cells},
            {"wildjailbreak", "wildguardmix"},
        )
        self.assertEqual(len(plan.cells), 509)
        counts = {kind.value: 0 for kind in {item.execution_kind for item in catalog.experiments.values()}}
        for cell in plan.cells:
            counts[catalog.experiments[cell.experiment_id].execution_kind.value] += 1
        self.assertEqual(
            counts,
            {"train": 289, "evaluate": 31, "analyze": 186, "manual": 3},
        )

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
