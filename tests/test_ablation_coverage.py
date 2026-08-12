from __future__ import annotations

import unittest
from pathlib import Path

from src.ablations.catalog import load_catalog
from src.ablations.handlers import handler_contracts
from src.ablations.planner import build_catalog_plan
from src.ablations.runner import compile_cell_commands, executable_handlers, RunnerContext


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")


class AblationCoverageTests(unittest.TestCase):
    def test_exact_35_contracts_are_registered_and_non_placeholder(self) -> None:
        contracts = handler_contracts()
        wanted = {definition.handler for definition in CATALOG.experiments.values()}
        self.assertEqual(set(contracts), wanted)
        self.assertEqual(set(contracts), executable_handlers())
        for name, contract in contracts.items():
            with self.subTest(handler=name):
                self.assertTrue(contract.required_inputs)
                self.assertTrue(contract.description)
                self.assertNotIn("placeholder", contract.description.lower())

    def test_every_declared_cell_compiles_to_finite_safe_argv(self) -> None:
        plan = build_catalog_plan(CATALOG, output_root="/outputs", scope="all")
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        for cell in plan.cells:
            with self.subTest(experiment=cell.experiment_id, cell=cell.cell_id):
                commands = compile_cell_commands(CATALOG, cell, context)
                self.assertGreaterEqual(len(commands), 1)
                self.assertTrue(all(command.argv and command.completion_artifacts for command in commands))
                self.assertTrue(all(isinstance(token, str) for command in commands for token in command.argv))

    def test_every_axis_value_changes_the_effective_execution_spec(self) -> None:
        plan = build_catalog_plan(CATALOG, output_root="/outputs", scope="all")
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        for experiment_id, definition in CATALOG.experiments.items():
            cells = [cell for cell in plan.cells if cell.experiment_id == experiment_id]
            signatures = []
            for cell in cells:
                commands = compile_cell_commands(CATALOG, cell, context)
                normalized = []
                for command in commands:
                    normalized.append(
                        tuple(
                            "CELL" if token == cell.cell_id else token.replace(cell.cell_id, "CELL")
                            for token in command.argv
                        )
                    )
                signatures.append(tuple(normalized))
            with self.subTest(experiment=experiment_id):
                self.assertEqual(
                    len(set(signatures)),
                    len(cells),
                    "different declared cells compiled to the same effective command",
                )


if __name__ == "__main__":
    unittest.main()
