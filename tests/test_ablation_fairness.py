from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.ablations.fairness import (
    FairnessLedgerError,
    resolve_fairness_configuration,
    validate_search_ledger_rows,
    load_search_ledger_snapshot,
)
from tests.fairness_evidence import attach_validation_evidence


METHODS = ("sft1", "random", "ours")
GLOBAL_FIXED = {
    "top_k": 5,
    "energy_threshold": 0.8,
    "rank_cap": 32,
    "layer_loss_weight": 0.25,
    "epochs": 3,
}


def _row(
    method: str,
    index: int,
    *,
    dataset: str = "wildjailbreak",
    config: str = "validation_selected",
    selected: bool = False,
    hyperparameters: dict | None = None,
) -> dict:
    row = {
        "trial_id": f"{dataset}-{config}-{method}-{index}",
        "dataset": dataset,
        "config": config,
        "method": method,
        "selection_split": "validation",
        "selected": selected,
        "validation_metric": 0.7 + index / 100,
    }
    if hyperparameters is not None:
        row["hyperparameters"] = hyperparameters
    return row


def _winner(method: str) -> dict:
    return {
        "top_k": 3,
        "energy_threshold": 0.7,
        "rank_cap": 8,
        "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
        "epochs": 5,
    }


class FairnessLedgerTests(unittest.TestCase):
    def test_ledger_snapshot_parses_and_hashes_one_atomic_byte_read(self) -> None:
        rows = [
            _row(
                method,
                0,
                dataset="pan",
                config="global",
                hyperparameters={
                    **GLOBAL_FIXED,
                    "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                },
            )
            for method in METHODS
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        with patch.object(Path, "read_bytes", autospec=True, return_value=payload) as read:
            parsed, digest = load_search_ledger_snapshot(Path("ledger.jsonl"))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(digest, __import__("hashlib").sha256(payload).hexdigest())
        self.assertEqual(read.call_count, 1)

    def _ledger(self, root: Path) -> Path:
        rows = [
            _row(method, index, selected=index == 1, hyperparameters=_winner(method))
            for method in METHODS
            for index in range(2)
        ]
        attach_validation_evidence(rows, root)
        path = root / "search-ledger.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_global_configuration_is_the_exact_preregistered_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "global.jsonl"
            fixed = dict(GLOBAL_FIXED)
            rows = [
                _row(
                    method,
                    0,
                    dataset="pan",
                    config="global",
                    selected=False,
                    hyperparameters={
                        **fixed,
                        "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                    },
                )
                for method in METHODS
            ]
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            configuration = resolve_fairness_configuration(
                {
                    "experiment_id": "P0-07",
                    "axes": {"dataset": "pan", "config": "global", "method": "ours"},
                    "inputs": {"search_ledger": str(ledger)},
                }
            )
        self.assertEqual(
            configuration.hyperparameters,
            {
                "top_k": 5,
                "energy_threshold": 0.8,
                "rank_cap": 32,
                "layer_loss_weight": 0.25,
                "epochs": 3,
            },
        )
        self.assertEqual(
            configuration.phase1_stage_extras,
            {
                "analyze": ["--top-k", "5"],
                "subspace": ["--energy-threshold", "0.8", "--rank-cap", "32"],
            },
        )
        self.assertEqual(
            configuration.phasef_updates,
            {"optim.layer_loss_weight": 0.25, "optim.epochs": 3},
        )
        self.assertIsNone(configuration.selected_trial_id)

    def test_matched_sft_global_configuration_keeps_layer_loss_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "global.jsonl"
            rows = [
                _row(
                    method,
                    0,
                    dataset="pan",
                    config="global",
                    hyperparameters={
                        **GLOBAL_FIXED,
                        "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                    },
                )
                for method in METHODS
            ]
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            configuration = resolve_fairness_configuration(
                {
                    "experiment_id": "P0-07",
                    "axes": {"dataset": "pan", "config": "global", "method": "sft1"},
                    "inputs": {"search_ledger": str(ledger)},
                }
            )
        self.assertEqual(configuration.hyperparameters["layer_loss_weight"], 0.0)
        self.assertEqual(configuration.phasef_updates["optim.layer_loss_weight"], 0.0)

    def test_validation_winner_for_the_current_method_drives_real_updates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = self._ledger(Path(td))
            configuration = resolve_fairness_configuration(
                {
                    "experiment_id": "P0-07",
                    "axes": {
                        "dataset": "wildjailbreak",
                        "config": "validation_selected",
                        "method": "random",
                    },
                    "inputs": {"search_ledger": str(ledger)},
                }
            )
        self.assertEqual(configuration.selected_trial_id, "wildjailbreak-validation_selected-random-1")
        self.assertEqual(configuration.hyperparameters, _winner("random"))
        self.assertEqual(configuration.phasef_updates["optim.epochs"], 5)
        self.assertEqual(configuration.phasef_updates["optim.layer_loss_weight"], 0.25)
        self.assertEqual(configuration.phase1_stage_extras["analyze"], ["--top-k", "3"])

    def test_validation_ledger_rejects_unsafe_or_incomplete_winner_parameters(self) -> None:
        base = [_row(method, 0, selected=True, hyperparameters=_winner(method)) for method in METHODS]
        bad_cases = []
        unsafe = [dict(row) for row in base]
        unsafe[0] = {**unsafe[0], "hyperparameters": {**unsafe[0]["hyperparameters"], "model.path": "/tmp/evil"}}
        bad_cases.append(unsafe)
        missing = [dict(row) for row in base]
        missing[1] = {**missing[1], "hyperparameters": {"top_k": 4}}
        bad_cases.append(missing)
        bad_range = [dict(row) for row in base]
        bad_range[2] = {**bad_range[2], "hyperparameters": {**bad_range[2]["hyperparameters"], "energy_threshold": 1.2}}
        bad_cases.append(bad_range)
        wrong_sft = [dict(row) for row in base]
        wrong_sft[0] = {**wrong_sft[0], "hyperparameters": {**wrong_sft[0]["hyperparameters"], "layer_loss_weight": 0.25}}
        bad_cases.append(wrong_sft)
        for index, rows in enumerate(bad_cases):
            with self.subTest(case=index), self.assertRaises(FairnessLedgerError):
                validate_search_ledger_rows(rows)

    def test_validation_ledger_requires_exact_methods_equal_counts_and_one_winner(self) -> None:
        valid = [
            _row(method, index, selected=index == 1, hyperparameters=_winner(method))
            for method in METHODS
            for index in range(2)
        ]
        cases = (
            [row for row in valid if row["method"] != "random"],
            valid + [_row("ours", 2)],
            [{**row, "selected": False} if row["method"] == "ours" else row for row in valid],
            [{**row, "selection_split": "test"} if row["trial_id"].endswith("ours-0") else row for row in valid],
        )
        for index, rows in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(FairnessLedgerError):
                validate_search_ledger_rows(rows)

    def test_validation_selected_is_limited_to_the_two_historical_override_corpora(self) -> None:
        for dataset in ("pan", "safety_tuned_llamas", "coconot", "c5"):
            rows = [
                _row(
                    method,
                    0,
                    dataset=dataset,
                    selected=True,
                    hyperparameters=_winner(method),
                )
                for method in METHODS
            ]
            with self.subTest(dataset=dataset), self.assertRaisesRegex(
                FairnessLedgerError,
                "historical override",
            ):
                validate_search_ledger_rows(rows)

    def test_validation_trials_require_identical_search_space_except_sft_lambda(self) -> None:
        rows = [
            _row(method, index, selected=index == 1, hyperparameters=_winner(method))
            for method in METHODS
            for index in range(2)
        ]
        rows[-2] = {
            **rows[-2],
            "hyperparameters": {**rows[-2]["hyperparameters"], "rank_cap": 64},
        }
        with self.assertRaisesRegex(FairnessLedgerError, "search space"):
            validate_search_ledger_rows(rows)

    def test_random_and_ours_require_the_same_joint_lambda_search_space(self) -> None:
        rows = []
        for method in METHODS:
            for index, (top_k, weight) in enumerate(((3, 0.25), (5, 0.5))):
                if method == "ours":
                    weight = (0.5, 0.25)[index]
                rows.append(
                    _row(
                        method,
                        index,
                        selected=index == 1,
                        hyperparameters={
                            **_winner(method),
                            "top_k": top_k,
                            "layer_loss_weight": 0.0 if method == "sft1" else weight,
                        },
                    )
                )
        with self.assertRaisesRegex(FairnessLedgerError, "search space"):
            validate_search_ledger_rows(rows)

    def test_selected_trial_must_have_the_best_validation_metric_for_each_method(self) -> None:
        rows = [
            {
                **_row(method, index, selected=index == 0, hyperparameters=_winner(method)),
                "validation_metric": 0.7 + index / 10,
            }
            for method in METHODS
            for index in range(2)
        ]
        with self.assertRaisesRegex(FairnessLedgerError, "best validation"):
            validate_search_ledger_rows(rows)

    def test_global_ledger_rows_are_exact_fixed_configs_and_not_search_selections(self) -> None:
        fixed = {
            "top_k": 5,
            "energy_threshold": 0.8,
            "rank_cap": 32,
            "layer_loss_weight": 0.25,
            "epochs": 3,
        }
        rows = [
            _row(
                method,
                0,
                dataset="pan",
                config="global",
                selected=False,
                hyperparameters={**fixed, "layer_loss_weight": 0.0 if method == "sft1" else 0.25},
            )
            for method in METHODS
        ]
        self.assertEqual(len(validate_search_ledger_rows(rows)), 3)
        for changed in (
            [{**row, "selected": True} if row["method"] == "ours" else row for row in rows],
            rows + [_row("ours", 1, dataset="pan", config="global", hyperparameters=fixed)],
            [{**row, "hyperparameters": {**row["hyperparameters"], "epochs": 4}} if row["method"] == "random" else row for row in rows],
        ):
            with self.assertRaises(FairnessLedgerError):
                validate_search_ledger_rows(changed)


if __name__ == "__main__":
    unittest.main()
