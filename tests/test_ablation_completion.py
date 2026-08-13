from __future__ import annotations

import json
import tempfile
import unittest
import hashlib
from pathlib import Path
from tests.fairness_evidence import attach_validation_evidence

from src.ablations.catalog import load_catalog
from src.ablations.completion import CompletionError, collect_training_contract


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class TrainingCompletionTests(unittest.TestCase):
    def _write_fairness_backend(
        self,
        phase1: Path,
        *,
        method: str,
        top_k: int,
        energy_threshold: float,
        rank_cap: int,
        layer_loss_weight: float,
        epochs: int,
    ) -> None:
        layer_path = phase1 / "layer_analysis" / "teacher_key_layers.json"
        layer = json.loads(layer_path.read_text(encoding="utf-8"))
        layer["top_k"] = top_k
        _write_json(layer_path, layer)
        subspace_path = phase1 / "safe_subspaces" / "manifest.json"
        subspace = json.loads(subspace_path.read_text(encoding="utf-8"))
        subspace.update(energy_threshold=energy_threshold, rank_cap=rank_cap)
        _write_json(subspace_path, subspace)
        training_path = phase1 / "training" / "manifest.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training.update(
            epochs=epochs,
            epochs_completed=epochs,
            layer_loss_weight=layer_loss_weight,
            target_mode="random_same_norm" if method == "random" else "semantic",
        )
        _write_json(training_path, training)

    def _pipeline(self, root: Path) -> Path:
        phase1 = root / "pipeline" / "phase1"
        permutation_path = phase1 / "training" / "target_permutation_train.json"
        _write_json(permutation_path, {"a": "b", "b": "a"})
        _write_json(
            phase1 / "training" / "manifest.json",
            {
                "trainable_parameters": 123,
                "total_parameters": 1000,
                "epochs_completed": 2,
                "train_num_samples": 20,
                "optimizer_steps": 8,
                "training_tokens_seen": 2048,
                "target_permutation_manifests": {"train": str(permutation_path)},
            },
        )
        _write_json(
            phase1 / "training" / "eval_suite" / "epoch_002" / "pan_results.json",
            {
                "status": "ok",
                "num_samples": 2,
                "harmful_refusal_rate": 0.5,
                "harmful_unsafe_output_rate": 0.5,
                "harmless_over_refusal_rate": 0.0,
                "generations": [
                    {"sample_id": "a", "id": "a", "label": "harmful", "final_text": "no", "is_refusal": True},
                    {"sample_id": "b", "id": "b", "label": "harmless", "final_text": "yes", "is_refusal": False},
                ],
            },
        )
        _write_json(phase1 / "training" / "eval_suite" / "epoch_002" / "final_summary.json", {"results": {"pan": {"asr": 0.5}}})
        _write_json(phase1 / "layer_analysis" / "teacher_key_layers.json", {"key_layers": [1, 3], "selection_metric": "effect_probe_sum"})
        _write_json(phase1 / "safe_subspaces" / "manifest.json", {"key_layers": [1, 3], "rank": 8, "files": ["layer_1.pt"]})
        _write_json(phase1 / "layer_pairing" / "teacher_student_layer_pairs.json", {"pairs": [{"teacher_layer": 1, "student_layer": 0}]})
        _write_json(phase1 / "semantic_bases" / "manifest.json", {"bridge_mode": "token_string", "bridge_artifact_path": "bridge_artifact.pt"})
        _write_json(phase1 / "semantic_bases" / "vocab_index_map.json", {"bridge_mode": "token_string", "vocab_size": 10, "tokenizer_shared": False})
        (phase1 / "semantic_bases" / "bridge_artifact.pt").write_bytes(b"real-torch-artifact")
        _write_json(phase1 / "hidden_states" / "teacher_alignment" / "manifest.json", {"sample_count": 20, "representation_mode": "last_prompt"})
        _write_json(phase1 / "semantic_coeffs_teacher_alignment" / "manifest.json", {"top_k": 128, "selection_mode": "abs"})
        failure = phase1 / "training" / "failure_boundary"
        training_jsonl = failure / "wildjailbreak_train.jsonl"
        training_jsonl.parent.mkdir(parents=True, exist_ok=True)
        training_jsonl.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (
                    {
                        "id": "train-a",
                        "label": "harmful",
                        "target_response": "I cannot help with that request.",
                        "metadata": {"data_type": "vanilla_harmful", "target_source": "upstream_completion"},
                    },
                    {
                        "id": "train-b",
                        "label": "harmful",
                        "target_response": "I cannot provide instructions for wrongdoing.",
                        "metadata": {"data_type": "adversarial_harmful", "target_source": "template_pool"},
                    },
                    {
                        "id": "train-c",
                        "label": "harmless",
                        "target_response": "Here is a benign answer.",
                        "metadata": {"data_type": "vanilla_benign", "target_source": "upstream_completion"},
                    },
                )
            ),
            encoding="utf-8",
        )
        common_result = failure / "common_eval" / "pan_results.json"
        _write_json(
            common_result,
            {
                "status": "ok",
                "harmful_refusal_rate": 0.75,
                "harmful_unsafe_output_rate": 0.25,
                "harmless_over_refusal_rate": 0.1,
                "generations": [{"id": "common-a", "label": "harmful", "final_text": "no", "is_refusal": True}],
            },
        )
        target_judge = failure / "target_judge_results.json"
        common_judge = failure / "common_judge_results.json"
        for path, asr in ((target_judge, 0.4), (common_judge, 0.2)):
            _write_json(
                path,
                {
                    "judge": "wildguard",
                    "judge_parse_rate": 1.0,
                    "num_unmatched_ids": 0,
                    "llm_judge_asr": asr,
                    "llm_judge_refusal_rate": 1.0 - asr,
                    "llm_judge_over_refusal": 0.1,
                },
            )
        checkpoint = phase1 / "training" / "checkpoints" / "epoch_002.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"real-adapter")
        target_test = failure / "wildjailbreak_test.jsonl"
        common_test = failure / "common_safety.jsonl"
        target_test.write_text(
            json.dumps({"id": "a", "label": "harmful", "prompt": "held out target"}) + "\n",
            encoding="utf-8",
        )
        common_test.write_text(
            json.dumps({"id": "common-a", "label": "harmful", "prompt": "common prompt"}) + "\n",
            encoding="utf-8",
        )
        _write_json(
            failure / "evaluation_manifest.json",
            {
                "schema_version": 1,
                "experiment_id": "P0-06",
                "pair": "qwen35_9b_to_08b",
                "config": "global",
                "curation": "off",
                "method": "ours",
                "training_jsonl": str(training_jsonl),
                "curation_summary": str(failure / "curation_summary.json"),
                "target_result": str(phase1 / "training" / "eval_suite" / "epoch_002" / "pan_results.json"),
                "target_judge": str(target_judge),
                "common_result": str(common_result),
                "common_judge": str(common_judge),
                "target_test_jsonl": str(target_test),
                "common_test_jsonl": str(common_test),
                "adapter_checkpoint": str(checkpoint),
                "adapter_checkpoint_sha256": hashlib.sha256(b"real-adapter").hexdigest(),
                "dataset_sha256": {
                    "training": hashlib.sha256(training_jsonl.read_bytes()).hexdigest(),
                    "target_test": hashlib.sha256(target_test.read_bytes()).hexdigest(),
                    "common_test": hashlib.sha256(common_test.read_bytes()).hexdigest(),
                },
                "split_audit": {
                    "train_target_overlap": 0,
                    "train_common_overlap": 0,
                    "common_target_overlap": 0,
                },
            },
        )
        _write_json(failure / "curation_summary.json", {"baseline": "wildjailbreak", "mode": "off", "output_count": 3})
        return phase1

    def test_every_training_contract_is_derived_from_nonempty_backend_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            search_ledger = root / "search-ledger.jsonl"
            search_ledger.write_text(
                "".join(
                    json.dumps(
                        {
                            "trial_id": f"pan-global-{method}",
                            "dataset": "pan",
                            "config": "global",
                            "method": method,
                            "selection_split": "validation",
                            "selected": False,
                            "validation_metric": 0.8,
                            "hyperparameters": {
                                "top_k": 5,
                                "energy_threshold": 0.8,
                                "rank_cap": 32,
                                "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                                "epochs": 3,
                            },
                        }
                    )
                    + "\n"
                    for method in ("ours", "sft1", "random")
                ),
                encoding="utf-8",
            )
            search_ledger_hash = hashlib.sha256(search_ledger.read_bytes()).hexdigest()
            for experiment_id, definition in CATALOG.experiments.items():
                if definition.execution_kind.value != "train":
                    continue
                output = root / "contracts" / experiment_id
                with self.subTest(experiment=experiment_id):
                    if experiment_id == "P0-07":
                        self._write_fairness_backend(
                            phase1,
                            method="ours",
                            top_k=5,
                            energy_threshold=0.8,
                            rank_cap=32,
                            layer_loss_weight=0.25,
                            epochs=3,
                        )
                    collect_training_contract(
                        output,
                        definition.completion_artifacts,
                        phase1,
                        cell_spec={
                            "experiment_id": experiment_id,
                            "axes": (
                                {"dataset": "pan", "config": "global", "method": "ours"}
                                if experiment_id == "P0-07"
                                else (
                                    {
                                        "pair": "qwen35_9b_to_08b",
                                        "config": "global",
                                        "curation": "off",
                                        "method": "ours",
                                    }
                                    if experiment_id == "P0-06"
                                    else {"seed": 42}
                                )
                            ),
                            "inputs": {"search_ledger": str(search_ledger)},
                            **(
                                {
                                    "fairness_configuration": {
                                        "hyperparameters": {
                                            "top_k": 5,
                                            "energy_threshold": 0.8,
                                            "rank_cap": 32,
                                            "layer_loss_weight": 0.25,
                                            "epochs": 3,
                                        },
                                        "selected_trial_id": None,
                                        "search_ledger_sha256": search_ledger_hash,
                                    }
                                }
                                if experiment_id == "P0-07"
                                else {}
                            ),
                        },
                    )
                    for name in definition.completion_artifacts:
                        self.assertGreater((output / name).stat().st_size, 0)
                    if "eval_predictions.jsonl" in definition.completion_artifacts:
                        rows = [
                            json.loads(line)
                            for line in (output / "eval_predictions.jsonl").read_text(encoding="utf-8").splitlines()
                        ]
                        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])

    def test_failure_boundary_requires_both_real_evaluations_and_reports_declared_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            output = root / "out"
            spec = {
                "experiment_id": "P0-06",
                "axes": {
                    "pair": "qwen35_9b_to_08b",
                    "config": "global",
                    "curation": "off",
                    "method": "ours",
                },
            }
            collect_training_contract(
                output,
                ["failure_analysis.json", "eval_predictions.jsonl"],
                phase1,
                cell_spec=spec,
            )
            report = json.loads((output / "failure_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"]["target_refusal_rate"], 1.0)
            self.assertEqual(report["metrics"]["template_diversity"], 1.0)
            self.assertEqual(report["metrics"]["common_test_asr"], 0.2)
            self.assertEqual(report["data_audit"]["template_pool_count"], 1)
            self.assertEqual(report["data_audit"]["data_type_distribution"]["adversarial_harmful"], 1)
            self.assertEqual(report["target_evaluation"]["wildguard"]["payload"]["llm_judge_asr"], 0.4)
            self.assertEqual(report["common_evaluation"]["keyword"]["payload"]["harmful_unsafe_output_rate"], 0.25)

            manifest = phase1 / "training" / "failure_boundary" / "evaluation_manifest.json"
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            checkpoint = phase1 / "training" / "epoch_002.pt"
            checkpoint.write_bytes(b"expected-checkpoint")
            raw["adapter_checkpoint"] = str(checkpoint)
            raw["adapter_checkpoint_sha256"] = "0" * 64
            manifest.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CompletionError, "checkpoint"):
                collect_training_contract(
                    root / "changed-checkpoint-out",
                    ["failure_analysis.json"],
                    phase1,
                    cell_spec=spec,
                )

            raw["adapter_checkpoint_sha256"] = hashlib.sha256(b"expected-checkpoint").hexdigest()
            manifest.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            Path(raw["common_result"]).unlink()
            with self.assertRaisesRegex(CompletionError, "common"):
                collect_training_contract(
                    root / "bad-out",
                    ["failure_analysis.json"],
                    phase1,
                    cell_spec=spec,
                )

    def test_missing_real_predictions_fails_instead_of_fabricating_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            phase1 = self._pipeline(Path(td))
            (phase1 / "training" / "eval_suite" / "epoch_002" / "pan_results.json").unlink()
            with self.assertRaisesRegex(CompletionError, "prediction"):
                collect_training_contract(
                    Path(td) / "out",
                    ["training_manifest.json", "eval_predictions.jsonl"],
                    phase1,
                    cell_spec={"experiment_id": "P1-11", "axes": {}},
                )

    def test_search_budget_uses_real_validation_ledger_and_exact_training_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            ledger = root / "search.jsonl"
            rows = [
                {
                    "trial_id": f"{method}-{index}",
                    "dataset": "wildjailbreak",
                    "config": "validation_selected",
                    "method": method,
                    "selection_split": "validation",
                    "selected": index == 1,
                    "validation_metric": 0.5 + index / 10,
                    "hyperparameters": {
                        "top_k": 3,
                        "energy_threshold": 0.7,
                        "rank_cap": 8,
                        "layer_loss_weight": 0.0 if method == "sft1" else 0.5,
                        "epochs": 5,
                    },
                }
                for method in ("ours", "sft1", "random")
                for index in range(2)
            ]
            attach_validation_evidence(rows, root)
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
            output = root / "out"
            self._write_fairness_backend(
                phase1,
                method="random",
                top_k=3,
                energy_threshold=0.7,
                rank_cap=8,
                layer_loss_weight=0.25,
                epochs=5,
            )
            collect_training_contract(
                output,
                ["search_ledger.jsonl", "budget_summary.json"],
                phase1,
                cell_spec={
                    "experiment_id": "P0-07",
                    "axes": {
                        "dataset": "wildjailbreak",
                        "config": "validation_selected",
                        "method": "random",
                    },
                    "inputs": {"search_ledger": str(ledger)},
                    "fairness_configuration": {
                        "hyperparameters": {
                            "top_k": 3,
                            "energy_threshold": 0.7,
                            "rank_cap": 8,
                            "layer_loss_weight": 0.25,
                            "epochs": 5,
                        },
                        "selected_trial_id": "random-1",
                        "search_ledger_sha256": ledger_hash,
                    },
                },
            )
            copied = [json.loads(line) for line in (output / "search_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
            summary = json.loads((output / "budget_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(copied, rows)
        self.assertEqual(summary["search_count"], 6)
        self.assertEqual(summary["search_count_by_method"], {"ours": 2, "random": 2, "sft1": 2})
        self.assertEqual(summary["training_budget"]["trainable_parameters"], 123)
        self.assertEqual(summary["training_budget"]["optimizer_steps"], 8)
        self.assertEqual(summary["training_budget"]["training_tokens_seen"], 2048)
        self.assertEqual(
            summary["selected_trial_ids"],
            {"ours": "ours-1", "random": "random-1", "sft1": "sft1-1"},
        )
        self.assertEqual(summary["current_method"], "random")
        self.assertEqual(summary["applied_trial_id"], "random-1")
        self.assertEqual(summary["applied_hyperparameters"]["epochs"], 5)

    def test_search_budget_rejects_backend_that_did_not_apply_the_declared_winner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            self._write_fairness_backend(
                phase1,
                method="ours",
                top_k=5,
                energy_threshold=0.8,
                rank_cap=32,
                layer_loss_weight=0.25,
                epochs=3,
            )
            ledger = root / "selected.jsonl"
            rows = []
            winner = {
                "top_k": 3,
                "energy_threshold": 0.7,
                "rank_cap": 8,
                "layer_loss_weight": 0.25,
                "epochs": 5,
            }
            for method in ("sft1", "random", "ours"):
                rows.append(
                    {
                        "trial_id": f"{method}-winner",
                        "dataset": "wildjailbreak",
                        "config": "validation_selected",
                        "method": method,
                        "selection_split": "validation",
                        "selected": True,
                        "validation_metric": 0.8,
                        "hyperparameters": {
                            **winner,
                            "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                        },
                    }
                )
            attach_validation_evidence(rows, root)
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
            with self.assertRaisesRegex(CompletionError, "backend fairness"):
                collect_training_contract(
                    root / "out",
                    ["search_ledger.jsonl", "budget_summary.json"],
                    phase1,
                    cell_spec={
                        "experiment_id": "P0-07",
                        "axes": {
                            "dataset": "wildjailbreak",
                            "config": "validation_selected",
                            "method": "ours",
                        },
                        "inputs": {"search_ledger": str(ledger)},
                        "fairness_configuration": {
                            "hyperparameters": winner,
                            "selected_trial_id": "ours-winner",
                            "search_ledger_sha256": ledger_hash,
                        },
                    },
                )

    def test_search_budget_rejects_ledger_changed_after_worker_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            self._write_fairness_backend(
                phase1,
                method="ours",
                top_k=5,
                energy_threshold=0.8,
                rank_cap=32,
                layer_loss_weight=0.25,
                epochs=3,
            )
            ledger = root / "global.jsonl"
            rows = [
                {
                    "trial_id": f"global-{method}",
                    "dataset": "pan",
                    "config": "global",
                    "method": method,
                    "selection_split": "validation",
                    "selected": False,
                    "validation_metric": 0.8,
                    "hyperparameters": {
                        "top_k": 5,
                        "energy_threshold": 0.8,
                        "rank_cap": 32,
                        "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                        "epochs": 3,
                    },
                }
                for method in ("sft1", "random", "ours")
            ]
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            stale_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
            ledger.write_text("".join(json.dumps({**row, "validation_metric": 0.9}) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(CompletionError, "changed after"):
                collect_training_contract(
                    root / "out",
                    ["search_ledger.jsonl", "budget_summary.json"],
                    phase1,
                    cell_spec={
                        "experiment_id": "P0-07",
                        "axes": {"dataset": "pan", "config": "global", "method": "ours"},
                        "inputs": {"search_ledger": str(ledger)},
                        "fairness_configuration": {
                            "hyperparameters": {
                                "top_k": 5,
                                "energy_threshold": 0.8,
                                "rank_cap": 32,
                                "layer_loss_weight": 0.25,
                                "epochs": 3,
                            },
                            "selected_trial_id": None,
                            "search_ledger_sha256": stale_hash,
                        },
                    },
                )
    def test_search_budget_requires_worker_applied_configuration_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            ledger = root / "global.jsonl"
            rows = []
            for method in ("sft1", "random", "ours"):
                rows.append(
                    {
                        "trial_id": f"global-{method}",
                        "dataset": "pan",
                        "config": "global",
                        "method": method,
                        "selection_split": "validation",
                        "selected": False,
                        "validation_metric": 0.8,
                        "hyperparameters": {
                            "top_k": 5,
                            "energy_threshold": 0.8,
                            "rank_cap": 32,
                            "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                            "epochs": 3,
                        },
                    }
                )

    def test_search_budget_rejects_nonpositive_or_impossible_training_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            self._write_fairness_backend(
                phase1,
                method="ours",
                top_k=5,
                energy_threshold=0.8,
                rank_cap=32,
                layer_loss_weight=0.25,
                epochs=3,
            )
            ledger = root / "global.jsonl"
            rows = [
                {
                    "trial_id": f"global-{method}",
                    "dataset": "pan",
                    "config": "global",
                    "method": method,
                    "selection_split": "validation",
                    "selected": False,
                    "validation_metric": 0.0,
                    "hyperparameters": {
                        "top_k": 5,
                        "energy_threshold": 0.8,
                        "rank_cap": 32,
                        "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                        "epochs": 3,
                    },
                }
                for method in ("sft1", "random", "ours")
            ]
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
            spec = {
                "experiment_id": "P0-07",
                "axes": {"dataset": "pan", "config": "global", "method": "ours"},
                "inputs": {"search_ledger": str(ledger)},
                "fairness_configuration": {
                    "hyperparameters": {
                        "top_k": 5,
                        "energy_threshold": 0.8,
                        "rank_cap": 32,
                        "layer_loss_weight": 0.25,
                        "epochs": 3,
                    },
                    "selected_trial_id": None,
                    "search_ledger_sha256": ledger_hash,
                },
            }
            training_path = phase1 / "training" / "manifest.json"
            baseline = json.loads(training_path.read_text(encoding="utf-8"))
            cases = (
                {**baseline, "optimizer_steps": 0},
                {**baseline, "training_tokens_seen": True},
                {**baseline, "trainable_parameters": baseline["total_parameters"] + 1},
            )
            for index, invalid in enumerate(cases):
                with self.subTest(case=index):
                    _write_json(training_path, invalid)
                    with self.assertRaisesRegex(CompletionError, "training budget"):
                        collect_training_contract(
                            root / f"out-{index}",
                            ["search_ledger.jsonl", "budget_summary.json"],
                            phase1,
                            cell_spec=spec,
                        )
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(CompletionError, "applied fairness"):
                collect_training_contract(
                    root / "out",
                    ["search_ledger.jsonl", "budget_summary.json"],
                    phase1,
                    cell_spec={
                        "experiment_id": "P0-07",
                        "axes": {"dataset": "pan", "config": "global", "method": "ours"},
                        "inputs": {"search_ledger": str(ledger)},
                    },
                )

    def test_search_budget_rejects_test_selected_or_unfair_search_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            ledger = root / "bad-search.jsonl"
            base = {
                "dataset": "pan",
                "config": "validation_selected",
                "selection_split": "validation",
                "validation_metric": 0.7,
            }
            bad_cases = (
                [
                    {**base, "trial_id": "ours-0", "method": "ours", "selected": True, "selection_split": "test"},
                    {**base, "trial_id": "sft-0", "method": "sft1", "selected": False},
                ],
                [
                    {**base, "trial_id": "ours-0", "method": "ours", "selected": True},
                    {**base, "trial_id": "ours-1", "method": "ours", "selected": False},
                    {**base, "trial_id": "sft-0", "method": "sft1", "selected": False},
                ],
            )
            for index, rows in enumerate(bad_cases):
                with self.subTest(case=index):
                    ledger.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
                    with self.assertRaises(CompletionError):
                        collect_training_contract(
                            root / f"out-{index}",
                            ["search_ledger.jsonl", "budget_summary.json"],
                            phase1,
                            cell_spec={
                                "experiment_id": "P0-07",
                                "axes": {"dataset": "pan", "config": "validation_selected"},
                                "inputs": {"search_ledger": str(ledger)},
                            },
                        )


if __name__ == "__main__":
    unittest.main()
