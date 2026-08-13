from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.ablations.fairness import FairnessLedgerError, load_search_ledger_snapshot
from src.ablations.fairness_search import (
    FairnessSearchError,
    build_fairness_search_trials,
    collect_fairness_search_ledger,
    compile_fairness_search_command,
    compile_fairness_judge_command,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class FairnessSearchTests(unittest.TestCase):
    def _materialize_trial(self, trial, *, metric_bias: float = 0.0) -> None:
        phase1 = trial.output_dir / "pipeline" / "phase1"
        values = trial.hyperparameters
        _write_json(
            phase1 / "layer_analysis" / "teacher_key_layers.json",
            {"top_k": values["top_k"], "key_layers": [1]},
        )
        _write_json(
            phase1 / "safe_subspaces" / "manifest.json",
            {
                "energy_threshold": values["energy_threshold"],
                "rank_cap": values["rank_cap"],
            },
        )
        _write_json(
            phase1 / "training" / "manifest.json",
            {
                "epochs": values["epochs"],
                "epochs_completed": values["epochs"],
                "layer_loss_weight": values["layer_loss_weight"],
                "target_mode": "random_same_norm" if trial.method == "random" else "semantic",
                "val_split": str(phase1 / "validation.jsonl"),
                "train_split": str(phase1 / "train.jsonl"),
            },
        )
        (phase1 / "validation.jsonl").write_text(
            '{"id":"x","label":"harmful","messages":[{"role":"user","content":"validation harmful"}]}\n'
            '{"id":"y","label":"harmless","messages":[{"role":"user","content":"validation harmless"}]}\n', encoding="utf-8")
        (phase1 / "train.jsonl").write_text(
            '{"id":"t","label":"harmful","messages":[{"role":"user","content":"training harmful"}]}\n'
            '{"id":"u","label":"harmless","messages":[{"role":"user","content":"training harmless"}]}\n', encoding="utf-8")
        epoch = values["epochs"]
        hr = 0.8 + metric_bias
        over_refusal = 0.2
        generations = phase1 / "training" / "logs" / "val_generations" / f"epoch_{epoch:03d}.json"
        _write_json(generations, {"generations": [
            {"id": "x", "label": "harmful"}, {"id": "y", "label": "harmless"}
        ]})
        _write_json(generations.with_name(f"epoch_{epoch:03d}.wildguard.json"), {
            "judge": "wildguard", "judge_parse_rate": 1.0, "num_unmatched_ids": 0,
            "num_generations": 2, "pan_results": str(generations.resolve()),
            "judge_num_harmful_scored": 1, "judge_num_harmless_scored": 1,
            "llm_judge_refusal_rate": hr, "llm_judge_over_refusal": over_refusal,
            "generations": [
                {"id": "x", "label": "harmful"}, {"id": "y", "label": "harmless"}
            ],
        })

    def test_plan_has_two_equal_candidates_for_each_dataset_and_method(self) -> None:
        trials = build_fairness_search_trials(Path("/persistent/fairness-search"))
        self.assertEqual(len(trials), 12)
        self.assertEqual({trial.dataset for trial in trials}, {"wildjailbreak", "wildguardmix"})
        self.assertEqual({trial.method for trial in trials}, {"sft1", "random", "ours"})
        for dataset in ("wildjailbreak", "wildguardmix"):
            for method in ("sft1", "random", "ours"):
                selected = [
                    trial
                    for trial in trials
                    if trial.dataset == dataset and trial.method == method
                ]
                self.assertEqual({trial.candidate for trial in selected}, {"global", "historical_override"})
                self.assertEqual(len(selected), 2)

    def test_compiled_trial_reuses_the_real_worker_and_disables_implicit_overrides(self) -> None:
        trial = build_fairness_search_trials(Path("/persistent/fairness-search"))[0]
        command = compile_fairness_search_command(
            trial,
            project_root=ROOT,
            python_executable="python",
            device="npu",
            device_id=0,
        )
        self.assertEqual(command[0], "python")
        self.assertEqual(Path(command[1]).name, "30_run_ablation_cell.py")
        self.assertIn("--disable-dataset-overrides", command)
        self.assertIn("--skip-test-eval", command)
        self.assertIn(f"--dataset={trial.dataset}", command)
        self.assertIn(f"--method={trial.method}", command)
        phasef = json.loads(next(token for token in command if token.startswith("--phasef-updates=")).split("=", 1)[1])
        extras = json.loads(next(token for token in command if token.startswith("--phase1-stage-extras=")).split("=", 1)[1])
        self.assertEqual(phasef["optim.epochs"], trial.hyperparameters["epochs"])
        self.assertEqual(phasef["optim.layer_loss_weight"], trial.hyperparameters["layer_loss_weight"])
        self.assertEqual(extras["analyze"], ["--top-k", str(trial.hyperparameters["top_k"])])
        with tempfile.TemporaryDirectory() as td:
            local_trial = build_fairness_search_trials(Path(td))[0]
            self._materialize_trial(local_trial)
            judge = compile_fairness_judge_command(local_trial, project_root=ROOT,
                python_executable="python", judge_model="/models/wildguard", device="npu", device_id=0)
            self.assertEqual(Path(judge[1]).name, "22_judge_generations.py")
            self.assertIn("--test-jsonl", judge)

    def test_collect_builds_auditable_ledger_from_final_validation_epoch_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trials = build_fairness_search_trials(root / "runs")
            for trial in trials:
                self._materialize_trial(
                    trial,
                    metric_bias=0.05 if trial.candidate == "historical_override" else 0.0,
                )
            ledger = root / "search-ledger.jsonl"
            collect_fairness_search_ledger(trials, ledger)
            rows, _ = load_search_ledger_snapshot(ledger)
            validation = [row for row in rows if row["config"] == "validation_selected"]
            global_rows = [row for row in rows if row["config"] == "global"]
        self.assertEqual(len(rows), 30)
        self.assertEqual(len(global_rows), 18)
        self.assertEqual(len(validation), 12)
        self.assertTrue(all(row["selection_split"] == "validation" for row in validation))
        self.assertTrue(all(row["selection_metric"] == "wildguard_refusal_minus_over_refusal" for row in validation))
        self.assertTrue(all(row["validation_metric"] == row["validation_harmful_refusal"] - row["validation_over_refusal"] for row in validation))
        self.assertEqual(
            {row["candidate"] for row in validation if row["selected"]},
            {"historical_override"},
        )

    def test_tampered_validation_file_or_backend_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trials = build_fairness_search_trials(root / "runs")
            for trial in trials:
                self._materialize_trial(trial)
            ledger = root / "search-ledger.jsonl"
            collect_fairness_search_ledger(trials, ledger)
            epoch = trials[0].hyperparameters["epochs"]
            validation_path = trials[0].output_dir / "pipeline" / "phase1" / "training" / "logs" / "val_generations" / f"epoch_{epoch:03d}.wildguard.json"
            _write_json(validation_path, {"judge": "wildguard"})
            with self.assertRaisesRegex(FairnessLedgerError, "changed"):
                load_search_ledger_snapshot(ledger)

            self._materialize_trial(trials[0])
            broken = trials[1]
            manifest = broken.output_dir / "pipeline" / "phase1" / "training" / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["epochs"] += 1
            _write_json(manifest, payload)
            with self.assertRaisesRegex(FairnessSearchError, "backend"):
                collect_fairness_search_ledger(trials, root / "broken-ledger.jsonl")

    def test_rehashed_forged_backend_evidence_is_still_rejected_semantically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trials = build_fairness_search_trials(root / "runs")
            for trial in trials:
                self._materialize_trial(trial)
            ledger = root / "search-ledger.jsonl"
            collect_fairness_search_ledger(trials, ledger)
            rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            target = next(row for row in rows if row["config"] == "validation_selected")
            layer_path = Path(target["evidence"]["layer"]["path"])
            _write_json(layer_path, {"top_k": 99, "key_layers": [1]})
            target["evidence"]["layer"]["sha256"] = hashlib.sha256(layer_path.read_bytes()).hexdigest()
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(FairnessLedgerError, "backend evidence"):
                load_search_ledger_snapshot(ledger)

    def test_train_validation_prompt_overlap_is_rejected_even_with_valid_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trials = build_fairness_search_trials(root / "runs")
            for trial in trials:
                self._materialize_trial(trial)
            ledger = root / "search-ledger.jsonl"
            collect_fairness_search_ledger(trials, ledger)
            rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            target = next(row for row in rows if row["config"] == "validation_selected")
            train_path = Path(target["evidence"]["train_split"]["path"])
            val_path = Path(target["evidence"]["validation_split"]["path"])
            train_path.write_bytes(val_path.read_bytes())
            target["evidence"]["train_split"]["sha256"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(FairnessLedgerError, "split leakage"):
                load_search_ledger_snapshot(ledger)


if __name__ == "__main__":
    unittest.main()
