from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.ablations.handlers import HandlerBlocked, execute_handler


class AblationE2ETests(unittest.TestCase):
    def test_provenance_handler_completes_from_real_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "main_table.jsonl").write_text(
                json.dumps(
                    {
                        "cell_id": "c",
                        "model_hash": "m",
                        "dataset_hash": "d",
                        "config_hash": "x",
                        "checkpoint_hash": "k",
                        "commit": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            execute_handler(
                "provenance_matrix",
                {"experiment_id": "P0-01", "axes": {}, "inputs": {"model_registry": str(inputs / "main_table.jsonl"), "dataset_registry": str(inputs / "main_table.jsonl")}},
                output_dir=output,
                required_artifacts=["provenance_matrix.jsonl", "coverage_summary.json"],
            )
            summary = json.loads((output / "coverage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["coverage_rate"], 1.0)

    def test_main_table_provenance_cell_selects_only_its_declared_axes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                {"cell_id": "wanted", "pair": "p1", "dataset": "d1", "method": "ours", "model_hash": "m", "dataset_hash": "d", "config_hash": "c", "checkpoint_hash": "k", "commit": "a"},
                {"cell_id": "other", "pair": "p2", "dataset": "d1", "method": "ours", "model_hash": "m", "dataset_hash": "d", "config_hash": "c", "checkpoint_hash": "k", "commit": "a"},
            ]
            registry = root / "registry.jsonl"
            registry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "out"
            execute_handler(
                "provenance_matrix",
                {"axes": {"pair": "p1", "dataset": "d1", "method": "ours"}, "inputs": {"model_registry": str(registry), "dataset_registry": str(registry)}},
                output_dir=output,
                required_artifacts=["provenance_matrix.jsonl", "coverage_summary.json"],
            )
            result = [json.loads(line) for line in (output / "provenance_matrix.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["cell_id"] for row in result], ["wanted"])

    def test_missing_analysis_asset_is_blocked_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "out"
            with self.assertRaisesRegex(HandlerBlocked, "missing"):
                execute_handler(
                    "seed_and_paired_bootstrap",
                    {"experiment_id": "P0-04", "axes": {"seed": 42}, "inputs": {"aligned_sample_predictions": str(Path(td) / "missing.jsonl")}},
                    output_dir=output,
                    required_artifacts=["paired_bootstrap.json", "seed_summary.json"],
                )
            self.assertFalse(output.exists())

    def test_manual_double_annotation_handler_computes_real_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "wildguard"
            model.mkdir()
            annotations = root / "annotations"
            annotations.mkdir()
            predictions = [
                {"sample_id": "a", "wildguard_label": "unsafe", "stratum": "harmful"},
                {"sample_id": "b", "wildguard_label": "safe", "stratum": "benign"},
            ]
            (annotations / "judge_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
            )
            (annotations / "blind_key.json").write_text(
                json.dumps({"x": {"sample_id": "a", "stratum": "harmful"}, "y": {"sample_id": "b", "stratum": "benign"}}), encoding="utf-8"
            )
            for name, labels in (("rater_a.jsonl", ("unsafe", "safe")), ("rater_b.jsonl", ("unsafe", "safe"))):
                (annotations / name).write_text(
                    json.dumps({"blind_id": "x", "annotation": labels[0]}) + "\n"
                    + json.dumps({"blind_id": "y", "annotation": labels[1]}) + "\n",
                    encoding="utf-8",
                )
            output = root / "out"
            execute_handler(
                "judge_agreement_audit",
                {"axes": {"stratum": "harmful"}, "inputs": {"wildguard_model": str(model), "human_annotations": str(annotations)}},
                output_dir=output,
                required_artifacts=["judge_predictions.jsonl", "manual_audit_summary.json"],
            )
            summary = json.loads((output / "manual_audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stratum"], "harmful")
            self.assertEqual(summary["n"], 1)
            self.assertEqual(summary["human_human_kappa"], 1.0)
            self.assertEqual(summary["wildguard_human_agreement"], 1.0)

    def test_subspace_bootstrap_uses_the_requested_draw_and_real_hidden_shards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hidden = root / "hidden"
            hidden.mkdir()
            torch.save(
                {
                    "feature_type": "first_generated_token_hidden_state",
                    "representation_mode": "last_prompt",
                    "sample_ids": ["a", "b", "c", "d", "e", "f"],
                    "labels": ["harmful", "harmful", "harmful", "harmless", "harmless", "harmless"],
                    "hidden_by_layer": {
                        "0": torch.tensor([[2.0, 0], [2.1, 0], [1.9, 0], [0.0, 0], [0.1, 0], [-0.1, 0]]),
                        "1": torch.tensor([[0, 3.0], [0, 3.1], [0, 2.9], [0, 0.0], [0, 0.1], [0, -0.1]]),
                    },
                },
                hidden / "part_000.pt",
            )
            output = root / "out"
            execute_handler(
                "subspace_bootstrap",
                {"axes": {"draw": 7}, "inputs": {"alignment_hidden_states": str(hidden)}},
                output_dir=output,
                required_artifacts=["bootstrap_stability.json"],
            )
            result = json.loads((output / "bootstrap_stability.json").read_text(encoding="utf-8"))
            self.assertEqual(result["draw"], 7)
            self.assertEqual(result["sample_count"], 6)
            self.assertTrue(result["layers"])
            self.assertIn("baseline_key_layers", result)
            self.assertIn("bootstrap_key_layers", result)
            self.assertIn("layer_jaccard", result)

    def test_cross_corpus_suite_axis_filters_the_common_test(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            rows = [
                {"train_corpus": "pan", "test_suite": "pan_heldout", "score": 0.8},
                {"train_corpus": "pan", "test_suite": "common_safety", "score": 0.6},
            ]
            common = root / "scores.jsonl"
            common.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            wildguard = root / "wildguard"
            wildguard.mkdir()
            output = root / "out"
            execute_handler(
                "cross_corpus_matrix",
                {
                    "axes": {"test_suite": "common_safety"},
                    "inputs": {
                        "checkpoint_registry": str(checkpoints),
                        "common_test": str(common),
                        "wildguard_model": str(wildguard),
                    },
                },
                output_dir=output,
                required_artifacts=["cross_corpus_matrix.json"],
            )
            result = json.loads((output / "cross_corpus_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(result["test_suites"], ["common_safety"])

    def test_pan_grouping_axis_changes_the_real_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                {"sample_id": "a", "unsafe": 1, "attack_family": "jailbreak", "benign_length": "short", "benign_topic": "code"},
                {"sample_id": "b", "unsafe": 0, "attack_family": "direct", "benign_length": "short", "benign_topic": "writing"},
                {"sample_id": "c", "unsafe": 0, "attack_family": "direct", "benign_length": "long", "benign_topic": "code"},
            ]
            predictions = root / "predictions.jsonl"
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metadata = root / "metadata.json"
            metadata.write_text("{}\n", encoding="utf-8")
            results = {}
            for grouping in ("attack_family", "benign_length", "benign_topic"):
                output = root / grouping
                execute_handler(
                    "pan_subgroup_analysis",
                    {"axes": {"grouping": grouping}, "inputs": {"pan_predictions": str(predictions), "pan_metadata": str(metadata)}},
                    output_dir=output,
                    required_artifacts=["pan_subgroups.json"],
                )
                results[grouping] = json.loads((output / "pan_subgroups.json").read_text(encoding="utf-8"))
        self.assertEqual(set(results["attack_family"]), {"direct", "jailbreak"})
        self.assertEqual(set(results["benign_length"]), {"long", "short"})
        self.assertEqual(set(results["benign_topic"]), {"code", "writing"})

    def test_representation_label_axis_filters_before_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hidden = root / "hidden"
            hidden.mkdir()
            rows = [
                {"sample_id": "h1", "label": "harmful", "cosine_pre": 0.1, "cosine_post": 0.8, "behavior_delta": 1.0},
                {"sample_id": "h2", "label": "harmful", "cosine_pre": 0.2, "cosine_post": 0.7, "behavior_delta": 0.5},
                {"sample_id": "b1", "label": "harmless", "cosine_pre": 0.2, "cosine_post": 0.3, "behavior_delta": 0.2},
            ]
            aligned = root / "aligned.jsonl"
            aligned.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "out"
            execute_handler(
                "representation_behavior_analysis",
                {"axes": {"label": "harmful"}, "inputs": {"pre_post_hidden_states": str(hidden), "aligned_predictions": str(aligned)}},
                output_dir=output,
                required_artifacts=["representation_behavior.json"],
            )
            result = json.loads((output / "representation_behavior.json").read_text(encoding="utf-8"))
        self.assertEqual(result["label"], "harmful")
        self.assertEqual(result["n"], 2)

    def test_efficiency_phase_axis_filters_the_requested_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs = root / "runtime.jsonl"
            logs.write_text(
                json.dumps({"phase": "extract", "wall_seconds": 2, "peak_memory_bytes": 10, "memory_measurement": "process_tree_rss", "disk_delta_bytes": 3, "device_hours": 0.1}) + "\n"
                + json.dumps({"phase": "train", "wall_seconds": 5, "peak_memory_bytes": 20, "memory_measurement": "process_tree_rss", "disk_delta_bytes": 7, "device_hours": 0.3}) + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            execute_handler(
                "efficiency_profile",
                {"axes": {"phase": "train"}, "inputs": {"phase_runtime_logs": str(logs)}},
                output_dir=output,
                required_artifacts=["efficiency_profile.json"],
            )
            result = json.loads((output / "efficiency_profile.json").read_text(encoding="utf-8"))
        self.assertEqual(result["phase"], "train")
        self.assertEqual(result["wall_seconds"], 5.0)

    def test_efficiency_handler_preserves_unavailable_memory_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs = root / "runtime.jsonl"
            logs.write_text(
                json.dumps(
                    {
                        "phase": "train",
                        "wall_seconds": 5,
                        "peak_memory_bytes": None,
                        "memory_measurement": "unavailable",
                        "disk_delta_bytes": 7,
                        "device_hours": 0.3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            execute_handler(
                "efficiency_profile",
                {"axes": {"phase": "train"}, "inputs": {"phase_runtime_logs": str(logs)}},
                output_dir=output,
                required_artifacts=["efficiency_profile.json"],
            )
            result = json.loads((output / "efficiency_profile.json").read_text(encoding="utf-8"))
        self.assertIsNone(result["peak_memory_bytes"])
        self.assertFalse(result["memory_complete"])


if __name__ == "__main__":
    unittest.main()
