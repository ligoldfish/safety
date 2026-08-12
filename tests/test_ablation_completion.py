from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ablations.catalog import load_catalog
from src.ablations.completion import CompletionError, collect_training_contract


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class TrainingCompletionTests(unittest.TestCase):
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
                "target_permutation_manifests": {"train": str(permutation_path)},
            },
        )
        _write_json(
            phase1 / "training" / "eval_suite" / "epoch_002" / "pan_results.json",
            {
                "status": "ok",
                "num_samples": 2,
                "generations": [
                    {"sample_id": "a", "response": "no", "refused": True},
                    {"sample_id": "b", "response": "yes", "refused": False},
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
        return phase1

    def test_every_training_contract_is_derived_from_nonempty_backend_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = self._pipeline(root)
            for experiment_id, definition in CATALOG.experiments.items():
                if definition.execution_kind.value != "train":
                    continue
                output = root / "contracts" / experiment_id
                with self.subTest(experiment=experiment_id):
                    collect_training_contract(
                        output,
                        definition.completion_artifacts,
                        phase1,
                        cell_spec={"experiment_id": experiment_id, "axes": {"seed": 42}},
                    )
                    for name in definition.completion_artifacts:
                        self.assertGreater((output / name).stat().st_size, 0)
                    if "eval_predictions.jsonl" in definition.completion_artifacts:
                        rows = [
                            json.loads(line)
                            for line in (output / "eval_predictions.jsonl").read_text(encoding="utf-8").splitlines()
                        ]
                        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])

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


if __name__ == "__main__":
    unittest.main()
