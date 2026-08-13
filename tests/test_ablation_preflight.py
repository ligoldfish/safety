from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from src.ablations.preflight import (
    AssetRequirement,
    requirements_from_manifest,
    inspect_model_directory,
    inspect_submission_package,
    run_preflight,
    training_data_requirements,
    training_model_requirements,
)
from src.ablations.catalog import load_catalog
from src.ablations.planner import build_catalog_plan
from tests.fairness_evidence import attach_validation_evidence


ROOT = Path(__file__).resolve().parents[1]


class AblationPreflightTests(unittest.TestCase):
    def test_complete_fairness_ledger_is_ready_for_all_24_declared_cells(self) -> None:
        datasets = (
            "pan",
            "safety_tuned_llamas",
            "coconot",
            "c5",
            "wildjailbreak",
            "wildguardmix",
        )
        methods = ("sft1", "random", "ours")
        rows = []
        for dataset in datasets:
            for method in methods:
                rows.append(
                    {
                        "trial_id": f"{dataset}-global-{method}",
                        "dataset": dataset,
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
                )
                for index, top_k in enumerate((3, 5)) if dataset in {
                    "wildjailbreak",
                    "wildguardmix",
                } else ():
                    rows.append(
                        {
                            "trial_id": f"{dataset}-validation-{method}-{index}",
                            "dataset": dataset,
                            "config": "validation_selected",
                            "method": method,
                            "selection_split": "validation",
                            "selected": index == 1,
                            "validation_metric": 0.7 + index / 10,
                            "hyperparameters": {
                                "top_k": top_k,
                                "energy_threshold": 0.8,
                                "rank_cap": 32,
                                "layer_loss_weight": 0.0 if method == "sft1" else 0.25,
                                "epochs": 3,
                            },
                        }
                    )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            attach_validation_evidence(rows, root)
            ledger = root / "search.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            catalog = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")
            cells = [
                cell
                for cell in build_catalog_plan(catalog, output_root="/out", scope="all").cells
                if cell.experiment_id == "P0-07"
            ]
            self.assertEqual(len(cells), 24)
            for cell in cells:
                requirement = AssetRequirement(
                    "search_ledger",
                    ledger,
                    "file",
                    cell.cell_id,
                    selectors=tuple(sorted((key, str(value)) for key, value in cell.axes.items())),
                )
                with self.subTest(cell=cell.cell_id):
                    self.assertEqual(run_preflight([requirement]).status, "READY")

    @staticmethod
    def _complete_model(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")

    def test_complete_model_passes_and_missing_weights_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            blocked = inspect_model_directory(model)
            self.assertEqual(blocked.status, "BLOCKED")
            self.assertIn("MODEL_WEIGHTS_MISSING", {item.code for item in blocked.issues})
            (model / "model.safetensors").write_bytes(b"weights")
            self.assertEqual(inspect_model_directory(model).status, "READY")

    def test_required_assets_return_structured_blocked_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = run_preflight(
                [
                    AssetRequirement("train_split", root / "missing.jsonl", "file"),
                    AssetRequirement("checkpoint", root / "missing.pt", "file"),
                ]
            )
            self.assertEqual(report.status, "BLOCKED")
            self.assertEqual({item.asset_id for item in report.issues}, {"train_split", "checkpoint"})
            self.assertTrue(all(item.cell_id for item in report.issues))

    def test_submission_package_rejects_symlink_and_size_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package = root / "package"
            package.mkdir()
            (package / "small.py").write_bytes(b"x" * 10)
            oversized = package / "large.bin"
            oversized.write_bytes(b"x" * 20)
            report = inspect_submission_package(package, max_file_bytes=15, max_total_bytes=25)
            self.assertEqual(report.status, "BLOCKED")
            codes = {item.code for item in report.issues}
            self.assertIn("PACKAGE_FILE_TOO_LARGE", codes)
            self.assertIn("PACKAGE_TOTAL_TOO_LARGE", codes)

    def test_submission_package_allows_source_data_module_but_rejects_top_level_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package = Path(tmpdir)
            (package / "src" / "data").mkdir(parents=True)
            (package / "src" / "data" / "loader.py").write_text("pass", encoding="utf-8")
            self.assertEqual(inspect_submission_package(package).status, "READY")
            (package / "models").mkdir()
            self.assertEqual(inspect_submission_package(package).status, "BLOCKED")

    def test_model_index_rejects_missing_weight_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model / "part-1.safetensors").write_bytes(b"one")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}),
                encoding="utf-8",
            )
            report = inspect_model_directory(model)
            self.assertEqual(report.status, "BLOCKED")
            self.assertIn("MODEL_SHARD_MISSING", {item.code for item in report.issues})

    def test_output_requirement_checks_minimum_free_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            requirement = AssetRequirement(
                "run_output",
                Path(tmpdir),
                "output",
                min_free_bytes=2**63,
            )
            report = run_preflight([requirement])
            self.assertEqual(report.status, "BLOCKED")
            self.assertIn("OUTPUT_DISK_INSUFFICIENT", {item.code for item in report.issues})

    def test_report_does_not_echo_secret_environment_values(self) -> None:
        secret = "hf_this_must_never_appear"
        report = run_preflight(
            [AssetRequirement("token", Path("definitely-missing"), "file")],
            environment={"HF_TOKEN": secret, "PATH": "safe"},
        )
        self.assertNotIn(secret, str(report.to_dict()))
        self.assertNotIn("HF_TOKEN", str(report.to_dict()))

    def test_checkpoint_registry_is_semantically_validated_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "merged"
            self._complete_model(model)
            registry = root / "checkpoints.jsonl"
            registry.write_text(
                json.dumps(
                    {
                        "checkpoint_id": "pan-ours",
                        "pair": "qwen35_9b_to_08b",
                        "train_corpus": "pan",
                        "method": "ours",
                        "kind": "merged",
                        "checkpoint_hash": "sha256:abc",
                        "model_path": "merged",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ready = run_preflight(
                [AssetRequirement("checkpoint_registry", registry, "file", "p0-08")]
            )
            registry.write_text(
                json.dumps({"checkpoint_id": "broken", "kind": "merged"}) + "\n",
                encoding="utf-8",
            )
            blocked = run_preflight(
                [AssetRequirement("checkpoint_registry", registry, "file", "p0-08")]
            )
        self.assertEqual(ready.status, "READY")
        self.assertEqual(blocked.status, "BLOCKED")
        self.assertIn("CHECKPOINT_REGISTRY_INVALID", {issue.code for issue in blocked.issues})

    def test_search_ledger_blocks_test_selected_or_unequal_method_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "search.jsonl"
            rows = [
                {"trial_id": "a1", "dataset": "pan", "config": "global", "method": "ours", "selection_split": "validation", "selected": False, "validation_metric": 0.8},
                {"trial_id": "b1", "dataset": "pan", "config": "global", "method": "sft", "selection_split": "test", "selected": False, "validation_metric": 0.7},
                {"trial_id": "b2", "dataset": "pan", "config": "global", "method": "sft", "selection_split": "validation", "selected": False, "validation_metric": 0.6},
            ]
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = run_preflight(
                [AssetRequirement("search_ledger", ledger, "file", "p0-07")]
            )
        self.assertEqual(report.status, "BLOCKED")
        self.assertIn("SEARCH_LEDGER_INVALID", {issue.code for issue in report.issues})

    def test_search_ledger_blocks_validation_winners_without_executable_hyperparameters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "search.jsonl"
            rows = [
                {
                    "trial_id": method,
                    "dataset": "pan",
                    "config": "validation_selected",
                    "method": method,
                    "selection_split": "validation",
                    "selected": True,
                    "validation_metric": 0.8,
                }
                for method in ("sft1", "random", "ours")
            ]
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = run_preflight(
                [AssetRequirement("search_ledger", ledger, "file", "p0-07")]
            )
        self.assertEqual(report.status, "BLOCKED")
        self.assertIn("SEARCH_LEDGER_INVALID", {issue.code for issue in report.issues})

    def test_model_registry_rejects_placeholder_or_missing_provenance(self) -> None:
        required = {
            "cell_id": "abc",
            "pair": "qwen35_9b_to_08b",
            "dataset": "pan",
            "method": "ours",
            "model_hash": "a" * 64,
            "dataset_hash": "b" * 64,
            "config_hash": "c" * 64,
            "checkpoint_hash": "d" * 64,
            "commit": "e" * 40,
        }
        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "models.jsonl"
            registry.write_text(json.dumps(required) + "\n", encoding="utf-8")
            ready = run_preflight(
                [AssetRequirement("model_registry", registry, "file", "p0-01")]
            )
            registry.write_text(
                json.dumps({**required, "checkpoint_hash": "REPLACE_ME"}) + "\n",
                encoding="utf-8",
            )
            blocked = run_preflight(
                [AssetRequirement("model_registry", registry, "file", "p0-01")]
            )
        self.assertEqual(ready.status, "READY")
        self.assertEqual(blocked.status, "BLOCKED")
        self.assertIn("MODEL_REGISTRY_INVALID", {issue.code for issue in blocked.issues})

    def test_structured_registry_preflight_is_scoped_to_the_requesting_cell_axes(self) -> None:
        provenance = {
            "cell_id": "abc",
            "pair": "qwen35_9b_to_08b",
            "dataset": "pan",
            "method": "ours",
            "model_hash": "a" * 64,
            "dataset_hash": "b" * 64,
            "config_hash": "c" * 64,
            "checkpoint_hash": "d" * 64,
            "commit": "e" * 40,
        }
        search_rows = [
            {"trial_id": "a1", "dataset": "pan", "config": "global", "method": "ours", "selection_split": "validation", "selected": False, "validation_metric": 0.8},
            {"trial_id": "b1", "dataset": "pan", "config": "global", "method": "sft", "selection_split": "validation", "selected": False, "validation_metric": 0.7},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_registry = root / "models.jsonl"
            model_registry.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
            search_ledger = root / "search.jsonl"
            search_ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in search_rows), encoding="utf-8"
            )
            model_requirements, _ = requirements_from_manifest(
                ["model_registry"],
                {"model_registry": {"path": str(model_registry), "kind": "file"}},
                cell_id="missing-model-cell",
                selectors={"pair": "llama31_8b_to_1b", "dataset": "pan", "method": "ours"},
            )
            search_requirements, _ = requirements_from_manifest(
                ["search_ledger"],
                {"search_ledger": {"path": str(search_ledger), "kind": "file"}},
                cell_id="missing-search-cell",
                selectors={"dataset": "c5", "config": "global"},
            )
            model_report = run_preflight(model_requirements)
            search_report = run_preflight(search_requirements)
        self.assertEqual(model_report.status, "BLOCKED")
        self.assertEqual(search_report.status, "BLOCKED")
        self.assertIn("MODEL_REGISTRY_INVALID", {issue.code for issue in model_report.issues})
        self.assertIn("SEARCH_LEDGER_INVALID", {issue.code for issue in search_report.issues})

    def test_manifest_requirements_are_exact_and_support_explicit_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            requirements, missing = requirements_from_manifest(
                ["phase1_data", "wildguard_model", "not_declared"],
                {
                    "phase1_data": str(data),
                    "wildguard_model": {"path": str(model), "kind": "model"},
                },
                cell_id="cell-1",
                base_dir=root,
            )
            report = run_preflight(requirements)
        self.assertEqual(report.status, "READY")
        self.assertEqual(missing, ("not_declared",))
        self.assertEqual([item.kind for item in requirements], ["directory", "model"])

    def test_manifest_relative_paths_are_anchored_to_the_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "assets" / "phasef").mkdir(parents=True)
            requirements, missing = requirements_from_manifest(
                ["phasef_data"],
                {"phasef_data": {"path": "assets/phasef", "kind": "directory"}},
                cell_id="cell-relative",
                base_dir=root,
            )
            report = run_preflight(requirements)
        self.assertEqual(missing, ())
        self.assertEqual(report.status, "READY")
        self.assertEqual(requirements[0].path, (root / "assets" / "phasef").resolve())

    def test_manifest_expands_only_declared_platform_root_environment_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "processed").mkdir()
            with patch.dict("os.environ", {"SAFETY_DATA_ROOT": str(root)}, clear=False):
                requirements, missing = requirements_from_manifest(
                    ["phase1_data"],
                    {
                        "phase1_data": {
                            "path": "${SAFETY_DATA_ROOT}/processed",
                            "kind": "directory",
                        }
                    },
                    cell_id="cell-env",
                    base_dir=root / "manifest-dir",
                )
        self.assertEqual(missing, ())
        self.assertEqual(requirements[0].path, (root / "processed").resolve())

    def test_manifest_rejects_unknown_kind_and_non_path_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind"):
            requirements_from_manifest(
                ["x"], {"x": {"path": "/x", "kind": "network"}}, cell_id="c"
            )
        with self.assertRaisesRegex(ValueError, "path"):
            requirements_from_manifest(["x"], {"x": []}, cell_id="c")

    def test_cross_family_training_models_are_resolved_under_persistent_model_root(self) -> None:
        project = Path(__file__).resolve().parents[1]
        catalog = load_catalog(project / "configs" / "ablations" / "catalog.yaml")
        plan = build_catalog_plan(catalog, output_root="/out", scope="all")
        cell = next(
            item
            for item in plan.cells
            if item.experiment_id == "P2-04" and item.axes["bridge_mode"] == "token_string"
        )
        with tempfile.TemporaryDirectory() as td:
            model_root = Path(td) / "models"
            self._complete_model(model_root / "Qwen3-8B")
            self._complete_model(model_root / "Llama-3.2-1B-Instruct")
            requirements = training_model_requirements(
                cell,
                project_root=project,
                environment={"SAFETY_MODEL_ROOT": str(model_root)},
            )
            ready = run_preflight(requirements)
            (model_root / "Llama-3.2-1B-Instruct" / "model.safetensors").unlink()
            blocked = run_preflight(requirements)
        self.assertEqual(
            {item.path for item in requirements},
            {
                (model_root / "Qwen3-8B").resolve(),
                (model_root / "Llama-3.2-1B-Instruct").resolve(),
            },
        )
        self.assertEqual(ready.status, "READY")
        self.assertEqual(blocked.status, "BLOCKED")
        self.assertIn("MODEL_WEIGHTS_MISSING", {issue.code for issue in blocked.issues})

    def test_teacher_control_replaces_only_teacher_model_requirement(self) -> None:
        project = Path(__file__).resolve().parents[1]
        catalog = load_catalog(project / "configs" / "ablations" / "catalog.yaml")
        plan = build_catalog_plan(catalog, output_root="/out", scope="all")
        cell = next(
            item
            for item in plan.cells
            if item.experiment_id == "P2-03" and item.axes["teacher"] == "same_size_base"
        )
        with tempfile.TemporaryDirectory() as td:
            model_root = Path(td) / "models"
            requirements = training_model_requirements(
                cell,
                project_root=project,
                environment={"SAFETY_MODEL_ROOT": str(model_root)},
            )
        self.assertEqual(
            {item.asset_id: item.path for item in requirements},
            {
                "training_teacher_model": (model_root / "teacher-controls" / "same-size-base").resolve(),
                "training_student_model": (model_root / "Qwen3-0.6B").resolve(),
            },
        )

    def test_training_data_requirements_follow_the_effective_dataset_config(self) -> None:
        project = Path(__file__).resolve().parents[1]
        catalog = load_catalog(project / "configs" / "ablations" / "catalog.yaml")
        plan = build_catalog_plan(catalog, output_root="/out", scope="all")
        pan = next(item for item in plan.cells if item.experiment_id == "P1-11")
        wjb = next(item for item in plan.cells if item.experiment_id == "P0-06")
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td) / "data"
            environment = {"SAFETY_DATA_ROOT": str(data_root)}
            pan_requirements = training_data_requirements(
                pan, project_root=project, environment=environment
            )
            wjb_requirements = training_data_requirements(
                wjb, project_root=project, environment=environment
            )
        self.assertEqual(
            {item.asset_id for item in pan_requirements},
            {
                "training_pan_toxicity",
                "training_pan_safety",
                "training_pan_add_moderation",
                "training_pan_sr_moderation",
            },
        )
        self.assertEqual(
            {item.asset_id: item.path for item in wjb_requirements},
            {
                "training_safety_train": (data_root / "processed" / "safety" / "wildjailbreak_20k_train.jsonl").resolve(),
                "training_safety_eval": (data_root / "processed" / "eval" / "wildjailbreak_test.jsonl").resolve(),
                "training_pan_test": (data_root / "processed" / "pan_test_set.jsonl").resolve(),
            },
        )

    def test_training_splits_block_empty_duplicate_or_overlapping_prompts(self) -> None:
        def row(sample_id: str, prompt: str, label: str) -> dict:
            return {"id": sample_id, "user_text": prompt, "label": label}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            requirements = [
                AssetRequirement("training_safety_train", train, "file", "cell"),
                AssetRequirement("training_safety_eval", evaluation, "file", "cell"),
            ]
            train.write_text("", encoding="utf-8")
            evaluation.write_text(json.dumps(row("e", "unique eval", "harmful")) + "\n", encoding="utf-8")
            empty = run_preflight(requirements)
            train.write_text(
                "".join(json.dumps(item) + "\n" for item in [
                    row("t1", "duplicate", "harmful"),
                    row("t2", "duplicate", "harmless"),
                ]),
                encoding="utf-8",
            )
            evaluation.write_text(
                "".join(json.dumps(item) + "\n" for item in [
                    row("e1", "duplicate", "harmful"),
                    row("e2", "unique eval", "harmless"),
                ]),
                encoding="utf-8",
            )
            leaked = run_preflight(requirements)
        self.assertIn("ASSET_FILE_EMPTY", {item.code for item in empty.issues})
        self.assertIn("TRAINING_SPLIT_LEAKAGE", {item.code for item in leaked.issues})


if __name__ == "__main__":
    unittest.main()
