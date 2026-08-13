from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from src.ablations.preflight import (
    AssetRequirement,
    requirements_from_manifest,
    inspect_model_directory,
    inspect_submission_package,
    run_preflight,
    training_model_requirements,
)
from src.ablations.catalog import load_catalog
from src.ablations.planner import build_catalog_plan


class AblationPreflightTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
