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
)


class AblationPreflightTests(unittest.TestCase):
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
            )
            report = run_preflight(requirements)
        self.assertEqual(report.status, "READY")
        self.assertEqual(missing, ("not_declared",))
        self.assertEqual([item.kind for item in requirements], ["directory", "model"])

    def test_manifest_rejects_unknown_kind_and_non_path_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind"):
            requirements_from_manifest(
                ["x"], {"x": {"path": "/x", "kind": "network"}}, cell_id="c"
            )
        with self.assertRaisesRegex(ValueError, "path"):
            requirements_from_manifest(["x"], {"x": []}, cell_id="c")


if __name__ == "__main__":
    unittest.main()
