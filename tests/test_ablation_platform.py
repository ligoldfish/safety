from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ablations.platform import configure_visible_devices, resolve_portable_path
from src.utils.config import load_phase1_config


class AblationPlatformTests(unittest.TestCase):
    def test_category_roots_rebase_paths_without_yaml_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = {
                "SAFETY_MODEL_ROOT": str(root / "models"),
                "SAFETY_DATA_ROOT": str(root / "datasets"),
                "SAFETY_OUTPUT_ROOT": str(root / "outputs"),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(
                    resolve_portable_path("../models/Qwen3-0.6B", root, category="model"),
                    str(root / "models" / "Qwen3-0.6B"),
                )
                self.assertEqual(
                    resolve_portable_path("../data/processed/pan", root, category="data"),
                    str(root / "datasets" / "processed" / "pan"),
                )
                self.assertEqual(
                    resolve_portable_path("../outputs/run/a.pt", root, category="output"),
                    str(root / "outputs" / "run" / "a.pt"),
                )

    def test_phase1_loader_consumes_portable_roots(self) -> None:
        yaml_text = """
seed: 42
dataset:
  pan_repo_dir: ../data/PAN
  raw_dir: ../data/raw
  processed_dir: ../data/processed
  metadata_dir: ../data/meta
models:
  teacher: {name: teacher, path: ../models/Teacher}
  student: {name: student, path: ../models/Student}
extraction: {output_root: ../outputs/pair}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "configs" / "phase.yaml"
            config.parent.mkdir()
            config.write_text(yaml_text, encoding="utf-8")
            env = {
                "SAFETY_MODEL_ROOT": str(root / "portable-models"),
                "SAFETY_DATA_ROOT": str(root / "portable-data"),
                "SAFETY_OUTPUT_ROOT": str(root / "portable-output"),
            }
            with patch.dict(os.environ, env, clear=False):
                loaded = load_phase1_config(config)
            self.assertEqual(loaded.teacher.path, str(root / "portable-models" / "Teacher"))
            self.assertEqual(loaded.dataset.processed_dir, str(root / "portable-data" / "processed"))
            self.assertEqual(loaded.extraction.output_root, str(root / "portable-output" / "pair"))

    def test_existing_ascend_visibility_is_never_overwritten(self) -> None:
        env = {"ASCEND_RT_VISIBLE_DEVICES": "2,3"}
        configured = configure_visible_devices(env, backend="npu", requested_devices="0,1")
        self.assertEqual(configured["ASCEND_RT_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(
            configure_visible_devices({}, backend="npu", requested_devices="0")["ASCEND_RT_VISIBLE_DEVICES"],
            "0",
        )


if __name__ == "__main__":
    unittest.main()
