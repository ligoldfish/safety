from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.ablations.foundation_cache import (
    FoundationCacheError,
    foundation_is_ready,
    foundation_lock,
    mark_foundation_ready,
    required_foundation_artifacts,
)


class FoundationCacheTests(unittest.TestCase):
    def test_marker_is_written_only_after_every_declared_artifact_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "phase1"
            with self.assertRaises(FoundationCacheError):
                mark_foundation_ready(root, "key-a", validation_only=True)
            for path in required_foundation_artifacts(root, validation_only=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"evidence")
            mark_foundation_ready(root, "key-a", validation_only=True)

            self.assertTrue(foundation_is_ready(root, "key-a", validation_only=True))
            self.assertFalse(foundation_is_ready(root, "key-b", validation_only=True))
            self.assertFalse(foundation_is_ready(root, "key-a", validation_only=False))

    def test_lock_is_reusable_and_preserves_the_cache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "phase1"
            with foundation_lock(root):
                self.assertTrue((root / ".foundation.lock").is_file())
            with foundation_lock(root):
                self.assertTrue(root.is_dir())

    def test_oneclick_builder_runs_once_then_reuses_complete_foundation(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("foundation_cache_oneclick_test", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "phase1"
            calls: list[str] = []

            def builder() -> None:
                calls.append("build")
                for path in required_foundation_artifacts(root, validation_only=True):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"evidence")

            fake_config = SimpleNamespace(
                extraction=SimpleNamespace(output_root=str(root))
            )
            with patch.object(module, "load_phase1_config", return_value=fake_config):
                first_hit = module._run_cached_foundation(
                    Path("phase1.yaml"),
                    cache_key="key-a",
                    validation_only=True,
                    dry_run=False,
                    builder=builder,
                )
                second_hit = module._run_cached_foundation(
                    Path("phase1.yaml"),
                    cache_key="key-a",
                    validation_only=True,
                    dry_run=False,
                    builder=builder,
                )

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls, ["build"])


if __name__ == "__main__":
    unittest.main()
