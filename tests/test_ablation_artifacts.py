from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ablations.artifacts import canonical_hash, phase1_artifact_key, sha256_file


class AblationArtifactTests(unittest.TestCase):
    def test_file_hash_is_streaming_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.bin"
            path.write_bytes(b"a" * 10000)
            first = sha256_file(path, chunk_size=257)
            self.assertEqual(len(first), 64)
            path.write_bytes(b"a" * 9999 + b"b")
            self.assertNotEqual(first, sha256_file(path, chunk_size=257))

    def test_canonical_hash_is_mapping_order_independent(self) -> None:
        self.assertEqual(
            canonical_hash({"a": 1, "b": {"x": 2, "y": 3}}),
            canonical_hash({"b": {"y": 3, "x": 2}, "a": 1}),
        )

    def test_phase1_key_changes_for_every_semantic_input(self) -> None:
        base = {
            "teacher": "teacher-hash",
            "student": "student-hash",
            "teacher_tokenizer": "ttok",
            "student_tokenizer": "stok",
            "dataset": "data-hash",
            "seed": 42,
            "representation": "last_prompt",
            "layer_selection": "effect_probe_sum",
            "subspace": "learned",
            "bridge": "vocabulary",
            "pairing": "relative_depth",
            "target": "semantic",
            "commit": "abc123",
            "schema_version": 1,
        }
        original = phase1_artifact_key(base)
        for key, value in base.items():
            changed = dict(base)
            changed[key] = f"{value}-changed" if not isinstance(value, int) else value + 1
            with self.subTest(key=key):
                self.assertNotEqual(original, phase1_artifact_key(changed))

    def test_phase1_key_rejects_missing_required_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing phase1 artifact fields"):
            phase1_artifact_key({"teacher": "only-one-field"})


if __name__ == "__main__":
    unittest.main()
