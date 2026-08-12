"""Tests for 01_extract_hidden_states shard-freshness guard.

Regression cover for the stale-shard bug: when the prepared split is rebuilt
with different sample ids, the existing hidden-state shards must NOT be reused
(the old skip_existing logic only checked shard_path.exists(), so a data rebuild
silently reused stale shards -> downstream target_map join collapse).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "01_extract_hidden_states.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("extract01", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_fingerprint_changes_when_ids_change():
    fp_a = mod._split_fingerprint([{"id": "a"}, {"id": "b"}], shard_size=32)
    fp_b = mod._split_fingerprint([{"id": "x"}, {"id": "y"}], shard_size=32)
    assert fp_a["id_hash"] != fp_b["id_hash"]
    assert fp_a["num_records"] == 2


def test_fingerprint_changes_when_shard_size_changes():
    recs = [{"id": "a"}, {"id": "b"}]
    assert mod._split_fingerprint(recs, shard_size=32) != mod._split_fingerprint(recs, shard_size=64)


def test_fingerprint_changes_when_representation_mode_changes():
    recs = [{"id": "a"}, {"id": "b"}]
    prompt = mod._split_fingerprint(recs, shard_size=32, representation_mode="last_prompt")
    generated = mod._split_fingerprint(recs, shard_size=32, representation_mode="first_generated")
    assert prompt != generated
    assert prompt["representation_mode"] == "last_prompt"


def test_shards_stale_when_ids_change(tmp_path):
    # Prior run wrote a fingerprint for the OLD ids.
    old = mod._split_fingerprint([{"id": "old_a"}, {"id": "old_b"}], shard_size=32)
    mod._write_fingerprint(tmp_path, old)
    # Data rebuilt with NEW ids -> shards must be treated as stale.
    new = mod._split_fingerprint([{"id": "new_a"}, {"id": "new_b"}], shard_size=32)
    assert mod._shards_are_fresh(tmp_path, new) is False


def test_shards_fresh_when_ids_match(tmp_path):
    fp = mod._split_fingerprint([{"id": "a"}, {"id": "b"}], shard_size=32)
    mod._write_fingerprint(tmp_path, fp)
    assert mod._shards_are_fresh(tmp_path, fp) is True


def test_missing_fingerprint_not_fresh(tmp_path):
    fp = mod._split_fingerprint([{"id": "a"}], shard_size=32)
    assert mod._shards_are_fresh(tmp_path, fp) is False


def test_clear_stale_shards_removes_parts_and_fingerprint(tmp_path):
    (tmp_path / "part_000.pt").write_bytes(b"x")
    (tmp_path / "part_001.pt").write_bytes(b"y")
    mod._write_fingerprint(tmp_path, mod._split_fingerprint([{"id": "a"}], shard_size=32))
    keep = tmp_path / "manifest.json"
    keep.write_text("{}", encoding="utf-8")

    mod._clear_stale_shards(tmp_path)

    assert list(tmp_path.glob("part_*.pt")) == []
    assert not mod._fingerprint_path(tmp_path).exists()
    # Unrelated files are left untouched.
    assert keep.exists()
