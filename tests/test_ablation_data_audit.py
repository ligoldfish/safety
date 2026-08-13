from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ablations.data_audit import (
    audit_train_eval_splits,
    canonical_prompt,
    exclude_protected_prompts,
    stratified_holdout,
    write_data_audit,
    prompt_sha256,
)
from src.data.dataset_io import _take_frame_excluding_prompts
import pandas as pd


def record(record_id: str, prompt: str, label: str) -> dict:
    return {
        "id": record_id,
        "label": label,
        "messages": [{"role": "user", "content": prompt}],
        "target_response": "response",
    }


class AblationDataAuditTests(unittest.TestCase):
    def test_prompt_normalization_is_unicode_and_whitespace_stable(self) -> None:
        self.assertEqual(canonical_prompt("  Ｈello\n WORLD  "), canonical_prompt("hello world"))

    def test_cross_split_exclusion_keeps_internal_duplicates(self) -> None:
        train = [record("a", "same", "harmful"), record("b", "same", "harmful"), record("c", "keep", "harmless")]
        test = [record("t", "same", "harmful")]
        kept, dropped = exclude_protected_prompts(train, test, reason="train_eval_prompt_overlap")
        self.assertEqual([item["id"] for item in kept], ["c"])
        self.assertEqual([item["id"] for item in dropped], ["a", "b"])
        self.assertTrue(all(item["drop_reason"] == "train_eval_prompt_overlap" for item in dropped))

    def test_stratified_holdout_is_deterministic_and_disjoint(self) -> None:
        records = [record(f"h{i}", f"harm-{i}", "harmful") for i in range(10)] + [record(f"b{i}", f"benign-{i}", "harmless") for i in range(10)]
        train, evaluation = stratified_holdout(records, fraction=0.2, seed=42)
        again = stratified_holdout(list(reversed(records)), fraction=0.2, seed=42)
        self.assertEqual([item["id"] for item in train], [item["id"] for item in again[0]])
        self.assertEqual(len(evaluation), 4)
        self.assertEqual(audit_train_eval_splits(train, evaluation)["overlap_count"], 0)

    def test_holdout_keeps_duplicate_prompt_group_on_one_side(self) -> None:
        records = [
            record("h0", "duplicate", "harmful"),
            record("h1", "duplicate", "harmful"),
            record("h2", "unique harmful", "harmful"),
            record("h3", "another harmful", "harmful"),
            record("b0", "safe zero", "harmless"),
            record("b1", "safe one", "harmless"),
            record("b2", "safe two", "harmless"),
        ]
        train, evaluation = stratified_holdout(records, fraction=0.34, seed=9)
        self.assertEqual(audit_train_eval_splits(train, evaluation)["overlap_count"], 0)
        train_duplicate_ids = {item["id"] for item in train if prompt_sha256(item) == prompt_sha256(records[0])}
        eval_duplicate_ids = {item["id"] for item in evaluation if prompt_sha256(item) == prompt_sha256(records[0])}
        self.assertIn({"h0", "h1"}, (train_duplicate_ids, eval_duplicate_ids))
        self.assertIn(set(), (train_duplicate_ids, eval_duplicate_ids))

    def test_audit_separates_internal_duplicates_from_cross_split_overlap(self) -> None:
        train = [record("a", "dup", "harmful"), record("b", "dup", "harmful")]
        evaluation = [record("t", "other", "harmless")]
        report = audit_train_eval_splits(train, evaluation)
        self.assertEqual(report["train_duplicate_count"], 1)
        self.assertEqual(report["overlap_count"], 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_data_audit(
                Path(tmpdir) / "audit.json",
                dataset="demo",
                train=train,
                evaluation=evaluation,
                drops=[],
                license_name="test-only",
                intended_use="unit test",
            )
            self.assertTrue(path.is_file())

    def test_pan_sampler_backfills_after_protected_prompt(self) -> None:
        frame = pd.DataFrame(
            [
                {"source_row": 0, "source_dataset": "x", "jailbroken_prompt": "held out"},
                {"source_row": 1, "source_dataset": "x", "jailbroken_prompt": "train one"},
                {"source_row": 2, "source_dataset": "x", "jailbroken_prompt": "train two"},
            ]
        )
        selected, drops = _take_frame_excluding_prompts(
            frame,
            prompt_column="jailbroken_prompt",
            protected_prompts={canonical_prompt("held out")},
            count=2,
            max_prompt_chars=2048,
            source_group="fixture",
        )
        self.assertEqual(selected["source_row"].tolist(), [1, 2])
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["drop_reason"], "train_eval_prompt_overlap")


if __name__ == "__main__":
    unittest.main()
