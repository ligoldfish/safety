from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.ablations.data_audit import audit_train_eval_splits
from src.data import safety_datasets
from src.data.dataset_io import (
    _take_frame_excluding_prompts,
    build_pan_train_test_records,
)
from src.data.safety_datasets import (
    build_coconot_records,
    build_safety_tuned_llamas_records,
    build_wildguardmix_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class PanSplitIntegrityTests(unittest.TestCase):
    def test_zero_budget_returns_empty_and_negative_budget_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [{"source_row": 0, "jailbroken_prompt": "unused", "source_dataset": "x"}]
        )

        selected, drops = _take_frame_excluding_prompts(
            frame,
            prompt_column="jailbroken_prompt",
            protected_prompts=set(),
            count=0,
            max_prompt_chars=2048,
            source_group="test",
        )

        self.assertTrue(selected.empty)
        self.assertEqual(drops, [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _take_frame_excluding_prompts(
                frame,
                prompt_column="jailbroken_prompt",
                protected_prompts=set(),
                count=-1,
                max_prompt_chars=2048,
                source_group="test",
            )

    def test_real_pan_sources_are_backfilled_to_unique_prompt_budget(self) -> None:
        raw_dir = PROJECT_ROOT / "external" / "safety-residual-space" / "data"
        train, evaluation, metadata = build_pan_train_test_records(raw_dir)

        audit = audit_train_eval_splits(train, evaluation)

        self.assertEqual(len(train), 2600)
        self.assertEqual(len(evaluation), 960)
        self.assertEqual(audit["overlap_count"], 0)
        self.assertEqual(audit["train_duplicate_count"], 0)
        self.assertEqual(audit["eval_duplicate_count"], 0)
        reconstruction = metadata["pan_reconstruction"]
        drops = reconstruction["split_drops"]
        overlap_drops = sum(
            item["drop_reason"] == "train_eval_prompt_overlap" for item in drops
        )
        duplicate_drops = sum(item["drop_reason"] == "duplicate_prompt" for item in drops)
        self.assertEqual(
            reconstruction["train_test_prompt_overlap_removed"], overlap_drops
        )
        self.assertEqual(reconstruction["duplicate_prompt_removed"], duplicate_drops)

    def test_cross_label_eval_prompt_is_excluded_from_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "Unnamed: 0": index,
                        "idx_org": index,
                        "method": "method-a",
                        "jailbroken_prompt": prompt,
                        "accept": f"unsafe {index}",
                        "rejected": f"refuse {index}",
                        "forbidden_prompt": f"forbidden {index}",
                    }
                    for index, prompt in enumerate(
                        ["harmful eval", "Cross Label Prompt", "harmful spare"]
                    )
                ]
            ).to_csv(raw_dir / "toxicity.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Unnamed: 0": index,
                        "idx_org": index,
                        "method": "or-bench",
                        "jailbroken_prompt": prompt,
                        "accept": f"help {index}",
                        "rejected": f"refuse safe {index}",
                    }
                    for index, prompt in enumerate(
                        ["safe one", "safe two", "  cross   label prompt  "]
                    )
                ]
            ).to_csv(raw_dir / "safety.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Unnamed: 0": 0,
                        "idx_org": 0,
                        "method": "data/sr_moderation.csv",
                        "jailbroken_prompt": "additional harmful",
                        "accept": "unsafe additional",
                        "rejected": "refuse additional",
                    }
                ]
            ).to_csv(raw_dir / "add_moderation.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Unnamed: 0": index,
                        "category": "test",
                        "source": "synthetic",
                        "forbidden_prompt": f"forbidden {index}",
                        "moderation_truth": 2,
                    }
                    for index in range(3)
                ]
            ).to_csv(raw_dir / "sr_moderation.csv", index=False)

            train, evaluation, _ = build_pan_train_test_records(
                raw_dir,
                exposure_size=1,
                pan_test_size_per_type=1,
                pan_train_size=4,
                seed=42,
            )

            self.assertEqual(len(train), 4)
            self.assertEqual(len(evaluation), 2)
            self.assertEqual(audit_train_eval_splits(train, evaluation)["overlap_count"], 0)


class SafetyTunedLlamasSplitIntegrityTests(unittest.TestCase):
    def test_default_contrast_is_balanced_after_harmful_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "safety_only_data_Instructions.json").write_text(
                json.dumps(
                    [
                        {"instruction": value, "input": "", "output": "refuse"}
                        for value in ("Duplicate", " duplicate ", "DUPLICATE")
                    ]
                ),
                encoding="utf-8",
            )
            (data_dir / "alpaca_small.json").write_text(
                json.dumps(
                    [
                        {"instruction": f"benign {index}", "input": "", "output": "help"}
                        for index in range(3)
                    ]
                ),
                encoding="utf-8",
            )

            records = build_safety_tuned_llamas_records(
                output_path=Path(tmpdir) / "train.jsonl",
                repo_or_data_path=data_dir,
                include_harmless_contrast=True,
            )

        labels = [record["label"] for record in records]
        self.assertEqual(labels.count("harmful"), 1)
        self.assertEqual(labels.count("harmless"), 1)

    def test_canonical_duplicates_are_removed_before_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            harmful = [
                {"instruction": "Caf\u00e9 request", "input": "", "output": "refuse"},
                {"instruction": "  CAFE\u0301   REQUEST  ", "input": "", "output": "refuse again"},
                *[
                    {"instruction": f"unsafe unique {index}", "input": "", "output": "refuse"}
                    for index in range(8)
                ],
            ]
            harmless = [
                {"instruction": "Benign duplicate", "input": "", "output": "help"},
                {"instruction": " benign   DUPLICATE ", "input": "", "output": "help again"},
                *[
                    {"instruction": f"safe unique {index}", "input": "", "output": "help"}
                    for index in range(8)
                ],
            ]
            (data_dir / "safety_only_data_Instructions.json").write_text(
                json.dumps(harmful, ensure_ascii=False), encoding="utf-8"
            )
            (data_dir / "alpaca_small.json").write_text(
                json.dumps(harmless, ensure_ascii=False), encoding="utf-8"
            )
            train_path = Path(tmpdir) / "train.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"

            train = build_safety_tuned_llamas_records(
                output_path=train_path,
                repo_or_data_path=data_dir,
                include_harmless_contrast=True,
                eval_output_path=eval_path,
                eval_holdout_fraction=0.2,
                seed=7,
            )
            evaluation = _read_jsonl(eval_path)
            audit = audit_train_eval_splits(train, evaluation)
            split_audit = json.loads(
                train_path.with_suffix(".jsonl.split_audit.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(train) + len(evaluation), 18)
        self.assertEqual(audit["overlap_count"], 0)
        self.assertEqual(audit["train_duplicate_count"], 0)
        self.assertEqual(audit["eval_duplicate_count"], 0)
        self.assertEqual(len(split_audit["drops"]), 2)
        self.assertTrue(
            all(item["drop_reason"] == "duplicate_prompt" for item in split_audit["drops"])
        )


class WildGuardMixSplitIntegrityTests(unittest.TestCase):
    def test_canonical_train_duplicates_are_dropped(self) -> None:
        train_rows = [
            {
                "id": "h0",
                "prompt": "How do I HACK?",
                "response": "I cannot help with that.",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "refusal",
            },
            {
                "id": "h1",
                "prompt": "  how do i   hack? ",
                "response": "I still cannot help.",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "refusal",
            },
            {
                "id": "h2",
                "prompt": "harmful unique",
                "response": "I cannot help.",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "refusal",
            },
            {
                "id": "b0",
                "prompt": "benign one",
                "response": "help one",
                "prompt_harm_label": "unharmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "compliance",
            },
            {
                "id": "b1",
                "prompt": "benign two",
                "response": "help two",
                "prompt_harm_label": "unharmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "compliance",
            },
        ]
        eval_rows = [
            {"id": "eh", "prompt": "eval harmful", "prompt_harm_label": "harmful"},
            {"id": "eb", "prompt": "eval benign", "prompt_harm_label": "unharmful"},
        ]

        def fake_load(_source, config, **_kwargs):
            return eval_rows if config == "wildguardtest" else train_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", side_effect=fake_load):
                train = build_wildguardmix_records(
                    output_path=train_path,
                    train_subset_mode=False,
                    eval_output_path=eval_path,
                )
            evaluation = _read_jsonl(eval_path)
            summary = json.loads(
                train_path.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8")
            )
            split_audit = json.loads(
                train_path.with_suffix(".jsonl.split_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            audit = audit_train_eval_splits(train, evaluation)

        self.assertEqual(len(train), 4)
        self.assertEqual(audit["train_duplicate_count"], 0)
        self.assertEqual(audit["overlap_count"], 0)
        self.assertEqual(summary["drops"]["duplicate_prompt"], 1)
        duplicate_drop = next(
            item for item in split_audit["drops"] if item["drop_reason"] == "duplicate_prompt"
        )
        self.assertEqual(duplicate_drop["id"], "h1")
        self.assertEqual(len(duplicate_drop["prompt_sha256"]), 64)


class CoCoNotSplitIntegrityTests(unittest.TestCase):
    def test_cross_label_deduplication_cannot_silently_empty_a_label(self) -> None:
        table = {
            ("original", "train"): [
                {
                    "id": "h0",
                    "prompt": "same prompt",
                    "response": "I cannot help.",
                    "category": "Requests with safety concerns",
                }
            ],
            ("pref", "train"): [
                {"id": "b0", "prompt": " SAME   PROMPT ", "chosen": "help"}
            ],
        }

        def fake_load(_source, config, *args, split=None, **_kwargs):
            return table[(config, split)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(safety_datasets, "_load_dataset", side_effect=fake_load):
                with self.assertRaisesRegex(RuntimeError, "both labels.*deduplication"):
                    build_coconot_records(
                        output_path=Path(tmpdir) / "train.jsonl",
                        train_subset_mode=False,
                    )

    def test_canonical_train_duplicates_are_removed_before_balancing(self) -> None:
        table = {
            ("original", "train"): [
                {
                    "id": "h0",
                    "prompt": "Refuse This Request",
                    "response": "I cannot help.",
                    "category": "Requests with safety concerns",
                },
                {
                    "id": "h1",
                    "prompt": " refuse   this request ",
                    "response": "I still cannot help.",
                    "category": "Requests with safety concerns",
                },
                {
                    "id": "h2",
                    "prompt": "different harmful request",
                    "response": "I cannot help with that either.",
                    "category": "Requests with safety concerns",
                },
            ],
            ("pref", "train"): [
                {"id": "b0", "prompt": "benign one", "chosen": "help one"},
                {"id": "b1", "prompt": "benign two", "chosen": "help two"},
            ],
            ("original", "test"): [
                {
                    "id": "eh",
                    "prompt": "  REFUSE this   request  ",
                    "category": "Requests with safety concerns",
                }
            ],
            ("contrast", "test"): [
                {"id": "eb", "prompt": "eval benign"}
            ],
        }

        def fake_load(_source, config, *args, split=None, **_kwargs):
            return table[(config, split)]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", side_effect=fake_load):
                train = build_coconot_records(
                    output_path=train_path,
                    eval_output_path=eval_path,
                    train_subset_mode=False,
                )
            evaluation = _read_jsonl(eval_path)
            split_audit = json.loads(
                train_path.with_suffix(".jsonl.split_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            audit = audit_train_eval_splits(train, evaluation)

        self.assertEqual(len(train), 2)
        self.assertEqual({item["label"] for item in train}, {"harmful", "harmless"})
        self.assertEqual(audit["train_duplicate_count"], 0)
        self.assertEqual(audit["overlap_count"], 0)
        drop_reasons = {item["drop_reason"] for item in split_audit["drops"]}
        self.assertEqual(
            drop_reasons,
            {"duplicate_prompt", "train_eval_prompt_overlap"},
        )
        self.assertTrue(
            all(len(item["prompt_sha256"]) == 64 for item in split_audit["drops"])
        )


if __name__ == "__main__":
    unittest.main()
