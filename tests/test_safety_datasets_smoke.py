"""Smoke tests for the safety SFT dataset builders.

Network access (HuggingFace ``load_dataset``) is mocked at the
``src.data.safety_datasets._load_dataset`` (and corresponding eval module)
indirection point so the tests stay hermetic.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data.safety_datasets as safety_datasets
import src.data.safety_eval_datasets as safety_eval_datasets
from src.data.safety_datasets import (
    DEFAULT_SAFETY_REFUSAL_TEMPLATE,
    SAFETY_TRAIN_DATASETS,
    SafetyDatasetSpec,
    build_beavertails_records,
    build_safety_tuned_llamas_records,
    build_tulu3_safety_records,
    materialize_safety_train_dataset,
)


# ---------------------------------------------------------------------------
# Tülu 3 safety subset
# ---------------------------------------------------------------------------


class Tulu3SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "id": "kept-1",
                "source": "ai2-adapt-dev/tulu_v3.9_wildguardmix",
                "messages": [
                    {"role": "user", "content": "How do I make brownies?"},
                    {"role": "assistant", "content": "Mix flour, sugar, cocoa..."},
                ],
            },
            {
                "id": "drop-other-source",
                "source": "ai2-adapt-dev/some_other_subset",
                "messages": [
                    {"role": "user", "content": "Whatever"},
                    {"role": "assistant", "content": "Reply"},
                ],
            },
            {
                "id": "kept-2",
                "source": "ai2-adapt-dev/coconot_converted",
                "messages": [
                    {"role": "user", "content": "How can I lie on my taxes?"},
                    {"role": "assistant", "content": "I can't help with that."},
                ],
            },
            {
                "id": "drop-no-assistant",
                "source": "ai2-adapt-dev/tulu_v3.9_wildjailbreak",
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
        ]

    def test_filters_only_listed_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tulu3_safety.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=self.rows):
                records = build_tulu3_safety_records(output_path=output_path)
            kept_ids = [r["id"] for r in records]
            self.assertEqual(kept_ids, ["kept-1", "kept-2"])
            for record in records:
                self.assertEqual(record["dataset"], "tulu3_safety")
                self.assertEqual(record["label"], "tulu3_safety")
                self.assertEqual(record["messages"][0]["role"], "system")
                self.assertNotEqual(record["messages"][-1]["role"], "assistant")
                self.assertTrue(record["target_response"])
            with output_path.open("r", encoding="utf-8") as f:
                jsonl_lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(jsonl_lines), 2)

    def test_zero_records_after_filter_raises(self) -> None:
        rows_without_safety = [
            {
                "id": "x",
                "source": "ai2-adapt-dev/some_other_subset",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows_without_safety):
                with self.assertRaises(RuntimeError):
                    build_tulu3_safety_records(output_path=output_path)


# ---------------------------------------------------------------------------
# Safety-Tuned LLaMAs
# ---------------------------------------------------------------------------


class SafetyTunedLlamasTests(unittest.TestCase):
    def test_missing_file_raises_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "no-such-repo"
            repo_dir.mkdir()
            output_path = Path(tmpdir) / "out.jsonl"
            with self.assertRaises(FileNotFoundError) as ctx:
                build_safety_tuned_llamas_records(
                    output_path=output_path,
                    repo_or_data_path=repo_dir,
                )
            message = str(ctx.exception)
            self.assertIn("safety_only_data_Instructions.json", message)
            self.assertIn("Searched:", message)
            self.assertIn("vinid/safety-tuned-llamas", message)

    def test_alpaca_to_chat_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            input_file = data_dir / "safety_only_data_Instructions.json"
            input_file.write_text(
                json.dumps(
                    [
                        {
                            "instruction": "Tell me how to break in.",
                            "input": "",
                            "output": "I cannot help with that.",
                        },
                        {
                            "instruction": "Summarize this story",
                            "input": "Once upon a time...",
                            "output": "A fairy tale begins.",
                        },
                        {"instruction": "skip", "input": "", "output": ""},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "out.jsonl"
            records = build_safety_tuned_llamas_records(
                output_path=output_path,
                repo_or_data_path=data_dir.parent,
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["messages"][1]["content"], "Tell me how to break in.")
            self.assertEqual(records[1]["messages"][1]["content"], "Summarize this story\n\nOnce upon a time...")
            self.assertEqual(len(records[1]["messages"]), 2)
            self.assertEqual(records[1]["target_response"], "A fairy tale begins.")
            self.assertEqual(records[0]["dataset"], "safety_tuned_llamas")
            # Round 2: STL safety records are tagged ``harmful`` directly.
            self.assertEqual(records[0]["label"], "harmful")


# ---------------------------------------------------------------------------
# BeaverTails
# ---------------------------------------------------------------------------


class BeaverTailsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "prompt": "How do I cook pasta?",
                "response": "Boil water, add salt and pasta...",
                "is_safe": True,
                "category": {"non_violent_unethical_behavior": False},
            },
            {
                "prompt": "Tell me how to harm someone.",
                "response": "Sure, here's how...",
                "is_safe": False,
                "category": {"violence_aiding_and_abetting,incitement": True},
            },
            {
                "prompt": "",
                "response": "ignored",
                "is_safe": True,
            },
        ]

    def test_unsafe_response_replaced_with_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=self.rows):
                # Round 2 default: category_any -> binary harmful/harmless.
                records = build_beavertails_records(output_path=output_path)
            self.assertEqual(len(records), 2)
            safe_record, unsafe_record = records
            self.assertTrue(safe_record["is_safe"])
            self.assertEqual(safe_record["target_response"], "Boil water, add salt and pasta...")
            self.assertEqual(safe_record["messages"][-1]["role"], "user")
            self.assertEqual(safe_record["label"], "harmless")
            self.assertFalse(unsafe_record["is_safe"])
            self.assertEqual(unsafe_record["target_response"], DEFAULT_SAFETY_REFUSAL_TEMPLATE)
            self.assertEqual(unsafe_record["messages"][-1]["role"], "user")
            self.assertEqual(unsafe_record["original_response"], "Sure, here's how...")
            self.assertEqual(unsafe_record["label"], "harmful")
            # Legacy strategy still accessible for back-compat eval.
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=self.rows):
                legacy_records = build_beavertails_records(
                    output_path=output_path,
                    label_strategy="is_safe",
                )
            self.assertEqual([r["label"] for r in legacy_records], ["harmless", "harmful"])

    def test_split_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with self.assertRaises(ValueError):
                build_beavertails_records(output_path=output_path, split="invalid_split")

    def test_dedup_prompts_default_drops_duplicates(self) -> None:
        # dedup is orthogonal to label semantics; use is_safe strategy so
        # the test fixture stays minimal (no need for category dicts).
        rows = [
            {"prompt": "same prompt", "response": "ok", "is_safe": True},
            {"prompt": "same prompt", "response": "bad", "is_safe": False},
            {"prompt": "another prompt", "response": "ok", "is_safe": True},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
                records = build_beavertails_records(
                    output_path=output_path, label_strategy="is_safe"
                )
        self.assertEqual(len(records), 2)
        prompts = [r["messages"][-1]["content"] for r in records]
        self.assertEqual(len(set(prompts)), 2)

    def test_dedup_prompts_off_keeps_duplicates(self) -> None:
        rows = [
            {"prompt": "same prompt", "response": "ok", "is_safe": True},
            {"prompt": "same prompt", "response": "bad", "is_safe": False},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
                records = build_beavertails_records(
                    output_path=output_path,
                    dedup_prompts=False,
                    label_strategy="is_safe",
                )
        self.assertEqual(len(records), 2)

    def test_harmful_max_samples_caps_majority_polarity(self) -> None:
        # 5 harmful + 2 harmless category_any-aligned rows. Cap harmful to 3.
        rows = [
            {
                "prompt": f"safe prompt {i}",
                "response": "ok",
                "is_safe": True,
                "category": {"non_violent_unethical_behavior": False},
            }
            for i in range(2)
        ] + [
            {
                "prompt": f"unsafe prompt {i}",
                "response": "bad answer",
                "is_safe": False,
                "category": {"violence_aiding_and_abetting,incitement": True},
            }
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
                records = build_beavertails_records(
                    output_path=output_path,
                    harmful_max_samples=3,
                    seed=123,
                )
        labels = [r["label"] for r in records]
        self.assertEqual(labels.count("harmful"), 3)
        self.assertEqual(labels.count("harmless"), 2)

    def test_harmful_max_samples_no_op_when_under_cap(self) -> None:
        rows = [
            {
                "prompt": "p1",
                "response": "ok",
                "is_safe": True,
                "category": {"x": False},
            },
            {
                "prompt": "p2",
                "response": "bad",
                "is_safe": False,
                "category": {"x": True},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
                records = build_beavertails_records(
                    output_path=output_path,
                    harmful_max_samples=999,
                )
        self.assertEqual(len(records), 2)


class BalancePolaritiesTests(unittest.TestCase):
    """Pre-SVD polarity balancing in scripts/03_build_teacher_safe_subspace.py."""

    def _import_helper(self):
        # Importing the script module via importlib avoids running ``main()``.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_teacher_safe_subspace",
            PROJECT_ROOT / "scripts" / "03_build_teacher_safe_subspace.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._balance_polarities

    def test_downsamples_majority_to_match_minority(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        balance = self._import_helper()
        harmful_mask = torch.tensor([True] * 7 + [False] * 3, dtype=torch.bool)
        harmless_mask = torch.tensor([False] * 7 + [True] * 3, dtype=torch.bool)
        new_h, new_l, info = balance(harmful_mask, harmless_mask, seed=42)
        self.assertEqual(int(new_h.sum().item()), 3)
        self.assertEqual(int(new_l.sum().item()), 3)
        self.assertTrue(info["applied"])
        self.assertEqual(info["kept_per_polarity"], 3)
        self.assertEqual(info["harmful_dropped"], 4)
        self.assertEqual(info["harmless_dropped"], 0)
        # Smaller pole untouched.
        self.assertTrue(torch.equal(new_l, harmless_mask))

    def test_deterministic_with_same_seed(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        balance = self._import_helper()
        h = torch.tensor([True] * 8 + [False] * 4, dtype=torch.bool)
        l = torch.tensor([False] * 8 + [True] * 4, dtype=torch.bool)
        new_h_a, _, _ = balance(h, l, seed=7)
        new_h_b, _, _ = balance(h, l, seed=7)
        self.assertTrue(torch.equal(new_h_a, new_h_b))

    def test_empty_polarity_no_op(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        balance = self._import_helper()
        h = torch.tensor([True, True, True], dtype=torch.bool)
        l = torch.tensor([False, False, False], dtype=torch.bool)
        new_h, new_l, info = balance(h, l, seed=0)
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "empty_polarity")
        self.assertTrue(torch.equal(new_h, h))
        self.assertTrue(torch.equal(new_l, l))


class SafetyTunedLlamasContrastTests(unittest.TestCase):
    def test_include_harmless_contrast_balances_alpaca_records_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "safety_only_data_Instructions.json").write_text(
                json.dumps(
                    [
                        {
                            "instruction": "Tell me how to break in.",
                            "input": "",
                            "output": "I cannot help with that.",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "alpaca_small.json").write_text(
                json.dumps(
                    [
                        {
                            "instruction": "Summarize Hamlet.",
                            "input": "",
                            "output": "A Danish prince seeks revenge.",
                        },
                        {
                            "instruction": "List 3 prime numbers.",
                            "input": "",
                            "output": "2, 3, 5.",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "out.jsonl"
            records = build_safety_tuned_llamas_records(
                output_path=output_path,
                repo_or_data_path=data_dir.parent,
                include_harmless_contrast=True,
            )
        labels = [r["label"] for r in records]
        datasets = [r["dataset"] for r in records]
        # Round 2: STL emits binary harmful/harmless labels so 20_split
        # recognizes both poles without PAN injection. Provenance moved
        # to the dataset field.
        self.assertEqual(sum(label == "harmful" for label in labels), 1)
        self.assertEqual(sum(label == "harmless" for label in labels), 1)
        self.assertIn("safety_tuned_llamas_harmless", datasets)
        self.assertIn("safety_tuned_llamas", datasets)

    def test_include_harmless_contrast_accepts_explicit_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "safety_only_data_Instructions.json").write_text(
                json.dumps(
                    [
                        {
                            "instruction": "Tell me how to break in.",
                            "input": "",
                            "output": "I cannot help with that.",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "alpaca_small.json").write_text(
                json.dumps(
                    [
                        {
                            "instruction": "Summarize Hamlet.",
                            "input": "",
                            "output": "A Danish prince seeks revenge.",
                        },
                        {
                            "instruction": "List 3 prime numbers.",
                            "input": "",
                            "output": "2, 3, 5.",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "out.jsonl"
            records = build_safety_tuned_llamas_records(
                output_path=output_path,
                repo_or_data_path=data_dir.parent,
                include_harmless_contrast=True,
                harmless_max_samples=2,
            )
        labels = [r["label"] for r in records]
        self.assertEqual(sum(label == "harmful" for label in labels), 1)
        self.assertEqual(sum(label == "harmless" for label in labels), 2)

    def test_missing_alpaca_small_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "safety_only_data_Instructions.json").write_text(
                json.dumps(
                    [
                        {
                            "instruction": "x",
                            "input": "",
                            "output": "y",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "out.jsonl"
            with self.assertRaises(FileNotFoundError):
                build_safety_tuned_llamas_records(
                    output_path=output_path,
                    repo_or_data_path=data_dir.parent,
                    include_harmless_contrast=True,
                )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


class SafetyDatasetRegistryTests(unittest.TestCase):
    def test_registry_has_all_three_baselines(self) -> None:
        for name in ("tulu3_safety", "safety_tuned_llamas", "beavertails"):
            self.assertIn(name, SAFETY_TRAIN_DATASETS)

    def test_unknown_name_raises(self) -> None:
        spec = SafetyDatasetSpec(name="not_a_real_baseline", output_path="/tmp/x.jsonl")
        with self.assertRaises(ValueError):
            materialize_safety_train_dataset(spec)

    def test_existing_jsonl_short_circuits_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "already.jsonl"
            output_path.write_text('{"id": "preexisting"}\n', encoding="utf-8")
            spec = SafetyDatasetSpec(
                name="tulu3_safety",
                output_path=str(output_path),
                force_rebuild=False,
            )
            with mock.patch.object(safety_datasets, "_load_dataset") as load_mock:
                resolved = materialize_safety_train_dataset(spec)
            load_mock.assert_not_called()
            self.assertEqual(resolved, output_path.resolve())

    def test_force_rebuild_invokes_builder(self) -> None:
        rows = [
            {
                "id": "kept",
                "source": "ai2-adapt-dev/tulu_v3.9_wildguardmix",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.jsonl"
            output_path.write_text("stale\n", encoding="utf-8")
            spec = SafetyDatasetSpec(
                name="tulu3_safety",
                output_path=str(output_path),
                force_rebuild=True,
            )
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
                materialize_safety_train_dataset(spec)
            with output_path.open("r", encoding="utf-8") as f:
                payload = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["id"], "kept")


# ---------------------------------------------------------------------------
# Eval-side loaders: verify the WildJailbreak loader passes the mandatory
# upstream loader options.
# ---------------------------------------------------------------------------


class SafetyEvalLoaderTests(unittest.TestCase):
    def test_wildjailbreak_eval_uses_required_loader_kwargs(self) -> None:
        recorded: dict = {}

        def fake_load_dataset(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return []

        with mock.patch.object(safety_eval_datasets, "_load_dataset", side_effect=fake_load_dataset):
            safety_eval_datasets.load_wildjailbreak_eval()

        self.assertEqual(recorded["args"], ("allenai/wildjailbreak", "eval"))
        self.assertEqual(recorded["kwargs"]["delimiter"], "\t")
        self.assertFalse(recorded["kwargs"]["keep_default_na"])

    def test_coconot_contrast_uses_test_split(self) -> None:
        recorded: dict = {}

        def fake_load_dataset(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return []

        with mock.patch.object(safety_eval_datasets, "_load_dataset", side_effect=fake_load_dataset):
            safety_eval_datasets.load_coconot_contrast()
        self.assertEqual(recorded["args"], ("allenai/coconot", "contrast"))
        self.assertEqual(recorded["kwargs"]["split"], "test")


# ---------------------------------------------------------------------------
# Config + factory wiring smoke
# ---------------------------------------------------------------------------


class ConfigWiringSmokeTests(unittest.TestCase):
    def test_safety_sft_configs_load(self) -> None:
        from src.baselines import load_sft_config

        for filename in (
            "configs/baseline_sft_qwen35_08b_tulu3_safety.yaml",
            "configs/baseline_sft_qwen35_08b_safety_tuned_llamas.yaml",
            "configs/baseline_sft_qwen35_08b_beavertails.yaml",
        ):
            cfg = load_sft_config(PROJECT_ROOT / filename)
            self.assertTrue(cfg.data.safety_dataset.name)
            self.assertIn(cfg.data.safety_dataset.name, SAFETY_TRAIN_DATASETS)

    def test_factory_constructs_three_specs(self) -> None:
        from src.baselines import load_sft_config

        # Round 2: tulu3 SFT YAML now points at the v2 builder.
        configs = {
            "tulu3_safety_v2": "configs/baseline_sft_qwen35_08b_tulu3_safety.yaml",
            "safety_tuned_llamas": "configs/baseline_sft_qwen35_08b_safety_tuned_llamas.yaml",
            "beavertails": "configs/baseline_sft_qwen35_08b_beavertails.yaml",
        }
        for expected_name, filename in configs.items():
            cfg = load_sft_config(PROJECT_ROOT / filename)
            spec = SafetyDatasetSpec(
                name=cfg.data.safety_dataset.name,
                output_path=cfg.data.train_split,
                source_name=cfg.data.safety_dataset.source_name or None,
                split=cfg.data.safety_dataset.split or None,
                sources=list(cfg.data.safety_dataset.sources) or None,
                repo_or_data_path=cfg.data.safety_dataset.repo_or_data_path or None,
                file_name=cfg.data.safety_dataset.file_name or None,
                refusal_template=cfg.data.safety_dataset.refusal_template or None,
            )
            self.assertEqual(spec.name, expected_name)
            self.assertIn(spec.name, SAFETY_TRAIN_DATASETS)


# ---------------------------------------------------------------------------
# OpenCompass dataset config presence
# ---------------------------------------------------------------------------


class OpenCompassConfigPresenceTests(unittest.TestCase):
    def test_required_dataset_configs_exist(self) -> None:
        oc_root = PROJECT_ROOT / "external" / "opencompass" / "opencompass" / "configs" / "datasets"
        if not oc_root.exists():
            self.skipTest(f"OpenCompass clone not present at {oc_root}")
        expected = [
            ("mmlu", "mmlu_gen.py"),
            ("gsm8k", "gsm8k_gen.py"),
            ("IFEval", "IFEval_gen.py"),
            ("humaneval", "humaneval_gen.py"),
            ("mbpp", "mbpp_gen.py"),
        ]
        for subdir, filename in expected:
            target = oc_root / subdir / filename
            self.assertTrue(target.exists(), f"Missing OpenCompass config: {target}")

    def test_eval_opencompass_lists_ifeval(self) -> None:
        # Late-imported because the script lives under scripts/ which isn't a package.
        spec_path = PROJECT_ROOT / "scripts" / "17_eval_opencompass.py"
        text = spec_path.read_text(encoding="utf-8")
        self.assertIn('"IFEval_gen"', text)
        self.assertIn("default=[]", text)
        self.assertIn("default=1024", text)

    def test_oneclick_and_shell_defaults_include_ifeval(self) -> None:
        oneclick_text = (PROJECT_ROOT / "scripts" / "15_run_oneclick.py").read_text(encoding="utf-8")
        shell_text = (PROJECT_ROOT / "run_experiments.sh").read_text(encoding="utf-8")
        self.assertIn('"IFEval_gen"', oneclick_text)
        self.assertIn("IFEval_gen", shell_text)

    def test_shell_exposes_safety_baselines(self) -> None:
        shell_text = (PROJECT_ROOT / "run_experiments.sh").read_text(encoding="utf-8")
        for name in ("tulu3_safety", "safety_tuned_llamas", "beavertails"):
            self.assertIn(name, shell_text)
        for name in (
            "distill_tulu3_safety",
            "distill_safety_tuned_llamas",
            "distill_beavertails",
            "ours_tulu3_safety",
            "ours_safety_tuned_llamas",
            "ours_beavertails",
            "safety_distill_all",
            "safety_full_all",
        ):
            self.assertIn(name, shell_text)
        self.assertIn("safety-distill", shell_text)
        self.assertIn("safety-full", shell_text)

    def test_safety_eval_wired_into_baseline_suite(self) -> None:
        eval_text = (PROJECT_ROOT / "scripts" / "12_eval_baseline_suite.py").read_text(encoding="utf-8")
        self.assertIn("SAFETY_EVAL_LOADERS", eval_text)
        self.assertIn("--safety-eval-datasets", eval_text)
        oneclick_text = (PROJECT_ROOT / "scripts" / "15_run_oneclick.py").read_text(encoding="utf-8")
        # Per-baseline routing replaces the old DEFAULT_SAFETY_EVAL_DATASETS knob.
        self.assertIn("SAFETY_EVAL_DATASETS_BY_BASELINE", oneclick_text)
        self.assertIn("SAFETY_EVAL_CONFIGS", oneclick_text)
        self.assertIn('"tulu3_safety": ("coconot_contrast",)', oneclick_text)
        self.assertIn("if safety_eval_datasets:", oneclick_text)
        self.assertIn('eval_args.extend(["--safety-eval-datasets", *safety_eval_datasets])', oneclick_text)


if __name__ == "__main__":
    unittest.main()
