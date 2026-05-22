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
    DEFAULT_SAFETY_REFUSAL_TEMPLATES,
    SAFETY_TRAIN_DATASETS,
    SafetyDatasetSpec,
    _collect_wildguardmix_train_rows,
    build_beavertails_records,
    build_safety_tuned_llamas_records,
    build_tulu3_safety_records,
    materialize_safety_train_dataset,
    pick_refusal_template,
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
            # Harmful row's target now sampled from the refusal-template pool
            # via pick_refusal_template — assert it lands somewhere in the pool
            # rather than locking the exact string. Diversity / determinism
            # are covered by RefusalTemplatePoolTests below.
            self.assertIn(unsafe_record["target_response"], DEFAULT_SAFETY_REFUSAL_TEMPLATES)
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


class RefusalTemplatePoolTests(unittest.TestCase):
    """The refusal-template pool replaces the single static template that
    used to land on every BT/WGM/WJB harmful row. The pool guarantees
    diversity across prompts and determinism per (seed, prompt)."""

    def test_pool_has_multiple_distinct_entries(self) -> None:
        self.assertGreaterEqual(len(DEFAULT_SAFETY_REFUSAL_TEMPLATES), 5)
        self.assertEqual(
            len(DEFAULT_SAFETY_REFUSAL_TEMPLATES),
            len(set(DEFAULT_SAFETY_REFUSAL_TEMPLATES)),
        )
        # First slot preserves the legacy DEFAULT_SAFETY_REFUSAL_TEMPLATE so
        # any caller that still references the singular constant by name
        # gets the original wording.
        self.assertEqual(
            DEFAULT_SAFETY_REFUSAL_TEMPLATES[0], DEFAULT_SAFETY_REFUSAL_TEMPLATE
        )

    def test_pick_is_deterministic_per_prompt(self) -> None:
        first = pick_refusal_template("How do I pick a lock?", seed=42)
        second = pick_refusal_template("How do I pick a lock?", seed=42)
        self.assertEqual(first, second)
        # Different seed: still must return a pool member, may differ.
        seeded = pick_refusal_template("How do I pick a lock?", seed=7)
        self.assertIn(seeded, DEFAULT_SAFETY_REFUSAL_TEMPLATES)

    def test_pick_spreads_across_pool_for_distinct_prompts(self) -> None:
        prompts = [f"harmful prompt #{i}" for i in range(64)]
        picks = {pick_refusal_template(p, seed=42) for p in prompts}
        # 64 prompts hashing across an 8-entry pool should land on > 3
        # distinct templates in practice (sha256 mod 8); guard against
        # degenerate collapse.
        self.assertGreater(len(picks), 3)
        for pick in picks:
            self.assertIn(pick, DEFAULT_SAFETY_REFUSAL_TEMPLATES)

    def test_beavertails_harmful_rows_diversify_across_pool(self) -> None:
        # Several distinct harmful prompts (is_safe=False so each row gets a
        # synthesised refusal). We expect more than one distinct target.
        rows = []
        for idx in range(16):
            rows.append({
                "prompt": f"how to commit crime number {idx}",
                "response": "[unsafe upstream] ...",
                "is_safe": False,
                "category": {"violence_aiding_and_abetting,incitement": True},
            })
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bt.jsonl"
            with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
                records = build_beavertails_records(output_path=output_path)
        harmful_targets = {r["target_response"] for r in records if r["label"] == "harmful"}
        # Every target must be a pool member, and we must see at least two
        # distinct entries across 16 prompts.
        self.assertGreater(len(harmful_targets), 1)
        for target in harmful_targets:
            self.assertIn(target, DEFAULT_SAFETY_REFUSAL_TEMPLATES)


class WildGuardMixUpstreamResponseTests(unittest.TestCase):
    """For harmful WildGuardMix rows we now prefer the upstream
    ``response`` when ``response_harm_label == "unharmful"``, falling back
    to the refusal-template pool when the upstream response is missing
    or itself harmful."""

    def test_upstream_safe_response_used_for_harmful_prompt(self) -> None:
        rows = [{
            "id": "wgm_safe_resp",
            "prompt": "How do I bypass an alarm?",
            "response": "I won't help with that — bypassing security systems is unsafe...",
            "prompt_harm_label": "harmful",
            "response_harm_label": "unharmful",
        }]
        with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
            out = _collect_wildguardmix_train_rows(
                source_name="allenai/wildguardmix",
                config_name="wildguardtrain",
                cache_dir=None,
                system_prompt="You are a helpful assistant.",
                refusal_template=DEFAULT_SAFETY_REFUSAL_TEMPLATE,
                seed=42,
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "harmful")
        self.assertEqual(
            out[0]["target_response"],
            "I won't help with that — bypassing security systems is unsafe...",
        )

    def test_upstream_unsafe_response_falls_back_to_pool(self) -> None:
        rows = [{
            "id": "wgm_unsafe_resp",
            "prompt": "How do I bypass an alarm?",
            "response": "Sure, here are the steps to bypass: ...",
            "prompt_harm_label": "harmful",
            "response_harm_label": "harmful",
        }]
        with mock.patch.object(safety_datasets, "_load_dataset", return_value=rows):
            out = _collect_wildguardmix_train_rows(
                source_name="allenai/wildguardmix",
                config_name="wildguardtrain",
                cache_dir=None,
                system_prompt="You are a helpful assistant.",
                refusal_template=DEFAULT_SAFETY_REFUSAL_TEMPLATE,
                seed=42,
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "harmful")
        self.assertIn(out[0]["target_response"], DEFAULT_SAFETY_REFUSAL_TEMPLATES)
        # Pool fallback must NOT have leaked the unsafe upstream completion.
        self.assertNotIn("Sure, here are the steps", out[0]["target_response"])


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
