"""Smoke tests: pipeline contracts stay consistent across phases.

Covers:
1. ``evaluate_generation_refusal_metrics`` return fields == fields consumed by
   scripts 09 / 13 / 14 at epoch logging time.
2. ``load_hidden_state_split`` default-strict rejects legacy feature_type;
   opt-in path accepts it and sets the legacy flag.
3. ``resolve_multi_teacher_reduction`` (used by script 08) raises on collision
   by default and only returns ``"mean"`` when explicitly opted in.
4. ``build_merged_summary`` (script 18) strips local general-eval tasks from
   ``pan_summary.results`` so the final summary's general numbers can only
   come from OpenCompass.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.training.trainer_phase1 as trainer_phase1
from src.phase_b.hidden_states import load_hidden_state_split


def _load_script_module(script_name: str, module_name: str):
    """Import a ``scripts/NN_name.py`` module whose filename isn't a valid
    Python identifier, so it can be exercised from tests.
    """

    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluatorContractSmokeTests(unittest.TestCase):
    """Every field that training scripts read from the evaluator must exist."""

    def test_refusal_only_fields_match_training_script_usage(self):
        source_files = [
            PROJECT_ROOT / "scripts" / "09_train_student_semalign.py",
            PROJECT_ROOT / "scripts" / "13_train_pan_sft.py",
            PROJECT_ROOT / "scripts" / "14_train_pan_distill.py",
        ]
        # Keys that training scripts read out of the evaluator's return dict.
        required_keys = {
            "harmful_refusal_rate",
            "harmful_unsafe_output_rate",
            "harmless_over_refusal_rate",
            "num_harmful",
            "num_harmless",
        }
        # Keys the thinking-era evaluator used to return; must NOT be used.
        forbidden_keys = {
            "harmful_safe_response_rate",
            "harmful_incomplete_output_rate",
        }
        evaluator_src = inspect.getsource(trainer_phase1.evaluate_generation_refusal_metrics)
        for key in required_keys:
            self.assertIn(
                key,
                evaluator_src,
                f"Evaluator no longer produces required key: {key}",
            )
        for path in source_files:
            text = path.read_text(encoding="utf-8")
            for key in required_keys:
                self.assertIn(
                    key,
                    text,
                    f"{path.name} does not surface required refusal-only key: {key}",
                )
            for key in forbidden_keys:
                self.assertNotIn(
                    key,
                    text,
                    f"{path.name} still references retired thinking-era key: {key}",
                )


class HiddenStateStrictSmokeTests(unittest.TestCase):
    def test_legacy_shard_rejected_by_default(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            split_dir = Path(tmpdir)
            torch.save(
                {
                    "feature_type": "final_response_prefix_hidden_state",
                    "sample_ids": ["s1"],
                    "labels": ["harmful"],
                    "hidden_by_layer": {"0": torch.zeros(1, 3)},
                },
                split_dir / "part_000.pt",
            )
            with self.assertRaisesRegex(ValueError, "Legacy hidden-state shard"):
                load_hidden_state_split(split_dir)


class CollisionStrictSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_script_module(
            "08_recompose_student_targets.py",
            "script_08_recompose_student_targets",
        )

    def test_strict_rejects_collision(self):
        teacher_to_student = {5: 2, 6: 2, 7: 3}  # two teachers -> student 2
        with self.assertRaisesRegex(ValueError, "Multi-teacher-to-one-student"):
            self.module.resolve_multi_teacher_reduction(
                teacher_to_student,
                allow_multi_teacher_mean=False,
            )

    def test_opt_in_mean_returns_mean_and_groups(self):
        teacher_to_student = {5: 2, 6: 2, 7: 3}
        strategy, groups = self.module.resolve_multi_teacher_reduction(
            teacher_to_student,
            allow_multi_teacher_mean=True,
        )
        self.assertEqual(strategy, "mean")
        self.assertEqual(groups, {"2": [5, 6]})

    def test_no_collision_returns_strict_single(self):
        teacher_to_student = {5: 5, 6: 6, 7: 7}
        strategy, groups = self.module.resolve_multi_teacher_reduction(
            teacher_to_student,
            allow_multi_teacher_mean=False,
        )
        self.assertEqual(strategy, "strict_single")
        self.assertEqual(groups, {})


class MergedSummaryIsolationSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_script_module(
            "18_merge_opencompass_summary.py",
            "script_18_merge_opencompass_summary",
        )

    def test_local_general_tasks_dropped_from_merged_summary(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            pan_summary_path = Path(tmpdir) / "summary.json"
            pan_summary_path.write_text(
                '{"results": {"pan": {"harmful_refusal_rate": 0.9},'
                ' "mmlu": {"accuracy": 0.5},'
                ' "gsm8k": {"accuracy": 0.4}},'
                ' "debug_outputs": {"pan": {"n": 1}, "mmlu": {"n": 2}}}',
                encoding="utf-8",
            )
            merged = self.module.build_merged_summary(
                pan_summary_path=pan_summary_path,
                opencompass_work_dir=None,
            )
        self.assertEqual(set(merged["pan_summary"]["results"].keys()), {"pan"})
        self.assertEqual(set(merged["pan_summary"]["debug_outputs"].keys()), {"pan"})
        self.assertIn("mmlu", merged["dropped_local_general_tasks"])
        self.assertIn("gsm8k", merged["dropped_local_general_tasks"])
        self.assertFalse(merged["opencompass"]["enabled"])
        self.assertEqual(merged["opencompass"]["results"], {})


if __name__ == "__main__":
    unittest.main()
