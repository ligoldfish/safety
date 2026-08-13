from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.config import load_distill_config, load_eval_config, load_sft_config
from src.ablations.efficiency import (
    append_efficiency_record,
    phase_for_script,
    run_profiled_subprocess,
)
from src.ablations.platform import resolve_portable_path
from src.utils.config import load_phase1_config, load_phasef_config
from src.pairs import DEFAULT_PAIR, PAIRS, apply_tokens

# Active teacher->student model pair for this invocation (set in main() from
# --pair). All config paths flow through _resolve(), which re-targets the
# canonical Qwen3.5-9B->0.8B config names to this pair via apply_tokens. For the
# default pair apply_tokens is the identity, so the original flow is byte-for-byte
# unchanged; only new pairs (Llama-3.x / Qwen3) get re-mapped filenames.
_ACTIVE_PAIR = DEFAULT_PAIR


def _pair_suffix() -> str:
    """'' for the default pair (legacy safety-full/sft1 output dirs unchanged);
    '_<pair>' otherwise, so different pairs' safety_full_<baseline>_<device>
    output roots stay disjoint (ours + sft1 still share one per-pair phase1)."""
    return "" if _ACTIVE_PAIR == DEFAULT_PAIR else f"_{_ACTIVE_PAIR}"


BASELINE_EVAL_CONFIGS = {
    ("npu", "0.8b"): "configs/baseline_eval_qwen35_08b_npu.yaml",
    ("ppu", "0.8b"): "configs/baseline_eval_qwen35_08b_ppu.yaml",
    ("npu", "9b"): "configs/baseline_eval_qwen35_9b_npu.yaml",
    ("ppu", "9b"): "configs/baseline_eval_qwen35_9b_ppu.yaml",
}

BASELINE_SFT_CONFIGS = {
    ("npu", "0.8b"): "configs/baseline_sft_qwen35_08b_npu.yaml",
    ("ppu", "0.8b"): "configs/baseline_sft_qwen35_08b_ppu.yaml",
    ("npu", "9b"): "configs/baseline_sft_qwen35_9b_npu.yaml",
    ("ppu", "9b"): "configs/baseline_sft_qwen35_9b_ppu.yaml",
}

BASELINE_DISTILL_CONFIGS = {
    "npu": "configs/baseline_distill_qwen35_9b_to_08b_npu.yaml",
    "ppu": "configs/baseline_distill_qwen35_9b_to_08b_ppu.yaml",
}

# Safety baselines added per data-augmentation plan (2026-05-04).
# Keyed by (device, model, baseline_name); only 0.8B has explicit configs
# because the upstream Tülu/Safety-Tuned-LLaMAs/BeaverTails recipes were
# specified for that scale.
SAFETY_SFT_BASELINES = (
    "tulu3_safety",
    "safety_tuned_llamas",
    "beavertails",
    "wildjailbreak",
    "wildguardmix",
    "hh_rlhf",
    "beavertails_category",
    "coconot",
    "c5",
)
# Keep all one-click experiment summaries comparable while this experiment
# matrix is still exploratory: safety-specific variants default to the same
# PAN transfer test as the ordinary baselines. External safety suites remain
# available via scripts/12_eval_baseline_suite.py --safety-eval-datasets.
DEFAULT_SAFETY_EVAL_DATASETS: tuple[str, ...] = ()
SAFETY_SFT_CONFIGS: dict[tuple[str, str, str], str] = {
    ("npu", "0.8b", "tulu3_safety"): "configs/baseline_sft_qwen35_08b_tulu3_safety_npu.yaml",
    ("ppu", "0.8b", "tulu3_safety"): "configs/baseline_sft_qwen35_08b_tulu3_safety_ppu.yaml",
    ("npu", "0.8b", "safety_tuned_llamas"): "configs/baseline_sft_qwen35_08b_safety_tuned_llamas_npu.yaml",
    ("ppu", "0.8b", "safety_tuned_llamas"): "configs/baseline_sft_qwen35_08b_safety_tuned_llamas_ppu.yaml",
    ("npu", "0.8b", "beavertails"): "configs/baseline_sft_qwen35_08b_beavertails_npu.yaml",
    ("ppu", "0.8b", "beavertails"): "configs/baseline_sft_qwen35_08b_beavertails_ppu.yaml",
    ("npu", "0.8b", "wildjailbreak"): "configs/baseline_sft_qwen35_08b_wildjailbreak_npu.yaml",
    ("npu", "0.8b", "wildguardmix"): "configs/baseline_sft_qwen35_08b_wildguardmix_npu.yaml",
    ("npu", "0.8b", "hh_rlhf"): "configs/baseline_sft_qwen35_08b_hh_rlhf_npu.yaml",
    ("npu", "0.8b", "beavertails_category"): "configs/baseline_sft_qwen35_08b_beavertails_category_npu.yaml",
    ("npu", "0.8b", "coconot"): "configs/baseline_sft_qwen35_08b_coconot_npu.yaml",
    ("npu", "0.8b", "c5"): "configs/baseline_sft_qwen35_08b_c5_npu.yaml",
}

# Distillation safety baselines: 9B teacher → 0.8B student, training corpus
# replaced by Tülu3 safety / Safety-Tuned LLaMAs / BeaverTails.
# Per-baseline eval configs that point datasets.pan.path at the baseline's
# own held-out test JSONL (built by scripts/21_build_baseline_eval_jsonls.py).
# Falls back to the canonical PAN eval YAML when no entry matches.
SAFETY_EVAL_CONFIGS: dict[tuple[str, str, str], str] = {
    ("npu", "0.8b", "tulu3_safety"): "configs/baseline_eval_qwen35_08b_tulu3_safety_npu.yaml",
    ("ppu", "0.8b", "tulu3_safety"): "configs/baseline_eval_qwen35_08b_tulu3_safety_ppu.yaml",
    ("npu", "0.8b", "safety_tuned_llamas"): "configs/baseline_eval_qwen35_08b_safety_tuned_llamas_npu.yaml",
    ("ppu", "0.8b", "safety_tuned_llamas"): "configs/baseline_eval_qwen35_08b_safety_tuned_llamas_ppu.yaml",
    ("npu", "0.8b", "beavertails"): "configs/baseline_eval_qwen35_08b_beavertails_npu.yaml",
    ("ppu", "0.8b", "beavertails"): "configs/baseline_eval_qwen35_08b_beavertails_ppu.yaml",
    ("npu", "0.8b", "wildjailbreak"): "configs/baseline_eval_qwen35_08b_wildjailbreak_npu.yaml",
    ("npu", "0.8b", "wildguardmix"): "configs/baseline_eval_qwen35_08b_wildguardmix_npu.yaml",
    ("npu", "0.8b", "hh_rlhf"): "configs/baseline_eval_qwen35_08b_hh_rlhf_npu.yaml",
    ("npu", "0.8b", "beavertails_category"): "configs/baseline_eval_qwen35_08b_beavertails_category_npu.yaml",
    ("npu", "0.8b", "coconot"): "configs/baseline_eval_qwen35_08b_coconot_npu.yaml",
    ("npu", "0.8b", "c5"): "configs/baseline_eval_qwen35_08b_c5_npu.yaml",
    # 9B teacher-model nosft eval configs (generated by scripts/gen_9b_baseline_eval_yamls.py)
    ("npu", "9b", "tulu3_safety"): "configs/baseline_eval_qwen35_9b_tulu3_safety_npu.yaml",
    ("ppu", "9b", "tulu3_safety"): "configs/baseline_eval_qwen35_9b_tulu3_safety_ppu.yaml",
    ("npu", "9b", "safety_tuned_llamas"): "configs/baseline_eval_qwen35_9b_safety_tuned_llamas_npu.yaml",
    ("ppu", "9b", "safety_tuned_llamas"): "configs/baseline_eval_qwen35_9b_safety_tuned_llamas_ppu.yaml",
    ("npu", "9b", "beavertails"): "configs/baseline_eval_qwen35_9b_beavertails_npu.yaml",
    ("ppu", "9b", "beavertails"): "configs/baseline_eval_qwen35_9b_beavertails_ppu.yaml",
    ("npu", "9b", "wildjailbreak"): "configs/baseline_eval_qwen35_9b_wildjailbreak_npu.yaml",
    ("npu", "9b", "wildguardmix"): "configs/baseline_eval_qwen35_9b_wildguardmix_npu.yaml",
    ("npu", "9b", "hh_rlhf"): "configs/baseline_eval_qwen35_9b_hh_rlhf_npu.yaml",
    ("npu", "9b", "beavertails_category"): "configs/baseline_eval_qwen35_9b_beavertails_category_npu.yaml",
    ("npu", "9b", "coconot"): "configs/baseline_eval_qwen35_9b_coconot_npu.yaml",
}

# Per-baseline external safety suites passed via --safety-eval-datasets.
# Tülu3 baseline gets CoCoNot contrast for over-refusal probing because
# WildGuardTest + WildJailbreak alone do not cover the "looks-harmful but
# legitimate" axis. The other baselines already carry both polarities in
# their per-baseline JSONL.
SAFETY_EVAL_DATASETS_BY_BASELINE: dict[str, tuple[str, ...]] = {
    "tulu3_safety": ("coconot_contrast",),
    "safety_tuned_llamas": (),
    "beavertails": (),
    "wildjailbreak": (),
    "wildguardmix": (),
    "hh_rlhf": (),
    "beavertails_category": (),
    "coconot": (),
    "c5": (),
}

# Per-baseline Phase-1 subspace + PhaseF overrides for safety-full / ours.
# WildJailbreak is heavily adversarial: its harmful/harmless contrast Δ_l is
# noisy, so a cleaner, more compact safety subspace + more PhaseF epochs help
# the intent alignment (L_layer) converge instead of chasing adversarial-style
# noise (which conflates "looks-like-jailbreak" with "harmful intent" and
# inflates over-refusal on adversarial-benign prompts):
#   * --top-k 3            : fewer, most safety-discriminative key layers.
#   * --energy-threshold 0.7: lower tau -> keep only the dominant singular
#                             directions per layer, drop the noisy low-energy tail.
#   * --rank-cap 8         : hard cap on per-layer effective rank (vs global 32).
#   * phasef_epochs 5      : more epochs (vs global 3) to converge on the harder
#                             distribution.
# Caller-supplied --phase1-analyze-extra / --phase1-subspace-extra are appended
# AFTER these, so an explicit CLI override still wins (argparse last-wins).
# Other baselines keep the global argparse / yaml defaults.
SAFETY_PHASE_OVERRIDES_BY_BASELINE: dict[str, dict[str, object]] = {
    "wildjailbreak": {
        # Dirty/adversarial contrast -> compact, denoised subspace + more epochs.
        "phasef_epochs": 5,
        "analyze_extra": ("--top-k", "3"),
        "subspace_extra": ("--energy-threshold", "0.7", "--rank-cap", "8"),
    },
    "wildguardmix": {
        # WGM is vanilla-only after round-2 -> a CLEAN contrast, the opposite of
        # WJB. A richer subspace (more key layers + higher energy threshold keeps
        # more genuine intent directions) plus a stronger L_layer weight sharpens
        # intent discrimination -> lower OR at a given HR and helps ours beat the
        # sft1 (L_layer=0) ablation. Epochs left at the default 3 (more epochs
        # lowers ours' HR further, away from the baseline HR band).
        "phasef_layer_loss_weight": 0.5,
        "analyze_extra": ("--top-k", "7"),
        "subspace_extra": ("--energy-threshold", "0.9"),
    },
}


def _safety_eval_config(device: str, model_size: str, baseline: str) -> str:
    """Return baseline-specific eval YAML or fall back to the PAN eval YAML."""

    return SAFETY_EVAL_CONFIGS.get(
        (device, model_size, baseline),
        BASELINE_EVAL_CONFIGS[(device, model_size)],
    )


SAFETY_DISTILL_CONFIGS: dict[tuple[str, str], str] = {
    ("npu", "tulu3_safety"): "configs/baseline_distill_qwen35_9b_to_08b_tulu3_safety_npu.yaml",
    ("ppu", "tulu3_safety"): "configs/baseline_distill_qwen35_9b_to_08b_tulu3_safety_ppu.yaml",
    ("npu", "safety_tuned_llamas"): "configs/baseline_distill_qwen35_9b_to_08b_safety_tuned_llamas_npu.yaml",
    ("ppu", "safety_tuned_llamas"): "configs/baseline_distill_qwen35_9b_to_08b_safety_tuned_llamas_ppu.yaml",
    ("npu", "beavertails"): "configs/baseline_distill_qwen35_9b_to_08b_beavertails_npu.yaml",
    ("ppu", "beavertails"): "configs/baseline_distill_qwen35_9b_to_08b_beavertails_ppu.yaml",
    ("npu", "wildjailbreak"): "configs/baseline_distill_qwen35_9b_to_08b_wildjailbreak_npu.yaml",
    ("npu", "wildguardmix"): "configs/baseline_distill_qwen35_9b_to_08b_wildguardmix_npu.yaml",
    ("npu", "hh_rlhf"): "configs/baseline_distill_qwen35_9b_to_08b_hh_rlhf_npu.yaml",
    ("npu", "beavertails_category"): "configs/baseline_distill_qwen35_9b_to_08b_beavertails_category_npu.yaml",
    ("npu", "coconot"): "configs/baseline_distill_qwen35_9b_to_08b_coconot_npu.yaml",
    ("npu", "c5"): "configs/baseline_distill_qwen35_9b_to_08b_c5_npu.yaml",
}

FULL_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_npu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_npu.yaml",
    },
    "ppu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_ppu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_ppu.yaml",
    },
}

RANDOM_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_npu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_npu_random.yaml",
    },
    "ppu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_ppu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_ppu_random.yaml",
    },
}

# sft1 ablation: sft_loss_weight=1.0, layer_loss_weight=0.0. Same phase1
# precompute as the main ours run; only the phaseF training step swaps in
# the sft1 yaml whose output_root is "<phase1>/training_sft1" (so it does
# not collide with the main "<phase1>/training" or the random ablation's
# "<phase1>/training_random_same_norm").
SFT1_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_npu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_npu_sft1.yaml",
    },
    "ppu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_ppu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_ppu_sft1.yaml",
    },
}

# bothpole ours variant: same Phase 1 precompute as the main ours run; the
# phaseF step swaps in the both-pole yaml (layer_loss_policy=label_weighted)
# whose output_root ends in "<phase1>/training_bothpole", so it never collides
# with the main "<phase1>/training" (harmful_only) run and can train in parallel.
BOTHPOLE_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_9b_to_08b_phase1_npu.yaml",
        "phasef": "configs/qwen35_9b_to_08b_phaseF_npu_bothpole.yaml",
    },
}

PIPELINE_SPLITS = ["alignment", "analysis_val", "pan_test", "sanity_test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click launcher for baseline and full-stage experiments on NPU/PPU."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument(
            "--device",
            choices=["npu", "ppu"],
            required=True,
            help="Accelerator backend to use.",
        )
        target_parser.add_argument(
            "--device-id",
            type=int,
            default=0,
            help="Primary accelerator ordinal. NPU maps to npu:<id>; PPU maps to ppu:<id>.",
        )
        target_parser.add_argument(
            "--num-devices",
            type=int,
            default=1,
            help="Requested accelerator count. The current code path is single-process single-device, so only 1 is supported.",
        )
        target_parser.add_argument(
            "--pair",
            choices=sorted(PAIRS),
            default=DEFAULT_PAIR,
            help=(
                "Teacher->student model pair (src/pairs.py). Default reproduces the "
                "original Qwen3.5-9B->0.8B flow unchanged. Other pairs require their "
                "configs to exist (scripts/gen_pair_configs.py) and models downloaded."
            ),
        )
        target_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the commands without executing them.",
        )
        target_parser.add_argument(
            "--enable-opencompass",
            action="store_true",
            help=(
                "Explicit gate for OpenCompass general-capability eval. Default: disabled. "
                "Even when enabled, --opencompass-dir must point at a cloned OpenCompass repo."
            ),
        )
        target_parser.add_argument(
            "--opencompass-dir",
            default="",
            help=(
                "Path to a cloned OpenCompass repo. Required when --enable-opencompass is set. "
                "When omitted (or --skip-opencompass is passed), only PAN safety eval runs and "
                "general datasets stay disabled in the final summary."
            ),
        )
        target_parser.add_argument(
            "--opencompass-config",
            default="",
            help=(
                "Optional extra config/path forwarded verbatim to scripts/17_eval_opencompass.py "
                "(via --extra-args). Reserved for future OpenCompass YAML configs; ignored when empty."
            ),
        )
        target_parser.add_argument(
            "--opencompass-datasets",
            nargs="+",
            default=["mmlu_gen", "gsm8k_gen", "IFEval_gen", "humaneval_gen", "mbpp_gen"],
            help="Datasets forwarded to scripts/17_eval_opencompass.py --datasets.",
        )
        target_parser.add_argument(
            "--skip-opencompass",
            action="store_true",
            help=(
                "Force-skip OpenCompass general-capability eval even when --enable-opencompass "
                "or --opencompass-dir is provided."
            ),
        )

    nosft_parser = subparsers.add_parser("nosft", help="Run no-SFT benchmark evaluation.")
    nosft_parser.add_argument(
        "--role",
        choices=["student", "teacher"],
        default="student",
        help="Which model of the --pair to eval (student or teacher). Replaces --model.",
    )
    nosft_parser.add_argument(
        "--model",
        choices=["0.8b", "9b"],
        default=None,
        help="DEPRECATED legacy alias (0.8b=student, 9b=teacher); use --role. Overrides --role if set.",
    )
    nosft_parser.add_argument(
        "--baseline",
        choices=["pan", *SAFETY_SFT_BASELINES, "all"],
        default="pan",
        help=(
            "Which test set to evaluate the base model on. "
            "'pan' (default) keeps back-compat with PAN eval. "
            "'tulu3_safety' / 'beavertails' / 'safety_tuned_llamas' use the "
            "per-baseline JSONL produced by scripts/21_build_baseline_eval_jsonls.py. "
            "'all' loops over PAN + the three safety baselines sequentially."
        ),
    )
    add_common_flags(nosft_parser)

    sft_parser = subparsers.add_parser("sft", help="Run PAN SFT and then benchmark evaluation.")
    sft_parser.add_argument(
        "--role",
        choices=["student", "teacher"],
        default="student",
        help="Which model of the --pair to SFT (student or teacher). Replaces --model.",
    )
    sft_parser.add_argument(
        "--model",
        choices=["0.8b", "9b"],
        default=None,
        help="DEPRECATED legacy alias (0.8b=student, 9b=teacher); use --role. Overrides --role if set.",
    )
    sft_parser.add_argument(
        "--baseline",
        choices=["pan", *SAFETY_SFT_BASELINES],
        default="pan",
        help="'pan' keeps the legacy PAN SFT flow; safety values route to safety-sft.",
    )
    sft_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety JSONL when --baseline is not pan.",
    )
    add_common_flags(sft_parser)

    distill_parser = subparsers.add_parser("distill", help="Run PAN distillation and then benchmark evaluation.")
    distill_parser.add_argument(
        "--baseline",
        choices=["pan", *SAFETY_SFT_BASELINES],
        default="pan",
        help="'pan' keeps the legacy PAN distill flow; safety values route to safety-distill.",
    )
    distill_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety JSONL when --baseline is not pan.",
    )
    add_common_flags(distill_parser)

    safety_parser = subparsers.add_parser(
        "safety-sft",
        help=(
            "Run a safety-augmented SFT baseline on Tülu 3 safety / "
            "Safety-Tuned LLaMAs / BeaverTails, followed by safety eval and "
            "OpenCompass general-capability eval."
        ),
    )
    safety_parser.add_argument(
        "--baseline",
        choices=list(SAFETY_SFT_BASELINES),
        required=True,
        help="Which safety SFT corpus to train on.",
    )
    safety_parser.add_argument("--model", choices=["0.8b"], default="0.8b")
    safety_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety training JSONL even when it exists.",
    )
    add_common_flags(safety_parser)

    safety_distill_parser = subparsers.add_parser(
        "safety-distill",
        help=(
            "Run 9B → 0.8B PAN-style distillation but with the training "
            "corpus replaced by Tülu 3 safety / Safety-Tuned LLaMAs / "
            "BeaverTails."
        ),
    )
    safety_distill_parser.add_argument(
        "--baseline",
        choices=list(SAFETY_SFT_BASELINES),
        required=True,
        help="Which safety corpus to distill on.",
    )
    safety_distill_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety training JSONL even when it exists.",
    )
    add_common_flags(safety_distill_parser)

    safety_full_parser = subparsers.add_parser(
        "safety-full",
        help=(
            "Run the full 00→11 SemAlign main experiment but with the "
            "training corpus replaced by Tülu 3 safety / Safety-Tuned "
            "LLaMAs / BeaverTails. The safety JSONL is split into PAN-style "
            "alignment/val/sanity sets so Phase 1-E can build a teacher "
            "safe subspace and student target map keyed by safety-record IDs."
        ),
    )
    safety_full_parser.add_argument(
        "--baseline",
        choices=list(SAFETY_SFT_BASELINES),
        required=True,
        help="Which safety corpus to train SemAlign on.",
    )
    safety_full_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety training JSONL even when it exists.",
    )
    add_common_flags(safety_full_parser)

    random_parser = subparsers.add_parser(
        "random",
        help="Run the random-vector baseline on the original 00->11 pipeline.",
    )
    random_parser.add_argument(
        "--baseline",
        choices=["pan", *SAFETY_SFT_BASELINES],
        default="pan",
        help="'pan' keeps the legacy PAN random flow; safety values run random on that safety corpus.",
    )
    random_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety JSONL when --baseline is not pan.",
    )
    add_common_flags(random_parser)

    sft1_parser = subparsers.add_parser(
        "sft1",
        help=(
            "Run the sft1 ablation (sft_loss_weight=1.0, layer_loss_weight=0.0). "
            "When --baseline is 'pan' (default) the PAN sft1 phaseF yaml is used "
            "and phase1 reuses ../outputs/qwen35_9b_to_08b_phase1_<device>/. When "
            "--baseline is a safety dataset the safety-full pipeline is run with "
            "the sft1 phaseF base, writing under phase1/training_sft1/ so the main "
            "ours / random / safety-full results stay intact."
        ),
    )
    sft1_parser.add_argument(
        "--baseline",
        choices=["pan", *SAFETY_SFT_BASELINES],
        default="pan",
        help=(
            "'pan' (default) runs the PAN sft1 ablation. The three safety values "
            "run safety-full's pipeline with the sft1 phaseF yaml on that dataset."
        ),
    )
    sft1_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety JSONL even when it exists (safety baselines only).",
    )
    add_common_flags(sft1_parser)

    bothpole_parser = subparsers.add_parser(
        "bothpole",
        help=(
            "Run the both-pole ours variant on the PAN corpus (L_layer supervises "
            "harmful AND harmless via layer_loss_policy=label_weighted, vs the main "
            "harmful_only run). Reuses ../outputs/qwen35_9b_to_08b_phase1_<device>/ "
            "and writes under phase1/training_bothpole/ so the main ours result is "
            "untouched; safe to run in parallel on another device."
        ),
    )
    add_common_flags(bothpole_parser)

    full_parser = subparsers.add_parser("full", help="Run the original 00->11 full-stage pipeline.")
    full_parser.add_argument(
        "--baseline",
        choices=["pan", *SAFETY_SFT_BASELINES],
        default="pan",
        help="'pan' keeps the legacy PAN full flow; safety values route to safety-full.",
    )
    full_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-materialize the safety JSONL when --baseline is not pan.",
    )
    full_parser.add_argument(
        "--phasef-config",
        default="",
        help=(
            "Optional PhaseF YAML override. Use this for ablations that reuse the "
            "same Phase1 artifacts but write PhaseF training/eval outputs to a "
            "different output.output_root. For safety baselines, this overrides "
            "the per-baseline PhaseF base yaml before _make_safety_full_overrides "
            "injects safety-specific paths."
        ),
    )
    full_parser.add_argument(
        "--phase1-config",
        default="",
        help=(
            "Optional Phase1 YAML override. Used for parallel sweep cells to "
            "isolate Phase1 outputs into per-cell directories without racing on "
            "the shared base yaml. For safety baselines, overrides the Phase1 "
            "base before _make_safety_full_overrides redirects processed_dir."
        ),
    )
    full_parser.add_argument(
        "--phase1-analyze-extra",
        default="[]",
        help=(
            "JSON list of extra CLI tokens forwarded to 02_analyze_teacher_layers.py, "
            'e.g. --phase1-analyze-extra=\'["--top-k","5"]\'. Use to ablate top-K '
            "key layers without modifying the script default. JSON form chosen so "
            'argparse does not mis-parse "--"-prefixed values as new flags.'
        ),
    )
    full_parser.add_argument(
        "--phase1-subspace-extra",
        default="[]",
        help=(
            "JSON list of extra CLI tokens forwarded to 03_build_teacher_safe_subspace.py, "
            'e.g. --phase1-subspace-extra=\'["--rank","8"]\' or '
            '--phase1-subspace-extra=\'["--no-balance-labels"]\'. JSON list (see '
            "--phase1-analyze-extra rationale)."
        ),
    )
    full_parser.add_argument(
        "--phase1-stage-extras",
        default="{}",
        help=(
            "JSON object mapping Phase-1 stage names to argv token lists. "
            "Supported keys: extract, analyze, subspace, pairing, bridge, "
            "project, decompose, recompose. Tokens are forwarded as argv and "
            "never passed through a shell."
        ),
    )
    full_parser.add_argument(
        "--cell-id",
        default="",
        help=(
            "Optional sweep cell identifier (e.g. 'B1'). When set on a safety "
            "baseline run, appends '_<cell-id>' to outputs/safety_full_<baseline>_<device>/ "
            "so concurrent sweep cells write to isolated dirs and do not clobber "
            "each other's checkpoints / eval_suite. PAN flow gets isolation via "
            "--phase1-config / --phasef-config output_root overrides instead."
        ),
    )
    full_parser.add_argument(
        "--disable-dataset-overrides",
        action="store_true",
        help="Disable WJB/WGM dataset-specific Phase1/PhaseF overrides for global-default fairness cells.",
    )
    add_common_flags(full_parser)
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    # Re-target canonical (qwen35_9b_to_08b) config paths to the active pair.
    # Identity for the default pair; only config FILE PATHS pass through here, so
    # the model-name tokens never match a path string (safe for all pairs).
    path_text = apply_tokens(path_text, _ACTIVE_PAIR)
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _run_script(
    script_name: str,
    args: Sequence[str],
    *,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path), *args]
    rendered = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(rendered)
    if env_overrides:
        print(
            "env:",
            " ".join(f"{key}={value}" for key, value in sorted(env_overrides.items())),
        )
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    profile_log = str(env.get("SAFETY_ABLATION_RUNTIME_LOG", "")).strip()
    phase = phase_for_script(script_name)
    if profile_log and phase:
        output_root = str(
            env.get("SAFETY_ABLATION_PROFILE_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
        )
        cell_id = str(env.get("SAFETY_ABLATION_CELL_ID", ""))
        try:
            device_count = int(env.get("SAFETY_ABLATION_DEVICE_COUNT", "1"))
        except ValueError as exc:
            raise ValueError("SAFETY_ABLATION_DEVICE_COUNT must be an integer") from exc
        returncode, record = run_profiled_subprocess(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stage=phase,
            script=script_name,
            output_root=output_root,
            cell_id=cell_id,
            device_count=device_count,
        )
        append_efficiency_record(profile_log, record)
        if returncode:
            raise subprocess.CalledProcessError(returncode, cmd)
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _all_epoch_checkpoints(training_dir: Path, *, epochs: int, dry_run: bool) -> list[Path]:
    """Every per-epoch checkpoint the trainer writes, oldest-first.

    On a dry run the checkpoints do not exist yet, so synthesize the expected
    ``epoch_001.pt .. epoch_{epochs}.pt`` names from the configured epoch count.
    """

    checkpoint_dir = training_dir / "checkpoints"
    if dry_run:
        return [checkpoint_dir / f"epoch_{idx:03d}.pt" for idx in range(1, max(int(epochs), 1) + 1)]
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_dir}")
    return candidates


def _validate_device_request(num_devices: int) -> None:
    if num_devices != 1:
        raise ValueError(
            "The current launcher only supports single-process single-device execution. "
            "Use --num-devices 1. Multi-device NPU/PPU parallelism would require a distributed training path."
        )


def _runtime_device_value(device: str, device_id: int) -> str:
    if device == "npu":
        return "npu:0"
    if device == "ppu":
        return f"ppu:{device_id}"
    raise ValueError(f"Unsupported device: {device}")


def _build_env_overrides(device: str, device_id: int) -> dict[str, str]:
    if device == "npu":
        inherited_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        if inherited_visible_devices:
            return {"ASCEND_RT_VISIBLE_DEVICES": inherited_visible_devices}
        return {"ASCEND_RT_VISIBLE_DEVICES": str(device_id)}
    return {}


def _should_run_opencompass(
    opencompass_dir: str,
    skip_opencompass: bool,
    enable_opencompass: bool = False,
) -> bool:
    if skip_opencompass:
        print("[INFO] OpenCompass step skipped (--skip-opencompass).")
        return False
    if not enable_opencompass and not opencompass_dir:
        # Default path: OpenCompass disabled. General datasets should show up in
        # the final summary as ``disabled`` rather than as placeholders.
        print("[INFO] OpenCompass disabled by default (--enable-opencompass not set).")
        return False
    if not opencompass_dir:
        print(
            "[WARN] --enable-opencompass was set but --opencompass-dir is empty; "
            "skipping OpenCompass general-capability eval."
        )
        return False
    if not Path(opencompass_dir).expanduser().exists():
        print(f"[WARN] OpenCompass dir not found: {opencompass_dir}; skipping.")
        return False
    return True


def _run_final_merge(
    *,
    pan_summary_path: Path,
    opencompass_work_dir: Path | None,
    output_path: Path,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Merge the PAN summary with the OpenCompass work_dir into final_summary.json.

    When ``opencompass_work_dir`` is None the merge still runs so the resulting
    ``final_summary.json`` explicitly records ``opencompass.enabled = false``
    rather than silently omitting general datasets.
    """

    args = [
        "--pan-summary",
        str(pan_summary_path),
        "--output",
        str(output_path),
    ]
    if opencompass_work_dir is not None:
        args.extend(["--opencompass-work-dir", str(opencompass_work_dir)])
    _run_script(
        "18_merge_opencompass_summary.py",
        args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_merge_lora(
    *,
    eval_config: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    merged_dir: Path,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    _run_script(
        "16_merge_lora_for_opencompass.py",
        [
            "--config",
            str(eval_config),
            "--adapter-manifest",
            str(manifest_path),
            "--adapter-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(merged_dir),
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_opencompass_eval(
    *,
    merged_model_dir: Path,
    work_dir: Path,
    opencompass_dir: str,
    datasets: Sequence[str],
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
    batch_size: int = 32,
) -> None:
    _run_script(
        "17_eval_opencompass.py",
        [
            "--merged-model-dir",
            str(merged_model_dir),
            "--opencompass-dir",
            str(Path(opencompass_dir).expanduser()),
            "--work-dir",
            str(work_dir),
            "--batch-size",
            str(int(batch_size)),
            "--datasets",
            *datasets,
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_opencompass_for_adapter(
    *,
    eval_config: Path,
    training_output_root: Path,
    checkpoint_path: Path,
    opencompass_dir: str,
    datasets: Sequence[str],
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
    merged_dir: Path | None = None,
    work_dir: Path | None = None,
) -> Path:
    if merged_dir is None:
        merged_dir = training_output_root / "merged_hf"
    if work_dir is None:
        work_dir = training_output_root / "opencompass"
    _run_merge_lora(
        eval_config=eval_config,
        manifest_path=training_output_root / "manifest.json",
        checkpoint_path=checkpoint_path,
        merged_dir=merged_dir,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_opencompass_eval(
        merged_model_dir=merged_dir,
        work_dir=work_dir,
        opencompass_dir=opencompass_dir,
        datasets=datasets,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    return work_dir


def _run_opencompass_for_base_model(
    *,
    eval_config_path: Path,
    opencompass_dir: str,
    datasets: Sequence[str],
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> Path:
    eval_cfg = load_eval_config(eval_config_path)
    base_model_dir = Path(eval_cfg.model.path).resolve()
    work_dir = Path(eval_cfg.output.output_root) / "opencompass"
    _run_opencompass_eval(
        merged_model_dir=base_model_dir,
        work_dir=work_dir,
        opencompass_dir=opencompass_dir,
        datasets=datasets,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    return work_dir


def _eval_one_checkpoint(
    *,
    eval_config: Path,
    training_output_root: Path,
    checkpoint_path: Path,
    eval_output_root: Path,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
    safety_eval_datasets: Sequence[str] = (),
) -> None:
    """Run safety eval (12_eval_baseline_suite) + general eval (OpenCompass) for
    a single checkpoint, writing every artifact under ``eval_output_root``."""

    if not dry_run:
        eval_output_root.mkdir(parents=True, exist_ok=True)
    eval_args = [
        "--config",
        str(eval_config),
        "--adapter-manifest",
        str(training_output_root / "manifest.json"),
        "--adapter-checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(eval_output_root),
    ]
    if safety_eval_datasets:
        eval_args.extend(["--safety-eval-datasets", *safety_eval_datasets])
    _run_script(
        "12_eval_baseline_suite.py",
        eval_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    oc_work_dir: Path | None = None
    if _should_run_opencompass(opencompass_dir, skip_opencompass, enable_opencompass):
        oc_work_dir = _run_opencompass_for_adapter(
            eval_config=eval_config,
            training_output_root=training_output_root,
            checkpoint_path=checkpoint_path,
            opencompass_dir=opencompass_dir,
            datasets=opencompass_datasets,
            dry_run=dry_run,
            env_overrides=env_overrides,
            merged_dir=eval_output_root / "merged_hf",
            work_dir=eval_output_root / "opencompass",
        )
    _run_final_merge(
        pan_summary_path=eval_output_root / "summary.json",
        opencompass_work_dir=oc_work_dir,
        output_path=eval_output_root / "final_summary.json",
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _eval_all_epoch_checkpoints(
    *,
    eval_config: Path,
    training_output_root: Path,
    epochs: int,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
    safety_eval_datasets: Sequence[str] = (),
) -> None:
    """Evaluate every per-epoch checkpoint the trainer wrote.

    Each checkpoint's safety + general eval lands in its own
    ``eval_suite/<epoch_NNN>/`` subdirectory so per-epoch progress is
    inspectable instead of only the final epoch.
    """

    checkpoints = _all_epoch_checkpoints(
        training_output_root, epochs=epochs, dry_run=dry_run
    )
    for checkpoint_path in checkpoints:
        eval_output_root = training_output_root / "eval_suite" / checkpoint_path.stem
        _eval_one_checkpoint(
            eval_config=eval_config,
            training_output_root=training_output_root,
            checkpoint_path=checkpoint_path,
            eval_output_root=eval_output_root,
            opencompass_dir=opencompass_dir,
            opencompass_datasets=opencompass_datasets,
            skip_opencompass=skip_opencompass,
            enable_opencompass=enable_opencompass,
            dry_run=dry_run,
            env_overrides=env_overrides,
            safety_eval_datasets=safety_eval_datasets,
        )


def _override_model_runtime(model_payload: dict[str, Any], device: str, device_id: int) -> None:
    model_payload["runtime_backend"] = device
    model_payload["runtime_device"] = _runtime_device_value(device, device_id)


def _make_runtime_override_config(config_path: Path, *, device: str, device_id: int) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    if isinstance(raw.get("model"), dict):
        _override_model_runtime(raw["model"], device, device_id)
    if isinstance(raw.get("teacher"), dict):
        _override_model_runtime(raw["teacher"], device, device_id)
    if isinstance(raw.get("student"), dict):
        _override_model_runtime(raw["student"], device, device_id)
    if isinstance(raw.get("models"), dict):
        for model_payload in raw["models"].values():
            if isinstance(model_payload, dict):
                _override_model_runtime(model_payload, device, device_id)

    override_dir = config_path.parent
    override_path = override_dir / f"{config_path.stem}_launcher_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    override_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return override_path


def _run_phase1_precompute(
    phase1_config: Path,
    *,
    smoke: bool,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
    skip_prepare: bool = False,
    analyze_extras: Sequence[str] | None = None,
    subspace_extras: Sequence[str] | None = None,
    stage_extras: Mapping[str, Sequence[str]] | None = None,
) -> None:
    allowed_stages = {
        "extract", "extract_alignment", "analyze", "subspace", "pairing", "bridge",
        "project", "decompose", "recompose",
    }
    normalized_stage_extras: dict[str, list[str]] = {}
    for stage, tokens in dict(stage_extras or {}).items():
        if stage not in allowed_stages:
            raise ValueError(f"Unsupported Phase-1 stage extra key: {stage}")
        if isinstance(tokens, (str, bytes)) or not all(isinstance(token, str) for token in tokens):
            raise ValueError(f"Phase-1 stage extras for {stage} must be a list of strings")
        normalized_stage_extras[stage] = list(tokens)
    if not skip_prepare:
        _run_script("00_prepare_data.py", ["--config", str(phase1_config)], dry_run=dry_run, env_overrides=env_overrides)

    for split in PIPELINE_SPLITS:
        split_args = [
            "--config",
            str(phase1_config),
            "--split",
            split,
            "--model",
            "teacher",
        ]
        split_args.extend(normalized_stage_extras.get("extract", ()))
        if split == "alignment":
            split_args.extend(normalized_stage_extras.get("extract_alignment", ()))
        _run_script("01_extract_hidden_states.py", split_args, dry_run=dry_run, env_overrides=env_overrides)

    for split in ("alignment", "analysis_val"):
        split_args = [
            "--config",
            str(phase1_config),
            "--split",
            split,
            "--model",
            "student",
        ]
        split_args.extend(normalized_stage_extras.get("extract", ()))
        if split == "alignment":
            split_args.extend(normalized_stage_extras.get("extract_alignment", ()))
        _run_script("01_extract_hidden_states.py", split_args, dry_run=dry_run, env_overrides=env_overrides)

    analyze_args = ["--config", str(phase1_config)]
    subspace_args = ["--config", str(phase1_config)]
    semantic_args_suffix: list[str] = ["--config", str(phase1_config)]
    if analyze_extras:
        analyze_args.extend(str(tok) for tok in analyze_extras)
    if subspace_extras:
        subspace_args.extend(str(tok) for tok in subspace_extras)
    analyze_args.extend(normalized_stage_extras.get("analyze", ()))
    subspace_args.extend(normalized_stage_extras.get("subspace", ()))
    if smoke:
        analyze_args += [
            "--top-k",
            "2",
            "--probe-max-iter",
            "20",
            "--train-max-samples-per-label",
            "16",
            "--val-max-samples-per-label",
            "8",
        ]
        subspace_args += ["--rank", "4"]
        semantic_args_suffix += ["--top-k", "64", "--vocab-chunk-size", "2048"]

    _run_script("02_analyze_teacher_layers.py", analyze_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("03_build_teacher_safe_subspace.py", subspace_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_script(
        "04_pair_layers.py",
        ["--config", str(phase1_config), *normalized_stage_extras.get("pairing", ())],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "05_build_semantic_bases.py",
        ["--config", str(phase1_config), *normalized_stage_extras.get("bridge", ())],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    for split in PIPELINE_SPLITS:
        _run_script(
            "06_project_teacher_safe_component.py",
            ["--config", str(phase1_config), "--split", split, *normalized_stage_extras.get("project", ())],
            dry_run=dry_run,
            env_overrides=env_overrides,
        )
        _run_script(
            "07_decompose_teacher_semantics.py",
            [*semantic_args_suffix, "--split", split, *normalized_stage_extras.get("decompose", ())],
            dry_run=dry_run,
            env_overrides=env_overrides,
        )
        _run_script(
            "08_recompose_student_targets.py",
            ["--config", str(phase1_config), "--split", split, *normalized_stage_extras.get("recompose", ())],
            dry_run=dry_run,
            env_overrides=env_overrides,
        )


def _run_baseline_nosft_one(
    device: str,
    model_size: str,
    *,
    baseline_name: str,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",  # reserved; not forwarded yet
) -> None:
    """Run base-model eval on a single baseline test set.

    ``baseline_name="pan"`` keeps the historical behaviour (PAN eval YAML).
    Any other value routes to the per-baseline eval YAML produced in
    Round 1 and lifts the ``--safety-eval-datasets`` over-refusal probe
    from ``SAFETY_EVAL_DATASETS_BY_BASELINE`` (e.g. CoCoNot for Tülu3).
    """

    _validate_device_request(num_devices)
    if baseline_name == "pan":
        eval_yaml_src = _resolve(BASELINE_EVAL_CONFIGS[(device, model_size)])
    else:
        if (device, model_size, baseline_name) not in SAFETY_EVAL_CONFIGS:
            raise ValueError(
                f"No per-baseline eval config registered for {(device, model_size, baseline_name)}. "
                "Add an entry to SAFETY_EVAL_CONFIGS and a matching "
                f"configs/baseline_eval_qwen35_{model_size}_{baseline_name}_{device}.yaml; "
                "for 9B autogenerate via scripts/gen_9b_baseline_eval_yamls.py."
            )
        eval_yaml_src = _resolve(_safety_eval_config(device, model_size, baseline_name))

    eval_config = _make_runtime_override_config(
        eval_yaml_src,
        device=device,
        device_id=device_id,
    )
    env_overrides = _build_env_overrides(device, device_id)
    safety_eval_datasets = (
        SAFETY_EVAL_DATASETS_BY_BASELINE.get(baseline_name, ())
        if baseline_name != "pan"
        else ()
    )
    eval_args: list[str] = ["--config", str(eval_config)]
    if safety_eval_datasets:
        eval_args.extend(["--safety-eval-datasets", *safety_eval_datasets])
    _run_script(
        "12_eval_baseline_suite.py",
        eval_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    eval_cfg = load_eval_config(eval_config)
    pan_output_root = Path(eval_cfg.output.output_root)
    oc_work_dir: Path | None = None
    if _should_run_opencompass(opencompass_dir, skip_opencompass, enable_opencompass):
        oc_work_dir = _run_opencompass_for_base_model(
            eval_config_path=eval_config,
            opencompass_dir=opencompass_dir,
            datasets=opencompass_datasets,
            dry_run=dry_run,
            env_overrides=env_overrides,
        )
    _run_final_merge(
        pan_summary_path=pan_output_root / "summary.json",
        opencompass_work_dir=oc_work_dir,
        output_path=pan_output_root / "final_summary.json",
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_baseline_nosft(
    device: str,
    model_size: str,
    *,
    baseline_name: str = "pan",
    **kwargs,
) -> None:
    """Outer dispatch: ``--baseline all`` loops over PAN + the three
    safety baselines; otherwise runs one ``_run_baseline_nosft_one``."""

    if baseline_name == "all":
        targets: tuple[str, ...] = ("pan", *SAFETY_SFT_BASELINES)
    else:
        targets = (baseline_name,)
    for target in targets:
        _run_baseline_nosft_one(
            device,
            model_size,
            baseline_name=target,
            **kwargs,
        )


def _invoke_phase1_curation(
    *,
    baseline_name: str,
    processed_dir: Path,
    phase1_yaml: Path,
    dry_run: bool,
    env_overrides: dict[str, str] | None,
) -> None:
    """Run scripts/19b_curate_phase1_subset.py with teacher info from phase1 yaml.

    For ``mode=off`` baselines this is a no-op copy that 19b handles
    internally (no teacher load). For ``minimal`` / ``strict`` baselines 19b
    loads the teacher and runs a forward pass on every prompt — so we pass
    it the same teacher path + runtime that Phase 1 uses.
    """

    teacher_path, teacher_runtime = _read_phase1_teacher(phase1_yaml)
    curate_args = [
        "--baseline",
        baseline_name,
        "--processed-dir",
        str(processed_dir),
        "--mode",
        "auto",
        "--force-rebuild",
    ]
    if teacher_path:
        curate_args.extend(["--teacher-path", str(teacher_path)])
    if teacher_runtime.get("backend"):
        curate_args.extend(["--runtime-backend", teacher_runtime["backend"]])
    if teacher_runtime.get("device"):
        curate_args.extend(["--runtime-device", teacher_runtime["device"]])
    if teacher_runtime.get("attn"):
        curate_args.extend(["--attn-implementation", teacher_runtime["attn"]])
    _run_script(
        "19b_curate_phase1_subset.py",
        curate_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _read_phase1_teacher(phase1_yaml_path: Path) -> tuple[str, dict]:
    """Pull (teacher_path, runtime_dict) out of a generated Phase 1 override.

    runtime_dict is {backend, device, attn} — empty strings when unset. Used
    by ``_run_safety_full`` to forward the teacher model + runtime knobs to
    ``scripts/19b_curate_phase1_subset.py`` so its forward pass runs on the
    same accelerator as Phase 1.
    """

    try:
        raw = yaml.safe_load(Path(phase1_yaml_path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return "", {"backend": "", "device": "", "attn": ""}
    teacher = ((raw.get("models") or {}).get("teacher") or {}) if isinstance(raw, dict) else {}
    path = str(teacher.get("path") or "")
    if path and not Path(path).is_absolute():
        path = str((Path(phase1_yaml_path).resolve().parent / path).resolve())
    backend = str(teacher.get("runtime_backend") or "")
    device = str(teacher.get("runtime_device") or "")
    attn = str(teacher.get("attn_implementation") or "")
    return path, {"backend": backend, "device": device, "attn": attn}


def _make_safety_full_overrides(
    *,
    device: str,
    device_id: int,
    baseline_name: str,
    safety_processed_dir: Path,
    safety_phase1_output_root: Path,
    safety_phasef_output_root: Path,
    phasef_base_override: str = "",
    phase1_base_override: str = "",
    apply_dataset_overrides: bool = True,
) -> tuple[Path, Path]:
    """Generate runtime override yamls for Phase 1 + PhaseF on safety data.

    Phase 1 yaml:
      * dataset.processed_dir → safety_processed_dir
      * extraction.output_root → safety_phase1_output_root

    PhaseF yaml:
      * inputs.train_split / val_split → safety_processed_dir/{alignment,analysis_val}_set.jsonl
      * inputs.train_targets_dir / val_targets_dir → safety_phase1 student_targets dirs
      * inputs.pairing_path → safety_phase1 layer_pairing file
      * output.output_root → safety_phasef_output_root
    """

    base_phase1 = (
        _resolve(phase1_base_override)
        if phase1_base_override
        else _resolve(FULL_PIPELINE_CONFIGS[device]["phase1"])
    )
    base_phasef = (
        _resolve(phasef_base_override)
        if phasef_base_override
        else _resolve(FULL_PIPELINE_CONFIGS[device]["phasef"])
    )

    phase1_raw = yaml.safe_load(base_phase1.read_text(encoding="utf-8"))
    if not isinstance(phase1_raw, dict):
        raise ValueError(f"Phase 1 config must be a mapping: {base_phase1}")
    phase1_raw.setdefault("dataset", {})["processed_dir"] = str(safety_processed_dir)
    phase1_raw.setdefault("extraction", {})["output_root"] = str(safety_phase1_output_root)
    if isinstance(phase1_raw.get("models"), dict):
        for entry in phase1_raw["models"].values():
            if isinstance(entry, dict):
                _override_model_runtime(entry, device, device_id)
    if isinstance(phase1_raw.get("model"), dict):
        _override_model_runtime(phase1_raw["model"], device, device_id)

    phase1_override_path = (
        base_phase1.parent
        / f"{base_phase1.stem}_safety_{baseline_name}_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    )
    phase1_override_path.write_text(
        yaml.safe_dump(phase1_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    phasef_raw = yaml.safe_load(base_phasef.read_text(encoding="utf-8"))
    if not isinstance(phasef_raw, dict):
        raise ValueError(f"PhaseF config must be a mapping: {base_phasef}")
    phasef_inputs = phasef_raw.setdefault("inputs", {})
    # Phase F trains on the full SFT data (train_set.jsonl). The curated
    # contrast subset lives in alignment_set.jsonl and is consumed only by
    # Phase 1 SVD via scripts 01-08. See src/data/curation.py.
    phasef_inputs["train_split"] = str(safety_processed_dir / "train_set.jsonl")
    phasef_inputs["val_split"] = str(safety_processed_dir / "analysis_val_set.jsonl")
    phasef_inputs["train_targets_dir"] = str(
        safety_phase1_output_root / "student_targets" / "student_safe_targets_alignment"
    )
    phasef_inputs["val_targets_dir"] = str(
        safety_phase1_output_root / "student_targets" / "student_safe_targets_val"
    )
    phasef_inputs["pairing_path"] = str(
        safety_phase1_output_root / "layer_pairing" / "teacher_student_layer_pairs.json"
    )
    phasef_inputs["train_anchor_dir"] = str(
        safety_phase1_output_root / "hidden_states" / "student_alignment"
    )
    phasef_inputs["val_anchor_dir"] = str(
        safety_phase1_output_root / "hidden_states" / "student_analysis_val"
    )
    phasef_raw.setdefault("output", {})["output_root"] = str(safety_phasef_output_root)
    _ov = SAFETY_PHASE_OVERRIDES_BY_BASELINE.get(baseline_name, {}) if apply_dataset_overrides else {}
    # Per-baseline PhaseF epoch override (e.g. WildJailbreak -> 5 epochs).
    # Applied to ours + sft1 + random alike so they share the same epoch budget
    # (fair ablation comparison on the harder distribution).
    if _ov.get("phasef_epochs") is not None:
        phasef_raw.setdefault("optim", {})["epochs"] = int(_ov["phasef_epochs"])
    # Per-baseline L_layer weight override (e.g. WildGuardMix -> 0.5). Gated on a
    # non-zero base weight so the sft1 ablation (layer_loss_weight=0.0) stays 0;
    # applies to ours + random (both carry the real L_layer term).
    if _ov.get("phasef_layer_loss_weight") is not None:
        _opt = phasef_raw.setdefault("optim", {})
        if float(_opt.get("layer_loss_weight", 0.0)) != 0.0:
            _opt["layer_loss_weight"] = float(_ov["phasef_layer_loss_weight"])
    if isinstance(phasef_raw.get("model"), dict):
        _override_model_runtime(phasef_raw["model"], device, device_id)

    phasef_override_path = (
        base_phasef.parent
        / f"{base_phasef.stem}_safety_{baseline_name}_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    )
    phasef_override_path.write_text(
        yaml.safe_dump(phasef_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return phase1_override_path, phasef_override_path


def _configured_output_root(config_path: str, section: str) -> Path:
    path = _resolve(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get(section), dict):
        raise ValueError(f"{path} must contain a {section!r} mapping")
    value = raw[section].get("output_root")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must define {section}.output_root")
    return Path(
        resolve_portable_path(value, path.parent, category="output")
    ).resolve()


def _resolve_safety_full_roots(
    *,
    baseline_name: str,
    device: str,
    cell_id: str,
    phase1_config_path: str,
    phasef_config_path: str,
) -> tuple[Path, Path, Path, Path]:
    """Resolve persistent data roots and isolated outputs for a safety cell.

    The ablation backend stages configs with cell-owned output roots. Those
    explicit roots are authoritative; legacy one-click calls without staged
    configs keep their historical per-baseline naming.
    """

    safety_processed_dir = Path(
        resolve_portable_path(
            str(PROJECT_ROOT / "data" / "processed" / f"safety_full_{baseline_name}"),
            PROJECT_ROOT,
            category="data",
        )
    ).resolve()
    pan_processed_dir = Path(
        resolve_portable_path(
            str(PROJECT_ROOT / "data" / "processed"),
            PROJECT_ROOT,
            category="data",
        )
    ).resolve()

    if phase1_config_path:
        phase1_root = _configured_output_root(phase1_config_path, "extraction")
    else:
        cell_suffix = f"_{cell_id}" if cell_id else ""
        legacy = (
            PROJECT_ROOT
            / "outputs"
            / f"safety_full_{baseline_name}_{device}{_pair_suffix()}{cell_suffix}"
            / "phase1"
        )
        phase1_root = Path(
            resolve_portable_path(str(legacy), PROJECT_ROOT, category="output")
        ).resolve()

    phasef_root = (
        _configured_output_root(phasef_config_path, "output")
        if phasef_config_path
        else (phase1_root / "training").resolve()
    )
    return safety_processed_dir, pan_processed_dir, phase1_root, phasef_root


def _run_safety_full(
    device: str,
    *,
    baseline_name: str,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    force_rebuild: bool,
    smoke: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
    phasef_config_path: str = "",
    phase1_config_path: str = "",
    analyze_extras: Sequence[str] | None = None,
    subspace_extras: Sequence[str] | None = None,
    stage_extras: Mapping[str, Sequence[str]] | None = None,
    cell_id: str = "",
    disable_dataset_overrides: bool = False,
) -> None:
    _validate_device_request(num_devices)
    if smoke:
        raise NotImplementedError(
            "smoke variant of safety-full is not wired up; run "
            "`safety-full` with the full Phase 1 budget instead."
        )
    config_key = (device, "0.8b", baseline_name)
    if config_key not in SAFETY_SFT_CONFIGS:
        raise ValueError(
            f"safety-full needs an SFT-style safety config for {config_key}; "
            f"known: {sorted(SAFETY_SFT_CONFIGS.keys())}."
        )
    sft_safety_config = _make_runtime_override_config(
        _resolve(SAFETY_SFT_CONFIGS[config_key]),
        device=device,
        device_id=device_id,
    )
    eval_config_src = _resolve(_safety_eval_config(device, "0.8b", baseline_name))
    eval_config = _make_runtime_override_config(
        eval_config_src,
        device=device,
        device_id=device_id,
    )
    safety_eval_datasets = SAFETY_EVAL_DATASETS_BY_BASELINE.get(baseline_name, ())
    sft_cfg = load_sft_config(sft_safety_config)
    safety_jsonl_path = Path(sft_cfg.data.train_split).resolve()
    env_overrides = _build_env_overrides(device, device_id)

    # 1) Materialize the safety SFT JSONL via 19.
    prep_args = ["--config", str(sft_safety_config)]
    if force_rebuild:
        prep_args.append("--force-rebuild")
    _run_script(
        "19_prepare_safety_data.py",
        prep_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    # 2) Split into PAN-style 5 JSONLs under a per-baseline processed_dir.
    (
        safety_processed_dir,
        pan_processed_dir,
        safety_phase1_output_root,
        safety_phasef_output_root,
    ) = _resolve_safety_full_roots(
        baseline_name=baseline_name,
        device=device,
        cell_id=cell_id,
        phase1_config_path=phase1_config_path,
        phasef_config_path=phasef_config_path,
    )
    if not dry_run:
        safety_processed_dir.mkdir(parents=True, exist_ok=True)
    _run_script(
        "20_split_safety_for_semalign.py",
        [
            "--safety-jsonl",
            str(safety_jsonl_path),
            "--output-dir",
            str(safety_processed_dir),
            "--pan-processed-dir",
            str(pan_processed_dir),
            "--harmless-source",
            "auto",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    # 3) Generate Phase 1 + PhaseF override yamls pointing at the persistent
    # safety split and the cell-owned output roots resolved above.
    phase1_override, phasef_override = _make_safety_full_overrides(
        device=device,
        device_id=device_id,
        baseline_name=baseline_name,
        safety_processed_dir=safety_processed_dir,
        safety_phase1_output_root=safety_phase1_output_root,
        safety_phasef_output_root=safety_phasef_output_root,
        phasef_base_override=phasef_config_path,
        phase1_base_override=phase1_config_path,
        apply_dataset_overrides=not disable_dataset_overrides,
    )

    # 3b) Curate the Phase 1 contrast subset (alignment_set.jsonl) from
    # train_set.jsonl. For clean baselines (PAN/STL/coconot/HH-RLHF) this is
    # a byte-identical copy via mode=off. For WJB/WGM it applies a strict
    # per-baseline pre-filter plus a universal teacher-confidence filter.
    _invoke_phase1_curation(
        baseline_name=baseline_name,
        processed_dir=safety_processed_dir,
        phase1_yaml=phase1_override,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    # 4) Run Phase 1-E (skip 00 — safety splits are already on disk).
    # Per-baseline cleaner-subspace knobs (e.g. WJB: --top-k 3 / --energy-threshold
    # 0.7 / --rank-cap 8) go first; caller-supplied extras append after so an
    # explicit CLI --phase1-analyze-extra / --phase1-subspace-extra still wins.
    phase_overrides = (
        {} if disable_dataset_overrides else SAFETY_PHASE_OVERRIDES_BY_BASELINE.get(baseline_name, {})
    )
    merged_analyze_extras = [
        *phase_overrides.get("analyze_extra", ()),
        *(analyze_extras or ()),
    ]
    merged_subspace_extras = [
        *phase_overrides.get("subspace_extra", ()),
        *(subspace_extras or ()),
    ]
    _run_phase1_precompute(
        phase1_override,
        smoke=False,
        dry_run=dry_run,
        env_overrides=env_overrides,
        skip_prepare=True,
        analyze_extras=merged_analyze_extras,
        subspace_extras=merged_subspace_extras,
        stage_extras=stage_extras,
    )

    # 5) PhaseF training.
    _run_script(
        "09_train_student_semalign.py",
        ["--config", str(phasef_override)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    # 6) Sanity eval + tables (still PAN-comparable; uses safety processed_dir).
    _run_script(
        "10_sanity_eval.py",
        ["--config", str(phase1_override)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        ["--config", str(phase1_override)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    # 7) PAN safety eval + OpenCompass general eval against every epoch checkpoint.
    phasef_cfg = load_phasef_config(phasef_override)
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=safety_phasef_output_root,
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        safety_eval_datasets=safety_eval_datasets,
        eval_config_path=str(eval_config_src),
    )


def _run_safety_distill(
    device: str,
    *,
    baseline_name: str,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    force_rebuild: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    _validate_device_request(num_devices)
    config_key = (device, baseline_name)
    if config_key not in SAFETY_DISTILL_CONFIGS:
        raise ValueError(
            f"No safety distill config registered for {config_key}. "
            f"Known combinations: {sorted(SAFETY_DISTILL_CONFIGS.keys())}."
        )
    train_config = _make_runtime_override_config(
        _resolve(SAFETY_DISTILL_CONFIGS[config_key]),
        device=device,
        device_id=device_id,
    )
    eval_config_src = _resolve(_safety_eval_config(device, "0.8b", baseline_name))
    eval_config = _make_runtime_override_config(
        eval_config_src,
        device=device,
        device_id=device_id,
    )
    safety_eval_datasets = SAFETY_EVAL_DATASETS_BY_BASELINE.get(baseline_name, ())
    cfg = load_distill_config(train_config)
    env_overrides = _build_env_overrides(device, device_id)

    prep_args = ["--config", str(train_config)]
    if force_rebuild:
        prep_args.append("--force-rebuild")
    _run_script(
        "19_prepare_safety_data.py",
        prep_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "14_train_pan_distill.py",
        ["--config", str(train_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _eval_all_epoch_checkpoints(
        eval_config=eval_config,
        training_output_root=Path(cfg.output.output_root),
        epochs=int(cfg.optim.epochs),
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        dry_run=dry_run,
        env_overrides=env_overrides,
        safety_eval_datasets=safety_eval_datasets,
    )


def _run_safety_sft(
    device: str,
    *,
    baseline_name: str,
    model_size: str,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    force_rebuild: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    _validate_device_request(num_devices)
    config_key = (device, model_size, baseline_name)
    if config_key not in SAFETY_SFT_CONFIGS:
        raise ValueError(
            f"No safety SFT config registered for {config_key}. "
            f"Known combinations: {sorted(SAFETY_SFT_CONFIGS.keys())}."
        )
    train_config = _make_runtime_override_config(
        _resolve(SAFETY_SFT_CONFIGS[config_key]),
        device=device,
        device_id=device_id,
    )
    eval_config_src = _resolve(_safety_eval_config(device, model_size, baseline_name))
    eval_config = _make_runtime_override_config(
        eval_config_src,
        device=device,
        device_id=device_id,
    )
    safety_eval_datasets = SAFETY_EVAL_DATASETS_BY_BASELINE.get(baseline_name, ())
    cfg = load_sft_config(train_config)
    env_overrides = _build_env_overrides(device, device_id)

    prep_args = ["--config", str(train_config)]
    if force_rebuild:
        prep_args.append("--force-rebuild")
    _run_script(
        "19_prepare_safety_data.py",
        prep_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "13_train_pan_sft.py",
        ["--config", str(train_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _eval_all_epoch_checkpoints(
        eval_config=eval_config,
        training_output_root=Path(cfg.output.output_root),
        epochs=int(cfg.optim.epochs),
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        dry_run=dry_run,
        env_overrides=env_overrides,
        safety_eval_datasets=safety_eval_datasets,
    )


def _run_baseline_sft(
    device: str,
    model_size: str,
    *,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    _validate_device_request(num_devices)
    train_config = _make_runtime_override_config(
        _resolve(BASELINE_SFT_CONFIGS[(device, model_size)]),
        device=device,
        device_id=device_id,
    )
    eval_config = _make_runtime_override_config(
        _resolve(BASELINE_EVAL_CONFIGS[(device, model_size)]),
        device=device,
        device_id=device_id,
    )
    cfg = load_sft_config(train_config)
    env_overrides = _build_env_overrides(device, device_id)

    _run_script(
        "13_train_pan_sft.py",
        ["--config", str(train_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _eval_all_epoch_checkpoints(
        eval_config=eval_config,
        training_output_root=Path(cfg.output.output_root),
        epochs=int(cfg.optim.epochs),
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_baseline_distill(
    device: str,
    *,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    _validate_device_request(num_devices)
    train_config = _make_runtime_override_config(
        _resolve(BASELINE_DISTILL_CONFIGS[device]),
        device=device,
        device_id=device_id,
    )
    eval_config = _make_runtime_override_config(
        _resolve(BASELINE_EVAL_CONFIGS[(device, "0.8b")]),
        device=device,
        device_id=device_id,
    )
    cfg = load_distill_config(train_config)
    env_overrides = _build_env_overrides(device, device_id)

    _run_script(
        "14_train_pan_distill.py",
        ["--config", str(train_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _eval_all_epoch_checkpoints(
        eval_config=eval_config,
        training_output_root=Path(cfg.output.output_root),
        epochs=int(cfg.optim.epochs),
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_adapter_eval(
    *,
    device: str,
    model_size: str,
    training_output_root: Path,
    epochs: int,
    device_id: int,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
    opencompass_dir: str = "",
    opencompass_datasets: Sequence[str] = (),
    skip_opencompass: bool = True,
    enable_opencompass: bool = False,
    safety_eval_datasets: Sequence[str] = (),
    eval_config_path: str | None = None,
) -> None:
    eval_config_src = (
        Path(eval_config_path).resolve()
        if eval_config_path
        else _resolve(BASELINE_EVAL_CONFIGS[(device, model_size)])
    )
    eval_config = _make_runtime_override_config(
        eval_config_src,
        device=device,
        device_id=device_id,
    )
    _eval_all_epoch_checkpoints(
        eval_config=eval_config,
        training_output_root=training_output_root,
        epochs=epochs,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        dry_run=dry_run,
        env_overrides=env_overrides,
        safety_eval_datasets=safety_eval_datasets,
    )


def _run_random_baseline(
    device: str,
    *,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    _validate_device_request(num_devices)
    phase1_config = _make_runtime_override_config(
        _resolve(RANDOM_PIPELINE_CONFIGS[device]["phase1"]),
        device=device,
        device_id=device_id,
    )
    phasef_config = _make_runtime_override_config(
        _resolve(RANDOM_PIPELINE_CONFIGS[device]["phasef"]),
        device=device,
        device_id=device_id,
    )
    phasef_cfg = load_phasef_config(phasef_config)

    env_overrides = _build_env_overrides(device, device_id)
    _run_phase1_precompute(phase1_config, smoke=False, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("09_train_student_semalign.py", ["--config", str(phasef_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script(
        "10_sanity_eval.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir",
            str(Path(phasef_cfg.output.output_root)),
            "--output-dir-name",
            "sanity_eval_random_same_norm",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir-name",
            Path(phasef_cfg.output.output_root).name,
            "--sanity-dir-name",
            "sanity_eval_random_same_norm",
            "--tables-dir-name",
            "tables_random_same_norm",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=Path(phasef_cfg.output.output_root),
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
    )


def _run_pan_sft1_ablation(
    device: str,
    *,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    """sft1 ablation on the PAN corpus.

    Same Phase 1 precompute (00->08) as the main ours run; the scripts
    detect existing shards via ``extraction.skip_existing`` so this is a
    no-op when Phase 1 is already on disk. Only the PhaseF training stage
    swaps in the sft1 yaml whose output_root ends in ``/training_sft1``,
    keeping main / random / sft1 results isolated.
    """

    _validate_device_request(num_devices)
    phase1_config = _make_runtime_override_config(
        _resolve(SFT1_PIPELINE_CONFIGS[device]["phase1"]),
        device=device,
        device_id=device_id,
    )
    phasef_config = _make_runtime_override_config(
        _resolve(SFT1_PIPELINE_CONFIGS[device]["phasef"]),
        device=device,
        device_id=device_id,
    )
    phasef_cfg = load_phasef_config(phasef_config)

    env_overrides = _build_env_overrides(device, device_id)
    _run_phase1_precompute(phase1_config, smoke=False, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("09_train_student_semalign.py", ["--config", str(phasef_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script(
        "10_sanity_eval.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir",
            str(Path(phasef_cfg.output.output_root)),
            "--output-dir-name",
            "sanity_eval_sft1",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir-name",
            Path(phasef_cfg.output.output_root).name,
            "--sanity-dir-name",
            "sanity_eval_sft1",
            "--tables-dir-name",
            "tables_sft1",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=Path(phasef_cfg.output.output_root),
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
    )


def _run_pan_bothpole(
    device: str,
    *,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    """Both-pole ours variant on the PAN corpus.

    Same Phase 1 precompute (00->08) as the main ours run -- the scripts skip
    existing shards, so this is a no-op when Phase 1 is already on disk and the
    same subspace / student targets are reused. Only the PhaseF training stage
    swaps in the both-pole yaml (layer_loss_policy=label_weighted) whose
    output_root ends in ``/training_bothpole``, keeping the main harmful_only
    ours result intact and allowing parallel execution on a separate device.
    """

    _validate_device_request(num_devices)
    if device not in BOTHPOLE_PIPELINE_CONFIGS:
        raise ValueError(
            f"bothpole is only wired for {sorted(BOTHPOLE_PIPELINE_CONFIGS)}; got device={device!r}."
        )
    phase1_config = _make_runtime_override_config(
        _resolve(BOTHPOLE_PIPELINE_CONFIGS[device]["phase1"]),
        device=device,
        device_id=device_id,
    )
    phasef_config = _make_runtime_override_config(
        _resolve(BOTHPOLE_PIPELINE_CONFIGS[device]["phasef"]),
        device=device,
        device_id=device_id,
    )
    phasef_cfg = load_phasef_config(phasef_config)

    env_overrides = _build_env_overrides(device, device_id)
    _run_phase1_precompute(phase1_config, smoke=False, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("09_train_student_semalign.py", ["--config", str(phasef_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script(
        "10_sanity_eval.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir",
            str(Path(phasef_cfg.output.output_root)),
            "--output-dir-name",
            "sanity_eval_bothpole",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir-name",
            Path(phasef_cfg.output.output_root).name,
            "--sanity-dir-name",
            "sanity_eval_bothpole",
            "--tables-dir-name",
            "tables_bothpole",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=Path(phasef_cfg.output.output_root),
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
    )


def _make_safety_sft1_overrides(
    *,
    device: str,
    device_id: int,
    baseline_name: str,
    safety_processed_dir: Path,
    safety_phase1_output_root: Path,
    safety_phasef_output_root: Path,
) -> tuple[Path, Path]:
    """Mirror of ``_make_safety_full_overrides`` but with the sft1 phaseF
    yaml as the base (same Phase 1 yaml, since the precompute is shared).
    Override file names use the ``_sft1_`` infix so they do not collide
    with the main-ours safety overrides on disk."""

    base_phase1 = _resolve(FULL_PIPELINE_CONFIGS[device]["phase1"])
    base_phasef = _resolve(SFT1_PIPELINE_CONFIGS[device]["phasef"])

    phase1_raw = yaml.safe_load(base_phase1.read_text(encoding="utf-8"))
    if not isinstance(phase1_raw, dict):
        raise ValueError(f"Phase 1 config must be a mapping: {base_phase1}")
    phase1_raw.setdefault("dataset", {})["processed_dir"] = str(safety_processed_dir)
    phase1_raw.setdefault("extraction", {})["output_root"] = str(safety_phase1_output_root)
    if isinstance(phase1_raw.get("models"), dict):
        for entry in phase1_raw["models"].values():
            if isinstance(entry, dict):
                _override_model_runtime(entry, device, device_id)
    if isinstance(phase1_raw.get("model"), dict):
        _override_model_runtime(phase1_raw["model"], device, device_id)

    phase1_override_path = (
        base_phase1.parent
        / f"{base_phase1.stem}_sft1_{baseline_name}_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    )
    phase1_override_path.write_text(
        yaml.safe_dump(phase1_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    phasef_raw = yaml.safe_load(base_phasef.read_text(encoding="utf-8"))
    if not isinstance(phasef_raw, dict):
        raise ValueError(f"PhaseF config must be a mapping: {base_phasef}")
    phasef_inputs = phasef_raw.setdefault("inputs", {})
    # Phase F trains on the full SFT data (train_set.jsonl). The curated
    # contrast subset lives in alignment_set.jsonl and is consumed only by
    # Phase 1 SVD via scripts 01-08. See src/data/curation.py.
    phasef_inputs["train_split"] = str(safety_processed_dir / "train_set.jsonl")
    phasef_inputs["val_split"] = str(safety_processed_dir / "analysis_val_set.jsonl")
    phasef_inputs["train_targets_dir"] = str(
        safety_phase1_output_root / "student_targets" / "student_safe_targets_alignment"
    )
    phasef_inputs["val_targets_dir"] = str(
        safety_phase1_output_root / "student_targets" / "student_safe_targets_val"
    )
    phasef_inputs["pairing_path"] = str(
        safety_phase1_output_root / "layer_pairing" / "teacher_student_layer_pairs.json"
    )
    phasef_inputs["train_anchor_dir"] = str(
        safety_phase1_output_root / "hidden_states" / "student_alignment"
    )
    phasef_inputs["val_anchor_dir"] = str(
        safety_phase1_output_root / "hidden_states" / "student_analysis_val"
    )
    phasef_raw.setdefault("output", {})["output_root"] = str(safety_phasef_output_root)
    _ov = SAFETY_PHASE_OVERRIDES_BY_BASELINE.get(baseline_name, {})
    # Per-baseline PhaseF epoch override (e.g. WildJailbreak -> 5 epochs).
    # Applied to ours + sft1 + random alike so they share the same epoch budget
    # (fair ablation comparison on the harder distribution).
    if _ov.get("phasef_epochs") is not None:
        phasef_raw.setdefault("optim", {})["epochs"] = int(_ov["phasef_epochs"])
    # Per-baseline L_layer weight override (e.g. WildGuardMix -> 0.5). Gated on a
    # non-zero base weight so the sft1 ablation (layer_loss_weight=0.0) stays 0;
    # applies to ours + random (both carry the real L_layer term).
    if _ov.get("phasef_layer_loss_weight") is not None:
        _opt = phasef_raw.setdefault("optim", {})
        if float(_opt.get("layer_loss_weight", 0.0)) != 0.0:
            _opt["layer_loss_weight"] = float(_ov["phasef_layer_loss_weight"])
    if isinstance(phasef_raw.get("model"), dict):
        _override_model_runtime(phasef_raw["model"], device, device_id)

    phasef_override_path = (
        base_phasef.parent
        / f"{base_phasef.stem}_sft1_{baseline_name}_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    )
    phasef_override_path.write_text(
        yaml.safe_dump(phasef_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return phase1_override_path, phasef_override_path


def _make_safety_random_overrides(
    *,
    device: str,
    device_id: int,
    baseline_name: str,
    safety_processed_dir: Path,
    safety_phase1_output_root: Path,
    safety_phasef_output_root: Path,
) -> tuple[Path, Path]:
    base_phase1 = _resolve(FULL_PIPELINE_CONFIGS[device]["phase1"])
    base_phasef = _resolve(RANDOM_PIPELINE_CONFIGS[device]["phasef"])

    phase1_raw = yaml.safe_load(base_phase1.read_text(encoding="utf-8"))
    if not isinstance(phase1_raw, dict):
        raise ValueError(f"Phase 1 config must be a mapping: {base_phase1}")
    phase1_raw.setdefault("dataset", {})["processed_dir"] = str(safety_processed_dir)
    phase1_raw.setdefault("extraction", {})["output_root"] = str(safety_phase1_output_root)
    if isinstance(phase1_raw.get("models"), dict):
        for entry in phase1_raw["models"].values():
            if isinstance(entry, dict):
                _override_model_runtime(entry, device, device_id)
    if isinstance(phase1_raw.get("model"), dict):
        _override_model_runtime(phase1_raw["model"], device, device_id)

    phase1_override_path = (
        base_phase1.parent
        / f"{base_phase1.stem}_random_{baseline_name}_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    )
    phase1_override_path.write_text(
        yaml.safe_dump(phase1_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    phasef_raw = yaml.safe_load(base_phasef.read_text(encoding="utf-8"))
    if not isinstance(phasef_raw, dict):
        raise ValueError(f"PhaseF config must be a mapping: {base_phasef}")
    phasef_inputs = phasef_raw.setdefault("inputs", {})
    # Phase F trains on the full SFT data (train_set.jsonl). The curated
    # contrast subset lives in alignment_set.jsonl and is consumed only by
    # Phase 1 SVD via scripts 01-08. See src/data/curation.py.
    phasef_inputs["train_split"] = str(safety_processed_dir / "train_set.jsonl")
    phasef_inputs["val_split"] = str(safety_processed_dir / "analysis_val_set.jsonl")
    phasef_inputs["train_targets_dir"] = str(
        safety_phase1_output_root / "student_targets" / "student_safe_targets_alignment"
    )
    phasef_inputs["val_targets_dir"] = str(
        safety_phase1_output_root / "student_targets" / "student_safe_targets_val"
    )
    phasef_inputs["pairing_path"] = str(
        safety_phase1_output_root / "layer_pairing" / "teacher_student_layer_pairs.json"
    )
    phasef_inputs["train_anchor_dir"] = str(
        safety_phase1_output_root / "hidden_states" / "student_alignment"
    )
    phasef_inputs["val_anchor_dir"] = str(
        safety_phase1_output_root / "hidden_states" / "student_analysis_val"
    )
    phasef_raw.setdefault("output", {})["output_root"] = str(safety_phasef_output_root)
    _ov = SAFETY_PHASE_OVERRIDES_BY_BASELINE.get(baseline_name, {})
    # Per-baseline PhaseF epoch override (e.g. WildJailbreak -> 5 epochs).
    # Applied to ours + sft1 + random alike so they share the same epoch budget
    # (fair ablation comparison on the harder distribution).
    if _ov.get("phasef_epochs") is not None:
        phasef_raw.setdefault("optim", {})["epochs"] = int(_ov["phasef_epochs"])
    # Per-baseline L_layer weight override (e.g. WildGuardMix -> 0.5). Gated on a
    # non-zero base weight so the sft1 ablation (layer_loss_weight=0.0) stays 0;
    # applies to ours + random (both carry the real L_layer term).
    if _ov.get("phasef_layer_loss_weight") is not None:
        _opt = phasef_raw.setdefault("optim", {})
        if float(_opt.get("layer_loss_weight", 0.0)) != 0.0:
            _opt["layer_loss_weight"] = float(_ov["phasef_layer_loss_weight"])
    if isinstance(phasef_raw.get("model"), dict):
        _override_model_runtime(phasef_raw["model"], device, device_id)

    phasef_override_path = (
        base_phasef.parent
        / f"{base_phasef.stem}_random_{baseline_name}_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    )
    phasef_override_path.write_text(
        yaml.safe_dump(phasef_raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return phase1_override_path, phasef_override_path


def _run_safety_random(
    device: str,
    *,
    baseline_name: str,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    force_rebuild: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    _validate_device_request(num_devices)
    config_key = (device, "0.8b", baseline_name)
    if config_key not in SAFETY_SFT_CONFIGS:
        raise ValueError(
            f"safety-random needs an SFT-style safety config for {config_key}; "
            f"known: {sorted(SAFETY_SFT_CONFIGS.keys())}."
        )
    sft_safety_config = _make_runtime_override_config(
        _resolve(SAFETY_SFT_CONFIGS[config_key]),
        device=device,
        device_id=device_id,
    )
    eval_config_src = _resolve(_safety_eval_config(device, "0.8b", baseline_name))
    safety_eval_datasets = SAFETY_EVAL_DATASETS_BY_BASELINE.get(baseline_name, ())
    sft_cfg = load_sft_config(sft_safety_config)
    safety_jsonl_path = Path(sft_cfg.data.train_split).resolve()
    env_overrides = _build_env_overrides(device, device_id)

    prep_args = ["--config", str(sft_safety_config)]
    if force_rebuild:
        prep_args.append("--force-rebuild")
    _run_script(
        "19_prepare_safety_data.py",
        prep_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    safety_processed_dir = (
        PROJECT_ROOT / "data" / "processed" / f"safety_full_{baseline_name}"
    ).resolve()
    pan_processed_dir = (PROJECT_ROOT / "data" / "processed").resolve()
    if not dry_run:
        safety_processed_dir.mkdir(parents=True, exist_ok=True)
    _run_script(
        "20_split_safety_for_semalign.py",
        [
            "--safety-jsonl",
            str(safety_jsonl_path),
            "--output-dir",
            str(safety_processed_dir),
            "--pan-processed-dir",
            str(pan_processed_dir),
            "--harmless-source",
            "auto",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    safety_phase1_output_root = (
        PROJECT_ROOT
        / "outputs"
        / f"safety_full_{baseline_name}_{device}"
        / "phase1"
    ).resolve()
    safety_phasef_output_root = (
        safety_phase1_output_root / "training_random_same_norm"
    ).resolve()
    phase1_override, phasef_override = _make_safety_random_overrides(
        device=device,
        device_id=device_id,
        baseline_name=baseline_name,
        safety_processed_dir=safety_processed_dir,
        safety_phase1_output_root=safety_phase1_output_root,
        safety_phasef_output_root=safety_phasef_output_root,
    )

    _invoke_phase1_curation(
        baseline_name=baseline_name,
        processed_dir=safety_processed_dir,
        phase1_yaml=phase1_override,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    _run_phase1_precompute(
        phase1_override,
        smoke=False,
        dry_run=dry_run,
        env_overrides=env_overrides,
        skip_prepare=True,
    )
    _run_script(
        "09_train_student_semalign.py",
        ["--config", str(phasef_override)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "10_sanity_eval.py",
        [
            "--config",
            str(phase1_override),
            "--training-dir",
            str(safety_phasef_output_root),
            "--output-dir-name",
            "sanity_eval_random_same_norm",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        [
            "--config",
            str(phase1_override),
            "--training-dir-name",
            "training_random_same_norm",
            "--sanity-dir-name",
            "sanity_eval_random_same_norm",
            "--tables-dir-name",
            "tables_random_same_norm",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    phasef_cfg = load_phasef_config(phasef_override)
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=safety_phasef_output_root,
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        safety_eval_datasets=safety_eval_datasets,
        eval_config_path=str(eval_config_src),
    )


def _run_safety_sft1(
    device: str,
    *,
    baseline_name: str,
    device_id: int,
    num_devices: int,
    dry_run: bool,
    force_rebuild: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
) -> None:
    """sft1 ablation on a safety corpus (BT / STL / Tülu3).

    Shares the full safety-full pipeline (19_prepare + 20_split + Phase 1
    precompute against the per-baseline safety processed_dir) so the
    phase1 artefacts under ``safety_full_<bl>_<device>/phase1/`` are
    reused by both the main safety-full run and this ablation. Only the
    PhaseF stage swaps in the sft1 yaml, with output_root pointing at
    ``safety_full_<bl>_<device>/phase1/training_sft1`` to keep the main
    safety-full training_dir intact.
    """

    _validate_device_request(num_devices)
    config_key = (device, "0.8b", baseline_name)
    if config_key not in SAFETY_SFT_CONFIGS:
        raise ValueError(
            f"safety-sft1 needs an SFT-style safety config for {config_key}; "
            f"known: {sorted(SAFETY_SFT_CONFIGS.keys())}."
        )
    sft_safety_config = _make_runtime_override_config(
        _resolve(SAFETY_SFT_CONFIGS[config_key]),
        device=device,
        device_id=device_id,
    )
    eval_config_src = _resolve(_safety_eval_config(device, "0.8b", baseline_name))
    eval_config = _make_runtime_override_config(
        eval_config_src,
        device=device,
        device_id=device_id,
    )
    safety_eval_datasets = SAFETY_EVAL_DATASETS_BY_BASELINE.get(baseline_name, ())
    sft_cfg = load_sft_config(sft_safety_config)
    safety_jsonl_path = Path(sft_cfg.data.train_split).resolve()
    env_overrides = _build_env_overrides(device, device_id)

    prep_args = ["--config", str(sft_safety_config)]
    if force_rebuild:
        prep_args.append("--force-rebuild")
    _run_script(
        "19_prepare_safety_data.py",
        prep_args,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    safety_processed_dir = (
        PROJECT_ROOT / "data" / "processed" / f"safety_full_{baseline_name}"
    ).resolve()
    pan_processed_dir = (PROJECT_ROOT / "data" / "processed").resolve()
    if not dry_run:
        safety_processed_dir.mkdir(parents=True, exist_ok=True)
    _run_script(
        "20_split_safety_for_semalign.py",
        [
            "--safety-jsonl",
            str(safety_jsonl_path),
            "--output-dir",
            str(safety_processed_dir),
            "--pan-processed-dir",
            str(pan_processed_dir),
            "--harmless-source",
            "auto",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    safety_phase1_output_root = (
        PROJECT_ROOT
        / "outputs"
        / f"safety_full_{baseline_name}_{device}{_pair_suffix()}"
        / "phase1"
    ).resolve()
    # sft1 lands beside the main "training/" so the two ablations share
    # the same phase1 artefacts without overwriting each other's
    # checkpoints, eval_suite, sanity_eval or tables.
    safety_phasef_output_root = (safety_phase1_output_root / "training_sft1").resolve()
    phase1_override, phasef_override = _make_safety_sft1_overrides(
        device=device,
        device_id=device_id,
        baseline_name=baseline_name,
        safety_processed_dir=safety_processed_dir,
        safety_phase1_output_root=safety_phase1_output_root,
        safety_phasef_output_root=safety_phasef_output_root,
    )

    _invoke_phase1_curation(
        baseline_name=baseline_name,
        processed_dir=safety_processed_dir,
        phase1_yaml=phase1_override,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    _run_phase1_precompute(
        phase1_override,
        smoke=False,
        dry_run=dry_run,
        env_overrides=env_overrides,
        skip_prepare=True,
    )

    _run_script(
        "09_train_student_semalign.py",
        ["--config", str(phasef_override)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    _run_script(
        "10_sanity_eval.py",
        [
            "--config",
            str(phase1_override),
            "--training-dir",
            str(safety_phasef_output_root),
            "--output-dir-name",
            "sanity_eval_sft1",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        [
            "--config",
            str(phase1_override),
            "--training-dir-name",
            "training_sft1",
            "--sanity-dir-name",
            "sanity_eval_sft1",
            "--tables-dir-name",
            "tables_sft1",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    phasef_cfg = load_phasef_config(phasef_override)
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=safety_phasef_output_root,
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
        safety_eval_datasets=safety_eval_datasets,
        eval_config_path=str(eval_config_src),
    )


def _run_full_pipeline(
    device: str,
    *,
    device_id: int,
    num_devices: int,
    smoke: bool,
    dry_run: bool,
    opencompass_dir: str,
    opencompass_datasets: Sequence[str],
    skip_opencompass: bool,
    enable_opencompass: bool,
    opencompass_config: str = "",
    phasef_config_path: str = "",
    phase1_config_path: str = "",
    analyze_extras: Sequence[str] | None = None,
    subspace_extras: Sequence[str] | None = None,
    stage_extras: Mapping[str, Sequence[str]] | None = None,
) -> None:
    _validate_device_request(num_devices)
    config_map = FULL_PIPELINE_CONFIGS
    phase1_base = (
        _resolve(phase1_config_path)
        if phase1_config_path
        else _resolve(config_map[device]["phase1"])
    )
    phase1_config = _make_runtime_override_config(
        phase1_base,
        device=device,
        device_id=device_id,
    )
    phasef_config_src = (
        _resolve(phasef_config_path)
        if phasef_config_path
        else _resolve(config_map[device]["phasef"])
    )
    phasef_config = _make_runtime_override_config(
        phasef_config_src,
        device=device,
        device_id=device_id,
    )

    phase1_cfg = load_phase1_config(phase1_config)
    phasef_cfg = load_phasef_config(phasef_config)

    env_overrides = _build_env_overrides(device, device_id)
    _run_phase1_precompute(
        phase1_config,
        smoke=smoke,
        dry_run=dry_run,
        env_overrides=env_overrides,
        analyze_extras=analyze_extras,
        subspace_extras=subspace_extras,
        stage_extras=stage_extras,
    )

    training_output_root = Path(phasef_cfg.output.output_root)
    phase1_output_root = Path(phase1_cfg.extraction.output_root)
    default_training_output_root = phase1_output_root / "training"
    uses_default_training_dir = (
        training_output_root.resolve() == default_training_output_root.resolve()
    )

    sanity_args = ["--config", str(phase1_config)]
    tables_args = ["--config", str(phase1_config)]
    if not uses_default_training_dir:
        try:
            training_dir_name = str(training_output_root.resolve().relative_to(phase1_output_root.resolve()))
        except ValueError as exc:
            raise ValueError(
                "Custom PhaseF output.output_root must live under the Phase1 "
                "extraction.output_root so 10_sanity_eval.py and 11_make_tables.py "
                "can keep experiment artifacts isolated by directory name. "
                f"Got PhaseF output_root={training_output_root} and "
                f"Phase1 output_root={phase1_output_root}."
            ) from exc
        suffix = training_output_root.name
        sanity_dir_name = f"sanity_eval_{suffix}"
        tables_dir_name = f"tables_{suffix}"
        sanity_args += [
            "--training-dir",
            str(training_output_root),
            "--output-dir-name",
            sanity_dir_name,
        ]
        tables_args += [
            "--training-dir-name",
            training_dir_name,
            "--sanity-dir-name",
            sanity_dir_name,
            "--tables-dir-name",
            tables_dir_name,
        ]
    if smoke:
        sanity_args += ["--max-samples-per-label", "8", "--max-new-tokens", "32"]

    _run_script("09_train_student_semalign.py", ["--config", str(phasef_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script("10_sanity_eval.py", sanity_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("11_make_tables.py", tables_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_adapter_eval(
        device=device,
        model_size="0.8b",
        training_output_root=training_output_root,
        epochs=int(phasef_cfg.optim.epochs),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
        opencompass_dir=opencompass_dir,
        opencompass_datasets=opencompass_datasets,
        skip_opencompass=skip_opencompass,
        enable_opencompass=enable_opencompass,
    )

    if not dry_run:
        summary = {
            "device": device,
            "device_id": device_id,
            "num_devices": num_devices,
            "smoke": smoke,
            "phase1_output_root": phase1_cfg.extraction.output_root,
            "phasef_output_root": phasef_cfg.output.output_root,
            "phasef_config": str(phasef_config_src),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    global _ACTIVE_PAIR
    _ACTIVE_PAIR = getattr(args, "pair", DEFAULT_PAIR)
    # Resolve the teacher/student slot ("0.8b"=student, "9b"=teacher — legacy keys
    # into the canonical config dicts) from --role, with --model as a legacy
    # override. apply_tokens() then maps the canonical config name to the pair's
    # actual model. Only nosft/sft expose --role; other commands fix their slot.
    if getattr(args, "model", None):
        _role_slot = args.model
    elif hasattr(args, "role"):
        _role_slot = "9b" if args.role == "teacher" else "0.8b"
    else:
        _role_slot = "0.8b"
    oc_kwargs = {
        "opencompass_dir": args.opencompass_dir,
        "opencompass_datasets": args.opencompass_datasets,
        "skip_opencompass": args.skip_opencompass,
        "enable_opencompass": bool(getattr(args, "enable_opencompass", False)),
        "opencompass_config": getattr(args, "opencompass_config", "") or "",
    }
    if args.command == "nosft":
        _run_baseline_nosft(
            args.device,
            _role_slot,
            baseline_name=args.baseline,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            **oc_kwargs,
        )
        return
    if args.command == "sft":
        if args.baseline != "pan":
            _run_safety_sft(
                args.device,
                baseline_name=args.baseline,
                model_size=_role_slot,
                device_id=args.device_id,
                num_devices=args.num_devices,
                dry_run=args.dry_run,
                force_rebuild=bool(args.force_rebuild),
                **oc_kwargs,
            )
            return
        _run_baseline_sft(
            args.device,
            _role_slot,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            **oc_kwargs,
        )
        return
    if args.command == "distill":
        if args.baseline != "pan":
            _run_safety_distill(
                args.device,
                baseline_name=args.baseline,
                device_id=args.device_id,
                num_devices=args.num_devices,
                dry_run=args.dry_run,
                force_rebuild=bool(args.force_rebuild),
                **oc_kwargs,
            )
            return
        _run_baseline_distill(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            **oc_kwargs,
        )
        return
    if args.command == "safety-sft":
        _run_safety_sft(
            args.device,
            baseline_name=args.baseline,
            model_size=args.model,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            force_rebuild=bool(args.force_rebuild),
            **oc_kwargs,
        )
        return
    if args.command == "safety-distill":
        _run_safety_distill(
            args.device,
            baseline_name=args.baseline,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            force_rebuild=bool(args.force_rebuild),
            **oc_kwargs,
        )
        return
    if args.command == "safety-full":
        _run_safety_full(
            args.device,
            baseline_name=args.baseline,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            force_rebuild=bool(args.force_rebuild),
            smoke=False,
            **oc_kwargs,
        )
        return
    if args.command == "random":
        if args.baseline != "pan":
            _run_safety_random(
                args.device,
                baseline_name=args.baseline,
                device_id=args.device_id,
                num_devices=args.num_devices,
                dry_run=args.dry_run,
                force_rebuild=bool(args.force_rebuild),
                **oc_kwargs,
            )
            return
        _run_random_baseline(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            **oc_kwargs,
        )
        return
    if args.command == "sft1":
        if args.baseline == "pan":
            _run_pan_sft1_ablation(
                args.device,
                device_id=args.device_id,
                num_devices=args.num_devices,
                dry_run=args.dry_run,
                **oc_kwargs,
            )
        else:
            _run_safety_sft1(
                args.device,
                baseline_name=args.baseline,
                device_id=args.device_id,
                num_devices=args.num_devices,
                dry_run=args.dry_run,
                force_rebuild=bool(args.force_rebuild),
                **oc_kwargs,
            )
        return
    if args.command == "bothpole":
        _run_pan_bothpole(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
            **oc_kwargs,
        )
        return
    if args.command == "full":
        try:
            analyze_extras_raw = getattr(args, "phase1_analyze_extra", "[]") or "[]"
            analyze_extras = json.loads(analyze_extras_raw) if isinstance(analyze_extras_raw, str) else list(analyze_extras_raw)
            subspace_extras_raw = getattr(args, "phase1_subspace_extra", "[]") or "[]"
            subspace_extras = json.loads(subspace_extras_raw) if isinstance(subspace_extras_raw, str) else list(subspace_extras_raw)
            stage_extras_raw = getattr(args, "phase1_stage_extras", "{}") or "{}"
            stage_extras = json.loads(stage_extras_raw) if isinstance(stage_extras_raw, str) else dict(stage_extras_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Phase-1 extra arguments must be valid JSON; "
                f"got analyze={args.phase1_analyze_extra!r} subspace={args.phase1_subspace_extra!r}"
            ) from exc
        if not isinstance(analyze_extras, list) or not all(isinstance(x, str) for x in analyze_extras):
            raise ValueError("--phase1-analyze-extra must be a JSON list of strings")
        if not isinstance(subspace_extras, list) or not all(isinstance(x, str) for x in subspace_extras):
            raise ValueError("--phase1-subspace-extra must be a JSON list of strings")
        if not isinstance(stage_extras, dict) or not all(
            isinstance(key, str)
            and isinstance(tokens, list)
            and all(isinstance(token, str) for token in tokens)
            for key, tokens in stage_extras.items()
        ):
            raise ValueError("--phase1-stage-extras must be a JSON object of string lists")
        phase1_config_path = getattr(args, "phase1_config", "") or ""
        if args.baseline != "pan":
            _run_safety_full(
                args.device,
                baseline_name=args.baseline,
                device_id=args.device_id,
                num_devices=args.num_devices,
                dry_run=args.dry_run,
                force_rebuild=bool(args.force_rebuild),
                smoke=False,
                phasef_config_path=args.phasef_config,
                phase1_config_path=phase1_config_path,
                analyze_extras=analyze_extras,
                subspace_extras=subspace_extras,
                stage_extras=stage_extras,
                cell_id=getattr(args, "cell_id", "") or "",
                disable_dataset_overrides=bool(getattr(args, "disable_dataset_overrides", False)),
                **oc_kwargs,
            )
            return
        _run_full_pipeline(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            smoke=False,
            dry_run=args.dry_run,
            phasef_config_path=args.phasef_config,
            phase1_config_path=phase1_config_path,
            analyze_extras=analyze_extras,
            subspace_extras=subspace_extras,
            stage_extras=stage_extras,
            **oc_kwargs,
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
