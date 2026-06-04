#!/usr/bin/env bash
# One experiment-unit on one NPU card: train (new Phase 1 curation) + LLM judge.
#
# An "experiment unit" is the smallest block that can run on its own card
# without colliding with another card:
#   ours    = safety-full -> safety-sft1 -> safety-random  (SERIAL: they share
#             outputs/safety_full_<X>/phase1 and data/processed/safety_full_<X>)
#   sft     = SFT-only baseline (13_train_pan_sft)
#   distill = 9B->0.8B distillation (14_train_pan_distill)
#
# Different datasets never collide; same dataset's ours/sft/distill collide on
# 19_prepare only -> run prewarm_curation_data.sh FIRST, then launch all units
# WITHOUT --force-rebuild (this script does that).
#
# Usage:
#   bash scripts/run_curation_exp.sh <group> <baseline> <device_id> [--judge-only|--no-judge]
#     group    : ours | sft | distill
#     baseline : wildjailbreak | wildguardmix | beavertails_category | ...
#     device_id: NPU card index (e.g. 0, 9, 15)
#
# Env:
#   WILDGUARD_MODEL  path to allenai/wildguard           (default: models/wildguard)
#   DEVICE           runtime backend                     (default: npu)
#   JUDGE_BATCH      judge batch size                    (default: 16)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GROUP="${1:?group required: ours|sft|distill}"
BASELINE="${2:?baseline required}"
DEV="${3:?device_id required}"
MODE="${4:-both}"   # both | --judge-only | --no-judge

DEVICE="${DEVICE:-npu}"
WILDGUARD="${WILDGUARD_MODEL:-models/wildguard}"
JUDGE_BATCH="${JUDGE_BATCH:-16}"
TEST_JSONL="data/processed/eval/${BASELINE}_test.jsonl"

DO_TRAIN=1; DO_JUDGE=1
case "$MODE" in
  --judge-only) DO_TRAIN=0 ;;
  --no-judge)   DO_JUDGE=0 ;;
  both|"")      ;;
  *) echo "unknown mode: $MODE (use --judge-only|--no-judge)" >&2; exit 2 ;;
esac

PY=python
ONECLICK="scripts/15_run_oneclick.py"

run_train () {  # $@ = oneclick subcommand args
  echo "=== [train] $* (device=${DEVICE}:${DEV}) ==="
  $PY "$ONECLICK" "$@" --baseline "$BASELINE" --device "$DEVICE" --device-id "$DEV"
}

run_judge () {  # $1 = eval_suite dir
  local suite="$1"
  if [ ! -d "$suite" ]; then
    echo "[judge] skip missing $suite" >&2
    return 0
  fi
  if [ ! -f "$TEST_JSONL" ]; then
    echo "[judge] ERROR test jsonl missing: $TEST_JSONL (run prewarm first)" >&2
    return 1
  fi
  echo "=== [judge] $suite ==="
  $PY scripts/22_judge_generations.py \
    --eval-suite-dir "$suite" \
    --test-jsonl "$TEST_JSONL" \
    --judge-model "$WILDGUARD" \
    --runtime-backend "$DEVICE" --runtime-device "${DEVICE}:${DEV}" \
    --batch-size "$JUDGE_BATCH" --merge-summary
}

OURS_BASE="outputs/safety_full_${BASELINE}_${DEVICE}/phase1"
SFT_SUITE="outputs/baselines/sft_qwen35_08b_${BASELINE}_${DEVICE}/eval_suite"
DISTILL_SUITE="outputs/baselines/distill_qwen35_9b_to_08b_${BASELINE}_${DEVICE}/eval_suite"

case "$GROUP" in
  ours)
    if [ "$DO_TRAIN" -eq 1 ]; then
      run_train safety-full
      [ "$DO_JUDGE" -eq 1 ] && run_judge "${OURS_BASE}/training/eval_suite"
      run_train safety-sft1
      [ "$DO_JUDGE" -eq 1 ] && run_judge "${OURS_BASE}/training_sft1/eval_suite"
      run_train safety-random
      [ "$DO_JUDGE" -eq 1 ] && run_judge "${OURS_BASE}/training_random_same_norm/eval_suite"
    else
      run_judge "${OURS_BASE}/training/eval_suite"
      run_judge "${OURS_BASE}/training_sft1/eval_suite"
      run_judge "${OURS_BASE}/training_random_same_norm/eval_suite"
    fi
    ;;
  sft)
    [ "$DO_TRAIN" -eq 1 ] && run_train sft
    [ "$DO_JUDGE" -eq 1 ] && run_judge "$SFT_SUITE"
    ;;
  distill)
    [ "$DO_TRAIN" -eq 1 ] && run_train distill
    [ "$DO_JUDGE" -eq 1 ] && run_judge "$DISTILL_SUITE"
    ;;
  *)
    echo "unknown group: $GROUP (use ours|sft|distill)" >&2; exit 2 ;;
esac

echo "=== [done] group=$GROUP baseline=$BASELINE device=${DEVICE}:${DEV} mode=$MODE ==="
