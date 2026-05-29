#!/usr/bin/env bash
# One-click experiment runner for the 4 new safety baselines:
#   wildjailbreak / wildguardmix / hh_rlhf / beavertails_category
#
# Default sequence per dataset (~6-10h on single NPU, total ~24-40h × 4):
#   1. nosft_08b      base 0.8b eval (no training)
#   2. nosft_9b       teacher reference eval
#   3. safety_sft     SFT-only on safety data
#   4. safety_distill teacher-supervised CE 9b→0.8b
#   5. ours           safety-full (PAN-Ours equivalent)
#   6. sft1           ablation (sft_loss=1.0, layer_loss=0.0)
#   7. random         random-vector ablation
#
# Usage:
#   bash scripts/run_new_baselines.sh                      # all 4 datasets, all 7 experiments
#   bash scripts/run_new_baselines.sh wildjailbreak        # one dataset, all experiments
#   bash scripts/run_new_baselines.sh wildjailbreak ours   # one dataset, single experiment
#   DEVICE_ID=13 bash scripts/run_new_baselines.sh         # change NPU id (default 0)
#   EXPERIMENTS="safety_sft safety_distill ours" bash scripts/run_new_baselines.sh
#   DRY_RUN=1 bash scripts/run_new_baselines.sh wildjailbreak ours   # dry-run wiring only
#
# Env:
#   DEVICE       npu (default) or ppu (only old 3 baselines have PPU configs)
#   DEVICE_ID    accelerator ordinal (default 0)
#   DRY_RUN=1    pass --dry-run to launcher (no GPU work)
#   FORCE_REBUILD=1  rebuild safety JSONL via 19_prepare_safety_data.py
#   ENABLE_OC=1  pass --enable-opencompass --opencompass-dir external/opencompass
#   EXPERIMENTS  space-separated subset of {nosft_08b,nosft_9b,safety_sft,safety_distill,ours,sft1,random}
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="$PROJECT_ROOT/scripts/15_run_oneclick.py"

DEVICE="${DEVICE:-npu}"
DEVICE_ID="${DEVICE_ID:-0}"
DRY_RUN_FLAG=""
[[ -n "${DRY_RUN:-}" ]] && DRY_RUN_FLAG="--dry-run"
FORCE_REBUILD_FLAG=""
[[ -n "${FORCE_REBUILD:-}" ]] && FORCE_REBUILD_FLAG="--force-rebuild"
OC_FLAGS=""
if [[ -n "${ENABLE_OC:-}" ]]; then
  OC_FLAGS="--enable-opencompass --opencompass-dir external/opencompass"
fi

NEW_BASELINES_DEFAULT=("wildjailbreak" "wildguardmix" "hh_rlhf" "beavertails_category")
ALL_EXPERIMENTS_DEFAULT=("nosft_08b" "nosft_9b" "safety_sft" "safety_distill" "ours" "sft1" "random")

# Determine target baselines + experiments
TARGET_BASELINE="${1:-}"
TARGET_EXP="${2:-}"

if [[ -n "$TARGET_BASELINE" ]]; then
  BASELINES=("$TARGET_BASELINE")
else
  BASELINES=("${NEW_BASELINES_DEFAULT[@]}")
fi

if [[ -n "$TARGET_EXP" ]]; then
  EXPERIMENTS_ARR=("$TARGET_EXP")
elif [[ -n "${EXPERIMENTS:-}" ]]; then
  # shellcheck disable=SC2206
  EXPERIMENTS_ARR=(${EXPERIMENTS})
else
  EXPERIMENTS_ARR=("${ALL_EXPERIMENTS_DEFAULT[@]}")
fi

run_exp() {
  local baseline="$1" exp="$2"
  local cmd=""
  case "$exp" in
    nosft_08b)
      cmd="python $LAUNCHER nosft --model 0.8b --baseline $baseline --device $DEVICE --device-id $DEVICE_ID"
      ;;
    nosft_9b)
      cmd="python $LAUNCHER nosft --model 9b --baseline $baseline --device $DEVICE --device-id $DEVICE_ID"
      ;;
    safety_sft)
      cmd="python $LAUNCHER safety-sft --baseline $baseline --device $DEVICE --device-id $DEVICE_ID $FORCE_REBUILD_FLAG"
      ;;
    safety_distill)
      cmd="python $LAUNCHER safety-distill --baseline $baseline --device $DEVICE --device-id $DEVICE_ID $FORCE_REBUILD_FLAG"
      ;;
    ours)
      cmd="python $LAUNCHER safety-full --baseline $baseline --device $DEVICE --device-id $DEVICE_ID $FORCE_REBUILD_FLAG"
      ;;
    sft1)
      cmd="python $LAUNCHER sft1 --baseline $baseline --device $DEVICE --device-id $DEVICE_ID $FORCE_REBUILD_FLAG"
      ;;
    random)
      cmd="python $LAUNCHER random --baseline $baseline --device $DEVICE --device-id $DEVICE_ID $FORCE_REBUILD_FLAG"
      ;;
    *)
      echo "[run_new_baselines] unknown experiment: $exp" >&2
      return 1
      ;;
  esac

  # OpenCompass flags only apply to subcommands that route through PhaseF/eval (skip nosft for now if undesired)
  cmd="$cmd $OC_FLAGS $DRY_RUN_FLAG"

  echo ""
  echo "============================================================"
  echo "[run_new_baselines] baseline=$baseline exp=$exp device=$DEVICE:$DEVICE_ID"
  echo "[run_new_baselines] $cmd"
  echo "============================================================"
  # shellcheck disable=SC2086
  eval $cmd || echo "[run_new_baselines][WARN] $baseline/$exp failed; continuing"
}

cd "$PROJECT_ROOT"

START=$(date +%s)
for baseline in "${BASELINES[@]}"; do
  for exp in "${EXPERIMENTS_ARR[@]}"; do
    run_exp "$baseline" "$exp"
  done
done
END=$(date +%s)

echo ""
echo "[run_new_baselines] Done. Elapsed: $((END-START))s"
echo "[run_new_baselines] Targets: ${BASELINES[*]} × ${EXPERIMENTS_ARR[*]}"
