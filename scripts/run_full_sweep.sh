#!/usr/bin/env bash
# Round 1 sweep orchestrator. 42 ablation cells across 7 axes.
#
# Usage:
#   bash scripts/run_full_sweep.sh [device=npu] [device_id=0] [axis_filter=""]
#
# Examples:
#   bash scripts/run_full_sweep.sh ppu 0           # everything on ppu:0
#   bash scripts/run_full_sweep.sh npu 0 "B|A"     # only axis B and A
#   bash scripts/run_full_sweep.sh npu 0 "G"       # only top-K ablations
#
# Each cell:
#  - patches phaseF / phase1 yaml in place
#  - patches launcher analyze_args / subspace_args if needed
#  - runs scripts/15_run_oneclick.py safety-full ...
#  - parses final_summary.json and appends to sweep_results.csv
#  - restores all touched files
set -euo pipefail

DEVICE="${1:-npu}"
DEVICE_ID="${2:-0}"
AXIS_FILTER="${3:-}"   # regex over axis label; empty = all
DRY_RUN_FLAG="${DRY_RUN:-}"   # set DRY_RUN=1 env to pass --dry-run through

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$PROJECT_ROOT/scripts/run_param_sweep.py"

echo "[sweep] device=$DEVICE device_id=$DEVICE_ID filter=${AXIS_FILTER:-<all>}"

# Cells encoded as:  AXIS|BASELINES|EXTRA_ARGS
# BASELINES one of: pan,beavertails,safety_tuned_llamas (any subset; comma-sep)
# EXTRA_ARGS = remaining CLI passed to run_param_sweep.py
CELLS=(
  # --- Axis B: layer_loss_weight (lambda) sweep ---
  'B1|pan,beavertails,safety_tuned_llamas|--phasef-set={"optim.layer_loss_weight":0.1}'
  'B2|pan,beavertails,safety_tuned_llamas|--phasef-set={"optim.layer_loss_weight":0.05}'
  'B3|pan,beavertails,safety_tuned_llamas|--phasef-set={"optim.layer_loss_weight":0.5}'
  # --- Axis A: layer_loss_policy ---
  'A1|pan,beavertails,safety_tuned_llamas|--phasef-set={"target.layer_loss_policy":"all"}'
  'A2|pan,beavertails,safety_tuned_llamas|--phasef-set={"target.layer_loss_policy":"label_weighted","target.harmless_layer_weight":0.5}'
  # --- Axis D: (sft, lambda) pair ---
  'D1|pan,beavertails,safety_tuned_llamas|--phasef-set={"optim.sft_loss_weight":1.0,"optim.layer_loss_weight":0.5}'
  'D2|pan,beavertails,safety_tuned_llamas|--phasef-set={"optim.sft_loss_weight":0.5,"optim.layer_loss_weight":0.5}'
  # --- Axis G: key layer top-K ---
  'G1|pan,beavertails,safety_tuned_llamas|--analyze-extra=["--top-k","1"]'
  'G2|pan,beavertails,safety_tuned_llamas|--analyze-extra=["--top-k","5"]'
  'G3|pan,beavertails,safety_tuned_llamas|--analyze-extra=["--top-k","7"]'
  # --- Axis C: SVD rank (skip PAN; insensitive) ---
  'C1|beavertails,safety_tuned_llamas|--subspace-extra=["--rank","8"]'
  'C2|beavertails,safety_tuned_llamas|--subspace-extra=["--rank","32"]'
  # --- Axis E: LoRA rank (alpha = 2*rank) ---
  'E1|pan,beavertails,safety_tuned_llamas|--phasef-set={"lora.rank":8,"lora.alpha":16.0}'
  'E2|pan,beavertails,safety_tuned_llamas|--phasef-set={"lora.rank":32,"lora.alpha":64.0}'
  # --- Axis F: balance_labels off (BT only; worst class imbalance) ---
  'F1|beavertails|--subspace-extra=["--no-balance-labels"]'
)

run_cell() {
  local axis="$1"
  local baseline="$2"
  local extra="$3"
  echo ""
  echo "============================================================"
  echo "[sweep] axis=$axis baseline=$baseline device=$DEVICE:$DEVICE_ID"
  echo "[sweep] extra: $extra"
  echo "============================================================"
  local dry_flag=""
  if [[ -n "$DRY_RUN_FLAG" ]]; then
    dry_flag="--dry-run"
  fi
  # shellcheck disable=SC2086
  python "$RUNNER" \
    --axis "$axis" \
    --baseline "$baseline" \
    --device "$DEVICE" \
    --device-id "$DEVICE_ID" \
    $extra \
    $dry_flag \
    || echo "[sweep][WARN] axis=$axis baseline=$baseline failed; continuing"
}

for cell in "${CELLS[@]}"; do
  AXIS="${cell%%|*}"
  rest="${cell#*|}"
  BASELINES="${rest%%|*}"
  EXTRA="${rest#*|}"

  if [[ -n "$AXIS_FILTER" && ! "$AXIS" =~ $AXIS_FILTER ]]; then
    continue
  fi

  IFS=',' read -r -a BASELINE_ARR <<<"$BASELINES"
  for BL in "${BASELINE_ARR[@]}"; do
    run_cell "$AXIS" "$BL" "$EXTRA"
  done
done

echo ""
echo "[sweep] All Round 1 ablations done."
echo "[sweep] Results: $PROJECT_ROOT/sweep_results.csv"
