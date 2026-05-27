#!/usr/bin/env bash
# Unified sweep dispatcher. Run from project root.
#
# Subcommands:
#   axis <ID> [DEVICE=npu] [DEV_ID=0]              -- run all cells in axis (e.g. B, A, G, C, ...)
#   cell <CELL_ID> <BASELINE> [DEVICE] [DEV_ID]    -- run single cell (B1+beavertails etc)
#   combo <NAME> [DEVICE] [DEV_ID]                 -- run named combo (defined below)
#   summary                                        -- print result CSV grouped by axis
#   winners                                        -- auto-pick per-axis winner by mean F1
#   status                                         -- pending vs completed counts
#   list                                           -- list all known cells / combos
#
# Examples:
#   bash scripts/sweep.sh axis B               # B1+B2+B3 × 3 baselines = 9 runs
#   bash scripts/sweep.sh axis G ppu 1
#   bash scripts/sweep.sh cell A2 beavertails
#   bash scripts/sweep.sh combo R2_C1          # Round-2 stack combo
#   bash scripts/sweep.sh summary
#   bash scripts/sweep.sh winners
#
# Env:
#   DRY_RUN=1   pass --dry-run through (wiring test only)
#   FORCE_REBUILD=0  disable --force-rebuild (default: enabled)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$PROJECT_ROOT/scripts/run_param_sweep.py"
SUMMARY="$PROJECT_ROOT/scripts/sweep_summary.py"
CSV="$PROJECT_ROOT/sweep_results.csv"

# ---------------------------------------------------------------------------
# Cell registry  (ID, BASELINES, EXTRA_ARGS for run_param_sweep.py)
# ---------------------------------------------------------------------------
ALL_BASELINES="pan,beavertails,safety_tuned_llamas"
BT_STL="beavertails,safety_tuned_llamas"
BT_ONLY="beavertails"

declare -A CELL_BASELINES
declare -A CELL_EXTRA

CELL_BASELINES[B1]="$ALL_BASELINES"; CELL_EXTRA[B1]='--phasef-set={"optim.layer_loss_weight":0.1}'
CELL_BASELINES[B2]="$ALL_BASELINES"; CELL_EXTRA[B2]='--phasef-set={"optim.layer_loss_weight":0.05}'
CELL_BASELINES[B3]="$ALL_BASELINES"; CELL_EXTRA[B3]='--phasef-set={"optim.layer_loss_weight":0.5}'

CELL_BASELINES[A1]="$ALL_BASELINES"; CELL_EXTRA[A1]='--phasef-set={"target.layer_loss_policy":"all"}'
CELL_BASELINES[A2]="$ALL_BASELINES"; CELL_EXTRA[A2]='--phasef-set={"target.layer_loss_policy":"label_weighted","target.harmless_layer_weight":0.5}'

CELL_BASELINES[D1]="$ALL_BASELINES"; CELL_EXTRA[D1]='--phasef-set={"optim.sft_loss_weight":1.0,"optim.layer_loss_weight":0.5}'
CELL_BASELINES[D2]="$ALL_BASELINES"; CELL_EXTRA[D2]='--phasef-set={"optim.sft_loss_weight":0.5,"optim.layer_loss_weight":0.5}'

CELL_BASELINES[G1]="$ALL_BASELINES"; CELL_EXTRA[G1]='--analyze-extra=["--top-k","1"]'
CELL_BASELINES[G2]="$ALL_BASELINES"; CELL_EXTRA[G2]='--analyze-extra=["--top-k","5"]'
CELL_BASELINES[G3]="$ALL_BASELINES"; CELL_EXTRA[G3]='--analyze-extra=["--top-k","7"]'

CELL_BASELINES[C1]="$BT_STL"; CELL_EXTRA[C1]='--subspace-extra=["--rank","8"]'
CELL_BASELINES[C2]="$BT_STL"; CELL_EXTRA[C2]='--subspace-extra=["--rank","32"]'

CELL_BASELINES[E1]="$ALL_BASELINES"; CELL_EXTRA[E1]='--phasef-set={"lora.rank":8,"lora.alpha":16.0}'
CELL_BASELINES[E2]="$ALL_BASELINES"; CELL_EXTRA[E2]='--phasef-set={"lora.rank":32,"lora.alpha":64.0}'

CELL_BASELINES[F1]="$BT_ONLY"; CELL_EXTRA[F1]='--subspace-extra=["--no-balance-labels"]'

# ---------------------------------------------------------------------------
# Combo registry (NAME, all 4 args concatenated)
# ---------------------------------------------------------------------------
declare -A COMBO_BASELINES
declare -A COMBO_EXTRA

# Round-2 templates (edit after Round 1 winners known)
COMBO_BASELINES[R2_AB]="$ALL_BASELINES"
COMBO_EXTRA[R2_AB]='--phasef-set={"target.layer_loss_policy":"label_weighted","target.harmless_layer_weight":0.5,"optim.layer_loss_weight":0.1}'

COMBO_BASELINES[R2_AD]="$ALL_BASELINES"
COMBO_EXTRA[R2_AD]='--phasef-set={"target.layer_loss_policy":"label_weighted","target.harmless_layer_weight":0.5,"optim.sft_loss_weight":1.0,"optim.layer_loss_weight":0.5}'

COMBO_BASELINES[R2_ABC]="$BT_STL"
COMBO_EXTRA[R2_ABC]='--phasef-set={"target.layer_loss_policy":"label_weighted","target.harmless_layer_weight":0.5,"optim.layer_loss_weight":0.1} --subspace-extra=["--rank","8"]'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
get_dry_flag() {
  if [[ -n "${DRY_RUN:-}" ]]; then echo "--dry-run"; else echo ""; fi
}

get_rebuild_flag() {
  # Default OFF: data unchanged across cells. Set FORCE_REBUILD=1 to force
  # 19_prepare_safety_data.py to rebuild the safety JSONL.
  if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then echo "--force-rebuild"; else echo ""; fi
}

run_one() {
  local axis_id="$1" baseline="$2" device="$3" device_id="$4" extra="$5"
  local dry_flag rebuild_flag
  dry_flag=$(get_dry_flag)
  rebuild_flag=$(get_rebuild_flag)
  echo ""
  echo "============================================================"
  echo "[sweep] axis=$axis_id baseline=$baseline device=$device:$device_id"
  echo "[sweep] extra: $extra"
  echo "============================================================"
  # shellcheck disable=SC2086
  python "$RUNNER" \
    --axis "$axis_id" \
    --baseline "$baseline" \
    --device "$device" \
    --device-id "$device_id" \
    $extra \
    $rebuild_flag \
    $dry_flag \
    || echo "[sweep][WARN] axis=$axis_id baseline=$baseline failed; continuing"
}

cmd_axis() {
  local axis="${1:?axis prefix required (e.g. B, A, G, C, E, F, D)}"
  local device="${2:-npu}"
  local device_id="${3:-0}"
  local found=0
  for cid in $(echo "${!CELL_BASELINES[@]}" | tr ' ' '\n' | sort); do
    if [[ "$cid" =~ ^${axis} ]]; then
      found=1
      IFS=',' read -r -a bls <<<"${CELL_BASELINES[$cid]}"
      for bl in "${bls[@]}"; do
        run_one "$cid" "$bl" "$device" "$device_id" "${CELL_EXTRA[$cid]}"
      done
    fi
  done
  if [[ $found -eq 0 ]]; then
    echo "[sweep] no cells match axis prefix: $axis" >&2
    return 1
  fi
}

cmd_cell() {
  local cell="${1:?cell ID required (e.g. B1, A2)}"
  local baseline="${2:?baseline required (pan/beavertails/safety_tuned_llamas)}"
  local device="${3:-npu}"
  local device_id="${4:-0}"
  if [[ -z "${CELL_EXTRA[$cell]:-}" ]]; then
    echo "[sweep] unknown cell: $cell" >&2
    return 1
  fi
  run_one "$cell" "$baseline" "$device" "$device_id" "${CELL_EXTRA[$cell]}"
}

cmd_combo() {
  local combo="${1:?combo name required (e.g. R2_AB)}"
  local device="${2:-npu}"
  local device_id="${3:-0}"
  if [[ -z "${COMBO_EXTRA[$combo]:-}" ]]; then
    echo "[sweep] unknown combo: $combo" >&2
    echo "[sweep] known combos: ${!COMBO_EXTRA[*]}" >&2
    return 1
  fi
  IFS=',' read -r -a bls <<<"${COMBO_BASELINES[$combo]}"
  for bl in "${bls[@]}"; do
    run_one "$combo" "$bl" "$device" "$device_id" "${COMBO_EXTRA[$combo]}"
  done
}

cmd_summary() {
  if [[ ! -f "$CSV" ]]; then
    echo "[sweep] no results yet: $CSV missing" >&2
    return 1
  fi
  python "$SUMMARY" --csv "$CSV" --mode all
}

cmd_winners() {
  if [[ ! -f "$CSV" ]]; then
    echo "[sweep] no results yet: $CSV missing" >&2
    return 1
  fi
  python "$SUMMARY" --csv "$CSV" --mode winners
}

cmd_status() {
  if [[ ! -f "$CSV" ]]; then
    echo "[sweep] no results yet; 0 / $(count_total_cells) cells done"
    return 0
  fi
  python "$SUMMARY" --csv "$CSV" --mode status \
    --total-cells "$(count_total_cells)"
}

count_total_cells() {
  local total=0
  for cid in "${!CELL_BASELINES[@]}"; do
    IFS=',' read -r -a bls <<<"${CELL_BASELINES[$cid]}"
    total=$((total + ${#bls[@]}))
  done
  echo $total
}

cmd_list() {
  echo "=== Cells (Round 1) ==="
  for cid in $(echo "${!CELL_BASELINES[@]}" | tr ' ' '\n' | sort); do
    printf "  %-4s  %-30s  %s\n" "$cid" "${CELL_BASELINES[$cid]}" "${CELL_EXTRA[$cid]}"
  done
  echo ""
  echo "=== Combos (Round 2 templates) ==="
  for n in $(echo "${!COMBO_BASELINES[@]}" | tr ' ' '\n' | sort); do
    printf "  %-8s  %-30s  %s\n" "$n" "${COMBO_BASELINES[$n]}" "${COMBO_EXTRA[$n]}"
  done
  echo ""
  echo "=== Total Round 1 cell-runs: $(count_total_cells) ==="
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
SUB="${1:-help}"
shift || true

case "$SUB" in
  axis)     cmd_axis "$@" ;;
  cell)     cmd_cell "$@" ;;
  combo)    cmd_combo "$@" ;;
  summary)  cmd_summary ;;
  winners)  cmd_winners ;;
  status)   cmd_status ;;
  list)     cmd_list ;;
  help|*)
    head -n 25 "$0" | grep -E "^#( |$)" | sed 's/^# \?//'
    echo ""
    echo "Quick examples:"
    echo "  bash scripts/sweep.sh list                    # see all cells/combos"
    echo "  bash scripts/sweep.sh axis B                  # all B cells"
    echo "  bash scripts/sweep.sh cell A2 beavertails     # single cell"
    echo "  bash scripts/sweep.sh combo R2_AB             # Round 2 combo"
    echo "  bash scripts/sweep.sh summary"
    echo "  bash scripts/sweep.sh winners"
    echo "  bash scripts/sweep.sh status"
    ;;
esac
