#!/usr/bin/env bash
# Unified sweep dispatcher. Run from project root.
#
# Subcommands:
#   axis <ID> [DEVICE=npu] [DEV_ID=0]              -- sequential: all cells in axis on one card
#   axis_fanout <ID> [N_CARDS=3] [START_ID=0]      -- parallel: one cell per card, queue refills
#                                                    (e.g. axis B has 9 pairs, N=9 cards => all
#                                                    9 run concurrently, finish ~3-6h)
#   axis_parallel <ID> [N_CARDS=3] [START_ID=0]    -- per-cell baselines parallel, cells sequential
#   axis_per_cell <ID> [N_CARDS=3] [START_ID=0]    -- one cell per card (all baselines sequential
#                                                    within card); e.g. axis B 3 cells -> 3 cards
#                                                    means card 0=B1, card 1=B2, card 2=B3
#   cell <CELL_ID> <BASELINE> [DEVICE] [DEV_ID]    -- run single cell
#   cell_loop <CELL_ID> [DEVICE] [DEV_ID]          -- run single cell on all baselines (one card)
#   combo <NAME> [DEVICE] [DEV_ID]                 -- run named combo
#   summary                                        -- print result CSV grouped by axis
#   winners                                        -- auto-pick per-axis winner by mean F1
#   status                                         -- pending vs completed counts
#   list                                           -- list all known cells / combos
#
# Examples (set JUDGE=1 to score by WildGuard judge net = best-net epoch):
#   JUDGE=1 bash scripts/sweep.sh axis TK              # all top_k cells x $SWEEP4
#   JUDGE=1 bash scripts/sweep.sh axis_fanout TK npu 16  # fan TK pairs over 16 dies
#   JUDGE=1 bash scripts/sweep.sh cell DEF pan         # single anchor cell
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
ALL_BASELINES="pan,beavertails_category,safety_tuned_llamas"
BT_STL="beavertails_category,safety_tuned_llamas"
BT_ONLY="beavertails_category"
# Per-dataset override search (judged re-baseline, post placement-fix a84bec0).
# These are the four datasets we now sweep to derive each one's
# SAFETY_PHASE_OVERRIDES_BY_BASELINE entry (like wjb/wgm already have).
SWEEP4="pan,safety_tuned_llamas,coconot,c5"

declare -A CELL_BASELINES
declare -A CELL_EXTRA

# ===========================================================================
# Per-dataset override search (judged re-baseline, post placement-fix a84bec0).
# Datasets = $SWEEP4 (pan, STL, coconot, c5). Objective = judge net (HR-OR),
# best-net epoch (run with JUDGE=1). DEF = current live defaults
# (top_k 5 / energy-threshold 0.8 / rank-cap 32 / lambda 0.25 / epochs 3 /
# harmful_only). MAIN search tunes only top_k, lambda, epochs.
#
# Option A (paper-grounded, agreed): energy-threshold tau is fixed at PAN's
# representative 0.8 and rank-cap at 32 for ALL main cells. Pan et al. 2025 (ICML)
# define the energy-threshold effective rank in their section 4 ONLY as a
# diagnostic of the residual space's dimensionality (Fig 2, plotted across
# tau in {0.4,0.6,0.7,0.8,0.9}); they do not tune it, and they apply SVD to
# W-I, not to the harmful-harmless contrast this repo uses. So tau / rank-cap
# are demoted to an APPENDIX sensitivity sweep (SE*/RC* below) -- reported as a
# net-vs-tau curve, NOT cherry-picked.
#
# Legacy Round-1 cells A-G were removed: they ran under the LoRA placement bug
# (pre a84bec0) so every layer-loss axis was measured with L_layer near-inert,
# and they tuned axes (policy, fixed --rank, LoRA rank, no-balance) dropped from
# this search. See git history before a84bec0 if needed.
# ===========================================================================

# --- Anchors -----------------------------------------------------------------
CELL_BASELINES[DEF]="$SWEEP4";  CELL_EXTRA[DEF]=''                                               # ours @ current defaults
CELL_BASELINES[SFT1]="$SWEEP4"; CELL_EXTRA[SFT1]='--phasef-set={"optim.layer_loss_weight":0.0}'  # L_layer ablation (lambda=0)

# --- TK: top_k = #key layers = #supervised LoRA layers (the core lever) ------
# DEF already covers top_k=5.
CELL_BASELINES[TK2]="$SWEEP4"; CELL_EXTRA[TK2]='--analyze-extra=["--top-k","2"]'
CELL_BASELINES[TK3]="$SWEEP4"; CELL_EXTRA[TK3]='--analyze-extra=["--top-k","3"]'
CELL_BASELINES[TK4]="$SWEEP4"; CELL_EXTRA[TK4]='--analyze-extra=["--top-k","4"]'
CELL_BASELINES[TK6]="$SWEEP4"; CELL_EXTRA[TK6]='--analyze-extra=["--top-k","6"]'
CELL_BASELINES[TK7]="$SWEEP4"; CELL_EXTRA[TK7]='--analyze-extra=["--top-k","7"]'
CELL_BASELINES[TK8]="$SWEEP4"; CELL_EXTRA[TK8]='--analyze-extra=["--top-k","8"]'

# --- LW: L_layer weight lambda ----------------------------------------------
# DEF covers lambda=0.25; SFT1 covers lambda=0.
CELL_BASELINES[LW05]="$SWEEP4";  CELL_EXTRA[LW05]='--phasef-set={"optim.layer_loss_weight":0.05}'
CELL_BASELINES[LW10]="$SWEEP4";  CELL_EXTRA[LW10]='--phasef-set={"optim.layer_loss_weight":0.1}'
CELL_BASELINES[LW50]="$SWEEP4";  CELL_EXTRA[LW50]='--phasef-set={"optim.layer_loss_weight":0.5}'
CELL_BASELINES[LW100]="$SWEEP4"; CELL_EXTRA[LW100]='--phasef-set={"optim.layer_loss_weight":1.0}'

# --- EP: PhaseF epochs (DEF covers 3) ---------------------------------------
CELL_BASELINES[EP5]="$SWEEP4"; CELL_EXTRA[EP5]='--phasef-set={"optim.epochs":5}'

# --- APPENDIX sensitivity (NOT main search) ---------------------------------
# tau (energy-threshold) sweep @ rank-cap 32 -- report net-vs-tau, do not pick a
# winner. DEF covers tau=0.8.
CELL_BASELINES[SE06]="$SWEEP4";  CELL_EXTRA[SE06]='--subspace-extra=["--energy-threshold","0.6"]'
CELL_BASELINES[SE07]="$SWEEP4";  CELL_EXTRA[SE07]='--subspace-extra=["--energy-threshold","0.7"]'
CELL_BASELINES[SE09]="$SWEEP4";  CELL_EXTRA[SE09]='--subspace-extra=["--energy-threshold","0.9"]'
CELL_BASELINES[SE095]="$SWEEP4"; CELL_EXTRA[SE095]='--subspace-extra=["--energy-threshold","0.95"]'
# rank-cap sweep @ tau 0.8 (repo engineering knob, NOT in PAN). DEF covers cap=32.
CELL_BASELINES[RC8]="$SWEEP4";  CELL_EXTRA[RC8]='--subspace-extra=["--rank-cap","8"]'
CELL_BASELINES[RC16]="$SWEEP4"; CELL_EXTRA[RC16]='--subspace-extra=["--rank-cap","16"]'
CELL_BASELINES[RC64]="$SWEEP4"; CELL_EXTRA[RC64]='--subspace-extra=["--rank-cap","64"]'

# ---------------------------------------------------------------------------
# Combo registry (NAME, all 4 args concatenated)
# ---------------------------------------------------------------------------
declare -A COMBO_BASELINES
declare -A COMBO_EXTRA

# Stage-2 confirmation combos: filled in per dataset AFTER stage-1 (TK/LW/EP)
# winners are known -- stack each dataset's best top_k with its best lambda and
# run >=2 seeds + JUDGE to confirm and to verify ours > sft1. Example template
# (edit the top_k / lambda to the stage-1 winner, run per dataset):
#   COMBO_BASELINES[S2_pan]="pan"
#   COMBO_EXTRA[S2_pan]='--analyze-extra=["--top-k","6"] --phasef-set={"optim.layer_loss_weight":0.5}'

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

get_judge_flag() {
  # JUDGE=1 -> run the WildGuard judge after each cell and score by best-net
  # (max judge_HR - judge_OR) epoch, matching D:/output/_table.py. The judge
  # reuses the cell's own die (it runs after training frees it). Override the
  # judge die with JUDGE_DEVICE_ID and the model with JUDGE_MODEL if needed.
  if [[ "${JUDGE:-0}" == "1" ]]; then
    local f="--judge"
    [[ -n "${JUDGE_MODEL:-}" ]] && f="$f --judge-model ${JUDGE_MODEL}"
    [[ -n "${JUDGE_DEVICE_ID:-}" ]] && f="$f --judge-device-id ${JUDGE_DEVICE_ID}"
    echo "$f"
  else
    echo ""
  fi
}

run_one() {
  local axis_id="$1" baseline="$2" device="$3" device_id="$4" extra="$5"
  local dry_flag rebuild_flag judge_flag
  dry_flag=$(get_dry_flag)
  rebuild_flag=$(get_rebuild_flag)
  judge_flag=$(get_judge_flag)
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
    $judge_flag \
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

cmd_cell_loop() {
  # Run a single cell value across all its baselines sequentially on one card.
  # Use this as the unit of work when fanning parameter VALUES (not pairs)
  # across cards: e.g. card 0 runs B1 on pan+bt+stl, card 1 runs B2 on
  # pan+bt+stl, etc.
  local cell="${1:?cell ID required (e.g. B1)}"
  local device="${2:-npu}"
  local device_id="${3:-0}"
  if [[ -z "${CELL_EXTRA[$cell]:-}" ]]; then
    echo "[sweep] unknown cell: $cell" >&2
    return 1
  fi
  IFS=',' read -r -a bls <<<"${CELL_BASELINES[$cell]}"
  echo "[sweep] cell_loop cell=$cell baselines=(${bls[*]}) device=$device:$device_id"
  for bl in "${bls[@]}"; do
    run_one "$cell" "$bl" "$device" "$device_id" "${CELL_EXTRA[$cell]}"
  done
}

cmd_axis_per_cell() {
  # Each parameter VALUE (cell) in <axis> gets its own card. Baselines within
  # a cell run sequentially on that card.
  #
  # Examples:
  #   axis B has 3 cells (B1, B2, B3), each cell runs 3 baselines.
  #   - 3 cards: B1->npu:0, B2->npu:1, B3->npu:2; each card sequential 3
  #     baselines, ~9-18h per card, total wall-clock ~9-18h.
  #   - 1 card:  queue: B1 then B2 then B3, ~27-54h.
  #
  # Requires parallel-safe (a) refactor.
  local axis="${1:?axis prefix required}"
  local num_cards="${2:-3}"
  local start_device_id="${3:-0}"
  local device="${DEVICE:-npu}"

  local -a cells=()
  for cid in $(echo "${!CELL_BASELINES[@]}" | tr ' ' '\n' | sort); do
    if [[ "$cid" =~ ^${axis} ]]; then
      cells+=( "$cid" )
    fi
  done

  if [[ ${#cells[@]} -eq 0 ]]; then
    echo "[sweep] no cells match axis prefix: $axis" >&2
    return 1
  fi

  echo "[sweep] axis_per_cell axis=$axis num_cells=${#cells[@]} num_cards=$num_cards start=$start_device_id"

  local -A pid_dev
  local -a free_cards
  local d
  for (( d = 0; d < num_cards; d++ )); do
    free_cards+=( "$(( start_device_id + d ))" )
  done

  for cid in "${cells[@]}"; do
    while [[ ${#free_cards[@]} -eq 0 ]]; do
      for pid in "${!pid_dev[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          wait "$pid" || echo "[sweep][WARN] child pid=$pid failed"
          free_cards+=( "${pid_dev[$pid]}" )
          unset 'pid_dev[$pid]'
        fi
      done
      [[ ${#free_cards[@]} -eq 0 ]] && sleep 5
    done
    local dev_id="${free_cards[0]}"
    free_cards=("${free_cards[@]:1}")
    echo "[sweep] dispatch cell=$cid -> device=$device:$dev_id (all baselines sequential)"
    cmd_cell_loop "$cid" "$device" "$dev_id" &
    local pid=$!
    pid_dev[$pid]="$dev_id"
  done

  for pid in "${!pid_dev[@]}"; do
    wait "$pid" || echo "[sweep][WARN] child pid=$pid failed"
  done
  echo "[sweep] axis_per_cell axis=$axis done"
}

cmd_axis_fanout() {
  # Fan out ALL (cell, baseline) pairs in <axis> across <num_cards> NPUs,
  # one cell per card at a time. Each card pulls the next pair from queue
  # when it finishes its current pair (no waiting for full batch).
  #
  # Examples:
  #   axis B has 3 cells x 3 baselines = 9 pairs
  #   - 9 cards: all 9 pairs run simultaneously, finish ~3-6h
  #   - 3 cards: 3 batches of 3, finish ~9-18h
  #   - 1 card:  sequential (= cmd_axis behavior)
  #
  # Requires parallel-safe (a) refactor (yaml copy isolation, per-cell
  # output dirs). Each pair gets unique cell_id, zero race.
  local axis="${1:?axis prefix required}"
  local num_cards="${2:-3}"
  local start_device_id="${3:-0}"
  local device="${DEVICE:-npu}"

  # Collect all (cell_id, baseline) pairs matching axis prefix.
  local -a pairs=()
  for cid in $(echo "${!CELL_BASELINES[@]}" | tr ' ' '\n' | sort); do
    if [[ "$cid" =~ ^${axis} ]]; then
      IFS=',' read -r -a bls <<<"${CELL_BASELINES[$cid]}"
      for bl in "${bls[@]}"; do
        pairs+=("$cid|$bl")
      done
    fi
  done

  if [[ ${#pairs[@]} -eq 0 ]]; then
    echo "[sweep] no cells match axis prefix: $axis" >&2
    return 1
  fi

  echo "[sweep] axis=$axis num_pairs=${#pairs[@]} num_cards=$num_cards start_device_id=$start_device_id"

  # Track running pids and their dev_ids.
  local -A pid_dev
  local -a free_cards
  local d
  for (( d = 0; d < num_cards; d++ )); do
    free_cards+=( "$(( start_device_id + d ))" )
  done

  for pair in "${pairs[@]}"; do
    # Wait until at least one card is free.
    while [[ ${#free_cards[@]} -eq 0 ]]; do
      # Reap one finished child.
      for pid in "${!pid_dev[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          wait "$pid" || echo "[sweep][WARN] child pid=$pid failed"
          free_cards+=( "${pid_dev[$pid]}" )
          unset 'pid_dev[$pid]'
        fi
      done
      [[ ${#free_cards[@]} -eq 0 ]] && sleep 5
    done

    local cid="${pair%%|*}"
    local bl="${pair#*|}"
    local dev_id="${free_cards[0]}"
    free_cards=("${free_cards[@]:1}")

    echo "[sweep] dispatch cell=$cid baseline=$bl -> device=$device:$dev_id"
    run_one "$cid" "$bl" "$device" "$dev_id" "${CELL_EXTRA[$cid]}" &
    local pid=$!
    pid_dev[$pid]="$dev_id"
  done

  # Wait for remaining children.
  for pid in "${!pid_dev[@]}"; do
    wait "$pid" || echo "[sweep][WARN] child pid=$pid failed"
  done
  echo "[sweep] axis=$axis fanout done"
}

cmd_axis_parallel() {
  # Run all cells in <axis> sequentially, but within each cell fan out
  # baselines across <num_cards> NPUs starting at <start_device_id>.
  # Requires the parallel-safe (a) refactor: yaml copy isolation + per-cell
  # output dirs (each child run_param_sweep.py uses a unique cell_id, so no
  # yaml or output dir collisions).
  local axis="${1:?axis prefix required}"
  local num_cards="${2:-3}"
  local start_device_id="${3:-0}"
  local device="${DEVICE:-npu}"
  local found=0
  for cid in $(echo "${!CELL_BASELINES[@]}" | tr ' ' '\n' | sort); do
    if [[ "$cid" =~ ^${axis} ]]; then
      found=1
      IFS=',' read -r -a bls <<<"${CELL_BASELINES[$cid]}"
      echo ""
      echo "[sweep] cell=$cid baselines=(${bls[*]}) fanout=${num_cards} cards start=$start_device_id"
      local pids=()
      local i=0
      for bl in "${bls[@]}"; do
        local dev_id=$(( (i + start_device_id) % num_cards + start_device_id ))
        # dev_id wraps within [start_device_id, start_device_id + num_cards)
        dev_id=$(( start_device_id + (i % num_cards) ))
        run_one "$cid" "$bl" "$device" "$dev_id" "${CELL_EXTRA[$cid]}" &
        pids+=($!)
        i=$((i+1))
      done
      local rc=0
      for pid in "${pids[@]}"; do
        wait "$pid" || { echo "[sweep][WARN] child pid=$pid failed"; rc=1; }
      done
      echo "[sweep] cell=$cid done (rc summary=$rc)"
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
  axis)           cmd_axis "$@" ;;
  axis_fanout)    cmd_axis_fanout "$@" ;;
  axis_parallel)  cmd_axis_parallel "$@" ;;
  axis_per_cell)  cmd_axis_per_cell "$@" ;;
  cell)           cmd_cell "$@" ;;
  cell_loop)      cmd_cell_loop "$@" ;;
  combo)          cmd_combo "$@" ;;
  summary)        cmd_summary ;;
  winners)        cmd_winners ;;
  status)         cmd_status ;;
  list)           cmd_list ;;
  help|*)
    head -n 25 "$0" | grep -E "^#( |$)" | sed 's/^# \?//'
    echo ""
    echo "Quick examples (JUDGE=1 -> score by WildGuard judge net):"
    echo "  bash scripts/sweep.sh list                        # see all cells/combos"
    echo "  JUDGE=1 bash scripts/sweep.sh axis TK             # all top_k cells x SWEEP4"
    echo "  JUDGE=1 bash scripts/sweep.sh axis_fanout TK npu 16  # fan TK over 16 dies"
    echo "  JUDGE=1 bash scripts/sweep.sh cell DEF pan        # single anchor cell"
    echo "  bash scripts/sweep.sh summary"
    echo "  bash scripts/sweep.sh winners"
    echo "  bash scripts/sweep.sh status"
    ;;
esac
