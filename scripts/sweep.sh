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
ALL_BASELINES="pan,beavertails_category,safety_tuned_llamas"
BT_STL="beavertails_category,safety_tuned_llamas"
BT_ONLY="beavertails_category"

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
