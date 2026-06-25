#!/usr/bin/env bash
# Run the cross-scale safety-transfer matrix for the extension model pairs:
#   4 pairs x 6 datasets x 5 variants (ours, sft1, sft, distill, nosft),
# scheduling one (pair,dataset) UNIT per NPU die across dies 0-15.
#
# Why UNIT = (pair,dataset): ours + sft1 SHARE one per-pair Phase-1 precompute
# (outputs/safety_full_<ds>_npu_<pair>/phase1) and must not race, so a unit runs
# its variants SEQUENTIALLY on a single die (ours computes phase1, sft1 reuses
# it; sft/distill/nosft are phase1-independent). Distinct (pair,dataset) units
# run concurrently on different dies with no output collision -- every variant is
# pair/student-namespaced.
#
# Prereqs: models downloaded (scripts/download_models.sh); per-pair configs
# generated (scripts/gen_pair_configs.py, already committed); shared dataset data
# already prepared by the Qwen runs (data/processed/safety_full_<ds>/ + eval
# jsonls -- reused as-is, pair-independent). Source your NPU env first:
#   conda activate safety310 && source set_env.sh
#   export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
# (this script does NOT source them.)
#
# Usage:
#   bash scripts/run_pairs.sh                                 # all 4 pairs x 6 ds, dies 0-15
#   PAIRS="llama31_8b_to_1b" bash scripts/run_pairs.sh        # one pair (smoke)
#   DATASETS="pan c5" bash scripts/run_pairs.sh               # subset of datasets
#   VARIANTS="ours sft1" bash scripts/run_pairs.sh            # subset of variants
#   NUM_DIES=8 START_DIE=0 bash scripts/run_pairs.sh          # fewer dies
#   DRY_RUN=1 bash scripts/run_pairs.sh                       # print commands only
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
LOGDIR="${LOGDIR:-$PROJECT_ROOT/outputs/run_pairs_logs}"
mkdir -p "$LOGDIR"

PAIRS="${PAIRS:-llama31_8b_to_1b qwen3_8b_to_06b qwen3_8b_to_4b qwen3_4b_to_06b}"
DATASETS="${DATASETS:-pan safety_tuned_llamas coconot c5 wildjailbreak wildguardmix}"
VARIANTS="${VARIANTS:-ours sft1 sft distill nosft}"
NUM_DIES="${NUM_DIES:-16}"
START_DIE="${START_DIE:-0}"
DRY=""; [[ -n "${DRY_RUN:-}" ]] && DRY="--dry-run"

run_variant() {
  local variant="$1" pair="$2" ds="$3" die="$4"
  local common=(--device npu --device-id "$die" --pair "$pair" --baseline "$ds")
  # Unified subcommands route pan vs safety internally:
  #   full    --baseline pan -> PAN pipeline; safety -> safety-full (ours)
  #   sft1 / sft / distill / nosft  handle both pan and safety datasets.
  case "$variant" in
    ours)    python scripts/15_run_oneclick.py full    "${common[@]}" $DRY ;;
    sft1)    python scripts/15_run_oneclick.py sft1    "${common[@]}" $DRY ;;
    sft)     python scripts/15_run_oneclick.py sft     "${common[@]}" $DRY ;;
    distill) python scripts/15_run_oneclick.py distill "${common[@]}" $DRY ;;
    nosft)   python scripts/15_run_oneclick.py nosft --role student "${common[@]}" $DRY ;;
    *) echo "[run][ERR] unknown variant: $variant" >&2; return 1 ;;
  esac
}

run_unit() {
  local pair="$1" ds="$2" die="$3"
  local log="$LOGDIR/${pair}__${ds}__npu${die}.log"
  {
    echo "[run] UNIT pair=$pair ds=$ds die=$die start=$(date '+%F %T')"
    for v in $VARIANTS; do
      echo ""
      echo "================ [$pair / $ds / die$die] variant=$v ================"
      run_variant "$v" "$pair" "$ds" "$die" || echo "[run][WARN] variant=$v FAILED ($pair/$ds)"
    done
    echo "[run] UNIT done pair=$pair ds=$ds die=$die end=$(date '+%F %T')"
  } >"$log" 2>&1
}

# Build the unit queue (pair x dataset).
units=()
for p in $PAIRS; do for d in $DATASETS; do units+=("$p|$d"); done; done
echo "[run] units=${#units[@]} dies=$NUM_DIES start=$START_DIE variants=($VARIANTS) logs=$LOGDIR ${DRY:+(DRY-RUN)}"

# Fan units across dies with queue-refill (one unit per die at a time).
declare -A pid_die
free=(); for ((i=0; i<NUM_DIES; i++)); do free+=("$((START_DIE+i))"); done
for unit in "${units[@]}"; do
  while [[ ${#free[@]} -eq 0 ]]; do
    for pid in "${!pid_die[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true; free+=("${pid_die[$pid]}"); unset 'pid_die[$pid]'
      fi
    done
    [[ ${#free[@]} -eq 0 ]] && sleep 5
  done
  p="${unit%%|*}"; d="${unit#*|}"; die="${free[0]}"; free=("${free[@]:1}")
  echo "[run] dispatch pair=$p ds=$d -> die$die  ($(date '+%T'))"
  run_unit "$p" "$d" "$die" &
  pid_die[$!]="$die"
done
for pid in "${!pid_die[@]}"; do wait "$pid" || true; done
echo "[run] ALL UNITS DONE. logs in $LOGDIR"
