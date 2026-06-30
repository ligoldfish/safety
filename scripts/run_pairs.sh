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
# Explicit die pool. Default = START_DIE..START_DIE+NUM_DIES-1. Pass DIES to
# EXCLUDE dies you know are occupied, e.g. DIES="1 2 4 5 6 7" (the surest way to
# avoid dispatching onto a busy die).
if [[ -n "${DIES:-}" ]]; then
  read -r -a DIE_POOL <<<"$DIES"
else
  DIE_POOL=(); for ((i=0; i<NUM_DIES; i++)); do DIE_POOL+=("$((START_DIE+i))"); done
fi
# OOM avoidance knobs:
#   IDLE_GATE=1        -> before dispatch, skip a die whose npu-smi HBM usage is
#                         above IDLE_MAX_USED_MIB (i.e. occupied by another proc).
#                         Best-effort: if npu-smi is absent/unparseable, ALLOW and
#                         rely on MAX_RETRIES. Set IDLE_GATE=0 to disable.
#   MAX_RETRIES=1      -> a UNIT whose every variant failed (contention signature)
#                         is re-queued onto another free die, up to this many times.
IDLE_GATE="${IDLE_GATE:-1}"
IDLE_MAX_USED_MIB="${IDLE_MAX_USED_MIB:-2000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
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

# Best-effort idle check: return 0 (idle) when die's HBM usage < threshold, OR
# when npu-smi is unavailable/unparseable (allow + rely on retry). Never blocks
# falsely on uncertainty -- the reliable guard is the explicit DIES pool.
die_idle() {
  local die="$1"
  [[ "$IDLE_GATE" == "1" ]] || return 0
  command -v npu-smi >/dev/null 2>&1 || return 0
  local used
  used=$(npu-smi info 2>/dev/null | awk -v d="$die" '
    $2==d && $0 ~ /[0-9]+\/[0-9]+/ {
      for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\/[0-9]+$/) { split($i,a,"/"); print a[1]; exit }
    }')
  [[ "$used" =~ ^[0-9]+$ ]] || return 0          # parse failed -> allow
  (( used <= IDLE_MAX_USED_MIB ))
}

run_entry() {
  # vspec = "ALL" (every variant, sequential) OR a single variant name (one
  # dedicated die for just that experiment, e.g. JOBS mode re-running failures).
  local pair="$1" ds="$2" vspec="$3" die="$4"
  local vars="$VARIANTS" tag=""
  if [[ "$vspec" != "ALL" ]]; then vars="$vspec"; tag="__$vspec"; fi
  local log="$LOGDIR/${pair}__${ds}${tag}__npu${die}.log"
  local fails=0 nvar=0 v
  {
    echo "[run] JOB pair=$pair ds=$ds variants=($vars) die=$die start=$(date '+%F %T')"
    for v in $vars; do
      nvar=$((nvar+1))
      echo ""
      echo "================ [$pair / $ds / die$die] variant=$v ================"
      if ! run_variant "$v" "$pair" "$ds" "$die"; then
        echo "[run][WARN] variant=$v FAILED ($pair/$ds)"
        fails=$((fails+1))
      fi
    done
    echo "[run] JOB done pair=$pair ds=$ds die=$die fails=$fails/$nvar end=$(date '+%F %T')"
  } >"$log" 2>&1
  # Wholesale failure (every variant in this job) smells like a busy/bad die ->
  # signal a retry. Partial failure returns 0 (retry would not help -- capacity
  # OOM needs the bf16/LoRA fix, not rescheduling).
  [[ $nvar -gt 0 && $fails -ge $nvar ]] && return 1 || return 0
}

# Build the job queue. Each entry: "pair|ds|vspec|retries".
#   - default: one entry per (pair,dataset), vspec=ALL -> 5 variants SEQUENTIAL on
#     one dedicated die (ours+sft1 share Phase-1, must not race).
#   - JOBS set: one entry per explicit "pair:ds:variant" -> each SINGLE variant on
#     its OWN dedicated die. Use to re-run ONLY specific failed experiments
#     (e.g. just the 8B-teacher distills). Safe for phase1-independent variants
#     (sft/distill/nosft); do NOT split ours/sft1 this way (they each recompute
#     Phase-1 and would race on the shared per-pair phase1 dir).
queue=()
if [[ -n "${JOBS:-}" ]]; then
  for j in $JOBS; do
    IFS=':' read -r jp jd jv <<<"$j"
    [[ -z "$jp" || -z "$jd" || -z "$jv" ]] && { echo "[run][ERR] bad JOB '$j' (want pair:ds:variant)" >&2; exit 1; }
    queue+=("$jp|$jd|$jv|0")
  done
else
  for p in $PAIRS; do for d in $DATASETS; do queue+=("$p|$d|ALL|0"); done; done
fi
echo "[run] jobs=${#queue[@]} die_pool=(${DIE_POOL[*]}) idle_gate=$IDLE_GATE max_retries=$MAX_RETRIES ${JOBS:+(JOBS mode: 1 variant/die)} logs=$LOGDIR ${DRY:+(DRY-RUN)}"
if command -v npu-smi >/dev/null 2>&1; then echo "[run] npu-smi snapshot:"; npu-smi info 2>/dev/null | head -25; fi

declare -A pid_die pid_unit pid_rty
free=("${DIE_POOL[@]}")
parked=()   # dies set aside this round because npu-smi shows them occupied

# Reap finished children: reclaim their die; re-queue a wholesale-failed unit
# (every variant failed -> likely a busy/bad die) up to MAX_RETRIES.
reap() {
  local pid rc die unit rty
  for pid in "${!pid_die[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      die="${pid_die[$pid]}"; unit="${pid_unit[$pid]}"; rty="${pid_rty[$pid]}"
      unset 'pid_die[$pid]' 'pid_unit[$pid]' 'pid_rty[$pid]'
      free+=("$die")
      if [[ $rc -ne 0 && $rty -lt $MAX_RETRIES ]]; then
        echo "[run] RETRY $unit (rc=$rc -> attempt $((rty+2)))"
        queue+=("${unit}|$((rty+1))")
      elif [[ $rc -ne 0 ]]; then
        echo "[run][WARN] $unit gave up (rc=$rc) after $((MAX_RETRIES+1)) tries -- likely CAPACITY OOM (needs bf16/LoRA, not rescheduling)"
      fi
    fi
  done
}

# Non-blocking: pop one free + npu-smi-idle die into ACQUIRED_DIE; return 1 if
# none is idle right now (occupied free dies are parked and rechecked next call).
ACQUIRED_DIE=""
try_acquire_die() {
  local n i cand
  n=${#free[@]}; ACQUIRED_DIE=""
  for ((i=0; i<n; i++)); do
    cand="${free[0]}"; free=("${free[@]:1}")
    if die_idle "$cand"; then ACQUIRED_DIE="$cand"; return 0; else parked+=("$cand"); fi
  done
  if [[ ${#parked[@]} -gt 0 ]]; then free=("${parked[@]}"); parked=(); fi
  return 1
}

# One unified scheduler loop: runs while units remain queued OR children run.
# Retries appended by reap() (even during drain) are picked up because the loop
# condition re-checks the (growing) queue length each iteration.
qi=0
while (( qi < ${#queue[@]} )) || [[ ${#pid_die[@]} -gt 0 ]]; do
  reap
  dispatched=0
  if (( qi < ${#queue[@]} )) && try_acquire_die; then
    entry="${queue[$qi]}"; qi=$((qi+1))
    IFS='|' read -r p d vspec rty <<<"$entry"
    die="$ACQUIRED_DIE"
    echo "[run] dispatch pair=$p ds=$d variant=$vspec -> die$die  (try $((rty+1)), $(date '+%T'))"
    run_entry "$p" "$d" "$vspec" "$die" &
    pid=$!; pid_die[$pid]="$die"; pid_unit[$pid]="$p|$d|$vspec"; pid_rty[$pid]="$rty"
    dispatched=1
  fi
  [[ $dispatched -eq 0 ]] && sleep 5
done
echo "[run] ALL UNITS DONE. logs in $LOGDIR"
