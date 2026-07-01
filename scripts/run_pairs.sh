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
  local variant="$1" pair="$2" ds="$3" dies="$4"   # dies = "12" or "12,13" (multi-card)
  # Expose the acquired die(s) via ASCEND_RT_VISIBLE_DEVICES; the launcher inherits
  # it (_build_env_overrides) and always sets runtime_device=npu:0, so physical dies
  # are chosen HERE. Pass --device-id 0 (device_id only picks the die when ASCEND is
  # NOT pre-set; we always pre-set it). Two dies + a device_map="auto" config ->
  # HF/accelerate shards the model across them (naive model-parallel; fp32/AdamW
  # unchanged). One die -> the original single-card path.
  local common=(--device npu --device-id 0 --pair "$pair" --baseline "$ds")
  local envp=(env "ASCEND_RT_VISIBLE_DEVICES=$dies")
  case "$variant" in
    ours)    "${envp[@]}" python scripts/15_run_oneclick.py full    "${common[@]}" $DRY ;;
    sft1)    "${envp[@]}" python scripts/15_run_oneclick.py sft1    "${common[@]}" $DRY ;;
    sft)     "${envp[@]}" python scripts/15_run_oneclick.py sft     "${common[@]}" $DRY ;;
    distill) "${envp[@]}" python scripts/15_run_oneclick.py distill "${common[@]}" $DRY ;;
    nosft)   "${envp[@]}" python scripts/15_run_oneclick.py nosft --role student "${common[@]}" $DRY ;;
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
  # vspec = "ALL" (every variant, sequential) OR a single variant name.
  # dies = comma-joined die list ("12" or "12,13"); a job with 2 dies + a
  # device_map="auto" config runs multi-card (naive model-parallel).
  local pair="$1" ds="$2" vspec="$3" dies="$4"
  local vars="$VARIANTS" tag=""
  if [[ "$vspec" != "ALL" ]]; then vars="$vspec"; tag="__$vspec"; fi
  local log="$LOGDIR/${pair}__${ds}${tag}__npu${dies//,/_}.log"
  local fails=0 nvar=0 v
  {
    echo "[run] JOB pair=$pair ds=$ds variants=($vars) dies=$dies start=$(date '+%F %T')"
    for v in $vars; do
      nvar=$((nvar+1))
      echo ""
      echo "================ [$pair / $ds / dies$dies] variant=$v ================"
      if ! run_variant "$v" "$pair" "$ds" "$dies"; then
        echo "[run][WARN] variant=$v FAILED ($pair/$ds)"
        fails=$((fails+1))
      fi
    done
    echo "[run] JOB done pair=$pair ds=$ds dies=$dies fails=$fails/$nvar end=$(date '+%F %T')"
  } >"$log" 2>&1
  # Wholesale failure (every variant in this job) -> signal a retry (busy/bad die).
  # Partial failure returns 0 (retry would not help).
  [[ $nvar -gt 0 && $fails -ge $nvar ]] && return 1 || return 0
}

# Build the job queue. Each entry: "pair|ds|vspec|ndie|retries".
#   - default: one entry per (pair,dataset), vspec=ALL, ndie=1 -> 5 variants
#     SEQUENTIAL on one dedicated die (ours+sft1 share Phase-1, must not race).
#   - JOBS set: one entry per explicit "pair:ds:variant[:N]" -> that single variant
#     on its OWN N dedicated die(s). Use N=2 for the 4B full-finetune jobs
#     (qwen3_8b_to_4b sft/distill) whose config sets device_map="auto" so the model
#     shards across the 2 cards. N defaults to 1. Do NOT split ours/sft1 via JOBS
#     (they recompute Phase-1 and would race the shared per-pair phase1 dir).
queue=()
if [[ -n "${JOBS:-}" ]]; then
  for j in $JOBS; do
    IFS=':' read -r jp jd jv jn <<<"$j"
    [[ -z "$jp" || -z "$jd" || -z "$jv" ]] && { echo "[run][ERR] bad JOB '$j' (want pair:ds:variant[:N])" >&2; exit 1; }
    jn="${jn:-1}"
    queue+=("$jp|$jd|$jv|$jn|0")
  done
else
  for p in $PAIRS; do for d in $DATASETS; do queue+=("$p|$d|ALL|1|0"); done; done
fi
echo "[run] jobs=${#queue[@]} die_pool=(${DIE_POOL[*]}) idle_gate=$IDLE_GATE max_retries=$MAX_RETRIES ${JOBS:+(JOBS mode)} logs=$LOGDIR ${DRY:+(DRY-RUN)}"
if command -v npu-smi >/dev/null 2>&1; then echo "[run] npu-smi snapshot:"; npu-smi info 2>/dev/null | head -25; fi

declare -A pid_dies pid_unit pid_rty
free=("${DIE_POOL[@]}")
parked=()   # dies set aside this round because npu-smi shows them occupied

# Reap finished children: reclaim ALL their dies; re-queue a wholesale-failed job.
reap() {
  local pid rc dies unit rty d
  for pid in "${!pid_dies[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      dies="${pid_dies[$pid]}"; unit="${pid_unit[$pid]}"; rty="${pid_rty[$pid]}"
      unset 'pid_dies[$pid]' 'pid_unit[$pid]' 'pid_rty[$pid]'
      for d in $dies; do free+=("$d"); done
      if [[ $rc -ne 0 && $rty -lt $MAX_RETRIES ]]; then
        echo "[run] RETRY $unit (rc=$rc -> attempt $((rty+2)))"
        queue+=("${unit}|$((rty+1))")
      elif [[ $rc -ne 0 ]]; then
        echo "[run][WARN] $unit gave up (rc=$rc) after $((MAX_RETRIES+1)) tries -- likely CAPACITY OOM"
      fi
    fi
  done
}

# Non-blocking: pop N free + npu-smi-idle dies into ACQUIRED_DIES (space-joined);
# return 1 if fewer than N are idle right now (grabbed dies are returned).
ACQUIRED_DIES=""
try_acquire_dies() {
  local n="$1" got=() i cand avail
  ACQUIRED_DIES=""
  avail=${#free[@]}
  for ((i=0; i<avail && ${#got[@]}<n; i++)); do
    cand="${free[0]}"; free=("${free[@]:1}")
    if die_idle "$cand"; then got+=("$cand"); else parked+=("$cand"); fi
  done
  if [[ ${#got[@]} -eq $n ]]; then
    ACQUIRED_DIES="${got[*]}"; return 0
  fi
  [[ ${#got[@]} -gt 0 ]] && free+=("${got[@]}")
  [[ ${#parked[@]} -gt 0 ]] && { free+=("${parked[@]}"); parked=(); }
  return 1
}

# Unified scheduler loop: runs while jobs remain queued OR children run. A job is
# dispatched only when its ndie idle dies are simultaneously free.
qi=0
while (( qi < ${#queue[@]} )) || [[ ${#pid_dies[@]} -gt 0 ]]; do
  reap
  dispatched=0
  if (( qi < ${#queue[@]} )); then
    entry="${queue[$qi]}"
    IFS='|' read -r p d vspec ndie rty <<<"$entry"
    if try_acquire_dies "$ndie"; then
      qi=$((qi+1))
      dies_csv="${ACQUIRED_DIES// /,}"
      echo "[run] dispatch pair=$p ds=$d variant=$vspec ndie=$ndie -> dies=$dies_csv  (try $((rty+1)), $(date '+%T'))"
      run_entry "$p" "$d" "$vspec" "$dies_csv" &
      pid=$!; pid_dies[$pid]="$ACQUIRED_DIES"; pid_unit[$pid]="$p|$d|$vspec|$ndie"; pid_rty[$pid]="$rty"
      dispatched=1
    fi
  fi
  [[ $dispatched -eq 0 ]] && sleep 5
done
echo "[run] ALL UNITS DONE. logs in $LOGDIR"
