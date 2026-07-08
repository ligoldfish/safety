#!/usr/bin/env bash
# ============================================================
# Global WildGuard LLM-judge backfill across ALL experiments (all model pairs +
# legacy qwen35 runs). Ensures every result carries llm_judge_* metrics so
# D:/output/_table.py (judge-only) can consume them.
#
# What it does:
#   1. Discovers every eval_suite/ dir and every standalone (nosft) pan_results.json
#      under outputs/.
#   2. Infers the dataset from the path (safety_tuned_llamas/coconot/c5/
#      wildjailbreak/wildguardmix/beavertails_category/... else pan) and picks the
#      matching eval test-jsonl (the judge's id->prompt join source -- the SAME
#      jsonl the eval configs point at).
#   3. Skips anything already judged (judge_results.json present for every
#      epoch's pan_results.json). Idempotent -- safe to re-run. FORCE=1 re-judges.
#   4. Fans jobs across the die pool (default 0-11; dies 12-15 reserved), one
#      judge process per die, queue-refill scheduling.
#
# Usage (on the NPU box):
#   conda activate safety310 && source set_env.sh
#   bash scripts/judge_backfill.sh                 # everything, dies 0-11
#   DIES="0 1 2 3" bash scripts/judge_backfill.sh  # restrict dies
#   ONLY=qwen3_8b_to_4b bash scripts/judge_backfill.sh   # path filter (regex)
#   DRY_RUN=1 bash scripts/judge_backfill.sh       # list jobs only
#   FORCE=1  bash scripts/judge_backfill.sh        # re-judge even if judged
# env: REPO_ROOT WILDGUARD_MODEL PYBIN LOGDIR
# ============================================================
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/root/safety}"
WILDGUARD_MODEL="${WILDGUARD_MODEL:-models/wildguard}"
PYBIN="${PYBIN:-python}"
cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512

LOGDIR="${LOGDIR:-$REPO_ROOT/outputs/judge_backfill_logs}"
mkdir -p "$LOGDIR"
NUM_DIES="${NUM_DIES:-12}"
START_DIE="${START_DIE:-0}"
if [[ -n "${DIES:-}" ]]; then read -r -a DIE_POOL <<<"$DIES"; else
  DIE_POOL=(); for ((i=0; i<NUM_DIES; i++)); do DIE_POOL+=("$((START_DIE+i))"); done
fi
ONLY="${ONLY:-}"          # optional regex filter on paths
FORCE="${FORCE:-0}"
DRY="${DRY_RUN:-}"

# --- dataset inference from a path -----------------------------------------
# Order matters: beavertails_category before beavertails; explicit tokens else pan.
ds_of_path() {
  local p="$1"
  case "$p" in
    *beavertails_category*) echo beavertails_category ;;
    *beavertails*)          echo beavertails ;;
    *safety_tuned_llamas*)  echo safety_tuned_llamas ;;
    *tulu3_safety*)         echo tulu3_safety ;;
    *hh_rlhf*|*hhrlhf*)     echo hh_rlhf ;;
    *coconot*)              echo coconot ;;
    *wildjailbreak*)        echo wildjailbreak ;;
    *wildguardmix*)         echo wildguardmix ;;
    *_c5_*|*/c5_*|*c5/*)    echo c5 ;;
    *)                      echo pan ;;
  esac
}
test_jsonl_of() {  # $1 = dataset
  if [[ "$1" == "pan" ]]; then echo "data/processed/pan_test_set.jsonl"
  else echo "data/processed/eval/$1_test.jsonl"; fi
}

# --- already-judged checks ---------------------------------------------------
suite_needs_judge() {  # $1 = eval_suite dir; 0 = needs judging
  [[ "$FORCE" == "1" ]] && return 0
  local pr n=0
  while IFS= read -r pr; do
    [[ -f "${pr%pan_results.json}judge_results.json" ]] || { n=1; break; }
  done < <(find "$1" -name pan_results.json 2>/dev/null)
  [[ $n -eq 1 ]]
}
single_needs_judge() {  # $1 = pan_results.json
  [[ "$FORCE" == "1" ]] && return 0
  [[ ! -f "${1%pan_results.json}judge_results.json" ]]
}

# --- build the job queue: "suite|<dir>|<ds>" / "single|<file>|<ds>" ----------
jobs=()
skipped=0
while IFS= read -r suite; do
  [[ -z "$suite" ]] && continue
  [[ -n "$ONLY" && ! "$suite" =~ $ONLY ]] && continue
  # only suites that actually contain generations
  [[ -z "$(find "$suite" -name pan_results.json -print -quit 2>/dev/null)" ]] && continue
  if suite_needs_judge "$suite"; then
    jobs+=("suite|$suite|$(ds_of_path "$suite")")
  else skipped=$((skipped+1)); fi
done < <(find outputs -type d -name eval_suite 2>/dev/null | sort)

while IFS= read -r pr; do
  [[ -z "$pr" ]] && continue
  [[ -n "$ONLY" && ! "$pr" =~ $ONLY ]] && continue
  if single_needs_judge "$pr"; then
    jobs+=("single|$pr|$(ds_of_path "$pr")")
  else skipped=$((skipped+1)); fi
done < <(find outputs -name pan_results.json -not -path "*/eval_suite/*" 2>/dev/null | sort)

echo "[judge] pending=${#jobs[@]} already_judged_skipped=$skipped die_pool=(${DIE_POOL[*]}) ${ONLY:+only=$ONLY} ${DRY:+(DRY-RUN)}"
for j in "${jobs[@]}"; do echo "  - $j"; done
[[ -n "$DRY" || ${#jobs[@]} -eq 0 ]] && { echo "[judge] nothing to run."; exit 0; }

# missing test-jsonl guard (fail early, per unique ds)
for j in "${jobs[@]}"; do
  ds="${j##*|}"; tj="$(test_jsonl_of "$ds")"
  [[ -f "$tj" ]] || { echo "[judge][ERR] test jsonl missing for ds=$ds: $tj (build it first via 21_build_baseline_eval_jsonls / 19_prepare)"; exit 1; }
done

run_job() {  # $1 = job entry, $2 = die
  local kind path ds tj tag log
  IFS='|' read -r kind path ds <<<"$1"
  tj="$(test_jsonl_of "$ds")"
  tag="$(echo "$path" | sed 's#^outputs/##; s#[/ ]#_#g')"
  log="$LOGDIR/${tag}__npu$2.log"
  if [[ "$kind" == "suite" ]]; then
    $PYBIN scripts/22_judge_generations.py \
      --eval-suite-dir "$path" --test-jsonl "$tj" \
      --judge-model "$WILDGUARD_MODEL" \
      --runtime-backend npu --runtime-device "npu:$2" --merge-summary \
      >"$log" 2>&1
  else
    $PYBIN scripts/22_judge_generations.py \
      --pan-results "$path" --test-jsonl "$tj" \
      --judge-model "$WILDGUARD_MODEL" \
      --runtime-backend npu --runtime-device "npu:$2" --merge-summary \
      >"$log" 2>&1
  fi
}

# --- fan across dies (queue refill, one judge proc per die) ------------------
declare -A pid_die pid_job
free=("${DIE_POOL[@]}")
qi=0
fails=0
while (( qi < ${#jobs[@]} )) || [[ ${#pid_die[@]} -gt 0 ]]; do
  for pid in "${!pid_die[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      [[ $rc -ne 0 ]] && { echo "[judge][WARN] FAILED (rc=$rc): ${pid_job[$pid]}"; fails=$((fails+1)); }
      free+=("${pid_die[$pid]}"); unset 'pid_die[$pid]' 'pid_job[$pid]'
    fi
  done
  if (( qi < ${#jobs[@]} )) && [[ ${#free[@]} -gt 0 ]]; then
    die="${free[0]}"; free=("${free[@]:1}")
    job="${jobs[$qi]}"; qi=$((qi+1))
    echo "[judge] dispatch [$qi/${#jobs[@]}] die$die :: $job"
    run_job "$job" "$die" &
    pid=$!; pid_die[$pid]="$die"; pid_job[$pid]="$job"
  else
    sleep 5
  fi
done
echo "[judge] ALL DONE. failures=$fails logs=$LOGDIR"
echo "[judge] verify: find outputs -name judge_results.json | wc -l ; re-run this script -- pending should be 0."
