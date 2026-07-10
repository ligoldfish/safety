#!/usr/bin/env bash
# ============================================================
# Global WildGuard LLM-judge backfill across ALL experiments (all model pairs +
# legacy qwen35 runs). Ensures every result carries llm_judge_* metrics so
# D:/output/_table.py (judge-only) can consume them.
#
# What it does:
#   1. Discovers every eval_suite/ dir and every standalone (nosft) pan_results.json
#      under outputs/, excluding sweep artifacts by default.
#   2. Infers the dataset from the path (safety_tuned_llamas/coconot/c5/
#      wildjailbreak/wildguardmix/beavertails_category/... else pan) and picks the
#      matching eval test-jsonl (the judge's id->prompt join source -- the SAME
#      jsonl the eval configs point at).
#   3. Skips anything already judged (complete judge_results.json with core
#      llm_judge_* scalars for every epoch's pan_results.json). If judge_results.json
#      is complete but summary.json lacks llm_judge_* scalars, it merge-fixes
#      summary.json without re-running WildGuard. Idempotent -- safe to re-run.
#      Complete judge_results are re-run only with FORCE=1 ALLOW_REJUDGE=1.
#   4. Fans jobs across the die pool (default 0-7), one
#      judge process per die, queue-refill scheduling.
#
# Usage (on the NPU box):
#   conda activate safety310 && source set_env.sh
#   bash scripts/judge_backfill.sh                 # everything, dies 0-7
#   DIES="0 1 2 3" bash scripts/judge_backfill.sh  # restrict dies
#   ONLY=qwen3_8b_to_4b bash scripts/judge_backfill.sh   # path filter (regex)
#   DRY_RUN=1 bash scripts/judge_backfill.sh       # list jobs only
#   FORCE=1 ALLOW_REJUDGE=1 bash scripts/judge_backfill.sh # explicit re-judge
#   INCLUDE_SWEEP=1 bash scripts/judge_backfill.sh # include sweep outputs too
#   EXCLUDE_LIVE_SWEEP=1 bash scripts/judge_backfill.sh # also skip live sweep cells
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
NUM_DIES="${NUM_DIES:-8}"
START_DIE="${START_DIE:-0}"
if [[ -n "${DIES:-}" ]]; then read -r -a DIE_POOL <<<"$DIES"; else
  DIE_POOL=(); for ((i=0; i<NUM_DIES; i++)); do DIE_POOL+=("$((START_DIE+i))"); done
fi
ONLY="${ONLY:-}"          # optional regex filter on paths
FORCE="${FORCE:-0}"
ALLOW_REJUDGE="${ALLOW_REJUDGE:-0}"
DRY="${DRY_RUN:-}"
INCLUDE_SWEEP="${INCLUDE_SWEEP:-0}"
EXCLUDE_LIVE_SWEEP="${EXCLUDE_LIVE_SWEEP:-0}"
EXCLUDE="${EXCLUDE:-}"    # optional extra regex filter on normalized paths

norm_path() { printf '%s' "$1" | tr '\\' '/'; }

is_sweep_path() {
  local p
  p="$(norm_path "$1")"
  [[ -n "$EXCLUDE" && "$p" =~ $EXCLUDE ]] && return 0
  [[ "$INCLUDE_SWEEP" == "1" ]] && return 1
  # Default policy is conservative for formal experiments: only archived sweep
  # roots are excluded. Live safety cells may look like sweep outputs, but are
  # included by default so formal reruns are not missed. Set
  # EXCLUDE_LIVE_SWEEP=1 to drop known run_param_sweep.py live cells too.
  [[ "$p" =~ (^|/)outputs/(sweep|sweep_runs)(/|$) ]] && return 0
  if [[ "$EXCLUDE_LIVE_SWEEP" == "1" ]]; then
    # Live safety sweep cells from run_param_sweep.py:
    # outputs/safety_full_<baseline>_<device>_<axis>_<baseline>_<device>_<die>/...
    # Keep real pair-suffixed runs such as safety_full_c5_npu_qwen3_8b_to_06b.
    [[ "$p" =~ (^|/)outputs/safety_full_[^/]+_(npu|ppu)_[A-Z][0-9]_[^/]+_(npu|ppu)_[0-9]+(/|$) ]] && return 0
  fi
  return 1
}

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
summary_has_llm_judge() {  # $1 = pan_results.json; 0 = summary already has merged llm_judge scalars
  local summary="${1%pan_results.json}summary.json"
  [[ -f "$summary" ]] || return 1
  $PYBIN - "$summary" <<'PY' >/dev/null 2>&1
import json, sys
try:
    pan = json.load(open(sys.argv[1], encoding="utf-8")).get("results", {}).get("pan", {})
except Exception:
    sys.exit(1)
need = ("llm_judge_asr", "llm_judge_over_refusal", "llm_judge_refusal_rate")
sys.exit(0 if all(k in pan for k in need) else 1)
PY
}

judge_results_complete() {  # $1 = pan_results.json; 0 = sibling judge_results.json has core metrics
  local judge="${1%pan_results.json}judge_results.json"
  [[ -f "$judge" ]] || return 1
  $PYBIN - "$judge" <<'PY' >/dev/null 2>&1
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
need = ("llm_judge_asr", "llm_judge_over_refusal", "llm_judge_refusal_rate")
sys.exit(0 if all(k in data for k in need) else 1)
PY
}

single_needs_judge() {  # $1 = pan_results.json
  [[ "$FORCE" == "1" && "$ALLOW_REJUDGE" == "1" ]] && return 0
  ! judge_results_complete "$1"
}
single_needs_merge() {  # $1 = pan_results.json
  [[ "$FORCE" == "1" && "$ALLOW_REJUDGE" == "1" ]] && return 1
  judge_results_complete "$1" || return 1
  ! summary_has_llm_judge "$1"
}

# --- build the job queue: "suite|<dir>|<ds>" / "single|<file>|<ds>" ----------
jobs=()
skipped=0
while IFS= read -r suite; do
  [[ -z "$suite" ]] && continue
  is_sweep_path "$suite" && continue
  [[ -n "$ONLY" && ! "$suite" =~ $ONLY ]] && continue
  prs=()
  while IFS= read -r pr; do prs+=("$pr"); done < <(find "$suite" -name pan_results.json 2>/dev/null | sort)
  [[ ${#prs[@]} -eq 0 ]] && continue

  missing=()
  merge_only=()
  for pr in "${prs[@]}"; do
    if single_needs_judge "$pr"; then
      missing+=("$pr")
    elif single_needs_merge "$pr"; then
      merge_only+=("$pr")
    fi
  done

  if [[ ${#missing[@]} -eq 0 && ${#merge_only[@]} -eq 0 ]]; then
    skipped=$((skipped+1))
    continue
  fi

  ds="$(ds_of_path "$suite")"
  if [[ ${#missing[@]} -eq ${#prs[@]} && ${#prs[@]} -gt 1 ]]; then
    # Fresh suite: judge all epochs in one process so WildGuard loads once.
    jobs+=("suite|$suite|$(ds_of_path "$suite")")
  else
    # Partial backfill: judge ONLY missing epochs; do not re-run existing judge.
    for pr in "${missing[@]}"; do jobs+=("single|$pr|$ds"); done
  fi
  for pr in "${merge_only[@]}"; do jobs+=("merge_single|$pr|$ds"); done
done < <(find outputs -type d -name eval_suite 2>/dev/null | sort)

while IFS= read -r pr; do
  [[ -z "$pr" ]] && continue
  is_sweep_path "$pr" && continue
  [[ -n "$ONLY" && ! "$pr" =~ $ONLY ]] && continue
  if single_needs_judge "$pr"; then
    jobs+=("single|$pr|$(ds_of_path "$pr")")
  elif single_needs_merge "$pr"; then
    jobs+=("merge_single|$pr|$(ds_of_path "$pr")")
  else skipped=$((skipped+1)); fi
done < <(find outputs -name pan_results.json -not -path "*/eval_suite/*" 2>/dev/null | sort)

echo "[judge] pending=${#jobs[@]} already_judged_skipped=$skipped die_pool=(${DIE_POOL[*]}) include_sweep=$INCLUDE_SWEEP exclude_live_sweep=$EXCLUDE_LIVE_SWEEP force=$FORCE allow_rejudge=$ALLOW_REJUDGE ${ONLY:+only=$ONLY} ${DRY:+(DRY-RUN)}"
for j in "${jobs[@]}"; do echo "  - $j"; done
[[ -n "$DRY" || ${#jobs[@]} -eq 0 ]] && { echo "[judge] nothing to run."; exit 0; }

# missing test-jsonl guard (fail early, per unique ds)
for j in "${jobs[@]}"; do
  ds="${j##*|}"; tj="$(test_jsonl_of "$ds")"
  [[ "$j" == merge_* ]] && continue
  [[ -f "$tj" ]] || { echo "[judge][ERR] test jsonl missing for ds=$ds: $tj (build it first via 21_build_baseline_eval_jsonls / 19_prepare)"; exit 1; }
done

merge_existing_one() {  # $1 = pan_results.json with sibling judge_results.json
  $PYBIN - "$1" <<'PY'
import json, sys
from pathlib import Path

pan_results = Path(sys.argv[1])
judge_path = pan_results.with_name("judge_results.json")
summary_path = pan_results.with_name("summary.json")
if not judge_path.exists() or not summary_path.exists():
    sys.exit(0)
judge = json.loads(judge_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
pan = summary.setdefault("results", {}).setdefault("pan", {})
for key in (
    "llm_judge_asr",
    "llm_judge_over_refusal",
    "llm_judge_refusal_rate",
    "judge_keyword_agreement",
    "judge_cohen_kappa",
    "judge_parse_rate",
    "judge_num_harmful_scored",
    "judge_num_harmless_scored",
):
    if key in judge:
        pan[key] = judge[key]
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

merge_existing_suite() {  # $1 = eval_suite dir
  local pr
  while IFS= read -r pr; do
    merge_existing_one "$pr"
  done < <(find "$1" -name pan_results.json 2>/dev/null | sort)
}

judge_single_on_die() {  # $1 = pan_results.json, $2 = dataset, $3 = die
  local tj
  tj="$(test_jsonl_of "$2")"
  $PYBIN scripts/22_judge_generations.py \
    --pan-results "$1" --test-jsonl "$tj" \
    --judge-model "$WILDGUARD_MODEL" \
    --runtime-backend npu --runtime-device "npu:$3" --merge-summary
}

judge_suite_on_die() {  # $1 = eval_suite dir, $2 = dataset, $3 = die
  local tj
  tj="$(test_jsonl_of "$2")"
  $PYBIN scripts/22_judge_generations.py \
    --eval-suite-dir "$1" --test-jsonl "$tj" \
    --judge-model "$WILDGUARD_MODEL" \
    --runtime-backend npu --runtime-device "npu:$3" --merge-summary
}

run_job() {  # $1 = job entry, $2 = die
  local kind path ds tag log pr
  local -a prs missing merge_only
  IFS='|' read -r kind path ds <<<"$1"
  tag="$(echo "$path" | sed 's#^outputs/##; s#[/ ]#_#g')"
  log="$LOGDIR/${tag}__npu$2.log"
  {
    echo "[judge] job=$kind path=$path ds=$ds die=npu:$2"
    if [[ "$kind" == "merge_suite" ]]; then
      merge_existing_suite "$path"
    elif [[ "$kind" == "merge_single" ]]; then
      if single_needs_merge "$path"; then
        merge_existing_one "$path"
      else
        echo "[judge] skip merge; summary already has llm_judge metrics or judge is incomplete: $path"
      fi
    elif [[ "$kind" == "suite" ]]; then
      # Re-check right before loading WildGuard so a stale queue never re-judges
      # epochs that were completed by an earlier run.
      prs=()
      missing=()
      merge_only=()
      while IFS= read -r pr; do prs+=("$pr"); done < <(find "$path" -name pan_results.json 2>/dev/null | sort)
      for pr in "${prs[@]}"; do
        if single_needs_judge "$pr"; then
          missing+=("$pr")
        elif single_needs_merge "$pr"; then
          merge_only+=("$pr")
        fi
      done
      if [[ ${#missing[@]} -eq 0 ]]; then
        echo "[judge] skip suite; all pan_results already have complete judge_results: $path"
      elif [[ ${#missing[@]} -eq ${#prs[@]} && ${#prs[@]} -gt 1 ]]; then
        judge_suite_on_die "$path" "$ds" "$2"
      else
        for pr in "${missing[@]}"; do
          if single_needs_judge "$pr"; then
            judge_single_on_die "$pr" "$ds" "$2"
          else
            echo "[judge] skip single after re-check: $pr"
          fi
        done
      fi
      for pr in "${merge_only[@]}"; do
        single_needs_merge "$pr" && merge_existing_one "$pr"
      done
    else
      if single_needs_judge "$path"; then
        judge_single_on_die "$path" "$ds" "$2"
      elif single_needs_merge "$path"; then
        merge_existing_one "$path"
      else
        echo "[judge] skip single; judge_results already complete and merged: $path"
      fi
    fi
  } >"$log" 2>&1
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
