#!/usr/bin/env bash
# ============================================================
# WildGuard backfill for the exact formal 5-pair x 6-dataset x 5-method matrix.
# Search/sweep runs are excluded structurally rather than by name heuristics.
#
# What it does:
#   1. Reads the same explicit formal target matrix as the CSV collector.
#      FORMAL_ONLY=0 restores broad discovery for diagnostics only.
#   2. Reports expected formal targets whose pan_results.json is absent; it never
#      launches eval or training to manufacture those missing generations.
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
#   bash scripts/judge_backfill.sh                 # formal matrix, dies 0-7
#   DIES="0 1 2 3" bash scripts/judge_backfill.sh  # restrict dies
#   ONLY=qwen3_8b_to_4b bash scripts/judge_backfill.sh   # path filter (regex)
#   DATASETS="pan c5" bash scripts/judge_backfill.sh     # restrict datasets
#   DRY_RUN=1 bash scripts/judge_backfill.sh       # list jobs only
#   FORCE=1 ALLOW_REJUDGE=1 bash scripts/judge_backfill.sh # explicit re-judge
#   FORMAL_ONLY=0 bash scripts/judge_backfill.sh   # broad diagnostic discovery
#   FORMAL_ONLY=0 INCLUDE_SWEEP=1 bash scripts/judge_backfill.sh # include sweeps
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
FORMAL_ONLY="${FORMAL_ONLY:-1}"
INCLUDE_SWEEP="${INCLUDE_SWEEP:-0}"
EXCLUDE_LIVE_SWEEP="${EXCLUDE_LIVE_SWEEP:-0}"
EXCLUDE="${EXCLUDE:-}"    # optional extra regex filter on normalized paths
DATASETS="${DATASETS:-pan safety_tuned_llamas coconot wildguardmix wildjailbreak c5}"

norm_path() { printf '%s' "$1" | tr '\\' '/'; }

is_sweep_path() {
  local p
  p="$(norm_path "$1")"
  [[ -n "$EXCLUDE" && "$p" =~ $EXCLUDE ]] && return 0
  [[ "$INCLUDE_SWEEP" == "1" ]] && return 1
  # This predicate is used only by broad discovery (FORMAL_ONLY=0). Formal mode
  # cannot construct a sweep path in the first place.
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
  local primary fallback
  if [[ "$1" == "pan" ]]; then
    primary="data/processed/pan_test_set.jsonl"
  else
    primary="data/processed/eval/$1_test.jsonl"
  fi
  if [[ -f "$primary" ]]; then
    echo "$primary"
    return
  fi
  fallback="${primary/data\/processed\//data/processed/processed/}"
  if [[ "$fallback" != "$primary" && -f "$fallback" ]]; then
    echo "$fallback"
    return
  fi
  echo "$primary"
}

canon_dataset() {
  case "$1" in
    STL|stl) echo safety_tuned_llamas ;;
    WGM|wgm) echo wildguardmix ;;
    WJB|wjb) echo wildjailbreak ;;
    *)       echo "$1" ;;
  esac
}

dataset_enabled() {
  local want item
  want="$(canon_dataset "$1")"
  for item in $DATASETS; do
    [[ "$(canon_dataset "$item")" == "$want" ]] && return 0
  done
  return 1
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
sys.path.insert(0, "scripts")
from formal_llm_judge_targets import judge_payload_is_complete

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
sys.exit(0 if judge_payload_is_complete(data) else 1)
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
skipped_dataset=0
missing_expected_pan=()

queue_suite() {  # $1 = eval_suite dir, $2 = dataset
  local suite="$1" ds="$2" pr
  local -a prs missing merge_only
  prs=()
  while IFS= read -r pr; do prs+=("$pr"); done < <(find "$suite" -name pan_results.json 2>/dev/null | sort)
  [[ ${#prs[@]} -eq 0 ]] && return

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
    return
  fi

  if [[ ${#missing[@]} -eq ${#prs[@]} && ${#prs[@]} -gt 1 ]]; then
    # Fresh suite: judge all epochs in one process so WildGuard loads once.
    jobs+=("suite|$suite|$ds")
  else
    # Partial backfill: judge ONLY missing epochs; do not re-run existing judge.
    for pr in "${missing[@]}"; do jobs+=("single|$pr|$ds"); done
  fi
  for pr in "${merge_only[@]}"; do jobs+=("merge_single|$pr|$ds"); done
}

queue_single() {  # $1 = standalone pan_results.json, $2 = dataset
  local pr="$1" ds="$2"
  [[ -f "$pr" ]] || return
  if single_needs_judge "$pr"; then
    jobs+=("single|$pr|$ds")
  elif single_needs_merge "$pr"; then
    jobs+=("merge_single|$pr|$ds")
  else skipped=$((skipped+1)); fi
}

if [[ "$FORMAL_ONLY" == "1" ]]; then
  [[ "$INCLUDE_SWEEP" == "1" ]] && echo "[judge][WARN] INCLUDE_SWEEP is ignored while FORMAL_ONLY=1"
  if ! formal_output="$($PYBIN scripts/formal_llm_judge_targets.py --outputs-root outputs --no-header)"; then
    echo "[judge][ERR] failed to construct the formal target matrix"
    exit 1
  fi
  formal_suites=()
  formal_singles=()
  declare -A seen_formal_suites seen_formal_singles
  while IFS=$'\t' read -r pair ds method epoch kind owner pan_path run_root; do
    [[ -z "$pair" ]] && continue
    if ! dataset_enabled "$ds"; then
      skipped_dataset=$((skipped_dataset+1))
      continue
    fi
    [[ -n "$ONLY" && ! "$owner" =~ $ONLY && ! "$pan_path" =~ $ONLY ]] && continue
    [[ -n "$EXCLUDE" && ( "$owner" =~ $EXCLUDE || "$pan_path" =~ $EXCLUDE ) ]] && continue
    if [[ ! -f "$pan_path" ]]; then
      missing_expected_pan+=("$pair|$ds|$method|$epoch|$pan_path")
    fi
    if [[ "$kind" == "suite" ]]; then
      if [[ -z "${seen_formal_suites[$owner]:-}" ]]; then
        formal_suites+=("$owner|$ds")
        seen_formal_suites[$owner]=1
      fi
    elif [[ -z "${seen_formal_singles[$pan_path]:-}" ]]; then
      formal_singles+=("$pan_path|$ds")
      seen_formal_singles[$pan_path]=1
    fi
  done <<<"$formal_output"

  for entry in "${formal_suites[@]}"; do
    IFS='|' read -r suite ds <<<"$entry"
    queue_suite "$suite" "$ds"
  done
  for entry in "${formal_singles[@]}"; do
    IFS='|' read -r pr ds <<<"$entry"
    queue_single "$pr" "$ds"
  done
else
  while IFS= read -r suite; do
    [[ -z "$suite" ]] && continue
    is_sweep_path "$suite" && continue
    [[ -n "$ONLY" && ! "$suite" =~ $ONLY ]] && continue
    ds="$(ds_of_path "$suite")"
    if ! dataset_enabled "$ds"; then
      skipped_dataset=$((skipped_dataset+1))
      continue
    fi
    queue_suite "$suite" "$ds"
  done < <(find outputs -type d -name eval_suite 2>/dev/null | sort)

  while IFS= read -r pr; do
    [[ -z "$pr" ]] && continue
    is_sweep_path "$pr" && continue
    [[ -n "$ONLY" && ! "$pr" =~ $ONLY ]] && continue
    ds="$(ds_of_path "$pr")"
    if ! dataset_enabled "$ds"; then
      skipped_dataset=$((skipped_dataset+1))
      continue
    fi
    queue_single "$pr" "$ds"
  done < <(find outputs -name pan_results.json -not -path "*/eval_suite/*" 2>/dev/null | sort)
fi

# Missing test-jsonl guard. Do not let one missing prompt source block all other
# judge backfill jobs; skip only the affected dataset/path and keep going.
skipped_missing_test_jsonl=0
runnable_jobs=()
declare -A missing_test_seen
for j in "${jobs[@]}"; do
  if [[ "$j" == merge_* ]]; then
    runnable_jobs+=("$j")
    continue
  fi
  ds="${j##*|}"
  tj="$(test_jsonl_of "$ds")"
  if [[ -f "$tj" ]]; then
    runnable_jobs+=("$j")
    continue
  fi
  skipped_missing_test_jsonl=$((skipped_missing_test_jsonl+1))
  if [[ -z "${missing_test_seen[$ds]:-}" ]]; then
    echo "[judge][WARN] test jsonl missing for ds=$ds: $tj"
    if [[ "$ds" == "pan" ]]; then
      echo "[judge][WARN] build PAN data first: $PYBIN scripts/00_prepare_data.py"
    else
      echo "[judge][WARN] build it first: $PYBIN scripts/21_build_baseline_eval_jsonls.py --baseline $ds --force-rebuild"
    fi
    missing_test_seen[$ds]=1
  fi
done
jobs=("${runnable_jobs[@]}")

echo "[judge] pending=${#jobs[@]} already_judged_skipped=$skipped missing_expected_pan=${#missing_expected_pan[@]} skipped_dataset=$skipped_dataset skipped_missing_test_jsonl=$skipped_missing_test_jsonl die_pool=(${DIE_POOL[*]}) datasets=($DATASETS) formal_only=$FORMAL_ONLY include_sweep=$INCLUDE_SWEEP force=$FORCE allow_rejudge=$ALLOW_REJUDGE ${ONLY:+only=$ONLY} ${DRY:+(DRY-RUN)}"
for j in "${jobs[@]}"; do echo "  - $j"; done
for item in "${missing_expected_pan[@]}"; do
  IFS='|' read -r pair ds method epoch pan_path <<<"$item"
  echo "[judge][WARN] cannot judge; pan_results missing: pair=$pair dataset=$ds method=$method epoch=$epoch path=$pan_path"
done
if [[ -n "$DRY" || ${#jobs[@]} -eq 0 ]]; then
  echo "[judge] nothing to run."
  [[ ${#missing_expected_pan[@]} -gt 0 ]] && exit 2
  exit 0
fi

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
    "judge_num_items",
    "judge_num_parsed",
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
[[ $fails -gt 0 ]] && exit 1
[[ ${#missing_expected_pan[@]} -gt 0 ]] && exit 2
exit 0
