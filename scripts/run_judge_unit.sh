#!/usr/bin/env bash
# ============================================================
# 单 baseline 的 LLM judge backfill (前台串行, 单卡)。
# 不自建 tmux — 在已有 session 内用 send-keys 后台提交:
#   tmux send-keys -t exp 'bash scripts/run_judge_unit.sh pan 7 > logs/judge_pan.log 2>&1 &' C-m
#
# 自动发现该 baseline 的 eval_suite + nosft 单文件 pan_results.json, 逐个 judge。
#   - strict/minimal (wildjailbreak/wildguardmix/beavertails_category):
#       只判 nosft/sft/distill;  ours+ours_sft1 由 run_ours_unit.sh 训练后判 (这里跳过)。
#   - off-mode (pan/safety_tuned_llamas/coconot):
#       判全部 5 变体 (nosft/sft/distill/ours/ours_sft1) — ours 不重训, 用已有 generation。
#   含 nosft (--pan-results 单文件); 其余 --eval-suite-dir (一次判所有 epoch)。
#
# 用法: bash scripts/run_judge_unit.sh <baseline> <card>
# env: REPO_ROOT WILDGUARD_MODEL PYBIN
# ============================================================
set -uo pipefail

BL="${1:?usage: run_judge_unit.sh <baseline> <card>}"
CARD="${2:?usage: run_judge_unit.sh <baseline> <card>}"
REPO_ROOT="${REPO_ROOT:-/root/safety}"
WILDGUARD_MODEL="${WILDGUARD_MODEL:-models/wildguard}"
PYBIN="${PYBIN:-python}"

cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512

# test jsonl + 路径匹配模式 (PAN 目录名不含 'pan', 需显式模式)
case "$BL" in
  pan)
    TEST="data/processed/pan_test_set.jsonl"
    PAT='qwen35_9b_to_08b_phase1_npu/training|baselines/sft_qwen35_08b_npu/|baselines/distill_qwen35_9b_to_08b_npu/|baselines/no_sft_qwen35_9b_npu/'
    JUDGE_OURS=1 ;;
  safety_tuned_llamas|coconot)
    TEST="data/processed/eval/${BL}_test.jsonl"
    PAT="$BL"
    JUDGE_OURS=1 ;;            # off-mode: 也判 ours/ours_sft1
  wildjailbreak|wildguardmix|beavertails_category)
    TEST="data/processed/eval/${BL}_test.jsonl"
    PAT="$BL"
    JUDGE_OURS=0 ;;           # strict/minimal: ours/sft1 由 run_ours_unit 判, 这里跳过
  *)
    echo "unknown baseline: $BL"; exit 2 ;;
esac

if [ ! -f "$TEST" ]; then echo "[$BL][judge] test jsonl missing: $TEST"; exit 1; fi

judge_suite () {  # $1 = eval_suite dir
  $PYBIN scripts/22_judge_generations.py \
    --eval-suite-dir "$1" --test-jsonl "$TEST" \
    --judge-model "$WILDGUARD_MODEL" \
    --runtime-backend npu --runtime-device "npu:${CARD}" --merge-summary \
    || echo "[$BL][judge] FAILED suite $1"
}
judge_file () {   # $1 = pan_results.json (nosft)
  $PYBIN scripts/22_judge_generations.py \
    --pan-results "$1" --test-jsonl "$TEST" \
    --judge-model "$WILDGUARD_MODEL" \
    --runtime-backend npu --runtime-device "npu:${CARD}" --merge-summary \
    || echo "[$BL][judge] FAILED single $1"
}

echo "[$BL][card $CARD] === judge backfill (JUDGE_OURS=$JUDGE_OURS) ==="

# 1) eval_suite 目录 (sft/distill + off-mode 的 ours/ours_sft1)
while IFS= read -r suite; do
  [ -z "$suite" ] && continue
  if [ "$JUDGE_OURS" = "0" ]; then
    # 跳过脚本1 负责的 ours/ours_sft1 (safety_full_<BL>/phase1/training[_sft1])
    case "$suite" in *safety_full_${BL}_npu/phase1/training*) echo "[$BL][judge] skip (run_ours_unit) $suite"; continue ;; esac
  fi
  echo "[$BL][judge] suite: $suite"
  judge_suite "$suite"
done < <(find outputs -type d -name eval_suite 2>/dev/null | grep -E "$PAT" | sort)

# 2) nosft 单文件 (pan_results.json 不在 eval_suite/ 下)
while IFS= read -r pr; do
  [ -z "$pr" ] && continue
  echo "[$BL][judge] nosft: $pr"
  judge_file "$pr"
done < <(find outputs -name pan_results.json 2>/dev/null | grep -v '/eval_suite/' | grep -E "$PAT" | sort)

echo "[$BL][card $CARD] === judge DONE ==="
