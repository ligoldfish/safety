#!/usr/bin/env bash
# ============================================================
# 单数据集 ours + ours_sft1 重训 + judge (前台串行, 单卡)。
# 不自建 tmux — 在已有 session 内用 send-keys 后台提交:
#   tmux send-keys -t exp 'bash scripts/run_ours_unit.sh wildjailbreak 0 > logs/ours_wjb.log 2>&1 &' C-m
#
# 卡内串行: safety-full --force-rebuild -> judge(ours)
#           -> sft1 (复用 phase1) -> judge(ours_sft1)
# judge 直接接训练后。仅用于 curation 受影响数据集:
#   wildjailbreak(strict) wildguardmix(strict) beavertails_category(minimal)
#
# 用法: bash scripts/run_ours_unit.sh <baseline> <card>
# env: REPO_ROOT WILDGUARD_MODEL PYBIN
# ============================================================
set -uo pipefail

BL="${1:?usage: run_ours_unit.sh <baseline> <card>}"
CARD="${2:?usage: run_ours_unit.sh <baseline> <card>}"
REPO_ROOT="${REPO_ROOT:-/root/safety}"
WILDGUARD_MODEL="${WILDGUARD_MODEL:-models/wildguard}"
PYBIN="${PYBIN:-python}"

cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
TEST="data/processed/eval/${BL}_test.jsonl"

judge () {  # $1 = eval_suite dir
  if [ ! -d "$1" ]; then echo "[$BL][judge] skip missing $1"; return 0; fi
  $PYBIN scripts/22_judge_generations.py \
    --eval-suite-dir "$1" --test-jsonl "$TEST" \
    --judge-model "$WILDGUARD_MODEL" \
    --runtime-backend npu --runtime-device "npu:${CARD}" --merge-summary
}

set -e
echo "[$BL][card $CARD] === safety-full (ours) ==="
$PYBIN scripts/15_run_oneclick.py safety-full --baseline "$BL" --device npu --device-id "$CARD" --force-rebuild
cat "data/processed/safety_full_${BL}/curation_summary.json" 2>/dev/null || true
echo "[$BL][card $CARD] === judge ours ==="
judge "outputs/safety_full_${BL}_npu/phase1/training/eval_suite"

echo "[$BL][card $CARD] === sft1 (ours_sft1, reuse phase1) ==="
$PYBIN scripts/15_run_oneclick.py sft1 --baseline "$BL" --device npu --device-id "$CARD"
echo "[$BL][card $CARD] === judge ours_sft1 ==="
judge "outputs/safety_full_${BL}_npu/phase1/training_sft1/eval_suite"

echo "[$BL][card $CARD] === ALL DONE ==="
