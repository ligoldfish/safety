#!/usr/bin/env bash
# ============================================================
# 单 baseline 的 sft 或 distill 重训 + LLM judge (前台串行, 单卡)。
# sft / distill 拆两条各占一卡 -> 并行。judge(WildGuard LLM)直接接训练后。
# 不自建 tmux — 在已有 session(rerun)内 send-keys 后台提交:
#   tmux send-keys -t rerun 'bash scripts/run_base_unit.sh wildguardmix sft 2     > logs/base_wgm_sft.log 2>&1 &' C-m
#   tmux send-keys -t rerun 'bash scripts/run_base_unit.sh wildguardmix distill 3 > logs/base_wgm_distill.log 2>&1 &' C-m
#
# 不带 --force-rebuild: 复用 run_ours_unit.sh 已重建的共享源
#   data/processed/safety/<bl>_20k_train.jsonl。务必在该数据集 ours-unit 建好数据后再跑。
# STL(safety_tuned_llamas) sft: judge 后自动 select_best_epoch_by_hr。
#
# 用法: bash scripts/run_base_unit.sh <baseline> <sft|distill> <card>
# env: REPO_ROOT WILDGUARD_MODEL PYBIN
# ============================================================
set -uo pipefail

BL="${1:?usage: run_base_unit.sh <baseline> <sft|distill> <card>}"
KIND="${2:?usage: run_base_unit.sh <baseline> <sft|distill> <card>}"
CARD="${3:?usage: run_base_unit.sh <baseline> <sft|distill> <card>}"
REPO_ROOT="${REPO_ROOT:-/root/safety}"
WILDGUARD_MODEL="${WILDGUARD_MODEL:-models/wildguard}"
PYBIN="${PYBIN:-python}"

cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512

if [ "$BL" = "pan" ]; then
  TEST="data/processed/pan_test_set.jsonl"
else
  TEST="data/processed/eval/${BL}_test.jsonl"
fi

# (训练命令, eval_suite 目录) 按 (baseline, kind) 选择
if [ "$BL" = "pan" ]; then
  if [ "$KIND" = "sft" ]; then
    TRAIN=("$PYBIN" scripts/15_run_oneclick.py sft --model 0.8b --baseline pan --device npu --device-id "$CARD")
    SUITE="outputs/baselines/sft_qwen35_08b_npu/eval_suite"
  else
    TRAIN=("$PYBIN" scripts/15_run_oneclick.py distill --baseline pan --device npu --device-id "$CARD")
    SUITE="outputs/baselines/distill_qwen35_9b_to_08b_npu/eval_suite"
  fi
else
  if [ "$KIND" = "sft" ]; then
    TRAIN=("$PYBIN" scripts/15_run_oneclick.py safety-sft --baseline "$BL" --device npu --device-id "$CARD")
    SUITE="outputs/baselines/sft_qwen35_08b_${BL}_npu/eval_suite"
  else
    TRAIN=("$PYBIN" scripts/15_run_oneclick.py safety-distill --baseline "$BL" --device npu --device-id "$CARD")
    SUITE="outputs/baselines/distill_qwen35_9b_to_08b_${BL}_npu/eval_suite"
  fi
fi

if [ ! -f "$TEST" ]; then echo "[$BL][$KIND] test jsonl missing: $TEST"; exit 1; fi

set -e
echo "[$BL][$KIND][card $CARD] === train ==="
"${TRAIN[@]}"

echo "[$BL][$KIND][card $CARD] === LLM judge ($SUITE) ==="
$PYBIN scripts/22_judge_generations.py \
  --eval-suite-dir "$SUITE" --test-jsonl "$TEST" \
  --judge-model "$WILDGUARD_MODEL" \
  --runtime-backend npu --runtime-device "npu:${CARD}" --merge-summary

# STL sft: judge 完后按 judge HR 选部署 epoch
if [ "$BL" = "safety_tuned_llamas" ] && [ "$KIND" = "sft" ]; then
  echo "[$BL][$KIND] === select best epoch by HR (judge) ==="
  $PYBIN scripts/select_best_epoch_by_hr.py \
    "outputs/baselines/sft_qwen35_08b_${BL}_npu" --metric judge --write
fi

echo "[$BL][$KIND][card $CARD] === DONE ==="
