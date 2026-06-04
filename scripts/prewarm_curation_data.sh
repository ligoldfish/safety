#!/usr/bin/env bash
# Serial pre-warm of per-dataset raw safety JSONL + eval test JSONL.
#
# WHY: ours / sft / distill flows all call 19_prepare_safety_data.py with the
# SAME per-dataset config, writing the SAME data/processed/safety/<X>.jsonl.
# Running them in parallel with --force-rebuild races on that file. So we
# materialize each dataset's raw jsonl ONCE here (serial), and the parallel
# experiment units below run WITHOUT --force-rebuild (19_prepare then no-ops
# because the file already exists). The ours flow still rebuilds train_set /
# alignment_set per-dataset inside its own dir, which is conflict-free.
#
# Also builds data/processed/eval/<X>_test.jsonl which the LLM judge joins on.
#
# Usage:
#   bash scripts/prewarm_curation_data.sh
#   BASELINES="wildjailbreak wildguardmix" bash scripts/prewarm_curation_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BASELINES="${BASELINES:-wildjailbreak wildguardmix beavertails_category}"

declare -A SFT_CONFIG=(
  [wildjailbreak]="configs/baseline_sft_qwen35_08b_wildjailbreak_npu.yaml"
  [wildguardmix]="configs/baseline_sft_qwen35_08b_wildguardmix_npu.yaml"
  [beavertails_category]="configs/baseline_sft_qwen35_08b_beavertails_category_npu.yaml"
)

for bl in $BASELINES; do
  cfg="${SFT_CONFIG[$bl]:-}"
  if [ -z "$cfg" ]; then
    echo "[prewarm] no SFT config for baseline=$bl; skip" >&2
    continue
  fi
  echo "=== [prewarm] $bl: materialize raw safety jsonl (19_prepare --force-rebuild) ==="
  python scripts/19_prepare_safety_data.py --config "$cfg" --force-rebuild

  echo "=== [prewarm] $bl: build eval test jsonl (21_build_baseline_eval_jsonls) ==="
  python scripts/21_build_baseline_eval_jsonls.py --baseline "$bl" --force-rebuild
done

echo "=== [prewarm] done. Safe to launch parallel experiment units (no --force-rebuild). ==="
