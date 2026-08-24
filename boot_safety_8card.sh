#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

APP_ROOT="/home/work/user-job-dir/app"
SRC_ROOT="${SAFETY_SOURCE_ROOT:-$APP_ROOT/safety-src-886760f}"
MODEL_ROOT="${SAFETY_MODEL_ROOT:-/opt/dpcvol/models/safetytransfer}"
DATA_ROOT="${SAFETY_DATA_ROOT:-/opt/dpcvol/datasets/safetytransfer}"
OUTPUT_ROOT="${SAFETY_OUTPUT_ROOT:-$DATA_ROOT/ablation-outputs/iclr-886760f}"
ACTIVATE="${SAFETY_ACTIVATE:-$MODEL_ROOT/_runtime_assets/activate-safety-ablation.sh}"
ASSET_MANIFEST="${SAFETY_ASSET_MANIFEST:-$SRC_ROOT/configs/ablations/assets.modelmate.template.json}"
CAMPAIGN_SCRIPT="$SRC_ROOT/scripts/37_modelmate_ablation_campaign.py"
POOL_SCRIPT="$SRC_ROOT/scripts/35_modelmate_8card_pool.py"
GATE_SCRIPT="$SRC_ROOT/scripts/36_modelmate_ablation_final_gate.py"

for path in \
  "$SRC_ROOT" \
  "$ACTIVATE" \
  "$ASSET_MANIFEST" \
  "$CAMPAIGN_SCRIPT" \
  "$POOL_SCRIPT" \
  "$GATE_SCRIPT"
do
  if [ ! -e "$path" ]; then
    echo "[FAIL] missing required runtime path: $path" >&2
    exit 20
  fi
done

# shellcheck disable=SC1090
source "$ACTIVATE"

JOB_PY="${SAFETY_JOB_PYTHON:-$(command -v python3)}"
if [ ! -x "$JOB_PY" ]; then
  echo "[FAIL] canonical Python is not executable: $JOB_PY" >&2
  exit 21
fi

export PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAFETY_MODEL_ROOT="$MODEL_ROOT"
export SAFETY_DATA_ROOT="$DATA_ROOT"
export SAFETY_OUTPUT_ROOT="$OUTPUT_ROOT"
export HF_HOME="$DATA_ROOT/_hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$JOB_PY" -u "$CAMPAIGN_SCRIPT" \
  --model-root "$MODEL_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --asset-manifest "$ASSET_MANIFEST" \
  --devices "${SAFETY_POOL_DEVICES:-8}" \
  --logical-shards "${SAFETY_LOGICAL_SHARDS:-16}" \
  --launch-stagger-seconds "${SAFETY_LAUNCH_STAGGER_SECONDS:-15}" \
  "$@"
