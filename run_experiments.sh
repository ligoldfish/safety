#!/usr/bin/env bash
# Keep this script in LF format so it runs correctly under Linux bash.
set -e
set -u
set -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/safety}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

DEVICE=""
VISIBLE_DEVICES=""
EXPERIMENTS=()
EFFECTIVE_DEVICE_ID="0"
OPENCOMPASS_DIR="${OPENCOMPASS_DIR:-}"
OPENCOMPASS_DATASETS="${OPENCOMPASS_DATASETS:-mmlu_gen gsm8k_gen humaneval_gen mbpp_gen}"
SKIP_OPENCOMPASS="0"

# Auto-pick the co-located OpenCompass clone under external/opencompass when present.
if [[ -z "${OPENCOMPASS_DIR}" && -f "${PROJECT_ROOT}/external/opencompass/run.py" ]]; then
  OPENCOMPASS_DIR="${PROJECT_ROOT}/external/opencompass"
fi

usage() {
  cat <<'EOF'
Usage:
  bash /root/run_experiments.sh --device <npu|tpu> --visible-devices <ids> \
    [--opencompass-dir <path>] [--opencompass-datasets "<ds1 ds2 ...>"] [--skip-opencompass] \
    <exp1> [exp2] ...

Experiments:
  nosft_08b
  nosft_9b
  sft_08b
  sft_9b
  distill
  random
  full
  smoke
  all

OpenCompass:
  --opencompass-dir <path>         Path to a cloned OpenCompass repo. When set, general-capability
                                   eval (MMLU/GSM8K/HumanEval/MBPP) runs automatically after each
                                   experiment's safety eval. Environment variable OPENCOMPASS_DIR
                                   also works.
  --opencompass-datasets "<list>"  Space-separated dataset config names forwarded to OpenCompass.
                                   Default: "mmlu_gen gsm8k_gen humaneval_gen mbpp_gen".
  --skip-opencompass               Explicitly skip OpenCompass even if --opencompass-dir is set.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --device)
        DEVICE="$2"
        shift 2
        ;;
      --visible-devices)
        VISIBLE_DEVICES="$2"
        shift 2
        ;;
      --opencompass-dir)
        OPENCOMPASS_DIR="$2"
        shift 2
        ;;
      --opencompass-datasets)
        OPENCOMPASS_DATASETS="$2"
        shift 2
        ;;
      --skip-opencompass)
        SKIP_OPENCOMPASS="1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        clean_exp="$(printf '%s' "$1" | tr -d '\r' | xargs)"
        EXPERIMENTS+=("${clean_exp}")
        shift
        ;;
    esac
  done

  if [[ -z "${DEVICE}" || -z "${VISIBLE_DEVICES}" || ${#EXPERIMENTS[@]} -eq 0 ]]; then
    usage
    exit 1
  fi

  if [[ "${DEVICE}" != "npu" && "${DEVICE}" != "tpu" ]]; then
    echo "[ERROR] --device must be npu or tpu"
    exit 2
  fi
}

setup_runtime() {
  cd "${PROJECT_ROOT}"

  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    echo "[INFO] Installing benchmark dependencies..."
    "${PYTHON_BIN}" -m pip install -U datasets pyarrow
  fi

  IFS=',' read -r FIRST_VISIBLE_ID _ <<< "${VISIBLE_DEVICES}"
  if [[ "${DEVICE}" == "npu" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
    EFFECTIVE_DEVICE_ID="0"
    echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "[INFO] Using logical device npu:${EFFECTIVE_DEVICE_ID} mapped onto visible physical NPU(s): ${VISIBLE_DEVICES}"
  else
    EFFECTIVE_DEVICE_ID="${FIRST_VISIBLE_ID}"
  fi
}

run_cmd() {
  echo
  echo "============================================================"
  echo "[RUN] $*"
  echo "============================================================"
  "$@"
}

run_launcher() {
  local subcommand="$1"
  shift

  local -a launcher_args=(
    "${subcommand}"
    --device "${DEVICE}"
    --device-id "${EFFECTIVE_DEVICE_ID}"
    --num-devices 1
  )
  if [[ "${SKIP_OPENCOMPASS}" == "1" ]]; then
    launcher_args+=(--skip-opencompass)
  elif [[ -n "${OPENCOMPASS_DIR}" ]]; then
    launcher_args+=(--opencompass-dir "${OPENCOMPASS_DIR}")
    if [[ -n "${OPENCOMPASS_DATASETS}" ]]; then
      # Split dataset list on whitespace so argparse nargs="+" receives multiple values.
      # shellcheck disable=SC2206
      local -a oc_ds=(${OPENCOMPASS_DATASETS})
      launcher_args+=(--opencompass-datasets "${oc_ds[@]}")
    fi
  fi

  run_cmd "${PYTHON_BIN}" scripts/15_run_oneclick.py "${launcher_args[@]}" "$@"
}

run_nosft_08b() { run_launcher nosft --model 0.8b; }
run_nosft_9b() { run_launcher nosft --model 9b; }
run_sft_08b()   { run_launcher sft --model 0.8b; }
run_sft_9b()   { run_launcher sft --model 9b; }
run_distill()  { run_launcher distill; }
run_random()   { run_launcher random; }
run_full()     { run_launcher full; }
run_smoke()    { run_launcher smoke; }

run_all() {
  run_nosft_08b
  run_nosft_9b
  run_sft_08b
  run_sft_9b
  run_distill
  run_random
  run_full
}

main() {
  parse_args "$@"
  setup_runtime

  for exp in "${EXPERIMENTS[@]}"; do
    case "${exp}" in
      nosft_08b) run_nosft_08b ;;
      nosft_9b) run_nosft_9b ;;
      sft_08b) run_sft_08b ;;
      sft_9b) run_sft_9b ;;
      distill) run_distill ;;
      random) run_random ;;
      full) run_full ;;
      smoke) run_smoke ;;
      all) run_all ;;
      *)
        echo "[ERROR] Unknown experiment: ${exp}"
        printf '[DEBUG] experiments:'
        for item in "${EXPERIMENTS[@]}"; do
          printf ' <%s>' "$item"
        done
        printf '\n'
        usage
        exit 3
        ;;
    esac
  done

  echo
  echo "[DONE] Requested experiments finished."
}

main "$@"
