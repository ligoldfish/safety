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

usage() {
  cat <<'EOF'
Usage:
  bash /root/run_experiments.sh --device <npu|tpu> --visible-devices <ids> <exp1> [exp2] ...

Experiments:
  nosft_1b
  nosft_9b
  sft_1b
  sft_9b
  distill
  random
  full
  smoke
  all
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
  EFFECTIVE_DEVICE_ID="${FIRST_VISIBLE_ID}"
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

  run_cmd "${PYTHON_BIN}" scripts/15_run_oneclick.py "${subcommand}" \
    --device "${DEVICE}" \
    --device-id "${EFFECTIVE_DEVICE_ID}" \
    --num-devices 1 \
    "$@"
}

run_nosft_1b() { run_launcher nosft --model 1b; }
run_nosft_9b() { run_launcher nosft --model 9b; }
run_sft_1b()   { run_launcher sft --model 1b; }
run_sft_9b()   { run_launcher sft --model 9b; }
run_distill()  { run_launcher distill; }
run_random()   { run_launcher random; }
run_full()     { run_launcher full; }
run_smoke()    { run_launcher smoke; }

run_all() {
  run_nosft_1b
  run_nosft_9b
  run_sft_1b
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
      nosft_1b) run_nosft_1b ;;
      nosft_9b) run_nosft_9b ;;
      sft_1b) run_sft_1b ;;
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
