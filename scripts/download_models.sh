#!/usr/bin/env bash
# Download teacher/student models for the cross-scale safety-transfer pairs
# (src/pairs.py). Models land in <repo>/models/<local-name>; configs reference
# ../models/<name> relative to the scripts/ cwd, i.e. <repo>/models/<name>.
#
# !!! ALIGNED vs BASE -- do NOT mix up (the two families use OPPOSITE suffixes):
#   * Llama : ALIGNED = "-Instruct" suffix.  Bare "Llama-3.1-8B" is the BASE model.
#             -> we download meta-llama/Llama-3.1-8B-Instruct (and 3.2-1B-Instruct).
#   * Qwen3 : ALIGNED = NO suffix (Qwen3-8B). The BASE model is "Qwen3-8B-Base".
#             -> we download the bare Qwen/Qwen3-8B / 4B / 0.6B.
# We need the ALIGNED (chat) weights everywhere: the SVD safety axis requires an
# aligned teacher, and the pipeline needs a chat_template. The script verifies
# each download exposes a chat_template and warns if it looks like a base model.
#
# GATED MODELS: meta-llama/* require accepting the license on HuggingFace and a
# token. Run `hf auth login` (or export HF_TOKEN=...) before this script.
# Qwen/* are open. Needs: pip install -U huggingface_hub  (provides the `hf` CLI).
#
# Usage:
#   bash scripts/download_models.sh            # all 5 models
#   bash scripts/download_models.sh llama      # only the 2 Llama models
#   bash scripts/download_models.sh qwen3      # only the 3 Qwen3 models
#   MODELS_DIR=/root/models bash scripts/download_models.sh   # custom dest
#   HF_ENDPOINT= bash scripts/download_models.sh             # disable the CN mirror
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
# China mirror by default (override or empty to use the official HF endpoint).
# NOTE: gated meta-llama models may need the OFFICIAL endpoint + HF_TOKEN; if the
# mirror 401/404s on Llama, re-run with `HF_ENDPOINT= HF_TOKEN=... bash ...`.
export HF_ENDPOINT="${HF_ENDPOINT-https://hf-mirror.com}"

FILTER="${1:-all}"

# repo_id|local_dir|family  (local_dir MUST match the paths in src/pairs.py)
ALL_MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|llama"   # llama teacher  (ALIGNED, -Instruct)
  "meta-llama/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct|llama"   # llama student  (ALIGNED, -Instruct)
  "Qwen/Qwen3-8B|Qwen3-8B|qwen3"                                   # qwen3 teacher  (ALIGNED, no suffix)
  "Qwen/Qwen3-4B|Qwen3-4B|qwen3"                                   # qwen3 teacher/student (ALIGNED)
  "Qwen/Qwen3-0.6B|Qwen3-0.6B|qwen3"                               # qwen3 student  (ALIGNED)
)

# Resolve the HF download CLI. huggingface_hub renamed `huggingface-cli` to `hf`
# (the old name is now a deprecated no-op that prints a hint and downloads
# nothing). Prefer `hf`; fall back to the legacy CLI only on older envs.
if command -v hf >/dev/null 2>&1; then
  HF_DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL=(huggingface-cli download)
else
  echo "[dl] no HF CLI found. Install: pip install -U huggingface_hub" >&2
  exit 1
fi
echo "[dl] using CLI: ${HF_DL[*]}"

echo "[dl] dest=$MODELS_DIR  endpoint=${HF_ENDPOINT:-<official>}  filter=$FILTER"
mkdir -p "$MODELS_DIR"

for entry in "${ALL_MODELS[@]}"; do
  IFS='|' read -r repo local_dir family <<<"$entry"
  if [[ "$FILTER" != "all" && "$FILTER" != "$family" ]]; then
    continue
  fi
  dest="$MODELS_DIR/$local_dir"
  echo ""
  echo "=== $repo  ->  $dest ==="
  "${HF_DL[@]}" "$repo" --local-dir "$dest" || {
    echo "[dl][ERROR] failed: $repo (gated? run 'hf auth login' / set HF_TOKEN; or HF_ENDPOINT= for official)" >&2
    continue
  }
  # Sanity: aligned/chat models ship a chat_template; base models do not.
  if grep -q "chat_template" "$dest/tokenizer_config.json" 2>/dev/null; then
    echo "[dl] OK chat_template present (aligned) -> $local_dir"
  else
    echo "[dl][WARN] NO chat_template in $local_dir/tokenizer_config.json -- did you grab a BASE model? Expected the ALIGNED variant." >&2
  fi
done

echo ""
echo "[dl] done. Models under $MODELS_DIR. Verify names match src/pairs.py paths (../models/<name>)."
