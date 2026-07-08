#!/usr/bin/env bash
# Download REAP-pruned NVFP4 GLM-5.2 (long-context-recall variant, ~294–315GB)
# to DEST (default /data/glm-5.2-reap). Used by the fast TP4+MTP+Marlin recipe.
set -u
export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
if command -v hf >/dev/null 2>&1; then
  HF=hf
elif [ -x "$HOME/.local/bin/hf" ]; then
  HF="$HOME/.local/bin/hf"
else
  echo "Install Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi
LOG="${LOG:-/data/glm-5.2/download-reap.log}"
MODEL="${MODEL_ID:-brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172}"
DEST="${DEST:-/data/glm-5.2-reap}"
mkdir -p "$(dirname "$LOG")" "$DEST" "$HF_HOME" 2>/dev/null || true

echo "=== REAP download start $(date) : $MODEL ===" | tee -a "$LOG"
n=0
until [ "$n" -ge 50 ]; do
  "$HF" download "$MODEL" --local-dir "$DEST" --max-workers 8 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "=== download COMPLETE $(date) ===" | tee -a "$LOG"; du -sh "$DEST" | tee -a "$LOG"; exit 0
  fi
  n=$((n+1)); echo "=== retry $n rc=$rc $(date) ===" | tee -a "$LOG"; sleep 30
done
echo "FINAL_RC=$rc FAILED" | tee -a "$LOG"; exit 1
