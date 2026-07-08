#!/usr/bin/env bash
# Download nvidia/GLM-5.2-NVFP4 (official ModelOpt quant, ~465GB) to /data/glm-5.2-nvfp4-nvidia.
# Mirrors download.sh (Mapika). Keeps Mapika copy intact as a fallback.
set -u
export HF_HOME=/data/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
# Resolve + export an auth token (HF_HOME override above hides the default stored token).
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
HF=/home/cuba6112/.local/bin/hf
[ -x "$HF" ] || HF=hf
LOG=/data/glm-5.2/download-nvidia.log
MODEL=nvidia/GLM-5.2-NVFP4
DEST=/data/glm-5.2-nvfp4-nvidia

echo "=== nvidia/GLM-5.2-NVFP4 download start $(date) ===" | tee -a "$LOG"
n=0
until [ "$n" -ge 50 ]; do
  "$HF" download "$MODEL" --local-dir "$DEST" --max-workers 8 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "=== download COMPLETE $(date) ===" | tee -a "$LOG"
    du -sh "$DEST" | tee -a "$LOG"
    exit 0
  fi
  n=$((n+1))
  echo "=== retry $n after rc=$rc at $(date), sleeping 30s ===" | tee -a "$LOG"
  sleep 30
done
echo "FINAL_RC=$rc FAILED" | tee -a "$LOG"
exit 1
