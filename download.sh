#!/usr/bin/env bash
# Download Mapika/GLM-5.2-NVFP4 (~410GB) to /data/glm-5.2-nvfp4
set -u
export HF_HOME=/data/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/home/cuba6112/.local/bin/hf
LOG=/data/glm-5.2/download.log
MODEL=Mapika/GLM-5.2-NVFP4
DEST=/data/glm-5.2-nvfp4

echo "=== GLM-5.2-NVFP4 download start $(date) ===" | tee -a "$LOG"
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
