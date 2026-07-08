#!/usr/bin/env bash
# Easy control for GLM-5.2-NVFP4 (vLLM b12x, DSA sparse attn) on this rig.
#
#   glm.sh start     boot the server (FAST config: TP4 GPUs 0-3 + MTP3 + marlin, 160K ctx, ~119 tok/s)
#   glm.sh stop      stop & remove the container
#   glm.sh restart   stop then start
#   glm.sh status    container state + /health + served model + GPU memory
#   glm.sh logs      follow the server log
#   glm.sh test      send a quick chat request
#
# Override baked-in defaults via env, e.g.:
#   DCP_SIZE=2 MAX_MODEL_LEN=327680 glm.sh start   # full 327K ctx back (KV sharded 2-way), ~85 tok/s
#   TP_SIZE=2 PP_SIZE=3 CUDA_VISIBLE=0,1,2,3,4,5 MTP=0 MAX_MODEL_LEN=1048576 GPU_MEMORY_UTILIZATION=0.92 glm.sh start  # old 6-GPU 1M layout, ~39 tok/s
set -uo pipefail

HERE=/data/glm-5.2
LAUNCH="$HERE/launch-vllm.sh"
CONTAINER=glm-5.2-vllm
PORT="${PORT:-30001}"

# DEFAULT = REAP-pruned model (172/256 experts), FAST single-stream recipe (2026-07-08):
# TP4 on GPUs 0-3 + MTP spec decode (nspec=3) + marlin MoE + eldritch-enlightenment image
# = ~119 tok/s decode (thinking), ~107 (chat), ~98 (creative) at 160K ctx. GPUs 4,5 free.
# 3x the old TP2xPP3 setup (~39 tok/s). MTP needs PP=1, so TP4/4-GPU is mandatory for spec decode.
#   Full 327K ctx back (KV sharded, ~85 tok/s):  DCP_SIZE=2 MAX_MODEL_LEN=327680 ./glm.sh start
#   Old 6-GPU 1M-ctx layout (~39 tok/s):         TP_SIZE=2 PP_SIZE=3 CUDA_VISIBLE=0,1,2,3,4,5 MTP=0 MAX_MODEL_LEN=1048576 GPU_MEMORY_UTILIZATION=0.92 ./glm.sh start
#   Full 753B model (max quality, 6 GPUs):       MODEL=/data/glm-5.2-nvfp4-nvidia TP_SIZE=2 PP_SIZE=3 CUDA_VISIBLE=0,1,2,3,4,5 MTP=0 MAX_MODEL_LEN=327680 GPU_MEMORY_UTILIZATION=0.95 CPU_OFFLOAD_GB=40 ./glm.sh start
export MODEL="${MODEL:-/data/glm-5.2-reap}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"

cmd="${1:-start}"
case "$cmd" in
  start)
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "Already running. '$0 restart' to reload, '$0 status' to check."; exit 0
    fi
    # wait for GPU memory to fully release (docker rm -f returns before driver frees it);
    # only check the GPUs this config uses so jobs on other GPUs don't block the boot
    GPUS="${CUDA_VISIBLE:-0,1,2,3}"
    echo "Waiting for GPUs $GPUS to be free..."
    until [ "$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 4; done
    echo "Starting GLM-5.2 (ctx=$MAX_MODEL_LEN, util=$GPU_MEMORY_UTILIZATION)..."
    ( cd "$HERE" && ./launch-vllm.sh )
    echo "Booting (~5 min: weight load + cudagraph capture). Watch: $0 logs | Check: $0 status"
    ;;
  stop)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "Stopped." || echo "Not running."
    ;;
  restart)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    exec "$0" start
    ;;
  status)
    echo "=== container ==="; docker ps -a --filter "name=$CONTAINER" --format '{{.Names}}  {{.Status}}'
    echo "=== /health ==="
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "OK (ready)"
      curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"][0];print(f"  model={d[\"id\"]}  ctx={d[\"max_model_len\"]}")' 2>/dev/null || true
    else
      echo "not ready (still booting? see: $0 logs)"
    fi
    echo "=== GPU memory ==="; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    ;;
  logs)
    exec docker logs -f "$CONTAINER"
    ;;
  test)
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d '{
      "model":"glm-5.2","messages":[{"role":"user","content":"Give me one surprising fact about the deep ocean."}],
      "max_tokens":150,"temperature":0.7}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null \
      || echo "request failed (server not ready?)"
    ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
