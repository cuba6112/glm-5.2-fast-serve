#!/usr/bin/env bash
set -euo pipefail
# Fast single-stream recipe (2026-07-08): TP4 on GPUs 0-3 + MTP spec decode (nspec=3)
# + marlin MoE on the eldritch-enlightenment image = ~119 tok/s decode (thinking),
# ~3x the old TP2xPP3 39 tok/s. Old image/topology still reachable via env overrides.
IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707}
NAME=glm-5.2-vllm
PORT=${PORT:-30001}
MODEL=${MODEL:-/data/glm-5.2-nvfp4}   # override e.g. MODEL=/data/glm-5.2-nvfp4-nvidia for the official NVIDIA quant
RECIPE_DIR=/data/glm-5.2/recipe
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
DCP_SIZE=${DCP_SIZE:-1}
MTP=${MTP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-163840}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.96}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}   # GB of weights/GPU offloaded to host RAM (frees VRAM for KV)
LOAD_FORMAT=${LOAD_FORMAT:-fastsafetensors}   # fastsafetensors bypasses cpu-offload-gb; use 'auto' when offloading
V2_RUNNER=${V2_RUNNER:-1}   # V2 model runner never installs the offloader -> set 0 when CPU_OFFLOAD_GB>0
PCIE_AR=${PCIE_AR:-0}   # 1 = b12x PCIe oneshot allreduce (custom-AR path for PCIe-only rigs; cuts TP allreduce latency)
SPEC_EXTEND_AS_DECODE=${SPEC_EXTEND_AS_DECODE:-0}   # 1 = route MTP verify through fast decode kernel instead of extend/prefill path
MOE_FORCE_A16=${MOE_FORCE_A16:-1}   # 1 = W4A16 MoE (safe default); 0 = native FP4 activations (faster, verify output quality!)
MOE_BACKEND=${MOE_BACKEND:-marlin}   # marlin ~9% faster than b12x for batch-1 decode; flashinfer_cutedsl unsupported on SM120, flashinfer_b12x crashes
LINEAR_BACKEND=${LINEAR_BACKEND:-auto}   # auto | b12x | marlin | machete | ... (see kernel.py LinearBackend)
CUDAGRAPH_SIZES=${CUDAGRAPH_SIZES:-}   # optional explicit capture sizes, e.g. "1,2,3,4,6,8,12" to capture the exact MTP verify batch (1+nspec)
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}   # fp8 | nvfp4 | turboquant_3bit_nc | turboquant_k3v4_nc — sub-fp8 shrinks KV to fit long ctx
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-0}   # bytes; if >0, PIN KV pool size (bypasses the non-deterministic auto-profiler for reliable boots)
CUDA_VISIBLE=${CUDA_VISIBLE:-0,1,2,3}   # TP4 default; use 0,1,2,3,4,5 for TP2xPP3 6-GPU layouts
INDEX_TOPK_PATTERN=FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
HF_OVERRIDES="{\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"

docker rm -f "$NAME" 2>/dev/null || true

KVMEM_ARG=""
[[ "${KV_CACHE_MEMORY_BYTES}" != "0" ]] && KVMEM_ARG="--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"

# custom allreduce must be ENABLED for the b12x PCIe oneshot path to initialize
DISABLE_CAR_FLAG="--disable-custom-all-reduce"
[[ "$PCIE_AR" == "1" ]] && DISABLE_CAR_FLAG=""

COMPILATION_FLAG=""
[[ -n "$CUDAGRAPH_SIZES" ]] && COMPILATION_FLAG="--compilation-config '{\"cudagraph_capture_sizes\":[${CUDAGRAPH_SIZES}]}'"

SPEC_ARGS=()
MAX_CUDAGRAPH=${MAX_NUM_SEQS}
SPEC_FLAG=""
if [[ "$MTP" == "1" ]]; then
  NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
  DRAFT_SAMPLE=${DRAFT_SAMPLE:-greedy}   # greedy | probabilistic
  DRAFT_TP=${DRAFT_TP:-}   # set to 1 to run the 1-layer MTP draft replicated (no TP allreduce in draft)
  DRAFT_TP_JSON=""
  [[ -n "$DRAFT_TP" ]] && DRAFT_TP_JSON=",\"draft_tensor_parallel_size\":${DRAFT_TP}"
  SPEC_JSON="{\"model\":\"${MODEL}\",\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"moe_backend\":\"${MOE_BACKEND}\",\"draft_sample_method\":\"${DRAFT_SAMPLE}\"${DRAFT_TP_JSON}}"
  SPEC_FLAG="--speculative-config '${SPEC_JSON}'"
  MAX_CUDAGRAPH=$((MAX_NUM_SEQS * (NUM_SPEC_TOKENS + 1)))
fi

docker run -d \
  --name "$NAME" \
  --gpus all --ipc=host --shm-size=64g --network host --privileged \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_MIN_NCHANNELS=8 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_DEBUG=WARN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER="$V2_RUNNER" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="$PCIE_AR" \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e USES_B12X=True \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16="$MOE_FORCE_A16" \
  -e HF_OVERRIDES="$HF_OVERRIDES" \
  -e XDG_CACHE_HOME=/cache/jit \
  -e CUDA_CACHE_PATH=/cache/jit \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e VLLM_TORCH_PROFILER_DIR=/cache/jit/profile \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE="$SPEC_EXTEND_AS_DECODE" \
  -v "$MODEL:$MODEL:ro" \
  -v "$RECIPE_DIR:/recipe:ro" \
  -v glm52-jit:/cache/jit \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc "unset NCCL_GRAPH_FILE NCCL_TOPO_FILE 2>/dev/null; exec /opt/venv/bin/vllm serve '${MODEL}' \
    --served-model-name glm-5.2-nvfp4 glm-5.2 \
    --trust-remote-code --host 0.0.0.0 --port '${PORT}' \
    --tensor-parallel-size '${TP_SIZE}' \
    --pipeline-parallel-size '${PP_SIZE}' \
    --decode-context-parallel-size '${DCP_SIZE}' \
    ${DISABLE_CAR_FLAG} \
    --enable-chunked-prefill --enable-prefix-caching \
    --load-format '${LOAD_FORMAT}' --async-scheduling \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
    ${KVMEM_ARG} \
    --cpu-offload-gb '${CPU_OFFLOAD_GB}' \
    --max-num-batched-tokens 8192 \
    --max-num-seqs '${MAX_NUM_SEQS}' \
    --max-cudagraph-capture-size '${MAX_CUDAGRAPH}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --quantization modelopt_fp4 \
    --attention-backend B12X_MLA_SPARSE \
    --moe-backend '${MOE_BACKEND}' \
    --linear-backend '${LINEAR_BACKEND}' \
    ${COMPILATION_FLAG} \
    --kv-cache-dtype '${KV_CACHE_DTYPE}' \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --chat-template /recipe/chat_template.maxthink.jinja \
    --hf-overrides '${HF_OVERRIDES}' \
    ${SPEC_FLAG}"

echo "Started $NAME TP=$TP_SIZE PP=$PP_SIZE DCP=$DCP_SIZE port=$PORT"
