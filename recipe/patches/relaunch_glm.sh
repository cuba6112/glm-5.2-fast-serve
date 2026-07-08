#!/bin/bash
# Relaunch glm-5.2-vllm with patched tool parser + reasoning parser.
# Generated /data/glm-5.2/recipe/patches/gen_relaunch.py. Original container kept as glm-5.2-vllm-old.
set -euo pipefail
docker stop glm-5.2-vllm
docker rename glm-5.2-vllm glm-5.2-vllm-old
docker run -d --name glm-5.2-vllm \
  --network host \
  --ipc host \
  --shm-size 68719476736 \
  --privileged \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --gpus all \
  --restart no \
  --env-file /data/glm-5.2/recipe/patches/glm52.env \
  -v /data/glm-5.2/recipe:/recipe:ro \
  -v glm52-jit:/cache/jit \
  -v /data/glm-5.2-reap:/data/glm-5.2-reap:ro \
  -v /data/glm-5.2/recipe/patches/glm4_moe_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/glm4_moe_tool_parser.py:ro \
  -w / \
  --entrypoint /bin/bash \
  voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608 \
  -lc 'unset NCCL_GRAPH_FILE NCCL_TOPO_FILE 2>/dev/null; exec /opt/venv/bin/vllm serve '"'"'/data/glm-5.2-reap'"'"'     --served-model-name glm-5.2-nvfp4 glm-5.2     --trust-remote-code --host 0.0.0.0 --port '"'"'30001'"'"'     --tensor-parallel-size '"'"'2'"'"'     --pipeline-parallel-size '"'"'3'"'"'     --decode-context-parallel-size '"'"'1'"'"'     --disable-custom-all-reduce     --enable-chunked-prefill --enable-prefix-caching     --load-format fastsafetensors --async-scheduling     --gpu-memory-utilization '"'"'0.92'"'"'          --cpu-offload-gb '"'"'0'"'"'     --max-num-batched-tokens 8192     --max-num-seqs '"'"'2'"'"'     --max-cudagraph-capture-size '"'"'2'"'"'     --max-model-len '"'"'1048576'"'"'     --quantization modelopt_fp4     --attention-backend B12X_MLA_SPARSE     --moe-backend b12x     --kv-cache-dtype '"'"'fp8'"'"'     --tool-call-parser glm47     --reasoning-parser glm45     --enable-auto-tool-choice     --chat-template /recipe/chat_template.maxthink.jinja     --hf-overrides '"'"'{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'"'"'     ${SPEC_ARGS[@]:-}'
echo "New container started. Poll: curl -s localhost:30001/health"
