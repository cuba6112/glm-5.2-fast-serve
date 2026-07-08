#!/bin/bash
# Roll back to the original glm-5.2-vllm container.
set -euo pipefail
docker stop glm-5.2-vllm || true
docker rm glm-5.2-vllm || true
docker rename glm-5.2-vllm-old glm-5.2-vllm
docker start glm-5.2-vllm
echo "Rolled back. Poll: curl -s localhost:30001/health"
