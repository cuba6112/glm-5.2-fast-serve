# GLM-5.2 Fast Serve Recipe

Shareable **vLLM + b12x** recipe for high single-stream decode on multi-GPU Blackwell.

Validated on a 6× RTX PRO 6000 (96 GB) box: single-stream decode from ~**39 tok/s → ~100–122 tok/s** while keeping a stable OpenAI-compatible `/v1` API.

> Weights are **not** in this repo (~294 GB). Scripts download them from Hugging Face.

## Results (A/B, 2026-07-08)

| Stage | Change | ~Decode tok/s |
|-------|--------|----------------|
| Baseline | TP2×PP3, 6 GPUs | ~39 |
| +topology | **TP4, PP=1** on GPUs 0–3 | ~53 |
| +spec | **MTP** (3 draft tokens) actually enabled | ~85 |
| +MoE | **Marlin** MoE backend | ~+9% |
| +image | `eldritch-enlightenment` 2026-07-07 | **≥100** (peak ~122 thinking) |

**Default context after the fast recipe:** `163840` (was 327K / 1M layouts).  
Full 327K: `DCP_SIZE=2 MAX_MODEL_LEN=327680 ./glm.sh start` (~85 tok/s).

## What you need

| Requirement | Notes |
|-------------|--------|
| GPUs | Prefer **≥4× large Blackwell** (~80–96 GB) for the full recipe |
| Docker | NVIDIA Container Toolkit |
| Disk | ~300 GB+ free for REAP weights + image |
| Image | `voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707` |
| Model | `brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172` (172/256 experts) |

Heads = **64** → valid TP sizes: 1, 2, 4, 8, …

## Quick start

```bash
# 1) clone
git clone https://github.com/cuba6112/glm-5.2-fast-serve.git
cd glm-5.2-fast-serve

# 2) layout (defaults match the original box)
sudo mkdir -p /data/glm-5.2 /data/glm-5.2-reap /data/huggingface
sudo chown -R "$USER" /data/glm-5.2 /data/glm-5.2-reap /data/huggingface
cp -a glm.sh launch-vllm.sh download-reap.sh recipe /data/glm-5.2/
chmod +x /data/glm-5.2/*.sh

# 3) weights (~294G)
export HF_TOKEN=…   # if gated / rate-limited
/data/glm-5.2/download-reap.sh

# 4) image
docker pull voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707

# 5) serve
cd /data/glm-5.2
./glm.sh start          # ~5 min: load + cudagraphs
./glm.sh status
./glm.sh test
```

API:

```text
http://HOST:30001/v1
models: glm-5.2 , glm-5.2-nvfp4
max context (default): 163840
```

Pi / client example:

```json
{
  "contextWindow": 163840,
  "baseUrl": "http://HOST:30001/v1"
}
```

## Controls (`glm.sh`)

```bash
./glm.sh start | stop | restart | status | logs | test
```

### Useful overrides

```bash
# full 327K context (slower)
DCP_SIZE=2 MAX_MODEL_LEN=327680 ./glm.sh start

# old 6-GPU 1M layout (~39 tok/s)
TP_SIZE=2 PP_SIZE=3 CUDA_VISIBLE=0,1,2,3,4,5 MTP=0 \
  MAX_MODEL_LEN=1048576 GPU_MEMORY_UTILIZATION=0.92 ./glm.sh start

# different GPUs
CUDA_VISIBLE=0,1,2,3 TP_SIZE=4 ./glm.sh start
```

## Why it got fast (do not undo these)

1. **PP=1 for single-stream** — pipeline parallel serializes decode.  
2. **MTP actually wired** — pass `--speculative-config '…'` as a **string** into the container; do **not** expand a host bash array inside the container (silent no-op).  
3. **MTP needs PP=1** — draft path does not support PP.  
4. **Marlin MoE** — faster batch-1 and frees VRAM for ~160K.  
5. **New b12x image** — batch-1 decode kernels.

### Dead ends (already tried)

PCIe oneshot allreduce, greedy drafting, draft TP1, 450 W power caps (decode is kernel-latency-bound), GPU set `{0,2,3,4}`, native-FP4 activations, exact-size cudagraphs, flashinfer backends.  
`cpu-offload-gb` is a **silent no-op** under V2 model runner (`V2_RUNNER=1`).

## Repo layout

```text
glm.sh                 # start/stop/status/test
launch-vllm.sh         # docker + vLLM flags (source of truth)
download-reap.sh       # HF download for REAP weights
download-nvidia.sh     # optional full NVFP4 path
download.sh            # legacy helper
recipe/
  chat_template.*.jinja
  patches/             # historical patches / env notes
```

## Security

- No model weights, tokens, or host passwords are stored here.  
- Set `HF_TOKEN` yourself when downloading.  
- Bind the API carefully if the host is on a public network (auth proxy / firewall).

## License

MIT (scripts and templates). Model weights remain under their upstream licenses (Hugging Face model cards).

## Credits & attribution

This repo only packages a **serving recipe**. Nearly all of the intelligence, weights, kernels, and compression work came from others. Please credit them when you share results.

### Model lineage

| Who | What | Links |
|-----|------|--------|
| **Z.ai / Zhipu AI** (`zai-org`) | Original **GLM-5.2** architecture, training, MTP layer, DSA/MLA design, open weights (MIT) | [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), [blog](https://huggingface.co/blog/zai-org/glm-52-blog) |
| **Luke Alonso** | **NVFP4** ModelOpt quantization of GLM-5.2 (Blackwell-oriented base quant) | [lukealonso/GLM-5.2-NVFP4](https://huggingface.co/lukealonso/GLM-5.2-NVFP4) |
| **Cerebras Research** | **REAP** — Router-weighted Expert Activation Pruning (one-shot MoE compression) | [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), [blog](https://www.cerebras.ai/blog/reap), [github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) |
| **brandonmusic** / **brandonmmusic-max** | **REAP-Recall N=172** checkpoint used here: knowledge/legal-balanced calibration, block-wise NVFP4 saliency, self-consistent prune (172 experts), serving docs | [HF model](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172), [GLM-5.2-Reap repo](https://github.com/brandonmmusic-max/GLM-5.2-Reap) |

Model-card attribution (from the REAP-Recall README):

> **GLM-5.2** by z.ai · **NVFP4 quantization** by **Luke Alonso** · **REAP** by **Cerebras Research** (arXiv:2510.13999, ICLR 2026).

### Inference stack (kernels, image, engine)

| Who | What | Links |
|-----|------|--------|
| **vLLM project** | Open-source high-throughput serving engine this stack builds on | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **b12x / Festr** & contributors | Blackwell (SM120) sparse-MLA, MoE, and related kernels that make GLM DSA usable on RTX PRO 6000 | Community builds published via `voipmonitor/vllm` |
| **local-inference-lab / rtx6kpro** community | Documented RTX PRO 6000 optimization paths, image matrix, and serving notes | [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) |
| **voipmonitor** (Docker Hub) | Publishes the **`eldritch-enlightenment`** vLLM+b12x CUDA 13.2 images used by this recipe | [hub.docker.com/r/voipmonitor/vllm](https://hub.docker.com/r/voipmonitor/vllm) |
| **NVIDIA** | CUDA / cuDNN / NCCL, ModelOpt NVFP4 tooling, Blackwell hardware | — |
| **FlashInfer** contributors | Sampling / attention building blocks pulled into the image stack | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |

Image pin used by this recipe:

```text
voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
```

(The “eldritch” / b12x line has had many community hands; if you know an author we missed, open a PR.)

### Related community work (not all in this recipe, but worth citing)

| Who | What |
|-----|------|
| **Mapika**, **mmangkad**, **Lorbus**, others | Additional GLM-5.2 NVFP4 / quant variants |
| **Unsloth** | GLM-5.2 GGUF Dynamic 2.0 quants (CPU/Apple/llama.cpp path) |
| **0xSero** and SM120 GLM runners | Early SM120 multi-GPU / NCCL notes for GLM-5.x |
| **verdictai** | Alternate Docker serving image referenced on the REAP-Recall model card (`glm52-nvfp4-dcpmtp`) |
| RTX 6000 / local-inference Discord & forum posters | Shared failure modes, PCIe NCCL flags, power/decode measurements |

### This repository

| Who | What |
|-----|------|
| **Jose Munoz** (`cuba6112`) | Packaging, launcher hardening (MTP `SPEC_ARGS` fix), A/B topology tuning (TP4 + MTP + Marlin), default 160K fast profile, docs |

If you publish numbers from this stack, a fair one-liner is:

> Served with community **b12x/eldritch vLLM** on Blackwell; weights **brandonmusic REAP-Recall N172** of **Luke Alonso NVFP4** of **Z.ai GLM-5.2**, compressed with **Cerebras REAP**; recipe packaging by **cuba6112**.

See also [CREDITS.md](./CREDITS.md).
