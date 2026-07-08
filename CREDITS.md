# Credits

This file is the long-form attribution list for **glm-5.2-fast-serve**.  
Please keep names attached when you fork, blog, or ship products on this stack.

---

## 1. Foundation model

### Z.ai / Zhipu AI (`zai-org`)
- **GLM-5.2** base model: MoE architecture, DSA sparse attention, MLA, IndexShare, native MTP head, training and open release (MIT).
- Links:
  - https://huggingface.co/zai-org/GLM-5.2
  - https://huggingface.co/blog/zai-org/glm-52-blog

Without GLM-5.2 there is nothing to quantize, prune, or serve.

---

## 2. Quantization

### Luke Alonso
- **NVFP4** (ModelOpt / TensorRT-Model-Optimizer style) quantization of GLM-5.2 for Blackwell-class GPUs.
- Primary HF reference used in community lineage:
  - https://huggingface.co/lukealonso/GLM-5.2-NVFP4

Other community NVFP4 / alternate quant authors (Mapika, mmangkad, Lorbus, Unsloth GGUF, cyankiwi AWQ, etc.) also advanced the ecosystem; this recipe’s default weights descend from the Alonso NVFP4 line via REAP-Recall.

---

## 3. MoE compression (REAP)

### Cerebras Research
- **REAP** — *Router-weighted Expert Activation Pruning* for one-shot MoE expert pruning.
- Paper: https://arxiv.org/abs/2510.13999 (ICLR 2026)
- Blog: https://www.cerebras.ai/blog/reap
- Code: https://github.com/CerebrasResearch/reap

### brandonmusic / brandonmmusic-max
- **GLM-5.2-NVFP4-REAP-Recall-N172** — the default weights this recipe downloads.
- Re-ran REAP with knowledge/legal-balanced calibration so closed-book recall does not collapse.
- Block-wise NVFP4 saliency runner, self-consistent 172-expert prune, docs and alternate serving notes.
- Links:
  - https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172
  - https://github.com/brandonmmusic-max/GLM-5.2-Reap

Their model card states:

> Attribution: **GLM-5.2** by z.ai · **NVFP4 quantization** by **Luke Alonso** · **REAP** by **Cerebras Research** (arXiv:2510.13999, ICLR 2026).

---

## 4. Serving engine & Blackwell kernels

### vLLM project
- Core continuous-batching OpenAI-compatible server this stack extends.
- https://github.com/vllm-project/vllm

### b12x / Festr and kernel contributors
- Sparse MLA, MoE, and related SM120 paths that make DSA GLM models practical on RTX PRO 6000-class GPUs.
- Work surfaces in community Docker tags under `voipmonitor/vllm`.

### local-inference-lab / rtx6kpro community
- Optimization matrix, image documentation, and RTX PRO 6000 serving notes.
- https://github.com/local-inference-lab/rtx6kpro

### voipmonitor (Docker Hub publisher)
- Hosts **eldritch-enlightenment** vLLM + b12x + CUDA 13.2 images used by this recipe.
- https://hub.docker.com/r/voipmonitor/vllm

Pinned image:

```text
voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
```

### FlashInfer contributors
- Sampling / attention components commonly bundled in these images.
- https://github.com/flashinfer-ai/flashinfer

### NVIDIA
- CUDA, drivers, NCCL, ModelOpt NVFP4 tooling, Blackwell GPUs.

### Related serving references
- **verdictai** — alternate GLM-5.2 REAP Docker path referenced on the model card (`glm52-nvfp4-dcpmtp`).
- **0xSero** and other SM120 GLM runners — multi-GPU / NCCL hard-won notes.
- Countless RTX 6000 Discord/forum posters who published PCIe, power, and decode measurements.

---

## 5. This packaging / tuning repo

### Jose Munoz (`cuba6112`)
- Public packaging of `glm.sh` / `launch-vllm.sh` / chat templates.
- Fixed silent MTP no-op (`SPEC_ARGS` host-array vs container string).
- A/B topology for single-stream: **TP4 + PP=1 + MTP + Marlin MoE** on the eldritch image.
- Default ~160K context fast profile and documentation.

Repo: https://github.com/cuba6112/glm-5.2-fast-serve

---

## Suggested citation (results / blogs)

```text
GLM-5.2 (Z.ai) → NVFP4 (Luke Alonso) → REAP-Recall N172 (brandonmusic, method: Cerebras REAP)
served with community b12x / eldritch-enlightenment vLLM (voipmonitor/vllm, rtx6kpro community)
recipe packaging & TP4+MTP+Marlin tuning: cuba6112/glm-5.2-fast-serve
```

---

## Corrections

If your name or project is missing or mis-attributed, please open a PR or issue on this repository. We will fix it promptly.
