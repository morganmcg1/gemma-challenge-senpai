STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["6i58gvjk"],"primary_metric":{"name":"tps_local_control_triton2d","value":205.38},"test_metric":{"name":"ppl_local_control","value":1.9929}}

## Results — FlashInfer decode-backend screen: REFUTED (terminal negative, LOCAL ONLY)

**Verdict:** FlashInfer attention is **incompatible with Gemma4-E4B** on this stack (vLLM 0.22.0, flashinfer 0.6.11.post2, A10G/sm86). It does **not load** — it crashes at engine **warmup, before any decode** — so it cannot serve as the bi0 decode backend. The hypothesis (FlashInfer → faster, greedy-identical decode) is dead on arrival. **No HF job warranted; no submission made.**

### Variant (FlashInfer) — DEAD ON ARRIVAL
Selected via the engine arg `attention_backend=FLASHINFER` (the body's `VLLM_ATTENTION_BACKEND` env is a dead/ignored var in this build — confirmed). Two independent FlashInfer blockers, both at engine init/warmup:

**1. Sampler JIT crash** (FlashInfer top-k/top-p sampler auto-enables with the FlashInfer backend):
```
flashinfer/data/include/flashinfer/sampling.cuh:20:10: fatal error: curand.h: No such file or directory
   20 | #include <curand.h>
```
The JIT (ninja/nvcc) build of `csrc/sampling.cu` fails because `curand.h` is absent from the local CUDA include path. **Worked around** with `VLLM_USE_FLASHINFER_SAMPLER=0` → PyTorch-native sampler, which is **numerically free for greedy** (temp=0 short-circuits to `argmax` before the top-k/top-p path is ever called). This unblocks the sampler but not the decisive blocker below.

**2. Attention paged-prefill dispatch crash (decisive, unfixable from our side):**
```
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
  at flashinfer/attention/prefill.cuh:2978: FlashInfer Internal Error:
  Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32
  NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```
fires at `kernel_warmup.py:97 Warming up FlashInfer attention`. `NUM_MMA_D_QK=32` ⟹ **head_dim = 512** — Gemma4-E4B's **global**-attention head dim. FlashInfer has **no valid paged-prefill kernel template for head_dim=512 on sm86**, so the dispatch raises before producing a single token. This is the same architectural reason vLLM **force-pins TRITON_ATTN** for Gemma4 (heterogeneous head dims 256 sliding / 512 global exceed the ≤256 ceiling of FA-style kernels).

**The PR's K=0 fallback is ALSO dead.** I reproduced crash #2 in an **offline `LLM()` smoke with `num_speculative_tokens=0` (spec OFF) and `enforce_eager=True`** — i.e. plain FlashInfer paged-prefill, no MTP, no cudagraph. It still crashes identically. So the incompatibility is in FlashInfer's attention kernel itself, **independent of speculative decoding**; re-running both control+variant at K=0 cannot succeed, so I did not spend a GPU window on it.

This **reconfirms a prior signal** (earlier vLLM build crashed Gemma4 FlashInfer at warmup with `Unsupported max_mma_kv`) — now on 0.22.0 + flashinfer 0.6.11.post2 with a refined signature. **The Triton force-pin is load-bearing, not cautious.**

### Control (bi0) — clean local anchor (backend correction)
**Correction to PR body:** the shipped bi0 backend is **TRITON_ATTN force-2D (surgattn)**, *not* FlashAttention-2. Serve log: `INFO ... cuda.py:318 Using AttentionBackendEnum.TRITON_ATTN backend.` `vllm_force2d_attn_patch.py` patches `vllm.v1.attention.backends.triton_attn` and nulls `softmax_segm_*` to force the 2D single-pass branch (vs 3D split-KV) for M=1 decode and M=K verify. So this screen is **FlashInfer vs Triton-force2D**.

Local exploratory prevalidate (shared/contended A10G — *not* the official a10g-small):

| metric | control (Triton-2D, local) | baseline (official a10g-small) |
|---|---|---|
| tps | **205.38** | 218.02 |
| ppl | **1.9929**  (≤ 2.42 ✓) | 2.0058 |
| completed | 32/32 ppl, 16 decode (token_ids captured ✓) | 128/128 |

Local TPS trails official as expected (shared GPU, 32-record exploratory harness, not the 128-prompt official run). The point of the control is **harness health + quality pass**, confirmed: PPL 1.9929 ≤ 2.42, all records completed, decode token_ids returned.

**Greedy token-identity: N/A** — there is no loadable FlashInfer variant to diff token_ids against (variant never reaches decode).

### Commands
Decisive variant smoke (offline, no-spec, sampler-off — proves the K=0 fallback is dead too):
```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python research/bi0_flashinfer_backend/smoke_flashinfer_grab.py
# (LLM(attention_backend=FLASHINFER, num_speculative_tokens=0, enforce_eager=True) → RESULT=FAIL_FLASHINFER, exit 2)
```
Control prevalidate (Triton-2D anchor):
```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python scripts/local_prevalidate.py \
  --submission submissions/int4_mtp_bi0_surgattn \
  --venv-python /workspace/senpai/target/.venv/bin/python \
  --ppl-records 32 --decode-num-prompts 16 --decode-output-len 256 \
  --server-log research/bi0_flashinfer_backend/control_serve.log
# → SENPAI-LOCAL tps=205.3762 ppl=1.9929 completed=32 decode_sample=16
```

### Peak memory
Control serve @ `gpu_memory_utilization=0.9`: model load **9.86 GiB** + **8.45 GiB** KV cache (336,844 tokens) on the 22.06 GiB A10G (reserve ≤ ~19.85 GiB). The variant never allocated KV (crashed at attention warmup).

### W&B
Run `6i58gvjk`, group `bi0-flashinfer` (liveness/orientation + verdict). `local_prevalidate.py` writes JSON artifacts, not W&B, so the control numbers live in `research/local_validation/int4_mtp_bi0_surgattn/{local,ppl,decode}_summary.json` and the raw crash logs in `research/bi0_flashinfer_backend/`.

### What happened — honest analysis
FlashInfer fundamentally cannot serve Gemma4-E4B's `head_dim=512` global attention on sm86: its paged-prefill kernel has no template for that head dim, so it aborts at warmup. This is an upstream **kernel coverage gap, not a config knob** — there is nothing to tune. The result is robust: it holds with spec OFF and enforce_eager, so it is not an MTP or cudagraph interaction. The bi0 Triton-force2D control is healthy and quality-passing, which is why bi0 ships on Triton in the first place.

### Suggested follow-ups
- **Do not revisit FlashInfer for Gemma4 on sm86** unless upstream flashinfer adds a `head_dim=512` paged-prefill template (track the flashinfer repo; this is a kernel PR, not a flag).
- The decode-speed goal still stands, but the realizable levers live on the **Triton-force2D path** (speculative `K`, the drafter, Marlin int4 body) — not the attention backend. The attention backend is fixed by architecture.
- **Infra:** a persistent co-tenant holds ~19.6 GiB on the shared physical A10G; only brief free windows open. Long local runs need opportunistic windowing (the grabber I used catches a window the instant free VRAM clears).
