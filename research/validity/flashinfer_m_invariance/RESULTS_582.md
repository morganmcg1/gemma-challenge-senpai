STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["qd89l33g"],"primary_metric":{"name":"flashinfer_tps","value":null},"test_metric":{"name":"flashinfer_self_det","value":null}}

## Results

**Verdict: FlashInfer is NOT a free #319-safe M-invariant lever on base_fullhead Gemma4-E4B — it is not even loadable on this stack.** Both FlashInfer paths fail at engine init, for two distinct, documented reasons; the default TRITON stack confirms the PR premise (not byte-exact M-invariant). `flashinfer_free_identity_lever = false`.

### Deliverables (W&B summary, run `qd89l33g`, group `flashinfer-determinism-zoomout`)

| field | value |
|---|---|
| `analysis_only` | `true` |
| `official_tps` | `0` |
| `flashinfer_byte_exact_m_invariant` | **`false`** (never loaded) |
| `flashinfer_self_det` | `null` (never loaded — undefined) |
| `flashinfer_tps` | `null` (never loaded) |
| `flashinfer_vs_default_tps_delta` | `null` |
| `flashinfer_free_identity_lever` | **`false`** |
| `default_tps_measured` (clean harness) | `96.60` TPS |
| `default_tps_anchor` (ship serve, context only) | `252.69` TPS |

### What happened — three legs, one process (run `qd89l33g`)

**1. Sampler knob `VLLM_USE_FLASHINFER_SAMPLER=1` (the literal PR instruction) → no-op AND crash.**
- *Code-level no-op for greedy:* `v1/sample/sampler.py::Sampler.sample` computes `greedy_sample = logits.argmax(-1)` and returns when `all_greedy` **before** ever reaching `self.topk_topp_sampler`. The flag only routes top-k/top-p **random** sampling, so at temp=0 it cannot change token identity.
- *Hard crash at warmup:* even though real decode is greedy, the warmup/profiling forward exercises the top-k/top-p path, which JIT-compiles the flashinfer sampling kernel. `flashinfer/.../sampling.cuh:20  #include <curand.h>` is **absent on this CUDA box** → ninja build fails → `EngineCore` init `RuntimeError` (rc=3, `_result_fi_sampler_on.log`). So the literal PR knob is **unrunnable** here, not merely inert.

**2. Attention / GEMV path `attention_backend=FLASHINFER` (the only FlashInfer lever that touches the forward/reduction order) → loads the backend, crashes the kernel.**
- *Mechanism note:* in this build (vLLM `0.22.0` dev307) the legacy `VLLM_ATTENTION_BACKEND` **env var is removed** — it logs `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND` and is ignored, so the Gemma4 TRITON force-pin still fires. The backend is now an **engine arg**: `LLM(attention_backend="FLASHINFER")` / `--attention-backend FLASHINFER`. (My first orchestrator pass mistakenly used the dead env var, which silently fell back to TRITON and then the sampler-flag crash masked the test — fixed.)
- *Clean isolation result:* with the engine arg + sampler **off**, the force-pin did **not** fire (`cuda.py:318 Using AttentionBackendEnum.FLASHINFER backend`), so FlashInfer attention **was** selected uniformly — but it crashes at warmup:
  ```
  RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
    .../flashinfer/attention/prefill.cuh:2964: Unsupported max_mma_kv: 0
  ```
  FlashInfer's prefill kernel cannot dispatch for Gemma4-E4B's **heterogeneous head dims** (`head_dim=256` sliding vs `global_head_dim=512` global). This is exactly the mixed-backend divergence the TRITON force-pin (`config.py:100`) exists to prevent — the force-pin is **load-bearing**, not merely cautious.

**3. Default TRITON stack confirms the PR premise — not byte-exact M-invariant.** Greedy token IDs of a fixed reasoning target, decoded at fixed batch width M by co-scheduling distinct fillers (`ignore_eos` + fixed `max_tokens` ⇒ constant width), 400 tokens, ≥2 repeats:

| pair | flip rate | steady flip | first divergence |
|---|---|---|---|
| M=1 ↔ M=8 | **0.6825** | 0.711 | pos 125 |
| M=8 ↔ M=16 | **0.6825** | 0.711 | pos 125 |
| M=1 ↔ M=16 | 0.0* | 0.0* | — |

self-determinism (rep0 vs rep1): M=1 → 1.0, M=8 → 1.0, **M=16 → 0.3225**. *The M1↔M16 "0.0" is a coincidental rep0 alignment and is **not** robust: at M=16 the stack is not even self-deterministic run-to-run (int4+vLLM async-schedule nondeterminism at width 16). The robust signal is a single near-tie first-divergence at pos 125 that cascades — a knife-edge identity break, consistent with prior decode-width findings. Either way `byte_exact_m_invariant = false`.

### TPS basis (important)
The `252.69` anchor is the **ship serving stack** (render / argmax-fusion / detok serving-layer opts that are orthogonal to the attention/sampler backend identity question). The apples-to-apples basis for any FlashInfer-vs-default *backend* comparison is the same clean `LLM()` harness: **default = 96.60 TPS** (warm-median decode, prefill-corrected; runs 96.64 / 96.51 / 96.60 — very stable). Since FlashInfer never loaded, there is no FI TPS to compare, so `flashinfer_vs_default_tps_delta = null`.

### Command
```bash
cd /workspace/senpai/target/research/validity/flashinfer_m_invariance && \
  /workspace/senpai/target/.venv/bin/python run_all.py
# runs probe_one.py in 3 fresh subprocesses on CUDA_VISIBLE_DEVICES=0:
#   default_triton  VLLM_USE_FLASHINFER_SAMPLER=0
#   fi_sampler_on   VLLM_USE_FLASHINFER_SAMPLER=1            (curand JIT crash)
#   fi_attention    PROBE_ATTN_BACKEND=FLASHINFER, sampler 0 (max_mma_kv crash)
# model /tmp/gemma4-e4b-qat-w4a16-ct, full 262k head, cudagraph on, greedy temp=0,
# enable_prefix_caching=False, seed=0, PROBE_N=400, PROBE_TPS_N=512.
```

### Peak memory
A10G 24 GiB at `gpu_memory_utilization=0.90`: model weights **9.93 GiB**, KV cache **8.19 GiB** (151,637 tokens), CUDA-graph pool 0.11 GiB. Within the 0.90 cap. (FlashInfer legs crashed before steady state.)

### Stretch — SGLang batch-1 path
PR gates this note on "only if FlashInfer is byte-exact M-invariant." It is not (it is unloadable), so the precondition fails. Concretely, **SGLang is not worth a follow-up card on this axis**: its attention backends are the same FlashInfer / FlashAttention kernels that hit the identical Gemma4 heterogeneous-head-dim dispatch wall, and byte-exact batch-invariance is a property of *purpose-built batch-invariant kernels*, not of FlashInfer/SGLang by default. Recommend closing the alternate-kernel "free identity lever" axis.

### Public evidence used
- **Batch-invariance framing:** Thinking Machines, *"Defeating Nondeterminism in LLM Inference"* (2025) — byte-exact batch-invariance needs dedicated batch-invariant kernels; stock Marlin/Triton/FlashInfer reductions do not guarantee M-invariance by construction. This is *why* the default stack flips and why "just switch to FlashInfer" was a plausible but ultimately unfounded free-lever hypothesis.
- **Code-level (this `0.22.0` dev307 venv):** `sampler.py` greedy-returns-before-topk_topp; `config.py:100` Gemma4 TRITON force-pin on heterogeneous head dims; `flashinfer/attention/prefill.cuh:2964` `max_mma_kv` dispatch; `flashinfer/.../sampling.cuh:20` curand dependency.
- **Context:** #481 Direction 1 (origin of this wild-card); ship 375.857 TPS, 500 leaderboard gate, public #1 481.53 (speed-only, fails all quality gates); default-stack NO-FIRE measured-closed (#556 / #562 / #571).

### Suggested follow-ups
1. **Close the FlashInfer / SGLang alternate-kernel free-lever axis.** Neither FlashInfer path loads on Gemma4-E4B; the head-dim incompatibility is structural (kernel dispatch), not a config knob we can flip. This reaffirms the existing NO-FIRE on the default stack from a fresh axis.
2. If a batch-invariant identity lever is still wanted, the only credible path is **purpose-built batch-invariant kernels** (thinking-machines `batch_invariant_ops` style) for the Marlin GEMMs + attention — a large kernel-engineering effort, not a free flag. Out of scope for a wild-card card; flag if the team wants to scope it.
3. The identity break is a **localized knife-edge near-tie** (single first-divergence at pos 125 cascading), not a pervasive reduction-order rebuild — consistent with the decode-width census. Worth keeping in mind if a future margin-aware or selective-recompute idea targets the specific near-tie positions rather than the whole stream.
