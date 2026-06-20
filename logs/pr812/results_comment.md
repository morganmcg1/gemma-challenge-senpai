STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["7rvzl0l9"],"primary_metric":{"name":"int4head_bf16_gemm_pct","value":0.5},"test_metric":{"name":"int4head_ppl","value":2.002908}}

## Results — Step 1 attribution: the 17% bf16 `aten::mm` is the **bf16 tied lm_head**, and it is **already converted to int4** on the firing int4head. STOP.

**TL;DR.** The 17% bf16 GEMM is neither outcome the PR posed. It is the **bf16 tied lm_head** full-vocab logits projection. The cited #809 run was profiled on **`w4a16-ct` (bf16 lm_head)**, *not* the int4head — so the premise "despite an int4 lm_head, 17% bf16" rests on a model mislabel. On the real int4head the lm_head is int4 Marlin and the bf16 GEMM is **gone (0.5% / 0.0%)**. That bf16→int4 conversion *is* the int4head **+17.053%** gain (219.34→256.74 TPS) that landed in #801. There is no remaining lever.

### Per-owner bucket of the 114.75 ms (the Step-1 deliverable)

| owner bucket | w4a16-ct = #809 `wsr5i6qb` (bf16 head) | int4head (firing submission) |
|---|---|---|
| **(c) lm_head** — full-vocab `[M,2560]@[2560,262144]` | **`aten::mm` 114.75 ms / 17.0% (×59)** + device `ampere_bf16_s16816gemm` 111.83 ms / 16.6% (×40, grid=(4096,1,1)→N=262144) | **int4 Marlin GEMV** (the 0.75 ms / 6% in stark #798); `ampere_bf16` → 2.09 ms / **0.5%**, `aten::mm` → 0.13 ms / **0.0%** |
| (a) drafter (`gemma4_mtp`) | small in-graph projections (`ampere_bf16` grid=(4,1,16), no cpu_op); centroid `get_top_tokens` at decode is **T=1, size=1 graph ≈ 21 MB/step** (negligible) | same — negligible |
| (b) body | int4 Marlin (the 133/120 ms lines) — not bf16 | int4 Marlin 290.19 ms / 70.4% |
| (d) PLE / (e) embed | int4 Marlin / gather — not bf16 | same |

≈100% of the bf16 slice → **lm_head**. So the STOP condition is met, but for a cleaner reason than "drafter, dead per #786": the bf16 GEMM is the head, and on int4head it is **already int4**.

### Why this is airtight (3 independent confirmations of the fingerprint)
The bf16 lm_head reads the full `embed_tokens` `[262144, 2560]` bf16 = **1.3422 GB/tok**, ~**2.78 ms/tok** GEMV, bandwidth-bound (~480 GB/s on A10G, AI ≈ 1 FLOP/byte):
- **#809 trace** (`decode_window.pt.trace.json.gz`): the ×40 `ampere_bf16_s16816gemm` have **grid=(4096,1,1)** → N = 4096·64 = **262144** (full vocab), launched by `aten::mm`.
- **`9tcygwjf`** (int4head TPS run): `lmhead_bytes_gb_bf16 = 1.3422`, `gemv_ms_per_tok_bf16 = 2.777`; base labeled *"int4 body, bf16 lm_head, tied"*; quantizing it → `tps_int4_gain_pct = 17.053`.
- **`dpc36210`** (stark #798): `lmhead_bf16_ms = 2.724`.

### The model mislabel (root of the PR premise)
`wsr5i6qb` (cited as "the int4head path") has `config.model = google/gemma-4-E4B-it-qat-w4a16-ct` — int4 body + **bf16 tied lm_head**, not `int4_g32_lmhead`. That is why its lm_head shows as a 17% bf16 GEMM. stark #798's own "lm_head GEMV 0.75 ms (6%)" is the *int4* version and is internally inconsistent with a 114.75 ms bf16 lm_head on the same model — the resolution is that the two profiles ran on different checkpoints (bf16-head vs int4-head).

### Decisive re-profile of the **real int4head** (my Step-1 measurement)
`lm_head: cls=ParallelLMHead has_packed=True` (int4), GPU-busy 412.4 ms:
```
matmul_gemm            315.12 ms  76.4%
  marlin (all)         290.19 ms  70.4%   <- int4 body + int4 lm_head
ampere_bf16_s16816gemm   2.09 ms   0.5%   <- was 16.6% on w4a16-ct (tiny drafter projs only)
aten::mm (host op)       0.13 ms   0.0%   <- was 17.0% on w4a16-ct
```
The 17% bf16 slice does not exist on the int4head.

### Centroid drafter ruled out (measured, not assumed)
I instrumented `Gemma4Proposer._greedy_sample`: at decode it is called with **T=1 every step**, replaying the **size=1** centroid graph (21 MB gather, ~0.04 ms). The 1.342 GB only appears in the *one-time* size=64 graph **capture at load** (`_setup_centroids_cuda_graphs`, sizes [1,2,4,8,16,32,64]) — never per decode step. So the sparse centroid path is not the bf16 slice.

### Commands
```bash
cd target/
# real int4head CUPTI kernel profile (graphs ON, uniproc, K=6):
CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_BATCH_INVARIANT=0 \
  VLLM_USE_FLASHINFER_SAMPLER=0 MODEL_ID=/workspace/gemma_build/int4_g32_lmhead \
  DRAFTER_MODEL=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant NUM_SPECULATIVE_TOKENS=6 \
  PROFILE_TOKENS=192 PYTHONPATH=submissions/int4_mtp_bi0_int4head:. \
  .../python research/bf16_gemm_attribution/profile_int4head_kernels.py
# decode-time centroid replay T/size:        research/bf16_gemm_attribution/measure_centroid_replay.py
# #809 trace owner correlation (no GPU):      research/bf16_gemm_attribution/analyze_trace.py
```
- **Peak memory:** ~9.9 GiB model load (int4 body + int4 lm_head + bf16 drafter); KV cache 322,912 tok @ `gpu_memory_utilization=0.90`. Local A10G in-process (EXPLORATORY — no HF job).
- **W&B:** `7rvzl0l9` (group `bf16-gemm-attribution`).

### What happened
The PR's two outcomes (drafter→dead-per-#786; un-Marlin'd body/head→convert) both miss the actual owner. The bf16 GEMM is the **lm_head**, and the lever was **already pulled**: int4head quantizes it to int4 Marlin, which is exactly its +17.053% TPS gain over the bf16-head base, at PPL 2.0029 (within the 2.42 gate), GSM8K 0.92. Nothing left to convert on the firing submission.

### Suggested follow-ups
- **Close #812 as NULL/resolved** — no Step-2 arm; the only bf16 GEMM was the lm_head and it is already int4 on the firing path.
- **Correct the #809 reference** in the program notes: `wsr5i6qb` is a **w4a16-ct (bf16-head)** profile, not int4head — future "17% bf16 on int4head" claims will misfire on this.
- If a residual byte-lever is still wanted on the int4head decode wall, it is the **int4 Marlin body verify-GEMM (47.6%, 290 ms)**, not a bf16 op — that is the dominant remaining matmul bucket.

### Public evidence used
- vLLM 0.22.0 `gemma4_mtp` source: `Gemma4Proposer._setup_centroids_cuda_graphs` / `_greedy_sample` (`vllm/v1/spec_decode/gemma4.py`), `Gemma4MTPMaskedEmbedder` + draft lm_head tied to `embed_tokens` at backbone-dim (`vllm/model_executor/models/gemma4_mtp.py`).
- A10G HBM roofline (~469–518 GB/s) for the 1.3422 GB bf16 read = ~2.78 ms.
- No external/borrowed PR content; all numbers from `approval-gated-8gpu-20260613` runs (`wsr5i6qb`, `9tcygwjf`, `dpc36210`) and my local int4head re-profile.
