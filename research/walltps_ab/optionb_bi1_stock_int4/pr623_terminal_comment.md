STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["3igf1sq0","xg74ajc3"],"primary_metric":{"name":"tps_arm_b_bi1_median_local_walltps","value":152.29},"test_metric":{"name":"ppl_arm_b_bi1","value":2.0055},"tps_arm_a_no_bi":253.98,"tps_arm_b_bi1":152.29,"bi1_tps_cost_pct":40.04,"arm_b_beats_126378":true,"ppl_arm_b":2.0055,"verdict":"BI1_SPEED_VIABLE__beats_126"}

## Results

**Verdict: `BI1_SPEED_VIABLE__beats_126`.** BI=1 is expensive (**−40%** local wall_tps) but does **not** kill Option-B on speed: the #319-compliant arm still clears the locked 126.378 rung at the raw local wall_tps. The card's pessimistic "tanks toward ~120" did **not** happen; the optimistic "427→~350 (−18%)" was also too rosy — the real local cost is −40%.

### 1. Two arms (single-stream, batch=1, 128×512, n=3 fresh servers/arm, median-of-N)

| Arm | `VLLM_BATCH_INVARIANT` | attn path | wall_tps median | wall_tps mean±std | e_accept |
|---|---|---|---|---|---|
| **A** (fast, non-#319) | 0 (3D split-KV) | num_splits>1 on M=1 | **253.98** | 252.85 ± 2.05 | 3.81 |
| **B** (#319-candidate) | 1 (2D, num_splits=1) | forced 2D | **152.29** | 151.75 ± 0.97 | 3.83 |

- **BI=1 cost: −40.04% median** (Δ −101.69 TPS), A/B verdict **REAL** (op@N3=0.10%, raw-powered-MDE 0.080%, observed CI95 ±1.01%).
- **Acceptance is unchanged** (3.81→3.83), so the −40% is **pure per-step forward time**, not a drafter-acceptance regression under BI.

### 2. Decision number (instruction #5 — the headline)

**Arm B beats the locked 126.378 by +25.91 TPS (+20.50%)** at local wall_tps.

> Caveat (honesty): 152.29 is **local single-stream wall_tps**; 126.378 is an **official HF-Job tps** (`int4_g128_lmhead`, PR #4, a non-spec AR config). These are not the same measurement, and Arm B is a *spec* config vs an *AR* rung — so this is a **screening signal**, not an official head-to-head. The +20.5% local margin is large enough to justify a (human-approved) HF Job to get Option-B/BI=1's official number, but `official_tps=0` here by design.

### 3. PPL gate (instruction #3) — both arms, **spec-on, full Option-B stack**

`prompt_logprobs` worked under speculative decoding (no target-only fallback needed), so PPL is teacher-forced over the faithful served stack for both arms:

| Arm | `VLLM_BATCH_INVARIANT` | PPL | records | tokens |
|---|---|---|---|---|
| A | 0 | **2.00566** | 128/128 | 61,797 |
| B | 1 | **2.00553** | 128/128 | 61,797 |

- **Arm B PASSES** the ≤2.42 gate with huge margin (2.00553 ≪ 2.42; reference PPL ≈ 2.305).
- **BI=1 − BI=0 = −0.000124** (−0.006%): identical to reduction-order noise, exactly as predicted (same int4 W4A16 weights; BI only reorders reductions).

### 4. Attribution (instruction #4 — cost >10%, so required)

Where does the −40% (≈ +10.08 ms per spec cycle) come from? I microbenched the served `unified_attention` 2D-vs-3D split-KV at ctx∈{256,512,1024}, M∈{1,8}:

- **The M=8 spec-verify forward is identical in both arms** (Δ ≈ **+0.0003 ms/cycle** @ctx512). The served TRITON_ATTN kernel already forces 2D for `max_seqlen_q>1`, so verify-attention never used 3D split-KV → BI=1's `num_splits=1` is a no-op there.
- Only the **M=1 forwards** lose 3D split-KV under BI=1, upper-bounded at **+1.516 ms/cycle** (full-target M=1 proxy; the real draft head is only 4 layers, so this is a loose ceiling).
- → **Attention ≤ 15% of the cost (upper bound); deterministic-GEMM tax ≥ 85% (lower bound).**

**This refutes the card's prime suspect.** The −40% is **not** the verify-attention KV reduction serializing under `num_splits=1` — it's the **batch-invariant GEMM tax** (BI=1 pins matmul reductions to a fixed, non-split order). A future split-KV attention kernel would claw back ≤15%; the real lever is a batch-invariant GEMM that recovers throughput.

### Config / commands

- **Submission:** `submissions/int4_mtp_batchinv` (int4 W4A16 `google/gemma-4-E4B-it-qat-w4a16-ct` + gemma4_assistant MTP drafter `/tmp/qat-assistant`, K=7/M=8), vLLM 0.22.0, dev307.
- **TPS A/B:** `scripts/profiler/paired_tps_ab.py`, two arms differing only in `VLLM_BATCH_INVARIANT` (0 vs 1), `--wandb_group optionb-bi1-speed-cost`, n=3, 128 prompts × 512 out, seed 1.
- **PPL:** `research/walltps_ab/optionb_bi1_stock_int4/run_ppl_ab.sh` — serves the full spec stack per arm, teacher-forced `prompt_logprobs=1, max_tokens=1, add_special_tokens=false` over the official 128-prompt `ppl_ground_truth_tokens.jsonl`.
- **Attribution:** `research/walltps_ab/optionb_bi1_stock_int4/attn_2d_vs_3d_probe.py`.
- **Peak memory:** ~20.3 GiB (`GPU_MEMORY_UTILIZATION=0.90` of the 22.5 GiB A10G), KV cache 8.44 GiB / 336k tokens — **identical across arms** (BI does not change footprint). GPU pinned 1710 MHz, util 93–97%, no OOM/disk issue.
- **Local only — no HF Job, no submission.** `analysis_only=true`, `official_tps=0`, as carded.

### W&B

- **`3igf1sq0`** — `land/optionb-bi1-stock-int4` (A/B TPS run), group `optionb-bi1-speed-cost`.
- **`xg74ajc3`** — `land/optionb-bi1-closeout` (decision + PPL gate + 2D/3D attribution), same group. ([link](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xg74ajc3))

### What happened — honest analysis

- The central open question is answered: **BI=1 costs −40% locally, and Arm B (152.29) still clears 126.378 (+20.5%).** The #319-compliant Option-B is **not dead on speed** — so identity (stark #621) and speed both point to it being worth an official run.
- The cost is **larger than the card's optimistic −18%** but **far from fatal**. Crucially it's **not** an acceptance regression (3.81→3.83) and **not** the verify-attention reduction (Δ≈0) — it's the deterministic-GEMM tax (≥85%). That reframes the future lever: a batch-invariant **GEMM** that recovers throughput, not a split-KV attention kernel.
- PPL is a non-issue: 2.00553 vs 2.00566, both ≪ 2.42, identical to noise — BI=1 is quality-neutral on the same weights.

### Bug fix included in this PR

`run_ppl_ab.sh` originally omitted `VLLM_USE_FLASHINFER_SAMPLER=0`, so all three PPL servers crashed at `profile_run` building the flashinfer sampling kernel (`fatal error: curand.h: No such file or directory` — `/usr/local/cuda` is runtime-only locally; the header ships in the pip `nvidia/cu13` package, off nvcc's include path). The entire profiler family already sets this flag for exactly this reason (e.g. `spec_cost_model.py:52`). Added it (one line + a why-comment). Teacher-forced PPL reads `prompt_logprobs`, never the sampler, so the flag has **zero** effect on the number — it only lets the server stand up. This is a local-env workaround, not a change to the served model (the official `vllm/vllm-openai` image has full CUDA dev headers and is unaffected).

### Suggested follow-ups

1. **Official number for BI=1 Option-B:** open an HF-Job approval issue to convert the 152.29 local screening signal into an official `summary.json:tps` head-to-head vs 126.378. This is the only way to settle the local-wall-vs-official caveat.
2. **Chase the GEMM tax, not attention:** since ≥85% of the cost is the batch-invariant GEMM (and verify-attention is already 2D/free), the highest-leverage next experiment is a batch-invariant GEMM kernel (or a deterministic-but-split reduction) that recovers per-step forward time without breaking #319. A split-KV attention kernel would only recover ≤15%.
3. **Smaller-M sensitivity:** the M=1 draft-head forwards are where BI bites; a lower K (e.g. K=3–4) trades acceptance for fewer M=1 BI-taxed forwards — worth a cheap local sweep to see if the −40% shrinks at lower K.
