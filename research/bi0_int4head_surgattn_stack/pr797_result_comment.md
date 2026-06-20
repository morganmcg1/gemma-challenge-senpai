STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["n84urlmc"],"primary_metric":{"name":"local_decode_tps_stack","value":272.78},"test_metric":{"name":"ppl","value":2.0028}}

## Results — int4head × surgattn-3D: the two levers COMPOUND (rung-2 fire candidate)

**Verdict: GATE PASS.** STACK = **272.78 local decode TPS** ≥ 265, PPL **2.0028** ≤ 2.42, **128/128**. surgattn-3D keeps **89.4%** of its isolated gain on top of int4head → this is a clear compound, not an evaporation. Tagged a **#784-band candidate** (PPL-gated; byte-identity intentionally given up by the 3D path, as expected).

### Per-arm (same #788 harness, median of warm reps rep1/rep2, cold rep0 dropped)

| Arm | Attention path (live, serve.log) | Decode TPS (warm median) | reps (0/1/2) | PPL | completed |
|---|---|---|---|---|---|
| **CONTROL** = `int4_mtp_bi0_int4head` (force-2D) | 2D single-pass (`use_3d=False`) — force-2D line PRESENT | **257.39** | 257.42 / 257.37 / 257.42 | 2.0028 | 128/128 |
| **STACK** = `int4_mtp_bi0_int4head_surgattn3d` (force-2D OFF) | native 3D split-KV (flash-decoding) — force-2D line ABSENT (grep count 0) | **272.78** | 271.83 / 273.20 / 272.37 | 2.0028 | 128/128 |

- CONTROL reproduces the MERGED #788 number (256.74 expected; got 257.39) → **harness sanity-check passed** (same as my K=6=220.47-vs-official-218.02 cross-check).
- **Attention toggle verified in serve.log:** CONTROL prints `[int4_mtp_force2d] unified_attention wrapped: forcing 2D single-pass attention (use_3d=False)`; STACK does **not** (count 0) → 3D split-KV confirmed live on the M=1 decode.
- **Within-arm determinism (sha256 of decode_outputs.jsonl):** all 3 CONTROL reps identical hash; all 3 STACK reps identical hash. Each path is byte-deterministic across reps, so the cross-arm divergence below is a genuine, reproducible kernel-path difference — not run-to-run noise.

### Compounding analysis (the open question this card settles)

- STACK gain on top of int4head: **+5.98%** (272.78 / 257.39).
- surgattn-3D isolated (wirbel #785, bf16 head): +6.69% → **89.4% retained** once lm_head is already int4. The attention kernel is still a meaningful marginal cost after the lm_head GEMV is cheapened; the two levers attack different bottlenecks and stack near-multiplicatively.
- Absolute STACK 272.78 vs the PR's multiplicative projection (257.39 × 1.0669 = **274.61**): within 0.7%. vs the #788 bf16-head control (219.34): **+24.4%** ≈ the projected 1.170 × 1.067 = 1.248. **The "250–270 via stacking" projection to the human team is now MEASURED, not projected.**

### Identity-class (#784 gate) — STACK greedy token_ids vs CONTROL greedy

surgattn-3D is **NOT byte-identical** (expected; wirbel #785 saw the same class). Diff (any control rep vs any stack rep — within-arm deterministic so representative):

- **prompts diverged: 4 / 128 (3.12%)**
- **first-token flips: 0 / 128 (0.0%)** — the very first decode step is byte-identical on every prompt; this is pure deep-sequence autoregressive drift, not an immediate kernel-path flip.
- **token mismatches: 112 / 65536 aligned positions (0.171%)** — upper bound (counts all post-divergence positions).
- first-divergence locus (of the 4 that diverge): **min 392, median 456.5, max 510** of 512 — all very late in the sequence.

This is *cleaner* than surgattn-3D's isolated signal on the bf16 head (wirbel #785: 1.76% tok / 6.25% prompts), and PPL is flat at 2.0028 across both arms — but note PPL is teacher-forced / decode-path-partially-blind, so **the token_id diff is the real identity proof, not PPL.** Per the PR, surgattn-3D → #784 candidate (PPL ≤ 2.42, identity optional), tagged accordingly — **not** a reject.

### Gate decision

```
STACK TPS 272.78 ≥ 265   ✓
PPL      2.0028 ≤ 2.42    ✓
completed 128/128         ✓
→ rung-2 fire candidate. The levers COMPOUND.
```

### Exact command

```bash
# Build int4 g32 lm_head (CONTROL weights; STACK reuses the SAME build via model_id):
python submissions/int4_mtp_bi0_int4head/build_lmhead_quant.py \
  --src google/gemma-4-E4B-it-qat-w4a16-ct --out /workspace/gemma_build/bi0_int4head_g32 \
  --num-bits 4 --head-group-size 32

# Each rep = one fresh-server pass of the #788 harness (PPL 128 validity gate + 128-prompt decode):
bash research/_int8head_smoke/run_prevalidate.sh int4_mtp_bi0_int4head            8021 pr797_control_repN
bash research/_int8head_smoke/run_prevalidate.sh int4_mtp_bi0_int4head_surgattn3d 8022 pr797_stack_repN
# venv /tmp/senpai-venvs/20f658587e8a6643/bin/python ; CUDA_VISIBLE_DEVICES=0 ;
# VLLM_USE_FLASHINFER_SAMPLER=0 (native sampler, confirmed in serve.log) ; TRITON_ATTN ; MTP K=6 ; max_num_seqs=1 ; BI=0
```

### Peak memory

Identical for both arms (same `model_id` = `/workspace/gemma_build/bi0_int4head_g32`; only the attention kernel differs). From serve.log: model load **10.22 GiB** + KV cache **8.1 GiB** (322,912 tokens) + CUDA graph 0.05 GiB, under `gpu_memory_utilization=0.9` on the A10G (24 GiB → ~21.6 GiB reserved). No OOM.

### W&B

Run **`n84urlmc`** — group `bi0-int4head-surgattn-stack` — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/n84urlmc (`analysis_only=1, no_hf_job=1, fires=0`; carries both arms' per-rep TPS, the compounding analysis, the live attention-path flags, and the full identity diff).

### What happened

The hypothesis held. int4head cuts the un-amortized per-accepted-token lm_head GEMV read (1.342 → 0.378 GB), surgattn-3D speeds the M=1 decode attention kernel — orthogonal bottlenecks, and on the trustworthy bi0 harness they stack near-multiplicatively (89.4% of surgattn's isolated gain survives on top of int4head). The negative-result branch (3D gain evaporates once lm_head is cheap) did **not** materialize: attention is still a real marginal cost after the head is quantized. Identity behaves exactly as wirbel #785 predicted for the 3D path — late, rare, 0 first-token flips — so the stack is a PPL-gated #784 candidate, not a byte-exact ship.

### Suggested follow-ups

1. **Panel-gate the stack (MMLU-Pro / AIME / GPQA-Diamond)** once wirbel #791 clears surgattn-3D quality in isolation. The token diff is tiny and late, but #784 requires within-5%-of-base on AIME/MMLU/GPQA before fire — the 4 diverging prompts should be checked for grade-flips on the reasoning sets.
2. **One official a10g-small HF Job** to convert this local 272.78 into a leaderboard number — but only via a separate `Approval request:` issue + explicit human OK, and only after the quality panel passes. The submission is HF-launch-ready *except* `model_id` is the local build path `/workspace/gemma_build/bi0_int4head_g32`; it must be published to a private Hub repo first (same gate as #788).
3. **Coordinate the attribution with wirbel #791** — if wirbel's isolated surgattn number on the bf16 head shifts, the "89.4% retained" framing updates with it; wirbel owns the surgattn mechanism + quality gate.
