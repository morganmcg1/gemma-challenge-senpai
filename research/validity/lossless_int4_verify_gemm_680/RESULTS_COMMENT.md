STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"fires":false,"wandb_run_ids":["5iy1mhe4"],"verdict":"LOSSLESS_VERIFY_NEEDS_KERNEL","primary_metric":{"name":"verify_gemm_byte_identical_achievable","value":1},"test_metric":{"name":"lossless_verify_tps_cost","value":0.0}}

## Results

**Verdict: `LOSSLESS_VERIFY_NEEDS_KERNEL` — but premise-refuting.** The config sweep is decisive in an unexpected direction: **the deployed int4 g=128 Marlin GEMM is already byte-identical across batch width M** (M=1 AR vs M=6 verify), under *every* reduction-order knob. So no Marlin config "recovers" identity because identity was **never lost at the GEMM**. The width-(K+1) verify break that kanna #673 reports is **not** the Marlin GEMM — it is the M-dependent flash split-KV **attention** reduction. A width-invariant *attention* kernel is required; the Marlin epilogue needs no change. The recompute-rescue (stark #669) stays the only path until attention is pinned.

| scalar | value | meaning |
|---|---|---|
| **`verify_gemm_byte_identical_achievable`** | **1** | the int4 verify-GEMM is byte-identical to AR-width at M=6 for all 5 deployed shapes — under the *deployed* config (no change needed) |
| **`lossless_verify_tps_cost`** | **0.0** | the byte-identical GEMM config = the deployed default → zero added TPS. **But moot:** GEMM byte-identity does not make spec-dec lossless (see below) |
| `marlin_gemm_is_m_invariant_as_deployed` | 1 | g=128, deployed knob, M=6, all shapes: m_inv=1.0000, max_abs_diff=**0.0** |
| `fullforward/ar_vs_ar_token_identity` | **1.000000** (768/768) | instruction-4 linchpin: the M=1 AR reference is deterministic |
| `fullforward/break_per_position` | 0.00163 (5/3072) | full-forward M=6-verify argmax break vs M=1 AR (this IS attention) |
| `fullforward/seq_break_rate` | 0.078 (5/64 @ n_new=48) | extrapolates toward #673's 0.33–0.38 at length 512 |

### Method — two isolated legs (LOCAL, analysis_only, no HF Job, no served-file change)

**Leg A — isolated Marlin GEMM width microbench** (`research/validity/lossless_int4_verify_gemm_680/gemm_width_microbench.py`). Builds a real GPTQ-Marlin int4 weight at the exact deployed shapes (qkv 2560→3072, o 2048→2560, gate_up 2560→20480, down 10240→2560, **lm_head 2560→16384**), **group_size=128, symmetric, act_order=False** — the `int4_g128_lmhead` quant. For every shape, sweeps the reduction-order knobs and measures whether the width-M GEMM output byte-matches the per-row width-1 (AR) output for the same rows. 20 (shape×knob) cells × M∈{1,2,4,6,8} × 8 trials.

This closes the one gap in ubel #491's prior microbench (which used `group_size=-1`/M=8) and settles the standing #23-vs-#491 conflict at the exact PR-#680 spec (g=128, M=6).

**Config sweep result (g=128, M=6 verify) — every cell identical:**

| knob | `use_atomic_add` | `use_fp32_reduce` | m_inv @ M6 (all shapes) | max_abs_diff @ M6 |
|---|---|---|---|---|
| **deployed** | False | True | **1.0000** | **0.0** |
| atomic_on (split-K) | True | True | 1.0000 | 0.0 |
| fp16_reduce | False | False | 1.0000 | 0.0 |
| atomic_fp16 | True | False | 1.0000 | 0.0 |

- **Forcing split-K / atomic-add ON is inert** — it introduces *zero* M-variance at these shapes (the global-reduce epilogue's `slice_count>1` path is never activated because k_tiles×n_tiles ≤ SM count). `should_use_atomic_add_reduce` returns False at M=1, 6, **and** 8 for every shape (all n≥2048).
- Positive control valid: a perturbed input row flips the bytes (harness=0.875<1.0) for every shape/knob → the 1.0 is a real invariance, not a stuck comparator. Run-to-run = 1.0 (deterministic).
- `group_size=-1` control reproduces #491 exactly.

**Leg B — full-forward width isolation** (`verify_break_fullforward.py`, reusing #491's validated `phase_margin` geometry at M=6 on the in-scope public STEM prompts; loadable full-vocab int4 ckpt, same documented #491 fidelity caveat — the deployed pruned 16384-row head cannot load in vanilla vLLM). M=1 greedy AR generate → reference; width-6 re-forward (`max_num_batched_tokens=6`) → M=6-occupancy logprobs at the same positions; break = (argmax_M6 ≠ ref_tok).

- **AR-vs-AR token identity = 1.000000 (768/768)** — the reference is deterministic (instruction 4 ✓); the break is a *width* effect, not run-to-run noise.
- **`frac_steps_bitdiff` = 0.8955** — the M=6 verify top-1 logit differs *in bits* from M=1 AR at ~90% of positions. **Yet Leg A proves the GEMMs are byte-identical across M (max_abs_diff=0).** Therefore this 90% logit bit-difference **cannot be the GEMM** — it is the attention reduction, propagated through the M-invariant body+lm_head.
- **All 5/5 argmax flips are near-ties** (M=1 top1–top2 margin ≤ 0.125; min=0.0, i.e. exact bf16 ties / one-ULP gaps). Exactly the "near-tie argmax flip" the PR describes — but sourced from attention, consistent with PR #654's irreducible exact-tie residual.

### Answering the PR instructions

1. **Config sweep / break_rate per config** — the GEMM-isolated break_rate (=1−m_inv) is **0.0 for every config** (deployed, split-K on/off, fp32/fp16 reduce). No config changes it because the GEMM is already width-invariant. The full-forward break_rate (Leg B, 0.00163/position; all near-tie) is **unchanged by any Marlin knob** because the GEMM output is byte-identical under all of them.
2. **Config that recovers byte-identity → TPS cost** — the byte-identical GEMM config *is* the deployed default → `lossless_verify_tps_cost = 0.0`. This is **not the fire-unlock case**, because GEMM byte-identity does not deliver lossless spec-dec (the break is attention).
3. **Minimal kernel change (no config fixes it)** — *for the Marlin epilogue: none required* (already cross-M byte-identical). The actual minimal change is a **width-invariant attention reduction** (fixed split-KV count independent of M occupancy). Cost is the *attention*-pin, not a GEMM change. In-scope anchors bound it: ubel #491/#484 price an attention-only reduction pin at ≈5.10% tax (≈ −6.4 TPS on the 126.378 anchor), trending toward ≈free with a targeted fixed-split (#363). A counterfactual GEMM M-padding fix (researcher Option 1, pad M→8) would have cost ~15–30% at M=1 (HBM-bound per denken #676's 2.38 GB/token ÷ 469 GB/s, not 8×) — but it is **unnecessary**, since the GEMM is already invariant.
4. **AR-vs-AR = 0** — confirmed: 768/768 token-identical (Leg B) and run-to-run byte-rate 1.0 (Leg A). Any divergence is genuine width-sensitivity, not noise.

### Why `verify_gemm_byte_identical_achievable=1` yet verdict `NEEDS_KERNEL`

The two are not in tension — together they are the headline. The GEMM **can be** (and already is) byte-identical across M, so the literal answer to the primary metric is 1. But the PR's premise — that making the GEMM byte-identical makes spec-dec lossless "by construction" — is **refuted**: the GEMM was never the break source. Pinning a kernel that is already invariant buys nothing. The lossless-verify lever lives one op over, in attention.

### Commands

```bash
# Leg A — isolated GEMM width microbench (g=128 + g=-1 control), ~9s
CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/20f658587e8a6643/bin/python \
  research/validity/lossless_int4_verify_gemm_680/gemm_width_microbench.py --n-trials 8

# Leg B — full-forward width isolation (M=1 AR vs M=6 verify) + AR-vs-AR control, ~9 min
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /tmp/senpai-venvs/20f658587e8a6643/bin/python \
  research/validity/lossless_int4_verify_gemm_680/verify_break_fullforward.py \
  --n-prompts 64 --n-new 48 --det-prompts 16 --ctx-cap 512

# W&B log
WANDB_ENTITY=wandb-applied-ai-team WANDB_PROJECT=gemma-challenge-senpai \
  /usr/bin/python research/validity/lossless_int4_verify_gemm_680/log_wandb.py
```

- **Peak memory:** Leg A 0.80 GiB (GEMM-only); Leg B model-load 9.77 GiB (full-vocab int4) on one A10G.
- **W&B run:** [`5iy1mhe4`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/5iy1mhe4) (`analysis_only=1`, `official_tps=0`, `fires=0`; config-sweep m-invariance table attached).
- **Greedy identity:** never broken; baseline `int4_g128_lmhead` (126.378 / PPL 2.019 / 128-128) untouched.

### What happened — honest analysis

The hypothesis named the wrong op. The int4 Marlin GEMM **is not batch-width-sensitive** at the deployed g=128 / M=6 spec — measured byte-identical (max_abs_diff=0) across M for all body+lm_head shapes, under split-K on/off, atomic on/off, and fp32/fp16 reduce. The researcher digest corroborates the mechanism: `should_use_atomic_add_reduce`→False for all n≥2048 means the M-dependent global-reduce epilogue is never even entered, and **`VLLM_BATCH_INVARIANT=1` is a no-op for `ops.marlin_gemm`** (it only overrides aten ops). So there is no Marlin config — and no Marlin kernel change — that turns the spec-dec verify lossless.

The break is real (Leg B: M=6 logits differ in bits at 90% of positions, flipping the argmax at 0.16% of positions, all exact-tie/one-ULP) but it is sourced **entirely** in the attention split-KV reduction, whose split count is occupancy-dependent (M=1 AR vs M=6 verify). This reproduces and extends ubel #491's isolated-GEMM result to the exact deployed g=128/M=6 spec and resolves the #23-vs-#491 conflict in favor of #491: #23 measured the *full forward* (attention + GEMM both vary with M) and mis-attributed the flips to the GEMM; their flip rate (0.00521) nearly matches the attention-only rate here/in #491 (~0.0016–0.0052), confirming attention was always the driver.

**Net:** the single highest-impact framing of the PR (fix the kernel determinism, not the symptom) is still right — but the kernel is **attention**, not Marlin. Lossless spec-dec by construction requires a batch-invariant attention reduction; until then stark #669's recompute-rescue remains necessary.

### Public evidence used

- **Extends** ubel #491 (`reduction_sensitivity_census`, on this advisor branch) — its isolated-GEMM microbench (group_size=−1, M=8, max_abs_diff=0) is reproduced here and closed at the deployed **g=128 / M=6** spec; its attention-reduction attribution is confirmed end-to-end at M=6.
- **Refutes** the PR-#680 premise (sourced to kanna #673 / #122 as cited *in the PR body*) that the int4 Marlin GEMM is the batch-width-sensitive root cause. #673's full-forward break (0.33–0.38) is real but attention-sourced; my run_identity probe independently saw seq_exact≈0.31 on the served g=128 stack.
- **Literature:** Thinking Machines "Defeating Nondeterminism in LLM Inference" (batch-invariant-ops) — the batch-invariance theory; researcher-agent confirmed against vLLM 0.22.0 source that Marlin's atomic/split-K path is dimension-gated off here and `VLLM_BATCH_INVARIANT` does not intercept the Marlin custom op.

### Suggested follow-ups (not implemented — flagging only)

1. **Attention is the lossless-verify lever, not the GEMM.** Measure the served TPS cost of a batch-invariant / fixed-split-KV attention on the `int4_g128_lmhead` anchor and whether lossless spec-dec then clears AR break-even. This is the real "fix the kernel determinism" path.
2. **Near-tie don't-care band.** All flips are exact-tie/one-ULP (margin ≤ 0.125) — the PR #654 irreducible-tie residual. A tie-tolerant verify acceptance (accept the draft at sub-ULP margins) could moot most of the break without any kernel change. Cheaper than pinning attention.
3. **`verify_gemm_byte_identical_achievable=1` is a free, durable fact** for the program: the int4 body+lm_head are width-invariant by kernel property (distribution-robust), so any future spec/tree-verify work can treat the GEMM path as lossless and budget determinism effort solely on attention.
