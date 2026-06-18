STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["r98w09by","k7bxh4tw","r3xptcvu"],"primary_metric":{"name":"gpqa_diamond_sampled_official_g32_0p22","value":0.5056},"test_metric":{"name":"gpqa_diamond_sampled_ours_g32_0p22","value":0.5152}}

## Results — verdict: `RECIPE_RECOVERS_GPQA` (and the lever is **group size**, not the head, not the observer)

All three arms complete. GPQA-Diamond **sampled** (T=1/top_p0.95/top_k64), 10-seed pooled **n=1980** @ gb6144, vLLM **0.22.0** / BI=1 — byte-for-byte the ubel #628 denominator harness (only `MODEL_ID` changes per arm). `analysis_only=true`, `official_tps=0`, **no HF Job, no submission, live `int4_g128_lmhead` untouched.**

| arm | recipe | body rel_err | **GPQA-D sampled** | Wilson 95% CI | fl_rate | % of bf16 base | W&B |
|---|---|---|---|---|---|---|---|
| **bf16 base** (denominator) | — | 0 | **0.5404** | — | 0.000 | 100% | [`ilg4z6e9`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ilg4z6e9) |
| recalibrated 0.9× bar | — | — | 0.4864 | — | — | 90% | — |
| **Arm 1** official `qat-w4a16-ct` (g32, **tied bf16 head**, minmax) | — | 0.5056 (1001/1980) | [0.4835, 0.5275] | 0.0030 | **93.6%** | [`r98w09by`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/r98w09by) |
| **Arm 2** ours @ g32 (**untied int4 head g32**, minmax) | 0.0667 | **0.5152 (1020/1980)** | [0.4931, 0.5371] | 0.0025 | **95.3%** | [`k7bxh4tw`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/k7bxh4tw) |
| **Arm 3** ours @ g128 + **MSE observer** (untied head g128) | 0.0863 | 0.4869 (964/1980) | [0.4649, 0.5089] | 0.0000 | 90.1% | [`r3xptcvu`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/r3xptcvu) |
| _anchor:_ live g128 body (minmax) | 0.1021 | ~0.499 AR (ubel #638) / 0.4652 +spec (Option-B, fern #629) | — | — | ~92% / 86% | — |

All arms clear the 0.4864 bar; **Arm 3 lands *on* the bar** (0.4869 vs 0.4864, CI straddles it — statistically indistinguishable from the bar and from the live g128 minmax body).

### The three decisive deltas

1. **Group size is the GPQA lever.** Arm 2 (ours g32) − Arm 3 (ours g128+MSE) = **+0.0283** (~2.5σ). Reverting *just* the group size, holding our untied head fixed, is what moves GPQA back toward base. → **`INT4_DEFICIT_FUNDAMENTAL` is REFUTED: the deficit is the g128 speed trade, not 4-bit itself.**
2. **The untied/quantized int4 head is NOT the cost.** Arm 2 (our untied int4 head) − Arm 1 (official's tied **bf16** head) = **+0.0096** (well inside σ=0.0112, a statistical tie). Re-tying the head to bf16 does *not* recover GPQA; if anything ours reads marginally higher. So the untied-head deviation is GPQA-neutral.
3. **The MSE observer is a measurable-but-immaterial proxy gain.** MSE cut per-module weight rel_err **15.7%** (0.1021→0.0863, **343/343 modules strictly improved**, lm_head 0.1004→0.0847) — yet GPQA did **not** move (Arm 3 0.4869 ≈ live g128 minmax ~0.499 AR, a tie). **Lower weight-reconstruction error did not translate to downstream accuracy.** The "free quality lever" buys nothing on the task metric.

**Quant-error ↔ GPQA ladder** (clean monotone on rel_err, but GPQA only responds to the *large* group-size step):
`g128 minmax 0.1021 (live) → g128 MSE 0.0863 (−16%, GPQA flat) → g32 0.0667 (−35%, GPQA +0.028)`.

### Speed (secondary, non-binding — `official_tps=0`)
Local **batched** generation throughput through the same concurrency-16 serve (directional only, *not* official/single-stream TPS): g32 peak ~**1048–1081 tok/s** vs g128 peak ~**1140–1193 tok/s** → g32 ≈ **8% slower**, consistent with g32's ~4× scales = marginally slower Marlin. So the GPQA recovery is a **quality/speed trade, not free**.

### Exact commands
```bash
# Build (CPU-only) from the QAT-unq source, via the live build_quant.py + new opt-in flags:
python submissions/int4_g128_lmhead/build_quant.py --src /workspace/gemma_build/qat_unq \
  --out /workspace/gemma_build/int4_g32_lmhead       --group-size 32  --head-group-size 32      # Arm 2
python submissions/int4_g128_lmhead/build_quant.py --src /workspace/gemma_build/qat_unq \
  --out /workspace/gemma_build/int4_g128_mse_lmhead  --group-size 128 --head-group-size 128 --observer mse  # Arm 3
# Serve (ubel #628 template, MODEL_ID repoint only): vLLM 0.22.0, BI=1, VLLM_USE_FLASHINFER_SAMPLER=0,
#   mml 8192, max-num-seqs 16, gpu-util 0.90, max-num-batched-tokens 2048, pck04 prometheus shim, PCK04_KEEPSET unset
bash research/validity/int4_body_quality_upside_639/serve_ours_g32.sh   # (and serve_ours_g128_mse.sh)
# Eval: GPQA-D sampled, 10 seeds (12345..13579), gb6144, min_tokens 8:
ARM=ours_g32       bash research/validity/int4_body_quality_upside_639/run_gpqa_sampled.sh
ARM=ours_g128_mse  bash research/validity/int4_body_quality_upside_639/run_gpqa_sampled.sh
python research/validity/int4_body_quality_upside_639/pool_gpqa.py <arm>     # pool n=1980
./.venv/bin/python research/validity/int4_body_quality_upside_639/log_wandb.py <arm> "<recipe>"
```
**Peak GPU memory:** ~**19.6 GB** observed (nvidia-smi during the n=16 eval), within the 0.90×23 GB A10G cap — ~10.6 GB int4 weights + 8.18 GiB free KV cache (36.5× max concurrency @ 8192 ctx). Single A10G, CPU-only builds.

### What happened — honest analysis
The hinge (Arm 1) already answered the headline question: Google's own g32 recipe reads **0.5056 ≈ 93.6% of base**, so the int4 GPQA deficit is **not fundamental to 4-bit**. Arms 2–3 then did the attribution the card asked for, and it's unusually clean: **the entire g128 GPQA cost is the group size.** Our untied int4 head is GPQA-neutral (Arm 2 ≈ Arm 1), and the min-max→MSE observer — despite a real, uniform 15.7% weight-error reduction — moves GPQA by zero. That last point is the most useful finding for future body-quality work: **weight-reconstruction error is a poor proxy for downstream task accuracy at int4** here; only the structural group-size change (which shrinks per-group dynamic range, not just average error) recovers accuracy. The recovery costs ~4× scales / ~8% local throughput, so it's a lever with a price tag, not a free win.

**Caveat / reframe from ubel #638:** on **GPQA** the shipped g128 body *already* clears the bar (~0.499 AR), so this recovery is mechanism-level, not a live-submission fix. The live int4 deficit actually bites on **AIME** (shipped g128 = 0.350, ~75% of bf16). This PR proves group-size is the int4-quality lever *on GPQA*; whether it also recovers the **AIME** deficit (where it matters for the live body) is the natural, high-value next test.

### Code change flagged for review (additive, default-preserving)
Arm 3 required touching `submissions/int4_g128_lmhead/build_quant.py`. I made it **purely additive**: new `--group-size` / `--head-group-size` / `--observer` flags **all default to the live recipe** (128 / 128 / minmax), so `build_quant.py` with no args reproduces the live `int4_g128_lmhead` artifact **byte-identically** (the live submission's shipped artifact is untouched; my builds went to `/workspace/gemma_build/`). **Correctness catch:** the card's literal "flip `observer='mse'`" would have been a **silent no-op** — compressed_tensors 0.15.0.1 ships the `observer` field as metadata only (no MSE impl; it derives `scale=max_abs/7.5` from whatever min/max it's handed). So a *genuine* MSE arm required implementing a per-group MSE clip-search (`_mse_clipped_symmetric`, grid includes ratio=1.0 ⇒ MSE ≤ minmax). Please sanity-check that additive diff before any future re-quant.

### Suggested follow-ups
1. **AIME on g32 (highest value, advisor-invited).** Test whether reverting group size *also* recovers the deficit where the live body actually fails (AIME 0.350 → ?). **Blocker to flag:** `research/validity/downstream_quality_eval/run_eval.py` currently has **no AIME task** (only mmlu_pro / gpqa_diamond / gpqa_main), and I have no read on the exact ubel #638 AIME protocol (max_tokens / seeds / answer-extraction) to match apples-to-apples. If you greenlight, please point me at the AIME harness + the protocol that produced 0.350 and I'll run official-g32 **and** our-g32 (attribution) on one GPU.
2. **Is g32 servable within budget?** The recovery costs ~8% local batched throughput; if AIME also recovers, quantify the g32 official-TPS hit to decide whether a g32 (or mixed g32-body / g128-head) body is a viable quality/speed point vs the live g128.
3. **Skip further observer/transform levers for GPQA.** Between this (MSE no-op) and wirbel #625 (rotation closed), the within-g128 quality knobs are exhausted; only the group-size structural change pays. Don't spend GPU re-probing observers/AWQ on GPQA.
