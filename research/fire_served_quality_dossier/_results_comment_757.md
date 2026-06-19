STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3ymlxjgl"],"primary_metric":{"name":"fire_pct_of_bf16_panel_mean","value":93.76338303004525},"test_metric":{"name":"int4_quant_factor_panel_mean","value":93.5938366261495}}

## Results — bf16 full-precision denominator (completes the %-of-original-model quality leg)

**Headline (panel mean):** *The int4 + MTP-spec-dec fire retains **93.8%** of the full-precision bf16 base across MMLU-Pro / GSM8K / AIME, decomposed as **93.6%** (int4-QAT quantization) × **100.2%** (speculative decoding).*

**The more useful headline (well-powered tasks):** *On GSM8K (n=300) and MMLU-Pro (n=250) the fire retains **97.0%** and **98.1%** of full precision. The panel mean is dragged to 93.8% only by AIME (n=30), where full-precision bf16 solves exactly **one** more problem (12/30 vs the int4 11/30) — inside the n=30 sampling CI.*

This adds the third arm the blog needs: a **full-precision bf16** denominator served drafter-OFF on the same vLLM 0.22.0 stack, turning #753's "% of the int4 base" into "% of the original full-precision model." The decomposition is exact per task: `fire_pct_of_bf16 = int4_quant_factor × specdec_factor`.

### 3-arm panel (sampled T=1.0, top_p=0.95, top_k=64; min_tokens=8 EOS-guard; VLLM_BATCH_INVARIANT=1)

| Task | Metric | fire (int4, drafter ON) | base (int4, drafter OFF) | **bf16 (full-prec, drafter OFF)** | int4_quant_factor (base/bf16) | specdec_factor (fire/base) | **fire_pct_of_bf16 (fire/bf16)** |
|---|---|---|---|---|---|---|---|
| **GSM8K** | acc | 0.8667 (260/300) | 0.8767 (263/300) | **0.8933 (268/300)** | 98.13% | 98.86% | **97.01%** |
| **MMLU-Pro** | acc | 0.6320 (158/250) | 0.6280 (157/250) | **0.6440 (161/250)** | 97.52% | 100.64% | **98.14%** |
| **AIME 2024** | maj@8 | 0.3667 (11/30) | 0.3667 (11/30) | **0.4000 (12/30)** | 91.67% | 100.00% | **91.67%** |
| AIME 2024 | mean pass-rate | 0.3125 | 0.3083 | **0.3542** | 87.06% | 101.35% | **88.24%** |
| | | | | **panel mean** | **93.59%** | **100.21%** | **93.76%** |

**Product check (exact, per task):** `int4_quant_factor × specdec_factor / 100` reproduces `fire_pct_of_bf16` to the digit on every row (e.g. MMLU-Pro 97.52% × 100.64% = 98.14%). Product of panel means = 93.79% ≈ panel-mean fire_pct_of_bf16 93.76% (mean-of-products ≠ product-of-means, so this is a coarse cross-check; the per-task identity is exact).

### Model-lineage honesty (denominator pinned)

- **Denominator used: `google/gemma-4-E4B-it`** — the original full-precision instruct model Google released. The submission's int4 target `google/gemma-4-E4B-it-qat-w4a16-ct` descends from it (original → QAT → W4A16 compressed-tensors). This is the denominator a skeptical blog reader means by "the original model," so `fire_pct_of_bf16` = "% of the original full-precision model," and `int4_quant_factor` bundles **QAT adaptation + W4A16 rounding** (the headline "int4-QAT quantization" cost).
- **Alternative (noted, not used): `google/gemma-4-E4B-it-qat-q4_0-unquantized`** — the QAT bf16 checkpoint the W4A16 was *directly* quantized from. Against it, `int4_quant_factor` would isolate W4A16 rounding **alone** (excluding QAT adaptation) and would read higher. I picked the plain `-it` because the blog claim is about the *original* model, not the QAT intermediate.
- **Served-arm proof:** `model='google/gemma-4-E4B-it', speculative_config=None, quantization=None, quantization_config=None, dtype=torch.bfloat16` + `[serve] SENPAI_REFERENCE_MODE active: forcing num_speculative_tokens=0 (drafter OFF)`. Clean full-precision, drafter-off, native bf16 (no Marlin) — matched to fire/base on every eval knob.

### What happened — honest read

- **Spec-dec is lossless, now anchored to full precision.** `specdec_factor` panel-mean = **100.21%**, reproducing #753 to the digit (fire/base unchanged: the bf16 arm doesn't touch that leg). The MTP drafter's accepted tokens are verified against the target's own distribution, so it moves task accuracy by ~0. This is the robust, well-powered finding.
- **int4-QAT quantization costs a small, consistent amount on well-powered tasks:** 1.9% (GSM8K) and 2.5% (MMLU-Pro). bf16 ≥ int4 on every task and metric (physically sensible — quantization can only hurt or be neutral; full precision is uniformly ≥).
- **The sub-97% panel mean is AIME n=30 noise, not a real quality cliff.** bf16 solves 12/30 vs int4's 11/30 — a **single-problem** majority flip = 3.3pp on maj@8, and the mean-pass-rate gap (0.354 vs 0.308) is well inside the ±~9pp n=30 CI. AIME contributes 2 of the 4 equal-weighted panel metrics, so this one problem pulls the mean from ~97.5% (large-n only) down to 93.8%. The "true" retention is most likely ~96–98%; I'm reporting the panel mean as-measured and flagging the cause rather than cherry-picking the large-n subset.
- **Hypothesis verdict:** the predicted "≳97% panel mean" is **met on the well-powered tasks (97–98%) but not on the equal-weighted 4-metric panel (93.8%)**, solely because of AIME (n=30) variance. The decomposition the blog asked for is clean and complete.

### Harness fix (score-neutral, flagged for review)

`research/downstream_quality_aime/aime_eval.py`: added a **transient-error retry** (up to 4 attempts, 2s/4s/6s backoff) around the chat-completions POST. It retries **only** `TimeoutError`/`URLError`/`ConnectionError`; HTTP 4xx/5xx are re-raised immediately. With `VLLM_BATCH_INVARIANT=1` + fixed per-request seed a retried request returns the **identical** completion, so it is score-neutral — pure resilience. Without it, the first bf16 AIME attempt (`--client-concurrency 16` → 16×k=8=128 decode seqs oversubscribing 16 server slots) had requests wait past the per-request timeout and crashed the whole maj@k run at rc=1 / wall=2401s. Fix = drop `--client-concurrency` to 2 (2×8=16 exactly fills the slots) + the retry; the rerun completed clean (rc=0, wall=2922s, extract_fail=0.0). fire/base were already complete, are unaffected (retry never changes outputs), and stay matched.

### Protocol / matched-arm knobs (identical to #753)
- Decode: `generation_config.json` sampling T=1.0, top_p=0.95, top_k=64 (lewtun #31). EOS-guard: `min_tokens=8` (#541), empty_rate=0.0 on all MMLU-Pro arms.
- `VLLM_BATCH_INVARIANT=1` + per-request seeds → each request's decode is batch-invariant, so `MAX_NUM_SEQS=16` (raised for eval tractability) and `--client-concurrency` leave per-request outputs unchanged → arms matched.
- Native torch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`; this box's CUDA toolkit ships no `curand.h` for the flashinfer JIT sampler). `CUDA_VISIBLE_DEVICES=0`.
- N: GSM8K 300, MMLU-Pro 250, AIME 2024 = 30 problems × k=8. One arm served at a time.

### Commands
```bash
cd /workspace/senpai/target
# bf16 full-precision denominator arm (drafter OFF, native bf16; AIME at cc=2 for the slow arm)
python3 research/fire_served_quality_dossier/run_dossier.py --arm bf16 \
  --bf16-model google/gemma-4-E4B-it --gpu-mem-util 0.93 --max-num-seqs 16 \
  --mmlu-n 250 --gsm8k-n 300 --aime-years 2024 \
  --aime-client-concurrency 2 --aime-request-timeout-s 3600 --tasks gsm8k,mmlu,aime
# re-aggregate the 3-arm panel + decomposition, log to W&B group fire_served_quality_dossier
python3 research/fire_served_quality_dossier/aggregate_dossier.py
```

### Run facts
- **W&B run id:** `3ymlxjgl` (group `fire_served_quality_dossier`, same as #753 `0eob690h`) — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/3ymlxjgl
- **Peak GPU mem (bf16 arm):** model weights **15.09 GiB** + KV cache **3.83 GiB** (152,622 tokens, 37.3× concurrency @ 4096 ctx) ≈ **~21 GB of the 23 GB A10G** at `gpu_memory_utilization=0.93`. The cap was raised 0.90→0.93 because the bf16 weights (15 GB) need more headroom than the int4 target (~4 GB); KV stayed comfortable. No HF Job.
- **Wall:** bf16 arm ≈ GSM8K 268s + MMLU-Pro 1013s + AIME 2922s (slow native bf16, no Marlin, cc=2) ≈ 70 min. fire/base reused from #753.
- **No HF Job; LOCAL served evaluation only.** `analysis_only=1`, `no_hf_job=1`, `official_tps=0` — this is a quality card, no TPS target, no baseline change.

### Suggested follow-ups
- **Tighten the AIME CI to nail the headline ≥97%** (the optional stretch). Add AIME 2025 (→ n=60) and/or bump k, re-run **all three arms** matched (~2–3 h: bf16 AIME alone is ~50 min). At n=60 the current 1-problem maj@8 gap shrinks to ~1.7pp and the panel mean should rise to ~96–97%. I did **not** run this speculatively — the PR flagged it optional and the decomposition is the deliverable; happy to launch if you want the single clean number for the blog.
- **Report the large-n retention separately** in the blog (GSM8K 97.0%, MMLU-Pro 98.1%) alongside the panel mean, since AIME n=30 is the only noisy leg.
