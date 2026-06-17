# base_fullhead GPQA-Diamond gate margin: CI-robust under the sampling protocol? (PR #589)

**Analysis-only, NO FIRE.** `analysis_only=true`, `official_tps=0`. No HF Job, no
`train.py --launch`, no `/v1/jobs:run`, no submission, no served-file change. Local serve +
inference only, on the assigned A10G. `--wandb_group gpqa-ci-tighten`.

## The soft spot this card pins down
The "quality satisfied" leg of the convergence claim rests on `base_fullhead` clearing the four
re-anchored quality gates (ubel #580 / land #581, merged). GPQA-Diamond is the thinnest: point
estimate **0.4798 vs the ≥0.471 bar = +0.009 margin**. GPQA-Diamond is only n≈198, so a +0.009
point margin may sit inside the sampling noise. If the CI-lb (or a plausible single seed) dips
below 0.471, "quality satisfied" is NOT robust on GPQA — the single load-bearing soft spot.

The program's gate point (0.4798) was measured under **greedy** (ubel #564, run `7bi4e2ne`).
But downstream evals are **NOT greedy** (lewtun #31): the model's own `generation_config.json`
is `do_sample=true, temperature=1.0, top_p=0.95, top_k=64`. This card re-measures GPQA-Diamond
under that **sampling** protocol across many seeds and asks whether the gate is CI-robust.

## What is already known (internal anchors, all on this advisor branch)
- **ubel #564 (`7bi4e2ne`, greedy):** base_fullhead GPQA-D 0.4798 (95/198), Wilson 95%
  [0.411, 0.549]. Proved the n=198 Wilson half-width (~0.069) makes a CI-lb on a ~0.48 point
  **un-tightenable** — would need n≈830 to clear 0.471 even if the true gap were zero.
- **#574 (paired bootstrap/Beta):** confirmed the un-passable-by-construction Wilson finding
  on the 198-item ceiling; base_fullhead↔denominator GPQA gap is sampling noise (McNemar p≈0.76).
- **kanna #563 (sampling, PLAIN vLLM serve):** GPQA-D sampled, 3 seeds → mean 0.4848,
  sd 0.046, vals [0.4747, 0.5354, **0.4444**]. The worst seed (0.4444) already dips below 0.471.
  Proved **min_tokens=8 is a mechanical no-op** on GPQA CoT (empty_rate=0 everywhere).

This card extends #563: the FAITHFUL `base_fullhead` serve recipe (not plain vLLM), **≥5 seeds**
(target 10), the **absolute** ≥0.471 bar (not the relative 0.90× ratio), and a proper CI-lb +
worst-seed verdict.

## Design
- **Serve = base_fullhead anchors recipe** (ubel #564 ARM_ENV): dev307 build
  (`vllm-0.22.1rc1.dev307+g3e8afdf78`) + `serve_inject` sitecustomize (prometheus shim +
  `FA_SLIDING=1` flash routing + `SURGICAL_ATTN_USE_3D_OFF=1` 2D order-preserving attention) +
  `PLE_FOLD_EMBED_SCALE=1`. Checkpoint = local int4-W4A16-g32 QAT snapshot
  `google/gemma-4-E4B-it-qat-w4a16-ct` (native 262k bf16 lm_head; `lm_head` is in the quant
  `ignore` list). `VLLM_USE_FLASHINFER_SAMPLER=0`, max_num_seqs=16, max-connections 16.
- **Eval = ubel #511 `run_eval.py`** (inspect_evals 0.14.0, inspect_ai 0.3.240), task
  `gpqa_diamond` (full 198), dataset_seed=12345 (byte-identical to #563/#564; verified via
  `prompt_sha`). Per-request sampling params **temp=1.0 / top_p=0.95 / top_k=64**,
  `--min-tokens 8`, `--max-tokens 3072`, `--sampling-seed` varied over seeds 0..9.
- Server stays up across all seeds (serve-once / eval-many).

## Statistics (report every lens, no thumb on the scale)
Same 198 items each seed; decode stochasticity is the only moving part.
- **per-seed accuracy**, **mean ± std** (sample, ddof=1), **worst single seed** (min).
- **Item-level Wilson 95% CI-lb at n=198** on the pooled point — the irreducible binomial
  (#564/#574 lens; ~±0.07; un-tightenable on the 198-item Diamond ceiling).
- **Clustered bootstrap 95% CI-lb over the 198 items** (resample items, each item's
  mean-over-seeds correctness; B=20000) — the statistically correct CI on the population
  sampling-protocol accuracy. **Primary CI-lb.**
- **Pooled Wilson 95% CI-lb** over all K×198 draws — anti-conservative (ignores item
  repetition); a "tightest defensible single number" reference.
- **Seed-mean t-CI 95% lb** over the K seed accuracies — decode/seed-noise-only lens.

## Verdict
`gpqa_ci_lb_clears_0471` (bool) = does the **primary CI-lb (clustered bootstrap)** AND the
**worst single seed** stay ≥ 0.471? Also report the per-lens clears-bools and, if it does NOT
clear, how far under each lens lands + `n_for_wilson_cilb_pass_at_point`. If it clears
comfortably under every lens, the soft spot is retired.

Expected: genuinely uncertain — the thinnest gate. Report straight.

`primary_metric = gpqa_ci_lb_clears_0471` (or the binding CI-lb margin to 0.471).
NO HF FIRE — hardens the quality half of the two-gate convergence claim; does not touch speed.
