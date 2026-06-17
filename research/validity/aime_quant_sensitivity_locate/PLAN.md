# aime-quant-sensitivity-locate (#586)

**DIAGNOSTIC ONLY. `analysis_only=true`, `official_tps=0`. NO HF Job, NO `--launch`,
NO submission, NO served-file change, NO ship.** Local A10G only. #319 strict
greedy-identity is the live launch contract — NO FIRE.

## Question
Is the int4→AIME quantization sensitivity **CONCENTRATED** in a few body decoder
layers or **DIFFUSE** across all 42? Zooms WITHIN the BODY locus established by
stark #536 / kanna #539.

## Models (both local, loaded `dtype=bfloat16`, `trust_remote_code=True`)
- bf16 reference / restore source: `google/gemma-4-E4B-it` (100% reference).
  Loaded from canonical Hub snapshot `fee6332c1abaafb77f6f9624236c63aa2f1d0187`.
- int4-QAT body precision (= base_fullhead body): `google/gemma-4-E4B-it-qat-w4a16-ct`
  (compressed-tensors W4A16; lm_head + embeddings already bf16/tied; only the 42
  decoder layers' linear projections are quantized).
- Arch: 42 decoder layers (indices 0–41), hidden 2560, full-attn at 5/11/17/23/29/35/41.

## Method
1. **PRIMARY (teacher-forced, NO generation).** Fixed batch of AIME-style reasoning
   prompts + reference traces. Working model = int4 dequant-in-bf16. For each layer
   ℓ, restore ONLY layer ℓ's weights to bf16, measure reduction in final-logit
   divergence (KL + argmax-flip on reasoning tokens) vs the full-bf16 reference.
   `s_ℓ = D_int4 − D_restore_ℓ`. Report ranked sensitivity vector +
   `top5_divergence_fraction = sum(top5 s_ℓ)/sum(all s_ℓ)`, `top1_divergence_fraction`.
   Cross-check: per-layer hidden-state relative-L2 int4-vs-bf16 (one paired forward).
2. **STRETCH (time-gated).** Rank by step 1; restore top-k to bf16 for k∈{0,2,5,10,all-body};
   re-measure AIME pass rate (60 problems, greedy temp=0, **min_tokens=8 EOS-guard #541**).
   `n_layers_for_90pct_recovery`.
3. **Verdict.** `aime_collapse_concentrated` (bool): TRUE if ≤~5 layers carry most
   sensitivity (top5_divergence_fraction ≥ ~0.6); FALSE if diffuse.

## Caveat (documented)
Divergence is computed in bf16 arithmetic on dequantized weights — isolates the
WEIGHT-quantization error (the meaningful quantity for "which layers hurt AIME"),
not Marlin-kernel numerics. This is exactly what the PR asks (restore *weights*).

## Self-determinism
Re-run one config; assert byte-identical divergence. NaN-clean.
