# PR #466 — Realize the strict frontier e2e: 467.14 or collapse to 162?

**stark · group `strict-frontier-realize` · LOCAL A10G (sm_86) · MEASUREMENT + analysis ONLY.
NO HF job, NO submission, NO served-file change, NO deploy. `analysis_only=true`.**

## The decision-critical question (the strict-side twin of #452)

The realized **blanket-strict equivalence frontier 467.14** (denken #423 `5a6zq2yz`; house
re-anchor lawine #455 466.02 ± 0.22 `0r0ounl8`) is **NOT a wall-clock serve** — it is a
*composition*: `OFFICIAL_TPS / (1 + eta_attn_decode)` = `481.53 / 1.0308`, assuming forcing
the verify-path attention reduction order-preserving costs only **~3.08 % of decode**.

But the only strict config ever *measured* end-to-end is **M=1 AR at 161.70 official**
(lawine #438), because forcing strict serial verify (BI=1) kills the K=7/M=8 CUDA graph
(ONEGRAPH). So the strict frontier is either **~467** (the +3 % composition holds, the
M=8 cudagraph survives an order-preserving attention reduction) or it **collapses toward
~162** (order-preserving attention forces BI=1 / serial verify, cudagraph dies, ~66 % tax)
— exactly the way #452's relax projection collapsed from 498.6 → 466.20.

**The #452 lesson: a composed frontier number is a hypothesis until it survives end-to-end
realization. I am the one who showed composed ≠ realized. Realize it.**

## The lever (the strict-attention analogue of #452's `use_fp32_reduce`)

The deployed stack routes the M=8 spec-verify attention to vLLM's **3D split-KV
(FlashDecoding)** path (`submissions/*/splitkv_verify_patch.py`). The cross-segment
online-softmax merge (`reduce_segments`, `num_par_softmax_segments` KV partitions) is the
**non-order-preserving reduction**. The strict lever = force a **single KV segment
(`num_splits=1` / sequential-KV reduction)** so the attention reduction is order-preserving
(confirmed by #381: the last decode-width identity coin-flip is the `num_splits=1`
varlen-combine; a single-segment combine closes it).

**CONFIG-level forcing on the existing served kernel only.** If realizing an order-preserving
attention reduction requires a served-kernel rebuild/patch, STOP and flag it (Directive #3)
rather than patching the served source.

## What I will measure (realized end-to-end, NOT a composed re-derivation)

1. **Full served decode cycle, CUDA-graph captured, M=8 verify width**, two arms:
   - **permissive**: deployed 3D split-KV multi-segment attention reduction.
   - **strict**: forced single-KV-segment (`num_splits=1`) order-preserving reduction.
   Paired per-round differencing (N≥3, target N≥7), median + σ. Apply the MEASURED attention
   Δ to the banked decode cycle (CYCLE_WALL_US=7903, base 467.14) → `realized_strict_frontier_tps`.
2. **The decisive cudagraph question.** Does the M=8 batched spec-verify + ONEGRAPH **survive**
   order-preserving attention (`strict_frontier_is_e2e_measurable=True` → ~467) or does forcing
   it force BI=1/serial verify (`strict_frontier_collapses_to_m1=True` → ~162)? Realize-or-collapse.
3. **Confirm the variant is actually strict + reconcile vs the composition.** 128-prompt eval:
   byte-exact greedy identity (expect 1.000, 0 flips vs the strict int4 argmax; <1.0 ⇒ strict
   forcing incomplete = bug). `composed_vs_realized_drift = 467.14 − measured TPS`; `|drift| ≤
   σ_hw (4.8)` ⇒ composition HOLDS, else the #452 lesson repeats. PPL vs ≤2.42 gate.
4. **Self-test + honest W&B logging.** `realized_strict_frontier_tps`,
   `strict_frontier_is_e2e_measurable`, `strict_frontier_collapses_to_m1`,
   `strict_variant_identity_fraction`, `strict_variant_token_flips`, `composed_vs_realized_drift`,
   `strict_frontier_realize_self_test_passes`, `analysis_only=true`, `no_served_file_change=true`,
   `official_tps=0`, `ppl=2.3772`. Apply through the END-TO-END wall clock, not the isolated-op Δ.

## Anchors (banked)

- Deployed incumbent (NON-equivalent, identity 0.9966, 3 flips @ {11,18,118}): **481.53** /
  PPL 2.3772 (PR #52 `2x9fm2zx`).
- Composed blanket-strict frontier (the number to REALIZE): **467.14** (denken #423 `5a6zq2yz`);
  house re-anchor 466.02 ± 0.22 (lawine #455 `0r0ounl8`). `= 481.53/(1+eta)`, eta_attn ≈ 3.08 %.
- Only e2e-MEASURED strict config (collapse floor): **M=1 AR 161.70 official** (lawine #438); BI=1
  kills cudagraph/ONEGRAPH.
- Methodology twin: relax-prize #452 `daqrzr99` — composed 498.6 realized 466.20 (−0.94).
- σ_hw ≈ 4.8 TPS. PPL gate ≤ 2.42. Identity gate: strict byte-exact (THIS variant MUST pass: 1.000 / 0 flips).
