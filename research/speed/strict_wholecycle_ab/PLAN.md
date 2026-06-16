# PR #472 — Whole-cycle strict A/B: tighten realized #466 with in-graph overlap

**stark · group `equivalence-escalation-anchors` · LOCAL A10G (sm_86) · MEASUREMENT ONLY.
NO HF job, NO submission, NO served-file change, NO `train.py --launch`, NO kernel rebuild.
`analysis_only=true`.**

## The decision-critical question (follow-up #1 from #466)

#466 realized the blanket-strict equivalence frontier at **456.36 TPS (L=640 headline)**,
byte-exact (identity 1.0000, 0 flips), and refuted the collapse-to-162 hypothesis. But that
number is an **isolated-attention-locus Δ** (+422.9µs/cycle — the 7 full-attn reductions timed
ALONE) applied to the banked decode cycle. Isolation is the most *precise* capture of the lever,
but it **cannot capture in-graph attention/GEMM overlap**, so #466 is a **conservative lower
bound**: the true realized strict frontier lies in **[456.5, ≤467.14]**.

The "holds-within-σ vs optimistic-composition" verdict flips only if in-graph overlap hides
**≳28%** of the measured +422.9µs/cycle Δ. That fraction was unmeasured in #466. This PR measures
it directly with a **whole-cycle A/B**, the #452-identical construction, so the approval issue can
quote the **honest, overlap-captured** board number — not a lower bound that under-sells by ~11 TPS,
nor an optimistic composition that over-sells.

## The lever (config-level, no source edit — identical to #466)

The deployed M=8 spec-verify attention routes to vLLM's **3D split-KV / FlashDecoding**
(`num_par=16`, `max_seqlen_q→1` override) — a non-order-preserving cross-segment online-softmax
merge (identity 0.9966). The strict lever forces the kernel's **natural M=8 2D single-segment
sequential-KV reduction** (`max_seqlen_q=8 → use_3d=False`), order-preserving and byte-exact. This
is exactly what `VLLM_BATCH_INVARIANT=1` does at serve time (auto-gates-off splitkv per PR #122) —
**config-reachable on the existing served kernel, no served-source patch, no rebuild.**

## What I measured (whole-cycle, overlap-captured — NOT an isolated-locus Δ)

The full deployed MTP K=7 / M=8 verify decode cycle, single-stream ONEGRAPH CUDA-graph captured:

- **37-layer body**: self-built g=128 int4-Marlin GEMMs (`apply_gptq_marlin_linear → ops.marlin_gemm`,
  served shapes qkv/o/gate_up/down), value-independent BW so random weights reproduce deployed timing.
- **7 full-attention** (hd=512) served Triton `unified_attention` at indices [2,8,14,20,26,32,36],
  with the lever toggled permissive 3D ↔ strict 2D.
- **30 sliding** (hd=256) sdpa + **int4 12k lm_head**.

Two whole-cycle arms (**permissive** vs **strict**), paired per-round differencing, **N=21 rounds**,
median + σ. `whole_cycle_strict_delta_us = whole_strict − whole_perm` captures the strict tax
**under in-graph overlap**. The **isolated** #466 arm (7 full-attn reductions alone) is co-measured
on the same clock → `overlap_recovery_fraction = (iso_Δ − whole_Δ) / iso_Δ`. Realized via
`tps_from_added_us(Δ) = 481.53 × CYCLE_PERM(7666.83)/(CYCLE_PERM + Δ)` on the banked cycle.

KV sweep L ∈ {128, 384, 640}; **headline L=640** (deployed-faithful longest KV, #466's headline).

## Calibration guards (whole_cycle_perm_tps=481.53 is definitional — these are the real checks)

The permissive arm IS the deployed baseline (self-tax 0 → 481.53 by construction), so it is **not**
an independent calibration. The genuine guards are: (a) in-harness **isolated Δ reproduces #466's
+422.9µs within 12%**; (b) **body GEMM reproduces #450's 4152.96µs within 20%**. Both are self-test
conditions — if either fails the strict number is suspect.

## Verdict metrics

`whole_cycle_strict_tps`, `whole_cycle_perm_tps`, `whole_cycle_strict_delta_us`,
`overlap_recovery_fraction` (0 → stays ~456; ≥0.28 → reaches ~467),
`whole_cycle_holds_within_sigma_hw` (bool, σ_hw≈4.8153),
`realized_strict_frontier_best_estimate_tps` (the single honest point estimate for the approval issue),
`whole_cycle_strict_identity_fraction` (1.0), `whole_cycle_strict_token_flips` (0),
`whole_cycle_self_test_passes`, `analysis_only=true`, `no_served_file_change=true`,
`official_tps=0`, `ppl=2.3772`. Identity re-confirm is the **locus M-invariance check**, NOT the
128-prompt served census (denken #471's job — not duplicated here).

## Anchors (banked)

- #466 isolated-locus realization (conservative lower bound): **456.36** headline, identity 1.0000,
  0 flips, `composed_vs_realized_drift=+10.78` (`sxigz7dp`/`gmd8v9sw`).
- Composition being tightened against: **467.14** (denken #423 `5a6zq2yz`); re-anchor 466.02 ± 0.22
  (lawine #455 `0r0ounl8`). `= 481.53/(1+0.0308)`, eta_attn ≈ 3.08%.
- Relax-side precedent the A/B mirrors: **#452** (composed 498.6, realized −0.94, held within σ_hw).
- Deployed (non-equivalent, identity 0.9966): **481.53** / PPL 2.3772 (PR #52 `2x9fm2zx`).
- Strict collapse floor: M=1 AR **161.70** (lawine #438). σ_hw ≈ 4.8153. PPL gate ≤ 2.42.
