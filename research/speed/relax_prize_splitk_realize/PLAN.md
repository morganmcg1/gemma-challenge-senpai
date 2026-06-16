# PR #452 — Realize the relax-equivalence prize: does greedy-unsafe split-K reach 498.6?

**stark · group `relax-equivalence-prize` · LOCAL A10G (sm_86) · MEASUREMENT + analysis ONLY.
NO HF job, NO submission, NO served-file change, NO deploy.**

## The decision-critical question
ubel #450 (`c5oyb7gv`) PROJECTS that a "realistic split-K" int4 verify-GEMM re-tile reaches
**498.6 TPS** (= `467.14 × CYCLE/(CYCLE − gemm_us×0.12)`, a *literature-assumed* 5–12%-of-GEMM
recovery; `realistic_splitk_greedy_safe=False`). That is **+17 over the deployed 481.53**. But my
own #433 MEASURED a Triton split-K at **−5.82 TPS**, and #130's committed 192-config Triton
re-tile sweep found **0.0% speedup (every config slower than Marlin)**. Is #450's +17 REAL or does
it COLLAPSE to ≈0/−6 when realized END-TO-END? This determines the whole human escalation.

## What I will measure (realized, NOT modeled-in-isolation)
1. **End-to-end full verify-cycle wall-clock** (37-layer self-built g=128 Marlin body, the SAME
   `apply_gptq_marlin_linear → ops.marlin_gemm` ubel #450 profiled), graph-captured, N≥7 rounds,
   median+σ, for:
   - **strict**: `use_fp32_reduce=True` (served default)
   - **relax**: `use_fp32_reduce=False` (the in-wheel FP-reassociating split-K reduction — the ONLY
     realizable served-numeric split-K lever, #448) → `realized_relax_prize_tps`.
   - **Triton re-tile confirmation**: a focused split-K sweep (SPLIT_K∈{2,4,8}) on gate_up/down,
     best achieved BW vs Marlin (reproduce #130's "slower than Marlin" on THIS pod), fed end-to-end.
   TPS via the banked cycle anchor (CYCLE_WALL_US=7903, base 467.14). Apply through the END-TO-END
   frontier, not the isolated-op Δ (the #433/#437/#442 trap).
2. **Reconcile vs #433 (−5.82) + #450 (+31/498.6) + #130 (0%)** — one clear sentence.
3. **Identity cost of the relax config** on the 128-prompt eval: byte-exact greedy identity
   fraction (vs strict int4 argmax), token-flip count, PPL (vs ≤2.42 gate). If identity stays
   1.0 / 0 flips → flag LOUD (`relax_prize_is_effectively_strict=True`).
4. **Self-test + honest W&B logging** of every required metric. `analysis_only=true`,
   `no_served_file_change=true`, `official_tps=0`.

## Anchors (banked)
- Deployed incumbent **481.53** / PPL 2.3772 (PR #52 `2x9fm2zx`, non-equivalent, 3 flips).
- Realized blanket-strict frontier **467.14** (denken #423 `5a6zq2yz`).
- Roofline relax-prize **498.6 / 510.87** (ubel #450 `c5oyb7gv`, `realistic_splitk_greedy_safe=False`).
- Prior split-K: −5.82 realized (stark #433); fp32_reduce=False +0.64 GEMM breaks 3/4 shapes (stark #448).
- σ_hw ≈ 4.8 TPS. PPL gate ≤ 2.42. Identity gate: strict byte-exact (this card measures the COST of relaxing it).
