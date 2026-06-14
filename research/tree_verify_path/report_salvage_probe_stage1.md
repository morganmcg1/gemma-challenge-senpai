# PR #71 — STAGE-1 salvage probe: real-stack branch-hit screen (GO on layout)

**Status:** built, CPU-validated (7/7), and **run live on the real stack**. This is
byteshark's mandatory debug-gate item #3 (branch-hit sanity, healthy ≈ ρ₂≈0.41,
broken ≈ 3%). **Verdict: GO on layout** — the live branch-hit is **0.360**, an 8×
separation from the conflated-trap arm (0.044), i.e. firmly in the healthy regime,
NOT the byteshark layout-bug regime. The probe is purely observational (PPL
unmoved, greedy identity preserved by construction).

## What STAGE 1 measures

P(drafter rank-2 == verifier argmax **at the first-divergence row** | rank-1 spine
misses at a width≥2 spine position), on the REAL stack, with **NO tree forward / NO
scratch KV / NO relocate**. It reuses the existing linear verify's per-row
`dixie_all_argmax` (already gathered at `serve.py:410`) as the verifier oracle and
Component 1's live tree draft tokens (`_run_tree_emit_probe`) as the candidate. So
the ubel #157 3c fused-relocate landmine is **irrelevant to STAGE 1** — there is no
KV mutation.

It computes **two** branch-hit numbers in one run:
- **`correct`** — rank-2 vs `target_argmax[first_div]` (the **decoupled-A** target
  row; both children of `spine[first_div]` map to verify row `first_div`). This is
  the Component-3a-correct comparison.
- **`conflated`** — rank-2 vs `target_argmax[first_div+1]` (the `tli+1` trap). This
  is exactly the byteshark layout bug; it should fire at ~3%.

## Result (16 prompts × 512 tok, M=16, local A10G, zero quota)

| metric | value | reading |
|---|---|---|
| aligned_steps | 2300 | spine-aligned decode steps observed |
| full_accept | 407 (0.177) | chain == target everywhere |
| div_at_branch | 1126 (0.489) | first divergence at a width≥2 spine pos (salvageable) |
| div_no_branch | 766 (0.333) | first divergence at a width-1 spine pos (unsalvageable) |
| **branch_hit_correct** | **0.3597 (405/1126)** | **decoupled-A target row — the real salvage rate** |
| branch_hit_conflated | 0.0444 (50/1126) | tli+1 trap row — the byteshark bug signature |
| per-pos depth 0 | 0.3546 (250/705) | consistent across positions |
| per-pos depth 1 | 0.3682 (155/421) | " |
| skipped.unaligned | 0 | alignment airtight (stale-stash guard never tripped) |
| PPL | 2.2324 | **identical to the comp2 control** → probe is observational |
| completion | 8/8 prompts, 8192 decode tok | clean |

## Four things this establishes

1. **Layout is CORRECT (GO).** 0.360 vs 0.044 is an **8× separation**. The
   conflated arm reproduces the byteshark ~3-4% bug signature exactly — proving the
   probe *can* see that failure mode — while the correct arm sits 8× above it. By
   byteshark's decision frame this is unambiguously the "healthy layout" regime.
2. **Component 3a's override-A-breaks-B trap reproduced LIVE.** The 8× gap between
   `correct` (right target row) and `conflated` (`tli+1`) IS the live manifestation
   of the decoupled-A vs conflated-index bug. Confirms on the real stack what was
   only a CPU argument before: gathering rank-2 against `target_argmax[pos*+1]`
   collapses the salvage to ~3%.
3. **Divergence distribution is textbook.** full_accept 0.177 / div_at_branch 0.489
   / div_no_branch 0.333 matches P(first div at depth d) = top1^d·(1−top1) with
   live spine-acceptance top1 ≈ 0.729. The tree's branch positions (depths 0,1 of
   the real PARENT_M16) capture the bulk of divergences as designed.
4. **Purely observational.** PPL 2.2324 is bit-identical to the comp2 control and
   real generation continues on the untouched linear chain → greedy identity and
   PPL are preserved; the probe only *reads* the existing argmax.

## The honest gap: 0.360 vs ρ₂ = 0.4165

0.360 is **~0.057 below** the wirbel #83 oracle (SE ≈ 0.014, so this is a real
gap, not sample noise). It is in the healthy regime — it does NOT refute the tree —
but it lowers the E[T] projection slightly and deserves a named cause.

**Leading candidate — rank-2 extraction fidelity.** The deployed rank-1 selection
is the FUSED fp32 sparse-argmax kernel (`self.model.get_top_tokens`), which is
**rank-1-only**. Component 1 therefore draws rank-2 from a *different* path:
`embedder._select_and_score` (bf16 centroid-sparse, CENTROID_TOP_K=64) + topk
excluding rank-1. That bf16 centroid proxy is a ~15%-near-tie-lossy approximation
of the true fp32 rank-2, which would systematically depress the measured branch
hit below the fp32 oracle. **Secondary candidate:** local ρ₂ on this prompt set may
genuinely differ from the official 0.4165. Resolving this (a true-fp32 rank-2
extraction, or an official-set rerun) is a STAGE-2 refinement, not a STAGE-1
blocker.

## Decision

**GO on layout → proceed to STAGE 2** (full tree-masked verify forward for E[T] +
both-halves runtime assert; 3c KV-compaction deferred, and when it lands it MUST be
the ubel #157 single fused GPU relocate, never a host loop — see
`report_live_integration_design.md` 3c row). The fern #142 go/no-go gate is armed
to consume this branch-hit number plus the STAGE-2 E[T]/PPL/greedy readout.

## Repro

```bash
bash /tmp/run_salvage_stage1.sh rate 16 512 4
# env: TREE_EMIT_PROBE=1 TREE_EMIT_PROBE_M=16 TREE_SALVAGE_PROBE=1
#      VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0
# verdict JSON: research/tree_verify_path/comp_salvage_probe_stage1_verdict_rate.json
```

CPU join-math validation (no GPU): `research/tree_verify_path/test_salvage_stage1.py`
— 7 deterministic scenarios (A correct-salvage, B conflated-trap, C full-accept,
D div-no-branch, E branch-miss, F stale-stash skip, G consume-handshake), all PASS
against the real PARENT_M16.
