STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["itrpyg25"],"primary_metric":{"name":"int4_divergence_m_sensitivity_self_test_passes","value":1},"test_metric":{"name":"projected_divergence_at_M16","value":0.007291666666666696}}

## Results

**The deployed 0.73% int4 divergence is an M=8 fact; at the tree build's M=16 it PLATEAUS — `projected_divergence_at_M16 = 0.007292` (identity 0.992708), bound `[5.10e-05, 0.014583]`, near-greedy SURVIVES.** The split-K FP-reduction-order mechanism, with my #232 locus correction, says the M=16 divergence does NOT degrade past M=8 except in a bounded tiling-boundary tail — so denken #236's λ-invariant-PPL assumption holds at the build's M, and the M=16 number is a one-probe confirm land #71's build must report.

### Primary + test metrics

| metric | value |
|---|---|
| **`int4_divergence_m_sensitivity_self_test_passes`** (PRIMARY) | **True** ✅ (all 5 checks) |
| **`projected_divergence_at_M16`** (TEST) | **0.007292** (plateau central; identity 0.992708) |
| bound | **[5.10e-05, 0.014583]** (lower = Curve-B decay tail; upper = 2× M=8 tiling-boundary tail) |
| bf16-floor cross-check (#221) | 0.010559 |
| near-greedy verdict (identity ≥ 0.99) | **SURVIVES** |

Self-test breakdown (PRIMARY), all True: `a_curve_through_both_anchors`, `b_M16_projection_has_explicit_bound`, `c_monotonicity_and_M1_to_M8_direction`, `d_near_greedy_verdict_with_threshold`, `e_key_scalars_finite`. NaN-clean over the full payload.

### The divergence(M) table (the deliverable)

| M | divergence (clean) | divergence (envelope) | identity | measured/projected | confidence |
|---|---|---|---|---|---|
| **1** | 0.000000 (det control) | 0.560776 (native-spec) | 1.000000 | **MEASURED** — two *different* quantities | high (flagged) |
| **4** | 0.007292 (plateau) | 0.031009 (decay) | 0.992708 | PROJECTED (interior) | medium |
| **8** | **0.007292** | 0.007292 | **0.992708** | **MEASURED** (#232) | high (controls 1.0) |
| **16** | **0.007292** (plateau) | 0.001715 (decay) | **0.992708** | **PROJECTED** | medium-high (one-probe-confirmable) |

Two readings, both anchored to the two hard measured points; both agree on the projection direction:
- **Curve A (clean per-token verify — the quantity the build runs).** Boundary `div_A(1)=0` (#232 determinism control `det_M1_vs_M1=1.0`: M=1-verify *is* M=1-AR), measured `div_A(8)=0.007292`. The clean batch-width order saturates once the GEMM enters batched mode (M≥2) → **plateau** at the M=8 floor. `div_A(16)=0.007292`.
- **Curve B (the PR two-anchor envelope).** Monotone-decreasing fit through (M=1, 0.560776 #114) and (M=8, 0.007292 #232): power `p=2.088`, exp `λ=0.620`. Projects M=16 to **[5.10e-05, 0.001715]** — i.e. *continued decay*, even smaller than M=8.

### Headline: PLATEAU, near-greedy survives

`projected_divergence_at_M16 = 0.007292` is the **PLATEAU central** call (Curve A). The bound `[5.10e-05, 0.014583]`:
- **lower 5.10e-05** = Curve-B exponential "more-batch-more-stable" continued-decay tail (optimistic);
- **upper 0.014583** = Curve-A tiling-boundary re-randomization tail — *only if* M=16 crosses a split-K tiling boundary and re-randomizes the near-tie flips vs M=1, capped at 2× the M=8 mass (cross-checked against the #221 bf16 floor 0.010559).

Near-greedy threshold = identity ≥ 0.99 (divergence ≤ 0.01). The central projection **0.73% SURVIVES** it (identity 0.9927, *identical* to M=8). The strict 0.99 line is at risk **only** in the pessimistic upper tail (1.46%, identity 0.9854 — still overwhelmingly near-greedy), and that tail requires a one-probe-confirmable tiling-boundary crossing.

### The locus correction (why the projection is confidently a plateau)

The PR frames the root cause as "the int4 Marlin split-K reduction order = f(M)." My #232 in-process diagnostic **corrects the locus**, and this is load-bearing for the M=16 call: all four int4-Marlin **body** GEMMs (qkv/o/gate_up/down) are **bit-exact across M ∈ {1,8}** (`max_abs_diff = 0.0` each) → the int4 body's split-K is **M-invariant** and contributes **zero** batch-width divergence. The residual 0.73% is the **bf16 tied lm_head + bf16 attention/norm** accumulation being batch-variant (below the #221 bf16 floor, 0.69×). Consequence for M=16: the int4 body stays bit-exact (0 contribution); only the bf16 lm_head's per-row K-reduction order can move — and that order is set by the hidden-dim K-partition (shared across all rows), **not** by M — so it holds from M=8 to M=16 unless M=16 crosses a tiling-config boundary. That biases the projection toward **plateau** with a bounded upper tail.

### The two-quantities caveat (honest)

The M=1 anchor (#114, 0.5608) and the M=8 anchor (#232, 0.0073) are **different quantities**, and I do not silently fit them as one clean curve. #114's `9q5yy9l1` is native-spec-vs-M1 (`reference_kind="unknown"`, onset signature *"FP-reduction near-tie flips"*) — it conflates draft-trajectory branching, an **upper envelope**. #232's 0.0073 is the isolated "hold weights, vary only M" batch-width effect. The clean curve's M=1 value is **0** (the #232 determinism control), not 0.5608; the "drop" 0.5608→0.0073 is the native-spec envelope collapsing to the clean batched floor. Curve B honors the PR's literal two-anchor instruction (and still gives M=16 ≤ M=8); Curve A is the mechanism-faithful clean curve (plateau). Both agree near-greedy survives at M=16.

### Comparison against baseline

- **Official baseline 481.53 TPS** (PPL 2.3772, 128/128; PR #52). **This leg adds 0 TPS — it is a projection, not a speed change.** Greedy/PPL path untouched (no GPU / vLLM / draw / served-file / submission change; CPU-only).
- Extends my own #232 (the M=8 measurement) along the M-axis; distinct from denken #236 (consumes the rate for PPL), stark #233 (f_priv), fern (card). **NOT a launch. NOT open2.**

### Exact command

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python research/validity/int4_divergence_m_sensitivity/int4_divergence_m_sensitivity.py \
  --self-test --wandb_group issue192-reading-calibration --wandb_name lawine/int4-divergence-m-sensitivity
```

(Run under `.venv/bin/python` so wandb resolves to the installed 0.27.2 — the repo's local `./wandb` logs dir is not a package and does not shadow the import. A no-wandb dry-run of the same script passed the self-test first.)

### Run facts

- **Peak memory:** 13.66 MiB (CPU-only analytic; `CUDA_VISIBLE_DEVICES=""`, no GPU touched)
- **W&B run ID:** `itrpyg25` (`wandb-applied-ai-team/gemma-challenge-senpai`, group `issue192-reading-calibration`, 38 summary keys + result artifact)
- **Anchors imported (NOT re-derived):** #232 `nxwv6pam` (M=8 div 0.007292, identity 0.992708, int4-body bit-exact across M); #114 `9q5yy9l1` (M=1 native-spec env 0.560776, read from the banked interlock report); #221 `6m40u2bg` (bf16 floor 0.010559, via the #232 report); land #71 (tree verify width M=16).

### What happened — honest analysis

**It worked, and the answer is HOLD (plateau), not degrade.** Pricing `divergence(M)` from the FP-reduction-order mechanism across M ∈ {1,4,8,16} gives a central M=16 projection that is *identical* to the measured M=8 value (0.73%), with a bound `[5.10e-05, 0.0146]`. The mechanism makes plateau the natural call: each output row's K-reduction order is set by the K-partition (shared across rows), not by M, so it saturates once the verify GEMM is batched (M≥2) and holds from M=8 to M=16. My #232 locus correction strengthens this — the int4 body is bit-exact across M, so the only M-mover is the bf16 lm_head, whose reduction order moves only if M=16 crosses a split-K tiling boundary (the bounded 2×-M8 upper tail). The honest band: (a) **projection, not measurement** — the M=16 confirm needs land #71's built tree-decode verify; I own the M-axis projection, not a re-measure. (b) The two anchors are **different quantities** (#114 native-spec envelope vs #232 clean) and I flag, rather than hide, that the clean M=1 value is 0; Curve B is the PR's literal envelope reading, Curve A the mechanism-faithful clean curve, and both keep near-greedy at M=16. (c) The pessimistic upper tail (1.46%) is the *only* path off near-greedy, it requires a tiling-boundary crossing, and even there identity stays > 98.5%.

**Consequence for the launch:** the near-greedy framing the downstream legs ride on is **safe at the build's M**. denken #236's λ-invariant-PPL assumption holds regardless (PPL is output-equivalence-pinned: the served stream is the int4 greedy stream, so even the 1.46% upper tail does not drift the served PPL). The M=16 divergence is a one-probe-confirmable readout (re-run the #232 M=1-AR-vs-M=16-verify identity at the tree width) — NOT free, but cheap once land #71's build lands.

### Hand-off (one sentence) to denken #236 + fern's card + land #71

> *The deployed 0.73% int4 divergence is an M=8 fact; projecting the split-K FP-reduction-order model (with #232's locus correction: int4 body bit-exact, bf16 lm_head is the M-variant locus) to the tree build's M=16 gives `projected_divergence_at_M16 = 0.007292` (bound `[5.10e-05, 0.014583]`, near-greedy **SURVIVES**), so denken #236's λ-invariant-PPL assumption holds at the build's M (PPL is output-equivalence-pinned regardless) — and land #71's build MUST report the M=16 divergence as a one-probe confirm (plateau vs the bounded tiling tail).*

### Public evidence used

Internal validity-chain leg (Issue #192 reading-calibration, group `issue192-reading-calibration`). Imported: my #232 (`nxwv6pam`) banked report, kanna #114 (`9q5yy9l1`) interlock report, #221 (`6m40u2bg`) bf16 floor, land #71 (tree build M=16). No public leaderboard/bucket reproduction (CPU-only projection; no submission, no draw).

### Suggested follow-ups

- **The M=16 confirm (owned by land #71's build):** once the tree-decode verify lands, re-run the #232 probe at M=1-AR vs M=16-verify (the tree width) to confirm plateau vs the bounded tiling-boundary tail. Cheap (same harness, +0 TPS).
- **Profile the bf16 lm_head split-K schedule at M ∈ {8,16}:** a one-shot in-process check of whether the lm_head GEMM selects the same tiling config (same BLOCK_M bucket / split-K factor) at M=8 and M=16 would collapse the bound to the plateau point (pins whether the upper tail is even reachable). This is the *locus* the residual divergence lives in.
- **Deployed lmhead12k checkpoint:** the M-curve is on the canonical Hub int4 + bf16-class lm_head; re-running on the flagship untied 12k-row lm_head at M=16 would confirm the same plateau on the deployed head (currently inferred from body bit-exactness + shared lm_head dtype).
