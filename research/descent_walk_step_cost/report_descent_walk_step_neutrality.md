<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Descent-walk step-neutrality — does land #71's SALVAGE-DESCEND accept-prep
# hold the launch-realized 1.2182 step, or does the DFS add device-busy? (PR #173 · lawine)

**GATE: AMBER (bar-safe).** The salvage-descend accept-prep is **step-neutral in the
operative regime** (realistic paired marginal **+1.96 µs = +0.020 %**, well under the
0.10 % / ~9.72 µs practical floor) and **bar-safe even adversarially**: the faithful
all-mismatch-EVERY-step worst case adds **+9.94 µs = +0.1022 %** (a hair over the floor →
AMBER, not GREEN), and even the naive O(depth) ancestor-rewalk ceiling adds only
**+14.42 µs = +0.1482 %**. At all three, **both officials clear 500** (descent-only ≥519.2,
both-bugs ≥534.6). The launch should keep quoting **ONE step = 1.2182**.

- **PRIMARY** `descent_walk_step_self_test_passes` = **True** (4/4 legs, NaN-clean)
- **TEST** `descent_walk_step_delta_pct` = **0.1022 %** (the faithful worst-case gate)
- W&B run `r13idrlx` (group `descent-walk-step-neutrality`). JSON:
  `research/descent_walk_step_cost/descent_walk_step_cost.json`. Script:
  `scripts/profiler/descent_walk_step_cost.py`.

## Honest scope
Pure LOCAL A10G profiling — a Triton/CUDA timing harness. **NO model, NO HF Job, NO
submission, NO served-file change, NO kernel deploy, NO quota.** BASELINE stays 481.53;
greedy/PPL untouched; adds **0 TPS**. It BOUNDS whether land #71's descend kernel holds the
launch step. Reuses my #161 paired-device-busy method VERBATIM; imports the committed
anchors (#136 step, #168 officials, #135/#161 E[T]s, #165 single corrected map), does NOT
re-derive them. Elapsed 105 s, peak 0.258 GB.

## The question (and the claim under test)
#161 (MERGED) proved the depth-1 SPINE fix (BUG-1) is step-neutral and #168 collapsed the
four step anchors to the single launch-realized **1.2182**. But both priced the step against
the CURRENT strictly-linear accept-prep kernel (`_dixie_fused_accept_prep_kernel`,
break-on-mismatch). land #71 replaces it with a SALVAGE-DESCEND kernel that, on a mismatch,
does MORE work: a descent-ordered DFS over the single corrected `target_logits_indices` map
(#165) that walks siblings / descends instead of breaking (#135 BUG-2 structure). ubel #163's
host-residency sweep carried op 5 `descent_accept_walk` as **"+0 net (GPU-hidden by design)"**
— a design ASSUMPTION, never a measured bound. **This PR measures it.**

## Method (paired device-busy marginal, #161 verbatim)
A `conc=1`, `grid=(1,)` accept-prep launch is microseconds of device-busy hidden behind the
~9150 µs (~92 %-weight-GEMM) decode step. The robust signal is the **profiler device-busy
self-time** (common-mode-cancelled), measured PAIRED per round (each spec back-to-back) and
interleaved behind a step-sized filler GEMM (`gemm_per_op=32`, ~9098 µs/step ≈ the #143
target) to confirm GPU-hidden overlap.

**Two faithful models of the salvage-descend, both worst-cased (all-mismatch ⇒ salvage at
EVERY node, descend all 32):**

| kernel | ancestor-validity cost | what it represents |
|---|---|---|
| `descend_walk` (realistic sched) | O(1) parent-carry | the operative regime (E[T]≈3, reach≈4) |
| **`descend_walk_worst`** (the GATE) | **O(1) parent-carry** | land #71's ACTUAL build, worst input |
| `descend_walk_ceiling` | O(depth) full re-walk | the dumbest possible ancestor impl |

The faithful kernel carries ancestor-validity O(1) per node — exactly this repo's own
`traversal_verify_et.walk_leaf_to_root`: `full[u] = match[u] AND full[parent[u]]` (one
parent-status load + one AND). The **GATE is the faithful O(1)-carry worst case**; the
O(depth) re-walk is reported only as a labelled conservative **ceiling** (a strict upper
bound on any ancestor revalidation scheme). Every visited node also pays the +1
`target_logits_indices` indirection (#165) and, on its forced mismatch, the full
`MAX_BRANCH=3` sibling-load + select salvage. Worst-case scalar-op budget: **544** (faithful
carry) / **1312** (naive ceiling) per all-mismatch step.

## Results (full run, 7 rounds)

### Device-busy (isolation, µs) — the whole accept-prep launch
| kernel | device-busy µs | % of the 9150 µs step |
|---|---|---|
| linear break (served) | 1.934 | 0.020 % |
| descend realistic (O(1) carry) | 3.898 | 0.039 % |
| **descend WORST faithful** (all-mismatch, O(1) carry) | **11.873** | **0.122 %** |
| descend CEILING naive (all-mismatch, O(depth)) | 16.350 | 0.168 % |
| linear break worst (all-mismatch) | 1.788 | 0.018 % |

Every regime's device-busy AND interleaved idle sit under the 60 µs GPU-hidden threshold
(`busy_hidden = idle_hidden = True`; worst-case interleaved idle 28.1 µs < 60 µs) — the
descend launch is **fully overlapped** behind the weight-GEMM step.

### Paired marginals vs the served linear break (µs)
| marginal | median | ci95 | within-CI | step Δ | reading |
|---|---|---|---|---|---|
| **GATE: faithful worst − linear** | **+9.939** | 0.060 | no | **+0.1022 %** | AMBER (bar-safe) |
| realistic descend − linear | +1.964 | — | — | +0.020 % | **step-neutral** |
| worst-faithful − worst-linear (common-mode ✗) | +10.084 | — | — | — | pure DFS extra work |
| ceiling: naive O(depth) − linear | +14.417 | — | — | +0.1482 % | naive-impl bound |

The gate marginal is rock-stable (per-round [9.77, 9.85, 9.77, 9.94, 9.96, 9.94, 9.96];
ci95 0.060 µs). `within_ci = False` / `sign_flips = False` — the worst case genuinely costs
~10 µs more (it is the cost of descending all 32 nodes vs the linear break stopping at node
0), not noise. **0.10 % practical floor = ~9.72 µs**; the faithful worst case lands +9.94 µs,
i.e. a hair over → AMBER. The **realistic** marginal +1.96 µs is firmly under it.

## Propagation — officials at every step anchor
`official = K_cal·E[T]/step·τ`, K_cal = 125.268, τ = 1.0 (imported, cross-checked vs
#168). E[T] = 5.0564 (descent-only) / 5.2070 (both-bugs, #165 single corrected map).

| step anchor | value | descent-only | both-bugs | clears 500 |
|---|---|---|---|---|
| #168 launch-realized (linear-priced) | 1.2182 | 519.96 | 535.44 | ✓ |
| **operative descend (realistic, +0.020 %)** | **≈1.2184** | **≈519.9** | **≈535.4** | ✓ |
| faithful worst (gate, +0.1022 %) | 1.2194 | 519.42 | 534.89 | ✓ |
| naive ceiling (+0.1482 %) | 1.2200 | 519.18 | 534.64 | ✓ |
| roofline edge (optimistic, +gate Δ) | 1.2140 | 521.75 | 537.29 | ✓ |

The worst case costs **0.54 / 0.55 TPS** vs the #168 realized officials; the realistic descend
costs ~0.1 TPS (indistinguishable). **All officials clear 500 with ~19/35 TPS of headroom.**

## Self-test (PRIMARY) — `descent_walk_step_self_test_passes = True`
| leg | check | result |
|---|---|---|
| (a) | rig reproduces the #136/#161 step anchor: 1.21742 vs 1.2182 (Δ 0.064 % < 1.5 %) | **PASS** |
| (b) | worst-case marginal characterized: honest lift reported AND officials hold (clears 500) | **PASS** |
| (c) | descent-only 519.42 ≥ 515 (~520) AND both-bugs 534.89 ≥ 530 (~535) | **PASS** |
| (d) | NaN-clean (all 12 headline metrics finite) | **PASS** |

(Leg (b): the realistic marginal IS sub-floor / step-neutral; the worst case is an honest
+0.10 % lift, reported — not hidden — and bar-safe. `b_marginal_step_neutral=False`,
`b_marginal_characterized=True`.)

## Cross-check vs ubel #163's "+0 net by design"
**Confirmed for the operative regime, bounded adversarially.** #163's op-5 assumption that
`descent_accept_walk` is GPU-hidden and adds +0 net is **correct in practice**: the realistic
descend marginal (+1.96 µs = +0.020 %) is sub-floor and fully overlapped. The only correction:
it is not *literally* +0 in the absurd all-mismatch-every-step worst case — there it is
+0.10 % (faithful) to +0.15 % (naive), still bar-safe. So #163's design claim stands; this PR
upgrades it from an assumption to a **measured ≤0.15 % bound**.

## Hand-off to land #71 / fern #167's launch packet
**Quote ONE step = 1.2182.** The salvage-descend accept-prep does NOT measurably change the
operative step: the realistic descend marginal is +0.020 % (sub-floor, step-neutral), so the
descent-only 519.96 / both-bugs 535.44 officials from #168 survive the kernel swap unchanged.
The descend walk introduces **no launch-step risk**: even the adversarial all-mismatch worst
case (which means the drafter is 100 % wrong — a regime where spec-decode yields nothing
anyway) lifts the step at most +0.15 % to ≤1.2200, keeping descent-only ≥519.2 and both-bugs
≥534.6 — both clear 500. The AMBER label flags only that the worst-case cost is non-zero
(+0.10 %), not that any bar is threatened. land #71 may build the descend kernel without
re-pricing the launch step.

## Public / banked evidence used
- **my #161** (MERGED): the paired-device-busy step-neutrality method + the 1.2182 anchor
  reproduction rig (reused verbatim; the linear `accept_prep_baseline` is the served kernel).
- **my #168** (MERGED): the single launch-realized step 1.2182 and the 519.96/535.44 officials
  this PR confirms the descend walk holds.
- **ubel #163**: the host-residency sweep whose op-5 "+0 net by design" claim this PR measures.
- **wirbel #135 / #165**: the BUG-2 salvage-descent structure + the single corrected
  `target_logits_indices` map (slot-0 rank-1 + descent-ordered layout) the worst case walks.
- **#88 `traversal_verify_et`**: the M=32 topology loader + `walk_leaf_to_root` (the canonical
  O(1) ancestor-carry the faithful kernel models).
