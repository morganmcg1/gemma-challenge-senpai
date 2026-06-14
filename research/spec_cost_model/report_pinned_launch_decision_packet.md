# Pinned-operating-point launch decision + readiness packet (PR #167)

**Gate: ARMED** — `launch_packet_self_test_passes = 1` (all four assertions). Instantiating the decision-geometry arc at stark #156's **pinned drop** (1.80% descent-only / 1.86% both-bugs) and ubel #154's **realistic bar** (E[T]≥4.809, step 1.2047), the full-recovery corner (λ=μ=1) projects **descent-only 519.6 TPS @ P(clear-500)=96.3%** and **both-bugs 534.8 TPS @ 99.9%**. Because the pinned drop (1.80%) sits far below #162's ~6% BUG-1 threshold, the recommended first shot is **descent-ONLY, BUG-1 deferred** — and its margin (519.6) EXCEEDS #162's 511.1 (which sat at the harsher GT-4.3%). `descent_only_p_clear500_at_pinned_drop = 0.9630`.

This is a **pure-analytic CPU synthesis**: it adds 0 TPS, does not touch the build, files no issue, and BASELINE stays 481.53. It composes the banked legs — #155 consolidator (GO/NO-GO + CI + binding-leg), #162 frontier (private retention + build gate), stark #156 pinned drop, lawine #161 both-bugs step, ubel #154 realistic bar — into ONE instantiated launch decision at the realized operating point, then assembles the pre-filled `Approval request: HF job` projection+validity body parameterized on land #71's PENDING tuple. **It does NOT authorize a launch** — the packet is a draft awaiting land's measured (λ,μ) and a human approval on the eventual filed issue.

## Public evidence used (imported VERBATIM — one source of truth per constant, not re-derived)
- **#155 consolidator** (`approval_projection_consolidator.py`): `consolidate()` / `SamplingModel` / `validity_gate` imported verbatim. P(clear-500)=min(Φ((proj−500)/σ), Φ((geom_tps−500)/σ)) (conservative union; geometry can only suppress), `combined_rel=√(samp²+calib²+step²)`, K_cal=125.268. Banked anchors: oracle 269.5/NO-GO, both-bugs 535.4/GO, boundary 500/HOLD.
- **#162 frontier** (`tightened_private_500_frontier.py`): `r_tree(d, topo)` private retention (piecewise-linear through stark's banked breakevens) + build gate **(λ_min,μ_min)=(0.8809, 0.7353)** @ P≥0.5. Re-exports the #155 consolidator; surface machinery untouched.
- **stark #156** (`tree_private_drop_reconcile/results.json`): pinned private drop **1.8015% descent-only / 1.8626% both-bugs**, anchored to flagship GT-4.3%. Banked projections 510.58 (descent) / 525.46 (both) at the measured-step bar.
- **lawine #161** (`both_bugs_step_cost.json`): both-bugs accept-prep is **step-NEUTRAL** (Δ=0.0%) → both-bugs official **537.8** locks in; the served accept-prep kernel is byte-identical (denken #133 plumbing fix, 0 added kernel ops).
- **ubel #154 realistic bar**: E[T]=4.809, step 1.2047 (scatter + LP-avoidance, greedy-token-identical).
- **denken #150 / #158**: validity contract (PPL≤2.42 & boots & 128/128) and greedy-exactness (`--audit-kernel-symbol`) — BANKED gates.
- **land #71 / denken #166 / kanna #159 / lawine step-reconcile**: PENDING legs (measured tuple, PPL-margin bound, σ_hw, final depth-9 step reconcile).

## Map
```
proj_private = K_cal · E[T]_land · r_tree(d_pinned, topo) / step          (K_cal=125.268)
P(clear 500) = min( Φ((proj−500)/σ), Φ((geom_tps(λ,μ)−500)/σ) )
σ            = proj · √(samp² + calib² + step²)   [+ σ_hw PENDING kanna #159]
launch GO  iff  P(clear 500) ≥ 0.9  AND  (λ,μ) ≥ (0.8809, 0.7353)  AND  validity READY
```

## Step 1 — PINNED instantiation (full-recovery corner λ=μ=1, realistic bar)
| topology | E[T] | r_tree(d_pinned) | proj_private | P(clear 500) | LCB(P≥0.9) | CI99 | conf-99 | launch (P≥0.9) | binding leg |
|---|---|---|---|---|---|---|---|---|---|
| **descent-only (BUG-1 deferred)** | 5.0564 | 0.9883 | **519.6** | **96.3%** | 505.6 | [491.3, 547.9] | INDETERMINATE | **GO** | sampling |
| both-bugs (BUG-1 fixed) | 5.2070 | 0.9877 | 534.8 | 99.9% | 520.6 | [506.3, 563.3] | robust-GREEN | GO | sampling |

The private haircut at the pinned drop is tiny (r_tree≈0.988, ~topology-independent — consistent with #162's finding that the haircut acts on the level, not the shape). The binding leg is **sampling** (E[T] variance), not geometry or validity: at full recovery the geometry membership is satisfied (geom_tps 525.8) and the projection clears with margin. **Cross-check vs stark #156:** stark's banked 510.58/525.46 were at the *measured-step* bar (1.2182); the realistic bar (1.2047) buys ~9 TPS of headroom → 519.6/534.8, exactly the step-reduction effect #162 Step-1 documents.

## Step 2 — BUG-1 resolution: pinned drop 1.80% ≪ 6% → DEFERRED
- The pinned drop (1.80%) sits far below #162 Step-5's ~6% BUG-1 binding threshold. BUG-1 (wirbel #160 depth-1 spine 0.679→0.7287) is therefore **not mandatory** for the first shot.
- Descent-only at the pinned point projects **519.6 > #162's 511.1** — and #162's 511.1 was computed at the *harsher* GT-4.3% drop. The lower pinned drop (1.80%) plus the realistic bar together lift the margin by **+8.5 TPS** over #162's already-clearing figure. `exceeds_162_gt_511 = True`.
- The both-bugs spine upgrades the shot to conf-99 robust-GREEN (534.8, CI99 floor 506.3 ≥ 500) and covers the 9% band-ceiling, but is **deferrable**: the spine is insurance for a drop that did not materialize.
- `descent_only_p_clear500_at_pinned_drop = 0.9630` **[TEST]**.

## Step 3 — Readiness packet (pre-filled issue body; 6 BANKED / 4 PENDING)
The script renders the verbatim projection+validity block of the eventual `Approval request: HF job` issue, parameterized on land #71's PENDING tuple `(E[T], ρ₂, λ, μ, step, ppl, boots, completed)`. Full markdown in `pinned_launch_decision_packet_results.json → step3_readiness_packet.packet_markdown`. Leg ledger:

| status | leg | role |
|---|---|---|
| BANKED | fern #155 consolidator | GO/NO-GO + CI + binding-leg union |
| BANKED | fern #162 build gate | (λ_min,μ_min)=(0.8809, 0.7353) @ P≥0.5 |
| BANKED | stark #156 pinned drop | private drop 1.80% desc / 1.86% both |
| BANKED | lawine #161 both-bugs step | step-NEUTRAL → both-bugs official 537.8 |
| BANKED | denken #150 validity | PPL≤2.42 & boots & 128/128 contract |
| BANKED | denken #158 greedy-exactness | per-token committed==argmax (BUG-2 catcher) |
| **PENDING** | lawine step-reconciliation | final depth-9 step reconcile (overlap vs roofline) |
| **PENDING** | denken #166 PPL-margin bound | M=32 batched-verify aggregate-PPL worst-case vs 2.42 |
| **PENDING** | kanna #159 σ_hw | 4th quadrature term (A10G clock/thermal/cold-start) |
| **PENDING** | land #71 measured tuple | (E[T], ρ₂, λ, μ, step, ppl, boots, completed) |

The packet is **complete except for the 4 in-flight legs** — three widen/validate the projection (step-reconcile, PPL-margin, σ_hw) and one fills the operating point itself (land's tuple). Once land #71 lands a tuple reaching (λ,μ)≥(0.881, 0.735), the packet is filed-ready.

## Step 4 — self-test (PRIMARY = all four pass)
- **(a) oracle E[T]=2.621 → NO-GO** — projects 269.2, far under 500. The decision correctly rejects the no-speculation floor. ✓
- **(b) both-bugs E[T]=5.207 → GO** — projects 534.8 @ P=99.9%, conf-99 robust-GREEN. ✓
- **(c) descent-only is operating-point-specific** — at the pinned 1.80% drop it returns **GO** (proj 519.6, p=96.3%); at the 9% band-ceiling the SAME topology returns **NO-GO** (proj 494.8, p=31.7%). The gate is a function of the realized drop, not a static topology verdict — this reproduces #162 Step-5's 494.8 ceiling exactly. ✓
- **(d) packet PENDING-list correct** — exactly the 4 in-flight legs PENDING, the 6 evidence legs BANKED (sets match). ✓

`launch_packet_self_test_passes = 1`. Surface-corner consistency check: `joint_et(1,1,b_both)=5.2070` and `joint_et(1,1,b_desc)=5.0564` reproduce lawine #161's E[T] anchors to <0.5%.

## Step 5 — hand-off
At the pinned operating point the recommended first shot is **descent-only, P(clear-500)=96.3%** (proj 519.6, LCB(P≥0.9) 505.6 ≥ 500), pending land #71's measured (λ,μ) reaching the (0.881, 0.735) build gate and the four PENDING legs (lawine step-reconcile, denken #166 PPL-margin, kanna #159 σ_hw) — **the packet is the pre-filled issue body awaiting only land's tuple; it does NOT authorize a launch.**

## Reproduce
```
cd target && python scripts/profiler/pinned_launch_decision_packet.py \
  --wandb --wandb_group pinned-launch-decision-packet \
  --wandb_name fern/pinned-launch-decision-packet
# CPU-only, ~5 s. Imports committed leg outputs; re-derives nothing.
```
Output: `research/spec_cost_model/pinned_launch_decision_packet_results.json`. W&B run `l3pdlh22`.

**Primary:** launch_packet_self_test_passes = 1. **Test:** descent_only_p_clear500_at_pinned_drop = 0.9630.
