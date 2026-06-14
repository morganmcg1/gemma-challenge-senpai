# Launch verdict at the conservative launch-realized step (PR #174)

**Lane:** decision-geometry — refresh my own #167 launch-decision packet at the **corrected
launch-realized decode step**, holding stark #156's pinned private drop fixed, and **settle the
first-shot recommendation**.
**Scope:** LOCAL CPU-only analytic re-instantiation. NO GPU / vLLM / HF Job / submission / served-file
change. BASELINE stays **481.53 TPS** (PPL 2.3777), greedy/PPL untouched. Adds **0 TPS**. **Corrects the
#167 headline step + settles the first shot; does NOT authorize a launch. NOT open2.**

Tool: `scripts/profiler/conservative_step_launch_verdict.py` · JSON:
`research/spec_cost_model/conservative_step_launch_verdict_results.json` · W&B run `s2vihqh1`
(group `conservative-step-launch-verdict`).

---

## The decision this answers

My #167 packet (MERGED) closed the decision-geometry arc, but its headline — descent-only **519.6 TPS,
P=96.3%** — was instantiated at the **optimistic conditional step 1.2047** (ubel #154's scatter+LP-reduced
decode). lawine #168 (MERGED this cycle) then ruled 1.2047 **CONDITIONAL**: it requires the argmax-only
decode build, which **has not shipped**, so the **launch-realized step is 1.2182**. denken #166 (MERGED)
banked the PPL stamp. Both of #167's refresh items have landed; the headline step needs correcting.

The decision-critical question: **at the shipped 1.2182, does descent-only-first still clear the
conservative P≥0.9 / LCB≥500 one-shot bar, or does the recommended first shot flip to both-bugs?**

**Why this is a clean re-instantiation, not a re-derivation.** In the #155 consolidator,
`proj = K_cal·E[T]·r_tree(drop)/step` scales **exactly** as 1/step, while
`combined_rel = sqrt(samp² + calib² + step_anchor²)` is **step-invariant** (sampling depends on E[T]
only; calibration and step-anchor are fixed). So building the sampling-CI model **once** at
base_step=1.2182 (exactly as #167 did) and varying only the `step` argument reproduces #167 at 1.2047 to
machine precision and cleanly recomputes proj / LCB / P(clear 500) at 1.2086 / 1.2182. r_tree at the
pinned drop is also step-invariant. **ONE physical operating point, three decode-step framings.**

---

## Result (headline)

| metric | value |
|---|---|
| **PRIMARY** `conservative_step_verdict_self_test_passes` | **1** (5/5 self-tests PASS) |
| **TEST** `descent_only_p_clear500_at_conservative_step` | **0.8994** (< 0.90 → descent-only misses the bar) |
| First-shot verdict at the shipped 1.2182 | **flips-to-both-bugs** (descent-only-first does NOT survive) |
| Recommended first shot | **both-bugs** |
| Cost of NOT shipping #154 (argmax-only decode) | **+3.96 TPS** descent LCB headroom (+4.07 proj) |
| Leg ledger | **8 BANKED / 4 PENDING** (#168 + #166 newly banked) |
| metrics NaN-clean | **1** |

**Verdict: at the conservative launch-realized step 1.2182, descent-only-first FLIPS to both-bugs.** It is
a **knife-edge** miss — descent-only LCB(P≥0.9) = **499.97** (margin **−0.035 TPS**), P(clear-500) =
**89.94%** vs the 90% bar — but it is on the wrong side of the conservative one-shot gate. both-bugs is
**robustly GO at all three steps** (LCB 514.9 → 520.6, conf-99 robust-GREEN). The descent-only miss is
fully recovered by **either** shipping #154's argmax-only decode (→ realizable 1.2086, LCB 503.9, GO)
**or** fixing BUG-1 (→ both-bugs, LCB 514.9, GO).

---

## (1) Three-step instantiation — same pinned drop, three decode-step framings

Full-recovery corner (λ=μ=1), pinned drop 1.80% (descent) / 1.86% (both). `official` = public projection
(no private haircut); `proj_priv` carries the r_tree(drop) haircut; LCB = LCB(P≥0.9); launch = conservative
P≥0.9 one-shot verdict.

| step framing | topology | official | proj_priv | P(clear 500) | LCB(P≥0.9) | conf-99 | launch (P≥0.9) |
|---|---|---|---|---|---|---|---|
| **conservative 1.2182 (SHIPPED)** | **descent-only** | 519.96 | **513.89** | **89.94%** | **499.97** | INDETERMINATE | **HOLD** |
| conservative 1.2182 (SHIPPED) | both-bugs | 535.44 | 528.89 | 99.59% | 514.88 | robust-GREEN | **GO** |
| realizable 1.2086 (if #154 ships) | descent-only | 524.08 | 517.96 | 94.95% | 503.92 | INDETERMINATE | GO |
| realizable 1.2086 (if #154 ships) | both-bugs | 539.68 | 533.07 | 99.87% | 518.95 | robust-GREEN | GO |
| optimistic 1.2047 (#167 original) | descent-only | 525.78 | 519.64 | 96.30% | 505.56 | INDETERMINATE | GO |
| optimistic 1.2047 (#167 original) | both-bugs | 541.43 | 534.80 | 99.92% | 520.63 | robust-GREEN | GO |

The optimistic row **reproduces #167 exactly** (519.64 / 96.30% / LCB 505.56 vs #167's
519.639 / 96.30% / 505.555). The step-framing column is the whole story: descent-only's LCB walks
505.6 → 503.9 → **499.97** as the step stiffens from optimistic to shipped, crossing the 500 bar between
realizable and conservative.

---

## (2) First-shot settlement — descent-only-first vs both-bugs

| framing | descent LCB − 500 | descent launch | both-bugs launch |
|---|---|---|---|
| conservative 1.2182 (SHIPPED) | **−0.035** | **HOLD (flips)** | GO |
| realizable 1.2086 | +3.92 | GO | GO |
| optimistic 1.2047 | +5.56 | GO | GO |

**descent-only-first is `flips-to-both-bugs` at the shipped step.** The recommended-first-shot calculus is
indeed step-framing-sensitive (as the PR anticipated): descent-only-first is robust at the realizable step
but **does not clear** the conservative one-shot bar at the shipped step. The **cost of NOT shipping #154's
argmax-only decode build** is **+3.96 TPS of descent-only LCB headroom** (the gap between the realizable
+3.92 margin and the conservative −0.035 margin) — exactly the headroom that would put descent-only
comfortably over 500.

**both-bugs is the safe first shot at the shipped step:** proj 528.9, P 99.6%, LCB 514.9, conf-99
robust-GREEN — comfortable margin at 1.2182 and every framing. Fixing BUG-1 (wirbel #160's depth-1 spine)
is the cleanest way to a robust descent launch under the conservative reality; it is no longer merely
9%-band-ceiling insurance.

---

## (3) Refreshed readiness packet — 8 BANKED / 4 PENDING

lawine #168 (step reconciliation) and denken #166 (PPL-margin bound, **M=32 worst-case PPL 2.4134 ≤ 2.42**,
margin 0.0066) are now **BANKED** — they were the two PENDING legs #167 named that have since merged. The
PENDING set is now the two long-poles (kanna #159 σ_hw, land #71 tuple) **plus the two NEW in-flight
descent-validity upgrades** the PR flagged: **denken #172** (replaces the point E[T]=5.0564 with a
conservative central±floor lower bound) and **lawine #173** (confirms the actual descent kernel realizes
the 1.2182 step). The full verbatim `Approval request: HF job` projection+validity block — re-emitted at
the corrected step with the refreshed ledger — is in the results JSON (`step3_readiness_packet.packet_markdown`).

**Why the two new PENDING legs matter to this packet specifically:** both attack the exact knife-edge.
denken #172 hardens the descent-only **numerator** (the point E[T]=5.0564 this packet consumes) into a
conservative lower bound; lawine #173 confirms the descent-walk **denominator** (that the shipped descent
kernel actually holds 1.2182, not worse). A −0.035 TPS miss is within the resolution of both — so they,
not σ_hw, are the legs that will decide whether a descent-only first shot is ever recommendable.

---

## (4) Self-validation (PRIMARY)

5/5 self-tests pass → `conservative_step_verdict_self_test_passes = 1`:

1. ✓ **(a)** the optimistic 1.2047 reproduces #167's 519.6 / 96.3% / LCB 505.6 within tolerance, AND the
   realizable 1.2086 lane stays GO (LCB 503.9 ≥ 500).
2. ✓ **(b)** at the conservative 1.2182 the descent-only proj_private (513.9) and LCB (499.97) are
   recomputed and the GO/marginal/flip verdict is **explicit** (`flips-to-both-bugs`) and **consistent**
   with the launch gate (flip ⟺ launch ≠ GO).
3. ✓ **(c)** both-bugs remains **GO at all three steps** (1.2182 / 1.2086 / 1.2047).
4. ✓ **(d)** the PENDING/BANKED ledger matches the current merged state (kanna #159 + land #71 + denken
   #172 + lawine #173 PENDING; #168 / #166 / #156 / #161 / #150 / #158 BANKED).
5. ✓ **(e)** NaN-clean (every headline numeric finite).

---

## (5) Hand-off

At the conservative launch-realized step **1.2182** and pinned **1.80%** drop, the recommended first shot
is **both-bugs**, P(clear-500) = **99.6%**, LCB(P≥0.9) = **514.9** — descent-only-first is
**flips-to-both-bugs** at the shipped step (LCB 499.97), **robust** at the realizable step (LCB 503.9);
pending **kanna #159, land #71, denken #172, lawine #173**. The refreshed packet remains a pre-filled
draft — it does **NOT** authorize a launch.

---

## What happened

The step-framing correction is decisive. #167's comfortable descent-only GO (P=96.3%) lived entirely at
the optimistic 1.2047. Once lawine #168 pins the launch-realized step at the shipped 1.2182, the
descent-only LCB falls to 499.97 — a **0.035-TPS / 0.06-pp** miss of the conservative one-shot bar — so the
recommended first shot **flips to both-bugs**. This is not a margin collapse; it is a knife-edge that two
cheap, already-in-flight legs (#154's argmax-only decode for +3.96 TPS, or the BUG-1 spine fix for the
both-bugs +14.9 TPS LCB cushion) each clear comfortably. The honest reading: under the **shipped** decode
reality, a **descent-only** first shot is not de-risked to the conservative bar, but a **both-bugs** first
shot is robustly GO at every framing.

This does **not** authorize a launch. It corrects the #167 packet headline to the shipped step and settles
the first-shot recommendation; the actual launch still requires a human-approved `Approval request: HF job`.

## Suggested follow-ups

- **denken #172 (descent E[T] conservative LB)** is now the load-bearing PENDING leg for a descent-only
  first shot: a central±floor lower bound on E[T]=5.0564 directly moves the −0.035 TPS margin. If #172's
  floor sits below the point estimate, descent-only-first stays flipped even at the realizable step.
- **lawine #173 (descent-walk step-neutrality)** is the denominator confirmation: if the shipped descent
  kernel realizes worse than 1.2182, the flip deepens; if it confirms 1.2182, the −0.035 margin is firm.
- **Fold kanna #159 σ_hw** when it lands — it widens the CI (adds a 4th quadrature term), which can only
  push the already-sub-500 descent LCB further down, reinforcing the both-bugs-first recommendation.
- **If #154's argmax-only decode ships**, re-run with the realizable step as the operating point: that
  alone restores descent-only-first to GO (LCB 503.9) without the BUG-1 spine.
