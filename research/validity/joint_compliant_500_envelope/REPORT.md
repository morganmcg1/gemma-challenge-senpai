# PR #325 — Joint compliant-500 envelope: identity-kernel ceiling × E[T]-lever under the #192 gate

**Verdict: 🟡 YELLOW (governed by the private tax, capped by the kernel).** Composed under ONE
compliant ledger, the two priced halves do **not** close to GREEN. At central private-tax the joint
compliant TPS is **520.95 — CAPPED by the identity kernel** (the E[T]-lever's 586 is unreachable;
the cap eats 65.1 TPS), clearing 500 by **+4.19%**. At the worst credible private tax it is
**492.87 — below the cap, floored by ρ_worst**, missing 500 by **−1.43%**. The 500 bar sits strictly
**between** the worst-case floor (492.87) and the kernel cap (520.95). **Bank-the-analysis, 0 TPS,
BASELINE 481.53 unchanged.**

W&B `xk1pghy4` · primary `joint_compliant_tps_worstcase=492.87` · test `joint_compliant_tps_central=520.95`.

## The integrator question

We had priced each half of the compliant >500 path in isolation but never **composed** them under
one ledger:
- **IDENTITY half** — a batch-invariant int4 verify kernel, ceiling λ=**520.95** (wirbel #216
  `pc8g6s04` / #227 `o674wmna`, UNBUILT). The deployed 481.53 split-K verify diverges 0.73% at M=8
  (denken #232) → FAILS the #192 gate, so 520.95 is the relevant identity ceiling, not 481.53.
- **SPEED half** — an EAGLE-3 E[T]-lever, honest public 622.08 × private-tax ρ (fern #318 `xe8ff7hq`:
  central 0.9421→586.1, worst 0.7923→492.9).

The governing question: **does any point in (identity-kernel ceiling × E[T]-lever realization) clear
500 while staying greedy-identical, and what binds — the 520.95 cap, the E[T]-realization, or the
private worst-case?**

## The ledger (one composition law, nothing re-measured)

    official_compliant(E[T], kernel) = min( K_cal · (E[T] · τ / step_kernel),  λ_ceiling )

- `K_cal=125.268, τ=1.218, step_deployed=1.2182` (denken #278); τ/step=0.99984≈1, so the law
  collapses to ≈ K_cal·E[T]; deployed E[T]=3.844 → 481.53.
- **step_kernel = step_deployed.** wirbel #216's kernel model (reconciled by #235 twoceiling): the
  batch-invariant split-K fix changes argmax reduction-**ORDER** (determinism), **not** the GEMM
  step-cost or topology. The int4-spec ceiling 520.95 is banked **at the deployed step**. So the
  identity kernel enters the ledger **only as the 520.95 throughput CAP**, not as a heavier
  denominator. *(Sensitivity: a slower batch-invariant kernel would both raise the denominator AND
  lower the realized ceiling below 520.95 → the cap binds EARLIER. So 520.95 / deployed-step is the
  **optimistic** kernel corner; this YELLOW is an **upper bound** on compliance.)*
- **E[T]-lever realization (the SPEED input):** fern #318 honest_public_611 = 622.08 = K_cal ·
  realized_public_E[T] (4.966 = EAGLE-3 paper public acceptance length, the wall-/rewrite-honest
  realization of the free 6.11 build target); private TPS = 622.08·ρ.

## (1) Joint compliant TPS — the cap binds at central, the private tax binds at worst

| ρ scenario | ρ | E[T]-lever (uncapped) | **joint compliant** | binds | clears 500? |
|---|---|---|---|---|---|
| central | 0.9421 | 586.08 | **520.95** | **kernel cap** | ✅ +4.19% |
| break-even | 0.8038 | 500.00 | **500.00** | E[T]-realization | ✅ +0.00% |
| **worst-case** | **0.7923** | **492.87** | **492.87** | **private tax (d)** | ❌ **−1.43%** |

**The headline integrator finding:** the joint compliant central is **NOT 586** — the identity
kernel caps it at **520.95**, eating **65.1 TPS** of the E[T]-lever's headroom. The cap clears 500
(+20.95 TPS, +4.19%), but at the worst credible private tax the result drops **below** the cap to
492.87, so the kernel ceiling is **not** the binding constraint there — the **#318 private worst-case
is**.

## (2) Binding-constraint map (free E[T] ∈ [3.844, 6.11])

- **Central ρ:** the binding constraint switches from **E[T]-realization (b, #298-throttled)** to the
  **520.95 kernel ceiling (a)** at **free E[T]\* = 5.041** (realized 4.415). Below 5.041, raising E[T]
  raises TPS; above it, the cap pins TPS at 520.95 — more E[T] buys nothing.
- **Worst ρ:** the cap **never** binds in range (free E[T]\*_worst = 6.79 > 6.11). The **#318 private
  worst-case (d)** is the limiter across the whole sweep; the curve maxes at 492.87 < 520.95.
- **#316 coverage clip (c):** the deployed deep spine caps realized E[T] at 4.9097. At central ρ that
  realized cap already implies TPS > 520.95, so the **kernel ceiling (a) pre-empts the coverage clip
  (c)**; (c) survives only as a build-feasibility gate (the build must clear max_frac_beyond_top4 ≤
  0.2907 vs linear demand 0.3468 to reach realized 4.966 at all).
- **#314 rewrite gate:** the whole `K_cal·E[T]` composition assumes the #312 loopgraph rewrite. The
  **eager path floors TPS at 360–481 (<500) at every E[T]** (rewrite worth +57…+140), so it cannot
  clear 500 regardless of the lever — a prerequisite, not a tunable.

So at the build target the limiter is **(a) the 520.95 kernel ceiling at central ρ** and **(d) the
private worst-case at worst ρ**. The bar 500 lives in the gap between them.

## (3) Feasibility verdict

**🟡 YELLOW.** Clears 500 at central ρ (capped at 520.95, **+4.19%**) and at break-even (500.0), but
the worst-case private tax lands at **492.87, −7.13 TPS (−1.43%)** below 500. The verdict flips to
GREEN only if the held-out private set is **not** as out-of-distribution to the {2,21,39} fusion head
as the worst cross-DOMAIN shift in the EAGLE-3 paper (CNN/DM ÷ HumanEval). This inherits fern #318's
YELLOW and **adds** the binding-constraint structure: even in the best case the compliant headroom is
only **+4.19% (capped)**, and the worst case is **private-tax-bound, not kernel-bound**.

## (4) The single cheapest measurement that flips YELLOW → GREEN

The YELLOW driver is **ρ_worst = 0.7923 < break-even 0.8038** (a gap of 0.0115 in ρ, ≈ −7 TPS).

- **(A) 0-GPU — credit the organizer-verified M=8 tree a₁-recovery (c₁=1.0, #316/#323) into the 0.792
  bound. CANNOT flip alone.** 0.792 is a *raw aggregate* τ-ratio (5.34/6.74) and EAGLE-3 has **no
  per-depth α table** to isolate a₁ vs deep. The most-generous analytic a₁-credit (attribute 100% of
  the cross-domain degradation to deep positions, a₁ fully held) is fern #318's
  `implied_f_deep_a1_held` = **0.9083** for the 0.792 case — **still below the break-even f_deep
  0.9163**, missing by ≈7 TPS. The clean fold is **blocked by the missing per-depth table**.
- **(B) checkpoint-gated — the #319 staged per-depth private-α read on a trained {2,21,39} fusion
  head. FLIPS GREEN.** It directly measures ρ_priv_e3 (and the a₁/deep split that unblocks (A)),
  replacing the 0.792 cross-DOMAIN literature bound. The **within-task measured analogue is 0.957**
  (Δ4.3%, organizer-verified on the linear stack) → 595.4 TPS, ~5× inside break-even. So the measured
  fusion ρ would very likely clear 500 at worst-case.

**Recommendation:** the **#319 per-depth read** is the single cheapest measurement that *actually*
flips YELLOW→GREEN. The 0-GPU credit is cheaper but cannot flip alone (the missing per-depth table
leaves f_deep 0.9083 < break-even 0.9163). #319 supplies that very table (unblocking the credit) AND
directly measures ρ — it is both the cheapest unblocking move and the decisive one. **Reinforces
issue #319.**

## Scope / caveats

- LOCAL CPU-only analytic over banked constants. **0 TPS; BASELINE 481.53 untouched; greedy/PPL
  untouched. NO GPU / vLLM / HF Job / submission / served-file change. Authorizes NOTHING. NOT a
  launch. NOT a build.** Peak RSS 13.66 MiB.
- This composes priced **bounds**, not measurements. The 520.95 ceiling is UNBUILT; the ρ envelope is
  a worst-case literature bound; the 6.11 E[T] and the loopgraph rewrite are unbuilt. The card sizes
  the joint feasibility envelope and the binding constraint — it does not change the baseline.
- The kernel-step = deployed-step reading is the **optimistic** corner (a slower batch-invariant
  kernel only tightens the verdict). The #298 wall-realization (0.477) used in the sweep lands
  realized E[T] at 4.925 (free 6.11), ~0.8% below the #318 anchor 4.966 — i.e. if the wall-realization
  governs over the EAGLE-3-paper anchor it shaves another ~5 TPS, deepening YELLOW, not changing it.
- Out of scope (do not re-derive here): the ×0.804 reconciliation (settled #310), the a₁-cliff
  trainability (denken #308), E[T]=6.11 reachability, greedy/PPL identity (Issue #192), the
  rank-coverage mass axis (lawine #316).

## Reproduce

```
cd target/ && .venv/bin/python \
  research/validity/joint_compliant_500_envelope/joint_compliant_500_envelope.py --self-test \
  --wandb_group eagle3-joint-compliant-envelope --wandb_name fern/joint-compliant-500-envelope
```

Imports (exact): fern #318 `xe8ff7hq` (honest_public_611 622.08, ρ central/breakeven/worst
0.9421/0.8038/0.7923, private TPS 586.08/492.87, implied_f_deep 0.9083, breakeven f_deep 0.9163,
within-task ρ 0.957) · wirbel #216 `pc8g6s04` / #227 `o674wmna` / #235 twoceiling (int4-spec ceiling
520.95, E[T]_int4 5.0661, kernel = argmax-order-not-step) · stark #298 `xp974x58` (wall-realization
0.4769) · wirbel #314 `fwqbz7zf` (eager 360–481, rewrite +57…+140, loopgraph@611=500) · lawine #316
`5lnz5jgb` (cov bar 0.2907 vs linear 0.3468, deployed-spine E[T] cap 4.9097) · denken #278 / kanna
#269 (K_cal 125.268, step 1.2182, τ 1.218).
