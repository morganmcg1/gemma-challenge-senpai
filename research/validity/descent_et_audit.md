# Descent-E[T] DP audit: is the 5.0564 first-shot numerator independently re-derivable and floored? (PR #172)

**Verdict: BOUNDED — NOT ROBUST.** The descent-only first-shot numerator
**E[T] = 5.0564** re-derives exactly by a method *distinct* from #135's forward DP,
but its conservative lower bound **3.5346 → 363 official TPS FAILS 500.** The 522
projection is real only if openevolve **cause #2** (depth>0 self-KV starvation) is a
*fixable build defect*: clearing 500 requires **≥ 91% deep-spine spread recovery**
(λ\* = 0.908 overlap / 0.890 realizable). This is the opposite robustness posture to
#166's PPL stamp — and the honest finding the launch packet needs.

- **PRIMARY** `descent_et_audit_self_test_passes = True`
- **TEST** `descent_only_E_T_lower_bound = 3.5346` (binding adversarial floor) → 363 TPS, FAILS 500
- `descent_only_E_T_recomputed = 5.0564` (M1 ≡ M2 ≡ imported #135, resid 0.0)
- W&B run `gh8pa4f3` (group `descent-et-dp-audit`). Evidence:
  `research/validity/descent_et_audit/runs/20260614T150434Z/descent_et_dp_audit_result.json`.
  Pure-analytic, CPU-only (peak 26 MiB, no GPU / vLLM / model load). BASELINE
  untouched: 481.53 TPS. Adds 0 TPS — it hardens the **numerator**, the E[T] twin of
  #166's PPL-denominator stamp.

---

## Why this bound

Every launch leg now consumes **descent-only E[T] = 5.0564 (→ ~522 official TPS)** as
the first-shot numerator: fern #167's packet, ubel #163's bars, wirbel #165's composed
5.2070 supply ceiling. That 5.0564 traces to a **single source** — wirbel #135's
descent DP (`score_tree_depthrank`, a forward path-product over the oracle's measured
tree). It has **never been independently re-derived or bounded.** A single-method
number driving the whole packet's headline is exactly the validity gap #166 closed for
the PPL denominator; this is its numerator twin.

Two questions, both unanswered before this PR:
1. **Is 5.0564 a DP artefact?** Re-derive it by a method that never touches #135's
   forward accumulation. If an independent recursion lands on a *different* number, the
   packet's headline is wrong.
2. **How low can descent-only E[T] go** if every favourable modelling assumption is
   taken adversarially — specifically if openevolve cause #2 (self-KV starvation at
   depth>0) is *intrinsic*, not a build defect? Quote E[T] as *central ± a defensible
   floor*, not a single point.

This is a **synthesis**, not a new measurement: it imports committed advisor-branch
outputs (#135 DP, the oracle primitives, #76 spine, #79 rank-rescue ladder, the
deep-spine decomp lattice, openevolve's localizer) and propagates them analytically.

---

## The anchors (committed advisor-branch artifacts)

| Role | Anchor (committed) | Value |
|---|---|---|
| **The number under audit** | `spec_cost_model/bug2_salvage_descent_results.json` (#135) | `bug2_et_full_alt_d1_0679` = **5.056405** (the 5.0564); `bug2_et_full` = 5.041271 (d1=0.674); `combined_et_both_fixed` = 5.206954 |
| **The committed floor** | same file, `step1.mb3_descending_same_ladder` | **3.534581** (mb3 topology, declining ladder = config C) |
| **Tree topology** | `spec_cost_model/rho_optimal_topology_results.json` | M=32 / depth-9 / max-branch-3 parent array (32 nodes, 7 leaves, irregular) |
| **Rising linear spine** | `accept_calibration/accept_calibration_results.json` (#76) | conditional [0.7287, 0.7590, 0.7925, 0.8217, 0.8343, 0.8353, 0.8473] |
| **Rank-rescue ladder** | `rank_coverage/rank_coverage_results.json` (#79) | ρ_cond = [0.4165, 0.2655, 0.1908] |
| **Measured (defective) ladder** | `oracle_readout` block (tree-488-pw-fp32-v0) | cumulative [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037]; realized E[T] = 2.621; salvages 391 / full 37 / 1024 steps |
| **Cause #2** | openevolve oracle localizer (board 20260614-140843, 14:08 UTC) | depth>0 self-KV starvation; **not retracted** by the 14:56 correction (which only retracts cause #1, depth-1) |

---

## 1. Independent recompute — the headline (distinct from #135's forward DP)

#135 scores E[T] with `score_tree_depthrank`: a **forward** pass over nodes in id-order
accumulating reach-probabilities `pp[c] = pp[parent]·pv[depth][rank]` and summing
`F = 1 + Σpp`. The topology is **irregular** (branch-counts differ per depth), so there
is no pure depth-only closed form — the distinct method must respect the tree shape.
Two genuinely independent routes, neither touching that forward accumulation:

- **M1 — backward renewal-reward DP** (post-order). The greedy walk's sibling edges are
  mutually exclusive (at most one child token can equal the single greedy-target
  argmax), so the accepted set is always a **chain** and E[T] = 1 + E[walk length].
  Define `D(u)` = expected accepted strict-descendants of `u` *given* `u` accepted:
  `D(u) = Σ_r pv[d+1][r]·(1 + D(child_r))`, `D(leaf) = 0`, `E[T] = 1 + D(root)`. This is
  the renewal-reward dual of #135's forward flow — a *different recursion direction*
  propagating a *different quantity* (conditional remaining length, not forward reach).
- **M2 — brute-force explicit path enumeration.** Enumerate every root→node path and sum
  its path-product `∏_k pv[k][r_k]` independently (no DP, no memoisation) — the direct
  combinatorial expectation.

Both reconstruct the per-rank marginals from **first principles** (chain rule
`pv[d][1] = q[d]`; `pv[d][r] = (∏_{j<r}(1−ρ_j))·ρ_r·(1−q[d])` for r≥2) — they never
import #135's `build_depth_pvecs_measured`. #135's literal DP is run *only* as an
imported reference cross-check.

### Result

| quantity | value | vs imported #135 |
|---|---|---|
| **M1 backward renewal-reward DP** | **5.056404568844709** | resid **0.0** |
| **M2 path enumeration** | 5.056404568844712 | M1≡M2 to **2.7e-15** |
| imported `score_tree_depthrank` (ref) | 5.056404568844709 | resid 0.0 |
| d1=0.674 variant (M1) | 5.041270826829536 | == `bug2_et_full` (resid ~1e-15) |
| both-bugs ceiling (M1) | 5.206954309441966 | == `combined_et_both_fixed` |

**The 5.0564 is not a DP artefact.** Three independent routes (backward DP, path
enumeration, #135's forward DP) agree to machine precision. The packet's headline
numerator is real.

---

## 2. Conservative lower bound — adversarial deep-node self-KV starvation (cause #2)

The 5.0564 model's single most optimistic input is the **deep-spine spread**: it assumes
the depth≥2 rank-1 conditional **rises** (0.76→0.85), the rate the *same* drafter hits
in the LINEAR chain (which has self-KV). openevolve's localizer names this as **cause
#2**: *"self-context at depth>0 … if the tree emit re-enters the [MTP] head per-node
without the chain's self-KV, depth>0 collapses."* The 14:56 correction retracts cause
#1 (depth-1) into the BUG-1 margin but **does not** retract cause #2 — it is the live
risk to the descent numerator.

Model cause #2 as the **floor of the deep-node acceptance term** (the PR's binding
extreme — "the single most pessimistic input"): if self-KV starvation persists, the
descent fix re-seeds branches (the buildable topology change) but the deep-spine
conditional does **not** recover — it stays at the **measured declining oracle ladder**
(conditional `[0.674, 0.519, 0.580, 0.645, 0.679, 0.674, 0.617]`, the self-KV-starved
rates the oracle actually measured). Scoring the descent-fixed re-seeding topology with
that declining ladder gives:

> **`descent_only_E_T_lower_bound` = 3.534581** == #135's committed
> `mb3_descending_same_ladder` (resid 8.9e-16). Every favourable assumption (rising
> deep spine) replaced by its measured-adversarial counterpart; the branch re-seeding
> (sibling-salvage credit) is *granted*, isolating deep-node acceptance as the binding
> input.

### Graded spread-recovery ladder

λ interpolates q[d≥2] from declining (λ=0) to rho-optimal rising (λ=1), depth-1 held at
the central 0.679, branch width restored:

| λ (deep-spine spread recovery) | E[T] | overlap TPS | clears 500? |
|---|---|---|---|
| 0.00 | 3.5445 | 364.5 | ✗ |
| 0.25 | 3.8217 | 393.0 | ✗ |
| 0.50 | 4.1548 | 427.3 | ✗ |
| 0.75 | 4.5596 | 469.0 | ✗ |
| 0.90 | 4.8451 | 498.3 | ✗ (just under) |
| 1.00 | 5.0564 | 520.0 | ✓ |

**clear-500 spread-recovery threshold λ\* = 0.908 (overlap 1.2182) / 0.890 (realizable
1.2086)** by bisection. Below ~91% deep-spine recovery, descent-only is sub-500 even
with branch width fully restored — a **spread** failure, exactly cause #2's signature.

Even-lower context floors if the topology re-seeding *also* fails (reported, not the
bound): measured-realized **2.621**, spine-only-no-branches **2.544**.

---

## 3. Propagate — official = K_cal·(E[T]/step)·τ ; clear-500 verdict at the floor

K_cal = 125.268, τ = 1, step ∈ {1.2182 overlap, 1.2086 ubel#163 realizable}; clear-500
bar E[T] = 500·step/K_cal = **4.862 / 4.824**.

| config | E[T] | overlap TPS | realizable TPS | clears 500? |
|---|---|---|---|---|
| **central 5.0564** | 5.0564 | **519.95** | **524.08** | ✓ |
| **lower bound 3.5346** | 3.5346 | **363.46** | **366.35** | ✗ |
| both-bugs 5.2070 | 5.2070 | 535.43 | 539.69 | ✓ |

The central clears comfortably; the **lower bound fails by ~137 TPS.** The 522
projection **requires** ≥λ\* deep-spine spread recovery — i.e. cause #2 must be a
*fixable build defect*, not intrinsic starvation.

---

## 4. Self-test (PRIMARY)

`descent_et_audit_self_test_passes = True` requires all four:

| condition | check | result |
|---|---|---|
| (a) central reproduces 5.0564 | `\|recomputed − 5.056405\|` ≤ 1e-6 (actual 0.0) | ✅ |
| (b) conservative ordering | 3.5346 ≤ 5.0564 ≤ 5.2070 (both-bugs) | ✅ |
| (c) clear-500 verdict explicit | lower bound pass/fail stated at **both** 4.862 AND 4.824 | ✅ (FAIL at both) |
| (d) cross-method M1 ≡ M2 | resid ≤ 1e-9 at central, floor, both-bugs (actual ≤ 2.7e-15) | ✅ |

NaN-clean across the full payload. The pass is gated on the **honest** ordering and the
explicit *failing* verdict at the floor — it does not hide that the bound is not robust.

---

## 5. Scope and limitations (honest)

- **Bounds the descent-only E[T] numerator only.** It is the E[T] twin of #166's PPL
  denominator stamp; together they harden both ends of `official = K_cal·E[T]/step`. It
  does not re-measure step time (ubel #163) or K_cal (#100).
- **The floor is a *modelled* adversarial, not a measured build.** It transplants the
  oracle's *measured* declining ladder onto the descent-fixed topology — the strongest
  defensible pessimism given committed data, but the *real* built descent ladder is
  unmeasured. That is precisely what the hand-off asks openevolve to close.
- **Branch re-seeding is granted** (sibling-salvage independence + branch-hit credit
  taken favourably), isolating deep-node acceptance as the single binding knob per the
  PR. If re-seeding *also* fails, the deeper context floors (2.621 / 2.544) apply.
- **Greedy chain invariant assumed** (mutual-exclusivity of siblings under single-argmax
  target) — the same structural fact #135/#158 rely on; it makes E[T] = 1 + E[chain
  length] exact, so M1/M2 are exact, not approximations.
- Anchors are committed advisor-branch artifacts; no external-PR borrow, no official
  draws, no served-file change. NOT a launch; NOT open2 (this is tree economics, not
  drafter architecture).

---

## 6. Hand-off (launch evidence-line)

**BOUNDED — NOT ROBUST: descent-only first-shot E[T] = central 5.0564 (→ 520 official,
clears 500) ± conservative lower bound 3.5346 (→ 363, FAILS 500).** Re-derived 5.0564 by
a method *distinct* from #135's forward DP (backward renewal-reward DP + path
enumeration, M1≡M2 to 1e-15; reproduces imported #135 to resid 0.0); ordering 3.5346 ≤
5.0564 ≤ 5.2070 holds. **clear-500 does NOT survive the worst case:** the 522 projection
requires ≥91% deep-spine spread recovery (λ\* = 0.908 / 0.890) — i.e. openevolve cause
#2 (depth>0 self-KV starvation) must be a *fixable build defect*, not intrinsic.

**Single-knob hand-off to openevolve's oracle:** the bound is tight, so MEASURE land
#71's *built* descent ladder q[2..9] (+ branch-hit ρ2) on the re-seeding topology. That
one measurement converts this modelled floor to a measured E[T] and resolves the
λ-versus-λ\* verdict. Pairs with wirbel's depth>0 self-KV leg and the deep-spine decomp
lattice's `spread_recovery_map` (which independently puts the both-bugs clear-500
threshold at λ≥0.9). Feeds fern's consolidator packet as the numerator-side validity
stamp.

**Non-collision:** complements #166 (PPL denominator bound) with the E[T] numerator
bound. Distinct from #135 (the DP it audits), #158 (per-token argmax), ubel #163 (step
time / realizable bars), the deep-spine decomp (which models the *both-bugs* spread
recovery; this isolates *descent-only*).

---

## 7. Reproduce

```bash
cd target/
# pure-analytic, CPU-only (no GPU / vLLM / model load); ~26 MiB peak
python -m research.validity.descent_et_audit.descent_et_dp_audit --self-test \
    --wandb-name "denken/descent-et-dp-audit" \
    --wandb-group "descent-et-dp-audit"
```

Exit 0 ⇔ `descent_et_audit_self_test_passes` and NaN-clean. All anchors are read from
committed paths (`--bug2-anchor`, `--topo-json`, `--accept-json`, `--rankcov-json`,
`--decomp-json` overridable).
