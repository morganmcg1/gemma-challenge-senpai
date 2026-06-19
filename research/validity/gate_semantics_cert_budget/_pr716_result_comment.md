STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["kyp59hd6"],"primary_metric":{"name":"int8_certify_n_pointclears","value":1040},"test_metric":{"name":"asapplied_gate_is_point","value":1}}

## Results — `POINT_GATE_LANE1_ALIVE` (both worlds priced)

**Headline:** The as-applied #481/#515 fire-gate is **POINT-based** (`asapplied_gate_is_point=1`) — lawine #703 gated PASS on the point and used the CI as an honesty caveat. Under that standard **Lane 1 is alive on its point** (full-g32 0.438 ≥ 0.420). #710/#714 `DEAD` bounds **only** the stricter CI-certification the fire-gate does *not* require. The CI world is fully populated as the alternative the human may elect; under it **only the int8-locus is even edge-feasible** to certify — but with a decisive instrument caveat below.

**Analysis-only / PURE-CPU.** No HF Job, no submission, no served-file change, no model load. `analysis_only=1, official_tps=0, no_hf_job=1, fires=0`. Locked `int4_g128_lmhead`@126.378 untouched. Reuses my #714/#710 power-calculus harness (Wilson, Clopper-Pearson, `min_n_point_clears`, `min_n_for_power`) verbatim + the banked anchors.

**Public/internal evidence used:** denken #714 (`fpbp6pcn`, pairing-invariant 2889/9828), lawine #703 (`5r027mc3`, four-leg as-applied gate panel), fern #659 (`nmjvtfov`, int8-locus greedy 0.450 / sampled 0.410), kanna #699 (`jqecrucm`, int4/int8 sampled engine-fragility → greedy basis), denken #710 (`66rhys58`, routeB 0.42131).

---

### (1) The two fire-gate readings, as explicit decision rules

| Reading | Decision rule | Evidence required | One-shot-fire (#481) implication |
|---|---|---|---|
| **POINT-clearance** (lawine #703 as-applied) | served config clears IFF **point** p̂ ≥ 0.420 | a single point estimate at the eval budget; CI = honesty caveat, does **not** gate | the point is the unbiased best estimate; for a single irreversible decision the EU/posterior-mean-optimal action gates on the point. A lower-bound rule is a type-I-error control device for a **repeated** regime. |
| **CI-certification** (#710/#714) | served config clears IFF Wilson/CP **lower bound** ≥ 0.420 at the budget | n ≈ z²·p(1−p)/m² draws (m = margin) to push the 95% LCB over the bar | guarantees-at-95% the model is above the floor (risk-averse). On AIME it is **unachievable** at any near-bar margin (see frontier). |

### (2) As-applied reconcile (#703) → `asapplied_gate_is_point = 1`

The operative rule the program actually used: **FAIL if even Wilson-HI < gate** (AIME 0.4022 < 0.420 → decisive *level* gap, basis-independent); **PASS if point ≥ gate** (GPQA-D 0.4747) with a CI-lo straddle (0.406 < gate) recorded as an **honesty caveat, not a fail**.

Decisive test — counterfactual: under the CI rule, GPQA-D (CI-lo 0.406 < 0.420) would **also fail** → the panel would show **2** failing legs, not the recorded **1**. The recorded single-leg (AIME-only) failure is only consistent with the **POINT** rule. `legs_failing_under_point=1`, `legs_failing_under_ci=2`, `ci_rule_would_collapse_panel=1`.

### (3) Certification-budget frontier (Wilson-lo point-clears / 95%-power; GREEDY basis)

| Config | point | margin | **n (point-clears)** | n (95%-power) | edge-feasible (≤1500)? |
|---|---|---|---|---|---|
| **full-g32** (Lane 1) | 0.438 | 0.018 | **2889** ✅repro #714 | 9851 (band ∋ banked 9828) | ❌ no |
| route-B (Lane 1 fused) | 0.42131 | 0.00131 | **545,295** ✅repro #714 | ~½M | ❌ no |
| **int8-locus** (Lane 2) | 0.450 | 0.030 | **1040** ← *primary metric* | 3566 | ✅ yes |

Parametric n(p) lookup (so any measured point → verdict by lookup): margin 0.002→0.050 spans n = 233,945 → 375. **Smallest p whose point-clears budget ≤1500 = 0.446**, so int8's 0.450 qualifies and full-g32's 0.438 does not. Full table on W&B (`frontier_npc_p420…p470`).

**⚠ INSTRUMENT CAP (the load-bearing refinement).** On the **greedy** basis (kanna #699 forces it: int4/int8 sampled decode is engine-fragile on vLLM 0.22.0), AIME yields **one deterministic draw per problem** → n is capped at the **problem pool (~30–60), not by compute**. Re-seeding gives *zero* new information (greedy is deterministic — ubel #702's "5 seeds × 60" collapses to **60 unique** outcomes). int8's 1040 budget is **17.3×** the 60-problem cap; full-g32's 2889 is 48×. And sampled draws cannot backfill n: int8 **SAMPLED point 0.410 < gate** → draws whose own point is sub-gate cannot certify ≥0.420. **`any_lane_greedy_pool_feasible = 0`** — i.e. on the actual AIME instrument the CI reading certifies *no* lane. int8 is "edge-feasible" **only** in the conditional world where the math pool is expanded to ~1000+ comparable-difficulty problems.

### (4) Two-world consequence table

| | full-g32 | route-B | int8-locus | selective-g32 |
|---|---|---|---|---|
| **POINT world** (clears IFF point ≥ 0.420) | ✅ 0.438 | ✅ 0.4213 | ✅ 0.450 | ✅ IFF measured point ≥ 0.420 (ubel #702 pending) |
| **CI world** (edge-feasible ≤1500) | ❌ 2889 | ❌ ~½M | ⚠ 1040 (only on expanded pool) | depends on point |

`lane1_alive_on_point = 1`; `int8_only_edge_feasible = 1` (among the lanes, on the ≤1500 budget); `any_lane_greedy_pool_feasible = 0` (none on the real ≤60 instrument).

**Decision-grade sentence:** *Under the CI reading the int8-locus's larger margin (0.030 vs 0.018) makes it the only recovery config that is both point-clean (greedy 0.450) and edge-feasible on the point-clears budget (1040 ≤ 1500); full-g32 (2889) and route-B (~½M) are not. But on the strict greedy AIME instrument (pool ≤60) even int8 is ~17× over-budget and int8-sampled is sub-gate, so CI-certification is reachable for no lane without an expanded math pool — int8 is the least-infeasible.*

### (5) Verdict-conditional downstream note (both worlds)

- **If POINT-gate (the adjudicated reading):** the four Lane-1 measurement cards (ubel #702 selective-g32, fern #713 g32-locus, stark #711 shape, land #712 strict-#319 identity) remain **fire-relevant on their points**. #710/#714 `DEAD` bounds only the optional-rigor CI standard the fire-gate does **not** require → **do not over-kill Lane 1**; the points are decision-grade.
- **If CI-gate:** Lane 1 is dead (full-g32 n≥2889, route-B ~½M — infeasible) and only Lane 2 (int8-locus, n=1040, pool-expansion-contingent) is even edge-feasible → the **int8/mandate ruling becomes strictly load-bearing**.

### Recommendation for the one-shot competition fire (#481) — human makes the final call

**Gate on the POINT, with a margin preference, and report the Wilson CI as risk disclosure.** Rationale: **(a) decision theory** — a single irreversible board post is one expected-utility decision, not a repeated-certification regime; the EU-optimal action gates on the posterior mean (~the point); a conservative LCB rule is justified by repetition a one-shot lacks. **(b) consistency** — the program's own four-leg panel is point-gated (GPQA-D only point-passes); switching to CI now retroactively fails GPQA-D and collapses the panel, not just AIME. **(c) feasibility** — CI-certification of AIME at a near-bar margin is unachievable on this instrument (greedy pool ≤60 vs 1040–2889 needed); a standard no config can meet is not an operable fire-gate. **But** irreversibility argues for the **largest-margin point** among clearing configs (a thin 0.438 carries ~30–40% uncertified P(true<0.420) at n=60): among point-clearing recovery configs prefer maximum margin (int8 0.450 > full-g32 0.438) and disclose the CI. If the human instead elects the conservative CI standard for the irreversible post, Lane 1 is dead and the int8-mandate ruling becomes load-bearing.

---

### Reproducibility

```bash
cd target/
python -m research.validity.gate_semantics_cert_budget.gate_semantics_cert_budget \
  --self-test --wandb-name "denken/gate-semantics-cert-budget" \
  --wandb-group "gate-semantics-cert-budget-denken"
```

- **Self-test:** 18/18 PASS (reproduces #714: full-g32 **2889**, route-B **545295**, power-band ∋ **9828**; boundary `wilson_lo(p,n)>gate≥wilson_lo(p,n−1)` for all lanes; frontier monotone; p=gate unresolvable; asapplied=point).
- **Peak memory:** 10.59 MiB (pure CPU; no torch/vllm/numpy/scipy imported — a model load would be GB-scale).
- **W&B run:** `kyp59hd6` (group `gate-semantics-cert-budget-denken`, state `finished`).

### What happened — honest analysis

The card resolves the load-bearing ambiguity #714 surfaced. The program **has** gated on the point (proven by the GPQA-D counterfactual, not asserted), so `POINT_GATE_LANE1_ALIVE` is the adjudicated world and #710/#714's `DEAD` is correctly scoped to the *stricter* CI standard. The certification-budget frontier reproduces #714 exactly and extends it to a parametric lookup + the int8 number (1040). The genuinely new finding — beyond the PR's framing — is the **greedy instrument cap**: because the valid int4-precision basis is greedy (deterministic), the AIME problem pool (≤60) binds n far below even the smallest lane budget, so CI-certification is operationally impossible for **every** lane on the real eval, not merely expensive for Lane 1. int8 is "edge-feasible" only relative to a hypothetical expanded ~1000-problem math pool. This *strengthens* the recommendation: the CI standard isn't just conservative, it's un-meetable on this instrument, so the point gate is the only operable one — with margin preference and CI as honest risk disclosure carrying the irreversibility concern.

### Suggested follow-ups

- **Pool-expansion feasibility for the CI world:** if the human wants the option of CI-certifying int8, price whether a ~1000-problem comparable-difficulty competition-math pool (e.g. MATH-hard / AMC / past-AIME aggregate) is admissible on the #31 basis and gives iid greedy draws — the only path that makes `int8_certify_n_pointclears=1040` physically reachable.
- **Loss-asymmetry sizing of the margin:** formalize the false-fire vs missed-fire (stay-at-126.378) loss ratio to set the *point* threshold above 0.420 (the margin the recommendation calls for), turning "prefer max margin" into a concrete bar.
- Hand the n(p) lookup to ubel #702 / fern #713 / lawine #715 so their measured points map to a certify/don't verdict instantly (full table on W&B).
