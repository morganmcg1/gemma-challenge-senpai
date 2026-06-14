<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Descent over-acceptance signature — the joint (E[T], greedy-violation) region
# land's measured tuple must fall in (PR #170 · wirbel)

**PRIMARY** `overaccept_signature_self_test_passes` = **True** (12/12 checks, NaN-clean)
**TEST** `et_inflation_at_unit_overaccept` = **1.0** (δ(ε=1); one extra node = +1.0 E[T], both topologies)

## Honest scope
Pure-analytic **CPU-only** magnitude complement to denken #158's binary detector. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays 481.53; **0 TPS**; greedy untouched by construction. Imports wirbel #160 (`x8vffgbs`) descent E[T]-DP ceilings + denken #158 (`opbbrnce`) measured operating point — **does NOT re-derive them**. Builds on wirbel #165 (`laxllfjl`), which named **BUG-2 the binding build-risk** (~19× BUG-1's E[T] lever).

## The model (node-counting; nothing re-derived)
Over-acceptance commits **ε** extra nodes per step past the true greedy boundary. Each over-accepted node is (1) one extra committed token → **+1 E[T]**, and (2) one greedy violation (past the boundary the draft has diverged). With the #160 greedy-exact ceiling `E[T]*`:

```
E[T](ε) = E[T]* + ε                  v(ε) = ε / (E[T]* + ε)
over-accept locus:   E[T] = E[T]*/(1 − v)   ⇔   v = 1 − E[T]*/E[T]
```

Imported ceilings (`x8vffgbs`): descent-only **E[T]\*=5.056405**, both-bugs **E[T]\*=5.206954** (cross-source match = True). Per-step committed cap = max_spec_len+1 = **8** (accept-all extreme).

## 1. Over-accept curve {ε, E[T], v}
**both-bugs (E[T]\*=5.20695):**

| ε | E[T] | v | exactness |
|---|---|---|---|
| 0.0000 | 5.20695 | 0.00000 | 1.00000 |
| 0.2500 | 5.45695 | 0.04581 | 0.95419 |
| 0.5000 | 5.70695 | 0.08761 | 0.91239 |
| 1.0000 | 6.20695 | 0.16111 | 0.83889 |
| 2.0000 | 7.20695 | 0.27751 | 0.72249 |
| 2.7930 | 8.00000 | 0.34913 | 0.65087 |

**descent-only (E[T]\*=5.05640):**

| ε | E[T] | v | exactness |
|---|---|---|---|
| 0.0000 | 5.05640 | 0.00000 | 1.00000 |
| 0.2500 | 5.30640 | 0.04711 | 0.95289 |
| 0.5000 | 5.55640 | 0.08999 | 0.91001 |
| 1.0000 | 6.05640 | 0.16511 | 0.83489 |
| 2.0000 | 7.05640 | 0.28343 | 0.71657 |
| 2.9436 | 8.00000 | 0.36795 | 0.63205 |

## 2. Inversion → 2D trustworthy region + over-accept locus
`trustworthy_region` = {(E[T], v): v ≤ v_tol AND E[T] ≤ E[T]*/(1−v)} (upper boundary IS the locus). **Degenerate at v=0**: v=0 ⇔ ε=0 ⇔ E[T]=E[T]* (UNIQUE) ⇒ **`max_et_inflation_at_v0` = 0.0** for both ceilings. Any E[T] > E[T]* **requires** v>0.

- both-bugs corners: greedy-exact `(5.20695, 0)`; locus@v_tol `(5.20695, 0.0)`.
- descent-only corners: greedy-exact `(5.05640, 0)`.
- noise-floor (v_tol = 1/65536 = one spurious token in the 128×512 benchmark budget): E[T] inflation budget = **7.95e-05** — i.e. even one spurious violation buys < 1e-4 E[T]; any meaningful E[T] readout above the ceiling is over-acceptance.

## 3. Cross-check vs denken #158's binary detector
#158's BUG-2 battery (`opbbrnce`): exactness 0.9194915254 = 217/236; violations **19/236**. The analytic v form (over-accepted committed fraction) reproduces the detector's per-token differential **exactly**:

- **`v_at_denken158_point` = 0.080508474576** = 1 − exactness 0.919491525424
- **`matches_detector` = True** (v = num_violations/total_committed = 1 - exactness_rate (exact complement))

The continuous `v(ε)` and the binary detector are the **same quantity**, agreeing at the same operating point. (#158's battery is synthetic stress, not the deployed tree — the cross-check is the count-identity, topology-free.)

## 4. The gate handed to land #71
```
land_tuple_in_trustworthy_region(E_T, v, E_T_star, v_tol=0, et_abs_tol=1e-9):
    return (v <= v_tol) and (E_T <= E_T_star/(1 - v_tol))
```
Strict (v_tol=0): **trustworthy ⇔ v=0 AND E[T] ≤ ceiling**. If land measures **E[T] > 5.2070 with v>0 → over-acceptance (FAILS greedy-exact)**, not a faster descent. Three regions: **TRUSTWORTHY** (v≈0, E[T]≤ceiling; under-accept is greedy-SAFE-but-slow) · **OVER-ACCEPT/BUG-2** (v>0, E[T]>ceiling on the locus) · **ANOMALOUS** (E[T]>ceiling but v≈0 — model-inconsistent, investigate).

**Composition with #158** — #158 = binary *"any violation?"* (catches substitution-only violations with no inflation); this = magnitude *"is the E[T] inflation explained by violation?"* (catches inflated-E[T]-read-as-headroom). Together they bound BUG-2 from both sides; a descent passing **both** is a trustworthy greedy-exact speedup.

## 5. Self-validate (PRIMARY)
12/12 checks pass (anchors reproduced at ε=0 with v=0; degenerate-at-v=0; locus inverts; #158 cross-check; δ(ε=1)=1.0 both topologies; E[T]>ceiling⇒v>0; NaN-clean). **`overaccept_signature_self_test_passes` = True**. **`et_inflation_at_unit_overaccept` = 1.0**.

## Public / banked evidence used
- wirbel #160 (`x8vffgbs`): descent E[T]-DP ceilings 5.0564 / 5.2070 (imported).
- denken #158 (`opbbrnce`): per-token `committed==in_step_target_argmax` binary detector, measured BUG-2 over-accept operating point exactness 0.91949 / 19 violations (imported).
- wirbel #165 (`laxllfjl`): SHARED index-map; named BUG-2 the binding build-risk (over-acceptance the only greedy-breaking path).

Official projection (context only; 0 TPS): the signature does **not** move the clear-500 bar — it certifies the measured E[T] feeding `official = K_cal·(E[T]/step)·τ` is TRUSTWORTHY, not over-accept-inflated.
