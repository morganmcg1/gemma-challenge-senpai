STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["2evhfxi7"],"primary_metric":{"name":"cb3_supply_lift_mtp_self_test_passes","value":1},"test_metric":{"name":"honest_supply_lift_tps_m1_floor","value":42.91}}

## Results

**Verdict: the supply route SURVIVES draft-separation. +38.3 was +5.69 TPS optimistic (~15%), but the honest verify-body-only lift still clears BOTH #383 targets at every cell — decision-critical answer is YES, on #388's banked M=1 number, independent of #391.**

Pure-CPU analytic card (0 GPU, 0 official TPS, 0 HF Job, no served-file change). Baseline **481.53 TPS / PPL 2.3772 UNCHANGED**. W&B `2evhfxi7` (group `cb3-supply-lift-mtp-honest`). Self-test **32/32** (≥20 required). New file: `research/validity/cb3_supply_lift_mtp_honest/`.

### The draft-fraction optimism, removed

The served stack is MTP K=7 spec-decode: **1 drafter forward** (separate small model, NOT cb3-quantized → un-shrunk) + **1 verify forward** (M=8 target body, cb3-shrinkable). #388's headline applied the speedup to `f_comp = 1 − f_attn − f_lmhead = 0.8825`, which **lumps the #378 `draft` fraction (0.1201) into the shrinkable body**. The honest fix credits cb3 only to `f_verify_body = 1 − draft − lm_head − attn = 0.7624` and composes through the acceptance-weighted step `TPS = (1 + E[accepted]) / T_step` (only `T_verify` shrinks).

| quantity | M=1 tier (1.1234×, #388 banked) | M=8/roofline tier (1.2744×, #391 PENDING) |
|---|---|---|
| #388 headline (credit-whole-complement, off-the-shelf) | **+38.34** | +83.82 |
| **honest verify-body-only, off-the-shelf (357.32)** | **+32.65** | +70.17 |
| **honest verify-body-only, floor (469.68)** | **+42.91** | +92.24 |
| **`optimism_gap_tps`** (headline − honest, off-the-shelf, M=1) | **+5.69** (14.9% of +38.3) | — |

`E[accepted]` from the #289 ladder = **2.851** → `E[T] = 3.851`, which matches the deployed `E_T_REALIZED = 3.844` to **0.19%** (independent cross-check that the ladder composition is well-calibrated). The explicit MTP-loop decomposition reproduces `base × lift_factor(speedup, 0.7624)` exactly at every cell (`E[T]` cancels) — the value of the explicit form is that it pins the draft component **fixed** across the cb3 shrink.

> NB: #388's own *gate* (`closes_383_supply_gap_floor`) already used the honest body-only number (32.647); only its **headline** carried the optimistic +38.3 complement. This card makes the headline honest and extends it to the floor base, the M=8 tier, and the combined private-500 route.

### Does the honest lift clear #383's targets? (deliverable 4)

#383 supply targets (lifts above the 469.68 floor): **+17.22 floor (joint) / +23.75 robust (E[T]-only)**.

| cell | honest lift | clears +17.22 | clears +23.75 |
|---|---|---|---|
| off-the-shelf, M=1 (strictest) | +32.65 | ✅ | ✅ |
| floor, M=1 (**headline**) | +42.91 | ✅ | ✅ |
| off-the-shelf, roofline | +70.17 | ✅ | ✅ |
| floor, roofline | +92.24 | ✅ | ✅ |

`honest_lift_clears_383_floor = True`, `honest_lift_clears_383_robust = True`. **Every cell clears both** — even the strictest (off-the-shelf, M=1, +32.65). The draft-separation does NOT flip the supply lane's readiness; it strengthens it (per the #383/#387 honest-band discipline).

### Combined supply+demand route to private-500 (deliverable 5)

Inverting #387's `reprice_anchor_band` at `base = 469.68 + honest_supply_lift` (`pstar = 524.95` public for private-500; round-trips #383's demand-alone **+0.0572 @ 469.68**, which busts the +0.031 budget):

- **M=1 tier:** supply 469.68 → **512.60**; `residual_demand_dcov_honest = +0.0117` (**38% of the #336 +0.031 budget**) → `combined_route_reaches_500_honest = True`.
- **roofline tier:** supply 469.68 → **561.93** > 524.95 → **supply-alone** reaches private-500 (residual demand ≤ 0).

This is the number fern #357 needs: the **measured-but-honest** cb3 supply lift (M=1) + a **+0.0117 demand sliver** (well inside the #336 budget) closes private-500. The supply lane is the load-bearing lever; demand only has to supply the remainder.

### Deliverables (W&B `summary/`)

`honest_supply_lift_tps` M=1 [off +32.65 / floor +42.91], roofline band [off +70.17 / floor +92.24]; `optimism_gap_tps = +5.69`; `honest_lift_clears_383_floor = True`; `honest_lift_clears_383_robust = True`; `residual_demand_dcov_honest = +0.0117`; `combined_route_reaches_500_honest = True`; `cb3_supply_lift_mtp_self_test_passes = True` (**PRIMARY**, 32/32).

### Command

```
cd target/ && .venv/bin/python research/validity/cb3_supply_lift_mtp_honest/cb3_supply_lift_mtp_honest.py --self-test
cd target/ && .venv/bin/python research/validity/cb3_supply_lift_mtp_honest/cb3_supply_lift_mtp_honest.py \
  --wandb_group cb3-supply-lift-mtp-honest --wandb_name denken/cb3-supply-lift-mtp-honest
```

Peak memory: negligible (stdlib `math`/`json`, no torch/GPU). W&B run `2evhfxi7`.

### What happened

The hypothesis asked whether +38.3 is optimistic because it credits the un-shrunk MTP drafter forward. It **is** optimistic — by +5.69 TPS (14.9% of the headline) — but the honesty correction does **not** change the conclusion: even at the conservative M=1 tier, the honest verify-body-only lift (+32.65 off-the-shelf / +42.91 floor) clears both #383 targets, and the combined route (cb3 M=1 + a 38%-of-budget demand sliver) reaches private-500. The self-test round-trips #388's +38.34 exactly under the credit-whole-complement assumption, then shows the honest number — so the gap is purely the draft-fraction credit, not a model change. The #289 ladder E[T] = 3.851 matching the deployed 3.844 (0.19%) gives confidence the acceptance composition is calibrated, not invented.

The honest end-to-end model is algebraically `base × lift_factor(speedup, f_verify_body)` because E[T] cancels between numerator and base — but writing it through the explicit `(1+E[accepted])/T_step` decomposition is what makes the draft-separation auditable (the `T_draft` component is held fixed while only `T_verify_body` divides by the speedup).

**Caveat (inherited, out of scope):** this composes the SPEED lift only. It treats cb3 as acceptance/identity-neutral (the #372 PPL-feasibility gate 2.3812 ≤ 2.42 is the precondition, same assumption as #388). Whether cb3-on-target preserves byte-exact greedy-decode identity is a **separate gate** not adjudicated here.

### Suggested follow-ups

- **Sharpen the M=8 tier when lawine #391 (`cb3-m8-verify-body-speedup`) posts.** The verify is M=8, so the correct cb3 speedup sits in [M=1 1.1234×, roofline 1.2744×]; M=1 is the conservative end. The band tightens to a point once #391 measures the realized M=8 BW-efficiency. The decision (draft-separation) already holds at M=1, so #391 only sharpens the combined-route margin, not the verdict.
- **The cb3-on-target greedy-identity gate.** Before any of this becomes deployable, cb3's 3.2369-bpw target body must be shown token-identical to int4-Marlin at M=8 decode (or the verify must re-prove greedy identity). That is the real blocker for the supply lane, not the speed composition.
- **Recompose against the off-the-shelf vs floor base disagreement.** The #378 band is wide ([357.32, 469.68]); the residual demand at M=1 swings from supply-alone-insufficient (off-the-shelf) to +0.0117 (floor). Pinning which base is operative (the lm_head-BI determinism corner) would collapse the combined-route uncertainty.

### Public evidence used

This is a 0-submission internal supply-route composition card; **no public leaderboard method was reproduced**. I checked the shared digest (`?as=senpai`): the public a10g frontier is ~489 TPS (split-KV verify stacks, e.g. `frantic-penguin` / `hayai-agent`), all PPL ≈ 2.3774 ≤ cap — confirming the public ceiling sits below the **private-500** bar this card targets. The card is grounded in the internal frontier (PR #52, 481.53 / `2x9fm2zx`) and lawine #388's banked microbench (`g5lfdpgw`), per the PR baseline table.
