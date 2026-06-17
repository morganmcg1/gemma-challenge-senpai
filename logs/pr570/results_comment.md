STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["d595j39x"],"self_det":true,"two_gate_satisfiable":false,"fire_decision":"NO-FIRE","quality_safe_319_safe_ceiling_tps":291.36,"quality_safe_ppl_only_ceiling_tps":299.29,"magically_free_floor_tps":311.25,"gap_to_ship_min":64.61,"self_tests_passed":32,"primary_metric":{"name":"quality_safe_ppl_only_ceiling_tps","value":299.29},"test_metric":{"name":"gap_to_ship_min","value":64.61}}

## Results — the 3 quality-safe head ceilings, reconciled to #553's corrected **299.29**: NO-FIRE, **UNCHANGED**

**TL;DR (read once).** The capstone (#561, `v74ad5jb`) cited the **superseded 292.1** projection for the precision head lever. wirbel #553 (`bo43du3w`) ground-truthed that lever served and it **over-realizes**: **+46.6 (1.217×)** → the plain-int4-head ceiling corrects **UPWARD to 299.29 local / 309.83 official** (+7.19 = **+1.48 σ_hw** ceiling-space, `lawine544_ceiling_confirmed=FALSE`). I re-derived the three banked quality-safe head-lever ceilings into one reconciled record with that corrected number. They order **candidate-verify 291.36 (#319 PASS) < plain-int4-head 299.29 (#319 FAIL, PPL-safe) < magically-free floor 311.25 (upper bound)** — and even the **loosest** (311.25) sits **64.61 below the 375.857 ship** and **188.75 below the 500 gate** → **`two_gate_satisfiable=FALSE`, the conjunction is UNCHANGED after the upward correction.** Every cited number is loaded + asserted against its on-branch merged W&B source artifact at runtime (`self_det=true`, **32/32**, `all_numbers_grounded=True`). Pure synthesis — `analysis_only=true`, `official_tps=0`, **0 GPU**, no HF job, no served-file change, BASELINE 481.53 untouched.

### The reconciled 3-ceiling table (PRIMARY)

| quality-safe ceiling | served_tps **local** | official_proxy (×τ_lo) | **#319** | gap_to_ship (375.857−local) | gap_to_500 (500−local) | source (W&B) |
|---|---|---|---|---|---|---|
| **candidate-verify** (identity-safe lever) | **291.36** | 301.63 | **PASS** | 84.49 | 208.64 | fern #560 (`ufv4nk21`) |
| **plain-int4-head** (PPL-safe, breaks identity) | **299.29** | 309.83 | **FAIL** | 76.57 | 200.71 | wirbel #553 (`bo43du3w`) |
| **magically-free floor** (upper bound) | **311.25** | 322.22 | N/A | **64.61** | **188.75** | lawine #554 (`fi8vr1nb`) |

- **candidate-verify** = cheap int4 top-8 nominator → exact bf16 verify of those 8 rows → re-argmax; `argmax_identity_rate=1.0` **IFF** the verify breaks exact-bf16 ties by **lowest vocab index** (vLLM `torch.argmax` first-index). Realized `cv_realized_quality_safe_tps=291.36` (+39.05 on the 252.31 anchor) — this **supersedes** the 305.4 *projection* #561 carried.
- **plain-int4-head** = the 262k-row int4 head GEMV, body+attn byte-identical. PPL-safe (`ppl_delta_int4−bf16 = −0.003`) but **breaks #319** (served greedy flips ~0.76% free-run) → a speed *ceiling*, not a shippable number. **This is the correction: 292.1 → 299.29.**
- **magically-free floor** = head bytes→0, body intact, fixed-overhead floor kept (#554) — an **unrealizable** upper bound; no real lever reaches it.

### The #553 correction, reconciled against #561 (PRIMARY)

| lever | #561 carried | this card (reconciled) | Δ | direction |
|---|---|---|---|---|
| plain-int4-head | **292.1** (lawine #544 projection) | **299.29** (wirbel #553 realized) | **+7.19** (+1.48 σ_hw) | **UPWARD** (the correction) |
| candidate-verify | 305.4 (fern #549 projection) | **291.36** (fern #560 realized) | −14.05 | downward (now measured) |
| magically-free floor | 311.25 (lawine #554) | **311.25** | 0 | unchanged |

The correction also **re-orders** the two head levers: the identity-**breaking** plain-int4-head (299.29) is now ~+8 TPS **faster** than the identity-**safe** candidate-verify (291.36) — exactly the cost of candidate-verify's extra exact-verify pass. Both still sit far below the floor and the ship.

### The two-gate NO-FIRE conjunction — re-confirmed UNCHANGED (PRIMARY)

`two_gate_satisfiable` is True **iff some quality-safe ceiling ≥ the ship AND ≥ the gate.** The loosest of the three (magically-free 311.25) clears **neither**: 64.61 below the 375.857 ship, 188.75 below the 500 gate. → **`two_gate_satisfiable=FALSE`, `fire_decision="NO-FIRE"`.** The #553 upward correction (292.1→299.29) **does not change this** — 299.29 is still 76.57 below the ship; the conjunction would only flip if a quality-safe ceiling exceeded 375.857, which the hardware floor (311.25) forbids. `correction_changes_conjunction=0`.

### KEY OUTPUTS (W&B `d595j39x`, group `quality-safe-ceiling-reconcile`)

- `quality_safe_319_safe_ceiling_tps` = **291.36** · `quality_safe_ppl_only_ceiling_tps` = **299.29** · `magically_free_floor_tps` = **311.25**
- `gap_to_ship_min` = **64.61** (from the loosest 311.25 ceiling) · `gap_to_500_min` = **188.75**
- `two_gate_satisfiable` = **false** · `fire_decision` = **NO-FIRE** · `correction_is_upward`=1 · `correction_changes_conjunction`=0 · `correction_delta_tps`=7.19 · `correction_delta_sigma_hw`=1.48
- `#319`: candidate-verify **PASS** / plain-int4-head **FAIL** / floor **N/A**
- `self_det=TRUE` (**32/32** self-tests), `all_numbers_grounded=True`, **peak GPU 0**, `analysis_only=true`, `official_tps=0`. `primary_metric = quality_safe_ppl_only_ceiling_tps = 299.29`.

### Self-test / grounding discipline (your #561 standard, made literal)

Each cited constant is **loaded + asserted against its on-branch, already-merged, W&B-verified source artifact** at runtime — no asserted-but-unchecked numbers:

- candidate-verify 291.36 ← `research/candidate_verify_realize/stage2_reproject.json` (`cv_realized_quality_safe_tps`, `argmax_identity_rate=1.0`, fern #560 `ufv4nk21`)
- plain-int4-head 299.29 / 309.83 / +46.6 / 1.217× / ppl −0.003 / `lawine544_ceiling_confirmed=False` ← `research/realized_anchor_tps/summary.json` (wirbel #553 `bo43du3w`)
- magically-free floor 311.25 ← `research/speed/fixed_overhead_ceiling/fixed_overhead_ceiling.json` (`fixed_overhead_bounded_ceiling_tps`, lawine #554 `fi8vr1nb`)
- capstone cross-check ← `research/two_gate_fire_decision/two_gate_fire_decision.json` (carries 311.25 + the superseded 292.1 + `two_gate_satisfiable=False`, lawine #561 `v74ad5jb`)

Plus arithmetic + conjunction assertions (ordering CV<plain<floor; all three < ship < gate; gap_to_ship_min = loosest's gap = 64.61; correction upward & non-conjunction-changing; NaN-clean). **32/32 passed.** W&B read-back confirmed all load-bearing summary fields MATCH.

### Comparison vs baseline (PR body)

| | PR body | this card |
|---|---|---|
| `quality_safe_319_safe_ceiling_tps` | 291.36 | **291.36** ✓ (grounded fern #560) |
| `quality_safe_ppl_only_ceiling_tps` | 299.29 (corrected from 292.1) | **299.29 / 309.83 official** ✓ (grounded wirbel #553) |
| `magically_free_floor_tps` | 311.25 | **311.25** ✓ (grounded lawine #554) |
| `gap_to_ship_min` | 64.61 | **64.61** ✓ (375.857 − 311.25) |
| `two_gate_satisfiable` | false | **false** ✓ (UNCHANGED after the +7.19 correction) |

### Exact command (LOCAL, analysis_only, 0 GPU, no fire)

```
cd target/ && .venv/bin/python research/quality_safe_ceiling_reconcile/quality_safe_ceiling_reconcile.py \
  --wandb-group quality-safe-ceiling-reconcile --wandb-name lawine/quality-safe-ceiling-reconcile
```

Artifact `research/quality_safe_ceiling_reconcile/quality_safe_ceiling_reconcile.json`. **Peak GPU 0** (no torch import, no served job, no microbench). W&B `d595j39x` · `analysis_only=true`, `official_tps=0`, no HF job. Cited runs: fern #560 `ufv4nk21` · wirbel #553 `bo43du3w` · lawine #554 `fi8vr1nb` · lawine #561 `v74ad5jb`.

### Public evidence used

Public digest (`GET /v1/digest?as=senpai`, pulled 2026-06-17): the message `20260617-044020-004_senpai.md` banks the candidate-verify lever's **305.4 read-bound projection** (W&B `p9ga96xo`, "NOT served") — this card uses fern #560's **realized 291.36**, which supersedes that projection. The live public **#1 is `ff-splitkv-frantic-fawindow-clean-v0-w256` at 508.63 TPS** with the unsafe **osoi5 lineage at #4–5 ~489.63** — all *above* the 481.53 baseline, which only **widens** the gap to #1 and **hardens** NO-FIRE; the challenge remains **PAUSED on downstream quality**, the precise reason this is a two-gate decision. This card **synthesizes** banked evidence (does not re-measure).

### What happened

The card did exactly what it asked: it brought the capstone numerically **current** without changing the verdict. wirbel #553's correction is genuinely **upward** (+7.19 TPS, +1.48 σ_hw over the 292.1 projection — lawine #544's +38 speed lever is real and then-some on a served path), and it even re-orders the two head levers (the identity-breaking plain head is now faster than the identity-safe verify head, by the cost of the verify pass). But the conjunction is **robust**: every one of the three quality-safe ceilings — including the unrealizable 311.25 upper bound — sits below the ship and the gate, so `two_gate_satisfiable` stays **FALSE**. The closing artifact now cites the realized **299.29**, not the superseded 292.1; the reconciled record is the single source of truth for the three quality-safe ceilings feeding #561.

### Suggested follow-ups

- **Re-point #561's `base_fullhead_quality_safe_ceiling_tps_strict` (292.1) at this reconciled 299.29** if the capstone artifact is ever re-emitted — the verdict text already holds (gap 76.57 vs the strict-line 83.76 it quoted), but the number should read current.
- **When fern #566 lands the served candidate-verify e2e number**, drop it into the `candidate-verify` row in place of the 291.36 re-projection (`cv_served_tps_is_measured=False` is the one honest residual) — predicted ~291 < ship, so it confirms NO-FIRE either way.
- **Do not re-open any head lane** — precision (over-realizes but #319-breaking), candidate-verify (identity-safe but slower), and the magically-free floor are now all reconciled below the ship; the three-ceiling band [291.36 … 311.25] is closed.
