STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["iv9i2wks"],"primary_metric":{"name":"cb3_conservative_k_self_test_passes","value":1},"test_metric":{"name":"m8_lift_at_kstar","value":15.604}}

## Results

**Verdict: cb3 IS conservatively deployable. Bankable k\*=229 → served-regime (M=8) supply lift = +15.60 TPS — essentially the full headline +15.67, because the lift-vs-k curve is FLAT here. The supply lane is NOT dead. Residual gap-to-500 after conservative cb3 = 16.93 TPS (cb3 closes 48.0% of the 32.53 strict gap); demand must supply the rest.**

Analysis-only re-cost card: **0 official TPS, no HF job, no served-file change, no cb3 kernel build**. Deployed baseline **481.53 TPS / PPL 2.3772 / 128÷128 UNCHANGED** (PR #52, W&B `2x9fm2zx`). PPL gate **≤2.42**; self-imposed conservative bar **≤2.41** (~0.01 bankable margin). Self-test **30/30** (≥20 required). W&B `iv9i2wks` (group `cb3-conservative-k-deployable-lift`). New file: `research/validity/cb3_conservative_k_deployable_lift/`.

### 1. k\* — the bankable PPL-safe point

I re-ran the #394 held-out-worst + OOD PPL-vs-k harness on a dense integer grid (k=205…247, plus coarse anchors). k\* is defined as the **largest-lift k whose held-out WORST-seed PPL AND OOD PPL are both ≤ 2.41**, taken **robustly** (no positive-lift k below it may breach — so a noise dip cannot resurface on private).

| | held-out worst | OOD | binding | margin→2.42 | margin→2.41 |
|---|---|---|---|---|---|
| **k\*=229 (bankable, robust)** | **2.3780** | **2.4067** | OOD | **+0.0133** | +0.0033 (OOD) |
| k=232 (the "still-clears" point) | 2.3843 | **2.4141** | OOD | +0.0059 | **−0.0041 (breach)** |
| k=236 (largest-clearing, rejected) | 2.3959 | 2.4099 | OOD | +0.0101 | +0.0001 |

`k_star=229`, `heldout_worst_ppl_at_kstar=2.3780`, `ood_ppl_at_kstar=2.4067`, `ppl_margin_to_242_at_kstar=+0.0133`, `cb3_supply_lane_dead=False`.

**Two honest corrections to the premise:**

- **k=232 is gate-legal but NOT conservatively bankable.** The PR premise ("k=232 still clears ~2.39 held-out") is right *on the held-out leg* (worst-seed 2.3843 ≤ 2.41) but the **OOD leg breaches**: OOD PPL at k=232 = **2.4141 > 2.41** (`kstar_232_survives_worst_bar=False`). 2.4141 still clears the 2.42 *gate*, so k=232 isn't illegal — it just isn't bankable at the 0.01-margin bar. The honest bankable point is one notch tighter: **k=229**.
- **The OOD curve is non-monotone — and it realized exactly the winner's-curse trap this card exists to avoid.** `k_star_largest=236 ≠ k_star_robust=229`. Between them is a 4-wide OOD breach band (k=230–233, OOD 2.411–2.415) followed by a recovery (k=234–236 dip back under 2.41). Picking the naive "largest clearing k"=236 would perch the choice on a noise recovery *right after a breach band* — precisely the "winner's-curse resurfacing on private" the PR rationale warns against. The robust k\*=229 is conservative by construction. The lift cost of choosing 229 over 236 is only +0.39 TPS (15.60 vs 15.99) — not worth the private-data risk.

### 2. Re-cost the supply lift at k\*

The cb3 body-read byte ratio is a function of the cb3 **param fraction** φ(k) along the #372 ascending-sensitivity ordering (`set_config(order_ascending[:k])` puts the k least-sensitive body linears on sub-int4 cb3, rest on int4): `eff_bpw(k)=φ(k)·3.125+(1−φ)·4.125`, `r(k)=eff_bpw/4.125`. The three banked tiers (#388/#391/#392 closed forms; Marlin eff is ~flat in both M and k, so only r moves with k) re-price at φ(229)=0.8848:

| quantity | at k=232 (φ=0.888, headline) | **at k\*=229 (φ=0.885)** | Δ |
|---|---|---|---|
| body-read shrink | −21.53% | **−21.45%** | −0.08pp |
| M1 honest lift (#392, off-the-shelf) | +32.65 | **+32.51** | −0.14 |
| **M8 measured-floor lift (#391, served)** | **+15.67** | **+15.60** | **−0.06** |
| M8 floor-base lift (better-case) | +15.7 | +20.51 | — |

`bodyread_shrink_at_kstar=0.2145`, `m1_lift_at_kstar=+32.51`, `m8_lift_at_kstar=+15.60`. The re-cost reproduces #391's +15.67 (M8) and #392's +32.65 (M1) at φ=0.888 to within 0.003 TPS (self-test asserts).

**The flatness insight (why conservative ≈ headline):** 88.5% of body params are already on cb3 by k=229; the sensitivity-ordered allocation front-loads the param mass, so backing k off from 232→229 sheds only −0.08pp of shrink → **−0.06 TPS of M8 lift**. The lift curve is flat; the **PPL gate**, not the lift, is what binds. The "conservative penalty" on the supply number is negligible (~0.4%).

### 3. The deployable supply number + residual

**Conservative cb3 supply lift (served M=8, PPL-safe by construction) = +15.60 TPS.** Against the corrected strict base **467.48** (#393, `0q7ynumg`; strict gap-to-500 = **32.53**):

- `residual_gap_to_500_after_cb3 = 32.53 − 15.60 = **16.93 TPS**`
- `frac_of_gap_closed_by_cb3 = **48.0%**`

Conservative cb3 **alone does not close the 500 gap** — it closes just under half (48.0%) of the 32.53 strict gap. The **demand route must still supply +16.93 TPS** on top of it. (Public frontier corroboration: leaderboard #1 is ~489.66 TPS on the osoi5/skv64/ctk48 split-KV stack; 500 is still open, and the public demand lane — tree spec-decode — is itself contested, e.g. openevolve's 2026-06-15 finding that the e1-drafter tree caps ~2.57 tok/step **below** linear 4.28. So the +16.93 demand leg is not yet banked anywhere.)

### 4. Decision

| flag | value |
|---|---|
| `cb3_conservative_deployable` | **True** |
| `cb3_supply_lane_dead` | **False** |
| `k_star` | 229 |
| `m8_lift_at_kstar` (deployable supply) | **+15.60 TPS** |
| `residual_gap_to_500_after_cb3` | 16.93 TPS |
| `cb3_conservative_k_self_test_passes` (PRIMARY) | True (30/30) |

**Verdict: DEPLOYABLE-SMALL-LIFT.** A k\* with both legs ≤2.41 AND positive M=8 lift exists (k\*=229). The supply lane is alive; the honest deployable supply number is **+15.60 TPS**, ≈ the headline +15.67 (the conservative back-off costs ~0.4%). cb3 is a real, bankable but **partial** supply leg — it halves the gap-to-500, no more.

### What happened

The headline cb3 lift was never PPL-blocked *as a lift* — it was blocked only at the over-aggressive selected k. The honest re-cost shows the deployable number barely moves when you back off to a bankable k, because the lift-vs-k curve is flat where the PPL gate bites (param mass is front-loaded by sensitivity order). So the right framing for the program is **not** "cb3 supply is capped below +33, exact value un-costed" — it's "**conservative cb3 supply = +15.6 TPS, banked and PPL-safe; the +33 M1 number was always the wrong (un-served, M=1) regime.**" The served-regime (M=8) number is the one that feeds the route, and it's +15.6, closing 48% of the gap. The non-monotone OOD band is the methodologically important part: this card was designed to refuse a winner's-curse pick, and the data handed it exactly that scenario (a clearing k=236 sitting on a noise recovery). The robust k\*=229 is the defensible answer.

### Suggested follow-ups

- **Cost the demand leg's residual target precisely.** cb3 banks +15.6; the route needs +16.93 more. That is the number the demand route (descent / acceptance-lift / tree, per #289 ladder and the public e1-tree work) must hit at the served M=8 width. A "combined supply(+15.6) + demand(+16.93) = 500" portfolio card would close the loop.
- **OOD-stability of k\*=229.** The binding OOD margin to 2.41 is thin (+0.0033) on a noisy OOD slice (n=96 sharegpt). A larger / multi-seed OOD slice would tighten whether 229 holds or should drop a notch. Cheap (it's the same harness, bigger n).
- **Flag the PR-body k-direction parenthetical.** The "(most aggressive, smallest k)" / "larger k = smaller lift" phrasing is inverted vs the #372/#394 harness (larger k = more cb3 modules = more shrink = larger lift = higher PPL; the curve confirms it). I implemented the unambiguous operative goal (maximize lift s.t. both legs ≤2.41) and documented the discrepancy in the module docstring; result is unaffected.

---
**Repro / provenance**

```
cd target/ && .venvs/vllm022/bin/python -m research.validity.cb3_conservative_k_deployable_lift.cb3_conservative_k_deployable_lift --self-test
cd target/ && CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python -m research.validity.cb3_conservative_k_deployable_lift.cb3_conservative_k_deployable_lift \
  --wandb_group cb3-conservative-k-deployable-lift --wandb_name kanna/cb3-conservative-k-deployable-lift
```

- W&B run: `iv9i2wks` (project `wandb-applied-ai-team/gemma-challenge-senpai`); per-k PPL-vs-k curve logged as a Table.
- Peak GPU mem: **19586.5 MiB** (single A10G); elapsed **2356.8 s** (~39 min); 128 official PPL records + 96 OOD sharegpt_v3.
- Anchors reused (all tracked on `approval-gated-8gpu-20260613`): #394 `cb3_ppl_heldout_margin` (PPL/OOD harness), #391 `cb3_kernel_realized_bw` (`marlin_m8_hbm_eff=0.2559`, M8 measured-floor +15.666), #392 `cb3_supply_lift_mtp_honest` (M1 +32.647), #372 mixed-optimum ordering, #393 corrected base 467.48.
- Guards: `analysis_only=True, no_hf_job=True, no_served_file_change=True, no_launch=True, no_kernel_build=True, official_tps=0`.
