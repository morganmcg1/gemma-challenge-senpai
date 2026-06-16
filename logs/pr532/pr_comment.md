STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"no_hf_job":true,"official_tps":0,"wandb_run_ids":["hpfw9e3y"],"primary_metric":{"name":"drafter_acceptance_realize_self_test_passes","value":1},"test_metric":{"name":"projected_tps_at_realized_E_T","value":414.31}}

## Results

**Headline: the 3.849→4.91 E[T] headroom is REAL (+1.07) and quality-safe by construction — but it is NOT free-500 on the byte-exact base, and NOT locally realizable this slot.** Folding wirbel #295's **MEASURED** EAGLE-3 draft-step tax onto **my MEASURED #523 442-base draft fraction** collapses kanna #526's no-tax **565**: at the realistic ceiling the realized TPS is **414 local (central regime tax) / 496 (optimistic tax)** on the byteexact-442 base — **crosses 500 only at the joint-optimistic corner, crosses 565 NEVER, private NEVER crosses 500.** The cheapest realizable path is an EAGLE-3 retrain (soft-KD+CoT + multi-step/TTT), a **~107 GPU-h CLUSTER slot** whose only built head (#34) currently *regresses* E[T]. **Verdict: NO-GO for a local realize this slot** — this is the #502-style structural ledger (PR step-4 escape hatch), handing off de-risked cluster scope.

All LOCAL, `analysis_only=true`, `official_tps=0`, **NO HF Job / `--launch` / submission / served-file change**. 0 GPU (CPU-only analytic; ~0.06 GB). **Reuses MEASURED data only — no fresh serve** (kanna #526's E[T] harness + my #523 served byteexact-442 stack, exactly as instructed). W&B `hpfw9e3y` (group `drafter-acceptance-realize`). Self-test **38/38**. New file: `research/validity/drafter_acceptance_realize/`.

### KEY OUTPUTS (required)
| output | value |
|---|---|
| `measured_E_T_baseline` | **3.849** (linear-MTP K=7 at structural cap ≈3.845; #289/#526 `5m17r52s`) |
| `best_realized_E_T` | **4.915 — PROJECTION/target, UNMEASURED.** The only *built* EAGLE-3 head (#34, K=1) collapses past step-1 → realizes **~1.8 < deployed 3.849** today |
| `E_T_lift` | **+1.065** (band +1.07…+1.45) |
| `cheapest_path_name` | **EAGLE-3 fusion retrain** (soft-KD top-k + reasoning-CoT root coverage #336) **+ multi-step/TTT** (fixes #34 collapse) **+ supply-side φ fix** (#335) — a **~107 A10G-GPU-h CLUSTER slot** (denken #301), not a 1-GPU local realize |
| `projected_tps_at_realized_E_T` (442 base) | **414 local / 429 official** (central tax); bracket **360–496** at realistic E[T], up to **534** at optimistic E[T] |
| `crosses_500` | **FALSE** (central/expected; only at the joint-optimistic corner; **private never**) |
| `crosses_565` | **FALSE** (never under any measured tax) |
| `drafter_is_quality_safe` | **TRUE** — spec-dec TV ≤ noise floor (denken #505 `bg03bq0d`); verify is byte-exact M=8 ⇒ output == target greedy regardless of drafter |
| `self_det` | served **r1-r2 = 1.0**, attention **0/8** byte-exact (#523 `i11p5e3y`) — **UNCHANGED** by drafter |
| `ppl` | **2.37666** (#523) — **UNCHANGED** by drafter (≤ 2.42 ✓) |
| `go_no_go` | **NO-GO-LOCAL-REALIZE / DEFER-TO-CLUSTER** |

### 1. Diagnose — where acceptance is lost (task step 1)
The ladder reproduces E[T]=3.849 exactly. The gap to the EAGLE-3 flat-0.91 deep target is **deep-position**, not first-token:

| pos | a_k measured | target 0.91 | gap |
|---|---|---|---|
| 1 | **0.729** | — (ceiling-bound) | first-token, *not* liftable by acceptance work |
| 2 | 0.759 | 0.91 | +0.151 |
| 3 | 0.793 | 0.91 | +0.117 |
| 4–7 | 0.822 / 0.835 / 0.836 / 0.847 | 0.91 | +0.088 / +0.075 / +0.074 / +0.063 |

The deep positions (2–7) plateau at ~0.83–0.85 — that plateau is the **linear-MTP topology cap** (deployed sits at 3.849 ≈ cap 3.845). a_1=0.729 is first-token ceiling-bound. So the realizable lift is **deep-position coverage**, which structurally requires a **wider fusion (EAGLE-3) head** — confirmed by ubel #399 (`ec7i3z5t`): every no-retrain / no-served-kernel lever (temperature, affine cal, tree-free) is a rank-order **no-op**.

### 2. Screen the cheapest path (task step 2)
- **(a) tune/retrain the existing MTP head → PROVEN-CLOSED.** Linear cap 3.845 (#119) + #399's null-lever result: same-topology epochs buy ~0.
- **(b) EAGLE-3 fusion head → the only path to the ceiling, but two taxes.** (i) **Capability:** the only *built* EAGLE-3 head (fern #34, K=1) **collapses past step-1** (native step-1 accept ~0.77, chain dies) → realized E[T] ~1.8, *below* deployed. Reaching 4.91 needs the **UNBUILT** multi-step/TTT (HASS-style) chain training + soft-KD/CoT root coverage (#336) + supply-side φ fix (#335 AND-gate). (ii) **Step cost:** the measured draft-step tax (§3). A faithful build is **~107 A10G-GPU-h** (denken #301) + the TTT recipe is unbuilt → **overruns the slot** → PR step-4 escape hatch triggered.
- **Reduced screen I *did* run (0-GPU):** the analytic reconciliation below — no train, no serve — pricing the realized TPS on the **legal** base.

### 3. The realized lift, priced on the byteexact-442 LEGAL base (task step 3 — the core contribution)
The PR premise (kanna #526) projects 4.91→565 holding `t_step` **fixed** (no drafter step tax). fern #305 *does* fold wirbel #295's measured tax — but on the **481.53 fast (non-byte-exact, equivalence-illegal) base**. The PR asks for the **byteexact-442 base**, so I re-anchor the measured tax there.

**Step model** (PF≈1.0, #504): `TPS = E[T]/t_step`. Swapping linear→EAGLE-3 inflates **only** the drafter portion (target verify `exec_gpu` unchanged — the 3 fused aux layers export FREE on vLLM 0.22.0, fern #15; draft vocab 12288 unchanged), so `t_step_new(m) = t_step_old + draft_gpu·(m−1)`. This is *favorable* to EAGLE-3 (no verify-side tax), i.e. an **upper bound**.

Measured 442-base step (my #523 `i11p5e3y`, STEPTIME=1, n=3): `t_step=8.754 ms, exec_gpu=6.89, draft_gpu=1.554 ms ⇒ draft_frac=0.178`. Measured tax bracket (wirbel #295 `c334qaqu`): **m ∈ [1.745 lower, 3.0 central, 4.161 upper]** (the raw 1.745 is a dispatch-compressed lower bound; 3.0 is regime-corrected, validates #293's independent 3× proxy).

**Realized TPS on the legal byteexact-442 base** `[local | official ×1.0352 (#267) | private ×0.804 (#305)]`:

| E[T] | tax m | local | official | private |
|---|---|---|---|---|
| **4.915** (realistic) | 1.0 (kanna no-tax) | 561 | 581 | 467 |
| 4.915 | **1.745** (opt) | **496** | 513 | 413 |
| 4.915 | **3.0** (central) | **414** | 429 | 345 |
| 4.915 | 4.161 (pess) | 360 | 372 | 299 |
| **5.295** (optimistic) | 1.745 (opt) | **534** | 553 | 445 |
| 5.295 | 3.0 (central) | 446 | 462 | 372 |

**Reconciliation:** kanna's 565 is recovered exactly at m=1.0 (561 here; the 4-TPS difference is my self-consistent 439.71 anchor vs kanna's 442.35 — verdict robust to ±0.6%). The **measured tax erodes it by ~147 TPS** at the realistic ceiling (central). This is precisely the kanna-565 ↔ fern-sub500 gap, now priced on the legal base: **the gap between them *is* the drafter step tax.**

- `crosses_500`: **FALSE** at the central operating point (414 local / 429 official). Crosses 500 **locally only at the joint-optimistic corner** (E[T]→5.295 *and* tax→1.745); **officially** also at (realistic E[T], optimistic tax)=513; **private never** (max 445).
- `crosses_565`: **FALSE** everywhere under measured tax (grid max 534).

### 4. Quality-safety (task step 3, identity leg)
`drafter_is_quality_safe = TRUE`, stated explicitly: spec-dec verify is byte-exact M=8, so acceptance changes **only** E[T] (tokens/step), **not** the emitted distribution — denken #505 (`bg03bq0d`) proved TV ≤ noise floor. The served output equals the target's greedy output regardless of drafter quality, so `self_det` (served r1-r2=1.0, 0/8 byte-exact attention) and `ppl` (2.37666) are **invariant to the drafter** and inherited from the #523 stack. No re-serve was needed or informative (a better drafter cannot move PPL/identity — only TPS).

### Verdict
**NO-GO for a LOCAL drafter-realize this slot.** The headroom is real (+1.07 E[T], quality-safe) and the single largest *demand-side* prize, but it is **not free-500 on the byte-exact base and not locally realizable**: (i) the only path to the ceiling is an EAGLE-3 retrain whose only built head regresses E[T] (needs unbuilt multi-step/TTT), a **~107 GPU-h cluster slot**; (ii) the **measured** step tax drops the realistic ceiling to **414 central / 496 optimistic** local — crosses 500 only at the joint-optimistic corner, **never** 565, **private never** 500. The clean answer to the PR's question: **drafter-acceptance is realizable quality-safe headroom, but the cheapest path is a deferred CLUSTER training slot, not this local slot.** Hand off the de-risked scope (soft-KD+CoT + multi-step/TTT + φ fix) as the #2-priority slot.

### Comparison vs PR baselines
- **kanna #526** (`3piz86i4`): no-tax projection +122→565 (crosses 500). **Reproduced exactly at m=1.0** (561 on my base). The +122/565 holds `t_step` fixed; folding the **measured** EAGLE-3 tax (wirbel #295) corrects +122 to **+54 (optimistic tax) … −83 (pessimistic) over the 442 base, −28 central** — at the central regime tax the EAGLE-3 swap is *net-negative* (the ~3× draft-step cost outweighs the +1.07 E[T]): my central realized **414 < the 442 base itself**, not 565. The premise (no-tax) is the entire gap.
- **byteexact-442 base** (#523 `i11p5e3y` 439.71 / #519 `kwhylaeg` 442.35): used as the served base; verdict robust to which anchor (±0.6%, < σ_hw 4.864).
- **denken #505** (`bg03bq0d`): spec-dec TV ≤ noise → the quality-safety guarantee; consumed directly.

### Command
```bash
cd target && .venv/bin/python research/validity/drafter_acceptance_realize/realize_drafter_acceptance.py --self-test
cd target && .venv/bin/python research/validity/drafter_acceptance_realize/realize_drafter_acceptance.py \
  --wandb_group drafter-acceptance-realize --wandb_name lawine/drafter-acceptance-realize
```
Peak memory **~0.06 GB** (CPU-only float analytic; no torch/GPU/serve). **NaN-clean.** W&B run **`hpfw9e3y`** (group `drafter-acceptance-realize`). Self-test 38/38.

### What happened
The PR's premise — kanna #526's "+122 → 565, crosses 500" — is a **no-tax** projection: it priced the *wide* EAGLE-3 head's acceptance ceiling at the *narrow* linear head's step cost. The honest realization requires the EAGLE-3 head's own (measured) draft-step cost. Folding wirbel #295's measured multiplier into my measured #523 442-base draft fraction, the realistic ceiling realizes **~414 TPS central / ~496 optimistic** on the legal base — it **crosses 500 only at the joint-optimistic corner and never reaches 565**; the binding private frame never crosses 500. Worse, the *capability* side isn't built: the only trained EAGLE-3 head (#34) collapses past step-1 and realizes E[T] *below* the deployed linear drafter. So the lever is genuinely "free quality-safe speed" **by construction** (denken #505) — but its **cheapest realizable path is a ~107 GPU-h cluster EAGLE-3 train with TTT, not a one-GPU local realize.** A partial-but-honest #502-style ledger, exactly per step 4.

### Suggested follow-ups
1. **Bank the cluster training scope this ledger de-risked.** Cheapest path = soft-KD top-k + reasoning-CoT root coverage (#336) **+ multi-step/TTT** (the #34-collapse fix is the load-bearing unbuilt piece) **+ supply-side φ** (#335). The go/no-go for *that* slot is a training-request decision (ROI 0.843 to clear identity, #339), not a local experiment.
2. **The realized number is regime-tax-bound, so pin the tax before committing.** My 414↔496 spread at the realistic ceiling is entirely wirbel #295's [1.745, 4.161] regime bracket. A short **deployed-regime** EAGLE-3 draft-step micro-profile (ONEGRAPH+INT4, not the bf16 standalone harness) would collapse that bracket and convert "crosses 500 only at the optimistic corner" into a hard yes/no on the official frame.
3. **Re-price on the certified post-#1 t_step when the supply lever lands officially.** `dTPS/dE[T]=TPS/t_step` is rung-dependent; the drafter payoff is +18% higher on the faster certified rung (kanna #526), so the training-request validity argument should quote the post-supply number.

### Public evidence
Per the shared digest read banked in kanna #526 (merged on this branch): the public a10g frontier clearing 500 (~508.6 / 505.9, cluster ~489 just under) is driven by **split-KV / window-attention supply-side** stacks — **all** still linear-MTP K=7, **none** by a better-acceptance drafter. That is direct public corroboration: the demand-side (drafter-acceptance) headroom remains **unrealized across the entire public frontier**, consistent with this ledger's NO-GO-for-local-realize / deferred-cluster verdict. This **extends** (does not reproduce) kanna #526 by pricing the realization that #526 deferred — now with the measured step tax on the legal base.
