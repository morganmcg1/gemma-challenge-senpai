STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["on4u78ul"],"primary_metric":{"name":"lambda_bar_reconciliation_self_test_passes","value":1},"test_metric":{"name":"build_lambda_operative_gate","value":0.9780112973731208}}

## Results

**Headline: the three floating constraints do NOT contradict — they price two different risk axes. Recommendation: `build_lambda_operative_gate = 0.9780` (P95 VALIDITY, the bar a measured build MUST clear to launch) and `build_lambda_defended_target = 0.9808` (divergence-informed DRAW-RISK, the bar it SHOULD clear). Under #124 publish-first the operative gate is the launch trigger; the defended target is advisory headroom because private-draw risk is ACCEPTED post-hoc.**

This is a CPU-only analytic **reconciliation** leg (bank-the-analysis; PRIMARY = self-test). **0 TPS, BASELINE stays 481.53, authorizes nothing. NOT a launch. NOT open2.** Imports #239 and #243 VERBATIM (recomputed from their modules and round-tripped to the committed JSON at **0.0 error**, ≤ 1e-6).

### Step 1 — the three constraints on one λ-axis

| λ | risk axis | constraint | provenance | role |
|---|---|---|---|---|
| **0.978011** | VALIDITY (P95 LCB) | P95 validity bar | stark #191, via #239 `model["p95"]` | **operative gate (MUST clear)** |
| 0.978413 | VALIDITY (point mean=500 @ adverse NLS vertex) | worst-case-vertex floor under measured 0.73% div | #243 `lambda_floor_under_measured_div_linear` (breakeven 0.959780 holds) | consistency check — **confirms** the gate (+4.0e-4), does NOT raise it |
| **0.980752** | DRAW-RISK (f_priv-integrated P(draw<500)) | integrated-5% (divergence-informed) | #239 `lambda_integrated_risk5_divinformed`, Beta(2,1) | **defended target (SHOULD clear)** |
| 0.986058 | DRAW-RISK (f_priv-integrated P(draw<500)) | integrated-5% (uniform) | #239 `lambda_integrated_risk5_uniform`, Beta(1,1) | defended target IF divergence evidence discounted |

### Step 2 — the two risk axes (why they don't reduce to one number)

- **(i) VALIDITY** — *will the private re-draw be a valid ≥500 result?* The P95-LCB axis. Three independent derivations land here and **AGREE inside a 4.0e-4 band**: P95 bar 0.978011, #233 central λ_floor 0.978044, #243 worst-case-vertex floor 0.978413. Residual at the gate: `P_invalid = 0.05` (the P95 LCB construction, stark #191).
- **(ii) DRAW-RISK** — *probability the private draw lands below 500?* The integrated-5% axis (0.9808 div-informed / 0.9861 uniform). It sits **~3e-3 ABOVE** the validity cluster because the draw adds `sigma_draw = 7.391` on top of the mean — a draw can fall below 500 even when the mean clears it.
- **Why not one number:** VALIDITY prices whether λ̂ itself is a high-enough LCB; DRAW-RISK prices whether a Gaussian draw (σ=7.391) around the mean lands ≥500. Different objects → 0.9780 (validity) and 0.9808+ (draw-risk) do not collapse.

### Step 3/4 — the recommendation (ONE headline + risk statement)

> **build to λ̂ ≥ 0.9780** (OPERATIVE gate, P95 validity); at 0.9780 the accepted residual is: **validity P_invalid = 0.05** (P95 LCB construction) and **draw-below-500 P = 0.0589** (divergence-informed) — the draw risk is ACCEPTED under #124 publish-first as post-hoc defence. **SHOULD additionally clear the DEFENDED target λ̂ ≥ 0.9808** (divergence-informed 5% draw-risk) to drive draw-below-500 P down to 0.05.

- **Operative gate = 0.9780, not the higher 0.978413:** 0.9780 is a *distributional* 95% confidence bound (margin baked in); 0.978413 is a *point-estimate* mean=500 floor at the adverse NLS vertex — it confirms the location but carries no extra confidence, and the +4.0e-4 gap is inside the modeling resolution of the two derivations. The conservative, well-defined operative choice is the P95 bar.
- **Defended target = 0.9808 (divergence-informed), not 0.9861 (uniform):** lawine #232/#242 **MEASURED** the 0.73% near-greedy divergence, so realizable f_priv mass leans to the clean ceiling. Uniform ignores that measured evidence and over-states the bar by +0.0053.
- **Monotone/consistent:** defended (0.9808) ≥ operative (0.9780) ✓; uniform (0.9861) ≥ div-informed (0.9808) ✓.

### Step 5 — sensitivity

| perturbation | operative gate | defended target |
|---|---|---|
| **headline** (measured 0.73%, div-informed) | 0.9780 | 0.9808 |
| **discount divergence** (uniform prior) | 0.9780 (unchanged) | 0.9861 (**+0.0053**) |
| f_priv pinned at assumed ceiling 0.969107 (point) | — | 0.9700 (== #237 point λ_risk5) |
| f_priv pinned at grounded floor 0.957054 (point) | — | 0.9987 (near λ=1) |

- The **operative gate is invariant** to the f_priv prior (it lives on the validity axis, not the draw distribution).
- The **defended target is the prior-sensitive number:** discounting the measured divergence raises it +0.0053; pinning f_priv at a single endpoint instead of integrating spans 0.9700 (assumed) → 0.9987 (grounded). The integrated 0.9808 sits between, near the assumed end under the divergence-informed lean.

### Self-test (PRIMARY) — `lambda_bar_reconciliation_self_test_passes = True`

- (a) provenance: every #239/#243 headline reproduces from the module to **0.0 error** (≤ 1e-6) ✓
- (b) operative gate + defended target each pinned to a stated risk axis with a numeric residual (P_invalid=0.05; P(draw<500)=0.05) ✓
- (c) monotone/consistent: defended ≥ operative; uniform ≥ div-informed; validity cluster width 4.0e-4 < 1e-3 ✓
- (d) defended target delivers ~5% integrated draw-risk by construction; operative gate accepts strictly more (0.0589 > 0.05) ✓
- (e) NaN-clean ✓

### Comparison vs the PR baseline anchors

| anchor | PR body | this leg |
|---|---|---|
| P95 validity bar | 0.9780 | imported 0.978011 (operative gate) |
| #243 worst-case-vertex point | 0.978413 (≈ central 0.978044) | imported 0.978413 (confirms gate, +4.0e-4) |
| #239 integrated-5% div-informed / uniform | 0.9808 / 0.9861 | imported 0.980752 / 0.986058 (defended target) |
| publish-first breakeven | 0.959780 | imported unchanged (holds at the worst-case vertex) |

### Reproduce

```
cd target/ && CUDA_VISIBLE_DEVICES="" python \
  research/validity/build_lambda_bar/build_lambda_bar.py \
  --self-test --wandb_group build-lambda-bar-reconciliation \
  --wandb_name fern/build-lambda-bar
```

- **W&B run:** `on4u78ul` (wandb-applied-ai-team/gemma-challenge-senpai, group `build-lambda-bar-reconciliation`)
- **Peak memory:** 30.1 MiB (CPU-only)
- **summary.json fields:** N/A — no benchmark/draw this leg (0 TPS; `tps`/`ppl`/`completed`/`run_prefix` unchanged from the served baseline 481.53 TPS, PPL 2.3772, 128/128, PR #52). This is a pure analytic reconciliation.

### What happened — honest analysis

It worked, and the reconciliation is cleaner than the "three-way argument" framing implied. The decisive finding is that the four banked numbers split **2-2 across two risk axes**, and within the validity axis the three independent derivations (P95 LCB, #233 central λ_floor, #243 worst-case-vertex floor under the corrected 0.73% physics) cluster inside **4.0e-4** — they are not in tension, they corroborate. So the apparent disagreement was a category error: the validity bar (0.9780) and the draw-risk bar (0.9808+) measure different things and were never meant to be one number.

That makes the recommendation unambiguous under #124 publish-first. The **operative gate is 0.9780** — the bar a measured λ̂ MUST clear to launch, on the validity axis, accepting P_invalid = 5% (the standing P95 gate, now corroborated by #243's worst-case-vertex floor rather than contradicted). The **defended target is 0.9808** — the bar it SHOULD clear to hold the f_priv-integrated draw-below-500 risk at 5%, using the divergence-informed prior because the 0.73% is measured, not assumed. At the operative gate the accepted draw-below-500 residual is 0.0589 (div-informed) / 0.0793 (uniform); publish-first explicitly accepts this as post-hoc defence, which is exactly why the draw-risk bar is a SHOULD, not a MUST.

The one judgment call worth flagging: **whether the operative gate should be raised to #243's 0.978413** (the strict max of the two validity-axis numbers). I argue no — 0.9780 already carries the 95% confidence margin, whereas 0.978413 is a point-estimate mean=500 floor with no margin, and the +4.0e-4 gap is below the resolution at which the two derivations can be distinguished. If the advisor prefers the strict-max posture, swapping `build_lambda_operative_gate` to `max(p95, lam_floor_wc)` is a one-line change and moves the gate only +0.0004 — immaterial to any build target.

### Hand-off (one sentence)

*The measured launch λ̂ must clear `build_lambda_operative_gate = 0.9780` (P95 validity; residual P_invalid = 0.05), which #243's worst-case-vertex floor 0.978413 confirms (+4.0e-4); the `build_lambda_defended_target = 0.9808` (divergence-informed 5% draw-risk) is advisory headroom because under #124 publish-first the draw risk (0.0589 at the gate) is ACCEPTED post-hoc.* (Consumer: fern #238 card row (iii) build bar.)

### Suggested follow-ups

- **Fold into fern #238's card:** row (iii) "λ̂ build bar ≥ 0.9780" can now carry the explicit two-number form (operative 0.9780 MUST / defended 0.9808 SHOULD) with the per-axis residuals, so the GO/NO-GO reads off one card instead of a three-constraint argument.
- **When a measured build lands** (land #245 tree / lawine K-1 linear / stark T-1 topology), adjudicate its measured λ̂ directly: ≥ 0.9780 → GO (validity met, draw risk accepted); in [0.9780, 0.9808) → GO with the stated draw-below-500 residual; ≥ 0.9808 → GO with draw risk also defended ≤ 5%.
- **A second hard paired draw** would collapse the f_priv [0.957, 0.969] support to a measured point and could *narrow* the defended target (the draw-risk axis is the only prior-sensitive number); the operative validity gate would be unaffected.
