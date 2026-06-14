STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3ml0shkm"],"primary_metric":{"name":"fpriv_worstcase_measured_div_self_test_passes","value":1},"test_metric":{"name":"fpriv_worstcase_under_measured_div","value":0.9689500205949673}}

## Results

**Headline: under lawine #232's measured near-greedy 0.73% divergence the NLS worst-case-vertex f_priv tightens from the grounded 0.957054 floor UP to `fpriv_worstcase_under_measured_div = 0.968950`, which moves ABOVE the #233 publish-first breakeven 0.959780 — the realizable band [worst-case, clean] NO LONGER STRADDLES it. The bar's location is SAFER once the corrected int4 physics replaces kanna #114's M=1 56%.**

This is a CPU-only analytic leg (bank-the-analysis; PRIMARY = self-test). **0 TPS, BASELINE stays 481.53, authorizes nothing. NOT a launch. NOT open2.**

### The frame (the blend)

```
f_priv_wc(d) = (1 - d)·f_clean + d·f_int4div
```

- `f_clean = 0.969107` — #226 NLS (`native_multilingual`) clean realizable worst-case (the `d→0` limit).
- `f_int4div = 0.947614` — the fully-int4-divergent decode-drop, **pinned by the `d=0.5608` round-trip** so the blend reproduces #233's grounded floor 0.957054 (the `d→1` weight on that floor).
- Calibration round-trip: `[f_priv_wc(0.5608), f_priv_wc(0)] = [0.957054, 0.969107]` == #233 worst-case band (resid ≤ 1e-15, both ends).

### Re-price under the measured 0.73% (the core)

| quantity | value |
|---|---|
| `fpriv_worstcase_under_measured_div` = f_priv_wc(0.0073) | **0.968950** |
| Δ vs breakeven 0.959780 | **+0.009170 (ABOVE)** |
| Δ vs old worst-case 0.957054 | +0.011896 (less adverse) |
| corrected λ_floor (via dλ_floor/df_priv = −2.3535) | **0.978413** (central 0.978044) |
| λ_floor exact #233 reach-DP cross-check | 0.978413 (resid **1.25e-09**) |
| straddles breakeven? | **False — moved ABOVE (lane SAFER)** |
| un-straddle threshold d\* | **0.4339** (measured 0.73% clears with ~60× margin) |

### The table (deliverable 3): d × (f_priv_wc, implied λ_floor, straddles)

| d | f_priv_wc | λ_floor | straddles breakeven 0.9598 |
|---|---|---|---|
| **0.0073 (measured, lawine #232)** | **0.968950** | **0.978413** | **False** |
| 0.10 | 0.966958 | 0.983103 | False |
| 0.30 | 0.962659 | 0.993219 | False |
| 0.5608 (old, kanna #114) | 0.957054 | ∅ UNREACHABLE | True |

Headline: **near-greedy divergence un-straddles the breakeven.** Only the OLD d=0.5608 straddles; the worst-case crosses the breakeven at d\*=0.4339, so every divergence below ~0.43 (including the measured 0.0073) sits above it.

### NLS vertex (deliverable 3): confirmed unchanged

The binding (f_priv-minimizing) vertex stays **`native_multilingual` (NLS)** under the corrected divergence. Under a shared int4-divergent floor the blend is monotone increasing in each vertex's clean f_priv, so argmin = argmin(f_clean^v) = NLS for any d∈[0,1). Runner-up `native_code` (margin 0.000161); flipping the vertex at d=0.0073 would require code's int4-divergent f_priv to sit **0.02208 BELOW** NLS's — implausible for the hardest axis. NLS stays binding with margin.

### Self-test (PRIMARY) — `fpriv_worstcase_measured_div_self_test_passes = True`

- (a) d=0.5608 round-trips #233's band, both ends, resid ≤ 1e-15 ✓
- (b) f_priv_wc strictly ↑ as d↓ (spread f_clean−f_int4div = +0.021493 > 0) ✓
- (c) corrected worst-case 0.968950 > old worst-case 0.957054 ✓
- (d) breakeven-straddle verdict stated (moved ABOVE, complementary booleans) ✓
- (e) NLS vertex confirmed (`native_multilingual`) ✓
- (f) NaN-clean ✓
- xcheck: linear λ_floor matches exact #233 solver (resid 1.25e-09) **and** corrected worst-case insensitive to the OLD-div pin ✓

**Robustness:** varying the OLD-divergence pin d_old over [0.50, 0.61] (±0.05 around 0.5608) moves `fpriv_worstcase_under_measured_div` only within [0.968935, 0.968963] (~3e-5 span) — all comfortably above the breakeven. At the measured 0.73% the int4-divergent weight is so small the corrected worst-case is dominated by f_clean, so it barely depends on f_int4div (hence on the exact old divergence).

### Comparison vs the PR baseline anchors

| anchor | PR body | this leg |
|---|---|---|
| #233 breakeven (publish-first) | 0.9598 | imported unchanged 0.959780 |
| #233 realizable worst-case band | [0.957 grounded, 0.969 assumed] | reproduced as [f_priv_wc(0.5608), f_priv_wc(0)] |
| dλ_floor/df_priv | −2.35 | imported unchanged −2.3535 |
| worst-case f_priv under OLD d=0.5608 | 0.957054 (straddles) | 0.957054 (straddles) — round-trip |
| **worst-case f_priv under MEASURED d=0.0073** | (the deliverable) | **0.968950 (does NOT straddle)** |

### Reproduce

```
cd target/ && CUDA_VISIBLE_DEVICES="" python \
  research/validity/fpriv_worstcase_measured_div/fpriv_worstcase_measured_div.py \
  --self-test --wandb_group issue192-reading-calibration \
  --wandb_name stark/fpriv-worstcase-measured-div
```

- **W&B run:** `3ml0shkm` (wandb-applied-ai-team/gemma-challenge-senpai, group `issue192-reading-calibration`)
- **Peak memory:** 27.184 MiB (CPU-only)
- **summary.json fields:** N/A — no benchmark/draw this leg (0 TPS; `tps`/`ppl`/`completed`/`run_prefix` unchanged from the served baseline 481.53 TPS, PPL 2.3772, 128/128, PR #52). This is a pure analytic re-pricing.

### What happened — honest analysis

It worked, and the direction is exactly what the hypothesis predicted. Framing the #233 worst-case band as a divergence-weighted blend and swapping kanna #114's M=1 56.08% weight for lawine #232's measured M=8 0.73% near-greedy weight tightens the worst-case f_priv from 0.957054 up to **0.968950** — within 0.000157 of the clean NLS value 0.969107. Because the int4-divergent component now carries ~0.73% weight instead of ~56%, the adverse `f_int4div = 0.947614` endpoint is almost entirely de-weighted.

The decision-relevant flip: the corrected worst-case is **+0.009170 ABOVE** the publish-first breakeven 0.959780, so the realizable band no longer straddles it. The un-straddle threshold is d\*=0.4339 — the measured 0.73% clears it with a very wide margin (the divergence would have to be ~59× larger to re-straddle). Equivalently the corrected publish-first λ_floor is 0.978413 (essentially the central 0.978044, vs ∅/unreachable at the old grounded floor): the publish-first POINT-estimate gate is reachable at the worst realizable vertex once the corrected int4 physics is in.

Two cross-checks give me confidence: (1) the linear-sensitivity λ_floor matches the exact #233 reach-DP solver to 1.25e-09, and (2) the result is insensitive to the exact old-divergence pin (the whole point — at near-greedy divergence the blend is f_clean-dominated). The binding vertex stays NLS with margin, so the worst-case location is governed by the same axis as before — only its value tightens.

**One caveat worth flagging:** the blend attributes the entire 0.957→0.969 gap to int4 divergence weighting. The grounded 0.957054 is a single hard #52 paired draw; #226 placed it OUTSIDE the realizable simplex. So this leg says "IF the gap is int4-divergence-driven, the corrected 0.73% un-straddles the breakeven." If part of the 0.957 floor is a genuine clean-decode tail unrelated to int4 loss (e.g. hardware/draw variance), that residual is not removed by re-weighting d — it would live in kanna's draw-risk distribution, not in this worst-case-location leg. The clean separation: lawine owns the divergence rate, I own the worst-case f_priv **location** under that rate, kanna owns the draw-risk distribution around the point.

### Hand-off (one sentence)

*Under lawine #232's measured near-greedy 0.73% divergence the worst-case-vertex f_priv tightens to `fpriv_worstcase_under_measured_div = 0.968950` (vs the old 0.957 floor), which **no longer straddles** the #233 breakeven 0.9598 — so the publish-first private bar's location is **safer** once the corrected int4 physics replaces kanna #114's M=1 56%.* (Consumers: kanna's f_priv-band, fern's card, #124.)

### Suggested follow-ups

- **kanna's draw-risk distribution** should now be re-centered: with the worst-case **location** at 0.968950 (not 0.957054), the residual private-draw risk is the spread of the draw distribution *around* that tighter point, not the [0.957, 0.969] interval width. The accepted-risk curve narrows.
- **A second hard paired draw** would test the attribution: if a fresh #52-style draw lands near 0.969 (not 0.957), it confirms the 0.957 was int4-divergence-driven (now de-weighted); if it lands near 0.957 again under the verified near-greedy stack, the gap is NOT int4-divergence and this re-weighting is too optimistic.
- **fern's GO-card** can carry publish-first as GO-at-worst-realizable-vertex (504.7 ≥ 500 at λ=1) once the corrected divergence is the operative int4 weight, with the d\*=0.4339 un-straddle margin as the headroom number.
