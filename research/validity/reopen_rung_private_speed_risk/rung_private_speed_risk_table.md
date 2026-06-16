# PR #522 — Reopen-rung private SPEED-drift risk (decision-tree feed)

`analysis_only=true`, `official_tps=0`. No HF Jobs / no `--launch` / no submission. W&B group `reopen-rung-private-speed-risk`.

**Boundary:** speed-side reopen-rung risk only. Quality-side = denken #513/#520 (downstream exposure **0.0**, pure speed). Public split-KV anchor = stark #519. Scored accuracy = ubel #511.

## Decision-tree feed table

| rung | role | public_tps | proj. private_tps (mean) | σ_hw 95% band | **private_tps_worstcase** | quality_verdict |
|---|---|---|---|---|---|---|
| `surgical357` | control (shipped primary) | 375.86 (official) | 359.69 | [352.6, 366.7] | **336.44** | 0 exposure (pure-speed) |
| `splitkv399` | upgrade candidate (byte-exact fixed-order split-KV) | 399.75 (provisional (local/#496)) | 382.55 | [375.1, 390.0] | **357.83** | 0 exposure (pure-speed) |
| `frontier457` | upgrade candidate (strict-frontier prediction) | 457.50 (prediction-only) | 437.82 | [429.2, 446.4] | **409.52** | 0 exposure (pure-speed) |
| `floor-lock` | literal-1.0 fallback (M=1 AR) | 166.23 | 166.23 (0 breach) | [163.0, 169.5] | 163.5 | literal-identical (guaranteed) |

**KEY OUTPUTS**
- `splitkv399_projected_private_tps` = **382.55**
- `splitkv399_private_tps_worstcase` = **357.83**
- `surgical357_private_tps_worstcase` (control) = **336.44**
- `frontier457_projected_private_tps` = **437.82** ; `frontier457_private_tps_worstcase` = **409.52** (prediction-only)
- `best_riskadj_rung` = **frontier457** ; worst-case floor **409.52**

## Why the table is a shared multiplier (the load-bearing fact)

surgical357 + splitkv399 share SPECULATIVE_CONFIG mtp K=7 and DRAFTER_SHA256 ed159e33...dd18e and are byte-exact greedy-identical; acceptance variance is identical across rungs by construction.

All three rungs share `SPECULATIVE_CONFIG mtp K=7` + `DRAFTER_SHA256 ed159e33…dd18e` and are byte-exact greedy-identical; they differ ONLY in the attention reduction path (a per-step `t_step` change that is acceptance-INDEPENDENT). Since `TPS = E[T]/t_step` and every step runs the same M=8 verify regardless of how many draft tokens are accepted, the acceptance-driven private drift is a **shared multiplicative factor**: central **0.9570**, combined-worstcase **0.8951**. Private ranking = public ranking at every percentile.

## Acceptance variance input (measured, reused — shared drafter)

Reused the **6 real served private draws** from PR #44 `private_gap_probe` on the shared-drafter parent `fa2sw_precache_kenyan` (sharegpt + 5 native domains). Re-serving each rung would reproduce the same acceptance (drafter property, not attention-path).

- single-draw acceptance ratio `R_ea = E[T]_priv/E[T]_pub`: mean **0.8877**, sd **0.0285**, range [0.8509, 0.9387]
- breach_acc = 1−R_ea: mean **0.1123**, sd **0.0285** (= σ_accdraw, the single-draw acceptance variance)
- served R_tps: mean 0.8700 sd 0.0094 (incl. precache-off ~1.24%)

These proxies are deliberately HARD (over-estimate true private breach ~2–3×, see `private_gap_probe.md`). We **headline the grounded central breach** and use the proxy **spread** (σ) as the single-draw acceptance variance.

## Framework (banked; kanna #504/#478/#508)

- propagation factor PF (acceptance→TPS) = **0.99992** ≈ 1.0 (#504 `0urxqwob`)
- grounded central breach = **4.295%** (#504/#508; board honest band 3.9–7.2%)
- σ_hw = **1.00% FRACTIONAL** one-shot (#478 `mssuss3f`).
  - ⚠ PR's `sigma_hw 4.864` is the ABSOLUTE between-leg @~482 TPS — provenance only. Applying 4.864 as a fixed constant at 375.857 would over-state the band by 1.29× ([[private_bar_convention]] trap).
- σ_accdraw used = **0.0285** (cross-domain breach spread)
- denken-24% refuted worst-case carried as extreme tail only

## Worst-case construction (single private draw, one-sided 95%)

```
mean       = P · (1 − breach_central) · PF
σ_hw(abs)  = mean · σ_hw_frac                      (fractional, per-rung)
band95     = mean ± 1.96·σ_hw                       (hardware-only, dossier-style)
breach_p95 = breach_central + 1.645·σ_accdraw       (acceptance-draw downside)
worstcase  = P · (1 − breach_p95) · (1 − 1.645·σ_hw_frac)   (acceptance ⊗ hardware downside)
```

## Verdict

Acceptance is a SHARED drafter property (identical DRAFTER_SHA256 + MTP K=7 + byte-exact output) -> all rungs inherit the SAME private-drift multiplier (central 0.9570, combined-worstcase 0.8951). Private TPS ranking therefore equals public ranking at EVERY percentile: frontier457 > splitkv399 > surgical357. Best risk-adjusted reopen rung = frontier457 (worst-case private floor 409.5 TPS). splitkv399 worst-case floor 357.8 ~= surgical357 EXPECTED private 359.7: the upgrade's downside lands at the control's median, and splitkv399 beats surgical357 by +21.4 at the worst case. UPGRADE rule: private-speed-drift NEVER inverts the ranking, so upgrade to the fastest rung whose PUBLIC anchor is validated (stark #519 for splitkv399; frontier457 is prediction-only). Speed-side risk does not gate the upgrade; quality is denken-cleared (0 exposure). Floor-lock 166.23 remains the only literal-identity fallback.

### Upgrade rule for #517's tree

**Upgrade — private-speed-drift never inverts the rung ranking.** Because the multiplier is shared, `splitkv399` worst-case floor (**357.83**) ≈ `surgical357` *expected* private (**359.69**) and exceeds `surgical357` worst-case (**336.44**) by **+21.4**. The upgrade has ~zero speed downside vs the control. The binding gates are the PUBLIC anchor (stark #519 validates split-KV; frontier457 is prediction-only) and the identity rule (denken-cleared for quality at 0 exposure; floor-lock 166.23 is the only literal-identity insurance).

> `analysis_only` composition of measured PR#44 acceptance draws + banked #504/#478/#508 framework. No server stood up (acceptance is a shared-drafter property). Self-test NaN-clean, ranking + dossier reproduction green.