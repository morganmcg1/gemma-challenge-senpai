# Margin-gated top-k accept projection (PR #823) — twin of #820

- blocks: **19538**  vocab: 262144  (reuses #820 capture `0r80mau9`, NO re-serve, 0 GPU)
- oracle fit (same anchors as #820): `TPS = 43.60 + 61.37*L`  (R²=0.9989)
- **strict self-consistency**: E_accept=3.3800 (baseline 3.379, ok=True), r=0.3967 (baseline 0.397, ok=True), TPS@strict=251.0
- **margin = dp/ap at reject positions**: median **0.030**, mean 0.160, p90 0.535, max 1.000  |  at rank-2 rejects: median **0.287**, max **0.883**
- fraction of all rejects with margin ≥ 0.5: **0.121** (4901), 0.8: **0.034** (1381), 0.95: **0.010** (420)

## Step 1 — margin-conditioned reject-rank CDF (HIGH=near-tie vs LOW)

| margin thr | subset | ≤2 | ≤5 | ≤16 | n |
|---|---|---|---|---|---|
| 0.10 | high (m≥thr, near-tie) | 0.640 | 0.960 | 1.000 | 14868 |
| 0.10 | low (m<thr) | 0.141 | 0.421 | 0.721 | 25770 |
| 0.30 | high (m≥thr, near-tie) | 0.790 | 0.994 | 1.000 | 8175 |
| 0.30 | low (m<thr) | 0.206 | 0.523 | 0.779 | 32463 |
| 0.50 | high (m≥thr, near-tie) | 0.882 | 0.999 | 1.000 | 4901 |
| 0.50 | low (m<thr) | 0.247 | 0.566 | 0.799 | 35737 |
| 0.80 | high (m≥thr, near-tie) | 0.977 | 1.000 | 1.000 | 1381 |
| 0.80 | low (m<thr) | 0.301 | 0.605 | 0.817 | 39257 |
| 0.95 | high (m≥thr, near-tie) | 1.000 | 1.000 | 1.000 | 420 |
| 0.95 | low (m<thr) | 0.316 | 0.614 | 0.821 | 40218 |

### Side-by-side: entropy-conditioned CDF (#820, REFUTED — does NOT separate)

| H thr (nats) | subset | ≤2 | ≤5 | ≤16 | n |
|---|---|---|---|---|---|
| 0.10 | high (H>thr) | 0.323 | 0.624 | 0.831 | 36044 |
| 0.10 | low (H≤thr) | 0.324 | 0.573 | 0.762 | 4594 |
| 0.25 | high (H>thr) | 0.320 | 0.621 | 0.830 | 33806 |
| 0.25 | low (H≤thr) | 0.341 | 0.603 | 0.788 | 6832 |
| 0.50 | high (H>thr) | 0.310 | 0.614 | 0.827 | 30270 |
| 0.50 | low (H≤thr) | 0.364 | 0.630 | 0.811 | 10368 |
| 1.00 | high (H>thr) | 0.254 | 0.572 | 0.810 | 20602 |
| 1.00 | low (H≤thr) | 0.395 | 0.666 | 0.837 | 20036 |
| 2.00 | high (H>thr) | 0.151 | 0.410 | 0.722 | 5081 |
| 2.00 | low (H≤thr) | 0.348 | 0.648 | 0.838 | 35557 |
| 3.74 | high (H>thr) | 0.000 | 0.067 | 0.533 | 15 |
| 3.74 | low (H≤thr) | 0.324 | 0.618 | 0.823 | 40623 |

### P(rank≤k | margin decile) — where do the near-misses live?

| margin bucket | count | P(rank=2) | P(≤2) | P(≤5) | P(≤16) |
|---|---|---|---|---|---|
| [0.0,0.1) | 25770 | 0.141 | 0.141 | 0.421 | 0.721 |
| [0.1,0.2) | 4413 | 0.422 | 0.422 | 0.896 | 0.999 |
| [0.2,0.3) | 2280 | 0.524 | 0.524 | 0.962 | 1.000 |
| [0.3,0.4) | 1672 | 0.625 | 0.625 | 0.981 | 1.000 |
| [0.4,0.5) | 1602 | 0.680 | 0.680 | 0.989 | 1.000 |
| [0.5,0.6) | 883 | 0.771 | 0.771 | 0.998 | 1.000 |
| [0.6,0.7) | 1756 | 0.847 | 0.847 | 0.999 | 1.000 |
| [0.7,0.8) | 881 | 0.915 | 0.915 | 1.000 | 1.000 |
| [0.8,0.9) | 961 | 0.967 | 0.967 | 1.000 | 1.000 |
| [0.9,1.0) | 420 | 0.000 | 1.000 | 1.000 | 1.000 |

## Step 2 — uniform top-k (= margin 0; identical to #820)

| k | E_accept | TPS | Δ% vs ~253 | non-argmax emitted |
|---|---|---|---|---|
| 2 | 4.234 | 303.4 | +19.9 | 7793 |
| 5 | 5.219 | 363.9 | +43.8 | 18360 |
| 16 | 6.065 | 415.8 | +64.4 | 28419 |

## Step 2 — (k, margin) grid

| k | margin | TPS_uniform | TPS_gated | Δ% vs ~253 | retain% (same k) | retain% (best uni) | risky low-m relaxes dropped | safe high-m kept |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.00 | 303.4 | 303.4 | +19.9 | 100 | 31 | 0 | 7793 |
| 2 | 0.50 | 303.4 | 265.9 | +5.1 | 26 | 8 | 5322 | 2471 |
| 2 | 0.80 | 303.4 | 255.4 | +0.9 | 5 | 1 | 7024 | 769 |
| 2 | 0.95 | 303.4 | 252.4 | -0.2 | -1 | -0 | 7547 | 246 |
| 5 | 0.00 | 363.9 | 363.9 | +43.8 | 100 | 68 | 0 | 18360 |
| 5 | 0.50 | 363.9 | 267.4 | +5.7 | 13 | 9 | 14834 | 3526 |
| 5 | 0.80 | 363.9 | 255.5 | +1.0 | 2 | 2 | 17355 | 1005 |
| 5 | 0.95 | 363.9 | 252.4 | -0.2 | -1 | -0 | 18032 | 328 |
| 16 | 0.00 | 415.8 | 415.8 | +64.4 | 100 | 100 | 0 | 28419 |
| 16 | 0.50 | 415.8 | 267.4 | +5.7 | 9 | 9 | 24217 | 4202 |
| 16 | 0.80 | 415.8 | 255.5 | +1.0 | 2 | 2 | 27245 | 1174 |
| 16 | 0.95 | 415.8 | 252.4 | -0.2 | -0 | -0 | 28045 | 374 |

## Step 2 — quality proxy: uniform-k relaxes by margin × entropy (H thr = 1.00 nats, #820's headline boundary)

| k | margin thr | total relaxes | low-margin (risky, gate drops) | frac low-margin | frac low-entropy (#820) | low-m∧low-H | low-m∧high-H |
|---|---|---|---|---|---|---|---|
| 2 | 0.50 | 7793 | 5322 | 0.683 | 0.655 | 4184 | 1138 |
| 2 | 0.80 | 7793 | 7024 | 0.901 | 0.655 | 4828 | 2196 |
| 2 | 0.95 | 7793 | 7547 | 0.968 | 0.655 | 5014 | 2533 |
| 5 | 0.50 | 18360 | 14834 | 0.808 | 0.568 | 9303 | 5531 |
| 5 | 0.80 | 18360 | 17355 | 0.945 | 0.568 | 10089 | 7266 |
| 5 | 0.95 | 18360 | 18032 | 0.982 | 0.568 | 10311 | 7721 |
| 16 | 0.50 | 28419 | 24217 | 0.852 | 0.522 | 13582 | 10635 |
| 16 | 0.80 | 28419 | 27245 | 0.959 | 0.522 | 14454 | 12791 |
| 16 | 0.95 | 28419 | 28045 | 0.987 | 0.522 | 14704 | 13341 |

## Step 3 — does margin DOMINATE uniform-k?

**NO — clean negative.** No `(k, margin>0)` keeps ≥ 90% of uniform-k's gain. The best any margin>0 achieves is **26%** retention (k=2, margin=0.50, TPS 265.9). Uniform-k's speed lives in the LOW-margin (genuine-disagreement) region, so a quality-safe margin gate cannot keep the speed.

**=> Ship plain uniform-k.** The relaxation lever's quality cost is intrinsic (coupled to the speed) and must be MEASURED on the real quality floors, not gated away by margin.

## Verdict

- relaxation-lever ceiling (best uniform, unchanged from #820): **415.8 TPS** (+64.4% vs ~253), E_accept=6.065
- **margin dominates uniform-k: False**
- **SHIP to stark #816: uniform_topk** k=16 margin=0.00 → 415.8 TPS (+64.4%)
- quality floors to protect (NOT re-evaluated here): {'ppl_max': 2.42, 'aime_min': 0.09, 'modality': '128/128', 'mmlu_pro_min': 0.572, 'gpqa_min': 0.471, 'gsm8k_min': 0.807}
