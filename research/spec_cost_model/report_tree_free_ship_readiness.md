# PR #109 — Tree-free-500 ship-readiness: the official-submission go/no-go

**Verdict: 🟡 AMBER — HOLD the official shot.** The tree-free build-complete stack
clears 500 at denken #105's *central* (SplitK 4.44%), but **not at the conservative
corner the ship decision requires**: the corner needs **SplitK ≥ 14.3%** (margin 0),
**16.7%** (+1%), **19.0%** (+2%) — all well above the SplitK ubel #84 can plausibly
deliver (≤ ~8.5%). At the plausible SplitK, clearing 500 needs **τ ≥ 0.986** (vs the
[0.96,1.00] band floor), so we are betting on near-perfect realization. Because SplitK
is a *new* verify-GEMM kernel mix the deployed multiplier was never measured on,
**`tau_official_reanchor_required = yes`**: spend exactly ONE approval-gated official
anchor of the SplitK-built submission to convert τ from assumed→measured **before**
trusting 500.

- **Primary metric** `min_splitk_for_confident_ship_pct` = **14.34%** (margin 0).
- **Test metric** `tau_official_reanchor_required` = **yes — one official anchor**.
- W&B `pyjib2k8` (group `tree-free-ship-readiness`). Repro:
  `python scripts/profiler/tree_free_ship_readiness.py`.
- LOCAL, CPU-only, analytic. No HF Job, no submission, greedy identity untouched.

## What changed vs #105 (two honest corrections)

1. **Byte lever = wirbel PALETTE, not INT8 double-quant.** wirbel #104 returned a
   **KILL**: the deployed FP16 g128 verify-GEMM scales do not double-quantize to INT8
   bit-exactly (13.1% round-trip; best lossless hybrid is net-negative −1.27% bytes;
   achievable ≈ −0.02% TPS) — an information-theoretic barrier, not tuning. The byte
   lever that *exists* is the lossless scale palette/LUT (~0.3% TPS, bit-exact) but it
   is **not built** (a #104 follow-up). So byte-lever central ~0.3%, **conservative
   corner 0** (vs #105's INT8-dq band {0.4,0.5,1.1}%). This alone moves the central
   SplitK-for-500 from **4.44% → 4.84%**.
2. **Multiplier (lawine #99) factored explicitly with its CI.** Central 1.06019;
   measured **local-side CI ±0.018%** (config-stable), official-CV **envelope ±1.96%**.
   We keep #105's K_cal central (1.05985 vs 1.06019 = +0.032% sub-MDE self-check
   residual; continuity with merged #105 = 518.1) and use #99 only for the CI band.

**No double-count (load-bearing):** the multiplier's official-CV envelope (±1.96%) and
τ∈[0.96,1.00] are the *same* official-side risk ("will measured official match the
projection?"). We carry it **once**, in τ (the PR-named gate), and use only the
multiplier's *local-side* CI in the corner. τ=0.96 (−4%) is strictly more conservative
than the envelope-low (−1.96%), so the primary corner already dominates it; the
envelope-low is reported only as a worst-of-worst stress (corner → 19.0%).

## Step 1 — minimum SplitK for a CONFIDENT ship (the corner, not the central)

Corner = τ=0.96 · multiplier local-CI-low · LK floor 1.010 · byte-lever(palette)=0 ·
fp32 worst · persist 0.

| margin | target | **corner SplitK** | ≤ ubel 8.5%? | central SplitK | envelope stress |
|---|---|---|---|---|---|
| 0% | 500 | **14.34%** | no | 4.84% | 19.00% |
| +1% | 505 | **16.67%** | no | 6.89% | 21.47% |
| +2% | 510 | **19.05%** | no | 8.97% | 24.00% |

The corner SplitK is **~3× #105's central 4.44%** and sits above ubel's plausible
delivery at every margin. This 14.34% is the bar ubel #84 must beat for a
projection-only ship.

## Step 2 — does pinning τ need an approval-gated official re-anchor?

**Decision: YES — one official anchor.** Rule (applied at ubel-central SplitK 8.5%):

- τ_required ≤ 0.96 (floor) → **NO** re-anchor (floor already clears; the ~0.3%
  kernel-mix shift is immaterial). *Not our case.*
- 0.96 < τ_required ≤ 1.00 → **YES**, one official anchor (we rely on τ above its
  floor; SplitK changes the kernel mix the multiplier was measured on, so the mix-shift
  is no longer immaterial against the thin margin). **τ_required = 0.986 → this is our
  case.**
- τ_required > 1.00 → moot: a SplitK/lever-magnitude problem, not a τ problem.

**Why SplitK can move τ:** the lawine #99 multiplier is a box transfer factor measured
on the *deployed* kernel mix. SplitK shrinks the verify-GEMM slice and re-weights the
step toward the GPU-busy small-kernel tail (denken #97). If per-kernel-class transfer
differs between the local A10G and a10g-small, the aggregate multiplier shifts by an
estimated ~0.1–0.3% — immaterial against a 4% τ-floor cushion, but **not** against the
~1.4% cushion we actually have at the plausible SplitK. One official run of the
SplitK-built submission measures τ directly and retires the assumption.

## Step 3 — the GO/HOLD submit-decision table (hand to the human team)

Cells = projected official TPS at the **corner-conservative** lever bundle; **GO** iff
≥ 500.

| SplitK % | τ=0.96 | τ=0.98 | τ=1.00 |
|---|---|---|---|
| 0.00 | 466.8 HOLD | 476.5 HOLD | 486.2 HOLD |
| 4.44 | 477.5 HOLD | 487.5 HOLD | 497.4 HOLD |
| 5.00 | 478.8 HOLD | 488.8 HOLD | 498.8 HOLD |
| 6.50 | 482.4 HOLD | 492.4 HOLD | 502.5 **GO** |
| **8.50** (ubel central) | 487.0 HOLD | 497.1 HOLD | 507.3 **GO** |
| 10.00 | 490.4 HOLD | 500.6 **GO** | 510.8 **GO** |
| 12.00 (ubel high) | 494.9 HOLD | 505.2 **GO** | 515.5 **GO** |
| 14.00 | 499.3 HOLD | 509.7 **GO** | 520.1 **GO** |
| 17.00 | 505.7 **GO** | 516.2 **GO** | 526.8 **GO** |
| 20.00 | 512.0 **GO** | 522.6 **GO** | 533.3 **GO** |
| 29.70 (gap ceiling) | 531.2 **GO** | 542.3 **GO** | 553.4 **GO** |

**Most-likely cell (ubel 8.5% × τ band): 487.0 → 507.3** — it **straddles 500**. We
clear 500 only if τ ≥ ~0.986; at the τ floor we land at 487. This straddle is precisely
why the verdict is AMBER and why the single official τ-anchor is the gating measurement.

## Public evidence

Public #1 is now **frantic-penguin `skv64` 489.63** (digest `?as=senpai`, 2026-06-14).
A cluster of SplitK/argmax-block-class submissions sits at ~484–490 (byteshark
`splitkv-k7-argmaxblock64` 484.62, need-for-speed `mao-gemma-fast-skv64` 488.07).
**Realized SplitK-class gains in the competitive field are ~+0.6–1.7% over 481.53 —
below the 4.44% central, and none clear 500.** This independently corroborates the
conservative corner: a tree-free 500 needs a SplitK lift the field has not yet realized.

## Bottom line for the official-submission approval request

- **Do not** spend the official shot on a projection-only basis: at the SplitK ubel can
  plausibly deliver, the corner straddles 500.
- The official shot is **best spent as the τ-anchor itself**: launch the SplitK-built
  submission once, read its official TPS + τ. If it lands ≥ ~507 (τ≈1.0 confirmed at
  8.5% SplitK), we have 500 with the corner pinned; if it lands ~487–500, we learn τ and
  re-decide with measured (not assumed) τ.
- If ubel #84 cannot push SplitK toward ~14% **and** palette/LK do not realize, the
  tree (land #71) re-enters the critical path for a *comfortable* 500 (lawine #107 step
  ratio governs that corner) — but the tree-free path is not RED: it reaches 500 with
  τ≈1.0 at SplitK ≥ ~6.5%, just without conservative-corner margin.
