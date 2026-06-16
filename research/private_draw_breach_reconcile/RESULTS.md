# PR #504 — Private-TPS-breach reconciliation (4.3% linear vs 24% amplified)

**Run:** `0urxqwob` · `--wandb_group private-draw-breach-reconcile` · analysis_only, official_tps=0, CPU-only.

## Verdict
Deployed anchor (481.53→460.85 = **4.295% TPS gap**) ÷ denken #489 Δα=4.295% realize a
propagation factor of **1.00 (linear)**; the natural-elasticity envelope is [0.74, 2.24],
all ≪ the **5.6** a 24% breach requires. → surgical-357 private TPS ≈ **341.9** [95% 335.2–348.6,
±σ_hw 1%], breach **4.29%**. **denken #489's 24% is WORST-CASE-ONLY** (needs a deep-block-survival
acceptance model far outside the realized/natural range), data-refuted as the *expected* outcome.

## The structural crux
In MTP K=7 spec-dec every verify step **always** drafts K and **always** verifies K positions;
acceptance changes only the per-step *yield*, not the per-step *cost*. So `C_step` is α-invariant and

```
TPS = E[T] / C_step   →   dTPS/TPS = dE[T]/E[T]   →   PF = elasticity(E[T], α)
```

The 4.3%-vs-24% question reduces to **which acceptance elasticity the data realizes**. denken owns
the acceptance leg (Δα=4.295%, taken as INPUT); this PR owns the **TPS-propagation leg**.

## Legs
| leg | result |
|---|---|
| **1b analytic** (E[T]=3.8512, K=7) | (A) mean-accept ε=0.74→3.18% · (B) geometric ε=2.24→9.62% · (C) per-pos compound ε=2.21→9.47%. Natural envelope **[0.74, 2.24]**. |
| **deployed anchor** (PR #52, real private leaderboard) | 481.53/460.85 = 4.295% gap ÷ 4.295% Δα = **PF 1.000** |
| **within-run GPU xcheck** (#478 `jb1a0lab`, exact deployed stack, n=10) | steady-rate elasticity **1.92** (r=0.50, 95% CI [-0.66, **3.95**]); wall elasticity 0.11 (overhead-pinned). CI upper bound excludes 5.6. |
| **24% requirement** | PF=**5.59** ≈ deep-block survival G(K)=α^5.6 — outside the natural envelope → worst-case only |
| **leg 3 composition** | surgical-357 private 341.9, 68% [338.5,345.3], 95% [335.2,348.6], P(<0.95×pub)=**0.231** |

## Why fractional breach transfers from the 481 stack to the 357 ship
surgical-357 keeps the **same** MTP K=7 drafter (kenyan-duma `ft-v1-epoch_001`); it differs only in
the attention path (`SURGICAL_ATTN_USE_3D_OFF=1`), which raises `C_step` (→ lower absolute TPS 357 vs
481) but does **not** change ΔE[T]/E[T]. Since the fractional breach = ΔE[T]/E[T] is drafter-determined
and α-invariant in `C_step`, the **fractional** 4.3% breach is identical on both stacks.

## GPU bracket leg (primary) — blocked, substituted
Full production serve not stood up on pod-A10G: vLLM not importable; custom cu129 wheel vs pod CUDA 13.2;
40KB serve.py + 52KB sitecustomize + 8 custom-kernel patches + multi-bucket downloads, none cached.
Disproportionate to a cross-check that cannot beat the deployed private-leaderboard anchor. **Substituted**
with my own #478 banked (E[T],TPS) pairs from the exact deployed stack. The #497 splits carry identity-flip
rates only (not acceptance/TPS), so they give qualitative off-distribution context (private flip ratio
0.50×/0.37× < 1 ⇒ bounded, not collapsing) but no bracket-resolved factor.
