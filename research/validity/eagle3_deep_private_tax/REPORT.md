# PR #318 — Fusion deep-private-tax: does the ρ_priv_e3 worst-case lower bound still clear 500?

**Verdict: 🟡 YELLOW (thin-margin).** The deep-private-tax YELLOW does **not** close to GREEN.
Under the central deep-fidelity model the build clears PRIVATE-500 comfortably (586 TPS, +17.2%),
but the **worst credible published OOD bound lands ~1.4% BELOW the 500 bar**. The verdict flips on
whether the held-out private set is as out-of-distribution to the fusion head as the worst
cross-DOMAIN shift in the EAGLE-3 paper. **Bank-the-analysis, 0 TPS, BASELINE 481.53 unchanged.**

W&B `xe8ff7hq` · primary `rho_priv_e3_min=0.7923` · test `worstcase_private_tps=492.87`.

## The question (the #1 residual YELLOW on the GREEN-pending-build verdict)

fern #310 (`2u3kcnv5`) cleared PRIVATE-500 at 586.08 TPS @ E[T]=6.11 on `ρ_priv_e3 = 0.9421`. But
that ρ was **MODELED from the deployed LINEAR spine** (lawine #300, `8t5q6sr0`: c₁=1.0 held by the
M=8 tree, c_deep=0.97135 on j≥2, calibrated to the organizer-verified 460.85). It was **never
measured on the actual {2,21,39}-fusion EAGLE-3 head**. The risk: the fusion head's deeper
(high-layer) features can overfit the public distribution, so its true public→private **deep tax**
may exceed the linear spine's, dragging ρ_priv_e3 toward — or below — the **0.8038 break-even**.

## The model (decomposition the PR asked for)

ρ_priv_e3 = E[T]_priv / E[T]_pub is, by definition, an acceptance-**length** ratio (τ_priv/τ_pub).
Per-position profile of the EAGLE-3 deep-flat build (public E[T]=4.966):

- **public:** a₁=0.72925 (held), a₂..₇ = deep_pub = 0.91443 → E[T]_pub = 4.966
- **private:** a₁ held (c₁=1.0 — the M=8 tree recovers the spine cliff, organizer-verified on the
  linear stack at Δ4.3%), a₂..₇ = deep_pub · **c_deep_e3**, with

      c_deep_e3 = c_deep_lin · f_deep ,   c_deep_lin = 0.97135   (lawine #300)

`f_deep` ∈ (0,1] is the **incremental fusion-deep retention** (1 = fusion inherits the linear
spine's deep fidelity exactly; <1 = the fusion head degrades MORE OOD). The #310 headline mapping is
`private_tps(6.11) = honest_public(6.11) · ρ = 622.08 · ρ`, break-even ρ = 500/622.08 = **0.8038**.

## The worst-case ladder (ρ → private TPS @ E[T]=6.11)

| scenario | ρ_priv_e3 | private TPS | verdict | source |
|---|---|---|---|---|
| measured within-task (deployed linear, Δ4.3%) | 0.9571 | 595.4 | **CLEAR** | organizer-verified 460.85/481.53 |
| central: linear deep fidelity inherited (f_deep=1) | 0.9421 | 586.1 | **CLEAR** +17.2% | lawine #300 / fern #310 |
| EAGLE-3 Vicuna-13B worst cross-dataset | 0.8183 | 509.1 | CLEAR | arXiv:2503.01840 T1 (Alpaca/HumanEval) |
| **break-even** | **0.8038** | **500.0** | — | fern #310 |
| **PRIMARY worst-case: EAGLE-3 worst cross-dataset** | **0.7923** | **492.9** | **MISS −1.4%** | arXiv:2503.01840 T1 (LLaMA-3.1-8B CNN/DM÷HumanEval) |
| #310 raw / no-tree-recovery (independent line) | 0.7797 | 485.1 | MISS −3.0% | fern #310 banked |
| EAGLE-3 padded recommendation δ=0.78 | 0.7800 | 485.2 | MISS −3.0% | researcher pad on 0.792 |
| _off-axis floor:_ #263 branch −34.5% on deep cond. | 0.5446 | 338.8 | MISS | ubel #263 (too harsh — see caveat) |

**Two independent on-axis lines converge just under 500:** EAGLE-3's worst cross-dataset τ-ratio
(0.7923→493) and #310's raw/no-tree-recovery worst case (0.7797→485). The worst credible cases
cluster at ρ ≈ 0.78–0.82, private TPS ≈ 485–509 — **straddling the 500 bar**.

## Sensitivity / break-even (the number the human needs)

- **Break-even f_deep = 0.9163** (c_deep_e3 from 0.97135 → **0.8901**; deep conditional from 0.8882
  → 0.8138). With a₁ held by the tree, the fusion head's deep-position conditional can suffer an
  **incremental 8.4% deep tax** beyond the linear spine before private-500 breaks.
- **Worst-case ρ headroom to break-even = 0.7923 − 0.8038 = −0.0115** (slightly underwater).
- **Central ρ headroom = 0.9421 − 0.8038 = +0.1383** (+17.2%).

**Decision framing:** private-500 survives any public→private acceptance-length degradation up to
**19.6%** (= 1 − 0.8038). The **measured within-task** degradation on this exact stack is **4.3%**
(clears with ~4.5× headroom). EAGLE-3's **worst cross-DOMAIN** degradation is **20.8%** (just over
the line). So the verdict flips to NO-GO only if the held-out private set is as OOD to the fusion
head as summarization is to a code-trained drafter.

## Why 0.7923 is a defensible (conservative) lower bound, not a point estimate

1. **Cross-DOMAIN ≫ within-task.** 0.7923 is the τ ratio between the *most different task domains*
   in the EAGLE-3 eval (CNN/DM summarization vs HumanEval code). The challenge's public→private is a
   *held-out set of the same task mix* — the measured linear-stack shift was 4.3%, ~5× milder.
2. **Tree a₁-recovery not credited.** The 0.7923 is a *raw aggregate* τ-ratio that includes the
   shallow-position (a₁) degradation. Our deployed M=8 tree *recovers* the a₁ cliff (c₁=1.0,
   organizer-verified on the linear stack). The EAGLE-3 paper has no per-depth α table, so the
   recovery cannot be cleanly subtracted — meaning 0.7923 *under*-credits the fusion head.
3. EAGLE-3's cross-dataset spread (18–21%) is *narrower* than EAGLE-2's (22.2%); the {2,21,39}
   fusion *raises* baseline τ (6.65 vs EAGLE-1 3.98 on MT-Bench). No measured layer-fusion
   overfitting evidence exists in the paper — it is an open empirical question, hence YELLOW.

The #263 branch-collapse floor (0.5446) is reported but **off-axis**: it is the rank-coverage-MASS
axis (lawine #316's), measured on a pessimistic ShareGPT chat proxy that over-states the real
private set (that proxy class predicted 12.4% for the linear stack vs the real 4.3%). The #310
raw/no-tree-recovery 0.7797 scenario already prices the "tree fails to recover" case.

## Scope / caveats

- LOCAL CPU-only analytic over banked constants + published EAGLE-3 numbers. **0 TPS; BASELINE
  481.53 untouched; greedy/PPL untouched. NO GPU / vLLM / HF Job / submission / served-file change.
  Authorizes NOTHING. NOT a launch.**
- This is a worst-case **bound**, not a measurement. The actual {2,21,39}-fusion private tax needs a
  trained head (checkpoint-gated). This leg de-risks the build GO/NO-GO; it does not change baseline.
- Out of scope (do not re-derive here): the ×0.804 reconciliation (settled #310), rank-coverage mass
  (lawine #316), a₁-cliff trainability (denken #308), E[T]=6.11 reachability, greedy/PPL (Issue #192).

## Reproduce

```
cd target/ && .venv/bin/python \
  research/validity/eagle3_deep_private_tax/eagle3_deep_private_tax.py --self-test \
  --wandb_group eagle3-deep-private-tax --wandb_name fern/eagle3-deep-private-tax
```

Imports (exact, self-test 0): lawine #300 `private_bar_eagle3_results.json` (a₁, deep, c_deep,
ρ_priv_e3, public_et) · fern #310 `eagle3_private_perposition_reconcile_results.json`
(honest_public_611, breakeven_rho, private_tps_611, deployed_priv_over_pub). Literature:
EAGLE-3 arXiv:2503.01840 Table 1. Banked OOD: ubel #263 `he7glotf`, ubel #258/#250.
