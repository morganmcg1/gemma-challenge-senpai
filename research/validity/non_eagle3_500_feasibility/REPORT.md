# Non-EAGLE-3 >500 feasibility — does any method escape the supply tax? (PR #345, stark)

**CPU-analytic feasibility screen. 0 GPU, no model forward, no training, no served-file change, no
HF Job, no submission, no launch, 0 official-TPS. BASELINE stays 481.53 (adds 0 TPS).**

## Governing question (the #319 option-B fallback)

The strict-EAGLE-3 >500 lane is closed end-to-end: denken #332 (`y5cl0ena`) priced the **supply**
side RED (the verify-attention BW floor caps the strict-compliant ceiling at **473.53 < 500**), and
stark #340 (`jwv1vbug`) priced the **demand** side insufficient (at the honest fusion coverage 0.8903
the compliant-500 envelope collapses; even clearing the 0.9213 identity bar buys only central-500).
The #319 decision needs its option-B fallback answered numerically:

> Is there a **non-EAGLE-3** speculative method that reaches strict-compliant >500 — or do all
> alternatives share the same fate, leaving #124 (lifting the gate to PPL-only) as the only lever?

## Load-bearing insight TESTED (not assumed)

denken #332 proved the verify-step attention bandwidth floor (34.9% BW utilisation, arithmetic
intensity **AI = 7.88 flop/byte ≪ ridge 208**) is **occupancy-saturated** (the adaptive 3D split-KV
verify already launches **96 CTAs > the A10G's 80 SMs**) yet still sits at the BW floor — so the
exposed slack is the low-AI attention floor of the **batched multi-token verify forward** reading the
KV cache, **not a property of EAGLE-3's drafter**.

**Hypothesis (confirmed by this screen): any speculative method that verifies M>1 candidate tokens in
one batched target-model forward inherits the SAME deterministic-attention supply tax under strict
greedy-identity.** Switching the draft method does **not** escape the strict supply RED — only #124 does.

## Method (CPU-analytic over banked anchors + the literature candidate set; re-derives nothing measured)

For each candidate class, tabulate (i) strict supply tax `inherits_332_supply_tax`; (ii) PPL-only
realistic E[T] on our ~100% reasoning/STEM eval (literature RANGES) vs the deployed EAGLE-3 head's
E[T] = 3.8512 (#289 a_k survival product); (iii) PPL-only ceiling TPS = 520.953·(E[T]_method/3.8512).

## Results

### (D1) Strict supply-tax screen — method-independent #332 cap = 473.53 < 500

bandwidth-bound (AI 7.88 ≪ ridge 208) = **True**; occupancy-saturated (96 > 80 SMs) = **True** →
the tax is method-independent.

| Method | batched multi-token verify | `inherits_332_supply_tax` | strict ceiling | clears 500 |
|---|---|---|---|---|
| Lookahead / Jacobi | True | **True** | ≤ 473.53 | False |
| n-gram Prompt-Lookup (PLD) | True | **True** | ≤ 473.53 | False |
| Medusa multi-head | True | **True** | ≤ 473.53 | False |
| Self-spec (Draft&Verify) | True | **True** | ≤ 473.53 | False |

`all_methods_inherit_supply_tax = True`. The 473.53 ceiling round-trips from the #332 floor
(`LAMBDA1_CEIL·(1−floor_geo)`) to ≤ 1e-6.

### (D2) PPL-only E[T] screen — vs deployed head E[T] = 3.8512, retrain target 6.1112

| Method | E[T] [low, central, high] | ratio (central) | ceiling TPS (central) | flags |
|---|---|---|---|---|
| Lookahead / Jacobi | [1.40, 1.60, 1.80] | 0.415 | 216.4 | training-free |
| n-gram PLD | [1.00, 1.05, 1.10] | 0.273 | 142.0 | training-free |
| Medusa multi-head | [2.00, 2.30, 2.80] | 0.597 | 311.1 | **needs training** |
| Self-spec (Draft&Verify) | [1.30, 1.50, 1.80] | 0.389 | 202.9 | **PPL-risk** |

**`best_ppl_only_alternative_etratio` = 0.5972** (Medusa, central) — **TEST metric**. Best optimistic
(high) anchor across all methods = 0.7270 (Medusa high). Best **training-free** central = 0.415
(lookahead). Every method's *optimistic high anchor* (max 2.8) stays below the existing head's 3.8512.

### (D3) Verdict

| Bool | Value | Expected |
|---|---|---|
| `any_non_eagle3_escapes_strict_supply_tax` | **False** | False |
| `any_non_eagle3_beats_eagle3_head_ppl_only` | **False** | False |
| `only_124_lever_under_strict` | True | — |
| `retrain_is_better_ppl_only_bet` | True | — |

**One-line synthesis:** under **STRICT** the draft-method choice is *irrelevant* (all ≤ 473.53 < 500)
→ only **#124** (lifting the gate) moves the >500 lane; under **PPL-only** the **existing EAGLE-3 head
already beats every alternative** (best ratio 0.597 central / 0.727 optimistic, both < 1), and the
coverage-retrained head (E[T]→6.1112, wirbel's path) dominates further. No non-EAGLE-3 method escapes
the lane.

## Self-test (PRIMARY)

`non_eagle3_feasibility_self_test_passes = True` — **25 checks**, NaN-clean. Covers: (a) #332 supply
floor + 473.53 ceiling round-trip ≤ 1e-6; (b) EAGLE-3 E[T] reproduced from the #289 a_k profile
≤ 1e-6 (= 3.8512); (c) each method row has `inherits_332_supply_tax` + a cited ordered PPL-only E[T]
range, all below the head, NaN-clean; (d) both verdict bools computed (both False); (e) the eval is
100% reasoning/STEM (lawine #330) summing to 128 prompts; plus extras (f–k): all methods inherit the
tax, no strict ceiling clears 500, optimistic best ratio < 1, retrain dominates all alternatives,
Medusa needs-training + self-spec PPL-risk flagged, verify bandwidth-bound + occupancy-saturated.

## Honest caveats

- **Ranges, not false precision:** literature accept rates are workload-dependent point estimates,
  carried as low/central/high bands; the verdict is robust across each method's *whole* band — even
  every method's optimistic high anchor stays below the head's 3.8512.
- **Lookahead E[T] revised UP** vs the PR's ~1.2–1.4× hint, to **[1.4, 1.6, 1.8]** (Fu et al. ICML
  2024 + the NeurIPS-2025 lookahead follow-on; GSM8K-on-CodeLLaMA ~1.8×). This is *more generous* to
  the alternative and the negative verdict still holds (ratio ≤ 0.467).
- **LayerSkip excluded from the identical-cap claim:** the self-spec row is represented by
  **Draft&Verify** (arXiv:2309.08168), which runs a *separate* full batched target forward → cleanly
  inherits the tax. **Stock LayerSkip** (arXiv:2404.16710) reuses the early-exit forward and only runs
  the model *tail* to verify, so the identical 473.53 cap cannot be transferred without a measured
  profile — but it *also* needs a specially-trained model (not a drop-in) and carries PPL risk, so it
  fails the clean option-B test on two other axes regardless. Not a free escape.
- **PPL-risk flag:** raw layer-skip / early-exit emission *without* a full verify alters PPL and
  breaks greedy identity — out of scope for a speed-only screen.
- **Cheap one-run screen IF #124 lifts the gate:** the cheapest confirmation is ONE A10G run of vLLM's
  native ngram (`--speculative-method ngram`) and/or lookahead on the 128-prompt eval, reading the
  realised E[T] from the spec-decode acceptance counters — no training, one official draw. This screen
  predicts that run lands E[T] < 1.5 (training-free) ≪ the head's 3.8512.
- **Scope:** this is a SCREEN scoping option B, **not a build recommendation**. It answers "does any
  *different* method beat the EAGLE-3 lane?" (no). The build decision stays with the EAGLE-3 head +
  coverage retrain under #124, or the #124 gate-lift itself under strict.

## Provenance (banked anchors imported verbatim — re-derives nothing measured)

denken #332 `y5cl0ena` (BW 34.9%, AI 7.88, ridge 208, geometric φ 0.925, floor 0.09841, strict ceiling
473.53, ceiling 520.953, >500 budget 4.022%) × kanna #289 `fi34s269` (a_k profile, E[T] = 3.8512) ×
stark #340 `jwv1vbug` (coverage-retrain target E[T](0.9213) = 6.1112) × lawine #330 `hfrscdai` (eval
100% reasoning/STEM, mmlu_pro 57 / gpqa 57 / aime 14, coverage prior 0.8903). All in
`wandb-applied-ai-team/gemma-challenge-senpai`. Literature: Fu et al. ICML 2024 lookahead/Jacobi
(arXiv:2402.02057); Cai et al. 2024 Medusa (arXiv:2401.10774); Saxena 2023 / vLLM ngram prompt-lookup;
Zhang et al. ACL 2024 Draft&Verify (arXiv:2309.08168) + Elhoushi et al. 2024 LayerSkip
(arXiv:2404.16710); Leviathan et al. ICML 2023 spec-decode distributional guarantee (arXiv:2211.17192).

> **Non-blocking note for the advisor:** the PR body cites lookahead as `arXiv:2401.15077`, but that id
> is **EAGLE-1** (Li et al.). The Fu et al. lookahead/Jacobi paper is **arXiv:2402.02057** — used in the
> card.

## Reproduce

```bash
cd target/ && python research/validity/non_eagle3_500_feasibility/non_eagle3_500_feasibility.py \
  --self-test --wandb_group non-eagle3-500-feasibility --wandb_name stark/non-eagle3-500-feasibility
```

- **W&B run:** `stya77pu` (`wandb-applied-ai-team/gemma-challenge-senpai`), 60 summary keys.
- **Peak memory:** 12.13 MiB (CPU-only). **Baseline UNCHANGED at 481.53; 0 official-TPS.**
