<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #348 — REST gate-free >500 feasibility: a no-retrain method on OUR reasoning eval?

**PRIMARY `rest_feasibility_self_test_passes` = True** (all 11 checks)
**TEST `rest_etratio_vs_eagle3` = 0.5219** (E[T]_REST generous-upper / E[T]_eagle3; <1 across the whole band)
**TEST `rest_is_gatefree_deployable` = True** (REST needs NO gated retrain — its headline advantage)
**W&B `usfgt24i`** (group `rest-gatefree-500-feasibility`) · LOCAL read-only analytic, 0 GPU, 0 TPS

> **Verdict: REST is gate-free but NOT a viable >500 path on our eval.** REST is the one >500 candidate
> that deploys with **no #319 gated retrain** (offline CPU suffix-array datastore, no gradient training,
> exact Leviathan-2023 accept rule → PPL-preserving). But our eval is **100% reasoning/STEM** (lawine
> #330) — REST's *weak* case. Its mean accepted length on our eval falls in the band **E[T] ∈ [1.0
> floor, 2.01 generous chat-analog upper]** vs the deployed spec head's **E[T] = 3.8512** (denken #289).
> So its PPL-only ceiling is **135–272 TPS**, and even REST's **global best** published number (code,
> M=2.69) reaches only **364 TPS** — below **both** 500 and the deployed **481.53**. The no-retrain
> advantage is **real but irrelevant**: REST is demand-capped *below the existing already-priced lane*.

## REST E[T] on our eval vs the deployed spec head (deliverable 1 — TEST)

REST's "mean generation length" M (tokens emitted per target forward step, M=1.0 = no-draft bonus floor)
is the **same axis** as the deployed head's E[T]. Literature (He et al. 2024, arXiv:2311.08252, carried
as **ranges**):

| workload | REST mean accepted length M | relation to our eval |
|---|---|---|
| **code** (HumanEval) | **2.53 – 2.69** | REST's STRONG case — **NOT** our eval |
| **chat** (MT-Bench) | **1.97 – 2.01** | closest published analog to general text |
| **reasoning / math / CoT** | **~1.0** (no published REST number) | principled no-draft floor as retrieval misses novel CoT |
| **deployed spec head** E[T]_eagle3 | **3.8512** (denken #289 `fi34s269`, 1+Σcumprod a_k) | the existing already-priced lane |

Our eval is **100% reasoning/STEM** (mmlu_pro 57 / gpqa 57 / aime 14, n=128; lawine #330 `hfrscdai`) —
novel LaTeX/math CoT, the **worst** case for verbatim retrieval. So the relevant band is **[1.0 floor,
2.01 generous chat upper]**. The decisive structural fact: **even REST's GLOBAL BEST (code 2.69) <
deployed 3.8512** — REST under-accepts vs the existing head on *every* workload, far worse on reasoning.

- `rest_etratio_vs_eagle3` (headline TEST) = **2.01 / 3.8512 = 0.5219** (uses the generous chat-analog
  upper — a *measured* number that over-states reasoning; the verdict is range-robust).
- floor ratio = **0.2597**; code-sanity ratio = **0.6985**. All **< 1**.

## Strict supply tax (deliverable 2)

REST verifies a **Trie of up to c=64 retrieved candidates in a single batched tree-attention forward**
(m≤10 draft tokens/step; Medusa/SpecInfer-class). That **is** a batched multi-token verify, so REST
**inherits denken #332's method-independent determinism BW floor**: `inherits_332_supply_tax = True`.
Under a strict greedy-identity gate its ceiling is further taxed to `520.953·ratio·(1−0.09103)`. The
method-independent strict ceiling **520.953·(1−0.09103) = 473.53** round-trips denken #332 (`y5cl0ena`)
to ≤1e-6. **But the supply tax is moot** — REST is already demand-capped far below 500.

## PPL-only ceiling (deliverable 3)

`REST_ppl_only_ceiling = 520.953 · (E[T]_REST / E[T]_eagle3)` (the PR's exact step-3 formula). The
deployed head's own PPL-only ceiling is `520.953·(E[T]_eagle3/E[T]_eagle3) = 520.953` (the λ=1 ceiling).

| E[T]_REST regime | PPL-only ceiling (TPS) | vs 500 | vs deployed 481.53 | vs eagle3 520.95 |
|---|---|---|---|---|
| reasoning floor (1.0) | **135.3** | ✗ | ✗ | ✗ |
| generous chat upper (2.01) | **271.9** | ✗ | ✗ | ✗ |
| global best — code (2.69) | **363.9** | ✗ | ✗ | ✗ |
| deployed eagle3 (3.8512) | 520.95 | — | — | — |

`rest_ceiling_beats_eagle3 = False`; `rest_ceiling_reaches_500_even_code = False`;
`rest_ceiling_beats_deployed_even_code = False`. REST falls short everywhere.

## The operational verdict (deliverable 4 — the distinctive deliverable)

> **REST IS gate-free deployable** (`rest_is_gatefree_deployable = True`) — offline CPU suffix-array
> datastore, **no** gradient training, **no** #319 gated cluster retrain, exact Leviathan-2023 accept
> rule (PPL-preserving). The deploy-today advantage is **real**.
> **But `rest_is_viable_500_path = False`** — on our 100%-reasoning eval REST under-accepts so badly
> (E[T] band [1.0, 2.01] vs 3.8512) that its ceiling (135–272 TPS) sits far below the existing
> EAGLE-3 head's already-priced lane (520.95 ceiling / 481.53 deployed).

`gatefree_advantage_real_but_irrelevant = True`: the no-retrain advantage **does not rescue** a method
that is demand-capped below the existing lane. There is (per this screen) **no free >500** via REST on
our eval.

## Honest caveats (deliverable 5 — carried in the artifact)

1. **Literature accept rates are workload-dependent point estimates → carried as RANGES** (code
   [2.53,2.69], chat [1.97,2.01], reasoning has **no published REST number**, principled floor ~1.0).
   The headline ratio uses the **generous** chat upper (2.01), which over-states reasoning; the verdict
   loses across the whole band, so it is **range-robust**.
2. **Datastore quality/size is a free parameter.** REST's M scales log-linearly with datastore size on
   code (0.9 GB→1.96 … 27 GB→2.65) **but out-of-domain collapses to ~1.0** — domain match dominates
   size. There is no large verbatim-reasoning-CoT corpus, so even a huge datastore won't lift novel
   reasoning M much above the floor. Assumed regime: best-case in-domain reasoning datastore ≈ chat
   hit-rate (the generous upper); realistic ≈ floor.
3. **A measured REST E[T] on our eval needs a (cheap, gated) run** — *named*: build a reasoning
   datastore offline on CPU (suffix array over a STEM/math corpus) → serve REST tree-verify → measure
   mean accepted length on the 128-prompt eval (gated local-GPU smoke or HF Job). **NOT drawn here**; no
   build is claimed beyond the screen.
4. **"EAGLE-3 head" = the deployed/existing already-priced spec head** that denken #289 decomposed
   (E[T]=3.8512), not the unbuilt EAGLE-3 coverage-retrain candidate (which itself needs the #319 gated
   spend). REST is compared against the existing lane.

## Self-test (NaN-clean, deterministic — 11 checks)

(a) E[T]_eagle3 reproduced from #289 a_k ≤1e-6 (3.851186 == 1+48684/17075; cumprod(a_k)==survival) ·
(b) #332 supply floor + 473.5 strict ceiling round-trip ≤1e-6 · (c) REST E[T] ranges cited, finite,
ordered · (d) `inherits_332_supply_tax` computed (True) · (e) #330 eval composition cited (100%, 57/57/14,
n=128) · (f) TEST ratio ∈ (0,1) and <1 · (g) ceiling loses (no beat-eagle3, no reach-500, no
beat-deployed) · (h) gate-free True yet viable-500 False · (i) REST under-accepts globally (code 2.69 <
3.8512) · (j) imports exact · (k) NaN-clean.

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no datastore build, no model
forward, no HF Job, no submission, NOT a launch, NOT a build. REST itself is exactness-preserving
(Leviathan 2023 accept rule), so it would not change PPL **if** built — but nothing is built here.
BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off

REST is **the** gate-free no-retrain >500 candidate, and that part holds (`rest_is_gatefree_deployable =
True`). But its accepted length does **not** hold on our 100%-reasoning eval: E[T] band [1.0, 2.01] vs
the deployed 3.8512 → PPL-only ceiling 135–272 TPS, and even REST's global best (code 2.69) reaches only
364 TPS, below both 500 and the deployed 481.53. `rest_etratio_vs_eagle3 = 0.5219` (<1 across the whole
band). REST also inherits denken #332's batched-verify supply tax. **Verdict: gate-free but not viable
on our eval — the no-retrain advantage is irrelevant when the method is demand-capped below the existing
lane.** A measured E[T]_REST would need a cheap gated reasoning-datastore run (named, not drawn).

## Public evidence used

- **Banked W&B anchors** (all `wandb-applied-ai-team/gemma-challenge-senpai`): denken #289 `fi34s269` +
  lawine #282 `2j0e8xgg` (deployed head a_k → E[T]=3.8512); denken #332 `y5cl0ena` (supply floor
  0.09103, strict ceiling 473.5, λ ceiling 520.953); lawine #330 `hfrscdai` (eval = 100% reasoning/STEM,
  57/57/14, cov prior 0.8903). Official frontier 481.53 TPS (PR #52, `2x9fm2zx`).
- **Literature:** He et al. 2024, *REST: Retrieval-Based Speculative Decoding* (arXiv:2311.08252) — mean
  accepted length M code [2.53,2.69] / chat [1.97,2.01], datastore-size ablation (Table 2), Trie c=64
  tree-attention verify, offline CPU suffix array; Leviathan et al. ICML 2023 (arXiv:2211.17192) —
  distributional-equivalence guarantee.
- **Challenge board:** the public digest shows **no** prior REST / retrieval-spec-decode message or
  result — REST is an untried lane, consistent with the PR's "is there a no-retrain >500 method?" framing.
  This card **refutes** REST as a viable >500 path on our reasoning eval (negative screen).
