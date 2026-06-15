<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #330 — What unconditional top-4 coverage prior does the "ShareGPT" eval carry vs the 0.9213 bar?

**0-GPU analytic card. No build, no HF job, no served-file change, 0 TPS. Greedy/PPL untouched.
BASELINE 481.53 unchanged. W&B run `hfrscdai`.**

## TL;DR — VERDICT: **LIKELY-MISSES** (P(clears 0.9213) ≈ **0.06**, range 0.001–0.15)

The premise this card was asked to quantify — *"the official 128 ShareGPT eval is free-form
conversational with ~0 MCQ-answer-letter tokens, so it sits toward the clearing aime end"* — is
**refuted by the repo's own benchmark data**. The official 128 eval is **100% reasoning/STEM**
(mmlu_pro 57 / gpqa 57 / aime 14), 89% multiple-choice. The honest coverage prior is the
57/57/14-weighted per-source coverage = **0.8903**, which **misses** the bar by **0.0310**. This
resolves lawine #323's "binding unknown" — but flips its optimistic lean.

| quantity | value |
|---|---|
| Official-eval uncond top-4 **prior (point estimate)** | **0.8903** |
| Gap to build bar (0.9213) | **−0.0310 (MISS)** |
| Conservative record-Bernoulli SE | 0.0200 |
| 95% band | [0.8511, 0.9295] (bar near **upper** edge) |
| **P(official-eval uncond top-4 ≥ 0.9213)** | **0.060** (range 0.001–0.15) |
| Verdict | **LIKELY-MISSES** |

## The two premise errors (both push the same way)

**Error 1 — the eval is not free-form ShareGPT.** The file
`official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json` is, despite the
name, **100% reasoning/STEM**: by `id` prefix, **mmlu_pro 57 / gpqa_diamond 57 / aime 14**. **114/128
(89.1%)** are multiple-choice (≥2 lettered options) "ending in `ANSWER: $LETTER`"; the rest are
open-form competition math. **Zero** conversational chat. This was already documented in
`research/DATASET_ANALYSIS.md` (2026-06-13: *"Despite the filename, the public 128 are not ShareGPT
chit-chat… 100% reasoning/STEM"*). The greedy-reference decode
(`research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl`) confirms it: every generation
is a 512-token LaTeX/math/science reasoning chain (latex char-density ≈ 0.066, digit ≈ 0.028), not prose.

**Error 2 — the coverage drag is not the "MCQ-answer-letter tail".** With `ignore_eos=true` +
`max_tokens=512`, only **7.8%** of the 128 generations even *reach* `ANSWER:`; the literal committed
answer-letter token is **~0.0002 (0.02%)** of all generated tokens. Yet mmlu_pro's **whole-CoT**
coverage is 0.8465 (vs aime 0.9570). So the drag is the reasoning-CoT **vocabulary breadth** (mmlu_pro
spans law/philosophy/business/health/… → highest entropy → lowest coverage), not a near-absent letter.
The correct axis is the **source mix**, not a letter fraction.

## Deliverable 1 — token-type composition of the official 128 eval

- **Source mix:** mmlu_pro 57 / gpqa 57 / aime 14 → reasoning/STEM frac **1.000**, free-form chat
  frac **0.000**. MCQ prompts **114/128 (89.1%)**.
- **Uniform token weighting:** `ignore_eos=true` ⇒ every prompt emits exactly **512** completion
  tokens ⇒ token-weight == prompt-weight == source proportions. The per-source aggregate **is** the
  honest token-level prior.
- **Generation signatures (greedy decode):** latex_density 0.066, digit_density 0.028, option-enum
  0.070, answer-tail 0.078. LaTeX/math reasoning, not conversational prose.
- **Literal answer-letter token fraction ≈ 0.0002** → ~420× below the level the letter-contamination
  model would need to matter (see Deliverable 3).

## Deliverable 2 — composition → coverage

Push the composition through #323's per-source curves (fern #34 `gua9x68j`, teacher-forced, the only
trained fusion head):

| source | uncond top-4 | official weight |
|---|---|---|
| aime | 0.9570 | 14/128 = 0.109 |
| gpqa | 0.9176 | 57/128 = 0.445 |
| mmlu_pro | 0.8465 | 57/128 = 0.445 |

- **Point estimate** = 57/57/14-weighted = **0.89027**. This equals fern's *benchmark-matched*
  aggregate (0.89026) to **1e-5** — because the holdout proportions (107/107/26) match the official
  eval (57/57/14). fern's aggregate was **already an on-distribution estimate**.
- **Uncertainty band:** conservative record-Bernoulli SE **0.0200** → 95% band **[0.8511, 0.9295]**,
  bar near the **upper** edge. Native-vs-tf upside (+0.0097 at root) → optimistic central ~0.900.
- **P(clears 0.9213) ≈ 0.060** central (0.001 at tight SE=0.01, 0.150 at wide SE=0.03, 0.143
  native-adjusted). Even the published fully-trained-head central (0.913) misses this reasoning eval;
  only the upper lit edge (0.929) clears.

## Deliverable 3 — sensitivity

**(a) Source-mix axis (correct).** With free-form(aime) fraction `f` and the rest the 57/57 gpqa/mmlu
blend (cov 0.8821): `cov(f) = f·0.9570 + (1−f)·0.8821` crosses the bar at **f ≈ 0.524**. The eval sits
at **f = 0.109** → deep miss (needs ~5× more open-math content to clear).

**(b) The PR's literal X% MCQ-answer-letter model.** Base = free-form @ aime (0.9570) + X% answer-letter
tokens @ `c_mcq`:

| c_mcq | X-cross | X=0% | X=5% | X=10% |
|---|---|---|---|---|
| 0.30 | 5.4% | 0.9570 ✓ | 0.9242 ✓ | 0.8913 ✗ |
| 0.40 | 6.4% | 0.9570 ✓ | 0.9292 ✓ | 0.9013 ✗ |
| 0.55 | 8.8% | 0.9570 ✓ | 0.9367 ✓ | 0.9163 ✗ |

So under this model X∈{0,5} clears and X=10 misses. **But the eval's literal answer-letter fraction is
~0.0002 — ~420× below the ~6% crossing** — so this model predicts a **CLEAR**. That is exactly why the
premise looked optimistic: it localizes the drag to a near-absent token. The data say the drag is the
whole reasoning-CoT body (mmlu_pro 0.8465), which the source-mix axis captures and the letter axis
misses.

## Deliverable 4 — verdict + framing

**LIKELY-MISSES.** Prior 0.8903 ± 0.0200, 0.0310 below the bar, P(clears) ≈ 0.06.

- **Reframes fern #329's GPU-gated RANKPROBE_W=4 read:** because the official 128 **are** the
  reasoning/MCQ distribution (not free-form), the prior predicts the read lands ~0.89 (**NO-GO**),
  *confirming* the miss rather than discovering a clear. #323's "what flips it" framed the official 128
  as *different* from the reasoning holdout; they are the **same distribution** (the holdout is
  "benchmark-matched", deduped against these 128), so the read would tighten the CI, not move the
  central.
- **What would flip it:** (i) the official 128 being unexpectedly aime-like (free-form frac ~0.52 vs the
  actual 0.11 — contradicted by the on-disk ids), or (ii) a **better-trained** fusion head than fern #34
  (soft-KD top-k calibration / more root training) lifting per-source coverage by ≥0.031 — plausible but
  untested, and even the lit-central fully-trained head (0.913) still misses this reasoning eval.

## Honesty / caveats

PRIOR, not a measurement. Built on fern #34's **teacher-forced** per-source coverage (root tf≈native,
gap +0.0097) over a holdout that matches but is disjoint from the official 128. The record-Bernoulli SE
is conservative (token-level SE is tighter → even lower P). The 0.9213 bar is position-1 root coverage:
necessary, **not sufficient** — the deployed deep spine still caps E[T] at 4.91 (#316). fern #34 is the
only trained head and is undertrained (K=1, hard-CE, no soft-KD, no TTT). No fusion checkpoint is
deployed; the drafter BUILD stays human-gated.

## Reproduce

```bash
cd target/ && .venv/bin/python \
    research/validity/eagle3_sharegpt_coverage_prior/eagle3_sharegpt_coverage_prior.py \
    --self-test --wandb_group eagle3-sharegpt-prior --wandb_name lawine/eagle3-sharegpt-coverage-prior
```

Self-test: **21/21 checks PASS**, NaN-clean, peak mem 21.1 MiB. W&B run `hfrscdai`.

## Provenance

Build bar lawine #316 (`5lnz5jgb`) via lawine #323 (`ceddxj20`); per-source + aggregate top-4 fern #34
(`gua9x68j`, W&B-verified `eval/*`); linear spine wirbel #79 (`z6wi4z4v`); official 128 eval composition
loaded live from `eval_prompts_sharegpt.json` + `decode_outputs.jsonl`; source identity confirmed by
`research/DATASET_ANALYSIS.md` (2026-06-13). Composition→coverage prior, sensitivity, and verdict are
this card (#330).
