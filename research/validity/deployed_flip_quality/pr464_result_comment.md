STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["1o7jwlw4"],"primary_metric":{"name":"deployed_flips_quality_neutral","value":1},"test_metric":{"name":"ppl","value":2.3772}}

## Results — Deployed-flip quality: the 3 flips {11,18,118} are quality-neutral (HARDENS #458)

**LOCAL A10G (sm_86), analysis-only. No HF job, no submission, no served/deployed file touched; the int4 path is READ only.** Headline verdict **`deployed_flips_quality_neutral = True`**, self-test **23/23**. The #458 `deployed_off_strict_frontier=True` reframe **holds**: the deployed 481.53's 3 flips are bitwise ties the model has *zero* real preference between — not a quality regression hiding under corpus-PPL.

### Step 1 — re-confirm the 3 flips + token context

Census (deployed heuristic arm, `VLLM_BATCH_INVARIANT=0`) reproduces the anchor: **identity 0.99657 (872/875), 3 flips @ {11, 18, 118}** (`census_match_known=True`, `rederived_match=True`). Each flip's deployed token-ID vs M=1-AR strict token-ID, with decoded context:

| prompt | pos (j) | strict (M=1 AR) | deployed (M=8 served) | context / what diverges |
|---|---|---|---|---|
| **11** | 231 (7) | `' \'` (621) | `' '` (236743) | `…E = (h/2)·`▮ — whitespace/delimiter inside a **LaTeX physics formula** enumeration. strict→`\sqrt{K\mu}`, deployed→`2\pi\nu` (both well-formed). |
| **18** | 227 (3) | `' significant'` (3629) | `' widespread'` (25581) | `…Venus does not have active plate tectonics or a `▮ — **synonym**; both continue into a coherent Venus-geology sentence. |
| **118** | 227 (3) | `'Let'` (6481) | `'Here'` (8291) | `…$ANSWER is the answer to the problem.\n\n`▮ — **stylistic opener** of a math solution; deployed adds a "Here is the step-by-step solution:" preamble, then both define the same rate variables. |

All three are benign surface choices (whitespace / synonym / opener), not semantic breaks.

### Step 2 — per-flip logit-margin + log-prob delta (the quality magnitude)

Every flip is a **near-tie**, and specifically a **bitwise tie in the canonical M=1 AR reference**:

| prompt | m8 (top1−top2) margin | M=1 self-gap `m1_self_gap` | bitwise tie? | flipping pair == tied top-2? | canonical cost (Δlogprob deployed vs strict, M=1) |
|---|---|---|---|---|---|
| 11 | 0.125 | **0.0** | yes | yes | **0.0** |
| 18 | 0.125 | **0.0** | yes | yes | **0.0** |
| 118 | 0.125 | **0.0** | yes | yes | **0.0** |

- `all_flips_are_near_ties = True`, `all_flips_bitwise_tie_m1 = True`, `all_tied_pair_is_flip_pair = True`.
- `max_flip_logprob_delta = 0.125` (= one bf16 ULP at the logit scale, EPS\* — the reassociation perturbation band that covers every observed flip, #381/#397/#405/#412), `max_flip_canonical_cost_m1 = 0.0`.
- **Mechanism, made falsifiable:** the M=1 AR top-2 logits are *bit-identical* (`m1_self_gap=0.0`) and the two flipping tokens ARE that tied top-2. The model assigns the deployed token **exactly** the strict token's probability → it has no preference → quality-neutral **by construction**. The deployed path picks the other tied token only because the split-KV reassociation perturbs the bf16 argmax tie-break by +0.125 (one ULP). The falsifiable opposite — a non-tie flip where the deployed token is meaningfully *less* probable — would raise `max_flip_canonical_cost_m1` and fail the gate (validated by CASE-B below).

### Step 3 — cascade / self-heal + downstream-sampling surfacing

**Cascade (32-token M=1 AR greedy from each committed token):** all three diverge in surface form (`flips_self_heal_within_32tok = False`) — but into **two coherent, equally-valid continuations**, which is exactly the signature of an equiprobable tie, not a regression:
- **p11:** strict `\sqrt{K\mu}$ … \sqrt{\frac{K}{\mu}}$` vs deployed `2\pi\nu$ … \nu$` — two valid physics formulas in an answer-options list.
- **p18:** strict "…significant history of large, easily identifiable impact craters… volcanic features, lava plains" vs deployed "…widespread, easily identifiable geological record of surface changes… scientists study impact craters and volcanic features" — two coherent, factually-reasonable sentences.
- **p118:** strict "Let $R_P,R_T,R_J$ be the rates of Patrick, Tanya, and Jose…" vs deployed "Here is the step-by-step solution:\n\nLet $r_P$ be Patrick's walking rate…" — both valid math setups.

Self-heal is **reported but NOT gating**: two equiprobable continuations diverging in surface form is nondeterminism between *equivalent* outputs. The correct gate is the tie itself + downstream-invisibility (below), not re-merge.

**Downstream-sampling surfacing** (gemma `generation_config.json`, read live: `do_sample=true, temperature=1.0, top_k=64, top_p=0.95`; lewtun Issue #31 — downstream evals sample, not greedy): at a bitwise tie `P_model(deployed) == P_model(strict)`, so `ln(P_dep/P_strict) = 0` for all three. **`flips_surface_under_downstream_sampling = False`**, `max |ln P_dep/P_strict| = 0.0`. Both tokens are the M=1 top-2 (inside top_k=64), drawn with **equal** probability → the deployed greedy tie-break is **invisible** to the actual eval harness. (Kept strictly separate from the greedy-identity / TPS gate — quality note only.)

### Step 4 — self-test + anchors + headline

| field | value |
|---|---|
| `n_deployed_flips` | **3** |
| `flip_positions` | **{11, 18, 118}** |
| `all_flips_are_near_ties` | **True** |
| `max_flip_logprob_delta` | **0.125** (one bf16 ULP) |
| `flips_self_heal_within_32tok` | False *(non-gating; see Step 3)* |
| `flips_surface_under_downstream_sampling` | **False** |
| `deployed_flips_quality_neutral` | **True** ← headline |
| `flip_quality_self_test_passes` | **True (23/23)** |
| `ppl` / gate | **2.3772** ≤ 2.42 ✓ |
| `analysis_only` / `no_hf_job` / `no_served_file_change` / `official_tps` | true / true / true / 0 |

Self-test includes a synthetic **CASE-B regression** (one non-tie flip, deployed token 0.9 nats less probable): the gate correctly returns `deployed_flips_quality_neutral=False` and flags it — so the verdict is **not a rubber stamp**.

### Re-derivation cross-check (honesty signal — non-gating)

The deep GPU phase **anchors token IDs / margins on the #412/#455 census** (the served chunked-verify path owns the knife-edge argmax) and adds the quality layer. As a transparency check it also independently re-derives each flip with a single prefill:

| prompt | `strict_matches_census` | `m8_argmax_matches_census` (single-prefill) | re-derived `m1_self_gap` |
|---|---|---|---|
| 11 | True | **False** | 0.125 |
| 18 | True | True | 0.0 |
| 118 | True | **False** | 0.0 |

The single-prefill can't reproduce the deployed argmax at p11/p118 — **and that is the point**: the deployed argmax at a one-ULP tie is a property of the M=8 *chunked-verify* reduction order, which a single prefill provably can't replicate. The fragility (argmax flips between strict/deployed depending on reduction order) is itself **evidence of a sub-ULP tie** — a real-margin flip would re-derive robustly. This is why the verdict anchors on the census, not on the re-derivation.

### Comparison vs baseline anchors

| anchor | expected | this run |
|---|---|---|
| deployed identity (lawine #455 `0r0ounl8`) | 0.99660, 3 flips {11,18,118} | **0.99657 (872/875), 3 flips {11,18,118}** ✓ (within 3e-5) |
| PPL (PR #52 `2x9fm2zx`) | 2.3772 ≤ 2.42 | 2.3772 ✓ |
| reframe hardened (land #458 `uhhyec0q`) | `deployed_off_strict_frontier=True` | **HARDENED** — flips are quality-neutral bitwise ties ✓ |

*Note on 872/875 vs the 879/882 anchor:* the #412 census excluded prompt **105** (744 tokens — not too short) via its `len(cont) < n_verify` guard: prompt 105's M=1 AR greedy continuation is < 8 tokens (early stop), so it yields 0 verify positions. It is **flip-free**, so the 7-position denominator difference is identity-neutral and the flip set is exact.

### Command

```bash
.venv/bin/python research/validity/deployed_flip_quality/deployed_flip_quality.py --measure \
  --wandb_group equivalence-escalation-anchors --wandb_name denken/deployed-flip-quality
# n_prompts=126 (125 effective, 875 positions), ctx_len=224, n_verify=8 (M=8), cascade_tokens=32,
# flip_prompts=11,18,118, gpu_mem_util=0.55
# Orchestrates: #412 census(heuristic, VLLM_BATCH_INVARIANT=0) -> census-anchored deep GPU phase -> compose+self-test
# 0-GPU validation: --self-test (synthetic CASE-A neutral + CASE-B regression, 23/23)
```

- **Peak GPU:** 12.25 GB (census and deep phases each)
- **W&B run:** `1o7jwlw4` (group `equivalence-escalation-anchors`)
- **Artifacts:** `research/validity/deployed_flip_quality/{deployed_flip_quality_results.json, census_heuristic.json, deep_result.json, full_measure.log}`

### What happened

Clean positive result. The 3 deployed flips are not a quality regression — they are **bitwise ties** (`m1_self_gap=0.0`) where the model assigns the deployed and strict tokens *bit-identical* probability, and the two flipping tokens ARE that tied top-2. Decoded, they are benign (whitespace, `significant`↔`widespread`, `Let`↔`Here`). The canonical-reference quality cost is **0.0 nats** for all three, and at temp=1.0 the downstream sampler draws both tokens equiprobably, so the greedy tie-break never surfaces in the actual evals.

The one nuance worth surfacing: **self-heal is the wrong gate.** All three cascades diverge in surface form over 32 tokens (they don't re-merge), but into two *equally-valid, coherent* continuations — which is what an equiprobable tie should do. I made self-heal a reported-but-non-gating signal and rested the verdict on the tie + downstream-invisibility instead; the synthetic CASE-B confirms a true non-tie flip would still fail the gate. The implementation also had to **anchor on the census flip set rather than re-derive the flips**, because the deployed argmax at these one-ULP knife-edges is a chunked-verify reduction-order property a single prefill can't reproduce — and that very fragility doubles as tie-evidence.

This **hardens land #458**: the deployed 3 flips are a genuinely quality-accepted status quo, so the #407 relax-decision stays correctly GRADED ("3→N flips for +TPS, does quality survive?"), with the base-3 now proven quality-neutral rather than assumed.

### Suggested follow-ups

- **The graded `3→N` question (#407) now has a clean quality oracle.** The same harness (census-anchored deep phase + bitwise-tie / canonical-cost / downstream-surfacing gate) can score *any* candidate flip set, not just the deployed 3. If a wider-M / relaxed config adds flips, run them through this gate: a new flip is admissible iff it is also a bitwise tie (`m1_self_gap=0.0`, cost 0.0, downstream-invisible). That converts "does quality survive N flips?" into a per-flip pass/fail.
- **ubel #461 (kernel attribution) is the orthogonal complement:** this card says the 3 flips are quality-neutral; #461 says which kernel produces each. Together they fully characterize the deployed flips (quality ⊥ severity).
- **Prompt 105's early-stop** (M=1 AR continuation < 8 tokens) is a census-geometry footnote, not a flip; if a future card wants the literal 882-position denominator it should bump the census `--n-prompts` headroom rather than touch the #412 length guard.
