STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["mfda3gmr"],"primary_metric":{"name":"admissible_iid_greedy_pool_size","value":0},"test_metric":{"name":"ci_world_constructible","value":0}}

## Results — `CI_WORLD_UNREACHABLE`

**Verdict: the CI world is NOT constructible. The human's "choose CI" branch collapses to shipping nothing; the as-applied POINT gate is effectively forced — my #716 recommendation is strengthened to a near-necessity.** The result is robust under *every* reading of the admissibility criterion (table below): the pool falls short of the int8-locus's required `n=1040` whether you demand strict difficulty-comparability (admissible **0**), relax to CI-overlap comparability (**767**), or ignore comparability entirely and count the whole realized corpus (**919**) or even the complete theoretical AIME universe (**975**). All four are `< 1040`. The binding constraint is **COUNT** — and it binds *dataset-independently*.

### 1. COUNT binds against the entire AIME universe — the robust, load-bearing kill

CI-certifying the int8-locus greedy point (0.450) at 95% power needs **n=1040 distinct iid greedy draws** (#716 frontier `kyp59hd6`, carried verbatim). One greedy draw = one distinct problem (the #716 deterministic cap: re-seeding a solved problem yields zero new information; int8-*sampled* 0.410 < gate, so sampling reaches n but fails the point — problem-pool expansion is the *sole* CI-rescue path). So the CI world needs ≥1040 distinct, gate-gradeable, comparable problems.

The **entire history of AIME is too small, dataset-independently:**
- AIME began 1983; **one** exam/yr (15 problems) through 1999; the alternate AIME II was introduced in 2000 → **two** exams/yr (30) from 2000 on.
- Complete non-reference universe (1983–2023, excluding the 2024+2025 gate reference): `1983–1999: 17×15 = 255` + `2000–2023: 24×30 = 720` = **975 distinct same-format problems.**
- **975 < 1040 — shortfall 65 problems — even ignoring comparability, contamination, and dataset gaps.** No past-AIME corpus, however complete, can reach 1040. Against full-g32's `n=2889` the gap is **1914**.

Realized from the public `di-zhang-fdu/AIME_1983_2024` corpus (independently re-censused here): **933** distinct integer-answer problems; minus the 14 of them that are 2024 (overlap with the gate reference) = **919** addable distinct → 919 < 1040, shortfall **121**.

### 2. COMPARABILITY compounds it — no historical year-band matches the canonical base rate

Anchor (the gate's canonical 2024+2025, 60 problems, **base-bf16 greedy**, the *identical* `aime_eval.py` harness, T=0, no-thinking, max_tokens 6144): **0.50 (30/60)** — 2024 0.600 / 2025 0.400; extract_fail 0; 7/60 length-truncated (mostly the harder recent years). Comparability band **±0.05 → [0.45, 0.55]** (justified below).

| source | sampled n | base greedy | Wilson 95% | Δ vs anchor | grader-faithful | in band (point) | admissible | distinct avail |
|---|---|---|---|---|---|---|---|---|
| past-AIME 1983–1994 | 40 | 0.775 | [0.625, 0.877] | +0.275 | yes | no | no | 152 |
| past-AIME 1995–2004 | 40 | 0.675 | [0.520, 0.799] | +0.175 | yes | no | no | 214 |
| past-AIME 2005–2014 | 40 | 0.600 | [0.446, 0.737] | +0.100 | yes | no | no | 284 |
| past-AIME 2015–2023 | 40 | 0.600 | [0.446, 0.737] | +0.100 | yes | no | no | 269 |
| secondary: AMC-12 | 77 | 0.805 | [0.703, 0.878] | +0.305 | **no** (5-way MC instrument) | no | no | 0 (77 if faithful) |
| secondary: MATH L5 | 65 | 0.815 | [0.705, 0.891] | +0.315 | **no** (48.5% integer golds) | no | no | 0 (65 if faithful) |

Base greedy **declines monotonically with recency** (1983–94 0.775 → 1995–04 0.675 → 2005–14 0.600 → 2015–23 0.600 → anchor 0.50), consistent with older problems being both genuinely easier and more contamination-exposed (1983–2014 problems + solutions are ubiquitous online). **No historical year-band's point estimate lands inside [0.45, 0.55]** — the most-recent band (2015–23, 0.600) is the closest and is still +0.10 above. So under strict point-comparability the admissible pool is **0**.

**Admissibility robustness ladder (every reading still < 1040):**

| reading | admissible pool | shortfall vs 1040 |
|---|---|---|
| strict point-comparability ([0.45,0.55] on the point) | **0** | 1040 |
| CI-overlap comparability (Wilson CI intersects [0.45,0.55]) | **767** (1995–04 + 2005–14 + 2015–23; 1983–94's CI lo 0.625 > 0.55) | 273 |
| ignore comparability, realized corpus (dedup'd) | **919** | 121 |
| ignore comparability, complete AIME universe 1983–2023 | **975** | 65 |

Honest caveat I want to flag, *because it makes the count argument the decisive one rather than comparability:* the anchor's depression to 0.50 is half-driven by a single hard exam. 2024 alone is 0.600 — exactly the recent-historical band rate — and it is the 2025 exam (0.400) that pulls the 60-problem anchor down. The two most-recent bands' Wilson CIs ([0.446, 0.737]) **overlap** the comparability band, so I do *not* claim they are *significantly* too easy at n=40; their point estimates are out-of-band, nothing stronger. That is exactly why I rest the verdict on **COUNT** (hard arithmetic, dataset-independent, CI-free) and treat comparability as a compounding-but-softer second constraint — not the reverse.

**Secondary sources cannot rescue the count** (the card's named format/grader criterion):
- **AMC-12** (`AI-MO/aimo-validation-amc`): 77/83 answers normalize to integers 0–999, so a numeric grader *is* bolt-on-able — but the SOURCE instrument is 5-way **multiple-choice** (≠ the gate's free-form integer generation), and AMC is *by construction the exam that qualifies entrants for AIME* → strictly easier difficulty class. Measured base greedy **0.805** (Wilson [0.703, 0.878], Δ **+0.305** vs anchor — far above the band, "too easy" exactly as expected for the AIME-qualifier exam). Excluded on format **and** difficulty.
- **MATH level-5** (`nlile/hendrycks-MATH-benchmark`, test, `level==5`): only **65/134 (48.5%)** gold answers are bare integers 0–999; 51.5% are fractions/expressions/tuples → the gate's boxed-int-0-999 grader is unfaithful on the majority; the gradeable 48.5% is a biased non-contest slice (different instrument than AIME). Measured base greedy on the gradeable subset **0.815** (Wilson [0.705, 0.891], Δ +0.315 — the easy-slice bias confirmed; shown for completeness, excluded on grader regardless of rate).

### iid-greedy premise (instruction #4)
Greedy is deterministic and batch-invariant (#716 cap) → distinct problems give genuine independent draws. Determinism spot-check (first 4 problems of band 1, re-run): **4/4 identical answers (agree_frac 1.0)**. AIME problems are freshly authored per exam → no cross-year near-duplication inflating the count (no near-dup pruning needed; effective independent count = distinct count). (Contamination on the old, count-rich eras does not reduce the *distinct* count but does undermine whether those draws are independent *capability* samples vs recall — a further reason the count-rich old eras are inadmissible even setting comparability aside.)

### n(p) frontier carried forward (#716, `kyp59hd6`)
`p=0.420→∞ / 0.446→1385 / 0.450→1040 / 0.470→375`. At the constructible n (≤919 raw, ≤975 universe, 0 strict-comparable — all < 1040), the int8-locus greedy point (0.450) is **NOT** CI-certifiable on any assemblable AIME pool. full-g32 (0.438, n=2889) is even further out of reach (universe is 1914 short).

### Residual definitional note for the human
*Does an expanded-year AIME pool still count as "AIME" for the #515 gate?* — **moot.** Even granting the most permissive answer (yes, any AIME year counts; ignore comparability and contamination), the complete universe (975) is still 65 short of 1040. The definitional question never gets to bind because the arithmetic fails first. The CI option is dead by counting, before any judgment call is needed.

### A side finding worth flagging (gate-denominator hygiene)
The base AIME greedy rate is extremely **max_tokens-sensitive**: bf16 base = **0.10 @ max_tokens 3072** (the historical `base_fullhead_aime_n60` gate denominator) vs **0.50 @ 6144** — AIME reasoning is truncated below ~6k tokens (7/60 still truncate at 6144). Anchor and all bands here use 6144 throughout, so this comparison is internally apples-to-apples, but the gate's nominal "base" denominator (and therefore the exact "≥90% of base" bar) is budget-dependent and worth re-confirming at the gb6144 budget the arms are scored on.

---
**Public evidence used:** my #716 cert-budget frontier (`kyp59hd6`, `any_lane_greedy_pool_feasible=0`, the n(p) lookup this card carries); the live AIME harness `research/downstream_quality_aime/aime_eval.py` (canonical 2024+2025 basis, the same harness lawine #703's gate panel uses); fern #659 int8-locus 0.450 (`nmjvtfov`); public math corpora `di-zhang-fdu/AIME_1983_2024`, `nlile/hendrycks-MATH-benchmark`, `AI-MO/aimo-validation-amc`.

### Command
```
# bf16 BASE master served locally (analysis-only, NO HF job, in-memory, port 8000):
#   vLLM google/gemma-4-E4B-it, --dtype bfloat16, --max-model-len 8192
bash research/validity/ci_world_pool_feasibility_719/serve_bf16_base.sh   # base server
bash research/validity/ci_world_pool_feasibility_719/run_measure.sh        # anchor (aime_eval) + 3 bands + determinism
bash research/validity/ci_world_pool_feasibility_719/run_band_1995_2004.sh # + 1995-2004 era
bash research/validity/ci_world_pool_feasibility_719/run_secondary.sh      # AMC + MATH-L5 (secondary)
python research/validity/ci_world_pool_feasibility_719/aggregate_and_log.py \
  --band-results results/band_results.json results/band_results_1995_2004.json \
  --secondary-results results/secondary_results.json \
  --anchor-acc 0.50 --anchor-n 60 --anchor-label "AIME2024+2025" \
  --tol 0.05 --exclude-years "2024" --wandb \
  --out results/ci_world_feasibility_summary.json
```
- **Peak GPU memory:** 19.5 GiB (bf16 base server, single A10G). **analysis_only=1, official_tps=0, no_hf_job=1, fires=0.**
- **W&B run:** `mfda3gmr` (group `ci-world-pool-feasibility-denken`). Locked submission `int4_g128_lmhead` @ 126.378 untouched; this is a MEASUREMENT card, not a fire.

### What happened
The hypothesis asked whether the "unless" in #716 — int8 edge-feasible at n=1040 *on a hypothetical ~1000-problem pool* — could be realized. **It cannot, by counting.** The required pool size (1040) exceeds the entire AIME problem universe ever set (975), so the constraint is structural, not a tuning shortfall, and it holds without leaning on any of my measurements. The measured comparability data compounds it (every historical band is easier than the canonical set; the count-rich old eras are also the most contamination-exposed), and the only count-rich alternatives (AMC, MATH) fail the gate's format/grader by construction. The CI standard is therefore operationally unachievable: "choose CI" = ship nothing → the as-applied POINT gate is forced.

### Suggested follow-ups
- **Hand the gate-semantics call to the human as POINT-forced.** With CI proven unreachable by counting, the only live question is the POINT margin preference among already-passing lanes (int8-locus 0.450, +0.030 margin > full-g32 0.438, +0.018) — no new measurement needed for the CI branch.
- If anyone still wants a CI-style guarantee, it must come from a *different* assurance design (a tighter one-sided test on a smaller pre-registered set, or a Bayesian posterior with an informative prior) — **not** from problem-pool expansion, which is now closed.
- Re-confirm the gate's base AIME denominator at the gb6144 budget (the 3072-era 0.100 is stale by 5×); this changes the exact "≥90% of base" bar if the program revisits the gate.
