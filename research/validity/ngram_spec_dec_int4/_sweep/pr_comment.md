STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["7gzefuwk"],"primary_metric":{"name":"best_ngram_proj_official_tps","value":108.12},"test_metric":{"name":"gsm8k_ngram_sampled_acc","value":0.890}}

## Results

**Verdict — draft-free ngram/prompt-lookup spec-decode is a LIGHT, deploy-trivial option-B flavor (no head build, ~10 GB serve) but it is NET-NEGATIVE on speed on the int4 substrate (≈ −16% TPS vs plain AR), and it breaks strict #319. Quality is intact (GSM8K holds). It is NOT speed-competitive with the MTP candidate (427.7), nor even with plain AR (126.378).** `analysis_only=true`, `official_tps=0` (local A10G proxy), single GPU, no HF Job / no submission / no served-file change.

### 1. Speed Pareto — M=1 single-stream, official `decode_outputs` workload (128-prompt sharegpt)

Proxy = vLLM's own *"Avg generation throughput"* (steady, warmup-excluded; the honest M=1 single-stream rate). Local→official via τ=1.0245 anchored on AR.

| config (`prompt_lookup_min=2`) | S_local TPS | proj official TPS | vs AR 126.378 | E[accept] | draft accept |
|---|---|---|---|---|---|
| AR (shipped serve, no spec) | 123.35 | 126.38 | — | — | — |
| ng max2 k3 | 98.39 | 100.80 | **−20.2%** | 2.09 | 0.364 |
| ng max2 k5 | 102.40 | 104.91 | −17.0% | 2.34 | 0.268 |
| ng max2 k7 | 102.40 | 104.91 | −17.0% | 2.47 | 0.211 |
| ng max3 k3 | 99.86 | 102.31 | −19.0% | 2.13 | 0.376 |
| ng max3 k5 | 102.40 | 104.91 | −17.0% | 2.38 | 0.276 |
| ng max3 k7 | 102.76 | 105.29 | −16.7% | 2.51 | 0.217 |
| ng max4 k3 | 99.66 | 102.11 | −19.2% | 2.16 | 0.388 |
| **ng max4 k5 (best)** | **105.53** | **108.12** | **−14.5%** | 2.41 | 0.283 |
| ng max4 k7 | 102.44 | 104.95 | −17.0% | 2.55 | 0.222 |

**Every one of the 9 configs is net-negative.** Best (least-bad) = `ng_max4_k5` (prompt_lookup_max=4, num_speculative_tokens=5).

- **Full-census confirmation (128×512, not the 16-prompt screen):** `ng_max4_k5` = **105.16 TPS** steady = **−16.4%** vs the full-census AR mean (125.8 TPS). The full-census AR (ar_ref 126.48 / ar_ref2 125.11) reproduces the official **126.378** to within **0.5%** — the local proxy is well-calibrated, so the negative delta is real, not a proxy artifact.
- **vs MTP candidate (427.7):** ngram is ~**4× slower**. **vs plain AR (126.378):** ngram is slower.
- **Mechanism (why it loses):** on the real sharegpt mix the draft acceptance is only **25.2%** (full census) → mean acceptance length **2.25 tokens** per verify step, with geometric per-position decay `[0.486, 0.303, 0.203, 0.144, 0.11]`. But each verify pass runs the int4-Marlin body at **M=K+1=6**, which costs ≈**2.7× an M=1 AR step** (2.25 tok / 105 TPS ⇒ 21.4 ms/step vs AR's 7.9 ms/step). 2.25 tokens cannot pay for a 2.7× step → net throughput **loss**. This is the same int4-Marlin M-dependence that breaks MTP strict-identity (my #600, fern #597) and that made the verify-GEMM at M>1 ~2× slower (#144).
- **Why the warm `probe_tps` looked like a 4× win (446–508 TPS) but isn't:** that probe is a *single* fixed prompt — *"Explain step by step how a transformer decodes one token at a time."* — whose output is highly self-referential, a near-100%-copy best case for prompt-lookup. It is the harness's labelled "secondary upper bound only," not the real workload. On the diverse 128-prompt set the steady rate is net-negative.

### 2. Identity (#319) — WARM free-run greedy census, full 128×512, seed 1, official verifier

| comparison | verdict | prompts differ | tokens differ |
|---|---|---|---|
| **`ng_max4_k5` vs AR reference** | **DIVERGENT (invalid)** | **99 / 128** | **28,553 / 65,536 (43.6%)** |
| AR vs AR' (cross-start floor, dev307 control) | **GREEDY_IDENTICAL** | 0 / 128 | 0 |

- **ngram BREAKS strict #319** — exactly as predicted: the verify pass is M>1, so it hits the int4-Marlin batch-variance, flips near-tie argmaxes, and greedy decode cascades (first divergence then propagates).
- **The cross-start floor is a clean ZERO** — two independent fresh server starts produce byte-identical greedy output. So the divergence is **100% attributable to the spec path**, not to dev307 cross-start nondeterminism (lawine #601). In this local env, ref-vs-ref cross-start noise does not manifest, which makes the DIVERGENT verdict unambiguous.

### 3. Quality — GSM8K (sampled T=1.0/top_p=0.95/top_k=64, min_tokens=8 per #541, n=500, seed 1234)

| arm (same int4_g128_lmhead substrate) | accuracy | strict | trunc | verdict (bar 0.807) |
|---|---|---|---|---|
| AR control (no spec) | 0.884 (442/500) | 0.920 | 0.088 | **PASS** |
| **`ng_max4_k5` ngram** | **0.890 (445/500)** | 0.932 | 0.078 | **PASS** |

- **No quality cliff.** ngram 0.890 vs AR-on-the-same-substrate 0.884: Δ = +0.006, well inside sampling noise (SE ≈ 0.014 at n=500). Speculative decoding with rejection sampling is distribution-preserving, and the result confirms it — the tiny int4 verify-logit perturbation does not move GSM8K. Both clear 0.807 with wide margin (base int4 = 0.878 under the same harness).
- **Bonus (CoT acceptance):** during the GSM8K run (concurrency=32) ngram draft acceptance was only **14.9%** — *lower* than the sharegpt M=1 single-stream 25.2%, because at batch=32 the verify width is already filled by batching, so spec adds little even on the structured CoT it was hypothesized to help most.

### Exact commands
```bash
V=/tmp/senpai-venvs/20f658587e8a6643/bin/python   # serve+decode (vLLM 0.22.0)
cd /workspace/senpai/target
# rebuild the named int4 base (shared-disk cleanup had reclaimed it); PPL gate passed 2.0197 (cap 2.42)
bash research/validity/ngram_spec_dec_int4/rebuild_base.sh
# speed screen (AR + 9 ngram configs, 16x512) and full-census identity (128x512)
$V research/validity/ngram_spec_dec_int4/sweep.py screen
$V research/validity/ngram_spec_dec_int4/sweep.py ar_census          # ar_ref + ar_ref2 (cross-start floor)
$V research/validity/ngram_spec_dec_int4/sweep.py census 4 5          # best config ng_max4_k5
# GSM8K quality, two matched arms (AR control + ngram) on the SAME substrate
bash research/validity/ngram_spec_dec_int4/run_gsm8k.sh
# aggregate + official #319 verifier + W&B (MUST run under the project venv that has wandb)
uv run python research/validity/ngram_spec_dec_int4/report_wandb.py
```
ngram serve flag (only delta vs the shipped serve): `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'`.

### Peak memory
ngram serve peak **20,753 MiB** (~20.3 GB; +~0.4 GB of spec scratch over the AR serve's 20,333 MiB) on the 23 GB A10G — comfortably within budget; no OOM. No 163 GB build needed (plain int4, ~10 GB checkpoint).

### W&B
Run **`7gzefuwk`** — `denken/ngram-promptlookup-optionb`, group `ngram-promptlookup-optionb-lane` (metrics under `summary/` prefix; `speed_pareto` table logged). https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/7gzefuwk

### What happened — honest analysis
The hypothesis (a light, draft-free spec-decode flavor for the relaxed-#319 option-B world) is **directionally false on speed for this int4 substrate**: prompt-lookup's modest, realizable acceptance (25% / E[L]=2.25 at M=1) is overwhelmed by the int4-Marlin M=K+1 verify cost (~2.7× an M=1 step). It is net-negative in every swept config and ≈4× off the MTP candidate. The identity break (#319) was expected and is confirmed via the same M>1 mechanism, with a clean cross-start floor (0) proving it is the spec path, not env noise. Quality is the one fully positive finding: no cliff, distribution-preserving, GSM8K holds. So as a *standalone* option-B flavor it is dominated; its only real assets (zero build, trivial deploy) don't offset a −16% speed regression. This is consistent with my prior PR #89 finding that prompt-lookup is redundant with — not complementary to — the strong drafter on this stack.

### Suggested follow-ups (not implemented — flagging per workflow)
1. **Stop pursuing draft-free ngram as a speed lane on int4.** The verify cost is structural to the int4-Marlin M>1 path; no `prompt_lookup_*`/K tuning rescues it (full sweep shown). It only ever wins on degenerate near-100%-copy prompts, which the official sharegpt workload is not.
2. If a draft-free lane is still wanted, the M>1 verify cost would have to be cut (e.g. a verify-side kernel that is M-cheap on int4) — but that is exactly the wall #144/#569 already mapped; ngram doesn't change it.
3. The clean **GSM8K distribution-preservation + zero cross-start floor** here are reusable as a quality/identity control harness for *any* future spec-decode arm on this substrate (the matched-arm + ref-vs-ref-floor protocol).

### Public evidence used
- **Leaderboard / option-B comparison:** MTP candidate `int4_g128+MTP-K7` @ 427.7 official-proxy TPS (fern #597, W&B `p7jo2ap4`) is the option-B speed bar this lane is measured against; public frontier is ~489–508 TPS (e.g. `ff-splitkv-...` 508.6). The draft-free ngram lane (~106 TPS) is not in that regime.
- **Prior prompt-lookup touch:** ubel #503 earlier touched prompt-lookup; no public *result* file currently records a head-to-head ngram-on-int4 measurement (0 matches in the results board), so this card is the first such characterization. It **refutes** the "light spec-decode is a free option-B speedup" intuition for the int4-Marlin substrate.
- **Identity context:** consistent with the int4-body batch-variance that breaks spec-identity (my #600, run `jd8q9q4t`) and the dev307 cross-start caveat (lawine #601) — here the cross-start floor came out clean (0).
