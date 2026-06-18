STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["pr668-recoverable-aime-set","pr668-recoverable-aime-divergence","pr668-recoverable-aime-VERDICT"],"primary_metric":{"name":"divergence_early_frac","value":1.0},"test_metric":{"name":"int4ar_aime_greedy_maj@1","value":0.4}}

## Results

### VERDICT: `FUNDAMENTAL`

First-divergence is **EARLY on 8/8 of R** (mean position **0.21%** of completion, median 0.16%; 3/8 diverge at the *literal first generated token*). The loss is **diffuse** (no year/tier/type cluster dominates). Per the PR's own decision rule — "diffuse + early → recovery needs a body-wide precision bump that eats the speed headroom; a `FUNDAMENTAL` verdict says stop trying to recover int4-body AIME with a tweak" — this is the strategic stop signal: **the int4 body cannot clear 0.420 without a near-bf16 body; there is no concentrated late/head locus to target cheaply.**

A second, independent finding reinforces this and undercuts the "recoverable set" premise itself: **the int4-AR greedy AIME result is not reproducible across serve sessions** (details in §5). The specific problems int4 misses are dominated by near-tie serving noise, so "fix these N problems" is not a well-posed target.

---

### 1. Recoverable set R

From the #650 `_aime_int4ar_mt12288.out` / `_aime_bf16_mt12288.out` transcripts (60 problems, greedy, fl=0 both sides):

| | n | acc |
|---|---|---|
| int4-AR @12288 correct | 24/60 | 0.4000 |
| bf16 @12288 correct | 29/60 | 0.4833 |

- **R (bf16 right, int4 wrong) = 8**: `2024-II-7, 2024-I-14, 2024-II-12, 2025-I-05, 2025-I-08, 2025-II-04, 2025-II-09, 2025-II-12`
- **reverse (int4 right, bf16 wrong) = 3**: `2025-II-06, 2025-II-10, 2025-II-11`
- **net deficit = |R| − |reverse| = 5** (= 29 − 24).
- Flips of R needed: **to bar 0.420 → 2** (24→26/60 = 0.4333); **to bf16 0.4833 → 5** (24→29/60).

### 2. Concentration — **SPREAD, not clustered**

| axis | distribution over R (n=8) | dominant share |
|---|---|---|
| year | 2024: 3, 2025-I: 2, 2025-II: 3 | 0.375 |
| difficulty tier (problem #) | easy(1-5): 2, medium(6-10): 3, hard(11-15): 3 | 0.375 |
| coarse type (keyword heuristic*) | number_theory: 5, geometry: 3 | 0.625 |

No year or difficulty tier holds a majority (max 3/8 = 37.5% < 60% cluster threshold) → **R is spread**. The type axis leans number-theory (5/8) but the keyword classifier is a screening proxy only (it collapsed all 8 into just two buckets), so I do not let it drive the verdict; year/tier are the reliable axes and both say spread.

### 3. First-divergence locus — **EARLY on 8/8**

Re-served int4-AR and bf16 greedy (T=0, BI=1, `--max-model-len 16384`, mt=12288, no-thinking, min_tokens=8) and captured per-token argmax streams + logprobs. First position where the two argmax streams disagree on identical context, normalized by the bf16 (correct-reference) completion length:

| problem | div_idx | frac (bf16 len) | bucket | bf16 tok → int4 tok |
|---|---|---|---|---|
| 2025-I-05 | 0 | 0.00% | EARLY | `Let` → `The` |
| 2025-II-04 | 0 | 0.00% | EARLY | `Let` → `The` |
| 2025-II-12 | 0 | 0.00% | EARLY | `The` → `This` |
| 2024-II-12 | 5 | 0.10% | EARLY | ` squared` → ` value` |
| 2025-II-09 | 5 | 0.16% | EARLY | ` sum` → ` value` |
| 2025-I-08 | 9 | 0.34% | EARLY | ` real` → ` values` |
| 2024-I-14 | 10 | 0.21% | EARLY | ` with` → ` $` |
| 2024-II-7 | 45 | 0.89% | EARLY | ` \` → `,` |

- **buckets: EARLY 8, MID 0, LATE 0.** mean frac **0.21%**, median **0.16%**.
- The int4 and bf16 trajectories split at the *opening* of the response and write entirely different reasoning bodies — this is body-wide drift, **the opposite of a final-answer / head locus**. A head-only or final-layer precision bump cannot recover it.
- **Robust to int4 serving noise:** identical div_idx whether int4 is served at batch-of-1 or at concurrency-11 — the EARLY conclusion does not depend on which (noisy) int4 trajectory is used.
- `finish_length_rate = 0.0` both arms (every completion ended on `stop`, none truncated at the 12288 budget — confirms the clean-budget `fl≈0` condition).

### 4. Verdict scalars (W&B `recoverable-aime-VERDICT`)

`verdict=FUNDAMENTAL`, `early_frac=1.0`, `late_frac=0.0`, `clustered=false`, `R_size=8`, `net_deficit=5`, `flips_to_bar=2`, `flips_to_bf16=5`, `finish_length_rate=0.0`, `analysis_only=true`, `official_tps=0`.

### 5. ⚠️ Major finding: int4-AR greedy AIME is **not cross-session reproducible**

While capturing the divergence streams I found that the int4 transcripts **do not reproduce** across serve processes — even at greedy (T=0) with `VLLM_BATCH_INVARIANT=1`. bf16 (same live process) reproduced #650 **11/11 bit-exact**; int4 (fresh process) did not:

- **Within a single int4 process**, decode batch size flips the answer: conc-1 (batch-of-1) vs conc-11 disagree on **8/11** boundary problems. Prefill chunking (11 prompts × ~300 tok > the 2048 batched-token cap → chunked) perturbs near-tie argmaxes. `VLLM_BATCH_INVARIANT=1` does **not** neutralize this on long AIME trajectories. (`research/_probe/greedy_samecfg_check.sh` independently documents the same model flipping **53/128** prompts when the prefill chunk size changes.)
- **Across sessions, fresh full-60 int4 greedy at #650's exact protocol scored 21/60 (0.350)** vs #650's 24/60 (0.400) — bf16 reference held at 29/60. So the *net deficit is 5 in one session, 8 in another*.
- **36/60 (60%) of all problems give a different greedy answer** across the two int4 sessions; **11/60 flip correctness**.
- The recoverable set itself is only **Jaccard 0.46** reproducible: of #650's 8 R problems, 2 (`2025-I-05`, `2025-II-04`) become *correct* in a fresh session, and 5 brand-new problems enter R. Problem `2024-I-14` produced four different wrong answers across four serving conditions (115 / 250 / 100 / 459; gold 104).

This is consistent with the PR's sampled-AIME anchor (int4 0.3467 vs bf16 0.4600, non-overlapping CIs): int4 is genuinely ~5–11 pts weaker **on average** (a real body-level capability gap), but **which specific problems it misses in any one greedy run is near-tie noise**. Both facts point the same way — the deficit is body-wide and fundamental, not a fixed, concentrated, recoverable set.

---

### Commands

```bash
# recoverable set from #650 .out (no GPU)
python3 research/validity/int4ar_aime_recoverable/decompose_recoverable.py recoverable \
  --int4-out research/validity/int4ar_denom_harden/_aime_int4ar_mt12288.out \
  --bf16-out research/validity/int4ar_denom_harden/_aime_bf16_mt12288.out --tag-type

# greedy+logprobs streams (bf16 from live #650 server; int4 from a fresh server, BI=1)
#   serve: VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve \
#     /workspace/gemma_build/int4_g128_lmhead --served-model-name gemma-4-e4b-it --port 8000 \
#     --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
#     --max-num-seqs 16 --seed 0 --trust-remote-code
python3 .../decompose_recoverable.py harvest --arm bf16 --include-reverse --save-text --out bf16_streams.json
python3 .../decompose_recoverable.py harvest --arm int4 --include-reverse --save-text --client-concurrency 1 --out int4_streams_seq.json

# first-divergence + session-noise + W&B
python3 .../decompose_recoverable.py diverge --bf16 bf16_streams.json --int4 int4_streams_seq.json --out divergence_c1.json
python3 .../decompose_recoverable.py reproducibility   # boundary + fresh-60 vs #650
python3 .../decompose_recoverable.py wandb --divergence divergence_c1.json --int4 int4_streams_seq.json

# fresh full-60 int4 session-noise re-run (matched #650 protocol)
python3 research/downstream_quality_aime/aime_eval.py --base-url http://127.0.0.1:8000 \
  --model gemma-4-e4b-it --years 2024,2025-I,2025-II --k 1 --temperature 0 \
  --max-tokens 12288 --min-tokens 8 --no-thinking --client-concurrency 16 --seed 0
```

- **W&B group:** `int4ar-aime-recoverable-ubel` (entity `wandb-applied-ai-team`, project `gemma-challenge-senpai`); runs `pr668-recoverable-aime-set`, `pr668-recoverable-aime-divergence`, `pr668-recoverable-aime-VERDICT`.
- **Peak GPU memory:** 19.5 GB (int4 server, single A10G); bf16 server comparable.
- **analysis_only:** no HF Job, no submission, no served-file change. `int4_g128_lmhead` @ 126.378 untouched.

### What happened

The decomposition answers the PR's decision question cleanly and in the direction that says *stop*: the int4 deficit is **diffuse + early-divergence → FUNDAMENTAL**. int4 and bf16 split at the first few tokens of the response and reason down different paths; there is no late/head locus a cheap targeted bump could catch.

The reproducibility finding (§5) is the deeper result. The int4 model decides these long reasoning chains on pervasive near-tie argmaxes, so the greedy trajectory — and the final answer — is unstable to ordinary serving perturbations (batch composition, prefill chunking, process epoch), 60% of problems flipping answer across sessions, despite batch-invariance. The aggregate 0.400 carries ≈±3/60 session noise, which is itself comparable to the +1.2-problem gap to the 0.420 bar. **A 0.400-vs-0.420 framing on a single greedy run is within the noise floor of the measurement.** Recovering int4-body AIME by chasing a single run's miss-set is not well-posed; if the body is to be made quality-safe it needs a genuine body-wide precision increase (toward bf16), which is exactly the speed-costly direction the PR flags.

### Suggested follow-ups

- **Re-baseline int4 AIME as a multi-session band, not a point.** Run the greedy @12288 eval over ≥3 fresh serve sessions (and/or ≥3 batch configs) and report mean ± range; the current 0.400 point is one draw from a ≈±3/60 distribution.
- **Quantify the near-tie margin directly.** I captured per-token runner-up logprob gaps; a histogram of int4 argmax margins on AIME decode steps would put a number on "how near-tie" and predict the flip rate — cheap, uses the streams already on disk.
- **Decide whether the speed-safe ship even needs greedy AIME parity.** Since spec carries no quality cost and the body is the ceiling, the strategic question is now "is a body-wide precision bump worth its TPS cost vs the 126.378 margin?" — the FUNDAMENTAL verdict says a tweak won't do it.
- (Bug-adjacent, not fixed here) `VLLM_BATCH_INVARIANT=1` not delivering batch-invariance for the int4 Marlin path on long decode is worth a focused repro for the serving stack owners; I can file it separately if useful.
