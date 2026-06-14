# TPS Measurement Protocol — closing the #56 noise-floor problem (PR #72)

**Stack under measurement:** `submissions/fa2sw_precache_kenyan` (PR #52, linear MTP K=7 + #43 split-KV), **unchanged**. Local AWS A10G, single GPU, conc=1, 128 prompts × 512 tok, seed 1, decode-only timing. N=12 identical-config runs, fresh server per run.

---

## TL;DR — the noise floor was in the *statistic*, not the hardware

The "±4.4% same-config TPS swing" that #56 surfaced (429.04 vs 448.01) is **not** physical throughput variance. It is an artifact of the statistic the team has been A/B-ing on: `steady_gen_tps_mean`, the **unweighted mean of vLLM's per-interval "Avg generation throughput" log meter**. That estimator is corrupted by (1) the cold first interval (~29% below steady) and (2) PPL-phase intervals leaking into the window. Across the *same 12 runs*:

| metric | mean TPS | std | **CV** | range | what it is |
|---|---|---|---|---|---|
| `steady_gen_tps_mean` (raw interval-mean) | 449.06 | 1.50 | **0.33 %** | 1.19 % | status-quo local A/B metric — fragile |
| **`wall_tps`** = `num_completion_tokens / decode_duration_s` | 454.12 | 0.16 | **0.035 %** | 0.10 % | **= official leaderboard `output_throughput` definition** |
| windowed steady mean (drop first W=3 of 14 intervals) | 459.83 | 0.23 | 0.05 % | 0.17 % | robust interval-meter variant |

`wall_tps` — which is the **same definition the official benchmark scores** (`sglang.bench_serving` `output_throughput` = output tokens / wall duration; `hf_bucket_single_job.py:344`) — has a **CV of 0.035 %**, ~125× tighter than the apparent 4.4 %. The 4.4 % was never throughput noise; it was a broken estimator.

> The team's local A/B metric (`steady_gen_tps_mean`) is **both** off-spec (not the leaderboard metric) **and** fragile. Switching to `wall_tps` fixes both at once — no extra runs required.

---

## RECOMMENDED PROTOCOL (copy-paste)

For every local TPS A/B on this programme:

1. **Metric:** report **`wall_tps = num_completion_tokens / decode_duration_s`** from the official `decode_outputs.py` summary (`duration_s`, `num_completion_tokens` fields). This is the official `output_throughput` definition. **Do not** decide merges on `steady_gen_tps_mean` (the vLLM interval-meter mean).
   - If you must use the interval meter, use the **warmup-discarded windowed mean**: drop the first **W=3** intervals, average the rest. Never include PPL-phase intervals (decode-only timing).
2. **Warmup:** the workload itself supplies warmup — `wall_tps` over the full 128-prompt decode already amortizes the ~1-prompt cold start to <0.04 % CV. (The official harness additionally discards 4 warmup requests; both are robust.)
3. **Statistic:** **median of N runs**, fresh server per run. **N=3** is plenty (N=1 already gives CV 0.04 %; N=3 buys MDE ≈ 0.06 %).
4. **Sequential is fine.** Measured across-run drift is **0.000 tps/run** (SM clock pinned at 1710 MHz, no thermal throttle over 49 min). Interleaving (ABAB) cancels drift, but there is no drift to cancel, so **all-A-then-all-B sequential = interleaved** here. Reserve interleaving only if a future regime shows drift (re-run this harness to check).
5. **Decode-only timing.** Run the PPL validity pass *separately, after* the timed loop — never let `prompt_logprobs` perturb the timing window (this leak is exactly what crashed #56's raw estimator to 410–429).

**Minimum Detectable Effect this buys** (paired A/B, α=0.05, power 0.80):

| N runs / arm | MDE on `wall_tps` (recommended) | MDE on windowed interval-mean |
|---|---|---|
| 1 | **0.10 %** (0.14 % powered) | 0.13 % (0.19 % powered) |
| 2 | 0.07 % | 0.10 % |
| 3 | 0.07 % | 0.09 % |
| 4 | 0.06 % | 0.08 % |
| 8 | 0.05 % | 0.06 % |

**Any A/B delta ≥ 0.2 % is real with N=1; ≥ 0.1 % with N=3.** The historical "all inter-arm deltas ≤1.2 % are below the noise floor" problem (#56) is solved by changing the metric, not by adding runs.

---

## Variance decomposition (PR #72 task 2)

Identical config, fixed seed ⇒ identical workload ⇒ every TPS difference is pure measurement/hardware/FP noise. Attribution:

- **(a) Warmup / cold-start transient — DOMINANT contributor to the raw-estimator noise.** First interval = 326.2 tps vs steady 459.8 → **29.1 % deficit**. At 14 intervals, that one cold interval drags the unweighted mean down ~10 tps *and* makes it fragile to exactly how many intervals get logged. Discarding W≥1 collapses CV from 0.33 % → 0.07 %. This is a per-decode CUDA-graph/cache ramp, **not** server-restart cold start (see below).
- **(b) Steady-state scheduler/timing jitter — small.** Windowed (W=3) across-run CV = 0.05 %; `wall_tps` CV = 0.035 %. This is the true irreducible floor and it is tiny.
- **(c) Thermal / clock drift — ZERO.** SM clock pinned at **1710 MHz** (mean=min=max) across all 12 runs; temp flat ~53 °C (max 59 °C, far below throttle); TPS~wall-time slope = −0.006 tps/run (r=−0.10, n.s.); lag-1 autocorr −0.29. The A10G never throttled.
- **(d) Token-level nondeterminism — negligible.** E[accept] CV = **0.071 %** (range 3.85–3.86); greedy decode acceptance is near-deterministic. Pearson(windowed TPS, E[accept]) = −0.51 but **not significant at n=12** (|r|<0.576 critical) and over a 0.07 %-CV range, so it can drive at most a sliver of the (already 0.05 %) floor. **Token nondeterminism is not a meaningful TPS-noise source here** (kanna characterizes the token-identity side separately on `greedy-determinism`).

**Server-restart (cold-start) contribution:** each of the 12 fresh runs is a full server restart (model load + CUDA-graph capture). Yet `wall_tps` CV across all 12 restarts is **0.035 %** ⇒ **server restart adds no measurable throughput variance** (weights are pre-baked in `/tmp`, graphs recapture deterministically, clock pins immediately). So fresh-per-arm vs warm-server reuse is a *cost* choice, not an accuracy one.

---

## Root cause of the #56 4.4 % (reproduced — `pr56_crosscheck.json`)

| #56 run | `steady_gen_tps_mean` | `wall_tps` |
|---|---|---|
| full_run1 | 429.04 | 454.30 |
| full2_run2 | 448.01 | 454.35 |
| **swing** | **+4.42 %** | **+0.01 %** |

`wall_tps` was identical (+0.01 %); only the unweighted interval-mean swung. Decomposing full2's interval series: clean 14-interval decode → 448.01; but include the trailing PPL-phase intervals (282.1, 6.4) → **410.04**; drop the cold first interval → **458.9**. The whole 4.4 % is estimator fragility (cold-interval inclusion + PPL leak), confirmed against an independent N=12 where `wall_tps` holds to 0.035 % CV.

---

## What this certifies for the active frontier

Land #71 (tree-verify) and stark #70 (int4 drafter) deltas must clear this MDE to count. Under this protocol: **report `wall_tps`, median of N=3 fresh runs; a delta ≥ 0.1 % is real, < 0.1 % is noise.** Sub-5 % gains are now far above the floor — the measurement is no longer the bottleneck.

**Reproduce:**
```bash
.venv/bin/python -m research.tps_noise_floor.run_noise_floor \
    --submission fa2sw_precache_kenyan --mode fresh --n-runs 12 \
    --wandb-name lawine/noise-floor-fresh --wandb-group tps-noise-floor
.venv/bin/python -m research.tps_noise_floor.analyze_noise_floor \
    --inputs research/tps_noise_floor/fresh_n12/noise_floor_fresh.json \
    --out research/tps_noise_floor/analysis.json
```
