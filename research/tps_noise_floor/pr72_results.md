STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["n07jrhxl"],"primary_metric":{"name":"tps_noise_floor_cv","value":0.035},"test_metric":{"name":"tps_mde_pct_wall_paired_n1","value":0.095}}

## Results — TPS measurement protocol: the ±4.4% was an estimator artifact, not throughput noise

**Headline:** Running the deployed `fa2sw_precache_kenyan` stack **unchanged, N=12 times** (128×512, conc=1, seed 1, fresh server per run, decode-only) shows the #56 "±4.4% same-config noise" is **not physical throughput variance** — it is an artifact of the statistic the team A/Bs on (`steady_gen_tps_mean`, the unweighted mean of vLLM's per-interval throughput meter). The robust metric, **`wall_tps`, has CV 0.035%** — ~125× tighter — and it is *also the official leaderboard metric definition*.

### Noise-floor table (N=12 identical-config fresh runs)

| metric | mean TPS | std | **CV** | range | what it is |
|---|---|---|---|---|---|
| `steady_gen_tps_mean` (raw interval-mean) | 449.06 | 1.50 | **0.33%** | 1.19% | status-quo local A/B metric — **fragile** |
| **`wall_tps` = completion_tokens / decode_duration_s** | 454.12 | 0.16 | **0.035%** | 0.10% | **= official `output_throughput` defn** |
| windowed steady (drop first W=3 of 14 intervals) | 459.83 | 0.23 | 0.05% | 0.17% | robust interval-meter variant |
| `e_accept_exact` | 3.855 | 0.003 | 0.07% | 0.24% | acceptance length (greedy, near-deterministic) |

All 12 runs completed **128/128** prompts, 65536 completion tokens each. Total wall 49.4 min.

### Why the 4.4% was an artifact (reproduced #56 — `pr56_crosscheck.json`)

| #56 run | `steady_gen_tps_mean` | `wall_tps` |
|---|---|---|
| full_run1 | 429.04 | 454.30 |
| full2_run2 | 448.01 | 454.35 |
| **swing** | **+4.42%** | **+0.01%** |

`wall_tps` was identical (+0.01%); only the unweighted interval-mean swung. The corruptors: (1) the **cold first interval** (326 vs 460 steady = **29% deficit**) drags the unweighted mean and makes it fragile to interval count; (2) **PPL-phase intervals** (282, 6.4 tps) leaking into the window crash it to 410. Note the historical "428.37" headline for this stack is itself a low point-estimate of this fragile metric — my 12 clean runs average **449** on the same metric, while `wall_tps` holds at **454** in both #56 and here. The throughput never moved; the estimator did.

### Variance decomposition (task 2)

- **(a) warmup/cold-start — dominant** for the raw estimator. First interval 29% below steady; dropping W≥1 collapses raw CV 0.33%→0.07%. (Per-decode CUDA-graph/cache ramp — *not* server-restart: `wall_tps` CV across 12 full restarts is 0.035%, so restart adds no measurable variance.)
- **(b) steady jitter — small.** windowed CV 0.05%, wall CV 0.035%. The true irreducible floor.
- **(c) thermal/clock drift — ZERO.** A10G SM clock pinned **1710 MHz** all session; temp flat ~53°C (max 59, no throttle); TPS~time slope −0.006 tps/run (r=−0.10, n.s.).
- **(d) token nondeterminism — negligible.** E[accept] CV 0.07%; Pearson(TPS, E[accept]) = −0.51 but **not significant at n=12** (|r|<0.576 crit). Greedy acceptance is near-deterministic — not a meaningful TPS-noise source. (kanna characterizes token-identity separately on `greedy-determinism`; shared N-run harness.)

### Recommended protocol + MDE (task 3/4)

**Decide every local TPS A/B on `wall_tps` (= `num_completion_tokens / decode_duration_s`, the official `output_throughput` definition), median of N=3 fresh runs, decode-only (PPL separate).** Drift is 0.000 tps/run, so sequential = interleaved (no drift to cancel). MDE (paired, α=0.05, power 0.80):

| N / arm | MDE `wall_tps` (rec.) | MDE windowed interval-mean |
|---|---|---|
| 1 | **0.095%** (0.136% powered) | 0.13% (0.19%) |
| 3 | 0.07% | 0.09% |
| 4 | 0.06% | 0.08% |

**Any delta ≥0.2% is real at N=1; ≥0.1% at N=3.** The #56 problem ("inter-arm deltas ≤1.2% all below the floor") is solved by **changing the metric**, not by adding runs. Full copy-paste protocol: `research/tps_noise_floor/PROTOCOL.md`.

### Validity (unchanged-stack check)

Zero served-file changes — the measured stack is byte-identical to deployed PR #52. PPL validity pass (decode-only timing kept separate, run after the loop): **ppl = 2.3767** (128/128 records; PR baseline 2.3767 local / 2.3772 official — exact match, greedy identity preserved).

### Run facts

- **summary.json fields:** tps (wall) = 454.12 mean (steady-meter 449.06); ppl = 2.3767; completed = 128/128 ×12; run_prefix = N/A (local research harness, no HF job); no failures.
- **Exact command:**
  ```bash
  .venv/bin/python -m research.tps_noise_floor.run_noise_floor \
      --submission fa2sw_precache_kenyan --mode fresh --n-runs 12 --wandb-group tps-noise-floor
  .venv/bin/python -m research.tps_noise_floor.analyze_noise_floor \
      --inputs research/tps_noise_floor/fresh_n12/noise_floor_fresh.json --out research/tps_noise_floor/analysis.json
  ```
- **Peak GPU memory:** 21395 MiB (20.9 GiB), captured across the memory-tight decode+PPL phase (serve `GPU_MEMORY_UTILIZATION=0.90` on the 23 GiB A10G ≈ 20.7 GiB reserved cap).
- **W&B:** run `n07jrhxl`, group `tps-noise-floor` — per-run series, warmup/MDE/per-run tables, analysis + protocol artifacts. Primary `tps_noise_floor_cv = 0.0354%`.

### What happened — honest analysis

The hypothesis ("there's a ±4.4% noise floor larger than our deltas") is **half right and high-leverage**: the *swing* is real in the *reported number*, but its cause is the **estimator, not the hardware or the algorithm**. The team has been making merge decisions on `steady_gen_tps_mean` — an unweighted mean over vLLM's per-interval meter — which is (a) **not** the metric the leaderboard scores (official `tps` = `sglang.bench_serving` `output_throughput` = wall-based output tokens/sec, conc=1, 4 warmup reqs discarded; `hf_bucket_single_job.py:344`) and (b) fragile to cold-interval inclusion and PPL leak. Switching the local A/B metric to `wall_tps` (which *is* the official definition) drops the floor to CV 0.035% / MDE 0.10% **at no extra cost** — no warmup tuning, no n-run averaging required. The deeper lesson: our local proxy diverged from the thing we actually optimize. This is pure infra leverage — every future TPS A/B (incl. land #71 tree-verify, stark #70 int4 drafter) now has a defensible 0.1% MDE instead of a phantom 4.4% floor.

### Suggested follow-ups

1. **Adopt `wall_tps` as the local A/B metric of record** in `serve_profile` / the bench reporting path so screens report the official definition by default (keep `steady_gen_tps_mean` as a secondary diagnostic only).
2. **Re-score recent sub-1.2% "noise" screens** (e.g. the #56 mbt sweep) on `wall_tps` — some "below the floor" deltas may now be resolvable.
3. **Reuse mode is implemented** (`--mode reuse`, warm-server back-to-back) but I did **not** run the full N=12: its raw-steady metric is *more* artifact-prone (slice-boundary idle ticks → 392–410, smoke-confirmed) — itself further evidence for "use `wall_tps`." A future tidy-up could fix per-slice E[accept] parsing if a warm-server interleaved harness is ever wanted (not needed for this deliverable).
4. **Drift was zero on this A10G/session** — if a future regime shows thermal throttling (e.g. longer sweeps, hotter pod), re-run this harness; the interleaved-MDE path is already built for it.
