<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Draft-verify overlap gate — can the 15.5% drafter be hidden behind verify on a bandwidth-bound A10G? (#94)

**PR:** #94 · **Author:** denken · **Date:** 2026-06-14 · **Type:** LOCAL profiling/analysis
gate (no HF Job) · **Builds on:** #75 (drafter-forward roofline = 15.46% of the decode step,
~47% HBM), #68 (verify-GEMM roofline = 77% HBM at M=8), #85 (M=8→M=32 tree non-GEMM
amortization), frontier_decode (#69/#30: measured drafter/verify STEPTIME + decode
composition), accept_calibration (E[T]=3.844, K=7)

**Question:** Saguaro-style parallel speculative decoding (arXiv:2603.03251) enqueues the
drafter forward for decode-step *N+1* onto a **secondary CUDA stream** so it runs concurrently
with the verify forward for step *N* on the primary stream — hiding the ~15.5% drafter cost
"for free." On a single A10G at conc=1, verify is memory-bandwidth-bound (verify GEMM 77% HBM,
#68) and so is the drafter (~47% HBM, #75/#77). **Both streams share one A10G HBM bus.** Does
a drafter-sized memory-bound forward actually HIDE behind a verify-sized one, or do the two
streams contend on the bus and serialize? Report the bandwidth-limited overlap ceiling vs the
naive compute-limited ceiling, and gate the idea.

## Verdict / headline

> **AMBER → recommend CLOSE. The naive "+18% for free" collapses to a *measured* +4.2% wall /
> +4.4% TPS ceiling, and ~+1.2–2.9% realized after the serial accept-boundary haircut.** The
> timing gate is wide open — `r = drafter/verify = 0.183 ≪ 0.85` (M=8) and `0.178` (M=32), so
> the drafter is cheap relative to verify and a *compute-limited* world would hide it almost
> fully (naive ceiling +15.46% wall / +18.29% TPS). **But the binding constraint is the HBM
> bus, not compute.** A direct two-stream probe on this A10G shows two memory-bound streams
> **fully serialize**: symmetric dual-verify speedup **1.01×** (2.0 = perfect overlap),
> bus-contention factor **0.506** (0.5 = full serialization), combined bandwidth **498 GB/s ≈
> single-stream 491 GB/s** — *not* the 982 GB/s additive overlap would require. The drafter
> claws back only **27.3%** of itself (the ~18% bus headroom left under peak), so the
> bandwidth-limited overlap ceiling is `0.273 × 15.46% = 4.22%` wall. Add the serial
> accept-boundary dependency (drafter(N+1) needs verify(N)'s accept set → must *speculate* the
> continuation; the single most-likely outcome is **zero** accepted tokens at P=0.271, best
> single-path hit 0.271, 2-path tree 0.484) and the realized TPS gain is **+1.16% / +2.09% /
> +2.86%** for 1/2/3-path continuation trees. **Not worth the dual-stream scheduler +
> speculative-drafter + rollback machinery for a sub-3% realized gain on a single
> bandwidth-saturated A10G.** Saguaro's gains are *separate-device* (the drafter runs on a
> second GPU with its own HBM bus); that premise does not hold for our single-A10G conc=1
> deployment.

| metric | naive (compute-limited) | **measured (bandwidth-limited)** | realized (after accept haircut) |
|---|---|---|---|
| overlap ceiling, wall % | 15.46% | **4.22%** | — |
| overlap ceiling, TPS % | 18.29% | **4.41%** | +1.16% (1-path) … +2.86% (3-path) |
| official TPS projection | 569.6 | 502.8 | 491.6 (2-path) |

Primary metric `bandwidth_limited_overlap_ceiling_pct = 4.22` (AMBER band 1–5%). Test metric
`drafter_verify_step_time_ratio = 0.183`.

---

## Step 1 — timing gate (r = drafter_step_time / verify_step_time)

From the merged frontier_decode profile (conc=1, the deployed M=8 frontier, #69/#30):

| | drafter | verify | gpu-busy |
|---|---|---|---|
| step time (ms, p50) | 1.446 | 7.906 | 9.352 |
| fraction of gpu-busy | 15.46% | 81.54% (+3.0% sampling) | 100% |

**`r_M8 = 1.446 / 7.906 = 0.183`.** PR primary gate: `r > 0.85 → CLOSE immediately`. We are far
below it — the drafter is ~5.5× cheaper than verify, so in a *compute-limited* world it would
hide almost entirely behind verify. The naive compute-limited ceiling (PR formula
`min(drafter,verify)/total`):

- **M=8:** hide = min(1.446, 7.906) = 1.446 → **15.46% wall / 18.29% TPS** (`9.352/7.906 − 1`).
  This reproduces the PR's projected "+18%" exactly, validating the byte/step accounting.

**M=32 tree-verify projection** (land #71 builds the M=32 tree). Drafter grows by the M-row
candidate sampler (#85: centroid_sampler 85.18→241.97 µs, +0.157 ms → drafter 1.603 ms);
verify grows by GEMM (×1.184, #68) with attention amortized ×1.06 (#85, split-KV reads the
shared prefix once) + M-row argmax (+0.037 ms) → verify 9.005 ms. **`r_M32 = 1.603 / 9.005 =
0.178`** — still ≪ 0.85, naive ceiling **15.11% wall / 17.80% TPS**. The timing gate does not
close the idea; the question is entirely whether the bus lets the overlap happen.

## Step 2 — HBM-bandwidth contention (THE crux)

### 2a. Byte-fit test (necessary, NOT sufficient)

Does `(drafter_bytes + verify_bytes) / HBM_bandwidth` fit inside `verify_step_time`?

- verify bytes/step = 2.295 GB (GEMM 2.251 GB from #68 + ~0.044 GB KV/lm_head/act)
- drafter bytes/step = 0.171 GB (#75 chain 0.160 GB + ~0.010 GB embed/act)
- combined = 2.466 GB → **bandwidth floor = 2.466 GB / 600 GB/s = 4.11 ms ≤ 7.906 ms verify.**
  Byte-fit **PASSES** with 3.80 ms margin; the featherweight drafter "fits" the time-averaged
  non-GEMM slack ~10× over (verify averages only 290 GB/s = 48.4% HBM across the whole step).

**This is an averaging artifact and necessary-not-sufficient.** Verify's 48% time-average hides
a bimodal bus profile: the **GEMM core (66% of verify wall) runs at 77% HBM** (#68), while the
brief non-GEMM windows (attention/norm/lm_head, 34% of wall) are bus-idle. A naive secondary
stream **cannot pin** the drafter into those short idle windows — its kernels land wherever the
scheduler puts them, overwhelmingly on top of the saturated GEMM core. So the question is not
"do the bytes fit on average" but "what happens when two memory-bound streams actually
co-run." We measured it.

### 2b. Direct A10G dual-stream probe (`dual_stream_hbm_contention.py`)

A faithful skinny-GEMM proxy at M=8 (the deployed verify width: few query rows → SM-light, but
streams a big weight once → bandwidth-bound, exactly the decode regime). "verify" GEMM ≈ 2.25
GB weight, "drafter" GEMM ≈ 0.16 GB, run solo vs concurrent on two `torch.cuda.Stream`s:

| measurement | ms | interpretation |
|---|---|---|
| verify solo | 4.583 | 491 GB/s = **82% of HBM peak** (one stream nearly saturates the bus) |
| drafter solo | 0.321 | — |
| **two verify (concurrent)** | **9.035** | = 1.97× one verify → **fully serialized** |
| **verify + drafter (concurrent)** | **4.817** | drafter adds only 0.234 ms of its 0.321 ms |

Derived contention metrics:

- **`symmetric_overlap_speedup = (4.583+4.556)/9.035 = 1.01×`** (2.0 = perfect overlap, 1.0 =
  full serialization). Two identical memory-bound streams get **no overlap at all** on this
  bus.
- **`bus_contention_factor = two_verify_BW / (solo_BW + solo_BW) = 498.1 / 984.8 = 0.506`**
  (1.0 = additive bandwidth, 0.5 = serialized). Combined achieved bandwidth is **498 GB/s ≈
  single-stream**, *not* the ~982 GB/s additive overlap would need. **The A10G HBM bus is
  saturated by one verify-sized stream; there is no second-stream bandwidth to give.**
- **`drafter_overlap_efficiency = (4.583+0.321−4.817)/0.321 = 0.273`.** Only **27.3%** of the
  drafter hides — exactly the ~18% headroom under peak (82% used) plus a little kernel-launch
  slack. 72.7% of the drafter still costs full wall time.

### 2c. Bandwidth-limited overlap ceiling

Scaling the naive ceiling by the measured fraction of the drafter the bus actually lets hide:

> **`bandwidth_limited_overlap_ceiling_pct = 0.273 × 15.46% = 4.22%` wall / `4.41%` TPS.**

This is the PRIMARY METRIC, and it is **measured, not estimated**. The PR's own framing —
"if verify saturates HBM → overlap saves ~0; if verify has slack → overlap reclaims it" — is
answered: verify *does* saturate the bus (82% solo, no additive second-stream bandwidth), so
overlap reclaims only the thin 27% headroom, not the naive 15.5%.

## Step 3 — serial accept-boundary realizability haircut

Even the 4.22% is optimistic, because the overlap is not unconditional. drafter(*N+1*)'s input
is the *accepted prefix* from verify(*N*) — which is not known until verify(*N*) finishes. To
overlap, you must **speculate** which continuation verify(*N*) will accept and draft from it; on
a wrong guess the drafter work is discarded and redone serially (no benefit). The measured
accept-boundary distribution (accept_calibration, K=7) is:

| accepted draft tokens *j* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| P(boundary = *j*) | **0.271** | 0.176 | 0.115 | 0.078 | 0.060 | 0.050 | 0.038 | **0.213** |

The single most-likely outcome is **zero accepted tokens (P=0.271)** — the verify rejects the
whole draft 27% of the time. Best continuation-tree hit rates (cover the top-*p* boundaries):

- single-path: 0.271 · 2-path tree: 0.484 · 3-path tree: 0.660

Composing both haircuts (overlap only when the guess hits AND only 27.3% of the drafter hides
on a hit), realized TPS gain:

| continuation scheme | hit rate | **realized TPS %** | official TPS |
|---|---|---|---|
| single-path | 0.271 | **+1.16%** | 487.1 |
| 2-path tree | 0.484 | **+2.09%** | 491.6 |
| 3-path tree | 0.660 | **+2.86%** | 495.3 |

## Step 3 — gate verdict

| gate input | value | threshold | result |
|---|---|---|---|
| `drafter_verify_step_time_ratio` (r, M=8) | 0.183 | CLOSE if > 0.85 | open |
| `bandwidth_limited_overlap_ceiling_pct` | **4.22%** | GREEN ≥5 / AMBER 1–5 / RED <1 | **AMBER** |
| realized (1-path) | +1.16% | — | low-AMBER |

**AMBER, recommend CLOSE.** The idea is not broken — it would deliver a small positive gain —
but it does not clear the bar that would justify building it:

1. **The bus is the wall, and we don't own a second one.** Saguaro/AMUSD's "free drafter" works
   because the drafter runs on a **separate device** with its own HBM. On our single A10G the
   drafter and verify share one saturated bus; the probe shows two memory-bound streams
   serialize (1.01× symmetric speedup). The core premise of the technique does not transfer.
2. **The headline collapses 4×.** +15.46% naive → +4.22% bandwidth-limited (measured) → +1.2–
   2.9% realized after the accept-boundary haircut. A sub-3% realized gain does not justify a
   dual-stream scheduler + a speculative continuation tree + a rollback/re-draft path on every
   mispredicted boundary.
3. **It fights the strongest base.** The gain shrinks at M=32 (`r` 0.183→0.178) and the
   non-GEMM slack the drafter would need is exactly what #85's tree machinery is starting to
   consume.

If revisited, the only regime where this pays is a **second GPU** (true AMUSD separate-device
overlap, no shared bus) — which is a different deployment, not a conc=1 single-A10G change.

## Reproduce

```bash
# (1) measured dual-stream HBM-contention probe on the A10G (the crux)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/profiler/dual_stream_hbm_contention.py
# (2) compose the gate (pure CPU; reads the probe + merged roofline/accept JSONs)
.venv/bin/python scripts/profiler/draft_verify_overlap_gate.py            # + W&B
.venv/bin/python scripts/profiler/draft_verify_overlap_gate.py --no-wandb # offline
```

Artifacts: `research/draft_verify_overlap/dual_stream_contention.json` (probe),
`research/draft_verify_overlap/overlap_gate.json` (gate). W&B: group
`draft-verify-overlap-gate`, run `denken/draft-verify-overlap-gate` (1127zef4). No HF Job, no
submission — this is a local analysis gate; greedy token-identity is preserved by construction
(overlap only reorders the GPU-timeline execution of work that is already done serially).
