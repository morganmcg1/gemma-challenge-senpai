# BASELINE — Fast Gemma Challenge (advisor branch `approval-gated-8gpu-20260613`)

Primary metric: **`summary.json:tps` (output-token throughput, higher is better)**, measured
single-stream (max concurrency 1), output_len 512, on the fixed 128 public prompts,
on **a10g-small** via HF Jobs. Local AWS A10G numbers are **exploratory only**.

Validity gates (a submission is invalid if any fail):
- **PPL ≤ ~2.42** (reference 2.30 + 5%).
- **128/128** prompts completed.
- **Greedy decode token-identical** to plain greedy AR decode of the *submitted* checkpoint.
- **All modalities loaded** (text/image/audio) — no text-only shortcut.

## Public frontier target (what we are reproducing, then beating)

Top **VALID** leaderboard entry as of 2026-06-13:
- **kenyan-duma `osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache` — 421.12 TPS / PPL 2.3774, 128/128** (job `6a2c7688871c005b5352b87a`).
- Other VALID repros at ~420.6–420.8 (frantic-penguin `fa2sw-fp`, agent-smith `fa2sw-v3`).
- The 3 entries above it (446–449 TPS: `ff-lf29cap432`, `mao-gemma-fast-cap433`, `pupa-lf29cap-repro`, all using a `DECODE_TPS_CAP`) are **PENDING / unverified** and look like decode-TPS-cap gaming — **not** our target. We target legitimately reproducing and beating the ~420 VALID frontier.

## The climb (intermediate milestones — our reproduction ladder)

| milestone | TPS (a10g-small) | PPL | lever |
|---|---|---|---|
| bf16 stock (`vllm_baseline`) | ~44.0 | ~2.30 | none (reference) |
| int4 QAT W4A16 (Marlin), as-is | ~95.4 | ~2.01 | 4× less weight bandwidth (dominant lever) |
| + untied int4 lm_head + full-body g128 | ~126.8 | ~2.02 | int4-Marlin **weight-byte floor** on Ampere |
| + MTP / QAT-drafter spec decode (K≈6) | ~273–286 | ~2.0–2.4 | amortize weight read over ~3.3 accepted tok/step |
| + lmhead12k sparse-verify + fa2sw + onegraph + precache | **~420** | **~2.377** | verify-cost + runtime + warmup levers (the frontier) |

Decode at conc=1 is **memory-bandwidth-bound** (profiler: ~92% weight-GEMM, attn ~2.6%, sampling ~0.2%).
Levers: (a) fewer weight-bytes/token, (b) more accepted tokens per weight read (better drafter),
(c) erase per-step overhead / cheaper 262k-vocab verification.

## Key risk for any near-frontier submission
The verifier re-runs on a **private** prompt set; top drafter stacks lose **4–9% TPS** on it
(prompt-distribution shift). Submissions die on the **5% TPS-reproduction gap, not on PPL**.
Private-stable acceptance (drafter trained on a wide distribution; prompt-content-invariant
verify paths) is a first-class objective, not an afterthought.

## Current local baseline in this repo
- `submissions/vllm_baseline` — bf16 stock vLLM 0.22.0 endpoint. Prior HF smoke job
  `6a2c5fb77c68f455eff14260` (run prefix `results/senpai/vllm-baseline-20260612T193622Z`)
  reported **tps=44.018, completed=128** on a10g-small.
- **PPL-artifact resolution (priority #1, fern, PR #2) — RESOLVED 2026-06-13.**
  - Local PPL **confirmed 2.3012** over all 128 GT records (61,797 scored tokens) via the
    official `ppl_endpoint.py` against a local bf16 `serve.py` endpoint — within the ≤2.42 gate.
  - The prior job's missing `ppl_summary.json` was **not** disabled / OOM / unfetched: it was the
    **40-min HF Job wall-clock timeout**. Evidence (`job_status.json` timed_out@40m stage=RUNNING,
    `run_environment.json` ppl.enabled=true, `summary.json` duration_s=1488.8s) shows 11.9-min cold
    startup + 24.8-min benchmark left only ~6.5 min, so decode-capture (another ~24.8-min workload)
    and the PPL stage that runs *after* it never completed. At 44 TPS the bf16 baseline cannot fit
    startup+benchmark+decode+PPL inside 40 min; faster submissions will.
  - Reusable one-command local pre-validation harness: **`scripts/local_prevalidate.py`** (serves
    bf16, runs PPL + decode capture, prints `tps`/`ppl`/`completed`, no HF Jobs quota). Evidence
    artifacts under `research/local_validation/`.

## Merge history

### 2026-06-13 08:40 — PR #2: Resolve PPL artifact path + validate bf16 baseline locally

- **Priority #1 resolved.** Root cause: 40-min HF Job wall-clock timeout (not OOM / disabled / unfetched).
- **Local PPL:** 2.3012 (128/128 GT records; within ≤2.42 gate) ✓
- **Local TPS (exploratory, A10G):** ~44.01 (16-prompt sample — not official a10g-small)
- **W&B run:** none (local validation + infra, no training)
- **New shared infra:** `scripts/local_prevalidate.py` — one-command local pre-validation for all future submissions.
- **Reproduce:** `cd target/ && VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/local_prevalidate.py --submission submissions/vllm_baseline --decode-num-prompts 16`
  (Env-var is a local-box workaround for broken FlashInfer JIT; not needed on official a10g-small image.)

### 2026-06-13 09:45 — PR #10: Offline suffix-run token-budget analysis for SAM-Decoding feasibility

- **Finding (GO on causal budget):** causal SAM-Decoding realized budget = **8.93%** free tokens at K>8 (K>4: 15.4%, K>6: 11.6%); clear **GO** for the Triton-kernel follow-up (threshold >3.6%). Robust across datasets (aime 10.74%, gpqa 9.23%, mmlu_pro 8.19%); greedy-safe by construction (zero PPL risk).
- **PR-spec proxy (`m(t)`):** 1.21% — *not* the decision metric. `m(t)` fires only on adjacent-period repetition; the exploitable structure is non-adjacent. Causal estimate cross-validated against brute-force O(n²) reference: 0 mismatches / 600 positions.
- **Caveat:** gain is *incremental* over the existing MTP/QAT drafter (~3.3 tok/step). Net headroom requires per-step acceptance trace from kanna's #5 — measuring SAM-drafter overlap de-risks the Triton build before GPU spend.
- **New shared infra:** `scripts/analyze_suffix_budget.py` — offline CPU-only suffix-budget analyzer; designed for extension to ingest a drafter acceptance trace for overlap quantification.
- **W&B run:** none (CPU-only offline analysis). 128/128 prompts captured (bf16, 43.94 TPS local).
- **Reproduce:** `cd target/ && python scripts/analyze_suffix_budget.py --input research/local_validation/vllm_baseline/decode_outputs_128.jsonl --output research/local_validation/suffix_budget/suffix_budget_analysis.json`

## Confirmed dead ends (do not re-spend on these)
sub-4-bit weight kernels (AWQ/GPTQ/AQLM/QuIP#/2:4-Sparse-Marlin/NVFP4) — no loadable Ampere
sm_86 kernel in vLLM 0.22; fp8 KV cache — rejected by A10G + Gemma4 attn; n-gram/prompt-lookup
spec decode — loses at conc=1; runtime knobs (attn-backend swap, max_num_seqs, MARLIN_USE_ATOMIC_ADD) —
parity/noise; body channel-wise quant — trades PPL for no TPS; widening draft centroid top_k — no gain;
**provable greedy-safe cert (Cauchy-Schwarz) for sparse lm_head verify on gemma-4-E4B** — model-intrinsic
geometry obstruction (flat row norms, near-full-rank embedding: R_complement_max_norm=1.630 >> z_max/||h||≈0.59),
0%-fire on 16,384 real decode steps, nets −8% TPS (cert + full fallback > full alone); harness on
`ubel/vocab-prune-sparse-verify` branch; empirical pruned-weights lmhead12k (no cert) is the viable lever.

_Last updated: 2026-06-13 (PR #10 merged — SAM-Decoding offline budget analysis: causal realized budget 8.93% at K>8, GO verdict, drafter-overlap quantification staged as next de-risking step)._
