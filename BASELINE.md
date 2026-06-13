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
  reported **tps=44.018, completed=128**, but **`ppl_summary.json` was not confirmed** —
  PPL-artifact resolution is research priority #1 (assigned to fern).

## Confirmed dead ends (do not re-spend on these)
sub-4-bit weight kernels (AWQ/GPTQ/AQLM/QuIP#/2:4-Sparse-Marlin/NVFP4) — no loadable Ampere
sm_86 kernel in vLLM 0.22; fp8 KV cache — rejected by A10G + Gemma4 attn; n-gram/prompt-lookup
spec decode — loses at conc=1; runtime knobs (attn-backend swap, max_num_seqs, MARLIN_USE_ATOMIC_ADD) —
parity/noise; body channel-wise quant — trades PPL for no TPS; widening draft centroid top_k — no gain.

_Last updated: 2026-06-13 (initial frontier reproduction track)._
