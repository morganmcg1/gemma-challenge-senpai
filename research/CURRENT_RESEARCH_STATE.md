# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13
- **Advisor branch:** `approval-gated-8gpu-20260613` · **Research tag:** `gemma-8gpu-progress-20260613`
- **Most recent human directive:** none yet. Operating under the launch operator rules:
  **no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch`** without a
  human-approved GitHub issue titled `Approval request: HF job for <submission-name>`.
  All student GPU work is LOCAL on the assigned AWS A10G (build, model-load, serve,
  greedy-identity, local PPL, exploratory profiling/TPS). Advisor consumes no GPU.

## Current focus — reproduce the public frontier, locally validated

The first milestone (per `program.md`) is a clean, locally-validated reproduction of the
strongest public **VALID** frontier package — kenyan-duma's
`osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache` at **~421 TPS / PPL 2.377** — built up the
reproduction ladder (bf16 ~44 → int4 QAT ~95 → +g128/lm_head ~127 → +drafter ~285 → ~420).
We push past it only after the base is reproduced and the local validation gates are green.

Single-stream decode is **memory-bandwidth-bound** (~92% weight-GEMM). The live levers are:
1. **Drafter acceptance quality + private-set stability** — THE binding constraint above ~286
   TPS. The verifier's private re-run costs 4–9% TPS on drafters overfit to the 128 public
   prompts; the fix is a drafter trained on a wide, distribution-matched corpus.
2. **Cheaper greedy verification of the 262k-vocab tail** (lmhead12k / PCK04 vocab-prune) and
   **per-step overhead erasure** (fa2sw, onegraph), with hard greedy-identity guards.
3. **Fewer weight-bytes/token** — already at the int4-Marlin Ampere floor; sub-4-bit kernels
   are blocked on vLLM 0.22 + sm_86 (dead end unless someone ships an Ampere kernel).

## Round 1 assignments (all 8 students, local-only)

| student | track | target |
|---|---|---|
| fern | bf16 baseline + **PPL artifact-path resolution** (priority #1) | reproducible local PPL+decode harness; explain missing `ppl_summary.json` |
| stark | **int4 QAT W4A16** reproduction | ~95 TPS / PPL ~2.01, base of the stack |
| lawine | **int4 g128 + untied int4 lm_head** re-quant | ~127 TPS / PPL ~2.02, weight-byte floor |
| kanna | **MTP / QAT-drafter** spec-decode stand-up | ~285 TPS, greedy-identical, acceptance ~3.3 |
| ubel | **vocab-prune / top-k sparse-verify** w/ greedy-identity guard | private-stable verify-cost lever |
| denken | **fa2sw + onegraph** target-side runtime levers | per-step overhead erasure, greedy-identical |
| wirbel | **local validation + profiling infra** | greedy-identity gate, local PPL, decode profiler, one-cmd validate |
| land | **wide-distribution KL-distilled drafter** (PARD option) | acceptance above ~286 + private stability |

## Potential next research directions (round 2+)

- **PARD parallel-draft adaptation** of the E4B assistant (flat acceptance curve → deeper K →
  potentially ~350–500 TPS). Heavy: needs a Gemma-adapted TRL trainer + a wide corpus.
- **EAGLE-3-style draft head** for E4B (none published) — train one.
- **Target-aligned drafter** matched to the *exact served quant* (g128-chanhead, not official g32).
- **Provably-greedy sparse verification** (top-k verify of the 262k tail with a full-vocab
  fallback guard) — legitimate version of the verify-cost lever; check if it's a real win.
- **Investigate the `DECODE_TPS_CAP` PENDING entries** — confirm whether they are a measurement
  artifact (cap gaming) or contain a legitimate throughput-accounting insight.
- Researcher-agent deep-dive in flight → `research/RESEARCH_IDEAS_2026-06-13_round2.md`.

_Living document — prune and update as rounds complete._
