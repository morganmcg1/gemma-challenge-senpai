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

Round-2 researcher-agent deep-dive complete → full writeup + dead-end map + decision tree in
`research/RESEARCH_IDEAS_2026-06-13_round2.md`. Five ranked ideas, staged for the next idle slot
(0 idle right now — all 8 on round-1). Ranked:

1. **KL-distilled ≥9k-corpus drafter** (private-stability) — **already in flight as land's #9.**
   Independent confirmation this is the top lever; no new assignment needed.
2. **PARD-2 parallel drafting + CAT** — break the MTP autoregressive chain (acceptance decay
   0.69→0.17 over K). Projected 450–520 TPS if tok/fwd lifts 3.55→5–6. Needs a new drafter +
   TRL/COD trainer. Natural successor to kanna's MTP base (#5) / land's corpus (#9).
3. **EAGLE-3 draft head** (multi-layer feature fusion) — gated on whether vLLM exposes Gemma-4
   intermediate features; degrades to (community-explored) EAGLE-2 if not. Verify export first.
4. **P-EAGLE `parallel_drafting:true` probe** — *near-zero-cost diagnostic*: one serve-flag, one
   local run on the existing drafter to read tok/fwd. **Cheapest next probe** — best as a fast
   follow-up once kanna's drafter (#5) serves, or the first round-2 assignment.
5. **GPU-side SAM-Decoding suffix match** (in-graph, no host round-trip) — exploits 3.6–3.9% of
   tokens in verbatim runs > K=8. High ceiling, heavy Triton/CUDA-graph work; gate on an offline
   token-budget replay before committing.

Pending earlier threads (still open): provably-greedy sparse verification (ubel's #6 is the
round-1 version); the `DECODE_TPS_CAP` PENDING leaderboard entries (confirm cap-gaming vs. real
accounting) — deprioritized vs. the drafter ladder.

_Living document — prune and update as rounds complete._
