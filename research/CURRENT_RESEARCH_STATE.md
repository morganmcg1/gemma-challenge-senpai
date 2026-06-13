# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 6)
- **Advisor branch:** `approval-gated-8gpu-20260613` · **Research tag:** `gemma-8gpu-progress-20260613`
- **Most recent human directive:** none yet. Operating under the launch operator rules:
  **no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch`** without a
  human-approved GitHub issue titled `Approval request: HF job for <submission-name>`.
  All student GPU work is LOCAL on the assigned AWS A10G (build, model-load, serve,
  greedy-identity, local PPL, exploratory profiling/TPS). Advisor consumes no GPU.
- **Current bottleneck (2026-06-13):** two locally-validated ladder rungs are queued behind human
  HF-Job approval — PR #3 int4 (~96 TPS local, issue #11) and PR #4 int4+g128+lmhead (~128 TPS local,
  issue #12). Both are GREEDY_IDENTICAL 128/128 with PPL well under the 2.42 gate locally, and the
  W&B serving runs corroborate. Nothing merges to an official a10g-small number until a human approves
  a job. stark and lawine hold their slots on these approval-blocked PRs (not idle-by-neglect); the
  program's current rate limiter is human approval, not research throughput.
- **LINCHPIN question (2026-06-13, from kanna PR #5) — gates rungs 4–5 / the path to 420:** int4
  batched-verify spec-decode is **structurally greedy-DIVERGENT** in vLLM 0.22.0 — the M=K+1 verify
  forward and the M=1 AR reference flip ~0.33%/token on int4 Marlin near-ties, compounding to
  DIVERGENT over 512 tokens (no batch-invariant knob in 0.22.0; K0-vs-K0 control is identical, so it
  is 100% the spec verify path). The official verifier is **strict bit-exact** (advisor-confirmed,
  zero tolerance) and the organizer reference is plain **M=1 AR of the submitted checkpoint** — so
  this is very likely the official verdict, not a local artifact. If int4+vLLM-spec cannot be
  greedy-valid, the planned drafter ladder (~285 → ~420) is not what we think. **Resolving this is
  now the #1 strategic question**, ahead of further drafter-quality spend. kanna's v1
  precision-localization experiment (int4 vs bf16 vs fp8 greedy flip-rate) + a manifest-vLLM-version
  check decide whether the ladder is salvageable or the team pivots to a non-vLLM-spec mechanism.

## Current focus — reproduce the public frontier, locally validated

The first milestone (per `program.md`) is a clean, locally-validated reproduction of the
strongest public **VALID** frontier package — kenyan-duma's
`osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache` at **~421 TPS / PPL 2.377** — built up the
reproduction ladder (bf16 ~44 → int4 QAT ~95 → +g128/lm_head ~127 → +drafter ~285 → ~420).
We push past it only after the base is reproduced and the local validation gates are green.

Single-stream decode is **memory-bandwidth-bound** (~92% weight-GEMM). The live levers are:
0. **Is int4 spec-decode greedy-valid at all?** (NEW linchpin — see bottleneck above). A drafter is
   worthless if the served int4+spec stack can't pass the strict greedy gate. Must resolve before (1)
   pays off. kanna #5 owns the deciding experiment.
1. **Drafter acceptance quality + private-set stability** — the binding constraint above ~286
   TPS *if (0) resolves favorably*. The verifier's private re-run costs 4–9% TPS on drafters overfit
   to the 128 public prompts; the fix is a drafter trained on a wide, distribution-matched corpus
   (land #9: wide corpus lifts the tf gate +10.3% incl. private-proxy floor +10.9%, but a
   free-running training schedule is needed to convert that to native serving acceptance).
2. **Cheaper greedy verification of the 262k-vocab tail** (lmhead12k / PCK04 vocab-prune) and
   **per-step overhead erasure** (fa2sw, onegraph), with hard greedy-identity guards.
3. **Fewer weight-bytes/token** — already at the int4-Marlin Ampere floor; sub-4-bit kernels
   are blocked on vLLM 0.22 + sm_86 (dead end unless someone ships an Ampere kernel).

## Active assignments

| student | PR | track | target |
|---|---|---|---|
| fern | #16 (WIP) | **EAGLE-3 draft-head training pipeline** — build distillation harness (`gen_eagle3_corpus.py` + `train_eagle3.py` + `eval_eagle3.py`), run debug-viability 1k-step training on 200 MATH samples, report offline teacher-forced acceptance rate. Full-scale run + HF Job gated on (a) debug viability and (b) kanna #5 linchpin. Aux layers `(2,21,39)`, `[T,7680]` fused input. | tf_acceptance_rate_debug_1k ≥ 3.5 tok/step (vs. QAT-MTP baseline ~2.2–3.3) |
| stark | #3 (WIP) | **int4 QAT W4A16** reproduction — local PPL 2.0055 ✓, local TPS ~96 ✓; **awaiting HF Job approval (GitHub issue #11)** | ~95 TPS / PPL ~2.01; after approval: run job, post terminal result, merge |
| lawine | #4 (WIP) | **int4 g128 + untied int4 lm_head** re-quant — local PPL 2.0190 ✓, local TPS ~128 ✓, GREEDY_IDENTICAL 128/128 ✓; **awaiting HF Job approval (GitHub issue #12)** | ~127 TPS / PPL ~2.02, weight-byte floor; after approval: run job, post terminal result, merge |
| kanna | #5 (WIP) | **int4+MTP spec-decode** — `{8,4}` engine blocker SOLVED (vLLM PR #43543 backport), but int4 batched-verify spec is **structurally greedy-DIVERGENT** vs M=1 AR (~0.33%/tok); acceptance caps ~2.2. v1: precision-localization (int4 vs bf16 vs fp8 greedy flip-rate) + confirm whether a10g-small honors manifest vLLM version | **resolve the linchpin**: is int4 spec greedy-valid at all? |
| ubel | #14 (WIP) | **Empirical lmhead12k** — CPU feasibility done (kept_ids.json, implementation complete); **GPU void + int4 base absent** (pod intermittent). Path unblocked: self-build int4+g128 via path-(a) (prune bf16→quantize), general-12,288 corpus cut (hard-include GT + STEM fill), regenerate 128-capture. PPL scorer requires hard-include of GT tokens or gate fails. **DRAFTER-INDEPENDENT** rung. | served TPS/PPL/greedy-identity 128/128; both 7,584 and 12,288 bandwidth numbers |
| denken | #7 (WIP) | **fa2sw + onegraph** target-side runtime levers | per-step overhead erasure, greedy-identical |
| wirbel | #8 (WIP) | **local validation + profiling infra** | greedy-identity gate, local PPL, decode profiler, one-cmd validate |
| land | #9 (WIP) | **wide KL-distilled drafter** — tf-gate **PASS +10.3%** (private-proxy floor +10.9%) but native serving **−4.6%** (train↔serve schedule mismatch + undertrained). v1: free-running / EAGLE-3-style schedule + full ~82-min budget, log eval to W&B | native accept > stock 3.553 (v0 = 3.388) |

## Completed (round 1)

| student | PR | result |
|---|---|---|
| fern | #2 ✓ **MERGED** | **Priority #1 resolved.** Local PPL=2.3012, root cause=40-min HF Job timeout. `scripts/local_prevalidate.py` merged — the team's local pre-validation gate before any HF Job. |
| fern | #10 ✓ **MERGED** | **SAM-Decoding GO verdict.** Causal budget 8.93% K>8 (>3.6% threshold); `analyze_suffix_budget.py` merged; drafter-overlap analysis (#13) merged next. |
| fern | #13 ✓ **MERGED** | **SAM drafter-overlap tooling.** `--drafter-trace` extension + 13/13 mock tests; canonical trace format (`output_start`); template JSON with thresholds. Net-headroom number awaits kanna's acceptance trace. |
| fern | #15 ✓ **MERGED** | **EAGLE-3 feasibility ACCESSIBLE → GO.** `SupportsEagle3` natively in vLLM 0.22.0 + Gemma-4 E4B; aux layers `(2,21,39)`, `[T,2560]` bf16, CUDA-graph safe; drafter head arch exists (`llama_eagle3.py`). Zero patching. Empirical probe confirmed. Report at `research/eagle3_feasibility/`. |
| ubel | #6 ✗ **CLOSED** (negative) | **Cert dead end.** Cauchy-Schwarz cert 0%-fire on Gemma4 (R_complement=1.630, geometry obstruction). Empirical lmhead12k authorized (PR #14). |

## Potential next research directions (round 2+)

Round-2 researcher-agent deep-dive complete → full writeup + dead-end map + decision tree in
`research/RESEARCH_IDEAS_2026-06-13_round2.md`. Five ranked ideas. Rank 5 now in flight with fern
(PR #10). Remaining staged for the next idle slot. Ranked:

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
5. **GPU-side SAM-Decoding suffix match** (in-graph, no host round-trip) — **PR #10 ✓ MERGED
   (GO verdict, causal 8.93% K>8). PR #13 in flight (drafter-overlap analysis, step 2).** Triton
   kernel follow-up gated on net_frac > 3% from PR #13 (requires kanna's #5 drafter trace).

**EAGLE-3 training in flight (fern PR #16):** `fern/eagle3-training-pipeline` — debug-viability
1k-step run, then full-scale once debug confirms + kanna linchpin clears. Rank 3 projected ceiling
480-550 TPS; highest-ceiling drafter in the pipeline. Serving validity gated on kanna #5.

Pending earlier threads (still open): provably-greedy sparse verification (ubel's #6 is the
round-1 version); the `DECODE_TPS_CAP` PENDING leaderboard entries (confirm cap-gaming vs. real
accounting) — deprioritized vs. the drafter ladder.

_Living document — prune and update as rounds complete._
