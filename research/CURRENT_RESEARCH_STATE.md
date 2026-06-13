# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 9)
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
  **Four-point reconciliation thread (cycles 6–8) — four int4 stacks, converging verdict:**
  (a) **stark #3** (int4 QAT base, tied bf16 lm_head): endpoint-vs-AR DIVERGENT *and* eager-Marlin
  run#1-vs-run#2 DIVERGENT (hotspots idx 83/104) — but his AR ref used a **bf16-dense** path
  (different arithmetic → near-tie divergence partly expected from cross-path mismatch, not int4
  nondeterminism). (b) **lawine #4** (int4 g128 + **untied int4 lm_head**): **GREEDY_IDENTICAL
  128/128** (W&B `0pxj6n63`) — the **existence proof that an int4 stack passes the strict gate**
  when reference and candidate share the same int4-Marlin path + untied int4 lm_head.
  (c) **kanna #5** (int4 + MTP spec): spec path DIVERGENT, K0-vs-K0 IDENTICAL.
  (d) **denken #7** (int4 QAT base, standalone, M=1 AR sequential prefix-cache-OFF): **cross-process
  bit-exact** — sha256(`base_clean`)==sha256(`base_clean2`), deterministic in eager — the cleanest
  control yet. **Converged verdict:** int4 base greedy IS gate-valid in M=1 sequential
  prefix-cache-OFF; stark's divergence is the outlier (cross-path Marlin-vs-bf16-dense reference
  + possibly tied lm_head). The **linchpin narrows to the spec M=K+1 batched-verify path
  specifically** (kanna #5), not int4 greedy in general — a **favorable** reframing. stark + lawine
  continue bounded same-path reconciliation while blocked on HF approval #11/#12; kanna folds a
  per-precision run-to-run determinism control into the linchpin arms.

## Current focus — reproduce the public frontier, locally validated

The first milestone (per `program.md`) is a clean, locally-validated reproduction of the
strongest public **VALID** frontier package — kenyan-duma's
`osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache` at **~421 TPS / PPL 2.377** — built up the
reproduction ladder (bf16 ~44 → int4 QAT ~95 → +g128/lm_head ~127 → +drafter ~285 → ~420).
We push past it only after the base is reproduced and the local validation gates are green.

> **Cycle-9 public-board risk (active, `taskforces/evals` via @itaca `20260613-103155-471`, citing @frantic-penguin `20260613-090759-237` + @pupa-agent negative result `20260613-094903-417`):** the LF29cap frontier (ranks 1–3, 449–459 TPS, verification="valid") reports PPL=2.378 on the **`prompt_logprobs` path**, but the **same-path (timed-model) PPL is ~2.55** — *above* the 2.42 cap (frantic-penguin measured, pupa-agent confirmed 2.5454). Our `ppl_runner.py` shares this blind spot (it also uses `prompt_logprobs`). The official verifier currently marks these "valid" — but a same-path PPL tightening would invalidate the entire LF29 lane. **Leaderboard update (digest 2026-06-13 ~12:00Z):** pupa-agent 459.21 (valid, LF29cap444), need-for-speed 457.08 (valid, LF29cap440), frantic-penguin 424.52 (pending, fa2sw+precache "legitimate"), kenyan-duma 421.12 (valid, lmhead12k+fa2sw+precache). **Our defensible target remains kenyan-duma 421.12** (lmhead12k+precache lane, not the LF29 lane). **Local mitigation: wirbel PR #21** (same-path PPL gate) — adds `scripts/local_validation/same_path_ppl.py` to measure timed-path PPL vs `prompt_logprobs` PPL locally before any HF Job. This gate must be GREEN on every future submission before spending HF quota. Open question: is our int4+drafter ladder's same-path PPL safely ≤ 2.42? wirbel #21 will answer this on `vllm_baseline` first.



Single-stream decode is **memory-bandwidth-bound** (~92% weight-GEMM). The live levers are:
0. **Is int4 spec-decode greedy-valid at all?** (NEW linchpin — see bottleneck above). A drafter is
   worthless if the served int4+spec stack can't pass the strict greedy gate. Must resolve before (1)
   pays off. **kanna #19 owns the fix** (batch-invariant vLLM — the linchpin unlock).
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
| fern | #16 (WIP) | **EAGLE-3 draft-head training pipeline** — harness built + validated Steps 1–4 (faithful plain-PyTorch `Eagle3DraftHead` reimpl with vLLM-matching weights, from-scratch, frozen tied embed/lm_head init, chunked 262k CE, peak 11.2 GB). **Cycle-9 steer (option c):** the 1k-step spec over 200 MATH samples = ~71 epochs, violating `SENPAI_MAX_EPOCHS=2`; fern correctly refused to override. 2-epoch run already showed viability (held-out tf 4e-6→0.248, monotone, still climbing). Steered to **enlarge corpus to ~8k MATH samples so 1000 steps = 2 epochs** — cap-compliant AND a cleaner, less-overfit held-out signal — then terminalize. Serving/full-scale still gated on (a) defensible debug number and (b) kanna linchpin (#19). | held-out `tf_acceptance_rate_debug_1k`: <0.50 underfit / 0.50–0.70 schedule full / ≥0.70 strong |
| stark | #3 (WIP) | **int4 QAT W4A16** reproduction — local PPL 2.0055 ✓, local TPS ~96 ✓; **awaiting HF Job approval (GitHub issue #11, 0 human comments)**. While blocked: bounded **run-to-run determinism reconciliation** vs kanna's K0 control (linchpin-relevant, interruptible). | ~95 TPS / PPL ~2.01; after approval: run job, post terminal result, merge |
| lawine | #4 (WIP) | **int4 g128 + untied int4 lm_head** re-quant — local PPL 2.0190 ✓, local TPS ~128 ✓, **GREEDY_IDENTICAL 128/128 ✓ (the int4-passes existence proof)**; **awaiting HF Job approval (GitHub issue #12, advisor-endorsed, no human approval)**. While blocked: document GREEDY_IDENTICAL methodology (same-path vs bf16-dense ref; run-to-run) for the linchpin reconciliation. | ~127 TPS / PPL ~2.02, weight-byte floor; after approval: run job, post terminal result, merge |
| kanna | #19 (WIP) | **Batch-invariant vLLM: rescue greedy-valid spec decode** — cashes in her PR #5 finding (vLLM 0.22.0 rejection-sampler spec greedy-INVALID at every precision: M=K+1 batched-verify GEMM shape divergence, precision-independent). A vLLM build with batch-invariant kernels (`VLLM_BATCH_INVARIANT` line of work, Thinking Machines Sep 2025, integrated into `model_executor.layers.batch_invariant`) makes M=K+1 verify forward numerically equal M=1 decode forward. **Goal:** `flip_rate_per_token = 0.0` (GREEDY_IDENTICAL 32/32, then 128-prompt confirm). This is **the linchpin** — gates drafter ladder rungs 4–5 and ALL speculative decode PRs (#9/#16/#18). | `flip_rate_per_token = 0.0`, GREEDY_IDENTICAL 128/128 on int4+MTP |
| ubel | #14 (WIP) | **Empirical lmhead12k** — CPU feasibility done (kept_ids.json, implementation complete); **GPU void + int4 base absent** (pod intermittent). Path unblocked: self-build int4+g128 via path-(a) (prune bf16→quantize), general-12,288 corpus cut (hard-include GT + STEM fill), regenerate 128-capture. PPL scorer requires hard-include of GT tokens or gate fails. **DRAFTER-INDEPENDENT** rung. | served TPS/PPL/greedy-identity 128/128; both 7,584 and 12,288 bandwidth numbers |
| denken | #18 (WIP) | **int4 decode-step cost model vs K** — measure batched M=K+1 verify-forward latency + component shares (weight-GEMM/attn/overhead) across M ∈ {1,2,4,6,8,10,12,16} at ctx 256 & 512, on the public int4 QAT base + synthetic K-token batched forwards. Derive TPS_ideal(K) ceiling curve + realistic TPS under flat and geometric acceptance models. Identify knee M\* and optimal K\* per drafter acceptance model. LOCAL ONLY, not linchpin-gated, not drafter-dependent. | hardware-grounded TPS ceiling curve; M\* knee; attention-share-vs-M (bounds when fa2sw/attn levers matter) |
| wirbel | #21 (WIP) | **Same-path PPL gate** — close the `prompt_logprobs`-vs-timed-model blind spot. Our `ppl_runner.py` scores via `prompt_logprobs` (same path the LF29 frontier games). Add `scripts/local_validation/same_path_ppl.py`: score reference continuations via the timed-throughput `/v1/completions` config, compare to `prompt_logprobs` PPL, gate on `\|Δ\| < 0.05`. Calibrate on `vllm_baseline` (expect both ≈ 2.30); wire as `--check-same-path` in `validate_submission`. **No HF Job, no GPU quota.** Public evidence: itaca `20260613-103155-471`, frantic-penguin + pupa-agent negative `20260613-094903-417` (same-path PPL 2.5454). | `\|same_path_ppl − prompt_logprobs_ppl\| < 0.02` on honest baseline; same-path PPL ≈ 2.30; gate wired |
| land | #9 (WIP) | **wide KL-distilled drafter** — tf-gate **PASS +10.3%** (private-proxy floor +10.9%) but native serving **−4.6%** (train↔serve schedule mismatch + undertrained). v1: free-running / EAGLE-3-style schedule + full ~82-min budget, log eval to W&B | native accept > stock 3.553 (v0 = 3.388) |

## Completed (round 1)

| student | PR | result |
|---|---|---|
| fern | #2 ✓ **MERGED** | **Priority #1 resolved.** Local PPL=2.3012, root cause=40-min HF Job timeout. `scripts/local_prevalidate.py` merged — the team's local pre-validation gate before any HF Job. |
| fern | #10 ✓ **MERGED** | **SAM-Decoding GO verdict.** Causal budget 8.93% K>8 (>3.6% threshold); `analyze_suffix_budget.py` merged; drafter-overlap analysis (#13) merged next. |
| fern | #13 ✓ **MERGED** | **SAM drafter-overlap tooling.** `--drafter-trace` extension + 13/13 mock tests; canonical trace format (`output_start`); template JSON with thresholds. Net-headroom number awaits kanna's acceptance trace. |
| fern | #15 ✓ **MERGED** | **EAGLE-3 feasibility ACCESSIBLE → GO.** `SupportsEagle3` natively in vLLM 0.22.0 + Gemma-4 E4B; aux layers `(2,21,39)`, `[T,2560]` bf16, CUDA-graph safe; drafter head arch exists (`llama_eagle3.py`). Zero patching. Empirical probe confirmed. Report at `research/eagle3_feasibility/`. |
| wirbel | #8 ✓ **MERGED** | **Local validation + profiling infra.** `scripts/local_validation/` — one-command `validate_submission`, served spec-off greedy reference (offline AR falsely fails 26/128 — 20.3%), local PPL runner, decode profiler. Profiler finding: lm_head vocab GEMV = 26.4% of decode GPU time → confirms lmhead12k as top non-block lever. Canonical greedy reference committed: `research/greedy_reference/google__gemma-4-E4B-it/` (bf16, 128 prompts, served spec-off). |
| denken | #7 ✗ **CLOSED** (negative) | **fa2sw + onegraph both dead ends standalone** on int4 base at conc=1 (rigorous isolation). fa2sw: −4.9% TPS + DIVERGENT 82/128. onegraph: parity + DIVERGENT 1/128. Mechanism: ~92% weight-GEMM/BW-bound, CUDA graph already collapses the step. **Bonus finding:** int4 base cross-process bit-exact (sha256 run#1==run#2) — 4th reconciliation data point confirming base is gate-valid in M=1 sequential prefix-cache-OFF. fa2sw needs vLLM worker-plugin (V1 EngineCore separate process). |
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
480-550 TPS; highest-ceiling drafter in the pipeline. Serving validity gated on **kanna #19** (linchpin).

Pending earlier threads (still open): provably-greedy sparse verification (ubel's #6 is the
round-1 version); the `DECODE_TPS_CAP` PENDING leaderboard entries (confirm cap-gaming vs. real
accounting) — deprioritized vs. the drafter ladder.

_Last updated: 2026-06-13 (cycle 9 — wirbel #21 assigned: same-path PPL gate; kanna #19 active: batch-invariant vLLM linchpin fix; LF29 lane same-path PPL 2.55 confirmed; wirbel #8 moved to completed; land #9 rebase nudge sent)._
