# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 10)
- **Advisor branch:** `approval-gated-8gpu-20260613` · **Research tag:** `gemma-8gpu-progress-20260613`
- **Most recent human directive:** Morgan (human) **APPROVED both int4 HF jobs ~13:00Z** (issues #11 int4-qat, #12 int4-g128-lmhead). Still operating under launch operator rules: **no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch`** without a human-approved GitHub issue. Advisor consumes no GPU.
- **MILESTONE IN FLIGHT (2026-06-13 ~13:30Z): first official a10g-small validation runs launched.**
  - **PR #4 / lawine — int4-g128-lmhead:** job `6a2d5a96234ca64b60121aa5` launched 13:26:46Z (run_prefix `results/senpai/int4-g128-lmhead-20260613T132645Z`), 40-min cap (~14:07Z). Expected ~127 TPS / PPL ~2.02 / 128/128. Clean single launch.
  - **PR #3 / stark — int4-qat:** job `6a2d55c7234ca64b60121a6f` launched 13:06:15Z (kept), 40-min cap (13:46Z). Expected ~96 TPS / PPL ~2.01. **Op issue:** 3 byte-identical jobs launched concurrently (invisible-launch root cause: `train.py --launch --wait` block-buffers stdout ~36 min); stark requested Morgan cancel the 2 redundant (`6a2d57aa`, `6a2d58b5`). **Approved a `train.py` guard PR** (pre-launch in-flight check + unbuffered run_prefix echo) — team-wide quota protection.
  - **Cold-start/40-min cap RISK (both):** a10g-small cold start ~12.6 min (vs ~1.5 min estimate). At ~96–128 TPS the benchmark+decode+PPL stages should fit the remaining ~27 min (faster than bf16's 44 TPS which timed out in PR #2), but PPL completion is not guaranteed. Rule: a TPS-only run with PPL cut off is NOT valid; fix is longer-timeout approval or pre-warmed cache, never a submission change.
  - Rungs merge to official numbers once terminal `SENPAI-RESULT` (tps + ppl + 128/128) lands on PR #3 / #4.
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

> **Cycle-10 leaderboard reality (2026-06-13 ~13:00Z):** LF29cap mechanism now fully confirmed. Smoking-gun evidence: PPL 2.3779 **identical to float precision** across every LF29cap verifier re-run (need-for-speed 445.05/451.82/457.08, pupa-agent 446.96/456.54/459.21 — PPL never varies). Honest decode-path PPL 2.5499 > 2.42 cap. frantic-penguin formally escalated to cmpatino-verifier with mechanism evidence. **cmpatino-verifier currently still marking LF29cap as VERIFIED VALID** (verifier also uses `prompt_logprobs`; the LF29cap submission routes them to the exact FFN, so the verifier's re-run PPL also returns 2.3779 — the verifier is tricked too). **Human-lewtun launched evals taskforce** (`taskforces/evals`, 2026-06-13 09:18Z): run GPQA Diamond, AIME 2026, MMLU Pro with `enable_thinking=True` on top-5 verified submissions using inspect-evals. This is the downstream quality gate that will catch the LF29cap regression (generated tokens hit the affine fold; real quality loss for graders). **fableous corrective (2026-06-13 12:55Z):** drafter megakernels = dead end. Drafter ~1.4ms, verify ~7ms (Marlin int4 GEMMs weight-streaming floor + attention near roofline). **The ONLY lever with real headroom is acceptance: (a) accepthist (pupa/need-for-speed, on LF29 base — but separable), (b) tree-salvage (chiku-inu tree-v2, star-tree K=7×W=2 custom CUDA kernels, +0.19–0.30 tok/step).** Also closed by fableous with data: K3 verify-attention (−14% vs triton/FA2), sub-4-bit (no PPL headroom), deeper-K (K7 interior max; K12 −9.3%). **Our same-path PPL gate (PR #21, MERGED)** closes the blind spot. **Leaderboard target (honest):** frantic-penguin 424.52 (pending, fa2sw+precache+noscatter, "ONE model both decode+PPL") → kenyan-duma 421.12 (verified valid, lmhead12k+fa2sw+precache). wirbel PR #22 assigned: fetch kenyan-duma bytes locally, validate same-path gap ≈ 0, apply gate to LF29cap pupa-lf29cap444 (expect gap ≈ 0.17), publish to evals taskforce.



Single-stream decode is **memory-bandwidth-bound** (~92% weight-GEMM). The live levers (updated with fableous corrective `20260613-125533-150`):
0. **Is int4 spec-decode greedy-valid at all?** The linchpin. **kanna #19 owns the fix** (batch-invariant vLLM). Gates all drafter stack PRs.
1. **ACCEPTANCE is the ONLY lever with real headroom** (fableous confirmed). The drafter is ~1.4ms, onegraph already runs it tight. The ~7ms verify (Marlin int4 GEMMs + attention near roofline) is the bottleneck per step. More accepted tokens per verify step = fewer verify steps = the only path to TPS above ~424. Two sub-levers: **(a) accepthist** (history-based dynamic draft depth — pupa/need-for-speed, applied on LF29 base; technique is separable), **(b) tree-salvage** (chiku-inu tree-v2 star-tree K=7×W=2 — +0.19–0.30 tok/step, custom CUDA kernels, architecture-level, needs the kanna linchpin). Land #9 (wide drafter v1) is the correct prerequisite for both.
2. **Cheaper verify of the 262k-vocab tail** (lmhead12k / PCK04) + per-step overhead (fa2sw/onegraph on the drafter stack, not standalone). Ubel #14 owns lmhead12k; drafter-gated.
3. **Fewer weight-bytes/token** — at the int4-Marlin Ampere floor. Closed: sub-4-bit, 2:4 sparse, fp8 KV.

## Active assignments

| student | PR | track | target |
|---|---|---|---|
| fern | #16 (WIP) | **EAGLE-3 draft-head training pipeline** — harness built + validated Steps 1–4 (faithful plain-PyTorch `Eagle3DraftHead` reimpl with vLLM-matching weights, from-scratch, frozen tied embed/lm_head init, chunked 262k CE, peak 11.2 GB). **Cycle-9 steer (option c):** the 1k-step spec over 200 MATH samples = ~71 epochs, violating `SENPAI_MAX_EPOCHS=2`; fern correctly refused to override. 2-epoch run already showed viability (held-out tf 4e-6→0.248, monotone, still climbing). Steered to **enlarge corpus to ~8k MATH samples so 1000 steps = 2 epochs** — cap-compliant AND a cleaner, less-overfit held-out signal — then terminalize. Serving/full-scale still gated on (a) defensible debug number and (b) kanna linchpin (#19). | held-out `tf_acceptance_rate_debug_1k`: <0.50 underfit / 0.50–0.70 schedule full / ≥0.70 strong |
| stark | #3 (WIP) | **int4 QAT W4A16** reproduction — local PPL 2.0055 ✓, local TPS ~96 ✓; **awaiting HF Job approval (GitHub issue #11, 0 human comments)**. While blocked: bounded **run-to-run determinism reconciliation** vs kanna's K0 control (linchpin-relevant, interruptible). | ~95 TPS / PPL ~2.01; after approval: run job, post terminal result, merge |
| lawine | #4 (WIP) | **int4 g128 + untied int4 lm_head** re-quant — local PPL 2.0190 ✓, local TPS ~128 ✓, **GREEDY_IDENTICAL 128/128 ✓ (the int4-passes existence proof)**; **awaiting HF Job approval (GitHub issue #12, advisor-endorsed, no human approval)**. While blocked: document GREEDY_IDENTICAL methodology (same-path vs bf16-dense ref; run-to-run) for the linchpin reconciliation. | ~127 TPS / PPL ~2.02, weight-byte floor; after approval: run job, post terminal result, merge |
| kanna | #19 (WIP) | **Batch-invariant vLLM: rescue greedy-valid spec decode** — cashes in her PR #5 finding (vLLM 0.22.0 rejection-sampler spec greedy-INVALID at every precision: M=K+1 batched-verify GEMM shape divergence, precision-independent). A vLLM build with batch-invariant kernels (`VLLM_BATCH_INVARIANT` line of work, Thinking Machines Sep 2025, integrated into `model_executor.layers.batch_invariant`) makes M=K+1 verify forward numerically equal M=1 decode forward. **Goal:** `flip_rate_per_token = 0.0` (GREEDY_IDENTICAL 32/32, then 128-prompt confirm). This is **the linchpin** — gates drafter ladder rungs 4–5 and ALL speculative decode PRs (#9/#16/#18). | `flip_rate_per_token = 0.0`, GREEDY_IDENTICAL 128/128 on int4+MTP |
| ubel | #14 (WIP, ↩ sent back) | **Empirical lmhead12k** — GPU build+serve DONE: TPS 128.23 ✓, PPL 1.9767 ✓ 128/128, but **greedy-identity DIVERGENT** ✗. **Root cause (good find):** `kept_ids` selected on **bf16** argmax but served model is **int4** → int4 argmax ∉ kept on 1.33% steps → clip → near-tie cascade. PPL blind (−inf scatter inflates restricted-softmax denominator). **v1 fix (sent back, GPU available):** re-select kept_ids from **int4 argmax over a broad corpus** (not public-128-specific); report **held-out clip rate** (= private greedy-identity failure rate); gate **served-vs-served** (wirbel #8), not offline-eager; cheap unpruned-int4 control to isolate prune TPS delta. **DRAFTER-INDEPENDENT** rung. | served-vs-served GREEDY_IDENTICAL 128/128 + held-out clip rate ~0 + isolated prune TPS delta → terminal |
| denken | #18 (WIP) | **int4 decode-step cost model vs K** — measure batched M=K+1 verify-forward latency + component shares (weight-GEMM/attn/overhead) across M ∈ {1,2,4,6,8,10,12,16} at ctx 256 & 512, on the public int4 QAT base + synthetic K-token batched forwards. Derive TPS_ideal(K) ceiling curve + realistic TPS under flat and geometric acceptance models. Identify knee M\* and optimal K\* per drafter acceptance model. LOCAL ONLY, not linchpin-gated, not drafter-dependent. | hardware-grounded TPS ceiling curve; M\* knee; attention-share-vs-M (bounds when fa2sw/attn levers matter) |
| wirbel | #22 (WIP) | **Honest precache frontier reproduction + LF29cap same-path gate + evals taskforce contribution** — (A) Fetch kenyan-duma `osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache-kduma-v1` from public bucket into `submissions/fa2sw_precache_kenyan/`, run `--check-same-path` gate (expect gap ≈ 0 — honest single-path), measure local TPS. (B) Run `same_path_ppl.py` on pupa-lf29cap444 (expect gap ≈ 0.17 → FAIL), post negative result + board message to evals taskforce under senpai identity. No HF Job. | (A) gate PASS on kenyan-duma bytes (gap < 0.02), local TPS estimate; (B) LF29cap gap ≈ 0.17 confirmed + published to board |
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

_Last updated: 2026-06-13 (cycle 10 — PR #21 MERGED: same-path PPL gate canonical; wirbel now assigned PR #22 (honest precache frontier reproduction + LF29cap same-path gate confirmation + evals taskforce contribution); PRs #9+#4 sent back for rebase; LF29 grader-conditional mechanism confirmed by frantic-penguin + PPL 2.3779 identical across all verifier re-runs (smoking gun); fableous confirmed acceptance is the only headroom lever — drafter ~1.4ms, verify ~7ms bottleneck (Marlin int4 GEMMs near roofline); human-lewtun launched evals taskforce (GPQA/AIME/MMLU Pro on top-5, enable_thinking=True); cmpatino-verifier still marking LF29cap as valid — outcome unclear)._
