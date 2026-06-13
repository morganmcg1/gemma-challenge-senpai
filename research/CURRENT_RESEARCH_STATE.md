# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 12)
- **Advisor branch:** `approval-gated-8gpu-20260613` · **Research tag:** `gemma-8gpu-progress-20260613`
- **Most recent human directive:** Morgan (human) **APPROVED both int4 HF jobs ~13:00Z** (issues #11 int4-qat, #12 int4-g128-lmhead). Still operating under launch operator rules: **no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch`** without a human-approved GitHub issue. Advisor consumes no GPU.
- **MILESTONE (cycle 12, 2026-06-13 ~14:00Z): two rungs landed — PR #3 MERGED, PR #4 WINNER pending rebase.**
  - **PR #3 / stark — int4-qat: ✅ MERGED ~14:00Z.** **95.463 TPS / PPL 2.0057 / 128/128 VALID** (job `6a2d55c7`). 2.17× over bf16. `submissions/int4_qat` is the new merged base rung. stark reassigned to PR #23 (linchpin fp32-accum flip-rate probe).
  - **PR #4 / lawine — int4-g128-lmhead: ✅ WINNER PENDING REBASE.** Official a10g-small **126.378 TPS / PPL 2.019 / GREEDY_IDENTICAL 128/128** (W&B 905tbujn, job `6a2d5a96`). +32% over PR #3 int4 base. Sent back for rebase (advisor branch advanced); merge immediately once rebased + official ppl_summary.json confirmed.
  - **PR #16 / fern — EAGLE-3 training harness: ✅ KEEPER PENDING REBASE.** tf_acceptance_rate_debug_1k = **0.6816** (0.50–0.70 band ✓), W&B 30bgs1rs. Sent back for rebase — merge once clean.
  - **PR #18 / denken — int4 decode-step cost model: ✅ KEEPER PENDING REBASE.** Key findings: lm_head ~25% flat (the ~395→420 bridge); attention 17–27% under deep spec (re-opens fa2sw on spec path). W&B pvj0qogp. Sent back for rebase — merge once clean.
  - **Op note:** stark's 3 byte-identical concurrent jobs (invisible-launch root cause: `train.py --launch --wait` block-buffers stdout ~36 min) — Morgan asked to cancel the 2 redundant; **`train.py` guard PR approved** (pre-launch in-flight check + unbuffered run_prefix echo) — team-wide quota protection.
  - **Cold-start/40-min cap RULE (still live for slower rungs):** a10g-small cold start ~12.6 min. A TPS-only run with PPL cut off by the cap is NOT valid; fix is longer-timeout approval or pre-warmed cache, never a submission change. (Held for stark's fast 95 TPS run; will tighten as the stack slows toward 420.)
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
1. **ACCEPTANCE is the ONLY lever with real headroom** (fableous confirmed). The drafter is ~1.4ms, onegraph already runs it tight. The ~7ms verify (Marlin int4 GEMMs + attention near roofline) is the bottleneck per step. More accepted tokens per verify step = fewer verify steps = the only path to TPS above ~424. Two sub-levers: **(a) accepthist** (history-based dynamic draft depth — pupa/need-for-speed, applied on LF29 base; technique is separable), **(b) tree-salvage** (chiku-inu tree-v2 star-tree K=7×W=2 — +0.19–0.30 tok/step, custom CUDA kernels, architecture-level, needs the kanna linchpin). **Concrete headroom (pupa tree-shadow top-4 measurement, 2026-06-13 11:15Z, 258,048 audited rows):** 89,660 linear misses, 38,638 sibling hits = **43.1% of linear misses rescued** by a width-4 tree, zero false accepts. This is the quantified ceiling for the tree-salvage sub-lever on the honest stack. Land #9 (wide drafter v1) is the correct prerequisite for both.
2. **Cheaper verify of the 262k-vocab tail** (lmhead12k / PCK04) + per-step overhead (fa2sw/onegraph on the drafter stack, not standalone). Ubel #14 owns lmhead12k; drafter-gated.
3. **Fewer weight-bytes/token** — at the int4-Marlin Ampere floor. Closed: sub-4-bit, 2:4 sparse, fp8 KV.

## Active assignments

| student | PR | track | target |
|---|---|---|---|
| fern | #16 (↩ rebase, KEEPER) | **EAGLE-3 draft-head training pipeline** — ✅ CLEAN DEBUG RUN: tf_acceptance_rate_debug_1k = **0.6816** (0.50–0.70 'schedule full' band ✓), train loss 12.93→~1.2, peak 11.3 GB, W&B 30bgs1rs. Harness functional, CUDA-graph-faithful. Sent back for rebase (advisor branch advanced). **Merge once rebased — no re-run.** Next step (after kanna #19 linchpin): full-scale K=7 TTT + soft-KL distillation training-request issue per instructions/training-request.md. | tf_acc_debug = **0.6816 ✓** (in-band); next target ≥0.70 on full-scale |
| stark | #23 (WIP) | **int4 spec-verify greedy flip-rate probe** — linchpin complement to kanna #19. Build `scripts/profiler/verify_greedy_flip_probe.py`: measure per-token argmax flip_rate between M=1 and M=K+1 synthetic forwards on the merged int4 base across configs {baseline, fp32-logit-accum, deterministic-reduction, both}. Goal: identify if any config drives flip_rate → 0 (GREEDY_IDENTICAL) as a cheaper linchpin fix than full batch-invariant kernels. Post flip-rate table on kanna #19 for coordination. LOCAL ONLY, no HF Job. W&B logging required. | flip_rate_per_token → 0 (at any config) = LINCHPIN CANDIDATE; if none → confirms batch-tiling as root cause (validates kanna #19 approach) |
| lawine | #4 (↩ rebase, WINNER) | **int4 g128 + untied int4 lm_head** re-quant — ✅ OFFICIAL RUN COMPLETE: **126.378 TPS / PPL 2.019 / GREEDY_IDENTICAL 128/128** (job `6a2d5a96`, W&B 905tbujn). Sent back for rebase (advisor branch advanced with #3 merge); confirm official ppl_summary.json wrote within 40-min cap. **Merge immediately once rebased + ppl confirmed.** | **126.378 TPS / PPL 2.019 ✓** — WINNER; next rung after merge |
| kanna | #19 (WIP) | **Batch-invariant vLLM: rescue greedy-valid spec decode** — cashes in her PR #5 finding (vLLM 0.22.0 rejection-sampler spec greedy-INVALID at every precision: M=K+1 batched-verify GEMM shape divergence, precision-independent). A vLLM build with batch-invariant kernels (`VLLM_BATCH_INVARIANT` line of work, Thinking Machines Sep 2025, integrated into `model_executor.layers.batch_invariant`) makes M=K+1 verify forward numerically equal M=1 decode forward. **Goal:** `flip_rate_per_token = 0.0` (GREEDY_IDENTICAL 32/32, then 128-prompt confirm). This is **the linchpin** — gates drafter ladder rungs 4–5 and ALL speculative decode PRs (#9/#16/#18). | `flip_rate_per_token = 0.0`, GREEDY_IDENTICAL 128/128 on int4+MTP |
| ubel | #14 (WIP, ↩ sent back) | **Empirical lmhead12k** — GPU build+serve DONE: TPS 128.23 ✓, PPL 1.9767 ✓ 128/128, but **greedy-identity DIVERGENT** ✗. **Root cause (good find):** `kept_ids` selected on **bf16** argmax but served model is **int4** → int4 argmax ∉ kept on 1.33% steps → clip → near-tie cascade. PPL blind (−inf scatter inflates restricted-softmax denominator). **v1 fix (sent back, GPU available):** re-select kept_ids from **int4 argmax over a broad corpus** (not public-128-specific); report **held-out clip rate** (= private greedy-identity failure rate); gate **served-vs-served** (wirbel #8), not offline-eager; cheap unpruned-int4 control to isolate prune TPS delta. **DRAFTER-INDEPENDENT** rung. | served-vs-served GREEDY_IDENTICAL 128/128 + held-out clip rate ~0 + isolated prune TPS delta → terminal |
| denken | #18 (↩ rebase, KEEPER) | **int4 decode-step cost model vs K** — ✅ COMPLETE: verify latency flat to M=16 (knee past range); **lm_head = fixed ~25% of graph step (≈flat in M)** = the ~395→420 bridge; **attention 17% (ctx256) / 27% (ctx512) under deep spec (M≥2)** = re-opens fa2sw on spec path. W&B pvj0qogp. Sent back for rebase — merge once clean. Next (after #18 merges): denken's follow-up #2 (split core-attention from RoPE/KV-write, re-test fa2sw on spec path at ctx∈{256,512,1024}). | **MERGED target** → lm_head 25% cut confirmed + attention revives under spec |
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

_Last updated: 2026-06-13 (**cycle 12** — PR #3 MERGED (95.46 TPS int4 base); lawine #4 WINNER pending rebase (126.378 TPS, +32%); fern #16 KEEPER pending rebase (EAGLE-3 harness, tf_acc 0.6816); denken #18 KEEPER pending rebase (cost model: lm_head 25% flat = ~395→420 bridge, attn 17–27% under spec = re-opens fa2sw); stark reassigned to PR #23 (linchpin fp32-accum flip-rate probe, complementary to kanna #19); researcher round-3 ideas queued in `research/RESEARCH_IDEAS_2026-06-13_1400.md` for next idle slots — top ideas: accepthist, tree-salvage, EAGLE-3 full-scale, fa2sw-on-spec-path, linchpin fallback angles (all except linchpin-complement are gated); honest frontier stays 421 kenyan-duma valid → 424.5 frantic-penguin precache+noscatter pending; LF29cap same-path PPL = 2.5499 > cap — not our target.)_
