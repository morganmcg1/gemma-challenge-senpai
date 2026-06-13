STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["r0ahjs45"],"primary_metric":{"name":"verify_attn_share_postsplitkv_pct","value":7.6},"test_metric":{"name":"frontier_E_accept_tokens_per_cycle","value":3.847}}

## Results — post-#43 decode re-profile (split-KV ACTIVE)

**TL;DR — hypothesis confirmed.** Split-KV (#43) collapsed verify-attention from the **#2 block (19.6%)** to **#3 (7.6%)** of decode GPU time, and **drafter-forward (K=7 MTP) is the new #2 block (18.1%, 1445 µs/step)**. Verify forward dropped −17.5% (7.906→6.519 ms); decode stays 99.4% GPU-bound. This **promotes stark #47 (W8A8 INT8 drafter)** to the top tractable kernel lever and — because split-KV also flattened verify cost(M) ~6× — makes **stark #47 + denken #51 dynamic-K complementary**. Acceptance (land #9 / fern #34) remains the dominant TPS lever overall. **Separately, per the advisor's request I evaluated denken #51 (Task 1)'s split-KV context-gate on the served stack — it is NOT supported:** the cost-model M=8 short-ctx penalty does not transfer (predicted **+9.7% slower** vs measured **−17.5% faster** over this run's real ctx distribution — a 27 pp gap; see the dedicated cross-check below).

> Local A10G exploratory probe — NOT the official a10g-small TPS. No-precache locally (same conditions as #30 and the #43 A/B); the **relative** shift vs #30 is the apples-to-apples result. Read-only profile: **zero submission-file changes**, PPL/greedy/serving untouched.

### Split-KV engagement confirmed (not a silent 2D no-op)
Server log: `[splitkv-verify] wrapped unified_attention (redirect 1<M<=64 verify batches to 3D split-KV)` → `verify batch M=8 q_rows=8 -> 3D split-KV (n=1..5)` (n log-capped at 5; redirect continues every verify step). No `redirect skipped` / `patch error` / 2D-fallback. `gpu_busy_share_of_wall_raw=99.4%`.

### Cycle (p50 steady-state) — diff vs #30
| quantity | #30 pre-splitKV | **post-#43** | Δ |
|---|--:|--:|--:|
| drafter forward GPU (ms) | 1.446 | **1.445** | −0.001 |
| verify forward GPU (ms) | 7.906 | **6.519** | −1.387 (−17.5%) |
| GPU-busy / cycle (ms) | 9.352 | **7.964** | −1.388 (−14.8%) |
| inter-op gap / host (ms) | 0.064 | **0.047** | −0.017 |
| cycle wall (ms) | 9.416 | **8.011** | −1.405 (−14.9%) |
| E_accept (tok/cycle) | 3.817 | **3.847** | +0.030 |
| local steady gen TPS | 391.3 | **455.3** | +64.0 (+16.4%) |
| wall-clock token-wtd TPS | 391.2 | **454.4** | +63.2 (+16.1%) |

### De-duped decode GPU-time composition (% of GPU-busy + µs/step)
| block | #30 % | **post-#43 %** | Δ pp | post-#43 µs/step |
|---|--:|--:|--:|--:|
| #1 verify body int4-Marlin GEMM | 53.2% | **60.6%** | +7.4 | 4826 |
| **#2 drafter forward (K=7 MTP)** | 15.5% | **18.1%** | +2.7 | **1445** |
| #3 verify attention (fa2sw) | **19.6%** | **7.6%** | **−12.0** | 605 |
| #4 verify norm / elementwise | 6.7% | 7.5% | +0.8 | 595 |
| #5 sampling | 2.6% | 2.9% | +0.3 | 233 |
| #6 verify lm_head12k GEMV | 1.0% | 0.3% | −0.6 | 26 |
| inter-op gap / CUDA-graph launch | — | — | — | +47 (host fully overlapped, ~0% of cycle) |

- **Attention collapse CONFIRMED:** 19.6%→**7.6%** (1836→605 µs, **3.03×** absolute). Slightly above the predicted ~5–6% because the in-serving blend is less than the op-level 4.38× sliding microbench — only the Triton-path verify layers (global head-512, KV-shared) get the full M=8→3D speedup; FA2 sliding layers and the already-3D M=1 decode attention don't move. Direction + magnitude hold.
- **New #2 block = drafter-forward** (18.1%, 1445 µs). Body GEMM stays #1; its share rose +7.4 pp purely because the GPU-busy denominator shrank (absolute µs flat, 4971→4826 within trace noise).

### Isolation: split-KV flattened verify cost(M) ~6× (the key enabler for K-raising)
| verify GPU (ms) | M=1 (spec_off) | M=8 (frontier) | slope M=1→8 |
|---|--:|--:|--:|
| #30 | 6.326 | 7.906 | **+25.0%** |
| **post-#43** | 6.252 | 6.519 | **+4.3%** |

The +25% pre-split-KV slope **was** the M=8 attention under-occupancy; split-KV removed it. Verify is now near-flat in M, so **raising K is almost free on the verify side** — the only cost that grows with K is now the drafter (#2 block).

### Next lever — ranked, quantified (local steady cost model, cycle ≈ GPU-busy)
| lever | block hit | projected ΔTPS (local 455 →) | tractability |
|---|---|--:|---|
| **acceptance E_accept 3.85→4.5** (land #9 / fern #34) | none (multiplier) | **+17.0% → 533** | drafter training (cluster) |
| body GEMM −15% (sub-int4 / faster Marlin) | #1 (60.6%) | +9.9% → 501 | hard (new weight format) |
| **drafter W8A8 INT8 −30%** (stark #47) | **#2 (18.1%)** | **+5.7% → 481** | tractable kernel lever |
| drafter W8A8 INT8 −40% (stark #47) | #2 | +7.8% → 491 | tractable |
| attention −100% (ceiling, unreachable) | #3 (7.6%) | +8.2% → 492 | de-prioritised |
| denken #51 (Task 1) split-KV ctx-gate (M=8) | #3 attn | **~0% on served (cost-model penalty doesn't transfer)** | **NOT supported by served data** |

**In-flight-lever cross-check:**
- **stark #47 (W8A8 INT8 draft-forward) — SUPPORTED ↑↑.** Drafter is now the **#2 block (18.1%)**, up from #3, and is currently **BF16** (`/tmp/qat-assistant`, 49 BF16 tensors). Its K=7 bandwidth-bound GEMVs are exactly what W8A8 halves the weight stream on → **+5.7% TPS at −30%**. Single highest-leverage *kernel* lever after the (hard) body GEMM.
- **denken #51 — two facets, split verdict.** *(a) Dynamic-K consequence — SUPPORTED ↑:* verify cost(M) now nearly flat (+4.3% vs +25%), so deeper K is ~free on verify; the per-K cost is now the drafter (#2, ~linear in K), so this is **complementary with stark #47** (a cheaper drafter buys deeper-K headroom). *(b) Task-1 split-KV context-gate — NOT supported on the served stack:* denken's M=8 cost-model curve says split-KV is net-negative at every served ctx (+15.5%@256 → +5.1%@1024), but my served re-profile + the #43 served A/B measure **−17.5% (a win)** over exactly that ctx distribution. Full quantified cross-check below.
- **land #9 / fern #34 (drafter acceptance) — SUPPORTED ↑ (still dominant).** Acceptance is a linear TPS multiplier at zero per-step cost (E_accept 3.85→4.5 = +17%); split-KV is orthogonal (cut per-step cost) — the two **multiply**.

**Recommendation:** the dominant lever is unchanged — **acceptance (land #9 / fern #34)**. The re-profile's new contribution: drafter-forward is the #2 per-step block, making **stark #47 (W8A8 INT8 drafter, +5.7–7.8%)** the top tractable kernel lever and **complementary with denken #51 dynamic-K**. **De-prioritise further attention work** — split-KV already captured it (7.6%, ceiling +8.2%). The advisor-requested **denken #51 (Task 1) split-KV context-gate is NOT supported by the served data** (cross-check below).

### denken #51 (Task 1) split-KV context-gate — evaluated on the served stack: NOT supported
The advisor asked me to fold in denken #51's cost-model finding that the #43 split-KV redirect is **net-negative at small-M / short-ctx, net-positive at long ctx**, and to evaluate a **`seqlen_kv ≥ threshold` context-gate** (explicitly NOT an `M≥33` gate — our linear MTP K=7 stack is always M=8) that would recover the early-decode penalty on the deployed stack. I reproduced denken's curve (`compare_splitkv_curves.py --operating-M 8` on the merged `results_pr51_*` JSONs) and cross-checked it against this run's **real** per-cycle context distribution. **It does not pencil out — the cost-model M=8 penalty does not transfer to the served stack.**

denken's M=8 cost-model curve (split-KV ON vs OFF), reproduced:

| M=8 verify Δ% | ctx256 | ctx512 | ctx1024 |
|---|--:|--:|--:|
| split-KV vs baseline | **+15.5%** | **+8.0%** | **+5.1%** |

At **M=8** the redirect is **slower at every measured ctx** — on only 8 query rows the `reduce_segments` merge + extra CTA launches exceed the small attention saving; the crossover to net-positive never occurs within the served range. (Cross-check at M=45 reproduces denken's headline exactly: −2.6 / −6.3 / −7.1% @ctx256/512/1024 — confirming the win is an **M≥33 / tree-path** effect, not a linear-M=8 effect.)

This run's served ctx distribution (128 prompts, prompt-len median 230 / mean 272, all generate 512 → ~133 cycles/prompt, **17,024 verify cycles**):

| seqlen_kv regime | % of served verify cycles | denken M=8 says |
|---|--:|---|
| ctx < 256 | 8.2% | +15.5% slower |
| ctx < 512 | **52.0%** | +8.0–15.5% slower |
| ctx ≥ 1024 | 2.6% | +5.1% slower |

The **majority of decode (52% of cycles below ctx512)** sits squarely in denken's predicted M=8 penalty band. Weighting denken's M=8 Δ%(ctx) by this exact distribution (cost-weighted):

> **cost-model PREDICTS aggregate verify ≈ +9.7% SLOWER; SERVED MEASURED = −17.5% FASTER. Gap = −27.2 pp.**

The served stack (this re-profile + the #43 served A/B, both p50 over all steps) shows split-KV is a **clean −17.5% verify win over exactly the ctx range the cost model says should be +9.7% slower**. A cost-model-honest `seqlen_kv` gate would fire **~never at M=8** (the M=8 curve is positive at all served ctx) — i.e. it would be equivalent to split-KV OFF and would **forfeit the measured win, not recover a penalty.**

**Why the gap (most likely):** the cost model is a synthetic verify-only microbench — its M=8 step is ~1.6× the served verify in absolute terms (11.7–12.8 vs 6.5–7.9 ms) and its M=8 attention fraction (17.6–30.2%) is far above the served verify-attention (7.6% of GPU-busy). It therefore over-weights both the attention the redirect targets and the segment-reduction overhead. The served path is CUDA-graph-captured (`ONEGRAPH`), 99.4% GPU-bound with host fully overlapped, so the extra-CTA-launch / `reduce_segments` overhead denken flags is captured in-graph and overlapped rather than charged to wall time.

**Verdict: do NOT add the context-gate on current evidence.** The #53 served re-profile directly measures the deployed M=8 stack across denken's flagged ctx range and finds a win, not a penalty. **Definitive test (follow-up):** a ctx-resolved served A/B — log per-step `(seqlen_kv, verify_ms)` in both `SPLITKV_VERIFY=0/1` arms and bin Δ%(ctx); if that *served* curve shows an early-ctx penalty, a gate is warranted and I can locate the served crossover. Until then the served evidence says leave split-KV unconditionally ON. (Artifact + reproduce: `ctx_gate_analysis.py`.)

### Run details
- **Command:** `python -m scripts.local_validation.profile_decode --submission submissions/fa2sw_precache_kenyan --num-prompts 128 --output-len 512 --variants frontier,lmhead_off,spec_off --wandb-name wirbel/decode-reprofile-postsplitkv-128 --wandb-group decode-reprofile-postsplitkv --out-dir research/profiling/frontier_decode_postsplitkv/`
- **W&B run:** `r0ahjs45` (group `decode-reprofile-postsplitkv`) · entity `wandb-applied-ai-team/gemma-challenge-senpai`
- **Peak GPU mem:** ~19.4 GiB observed (model load 8.85 GiB, KV 9.46 GiB; budget 0.90×23 = 20.7 GiB; #43 peak 21.0 GiB)
- **No HF job / submission.** Local-only profile (Issue #46 / lawine #52 own the split-KV official launch). No `summary.json`/`run_prefix` — this is a composition profile, not a benchmark run.
- **Validity:** `git status` shows only `research/profiling/frontier_decode_postsplitkv/` added; #43 already validated PPL 2.3767 ≤ 2.42 + greedy-equivalence for this exact stack.
- **Public evidence used:** digest 2026-06-13 — our base `kenyan-duma` osoi5 stack = leaderboard rank 7 (~421 TPS official); #1 VALID rock-ai 459.72 (accepthist+cap); top-4 all use accepthist+cap, corroborating the land #9/fern #34 + denken #51 lanes. Builds on #30 (`07kg6bn7`, pre-split-KV composition) and #43 (split-KV A/B, −17.5% verify).
- Artifacts: `research/profiling/frontier_decode_postsplitkv/{FINDING.md, breakdown.md, frontier_decode_profile.json, analyze_diff.py, ctx_gate_analysis.py}`.

### What happened
The hypothesis held: split-KV reshaped the composition exactly as predicted in direction (attention collapse, drafter promoted to #2), with the only nuance being attention landed at 7.6% rather than 5–6% (blended layer-type speedup < op-level 4.38×). The surprise upside is the **verify cost(M) flattening (+25%→+4.3%)**, which de-risks the K-raising levers (denken #51 dynamic-K) and ties them to the now-#2 drafter cost — so stark #47 and denken #51 dynamic-K are complementary, not competing.

Acting on the advisor's mid-run note, I also cross-checked **denken #51 (Task 1)'s split-KV cost(M,ctx) curve** against this run's real ctx distribution. At the deployed **M=8** the cost model predicts a short-ctx *penalty* (+9.7% weighted over this run's contexts), but the served stack measures **−17.5%** — a **27 pp gap** — so the proposed `seqlen_kv` context-gate is **not supported by served evidence** (most plausibly a synthetic-microbench-vs-CUDA-graph artifact: the microbench over-weights attention/segment-reduction and doesn't overlap the extra CTA launches the served onegraph hides). The honest resolver is a ctx-resolved served A/B (follow-up 1). Net: this is a useful *negative* on a proposed micro-opt — it does not change the #53 headline (split-KV is a clean served win and attention is captured).

### Suggested follow-ups
1. **(Resolves the denken #51 context-gate question) ctx-resolved served A/B.** Log per-step `(seqlen_kv, verify_ms)` in both `SPLITKV_VERIFY=0/1` arms (n128, output_len 512) and bin served verify Δ%(ctx). This is the apples-to-apples served test of denken's M=8 cost-model penalty. If the *served* curve also shows an early-ctx penalty, a `seqlen_kv ≥ threshold` gate is warranted and I can locate the served crossover; if (as the aggregate −17.5% and the +9.7%-predicted-vs-−17.5%-measured gap suggest) it does not, leave split-KV unconditionally ON. Cheap: reuses the existing A/B harness + a per-step seqlen tap.
2. **stark #47 W8A8 INT8 drafter** is the top tractable kernel lever the re-profile surfaces (+5.7–7.8%); the drafter is BF16 and bandwidth-bound, so the win should be clean. Pair it with denken #51 dynamic-K (flatter verify cost(K) leaves K-headroom that a cheaper drafter unlocks).
3. The queued **#49 tree-ceiling tightening of `tree_acceptance_model.py`** should re-fit on the post-split-KV verify curve (verify_base ≈ 6.5 ms, attention no longer M-growing) — the flatter cost(M) likely shifts the tree K\*-optimum deeper.
4. A per-layer-type attention attribution (the #39 attention-detail profiler on the post-split-KV stack) would confirm the 3.03×-vs-4.38× gap is the FA2-sliding / already-3D-M=1 layers, and quantify any residual Triton-path verify-attention headroom (likely small).
