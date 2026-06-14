# PR #154 — Step-denominator reduction audit: decode-path scatter avoidance + CUDA-graph launch overhead

**Verdict: AUDIT COMPLETE. Two greedy-safe, NON-drafter step_time (denominator)
levers quantified.**
**PRIMARY `step_reduction_audit_self_test_passes = 1` (8/8).**
**TEST `recoverable_step_pct = 0.86% (conservative) / 1.11% (realistic)`** (tree
M=32 @ the clear-500 bar). LOCAL A10G profiling + analysis only. No HF Job, no
submission, no served-file change. **BASELINE stays 481.53 (PPL 2.3777).** This
bounds a lever + hands a build design; it does **not** authorize a launch.

## TL;DR

| leg | finding | step headroom | build status |
|---|---|---|---|
| **1. decode-path [M,262144] scatter + LP avoidance** | **REAL, greedy-safe, eager (outside the graph)** | **~0.8–1.3% (cons) / ~1.0–1.8% (real)** across E[T]; ~flat **+3.6 to +5.6 TPS** | hand to land/build — the remaining denominator lever |
| **2. CUDA-graph capture of the tree step** | **ALREADY DONE in the deployed stack** (ONEGRAPH+LOOPGRAPH+verify cudagraph; #136 `gemm_all_graphed=true`) | **~0 additional** (461/500 launches already captured) | CLOSED; residual (39 data-dependent launches) PENDING land's #147 trace |

The fleet has attacked the **E[T] numerator** (the descent). This audit prices the
**denominator**. Leg 1 is a real ~1% greedy-safe lever that **STACKS** with the
descent and **lowers the clear-500 bar** by ~0.04–0.08 in E[T] (4.862 → ~4.81) —
direct insurance on the fern #145 ≥90%-spread-recovery risk. Leg 2 is already
realized in the baseline; it is **not** new headroom (do not double-count it
against the 1.2182 step, which is measured *with* capture on).

Artifacts: `step_denominator_reduction_audit.{py,json}`,
`bench_launch_overhead.py`, `launch_overhead_graph_leg.json`.

## Public evidence used

- Leaderboard rank 1 (digest, 2026-06-14): `frantic-penguin`
  `osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache-skv64-v1` **489.63 TPS / PPL 2.3774**
  — this is the deployed lmhead12k + scatter stack being audited; confirms the
  scatter path is live on the frontier.
- denken #144 `research/lmhead_verify_audit/` int4-Marlin anchors (reproduced).
- my #148 `research/kcal_tree_transfer/` de-risked K_cal band (0.787% one-sided↓).
- #136 `tree_step_denominator.json` step anchor + `gemm_all_graphed=true`.

## Leg 1 — decode-path scatter + LogitsProcessor avoidance (the real lever)

### 1a. Decomposition (denken #144 anchors reproduced)

The deployed `compute_logits` (patched, `serve_patch_pck04.py:335-342`) does:
int4-Marlin lm_head GEMM → `[M,12288]`, then `_scatter_to_full_vocab` → `[M,262144]`,
then the sampling LogitsProcessor. denken #144's served-stack microbench split it
(M=8):

| stage (M=8) | denken µs | this audit |
|---|---|---|
| int4 Marlin GEMM → [8,12288] (**UNAVOIDABLE**) | 38.27 | BW-floor cross-check **32.86 µs** ≤ 38.27 ≤ 2× floor ✓ |
| scatter `index_copy_ [8,12288]→[8,262144]` | 8.15 | **fresh measure 8.6–8.8 µs (≤7.4% rel)** ✓ |
| full `compute_logits` (GEMM+scatter+LP) | 135.82 | LP-wrapper share = 135.82−38.27−8.15 = **89.40 µs** ✓ |

The **GEMM is unavoidable** (the argmax-only path still reads the `[M,12288]` head).
The **scatter (8.15) + LP (89.40) = 97.55 µs is avoidable on decode**: greedy token
selection only needs `target_argmax`, which equals `kept_ids[argmax(pruned[M,12288])]`
with no scatter and no sampling LP.

### 1b. Avoidable µs/step (gross / conservative / realistic)

| M | gross (scatter+LP removed) | net-conservative¹ | net-realistic² |
|---|---|---|---|
| **8 (linear)** | **97.5 µs** | 71.0 µs | 97.5 µs |
| **32 (tree)** | **111.9 µs** | 86.6 µs | 111.9 µs |

¹ net-conservative charges the FULL eager `argmax(pruned)+gather` (2 launches,
launch-overhead-inflated ~26 µs) as if it were pure net-new add-back.
² net-realistic = gross: the proposed argmax over `[M,12288]` **replaces** the
262144-argmax the stack does anyway (21× fewer elements → strictly cheaper), so
nothing is truly added. Truth ≥ realistic.

> Honest note: a *naive eager* pure-torch reproduction of the LP work over
> `[M,262144]` (cast+softcap+argmax) measures 160 µs (M=8) → 617 µs (M=32) — far
> **slower** than vLLM's fused 89–104 µs LP. So the eager number is a loose UPPER
> bound, not the estimate; the trustworthy avoidable is anchored to denken's real
> served LP. (At tree M=32 the scatter is also *transient-allocated*, not cached —
> `serve_patch_pck04.py:144,156` caches only M≤16 — so the tree's avoidable is if
> anything understated here.)

### 1c. Step-fraction + TPS effect (`official = K_cal·(E[T]/step)·τ`, K_cal=125.268)

per-output-token budget = 1/481.53 = **2076.7 µs**; step_abs(E[T]) = E[T]×budget.
recoverable_step_pct(E[T]) = avoidable / step_abs(E[T]). dTPS = official/(1−frac) −
official (≈ flat in E[T] because avoidable is per-step, fixed).

| E[T] | regime | rec_step% (cons) | rec_step% (real) | dTPS (real) | bar_new (real) |
|---|---|---|---|---|---|
| 2.6 | linear M=8 | 1.32% | 1.81% | — | 4.78 |
| 3.844 | linear/tree | ~0.89% | ~1.22% | — | — |
| **4.862 (bar)** | tree M=32 | **0.86%** | **1.11%** | **+5.6** | **4.808** |
| 5.207 (ceiling) | tree M=32 | 0.79% | 1.04% | +5.6 | 4.812 |

**Magnitude, honestly:** ~0.8–1.3% conservative, ~1.0–1.8% realistic — at the
**low-to-mid** of the PR's "~1–2.5%" estimate. Worth a ~**flat +3.6 (cons M=8) to
+5.6 (real M=32) TPS** that **stacks** with the descent, and **lowers the clear-500
bar** by ~0.04–0.08 in E[T]. **Not a standalone 500-clearer** — denominator
insurance, exactly as scoped.

### 1d. Greedy-safety + the seam land must guard

**Token-identity (proved):** `kept_ids[argmax(pruned[M,12288])] ≡ argmax(scatter[M,262144])`
at **equivalence_rate = 1.0** on real weights AND adversarial ties
(`lmhead12k_scatter_equiv.json`). Holds because `kept_ids` is strictly ascending,
so argmax's first-occurrence tiebreak picks the smallest kept-row = smallest
original vocab id = exactly what full-vocab argmax returns.

**Softcap-invariance (verified here, all M):** softcap `30·tanh(x/30)` is strictly
increasing ⇒ `argmax(softcap(x)) == argmax(x)`; the decode path need not even apply
softcap for token selection.

**scatter+LP is PREFILL/PPL-only:** `serve_patch_pck04.py:5-11` — the full-vocab
scatter exists so "Downstream sampler / **prompt_logprobs** sees full-vocab logits
with original token IDs." For **greedy decode** the *only* consumer of `[M,262144]`
is the argmax ⇒ removable. PPL is measured on the **prefill** `prompt_logprobs`
path, which keeps the full scatter unchanged.

**The seam:** `compute_logits` must branch on **token-selection vs prompt_logprobs**,
not on M. (a) greedy decode/verify argmax (linear M=8 / tree M=32) → argmax over
`[M,12288]` + `kept_ids` remap, **no scatter, no LP**. (b) prompt_logprobs /
non-greedy → keep `[M,262144]` scatter+LP. The existing M≤16 vs M>16 branch
(`serve_patch_pck04.py:140-144`) is only a buffer-caching proxy — **not** the guard
the lever needs.

**Orthogonality:** `compute_logits` runs **eagerly outside the CUDA graph**
(`serve_patch_pck04.py:17-20`) ⇒ Leg 1 is independent of Leg 2; no double-count.

## Leg 2 — CUDA-graph launch overhead (ALREADY CLOSED in the deployed stack)

Hypothesis #2 was "if the multi-launch tree step is NOT CUDA-graph-captured,
per-launch overhead is a step tax." **It IS captured:**

- **drafter propose loop** (K=7 width-1 iters): `sitecustomize.py` `_capture_graph`
  → `torch.cuda.CUDAGraph()`, `LOOPGRAPH_TARGET=vllm.v1.spec_decode.gemma4`,
  `ONEGRAPH=1`, `LOOPGRAPH_REQUIRE_CAPTURE=1`.
- **target verify forward** (M=32, 42 layers): vLLM cudagraph; #136 reports
  `gemm_all_graphed = true` (verify GEMMs launch-free).

**Measured per-launch overhead (A10G):** eager tiny-kernel launch **7.4 µs** vs
graph-replay **1.1 µs** ⇒ tiny kernels are launch-bound (the regime where launches
tax the step). **Modeled launch budget:** ~500 eager launches/step, of which
**~461 are already captured**; the **residual ~39** are the data-dependent
accept-walk + sampling glue.

**Residual headroom:** zero-overlap UPPER bound 39×7.4 µs = 289 µs (2.87% of the
bar-step) — but async launches overlap GPU exec, so the true tax is the GPU-idle
fraction only (illustrative at 80% overlap: ~0.57%), and the residual is mostly
**un-capturable** (data-dependent control flow). **Reliable conclusion: ~0
additional headroom.** The precise overlap-limited residual is **ARMED/PENDING
land's real launch trace** (reuses lawine #147 `--trace`).

## Self-test (PRIMARY = 8/8 pass)

1. GEMM 38.27 µs consistent with measured BW floor (32.86 µs ≤ 38.27 ≤ 2×). ✓
2. scatter reproduces denken 8.15 µs within tolerance (8.6–8.8 µs, ≤7.4%). ✓
3. LP-wrapper share positive (89.40 µs). ✓
4. avoidable µs + argmax-only finite & NaN-clean at all M. ✓
5. ordering 0 < net_conservative ≤ net_realistic = gross < full_compute_logits. ✓
6. softcap monotone: argmax(softcap(x))==argmax(x) at all M. ✓
7. bar-drop arithmetic: 2% step cut → 4.862×0.98 ≈ 4.765. ✓
8. K_cal·bar/step ≈ 500 (anchor closes: 499.96). ✓

## Build hand-off (to land / build-team)

**Lever:** decode-path argmax-only logits. Replace, on the greedy token-selection
path only, `scatter[M,262144] + LP + argmax_262144` with `argmax(pruned[M,12288]) →
kept_ids remap`. Keep the full scatter+LP on the prompt_logprobs (prefill PPL) path.

**Guard seam:** branch `compute_logits` on token-selection vs prompt_logprobs (NOT
on M). Cite `serve_patch_pck04.py:335-342` (scatter site) and `:10,140-144`
(prompt_logprobs consumer / decode-prefill split). The fused accept-prep
(`sitecustomize.py:927,945-951`) already consumes a `target_argmax` — feed it the
remapped argmax directly.

**Expected payoff:** ~flat **+3.6 to +5.6 TPS**, **stacks** with the descent,
**lowers the clear-500 bar** to ~4.81 — insurance on fern #145's ≥90%-recovery
risk. Greedy-exact (PPL untouched). Pairs with #136 (step anchor) + #147
(sync-audit) as the **denominator leg**.

**Do NOT pursue Leg 2 as new headroom** — it is already captured; only the
overlap-limited residual remains, PENDING land's trace.

## Method / scope

LOCAL A10G pure-torch microbench of the avoidable memory-bound work (scatter +
fp32 cast + softcap + argmax over 262144) vs the argmax-only replacement; int4
Marlin GEMM anchored to denken #144 + BW roofline cross-check; per-launch overhead
via eager-vs-graph replay; propagated through K_cal (#148 de-risked band). All
metrics NaN-clean. No HF Job / no submission / no served-file change / no baseline
move. Rides Issue #124 RESOLVED (greedy-exact, PPL ≤ 2.42 binding).
