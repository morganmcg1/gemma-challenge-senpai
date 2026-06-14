# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~12:42Z (cycle 49)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## Strategic Snapshot

**We sit at 481.53 TPS (public #8), at frontier *parity* with the linear-stack leaders — the gap to #1 is noise, not a lever.** stark #137 (CLOSED ⚪) confirmed the "8-TPS gap to #1" is **best-of-N official-scorer variance (~1.9%), not a real deficit** — frantic-penguin's own 3-draw spread (489.63 / 483.80 / 480.41) brackets our 481.53. block64 (`FUSED_SPARSE_ARGMAX_BLOCK=16→64`) is a **+0.085% NULL lever**, adopted as a free config line in land's tree manifest (provably greedy/PPL-safe), NOT a launch. ⇒ **500 is a tree-path E[T] story (land #71's descent), not a linear micro-opt story.** kanna #138 re-checks K* at the block64 operating point. The memory-BW micro-lever surface is fully exhausted; a researcher-agent sweep (Plateau Protocol) is in flight for genuinely independent levers.

**THE DECISIVE FINDING THIS CYCLE — the tree's 500-lever is localized.** The oracle has now *run* (openevolve `tree-488-pw-fp32-v0`, board `20260614-100550-487`): **measured E[T]=2.621 → ~271 official (FAILS 500).** Two independent methods now **double-confirm** that the **descending accept walk (BUG-2)** — not the depth-1 spine — is the whole 500-game:
- **fern #134** (official-TPS matrix): descent-only fix → **522 ✅** (E[T] 5.056, clears 500 *even with the depth-1 spine left broken*); spine-only → 283 ❌; both → 538.
- **wirbel #135** (independent E[T]-DP): descent-only → **E[T] 5.041 ✅**; spine-only → 2.746 ❌. **BUG-2 / BUG-1 = 19.3×.** The 391/1024 salvages fire but contribute only +0.077 E[T] (2.9%) — they do not descend.

⇒ **Fix the accept walk and the tree clears 500; the depth-1 spine (0.679 vs 0.7287) is only the 522→538 margin.** Localized on our stack (land #71): `sitecustomize.py` `_dixie_fused_accept_prep_kernel` is a strictly-linear break-on-mismatch chain-reject reused on a flat tree → branches unreachable → ~3% salvage. **land is building the descending replacement now (descent FIRST, spine second).**

**Step-time denominator is FIRM** (lawine #136): measured depth-9 step = **1.2182 (+0.45% vs the 1.2127 roofline)** — the eager attn-launch idle is hidden behind per-layer GEMM. Operative clear-500 bar moves only **4.841 → 4.862**. The binding lever is the **numerator** (descent), not the step.

## Official Standing

| rank | agent | TPS | status |
|---:|---|---:|---|
| 1 | frantic-penguin | 489.63 | valid |
| 2 | need-for-speed | 488.07 | valid |
| 4 | byteshark | 484.62 | valid |
| **8** | **senpai** | **481.53** | **valid** |

- **Our official best:** 481.53 TPS (`fa2sw_precache_kenyan`, PR #52, lawine), private-verified VALID 2026-06-13 23:04Z (PPL 2.3777, 128/128, gap Δ4.3% ≤ 5%)
- **Target:** >500 TPS (~+3.8% needed); past-530 = tree-only (descent fix → 522, both bugs → 538)
- **No linear-stack reclaim remains:** the gap to #1 is best-of-N scorer variance (stark #137 CLOSED — frantic-penguin's own spread brackets us); the only path past parity is the **tree** (land #71 descent → 522, both bugs → 538).

## Human Directives (Standing)

- **lewtun (Issue #31):** downstream evals must use `generation_config.json` params — NOT greedy. Does NOT apply to TPS benchmark or greedy-identity gate.
- **theykk:** "target is 500tps."
- **No HF job launches without human approval.** Open GitHub issue titled `Approval request: HF job for <name>`, launch only after explicit human go. (Validity is no longer the blocker — launch *authorization* is.)
- **Advisor consumes no GPU.** All GPU usage via assigned students only.

## Greedy-Identity Decision (Issue #124) — RESOLVED by advisor ruling

Per human directive (theykk 10:30Z: "come to your own decision, then flag it to the broader challenge collective and monitor if other agents disagree"), the operative interpretation: **the greedy-identity contract binds but is SATISFIED** by greedy-exact int4 speculative decoding + the PPL ≤ 2.42 gate. The deployed stack's 56% AR-divergence (kanna #114/#122) is inherent int4-Marlin batch-variance (split-K geometry = f(M), no batch-invariant Marlin in the pinned wheel), **not** a contract violation; the official auto-scorer (`hf_bucket_single_job.py`) runs **no token-identity check** (TPS + PPL + 128/128 only). The tree rides on the **same int4-spec basis as the 481.53 frontier**. Socialized on the board (decision `20260614-104624-532`); **no dissent has appeared** — it STANDS as operative. The official shot is still approval-gated, but on *launch authorization*, not validity.

## Active 8-Seat Roster (Cycle 48) — ZERO IDLE

| student | PR | lever | status |
|---|---|---|---|
| **land** | **#71** | **Tree descent-walk BUILD — ★ CRITICAL-PATH / the 500-lever.** Replace the linear `_dixie_fused_accept_prep_kernel` chain-reject with a **descending** accept walk on the flat tree (BUG-2). Build descent FIRST (clears 500 alone per fern #134); depth-1 spine is the 522→538 margin. Gate: local M=16 ≈ 5.0 tok/step with branch-hit ≈ ρ₂=0.4165, both-halves asserts. | **WIP** (build team holds the runnable star-attn build) |
| **stark** | **#151** | **Tree private-acceptance gap** (parallel-advisor assigned): does the descent-walk E[T] survive the ~4.3% public→private acceptance gap — i.e. does the 522/538 projection hold on the private prompt distribution? Stress-tests the 500 verdict against distribution shift. | **WIP** |
| **kanna** | **#138** | **K-sweep with block64**: faster verify from block64 may shift K-optimal above 7. Sweep K=6..9, pick K* for the next HF shot. | **WIP** |
| **denken** | **#150** | **Tree-submission local preflight harness** (NEW): validate the assembled tree stack against the scorer's 3 hard gates — boots/serves, PPL≤2.42, 128/128 — emit READY/NOT-READY *before* the one shot. Self-validates against the known-good 481.53 linear stack + an injected fault. The **validity** leg of the launch evidence-line. (#144 MERGED 🔴: verify GEMM is full-12288 LIVE but candidate-prune net-slower 2.1× + greedy-unsafe — lane closed.) | **WIP** (NEW) |
| **fern** | **#149** | **Joint (spread×width) clears-500 frontier** (NEW): upgrade #145's two 1-D recovery slices into the 2-D (λ spread × μ width) decision surface, so land's partial-both-facet landing point gets a *correct* GO/NO-GO (not one assuming the unrecovered facet is at ρ-opt). The **decision-geometry** leg. (#145 MERGED 🟢: deep-spine-spread IS the 161.6-TPS swing; ≥90% spread recovery to clear 500.) | **WIP** (NEW) |
| **wirbel** | **#152** | **Topology re-opt** (parallel-advisor assigned): given the now-measured ladder [0.674, 0.350, 0.203, …], is the M=32/depth-9/max-branch-3 build budget (pinned *before* the oracle ran) E[T]-optimally allocated — does re-allocating clear 530? (#146 MERGED 🟢: gate CI self-test passes; required_n=5 for a robust 500 verdict; 522 GREEN holds until sampling ½-width > 4.19%.) | **WIP** |
| **ubel** | **#154** | **Step-denominator reduction audit** (NEW): decode-path `[M,262144]` scatter avoidance (greedy-safe argmax-only — researcher RANK-1, token-identical) + CUDA-graph launch-overhead capture → recoverable step% → clear-500-bar reduction (insurance on fern #145's ≥90% spread-recovery risk). The **denominator** leg, multiplicative with the descent. (#148 MERGED 🟢: K_cal transfers to the tree, band 0.787% one-sided↓; `PRECACHE_BENCH=1` named as the calibration dependency.) | **WIP** (NEW) |
| **lawine** | **#153** | **Verify-step(M) cost curve** (parallel-advisor assigned): is the depth-9 verify step **flat in M** up to some M_crit (free tree-growth headroom), or does step(M) rise — extends the single M=32 point #136 measured into a curve. (#147 MERGED 🟢: sync-audit harness — the sync-free constraint is now verifiable the instant land's kernel lands.) | **WIP** |

## Current Research Themes

### Theme 1: Linear-Stack Parity — CLOSED as a lever (kanna #138 K* check only)
stark #137 (CLOSED ⚪) settled it: block64 is a **+0.085% NULL**, and the "8-TPS gap to #1" is **best-of-N scorer variance (~1.9%)** — frantic-penguin's own 3-draw spread (489.63 / 483.80 / 480.41) brackets our 481.53. We are at frontier *parity* on the linear stack; **no linear-stack reclaim remains**. block64 is adopted as a free, greedy/PPL-safe config line in land's tree manifest (no dedicated launch). kanna #138 still validates K* at the block64 operating point for the eventual tree shot. All linear micro-opt energy now redirects to the tree (Theme 2) and bulletproofing the one shot (Theme 3).

### Theme 2: Tree Descent-Walk Build (THE 500-PATH — land #71 + support seats)
Tree is the sole 500+ path (tree-free caps 491.8, denken #123 RED). The decisive finding: **descent (BUG-2) alone clears 500** (fern #134 + wirbel #135, two independent methods). land #71 builds the descending accept walk; the support instruments are now mostly **banked**: **lawine #147** (sync-audit harness — MERGED 🟢, verifies the sync-free build + measures the real step; lawine → #153), **wirbel #146** (measured-gate confidence envelope — MERGED 🟢, sampling CIs + required_n=5; wirbel → #152). Still live: **fern #149** (joint spread×width clears-500 frontier — the 2-D decision surface; #145 MERGED 🟢 established deep-spine-spread IS the 161.6-TPS swing, ≥90% spread recovery to clear 500) and a new **denominator** support seat **ubel #154** (decode-path scatter avoidance + CUDA-graph capture → lowers the clear-500 bar). **BUG-1 depth-1 is now CLOSED-and-handed (denken #133 MERGED):** fp32 GPU-confirmed not-the-fix (lane closed, tree-488-pw-fp32-v0 quota saved); the 13.1pp deficit is ~96% a fixable wrong-rank `target_logits_indices` plumbing bug → handed to land as the rank-1 spine fix, and the **same index-map class may fix BOTH the depth-1 spine AND the descent**. BUG-3 (CUDA-graph illegal-memory-access) lives in the external star-attn build → re-homed to land #71 / build team (ubel #139 closed as out-of-scope for an isolation-scoped seat).

### Theme 3: Launch-Readiness / One-Shot Evidence-Line Cluster (#142/#145/#146/#147/#148 banked; fern #149 + denken #150 live)
The memory-BW micro-lever surface is now **fully exhausted** — weights floored at int4 (kanna #132), verify-GEMM weight-BW maxed (#117/#130/#108), scale bytes palette-harvested (#110) with coarser-grouping closed (ubel #140 RED), the bf16 KV-cache lever closed (wirbel #141 RED), and the verify-candidate column-prune closed (denken #144 RED). With no independent BW lever and the drafter-acceptance lane owned by the parallel team, the freed seats built **bulletproofing for the one human-approved 500 shot** — for a one-shot irreversible spend, heavy verification is rational. The legs of the eventual `Approval request: HF job` evidence-line (three now MERGED-and-banked):
- **#146 (wirbel, MERGED 🟢)** — *sampling* leg: CIs + required-N (=5) on land's measured E[T]; the 522 GREEN survives until the sampling ½-width alone exceeds **4.19%**. wirbel → #152.
- **#148 (ubel, MERGED 🟢)** — *calibration* leg: K_cal=125.268 **transfers** to the tree (band 0.787% one-sided↓); the residual +6.019% is a hardware bus ratio held invariant by **`PRECACHE_BENCH=1`** (named launch dependency — the tree submission MUST retain it). ubel → #154.
- **#147 (lawine, MERGED 🟢)** — *measurement* leg: sync-audit harness verifies land honored the sync-free rule (PASS→bar ~4.88 / FAIL→names syncs) + produces the real measured step. lawine → #153.
- **fern #149 (LIVE)** — *decision-geometry* leg: the 2-D (spread×width) joint clears-500 frontier for land's partial-recovery landing point.
- **denken #150 (LIVE)** — *validity* leg: local preflight that proves the assembled tree stack boots, passes PPL≤2.42, and completes 128/128 *before* we spend the shot (catches a #141-class crash or a PPL regression).

Plus fern #142 (scalar point gate) + #145 (facet decomp), and now a **denominator leg (ubel #154, NEW)** — decode-path scatter avoidance + CUDA-graph capture *lowers the clear-500 bar*, multiplicative with the descent. **Honest note:** the **Plateau-Protocol researcher-agent sweep has RETURNED** (`research/RESEARCH_IDEAS_2026-06-14_12:20.md`): it *confirmed* the tree-descent premise (TTFT/prefill ruled out — <1% at output_len 512) and surfaced the genuine independent path as **step-denominator reductions** — RANK-1 scatter-free verify argmax (`kept_ids[argmax(partial[M,12k])]`, validated token-identical, equivalence_rate=1.0) is now ubel #154. The concentration on one number (land #71's) was justified by the one-shot structure; the sweep pressure-tested it and produced a real orthogonal lever rather than more instrumentation.

### Theme 4: Closed Lanes (DO NOT RE-ASSIGN)
- **GEMM-bandwidth** — triple-confirmed closed (#117 roofline + #130 re-tile + #108 SplitK: Marlin 79.4% = 95% of the 1-wave HBM wall).
- **Sub-4-bit weight body** — closed (kanna #132: no servable sub-4-bit GEMM kernel on sm_86; Marlin floors weights at 4 bits).
- **QuantSpec drafter-KV** — moot (stark #121: the MTP drafter is Q-only, allocates zero KV).
- **fp8 KV-cache (a10g-small / sm_86)** — closed (wirbel #141 RED: e4m3 `fp8e4nv` hardware-impossible on sm_86; e5m2 software-blocked by vLLM #39137 compressed-tensors guard, + FA_SLIDING `NotImplementedError` PR #14221). The KV-read BW stream is un-attackable via fp8 on this hardware. DO NOT re-propose.
- **Marlin coarser-group scale-BW** — closed (ubel #140 RED: g=256 unservable [pinned Marlin max group 128]; the only coarser servable group g=-1 fails PPL at +0.122 = 3× the 0.039 cap headroom). The scale-byte slice is already palette-harvested losslessly (#110). Conditional re-open only if a wheel bump dispatches g=256 (delta ≈ +0.06 might fit).

## Tree Build Status (land #71) — the 500-lever

- **Topology pinned:** M=32, depth-9, max-branch-3 build array (per wirbel #83 / denken #101).
- **Measured oracle (openevolve `20260614-100550-487`):** E[T]=2.621, depth-1 accept 0.679 (target 0.7287), per-position cumulative ladder [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037], 391 salvages / 37 full / 1024 steps, eager star-attn PIECEWISE.
- **Recovery (fern #134 / wirbel #135):** descent-only fix → E[T] 5.04–5.06 → ~522 ✅; both bugs → 5.207 → ~538.
- **Step denominator (lawine #136):** measured depth-9 step 1.2182 (+0.45% vs roofline); operative clear-500 bar 4.841→4.862, well under the 5.207 ceiling.
- **Build sequence:** descending accept walk FIRST (clears 500 alone), then the depth-1 spine for the 522→538 margin. land's local gate: M=16 ≈ 5.0 tok/step, branch-hit ≈ ρ₂=0.4165.
- **Launch:** approval-gated HF job only, behind a human-approved `Approval request: HF job` issue.

## Recent Closed/Merged (cycles 47–49)

| PR | verdict | significance |
|---|---|---|
| #148 | MERGED GREEN (ubel, K_cal tree-transfer) | K_cal=125.268 **transfers** to the tree — band **0.787% one-sided↓** (far inside the 3%-drift tripwire). The +6.019% local→official multiplier decomposes to a hardware bus ratio held invariant by `PRECACHE_BENCH=1` (named launch dependency); calibration is NOT the binding 500-boundary constraint (522 GREEN holds until sampling > 4.19%). Flagged fern #142's τ-floor optimism (0.9983 SplitK vs 0.9924 tree-class). ubel → #154. |
| #147 | MERGED GREEN (lawine, sync-audit harness) | Extends #143's profiler with `--trace`/`--self-test`; classifies sync-free (0 syncs, bar 4.880, PASS) vs sync-bound (4.24 syncs/step, bar 4.956, FAIL, names 4 sites). The sync-free constraint is now verifiable the instant land's kernel lands. lawine → #153. |
| #146 | MERGED GREEN (wirbel, gate confidence envelope) | Wraps fern #142's point gate with bootstrap CIs: oracle 2.621→[253,286] robust-RED, ρ-opt 5.207→[507,564] robust-GREEN, boundary 4.862→[472,528] INDETERMINATE; **required_n=5** for a robust 500 verdict at the oracle point. wirbel → #152. |
| #145 | MERGED GREEN (fern, width-vs-spread decomp) | Splits realized tree E[T] into 3 TPS facets; the 161.6-TPS watched swing is ~entirely **deep-spine-spread** (nested +160.9 / Shapley +151.1). ≥90% spread recovery (width restored) to clear 500. Names the binding facet on a sub-GREEN ladder. Advisor confirmed the width/spread naming. fern → #149 (2-D joint frontier). |
| #144 | MERGED RED (denken, lm_head verify-candidate) | Verify lm_head GEMM is full-12288 (LIVE — 8192-sparse is the *drafter*). Candidate-prune NO-GO: gather-GEMM 2.1× slower than dense int4 Marlin at M=8 + argmax over 263 cands can't certify true argmax (breaks greedy/PPL). Verifier-side active-vocab closed. denken → #150 (preflight). |
| #137 | CLOSED NULL (stark, block64 reclaim) | block64 = +0.085% NULL. Valuable finding: the "8-TPS gap to #1" is best-of-N scorer variance (~1.9%) — frantic-penguin's own spread brackets us → we're at linear-stack *parity*, 500 is a tree story. block64 folded into land's manifest as a free config line; no launch. stark → reassign (Plateau sweep). |
| #142 | MERGED GREEN (fern, go/no-go gate) | One-call measured-M16 → official verdict; self-tests both anchors (271 RED / 538 GREEN within 0.1%) + bit-matches #134 matrix; wires PPL/tok-floor/branch-hit/greedy preconditions. Armed for land's readout. Decision input only — no launch authority. fern → #145. |
| #133 | MERGED GREEN (denken, depth-1 root-cause) | fp32 GPU-confirmed NOT the fix (NET ~0pp, exact-tie reshuffles) → fp32 lane CLOSED, quota saved; 13.1pp depth-1 deficit = ~96% fixable `target_logits_indices` plumbing (rank-2 contamination reproduces 0.598), handed to land as the rank-1 fix; shared-index-map may fix BOTH bugs. denken → #144. |
| #136 | MERGED AMBER (lawine, step-anchor) | Measured depth-9 step 1.2182 (+0.45% vs roofline); eager attn idle hidden behind GEMM; root-row clears 530; operative bar 4.841→4.862. Denominator FIRM. Methodology catch: isolation-only bench would have falsely read RED. |
| #135 | MERGED GREEN (wirbel, BUG-2 E[T]-DP) | Independent confirmation: descent-only fix → E[T] 5.041 (clears bar alone); BUG-2/BUG-1 = 19.3×. Salvages fire but don't descend (+0.077 E[T]). |
| #134 | MERGED GREEN (fern, live oracle readout) | Official-TPS matrix: as-built 270.7 ❌ / spine-only 283 ❌ / **descent-only 522 ✅** / both 538. Descent is the decisive 500-lever. |
| #132 | CLOSED (kanna, Q-Palette) | No servable sub-4-bit GEMM kernel on sm_86/vLLM-0.22; weight body floored at int4. Sub-4-bit weight lane closed. |
| #131 | MERGED AMBER (lawine, fp32 step-time) | Selective root-row-only fp32 → 563 TPS central, clears 530; full upcast compute-exposed at M=32. |
| #130 | MERGED RED (wirbel, gate_up re-tile) | 0% benefit — 1-wave HBM saturation wall. Final proof of GEMM-BW closure. |
| #129 | MERGED AMBER (fern, oracle harness) | Harness armed; operative clear-500 bar = E[T] 4.841 at the depth-9 step; fed the live oracle run. |
| #128 | MERGED RED (denken, fp32 cross-check) | fp32 is NOT the depth-1 fix (closes ≤1.4pp of 13.1pp). Real cause = drafter-spine/index-map under LIVE masking. |
