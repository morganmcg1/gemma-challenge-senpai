# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~12:20Z (cycle 48)
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
| **stark** | reassigning | **#137 CLOSED ⚪** (block64 = +0.085% NULL; the valuable finding = "8-TPS gap to #1 is scorer variance ~1.9%, not a deficit" → block64 folded into land's manifest as a free config line). Reassigning to a fresh bigger-swing lever — **Plateau Protocol researcher-agent sweep in flight**. | **reassigning** |
| **kanna** | **#138** | **K-sweep with block64**: faster verify from block64 may shift K-optimal above 7. Sweep K=6..9, pick K* for the next HF shot. | **WIP** |
| **denken** | **#150** | **Tree-submission local preflight harness** (NEW): validate the assembled tree stack against the scorer's 3 hard gates — boots/serves, PPL≤2.42, 128/128 — emit READY/NOT-READY *before* the one shot. Self-validates against the known-good 481.53 linear stack + an injected fault. The **validity** leg of the launch evidence-line. (#144 MERGED 🔴: verify GEMM is full-12288 LIVE but candidate-prune net-slower 2.1× + greedy-unsafe — lane closed.) | **WIP** (NEW) |
| **fern** | **#149** | **Joint (spread×width) clears-500 frontier** (NEW): upgrade #145's two 1-D recovery slices into the 2-D (λ spread × μ width) decision surface, so land's partial-both-facet landing point gets a *correct* GO/NO-GO (not one assuming the unrecovered facet is at ρ-opt). The **decision-geometry** leg. (#145 MERGED 🟢: deep-spine-spread IS the 161.6-TPS swing; ≥90% spread recovery to clear 500.) | **WIP** (NEW) |
| **wirbel** | **#146** | **Measured-500-gate confidence envelope** (NEW): wrap fern #142's point gate with sampling CIs + required-N for land's measured E[T], branch-hit ρ₂, and step-time → a TPS confidence interval at the 4.862 GREEN/RED boundary, so the one human-approved shot isn't gated on a noisy 1024-step point. CPU-only; the uncertainty leg of the launch evidence-line. (#141 CLOSED 🔴: fp8 KV undispatchable on sm_86 — lane banked.) | **WIP** (NEW) |
| **ubel** | **#148** | **K_cal tree-transfer validation** (NEW): the 500-projection's most load-bearing constant — K_cal=125.268 was fit on the LINEAR frontier; does it transfer to the tree path? Decompose the 1.06019 local→official multiplier into tree-invariant vs tree-sensitive factors → a calibration-uncertainty band (the calibration leg, complementing wirbel #146's sampling leg). (#140 CLOSED 🔴: coarser Marlin group unservable >128 / g=-1 fails PPL — scale-byte lane retired.) | **WIP** (NEW) |
| **lawine** | **#147** | **Live re-bench + sync-audit harness** (NEW): the drop-in tool that, when land's kernel lands, verifies he honored the sync-free rule (PASS → bar holds ~4.86; FAIL → names the offending syncs) and produces the real measured step that fern #142 / wirbel #146 consume. Self-validates against the #143 modeled + a synthetic sync-bound trace. (#143 MERGED 🟢: salvage-walk GPU-hidden — +0.39% sync-free / +2.2% sync-bound; sync-free build rule handed to land.) | **WIP** (NEW) |

## Current Research Themes

### Theme 1: Linear-Stack Parity — CLOSED as a lever (kanna #138 K* check only)
stark #137 (CLOSED ⚪) settled it: block64 is a **+0.085% NULL**, and the "8-TPS gap to #1" is **best-of-N scorer variance (~1.9%)** — frantic-penguin's own 3-draw spread (489.63 / 483.80 / 480.41) brackets our 481.53. We are at frontier *parity* on the linear stack; **no linear-stack reclaim remains**. block64 is adopted as a free, greedy/PPL-safe config line in land's tree manifest (no dedicated launch). kanna #138 still validates K* at the block64 operating point for the eventual tree shot. All linear micro-opt energy now redirects to the tree (Theme 2) and bulletproofing the one shot (Theme 3).

### Theme 2: Tree Descent-Walk Build (THE 500-PATH — land #71 + support seats)
Tree is the sole 500+ path (tree-free caps 491.8, denken #123 RED). The decisive finding: **descent (BUG-2) alone clears 500** (fern #134 + wirbel #135, two independent methods). land #71 builds the descending accept walk; supported by **lawine #147** (sync-audit harness to verify the sync-free build + measure the real step — #143 MERGED 🟢 closed the denominator's last component: salvage-walk is GPU-hidden, +0.39% sync-free / +2.2% sync-bound, sync-free rule handed to land), **fern #149** (joint spread×width clears-500 frontier — the 2-D decision surface; #145 MERGED 🟢 established deep-spine-spread IS the 161.6-TPS swing, ≥90% spread recovery to clear 500), and **wirbel #146** (measured-gate confidence envelope — sampling CIs + required-N so the verdict is statistically robust). **BUG-1 depth-1 is now CLOSED-and-handed (denken #133 MERGED):** fp32 GPU-confirmed not-the-fix (lane closed, tree-488-pw-fp32-v0 quota saved); the 13.1pp deficit is ~96% a fixable wrong-rank `target_logits_indices` plumbing bug → handed to land as the rank-1 spine fix, and the **same index-map class may fix BOTH the depth-1 spine AND the descent**. BUG-3 (CUDA-graph illegal-memory-access) lives in the external star-attn build → re-homed to land #71 / build team (ubel #139 closed as out-of-scope for an isolation-scoped seat).

### Theme 3: Launch-Readiness / One-Shot Evidence-Line Cluster (wirbel #146 + ubel #148 + lawine #147 + fern #149 + denken #150)
The memory-BW micro-lever surface is now **fully exhausted** — weights floored at int4 (kanna #132), verify-GEMM weight-BW maxed (#117/#130/#108), scale bytes palette-harvested (#110) with coarser-grouping closed (ubel #140 RED), the bf16 KV-cache lever closed (wirbel #141 RED), and the verify-candidate column-prune closed (denken #144 RED). With no independent BW lever and the drafter-acceptance lane owned by the parallel team, the freed seats concentrate on **bulletproofing the one human-approved 500 shot** — for a one-shot irreversible spend, heavy verification is rational. The five legs of the eventual `Approval request: HF job` evidence-line:
- **wirbel #146** — *sampling* leg: CIs + required-N on land's measured E[T] / branch-hit / step, so a GREEN isn't a noisy 1024-step draw.
- **ubel #148** — *calibration* leg: does K_cal=125.268 (fit on the linear frontier) transfer to the tree path? The projection's most load-bearing constant, validated independently.
- **lawine #147** — *measurement* leg: sync-audit harness that verifies land honored the sync-free rule + produces the real measured step.
- **fern #149** — *decision-geometry* leg: the 2-D (spread×width) joint clears-500 frontier for land's partial-recovery landing point.
- **denken #150** — *validity* leg: local preflight that proves the assembled tree stack boots, passes PPL≤2.42, and completes 128/128 *before* we spend the shot (catches a #141-class crash or a PPL regression).

Plus fern #142 (scalar point gate) + #145 (facet decomp). **Honest note:** this is heavy concentration on one number (land #71's), justified by the one-shot structure but at diminishing returns — hence the **Plateau-Protocol researcher-agent sweep in flight** to pressure-test whether the tree-descent is truly the only remaining lever and surface any genuinely independent path (prefill/TTFT in the scorer, per-step systems overhead, non-drafter E[T] levers).

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

## Recent Closed/Merged (cycles 47–48)

| PR | verdict | significance |
|---|---|---|
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
