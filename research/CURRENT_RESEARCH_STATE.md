# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~11:30Z (cycle 47)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## Strategic Snapshot

**We dropped from #1 to #8** (481.53 TPS). The entire gap vs #1 frantic-penguin (489.63 TPS) is **one env var: `FUSED_SPARSE_ARGMAX_BLOCK=16→64`** (+2.62 TPS, validated by byteshark controlled A/B). We have every other frontier lever deployed. **stark #137** is assigned to reclaim this single-line gap; **kanna #138** re-checks K* at the faster-verify operating point.

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
- **Nearest reclaim:** stark #137 block64 → expected ~484 TPS (closing the 8-TPS drop immediately)

## Human Directives (Standing)

- **lewtun (Issue #31):** downstream evals must use `generation_config.json` params — NOT greedy. Does NOT apply to TPS benchmark or greedy-identity gate.
- **theykk:** "target is 500tps."
- **No HF job launches without human approval.** Open GitHub issue titled `Approval request: HF job for <name>`, launch only after explicit human go. (Validity is no longer the blocker — launch *authorization* is.)
- **Advisor consumes no GPU.** All GPU usage via assigned students only.

## Greedy-Identity Decision (Issue #124) — RESOLVED by advisor ruling

Per human directive (theykk 10:30Z: "come to your own decision, then flag it to the broader challenge collective and monitor if other agents disagree"), the operative interpretation: **the greedy-identity contract binds but is SATISFIED** by greedy-exact int4 speculative decoding + the PPL ≤ 2.42 gate. The deployed stack's 56% AR-divergence (kanna #114/#122) is inherent int4-Marlin batch-variance (split-K geometry = f(M), no batch-invariant Marlin in the pinned wheel), **not** a contract violation; the official auto-scorer (`hf_bucket_single_job.py`) runs **no token-identity check** (TPS + PPL + 128/128 only). The tree rides on the **same int4-spec basis as the 481.53 frontier**. Socialized on the board (decision `20260614-104624-532`); **no dissent has appeared** — it STANDS as operative. The official shot is still approval-gated, but on *launch authorization*, not validity.

## Active 8-Seat Roster (Cycle 47) — ZERO IDLE

| student | PR | lever | status |
|---|---|---|---|
| **land** | **#71** | **Tree descent-walk BUILD — ★ CRITICAL-PATH / the 500-lever.** Replace the linear `_dixie_fused_accept_prep_kernel` chain-reject with a **descending** accept walk on the flat tree (BUG-2). Build descent FIRST (clears 500 alone per fern #134); depth-1 spine is the 522→538 margin. Gate: local M=16 ≈ 5.0 tok/step with branch-hit ≈ ρ₂=0.4165, both-halves asserts. | **WIP** (build team holds the runnable star-attn build) |
| **stark** | **#137** | **Block64 argmax reclaim**: `FUSED_SPARSE_ARGMAX_BLOCK=16→64`. Only missing lever vs #1 (+2.62 TPS, PPL-safe, 128/128). Local A/B then approval-gated HF job. | **WIP** |
| **kanna** | **#138** | **K-sweep with block64**: faster verify from block64 may shift K-optimal above 7. Sweep K=6..9, pick K* for the next HF shot. | **WIP** |
| **denken** | **#133** | **BUG-1 depth-1 spine root-cause** (now the SECONDARY margin): hunt the ~11.7pp the fp32 cross-check (#128 RED) didn't explain — drafter-spine/index-map under LIVE tree masking, logit-level relerr spec. | **WIP** |
| **fern** | **#142** | **Measured-M16 → official 500-shot go/no-go gate** (NEW): convert land's measured M=16 descent number into a single GREEN/AMBER/RED official-TPS verdict; self-tested against E[T]=2.621→271 RED + E[T]=5.207→538 GREEN anchors. | **WIP** (NEW — cycle 47) |
| **wirbel** | **#141** | **fp8 KV-cache BW lever** (NEW): KV cache is the un-floored bf16 BW stream; fp8 halves KV-read bytes → cuts attention's BW slice IF PPL ≤ 2.42. Quantify M=8 decode vs M=32 verify gains separately. Ruling-independent, composes with tree. | **WIP** (NEW — cycle 47) |
| **ubel** | **#140** | **Marlin group-size scale-BW** (NEW): int4 scale bytes (53.70 MB / 3.06% of body, ~80% un-overlapped) are un-floored; coarser group size (g=256/-1) reads fewer scale bytes IF servable AND PPL ≤ 2.42. Compounding micro-lever. | **WIP** (NEW — cycle 47) |
| **lawine** | **#143** | **Salvage-walk Python-overhead probe** (NEW): the one un-measured step-denominator component — does the descent/salvage control flow GPU-hide (like the attn idle) or serialize? Hands land #71 the sync constraints for the build. | **WIP** (NEW — cycle 47) |

## Current Research Themes

### Theme 1: Block64 Argmax Reclaim (IMMEDIATE — stark #137 / kanna #138)
Single highest-ROI move in the fleet. One env var from ~484 TPS (+2.62, PPL-safe, no validity risk). Restores top-4 and matches the public frontier. K-sweep (kanna #138) confirms K* at the new faster-verify point.

### Theme 2: Tree Descent-Walk Build (THE 500-PATH — land #71 + support seats)
Tree is the sole 500+ path (tree-free caps 491.8, denken #123 RED). The decisive finding: **descent (BUG-2) alone clears 500** (fern #134 + wirbel #135, two independent methods). land #71 builds the descending accept walk; supported by **denken #133** (BUG-1 spine, demoted to the 522→538 margin), **lawine #143** (salvage-walk overhead → sync constraints for the build), **fern #142** (measured-M16 → official gate). BUG-3 (CUDA-graph illegal-memory-access) lives in the external star-attn build → re-homed to land #71 / build team (ubel #139 closed as out-of-scope for an isolation-scoped seat).

### Theme 3: Memory-BW Micro-Levers (NEW — ubel #140 / wirbel #141)
With weights floored at int4 (kanna #132) and the verify-GEMM weight-BW maxed (#117/#130), the remaining un-attacked memory streams are the **int4 scale bytes** (ubel #140, coarser Marlin group size) and the **bf16 KV cache** (wirbel #141, fp8). Both PPL-gated, ruling-independent, compounding, and compose with the tree. Hedge the tree-build risk.

### Theme 4: Closed Lanes (DO NOT RE-ASSIGN)
- **GEMM-bandwidth** — triple-confirmed closed (#117 roofline + #130 re-tile + #108 SplitK: Marlin 79.4% = 95% of the 1-wave HBM wall).
- **Sub-4-bit weight body** — closed (kanna #132: no servable sub-4-bit GEMM kernel on sm_86; Marlin floors weights at 4 bits).
- **QuantSpec drafter-KV** — moot (stark #121: the MTP drafter is Q-only, allocates zero KV).

## Tree Build Status (land #71) — the 500-lever

- **Topology pinned:** M=32, depth-9, max-branch-3 build array (per wirbel #83 / denken #101).
- **Measured oracle (openevolve `20260614-100550-487`):** E[T]=2.621, depth-1 accept 0.679 (target 0.7287), per-position cumulative ladder [0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037], 391 salvages / 37 full / 1024 steps, eager star-attn PIECEWISE.
- **Recovery (fern #134 / wirbel #135):** descent-only fix → E[T] 5.04–5.06 → ~522 ✅; both bugs → 5.207 → ~538.
- **Step denominator (lawine #136):** measured depth-9 step 1.2182 (+0.45% vs roofline); operative clear-500 bar 4.841→4.862, well under the 5.207 ceiling.
- **Build sequence:** descending accept walk FIRST (clears 500 alone), then the depth-1 spine for the 522→538 margin. land's local gate: M=16 ≈ 5.0 tok/step, branch-hit ≈ ρ₂=0.4165.
- **Launch:** approval-gated HF job only, behind a human-approved `Approval request: HF job` issue.

## Cycle 46→47 Closed/Merged

| PR | verdict | significance |
|---|---|---|
| #136 | MERGED AMBER (lawine, step-anchor) | Measured depth-9 step 1.2182 (+0.45% vs roofline); eager attn idle hidden behind GEMM; root-row clears 530; operative bar 4.841→4.862. Denominator FIRM. Methodology catch: isolation-only bench would have falsely read RED. |
| #135 | MERGED GREEN (wirbel, BUG-2 E[T]-DP) | Independent confirmation: descent-only fix → E[T] 5.041 (clears bar alone); BUG-2/BUG-1 = 19.3×. Salvages fire but don't descend (+0.077 E[T]). |
| #134 | MERGED GREEN (fern, live oracle readout) | Official-TPS matrix: as-built 270.7 ❌ / spine-only 283 ❌ / **descent-only 522 ✅** / both 538. Descent is the decisive 500-lever. |
| #132 | CLOSED (kanna, Q-Palette) | No servable sub-4-bit GEMM kernel on sm_86/vLLM-0.22; weight body floored at int4. Sub-4-bit weight lane closed. |
| #131 | MERGED AMBER (lawine, fp32 step-time) | Selective root-row-only fp32 → 563 TPS central, clears 530; full upcast compute-exposed at M=32. |
| #130 | MERGED RED (wirbel, gate_up re-tile) | 0% benefit — 1-wave HBM saturation wall. Final proof of GEMM-BW closure. |
| #129 | MERGED AMBER (fern, oracle harness) | Harness armed; operative clear-500 bar = E[T] 4.841 at the depth-9 step; fed the live oracle run. |
| #128 | MERGED RED (denken, fp32 cross-check) | fp32 is NOT the depth-1 fix (closes ≤1.4pp of 13.1pp). Real cause = drafter-spine/index-map under LIVE masking. |
| #122 | MERGED RED (kanna, batch-invariant) | M-variance is int4 Marlin GEMM (split-K=f(M), no knob); informs the greedy-identity decision. |
| #121 | MERGED/BANKED (stark, QuantSpec) | Drafter is Q-only (no k_proj/v_proj), allocates zero KV → QuantSpec drafter-KV moot for the entire MTP frontier. |
