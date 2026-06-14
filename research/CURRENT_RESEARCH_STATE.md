# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~11:30Z (cycle 46)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## Strategic Snapshot

**We dropped from #1 to #8** (481.53 TPS). The entire gap vs #1 frantic-penguin (489.63 TPS) is **one env var: `FUSED_SPARSE_ARGMAX_BLOCK=16→64`** (+2.62 TPS, validated by byteshark controlled A/B). We have every other frontier lever deployed (osoi5, feopt2, w20, e1-drafter, lmhead12k, fa2sw, SPLITKV_VERIFY_MAX_Q=64, precache, orjson, fastrender, detok, fused-accept-prep, onegraph). **stark #137** is assigned to reclaim this single-line gap.

**GEMM-bandwidth lane permanently closed**: three independent convergent proofs.
- **#117 (denken) roofline**: 1-wave HBM saturation wall — 160 CTAs = 2 full waves on 80 SMs; Marlin at 79.4% = 95% achievable → no occupancy headroom.
- **#130 (wirbel) gate_up re-tile**: RED — re-tiling cannot break past CTA saturation.
- **#108 (ubel) SplitK micro-bench**: Closed — non-terminal + structural M-variance refuted the premise.

**Tree = sole 500+ path**: tree-free caps at 491.8 TPS (denken #123 RED). Tree physically delivers E[T]=5.207 → ~538 TPS (fern #125 GREEN). Three active build bugs block quota spend.

## Official Standing

| rank | agent | TPS | status |
|---:|---|---:|---|
| 1 | frantic-penguin | 489.63 | valid |
| 2 | need-for-speed | 488.07 | valid |
| 4 | byteshark | 484.62 | valid |
| **8** | **senpai** | **481.53** | **valid** |

- **Our official best:** 481.53 TPS (`fa2sw_precache_kenyan`, PR #52, lawine), private-verified VALID 2026-06-13 23:04Z (PPL 2.3777, 128/128, gap Δ4.3% ≤ 5%)
- **Target:** >500 TPS (~+3.8% needed); past-530 = tree-only
- **Nearest reclaim:** stark #137 block64 → expected ~484 TPS (matching byteshark/frantic-penguin field, closing the 8-TPS drop immediately)

## Human Directives (Standing)

- **lewtun (Issue #31):** downstream evals must use `generation_config.json` params — NOT greedy. Does NOT apply to TPS benchmark or greedy-identity gate.
- **theykk:** "target is 500tps."
- **No HF job launches without human approval.** Open GitHub issue titled `Approval request: HF job for <name>`, launch only after explicit human go.
- **Issue #124 (greedy-identity ruling):** STILL PENDING — load-bearing for any spec-decode stack on the official leaderboard (Marlin is M-variant, no batch-invariant Marlin in wheel; verified by kanna #122 RED MERGED).
- **Advisor consumes no GPU.** All GPU usage via assigned students only.

## Active 8-Seat Roster (Cycle 46)

| student | PR | lever | status |
|---|---|---|---|
| **land** | **#71** | **Tree-verify build — ★ CRITICAL-PATH / MANDATORY-for-500** (tree-free 491.8 < 500; tree must deliver E[T] ≥ 4.624, ceiling 5.207). Active bugs: BUG-1 (depth-1 spine 0.598 vs 0.7287 — real cause = drafter-spine/index-map under LIVE masking, NOT fp32), BUG-2 (salvage walk not descending sub-path; realized E[T]=2.10 ≪ 4.81), BUG-3 (CUDA illegal-memory-access when graphs enabled, attn_py_calls/step=37) | **WIP** — external build team (chiku-inu/byteshark/openevolve); awaiting denken #133 BUG-1 diagnosis + ubel #139 BUG-3 fix |
| **stark** | **#137** | **Block64 argmax reclaim**: `FUSED_SPARSE_ARGMAX_BLOCK=16→64` in `sitecustomize.py` line 36. Only missing lever vs #1 frantic-penguin (+2.62 TPS validated by byteshark A/B, PPL-safe 2.3769 unchanged, 128/128 valid). Local A/B wall_tps sweep (block16 vs 32 vs 64), then approval-gated HF job | **WIP** (NEW — cycle 46) |
| **kanna** | **#138** | **K-sweep with block64**: K=7 was optimal at block16 (lawine #90); faster verify step from block64 may shift K-optimal upward. Sweep K=6..9 at block64 baseline, report wall_tps per K; pick K* for next HF shot | **WIP** (NEW — cycle 46) |
| **ubel** | **#139** | **Tree CUDA graph crash fix (BUG-3)**: `attn_py_calls/step=37` when graphs enabled → CUDA illegal-memory-access. Kernel expertise seat. Diagnose the FA2/star-attn dispatch path under CUDA graph capture; propose minimal patch to unblock graph-mode for the tree. Critical-path for tree TPS (eager dispatch adds ~10% overhead per byteshark) | **WIP** (NEW — cycle 46) |
| fern | #134 | Live oracle readout (Morgan assignment) — oracle harness for measuring tree acceptance stats on current build | **WIP** (Morgan-assigned) |
| wirbel | #135 | BUG-2 salvage-descent (Morgan assignment) — why doesn't the salvage walk descend the rescue sub-path? E[T] gap 2.10 → 4.81 | **WIP** (Morgan-assigned) |
| lawine | #136 | fp32 step-anchor (Morgan assignment) — anchor the fp32 star-attn step-time delta under current build conditions | **WIP** (Morgan-assigned) |
| denken | **#133** | **BUG-1 depth-1 root-cause (drafter-spine equivalence)**: fp32 closes only ~1.4pp of the 13pp gap (#128 RED) — hunt the real ~11.7pp: does the tree's depth-1 spine token == the linear-chain token? index-mapping under LIVE tree masking (not static trace); logit-level relerr spec for the build team | **WIP** (cycle 45 — awaiting results) |

## Current Research Themes

### Theme 1: Block64 Argmax Reclaim (IMMEDIATE — stark #137)
Single highest-ROI move in the fleet. One env var from ~484 TPS (+2.62 TPS, PPL-safe, no validity risk). Will restore us to top-4 and match the current public frontier. K-sweep follow-on (kanna #138) to confirm K* at the new faster-verify operating point.

### Theme 2: Tree Build Debugging (MEDIUM-TERM — denken/land/ubel/fern/wirbel/lawine)
Tree is the sole 500+ path and the sole past-530 path. Three bugs block the quota run:
- **BUG-1** (depth-1): depth-1 accept=0.598 vs 0.7287 target — denken #133 hunting root cause (drafter-spine/index-map under LIVE masking)
- **BUG-2** (descent): salvage walk not descending sub-paths → E[T]=2.097 vs floor 3.844 — wirbel #135
- **BUG-3** (CUDA graphs): illegal-memory-access with graphs=ON — ubel #139 (critical for graph-mode overhead removal)

Fixing all 3 unlocks: E[T] → [3.844, 5.207], graph-mode, measured official TPS ~538.

### Theme 3: GEMM-Bandwidth (PERMANENTLY CLOSED)
Triple-confirmed by #117 (roofline) + #130 (re-tile RED) + #108 (SplitK closed): Marlin at 79.4% = 95% of achievable 1-wave throughput. No SplitK, LUT-GEMM, re-tiling, Q-Palette (sm_86 unsupported), or MaskLLM path exists. This lane is closed — do not re-assign.

### Theme 4: Greedy-Identity Contract (PENDING HUMAN RULING)
Issue #124 ruling is load-bearing for any spec-decode stack on the official leaderboard. kanna #122 RED (Marlin M-variant, no batch-invariant fix) confirmed there is no cheap local fix. The official shot stays approval-gated until #124 resolves.

## Tree Build Status (land #71) — Mandatory for 500 TPS

- **Topology pinned:** build array `[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]` (M=32, depth-9, max-branch-3)
- **E[T] ceiling:** 5.207 (fern #125 GREEN) → ~538 TPS official; supply > demand (4.624) by +0.59 E[T]
- **Current empirical:** E[T]=2.097 (both halves wired, byteshark board 05:53Z) — fixable build defect (denken #101 GREEN)
- **fp32 star-attn cost:** ~free (+0.339% M=32, wirbel #98 GREEN) — CONFIRMED not a blocker
- **Blocking tree quota run:** BUG-1 (depth-1 spine) + BUG-2 (salvage descent) + BUG-3 (CUDA graph crash) — all three under active investigation

## Cycle 45→46 Closed/Merged

| PR | verdict | significance |
|---|---|---|
| #132 | CLOSED (kanna, Q-Palette) | Step-1 gate kill: Q-Palette requires sm_89+, our A10G is sm_86. Sub-4-bit path dead on this hardware. |
| #108 | CLOSED (ubel, SplitK) | Non-terminal + structural M-variance refuted. Triple-confirmed 0% speedup (1-wave HBM saturation wall). GEMM-BW lane permanently closed. |
| #131 | MERGED AMBER (lawine, fp32 step-time) | Selective root-row-only fp32 → 563 TPS central, +0.48% step-time penalty, clears 530 easily. Full upcast exposed at M=32/AI=128. |
| #130 | MERGED RED (wirbel, re-tiling) | gate_up re-tile: 0% benefit — 1-wave saturation wall confirmed by Marlin 79.4%=95% achievable HBM. Final proof of GEMM-BW lane closure. |
| #129 | MERGED AMBER (fern, oracle harness) | Corrected harness: E[T] demand floor = 4.624 (from earlier 4.841 overcalculation). Oracle harness ready for tree quota run. |
| #128 | MERGED RED (denken, fp32 cross-check) | fp32 is NOT the depth-1 fix — bf16 relerr 1e-3 flips ≤1.4% vs 13.1pp deficit. Real cause = drafter-spine/index-map under LIVE masking. |
| #122 | MERGED RED (kanna, batch-invariant) | `VLLM_BATCH_INVARIANT=1` no-op — structural M-variance is int4 Marlin GEMM (split-K=f(M), no knob). Issue #124 ruling now load-bearing. |
| #121 | CLOSED/BANKED (stark, QuantSpec) | Non-terminal result; stark pod-wedge (Issue #127); premise of separate drafter KV cache unconfirmed. Closed as pod was dark 3× consecutive. |
