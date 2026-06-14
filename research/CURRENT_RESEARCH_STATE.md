# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~04:50Z (cycle 39)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## Frontier

- **Official best:** 481.53 TPS (`fa2sw_precache_kenyan`, PR #52, lawine) — private-verified VALID 2026-06-13 23:04Z (460.85 private Δ4.3% ≤ 5%, PPL 2.3777, 128/128). **Target: >500 TPS (~+3.8% needed).**
- **Local baseline:** wall_tps = 454.09 (PR #82, median N=3, CV 0.007%). Official metric = `summary.json:tps`, greedy, 128 prompts, output_len 512, a10g-small.
- **Decode budget (M=8 linear-MTP):** verify-GEMM ~53% (int4 W4A16 Marlin, weight-BW-bound, FLAT M≤32, HARD tile cliff M=33), drafter ~7%, attention <8% (at irreducible conc=1 floor, #43 already harvested), **~32% other/overhead** (host–device scheduling, Python round-trips — largely un-mined).
- **Greedy-identity gate (HARD):** any emitted-token change vs plain greedy AR is DISQUALIFYING (internal contract). All speed levers must be provably lossless.

## Human directives (standing)

- **lewtun (Issue #31):** downstream quality evals must use `generation_config.json` params — NOT greedy. Does NOT apply to the TPS benchmark or greedy-identity gate.
- **theykk:** "target is 500tps."
- **No HF job launches without human approval.** Open a GitHub issue titled `Approval request: HF job for <name>`, launch only after explicit human go.
- **Advisor consumes no GPU.** Keep usage to the assigned student GPU.

## Current 8-seat roster (cycle 39)

| student | PR | lever | status |
|---|---|---|---|
| **land** | **#71** | **Tree-verify build (THE #1 lever, ~576 TPS projected)** | WIP |
| stark | #78 | Drafter GEMM/pass fusion (~+2.6% ceil) | WIP — Morgan check-in posted 04:14Z |
| ubel | #84 | SplitK W4A16 verify-GEMM (~+5–12%) | WIP |
| kanna | #87 | Argmax-margin greedy-safety gate (protects SplitK + tree) | WIP |
| lawine | #90 | MTP K sweep (confirm K=7 optimal, empirical A/B) | WIP |
| denken | — | **Persistent-kernel scheduling** (target ~32% overhead, +8–15% if inter-kernel gap ≥5%; nsys profile first) | being assigned |
| wirbel | — | **Double-quant verify-GEMM scales** (lossless HBM, INT8 round-trip scan first, +0.4–2%) | being assigned |
| fern | — | **LK-loss draft-head fine-tuning** (+8% E[T] via acceptance-rate loss; analytical E[T] ceiling first) | being assigned |

## Land #71 de-risking status (the #1 lever)

Build the multi-candidate tree-verify serving path with the M=32, depth-9, max-branch-3 topology:
- **Topology:** max-branch-3 CONFIRMED empirically +0.9614% E[T] over max-branch-4 (fern #91, three independent estimators). **Build array:** `[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]`
- **Verify-GEMM free to M≤32:** weight-BW-bound (77.1% HBM at M=8), flat to M≤32, hard tile cliff at M=33 (denken #68 MERGED).
- **Overhead:** non-GEMM tree machinery 2.597% of decode (denken #85 MERGED); ~8× smaller than gain; net +19.82% / ~576 official projected.
- **Acceptance model validated:** `score_tree_depthrank` ≈ MC to ~1e-3 tok (fern #91 MERGED); future DP topology results trusted analytically.
- **E[T] / acceptance rule:** root-to-leaf standard verify is correct; Traversal Verification adds zero under greedy (fern #88 MERGED, RED, structural proof).
- **FP numerics:** argmax-margin gate in flight (kanna #87); SplitK is the ONLY remaining live verify-GEMM kernel lever (KV shared, mask~0, BW-bound — denken #85 side-finding).
- **Salvage oracle (debug gate):** ρ₂=0.4165 at each divergence step; byteshark broken = 0.033 (12× gap = layout bug); per-op cost-budget oracle delivered (denken #85).

## Cycle 39 — lanes closed this session

| PR | student | verdict | significance |
|---|---|---|---|
| #91 | fern | Topology CONFIRMED: mb3 +0.9614% E[T] > mb4, acceptance model validated | topology-analytics CLOSED |
| #86 | wirbel | Entropy-branching: r=−0.9688 (sign-reversed), oracle +0.27% E[T] — non-actionable | entropy-branching CLOSED |
| #89 | denken | Prompt-lookup augment: +1.67% gross, structural redundancy (corr=+0.354) | prompt-lookup-augment CLOSED |
| #88 | fern | Traversal Verification RED: provably zero under greedy (structural) | acceptance-rule CLOSED |
| #82 | lawine | Infra keeper: paired-A/B runner + re-baseline 454.09 | measurement infra |
| #85 | denken | Tree non-GEMM overhead 2.597%; SplitK = ONLY live verify-GEMM kernel lever | overhead audit + kernel ruling |

## Confirmed dead ends (all cycles)

- **CUDA-graph:** decode 99.41% GPU-bound, ≈0 launch-overhead headroom (#65).
- **Decode norm/elementwise fusion:** ceiling <0.5% (#67).
- **Attention at M=8:** at irreducible conc=1 floor; #43 already harvested 4.38× (#69).
- **Traversal/leaf-to-root acceptance:** zero under greedy, structural (#88).
- **Entropy-conditioned branching:** oracle +0.27% E[T], non-actionable (#86).
- **Prompt-lookup augment:** +1.67% gross, structural corr ceiling +2.38% (#89).
- **EAGLE-3 drafter training:** CLOSED — MTP parity (E[T] 3.844, HASS +57% offline ≪ MTP, #80).
- **MBT > 512:** small REAL regressions; deployed MBT=512 optimal (#56/#82).
- **KV-layout / fused-mask / tile-scheduler at M≤32:** ruled out by #85 (KV shared, mask~0, BW-bound).
- **int4 drafter GEMM refitted:** ~+5% realistic ceiling, too small (#75).
- **Drafter non-GEMM subtasks:** no addressable headroom (#77).
- **MAX_NUM_BATCHED_TOKENS > 512:** small real regressions (#56/#82).
- **Verify-rollback, strict M=1 greedy-valid spec:** cost theorem / official gate has no identity check (#24/#38).

## Next directions (priority order)

1. **Land #71 completes tree-verify build** → served measurement (openevolve oracle first, then lawine A/B, then approval-gated HF submission). ~576 TPS projected.
2. **Persistent-kernel scheduling (denken, cycle 39):** nsys profile gate — if inter-kernel gap ≥5% of step, persistent GPU loop eliminates host–device round-trips; +8–15% TPS ceiling. Biggest fresh lever.
3. **LK-loss draft-head fine-tuning (fern, cycle 39):** E[T] analytical ceiling first (measure current P(accept|k) on validation set); if +8% E[T] projected, proceed to fine-tuning run. ~+8% TPS.
4. **Double-quant verify-GEMM scales (wirbel, cycle 39):** INT8 round-trip exactness scan on weight scale tensors; if >98% bit-exact, lossless +0.4–2% HBM bandwidth. CPU-analyzable first.
5. **SplitK W4A16 verify-GEMM (ubel #84):** close 23% HBM gap at M=8; ~+5–12% wall_tps if kernel phase succeeds.
6. **Composition:** tree-verify × SplitK × persistent-kernel — orthogonal levers that compound.
7. **Researcher ideas (cycle 39, `research/RESEARCH_IDEAS_2026-06-14_04:40.md`):** three fresh levers surfaced: persistent-kernel scheduling (+8–15%), double-quant scales (+0.4–2%), LK-loss draft head (+8%). All assigned.
