# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~05:00Z (cycle 39)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## Frontier

- **Official best:** 481.53 TPS (`fa2sw_precache_kenyan`, PR #52, lawine) — private-verified VALID 2026-06-13 23:04Z (460.85 private Δ4.3% ≤ 5%, PPL 2.3777, 128/128). **Target: >500 TPS (~+3.8% needed).**
- **Local baseline:** wall_tps = 454.09 (PR #82, median N=3, CV 0.007%). Official metric = `summary.json:tps`, greedy, 128 prompts, output_len 512, a10g-small.
- **Decode budget (M=8 linear-MTP):** verify-GEMM ~53% (int4 W4A16 Marlin, weight-BW-bound, FLAT M≤32, HARD tile cliff M=33), drafter ~7%, attention <8% (at irreducible conc=1 floor), **~32% other/overhead** (host–device scheduling, Python round-trips — largely un-mined).
- **Greedy-identity gate (HARD):** any emitted-token change vs plain greedy AR is DISQUALIFYING. All speed levers must be provably lossless.

## Human directives (standing)

- **lewtun (Issue #31):** downstream quality evals must use `generation_config.json` params — NOT greedy. Does NOT apply to the TPS benchmark or greedy-identity gate.
- **theykk:** "target is 500tps."
- **No HF job launches without human approval.** Open a GitHub issue titled `Approval request: HF job for <name>`, launch only after explicit human go.
- **Advisor consumes no GPU.** Keep usage to the assigned student GPU.

## Current 8-seat roster (cycle 39)

| student | PR | lever | status |
|---|---|---|---|
| **land** | **#71** | **Tree-verify build (THE #1 lever, ~576 TPS projected)** | WIP |
| stark | #78 | Drafter GEMM/pass fusion (~+2.6% ceil) | WIP — check-in posted 04:14Z |
| ubel | #84 | SplitK W4A16 verify-GEMM (~+5–12%) | WIP |
| kanna | #87 | Argmax-margin greedy-safety gate (protects SplitK + tree) | WIP |
| lawine | #90 | MTP K sweep (confirm K=7 optimal, empirical A/B) | WIP |
| fern | **#92** | **Tree E[T] independence-gap:** realized E[T] under real correlated drafter draws vs the independent DP model — the last untested assumption behind +18.2% | WIP (Morgan assigned) |
| wirbel | **#93** | **Star-attention greedy-equivalence gate:** does the tree-mask numerical path preserve greedy argmax? attention-side twin of kanna #87's GEMM gate | WIP (Morgan assigned) |
| denken | — | **Persistent-kernel scheduling** (target ~32% overhead, +8–15%; nsys profile first) | **being assigned** |

## Land #71 de-risking status (THE #1 lever, ~576 TPS projected)

Build the multi-candidate tree-verify serving path with M=32, depth-9, max-branch-3 topology:
- **Topology:** max-branch-3 CONFIRMED empirically +0.9614% E[T] over max-branch-4 (fern #91, three independent estimators). **Build array:** `[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]`
- **Verify-GEMM free to M≤32:** bandwidth-bound (77.1% HBM at M=8), flat to M≤32, hard tile cliff at M=33 (denken #68).
- **Overhead:** non-GEMM tree machinery 2.597% of decode (denken #85); ~8× smaller than gain; net +19.82% / ~576 official projected.
- **E[T] / acceptance rule:** root-to-leaf standard verify is correct; Traversal Verification adds zero under greedy (fern #88 RED, structural proof).
- **Static topology fully pinned:** entropy-gated dynamic tree closes (oracle +0.27% E[T], sign-reversed, within-step — explains #83's flat per-depth ρ₂; wirbel #86). Uniform max-branch-3 (E[T]=5.207) is at the STRUCTURAL LIMIT for any static topology.
- **FP numerics:** argmax-margin gate in flight (kanna #87 WIP). SplitK = ONLY remaining live verify-GEMM kernel lever (denken #85 side-finding: KV shared, mask~0, BW-bound).
- **Salvage oracle (debug gate):** ρ₂=0.4165 at divergence steps (wirbel #83). Byteshark broken tree = 0.033 (12× gap = layout bug).
- **⚠️ SALVAGE-COLLAPSE ROOT CAUSE (chiku-inu, ~04:18Z board, relayed to land #71):** the 0.033 salvage signature = having only ONE of two required halves wired — (1) star-attention DISPATCH installed for tree rows **AND** (2) the fused reject/salvage WALK called on the tree layout. Every prior broken run missed exactly one half. **Runtime double-assert required:** assert (a) star-attn is the dispatched path AND (b) salvage walk is the rejection code path, before any quota spend. Chiku-inu's both-halves-wired package is being oracle-benched by openevolve.
- **E[T] assumption (fern #92 WIP):** the +18.2% projection assumes independent drafter draws (DP model); fern #92 measures whether real correlated drafter draws match the model — the last untested assumption before quota.
- **Attention greedy-equivalence (wirbel #93 WIP):** does the tree-mask numerical path (validated externally to relerr 1e-3) preserve greedy argmax? — attention-side twin of kanna #87's GEMM gate.

## Cycle 39 — lanes closed / confirmed this session

| PR | student | verdict | significance |
|---|---|---|---|
| #91 | fern | Topology CONFIRMED: mb3 +0.9614% E[T] > mb4; acceptance model validated to ~1e-3 tok | topology-analytics CLOSED |
| #86 | wirbel | Entropy-branching: r=−0.9688 (sign-reversed, within-step); oracle +0.27% E[T] — non-actionable | entropy-branching + dynamic-tree CLOSED |
| #89 | denken | Prompt-lookup augment: +1.67% gross, structural redundancy (corr=+0.354) — DROP | prompt-lookup-augment CLOSED |
| #88 | fern | Traversal Verification RED: provably zero under greedy (structural) | acceptance-rule CLOSED |
| #82 | lawine | Infra keeper: paired-A/B runner + re-baseline 454.09 | measurement infra |
| #85 | denken | Tree non-GEMM overhead 2.597%; SplitK = ONLY live verify-GEMM kernel lever | overhead audit + kernel ruling |

## Confirmed dead ends (comprehensive)

- **CUDA-graph:** decode 99.41% GPU-bound; ≈0 launch-overhead headroom (#65).
- **Decode norm/elementwise fusion:** ceiling <0.5% (#67).
- **Attention at M=8:** at irreducible conc=1 floor; #43 harvested 4.38× (#69).
- **Traversal/leaf-to-root acceptance:** zero under greedy, structural (#88).
- **Entropy-conditioned / dynamic-tree branching:** oracle +0.27% E[T], sign-reversed (#86).
- **Prompt-lookup augment:** +1.67% gross, structural corr ceiling +2.38% (#89).
- **EAGLE-3 drafter training:** CLOSED — MTP parity (HASS +57% offline ≪ MTP, #80).
- **MBT > 512:** small REAL regressions; MBT=512 optimal (#56/#82).
- **KV-layout / fused-mask / tile-scheduler at M≤32:** ruled out by #85 (KV shared, mask~0, BW-bound).
- **int4 drafter GEMM refitted:** ~+5% realistic ceiling (#75).
- **Drafter non-GEMM subtasks:** no addressable headroom (#77).
- **Verify-rollback, strict M=1 greedy-valid spec:** cost theorem / official gate has no identity check (#24/#38).

## Next directions (priority order)

1. **Land #71 completes tree-verify build** → openevolve oracle bench → lawine served A/B → approval-gated HF submission. ~576 TPS projected. Both pre-quota gates (kanna #87 GEMM, wirbel #93 attention) must clear first.
2. **Persistent-kernel scheduling (denken, cycle 39):** nsys profile gate — if inter-kernel gap ≥5% of step, persistent GPU loop eliminates host–device round-trips; +8–15% TPS ceiling. Targets the ~32% uncharacterized overhead budget.
3. **SplitK W4A16 verify-GEMM (ubel #84):** close 23% HBM gap at M=8; ~+5–12% wall_tps.
4. **MTP K sweep (lawine #90):** confirm K=7 optimal empirically or find a free config win.
5. **Drafter GEMM fusion (stark #78):** ~+2.6% ceiling once stark responds to check-in.
6. **Composition:** tree-verify × SplitK × persistent-kernel — orthogonal levers that compound.
7. **Next-wave levers (researcher-agent, cycle 39 — `research/RESEARCH_IDEAS_2026-06-14_04:40.md`):** LK-loss draft head (+8% E[T]), double-quant verify-GEMM scales (+0.4–2%) — reserved for next freed seats after fern/wirbel complete #92/#93.
