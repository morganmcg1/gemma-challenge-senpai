# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~05:35Z (cycle 40)
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

## Current 8-seat roster (cycle 40)

| student | PR | lever | status |
|---|---|---|---|
| **land** | **#71** | **Tree-verify build (THE #1 lever, ~576 TPS projected)** | WIP |
| stark | #78 | Drafter GEMM/pass fusion (~+2.6% ceil) | WIP — check-in posted 04:14Z |
| ubel | #84 | SplitK W4A16 verify-GEMM (~+5–12%) | WIP |
| kanna | **#96** | **Network-wide greedy-compounding gate:** do per-layer ≤1-ULP perturbations compound to flip argmax on the composed land#71×ubel#84 frontier? (closes #87's named upstream residual) | WIP |
| lawine | **#99** | **Local→official projection calibration + tree-A/B harness readiness for land #71** (Morgan assigned) | WIP (#90 ✅ MERGED — K=7 confirmed, 454.338 locked) |
| fern | **#95** | **Drafter loss-objective gate:** is the MTP draft head acceptance-optimal or only likelihood-optimal? (LK-Loss headroom) (Morgan assigned) | WIP |
| wirbel | **#98** | **fp32 star-attn cost-erosion gate:** does the #93-mandated fp32 accumulation erode the tree's +18.2%? (Morgan assigned) | WIP (#93 🔴 RED MERGED → land #71 needs fp32-accum) |
| denken | **#97** | **Persistent-kernel overhead gate:** is the ~32% "other" bucket GPU-idle (megakernel-reclaimable, +8-15%) or GPU-busy (#65 extended)? CPU-first | WIP (#94 ✅ MERGED, overlap CLOSED) |

## Land #71 de-risking status (THE #1 lever, ~576 TPS projected)

Build the multi-candidate tree-verify serving path with M=32, depth-9, max-branch-3 topology:
- **Topology:** max-branch-3 CONFIRMED empirically +0.9614% E[T] over max-branch-4 (fern #91, three independent estimators). **Build array:** `[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]`
- **Verify-GEMM free to M≤32:** bandwidth-bound (77.1% HBM at M=8), flat to M≤32, hard tile cliff at M=33 (denken #68).
- **Overhead:** non-GEMM tree machinery 2.597% of decode (denken #85); ~8× smaller than gain; net +19.82% / ~576 official projected.
- **E[T] / acceptance rule:** root-to-leaf standard verify is correct; Traversal Verification adds zero under greedy (fern #88 RED, structural proof).
- **Static topology fully pinned:** entropy-gated dynamic tree closes (oracle +0.27% E[T], sign-reversed, within-step — explains #83's flat per-depth ρ₂; wirbel #86). Uniform max-branch-3 (E[T]=5.207) is at the STRUCTURAL LIMIT for any static topology.
- **FP numerics:** GEMM argmax-margin gate ✅ GREEN/MERGED (kanna #87): M-widen M=16 bit-identical, M=32 0-flip, SplitK 0-flip — land #71 M-width is DIRECT GREEN to quota. SplitK = ONLY remaining live verify-GEMM kernel lever (denken #85 side-finding: KV shared, mask~0, BW-bound).
- **Salvage oracle (debug gate):** ρ₂=0.4165 at divergence steps (wirbel #83). Byteshark broken tree = 0.033 (12× gap = layout bug).
- **⚠️ SALVAGE-COLLAPSE ROOT CAUSE (chiku-inu, ~04:18Z board, relayed to land #71):** the 0.033 salvage signature = having only ONE of two required halves wired — (1) star-attention DISPATCH installed for tree rows **AND** (2) the fused reject/salvage WALK called on the tree layout. Every prior broken run missed exactly one half. **Runtime double-assert required:** assert (a) star-attn is the dispatched path AND (b) salvage walk is the rejection code path, before any quota spend. Chiku-inu's both-halves-wired package is being oracle-benched by openevolve.
- **E[T] assumption (fern #92 ✅ GREEN/MERGED):** realized tree E[T]=5.20824 under real correlated draws = independent model to **+0.025%**; gap within [−1.8%, +2.3%] across 3 cross-checks. Correlation is strong (r=−0.97) but E[T]-neutral. ~568 projection STANDS (carry ±2–3% band → 558–581). **Last analytical assumption DE-RISKED.** Only true channel-4 test = land's first tree `accept_length` run.
- **Attention greedy-equivalence (wirbel #93 ✅ MERGED — 🔴 RED):** the relerr-1e-3 star-attention path FLIPS 0.59% of greedy tokens (noise floor 0/65,536; 82% genuine fp32-propagation, only 18% bf16-readout). **HARD PRE-QUOTA REQUIREMENT: land #71 must run the star-attention reduction (softmax·V / o_proj) in fp32 accumulation** (not just a higher-precision readout) before any quota — the near-tie tail (~4.8% of positions) needs ~bit-exactness (≲1e-6). Re-verify with `scripts/profiler/star_attn_greedy_gate.py` against the kernel's measured relerr. fp32-accum COST now gated by **wirbel #98**; margin-gated salvage (recompute only the ~4.8% near-tie positions) is the fallback if blanket-fp32 dents the projection (caveat: residual-stream propagation tail may defeat a purely local detector).
- **Network-wide compounding (kanna #96 WIP):** closes #87's named residual — do the per-layer ≤1-ULP perturbations compound across ~30 layers to flip the final argmax on the composed land#71×ubel#84 frontier? Integration test consuming wirbel #93 + kanna #87 per-op GREENs.
- **Pre-quota gate status:** GEMM ✅ (kanna #87 MERGED) · attention 🔴 RED→land needs fp32-accum (wirbel #93 MERGED; cost-erosion gated by wirbel #98) · network-wide compounding 🔄 (kanna #96 WIP). All ANALYTICAL assumptions closed (tree-economics lane saturated: #88/#86/#91/#92); the live pre-quota work is now the **fp32-accum BUILD requirement + its TPS cost**.

## Cycle 39–40 — lanes closed / confirmed this session

| PR | student | verdict | significance |
|---|---|---|---|
| #93 | wirbel | Star-attn greedy-equiv 🔴 RED: relerr-1e-3 flips 0.59% greedy tokens (noise floor 0; 82% fp32-propagation) | **land #71 MUST add fp32 accum on star-attn reduction before quota** (cost gated by #98) |
| #90 | lawine | MTP K sweep: clean inverted-U, K=7 optimal (K6 −0.72%, K8 −3.09%); E[accept] monotone 3.49→4.08 | draft-length CLOSED; **454.338 locked as land #71's linear-chain reference** |
| #94 | denken | Draft-verify overlap AMBER: naive +18% → +4.22% BW-limited → +1.2-2.9% realized; A10G bus serializes 2 streams (contention 0.506) | overlap lane CLOSED (single-GPU conc=1); banks bus_contention_factor=0.506 |
| #87 | kanna | Argmax-margin GREEN: verify-GEMM SplitK + M-widen 0/65,536 flips; 98.13% provably flip-proof; M=16 bit-identical | pre-quota GEMM numerics gate CLEARED (1 of 2) |
| #92 | fern | E[T] independence GREEN: realized 5.20824 = independent +0.025%; corr strong but E[T]-neutral | tree-economics-analytics CLOSED (de-risk) |
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
- **Draft-verify stream overlap (single-GPU conc=1):** A10G bus serializes the two memory-bound streams (contention 0.506); +1.2-2.9% realized, sub-build-bar (#94). Re-opens only on a 2nd GPU or a compute-bound regime.

## Next directions (priority order)

1. **Land #71 completes tree-verify build — now WITH fp32 accumulation on the star-attention reduction** (wirbel #93 RED requirement) → openevolve oracle bench → lawine served A/B (#99 readies the harness + projection calibration) → approval-gated HF submission. ~576 projected, less the fp32-accum cost (wirbel #98 is measuring whether it erodes the +18.2%). Pre-quota gates: GEMM ✅ (kanna #87); attention 🔴 → **fp32-accum REQUIRED** (wirbel #93); network-wide compounding (kanna #96) pending.
2. **Persistent-kernel overhead-reclamation (denken #97):** the ~32% "other/overhead" is the biggest un-mined bucket; a megakernel could reclaim GPU-idle/launch/host slack (+8-15% IF idle). Crux: tested against denken's own #65 (decode 99.41% GPU-bound) — is the 32% GPU-idle (reclaimable, GREEN) or GPU-busy small-kernel-tail/bus-spillover (#65 extended, CLOSE)? CPU-first gate. _[Draft-verify overlap CLOSED: denken #94 AMBER — the A10G bus serializes the two streams → +1.2-2.9% realized, sub-build-bar.]_
3. **SplitK W4A16 verify-GEMM (ubel #84):** close 23% HBM gap at M=8; ~+5–12% wall_tps.
4. **MTP K=7 CONFIRMED (lawine #90 ✅ MERGED):** clean inverted-U on the robust runner; 454.338 locked as land #71's linear-chain reference; no free config win (±4.4% fragile-estimator caveat retired). _[lawine → #99 projection-calibration + tree-A/B harness.]_
5. **Drafter GEMM fusion (stark #78):** ~+2.6% ceiling once stark responds to check-in.
6. **Composition:** tree-verify × SplitK × draft-verify-overlap — orthogonal levers that multiply.
7. **Next-wave levers (queued for freed seats):** double-quant verify-GEMM scales (+0.4–2%, compounds with ubel SplitK), QuantSpec INT4 drafter-KV (+10–15% claim — premise-check first: MTP may lack a separate drafter KV to quantize), Token Recycling (+4–10%; likely #89-redundant). _[Now in-flight: LK-loss draft head → fern #95; persistent-kernel scheduling → denken #97.]_
