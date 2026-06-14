# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-14 ~06:30Z (cycle 40)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## Frontier

- **Official best:** 481.53 TPS (`fa2sw_precache_kenyan`, PR #52, lawine) — private-verified VALID 2026-06-13 23:04Z (460.85 private Δ4.3% ≤ 5%, PPL 2.3777, 128/128). **Target: >500 TPS (~+3.8% needed).**
- **Local baseline:** wall_tps = 454.338 (PR #90, lawine locked linear-chain reference, CV 0.001%). Official metric = `summary.json:tps`, greedy, 128 prompts, output_len 512, a10g-small.
- **Decode budget (M=8 linear-MTP):** verify-GEMM ~53% (int4 W4A16 Marlin, weight-BW-bound, FLAT M≤32, HARD tile cliff M=33), drafter ~7%, attention <8% (at irreducible conc=1 floor; **real-kernel M=32 attn = 1.83× M=8 = 8.12% of step, wirbel #98**), **~32% "other"** — re-labeled by denken #97 (MERGED): **~30pp is GPU-BUSY small-kernel tail** (attention/drafter/norm/sampling/lm_head/elementwise — a megakernel reorders but CANNOT remove; bus is the wall) + only **~2.17pp reclaimable GPU-idle**. Persistent-kernel/megakernel lane CLOSED.
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
| stark | **#103** | **QuantSpec drafter-KV premise-check:** does the MTP drafter expose a SEPARATE KV to quantize (lever live) or share verify's KV (moot)? CPU | WIP (#78 drafter-fusion CLOSED — 5h silent/zero-commit, repointed) |
| ubel | #84 | SplitK W4A16 verify-GEMM (~+5–12%) | WIP — ⚠️ DEADLINE: 3h20m silent, repoint at next review if no Phase-1 signal |
| kanna | **#96** | **Network-wide greedy-compounding gate:** do per-layer ≤1-ULP perturbations compound to flip argmax on the composed land#71×ubel#84 frontier? (closes #87's named upstream residual) | WIP |
| lawine | **#99** | **Local→official projection calibration:** pin the local-wall_tps→official multiplier + ready the tree-A/B harness so land #71's build is a zero-lag ≥500 decision | WIP (#90 ✅ MERGED: K=7 confirmed, 454.338 locked) |
| fern | **#102** | **Tree E[T] break-even / margin-of-safety:** invert the #100 model for the MIN accept_length that clears 500 (alone + per lever stack); place byteshark's 2.10 + denken #101's recoverable band on that axis | WIP (#100 ✅ MERGED 🟢 GREEN: tree-sufficient @ E[T]=5.207, but build gives 2.10 — caveat) |
| wirbel | **#104** | **Double-quant verify-GEMM scales:** INT8 scale-of-scales (QLoRA) + FP16 sparse exceptions, greedy-lossless ~0.4–1.1% byte lever on the #1 decode block. CPU round-trip bit-exact build-or-kill first (>98%→viable), then standalone scale-stream microbench. Scoped OFF ubel #84's kernel | WIP (#98 ✅ MERGED 🟢 GREEN: fp32 star-attn ~free, #93 constraint NOT load-bearing, tree +18.2% stands) |
| denken | **#105** | **Tree-free 500-path ceiling:** can the build-COMPLETE stack (SplitK #84 + LK #95 + double-quant #104, NO tree) clear 500, and at what SplitK threshold? go/no-go: tree critical-path vs insurance? CPU | WIP (#101 ✅ MERGED 🟢 GREEN: 2.10 = FIXABLE BUILD DEFECT not ceiling; ~568 survives) |

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
- **⚠️ FIRST EMPIRICAL TREE NUMBER (byteshark, ~05:53Z board — `tree-v2-merge-eager-v1`, 40m-timeout diagnostic):** both halves wired → **salvage jumped 0.033 → 0.358 per non-full step → ROOT CAUSE CONFIRMED.** BUT `tok/step=2.097` — BELOW the linear-MTP accept_length 3.844, far below analytical E[T]=5.207 (`full=164/14848`=1.1%; accept hist `[0,5761,5061,1765,854,355,214,126,200]` mean ~2.10). Blockers per byteshark, **NOT quota-ready:** (1) accept_length collapse (2.10 « 5.207), (2) eager-dispatch overhead (`attn_py_calls/step=37`, graph path needed), (3) fp32 star-attn still required (wirbel #93). **denken #101 RESOLVED the gap (✅ MERGED 🟢 GREEN): 2.10 is a FIXABLE BUILD DEFECT, not acceptance collapse** — it sits 1.74 tok BELOW the linear floor 3.844 that the same drafter+verifier already hit, so a correct tree (spine = linear chain) cannot accept less ⇒ ≥56.1% provably build-defect, 100% fixable, (D)-ceiling 0% (fern #92). Mechanism: depth-1 spine collapses to **0.598 vs required q[1]=0.7287** (verify/dispatch corrupts the spine before branch logic) + salvage walk recovers the rescue node but **doesn't descend its sub-path** (full-reach 1.10% vs 60.8% model). ρ-ladder sound and un-exercised. **Build hand-off (board 20260614-062536):** (1) assert depth-1==0.7287 = fastest localizer, (2) make the walk descend, (3) re-measure accept_length on the **fp32** star-attn path (NOT relerr eager — wirbel #93 separate blocker). Corrected band for fern #102 = **[3.844 floor, 5.207 ceiling], re-measure-pending** (don't plug 2.10 or 5.207 directly). Next BUILD gate (byteshark) = bounded fp32/bit-exact package w/ double-asserts (star-attn dispatched + reject/walk ran) + per-position branch-hit + perf vs ≤89us/≤452us budgets. **The analytical economics are clean (#85/#88/#86/#91/#92) AND the empirical gap is now diagnosed as fixable — the tree is no longer a single point of failure: denken #105 prices the tree-free 500-path (SplitK+LK+double-quant) as the build-slip backup.**
- **E[T] assumption (fern #92 ✅ GREEN/MERGED):** realized tree E[T]=5.20824 under real correlated draws = independent model to **+0.025%**; gap within [−1.8%, +2.3%] across 3 cross-checks. Correlation is strong (r=−0.97) but E[T]-neutral. ~568 projection STANDS (carry ±2–3% band → 558–581). **Last analytical assumption DE-RISKED.** Only true channel-4 test = land's first tree `accept_length` run.
- **⚠️ Attention greedy-equivalence (wirbel #93 ✅ MERGED — 🔴 RED → HARD CONSTRAINT):** relerr-1e-3 star-attn flips **0.59% of greedy tokens** (clean-vs-clean noise floor provably 0) → **land #71 MUST run star-attn accumulation in fp32 (or prove bit-exact) before quota; a relerr-1e-3 build is DISQUALIFYING.** Margin bimodal: safe bulk (median rel-margin 18%) + near-tie tail (0.537% < 1e-3 = the entire flip-risk set). Relayed to #71 as a hard gate. **The mandated-fp32 cost is now CONFIRMED ~free by wirbel #98 (✅ MERGED 🟢 GREEN): conservative +0.339% M=32 / +0.010% M=8, realized NEGATIVE (A10G L2-residency of fp32 partials, KV stays bf16); haircut 0.404pp → tree net +19.41%. fp32 is both SAFE and FREE — the #93 constraint is NOT load-bearing. Tail-only-fp32 hybrid NOT needed (the #93 margin map stays banked only if #71's real kernel ever measures >3%).**
- **Network-wide compounding (kanna #96 WIP):** closes #87's named residual — do the per-layer ≤1-ULP perturbations compound across ~30 layers to flip the final argmax on the composed land#71×ubel#84 frontier? Integration test consuming wirbel #93 + kanna #87 per-op GREENs.
- **Pre-quota gate status:** GEMM ✅ (kanna #87 MERGED, 0/65,536) · attention 🔴→✅-COST (wirbel #93 MERGED RED → **fp32 star-attn REQUIRED**; cost CONFIRMED ~free by wirbel #98 MERGED GREEN — then re-verify 0-flip after land implements) · network-wide compounding 🔄 (kanna #96 WIP). All ANALYTICAL assumptions closed (tree-economics saturated: #88/#86/#91/#92); the fp32 cost is now closed too. **The live blocker is the EMPIRICAL build (tok/step=2.10) — now DIAGNOSED by denken #101 (✅ MERGED 🟢) as a FIXABLE build defect (depth-1 spine corruption + non-descending salvage walk), handed to the build team; fern #102 sizes break-even, denken #105 prices the tree-free backup. NOT an open analytical question, and NO LONGER a single point of failure for 500.**

## Cycle 39–40 — lanes closed / confirmed this session

| PR | student | verdict | significance |
|---|---|---|---|
| #101 | denken | Accept-length reconciliation GREEN: as-built 2.10 is a FIXABLE BUILD DEFECT (≥56.1% provable, 100% fixable, (D)-ceiling 0%) — sits below the linear floor 3.844 the same drafter+verifier already hit; depth-1 spine 0.598 vs q[1]=0.7287 + non-descending salvage walk (reach 1.10% vs 60.8%) | tree 2.10 is build-blocked NOT ceiling-capped → ~568 survives; corrected band [3.844,5.207] to fern #102; tree no longer SPOF (denken #105 prices tree-free path) |
| #98 | wirbel | fp32 star-attn cost GREEN: conservative +0.339% M=32 / +0.010% M=8, realized NEGATIVE; haircut 0.404pp → tree net +19.41%. The #93 fp32 mandate is NOT load-bearing (safe AND free, L2-residency) | clears the last #71 fp32 numerics blocker cost-free; tail-only hybrid not needed → wirbel #104 double-quant |
| #100 | fern | Composition GREEN: tree alone clears 500 (cons 518/centr 563); composition ORDER-INDEPENDENT; min_levers=1. CAVEAT: assumes E[T]=5.207, build gives 2.10 | composition framework banked; conditional GREEN → denken #101 + fern #102 close the E[T] loop |
| #97 | denken | Persistent-kernel AMBER→CLOSE: decode 97.83% GPU-busy, only 2.17% reclaimable idle; the ~32% "other" is 93% GPU-busy small-kernel tail (megakernel reorders but can't remove — bus is the wall, #94) | persistent-kernel/megakernel lane CLOSED; #65 extended to megakernel |
| #95 | fern | LK-Loss AMBER: greedy headroom +1.0–2.4% E[T] (NOT +8% headline); re-rank channel CLOSED (drafter argmax already acceptance-ordered, rank-1 best by +0.6 margin); prediction channel untested | LK lane SIZED (don't transfer headline); LoRA/projection probe queued |
| #90 | lawine | K=7 CONFIRMED optimal (inverted-U; every non-K7 a REAL regression 7-35x MDE); locks 454.338 linear-chain ref; retires ±4.4% caveat | draft-length lane CLOSED (confirm) |
| #93 | wirbel | Star-attn relerr-1e-3 RED: flips 0.59% greedy tokens (noise floor 0) -> fp32 star-attn REQUIRED pre-quota | attention gate -> HARD fp32 constraint on land #71 |
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

1. **Land #71 completes tree-verify build (with fp32 star-attn per wirbel #93)** → openevolve oracle bench → lawine #99 served A/B → approval-gated HF submission. ~568 TPS projected (fern #92 band 558-581). Pre-quota: GEMM ✅ (kanna #87); attention needs the **fp32 star-attn implementation + 0-flip re-verify** (wirbel #93 RED, cost sized by #98); kanna #96 network-wide in flight.
2. **Persistent-kernel overhead-reclamation (denken #97):** the ~32% "other/overhead" is the biggest un-mined bucket; a megakernel could reclaim GPU-idle/launch/host slack (+8-15% IF idle). Crux: tested against denken's own #65 (decode 99.41% GPU-bound) — is the 32% GPU-idle (reclaimable, GREEN) or GPU-busy small-kernel-tail/bus-spillover (#65 extended, CLOSE)? CPU-first gate. _[Draft-verify overlap CLOSED: denken #94 AMBER — the A10G bus serializes the two streams → +1.2-2.9% realized, sub-build-bar.]_
3. **SplitK W4A16 verify-GEMM (ubel #84):** close 23% HBM gap at M=8; ~+5–12% wall_tps.
4. **Projection calibration (lawine #99):** K=7 confirmed (#90, 454.338 locked); now pin the local→official multiplier so land #71's ~568 is a measured band that clears 500 with margin.
5. **Drafter GEMM fusion (stark #78):** ~+2.6% ceiling once stark responds to check-in.
6. **Lever-composition economics (fern #100):** compose tree #71 × SplitK #84 × persistent-kernel #97 × LK #95 into the official-TPS landscape + the minimal lever ordering that clears 500 with most margin. Models anti-compounding (the tree amortizes fixed per-step overhead → shrinks the persistent-kernel bucket) vs compounding (tree × LK on the E[T] numerator; SplitK on the verify-GEMM denominator). Consumes the #98/#97/#99 bands.
7. **Next-wave levers (queued for freed seats):** double-quant verify-GEMM scales (+0.4–2%, compounds with ubel SplitK), QuantSpec INT4 drafter-KV (+10–15% claim — premise-check first: MTP may lack a separate drafter KV to quantize), Token Recycling (+4–10%; likely #89-redundant), **LK prediction-channel LoRA/projection probe** (fern #95 sized the channel at +1.0–2.4% greedy; a trained-head probe is the only way to realize it — approval-gated if it needs quota). _[Now in-flight: lever-composition → fern #100; persistent-kernel scheduling → denken #97.]_
