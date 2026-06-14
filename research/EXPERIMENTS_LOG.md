# SENPAI Research Results

## 2026-06-14 07:13 — PR #107: Tree-step denominator measurement — pin the REAL M=8→M=32 verify-step ratio 🟢 GREEN — MERGED (measured verify-forward floor 1.237×; whole-step bracket [1.145,1.156] CONFIRMS fern's 1.16×; break-even 4.614 holds vs 4.624; GEMM NOT flat — Marlin 16-row tile staircase 1.169× — but offset by attention-as-modeled 1.83× → nets to fern's 1.158)

- **Branch:** `lawine/tree-step-denominator` · **Student:** lawine · merged 07:14Z (LOCAL A10G microbench — no HF Job, timing only, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** convert the load-bearing 1.16× M=8→M=32 tree-step denominator (under fern #102's break-even + the 569 projection) from a back-solved model assumption into a MEASURED number with a CI, on the GEMM+causal-attn floor (no star-attn tree-mask kernel needed).
- **Primary metric:** `measured_M32_M8_step_ratio = 1.2370` [1.2268, 1.2472], CV 0.944% (median N=5, verify-FORWARD floor = GEMM+attn). **Test:** `corrected_breakeven_ET = 4.614` [4.568, 4.614] vs fern #102's 4.624 (Δ −0.21%, holds). W&B `tbhywbmw`.

| component | M=8 | M=32 | ratio |
|---|--:|--:|--:|
| **verify-forward floor (GEMM+attn)** | 5588 µs | 6910 µs | **1.2370** |
| GEMM (int4 W4A16 Marlin, ×42 layers) | 5013 µs | 5856 µs | 1.1686 |
| attention (deployed fp32 split-KV `unified_attention`) | 575 µs | 1054 µs | 1.8325 |

- **Verdict — 🟢 GREEN, fern's 1.16× whole-step denominator CONFIRMED by direct measurement.** Mapping the verify-forward floor (1.237×, on the ~61% GEMM+attn share) to the WHOLE step via two transparent maps gives bracket **[1.1446 lumped, 1.1560 budget-share]** vs fern's modeled **1.1584** → confirmed (not corrected). The component nuance is the real find: **GEMM is NOT flat within M≤32** (r=1.169, the 16-row Marlin tile staircase: M=8 fills 1 tile, M=32 fills 2 → +17%; still ≪ the +29% M=33 cliff denken #68 flagged) — this *corrects* denken #68's "~1.0× flat" piece — **but attention is exactly wirbel #98's 1.83×**, and in the budget-share map the GEMM-not-flat surplus and the as-modeled attention move opposite to the model and cancel, so the whole-step lands on 1.158. Break-even 4.614 sits comfortably between beat-linear 4.45 and the 5.207 ceiling → tree go/no-go unchanged.
- **The one open denominator RISK (lawine's caveat, carried forward):** the measured floor holds the drafter+host remainder (~39% of the M=8 step) FLAT — fern's M-invariance assumption. Two residuals sit ON TOP and are unmeasurable until the build: (1) the star-attn **tree-mask** kernel delta (causal attn is its lower bound); (2) **drafter tree-expansion** (a 32-node tree may issue more drafter passes than the linear 8-chain). If tree-expansion adds real M-variant cost, the ratio rises above 1.16 and the break-even tightens — this is what land #71's build must resolve and what the harness stays armed for.
- **Bug fixed (banked):** the first timed CUDA-graph kernel in a cold process runs at A10G BASE clock (~2× slow — no-warmup GEMM M=8 = 10013 µs vs warm 5013 µs); added a mandatory sustained warmup (clock ramp + Triton JIT both widths) before timed repeats. Protects every future microbench on this rig.
- **Banks:** `scripts/profiler/tree_step_denominator.py` + W&B `tbhywbmw`. **Next:** lawine → **#112 (harden the tree-free-500 projection + bound τ)** — denken #105 made tree-free the primary 500-path; arm the zero-lag SplitK%→official-vs-500 meter + bound τ from local data for denken #109's ship decision.

## 2026-06-14 06:55 — PR #106: Tree-vs-tree-free crossover + build-milestone ladder — at what realized E[T] does the tree overtake tree-free? 🟡 AMBER — MERGED (crossover@C=500 = 4.624 = #102 break-even verbatim; with denken #105's landed C=518.1 the tree is UPSIDE not critical-path; corner-matched recovery gate E[T]≥~4.7; CPU-only, greedy untouched)

- **Branch:** `fern/tree-vs-treefree-crossover` · **Student:** fern · merged 06:56Z (analysis-only, BASELINE unchanged 481.53)
- **Hypothesis:** generalize #102's break-even from target=500 to target=C (denken #105's tree-free ceiling); find the crossover E[T]ₓ(C) where `tree_official(E[T]) = C`, plus a build-milestone ladder official(E[T]) for ship-gates.
- **Primary metric:** `tree_vs_treefree_crossover_ET = 4.727` (corner-matched central). **Test:** `build_milestone_ladder_clear500_ET = 4.624`. W&B `1qkiheqb` (CPU-only ~27 MiB/~1s; reuses #102 `breakeven_raw_et` verbatim, rescale error 0.0).

| denken #105 ceiling C | crossover E[T]ₓ | verdict | meaning |
|---|--:|---|---|
| C < 500 | < 4.624 | 🟢 GREEN | tree-free can't hit 500 → tree CRITICAL-path |
| **500 ≤ C < 540.7** | 4.624–5.000 | 🟡 **AMBER** | tree-free clears 500 → tree is **UPSIDE** |
| C ≥ 540.7 | ≥ 5.000 | 🔴 RED | tree barely beats tree-free → pivot |
| C ≥ 563.1 | > 5.207 | 🔴 deep-RED | tree never beats tree-free → pivot+escalate |

- **Verdict — 🟡 AMBER, settled by denken #105's landed C=518.1.** 500 ≤ 518.1 < 540.7 → the tree is **bounded UPSIDE, not the critical path**. Corner-matched crossover (same optimism lifts both sides) collapses to a tight **4.834 / 4.727 / 4.737** → whatever corner reality picks, **the build must recover to E[T] ≈ 4.7–4.8 to overtake tree-free** (alone; ~4.52 with splitk+lk also built). Milestone ladder: beat-linear 4.45 → clear-500 4.62 → overtake-tree-free ~4.7; ~10.8 official TPS per +0.1 accept_length. Tree's recoverable official band [416, 563] central — overtakes tree-free only in the upper part of denken #101's [3.844, 5.207] band (the floor 3.844 never clears any plausible C). Caveat: at the conservative ceiling corner (496.8 < 500) the verdict flips GREEN (tree critical) → the AMBER call rests on the tree-free *central*, which denken #109 ship-readiness pins.
- **Jointly with denken #101 + #105:** converts the tree from a single point of failure into bounded upside on a hard recovery gate (E[T] ≥ ~4.7).
- **Banks:** `scripts/profiler/tree_vs_treefree_crossover.py` + `tree_vs_treefree_crossover_results.json` + report. **Next:** fern → **#111 (settle headline at C=518.1 + post-500 lever-ROI climb)** — rank the 500→556 levers by official-TPS-per-build-effort.

## 2026-06-14 06:52 — PR #105: Tree-free 500-path ceiling — does the build-complete stack clear 500 with NO tree, at what SplitK threshold? 🟢 GREEN — MERGED (tree-free C=518.1 central [496.8,540.8]; SplitK-for-500 = 4.44% < ubel's +5% floor; ceiling 556; the tree is now INSURANCE; binding gate moves SplitK→τ)

- **Branch:** `denken/tree-free-500-ceiling` · **Student:** denken · merged 06:53Z (analysis-only, BASELINE unchanged 481.53)
- **Hypothesis:** can SplitK #84 + LK #95 + double-quant #104 clear 500 with NO tree (which is build-blocked per #101), and at what SplitK threshold? — the go/no-go deciding whether the tree is critical-path or insurance.
- **Primary metric:** `tree_free_max_official_tps = 518.1` [496.8, 540.8] (ubel-high SplitK 12%). **Test:** `splitk_threshold_for_500 = 0.0444` (4.44% central, below ubel's +5% floor). W&B `0kiktnqt` (CPU-only; composes merged levers via fern #100 model, K_cal=125.268).
- **Verdict — 🟢 GREEN, 500 is reachable TREE-FREE.** The build-complete linear stack clears 500 at SplitK ≥ **4.44%** (central) — below the +5% floor ubel #84 already targeted; SplitK 8.5%→509.9, 12%→518.1; full ceiling **556.0 [533.2, 581.1]** at 29.7% gap-close. **Strategic consequence: the tree (land #71) is now INSURANCE/UPSIDE (500→556-581), not critical-path** — the #101 build defect no longer blocks 500. **The binding gate moves from SplitK to τ** (realization factor [0.96,1.00]): τ→1.00 is ~3× cheaper than any other margin lever. SplitK × double-quant netting is orthogonal (multiply, no double-count).
- **Critical-path handoff:** land ubel #84 SplitK to ~8.5% (→ 509.9 central) + pin lawine #99's τ. **Next:** denken → **#109 (tree-free-500 ship-readiness)** — min SplitK for a *confident* (conservative-corner) ship + whether pinning τ needs an approval-gated official re-anchor or ships on lawine #99 local calibration.
- **Banks:** `scripts/profiler/tree_free_500_ceiling.py` + `tree_free_500_ceiling_results.json` + report.

## 2026-06-14 06:52 — PR #104: Double-quant verify-GEMM scales 🔴 KILL (banked) — MERGED (bit-exact frac 13.1% « 98% gate; info-theoretic dead: FP16 10 mantissa bits vs ~1.4-1.9-octave scale spread; corrected byte estimate 53.70 MB = 3.06% of int4 body; successor = lossless scale-palette/LUT)

- **Branch:** `wirbel/double-quant-verify-gemm-scales` · **Student:** wirbel · merged 06:52Z (analysis-only, BASELINE unchanged 481.53)
- **Hypothesis:** double-quantize the verify-GEMM int4 scales (quant-the-scales) for a greedy-lossless byte-saving → wall_tps lift.
- **Primary metric:** `dq_scale_roundtrip_bitexact_frac = 0.1309` (gate >0.98 → **FAILED**). **Test:** `dq_tps_lift_est_pct = -0.02%`. W&B `6or2w3ee` (CPU-only byte/precision analysis).
- **Verdict — 🔴 KILL, info-theoretic (not tunable).** FP16 carries 10 mantissa bits; the per-group scale spread is ~1.4–1.9 octaves, so any 8-bit re-code of the scales loses ≥2 bits → only 13.1% of scales round-trip bit-exact « the 98% greedy-safety gate. Not fixable by a better codebook — it is an information bound. **Banked value outlives the negative:** (1) corrected byte accounting — core-7 verify-GEMM scales = **53.70 MB = 3.06%** of the 1754.7 MB int4 body (**g=128** confirmed for the folded osoi5-v0-baked weights; the earlier g=32 was the *unfolded* base); (2) the **successor that survives** — a lossless **scale-palette/LUT**: the scales take only **1,009 distinct FP16 values globally** (per-tensor median 427) → a 10-bit global / 9-bit per-tensor index into a palette of the *exact* values is **bit-exact by construction**, ~37.5% scale-byte saving (~20 MB), ~0.3% TPS.
- **Banks:** `scripts/profiler/dq_scale_roundtrip.py` + `dq_scale_roundtrip.json` + report. **Next:** wirbel → **#110 (lossless scale-palette/LUT)** — build-or-kill: are scale bytes on the BW-critical path or already hidden by Marlin?

## 2026-06-14 06:33 — PR #102: Tree E[T] break-even / margin-of-safety — what MIN accept_length clears 500? 🟡 AMBER — MERGED (break-even E[T]*=4.624 tree-alone; the M=32 step is ~1.16× heavier so the tree needs E[T]≥4.45 just to TIE linear 481.53, not 3.844; byteshark's 2.10 is a regression; no lever stack pulls break-even <4.0)

- **Branch:** `fern/tree-et-breakeven` · **Student:** fern · merged ~06:33Z (analysis-only, BASELINE unchanged — official bar UNCHANGED 481.53)
- **Hypothesis:** invert fern #100's forward model (`official_TPS = K_cal·E[T]/step_time·τ`) — solve `official=500` for the threshold accept_length E[T]*, tree-alone + per compounding lever stack; place byteshark's 2.097 + denken #101's recoverable band on that axis.
- **Primary metric:** `breakeven_ET_tree_alone = 4.624` [4.481 opt, 5.026 cons]. **Test:** `ET_recovery_needed_from_2p10 = 2.527` (tree-alone, central). W&B `l12ikxea` (CPU-only; imports #100's `lever_composition.py` verbatim, max |direct−rescale| = 1.8e-15 machine-zero; reproduces #100's 563.1 at E[T]=5.207 exactly).

| break-even ladder (raw accept_length to clear 500) | cons | central | opt | recovery from 2.097 |
|---|--:|--:|--:|--:|
| **tree alone** | 5.026 | **4.624** | 4.481 | **2.527** |
| tree+splitk #84 | 4.922 | 4.458 | 4.254 | 2.361 |
| tree+lk #95 | 5.001 | 4.587 | 4.376 | 2.490 |
| tree+lk+splitk | 4.897 | 4.422 | 4.155 | 2.325 |
| full stack (+persist #97) | 4.897 | 4.339 | 3.648 | 2.242 |

- **Verdict — 🟡 AMBER, must-recover-most-of-the-way.** The load-bearing reframe: fern separated the M=32 *denominator* widening (a step-time fact) from the accept-length *numerator* (free variable), and showed the binding floor is NOT denken #101's structural 3.844 (accept-length units) but **4.45 in OFFICIAL-TPS units** — because the ~1.16× heavier M=32 step means a 'merely correct' tree at 3.844 is still a TPS *regression* (415 official). The tree needs E[T] ≥ **4.45 to TIE 481.53** (the abort line), ≥ **4.624 to clear 500 alone**, and **no lever stack pulls the central break-even under 4.0** (SplitK is the most useful, −0.17; LK barely moves it −0.04). byteshark's as-built **2.097 is a regression** (~227 official, <½ the linear frontier). Critical recoverable threshold (central): <4.34 → no path to 500 with any stack (escalate); [4.34, 4.62) → needs compounding levers; ≥4.62 → tree alone clears. Conservative corner: break-even 5.026 ≈ the 5.207 ceiling → margin only +0.181 even at full recovery.
- **Reframes #100's GREEN:** #100 ('tree clears 500 with margin') is true *given* E[T]=5.207, but in accept-length units the margin is thin (0.58 central / 0.18 cons below ceiling). byteshark's 2.097 falsifies the 5.207 assumption, so the binding variable is the realized E[T] — GREEN→AMBER not because the model changed but because the build's accept-length collapsed.
- **Composes with denken #101:** denken says 2.10→≥3.844 recovery is structurally guaranteed once the spine+salvage defects are fixed; fern says 3.844→4.62 then needs ~75% of the branch premium. Build job precisely bounded: fix defect (→3.844 floor), land full-depth traversal (→toward 5.207). **Abort line E[T]<4.45.**
- **Banks:** `scripts/profiler/tree_et_breakeven.py` + `research/spec_cost_model/tree_et_breakeven_results.json` + report. **Next:** fern → **#106 (tree-vs-tree-free crossover + build-milestone ladder)** — at what realized E[T] does the tree overtake denken #105's tree-free ceiling; partial-recovery curve for build ship-gates.

## 2026-06-14 06:33 — PR #99: Local→official projection calibration + tree-A/B harness 🟢 GREEN — MERGED (multiplier 1.06019 config-stable to 0.056%; closed-loop self-check reproduces 454.338 AND maps back onto 481.53 within 0.014%; zero-lag build-agnostic projection harness armed; current-spec tree projects 569 [552,587])

- **Branch:** `lawine/projcal-tree-harness` · **Student:** lawine · merged ~06:33Z (calibration-only; live dry-run was a tree=OFF self-null on the frontier — BASELINE unchanged 481.53)
- **Hypothesis:** pin the local-wall_tps→official-TPS multiplier + ready the tree-A/B harness so land #71's build is a zero-lag ≥500 decision.
- **Primary metric:** `local_to_official_multiplier = 1.06019` [1.05999, 1.06038] (±0.018%). **Test:** `linear_chain_wall_tps_reproduced = 454.258` (Δ0.018% < 0.10% MDE). W&B `zcfjgog9`.
- **Verdict — 🟢 GREEN.** Multiplier is a pure hardware/environment transfer factor (`wall_tps` is *definitionally* the official `output_throughput`), config-stable to **0.056%** across 5 committed sessions (454.085–454.338). Closed-loop self-check PASSES end-to-end: the live meter reproduces 454.338 within MDE AND the multiplier maps it back onto the 481.53 anchor (both arms within 0.014%). The projection harness (`local_official_projection.py`, 12 unit tests, build-agnostic) maps any candidate's measured wall_tps → projected-official band in one command. Current-spec tree (E[T]=5.207, +18.2%) → **569.3 official [552.1, 586.6]**, clears 500 by +10.4% at the conservative low edge; gate robust unless the official anchor is ~12% below its private-VERIFIED value.
- **Advisor scoping note (recorded with merge):** the 569 GREEN is conditional on E[T]=5.207 — the projection MATH + harness are sound and armed, but the binding variable (realized tree E[T]) is contested by byteshark's 2.097 + fern #102's 4.624 break-even. GREEN = 'instrument calibrated and ready,' not 'tree will score 569.' Honest caveat from lawine: cross-*precision* invariance (int4 vs bf16) not cleanly testable from repo (older rungs metered with the 16-prompt meter) — but NOT the binding axis for the tree (same int4/split-KV precision as anchor, only widens drafter M=8→M=32).
- **Banks:** `scripts/profiler/local_official_projection.py` + `paired_tps_ab.py` projection layer + `scripts/tests/test_local_official_projection.py` (12 tests). **Next:** lawine → **#107 (tree-step denominator measurement)** — pin the real M=8→M=32 verify-step wall-time ratio (the load-bearing 1.16× under fern #102 + this 569 projection), measurable now without land's star-attn kernel.

## 2026-06-14 06:25 — PR #101: Tree accept-length reconciliation — why is the as-built tok/step=2.10 vs analytical 5.207? 🟢 GREEN — MERGED (2.10 is a FIXABLE BUILD DEFECT, not acceptance collapse; ≥56.1% provably build-defect, 100% fixable, (D)-ceiling 0%; ~568 survives; tree marked BUILD-BLOCKED/re-measure-pending)

- **Branch:** `denken/tree-accept-reconciliation` · **Student:** denken · merged ~06:24Z (analysis-only, BASELINE unchanged — official bar UNCHANGED 481.53)
- **Hypothesis:** byteshark's first real tree build (`tree-v2-merge-eager-v1`) delivered tok/step=2.097 — BELOW even linear-MTP 3.844, far below analytical 5.207. Back out implied per-position ρ̂ from the accept hist, compare to wirbel #79's ρ-ladder, classify the defect (A truncated walk / B draft-quality collapse / C eager artifact / D model-optimistic), and hand fern an honest E[T] band.
- **Primary metric:** `tree_accept_length_gap_explained_pct = 100.0` (fixable). **Test:** `implied_tree_rho_hat_depth1 = -0.481` (negative ⇒ impossible for a real tree ⇒ build corrupts the spine). W&B `c5nbdjic` (CPU-only, no GPU; E[T] anchors reproduce wirbel 5.207 to 4.6e-5).

| reference (tok/step, bonus incl) | E[T] | note |
|---|--:|---|
| model full tree (M=32) | 5.2070 | wirbel, \|Δ\|=4.6e-5 |
| model spine-only (rank-1, depth-9) | 4.1773 | |
| **deployed linear-MTP (measured)** | **3.8441** | hard floor a correct tree MUST clear |
| **as-built (byteshark eager-diag)** | **2.102** | 40.4% of model |

- **Verdict — 🟢 GREEN, fixable build defect.** The decisive observation: 2.10 sits 1.74 tok BELOW a *measured* floor (linear 3.844) that the *same drafter+verifier* already achieve — and the tree spine IS the linear chain, so a correct tree cannot accept less. ⇒ not an acceptance question at all. Gap = 5.207−2.102 = 3.105 tok decomposes as: **(A) sub-linear collapse 3.844→2.102 = 1.742 tok = 56.1% = PROVABLE build defect (dominant)**; spine depth-9 ext 0.333 tok = 10.7% (unrealized premium); branch premium 1.030 tok = 33.2% (unrealized premium). **(B) draft-quality** bounded SMALL (salvage fires 0.358/step = 1.07× model-expected 0.336 → drafter IS producing correct rank-2+ candidates); **(C) eager-mode** ~0% of accept-length (overhead/TPS axis, not numerics); **(D) optimistic model = 0%** (fern #92 realized 5.208 vs independent 5.207, +0.025%).
- **Mechanism (the build defect):** (1) depth-1 spine continuation collapses to **0.598 vs the required q[1]=0.7287** (depth-1 is drafter-identical to linear, no tree-attn → a correct build MUST hit 0.7287) ⇒ verify/dispatch corrupts the spine before any branch logic; (2) salvage walk recovers the immediate rescue node but does NOT descend its sub-path — full-tree reach **1.10% as-built vs ~60.8% model**. wirbel #79's ρ-ladder is sound and **un-exercised** (build corrupts below its own rank-1 floor).
- **Corrected band handed to fern #100/#102:** compose against **[3.844 floor, 5.207 ceiling]**, central 5.14–5.21 IF full-depth traversal + fp32 star-attn land; tree marked **BUILD-BLOCKED / re-measure-pending**. Do NOT plug 2.10 (defect artifact) or 5.207 (unrealized) directly. JSON `step3_gate.corrected_ET_band_for_fern100` is machine-readable.
- **Build hand-off (relayed to land #71 / byteshark / chiku-inu / openevolve, board 20260614-062536):** (1) fastest localizer = assert depth-1 == 0.7287; (2) make the salvage walk descend the recovered sub-tree (full-reach ≫1.1%); (3) re-measure accept_length on the **fp32** star-attn path (NOT relerr eager — wirbel #93 is a *separate* greedy blocker).
- **Banks:** `scripts/profiler/tree_accept_reconciliation.py` + `research/spec_cost_model/tree_accept_reconciliation_results.json` + `report_tree_accept_reconciliation.md`. **Strategic consequence:** the tree is no longer a single point of failure for 500 — denken reassigned to **#105 (tree-free 500-path ceiling)**: can SplitK #84 + LK #95 + double-quant #104 clear 500 with NO tree, and at what SplitK threshold? (go/no-go: tree critical-path vs insurance). **Next (queued, denken on request):** re-run this exact reconciliation once byteshark posts a per-position branch-hit histogram from the fixed build → converts 5.207 into a realized band + bounds the deep-tail (B).

## 2026-06-14 06:12 — PR #98: fp32 star-attn cost gate — does the #93-mandated fp32 accumulation erode the tree's +18.2%? 🟢 GREEN — MERGED (fp32 star-attn is ~free: conservative +0.34% M=32 / +0.01% M=8, realized NEGATIVE; haircut 0.404pp → tree net +19.41%; the #93 fp32 constraint is NOT load-bearing)

- **Branch:** `wirbel/fp32-starattn-cost-gate` · **Student:** wirbel · merged ~06:12Z (analysis-only, BASELINE unchanged — official bar UNCHANGED 481.53)
- **Hypothesis:** #93 (RED) mandated fp32 star-attn accumulation (bf16-relerr-1e-3 flips 0.59% greedy tokens). This gate PRICES that mandate: does fp32 erode the tree's +18.2%, or is a tail-only-fp32 hybrid needed?
- **Primary metric:** `fp32_starattn_tree_gain_haircut_pp = 0.404`. **Test:** `fp32_starattn_cost_pct_M32_conservative = 0.339`. W&B `r8xckc7s` (primary) + `jbooswq1` (bit-identical replicate); advisor-verified (both finished, no NaN, verdict_green=1, Δ<0.001pp).

| layout | realized cost% | conservative cost% | attn % of decode |
|---|--:|--:|--:|
| M=8 frontier | −0.305 | +0.010 | 6.43% |
| M=32 tree-verify | −0.572 | +0.339 | 8.12% |

- **Verdict — GREEN. fp32 star-attn is ~free; the #93 constraint is NOT load-bearing.** Priced the only BW-relevant channel (per-segment softmax partial buffer) on the real deployed 3D split-KV `unified_attention` kernel: fp32 vs bf16 partials, everything else identical (KV stays bf16 → read bytes flat in M). Conservative worst-case **+0.339% M=32 / +0.010% M=8** (≤1% → GREEN); realized NEGATIVE. **Mechanism = A10G 6MB L2-residency:** fp32 partials fit L2 for every layer-type except full-M=32 (spills by 2.4MB = the ONLY place fp32 costs wall-time, +5.6µs/op × 7 full layers). Haircut on denken #85's net = **0.404pp → +19.82%→+19.41%** (576.96→575.02 official-projected, −1.94 TPS). The bf16 reduction that would "save" 0.34% is exactly the #93-unsafe path (flips 0.59% greedy tokens) → **fp32 is both SAFE and FREE.** Tail-only-fp32 hybrid NOT needed (#93's 0.537% margin map stays banked if #71's real kernel ever measures >3%).
- **Secondary finding (orthogonal, routed to denken #101 / land #71):** real-kernel M=32 split-KV attention = **1.83× M=8** (8.12% of step) vs denken #85's SDPA-proxy 1.06× — a small attention-BASELINE haircut (NOT fp32) for denken #101 to fold in with the real paged kernel. denken #85's KV-bytes-flat-in-M claim still holds; the wall-time doesn't at the conc=1 floor (the 32-row Q·K·softmax·V compute becomes visible above the tiny KV-read). Caveat: measured on vLLM `unified_attention` (causal), not #71's custom star-attention tree-mask kernel — worth re-confirming with the real paged kernel.
- **Banks:** `scripts/profiler/star_attn_fp32_cost.py` + `research/star_attn_gate/fp32_cost_results.json`. **Next:** wirbel → **#104** (double-quant verify-GEMM scales — INT8 scale-of-scales, greedy-lossless ~0.4–1.1% byte lever; CPU round-trip build-or-kill first).

## 2026-06-14 06:05 — PR #100: Lever-composition economics — composed official-TPS landscape + minimal lever ordering to clear 500 🟢 GREEN — MERGED (tree-sufficient: tree alone clears 500 at the conservative corner; composition is order-independent; min_levers=1) — verdict conditional on E[T]≈5.207, which byteshark's empirical 2.10 now contests

- **Branch:** `fern/lever-composition-economics` · **Student:** fern · merged ~06:08Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** the in-flight 500-path levers are priced in isolation. Compose tree #71 × SplitK #84 × persistent-kernel #97 × LK #95 into one official-TPS landscape + the minimal lever ordering that clears 500, accounting for compounding vs anti-compounding.
- **Primary metric:** `composed_official_tps = 600.0` (full stack, band [531.6, 713.7]). **Test:** `min_levers_to_clear_500 = 1` (`['tree']`).

| lever stack | conservative | central | optimistic | clears 500? |
|---|--:|--:|--:|---|
| frontier | 462.3 | 481.5 | 481.5 | no |
| + LK #95 | 466.9 | 486.3 | 493.1 | no |
| + persist #97 | 462.3 | 496.4 | 553.5 | no |
| + SplitK #84 | 474.2 | 502.4 | 510.5 | central only |
| **+ tree #71** | **518.0** | **563.1** | **581.0** | **YES (even conservative)** |
| full stack | 531.6 | 600.0 | 713.7 | yes |

- **Verdict — GREEN (tree-sufficient), with a load-bearing caveat.** Model: `official_TPS = K_cal·(E[T]/step_time)·τ`, `K_cal = 481.53/3.844 = 125.268` (frontier reproduces exactly); step decomposed into ABS slices (verify-GEMM 0.53 / drafter 0.07 / attn 0.08 / other 0.32). Numerator levers (tree, LK) multiply E[T]; denominator levers (SplitK, persist) subtract an **absolute** saving from their slice. **Key structural finding: the final step is ORDER-INDEPENDENT** — the "lever ordering" is purely a relative-attribution artefact, which collapses the sequencing question. `min_levers_to_clear_500 = 1 (['tree'])` at BOTH conservative (518.0) and central (563.1) → the tree alone is sufficient, every other lever is insurance/upside. **CAVEAT (the reason this is not an operational green-light):** the entire result assumes the tree realizes E[T]≈5.207 (+18.2%). byteshark's first empirical build delivers **tok/step=2.097** — outside the conservative [558,581] band. So GREEN holds *conditional on the build working as analyzed*, a condition now empirically open. **denken #101** diagnoses the recoverable E[T]; **fern #102** (break-even inverse) computes the required E[T]* — their intersection is the real go/no-go.
- **W&B:** `ncseu3ar`. **Next:** fern → #102 (tree E[T] break-even / margin-of-safety — invert this model for the minimum accept_length that clears 500, alone + per lever stack; place byteshark's 2.10 + denken #101's recoverable band on that axis).

## 2026-06-14 05:48 — PR #97: Persistent-kernel overhead gate — is the ~32% "other" GPU-idle (megakernel-reclaimable) or GPU-busy? 🟡 AMBER — MERGED → persistent-kernel/megakernel LANE CLOSED (only 2.17% reclaimable GPU-idle; #65's GPU-bound finding EXTENDS to the megakernel objective)

- **Branch:** `denken/persistent-kernel-overhead-gate` · **Student:** denken · merged ~05:59Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** the parallel-advisor LEVER 1 prices a persistent-kernel/megakernel at +8–15% by assuming the ~32% "other/overhead" is reclaimable GPU-idle (launch latency, host round-trips, inter-kernel bubbles). Tested against denken's own #65 (decode 99.41% GPU-bound). Is the 32% GPU-idle (reclaimable, GREEN) or GPU-busy small-kernel-tail/bus-spillover (#65 extended, CLOSE)?
- **Primary metric:** `persistent_kernel_reclaimable_pct = 2.17%` (GPU-idle ceiling). **Test:** `decode_gpu_idle_fraction = 0.0217` (2.173% ± 0.024% across 39 cycles).

| idle bucket (a+b+c = reclaimable) | % of decode wall |
|---|---|
| (a) kernel-launch / API overhead | 0.53% |
| (b) host-device sync / Python round-trip | 0.33% |
| (c) inter-kernel GPU-idle bubble | 1.31% |
| **total GPU-idle (reclaimable ceiling)** | **2.17%** |
| (d) GPU-BUSY real kernels (the other ~29.8pp of "32%") | **93% of the bucket — NOT reclaimable** |

- **Verdict — AMBER (2.17% < 3% GREEN) → CLOSE the persistent-kernel/megakernel lane.** A trace-direct timeline of the deployed frontier decode step (CUDA graphs ON, conc=1, committed #43 post-split-KV trace) shows the GPU is **97.83% busy / 2.17% idle** in steady decode (1049 kernels per 8.16 ms cycle). Of the coarse "~32% other", only **2.17pp is GPU-idle**; **29.8pp (93%) is GPU-busy** under-counted attention + drafter + norm/sampling/lm_head/elementwise — all real kernels a megakernel *reorders* but **cannot remove** (the bus is the wall, #94). LEVER 1's "+8–15% reclaimable scheduling idle" premise is **refuted**; **#65's 99.41%-GPU-bound finding EXTENDS** from launch-overhead to the full megakernel objective (a sharper closure: not just CUDA-graph-immune but megakernel-immune). Even the 2.17% is mostly intra-graph bubble CUDA-graphs already minimized. Re-labels the 32% bucket correctly: it is the GPU-busy small-kernel tail + bus-bound spillover, not idle slack.
- **W&B:** `gro3qa0d`. **Next:** denken → #101 (tree accept-length reconciliation — why the as-built tree gives `tok/step=2.10` vs analytical E[T]=5.207; the #1 lever's first empirical number).

## 2026-06-14 05:43 — PR #95: Drafter loss-objective gate — is the MTP draft head acceptance-optimal or only likelihood-optimal? (LK-Loss headroom) 🟡 AMBER — MERGED (LK headroom is +1.0–2.4% E[T] under greedy, NOT the +8% headline; re-rank channel rigorously CLOSED; prediction channel untested; banks the measured acceptance profile)

- **Branch:** `fern/drafter-accept-objective` · **Student:** fern · merged ~05:43Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** the MTP draft head is trained likelihood-optimal; an acceptance-aware (LK-Loss / rank-calibrated) objective claims +8–10% E[T]. Does that headroom survive our GREEDY (T=0) verify, and is it a re-ranking win (free, re-order existing draft logits) or a prediction win (needs a trained head)?
- **Primary metric:** `lk_implied_ET_headroom_pct = +2.4%` (greedy ceiling). **Test:** `measured_drafter_top1_accept = 0.7287`.

| channel | headroom under greedy | status |
|---|---|---|
| LK paper headline (T=1 / sampling) | +8–10% E[T] | NOT our regime |
| re-ranking (re-order existing draft logits) | **0.0%** | rigorously CLOSED — drafter argmax already acceptance-ordered (rank-1 best by +0.6 margin) |
| prediction-improvement (trained head) | **+1.0–2.4%** | only surviving channel (EAGLE-3 +2.4 / Medusa +1.0 / MLP-spec +1.2); #80 likelihood-only never tested it |

- **Verdict — AMBER → SIZE, DON'T TRANSFER.** The +8–10% is the paper's sampling-regime figure; under our greedy verify it collapses 3–8× to **+1.0–2.4% E[T]** (~486–493 official). The **re-ranking channel is rigorously CLOSED**: the drafter's argmax is already acceptance-ordered (rank-1 is the best candidate by +0.6 acceptance margin → 0.0% from re-weighting existing logits). Only the **prediction-improvement channel** (a head trained on the acceptance objective — which #80's likelihood-only EAGLE training never isolated) remains live. Positive selection confirmed: P(accept | position k) rises 0.7287→0.8473 across the chain; measured top-1 accept 0.7287 reconciles E[T]=3.8445 to 3e-4. **Banks the measured per-position acceptance profile.** Student rec: do NOT transfer the +8% headline; do NOT full-launch unsized; size the prediction channel with a cheap LoRA/projection probe first; do NOT close the lane. **LK LoRA/projection probe QUEUED** (a separate, approval-gated run if it needs real quota).
- **W&B:** `8kzjyzxb`. **Next:** fern → #100 (lever-composition economics — compose tree #71 × SplitK #84 × persistent-kernel #97 × LK #95 into the official-TPS landscape + minimal lever ordering to clear 500).

## 2026-06-14 05:31 — PR #93: Star-attention greedy-equivalence gate — does the tree-mask numerical path preserve greedy argmax? 🔴 MERGED — RED (relerr-1e-3 star-attention is NOT greedy-safe; land #71 MUST run fp32 accumulation before quota; banks the reusable attention-side flip-gate)

- **Branch:** `wirbel/star-attn-greedy-gate` · **Student:** wirbel · merged ~05:31Z (analysis-only, BASELINE unchanged; official bar 481.53)
- **Hypothesis:** land #71 / chiku-inu implement the tree mask via a triton star-attention path (per-row prefix + rank-1 self, no dense mask; vLLM force-overrides FlexAttention→TRITON_ATTN on gemma-4-E4B), validated externally only to relerr ~1e-3. But the greedy-identity contract (program.md 27-28) is bit-for-bit. Does a ~1e-3 relative attention-output perturbation flip the top1-vs-top2 decision at close-margin positions? Attention-side twin of kanna #87's now-GREEN GEMM gate — the LAST un-audited half of land #71's pre-quota numerical surface.
- **Primary metric:** `greedy_flip_rate_at_1e3 = 0.005927` (0.59%, RED). **Test:** `min_greedy_margin_p1 = 0.001965` (1st-pct relative final-logit top1−top2 margin).

| metric | value | read |
|---|---|---|
| `greedy_flip_rate_at_1e3` (primary, bf16/deployed argmax) | **0.59%** | RED (>0 = bit-for-bit contract violation) |
| noise floor (clean-vs-clean, bf16 & fp32) | **0.0** | control — every flip is perturbation-attributable, not nondeterminism |
| fp32-propagation share of flips | **82%** | TRUE top-1 changes → needs fp32 ACCUM, not just fp32 readout |
| bf16-tie-readout share | 18% | a higher-precision argmax removes only these |
| eps sweep 1e-4 / 1e-3 / 1e-2 (bf16) | 0.597 / 0.593 / 0.734% | FLAT 1e-4→1e-3 — near-tie tail governs, not eps magnitude |
| near-tie population (rel-margin <1e-2 / <1e-3) | 4.77% / 0.537% | the flip-risk set |
| eps_first_flip | 1e-4 (smallest tested) | even realized relerr 3.3e-5 flips 0.485% |

- **Verdict — RED.** A star-attention-magnitude (relerr 1e-3) perturbation flips 0.59% of greedy tokens; noise floor provably 0/65,536, so it's a real effect. Decisive decomposition: 82% are genuine fp32-propagation flips (the TRUE top-1 token changes) → a higher-precision argmax readout removes only ~18%; **land #71 needs fp32 accumulation on the star-attention reduction (softmax·V / o_proj), not merely a higher-precision readout.** Flip rate is FLAT across eps=1e-4→1e-3 because the margin distribution has a thin near-tie tail (~0.5% of positions <1e-3 rel-margin) that any surviving perturbation tips → land needs ~bit-exactness (≲1e-6) at near-tie positions; fp32 accum typically reaches ~1e-6. Flipped positions: median fp32 margin 0.075, with a rare large-margin tail (max 3.74) from cross-position residual-stream propagation (the reason a purely local margin-gated salvage may under-recall).
- **Routing to land #71 (Morgan, 05:30Z — HARD PRE-QUOTA CONSTRAINT):** star-attention path MUST accumulate in fp32 (or be proven bit-exact) before ANY quota spend; re-verify with this gate's script against the kernel's *measured* relerr to confirm flip_rate→0. Discovering this post-launch would have burned the approved HF Job. Salvage option (wirbel follow-up #3): margin-gated fallback recomputing only the ~4.8% near-tie positions in fp32 — folded into wirbel #98's cost-erosion gate.
- **Banks:** `scripts/profiler/star_attn_greedy_gate.py` (reusable attention-side flip-gate). CPU/eager teacher-forcing of the deployed int4 checkpoint (bf16 argmax reproduced exactly per serve.py:410; cross-kernel caveat quantified: eager-vs-vLLM disagrees at 1.48% of positions — itself near-tie fragility, which reinforces the finding); 15.54 GB single-GPU, no HF launch.
- **W&B:** `ut6a94qa` (group `star-attn-greedy-gate`, 428s, 5 seeds). Advisor-verified independently (wandb-query): all substantive metrics (primary/test/noise-floor/full eps-sweep/margin-distribution) match to 6+ sig figs; `verdict_red=1`; `step1/n_positions=65536`; run `finished` (`peak_gpu_mem_gb` unlogged + `_runtime=0` are cosmetic W&B artifacts).
- **Pre-quota numerics surface:** GEMM ✅ GREEN (kanna #87) · attention 🔴 RED→needs-fp32 (this) · network-wide compounding 🔄 (kanna #96). **wirbel → #98** (fp32 star-attn cost-erosion gate: does the #93-mandated fp32 accum erode the tree's +18.2%?).

## 2026-06-14 05:31 — PR #90: MTP draft-length K sweep — empirical wall_tps confirmation of K=7 optimality ✅ MERGED — GREEN/CONFIRM (clean inverted-U, K=7 optimal; locks 454.338 linear-chain reference for land #71; retires the ±4.4% fragile-estimator caveat)

- **Branch:** `lawine/mtp-k-sweep-wall-tps` · **Student:** lawine · merged ~05:31Z (config A/B, BASELINE unchanged; official bar 481.53)
- **Hypothesis:** the deployed linear MTP K=7 was confirmed "near-optimal static" only on the OLD fragile estimator (±4.4% floor); never directly verified with the new robust `wall_tps` runner (lawine #82, CV 0.007%, MDE ≥0.1% N=3). A direct K sweep closes it empirically — confirm K=7 or find a free serve-config win. First "real lever" test of the #82 paired-A/B runner.
- **Primary metric:** `mtp_k_optimal_wall_tps = 454.338` (best K=7). **Test:** `mtp_k7_confirmed_optimal_bool = 1`.

| K | median wall_tps | Δ% vs K=7 | verdict | E[accept] tok/step |
|---|---|---|---|---|
| 5 | 438.412 | −3.505% | REAL | 3.4902 |
| 6 | 451.047 | −0.724% | REAL | 3.7160 |
| **7 (ref)** | **454.338** | — | REF | **3.8555** |
| 8 | 440.282 | −3.094% | REAL | 3.9720 |
| 9 | 440.784 | −2.983% | REAL | 4.0794 |

- **Verdict — K=7 CONFIRMED OPTIMAL.** Clean inverted-U; every non-K7 arm clears the REAL bar (|Δ|≥0.10% N=3) by 7–35×. E[accept] rises monotonically with K (3.49→4.08) with shrinking increments — wall_tps peaks at K=7 because past it the marginal acceptance gain no longer repays the per-step drafter+verify cost (exactly denken #51's analytic K*≈7, now on the robust meter → the ±4.4% caveat is retired). Asymmetric curve: the K7→K8 step-cost jump (+0.54ms) is anomalously large vs ~0.23–0.25ms/K elsewhere → hypothesis: LOOPGRAPH/ONEGRAPH capture sized for M=8 (K=7); K≥8 (M≥9) falls outside the captured bucket and pays a re-pad penalty → K=7 is the *engineered* sweet spot the whole capture+precache stack is tuned around. No free config win.
- **Locks:** **454.338 wall_tps = the linear-chain reference** for land #71's tree-verify gain measurement (paired, CV 0.001%; E[T]=3.8555 matches deployed 3.844 +0.3%). Don't re-derive it.
- **Banks:** `research/walltps_ab/{run_k_sweep.sh,analyze_k_sweep.py,mtp_k{5,6,8,9}/...}`. Single-variable (only num_speculative_tokens via SPECULATIVE_CONFIG env; no served-file change to fa2sw_precache_kenyan); shared fresh K=7 baseline byte-identical across reuse arms; completed 128/128 every run; PPL/greedy untouched (serve-config knob, verifier argmax unchanged). Local-only, no quota.
- **W&B:** K6 `vz5whvxs` (holds shared K7 baseline) · K5 `7ven5w5b` · K8 `bvms4yto` · K9 `ela8jaqt`. Advisor-verified independently (wandb-query): all per-K wall_tps Δ + E[accept] match (3 sig figs); K7 baseline byte-identical CV 0.001%; all 4 runs `finished`, no NaN. **lawine → #99** (local→official projection calibration + tree-A/B harness readiness for land #71).

## 2026-06-14 05:22 — PR #94: Draft-verify overlap gate — can the drafter be hidden behind verify on a BW-bound A10G? ✅ MERGED — AMBER / OVERLAP LANE CLOSED (single-GPU conc=1; banks the reusable A10G dual-stream contention probe + `bus_contention_factor=0.506`)

- **Branch:** `denken/draft-verify-overlap-gate` · **Student:** denken · merged ~05:22Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** at conc=1 the decode loop is serial drafter(N)→verify(N)→drafter(N+1)…; Saguaro/AMUSD-style secondary-stream overlap runs drafter(N+1) concurrently with verify(N). #75 priced the drafter block at 15.5% → naive fully-hidden ceiling ~+18% TPS. Gate: does it survive the A10G single-HBM-bus contention?
- **Primary metric:** `bandwidth_limited_overlap_ceiling_pct = 4.22%` wall / 4.41% TPS (AMBER). **Test:** `drafter_verify_step_time_ratio r = 0.183` (M=8), 0.178 (M=32).

| metric | value | read |
|---|---|---|
| `drafter_verify_step_time_ratio` (r, M=8) | **0.183** | drafter cheap → timing gate stays OPEN (CLOSE iff r>0.85) |
| naive compute-limited ceiling | +15.46% wall / **+18.29% TPS** | reproduces the PR's +18% projection exactly |
| verify solo HBM | **491 GB/s (82% peak)** | one stream nearly saturates the bus |
| two concurrent verify | **1.97× one verify** | fully serialized (symmetric speedup 1.01×) |
| `bus_contention_factor` | **0.506** | ~full serialization on the shared bus |
| `drafter_overlap_efficiency` | **0.273** | only 27% of the drafter hides |
| **bandwidth-limited ceiling** | **+4.22%** wall / +4.41% TPS | 0.273 × 15.46% — AMBER |
| realized after accept-boundary haircut | **+1.16 / +2.09 / +2.86%** (1/2/3-path) | official 487.1 / 491.6 / 495.3 |

- **Verdict — AMBER → LANE CLOSED.** The timing gate does NOT close it (r=0.183 ≪ 0.85: the drafter is ~5.5× cheaper than verify, so a *compute-limited* world would hide it almost fully). **The A10G's single HBM bus is the wall:** verify alone pulls 82% of HBM peak, two memory-bound streams serialize (1.01× symmetric speedup, contention 0.506), combined bandwidth ≈ single-stream (498 GB/s, not the ~982 GB/s additive overlap needs). So the naive +18% collapses ~4× to **+4.22% bandwidth-limited**, then to **+1.2-2.9% realized** after the serial accept-boundary haircut (zero-accept P=0.271). Saguaro/AMUSD's "free drafter" relies on a **separate device with its own HBM**; the premise does not transfer to one A10G at conc=1. Not worth a dual-stream scheduler + speculative continuation tree + rollback for sub-3%, and it **fights the tree** for the same non-GEMM slack (#85).
- **Banks:** `scripts/profiler/dual_stream_hbm_contention.py` (reusable A10G dual-stream probe), `scripts/profiler/draft_verify_overlap_gate.py`, and the reusable **`bus_contention_factor=0.506`** A10G constant — any future "overlap two memory-bound kernels at conc=1" idea should assume ~full serialization. CPU gate + ~4.7 GB GPU probe; greedy token-identity preserved by construction (overlap reorders the GPU timeline only). Re-run regime: a compute-bound decode (much higher concurrency / larger M) returns the bus headroom.
- **W&B:** `1127zef4`. **Next:** denken → #97 (persistent-kernel overhead-reclamation gate — is the ~32% "other" GPU-idle or GPU-busy?).

## 2026-06-14 05:17 — PR #87: Verify-GEMM argmax-margin greedy-safety gate ✅ MERGED — GREEN (both verify-GEMM levers clear the FP-numerics gate; lm_head-isolated; banks the 65,536-position margin map)

- **Branch:** `kanna/verify-gemm-argmax-margin` · **Student:** kanna · merged ~05:17Z
- **Hypothesis:** the "lossless by construction" claim for ubel #84 SplitK + land #71 M-widen is unverified at the FP-numerics level — a reduction-order/tiling change could flip the greedy argmax (kanna's own #73 atomic-add control proved ~36% flips out-of-regime). Map the deployed verify's top-2 logit margin, emulate the SplitK K-partition + M=16/32, count argmax flips. GREEN = 0 flips → protects both levers from an FP-nonassoc disqualification BEFORE quota.
- **Primary metric:** `verify_gemm_argmax_flip_count_splitk = 0`. **Test:** `verify_gemm_min_top2_margin_ulp = 0` (min-positive 0.5 ULP, median 39 ULP).

| metric | value | read |
|---|---|---|
| `verify_gemm_argmax_flip_count_splitk` (primary) | **0** | GREEN — SplitK S∈{2,4,8} isolated reduction order, ≤1 bf16-ULP, split-INDEPENDENT |
| M-widen M=16 / M=32 (real Marlin) | **0 / 0** | M=16 bit-identical to M=8; M=32 ≤0.25 logit |
| provably flip-proof (margin > 2·max\|Δ\|) | **98.13%** | 64,310/65,536; residual 1,226 → 0 measured flips |
| exact bf16 ties | 907 (1.38%) | deterministic lowest-index tie-break (greedy-safe) |
| SplitK-vs-real-M=8 disagreements | 186 (0.28%) | = emu fidelity gap (FP32-emu vs FP16-Marlin-MAC), split-independent — NOT a lever indictment |
| positions audited | 65,536 | 128 prompts × 512 tok, official config |

- **W&B:** `875cujdk` (~45min GPU decode + ~3min CPU analysis, peak ~19.5GB/23GB A10G). Advisor-verified independently: all 7 numeric metrics + the logged artifact match reported values to exact; run `finished`, `verdict: GREEN`. Honest residual flags (`comfortable_headroom: false`, 1.87% measurement-dependent, 907 tie-bounded) all logged transparently.
- **Analysis/conclusions:** The #73 mechanism (reduction-order swap → argmax flip) does NOT trigger for the claimed-lossless swaps: margins are wide where they matter (median 39 ULP, 98.13% provably flip-proof) and at the genuinely thin positions the in-regime perturbation (≤1 bf16-ULP) is too small to flip — 0/907 ties broken even where the swap reached a full ULP. Mechanism: the n=12,288 vocab head forces vLLM Marlin into `use_fp32_reduce=True` + atomic-add OFF (`should_use_atomic_add_reduce()` False for n≥2048), so the only lossy step is the single final FP32→bf16 cast → reduction-order change capped at ±1 bf16-ULP. **Hand-off:** land #71 M-widen DIRECT GREEN (M=16 literally bit-exact), proceed to quota; ubel #84 SplitK GREEN with residual = 907 ties to confirm under the real kernel (margin map handed over). **Honest scope:** audits the lm_head projection (the GEMM feeding argmax); upstream network-wide compounding bounded by the per-layer ≤1-ULP regime argument + ultimately the official 128/128 gate → kanna #96 closes that residual directly. **ONE of two pre-quota numerics gates CLEARED** (wirbel #93 attention-side remains). Banks the 65,536-position margin map as a standing safety contract for any future verify-GEMM kernel change. kanna → #96 (network-wide compounding gate).

## 2026-06-14 05:07 — PR #92: Tree E[T] independence-gap ✅ MERGED — GREEN / DE-RISKED (realized tree E[T] matches independent model +0.025%; last analytical assumption in the 500-path confirmed under real correlated draws)

- **Branch:** `fern/tree-et-independence-gap` · **Student:** fern · merged ~05:07Z by morganmcg1
- **Hypothesis:** wirbel #83's +18.2% E[T] (~568 official) and fern #91's topology confirm both assume chain-rule independence (per-rank AND per-position) of drafter acceptance. Real top-W emissions are correlated (wirbel #86: confident drafter → higher ρ₂, r=−0.97). If correlation lowers realized tree E[T] below 5.207, land #71's projection is inflated; if it matches, the last untested assumption is de-risked.
- **Primary metric:** `ET_independence_gap_pct = +0.0247%`. **Test:** `realized_tree_ET = 5.20824`.

| metric | value | read |
|---|---|---|
| `ET_independence_gap_pct` (primary) | **+0.0247%** | GREEN if \|·\|≤3% — independence holds |
| `realized_tree_ET` (B, MTP regime mixture) | **5.20824** | == #86 ET_uniform_global to 8.9e-16 |
| `independent_tree_ET` (A, analytic) | 5.20695 | wirbel #83 ≈5.207 / fern #91 5.20695 ✓ reproduced |
| `independent_tree_ET` (A, MC 5×400k) | 5.20559 ± 0.00248 | fern #91 MC 5.2056 ✓ reproduced |
| conservative analytic \|gap\| bound | +2.267% | < 3% (GREEN even at the bound) |
| EAGLE-3 real-vs-shuffle (cross-position xcheck) | −1.78% | within ±3% |
| `land71_official_proj_recalibrated` | 568.1 | ~unchanged |

- **W&B:** `r9pq2qon` (CPU-only, 52s, 33.4 MiB RSS). Advisor-verified independently: all 5 numeric metrics + both tables (`et_estimators` 6 rows, `realized_per_regime` 5 bins) match reported values to full FP precision; run `finished` clean (the `_runtime=1s` is a known W&B CPU-only artifact, not a crash).
- **Analysis/conclusions:** Independence is **confirmed**, but the deeper reason is that draws are strongly correlated and the correlation is E[T]-neutral across all four channels: (1) spine cross-depth survivorship — ZERO gap by construction (depth-dependent q[d] ≡ ∏ chain-rule survivorship); (2) rescue depth-dependence — #79 ρ₂-by-depth flat (slope +0.0032/depth), pooling justified; (3) within-step confidence↔rescue (wirbel #86's headline r=−0.97) — REALIZED via a freq-weighted entropy-regime mixture over 13,491 steps → 5.20824 = pooled to machine precision (the pooled ladder IS the freq-weighted mean of regime ladders; E[T] near-linear → only a +0.025% Jensen residual, correlation marginally HELPS); (4) branch-continuation/cross-position — unmeasurable on the deployed linear MTP chain (cancels in the gap), cross-checked on the independent #80 EAGLE-3 trace via real-order-vs-shuffle bootstrap = −1.78% (mechanism: rank-1 runs 17× over-dispersed vs geometric → a few long runs spill past the spine-depth cap). No fresh GPU capture: a full-joint tree accept/reject capture is **structurally impossible** on a linear MTP chain (a rank-2 branch's own continuation is never drafted), so the committed ≥13k-step captures the PR pointed to are the correct data. **For land #71:** ~568–569 (denken #85 net ~576) STANDS; carry ±2–3% band (≈558–581). The only true test of channel 4 is land's first tree `accept_length` run (prior: the −1.78% EAGLE-3 number). **Analytical tree-economics lane now fully saturated** (#88 RED, #86 RED, #91 confirm, #92 confirm). fern → #95 (LK-Loss headroom gate).

## 2026-06-14 04:44 — PR #89: Prompt-lookup × MTP first-reject overlap ✅ MERGED — DROP / LANE CLOSED (realized +1.67% gross below +2% build bar; structural positive-correlation ceiling; prompt-lookup-augment lane CLOSED)

- **Branch:** `denken/promptlookup-augment-overlap` · **Student:** denken · merged ~04:44Z
- **Hypothesis:** prompt-lookup (n-gram draft) hits MTP first-reject misses (m=0 steps, 27% of decode) → complementary, not redundant; augmenting MTP with free PLD tokens at m=0 steps is the highest-value cheap augment behind land #71's fork.
- **Primary metric:** `promptlookup_realized_augment_tps_pct = +1.67%` [CI +1.36, +2.02]. **Test:** `promptlookup_mtp_firstreject_overlap_frac = 0.0354`.

| metric | value | read |
|---|---|---|
| `promptlookup_realized_augment_tps_pct` | **+1.67%** [+1.36, +2.02] | BELOW +2% build bar (gross, no fork cost subtracted) |
| oracle upper bound | +2.38% | hard ceiling (best-match pick) |
| `corr_q_vs_m` | **+0.354** | POSITIVE → redundant, not complementary |
| `promptlookup_mtp_firstreject_overlap_frac` | **0.0354** | 3.54% of m=0 misses get PLD accept-extending hit |
| `share_extra_from_m0` | 0.389 | below #81's independence assumption (54.5%) |
| independence UB (this trace) | +7.87% | realized only 21% of it |

- **Verdict: DROP.** Structural limiter: PLD hits POSITIVELY correlate with MTP acceptance (corr=+0.354) — redundant, not complementary. Oracle best-case caps at +2.38%. No realistic path to ≥+2% build-worthy net. **Prompt-lookup-augment lane CLOSED. Do NOT queue behind land #71.**
- **W&B:** `tz2oaemz`. denken → cycle-39 reassignment (persistent-kernel scheduling).

## 2026-06-14 04:27 — PR #86: Entropy–branching correlation ✅ MERGED — STRONG SIGNAL, NON-ACTIONABLE / LANE CLOSED (r=−0.9688, sign-reversed; oracle ceiling +0.27% E[T] / +0.33pp TPS; entropy-branching lane CLOSED)

- **Branch:** `wirbel/entropy-branching-correlation` · **Student:** wirbel · merged by morganmcg1 ~04:27Z
- **Hypothesis:** drafter uncertainty (token-level entropy) predicts rank-2 branching value — high-entropy steps benefit from deeper branching → entropy-gated dynamic tree delivers free E[T] over a static topology.
- **Primary metric:** `rho2_entropy_correlation_r = −0.9688`. **Test:** `entropy_gated_tree_E_T_gain_pct = 0.273`.

| metric | value | read |
|---|---|---|
| `rho2_entropy_correlation_r` | **−0.9688** | ONE OF STRONGEST SIGNALS IN PROGRAMME — but SIGN-REVERSED |
| sign direction | drafter CONFIDENCE (low entropy) → HIGHER ρ₂ | anti-direction from hypothesis |
| `entropy_gated_tree_E_T_gain_pct` | **+0.273%** | oracle ceiling on dynamic entropy-gated tree |
| TPS equivalent | ~+0.33pp | BELOW cost of forfeiting onegraph CUDA graph |
| pooled ρ₂ | 0.4172 | matches #79's 0.4165 ✓ (acceptance model self-consistent) |
| data collected | 13,491 first-reject steps | 128 prompts × 512 tok, greedy, seed 1, conc 1 |

- **Verdict: NON-ACTIONABLE.** drafter CONFIDENCE (low entropy) predicts HIGHER ρ₂ — so high-entropy steps (where branching would theoretically help most) are precisely where MTP acceptance is LOWEST. Signal is purely within-step; a static depth-indexed tree captures none of it (consistent with #83's flat per-depth ρ₂). Oracle ceiling +0.27% E[T] / +0.33pp TPS < cost of forfeiting onegraph CUDA graph. **Lane closes on actionability, not on null correlation.** The strongest r in the programme closes the weakest lever.
- **W&B:** `59u7qcwa` / `79u01jm8`. wirbel → cycle-39 reassignment (double-quant verify-GEMM scales).

## 2026-06-14 04:26 — PR #91: Tree topology E[T] — max-branch-3 vs max-branch-4 ✅ MERGED — CONFIRMED (+0.9614%, all three estimators agree, validates acceptance model; land #71: build max-branch-3)

- **Branch:** `fern/tree-topology-et-comparison` · **Student:** fern · merged by morganmcg1 ~04:26Z
- **Hypothesis:** wirbel #83's DP-optimal max-branch-3 topology buys +0.96% E[T] / +1.13pp TPS over max-branch-4 (both depth-9, M=32). This analytic prediction is directly measurable by MC simulation using fern's #88 harness.
- **Primary metric:** `topology_et_delta_pct = +0.9614%`. **Test:** `topology_et_confirmed = 1`.

| estimator | E[T] mb3 | E[T] mb4 | delta_pct | SE |
|---|---|---|---|---|
| **analytic** (exact, `score_tree_depthrank`) | 5.206954 | 5.157273 | **+0.9633%** | 0 |
| **independent MC** (2M trials/topo, 5 seeds) | 5.20559 | 5.15602 | **+0.9614%** | ±0.073pp |
| **CRN paired** (6M trials, 3 seeds) | 5.20790 | 5.15859 | **+0.9560%** | ±0.003pp |
| wirbel #83 predicted | — | — | +0.9633% | — |

- CRN 95% CI = **[+0.950%, +0.962%]** — entirely above the +0.8% CONFIRMED threshold; gap to wirbel's analytic 0.002pp. #88 Leg A reproduced bit-for-bit (engine integrity). 0 greedy violations on both topologies. `score_tree_depthrank` ≈ MC to ~1e-3 tok → future tree-topology DP results trusted analytically without per-candidate MC. **Build recommendation for land #71: max-branch-3 array** `[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]` (depth-9, spine widths [3,3,2,2,1,1,1,1,1]). Mechanism: only ~23.8% of decode steps show any topology difference; mb4 wastes a node on rank-4 (marginal ≈0.022, ρ₄=0.1908) while mb3 reallocates to rank-2 breadth (ρ₂=0.4165 ≫ ρ₄).
- **W&B:** `exkahicq` (CPU-only, 33.3 MiB RSS, 155s). fern → cycle-39 reassignment (LK-loss draft head).

## 2026-06-14 04:07 — PR #88: Traversal Verification E[T] gate ✅ MERGED — RED / AXIS CLOSED (provably zero under greedy; standard root-to-leaf confirmed; land #71 keeps existing acceptance rule)

- **Branch:** `fern/traversal-verify-et` · **Student:** fern · merged ~04:07Z
- **Status:** MERGED as decisive RED characterization keeper (CPU analytical + MC simulation; no GPU training, no served-file change; official bar UNCHANGED 481.53). Banks `scripts/profiler/traversal_verify_et.py` + `research/spec_cost_model/traversal_verify_et_results.json`.
- **Hypothesis:** Traversal Verification (NeurIPS 2025 OpenReview 8nOMhDFpkU) — leaf-to-root tree acceptance — recovers sibling-subtree mass from wirbel #83's salvage oracle (rho2=0.4165), potentially delivering a free, provably-lossless E[T] uplift on land #71's M=32 tree.
- **Primary metric:** `traversal_et_uplift_pct = 0.000`. **Test:** `traversal_greedy_violation_count = 0`.

| Leg | Regime | E[T] root→leaf | E[T] leaf→root | uplift % | greedy viol. |
|---|---|---|---|---|---|
| **A** physical M=32, 400k MC | **greedy (T=0)** | **5.2140** | **5.2140** | **+0.000** | **0** |
| **B** sampling-proxy contrast | sampling proxy | 4.4324 | 4.6348 | +4.567 | 26,984 |
| **C** real #80 ranks, 1868 steps | greedy | 3.3330 | 3.3330 | +0.000 | 0 |
| **D** exhaustive all trees n≤6 | greedy-valid | — | — | 0.000 | 0 / 872 trees |

- **Verdict: RED.** Traversal Verification is **provably zero under greedy decode** — structural, not a measurement artefact. Under temperature 0, the target argmax at each position is a single token, so at most one child can match at any tree node. The consistent paths form a unique chain; both walks return the same chain → E[T] uplift 0, greedy violations 0, for any tree/corpus. wirbel's rho2=0.4165 salvage mass is **fully realized by root-to-leaf** (it is the value of the tree topology over the linear chain, not incremental headroom for a different acceptance rule). Leg B confirms the mechanism exists in sampling regimes (+4.57%) but is vacuous at T=0. **Acceptance-rule axis CLOSED. land #71: keep standard root-to-leaf verification.**
- **W&B:** `yiwl2jfj`. fern → #91 (tree-topology E[T] comparison: max-branch-3 vs max-branch-4, using this harness).

## 2026-06-14 03:53 — PR #82: Operationalized wall_tps: paired-A/B runner + re-baseline + #56 re-screen ✅ MERGED (infra keeper: canonical A/B entrypoint + locked re-baseline 454.09 + confirmed deployed MBT=512 optimal)

- **Branch:** `lawine/walltps-ab-runner` · **Student:** lawine · merged ~03:53Z
- **Status:** MERGED as infrastructure keeper (no served-file change; official bar UNCHANGED 481.53). Banks `scripts/profiler/paired_tps_ab.py` as the canonical one-command paired-`wall_tps` A/B entrypoint.
- **Hypothesis:** the #72 `wall_tps` protocol (CV 0.035%, MDE ≥0.1% N=3) is proven but not yet operationalized into a reusable tool. A one-command paired-A/B runner lets every lever-builder (land #71, ubel #84, stark #78, kanna #87) decide on the same robust metric without re-implementing the harness.
- **Primary metric:** `deployed_local_wall_tps = 454.085` (locks re-baseline). **Test:** `paired_ab_self_null_gain_pct = 0.030` (NULL ✓ = unbiased).

| arm | median wall_tps | Δ vs deployed 512 | verdict |
|---|---|---|---|
| **A=B self-null (baseline check)** | 454.09 | +0.030% | NULL ✓ (unbiased) |
| MBT 512→2048 (real-change validation) | 450.83 | −0.716% | REAL ✓ (~9× MDE, sensitive) |
| MBT 512→4096 | 453.54 | −0.120% | REAL (small regression) |
| MBT 512→8192 | 453.06 | −0.226% | REAL (small regression) |

- **Conclusions:** (1) Runner validated — unbiased (self-null NULL +0.030%) + sensitive (real change REAL at ~9× MDE). (2) Re-baseline locked at **454.09 wall_tps** (CV 0.007%, confirms #72's N=12 454.12; retires fragile 428.37). (3) **#56 re-screen: no hidden win** — deployed MBT=512 is at/near optimum; all increases are small REAL regressions (E[accept] drifts 3.853→3.879 but scheduling overhead dominates at conc=1). (4) `paired_tps_ab.py` supports `--candidate-env` serve-time overrides, `--reuse-baseline-from` for multi-arm re-screens, structured `paired_ab.json` + W&B logging. First real lever job (topology A/B, re-opt max-branch-3 vs #74, ~+1.13pp) queued behind land #71. lawine → #90 (MTP K sweep).
- **W&B (group walltps-ab-runner):** selfnull `2mq96qz1` · detok_off `dorrmq8l` · mbt2048 `xmwqvtmk` · mbt4096 `5ny0egab` · mbt8192 `pvg56gnm`.

## 2026-06-14 03:29 — PR #85: Tree-verify non-GEMM overhead audit at M=32 ✅ MERGED (decisive GO: non-GEMM tree machinery 2.597% decode, ~8× smaller than the +21.8% GEMM gain → net +19.82% survives; no O(M²); attention amortizes 1.06×; hands land #71 a per-op cost-budget oracle)

- **Branch:** `denken/tree-overhead-audit` · **Student:** denken · merged by morganmcg1 ~03:28Z
- **Status:** MERGED as research artifact (local profiling, no GPU-train, no served-file change → BASELINE official bar UNCHANGED 481.53; net +19.82% is a PROJECTION off the frontier, not a measured submission). Banks `scripts/profiler/tree_nongemm_overhead.py` + `research/spec_cost_model/report_tree_nongemm_overhead.md` + `tree_nongemm_overhead.json`.
- **Hypothesis:** the +21.8% M=32 tree-verify re-price (wirbel #79/#83) prices the GEMM SAVINGS but not the tree's NON-GEMM systems OVERHEAD (mask construct, scatter/gather, M-row sampler-prep, valid_counts scheduling). Auditing it in isolation either confirms the net gain holds or reveals erosion before land #71 spends an approval-gated launch.
- **Primary metric:** `tree_overhead_nongemm_pct_decode = 2.597` (% decode at M=32 static, W&B `f0c8mb39`). **Test:** `net_tree_gain_after_overhead_pct = 19.82`.

| quantity (M=32 DP-tree vs M=8 linear, 11.6ms step) | value |
|---|--:|
| non-GEMM overhead (static, mask precomputed) | 2.597% (301µs) vs +21.8% GEMM → ~8× smaller |
| Δ vs M=8 linear (the slice eroding the gross) | +1.65pp (192µs) |
| verify-side ONLY (excl. drafter M-row sampler) | 0.512% (59µs) |
| net tree gain after overhead (static) | +19.82% (gross 21.8 − 1.98pp) |
| attention M=32/M=8 (split-KV 3D FlashDecoding) | 1.06× (≪4×, KV read shared) |
| only [M,M] op (ancestor mask) scaling exp | 0.16 (≈flat); 0/step precomputed static |

- **Conclusions:** (1) Tree non-GEMM machinery is ~8× smaller than the GEMM gain it unlocks → net **+19.82%** (≈576 official projected, 3-base). (2) **NO O(M²)** anywhere — ancestor mask exp 0.16 ≈flat, refutes the byteshark O(M²)-mask risk; the two M-growing ops (drafter M-row sampler, full-vocab verify-argmax) are ≈O(M) linear. (3) **Attention amortizes 1.06×** — #43 split-KV routes all M≤64 verify rows to 3D FlashDecoding, shared-prefix KV read once → attention stays at floor (closes the #69-at-M=32 question). (4) Hands land #71 a **per-op cost-budget oracle** (expected µs + 1.5× ceiling) — PERFORMANCE half of its debug gate, pairs with wirbel #83's ≈0.41 salvage (correctness half): op over budget = byteshark layout bug, caught pre-launch. (5) **Side-finding:** denken's roofline (weight-bandwidth-bound, free-to-M≤32) + this audit (KV shared, mask ~0) RULE OUT the tile-scheduler / KV-layout / fused-mask kernel levers at M≤32 → **SplitK (ubel #84) is the only live verify-GEMM kernel lever** (a useful cohort negative). (6) Methodology: eager 327µs → graph-basis 37µs (8.8× launch artifact) — the #77 lesson. (7) denken → #89 (prompt-lookup × first-reject overlap build-or-kill).

## 2026-06-14 03:25 — PR #80: Multi-step (HASS) drafter training — break the K=1 chain-collapse ceiling ✅ MERGED (thesis CONFIRMED +57.8% native accept/step, but E[T]=2.23 ≪ MTP 3.844 → bank-and-close; EAGLE-3 single-layer drafter-training lane CLOSED)

- **Branch:** `fern/eagle3-multistep-hass` · **Student:** fern · merged by advisor ~03:13Z
- **Status:** MERGED as research artifact (offline training/eval, no served-file change → BASELINE official bar UNCHANGED 481.53; confirmed BELOW frontier). Banks the serve-faithful HASS unroll machinery (`train_eagle3.py --unroll_steps`, detached depth unroll, native-acceptance sim).
- **Hypothesis:** the K=1 teacher-forced regime (not the corpus) was the binding native-acceptance ceiling on the EAGLE-3 drafter; HASS multi-step (J=3) unroll — feeding the draft its own rolled-forward hidden — lets per-step acceptance sustain past step-1 instead of collapsing.
- **Primary metric:** `native_accept_per_step_bench_holdout = 1.2294` (HASS J=3, K=8, W&B `at46onde`). **Test:** `..._34ckpt = 0.7792` (#34 K=1 baseline, W&B `bsu901oj`; training `pkcmx1zl`).

| metric (240-rec holdout, K=8, shared harness) | #34 (K=1) | HASS J=3 | Δ |
|---|--:|--:|--:|
| native accept/step (primary) | 0.7792 | 1.2294 | +57.8% |
| E[T] = tok/target-forward | 1.7792 | 2.2294 | +0.4502 |
| step-2 conditional accept | 0.8% | 32.4% | 38.6× |
| tf top-1 (secondary) | 0.7617 | 0.7475 | −1.9% |

- **Conclusions:** (1) Thesis CONFIRMED decisively — HASS unroll lifts step-2 conditional accept 0.8%→32.4% (the own-hidden hand-off the chain died at) and native accept/step +57.8%; the draft genuinely learned to condition on its own rolled-forward hidden. (2) **NOT a frontier candidate** — E[T]=2.23 = 58% of MTP's 3.844, ~1.6 tok short; the ceiling is architectural (single-layer head capacity), not a training schedule. (3) openevolve independently found every retrained MTP head lands at parity too → the parity finding generalizes across head families. (4) Measurement handled right — gated on a serve-faithful sim with a large margin to the 3.844 bar, not a fragile HF proxy (the openevolve over-report caveat). (5) **EAGLE-3 single-layer drafter-training lane CLOSED** per fern's own recommendation; HASS machinery banked as a drop-in if head capacity is ever raised. (6) fern → #88 (Traversal Verification E[T] gate).

## 2026-06-14 03:24 — PR #73: Greedy-identity — is the frontier stack bit-exact or distributional? ✅ MERGED (refutes the premise: deployed stack is BIT-EXACT run-to-run AND satisfies the contract at ~489 local TPS; determinism is ENGINEERED — atomic-add-off is load-bearing; + analyze_determinism.py bugfix)

- **Branch:** `kanna/greedy-determinism` · **Student:** kanna · merged by advisor ~03:13Z
- **Status:** MERGED as research artifact (local measurement, served stack UNCHANGED → BASELINE official bar UNCHANGED 481.53). Banks `scripts/validity/greedy_determinism.py` + `analyze_determinism.py` (with bugfix) + captures.
- **Hypothesis:** the deployed stack is run-to-run token-nondeterministic, so greedy-identity can only be a distributional property (the #66 contract-foundation question).
- **Primary metric:** `greedy_identity_verdict = 0` (bit-exact, W&B `lr1ornnl` N=10 / `45y7ui1o` N=7). **Test:** `fa_sliding0_tps_cost_pct = 0.03` (FA_SLIDING=0 is a no-op, +0.03% within noise).

| config | N | mean byte-id | official greedy_gate | verdict |
|---|--:|--:|---|---|
| **default (deployed)** | 10 | **1.0** | GREEDY_IDENTICAL (0/65536 div) | bit-exact |
| splitkv_off | 3 | 1.0 | GREEDY_IDENTICAL | stable |
| **atomic_on (positive control)** | 7 | **0.8214** | DIVERGENT (35.7% tok) | breaks bit-exactness |

- **Conclusions:** (1) Premise REFUTED — the deployed spec-ON M=1 greedy frontier is BIT-EXACT run-to-run (N=10 fresh reloads, official GREEDY_IDENTICAL, 0/65,536 divergent tokens, flip hazard 0.0) AND satisfies the contract at ~489 local TPS — a third option the PR's dichotomy excluded. (2) **Determinism is ENGINEERED, not luck** — the atomic-add positive control (forcing VLLM_MARLIN_USE_ATOMIC_ADD=1 flips ~36% tokens) proves keeping it OFF is load-bearing; FA2 sliding is inert (flips 0 layers), #43 split-KV is frozen inside the captured graph. (3) The prior "non-reproducible" numbers (BASELINE line 49, lawine #56) were PROXY configs (spec-OFF, plain int4), not the deployed stack. (4) Churn cannot move the gates — PPL invariant to 12 digits (teacher-forced), private TPS gate stable (~0.2% spread). (5) **Bugfix:** analyze_determinism.py had a false hardcoded "atomic-add hardware-gated OFF" prior contradicted by its own data → replaced with data-driven branching + cluster_signatures() + atomic_add_breaks_determinism field. (6) The atomic-add control PROVES the kernel-swap→argmax-flip mechanism → kanna's own follow-up #3 (margin map) = the **argmax-margin gate**, reassigned to her as **#87**.

## 2026-06-14 03:15 — PR #83: Re-optimize M=32 tree topology with measured rho ladder + salvage oracle ✅ MERGED (decisive positive: max-branch-3 optimal, +1.13pp over #74; salvage oracle delivered; headline corrected to +18.2% / ~569 official; wirbel acceptance-cost-model axis CLOSED)

- **Branch:** `wirbel/rho-optimal-topology` · **Student:** wirbel · merged by advisor ~03:15Z
- **Status:** MERGED as research artifact (CPU-only analytic, no GPU, no served-file change -> BASELINE official bar UNCHANGED 481.53). Banks `scripts/profiler/rho_optimal_topology.py` + `research/spec_cost_model/report_rho_optimal_topology.md` + `rho_optimal_topology_results.json`.
- **Hypothesis:** re-running the Sequoia/DP tree optimization with the measured DECLINING rho ladder (rho2=0.4165 >> rho3=0.2655 > rho4=0.1908) instead of #74's borrowed flat rho=0.565 yields the true-rho-optimal M=32 topology (parent arrays land #71 should build) + the expected per-position salvage curve for land's debug-gate oracle.
- **Primary metric:** `measured_rho_optimal_M32_gain_pct = 0.1817` (+18.17% drafter-aware re-priced gain, W&B `6tghbnjn`). **Test:** `expected_pooled_branch_hit_salvage = 0.4165` (rho2 debug-gate target).
- **Results:**

| topology | E[T] | max-branch | gain (drafter-aware) | wall_tps proj |
|---|---|---|---|---|
| #74 (flat rho=0.565) | 5.157 | 4 | +17.04% | 531.5 |
| **#83 measured-rho-optimal** | **5.207** | **3** | **+18.17%** | **536.6** |
| delta re-opt | +0.96% | -1 (rank-4 dropped) | +1.13pp | +5.1 |

3 bases: **+18.2% relative / wall_tps x454.1 -> 536.6 / official x481.53 -> ~569 projected**. MC cross-check: 400k-trial sim E[T]=5.214 vs analytic 5.207 (|err|=0.007). Anchor: F_linear(8)=3.84445 == measured 3.8441.

- **Salvage oracle (land #71 per-position debug gate):**

| spine pos | width | E[salvage rank-2] |
|---|---|---|
| 1 | 3 | 0.397 |
| 2 | 3 | 0.431 |
| 3 | 2 | 0.413 |
| 4 | 2 | 0.428 |
| 5-9 | 1 | 0 |

Universal rank-2 gate = rho2 0.4165. A correct width-2 branch at any divergence reads ~0.41; byteshark broken tree-v2 read 0.033 (12x discrepancy = layout-bug signature).

- **Cost-model deviation:** g_d=0.168 drafter-depth term added (MTP runs `depth` sequential passes, 15.5-18.1% of step). Under the SAME M-only cost as #79, re-opt reads +23.3% vs +22.2% for #74 -> topology improved; the headline drop (from +21.8% to +18.2%) is an honesty correction for depth cost. The re-opt DELTA (+0.96%/+1.13pp) is cost-model-independent (both depth-9).
- **Conclusions:** (1) Declining rho ladder makes max-branch-3 optimal (width-4 buys +0.00pp under measured rho); (2) beyond-width-4 never pays (rank-5 leaf marginal 0.0179 < least placed node 0.0272); (3) salvage oracle delivered to land #71 (universal rank-2 gate = 0.4165); (4) headline corrected to +18.2% (~569 official) -- still well above 500 target and 488.07 competitor; (5) +1.13pp topology delta should be confirmed by lawine #82 A/B before banking; (6) wirbel's acceptance-cost-model axis (#49->74->76->79->83) is CLOSED for the tree build.

## 2026-06-14 02:54 — PR #81: Prompt-lookup/n-gram hybrid drafter — free accepted tokens on top of MTP (Step-0 gate) ✅ MERGED (decisive Step-0 gate: CLEARS at 0.4066 extra-accept, but realistic +1-3% & not buildable in stock vLLM → do-not-build-now, queued behind land #71 fork)

- **Branch:** `denken/prompt-lookup-drafter` · **Student:** denken · merged by morganmcg1 02:54Z
- **Status:** MERGED as research artifact (CPU-only, no-GPU, no-HF-Job, zero served-file change → BASELINE official bar UNCHANGED 481.53). Banks `scripts/analyze_prompt_lookup.py` + the verdict.
- **Hypothesis:** prompt-lookup/n-gram (PLD) gives training-free accepted tokens from self-repetition; AUGMENT mode (PLD on top of MTP, not replace) could add free tokens where MTP misses. Step-0 gate: self-ngram ≥~0.3 extra accept tok/step on the 128 public prompts → build; below → kill.
- **Primary metric:** `promptlookup_extra_accept_tokens_per_step = 0.4066` (W&B `ed46yvkz`). **Test:** `promptlookup_augment_tps_uplift_pct_independence_ub = 10.64`.

| metric | value | notes |
|---|--:|---|
| extra accept tok/step (vLLM-faithful) | **0.4066** | clears the ≥0.3 gate |
| TPS uplift (independence UB) | **+10.6%** | upper bound (q–m independent) |
| oracle best-occurrence UB | 0.494/step / +12.9% | absolute ceiling |
| n=2 ngram hits: generated vs prompt | 0.316 vs 0.207 | reasoning DOES self-repeat |
| extra tokens from MTP-full-miss steps (m=0) | 54.5% | the most valuable free tokens |
| realistic uplift (q–m correlated) | **+1-3%** | conservation-constrained full-span, a_H 0.90-0.95 |
| REPLACE mode E[T] | 1.45-1.51 | ngram-only LOSES to MTP 3.84 |

**Conclusions:**
1. **Gate CLEARS but the lever is modest + blocked.** Extra-accept 0.4066/step (>0.3) and reasoning self-repeats (generated n-gram 0.316 > prompt 0.207), refuting "reasoning is generated not copied → PLD won't fire." 54.5% of free tokens come from steps where MTP fully missed (most valuable kind).
2. **+10.6% is an INDEPENDENCE upper bound; realistic = +1-3%.** Under realistic positive q–m correlation (PLD fires on predictable spans where MTP already wins), the conservation-constrained sweep pulls it to +1-3%, → 0 at perfect correlation. True gain needs the q–m correlation pinned.
3. **Decisive blocker is COMPOSABILITY, not magnitude.** vLLM 0.22.0 `SpeculativeConfig.method` is single-choice — mtp XOR ngram. Only stock mode is REPLACE (ngram-only E[T]=1.45-1.51 LOSES to MTP 3.84). AUGMENT (the valuable mode) needs a fork of the spec-decode proposer loop + composition with land #71's tree-verify → do-not-build-now.
4. **denken reassigned → #85 (M=32 tree-verify non-GEMM overhead audit).** Prompt-lookup queued behind land #71's proposer-loop fork (revisit + pin q–m correlation after tree-verify lands). denken's GPU goes to de-risking the headline: the tree-overhead performance oracle complementing wirbel #83's salvage correctness oracle.

## 2026-06-14 02:46 — PR #36: int4-quantize the pruned 12k lm_head 🔒 CLOSED-BANKED (clean terminal LOCAL rung; lm_head lever exhausted; ubel → #84 SplitK verify-GEMM)

- **Branch:** `ubel/...` (int4 lm_head) · **Student:** ubel
- **Status:** CLOSED-BANKED (terminal local rung, NOT a frontier advance → no BASELINE change; 481.53 official stays). Result preserved here + on branch (recoverable). PR was never un-drafted and carried only a `terminal:false, pending_arms:true` marker (the `terminal:true` text in-thread was the advisor's template with `[...]` placeholder run-ids).
- **Hypothesis:** int4-quantize the pruned-12k lm_head (4× head-byte cut) on the PR #14 bf16-12k rung (131.60 local) for a cross-session-deterministic, contract-safe TPS bump.
- **Result:** **133.299 local TPS (+1.3% over 131.60), PPL 1.9713** (drift +0.0001), **GREEDY_IDENTICAL 128/128**, head 62.9MB→16.22MB, cross-session deterministic (real int4 Marlin GEMV bit-exact). 4th datapoint on the lm_head-bytes↔TPS bandwidth model, landed on projection (133.3 predicted).

| metric | value | notes |
|---|--:|---|
| local TPS (lmhead12k rung) | **133.299** | +1.3% over 131.60 bf16-12k rung |
| PPL (128rec) | **1.9713** | drift +0.0001 vs bf16-12k |
| greedy identity | **128/128** | int4 Marlin GEMV bit-exact, cross-session deterministic |
| head bytes | 62.9MB → 16.22MB | 4× cut |

**Conclusions:**
1. **Clean, validated work — but the lm_head lever is EXHAUSTED.** lm_head is only ~1% of decode (wirbel #30), so even a 4× byte cut is a negligible full-stack lever; the 133.3 rung is sub-frontier vs the 481.53 fa2sw split-KV frontier (PR #52).
2. **Cannot compose into the frontier:** the pruned-12k vocab would break full-vocab greedy-identity if dropped into the 481.53 stack. Issue #35 (Morgan-closed: "close this and move on") retired the lmhead12k LAUNCH lane → no official-benchmark path remains for this rung.
3. **Banked, not merged:** no valid terminal marker + sub-frontier + lever exhausted → closed with the record preserved rather than adding a sub-frontier submission to the tree.
4. **ubel reassigned → #84 (SplitK W4A16 verify-GEMM):** promote ubel's int4-Marlin-kernel experience to the #1 decode block — close denken #68's 23% HBM gap on the verify-GEMM (53% of decode, 77.1% HBM at M=8) via SplitK K-decomposition (arXiv:2402.00025). Lossless/greedy-safe, composes with land #71, ~+5-12% wall_tps ceiling.

## 2026-06-14 02:50 — PR #79: Pin rank-2+ drafter coverage (ρ) — the last borrowed input to the +18.7% tree gain ✅ MERGED (decisive measurement positive: ρ₂=0.4165/ρ₃=0.2655/ρ₄=0.1908; cov₂₋₄=0.6532 > borrowed 0.565 — gain was CONSERVATIVE; byteshark cross-val PASS; full max-branch-4 justified; ~586 official)

- **Branch:** `wirbel/rank-coverage` · **Student:** wirbel
- **Status:** MERGED as research artifact (cost-model measurement; no served-file change → BASELINE official bar UNCHANGED 481.53). Lands `scripts/profiler/rank_coverage.py`, `scripts/profiler/rankprobe_patch.py`, `scripts/profiler/treeshape_measured_accept.py`, `research/rank_coverage/rank_coverage_results.json` + pr79_report.md, `research/accept_calibration/treeshape_measured_results.json`.
- **Hypothesis:** The +18.7% M=32 tree-verify gain (wirbel #74, borrowed flat ρ=0.565 from EAGLE-3) has one open parameter: **ρ = rank-2+ drafter coverage** (P(target in drafter top-k | rank-1 rejected at first divergence)). If ρ < 0.565 the gain is overstated; if ρ > 0.565 it was conservative. Measuring locally from the deployed kenyan-duma MTP drafter's top-4 outputs vs the verifier's greedy path pins the last borrowed input, and cross-validating against byteshark's official-stack ρ₂=0.4130 (BLOCK=64) confirms the measurement is a drafter intrinsic, not a config artifact.
- **Primary metric:** `drafter_rank2_coverage = 0.4165` (ρ₂, W&B `z6wi4z4v` + `6wr8r2y0`). **Test:** `reprice_M32_proj_tps = 521.64` (M=32 re-priced on measured ρ ladder, old local base).

| metric | value | notes |
|---|--:|---|
| ρ₂ (rank-2 coverage) | **0.4165** | 12,869/30,874 first-reject divergence events where target in top-2 |
| ρ₃ (rank-3 coverage) | **0.2655** | declining ladder (not flat) |
| ρ₄ (rank-4 coverage) | **0.1908** | above cost threshold ~0.10 → width-4 still pays |
| cov₂₋₄ (aggregate top-4) | **0.6532** | > borrowed EAGLE-3 0.565 → gain was CONSERVATIVE |
| beyond-top-4 hard miss | **34.7%** | rank-5+ below GEMM-row cost → max-branch-4 confirmed optimal |
| top-1 q0 (cross-check #76) | **0.7335** | vs #76's 0.7287 (Δ0.0048) ✓ |
| per-depth ρ₂ | flat 0.397–0.445 | no depth trend across depths 1–7 |
| align_bad (greedy preserved) | **0** | 16,524 records; greedy-identity INTACT |
| M=32 re-price (old local base) | **+21.8% / 521.6 TPS** | ≈586 official (481.53×1.218) |
| byteshark cross-val Δ ρ₂ | 0.85% (0.4165 vs 0.4130) | PASS — drafter property, not block-size artifact |
| byteshark cross-val Δ cov₂₋₄ | 1.16% (0.6532 vs 0.6609) | PASS |
| byteshark cross-val Δ mean_emit | 1.96% (3.844 vs 3.921) | PASS |

**Conclusions:**
1. **ρ₂=0.4165 measured (replaces borrowed EAGLE-3 0.565).** The declining ladder ρ₂>ρ₃>ρ₄ reflects kenyan-duma MTP drafter's actual per-rank distribution; the borrowed scalar (35% above ρ₂) was wrong but the aggregate cov₂₋₄=0.6532 > 0.565 means the gain estimate was CONSERVATIVE, not overstated.
2. **Full max-branch-4 is justified.** ρ₂=0.4165 >> ρ₃=0.2655 > ρ₄=0.1908 >> ~0.10 threshold; 34.7% hard-miss beyond top-4 is below GEMM-row cost → M=32 max-branch-4 confirmed optimal per #68's roofline. wirbel #74's parent arrays stand; land #71 should build.
3. **M=32 re-price: +21.8% central → ~586 official.** 481.53×1.218≈586 TPS; well past 500-TPS target, clears need-for-speed 488.07 by ~20%.
4. **byteshark cross-validation PASS at all 3 metrics (<2%).** ρ₂ 0.85% / cov₂₋₄ 1.16% / mean_emit 1.96% between local BLOCK=16 and byteshark official-stack BLOCK=64 — coverage ladder is a drafter intrinsic.
5. **Per-depth ρ₂ flat (0.397–0.445, depths 1–7)** — depth-flat DP model used in #74 is justified; no depth correction needed for the topology.
6. wirbel reassigned → **#83 (rho-optimal-topology)**: re-run Sequoia DP with measured declining ρ (ρ₂=0.4165/ρ₃=0.2655/ρ₄=0.1908) to verify #74 is ρ-robust or find a better topology, and produce per-position expected-salvage oracle for land #71's debug gate (target ≈0.41 pooled vs byteshark's broken 3.3%).

## 2026-06-14 02:12 — PR #72: TPS measurement protocol — tighten the ±4.4% noise floor ✅ MERGED (decisive measurement positive: ±4.4% was estimator artifact; wall_tps CV 0.035% / MDE 0.2% N=1; wandb_logging bug fixed)

- **Branch:** `lawine/tps-noise-floor` · **Student:** lawine
- **Status:** MERGED as a research artifact + bug fix (measurement harness + `scripts/wandb_logging.py` 1-line fix; no served-file change → BASELINE official bar UNCHANGED 481.53). Lands `research/tps_noise_floor/` harness (N=12 run data, analysis scripts, PROTOCOL.md) + 1-line fix to `scripts/wandb_logging.py` (group kwarg was hardcoded → `--wandb_group` was a no-op for ALL callers).
- **Hypothesis:** The ±4.4% same-config TPS noise floor (from #56) is larger than our experiment deltas → the team can't tell a real <5% gain from noise. A hardened measurement protocol would shrink the effective noise floor and give a defensible MDE for every future TPS A/B.
- **Primary metric:** `tps_noise_floor_cv = 0.035` (wall_tps CV = **0.035%**, W&B `n07jrhxl`). **Test:** `tps_mde_pct_wall_paired_n1 = 0.095` (MDE = **0.095%** at N=1 paired on wall_tps).

| metric (A10G, deployed fa2sw_precache_kenyan, N=12 fresh, 128×512, conc=1) | mean TPS | CV | notes |
|---|--:|--:|---|
| `steady_gen_tps_mean` (old metric, fragile) | 449.06 | **0.33%** | unweighted interval-mean; cold 1st interval drags it |
| **`wall_tps` = tokens / decode_duration_s** | **454.12** | **0.035%** | = official `output_throughput` defn; **NEW STANDARD** |
| windowed steady (drop W=3 cold intervals) | 459.83 | 0.05% | robust interval-meter variant |
| `e_accept_exact` | 3.855 | 0.07% | near-deterministic greedy acceptance |
| **#56 reproduced on wall_tps** | 429.04→448.01 (+4.42%) | → **454.30→454.35 (+0.01%)** | throughput never moved; estimator did |

**Variance decomposition:**
- **(a) Warmup/cold-start — dominant for raw estimator.** First interval 29% below steady; dropping W≥1 collapses raw CV 0.33%→0.07%.
- **(b) Steady jitter — small.** windowed CV 0.05%, wall CV 0.035% (irreducible floor).
- **(c) Thermal/clock drift — ZERO.** A10G SM clock pinned 1710 MHz; temp flat ~53°C; TPS~time slope −0.006 tps/run (n.s.).
- **(d) Token nondeterminism — negligible.** E[accept] CV 0.07%; not a meaningful TPS-noise source.

**Recommended protocol (copy-paste):** decide every local TPS A/B on `wall_tps`, median N=3 fresh decode-only runs; MDE: **≥0.2% real at N=1; ≥0.1% at N=3**. Any delta ≥0.2% is real. Full protocol: `research/tps_noise_floor/PROTOCOL.md`.

**Conclusions:**
1. **The ±4.4% noise floor was an estimator artifact** — wall_tps was identical (+0.01%) across both #56 runs. The sub-5% wins concern is SOLVED by changing the metric, not by adding runs.
2. **New canonical local A/B metric: wall_tps ≈ 454** (replaces fragile "428.37 steady"; the 428.37 headline was the fragile metric's low point-estimate). Official bar unchanged 481.53.
3. **MDE 0.2% at N=1 / 0.1% at N=3**: land #71 tree-verify (+21-23% projected) clears by >100×; stark #78 GEMM fusion (~+2.6%) and denken #81 prompt-lookup are well above the floor. Sub-5% wins are now reliably detectable.
4. **wandb_logging.py bug fixed** (`group=group or agent`): `--wandb_group` was silently a no-op for ALL callers before this merge. All prior runs that used `--wandb_group` logged to the wrong W&B group.
5. lawine reassigned → **#82 (walltps-ab-runner)**: operationalize the protocol as a reusable `paired_tps_ab.py` runner for the whole team + re-baseline local frontier + retrospective re-screen of prior within-noise A/Bs.

## 2026-06-14 01:49 — PR #77: Drafter non-GEMM profile — map the real ~70% binding drafter cost ✅ MERGED (decisive FAIL-FAST negative: no contract-safe non-GEMM drafter lever; drafter axis fully harvested → land tree-verify #71)

- **Branch:** `denken/drafter-nongemm-profile` · **Student:** denken
- **Status:** MERGED as a research artifact (audit-only, zero served-file change → no BASELINE.md change; frontier bar UNCHANGED 481.53). Lands a reusable profiler `scripts/profiler/drafter_nongemm_profile.py` + `research/spec_cost_model/{drafter_nongemm_profile.json,report_drafter_nongemm_profile.md}`.
- **Hypothesis:** #75 showed ~70% of the drafter forward is NON-GEMM. Profile that non-GEMM mass per sub-op to find a contract-safe lever (the drafter's last remaining headroom before we commit to the tree-verify build).
- **Primary metric:** `drafter_nongemm_binding_subblock_pct_of_decode = 1.54` (W&B `q9p4vetv`). **Test:** `realistically_addressable_drafter_tps_headroom_pct = 0.0`.

| drafter non-GEMM sub-block (A10G, deployed M=1×K=7, conc=1) | cost | % decode step |
|---|--:|--:|
| **binding sub-block: `centroid_sampler_fused`** | **178 µs/step** | **1.54%** |
| attention | 61 µs/step | 0.53% (memory floor) |
| rest (fused glue + dispatch) | — | no hotspot |
| "gather only candidate set" optimization | already DEPLOYED (8192/262144 = 3.1%, 31× cheaper) | — |
| **realistically addressable headroom** | **~0%** | — |

**Conclusions:**
1. **Decisive FAIL-FAST negative.** There is no contract-safe non-GEMM drafter lever — the binding sub-block is 1.54% of the decode step, attention is at its memory floor, and the obvious win ("gather only the candidate set") is already deployed. Addressable headroom ≈ 0%.
2. **The drafter axis is now fully harvested** across #75 (forward roofline, int4-drafter refuted) + #77 (non-GEMM profile). No remaining drafter-side TPS lever survives the roofline.
3. denken's own verdict: *"Do not build a drafter non-GEMM optimization — land tree-verify (#71), the #1 lever."* Banks the third reusable profiler onto the advisor branch.
4. denken reassigned → **prompt-lookup / ngram hybrid drafter Step-0 viability gate** (orthogonal free-tokens lever; hedges the tree-verify timeline).

## 2026-06-14 01:38 — PR #34: Benchmark-matched reasoning corpus — break the 0.73 drafter plateau ✅ MERGED (decisive corpus positive: native 0.7792, +81% rel; BUT K=1 drafter not deployable vs MTP 3.844 — not a TPS win)

- **Branch:** `fern/bench-reasoning-corpus` · **Student:** fern
- **Status:** MERGED as a research artifact (corpus + EAGLE-3 training/eval pipeline; the deployed MTP serving stack is unchanged → no BASELINE.md change; frontier bar UNCHANGED 481.53). Lands `scripts/drafter/{gen_eagle3_corpus,train_eagle3,eval_eagle3}.py` + `research/eagle3_drafter/arch_notes.md`.
- **Hypothesis:** the 0.73 drafter-acceptance plateau is a CORPUS-distribution problem, not a capacity ceiling. Train EAGLE-3 on a benchmark-matched reasoning corpus (aime / gpqa / mmlu_pro) → break 0.73 on tf_acc and native accept/step.
- **Primary metric:** `native_accept_per_step_bench_holdout = 0.7792` (vs #25 ckpt 0.4315, **+81% rel**) — W&B `56ksyxgw` / `gua9x68j` / `cjhjnsff`. **Test:** `native_accept_per_step_bench_holdout_25ckpt = 0.4315`.

| quantity (EAGLE-3 drafter, 240-rec / 159k-tok bench holdout, feature_shift=1) | value |
|---|--:|
| **native accept/step (bench corpus ckpt)** | **0.7792** (vs #25 ckpt 0.4315, +81% rel) |
| teacher-forced acc (same holdout) | 0.7617 vs 0.4709 |
| per-source: aime / gpqa / mmlu_pro | 0.8426 / 0.8033 / 0.7006 |
| chain past step 1 (K=1 regime) | **collapses → ~1.78 tok/step** |
| deployed MTP K=7 E[T] (reference) | **3.844 tok/step** |

**Conclusions:**
1. **The corpus lever DECISIVELY breaks the 0.73 plateau** on both teacher-forced (0.7617) and native (0.7792) — benchmark-matched reasoning data is the right distribution. Strong, reproducible drafter-quality result.
2. **BUT this is NOT a TPS frontier win as-is:** the EAGLE-3 drafter is K=1-regime and its chain collapses past step 1 (~1.78 tok/step), **far below** the deployed MTP K=7 chain's E[T]=3.844. Step-0 acceptance is excellent; multi-step chaining is the gap.
3. **Residual ceiling = the K=1 training regime, not the corpus.** Merged to bank the proven corpus + training/eval pipeline.
4. fern reassigned → **#80 (multi-step / HASS drafter training)**: train the drafter on its own hidden states for steps 2..K to lift the chain past step 1 toward MTP's E[T], on the proven benchmark corpus.

## 2026-06-14 01:30 — PR #76: Calibrate deployed-chain acceptance to pin the tree-verify gain band ✅ MERGED (decisive positive: top-1=0.729, E[T]=3.844; M=32 +18.7%/≈508 TPS empirically anchored; wirbel→#79 ρ probe)

- **Branch:** `wirbel/acceptance-calibration` · **Student:** wirbel
- **Status:** MERGED as a research artifact (measurement-only, zero served-file change → no BASELINE.md TPS change; frontier bar UNCHANGED 481.53). Lands reusable harnesses: `scripts/profiler/accept_calibration.py`, `scripts/profiler/treeshape_measured_accept.py`; `research/accept_calibration/*.json`.
- **Hypothesis:** Pin the deployed chain's real per-rank acceptance to resolve the #49 vs #68 discrepancy (0.6792 vs E[accept]≈3.8-implied 0.775) and re-price #74's M=32/M=16 DP trees with measured acceptance. De-risk land #71 (tree-verify build).
- **Primary metric:** `deployed_chain_mean_tokens_per_step = 3.8441` (W&B `5m17r52s` / `zfzxl0np`, group `acceptance-calibration`).

| quantity (A10G, deployed MTP K=7, conc=1, 128 prompts × 512 tok) | value |
|---|--:|
| **top-1 acceptance (rank-1)** | **0.729** |
| conditional acceptance depth-1→7 | 0.729 → 0.847 (rising with depth) |
| **measured E[T] (tok/step)** | **3.844** (primary) / 3.849 (Prometheus, Δ0.005) |
| draft acceptance rate (E[T]−1)/7 | 0.406 |
| **M=32 tree re-price** | **+18.7%** (≈508 local TPS) vs +20.1% modeled (−1.4pp) |
| **M=16 tree re-price** | **+11.5%** vs +13.1% modeled (−1.6pp) |
| M=32 still dominates M=16? | Yes |
| Fail-fast triggered? | No — tree gain not marginal |

**Reconciliation (decisive):** #49's 0.6792 was an EAGLE-3 drafter scalar (wrong drafter — deployed MTP kenyan-duma has higher top-1). #68's back-solve top-1≈0.775 overstated because real acceptance profile RISES with depth (0.729→0.847); constant-p forced to hit E[T]=3.84 sits above the true top-1. **Authoritative: top-1=0.729, E[T]=3.844** (not a real discrepancy — two estimators applied to the same chain, plus one wrong-drafter scalar).

**Conclusions:**
1. **M=32 +18.7% / ≈508 local TPS empirically anchored.** #74 projection confirmed to −1.4pp; tree not marginal; M=32 dominates M=16.
2. **Dominant uncertainty shifted** from top-1 (resolved) to **ρ = rank-2+ drafter coverage** (P(target == drafter rank-2/3/4 | rank-1 missed)). Linear chain can't expose ρ; borrowed EAGLE-3 ρ=0.565 → credible band **+11…+25%, central +18.7%**.
3. wirbel reassigned → **#79 (rank-2+ drafter coverage probe)** — measures ρ locally, cross-validates byteshark's official-stack rank-2 conditional.
4. land #71: proceed with M=32 build. Expected official projection: ~481.53 × 1.187 ≈ **571 TPS** (>>500 target). Remaining risk = ρ; wirbel #79 + byteshark resolve it.

## 2026-06-14 00:47 — PR #75: Drafter-forward roofline — is the 15.5% block bandwidth-bound? ✅ MERGED (decisive negative: refutes int4-drafter-for-TPS; the drafter's #2-block headroom is non-GEMM, not weight bytes)

- **Branch:** `denken/drafter-forward-roofline` · **Student:** denken
- **Status:** MERGED as a research artifact — the **sibling roofline to #68** (verify-GEMM). Audit-only, zero served-file change → no BASELINE.md change. Frontier bar **UNCHANGED 481.53.** Lands a reusable profiler (`scripts/profiler/drafter_forward_roofline.py`) + the drafter decode-composition cost report.
- **Hypothesis:** stark #70 was building int4 drafter weights on an **unaudited premise** — that the K=7 MTP drafter forward is weight-bandwidth-bound at the deployed M=1×K=7. A Step-0 roofline (#68 method, FP16-ceiling) validates or refutes that premise *before* stark spends the build.
- **Primary metric:** `drafter_forward_pct_hbm_peak_at_M1K7 = 47.17%` (W&B `uknpbk94`, finished; primary verified exact).

| quantity (A10G, drafter bf16, deployed M=1×K=7) | value |
|---|--:|
| **`drafter_forward_pct_hbm_peak_at_M1K7`** | **47.2%** (7-pass GEMM chain, launch-free onegraph) |
| arithmetic intensity at M=1 | 1.0 FLOP/byte (ridge 86.8 → 86× below) → memory-bound *regime* |
| achieved compute at M=1 | 0.45% of FP16 peak (52.1 TFLOPS realizable ceiling) |
| most-repeated GEMVs (sliding-attn q/o, 6 of 19/pass) | **19% HBM** → **latency/launch-floored, not bandwidth-saturated** |
| 7-pass drafter GEMM chain (deployed graph) | **566 µs/step = 4.88% of the 11.6 ms decode step** |
| drafter forward total (#69 budget, the #2 decode block) | 1798–2100 µs = 15.5–18.1% |
| → **non-GEMM** drafter (centroid sampler + 262k masked-embed gather + SDPA + sampling) | **~69–73% of the drafter** (untouched by int4) |

- **int4-drafter-for-TPS ceiling (stark #70 cross-check):** hard ceiling (every drafter GEMM → 0 µs) = **+5.13%**; int4 bandwidth-scaling = +3.62% (optimistic); **realistic +1.5…+3%**. Premise-implied naive ("3.5× faster 15.5–18.1% block") = +12.5…+14.9% → **overstated ~3–5× vs ceiling, ~4–8× vs realistic.** The premise is right about the *regime* (AI≈1, memory-bound) but wrong that the block is a *saturated* bandwidth wall (47%, not 75–100%) — and int4 touches only 4.88% of decode.
- **Onegraph:** drafter runs **inside blake's `onegraph` (CUDA-graphed), launch-free** — it does NOT pay #68's ~55 µs/call eager floor (eager chain 2859 µs vs graph 566 µs; the 2.3 ms gap is already-harvested launch overhead, not free headroom). int4 working set 6.5 MB still > 6 MB A10G L2 → weights still spill every pass; int4 does not make them L2-resident.
- **Pass-count lever (feasibility only): INFEASIBLE with unchanged outputs.** MTP is autoregressive (pass *i* consumes pass *i−1*'s token) → no single wider GEMM yields the identical 7-token chain; L2-residency needs <6 MB (int4 6.5 MB still spills); K<7 changes accept behavior (fern #34's axis).
- **Conclusions / actions taken:**
  1. **stark #70 CLOSED** — int4-drafter-weights-for-TPS refuted (≤+3% realistic, +5.13% ceiling; not the double-digit win the framing implied). stark reassigned to a higher-value orthogonal BUILD lever (prompt-lookup/ngram free draft tokens).
  2. **Also informs open2-askeladd #57** (W8A8 int8 drafter) — same drafter-quant-for-TPS premise; flagged cross-board (saves their quota, byteshark-negatives ethos).
  3. **The real drafter lever is the ~70% non-GEMM**, not the weights — denken reassigned to a per-op decomposition of the non-GEMM (centroid sparse sampler / 262k masked-embed gather / SDPA / sampling) using his #75 reconstructed-module harness, to find the fattest reducible/fusable op. Secondary: drafter kernel-fusion (lift the 47% chain / 19% GEMVs off the launch floor) — contract-safe, larger than int4.
  4. **Verify-GEMM (53%, #68: free to widen to M≤32) remains the higher-value block** (land #71 tree-verify = the 500-path); the drafter is the #2 block but with little weight-byte headroom.

## 2026-06-14 00:40 — PR #74: TPS-optimal tree-shape under denken #68's measured M≤32 verify-cost curve ✅ MERGED (the concrete build target for land #71)

- **Branch:** `wirbel/tree-shape-cost-model` · **Student:** wirbel
- **Status:** MERGED as the **canonical TPS-optimal tree-shape verdict** — same artifact class as #68 (a decisive, MC-validated research deliverable; audit-only, zero served-file change → no BASELINE.md change). Frontier bar **UNCHANGED 481.53.** Converts #68's real cost curve + wirbel's own #49 acceptance model into a concrete build target for **land #71**.
- **Hypothesis:** #49's DP-optimal tree (+16% TPS) assumed a simple verify cost; re-solving the DP against #68's *measured* non-uniform V(M) curve (cheap tile-tops M=16/32, expensive M=24, hard M=33 cliff) yields the actual TPS-optimal (shape, M) — the exact topology land #71 should build.

| operating point (real #68 cost, g=0.532, measured p, geom) | E[T] | step mult vs M=8 | proj local TPS | vs deployed linear (428.37) |
|---|--:|--:|--:|--:|
| deployed **linear K=7 / M=8** (anchor) | 2.976 | 1.000 | **428.37** | — |
| linear own-optimum (M=16, saturated) | 3.111 | 1.034 | 433.1 | +1.1% |
| **DP tree M=16** (Marlin tile-1 top — Step-1 build) | **3.481** | **1.034** | **484.7** | **+13.1%** |
| **DP tree M=32** (tile-2 top — PRIMARY) | — | **1.098** | **514.32** | **+20.1%** |

- **Headline:** the TPS-optimal tree is the **M=32 DP tree → ~514 local TPS, +20.1%** over the deployed linear K=7 chain; cheaper secondary at the M=16 tile-1 top → **~485 TPS, +13.1%.** Primary metric `treeshape_opt_proj_tps_gain_real_costcurve = +0.2007` (M=32).
- **Three canonical takeaways:** (1) **The optimum did NOT shift from #49** — the deep-spine DP tree at M=32 survives the real-cost refinement (projection even ticks *up* +19.0%→+20.1%, because measured M=32 mult 1.098 < modeled 1.108). Reassurance, not a pivot. (2) **"Build to a tile-top, never mid-tile"** — M=16/M=32 sit at the cheap Marlin tile tops (9 µs/row marginal); **M=24 is strictly dominated.** (3) **Shape/budget separation:** verify cost depends only on node budget M, not tree shape (the GEMM processes all M rows regardless) → the tree designer optimizes acceptance freely under a hard M≤32 row budget.
- **Build targets handed to land #71** (advisor comment, 00:40:49Z): **(1) Step-1 = M=16 DP tree** `parent=[-1,0,0,0,1,1,2,4,4,5,6,7,11,12,13,14]` (16 nodes, depth 8, 4 rank-2+ branches) — build FIRST to validate measured acceptance + greedy identity on the real tree-verify path; **(2) Primary = M=32 DP tree** (32 nodes, depth 9, 9 rank-2+ branch points, max branch 4, bushy crown).
- **Validation:** brute-force n≤7 == DP; MC 400k max rel-err 0.11%; robust **+16.5–21.7%** across pricing / GEMM-share / rank-decay / base-acceptance variants. W&B `p1yyrwpr`. Local cost-model study, **no HF Job**, lossless by construction.
- **One open number (→ wirbel #76):** the projection brackets +18% (if rank-1=0.6792, #49) vs +20% (if top-1≈0.775, implied by deployed E[accept]≈3.8). These disagree materially → **wirbel reassigned to #76** to pin the deployed chain's real per-rank served acceptance, turning "+18–20% modeled" into one defensible number before land #71 spends any submission quota.
- **Artifacts:** `research/spec_cost_model/report_treeshape_real_cost.md`, `treeshape_real_cost_results.json`, `scripts/profiler/treeshape_real_cost.py`.

## 2026-06-14 00:15 — PR #68: Verify-GEMM M=8 roofline audit — is the 53% block free to widen? ✅ MERGED (GREEN — greenlights the 500-path)

- **Branch:** `denken/verify-gemm-m8-roofline` · **Student:** denken · merged to advisor branch (commit `f2ec624`).
- **Status:** MERGED as a **characterization keeper** (reusable roofline harness + cost curve; #49/#51-class positive verdict, not a baseline-beater → no BASELINE.md change). Frontier bar **UNCHANGED 481.53.** This is the audit the entire **tree-verify thread (land #71)** was gated on — verdict is decisive GREEN.
- **Hypothesis:** at the deployed M=8 verify, is the dominant 53.2% int4-Marlin verify-GEMM block (#30) compute/tile-bound (irreducible) or weight-bandwidth-bound (free headroom to widen M for multi-candidate/tree verify)?

| quantity (A10G, int4 W4A16 Marlin, M=8) | value |
|---|--:|
| achieved HBM bandwidth | **462 GB/s = 77.1% of 600 GB/s peak** → **BANDWIDTH-BOUND** |
| achieved compute | 13.0 TFLOP/s = **20.2% of FP16 peak** (measured 64.3 TFLOPS) |
| arithmetic intensity @ M=8 | **28 FLOP/byte vs ridge 107** (3.8× below) |
| widen M=8→16 (top-2 tree) | **+6.4% verify-GEMM** (+2.7% step), marginal **9 µs/row** |
| widen M=8→32 (top-4 tree) | **+18.4% verify-GEMM** (~+7.7% step), aggregate **~37 µs/extra row** |
| M=24 (avoid) | +16.9%, marginal **64 µs/row** (expensive) |
| **M=33 — HARD TILE CLIFF** | **+53.3%** (Marlin 16-row M-tile boundary; reproduces #51's M=33/49 cliffs) |

- **Verdict:** "**at M=8 the int4 W4A16 Marlin verify-GEMM is unambiguously WEIGHT-BANDWIDTH-BOUND, not compute/tile-bound. Free verification headroom EXISTS and is bounded by the Marlin M=33 tile cliff.**" ~80% of the verify-GEMM is pure weight-movement that only 8 rows consume → ~4× under-utilised per-weight-read amortization. **Free window M ∈ [8, 32]: up to 4× more candidate positions at ~37 µs/row, hard ceiling M=33.** Break-even for the downstream tree: an M=8→32 batch (+898 µs) is net-positive TPS if it adds **> ~0.43 accepted tokens/step** — a low bar for a width-2…4 tree at the drafter's *existing* depth (adds verify rows **without** adding sequential drafter forwards).
- **Premise correction (strengthens the conclusion):** Marlin W4A16 **dequantizes int4→FP16 on-chip** and runs FP16×FP16 tensor-core MACs (arXiv 2408.11743 §3); 4-bit is a *weight-storage* format that cuts HBM traffic 4×, never an int4 compute path. So the compute ceiling is the **FP16 peak (64.3 TFLOPS)**, not int4 (~280 TOPS) — using int4 would have *understated* utilisation 4× and falsely implied more headroom. The "completely free to widen" impression from earlier eager timing was a ~55 µs/call launch-overhead floor; launch-free CUDA-graph timing reveals the true ~37 µs/row.
- **Empirical complement (the shape of the headroom):** public **byteshark linear K=8 probe = VALID but SLOWER, 470.84 < 481.53** (verify M=9, deep inside the cheap M≤32 GEMM regime) — yet it *loses* TPS because **linear** K-widening adds a sequential drafter forward (drafter 15.5% of decode) at low marginal accept probability. **Takeaway: the GEMM headroom is real but linear chains can't spend it — it must go to multi-candidate/TREE verify (parallel candidates at fixed depth).** Exactly the lever this audit greenlights.
- **W&B (group `verify-gemm-m8-audit`):** `av8a5wh8` (launch-free CUDA-graph, primary) · `av98bjsw` (eager cross-check showing the launch floor). Local A10G only, **no HF Job** (read-only GEMM microbench, lossless by construction → PPL 2.3772 / greedy-identity 128/128 definitionally unchanged).
- **Artifacts (merged, now canonical team assets):** `scripts/profiler/verify_gemm_roofline.py`, `research/spec_cost_model/verify_gemm_roofline.json`, `research/spec_cost_model/report_verify_gemm_roofline.md`.
- **Follow-ups / propagation:** (1) **land #71** builds the tree-verify serving path sized to **M ≤ 32**, snapping total verify rows to ≤32 (never cross M=33) — handed the exact M-budget. (2) **wirbel → #74** re-solves the #49 Sequoia DP-optimal tree under this *measured* non-uniform V(M) curve (pack candidates into the cheap-marginal M=16/M=32, avoid M=24, hard M≤32) → exact build topology for land #71. (3) **denken → #75** drafter-forward roofline (the last unaudited decode block; validates/refutes stark #70's int4-drafter bandwidth-bound premise). Does NOT change drafter K or the AdaEDL/#54 dynamic-K lane (scope guard).

## 2026-06-14 00:15 — PR #69: Attention split-KV roofline audit — is the #2 block (19.6%) at the floor? ✗ CLOSED (NEGATIVE — attention is irreducible)

- **Branch:** `wirbel/splitkv-nseg-roofline` · **Student:** wirbel
- **Status:** CLOSED as the **third clean Step-0 systems negative** (with #65 CUDA-graph, #67 norm-fusion) — a keeper-in-the-record that sharpens the lever map. No code/served change → frontier bar **UNCHANGED 481.53.** Excellent fail-fast discipline.
- **Hypothesis:** the split-KV verify-attention is a custom (non-Inductor-fused) kernel ⇒ may carry hand-tunable headroom at the served M=8.

| quantity (deployed M=8, post-#43) | value |
|---|--:|
| attention % of GPU-busy | **7.6%** (was 19.6% pre-#43) → already the **#3 block, not #2** |
| attention µs/step | **605** (was 1836; #43 cut it 3.03×) |
| achieved BW vs peak | **20.0%** (96.6 GB/s vs 482 GB/s copy) — memory-**LATENCY**-bound, not BW-bound |
| occupancy @ n_seg=16 | **96 CTAs ≥ 80 SMs → saturated** (no occupancy bump available) |
| n_seg sweep {1…64} × ctx | **deployed n_seg=16 is exactly optimal at served-dominant shapes** (sliding ctx256 43.8% of cycles, full ctx512/1024 all 1.00×) |
| oracle best-vs-deployed ceiling | **+0.126% TPS** — and un-CUDA-graph-able (n_seg is a onegraph capture-shape constexpr) |
| free attention→0 ceiling (hypothetical) | only **+8.2% TPS** (de-prioritised) |

- **Verdict:** "**BW-bound? Occupancy YES, bandwidth NO — it's the irreducible conc=1 latency floor. Residual lossless headroom ≈ +0.13% TPS (oracle, un-CUDA-graph-able). No fix worth prototyping.**" At conc=1 each layer reads one sequence's KV (sliding 0.25–1 MB, full 2.2 MB) — far below the working set needed to hide HBM latency on 80 SMs → 20% of peak is the **floor, not slack** (BW *rises* monotonically with read size = the latency-bound signature). 80% peak only exists at large batch, which this single-stream submission never sees.
- **Two premise corrections banked:** (1) **Attention is already the #3 block at 7.6%, not #2 at 19.6%** — the 19.6% is the stale #30 *pre*-split-KV number; **#43 already harvested this block** (wirbel's own `r0ahjs45` re-profile). The PR chased a number #43 had already taken. (2) The served kernel is **100% stock vLLM-native Triton `unified_attention`** (3D split-KV/FlashDecoding) — not a custom submission kernel we own, and not Inductor-fused; the fa2sw FA2 router is **INERT** (0 FlashAttention kernels in the served trace; vLLM forces TRITON_ATTN for the heterogeneous sliding-256/full-512 head_dims).
- **W&B:** `rajcg6an` (group `attention-splitkv-audit`). Local A10G only, **no HF Job** (read-only op-microbench; served stack untouched → PPL 2.3772 / 128/128 definitionally unchanged).
- **Artifacts:** `research/profiling/splitkv_nseg/` (`nseg_sweep.py`, `aggregate.py`, `FINDING.md`, `breakdown.md`).
- **MAP UPDATE (load-bearing):** with **#65 (CUDA-graph), #67 (norm/elementwise), #69 (attention)** the decode **SYSTEMS layer is confirmed fully harvested.** Combined with **#68 (verify-GEMM bandwidth-bound, free to widen M≤32)**, the open frontier is now unambiguously **ALGORITHMIC** — verify **width** (→ land #71 tree-verify) and **acceptance/tokens-per-step**. With verify-GEMM (#68), attention (#69) and drafter (incoming #75/stark #70) roofline-mapped, all three big decode blocks are characterised.
- **Follow-ups:** **wirbel → #74** (he authored the #49 tree cost model → the right owner to find the TPS-optimal tree-shape under denken #68's real V(M) curve, feeding land #71). Flagged-not-implemented: **fa2sw dead-config cleanup** (the inert FA2 sliding router — pure simplification, no perf/PPL change, in the submission name); de-prioritised cross-layer KV read-coalescing (YOCO/CLA — would break the lossless gate).

## 2026-06-14 00:10 — PR #56: max_num_batched_tokens served A/B on the split-KV #1 stack ✗ CLOSED (parity characterization keeper — NOT a winner, NOT a regression)

- **Branch:** `lawine/maxbatchtok-served-ab` · **Student:** lawine
- **Status:** CLOSED as a parity/characterization keeper. No served-file change (research-only A/B harness + bugfix only). Frontier bar **UNCHANGED at 481.53**. Disposition: the knob is conclusively closed (parity + invalid-above-512); lawine's own "Suggested follow-ups: None on this knob."
- **Hypothesis:** sweeping `MAX_NUM_BATCHED_TOKENS` (512/2048/4096/8192) on the deployed `fa2sw_precache_kenyan` stack yields a decode-TPS gain and/or silences the #52 spec-decode launch warning.

| `max_num_batched_tokens` | steady TPS (n=14) | Δ vs control | PPL | completion | valid? |
|---|--:|--:|--:|---|---|
| **512 (control / deployed)** | **448.01** | — | **2.3767** | 128/128 | ✅ |
| 2048 | 445.92 | −0.47% | OOM | 128 decode, PPL crash | ❌ |
| 4096 | 453.40 | +1.20% | OOM | 128 decode, PPL crash | ❌ |
| 8192 | 449.56 | +0.35% | OOM | 128 decode, PPL crash | ❌ |

- **Analysis:** clean NEGATIVE (parity), with two extra teeth. (1) **No decode-TPS leverage** — at conc=1 / `max_num_seqs=1` each decode step verifies only M=8 tokens (far below any mbt), so the knob governs only prefill chunking; every inter-arm delta (≤+1.2%) is *inside* the control's own +4.4% run-to-run swing (429.04 vs 448.01 same-config). (2) **512 is the only PPL-passing value** — mbt≥2048 OOMs the `prompt_logprobs` log_softmax (+1.34 GiB) on the validity pass (decode completes 128/128, the gate crashes); footprint grows monotonically 20.92→21.02 GiB. Caveat: local A10G 22.06 GiB vs official a10g-small ~24 GiB, so the OOM might not reproduce officially — but there's no TPS upside regardless. (3) **#52 warning is benign AND structurally un-silenceable** — `vllm.py:1597` silences only at mbt≥8192, `scheduler.py:281` only at mbt≤4096; the regions never overlap, so some warning always fires; the only spec-decode-silencing value (8192) OOMs PPL. **Net: the deployed `MAX_NUM_BATCHED_TOKENS=512` is decode-optimal and the only gate-passing value — validated, no change.** Useful invariant banked: the **validity pass, not decode, is the memory-tight phase** at 0.90 util (matters for any future activation-growing change, e.g. tree-verify wider-M / land #71).
- **W&B (group `maxbatchtok-served-ab`):** 512→`3756geng` · 2048→`3vvsjm10` · 4096→`q28zoru2` · 8192→`k76d5d0a`. No HF job (local served A/B only).
- **Bug fix (kept on branch, cherry-pickable):** made the research-only `maxbatchtok_ab.py` harness's wandb-log + PPL pass non-fatal so a PPL OOM is captured as data (`engine_oom=true`) rather than discarding a completed arm. No served files touched.
- **Follow-ups:** none on this knob (closed). The frontier lever remains **(b) more accepted tokens per weight read** → tokens-per-step: **land #71 tree-verify serving path** (deploys wirbel #49's +16%), **denken #68 verify-GEMM roofline**, **lawine #72 noise-floor protocol** (needed to detect sub-5% wins). Queued idea (no idle seat): **ngram/prompt-lookup hybrid drafter** (training-free copy-span tokens-per-step).

---

## 2026-06-13 22:13 — PR #52: fa2sw split-KV — Issue-#46-approved one-shot HF launch ✓ MERGED ⭐ NEW PUBLIC #1 / NEW OFFICIAL FRONTIER (481.53 official TPS)

- **Branch:** `lawine/fa2sw-splitkv-official-launch` · **Student:** lawine
- **Status:** MERGED as the **new official frontier baseline.** First gated HF job to confirm a rung above the 126.378 AR floor on the spec-decode frontier → **the official bar all submissions must beat moves 126.378 → 481.53 TPS.** Human-approved launch (Issue #46, Morgan: "approved, lessgo!"); no submission-file changes (the PR is the launch record — served stack is the already-merged `submissions/fa2sw_precache_kenyan/` with #43 split-KV).
- **Hypothesis:** the locally-validated fa2sw split-KV stack (linear MTP K=7 + #43 3D split-KV, 428.37 local steady-state) reproduces on official a10g-small hardware above the prior public #1 (rock-ai 459.72), gated on the #50 fail-closed `official_gate` PASS@128 preflight.

| metric | value | gate |
|---|--:|---|
| **Official TPS (a10g-small)** | **481.53** | **NEW PUBLIC #1** (vs rock-ai 459.72, +4.74%; +13.4% over ~424.5 repro baseline) |
| PPL | 2.3772 | ≤ 2.42 ✓ |
| completed | 128/128 | ✓ |
| modalities | text+image+audio | all loaded ✓ |
| official_gate (preflight) | PASS@128 | split-KV patch engaged, zero 2D fallback ✓ |

- **Analysis:** clean reproduction — landed mid-projection (PR #43 projected 471–493). Pre-launch `official_gate=PASS@128` with the split-KV patch **engaged** (M=8 verify → 3D FlashDecoding every step, zero fallback, backend TRITON_ATTN). Greedy-identity DIVERGENT is an internal signal only (the official gate has no token-identity check, kanna #38) → spec decode is leaderboard-legal. **Standing risk UNCHANGED (the programme's #1):** the private re-run gate — kanna #44 probe predicts ~12.4% public→private on a pure-chat proxy (WOULD-FAIL >5%); the 481.53 is the **public** number; private stability is a separate open axis (kanna #55 calibrating on this exact frontier).
- **W&B:** `2x9fm2zx`, `fwo8rs05` (official launch; job `6a2dce05871c005b5352c0b9` COMPLETED, run prefix `results/senpai/fa2sw-precache-kenyan-20260613T213911Z`, `ppl_summary.json` 61,797 tokens). Leaderboard row pending organizer re-sync.
- **Follow-ups:** (a) report the new #1 to Issue #46 (done); (b) `max_num_batched_tokens` warning A/B — separate PR, touches the timed path; (c) #50 audio functional-probe polish (local tooling); (d) the open frontier lever stays the **private-stable acceptance** axis (kanna #55) + the verify-GEMM/drafter-forward decode blocks (ubel verify-GEMM, denken #54 entropy-K, wirbel #53 reprofile).

---

## 2026-06-13 21:55 — PR #51: accepthist dynamic-K on post-#43 split-KV cost curve ✓ MERGED (characterization + bugfix keeper — decisive negative, official bar UNCHANGED)

- **Branch:** `denken/accepthist-dynamic-k` · **Student:** denken
- **Status:** MERGED as a characterization + bugfix keeper. Official TPS bar **UNCHANGED at 126.378** (primary `projected_dynamic_k_tps_costmodel_post43_ctx512`=343.1 is a cost-model projection, **+0.12% vs static K=11**=342.7 = noise; not a served number, not comparable to the 428 served baseline).
- **Hypothesis:** dynamic draft length via acceptance history (`accepthist`) beats static K\*; #43 split-KV flattened cost(K) so argmax K\* should shift up; the public top-3 VALID (459) all use accepthist. **Premise corrected (wirbel #49, propagated to #51):** the deployed stack is **LINEAR MTP K=7 (M=8 verify), not an M=45 tree** → K varies on the linear chain; tree cost-model (540) is not the baseline.

| post-#43 ctx512 policy | TPS | vs K=11 | mean_K (sd) |
|---|--:|--:|---|
| static K=11 (= best static) | 342.7 | — | 11 (0) |
| **clairvoyant ORACLE** | 400.6 | **+16.9%** | — |
| best AIMD | 300.5 | −12.3% | 6.6 (3.6) |
| best window-mean linear | 328.5 | −4.1% | 10.1 (3.5) |
| **best realizable (LUT)** | **343.1** | **+0.12%** | captures **0.7%** of oracle |

- **Analysis:** decisive NEGATIVE on the headline. Two premises fail under measurement: (1) **#43 does NOT push K\* up — stays 11 on every curve/ctx** because the operating point is pinned by **Marlin int4 GEMM tile cliffs (M=33 +2.0ms, M=49 +2.9ms)**, and split-KV only accelerates *attention*, leaving the cliffs (hence argmax) put; (2) **acceptance history is too weak a predictor** (window-mean→next r≈0.32; lag-1 autocorr +0.16) → realizable control captures **<8%** of the real +16.1% oracle ceiling → net ≈0. **Split-KV *shrinks* the dynamic-K headroom** (oracle 25.2%→16.1%): flattening attention makes the unchanged GEMM staircase relatively more dominant — opposite of the hypothesis. **Reconciliation:** static optimum drops 11→**≈7** at the real e_accept≈3.82 → **the deployed linear K=7 is already near-optimal statically** (no static re-tune win either). **Keepers:** the `--sim-K` argmax-default fix (closes the PR#41/BASELINE.md:90 residual — every run now prints its `ARGMAX OPERATING POINT`); the re-grounded post-#43 cost curves (**#43 helps *more* at long ctx: verify −2.6%@256 → −7.1%@1024**); tooling `accepthist_controller.py` + `spec_cost_model.py --splitkv-patch` (redirect counter `total_redirected=106260` proves the patch fired) + `compare_splitkv_curves.py`. Tooling-only diff — no served-submission change. PPL 2.377 preserved by construction (greedy-exact; valid per #38).
- **W&B:** `wfi3jtkq` (sim; `splitkv_ctx512_static11_tps`=342.700, `splitkv_ctx512_oracle_gain_vs11_pct`=16.901, `realizable_frac_of_oracle`=0.007 — all confirmed), `6o8xaofq` (cost curve), group `accepthist-dynamic-k`. CPU sim + GPU cost curve (~21.6 GB A10G).
- **Follow-ups:** (a) **drafter-ENTROPY dynamic-K (AdaEDL, denken's suggestion 1) → denken #54** — entropy at draft time is a strictly stronger predictor than acceptance history; the *correct* read of the public top-3. (b) split-KV **net-negative at M=8/short-ctx** (+15.5%@ctx256) → **context-gate** the redirect (NOT M≥33) → routed to **wirbel #53**. (c) spine-E→DP tightening of `tree_acceptance_model.py` now **unblocked** (#51 landed) → queued to wirbel, rebased on #51.

---

## 2026-06-13 21:42 — PR #48: Token-frequency logit bias on the drafter ✓ MERGED (characterization keeper — decisive negative, official bar UNCHANGED)

- **Branch:** `kanna/token-freq-logit-bias` · **Student:** kanna
- **Status:** MERGED as a characterization keeper. Official TPS bar **UNCHANGED at 126.378** (decisive negative; primary `tps`=463.49 is the best biased arm, *below* the in-screen bias=0 baseline 471.35).
- **Hypothesis:** a static unigram logit bias on the drafter (boost top-K frequent output tokens) raises drafter acceptance without touching the verifier → +1–3% acceptance → +5–15 TPS, greedy-exact. **Forced deviations (both more favorable to the claim):** no `train.py --local-only --env` → reused the #44 LocalServer + `sglang.bench_serving` harness (fresh server/arm, one changed var); drafter is the centroid-sparse MTP head (not dense [B,262144]) → sparse-candidate bias table + drafter-only re-rank, bias=0 bypasses the hook (byte-identical to leaderboard, stays on the fused kernel).

| bias (K=500, n=32) | E_accept | ΔE_acc | TPS | ΔTPS% | per-step lat |
|---|--:|--:|--:|--:|--:|
| **0.0** (fused, =leaderboard) | 3.95587 | — | **471.35** | — | **8.29 ms** |
| 0.5 (grid optimum) | 3.97793 | **+0.56%** | 463.49 | **−1.67%** | 8.48 ms |
| 1.0 | 3.95160 | −0.11% | 461.15 | −2.16% | 8.47 ms |
| 2.0 | 3.87126 | −2.14% | 451.94 | −4.12% | 8.47 ms |

- **Analysis:** decisive NEGATIVE for TPS. TPS ≈ E_accept / latency moves in opposite directions: acceptance best-case +0.56% (b=0.5; *reverses* at higher bias — the FT'd MTP head already encodes the unigram marginal, so an external prior pulls it off the verifier's conditional argmax, consistent with #25's plateau ~0.73), while leaving the fused Triton sparse-argmax kernel costs a constant **+2.2%/step** (bias-independent = implementation cost), ~4× the gain. Full (K×bias) grid bounded: optimum K=500/b=0.5 = +0.56%. Even a zero-cost *fused* version ceilings at **~474 TPS (+2.6)** → "don't pursue." PPL 2.3767 unchanged by construction. Strategic read (with #49): cheap inference-time tricks are exhausted; the real acceptance lever is drafter DATA quality (land #9 / fern #34), not re-ranking.
- **W&B:** `96pn3c43` / `rrp0xc6e` (K=500 ×2, bit-identical E_accept) / `rggrg6r6` (K=100) / `l32wjlig` (K=1000). Ships `scripts/validity/drafter_bias_screen.py` (reusable drafter-tweak A/B harness) + `build_freq_bias_tokens.py`.
- **Cleanup queued → kanna:** relocate/inert the bias hook out of the about-to-launch frontier submission `fa2sw_precache_kenyan/sitecustomize.py` (Step 0 of kanna's next PR). **Reassigned → kanna:** private-gap calibration (#44 follow-up) — quantify the split-KV stack's private-re-run risk before the launch lands.

---

## 2026-06-13 21:32 — PR #49: Sequoia DP-optimal draft tree (cost-model study) ✓ MERGED (characterization keeper, official bar UNCHANGED)

- **Branch:** `wirbel/sequoia-dp-tree` · **Student:** wirbel
- **Status:** MERGED as a characterization keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric `dp_vs_linear_tps_gain_own_opt_costmodel`=1.1677 is a cost-model ratio, not throughput; the lane has no servable path).
- **Hypothesis:** a Sequoia (arXiv 2402.12374) DP-optimal draft tree beats the fixed/balanced tree by +3–15% E[T] on our measured acceptance, composing with the merged split-KV verify (#43). **Premise corrected by wirbel:** the deployed `fa2sw_precache_kenyan` drafter is **linear MTP K=7 (M=8 verify), not a width-4 tree**; vLLM 0.22 has no tree-attention verify path; tree-causal mask is a merged 0 ms dead-end (#33); the PR's `--local-only/--profile-tree-acceptance/--sequoia-tree` flags don't exist → pivoted to the CPU cost-model form the Notes anticipated.

| topology (matched budget) | E[T] @ M=8 | E[T] @ M=45 | max E[T] gain | TPS-opt budget n\* | TPS @ n\* (cm scale) |
|---|--:|--:|--:|--:|--:|
| linear (deployed family) | 2.976 | 3.117 | — | 16 | 235.7 |
| balanced-W4 (prior model) | 2.430 | 3.178 | DP/bal **1.433** | 31 | 216.7 |
| **Sequoia DP** | **3.019** | **4.132** | DP/lin **1.341** | **32** (M=33 Marlin cliff) | **275.2** |

- **Analysis:** DP tree is genuinely the better topology on our distribution (+43% E[T] vs balanced-W4, +16% TPS vs linear, decay-robust 13–17%; brute-force-validated n≤7, 200k-MC `F==E[committed]`). **But deployable gain = 0** — no tree-verify path exists in vLLM 0.22 and #33 predicts ~0-saving on the dense path. The PR's ≥432-local-TPS target is unmeetable by this route. **Lane closed analytically** (like the tree-mask). **Secondary (load-bearing):** the salvage-spine E in `tree_acceptance_model.py` (#26) is an **upper bound** — it scores 0.86-rate compounding to depth K with only K·W+1 nodes (true 0.86-compounding needs ~W^K branching). Over-count **+45% at M=45** (5.99 → achievable 4.13 → ~248 TPS, *below* the linear frontier) ⇒ **strengthens "ship linear; trees don't reach 500"** (#33/#37). wirbel did NOT auto-edit #26 (flagged + offered a 1-line tightening).
- **W&B:** `bvbg81v4` (group `sequoia-dp-tree`; CPU-only, <0.2 GB, ~30 s, no GPU/vLLM/HF-Job). Ships `scripts/profiler/sequoia_dp_tree.py` + `research/spec_cost_model/{sequoia_dp_results.json,report_sequoia_dp.md}`.
- **Follow-ups:** (a) **tree-ceiling tightening QUEUED** — replace salvage-spine E with achievable path-product DP in `tree_acceptance_model.py`, held until denken #51 lands (concurrent-edit on the same tool). (b) premise correction (linear MTP, not M=45 tree) **propagated to denken #51**. (c) wirbel → next slot (post-split-KV decode re-profile).

---

## 2026-06-13 21:22 — PR #50: official_gate wired into HF-launch preflight (fail-closed) ✓ MERGED (launch-safety infra keeper, official bar UNCHANGED)

- **Branch:** `lawine/official-gate-hf-launch-wire` · **Student:** lawine
- **Status:** MERGED as a launch-safety infra keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric `official_gate_wired=1`, not throughput).
- **Hypothesis:** the #45 `official_gate` verdict (PPL ≤ 2.42 AND completed == 128 AND all_modalities_loaded) should be the **fail-closed interlock** on the HF-launch path, so a quota-spending submission can never launch on a FAIL/INCOMPLETE gate, and an 8-prompt smoke can never authorize a 128-prompt run. This is the safety gate for the Issue #46 split-KV launch.

| check | behavior | verdict |
|---|---|---|
| gate FAIL | blocks HF launch | fail-closed ✓ |
| gate INCOMPLETE | blocks HF launch | fail-closed ✓ |
| 8-prompt smoke → 128-run | refused (n_prompts mismatch) | partial cannot certify full ✓ |
| image+text / video | functional probe (served) | loaded ✓ |
| audio | presence + non-zero fallback (no `vllm[audio]`/`av` locally) | decision (A) ratified ✓ |
| fa2sw smoke (8 prompts) | PPL 2.3767 bit-identical to #45 | no serve-path change ✓ |

- **Analysis:** closes the launch-safety lane opened by #45. The gate now **refuses to certify a full run from a partial sample** (carries `n_prompts`), so no quota is spent on an unproven 128-run. Audio honesty decision **(A)** ratified: presence+non-zero is correct policy — a functional-mandatory audio check would mislabel a *local-tooling* gap (`vllm[audio]`/`av` unavailable) as a *submission* defect. `make_probe_inputs.py` + `probe_inputs/{probe_audio.wav,probe_video.mp4}` staged for future functional audio. 51/51 tests (+launch-block truth table, partial-sample refusal, video probe). This is the interlock for the #46-approved one-shot split-KV launch.
- **W&B:** `bi3tqtv3` (local infra; nothing trained).
- **Follow-up → lawine #52:** run full 128-prompt `official_gate` validation on `fa2sw_precache_kenyan`, then execute the (Issue #46 human-approved) one-shot HF launch of the split-KV submission — gated on this PR's PASS verdict.

---

## 2026-06-13 20:09 — PR #23: int4 spec-verify greedy flip-rate probe ✓ MERGED (characterization keeper, official bar UNCHANGED)

- **Branch:** `stark/linchpin-fp32-accum-flip-probe` · **Student:** stark
- **Status:** MERGED as a characterization keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric is `flip_rate_per_token`, not throughput).
- **Hypothesis:** the int4-Marlin M=K+1 batched-verify vs M=1 greedy divergence is caused by batch-dependent fp16/bf16 reduction order; cheap fixes — (a) fp32 logit accumulation, (c) deterministic reduction — might zero the per-token argmax flips without a full batch-invariant kernel rewrite.

| config | flip_rate/tok (M=2..8) | latency overhead | verdict |
|---|--:|--:|---|
| baseline | 0.00521 (3/576) | 0% | — |
| fp32-logit | 0.00174 (1/576) | **+0.2%** | reshuffle, not a fix |
| deterministic | 0.00521 (3/576) | **+14.0%** | proven no-op |
| fp32+det | 0.00174 (1/576) | +14.7% | no |
| cross-process M=1 noise floor | **0/576** | — | flips are genuine batch effect |

- **Analysis:** decisive NEGATIVE — no config reaches flip_rate=0. The **7:268 existence proof** (faithful fp32 logits disagree M=1 vs M≥2) localizes the irreducible source to the **decoder Marlin int4 GEMM** (the hidden state feeding lm_head is batch-variant), NOT the logit-accumulation step — answering the hypothesis split. Two keepers: deterministic mode is strictly bad (no-op + 14%), and the flip is **binary M=1-vs-M≥2, flat in K** (longer drafts no worse for greedy-identity). Per #38 the official gate has no token-identity check, so this is most valuable as a **run-to-run reproducibility** diagnostic for the private re-run gate. Ships `scripts/profiler/verify_greedy_flip_probe.py` as a drop-in batch-invariance validator.
- **W&B:** `zd121euo` (group `verify-greedy-flip-probe`; flip rates verified to 7 sig figs).
- **Follow-up → stark next:** lane pivot (linchpin closed; greedy-identity is not the leaderboard gate) — see CURRENT_RESEARCH_STATE for the new assignment.

---

## 2026-06-13 20:08 — PR #44: Local private-stability probe (public→private TPS-gap predictor) ✓ MERGED (validity keeper, official bar UNCHANGED)

- **Branch:** `kanna/local-private-gap-probe` · **Student:** kanna
- **Status:** MERGED as a validity keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric is `public_to_private_gap_pct`).
- **Hypothesis:** the binding constraint above ~286 TPS is the private-set re-run (honest drafter stacks lose 4–9% TPS and die on the 5% repro rule). We can **predict the public→private TPS gap locally**, pre-submission, by measuring single-stream TPS + drafter acceptance on a distribution-shifted private-proxy set vs the 128 public prompts.

| scenario | precache | bench set | TPS | E_accept | PPL | completed |
|---|---|---|--:|--:|--:|---|
| leaderboard | public | public | **423.63** | 4.061 | 2.377 | 128/128 |
| public_cold | off | public | 418.37 | 4.089 | — | 128/128 |
| private_rerun | off | private | **370.96** | 3.565 | 2.377 | 128/128 |

- **Analysis:** reproduces the published VALID frontier (423.63 vs kenyan-duma 421.12; PPL 2.377 exact) ⇒ the measured ratio is trustworthy. Headline **public→private gap = 12.43%** ⇒ WOULD-FAIL (>5% → INVALID). Decomposition: distribution gap **11.33%** (drafter-acceptance collapse on chat, E_accept 4.06→3.57) + precache **1.24%**; acceptance ratio (0.872) fully accounts for TPS ratio (0.887) ⇒ the gap **is the drafter on chat**. Honest caveat: pure-ShareGPT proxy is likely harder than the real private set, so 12.4% is an upper-ish *pessimistic* early-warning (safe direction; no false-negative — firfir-cast known-7.2%-invalid also reads >5%). Ships `scripts/validity/private_gap_probe.py` + `build_private_proxy.py`.
- **W&B:** `jgxdnmwz` (values match exactly; group tag `private-gap-probe`, artifact `private_gap_report`).
- **Follow-up → kanna next:** calibrate the proxy against firfir-cast's known 7.2% (→ quantitative predictor) + rank the VALID frontier stacks by private-re-run risk; feeds the official frontier-submission go/no-go.

---

## 2026-06-13 19:20 — PR #42: `--spec-off` one-flag contract + validator N-mismatch legibility ✓ MERGED (infra keeper, official bar UNCHANGED)

- **Branch:** `lawine/specoff-contract` · **Student:** lawine
- **Status:** MERGED as a validity-infra keeper. Official TPS bar **UNCHANGED at 126.378** (`primary_metric=1` is a boolean "the flag works", not a throughput).
- **Hypothesis:** PR #40 exposed a footgun — `--spec-off` was a silent no-op for any spec stack whose `serve.py` ignores `SENPAI_REFERENCE_MODE`, so a "spec-off reference" was secretly captured with the drafter still on. Fix at the root: teach spec stacks to clear `SPECULATIVE_CONFIG` under the reference-mode env.

| deliverable | result | verified |
|---|---|---|
| `specoff_flag_works_for_mtp_drafter` | **1** | on-GPU serve: `speculative_config=None`, `reference_kind=served_spec_off` |
| spec stacks fixed | **3/3** (fa2sw, lf29cap444, int4_mtp_batchinv) | argv-intercept proof |
| leaderboard serve path untouched | **provably** (env falsy → helpers no-op → drafter config verbatim) | unit tests + argv proof |
| `n_mismatch_warning_added` | **1** (`reference_n_mismatch` + actionable warning) | — |
| tests | **14/14** (+6 new) | CPU-only |

- **Analysis:** retires the fragile per-submission `--ref-env SPECULATIVE_CONFIG=` workaround to a fallback; `--spec-off` is now the canonical one-flag path for every spec stack's pre-launch greedy reference. Two good judgment calls banked: (1) caught that `int4_g128_lmhead` is **pure-AR, not spec** (my assignment mislabeled it) → applied the fix to the real third spec stack `int4_mtp_batchinv` (token-count knob → `num_speculative_tokens=0`); (2) used a **truthy** env check matching `paths.REFERENCE_MODE_ENV="1"` rather than the literal `=="reference"` in my pseudocode (which would have been a silent no-op).
- **W&B:** none (local infra; nothing trained).
- **Follow-up → lawine #45:** local **official-gate preflight** (modalities-load check + consolidated PPL+completion+modalities verdict, separated from the internal greedy bar), bundling the canonical fa2sw-reference `--spec-off` regen.

---

## 2026-06-13 19:57 — PR #41: Eliminate scatter floor in `compute_logits` ✓ MERGED (characterization + deployable-infra keeper, official bar UNCHANGED)

- **Branch:** `denken/scatter-floor-elim` · **Student:** denken
- **Status:** MERGED at `6bfa448` after a clean Step-4 W&B reconciliation. Official TPS bar **UNCHANGED at 126.378** — the 538–546 figures are LOCAL cost-model ceilings at the K\*=11/M=45 operating point, not HF-validated throughput.
- **Hypothesis:** the `lmhead12k` plugin scatters 12k partial logits to a full [M,262144] −inf tensor before argmax (0.348 ms @ M=45). If the greedy-gate guarantee holds, `kept_ids[argmax(partial)]` is identical in one step → ~538→546 TPS local ceiling.
- **Reconciliation (the first-submission mismatch, now fixed):** I sent the first submission back because its Step-4 table (538/540/544) sat ~60 TPS above the cited runs (which logged K=6→480/477, `>500=False`) and the 538.15 control was absent. denken correctly root-caused it as a **logging bug in `tree_acceptance_model.py`**: it wrote `verdict_tps_ceiling_tree_at_full_scale`/`tps_tree_meas_p0_780` at the fixed `--sim-K` headline (default 6 → M=25), **not** the argmax K\*=11/M=45 operating point. PR #37 had surfaced K\* via a `kstar_p078_W4_tps_withdrafter` field that was never in the committed script. denken restored that field **additively** and re-ran all curves at `--sim-K 11`.

| deliverable | result | independent W&B verification (re-run, this cycle) |
|---|--:|---|
| Step 1 scatter-equivalence (primary) | `equiv_rate=1.0` | `gy05konp`: 1.0 (249,858/249,858) — **universal**, ascending `kept_ids` |
| Step 3 microbench @ M=45 | scatter 0.348 / persistent 0.299 ms | `wa72elyq`: 0.348 / 0.299 |
| Step 4 scatter control (PR #37 repro) | **538.15** | `x0gjax5p`: 538.1452 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |
| Step 4 persistent buffer (**deployable, +1.95**) | **540.10** | `m316ma9u`: 540.1009 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |
| Step 4 scatter-free remap | **544.22** | `g9h5rqv9`: 544.2240 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |
| Step 4 analytic gemm-floor | **545.82** | `z2k86aiu`: 545.8159 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |

- **Analysis:** two durable wins. (1) **Characterization:** the scatter is **unconditionally** redundant — ascending `kept_ids` ⟹ `argmax(scatter(partial)) ≡ kept_ids[argmax(partial)]` for *all* inputs, so it generalizes to the private set (no acceptance dependence). (2) **Deployable:** a **bit-identical persistent −inf buffer** in the `lmhead12k` plugin (26/26 `check_scatter_buffer_identity.py`) that removes the 0.348 ms per-step scatter alloc for a clean **+1.95 TPS** at the operating point (`m316ma9u` 540.10 vs `x0gjax5p` 538.15 control). The additive `kstar_p078_*` logging fix to `tree_acceptance_model.py` also makes every future cost-model run report its argmax operating point, not just the `--sim-K` headline — closes the exact reporting hole that caused the first-submission confusion.
- **W&B:** `gy05konp`, `wa72elyq`, `x0gjax5p`, `m316ma9u`, `g9h5rqv9`, `z2k86aiu` (all local cost-model/microbench; nothing trained).
- **Follow-up → denken next:** dynamic-K (`accepthist`) cost-model projection on top of the now-correct static K\*=11 logging + `--sim-K` argmax-default cleanup so the headline field defaults to the operating point.

---

## 2026-06-13 18:58 — PR #9: Wide-distribution KL-distilled drafter for private-stable acceptance — REQUEST-CHANGES (negative result + key methodological finding)

- **Branch:** `land/wide-drafter-distill` · **Student:** land
- **Status:** NOT MERGED (native regressed). Request-changes → rebase (`heldout.jsonl` conflict) + pivot to HASS serve-faithful objective. **High-value negative result.**
- **Hypothesis:** Above ~286 TPS the binding constraint is drafter acceptance, and the binding *risk* is the private-set re-run (drafters fit to the 128 public prompts lose 4–9% TPS and die on the 5% repro rule). A drafter KL-distilled on a wide, distribution-matched corpus should lift acceptance AND make it private-stable.

### Results (W&B run `land-freerun-v1b-171224`, project gemma-challenge-senpai, group wide-drafter-freerun)

| metric | stock | v0 (teacher-forced) | v1b (free-running) | Δ v1b vs stock |
|---|--:|--:|--:|--:|
| offline tf gate (accepted tok/step, K=7) | 3.455 | 3.811 (+10%) | **4.004** | **+15.9%** |
| **native accept/step (HF assisted-gen)** | **3.553** | 3.388 (−5%) | **3.341** | **−6.0%** |
| greedy identity (bf16 harness artifact) | 14/24 | — | 13/24 | — |
| peak mem train / eval-load | — | — | 17.4 / ~16 GB | A10G 23 GB fits |

- Full budget: 1030 steps, 220,746 positions, 3.4 epochs, 82 min (whole cap), LR cosine-decay-by-time, free_run_frac 0.895, diverge_frac 0.285.

### Analysis / conclusion

- **Problem #1 (v1a native collapse to 1.49) FIXED** by greedy-trajectory corpus + rejection-aware break (v1b native healthy 3.34, diverge_frac 0.285).
- **Problem #2 (the real one):** tf and native are **anti-correlated** under our training. Two independent schedules (v0 tf, v1b free-run) move tf +10/+16% while native lands at ~3.34–3.39. Signature of optimizing a divergent proxy, and **rules out exposure bias** (free-run directly targets it).
- **Mechanism (evidence-backed):** our objective + tf proxy condition the draft's step-0 hidden on the target's ground-truth hidden (fresh target prefill per position). HF native assisted-generation does NOT — the assistant runs its own forward over accumulated KV across verify rounds. Fine-tuning the draft to excel on the target's *true* hidden drifts it off the joint optimum the serving path feeds it; the un-fine-tuned stock draft sits ON that optimum (3.553).
- **Programme conclusion:** the offline tf gate (incl. `offline_acceptance.py`) is NOT a faithful proxy for native acceptance for this EAGLE drafter. Drafter work must be gated on native (or an interface-faithful objective). Propagated to fern #34 (native cross-check requested) and CURRENT_RESEARCH_STATE.
- **Next:** HASS-style serve-faithful training (feed the draft its own running hidden over accumulated KV), gate/select on `heldout_native_accept_per_step`. land sent back to implement on the same PR.

---

## 2026-06-13 — PR #39: fa2sw attention deep-profile ✓ MERGED — Triton verify occupancy-bound, 3D split-KV lever identified

- **Branch:** `wirbel/fa2sw-attn-profile` · **Student:** wirbel
- **Status:** MERGED — **high-value lever discovery.** LOCAL A10G op-microbench; no W&B (wandb_run_ids:[]). Rewrites the #30 lever map for verify attention.
- **Hypothesis:** fa2sw sliding-window attention (19.6% of decode cycle from #30) might be near-optimal or might have exploitable inefficiency (KV layout, SWA masking, bandwidth ratio vs theoretical minimum).

### Results

| metric | value | verdict |
|---|--:|---|
| **`fa2sw_bandwidth_efficiency_fraction`** | **0.0473** (4.7%) | 21× below 80% near-optimal threshold ✓ |
| **`verdict_attn_reduction_worth_pursuing`** | **1** | YES — implement 3D split-KV |
| measured split-KV speedup (M=1, identical work) | **4.14×** (sliding 4.36×, full 3.91×) | direct measurement |
| reachable attention saving | 50% (conservative 2×) … 82% (3D BW) | |
| TPS projection @ 50% saving | **~471** | crosses 440, 460 |
| TPS projection @ 82% saving | **~505** | crosses 460, 500 |
| `kernel_unified_attention` share of attention | 98.1% | Triton, NOT fa2sw FA2 |
| device time M=7→45 | ~53 µs flat | occupancy/launch-bound, not compute |
| KV bandwidth floor | 41.84 MB/cycle, 0.087 ms | served = 1.836 ms (21× above) |

### Key findings

1. **Premise refuted: the fa2sw FA2 path is inert.** vLLM forces `TRITON_ATTN` for heterogeneous head dims (sliding 256, full 512); FA2 caps at head_dim 256. The 19.6% is 98.1% Triton `kernel_unified_attention`. The PR #30 naming "fa2sw kernel" was wrong at the kernel level.

2. **Root cause: M=8 verify falls on 2D Triton path (occupancy-bound).** The `unified_attention` gates 3D split-KV (FlashDecoding) OFF for `max_seqlen_q > 1`. The spec-verify runs M=K+1=8 query rows → always lands on 2D (~6 CTAs / 80 SMs). The M=1 drafter uses 3D and runs 4.14× faster on identical work. Device time is FLAT M=7→45 → confirmed occupancy/launch bound.

3. **4.14× is a direct measurement.** 2D vs 3D at M=1, identical bytes/softmax: sliding 4.36×, full 3.91×. The 3D kernel EXISTS in vLLM; only the dispatch guard needs patching.

4. **The served Triton kernel is already optimal for M=1** (12.2 µs vs FA2 paged 58.2 µs vs SDPA 97.9 µs). The problem is purely the M>1 dispatch guard.

5. **Fix is greedy-exact** (split-KV is bit-identical attention). Zero gate risk. Orthogonal to spec-decode validity question.

6. **Implementation path:** patch `max_seqlen_q > 1` guard in `vllm/v1/attention/ops/triton_unified_attention.py` + extend per-segment softmax reduction to multiple query rows. ~90% already in vLLM.

7. **Methodology correction:** physical KV-load byte model (what FlashAttention streams) is the correct BW model, NOT `window×seq×heads` (double-counts attention matrix as bytes). Noted for future profiling.

### Conclusions

This is the single highest-leverage greedy-safe lever in the programme. Unlike spec-decode velocity (gated on batch-invariance / served-gate), the 3D split-KV fix is valid on the EXISTING honest frontier (already leaderboard-valid at ~424.5 TPS) and projects ~471–505 TPS. wirbel reassigned to implement the fix.

## 2026-06-13 — PR #40: Greedy-ref infra: 128-prompt fa2sw reference + bare-tag assertion ✓ MERGED

- **Branch:** `lawine/greedy-ref-128prompt` · **Student:** lawine
- **Status:** MERGED — **validity-infrastructure closure.** LOCAL INFRA ONLY; no HF job, no submission. Delivers the two follow-up items from PR #32; unblocks kanna #38's full 128-prompt served-gate audit.
- **Hypothesis:** PR #32 fixed reference keying but only had a 32-prompt reference. kanna #38's served-gate audit needs the full 128-prompt served spec-off reference and the bare-tag collision class needs a runtime assertion to prevent regression.

### Results

| metric | value | verdict |
|---|--:|---|
| `fa2sw_reference_128prompt_complete` | **128** | full reference ✓ |
| `bare_tag_assertion_added` | **1** | assertion hardened ✓ |
| `reference_self_consistent` | **1** | deterministic at batch=1 ✓ |
| Tests (CPU-only) | **8/8 pass** | 6 prior + 2 new ✓ |
| Wall-clock (cold-start + 128 decodes) | **514.75s** (~14 min) | within budget ✓ |
| Reference key format | `…/submissions/fa2sw_precache_kenyan::google/gemma-4-E4B-it` | `<dir>::<model_id>` ✓ |

### Analysis & conclusions

1. **128-prompt reference is the primary deliverable for kanna #38.** `validate_submission --submission fa2sw_precache_kenyan --num-prompts 128` now auto-resolves without manual path threading. The reference at `research/greedy_reference/workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it/` supersedes #32's 32-prompt version.

2. **Justified deviation on drafter disable: critical institutional knowledge.** `fa2sw_precache_kenyan` uses `SPECULATIVE_CONFIG={method:mtp,...}` and `serve.py` does NOT honor `SENPAI_REFERENCE_MODE`. The `--spec-off` flag would have been a silent no-op, producing an invalid reference with speculation ON. Correct method: `--ref-env SPECULATIVE_CONFIG=` (same as #32). `reference_kind=served_spec_off` confirmed via meta. **Every future spec submission that doesn't honor `SENPAI_REFERENCE_MODE` needs this `--ref-env` flag — should teach `serve.py` to honor it (follow-up item).**

3. **Self-consistency (1/1):** bit-identical output from two separate processes on 16 prompts confirms the int4 + CUDA-graph stack is deterministic at batch=1 served. This is expected but now empirically confirmed.

4. **Bare-tag assertion:** `harness.assert_submission_reference_tag(ref_tag)` placed at both generator and validator sites (lockstep). Smart adaptation: real function is 1-arg, takes already-resolved tag. Bare-baseline branch (pure model-id key) intentionally NOT guarded — correct design.

5. **Wall-clock fast:** 514.75s total (~14 min) vs the feared 2+ hours. The reasoning decodes ran faster than worst-case; 128 × 512-token completions total.

## 2026-06-13 — PR #37: lmhead12k verify-forward cost model + tile-corrected canonical curve ✓ MERGED

- **Branch:** `denken/lmhead12k-verify-cost` · **Student:** denken
- **Status:** MERGED — **cost-model closure + infra (tile-fold).** LOCAL profiling only; no HF job, no submission. Establishes the lmhead12k ceiling on the spec-verify path via directly-measured pod latencies.
- **Hypothesis:** Ubel #14's lmhead12k prune removes ~3 ms from the AR lm_head (PR #30: 1% of decode). Does it also remove a comparable fraction from the *verify* lm_head? The verify head runs on M=K+1=45 tokens simultaneously — if the head is memory-bandwidth-bound there too, the savings may be larger and flip PR #33's ">500 @ p=0.78 = NO" verdict.

### Results

| quantity (canonical = graph, ctx256) | full head (#33) | lmhead12k (measured) | analytic ceiling |
|---|--:|--:|--:|
| lm_head verify cost @ M=45 | 3.367 ms | **0.348 ms** (scatter floor) | 0.158 ms (×0.0469) |
| V_tree step @ M=45 | 15.235 ms | **12.212 ms** (−3.02 ms, −19.8%) | 12.022 ms |
| tree K* @ p=0.78 w/ drafter | K11/M45: 440.4 | **K11/M45: 538.1** | K11/M45: 545.8 |
| tree K* @ p=0.78 verify-only | K11/M45: 480.8 | **K11/M45: 599.8** | K11/M45: 609.4 |
| tree K* @ p=0.6792 w/ drafter | K11/M45: 359.9 | K7/M29: 446.6 (<500) | K7/M29: 451.7 |
| >500 @ p=0.78, K*-optimum, w/ drafter? | **NO** (440.4) | **YES (538.1)** | YES (545.8) |
| `primary_metric` `tree_tps_ceiling_p078_lmhead12k` | — | **538.1** | — |
| `test_metric` `verdict_exceeds_500_at_p078_lmhead12k` | — | **1** | — |

**W&B (verified by direct query):**

| run | name | key scalar | value | W&B |
|---|---|---|---|---|
| `klvpfk7g` | lmhead12k-verify-derive-measure | `V_full_M45`, `meas_k12_scatter_M45`, `lmhead_fixed_share_at_M45` | 15.235 ms, 0.348 ms, 0.860 | finished ✓ |
| `ruch259z` | lmhead12k-tree-ceiling-measured | `kstar_p078_W4_tps_withdrafter`, `verdict_exceeds_500` | **538.150, True** | finished ✓ |
| `6c9r3lih` | lmhead12k-tree-ceiling-analytic | `kstar_p078_W4_tps_withdrafter` | 545.816 | finished ✓ |

Group `spec-verify-lmhead12k` in `gemma-challenge-senpai`. Minor cosmetic gap: `V_lmhead12k_M45` logged 12.022 (analytic) vs PR table's 12.212 (measured) — label swap; does not touch the verified 538.1 headline (logged independently in `ruch259z`).

### Analysis & conclusions

1. **The verify-head prune is real and bounded.** Pruning to 12k rows removes ~3.0 ms from V_tree @ M=45 (−19.8%), because the verify forward streams the full bf16 head for each of the M=45 tokens in the speculative proposal. The saving is ~flat in absolute ms across M (it's a fixed head-weight bandwidth term), so its *fractional* contribution falls with M.

2. **The scatter floor is the correct honest ceiling.** The production `compute_logits` path scatters 12k partial logits back to a full [M,262144] −inf tensor + argmaxes over the full vocab for greedy-identity correctness (cannot be removed without a kernel rewrite). This costs 0.348 ms @ M=45 = ~2.2× the bare GEMM. Measured ceiling 538.1, not over-claimed analytic 546.

3. **Two-lens honest >500 reporting:** K*-optimum (538.1, >500 ✓ — matches #33's baseline frame, the headline lens) vs conservative fixed-K=6 with-drafter (476.5, <500 ✗). The flip needs p≥0.78 AND the K*-optimum lens. At realistic p=0.6792 with-drafter optimum stays <500 (446.6). Both lenses W&B-logged.

4. **Pipeline validated:** baseline column reproduces #33's K=11/M=45 440/481 @ p=0.78 exactly. Reduced curve trustworthy.

5. **K\*=11/M=45 serving guidance LOCKED for kanna #24 / any future spec submission.** PR #33's `optimal_k=15` scalars were the linear-W=1 lens artifact in run `36hkaj14` — corrected here (Step 5). Realistic W=4 tree optimum is K=11 (M=45) at both p=0.6792 and p=0.78.

6. **Infra: tile-fold into canonical msweep.** `fold_tile_into_msweep.py` folds #33's measured Marlin cliffs into `results_msweep.json` in place (pre-fold provenance at `results_msweep_prefold.json`). #26/#28 consumers now inherit the correct non-linear curve automatically.

7. **Suggested follow-ups from denken:** (a) eliminate the scatter floor (kernel argmax over 12k partial + remap full-vocab id — correctness proof needed, ~546 vs 538 ceiling); (b) tile-correct `eager/*` and `*/ctx512` keys (only `graph|ctx256` carries measured cliffs now); (c) validate ceiling against a real end-to-end spec-decode serving run.

## 2026-06-13 18:xx — PR #32: Greedy-gate reference-keying fix ✓ MERGED — validity-infrastructure correction

- **Branch:** `lawine/greedy-gate-ref-keying-fix` · **Student:** lawine
- **Status:** MERGED as a **validity-infrastructure fix**, NOT a TPS change. Served decode path byte-for-byte unchanged. CPU-only, no W&B.
- **Hypothesis:** The greedy-reference cache is keyed on `model_id` alone — two submissions sharing the same base checkpoint collide on a single cached reference, potentially causing silent false-PASS / false-FAIL on the greedy-identity gate.

### Results

| metric | value | verdict |
|---|--:|---|
| `collision_free` | **1.0** | collision hole CLOSED ✓ |
| `distinct_tags` | **2** | two submissions → two references ✓ |
| test guards (CPU-only, 6 assertions) | **6/6 pass** | correctness confirmed ✓ |
| fa2sw_precache_kenyan vs own M=1 AR (32 prompts, correct keying) | **DIVERGENT 27/32** | out-of-scope finding; routes to kanna |

### Analysis & conclusions

- **Root cause fixed:** reference cache keyed on `model_id` alone → submissions sharing a base model collide. Fixed by keying on `<submission_dir>::<model_id>` and threading a separate `reference_model_id` through `harness.py` / `gen_greedy_reference.py` / `validate_submission.py`. Audit trail: the resolved tag is now recorded.
- **`distinct_tags=2` confirms the old collision was real** — previously both submissions resolved to the same reference, rendering the greedy-gate meaningless for same-base-model submissions.
- **Keeper finding (routes to kanna):** under correct per-submission keying, `fa2sw_precache_kenyan` is **DIVERGENT 27/32** against its own M=1 AR reference. This is the data point kanna's served-gate validity audit must reconcile: the stack is leaderboard-valid at ~424.5 TPS but fails our strict M=1 bar — strong evidence our bar is over-conservative vs the leaderboard's served gate.
- **Unit-tested at the boundary:** `scripts/tests/test_greedy_ref_keying.py` (6 CPU-only guards: collision-free keying, distinct tags, key format). Correct test strategy for a correctness-of-validation change.
- **Next:** lawine reassigned — regenerate fa2sw_precache_kenyan reference at full 128 prompts + add runtime assert that resolved reference tag is never bare `"model"`. kanna → served-gate validity audit using the now-trustworthy keying.

---

## 2026-06-13 18:xx — PR #30: Frontier decode composition profile ✓ MERGED — authoritative component breakdown of ~420 TPS stack

- **Branch:** `wirbel/frontier-decode-profile` · **Student:** wirbel
- **Status:** MERGED as a **frontier decode characterization artifact**, NOT a TPS improvement. On-device component-resolved profile of `fa2sw_precache_kenyan` decode loop — the most strategically clarifying measurement of the cycle.
- **Hypothesis:** Decompose the decode cycle of the ~420 frontier (`fa2sw_precache_kenyan`) into GPU-time fractions by component (int4 body GEMM, sliding-window attention, drafter, lm_head) to rank remaining addressable levers and set priorities.

### Results

| component | fraction of decode cycle | verdict / implication |
|---|--:|---|
| Total GPU-bound | **99.3%** | host/launch overhead already negligible |
| **Verify-body int4 GEMM** | **53.2%** | dominant cost; walled at int4-Marlin floor |
| **fa2sw sliding-window attention** | **19.6%** | **second lever — most addressable** |
| Drafter | **15.5%** | third lever (drafter quality / steps) |
| lm_head | **1.0%** | collapsed from ~26.4% — validates lmhead12k (#14) ✓ |
| Verify bandwidth-bound / flat-in-M | M=1→8: **+25%** | tree widening nearly free on verify; K* set by acceptance geometry |
| E_accept | **3.817 tok/cycle** | current drafter acceptance at frontier |

W&B: `07kg6bn7` (authoritative, group `frontier-decode-profile`). `og7z6w0c` superseded.

### Analysis & conclusions

- **The decode loop is 99.3% GPU-bound.** Every remaining TPS gain must come from bytes-moved or FLOPs-cut inside kernels. This kills the "optimize launch/Python overhead" hypothesis for the frontier stack.
- **Verify-body GEMM (53.2%) is walled at the int4-Marlin floor.** There is no cheaper exact int4 matmul in vLLM 0.22.0. This eliminates the "find a faster verify GEMM" direction without a major kernel rewrite.
- **fa2sw attention (19.6%) is the live second lever.** It's large enough to matter (~100 TPS headroom if fully eliminated) and it's a kernel-addressable path (KV layout, SWA masking efficiency). This is where wirbel's next investigation goes.
- **lm_head collapsed to 1.0%** — independent validation that lmhead12k's 21.3× row-cut lands on the decode path, corroborating ubel #14 and wirbel #8. The lm_head lever is fully exploited.
- **Verify is bandwidth-bound / flat-in-M** — widening the tree is cheap on the verify side; the K* ceiling is set by acceptance geometry (acceptance rate p), not by verify cost per token. This corroborates PR #28/#33 cost-model findings and confirms the drafter quality (p) lever is the path to >500 TPS.
- **Cross-path validation:** `fa2sw_precache_kenyan` is the same stack lawine #32 used as the "out-of-scope" divergence case — now feeding directly into kanna's served-gate audit.
- **Next:** wirbel → fa2sw attention kernel-level deep-profile (19.6% second lever). kanna → served-gate validity audit using the #32-corrected keying. Artifacts: `research/profiling/frontier_decode/`, `scripts/local_validation/profile_decode.py`.

---

## 2026-06-13 17:52 — PR #24: Verify-rollback gate ✓ MERGED — THE LINCHPIN's final closure (greedy-valid spec-decode-for-speed is DEAD in vLLM 0.22.0)

- **Branch:** `kanna/verify-rollback-gate` · **Student:** kanna
- **Status:** MERGED as the **verify-rollback lane closure** (research artifact completing the #19→#24 arc), NOT a TPS baseline change. Official headline stays PR #4 (126.378).
- **Hypothesis:** Verify-rollback (per-step re-verify of accepted spec tokens under an M=1 AR forward; commit on match, rollback on mismatch) can restore greedy-valid spec decode **AND** maintain net-positive TPS over int4 AR — the only remaining greedy-valid-spec route after PR #19 closed the invariant-kernel lane.

### Results — hypothesis HALF-confirmed; the failing half is provably unfixable

| metric (eager n=32, W&B `ibmlc871`) | value | verdict |
|---|--:|---|
| flip_rate/tok, **verify-rollback** (vr vs M=1 ref) | **0.0** (`GREEDY_IDENTICAL` 32/32, 0/16384 divergent) | identity RESTORED ✓ |
| flip_rate/tok, raw spec (cand vs ref) | 0.332% | matches PR #19's 0.376% (CIs overlap) |
| rollback_rate/spec step (K=6) | 1.98% | matches ~2.2% theory |
| TPS int4 AR (spec-off) | 22.46 | the floor VR must beat |
| TPS int4 spec K=6 (raw, greedy-INVALID) | 49.75 | fast but fails the gate |
| **TPS verify-rollback (composed)** | **15.48 (0.69× AR)** | net-NEGATIVE ✗ |

Cudagraph n=16 (`354tydww`): VR flip 0.0 (16/16), AR 93.24, spec 229.71, **VR 66.32 (0.71× AR)** — also net-negative, far below the 126.378 official AR floor. All W&B arms verified to 4 sig-figs (no NaN); `tps_vr_composed` is transparently a derived field = 1/(1/AR+1/spec).

### Analysis & conclusions

- **The cost theorem (the keeper).** Net-positive TPS is impossible *by construction*, not by tuning: **you cannot know which 2.2% of steps roll back without computing the M=1 reference for ALL of them** — detecting a flip *is* running the M=1 forward (= one AR step). So re-verifying the j tokens a spec step accepts runs j sequential M=1 forwards = identical to the j forwards AR would run anyway. `TPS_VR = 1/(1/TPS_AR + 1/TPS_spec) < TPS_AR`, exact, implementation-independent. The PR's "extra M=1 only on the 2.2% that roll back" undercounted the re-verify work ~45× (re-verify rate is 100% of tokens). **Per-token M=1 → identity ✓ speed ✗; batched M=K → speed ✓ identity ✗ (M=K≠M=1 reintroduces the flips); no third option in a non-batch-invariant stack.**
- **Methodology accepted — composition, not a live engine.** Realized by composition (deliberate, disclosed): output identity is *definitional* (per-token rollback emits the M=1 AR argmax at every position → VR stream = M=1 AR stream bit-for-bit, confirmed on the real stream); cost is a *theorem* (both TPS arms real wall-clock; only the interleave composed). The PR's `spec_decode_worker.py` hook is a vLLM v0 path absent in 0.22.0 (v1 accept is in `rejection_sampler.py`/`gpu_model_runner._sample`); a live inline engine would burn GPU-days to reproduce a provable verdict. Advisor endorsed NOT building it.
- **Paper-premise correction (keeper).** arxiv 2601.17768 ("LLM-42", Gond et al.) targets **batch-self-consistency** (fixed-shape 256-wide re-verify; Obs. O3 relaxes to "position-consistent across runs"), **not** M=1-greedy-identity — greedy-DIVERGENT against our served reference if applied verbatim. Closes the "just implement the determinism paper" expectation.
- **Strategic consequence.** #19 closed the invariant-kernel route, #24 closes the rollback route → **spec-decode-for-speed under a strict M=1-greedy-identity gate is DEAD in vLLM 0.22.0.** The only net-positive greedy-valid-drafter route left is **source-level batch-invariance of the M=K+1 verify forward** (kanna follow-up #2 = stark #23; would make spec valid with ZERO rollback, strictly dominating VR). kanna follow-up #1 (is the ~420 frontier greedy-valid under the *served* gate without spec — is our strict M=1 bar stricter than the leaderboard enforces?) is the other open thread (feeds off wirbel #30).
- **Next:** kanna reassigned (verify-rollback lane closed); routed per the #30 frontier picture. Artifacts: `research/verify_rollback/{paper_notes.md,verify_rollback_patch.py,run_vr_arm.py,arms/}`.

---

## 2026-06-13 17:40 — PR #33: Tree-causal mask (dead) + Marlin tile-boundary correction ✓ MERGED — cost-model closure (NOT a TPS change)

- **Branch:** `denken/tree-causal-mask-verify-cost` · **Student:** denken
- **Status:** MERGED as a **LOCAL cost-model closure / profiler-infrastructure landing**, NOT a leaderboard/baseline change. Official headline stays PR #4 (126.378 a10g-small); best-LOCAL rung stays PR #14 (131.60 local). Directly refines PR #28's verify-latency curve.
- **Hypothesis:** A sparse tree-causal attention mask (each node attends only to its ancestors) cuts the attention term of the int4 verify forward at tree shapes K=6/8/12, potentially shifting the K=12 ceiling from PR #28's 452 toward 470–490 (or across 500) @ p=0.78. Secondary: the GEMM ramp steps at M≈20/40 are Marlin tile-boundary effects; a fine M-sweep finds the "free" plateau tree shapes.

### Results

| quantity (graph, ctx256, p=0.78, W=4) | PR #28 dense baseline | this PR (tree-masked + tile-corrected) |
|---|--:|--:|
| tree-mask saving M=25/33/49 — **production SDPA** | — | **0.000 / 0.000 / 0.000 ms** |
| tree-mask saving M=25/33/49 — FLOP-ideal ceiling | — | 0.076 / 0.104 / 0.175 ms (≤1.1% of step) |
| Marlin cliff Δ at M=17 / 33 / 49 | (interpolated, hidden) | **+0.772 / +2.176 / +2.869 ms** |
| **V_tree(M=49)** direct | 15.28 ms (interp) | **18.13 ms** (interp under-stated 2.68 ms / 17%) |
| tree K\* @ p=0.78 (drafter / verify-only) | K=12 (M=49): 452.4 / 493.4 (artifact) | **K=11 (M=45): 440.4 / 480.8** |
| **K12 tree TPS @ p=0.78** (primary metric, variant B) | 452.4 (artifact) | **393.9** |
| **verdict_exceeds_500 @ p=0.78** (test metric) | FALSE | **FALSE** (max 440 / 481) |

**W&B runs:** `k56d6cxe` (tree-mask), `36hkaj14` (tile boundary), `aid45far` (tree model), group `spec-verify-tree-mask` — all finished. Advisor sub-agent verified the tile deltas, M=49=18.134 ms, and `verdict_exceeds_500_at_full_scale_withdrafter=False` to logged precision.

### Analysis & conclusions

- **Finding 1 — tree-causal mask is DEAD for this model/hardware.** On the production dense-SDPA + topology-mask path (SpecInfer Eq.4 / EAGLE / Medusa / vLLM) the saving is **exactly 0 by construction** — a tree mask changes *which* scores are masked, not *how many* are computed. Even the unrealizable FLOP-ideal kernel saves ≤0.18 ms (≤1.1% of the step); FlexAttention is *negative* (the whole M≤49 tree fits one 128×128 block → partial-block overhead, pytorch #133562). Attention is only ~2.6% of the int4 verify step; the GEMM ramp dominates and is sparsity-invariant. Added to BASELINE.md dead-ends.
- **Finding 2 (the keeper) — Marlin tile-boundary cost-model bug-fix.** Step jumps at M=17/33/49 land *exactly* where `thread_m_blocks = ceil(M/16)` predicts (Marlin arXiv:2408.11743), and they are large (+2.18, +2.87 ms). PR #28's `LatencyCurve` linearly interpolated across them, **under-stating M=49 by 2.68 ms (17%)**. The corrected curve carries directly-measured boundaries — protects every future drafter-ladder TPS projection.
- **Net on the programme:** >500 TPS @ p=0.78 stays **FALSE — now firmer**; the only reading that approached 500 (variant-C 499.1) *was* the interpolation artifact this PR removes. **Serving guidance for kanna #24: target the M=45 (K=11) tmb=3 plateau, avoid M=17/33/49** — same accepted length, ~12% cheaper verify, no code change beyond tree shape.
- **One open reconciliation (non-blocking, flagged on PR):** report optimum K\*=11 (M=45) vs W&B-logged `optimal_k_*=15` (range-cap; likely the optimistic-accept scenarios — p=0.85 pushes K deeper to 511/558); `tps_tree_meas_p0_780=377.1` matches the K=6 sim exactly. denken to confirm scenario keying before the M=45 guidance is locked.
- **Suggested follow-ups (from denken):** (1) tell kanna #24 to target M=45 not M=49; (2) don't pursue tree-mask kernels here; (3) re-measure the M=45 plateau with a real per-position accept trace once a drafter lands; (4) fold the tile curve back into canonical `results_msweep.json`.
- **Next:** denken → fresh local profiling/cost-model assignment (incl. folding the tile correction into the canonical curve + the highest-value next decode-cost question).

---

- **Branch:** `ubel/empirical-lmhead12k` · **Student:** ubel
- **Status:** MERGED as a **validated lever + best-LOCAL rung**, NOT a new official baseline. Official a10g-small TPS + private-PPL await a gated HF job (approval issue opened). Official baseline headline stays PR #4 126.378.
- **Hypothesis:** Pruning the `lm_head` weight matrix to the top-12,288 most-frequent token rows (bf16, sliced from tied embeddings) cuts the lm_head GEMV bandwidth ~21× and yields a measurable single-stream TPS gain on the int4 base; it passes the official greedy-identity gate empirically (the pruned model is self-consistent) even though it is not adversarially safe.

### Results

| metric | unpruned control (bf16-262k head) | pruned (bf16-12k head) | delta | verdict |
|---|---|---|---|---|
| **tps_local_single_stream** (isolated, single-variable) | 97.65 | **131.60** | **+34.8%** | lm_head prune is real & standalone-positive |
| implied lm_head decode fraction | — | **27.1%** | matches wirbel #8's 26.4% | two independent measurements agree |
| local-to-local net vs PR #4 (int4-262k head, 128.13 local) | — | 131.60 | **+2.7%** | honest cross-config net (student's +3.6% mixed local-vs-official) |
| served_ppl (token-wtd) | — | **1.9712** | better than int4-head ~2.02 | ≤ 2.42 cap ✓ |
| greedy gate (served-vs-served, spec-off) | **GREEDY_IDENTICAL 128/128** | **GREEDY_IDENTICAL 128/128** | 0 divergent | valid (self-consistency) ✓ |
| completed | 128/128 | 128/128 | — | ✓ |

**W&B runs:** NONE (`wandb_run_ids: []`) — serve+validate experiment, no training run. Fully auditable via **38 committed evidence JSONs** under `research/local_validation/lmhead12k_empirical/` (`stage1_evidence/evidence.json`, `greedy_report.json`, `ppl_summary.json`, `control_int4_served/control_result.json`, `clip_floor_ksweep.json`, plus `vllm_baseline_128/` control). Advisor confirmed the marker progression (blocked_local_gpu → greedy_identity_divergent → running_corrected_gate → terminal) and the evidence-file backing; merge preflight passed.

### Analysis & conclusions

- **The lever is real and standalone-positive.** +34.8% isolated single-variable (only head row count differs) with an implied 27.1% lm_head decode-bandwidth fraction that independently matches wirbel #8's 26.4% profiler split. lmhead12k is **rung 5 of the BASELINE.md ladder** ("lmhead12k sparse-verify … the frontier"); this is the first in-repo standalone confirmation.
- **Three keeper validity findings** (sharpen our instrument): (1) the greedy gate is **self-consistency** (served-pruned vs plain-greedy-pruned, *same* checkpoint) — clipping cannot fail it by construction; the PRUNE-EFFECT (pruned-vs-*unpruned*) A/B measures fidelity to a model the gate never tests, not the gate. (2) The earlier 107/128 unpruned "control failure" was an **offline-batched-reference (batch≈128) vs strictly-sequential-candidate (batch=1) FP-reduction artifact** — *every* future greedy-gate run must use a batch=1 served-vs-served reference (wirbel #8's warning, larger here). (3) The int4-argmax clip rate has an **irreducible frequency-selection floor** (~0.78% public / 1.15% held-out) because some argmax tokens appear in *no* selection corpus — "held-out clip ~0" is unreachable by selection, and per finding (1) it isn't the gate anyway.
- **Honest framing:** per BASELINE.md, local A10G is exploratory-only; the official metric is a10g-small HF-Job TPS. So this merges as a validated lever/best-local rung, not a new official baseline. The +2.7% local net over PR #4 is plausible-but-unconfirmed officially (and the head dtypes differ: bf16-12k vs int4-262k).
- **Standing residual risk — private PPL** (not closable locally): a private GT-*target* token outside `kept_ids` → −∞ → +∞ PPL on the private re-run. Greedy-identity passes private by self-consistency, so this is purely a PPL axis. Only a gated a10g-small HF job on the private set closes it.
- **Next:** ubel → follow-up #3 (int4-pruned head, another ~4× head-byte cut, orthogonal to the kept-set). Also compounds in the spec-verify forward (gated on kanna #24). HF-approval issue opened for the official confirmation.

---

## 2026-06-13 17:30 — PR #25: EAGLE-3 full-scale training ✓ MERGED — keeper (drafter asset, reasoning acceptance 0.7314; DATA-bottlenecked)

- **Branch:** `fern/eagle3-full-scale-training` · **Student:** fern
- **Status:** MERGED as a research keeper (drafter asset). No TPS-baseline change — baseline stays PR #4 126.378 TPS (the drafter cannot deploy until kanna's verify-rollback #24 unlocks greedy-valid serving). The asset is the current-best drafter checkpoint, banked for the moment serving is unlocked.
- **Hypothesis:** Training the EAGLE-3 drafter at full scale (20k-step budget, benchmark-distribution data) past the PR #16 harness debug head (tf_acc 0.6816) pushes teacher-forced top-1 acceptance toward 0.78 — the level PR #28 says is needed to approach >500 TPS. Reframed mid-run: full MATH+ShareGPT as a per-source-decomposed arm to isolate whether chat data helps or hurts reasoning acceptance.

### Results

| metric | debug (MATH-only, 898 steps) | full (MATH+SG, 3500 steps) | delta | verdict |
|---|---|---|---|---|
| **tf_acceptance_rate, MATH holdout (n=48,142)** | 0.7051 | **0.7314** | **+0.026** | the benchmark-relevant number (128 public prompts are 100% reasoning) |
| tf_acceptance_rate, ShareGPT holdout | 0.1529 | **0.3444** | +0.19 | chat doubled but intrinsically hard to draft (high-entropy/multilingual/code) |
| tf_acceptance_rate, combined holdout | 0.5839 | 0.6464 | +0.063 | combined understates benchmark-relevant quality |
| val_loss, MATH holdout (final) | — | **1.2876** | — | reasoning fit |
| val_loss, combined (overfit signature) | — | 1.8516@2000 → 1.9519@3500 | +0.10 | overfits after ~2000 steps |

**W&B runs:** `7domtiin` (training — "crashed" = external interruption @ step 3670, `model_best.pt` step 3500 checkpoint intact) · evals `egv59ku0` (full·MATH 0.73136) · `xqtvcj58` (full·SG 0.3444) · `udb18hnh` (full·combined 0.6464) · `y0yupavk` (debug·MATH 0.7051) · `yxkh2739` (debug·SG 0.1529) · `1j8afmzk` (debug·combined 0.5839). All six eval runs finished clean; advisor independently verified headline 0.73136 and all per-source numbers to 4 s.f., no NaN. Training "crashed" status is an external interruption, not a divergence — checkpoint and eval lineage are intact.

### Analysis & conclusions

- **Reasoning acceptance is DATA-bottlenecked, not step-bottlenecked.** MATH-holdout tf_acc plateaus ~0.72–0.73 by step ~2000 (gains <0.004 per 500 steps thereafter), and combined val/loss *overfits* after step 2000 (1.8516→1.9519). More steps on this corpus will not break 0.73. The lever is **benchmark-matched reasoning CoT** (MMLU-Pro / GPQA / AIME-math), not more MATH and not more chat.
- **ShareGPT did not hurt reasoning** — it slightly *helped* MATH acceptance (0.7051→0.7314, via more total steps) while doubling its own acceptance (0.15→0.34). So mixing chat is safe, but chat is intrinsically low-acceptance (the combined 0.6464 is dragged down by the hard SG tail and understates the benchmark-relevant figure, since the 128 public prompts are 100% reasoning: mmlu_pro 57 / gpqa_diamond 57 / aime2026 14).
- **Ceiling caveat (PR #28 linkage):** tf_acc is a *teacher-forced UPPER BOUND* on free-running acceptance. PR #28 established >500 TPS needs free-running top-1 p≥0.85; 0.73 tf_acc maps to something lower free-running. So this asset, while the best drafter we have, is not yet the >500 TPS key — it sets up the next two levers.
- **Asset banked:** `research/eagle3_drafter/checkpoints/full_20k/model_best.pt` (step 3500, 0.7314 reasoning tf_acc). Corpus 2.21M tok (1.76M MATH + 0.45M SG), de-contaminated vs the 128 eval ids. Deploys the moment verify-rollback (#24) unlocks greedy-valid serving.
- **Student's flagged next step (correct):** a benchmark-matched reasoning corpus distilled from the served target on MMLU-Pro/GPQA/AIME. That is fern's next assignment — the corpus that should break the 0.73 plateau toward 0.78. On-policy distillation (Draft-OPD, round-3 H1) is the follow-on lever if static-corpus distillation plateaus below 0.85.

---

## 2026-06-13 17:00 — PR #28: Extended verify-latency M-sweep ✓ MERGED — keeper (ceiling corrected, extrapolation killed)

- **Branch:** `denken/verify-latency-msweep` · **Student:** denken
- **Status:** MERGED as a research keeper. Replaces the only extrapolated input in the PR #26 tree-salvage cost model with measured data. No TPS-baseline change — baseline stays PR #4 126.378 TPS.
- **Hypothesis:** The int4 verify forward stays bandwidth-bound and ~flat in M well beyond M=16, so extrapolating the PR #18 curve to M=25 (K=6 tree) and M=41 (K=10 tree) is safe, and the >500 TPS @ p=0.78 claim from PR #26 holds on measured data.

### Results

| metric | PR #26 extrapolated | PR #28 measured | verdict |
|---|---|---|---|
| V_tree(M=25) / V_lin(M=7) — K=6 tree overhead | 1.057× | **1.113×** | higher than extrapolated but ≪ 4× naive fear |
| K=6 tree TPS @ p=0.6792 | 346.8 | **331.2** (−4.5%) | net-positive 1.46×, holds |
| Tree K* @ p=0.78 | K=20 (M=81): **616 TPS** (extrapolated) | **K=12 (M=49): 452.4 TPS** | **30% overstatement** — interior optimum found |
| >500 TPS @ p=0.78? | YES (extrapolated K≈10) | **NO — max 452/493 TPS** | ceiling refuted at debug-head acceptance |
| Knee M* | ≥16 (edge of old sweep) | **M≈24** (ramp starts M≈20) | step-structure from tile quantization |

**W&B runs:** `2mk0z0c3` (latency M-sweep, group `spec-verify-msweep`) · `imoi4mx1` (tree acceptance model, group `spec-verify-msweep`). Both finished; all cited numbers verified vs W&B artifacts (60-row cost table, 120-row tree table).

### Analysis & conclusions

- **The hypothesis is partially refuted — and that's the finding.** The verify forward IS flat through M≈32 (+2.6%), so the K=6 moderate tree (M=25) extrapolation was essentially sound (1.057→1.113×). But beyond M≈32 the int4 Marlin W4A16 GEMM goes compute-bound and ramps: M=40 +31%, M=64 +60% over M=1. Discrete steps at M≈20, 32, 64 are Marlin tile-boundary quantization effects.
- **The ramp is GEMM, not lm_head.** The forward GEMM share rises 62%→68% through the ramp; lm_head grows smoothly (2.86→3.57 ms). CUDA-graph mode exposes the ramp (eager masks it with fixed CPU-launch overhead).
- **The REAL interior optimum is K*≈8–12** (not K=20). At p=0.78: K=8 (M=33) gives 429.3 TPS → peaks at K=12 (M=49): 452.4 TPS → then declines as ramp outpaces saturating acceptance.
- **>500 TPS requires drafter quality, not deeper trees.** Only at p≥0.85 (top-1 acceptance ≥0.85) does the K=12 tree clear 500 (531 TPS). The debug-head acceptance regime (p≈0.68) caps at ~366–406 TPS (K*=8). **This re-anchors the entire team's focus on fern #25 (EAGLE-3 full-scale training) as the ceiling-setter.**
- **Dense-M upper-bound caveat** (reported by student): the profiler times a dense/full-causal M-token forward (upper bound). The true tree-causal-masked cost is cheaper only in the attention term (16%→13% of the ramp), so the GEMM-dominated correction is sub-2 ms at M≈49 — tight upper bound.
- **Strategic re-anchor:** K*≈8–12, not K≈20. The next steps are (a) tree-causal mask measurement to tighten the dense-M upper bound, (b) EAGLE-3 training to push p toward 0.85, (c) kanna's verify-rollback to unlock serving.

---

## 2026-06-13 16:20 — PR #27: int4 channel-wise lm_head sweep ✗ CLOSED — confirmed NEGATIVE (g128 stays the floor)

- **Branch:** `lawine/int4-channel-lmhead-sweep` · **Student:** lawine
- **Status:** CLOSED as a clean, fully-characterized NEGATIVE. No TPS-baseline change — baseline stays PR #4 126.378 TPS. The channel submission dir stays on the student branch (dead-end; not merged).
- **Hypothesis:** channel-wise (`group_size=-1`) int4 lm_head gives +~1 TPS over g128 (PR #4) because per-output-channel dequantization requires a simpler scale lookup in the Marlin GEMV kernel; PPL cost small (lm_head error affects low-confidence vocab tail). Single-variable change: one line in `submissions/int4_g128_lmhead/build_quant.py`.

### Results

| metric | g128 control (PR #4) | channel-wise (g=-1) | delta | verdict |
|---|---|---|---|---|
| local TPS (A10G, 128 prompts) | **128.13** | 127.74 | **−0.39** | NO GAIN — within noise |
| local PPL (128 prompts / 61,797 tok) | **2.0188** | **2.0212** | +0.0024 | ≤ 2.42 cap ✓ |
| greedy identity (self spec-off) | GREEDY_IDENTICAL 128/128 | **GREEDY_IDENTICAL 128/128** | — | valid ✓ (0 divergent / 65,536 tok) |
| same-path PPL gate | SAME_PATH_OK (gap 0.0) | **SAME_PATH_OK (gap 0.0)** | — | honest ✓ |
| completed | 128/128 | **128/128** | — | ✓ |
| Marlin g=-1 support | — | confirmed (no g=32 fallback needed) | — | — |

**W&B runs:** `gtlruguu` (channel prevalidate, TPS 127.74/PPL 2.0213) · `a0xtk79t` (g128-ctrl prevalidate, TPS 128.13/PPL 2.0188) · `c9qy6rcq` (channel validation, same_path_gap 0/SAME_PATH_OK/128/128). All three in `gemma-challenge-senpai` or `wandb-applied-ai-team/senpai`; all finished; independently verified by advisor to >3 sig figs, no NaN.

### Analysis & conclusions

- **The TPS gain did not materialize.** The lm_head is a single GEMV per decode step over a tiny fraction of total decode traffic; the scale-lookup simplification for g=-1 vs g128 is sub-noise at the whole-model level. The PPL moved +0.0024 (well under +0.011 projection and far under the 2.42 cap), and the greedy self-gate is byte-exact 128/128 — the coarser head did NOT flip any near-tie argmax.
- **Net verdict:** channel-wise is SAFE but POINTLESS as a speed lever. **lm_head quant granularity is not a TPS knob.** A head-side TPS lever must come from a smaller effective vocab at decode (the lmhead12k direction), not from g128→channel.
- **HF approval issue:** correctly NOT opened by lawine (no improvement to confirm). Correct protocol.
- **The real deliverable:** lawine's **bug flag** — a **silent-correctness hazard on the greedy-gate auto-reference resolution** (`harness.py:84-92` manifest `env.MODEL_ID="model"` copied into serve env before `setdefault` → `srv.model_id` stays the relative literal `"model"` → `reference_for("model")` keys shared `greedy_reference/model/` tag → NO_REFERENCE AND every `env.MODEL_ID="model"` submission collides on the same tag → silent wrong-reference verdict risk). The actual GREEDY_IDENTICAL was confirmed offline via `--reference` flag (sound). **lawine reassigned to harness fix → PR #32**.

---

## 2026-06-13 15:49 — PR #22: Honest fa2sw-precache frontier in-repo + LF29 dual-gate-blind finding ✓ MERGED — keeper (asset + validity)

- **Branch:** `wirbel/fa2sw-precache-validate-and-lf29-check` · **Student:** wirbel
- **Status:** MERGED as a research keeper (plain squash; no TPS-baseline change — baseline stays PR #4 126.378 TPS). Two deliverables: (A) the honest ~420 TPS frontier stack is now an in-repo VALID base; (B) a validity finding about our own tooling.
- **Hypothesis (two-part):** (A) reproduce kenyan-duma's honest precache frontier locally; it should pass the same-path PPL gate (gap ≈ 0). (B) the pupa-lf29cap444 lane is a grader-conditional FFN bypass → same-path PPL gate should return gap ≈ 0.17 → FAIL.

### Results

| part | gate | result | verdict |
|---|---|---|---|
| **A** — kenyan-duma honest frontier | same-path PPL (`same_path_ppl.py`) | gap **0.0000**, both paths PPL **2.37688**, bit-identical NLL (11 sig figs) | `SAME_PATH_OK` — confirmed single-path honest ✓ |
| **B** — pupa-lf29cap444 | same-path PPL (teacher-forced) | gap **0.0000**, PPL **2.37794** (NOT the predicted 0.17) | `SAME_PATH_OK` — gate is **blind** to this fold |
| **B** — pupa-lf29cap444 | greedy identity (fold-on vs exact-FFN, spec-off AR, 65,536 tok) | **0 flips / 128 prompts identical**, `flip_rate_per_token=0` | `GREEDY_IDENTICAL` — fold is argmax-safe |
| W&B | `jg99477i` (Part A), `tju905db` (Part B same-path), `gz5b064e` (greedy gate) | all 3 finished; metrics verified vs logged summary (5+ sig figs) | no fabrication |

### Analysis & conclusions

- **Part A asset:** `submissions/fa2sw_precache_kenyan/` (serve.py + patches, no weights — synced at runtime) is now an in-repo VALID base for future TPS work (tree-salvage, accepthist, EAGLE-3 can branch from the real frontier stack). Mechanism documented component-by-component in `research/validity/fa2sw_precache_notes.md`. Local exploratory TPS 867 tok/s (NOT official a10g-small — liveness only).
- **The headline finding — both output gates are BLIND to the LF29 fold class.** The pupa LF29 lane keys layer-29 FFN on `num_prompt_logprobs` (exact FFN when PPL is graded; cheap affine fold for timed decode) — confirmed in `serve.py:411-415`. But the deployed fold is **both teacher-forced-PPL-neutral AND argmax-safe**: same-path PPL gap 0.0000 (forcing the fold ON every request gives 2.3767, marginally *below* exact-FFN 2.3779) and greedy flip_rate 0/65,536. **Neither same-path PPL nor greedy_gate can detect this lane.** The only detector is **static mechanism inspection** of the grader-conditional branch. This corrects the prior research-state assumption that `greedy_gate` is the load-bearing detector for fold-class lanes — it is also clean here. BASELINE.md's "every HF-approval issue requires `--check-same-path` output" reads PASS even for this invalid lane.
- **The 2.55 mystery:** neither output gate reproduces frantic-penguin/itaca's community 2.55. Since greedy text is byte-identical to exact-FFN (0 flips ⇒ no prefix divergence ⇒ no error compounding), free-running greedy PPL on pupa's deployed weights is ≈2.378. The 2.55 is most likely a **reconstructed** fold (R²≈0.80, not pupa's weights) or a non-greedy regime — needs the external frantic-penguin method to settle.
- **Intellectual honesty:** wirbel falsified their own hypothesis (predicted gap 0.17 / flip>0; measured 0/0), reported faithfully, and held the board post for human approval (Issue #29). Excellent diligence.
- **Scope-limit doc kept:** `research/validity/same_path_ppl.md` now permanently documents that same-path PPL + greedy_gate are blind to argmax-preserving / decode-compounding folds; mechanism inspection is load-bearing.
- **Follow-ups:** (1) wirbel reassigned → **PR #30** (frontier decode-step profile on the new in-repo `fa2sw_precache_kenyan` base — find the next TPS lever beyond 421). (2) **Issue #29** opened (board post to evals taskforce) — HELD, human-gated; advisor verified the W&B evidence but is NOT approving publication. (3) Suggested team direction: a static mechanism-scanner for grader-conditional request-field branching — the only detector for this fold class.

---

## 2026-06-13 15:20 — PR #26: Tree-salvage acceptance model (width-4 tree vs linear K) ✓ MERGED — keeper (cost model)

- **Branch:** `denken/tree-salvage-acceptance-model` · **Student:** denken
- **Status:** MERGED as a research keeper (no served checkpoint / no TPS-baseline change; baseline stays PR #4 126.378 TPS). Plain squash-merge. `scripts/profiler/tree_acceptance_model.py` + extended `eval_eagle3.py` (top-k + trace) now canonical.
- **Hypothesis:** width-4 tree decoding raises E[accepted tok/invoke] substantially over linear K=6 for our EAGLE-3 head, and the acceptance gain outweighs the tree-verify overhead → realistic TPS ceiling >500 at full-scale acceptance.

### Results

| metric | value | note |
|---|---|---|
| top-1 acc | 0.6792 | reproduces PR #16 tf_acc 0.6816 (within 0.4%) |
| top-4 acc | 0.8605 | hypothesis ≥0.82 ✓ |
| **rescue_rate (width-4)** | **0.5651** | **beats fableous 0.431 by +0.134** — our head is more tree-salvageable |
| E_accept tree4 / linear (empirical) | **1.5923** | primary metric; i.i.d. model agrees (1.60) |
| **measured tree-verify overhead** | **1.06×** | M=25 forward ≈ as cheap as M=7 (PR #18 flat-in-M); NOT the feared 4× |
| K=6 tree TPS @ p=0.6792 | 346.8 (+53% vs linear 227.3) | verify V=12.05ms **extrapolated** at M=25 |
| full-scale ceiling @ p=0.78, K=6 | **393 TPS** (w/ drafter) | `verdict_exceeds_500_at_full_scale = False` at K=6 |
| >500 TPS @ p=0.78 | only at K≈10 (M≈41, **extrapolated**) | beyond PR #18 measured M≤16 |
| W&B | eval `8idbwjk1`, cost-model `zlzti9h0` (group `tree-salvage-acceptance-model`) | all metrics independently verified vs logged summary |

### Analysis & conclusions

- **Tree-salvage is real and net-positive on this hardware.** The decisive fact is the **1.06× measured verify overhead**, not the acceptance gain alone: under a 4×/additive verify model the tree is net-negative; under PR #18's measured bandwidth-bound (flat-in-M) curve it's +53%. The tree-salvage case **depends on the int4-verify-flat-in-M finding** — a clean, physically-grounded refutation of the naive "4× tree cost" framing.
- **Validates the acceptance lever for kanna's verify-rollback path (#24).** With overhead ~1.06× and E gain ~1.6×, width-4 tree at K≈6–8 is the concrete config to prototype once spec decode is greedy-valid.
- **Honest limits (denken flagged all):** (1) the >500 @ full-scale is conditional — needs p→0.78 AND deep K≈10 where M≈41 is **extrapolated** beyond PR #18's measured M≤16; (2) empirical trace is slightly *sub*-geometric (0.96× i.i.d.) — the "easy-span" positive correlation hypothesized did NOT appear on this head+MATH set, though the tree/linear ratio is preserved so the gain conclusion is robust; (3) D=1.4ms is fableous's *linear* drafter cost — a width-4 tree drafter expands K·W nodes so may cost more (verify-only vs +drafter band brackets it).
- **Checkpoint-provenance catch (excellent diligence):** the PR-named `debug_1k/` is a 28-step underfit (tf_acc 0.2484); the real 0.6816 head is `debug_1k_2ep/` (898 steps), confirmed against W&B `30bgs1rs`. denken evaluated the correct head on held-out `debug_1k_eval_corpus.pt` and staged canonical paths. **Note for fern #25 / future drafter work: use `debug_1k_2ep/`, not `debug_1k/`.**
- **Follow-up assigned → denken PR #28:** extend the PR #18 verify sweep to M∈{20,24,28,32,40,48,64} to replace the M=25/M=41 extrapolation with measured latency — the only soft spot in the >500 projection.

---

## 2026-06-13 14:38 — PR #4: int4 g128 + untied int4 lm_head (~127 TPS) ✓ MERGED — new leaderboard baseline rung

- **Branch:** `lawine/int4-g128-lmhead` · **Student:** lawine
- **Status:** MERGED — new best merged rung. `submissions/int4_g128_lmhead` is now the best merged submission. All future submissions beat 126.38 TPS.
- **Hypothesis:** untied int4 lm_head (eliminating the bf16 GEMV for 262k-vocab verify = 26.4% of decode GPU time per PR #8 profiler) + full-body g128 granularity (slight additional weight-byte reduction vs per-layer) → reaches the int4 Marlin weight-byte floor on Ampere.

### Results

| metric | value (official a10g-small) | vs PR #3 base |
|---|---|---|
| tps / output_tps | **126.378** | 1.32× (**+32%**) |
| ppl (served) | **2.019** | ≤ 2.42 ✓ |
| completed | **128 / 128** ✓ | — |
| greedy identity | **GREEDY_IDENTICAL 128/128** (served-vs-served cap=512) ✓ | — |
| same-path gate | **SAME_PATH_OK (gap 0.0000)** ✓ | — |
| job | `6a2d5a96234ca64b60121aa5` | — |
| W&B | `905tbujn` (official a10g-small) · `0pxj6n63` (local proxy + greedy) | — |

**Overall: 2.87× over bf16 (44.018 TPS), 1.32× over PR #3 int4 base.**

### Analysis & conclusions

- **Confirms lmhead profiler finding** (PR #8): 26.4% of decode GPU time was the 262k-vocab bf16 GEMV. Untied int4 lm_head eliminates it, explaining the +32% TPS gain. This is the exact profiler prediction.
- **This is the weight-byte floor.** Sub-4-bit (no sm_86 kernel) and fp8 KV (no A10G support) are dead ends. No further weight-bandwidth reduction is achievable in vLLM 0.22.0 on Ampere. Every remaining TPS lever is either (a) the drafter ladder (spec decode, gated on kanna verify-rollback), (b) lmhead12k (ubel #14, cheaper verify), or (c) runtime/warmup (precache, onegraph — the frontier stack).
- **Greedy validity methodology confirmed:** served-vs-served (spec-off) via `check_greedy_identity.py` passes cleanly (GREEDY_IDENTICAL 128/128). This is the gold-standard test.
- **lawine confirmed official PPL artifact** present on the HF job result — closing the near-cap timing question from last cycle.

---

## 2026-06-13 14:38 — PR #19: Batch-invariant vLLM spec decode ✓ MERGED — LINCHPIN DEFINITIVE NEGATIVE

- **Branch:** `kanna/batch-invariant-vllm-spec` · **Student:** kanna
- **Status:** MERGED — definitive negative. Closes the invariant-kernel lane. Next lane: verify-rollback (kanna PR #24).
- **Hypothesis:** `VLLM_BATCH_INVARIANT=1` (aten-override batch-invariant kernels) makes the M=K+1 verify forward bit-match the M=1 AR forward → greedy-identical spec decode.

### Results

| arm | INV | target GEMM | flip/tok | 95% CI | identical/32 | W&B |
|---|---|---|---|---|---|---|
| int4 ON (decisive) | 1 | Marlin `_C` (un-covered) | **0.376%** | [0.234, 0.518]% | 5/32 | `hz8jkc5h` |
| int4 OFF (control) | 0 | Marlin `_C` (un-covered) | 0.332% | [0.205, 0.460]% | 6/32 | `8wne15eh` |
| bf16 ON (discriminator) | 1 | aten linear (covered) | **0.111%** | [0.057, 0.166]% | 16/32 | `z0mclftv` |
| bf16 OFF (PR #5 ref) | 0 | aten linear | 0.72% | — | — | — |

**Primary metric:** int4_mtp_batchinv_greedy_flip_rate_per_token = **0.00376** (0.376%) — NOT zero. **Verdict: DIVERGENT, invariant-kernel lane CLOSED.**

### Analysis & conclusions

The bf16 control arm is the key insight. By removing int4 Marlin (using aten-covered bf16 GEMM) while keeping INV=1, we isolate TWO independent un-coverable root causes:

- **(a) int4 Marlin `_C` op:** contributes ~0.265%/tok excess above bf16 floor. The Marlin custom op is outside aten's scope; batch-invariance cannot intercept it. This was the main prior hypothesis (Marlin was "plausibly already M-invariant") — REFUTED.
- **(b) Spec verify path non-aten residual:** bf16 ON (full aten coverage, zero Marlin) is STILL divergent at 0.111%/tok. An irreducible non-aten component in the spec verify forward (attention-metadata build, rejection-sampler logits compare, or a fused step) remains batch-variant. Corroborated by vLLM issue #27433: "batch-invariance does not currently integrate with speculative decoding."
- **Consistency check:** 0.265% (a) + 0.111% (b) ≈ 0.376% (observed int4 ON). The two sources are independent and additive.
- **Implication:** neither int4 nor bf16 target drafter ladders are rescuable by `VLLM_BATCH_INVARIANT`. The invariant-kernel lane is closed for greedy-valid spec decode at ANY precision in vLLM 0.22.0.
- **Next lane:** verify-rollback (arxiv 2601.17768) — re-verify accepted tokens under fixed-shape M=1 reduction after each spec step; commit consistent / roll back violators. This targets both causes: (a) is dodged (rollback uses M=1 AR path, no Marlin batch-size dependency on committed path), (b) is caught and corrected by the re-verify. Assigned to kanna PR #24.

---

## 2026-06-13 14:38 — PR #16: EAGLE-3 draft-head training harness ✓ MERGED — keeper research artifact

- **Branch:** `fern/eagle3-training-pipeline` · **Student:** fern
- **Status:** MERGED — keeper (training harness + asset). No leaderboard TPS improvement; infrastructure needed for the drafter ladder.
- **Hypothesis:** An EAGLE-3 draft head trained via offline distillation from Gemma-4 E4B (using aux hidden states from layers 2, 21, 39) can achieve teacher-forced acceptance ≥ 3.5 tok/step on a held-out STEM corpus at debug scale.

### Results

| metric | value | note |
|---|---|---|
| tf_acceptance_rate_debug_1k | **0.6816** | at 1k steps, 200 MATH train samples |
| final_val_loss_debug_1k | 1.3372 | still converging |
| W&B | `30bgs1rs` (group `eagle3-drafter-training`) | |

**Verdict:** pipeline confirmed functional. 0.6816 is in the "0.50–0.70 → schedule full run" range.

### Analysis & conclusions

- **Harness architecture:** faithful PyTorch reimplementation of vLLM's Eagle3DraftHead with vLLM-matching weight names/shapes (deployable checkpoint). Llama decoder layers (not Gemma), RoPE/RMSNorm/GQA/SwiGLU. feature_shift=1 vLLM-faithful alignment. Chunked 262k-way CE to avoid OOM.
- **Corpus:** EleutherAI/hendrycks_math (allenai/MATH 404s), 200 train samples, 52,751 tokens.
- **Key finding:** no public Gemma-4 E4B EAGLE-3 checkpoint exists (thoughtworks/Gemma-4-31B-Eagle3 is shape-incompatible) → trained from scratch.
- **Next:** full-scale training (2000 MATH + 500 ShareGPT samples, 20k steps, targeting tf_acc ≥ 0.78) assigned to fern PR #25. Serving is gated on kanna's verify-rollback PR #24.

---

## 2026-06-13 14:38 — PR #18: int4 decode-step cost model vs K ✓ MERGED — keeper research artifact

- **Branch:** `denken/spec-verify-cost-model` · **Student:** denken
- **Status:** MERGED — keeper (analytical cost model). No leaderboard TPS improvement; foundational analysis for drafter-ladder decisions.
- **Hypothesis:** characterize the ideal TPS ceiling of int4 spec decode as a function of K (draft count) and acceptance probability p.

### Results

| metric | value | note |
|---|---|---|
| tps_ceiling_ideal_at_kstar | **1,269.5 TPS** | at K*=15, acceptance p=0.7 |
| optimal_k_geom_p0.7 | **K*=15** | geometric acceptance, 40% of weight-GEMM time is verify |
| W&B | `pvj0qogp` (group `spec-cost-model`) | |

### Analysis & conclusions

- **The sky is high:** 1,269.5 TPS ideal ceiling (at p=0.7, optimal K) confirms the drafter ladder has massive headroom. Even at p=0.5, the ceiling is > 600 TPS.
- **K=6 is suboptimal:** at p=0.7, ideal K*=15. The current MTP drafter at K=6 leaves TPS on the table even at full acceptance. Higher acceptance rate raises K* — tree decoding (fableous: width-4 rescues 43.1% of linear misses) could change the optimal strategy.
- **Feeds verify-rollback net-value:** the cost model now establishes the ceiling. denken's next assignment (PR #26) extends it to tree decoding.
- **Dropped dependency in rebase:** no functional issue — the cost model files (research/spec_cost_model/ + scripts/profiler/spec_cost_model.py) are self-contained; the dropped dependency was an unmerged PR-specific hook that was correctly removed.

---

## 2026-06-13 14:15 — PR #22: Honest precache frontier + LF29cap same-path validity (SENT BACK, WIP)

- **Branch:** `wirbel/fa2sw-precache-validate-and-lf29-check` · **Student:** wirbel
- **Status:** NON-TERMINAL (pending_arms=true). Sent back for greedy_gate on pupa-lf29cap444 + terminal marker.
- **Hypothesis:** (A) reproduce kenyan-duma honest precache frontier (PPL ~2.377); (B) test whether pupa-lf29cap444 fails the same-path PPL gate (gap ~0.17).

### Part A results (PASS — clean asset)

| metric | value |
|---|---|
| same_path_ppl_gap (fa2sw_precache) | **0.0000** (SAME_PATH_OK, exit 0) |
| same_path_ppl | **2.37688** |
| NLL equality | byte-identical to 11 sig figs — single-path confirmed |
| W&B | `jg99477i` |

Part A confirmed: kenyan-duma honest precache frontier is single-path at the strongest possible resolution. Clean VALID base for tree-salvage / accepthist / EAGLE-3 branching.

### Part B results (UNEXPECTED finding — important tooling insight)

| metric | predicted | measured | verdict |
|---|---|---|---|
| same_path_ppl_gap (pupa-lf29cap444) | ~0.17 / FAIL | **0.0000 / SAME_PATH_OK** | gate is BLIND to this class |
| fold-forced same_path_ppl | — | 2.3767 (−0.0013 vs exact) | fold is teacher-forced-neutral |
| W&B | — | `tju905db` | |

**Critical finding (structural — affects all future validity work):** the same-path PPL gate (merged PR #21) is **teacher-forced-blind** — it cannot detect argmax-preserving / decode-compounding folds. The LF29 affine fold (ridge approximation of layer-29 FFN, R²≈0.80) is teacher-forced-neutral because each token is scored on the ground-truth prefix; the fold's cost is in free-running decode where argmax flips compound. Two independent mechanisms: (1) teacher-forced scoring is fold-neutral by construction; (2) `echo+logprobs` is coupled to `prompt_logprobs` in vLLM (`completion/protocol.py:276-277`), tripping the same bypass exemption. **→ `greedy_gate` (served-token identity) is the load-bearing validity instrument for fold-class lanes.** The same-path gate catches logit-level path splits (request-field branching on `prompt_logprobs`).

This corrects the BASELINE.md scope statement: "every future HF-approval issue must attach `--check-same-path` output" still holds for logit-path split detection, but greedy_gate is ALSO required for fold-class lanes. The `research/validity/same_path_ppl.md` scope-limit update (wirbel PR #22) will land when the PR merges.

### Next steps (pending)

wirbel authorized to run greedy_gate on pupa-lf29cap444 (local, spec-off served-vs-served). Expected: flip_rate > 0 (the fold changes decode-path argmax where the approximation crosses a decision boundary). Board post held for human approval. Terminal marker expected once greedy_gate completes.

---

## 2026-06-13 14:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) ✓ MERGED — first official int4 base rung

- **Branch:** `stark/int4-qat-w4a16` · **Student:** stark
- **Status:** MERGED — new official base rung of the reproduction ladder. `submissions/int4_qat` is now the best merged submission.
- **Hypothesis:** int4 W4A16 (Marlin) is the dominant single-stream speed lever (decode is memory-bandwidth-bound; quartering text-linear weight bytes bf16→int4 lifts ~44→~95 TPS). Google's QAT checkpoint keeps PPL *below* the bf16 reference (~2.01 vs 2.30), so faster AND safely inside the 2.42 cap.

### Results

| metric | value (official a10g-small) |
|---|---|
| tps / output_tps | **95.463** (2.17× over bf16 44.018) |
| ppl | **2.0057** (≤ 2.42 cap ✓; better than bf16 2.30) |
| completed | **128 / 128** ✓ |
| total_tps | 144.53 (diagnostic) |
| duration_s | 686.5 · job_status COMPLETED ✓ |
| greedy identity | valid within same serve/job stack (no token-changing optimization added) |
| job / run | `6a2d55c7234ca64b60121a6f` / `results/senpai/int4-qat-20260613T130614Z` |

**W&B run:** N/A (serving-submission reproduction, no training). Official artifacts under `results/senpai/int4-qat-20260613T130614Z/`. Local proxy ≈ 95.99 TPS / 2.0055 PPL (<0.6% off official).

### Analysis & conclusions

- int4 W4A16 confirmed as the **dominant single-stream lever on official hardware**: ~4× less weight bandwidth, the foundation the entire ~420 frontier stack builds on. Base rung is now an official, valid, merged result.
- **Cold-start/40-min-cap did NOT bite** for a submission this fast: `ppl_summary.json` wrote 13:42:23Z, ~3.5 min before the cap. PPL is cheap (one forward pass) and benchmark+decode run ~2.2× faster than bf16. (Slower stack rungs later will tighten this margin — keep the watch.)
- All modalities loaded (vision/audio bf16 via QAT `ignore` list, no `--limit-mm-per-prompt`). No text-only shortcut.
- **Next rung already landed:** lawine PR #4 (int4 g128 + untied int4 lm_head) reports official **126.378 TPS / PPL 2.019 / GREEDY_IDENTICAL 128/128**, +32% on this base — merging once rebased onto this commit + official-ppl artifact confirmed.

## 2026-06-13 13:00 — PR #21: Same-path PPL gate ✓ MERGED

- **Branch:** `wirbel/same-path-ppl-gate` · **Student:** wirbel
- **Status:** MERGED — validity tooling protecting all future HF submissions. No TPS change.
- **Hypothesis:** for an honest single-path submission, timed-generation-path PPL equals prompt_logprobs-path PPL; a non-zero gap (>0.05) reveals grader-conditional branching on `bool(num_prompt_logprobs)`.

### Results

| metric | value |
|---|---|
| `same_path_ppl` (echo/no `prompt_logprobs`) | **2.3012128792** |
| `prompt_logprobs_ppl` (official path) | **2.3012128792** |
| `\|gap\|` | **8.88e-16 ≈ 0.0000** |
| gate verdict | **SAME_PATH_OK** (exit 0) |
| GT records | 128/128 |
| scored tokens | 61,797/61,797 |

**W&B run:** `b9igh00q` (wandb-applied-ai-team/gemma-challenge-senpai, group `same-path-ppl-gate`, finished, all values verified).

### What was built

- `scripts/local_validation/same_path_ppl.py` — scores reference continuations via the generation path with **no `prompt_logprobs` field** in the request (indistinguishable from timed throughput). Uses `echo:true` + `logprobs:1` to read per-token logprobs without triggering the branch a gamed submission would key on.
- `--check-same-path` flag wired into `validate_submission.py` — non-zero exit if `|gap| > 0.05`.
- Calibration artifacts at `research/validity/vllm_baseline/` (both `*_summary.json` + `*_results.jsonl`).
- Documentation at `research/validity/same_path_ppl.md` with honest-vs-gamed reference points.

### Why this matters (public context)

The LF29cap lane (pupa-agent 459 TPS / need-for-speed 457 TPS, cmpatino-verifier "VERIFIED VALID") was confirmed grader-conditional by frantic-penguin (`20260613-090759-237`): `lffn_ppl_exact_active = (LFFN_PPL_EXACT==1 and bool(num_prompt_logprobs))` — `prompt_logprobs` grader gets exact FFN (PPL 2.378), decode gets cheap affine fold (same-path PPL 2.5499, > 2.42 cap). PPL 2.3779 identical across ALL LF29cap verifier re-runs (smoking gun: frozen artifact). frantic-penguin escalated to cmpatino-verifier + evals taskforce. Our gate cleanly separates honest (gap ≈ 0) from gamed (gap ≈ 0.17).

**Required from now on:** every HF-approval issue must attach both `greedy_gate` verdict + `--check-same-path` output.

### Critical scope note

Gate catches request-field branching on `prompt_logprobs`. Does NOT catch `echo`-branching or prefix-cache replay keyed on public-prompt content. Named residual attack surfaces in `research/validity/same_path_ppl.md`.

### Advisory action

- PR comments addressed (advisor guided probe design: no `prompt_logprobs` in request).
- wirbel assigned next task (#22): reproduce kenyan-duma honest precache frontier locally + apply gate to LF29cap lane + publish to evals taskforce.

---

## 2026-06-13 12:55 — PR #14: Empirical lmhead12k ↩ REVIEW → request-changes (int4-argmax re-selection)

- **Branch:** `ubel/empirical-lmhead12k` · **Student:** ubel
- **Status:** WIP (non-terminal; `greedy_identity_divergent_pending_decision`). Reviewed, requested changes, sent back. NOT merged (greedy-invalid), NOT closed (alive + crisp fix).
- **Hypothesis:** pruning the 262k lm_head to a ~12,288 kept-vocab set cuts lm_head GEMV bandwidth (~5–8% TPS over the int4 base) while preserving PPL ≤ 2.42 and greedy identity.
- **Results (local A10G, exploratory):**

| metric | pruned lmhead12k (12,288) | bf16 stock | gate | verdict |
|---|---|---|---|---|
| TPS (single-stream) | 128.23 | 43.95 | higher | ≈ int4 base (prune delta unmeasured — no unpruned-int4 control) |
| served PPL | 1.9767 | 2.3012 | ≤ 2.42 | ✓ (but blind — see below) |
| completed | 128/128 | 128/128 | =128 | ✓ |
| greedy-identity | **DIVERGENT** | (ref) | required | ✗ **invalid** |

- W&B: none logged (local serve/validate, no training). Artifacts: `research/local_validation/lmhead12k_empirical/{greedy_identity_summary,greedy_prune_effect_int4full_vs_pruned,select_analysis,*_summary}.json`.
- **Root-cause finding (valuable, non-obvious):** `kept_ids` was selected from the **bf16** model's argmax, but the served model is **int4**. int4 quantization moves ~1.33% of greedy-argmax decisions (874/65,536; 114/128 prompts) to tokens bf16 never emits → pruning clips them → near-tied survivors flip across numeric paths → DIVERGENT. Clean offline-eager A/B (int4full vs pruned) confirms the prune itself diverges (10/128), independent of serving config. **The kept set covers the wrong model.**
- **PPL is blind to greedy clips:** the −inf scatter on 250k pruned rows shrinks the softmax denominator, *inflating* every kept token's logprob (PPL 1.98 < bf16 2.30). Teacher-forced PPL cannot see a greedy argmax clip. Reinforces why same-path/greedy gates (not PPL) are the validity backstop.
- **Decision & rationale (request changes):** fix = re-select `kept_ids` from the **int4** model's argmax over a **broad corpus** (not public-128-specific), sized so the int4-argmax-outside-kept clip rate is ~0 on public AND a held-out split. Report the **held-out clip rate** = private greedy-identity failure rate (the lmhead12k analog of private TPS drift). Re-run the gate **served-vs-served** (wirbel #8), not offline-eager (avoids ~20% false divergence). Cheap add: serve an unpruned-int4 control to isolate the prune's conc=1 TPS delta — if ~neutral, lmhead12k's value lives in the spec-decode verify forward (gated kanna #19), not standalone. Drafter-independent rung; GPU now available.

## 2026-06-13 (cycle 9) — PR #16: EAGLE-3 draft-head training pipeline ↩ INTERIM REVIEW → sent back (option c)

- **Branch:** `fern/eagle3-training-pipeline` · **Student:** fern
- **Status:** WIP (not terminal). Reviewed an interim/blocking-question update; steered, did not merge or close.
- **Hypothesis:** an EAGLE-3 head distilled from Gemma-4 E4B aux states `(2,21,39)` can reach offline teacher-forced acceptance well above the QAT-MTP baseline, and the training pipeline is functional + CUDA-graph-compatible.
- **What landed (Steps 1–4, validated):** faithful plain-PyTorch `Eagle3DraftHead` (vLLM-matching weight names/shapes; the vLLM head is inference-only/no-autograd), from-scratch (no compatible public Gemma-4 EAGLE-3 ckpt), frozen tied embed/lm_head init, chunked 262k-way CE (avoids `[N,262144]` fp32 OOM), `feature_shift=1` vLLM-faithful alignment. Corpus: `EleutherAI/hendrycks_math` (allenai/MATH 404s), 200 train + 20 held-out, 52,751 tokens. Peak GPU **11.2 GB**.

### Interim result (accidentally cap-constrained 2-epoch run)

| epoch | step | held-out tf_acceptance | held-out loss |
|---|---|---|---|
| ~0.5 | 7 | 0.066 | 5.68 |
| ~1.0 | 14 | 0.192 | 4.64 |
| ~1.5 | 21 | 0.236 | 4.19 |
| **~2.0** | **28** | **0.248** | **4.10** |

Train loss 12.97→3.72, train acc 0→0.295. W&B `rxxd8yen` (group `eagle3-drafter-training`).

### Decision & rationale

fern flagged a **binding conflict**: the PR's "1000 steps" over the 200-sample corpus = ~71 epochs, violating the live launch's accidental `SENPAI_MAX_EPOCHS=2` bound. fern correctly **refused to override** the bound and ran the max-compliant 2-epoch run. The held-out acceptance is monotone and **still climbing steeply at the cap** (chance ≈ 4e-6) — viability is demonstrated, but 0.248@28-steps is too weak to anchor the full-scale go/no-go.

**Revised steer:** the pod cap has been raised to `SENPAI_MAX_EPOCHS=9999`, so the student should run the intended 1000-step debug training directly, using a corpus broad enough to avoid a public-slice memorization artifact. Terminalize with a defensible `tf_acceptance_rate_debug_1k`. Serving/full-scale remain gated on (a) this number and (b) the int4 spec greedy-identity linchpin (#19). EAGLE-3 is the highest-ceiling drafter (lit. ~480–550 TPS) and is deployable on the public VALID frontier's drafter (`e1`) spec path independent of the int4 linchpin.

## 2026-06-13 (cycle 8) — PR #7: fa2sw + onegraph runtime levers ✗ CLOSED (negative)

- **Branch:** `denken/fa2sw-onegraph`
- **Student:** denken
- **Status:** CLOSED — rigorous, well-isolated NEGATIVE. Both runtime levers are dead ends standalone on the int4 base at conc=1. Knowledge preserved here and in BASELINE.md "Confirmed dead ends."
- **Hypothesis:** fa2sw (route 35× hd-256 sliding-window local layers to FlashAttention-2) + onegraph (`cudagraph_mode=FULL`) erase per-step overhead at conc=1, enabling a TPS gain over the int4 base without drafter or lmhead changes.

### Results

| variant | TPS (local, conc=1) | Δ vs base | greedy (official verifier, 128-prompt) |
|---|---|---|---|
| base (int4 QAT W4A16) | **96.89 ±0.01** | — | REFERENCE |
| fa2sw only | 92.11 ±0.02 | **−4.9%** | **DIVERGENT** 82/128 (12,075 tok) |
| onegraph only | 96.82 ±0.00 | ~0% (parity) | **DIVERGENT** 1/128 (59 tok, @idx 197) |
| both | 92.12 ±0.00 | **−4.9%** | **DIVERGENT** 82/128 (11,767 tok) |

**W&B run:** `57bb3a6s` — ablation matrix table + per-variant metrics.

### Analysis

Both levers **fail the strict zero-tolerance greedy gate**, so neither can ship standalone regardless of TPS:
- **fa2sw:** FA2 sliding-window numerics ≠ Triton → near-tie argmax flips on 82/128 prompts. The mixed FA2+Triton backend also *blocks* a single full-graph capture, producing the −4.9% TPS regression.
- **onegraph:** A pure graph-capture knob (`cudagraph_mode=FULL`) still perturbs the numeric path (one near-tie argmax flip) — confirms the "different numeric path even from a pure graph-capture knob" warning.
- **fa2sw dominates** — `both` == fa2sw's divergence set; onegraph's addition doesn't expand the failure set.

**Root cause of no TPS win:** Decode at conc=1 is **~92% weight-GEMM / bandwidth-bound** (attn ≈2.6%, sampling ≈0.2%). The existing CUDA graph already collapses the decode step into one launch. There is **no per-step overhead left to reclaim** standalone at conc=1. This closes the "per-step overhead gap" hypothesis for these two levers.

**Determinism control (bonus finding — 4th int4 greedy-determinism reconciliation data point):**
Int4 base is **cross-process bit-exact** (sha256 `base_clean`==`base_clean2`, also deterministic in eager mode). The divergences above are a real mechanism, not run noise. This is the clearest data point yet: int4 base greedy **IS gate-valid in M=1 sequential prefix-cache-OFF**, narrowing the linchpin to the *spec M=K+1 batched-verify path* specifically.

**fa2sw serving caveat:** fa2sw cannot be served via a serve-process monkeypatch — vLLM V1 spawns a separate EngineCore process; a real fa2sw serve path requires a **vLLM worker-plugin** entry point. Moot since it's invalid, but prevents wasted re-discovery.

### Suggested follow-up (from denken, evaluated by advisor)
fa2sw layered *on top of the MTP drafter* (where attention share under spec verify may be higher) — valid direction but drafter-gated (kanna #5 linchpin). Assigned denken the hardware-grounded TPS ceiling curve instead (PR #18: decode-step cost model vs K), which directly quantifies when attention-share rises enough for fa2sw to matter.

---

## 2026-06-13 11:15 — PR #15: EAGLE-3 feature-export feasibility ✓ MERGED

- **Branch:** `fern/eagle3-feature-export-feasibility`
- **Student:** fern
- **Status:** MERGED — binary feasibility verdict: ACCESSIBLE → GO. Research report + reusable probe script. No TPS change; foundational prerequisite for the highest-ceiling drafter path.
- **Hypothesis:** Multi-layer intermediate hidden states from Gemma-4 E4B ARE accessible from vLLM 0.22.0's model executor (either natively or via a minimal model-class override).

### Results

| field | value |
|---|---|
| `eagle3_hiddens_accessible` | **1 (yes, natively)** |
| Access mechanism | Built-in `SupportsEagle3` interface — zero patching |
| Model-class override effort | **0 hours** (already implemented) |
| Aux layers (default) | `(2, 21, 39)` over the 42-layer E4B body |
| Aux shape/dtype | `[num_tokens, 2560]` bf16 per layer |
| CUDA-graph compatible | **Yes** (persistent buffers pre-allocated at capture) |
| Drafter head arch | Already exists: `llama_eagle3.py`, `v1/spec_decode/eagle.py` |
| W&B run | None (source audit + single model-load probe) |

**Empirical probe (PR #15 `probe_result.json`):** `supports_eagle3=True`, `default_aux_layers=[2,21,39]`, 3 tensors `[5,2560]` no NaN; vision+audio towers intact; 15.3 GiB peak bf16 on A10G.

**Key vLLM source refs (vLLM 0.22.0):**
- `model_executor/models/interfaces.py:1285-1392` — `EagleModelMixin` + `SupportsEagle3` Protocol
- `gemma4_mm.py:917-923` — `Gemma4ForConditionalGeneration implements SupportsEagle3`
- `gemma4.py:958` — `Gemma4Model is EagleModelMixin` (42 layers)
- `v1/worker/gpu_model_runner.py:4861-4987` — concatenates 3 aux layers `dim=-1` (that's the EAGLE-3 multi-layer fusion)
- `v1/worker/gpu/cudagraph_utils.py:382-395` — persistent aux buffers for CUDA-graph safe capture

**Serving-validity gate:** greedy-identity of EAGLE-3 spec decode on int4 is gated on kanna #5 linchpin (int4 batched-verify greedy-validity).

### New shared infra
`research/eagle3_feasibility/{feasibility_report.md, probe_eagle3_export.py, probe_result.json, probe.log}`

### Recommendation → GO
Full EAGLE-3 drafter head training assigned to fern (PR #16). Literature projects **480–550 TPS** at ~4–5+ accepted tok/step. Serving run gated on kanna #5 linchpin.

---

## 2026-06-13 10:45 — PR #13: SAM-Decoding drafter-overlap intersection analysis ✓ MERGED

- **Student:** fern
- **Status:** MERGED — CPU-only infra extension to `analyze_suffix_budget.py`. No TPS change; shared tooling for net-headroom decision.
- **What was built:** `--drafter-trace <file>` extension; `drafter_overlap` block with `net_sam_beyond_drafter_frac` (the GO/marginal/retire decision number); 13/13 mock tests pass; no-drafter path byte-identical (regression-safe). Canonical trace format (`output_start` for spec interleave alignment). `research/sam_drafter_overlap/overlap_analysis_template.json`. Dev dep `pytest>=8` added.
- **Metrics:** `sam_causal_frac_gt_k8_base_reproduced=0.0893` (PR #10 anchor), `mock_tests_passed=13`.
- **Net-headroom thresholds:** `net_frac > 3%` → Triton kernel GO; `1–3%` → marginal; `< 1%` → retire SAM.
- **Caveat (fern):** real MTP drafter concentrates acceptances on predictable/repetitive spans — exactly where SAM runs live — so real overlap likely HIGHER → real net LOWER than naive intuition. Base 8.93% is small; brace for marginal/retire.
- **Next:** tool ready; trace landing depends on kanna's linchpin outcome (PR #5 → real acceptance trace gated on greedy-validity resolution).
- **Reproduce:** `cd target/ && uv run python -m pytest scripts/tests/test_drafter_overlap.py -v`

## 2026-06-13 10:45 — PR #14: Empirical lmhead12k (pruned-weights top-12k vocab) — IN PROGRESS (non-terminal, blocked)

- **Student:** ubel
- **Status:** NON-TERMINAL (`terminal=false`, `status=blocked_local_gpu`) — sent back to WIP with advisor answers. GPU void on pod (intermittent); int4 base checkpoint not on node. Implementation complete (CPU feasibility done, GPU steps pending).
- **Key findings (change the plan):**
  1. **12k underspecified:** 128 benchmark prompts have only 7,338 unique tokens — can't frequency-fill to 12,288 from the benchmark alone. Tight kept set = 7,584 (34.6× bandwidth). Must use a general corpus to reach 12,288 faithfully.
  2. **Hard-include public GT tokens is NECESSARY:** official PPL scorer (`ppl_endpoint.py:163-183`) does NOT floor −∞ for out-of-vocab tokens → GT target token outside kept vocab → −∞/missing → gate fail. The tight set is intrinsically public-tailored; would fail private PPL re-run. General-12,288 cut is required for private validity.
  3. **Only 31/128 decode captures available locally** (fern's 128-capture gitignored, not on scratch bucket); greedy-identity proven on 31 only.
- **Serving design (correct):** custom vLLM model class `Gemma3ForCausalLMLMHead12k` — scatters kept-row logits into full 262,144 (−∞ on pruned) inside `compute_logits` (VOCABTRIM-style); `LogitsProcessor` path insufficient (V1 reads `prompt_logprobs` before logits processors).
- **Advisor answers:** self-build int4+g128 base via path-(a) (prune bf16 → quantize, deterministic from public source, no cross-node dep); build general-12,288 cut from broad STEM corpus; regenerate full 128 decode capture; report both bandwidth numbers.
- **Note: DRAFTER-INDEPENDENT** — not affected by kanna's spec-decode linchpin. Building block toward ~420 regardless of linchpin outcome.

## 2026-06-13 10:30 — PR #5: int4 + MTP/QAT drafter spec-decode ({8,4} engine fix + greedy-validity finding) — REQUEST CHANGES (→ WIP)

- **Branch:** `kanna/int4-mtp-drafter`
- **Student:** kanna
- **Status:** REQUEST CHANGES — terminal SENPAI-RESULT but submission **INVALID** (greedy DIVERGENT). Sent back to WIP for a decisive precision-localization experiment. The `{8,4}` backport + wandb-scraper fix are keepers on the branch.
- **Hypothesis:** int4 W4A16 target + QAT-MTP drafter spec-decode reaches ~285 TPS greedy-identical once the vLLM 0.22.0 `{8,4}` attention-group blocker is fixed.

### Results (local A10G, exploratory; W&B group `int4-mtp-drafter`)

| K | mean accepted tok/step | exploratory TPS (A10G) | PPL | greedy | W&B run |
|---|---|---|---|---|---|
| 5 | 2.151 | 164.45 | 2.0064 | DIVERGENT | zbt1fras |
| 6 | 2.197 | 163.87 | 2.0064 | DIVERGENT | 7vnkis8z |
| 7 | 2.188 | 160.28 | 2.0064 | DIVERGENT | 0fa5c8fx |

W&B cross-check (advisor): tps/ppl/accept match the PR verbatim; `greedy_identical=0` boolean = DIVERGENT confirmed; the malformed `spec/accept_rate_posN` values are the pre-fix scraper bug kanna disclosed and fixed.

### Engineering win — `{8,4}` blocker SOLVED
Backported upstream vLLM PR #43543 / commit `dede691c9536` ("split attention groups by `num_heads_q` for spec-decode drafts") as a fork/spawn-safe runtime monkeypatch (`vllm_attn_group_patch.py` + `sitecustomize.py`). Serves cleanly eager + cudagraph. (The PR-cited commit `3e8afdf7` is WRONG — that's a Cohere2MoE fix; the real fix is #43543.)

### CRITICAL FINDING — int4 spec-decode is structurally greedy-DIVERGENT in vLLM 0.22.0
At temp=0 vLLM's rejection sampler emits `argmax(target_logits)` from the **batched M=K+1 verify forward**; plain AR (the reference) emits `argmax` from the **M=1 decode forward**. int4 Marlin accumulation is batch-shape-dependent → logits differ in the last bits → ~0.33%/token argmax flips on near-ties → compounds to DIVERGENT over 512 tokens (6/32 prompts identical). Structural for any K≥1; no batch-invariant/deterministic knob exists in 0.22.0 (kanna grep-confirmed). K0-vs-K0 control is IDENTICAL → divergence is 100% the spec verify path.

### Advisor verification of the gate mechanics (this cycle)
- Read the official verifier (`gemma_greedy_identity_verifier_flowian-powers/greedy_identity.py`): **strict bit-exact**, full `completion_token_ids`, zero tolerance — any 1 flipped token → DIVERGENT.
- Traced the harness (`speed_benchmark/scripts/{hf_bucket_single_job,decode_outputs}.py`): it generates ONLY the candidate decode (128×512, seed 1, temp 0, ignore_eos); the **reference is organizer-held** = "plain greedy decode of the submitted checkpoint" = int4 M=1 AR — exactly what kanna compared against. **kanna's DIVERGENT is very likely the official verdict.** Refutes her hypothesis (c) "audit is lenient."

### LINCHPIN question (gates rungs 4–5 / the path to 420)
If int4+vLLM-spec cannot be greedy-valid in 0.22.0, how is the ~420 frontier VALID? Remaining hypotheses: **(a)** higher-precision target (fewer near-tie flips, but can't hit 420 at int4 bandwidth) or **(b)** batch-invariant kernels in a newer vLLM (only if the harness honors manifest `python_packages`). **Next experiment (assigned to kanna):** hold the spec stack fixed, vary target precision (int4 vs bf16 vs fp8), measure greedy flip-rate per arm — localizes the divergence and decides whether the drafter ladder is salvageable. Plus: definitively confirm whether a10g-small honors the manifest vLLM version.

### Secondary
Acceptance underdelivers: 2.20 tok/step (vs ~3.3 target) — strong pos0 (87%) but steep decay caps speedup ~2.2× (~270 effective TPS). Real-prompt corroboration: K6 340.9s vs K0 730.2s = 2.14×.

## 2026-06-13 10:30 — PR #9: Wide-distribution KL-distilled drafter (private-stable acceptance) — REQUEST CHANGES (→ WIP)

- **Branch:** `land/wide-drafter-distill`
- **Student:** land
- **Status:** REQUEST CHANGES — tf-gate PASSES but native serving regressed; sent back for v1 (free-running schedule). Drafter infra + deduped corpus are keepers on the branch.
- **Hypothesis:** A wide, distribution-matched (4-dist) KL-distilled drafter lifts acceptance uniformly — including the chat/private-proxy floor — improving private-set stability over the reasoning-skewed stock drafter.

### Results (offline acceptance, held-out shard; committed JSONs `research/wide_drafter/eval/{stock,wide}.json`)

| metric | stock | wide (v0) | Δ |
|---|---|---|---|
| tf accepted-tok/step (the gate), overall | 3.455 | 3.811 | **+0.356 (+10.3%)** |
| tf — chat (private proxy) | 2.753 | 3.052 | **+0.299 (+10.9%)** |
| native `generate(assistant_model=)` overall | 3.553 | 3.388 | **−0.165 (−4.6%)** |

W&B run `eqqdeodf` (group `wide-drafter-distill`). **Reporting gap (advisor W&B check):** the cited run logged only `train/*` loss curves — the acceptance numbers live in committed JSONs + reproduce commands, NOT in W&B. v1 must log the heldout eval to W&B.

### Analysis
- Width corpus works on the metric it optimizes: +10.3% tf, **uniform incl. chat/private-proxy floor (+10.9%)** — the target signal. Dedup proof: zero overlap with the 128 public prompts.
- **Native regressed −4.6%, uniformly** — train↔serve schedule mismatch (teacher-forced training vs free-running serving) + undertraining (0.87 epoch, 40 of 90 budget-min unused, losses still falling). Correctly diagnosed by land.

### Next (v1, assigned to land)
Change ONE variable: **free-running / scheduled-sampling (EAGLE-3-style) unroll** to close the exposure-bias gap; same ~5k corpus + recipe; full ~82-min budget; primary = `heldout_native_accept_per_step` (beat stock 3.553); log eval to W&B. Optional 2nd arm: narrow-corpus contrast to isolate the width variable.

### Infra/methodology notes
- `scripts/drafter/offline_eval.py` is the correct EAGLE-aware acceptance tool (the reference `shared_resources/.../offline_acceptance.py` mis-measures EAGLE drafters as standalone CausalLM — flagged to wirbel #8).
- `google/gemma-4-E4B-it-assistant` is the correct control; `Tonykip/...` baseline didn't resolve (fine). hf_xet wedge → `HF_HUB_DISABLE_XET=1`.
- Coupling: converting acceptance → served TPS depends on int4 spec being greedy-valid (kanna #5's linchpin question).

## 2026-06-13 10:00 — PR #6: Greedy-safe vocab-prune / top-k sparse-verify (verify-cost lever) ✗ CLOSED (negative)

- **Branch:** `ubel/vocab-prune-sparse-verify`
- **Student:** ubel
- **Status:** CLOSED — confirmed dead end (provable Cauchy-Schwarz certificate, 0%-fire on Gemma4 geometry). Option A authorized: empirical lmhead12k (new PR incoming).
- **Hypothesis:** A Cauchy-Schwarz sufficient certificate determines per decode step whether the greedy
  argmax is within the top-K kept set — allowing the step to skip the full 262k GEMM if certified,
  with a greedy-safe adversarial fallback when not.

### Results (measured on A10G, K=12000, 64 prompts × 256 tokens = 16,384 decode steps)

| metric | value | verdict |
|---|---|---|
| Certificate fire rate | **0.0%** (0 / 16,384 steps) | dead end |
| Fallback rate | **100%** | always pays full 262k GEMM |
| Isolated lm_head GEMM speedup (12k vs 262k kept) | **20.1×** | ceiling for the empirical approach |
| Effective speedup with cert overhead | **0.92×** (−8% slower) | provable lever LOSES |
| TPS (net) | null (slower than baseline) | — |
| PPL (128/128 GT records, 61,797 tokens) | 2.304 | ≤ 2.42 ✓ |
| Greedy identity (128 public prompts) | GREEDY_IDENTICAL (trivially — 100% fallback) | ✓ |
| Adversarial fallback (rare-token test) | PASS (cert correctly refuses → full GEMM emits true argmax) | ✓ |
| Unit tests | 7/7 PASS | ✓ |
| W&B run | none | — |

### Root cause — model-intrinsic geometry obstruction

`R_complement_max_norm = 1.630` vs real `z_max/||h|| ≈ 0.59` → the Cauchy–Schwarz sufficient
condition **provably cannot fire** on real Gemma4 hidden states. The model has flat row norms, tiny
kept-vs-pruned margins, and a near-full-rank embedding. No kept-set construction rescues the cert
on this lm_head. The **Cauchy-Schwarz provable-greedy-cert family is a confirmed dead end on
`gemma-4-E4B-it`**.

### Key program finding

The frontier's `lmhead12k` (kenyan-duma, 421.12 TPS VALID) is the **empirical prune**: compute
only top-12k logits, emit the kept-argmax, **no per-step certificate**. It captures the ~20×
isolated GEMM speedup. It is NOT adversarially safe — the rare-token case diverges (ubel measured
this: id 258090 outside 12k → kept-only emits 188798). It passes the official greedy-identity
check because benchmark prompts apparently do not generate rare tokens. The empirical approach is
what the leaderboard rewards; the provable approach cannot compete on this geometry.

**On this lm_head: provable safety OR TPS win — not both.**

### Decision

- Provable greedy-safe cert (Cauchy-Schwarz) on Gemma4: **DEAD END**. Added to BASELINE.md.
- **Option A authorized:** build the pruned-weights empirical `lmhead12k` checkpoint (top-12k
  rows of the int4+g128 lm_head), serve it, measure TPS/PPL/greedy-identity + rare-token divergence
  rate. New PR for ubel: `empirical-lmhead12k`.

---

## 2026-06-13 09:45 — PR #10: Offline suffix-run token-budget analysis for SAM-Decoding feasibility ✓ MERGED

- **Branch:** `fern/sam-decoding-offline-analysis`
- **Student:** fern
- **Status:** MERGED (`c8dfdb3`) — analysis deliverable + shared infra (`scripts/analyze_suffix_budget.py`).
- **Hypothesis:** The SAM-Decoding paper (arXiv 2411.10666) claims a 3.6–3.9% verbatim-suffix-run
  budget on reasoning prompts. Confirm on our 128 benchmark prompts; produce a go/no-go for the
  Triton in-graph suffix-match kernel (Rank 5 from round-2 research).

### Results

| budget definition | K>4 | K>6 | **K>8** | K>10 | verdict (K>8) |
|---|---|---|---|---|---|
| `m(t)` (PR spec; adjacent-only, non-causal) | 1.47% | 1.37% | **1.21%** | 1.14% | no-go (flawed proxy) |
| **Causal SAM realized** (actionable, greedy-safe) | 15.37% | 11.60% | **8.93%** | 7.16% | **GO** |
| ↳ causal decode-steps-saved (TPS-correct) | 13.74% | 10.66% | **8.35%** | 6.77% | — |
| LPF forward-oracle (loose upper ref) | 30.56% | 21.37% | 16.21% | 12.42% | — |

**Per-dataset causal K>8:** aime2026 10.74% | gpqa_diamond 9.23% | mmlu_pro 8.19% (uniform 8–11%).

SENPAI-RESULT: `{"terminal":true,"status":"complete","frac_tokens_gt_k8":0.0121,"causal_sam_realized_frac_gt_k8":0.0893}`

**Decision metric:** causal_sam_realized_frac_gt_k8 = **8.93%** → **GO** (>3.6% threshold).
`frac_tokens_gt_k8` (0.0121) is the literal PR-spec `m(t)` value — documented but *not* the decision metric.

### Key points

- **`m(t)` is a flawed proxy:** fires only on adjacent-period repetition (the s tokens immediately before t
  reappearing at t). Only 127 such runs across all 128 prompts (~1/prompt). The exploitable structure is
  non-adjacent — prompt re-quotes, formula restatements, repeated option text — which `m(t)` cannot see.
- **Causal estimate validated:** cross-checked against brute-force O(n²) causal reference: 0 mismatches
  over 600 positions. Robust to nondeterminism: 10.51% (PR #2's 16-prompt capture) vs 10.49% (this
  run's first 16 prompts) — Δ0.02pp.
- **Greedy-safe:** SAM-Decoding verifies each drafted token against live target logits → greedy-safe by
  construction → zero PPL risk.
- **Critical caveat:** the ~420 TPS frontier already runs an MTP/QAT model-drafter (~3.3 tok/step).
  SAM adds to it; the incremental gain = causal budget MINUS drafter-accepted positions. Net headroom
  can only be measured by intersecting causal suffix runs with the drafter's per-step acceptance trace
  (needs kanna's #5 to serve). This is the de-risking step before the Triton kernel build.

### New shared infra

`scripts/analyze_suffix_budget.py` — offline CPU-only suffix-budget analyzer. Designed for extension
with a `--drafter-trace` flag to intersect causal suffix runs with a drafter acceptance trace and
output the net incremental headroom.

**W&B run:** none (CPU-only offline analysis). 128/128 prompts captured (bf16, 43.94 TPS local).
**Artifacts:** `research/local_validation/suffix_budget/suffix_budget_analysis.json` (committed).

### Next steps

- **fern** extends `analyze_suffix_budget.py` with drafter-overlap intersection + synthetic mock-trace
  validation (non-blocked, CPU-only). Once kanna's #5 drafter serves and emits an acceptance trace,
  the net-headroom number is one command away.
- If net_headroom > 3%: assign Triton in-graph suffix-match kernel PR.
- If net_headroom < 1%: SAM direction adds near-nothing to the drafter stack — retire.

---

## 2026-06-13 09:30 — PR #4: int4 g128 + untied int4 lm_head re-quant (~127 TPS weight floor) [IN PROGRESS — awaiting HF Job]

- **Branch:** `lawine/int4-g128-lmhead`
- **Student:** lawine
- **Status:** WIP — local evidence complete; **awaiting human approval of HF Job (GitHub issue #12)**
  before posting terminal SENPAI-RESULT with official a10g-small numbers. Held at the int4 (PR #3)
  rung deliberately: the ladder is confirmed bottom-up and, per BASELINE.md, local A10G numbers are
  exploratory only — no merge to a confirmed TPS rung without the official a10g-small score.
- **Hypothesis:** Re-quantizing the QAT base (`gemma-4-E4B-it-qat-q4_0-unquantized`) to group_size=128
  across all 343 body modules plus an **untied int4 `lm_head`** (`embed_tokens` kept bf16) hits the
  int4-Marlin Ampere **weight-byte floor**, lifting single-stream TPS from the ~95 int4 base to ~127
  with PPL essentially unchanged (~2.02). This is the last "fewer weight-bytes/token" lever before
  sub-4-bit (a confirmed sm_86 dead end).

### Local Results (exploratory, A10G — NOT official a10g-small)

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (served, 128/128 GT records, 61797 tokens) | **2.0190** | ≤ 2.42 | ✓ |
| Offline fake-quant PPL | 2.0197 | ≤ 2.42 | ✓ |
| Local TPS (exploratory, A10G, single-stream) | **127.99** | — | on target ~126.8 (+33% over int4 base ~96) |
| Greedy identity (official served-vs-served, standard cap=512 config) | **GREEDY_IDENTICAL** 128/128 prompts, 16384/16384 tok, 0 divergent | byte-exact | ✓ |
| Quantized modules | 343 body @ g128 + untied int4 lm_head = 344 total, 9.62 GiB on disk | — | ✓ |
| compressed_tensors version | 0.15.0.1 (vLLM 0.22.0's shipped version) | — | ✓ (see note) |
| All modalities | vision/audio loaded | — | ✓ |
| W&B run | `0pxj6n63` (`wandb-applied-ai-team/senpai-v1`, finished) | — | ✓ corroborates tps 127.99 / ppl 2.019 / GREEDY_IDENTICAL, logged verbatim |

### Key points

- **TPS lever:** 127.99 local = +33% over the int4 base (~96 local) and +0.9% above the ~126.8 public
  ladder target — confirms the int4-Marlin weight-byte floor on Ampere. group_size 128 + untied int4
  `lm_head` is the last weight-bytes/token reduction available (sub-4-bit AWQ/GPTQ/etc. have no
  loadable sm_86 kernel in vLLM 0.22 — confirmed dead end in BASELINE.md). lawine's track is at its
  natural floor; the next lever above this rung is the drafter (kanna #5 / land #9), not more quant.
- **Greedy identity (same resolution as stark's PR #3):** the official gate is served-vs-served at a
  SHARED config. lawine proved **GREEDY_IDENTICAL 128/128 at the standard cap=512 config**; spurious
  divergence only appears under cross-config (no-cap reference vs cap=512 candidate). Not a blocker.
- **Version note:** the PR body states compressed_tensors==0.10.2 but lawine actually built against
  **0.15.0.1** — the version vLLM 0.22.0 ships. 0.15.0.1 is the correct/required choice; 0.10.2 is
  incompatible with vLLM 0.22.0. Acknowledged on the PR; the built checkpoint is the valid artifact.
- **PPL-metric note (reusable):** the scored gate metric is the token-weighted `served_ppl=2.0190`
  (`exp(Σnll/Σtok)` over all 61,797 tokens). The W&B run also logs an unweighted per-record mean
  `served_mean_record_ppl=2.1787`, which runs higher because short records weigh equally — it is
  informational only, not the contract metric, and both are under the 2.42 gate.

### Next Steps

- Human approves GitHub issue #12 → lawine runs
  `python train.py --submission submissions/int4_g128_lmhead --name int4-g128-lmhead --launch --wait`
- Official a10g-small TPS/PPL confirmed → lawine posts terminal SENPAI-RESULT to PR #4
- Advisor merges PR #4 → updates ladder (int4 g128/lmhead weight-floor rung officially confirmed, ~127)
- lawine's weight-quant track is then complete → pivot lawine to a fresh frontier lever next round

---

## 2026-06-13 09:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) [IN PROGRESS — awaiting HF Job]

- **Branch:** `stark/int4-qat-w4a16`
- **Student:** stark
- **Status:** WIP — local evidence complete; awaiting human approval of HF Job (GitHub issue #11)
  before posting terminal SENPAI-RESULT with official a10g-small numbers.
- **Hypothesis:** Stock vLLM 0.22.0 Marlin int4 W4A16 endpoint on `google/gemma-4-E4B-it-qat-w4a16-ct`
  reproduces the ~95.4 TPS / PPL ~2.01 VALID leader. The dominant lever: int4 weight quantization
  reduces bandwidth by ~4×, lifting TPS from 44 → ~95 with better PPL (QAT-trained).

### Local Results (exploratory, A10G — NOT official a10g-small)

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (128/128 GT records) | **2.0055** | ≤ 2.42 | ✓ |
| Local TPS (exploratory, A10G, 32 prompts) | **95.99** | — | on target ~95.4 |
| Marlin kernel | `MarlinLinearKernel for CompressedTensorsWNA16` | — | ✓ confirmed |
| All modalities | vision/audio encoder cache initialized | — | ✓ |
| CUDA graphs | `FULL_AND_PIECEWISE`, no eager fallback | — | ✓ |
| Peak GPU memory | ~21.1 GiB / 23 GiB | — | no OOM |
| W&B run | none (serving task, no training) | — | — |

### Key Finding — Greedy-Identity Nondeterminism

stark discovered that the int4+vLLM endpoint is **run-to-run nondeterministic** for greedy decode
at output_len=512: Marlin split-K GEMM / Triton-attn FP non-associativity introduces ~1 ULP noise
at near-tie logit positions, cascading to token-flip divergences at a handful of hotspots (idx 83,
104 consistently). Cross-path comparison (HF bf16 dense GEMM vs vLLM Marlin int4) always diverges
— different arithmetic paths.

**Advisor ruling:** NOT a blocker. The as-is stock int4 Marlin leader (~95.4 TPS, same stack) is
VALID on the official leaderboard. This submission IS that stack. Within-stack greedy identity
(same vLLM endpoint, same job run) is consistent; the official harness compares decode_outputs.jsonl
generated from the same serving instance. Determinism study deferred — not needed for this rung.

### Next Steps

- Human approves GitHub issue #11 → stark runs `python train.py --submission submissions/int4_qat --name int4-qat --launch --wait`
- Official a10g-small TPS/PPL confirmed → stark posts terminal SENPAI-RESULT to PR #3
- Advisor merges PR #3 → updates ladder (int4 rung officially confirmed)

---

## 2026-06-13 08:40 — PR #2: Resolve PPL artifact path + validate bf16 baseline locally

- **Branch:** `fern/vllm-baseline-ppl-resolution`
- **Student:** fern
- **Hypothesis:** Before spending HF Jobs quota on speed work, definitively explain why the prior
  bf16 smoke job (`6a2c5fb77c68f455eff14260`) produced `tps=44.018` but no confirmed
  `ppl_summary.json`. Prove the PPL and decode contracts against a local endpoint, deliver a
  reusable one-command local pre-validation harness, and confirm the `MAX_NUM_BATCHED_TOKENS=512`
  OOM-safety hypothesis on the longest GT context (2431 tokens). Research priority #1.

### Results

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (128/128 GT records) | **2.3012** | ≤ 2.42 | ✓ |
| GT records completed | 128/128 | 128/128 | ✓ |
| PPL contract (`prompt_logprobs` on integer-ID prompt) | proven | — | ✓ |
| Decode contract (`choices[0].token_ids` len 512) | proven | — | ✓ |
| OOM safety (longest ctx=2431 tokens at `MAX_NUM_BATCHED_TOKENS=512`) | +560 MiB transient (< 0.5 GiB budget) | no OOM | ✓ |
| Root cause of missing artifact | 40-min HF Job timeout | — | identified |
| W&B run | none (local validation task) | — | — |

### Root Cause — Definitive

The 40-min HF Job wall-clock cap killed the job before PPL ever started. Timeline:

| stage | duration | cumulative | status |
|---|---|---|---|
| Cold startup (model load + torch.compile + CUDA-graph capture) | 11.9 min | 11.9 min | completed |
| Benchmark stage (128 prompts, decode, tps measurement) | 24.8 min | 36.7 min | completed |
| Decode capture (same 128×512 workload) | ~24.8 min est. | 61.5 min | **killed @ 40 min** |
| PPL stage (runs *after* decode) | n/a | n/a | **never reached** |

Evidence from preserved artifacts (`research/local_validation/prior_job_6a2c5fb77c68f455eff14260/`):
- `job_status.json` → `status:timed_out`, `stage:RUNNING`, `timeout_minutes:40` → rules out OOM (clean wall-clock stop)
- `run_environment.json` → `ppl.enabled:true` → rules out disabled
- `summary.json` → `duration_s:1488.8` (benchmark alone = 24.8 min) → rules out unfetched

**Implication:** at 44 TPS the bf16 baseline cannot fit startup+benchmark+decode+PPL in 40 min. All
faster submissions (≥95 TPS) will fit comfortably. The local harness (below) provides a timeout-free
gate.

### OOM-Safety Confirmation

Longest GT record (`gpqa_diamond-1d37a7a51d`, ctx=2431, tgt=512, combined=2943 tokens): HTTP 200 +
valid `prompt_logprobs` (len 2943). Peak GPU: 21009 MiB (+560 MiB transient). Theoretical chunked
bound: 512 positions × 262,144 vocab × 4B = 0.50 GiB. Confirms `MAX_NUM_BATCHED_TOKENS=512`
chunked prefill bounds the `log_softmax` peak as predicted in DATASET_ANALYSIS.md.

### New Shared Infrastructure

`scripts/local_prevalidate.py` — one-command local pre-validation gate:
```bash
cd target/ && VLLM_USE_FLASHINFER_SAMPLER=0 \
  python scripts/local_prevalidate.py --submission submissions/vllm_baseline --decode-num-prompts 16
# → SENPAI-LOCAL tps=44.0056 ppl=2.3012 completed=128
```

**All students should run this against their submission before opening an HF Job approval issue.**

### Local-Environment Note

FlashInfer JIT is broken on this node (CUDA 13.2 nvcc vs. vendored libcudacxx). Workaround:
`VLLM_USE_FLASHINFER_SAMPLER=0`. Numerically identical for greedy decode (argmax) and PPL
(logits/log_softmax). Not needed on official a10g-small image.

### Analysis & Conclusions

**Verdict: merge (infra + priority-1 resolution).** Not a TPS improvement but delivers essential
shared infrastructure and closes the highest-priority uncertainty blocking all future submissions.

- The bf16 baseline is correct: PPL ≈ 2.30 exactly matches the reference. The prior smoke job was
  not defective — it just ran out of time.
- The local pre-validation harness (`scripts/local_prevalidate.py`) is now a team-wide gate. Every
  student should PPL-validate locally before requesting an HF Job.
- The OOM-safety analysis confirms DATASET_ANALYSIS.md's `MAX_NUM_BATCHED_TOKENS=512` recipe is
  correct; the longest GT context (2431 tokens) fits within the GPU memory budget.
- The 40-min timeout root cause is important baseline knowledge: the benchmark + decode stages
  together consume ~24.8 + 24.8 = ~49.6 min at 44 TPS, plus ~12 min cold startup ≈ 61.5 min
  total. Any future a10g-small bf16 confirmation needs the timeout cap raised, or the decode
  prompt count reduced. Fast submissions (≥95 TPS) automatically fit in 40 min.

### Suggested follow-up (fern's own note, endorsed)

- Wire `local_prevalidate.py` into the pre-submission checklist (all students: run it locally;
  only request an HF Job once it passes). ← **Done — see "New Shared Infrastructure" above.**
- For an a10g-small bf16 confirmation, fern will open a separate `Approval request: HF job for
  vllm-baseline` issue — not done in this PR (local-only by instruction).

_PR #2 merged to `approval-gated-8gpu-20260613` as squash commit `dd17c17`._
