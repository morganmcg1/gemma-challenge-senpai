# PR #331 — Does the EAGLE-3 E[T]=6.11 tree stay sub-cliff, or does the verify-GEMM M=33 tile jump bind fern #325 earlier?

**PRIMARY `eagle3_verify_gemm_mcliff_self_test_passes` = True** (all 9 conditions a–i)
**TEST `compliant_tps_worstcase_with_mcliff` = 492.87** (sub-cliff; the cliff is inert, so this == fern #325's banked worst-case)
**W&B `mllu8p23`** (group `eagle3-mcliff`) · LOCAL read-only analytic, 0 GPU, 0 TPS

> **Verdict: SUB-CLIFF-SAFE.** The E[T]=6.11 EAGLE-3 top-4 salvage tree verifies at **M = W·K+1 =
> 4·7+1 = 29** tokens — **2 Marlin `thread_m_blocks` (ceil(29/16))**, the **same** GEMM tile as fern
> #325's M=32 anchor. M=29 ≤ knee_Mstar=32, so the verify-GEMM denominator is **unchanged**, fern
> #325's compliant-500 **YELLOW envelope STANDS** (central capped 520.95, worst 492.87), and the
> **M=33 tile cliff does NOT bind earlier.** It is **tight**: E[T]=6.11 is simultaneously the
> depth-7 top-4 acceptance ceiling **and** the M=32 tile knee — any push to E[T]>6.11 needs depth-8
> (M=33) or top-5 breadth (M=36), and **both cross the cliff** → RED.

## 1. The mapping: accepted tree width → verify-GEMM M (deliverable 1)

The Marlin int4 W4A16 verify GEMM is weight-bandwidth-bound and quantizes the token dimension M into
tiles of `thread_m_blocks = ceil(M/16)`: throughput is **flat inside a tile** and **cliffs** at each
16-boundary. A speculative draft tree of depth K with breadth W (candidates retained per depth)
presents

> **M = W·K + 1** (root + W candidate siblings at each of K depths)

tokens to the single tree-verify forward pass. Two corpus anchors pin the formula exactly:

| config | W | K | M = W·K+1 | `ceil(M/16)` | sub-cliff (≤32)? | anchor |
|---|---|---|---|---|---|---|
| deployed linear MTP chain | 1 | 7 | **8** | 1 block | ✅ | `m_verify_deployed=8` (ubel #311) — **exact** |
| EAGLE-3 top-4 salvage tree (E[T]=6.11) | 4 | 7 | **29** | 2 blocks | ✅ | lawine #101 size-29 tree — **exact** |

**Why W=4:** the per-depth effective acceptance that reaches E[T]=6.11 over the deployed K=7 spine is
the **top-4 salvage** `c_eff = a₁ + (1−a₁)·cov4 = 0.7731 + 0.2269·0.6532 = 0.9213` (denken #304
`a1_required_611` × wirbel #79 `cov4`). Realizing top-4 rank-coverage salvage **requires carrying
W=4 candidate siblings per depth**, so M = 4·7+1 = **29** — **3 nodes of growth headroom** (M=29→32
stays sub-cliff; a 4th node lands on M=33 and crosses).

**Capture-safety is orthogonal.** ubel #311's `max_safe_tree_width=16` and the cudagraph buckets
{8,16,32} are a separate **correctness** axis (lawine #101 IndexError), not throughput. A 29-node
tree pads up to the **M=32 capture bucket** — still 2 Marlin blocks, **still sub-cliff**. The
throughput cliff priced here (M=33, 3 blocks) is reached only by a tree of **≥33 real nodes**.

## 2. The cliff cost if crossed (deliverable 2)

Directly measured (A10G, int4 W4A16, ctx=256; `research/spec_cost_model/results_tile_boundary.json`,
`knee_Mstar=32`):

| transition | step | forward (GEMM) | jump |
|---|---|---|---|
| M=32 → M=33 (2→3 block) | 12.812 → 14.987 ms (**×1.16981, +16.98%**) | 9.566 → 11.731 ms (+22.64%) | +2.176 ms (== corpus `marginal_ms_per_token[33]`) |
| M=48 → M=49 (3→4 block) | 15.265 → 18.134 ms (×1.1879, +18.79%) | — | +2.869 ms (second cliff) |

Crossing to M≥33 scales every verify step by **μ=1.16981**, so fern #325's ledger denominator grows
by μ and the **whole envelope divides by μ** — both the E[T]-lever public TPS **and** the 520.95
identity-kernel ceiling (that ceiling is itself banked at the deployed step):

| corner | sub-cliff (banked) | **if crossed (÷μ)** | vs 500 |
|---|---|---|---|
| central | 586.08 → capped 520.95 | min(501.0, 445.33) = **445.33** | **−10.93%** (RED) |
| worst | 492.87 | 492.87/μ = **421.32** | **−15.74%** (RED) |

So crossing the cliff would **flip fern's YELLOW envelope to RED** — the tile cliff would bind
**before** the private tax does.

## 3. Max sub-cliff width — is E[T]=6.11 reachable inside it? (deliverable 3)

- **Largest breadth at K=7:** W=4 (M=29) fits; **W=5 → M=36 crosses**.
- **Largest depth at W=4:** K=7 (M=29) fits; **K=8 → M=33 crosses** (lands exactly on the cliff).
- **E[T] ceiling the sub-cliff imposes:** the W=4/K=7 tree's uniform-`c_eff` ceiling is
  `1 + Σ_{d=1..7} 0.9213^d = 6.1112` — i.e. **E[T]=6.11 IS the depth-7 top-4 ceiling**, exactly
  reachable sub-cliff. But it sits **at the frontier**: raising E[T] above 6.11 needs more depth
  (K=8 → M=33) or more breadth (W=5 → M=36), and **both cross**. The acceptance ceiling and the
  tile knee **coincide at 6.11**.

## 4. Verdict: SUB-CLIFF-SAFE (deliverable 4)

M=29 ≤ knee_Mstar=32 ⇒ the E[T]=6.11 tree verifies on the **same 2-block tile** as fern #325's M=32
anchor. fern's `step_deployed` denominator is **correct** for this tree, the **YELLOW envelope
stands** (central 520.95, worst 492.87), and the M=33 cliff **does not bind earlier**. The
counterfactual cliff-crossed envelope (central 445.33, worst 421.32, both RED) is what **would** bind
**only if** the tree were pushed past M=32 — which is also exactly what is needed to push E[T] past
the depth-7 ceiling of 6.11. **E[T]=6.11 sits at the joint frontier of the depth-7 acceptance
ceiling and the M=32 tile knee: the cheapest move past either crosses both at once.**

## 5. Anchor cross-check + self-test (NaN-clean)

All 9 PRIMARY conditions pass (`max_abs_err ≤ 1e-6`):
**a** M=W·K+1 reproduces deployed M=8 (W=1,K=7) ✓ · **b** top-4 tree M=29 == size-29 corpus, 2
blocks, sub-cliff ✓ · **c** salvage identity `a₁+(1−a₁)·cov4` == denken #304 `a1_required_611`
(pins W=4) ✓ · **d** cliff μ=1.16981 from measured tile, jump 2.176ms == corpus marginal ✓ · **e**
crossed envelope ÷μ → both corners RED ✓ · **f** sub-cliff envelope == fern #325 banked (cap binds
central, worst 492.87) ✓ · **g** max sub-cliff W=4/K=7, W=5 & K=8 cross, E[T]=6.11 reachable ✓ ·
**h** verdict SUB-CLIFF-SAFE, fern stands, cliff would flip YELLOW→RED, frontier tight ✓ · **i**
NaN-clean ✓ → **PRIMARY PASS**.

## 6. Honest caveats (carried in the artifact)

1. **DERIVED, not measured:** no EAGLE-3 checkpoint exists (training-gated). The W=4 → M=29 mapping
   prices the tree the top-4 salvage acceptance model **implies**, not a running `EagleProposer`.
2. **cov-transfer YELLOW inherited:** `cov4=0.6532` is measured on the deployed **linear** spine
   (wirbel #79). A {2,21,39}-fusion draft could lower cov4 and **raise** the W needed to reach 6.11
   — denken #320's cov-transfer YELLOW is orthogonal to this **step-regime** card and unchanged.
3. **`step_deployed` tile basis:** fern #325 labels its denominator the "M=32 step" (2-block plateau
   top); the normalized 1.2182 unit traces (denken #278) to the deployed **M=8** verify (1 block).
   Either reading is sub-cliff, and the M=29 tree shares the 2-block tile with the M=32 anchor — so
   the only cliff above is M=33. The headline does not depend on which reading is taken.
4. **Capture-safety is a separate axis:** ubel #311 (`max_safe_tree_width=16`, lawine #101
   IndexError) is correctness, not throughput; a 29-node tree pads to the M=32 bucket, still
   sub-cliff. This card prices the verify-GEMM **throughput** cliff (M=33).
5. **0 TPS / denominator-correctness property:** depends only on integer node-counts and the measured
   tile boundary, not tensor values. **NOT a launch / build / served-file change / HF Job.**

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT
a launch, NOT a build. BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**; greedy decode and
PPL untouched.

## Hand-off

The EAGLE-3 E[T]=6.11 build verifies its top-4 salvage tree at **M=29 ≤ knee_Mstar=32** — **2 Marlin
blocks, sub-cliff**. fern #325's compliant-500 envelope is priced on the **correct** verify-GEMM
denominator; its **YELLOW stands** (central capped 520.95, worst 492.87) and the M=33 tile cliff does
**not** move the binding constraint. The human GO/NO-GO can treat the verify-GEMM denominator as
**not the binding term at E[T]=6.11** — but should note the **tight joint frontier**: any E[T]>6.11
(via depth-8 or top-5 breadth) crosses M=33 and divides the envelope by 1.16981 → RED. The single
cheapest YELLOW→GREEN move remains denken #319's per-depth private-α read (fern #325), unchanged by
this card; this card removes the M-cliff as a competing un-priced denominator risk.
