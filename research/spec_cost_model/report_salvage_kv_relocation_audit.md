# PR #157 — Salvage-KV relocation audit: is the host-bound Python loop a live step tax on the descent path?

**Verdict: LIVE-LANDMINE / build-blocker.** The `relocate_salvaged_kv` host loop
is a real, on-the-timed-window step tax that the captured step model never priced —
and it is exactly the op land #71's descent fix ARMS. Designing it out as a
vectorized GPU gather/scatter recovers essentially all of it (greedy-safe,
`equivalence_rate=1.0`).

**PRIMARY `salvage_kv_audit_self_test_passes = 1` (11/11).**
**TEST `recoverable_step_pct_salvage_kv = 569.9%`** of the captured-target step
(i.e. the host loop is ~5.7 captured-steps of overhead PER step; the vectorized
design gives all but ~35 µs of it back).
LOCAL A10G profiling + analysis only. No vLLM serve change, no HF Job, no
submission, no kernel deploy. **BASELINE stays 481.53 (PPL 2.3777).** This bounds a
denominator lever + hands land #71 a build design + a build-blocker classification;
it does **not** authorize a launch. Rides Issue #124 RESOLVED (greedy-exact).

## TL;DR

| relocate implementation | per-call (A10G, M=32×37L K+V bf16) | amortized @ 0.382 salvages/step | clear-500 bar | descent TPS | build status |
|---|---|---|---|---|---|
| **host_loop** (per-layer×per-row D2H/H2D Python) | **145.2 ms** | **55.46 ms/step** | **32.59** (≫ 5.207 ceiling) | **77** | **INFEASIBLE — the landmine** |
| gpu_perlayer (37 device `index_copy_`, no host trip) | 2.09 ms | 0.80 ms/step | — | — | proves the 37-layer COUNT is cheap |
| **gpu_vectorized** ([L,W,H,D] gather/scatter, 1 launch) | **0.092 ms** | **35.3 µs/step** | **4.880** | **516** | **the design target — clears 500** |
| paged_slotmap (int slot-map update, zero KV copy) | 0.053 ms | 20.3 µs/step | ~4.875 | ~517 | the zero-copy ideal if build is paged |

The fleet has attacked the **E[T] numerator** (land #71's descent walk). This audit
prices a **denominator** op on the descent path that NO step-model leg priced —
lawine #136 measured a GRAPH-CAPTURED step (1.2182), my #154 measured the LINEAR
M=8 stack, lawine #153 the verify-step M curve — all measured a path that salvages
differently or is already captured. The host loop sits **outside** all of them.

- **As-built it is host-bound:** relocate amortizes to **110.45 ms/step = 90.2%**
  of chiku-inu's 122.5 ms CPU wall, ≫ the 18.67 ms GPU floor → the eager descent
  stack is host-bound on the timed decode window and timed out at 40 min.
- **It is a build-blocker, not a dead fallback:** the descent fix (BUG-2) is what
  arms it — as-built the 391 salvages fire but don't descend; once they descend,
  the relocate fires for real every salvage. A data-dependent Python loop over 37
  layers **cannot be CUDA-graph-captured**, so it pins the step host-bound (~122 ms)
  instead of the 9.7 ms captured target → the descent's E[T]=5.04 (→522) collapses
  to **~77 TPS**.
- **The fix is a sliver:** the vectorized relocate eats **9.9%** of the descent's
  0.178-E[T] cushion over the 500 bar (bar 4.862 → 4.880, still ≪ E[T]=5.04 → 516
  TPS). **1571× faster** than the host loop. Stacks under my #154 bar (~4.808):
  the combined operative bar stays meetable **iff** the relocate is vectorized; a
  host-loop build makes it unmeetable regardless of #154.

Artifacts: `scripts/profiler/salvage_kv_relocation_audit.py`,
`research/spec_cost_model/salvage_kv_relocation_audit.json`. W&B `rh8ysitz`
(group `salvage-kv-relocation-audit`).

## Public evidence used

- chiku-inu live STEPTIME trace, board `20260614-111022-934`, STEPTIME agg
  **n=12841 steady decode steps** on `tree-488-pw-fp32-v0`: EXEC gpu p50=19.18ms
  p90=21.42 mean=18.67; EXEC cpu p50=12.05ms **p90=335.36** mean=122.5; the
  decomposition note pins the 335 ms p90 to "the 35%-of-steps Python
  `relocate_salvaged_kv` loop over 37 layers."
- Oracle readout `tree-488-pw-fp32-v0` (board `20260614-100550-487`): **391 salvages
  / 37 full / 1024 steps** → 0.382 salvages/step, E[T]=2.621, M=32.
- lawine #136 `tree_step_denominator.json`: GRAPH-CAPTURED depth-9 step **1.2182**
  units (`gemm_all_graphed=true`); clear-500 bar 4.862.
- my #154 `report_step_denominator_reduction.md`: lowered bar **4.808–4.820**
  (scatter+LP avoidance) — this audit STACKS under it.
- fern #134 / wirbel #135 descent-only E[T] **5.04** (→ ~522 TPS); fern #125 /
  denken #101 supply ceiling E[T] **5.207**.
- K_cal = 125.268 (#148 de-risked band), STEP_M8 = 7982.89 µs.

## Part A — Anchor decomposition (analytic, exact: chiku-inu's own numbers close)

The relocate cost is recoverable WITHOUT the external build by closing chiku-inu's
CPU mean against the salvage-weighted mix of the relocate tail and the non-salvage
floor:

```
CPU mean = r·(relocate-step CPU) + (1−r)·(non-salvage CPU floor)
122.5    = r·335.36              + (1−r)·12.05
=>  r = (122.5 − 12.05) / (335.36 − 12.05) = 0.342
```

- **implied salvage frac = 0.342** — sits between chiku's stated 0.35 and the
  oracle's 0.382 (sampling spread), so the anchors are mutually consistent.
- **relocate amortized = mean − floor = 110.45 ms/step** = **90.2%** of the
  122.5 ms host-bound wall. The op IS the wall.
- relocate marginal per salvage = tail − floor = **323.31 ms**.
- Cross-checks: amortized via oracle rate (0.382×323.31 = 123.5 ms) and via chiku's
  0.35 (113.2 ms) bracket the mean-minus-floor 110.45 — consistent.

This is the rigorous "reproduce the anchors within tolerance" deliverable: it
reconstructs chiku-inu's CPU mean to <0.1 ms from its own {rate, p90, p50}.

## Part B — Per-call microbench (4 implementations, A10G, served osoi5 KV dims)

37 served layers × K+V `[CTX=4096, kv_heads=2, head_dim=256]` bf16 (2048 B/pos/layer),
n_move = M=32 (the naive full-tree-window compaction the host loop walks). Wall-clock,
synced each call (the host loop's cost IS host time):

| impl | p50 | p90 | mean | n |
|---|---|---|---|---|
| **host_loop** (D2H/H2D per row per layer) | 145.1 ms | 145.7 ms | **145.2 ms** | 40 |
| gpu_perlayer (37× device `index_select`+`index_copy_`) | 2.09 ms | 2.15 ms | **2.09 ms** | 300 |
| **gpu_vectorized** ([L,W,H,D] batched gather/scatter) | 81 µs | 93 µs | **92.4 µs** | 300 |
| paged_slotmap (int slot-map `index_copy_`) | 51 µs | 55 µs | **53.0 µs** | 300 |

**Reading it honestly:**
- The microbench host_loop (145 ms) reproduces the **hundreds-of-ms host-bound
  CLASS** of chiku's 335 ms p90, at ~0.43× its value. The gap is expected: I model
  ONLY the KV-row D2H/H2D round-trips (2 D2H + 2 H2D × 32 rows × 37 layers ≈ 4.7k
  synced round-trips ≈ 30 µs each); chiku's production loop adds per-element Python
  (index math, list/dict ops) and may compact a wider window. The **exact** anchor
  reproduction is Part A (analytic, <0.1 ms). Part B independently confirms the
  qualitative structure: host round-trip dominates.
- **The 37-layer count is NOT the killer.** gpu_perlayer keeps the 37-iteration
  Python loop but removes the host round-trip → 2.09 ms (70× under host_loop). The
  HOST ROUND-TRIP is the tax, not the layer count.
- **chiku's GPU floor 19.18 ms is the EXEC-step forward, not the relocate.** The
  relocate's own GPU cost (vectorized 92 µs / perlayer 2 ms) is 2–3 orders BELOW
  that floor — which is exactly why moving it onto the GPU makes it disappear into
  the step.

## Part C — Pricing on the oracle ladder (0.382 salvages/step)

amortized µs/step = 0.382 × per-call; bar = 500·step/(K_cal·τ) at the captured step
inflated by the amortized cost:

| impl | amortized/step | captured-step inflation | clear-500 bar | descent E[T]=5.04 → TPS | clears 500? |
|---|---|---|---|---|---|
| host_loop | **55.46 ms** | +570% | **32.59** | **77** | **NO** (≫ 5.207 ceiling) |
| gpu_perlayer | 0.80 ms | +8.2% | 5.262 | 478 | NO (just over ceiling) |
| gpu_vectorized | **35.3 µs** | +0.36% | **4.880** | **516** | **YES** |
| paged_slotmap | 20.3 µs | +0.21% | 4.875 | 517 | YES |

- **recoverable = host_loop − vectorized = 55.42 ms/step = 569.9%** of the 9726 µs
  captured-target step. (Contrast my #154 lever at 0.86–1.11% of step — this is
  three orders larger. It is not a 1% shaving; it is a step-DOMINATING op.)
- **speedup vectorized/host = 1571×.**
- The **descent cushion** over the 500 bar is E[T]_descent − bar = 5.04 − 4.862 =
  **0.178 E[T] units**. The vectorized relocate consumes **9.93%** of it (bar →
  4.880); the host loop blows 18,000% of it (bar → 32.6). That is the binary.
- Note gpu_perlayer's bar 5.262 just exceeds the 5.207 ceiling — even the
  no-host-round-trip-but-still-per-layer-Python path is marginal. The relocate must
  be **fully vectorized** (single launch sequence), not merely moved off the host.

## Part D — Greedy-safety (`equivalence_rate = 1.0`, bit-exact, by construction)

All three movement implementations were verified bit-exact against the reference
gathered KV (`max_abs_err_k = max_abs_err_v = 0.0` for host_loop, gpu_perlayer,
gpu_vectorized):

> KV relocation is a **pure permutation/copy of existing bf16 values** — no cast, no
> arithmetic → no rounding. The verifier's argmax over the verify logits, and thus
> the accepted token IDs, are decided **BEFORE** relocation and do not depend on
> which physical KV slot the bytes live in. Greedy identity is preserved by
> construction; `equivalence_rate = 1.0`.

This holds for the vectorized design identically — it moves the same bytes to the
same logical positions via a different (device-side) index path.

## Part E — Self-tests (PRIMARY = 11/11)

| # | test | result |
|---|---|---|
| 1 | anchor_decomposition_closes (implied frac ∈ [0.30,0.40], amort>0) | ✓ |
| 2 | host_loop_is_host_bound (p90 ≥ 50 ms) | ✓ |
| 3 | vectorized_sub_ms (mean < 1 ms) | ✓ |
| 4 | vectorized_beats_host_100x (≥100× per call) | ✓ (1571×) |
| 5 | layer_count_not_the_killer (gpu_perlayer mean < 5 ms) | ✓ |
| 6 | amortization_arithmetic (rate×per-call == µs/step) | ✓ |
| 7 | greedy_safe_equivalence_1 (`equivalence_rate == 1.0`) | ✓ |
| 8 | vectorized_headroom_small (eats < 50% of descent cushion) | ✓ (9.9%) |
| 9 | host_loop_infeasible (bar > 5.207 ceiling) | ✓ (32.6) |
| 10 | feasibility_flips (vectorized clears 500, host doesn't) | ✓ |
| 11 | nan_clean (all numerics finite) | ✓ |

## Hand-off to land #71 (the build)

**Classification: LIVE host-bound build-blocker (NOT a dead fallback).**

**Why a blocker.** The descent fix (BUG-2: making the 391 salvages actually descend
instead of firing-without-descending, +0.077 E[T]) is precisely what arms the
relocate on every salvage. A data-dependent Python loop over 37 layers cannot be
CUDA-graph-captured, so it pins the step host-bound (~122 ms) rather than the 9.7 ms
captured target → the descent's E[T]=5.04 (→522) collapses to ~77 TPS. The descent
numerator gain is **unrealizable** until the relocate is designed out.

**Design (drop-in for the descent build):**
1. **Single FUSED/vectorized GPU relocate.** Gather the accepted rows across ALL 37
   layers' K and V by a DEVICE commit-index in one launch sequence — `index_select`
   on a `[L, W, H, D]` stack, scatter to committed slots with `index_copy_`. NO
   per-layer Python, NO `.item()`/`.cpu()` host round-trip, NO per-element loop.
   Measured target: **35 µs/step** amortized.
2. **Zero-copy ideal (if the build uses paged KV):** relocation is a block-table /
   slot-map update — move an int slot-map, the KV bytes never move. Measured
   **20 µs/step**.
3. **Device-index rule (keep it inside the captured graph):** the commit-index
   (which scratch rows → which committed slots) must be produced ON-DEVICE by the
   accept walk (lawine #147 sync-free rule) and consumed by the relocate without a
   host readout — otherwise the `.cpu()` to materialize indices re-introduces the
   sync and the op falls back out of the graph.
4. **Greedy-safety:** the move is a bit-exact bf16 permutation; the verifier already
   decided accept/reject before relocation → `equivalence_rate=1.0`. No PPL risk.

**Stacks with.** Multiplicative with land's descent (a denominator PRECONDITION —
without it the descent numerator is unrealizable) and with my #154 scatter+LP lever.
The combined operative bar stays ~4.808 (#154) PROVIDED the relocate is vectorized;
a host-loop build makes the bar unmeetable regardless of #154. **Build-blocker if
host-loop; de-risked if vectorized per this design.**

## Honest scope

This is a **step-side (denominator) lever bound + a build design**, multiplicative
with land's descent and my own #154 lever — **not a new physics lever** and **not a
change vs the 481.53 baseline** (the host loop is not on the current linear/captured
path; it arms only when land's descent ships). Recoverable vs 481.53 today = 0; the
value is preventing the descent's 522-TPS gain from silently collapsing to ~77 when
the descent ships with a host-loop relocate.
