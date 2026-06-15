# Drafter-fusion build scope (PR #424) — the human-facing GO/NO-GO

**Verdict: NO-GO (do not build).** W&B `4ea99xzw` (group `drafter-fusion-scope`,
`analysis_only=True`, `no_hf_job=True`, `no_served_file_change=True`,
`official_tps=0`, self-test 13/13). Baseline UNCHANGED 481.53.

Built strictly from merged artifacts on `approval-gated-8gpu-20260613` + the
deployed served submission `submissions/fa2sw_precache_kenyan`.

---

## TL;DR for the human

The card asks to validate a "~+52 TPS roofline" from fusing the deployed MTP K=7
drafter's 7 heads toward a 248 µs bandwidth floor, and to scope the build. The
honest answer, from the merged record and the served code:

1. **The premise is already false.** The K=7 drafter does **not** run as "separate
   per-head launches." It is already a **single ONEGRAPH CUDA-graph replay**
   (`sitecustomize.py:25-33`, default `ONEGRAPH=1`). The per-head launch overhead
   is already harvested: **eager 2859 µs → graph 566 µs** (PR #75 `uknpbk94`; PR
   #261 `egaz6m2f` measured the draft launch tax as `realizable_bound_pct≈0`).
2. **The honest roofline is ~+20 TPS, not +52.** The 248 µs floor is the *GEMM-
   chain* BW floor (566 µs @ 47.17 % HBM peak). Even granting the floor, only the
   566 µs GEMM chain can move → **+20.0 TPS** roofline (merged 267 µs floor →
   +18.8). **+52 TPS (= +10.8 %) needs 781 µs of drafter saving — more than the
   entire 566 µs GEMM chain exists** — so +52 silently consumes ~215 µs of the
   ~879 µs non-GEMM mass that PR #75 shows GEMM fusion cannot touch. +52 even
   exceeds the hard ceiling of *deleting the whole GEMM chain* (+36.6 TPS).
3. **Realistic recovery ≈ 0** (band [0, +5], optimistic cap +11.4). The launch
   floor is harvested; the BW floor is M=1-unreachable (occupancy-bound,
   autoregressive-serial — PR #75/#269); and the one separable micro-lever
   (#269 GeluAndMul fold, +4.4 % *composition* ceiling) **measured negative
   realization** in #273 (`realization_ratio=-2.02`).
4. **Blast radius is HIGH for ≤+5 TPS.** The ONEGRAPH loopgraph *is* the engine of
   481.53; rewriting it risks the measured **−16.5 % regression to ~402 TPS**
   (PR #312/#315) and reopens all four gates.

**Recommendation:** do not request a served-kernel build for this lever. If any
drafter-side time is worth a build, it is the verify-SDPA `num_stages` tune
(#270, +1.097×, bit-identical) on the *verify* side — not drafter head-fusion.

---

## Note on the card's cited artifacts (premise/citation correction)

The card cites "#295/#301/#307/#311/#312/#313" as the cost basis. Those are the
**EAGLE-3 layer-fusion** drafter cluster — a *different drafter architecture*
(fuse {2,21,39} hidden states; full retrain; a₁ 0.73→0.92), not "fuse the 7 MTP
heads of the deployed drafter." Their "fusion" ≠ this card's head-fusion. The
correct cost basis for the **deployed MTP K=7** drafter is the drafter-roofline
cluster: **#75** (`uknpbk94`, drafter-forward roofline), **#284** (`u58fxtu6`,
measured step decomposition), **#261** (`egaz6m2f`, draft launch already
captured), **#269/#270** (draft MLP/attn intrinsic-M=1), **#289/#392**
(acceptance ladder E[T]=3.851). This memo is built on those.

---

## 1. Pinned current drafter cost (merged)

| quantity | value | source |
|---|--:|---|
| deployed per-step decode wall | **8017 µs** | #284 `u58fxtu6` |
| ├─ verify forward | 6532 µs (81.4 %) | #284 |
| ├─ **drafter forward (full, in-stack)** | **1445 µs (18.0 %)** | #284 |
| └─ host/serving residual | 40 µs (0.50 %) | #284 |
| drafter **GEMM/GEMV chain** (7-pass, ONEGRAPH) | **566 µs** | #75 `uknpbk94` |
| drafter **non-GEMM** (sampler + 262k embed-gather + SDPA + sampling) | **879 µs (~61 %)** | #75 (≈70 % of drafter) |
| GEMM chain HBM utilisation @ M=1×K=7 | **47.17 %** | #75 |
| drafter chain **eager** (no ONEGRAPH) | 2859 µs | #75 |
| per-head launch overhead recovered by ONEGRAPH | **2293 µs, already harvested** | #75 (2859→566) |

The drafter is the #2 decode block, but its launch overhead is already gone and
its GEMVs are occupancy-floored at M=1.

## 2. Roofline (validate the ~+52)

TPS scales as `1/step` (tokens/step E[T]=3.851 fixed). On the realizable strict
stack (482.74, step 7977 µs):

| reading | drafter saving | TPS Δ |
|---|--:|--:|
| **GEMM chain → 248 µs floor (advisor's floor)** | 318 µs | **+20.0** ← PRIMARY |
| GEMM chain → 267 µs floor (merged 47.17 %) | 299 µs | +18.8 |
| hard ceiling: delete the **entire** GEMM chain (566→0) | 566 µs | +36.6 |
| **advisor's quoted roofline** | — | **+52.0** |
| (wrong) full drafter → 248 µs (collapses non-GEMM too) | 1197 µs | +84.5 |

**+52 needs 781 µs of saving — 215 µs more than the entire 566 µs GEMM chain.**
It sits between "delete the whole GEMM chain" (+36.6) and "collapse the full
forward to a GEMM-only floor" (+84.5), i.e. it requires removing time that PR #75
shows is non-GEMM and not GEMM-fusable. **The honest roofline is +20.0 TPS.**

## 3. Realistic recovery band

`drafter_fusion_realistic_tps = +1.5` (band **[0, +5]**, optimistic cap +11.4).
- Launch-fusion component: **0** (ONEGRAPH already harvested it — #75/#261/#246).
- BW-saturation component: **unreachable at M=1** — single-warp GEMVs hit 41-47 %
  HBM, "physically unreachable at M=1 without M≥16 batching" (#269), and the 7
  passes are **autoregressive-serial** so cannot be batched into one BW-saturating
  GEMM with unchanged outputs (#75: "INFEASIBLE with unchanged outputs").
- Separable micro-lever (GeluAndMul epilogue fold, #269 +4.4 % composition
  ceiling) **regressed on wall-clock realization** (#273 `realization_ratio
  =-2.02`) — so even the +5 cap is not assured and can realize ≤0.

## 4. Identity-verify cost (regimes A vs B)

- **Regime A — bit-identical fusion.** `fusion_identity_preserving=True`: identical
  proposals → acceptance unchanged → identity trivially preserved (and OUTPUT
  identity is anyway guaranteed by the M=8 verify gate regardless of drafter
  numerics). **But the only bit-identical fusion with positive speed headroom is
  ONEGRAPH, already deployed** → `regime_A_speed_headroom=False`.
- **Regime B — numerically-different-but-faster.** Not an identity effect (verify
  still emits M=1-exact tokens), but a TPS effect via acceptance. Sensitivity:
  `dTPS/dE[T] = 125.4 TPS` per unit E[T]; a **−0.04 E[T]** acceptance regression
  (`et_regression_that_wipes_realistic=0.0399`) erases the entire realistic speed
  band. No mechanism makes a faster-but-different drafter accept *better*
  (`fusion_acceptance_rate_delta` band [−0.10, +0.05], central 0,
  `regime_B_net_can_be_negative=True`). **Regime B is not worth the freedom;
  Regime A is the safe target — and it has ~0 headroom.**
- **Verify cost:** `fusion_identity_verify_cost_tps=0.0` — verification is a
  build-time CI gate (one served-vs-served greedy-identity pass + PPL≤2.42 over
  128 prompts, per BASELINE.md), not a served-runtime cost.

## 5. Build surface + blast radius

- **`served_kernel_surface`:** `submissions/fa2sw_precache_kenyan/sitecustomize.py`
  (ONEGRAPH loopgraph: `_run_graph_body`/`_capture_graph`/`_is_loopgraph_eligible`,
  keyed to `Gemma4Proposer` on `vllm.v1.spec_decode.gemma4` +
  `gemma4_mtp.get_top_tokens`) + a NEW fused CUDA/Triton kernel for the 7
  width-1 `Gemma4MTP` sub-forwards (q/o-proj GEMVs + 4-layer 256-dim gated MLP +
  centroid sampler + 262k masked-embed gather). **Served-file change ⇒
  human-approval-gated.**
- **`fusion_build_blast_radius`: HIGH.** (1) the ONEGRAPH loopgraph *is* the
  481.53 engine — a rewrite risks the measured **−16.5 % → ~402 TPS** regression
  (#312/#315, ~80 TPS debt) if capture breaks/goes inert; (2) reopens all 4 gates
  (greedy-identity, PPL≤2.42, boot-500, TPS) + the #272 boot-500 guard co-edit;
  (3) a 7-pass mega-kernel risks register-pressure/occupancy *regressions*;
  (4) the autoregressive serial dependency caps payoff at ~0 (occupancy-bound at
  M=1). High risk to the flagship for ≤+5 realistic TPS.

## 6. Stack projection

`stack_tps_if_realistic_recovery = 482.74 + 1.5 = 484.24` (band [482.74, 487.7],
optimistic cap ~494). The lever does not move the strict stack toward 500.

---

### Self-tests (13/13)
step decomposition sums; GEMM⊂full-drafter; non-GEMM>0; BW-floor<chain;
roofline<hard-ceiling; +52>hard-ceiling; +52 requires non-GEMM; +52 overdraws the
GEMM chain; realistic-high<roofline; launch already harvested (eager>4×graph);
dTPS/dE[T]>0; stack=base+realistic; E[T]-wipe<0.06.
