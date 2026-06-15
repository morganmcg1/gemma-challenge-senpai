STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["4ea99xzw"],"primary_metric":{"name":"drafter_fusion_roofline_tps","value":20.04},"test_metric":{"name":"drafter_fusion_self_test_passes","value":1}}

## Results — Drafter-fusion build scope: **NO-GO** (do not build)

**The ~+52 TPS roofline does not survive the merged record. Honest roofline ~+20 TPS, realistic ~0–5, blast radius HIGH.** This is a 0-TPS analysis card (no served-file change, no HF job, baseline UNCHANGED 481.53).

W&B run `4ea99xzw` (group `drafter-fusion-scope`, `analysis_only=True`, `no_hf_job=True`, `no_served_file_change=True`, `official_tps=0`, self-test **13/13**).
Command: `python research/drafter_fusion_scope/scope_analysis.py` (CPU-only analytic; 0 GPU, negligible RAM).
Public context: leaderboard top now 514.2 / 508.6 TPS (both > our 481.53) — the 500-push matters, but drafter head-fusion is not the route (below).

### ⚠️ Two premise/citation corrections (decision-critical — please read before flagging to the human)

**(1) The premise "[MTP K=7] runs as separate per-head launches — launch/sync/overhead-bound" is false against the deployed submission.** The served drafter is **already a single ONEGRAPH CUDA-graph replay** of the 7 width-1 iterations (`submissions/fa2sw_precache_kenyan/sitecustomize.py:25-33`, default `ONEGRAPH=1`; `_capture_graph`/`_run_graph_body`). The per-head launch overhead is **already harvested: eager 2859 µs → graph 566 µs** (PR #75 `uknpbk94`), and my own #261 (`egaz6m2f`) already measured the draft launch tax as `realizable_bound_pct≈0`.

**(2) The card's cited cost basis (#295/#301/#307/#311/#312/#313) is the EAGLE-3 *layer-fusion* cluster — a different drafter, a different "fusion."** Those price swapping in a {2,21,39}-hidden-state EAGLE-3 draft (full retrain, a₁ 0.73→0.92), not fusing the 7 MTP heads of the deployed drafter. I rebuilt this scope on the correct basis: **#75** (drafter-forward roofline), **#284** `u58fxtu6` (measured step decomposition), **#261** (draft launch already captured), **#269/#270** (draft MLP/attn intrinsic-M=1), **#289/#392** (E[T]=3.851).

### Deliverables (W&B summary fields)

| field | value | note |
|---|--:|---|
| **`drafter_fusion_roofline_tps`** (PRIMARY) | **+20.04** | GEMM chain → advisor's 248 µs floor, strict stack (merged 267 µs floor → +18.8). **Refutes +52.** |
| `drafter_fusion_realistic_tps` | **+1.5** (band [0, +5], cap +11.4) | launch already harvested; M=1 BW floor unreachable |
| `drafter_forward_current_us` / `_floor_us` | 1445 / 248 | full in-stack drafter / GEMM-chain BW floor |
| `drafter_gemm_chain_us` / `_nongemm_us` | 566 / 879 | only the 566 µs GEMM chain can reach a *GEMM* floor |
| `fusion_identity_preserving` | **True** (but `regime_A_speed_headroom=False`) | the only bit-identical fusion *is* ONEGRAPH, already deployed |
| `fusion_acceptance_rate_delta` | 0.0 (band [−0.10, +0.05]) | `regime_B_net_can_be_negative=True` |
| `fusion_identity_verify_cost_tps` | 0.0 | offline build-time CI gate, not a runtime cost |
| `stack_tps_if_realistic_recovery` | **484.24** | 482.74 + 1.5; does not approach 500 |
| `drafter_fusion_self_test_passes` | **True (13/13)** | |

### Why +52 is physically impossible on the deployed MTP K=7 drafter

Deployed step (PR #284 `u58fxtu6`): **8017 µs = verify 6532 + drafter 1445 + host 40**. The drafter splits into a **566 µs GEMM chain** (47.17 % HBM peak, #75) + **879 µs non-GEMM** (sampler + 262k embed-gather + SDPA + sampling).

- TPS scales `1/step` (E[T]=3.851 fixed). **+52 TPS = +10.8 % needs 781 µs of drafter saving** — but the entire GEMM chain is only **566 µs**. So +52 must remove **215 µs of non-GEMM mass that GEMM fusion cannot touch** (#75: non-GEMM is ~70 % of the drafter, untouched).
- +52 even exceeds the **hard ceiling of deleting the entire GEMM chain (566→0) = +36.6 TPS**.
- The roofline ladder: GEMM→248 floor **+20.0** · GEMM→267 floor +18.8 · delete-whole-GEMM +36.6 · (wrong) full-drafter→248 +84.5. **+52 sits in the impossible band between "delete the GEMM chain" and "collapse the full forward onto a GEMM-only floor."**

### Why realistic ≈ 0 (not just below roofline)

- **Launch-fusion = 0:** ONEGRAPH already harvested it (#75/#261/#246).
- **BW saturation unreachable at M=1:** single-warp GEMVs hit 41–47 % HBM, "physically unreachable at M=1 without M≥16 batching" (#269), and the 7 passes are **autoregressive-serial** → cannot batch into one BW-saturating GEMM with unchanged outputs (#75: "INFEASIBLE with unchanged outputs").
- **The one separable micro-lever regressed:** #269's GeluAndMul epilogue fold (+4.4 % *composition* ceiling) measured **negative wall-clock realization** in #273 (`realization_ratio=-2.02`).

### Identity (regimes A/B)

- **A — bit-identical fusion:** identity trivially preserved (and OUTPUT identity is M=8-verify-gated regardless), but the only bit-identical fusion with speed headroom is ONEGRAPH, already deployed → ~0 headroom.
- **B — numerically-different-but-faster:** a TPS effect, not identity. `dTPS/dE[T]=125.4`; a **−0.04 E[T]** acceptance regression wipes the whole realistic band, and nothing makes a faster-different drafter accept *better*. Regime B can net negative; Regime A is the safe (but ~0-payoff) target.

### Blast radius — HIGH for ≤+5 TPS

The ONEGRAPH loopgraph **is** the engine of 481.53. Rewriting `_run_graph_body`/`_capture_graph` risks the **measured −16.5 % regression to ~402 TPS** (#312/#315, ~80 TPS debt) if capture breaks/goes inert, reopens all 4 gates (greedy-identity, PPL≤2.42, boot-500, TPS) + the #272 boot-500 guard co-edit, and a 7-pass mega-kernel risks register-pressure/occupancy *regressions*.

### What happened — honest analysis

The lever's two interpretations both collapse: (i) "erase per-head launch overhead" is **already done** by ONEGRAPH; (ii) "saturate the GEMM bandwidth floor" is **M=1-unreachable** (occupancy-bound + autoregressive-serial). The ~+52 headline only arises by treating the full 1445 µs drafter as collapsible onto the 248 µs *GEMM-only* floor, which double-counts the ~879 µs non-GEMM mass. The defensible roofline is **+20 TPS** (a true ceiling), realistic recovery **~0–5**, and the build touches the exact code that produces 481.53. **Net EV is strongly negative: high flagship risk for ≤+5 realistic TPS. Do not request the build.**

### Suggested follow-ups

1. **If any drafter-adjacent build is worth it, it's the verify-SDPA `num_stages 3→2` tune (#270, +1.097× at M=8, bit-identical maxdiff=0.0)** on the *verify* side (batch=8 linear path) — far cheaper surface than rewriting the drafter loopgraph, and not premise-broken. (wirbel #279 was already measuring its linear-deploy transfer.)
2. **Retire the "fuse the MTP heads for the BW floor" lane** the way #261/#246/#251/#255 retired the other already-captured launch-tax lanes — it is the same already-harvested-by-ONEGRAPH pattern.
3. If the goal is the drafter's E[T] numerator (the real 500 lever), that is the **EAGLE-3 build** lane (#293/#295/#304/#312), which is a separate, retrain-gated, regression-risky decision — not in-place head fusion.
