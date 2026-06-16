#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #506 — M=1 batch-invariant GEMV kernel vs vLLM Triton matmul_persistent.

DRAW-SAFE / LOCAL-ONLY pure microbenchmark on ONE pod A10G (sm_86). NO serve, NO HF
Job, NO submission, NO served-file change, NO --launch. analysis_only=true, official_tps=0.
Local measurement is pre-authorized (PR #506); this card writes ONLY research/ files.

## The lever this card tests (land's own #490 suggestion #1)

#490 (mtiq3ys8) established that at M=1 decode, vLLM's batch-invariant Triton
`matmul_persistent` kernel carries a large overhead on tiny-N shapes (k/v_proj N=512,
per_layer_input_gate N=256) because it computes a full BLOCK_M=128 MMA tile while only
ONE row is real (127/128 of the M-MMAs are wasted) and under-occupies the GPU (N=512 ->
only 4 of 84 SMs busy). cuBLAS-det is faster but NOT batch-invariant -> identity-unsafe.

A dedicated **fixed-order M=1 GEMV** can be BOTH fast AND byte-exact. The key construction:
take the *identical* `matmul_kernel_persistent` body but set **BLOCK_SIZE_M = 16** (the
sm_86 tensor-core m16n8k16 minimum) instead of 128, and KEEP **BLOCK_SIZE_K = 64**. The
per-output-element K-reduction is a sequential Python-level loop over k-tiles of size 64,
each a `tl.dot` lowering to `mma.sync.aligned.m16n8k16.f32.bf16.bf16`. Shrinking BLOCK_M
from 128->16 changes only how many m16-tiles a warp owns (8 -> 1); it does NOT change the
order in which a single output element accumulates its K-reduction. So row-0 output is
**bit-identical** to the BLOCK_M=128 kernel — provably batch-invariant — while doing 8x
fewer wasted M-MMAs. Finer BLOCK_N (128->64/32) raises SM occupancy (more N-tile programs)
without touching the per-element K-order (N-tiling only reassigns which program computes an
element). The danger zone is num_warps: each warp must own >=1 full m16n8k16 tile, i.e.
`num_warps <= (BLOCK_M/16) * (BLOCK_N/8)`, else Triton may reassign work non-bit-exactly.
We verify bit-exactness EMPIRICALLY per shape (the ground truth), the mechanism only tells
us which configs to test.

## What is measured (logged to W&B group m1-gemv-invariant-kernel)

For every one of the REAL gemma-4-E4B-it decode-GEMM shapes (model introspection, same 15
shapes as #490), BF16, M=1:
  - reference (c): vLLM Triton `matmul_persistent` BLOCK_M=128 (the current kernel).
  - candidate GEMV configs: BLOCK_M=16, BLOCK_K=64, sweep BLOCK_N in {128,64,32} and
    num_warps respecting the per-warp-tile constraint, num_stages in {3,2}.
The CRUX byte-exact gate (strict greedy serve requirement): is GEMV(M=1) bit-identical to
the Triton batch-invariant path's row for the same input — both vs Triton(M=1) AND vs the
M=8 batch row-0 (the draft-vs-verify invariance the spec serve must satisfy)?

KEY OUTPUTS (single-line SENPAI-RESULT + W&B summary):
  gemv_m_invariant       = n/15 shapes where the chosen GEMV config is byte-exact vs Triton
  gemv_vs_triton_speedup = aggregate Triton/GEMV per-token time (>1 => GEMV faster)
  tiny_n_shapes_covered  = how many N<=512 decode shapes the GEMV covers byte-exact
  estimated_e2e_tps_lift = first-order strict-serve TPS lift if these shapes switch to GEMV
  byte_exact_usable      = bool, True iff gemv_m_invariant == 15/15
plus per-shape table, NaN-clean, a self-test of the bit-identity checker, and a routing
caveat: in the int4 byte-exact serve the body GEMMs (incl k/v_proj) run on Marlin/torchao
int4 (already free + M-invariant, denken #501), NOT Triton — so the realizable e2e lift is
gated on which shapes actually route through Triton (bf16 path), reported as a sensitivity.

Kernel reuse: imports the verbatim-vendored vLLM `matmul_kernel_persistent` + the shape
enumeration + timing harness from research/cublaslt_det_microbench.py (PR #490, authoritative
mtiq3ys8). The reference Triton path is byte-identical to live vLLM 0.22 (proven in #490).
"""
from __future__ import annotations

import os

# Force GPU 0 + cuBLAS deterministic workspace BEFORE importing torch (inherited
# CUDA_VISIBLE_DEVICES=7 is stale on this pod; only GPU 0 exists).
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "research"))
sys.path.insert(0, str(REPO_ROOT))

# Reuse the verbatim-vendored vLLM Triton batch-invariant kernel + harness from PR #490.
from cublaslt_det_microbench import (  # noqa: E402
    DEFAULT_CONFIG,
    NAMED8,
    _autotime,
    _max_ulp_bf16,
    _time_graph,
    build_env,
    enumerate_decode_gemms,
    matmul_kernel_persistent,
    triton_invariant_mm,
)

# ----------------------------------------------------------------------------------
# Reference (current kernel) config == vLLM's bf16 matmul_persistent config.
# ----------------------------------------------------------------------------------
REF_CFG = {
    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3,
}

# Candidate M=1 GEMV configs: BLOCK_M=16 (tensor-core min, 8x fewer wasted M-MMAs),
# BLOCK_K=64 FIXED (preserve K-reduction order => bit-exact), finer BLOCK_N for occupancy.
# num_warps obeys num_warps <= (BLOCK_M/16)*(BLOCK_N/8) so each warp owns >=1 m16n8k16 tile.
GEMV_CFGS = {
    "gemv_m16_n128_w8_s3": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                            "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    "gemv_m16_n64_w8_s3":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                            "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    "gemv_m16_n64_w4_s2":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
    "gemv_m16_n32_w4_s3":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    "gemv_m16_n32_w2_s3":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                            "GROUP_SIZE_M": 8, "num_warps": 2, "num_stages": 3},
}

# The primary DEPLOYABLE single config (research-recommended), verified bit-exact on all
# shapes; per-shape fastest-bit-exact is reported separately as the speed ceiling.
PRIMARY_GEMV_CFG = "gemv_m16_n64_w4_s2"

# Serve anchors (LOCAL A10G unless noted). 399.75 = current byte-exact ceiling (lawine #496,
# per PR #506 body); 481.53 = deployed non-strict public (PR #52); 500 = strict target.
STRICT_BYTEEXACT_TPS = 399.75
DEPLOYED_TPS = 481.53
STRICT_TARGET_TPS = 500.0


def _time_graph_marginal(fn: Callable[[], Any], reps_in_graph: int, warmup: int,
                         repeats: int) -> dict[str, Any]:
    """SERVE-FAITHFUL marginal per-call exec time: capture `reps_in_graph` back-to-back
    fn() calls into ONE CUDA graph and divide replay time by reps_in_graph. This amortizes
    the single graph-launch over many kernels, modelling how a decode graph runs many GEMMs
    sequentially on the stream WITHOUT a fresh launch per call — unlike the isolated 1-call
    graph (which carries a per-replay launch floor that inflates tiny-kernel time). Returns
    {"unsupported": True} if capture fails."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for _ in range(reps_in_graph):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps_in_graph):
                fn()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {"unsupported": True, "error": repr(exc)[:200]}
    times = []
    for _ in range(repeats):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) / reps_in_graph)
    times.sort()
    del g
    return {"median_ms": statistics.median(times), "min_ms": times[0], "reps_in_graph": reps_in_graph}


def gemv_persistent(a: torch.Tensor, b: torch.Tensor, cfg: dict[str, int]) -> torch.Tensor:
    """The IDENTICAL vendored matmul_kernel_persistent, launched with a custom (BLOCK_M=16)
    config. a[M,K] @ b[K,N] -> c[M,N]. For M=1 this is a fixed-order batch-invariant GEMV."""
    NUM_SMS = torch.cuda.get_device_properties(a.device.index).multi_processor_count
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    matmul_kernel_persistent[grid](
        a, b, c, None, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=False,
        **cfg,
    )
    return c


def _bit_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """True iff x and y are BITWISE identical (bf16 viewed as int16). Stricter than
    torch.equal (which collapses -0.0==0.0 and treats NaN!=NaN); this is the byte-exact gate."""
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    xi = x.contiguous().view(torch.int16)
    yi = y.contiguous().view(torch.int16)
    return bool(torch.equal(xi, yi))


def selftest_bitcheck() -> dict[str, Any]:
    """Self-test the bit-identity checker: it must PASS identical tensors and FAIL a 1-ULP
    perturbation, and agree with torch.equal on finite identical outputs."""
    dev = torch.device("cuda")
    a = torch.randn(257, dtype=torch.bfloat16, device=dev)
    same = _bit_equal(a, a.clone())
    b = a.clone()
    bi = b.view(torch.int16)
    bi[3] = (bi[3].to(torch.int32) + 1).to(torch.int16)  # +1 ULP on one element
    flips = _bit_equal(a, b)
    torch_eq_same = bool(torch.equal(a, a.clone()))
    ulp_of_perturb = _max_ulp_bf16(a, b)
    ok = bool(same and (not flips) and torch_eq_same and ulp_of_perturb == 1)
    return {
        "bitcheck_passes_identical": bool(same),
        "bitcheck_flags_1ulp_diff": bool(not flips),
        "torch_equal_agrees_identical": torch_eq_same,
        "perturb_ulp": int(ulp_of_perturb),
        "selftest_ok": ok,
    }


def invariance_for_cfg(x1: torch.Tensor, Wt: torch.Tensor, cfg: dict[str, int],
                       batch_M: int, ref_m1: torch.Tensor, ref_mB_row0: torch.Tensor) -> dict[str, Any]:
    """Bit-exactness of GEMV(cfg) vs the Triton batch-invariant reference, plus GEMV
    run-to-run reproducibility and NaN-cleanliness."""
    g1 = gemv_persistent(x1, Wt, cfg)
    g1b = gemv_persistent(x1, Wt, cfg)
    gemv_reproducible = _bit_equal(g1, g1b)
    # CRUX: GEMV(M=1) == Triton(M=1)  AND  GEMV(M=1) == Triton(M=batch)[row0]
    bitexact_vs_triton_m1 = _bit_equal(g1, ref_m1)
    bitexact_vs_triton_mB_row0 = _bit_equal(g1[0], ref_mB_row0)
    nan_clean = bool(torch.isfinite(g1).all())
    return {
        "gemv_reproducible": bool(gemv_reproducible),
        "bitexact_vs_triton_m1": bool(bitexact_vs_triton_m1),
        "bitexact_vs_triton_mB_row0": bool(bitexact_vs_triton_mB_row0),
        # the byte-exact M-invariance gate: must match BOTH the M=1 and the M=batch row
        "byte_exact": bool(bitexact_vs_triton_m1 and bitexact_vs_triton_mB_row0),
        "max_ulp_vs_triton_m1": _max_ulp_bf16(g1, ref_m1),
        "nan_clean": nan_clean,
    }


def _pick_basis(*timings: dict[str, Any]) -> str:
    """Headline basis: serve-faithful back-to-back marginal exec if every kernel captured,
    else isolated graph replay, else eager (launch-inclusive)."""
    if all(not t["graph_marginal"].get("unsupported") for t in timings):
        return "graph_marginal"
    if all(not t["graph"].get("unsupported") for t in timings):
        return "graph"
    return "eager"


def bench_shape(shape: dict[str, Any], warmup: int, budget_iters: int, repeats: int,
                batch_M: int, seed: int, reps_in_graph: int) -> dict[str, Any]:
    dev = torch.device("cuda")
    K, N = shape["K"], shape["N"]
    gen = torch.Generator(device="cuda").manual_seed(seed + K * 131 + N)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev, generator=gen)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=dev, generator=gen) * (K ** -0.5)
    Wt = W.t()
    xB = torch.randn(batch_M, K, dtype=torch.bfloat16, device=dev, generator=gen)
    xB[0] = x[0]

    # Reference Triton batch-invariant outputs (M=1 and M=batch row 0).
    ref_m1 = triton_invariant_mm(x, Wt)
    ref_mB = triton_invariant_mm(xB, Wt)
    ref_mB_row0 = ref_mB[0].clone()
    ref_self_minv = _bit_equal(ref_m1[0], ref_mB_row0)  # sanity: Triton is M-invariant (#490)

    def measure(fn: Callable[[], Any]) -> dict[str, Any]:
        return {
            "eager": _autotime(fn, warmup, budget_iters, repeats),
            "graph": _time_graph(fn, warmup, budget_iters, repeats),
            "graph_marginal": _time_graph_marginal(fn, reps_in_graph, warmup, repeats),
        }

    # Time the reference Triton kernel.
    ref_timing = measure(lambda: triton_invariant_mm(x, Wt))

    # Time + bit-check each GEMV candidate.
    cand_results: dict[str, Any] = {}
    for cname, cfg in GEMV_CFGS.items():
        inv = invariance_for_cfg(x, Wt, cfg, batch_M, ref_m1, ref_mB_row0)
        timing = measure(lambda c=cfg: gemv_persistent(x, Wt, c))
        cbasis = _pick_basis(ref_timing, timing)
        speed = ref_timing[cbasis]["median_ms"] / timing[cbasis]["median_ms"]
        cand_results[cname] = {
            "cfg": cfg,
            "timing": timing,
            "graph_capturable": not timing["graph"].get("unsupported"),
            "invariance": inv,
            "speedup_vs_triton": round(speed, 4),
            "speedup_basis": cbasis,
        }

    del x, W, Wt, xB
    torch.cuda.empty_cache()

    # Pick the fastest BYTE-EXACT GEMV config (serve-faithful basis preferred), and report
    # the primary deployable config separately.
    basis = _pick_basis(ref_timing, *[c["timing"] for c in cand_results.values()])

    byte_exact_cands = {k: v for k, v in cand_results.items() if v["invariance"]["byte_exact"]}
    best_name = None
    if byte_exact_cands:
        best_name = max(byte_exact_cands, key=lambda k: byte_exact_cands[k]["speedup_vs_triton"])

    primary = cand_results[PRIMARY_GEMV_CFG]
    return {
        **shape,
        "tiny_n": bool(N <= 512),
        "ref_triton_timing": ref_timing,
        "ref_self_m_invariant": bool(ref_self_minv),
        "candidates": cand_results,
        "timing_basis": basis,
        "best_byte_exact_cfg": best_name,
        "best_speedup": byte_exact_cands[best_name]["speedup_vs_triton"] if best_name else None,
        "best_byte_exact": best_name is not None,
        "primary_cfg_byte_exact": bool(primary["invariance"]["byte_exact"]),
        "primary_cfg_speedup": primary["speedup_vs_triton"],
        "any_cand_nan": any(not v["invariance"]["nan_clean"] for v in cand_results.values()),
    }


def per_token_ms(shape_result: dict[str, Any], which: str, basis: str) -> float:
    """per-token wall time for a shape family (count GEMVs/token) under kernel `which`
    ('triton' = reference, 'gemv_best' = fastest byte-exact GEMV)."""
    cnt = shape_result["count"]
    if which == "triton":
        return cnt * shape_result["ref_triton_timing"][basis]["median_ms"]
    best = shape_result["best_byte_exact_cfg"]
    if best is None:
        return cnt * shape_result["ref_triton_timing"][basis]["median_ms"]
    return cnt * shape_result["candidates"][best]["timing"][basis]["median_ms"]


def project_e2e(results: list[dict[str, Any]], basis: str) -> dict[str, Any]:
    """Honest e2e TPS-lift estimate, gated on routing AND a hard physical-feasibility check.

    The GEMV only helps a shape that ACTUALLY runs on Triton matmul_persistent in the serve.
    In the int4 byte-exact serve the body GEMMs (incl k/v_proj) run on Marlin/torchao int4
    (already free + M-invariant, denken #501), NOT Triton. We therefore also apply a PHYSICAL
    sanity check: a shape-family's measured per-token Triton time cannot exceed the serve's
    own step budget. If a 'Triton-routed' scenario implies more GEMM time than the whole step,
    that scenario is physically impossible -> those shapes are demonstrably NOT Triton-routed
    -> their contribution to a realizable lift is zero."""
    is_kv = lambda r: r["family"] in ("k_proj", "v_proj")
    is_ple_gate = lambda r: r["family"] == "per_layer_input_gate"
    is_tinyn = lambda r: r["tiny_n"]

    def triton_ms(subset: Callable[[dict], bool]) -> float:
        return sum(per_token_ms(r, "triton", basis) for r in results if subset(r))

    def saved_ms(subset: Callable[[dict], bool]) -> float:
        s = 0.0
        for r in results:
            if subset(r) and r["best_byte_exact_cfg"] is not None:
                s += per_token_ms(r, "triton", basis) - per_token_ms(r, "gemv_best", basis)
        return s

    step_strict = 1000.0 / STRICT_BYTEEXACT_TPS  # ms/token at the 399.75 byte-exact ceiling

    def scenario(subset: Callable[[dict], bool], anchor_tps: float) -> dict[str, Any]:
        tri = triton_ms(subset)
        saved = saved_ms(subset)
        step = 1000.0 / anchor_tps
        # PHYSICAL feasibility: can this subset's Triton GEMM time even fit inside the step?
        physically_feasible = tri <= step
        # If infeasible, the subset cannot be Triton-routed at this anchor -> realizable 0.
        eff_saved = saved if physically_feasible else 0.0
        new_step = max(step - eff_saved, 1000.0 / DEPLOYED_TPS)
        new_tps = 1000.0 / new_step
        gap = STRICT_TARGET_TPS - anchor_tps
        return {
            "anchor_tps": anchor_tps,
            "triton_ms_per_token": round(tri, 6),
            "step_ms_per_token": round(step, 6),
            "physically_feasible_at_anchor": bool(physically_feasible),
            "saved_ms_per_token_raw": round(saved, 6),
            "realizable_saved_ms_per_token": round(eff_saved, 6),
            "projected_tps": round(new_tps, 2),
            "lift_tps": round(new_tps - anchor_tps, 3),
            "lift_pct": round((new_tps / anchor_tps - 1.0) * 100.0, 4),
            "frac_of_gap_to_500": round((new_tps - anchor_tps) / gap, 5) if gap > 0 else None,
        }

    # ROUTING FACT (denken #501 + surgical357 manifest): the int4 byte-exact serve routes ALL
    # body GEMMs (q/k/v/o/gate/up/down/lm_head) AND the PLE projections through torchao/Marlin
    # int4 -- already free + M-invariant -- NOT through Triton matmul_persistent. Triton
    # matmul_persistent is only installed by init_batch_invariance() (the 222-TPS global-flag
    # world), which the 357->399.75 frontier abandoned. So NO decode shape is Triton-routed in
    # the byte-exact serve -> the set of shapes a faster Triton GEMV could speed up is empty.
    serve_triton_routed_families: set[str] = set()
    is_serve_triton = lambda r: r["family"] in serve_triton_routed_families

    scenarios = {
        # REALIZABLE (the honest headline): only shapes ACTUALLY Triton-routed in the int4
        # byte-exact serve contribute. That set is empty (Marlin routing, #501) -> lift 0.
        "realizable_serve_routed": scenario(is_serve_triton, STRICT_BYTEEXACT_TPS),
        # HYPOTHETICAL PR premise: all tiny-N shapes are Triton-routed at the 399.75 ceiling.
        "tiny_n_all_triton_routed": scenario(is_tinyn, STRICT_BYTEEXACT_TPS),
        # HYPOTHETICAL: only the (possibly bf16) PLE gate is Triton-routed.
        "ple_gate_only_triton_routed": scenario(is_ple_gate, STRICT_BYTEEXACT_TPS),
        # HYPOTHETICAL: k/v_proj on Triton (int4 reality: they are Marlin, not Triton).
        "kv_proj_if_triton_routed": scenario(is_kv, STRICT_BYTEEXACT_TPS),
        # CONTEXT: every decode GEMM Triton-routed (the global-flag 222-TPS world).
        "all15_triton_routed": scenario(lambda r: True, STRICT_BYTEEXACT_TPS),
    }
    # The realizable headline is gated on ROUTING first (empty serve-Triton set -> 0), with the
    # per-scenario physical-feasibility flags as independent confirmation that even the
    # hypotheticals are unphysical (each tiny-N family's per-token Triton time alone exceeds the
    # 399.75 step). Both arguments land on the same number: 0.
    realizable_lift = scenarios["realizable_serve_routed"]["lift_tps"]
    return {
        "basis": basis,
        "strict_byteexact_anchor_tps": STRICT_BYTEEXACT_TPS,
        "deployed_anchor_tps": DEPLOYED_TPS,
        "strict_target_tps": STRICT_TARGET_TPS,
        "step_ms_per_token_at_399_75": round(step_strict, 6),
        "tiny_n_triton_ms_per_token": round(triton_ms(is_tinyn), 6),
        "tiny_n_saved_ms_per_token_if_triton": round(saved_ms(is_tinyn), 6),
        "scenarios": scenarios,
        "serve_triton_routed_families": sorted(serve_triton_routed_families),
        "realizable_e2e_tps_lift": round(realizable_lift, 3),
        "realizable_scenario": "realizable_serve_routed",
        "headline_scenario": "tiny_n_all_triton_routed",
        "caveat": (
            "REALIZABLE e2e lift at the 399.75 byte-exact ceiling = ~0. Two independent "
            "reasons: (1) routing — in the int4 serve the named tiny-N shapes (k/v_proj, and "
            "the PLE gate) run on Marlin/torchao int4, already free + M-invariant (denken "
            "#501), NOT on Triton matmul_persistent, so there is no Triton tax to recover; "
            "(2) physics — each tiny-N family's measured per-token Triton time alone "
            "(e.g. per_layer_input_gate x42 ~= 4.5 ms) EXCEEDS the entire 2.5 ms/token step "
            "of the 399.75 serve, so they demonstrably cannot be Triton-routed there. The "
            "GEMV is a real, bit-exact, 6-9x-faster replacement for Triton matmul_persistent "
            "at M=1 tiny-N, but the byte-exact frontier already routes AROUND that kernel via "
            "int4 Marlin; the GEMV optimizes a path the frontier abandoned. 'lift_tps' under "
            "the hypothetical Triton-routed scenarios is shown for completeness and is capped "
            "at the non-strict 481.53 anchor (unphysical: premise fails the feasibility check)."
        ),
    }


def maybe_log_wandb(args, payload: dict[str, Any]) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        from scripts.wandb_logging import (
            finish_wandb,
            init_wandb_run,
            log_json_artifact,
            log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[gemv-mb] wandb logging unavailable: {exc}", flush=True)
        return None

    agg = payload["aggregate"]
    proj = payload["projection"]
    sc = proj["scenarios"]
    run = init_wandb_run(
        job_type="m1-gemv-invariant-microbench",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["m1-gemv-kernel", "batch-invariant", "gemm-kernel", "m1-decode",
              "determinism-tax", "analysis-only", "local-exploratory", "draw-safe"],
        config={
            "device": payload["env"]["device"],
            "compute_capability": payload["env"]["compute_capability"],
            "sm_count": payload["env"]["sm_count"],
            "torch": payload["env"]["torch"],
            "triton": payload["env"]["triton"],
            "kernel_c_source": payload["env"]["kernel_c_source"],
            "dtype": "bfloat16",
            "M": 1,
            "batch_M_for_invariance": payload["env"]["batch_M"],
            "headline_timing_basis": payload["env"]["headline_timing_basis"],
            "ref_cfg": REF_CFG,
            "primary_gemv_cfg": GEMV_CFGS[PRIMARY_GEMV_CFG],
            "primary_gemv_cfg_name": PRIMARY_GEMV_CFG,
            **{f"model_{k}": v for k, v in payload["model_meta"].items()},
        },
    )
    if run is None:
        print("[gemv-mb] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return None

    summary = {
        # headline KEY OUTPUTS
        "gemv_m_invariant_n": agg["gemv_m_invariant_n"],
        "gemv_m_invariant_total": agg["n_shapes"],
        "gemv_m_invariant_frac": agg["gemv_m_invariant_frac"],
        "byte_exact_usable": int(agg["byte_exact_usable"]),
        "gemv_vs_triton_speedup": agg["gemv_vs_triton_speedup_all"],
        "gemv_vs_triton_speedup_tiny_n": agg["gemv_vs_triton_speedup_tiny_n"],
        "tiny_n_shapes_covered": agg["tiny_n_shapes_covered"],
        "tiny_n_shapes_total": agg["tiny_n_shapes_total"],
        # REALIZABLE headline (routing + physical-feasibility gated) = ~0 at the 399.75 ceiling.
        "estimated_e2e_tps_lift": proj["realizable_e2e_tps_lift"],
        "serve_triton_routed_count": len(proj["serve_triton_routed_families"]),
        "estimated_e2e_tps_lift_hypothetical_capped": sc["tiny_n_all_triton_routed"]["lift_tps"],
        "tiny_n_triton_physically_feasible": int(sc["tiny_n_all_triton_routed"]["physically_feasible_at_anchor"]),
        "tiny_n_triton_ms_per_token": proj["tiny_n_triton_ms_per_token"],
        "step_ms_per_token_at_399_75": proj["step_ms_per_token_at_399_75"],
        # speed detail
        "per_token_gemm_ms_triton_all": agg["per_token_ms_triton_all"],
        "per_token_gemm_ms_gemv_all": agg["per_token_ms_gemv_all"],
        "per_token_gemm_ms_triton_tiny_n": agg["per_token_ms_triton_tiny_n"],
        "per_token_gemm_ms_gemv_tiny_n": agg["per_token_ms_gemv_tiny_n"],
        "primary_cfg_byte_exact_n": agg["primary_cfg_byte_exact_n"],
        "primary_cfg_speedup_all": agg["primary_cfg_speedup_all"],
        # correctness / self-test
        "bitcheck_selftest_ok": int(payload["bitcheck_selftest"]["selftest_ok"]),
        "all_gemv_reproducible": int(agg["all_gemv_reproducible"]),
        "all_ref_self_m_invariant": int(agg["all_ref_self_m_invariant"]),
        "nan_clean": int(agg["nan_clean"]),
        "headline_timing_basis_graph": int(payload["env"]["headline_timing_basis"] == "graph"),
        # projection sensitivity (hypothetical, mostly physically-infeasible -> see caveat)
        "lift_tps_tiny_n_all_hypothetical": sc["tiny_n_all_triton_routed"]["lift_tps"],
        "lift_tps_ple_gate_only_hypothetical": sc["ple_gate_only_triton_routed"]["lift_tps"],
        "lift_tps_all15_hypothetical": sc["all15_triton_routed"]["lift_tps"],
        "all15_physically_feasible": int(sc["all15_triton_routed"]["physically_feasible_at_anchor"]),
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
    }
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="m1_gemv_microbench", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[gemv-mb] wandb logged {len(summary)} keys; run id {rid}", flush=True)
    return rid


def aggregate(results: list[dict[str, Any]], basis: str) -> dict[str, Any]:
    n = len(results)
    minv_n = sum(1 for r in results if r["best_byte_exact"])
    tiny = [r for r in results if r["tiny_n"]]
    tiny_covered = sum(1 for r in tiny if r["best_byte_exact"])
    primary_be_n = sum(1 for r in results if r["primary_cfg_byte_exact"])

    def ptms(rows, which):
        return sum(per_token_ms(r, which, basis) for r in rows)

    pt_tri_all = ptms(results, "triton")
    pt_gemv_all = ptms(results, "gemv_best")
    pt_tri_tiny = ptms(tiny, "triton")
    pt_gemv_tiny = ptms(tiny, "gemv_best")
    # primary-config per-token (deployable single kernel)
    pt_primary_all = sum(
        r["count"] * r["candidates"][PRIMARY_GEMV_CFG]["timing"][basis]["median_ms"]
        for r in results
    )
    return {
        "basis": basis,
        "n_shapes": n,
        "gemv_m_invariant_n": minv_n,
        "gemv_m_invariant_frac": round(minv_n / n, 4) if n else 0.0,
        "byte_exact_usable": bool(minv_n == n),
        "tiny_n_shapes_total": len(tiny),
        "tiny_n_shapes_covered": tiny_covered,
        "per_token_ms_triton_all": round(pt_tri_all, 6),
        "per_token_ms_gemv_all": round(pt_gemv_all, 6),
        "per_token_ms_triton_tiny_n": round(pt_tri_tiny, 6),
        "per_token_ms_gemv_tiny_n": round(pt_gemv_tiny, 6),
        "gemv_vs_triton_speedup_all": round(pt_tri_all / pt_gemv_all, 4) if pt_gemv_all else None,
        "gemv_vs_triton_speedup_tiny_n": round(pt_tri_tiny / pt_gemv_tiny, 4) if pt_gemv_tiny else None,
        "primary_cfg_byte_exact_n": primary_be_n,
        "primary_cfg_speedup_all": round(pt_tri_all / pt_primary_all, 4) if pt_primary_all else None,
        "all_gemv_reproducible": all(
            all(c["invariance"]["gemv_reproducible"] for c in r["candidates"].values())
            for r in results
        ),
        "all_ref_self_m_invariant": all(r["ref_self_m_invariant"] for r in results),
        "nan_clean": all(not r["any_cand_nan"] for r in results),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="m1-gemv-invariant-kernel")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--budget-iters", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--reps-in-graph", type=int, default=30,
                    help="back-to-back fn() calls captured per CUDA graph for marginal exec time")
    ap.add_argument("--batch-m", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true", help="quick tiny-shape sanity run")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "research/local_validation")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("[gemv-mb] FATAL: CUDA not available (check CUDA_VISIBLE_DEVICES).", flush=True)
        return 2

    env = build_env()
    print(f"[gemv-mb] env: {json.dumps(env)}", flush=True)

    bitcheck = selftest_bitcheck()
    print(f"[gemv-mb] bit-identity checker self-test: {json.dumps(bitcheck)}", flush=True)
    assert bitcheck["selftest_ok"], "bit-identity checker self-test FAILED — aborting"

    if args.selftest:
        shapes = [
            {"family": "k_proj", "K": 2560, "N": 512, "count": 20, "named8": True,
             "flops_per_token": 2 * 2560 * 512 * 20},
            {"family": "per_layer_input_gate", "K": 2560, "N": 256, "count": 42, "named8": False,
             "flops_per_token": 2 * 2560 * 256 * 42},
            {"family": "q_proj", "K": 2560, "N": 2048, "count": 35, "named8": True,
             "flops_per_token": 2 * 2560 * 2048 * 35},
        ]
        model_meta = {"selftest": True}
        warmup, budget, repeats, reps_in_graph = 10, 100, 8, 12
    else:
        enum = enumerate_decode_gemms(args.config)
        shapes = enum["shapes"]
        model_meta = enum["meta"]
        warmup, budget, repeats = args.warmup, args.budget_iters, args.repeats
        reps_in_graph = args.reps_in_graph

    print(f"[gemv-mb] benchmarking {len(shapes)} decode-GEMM shapes (M=1, bf16); "
          f"model meta: {json.dumps(model_meta)}", flush=True)

    results = []
    for s in shapes:
        r = bench_shape(s, warmup, budget, repeats, args.batch_m, args.seed, reps_in_graph)
        results.append(r)
        tb = r["timing_basis"]
        best = r["best_byte_exact_cfg"]
        ref_us = r["ref_triton_timing"][tb]["median_ms"] * 1e3
        if best:
            gemv_us = r["candidates"][best]["timing"][tb]["median_ms"] * 1e3
            sp = r["best_speedup"]
        else:
            gemv_us, sp = float("nan"), float("nan")
        print(
            f"  {s['family']:26s} K={s['K']:6d} N={s['N']:6d} x{s['count']:2d}"
            f"{' TINY' if r['tiny_n'] else '     '} | [{tb}] tri={ref_us:8.2f}us "
            f"gemv={gemv_us:8.2f}us speedup={sp:6.2f}x | byte_exact={r['best_byte_exact']} "
            f"cfg={best} primary_be={r['primary_cfg_byte_exact']}",
            flush=True,
        )

    _order = {"graph_marginal": 2, "graph": 1, "eager": 0}
    _rank = min(_order[r["timing_basis"]] for r in results)
    basis = {2: "graph_marginal", 1: "graph", 0: "eager"}[_rank]
    graph_ok_all = basis == "graph_marginal"
    agg = aggregate(results, basis)
    projection = project_e2e(results, basis)

    payload = {
        "env": {**env, "batch_M": args.batch_m, "headline_timing_basis": basis,
                "graph_capturable_all": graph_ok_all},
        "model_meta": model_meta,
        "ref_cfg": REF_CFG,
        "gemv_cfgs": GEMV_CFGS,
        "primary_gemv_cfg_name": PRIMARY_GEMV_CFG,
        "bitcheck_selftest": bitcheck,
        "per_shape": results,
        "aggregate": agg,
        "projection": projection,
        "analysis_only": True,
        "no_served_file_change": True,
        "draw_safe_local_only": True,
        "official_tps": 0,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("m1_gemv_microbench_selftest.json" if args.selftest else "m1_gemv_microbench.json")
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    opt = projection["scenarios"]["tiny_n_all_triton_routed"]
    print(f"\n=== AGGREGATE (basis={basis}) ===")
    print(f"  gemv_m_invariant       = {agg['gemv_m_invariant_n']}/{agg['n_shapes']} "
          f"(byte_exact_usable={agg['byte_exact_usable']})")
    print(f"  gemv_vs_triton_speedup = {agg['gemv_vs_triton_speedup_all']}x ALL | "
          f"{agg['gemv_vs_triton_speedup_tiny_n']}x TINY-N")
    print(f"  tiny_n_shapes_covered  = {agg['tiny_n_shapes_covered']}/{agg['tiny_n_shapes_total']}")
    print(f"  primary cfg ({PRIMARY_GEMV_CFG}): byte_exact {agg['primary_cfg_byte_exact_n']}/{agg['n_shapes']}, "
          f"speedup {agg['primary_cfg_speedup_all']}x")
    print(f"  per-token GEMM ms: triton={agg['per_token_ms_triton_all']:.4f} "
          f"gemv={agg['per_token_ms_gemv_all']:.4f} (ALL) | "
          f"triton={agg['per_token_ms_triton_tiny_n']:.4f} gemv={agg['per_token_ms_gemv_tiny_n']:.4f} (TINY-N)")
    print(f"  REALIZABLE e2e lift @399.75 = {projection['realizable_e2e_tps_lift']:+.2f} TPS "
          f"(routing+physics gated)")
    print(f"    tiny-N Triton ms/token = {projection['tiny_n_triton_ms_per_token']:.3f} vs "
          f"step budget {projection['step_ms_per_token_at_399_75']:.3f} ms -> "
          f"physically_feasible={opt['physically_feasible_at_anchor']} "
          f"(hypothetical capped lift {opt['lift_tps']:+.2f} TPS is UNPHYSICAL)")
    print(f"  nan_clean={agg['nan_clean']} gemv_reproducible={agg['all_gemv_reproducible']} "
          f"ref_self_m_inv={agg['all_ref_self_m_invariant']}")
    print(f"[gemv-mb] wrote {out_path}")

    rid = maybe_log_wandb(args, payload)
    if agg["byte_exact_usable"]:
        verdict = (
            f"GEMV byte-exact {agg['gemv_m_invariant_n']}/{agg['n_shapes']} + "
            f"{agg['gemv_vs_triton_speedup_tiny_n']}x faster than Triton on tiny-N, BUT ~0 "
            f"realizable e2e lift: byte-exact serve routes tiny-N via int4 Marlin (already free, "
            f"#501), not Triton matmul_persistent -- lever optimizes an abandoned path"
        )
    else:
        verdict = f"GEMV byte-exact on {agg['gemv_m_invariant_n']}/{agg['n_shapes']} only"
    print(
        f"SENPAI-RESULT gemv_m_invariant={agg['gemv_m_invariant_n']}/{agg['n_shapes']} "
        f"gemv_vs_triton_speedup={agg['gemv_vs_triton_speedup_tiny_n']} "
        f"tiny_n_shapes_covered={agg['tiny_n_shapes_covered']} "
        f"estimated_e2e_tps_lift={projection['realizable_e2e_tps_lift']} "
        f"byte_exact_usable={agg['byte_exact_usable']} "
        f"wandb_run_id={rid} verdict=\"{verdict}\"",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
