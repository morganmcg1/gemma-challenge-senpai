#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #680 (land) -- LOSSLESS int4 verify-GEMM: is the deployed g=128 Marlin GEMM
byte-identical across batch width M, and can a config recover identity if not?

THE PR PREMISE (to be tested, not assumed)
------------------------------------------
PR #680 hypothesizes the compressed-tensors int4 Marlin GEMM is "batch-width
sensitive": the width-(K+1) verify forward (K=5 -> M=6) produces different logits
than width-1 (M=1) AR because the kernel's reduction/tiling depends on M, flipping
near-tie argmax and breaking greedy identity (strict-#319). The "bold alternative"
is to make the int4 verify-GEMM byte-identical to AR-width BY CONSTRUCTION via a
Marlin config.

This script is the DECISIVE, cheap, isolated test of that premise at the EXACT
deployed spec (group_size=128, the int4_g128_lmhead body+lm_head shapes, M=6
verify), settling a standing conflict in the codebase:
  * verify_flip_probe #23 (full forward): "Marlin int4 GEMM is the irreducible source."
  * reduction_sensitivity_census #491 (isolated GEMM microbench, group_size=-1, M=8):
    "GEMM is byte-identical across M (maxdiff=0); attention reduction is the source."
#23 ran the FULL forward (attention + GEMM both vary with M) and could not isolate;
#491 ran the direct GEMM microbench but used group_size=-1, not the deployed g=128.
This closes that one gap: same microbench, group_size=128, M in {1,2,4,6,8}.

The researcher digest (Marlin internals) says the M-dependence, IF present, lives in
the stripe-based global-reduce epilogue (slice_count>1 -> a barrier-sequenced FP32
accumulation whose order depends on prob_m_split). Whether that path is even ACTIVE
depends on whether k_tiles*n_tiles exceeds the SM count for a given shape/grid -- an
empirical question this microbench answers directly.

METHOD (LOCAL, analysis_only, no model load, no HF Job, no served change)
------------------------------------------------------------------------
For each group_size in {128 (deployed), -1 (#491 control)} and each deployed GEMM
shape (incl. the pruned lm_head 2560->16384), build a real GPTQ-Marlin int4 weight
and measure, over n_trials random inputs:
  * m_invariant_byte_rate[M]  = fraction of the M rows whose width-M GEMM output
        byte-matches the per-row width-1 (AR) GEMM output for the SAME row.
        m_inv==1.0  <=>  verify-width GEMM is byte-identical to AR-width -> the GEMM
        cannot flip any argmax -> GEMM contributes ZERO to break_rate.
  * max_abs_diff_vs_perrow[M]  the largest |bf16 delta| (0.0 == bit-identical).
Across the config sweep KNOBS = {use_atomic_add: off(deployed)/on} x
{use_fp32_reduce: on(deployed)/off}:
  * does ANY knob change m_inv? (the PR "split-K on/off, atomic toggles" sweep)
Plus two controls that make a 1.0 result trustworthy, not vacuous:
  * harness_sensitivity_byte_rate  -- a perturbed input MUST change the bytes (<1.0).
  * run_to_run_byte_rate           -- two launches of the SAME config (determinism floor).
And the call-site "config" the bold-alternative implies if the GEMM IS M-variant:
  * pad_to_canonical: pad BOTH M=1 and M=6 to M=8 (one kernel launch geometry) and
    check they become byte-identical to each other -- the minimal-kernel-change
    (researcher Option 1) feasibility probe (TPS cost measured separately if needed).

KEY OUTPUTS (-> PR #680 scalars):
  verify_gemm_byte_identical_achievable (1/0): is there a config (incl. the deployed
    one, i.e. "no change needed") under which the int4 verify-GEMM is byte-identical
    to AR-width at M=6 for ALL deployed shapes?
  marlin_gemm_is_m_invariant_as_deployed (1/0): the deployed knob, g=128, M=6.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Deployed int4_g128_lmhead geometry (gemma-4-E4B-it text_config). vLLM FUSES qkv and
# gate_up; the lm_head is the pruned 16384-row head (still int4 g=128). Every n>=2048.
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 256
LMHEAD_ROWS = 16384
DEPLOYED_GEMMS = {
    "qkv_proj":     (HIDDEN, N_Q_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM),  # 2560 -> 3072
    "o_proj":       (N_Q_HEADS * HEAD_DIM, HIDDEN),                              # 2048 -> 2560
    "gate_up_proj": (HIDDEN, 2 * INTERMEDIATE),                                  # 2560 -> 20480
    "down_proj":    (INTERMEDIATE, HIDDEN),                                      # 10240 -> 2560
    "lm_head":      (HIDDEN, LMHEAD_ROWS),                                       # 2560 -> 16384
}
DEPLOYED_GROUP_SIZE = 128          # the int4_g128_lmhead quant: group_size=128, symmetric, no act-order
CONTROL_GROUP_SIZE = -1            # ubel #491 used channel-wise (-1); included as a cross-check
VERIFY_M = 6                       # PR #680 K=5 -> M=6 verify width (primary)
M_LIST = [1, 2, 4, 6, 8]           # 1=AR ref; 6=K5 verify (PR #680); 8=K7 verify (#491 cross-check)
CANONICAL_PAD_M = 8                # researcher Option-1 canonical width (m_block_size_8 path, M<=8)

# config sweep: (use_atomic_add, use_fp32_reduce). "deployed" == the served default.
KNOBS = {
    "deployed":     (False, True),   # atomic OFF (heuristic for n>=2048), fp32 reduce ON
    "atomic_on":    (True,  True),    # force split-K atomic global-reduce (PR "split-K on")
    "fp16_reduce":  (False, False),   # fp16 global reduce (USE_FP32_REDUCE_DEFAULT off)
    "atomic_fp16":  (True,  False),   # both off-defaults
}


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return str(o)


def _build_marlin(dev, seed: int, size_k: int, size_n: int, group_size: int):
    """Real GPTQ-Marlin int4 GEMM at (size_k, size_n) with the given group_size,
    symmetric, act_order=False (the deployed quant). Returns:
      apply(x, use_atomic_add, use_fp32_reduce) -> y[M, size_n]
      heuristic_aa  -- vLLM should_use_atomic_add_reduce at M=VERIFY_M (deployed flag)
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mut
    from vllm.scalar_type import scalar_types

    qtype = scalar_types.uint4b8
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(size_k, size_n, generator=g, device=dev, dtype=torch.bfloat16) * 0.02)
    # marlin_quantize(w, quant_type, group_size, act_order) -> (w_ref, q_w, s, g_idx, sort_idx, rand_perm)
    _w_ref, q_w, s, g_idx, sort_idx, _ = mut.marlin_quantize(w, qtype, group_size, False)
    ws = mu.marlin_make_workspace_new(dev)
    empty_zp = torch.empty(0, dtype=torch.int, device=dev)
    heuristic_aa = bool(mu.should_use_atomic_add_reduce(
        m=VERIFY_M, n=size_n, k=size_k, device=dev, dtype=torch.bfloat16))

    def apply(x, use_atomic_add: bool, use_fp32_reduce: bool):
        xr = x.reshape(-1, size_k)
        return ops.marlin_gemm(
            xr, None, q_w, None, s, None, None, empty_zp, g_idx, sort_idx, ws, qtype,
            size_m=xr.shape[0], size_n=size_n, size_k=size_k, is_k_full=True,
            use_atomic_add=use_atomic_add, use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False).reshape(x.shape[:-1] + (size_n,))

    return apply, heuristic_aa


def _byte_rate(bat, ref) -> float:
    """Fraction of rows that byte-match (bf16 exact-equal across the full row)."""
    return float((bat == ref).all(dim=-1).float().mean().item())


def _maxdiff(bat, ref) -> float:
    return float((bat.float() - ref.float()).abs().max().item())


def run(out_path: Path, n_trials: int, seed: int, group_sizes: list[int]) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- launch with CUDA_VISIBLE_DEVICES=0")
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev)
    gpu = {"name": p.name, "sm_count": p.multi_processor_count,
           "cap": list(torch.cuda.get_device_capability(dev)),
           "total_mem_gib": round(p.total_memory / (1024 ** 3), 2)}
    print(f"[micro] gpu={gpu['name']} sm={gpu['sm_count']} cap={gpu['cap']}", flush=True)

    results: dict[str, Any] = {}
    any_nan = False
    t0 = time.time()

    for gs in group_sizes:
        gs_key = f"g{gs}"
        per_gemm: dict[str, Any] = {}
        for name, (size_k, size_n) in DEPLOYED_GEMMS.items():
            apply, heuristic_aa = _build_marlin(dev, seed, size_k, size_n, gs)
            shape_rec: dict[str, Any] = {
                "size_k": size_k, "size_n": size_n, "group_size": gs,
                "heuristic_use_atomic_add_at_M6": heuristic_aa, "knobs": {}}

            for knob_name, (use_aa, use_fp32) in KNOBS.items():
                m_inv_by_M: dict[str, float] = {}
                maxdiff_by_M: dict[str, float] = {}
                harness_rates, r2r_rates = [], []
                pad_byte_rates, pad_maxdiffs = [], []  # M=1 padded-to-8 vs M=6 padded-to-8

                for t in range(n_trials):
                    g = torch.Generator(device=dev).manual_seed(seed + 1000 * t)
                    xfull = torch.randn(max(M_LIST), size_k, generator=g, device=dev,
                                        dtype=torch.bfloat16)
                    # per-M: width-M batched vs per-row width-1 (AR) for the SAME rows
                    for M in M_LIST:
                        x = xfull[:M]
                        bat = apply(x, use_aa, use_fp32)
                        any_nan = any_nan or bool(torch.isnan(bat).any())
                        ref = torch.cat([apply(x[r:r + 1], use_aa, use_fp32)
                                         for r in range(M)], dim=0)
                        m_inv_by_M.setdefault(str(M), []).append(_byte_rate(bat, ref))
                        maxdiff_by_M.setdefault(str(M), []).append(_maxdiff(bat, ref))
                    # positive control: a perturbed input MUST change bytes (at M=8)
                    x8 = xfull[:CANONICAL_PAD_M]
                    base8 = apply(x8, use_aa, use_fp32)
                    xp = x8.clone()
                    xp[0, 0] = xp[0, 0] + torch.tensor(0.5, dtype=torch.bfloat16, device=dev)
                    harness_rates.append(_byte_rate(apply(xp, use_aa, use_fp32), base8))
                    # run-to-run determinism floor for THIS config
                    r2r_rates.append(_byte_rate(apply(x8, use_aa, use_fp32),
                                                apply(x8, use_aa, use_fp32)))
                    # pad-to-canonical (researcher Option 1): pad M=1 and M=6 to M=8 with
                    # zero rows -> identical kernel geometry -> expect byte-identical real rows.
                    def _padapply(M):
                        xx = xfull[:M]
                        pad = torch.zeros(CANONICAL_PAD_M - M, size_k, dtype=xx.dtype, device=dev)
                        y = apply(torch.cat([xx, pad], dim=0), use_aa, use_fp32)
                        return y[:M]
                    y1 = _padapply(1)        # AR token, padded to width 8
                    y6 = _padapply(VERIFY_M)  # verify, padded to width 8
                    # compare the shared first row (both are the SAME xfull[0])
                    pad_byte_rates.append(_byte_rate(y6[:1], y1[:1]))
                    pad_maxdiffs.append(_maxdiff(y6[:1], y1[:1]))

                def _mean(d):
                    return {k: sum(v) / len(v) for k, v in d.items()}
                def _mx(d):
                    return {k: max(v) for k, v in d.items()}
                shape_rec["knobs"][knob_name] = {
                    "use_atomic_add": use_aa, "use_fp32_reduce": use_fp32,
                    "m_invariant_byte_rate": _mean(m_inv_by_M),
                    "max_abs_diff_vs_perrow": _mx(maxdiff_by_M),
                    "harness_sensitivity_byte_rate": sum(harness_rates) / len(harness_rates),
                    "run_to_run_byte_rate": sum(r2r_rates) / len(r2r_rates),
                    "pad_to_canonical_M1_vs_M6_byte_rate": sum(pad_byte_rates) / len(pad_byte_rates),
                    "pad_to_canonical_M1_vs_M6_maxdiff": max(pad_maxdiffs),
                }
            dep = shape_rec["knobs"]["deployed"]
            print(f"[micro] {gs_key} {name:13s} k={size_k:5d} n={size_n:5d} "
                  f"heur_aa={heuristic_aa} | deployed m_inv@M6={dep['m_invariant_byte_rate'].get('6'):.4f} "
                  f"maxdiff@M6={dep['max_abs_diff_vs_perrow'].get('6'):.2e} "
                  f"harness={dep['harness_sensitivity_byte_rate']:.3f}", flush=True)
            per_gemm[name] = shape_rec
        results[gs_key] = per_gemm

    out = {
        "phase": "gemm_width_microbench", "gpu": gpu, "n_trials": n_trials,
        "verify_M": VERIFY_M, "M_list": M_LIST, "canonical_pad_M": CANONICAL_PAD_M,
        "group_sizes": group_sizes, "knobs": {k: list(v) for k, v in KNOBS.items()},
        "any_nan": bool(any_nan), "results": results,
        "elapsed_s": round(time.time() - t0, 1),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2),
    }

    # ---- derived PR scalars ----
    gdep = f"g{DEPLOYED_GROUP_SIZE}"

    def _all_shapes_m_inv(gs_key: str, knob: str, M: int) -> tuple[bool, float, float]:
        """All deployed shapes byte-identical at width M under (gs_key, knob)?"""
        rates, mds = [], []
        for name in DEPLOYED_GEMMS:
            kr = results[gs_key][name]["knobs"][knob]
            rates.append(kr["m_invariant_byte_rate"].get(str(M), 0.0))
            mds.append(kr["max_abs_diff_vs_perrow"].get(str(M), float("inf")))
        return (all(r >= 0.999 for r in rates), min(rates), max(mds))

    # is the GEMM M-invariant exactly as deployed (g=128, deployed knob, M=6)?
    dep_inv, dep_min_rate, dep_maxdiff = _all_shapes_m_inv(gdep, "deployed", VERIFY_M)
    # does ANY swept config achieve byte-identity at g=128/M=6? (deployed counts as "no change")
    cfg_found, winning_cfg = None, None
    for knob in KNOBS:
        ok, _, _ = _all_shapes_m_inv(gdep, knob, VERIFY_M)
        if ok:
            cfg_found, winning_cfg = True, knob
            break
    # pad-to-canonical feasibility (would recover identity if the GEMM were M-variant)
    pad_ok = all(
        results[gdep][name]["knobs"]["deployed"]["pad_to_canonical_M1_vs_M6_byte_rate"] >= 0.999
        for name in DEPLOYED_GEMMS)
    # harness positive control valid for every shape/knob (so a 1.0 is real, not stuck)
    harness_ok = all(
        results[gdep][name]["knobs"][knob]["harness_sensitivity_byte_rate"] < 0.999
        for name in DEPLOYED_GEMMS for knob in KNOBS)

    out["derived"] = {
        "marlin_gemm_is_m_invariant_as_deployed": bool(dep_inv),   # g=128, deployed knob, M=6
        "deployed_min_byte_rate_M6": dep_min_rate,
        "deployed_max_abs_diff_M6": dep_maxdiff,
        "verify_gemm_byte_identical_achievable": int(bool(cfg_found)),
        "winning_config": winning_cfg,
        "pad_to_canonical_recovers_identity": bool(pad_ok),
        "harness_positive_control_valid": bool(harness_ok),
        "note": ("verify_gemm_byte_identical_achievable=1 means a config (incl. the deployed "
                 "one, i.e. no change) makes the int4 verify-GEMM byte-identical to AR-width at "
                 "M=6 for ALL deployed shapes. This is necessary but NOT sufficient for lossless "
                 "verify: if the GEMM is already invariant, the greedy break is sourced elsewhere "
                 "(attention reduction, #491) and verify_gemm_byte_identical_achievable=1 carries "
                 "zero TPS cost but does not by itself fix greedy identity."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_jsonable(out), open(out_path, "w"), indent=2)
    d = out["derived"]
    print("\n" + "=" * 70, flush=True)
    print(f"[DERIVED] g=128 deployed knob, M={VERIFY_M}:", flush=True)
    print(f"  marlin_gemm_is_m_invariant_as_deployed = {d['marlin_gemm_is_m_invariant_as_deployed']} "
          f"(min_byte_rate={d['deployed_min_byte_rate_M6']:.4f}, max_abs_diff={d['deployed_max_abs_diff_M6']:.2e})", flush=True)
    print(f"  verify_gemm_byte_identical_achievable  = {d['verify_gemm_byte_identical_achievable']} "
          f"(winning_config={d['winning_config']})", flush=True)
    print(f"  pad_to_canonical_recovers_identity     = {d['pad_to_canonical_recovers_identity']}", flush=True)
    print(f"  harness_positive_control_valid         = {d['harness_positive_control_valid']}", flush=True)
    print(f"  result -> {out_path}", flush=True)
    print("=" * 70, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "gemm_width_microbench.json")
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[DEPLOYED_GROUP_SIZE, CONTROL_GROUP_SIZE])
    args = ap.parse_args()
    run(args.out, args.n_trials, args.seed, args.group_sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
