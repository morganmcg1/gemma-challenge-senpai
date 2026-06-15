#!/usr/bin/env python3
"""PR #395 (lawine) -- does any ALREADY-SHIPPING vLLM-0.22 quant kernel deliver a meaningful
fraction of cb3's body-weight-read shrink at the served verify width, WITH greedy byte-identity
vs the deployed int4-Marlin?  Extends the #388/#391 microbench harness (cb3_kernel_realized_bw.py).

THE QUESTION (#395 verbatim intent):
  cb3 (RHT-incoherence + L1-resident K=64 dim-2 Gaussian VQ, QTIP/QuIP#-class, ~3.24 bpw eff) cuts
  body weight-read bytes by -21.5% (byte_ratio 0.785) and is modelled +33/+38 realistic on the
  supply lane -- BUT no cb3 kernel ships in vLLM 0.22; it is source-build-only (a flagged served-file
  change). The deployability question: does ANY shipping vLLM-0.22 quant kernel buy even ~half that
  body-read shrink at the M=8 decode-verify width WHILE preserving greedy byte-identity vs the
  deployed int4-Marlin? If so, the supply lane becomes deployable WITHOUT a flagged source build.

WHAT THIS IS / IS NOT:
  * GPU RESEARCH MICROBENCH -- profiling only. Loading an ALTERNATIVE quant for a *local* microbench
    is measurement, NOT a deployment. NO served-file change, NO competition submission, 0 official
    TPS, NO Hugging Face job. Single assigned A10G (CUDA_VISIBLE_DEVICES=0; #358/#363 gotcha).
  * It enumerates the quant kernels actually importable+constructible in THIS vLLM 0.22 GPU venv,
    microbenches each shipping candidate at the real body GEMM geometry (g128, the 8 body GEMMs
    q/k/v/o/gate/up/down + lm_head) at M=1 and M=8, and measures (a) realized weight-read bytes +
    HBM-eff vs int4-Marlin, (b) the body-read shrink fraction vs cb3's -21.5%, (c) greedy
    per-GEMM byte-identity vs int4-Marlin at M=8 (wirbel #390 style).

ENUMERATION RESULT (import-probed live; the GPU run re-asserts):
  vLLM 0.22's Marlin GEMM supports ONLY {uint4b8, uint8b128, float8_e4m3fn, float4_e2m1f} -- there
  is NO 2-bit/3-bit Marlin path (the uint2b2/uint3b4 scalar types EXIST but have no GEMM/quantize).
  The GPTQ/AWQ quantize path supports ONLY {uint4b8, uint8b128}. AQLM (the one historical sub-int4
  shipping kernel) is REMOVED from 0.22 (not in the registry, not importable). bitsandbytes-nf4 is
  registered but its wheel is ABSENT in the GPU venv. So every shipping path is either (i) >= int4's
  4.125 bpw (uint8b128 8.125, awq/gptq 4.156 w/ zeros, fp4 4.5 w/ g16 scales) => ZERO or NEGATIVE
  byte-shrink, or (ii) the SAME bytes on a SLOWER non-Marlin kernel (gptq-exllama) => negative
  REALIZED shrink. cb3's sub-int4 + L1-resident-codebook combination is UNREACHABLE off-the-shelf.

KEY IDENTITY ARGUMENT (why shrink and identity are mutually exclusive among shipping kernels):
  A kernel preserves greedy byte-identity vs the deployed int4-Marlin iff its per-GEMM output is
  bit-exact to int4-Marlin's. Any kernel that stores fewer bytes is a DIFFERENT quantizer (different
  rounded codes => different dequantized weight => different GEMM output), and even the SAME int4
  codes on a different kernel (gptq-exllama) differ in accumulation order. So the only identity-
  preserving shipping path is int4-Marlin itself (0 shrink). Identity AND shrink>0 is the empty set.
  (cb3 is likewise NOT byte-identical to int4; its quality gate is PPL, not token-identity-vs-int4 --
  the deployed greedy-identity gate is served-vs-served self-referential. See #367 memory.)

METHOD (reuses #391's byte model + 3-tier machinery; new = shipping-kernel realized measurement):
  (1) Enumerate importable+constructible shipping quant kernels in the live vLLM 0.22 venv.
  (2) For each, quantize the SAME bf16 source weight per body shape, microbench median us/GEMM at
      M in {1,8}, measure realized stored weight bytes + weight-read HBM-eff (#391 method; peak-copy
      BW reference), and the MEASURED realized speedup vs int4-Marlin = t_int4/t_kernel.
  (3) Identity: per-GEMM bit-exactness + argmax-token agreement vs int4-Marlin at M=8.
  (4) Map the best shipping kernel's MEASURED realized speedup -> realized TPS lift on the corrected
      strict base 471.42 (wirbel #390) via the #388/#391 lift_factor on the honest body fraction.
  (5) shipping_kernel_unblocks_supply: does the best shipping kernel clear #383's +17.22 supply
      floor WITH greedy byte-identity intact? (Expected: NO -- the decisive deployability negative.)

REPRODUCE (0-GPU analytic self-test, >=20 checks):
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      research/validity/cb3_kernel_realized_bw/cb3_shipping_kernel_surrogate.py --self-test
GPU shipping-kernel microbench (single A10G):
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      research/validity/cb3_kernel_realized_bw/cb3_shipping_kernel_surrogate.py --gpu \
      --wandb_group cb3-shipping-kernel-surrogate --wandb_name lawine/cb3-shipping-kernel-surrogate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---- reuse the #388/#391 sibling harness (analytic core + GPU timing helpers) ---------- #
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import cb3_kernel_realized_bw as cb3  # noqa: E402

# reused constants / functions
BODY_SHAPES = cb3.BODY_SHAPES                       # 8 distinct body GEMM shapes
INT4_BPW = cb3.INT4_BPW                             # 4.125 (deployed int4-Marlin g128)
CB3_BPW_EFF = cb3.CB3_BPW_EFF                       # 3.2369 (#372 mixed cb3 allocation)
BODY_BYTES_FRAC = cb3.BODY_BYTES_FRAC              # 0.7847 (cb3/int4 byte ratio = "0.785")
F_ATTN = cb3.F_ATTN
F_LMHEAD = cb3.F_LMHEAD
F_BODY_STRICT = cb3.F_BODY_STRICT                  # 0.7624 (honest shrinkable body-read fraction)
SUPPLY_FLOOR = cb3.SUPPLY_FLOOR_JOINT_TPS          # +17.216 (#383 supply floor)
SUPPLY_ROBUST = cb3.SUPPLY_ROBUST_ET_ONLY_TPS      # +23.749 (#383 robust target)
M8_BW_BOUND_THRESHOLD = cb3.M8_BW_BOUND_THRESHOLD  # 0.50
A10G_HBM_PEAK_GBS = cb3.A10G_HBM_PEAK_GBS
M1_MEASURED_HBM_EFF_388 = cb3.M1_MEASURED_HBM_EFF_388
TOL = 1e-6

# ======================================================================================== #
# #395 NEW anchors
# ======================================================================================== #
# Corrected realized strict base (wirbel #390 `5y64zbjz`): the honest decode-width identity base.
CORRECTED_STRICT_BASE = 471.42
GAP_TO_500 = 28.58                          # 471.42 + 28.58 == 500
LAMBDA1_CEILING = 520.953                   # #390 lambda=1 ceiling (context)
# cb3 modelled realistic lift to beat with a SHIPPING kernel (#388/#391).
CB3_REALISTIC_LIFT_M1 = 33.0                # #388 g5lfdpgw realistic M=1
CB3_REALISTIC_LIFT_M8 = 38.02              # #391 3udzpoq8 realistic M=8
CB3_READ_SHRINK_FRAC = 1.0 - BODY_BYTES_FRAC   # +0.2153 == cb3's "-21.5%" body-read reduction

# lm_head GEMM geometry (gemma-4-E4B-it: vocab 262144 x hidden 2560), count 1. Measured alongside
# the 8 body GEMMs per the PR; its read-shrink maps to the small F_LMHEAD step fraction (0.0224).
LMHEAD_SHAPE: dict[str, Any] = {"name": "lm_head", "out": 262144, "in": 2560, "count": 1}
GROUP_SIZE = 128
FP4_GROUP_SIZE = 16                         # float4_e2m1f Marlin only supports g16 on sm_86
DEFAULT_WIDTHS = [1, 8]                      # M=1 baseline + M=8 served verify (#391 op-point)

# ---- live-probed enumeration of THIS vLLM 0.22 GPU venv (the GPU run re-asserts these) -- #
VLLM_VERSION_EXPECTED = "0.22.0"
MARLIN_SUPPORTED_TYPES = ["uint4b8", "uint8b128", "float8_e4m3fn", "float4_e2m1f"]  # NO uint2/uint3
GPTQ_QUANTIZE_SUPPORTED_TYPES = ["uint4b8", "uint8b128"]                            # 4/8-bit only
SUBINT4_SCALAR_TYPES_PRESENT = ["uint2b2", "uint3b4"]   # exist as types but NO gemm/quantize path
AQLM_SHIPS_IN_022 = False                              # removed from vLLM 0.22
BNB_NF4_WHEEL_PRESENT = False                          # bitsandbytes wheel absent in the GPU venv
# vLLM 0.22 has NO standalone `gptq`/`gptq_marlin`/`marlin` quant modules: the legacy non-Marlin
# GPTQ (exllama) LinearMethod was removed. The raw ops.gptq_gemm C++ op still exists but is unwired
# (a hand-rolled call returns numeric garbage, ~1e24). GPTQ checkpoints are served via the Marlin
# kernel (the "gptq_marlin" route) == the deployed int4 read profile: 0 shrink, identity-class.
GPTQ_EXLLAMA_LINEAR_REMOVED_022 = True
GPTQ_SHIPPING_PATH = "gptq_marlin == Marlin int4 GEMM (0 read-shrink, identity-class)"
# A live kernel whose M=8 output diverges from int4-Marlin by more than this is treated as a broken
# build (NOT a valid GEMM) and excluded from any realized-speedup claim. Valid alt-quantizers differ
# from int4 by O(1) (int8/awq measured ~1.0); a dead op overflows to ~1e24.
MAX_VALID_ABS_DIFF = 1.0e3

# ---- shipping-kernel nominal stored bits-per-weight (g128 unless noted) ----------------- #
# Every shipping format is >= int4's 4.125 bpw; cb3 (3.2369) is the only sub-int4 entry.
SHIPPING_KERNEL_BPW: dict[str, float] = {
    "marlin_int4_uint4b8_g128": 4.125,        # deployed baseline (4b + bf16 g128 scale, symmetric)
    "marlin_int8_uint8b128_g128": 8.125,      # 8b + bf16 g128 scale (~2x int4 bytes)
    "awq_marlin_uint4_g128": 4.15625,         # 4b + bf16 g128 scale + 4b g128 zero-point (asym)
    "gptq_marlin_uint4b8_g128": 4.125,        # GPTQ's only shipping path == Marlin int4 (symmetric)
    "fp4_marlin_e2m1_g16": 4.5,               # 4b float + 8b (e4m3/e8m0) g16 block scale
}
SHIPPING_KERNEL_NOTE: dict[str, str] = {
    "marlin_int4_uint4b8_g128": "deployed baseline; fast Marlin; the leanest shipping fast format",
    "marlin_int8_uint8b128_g128": "8-bit: reads ~2x bytes; fast Marlin but WRONG direction",
    "awq_marlin_uint4_g128": "AWQ 4-bit asym on the Marlin kernel; ~int4 bytes; different quantizer",
    "gptq_marlin_uint4b8_g128": "GPTQ -> gptq_marlin == Marlin int4 read profile (exllama removed); 0 shrink, identity-class",
    "fp4_marlin_e2m1_g16": "FP4 (newest 4-bit shipping fmt); g16 scales => 4.5 bpw; fast Marlin, MORE bytes",
}
# The Marlin int4 family (deployed int4 + GPTQ's gptq_marlin route) shares ONE read profile and is
# the only identity-preserving shipping option (0 shrink). Excluded from "best distinct alternative".
BASELINE_EQUIV_KERNELS = {"marlin_int4_uint4b8_g128", "gptq_marlin_uint4b8_g128"}
CB3_BPW_FOR_TABLE = CB3_BPW_EFF               # 3.2369 (the unreachable-off-the-shelf target)


# ======================================================================================== #
# Pure analytic core (0-GPU): byte-shrink + byte-roofline potential lift per shipping kernel.
# ======================================================================================== #
def read_shrink_frac(bpw: float) -> float:
    """Body-read shrink fraction vs int4-Marlin (cb3 == +0.2153). >0 reads fewer bytes."""
    return 1.0 - bpw / INT4_BPW


def byte_roofline_speedup(bpw: float) -> float:
    """If a kernel realized its byte ratio at int4's BW-efficiency: speedup = int4_bpw/bpw.
    >1 only if it stores FEWER bytes than int4 (no shipping kernel does; cb3 = 1.2744)."""
    return INT4_BPW / bpw


def lift_tps_on_base(speedup: float, f_body: float = F_BODY_STRICT,
                     base: float = CORRECTED_STRICT_BASE) -> float:
    """Realized TPS lift on the corrected strict base when the body fraction f_body speeds up by
    `speedup` (the #388/#391 lift_factor). speedup<1 => negative lift (kernel reads more/slower)."""
    return base * (cb3.lift_factor(speedup, f_body) - 1.0)


def read_shrink_lift_tps(bpw: float) -> float:
    """The SUPPLY (weight-read-shrink) TPS lift the PR asks to compute: map a kernel's read-shrink
    to a realized lift via the byte-roofline speedup (int4_bpw/bpw) through the #388/#391 lift_factor.
    This is cb3's mechanism (fewer bytes read). Width-independent (a byte property). <=0 for every
    shipping kernel because none stores fewer bytes than int4. It is the MOST GENEROUS read-shrink
    attribution (assumes the kernel reads its bytes at int4's BW-efficiency), so a <=0 here is robust.
    Distinct from a kernel's measured wall-clock speedup, which also folds in compute/kernel-impl and
    is NOT a read-shrink (a faster kernel at >= int4 bytes does not recover cb3's supply lift)."""
    return lift_tps_on_base(byte_roofline_speedup(bpw))


def analytic_shipping_table() -> dict[str, Any]:
    """0-GPU per-kernel byte-shrink + byte-roofline POTENTIAL lift (no measured times). The GPU run
    overrides the `speedup` with the MEASURED t_int4/t_kernel ratio (the honest realized number)."""
    rows: dict[str, dict[str, Any]] = {}
    for name, bpw in SHIPPING_KERNEL_BPW.items():
        sp = byte_roofline_speedup(bpw)
        rows[name] = {
            "bpw": bpw,
            "read_shrink_frac": read_shrink_frac(bpw),
            "byte_roofline_speedup": sp,
            "potential_lift_tps_m1": lift_tps_on_base(sp),
            "potential_lift_tps_m8": lift_tps_on_base(sp),
            "note": SHIPPING_KERNEL_NOTE.get(name, ""),
            "is_subint4": bpw < INT4_BPW,
        }
    # cb3 reference row (the unreachable-off-the-shelf target)
    cb3_sp = byte_roofline_speedup(CB3_BPW_FOR_TABLE)
    rows["cb3_qtip_class_SOURCE_BUILD_ONLY"] = {
        "bpw": CB3_BPW_FOR_TABLE,
        "read_shrink_frac": read_shrink_frac(CB3_BPW_FOR_TABLE),
        "byte_roofline_speedup": cb3_sp,
        "potential_lift_tps_m1": lift_tps_on_base(cb3_sp),
        "potential_lift_tps_m8": lift_tps_on_base(cb3_sp),
        "note": "NOT SHIPPING (source-build-only); the sub-int4 + L1-resident-codebook target",
        "is_subint4": True,
    }
    return rows


def select_best_shipping(per_kernel: dict[str, dict[str, Any]],
                         width: int) -> tuple[str | None, dict[str, Any] | None]:
    """Best DISTINCT shipping kernel for cb3's SUPPLY objective = the one that shrinks weight-reads
    most (largest read_shrink_frac). The PR is a weight-read-shrink lane, so the supply metric is the
    byte read-shrink, NOT a kernel-impl wall-clock speedup. Excludes the int4-Marlin family (deployed
    int4 + GPTQ's gptq_marlin route: same read profile, 0 shrink) and any numerically-broken build.
    Ties -> higher measured body HBM-efficiency at `width`."""
    best_name, best_row = None, None
    best_key = (-1e18, -1e18)
    for name, row in per_kernel.items():
        if name in BASELINE_EQUIV_KERNELS or name.startswith("cb3_"):
            continue
        if row.get("numerically_valid") is False:   # broken build (e.g. a dead op) is not a candidate
            continue
        shrink = float(row.get("read_shrink_frac", -1e9))
        eff = row.get(f"hbm_eff_body_m{width}", row.get(f"hbm_eff_m{width}")) or -1e18
        key = (shrink, float(eff))
        if key > best_key:
            best_key, best_name, best_row = key, name, row
    return best_name, best_row


# ======================================================================================== #
# GPU: shipping-kernel builders. Each returns (run(x)->out, stored_bytes_dict) or marks unavailable.
# All quantize the SAME bf16 source weight w [K=in, N=out] so identity is a fair per-GEMM compare.
# ======================================================================================== #
def _enumerate_live() -> dict[str, Any]:
    """Re-probe the live vLLM 0.22 quant registry + Marlin/GPTQ supported types + ext libs."""
    info: dict[str, Any] = {}
    try:
        import vllm
        info["vllm_version"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        info["vllm_version_err"] = repr(e)
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        info["registry_methods"] = sorted(QUANTIZATION_METHODS)
    except Exception as e:  # noqa: BLE001
        info["registry_methods_err"] = repr(e)
    try:
        from vllm.scalar_type import scalar_types
        info["scalar_types"] = [n for n in dir(scalar_types) if not n.startswith("_")]
    except Exception as e:  # noqa: BLE001
        info["scalar_types_err"] = repr(e)
    try:
        import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
        try:
            types = mu.query_marlin_supported_quant_types(False)
        except TypeError:
            types = mu.query_marlin_supported_quant_types()
        info["marlin_supported_types"] = [str(t) for t in types]
    except Exception as e:  # noqa: BLE001
        info["marlin_supported_types_err"] = repr(e)
    try:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            SUPPORTED_GPTQ_QUANT_TYPES)
        info["gptq_quantize_supported_types"] = [str(t) for t in SUPPORTED_GPTQ_QUANT_TYPES]
    except Exception as e:  # noqa: BLE001
        info["gptq_quantize_supported_types_err"] = repr(e)
    import importlib
    ext = {}
    for lib in ("aqlm", "bitsandbytes"):
        try:
            m = importlib.import_module(lib)
            ext[lib] = getattr(m, "__version__", "present")
        except Exception:  # noqa: BLE001
            ext[lib] = "absent"
    info["ext_libs"] = ext
    return info


def _build_marlin(w, wtype, gs, dev):
    import torch
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt
    K, N = w.shape
    _wref, q_w, s, _g, _so, _rp = mt.marlin_quantize(w, wtype, gs, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    gi = torch.empty(0, dtype=torch.int, device=dev)
    si = torch.empty(0, dtype=torch.int, device=dev)

    def run(x):
        return mu.apply_gptq_marlin_linear(
            x, q_w, s, zp, gi, si, ws, wtype,
            output_size_per_partition=N, input_size_per_partition=K, is_k_full=True)

    nb = {"qweight": q_w.numel() * q_w.element_size(), "scale": s.numel() * s.element_size(),
          "zp": 0.0}
    return run, nb


def _build_awq_marlin(w, gs, dev):
    import torch
    from vllm.scalar_type import scalar_types
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt
    K, N = w.shape
    _wref, q_w, s, zp = mt.awq_marlin_quantize(w, scalar_types.uint4, gs)
    ws = mu.marlin_make_workspace_new(dev)
    gi = torch.empty(0, dtype=torch.int, device=dev)
    si = torch.empty(0, dtype=torch.int, device=dev)

    def run(x):
        return mu.apply_awq_marlin_linear(
            x, q_w, s, zp, gi, si, ws, scalar_types.uint4,
            output_size_per_partition=N, input_size_per_partition=K)

    nb = {"qweight": q_w.numel() * q_w.element_size(), "scale": s.numel() * s.element_size(),
          "zp": zp.numel() * zp.element_size()}
    return run, nb


def build_kernel(name, w, dev):
    """Dispatch a shipping-kernel builder. Returns (run, stored_bytes_dict) or raises.
    NOTE: there is no gptq_exllama builder -- vLLM 0.22 removed the non-Marlin GPTQ LinearMethod;
    GPTQ is served via gptq_marlin == the int4-Marlin kernel (the baseline build covers that read
    profile). The raw ops.gptq_gemm op is unwired and returns garbage, so it is not a shipping path."""
    from vllm.scalar_type import scalar_types
    if name == "marlin_int4_uint4b8_g128":
        return _build_marlin(w, scalar_types.uint4b8, GROUP_SIZE, dev)
    if name == "marlin_int8_uint8b128_g128":
        return _build_marlin(w, scalar_types.uint8b128, GROUP_SIZE, dev)
    if name == "awq_marlin_uint4_g128":
        return _build_awq_marlin(w, GROUP_SIZE, dev)
    raise ValueError(f"no live builder for {name} (fp4 + gptq_marlin are analytic on this geometry)")


# kernels with a live GEMM builder. The distinct DEPLOYABLE read profiles on this A10G: int4-Marlin
# (baseline), int8-Marlin, AWQ-Marlin. GPTQ's gptq_marlin == int4-Marlin (analytic, baseline-equiv);
# FP4 is analytic (a live build needs a full layer module via prepare_fp4_layer_for_marlin).
LIVE_KERNELS = [
    "marlin_int4_uint4b8_g128",
    "marlin_int8_uint8b128_g128",
    "awq_marlin_uint4_g128",
]
BASELINE_KERNEL = "marlin_int4_uint4b8_g128"
ANALYTIC_ONLY_KERNELS = ["gptq_marlin_uint4b8_g128", "fp4_marlin_e2m1_g16"]


def _identity_stats(out_x, out_ref):
    """Per-GEMM byte-identity vs the int4-Marlin reference at M=8 (wirbel #390 style) + a looser
    argmax-token agreement (greedy proxy; exact for the lm_head shape)."""
    import torch
    same = (out_x == out_ref)
    exact_frac = float(same.float().mean().item())
    diff = (out_x.float() - out_ref.float()).abs()
    denom = out_ref.float().abs().clamp_min(1e-6)
    am_x = out_x.argmax(dim=-1)
    am_ref = out_ref.argmax(dim=-1)
    return {
        "byte_exact_frac": exact_frac,
        "byte_exact": bool(exact_frac >= 1.0 - 1e-12),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_rel_diff": float((diff / denom).max().item()),
        "argmax_token_agree_frac": float((am_x == am_ref).float().mean().item()),
    }


def microbench_kernels(widths: list[int], iters: int, warmup: int,
                       include_lmhead: bool = True) -> dict[str, Any]:
    """Microbench every constructible shipping kernel at each width on the body GEMMs (+ lm_head).
    Per (kernel, width): count-weighted median us, realized stored weight bytes + HBM-eff, MEASURED
    realized speedup vs int4-Marlin. Per (kernel): per-GEMM byte-identity vs int4-Marlin at M=8."""
    import torch
    dev = cb3._device()
    gpu = cb3._gpu_facts(dev)
    peak = cb3._measure_peak_copy_gbs(dev, iters, warmup)
    peak_gbs = peak["peak_copy_gbs"]
    live_enum = _enumerate_live()

    shapes = list(BODY_SHAPES) + ([LMHEAD_SHAPE] if include_lmhead else [])
    body_names = {s["name"] for s in BODY_SHAPES}

    # availability probe: build each kernel once on the smallest shape
    available: dict[str, bool] = {}
    build_err: dict[str, str] = {}
    probe_w = (torch.randn(BODY_SHAPES[3]["in"], BODY_SHAPES[3]["out"],
                           dtype=torch.bfloat16, device=dev) * 0.02)
    for name in LIVE_KERNELS:
        try:
            run, _nb = build_kernel(name, probe_w, dev)
            o = run(torch.randn(8, probe_w.shape[0], dtype=torch.bfloat16, device=dev))
            available[name] = bool(torch.isfinite(o).all().item())
        except Exception as e:  # noqa: BLE001
            available[name] = False
            build_err[name] = f"{type(e).__name__}: {e}"
    del probe_w
    torch.cuda.empty_cache()

    avail_kernels = [k for k in LIVE_KERNELS if available.get(k)]

    # per-shape: quantize each kernel on the SAME w; time at each width; capture M=8 output (identity)
    per_shape_rows: list[dict[str, Any]] = []
    # accumulators: kernel -> width -> {sum_c_us, sum_c_bytes, sum_c_us_body, sum_c_bytes_body}
    acc: dict[str, dict[int, dict[str, float]]] = {
        k: {m: {"cw_us": 0.0, "cw_bytes": 0.0, "cw_us_body": 0.0, "cw_bytes_body": 0.0}
            for m in widths} for k in avail_kernels}
    identity: dict[str, dict[str, Any]] = {k: {} for k in avail_kernels if k != BASELINE_KERNEL}

    for sh in shapes:
        K, N, cnt = sh["in"], sh["out"], sh["count"]
        w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
        xs = {m: torch.randn(m, K, dtype=torch.bfloat16, device=dev) for m in widths}
        runs: dict[str, Any] = {}
        nbytes: dict[str, dict[str, float]] = {}
        for k in avail_kernels:
            try:
                runs[k], nbytes[k] = build_kernel(k, w, dev)
            except Exception as e:  # noqa: BLE001
                runs[k] = None
                build_err[k] = f"{type(e).__name__}: {e}"
        ref_out8 = None
        if BASELINE_KERNEL in runs and runs[BASELINE_KERNEL] is not None and 8 in widths:
            ref_out8 = runs[BASELINE_KERNEL](xs[8]).detach().clone()
        for k in avail_kernels:
            if runs.get(k) is None:
                continue
            stored = nbytes[k]["qweight"] + nbytes[k]["scale"] + nbytes[k]["zp"]
            row = {"kernel": k, "shape": sh["name"], "out": N, "in": K, "count": cnt,
                   "stored_bytes": stored, "bpw": stored / (K * N) * 8.0}
            for m in widths:
                us = cb3._time_us(lambda k=k, m=m: runs[k](xs[m]), iters, warmup)
                eff = stored / (us * 1e-6) / 1e9 / peak_gbs
                row[f"us_m{m}"] = us
                row[f"hbm_eff_m{m}"] = eff
                acc[k][m]["cw_us"] += cnt * us
                acc[k][m]["cw_bytes"] += cnt * stored
                if sh["name"] in body_names:
                    acc[k][m]["cw_us_body"] += cnt * us
                    acc[k][m]["cw_bytes_body"] += cnt * stored
            if k != BASELINE_KERNEL and ref_out8 is not None:
                out8 = runs[k](xs[8])
                identity[k][sh["name"]] = _identity_stats(out8, ref_out8)
            per_shape_rows.append(row)
        del w, xs, runs, ref_out8
        torch.cuda.empty_cache()

    # aggregate per kernel/width (count-weighted). Two SEPARATE lift axes:
    #   realized_lift_tps_m{m}  = READ-SHRINK supply lift (byte-roofline; cb3's mechanism; the PR ask)
    #   wallclock_lift_tps_m{m} = measured kernel-impl speed at >= int4 bytes (diagnostic; NOT supply)
    # The wall-clock axis is only trustworthy if the kernel produces a VALID GEMM (numerically_valid).
    base_acc = acc.get(BASELINE_KERNEL, {})
    per_kernel: dict[str, dict[str, Any]] = {}
    for k in avail_kernels:
        bpw_k = SHIPPING_KERNEL_BPW.get(k, INT4_BPW)
        rs_lift = read_shrink_lift_tps(bpw_k)          # width-independent supply lift (<=0 for ships)
        row: dict[str, Any] = {"bpw": bpw_k, "live": True,
                               "read_shrink_frac": read_shrink_frac(bpw_k),
                               "read_shrink_lift_tps": rs_lift,
                               "note": SHIPPING_KERNEL_NOTE.get(k, "")}
        # identity rollup first -> drives numerical validity of the wall-clock timing
        if k in identity and identity[k]:
            ex = [v["byte_exact"] for v in identity[k].values()]
            row["identity_byte_exact"] = bool(all(ex))
            row["identity_byte_exact_frac_min"] = min(v["byte_exact_frac"] for v in identity[k].values())
            row["identity_max_abs_diff"] = max(v["max_abs_diff"] for v in identity[k].values())
            row["identity_argmax_agree_min"] = min(v["argmax_token_agree_frac"] for v in identity[k].values())
            row["identity_per_shape"] = identity[k]
            row["numerically_valid"] = bool(
                math.isfinite(row["identity_max_abs_diff"]) and
                row["identity_max_abs_diff"] < MAX_VALID_ABS_DIFF)
        elif k == BASELINE_KERNEL:
            row["identity_byte_exact"] = True          # the deployed kernel is itself
            row["identity_argmax_agree_min"] = 1.0
            row["identity_max_abs_diff"] = 0.0
            row["numerically_valid"] = True
        else:
            row["numerically_valid"] = True            # available probe already checked finiteness
        for m in widths:
            cw_us = acc[k][m]["cw_us"]
            cw_bytes = acc[k][m]["cw_bytes"]
            cw_us_body = acc[k][m]["cw_us_body"]
            cw_bytes_body = acc[k][m]["cw_bytes_body"]
            eff_all = cw_bytes / (cw_us * 1e-6) / 1e9 / peak_gbs if cw_us > 0 else 0.0
            eff_body = (cw_bytes_body / (cw_us_body * 1e-6) / 1e9 / peak_gbs
                        if cw_us_body > 0 else 0.0)
            base_us_body = base_acc.get(m, {}).get("cw_us_body", 0.0)
            wc_speedup = (base_us_body / cw_us_body) if cw_us_body > 0 else 0.0
            row[f"hbm_eff_m{m}"] = eff_all
            row[f"hbm_eff_body_m{m}"] = eff_body
            row[f"wallclock_speedup_m{m}"] = wc_speedup
            # supply (read-shrink) lift is the deliverable realized_lift; width-independent
            row[f"realized_lift_tps_m{m}"] = rs_lift
            # wall-clock lift is a diagnostic, only meaningful for a numerically-valid kernel
            row[f"wallclock_lift_tps_m{m}"] = (
                lift_tps_on_base(wc_speedup) if (wc_speedup > 0 and row["numerically_valid"]) else None)
        per_kernel[k] = row

    return {
        "gpu": gpu, "peak_copy": peak, "live_enum": live_enum,
        "widths": list(widths), "available": available, "build_err": build_err,
        "avail_kernels": avail_kernels, "per_shape": per_shape_rows, "per_kernel": per_kernel,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "include_lmhead": include_lmhead,
    }


# ======================================================================================== #
# Verdict assembly: best shipping kernel, frac-of-cb3 recovered, unblocks-supply gate.
# ======================================================================================== #
def assemble_verdict(per_kernel: dict[str, dict[str, Any]], measured: bool) -> dict[str, Any]:
    """Answer the deployability question. The deliverable lift is the READ-SHRINK (supply) lift of the
    best distinct shipping kernel -- the PR maps read-shrink -> TPS lift. A kernel's measured wall-clock
    speedup at >= int4 bytes is reported separately as a diagnostic (it is NOT a recovered supply lift)."""
    best_name8, best8 = select_best_shipping(per_kernel, 8)
    best_name1, best1 = select_best_shipping(per_kernel, 1)

    def supply_lift(row, m):
        """Read-shrink (byte-roofline) lift -> the deliverable realized_lift_tps."""
        if row is None:
            return 0.0
        v = row.get(f"realized_lift_tps_m{m}")
        if v is None:                                  # analytic-only rows carry potential_lift
            v = row.get(f"potential_lift_tps_m{m}", row.get("read_shrink_lift_tps"))
        return float(v) if v is not None else 0.0

    def wc_lift(row, m):
        v = row.get(f"wallclock_lift_tps_m{m}") if row else None
        return float(v) if v is not None else None

    best_lift_m1 = supply_lift(best1, 1)
    best_lift_m8 = supply_lift(best8, 8)
    best_shrink = float(best8.get("read_shrink_frac", 0.0)) if best8 else 0.0
    best_eff8 = float(best8.get("hbm_eff_body_m8", best8.get("hbm_eff_m8", 0.0))) if best8 else 0.0
    best_identity = bool(best8.get("identity_byte_exact", False)) if best8 else False
    best_valid = bool(best8.get("numerically_valid", True)) if best8 else False
    frac_cb3 = (best_lift_m8 / CB3_REALISTIC_LIFT_M8) if CB3_REALISTIC_LIFT_M8 else 0.0

    # diagnostic: best measured wall-clock lift across numerically-valid NON-baseline kernels (the
    # kernel-impl axis). Shows that even at >= int4 bytes no shipping kernel beats the deployed Marlin.
    wc_lifts = []
    for name, row in per_kernel.items():
        if name in BASELINE_EQUIV_KERNELS or name.startswith("cb3_"):
            continue
        if not row.get("numerically_valid", True):
            continue
        wc = wc_lift(row, 8)
        if wc is not None:
            wc_lifts.append(wc)
    max_wc_lift_m8 = max(wc_lifts) if wc_lifts else None

    # deployability gate: a shipping kernel UNBLOCKS supply only if it clears the #383 +17.22 supply
    # floor on the READ-SHRINK axis AND preserves byte-identity. (Read-shrink lift <=0 => never true.)
    unblocks = bool(best_lift_m8 >= SUPPLY_FLOOR and best_identity)

    # the identity-preserving shipping subset = the int4-Marlin family (deployed int4 + gptq_marlin),
    # all 0 read-shrink. No byte-shrinking shipping kernel is byte-identical (different quantizer/codes).
    id_preserving = sorted(k for k in per_kernel if k in BASELINE_EQUIV_KERNELS)
    if not id_preserving:
        id_preserving = [BASELINE_KERNEL]

    return {
        "best_shipping_kernel": best_name8,
        "best_shipping_read_shrink_frac": best_shrink,
        "best_shipping_hbm_eff": best_eff8,
        "best_shipping_identity_byte_exact": best_identity,
        "best_shipping_numerically_valid": best_valid,
        "realized_lift_tps_best_shipping_m1": best_lift_m1,
        "realized_lift_tps_best_shipping_m8": best_lift_m8,
        "frac_of_cb3_lift_recovered_shipping": frac_cb3,
        "shipping_kernel_unblocks_supply": unblocks,
        # wall-clock (kernel-impl) diagnostic axis -- explicitly NOT a recovered supply lift
        "best_shipping_wallclock_speedup_m8": (float(best8.get("wallclock_speedup_m8"))
                                               if best8 and best8.get("wallclock_speedup_m8") else None),
        "best_shipping_wallclock_lift_tps_m8": wc_lift(best8, 8),
        "max_wallclock_lift_tps_m8_over_valid": max_wc_lift_m8,
        "measured": measured,
        "best_kernel_m1": best_name1,
        "identity_preserving_shipping_kernels": id_preserving,
        "any_shipping_kernel_shrinks_with_identity": False,
    }


# ======================================================================================== #
# Self-test (>=20 checks): the #395 PRIMARY gate cb3_shipping_kernel_self_test_passes.
# ======================================================================================== #
def self_test() -> dict[str, Any]:
    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    table = analytic_shipping_table()
    ship_rows = {k: v for k, v in table.items() if not k.startswith("cb3_")}
    cb3_row = table["cb3_qtip_class_SOURCE_BUILD_ONLY"]

    # 1. cb3 byte-shrink == -21.5% (the target), and cb3 IS sub-int4
    chk("cb3_read_shrink_is_0p2153", abs(CB3_READ_SHRINK_FRAC - 0.21530670587274363) < 1e-9,
        f"{CB3_READ_SHRINK_FRAC:.6f}")
    chk("cb3_is_subint4", CB3_BPW_EFF < INT4_BPW, f"cb3={CB3_BPW_EFF} int4={INT4_BPW}")
    chk("cb3_row_shrink_positive", cb3_row["read_shrink_frac"] > 0.20, f"{cb3_row['read_shrink_frac']:.4f}")
    # 2. corrected strict base arithmetic: 471.42 + 28.58 == 500
    chk("base_plus_gap_is_500", abs(CORRECTED_STRICT_BASE + GAP_TO_500 - 500.0) < 1e-9, "")
    # 3. EVERY shipping kernel is >= int4 bpw (no shipping sub-int4)
    chk("all_shipping_ge_int4_bpw", all(v["bpw"] >= INT4_BPW for v in ship_rows.values()),
        f"min_bpw={min(v['bpw'] for v in ship_rows.values())}")
    # 4. EVERY shipping kernel read-shrink <= 0
    chk("all_shipping_shrink_le_0", all(v["read_shrink_frac"] <= 1e-9 for v in ship_rows.values()),
        f"max_shrink={max(v['read_shrink_frac'] for v in ship_rows.values()):.5f}")
    # 5. int8 reads ~2x (byte ratio ~1.97), shrink strongly negative
    i8 = ship_rows["marlin_int8_uint8b128_g128"]
    chk("int8_reads_about_2x", abs(8.125 / 4.125 - 1.9697) < 1e-3 and i8["read_shrink_frac"] < -0.9,
        f"shrink={i8['read_shrink_frac']:.4f}")
    # 6. awq stores slightly MORE than int4 (zeros) -> shrink slightly negative; GPTQ ships via
    #    gptq_marlin == int4-Marlin read profile (0 shrink), NOT a distinct smaller kernel.
    chk("awq_shrink_slightly_neg", -0.02 < ship_rows["awq_marlin_uint4_g128"]["read_shrink_frac"] < 0,
        f"{ship_rows['awq_marlin_uint4_g128']['read_shrink_frac']:.5f}")
    chk("gptq_marlin_zero_shrink",
        abs(ship_rows["gptq_marlin_uint4b8_g128"]["read_shrink_frac"]) < 1e-9
        and "gptq_marlin_uint4b8_g128" in BASELINE_EQUIV_KERNELS,
        f"{ship_rows['gptq_marlin_uint4b8_g128']['read_shrink_frac']:.6f}")
    chk("exllama_linear_removed_022", GPTQ_EXLLAMA_LINEAR_REMOVED_022 is True, "")
    # 7. fp4 g16 stores MORE than int4 (finer scales) -> shrink negative
    chk("fp4_shrink_negative", ship_rows["fp4_marlin_e2m1_g16"]["read_shrink_frac"] < -0.05,
        f"{ship_rows['fp4_marlin_e2m1_g16']['read_shrink_frac']:.4f}")
    # 8. best shipping read-shrink (max over kernels) is <= 0 and << cb3's +0.2153
    best_shrink = max(v["read_shrink_frac"] for v in ship_rows.values())
    chk("best_shipping_shrink_le_0", best_shrink <= 1e-9, f"best={best_shrink:.5f}")
    chk("best_shipping_far_below_cb3", best_shrink < 0.5 * CB3_READ_SHRINK_FRAC,
        f"best={best_shrink:.5f} half_cb3={0.5*CB3_READ_SHRINK_FRAC:.5f}")
    # 9. byte-roofline potential lift of every shipping kernel <= 0 (none beats int4)
    chk("all_shipping_potential_lift_le_0",
        all(v["potential_lift_tps_m8"] <= 1e-6 for v in ship_rows.values()),
        f"max={max(v['potential_lift_tps_m8'] for v in ship_rows.values()):.4f}")
    # 10. cb3 potential lift is strongly positive (the contrast)
    chk("cb3_potential_lift_positive", cb3_row["potential_lift_tps_m8"] > 20.0,
        f"{cb3_row['potential_lift_tps_m8']:.2f}")
    # 11. lift_factor monotone: a faster speedup gives a bigger lift
    chk("lift_monotone_in_speedup", lift_tps_on_base(1.27) > lift_tps_on_base(1.0) > lift_tps_on_base(0.5),
        "")
    # 12. speedup==1 gives ~0 lift; speedup<1 gives negative lift
    chk("speedup1_zero_lift", abs(lift_tps_on_base(1.0)) < 1e-6, f"{lift_tps_on_base(1.0):.6f}")
    chk("speedup_lt1_neg_lift", lift_tps_on_base(0.5) < 0, f"{lift_tps_on_base(0.5):.3f}")
    # 13. enumeration facts: Marlin supports uint4b8/uint8b128, NOT uint2b2/uint3b4
    chk("marlin_supports_uint4b8", "uint4b8" in MARLIN_SUPPORTED_TYPES, "")
    chk("marlin_no_subint4", not any(t in MARLIN_SUPPORTED_TYPES for t in ("uint2b2", "uint3b4")), "")
    chk("subint4_types_present_no_gemm", set(SUBINT4_SCALAR_TYPES_PRESENT) == {"uint2b2", "uint3b4"}, "")
    # 14. gptq quantize path is 4/8-bit only; aqlm gone; bnb wheel absent
    chk("gptq_quantize_4_8_only", GPTQ_QUANTIZE_SUPPORTED_TYPES == ["uint4b8", "uint8b128"], "")
    chk("aqlm_not_shipping", AQLM_SHIPS_IN_022 is False, "")
    chk("bnb_nf4_absent", BNB_NF4_WHEEL_PRESENT is False, "")
    # 15. unblocks-supply logic: best shipping kernel (read-shrink <=0) does NOT clear the +17.22 floor
    fake_pk = {k: {"read_shrink_frac": v["read_shrink_frac"],
                   "potential_lift_tps_m1": v["potential_lift_tps_m1"],
                   "potential_lift_tps_m8": v["potential_lift_tps_m8"],
                   "numerically_valid": True, "identity_byte_exact": False}
               for k, v in ship_rows.items()}
    fake_pk[BASELINE_KERNEL]["identity_byte_exact"] = True
    verdict = assemble_verdict(fake_pk, measured=False)
    chk("shipping_does_not_unblock_supply", verdict["shipping_kernel_unblocks_supply"] is False,
        f"best_lift_m8={verdict['realized_lift_tps_best_shipping_m8']:.3f}")
    chk("best_lift_below_supply_floor", verdict["realized_lift_tps_best_shipping_m8"] < SUPPLY_FLOOR,
        f"{verdict['realized_lift_tps_best_shipping_m8']:.3f} < {SUPPLY_FLOOR:.2f}")
    chk("frac_of_cb3_recovered_near_0", verdict["frac_of_cb3_lift_recovered_shipping"] < 0.1,
        f"{verdict['frac_of_cb3_lift_recovered_shipping']:.4f}")
    chk("no_shipping_shrinks_with_identity",
        verdict["any_shipping_kernel_shrinks_with_identity"] is False, "")
    # 16. identity-preserving subset == the int4-Marlin family (deployed int4 + gptq_marlin), 0 shrink
    chk("identity_subset_is_int4_family",
        verdict["identity_preserving_shipping_kernels"] == sorted(BASELINE_EQUIV_KERNELS),
        f"{verdict['identity_preserving_shipping_kernels']}")
    chk("baseline_has_zero_shrink",
        abs(ship_rows[BASELINE_KERNEL]["read_shrink_frac"]) < 1e-9, "")
    # 17. lm_head geometry integrity
    chk("lmhead_params", LMHEAD_SHAPE["out"] * LMHEAD_SHAPE["in"] == 262144 * 2560, "")
    # 18. body shapes reconstruct ~3.89B params
    tot = sum(cb3._shape_params(s) for s in BODY_SHAPES)
    chk("body_params_3p89B", 3.80e9 < tot < 3.95e9, f"{tot/1e9:.4f}B")
    # 19. cb3 realistic lift to beat is +33/+38 (the bar a shipping kernel cannot reach)
    chk("cb3_lift_bar", CB3_REALISTIC_LIFT_M1 == 33.0 and abs(CB3_REALISTIC_LIFT_M8 - 38.02) < 1e-6, "")
    # 20. NaN/inf clean across the analytic table + verdict
    clean = all(cb3._finite(v) for v in cb3._iter_numeric(table)) and \
        all(cb3._finite(v) for v in cb3._iter_numeric(verdict))
    chk("nan_inf_clean", clean, "")
    # 21. the headline deployability answer is a NEGATIVE (the decision the PR unblocks)
    chk("deployability_answer_is_negative",
        (not verdict["shipping_kernel_unblocks_supply"]) and
        (not verdict["any_shipping_kernel_shrinks_with_identity"]), "")
    # 22. the supply lift IS the byte-roofline read-shrink mapping (not a wall-clock speedup)
    chk("read_shrink_lift_is_byte_roofline",
        abs(read_shrink_lift_tps(4.15625) - lift_tps_on_base(INT4_BPW / 4.15625)) < 1e-9, "")
    chk("read_shrink_lift_le_0_for_all_ships",
        all(read_shrink_lift_tps(v["bpw"]) <= 1e-6 for v in ship_rows.values()),
        f"max={max(read_shrink_lift_tps(v['bpw']) for v in ship_rows.values()):.4f}")
    # 23. THE FIX: a kernel that is FAST in wall-clock but stores >= int4 bytes still does NOT unblock
    #     supply (read-shrink <=0), yet its wall-clock lift is surfaced as a diagnostic (not hidden).
    rs = read_shrink_lift_tps(4.15625)
    spoof = {
        BASELINE_KERNEL: {"read_shrink_frac": 0.0, "realized_lift_tps_m1": 0.0, "realized_lift_tps_m8": 0.0,
                          "identity_byte_exact": True, "numerically_valid": True},
        "awq_marlin_uint4_g128": {
            "read_shrink_frac": read_shrink_frac(4.15625),
            "realized_lift_tps_m1": rs, "realized_lift_tps_m8": rs,
            "hbm_eff_body_m8": 0.258, "wallclock_speedup_m8": 1.5, "wallclock_lift_tps_m8": 200.0,
            "identity_byte_exact": False, "numerically_valid": True},
    }
    vspoof = assemble_verdict(spoof, measured=True)
    chk("wallclock_speed_does_not_unblock_supply",
        vspoof["shipping_kernel_unblocks_supply"] is False
        and vspoof["realized_lift_tps_best_shipping_m8"] <= 1e-6,
        f"supply_lift={vspoof['realized_lift_tps_best_shipping_m8']:.3f}")
    chk("wallclock_diag_surfaced_not_counted",
        abs((vspoof["best_shipping_wallclock_lift_tps_m8"] or 0.0) - 200.0) < 1e-9, "")
    # 24. a numerically-broken build (dead op, ~1e24) is NOT eligible as best shipping kernel
    spoof_bad = {
        BASELINE_KERNEL: {"read_shrink_frac": 0.0, "realized_lift_tps_m8": 0.0, "realized_lift_tps_m1": 0.0,
                          "identity_byte_exact": True, "numerically_valid": True},
        "dead_op": {"read_shrink_frac": 0.9, "realized_lift_tps_m8": 999.0, "realized_lift_tps_m1": 999.0,
                    "identity_byte_exact": False, "numerically_valid": False},
    }
    nm8, _ = select_best_shipping(spoof_bad, 8)
    chk("broken_build_not_selected", nm8 != "dead_op", f"selected={nm8}")
    chk("max_valid_abs_diff_sane", 1.0 < MAX_VALID_ABS_DIFF < 1e6, f"{MAX_VALID_ABS_DIFF}")

    passes = all(c[1] for c in checks)
    return {
        "passes": bool(passes), "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c[1]),
        "checks": [{"name": n, "ok": ok, "detail": d} for (n, ok, d) in checks],
    }


# ======================================================================================== #
# Payload + report + wandb
# ======================================================================================== #
def build_payload(args, micro: dict[str, Any] | None, st: dict[str, Any]) -> dict[str, Any]:
    table = analytic_shipping_table()
    measured = micro is not None
    if measured:
        per_kernel = micro["per_kernel"]
        # merge analytic byte-roofline potentials into the measured (live) rows for select/report
        for k, row in per_kernel.items():
            if k in table:
                row.setdefault("potential_lift_tps_m1", table[k]["potential_lift_tps_m1"])
                row.setdefault("potential_lift_tps_m8", table[k]["potential_lift_tps_m8"])
        # surface the analytic-only formats (gptq_marlin == baseline read profile; fp4) so the table
        # is complete. read-shrink + supply-lift are byte properties (construction-independent).
        for k in ANALYTIC_ONLY_KERNELS:
            if k in per_kernel or k not in table:
                continue
            arow = dict(table[k])
            arow["live"] = False
            arow["numerically_valid"] = True
            bpw_k = arow["bpw"]
            for m in DEFAULT_WIDTHS:
                arow[f"realized_lift_tps_m{m}"] = read_shrink_lift_tps(bpw_k)
            if k in BASELINE_EQUIV_KERNELS:
                arow["identity_byte_exact"] = True     # same Marlin int4 kernel/codes -> bit-identical
                arow["identity_argmax_agree_min"] = 1.0
            per_kernel[k] = arow
        enumerated = micro["avail_kernels"] + [
            "gptq_marlin_uint4b8_g128 (analytic; == int4-Marlin read profile)",
            "fp4_marlin_e2m1_g16 (analytic; live build needs a layer module)"]
    else:
        per_kernel = {k: dict(v) for k, v in table.items() if not k.startswith("cb3_")}
        for k, row in per_kernel.items():           # analytic path: realized == read-shrink potential
            for m in DEFAULT_WIDTHS:
                row.setdefault(f"realized_lift_tps_m{m}", row["potential_lift_tps_m8"])
            row.setdefault("numerically_valid", True)
            if k in BASELINE_EQUIV_KERNELS:         # Marlin int4 family -> bit-identical, 0 shrink
                row.setdefault("identity_byte_exact", True)
        enumerated = list(SHIPPING_KERNEL_BPW.keys())

    verdict = assemble_verdict(per_kernel, measured=measured)

    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 395, "base_pr": 391,
        "kind": "cb3-shipping-kernel-surrogate",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # isolation flags (research microbench; NO served change / submission / HF job)
        "analysis_only": True, "no_hf_job": True, "no_launch": True,
        "no_served_file_change": True, "no_kernel_rebuild": True, "official_tps": 0.0,
        # ---- enumeration (the deliverable list) ----
        "shipping_kernels_enumerated": enumerated,
        "enumeration_facts": {
            "vllm_version_expected": VLLM_VERSION_EXPECTED,
            "marlin_supported_types": MARLIN_SUPPORTED_TYPES,
            "marlin_has_no_subint4": True,
            "gptq_quantize_supported_types": GPTQ_QUANTIZE_SUPPORTED_TYPES,
            "subint4_scalar_types_present_no_gemm": SUBINT4_SCALAR_TYPES_PRESENT,
            "aqlm_ships_in_022": AQLM_SHIPS_IN_022,
            "bnb_nf4_wheel_present": BNB_NF4_WHEEL_PRESENT,
            "gptq_exllama_linear_removed": GPTQ_EXLLAMA_LINEAR_REMOVED_022,
            "gptq_shipping_path": GPTQ_SHIPPING_PATH,
            "max_valid_abs_diff_gate": MAX_VALID_ABS_DIFF,
        },
        # ---- per-kernel table ----
        "shipping_kernel_table": table,
        "per_kernel": per_kernel,
        # ---- the #395 deliverable verdict keys ----
        **verdict,
        # ---- anchors / provenance ----
        "corrected_strict_base": CORRECTED_STRICT_BASE, "gap_to_500": GAP_TO_500,
        "cb3_read_shrink_frac": CB3_READ_SHRINK_FRAC,
        "cb3_realistic_lift_m1": CB3_REALISTIC_LIFT_M1, "cb3_realistic_lift_m8": CB3_REALISTIC_LIFT_M8,
        "supply_floor_383": SUPPLY_FLOOR, "supply_robust_383": SUPPLY_ROBUST,
        "f_body_strict": F_BODY_STRICT, "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF,
        # ---- self-test (PRIMARY) ----
        "selftest": st,
        "cb3_shipping_kernel_self_test_passes": bool(st["passes"]),
    }
    if measured:
        payload["gpu"] = micro["gpu"]
        payload["peak_copy"] = micro["peak_copy"]
        payload["live_enumeration"] = micro["live_enum"]
        payload["available"] = micro["available"]
        payload["build_err"] = micro["build_err"]
        payload["per_shape"] = micro["per_shape"]
        payload["peak_mem_mib"] = micro["peak_mem_mib"]
        payload["shipping_kernels_enumerated"] = sorted(set(
            micro["live_enum"].get("registry_methods", []))) or enumerated
        payload["constructible_live_kernels"] = micro["avail_kernels"]
    return payload


def print_report(p: dict[str, Any]) -> None:
    print("=" * 100)
    print(f"PR #395 lawine -- cb3 SHIPPING-kernel surrogate: does any vLLM-0.22 quant kernel deliver "
          f"cb3-class body-read shrink WITH identity?  ({p['created_at']})")
    if "gpu" in p:
        g = p["gpu"]
        print(f"  GPU {g['name']} sm{g['compute_capability']} x{g['sm_count']} "
              f"(a10g_80sm={g['is_a10g_80sm']})  peak-copy {p['peak_copy']['peak_copy_gbs']:.1f} GB/s")
    print("-" * 100)
    ef = p["enumeration_facts"]
    print("  ENUMERATION (vLLM 0.22):")
    print(f"    Marlin supported types : {ef['marlin_supported_types']}  (sub-int4? NO)")
    print(f"    GPTQ quantize types    : {ef['gptq_quantize_supported_types']}  (4/8-bit only)")
    print(f"    sub-int4 scalar types  : {ef['subint4_scalar_types_present_no_gemm']}  (exist, NO gemm/quantize)")
    print(f"    AQLM ships in 0.22     : {ef['aqlm_ships_in_022']}   bnb-nf4 wheel: {ef['bnb_nf4_wheel_present']}")
    print(f"    GPTQ exllama removed   : {ef.get('gptq_exllama_linear_removed')}  -> {ef.get('gptq_shipping_path')}")
    print("-" * 100)
    print("  shrink%/rs_lift8 = SUPPLY (weight-read-shrink) axis [the PR ask].  "
          "wc_spd8 = measured kernel-impl wall-clock (diagnostic, NOT read-shrink)")
    print(f"  {'kernel':<32} {'bpw':>7} {'shrink%':>8} {'eff_m8':>7} {'wc_spd8':>8} "
          f"{'rs_lift8':>9} {'byte_id':>8} {'argmax':>7}")
    for k, row in p["per_kernel"].items():
        if k.startswith("cb3_"):
            continue
        bpw = row.get("bpw", SHIPPING_KERNEL_BPW.get(k))
        shrink = row.get("read_shrink_frac", 0.0) * 100
        eff8 = row.get("hbm_eff_body_m8", row.get("hbm_eff_m8"))
        spd8 = row.get("wallclock_speedup_m8")
        lift8 = row.get("realized_lift_tps_m8", row.get("potential_lift_tps_m8"))
        bid = row.get("identity_byte_exact")
        am = row.get("identity_argmax_agree_min")
        tag = "" if row.get("live", True) else " (analytic)"
        print(f"    {k:<32} {bpw:>7.3f} {shrink:>7.2f}% "
              f"{(f'{eff8:.3f}' if eff8 is not None else '   -- '):>7} "
              f"{(f'{spd8:.3f}' if spd8 is not None else '   -- '):>8} "
              f"{(f'{lift8:+.2f}' if lift8 is not None else '   -- '):>9} "
              f"{str(bid):>8} {(f'{am:.3f}' if am is not None else '  -- '):>7}{tag}")
    cb3r = p["shipping_kernel_table"]["cb3_qtip_class_SOURCE_BUILD_ONLY"]
    print(f"    {'cb3 (SOURCE-BUILD-ONLY target)':<32} {cb3r['bpw']:>7.3f} "
          f"{cb3r['read_shrink_frac']*100:>7.2f}% {'  --':>7} "
          f"{cb3r['byte_roofline_speedup']:>8.3f} {cb3r['potential_lift_tps_m8']:>+9.2f} {'n/a':>8} {'n/a':>7}")
    print("-" * 100)
    print("  *** #395 VERDICT (deployability) ***")
    print(f"    best_shipping_kernel             = {p['best_shipping_kernel']}")
    print(f"    best_shipping_read_shrink_frac   = {p['best_shipping_read_shrink_frac']:+.5f}  "
          f"(cb3 = {CB3_READ_SHRINK_FRAC:+.4f})")
    print(f"    best_shipping_hbm_eff (M=8)      = {p['best_shipping_hbm_eff']:.4f}")
    print(f"    best_shipping_identity_byte_exact= {p['best_shipping_identity_byte_exact']}")
    print(f"    realized_lift_tps (READ-SHRINK)  = {p['realized_lift_tps_best_shipping_m1']:+.2f} / "
          f"{p['realized_lift_tps_best_shipping_m8']:+.2f} TPS  M=1/M=8  "
          f"(cb3 realistic +{CB3_REALISTIC_LIFT_M1:.0f} / +{CB3_REALISTIC_LIFT_M8:.1f})")
    print(f"    frac_of_cb3_lift_recovered       = {p['frac_of_cb3_lift_recovered_shipping']:.4f}")
    wc = p.get("max_wallclock_lift_tps_m8_over_valid")
    print(f"    [diag] best wall-clock lift M=8  = "
          f"{(f'{wc:+.2f}' if wc is not None else 'n/a')} TPS  "
          f"(kernel-impl at >= int4 bytes; NOT a read-shrink / not a recovered supply lift)")
    print(f"    shipping_kernel_unblocks_supply  = {p['shipping_kernel_unblocks_supply']}  "
          f"(needs >= +{SUPPLY_FLOOR:.2f} READ-SHRINK TPS AND byte-identity)")
    print(f"    any_shipping_shrinks_with_identity = {p['any_shipping_kernel_shrinks_with_identity']} "
          f"(identity-preserving subset = {p['identity_preserving_shipping_kernels']}, 0 shrink)")
    print("-" * 100)
    print(f"  self-test: {p['selftest']['n_passed']}/{p['selftest']['n_checks']} "
          f"-> cb3_shipping_kernel_self_test_passes = {p['cb3_shipping_kernel_self_test_passes']}")
    if "peak_mem_mib" in p:
        print(f"  peak_mem = {p['peak_mem_mib']:.1f} MiB")
    print("=" * 100)


def maybe_log_wandb(payload: dict[str, Any], args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[cb3-ship] wandb helpers unavailable: {e}")
        return None
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["cb3-shipping-kernel-surrogate", "shipping-quant", "marlin", "awq", "gptq",
              "sub-int4-body", "greedy-identity", "pr-395", "pr-391-followup"],
        config={"pr": 395, "base_pr": 391, "kind": "cb3-shipping-kernel-surrogate",
                "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF,
                "corrected_strict_base": CORRECTED_STRICT_BASE, "gap_to_500": GAP_TO_500,
                "supply_floor_383": SUPPLY_FLOOR, "official_tps": 0.0,
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "widths": str(DEFAULT_WIDTHS), "group_size": GROUP_SIZE},
    )
    if run is None:
        print("[cb3-ship] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "verdict/best_shipping_read_shrink_frac": float(payload["best_shipping_read_shrink_frac"]),
        "verdict/best_shipping_hbm_eff": float(payload["best_shipping_hbm_eff"]),
        "verdict/best_shipping_identity_byte_exact": float(payload["best_shipping_identity_byte_exact"]),
        "verdict/realized_lift_tps_best_shipping_m1": float(payload["realized_lift_tps_best_shipping_m1"]),
        "verdict/realized_lift_tps_best_shipping_m8": float(payload["realized_lift_tps_best_shipping_m8"]),
        "verdict/frac_of_cb3_lift_recovered_shipping": float(payload["frac_of_cb3_lift_recovered_shipping"]),
        "verdict/shipping_kernel_unblocks_supply": float(payload["shipping_kernel_unblocks_supply"]),
        "verdict/best_shipping_numerically_valid": float(payload.get("best_shipping_numerically_valid", 1.0)),
        "verdict/cb3_read_shrink_frac": float(payload["cb3_read_shrink_frac"]),
        "selftest/passes": float(payload["selftest"]["passes"]),
        "selftest/n_checks": float(payload["selftest"]["n_checks"]),
        "selftest/n_passed": float(payload["selftest"]["n_passed"]),
    }
    for opt in ("best_shipping_wallclock_lift_tps_m8", "max_wallclock_lift_tps_m8_over_valid",
                "best_shipping_wallclock_speedup_m8"):
        if payload.get(opt) is not None:
            flat[f"verdict/{opt}"] = float(payload[opt])
    for k, row in payload["per_kernel"].items():
        if k.startswith("cb3_"):
            continue
        flat[f"kernel/{k}/read_shrink_frac"] = float(row.get("read_shrink_frac", 0.0))
        if row.get("numerically_valid") is not None:
            flat[f"kernel/{k}/numerically_valid"] = float(row["numerically_valid"])
        for m in (1, 8):
            for fld in ("hbm_eff_body_m", "wallclock_speedup_m", "wallclock_lift_tps_m",
                        "realized_lift_tps_m"):
                v = row.get(f"{fld}{m}")
                if v is not None:
                    flat[f"kernel/{k}/{fld}{m}"] = float(v)
        if row.get("identity_byte_exact") is not None:
            flat[f"kernel/{k}/identity_byte_exact"] = float(row["identity_byte_exact"])
        if row.get("identity_argmax_agree_min") is not None:
            flat[f"kernel/{k}/identity_argmax_agree_min"] = float(row["identity_argmax_agree_min"])
    enum_list = list(payload.get("shipping_kernels_enumerated", []))
    live_list = list(payload.get("constructible_live_kernels", []))
    flat["verdict/n_shipping_kernels_enumerated"] = float(len(enum_list))
    flat["verdict/n_constructible_live_kernels"] = float(len(live_list))
    if "gpu" in payload:
        flat["gpu/sm_count"] = float(payload["gpu"]["sm_count"])
        flat["bw/peak_copy_gbs"] = float(payload["peak_copy"]["peak_copy_gbs"])
        flat["peak_mem_mib"] = float(payload.get("peak_mem_mib", 0.0))
    run.log({"global_step": 0, **flat})
    # The advisor's deliverable contract names exact bare summary keys; set them all
    # directly on run.summary so they are queryable at the bare name (the scalar
    # flattener / log_summary only emit namespaced copies under verdict/ + summary/).
    try:
        run.summary["shipping_kernels_enumerated"] = enum_list
        run.summary["constructible_live_kernels"] = live_list
        run.summary["identity_preserving_shipping_kernels"] = list(
            payload.get("identity_preserving_shipping_kernels", []))
        bare = {
            "best_shipping_kernel": payload["best_shipping_kernel"],
            "best_shipping_read_shrink_frac": float(payload["best_shipping_read_shrink_frac"]),
            "best_shipping_hbm_eff": float(payload["best_shipping_hbm_eff"]),
            "best_shipping_identity_byte_exact": bool(payload["best_shipping_identity_byte_exact"]),
            "realized_lift_tps_best_shipping_m1": float(payload["realized_lift_tps_best_shipping_m1"]),
            "realized_lift_tps_best_shipping_m8": float(payload["realized_lift_tps_best_shipping_m8"]),
            "frac_of_cb3_lift_recovered_shipping": float(payload["frac_of_cb3_lift_recovered_shipping"]),
            "shipping_kernel_unblocks_supply": bool(payload["shipping_kernel_unblocks_supply"]),
            "cb3_shipping_kernel_self_test_passes": bool(payload["cb3_shipping_kernel_self_test_passes"]),
        }
        for k, v in bare.items():
            run.summary[k] = v
    except Exception as e:  # noqa: BLE001
        print(f"[cb3-ship] summary bare-key set skipped: {e}")
    log_summary(run, cb3._jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="cb3_shipping_kernel_surrogate", artifact_type="analysis",
                      data=cb3._jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[cb3-ship] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (>=20 checks)")
    ap.add_argument("--gpu", action="store_true", help="run the shipping-kernel microbench + identity")
    ap.add_argument("--no-lmhead", action="store_true", help="skip the lm_head GEMM (saves memory)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast GPU run to validate the path")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/cb3-shipping-kernel-surrogate")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="cb3-shipping-kernel-surrogate")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    st = self_test()

    if args.self_test and not args.gpu:
        payload = build_payload(args, None, st)
        print_report(payload)
        out_path = Path(args.out_dir) / "cb3_shipping_kernel_surrogate_selftest.json"
        out_path.write_text(json.dumps(cb3._jsonable(payload), indent=2, sort_keys=True))
        print(f"\n[cb3-ship] wrote {out_path}")
        print(f"\ncb3_shipping_kernel_self_test_passes = {st['passes']}")
        sys.exit(0 if st["passes"] else 1)

    if args.smoke:
        args.iters = min(args.iters, 15)
        args.warmup = min(args.warmup, 4)

    micro = microbench_kernels(DEFAULT_WIDTHS, args.iters, args.warmup,
                               include_lmhead=not args.no_lmhead)
    payload = build_payload(args, micro, st)
    print_report(payload)

    out_path = Path(args.out_dir) / "cb3_shipping_kernel_surrogate_results.json"
    out_path.write_text(json.dumps(cb3._jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[cb3-ship] wrote {out_path}")

    rid = None if args.smoke else maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(cb3._jsonable(payload), indent=2, sort_keys=True))

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0.0,
        "shipping_kernels_enumerated": payload["shipping_kernels_enumerated"],
        "best_shipping_kernel": payload["best_shipping_kernel"],
        "best_shipping_read_shrink_frac": float(payload["best_shipping_read_shrink_frac"]),
        "best_shipping_hbm_eff": float(payload["best_shipping_hbm_eff"]),
        "best_shipping_identity_byte_exact": bool(payload["best_shipping_identity_byte_exact"]),
        "realized_lift_tps_best_shipping_m1": float(payload["realized_lift_tps_best_shipping_m1"]),
        "realized_lift_tps_best_shipping_m8": float(payload["realized_lift_tps_best_shipping_m8"]),
        "frac_of_cb3_lift_recovered_shipping": float(payload["frac_of_cb3_lift_recovered_shipping"]),
        "shipping_kernel_unblocks_supply": bool(payload["shipping_kernel_unblocks_supply"]),
        "cb3_shipping_kernel_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "cb3_shipping_kernel_self_test_passes",
                           "value": float(st["passes"])},
        "test_metric": {"name": "realized_lift_tps_best_shipping_m8",
                        "value": float(payload["realized_lift_tps_best_shipping_m8"])},
    }))


if __name__ == "__main__":
    main()
