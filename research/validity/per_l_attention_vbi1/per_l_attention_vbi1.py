#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #389 -- PIN the #386 pessimistic breach: GPU per-L attention latency under VBI=1 (#319).

WHAT THIS ANSWERS
-----------------
My #386 (`xxzujn7a`, MERGED b0de7eb) found the 0.633% irreducible gap floor inflates 2.07x ->
1.310% central under the deployable-strict `VLLM_BATCH_INVARIANT=1` stack, and -- decisively --
that the PESSIMISTIC corner BREACHES the 3.2% knife-edge at 3.5235% (-0.32pp). I flagged that
breach as resting on the single weakest input: the **interpolated** local attention penalty slope
on L in [528,658], taken from #375's CONSERVATIVE [528,2048] anchor segment rather than MEASURED
at the operating point. This card runs my own follow-up #1 -- the decisive hardening step: a direct
GPU per-L attention-latency measurement under VBI=1 that PINS the local penalty slope and converts
the breach from "slope-interpolated" to "measured".

THE PHYSICS BEING PINNED (un-pack == VBI=1 attention)
-----------------------------------------------------
`VLLM_BATCH_INVARIANT=1` forces deterministic, batch-size-independent reductions. For decode
attention this DISABLES the split-K / flash-decoding optimisation (the split COUNT and reduction
ORDER depend on batch/seqlen -> not batch-invariant), i.e. VBI=1 attention runs UN-PACKED. This is
exactly the denken #332 / wirbel #375 / #378 / #386 premise. On this pod (flash_attn 2.8.4; vLLM /
FlashInfer / transformers are NOT installed) the faithful, established (stark #363/#365) proxy is:
    num_splits=1  -> UN-PACKED single split == VBI=1 attention      (A_unpack)
    num_splits=0  -> deployed heuristic auto-split (packed)         (A_heuristic)
on the real gemma-4-E4B-it attention geometry (config: 8 q-heads / 2 kv-heads, head_dim 256 for
ALL layers, 42 layers = 35 sliding(window 512) + 7 full at idx {5,11,17,23,29,35,41}). Single-stream
batch=1, seqlen_q=1 decode step. Greedy identity is UNTOUCHED (we time kernels, we do not change them).

TWO L-SCALING CONVENTIONS (both reported; the gap between them is the deepest finding)
-------------------------------------------------------------------------------------
  * FULL convention (the #386 like-for-like PIN): the per-step attention scales with the FULL KV
    length L (what #386's shape_vbi1(L)=(L/L_ref)*penalty375(L)/penalty375(L_ref) implicitly assumes).
    Measured as the GLOBAL/full-attention un-pack shape:  shape_full(L) = A_full_unpack(L)/A_full_unpack(528).
    Replacing #386's INTERPOLATED shape with this MEASURED shape is the literal "pin the slope" ask;
    `pessimistic_breaches_3p2_measured` is computed on this convention.
  * COMPOSED convention (physically faithful per-step attention): 35 sliding layers SATURATE at the
    512 window (they attend to min(L,512) keys), only the 7 full layers grow with L. The real per
    decode-step attention is  A_comp(L) = 35*A_sliding(min(L,512)) + 7*A_full(L). Above the 512 window
    (L_ref=528 and the corners 578/658 are ALL > 512) the ctxlen sensitivity is carried by 7/42 layers
    only, so the composed shape rises FAR more gently than the full-L convention. This exposes whether
    the #386 floor (built on the full-L convention) OVERSTATES the ctxlen sensitivity by ignoring the
    sliding-window saturation -- the heart of the L=503-vs-528 anchor question (step 5: 503 is BELOW
    the window, 528 is ABOVE it).

RE-DERIVATION (inherit r_a/g_a UNCHANGED; recompute g_s with the MEASURED shape)
-------------------------------------------------------------------------------
The #386 model is EXACT and re-used verbatim:  floor = r_a*g_s,  gap = g_a + r_a*g_s,
  g_s(L_priv) = f_attn*(shape(L_priv)-1) / (1 + f_attn*(shape(L_priv)-1)),  f_attn=0.0951 (#378),
  r_a inherited per-corner from #379's deployed back-out (KERNEL-INVARIANT -- greedy identity preserved).
Only `shape(L_priv)` is swapped: interpolated #375 -> MEASURED (full and composed). The round-trip
self-test feeds the #386 INTERPOLATED shape back through this exact code and reproduces 3.5235%
pessimistic / 1.3097% central, proving the re-derivation matches #386 before the measured swap.

This is identity-safe GPU latency profiling on the local A10G (sm_86, on-target). 0 official TPS,
NO submission, NO served-file change, NO HF Job, NO --launch. Deployed best stays 481.53 / PPL 2.3772.

PRIMARY self-test : per_l_attention_vbi1_self_test_passes (bool).
Headline          : pessimistic_breaches_3p2_measured (bool), irreducible_gap_floor_pct_vbi1_measured
                    (central), measured_local_penalty_slope vs interp_375_slope (slope_ratio),
                    f_attn_measured, breakeven_prompt_shift_tok_measured, the composed (window-aware)
                    floor, and breach_is_anchor_artifact (L=503 leg).

Reproduce (on-target pod A10G):
    cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 python \\
      research/validity/per_l_attention_vbi1/per_l_attention_vbi1.py --gpu \\
      --proxy google/gemma-4-E4B-it-qat-w4a16-ct --vbi1 --measure-f-attn --self-test \\
      --wandb_group per-l-attention-vbi1 --wandb_name ubel/per-l-attention-vbi1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# research/validity/per_l_attention_vbi1/this.py -> repo root is 3 up.
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research" / "validity" / "per_l_attention_vbi1"
RESULTS_PATH = OUT_DIR / "per_l_attention_vbi1_results.json"

# --------------------------------------------------------------------------- #
# Served gemma-4-E4B-it attention geometry (text_config of the int4-ct proxy; VERIFIED from config.json).
# --------------------------------------------------------------------------- #
HIDDEN = 2560
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 256                         # head_dim for ALL layers (config has a single head_dim=256)
N_LAYERS = 42
SLIDING_WINDOW = 512
FULL_LAYER_IDX = frozenset({5, 11, 17, 23, 29, 35, 41})
N_FULL_LAYERS = len(FULL_LAYER_IDX)            # 7
N_SLIDING_LAYERS = N_LAYERS - N_FULL_LAYERS    # 35
SCALE = 1.0 / math.sqrt(HEAD_DIM)
A10G_SMS = 80                          # sm_86 A10G

# flash_attn split-K knobs: the VBI=1 (un-pack) and deployed (heuristic) attention proxies.
NS_HEURISTIC = 0                       # deployed auto-split (packed)
NS_UNPACK = 1                          # VBI=1 un-packed single split

# --------------------------------------------------------------------------- #
# Imported fleet anchors -- copied EXACTLY from #386 gap_floor_vbi1_regime (DO NOT re-derive).
# --------------------------------------------------------------------------- #
OFFICIAL_PUBLIC = 481.53
PRIVATE_VALID = 460.85
KNIFE_EDGE_PCT = 3.2
K_CAL = 125.268
ATTN_US = 557.90
BODY_US = 4474.19
LMHEAD_US = 131.62
L_REF = 528.0
OUT_LEN = 512
GAP_MEASURED = (OFFICIAL_PUBLIC - PRIVATE_VALID) / OFFICIAL_PUBLIC   # 4.2946% deployed gap

R_A_DEPLOYED_BANKED = 0.9570535584491102
R_A_DEPLOYED_CENTRAL = 0.9633874238374297
R_A_DEPLOYED_PESSIMISTIC = 0.973521608458741
FLOOR_379_CENTRAL_PCT = 0.6333865388319535
BREAKEVEN_379_TOK = 252.6103574841727

F_ATTN_VBI1 = 0.0951                   # #378 measured attention fraction under VBI=1 (eval-weighted)
EVAL_WEIGHTED_PENALTY_378 = 1.2257
# #375 un-pack penalty curve penalty(L)=A_unpack(L)/A_heuristic(L) anchors:
PENALTY_CURVE_375 = {110.0: 0.877, 352.0: 1.000, 528.0: 1.264, 2048.0: 3.027, 4096.0: 4.756}
PENALTY_AT_LREF = PENALTY_CURVE_375[L_REF]    # 1.264

# #386 banked results (the round-trip self-test must reproduce these from the interpolated shape):
FLOOR_386_CENTRAL_PCT = 1.3097036287951451
FLOOR_386_PESSIMISTIC_PCT = 3.523494549873982
FLOOR_386_BANKED_PCT = 0.0
BREAKEVEN_386_TOK = 118.61232172542913

# sensitivity-sweep corners (SAME as #379/#386): private prompt-length shift Delta-P (tokens).
DELTA_P_BANKED = 0.0
DELTA_P_CENTRAL = 50.0
DELTA_P_PESSIMISTIC = 130.0
CORNERS = (("banked", DELTA_P_BANKED), ("central", DELTA_P_CENTRAL), ("pessimistic", DELTA_P_PESSIMISTIC))

L_OPERATING_503 = 503.0                # #282 median decode length (the step-5 re-anchor operating point)
PUBLIC_MEAN_PROMPT_TOK = L_REF - OUT_LEN / 2.0   # ~272 tok (#379 convention, for the breakeven multiple)

# per-L measurement grid: #386 pessimistic bracket {528..658} + #375 anchors {352,2048} + #282 op pt 503,
# + 633 (= 503 + DELTA_P_PESSIMISTIC, needed for the L=503 re-anchor pessimistic corner).
L_GRID = (352.0, 503.0, 528.0, 553.0, 578.0, 603.0, 628.0, 633.0, 658.0, 2048.0)


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv)
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-13, max_it: int = 400) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(f"bisect: no sign change on [{lo},{hi}] -> {flo},{fhi}")
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def penalty_of_L_375(L: float) -> float:
    """Piecewise-linear interpolation of the #375 un-pack penalty curve (the value #386 used)."""
    xs = sorted(PENALTY_CURVE_375)
    if L <= xs[0]:
        x0, x1 = xs[0], xs[1]
    elif L >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]
    else:
        x0 = max(x for x in xs if x <= L)
        x1 = min(x for x in xs if x >= L)
        if x0 == x1:
            return PENALTY_CURVE_375[x0]
    y0, y1 = PENALTY_CURVE_375[x0], PENALTY_CURVE_375[x1]
    return y0 + (y1 - y0) * (L - x0) / (x1 - x0)


def shape_vbi1_modeled(L: float, L_ref: float = L_REF) -> float:
    """#386's MODELED un-pack attention shape A_vbi1(L)/A_vbi1(L_ref) = (L/L_ref)*penalty375(L)/penalty375(L_ref).
    This is exactly the curve #386 fed into g_s; the round-trip self-test reproduces #386 from it."""
    return (L / L_ref) * (penalty_of_L_375(L) / penalty_of_L_375(L_ref))


# --------------------------------------------------------------------------- #
# #386 step-loss model -- COPIED EXACTLY so the measured-shape re-derivation is identical except shape().
# --------------------------------------------------------------------------- #
F_ATTN_DEPLOYED = ATTN_US / (1.0e6 / K_CAL)    # 0.069887...


def g_s_deployed(delta_p_tokens: float) -> float:
    """#379 deployed step loss: A(L)=ATTN_US*L/L_ref (L-linear), f_attn~0.0699. Backs out KERNEL-INVARIANT r_a."""
    shape = (L_REF + delta_p_tokens) / L_REF
    x = F_ATTN_DEPLOYED * (shape - 1.0)
    return x / (1.0 + x)


def r_a_deployed(delta_p_tokens: float) -> float:
    """KERNEL-INVARIANT accept ratio backed out of the FIXED deployed gap at this delta_p (inherited unchanged)."""
    r_s = 1.0 - g_s_deployed(delta_p_tokens)
    return (1.0 - GAP_MEASURED) / r_s


def g_s_from_shape(shape_at_L_priv: float, f_attn: float = F_ATTN_VBI1) -> float:
    """g_s under VBI=1 given the (measured OR modeled) un-pack attention shape at L_priv. #386 eq, verbatim."""
    x = f_attn * (shape_at_L_priv - 1.0)
    return x / (1.0 + x)


def floor_pct(delta_p_tokens: float, shape_fn: Callable[[float], float], L_ref: float = L_REF,
              f_attn: float = F_ATTN_VBI1) -> float:
    """Irreducible ctxlen floor (absolute % of TPS gap) = 100 * r_a * g_s, with r_a inherited and g_s from shape_fn."""
    L_priv = L_ref + delta_p_tokens
    g_s = g_s_from_shape(shape_fn(L_priv), f_attn)
    return 100.0 * r_a_deployed(delta_p_tokens) * g_s


# --------------------------------------------------------------------------- #
# GPU per-L attention latency (flash_attn isolated, #363/#365 median-us methodology + mean/std).
# --------------------------------------------------------------------------- #
def _device():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. On this pod set CUDA_VISIBLE_DEVICES=0 (default device is dead) "
              "-- the #358/#363 gotcha.", file=sys.stderr)
        sys.exit(2)
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(dev)
    return {"name": p.name, "sm_count": p.multi_processor_count,
            "cc": f"{p.major}.{p.minor}", "total_mem_gib": round(p.total_memory / (1024 ** 3), 2),
            "is_a10g_sm86": ("A10G" in p.name) and (p.major == 8 and p.minor == 6)}


def _decode_closure(eff_L: int, is_full: bool, num_splits: int, seed: int, dev):
    """One served single-stream (batch=1) decode step: seqlen_q=1 query over an eff_L-long KV cache, at the
    given split-K. Sliding layers pass a window-capped cache (eff_L=min(L,512)); full layers pass eff_L=L."""
    import torch
    window = (-1, -1) if is_full else (SLIDING_WINDOW - 1, 0)
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(1, 1, N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    kc = torch.randn(1, eff_L + 1, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(1, eff_L + 1, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 1, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 1, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    cs = torch.tensor([eff_L], device=dev, dtype=torch.int32)
    from flash_attn import flash_attn_with_kvcache

    def call():
        return flash_attn_with_kvcache(q, kc, vc, k=k, v=v, cache_seqlens=cs, softmax_scale=SCALE,
                                       causal=True, window_size=window, num_splits=num_splits)
    return call


def _time_us(call, iters: int, warmup: int, dev) -> dict[str, float]:
    """CUDA-event timing of a single attention call. Returns mean/std/median/p10/p90 us over `iters` after warmup."""
    import torch
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        call()
        t1.record()
        torch.cuda.synchronize()
        samples.append(t0.elapsed_time(t1) * 1.0e3)   # ms -> us
    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / max(1, n - 1)
    return {"mean": mean, "std": math.sqrt(var), "median": samples[n // 2],
            "p10": samples[max(0, int(0.10 * n))], "p90": samples[min(n - 1, int(0.90 * n))], "n": n}


def measure_per_L(iters: int, warmup: int, seed: int, dev) -> dict[str, Any]:
    """For each L in the grid time the sliding and full attention decode step at num_splits in {0,1}, then
    compose the per-step attention as 35*sliding(min(L,512)) + 7*full(L). All us are batch=1 seqlen_q=1."""
    out: dict[str, Any] = {"grid": list(L_GRID), "iters": iters, "warmup": warmup, "by_L": {}}
    for L in L_GRID:
        eff_sliding = int(min(L, SLIDING_WINDOW))
        eff_full = int(L)
        row: dict[str, Any] = {"eff_sliding_L": eff_sliding, "eff_full_L": eff_full}
        for tag, is_full, eff in (("sliding", False, eff_sliding), ("full", True, eff_full)):
            for ns, nstag in ((NS_HEURISTIC, "heur"), (NS_UNPACK, "unpack")):
                stats = _time_us(_decode_closure(eff, is_full, ns, seed + ns + (1 if is_full else 0), dev),
                                 iters, warmup, dev)
                row[f"{tag}_{nstag}_us"] = stats
        # composed per-step attention for each split, built from the MEDIAN (robust central estimator for a
        # typical decode step; right-skew scheduling outliers bias the mean -- #363/#365 precedent uses median).
        for nstag in ("heur", "unpack"):
            row[f"composed_{nstag}_us"] = (N_SLIDING_LAYERS * row[f"sliding_{nstag}_us"]["median"]
                                           + N_FULL_LAYERS * row[f"full_{nstag}_us"]["median"])
        out["by_L"][f"{L:.0f}"] = row
    return out


# --------------------------------------------------------------------------- #
# Build the measured shape + penalty curves from the timed grid.
# --------------------------------------------------------------------------- #
def _A(meas: dict[str, Any], L: float, kind: str) -> float:
    """kind in {full_unpack, full_heur, composed_unpack, composed_heur}. MEDIAN us at grid point L (the robust
    central estimator; composed_* are already median-composed). mean/std are kept in the stats blob for reporting."""
    row = meas["by_L"][f"{L:.0f}"]
    if kind == "full_unpack":
        return row["full_unpack_us"]["median"]
    if kind == "full_heur":
        return row["full_heur_us"]["median"]
    if kind == "composed_unpack":
        return row["composed_unpack_us"]
    if kind == "composed_heur":
        return row["composed_heur_us"]
    raise KeyError(kind)


def build_curves(meas: dict[str, Any]) -> dict[str, Any]:
    """shape_full / shape_composed (un-pack L-scaling, normalized at 528 AND 503) + un-pack penalty curves."""
    def shape(kind_unpack: str, L: float, L_ref: float) -> float:
        return _A(meas, L, kind_unpack) / _A(meas, L_ref, kind_unpack)

    curves: dict[str, Any] = {"by_L": {}}
    for L in L_GRID:
        curves["by_L"][f"{L:.0f}"] = {
            "shape_full_528": shape("full_unpack", L, 528.0),
            "shape_full_503": shape("full_unpack", L, 503.0),
            "shape_composed_528": shape("composed_unpack", L, 528.0),
            "shape_composed_503": shape("composed_unpack", L, 503.0),
            "unpack_penalty_full": _A(meas, L, "full_unpack") / _A(meas, L, "full_heur"),
            "unpack_penalty_composed": _A(meas, L, "composed_unpack") / _A(meas, L, "composed_heur"),
            "shape_modeled_386_528": shape_vbi1_modeled(L, 528.0),   # #386 interpolated (for side-by-side)
        }
    return curves


def shape_fn_factory(curves: dict[str, Any], key: str):
    """Return shape_fn(L) reading the measured curve `key` at grid point L (L must be on the grid)."""
    def fn(L: float) -> float:
        return curves["by_L"][f"{L:.0f}"][key]
    return fn


# --------------------------------------------------------------------------- #
# Slopes on [528, 658] + provenance to #375 anchors.
# --------------------------------------------------------------------------- #
def slope_block(curves: dict[str, Any]) -> dict[str, Any]:
    def sl(key: str, lo: float, hi: float) -> float:
        c = curves["by_L"]
        return (c[f"{hi:.0f}"][key] - c[f"{lo:.0f}"][key]) / (hi - lo)

    measured_full = sl("shape_full_528", 528.0, 658.0)
    measured_composed = sl("shape_composed_528", 528.0, 658.0)
    interp_375 = (shape_vbi1_modeled(658.0) - shape_vbi1_modeled(528.0)) / (658.0 - 528.0)  # #386 modeled shape slope
    # raw #375 penalty-curve segment slope (the literal [528,2048] anchor slope cited in the hypothesis prose)
    interp_375_unpack_penalty_seg = (PENALTY_CURVE_375[2048.0] - PENALTY_CURVE_375[528.0]) / (2048.0 - 528.0)
    measured_unpack_penalty_full = sl("unpack_penalty_full", 528.0, 658.0)
    return {
        "L_lo": 528.0, "L_hi": 658.0,
        "measured_local_penalty_slope": measured_full,          # PR field: slope of measured full shape
        "interp_375_slope": interp_375,                          # PR field: the shape slope #386 used
        "slope_ratio": measured_full / interp_375 if interp_375 else float("nan"),
        "measured_composed_shape_slope": measured_composed,
        "slope_ratio_composed": measured_composed / interp_375 if interp_375 else float("nan"),
        "interp_375_unpack_penalty_seg_slope": interp_375_unpack_penalty_seg,  # 0.00116/tok (literal #375 segment)
        "measured_unpack_penalty_full_slope": measured_unpack_penalty_full,
    }


def provenance_375(curves: dict[str, Any]) -> dict[str, Any]:
    """Measured un-pack penalty (full geometry) at the #375 anchors L in {352, 2048} vs #375's {1.0, 3.027}."""
    c = curves["by_L"]
    p352 = c["352"]["unpack_penalty_full"]
    p2048 = c["2048"]["unpack_penalty_full"]
    p528 = c["528"]["unpack_penalty_full"]
    monotone = p352 <= p528 <= p2048
    # QUALITATIVE provenance (robust to local<->official absolute scaling; the RATIO/structure is portable):
    # the un-pack penalty must (i) sit near the #375 crossover at the short anchor, (ii) grow substantially by
    # 2048, and (iii) be monotone -- i.e. reproduce #375's curve SHAPE. Exact absolute reproduction is NOT
    # asserted (kernel/hardware-specific); the measured anchor values + relative deviation are reported.
    anchor_352_ok = 0.6 <= p352 <= 1.6
    anchor_2048_ok = p2048 >= 1.8
    return {
        "measured_penalty_352": p352, "ref_375_penalty_352": PENALTY_CURVE_375[352.0],
        "measured_penalty_2048": p2048, "ref_375_penalty_2048": PENALTY_CURVE_375[2048.0],
        "measured_penalty_528": p528, "ref_375_penalty_528": PENALTY_CURVE_375[528.0],
        "rel_dev_352": p352 / PENALTY_CURVE_375[352.0] - 1.0,
        "rel_dev_2048": p2048 / PENALTY_CURVE_375[2048.0] - 1.0,
        "penalty_monotone_in_L": bool(monotone),
        "anchor_352_in_band": bool(anchor_352_ok),
        "anchor_2048_grows": bool(anchor_2048_ok),
        "reproduces_375_anchors": bool(monotone and anchor_352_ok and anchor_2048_ok),
    }


# --------------------------------------------------------------------------- #
# Floor re-derivation at the 3 corners (full headline + composed physical), for a given anchor L_ref.
# --------------------------------------------------------------------------- #
def rederive(curves: dict[str, Any], shape_key: str, L_ref: float, f_attn: float = F_ATTN_VBI1) -> dict[str, Any]:
    fn = shape_fn_factory(curves, shape_key)
    out: dict[str, Any] = {"L_ref": L_ref, "shape_key": shape_key, "f_attn": f_attn, "corners": {}}
    for name, dp in CORNERS:
        L_priv = L_ref + dp
        shp = fn(L_priv)
        g_s = g_s_from_shape(shp, f_attn)
        r_a = r_a_deployed(dp)
        floor = 100.0 * r_a * g_s
        out["corners"][name] = {
            "delta_p_tokens": dp, "L_priv": L_priv, "shape_at_L_priv": shp,
            "r_accept_kernel_invariant": r_a, "g_s_vbi1_measured": g_s,
            "irreducible_gap_floor_abs_pct": floor,
            "gap_vbi1_total_pct": 100.0 * ((1.0 - r_a) + r_a * g_s),
            "clears_3p2": bool(floor < KNIFE_EDGE_PCT),
            "margin_pp": KNIFE_EDGE_PCT - floor,
        }
    cen = out["corners"]["central"]["irreducible_gap_floor_abs_pct"]
    pes = out["corners"]["pessimistic"]["irreducible_gap_floor_abs_pct"]
    out["central_floor_pct"] = cen
    out["pessimistic_floor_pct"] = pes
    out["banked_floor_pct"] = out["corners"]["banked"]["irreducible_gap_floor_abs_pct"]
    out["all_corners_clear_3p2"] = bool(all(c["clears_3p2"] for c in out["corners"].values()))
    out["pessimistic_breaches_3p2"] = bool(pes >= KNIFE_EDGE_PCT)
    out["floor_inflation_ratio_vs_379"] = cen / FLOOR_379_CENTRAL_PCT
    return out


def breakeven_shift(curves: dict[str, Any], shape_key: str, L_ref: float) -> float:
    """delta_p where the measured floor hits exactly 3.2%. Uses the measured shape interpolated between grid
    points (the un-pack shape is smooth in L); returns NaN if 3.2% is unreachable within the grid span."""
    # linear interpolation of the measured shape between grid points for off-grid L
    grid = sorted(L_GRID)
    keyed = [(L, curves["by_L"][f"{L:.0f}"][shape_key]) for L in grid]

    def shape_interp(L: float) -> float:
        if L <= keyed[0][0]:
            (x0, y0), (x1, y1) = keyed[0], keyed[1]
        elif L >= keyed[-1][0]:
            (x0, y0), (x1, y1) = keyed[-2], keyed[-1]
        else:
            lo = max(p for p in keyed if p[0] <= L)
            hi = min(p for p in keyed if p[0] >= L)
            if lo[0] == hi[0]:
                return lo[1]
            (x0, y0), (x1, y1) = lo, hi
        return y0 + (y1 - y0) * (L - x0) / (x1 - x0)

    def floor_at(dp: float) -> float:
        return 100.0 * r_a_deployed(dp) * g_s_from_shape(shape_interp(L_ref + dp))

    dp_hi = grid[-1] - L_ref
    try:
        if floor_at(0.0) < KNIFE_EDGE_PCT < floor_at(dp_hi):
            return bisect(lambda dp: floor_at(dp) - KNIFE_EDGE_PCT, 0.0, dp_hi)
    except ValueError:
        return float("nan")
    return float("nan")


# --------------------------------------------------------------------------- #
# f_attn re-derivation: replace #375's penalty(528)=1.264 with the MEASURED un-pack penalty at the anchor,
# holding the #378-implied body B_vbi1 fixed (B is NOT measured here; this is a penalty-driven f_attn check).
# --------------------------------------------------------------------------- #
def f_attn_measured(curves: dict[str, Any], penalty_key: str, L_ref: float) -> dict[str, Any]:
    A_vbi1_ref_modeled = ATTN_US * PENALTY_AT_LREF                 # 557.9 * 1.264
    B_vbi1 = A_vbi1_ref_modeled * (1.0 / F_ATTN_VBI1 - 1.0)        # body backed out from #378 f_attn=0.0951
    pen_meas_ref = curves["by_L"][f"{L_ref:.0f}"][penalty_key]
    A_vbi1_ref_meas = ATTN_US * pen_meas_ref
    f_meas = A_vbi1_ref_meas / (A_vbi1_ref_meas + B_vbi1)
    return {
        "penalty_key": penalty_key, "L_ref": L_ref,
        "measured_unpack_penalty_at_ref": pen_meas_ref, "ref_375_penalty_at_ref": PENALTY_AT_LREF,
        "B_vbi1_backed_out_us": B_vbi1, "f_attn_modeled_378": F_ATTN_VBI1, "f_attn_measured": f_meas,
        "note": ("f_attn_measured replaces #375's penalty(528)=1.264 with the MEASURED un-pack penalty at the "
                 "anchor; body B_vbi1 is backed out of #378's f_attn=0.0951 and held fixed (body NOT measured "
                 "in this attention-only card)."),
    }


# --------------------------------------------------------------------------- #
# Round-trip: feed the #386 INTERPOLATED shape through the EXACT re-derivation -> must reproduce #386.
# --------------------------------------------------------------------------- #
def roundtrip_386() -> dict[str, Any]:
    def modeled(L: float) -> float:
        return shape_vbi1_modeled(L, L_REF)
    cen = floor_pct(DELTA_P_CENTRAL, modeled)
    pes = floor_pct(DELTA_P_PESSIMISTIC, modeled)
    ban = floor_pct(DELTA_P_BANKED, modeled)
    return {
        "central_floor_pct": cen, "pessimistic_floor_pct": pes, "banked_floor_pct": ban,
        "matches_386_central": abs(cen - FLOOR_386_CENTRAL_PCT) <= 1e-6,
        "matches_386_pessimistic": abs(pes - FLOOR_386_PESSIMISTIC_PCT) <= 1e-6,
        "matches_386_banked": abs(ban - FLOOR_386_BANKED_PCT) <= 1e-9,
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY): provenance + round-trip + central clears + harness sanity.
# --------------------------------------------------------------------------- #
def self_test(meas: dict[str, Any], curves: dict[str, Any], prov: dict[str, Any], rt: dict[str, Any],
              floor_full_528: dict[str, Any], slopes: dict[str, Any], gpu: dict[str, Any],
              flags: dict[str, bool]) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (A) provenance: the measured un-pack penalty reproduces #375's anchor STRUCTURE (monotone, crossover-near
    #     -1.0 at 352, substantial growth by 2048). Absolute reproduction is reported, not gated (local-relative).
    c["a_provenance_reproduces_375_anchors"] = bool(prov["reproduces_375_anchors"])
    # (B) round-trip: the EXACT re-derivation, fed #386's INTERPOLATED shape, reproduces #386's banked floors.
    c["b_roundtrip_central_1p3097"] = bool(rt["matches_386_central"])
    c["b_roundtrip_pessimistic_3p5235"] = bool(rt["matches_386_pessimistic"])
    c["b_roundtrip_banked_zero"] = bool(rt["matches_386_banked"])
    # (C) the central corner clears 3.2% under the MEASURED slope (full convention).
    c["c_central_clears_3p2_measured"] = bool(floor_full_528["corners"]["central"]["clears_3p2"])
    # (D) measured-curve hygiene: shapes finite, >0, and monotone non-decreasing across the NOISE-RESOLVED
    #     anchors that carry the slope/floor conclusion: {352, 528, 578, 658, 2048}. Adjacent 25-tok grid steps
    #     move the full-attn latency by ~2us -- at/below the A10G CUDA-event jitter floor (~1us on ~85us) -- so
    #     strict point-to-point monotonicity over the FINE grid is not resolvable and is NOT gated; the worst
    #     fine-grid dip is reported as an honest diagnostic. The corner points (528/578/658) used by every
    #     reported floor ARE resolved (>=50-tok spans, +6-13% signal) and are included in the gated anchors.
    fulls = [curves["by_L"][f"{L:.0f}"]["shape_full_528"] for L in L_GRID]
    comps = [curves["by_L"][f"{L:.0f}"]["shape_composed_528"] for L in L_GRID]
    c["d_shapes_finite_positive"] = all(math.isfinite(x) and x > 0 for x in fulls + comps)
    resolved_anchors = [352.0, 528.0, 578.0, 658.0, 2048.0]
    res_full = [curves["by_L"][f"{L:.0f}"]["shape_full_528"] for L in resolved_anchors]
    c["d_full_shape_monotone_resolved"] = all(res_full[i] <= res_full[i + 1] + 1e-9 for i in range(len(res_full) - 1))
    c["d_shape_at_ref_is_one"] = abs(curves["by_L"]["528"]["shape_full_528"] - 1.0) <= 1e-9
    # honest (NON-gated) fine-grid diagnostics: worst local dip in the full shape across the 25-tok steps.
    Ls = sorted(L_GRID)
    fine_full = [curves["by_L"][f"{L:.0f}"]["shape_full_528"] for L in Ls]
    fine_diffs = [fine_full[i + 1] - fine_full[i] for i in range(len(Ls) - 1)]
    worst_dip = min(fine_diffs) if fine_diffs else 0.0
    diagnostics = {
        "fine_grid_worst_dip_shape": worst_dip,
        "fine_grid_worst_dip_pct_of_ref": 100.0 * worst_dip,   # shape is normalized at 528 -> dip as % of ref
        "fine_grid_strict_monotone": bool(worst_dip >= -1e-6),
        "resolved_anchors": resolved_anchors,
        "resolved_anchor_shapes_full": res_full,
    }
    # (E) latency hygiene: all timed means finite & positive; >= 200 iters as the PR requires.
    lat = [meas["by_L"][f"{L:.0f}"][f"{tag}_{ns}_us"]["mean"]
           for L in L_GRID for tag in ("sliding", "full") for ns in ("heur", "unpack")]
    c["e_latency_finite_positive"] = all(math.isfinite(x) and x > 0 for x in lat)
    c["e_iters_ge_200"] = int(meas["iters"]) >= 200
    # (F) slope block well-formed.
    c["f_slope_finite"] = all(math.isfinite(float(slopes[k])) for k in
                              ("measured_local_penalty_slope", "interp_375_slope", "slope_ratio"))
    # (G) provenance / on-target / launch flags.
    c["g_on_target_a10g_sm86"] = bool(gpu["is_a10g_sm86"])
    c["g_no_launch_flags"] = all(flags.values())
    gate = all(c.values())
    return {"checks": c, "n_checks": len(c), "diagnostics": diagnostics,
            "per_l_attention_vbi1_self_test_passes": bool(gate)}


# --------------------------------------------------------------------------- #
# Verdict.
# --------------------------------------------------------------------------- #
def verdict(floor_full: dict[str, Any], floor_comp: dict[str, Any], floor_full_503: dict[str, Any],
            slopes: dict[str, Any], be_full: float) -> dict[str, Any]:
    pes_full = floor_full["pessimistic_floor_pct"]
    cen_full = floor_full["central_floor_pct"]
    pes_comp = floor_comp["pessimistic_floor_pct"]
    breaches = floor_full["pessimistic_breaches_3p2"]
    pes_full_503 = floor_full_503["pessimistic_floor_pct"]
    breach_is_anchor_artifact = bool((pes_full >= KNIFE_EDGE_PCT) and (pes_full_503 < KNIFE_EDGE_PCT))
    slope_ratio = slopes["slope_ratio"]

    if breaches:
        band = "RED_breach_CONFIRMED_measured"
        action = "fern #357 re-derives the demand ceiling on the MEASURED VBI=1 floor (breach genuine)"
        summary = (
            f"BREACH CONFIRMED (measured). The GPU per-L pin gives a measured local shape slope "
            f"{slopes['measured_local_penalty_slope']:.5f}/tok = {slope_ratio:.2f}x #386's interpolated "
            f"{slopes['interp_375_slope']:.5f}/tok; the pessimistic corner floor is {pes_full:.3f}% "
            f">= 3.2% on the full-L convention (central {cen_full:.3f}%). #386's thin -0.32pp breach is "
            f"NOT a conservative-slope artifact -- the demand route's all-corner robustness pillar is "
            f"genuinely lost on the live contract; fern #357 must re-derive on >=1.31% with the pessimistic "
            f"corner breaching. NB the physically-faithful WINDOW-AWARE composed floor is {pes_comp:.3f}% "
            f"(35/42 sliding layers saturate at the 512 window) -- report both.")
    else:
        band = "GREEN_breach_LIFTED_measured"
        action = ("breach was a conservative-slope artifact; central VBI=1 floor "
                  f"{cen_full:.3f}% stands but every corner clears 3.2%")
        summary = (
            f"BREACH LIFTED (measured). The GPU per-L pin gives a measured local shape slope "
            f"{slopes['measured_local_penalty_slope']:.5f}/tok = {slope_ratio:.2f}x #386's interpolated "
            f"{slopes['interp_375_slope']:.5f}/tok; the pessimistic corner floor falls to {pes_full:.3f}% "
            f"< 3.2% on the full-L convention -- #386's -0.32pp breach was a CONSERVATIVE-SLOPE ARTIFACT. "
            f"Central floor {cen_full:.3f}% (clears +{KNIFE_EDGE_PCT - cen_full:.2f}pp), breakeven "
            f"+{be_full:.0f} tok. Moreover the physically-faithful WINDOW-AWARE composed floor is only "
            f"{pes_comp:.3f}% pessimistic (35/42 sliding layers saturate at the 512 window, so the per-step "
            f"ctxlen sensitivity above L=512 is carried by 7/42 full layers) -- the floor is even SMALLER "
            f"than the full-L convention. The demand route's all-corner robustness survives the measured pin.")
    return {
        "pessimistic_breaches_3p2_measured": breaches,
        "breach_is_anchor_artifact": breach_is_anchor_artifact,
        "verdict_band": band, "recommended_action": action, "verdict_summary": summary,
    }


# --------------------------------------------------------------------------- #
# Assemble.
# --------------------------------------------------------------------------- #
def build_report(meas: dict[str, Any], gpu: dict[str, Any], peak_mem_mib: float,
                 flags: dict[str, bool]) -> dict[str, Any]:
    curves = build_curves(meas)
    slopes = slope_block(curves)
    prov = provenance_375(curves)
    rt = roundtrip_386()

    # floors: full headline (528) + composed physical (528) + full re-anchored at 503.
    floor_full_528 = rederive(curves, "shape_full_528", 528.0)
    floor_comp_528 = rederive(curves, "shape_composed_528", 528.0)
    floor_full_503 = rederive(curves, "shape_full_503", 503.0)
    floor_comp_503 = rederive(curves, "shape_composed_503", 503.0)

    be_full = breakeven_shift(curves, "shape_full_528", 528.0)
    be_comp = breakeven_shift(curves, "shape_composed_528", 528.0)

    fattn_full = f_attn_measured(curves, "unpack_penalty_full", 528.0)
    fattn_comp = f_attn_measured(curves, "unpack_penalty_composed", 528.0)

    st = self_test(meas, curves, prov, rt, floor_full_528, slopes, gpu, flags)
    vrd = verdict(floor_full_528, floor_comp_528, floor_full_503, slopes, be_full)

    report = {
        "pr": 389, "issue": 319, "author": "ubel",
        "leg": "PIN the #386 pessimistic breach: GPU per-L attention latency under VBI=1 at L in [528,658]",
        "analysis_only": False, "gpu_used": True, **flags, "tps_added_by_this_card": 0,
        "proxy": "google/gemma-4-E4B-it-qat-w4a16-ct", "vbi1": True,
        "method": ("flash_attn 2.8.4 isolated decode timing (vLLM/FlashInfer NOT installed); num_splits=1 "
                   "(un-pack) == VBI=1 attention proxy, num_splits=0 == deployed heuristic; real gemma-4-E4B "
                   "geometry 8q/2kv head_dim 256, 35 sliding(512)+7 full layers; batch=1 seqlen_q=1 decode."),
        "gpu": gpu, "peak_mem_mib": peak_mem_mib,
        "imported_386": {
            "floor_central_pct": FLOOR_386_CENTRAL_PCT, "floor_pessimistic_pct": FLOOR_386_PESSIMISTIC_PCT,
            "breakeven_tok": BREAKEVEN_386_TOK, "f_attn_vbi1_378": F_ATTN_VBI1,
            "penalty_curve_375": PENALTY_CURVE_375, "knife_edge_pct": KNIFE_EDGE_PCT,
            "r_a_corners_379": {"banked": R_A_DEPLOYED_BANKED, "central": R_A_DEPLOYED_CENTRAL,
                                "pessimistic": R_A_DEPLOYED_PESSIMISTIC},
        },
        "measurement": meas, "curves": curves, "slopes": slopes, "provenance_375": prov, "roundtrip_386": rt,
        "floor_full_528": floor_full_528, "floor_composed_528": floor_comp_528,
        "floor_full_503": floor_full_503, "floor_composed_503": floor_comp_503,
        "breakeven_full_tok": be_full, "breakeven_composed_tok": be_comp,
        "f_attn_measured_full": fattn_full, "f_attn_measured_composed": fattn_comp,
        # ---- HEADLINE (full-L convention = the like-for-like #386 pin) ----
        "pessimistic_breaches_3p2_measured": floor_full_528["pessimistic_breaches_3p2"],
        "irreducible_gap_floor_pct_vbi1_measured": floor_full_528["central_floor_pct"],
        "irreducible_floor_pessimistic_pct_vbi1_measured": floor_full_528["pessimistic_floor_pct"],
        "irreducible_floor_banked_pct_vbi1_measured": floor_full_528["banked_floor_pct"],
        "floor_inflation_ratio_measured": floor_full_528["floor_inflation_ratio_vs_379"],
        "measured_local_penalty_slope": slopes["measured_local_penalty_slope"],
        "interp_375_slope": slopes["interp_375_slope"],
        "slope_ratio": slopes["slope_ratio"],
        "f_attn_measured": fattn_full["f_attn_measured"],
        "breakeven_prompt_shift_tok_measured": be_full,
        "all_corners_clear_3p2_measured": floor_full_528["all_corners_clear_3p2"],
        # ---- physical (window-aware composed) secondary ----
        "pessimistic_breaches_3p2_composed": floor_comp_528["pessimistic_breaches_3p2"],
        "irreducible_gap_floor_pct_composed_measured": floor_comp_528["central_floor_pct"],
        "irreducible_floor_pessimistic_pct_composed": floor_comp_528["pessimistic_floor_pct"],
        # ---- anchor-sensitivity leg (L=503) ----
        "pessimistic_floor_pct_at_L503": floor_full_503["pessimistic_floor_pct"],
        "pessimistic_floor_pct_at_L503_composed": floor_comp_503["pessimistic_floor_pct"],
        "breach_is_anchor_artifact": vrd["breach_is_anchor_artifact"],
        # ---- verdict + self-test ----
        "verdict_band": vrd["verdict_band"], "recommended_action": vrd["recommended_action"],
        "verdict_summary": vrd["verdict_summary"],
        "self_test": st["checks"], "n_checks": st["n_checks"],
        "self_test_diagnostics": st["diagnostics"],
        "per_l_attention_vbi1_self_test_passes": st["per_l_attention_vbi1_self_test_passes"],
        "official_baseline_unchanged": OFFICIAL_PUBLIC,
    }
    return report


# --------------------------------------------------------------------------- #
# W&B.
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    repo = str(ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import init_wandb_run, log_summary, log_json_artifact, finish_wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[plav] wandb helpers unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="ubel", name=name, group=group,
        tags=["per-l-attention", "vbi1", "deployable-strict", "knife-edge-3p2", "un-pack-penalty",
              "gap-floor", "demand-side", "sliding-window", "issue-319", "pr-389"],
        config={"pr": 389, "issue": 319, "kind": "per-l-attention-vbi1",
                "proxy": report["proxy"], "vbi1": True,
                "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
                "n_layers": N_LAYERS, "n_sliding": N_SLIDING_LAYERS, "n_full": N_FULL_LAYERS,
                "sliding_window": SLIDING_WINDOW, "L_grid": list(L_GRID),
                "f_attn_vbi1_378": F_ATTN_VBI1, "knife_edge_pct": KNIFE_EDGE_PCT,
                "ns_unpack": NS_UNPACK, "ns_heuristic": NS_HEURISTIC},
    )
    if run is None:
        print("[plav] wandb disabled (no API key / WANDB_MODE).", flush=True)
        return None
    try:
        import wandb
        sl = report["slopes"]
        flat = {
            "global_step": 0,
            "primary/per_l_attention_vbi1_self_test_passes": int(report["per_l_attention_vbi1_self_test_passes"]),
            "headline/pessimistic_breaches_3p2_measured": int(report["pessimistic_breaches_3p2_measured"]),
            "headline/irreducible_gap_floor_pct_vbi1_measured": report["irreducible_gap_floor_pct_vbi1_measured"],
            "headline/irreducible_floor_pessimistic_pct_vbi1_measured":
                report["irreducible_floor_pessimistic_pct_vbi1_measured"],
            "headline/floor_inflation_ratio_measured": report["floor_inflation_ratio_measured"],
            "headline/all_corners_clear_3p2_measured": int(report["all_corners_clear_3p2_measured"]),
            "slope/measured_local_penalty_slope": sl["measured_local_penalty_slope"],
            "slope/interp_375_slope": sl["interp_375_slope"],
            "slope/slope_ratio": sl["slope_ratio"],
            "slope/measured_composed_shape_slope": sl["measured_composed_shape_slope"],
            "slope/slope_ratio_composed": sl["slope_ratio_composed"],
            "fattn/f_attn_measured": report["f_attn_measured"],
            "fattn/f_attn_modeled_378": F_ATTN_VBI1,
            "breakeven/measured_tok": report["breakeven_prompt_shift_tok_measured"],
            "breakeven/386_tok": BREAKEVEN_386_TOK,
            "composed/pessimistic_breaches_3p2": int(report["pessimistic_breaches_3p2_composed"]),
            "composed/irreducible_floor_pessimistic_pct": report["irreducible_floor_pessimistic_pct_composed"],
            "composed/central_floor_pct": report["irreducible_gap_floor_pct_composed_measured"],
            "anchor503/pessimistic_floor_pct_full": report["pessimistic_floor_pct_at_L503"],
            "anchor503/pessimistic_floor_pct_composed": report["pessimistic_floor_pct_at_L503_composed"],
            "anchor503/breach_is_anchor_artifact": int(report["breach_is_anchor_artifact"]),
            "provenance/measured_penalty_352": report["provenance_375"]["measured_penalty_352"],
            "provenance/measured_penalty_2048": report["provenance_375"]["measured_penalty_2048"],
            "provenance/reproduces_375_anchors": int(report["provenance_375"]["reproduces_375_anchors"]),
            "roundtrip/central_floor_pct": report["roundtrip_386"]["central_floor_pct"],
            "roundtrip/pessimistic_floor_pct": report["roundtrip_386"]["pessimistic_floor_pct"],
            "diag/fine_grid_worst_dip_pct_of_ref": report["self_test_diagnostics"]["fine_grid_worst_dip_pct_of_ref"],
            "diag/fine_grid_strict_monotone": int(report["self_test_diagnostics"]["fine_grid_strict_monotone"]),
            "gpu/sm_count": float(report["gpu"]["sm_count"]),
            "peak_mem_mib": report["peak_mem_mib"], "tps_added_by_this_card": 0,
        }
        flat = {k: v for k, v in flat.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))}
        run.log(flat)
        for k, v in report["self_test"].items():
            run.summary[f"selftest/{k}"] = int(bool(v))
        run.summary["verdict_band"] = report["verdict_band"]
        run.summary["recommended_action"] = report["recommended_action"]

        # per-L curve table (the measured shape/penalty + #386 modeled side-by-side).
        ctbl = wandb.Table(columns=["L", "full_unpack_us", "full_heur_us", "composed_unpack_us",
                                    "shape_full_528", "shape_composed_528", "shape_modeled_386_528",
                                    "unpack_penalty_full"])
        for L in L_GRID:
            row = report["measurement"]["by_L"][f"{L:.0f}"]
            cv = report["curves"]["by_L"][f"{L:.0f}"]
            ctbl.add_data(L, row["full_unpack_us"]["mean"], row["full_heur_us"]["mean"],
                          row["composed_unpack_us"], cv["shape_full_528"], cv["shape_composed_528"],
                          cv["shape_modeled_386_528"], cv["unpack_penalty_full"])
        run.log({"per_L_curve": ctbl})

        # corner floor table (full + composed).
        ftbl = wandb.Table(columns=["corner", "delta_p", "L_priv", "shape_full", "floor_full_pct",
                                    "shape_composed", "floor_composed_pct", "floor_386_pct"])
        f386 = {"banked": FLOOR_386_BANKED_PCT, "central": FLOOR_386_CENTRAL_PCT,
                "pessimistic": FLOOR_386_PESSIMISTIC_PCT}
        for nm, _dp in CORNERS:
            cf = report["floor_full_528"]["corners"][nm]
            cc = report["floor_composed_528"]["corners"][nm]
            ftbl.add_data(nm, cf["delta_p_tokens"], cf["L_priv"], cf["shape_at_L_priv"],
                          cf["irreducible_gap_floor_abs_pct"], cc["shape_at_L_priv"],
                          cc["irreducible_gap_floor_abs_pct"], f386[nm])
        run.log({"corner_floor_measured": ftbl})

        log_summary(run, {k: v for k, v in report.items()
                          if isinstance(v, (int, float, bool, str))}, step=0, run_prefix=name)
        log_json_artifact(run, name="per_l_attention_vbi1", artifact_type="analysis", data=report)
        rid = getattr(run, "id", None)
        print(f"[plav] W&B run: {getattr(run, 'url', rid)}", flush=True)
        finish_wandb(run)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[plav] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def print_report(r: dict[str, Any]) -> None:
    sl = r["slopes"]
    bar = "=" * 98
    print("\n" + bar, flush=True)
    print(" PIN THE #386 PESSIMISTIC BREACH: GPU PER-L ATTENTION LATENCY UNDER VBI=1 (PR #389, #319)", flush=True)
    print(bar, flush=True)
    g = r["gpu"]
    print(f" GPU: {g['name']} | SMs {g['sm_count']} | cc {g['cc']} | a10g_sm86={g['is_a10g_sm86']} | "
          f"peak_mem {r['peak_mem_mib']:.0f} MiB", flush=True)
    print(f" proxy: {r['proxy']}  (VBI=1 == num_splits=1 un-pack; heuristic == num_splits=0)", flush=True)
    print(" --- MEASURED per-L un-pack attention (median us; mean+/-std @anchors below) + shape (norm @528) ---",
          flush=True)
    print(f"   {'L':>6} | {'full_med':>9} | {'full_std':>8} | {'comp_med':>9} | {'shp_full':>9} | "
          f"{'shp_comp':>9} | {'shp_386mdl':>10} | {'pen_full':>8}", flush=True)
    for L in L_GRID:
        row = r["measurement"]["by_L"][f"{L:.0f}"]
        cv = r["curves"]["by_L"][f"{L:.0f}"]
        print(f"   {L:>6.0f} | {row['full_unpack_us']['median']:>9.2f} | {row['full_unpack_us']['std']:>8.2f} | "
              f"{row['composed_unpack_us']:>9.1f} | {cv['shape_full_528']:>9.4f} | {cv['shape_composed_528']:>9.4f} | "
              f"{cv['shape_modeled_386_528']:>10.4f} | {cv['unpack_penalty_full']:>8.3f}", flush=True)
    print("   mean +/- std (full un-pack us) @anchors:", flush=True)
    for L in (352.0, 528.0, 578.0, 658.0, 2048.0):
        s = r["measurement"]["by_L"][f"{L:.0f}"]["full_unpack_us"]
        print(f"      L={L:>5.0f}: {s['mean']:>8.2f} +/- {s['std']:>6.2f} us  (n={s['n']}, p10={s['p10']:.2f}, "
              f"p90={s['p90']:.2f})", flush=True)
    dg = r["self_test_diagnostics"]
    print(f"   fine-grid worst local dip (full shape): {dg['fine_grid_worst_dip_pct_of_ref']:+.3f}% of ref "
          f"(strict-monotone={dg['fine_grid_strict_monotone']}; gated on resolved anchors "
          f"{[int(a) for a in dg['resolved_anchors']]})", flush=True)
    print(" --- SLOPE on [528,658] (the PINNED quantity) ---", flush=True)
    print(f"   measured_local_penalty_slope : {sl['measured_local_penalty_slope']:.5f}/tok (full shape)", flush=True)
    print(f"   interp_375_slope (#386 used) : {sl['interp_375_slope']:.5f}/tok", flush=True)
    print(f"   slope_ratio  measured/interp : {sl['slope_ratio']:.3f}x   "
          f"(composed slope {sl['measured_composed_shape_slope']:.5f}/tok, "
          f"ratio {sl['slope_ratio_composed']:.3f}x)", flush=True)
    print(" --- PROVENANCE to #375 anchors (un-pack penalty full) ---", flush=True)
    p = r["provenance_375"]
    print(f"   penalty(352)  meas {p['measured_penalty_352']:.3f} vs #375 {p['ref_375_penalty_352']:.3f} "
          f"(dev {p['rel_dev_352']*100:+.1f}%) | penalty(2048) meas {p['measured_penalty_2048']:.3f} vs "
          f"#375 {p['ref_375_penalty_2048']:.3f} (dev {p['rel_dev_2048']*100:+.1f}%) | "
          f"reproduces={p['reproduces_375_anchors']}", flush=True)
    print(" --- ROUND-TRIP (#386 interpolated shape -> exact re-derivation) ---", flush=True)
    rt = r["roundtrip_386"]
    print(f"   central {rt['central_floor_pct']:.4f}% (==1.3097 {rt['matches_386_central']}) | "
          f"pessimistic {rt['pessimistic_floor_pct']:.4f}% (==3.5235 {rt['matches_386_pessimistic']})", flush=True)
    print(" --- RE-DERIVED FLOOR (MEASURED slope) ---", flush=True)
    for conv, key in (("FULL  (like-for-like #386 pin)", "floor_full_528"),
                      ("COMPOSED (window-aware, physical)", "floor_composed_528")):
        fd = r[key]
        print(f"   [{conv}]", flush=True)
        for nm, _dp in CORNERS:
            c = fd["corners"][nm]
            flag = "clears" if c["clears_3p2"] else ">>> BREACHES 3.2% <<<"
            print(f"      {nm:>11s} (dP={c['delta_p_tokens']:>5.0f}, L_priv={c['L_priv']:>5.0f}, "
                  f"shape={c['shape_at_L_priv']:.4f}): floor {c['irreducible_gap_floor_abs_pct']:6.3f}%  "
                  f"margin {c['margin_pp']:+6.2f}pp  {flag}", flush=True)
    print(" --- HEADLINE ---", flush=True)
    print(f"   pessimistic_breaches_3p2_measured     : {r['pessimistic_breaches_3p2_measured']} "
          f"(#386 was True @3.5235%)", flush=True)
    print(f"   irreducible_gap_floor_pct_vbi1_meas   : {r['irreducible_gap_floor_pct_vbi1_measured']:.4f}% "
          f"(central; #386 1.3097%)", flush=True)
    print(f"   f_attn_measured                       : {r['f_attn_measured']:.4f} (#378 modeled 0.0951)", flush=True)
    print(f"   breakeven_prompt_shift_tok_measured   : {r['breakeven_prompt_shift_tok_measured']:.1f} tok "
          f"(#386 +118.6)", flush=True)
    print(f"   COMPOSED pessimistic floor            : {r['irreducible_floor_pessimistic_pct_composed']:.3f}% "
          f"(breaches={r['pessimistic_breaches_3p2_composed']})", flush=True)
    print(f"   pessimistic_floor_pct_at_L503 (full)  : {r['pessimistic_floor_pct_at_L503']:.3f}%  "
          f"breach_is_anchor_artifact={r['breach_is_anchor_artifact']}", flush=True)
    print(f"   verdict_band                          : {r['verdict_band']}", flush=True)
    print(f"   PRIMARY self_test_passes              : {r['per_l_attention_vbi1_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['n_checks']})", flush=True)
    if not r["per_l_attention_vbi1_self_test_passes"]:
        for k, v in r["self_test"].items():
            if not v:
                print(f"      FAILED: {k}", flush=True)
    print(bar + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="informational; this pod requires CUDA_VISIBLE_DEVICES=0")
    ap.add_argument("--proxy", default="google/gemma-4-E4B-it-qat-w4a16-ct", help="int4-ct proxy id (geometry source)")
    ap.add_argument("--vbi1", action="store_true", help="profile the VBI=1 (num_splits=1 un-pack) attention path")
    ap.add_argument("--measure-f-attn", action="store_true", help="re-derive f_attn from the measured penalty")
    ap.add_argument("--self-test", action="store_true", help="exit nonzero if the primary self-test fails")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="ubel/per-l-attention-vbi1")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="per-l-attention-vbi1")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.iters = min(args.iters, 30)
        args.warmup = min(args.warmup, 10)

    import torch
    torch.manual_seed(args.seed)
    dev = _device()
    gpu = _gpu_facts(dev)
    if os.environ.get("VLLM_BATCH_INVARIANT", "") != "1":
        print("[plav] NOTE: VLLM_BATCH_INVARIANT != 1 in env. The VBI=1 attention is proxied by the "
              "num_splits=1 (un-pack) flash path regardless; set VLLM_BATCH_INVARIANT=1 to match the "
              "deployable-strict environment exactly.", flush=True)

    meas = measure_per_L(args.iters, args.warmup, args.seed, dev)
    torch.cuda.synchronize()
    peak_mem_mib = round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2)
    flags = {"no_hf_job": True, "no_launch": True, "no_submission": True, "no_served_file_change": True}
    report = build_report(meas, gpu, peak_mem_mib, flags)
    report["created_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print_report(report)

    wid = None
    if not args.no_wandb and not args.smoke:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    report["wandb_run_ids"] = [wid] if wid else []
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"[plav] wrote {RESULTS_PATH}", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": report["wandb_run_ids"],
        "primary_metric": {"name": "pessimistic_breaches_3p2_measured",
                           "value": int(report["pessimistic_breaches_3p2_measured"])},
        "test_metric": {"name": "irreducible_gap_floor_pct_vbi1_measured",
                        "value": report["irreducible_gap_floor_pct_vbi1_measured"]},
        "headline": {
            "pessimistic_breaches_3p2_measured": report["pessimistic_breaches_3p2_measured"],
            "irreducible_gap_floor_pct_vbi1_measured": report["irreducible_gap_floor_pct_vbi1_measured"],
            "irreducible_floor_pessimistic_pct_vbi1_measured":
                report["irreducible_floor_pessimistic_pct_vbi1_measured"],
            "measured_local_penalty_slope": report["measured_local_penalty_slope"],
            "interp_375_slope": report["interp_375_slope"], "slope_ratio": report["slope_ratio"],
            "f_attn_measured": report["f_attn_measured"],
            "breakeven_prompt_shift_tok_measured": report["breakeven_prompt_shift_tok_measured"],
            "all_corners_clear_3p2_measured": report["all_corners_clear_3p2_measured"],
            "pessimistic_breaches_3p2_composed": report["pessimistic_breaches_3p2_composed"],
            "irreducible_floor_pessimistic_pct_composed": report["irreducible_floor_pessimistic_pct_composed"],
            "pessimistic_floor_pct_at_L503": report["pessimistic_floor_pct_at_L503"],
            "breach_is_anchor_artifact": report["breach_is_anchor_artifact"],
            "per_l_attention_vbi1_self_test_passes": report["per_l_attention_vbi1_self_test_passes"],
            "verdict_band": report["verdict_band"], "recommended_action": report["recommended_action"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and not report["per_l_attention_vbi1_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[plav] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
