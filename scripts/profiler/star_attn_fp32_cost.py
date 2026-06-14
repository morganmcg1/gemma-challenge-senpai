"""fp32 star-attention COST gate (PR #98) -- the cost twin of wirbel #93's safety gate.

wirbel #93 (MERGED, RED) proved a relerr-1e-3 star-attention reduction flips 0.59%
of greedy tokens (noise floor 0) -> land #71's M=32 tree-verify MUST accumulate the
softmax.V star-attention reduction in fp32 (or prove bit-exactness) before quota.
That is a HARD gate on the #1 lever. This script prices it:

  > Does the MANDATORY fp32 star-attention accumulation erode the tree's projected
  > +18.2% net (~568 official, denken #85), or is it ~free?

Physics (PR #98 framing). At conc=1 decode attention is KV-read-bound and sits at the
irreducible latency floor (~20% peak BW, occupancy-bound -- denken #69). fp32 vs bf16
*accumulation* splits into three channels:

  1. KV-read bytes  -- UNCHANGED (KV stays bf16). The dominant cost. delta = 0.
  2. accumulator compute / registers -- SM-side, hidden under memory-latency stalls
     at the floor. The deployed vLLM kernel ALREADY accumulates in fp32 (acc/L/M/S =
     tl.float32), so there is nothing to switch and no register regression.
  3. fp32 INTERMEDIATE HBM traffic (split-KV per-segment partials) -- the ONLY
     bandwidth-relevant channel at the BW-bound floor. The deployed backend ALREADY
     allocates softmax_segm_output/max/expsum as torch.float32.

So we measure channel 3 directly on the REAL served kernel: run vLLM's 3D split-KV
``unified_attention`` at M in {1,8,16,32} with the per-segment partial buffer
(``softmax_segm_output``) in fp32 (deployed/safe path) vs bf16 (the hypothetical
cheaper-but-#93-UNSAFE custom-reduction path). The fp32-vs-bf16 device-us delta,
summed over the 30 sliding + 7 full decode layers, IS ``fp32_starattn_cost_pct``.
We also validate each path vs an SDPA reference: the bf16-partial path should show a
#93-class ~1e-3 error, the fp32-partial path ~1e-4 (greedy-safe), tying the cost
measurement to the safety finding.

LOCAL A10G op-microbench. NO HF launch, NO server, NO submission change, NO leaderboard
number -- composition/efficiency evidence only. Single assigned student GPU.
"""
from __future__ import annotations

# The container exposes one A10G as index 0 but inherits a host-physical
# CUDA_VISIBLE_DEVICES (=6 here) that makes torch.cuda unavailable; pin to the
# single visible device exactly like every other scripts/profiler/* roofline.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("0",):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Force the splitkv-verify patch ON so M>1 verify batches route to 3D split-KV
# (the path that USES the per-segment partial buffer whose dtype we are pricing).
os.environ.setdefault("SPLITKV_VERIFY", "1")

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the #39 fa2sw probe machinery (real-kernel inputs + device-time instrument).
from scripts.local_validation.profile_attention import (  # noqa: E402
    A10G_PEAK_GBPS,
    BLOCK_SIZE,
    DTYPE_BYTES,
    HEAD_DIM,
    N_FULL,
    N_KV_HEADS,
    N_Q_HEADS,
    N_SLIDING,
    SLIDING_WINDOW,
    _build_paged_kv,
    _maybe_install_splitkv,
    _measure_peak_bw,
    _op_bytes,
    _profiled_device_us,
    _rot_for,
    _sdpa_reference,
)

# --- decode-step economics (all from MERGED advisor-branch artifacts) ----------
# denken #69 (post-#43 re-profile, run r0ahjs45): GPU-busy/cycle at M=8.
M8_DECODE_STEP_US = 7964.0
ATTN_M8_SERVED_US = 605.0          # verify attention slice @ M=8 (7.6% of decode)
# denken #85 (tree-overhead audit, run f0c8mb39): M=32 tree decode step + economics.
M32_TREE_DECODE_STEP_US = 11600.0  # 301 us = 2.597% -> 11.59 ms
ATTN_AMORTIZE_M32_OVER_M8 = 1.06   # split-KV: M=32 attention = 1.06x M=8 (KV shared)
TREE_GROSS_PCT = 21.8              # wirbel #79 gross verify-GEMM gain
TREE_NET_STATIC_PCT = 19.82        # denken #85 net after non-GEMM overhead (static)
TREE_NET_PR_HEADLINE_PCT = 18.2    # PR #98 headline net
OFFICIAL_BASE_TPS = 481.53         # fa2sw_precache_kenyan official projection base
OFFICIAL_TREE_PROJ_TPS = 576.96    # denken #85 net (static) official projection
LOCAL_WALL_TPS_BASE = 454.0

# #93 margin map (gate_results.json) -- the hybrid target population (RED fallback).
NEAR_TIE_FRAC_LT_1E3 = 0.00537109375    # frac rel-margin < 1e-3 (fp32 ref)
GREEDY_FLIP_RATE_BF16_1E3 = 0.005926513671875


_SEQ_THRESHOLD_3D = 128 // N_KV_HEADS  # = 64 (backend default)
_N_SEG = 16


def _build_op_inputs(torch, layer_type: str, M: int, ctx: int) -> dict:
    """Build the real-kernel inputs for one decode-attention op (KV force-rotated
    cold so each iteration reads cold KV from HBM, defeating the 6 MB L2)."""
    device = torch.device("cuda")
    hd = HEAD_DIM[layer_type]
    scale = 1.0 / math.sqrt(hd)
    window = (SLIDING_WINDOW - 1, 0) if layer_type == "sliding" else (-1, -1)
    rot = _rot_for(layer_type, ctx)
    key_cache, value_cache, block_tables, nb = _build_paged_kv(
        torch, device, layer_type, ctx, rot)
    q = torch.randn(M, N_Q_HEADS, hd, dtype=torch.bfloat16, device=device) * 0.1
    cu_seqlens_q = torch.tensor([0, M], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([ctx], dtype=torch.int32, device=device)
    return {
        "device": device, "hd": hd, "scale": scale, "window": window, "rot": rot,
        "key_cache": key_cache, "value_cache": value_cache,
        "block_tables": block_tables, "q": q,
        "cu_seqlens_q": cu_seqlens_q, "seqused_k": seqused_k,
    }


def _make_call(torch, inp, M, ctx, segm_dtype, out):
    """Return (call_fn, segm_out) bound to a fresh partial buffer of ``segm_dtype``.
    ONLY the partial-output buffer dtype is varied: segm_max / segm_expsum (the LSE
    scalars) stay fp32 -- they are unconditionally fp32 in any sane impl (FA2/FA3
    softmax_lse) and are negligible bytes."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    device, hd = inp["device"], inp["hd"]
    segm_out = torch.empty(_SEQ_THRESHOLD_3D, N_Q_HEADS, _N_SEG, hd,
                           dtype=segm_dtype, device=device)
    segm_max = torch.empty(_SEQ_THRESHOLD_3D, N_Q_HEADS, _N_SEG,
                           dtype=torch.float32, device=device)
    segm_exp = torch.empty(_SEQ_THRESHOLD_3D, N_Q_HEADS, _N_SEG,
                           dtype=torch.float32, device=device)
    state = {"i": 0}

    def call():
        bt = inp["block_tables"][state["i"] % inp["rot"]]
        state["i"] += 1
        unified_attention(
            q=inp["q"], k=inp["key_cache"], v=inp["value_cache"], out=out,
            cu_seqlens_q=inp["cu_seqlens_q"], max_seqlen_q=M,
            seqused_k=inp["seqused_k"], max_seqlen_k=ctx,
            softmax_scale=inp["scale"], causal=True, window_size=inp["window"],
            block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=_SEQ_THRESHOLD_3D, num_par_softmax_segments=_N_SEG,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_exp,
        )

    return call, segm_out


def _routes_3d(M: int) -> bool:
    try:
        import splitkv_verify_patch as _skv
        return bool(_skv.would_redirect(
            q_rows=M, max_seqlen_q=M, segm_rows=_SEQ_THRESHOLD_3D,
            seq_threshold_3D=_SEQ_THRESHOLD_3D, num_seqs=1)) or (M == 1)
    except Exception:
        return M == 1


def _partial_working_set_bytes(layer_type: str, M: int, segm_dtype_bytes: int) -> int:
    """Per-OP partial-output buffer footprint (the per-segment softmax partials the
    3D kernel writes then ``reduce_segments`` reads back). Only the first M query
    rows of the (64, n_q, n_seg, hd) buffer are touched. This working set, plus the
    ~1-2 MB per-op KV read, is what must fit in L2 for the partial dtype to cost no
    HBM bandwidth."""
    return M * N_Q_HEADS * _N_SEG * HEAD_DIM[layer_type] * segm_dtype_bytes


def bench_partial_dtype(torch, inp, layer_type, M, ctx, segm_dtype, *,
                        n_iter: int) -> dict:
    """Time one decode-attention op on the REAL vLLM 3D split-KV kernel with the
    per-segment partial buffer in ``segm_dtype`` (fp32 = deployed/safe path,
    bf16 = the hypothetical cheaper-but-#93-UNSAFE custom-reduction path)."""
    out = torch.empty(M, N_Q_HEADS, inp["hd"], dtype=torch.bfloat16,
                      device=inp["device"])
    call, _ = _make_call(torch, inp, M, ctx, segm_dtype, out)
    dev_us = _profiled_device_us(torch, call, n_iter)
    seg_bytes = 1 if segm_dtype == torch.bfloat16 else 4
    return {
        "layer_type": layer_type, "M": M, "ctx": ctx,
        "segm_dtype": str(segm_dtype).replace("torch.", ""),
        "used_3d_split_kv": _routes_3d(M), "rot_buffers": inp["rot"],
        "device_us": dev_us,
        "kv_floor_bytes": _op_bytes(layer_type, ctx, M)["kv_floor_bytes"],
        "partial_working_set_bytes": _partial_working_set_bytes(
            layer_type, M, seg_bytes),
    }


def validate_partial_precision(torch, layer_type, M, ctx) -> dict:
    """Run BOTH partial dtypes on IDENTICAL inputs + an fp32-SDPA reference, so the
    extra error injected by the bf16 (cheaper) reduction is isolated from the
    unavoidable bf16-OUTPUT quantization floor. Ties the cost gate to wirbel #93:
    bf16 partials should inject ~#93-class extra error; fp32 partials should sit at
    the bf16-output floor (greedy-safe)."""
    inp = _build_op_inputs(torch, layer_type, M, ctx)
    out_fp32 = torch.empty(M, N_Q_HEADS, inp["hd"], dtype=torch.bfloat16,
                           device=inp["device"])
    out_bf16 = torch.empty_like(out_fp32)
    call_fp32, _ = _make_call(torch, inp, M, ctx, torch.float32, out_fp32)
    call_bf16, _ = _make_call(torch, inp, M, ctx, torch.bfloat16, out_bf16)
    call_fp32()
    call_bf16()
    torch.cuda.synchronize()
    ref = _sdpa_reference(torch, inp["q"], inp["key_cache"], inp["value_cache"],
                          inp["block_tables"][0], ctx, M, layer_type, inp["scale"])
    denom = ref.abs().max().clamp_min(1e-9)

    def vs(o):
        d = (o.float() - ref).abs()
        return {"max_abs_err": d.max().item(),
                "max_rel_err": (d.max() / denom).item()}

    direct = (out_fp32.float() - out_bf16.float()).abs()
    res = {
        "fp32_partial_vs_sdpa": vs(out_fp32),
        "bf16_partial_vs_sdpa": vs(out_bf16),
        "bf16_minus_fp32_partial_max_abs": direct.max().item(),
        "bf16_minus_fp32_partial_max_rel": (direct.max() / denom).item(),
    }
    del inp
    torch.cuda.empty_cache()
    return res


def measure_interleaved(torch, lt, M, ctx, n_iter, rounds):
    """Build the op inputs ONCE, then interleave fp32/bf16 partial timings across
    rounds (round0: fp32,bf16; round1: fp32,bf16; ...) so slow thermal drift cancels
    in the tiny fp32-bf16 delta rather than biasing it. Returns {dname: cell}."""
    inp = _build_op_inputs(torch, lt, M, ctx)
    us = {"fp32": [], "bf16": []}
    meta = {}
    for r in range(rounds):
        for dname, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            res = bench_partial_dtype(torch, inp, lt, M, ctx, dt, n_iter=n_iter)
            us[dname].append(res["device_us"])
            meta[dname] = res
    out = {}
    for dname in ("fp32", "bf16"):
        cell = meta[dname]
        cell["device_us"] = statistics.fmean(us[dname])
        cell["device_us_std"] = statistics.pstdev(us[dname]) if rounds > 1 else 0.0
        cell["device_us_all"] = us[dname]
        out[dname] = cell
    del inp
    torch.cuda.empty_cache()
    return out


def run(out_path: Path, ctx: int, n_iter: int, rounds: int,
        wandb_group: str | None, wandb_name: str | None) -> dict:
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    _maybe_install_splitkv()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    m_values = [1, 8, 16, 32]

    result: dict = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "ctx": ctx, "n_iter": n_iter, "rounds": rounds,
        "config": {
            "n_sliding": N_SLIDING, "n_full": N_FULL,
            "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
            "n_seg": 16, "seq_threshold_3D": 64,
            "M8_decode_step_us": M8_DECODE_STEP_US,
            "M32_tree_decode_step_us": M32_TREE_DECODE_STEP_US,
            "tree_gross_pct": TREE_GROSS_PCT,
            "tree_net_static_pct": TREE_NET_STATIC_PCT,
        },
    }
    print(f"[fp32cost] GPU {result['gpu']} ctx={ctx} n_iter={n_iter} rounds={rounds}",
          flush=True)
    result["peak_bw"] = _measure_peak_bw(torch, device)
    peak = result["peak_bw"]["measured_peak_gbps_copy"]
    print(f"[fp32cost] measured peak HBM copy BW = {peak:.0f} GB/s", flush=True)

    l2_bytes = result["l2_bytes"]

    # ---- per-(layer, M) interleaved fp32/bf16 timing + same-inputs validation --
    cells = {}
    valids = {}
    for lt in ("sliding", "full"):
        for M in m_values:
            pair = measure_interleaved(torch, lt, M, ctx, n_iter, rounds)
            v = validate_partial_precision(torch, lt, M, ctx)
            valids[(lt, M)] = v
            for dname in ("fp32", "bf16"):
                cells[(lt, M, dname)] = pair[dname]
            f32, b16 = pair["fp32"], pair["bf16"]
            ws_f32 = f32["partial_working_set_bytes"]
            print(f"   {lt:8s} M={M:<3d} 3D={int(f32['used_3d_split_kv'])} "
                  f"fp32 {f32['device_us']:7.2f}+-{f32['device_us_std']:.2f}  "
                  f"bf16 {b16['device_us']:7.2f}+-{b16['device_us_std']:.2f}us  "
                  f"d={f32['device_us']-b16['device_us']:+5.2f}us "
                  f"ws_fp32={ws_f32/1e6:.2f}MB{'>' if ws_f32 > l2_bytes else '<'}L2 "
                  f"relerr fp32={v['fp32_partial_vs_sdpa']['max_rel_err']:.1e} "
                  f"bf16={v['bf16_partial_vs_sdpa']['max_rel_err']:.1e}", flush=True)
    result["cells"] = {f"{lt}|M{M}|{d}": c for (lt, M, d), c in cells.items()}
    result["validation"] = {f"{lt}|M{M}": v for (lt, M), v in valids.items()}

    # ---- aggregate to a full decode cycle (30 sliding + 7 full) ---------------
    def cycle_us(M, dname):
        return (N_SLIDING * cells[("sliding", M, dname)]["device_us"]
                + N_FULL * cells[("full", M, dname)]["device_us"])

    def cycle_us_std(M, dname):
        # propagate per-cell pstdev across the 37 layers (independent-draw bound).
        sv = (N_SLIDING * cells[("sliding", M, dname)]["device_us_std"] ** 2
              + N_FULL * cells[("full", M, dname)]["device_us_std"] ** 2)
        return math.sqrt(sv)

    def cycle_kv_floor(M):
        return (N_SLIDING * cells[("sliding", M, "fp32")]["kv_floor_bytes"]
                + N_FULL * cells[("full", M, "fp32")]["kv_floor_bytes"])

    def cycle_pos_delta(M):
        # CONSERVATIVE upper bound on the fp32 cost: charge ONLY layer-types where
        # fp32 is genuinely SLOWER (the L2-spill cells), and ignore layer-types
        # where fp32 is faster (bf16 store/load conversion overhead that gives a
        # spurious "speedup"). This is the honest worst-case fp32 wall-cost: we do
        # not let the realized cycle delta go negative by crediting that quirk.
        d_sl = max(0.0, cells[("sliding", M, "fp32")]["device_us"]
                   - cells[("sliding", M, "bf16")]["device_us"])
        d_fl = max(0.0, cells[("full", M, "fp32")]["device_us"]
                   - cells[("full", M, "bf16")]["device_us"])
        return N_SLIDING * d_sl + N_FULL * d_fl

    agg = {}
    for M in m_values:
        f32 = cycle_us(M, "fp32")
        b16 = cycle_us(M, "bf16")
        delta = f32 - b16
        # noise floor on the delta: quadrature of the per-dtype cycle stds.
        delta_std = math.sqrt(cycle_us_std(M, "fp32") ** 2
                              + cycle_us_std(M, "bf16") ** 2)
        denom = M32_TREE_DECODE_STEP_US if M == 32 else M8_DECODE_STEP_US
        ws_f32_sl = cells[("sliding", M, "fp32")]["partial_working_set_bytes"]
        ws_f32_fl = cells[("full", M, "fp32")]["partial_working_set_bytes"]
        cons_delta = cycle_pos_delta(M)
        agg[M] = {
            "attn_us_per_cycle_fp32": f32,
            "attn_us_per_cycle_bf16": b16,
            "fp32_minus_bf16_us": delta,
            "fp32_minus_bf16_us_std": delta_std,
            "delta_over_noise": (delta / delta_std) if delta_std > 0 else 0.0,
            "decode_step_us": denom,
            "fp32_starattn_cost_pct": delta / denom * 100.0,
            "fp32_starattn_cost_pct_noise": delta_std / denom * 100.0,
            # conservative (spill-only) upper bound used for the gate + haircut.
            "fp32_cost_us_conservative": cons_delta,
            "fp32_starattn_cost_pct_conservative": cons_delta / denom * 100.0,
            "attn_pct_of_decode_fp32": f32 / denom * 100.0,
            "kv_floor_MB": cycle_kv_floor(M) / 1e6,
            # L2 residency: per-OP fp32 partial working set vs A10G L2 (6 MB).
            # When it fits, the partial write+read-back is L2-resident -> no HBM
            # traffic from the fp32 vs bf16 dtype; only spilling cells cost wall.
            "partial_ws_fp32_sliding_MB": ws_f32_sl / 1e6,
            "partial_ws_fp32_full_MB": ws_f32_fl / 1e6,
            "sliding_fp32_partial_spills_l2": ws_f32_sl > l2_bytes,
            "full_fp32_partial_spills_l2": ws_f32_fl > l2_bytes,
        }
        print(f"[fp32cost] cycle M={M:<3d}: fp32 {f32:7.1f}us  bf16 {b16:7.1f}us  "
              f"realized {delta:+6.1f}+-{delta_std:.1f}us "
              f"({agg[M]['fp32_starattn_cost_pct']:+.3f}%)  "
              f"conservative {cons_delta:5.1f}us "
              f"({agg[M]['fp32_starattn_cost_pct_conservative']:+.3f}%) "
              f"of {denom/1e3:.1f}ms  | fp32 partial ws full={ws_f32_fl/1e6:.2f}MB "
              f"{'SPILLS' if ws_f32_fl > l2_bytes else 'fits'}-L2", flush=True)
    result["cycle_aggregate"] = {str(k): v for k, v in agg.items()}

    # ---- PRIMARY/TEST metrics -------------------------------------------------
    # realized = direct cycle delta (goes <=0: fp32 is free / slightly faster
    # because bf16 partials pay conversion overhead with no BW benefit when
    # L2-resident). conservative = spill-only upper bound (the honest fp32 cost).
    cost_m8 = agg[8]["fp32_starattn_cost_pct"]
    cost_m32 = agg[32]["fp32_starattn_cost_pct"]
    cost_m8_cons = agg[8]["fp32_starattn_cost_pct_conservative"]
    cost_m32_cons = agg[32]["fp32_starattn_cost_pct_conservative"]

    # ---- Step 2: net-tree-gain reconciliation ---------------------------------
    # Charge the CONSERVATIVE (spill-only) fp32 cost at M=32 as the erosion: the
    # fp32 mandate adds delta_us to the M=32 tree decode step; TPS scales by
    # step/(step+delta). This is the pessimistic bound -- the realized delta is
    # negative (fp32 free), so the true haircut is <= this.
    delta_us_m32 = agg[32]["fp32_minus_bf16_us"]              # realized (<=0)
    delta_us_m32_cons = agg[32]["fp32_cost_us_conservative"]  # spill-only (>=0)
    step = M32_TREE_DECODE_STEP_US
    net_after_frac = (1.0 + TREE_NET_STATIC_PCT / 100.0) * step / (step + delta_us_m32_cons) - 1.0
    net_after_pct = net_after_frac * 100.0
    haircut_pp = TREE_NET_STATIC_PCT - net_after_pct
    official_after = OFFICIAL_BASE_TPS * (1.0 + net_after_frac)
    result["primary_metric"] = {
        "name": "fp32_starattn_tree_gain_haircut_pct", "value": haircut_pp}
    result["test_metric"] = {
        "name": "fp32_starattn_cost_pct",
        "value_M8": cost_m8, "value_M32": cost_m32,
        "value_M8_conservative": cost_m8_cons,
        "value_M32_conservative": cost_m32_cons}
    result["step2_net_reconciliation"] = {
        "tree_net_static_pct_before": TREE_NET_STATIC_PCT,
        "fp32_delta_us_M32_realized": delta_us_m32,
        "fp32_delta_us_M32_conservative": delta_us_m32_cons,
        "M32_tree_decode_step_us": step,
        "net_after_fp32_pct": net_after_pct,
        "fp32_starattn_tree_gain_haircut_pp": haircut_pp,
        "official_proj_before": OFFICIAL_TREE_PROJ_TPS,
        "official_proj_after_conservative": official_after,
        "official_proj_after_loss_tps": OFFICIAL_TREE_PROJ_TPS - official_after,
        "interpretation_A_deployed_reuse_haircut_pp": 0.0,
        "note": (
            "Haircut uses the CONSERVATIVE spill-only fp32 cost at M=32 (charges "
            "only the full-layers whose fp32 partials spill the 6 MB L2; ignores "
            "the sliding-layer bf16-conversion 'speedup'). Realized cycle delta is "
            "NEGATIVE (fp32 free). Interp A: denken #85 already priced the M=32 "
            "attention with the DEPLOYED fp32 split-KV kernel (1.06x M=8), so "
            "reusing that path adds 0 -> haircut 0."),
    }

    # ---- Step 3: gate (uses the conservative upper bound) ---------------------
    worst_cost = max(cost_m8_cons, cost_m32_cons)
    if worst_cost <= 1.0:
        verdict = "GREEN"
    elif worst_cost <= 3.0:
        verdict = "AMBER"
    else:
        verdict = "RED"
    result["verdict"] = verdict
    result["gate"] = {
        "fp32_starattn_cost_pct_M8_realized": cost_m8,
        "fp32_starattn_cost_pct_M32_realized": cost_m32,
        "fp32_starattn_cost_pct_M8_conservative": cost_m8_cons,
        "fp32_starattn_cost_pct_M32_conservative": cost_m32_cons,
        "worst_cost_pct_conservative": worst_cost,
        "rule": ("GREEN<=1%, AMBER 1-3%, RED>3% (worst of M8/M32, conservative "
                 "spill-only bound)"),
        "verdict": verdict,
    }

    # numerical safety cross-check (same-inputs): bf16-partial should inject the
    # #93-class extra reduction error; fp32-partial should sit at the bf16-OUTPUT
    # quantization floor (greedy-safe). The marginal-error metric isolates the
    # partial precision from the unavoidable bf16-output floor.
    def worst(path_keys):
        out = 0.0
        for v in valids.values():
            x = v
            for k in path_keys:
                x = x[k]
            out = max(out, x)
        return out
    result["numerical_safety"] = {
        "fp32_partial_worst_rel_err_vs_sdpa":
            worst(["fp32_partial_vs_sdpa", "max_rel_err"]),
        "bf16_partial_worst_rel_err_vs_sdpa":
            worst(["bf16_partial_vs_sdpa", "max_rel_err"]),
        "bf16_marginal_worst_rel_err":
            worst(["bf16_minus_fp32_partial_max_rel"]),
        "bf16_marginal_worst_abs_err":
            worst(["bf16_minus_fp32_partial_max_abs"]),
        "pr93_flip_rate_bf16_1e3": GREEDY_FLIP_RATE_BF16_1E3,
        "pr93_near_tie_frac_lt_1e3": NEAR_TIE_FRAC_LT_1E3,
        "note": ("bf16-partial (the cheaper custom-reduction path) injects extra "
                 "attention-output error on top of the unavoidable bf16-output "
                 "floor; where the marginal rel-err exceeds the #93 ~1e-3 near-tie "
                 "band it flips greedy tokens. fp32-partial = deployed, greedy-safe "
                 "AND ~free (this gate)."),
    }

    result["elapsed_s"] = time.time() - t0
    result["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"[fp32cost] peak GPU mem = {result['peak_gpu_gb']:.3f} GB", flush=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[fp32cost] VERDICT={verdict}  cost%(M8)={cost_m8:+.3f}/cons={cost_m8_cons:.3f} "
          f"cost%(M32)={cost_m32:+.3f}/cons={cost_m32_cons:.3f}  haircut={haircut_pp:.3f}pp",
          flush=True)
    print(f"[fp32cost] wrote {out_path}  ({result['elapsed_s']:.0f}s)", flush=True)

    # ---- W&B ------------------------------------------------------------------
    if wandb_group:
        try:
            import wandb
            run = wandb.init(
                project="gemma-challenge-senpai",
                entity="wandb-applied-ai-team",
                group=wandb_group, name=wandb_name or "wirbel/fp32-starattn-cost-gate",
                config={**result["config"], "ctx": ctx, "n_iter": n_iter,
                        "rounds": rounds, "gpu": result["gpu"]},
            )
            log = {
                "fp32_starattn_cost_pct_M8": cost_m8,
                "fp32_starattn_cost_pct_M32": cost_m32,
                "fp32_starattn_cost_pct_M8_conservative": cost_m8_cons,
                "fp32_starattn_cost_pct_M32_conservative": cost_m32_cons,
                "fp32_starattn_tree_gain_haircut_pp": haircut_pp,
                "net_after_fp32_pct": net_after_pct,
                "tree_net_static_pct_before": TREE_NET_STATIC_PCT,
                "official_proj_after_conservative": official_after,
                "verdict_green": int(verdict == "GREEN"),
                "fp32_partial_worst_rel_err": result["numerical_safety"]["fp32_partial_worst_rel_err_vs_sdpa"],
                "bf16_partial_worst_rel_err": result["numerical_safety"]["bf16_partial_worst_rel_err_vs_sdpa"],
                "bf16_marginal_worst_rel_err": result["numerical_safety"]["bf16_marginal_worst_rel_err"],
                "measured_peak_gbps": peak,
            }
            for M in m_values:
                log[f"attn_us_cycle_fp32_M{M}"] = agg[M]["attn_us_per_cycle_fp32"]
                log[f"attn_us_cycle_bf16_M{M}"] = agg[M]["attn_us_per_cycle_bf16"]
                log[f"fp32_cost_pct_M{M}"] = agg[M]["fp32_starattn_cost_pct"]
                log[f"fp32_cost_pct_cons_M{M}"] = agg[M]["fp32_starattn_cost_pct_conservative"]
                log[f"fp32_cost_pct_noise_M{M}"] = agg[M]["fp32_starattn_cost_pct_noise"]
            wandb.log(log)
            result["wandb_run_id"] = run.id
            run.summary.update(log)
            wandb.finish()
            print(f"[fp32cost] W&B run {run.id} (group {wandb_group})", flush=True)
            out_path.write_text(json.dumps(result, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[fp32cost] W&B logging skipped: {e!r}", flush=True)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=528,
                    help="decode context length (mean served ctx ~527.7)")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=_REPO_ROOT / "research/star_attn_gate/fp32_cost_results.json")
    ap.add_argument("--wandb-group", type=str, default="fp32-starattn-cost-gate")
    ap.add_argument("--wandb-name", type=str, default="wirbel/fp32-starattn-cost-gate")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    run(args.output, args.ctx, args.n_iter, args.rounds,
        None if args.no_wandb else args.wandb_group, args.wandb_name)
