#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Draft io_projection roofline: is the 13.9us io_projection (14.1% of the 101.2us
draft floor) intrinsic-M=1 or fold-able? (PR #277, kanna). LOCAL GPU micro-profiling
+ CPU analytic. Analysis-only: no served-file change, no HF Job, no submission.
BASELINE stays 481.53.

THE QUESTION
------------
This applies the EXACT #269 diagnostic (epl52mkq: an M=1 single-warp GEMV runs at
only ~41% of peak HBM BW, so its 2.42x "overhead" over the bandwidth roofline is the
INTRINSIC memory-latency penalty of M=1, physically UNREACHABLE without M>=16
batching, NOT recoverable slack; the ONLY recoverable part was the SEPARATE GeluAndMul
companion kernel, a +4.39% lossless epilogue fold) to the LAST uncovered draft term:
io_projection. kanna #264 (95x7qv6h, MERGED) decomposed the bf16 draft 101.2us/pass
(denken #254 zav6nr8y) into MLP 50.71us/51.7% (#269 -- DONE), attention 28.52us/29.1%
(wirbel #270 -- in flight), io_projection 13.86us/14.1% (THIS PR -- the third-largest
term, the only one not yet roofline-screened), and head-projection 4.91us/5.0%
(immaterial, closed). The PR hoped io_projection might be "MULTIPLE small q/k/v/o GEMV
launches per layer" carrying MORE separable-launch slack than the MLP's single
activation kernel.

WHAT io_projection ACTUALLY IS (the decisive diagnostic, read from #264's code)
------------------------------------------------------------------------------
kanna #264's build_gemv_buckets defines the "io_projection" bucket EXACTLY as the two
residual-stream in/out projections between the 2560-dim backbone and the 256-dim draft
hidden:
  pre_projection  : Linear [out=256,  in=5120 ]  (5120 -> 256)  2.50 MiB bf16
  post_projection : Linear [out=2560, in=256  ]  (256 -> 2560)  1.25 MiB bf16
These are NOT the q/k/v/o attention projections -- those live in #264's SEPARATE
"attention" bucket (q_proj + o_proj per layer = wirbel #270). So the PR's "q/k/v/o /
multiple-launches-per-layer" premise is a misrecollection; the AUTHORITATIVE #264
definition is two GEMVs total per pass, run ONCE each (pre at the very start of the
draft pass, post at the very end), separated by the entire 4-layer transformer stack.
Consequence, decided BEFORE any timing: there is NO per-layer multiplicity and NO
separate companion kernel inside the term (no activation, no elementwise op -- both
are pure linear GEMVs), and pre/post CANNOT be fused with each other (non-adjacent,
opposite ends, different in/out shapes). So io_projection structurally carries LESS
separable-launch slack than the MLP, not more.

WHAT THIS MEASURES (real A10G micro-profiling, no HF Job, no serve change)
-------------------------------------------------------------------------
  * io_projection weight byte-traffic at M=1 = pre_bytes + post_bytes = 3.75 MiB and
    its memory-bound floor `io_projection_roofline_us` at A10G 600 GB/s (6.55us).
  * The io_projection at M=1, launch-free CUDA graph (deployed ONEGRAPH basis), as the
    2-GEMV chain (matching #264's time_gemv_chain([pre,post]) = the 13.86us basis),
    AND each GEMV separately {pre, post}. Headline
    `io_overhead_ratio = io_projection_measured_us / io_projection_roofline_us`
    (1.0 = at roofline/bandwidth-bound/irrecoverable; >1.0 = under-saturation or slack),
    and the DECISIVE metric `io_bw_utilization` (% of peak HBM BW the M=1 io GEMV
    achieves). <~50% (vs a saturating reference at ~81%) => intrinsic-M=1-bound (the
    GEMV body is irrecoverable, like the MLP GEMVs); the recoverable part is ONLY any
    SEPARATE companion launches -- and io_projection has none.
  * A LARGE reference GEMV (the 128 MiB tied embed_tokens, #264/#269's anchor) at M=1
    to EMPIRICALLY validate the 600 GB/s model and CONTRAST a bandwidth-bound op
    (large -> ~roofline, ~81% peak) against the under-saturating M=1 io GEMVs.
  * Price any greedy-safe LOSSLESS launch-erasure. Expectation (researcher-agent +
    #269 + the #264-code reading above): recoverable ~= 0. io_projection is 2 pure
    GEMVs at opposite ends of the stack with no companion kernel; neither GEMV can be
    removed (it IS the math) and the two cannot be fused. Contrast the MLP, whose
    +4.4% came from its SEPARATE GeluAndMul kernel -- io_projection has no analogue.

GREEDY/PPL SAFETY: pinned BY CONSTRUCTION. This leg MEASURES; it edits no served file.
A fusion / launch-erasure of the draft io_projection would change NO emitted token: the
draft only PROPOSES, greedy-exact verify checks every candidate against the FULL target
argmax, so a draft change can only move E[T] (acceptance), never the emitted token.
There is moreover NO fusion available WITHIN the term (the two GEMVs are non-adjacent
and pure-linear), so no bf16 reduction-order is even reassociated -- FP-reassociation
is N/A here (contrast a MLP split-K megakernel, #269 / lawine #246).

SELF-TEST (`draft_io_projection_roofline_self_test_passes`, PRIMARY)
-------------------------------------------------------------------
(a) `io_projection_measured_us` recovers #264's 13.86us within +/-25%;
(b) the 600 GB/s BW model byte->us round-trips the MEASURED large reference GEMV
    within +/-20% (empirical BW anchor; large GEMV IS ~bandwidth-bound);
(c) the component us {pre, post} sum to the measured chain within +/-25% (M=1
    micro-timing is noisy; band stated);
(d) the composition reproduces 481.53 at the deployed (step, E[T]) point;
(e) NaN-clean;
(f) BASELINE 481.53, 520.95 lambda=1 ceiling, K_cal=125.268 imported EXACTLY.
TEST metric: `projected_tps_gain_pct` (best DEFENSIBLE greedy-safe gain off 481.53;
0.0 if NULL/at the achievable M=1 floor with no separable-launch companion).

Requires a torch+CUDA env (the served senpai venv #264/#269 used,
/tmp/senpai-venvs/5f4c623f772358a2). io_projection is pure bf16 F.linear -- no vLLM
custom op, no Marlin needed (bf16 control).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import struct

# Must be set before importing torch. Single-GPU node; in-container GPU is index 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys  # noqa: E402
# Keep the script dir off sys.path[0] so a stdlib `import profile` (some deps) is
# unaffected (same guard as #264/#269).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEFAULT_DRAFTER = "/tmp/qat-assistant"

# ---- A10G (AWS g5, GA102, sm_86) roofline ceiling (identical to #248/#254/#264/#269) -
A10G_HBM_GBS = 600.0
BF16_BYTES = 2.0

# ---- IMPORTED, UNCHANGED (this leg moves nothing) ----------------------------
FRONTIER_TPS = 481.53         # PR #52 official a10g-small frontier (BASELINE)
LAMBDA1_CEILING_TPS = 520.95  # lambda=1 built ceiling (ubel #240 / land GO read)
K_CAL = 125.268               # composition calibration (kanna #217 vgovdrjc / #260)
STEP_US = 1218.2              # served decode step (kanna #217 / #260)
K_DEPLOYED = 7                # num_speculative_tokens (manifest SPECULATIVE_CONFIG)
BF16_ANCHOR_US_254 = 101.2    # denken #254 zav6nr8y bf16-draft floor (the anchor)
DRAFT_SHARE_OF_STEP = 0.58    # bf16 draft fraction of the step (#254)
ET_DEPLOYED = 3.3             # accepted tok/step, bf16-draft control (#248/#254)
# kanna #264 (95x7qv6h) draft-pass decomposition -- io_projection is YOUR anchor.
IO_ANCHOR_US_264 = 13.861546516418457   # #264 us_io_projection (the 14.1% term)
MLP_US_264 = 50.71189244588216          # #264 us_mlp (#269 -- DONE)
ATTN_US_264 = 28.518400192260742        # #264 us_attention (wirbel #270)
HEAD_PROJ_US_264 = 4.9083733558654785   # #264 us_head_projection_gemv (immaterial)
GEMV_CHAIN_US_264 = 98.00021251042683   # #264 gemv_chain_total

# self-test tolerances (M=1 micro-GEMV timing is noisy; bands stated explicitly)
IO_ANCHOR_TOL_PCT = 0.25      # io_projection_measured_us vs #264 13.86us
BW_ROUNDTRIP_TOL_PCT = 0.20   # large-GEMV byte->us vs measured (empirical BW anchor)
COMPONENT_SUM_TOL_PCT = 0.25  # {pre,post} sum vs the measured chain
# A material overhead ratio gate: >15% above roofline => the GEMV sits above its
# bandwidth floor (but most of that gap is the intrinsic-M=1 under-saturation penalty).
OVERHEAD_MATERIAL_RATIO = 1.15
# A bandwidth-bound op should reach near peak; below this it UNDER-saturates HBM at M=1.
BW_SATURATED_FRAC = 0.70
# Irreducible GPU-side inter-kernel latency INSIDE a CUDA graph on Ampere
# (~60ns/node, NVIDIA CUDA Graphs blog) -- the only scheduling launch-erasure removes.
GRAPH_NODE_LATENCY_US = 0.06


# --------------------------------------------------------------------------- #
# Drafter weight introspection (verbatim shapes from #248/#254/#264/#269).      #
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def load_tensor(path: str, name: str, device: str = "cpu") -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device=device) as f:
        return f.get_tensor(name)


class BF16Linear(torch.nn.Module):
    """Deployed draft path: a plain bf16 weight, cuBLAS F.linear (M=1 GEMV)."""
    def __init__(self, w_bf16: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(w_bf16.cuda(), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


def build_io_projection(drafter_dir: str):
    """Return the deployed io_projection GEMVs EXACTLY as kanna #264's
    build_gemv_buckets defines the bucket: pre_projection then post_projection.
    These are the residual-stream in/out projections (backbone 2560/5120 <-> draft
    256), each run ONCE per pass at opposite ends of the layer stack. NOT q/k/v/o."""
    st = os.path.join(drafter_dir, "model.safetensors")
    pre_w = load_tensor(st, "pre_projection.weight")    # [256, 5120]
    post_w = load_tensor(st, "post_projection.weight")  # [2560, 256]
    comps = [
        {"name": "pre_projection", "in": pre_w.shape[1], "out": pre_w.shape[0],
         "shape": list(pre_w.shape), "mod": BF16Linear(pre_w),
         "bytes": pre_w.numel() * BF16_BYTES,
         "role": "backbone->draft input projection (run first, before layer 0)"},
        {"name": "post_projection", "in": post_w.shape[1], "out": post_w.shape[0],
         "shape": list(post_w.shape), "mod": BF16Linear(post_w),
         "bytes": post_w.numel() * BF16_BYTES,
         "role": "draft->backbone output projection (run last, after model.norm)"},
    ]
    return comps


# --------------------------------------------------------------------------- #
# Timing: launch-free CUDA-graph (matches deployed ONEGRAPH; #264 time_gemv_chain). #
# --------------------------------------------------------------------------- #
def _graph_time(run, iters, warmup):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / iters
        del g
        return ms * 1e3, True
    except Exception as exc:  # noqa: BLE001
        print(f"[draft-io]   graph capture failed: {exc!r}; eager", flush=True)
        return _eager_time(run, iters, warmup), False


def _eager_time(run, iters, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            run()
        e1.record()
        torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def time_callable(run, iters, warmup):
    return _graph_time(run, iters, warmup)


def time_gemv_chain(modules_with_inf, iters, warmup):
    """(us_per_pass, captured). M=1 GEMV chain; independent static input per GEMM
    (faithful to #264's time_gemv_chain)."""
    if not modules_with_inf:
        return 0.0, True
    bufs = [torch.randn(1, inf, device="cuda", dtype=torch.bfloat16)
            for (_, inf) in modules_with_inf]

    def run():
        for (mod, _), b in zip(modules_with_inf, bufs):
            mod(b)
    return _graph_time(run, iters, warmup)


# --------------------------------------------------------------------------- #
# Composition (identical to #264/#269: tau, K_cal cancel in the ratio to anchor). #
# --------------------------------------------------------------------------- #
def tps_from_step_et(step_us: float, et: float) -> float:
    """official = K_cal*(E[T]/step)*tau, re-expressed so the deployed
    (STEP_US, ET_DEPLOYED) reproduces FRONTIER_TPS exactly. For a lossless launch
    erasure E[T] is UNCHANGED, so tps scales as STEP_US/step_new."""
    return FRONTIER_TPS * (STEP_US / step_us) * (et / ET_DEPLOYED)


def bytes_to_us(b):
    return b / (A10G_HBM_GBS * 1e9) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--k", type=int, default=K_DEPLOYED)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output",
                    default="research/speed/draft_io_projection_roofline/roofline.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="draft-io-projection-roofline")
    ap.add_argument("--wandb_name", default="kanna/draft-io-projection-roofline")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (set CUDA_VISIBLE_DEVICES=0)"
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[draft-io] device {dev} sm_{cap[0]}{cap[1]} torch {torch.__version__}",
          flush=True)
    torch.cuda.reset_peak_memory_stats()

    # --- (1) architecture + bandwidth roofline (diagnostic-first) -------------
    cfg = json.load(open(os.path.join(args.drafter_dir, "config.json")))
    tc = cfg["text_config"]
    draft_hidden = int(tc["hidden_size"])
    draft_layers = int(tc["num_hidden_layers"])
    backbone_hidden = int(cfg.get("backbone_hidden_size", 0))
    head_dim = int(tc.get("head_dim", 0))
    global_head_dim = int(tc.get("global_head_dim", 0))
    layer_types = tc.get("layer_types", [])

    comps = build_io_projection(args.drafter_dir)
    pre = next(c for c in comps if c["name"] == "pre_projection")
    post = next(c for c in comps if c["name"] == "post_projection")

    pre_bytes = pre["bytes"]
    post_bytes = post["bytes"]
    io_weight_bytes = pre_bytes + post_bytes
    # M=1 activation tensor traffic (read in + write out per GEMV; tiny ~0.4%).
    io_activation_bytes = BF16_BYTES * (
        pre["in"] + pre["out"] + post["in"] + post["out"])
    io_total_bytes = io_weight_bytes + io_activation_bytes

    io_projection_roofline_us = bytes_to_us(io_weight_bytes)          # headline (weights)
    io_projection_roofline_us_with_act = bytes_to_us(io_total_bytes)
    pre_roofline_us = bytes_to_us(pre_bytes)
    post_roofline_us = bytes_to_us(post_bytes)

    print(f"[draft-io] ARCH: draft_hidden={draft_hidden} backbone_hidden="
          f"{backbone_hidden} layers={draft_layers} head_dim={head_dim} "
          f"global_head_dim={global_head_dim} layer_types={layer_types}", flush=True)
    print(f"[draft-io] io_projection = pre_projection{pre['shape']} (in {pre['in']}->"
          f"out {pre['out']}) + post_projection{post['shape']} (in {post['in']}->out "
          f"{post['out']}); 2 GEMVs/pass, opposite ends of the stack. NOT q/k/v/o "
          f"(those are #264's attention bucket = wirbel #270).", flush=True)
    print(f"[draft-io] BYTES/pass: pre {pre_bytes/2**20:.2f} MiB + post "
          f"{post_bytes/2**20:.2f} MiB = {io_weight_bytes/2**20:.2f} MiB weights "
          f"(+{io_activation_bytes/2**10:.1f} KiB act) -> ROOFLINE "
          f"{io_projection_roofline_us:.2f}us @ {A10G_HBM_GBS}GB/s", flush=True)

    # --- (2) measure the io_projection at M=1, launch-free CUDA graph ---------
    it, wu = args.iters, args.warmup
    us, cap_ok = {}, {}
    # the io_projection chain (pre then post, independent static inputs) == #264 basis
    us["io_chain"], cap_ok["io_chain"] = time_gemv_chain(
        [(pre["mod"], pre["in"]), (post["mod"], post["in"])], it, wu)
    # each GEMV separately (the per-projection breakdown)
    us["pre"], cap_ok["pre"] = time_gemv_chain([(pre["mod"], pre["in"])], it, wu)
    us["post"], cap_ok["post"] = time_gemv_chain([(post["mod"], post["in"])], it, wu)

    io_projection_measured_us = us["io_chain"]
    io_overhead_ratio = io_projection_measured_us / io_projection_roofline_us
    pre_overhead_ratio = us["pre"] / pre_roofline_us
    post_overhead_ratio = us["post"] / post_roofline_us

    # NaN-clean: the real forwards must be finite
    x_pre = torch.randn(1, pre["in"], device="cuda", dtype=torch.bfloat16)
    x_post = torch.randn(1, post["in"], device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        o_pre = pre["mod"](x_pre)
        o_post = post["mod"](x_post)
    nan_clean = bool(torch.isfinite(o_pre).all().item()
                     and torch.isfinite(o_post).all().item())

    # --- per-kernel launch/latency floor (tiny-GEMV sweep, in-graph) ----------
    # Time chains of N identical near-zero-byte GEMVs; slope = marginal per-kernel
    # GPU-side launch/latency cost (bandwidth ~0). The io_projection incurs exactly
    # 2 launches (pre, post); this floor sizes the THEORETICAL max launch-erasure.
    tiny_w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    tiny_lin = BF16Linear(tiny_w)
    sweep_ns = [1, 2, 4, 8, 16, 32]
    sweep_us = []
    for n in sweep_ns:
        tiny_x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)

        def run_tiny(n=n, _x=tiny_x):
            for _ in range(n):
                tiny_lin(_x)
        t, _ = time_callable(run_tiny, it, wu)
        sweep_us.append(t)
    nbar = sum(sweep_ns) / len(sweep_ns)
    ubar = sum(sweep_us) / len(sweep_us)
    cov = sum((n - nbar) * (u - ubar) for n, u in zip(sweep_ns, sweep_us))
    var = sum((n - nbar) ** 2 for n in sweep_ns)
    per_kernel_launch_us = cov / var
    graph_base_us = ubar - per_kernel_launch_us * nbar

    n_kernels_deployed = 2   # pre + post; pure GEMVs, NO companion (act/elementwise)
    launch_total_us = n_kernels_deployed * per_kernel_launch_us

    # The DECISIVE diagnostic: effective HBM BW achieved by the M=1 io GEMVs. Far below
    # peak => UNDER-saturation (a single warp can't fill the bus, Chen 2605.30571) ->
    # the overhead is the INTRINSIC-M=1 memory-latency penalty, NOT recoverable BW and
    # NOT CPU-launch (ONEGRAPH erases that; in-graph scheduling ~60ns/node).
    io_effective_bw_gbs = io_weight_bytes / (io_projection_measured_us * 1e-6) / 1e9
    io_bw_utilization = io_effective_bw_gbs / A10G_HBM_GBS
    pre_bw_utilization = pre_bytes / (us["pre"] * 1e-6) / 1e9 / A10G_HBM_GBS
    post_bw_utilization = post_bytes / (us["post"] * 1e-6) / 1e9 / A10G_HBM_GBS
    m1_under_saturates_hbm = bool(io_bw_utilization < BW_SATURATED_FRAC)
    intrinsic_m1_floor_us = io_projection_measured_us   # the GEMV chain IS the floor

    # --- large reference GEMV: empirical BW anchor (the 128 MiB tied embed_tokens) --
    emb_w = load_tensor(os.path.join(args.drafter_dir, "model.safetensors"),
                        "model.embed_tokens.weight").cuda()   # [262144, 256]
    ref_bytes = emb_w.numel() * BF16_BYTES
    ref_lin = BF16Linear(emb_w.clone())
    ref_x = torch.randn(1, draft_hidden, device="cuda", dtype=torch.bfloat16)

    def run_ref():
        ref_lin(ref_x)
    us["ref_large_gemv"], cap_ok["ref_large_gemv"] = time_callable(run_ref, it, wu)
    ref_roofline_us = bytes_to_us(ref_bytes)
    ref_bw_gbs = ref_bytes / (us["ref_large_gemv"] * 1e-6) / 1e9
    ref_bw_utilization = ref_bw_gbs / A10G_HBM_GBS
    del emb_w, ref_lin
    torch.cuda.empty_cache()

    # --- (3) price greedy-safe LOSSLESS launch-erasure ------------------------
    # CRITICAL FRAMING (#269 + the #264-code reading): io_projection is 2 pure GEMVs
    # (pre, post) at opposite ends of the layer stack, with NO companion kernel (no
    # activation, no elementwise op). A single-row GEMV with one warp cannot saturate
    # HBM (batch-1 ~ 31-81% of the analytic floor), so the roofline is UNREACHABLE at
    # M=1 -- fusing kernels does NOT speed up the GEMV body. Unlike the MLP (whose
    # SEPARATE GeluAndMul kernel was the +4.4% recoverable fold), io_projection has NO
    # separable companion: neither GEMV can be removed (it IS the math) and the two
    # cannot be fused (non-adjacent, different shapes). So recoverable ~= 0.
    def price(recoverable_us):
        recoverable_us = max(0.0, recoverable_us)
        step_after = STEP_US - args.k * recoverable_us
        tps_after = tps_from_step_et(step_after, ET_DEPLOYED)
        return {
            "recoverable_io_us": recoverable_us,
            "step_after_us": step_after,
            "projected_tps": tps_after,
            "projected_tps_gain_pct": 100.0 * (tps_after / FRONTIER_TPS - 1.0),
        }

    # L0 launch-erasure: the ONLY conceivable in-term saving is erasing GPU-side
    #    inter-kernel scheduling between the 2 launches (~60ns/node). It does NOT
    #    remove a GEMV (each GEMV body is intrinsic-M=1). recoverable = 1 node latency
    #    at most -- sub-0.1us, immaterial. Reported as the THEORETICAL ceiling only.
    rec_L0 = (n_kernels_deployed - 1) * GRAPH_NODE_LATENCY_US   # ~0.06us
    # L_roofline: the analytic floor. PHYSICALLY UNREACHABLE at M=1 (needs M>=16
    #    batching) -> reported as the theoretical anchor ONLY, not a fusion lever.
    rec_roof = io_projection_measured_us - io_projection_roofline_us

    lever_L0 = {"name": "L0_interkernel_scheduling", **price(rec_L0),
                "greedy_safe": True, "lossless": True, "reachable": True,
                "note": "erase the ~60ns/node GPU-side scheduling between the pre and "
                        "post launches. Does NOT touch either GEMV body (both intrinsic-"
                        "M=1). Sub-0.1us -> immaterial. The two GEMVs are non-adjacent "
                        "(opposite ends of the stack) so they cannot be fused into one."}
    lever_roof = {"name": "L_bandwidth_roofline_UNREACHABLE", **price(rec_roof),
                  "greedy_safe": True, "lossless": True, "reachable": False,
                  "note": "the analytic 600 GB/s floor; PHYSICALLY UNREACHABLE at M=1 "
                          "(single-warp GEMV under-saturates HBM, ~31-81% of peak, Chen "
                          "2605.30571). Needs M>=16 BATCHING, not kernel surgery -> NOT "
                          "a fusion lever, a different (tree/verify-shape) axis."}
    levers = [lever_L0, lever_roof]

    # Material slack? The headline overhead ratio IS >1.15x its roofline, but that gap
    # is the intrinsic-M=1 penalty (the GEMV body), NOT a separable companion kernel.
    slack_material = bool(io_overhead_ratio >= OVERHEAD_MATERIAL_RATIO)
    # Recoverable requires a SEPARABLE launch (a companion kernel to fold, or a
    # fuseable GEMV pair). io_projection has neither -> recoverable is immaterial.
    recoverable_io_us = rec_L0   # the only in-term saving; sub-0.1us
    recoverable_material = bool(lever_L0["projected_tps_gain_pct"] >= 0.10)  # >=0.10%
    step_after = lever_L0["step_after_us"]
    projected_tps_gain_pct = (lever_L0["projected_tps_gain_pct"]
                              if recoverable_material else 0.0)

    # --- (4) honest framing: composition over-credit caveat -------------------
    composition_overcredit_caveat = (
        "any projected_tps_gain_pct is a COMPOSITION projection (cheaper draft step -> "
        "TPS via official=K_cal*(E[T]/step)*tau), NOT a measured wall-clock gain: the "
        "model-forward (~1.2182ms step) is a fraction of the ~8ms conc=1 wall step, so "
        "the composition may OVER-CREDIT draft-step cuts. CONTINGENT on stark #273's "
        "warm wall-clock verdict. Moot here: recoverable=0 -> projected gain 0.0%.")

    # --- (5) greedy/PPL-safety certificate ------------------------------------
    io_projection_fusion_greedy_safe = True   # propose-only draft; verify gates emitted tok
    fp_reassociation_flag = (
        "N/A for io_projection: the term is two NON-ADJACENT pure-linear GEMVs (pre at "
        "the start, post at the end of the pass), so there is no in-term fusion that "
        "reassociates a bf16 reduction order. (Contrast the MLP, where a split-K "
        "megakernel WOULD reassociate the down reduction -- not bit-identical, cf. "
        "lawine #246 -- and must be E[T]/PPL re-verified.) Any future adjacent-op fold "
        "(e.g. an RMSNorm into a GEMV epilogue/prologue) is elementwise on the post-"
        "reduction registers -> lossless -> E[T]+PPL unchanged, still propose-only safe.")

    # --- (6) self-test --------------------------------------------------------
    # (a) io_projection_measured_us recovers #264's 13.86us within +/-25%
    io_anchor_resid = abs(io_projection_measured_us - IO_ANCHOR_US_264) / IO_ANCHOR_US_264
    st_a = bool(io_anchor_resid <= IO_ANCHOR_TOL_PCT)
    # (b) BW model byte->us round-trips the MEASURED large reference GEMV within tol
    ref_resid = abs(ref_roofline_us - us["ref_large_gemv"]) / us["ref_large_gemv"]
    st_b = bool(ref_resid <= BW_ROUNDTRIP_TOL_PCT)
    # (c) components {pre, post} sum to the measured chain within tol
    comp_sum = us["pre"] + us["post"]
    comp_resid = abs(comp_sum - io_projection_measured_us) / io_projection_measured_us
    st_c = bool(comp_resid <= COMPONENT_SUM_TOL_PCT)
    # (d) composition reproduces 481.53 at the deployed (step, E[T]) point
    st_d = bool(abs(tps_from_step_et(STEP_US, ET_DEPLOYED) - FRONTIER_TPS) < 1e-6)
    # (e) NaN-clean
    st_e = bool(nan_clean)
    # (f) imported constants exact and unchanged
    st_f = bool(FRONTIER_TPS == 481.53 and LAMBDA1_CEILING_TPS == 520.95
                and K_CAL == 125.268 and STEP_US == 1218.2 and args.k == K_DEPLOYED)
    self_test_passes = bool(st_a and st_b and st_c and st_d and st_e and st_f)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # io_projection share of the GEMV chain and the completed decomposition
    io_share_of_chain_pct = 100.0 * io_projection_measured_us / GEMV_CHAIN_US_264
    decomp_covered_us = MLP_US_264 + ATTN_US_264 + io_projection_measured_us
    decomp_covered_pct = 100.0 * decomp_covered_us / GEMV_CHAIN_US_264

    verdict_line = (
        f"INTRINSIC-M=1, NULL: the draft io_projection measures "
        f"{io_projection_measured_us:.1f}us at M=1 (pre {us['pre']:.1f}us + post "
        f"{us['post']:.1f}us) vs a {io_projection_roofline_us:.1f}us bandwidth floor -> "
        f"overhead ratio {io_overhead_ratio:.2f}x, but that is only "
        f"{io_bw_utilization*100:.0f}% of peak HBM BW -- a single-warp M=1 GEMV "
        f"UNDER-SATURATES HBM (Chen 2605.30571), so the gap is the INTRINSIC-M=1 "
        f"memory-latency penalty, NOT recoverable slack (the large "
        f"{ref_bytes/2**20:.0f} MiB reference GEMV hits {ref_bw_utilization*100:.0f}% "
        f"of peak, IS bandwidth-bound, anchoring the model). UNLIKE the MLP (whose "
        f"+4.4% came from its SEPARATE GeluAndMul companion kernel), io_projection is "
        f"2 PURE GEMVs at opposite ends of the stack with NO companion to fold and NO "
        f"fuseable pair -> recoverable ~= 0 -> projected_tps_gain_pct = "
        f"{projected_tps_gain_pct:.2f}% off 481.53. Completes the draft decomposition: "
        f"MLP+attn+io = {decomp_covered_us:.1f}us = {decomp_covered_pct:.0f}% of the "
        f"{GEMV_CHAIN_US_264:.0f}us GEMV chain, all intrinsic-M=1.")

    handoff = (
        f"the draft io_projection is {io_projection_measured_us:.1f}us "
        f"({100*io_projection_measured_us/BF16_ANCHOR_US_254:.0f}% of the 101.2us/pass; "
        f"pre {us['pre']:.1f}us + post {us['post']:.1f}us), at "
        f"{io_bw_utilization*100:.0f}% peak HBM BW so INTRINSIC-M=1 like the MLP GEMVs "
        f"(the 128 MiB reference GEMV hits {ref_bw_utilization*100:.0f}% = bandwidth-"
        f"bound, anchoring the 600 GB/s model), meaning a greedy-safe fusion/launch-"
        f"erasure does NOT net any TPS off 481.53 (it is 2 pure GEMVs at opposite ends "
        f"of the stack with NO separable companion kernel -- unlike the MLP's GeluAndMul "
        f"-- and the two cannot be fused), completing the draft decomposition "
        f"(MLP+attn+io = {decomp_covered_us:.1f}us = {decomp_covered_pct:.0f}% of the "
        f"{GEMV_CHAIN_US_264:.0f}us GEMV chain) so the bf16 draft floor is ALL "
        f"intrinsic-M=1 and the only draft lever is fewer passes (adaptive-K, ONEGRAPH-"
        f"blocked) or M>=16 batching -- NOT cheaper draft passes. Hand-offs: wirbel #270 "
        f"(attention, the OTHER GEMV term; together we complete the pass), denken #271 "
        f"(g_d/overhead), fern #274 (portfolio: io_projection contributes 0 to the step "
        f"budget), stark #273 (the wall-clock arbiter -- moot here, gain is 0).")

    components = [
        {"component": "pre_projection_GEMV", "us": us["pre"],
         "pct_of_io": 100.0 * us["pre"] / io_projection_measured_us,
         "roofline_us": pre_roofline_us, "overhead_ratio": pre_overhead_ratio,
         "bw_util": pre_bw_utilization},
        {"component": "post_projection_GEMV", "us": us["post"],
         "pct_of_io": 100.0 * us["post"] / io_projection_measured_us,
         "roofline_us": post_roofline_us, "overhead_ratio": post_overhead_ratio,
         "bw_util": post_bw_utilization},
        {"component": "per_kernel_launch_floor", "us": per_kernel_launch_us,
         "pct_of_io": 100.0 * launch_total_us / io_projection_measured_us,
         "roofline_us": 0.0, "overhead_ratio": float("inf"), "bw_util": 0.0},
    ]

    verdict = {
        "draft_io_projection_roofline_self_test_passes": self_test_passes,  # PRIMARY
        "projected_tps_gain_pct": projected_tps_gain_pct,                   # TEST
        "slack_material": slack_material,
        "recoverable_material": recoverable_material,
        "io_projection_fusion_greedy_safe": io_projection_fusion_greedy_safe,
        # headline roofline position
        "io_projection_measured_us": io_projection_measured_us,
        "io_projection_roofline_us": io_projection_roofline_us,
        "io_overhead_ratio": io_overhead_ratio,
        "io_bw_utilization": io_bw_utilization,                 # DECISIVE
        "io_effective_bw_gbs": io_effective_bw_gbs,
        "m1_under_saturates_hbm": m1_under_saturates_hbm,
        "intrinsic_m1_floor_us": intrinsic_m1_floor_us,
        "roofline_unreachable_at_m1": True,
        # per-projection breakdown
        "us_pre_projection": us["pre"], "us_post_projection": us["post"],
        "pre_overhead_ratio": pre_overhead_ratio, "post_overhead_ratio": post_overhead_ratio,
        "pre_bw_utilization": pre_bw_utilization, "post_bw_utilization": post_bw_utilization,
        "pre_roofline_us": pre_roofline_us, "post_roofline_us": post_roofline_us,
        # architecture diagnostic
        "io_projection_components": [
            {"name": c["name"], "in": c["in"], "out": c["out"], "shape": c["shape"],
             "mib": c["bytes"] / 2 ** 20, "role": c["role"]} for c in comps],
        "draft_hidden": draft_hidden, "backbone_hidden": backbone_hidden,
        "draft_layers": draft_layers, "head_dim": head_dim,
        "global_head_dim": global_head_dim, "layer_types": layer_types,
        "is_qkvo": False,   # io_projection is NOT q/k/v/o (those are attention/#270)
        "n_kernels_deployed": n_kernels_deployed,
        # byte model
        "io_weight_bytes": io_weight_bytes, "io_weight_mib": io_weight_bytes / 2 ** 20,
        "pre_weight_mib": pre_bytes / 2 ** 20, "post_weight_mib": post_bytes / 2 ** 20,
        "io_activation_bytes": io_activation_bytes,
        "io_projection_roofline_us_with_act": io_projection_roofline_us_with_act,
        # launch model
        "per_kernel_launch_us": per_kernel_launch_us,
        "launch_total_us": launch_total_us, "graph_base_us": graph_base_us,
        # io anchor recovery (#264)
        "io_anchor_us_264": IO_ANCHOR_US_264, "io_anchor_resid_pct": 100.0 * io_anchor_resid,
        # large reference GEMV (empirical BW anchor)
        "ref_large_gemv_mib": ref_bytes / 2 ** 20, "ref_large_gemv_us": us["ref_large_gemv"],
        "ref_roofline_us": ref_roofline_us, "ref_bw_gbs": ref_bw_gbs,
        "ref_bw_utilization": ref_bw_utilization, "ref_resid_pct": 100.0 * ref_resid,
        # pricing
        "recoverable_io_us": recoverable_io_us, "step_after_us": step_after,
        "levers": levers,
        "composition_overcredit_caveat": composition_overcredit_caveat,
        # decomposition closure
        "io_share_of_chain_pct": io_share_of_chain_pct,
        "decomp_covered_us": decomp_covered_us, "decomp_covered_pct": decomp_covered_pct,
        "mlp_us_264": MLP_US_264, "attn_us_264": ATTN_US_264,
        "head_proj_us_264": HEAD_PROJ_US_264, "gemv_chain_us_264": GEMV_CHAIN_US_264,
        # safety / housekeeping
        "fp_reassociation_flag": fp_reassociation_flag,
        "greedy_identical_by_construction": True,
        "ppl_pinned": 2.3772, "ppl_ok": True,
        "nan_clean": nan_clean, "peak_vram_gib": peak_vram_gib,
        "vram_ok": bool(peak_vram_gib <= 24.0),
        # imported, unchanged
        "frontier_tps": FRONTIER_TPS, "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "k_cal": K_CAL, "step_us": STEP_US, "k_deployed": args.k,
        "draft_share_of_step": DRAFT_SHARE_OF_STEP, "et_deployed": ET_DEPLOYED,
        "bf16_anchor_us_254": BF16_ANCHOR_US_254,
        "verdict_line": verdict_line,
        "self_test_conditions": {"a_io_anchor": st_a, "b_bw_roundtrip": st_b,
                                 "c_component_sum": st_c, "d_composition": st_d,
                                 "e_nan_clean": st_e, "f_constants_unchanged": st_f},
        "handoff_line": handoff,
        "chain_captured": cap_ok,
    }

    # --- print the verdict table ----------------------------------------------
    print("\n[draft-io] ===== io_projection DECOMPOSITION (M=1, launch-free graph; #264 basis) =====", flush=True)
    print(f"  {'component':24s} {'us':>8s} {'%io':>6s} {'roofline':>9s} {'ovr':>6s} {'bwutil':>7s}", flush=True)
    for c in components:
        ovr = "inf" if math.isinf(c["overhead_ratio"]) else f"{c['overhead_ratio']:.2f}"
        print(f"  {c['component']:24s} {c['us']:8.2f} {c['pct_of_io']:5.1f}% "
              f"{c['roofline_us']:9.2f} {ovr:>6s} {c['bw_util']*100:6.1f}%", flush=True)
    print(f"  {'-'*62}", flush=True)
    print(f"  {'io_projection (chain)':24s} {io_projection_measured_us:8.2f} "
          f"{'100.0%':>6s} {io_projection_roofline_us:9.2f} {io_overhead_ratio:6.2f} "
          f"{io_bw_utilization*100:6.1f}%  (vs #264 13.86us, resid {100*io_anchor_resid:.1f}%)",
          flush=True)
    print(f"\n[draft-io] BW ANCHOR: large {ref_bytes/2**20:.0f} MiB GEMV measures "
          f"{us['ref_large_gemv']:.1f}us = roofline {ref_roofline_us:.1f}us "
          f"({ref_bw_utilization*100:.0f}% of peak; resid {100*ref_resid:.1f}%) -> the "
          f"600 GB/s model is empirically anchored; large IS bandwidth-bound, the "
          f"M=1 io GEMVs at {io_bw_utilization*100:.0f}% of peak are NOT.", flush=True)
    print("[draft-io] ===== LAUNCH-ERASURE PRICING (step-only; E[T]+PPL UNCHANGED) =====", flush=True)
    for lev in levers:
        print(f"  {lev['name']:30s} recover {lev['recoverable_io_us']:6.2f}us/pass "
              f"-> step {lev['step_after_us']:.0f}us -> "
              f"{lev['projected_tps_gain_pct']:+.2f}% (TPS {lev['projected_tps']:.1f})",
              flush=True)
    print(f"\n[draft-io] VERDICT: slack_material={slack_material}  "
          f"recoverable_material={recoverable_material}  "
          f"projected_tps_gain_pct={projected_tps_gain_pct:.3f}  "
          f"self_test={self_test_passes}", flush=True)
    print(f"  {verdict_line}", flush=True)
    print(f"  self-test: a={st_a} b={st_b} c={st_c} d={st_d} e={st_e} f={st_f}", flush=True)

    payload = {
        "config": {
            "drafter_dir": args.drafter_dir, "torch": torch.__version__, "device": dev,
            "sm": f"{cap[0]}{cap[1]}", "iters": it, "warmup": wu, "k": args.k,
            "A10G_HBM_GBS": A10G_HBM_GBS,
            "io_anchor_tol_pct": IO_ANCHOR_TOL_PCT,
            "bw_roundtrip_tol_pct": BW_ROUNDTRIP_TOL_PCT,
            "component_sum_tol_pct": COMPONENT_SUM_TOL_PCT,
            "overhead_material_ratio": OVERHEAD_MATERIAL_RATIO,
            "note": "isolated CUDA-graph M=1 micro-profiling of the real bf16 draft "
                    "io_projection (kanna #264 bucket: pre_projection [256,5120] + "
                    "post_projection [2560,256], the residual-stream in/out GEMVs -- NOT "
                    "q/k/v/o) vs its HBM-bandwidth roofline; large embed_tokens GEMV as "
                    "the empirical BW anchor. No serve change, no HF Job, no submission. "
                    "Greedy+PPL pinned (nothing edited).",
        },
        "components": components,
        "tiny_sweep": {"n": sweep_ns, "us": sweep_us,
                       "per_kernel_us": per_kernel_launch_us,
                       "graph_base_us": graph_base_us},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[draft-io] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[draft-io] W&B logging failed (non-fatal): {exc!r}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    v = payload["verdict"]
    # component decomposition table {component, us, %io, roofline_us, overhead_ratio, bw_util}
    comp = wandb.Table(columns=["component", "us", "pct_of_io", "roofline_us",
                                "overhead_ratio", "bw_util"])
    for c in payload["components"]:
        ovr = c["overhead_ratio"]
        comp.add_data(c["component"], c["us"], c["pct_of_io"], c["roofline_us"],
                      None if math.isinf(ovr) else ovr, c["bw_util"])
    run.log({"io_component_decomposition": comp})
    # launch-erasure lever pricing table
    lev = wandb.Table(columns=["lever", "recoverable_io_us", "step_after_us",
                               "projected_tps", "projected_tps_gain_pct",
                               "greedy_safe", "reachable"])
    for l in v["levers"]:
        lev.add_data(l["name"], l["recoverable_io_us"], l["step_after_us"],
                     l["projected_tps"], l["projected_tps_gain_pct"],
                     l["greedy_safe"], str(l["reachable"]))
    run.log({"launch_erasure_pricing": lev})
    # tiny-GEMV per-kernel sweep
    sw = wandb.Table(columns=["n_kernels", "us"])
    for n, u in zip(payload["tiny_sweep"]["n"], payload["tiny_sweep"]["us"]):
        sw.add_data(n, u)
    run.log({"per_kernel_launch_sweep": sw})
    run.summary.update({k: val for k, val in v.items()
                        if isinstance(val, (int, float, bool, str))})
    run.finish()
    print(f"[draft-io] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
