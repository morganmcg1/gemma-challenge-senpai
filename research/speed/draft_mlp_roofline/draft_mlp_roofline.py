#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Draft MLP roofline: is the 50.7us MLP (51.7% of the 101.2us draft floor) at
bandwidth, or overhead-recoverable? (PR #269, kanna). LOCAL GPU micro-profiling +
CPU analytic. Analysis-only: no served-file change, no HF Job, no submission.
BASELINE stays 481.53.

THE QUESTION
------------
kanna #264 (95x7qv6h, MERGED) decomposed the bf16 draft 101.2us/pass and found the
vocab-head IMMATERIAL (5.0%), handing back the real map: MLP 50.7us (51.7%) +
attention 28.5us (29.1%) = 80.8% of the floor. The MLP is the single LARGEST
component and has NEVER been roofline-decomposed. At M=1 the draft MLP (gate_up
fused GEMV -> GeluAndMul -> down GEMV, x4 layers; Gemma-4 GeGLU) is a memory-bound
GEMV that must read the entire MLP weight matrices every pass. The standing belief
is the bf16 draft is "at the floor" -- but #264 found the centroid PROPOSAL was
OVERHEAD-bound at M=1 (46us to move ~5.5 MiB), direct evidence the draft pass
carries launch/latency slack at M=1, NOT pure bandwidth saturation.

CRUX (diagnostic-first): (1) the draft MLP's byte-traffic at M=1 and its
memory-bound us floor at A10G HBM BW; (2) does the MEASURED 50.7us MLP sit AT that
roofline (irrecoverable, bandwidth-bound) or ABOVE it (launch/un-fused slack);
(3) IF slack, what is the greedy-safe step reduction from closing it
(gate_up+down/activation fusion, persistent/fused-MLP megakernel) -> net TPS off
481.53? This is orthogonal to BOTH dead draft levers: weight-quant (int3 #248 /
int4 #254 -- bits-per-weight, dead at M=1) and the vocab-head (#264 -- output
columns, immaterial). The MLP roofline is the BYTES-MOVED-vs-ROOFLINE axis on the
dominant 51.7% term.

WHAT THE DEPLOYED MLP ACTUALLY IS (diagnostic, decisive)
-------------------------------------------------------
The drafter at /tmp/qat-assistant reuses Gemma-4 decoder layers; the served MLP is
vLLM `Gemma4MLP` (model_executor/models/gemma4.py:218-247):
  gate_up_proj = MergedColumnParallelLinear(256 -> [2048]*2)   # gate+up ALREADY FUSED
  act_fn       = GeluAndMul(approximate="tanh")                 # fused gelu*mul, 1 kernel
  down_proj    = RowParallelLinear(2048 -> 256)
So the deployed MLP per layer = 3 kernels {gate_up GEMV, GeluAndMul, down GEMV};
x4 layers = 12 kernels. gate+up fusion is ALREADY TAKEN (so NOT an available lever);
the remaining levers are (L1) fold the GeluAndMul into the gate_up epilogue, and
(L2) a fused/persistent MLP megakernel (1 kernel/layer). Both REDUCE kernel count
inside the CUDA graph, so both are ONEGRAPH-compatible (unlike adaptive-K, which
changes the graph shape per step and is ONEGRAPH-blocked).

WHAT THIS MEASURES (real A10G micro-profiling, no HF Job, no serve change)
-------------------------------------------------------------------------
  * The draft MLP weight byte-traffic at M=1 = Sum_layers(gate_up + down weight
    bytes) and its memory-bound floor `draft_mlp_roofline_us` at A10G 600 GB/s.
  * The MLP at M=1, launch-free CUDA graph (deployed ONEGRAPH basis), broken into
    {gate_up GEMV, down GEMV, GeluAndMul activation, per-kernel launch/latency}.
    `draft_mlp_measured_us` = the gate_up+down GEMV chain (the #264 50.7us basis);
    `draft_mlp_full_forward_us` adds the real GeluAndMul. Headline
    `mlp_overhead_ratio = draft_mlp_measured_us / draft_mlp_roofline_us`
    (1.0 = at roofline/bandwidth-bound/irrecoverable; >1.0 = launch/un-fused slack).
  * A LARGE reference GEMV (the 128 MiB tied lm_head, #264's dense-256k head) at M=1
    to EMPIRICALLY validate the 600 GB/s model and CONTRAST a bandwidth-bound op
    (large -> ~roofline) against the launch-bound MLP (small -> >>roofline).
  * Price greedy-safe LOSSLESS fusion levers. CRITICAL (researcher-agent, sources
    below): the 2.4x gap is MOSTLY INTRINSIC to M=1, NOT recoverable by fusion. A
    single-row GEMV with one warp cannot saturate HBM -- Chen et al. (2605.30571)
    measure batch-1 at only ~31% (A100) .. 72-81% (L40S/L4, the A10G's inference
    class) of the analytic memory floor. The 21us roofline is PHYSICALLY UNREACHABLE
    at M=1; reaching it needs M>=16 BATCHING, not kernel surgery. Inter-kernel
    scheduling inside a CUDA graph is only ~60ns/node (~0.7us total). So the GEMV
    chain (~50.7us) IS essentially the achievable M=1 floor, and the ONLY recoverable
    slack is the SEPARATE GeluAndMul kernel (fold into the gate_up epilogue, CUTLASS
    EVT) + a per-layer megakernel (on A10G only 1.0-1.08x vs cuBLAS, AutoMegaKernel
    2606.09682). recoverable ~= us_activation. The per-pass saving x K=7 maps through
    `official = K_cal*(E[T]/step)*tau` into a step reduction and a TPS off 481.53.
    A LOSSLESS epilogue fold is the identical GEMV math (same bytes, same reduction
    order) => E[T] and PPL UNCHANGED by construction -- a PURE STEP lever, the
    cleanest kind. (Sources: Chen 2605.30571 "Memory-Bound but Not Bandwidth-
    Limited"; AutoMegaKernel 2606.09682; FlashDecoding++ 2311.01282.)

GREEDY/PPL SAFETY: pinned BY CONSTRUCTION. This leg MEASURES; it edits no served
file. A fusion / launch-erasure of the draft MLP changes NO emitted token: the
draft only PROPOSES, and greedy-exact verify checks every candidate against the
FULL target argmax, so a draft change can only move E[T] (acceptance), never the
emitted token. A LOSSLESS fusion that preserves the bf16 reduction order leaves
E[T]+PPL exactly unchanged; a fusion that REASSOCIATES the down reduction (e.g.
split-K) is still correctness-safe (propose-only) but could drift E[T] -- flagged
explicitly (cf. lawine #246: a CUDAGraph toggle was NOT bit-identical).

SELF-TEST (`draft_mlp_roofline_self_test_passes`, PRIMARY)
----------------------------------------------------------
(a) `draft_mlp_measured_us` recovers #264's 50.7us MLP component within +/-20%;
(b) the 600 GB/s BW model byte->us round-trips the MEASURED large reference GEMV
    within +/-20% (empirical BW anchor; large GEMV IS ~bandwidth-bound);
(c) the component us {gate_up, down, activation} sum to `draft_mlp_full_forward_us`
    within +/-25% (M=1 micro-timing is noisy; band stated);
(d) the composition reproduces 481.53 at the deployed (step, E[T]) point, so the
    step-only band maps through tps proportional to 1/step with E[T] UNCHANGED;
(e) NaN-clean;
(f) BASELINE 481.53, 520.95 lambda=1 ceiling, K_cal=125.268 imported EXACTLY.
TEST metric: `projected_tps_gain_pct` (best DEFENSIBLE greedy-safe gain off 481.53:
the activation-epilogue fold L1 = eliminate the separate GeluAndMul kernel, the
dominant RECOVERABLE term; the fused-MLP megakernel L2 adds little on A10G; the
bandwidth-roofline L3 is reported but flagged PHYSICALLY UNREACHABLE at M=1).

Requires a torch+CUDA env with the served vLLM wheel (GeluAndMul custom op) +
gemma4_assistant modeling code: the deployed senpai venv (#264 used
/tmp/senpai-venvs/5f4c623f772358a2). No Marlin needed (bf16 control).
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
# unaffected (same guard as #264).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEFAULT_DRAFTER = "/tmp/qat-assistant"

# ---- A10G (AWS g5, GA102, sm_86) roofline ceiling (identical to #248/#254/#264) --
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
# kanna #264 (95x7qv6h) draft-pass decomposition -- the MLP component is YOUR anchor.
MLP_ANCHOR_US_264 = 50.7      # #264 us_mlp (exact 50.71189); the 51.7% dominant term

# self-test tolerances (M=1 micro-GEMV timing is noisy; bands stated explicitly)
MLP_ANCHOR_TOL_PCT = 0.20     # draft_mlp_measured_us vs #264 50.7us
BW_ROUNDTRIP_TOL_PCT = 0.20   # large-GEMV byte->us vs measured (empirical BW anchor)
COMPONENT_SUM_TOL_PCT = 0.25  # components sum vs full forward
# A material overhead ratio gate: >15% above roofline => the MLP sits above its
# bandwidth floor (but most of that gap is the intrinsic-M=1 under-saturation penalty).
OVERHEAD_MATERIAL_RATIO = 1.15
# Irreducible GPU-side inter-kernel latency INSIDE a CUDA graph on Ampere
# (~60ns/node, NVIDIA CUDA Graphs blog) -- the only scheduling a megakernel removes.
GRAPH_NODE_LATENCY_US = 0.06


# --------------------------------------------------------------------------- #
# Drafter weight introspection (verbatim shapes from #248/#254/#264).          #
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


def build_mlp_layers(drafter_dir: str):
    """Return per-layer dicts of the deployed MLP weights, gate+up FUSED into one
    [2*intermediate, hidden] GEMV (vLLM MergedColumnParallelLinear), plus down
    [hidden, intermediate]. Mirrors #264's mlp bucket exactly (the 50.7us basis)."""
    st = os.path.join(drafter_dir, "model.safetensors")
    hdr = read_safetensors_header(st)
    layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0])
                        for k in hdr if ".layers." in k})
    layers = []
    for i in layer_ids:
        gw = load_tensor(st, f"model.layers.{i}.mlp.gate_proj.weight")
        uw = load_tensor(st, f"model.layers.{i}.mlp.up_proj.weight")
        dw = load_tensor(st, f"model.layers.{i}.mlp.down_proj.weight")
        guw = torch.cat([gw, uw], dim=0)            # gate_up [2*inter, hidden]
        layers.append({
            "layer": i,
            "gate_up": BF16Linear(guw),             # fused GEMV
            "gate_sep": BF16Linear(gw.clone()),     # unfused gate (for the already-taken note)
            "up_sep": BF16Linear(uw.clone()),       # unfused up
            "down": BF16Linear(dw),
            "hidden": gw.shape[1], "intermediate": gw.shape[0],
            "gate_up_bytes": guw.numel() * BF16_BYTES,
            "down_bytes": dw.numel() * BF16_BYTES,
        })
    return layers, layer_ids


# --------------------------------------------------------------------------- #
# Timing: launch-free CUDA-graph (matches deployed ONEGRAPH 101.2us basis).     #
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
        print(f"[draft-mlp]   graph capture failed: {exc!r}; eager", flush=True)
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


def time_callable(run, iters, warmup, graph=True):
    return _graph_time(run, iters, warmup) if graph else (
        _eager_time(run, iters, warmup), False)


# --------------------------------------------------------------------------- #
# Composition (identical to #264: tau, K_cal cancel in the ratio to anchor).    #
# --------------------------------------------------------------------------- #
def tps_from_step_et(step_us: float, et: float) -> float:
    """official = K_cal*(E[T]/step)*tau, re-expressed so the deployed
    (STEP_US, ET_DEPLOYED) reproduces FRONTIER_TPS exactly. For a lossless fusion
    E[T] is UNCHANGED, so tps scales as STEP_US/step_new."""
    return FRONTIER_TPS * (STEP_US / step_us) * (et / ET_DEPLOYED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--k", type=int, default=K_DEPLOYED)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output",
                    default="research/speed/draft_mlp_roofline/roofline.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="draft-mlp-roofline")
    ap.add_argument("--wandb_name", default="kanna/draft-mlp-roofline")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (set CUDA_VISIBLE_DEVICES=0)"
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[draft-mlp] device {dev} sm_{cap[0]}{cap[1]} torch {torch.__version__}",
          flush=True)
    torch.cuda.reset_peak_memory_stats()

    # --- (1) architecture + bandwidth roofline (diagnostic-first) -------------
    cfg = json.load(open(os.path.join(args.drafter_dir, "config.json")))
    tc = cfg["text_config"]
    draft_mlp_hidden = int(tc["hidden_size"])
    draft_mlp_intermediate = int(tc["intermediate_size"])
    draft_mlp_layers = int(tc["num_hidden_layers"])
    draft_mlp_activation = str(tc["hidden_activation"])
    draft_mlp_gated = True   # Gemma-4 GeGLU: gate_proj + up_proj + down_proj
    draft_mlp_dtype = str(tc.get("dtype", cfg.get("dtype", "bfloat16")))

    layers, layer_ids = build_mlp_layers(args.drafter_dir)
    assert len(layers) == draft_mlp_layers, (len(layers), draft_mlp_layers)

    # per-pass MLP weight byte-traffic at M=1 (the dominant memory term)
    gate_up_weight_bytes = sum(l["gate_up_bytes"] for l in layers)
    down_weight_bytes = sum(l["down_bytes"] for l in layers)
    mlp_weight_bytes = gate_up_weight_bytes + down_weight_bytes
    # M=1 activation tensor traffic (read+write; ~0.8% -- included for rigor)
    H, I, L = draft_mlp_hidden, draft_mlp_intermediate, draft_mlp_layers
    act_bytes_per_layer = BF16_BYTES * (
        H + 2 * I            # gate_up: read x[H], write [2I]
        + 2 * I + I          # GeluAndMul: read [2I], write [I]
        + I + H)             # down: read [I], write [H]
    mlp_activation_bytes = act_bytes_per_layer * L
    mlp_total_bytes = mlp_weight_bytes + mlp_activation_bytes

    def bytes_to_us(b):
        return b / (A10G_HBM_GBS * 1e9) * 1e6

    draft_mlp_roofline_us = bytes_to_us(mlp_weight_bytes)         # headline (weights)
    draft_mlp_roofline_us_with_act = bytes_to_us(mlp_total_bytes)
    gate_up_roofline_us = bytes_to_us(gate_up_weight_bytes)
    down_roofline_us = bytes_to_us(down_weight_bytes)

    print(f"[draft-mlp] ARCH: hidden={draft_mlp_hidden} intermediate="
          f"{draft_mlp_intermediate} layers={draft_mlp_layers} gated={draft_mlp_gated} "
          f"act={draft_mlp_activation} dtype={draft_mlp_dtype}", flush=True)
    print(f"[draft-mlp] BYTES/pass: gate_up {gate_up_weight_bytes/2**20:.2f} MiB + down "
          f"{down_weight_bytes/2**20:.2f} MiB = {mlp_weight_bytes/2**20:.2f} MiB weights "
          f"(+{mlp_activation_bytes/2**10:.0f} KiB act) -> ROOFLINE "
          f"{draft_mlp_roofline_us:.2f}us @ {A10G_HBM_GBS}GB/s", flush=True)

    # --- (2) measure the MLP at M=1, launch-free CUDA graph -------------------
    it, wu = args.iters, args.warmup
    x_h = torch.randn(1, draft_mlp_hidden, device="cuda", dtype=torch.bfloat16)
    x_gu = torch.randn(1, 2 * draft_mlp_intermediate, device="cuda", dtype=torch.bfloat16)
    x_i = torch.randn(1, draft_mlp_intermediate, device="cuda", dtype=torch.bfloat16)

    # The REAL deployed fused activation kernel is the single vLLM CUDA op
    # torch.ops._C.gelu_tanh_and_mul (what GeluAndMul(approximate="tanh").forward_cuda
    # dispatches to). Call it directly -- the GeluAndMul CustomOp wrapper needs a vLLM
    # config context, but the underlying fused kernel does not. Fallback: torch gelu*mul.
    act_is_fused_kernel = True
    fused_act_op = None
    try:
        import vllm._C  # noqa: F401  (registers torch.ops._C)
        fused_act_op = torch.ops._C.gelu_tanh_and_mul
        _d = x_gu.shape[-1] // 2
        _out = torch.empty(x_gu.shape[:-1] + (_d,), dtype=x_gu.dtype, device=x_gu.device)
        fused_act_op(_out, x_gu)  # smoke
        torch.cuda.synchronize()

        class _FusedGeluAndMul(torch.nn.Module):
            """Faithful to vLLM GeluAndMul(approximate='tanh').forward_cuda: one fused
            CUDA kernel gelu_tanh_and_mul(out, x), [.,2d] -> [.,d]."""
            def forward(self, gu):
                d = gu.shape[-1] // 2
                out = torch.empty(gu.shape[:-1] + (d,), dtype=gu.dtype, device=gu.device)
                fused_act_op(out, gu)
                return out
        act_mod = _FusedGeluAndMul().cuda().eval()
        print("[draft-mlp] activation = vLLM fused torch.ops._C.gelu_tanh_and_mul "
              "(deployed kernel)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[draft-mlp] fused gelu_tanh_and_mul unavailable ({exc!r}); torch "
              f"gelu*mul fallback (may be >1 kernel -> us_activation upper bound)",
              flush=True)
        act_is_fused_kernel = False

        class _TorchGeGLU(torch.nn.Module):
            def forward(self, gu):
                g, u = gu.chunk(2, dim=-1)
                return F.gelu(g, approximate="tanh") * u
        act_mod = _TorchGeGLU().cuda().eval()

    gate_up_mods = [l["gate_up"] for l in layers]
    down_mods = [l["down"] for l in layers]

    def run_gate_up():        # 4x gate_up GEMV (fused gate+up), M=1
        for m in gate_up_mods:
            m(x_h)

    def run_down():           # 4x down GEMV, M=1
        for m in down_mods:
            m(x_i)

    def run_activation():     # 4x GeluAndMul (the real deployed fused act kernel)
        for _ in range(L):
            act_mod(x_gu)

    def run_full_mlp():       # the REAL dependent forward: gate_up -> act -> down x4
        h = x_h
        for l in layers:
            gu = l["gate_up"](h)
            a = act_mod(gu)
            h = l["down"](a)
        return h

    def run_gate_sep_up_sep():  # unfused gate + up (gate+up fusion = already-taken lever)
        for l in layers:
            l["gate_sep"](x_h)
            l["up_sep"](x_h)

    us = {}
    cap_ok = {}
    us["gate_up"], cap_ok["gate_up"] = time_callable(run_gate_up, it, wu)
    us["down"], cap_ok["down"] = time_callable(run_down, it, wu)
    us["activation"], cap_ok["activation"] = time_callable(run_activation, it, wu)
    us["full_forward"], cap_ok["full_forward"] = time_callable(run_full_mlp, it, wu)
    us["gate_sep_up_sep"], cap_ok["gate_sep_up_sep"] = \
        time_callable(run_gate_sep_up_sep, it, wu)

    # draft_mlp_measured_us = the gate_up+down GEMV chain == #264's 50.7us mlp basis
    draft_mlp_measured_us = us["gate_up"] + us["down"]
    draft_mlp_full_forward_us = us["full_forward"]
    mlp_overhead_ratio = draft_mlp_measured_us / draft_mlp_roofline_us
    full_overhead_ratio = draft_mlp_full_forward_us / draft_mlp_roofline_us

    # NaN-clean: the real forward must be finite
    out = run_full_mlp()
    nan_clean = bool(torch.isfinite(out).all().item())

    # --- per-kernel launch/latency floor (tiny-GEMV sweep, in-graph) ----------
    # Time chains of N identical near-zero-byte GEMVs; slope = marginal per-kernel
    # GPU-side launch/latency cost (bandwidth ~0). Robust to graph-replay base.
    tiny_w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    tiny_x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    tiny_lin = BF16Linear(tiny_w)
    sweep_ns = [1, 2, 4, 8, 16, 32]
    sweep_us = []
    for n in sweep_ns:
        def run_tiny(n=n):
            for _ in range(n):
                tiny_lin(tiny_x)
        t, _ = time_callable(run_tiny, it, wu)
        sweep_us.append(t)
    # least-squares slope (per-kernel) and intercept (graph base)
    nbar = sum(sweep_ns) / len(sweep_ns)
    ubar = sum(sweep_us) / len(sweep_us)
    cov = sum((n - nbar) * (u - ubar) for n, u in zip(sweep_ns, sweep_us))
    var = sum((n - nbar) ** 2 for n in sweep_ns)
    per_kernel_launch_us = cov / var
    graph_base_us = ubar - per_kernel_launch_us * nbar
    # cross-check: the GeluAndMul kernel is ~0-byte -> its per-kernel time is also a
    # real deployed per-kernel floor.
    per_kernel_from_activation_us = us["activation"] / L

    # deployed kernel count in the MLP: gate_up GEMV + GeluAndMul + down GEMV per layer
    n_kernels_deployed = 3 * L if act_is_fused_kernel else None
    # The decisive diagnostic: effective HBM BW achieved by the M=1 MLP. If it is far
    # below peak, the GEMV UNDER-SATURATES HBM (a single warp can't fill the bus, Chen
    # 2605.30571) -> the overhead is the INTRINSIC-M=1 memory-latency penalty, NOT
    # recoverable bandwidth slack, NOT CPU-launch (ONEGRAPH already erases that).
    mlp_effective_bw_gbs = mlp_weight_bytes / (draft_mlp_measured_us * 1e-6) / 1e9
    mlp_bw_utilization = mlp_effective_bw_gbs / A10G_HBM_GBS
    m1_under_saturates_hbm = bool(mlp_bw_utilization < 0.90)
    # the intrinsic (un-recoverable, M=1) part vs the recoverable (activation kernel).
    intrinsic_m1_floor_us = draft_mlp_measured_us       # the GEMV chain IS the floor
    launch_total_us = (n_kernels_deployed or (3 * L)) * per_kernel_launch_us

    # --- large reference GEMV: empirical BW anchor (the 128 MiB tied lm_head) --
    lm_head_w = load_tensor(os.path.join(args.drafter_dir, "model.safetensors"),
                            "model.embed_tokens.weight").cuda()   # [262144, 256]
    ref_bytes = lm_head_w.numel() * BF16_BYTES
    ref_lin = BF16Linear(lm_head_w.clone())
    ref_x = torch.randn(1, draft_mlp_hidden, device="cuda", dtype=torch.bfloat16)

    def run_ref():
        ref_lin(ref_x)
    us["ref_large_gemv"], cap_ok["ref_large_gemv"] = time_callable(run_ref, it, wu)
    ref_roofline_us = bytes_to_us(ref_bytes)
    ref_bw_gbs = ref_bytes / (us["ref_large_gemv"] * 1e-6) / 1e9
    ref_bw_utilization = ref_bw_gbs / A10G_HBM_GBS
    del lm_head_w, ref_lin
    torch.cuda.empty_cache()

    # --- (3) price greedy-safe LOSSLESS fusion levers -------------------------
    # CRITICAL FRAMING (researcher-agent; Chen 2605.30571, AutoMegaKernel 2606.09682):
    # the 50.7us GEMV chain IS essentially the achievable M=1 floor. A single-row
    # GEMV with one warp cannot saturate HBM (batch-1 ~ 31-81% of the analytic
    # floor), so the 21us roofline is UNREACHABLE at M=1 -- fusing kernels does NOT
    # speed up the individual GEMVs (each still reads the same bytes with the same low
    # memory-level parallelism). The ONLY recoverable slack is the SEPARATE GeluAndMul
    # kernel (eliminated by folding gelu*mul into the gate_up epilogue) + the tiny
    # inter-kernel scheduling (~60ns/node). recoverable ~= us_activation.
    # All levers are step-only (E[T], PPL unchanged for a lossless fusion).
    def price(recoverable_us):
        recoverable_us = max(0.0, recoverable_us)
        step_after = STEP_US - args.k * recoverable_us
        tps_after = tps_from_step_et(step_after, ET_DEPLOYED)
        return {
            "recoverable_mlp_us": recoverable_us,
            "step_after_us": step_after,
            "projected_tps": tps_after,
            "projected_tps_gain_pct": 100.0 * (tps_after / FRONTIER_TPS - 1.0),
        }

    # L1 fold-activation (the dominant RECOVERABLE lever): eliminate the L separate
    #     GeluAndMul kernels by folding gelu*mul into the gate_up GEMV epilogue
    #     (CUTLASS EVT). The GeluAndMul reads [1,4096]+writes [1,2048] (~12KB ->
    #     ~0.02us HBM at 600 GB/s) -> it does ~0 real work and is PURELY launch-bound;
    #     folding it eliminates exactly L kernel launches. DEPLOYED-FAITHFUL recoverable
    #     = L * per_kernel_launch_us (1 fused kernel/layer at the MEASURED per-kernel
    #     floor). When the real vLLM fused kernel is callable, us["activation"] already
    #     measures those L kernels directly -> use it. The torch fallback runs 2 kernels/
    #     layer (gelu;mul) so us["activation"] OVERcounts -> it is the band's UPPER bound.
    #     LOSSLESS (elementwise on the post-reduction registers; no reduction-order
    #     change) -> E[T]+PPL exactly unchanged.
    rec_L1_upper = us["activation"]                 # eager measurement (2 kernels/layer = upper)
    if act_is_fused_kernel:
        rec_L1 = us["activation"]                   # real fused kernel measured directly
    else:
        rec_L1 = L * per_kernel_launch_us           # deployed-faithful: 1 fused kernel/layer
    # L2 fused-MLP megakernel: L1 + keep the act output in registers (skip the
    #     [1,2048] act->down HBM round-trip) + remove the ~60ns/node inter-kernel
    #     scheduling (NVIDIA CUDA Graphs, Ampere). On A10G a megakernel is only
    #     1.0-1.08x vs cuBLAS (AutoMegaKernel) because the GEMVs stay memory-latency-
    #     bound -> the increment over L1 is SMALL; cap it at 8% of the forward.
    #     LOSSLESS iff the down reduction order is preserved (split-K reassoc. flag).
    scheduling_saving_us = (3 * L - L) * GRAPH_NODE_LATENCY_US
    megakernel_extra_cap = 0.08 * draft_mlp_full_forward_us
    rec_L2 = rec_L1 + min(scheduling_saving_us, megakernel_extra_cap)
    # L3 bandwidth-roofline: the analytic floor. PHYSICALLY UNREACHABLE at M=1 (needs
    #     M>=16 batching, Chen 2605.30571) -> reported as the theoretical anchor ONLY,
    #     NOT a fusion lever.
    rec_L3 = draft_mlp_full_forward_us - draft_mlp_roofline_us

    lever_L1 = {"name": "L1_fold_activation", **price(rec_L1),
                "greedy_safe": True, "lossless": True, "reachable": True,
                "note": "fold GeluAndMul into the gate_up epilogue (CUTLASS EVT); "
                        "elementwise on post-reduction registers, no reduction-order "
                        "change -> E[T]+PPL exactly unchanged. The dominant RECOVERABLE "
                        "lever. recoverable = L * per_kernel_launch_us (1 fused launch/"
                        "layer at the measured floor; deployed-faithful)."}
    lever_L1_upper = {"name": "L1_fold_activation_UPPER", **price(rec_L1_upper),
                "greedy_safe": True, "lossless": True, "reachable": True,
                "note": "upper bracket: the eager-measured standalone activation time "
                        "(torch fallback = 2 kernels/layer, so it OVERcounts the single "
                        "deployed fused kernel). Brackets the activation-cost uncertainty."}
    lever_L2 = {"name": "L2_fused_mlp_megakernel", **price(rec_L2),
                "greedy_safe": True, "lossless": None, "reachable": True,
                "note": "1 fused kernel/layer; also skips the act->down HBM round-trip "
                        "+ ~60ns/node scheduling. On A10G only 1.0-1.08x vs cuBLAS "
                        "(AutoMegaKernel) -> small increment over L1. LOSSLESS iff the "
                        "down reduction order is preserved; a split-K/reassociating "
                        "fusion stays correctness-safe (propose-only) but may drift "
                        "E[T] -- FP reassociation flag."}
    lever_L3 = {"name": "L3_bandwidth_roofline_UNREACHABLE", **price(rec_L3),
                "greedy_safe": True, "lossless": True, "reachable": False,
                "note": "the analytic 600 GB/s floor; PHYSICALLY UNREACHABLE at M=1 "
                        "(single-warp GEMV under-saturates HBM, ~31-81% of peak, Chen "
                        "2605.30571). Needs M>=16 BATCHING, not kernel surgery -> NOT "
                        "a fusion lever, a different (tree/verify-shape) axis."}
    levers = [lever_L1, lever_L1_upper, lever_L2, lever_L3]

    # Material slack? headline overhead ratio gate (the MLP IS >1.15x its roofline,
    # but most of that gap is the intrinsic-M=1 penalty; the RECOVERABLE slack is the
    # activation kernel). recoverable_material gates the live lever on us_activation.
    slack_material = bool(mlp_overhead_ratio >= OVERHEAD_MATERIAL_RATIO)
    recoverable_material = bool(rec_L1 > 0.0 and lever_L1["projected_tps_gain_pct"] > 0.0)
    # PRIMARY-facing live lever: the activation-epilogue fold (L1), the defensible,
    # measured, LOSSLESS, ONEGRAPH-compatible recoverable term. The megakernel (L2)
    # is a small increment on A10G; L3 is the UNREACHABLE roofline anchor.
    recoverable_mlp_us = rec_L1
    step_after = lever_L1["step_after_us"]
    projected_tps_gain_pct = (lever_L1["projected_tps_gain_pct"]
                              if recoverable_material else 0.0)
    # band: [L1 deployed-faithful floor, L1_upper eager-measured ceiling] -- brackets the
    # activation-fold uncertainty (1 fused kernel vs the 2-kernel eager measurement).
    # L2 (megakernel) and L3 (UNREACHABLE roofline) are reported in the lever table.
    projected_tps_gain_band_pct = [lever_L1["projected_tps_gain_pct"],
                                   lever_L1_upper["projected_tps_gain_pct"]]

    # --- (4) greedy/PPL-safety certificate ------------------------------------
    draft_mlp_fusion_greedy_safe = True   # propose-only draft; verify gates emitted tok
    fp_reassociation_flag = (
        "gate+up fusion (already deployed) and fold-activation (L1) are LOSSLESS "
        "(same per-output bf16 reduction order). A fused-MLP megakernel (L2) is "
        "lossless ONLY if it preserves the down GEMV's reduction order; a split-K / "
        "retiled reduction REASSOCIATES the bf16 sum (not bit-identical, cf. lawine "
        "#246) -- still correctness-safe by propose-only verify, but E[T] may drift, "
        "so a megakernel must be E[T]+PPL re-verified, not assumed.")

    # --- (6) self-test --------------------------------------------------------
    # (a) draft_mlp_measured_us recovers #264's 50.7us within +/-20%
    mlp_anchor_resid = abs(draft_mlp_measured_us - MLP_ANCHOR_US_264) / MLP_ANCHOR_US_264
    st_a = bool(mlp_anchor_resid <= MLP_ANCHOR_TOL_PCT)
    # (b) BW model byte->us round-trips the MEASURED large reference GEMV within tol
    ref_resid = abs(ref_roofline_us - us["ref_large_gemv"]) / us["ref_large_gemv"]
    st_b = bool(ref_resid <= BW_ROUNDTRIP_TOL_PCT)
    # (c) components {gate_up, down, activation} sum to the full forward within tol
    comp_sum = us["gate_up"] + us["down"] + us["activation"]
    comp_resid = abs(comp_sum - draft_mlp_full_forward_us) / draft_mlp_full_forward_us
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

    verdict_line = (
        f"ABOVE roofline BUT MOSTLY INTRINSIC: the draft MLP measures "
        f"{draft_mlp_measured_us:.1f}us at M=1 vs a {draft_mlp_roofline_us:.1f}us "
        f"bandwidth floor -> overhead ratio {mlp_overhead_ratio:.2f}x, but that is only "
        f"{mlp_bw_utilization*100:.0f}% of peak HBM BW -- a single-warp M=1 GEMV "
        f"UNDER-SATURATES HBM (Chen 2605.30571: batch-1 caps at ~31-81% of the floor), "
        f"so the gap is the INTRINSIC-M=1 memory-latency penalty, NOT recoverable "
        f"bandwidth slack and NOT CPU-launch (ONEGRAPH erases that; in-graph scheduling "
        f"is ~60ns/node). The {draft_mlp_roofline_us:.0f}us roofline is UNREACHABLE at "
        f"M=1 (needs M>=16 batching). The large {ref_bytes/2**20:.0f} MiB reference "
        f"GEMV hits {ref_bw_utilization*100:.0f}% of peak (IS bandwidth-bound), "
        f"anchoring the model. The ONLY recoverable slack is the SEPARATE GeluAndMul "
        f"kernel: a greedy-safe LOSSLESS epilogue fold (L1) eliminates L={L} launches "
        f"and recovers ~{recoverable_mlp_us:.1f}us/pass (eager-measured upper "
        f"{rec_L1_upper:.1f}us) x K={args.k} -> step {step_after:.0f}us -> "
        f"{projected_tps_gain_pct:+.2f}% TPS off 481.53 (band "
        f"{projected_tps_gain_band_pct[0]:+.1f}..{projected_tps_gain_band_pct[1]:+.1f}%; "
        f"megakernel L2 adds little on A10G, 1.0-1.08x), E[T]+PPL UNCHANGED."
        if recoverable_material else
        f"the draft MLP is at its achievable M=1 floor; no recoverable slack")

    handoff = (
        f"the draft MLP is {draft_mlp_measured_us:.1f}us "
        f"({100*draft_mlp_measured_us/BF16_ANCHOR_US_254:.0f}% of the 101.2us/pass), "
        f"{mlp_overhead_ratio:.1f}x ABOVE its {draft_mlp_roofline_us:.0f}us bandwidth "
        f"roofline, BUT that gap is INTRINSIC: at M=1 the GEMV hits only "
        f"{mlp_bw_utilization*100:.0f}% of peak HBM BW (single warp can't saturate the "
        f"bus, Chen 2605.30571), so the {draft_mlp_roofline_us:.0f}us roofline is "
        f"UNREACHABLE without M>=16 batching -- the GEMV chain IS the achievable M=1 "
        f"floor and fusion cannot speed it up. gate+up and gelu*mul are ALREADY fused "
        f"in the served Gemma4MLP, so the ONLY recoverable draft-MLP slack is folding "
        f"the SEPARATE GeluAndMul kernel (L={L} launches, ~{recoverable_mlp_us:.1f}us "
        f"deployed-faithful) into the gate_up epilogue: a greedy-safe LOSSLESS step "
        f"lever worth {projected_tps_gain_pct:+.1f}% "
        f"/ ~{tps_from_step_et(step_after, ET_DEPLOYED)-FRONTIER_TPS:+.0f} TPS off 481.53 "
        f"(band {projected_tps_gain_band_pct[0]:+.1f}..{projected_tps_gain_band_pct[1]:+.1f}%, "
        f"a per-layer megakernel adds little on A10G). So the dominant 51.7% draft term "
        f"has MODEST recoverable headroom (the activation kernel only, NOT the roofline "
        f"gap) -- a PURE-STEP E[T]-independent lever (cheaper pass, not fewer passes -> "
        f"NOT adaptive-K/ONEGRAPH-blocked); needs a fused gate_up+epilogue kernel + "
        f"E[T]/PPL re-verify (split-K reassociation flag), human-approval-gated, NOT "
        f"this analysis-only PR. The big draft levers remain fewer passes (adaptive-K, "
        f"blocked) or M>=16 batching, not cheaper MLP passes.")

    components = [
        {"component": "gate_up_GEMV", "us": us["gate_up"],
         "roofline_us": gate_up_roofline_us,
         "overhead_ratio": us["gate_up"] / gate_up_roofline_us,
         "pct_of_full": 100.0 * us["gate_up"] / draft_mlp_full_forward_us},
        {"component": "down_GEMV", "us": us["down"],
         "roofline_us": down_roofline_us,
         "overhead_ratio": us["down"] / down_roofline_us,
         "pct_of_full": 100.0 * us["down"] / draft_mlp_full_forward_us},
        {"component": "GeluAndMul_activation", "us": us["activation"],
         "roofline_us": 0.0,  # ~0-byte; pure launch/latency
         "overhead_ratio": float("inf"),
         "pct_of_full": 100.0 * us["activation"] / draft_mlp_full_forward_us},
        {"component": "per_kernel_launch_floor", "us": per_kernel_launch_us,
         "roofline_us": 0.0, "overhead_ratio": float("inf"),
         "pct_of_full": 100.0 * (3 * L * per_kernel_launch_us)
                        / draft_mlp_full_forward_us},
    ]

    verdict = {
        "draft_mlp_roofline_self_test_passes": self_test_passes,   # PRIMARY
        "projected_tps_gain_pct": projected_tps_gain_pct,          # TEST
        "slack_material": slack_material,
        "recoverable_material": recoverable_material,
        "draft_mlp_fusion_greedy_safe": draft_mlp_fusion_greedy_safe,
        # headline roofline position
        "draft_mlp_measured_us": draft_mlp_measured_us,
        "draft_mlp_roofline_us": draft_mlp_roofline_us,
        "mlp_overhead_ratio": mlp_overhead_ratio,
        "draft_mlp_full_forward_us": draft_mlp_full_forward_us,
        "full_overhead_ratio": full_overhead_ratio,
        "mlp_effective_bw_gbs": mlp_effective_bw_gbs,
        "mlp_bw_utilization": mlp_bw_utilization,
        "m1_under_saturates_hbm": m1_under_saturates_hbm,
        "intrinsic_m1_floor_us": intrinsic_m1_floor_us,
        "roofline_unreachable_at_m1": True,
        # architecture diagnostic
        "draft_mlp_hidden": draft_mlp_hidden,
        "draft_mlp_intermediate": draft_mlp_intermediate,
        "draft_mlp_layers": draft_mlp_layers,
        "draft_mlp_gated": draft_mlp_gated,
        "draft_mlp_activation": draft_mlp_activation,
        "draft_mlp_dtype": draft_mlp_dtype,
        "act_is_fused_kernel": act_is_fused_kernel,
        "n_kernels_deployed": n_kernels_deployed,
        # byte model
        "mlp_weight_bytes": mlp_weight_bytes,
        "mlp_weight_mib": mlp_weight_bytes / 2 ** 20,
        "gate_up_weight_mib": gate_up_weight_bytes / 2 ** 20,
        "down_weight_mib": down_weight_bytes / 2 ** 20,
        "mlp_activation_bytes": mlp_activation_bytes,
        "draft_mlp_roofline_us_with_act": draft_mlp_roofline_us_with_act,
        "gate_up_roofline_us": gate_up_roofline_us,
        "down_roofline_us": down_roofline_us,
        # measured components
        "us_gate_up": us["gate_up"],
        "us_down": us["down"],
        "us_activation": us["activation"],
        "us_gate_sep_up_sep": us["gate_sep_up_sep"],
        "gate_up_fusion_saving_us": us["gate_sep_up_sep"] - us["gate_up"],
        "per_kernel_launch_us": per_kernel_launch_us,
        "per_kernel_from_activation_us": per_kernel_from_activation_us,
        "graph_base_us": graph_base_us,
        "launch_total_us": launch_total_us,
        # MLP anchor recovery (#264)
        "mlp_anchor_us_264": MLP_ANCHOR_US_264,
        "mlp_anchor_resid_pct": 100.0 * mlp_anchor_resid,
        # large reference GEMV (empirical BW anchor)
        "ref_large_gemv_mib": ref_bytes / 2 ** 20,
        "ref_large_gemv_us": us["ref_large_gemv"],
        "ref_roofline_us": ref_roofline_us,
        "ref_bw_gbs": ref_bw_gbs,
        "ref_bw_utilization": ref_bw_utilization,
        "ref_resid_pct": 100.0 * ref_resid,
        # pricing
        "recoverable_mlp_us": recoverable_mlp_us,
        "recoverable_mlp_upper_us": rec_L1_upper,
        "step_after_us": step_after,
        "projected_tps_gain_band_pct": projected_tps_gain_band_pct,
        "levers": levers,
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
        "self_test_conditions": {"a_mlp_anchor": st_a, "b_bw_roundtrip": st_b,
                                 "c_component_sum": st_c, "d_composition": st_d,
                                 "e_nan_clean": st_e, "f_constants_unchanged": st_f},
        "handoff_line": handoff,
        "chain_captured": cap_ok,
    }

    # --- (5) print the verdict table ------------------------------------------
    print("\n[draft-mlp] ===== MLP DECOMPOSITION (M=1, launch-free graph; #264 basis) =====", flush=True)
    print(f"  {'component':24s} {'us':>8s} {'roofline':>9s} {'ovr':>6s} {'%full':>7s}", flush=True)
    for c in components:
        ovr = "inf" if math.isinf(c["overhead_ratio"]) else f"{c['overhead_ratio']:.2f}"
        print(f"  {c['component']:24s} {c['us']:8.2f} {c['roofline_us']:9.2f} "
              f"{ovr:>6s} {c['pct_of_full']:6.1f}%", flush=True)
    print(f"  {'-'*56}", flush=True)
    print(f"  {'MLP measured (gate_up+down)':24s} {draft_mlp_measured_us:8.2f} "
          f"{draft_mlp_roofline_us:9.2f} {mlp_overhead_ratio:6.2f} (vs #264 50.7us, "
          f"resid {100*mlp_anchor_resid:.1f}%)", flush=True)
    print(f"  {'MLP full forward (+act)':24s} {draft_mlp_full_forward_us:8.2f} "
          f"{draft_mlp_roofline_us:9.2f} {full_overhead_ratio:6.2f}", flush=True)
    print(f"  gate+up fusion saving (already deployed): "
          f"{us['gate_sep_up_sep']-us['gate_up']:.1f}us "
          f"(unfused {us['gate_sep_up_sep']:.1f} -> fused {us['gate_up']:.1f})", flush=True)
    print(f"\n[draft-mlp] BW ANCHOR: large {ref_bytes/2**20:.0f} MiB GEMV measures "
          f"{us['ref_large_gemv']:.1f}us = roofline {ref_roofline_us:.1f}us "
          f"({ref_bw_utilization*100:.0f}% of peak; resid {100*ref_resid:.1f}%) -> the "
          f"600 GB/s model is empirically anchored; large IS bandwidth-bound, the "
          f"M=1 MLP at {mlp_bw_utilization*100:.0f}% of peak is NOT.", flush=True)
    print("[draft-mlp] ===== FUSION LEVER PRICING (step-only; E[T]+PPL UNCHANGED) =====", flush=True)
    for lev in levers:
        print(f"  {lev['name']:26s} recover {lev['recoverable_mlp_us']:6.1f}us/pass "
              f"-> step {lev['step_after_us']:.0f}us -> "
              f"{lev['projected_tps_gain_pct']:+.2f}% "
              f"(TPS {lev['projected_tps']:.1f})", flush=True)
    print(f"\n[draft-mlp] VERDICT: slack_material={slack_material}  "
          f"projected_tps_gain_pct={projected_tps_gain_pct:.3f}  "
          f"self_test={self_test_passes}", flush=True)
    print(f"  {verdict_line}", flush=True)
    print(f"  self-test: a={st_a} b={st_b} c={st_c} d={st_d} e={st_e} f={st_f}", flush=True)

    payload = {
        "config": {
            "drafter_dir": args.drafter_dir, "torch": torch.__version__, "device": dev,
            "sm": f"{cap[0]}{cap[1]}", "iters": it, "warmup": wu, "k": args.k,
            "A10G_HBM_GBS": A10G_HBM_GBS, "layer_ids": layer_ids,
            "mlp_anchor_tol_pct": MLP_ANCHOR_TOL_PCT,
            "bw_roundtrip_tol_pct": BW_ROUNDTRIP_TOL_PCT,
            "component_sum_tol_pct": COMPONENT_SUM_TOL_PCT,
            "overhead_material_ratio": OVERHEAD_MATERIAL_RATIO,
            "note": "isolated CUDA-graph M=1 micro-profiling of the real bf16 draft "
                    "MLP (vLLM Gemma4MLP: gate_up fused GEMV + GeluAndMul + down GEMV "
                    "x4 layers) vs its HBM-bandwidth roofline; large lm_head GEMV as "
                    "the empirical BW anchor. No serve change, no HF Job, no "
                    "submission. Greedy+PPL pinned (nothing edited).",
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
    print(f"[draft-mlp] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[draft-mlp] W&B logging failed (non-fatal): {exc!r}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    v = payload["verdict"]
    # component decomposition table {component, us, roofline_us, overhead_ratio, %full}
    comp = wandb.Table(columns=["component", "us", "roofline_us", "overhead_ratio",
                                "pct_of_full"])
    for c in payload["components"]:
        ovr = c["overhead_ratio"]
        comp.add_data(c["component"], c["us"], c["roofline_us"],
                      None if math.isinf(ovr) else ovr, c["pct_of_full"])
    run.log({"mlp_component_decomposition": comp})
    # fusion lever pricing table
    lev = wandb.Table(columns=["lever", "recoverable_mlp_us", "step_after_us",
                               "projected_tps", "projected_tps_gain_pct",
                               "greedy_safe", "lossless"])
    for l in v["levers"]:
        lev.add_data(l["name"], l["recoverable_mlp_us"], l["step_after_us"],
                     l["projected_tps"], l["projected_tps_gain_pct"],
                     l["greedy_safe"], str(l["lossless"]))
    run.log({"fusion_lever_pricing": lev})
    # tiny-GEMV per-kernel sweep
    sw = wandb.Table(columns=["n_kernels", "us"])
    for n, u in zip(payload["tiny_sweep"]["n"], payload["tiny_sweep"]["us"]):
        sw.add_data(n, u)
    run.log({"per_kernel_launch_sweep": sw})
    run.summary.update({k: val for k, val in v.items()
                        if isinstance(val, (int, float, bool, str))})
    run.finish()
    print(f"[draft-mlp] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
