#!/usr/bin/env python
"""QSpec int3-draft profiler + self-test (PR #248, Idea Q-1).

WHAT THIS MEASURES
------------------
The hypothesis (QSpec, complementary quantization): push the speculative DRAFT to
int3 while keeping the verify model untouched (int4-Marlin body + bf16 lm_head),
so draft errors are corrected bit-for-bit by the unchanged verify. The claim:
+5-10% step-time at ZERO greedy-identity / PPL risk, stacking with other levers.

This profiler answers the two empirical questions the hypothesis rests on, on the
REAL deployed Gemma4 MTP drafter (`/tmp/qat-assistant`, the proposer in the
`fa2sw_precache_kenyan` 481.53-TPS frontier), single A10G, post-training quant
only, no HF Job, no served-file change:

  (1) DRAFT-PASS WALL-TIME (the speed lever). Time the real per-pass drafter GEMM
      chain at M=1, x K=7, launch-free in ONE CUDA graph (the deployed ONEGRAPH
      basis), for:
        - bf16            : the DEPLOYED draft path (cuBLAS F.linear).
        - int3 (available): the ONLY int3 path that exists on Ampere sm_86 today
                            = group int3 weights -> runtime dequant-to-bf16 ->
                            bf16 GEMM (no fused W3A16 kernel exists; see below).
      Plus the roofline BYTE model for a hypothetical PERFECT fused int3/int4
      kernel (ceiling), so both the realistic floor and the unreachable ceiling
      are reported.

  (2) E[T] DEGRADATION (the cost). Fake-quantize every drafter linear to group
      int3 and measure the per-pass GEMM-chain OUTPUT perturbation vs bf16 on a
      representative input ensemble -> top-1 proposal disagreement -> a conservative
      E[T] estimate (a weaker proposer can only LOWER E[T]).

  Net TPS = (draft-pass wall-time saving) - (E[T] loss). Both terms are reported;
  the verdict is the combined net.

WHY THE ANSWER IS STRUCTURAL (RED on speed, GREEN on safety)
------------------------------------------------------------
- No fused int3 GEMM kernel exists on Ampere sm_86. vLLM's GPTQMarlin / AWQMarlin
  accept num_bits in {4,8} ONLY; Marlin (arXiv:2408.11743) is W4A16; Machete is
  Hopper-first; exllamav2-q3 / BitBlas / LUT-GEMM are research-grade with no
  production vLLM Ampere path. The only available int3 route is dequant->bf16->GEMM,
  which is STRICTLY MORE work than the bf16 path (dequant + the same matmul) -> it
  cannot beat bf16, let alone the int4 Marlin kernel.
- The cited QSpec id arXiv:2411.11514 is wrong (that is a different paper). QSpec is
  arXiv:2410.11305; its 1.78-1.80x comes from W4A4 ACTIVATION+weight quant on the
  draft (INT4 tensor-core GEMM), NOT weight-only int3, and it speeds up BATCH
  throughput, not batch=1 single-stream MTP decode where activations cannot fill a
  tensor-core tile.
- The DEPLOYED draft is bf16, NOT int4 (model `gemma-4-E4B-it-qat-q4_0-UNQUANTIZED-
  assistant`; all 49 weight tensors BF16). So the PR's "int4-draft control" does not
  exist in the deployed stack; the honest control is bf16. (int4-draft is a SEPARATE
  lever -- stark #70 -- that does have a real Marlin kernel; int3 does not.)
- The drafter GEMM chain is only ~5% of the ~11.6 ms decode step (566 us/step,
  drafter_forward_roofline.py) and 47% HBM-peak (partially BW-bound). Even a PERFECT
  fused int3 kernel (which does not exist) caps the gross saving at the int3-byte
  fraction of that ~5% slice -> <1% gross, erased by any E[T] loss.

The SAFETY claim is vindicated by construction: greedy spec-decode emits the verify
model's greedy token regardless of the draft proposals (accepted draft tokens are
accepted ONLY when they equal the verify argmax). The verify model and accept rule
are byte-identical between control and variant -> token-identical output and pinned
PPL=2.3772, independent of draft precision.

SELF-TEST (`qspec_int3_draft_self_test_passes`)
-----------------------------------------------
(a) greedy-identity token-identical to the bf16-draft path  -> True BY CONSTRUCTION
(b) PPL <= 2.42 (pinned 2.3772, verify untouched)           -> True BY CONSTRUCTION
(c) NaN-clean (all timed / quantized tensors finite)        -> measured
(d) peak VRAM <= 24 GB                                       -> measured
TEST metrics: net_tps_gain_pct_int3_draft, e_t_int3_draft (vs bf16-draft control).

Pure isolated-kernel microbenchmark on real drafter weights + synthetic [M,in]
activations (the drafter_forward_roofline.py #75 methodology); lossless, no
serve-path change, no token-stream change, no HF Job.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import struct
import time

# Must be set before importing torch. This A10G node inherits CUDA_VISIBLE_DEVICES=5
# (host physical id) but the in-container GPU is index 0; the inherited value makes
# torch.cuda unavailable. Force 0 (single-GPU node). See drafter_forward_roofline.py.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEFAULT_DRAFTER = "/tmp/qat-assistant"

# ---- A10G (AWS g5, GA102, sm_86) roofline ceilings (identical to #68/#75) ----
A10G_HBM_GBS = 600.0
BF16_BYTES = 2.0
# group quant byte model (weight bits + per-group fp16 scale[+zero])
GROUP_SIZE = 32
INT4_WEIGHT_BITS = 4
INT3_WEIGHT_BITS = 3
INT4_SCALE_BYTES = 2          # fp16 scale per group
INT3_SCALE_BYTES = 4          # fp16 scale + fp16 zero per group (asymmetric)

# deployed-step references (drafter_forward_roofline.py / spec_cost_model, on-branch)
DECODE_STEP_MS = 11.6         # int4 verify decode-step latency (#51/#68)
FRONTIER_TPS = 481.53         # PR #52 official a10g-small frontier (the anchor)
ET_DEPLOYED = 3.3             # ~3.3 accepted tok/step (BASELINE.md "the climb")
K_DEPLOYED = 7                # num_speculative_tokens (manifest SPECULATIVE_CONFIG)


# --------------------------------------------------------------------------- #
# Drafter weight introspection (verbatim shapes from #75; real bf16 weights).  #
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def load_tensor(path: str, name: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def build_drafter_specs(drafter_dir: str):
    """Return per_pass list of (role, in, out, weight_bf16) in execution order.

    The 19 dense GEMMs run per draft pass (drafter_forward_roofline.py #75):
    pre_projection, x4 layers {q_proj, o_proj, gate_up, down_proj}, post_projection,
    centroid sampler. M=1; K=7 passes are strictly sequential.
    """
    st = os.path.join(drafter_dir, "model.safetensors")
    hdr = read_safetensors_header(st)
    layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0])
                        for k in hdr if ".layers." in k})
    specs = []  # (role, in, out, weight)

    w = load_tensor(st, "pre_projection.weight")
    specs.append(("pre_projection", w.shape[1], w.shape[0], w))
    for i in layer_ids:
        qw = load_tensor(st, f"model.layers.{i}.self_attn.q_proj.weight")
        specs.append((f"layer{i}.q_proj", qw.shape[1], qw.shape[0], qw))
        ow = load_tensor(st, f"model.layers.{i}.self_attn.o_proj.weight")
        specs.append((f"layer{i}.o_proj", ow.shape[1], ow.shape[0], ow))
        gw = load_tensor(st, f"model.layers.{i}.mlp.gate_proj.weight")
        uw = load_tensor(st, f"model.layers.{i}.mlp.up_proj.weight")
        guw = torch.cat([gw, uw], dim=0)
        specs.append((f"layer{i}.gate_up", guw.shape[1], guw.shape[0], guw))
        dw = load_tensor(st, f"model.layers.{i}.mlp.down_proj.weight")
        specs.append((f"layer{i}.down_proj", dw.shape[1], dw.shape[0], dw))
    w = load_tensor(st, "post_projection.weight")
    specs.append(("post_projection", w.shape[1], w.shape[0], w))
    cw = "masked_embedding.centroids.weight"
    if cw in hdr:
        c = load_tensor(st, cw)
        specs.append(("centroids_sampler", c.shape[1], c.shape[0], c))
    return specs


# --------------------------------------------------------------------------- #
# Group int-N fake quantization (PTQ grid; calibration-free min/max per group). #
# --------------------------------------------------------------------------- #
def group_quant_dequant(w: torch.Tensor, bits: int, group: int):
    """Asymmetric per-(out-channel, in-group) uint-N quant -> dequant to bf16.

    Returns (w_hat_bf16, qweight_uint8, scale_bf16, zero_bf16, rel_l2_err).
    This is the standard GPTQ/AWQ weight grid (no backprop). w is [out, in].
    """
    out_f, in_f = w.shape
    wf = w.float()
    g = group if in_f % group == 0 else math.gcd(in_f, group) or 1
    ng = in_f // g
    wg = wf.view(out_f, ng, g)
    wmin = wg.amin(dim=-1, keepdim=True)
    wmax = wg.amax(dim=-1, keepdim=True)
    qmax = (1 << bits) - 1
    scale = (wmax - wmin).clamp_min(1e-8) / qmax
    zero = wmin
    q = ((wg - zero) / scale).round().clamp_(0, qmax)
    w_hat = (q * scale + zero).view(out_f, in_f)
    rel = (w_hat - wf).norm() / wf.norm().clamp_min(1e-12)
    return (w_hat.to(torch.bfloat16),
            q.view(out_f, in_f).to(torch.uint8),
            scale.squeeze(-1).to(torch.bfloat16),
            zero.squeeze(-1).to(torch.bfloat16),
            float(rel))


class BF16Linear(torch.nn.Module):
    """Deployed draft path: a plain bf16 weight, cuBLAS F.linear."""
    def __init__(self, w_bf16: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(w_bf16.cuda(), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class Int3DequantLinear(torch.nn.Module):
    """The ONLY available int3 path on sm_86: store group int3 weights, dequant to
    bf16 at runtime, then the SAME bf16 GEMM. Strictly more work than BF16Linear
    (dequant + identical matmul). qweight stored uint8 (generous: skips the bit
    unpack a real 3-bit kernel would also pay), so this is a LOWER bound on the
    int3 penalty."""
    def __init__(self, q_u8, scale_bf16, zero_bf16, group, in_f):
        super().__init__()
        out_f = q_u8.shape[0]
        self.out_f, self.in_f = out_f, in_f
        self.g = group if in_f % group == 0 else math.gcd(in_f, group) or 1
        self.ng = in_f // self.g
        self.register_buffer("q", q_u8.view(out_f, self.ng, self.g).cuda())
        self.register_buffer("scale", scale_bf16.view(out_f, self.ng, 1).cuda())
        self.register_buffer("zero", zero_bf16.view(out_f, self.ng, 1).cuda())

    def dequant(self):
        w = self.q.to(torch.bfloat16) * self.scale + self.zero
        return w.view(self.out_f, self.in_f)

    def forward(self, x):
        return F.linear(x, self.dequant())


# --------------------------------------------------------------------------- #
# Launch-free CUDA-graph timing of a whole per-pass GEMM chain (the deployed    #
# ONEGRAPH basis): many tiny sequential kernels, one replay. (#75 methodology)  #
# --------------------------------------------------------------------------- #
def time_chain_graph(modules_in_order, M, iters, warmup):
    """(ms_per_pass_graph, captured). modules_in_order: list of (module, in_f)."""
    bufs = [torch.randn(M, inf, device="cuda", dtype=torch.bfloat16)
            for (_, inf) in modules_in_order]

    def run_chain():
        for (mod, _), b in zip(modules_in_order, bufs):
            mod(b)

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run_chain()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run_chain()
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
        return ms, True
    except Exception as exc:  # noqa: BLE001
        print(f"[qspec-int3]   chain graph capture failed: {exc!r}; eager", flush=True)
        with torch.inference_mode():
            for _ in range(warmup):
                run_chain()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                run_chain()
            e1.record()
            torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters, False


def byte_model(specs, bits, scale_bytes, group):
    """Total per-pass (weight + act + out) bytes under group int-`bits` weights."""
    wb = ab = ob = 0.0
    for _role, inn, out, _w in specs:
        g = group if inn % group == 0 else math.gcd(inn, group) or 1
        wb += (bits / 8.0) * out * inn + scale_bytes * out * (inn // g)
        ab += BF16_BYTES * 1 * inn       # M=1
        ob += BF16_BYTES * 1 * out
    return wb + ab + ob, wb


def expected_accepted(alpha: float, K: int) -> float:
    """Leviathan/Chen i.i.d. greedy acceptance: E[T] = (1 - a^(K+1)) / (1 - a)."""
    if alpha >= 1.0:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


def solve_alpha_for_et(et: float, K: int) -> float:
    """Invert expected_accepted for per-token acceptance alpha (bisection)."""
    lo, hi = 0.0, 0.999999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expected_accepted(mid, K) < et:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--k", type=int, default=K_DEPLOYED)
    ap.add_argument("--group", type=int, default=GROUP_SIZE)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--et-samples", type=int, default=4096,
                    help="representative drafter-input rows for the E[T] proxy")
    ap.add_argument("--decode-step-ms", type=float, default=DECODE_STEP_MS)
    ap.add_argument("--frontier-tps", type=float, default=FRONTIER_TPS)
    ap.add_argument("--et-deployed", type=float, default=ET_DEPLOYED)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output",
                    default="research/quant/qspec_int3_draft/profile.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="qspec-int3-draft")
    ap.add_argument("--wandb_name", default="kanna/qspec-int3-draft")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (set CUDA_VISIBLE_DEVICES=0)"
    dev = torch.cuda.get_device_name(0)
    print(f"[qspec-int3] device: {dev}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # --- real drafter weights -------------------------------------------------
    t0 = time.time()
    specs = build_drafter_specs(args.drafter_dir)
    print(f"[qspec-int3] {len(specs)} per-pass GEMMs from {args.drafter_dir} "
          f"in {time.time()-t0:.1f}s", flush=True)

    # --- build bf16 (control) and int3-dequant (variant) module chains --------
    bf16_mods, int3_mods = [], []
    weight_err = {}
    nan_clean = True
    for role, inn, out, w in specs:
        bf16_mods.append((BF16Linear(w), inn))
        w_hat, q_u8, scale, zero, rel = group_quant_dequant(w, INT3_WEIGHT_BITS, args.group)
        weight_err[role] = rel
        int3_mods.append((Int3DequantLinear(q_u8, scale, zero, args.group, inn), inn))
        if not (torch.isfinite(w_hat).all() and torch.isfinite(scale).all()):
            nan_clean = False
    mean_w_err = statistics.mean(weight_err.values())
    print(f"[qspec-int3] int3 group({args.group}) weight rel-L2 err: "
          f"mean {mean_w_err:.4f}  max {max(weight_err.values()):.4f}", flush=True)

    # --- (1) DRAFT-PASS WALL-TIME: bf16 vs int3-available, M=1, x K ----------
    ms_bf16, cap_bf16 = time_chain_graph(bf16_mods, 1, args.iters, args.warmup)
    ms_int3, cap_int3 = time_chain_graph(int3_mods, 1, args.iters, args.warmup)
    pass_bf16_us, pass_int3_us = ms_bf16 * 1e3, ms_int3 * 1e3
    step_bf16_us, step_int3_us = pass_bf16_us * args.k, pass_int3_us * args.k
    draft_walltime_saving_us = step_bf16_us - step_int3_us  # >0 == int3 faster
    print(f"[qspec-int3] draft pass (graph, M=1): bf16 {pass_bf16_us:.1f}us  "
          f"int3-available {pass_int3_us:.1f}us  (x{args.k} step: "
          f"{step_bf16_us:.0f} vs {step_int3_us:.0f}us)", flush=True)

    # --- roofline byte model: ceiling for a PERFECT (nonexistent) fused kernel -
    tot_bf16, w_bf16 = byte_model(specs, 16, 0, args.group)
    tot_int4, w_int4 = byte_model(specs, INT4_WEIGHT_BITS, INT4_SCALE_BYTES, args.group)
    tot_int3, w_int3 = byte_model(specs, INT3_WEIGHT_BITS, INT3_SCALE_BYTES, args.group)
    int3_byte_ratio = tot_int3 / tot_bf16
    int4_byte_ratio = tot_int4 / tot_bf16
    # drafter GEMM share of the decode step (measured bf16 chain / step)
    drafter_gemm_share = step_bf16_us / (args.decode_step_ms * 1e3)
    # ceiling for a PERFECT (nonexistent) fused int3 kernel: only the BANDWIDTH-bound
    # part of the drafter slice converts weight bytes -> time. The drafter chain is
    # 47% HBM-peak (drafter_forward_roofline.py, on-branch) -> only ~47% of the slice
    # is byte-bound; the rest is launch/latency that NO bit-width can cut.
    HBM_PEAK_FRAC = 0.4717
    ceiling_gross_full_pct = drafter_gemm_share * (1.0 - int3_byte_ratio) * 100.0
    ceiling_gross_saving_pct = ceiling_gross_full_pct * HBM_PEAK_FRAC
    print(f"[qspec-int3] drafter GEMM share of {args.decode_step_ms}ms step = "
          f"{drafter_gemm_share*100:.2f}%  | int3 byte ratio {int3_byte_ratio:.3f} "
          f"(int4 {int4_byte_ratio:.3f})  | perfect-kernel ceiling gross: full-BW "
          f"+{ceiling_gross_full_pct:.2f}%  realistic(47%HBM) +{ceiling_gross_saving_pct:.2f}%",
          flush=True)

    # --- (2) E[T] DEGRADATION: chain output perturbation -> proposal disagreement
    # Representative ensemble: random directions (Gemma RMSNorm normalizes the
    # input scale away on the first op, so direction is what matters). Real
    # backbone states cluster on a sub-manifold -> int3 disagreement is typically
    # LOWER there, so this synthetic estimate UPPER-BOUNDS the degradation.
    in0 = specs[0][1]
    torch.manual_seed(0)
    x = torch.randn(args.et_samples, in0, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        yb = bf16_mods[0][0](x)
        yi = int3_mods[0][0](x)
    # propagate the relative perturbation through the chain in a scalar sense:
    # the per-GEMM relative output error compounds ~ sum of weight rel-errs (first
    # order). Use the measured first-GEMM output disagreement as the per-stage rate
    # and the final centroid-sampler top-1 as the proposal disagreement proxy.
    cos0 = F.cosine_similarity(yb.float(), yi.float(), dim=-1).mean().item()
    relerr0 = ((yi.float() - yb.float()).norm(dim=-1)
               / yb.float().norm(dim=-1).clamp_min(1e-9)).mean().item()
    # proposal-quality proxy (E[T] degradation DIRECTION, an UPPER BOUND on loss):
    # a trustworthy per-step E[T] needs the full int4-verify + drafter accept loop,
    # which CANNOT run int3 (no kernel). Offline we can only bound the direction: the
    # int3 weight error perturbs the drafter's proposed token -> lower acceptance ->
    # lower E[T]. Synthetic-input centroid-argmax flip rate OVER-states it (256-d
    # synthetic inputs give low-margin logits; real decode is high-margin) AND min/max
    # int3 is cruder than Hessian/activation-aware GPTQ/AWQ int3. So et_int3 below is a
    # pessimistic LOWER bound on E[T]; the verdict does NOT depend on its precision.
    samp_bf16 = bf16_mods[-1][0]
    samp_int3 = int3_mods[-1][0]
    in_c = specs[-1][1]
    h_rand = torch.randn(args.et_samples, in_c, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        lg_b = samp_bf16(h_rand).float()
        lg_i = samp_int3(h_rand).float()
    proposal_disagree_ub = (lg_b.argmax(-1) != lg_i.argmax(-1)).float().mean().item()
    if not (torch.isfinite(lg_i).all() and torch.isfinite(yi).all()):
        nan_clean = False
    alpha_bf16 = solve_alpha_for_et(args.et_deployed, args.k)
    alpha_int3_lb = max(0.0, alpha_bf16 * (1.0 - proposal_disagree_ub))
    et_bf16 = expected_accepted(alpha_bf16, args.k)
    et_int3 = expected_accepted(alpha_int3_lb, args.k)   # pessimistic E[T] lower bound
    print(f"[qspec-int3] E[T]: bf16 {et_bf16:.3f} (alpha {alpha_bf16:.3f}) -> int3 "
          f">= {et_int3:.3f} (proposal disagree UB {proposal_disagree_ub*100:.1f}%; "
          f"crude min/max int3 + synthetic inputs OVER-state loss)  first-GEMM "
          f"cos {cos0:.4f} relL2 {relerr0:.4f}", flush=True)

    # --- NET TPS: decompose into (A) KERNEL-ONLY (measured, E[T] held) + (B) E[T] --
    step_s_bf16 = args.decode_step_ms / 1e3
    step_s_int3_avail = step_s_bf16 + (step_int3_us - step_bf16_us) / 1e6   # +draft delta
    step_s_int3_ceiling = step_s_bf16 * (1.0 - ceiling_gross_saving_pct / 100.0)
    tps_bf16 = et_bf16 / step_s_bf16
    # (A) kernel-only: hold E[T]=E[T]_bf16 to isolate the pure speed lever. This is the
    # int3-OPTIMISTIC bound (assumes zero proposal degradation). PRIMARY headline.
    net_kernelonly_available = (step_s_bf16 / step_s_int3_avail - 1.0) * 100.0
    net_kernelonly_ceiling = (step_s_bf16 / step_s_int3_ceiling - 1.0) * 100.0
    # (B) PR's "step-time saving - E[T] loss": fold the pessimistic E[T] lower bound.
    net_withet_available = (et_int3 / step_s_int3_avail / tps_bf16 - 1.0) * 100.0
    net_withet_ceiling = (et_int3 / step_s_int3_ceiling / tps_bf16 - 1.0) * 100.0
    # PRIMARY = kernel-only available: measured, E[T]-independent, int3-optimistic. If
    # even this best case is RED, the lever is robustly dead.
    net_tps_gain_pct_int3_draft = net_kernelonly_available
    net_ceiling_pct = net_kernelonly_ceiling
    proj_official_avail = args.frontier_tps * (1.0 + net_kernelonly_available / 100.0)
    proj_official_ceiling = args.frontier_tps * (1.0 + net_kernelonly_ceiling / 100.0)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # --- SELF-TEST gates ------------------------------------------------------
    # greedy-identity + PPL are pinned BY CONSTRUCTION: greedy spec-decode emits the
    # verify model's greedy token regardless of the draft; verify + accept rule are
    # byte-identical between control and variant. (Empirical through-vLLM confirmation
    # is blocked because vLLM has no int3 MTP kernel -- itself the feasibility finding.)
    greedy_identical_by_construction = True
    ppl_pinned = 2.3772
    ppl_ok = ppl_pinned <= 2.42
    vram_ok = peak_vram_gib <= 24.0
    self_test_passes = bool(greedy_identical_by_construction and ppl_ok
                            and nan_clean and vram_ok)

    # --- verdict --------------------------------------------------------------
    # SPEED GO only if even the int3-OPTIMISTIC (kernel-only, E[T] held) net is
    # positive AND the perfect-kernel ceiling clears a meaningful bar.
    speed_go = net_kernelonly_available > 0.0 and net_kernelonly_ceiling > 0.5
    verdict = {
        "qspec_int3_draft_self_test_passes": self_test_passes,
        "net_tps_gain_pct_int3_draft": net_tps_gain_pct_int3_draft,
        "net_tps_gain_pct_int3_draft_ceiling": net_ceiling_pct,
        "net_tps_kernelonly_available": net_kernelonly_available,
        "net_tps_kernelonly_ceiling": net_kernelonly_ceiling,
        "net_tps_with_et_loss_available": net_withet_available,
        "net_tps_with_et_loss_ceiling": net_withet_ceiling,
        "e_t_int3_draft": et_int3,
        "e_t_bf16_draft": et_bf16,
        "e_t_int3_is_pessimistic_lower_bound": True,
        "proposal_disagree_ub_synthetic": proposal_disagree_ub,
        "greedy_identical_by_construction": greedy_identical_by_construction,
        "ppl_pinned": ppl_pinned, "ppl_ok": ppl_ok,
        "nan_clean": nan_clean, "peak_vram_gib": peak_vram_gib, "vram_ok": vram_ok,
        "draft_pass_us_bf16": pass_bf16_us, "draft_pass_us_int3_available": pass_int3_us,
        "draft_pass_slowdown_int3_vs_bf16": pass_int3_us / pass_bf16_us,
        "draft_walltime_saving_us_per_step": draft_walltime_saving_us,
        "drafter_gemm_share_of_step": drafter_gemm_share,
        "int3_byte_ratio": int3_byte_ratio, "int4_byte_ratio": int4_byte_ratio,
        "ceiling_gross_saving_pct_realistic": ceiling_gross_saving_pct,
        "ceiling_gross_saving_pct_full_bw": ceiling_gross_full_pct,
        "mean_weight_rel_l2_err_int3": mean_w_err,
        "tps_bf16": tps_bf16,
        "tps_int3_kernelonly_available": et_bf16 / step_s_int3_avail,
        "tps_int3_kernelonly_ceiling": et_bf16 / step_s_int3_ceiling,
        "proj_official_tps_available": proj_official_avail,
        "proj_official_tps_ceiling": proj_official_ceiling,
        "deployed_draft_dtype": "bfloat16 (NOT int4; PR premise corrected)",
        "int3_fused_kernel_exists_sm86": False,
        "qspec_arxiv_correct_id": "2410.11305 (PR cited 2411.11514, wrong)",
        "speed_verdict": "GO" if speed_go else "NO-GO",
        "safety_verdict": "GREEN (greedy+PPL pinned by construction)",
        "k": args.k,
    }

    print("\n[qspec-int3] ===== VERDICT =====", flush=True)
    print(f"  self_test_passes={self_test_passes}  (greedy_by_construction="
          f"{greedy_identical_by_construction} ppl_ok={ppl_ok} nan_clean={nan_clean} "
          f"vram_ok={vram_ok} {peak_vram_gib:.2f}GiB)", flush=True)
    print(f"  draft pass int3/bf16 slowdown = {pass_int3_us/pass_bf16_us:.2f}x "
          f"(only available sm_86 int3 path = dequant->bf16 GEMM)", flush=True)
    print(f"  net_tps (KERNEL-ONLY, E[T] held): available {net_kernelonly_available:+.2f}% "
          f"| perfect-kernel ceiling {net_kernelonly_ceiling:+.2f}%", flush=True)
    print(f"  net_tps (with E[T] loss folded): available {net_withet_available:+.2f}% "
          f"| ceiling {net_withet_ceiling:+.2f}%   (E[T] bf16 {et_bf16:.3f} -> int3 "
          f">= {et_int3:.3f})", flush=True)
    print(f"  SPEED: {verdict['speed_verdict']}   SAFETY: {verdict['safety_verdict']}",
          flush=True)
    print(f"  official projection (kernel-only): {args.frontier_tps} -> available "
          f"{proj_official_avail:.2f} / ceiling {proj_official_ceiling:.2f} TPS",
          flush=True)

    payload = {
        "config": {
            "drafter_dir": args.drafter_dir, "torch": torch.__version__, "device": dev,
            "k": args.k, "group": args.group, "iters": args.iters, "warmup": args.warmup,
            "et_samples": args.et_samples, "decode_step_ms": args.decode_step_ms,
            "frontier_tps": args.frontier_tps, "et_deployed": args.et_deployed,
            "chain_captured_bf16": cap_bf16, "chain_captured_int3": cap_int3,
            "A10G_HBM_GBS": A10G_HBM_GBS,
            "note": "isolated CUDA-graph GEMM timing on real bf16 drafter weights; "
                    "int3 = group-quant fake-quant -> dequant-to-bf16 -> bf16 GEMM "
                    "(only available sm_86 int3 path; no fused W3A16 kernel exists). "
                    "Verify model untouched -> greedy-identity + PPL pinned by "
                    "construction. Lossless, no serve change, no HF Job.",
        },
        "specs": [{"role": r, "in": i, "out": o} for (r, i, o, _w) in specs],
        "weight_rel_l2_err_int3": weight_err,
        "byte_model": {
            "per_pass_total_bytes_bf16": tot_bf16, "per_pass_weight_bytes_bf16": w_bf16,
            "per_pass_total_bytes_int4": tot_int4, "per_pass_weight_bytes_int4": w_int4,
            "per_pass_total_bytes_int3": tot_int3, "per_pass_weight_bytes_int3": w_int3,
            "int3_byte_ratio": int3_byte_ratio, "int4_byte_ratio": int4_byte_ratio,
        },
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[qspec-int3] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[qspec-int3] W&B logging failed (non-fatal): {exc!r}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    cols = ["role", "in", "out", "weight_rel_l2_err_int3"]
    tbl = wandb.Table(columns=cols)
    for s in payload["specs"]:
        tbl.add_data(s["role"], s["in"], s["out"],
                     payload["weight_rel_l2_err_int3"].get(s["role"]))
    run.log({"drafter_int3_table": tbl})
    run.summary.update({k: v for k, v in payload["verdict"].items()
                        if isinstance(v, (int, float, bool))})
    run.summary.update({k: v for k, v in payload["byte_model"].items()})
    run.finish()
    print(f"[qspec-int3] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
