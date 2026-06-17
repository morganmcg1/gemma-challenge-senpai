#!/usr/bin/env python
"""PR #625 -- QuaRot body re-quant: CHEAP Frobenius pre-screen. LOCAL, NO HF JOB.

Tests the advisor's core mechanism BEFORE any serving: is the int4-body weight
quantization error *outlier-concentrated*, i.e. does folding a Hadamard rotation
into the weights BEFORE the int4 grid reduce the group-wise (g128) reconstruction
error? If the QAT-conditioned weights are already near-optimal for int4 (rotation
gives ~0 reduction), that is a decisive cheap DEFICIT_IRREDUCIBLE negative and we
never need to confront the (architecturally blocked) offline-R1 serving problem.

Source of truth for the bf16 body = DEQUANTIZED official g32 QAT checkpoint
(google/gemma-4-E4B-it-qat-w4a16-ct). These ARE the QAT weights the shipped int4
body derives from. We re-quantize them to g128 int4 exactly as
submissions/int4_g128_lmhead/build_quant.py does, then compare:
  - naive            : g128 int4, no rotation (== shipped pipeline)
  - hadamard_g128blk : rotate input dim by a block-diagonal 128x128 Hadamard
                       (group-ALIGNED; what an online per-group Hadamard would give)
  - randorth_full    : rotate full input dim by a fixed random orthogonal
                       (QuaRot-R1-style incoherence CEILING; not offline-servable here)
  - r2_head256 (o_proj only): rotate input dim by per-head 256x256 Hadamard
                       (the ONLY exactly-offline-foldable rotation on this sandwich-norm arch)

Error metric per matrix: rel = ||W - dequant(quant(W'))||_F / ||W||_F, where W'
is the rotated weight. Because the rotation is orthogonal, ||W'||=||W|| and the
rotated weight-error propagates to the layer output with the same Frobenius norm,
so naive vs rotated rel-errors are apples-to-apples.

Run:  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python research/quarot_body_requant/frobenius_screen.py
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/quarot_body_requant"
QAT_G32 = Path(
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors"
)
SHIPPED_G128 = Path("/workspace/gemma_build/int4_g128_lmhead/model.safetensors")
MODULE_LIST = ROOT / "submissions/int4_g128_lmhead/official_quantized_modules.json"
HEAD_DIM = 256

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def qargs(gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=4, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def hadamard(n: int, device=DEV) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard, n a power of 2."""
    assert (n & (n - 1)) == 0, f"{n} not power of 2"
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).to(device=device, dtype=torch.float32)


_ORTH_CACHE: dict[int, torch.Tensor] = {}


def rand_orth(n: int, device=DEV) -> torch.Tensor:
    if n not in _ORTH_CACHE:
        g = torch.Generator(device="cpu").manual_seed(1234 + n)
        a = torch.randn(n, n, generator=g, dtype=torch.float32)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # fix signs -> unique
        _ORTH_CACHE[n] = q.to(device)
    return _ORTH_CACHE[n]


def quant_g128_relerr(W: torch.Tensor, gs: int = 128) -> tuple[float, float]:
    """Group-wise symmetric int4 quant rel error (matches build_quant). Returns
    (rel_err, sse) where sse = ||W - dequant(quant(W))||_F^2."""
    W = W.to(torch.float32)
    out_dim, in_dim = W.shape
    qa = qargs(gs)
    ng = in_dim // gs
    wg = W.reshape(out_dim, ng, gs)
    mn = wg.amin(dim=-1)
    mx = wg.amax(dim=-1)
    scale, zp = calculate_qparams(mn, mx, qa)
    q = quantize(W, scale, zp, qa)
    deq = dequantize(q, scale, zp, qa)
    diff = (W - deq)
    sse = diff.pow(2).sum().item()
    rel = math.sqrt(sse) / W.norm().clamp_min(1e-9).item()
    return rel, sse


def rot_input_block_hadamard(W: torch.Tensor, blk: int) -> torch.Tensor:
    """Rotate input dim (columns) by a block-diagonal blk x blk Hadamard."""
    out_dim, in_dim = W.shape
    assert in_dim % blk == 0
    H = hadamard(blk, W.device)
    return (W.reshape(out_dim, in_dim // blk, blk) @ H).reshape(out_dim, in_dim)


def rot_input_full_orth(W: torch.Tensor) -> torch.Tensor:
    """Rotate full input dim (columns) by a fixed random orthogonal."""
    Q = rand_orth(W.shape[1], W.device)
    return W @ Q


def rot_output_block_hadamard(W: torch.Tensor, blk: int) -> torch.Tensor:
    """Rotate output dim (rows) by block-diagonal blk x blk Hadamard (for v_proj R2)."""
    out_dim, in_dim = W.shape
    assert out_dim % blk == 0
    H = hadamard(blk, W.device)
    return (W.t().reshape(in_dim, out_dim // blk, blk) @ H).reshape(in_dim, out_dim).t().contiguous()


def dequant_g32_module(f, base: str) -> torch.Tensor:
    packed = f.get_tensor(base + ".weight_packed").to(DEV)
    scale = f.get_tensor(base + ".weight_scale").to(DEV).to(torch.float32)
    shape = f.get_tensor(base + ".weight_shape").tolist()
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q_int8 = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1)
    zp = torch.zeros_like(scale)
    W = dequantize(q_int8.to(DEV), scale, zp, qargs(32))
    return W.to(torch.float32)


def mtype(base: str) -> str:
    for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
              "down_proj", "per_layer_input_gate", "per_layer_projection"):
        if base.endswith(t):
            return t
    if base.endswith("per_layer_model_projection"):
        return "per_layer_model_projection"
    return "other"


def main() -> None:
    t0 = time.time()
    modules = sorted(json.load(open(MODULE_LIST)))
    lang = [m for m in modules if "language_model" in m]
    print(f"[screen] device={DEV}  body modules={len(modules)}  language={len(lang)}", flush=True)

    # ---- faithfulness sanity vs shipped g128: do dequant(g32)->q(g128) codes match shipped? ----
    sanity = {}
    with safe_open(str(QAT_G32), framework="pt", device="cpu") as fq, \
         safe_open(str(SHIPPED_G128), framework="pt", device="cpu") as fs:
        ship_keys = set(fs.keys())
        for base in ["model.language_model.layers.0.self_attn.q_proj",
                     "model.language_model.layers.0.mlp.gate_proj"]:
            W = dequant_g32_module(fq, base)
            out_dim, in_dim = W.shape
            qa = qargs(128)
            ng = in_dim // 128
            wg = W.reshape(out_dim, ng, 128)
            scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qa)
            q = quantize(W, scale, zp, qa).to(torch.int8)
            my_packed = pack_to_int32(q.cpu(), 4, packed_dim=1)
            if base + ".weight_packed" in ship_keys:
                ship_packed = fs.get_tensor(base + ".weight_packed")
                match = bool(torch.equal(my_packed, ship_packed))
                frac = (my_packed == ship_packed).float().mean().item()
                sanity[base] = {"codes_exact_match": match, "frac_int32_equal": round(frac, 4)}
                print(f"[sanity] {base.split('.')[-1]}: exact={match} frac_eq={frac:.4f}", flush=True)

    # ---- Frobenius screen, module by module ----
    agg = defaultdict(lambda: defaultdict(float))  # type -> metric -> value
    cnt = defaultdict(int)
    global_sse = defaultdict(float)
    global_wnorm2 = 0.0

    with safe_open(str(QAT_G32), framework="pt", device="cpu") as fq:
        keys = set(fq.keys())
        done = 0
        for base in lang:
            if base + ".weight_packed" not in keys:
                print(f"[warn] missing {base}", flush=True)
                continue
            W = dequant_g32_module(fq, base)
            t = mtype(base)
            wn2 = W.pow(2).sum().item()
            global_wnorm2 += wn2

            rel_naive, sse_naive = quant_g128_relerr(W)
            rel_hblk, sse_hblk = quant_g128_relerr(rot_input_block_hadamard(W, 128))
            rel_orth, sse_orth = quant_g128_relerr(rot_input_full_orth(W))

            agg[t]["rel_naive"] += rel_naive
            agg[t]["rel_hblk"] += rel_hblk
            agg[t]["rel_orth"] += rel_orth
            cnt[t] += 1
            global_sse["naive"] += sse_naive
            global_sse["hblk"] += sse_hblk
            global_sse["orth"] += sse_orth

            # R2: the only offline-foldable rotation. o_proj input = 8 heads x 256.
            if t == "o_proj":
                rel_r2, sse_r2 = quant_g128_relerr(rot_input_block_hadamard(W, HEAD_DIM))
                agg[t]["rel_r2_head256"] += rel_r2
                global_sse["r2_oproj"] += sse_r2
            else:
                global_sse["r2_oproj"] += sse_naive  # o_proj is the only one R2 touches
            # v_proj output rotation (the partner of o_proj's R2): measure its error too
            if t == "v_proj":
                rel_v, _ = quant_g128_relerr(rot_output_block_hadamard(W, HEAD_DIM))
                agg[t]["rel_vout_head256"] += rel_v

            del W
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(lang)}  (last {t} naive={rel_naive:.4f} "
                      f"hblk={rel_hblk:.4f} orth={rel_orth:.4f})", flush=True)

    # ---- report ----
    print("\n==== per-matrix-type mean relative g128-int4 error ====", flush=True)
    hdr = f"{'type':<26}{'n':>4}{'naive':>9}{'had_g128':>10}{'randorth':>10}{'extra':>16}"
    print(hdr)
    rows = {}
    for t in sorted(agg):
        n = cnt[t]
        rn = agg[t]["rel_naive"] / n
        rh = agg[t]["rel_hblk"] / n
        ro = agg[t]["rel_orth"] / n
        extra = ""
        if t == "o_proj":
            extra = f"r2head={agg[t]['rel_r2_head256']/n:.4f}"
        if t == "v_proj":
            extra = f"vout={agg[t]['rel_vout_head256']/n:.4f}"
        print(f"{t:<26}{n:>4}{rn:>9.4f}{rh:>10.4f}{ro:>10.4f}{extra:>16}")
        rows[t] = {"n": n, "rel_naive": rn, "rel_had_g128blk": rh,
                   "rel_randorth_full": ro,
                   "rel_extra": (agg[t].get("rel_r2_head256", 0)/n if t == "o_proj"
                                 else agg[t].get("rel_vout_head256", 0)/n if t == "v_proj" else None)}

    # global SSE-weighted reduction (the headline)
    def red(key):
        return 1.0 - global_sse[key] / global_sse["naive"]
    headline = {
        "global_rel_naive": math.sqrt(global_sse["naive"] / global_wnorm2),
        "global_rel_had_g128blk": math.sqrt(global_sse["hblk"] / global_wnorm2),
        "global_rel_randorth_full": math.sqrt(global_sse["orth"] / global_wnorm2),
        "sse_reduction_had_g128blk": red("hblk"),
        "sse_reduction_randorth_full": red("orth"),
        "sse_reduction_r2_oproj_only": red("r2_oproj"),
    }
    print("\n==== GLOBAL (param-SSE-weighted over language body) ====")
    for k, v in headline.items():
        print(f"  {k:<32} {v:+.5f}" if "reduction" in k else f"  {k:<32} {v:.5f}")

    out = {
        "pr": 625, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "source": "dequant(google/gemma-4-E4B-it-qat-w4a16-ct g32 QAT)",
        "quantizer": "compressed-tensors g128 symmetric int4 minmax (== build_quant.py)",
        "device": DEV,
        "faithfulness_vs_shipped_g128": sanity,
        "per_type": rows,
        "global": headline,
        "n_language_modules": len(lang),
        "seconds": round(time.time() - t0, 1),
        "note": ("had_g128blk = group-aligned 128-Hadamard on input dim (online-servable only); "
                 "randorth_full = QuaRot-R1 incoherence ceiling (NOT offline-foldable on sandwich-norm gemma4); "
                 "r2_head256 on o_proj = the ONLY exactly-offline-foldable rotation."),
    }
    (HERE / "frobenius_screen.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {HERE/'frobenius_screen.json'}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
