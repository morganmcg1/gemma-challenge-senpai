#!/usr/bin/env python
"""PR #625 -- Frobenius pre-screen on the TRUE bf16 QAT master (airtight source).

Re-runs the weight-error screen on google/gemma-4-E4B-it-qat-q4_0-unquantized --
the bf16 QAT master that submissions/int4_g128_lmhead/build_quant.py used as its
--src (qat_unq). This removes the g32-grid pre-snapping bias of the dequant(g32)
source and is the EXACT source the shipped int4 body was quantized from.

Faithfulness gate: quant_g128(bf16-master) codes must EXACTLY match the shipped
int4_g128 weight_packed (since the shipped body == quant_g128 of this master).

Run:  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python research/quarot_body_requant/frobenius_screen_bf16.py
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from frobenius_screen import (
    DEV, HEAD_DIM, qargs, quant_g128_relerr, rot_input_block_hadamard,
    rot_input_full_orth, rot_output_block_hadamard, mtype,
)

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/quarot_body_requant"
BF16_MASTER = Path(
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/"
    "dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6/model.safetensors"
)
SHIPPED_G128 = Path("/workspace/gemma_build/int4_g128_lmhead/model.safetensors")
MODULE_LIST = ROOT / "submissions/int4_g128_lmhead/official_quantized_modules.json"


def main() -> None:
    t0 = time.time()
    modules = sorted(json.load(open(MODULE_LIST)))
    lang = [m for m in modules if "language_model" in m]
    print(f"[bf16-screen] device={DEV}  language body modules={len(lang)}", flush=True)

    # ---- faithfulness: quant_g128(bf16 master) must EXACTLY reproduce shipped int4_g128 codes ----
    sanity = {}
    with safe_open(str(BF16_MASTER), framework="pt", device="cpu") as fm, \
         safe_open(str(SHIPPED_G128), framework="pt", device="cpu") as fs:
        ship_keys = set(fs.keys())
        for base in ["model.language_model.layers.0.self_attn.q_proj",
                     "model.language_model.layers.0.mlp.gate_proj",
                     "model.language_model.layers.20.mlp.down_proj"]:
            W = fm.get_tensor(base + ".weight").to(DEV).to(torch.float32)
            out_dim, in_dim = W.shape
            qa = qargs(128)
            wg = W.reshape(out_dim, in_dim // 128, 128)
            scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qa)
            q = quantize(W, scale, zp, qa).to(torch.int8)
            my_packed = pack_to_int32(q.cpu(), 4, packed_dim=1)
            if base + ".weight_packed" in ship_keys:
                ship = fs.get_tensor(base + ".weight_packed")
                match = bool(torch.equal(my_packed, ship))
                frac = (my_packed == ship).float().mean().item()
                sanity[base] = {"codes_exact_match": match, "frac_int32_equal": round(frac, 6)}
                print(f"[sanity] {'.'.join(base.split('.')[-3:])}: exact={match} frac_eq={frac:.6f}", flush=True)

    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    global_sse = defaultdict(float)
    global_wnorm2 = 0.0

    with safe_open(str(BF16_MASTER), framework="pt", device="cpu") as fm:
        keys = set(fm.keys())
        done = 0
        for base in lang:
            wk = base + ".weight"
            if wk not in keys:
                print(f"[warn] missing {wk}", flush=True)
                continue
            W = fm.get_tensor(wk).to(DEV).to(torch.float32)
            t = mtype(base)
            global_wnorm2 += W.pow(2).sum().item()

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

            if t == "o_proj":
                rel_r2, sse_r2 = quant_g128_relerr(rot_input_block_hadamard(W, HEAD_DIM))
                agg[t]["rel_r2_head256"] += rel_r2
                global_sse["r2_oproj"] += sse_r2
            else:
                global_sse["r2_oproj"] += sse_naive
            if t == "v_proj":
                rel_v, _ = quant_g128_relerr(rot_output_block_hadamard(W, HEAD_DIM))
                agg[t]["rel_vout_head256"] += rel_v

            del W
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(lang)} (last {t} naive={rel_naive:.4f} hblk={rel_hblk:.4f} orth={rel_orth:.4f})", flush=True)

    print("\n==== per-matrix-type mean relative g128-int4 error (TRUE bf16 master) ====")
    print(f"{'type':<26}{'n':>4}{'naive':>9}{'had_g128':>10}{'randorth':>10}{'extra':>16}")
    rows = {}
    for t in sorted(agg):
        n = cnt[t]
        rn, rh, ro = agg[t]["rel_naive"]/n, agg[t]["rel_hblk"]/n, agg[t]["rel_orth"]/n
        extra = (f"r2head={agg[t]['rel_r2_head256']/n:.4f}" if t == "o_proj"
                 else f"vout={agg[t]['rel_vout_head256']/n:.4f}" if t == "v_proj" else "")
        print(f"{t:<26}{n:>4}{rn:>9.4f}{rh:>10.4f}{ro:>10.4f}{extra:>16}")
        rows[t] = {"n": n, "rel_naive": rn, "rel_had_g128blk": rh, "rel_randorth_full": ro,
                   "rel_extra": (agg[t].get("rel_r2_head256", 0)/n if t == "o_proj"
                                 else agg[t].get("rel_vout_head256", 0)/n if t == "v_proj" else None)}

    def red(k):
        return 1.0 - global_sse[k]/global_sse["naive"]
    headline = {
        "global_rel_naive": math.sqrt(global_sse["naive"]/global_wnorm2),
        "global_rel_had_g128blk": math.sqrt(global_sse["hblk"]/global_wnorm2),
        "global_rel_randorth_full": math.sqrt(global_sse["orth"]/global_wnorm2),
        "sse_reduction_had_g128blk": red("hblk"),
        "sse_reduction_randorth_full": red("orth"),
        "sse_reduction_r2_oproj_only": red("r2_oproj"),
    }
    print("\n==== GLOBAL (param-SSE-weighted over language body) ====")
    for k, v in headline.items():
        print(f"  {k:<32} {v:+.5f}" if "reduction" in k else f"  {k:<32} {v:.5f}")

    out = {
        "pr": 625, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "source": "google/gemma-4-E4B-it-qat-q4_0-unquantized (TRUE bf16 QAT master == build_quant.py qat_unq)",
        "quantizer": "compressed-tensors g128 symmetric int4 minmax (== build_quant.py)",
        "device": DEV, "faithfulness_vs_shipped_g128": sanity,
        "per_type": rows, "global": headline,
        "n_language_modules": len(lang), "seconds": round(time.time()-t0, 1),
    }
    (HERE / "frobenius_screen_bf16.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {HERE/'frobenius_screen_bf16.json'} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
