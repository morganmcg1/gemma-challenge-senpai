#!/usr/bin/env python
"""PR #639 -- quantify the MSE observer's int4 round-trip error reduction vs the
live minmax recipe, on the REAL Gemma body weights (not synthetic). Loads the
QAT-unq source once and re-quantizes the 343 official body modules + untied
lm_head under both observers at g128 (the shipped group size). Mean rel_err is
the Frobenius ratio ||w-deq||/||w||; MSE <= minmax per module by construction
(the clip grid includes ratio=1.0). Report-only; writes nothing to disk."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "submissions/int4_g128_lmhead")
from build_quant import quantize_weight  # noqa: E402

SRC = Path("/workspace/gemma_build/qat_unq/model.safetensors")
MODS = set(json.load(open("submissions/int4_g128_lmhead/official_quantized_modules.json")))
EMBED = "model.language_model.embed_tokens.weight"
GS = 128

def main():
    qnames = {m + ".weight" for m in MODS}
    mm, ms, impr = [], [], []
    embed = None
    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name == EMBED:
                embed = f.get_tensor(name)
            if name in qnames:
                w = f.get_tensor(name)
                _, _, _, r_mm = quantize_weight(w, GS, "minmax")
                _, _, _, r_ms = quantize_weight(w, GS, "mse")
                mm.append(r_mm); ms.append(r_ms); impr.append((r_mm - r_ms) / max(r_mm, 1e-9))
    # untied head
    _, _, _, h_mm = quantize_weight(embed, GS, "minmax")
    _, _, _, h_ms = quantize_weight(embed, GS, "mse")
    n = len(mm)
    print(f"body modules: {n}")
    print(f"  minmax g128 mean rel_err = {sum(mm)/n:.4f}  (min {min(mm):.4f} max {max(mm):.4f})")
    print(f"  mse    g128 mean rel_err = {sum(ms)/n:.4f}  (min {min(ms):.4f} max {max(ms):.4f})")
    print(f"  mean per-module improvement = {100*sum(impr)/n:.2f}%  (max {100*max(impr):.2f}%)")
    print(f"  modules where mse strictly helps: {sum(1 for d in impr if d>1e-6)}/{n}")
    print(f"lm_head: minmax rel_err={h_mm:.4f}  mse rel_err={h_ms:.4f}  impr={100*(h_mm-h_ms)/h_mm:.2f}%")

if __name__ == "__main__":
    main()
