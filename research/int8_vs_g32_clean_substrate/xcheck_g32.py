#!/usr/bin/env python
"""GPU-free substrate pre-gate (PR #726 / ubel).

Proves whether the Path-B Arm1 weights (fake_quant of the qat_unq bf16 master at
int4-g32) are the SAME weights as the #702 full_g32 substrate (the served
w4a16-ct int4-g32 checkpoint, group_size=32 symmetric -- confirmed). If
rel_err ~ 0 across modules, qat_unq is provably the master w4a16-ct was built
from, so Arm1 MUST reproduce the #702 nqk9izab pooled acc 0.3867 (same weights,
same serve, same sampled protocol) -- de-risking the ~2h Arm1 GPU spend BEFORE
it runs. A non-zero rel_err is itself the advisor's flag that
'/tmp/gemma40L-int4 != g32-from-qat_unq'.

Compares, per body module:
  A = fake_quant(qat_unq.<mod>.weight bf16, int4 g32)        [Arm1 recipe]
  B = dequant(w4a16-ct.<mod>.weight_packed, g32)             [#702 full_g32 = served]
  rel_err = ||A - B|| / ||B||
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

LOCUS_LO, LOCUS_HI = 14, 27


def qargs(gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=4, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def fake_quant_g32(w: torch.Tensor, gs: int = 32) -> torch.Tensor:
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    a = qargs(gs)
    ng = in_dim // gs
    wg = w.reshape(out_dim, ng, gs)
    scale, zp = calculate_qparams(wg.amin(dim=-1), wg.amax(dim=-1), a)
    return dequantize(quantize(w, scale, zp, a), scale, zp, a)


def dequant_packed(packed, scale, shape) -> torch.Tensor:
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1).to(torch.float32)
    ng = scale.shape[1]
    gs = in_dim // ng
    qg = q.reshape(out_dim, ng, gs)
    return (qg * scale.float().unsqueeze(-1)).reshape(out_dim, in_dim)


def layer_of(mod):
    m = re.search(r"\.layers\.(\d+)\.", mod)
    return int(m.group(1)) if m else None


def main() -> None:
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = all 343 modules; else first N (sorted)")
    ap.add_argument("--out", default="research/int8_vs_g32_clean_substrate/xcheck_g32.json")
    args = ap.parse_args()

    qat = Path(snapshot_download("google/gemma-4-E4B-it-qat-q4_0-unquantized", local_files_only=True))
    w4 = Path(snapshot_download("google/gemma-4-E4B-it-qat-w4a16-ct", local_files_only=True))
    mods = sorted(json.load(open("submissions/int4_g128_lmhead/official_quantized_modules.json")))
    if args.sample:
        mods = mods[: args.sample]

    rows = []
    with safe_open(str(qat / "model.safetensors"), framework="pt", device="cpu") as fq, \
         safe_open(str(w4 / "model.safetensors"), framework="pt", device="cpu") as fw:
        for i, mod in enumerate(mods):
            A = fake_quant_g32(fq.get_tensor(mod + ".weight"))
            B = dequant_packed(fw.get_tensor(mod + ".weight_packed"),
                               fw.get_tensor(mod + ".weight_scale"),
                               fw.get_tensor(mod + ".weight_shape"))
            rel = float((A - B).norm() / B.norm().clamp_min(1e-9))
            rows.append({"mod": mod, "layer": layer_of(mod), "rel_err": rel,
                         "in_locus": layer_of(mod) is not None and LOCUS_LO <= layer_of(mod) <= LOCUS_HI})
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(mods)} (last rel_err {rel:.2e})", flush=True)

    rel_all = sorted(r["rel_err"] for r in rows)
    rel_locus = sorted(r["rel_err"] for r in rows if r["in_locus"])
    n = len(rel_all)

    def stats(xs):
        if not xs:
            return {}
        return {"n": len(xs), "min": xs[0], "max": xs[-1],
                "mean": sum(xs) / len(xs), "median": xs[len(xs) // 2]}

    worst = sorted(rows, key=lambda r: -r["rel_err"])[:8]
    # identity threshold: int4-g32 is the same quantization on both sides, so if qat_unq
    # IS the master, A and B differ only by float round-off in the (deterministic) minmax
    # calc -> rel_err << 1e-3. Anything above ~1e-2 means a different master.
    THRESH = 1e-2
    identical = rel_all[-1] < THRESH if rel_all else False
    out = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "comparison": "A=fake_quant(qat_unq.weight,int4-g32)  B=dequant(w4a16-ct packed g32)",
        "n_modules": n,
        "rel_err_all": stats(rel_all),
        "rel_err_locus_L14_27": stats(rel_locus),
        "identity_threshold": THRESH,
        "substrate_is_master": bool(identical),
        "worst_modules": worst,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("=" * 80)
    print(f"qat_unq-g32 vs w4a16-ct: n={n}  rel_err all "
          f"min={out['rel_err_all']['min']:.2e} mean={out['rel_err_all']['mean']:.2e} "
          f"max={out['rel_err_all']['max']:.2e}")
    if rel_locus:
        print(f"  locus L14-27 (n={len(rel_locus)}): mean={out['rel_err_locus_L14_27']['mean']:.2e} "
              f"max={out['rel_err_locus_L14_27']['max']:.2e}")
    print(f"  worst: " + ", ".join(f"{w['mod'].split('.')[-1]}@L{w['layer']}={w['rel_err']:.2e}" for w in worst[:5]))
    print(f"SUBSTRATE_IS_MASTER (max rel_err < {THRESH}): {identical}")
    print(f"  => {'Arm1 will reproduce 0.3867 (same weights as #702 full_g32)' if identical else 'FLAG: qat_unq != the w4a16-ct master; Arm1 may diverge from 0.3867'}")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
