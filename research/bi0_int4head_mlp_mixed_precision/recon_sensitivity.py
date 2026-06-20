#!/usr/bin/env python
"""PR #810 Step 1a — per-layer body-MLP weight-reconstruction sensitivity (W4/W3/W2).

OFFLINE, no model forward, no serving. Reference = the dequantized W4 weights the
int4head base actually serves (`W4deq`). For each (layer, proj) we requantize
W4deq to {W4, W3, W2} group_size=32 symmetric int (the same compressed-tensors
scheme as the body) and measure how much extra distortion each bit-width adds:

  rel_err = ||W4deq - W_b_deq|| / ||W4deq||
  mse     = mean((W4deq - W_b_deq)^2)
  sqnr_dB = 10*log10( sum(W4deq^2) / sum((W4deq - W_b_deq)^2) )

W4 self-requant is a sanity check (should be ~0). This is the cheap signal (a)
of the sensitivity map; signal (b) is the PPL-delta sweep (separate script).

Caveat: W4deq (not the original QAT high-precision weights, which are not
published) is the reference — this is the faithful, exactly-reproducible
"buildable from the shipped checkpoint" number.

LOCAL ONLY. Reuses compressed-tensors primitives (same as build_lmhead_quant.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # local A10G pod: torch must see GPU 0

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from compressed_tensors.quantization import QuantizationArgs  # noqa: E402
from compressed_tensors.quantization.lifecycle.forward import (  # noqa: E402
    quantize,
    dequantize,
)
from compressed_tensors.quantization.utils.helpers import calculate_qparams  # noqa: E402
from compressed_tensors.compressors.pack_quantized.helpers import (  # noqa: E402
    unpack_from_int32,
)

GROUP_SIZE = 32
N_LAYERS = 42
PROJS = ["gate_proj", "up_proj", "down_proj"]
BITS = [4, 3, 2]


def make_qargs(num_bits: int, group_size: int) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=num_bits, type="int", strategy="group", group_size=group_size,
        symmetric=True, observer="minmax",
    )


def requant_error(w4deq: torch.Tensor, num_bits: int, group_size: int):
    """Requantize a (already-dequantized) weight to num_bits g32 sym; return metrics."""
    out_dim, in_dim = w4deq.shape
    qargs = make_qargs(num_bits, group_size)
    ng = in_dim // group_size
    wg = w4deq.reshape(out_dim, ng, group_size)
    scale, zp = calculate_qparams(wg.amin(dim=-1), wg.amax(dim=-1), qargs)
    q = quantize(w4deq, scale, zp, qargs)
    deq = dequantize(q, scale, zp, qargs)
    diff = (w4deq - deq).to(torch.float64)
    num = torch.sum(diff * diff)
    den_w = torch.sum(w4deq.to(torch.float64) ** 2)
    rel = float(torch.sqrt(num / den_w.clamp_min(1e-12)))
    mse = float(num / w4deq.numel())
    sqnr_db = float(10.0 * torch.log10(den_w.clamp_min(1e-12) / num.clamp_min(1e-30)))
    return {"rel_err": rel, "mse": mse, "sqnr_db": sqnr_db}


def load_w4deq(f, prefix: str, device) -> torch.Tensor:
    packed = f.get_tensor(f"{prefix}.weight_packed").to(device)
    scale = f.get_tensor(f"{prefix}.weight_scale").to(device)
    shape = f.get_tensor(f"{prefix}.weight_shape").tolist()
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q4 = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1)
    qargs4 = make_qargs(4, GROUP_SIZE)
    zp = torch.zeros_like(scale)
    w4deq = dequantize(q4, scale, zp, qargs4).to(torch.float32)
    return w4deq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/"
        "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"))
    ap.add_argument("--output",
                    default="research/bi0_int4head_mlp_mixed_precision/recon_sensitivity.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get(
        "WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="body-mlp-mixed-precision")
    ap.add_argument("--wandb_name", default="stark/recon-sensitivity-map")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[recon] device={device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})",
          flush=True)

    st_path = Path(args.src) / "model.safetensors"
    assert st_path.exists(), f"missing {st_path}"

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, name=args.wandb_name,
                             job_type="analysis",
                             config={"group_size": GROUP_SIZE, "n_layers": N_LAYERS,
                                     "bits": BITS, "projs": PROJS,
                                     "reference": "W4deq (served int4 body)",
                                     "src": args.src})
            print(f"[recon] W&B run: {run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[recon] W&B init failed: {exc!r}", flush=True)

    t0 = time.time()
    rows = []  # per (layer, proj, bits)
    # bytes-per-element of the WEIGHT payload (excludes bf16 scale, which is fixed).
    bytes_per_elem = {4: 0.5, 3: 0.375, 2: 0.25}

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for L in range(N_LAYERS):
            for proj in PROJS:
                prefix = f"model.language_model.layers.{L}.mlp.{proj}"
                w4deq = load_w4deq(f, prefix, device)
                numel = w4deq.numel()
                for b in BITS:
                    m = requant_error(w4deq, b, GROUP_SIZE)
                    rows.append({
                        "layer": L, "proj": proj, "bits": b,
                        "numel": numel,
                        "weight_bytes": numel * bytes_per_elem[b],
                        **m,
                    })
                del w4deq
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if L % 8 == 0 or L == N_LAYERS - 1:
                print(f"[recon] layer {L:2d} done ({time.time()-t0:.1f}s)", flush=True)

    # ---- per-layer aggregate (sum over 3 projections), per bit-width ----
    per_layer = {}
    for L in range(N_LAYERS):
        per_layer[L] = {}
        for b in BITS:
            lr = [r for r in rows if r["layer"] == L and r["bits"] == b]
            tot_num = sum(r["mse"] * r["numel"] for r in lr)
            tot_den = sum(r["mse"] * r["numel"] / max(r["rel_err"] ** 2, 1e-30)
                          for r in lr)  # = sum ||w||^2
            tot_elem = sum(r["numel"] for r in lr)
            rel = math.sqrt(tot_num / max(tot_den, 1e-12))
            sqnr = 10.0 * math.log10(max(tot_den, 1e-12) / max(tot_num, 1e-30))
            per_layer[L][b] = {"rel_err": rel, "sqnr_db": sqnr,
                               "mse": tot_num / tot_elem}

    # ---- report: rank layers by W3 and W2 rel_err (most robust = lowest err) ----
    print("\n[recon] per-layer rel_err (W3 | W2), sorted by W3 robustness:", flush=True)
    order = sorted(range(N_LAYERS), key=lambda L: per_layer[L][3]["rel_err"])
    for L in order:
        print(f"  L{L:2d}: W3 rel={per_layer[L][3]['rel_err']:.4f} "
              f"sqnr={per_layer[L][3]['sqnr_db']:5.1f}dB | "
              f"W2 rel={per_layer[L][2]['rel_err']:.4f} "
              f"sqnr={per_layer[L][2]['sqnr_db']:5.1f}dB", flush=True)

    w4_self = max(r["rel_err"] for r in rows if r["bits"] == 4)
    print(f"\n[recon] W4 self-requant max rel_err (sanity, want ~0): {w4_self:.2e}", flush=True)

    payload = {
        "config": {"group_size": GROUP_SIZE, "n_layers": N_LAYERS, "bits": BITS,
                   "projs": PROJS, "reference": "W4deq (served int4 body)",
                   "src": args.src, "elapsed_s": time.time() - t0,
                   "w4_self_requant_max_rel_err": w4_self},
        "rows": rows,
        "per_layer": {str(L): per_layer[L] for L in per_layer},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[recon] wrote {args.output} ({len(rows)} rows, {time.time()-t0:.1f}s)", flush=True)

    if run is not None:
        try:
            import wandb
            cols = ["layer", "proj", "bits", "rel_err", "mse", "sqnr_db",
                    "numel", "weight_bytes"]
            tbl = wandb.Table(columns=cols)
            for r in rows:
                tbl.add_data(r["layer"], r["proj"], r["bits"], r["rel_err"],
                             r["mse"], r["sqnr_db"], r["numel"], r["weight_bytes"])
            run.log({"recon_sensitivity_table": tbl})
            # per-layer W3/W2 rel_err as line series over layer index
            for b in (3, 2):
                for L in range(N_LAYERS):
                    run.log({f"W{b}_rel_err_by_layer": per_layer[L][b]["rel_err"],
                             f"W{b}_sqnr_db_by_layer": per_layer[L][b]["sqnr_db"],
                             "layer_idx": L})
            run.summary.update({"w4_self_requant_max_rel_err": w4_self,
                                "n_rows": len(rows)})
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[recon] W&B log failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
