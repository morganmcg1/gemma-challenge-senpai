#!/usr/bin/env python
"""Byte-identity proof for the token-free startup-quant serve path (PR #801).

Proves that quantizing the PUBLIC base lm_head at serve startup (serve.py
LMHEAD_QUANT_AT_STARTUP=1 -> build_lmhead_quant.py) yields a checkpoint whose
lm_head is BIT-IDENTICAL to the validated, pre-built int4head checkpoint that was
published to the private Hub repo. If the three lm_head tensors
(weight_packed/weight_scale/weight_shape) match bit-for-bit, the served logits —
hence PPL, greedy decode, and the leaderboard gate — are identical by
construction, so the token-free path needs no quality re-validation beyond the
local gate smoke.

Two modes (both compare ONLY the lm_head tensors; the int4 body + towers are
copied byte-for-byte from the same source by the builder, so they are identical
by construction and not re-checked here):

  * --cand <dir>       compare ref-dir lm_head vs an already-built checkpoint dir
                       (use this on the checkpoint serve.py actually built at
                       startup, e.g. /tmp/int4head_startupq).
  * --from-base <dir>  re-quantize the base snapshot's lm_head IN MEMORY with the
                       exact builder primitive and compare to ref-dir (the
                       cheap determinism pre-check, no 10GB write).

Exit 0 iff every lm_head tensor is bit-identical, else 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Import the EXACT builder primitive so the in-memory re-quant uses identical math.
SUBMISSION_DIR = Path(__file__).resolve().parents[4] / "submissions" / "int4_mtp_bi0_int4head"
sys.path.insert(0, str(SUBMISSION_DIR))
import build_lmhead_quant  # noqa: E402

LM_HEAD_TENSORS = ("lm_head.weight_packed", "lm_head.weight_scale", "lm_head.weight_shape")


def _raw_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _load_lmhead(ckpt_dir: str) -> dict[str, torch.Tensor]:
    st = Path(ckpt_dir) / "model.safetensors"
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(st), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for name in LM_HEAD_TENSORS:
            if name not in keys:
                raise KeyError(f"{name} not in {st} (keys with lm_head: "
                               f"{[k for k in keys if 'lm_head' in k]})")
            out[name] = f.get_tensor(name)
    return out


def _quant_from_base(base_dir: str, num_bits: int, group_size: int) -> dict[str, torch.Tensor]:
    st = Path(base_dir) / "model.safetensors"
    with safe_open(str(st), framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight")
    packed, scale, shape, rel = build_lmhead_quant.quantize_weight(w, num_bits, group_size)
    print(f"[from-base] re-quantized lm_head rel_err={rel:.5f} "
          f"packed={tuple(packed.shape)} scale={tuple(scale.shape)}", flush=True)
    return {
        "lm_head.weight_packed": packed,
        "lm_head.weight_scale": scale,
        "lm_head.weight_shape": shape,
    }


def _compare(ref: dict[str, torch.Tensor], cand: dict[str, torch.Tensor]) -> bool:
    all_ok = True
    report = {}
    for name in LM_HEAD_TENSORS:
        a, b = ref[name], cand[name]
        equal = (a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b))
        ha, hb = _raw_sha256(a), _raw_sha256(b)
        report[name] = {
            "ref_shape": list(a.shape), "cand_shape": list(b.shape),
            "ref_dtype": str(a.dtype), "cand_dtype": str(b.dtype),
            "ref_sha256": ha, "cand_sha256": hb,
            "bit_identical": bool(equal and ha == hb),
        }
        mark = "OK " if report[name]["bit_identical"] else "MISMATCH"
        print(f"  [{mark}] {name:<24} ref={ha[:16]}… cand={hb[:16]}… "
              f"shape={list(a.shape)} dtype={str(a.dtype)}", flush=True)
        all_ok = all_ok and report[name]["bit_identical"]
    print(json.dumps({"all_bit_identical": all_ok, "tensors": report}, indent=2))
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="validated int4head checkpoint dir")
    ap.add_argument("--cand", default=None, help="built checkpoint dir to compare (serve.py output)")
    ap.add_argument("--from-base", default=None, help="base snapshot dir to re-quantize in memory")
    ap.add_argument("--num-bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()
    if (args.cand is None) == (args.from_base is None):
        ap.error("pass exactly one of --cand or --from-base")

    print(f"[ref]  {args.ref}", flush=True)
    ref = _load_lmhead(args.ref)
    if args.cand is not None:
        print(f"[cand] {args.cand}", flush=True)
        cand = _load_lmhead(args.cand)
    else:
        print(f"[from-base] {args.from_base} (num_bits={args.num_bits} group_size={args.group_size})", flush=True)
        cand = _quant_from_base(args.from_base, args.num_bits, args.group_size)

    ok = _compare(ref, cand)
    print(f"\nBYTE-IDENTITY: {'PASS — lm_head bit-identical' if ok else 'FAIL — lm_head DIFFERS'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
