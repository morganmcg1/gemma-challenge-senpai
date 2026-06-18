#!/usr/bin/env python
"""Build corner C = g128-body + tied-bf16-head (PR #649 speed-trade attribution).

Takes the locked ``int4_g128_lmhead`` rung (g128 body + **int4-untied** lm_head)
and produces a checkpoint with the SAME g128 int4 body but a **tied bf16** lm_head
-- i.e. the locked body with the official-recipe head. Comparing C against the
locked rung D isolates the head-precision speed trade (D - C); against the g32
official A it isolates the group-size trade (C - A).

The only change is the lm_head treatment:
  - config: ``tie_word_embeddings = True``; drop the ``re:.*lm_head`` quant group;
    add ``lm_head`` to ``ignore`` (mirrors the official g32 w4a16-ct config).
  - vLLM then ties the logits projection to the bf16 ``embed_tokens.weight`` and
    does NOT build a separate quantized lm_head.

Two modes:
  symlink  (default) -- symlink the original model.safetensors; rely on vLLM to
           tie lm_head and skip the now-unreferenced lm_head.* tensors. ZERO big
           disk. Validity must be confirmed by a load smoke + a logit check.
  rewrite  -- physically drop the lm_head.* tensors and write a fresh
           model.safetensors. Needs ~10 GiB free; use if vLLM rejects the
           leftover tensors in symlink mode.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SRC = Path("/workspace/gemma_build/int4_g128_lmhead")
DST = Path("/workspace/gemma_build/g128_bf16head")
LMHEAD_TENSOR_PREFIX = "lm_head."


def _free_gib(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def surgery_config(src_cfg: dict) -> dict:
    cfg = json.loads(json.dumps(src_cfg))  # deep copy
    cfg["tie_word_embeddings"] = True
    qc = cfg.get("quantization_config", {})
    groups = qc.get("config_groups", {})
    # drop any group whose targets reference lm_head
    drop = []
    for gname, gv in groups.items():
        tgts = gv.get("targets", []) or []
        if any("lm_head" in str(t) for t in tgts):
            drop.append(gname)
    for g in drop:
        groups.pop(g, None)
    qc["config_groups"] = groups
    ig = list(qc.get("ignore", []) or [])
    if "lm_head" not in ig:
        ig.append("lm_head")
    qc["ignore"] = ig
    cfg["quantization_config"] = qc
    return cfg, drop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["symlink", "rewrite"], default="symlink")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--force", action="store_true", help="overwrite existing dst")
    args = ap.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"src not found: {args.src}")
    if args.dst.exists():
        if not args.force:
            raise SystemExit(f"dst exists: {args.dst} (use --force)")
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    src_cfg = json.loads((args.src / "config.json").read_text())
    new_cfg, dropped = surgery_config(src_cfg)
    (args.dst / "config.json").write_text(json.dumps(new_cfg, indent=2))
    print(f"[buildC] config: tie_word_embeddings=True, dropped quant groups={dropped}, "
          f"lm_head added to ignore", flush=True)

    # copy/symlink all aux files (everything except the big safetensors)
    for p in sorted(args.src.iterdir()):
        if p.name in ("config.json", "model.safetensors"):
            continue
        (args.dst / p.name).symlink_to(p.resolve())
        print(f"[buildC] symlink {p.name}", flush=True)

    st_src = args.src / "model.safetensors"
    if args.mode == "symlink":
        (args.dst / "model.safetensors").symlink_to(st_src.resolve())
        print(f"[buildC] model.safetensors -> SYMLINK {st_src} (lm_head.* left in place; vLLM should skip them under tie)", flush=True)
    else:
        from safetensors import safe_open
        from safetensors.torch import save_file
        free = _free_gib(args.dst)
        size_gib = st_src.stat().st_size / 1024**3
        print(f"[buildC] rewrite: src safetensors {size_gib:.2f} GiB, free {free:.2f} GiB", flush=True)
        if free < size_gib + 2.0:
            raise SystemExit(f"insufficient disk: need ~{size_gib + 2.0:.1f} GiB free, have {free:.1f} GiB. "
                             "Free space first (e.g. remove an unused HF cache) or use --mode symlink.")
        tensors = {}
        meta = {}
        dropped_keys = []
        with safe_open(st_src, framework="pt") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                if k.startswith(LMHEAD_TENSOR_PREFIX):
                    dropped_keys.append(k)
                    continue
                tensors[k] = f.get_tensor(k)
        save_file(tensors, str(args.dst / "model.safetensors"), metadata=meta)
        print(f"[buildC] rewrote model.safetensors dropping {dropped_keys}", flush=True)

    print(f"[buildC] DONE -> {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
