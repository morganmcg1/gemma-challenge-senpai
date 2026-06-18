#!/usr/bin/env python
"""Offline int4 W4A16 re-quant at an arbitrary BODY group-size (PR #679).

Reuses submissions/int4_g128_lmhead/build_quant.py's EXACT quant math
(quantize_weight, minmax observer, compressed-tensors pack layout) but sources
the quantization_config from the on-disk g128 checkpoint instead of fetching it
from the Hub -- so the build is fully offline and needs no HF_TOKEN/network.

Sweep axis: BODY group_size only. The lm_head ("group_1") stays at the shipped
g128 grid for every arm ("everything else fixed", per the PR). Only group_0
(the 343 language-model Linear modules) changes group_size.

The QAT source (gemma-4-E4B-it-qat-q4_0-unquantized) is q4_0 -> native 32-elem
blocks, so g32 RTN reproduces the QAT grid bit-exactly (rel_err ~0). That makes
g32 the finest *meaningful* int4 grid for this checkpoint.

LOCAL build only -- launches no HF Job.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "submissions" / "int4_g128_lmhead"))
import build_quant as bq  # noqa: E402  (quantize_weight, EMBED_TOKENS, ASSET_FILES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="QAT-unquantized source checkpoint dir")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--body-group-size", type=int, required=True)
    ap.add_argument("--head-group-size", type=int, default=128,
                    help="lm_head grid; fixed at shipped g128 for the sweep")
    ap.add_argument("--body-observer", choices=["minmax", "mse"], default="minmax",
                    help="body scale-selection observer. minmax=shipped RTN (raw "
                         "amin/amax). mse=data-free per-group clip search minimizing "
                         "int4 round-trip MSE (the only calibration-ish lever the repo "
                         "ships; true activation/math-domain cal needs GPTQ/AWQ, not "
                         "installed). head stays minmax (recipe-fixed).")
    ap.add_argument("--template-config",
                    default="/workspace/gemma_build/int4_g128_lmhead/config.json",
                    help="on-disk g128 config.json used as the quant-config template")
    ap.add_argument("--module-list",
                    default=str(REPO / "submissions/int4_g128_lmhead/official_quantized_modules.json"))
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    body_gs, head_gs = args.body_group_size, args.head_group_size

    quant_modules = set(json.load(open(args.module_list)))
    assert len(quant_modules) == 343, f"expected 343 modules, got {len(quant_modules)}"
    quant_weight_names = {m + ".weight" for m in quant_modules}

    t0 = time.time()
    print(f"[build_offline] src={src} out={out} body_gs={body_gs} head_gs={head_gs} "
          f"body_observer={args.body_observer}", flush=True)

    tensors: dict[str, torch.Tensor] = {}
    n_quant = n_copy = 0
    rel_errs: list[float] = []
    embed_weight = None

    with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
        for name in list(f.keys()):
            t = f.get_tensor(name)
            if name == bq.EMBED_TOKENS:
                embed_weight = t
            if name in quant_weight_names:
                base = name[: -len(".weight")]
                packed, scale, shape, rel = bq.quantize_weight(t, body_gs, args.body_observer)
                tensors[base + ".weight_packed"] = packed
                tensors[base + ".weight_scale"] = scale
                tensors[base + ".weight_shape"] = shape
                rel_errs.append(rel)
                n_quant += 1
                if n_quant % 100 == 0:
                    print(f"  body {n_quant}/343 (last rel_err={rel:.5f})", flush=True)
            else:
                tensors[name] = t
                n_copy += 1

    assert n_quant == 343, f"quantized {n_quant} modules, expected 343"
    assert embed_weight is not None, "embed_tokens.weight not found"

    packed, scale, shape, rel_head = bq.quantize_weight(embed_weight, head_gs, "minmax")
    tensors["lm_head.weight_packed"] = packed
    tensors["lm_head.weight_scale"] = scale
    tensors["lm_head.weight_shape"] = shape

    body_mean = sum(rel_errs) / len(rel_errs)
    body_max = max(rel_errs)
    body_min = min(rel_errs)
    print(f"[build_offline] body rel_err: min={body_min:.5f} mean={body_mean:.5f} "
          f"max={body_max:.5f} | lm_head rel_err={rel_head:.5f} (head_gs={head_gs})", flush=True)

    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    # config.json: untied embeddings + group-size-edited quant config (offline template)
    cfg = json.load(open(src / "config.json"))
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    qc = copy.deepcopy(json.load(open(args.template_config))["quantization_config"])
    qc["config_groups"]["group_0"]["weights"]["group_size"] = body_gs
    qc["config_groups"]["group_1"]["weights"]["group_size"] = head_gs
    cfg["quantization_config"] = qc
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    # tokenizer/processor assets are NOT in the qat-unquantized source; the
    # shipped g128 build copied them from the w4a16-ct checkpoint. Reuse the
    # on-disk g128 template dir (same tokenizer) so the served model is complete.
    template_dir = Path(args.template_config).parent
    for fn in bq.ASSET_FILES + ["special_tokens_map.json", "preprocessor_config.json"]:
        s = template_dir / fn
        if s.exists() and fn != "config.json":
            shutil.copy2(s, out / fn)

    # record build provenance for the report
    json.dump(
        {
            "body_group_size": body_gs, "head_group_size": head_gs,
            "body_observer": args.body_observer, "head_observer": "minmax",
            "body_rel_err_min": body_min, "body_rel_err_mean": body_mean,
            "body_rel_err_max": body_max, "lm_head_rel_err": rel_head,
            "n_quant": n_quant, "n_copy": n_copy, "src": str(src),
            "build_secs": round(time.time() - t0, 1),
        },
        open(out / "_build_meta.json", "w"), indent=2,
    )
    sz = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"[build_offline] DONE {out} ({sz:.2f} GB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
