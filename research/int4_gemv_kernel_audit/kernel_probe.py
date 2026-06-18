#!/usr/bin/env python
"""Deliverable 1 (offline): resolve which int4 GEMV kernel vLLM dispatches for the
shipped int4_g128_lmhead body on this A10G (sm_86), and WHY each alternative is or
is not a candidate -- without standing up a server.

For each real Gemma4-E4B linear (qkv / o_proj / gate_up / down_proj / lm_head) we
build the exact ``MPLinearLayerConfig`` (W4A16, group_size=128, symmetric uint4b8,
bf16 activations, no act-order) that ``CompressedTensorsWNA16.create_weights`` would,
then call the SAME ``choose_mp_linear_kernel`` the scheme calls. We also enumerate
every CUDA candidate in priority order and print its ``can_implement`` verdict +
reason, so the "why not" for Machete/AllSpark/Exllama/Conch is evidence, not assertion.

Run under the dev307 serve venv with the GPU visible:
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      -m research.int4_gemv_kernel_audit.kernel_probe
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import vllm.model_executor.kernels.linear as _linmod
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

_POSSIBLE_KERNELS = _linmod._POSSIBLE_KERNELS

# Gemma4-E4B text-config dims (submissions/int4_g128_lmhead/model/config.json).
HIDDEN = 2560
INTERMEDIATE = 10240
Q_OUT = 8 * 256       # num_attention_heads * head_dim
KV_OUT = 2 * 256      # num_key_value_heads * head_dim
QKV_OUT = Q_OUT + 2 * KV_OUT  # fused qkv
VOCAB = 262144

# (name, in_features, out_features) for the body GEMVs + lm_head.
LAYERS = [
    ("qkv_proj", HIDDEN, QKV_OUT),
    ("o_proj", Q_OUT, HIDDEN),
    ("gate_up_proj", HIDDEN, 2 * INTERMEDIATE),
    ("down_proj", INTERMEDIATE, HIDDEN),
    ("lm_head", HIDDEN, VOCAB),
]


def make_config(in_features: int, out_features: int) -> MPLinearLayerConfig:
    # Mirrors CompressedTensorsWNA16.create_weights for this checkpoint:
    #   num_bits=4 symmetric -> weight_type uint4b8; group_size=128; bf16 acts;
    #   actorder=None -> has_g_idx=False; symmetric -> zero_points=False.
    return MPLinearLayerConfig(
        full_weight_shape=(in_features, out_features),
        partition_weight_shape=(in_features, out_features),  # TP=1: partition == full
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )


def main() -> int:
    cc = current_platform.get_device_capability()
    cc_int = cc.to_int() if cc is not None else None
    disabled = os.environ.get("VLLM_DISABLED_KERNELS", "")
    candidates = list(_POSSIBLE_KERNELS[current_platform._enum])

    print(f"# device_capability = {cc_int} (sm_{cc_int})  VLLM_DISABLED_KERNELS={disabled!r}")
    print(f"# CUDA MP candidate priority order: {[k.__name__ for k in candidates]}\n")

    report: dict = {
        "device_capability": cc_int,
        "vllm_disabled_kernels": disabled,
        "candidate_priority": [k.__name__ for k in candidates],
        "layers": {},
    }

    for name, in_f, out_f in LAYERS:
        cfg = make_config(in_f, out_f)
        rows = []
        for k in candidates:
            min_cap = k.get_min_capability()
            cap_ok = (cc_int is None) or (min_cap <= cc_int)
            try:
                impl, reason = k.can_implement(cfg)
            except Exception as exc:  # a candidate may hard-raise; capture it
                impl, reason = False, f"can_implement raised: {exc!r}"
            rows.append({
                "kernel": k.__name__,
                "min_capability": min_cap,
                "capability_ok": cap_ok,
                "can_implement": bool(impl),
                "reason": reason,
            })
        try:
            chosen = choose_mp_linear_kernel(cfg).__name__
        except Exception as exc:
            chosen = f"<no kernel: {exc!r}>"
        report["layers"][name] = {
            "in_features": in_f,
            "out_features": out_f,
            "chosen_kernel": chosen,
            "candidates": rows,
        }
        print(f"== {name}  [in={in_f}, out={out_f}]  -> CHOSEN: {chosen}")
        for r in rows:
            tick = "OK " if r["can_implement"] else "no "
            print(f"    {tick} {r['kernel']:24s} min_cap={r['min_capability']:>3} "
                  f"cap_ok={int(r['capability_ok'])}  {r['reason'] or ''}")
        print()

    out = Path(__file__).resolve().parent / "kernel_probe_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[probe] wrote {out}")

    chosen_set = {report["layers"][n]["chosen_kernel"] for n in report["layers"]}
    print(f"[probe] distinct chosen kernels across all layers: {sorted(chosen_set)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
