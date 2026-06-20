#!/usr/bin/env python
"""Verify the built checkpoint: PLE=bf16 .weight, body+head still int4-packed,
config ignore/groups correct."""
import json
import sys
from pathlib import Path
from safetensors import safe_open

OUT = Path("/workspace/gemma_build/bi0_int4head_pledequant")

cfg = json.load(open(OUT / "config.json"))
qc = cfg["quantization_config"]
print("tie_word_embeddings:", cfg.get("tie_word_embeddings"),
      "| text_config.tie:", cfg["text_config"].get("tie_word_embeddings"))
print("config_groups:", list(qc["config_groups"].keys()))
print("group_1 targets:", qc["config_groups"]["group_1"]["targets"],
      "bits:", qc["config_groups"]["group_1"]["weights"]["num_bits"],
      "gs:", qc["config_groups"]["group_1"]["weights"]["group_size"])
ign = qc["ignore"]
print("PLE in ignore?:", "re:.*per_layer_input_gate" in ign)
print("lm_head in ignore?:", "lm_head" in ign, "(should be False)")
print("ignore len:", len(ign))

# tensor checks
gate_weight = gate_packed = proj_packed = head_packed = 0
gate_shapes = set()
with safe_open(str(OUT / "model.safetensors"), framework="pt", device="cpu") as f:
    keys = list(f.keys())
    for k in keys:
        if ".per_layer_input_gate." in k:
            if k.endswith(".weight"):
                gate_weight += 1
                gate_shapes.add(tuple(f.get_slice(k).get_shape()))
            elif "packed" in k or "scale" in k or "shape" in k:
                gate_packed += 1
        elif ".per_layer_projection." in k and k.endswith(".weight_packed"):
            proj_packed += 1
        elif k == "lm_head.weight_packed":
            head_packed += 1
print("-" * 50)
print(f"per_layer_input_gate bf16 .weight tensors: {gate_weight} (expect 42), shapes={gate_shapes}")
print(f"per_layer_input_gate packed/scale/shape leftovers: {gate_packed} (expect 0)")
print(f"per_layer_projection.weight_packed (sibling, stays int4): {proj_packed} (expect 42)")
print(f"lm_head.weight_packed present: {head_packed} (expect 1)")
print(f"total tensors: {len(keys)}")

ok = (cfg.get("tie_word_embeddings") is False
      and "re:.*per_layer_input_gate" in ign
      and "lm_head" not in ign
      and gate_weight == 42 and gate_packed == 0
      and proj_packed == 42 and head_packed == 1
      and gate_shapes == {(256, 2560)})
print("=" * 50)
print("BUILD VERIFY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
