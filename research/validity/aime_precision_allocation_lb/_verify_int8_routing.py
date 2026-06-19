#!/usr/bin/env python
"""CPU-only verification that vLLM 0.22.0 resolves the build_mixed.py int8 config the way
PR #659 assumes: per-layer int8 regex -> num_bits=8 on the upgraded layers, int4 "Linear"
class -> num_bits=4 everywhere else, lm_head stays int4. Dangerous silent failure would be
4-bit dequant of 8-bit-packed data -> garbage. We exercise the REAL resolver
(CompressedTensorsConfig.get_scheme_dict) on the synthesized config, not a re-implementation.

No GPU: get_scheme_dict does should_ignore_layer + find_matched_target only (no device check).
"""
import json
import re
import sys

INT4_CFG = "/workspace/gemma_build/int4_g128_lmhead/config.json"
UPGRADE_LAYERS = [0, 1]  # exercise a 2-layer int8 regex like Phase B will use


def synth_int8_config(upgrade_layers):
    """Replicate build_mixed.py's int8 config transform exactly (group_int8 + layer regex)."""
    cfg = json.load(open(INT4_CFG))
    qc = cfg["quantization_config"]
    g0 = json.loads(json.dumps(qc["config_groups"]["group_0"]))   # Linear num_bits=4
    g1 = json.loads(json.dumps(qc["config_groups"]["group_1"]))   # re:.*lm_head num_bits=4
    alt = "|".join(str(l) for l in sorted(upgrade_layers))
    layer_regex = rf"re:.*language_model\.model\.layers\.({alt})\."
    g8w = json.loads(json.dumps(g0["weights"]))
    g8w["num_bits"] = 8
    qc["config_groups"] = {
        "group_int8": {"targets": [layer_regex], "weights": g8w},
        "group_0": g0,
        "group_1": g1,
    }
    qc["version"] = "0.15.0.1"
    qc["quantization_status"] = "compressed"
    qc["format"] = "pack-quantized"
    return qc, layer_regex


def main():
    qc, layer_regex = synth_int8_config(UPGRADE_LAYERS)
    print(f"[cfg] int8 layer regex: {layer_regex}")
    print(f"[cfg] groups: {list(qc['config_groups'])}")

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    ct = CompressedTensorsConfig.from_config(qc)
    print(f"[cfg] target_scheme_map keys: {list(ct.target_scheme_map.keys())}")
    print(f"[cfg] ignore entries: {len(ct.ignore)} (language? "
          f"{any('language_model' in x for x in ct.ignore)})")

    # vLLM running fused module names (hf_to_vllm_mapper: model.language_model -> language_model.model,
    # q/k/v -> qkv_proj, gate/up -> gate_up_proj).
    cases = [
        # (layer_name, dummy_class_name, expected_num_bits, why)
        ("language_model.model.layers.0.self_attn.qkv_proj", "QKVParallelLinear", 8, "upgraded L0 attn (fused)"),
        ("language_model.model.layers.0.self_attn.o_proj",   "RowParallelLinear", 8, "upgraded L0 o_proj"),
        ("language_model.model.layers.0.mlp.gate_up_proj",   "MergedColumnParallelLinear", 8, "upgraded L0 mlp (fused)"),
        ("language_model.model.layers.0.mlp.down_proj",      "RowParallelLinear", 8, "upgraded L0 down_proj"),
        ("language_model.model.layers.1.self_attn.qkv_proj", "QKVParallelLinear", 8, "upgraded L1 attn"),
        ("language_model.model.layers.2.self_attn.qkv_proj", "QKVParallelLinear", 4, "NON-upgraded L2 -> int4 Linear"),
        ("language_model.model.layers.5.mlp.down_proj",      "RowParallelLinear", 4, "NON-upgraded L5 -> int4 Linear"),
        ("language_model.model.layers.41.self_attn.qkv_proj","QKVParallelLinear", 4, "NON-upgraded L41 -> int4 Linear"),
        ("lm_head",                                          "ParallelLMHead",    4, "lm_head LOCKED int4 group_1"),
    ]

    def dummy(cls_name):
        return type(cls_name, (), {})()

    ok = True
    for layer_name, cls_name, want_bits, why in cases:
        mod = dummy(cls_name)
        sd = ct.get_scheme_dict(mod, layer_name)
        if sd is None:
            got = "IGNORE(unquantized/bf16)"
            passed = (want_bits is None)
        else:
            wq = sd.get("weights")
            got_bits = getattr(wq, "num_bits", None)
            got = f"num_bits={got_bits}"
            passed = (got_bits == want_bits)
        ok = ok and passed
        flag = "OK " if passed else "**FAIL**"
        print(f"[{flag}] {layer_name:55s} cls={cls_name:28s} -> {got:24s} want={want_bits}  ({why})")

    # Negative control: confirm the regex actually distinguishes L0/L1 from L2 at the string level.
    pat = layer_regex[3:]
    assert re.match(pat, "language_model.model.layers.0.self_attn.qkv_proj")
    assert re.match(pat, "language_model.model.layers.1.mlp.down_proj")
    assert not re.match(pat, "language_model.model.layers.2.self_attn.qkv_proj")
    assert not re.match(pat, "language_model.model.layers.10.self_attn.qkv_proj"), \
        "regex must not let L10 leak into the L0|L1 alternation"
    print("[regex] L0/L1 match, L2/L10 correctly excluded (no alternation leakage)")

    print("\nRESULT:", "INT8_ROUTING_VERIFIED" if ok else "INT8_ROUTING_BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
