#!/usr/bin/env python3
"""Recover osoi5's exact dropped-layer set + per-layer tensor inventory (PR #538).

osoi5's int4 body was carved from the STOCK int4 (google/gemma-4-E4B-it-qat-w4a16-ct)
by removing 5 transformer layers and renumbering the survivors 0..36. The per-layer
norm weights (input_layernorm / post_attention_layernorm / pre_feedforward_layernorm /
post_feedforward_layernorm) are NOT quantized in w4a16, so they are byte-exact
fingerprints: osoi5 baked layer i's norms == stock-int4 layer s_i's norms. Matching
them recovers S=[s_0..s_36] (surviving originals, in baked order) and D (the 5 dropped
original indices) with zero ambiguity.

Also inventories which layers carry k_proj/v_proj (Gemma4 KV-sharing: the last
num_kv_shared_layers reuse an earlier global layer's KV and omit their own k/v),
so the bf16 twin can be built with the exact same module set per position.
"""
import json
import re
import sys

import torch
from safetensors import safe_open

STOCK_INT4 = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors"
STOCK_BF16 = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187/model.safetensors"
OSOI5 = "/tmp/osoi5-12k-baked/model.safetensors"

LAY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
NORM_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
]


def layer_tensor_index(path):
    """Return {layer_idx: set(suffixes)} and a handle-free fingerprint loader."""
    inv = {}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
    for k in keys:
        m = LAY.match(k)
        if not m:
            continue
        li = int(m.group(1))
        inv.setdefault(li, set()).add(m.group(2))
    return inv, keys


def norm_fingerprint(path, layers):
    """{layer_idx: concatenated norm vector} for exact-equality matching."""
    fp = {}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for li in layers:
            parts = []
            ok = True
            for suf in NORM_SUFFIXES:
                key = f"model.language_model.layers.{li}.{suf}"
                if key not in keys:
                    ok = False
                    break
                parts.append(f.get_tensor(key).float().flatten())
            if ok:
                fp[li] = torch.cat(parts)
    return fp


def main():
    stock_inv, _ = layer_tensor_index(STOCK_INT4)
    osoi5_inv, _ = layer_tensor_index(OSOI5)
    bf16_inv, _ = layer_tensor_index(STOCK_BF16)

    stock_layers = sorted(stock_inv)
    osoi5_layers = sorted(osoi5_inv)
    print(f"stock-int4 language_model layers: {len(stock_layers)} ({stock_layers[0]}..{stock_layers[-1]})")
    print(f"osoi5 language_model layers:      {len(osoi5_layers)} ({osoi5_layers[0]}..{osoi5_layers[-1]})")
    print(f"stock-bf16 language_model layers: {len(sorted(bf16_inv))}")

    stock_fp = norm_fingerprint(STOCK_INT4, stock_layers)
    osoi5_fp = norm_fingerprint(OSOI5, osoi5_layers)

    # Match each osoi5 layer to the unique stock layer with identical norm vector.
    S = []  # surviving originals in baked order
    ambiguous = []
    for oi in osoi5_layers:
        ov = osoi5_fp[oi]
        matches = [si for si in stock_layers if stock_fp[si].shape == ov.shape and torch.equal(stock_fp[si], ov)]
        if len(matches) != 1:
            ambiguous.append((oi, matches))
            S.append(None)
        else:
            S.append(matches[0])

    if ambiguous:
        print("!! AMBIGUOUS/UNMATCHED osoi5 layers (oi, candidate stock idxs):")
        for oi, m in ambiguous:
            print(f"   osoi5 {oi} -> {m}")
    monotonic = all(S[i] is not None and S[i + 1] is not None and S[i] < S[i + 1] for i in range(len(S) - 1))
    dropped = sorted(set(stock_layers) - set(x for x in S if x is not None))

    print("\n=== EXACT layer map (osoi5 baked idx -> stock original idx) ===")
    print("baked:", list(range(len(S))))
    print("stock:", S)
    print(f"monotonic (order preserved): {monotonic}")
    print(f"\nDROPPED original indices ({len(dropped)}): {dropped}")

    # KV-sharing inventory: which layers carry their own k_proj/v_proj.
    def kv_layers(inv):
        has, no = [], []
        for li in sorted(inv):
            suf = inv[li]
            has_kv = any("k_proj" in s for s in suf) and any("v_proj" in s for s in suf)
            (has if has_kv else no).append(li)
        return has, no

    s_has, s_no = kv_layers(stock_inv)
    o_has, o_no = kv_layers(osoi5_inv)
    b_has, b_no = kv_layers(bf16_inv)
    print("\n=== KV-sharing (layers WITHOUT own k/v = shared) ===")
    print(f"stock-int4: own-kv={s_has}\n            shared(no-kv)={s_no}")
    print(f"stock-bf16: own-kv={b_has}\n            shared(no-kv)={b_no}")
    print(f"osoi5     : own-kv={o_has}\n            shared(no-kv)={o_no}")

    # Per-position module-set check: does osoi5 baked layer i carry the SAME
    # suffix family (mod the int4 packed/scale vs bf16 plain split) as stock s_i?
    def fam(suf):
        # collapse compressed-tensors variants to the plain module name
        out = set()
        for s in suf:
            s2 = s.replace(".weight_packed", ".weight").replace(".weight_scale", ".weight")
            s2 = s2.replace(".weight_shape", ".weight").replace(".weight_zero_point", ".weight")
            out.add(s2)
        return out

    mismatches = []
    for i, si in enumerate(S):
        if si is None:
            continue
        if fam(osoi5_inv[i]) != fam(stock_inv[si]):
            mismatches.append((i, si, fam(osoi5_inv[i]) ^ fam(stock_inv[si])))
    print(f"\nper-position module-family mismatches (osoi5 i vs stock s_i): {len(mismatches)}")
    for i, si, diff in mismatches[:10]:
        print(f"   baked {i} (stock {si}): symdiff={sorted(diff)}")

    out = {
        "dropped_original_indices": dropped,
        "surviving_originals_in_baked_order": S,
        "order_preserved": monotonic,
        "n_stock_layers": len(stock_layers),
        "n_osoi5_layers": len(osoi5_layers),
        "stock_own_kv_layers": s_has,
        "stock_shared_layers": s_no,
        "bf16_own_kv_layers": b_has,
        "bf16_shared_layers": b_no,
        "osoi5_own_kv_layers": o_has,
        "osoi5_shared_layers": o_no,
        "per_position_module_family_mismatches": [[i, si] for i, si, _ in mismatches],
        "stock_int4": STOCK_INT4,
        "stock_bf16": STOCK_BF16,
        "osoi5": OSOI5,
    }
    with open("layer_map.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote layer_map.json")
    return 0 if (not ambiguous and monotonic and len(dropped) == 5) else 1


if __name__ == "__main__":
    sys.exit(main())
