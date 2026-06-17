#!/usr/bin/env python3
"""Carve an arbitrary-depth GENUINE int4 body from stock gemma-4-E4B-it-qat-w4a16
(PR #546 follow-up confirm).

This is the int4 twin of build_dropN_bf16.py. It confirms whether the bf16 gate
winner (the largest tail-only drop-count that clears Morgan #515) STILL clears at
the deployed int4 precision -- using a from-scratch int4 carve of the stock QAT
checkpoint, NOT the #527 synthesized full-head proxy.

WHY THIS IS A GENUINE int4 CARVE (head + body both real int4-QAT weights)
-------------------------------------------------------------------------
Stock QAT (compressed-tensors, pack-quantized, w4a16) already ships:
  * decoder weights int4-packed (weight_packed I32 + weight_scale BF16 +
    weight_shape I64) per linear,
  * KV-sharing ALREADY baked in: shared originals 24..41 carry NO
    k_proj/v_proj/k_norm (only q_proj/q_norm/o_proj + mlp + norms),
  * tie_word_embeddings=True with lm_head in the quant ignore list, so the head
    is the model's real tied embedding head (bf16 lm_head.weight present) -- the
    actual deployed head, not a synthesized one.
So carving = renumber surviving layers VERBATIM (copy every quantized tensor of
each survivor unchanged) + slice the two layer-count-dependent model tensors.
Nothing is requantized or synthesized; every weight is a stock QAT weight.

KV-SHARE RENUMBER SAFETY: identical argument to the bf16 carve. Tail-only drops
(orig >= own_kv_boundary=24) keep the own-KV prefix [0..23] byte-identical, so
every surviving shared layer keeps its exact KV source. new num_kv_shared_layers
= count of survivors with orig >= 24.

LAYER-COUNT-DEPENDENT MODEL TENSORS (int4 specifics)
----------------------------------------------------
  embed_tokens_per_layer.weight  BF16 [V, 42*256]  -> slice COLUMNS to survivors
  per_layer_model_projection: int4-packed, rows are layer-major 256-blocks:
     weight_packed I32  [42*256, 320]  -> slice ROWS to survivors
     weight_scale  BF16 [42*256,  80]  -> slice ROWS to survivors
     weight_shape  I64  [2] = [42*256, 2560] -> set [n_new*256, 2560]

TWO ORACLES
-----------
1) (standing, drop-set independent) per-layer block layout: stock-int4 PLE sliced
   to osoi5's OWN survivors == osoi5's PLE, exact.
2) (int4 verbatim-copy proof) for several osoi5 survivor positions j (original
   S_osoi5[j]), osoi5.layers.j.<quant tensor> == stock-int4.layers.<orig>.<same>,
   exact -- proves stock QAT layer tensors are copied verbatim under renumber,
   exactly what THIS carve does.

Usage:
  build_dropN_int4.py --drop 36,37     --out /tmp/gemma40L-int4
  build_dropN_int4.py --drop 35,36,37  --out /tmp/gemma39L-int4
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK_INT4_DIR = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
STOCK_INT4 = os.path.join(STOCK_INT4_DIR, "model.safetensors")
OSOI5_DIR = "/tmp/osoi5-12k-baked"
OSOI5 = os.path.join(OSOI5_DIR, "model.safetensors")
OSOI5_LAYER_MAP = os.path.join(HERE, "..", "body_decomp_served_2x2", "layer_map.json")

N_STOCK = 42
PLI = 256  # hidden_size_per_layer_input

LAY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
PLMP_PACKED = "model.language_model.per_layer_model_projection.weight_packed"
PLMP_SCALE = "model.language_model.per_layer_model_projection.weight_scale"
PLMP_SHAPE = "model.language_model.per_layer_model_projection.weight_shape"
PLMP_KEYS = {PLMP_PACKED, PLMP_SCALE, PLMP_SHAPE}
# stock int4 shared layers carry NONE of these (KV-sharing baked in); used only
# as a sanity assert, never to drop (we copy survivors verbatim).
KV_ABSENT_ON_SHARED = {
    "self_attn.k_proj.weight_packed", "self_attn.k_proj.weight_scale",
    "self_attn.k_proj.weight_shape", "self_attn.v_proj.weight_packed",
    "self_attn.v_proj.weight_scale", "self_attn.v_proj.weight_shape",
    "self_attn.k_norm.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", required=True, help="comma-separated original layer indices to drop, e.g. 36,37")
    ap.add_argument("--out", required=True, help="output model dir")
    args = ap.parse_args()

    dropped = sorted(int(x) for x in args.drop.split(",") if x.strip() != "")
    assert all(0 <= d < N_STOCK for d in dropped), f"drop idx out of range: {dropped}"
    assert len(set(dropped)) == len(dropped), "duplicate drop idx"

    stock_cfg = json.load(open(os.path.join(STOCK_INT4_DIR, "config.json")))
    stock_tc = stock_cfg["text_config"]
    assert stock_tc["num_hidden_layers"] == N_STOCK, stock_tc["num_hidden_layers"]
    n_kv_shared_stock = stock_tc["num_kv_shared_layers"]
    own_kv_boundary = N_STOCK - n_kv_shared_stock  # 42-18=24
    stock_layer_types = stock_tc["layer_types"]
    assert len(stock_layer_types) == N_STOCK
    assert stock_cfg.get("tie_word_embeddings") is True
    assert "quantization_config" in stock_cfg, "stock int4 must carry quantization_config"

    assert all(d >= own_kv_boundary for d in dropped), (
        f"this carve requires tail-only (KV-shared) drops >= {own_kv_boundary}; got {dropped}"
    )

    S = [i for i in range(N_STOCK) if i not in dropped]
    n_new = len(S)
    survivors = set(S)
    new_own_kv = sum(1 for s in S if s < own_kv_boundary)
    new_shared = n_new - new_own_kv
    first_kv_shared_idx = new_own_kv
    for i, s in enumerate(S):
        assert (s >= own_kv_boundary) == (i >= first_kv_shared_idx), f"prefix/suffix broken at {i}({s})"
    new_layer_types = [stock_layer_types[s] for s in S]
    prefix_types = set(new_layer_types[:first_kv_shared_idx])
    suffix_types = set(new_layer_types[first_kv_shared_idx:])
    assert suffix_types <= prefix_types, f"shared types {suffix_types} not covered by prefix {prefix_types}"

    print(f"dropped={dropped}  n_new={n_new}  new_own_kv={new_own_kv}  "
          f"new_kv_shared={new_shared}  first_kv_shared_idx={first_kv_shared_idx}")

    os.makedirs(args.out, exist_ok=True)
    new_sd = {}

    with safe_open(STOCK_INT4, framework="pt") as f:
        keys = list(f.keys())
        per_orig = {}
        for k in keys:
            m = LAY.match(k)
            if m and int(m.group(1)) in survivors:
                per_orig.setdefault(int(m.group(1)), []).append((m.group(2), k))

        # 1) non-layer tensors verbatim (except PLE + the 3 PLMP tensors)
        n_nonlayer = 0
        for k in keys:
            if LAY.match(k) or k == PLE_KEY or k in PLMP_KEYS:
                continue
            new_sd[k] = f.get_tensor(k).contiguous()
            n_nonlayer += 1
        assert "lm_head.weight" in new_sd, "stock int4 must carry tied lm_head.weight (full int4-model head)"

        # 2) renumbered decoder layers, copied VERBATIM (KV-sharing already baked).
        n_shared_seen = 0
        for i, s_i in enumerate(S):
            shared = i >= first_kv_shared_idx
            sufs = {suf for suf, _ in per_orig[s_i]}
            if shared:
                n_shared_seen += 1
                # sanity: stock already omits k/v/k_norm on shared originals
                assert not (sufs & KV_ABSENT_ON_SHARED), (
                    f"orig {s_i} (shared) unexpectedly carries {sufs & KV_ABSENT_ON_SHARED}")
            else:
                assert "self_attn.k_proj.weight_packed" in sufs, f"orig {s_i} (own-KV) missing k_proj"
            for suf, key in per_orig[s_i]:
                new_sd[f"model.language_model.layers.{i}.{suf}"] = f.get_tensor(key).contiguous()
        assert n_shared_seen == new_shared, (n_shared_seen, new_shared)

        # 3) layer-count-dependent model tensors
        ple = f.get_tensor(PLE_KEY)            # BF16 [V, 42*256]  -> slice cols
        plmp_packed = f.get_tensor(PLMP_PACKED)  # I32  [42*256, 320] -> slice rows
        plmp_scale = f.get_tensor(PLMP_SCALE)    # BF16 [42*256, 80]  -> slice rows
        plmp_shape = f.get_tensor(PLMP_SHAPE)    # I64  [2] = [42*256, 2560]
    assert ple.shape[1] == N_STOCK * PLI, ple.shape
    assert plmp_packed.shape[0] == N_STOCK * PLI, plmp_packed.shape
    assert plmp_scale.shape[0] == N_STOCK * PLI, plmp_scale.shape
    assert int(plmp_shape[0].item()) == N_STOCK * PLI, plmp_shape

    ple_new = torch.cat([ple[:, s * PLI:(s + 1) * PLI] for s in S], dim=1).contiguous()
    plmp_packed_new = torch.cat([plmp_packed[s * PLI:(s + 1) * PLI, :] for s in S], dim=0).contiguous()
    plmp_scale_new = torch.cat([plmp_scale[s * PLI:(s + 1) * PLI, :] for s in S], dim=0).contiguous()
    plmp_shape_new = torch.tensor([n_new * PLI, int(plmp_shape[1].item())], dtype=plmp_shape.dtype)
    assert ple_new.shape == (ple.shape[0], n_new * PLI)
    assert plmp_packed_new.shape == (n_new * PLI, plmp_packed.shape[1])
    assert plmp_scale_new.shape == (n_new * PLI, plmp_scale.shape[1])
    new_sd[PLE_KEY] = ple_new
    new_sd[PLMP_PACKED] = plmp_packed_new
    new_sd[PLMP_SCALE] = plmp_scale_new
    new_sd[PLMP_SHAPE] = plmp_shape_new

    # 4a) standing block-layout oracle (PLE is bf16 in both, layout proven by osoi5)
    lm = json.load(open(OSOI5_LAYER_MAP))
    osoi5_S = lm["surviving_originals_in_baked_order"]
    with safe_open(OSOI5, framework="pt") as g:
        osoi5_ple = g.get_tensor(PLE_KEY)
        osoi5_keys = set(g.keys())
        with safe_open(STOCK_INT4, framework="pt") as g4:
            i4_ple = g4.get_tensor(PLE_KEY)
            i4_slice = torch.cat([i4_ple[:, s * PLI:(s + 1) * PLI] for s in osoi5_S], dim=1)
            layout_ok = bool(i4_slice.shape == osoi5_ple.shape and torch.equal(i4_slice.float(), osoi5_ple.float()))

            # 4b) int4 verbatim-copy proof: a few osoi5 survivor positions must
            # equal stock-int4's same original layer, exact (quantized tensors).
            probe_suffixes = [
                "self_attn.q_proj.weight_packed", "self_attn.q_proj.weight_scale",
                "mlp.down_proj.weight_packed", "input_layernorm.weight",
            ]
            verbatim_ok = True
            n_probe = 0
            for j, orig in enumerate(osoi5_S):
                if j % 9 != 0:  # sample ~every 9th to keep it cheap
                    continue
                for suf in probe_suffixes:
                    ok_key = f"model.language_model.layers.{j}.{suf}"
                    i4_key = f"model.language_model.layers.{orig}.{suf}"
                    if ok_key in osoi5_keys and i4_key in keys:
                        a = g.get_tensor(ok_key)
                        b = g4.get_tensor(i4_key)
                        eq = a.shape == b.shape and torch.equal(a, b)
                        verbatim_ok = verbatim_ok and eq
                        n_probe += 1
    print(f"oracle-1 PLE block layout (stock-int4 @osoi5 survivors == osoi5, exact): {layout_ok}")
    print(f"oracle-2 int4 verbatim-copy proof (osoi5 layer tensors == stock-int4 originals, {n_probe} probes): {verbatim_ok}")

    dtypes = {str(v.dtype) for v in new_sd.values()}
    print(f"built state_dict: {len(new_sd)} tensors | nonlayer={n_nonlayer} | dtypes={sorted(dtypes)}")

    # config: stock int4 (keeps quantization_config + tie) with carved geometry
    cfg = json.load(open(os.path.join(STOCK_INT4_DIR, "config.json")))
    cfg["text_config"]["num_hidden_layers"] = n_new
    cfg["text_config"]["num_kv_shared_layers"] = new_shared
    cfg["text_config"]["layer_types"] = new_layer_types
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    for fn in os.listdir(STOCK_INT4_DIR):
        if fn in ("model.safetensors", "config.json"):
            continue
        src = os.path.join(STOCK_INT4_DIR, fn)
        if os.path.isfile(src) or os.path.islink(src):
            shutil.copy(src, os.path.join(args.out, fn))

    save_file(new_sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    sz = os.path.getsize(os.path.join(args.out, "model.safetensors")) / 1e9
    print(f"wrote {args.out}/model.safetensors ({sz:.2f} GB) + config.json + aux files")

    meta = {
        "out_dir": args.out,
        "source_int4": STOCK_INT4_DIR,
        "dropped_original_indices": dropped,
        "surviving_originals_in_baked_order": S,
        "num_hidden_layers": n_new,
        "num_kv_shared_layers": new_shared,
        "first_kv_shared_idx": first_kv_shared_idx,
        "own_kv_boundary_stock": own_kv_boundary,
        "shared_new_positions": list(range(first_kv_shared_idx, n_new)),
        "tie_word_embeddings": True,
        "head": "full int4-model tied head (bf16 lm_head.weight from stock QAT, ignore-listed)",
        "new_layer_types": new_layer_types,
        "tail_only_drop": True,
        "own_kv_prefix_byte_identical_to_stock": True,
        "perlayer_block_layout_oracle_ok": layout_ok,
        "int4_verbatim_copy_oracle_ok": verbatim_ok,
        "int4_verbatim_probes": n_probe,
        "n_tensors": len(new_sd),
        "genuine_int4_carve_not_527_proxy": True,
    }
    json.dump(meta, open(os.path.join(HERE, f"build_drop{len(dropped)}_int4_meta.json"), "w"), indent=2)
    print(f"wrote build_drop{len(dropped)}_int4_meta.json")
    return 0 if (layout_ok and verbatim_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
