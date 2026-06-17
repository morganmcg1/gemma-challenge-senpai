#!/usr/bin/env python3
"""Carve an arbitrary-depth bf16 body from stock gemma-4-E4B-it (PR #546).

Generalizes #538's build_37L_bf16.py to drop an ARBITRARY set of layers from the
stock bf16 (42L), preserving the stock model's own KV-sharing semantics exactly.
This fills the depth<->gate Pareto interior (drop in {2,3}) between #538's two
measured endpoints (42L clears the gate, 37L=drop-5 fails it).

WHY RENUMBERING IS SAFE (the correctness crux)
-----------------------------------------------
Stock bf16: 42 layers, num_kv_shared_layers=18 -> own-KV layers are originals
0..23 (a contiguous PREFIX), KV-shared layers are originals 24..41 (a contiguous
SUFFIX). In vLLM's Gemma4Attention.__init__ each shared layer reuses the KV of
the LAST non-shared layer of the SAME attention type, and that source index is
RECOMPUTED from the carved config's `layer_types` + `num_kv_shared_layers` at
load time (gemma4.py: `kv_shared_layer_index = len(prev_layers)-1 -
prev_layers[::-1].index(current_layer_type)`). So renumbering survivors is safe
as long as the own-KV prefix still contains >=1 layer of every attention type
present in the shared suffix.

We further RESTRICT drops to the KV-shared suffix (original idx >= 24). Then:
 - the own-KV prefix [0..23] is byte-identical to stock,
 - every surviving layer keeps its exact stock weights AND its exact KV source
   (the last sliding/full own-KV layer, 22/23), so the carve is a PURE depth
   ablation: remove k transformer blocks, nothing else moves.
 - new num_kv_shared_layers = (#surviving layers with original idx >= 24).

PER-LAYER TENSOR INVENTORY (verified against stock + osoi5)
-----------------------------------------------------------
Stock stores {q,k,v}_proj + {q,k}_norm + o_proj on every layer. vLLM fuses q/k/v
into one qkv_proj; for a shared layer it computes k,v then DISCARDS them (reads
KV from the target's cache). So on surviving SHARED positions we drop
k_proj/v_proj (their qkv_proj k/v shards stay uninitialized but are never used)
and KEEP k_norm (vLLM instantiates k_norm per-layer; forward skips it on shared
layers, so its value is irrelevant -- keeping the stock weight just avoids a
missing-weight warning). This is exactly #538's verified DROP_ON_SHARED recipe
and is forward-identical to osoi5's real served body.

LAYER-COUNT-DEPENDENT MODEL TENSORS
-----------------------------------
Two model-level tensors are sliced to the surviving 256-wide per-layer blocks:
  embed_tokens_per_layer.weight     [V, 42*256] -> [V, n_new*256]
  per_layer_model_projection.weight [42*256, H] -> [n_new*256, H]
The 256-col block layout is PROVEN by the osoi5 oracle (stock-int4's per-layer
embed sliced to osoi5's own surviving originals == osoi5's, exact) -- a standing
proof independent of OUR drop set.

Usage:
  build_dropN_bf16.py --drop 36,37 --out /tmp/gemma40L-bf16
  build_dropN_bf16.py --drop 35,36,37 --out /tmp/gemma39L-bf16
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
STOCK_BF16_DIR = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187"
STOCK_BF16 = os.path.join(STOCK_BF16_DIR, "model.safetensors")
STOCK_INT4 = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors"
OSOI5_DIR = "/tmp/osoi5-12k-baked"
OSOI5 = os.path.join(OSOI5_DIR, "model.safetensors")
# osoi5's surviving originals (from #538 layer_map.json) -- used ONLY as the
# block-layout oracle, independent of our drop set.
OSOI5_LAYER_MAP = os.path.join(HERE, "..", "body_decomp_served_2x2", "layer_map.json")

N_STOCK = 42
PLI = 256  # hidden_size_per_layer_input

DROP_ON_SHARED = {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}
LAY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
PLMP_KEY = "model.language_model.per_layer_model_projection.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", required=True, help="comma-separated original layer indices to drop, e.g. 36,37")
    ap.add_argument("--out", required=True, help="output model dir")
    args = ap.parse_args()

    dropped = sorted(int(x) for x in args.drop.split(",") if x.strip() != "")
    assert all(0 <= d < N_STOCK for d in dropped), f"drop idx out of range: {dropped}"
    assert len(set(dropped)) == len(dropped), "duplicate drop idx"

    stock_cfg = json.load(open(os.path.join(STOCK_BF16_DIR, "config.json")))
    stock_tc = stock_cfg["text_config"]
    assert stock_tc["num_hidden_layers"] == N_STOCK, stock_tc["num_hidden_layers"]
    n_kv_shared_stock = stock_tc["num_kv_shared_layers"]
    own_kv_boundary = N_STOCK - n_kv_shared_stock  # 42-18 = 24; originals 0..23 own-KV, 24..41 shared
    stock_layer_types = stock_tc["layer_types"]
    assert len(stock_layer_types) == N_STOCK

    # Clean-carve constraint: drops must lie in the KV-shared suffix so the
    # own-KV prefix [0..own_kv_boundary) is byte-identical to stock and every
    # survivor keeps its exact KV source. (Prefix drops would still be handled
    # correctly by the general KV recompute below, but would no longer be a pure
    # depth ablation of the 42L endpoint -- this PR sweeps tail-only drops.)
    assert all(d >= own_kv_boundary for d in dropped), (
        f"this carve requires tail-only (KV-shared) drops >= {own_kv_boundary}; got {dropped}"
    )

    S = [i for i in range(N_STOCK) if i not in dropped]  # survivors, order preserved
    n_new = len(S)
    survivors = set(S)
    new_own_kv = sum(1 for s in S if s < own_kv_boundary)
    new_shared = n_new - new_own_kv
    first_kv_shared_idx = new_own_kv  # new positions >= this are shared

    # Contiguity: own-KV survivors form a prefix, shared survivors a suffix.
    for i, s in enumerate(S):
        assert (s >= own_kv_boundary) == (i >= first_kv_shared_idx), (
            f"prefix/suffix broken at new pos {i} (orig {s})"
        )
    new_layer_types = [stock_layer_types[s] for s in S]

    # KV-source existence: every attention type in the shared suffix must have a
    # same-type layer in the non-shared prefix (else vLLM finds no KV source).
    prefix_types = set(new_layer_types[:first_kv_shared_idx])
    suffix_types = set(new_layer_types[first_kv_shared_idx:])
    assert suffix_types <= prefix_types, (
        f"shared-suffix types {suffix_types} not all covered by prefix types {prefix_types}"
    )

    print(f"dropped={dropped}  n_new={n_new}  new_own_kv={new_own_kv}  "
          f"new_kv_shared={new_shared}  first_kv_shared_idx={first_kv_shared_idx}")
    print(f"shared new positions (drop own k/v): {list(range(first_kv_shared_idx, n_new))}")

    os.makedirs(args.out, exist_ok=True)
    new_sd = {}

    with safe_open(STOCK_BF16, framework="pt") as f:
        keys = list(f.keys())
        per_orig = {}
        for k in keys:
            m = LAY.match(k)
            if m and int(m.group(1)) in survivors:
                per_orig.setdefault(int(m.group(1)), []).append((m.group(2), k))

        # 1) non-layer tensors verbatim (except the two sliced ones)
        n_nonlayer = 0
        for k in keys:
            if LAY.match(k) or k in (PLE_KEY, PLMP_KEY):
                continue
            new_sd[k] = f.get_tensor(k).contiguous()
            n_nonlayer += 1
        assert "lm_head.weight" not in new_sd, "stock bf16 unexpectedly has untied lm_head"

        # 2) renumbered decoder layers; drop own k/v on shared positions
        n_dropped_kv = 0
        for i, s_i in enumerate(S):
            shared = i >= first_kv_shared_idx
            for suf, key in per_orig[s_i]:
                if shared and suf in DROP_ON_SHARED:
                    n_dropped_kv += 1
                    continue
                new_sd[f"model.language_model.layers.{i}.{suf}"] = f.get_tensor(key).contiguous()

        # 3) layer-count-dependent model tensors -> slice to surviving blocks
        ple = f.get_tensor(PLE_KEY)
        plmp = f.get_tensor(PLMP_KEY)
    assert ple.shape[1] == N_STOCK * PLI, ple.shape
    assert plmp.shape[0] == N_STOCK * PLI, plmp.shape
    ple_new = torch.cat([ple[:, s * PLI:(s + 1) * PLI] for s in S], dim=1).contiguous()
    plmp_new = torch.cat([plmp[s * PLI:(s + 1) * PLI, :] for s in S], dim=0).contiguous()
    assert ple_new.shape == (ple.shape[0], n_new * PLI)
    assert plmp_new.shape == (n_new * PLI, plmp.shape[1])
    new_sd[PLE_KEY] = ple_new
    new_sd[PLMP_KEY] = plmp_new

    # 4) block-layout oracle (independent of OUR drop set): stock-int4's per-layer
    # embed sliced to osoi5's OWN surviving originals must EQUAL osoi5's, exact.
    lm = json.load(open(OSOI5_LAYER_MAP))
    osoi5_S = lm["surviving_originals_in_baked_order"]
    with safe_open(OSOI5, framework="pt") as g:
        osoi5_ple = g.get_tensor(PLE_KEY)
    with safe_open(STOCK_INT4, framework="pt") as g4:
        i4_ple = g4.get_tensor(PLE_KEY)
    i4_slice = torch.cat([i4_ple[:, s * PLI:(s + 1) * PLI] for s in osoi5_S], dim=1)
    layout_ok = bool(i4_slice.shape == osoi5_ple.shape and torch.equal(i4_slice.float(), osoi5_ple.float()))
    print(f"per-layer block-layout oracle (stock-int4 slice @osoi5 survivors == osoi5, exact): {layout_ok}")

    dtypes = {str(v.dtype) for v in new_sd.values()}
    print(f"built state_dict: {len(new_sd)} tensors | nonlayer={n_nonlayer} | "
          f"dropped own-k/v on shared={n_dropped_kv} (expect {2*new_shared}) | dtypes={dtypes}")
    assert n_dropped_kv == 2 * new_shared, (n_dropped_kv, new_shared)

    # config: stock bf16 (full multimodal, tied full head, no quant) with the
    # carved depth geometry spliced in.
    cfg = json.load(open(os.path.join(STOCK_BF16_DIR, "config.json")))
    cfg["text_config"]["num_hidden_layers"] = n_new
    cfg["text_config"]["num_kv_shared_layers"] = new_shared
    cfg["text_config"]["layer_types"] = new_layer_types
    assert cfg.get("tie_word_embeddings") is True
    assert "quantization_config" not in cfg
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    # aux files verbatim
    for fn in os.listdir(STOCK_BF16_DIR):
        if fn in ("model.safetensors", "config.json"):
            continue
        src = os.path.join(STOCK_BF16_DIR, fn)
        if os.path.isfile(src) or os.path.islink(src):
            shutil.copy(src, os.path.join(args.out, fn))

    save_file(new_sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    sz = os.path.getsize(os.path.join(args.out, "model.safetensors")) / 1e9
    print(f"wrote {args.out}/model.safetensors ({sz:.2f} GB) + config.json + aux files")

    meta = {
        "out_dir": args.out,
        "source_bf16": STOCK_BF16_DIR,
        "dropped_original_indices": dropped,
        "surviving_originals_in_baked_order": S,
        "num_hidden_layers": n_new,
        "num_kv_shared_layers": new_shared,
        "first_kv_shared_idx": first_kv_shared_idx,
        "own_kv_boundary_stock": own_kv_boundary,
        "shared_new_positions": list(range(first_kv_shared_idx, n_new)),
        "tie_word_embeddings": True,
        "head": "full-262144 (tied to embeddings, bf16)",
        "new_layer_types": new_layer_types,
        "tail_only_drop": True,
        "own_kv_prefix_byte_identical_to_stock": True,
        "perlayer_block_layout_oracle_ok": layout_ok,
        "n_tensors": len(new_sd),
        "n_dropped_own_kv_tensors": n_dropped_kv,
    }
    json.dump(meta, open(os.path.join(HERE, f"build_drop{len(dropped)}_meta.json"), "w"), indent=2)
    print(f"wrote build_drop{len(dropped)}_meta.json")
    return 0 if layout_ok else 1


if __name__ == "__main__":
    sys.exit(main())
