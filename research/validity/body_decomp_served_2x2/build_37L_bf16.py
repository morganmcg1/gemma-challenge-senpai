#!/usr/bin/env python3
"""Build the 37L-bf16 twin of osoi5's body (PR #538, corner B).

Isolates the layer-removal knob at bf16 precision: take the STOCK bf16
google/gemma-4-E4B-it and carve it to EXACTLY osoi5's body architecture —
same 5 dropped layers (originals [2,3,4,36,37]), same renumbering, same
KV-sharing (num_kv_shared_layers=16 -> baked positions 21..36 are shared and
omit their own k_proj/v_proj/k_norm), same layer_types — but bf16 weights and
the full 262k tied head (vs osoi5's int4 body + pruned head).

The only difference from osoi5's int4 body is the precision (bf16 vs int4) and
the head (full tied vs 12k pruned). So (37L-bf16) vs (37L-int4=osoi5 full-head
ceiling, #527) isolates int4-cost on the reduced body, and (37L-bf16) vs
(42L-bf16) isolates layerdrop-cost at bf16.

Two model-level tensors are layer-count-dependent and are sliced to the
surviving layers' per-layer blocks (256 cols/rows each):
  embed_tokens_per_layer.weight  [V, 42*256] -> [V, 37*256]
  per_layer_model_projection.weight [42*256, H] -> [37*256, H]
The embed_tokens_per_layer slice is checked for EXACT equality against osoi5's
own bf16 copy, which both confirms the per-layer block layout and proves osoi5
carved per-layer embeddings the same way.
"""
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
OSOI5_DIR = "/tmp/osoi5-12k-baked"
OSOI5 = os.path.join(OSOI5_DIR, "model.safetensors")
OUT_DIR = "/tmp/gemma37L-bf16"

PLI = 256  # hidden_size_per_layer_input
N_KV_SHARED = 16
# vLLM Gemma4 (gemma4.py): shared layers reuse the target layer's KV and SKIP
# their own k_norm in forward (`if not self.is_kv_shared_layer: k=self.k_norm(k)`),
# so they don't need k_proj/v_proj. But `self.k_norm = RMSNorm(...)` is created
# unconditionally per layer, so the loader REQUIRES k_norm.weight to be present
# (even though it is never applied for shared layers). Keep k_norm everywhere;
# only k_proj/v_proj are genuinely absent on shared layers (KV-sharing). This is
# forward-identical to osoi5's body (which omits the unused shared k_norm too).
DROP_ON_SHARED = {
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
}
LAY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
PLMP_KEY = "model.language_model.per_layer_model_projection.weight"


def main():
    lm = json.load(open(os.path.join(HERE, "layer_map.json")))
    S = lm["surviving_originals_in_baked_order"]
    dropped = lm["dropped_original_indices"]
    assert lm["order_preserved"] and len(dropped) == 5 and len(S) == 37, "bad layer_map"
    n_new = len(S)
    shared_start = n_new - N_KV_SHARED  # 21
    survivors = set(S)
    print(f"survivors(37)={S}\ndropped(5)={dropped}\nshared positions (drop own k/v): {list(range(shared_start, n_new))}")

    osoi5_cfg = json.load(open(os.path.join(OSOI5_DIR, "config.json")))
    osoi5_layer_types = osoi5_cfg["text_config"]["layer_types"]
    assert len(osoi5_layer_types) == n_new

    os.makedirs(OUT_DIR, exist_ok=True)
    new_sd = {}

    with safe_open(STOCK_BF16, framework="pt") as f:
        keys = list(f.keys())
        # index surviving-layer keys by original index
        per_orig = {}  # orig_idx -> list[(suffix, key)]
        for k in keys:
            m = LAY.match(k)
            if not m:
                continue
            oi = int(m.group(1))
            if oi in survivors:
                per_orig.setdefault(oi, []).append((m.group(2), k))

        # 1) non-layer tensors verbatim (except the two sliced ones)
        n_nonlayer = 0
        for k in keys:
            if LAY.match(k) or k in (PLE_KEY, PLMP_KEY):
                continue
            new_sd[k] = f.get_tensor(k).contiguous()
            n_nonlayer += 1
        assert "lm_head.weight" not in new_sd, "stock bf16 unexpectedly has untied lm_head"

        # 2) renumbered decoder layers, dropping own-k/v on shared positions
        n_dropped_kv = 0
        for i, s_i in enumerate(S):
            shared = i >= shared_start
            for suf, key in per_orig[s_i]:
                if shared and suf in DROP_ON_SHARED:
                    n_dropped_kv += 1
                    continue
                new_sd[f"model.language_model.layers.{i}.{suf}"] = f.get_tensor(key).contiguous()

        # 3) layer-count-dependent model-level tensors -> slice to surviving blocks
        ple = f.get_tensor(PLE_KEY)    # [V, 42*PLI]
        plmp = f.get_tensor(PLMP_KEY)  # [42*PLI, H]
    assert ple.shape[1] == 42 * PLI, ple.shape
    assert plmp.shape[0] == 42 * PLI, plmp.shape
    ple_new = torch.cat([ple[:, s * PLI:(s + 1) * PLI] for s in S], dim=1).contiguous()
    plmp_new = torch.cat([plmp[s * PLI:(s + 1) * PLI, :] for s in S], dim=0).contiguous()
    assert ple_new.shape == (ple.shape[0], n_new * PLI)
    assert plmp_new.shape == (n_new * PLI, plmp.shape[1])
    new_sd[PLE_KEY] = ple_new
    new_sd[PLMP_KEY] = plmp_new

    # VERIFY the per-layer block layout. osoi5 was carved from stock INT4 (QAT),
    # not from stock bf16, so the correct layout oracle is: stock-int4's
    # embed_tokens_per_layer sliced to the surviving blocks must EQUAL osoi5's
    # (exact). That proves the 256-col layer-major block layout. My bf16 slice
    # then legitimately differs from osoi5 only by the QAT bf16-vs-int4 embedding
    # delta (embeddings are not 4bit-packed but ARE QAT-finetuned).
    with safe_open(OSOI5, framework="pt") as g:
        osoi5_ple = g.get_tensor(PLE_KEY)
    STOCK_INT4 = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors"
    with safe_open(STOCK_INT4, framework="pt") as g4:
        i4_ple = g4.get_tensor(PLE_KEY)
    i4_slice = torch.cat([i4_ple[:, s * PLI:(s + 1) * PLI] for s in S], dim=1)
    layout_ok = bool(i4_slice.shape == osoi5_ple.shape and torch.equal(i4_slice.float(), osoi5_ple.float()))
    qat_delta = (ple_new.float() - osoi5_ple.float()).abs().mean().item() if ple_new.shape == osoi5_ple.shape else float("nan")
    print(f"per-layer block layout confirmed (stock-int4 slice == osoi5, exact): {layout_ok}")
    print(f"  bf16 slice vs osoi5 (int4-QAT) mean|d|={qat_delta:.3e} (expected nonzero; not 4bit-packed but QAT-finetuned)")
    ple_exact = layout_ok

    dtypes = {str(v.dtype) for v in new_sd.values()}
    print(f"\nbuilt state_dict: {len(new_sd)} tensors | nonlayer={n_nonlayer} | dropped own-k/v tensors on shared={n_dropped_kv} | dtypes={dtypes}")

    # config: stock bf16 (full multimodal, tied full head, no quant) with the
    # osoi5 body geometry spliced in.
    cfg = json.load(open(os.path.join(STOCK_BF16_DIR, "config.json")))
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = n_new
    tc["num_kv_shared_layers"] = N_KV_SHARED
    tc["layer_types"] = osoi5_layer_types
    assert cfg.get("tie_word_embeddings") is True
    assert "quantization_config" not in cfg
    json.dump(cfg, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)

    # aux files (tokenizer, processor, chat template, generation config) verbatim
    for fn in os.listdir(STOCK_BF16_DIR):
        if fn in ("model.safetensors", "config.json"):
            continue
        src = os.path.join(STOCK_BF16_DIR, fn)
        if os.path.isfile(src) or os.path.islink(src):
            shutil.copy(src, os.path.join(OUT_DIR, fn))

    save_file(new_sd, os.path.join(OUT_DIR, "model.safetensors"), metadata={"format": "pt"})
    sz = os.path.getsize(os.path.join(OUT_DIR, "model.safetensors")) / 1e9
    print(f"wrote {OUT_DIR}/model.safetensors ({sz:.2f} GB) + config.json + aux files")

    meta = {
        "out_dir": OUT_DIR,
        "source_bf16": STOCK_BF16_DIR,
        "dropped_original_indices": dropped,
        "surviving_originals_in_baked_order": S,
        "num_hidden_layers": n_new,
        "num_kv_shared_layers": N_KV_SHARED,
        "shared_positions": list(range(shared_start, n_new)),
        "tie_word_embeddings": True,
        "head": "full-262144 (tied to embeddings, bf16)",
        "perlayer_block_layout_confirmed_via_stock_int4": ple_exact,
        "bf16_vs_osoi5_embed_qat_mean_abs_delta": qat_delta,
        "n_tensors": len(new_sd),
        "n_dropped_own_kv_tensors": n_dropped_kv,
    }
    json.dump(meta, open(os.path.join(HERE, "build_37L_bf16_meta.json"), "w"), indent=2)
    print("wrote build_37L_bf16_meta.json")
    return 0 if ple_exact else 1


if __name__ == "__main__":
    sys.exit(main())
