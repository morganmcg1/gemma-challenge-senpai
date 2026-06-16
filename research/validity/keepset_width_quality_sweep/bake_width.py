#!/usr/bin/env python3
"""Bake an osoi5 37L-int4 checkpoint with a width-K lm_head for the keepset sweep.

HYBRID head construction (one variable moves: lm_head keepset width K):
  * token present in osoi5-v0's real 16k substrate  -> copy osoi5-v0's REAL int4
    packed row + scale (bit-exact; the substrate's own head, QAT/calibrated).
  * token absent from the 16k substrate (tail)       -> reconstruct int4(embed[t])
    with the checkpoint's own quantizer recipe (symmetric per-channel,
    memoryless_minmax => scale=amax/7.5, codes=clamp(round(w/scale),-8,7)).
    This is exactly the BASE model's tied head (base tie_word_embeddings=True =>
    head==embed) quantized the same way the substrate quantizes its present rows,
    so a tail row is as faithful to base as int4 allows. Tail tokens were pruned
    (never modified) in osoi5, so raw int4(embed) is the right target.

Everything else (37L body, int4 body quant, embed_tokens, config) is copied
verbatim from osoi5-v0 -> ONLY the lm_head keepset width changes.

The result loads through the REAL submission serve path: serve_patch_pck04 rebuilds
lm_head to K=len(keep_ids) rows and the compressed-tensors loader fills it from the
stored [K,320] packed head; compute_logits scatters [M,K] -> [M,262144] with -inf.

Usage:
  bake_width.py --target-keepset /tmp/keepset32k.json --out /tmp/osoi5-32k-baked
  bake_width.py --full --out /tmp/osoi5-full-baked        # K=262144, no prune
  bake_width.py ... --no-write   # build+validate head in RAM only (fast, no 9GB write)
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32, unpack_from_int32,
)

SRC = "/tmp/osoi5-v0-baked"
FULL_VOCAB = 262144
HIDDEN = 2560
DENOM = 7.5  # compressed-tensors symmetric int4 minmax: scale = amax / (15/2)
AUX_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "processor_config.json", "chat_template.jinja", "README.md", ".gitattributes",
    "preprocessor_config.json", "special_tokens_map.json",
]


def log(m):
    print(f"[bake {time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_head(target_ids):
    """Return (packed[K,320] int32, scale[K,1] f16, n_present, n_tail) for target_ids."""
    K = len(target_ids)
    with safe_open(f"{SRC}/model.safetensors", framework="pt", device="cpu") as f:
        real_packed = f.get_tensor("lm_head.weight_packed")             # [16384,320] i32
        real_scale = f.get_tensor("lm_head.weight_scale")               # [16384,1] f16
        embed = f.get_tensor("model.language_model.embed_tokens.weight")  # [262144,2560] bf16
    src_keep = json.load(open(f"{SRC}/pck04_keepset.json"))["keep_ids"]   # 16384 ids
    tok2row = {int(t): i for i, t in enumerate(src_keep)}

    tgt = torch.tensor(target_ids, dtype=torch.long)
    present_mask = torch.tensor([int(t) in tok2row for t in target_ids], dtype=torch.bool)
    n_present = int(present_mask.sum())
    n_tail = K - n_present
    log(f"K={K}  present(real 16k rows)={n_present}  tail(recon int4(embed))={n_tail}")

    new_packed = torch.empty((K, 320), dtype=torch.int32)
    new_scale = torch.empty((K, 1), dtype=torch.float16)

    # present rows: gather REAL packed+scale by source-row index (bit-exact)
    pres_pos = torch.nonzero(present_mask, as_tuple=True)[0]
    if len(pres_pos):
        src_rows = torch.tensor([tok2row[int(target_ids[int(i)])] for i in pres_pos], dtype=torch.long)
        new_packed[pres_pos] = real_packed[src_rows]
        new_scale[pres_pos] = real_scale[src_rows]

    # tail rows: int4(embed[t], scale=amax/7.5), packed compressed-tensors layout
    tail_pos = torch.nonzero(~present_mask, as_tuple=True)[0]
    if len(tail_pos):
        tail_tok = tgt[tail_pos]
        emb = embed[tail_tok].to(torch.float32)                # [n_tail,2560]
        amax = emb.abs().amax(dim=1, keepdim=True)
        s = amax / DENOM                                       # [n_tail,1]
        codes = torch.clamp(torch.round(emb / s), -8, 7).to(torch.int8)
        packed_tail = pack_to_int32(codes, num_bits=4, packed_dim=1)  # [n_tail,320] i32
        new_packed[tail_pos] = packed_tail
        new_scale[tail_pos] = s.to(torch.float16)

    # ---- validation ----
    # present rows must be bit-exact to the real substrate
    if len(pres_pos):
        re_codes = unpack_from_int32(new_packed[pres_pos], num_bits=4,
                                     shape=torch.Size([len(pres_pos), HIDDEN]), packed_dim=1)
        real_codes = unpack_from_int32(real_packed[src_rows], num_bits=4,
                                       shape=torch.Size([len(pres_pos), HIDDEN]), packed_dim=1)
        assert bool((re_codes == real_codes).all()), "present rows not bit-exact to real substrate!"
        assert bool((new_scale[pres_pos] == real_scale[src_rows]).all()), "present scales differ!"
        log(f"validate: present {len(pres_pos)} rows bit-exact to osoi5-v0 (codes+scale) OK")
    # tail rows: dequant must track embed. NB the int4-per-row floor is intrinsic:
    # the REAL osoi5 head has cos down to ~0.69 on degenerate/special-token rows
    # (median 0.979). So we gate on the MEDIAN (catches packing/index bugs, which
    # would crater cos to ~0) and allow legit low-cos outlier rows.
    if len(tail_pos):
        smp = tail_pos[torch.linspace(0, len(tail_pos) - 1, steps=min(2048, len(tail_pos))).long()]
        dq = unpack_from_int32(new_packed[smp], num_bits=4, shape=torch.Size([len(smp), HIDDEN]),
                               packed_dim=1).to(torch.float32) * new_scale[smp].to(torch.float32)
        emb_smp = embed[tgt[smp]].to(torch.float32)
        cos = torch.nn.functional.cosine_similarity(dq, emb_smp, dim=1)
        log(f"validate: tail({len(smp)} smp) cos(dequant,embed) "
            f"min={cos.min():.4f} median={cos.median():.4f} mean={cos.mean():.4f} "
            f"(real-substrate ref: median 0.979, min 0.69)")
        assert float(cos.median()) > 0.95, "tail reconstruction MEDIAN cosine too low — likely a bug!"
        assert float(cos.min()) > 0.30, "tail reconstruction has near-zero cos — packing/index bug!"
    return new_packed, new_scale, n_present, n_tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-keepset", help="path to pck04_keepset.json with keep_ids")
    ap.add_argument("--full", action="store_true", help="K=262144, keep_ids=range(vocab)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-write", action="store_true", help="build+validate head only, skip 9GB write")
    a = ap.parse_args()

    if a.full:
        target_ids = list(range(FULL_VOCAB))
        src_name = "full(range)"
    else:
        target_ids = json.load(open(a.target_keepset))["keep_ids"]
        assert target_ids == sorted(target_ids), "keepset must be ascending"
        src_name = a.target_keepset
    K = len(target_ids)

    t0 = time.time()
    new_packed, new_scale, n_present, n_tail = build_head(target_ids)
    new_shape = torch.tensor([K, HIDDEN], dtype=torch.int64)
    log(f"head built in {time.time()-t0:.1f}s  packed={list(new_packed.shape)} scale={list(new_scale.shape)}")

    if a.no_write:
        log("--no-write: skipping checkpoint write")
        return 0

    os.makedirs(a.out, exist_ok=True)
    log(f"loading all osoi5-v0 tensors ({os.path.getsize(f'{SRC}/model.safetensors')/1e9:.1f} GB) ...")
    with safe_open(f"{SRC}/model.safetensors", framework="pt", device="cpu") as f:
        meta = f.metadata()
    tensors = load_file(f"{SRC}/model.safetensors")
    tensors["lm_head.weight_packed"] = new_packed.contiguous()
    tensors["lm_head.weight_scale"] = new_scale.contiguous()
    tensors["lm_head.weight_shape"] = new_shape.contiguous()
    log(f"writing {a.out}/model.safetensors ...")
    save_file(tensors, f"{a.out}/model.safetensors", metadata=meta)

    for fn in AUX_FILES:
        sp = os.path.join(SRC, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(a.out, fn))
    with open(os.path.join(a.out, "pck04_keepset.json"), "w") as f:
        json.dump({
            "keep_ids": target_ids, "pruned_vocab_K": K, "full_vocab": FULL_VOCAB,
            "source_keepset": src_name,
            "note": (f"keepset-width sweep K={K}; hybrid head: real osoi5-v0 rows for "
                     f"{n_present} present(16k) tokens, int4(embed,amax/{DENOM}) for {n_tail} tail tokens"),
        }, f)
    sz = os.path.getsize(f"{a.out}/model.safetensors") / 1e9
    log(f"DONE  {a.out}  model.safetensors={sz:.2f} GB  K={K}  ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
