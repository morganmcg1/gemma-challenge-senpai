#!/usr/bin/env python3
"""Read the safetensors header (no weight load) and bucket tensor bytes by
component, to derive the per-token HBM weight-read footprint at M=1 TEXT decode.

At strict batch-1 text decode every decoded token streams the full text-decoder
weights + the int4 lm_head once; the vision/audio towers are NOT read (no image/
audio input), and embed_tokens is a single-row gather (negligible). We sum the
on-disk tensor bytes (== HBM read bytes; vLLM/Marlin reads the packed int4 format
as stored) for exactly the tensors that fire per text token.
"""
import json
import re
import struct
import sys
from collections import defaultdict

ST = "/workspace/senpai/target/submissions/int4_g128_lmhead/model/model.safetensors"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr


def tensor_bytes(meta):
    a, b = meta["data_offsets"]
    return b - a


def bucket(name):
    # gemma3n / gemma-3 multimodal naming. Text decoder lives under
    # language_model / model.layers; vision under vision_tower / visual;
    # audio under audio_tower / embed_audio; multimodal embedders separate.
    low = name.lower()
    if any(k in low for k in ("vision", "visual", "siglip", "mobilenet", "image")):
        return "vision_tower"
    if any(k in low for k in ("audio", "speech", "conformer", "usm")):
        return "audio_tower"
    if "lm_head" in low:
        return "lm_head"
    if re.search(r"embed_tokens|embedder|tok_embeddings|wte", low):
        return "embed_tokens"
    if re.search(r"(^|\.)(model\.)?layers\.\d+", low) or "language_model" in low:
        return "text_decoder"
    return "other_textstack"


def main():
    hdr = read_header(ST)
    by_bucket = defaultdict(int)
    by_bucket_count = defaultdict(int)
    dtypes = defaultdict(int)
    samples = defaultdict(list)
    for name, meta in hdr.items():
        nb = tensor_bytes(meta)
        bk = bucket(name)
        by_bucket[bk] += nb
        by_bucket_count[bk] += 1
        dtypes[meta["dtype"]] += nb
        if len(samples[bk]) < 4:
            samples[bk].append((name, meta["dtype"], meta["shape"], nb))

    total = sum(by_bucket.values())
    print(f"total safetensors tensor bytes = {total/1e9:.4f} GB  ({total} B)")
    print("\n=== by bucket ===")
    for bk in sorted(by_bucket, key=lambda k: -by_bucket[k]):
        print(f"  {bk:16s} {by_bucket[bk]/1e9:8.4f} GB  n={by_bucket_count[bk]:4d}")
        for s in samples[bk]:
            print(f"      e.g. {s[0]}  {s[1]} {s[2]}  {s[3]/1e6:.3f} MB")
    print("\n=== by dtype ===")
    for dt in sorted(dtypes, key=lambda k: -dtypes[k]):
        print(f"  {dt:10s} {dtypes[dt]/1e9:8.4f} GB")

    # Per-token text-decode read footprint = text_decoder + lm_head + other_textstack
    # (final norm etc). embed_tokens excluded (single-row gather). vision/audio excluded.
    per_tok = by_bucket["text_decoder"] + by_bucket["lm_head"] + by_bucket["other_textstack"]
    print("\n=== per-token TEXT-decode weight read (M=1) ===")
    print(f"  text_decoder    = {by_bucket['text_decoder']/1e9:.4f} GB")
    print(f"  lm_head         = {by_bucket['lm_head']/1e9:.4f} GB")
    print(f"  other_textstack = {by_bucket['other_textstack']/1e9:.4f} GB")
    print(f"  -> W_bytes_per_token = {per_tok/1e9:.6f} GB  ({per_tok} B)")
    print(f"  (excluded: embed_tokens={by_bucket['embed_tokens']/1e9:.4f}GB, "
          f"vision={by_bucket['vision_tower']/1e9:.4f}GB, audio={by_bucket['audio_tower']/1e9:.4f}GB)")

    out = {
        "total_safetensors_bytes": total,
        "by_bucket_bytes": dict(by_bucket),
        "by_dtype_bytes": dict(dtypes),
        "W_bytes_per_token": per_tok,
        "W_GB_per_token": per_tok / 1e9,
        "excluded_bytes": {
            "embed_tokens": by_bucket["embed_tokens"],
            "vision_tower": by_bucket["vision_tower"],
            "audio_tower": by_bucket["audio_tower"],
        },
        "note": "W_bytes_per_token = text_decoder + lm_head + other_textstack; "
                "on-disk == HBM read bytes (packed int4 read as stored). M=1 text decode.",
    }
    with open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
