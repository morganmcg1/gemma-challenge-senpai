#!/usr/bin/env python3
"""Linchpin validation for the keepset-width sweep's hybrid head builder.

Claim under test: osoi5-v0's stored int4 lm_head row for token t is
`int4(embed_tokens[t])` with a per-row symmetric scale = amax/denom. If true across
ALL 16384 present rows (not just a 7-row sample), then reconstructing tail rows
(tokens absent from the 16k substrate) as int4(embed_tokens[t]) is exactly as
faithful as the real rows -> a full/32k reconstructed head is a valid measurement.

Also round-trips pack_to_int32(unpack(packed)) == packed to validate our pack usage.

CPU-only. Reads /tmp/osoi5-v0-baked.
"""
import json
import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.helpers import (
    unpack_from_int32, pack_to_int32,
)

CKPT = "/tmp/osoi5-v0-baked"
with safe_open(f"{CKPT}/model.safetensors", framework="pt", device="cpu") as f:
    packed = f.get_tensor("lm_head.weight_packed")          # [16384, 320] int32
    scale = f.get_tensor("lm_head.weight_scale").to(torch.float32)  # [16384, 1]
    wshape = f.get_tensor("lm_head.weight_shape").tolist()  # [16384, 2560]
    embed_t = f.get_tensor("model.language_model.embed_tokens.weight")  # [262144,2560] bf16

keep = json.load(open(f"{CKPT}/pck04_keepset.json"))["keep_ids"]
rows, cols = wshape
print(f"head [{rows},{cols}]  scale {list(scale.shape)}  keep_ids={len(keep)}")

# 1) pack round-trip on the REAL stored codes (validates our pack/unpack usage)
real_codes = unpack_from_int32(packed, num_bits=4, shape=torch.Size([rows, cols]), packed_dim=1)
repacked = pack_to_int32(real_codes, num_bits=4, packed_dim=1)
rt_ok = bool((repacked == packed).all().item())
print(f"[pack round-trip] pack(unpack(packed))==packed : {rt_ok}  "
      f"(codes dtype={real_codes.dtype} min={int(real_codes.min())} max={int(real_codes.max())})")

# 2) reconstruct all 16384 rows from embed_tokens and compare codes to the real stored codes
emb = embed_t[keep].to(torch.float32)                  # [16384, 2560], gather present rows
amax = emb.abs().amax(dim=1, keepdim=True)             # [16384, 1]
print("\n=== reconstruct int4(embed) vs REAL stored codes, ALL rows ===")
best = None
for denom in (7.0, 8.0):
    s = amax / denom
    q = torch.clamp(torch.round(emb / s), -8, 7).to(real_codes.dtype)
    exact_rows = int((q == real_codes).all(dim=1).sum().item())
    within1 = (((q - real_codes).abs() <= 1).float().mean().item())
    within0 = ((q == real_codes).float().mean().item())
    # scale agreement: does stored scale == amax/denom?
    rel_scale_err = ((s - scale).abs() / (scale.abs() + 1e-12)).mean().item()
    print(f"  scale=amax/{denom:.0f}: exact-rows {exact_rows}/{rows} ({exact_rows/rows:.3%})  "
          f"per-elem exact {within0:.4%}  within-1 {within1:.4%}  | mean|stored-amax/{denom:.0f}|/stored = {rel_scale_err:.3e}")
    if best is None or rel_scale_err < best[1]:
        best = (denom, rel_scale_err, exact_rows, within1)

denom, _, ex, w1 = best
print(f"\n[verdict] osoi5 scale recipe = amax/{denom:.0f}  ->  exact-rows {ex}/{rows} ({ex/rows:.3%}), within-1 {w1:.4%}")

# 3) logit-level sanity: with a random hidden vector, does reconstructed-row argmax-rank
#    track the real-row dot product? (cheap proxy that <=1 LSB code noise is harmless)
torch.manual_seed(0)
h = torch.randn(2560, dtype=torch.float32)
real_logit = (real_codes.to(torch.float32) * scale) @ h          # [16384]
s = amax / denom
qbest = torch.clamp(torch.round(emb / s), -8, 7).to(torch.float32)
recon_logit = (qbest * s) @ h
# top-1 agreement and rank correlation of the top-50
r_top = torch.topk(real_logit, 50).indices
q_top = torch.topk(recon_logit, 50).indices
top1 = bool(r_top[0].item() == q_top[0].item())
overlap = len(set(r_top.tolist()) & set(q_top.tolist()))
maxabs = (real_logit - recon_logit).abs().max().item()
rng = (real_logit.max() - real_logit.min()).item()
print(f"\n[logit proxy] random-h: top1 match={top1}  top50 overlap={overlap}/50  "
      f"max|Δlogit|={maxabs:.4f}  logit range={rng:.2f}  (Δ/range={maxabs/rng:.4%})")
