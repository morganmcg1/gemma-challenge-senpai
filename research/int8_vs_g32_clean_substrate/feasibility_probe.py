"""GPU-free feasibility probe for PR #726.

Question: on the bf16-fake-quant substrate (whose only available source is the
int4-g32 QAT checkpoint /tmp/gemma40L-int4), does an int8 fake-quant of a body
module differ meaningfully from the int4-g32 weights?

If int8-fake-quant(int4_g32_weights) ~= int4_g32_weights (rel_err << the g128
re-quant error ~0.10), then "int8-locus" and "full-g32" are weight-identical on
this substrate by construction -> the int8>g32 delta is pre-determined ~0
(INVERT) regardless of the locus module list. That is a substrate-feasibility
fact the advisor needs before any GPU run.
"""
import torch
from safetensors import safe_open

SRC = "/tmp/gemma40L-int4/model.safetensors"


def load(k):
    with safe_open(SRC, "pt") as f:
        return f.get_tensor(k)


def unpack_int4(packed, shape):
    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
            unpack_from_int32 as u,
        )
        return u(packed, 4, shape)
    except Exception:
        out_f, in_f = int(shape[0]), int(shape[1])
        nib = torch.zeros((out_f, in_f), dtype=torch.int32)
        for i in range(8):
            nib[:, i::8] = (packed >> (4 * i)) & 0xF
        return nib - 8  # unsigned store with zp offset 8 -> signed [-8,7]


def dequant_g(qint, scale, gs):
    out_f, in_f = qint.shape
    q = qint.reshape(out_f, in_f // gs, gs).to(torch.float32)
    W = (q * scale.to(torch.float32).unsqueeze(-1)).reshape(out_f, in_f)
    return W


def fq_intN(W, gs, bits):
    """Symmetric per-group fake-quant of bf16 weights W at given group size / bit width."""
    out_f, in_f = W.shape
    qmax = (1 << (bits - 1)) - 1
    Wg = W.reshape(out_f, in_f // gs, gs)
    amax = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    s = amax / qmax
    q = torch.clamp(torch.round(Wg / s), -qmax, qmax)
    return (q * s).reshape(out_f, in_f)


def rel_err(A, B):
    return (A - B).norm().item() / B.norm().item()


mods = []
for L in (0, 11, 23, 39):
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        mods.append(f"model.language_model.layers.{L}.{name}")

print(f"{'module':52s} {'i8g32/g32':>10s} {'i8pc/g32':>10s} {'g128/g32':>10s} {'g64/g32':>10s}")
agg = {"i8g32": [], "i8pc": [], "g128": [], "g64": []}
for m in mods:
    try:
        packed = load(m + ".weight_packed")
        scale = load(m + ".weight_scale")
        shape = load(m + ".weight_shape").tolist()
    except Exception as e:
        print(f"{m:52s} MISSING ({e})")
        continue
    qint = unpack_int4(packed, shape)
    W_g32 = dequant_g(qint, scale, 32)  # the substrate "full-g32" reference (== source)

    in_f = W_g32.shape[1]
    i8_g32 = fq_intN(W_g32, 32, 8)
    i8_pc = fq_intN(W_g32, in_f, 8)        # int8 per-channel (whole row = 1 group)
    g128 = fq_intN(W_g32, 128, 4) if in_f % 128 == 0 else torch.full_like(W_g32, float('nan'))
    g64 = fq_intN(W_g32, 64, 4) if in_f % 64 == 0 else torch.full_like(W_g32, float('nan'))

    re = {
        "i8g32": rel_err(i8_g32, W_g32),
        "i8pc": rel_err(i8_pc, W_g32),
        "g128": rel_err(g128, W_g32),
        "g64": rel_err(g64, W_g32),
    }
    for k, v in re.items():
        if v == v:  # not nan
            agg[k].append(v)
    print(f"{m.replace('model.language_model.',''):52s} {re['i8g32']:10.5f} {re['i8pc']:10.5f} {re['g128']:10.5f} {re['g64']:10.5f}")

print("\n--- mean rel_err vs full-g32 (the substrate reference) ---")
for k in ("i8g32", "i8pc", "g128", "g64"):
    xs = agg[k]
    if xs:
        print(f"  {k:8s}: mean={sum(xs)/len(xs):.5f}  max={max(xs):.5f}  n={len(xs)}")
print("\nInterpretation: i8* rel_err << g128 rel_err => int8 fake-quant of the int4-g32 "
      "source is ~lossless vs g32 => int8-locus == full-g32 by construction on this substrate.")
