#!/usr/bin/env python
"""PR #713 — IN-MEMORY int4 fake-quant for the g32-on-locus AIME experiment.

NO checkpoint build, NO disk write. The served body is the qat-unquantized bf16
master (read-only); this module fake-quantizes selected Linear weights in place
AFTER vLLM loads them onto GPU:

    w  ->  dequantize(quantize(w, scale, zp, qargs), scale, zp, qargs)

with symmetric int4 *group* quantization. Layers in FQ_G32_LAYERS use group_size
32 (the finer, AIME-recovering grid that is fully WITHIN the int4-QAT mandate);
every other quantized body module + lm_head uses group_size 128 (= the operative
int4_g128 body). The bf16 GEMM on these fake-quantized weights carries the
identical per-group rounding error of a real int4 serve, so AIME *quality* is a
faithful proxy (speed is not — that is priced from ubel #700's byte-law).

This is the within-mandate analogue of #659's int8-on-locus cell, but built
in-memory because the local qat_unq source + disk headroom from #659 are gone.
"""
from __future__ import annotations

import re

import torch
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams

# Per-layer quantized body module *suffixes* in the vLLM runtime model. q/k/v are
# fused into qkv_proj and gate/up into gate_up_proj; group quant is along in_dim so
# fusing (concat on out_dim) is identical to quantizing the unfused shards. The
# per_layer_* gates are the Gemma per-layer-embedding pathway (ubel #700's top
# energy locus). down_proj / o_proj are unfused 1:1.
LAYER_SUFFIXES = (
    "self_attn.qkv_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "per_layer_input_gate", "per_layer_projection",
)
# global (non-layer) quantized projections in the int4 body's 343-module list
GLOBAL_SUFFIXES = ("per_layer_model_projection",)
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def make_qargs(num_bits: int, group_size: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                            group_size=group_size, symmetric=True, observer="minmax")


# Bound the transient fp32 buffers (wf, q, deq) per fake-quant step. Group quant is
# per-(row, group) along in_dim, so processing the out_dim rows in chunks is numerically
# IDENTICAL to whole-tensor — it only caps peak memory. Only the tied embed/lm_head
# (262144x2560 = 671M elems -> ~8 GiB of simultaneous fp32 transients, which OOMs the
# 22 GiB card on top of the 15 GiB model) actually needs chunking; every body module
# is a single chunk (<= this bound), so their numerics/cost are unchanged.
_MAX_CHUNK_ELEMS = 1 << 25  # 32M elems -> ~128 MiB per fp32 buffer


def fake_quant_weight(w: torch.Tensor, num_bits: int, group_size: int) -> float:
    """Fake-quantize `w` IN PLACE (symmetric int4 group quant along in_dim); return
    rel_err (Frobenius). Rows (out_dim) are processed in chunks to bound transient
    fp32 memory — exact because scale/zp are per-(row, group)."""
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
    qargs = make_qargs(num_bits, group_size)
    ng = in_dim // group_size
    rows_per_chunk = max(1, _MAX_CHUNK_ELEMS // in_dim)
    num_sq = den_sq = 0.0
    for a in range(0, out_dim, rows_per_chunk):
        b = min(a + rows_per_chunk, out_dim)
        wf = w[a:b].to(torch.float32)  # read original chunk before in-place write-back
        wg = wf.reshape(b - a, ng, group_size)
        scale, zp = calculate_qparams(wg.amin(dim=-1), wg.amax(dim=-1), qargs)
        deq = dequantize(quantize(wf, scale, zp, qargs), scale, zp, qargs)
        num_sq += float(((wf - deq) ** 2).sum())
        den_sq += float((wf ** 2).sum())
        w[a:b].copy_(deq.to(w.dtype))
    return (num_sq ** 0.5) / max(den_sq ** 0.5, 1e-9)


def parse_layers(spec: str) -> set[int]:
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return set()
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


# lm_head / embed_tokens are kept at g128 to match the locked int4_g128_lmhead rung
# (the operative body quantizes the output projection at int4 g128). They are HELD
# FIXED across the N=0 and g32-locus cells, so they never affect the paired McNemar;
# quantizing them only matters for the absolute anchor matching the int4 body's 0.400.
HEAD_SUFFIXES = ("lm_head", "embed_tokens")


_EXCLUDE = ("vision_tower", "audio_tower", "embed_audio", "embed_vision")


def _is_excluded(name: str) -> bool:
    """Exclude the multimodal towers — at vLLM runtime the text decoder may be
    named `model.layers.{i}...` (the `language_model.` prefix is stripped by the
    WeightsMapper), so we target by tower-EXCLUSION + body suffix, not by a
    `language_model` substring (which is absent at runtime)."""
    return any(t in name for t in _EXCLUDE)


def is_head_target(name: str) -> bool:
    return (not _is_excluded(name)) and any(name.endswith(s) for s in HEAD_SUFFIXES)


def is_quant_target(name: str) -> bool:
    if _is_excluded(name):
        return False
    if any(name.endswith(s) for s in GLOBAL_SUFFIXES):
        return True
    if layer_of(name) is not None and any(name.endswith(s) for s in LAYER_SUFFIXES):
        return True
    return False


def group_for(name: str, g32_layers: set[int], g32: int, g128: int) -> int:
    L = layer_of(name)
    if L is not None and L in g32_layers:
        return g32
    return g128


def apply_fake_quant(model, g32_layers: set[int], g32: int = 32, g128: int = 128,
                     quant_head: bool = True, log=print) -> dict:
    """Walk the live model, fake-quant each text-decoder body Linear weight in place.
    Dedups by weight tensor id (handles tied lm_head/embed_tokens). Returns a report."""
    seen: set[int] = set()
    body_n = head_n = 0
    g32_n = g128_n = 0
    rel_g32: list[float] = []
    rel_g128: list[float] = []
    skipped_div = 0
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or not hasattr(w, "shape") or w.dim() != 2:
            continue
        is_body = is_quant_target(name)
        is_head = quant_head and is_head_target(name)
        if not (is_body or is_head):
            continue
        if id(w) in seen:
            continue
        gs = group_for(name, g32_layers, g32, g128) if is_body else g128
        if w.shape[1] % gs != 0:
            skipped_div += 1
            log(f"[fq] SKIP {name} in_dim {w.shape[1]} not divisible by {gs}")
            continue
        rel = fake_quant_weight(w.data, 4, gs)
        seen.add(id(w))
        if is_body:
            body_n += 1
            if gs == g32:
                g32_n += 1; rel_g32.append(rel)
            else:
                g128_n += 1; rel_g128.append(rel)
        else:
            head_n += 1
    rep = {
        "body_modules_quantized": body_n, "head_modules_quantized": head_n,
        "g32_modules": g32_n, "g128_modules": g128_n,
        "g32_layers": sorted(g32_layers), "skipped_div": skipped_div,
        "mean_rel_err_g32": (sum(rel_g32) / len(rel_g32)) if rel_g32 else None,
        "mean_rel_err_g128": (sum(rel_g128) / len(rel_g128)) if rel_g128 else None,
    }
    log(f"[fq] DONE body={body_n} (g32={g32_n} g128={g128_n}) head={head_n} "
        f"g32_layers={sorted(g32_layers)} skipped={skipped_div} "
        f"rel_g32={rep['mean_rel_err_g32']} rel_g128={rep['mean_rel_err_g128']}")
    return rep
