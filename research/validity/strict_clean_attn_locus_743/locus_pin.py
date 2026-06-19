#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #743 (land) -- pin the strict-clean spec divergence locus.

wirbel #736 / land #680 proved the int4 g=128 Marlin GEMV is BIT-EXACTLY
M-invariant (max bitdiff = 0, M=1..16). So the strict-#319 divergence between the
K-token spec-VERIFY forward and the single-token AR-DECODE forward is NOT in the
GEMM -- the leading candidate is the ATTENTION branch. This script PINS that
divergence to a specific layer + op on the in-scope accuracy stack and classifies
whether it is a deterministic reduction-ORDER artifact (byte-exact-fixable) or a
genuine numeric difference.

METHOD -- a controlled A/B over an IDENTICAL token prefix, in-process vLLM 0.22.0
serving the int4 base ``gemma-4-E4B-it`` (TRITON_ATTN, the served unified-attention
kernel whose 2D/3D split-KV reduction is the only M-dependent attention op:
triton_unified_attention.py:923-931 -> ``use_3d`` is False when ``max_seqlen_q>1``
(the M=K verify shape) or ``is_batch_invariant``, True for the M=1 decode shape):

  * (b) DECODE / M=1 path  : greedy AR generate. Each decode step is a 1-row
        forward -> max_seqlen_q==1 -> the 3D segmented-LSE split-KV reduction.
  * (a) VERIFY / M=K path  : re-forward [ctx+gen] with max_num_batched_tokens=M
        -> every position is computed at M-occupancy, max_seqlen_q>1 -> the 2D
        one-shot reduction (the spec-verify shape; 2D output is query-row-
        independent so this is a faithful proxy for the K-wide verify block).

We register forward hooks on EVERY decoder layer's ``self_attn.{qkv_proj, attn,
o_proj}`` (capturing the attention INPUTS q,k,v post-norm/RoPE and the attention
OUTPUT = flash context = o_proj input) plus the ``lm_head`` pre-hook (the
pre-lm_head hidden), key every captured row by its ABSOLUTE position, and diff the
M=1 decode capture against the M=K verify capture at matched positions. The first
op in residual order (qkv_proj -> q/k/v -> attn_out -> o_proj) with a non-zero
bitdiff at the earliest layer is the LOCUS.

CLASSIFICATION (instr 4): rerun with ``--batch-invariant 1`` -> the M=1 decode is
forced to num_splits=1 (2D), matching the M=K verify reduction order. If the
attn_out bitdiff collapses to ~0 under matched order, the locus is a reduction-
ORDER artifact -> byte-exact-fixable by pinning the split. If it persists, it is a
genuine numeric difference.

FIDELITY (honest, identical to the #491/#680 caveat): the loadable full-vocab int4
ckpt is the upstream QAT checkpoint (the deployed int4_g128_lmhead has a pruned
16384-row head vanilla vLLM cannot load). The GEMV byte-invariance was measured at
the EXACT deployed g=128 shapes in #680 Leg A; attention M-dependence is a kernel-
occupancy property independent of weight-quant granularity, so the full-vocab ckpt
is faithful for the attention-attribution question. analysis_only, LOCAL A10G, no
served-file change, NO HF Job.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
CENSUS_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
sys.path.insert(0, str(CENSUS_DIR))


def _set_env(batch_invariant: bool) -> None:
    # flashinfer sampler JIT-builds a curand kernel absent in this venv; greedy
    # argmax is unaffected by the native sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # in-process engine -> forward hooks registered here actually fire.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # the served unified-attention kernel with the documented 2D/3D split gating.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    # classification lever: forces M=1 decode to num_splits=1 (2D), matching verify.
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant else "0"


# residual-stream op order within a layer (for first-divergence reporting)
OP_ORDER = ["qkv", "attn_q", "attn_k", "attn_v", "attn_out", "oproj_out"]

# ---- capture state (populated by hooks; in-process so this module's globals are live)
CAP: dict[str, Any] = {
    "mode": None,          # "store" (M=1 decode) | "diff" (M=K verify)
    "positions": None,     # cpu long tensor [n_tok] for the current forward
    "store": {},           # (li, op, pos) -> cpu fp32 tensor   (M=1 3D decode rows only)
    "stats": defaultdict(lambda: {"n": 0, "n_bitdiff": 0, "max_abs": 0.0}),  # (li,op)->stat
}


def _pos_pre_hook(mod, args, kwargs):
    # Gemma4DecoderLayer calls self_attn(positions=.., hidden_states=..) -> kwargs
    p = args[0] if args else kwargs.get("positions")
    if p is not None:
        CAP["positions"] = p.detach().to("cpu")
    return None


def _record(li: int, op: str, t) -> None:
    """Shared store/diff body. ``t`` is the [n_tok, feat] tensor for op (li,op).
    Store mode keeps only 1-row (M=1 3D decode) rows; diff mode compares matched
    absolute positions and skips the GPU->CPU copy when no row was stored."""
    import torch
    pos = CAP["positions"]
    if pos is None or not torch.is_tensor(t) or t.shape[0] != pos.shape[0]:
        return
    n = t.shape[0]
    if CAP["mode"] == "store":
        if n != 1:               # only the M=1 decode forwards (3D split-KV)
            return
        CAP["store"][(li, op, int(pos[0]))] = t[0].detach().to("cpu", torch.float32)
    elif CAP["mode"] == "diff":
        rows = [(r, int(pos[r])) for r in range(n) if (li, op, int(pos[r])) in CAP["store"]]
        if not rows:             # ctx-prefill positions were never stored -> skip copy
            return
        tc = t.detach().to("cpu", torch.float32)
        st = CAP["stats"][(li, op)]
        for r, p in rows:
            ref = CAP["store"][(li, op, p)]
            st["n"] += 1
            if not torch.equal(tc[r], ref):
                st["n_bitdiff"] += 1
                d = (tc[r] - ref).abs().max().item()
                if d > st["max_abs"]:
                    st["max_abs"] = d


def _make_capture_hook(li: int, op: str, source: str, idx: int = 0):
    """forward_hook. source: 'out' (module output) | 'in' (input[idx])."""
    def hook(mod, inp, out):
        t = inp[idx] if source == "in" else (out[0] if isinstance(out, tuple) else out)
        _record(li, op, t)
        return None
    return hook


def get_model_runner(llm):
    """Reach the in-process v1 model_runner (VLLM_ENABLE_V1_MULTIPROCESSING=0).
    Mirrors research/speed/byte_identical_kernel/attention_byte_identical_ceiling.py."""
    cands = []
    try:
        cands.append(llm.llm_engine.engine_core.engine_core
                     .model_executor.driver_worker.worker.model_runner)
    except Exception:  # noqa: BLE001
        pass
    try:
        ec = llm.llm_engine.engine_core
        mexec = getattr(ec, "model_executor", None)
        if mexec is not None:
            dw = getattr(mexec, "driver_worker", None)
            w = getattr(dw, "worker", dw)
            mr = getattr(w, "model_runner", None)
            if mr is not None:
                cands.append(mr)
    except Exception:  # noqa: BLE001
        pass
    for mr in cands:
        if mr is not None and hasattr(mr, "model"):
            return mr
    raise RuntimeError("could not reach model_runner")


def _find_layers_and_lmhead(model):
    """Locate per-layer self_attn submodules, the final pre-lm_head norm, and lm_head."""
    import re
    attn_by_idx: dict[int, Any] = {}
    qkv_by_idx: dict[int, Any] = {}
    attnop_by_idx: dict[int, Any] = {}
    oproj_by_idx: dict[int, Any] = {}
    final_norm = None
    lm_head = None
    impl_classes = set()
    pat = re.compile(r"\.layers\.(\d+)\.self_attn$")
    for name, mod in model.named_modules():
        m = pat.search(name)
        if m:
            li = int(m.group(1))
            attn_by_idx[li] = mod
            qkv_by_idx[li] = getattr(mod, "qkv_proj", None)
            attnop_by_idx[li] = getattr(mod, "attn", None)
            oproj_by_idx[li] = getattr(mod, "o_proj", None)
            ai = getattr(mod, "attn", None)
            if ai is not None and hasattr(ai, "impl"):
                impl_classes.add(type(ai.impl).__name__)
        # final norm: "...language_model.norm" (per-layer ones end with "layernorm")
        if name.endswith(".norm") and ".layers." not in name:
            final_norm = mod
        if name.endswith("lm_head"):
            lm_head = mod
    return (attn_by_idx, qkv_by_idx, attnop_by_idx, oproj_by_idx, final_norm, lm_head,
            sorted(impl_classes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-width", type=int, default=6, help="M=K+1 verify occupancy")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--det-prompts", type=int, default=4, help="# prompts with AR-vs-AR control")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--batch-invariant", type=int, default=0, choices=(0, 1))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    _set_env(bool(args.batch_invariant))
    tag = "bi1" if args.batch_invariant else "bi0"
    out_path = args.out or (HERE / "runs" / f"locus_pin_{tag}.json")

    import torch
    from vllm import LLM, SamplingParams
    from reduction_sensitivity_census import (  # noqa: E402
        load_prompts, resolve_model_dir, _margin_model_full_vocab, entry_as_dict,
    )

    model_dir = resolve_model_dir()
    full_vocab = _margin_model_full_vocab(model_dir)
    prompts = load_prompts(args.n_prompts, args.ctx_cap)
    M = args.verify_width
    print(f"[locus {tag}] model={model_dir} full_vocab={full_vocab} "
          f"prompts={len(prompts)} M={M} n_new={args.n_new} "
          f"BATCH_INVARIANT={os.environ['VLLM_BATCH_INVARIANT']} "
          f"backend={os.environ['VLLM_ATTENTION_BACKEND']}", flush=True)

    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=max(1024, args.ctx_cap + args.n_new + 16),
              gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
              max_num_batched_tokens=M, enable_prefix_caching=False,
              enforce_eager=True, trust_remote_code=True,
              max_logprobs=max(20, args.topk + 2))

    # reach the in-process model + register hooks
    model = get_model_runner(llm).model
    attn_by_idx, qkv_by_idx, attnop_by_idx, oproj_by_idx, final_norm, lm_head, impl_classes = \
        _find_layers_and_lmhead(model)
    n_layers = len(attn_by_idx)
    print(f"[locus {tag}] hooked {n_layers} layers  attn_impl={impl_classes}  "
          f"final_norm={type(final_norm).__name__ if final_norm is not None else None}  "
          f"lm_head={type(lm_head).__name__ if lm_head is not None else None}", flush=True)
    assert n_layers > 0, "no decoder layers found"

    handles = []
    for li, attn in attn_by_idx.items():
        handles.append(attn.register_forward_pre_hook(_pos_pre_hook, with_kwargs=True))
        if qkv_by_idx[li] is not None:
            handles.append(qkv_by_idx[li].register_forward_hook(_make_capture_hook(li, "qkv", "out")))
        if attnop_by_idx[li] is not None:
            ao = attnop_by_idx[li]
            handles.append(ao.register_forward_hook(_make_capture_hook(li, "attn_q", "in", 0)))
            handles.append(ao.register_forward_hook(_make_capture_hook(li, "attn_k", "in", 1)))
            handles.append(ao.register_forward_hook(_make_capture_hook(li, "attn_v", "in", 2)))
            handles.append(ao.register_forward_hook(_make_capture_hook(li, "attn_out", "out")))
        if oproj_by_idx[li] is not None:
            handles.append(oproj_by_idx[li].register_forward_hook(_make_capture_hook(li, "oproj_out", "out")))
    if final_norm is not None:  # pre-lm_head hidden (end-to-end magnitude)
        handles.append(final_norm.register_forward_hook(_make_capture_hook(-1, "prelmhead", "out")))

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.n_new, logprobs=args.topk)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)

    # end-to-end (instr 1-2): argmax-flip via verify prompt_logprobs vs M=1 ref token
    e2e = {"n_pos": 0, "n_argflip": 0, "n_logprob_bitdiff": 0}
    det = {"n_match": 0, "n_pos": 0}     # AR-vs-AR determinism control (instr 4 linchpin)
    n_decode_pos_total = 0
    t0 = time.time()

    for pi, pr in enumerate(prompts):
        ctx = pr["context_token_ids"]
        c = len(ctx)
        base = {"prompt_token_ids": ctx}

        # ---- (b) M=1 AR decode: store 3D decode-step captures
        CAP["mode"] = "store"
        CAP["store"].clear()
        CAP["positions"] = None
        out = llm.generate([base], gen_sp, use_tqdm=False)[0]
        gen = list(out.outputs[0].token_ids)
        gen_lps = out.outputs[0].logprobs or []
        if not gen:
            continue
        n_decode_pos_total += sum(1 for k in CAP["store"] if k[1] == "attn_out")

        # AR-vs-AR determinism control
        if pi < args.det_prompts:
            CAP["mode"] = None  # don't capture during the control re-gen
            gen_b = list(llm.generate([base], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
            Lg = min(len(gen), len(gen_b))
            det["n_match"] += sum(1 for a, b in zip(gen[:Lg], gen_b[:Lg]) if a == b)
            det["n_pos"] += Lg

        # ---- (a) M=K verify: re-forward [ctx+gen], diff against stored decode
        CAP["mode"] = "diff"
        CAP["positions"] = None
        full = ctx + gen
        vout = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs

        # end-to-end argmax-flip at gen positions
        for g in range(len(gen)):
            j = c + g
            if pls is None or j >= len(pls) or pls[j] is None:
                continue
            ref_tok = int(gen[g])
            mv = entry_as_dict(pls[j])
            v_arg = max(mv, key=mv.get) if mv else ref_tok
            v_top = mv.get(v_arg, float("-inf"))
            m1 = entry_as_dict(gen_lps[g]) if g < len(gen_lps) else {}
            m1_top = m1.get(ref_tok, m1.get(max(m1, key=m1.get), float("-inf"))) if m1 else float("-inf")
            e2e["n_pos"] += 1
            if v_arg != ref_tok:
                e2e["n_argflip"] += 1
            if math.isfinite(v_top) and math.isfinite(m1_top) and v_top != m1_top:
                e2e["n_logprob_bitdiff"] += 1

        print(f"  [{pi+1}/{len(prompts)}] c={c} gen={len(gen)} "
              f"decode_pos_stored={sum(1 for k in CAP['store'] if k[1]=='attn_out')}", flush=True)

    for h in handles:
        h.remove()

    # per-(layer,op) bitdiff table (prelmhead is li=-1, the end-to-end pre-lm_head hidden)
    layers_tbl: dict[int, dict[str, Any]] = defaultdict(dict)
    for (li, op), st in CAP["stats"].items():
        frac = (st["n_bitdiff"] / st["n"]) if st["n"] else float("nan")
        layers_tbl[li][op] = {"n": st["n"], "n_bitdiff": st["n_bitdiff"],
                              "frac_bitdiff": frac, "max_abs": st["max_abs"]}
    prelm_cell = layers_tbl.get(-1, {}).get("prelmhead")

    # first divergent op at the EARLIEST layer (residual order; real layers only, li>=0)
    real_layers = [li for li in sorted(layers_tbl) if li >= 0]
    first_locus = None
    for li in real_layers:
        for op in OP_ORDER:
            cell = layers_tbl[li].get(op)
            if cell and cell["n_bitdiff"] > 0:
                first_locus = {"layer": li, "op": op, "frac_bitdiff": cell["frac_bitdiff"],
                               "max_abs": cell["max_abs"], "n": cell["n"]}
                break
        if first_locus:
            break

    # layer-0 clean chain (the airtight pin: qkv/attn-inputs identical, attn_out first-diverges)
    layer0 = layers_tbl.get(real_layers[0], {}) if real_layers else {}

    e2e_argflip = (e2e["n_argflip"] / e2e["n_pos"]) if e2e["n_pos"] else float("nan")
    e2e_lpbit = (e2e["n_logprob_bitdiff"] / e2e["n_pos"]) if e2e["n_pos"] else float("nan")
    ar_vs_ar = (det["n_match"] / det["n_pos"]) if det["n_pos"] else float("nan")

    result = {
        "phase": "strict_clean_attn_locus_pin", "tag": tag,
        "batch_invariant": bool(args.batch_invariant),
        "attn_backend": os.environ["VLLM_ATTENTION_BACKEND"],
        "attn_impl_classes": impl_classes,
        "model_dir": model_dir, "margin_model_full_vocab": full_vocab,
        "verify_width": M, "n_prompts": len(prompts), "n_new": args.n_new,
        "n_layers": n_layers, "n_decode_positions": n_decode_pos_total,
        # instr 1-2: end-to-end A/B (identical prefix; M=K verify vs M=1 decode)
        "e2e_positions": e2e["n_pos"],
        "e2e_argmax_flip_rate": e2e_argflip,
        "e2e_argmax_flips": e2e["n_argflip"],
        "e2e_logprob_bitdiff_rate": e2e_lpbit,
        # pre-lm_head hidden delta (instr 1-2 magnitude; logits = width-invariant lm_head(hidden))
        "prelmhead": prelm_cell,
        # instr 4 linchpin
        "ar_vs_ar_token_identity": ar_vs_ar, "ar_vs_ar_positions": det["n_pos"],
        # instr 3: per-layer per-op localization
        "first_divergent_locus": first_locus,
        "layer0_chain": {op: layer0.get(op) for op in OP_ORDER},
        "per_layer_op": {str(li): layers_tbl[li] for li in sorted(layers_tbl)},
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2, default=str)

    print("\n" + "=" * 72, flush=True)
    print(f"[LOCUS {tag}]  backend={impl_classes}  layers={n_layers}", flush=True)
    print(f"  e2e argmax-flip (M=K verify vs M=1 decode): {e2e_argflip:.5f} "
          f"({e2e['n_argflip']}/{e2e['n_pos']})", flush=True)
    print(f"  AR-vs-AR token identity (control):          {ar_vs_ar:.6f} "
          f"({det['n_match']}/{det['n_pos']})", flush=True)
    if prelm_cell:
        print(f"  pre-lm_head hidden delta: frac_bitdiff={prelm_cell['frac_bitdiff']:.4f} "
              f"max_abs={prelm_cell['max_abs']:.3e} (n={prelm_cell['n']})", flush=True)
    print(f"  FIRST divergent locus: {first_locus}", flush=True)
    print(f"  layer-0 chain (op: frac_bitdiff / max_abs):", flush=True)
    for op in OP_ORDER:
        cell = layer0.get(op)
        if cell:
            print(f"     {op:10s}: frac={cell['frac_bitdiff']:.4f}  max_abs={cell['max_abs']:.3e}  (n={cell['n']})", flush=True)
    print(f"  result -> {out_path}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
