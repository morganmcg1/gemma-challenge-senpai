#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #761 (lawine) -- DIVERGENCE-LOCUS census: where do the 104 served-spec
divergent positions FIRST diverge?

This is the direct, per-position first-divergence trace that #755 reached only by
ELIMINATION. #755 localized the residual M=1-decode-vs-M=K-verify divergence to the
"un-BI int4 Marlin matmul" because it survives num_splits-force + BI=1 + enforce_eager.
#761 asks WHERE precisely, and attributes the 104 divergent positions to a single op
family.

KEY FIDELITY UPGRADE over land #743's locus_pin.py: #743 traced the **Hub** QAT ckpt
``gemma-4-E4B-it-qat-w4a16-ct`` whose lm_head is **bf16** (tie_word_embeddings=True,
lm_head in the quant ``ignore`` list) -> under BI=1 the aten matmul patch makes that
bf16 head byte-invariant, so #743 saw 0/192 and a null locus. The DEPLOYED ckpt
``/workspace/gemma_build/int4_g128_lmhead`` instead has an **int4 g128 Marlin lm_head**
(weight_packed [262144,320], in quant group_1). That int4 Marlin head is the ONE op
present in the served stack but absent (as bf16) from #743's probe -- and it is a
custom Marlin op, never touched by enable_batch_invariant_mode() (which patches only
aten mm/addmm/bmm). This script traces the EXACT deployed stack in-process (the full
262144-vocab head loads in vanilla vLLM), so there is ZERO fidelity gap.

TWO complementary phases, both LOCAL A10G, in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0
so forward hooks fire), BI=1 + TRITON_ATTN + enforce_eager (the served kernel config;
#755 proved enforce_eager does not change the divergence):

  Phase 1 -- forward A/B per-position first-divergence map.
    M=1 AR greedy decode (each step max_seqlen_q=1) STORE every op activation keyed by
    absolute position. Then re-forward [ctx+gen] at max_num_batched_tokens=K+1 (the
    M=K+1 spec-verify occupancy) and DIFF every op against the stored decode rows. For
    each generated position record the FIRST op in compute order whose verify activation
    differs in ULP (torch.equal) from the decode row. Hooked op set per layer:
    input_layernorm, qkv_proj (Marlin), q/k/v attn-inputs (post norm+RoPE), attn
    (reduction), o_proj (Marlin), mlp.gate_up_proj (Marlin), mlp.down_proj (Marlin);
    plus the final model.norm (prelmhead). Op families: matmul/GEMM, attn-reduction,
    norm, rope. We also record the end-to-end argmax flip (M=K verify prompt_logprobs
    argmax vs the M=1 decode token) -- the actual "divergent position" signal.

  Phase 2 -- lm_head (and intermediate Marlin) M-dependence microbench, identical input.
    For positions whose prelmhead hidden is byte-identical M=1-vs-M=K (Phase 1 says the
    intermediate stack is clean), feed that EXACT hidden through the deployed Marlin
    GEMMs at M=1 vs M=K and bit-compare the outputs. This PINS the divergence to a
    specific GEMM with a controlled identical input (no position-matching needed) and
    tests the int4-Marlin M-invariance claim (#680/#736/#743) on the DEPLOYED weights.
    For the lm_head we also report the argmax flip + top-2 gap (is every flip a ULP tie?)
    and the targeted-fix probe: does any current toggle (VLLM_MARLIN_USE_ATOMIC_ADD)
    make the lm_head GEMM M-invariant?

Primary metric: top_op_divergence_share = fraction of divergent positions whose FIRST
divergence is the single top op family. test metric: literal_strict_achievable_targeted
= 1 iff a single targeted BI fix on the top op (current toggles only) reaches 128/128.

analysis_only, official_tps=0, no_hf_job. NO served-file change, NO --launch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
CENSUS_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
for p in (str(HERE), str(CENSUS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEPLOYED = "/workspace/gemma_build/int4_g128_lmhead"

# ---- compute-order op table: op_label -> (family, within_layer_rank) ----
# matmul = int4 Marlin GEMM; attn = split-KV reduction; norm = RMSNorm; rope = RoPE
OP_TABLE = {
    "input_ln":   ("norm",   0),
    "qkv":        ("matmul", 1),
    "attn_q":     ("rope",   2),
    "attn_k":     ("rope",   3),
    "attn_v":     ("norm",   4),
    "attn_out":   ("attn",   5),
    "oproj":      ("matmul", 6),
    "mlp_gateup": ("matmul", 7),
    "mlp_down":   ("matmul", 8),
}
PRELM_RANK = 10 ** 6  # final model.norm output (prelmhead hidden)
# matmul sub-op labels for the ranked breakdown
MATMUL_SUBOP = {"qkv": "matmul_qkv", "oproj": "matmul_oproj",
                "mlp_gateup": "matmul_mlp_gateup", "mlp_down": "matmul_mlp_down"}


def _set_env(batch_invariant: int = 1) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # harness pins host idx 2; in-container GPU is 0
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    # BI=1 freezes the attention is_batch_invariant global -> use_3d=False -> num_splits=1
    # for BOTH M=1 decode and M=K+1 verify. BI=0 lets M=1 decode take the 3D (num_splits>1)
    # split-KV path while M=K+1 verify stays 2D (num_splits=1) -> a reduction-order split.
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant else "0"


# ---------- capture state (in-process; module globals are live in the worker) ----------
CAP: dict[str, Any] = {
    "mode": None,         # "store" (M=1 decode) | "diff" (M=K verify)
    "positions": None,    # cpu long tensor [n_tok] for the current forward
    "store": {},          # (op, pos) -> cpu tensor   (M=1 1-row decode rows only)
    "prelm": {},          # pos -> cpu tensor (M=1 decode prelmhead hidden, for Phase 2)
    "pos_div": defaultdict(dict),   # pos -> {global_rank: {op,family,layer,max_abs}}
    "n_decode_rows": 0,
    "shape_mismatch": {},           # (op,li) -> (verify_row_shape, decode_row_shape)
}


def _pos_pre_hook(mod, args, kwargs):
    p = args[0] if args else kwargs.get("positions")
    if p is not None:
        CAP["positions"] = p.detach().to("cpu")
    return None


def _record(li: int, op: str, t) -> None:
    import torch
    pos = CAP["positions"]
    if pos is None or not torch.is_tensor(t) or t.shape[0] != pos.shape[0]:
        return
    n = t.shape[0]
    fam, wrank = OP_TABLE[op]
    grank = li * 10 + wrank
    if CAP["mode"] == "store":
        if n != 1:                      # only M=1 decode forwards
            return
        # key MUST include the layer index: this model is heterogeneous -- the 7
        # full_attention layers emit a 2x-wide qkv/attn (6144 vs the 35 sliding
        # layers' 3072), so an (op,pos) key would let a full-attn row clobber a
        # sliding-attn row at the same position and corrupt the width-matched diff.
        CAP["store"][(op, li, int(pos[0]))] = t[0].detach().to("cpu")
        if op == "attn_out" and li == 0:
            CAP["n_decode_rows"] += 1
    elif CAP["mode"] == "diff":
        tc = None
        for r in range(n):
            key = (op, li, int(pos[r]))
            ref = CAP["store"].get(key)
            if ref is None:
                continue
            if tc is None:
                tc = t.detach().to("cpu")
            row = tc[r]
            if row.shape != ref.shape:   # decode row vs verify row width mismatch -> skip
                CAP["shape_mismatch"][(op, li)] = (tuple(row.shape), tuple(ref.shape))
                continue
            if not torch.equal(row, ref):
                d = (row.float() - ref.float()).abs().max().item()
                CAP["pos_div"][int(pos[r])][grank] = {
                    "op": op, "family": fam, "layer": li, "max_abs": d}


def _prelm_record(t) -> None:
    """final model.norm output: store M=1 decode hidden (Phase 2) + diff for first-div."""
    import torch
    pos = CAP["positions"]
    if pos is None or not torch.is_tensor(t) or t.shape[0] != pos.shape[0]:
        return
    n = t.shape[0]
    if CAP["mode"] == "store":
        if n != 1:
            return
        CAP["prelm"][int(pos[0])] = t[0].detach().to("cpu")
    elif CAP["mode"] == "diff":
        tc = None
        for r in range(n):
            ref = CAP["prelm"].get(int(pos[r]))
            if ref is None:
                continue
            if tc is None:
                tc = t.detach().to("cpu")
            row = tc[r]
            if row.shape != ref.shape:
                CAP["shape_mismatch"][("prelmhead", -1)] = (tuple(row.shape), tuple(ref.shape))
                continue
            if not torch.equal(row, ref):
                d = (row.float() - ref.float()).abs().max().item()
                CAP["pos_div"][int(pos[r])][PRELM_RANK] = {
                    "op": "prelmhead", "family": "norm", "layer": -1, "max_abs": d}


def _hook(li: int, op: str, source: str, idx: int = 0):
    def h(mod, inp, out):
        t = inp[idx] if source == "in" else (out[0] if isinstance(out, tuple) else out)
        _record(li, op, t)
        return None
    return h


def _prelm_hook(mod, inp, out):
    t = out[0] if isinstance(out, tuple) else out
    _prelm_record(t)
    return None


def get_model_runner(llm):
    cands = []
    try:
        cands.append(llm.llm_engine.engine_core.engine_core
                     .model_executor.driver_worker.worker.model_runner)
    except Exception:
        pass
    try:
        ec = llm.llm_engine.engine_core
        ec2 = getattr(ec, "engine_core", ec)
        w = getattr(getattr(ec2, "model_executor", None), "driver_worker", None)
        w = getattr(w, "worker", w)
        mr = getattr(w, "model_runner", None)
        if mr is not None:
            cands.append(mr)
    except Exception:
        pass
    for mr in cands:
        if mr is not None and hasattr(mr, "model"):
            return mr
    raise RuntimeError("could not reach model_runner")


def _wire_hooks(model):
    import re
    handles, layers = [], {}
    # restrict to the TEXT decoder: vision_tower/audio_tower also expose
    # ".layers.N.self_attn" and would collide on the bare layer index.
    name2mod = dict(model.named_modules())
    is_lang = any("language_model" in n for n in name2mod)
    def _lang(n: str) -> bool:
        return (not is_lang) or ("language_model" in n)
    pat = re.compile(r"\.layers\.(\d+)\.self_attn$")
    by_li: dict[int, dict] = defaultdict(dict)
    final_norm = None
    for name, mod in model.named_modules():
        if not _lang(name):
            continue
        m = pat.search(name)
        if m:
            li = int(m.group(1))
            by_li[li]["attn_mod"] = mod
            by_li[li]["qkv"] = getattr(mod, "qkv_proj", None)
            by_li[li]["attn"] = getattr(mod, "attn", None)
            by_li[li]["oproj"] = getattr(mod, "o_proj", None)
            # the DECODER LAYER (parent of self_attn) receives ``positions`` as its
            # first forward arg and runs BEFORE input_layernorm -- so a pos pre-hook
            # here keys layer-0 input_ln correctly (a pre-hook on self_attn fires
            # AFTER input_ln and leaves L0 input_ln keyed to the prior forward's pos).
            layer_name = name[: -len(".self_attn")]
            by_li[li]["layer_mod"] = name2mod.get(layer_name)
        if name.endswith(".input_layernorm"):
            by_li[int(name.split(".layers.")[1].split(".")[0])]["input_ln"] = mod
        if name.endswith(".mlp.gate_up_proj"):
            by_li[int(name.split(".layers.")[1].split(".")[0])]["mlp_gateup"] = mod
        if name.endswith(".mlp.down_proj"):
            by_li[int(name.split(".layers.")[1].split(".")[0])]["mlp_down"] = mod
        if name.endswith(".norm") and ".layers." not in name and "self_attn" not in name:
            final_norm = mod
    for li in sorted(by_li):
        d = by_li[li]
        lm = d.get("layer_mod") or d.get("attn_mod")
        if lm is not None:
            handles.append(lm.register_forward_pre_hook(_pos_pre_hook, with_kwargs=True))
        if d.get("input_ln") is not None:
            handles.append(d["input_ln"].register_forward_hook(_hook(li, "input_ln", "out")))
        if d.get("qkv") is not None:
            handles.append(d["qkv"].register_forward_hook(_hook(li, "qkv", "out")))
        if d.get("attn") is not None:
            ao = d["attn"]
            handles.append(ao.register_forward_hook(_hook(li, "attn_q", "in", 0)))
            handles.append(ao.register_forward_hook(_hook(li, "attn_k", "in", 1)))
            handles.append(ao.register_forward_hook(_hook(li, "attn_v", "in", 2)))
            handles.append(ao.register_forward_hook(_hook(li, "attn_out", "out")))
        if d.get("oproj") is not None:
            handles.append(d["oproj"].register_forward_hook(_hook(li, "oproj", "out")))
        if d.get("mlp_gateup") is not None:
            handles.append(d["mlp_gateup"].register_forward_hook(_hook(li, "mlp_gateup", "out")))
        if d.get("mlp_down") is not None:
            handles.append(d["mlp_down"].register_forward_hook(_hook(li, "mlp_down", "out")))
        layers[li] = True
    if final_norm is not None:
        handles.append(final_norm.register_forward_hook(_prelm_hook))
    return handles, sorted(layers), (final_norm is not None)


def _lmhead_gemm(lm_head, x):
    """Run the deployed int4 Marlin lm_head GEMM on x[M,hidden] -> logits[M,vocab]."""
    qm = getattr(lm_head, "quant_method", None)
    if qm is not None and hasattr(qm, "apply"):
        return qm.apply(lm_head, x, bias=None)
    return lm_head(x)


def _marlin_gemm(linear, x):
    qm = getattr(linear, "quant_method", None)
    out = qm.apply(linear, x, bias=None) if (qm and hasattr(qm, "apply")) else linear(x)
    return out[0] if isinstance(out, tuple) else out


def phase2_microbench(model, prelm_hiddens, verify_width, topk_gap=True):
    """M-dependence of each deployed Marlin GEMM under an IDENTICAL input row.
    prelm_hiddens: list of cpu bf16 hidden vectors [hidden] (real prelmhead rows)."""
    import torch
    dev = next(model.parameters()).device
    # locate modules
    lm_head = None
    sample = {}
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            lm_head = mod
        for key, suf in (("qkv", "layers.0.self_attn.qkv_proj"),
                         ("oproj", "layers.0.self_attn.o_proj"),
                         ("mlp_gateup", "layers.0.mlp.gate_up_proj"),
                         ("mlp_down", "layers.0.mlp.down_proj")):
            if name.endswith(suf):
                sample[key] = mod
    res: dict[str, Any] = {}
    M = max(2, verify_width)
    H = prelm_hiddens[0].shape[0]
    hs = torch.stack(prelm_hiddens[:max(M, len(prelm_hiddens))]).to(dev, torch.bfloat16)  # [N,H]
    N = hs.shape[0]

    # ---- lm_head: the deployed int4 head (the suspect) ----
    n_div = n_flip = n_tie = n_pos = 0
    for i in range(N):
        h = hs[i:i + 1]                                   # [1,H]
        lg1 = _lmhead_gemm(lm_head, h)[0]                 # M=1 path
        # M=K batch with row 0 = h, fill remaining with neighbours
        idx = [i] + [(i + j) % N for j in range(1, M)]
        batch = hs[idx]                                   # [M,H], row0=h
        lgK = _lmhead_gemm(lm_head, batch)[0]             # row 0 logits at M=K
        n_pos += 1
        if not torch.equal(lg1, lgK):
            n_div += 1
            if int(lg1.argmax()) != int(lgK.argmax()):
                n_flip += 1
            if topk_gap:
                top2 = torch.topk(lg1.float(), 2).values
                gap = (top2[0] - top2[1]).item()
                # tie = the M-induced logit perturbation can cross the top-2 gap
                dmax = (lg1.float() - lgK.float()).abs().max().item()
                if gap <= max(dmax, 0) + 1e-6:
                    n_tie += 1
    res["lm_head"] = {"family": "matmul", "subop": "matmul_lmhead", "M": M, "n_pos": n_pos,
                      "n_divergent": n_div, "frac_divergent": (n_div / n_pos) if n_pos else None,
                      "n_argmax_flip": n_flip, "n_tie_within_gap": n_tie}

    # ---- intermediate Marlin GEMMs (reproduce/refute #680/#743 "M-invariant") ----
    for key, lin in sample.items():
        in_dim = None
        for a in ("input_size_per_partition", "input_size"):
            if hasattr(lin, a):
                in_dim = int(getattr(lin, a)); break
        if in_dim is None:
            in_dim = H
        torch.manual_seed(0)
        x1 = torch.randn(1, in_dim, device=dev, dtype=torch.bfloat16)
        xb = torch.randn(M, in_dim, device=dev, dtype=torch.bfloat16)
        xb[0] = x1[0]
        o1 = _marlin_gemm(lin, x1)[0]
        ob = _marlin_gemm(lin, xb)[0]
        div = not torch.equal(o1, ob)
        res[key] = {"family": "matmul", "subop": MATMUL_SUBOP[key], "M": M,
                    "in_dim": in_dim, "divergent_M1_vs_MK": bool(div),
                    "max_abs": (o1.float() - ob.float()).abs().max().item() if div else 0.0}
    return res


def targeted_fix_probe(model, prelm_hiddens, verify_width):
    """Does any CURRENT toggle make the deployed lm_head Marlin GEMM M-invariant?
    Tests VLLM_MARLIN_USE_ATOMIC_ADD on/off in-process by re-importing the marlin
    util gate. (enable_batch_invariant_mode patches only aten mm/addmm/bmm, never the
    custom Marlin op, so BI=1 alone cannot batch-invariant the int4 head.)"""
    import torch
    dev = next(model.parameters()).device
    lm_head = None
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            lm_head = mod
    M = max(2, verify_width)
    hs = torch.stack(prelm_hiddens[:M]).to(dev, torch.bfloat16)
    out = {}
    try:
        import vllm.envs as envs
        for tag, val in (("atomic_add_off", False), ("atomic_add_on", True)):
            try:
                envs.VLLM_MARLIN_USE_ATOMIC_ADD = val  # type: ignore[attr-defined]
            except Exception:
                pass
            os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "1" if val else "0"
            n_div = 0
            for i in range(M):
                h = hs[i:i + 1]
                lg1 = _lmhead_gemm(lm_head, h)[0]
                idx = [i] + [(i + j) % M for j in range(1, M)]
                lgK = _lmhead_gemm(lm_head, hs[idx])[0]
                if not torch.equal(lg1, lgK):
                    n_div += 1
            out[tag] = {"n_div_of_M": n_div, "M": M}
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4, help="spec tokens; verify width = k+1")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--bi", type=int, default=1, choices=(0, 1),
                    help="VLLM_BATCH_INVARIANT: 1=served byte-exact config (num_splits=1 "
                         "both M); 0=opens the attention 3D-decode vs 2D-verify split")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "locus_census" / "report.json")
    args = ap.parse_args()
    _set_env(args.bi)
    verify_width = args.k + 1
    t0 = time.time()

    import math
    import torch
    from vllm import LLM, SamplingParams
    from reduction_sensitivity_census import load_prompts, entry_as_dict

    prompts = load_prompts(args.n_prompts, args.ctx_cap)
    print(f"[locus761] DEPLOYED={DEPLOYED} prompts={len(prompts)} K={args.k} "
          f"verify_width={verify_width} n_new={args.n_new} BI=1 TRITON_ATTN eager", flush=True)

    llm = LLM(model=DEPLOYED, dtype="bfloat16",
              max_model_len=max(1024, args.ctx_cap + args.n_new + 16),
              gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
              max_num_batched_tokens=verify_width, enable_prefix_caching=False,
              enforce_eager=True, trust_remote_code=True, max_logprobs=max(20, args.topk + 2))
    model = get_model_runner(llm).model
    handles, layers, has_prelm = _wire_hooks(model)
    n_layers = len(layers)
    print(f"[locus761] hooked {n_layers} layers, prelmhead={has_prelm}", flush=True)
    assert n_layers > 0 and has_prelm

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.n_new, logprobs=args.topk)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)

    e2e = {"n_pos": 0, "n_argflip": 0}
    flip_positions: list[dict] = []
    prelm_collect: list = []     # representative prelmhead hiddens for Phase 2
    n_gen_total = 0

    for pi, pr in enumerate(prompts):
        ctx = pr["context_token_ids"]
        c = len(ctx)
        # ---- M=1 AR decode: store activations ----
        CAP["mode"] = "store"; CAP["store"].clear(); CAP["prelm"].clear(); CAP["positions"] = None
        out = llm.generate([{"prompt_token_ids": ctx}], gen_sp, use_tqdm=False)[0]
        gen = list(out.outputs[0].token_ids)
        if not gen:
            continue
        n_gen_total += len(gen)
        if len(prelm_collect) < 64 and CAP["prelm"]:
            for p in sorted(CAP["prelm"])[:8]:
                prelm_collect.append(CAP["prelm"][p].clone())
        # ---- M=K verify re-forward: diff activations + e2e argmax flip ----
        CAP["mode"] = "diff"; CAP["positions"] = None
        vout = llm.generate([{"prompt_token_ids": ctx + gen}], ver_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs
        for g in range(len(gen)):
            j = c + g
            if pls is None or j >= len(pls) or pls[j] is None:
                continue
            ref_tok = int(gen[g])
            mv = entry_as_dict(pls[j])
            v_arg = max(mv, key=mv.get) if mv else ref_tok
            e2e["n_pos"] += 1
            if v_arg != ref_tok:
                e2e["n_argflip"] += 1
                fd = min(CAP["pos_div"].get(j, {}), default=None)
                fdinfo = CAP["pos_div"].get(j, {}).get(fd) if fd is not None else None
                flip_positions.append({"prompt": pi, "pos": j, "gen_idx": g,
                                       "first_div": fdinfo})
        print(f"  [{pi+1}/{len(prompts)}] c={c} gen={len(gen)} "
              f"cum_flips={e2e['n_argflip']}/{e2e['n_pos']}", flush=True)

    for h in handles:
        h.remove()

    # ---------- aggregate Phase 1: per-position first-divergence ----------
    # every position that has ANY recorded op-divergence (well-powered, pre-argmax)
    first_by_family: dict[str, int] = defaultdict(int)
    first_by_op: dict[str, int] = defaultdict(int)
    n_div_positions = 0
    for pos, ranks in CAP["pos_div"].items():
        if not ranks:
            continue
        gr = min(ranks)
        info = ranks[gr]
        n_div_positions += 1
        fam = info["family"]
        op = info["op"]
        if fam == "matmul":
            op = MATMUL_SUBOP.get(op, op)
        first_by_family[fam] += 1
        first_by_op[op] += 1

    # ---------- Phase 2: microbench (pins lm_head; tests intermediate Marlin) ----------
    print(f"[locus761] Phase 2 microbench on {len(prelm_collect)} real prelmhead rows", flush=True)
    micro = phase2_microbench(model, prelm_collect, verify_width) if prelm_collect else {}
    fix = targeted_fix_probe(model, prelm_collect, verify_width) if prelm_collect else {}

    # ---- merge Phase 2 lm_head into the first-divergence accounting ----
    # positions clean through prelmhead but divergent at lm_head -> first-div = lm_head.
    # Phase 1 cannot hook past prelmhead, so we ADD the lm_head channel from Phase 2's
    # measured rate applied to the clean-to-prelmhead population (= e2e argmax flips,
    # which by construction first-diverge no earlier than the head when prelm is clean).
    lmhead_div = micro.get("lm_head", {})
    n_flip = e2e["n_argflip"]
    # flips whose first-div is upstream (prelm already diverged):
    flips_upstream = sum(1 for f in flip_positions if f.get("first_div"))
    flips_headonly = n_flip - flips_upstream

    total_div = n_div_positions + flips_headonly
    merged_family = dict(first_by_family)
    merged_op = dict(first_by_op)
    if flips_headonly > 0:
        merged_family["matmul"] = merged_family.get("matmul", 0) + flips_headonly
        merged_op["matmul_lmhead"] = merged_op.get("matmul_lmhead", 0) + flips_headonly

    fam_share = {k: v / total_div for k, v in merged_family.items()} if total_div else {}
    top_family = max(fam_share, key=fam_share.get) if fam_share else None
    top_share = fam_share.get(top_family, 0.0) if top_family else 0.0
    op_ranked = sorted(merged_op.items(), key=lambda kv: -kv[1])

    # ---- per-arm facts (the cross-arm targeted-fix verdict is computed by the
    #      bi_arm orchestrator, which compares the bi=0 locus arm to the bi=1 fix arm) ----
    head_M_dep = bool(lmhead_div.get("n_divergent", 0) > 0)
    inter_M_dep = any(micro.get(k, {}).get("divergent_M1_vs_MK") for k in
                      ("qkv", "oproj", "mlp_gateup", "mlp_down"))
    marlin_M_invariant = not head_M_dep and not inter_M_dep
    atomic_inert_on_head = (fix.get("atomic_add_off", {}).get("n_div_of_M", 1) ==
                            fix.get("atomic_add_on", {}).get("n_div_of_M", 0))
    byte_exact_this_arm = (total_div == 0 and e2e["n_argflip"] == 0)
    # In a SINGLE arm we can only state the locus + whether THIS config is byte-exact.
    # The targeted fix = VLLM_BATCH_INVARIANT=1 (a current toggle): when bi=1 this arm
    # IS byte-exact (0 divergence) and when bi=0 the locus is attention -> the toggle
    # that closes it is BI. literal-strict is realizable iff the bi=1 arm is byte-exact.
    literal_strict_achievable_targeted = 1 if byte_exact_this_arm else 0

    report = {
        "pr": 761, "phase": "served_divergence_locus", "analysis_only": True,
        "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "deployed_ckpt": DEPLOYED, "ckpt_lm_head": "int4_g128_marlin_full_vocab_262144",
        "hub_ckpt_lm_head_for_contrast": "bf16 (gemma-4-E4B-it-qat-w4a16-ct, #743 probe)",
        "config": {"k": args.k, "verify_width": verify_width, "batch_invariant": args.bi,
                   "attn_backend": "TRITON_ATTN", "enforce_eager": True,
                   "n_prompts": len(prompts), "n_new": args.n_new, "ctx_cap": args.ctx_cap,
                   "n_layers": n_layers},
        "phase1_forward_ab": {
            "n_gen_positions": n_gen_total,
            "n_decode_rows_stored": CAP["n_decode_rows"],
            "e2e_positions": e2e["n_pos"],
            "e2e_argmax_flips": e2e["n_argflip"],
            "e2e_argmax_flip_rate": (e2e["n_argflip"] / e2e["n_pos"]) if e2e["n_pos"] else None,
            "n_positions_with_intermediate_divergence": n_div_positions,
            "first_div_by_family_intermediate": dict(first_by_family),
            "first_div_by_op_intermediate": dict(first_by_op),
            "shape_mismatch_ops": {f"{k[0]}@L{k[1]}": v for k, v in CAP["shape_mismatch"].items()},
            "flips_upstream_of_head": flips_upstream,
            "flips_head_only": flips_headonly,
            "flip_sample": flip_positions[:40],
        },
        "phase2_microbench": micro,
        "targeted_fix_probe": fix,
        "attribution": {
            "total_divergent_positions": total_div,
            "first_div_by_family": merged_family,
            "first_div_family_share": fam_share,
            "first_div_by_op_ranked": op_ranked,
            "top_op_family": top_family,
            "top_op_divergence_share": top_share,
        },
        "verdict": {
            "lm_head_int4_marlin_M_dependent": head_M_dep,
            "intermediate_marlin_M_dependent": inter_M_dep,
            "marlin_M_invariant_incl_lm_head": marlin_M_invariant,
            "attention_reduction_is_locus": (top_family == "attn"),
            "atomic_add_inert_on_head": atomic_inert_on_head,
            "byte_exact_this_arm": byte_exact_this_arm,
            "literal_strict_achievable_targeted": literal_strict_achievable_targeted,
        },
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 80, flush=True)
    print(f"[PR761 LOCUS] deployed int4-lm_head | {len(prompts)}p x {args.n_new} | K={args.k}", flush=True)
    print(f"  e2e argmax flips: {e2e['n_argflip']}/{e2e['n_pos']} "
          f"(rate={report['phase1_forward_ab']['e2e_argmax_flip_rate']})", flush=True)
    print(f"  intermediate-op divergent positions: {n_div_positions} "
          f"by_family={dict(first_by_family)}", flush=True)
    if CAP["shape_mismatch"]:
        print(f"  SHAPE-MISMATCH ops (skipped): {dict(CAP['shape_mismatch'])}", flush=True)
    print(f"  Phase2 lm_head: {micro.get('lm_head')}", flush=True)
    print(f"  Phase2 intermediate Marlin M-dep: "
          f"{ {k: micro.get(k, {}).get('divergent_M1_vs_MK') for k in ('qkv','oproj','mlp_gateup','mlp_down')} }",
          flush=True)
    print(f"  ATTRIBUTION top_op_family={top_family} share={top_share:.4f} "
          f"total_div={total_div}", flush=True)
    print(f"  ranked ops: {op_ranked}", flush=True)
    print(f"  targeted_fix_probe={fix}", flush=True)
    print(f"  literal_strict_achievable_targeted={literal_strict_achievable_targeted}", flush=True)
    print(f"  peak_mem={report['peak_mem_gb']}GB elapsed={report['elapsed_s']}s -> {args.out}", flush=True)
    print("=" * 80, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
