#!/usr/bin/env python
"""PR #747 wirbel — strict-#319 route-b: is the forced-M=1 verify ATTENTION
byte-exact with served AR decode?  (the #736 GEMV M-invariance analog, for the
TRITON_ATTN unified-attention kernel that Gemma4 forces on A10G.)

Served stack: vLLM 0.22.0, transformers 5.9.0, VLLM_BATCH_INVARIANT=1, the int4
base (int4_g128_lmhead). Gemma4 forces AttentionBackendEnum.TRITON_ATTN
(heterogeneous head dims 256 sliding / 512 global), so decode (q_len=1) and spec
verify (q_len=K) both flow through ONE kernel: unified_attention. Under
VLLM_BATCH_INVARIANT, that kernel runs its 2D single-pass path (no split-KV);
verify (max_seqlen_q>1) is ALWAYS 2D. So the load-bearing question reduces to:
does a query row's attention output depend on how many query rows (q_len) share
the launch?

METHOD (maximally faithful — no hand-built tensors):
  1. Load the int4 base in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) with
     VLLM_BATCH_INVARIANT=1. Monkeypatch the module fn
     vllm.v1.attention.backends.triton_attn.unified_attention so every REAL
     decode call (max_seqlen_q==1, one seq) records: the exact query tensor, the
     reference attention-block output the served kernel produced, and the exact
     kwargs (scale, window, softcap, descales, kv_quant_mode, sinks, ...). We
     keep a live handle to each layer's paged (k,v) cache + block_table.
  2. After greedily decoding a prompt, snapshot that sequence's cache blocks and
     replay, for K candidate tokens (K in {2,4,6}) at many decode positions:
       (a) batched verify  -> ONE unified_attention call, q_len=K (the path
           land #680 says diverges), and
       (b) forced-M=1 verify -> K sequential q_len=1 calls (the decode path).
     Every non-q/seqlen kwarg is reused byte-identically from the captured
     decode call, so only the query-batching changes.
  3. Compare each verify output row against the AR-decode reference bit-for-bit
     (int16 view of the raw bf16 bytes + max|Δ|), exactly like #736.

ANALYSIS ONLY: analysis_only=1, official_tps=0, no_hf_job=1, fires=0. Assigned
local A10G only. Touches NO served file and does not change the locked
int4_g128_lmhead@126.378.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process worker
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # local curand.h gap
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = "/workspace/gemma_build/int4_g128_lmhead"
FULL_ATTENTION_LAYERS = {5, 11, 17, 23, 29, 35, 41}  # from config layer_types

PROMPTS = [
    "Explain step by step how a hydroelectric dam converts water flow into "
    "electricity, and mention three environmental trade-offs.",
    "Write a short story about a lighthouse keeper who discovers a message in "
    "a bottle from the future.",
    "Summarize the causes of the 1929 stock market crash and compare them to "
    "modern financial regulation.",
    "Describe the differences between TCP and UDP, then give two concrete "
    "examples of applications that prefer each.",
]

# ------------------------------------------------------------------ capture ---
_CAP = {
    "on": False,
    "layer_of_id": {},      # id(layer_module) -> layer_idx (first-seen order)
    "name_of_lid": {},      # layer_idx -> layer.layer_name (ground-truth index)
    "cur_lid": None,        # set by the forward() wrapper for the active layer
    "live": {},             # layer_idx -> (k, v, block_table_latest)
    "steps": {},            # layer_idx -> {pos -> {"q":cpu, "ref":cpu}}
    "kw": {},               # layer_idx -> captured static kwargs (cloned)
    "real": None,           # real module-global unified_attention
    "real_forward": None,   # real TritonAttentionImpl.forward
}


def _patched_forward(self, layer, *a, **kw):
    """Wrap TritonAttentionImpl.forward so the active layer module's identity is
    known when its unified_attention() call fires. The `layer` nn.Module is
    persistent (stable id), unlike the per-call kv_cache.unbind() views."""
    prev = _CAP["cur_lid"]
    lid = _CAP["layer_of_id"].setdefault(id(layer), len(_CAP["layer_of_id"]))
    if lid not in _CAP["name_of_lid"]:
        _CAP["name_of_lid"][lid] = getattr(layer, "layer_name", f"<lid{lid}>")
    _CAP["cur_lid"] = lid
    try:
        return _CAP["real_forward"](self, layer, *a, **kw)
    finally:
        _CAP["cur_lid"] = prev


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return x


def _patched_unified_attention(*args, **kw):
    real = _CAP["real"]
    # All call sites in triton_attn.forward use kwargs; be defensive anyway.
    real(*args, **kw)  # writes kw["out"] in place
    if not _CAP["on"] or _CAP["cur_lid"] is None:
        return
    try:
        seqused_k = kw["seqused_k"]
        max_q = kw["max_seqlen_q"]
        if int(max_q) != 1 or seqused_k.shape[0] != 1:
            return  # only single-seq decode steps are the AR reference
        lid = _CAP["cur_lid"]  # stamped by the forward() wrapper for this layer
        k = kw["k"]
        _CAP["live"][lid] = (k, kw["v"], kw["block_table"])
        if lid not in _CAP["kw"]:
            _CAP["kw"][lid] = {
                "softmax_scale": kw["softmax_scale"],
                "causal": kw["causal"],
                "window_size": kw["window_size"],
                "softcap": kw["softcap"],
                "q_descale": _clone(kw.get("q_descale")),
                "k_descale": _clone(kw.get("k_descale")),
                "v_descale": _clone(kw.get("v_descale")),
                "alibi_slopes": kw.get("alibi_slopes"),
                "use_alibi_sqrt": kw.get("use_alibi_sqrt", False),
                "sinks": _clone(kw.get("sinks")),
                "kv_quant_mode": kw.get("kv_quant_mode"),
                "chunk_lookback": kw.get("chunk_lookback", -1),
                "use_td": kw.get("use_td", False),
                "head_size": int(kw["q"].shape[2]),
                "num_heads": int(kw["q"].shape[1]),
                "num_kv_heads": int(kw["k"].shape[2]),
                "block_size": int(kw["k"].shape[1]),
                "is_global": int(seqused_k[0]) and bool(kw["window_size"][0] < 0),
                # 3D split-KV scratch (persistent per-KV-group buffers, shape
                # depends on headdim_padded so capture per layer). Reusing these
                # in replay lets a q_len=1 call take the real 3D decode path
                # under BI=0 (a "faithful forced-M=1 verify").
                "seq_threshold_3D": kw.get("seq_threshold_3D"),
                "num_par_softmax_segments": kw.get("num_par_softmax_segments"),
                "segm_output": kw.get("softmax_segm_output"),
                "segm_max": kw.get("softmax_segm_max"),
                "segm_expsum": kw.get("softmax_segm_expsum"),
            }
        pos = int(seqused_k[0].item()) - 1
        _CAP["steps"].setdefault(lid, {})[pos] = {
            "q": kw["q"].detach().to("cpu", copy=True),
            "ref": kw["out"].detach().to("cpu", copy=True),
        }
    except Exception as e:  # never break generation
        print(f"[capture] WARN {type(e).__name__}: {e}", flush=True)


# ------------------------------------------------------------------ replay ----
def _bitdiff(a: torch.Tensor, b: torch.Tensor):
    """Raw-byte diff of two bf16 tensors (int16 bit view) + max abs fp diff."""
    assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
    ai = a.reshape(-1).view(torch.int16)
    bi = b.reshape(-1).view(torch.int16)
    nbit = int((ai != bi).sum().item())
    maxd = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
    return nbit, maxd, a.numel()


def _call_unified(uni, q, k, v, seqlen_k, kwc, dev, faithful=False):
    """One unified_attention call: q [M,H,D] against (k,v) snapshot, a single
    sequence of length seqlen_k, query rows at positions [seqlen_k-M .. seqlen_k-1].
    Reuses captured static kwargs verbatim. When `faithful` and the captured 3D
    scratch is present, pass it too so a q_len=1 call takes the real 3D split-KV
    decode path under BI=0 (the served decode path). With q_len>1 or under BI=1,
    use_3d is forced False internally regardless, so it stays the 2D path."""
    M = q.shape[0]
    nblocks = k.shape[0]
    out = torch.empty_like(q)
    cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
    seqused = torch.tensor([seqlen_k], dtype=torch.int32, device=dev)
    bt = torch.arange(nblocks, dtype=torch.int32, device=dev).view(1, nblocks)
    extra = {}
    if faithful and kwc.get("segm_output") is not None:
        extra = {
            "seq_threshold_3D": kwc["seq_threshold_3D"],
            "num_par_softmax_segments": kwc["num_par_softmax_segments"],
            "softmax_segm_output": kwc["segm_output"],
            "softmax_segm_max": kwc["segm_max"],
            "softmax_segm_expsum": kwc["segm_expsum"],
        }
    uni(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu, max_seqlen_q=M, seqused_k=seqused, max_seqlen_k=seqlen_k,
        softmax_scale=kwc["softmax_scale"], causal=kwc["causal"],
        window_size=kwc["window_size"], block_table=bt, softcap=kwc["softcap"],
        q_descale=kwc["q_descale"], k_descale=kwc["k_descale"],
        v_descale=kwc["v_descale"], alibi_slopes=kwc["alibi_slopes"],
        use_alibi_sqrt=kwc["use_alibi_sqrt"], sinks=kwc["sinks"],
        kv_quant_mode=kwc["kv_quant_mode"], chunk_lookback=kwc["chunk_lookback"],
        use_td=kwc["use_td"], **extra,
    )
    return out


def snapshot_layer(lid, dev):
    """Compact the current sequence's cache blocks into a standalone (k,v)."""
    k, v, bt = _CAP["live"][lid]
    kwc = _CAP["kw"][lid]
    bsz = kwc["block_size"]
    positions = sorted(_CAP["steps"][lid].keys())
    L = positions[-1] + 1
    nblocks = math.ceil(L / bsz)
    phys = bt[0, :nblocks].to(torch.long)
    k_snap = k[phys].clone()
    v_snap = v[phys].clone()
    return k_snap, v_snap, L, positions


def _ltype(kwc):
    """global (full) attention has no sliding window -> window_size[0] < 0."""
    ws = kwc["window_size"]
    return "global" if (ws is None or int(ws[0]) < 0) else "sliding"


def run_prompt_replays(lid, ks, stride, uni, dev, results):
    kwc = _CAP["kw"][lid]
    k_snap, v_snap, L, positions = snapshot_layer(lid, dev)
    pset = set(positions)
    p0 = positions[0]
    ltype = _ltype(kwc)

    # sanity: faithful forced-M=1 (path c) must equal the captured AR ref byte
    # for byte (it re-runs the exact decode forward: same q, KV, path).
    for p in positions[: min(3, len(positions))]:
        st = _CAP["steps"][lid][p]
        q1 = st["q"].to(dev)
        oc = _call_unified(uni, q1, k_snap, v_snap, p + 1, kwc, dev, faithful=True)
        nbit, maxd, n = _bitdiff(oc.cpu(), st["ref"])
        results["sanity_b_vs_ref"].append(
            {"layer": lid, "ltype": ltype, "pos": p, "nbit": nbit, "maxd": maxd}
        )

    for K in ks:
        # window starts: need K consecutive captured positions p..p+K-1
        starts = [p for p in range(p0, p0 + L) if all((p + j) in pset for j in range(K))]
        starts = starts[::stride]
        for p in starts:
            qs = [_CAP["steps"][lid][p + j]["q"].to(dev) for j in range(K)]
            refs = [_CAP["steps"][lid][p + j]["ref"] for j in range(K)]
            # (a) batched verify: one q_len=K call (real spec verify; 2D path).
            qa = torch.cat(qs, dim=0)
            oa = _call_unified(uni, qa, k_snap, v_snap, p + K, kwc, dev, faithful=True).cpu()
            # (b) naive-2D forced-M=1: K sequential q_len=1 calls, 2D single-pass
            #     (what a standalone 2D-only verify kernel would compute).
            # (c) faithful forced-M=1: K sequential q_len=1 calls down the REAL
            #     decode path (3D split-KV under BI=0) -> the route-b kernel.
            ob_rows, oc_rows = [], []
            for j in range(K):
                ob_rows.append(
                    _call_unified(uni, qs[j], k_snap, v_snap, p + j + 1, kwc, dev).cpu()
                )
                oc_rows.append(
                    _call_unified(uni, qs[j], k_snap, v_snap, p + j + 1, kwc, dev,
                                  faithful=True).cpu()
                )
            for j in range(K):
                na, da, n = _bitdiff(oa[j : j + 1], refs[j])
                nb, db, _ = _bitdiff(ob_rows[j], refs[j])
                nc, dc, _ = _bitdiff(oc_rows[j], refs[j])
                # pure batching effect: batched row j vs forced-M=1 row j (both
                # take the SAME 2D single-pass path; differ only in q_len). Any
                # divergence here is the query-batching axis, isolated from the
                # decode reference's path (which may be 3D split-KV under BI=0).
                nab, dab, _ = _bitdiff(oa[j : j + 1], ob_rows[j])
                results["rows"].append(
                    {"layer": lid, "ltype": ltype, "K": K, "pos": p + j,
                     "ctx": p + j, "row_in_block": j,
                     "a_nbit": na, "a_maxd": da, "b_nbit": nb, "b_maxd": db,
                     "c_nbit": nc, "c_maxd": dc,
                     "ab_nbit": nab, "ab_maxd": dab, "n_elem": n}
                )


# ------------------------------------------------------------------ timing ----
def time_paths(lid, K, uni, dev, iters=200):
    kwc = _CAP["kw"][lid]
    k_snap, v_snap, L, positions = snapshot_layer(lid, dev)
    # pick a mid position with room for K
    pos_set = set(positions)
    cands = [p for p in positions if all((p + j) in pos_set for j in range(K))]
    if not cands:
        return None  # not enough consecutive captured positions for this K
    p = cands[len(cands) // 2]
    qs = [_CAP["steps"][lid][p + j]["q"].to(dev) for j in range(K)]
    qa = torch.cat(qs, dim=0)

    def once_a():  # batched M=K verify (one 2D launch)
        _call_unified(uni, qa, k_snap, v_snap, p + K, kwc, dev, faithful=True)

    def once_c():  # faithful forced-M=1 verify: K sequential real-decode-path calls
        for j in range(K):
            _call_unified(uni, qs[j], k_snap, v_snap, p + j + 1, kwc, dev, faithful=True)

    for _ in range(20):
        once_a(); once_c()
    torch.cuda.synchronize()

    def bench(fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters  # ms per verify

    return {"K": K, "layer": lid, "ltype": _ltype(kwc),
            "a_ms": bench(once_a), "c_ms": bench(once_c),
            "pos": p, "head_size": kwc["head_size"]}


# ------------------------------------------------------------------ main ------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default="2,4,6")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "verify_attn_report.json")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"[run] device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)} "
          f"BI={os.environ.get('VLLM_BATCH_INVARIANT')}", flush=True)

    import vllm.v1.attention.backends.triton_attn as ta
    from vllm import LLM, SamplingParams

    _CAP["real"] = ta.unified_attention
    ta.unified_attention = _patched_unified_attention
    _CAP["real_forward"] = ta.TritonAttentionImpl.forward
    ta.TritonAttentionImpl.forward = _patched_forward

    llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True)
    uni = _CAP["real"]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    results = {"rows": [], "sanity_b_vs_ref": [], "timing": []}
    layer_meta = {}

    for pi, prompt in enumerate(PROMPTS[: args.n_prompts]):
        # reset per-prompt capture (cache blocks get reused across prompts)
        _CAP["steps"].clear(); _CAP["live"].clear()
        _CAP["on"] = True
        out = llm.generate([prompt], sp)
        _CAP["on"] = False
        ntok = len(out[0].outputs[0].token_ids)
        lids = sorted(_CAP["steps"].keys())
        nlayers = len(lids)
        gl = [l for l in lids if _ltype(_CAP["kw"][l]) == "global"]
        sl = [l for l in lids if _ltype(_CAP["kw"][l]) == "sliding"]
        print(f"[run] prompt {pi}: gen {ntok} tok, captured {nlayers} layers "
              f"({len(sl)} sliding, {len(gl)} global)", flush=True)
        if pi == 0:
            print(f"[run] global lids={gl} (names: "
                  f"{[_CAP['name_of_lid'].get(l) for l in gl]})", flush=True)
        # replay every captured layer for this prompt
        for lid in lids:
            if lid not in layer_meta:
                layer_meta[lid] = _CAP["kw"][lid]
            run_prompt_replays(lid, ks, args.stride, uni, dev, results)
        # timing once (first prompt) on a representative sliding + global layer
        if pi == 0:
            tlids = ([sl[0]] if sl else []) + ([gl[0]] if gl else [])
            for lid in tlids:
                for K in ks:
                    t = time_paths(lid, K, uni, dev)
                    if t is not None:
                        results["timing"].append(t)

    # ---- aggregate verdicts -------------------------------------------------
    def agg(rows, key_prefix, sub=None):
        sel = rows if sub is None else [r for r in rows if r["ltype"] == sub]
        if not sel:
            return None
        return {
            "n": len(sel),
            "max_nbit": max(r[f"{key_prefix}_nbit"] for r in sel),
            "max_maxd": max(r[f"{key_prefix}_maxd"] for r in sel),
            "frac_rows_nonzero": sum(1 for r in sel if r[f"{key_prefix}_nbit"] > 0) / len(sel),
        }

    per_k = {}
    for K in ks:
        kr = [r for r in results["rows"] if r["K"] == K]
        per_k[K] = {
            "a_batched_vs_ar": agg(kr, "a"),
            "b_forcedm1_2d_vs_ar": agg(kr, "b"),
            "c_forcedm1_faithful_vs_ar": agg(kr, "c"),
            "ab_batching_effect": agg(kr, "ab"),
            "a_by_type": {t: agg(kr, "a", t) for t in ("sliding", "global")},
            "c_by_type": {t: agg(kr, "c", t) for t in ("sliding", "global")},
            "ab_by_type": {t: agg(kr, "ab", t) for t in ("sliding", "global")},
        }

    # route-b verdict keys on the FAITHFUL forced-M=1 (path c = real decode path).
    forced_m1_max_bitdiff = max((r["c_nbit"] for r in results["rows"]), default=0)
    forced_m1_2d_max_bitdiff = max((r["b_nbit"] for r in results["rows"]), default=0)
    batched_max_bitdiff = max((r["a_nbit"] for r in results["rows"]), default=0)
    batching_effect_max_bitdiff = max((r["ab_nbit"] for r in results["rows"]), default=0)
    sanity_max = max((r["nbit"] for r in results["sanity_b_vs_ref"]), default=0)

    verdict = (
        "VERIFY_ATTN_BYTE_EXACT" if forced_m1_max_bitdiff == 0
        else "VERIFY_ATTN_RESIDUAL_DIVERGENCE"
    )

    report = {
        "pr": 747, "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "backend": "TRITON_ATTN (unified_attention)",
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
        "ks": ks, "n_rows": len(results["rows"]),
        "verify_attn_forced_m1_max_bitdiff": forced_m1_max_bitdiff,
        "verify_attn_forced_m1_2d_max_bitdiff": forced_m1_2d_max_bitdiff,
        "verify_attn_batched_max_bitdiff": batched_max_bitdiff,
        "verify_attn_batching_effect_max_bitdiff": batching_effect_max_bitdiff,
        "sanity_c_vs_ref_max_bitdiff": sanity_max,
        "n_layers_captured": len(layer_meta),
        "name_of_lid": _CAP["name_of_lid"],
        "global_lids": sorted(l for l in layer_meta if _ltype(layer_meta[l]) == "global"),
        "expected_global_layers": sorted(FULL_ATTENTION_LAYERS),
        "verdict": verdict,
        "per_k": per_k,
        "timing": results["timing"],
        "layer_meta": {str(k): {kk: (str(vv) if isinstance(vv, torch.Tensor) else vv)
                                 for kk, vv in v.items()
                                 if kk in ("head_size", "num_heads", "num_kv_heads",
                                           "block_size", "window_size", "softmax_scale")}
                       for k, v in layer_meta.items()},
        "rows": results["rows"],
        "sanity": results["sanity_b_vs_ref"],
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[run] === VERDICT: {verdict} ===", flush=True)
    print(f"[run] BI={os.environ.get('VLLM_BATCH_INVARIANT')} | "
          f"forced_m1 FAITHFUL (path c) max bitdiff vs AR : {forced_m1_max_bitdiff}", flush=True)
    print(f"[run] forced_m1 naive-2D (path b) max bitdiff vs AR: {forced_m1_2d_max_bitdiff}", flush=True)
    print(f"[run] batched   (path a) max bitdiff vs AR : {batched_max_bitdiff}", flush=True)
    print(f"[run] batching-effect (a vs b)  max bitdiff: {batching_effect_max_bitdiff}", flush=True)
    print(f"[run] sanity c-vs-ref max bitdiff          : {sanity_max}", flush=True)
    for K in ks:
        pk = per_k[K]
        a = pk["a_batched_vs_ar"]; b = pk["b_forcedm1_2d_vs_ar"]
        c = pk["c_forcedm1_faithful_vs_ar"]; ab = pk["ab_batching_effect"]
        print(f"[run] K={K}: (a)batched nbit_max={a['max_nbit']} frac={a['frac_rows_nonzero']:.3f} "
              f"| (b)M1-2D nbit_max={b['max_nbit']} frac={b['frac_rows_nonzero']:.3f} "
              f"| (c)M1-faithful nbit_max={c['max_nbit']} frac={c['frac_rows_nonzero']:.3f} "
              f"| (a-vs-b)batch nbit_max={ab['max_nbit']}", flush=True)
    for t in results["timing"]:
        print(f"[run] timing L{t['layer']}({t['ltype']},hs{t['head_size']}) K={t['K']}: "
              f"a_batched={t['a_ms']*1000:.2f}us c_forcedM1={t['c_ms']*1000:.2f}us "
              f"ratio={t['c_ms']/t['a_ms']:.2f}x", flush=True)
    print(f"[run] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
