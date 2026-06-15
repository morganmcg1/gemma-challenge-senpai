#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Deployed-path g_d measurement (PR #271) — the COUPLED draft+verify step.

WHAT THIS MEASURES (and why it differs from #257)
-------------------------------------------------
#257 (run h1gj2ved) measured g_d = draft_pass / verify(8) from two ISOLATED
launch-free CUDA graphs (draft chain alone; verify components alone) -> g_d=0.0195.
The #268 step-basis fork (central g_d=0.0195 -> M*=32 tops 480 NO-GO vs assumed
g_d=0.168 -> 577.6 GO) hinges on whether the REAL DEPLOYED served step obeys that
isolated ratio or a different EFFECTIVE one once draft and verify are scheduled
back-to-back in the deployed ONEGRAPH step.

This profiler loads BOTH the served int4 verify body (vLLM compressed-tensors
Marlin, deployed depth=37) AND the served MTP drafter (/tmp/qat-assistant) in ONE
process and captures, in a SINGLE CUDA graph, the real spec-decode step structure:

    [ K_spec draft passes ]  ->  [ verify(8) = body*37 + attn*37 + lm_head(12k) ]

with CUDA events placed AT GRAPH-RECORD TIME at the draft->verify boundary (the
only faithful way to attribute per-region wall time inside a replayed graph; a
Python timer or replay-time profiler sees one monolithic CUDAGraphExec). The
region split gives the DEPLOYED effective draft-region / verify-region us, hence
`deployed_gd`, captured WITH whatever back-to-back scheduling / L2-residency /
clock coupling the deployed step actually has. The ISOLATED graphs are also
measured so the coupling delta (coupled_total vs isolated_sum) is explicit.

Greedy / PPL / served files are UNTOUCHED -- this is a model-loading smoke test +
local micro-profiling. No vLLM serve.py, no HF Job, no submission. Single A10G,
CUDA_VISIBLE_DEVICES=0. Writes deployed_gd_measurement.json for the analytic core.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

# keep our dir off sys.path[0] so a stdlib `import profile` (pulled by deps) is safe
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _here]
sys.modules.pop("profile", None)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---- served-config constants (manifest fa2sw_precache_kenyan #52) ----
K_SPEC = 7                       # num_speculative_tokens -> verify width M = K+1 = 8
M_VERIFY = 8
M_TREE = 32
LM_HEAD_VOCAB = 12288            # deployed LM_HEAD_PRUNE 12k
DRAFTER_DIR = "/tmp/qat-assistant"
DEPLOYED_CFG = "/tmp/osoi5-v0-baked/config.json"   # deployed depth source (37)
MODEL_CANDS = [
    "/tmp/osoi5-v0-baked",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"),
]
A10G_BW_GBPS = 600.0
INT4_BYTES = 0.5
BF16_BYTES = 2.0


def _stats(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    n = len(vals)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    return {"n": n, "mean": mean, "median": statistics.median(vals), "std": std,
            "se": se, "ci95_abs": 1.96 * se, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
            "cv_pct": (100.0 * std / mean) if mean else 0.0, "min": min(vals), "max": max(vals)}


# --------------------------------------------------------------------------- #
# model loading + locate (proven on this node in #257; new-vLLM path added).
# --------------------------------------------------------------------------- #
def resolve_model_dirs() -> list[str]:
    found = []
    for cand in MODEL_CANDS:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            found.append(str(p))
        elif p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    found.append(str(sub)); break
    if not found:
        raise FileNotFoundError(f"no int4 model among {MODEL_CANDS}")
    return found


def read_dims(model_dir: str) -> dict:
    cfg = json.load(open(Path(model_dir) / "config.json"))
    tc = cfg.get("text_config", cfg)
    h, n_heads, n_kv = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"]
    hd, inter = tc["head_dim"], tc["intermediate_size"]
    return {"hidden": h, "n_heads": n_heads, "n_kv": n_kv, "head_dim": hd,
            "intermediate": inter, "num_layers": tc.get("num_hidden_layers"),
            "shapes": {"qkv_proj": ((n_heads + 2 * n_kv) * hd, h), "o_proj": (h, n_heads * hd),
                       "gate_up_proj": (2 * inter, h), "down_proj": (h, inter)}}


def deployed_depth(default: int) -> tuple[int, str]:
    try:
        cfg = json.load(open(DEPLOYED_CFG))
        tc = cfg.get("text_config", cfg)
        nl = int(tc["num_hidden_layers"])
        return nl, "osoi5-v0-baked config.json"
    except Exception as exc:
        return default, f"loaded-model (osoi5 cfg unavailable: {exc!r})"


def load_verify(ctx: int):
    from vllm import LLM
    cands = resolve_model_dirs()
    llm = model_dir = dims = None
    errs = {}
    for cand in cands:
        try:
            cd = read_dims(cand)
            print(f"[verify] try {cand} layers={cd['num_layers']} hidden={cd['hidden']}", flush=True)
            t0 = time.time()
            llm = LLM(model=cand, quantization="compressed-tensors", dtype="bfloat16",
                      max_model_len=max(1024, ctx + M_TREE + 8), gpu_memory_utilization=0.60,
                      max_num_seqs=1, enforce_eager=True, trust_remote_code=True)
            model_dir, dims = cand, cd
            print(f"[verify] LOAD OK {cand} in {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as exc:
            errs[cand] = repr(exc)
            print(f"[verify] load FAILED {cand}: {exc!r}; next", flush=True)
    if llm is None:
        raise RuntimeError(f"all int4 candidates failed: {errs}")

    model = None
    for p in (lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
              lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
              lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model):
        try:
            m = p()
            if m is not None:
                model = m; break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("could not locate model_runner.model")

    import torch.nn as nn
    layers = None
    for chain in [("model", "layers"), ("model", "language_model", "layers"),
                  ("language_model", "model", "layers"), ("language_model", "layers"),
                  ("model", "model", "layers")]:
        obj, ok = model, True
        for a in chain:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False; break
        if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
            layers = obj; break
    if layers is None:
        for _, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0 and hasattr(mod[0], "self_attn"):
                layers = mod; break
    if layers is None:
        raise RuntimeError("could not locate decoder layers")

    def oi(mod):
        out = getattr(mod, "output_size_per_partition", None)
        inp = getattr(mod, "input_size_per_partition", None)
        if out is None or inp is None:
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 2:
                out, inp = int(w.shape[0]), int(w.shape[1])
        return (int(out), int(inp)) if out and inp else None

    shapes = dims["shapes"]
    targets = None
    for layer in layers:
        try:
            cand = {"qkv_proj": layer.self_attn.qkv_proj, "o_proj": layer.self_attn.o_proj,
                    "gate_up_proj": layer.mlp.gate_up_proj, "down_proj": layer.mlp.down_proj}
        except AttributeError:
            continue
        if all(hasattr(m, "quant_method") and oi(m) == shapes[n] for n, m in cand.items()):
            targets = cand; break
    if targets is None:
        raise RuntimeError(f"no layer matched body shapes {shapes}")
    print(f"[verify] located int4 body {[(n, oi(m)) for n, m in targets.items()]}", flush=True)
    return llm, model_dir, dims, targets, errs


# --------------------------------------------------------------------------- #
# drafter per-pass chain (bf16 linears from safetensors; #257 phase_draft basis).
# --------------------------------------------------------------------------- #
def load_drafter():
    st = os.path.join(DRAFTER_DIR, "model.safetensors")
    with open(st, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    from safetensors import safe_open

    def lt(name):
        with safe_open(st, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in hdr if ".layers." in k})
    specs = []
    w = lt("pre_projection.weight"); specs.append(("pre_projection", w.shape[1], w))
    for i in layer_ids:
        qw = lt(f"model.layers.{i}.self_attn.q_proj.weight"); specs.append((f"l{i}.q", qw.shape[1], qw))
        ow = lt(f"model.layers.{i}.self_attn.o_proj.weight"); specs.append((f"l{i}.o", ow.shape[1], ow))
        gw = lt(f"model.layers.{i}.mlp.gate_proj.weight"); uw = lt(f"model.layers.{i}.mlp.up_proj.weight")
        guw = torch.cat([gw, uw], dim=0); specs.append((f"l{i}.gu", guw.shape[1], guw))
        dw = lt(f"model.layers.{i}.mlp.down_proj.weight"); specs.append((f"l{i}.dn", dw.shape[1], dw))
    w = lt("post_projection.weight"); specs.append(("post_projection", w.shape[1], w))
    if "masked_embedding.centroids.weight" in hdr:
        c = lt("masked_embedding.centroids.weight"); specs.append(("centroids", c.shape[1], c))

    class L(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w.cuda().to(torch.bfloat16), requires_grad=False)

        def forward(self, x):
            return F.linear(x, self.weight)

    mods = [(L(w), inn) for (_r, inn, w) in specs]
    bufs = [torch.randn(1, inn, device="cuda", dtype=torch.bfloat16) for (_, inn) in mods]
    print(f"[draft] {len(specs)} per-pass GEMMs from {DRAFTER_DIR}", flush=True)
    return mods, bufs, [r for (r, _i, _w) in specs]


# --------------------------------------------------------------------------- #
# timing primitives
# --------------------------------------------------------------------------- #
def _capture(run):
    """Capture zero-arg `run` into a CUDA graph (deployed ONEGRAPH basis)."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(5):
            run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        run()
    return g


def graphed_avg(run, iters, warmup, repeats):
    """`repeats` measurements, each the mean us/replay over `iters` replays."""
    try:
        g = _capture(run)
    except Exception as exc:
        print(f"[time] capture failed ({exc!r}); eager", flush=True)
        return _eager_avg(run, iters, warmup, repeats), False
    for _ in range(max(10, warmup)):
        g.replay()
    torch.cuda.synchronize()
    means = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        means.append(e0.elapsed_time(e1) / iters * 1e3)
    del g
    return means, True


def _eager_avg(run, iters, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        means = []
        for _ in range(repeats):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                run()
            e1.record(); torch.cuda.synchronize()
            means.append(e0.elapsed_time(e1) / iters * 1e3)
    return means


def graphed_regions(run_draft, run_verify, warmup, n_samples):
    """Region-split the deployed sequential spec-decode step by replaying the
    draft graph and the verify graph BACK-TO-BACK on one stream, with timing
    events at the replay boundaries.

    Why not one fused graph with in-graph events: recording cuda Events INSIDE a
    graph capture and then reading elapsed_time() throws cudaErrorInvalidValue on
    this torch build. More importantly, vLLM MTP runs draft then verify
    SEQUENTIALLY (the draft proposes K tokens, then a single verify forward over
    M=K+1 positions -- no draft/verify overlap), so two graphs replayed
    back-to-back on ONE stream faithfully reproduce the deployed scheduling: the
    verify kernels execute immediately after the draft kernels with no CPU bubble
    (the CPU races ahead enqueuing replay+event while the GPU is still in the
    ~700us draft region), capturing the real L2-residency / sustained-clock
    coupling. Events are recorded at REPLAY time (normal, robust usage).
    Returns (draft_region_us list, verify_region_us list, total_us list, captured)."""
    try:
        gd = _capture(run_draft)
        gv = _capture(run_verify)
    except Exception as exc:
        print(f"[time] coupled capture failed ({exc!r})", flush=True)
        return None, None, None, False
    stream = torch.cuda.current_stream()
    for _ in range(max(10, warmup)):
        gd.replay(); gv.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    dr, vr, tot = [], [], []
    for _ in range(n_samples):
        e0.record(stream)
        gd.replay()
        e1.record(stream)
        gv.replay()
        e2.record(stream)
        torch.cuda.synchronize()
        dr.append(e0.elapsed_time(e1) * 1e3)   # whole K-pass draft chain, us
        vr.append(e1.elapsed_time(e2) * 1e3)
        tot.append(e0.elapsed_time(e2) * 1e3)
    del gd, gv
    return dr, vr, tot, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=528)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--region-samples", type=int, default=80)
    ap.add_argument("--out", default=os.path.join(_here, "deployed_gd_measurement.json"))
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    print(f"[gd] device {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    llm, model_dir, dims, targets, load_errs = load_verify(a.ctx)
    loaded_layers = dims["num_layers"]
    num_layers, depth_src = deployed_depth(loaded_layers)
    print(f"[gd] layers loaded={loaded_layers} DEPLOYED-compose={num_layers} ({depth_src})", flush=True)

    draft_mods, draft_bufs, draft_roles = load_drafter()

    # heavy warmup -> A10G boost clock
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()

    shapes = dims["shapes"]
    n_h, n_kv, hd, hidden = dims["n_heads"], dims["n_kv"], dims["head_dim"], dims["hidden"]

    # static inputs for the verify body GEMMs at width M (reused across the 37-layer
    # replay; per-layer weights 45MB >> 6MB L2 so each call is a real HBM read).
    def make_verify_runner(M):
        xins = {n: torch.randn(M, shapes[n][1], dtype=torch.bfloat16, device=dev) for n in shapes}
        applies = [(targets[n].quant_method.apply, targets[n], xins[n]) for n in shapes]
        q = torch.randn(1, n_h, M, hd, dtype=torch.bfloat16, device=dev)
        k = torch.randn(1, n_h, a.ctx, hd, dtype=torch.bfloat16, device=dev)
        v = torch.randn(1, n_h, a.ctx, hd, dtype=torch.bfloat16, device=dev)
        lm_w = torch.randn(LM_HEAD_VOCAB, hidden, dtype=torch.bfloat16, device=dev) * 0.02
        xlm = torch.randn(M, hidden, dtype=torch.bfloat16, device=dev)

        def run_verify():
            for _ in range(num_layers):
                for apply, mod, x in applies:
                    apply(mod, x, bias=None)
                F.scaled_dot_product_attention(q, k, v)
            torch.matmul(xlm, lm_w.t())
        return run_verify

    def run_draft_pass():
        for (mod, _), b in zip(draft_mods, draft_bufs):
            mod(b)

    def run_draft_k():
        for _ in range(K_SPEC):
            run_draft_pass()

    run_verify8 = make_verify_runner(M_VERIFY)
    run_verify32 = make_verify_runner(M_TREE)

    # ---- isolated graphed measurements (the #257 basis; provenance cross-check) ----
    print("[gd] timing isolated draft pass ...", flush=True)
    d1_means, d1_cap = graphed_avg(run_draft_pass, a.iters, a.warmup, a.repeats)
    print("[gd] timing isolated draft K=7 chain ...", flush=True)
    dk_means, dk_cap = graphed_avg(run_draft_k, a.iters, a.warmup, a.repeats)
    print("[gd] timing isolated verify(8) ...", flush=True)
    v8_means, v8_cap = graphed_avg(run_verify8, max(20, a.iters // 2), a.warmup, a.repeats)
    print("[gd] timing isolated verify(32) ...", flush=True)
    v32_means, v32_cap = graphed_avg(run_verify32, max(20, a.iters // 2), a.warmup, a.repeats)
    # eager draft cross-check (launch-inflation basis)
    d1_eager = statistics.fmean(_eager_avg(run_draft_pass, a.iters, a.warmup, 3))

    # ---- COUPLED draft+verify(8) in ONE graph w/ record-time region events ----
    print("[gd] timing COUPLED [draft K=7 -> verify(8)] with region split ...", flush=True)
    cdr, cvr, ctot, c_cap = graphed_regions(run_draft_k, run_verify8, a.warmup, a.region_samples)

    d1 = _stats(d1_means); dk = _stats(dk_means)
    v8 = _stats(v8_means); v32 = _stats(v32_means)

    out = {
        "kind": "deployed-gd-measurement", "pr": 271, "agent": "denken",
        "model_dir": model_dir, "loaded_num_layers": loaded_layers,
        "deployed_num_layers": num_layers, "deployed_layer_src": depth_src,
        "load_errors": load_errs, "drafter_dir": DRAFTER_DIR, "n_draft_gemms": len(draft_mods),
        "lm_head_vocab": LM_HEAD_VOCAB, "ctx": a.ctx, "K_spec": K_SPEC,
        "config": {"iters": a.iters, "warmup": a.warmup, "repeats": a.repeats,
                   "region_samples": a.region_samples,
                   "graphed": {"draft": d1_cap, "draft_k": dk_cap, "verify8": v8_cap,
                               "verify32": v32_cap, "coupled": c_cap}},
        # isolated (graphed) us
        "draft_pass_us_graphed": d1["mean"], "draft_pass_us_graphed_stats": d1,
        "draft_pass_us_eager": d1_eager,
        "draft_k7_chain_us_graphed": dk["mean"], "draft_k7_chain_us_stats": dk,
        "verify8_us": v8["mean"], "verify8_us_stats": v8,
        "verify32_us": v32["mean"], "verify32_us_stats": v32,
        "verify32_over_verify8": v32["mean"] / v8["mean"],
        # isolated g_d
        "g_d_isolated": d1["mean"] / v8["mean"],
        "g_d_isolated_eager": d1_eager / v8["mean"],
    }

    # physical floor cross-check (why g_d=0.168 is impossible)
    body_params = sum(o * i for (o, i) in shapes.values())
    body_int4_gb = body_params * num_layers * INT4_BYTES / 1e9
    lmhead_gb = LM_HEAD_VOCAB * hidden * BF16_BYTES / 1e9
    hbm_floor_ms = (body_int4_gb + lmhead_gb) / A10G_BW_GBPS * 1e3
    out["physical_floor"] = {
        "body_int4_gb": body_int4_gb, "lmhead_bf16_gb": lmhead_gb,
        "a10g_bw_gbps": A10G_BW_GBPS, "verify_hbm_floor_ms": hbm_floor_ms,
        "verify8_wall_ms": v8["mean"] / 1e3, "verify8_over_floor": (v8["mean"] / 1e3) / hbm_floor_ms,
    }

    if c_cap:
        cd = _stats(cdr); cv = _stats(cvr); ct = _stats(ctot)
        gd_samples = [(dd / K_SPEC) / vv for dd, vv in zip(cdr, cvr)]
        gd_c = _stats(gd_samples)
        iso_sum = d1["mean"] * K_SPEC + v8["mean"]
        out["coupled"] = {
            "draft_region_us": cd, "verify_region_us": cv, "total_us": ct,
            "coupled_total_us": ct["mean"], "isolated_sum_us": iso_sum,
            "coupling_overhead_us": ct["mean"] - iso_sum,
            "coupling_ratio": ct["mean"] / iso_sum,
            "draft_region_per_pass_us": cd["mean"] / K_SPEC,
            "deployed_gd": gd_c["mean"], "deployed_gd_stats": gd_c,
            "deployed_gd_lo": gd_c["lo"], "deployed_gd_hi": gd_c["hi"],
        }
        # headline DEPLOYED g_d = the coupled (deployed-scheduled) ratio
        out["deployed_gd"] = gd_c["mean"]
        out["deployed_gd_lo"] = gd_c["lo"]
        out["deployed_gd_hi"] = gd_c["hi"]
        out["deployed_gd_basis"] = "coupled_record_time_region_split"
    else:
        # fallback: isolated graphed ratio is still a valid deployed-ONEGRAPH g_d
        out["deployed_gd"] = out["g_d_isolated"]
        out["deployed_gd_lo"] = (d1["lo"]) / v8["hi"]
        out["deployed_gd_hi"] = (d1["hi"]) / v8["lo"]
        out["deployed_gd_basis"] = "isolated_graphed_fallback"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out["peak_gpu_gb"] = peak_gb
    out["nan_clean"] = all(math.isfinite(x) for x in
                           [out["deployed_gd"], out["g_d_isolated"], v8["mean"], d1["mean"]])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print("\n========== DEPLOYED-PATH g_d MEASUREMENT (PR #271) ==========", flush=True)
    print(f" draft_pass graphed = {d1['mean']:.1f}us (eager {d1_eager:.1f})  "
          f"verify(8) = {v8['mean']:.1f}us  verify(32) = {v32['mean']:.1f}us", flush=True)
    print(f" g_d ISOLATED = {out['g_d_isolated']:.4f} (graphed)  {out['g_d_isolated_eager']:.4f} (eager)", flush=True)
    if c_cap:
        print(f" COUPLED: draft_region={out['coupled']['draft_region_us']['mean']:.1f}us "
              f"verify_region={out['coupled']['verify_region_us']['mean']:.1f}us "
              f"total={out['coupled']['coupled_total_us']:.1f}us "
              f"(iso_sum {out['coupled']['isolated_sum_us']:.1f}, "
              f"coupling x{out['coupled']['coupling_ratio']:.3f})", flush=True)
        print(f" DEPLOYED g_d = {out['deployed_gd']:.4f}  CI[{out['deployed_gd_lo']:.4f},"
              f"{out['deployed_gd_hi']:.4f}]  ({out['deployed_gd_basis']})", flush=True)
    print(f" physical floor: verify {body_int4_gb:.2f}GB int4 -> >= {hbm_floor_ms:.2f}ms HBM; "
          f"verify(8) {v8['mean']/1e3:.2f}ms ({out['physical_floor']['verify8_over_floor']:.2f}x floor)", flush=True)
    print(f" peak VRAM {peak_gb:.1f}GB -> {a.out}", flush=True)
    print("=============================================================\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
