#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Deployed LINEAR-path step micro-measurement (PR #278).

WHAT THIS MEASURES (and why it is the deployed LINEAR verify, not the tree verify)
----------------------------------------------------------------------------------
The deployed manifest (fa2sw_precache_kenyan #52) runs LINEAR speculative decode:
the MTP drafter proposes a single chain of K_spec=7 tokens, then ONE target-model
forward verifies the K+1=8 chain positions (a degenerate "M=1" tree -> a single
linear branch, CAUSAL self-attention among the 8 new positions + full attention to
the ctx prefix). This is distinct from the M=8 / M=32 TREE verify (#271 verify8 /
verify32), which carry a tree-structured attention mask over 8 / 32 candidate nodes.

This profiler reuses the #271 ONEGRAPH-faithful setup (served int4 verify body via
vLLM compressed-tensors Marlin at deployed depth=37 + the MTP drafter in one
process, CUDA-graph captured) and CUDA-events, IN ISOLATION:

  (1) target_verify_m1_us  -- the deployed LINEAR verify: full int4 body forward
      over the K+1=8 linear-chain positions with a CAUSAL attention mask (the
      "step minus draft" core component the composition prices against).
  (2) a FULL-SDPA verify cross-check (the #271 verify8 basis) to show the linear
      causal mask vs the unmasked tree-8 forward differ negligibly (the verify is
      int4-body-GEMM dominated, HBM-weight-bound, so attention-mask shape barely
      moves it).
  (3) a batch=8 linear verify (8 seqs x 8 chain positions = 64 rows in one forward)
      to demonstrate the int4 body GEMMs are weight-HBM-bound -> ~M-invariant ->
      the per-forward verify is the SAME whether 8 or 64 rows (the weight read
      dominates), i.e. batch does not inflate the per-forward verify.
  (4) the draft K=7 chain (fresh, same run) -- the deployed draft component.

CUDA events, graphed (ONEGRAPH=1 deployed basis). Greedy / PPL / served files are
UNTOUCHED -- model-loading smoke test + local micro-profiling. No vLLM serve.py, no
HF Job, no submission. Single A10G, CUDA_VISIBLE_DEVICES=0. Writes
linear_verify_measurement.json for the analytic core (linear_step_decomposition.py).
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
K_SPEC = 7                       # num_speculative_tokens -> linear chain of K=7
M_LINEAR = K_SPEC + 1            # 8 verify positions (the K+1 linear chain)
DEPLOYED_BATCH = 8               # deployed serving batch (lawine #246 0qc5lk4y)
M_TREE8 = 8                      # tree-8 cross-check width (#271 verify8 basis)
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
# model loading + locate (identical to #271; proven on this node).
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
                      max_model_len=max(1024, ctx + 64), gpu_memory_utilization=0.60,
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
# drafter per-pass chain (bf16 linears from safetensors; #271 basis).
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
# timing primitives (identical to #271).
# --------------------------------------------------------------------------- #
def _capture(run):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(5):
            run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        run()
    return g


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


def graphed_avg(run, iters, warmup, repeats):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=528)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(_here, "linear_verify_measurement.json"))
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    print(f"[lin] device {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    llm, model_dir, dims, targets, load_errs = load_verify(a.ctx)
    loaded_layers = dims["num_layers"]
    num_layers, depth_src = deployed_depth(loaded_layers)
    print(f"[lin] layers loaded={loaded_layers} DEPLOYED-compose={num_layers} ({depth_src})", flush=True)

    draft_mods, draft_bufs, draft_roles = load_drafter()

    # heavy warmup -> A10G boost clock (match #271)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()

    shapes = dims["shapes"]
    n_h, n_kv, hd, hidden = dims["n_heads"], dims["n_kv"], dims["head_dim"], dims["hidden"]

    # verify runner over M positions x batch B. The int4 body GEMMs process B*M rows
    # (reused across the 37-layer replay; per-layer weights 45MB >> 6MB L2 so each call
    # is a real HBM read). attn: "causal" -> is_causal=True (the deployed linear chain:
    # M new positions attend to all prefix keys + causally among themselves, via the
    # bottom-right-aligned causal of SDPA, EFFICIENT kernel) / "full" -> no mask (#271
    # tree-8 basis). NOTE: an explicit boolean attn_mask forces SDPA's slow math
    # backend (materializes the QK^T matrix), inflating attention ~28% over all 37
    # layers -- a measurement artifact NOT representative of the deployed TRITON_ATTN
    # path; is_causal=True keeps the efficient kernel.
    def make_verify_runner(M, B, attn):
        rows = B * M
        xins = {n: torch.randn(rows, shapes[n][1], dtype=torch.bfloat16, device=dev) for n in shapes}
        applies = [(targets[n].quant_method.apply, targets[n], xins[n]) for n in shapes]
        q = torch.randn(B, n_h, M, hd, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, n_h, a.ctx, hd, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, n_h, a.ctx, hd, dtype=torch.bfloat16, device=dev)
        is_causal = (attn == "causal")
        lm_w = torch.randn(LM_HEAD_VOCAB, hidden, dtype=torch.bfloat16, device=dev) * 0.02
        xlm = torch.randn(rows, hidden, dtype=torch.bfloat16, device=dev)

        def run_verify():
            for _ in range(num_layers):
                for apply, mod, x in applies:
                    apply(mod, x, bias=None)
                F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            torch.matmul(xlm, lm_w.t())
        return run_verify

    def run_draft_pass():
        for (mod, _), b in zip(draft_mods, draft_bufs):
            mod(b)

    def run_draft_k():
        for _ in range(K_SPEC):
            run_draft_pass()

    # the deployed LINEAR verify (causal mask, per-sequence M=8 chain, batch=1)
    run_verify_lin = make_verify_runner(M_LINEAR, 1, "causal")
    # full-SDPA cross-check (the #271 verify8 / tree-8 basis, batch=1)
    run_verify_full8 = make_verify_runner(M_TREE8, 1, "full")
    # batch=8 linear verify (8 seqs x 8 positions = 64 rows -> weight-bound invariance)
    run_verify_lin_b8 = make_verify_runner(M_LINEAR, DEPLOYED_BATCH, "causal")

    half = max(20, a.iters // 2)
    print("[lin] timing deployed LINEAR verify (causal, M=8, B=1) ...", flush=True)
    vlin_means, vlin_cap = graphed_avg(run_verify_lin, half, a.warmup, a.repeats)
    print("[lin] timing FULL-SDPA verify (tree-8 basis, M=8, B=1) ...", flush=True)
    vfull_means, vfull_cap = graphed_avg(run_verify_full8, half, a.warmup, a.repeats)
    print("[lin] timing LINEAR verify batch=8 (64 rows, weight-bound check) ...", flush=True)
    vb8_means, vb8_cap = graphed_avg(run_verify_lin_b8, half, a.warmup, a.repeats)
    print("[lin] timing draft pass + K=7 chain (fresh) ...", flush=True)
    d1_means, d1_cap = graphed_avg(run_draft_pass, a.iters, a.warmup, a.repeats)
    dk_means, dk_cap = graphed_avg(run_draft_k, a.iters, a.warmup, a.repeats)

    vlin = _stats(vlin_means); vfull = _stats(vfull_means); vb8 = _stats(vb8_means)
    d1 = _stats(d1_means); dk = _stats(dk_means)

    # physical floor: a full int4 body forward must read body+lm_head weights once.
    body_params = sum(o * i for (o, i) in shapes.values())
    body_int4_gb = body_params * num_layers * INT4_BYTES / 1e9
    lmhead_gb = LM_HEAD_VOCAB * hidden * BF16_BYTES / 1e9
    hbm_floor_ms = (body_int4_gb + lmhead_gb) / A10G_BW_GBPS * 1e3

    out = {
        "kind": "linear-verify-measurement", "pr": 278, "agent": "denken",
        "model_dir": model_dir, "loaded_num_layers": loaded_layers,
        "deployed_num_layers": num_layers, "deployed_layer_src": depth_src,
        "load_errors": load_errs, "drafter_dir": DRAFTER_DIR, "n_draft_gemms": len(draft_mods),
        "lm_head_vocab": LM_HEAD_VOCAB, "ctx": a.ctx, "K_spec": K_SPEC,
        "M_linear": M_LINEAR, "deployed_batch": DEPLOYED_BATCH,
        "config": {"iters": a.iters, "warmup": a.warmup, "repeats": a.repeats,
                   "graphed": {"verify_linear": vlin_cap, "verify_full8": vfull_cap,
                               "verify_linear_b8": vb8_cap, "draft": d1_cap, "draft_k": dk_cap}},
        # THE headline: deployed linear-path M=1 verify (causal, per-seq M=8 chain)
        "target_verify_m1_us": vlin["mean"], "target_verify_m1_us_stats": vlin,
        # cross-checks
        "verify_full8_us": vfull["mean"], "verify_full8_us_stats": vfull,
        "verify_linear_b8_us": vb8["mean"], "verify_linear_b8_us_stats": vb8,
        "verify_linear_b8_per_seq_us": vb8["mean"] / DEPLOYED_BATCH,
        "linear_vs_full8_ratio": vlin["mean"] / vfull["mean"],
        "b8_over_b1_ratio": vb8["mean"] / vlin["mean"],
        # draft (fresh)
        "draft_pass_us_graphed": d1["mean"], "draft_pass_us_graphed_stats": d1,
        "draft_k7_chain_us_graphed": dk["mean"], "draft_k7_chain_us_stats": dk,
        # physical floor
        "physical_floor": {
            "body_int4_gb": body_int4_gb, "lmhead_bf16_gb": lmhead_gb,
            "a10g_bw_gbps": A10G_BW_GBPS, "verify_hbm_floor_ms": hbm_floor_ms,
            "verify_m1_wall_ms": vlin["mean"] / 1e3,
            "verify_m1_over_floor": (vlin["mean"] / 1e3) / hbm_floor_ms,
        },
    }
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out["peak_gpu_gb"] = peak_gb
    out["nan_clean"] = all(math.isfinite(x) for x in
                           [vlin["mean"], vfull["mean"], vb8["mean"], d1["mean"], dk["mean"]])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print("\n========== DEPLOYED LINEAR-PATH STEP MICRO-MEASUREMENT (PR #278) ==========", flush=True)
    print(f" target_verify_m1 (deployed LINEAR, causal, M=8 B=1) = {vlin['mean']:.1f}us", flush=True)
    print(f"   full-SDPA tree-8 cross-check = {vfull['mean']:.1f}us  "
          f"(linear/full = {out['linear_vs_full8_ratio']:.4f})", flush=True)
    print(f"   batch=8 linear (64 rows) = {vb8['mean']:.1f}us  per-seq = {out['verify_linear_b8_per_seq_us']:.1f}us  "
          f"(b8/b1 = {out['b8_over_b1_ratio']:.3f}, weight-bound => ~1.0)", flush=True)
    print(f" draft_pass = {d1['mean']:.1f}us  draft_K7_chain = {dk['mean']:.1f}us", flush=True)
    print(f" physical floor: verify reads {body_int4_gb:.2f}GB int4 + {lmhead_gb:.3f}GB lm_head "
          f"-> >= {hbm_floor_ms:.2f}ms HBM; verify_m1 = {vlin['mean']/1e3:.2f}ms "
          f"({out['physical_floor']['verify_m1_over_floor']:.2f}x floor)", flush=True)
    print(f" peak VRAM {peak_gb:.1f}GB -> {a.out}", flush=True)
    print("===========================================================================\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
