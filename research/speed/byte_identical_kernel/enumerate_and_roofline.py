#!/usr/bin/env python3
"""PR #550 Stage 1 (PRIMARY): enumerate the kernels the base_fullhead stack
actually dispatches at M=1 decode, and roofline the dominant byte-read ops.

base_fullhead = full native 262144-row bf16 lm_head (tied, in the quant `ignore`
list) + intact int4 W4A16 (compressed-tensors, group_size=32) body, served as
fern #535. This is the quality-safe slow ship (253.78 TPS local, lawine #544).

The kernel lever (PR #550) is orthogonal to lawine #544's precision lever: a
*different, faster* kernel that is ALSO byte-identical (preserves the exact bf16
argmax) is a quality-free + identity-safe TPS gain by construction. This script
answers the decisive first question: WHICH kernels are dispatched, and are the
dominant byte-read ops (int4 body Marlin GEMM, bf16 262k head) already at the
HBM-bandwidth floor (=> no faster byte-identical kernel possible) or is there
recoverable overhead?

analysis_only=true, official_tps=0. LOCAL serve + local profile only. No HF Job,
no submission, no served-file change. M=1 AR (spec-off) is the cleanest kernel
isolation and is exactly the #319 greedy-identity reference path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
A10G_HBM_GBS = 600.0  # A10G HBM bandwidth (same anchor as draft_head_vocab_roofline / #506)
LOCAL_CKPT = (
    "/senpai-run/home/student-denken/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
FULL_LM_HEAD_ROWS = 262144


# --------------------------------------------------------------------------- #
# self-test: roofline arithmetic only (no GPU, no model)                       #
# --------------------------------------------------------------------------- #
def roofline_us(num_bytes: float, gbs: float = A10G_HBM_GBS) -> float:
    return num_bytes / (gbs * 1e9) * 1e6  # microseconds


def self_test() -> bool:
    ok = True
    # bf16 262144x2560 head = 1.342 GB -> ~2235 us at 600 GB/s
    head_bytes = FULL_LM_HEAD_ROWS * 2560 * 2
    us = roofline_us(head_bytes)
    ok &= abs(head_bytes / 1e9 - 1.342) < 0.01
    ok &= abs(us - 2235.0) < 30.0
    # int4 (4-bit) 2560x4096 packed weight ~ 5.24 MB -> ~8.7 us
    body_bytes = 2560 * 4096 * 0.5
    ok &= abs(roofline_us(body_bytes) - 8.74) < 0.5
    print(f"[self-test] head_bytes={head_bytes/1e9:.3f}GB head_floor_us={us:.1f} "
          f"body_floor_us={roofline_us(body_bytes):.2f} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# engine introspection helpers                                                 #
# --------------------------------------------------------------------------- #
def get_model_runner(llm):
    """Reach the V1 GPUModelRunner with multiprocessing disabled. Try the
    documented path then fall back to a breadth search for a `.model` attr."""
    cands = []
    try:
        cands.append(llm.llm_engine.engine_core.engine_core
                     .model_executor.driver_worker.worker.model_runner)
    except Exception:  # noqa: BLE001
        pass
    try:
        ec = llm.llm_engine.engine_core
        for path in ("engine_core",):
            ec = getattr(ec, path, ec)
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


def short(obj) -> str:
    return type(obj).__name__


def find_first(model, name_substr, need_attr=None):
    for name, mod in model.named_modules():
        if name_substr in name:
            if need_attr is None or hasattr(mod, need_attr):
                return name, mod
    return None, None


def enumerate_kernels(model_runner) -> dict:
    import torch
    model = model_runner.model
    out: dict = {}

    # ---- find a representative body quantized Linear (gate_up / qkv) ----
    # NB: quantized linears carry weight_packed/qweight buffers, NOT a `.weight`
    # Parameter, so do not filter on hasattr(mod, "weight").
    body_name, body_lin = None, None
    for name, mod in model.named_modules():
        if "layers.0" in name and hasattr(mod, "quant_method") and ("proj" in name):
            qm = type(mod.quant_method).__name__
            if "Unquantized" not in qm:
                body_name, body_lin = name, mod
                break
    if body_lin is None:  # fall back: any quantized proj anywhere
        for name, mod in model.named_modules():
            if hasattr(mod, "quant_method") and ("proj" in name) and ("layers." in name):
                qm = type(mod.quant_method).__name__
                if "Unquantized" not in qm:
                    body_name, body_lin = name, mod
                    break
    out["body_linear_name"] = body_name
    if body_lin is not None:
        qm = body_lin.quant_method
        info = {"linear_class": short(body_lin), "quant_method": short(qm)}
        # compressed-tensors scheme + kernel
        for a in ("scheme",):
            if hasattr(qm, a):
                info["scheme"] = short(getattr(qm, a))
                sch = getattr(qm, a)
                for k in ("kernel",):
                    if hasattr(sch, k):
                        info["kernel"] = short(getattr(sch, k))
        if hasattr(qm, "kernel"):
            info["kernel"] = short(getattr(qm, "kernel"))
        # weight tensor inventory (for body byte floor)
        wbytes = 0
        wparams = {}
        for pn, p in body_lin.named_parameters(recurse=False):
            wbytes += p.numel() * p.element_size()
            wparams[pn] = {"shape": list(p.shape), "dtype": str(p.dtype),
                           "elt": p.element_size()}
        # also named_buffers (packed qweight may be a buffer)
        for pn, p in body_lin.named_buffers(recurse=False):
            wbytes += p.numel() * p.element_size()
            wparams[pn] = {"shape": list(p.shape), "dtype": str(p.dtype),
                           "elt": p.element_size()}
        info["weight_bytes"] = wbytes
        info["weight_tensors"] = wparams
        out["body_kernel"] = info

    # ---- lm_head (the bf16 262k head) ----
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        _, lm_head = find_first(model, "lm_head")
    if lm_head is not None:
        w = getattr(lm_head, "weight", None)
        hd = {"class": short(lm_head),
              "quant_method": short(lm_head.quant_method) if hasattr(lm_head, "quant_method") else None}
        if w is not None:
            hd["weight_shape"] = list(w.shape)
            hd["weight_dtype"] = str(w.dtype)
            hd["weight_bytes"] = w.numel() * w.element_size()
            hd["rows"] = int(w.shape[0])
        for a in ("num_embeddings", "org_num_embeddings", "embedding_dim"):
            if hasattr(lm_head, a):
                hd[a] = int(getattr(lm_head, a))
        out["lm_head"] = hd
        out["lm_head_rows"] = hd.get("rows", hd.get("num_embeddings"))
    # logits processor
    lp = getattr(model, "logits_processor", None)
    if lp is not None:
        out["logits_processor"] = {
            "class": short(lp),
            "scale": float(getattr(lp, "scale", 1.0)),
            "vocab_size": int(getattr(lp, "vocab_size", -1)),
            "org_vocab_size": int(getattr(lp, "org_vocab_size", -1)),
            "soft_cap": getattr(lp, "soft_cap", None),
        }

    # ---- attention backend ----
    attn_name, attn_mod = find_first(model, "self_attn.attn", "impl")
    if attn_mod is None:
        attn_name, attn_mod = find_first(model, ".attn", "impl")
    if attn_mod is not None:
        ad = {"module": attn_name, "attn_class": short(attn_mod),
              "impl_class": short(attn_mod.impl) if hasattr(attn_mod, "impl") else None}
        for a in ("backend",):
            if hasattr(attn_mod, a):
                ad["backend"] = str(getattr(attn_mod, a))
        out["attention"] = ad
    # model_runner-level attn backend
    for a in ("attn_backend", "attn_backends"):
        if hasattr(model_runner, a):
            out["model_runner_attn_backend"] = str(getattr(model_runner, a))
            break
    out["env_VLLM_ATTENTION_BACKEND"] = os.environ.get("VLLM_ATTENTION_BACKEND", "<unset/default>")
    return out, body_lin, lm_head


# --------------------------------------------------------------------------- #
# CUDA-event timing                                                            #
# --------------------------------------------------------------------------- #
def time_op(fn, iters=200, warmup=40):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(iters)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in evs)
    return {"median_ms": statistics.median(ms), "min_ms": ms[0],
            "p10_ms": ms[max(0, len(ms)//10)], "n": iters}


def roofline_head(lm_head):
    import torch
    import torch.nn.functional as F
    w = lm_head.weight
    H = w.shape[1]
    x = torch.randn(1, H, dtype=w.dtype, device=w.device)
    # raw GEMM that reads the head weight once (the bandwidth-bound op)
    t_linear = time_op(lambda: F.linear(x, w))
    out = F.linear(x, w)
    wbytes = w.numel() * w.element_size()
    floor = roofline_us(wbytes)
    return {
        "weight_shape": list(w.shape), "weight_dtype": str(w.dtype),
        "weight_bytes": wbytes, "weight_GB": wbytes / 1e9,
        "floor_us": floor,
        "measured_us": t_linear["median_ms"] * 1e3,
        "measured_min_us": t_linear["min_ms"] * 1e3,
        "overhead_pct": (t_linear["median_ms"] * 1e3 / floor - 1.0) * 100.0,
        "out_dtype": str(out.dtype),
        "bw_realized_GBs": wbytes / (t_linear["median_ms"] * 1e-3) / 1e9,
    }


def roofline_body(body_lin):
    import torch
    qm = body_lin.quant_method
    # infer input dim
    in_dim = None
    for a in ("input_size_per_partition", "input_size"):
        if hasattr(body_lin, a):
            in_dim = int(getattr(body_lin, a))
            break
    if in_dim is None:
        # fall back to weight inspection
        w = getattr(body_lin, "weight", None)
        in_dim = int(w.shape[-1]) if w is not None else 2560
    x = torch.randn(1, in_dim, dtype=torch.bfloat16, device="cuda")
    try:
        t = time_op(lambda: qm.apply(body_lin, x))
        outshape = list(qm.apply(body_lin, x).shape)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "in_dim": in_dim}
    wbytes = 0
    for _, p in body_lin.named_parameters(recurse=False):
        wbytes += p.numel() * p.element_size()
    for _, p in body_lin.named_buffers(recurse=False):
        wbytes += p.numel() * p.element_size()
    floor = roofline_us(wbytes)
    return {
        "in_dim": in_dim, "out_shape": outshape, "weight_bytes": wbytes,
        "floor_us": floor, "measured_us": t["median_ms"] * 1e3,
        "measured_min_us": t["min_ms"] * 1e3,
        "overhead_pct": (t["median_ms"] * 1e3 / floor - 1.0) * 100.0,
    }


def measure_m1ar_tps(llm, prompts, max_tokens=256):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens)
    # warm
    llm.generate([{"prompt_token_ids": prompts[0]}], sp, use_tqdm=False)
    rows = []
    for ptoks in prompts:
        t0 = time.perf_counter()
        o = llm.generate([{"prompt_token_ids": ptoks}], sp, use_tqdm=False)[0]
        dt = time.perf_counter() - t0
        n = len(o.outputs[0].token_ids)
        rows.append({"n": n, "dt": dt, "tps": n / dt})
    tpss = sorted(r["tps"] for r in rows)
    return {"per_prompt": rows, "median_tps": statistics.median(tpss),
            "mean_tps": statistics.mean(tpss), "n_prompts": len(rows)}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model-dir", default=LOCAL_CKPT)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=str(HERE / "enumerate_and_roofline.json"))
    args = ap.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM

    print(f"[load] model={args.model_dir}", flush=True)
    t0 = time.time()
    llm = LLM(model=args.model_dir, quantization="compressed-tensors",
              dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
              enable_prefix_caching=False, enforce_eager=True,
              trust_remote_code=True)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    mr = get_model_runner(llm)
    kern, body_lin, lm_head = enumerate_kernels(mr)

    rows = kern.get("lm_head_rows")
    full_ok = (rows == FULL_LM_HEAD_ROWS)
    print(f"[assert] lm_head rows={rows} full_ok={full_ok}", flush=True)
    if full_ok:
        print(f"verified full lm_head: {rows} rows", flush=True)

    print("[roofline] head ...", flush=True)
    head_rf = roofline_head(lm_head) if lm_head is not None else {"error": "no lm_head"}
    print(f"  head measured={head_rf.get('measured_us'):.1f}us floor={head_rf.get('floor_us'):.1f}us "
          f"overhead={head_rf.get('overhead_pct'):.1f}% bw={head_rf.get('bw_realized_GBs'):.0f}GB/s",
          flush=True)

    print("[roofline] body ...", flush=True)
    body_rf = roofline_body(body_lin) if body_lin is not None else {"error": "no body"}
    if "measured_us" in body_rf:
        print(f"  body measured={body_rf['measured_us']:.2f}us floor={body_rf['floor_us']:.2f}us "
              f"overhead={body_rf['overhead_pct']:.1f}%", flush=True)
    else:
        print(f"  body roofline: {body_rf}", flush=True)

    # tokenizer / prompts for M=1 AR TPS
    tok = llm.get_tokenizer()
    seed_texts = [
        "Explain why the sky is blue in a few sentences.",
        "Write a short paragraph about the history of computing.",
        "Summarize the plot of a typical hero's journey story.",
        "Describe how photosynthesis works step by step.",
    ]
    prompts = [tok.encode(t) for t in seed_texts]
    print("[tps] M=1 AR decode ...", flush=True)
    tps = measure_m1ar_tps(llm, prompts, max_tokens=args.max_tokens)
    print(f"  base_fullhead M=1 AR median_tps={tps['median_tps']:.2f} "
          f"(eager/spec-off; overhead-bound, NOT the served 253.78 regime)", flush=True)

    # self-determinism: same greedy prompt twice -> token-identical (#319 ref path)
    from vllm import SamplingParams
    sp_sd = SamplingParams(temperature=0.0, max_tokens=64, min_tokens=64)
    sd_total = sd_match = 0
    for ptoks in prompts:
        a = llm.generate([{"prompt_token_ids": ptoks}], sp_sd, use_tqdm=False)[0].outputs[0].token_ids
        b = llm.generate([{"prompt_token_ids": ptoks}], sp_sd, use_tqdm=False)[0].outputs[0].token_ids
        m = min(len(a), len(b))
        sd_total += m
        sd_match += sum(1 for i in range(m) if a[i] == b[i])
    self_det = (sd_match / sd_total) if sd_total else 0.0
    print(f"  self_det={self_det:.6f} ({sd_match}/{sd_total} greedy tokens identical r1-vs-r2)", flush=True)

    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    result = {
        "pr": 550, "stage": 1, "analysis_only": True, "official_tps": 0,
        "model_dir": args.model_dir, "device": torch.cuda.get_device_name(0),
        "sm": "".join(map(str, torch.cuda.get_device_capability(0))),
        "A10G_HBM_GBS": A10G_HBM_GBS,
        "current_kernels": kern,
        "lm_head_full_rows": rows, "lm_head_full_ok": full_ok,
        "head_roofline": head_rf, "body_roofline": body_rf,
        "base_fullhead_m1ar_tps": tps,
        "base_fullhead_m1ar_tps_caveat": "eager + spec-off; overhead-bound across 42 layers, NOT the graphed+spec-on served 253.78 regime. Per-op roofline is the regime-invariant kernel-cost truth.",
        "self_det": self_det,
        "peak_vram_gib": peak_gib,
        "self_test_passes": self_test(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {args.out}  peak_vram={peak_gib:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
