"""PR #501 microbench: localize the "-135 matmul tax" and test whether a
fixed-order split-K GEMM can make the deployed int4 Marlin matmuls (body +
lm_head) M-invariant (M=8 verify == M=1 AR, byte-exact) -- the GEMM analogue of
lawine #496's fixed-order split-KV attention fix.

WHY THIS HARNESS EXISTS
-----------------------
#496 proved fixed-order (fix the split SIZE not count) makes the attention
split-KV M-invariant -> byte-exact at ~0% cost (surgical 345.22 -> byteexact
399.75 TPS). PR #501 asks the same of the *matmul axis*: the cited "-135 matmul
tax" is surgical(345.22, byte-exact attn + STOCK matmul) - full_flag(212.28,
byte-exact attn + VLLM_BATCH_INVARIANT matmul) = 132.94. Can a fixed-order
split-K GEMM recover that 133 while staying byte-exact?

KEY STRUCTURAL FACTS (grounded in the serve-venv vLLM, advisor-branch deployed
config /tmp/osoi5-v0-baked):
  * The deployed BODY (qkv/o/gate_up/down) AND the pruned lm_head are BOTH int4
    compressed-tensors W4A16 (group_size=128) -> every production matmul goes
    through the Marlin custom op ``ops.marlin_gemm`` (torch.ops._C.marlin_gemm).
  * ``VLLM_BATCH_INVARIANT=1`` (full_flag) only overrides aten::mm/addmm/matmul/
    linear/bmm/softmax/mean (batch_invariant.py). NO quantization linear method
    has a batch-invariant branch -> full_flag does NOT touch the Marlin GEMMs.
  * On A10G sm_86 + bf16, ``should_use_atomic_add_reduce`` ALWAYS returns False
    (atomic-add+bf16 unsupported pre-SM90), so Marlin reduces split-K partials in
    fp32 global-reduce -- a deterministic, value-stable reduction.
  * ``marlin_gemm`` exposes NO num_splits / thread_k / split-K knob to Python
    (see _custom_ops.marlin_gemm signature). The split count is chosen inside
    compiled CUDA from (size_m,size_n,size_k,#SM). Pinning the split SIZE would
    require a gptq_marlin.cu rebuild -- out of scope (no served-file change).

So this harness MEASURES, at the kernel level (the only clean isolation -- the
e2e serve identity gate is confounded: #496 full_flag-vs-its-own-M1AR = 0.645,
m1ar_gate_valid=False), the property that actually matters:

  CENSUS  for each deployed int4 Marlin shape, M in {1..8} (+ boundary probe):
          feed the SAME input row at M=1 and as row-i of an M-row batch; compare
          the bf16 output bytes (torch.equal on the int16 view, #496 standard).
          flips==0  <=>  the matmul is already M-invariant (byte-exact) and there
          is NO reduction-order tax to collapse on that op.

  CONTROL bf16 cuBLAS (torch.mm) vs matmul_persistent (the batch_invariant
          fixed-order GEMM) at the gate_up / lm_head shapes: cuBLAS is the
          positive control (CAN flip -> proves harness sensitivity); persistent
          is the 0-flip fixed-order reference. Establishes that the fixed-order
          discipline is real but that the int4 production path simply does not
          need it.

  COST    marlin_gemm M=8 latency per shape (eager + cudagraph) = the deployed,
          already-byte-exact matmul cost. If the census is all-0-flip the
          byte-exact GEMM cost == deployed cost -> realized fixed-order-GEMM tax
          = 0 on the int4 path, and the "-135" is full_flag *machinery*
          (cudagraph/TF32/aten overrides), not a Marlin reduction cost.

Synthetic random weights/inputs are valid: M-invariance and split-K structure
are functions of (M,N,K,#SM) only, not weight VALUES (same argument as #496's
synthetic q/kv). analysis_only=true, official_tps=0, NO served-file change.

Run (serve venv):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/<hash>/bin/python \
      research/speed/byteexact_gemm/microbench_gemm_tax.py --out <json> [--wandb]
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "0,", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402

# ---- deployed Gemma-4-E4B-it int4 Marlin matmul stack (config /tmp/osoi5-v0-baked) ----
# hidden=2560, intermediate=10240, heads=8 head_dim=256 (q=2048), kv_heads=2 (kv=512),
# 37 layers, lm_head pruned to 16384 rows (PCK04 keepset). All int4 W4A16, group_size=128.
HIDDEN = 2560
INTERMEDIATE = 10240
Q_DIM = 8 * 256          # 2048
KV_DIM = 2 * 256         # 512
QKV_N = Q_DIM + 2 * KV_DIM  # 3072
LMHEAD_N = 16384         # pruned PCK04 keepset rows (divisible by 64)
N_LAYERS = 37
GROUP_SIZE = 128

# (name, N, K, per-step count across the served stack)
DEPLOYED_SHAPES = [
    ("qkv_proj", QKV_N, HIDDEN, N_LAYERS),
    ("o_proj", HIDDEN, Q_DIM, N_LAYERS),
    ("gate_up_proj", 2 * INTERMEDIATE, HIDDEN, N_LAYERS),
    ("down_proj", HIDDEN, INTERMEDIATE, N_LAYERS),
    ("lm_head_pck04", LMHEAD_N, HIDDEN, 1),
]

# spec verify width K_spec=7 -> decode M in {1..8}; probe a few larger M to locate
# the M at which Marlin's tiling/par would change (the invariance boundary).
CENSUS_M = (1, 2, 3, 4, 5, 6, 7, 8)
BOUNDARY_M = (16, 32, 64, 128)
SEEDS = (1234, 5678, 9012)
SCALES = (0.1, 1.0)
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

# banked #496 serve anchors (advisor-branch, run 42qroec1) for the tax reconciliation
DEPLOYED_TPS = 426.36
SURGICAL_TPS = 345.22       # byte-exact 2D attn + STOCK matmul
BYTEEXACT_ATTN_TPS = 399.75  # #496 fixed-order split-KV rung (the bar to beat)
FULL_FLAG_TPS = 212.28      # VLLM_BATCH_INVARIANT=1 (byte-exact attn + BI matmul)
MATMUL_TAX = SURGICAL_TPS - FULL_FLAG_TPS  # 132.94 ~ the cited "-135"
PPL_ANCHOR = 2.3767


def _marlin_imports():
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        GPTQ_MARLIN_MAX_PARALLEL,
        GPTQ_MARLIN_MIN_THREAD_N,
        should_use_atomic_add_reduce,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        MarlinWorkspace,
        marlin_quantize,
    )
    return (ops, scalar_types, GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
            should_use_atomic_add_reduce, MarlinWorkspace, marlin_quantize)


def _build_marlin(N, K, seed, scale):
    (ops, scalar_types, GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
     _should_atomic, MarlinWorkspace, marlin_quantize) = _marlin_imports()
    dev = torch.device(DEVICE)
    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.randn(K, N, device=dev, dtype=DTYPE, generator=g) * scale
    qtype = scalar_types.uint4b8
    w_ref, q_w, s, g_idx, sort_idx, _perm = marlin_quantize(W, qtype, GROUP_SIZE, False)
    ws = MarlinWorkspace(N, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)

    def gemm(x):
        return ops.marlin_gemm(
            x, None, q_w, None, s, None, None, None, g_idx, sort_idx, ws.scratch,
            qtype, size_m=x.shape[0], size_n=N, size_k=K, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )

    return gemm, qtype


def _atomic_add_context(N, K):
    """Record the Marlin reduce config the deployed call resolves to (the source
    of any non-determinism). On sm_86+bf16 atomic-add is force-off -> fp32 global
    reduce -> deterministic."""
    (_ops, _st, _maxp, _minn, should_use_atomic_add_reduce, _ws, _mq) = _marlin_imports()
    dev = torch.device(DEVICE)
    return {
        "use_atomic_add": bool(should_use_atomic_add_reduce(8, N, K, dev, DTYPE)),
        "use_fp32_reduce": True,
        "atomic_add_disabled_reason": "sm_86 (<SM90) + bf16: atomicAdd unsupported -> fp32 global reduce",
    }


# ----------------------------------------------------------------- census
def _census_one(gemm, Ms, seed_tag):
    """For each M, feed identical input rows at M=1 and as rows of an M-batch;
    compare bf16 output bytes (int16 view). flips==0 <=> M-invariant."""
    dev = torch.device(DEVICE)
    out = {}
    for M in Ms:
        g = torch.Generator(device=dev).manual_seed(99 + M + seed_tag)
        # X holds M distinct rows; the M=1 ref re-runs each row alone.
        X = _rand_like_input(gemm, M, g)
        Ybatch = gemm(X)
        flips = 0
        max_abs = 0.0
        for i in range(M):
            Y1 = gemm(X[i:i + 1])
            eq = torch.equal(Ybatch[i].view(torch.int16), Y1[0].view(torch.int16))
            md = float((Ybatch[i].float() - Y1[0].float()).abs().max().item())
            max_abs = max(max_abs, md)
            flips += 0 if eq else 1
        out[str(M)] = {"flips": flips, "n_rows": M, "max_abs_err": max_abs,
                       "byte_invariant": flips == 0}
    return out


def _rand_like_input(gemm, M, gen):
    # recover K from the closure's free vars
    K = None
    for cell, name in zip(gemm.__closure__ or [], (gemm.__code__.co_freevars or [])):
        if name == "K":
            K = cell.cell_contents
    if K is None:
        raise RuntimeError("could not recover K from gemm closure")
    return torch.randn(M, K, device=torch.device(DEVICE), dtype=DTYPE, generator=gen) * 0.1


# --------------------------------------------------------- bf16 controls
def _bf16_controls(N, K, seed):
    """Positive control: bf16 cuBLAS (torch.mm) CAN be M-variant; matmul_persistent
    (the batch_invariant fixed-order GEMM) is 0-flip by construction. These are NOT
    production ops (the deployed matmuls are int4 Marlin) -- they prove harness
    sensitivity + that the fixed-order discipline works for bf16."""
    from vllm.model_executor.layers.batch_invariant import matmul_persistent
    dev = torch.device(DEVICE)
    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.randn(K, N, device=dev, dtype=DTYPE, generator=g) * 0.1
    Mmax = 8
    X = torch.randn(Mmax, K, device=dev, dtype=DTYPE, generator=g) * 0.1

    def census(fn):
        Yb = fn(X)
        flips = 0
        maxabs = 0.0
        for i in range(Mmax):
            Y1 = fn(X[i:i + 1])
            eq = torch.equal(Yb[i].view(torch.int16), Y1[0].view(torch.int16))
            maxabs = max(maxabs, float((Yb[i].float() - Y1[0].float()).abs().max().item()))
            flips += 0 if eq else 1
        return flips, maxabs

    cub_flips, cub_max = census(lambda x: torch.mm(x, W))
    per_flips, per_max = census(lambda x: matmul_persistent(x, W))
    return {
        "shape": {"N": N, "K": K},
        "cublas_torch_mm": {"flips": cub_flips, "max_abs_err": cub_max,
                            "m_invariant": cub_flips == 0},
        "matmul_persistent_fixed_order": {"flips": per_flips, "max_abs_err": per_max,
                                          "m_invariant": per_flips == 0},
    }


# ----------------------------------------------------------------- timing
def _time_eager(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def _time_graph(fn, iters, warmup):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(5, warmup)):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)
    except Exception as exc:  # noqa: BLE001
        print(f"    [graph capture failed: {exc!r}]", flush=True)
        return None


def _time_marlin(N, K, M, iters, warmup):
    gemm, _ = _build_marlin(N, K, 1234, 0.1)
    dev = torch.device(DEVICE)
    X = torch.randn(M, K, device=dev, dtype=DTYPE,
                    generator=torch.Generator(device=dev).manual_seed(7)) * 0.1
    out = torch.empty_like(gemm(X))

    def call():
        out.copy_(gemm(X))

    return {"eager_us": round(_time_eager(call, iters, warmup), 3),
            "graph_us": (lambda v: round(v, 3) if v is not None else None)(_time_graph(call, iters, warmup))}


# ----------------------------------------------------------------- compose
def compose(args):
    dev = torch.device(DEVICE)
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    gpu = {"name": p.name, "sm_count": p.multi_processor_count,
           "cc": f"{cc[0]}.{cc[1]}", "is_a10g_sm86": bool("A10G" in p.name and cc == (8, 6))}
    print(f"device: {gpu['name']} cc={gpu['cc']} sms={gpu['sm_count']}", flush=True)

    # marlin_gemm split-K knob probe (instruction 2 structural fact)
    from vllm import _custom_ops as ops
    sig = inspect.signature(ops.marlin_gemm)
    knob_params = [pn for pn in sig.parameters if any(
        t in pn.lower() for t in ("split", "num_split", "thread_k", "max_par", "par", "slice"))]
    splitk_knob = {
        "marlin_gemm_params": list(sig.parameters.keys()),
        "exposes_splitk_knob": bool(knob_params),
        "matched_knob_params": knob_params,
        "note": "split count chosen inside compiled CUDA from (size_m,size_n,size_k,#SM); "
                "no Python knob -> pinning split SIZE needs a gptq_marlin.cu rebuild (out of scope)",
    }

    # ---- per-op int4 Marlin M-invariance census ----
    census = {}
    all_flips = 0
    boundary = {}
    for (name, N, K, count) in DEPLOYED_SHAPES:
        print(f"\n== census {name} N={N} K={K} ==", flush=True)
        per_shape = {"N": N, "K": K, "stack_count": count,
                     "reduce_config": _atomic_add_context(N, K), "by_seed_scale": {}}
        shape_flips = 0
        for seed in SEEDS:
            for scale in SCALES:
                gemm, _ = _build_marlin(N, K, seed, scale)
                res = _census_one(gemm, CENSUS_M, seed)
                tag = f"s{seed}_x{scale}"
                per_shape["by_seed_scale"][tag] = res
                f = sum(v["flips"] for v in res.values())
                shape_flips += f
        per_shape["total_flips_over_M1to8"] = shape_flips
        per_shape["m_invariant_M1to8"] = shape_flips == 0
        # boundary probe (one seed/scale): locate the M where Marlin config changes
        gemm, _ = _build_marlin(N, K, SEEDS[0], 0.1)
        bres = _census_one(gemm, BOUNDARY_M, SEEDS[0])
        boundary[name] = bres
        per_shape["boundary_probe"] = bres
        census[name] = per_shape
        all_flips += shape_flips
        print(f"  {name}: flips(M1..8)={shape_flips} invariant={shape_flips == 0} "
              f"| boundary " + " ".join(f"M{m}={bres[str(m)]['flips']}" for m in BOUNDARY_M),
              flush=True)

    int4_marlin_m_invariant = all_flips == 0

    # ---- bf16 controls ----
    print("\n== bf16 controls (cuBLAS vs matmul_persistent) ==", flush=True)
    controls = {}
    for name in ("gate_up_proj", "lm_head_pck04"):
        N, K = dict((s[0], (s[1], s[2])) for s in DEPLOYED_SHAPES)[name]
        controls[name] = _bf16_controls(N, K, 4242)
        c = controls[name]
        print(f"  {name}: cuBLAS flips={c['cublas_torch_mm']['flips']} "
              f"persistent flips={c['matmul_persistent_fixed_order']['flips']}", flush=True)
    cublas_can_flip = any(c["cublas_torch_mm"]["flips"] > 0 for c in controls.values())
    persistent_invariant = all(c["matmul_persistent_fixed_order"]["flips"] == 0 for c in controls.values())

    # ---- timing / cost accounting ----
    print("\n== timing (marlin_gemm, deployed already-byte-exact cost) ==", flush=True)
    timing = {}
    per_step_graph_us = 0.0
    for (name, N, K, count) in DEPLOYED_SHAPES:
        t = _time_marlin(N, K, 8, args.iters, args.warmup)
        timing[name] = {**t, "stack_count": count}
        g_us = t["graph_us"] if t["graph_us"] is not None else t["eager_us"]
        per_step_graph_us += g_us * count
        print(f"  {name:16s} M=8 eager={t['eager_us']}us graph={t['graph_us']}us x{count}", flush=True)
    # bf16 fixed-order cost contrast at the two control shapes (M=8)
    bf16_cost = {}
    from vllm.model_executor.layers.batch_invariant import matmul_persistent
    for name in ("gate_up_proj", "lm_head_pck04"):
        N, K = dict((s[0], (s[1], s[2])) for s in DEPLOYED_SHAPES)[name]
        g = torch.Generator(device=dev).manual_seed(11)
        W = torch.randn(K, N, device=dev, dtype=DTYPE, generator=g) * 0.1
        X = torch.randn(8, K, device=dev, dtype=DTYPE, generator=g) * 0.1
        outc = torch.empty(8, N, device=dev, dtype=DTYPE)
        cub = _time_eager(lambda: outc.copy_(torch.mm(X, W)), args.iters, args.warmup)
        per = _time_eager(lambda: outc.copy_(matmul_persistent(X, W)), args.iters, args.warmup)
        bf16_cost[name] = {"cublas_us": round(cub, 3), "persistent_us": round(per, 3),
                           "persistent_over_cublas": round(per / cub, 3) if cub else None}
        print(f"  bf16 {name}: cuBLAS={cub:.1f}us persistent={per:.1f}us "
              f"ratio={bf16_cost[name]['persistent_over_cublas']}", flush=True)

    # ---- verdict ----
    verdict = {
        "int4_marlin_m_invariant_M1to8": int4_marlin_m_invariant,
        "total_flips_int4_marlin": all_flips,
        "marlin_exposes_splitk_knob": splitk_knob["exposes_splitk_knob"],
        "bf16_cublas_positive_control_flips": cublas_can_flip,
        "bf16_matmul_persistent_invariant": persistent_invariant,
        "matmul_tax_tps_cited": round(MATMUL_TAX, 2),
        "matmul_tax_source": (
            "surgical(345.22, byte-exact attn + STOCK Marlin matmul) - full_flag(212.28, "
            "VLLM_BATCH_INVARIANT) = 132.94. Marlin GEMMs are unchanged by the flag "
            "(no quant method has a batch-invariant branch) AND are already 0-flip "
            "M-invariant -> the tax is full_flag MACHINERY (cudagraph/TF32/aten overrides), "
            "NOT a Marlin reduction-order cost recoverable by fixed-order split-K GEMM."
            if int4_marlin_m_invariant else
            "int4 Marlin DOES flip at decode M -> a real reduction-order tax, but no Python "
            "split-K knob exists (kernel rebuild required)."
        ),
        "fixed_order_gemm_recoverable_tps": 0.0 if int4_marlin_m_invariant else None,
        "byte_exact_gemm_ceiling_tps": BYTEEXACT_ATTN_TPS if int4_marlin_m_invariant else None,
        "per_step_marlin_matmul_graph_us": round(per_step_graph_us, 2),
        "ppl_anchor": PPL_ANCHOR,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
    }

    return {"gpu": gpu, "deployed_shapes": [
                {"name": n, "N": N, "K": K, "count": c} for (n, N, K, c) in DEPLOYED_SHAPES],
            "splitk_knob_probe": splitk_knob, "census": census, "boundary": boundary,
            "bf16_controls": controls, "timing": timing, "bf16_fixed_order_cost": bf16_cost,
            "verdict": verdict}


# ----------------------------------------------------------------- self-test
def self_test():
    checks = {}
    checks["matmul_tax_is_133"] = abs(MATMUL_TAX - 132.94) < 0.01
    checks["surgical_minus_fullflag"] = abs((SURGICAL_TPS - FULL_FLAG_TPS) - MATMUL_TAX) < 1e-9
    checks["qkv_n_3072"] = QKV_N == 3072
    checks["gate_up_n_20480"] = 2 * INTERMEDIATE == 20480
    checks["lmhead_div_64"] = LMHEAD_N % 64 == 0
    checks["census_covers_spec_width"] = max(CENSUS_M) == 8 and min(CENSUS_M) == 1
    checks["all_shapes_div_minthread"] = all(N % 64 == 0 and K % 128 == 0 for (_n, N, K, _c) in DEPLOYED_SHAPES)
    checks["analysis_only_guard"] = True
    passed = all(bool(v) for v in checks.values())
    return {"self_test_passes": passed, "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/speed/byteexact_gemm/gemm_tax_results.json")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default="fixed-order-gemm-byteexact")
    ap.add_argument("--wandb_name", default="denken/fixed-order-gemm-byteexact")
    args = ap.parse_args()

    if args.self_test:
        st = self_test()
        print(json.dumps(st, indent=2))
        sys.exit(0 if st["self_test_passes"] else 1)

    assert torch.cuda.is_available(), "need GPU (CUDA_VISIBLE_DEVICES=0)"
    res = compose(args)
    res["self_test"] = self_test()
    res["self_test_passes"] = res["self_test"]["self_test_passes"]
    res["timestamp"] = datetime.now(timezone.utc).isoformat()
    res["args"] = vars(args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    v = res["verdict"]
    print("\n=========== VERDICT ===========", flush=True)
    print(f"  int4_marlin_m_invariant_M1to8 = {v['int4_marlin_m_invariant_M1to8']} "
          f"(total flips={v['total_flips_int4_marlin']})", flush=True)
    print(f"  marlin_exposes_splitk_knob    = {v['marlin_exposes_splitk_knob']}", flush=True)
    print(f"  bf16 cuBLAS flips (control)    = {v['bf16_cublas_positive_control_flips']} | "
          f"persistent invariant = {v['bf16_matmul_persistent_invariant']}", flush=True)
    print(f"  matmul_tax_tps_cited          = {v['matmul_tax_tps_cited']}", flush=True)
    print(f"  byte_exact_gemm_ceiling_tps   = {v['byte_exact_gemm_ceiling_tps']}", flush=True)
    print(f"  per_step_marlin_matmul_graph_us = {v['per_step_marlin_matmul_graph_us']}", flush=True)
    print(f"  source: {v['matmul_tax_source']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)

    if args.wandb:
        _log_wandb(args, res)


def _log_wandb(args, res):
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb skipped] {exc}", flush=True)
        return
    v = res["verdict"]
    run = wandb.init(project="gemma-challenge-senpai", group=args.wandb_group,
                     name=args.wandb_name,
                     tags=["fixed-order-gemm-byteexact", "pr501", "analysis-only"],
                     config={"deployed_shapes": res["deployed_shapes"], "census_M": list(CENSUS_M),
                             "seeds": list(SEEDS), "scales": list(SCALES),
                             "analysis_only": True, "official_tps": 0})
    flat = {
        "int4_marlin_m_invariant": int(v["int4_marlin_m_invariant_M1to8"]),
        "total_flips_int4_marlin": v["total_flips_int4_marlin"],
        "marlin_exposes_splitk_knob": int(v["marlin_exposes_splitk_knob"]),
        "bf16_cublas_positive_control_flips": int(v["bf16_cublas_positive_control_flips"]),
        "bf16_matmul_persistent_invariant": int(v["bf16_matmul_persistent_invariant"]),
        "matmul_tax_tps_cited": v["matmul_tax_tps_cited"],
        "byte_exact_gemm_ceiling_tps": v["byte_exact_gemm_ceiling_tps"] or 0,
        "per_step_marlin_matmul_graph_us": v["per_step_marlin_matmul_graph_us"],
        "ppl_anchor": PPL_ANCHOR, "official_tps": 0,
        "self_test_passes": int(res["self_test_passes"]),
    }
    for name, sh in res["census"].items():
        flat[f"flips_M1to8/{name}"] = sh["total_flips_over_M1to8"]
    for name, t in res["timing"].items():
        flat[f"marlin_graph_us/{name}"] = t["graph_us"] if t["graph_us"] is not None else t["eager_us"]
    wandb.log(flat)
    wandb.summary.update(flat)
    print(f"wandb run: {run.id}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
