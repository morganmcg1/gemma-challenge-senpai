#!/usr/bin/env python
"""PR #257 — Built-step roofline grounding: empirically bracket the 1.085 ms tree step.

WHAT THIS GROUNDS
-----------------
The launch packet reads the TPS gate against a BUILT (tree-decode) per-step cost of
~1.085 ms (denken #252: served 1.2182 ms linear-MTP-K7 and built 1.085 ms tree are
DISTINCT valid per-step costs; E[T]_both=4.512 @ 1.085 -> 520.95 GO read). But 1.085
is currently grounded ONLY analytically (land's stated effective step + denken #252's
self-consistency round-trip + #241 step_eff_landread). This leg turns the analytic
anchor into an EMPIRICALLY-grounded one, AHEAD of land #245's one live build run, by a
measured step-decomposition roofline on the CURRENTLY-served stack.

THE COMPOSITION MODEL UNDER TEST (denken #75/#85, lawine #153)
-------------------------------------------------------------
  step = verify_us(M) + (draft passes).  draft_pass_us = g_d * verify_us(M=8).
  SERVED (linear MTP K=7, verify width M=8):  S_served = verify(8) + 7*draft_pass
                                                       = verify(8)*(1 + 7*g_d)
  BUILT  (tree decode, verify width M=32):    S_built  = verify(32) + tree_draft_cost
The deployed 481.53 frontier (fa2sw_precache_kenyan #52) serves
SPECULATIVE_CONFIG {method:mtp, num_speculative_tokens:7} -> verify width M=K+1=8, so
the served decomposition is verify(8)*(1+7*g_d). The built tree (land #245) is the
prospective M=32 path. This profiler MEASURES verify_us(M) and draft_pass_us on the
deployed int4-Marlin verify body + bf16 lm_head and the served MTP drafter, computes
the empirical g_d, round-trips the served step (provenance), and PROJECTS the built
step to compare against the 1.085 analytic anchor.

MEASUREMENT (LOCAL, single A10G, forward-pass only — NO tree-decode machinery)
-----------------------------------------------------------------------------
  verify_us(M) = body_gemm(M)*num_layers + attention(M)*num_layers + lm_head(M)
    - body_gemm: the 4 deployed int4-Marlin body GEMMs (qkv/o/gate_up/down) timed at
      M in {1,2,4,8,16,32} via vLLM's compressed-tensors quant kernel (the #221/#232
      recipe). The canonical Hub int4 ckpt is body-identical to deployed osoi5 (only
      the lm_head/embedding differ -- modeled separately); osoi5-v0-baked is tried
      first and the Hub ckpt is the documented fallback.
    - attention: SDPA at [M queries x ctx keys], GQA 8/2, head_dim 256 (a ~2.6%-of-step
      term per BASELINE.md; M-scaling captured).
    - lm_head: bf16 GEMM [M,hidden] -> deployed 12k-pruned vocab (weight-bound, ~flat).
  draft_pass_us = the served MTP drafter (/tmp/qat-assistant) per-pass GEMM chain at
    M=1 (the qspec #248 ONEGRAPH recipe), and the K=7 sequential chain cost.

All timing is one consistent GRAPHED basis (CUDA-graph replay = the deployed ONEGRAPH
served step; launch overhead erased), with heavy-warmup clock boost and an eager
launch-inflation cross-check. CRUCIAL BASIS NOTE: the deployed int4 verify body is 3.85 B
params -> 1.93 GB int4 -> a ~3.2 ms HBM floor on the A10G (600 GB/s), so a verify forward
is several ms of WALL-CLOCK -- the banked 1.2182 "ms" step is therefore NOT wall-clock; it
is a NORMALIZED composition unit (K_cal=125.27 absorbs the ~8x benchmark/overhead factor,
1000/125.27~=7.98). The single bridge c = STEP_SERVED / S_served thus RECONCILES the
isolated-wall-clock roofline to that normalized unit (c~=0.18, provenance, NOT a c~=1
validation -- this leg corrects the earlier mis-statement). What survives the bridge and
DRIVES the verdict is the basis-robust RATIO S_built/S_served, so step_built_proj =
1.2182 * S_built/S_served. That ratio hinges on (i) verify(32)/verify(8) (measured ~1.15,
the M=8->M=32 verify GROWTH; knee ~M=16) and (ii) g_d = draft_pass/verify(8). g_d is the
crux: measured full-forward g_d~=0.017 (drafter 4 layers vs verify 42 layers) is ~10x
BELOW the assumed 0.168 -- the projection is reported under BOTH g_d bases so the verdict's
g_d-sensitivity is explicit, and the 0.168 gap is FLAGGED for advisor reconciliation
(likely the fleet verify_us is not a full-forward wall-clock quantity).

HONEST BAND (carried from denken #252 follow-up #1)
--------------------------------------------------
This is a forward-pass PROJECTION (isolated deployed-kernel GEMM-chain + attn + lm_head
reconstruction bridged to the served anchor), NOT the live tree-decode step. It BRACKETS
land #245's number; it does not replace it. The served-step round-trip is exact BY the
bridge (provenance, like #252's honest band a); the empirical content is (i) g_d_measured
vs the assumed 0.168 and (ii) the basis-robust built/served cost ratio that sets the
verdict vs the INDEPENDENT 1.085 anchor. A DIVERGE (proj materially > 1.085) is a valid,
HIGH-VALUE terminal result: it pre-warns the packet that the 520.95 read may shift toward
served-path pricing before land spends the one build run.

SCOPE: no HF Job, no submission, no served-file change, no draw, no train. BASELINE stays
481.53; this leg adds 0 TPS (a measurement). Single GPU CUDA_VISIBLE_DEVICES=0.

SELF-TEST (`built_step_roofline_grounding_self_test_passes`)
  (a) served-step round-trip reproduces 1.2182 from verify(8)+empirical g_d (<= tol)
  (b) step_built_measured_proj reported w/ tree-draft model + CONFIRM/DIVERGE verdict
  (c) verify_us(M) curve over M in {1,2,4,8,16,32} + bandwidth->compute knee reported
  (d) NaN-clean   (e) peak VRAM <= 24 GB
TEST: step_built_measured_proj (ms, vs 1.085 analytic) and g_d_measured (vs 0.168).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# Imported fleet anchors (DO NOT re-derive). denken #252 (a7llo7o7) / #241 (hqewf1d6),
# lawine #153 (ma0qlpas) verify_step_m_curve, denken #75/#85 + kanna #217 (vgovdrjc)
# composition, kanna #126 tau band.
# --------------------------------------------------------------------------------------
STEP_SERVED = 1.2182                      # served step ms (linear MTP K=7, M=8) #252/#136
STEP_BUILT_ANALYTIC = 1.084952540947906  # the 1.085 analytic built step (tree M=32) #252/#241
K_CAL = 125.26795005202914               # official step-normalizer #148/#169/#217
G_D_ASSUMED = 0.168                       # drafter share g_d (denken #75/#85; lawine #153 drafter_share_m32)
E_T_SERVED = 4.6827608                    # served realized E[T] (tau=1) #252
E_T_BUILT = 4.512                         # built both-bugs E[T] projection #252
K_SPEC = 7                                # num_speculative_tokens (manifest)
M_VERIFY_SERVED = 8                       # = K+1, served linear verify width
M_TREE = 32                               # built tree verify width
OFFICIAL_BASELINE = 481.53               # #52 served official TPS
GO_READ = 520.9527323111674              # land reset GO read / ubel #240 lambda=1 ceiling
TARGET_TPS = 500.0
TAU_BAND = (0.9924, 1.0)                  # served tau band #252
TAU_TREE_CENTRAL = 1.0                    # kanna #126 tau_tree central
TREE_DRAFT_PASSES_EMPIRICAL = 5           # lawine #153 b_for_M[32] (frontier-batch count to fill M=32)

M_SWEEP = [1, 2, 4, 8, 16, 32]
# tree-draft-cost models priced from the measured draft_pass_us (instruction 4):
#   n_tree = 5  : lawine #153 empirical frontier-batch count for M=32 (central)
#   n_tree = 7  : conservative -- tree draft costs the SAME as the linear K=7 chain
#   n_tree = 9  : pessimistic -- lawine depth-9 tree, one pass per level
TREE_PASS_MODELS = {"central_b5": 5, "conservative_k7": 7, "pessimistic_depth9": 9}
CONFIRM_TOL_MS = 0.030                    # |proj - 1.085| <= tol AND proj <= 1.085+tol => CONFIRM
ROUNDTRIP_TOL_MS = 1e-6                   # served round-trip residual (exact by bridge)
A10G_BW_GBPS = 600.0                      # A10G (GA102, GDDR6) HBM bandwidth ~600 GB/s
INT4_BYTES = 0.5                          # W4 weight byte/param
BF16_BYTES = 2.0
K_CAL_OVERHEAD_FACTOR = 1000.0 / K_CAL    # ~7.98: normalized-unit <-> wall-clock-ms scale

# canonical Hub int4 ckpt (body-identical to deployed osoi5) + deployed osoi5 first
MODEL_CANDIDATES = [
    "/tmp/osoi5-v0-baked",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
]
DRAFTER_DIR = "/tmp/qat-assistant"
LM_HEAD_VOCAB_DEPLOYED = 12288            # deployed LM_HEAD_PRUNE 12k (manifest)
LM_HEAD_VOCAB_FULL = 262144
OUT_DIR = Path("research/validity/built_step_roofline")


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------
def _deshadow_stdlib_profile() -> None:
    """This file is named profile.py and the GPU phases run as `python .../profile.py`,
    putting our dir on sys.path[0]. vllm -> torch._dynamo -> cProfile does `import profile`
    and would pick US up instead of the stdlib `profile`. Drop our dir so stdlib wins."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != here]
    sys.modules.pop("profile", None)


def _stats(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    n = len(vals)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    return {"n": n, "mean": mean, "median": statistics.median(vals), "std": std,
            "se": se, "ci95_abs": 1.96 * se, "cv_pct": (100.0 * std / mean) if mean else 0.0,
            "min": min(vals), "max": max(vals), "values": vals}


def resolve_model_dirs() -> list[str]:
    """Ordered loadable candidates: deployed osoi5-v0-baked first, then the canonical
    Hub int4 ckpt (body-identical: same gemma-4-E4B w4a16 decoder; only lm_head/embed
    differ, modeled separately). osoi5 may not load under vanilla vLLM (PLE-folded /
    pruned lm_head needs serve patches; #221/#232) -> phase_verify falls back."""
    found = []
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            found.append(str(p))
        elif p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    found.append(str(sub))
                    break
    if not found:
        raise FileNotFoundError(f"no int4 model found among {MODEL_CANDIDATES}")
    return found


def read_text_dims(model_dir: str) -> dict:
    cfg = json.load(open(Path(model_dir) / "config.json"))
    tc = cfg.get("text_config", cfg)
    h, n_heads, n_kv = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"]
    hd, inter = tc["head_dim"], tc["intermediate_size"]
    return {
        "hidden": h, "n_heads": n_heads, "n_kv": n_kv, "head_dim": hd,
        "intermediate": inter, "num_layers": tc.get("num_hidden_layers"),
        "shapes": {
            "qkv_proj": ((n_heads + 2 * n_kv) * hd, h),
            "o_proj": (h, n_heads * hd),
            "gate_up_proj": (2 * inter, h),
            "down_proj": (h, inter),
        },
    }


def sample_sm_clock():
    try:
        import pynvml
        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetClockInfo(pynvml.nvmlDeviceGetHandleByIndex(0),
                                                 pynvml.NVML_CLOCK_SM))
    except Exception:
        return None


# ======================================================================================
# PHASE: verify  (deployed int4-Marlin body GEMM M-roofline + attention + lm_head)
# ======================================================================================
def phase_verify(out_path: str, ctx: int, iters: int, warmup: int, repeats: int) -> None:
    _deshadow_stdlib_profile()
    import torch
    from vllm import LLM

    dev = torch.device("cuda:0")
    candidates = resolve_model_dirs()
    llm = model_dir = dims = None
    load_errors = {}
    for cand in candidates:
        try:
            cd = read_text_dims(cand)
            print(f"[verify] trying model={cand} layers={cd['num_layers']} hidden={cd['hidden']} "
                  f"n_heads={cd['n_heads']} n_kv={cd['n_kv']} head_dim={cd['head_dim']} ctx={ctx}",
                  flush=True)
            t0 = time.time()
            llm = LLM(model=cand, quantization="compressed-tensors", dtype="bfloat16",
                      max_model_len=max(1024, ctx + max(M_SWEEP) + 8),
                      gpu_memory_utilization=0.60, max_num_seqs=1, enforce_eager=True,
                      trust_remote_code=True)
            model_dir, dims = cand, cd
            print(f"[verify] vLLM load OK ({cand}) in {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as exc:
            load_errors[cand] = repr(exc)
            print(f"[verify] load FAILED for {cand}: {exc!r}; trying next candidate", flush=True)
    if llm is None:
        raise RuntimeError(f"all int4 verify candidates failed to load: {load_errors}")
    shapes = dims["shapes"]
    loaded_num_layers = dims["num_layers"]
    # DEPLOYED depth: osoi5-v0-baked is 37 layers (num_kv_shared_layers=16, sliding/full
    # attention mix); the loadable Hub fallback is 42. Per-layer body GEMM shapes are
    # IDENTICAL, so per-layer timing transfers -- but the verify forward must be scaled by
    # the DEPLOYED layer count, not the Hub's. Read it from the deployed config; fall back
    # to the loaded count if unreadable.
    deployed_num_layers = loaded_num_layers
    deployed_layer_src = "loaded-model (osoi5 config unavailable)"
    try:
        dep_cfg = read_text_dims(MODEL_CANDIDATES[0])  # /tmp/osoi5-v0-baked
        if dep_cfg.get("num_layers"):
            deployed_num_layers = int(dep_cfg["num_layers"])
            deployed_layer_src = "osoi5-v0-baked config.json"
            if dep_cfg["shapes"] != shapes:
                print(f"[verify] WARN deployed per-layer shapes {dep_cfg['shapes']} != "
                      f"loaded {shapes}; per-layer transfer may be imperfect", flush=True)
    except Exception as exc:
        print(f"[verify] could not read deployed layer count ({exc!r}); using loaded {loaded_num_layers}", flush=True)
    num_layers = deployed_num_layers  # COMPOSE with the deployed depth
    body_identical_note = ("deployed osoi5-v0-baked" if "osoi5" in model_dir
                           else "canonical Hub int4 ckpt (per-layer body-identical to deployed "
                                "osoi5; osoi5 load fell back -- lm_head/embed differ, modeled "
                                "separately). COMPOSED with deployed depth, not Hub depth.")
    print(f"[verify] using {body_identical_note}", flush=True)
    print(f"[verify] layer count: loaded(Hub)={loaded_num_layers}  DEPLOYED(compose)="
          f"{num_layers} (src: {deployed_layer_src}); body+attn scaled by {num_layers}", flush=True)

    def get_model():
        for p in (
            lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        ):
            try:
                m = p()
                if m is not None:
                    return m
            except Exception:
                continue
        raise RuntimeError("could not locate model_runner.model")

    def find_layers(root):
        import torch.nn as nn
        chains = [("model", "layers"), ("model", "language_model", "layers"),
                  ("language_model", "model", "layers"), ("language_model", "layers"),
                  ("model", "model", "layers"), ("layers",)]
        for chain in chains:
            obj, ok = root, True
            for attr in chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return obj
        for _, mod in root.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                el = mod[0]
                if hasattr(el, "self_attn") and hasattr(el.self_attn, "qkv_proj"):
                    return mod
        raise RuntimeError("could not locate decoder ModuleList")

    def module_out_in(mod):
        out = getattr(mod, "output_size_per_partition", None)
        inp = getattr(mod, "input_size_per_partition", None)
        if out is None or inp is None:
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 2:
                out, inp = int(w.shape[0]), int(w.shape[1])
        return (int(out), int(inp)) if out and inp else None

    model = get_model()
    layers = find_layers(model)
    targets = None
    for layer in layers:
        try:
            cand = {"qkv_proj": layer.self_attn.qkv_proj, "o_proj": layer.self_attn.o_proj,
                    "gate_up_proj": layer.mlp.gate_up_proj, "down_proj": layer.mlp.down_proj}
        except AttributeError:
            continue
        if all(hasattr(m, "quant_method") and module_out_in(m) == shapes[name]
               for name, m in cand.items()):
            targets = cand
            break
    if targets is None:
        raise RuntimeError(f"no layer matched canonical body shapes {shapes}")
    print(f"[verify] located deployed int4 body GEMMs {[ (n, module_out_in(m)) for n,m in targets.items() ]}",
          flush=True)

    def bench_eager(fn, x):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(x)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0  # us per call

    def time_graphed(run, n_iters=iters, n_warm=warmup):
        """us per replay of zero-arg `run` captured in a CUDA graph (deployed ONEGRAPH
        basis: launch overhead erased, as in the served step). Eager fallback returns
        (eager_us, False). Many tiny sequential kernels -> one replay."""
        try:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(5):
                    run()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                run()
            for _ in range(max(10, n_warm)):
                g.replay()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n_iters):
                g.replay()
            e1.record(); torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / n_iters * 1000.0
            del g
            return us, True
        except Exception as exc:
            print(f"[verify]   graph capture failed ({exc!r}); eager fallback", flush=True)
            with torch.inference_mode():
                for _ in range(n_warm):
                    run()
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(n_iters):
                    run()
                e1.record(); torch.cuda.synchronize()
            return e0.elapsed_time(e1) / n_iters * 1000.0, False

    # heavy warmup -> A10G boost clock (cannot lock clocks here)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    clk = {"after_warmup": sample_sm_clock()}

    graphed_all = True
    Mmax = max(M_SWEEP)
    torch.manual_seed(0)
    gemm_inputs = {name: torch.randn(Mmax, inp, dtype=torch.bfloat16, device=dev)
                   for name, (out, inp) in shapes.items()}

    def int4_apply(mod):
        return lambda x: mod.quant_method.apply(mod, x, bias=None)

    # ---- (1) int4 body GEMM M-roofline: GRAPHED 4-GEMM chain (per-layer weights 45MB
    # >> 6MB L2, so each call is a real HBM read -> x num_layers is faithful), + eager
    # cross-check. The graphed basis matches the deployed ONEGRAPH served step. ----
    import torch.nn.functional as F
    body_gemm = {}
    for M in M_SWEEP:
        per_shape_eager = {}
        for name in shapes:
            xm = gemm_inputs[name][:M].contiguous()
            per_shape_eager[name] = _stats([bench_eager(int4_apply(targets[name]), xm)
                                            for _ in range(repeats)])
        sum4_eager = sum(per_shape_eager[n]["mean"] for n in shapes)
        # graphed: capture the 4-GEMM chain (one of each body GEMM) at width M
        xins = {name: gemm_inputs[name][:M].contiguous() for name in shapes}
        applies = [int4_apply(targets[name]) for name in shapes]

        def run_chain(_applies=applies, _xins=xins, _names=list(shapes)):
            for fn, name in zip(_applies, _names):
                fn(_xins[name])
        chain_us, cap = time_graphed(run_chain)
        graphed_all = graphed_all and cap
        body_gemm[M] = {"per_shape_us_eager": per_shape_eager, "sum_4gemm_us_eager": sum4_eager,
                        "sum_4gemm_us": chain_us, "graph_captured": cap,
                        "body_us": chain_us * num_layers, "body_us_eager": sum4_eager * num_layers}
        print(f"[verify] M={M:2d} body 4-GEMM graphed={chain_us:7.2f}us (eager {sum4_eager:7.2f}) "
              f"-> body(x{num_layers})={chain_us*num_layers/1e3:7.3f}ms cap={cap}", flush=True)

    # ---- (2) attention M-roofline (SDPA, GQA, M queries x ctx keys), graphed ----
    n_h, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    attn = {}
    for M in M_SWEEP:
        q = torch.randn(1, n_h, M, hd, dtype=torch.bfloat16, device=dev)
        k = torch.randn(1, n_h, ctx, hd, dtype=torch.bfloat16, device=dev)   # GQA expanded
        v = torch.randn(1, n_h, ctx, hd, dtype=torch.bfloat16, device=dev)

        def run_attn(_q=q, _k=k, _v=v):
            F.scaled_dot_product_attention(_q, _k, _v)
        us, cap = time_graphed(run_attn)
        graphed_all = graphed_all and cap
        attn[M] = {"mean": us, "graph_captured": cap}
        print(f"[verify] M={M:2d} attn(x{num_layers})={us*num_layers/1e3:6.3f}ms "
              f"(per-layer {us:.1f}us cap={cap})", flush=True)

    # ---- (3) lm_head bf16 GEMM (deployed 12k-pruned vocab), graphed ----
    hdn = dims["hidden"]
    lm_W12k = torch.randn(LM_HEAD_VOCAB_DEPLOYED, hdn, dtype=torch.bfloat16, device=dev) * 0.02
    lm_head = {}
    for M in M_SWEEP:
        x = torch.randn(M, hdn, dtype=torch.bfloat16, device=dev)

        def run_lm(_x=x):
            torch.matmul(_x, lm_W12k.t())
        us, cap = time_graphed(run_lm)
        graphed_all = graphed_all and cap
        lm_head[M] = {"mean": us, "graph_captured": cap}
    print(f"[verify] lm_head(12k) M=8={lm_head[8]['mean']:.1f}us M=32={lm_head[32]['mean']:.1f}us",
          flush=True)
    clk["after_timing"] = sample_sm_clock()
    timing_basis = "graphed" if graphed_all else "mixed_graphed_eager_fallback"

    # ---- compose verify_us(M) = body + attn*L + lm_head ----
    verify_us = {}
    for M in M_SWEEP:
        b = body_gemm[M]["body_us"]
        a = attn[M]["mean"] * num_layers
        h = lm_head[M]["mean"]
        verify_us[M] = {"body_us": b, "attn_us": a, "lmhead_us": h,
                        "verify_total_us": b + a + h, "verify_bodyonly_us": b,
                        "verify_body_attn_us": b + a}

    nan_clean = all(math.isfinite(verify_us[M]["verify_total_us"]) for M in M_SWEEP)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "verify", "model_dir": model_dir, "body_identical_note": body_identical_note,
        "load_errors": load_errors, "dims": dims, "ctx": ctx,
        "loaded_num_layers": loaded_num_layers, "deployed_num_layers": deployed_num_layers,
        "compose_num_layers": num_layers, "deployed_layer_src": deployed_layer_src,
        "deployed_arch_note": ("composed with deployed depth=%d (osoi5); osoi5 has "
                               "num_kv_shared_layers=16 + sliding/full attention mix, so the "
                               "full-ctx attention term is an UPPER bound -> verify slightly "
                               "OVER-estimated -> g_d slightly UNDER-estimated -> DIVERGE is "
                               "conservative." % num_layers),
        "config": {"iters": iters, "warmup": warmup, "repeats": repeats, "M_sweep": M_SWEEP,
                   "lm_head_vocab": LM_HEAD_VOCAB_DEPLOYED},
        "sm_clock_mhz": clk, "body_gemm": body_gemm, "attn_us": attn, "lmhead_us": lm_head,
        "verify_us": verify_us, "nan_clean": bool(nan_clean), "peak_gpu_gb": peak_gb,
        "timing_basis": timing_basis, "graphed_all": bool(graphed_all),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    print(f"[verify] verify_us(8)={verify_us[8]['verify_total_us']/1e3:.4f}ms "
          f"verify_us(32)={verify_us[32]['verify_total_us']/1e3:.4f}ms peak={peak_gb:.1f}GB", flush=True)
    print(f"VERIFY_DONE {out_path}", flush=True)


# ======================================================================================
# PHASE: draft  (served MTP drafter per-pass GEMM chain; qspec #248 recipe)
# ======================================================================================
def phase_draft(out_path: str, iters: int, warmup: int) -> None:
    _deshadow_stdlib_profile()
    import struct
    import torch
    import torch.nn.functional as F

    def read_header(path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        return hdr

    def load_tensor(path, name):
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    st = os.path.join(DRAFTER_DIR, "model.safetensors")
    hdr = read_header(st)
    layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in hdr if ".layers." in k})
    specs = []  # (role, in, out, weight)
    w = load_tensor(st, "pre_projection.weight"); specs.append(("pre_projection", w.shape[1], w.shape[0], w))
    for i in layer_ids:
        qw = load_tensor(st, f"model.layers.{i}.self_attn.q_proj.weight")
        specs.append((f"layer{i}.q_proj", qw.shape[1], qw.shape[0], qw))
        ow = load_tensor(st, f"model.layers.{i}.self_attn.o_proj.weight")
        specs.append((f"layer{i}.o_proj", ow.shape[1], ow.shape[0], ow))
        gw = load_tensor(st, f"model.layers.{i}.mlp.gate_proj.weight")
        uw = load_tensor(st, f"model.layers.{i}.mlp.up_proj.weight")
        guw = torch.cat([gw, uw], dim=0)
        specs.append((f"layer{i}.gate_up", guw.shape[1], guw.shape[0], guw))
        dw = load_tensor(st, f"model.layers.{i}.mlp.down_proj.weight")
        specs.append((f"layer{i}.down_proj", dw.shape[1], dw.shape[0], dw))
    w = load_tensor(st, "post_projection.weight"); specs.append(("post_projection", w.shape[1], w.shape[0], w))
    cw = "masked_embedding.centroids.weight"
    if cw in hdr:
        c = load_tensor(st, cw); specs.append(("centroids_sampler", c.shape[1], c.shape[0], c))
    print(f"[draft] {len(specs)} per-pass GEMMs from {DRAFTER_DIR}", flush=True)

    class BF16Linear(torch.nn.Module):
        def __init__(self, w_bf16):
            super().__init__()
            self.weight = torch.nn.Parameter(w_bf16.cuda().to(torch.bfloat16), requires_grad=False)

        def forward(self, x):
            return F.linear(x, self.weight)

    mods = [(BF16Linear(w), inn) for (_r, inn, _o, w) in specs]
    bufs = [torch.randn(1, inn, device="cuda", dtype=torch.bfloat16) for (_, inn) in mods]

    def run_chain():
        for (mod, _), b in zip(mods, bufs):
            mod(b)

    # eager per-pass (consistent basis with the verify phase)
    with torch.inference_mode():
        for _ in range(warmup):
            run_chain()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            run_chain()
        e1.record(); torch.cuda.synchronize()
        draft_pass_us_eager = e0.elapsed_time(e1) / iters * 1000.0

    # graphed cross-check (deployed ONEGRAPH basis)
    draft_pass_us_graphed, captured = draft_pass_us_eager, False
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run_chain()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run_chain()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        draft_pass_us_graphed = e0.elapsed_time(e1) / iters * 1000.0
        captured = True
        del g
    except Exception as exc:
        print(f"[draft] graph capture failed ({exc!r}); graphed=eager", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    nan_clean = math.isfinite(draft_pass_us_eager) and math.isfinite(draft_pass_us_graphed)
    out = {
        "phase": "draft", "drafter_dir": DRAFTER_DIR, "n_gemms_per_pass": len(specs),
        "draft_pass_us_eager": draft_pass_us_eager,
        "draft_pass_us_graphed": draft_pass_us_graphed, "graph_captured": captured,
        "k7_chain_us_eager": draft_pass_us_eager * K_SPEC,
        "k7_chain_us_graphed": draft_pass_us_graphed * K_SPEC,
        "config": {"iters": iters, "warmup": warmup},
        "specs": [{"role": r, "in": i, "out": o} for (r, i, o, _w) in specs],
        "nan_clean": bool(nan_clean), "peak_gpu_gb": peak_gb,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    print(f"[draft] draft_pass_us eager={draft_pass_us_eager:.1f} graphed={draft_pass_us_graphed:.1f} "
          f"(K=7 chain eager={draft_pass_us_eager*K_SPEC:.0f}us) peak={peak_gb:.1f}GB", flush=True)
    print(f"DRAFT_DONE {out_path}", flush=True)


# ======================================================================================
# Subprocess + orchestration
# ======================================================================================
def run_phase_subprocess(args_list: list[str]) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def detect_knee(body_gemm: dict, flat_tol_pct: float = 5.0) -> dict:
    """First M where the body-GEMM cost leaves the flat (weight-bound) regime.

    Flat == body(M) within flat_tol_pct of body(M=1) (bandwidth-bound: M rows nearly
    free over the single weight read). The knee is the first M that exceeds it.
    """
    b1 = body_gemm[1]["sum_4gemm_us"]
    rel = {M: body_gemm[M]["sum_4gemm_us"] / b1 for M in M_SWEEP}
    knee = None
    for M in M_SWEEP:
        if (rel[M] - 1.0) * 100.0 > flat_tol_pct:
            knee = M
            break
    m8_weightbound = (rel[8] - 1.0) * 100.0 <= flat_tol_pct
    m32_weightbound = (rel[32] - 1.0) * 100.0 <= flat_tol_pct
    return {
        "body_gemm_rel_m1": rel,
        "gemm_M32_over_M8": body_gemm[32]["sum_4gemm_us"] / body_gemm[8]["sum_4gemm_us"],
        "flat_tol_pct": flat_tol_pct,
        "knee_first_M_above_flat": knee,
        "m8_still_weightbound": bool(m8_weightbound),
        "m32_still_weightbound": bool(m32_weightbound),
        "regime_m32": "weight-bound (near-free vs M=8)" if m32_weightbound else "compute-bound (M=32 materially > M=8)",
    }


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify_json = str(OUT_DIR / "verify_phase.json")
    draft_json = str(OUT_DIR / "draft_phase.json")

    if a.measure:
        run_phase_subprocess(["--phase", "verify", "--out", verify_json, "--ctx", str(a.ctx),
                              "--iters", str(a.iters), "--warmup", str(a.warmup),
                              "--repeats", str(a.repeats)])
        run_phase_subprocess(["--phase", "draft", "--out", draft_json,
                              "--iters", str(a.iters * 2), "--warmup", str(a.warmup)])
    vp = json.load(open(verify_json))
    dp = json.load(open(draft_json))

    verify_us = {int(k): v for k, v in vp["verify_us"].items()}
    body_gemm = {int(k): v for k, v in vp["body_gemm"].items()}
    num_layers = vp.get("compose_num_layers", vp["dims"]["num_layers"])  # DEPLOYED depth (37)

    # measured scalars -- PRIMARY basis = GRAPHED (deployed ONEGRAPH): the served step is
    # a CUDA-graph replay, so verify_us(M) (body+attn+lm_head all captured) and the draft
    # pass must both live in the graphed basis for g_d and the bridge to be deployment-
    # faithful. Eager is carried only as a launch-inflation cross-check.
    timing_basis = vp.get("timing_basis", "graphed")
    v8 = verify_us[M_VERIFY_SERVED]["verify_total_us"]
    v32 = verify_us[M_TREE]["verify_total_us"]
    d_graphed = dp["draft_pass_us_graphed"]
    d_eager = dp["draft_pass_us_eager"]
    d = d_graphed                                   # primary draft-pass cost (graphed)

    # (3) empirical g_d + served-step decomposition (graphed, deployed basis)
    g_d_measured = d / v8
    g_d_eager_crosscheck = d_eager / v8
    s_served_abs = v8 + K_SPEC * d                  # us
    bridge_c = STEP_SERVED / s_served_abs           # ms/us: wall-clock(us) -> NORMALIZED(ms); folds us->ms
    bridge_dimensionless = STEP_SERVED / (s_served_abs / 1e3)  # 0.184: normalized-ms / wall-clock-ms
    served_step_proj_ms = bridge_c * s_served_abs   # == STEP_SERVED by construction (provenance)
    roundtrip_resid_ms = abs(served_step_proj_ms - STEP_SERVED)
    g_d_vs_assumed_delta = g_d_measured - G_D_ASSUMED

    # (4) built-step projection (instruction 4): step_built = verify(32) + tree_draft_cost,
    # tree_draft_cost = draft_pass_us * n_tree, bridged to deployed units by the SAME c.
    knee = detect_knee(body_gemm)
    proj = {}
    for label, n_tree in TREE_PASS_MODELS.items():
        s_built_abs = v32 + n_tree * d
        step_built_ms = bridge_c * s_built_abs       # = STEP_SERVED * s_built/s_served
        built_over_served = s_built_abs / s_served_abs
        delta_vs_analytic = step_built_ms - STEP_BUILT_ANALYTIC
        confirm = bool(step_built_ms <= STEP_BUILT_ANALYTIC + CONFIRM_TOL_MS)
        # what TPS the projected built step implies at E[T]_both=4.512 (vs the 520.95 GO read)
        tps_at_built = K_CAL * (E_T_BUILT / step_built_ms) * TAU_TREE_CENTRAL
        proj[label] = {
            "n_tree_passes": n_tree, "tree_draft_cost_us": n_tree * d,
            "s_built_abs_us": s_built_abs, "built_over_served_ratio": built_over_served,
            "step_built_measured_proj_ms": step_built_ms,
            "delta_vs_analytic_1p085_ms": delta_vs_analytic,
            "verdict": "CONFIRM" if confirm else "DIVERGE",
            "implied_tps_at_ET4512": tps_at_built,
            "implied_tps_clears_500": bool(tps_at_built >= TARGET_TPS),
        }

    central = proj["central_b5"]
    step_built_measured_proj_ms = central["step_built_measured_proj_ms"]
    headline_verdict = central["verdict"]

    # ---- PHYSICAL FLOOR (why 1.2182 is a normalized unit, not wall-clock ms) ----
    shapes = vp["dims"]["shapes"]
    body_params = sum(int(o) * int(i) for (o, i) in shapes.values())
    body_int4_gb = body_params * num_layers * INT4_BYTES / 1e9
    lmhead_bf16_gb = LM_HEAD_VOCAB_DEPLOYED * vp["dims"]["hidden"] * BF16_BYTES / 1e9
    hbm_floor_ms = (body_int4_gb + lmhead_bf16_gb) / A10G_BW_GBPS * 1e3
    physical_floor = {
        "body_params_billion": body_params * num_layers / 1e9,
        "body_int4_gb": body_int4_gb, "lmhead_bf16_gb": lmhead_bf16_gb,
        "a10g_bw_gbps": A10G_BW_GBPS, "verify_hbm_floor_ms": hbm_floor_ms,
        "verify8_wall_ms": v8 / 1e3, "verify8_over_floor": (v8 / 1e3) / hbm_floor_ms,
        "step_served_normalized_ms": STEP_SERVED,
        "served_wall_clock_est_ms": STEP_SERVED * K_CAL_OVERHEAD_FACTOR,
        "kcal_overhead_factor": K_CAL_OVERHEAD_FACTOR,
        "note": (f"verify reads {body_int4_gb:.2f}GB int4 -> >= {hbm_floor_ms:.2f}ms HBM "
                 f"floor on the A10G; the banked {STEP_SERVED}ms step is therefore a "
                 f"NORMALIZED composition unit (wall-clock served step ~= "
                 f"{STEP_SERVED*K_CAL_OVERHEAD_FACTOR:.2f}ms via K_cal). The bridge c "
                 f"reconciles the two bases; the verdict rides on the basis-robust ratio."),
    }

    # ---- g_d-SENSITIVITY: the verdict hinges on g_d. Re-price the built step under the
    # ASSUMED g_d=0.168 (fleet anchor) using the measured verify-growth ratio v32/v8, so
    # the gap between the measured-g_d verdict and the analytic 1.085 (which BAKES IN
    # g_d=0.168) is explicit. Under assumed g_d the served split is verify-light:
    #   verify(8)_n = STEP_SERVED/(1+7*g_d_assumed); draft_n = g_d_assumed*verify(8)_n.
    # Built keeps the SAME measured growth: verify(32)_n = verify(8)_n*(v32/v8). ----
    v_ratio = v32 / v8
    verify8_n = STEP_SERVED / (1.0 + K_SPEC * G_D_ASSUMED)
    draft_n = G_D_ASSUMED * verify8_n
    gd_sensitivity = {}
    for label, n_tree in TREE_PASS_MODELS.items():
        built_n = verify8_n * v_ratio + n_tree * draft_n
        gd_sensitivity[label] = {
            "n_tree_passes": n_tree, "step_built_proj_ms": built_n,
            "delta_vs_analytic_1p085_ms": built_n - STEP_BUILT_ANALYTIC,
            "verdict": "CONFIRM" if built_n <= STEP_BUILT_ANALYTIC + CONFIRM_TOL_MS else "DIVERGE",
            "implied_tps_at_ET4512": K_CAL * (E_T_BUILT / built_n) * TAU_TREE_CENTRAL,
        }
    gd_sensitivity_note = (
        f"measured full-forward g_d={g_d_measured:.4f} is ~{G_D_ASSUMED/max(g_d_measured,1e-9):.0f}x "
        f"BELOW the assumed {G_D_ASSUMED}. Under MEASURED g_d the built step prices at "
        f"{step_built_measured_proj_ms:.3f}ms ({headline_verdict}); under the ASSUMED g_d "
        f"(fleet anchor, which the 1.085 bakes in) at {gd_sensitivity['central_b5']['step_built_proj_ms']:.3f}ms "
        f"({gd_sensitivity['central_b5']['verdict']}). The 10x g_d gap is the entire verdict "
        f"fork and is FLAGGED for advisor reconciliation (the fleet verify_us is likely not "
        f"a full-forward wall-clock quantity).")

    # (5) tau-path note (closes #252 follow-up #2): kanna #126 tau_tree central=1.0; the
    # served band is [0.9924,1.0]. Both linear(M=8) and tree(M=32) plausibly share it ->
    # tightens e_t_served band edges. Pure accounting import, no new measurement.
    tau_note = {
        "served_tau_band": list(TAU_BAND), "tau_tree_central_kanna126": TAU_TREE_CENTRAL,
        "shared_band": [max(TAU_BAND[0], TAU_BAND[0]), TAU_BAND[1]],
        "note": ("served (linear M=8) and built (tree M=32) plausibly share tau in "
                 "[0.9924,1.0] (kanna #126 tau_tree central=1.0): the e_t_served band "
                 "4.6828(tau=1)..4.7186(tau=0.9924) #252 stands; no widening from the M jump."),
    }

    # ---- self-test (PRIMARY) ----
    curve_finite = all(math.isfinite(verify_us[M]["verify_total_us"]) for M in M_SWEEP)
    knee_found = knee["knee_first_M_above_flat"] is not None or knee["m32_still_weightbound"]
    roundtrip_ok = roundtrip_resid_ms <= ROUNDTRIP_TOL_MS
    projection_finite = all(math.isfinite(p["step_built_measured_proj_ms"]) for p in proj.values())
    verdict_assigned = headline_verdict in ("CONFIRM", "DIVERGE")
    nan_clean = bool(vp["nan_clean"]) and bool(dp["nan_clean"]) and curve_finite and projection_finite
    peak_gb = max(vp["peak_gpu_gb"], dp["peak_gpu_gb"])
    vram_ok = peak_gb <= 24.0

    self_test = {
        "a_served_roundtrip_reproduces_1p2182": roundtrip_ok,    # provenance (exact by bridge)
        "b_built_proj_reported_with_verdict": bool(projection_finite and verdict_assigned),
        "c_verify_curve_and_knee_reported": bool(curve_finite and knee_found),
        "d_nan_clean": nan_clean,
        "e_peak_vram_le_24gb": bool(vram_ok),
    }
    built_step_roofline_grounding_self_test_passes = bool(
        roundtrip_ok and projection_finite and verdict_assigned and curve_finite
        and knee_found and nan_clean and vram_ok
    )

    report = {
        "pr": 257, "agent": "denken",
        "leg": "built-step roofline grounding (local forward-pass projection)",
        "imported_anchors": {
            "step_served_ms": STEP_SERVED, "step_built_analytic_ms": STEP_BUILT_ANALYTIC,
            "K_cal": K_CAL, "g_d_assumed": G_D_ASSUMED, "E_T_served": E_T_SERVED,
            "E_T_built": E_T_BUILT, "K_spec": K_SPEC, "M_verify_served": M_VERIFY_SERVED,
            "M_tree": M_TREE, "official_baseline": OFFICIAL_BASELINE, "go_read": GO_READ,
            "tau_band": list(TAU_BAND), "tau_tree_central": TAU_TREE_CENTRAL,
            "tree_draft_passes_empirical_lawine153": TREE_DRAFT_PASSES_EMPIRICAL,
        },
        # PRIMARY + TEST
        "built_step_roofline_grounding_self_test_passes": built_step_roofline_grounding_self_test_passes,
        "step_built_measured_proj_ms": step_built_measured_proj_ms,
        "step_built_measured_proj_verdict": headline_verdict,
        "g_d_measured": g_d_measured, "g_d_measured_vs_assumed_delta": g_d_vs_assumed_delta,
        "g_d_eager_crosscheck": g_d_eager_crosscheck, "timing_basis": timing_basis,
        # basis + verdict sensitivity (the crux)
        "physical_floor": physical_floor,
        "gd_sensitivity_assumed_0p168": gd_sensitivity,
        "gd_sensitivity_note": gd_sensitivity_note,
        # verify roofline
        "verify_us_ms": {M: verify_us[M]["verify_total_us"] / 1e3 for M in M_SWEEP},
        "verify_bodyonly_ms": {M: verify_us[M]["verify_bodyonly_us"] / 1e3 for M in M_SWEEP},
        "verify_v8_us": v8, "verify_v32_us": v32, "verify_v32_over_v8": v32 / v8,
        "knee": knee,
        # draft
        "draft_pass_us_eager": d_eager, "draft_pass_us_graphed": d_graphed,
        "k7_chain_us_eager": K_SPEC * d_eager,
        # step decomposition + bridge
        "s_served_abs_us": s_served_abs, "bridge_c_wall_to_normalized": bridge_c,
        "bridge_dimensionless_normalized_over_wall": bridge_dimensionless,
        "served_step_proj_ms": served_step_proj_ms, "roundtrip_resid_ms": roundtrip_resid_ms,
        "built_projection_models": proj, "tau_note": tau_note,
        # bookkeeping
        "self_test": self_test, "nan_clean": nan_clean, "peak_gpu_gb": peak_gb,
        "model_dir": vp["model_dir"], "body_identical_note": vp.get("body_identical_note"),
        "ctx": vp["ctx"], "num_layers": num_layers,
        "loaded_num_layers": vp.get("loaded_num_layers"),
        "deployed_num_layers": vp.get("deployed_num_layers"),
        "deployed_arch_note": vp.get("deployed_arch_note"),
        "verify_phase_json": verify_json, "draft_phase_json": draft_json,
    }
    report_path = OUT_DIR / "built_step_roofline_report.json"
    json.dump(report, open(report_path, "w"), indent=2, default=float)

    # ---- console summary ----
    print("\n========== BUILT-STEP ROOFLINE GROUNDING (PR #257) ==========", flush=True)
    print(f" verify_us(M) ms : " + "  ".join(f"M{M}={verify_us[M]['verify_total_us']/1e3:.3f}" for M in M_SWEEP), flush=True)
    print(f" body-GEMM rel M1: " + "  ".join(f"M{M}={knee['body_gemm_rel_m1'][M]:.3f}" for M in M_SWEEP), flush=True)
    print(f" knee            : first M>flat = {knee['knee_first_M_above_flat']}  "
          f"M=32 {knee['regime_m32']}  (gemm M32/M8={knee['gemm_M32_over_M8']:.3f})", flush=True)
    print(f" timing basis    : {timing_basis} (verify+draft); deployed ONEGRAPH = graphed", flush=True)
    print(f" draft_pass_us   : graphed {d_graphed:.1f} (PRIMARY)  eager {d_eager:.1f} (x-check)", flush=True)
    print(f" g_d_measured    : {g_d_measured:.4f} graphed  (assumed {G_D_ASSUMED}; delta {g_d_vs_assumed_delta:+.4f}; "
          f"eager x-check {g_d_eager_crosscheck:.4f})", flush=True)
    print(f" physical floor  : verify {physical_floor['body_int4_gb']:.2f}GB int4 -> "
          f">= {physical_floor['verify_hbm_floor_ms']:.2f}ms HBM floor; verify(8) wall "
          f"{physical_floor['verify8_wall_ms']:.2f}ms ({physical_floor['verify8_over_floor']:.2f}x floor)", flush=True)
    print(f"   => banked {STEP_SERVED}ms is a NORMALIZED unit (wall-clock served step "
          f"~{physical_floor['served_wall_clock_est_ms']:.2f}ms via K_cal x{K_CAL_OVERHEAD_FACTOR:.2f})", flush=True)
    print(f" served round-trip: verify(8)*(1+7*g_d) -> {served_step_proj_ms:.4f}ms "
          f"(banked {STEP_SERVED}; resid {roundtrip_resid_ms:.2e}ms; bridge basis-factor "
          f"{bridge_dimensionless:.4f} = normalized/wall, unit-reconciler not c~=1)", flush=True)
    print(f" --- built-step projection vs 1.085 analytic  [MEASURED g_d={g_d_measured:.4f}] ---", flush=True)
    for label, p in proj.items():
        print(f"   {label:>18s} n_tree={p['n_tree_passes']}: proj={p['step_built_measured_proj_ms']:.4f}ms "
              f"(Δ{p['delta_vs_analytic_1p085_ms']:+.4f}) ratio b/s={p['built_over_served_ratio']:.4f} "
              f"-> {p['verdict']}  (TPS@4.512={p['implied_tps_at_ET4512']:.1f}, clears500={p['implied_tps_clears_500']})",
              flush=True)
    print(f" --- g_d SENSITIVITY: same built priced under ASSUMED g_d={G_D_ASSUMED} (fleet anchor) ---", flush=True)
    for label, p in gd_sensitivity.items():
        print(f"   {label:>18s} n_tree={p['n_tree_passes']}: proj={p['step_built_proj_ms']:.4f}ms "
              f"(Δ{p['delta_vs_analytic_1p085_ms']:+.4f}) -> {p['verdict']}  (TPS@4.512={p['implied_tps_at_ET4512']:.1f})",
              flush=True)
    print(f" g_d NOTE        : {gd_sensitivity_note}", flush=True)
    print(f" HEADLINE (central b5, MEASURED g_d): step_built_measured_proj={step_built_measured_proj_ms:.4f}ms "
          f"vs 1.085 -> {headline_verdict}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY): {built_step_roofline_grounding_self_test_passes}  {self_test}", flush=True)
    print(f" peak VRAM: {peak_gb:.2f}GB   report -> {report_path}", flush=True)
    print("=============================================================\n", flush=True)

    if not a.no_wandb:
        try:
            log_wandb(report, a)
        except Exception as exc:
            print(f"[wandb] logging failed (non-fatal): {exc!r}", flush=True)

    return built_step_roofline_grounding_self_test_passes


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="denken", name=a.wandb_name, group=a.wandb_group,
        notes="PR#257 built-step roofline grounding: empirically bracket the 1.085ms tree step",
        config={"pr": 257, "ctx": report["ctx"], "num_layers": report["num_layers"],
                "M_sweep": M_SWEEP, "step_served": STEP_SERVED,
                "step_built_analytic": STEP_BUILT_ANALYTIC, "g_d_assumed": G_D_ASSUMED,
                "tree_pass_models": TREE_PASS_MODELS, "model_dir": report["model_dir"]},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    summary = {
        "built_step_roofline_grounding_self_test_passes": report["built_step_roofline_grounding_self_test_passes"],
        "step_built_measured_proj_ms": report["step_built_measured_proj_ms"],
        "step_built_measured_proj_verdict": report["step_built_measured_proj_verdict"],
        "g_d_measured": report["g_d_measured"],
        "g_d_measured_vs_assumed_delta": report["g_d_measured_vs_assumed_delta"],
        "g_d_eager_crosscheck": report["g_d_eager_crosscheck"],
        "verify_v8_us": report["verify_v8_us"], "verify_v32_us": report["verify_v32_us"],
        "verify_v32_over_v8": report["verify_v32_over_v8"],
        "draft_pass_us_eager": report["draft_pass_us_eager"],
        "draft_pass_us_graphed": report["draft_pass_us_graphed"],
        "bridge_c_wall_to_normalized": report["bridge_c_wall_to_normalized"],
        "bridge_dimensionless_normalized_over_wall": report["bridge_dimensionless_normalized_over_wall"],
        "served_step_proj_ms": report["served_step_proj_ms"],
        "roundtrip_resid_ms": report["roundtrip_resid_ms"],
        "gemm_M32_over_M8": report["knee"]["gemm_M32_over_M8"],
        "knee_first_M_above_flat": report["knee"]["knee_first_M_above_flat"],
        "m32_still_weightbound": report["knee"]["m32_still_weightbound"],
        "step_built_proj_conservative_k7_ms": report["built_projection_models"]["conservative_k7"]["step_built_measured_proj_ms"],
        "step_built_proj_pessimistic_depth9_ms": report["built_projection_models"]["pessimistic_depth9"]["step_built_measured_proj_ms"],
        # physical floor (basis evidence) + g_d-assumed sensitivity (verdict fork)
        "verify_hbm_floor_ms": report["physical_floor"]["verify_hbm_floor_ms"],
        "verify8_wall_ms": report["physical_floor"]["verify8_wall_ms"],
        "verify8_over_floor": report["physical_floor"]["verify8_over_floor"],
        "body_int4_gb": report["physical_floor"]["body_int4_gb"],
        "served_wall_clock_est_ms": report["physical_floor"]["served_wall_clock_est_ms"],
        "step_built_proj_assumed_gd_central_ms": report["gd_sensitivity_assumed_0p168"]["central_b5"]["step_built_proj_ms"],
        "step_built_proj_assumed_gd_verdict": report["gd_sensitivity_assumed_0p168"]["central_b5"]["verdict"],
        "deployed_num_layers": report["deployed_num_layers"],
        "loaded_num_layers": report["loaded_num_layers"],
        "peak_gpu_gb": report["peak_gpu_gb"],
    }
    for M in M_SWEEP:
        summary[f"verify_us_ms_M{M}"] = report["verify_us_ms"][M]
        summary[f"verify_bodyonly_ms_M{M}"] = report["verify_bodyonly_ms"][M]
        summary[f"body_gemm_rel_m1_M{M}"] = report["knee"]["body_gemm_rel_m1"][M]
    for k, v in summary.items():
        run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["verify", "draft"], default=None,
                    help="internal: run a GPU phase (subprocess). Omit for the orchestrator.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true", help="emit the PRIMARY self-test")
    ap.add_argument("--measure", action="store_true",
                    help="run the GPU measurement phases (else reuse cached phase JSONs)")
    ap.add_argument("--ctx", type=int, default=528, help="served decode KV context length")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="built-step-roofline")
    ap.add_argument("--wandb_name", dest="wandb_name", default="denken/built-step-roofline-grounding")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.phase == "verify":
        phase_verify(a.out, a.ctx, a.iters, a.warmup, a.repeats)
        return 0
    if a.phase == "draft":
        phase_draft(a.out, a.iters, a.warmup)
        return 0
    ok = orchestrate(a)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
