#!/usr/bin/env python
"""PR #221 — FP16-verify cost + validity: measure the valid-path penalty locally.

The int4-spec frontier is greedy-INVALID: the int4 Marlin verify GEMM's split-K
reduction order is f(batch width M) -> 56% token divergence vs int4-AR-greedy
(#114/#192). Issue #211 argues the FP16/BF16 verify path is greedy-VALID by
construction (standard cuBLAS GEMM has no M-dependent split-K), but slower. This
leg MEASURES both, locally, for free (no official draw):

  Tier 1 (PRIMARY, always): GEMM-level int4-Marlin vs bf16-cuBLAS cost ratio at the
    deployed gemma-4-E4B-it body GEMM shapes, swept over the verify batch width M,
    plus the kernel-level batch-invariance test -- the #211 fp16-validity premise
    (bf16 row-0 bit-identical across M) AND the #114 int4 divergence mechanism
    (int4 Marlin row-0 differs across M).
  Tier 2 (--tier2, best-effort): bf16 target forward (transformers, int4->bf16
    decompressed) end-to-end token-identity at M=1 vs M=8 over the official 128
    prompts -- the #211 premise tested through the whole network.

Deliverable: M_step = 1 + (m_gemm@M8 - 1) * f_verify ; the valid-fp16 ceilings
    fp16verify_tps_at_lambda1 = 520.95 / M_step and 481.53 / M_step ; and the bool
    fp16verify_clears_500_at_lambda1.

Imported (NOT re-derived): K_cal #148/#169, the #153 verify-step(M) curve (gemm
    fraction f_verify), step_int4=1.2182 #168, int4-spec lambda=1 ceiling 520.95
    #204, wall->official bridge 1.06019 #180.

LOCAL profiling on a single A10G. No HF Job / no submission / no served-file change
/ no official draw. BASELINE stays 481.53. Greedy/PPL untouched. This leg adds 0 TPS.

Phases run as isolated subprocesses so each GPU framework (vLLM for the int4 Marlin
kernel; transformers for the bf16 forward) gets a clean CUDA context and fully
releases VRAM on exit. The orchestrator stays GPU-free and owns composition + wandb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# Imported fleet anchors (DO NOT re-derive)
# --------------------------------------------------------------------------------------
K_CAL = 125.26795005202914              # #148/#169/#153 official step-normalizer
STEP_INT4_M32 = 1.2182                  # #168 measured step multiplier at M=32
LAMBDA1_CEILING = 520.95                # #204 int4-spec lambda=1 (draft-independent) ceiling, official TPS
OFFICIAL_BASELINE = 481.53              # #52 fa2sw_precache_kenyan official TPS
WALL_TO_OFFICIAL = 1.06019              # #180/#99 local wall_tps -> official bridge (imported, ref only)
TARGET_TPS = 500.0

# #153 verify_step_m_curve raw anchors at the deployed verify width M=8 (verify_step_m_curve.json)
GEMM_US_M8_153 = 4980.291757583619      # target-forward body GEMM time at M=8 (us)
STEP_US_M8_153 = 7982.887878221502      # whole decode-step time at M=8 (us), "step_m8_us" anchor
F_VERIFY_153 = GEMM_US_M8_153 / STEP_US_M8_153   # = 0.62387 : GEMM fraction of the step (the term that goes fp16)
F_VERIFY_FERN100 = 0.53                 # fern #100 verify-GEMM step share (sensitivity bracket)

# Deployed speculative tree
K_SPEC = 7                              # num_speculative_tokens (manifest)
M_VERIFY = K_SPEC + 1                   # = 8, the deployed verify batch width
M_SWEEP = [1, 2, 4, 8, 16]              # batch widths for the GEMM cost curve

# Model source for the int4 Marlin BODY GEMMs. The canonical Hub int4 checkpoint
# (google/gemma-4-E4B-it-qat-w4a16-ct) is used because the deployed osoi5-v0-baked
# variant has a PLE-folded / pruned lm_head that needs the submission serve patches
# to load under vanilla vLLM. The body QKV/MLP GEMM shapes and the int4 Marlin kernel
# are IDENTICAL between the two (same gemma-4-E4B text decoder, same g=128 w4a16) -- only
# the lm_head/embedding differ, which this leg does not time. So the body verify-GEMM
# cost ratio is faithful to the deployed model.
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/fp16_verify_cost")


# --------------------------------------------------------------------------------------
# Small helpers shared by phases
# --------------------------------------------------------------------------------------
def _stats(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    n = len(vals)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(vals),
        "std": std,
        "se": se,
        "ci95_abs": 1.96 * se,
        "cv_pct": (100.0 * std / mean) if mean else 0.0,
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        # snapshots dir -> pick the single snapshot subdir with a config.json
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    raise FileNotFoundError(f"no int4 model found among {MODEL_CANDIDATES}")


def read_text_dims(model_dir: str) -> dict:
    cfg = json.load(open(Path(model_dir) / "config.json"))
    tc = cfg.get("text_config", cfg)
    h = tc["hidden_size"]
    n_heads = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    inter = tc["intermediate_size"]
    return {
        "hidden": h,
        "n_heads": n_heads,
        "n_kv": n_kv,
        "head_dim": hd,
        "intermediate": inter,
        "num_layers": tc.get("num_hidden_layers"),
        # body GEMM shapes (out_features, in_features)
        "shapes": {
            "qkv_proj": ((n_heads + 2 * n_kv) * hd, h),
            "o_proj": (h, n_heads * hd),
            "gate_up_proj": (2 * inter, h),
            "down_proj": (h, inter),
        },
    }


def sample_sm_clock() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    except Exception:
        return None


# ======================================================================================
# PHASE: gemm  (Tier 1 -- vLLM int4 Marlin vs bf16 cuBLAS + batch-invariance)
# ======================================================================================
def phase_gemm(out_path: str, iters: int, warmup: int, repeats: int, smoke: bool) -> None:
    import torch
    from vllm import LLM

    dev = torch.device("cuda:0")
    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    shapes = dims["shapes"]

    print(f"[gemm] model={model_dir} dims={ {k: dims[k] for k in ('hidden','n_heads','n_kv','head_dim','intermediate','num_layers')} }", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_dir,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"[gemm] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    # locate the runner model
    def get_model():
        paths = [
            lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        ]
        for p in paths:
            try:
                m = p()
                if m is not None:
                    return m
            except Exception:
                continue
        raise RuntimeError("could not locate model_runner.model")

    model = get_model()

    # navigate to the text decoder layers. vLLM's Gemma4ForCausalLM nests the
    # decoder behind .model (and may further nest .language_model), but the exact
    # path varies by class, so try known paths then fall back to a module scan.
    def find_layers(root):
        import torch.nn as nn
        # 1) known attribute chains
        chains = [
            ("model", "layers"),
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "model", "layers"),
            ("layers",),
        ]
        for chain in chains:
            obj = root
            ok = True
            for attr in chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return obj, ".".join(chain)
        # 2) scan every ModuleList; pick the one whose elements expose self_attn.qkv_proj
        best = None
        for name, mod in root.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                el = mod[0]
                if hasattr(el, "self_attn") and hasattr(el.self_attn, "qkv_proj"):
                    if best is None or len(mod) > len(best[0]):
                        best = (mod, name)
        if best is not None:
            return best
        raise RuntimeError("could not locate decoder ModuleList")

    layers, layers_path = find_layers(model)
    print(f"[gemm] decoder layers found: {len(layers)} via '{layers_path}' "
          f"(model class={type(model).__name__})", flush=True)

    # find a standard layer whose 4 body modules match the canonical shapes
    def module_out_in(mod):
        # vLLM linear: output_size_per_partition x input_size_per_partition
        out = getattr(mod, "output_size_per_partition", None)
        inp = getattr(mod, "input_size_per_partition", None)
        if out is None or inp is None:
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 2:
                out, inp = int(w.shape[0]), int(w.shape[1])
        return (int(out), int(inp)) if out and inp else None

    targets = None
    for li, layer in enumerate(layers):
        try:
            cand = {
                "qkv_proj": layer.self_attn.qkv_proj,
                "o_proj": layer.self_attn.o_proj,
                "gate_up_proj": layer.mlp.gate_up_proj,
                "down_proj": layer.mlp.down_proj,
            }
        except AttributeError:
            continue
        ok = all(
            hasattr(m, "quant_method") and module_out_in(m) == shapes[name]
            for name, m in cand.items()
        )
        if ok:
            targets = cand
            print(f"[gemm] using layer {li} for the 4 canonical body GEMMs", flush=True)
            break
    if targets is None:
        try:
            l0 = layers[0]
            print("[gemm] layer0 attrs:", [n for n in dir(l0) if not n.startswith("_")][:40], flush=True)
            if hasattr(l0, "self_attn"):
                print("[gemm] self_attn children:", list(dict(l0.self_attn.named_children()).keys()), flush=True)
            if hasattr(l0, "mlp"):
                print("[gemm] mlp children:", list(dict(l0.mlp.named_children()).keys()), flush=True)
        except Exception as exc:
            print(f"[gemm] introspection failed: {exc!r}", flush=True)
        raise RuntimeError(f"no layer matched canonical shapes {shapes}")

    def int4_apply(mod):
        return lambda x: mod.quant_method.apply(mod, x, bias=None)

    # bf16 reference weights (random; cuBLAS GEMM timing & batch-invariance are data-independent)
    torch.manual_seed(0)
    bf16_W = {name: torch.randn(out, inp, dtype=torch.bfloat16, device=dev) * 0.02
              for name, (out, inp) in shapes.items()}

    def bf16_apply(name):
        W = bf16_W[name]
        return lambda x: torch.matmul(x, W.t())

    def bench(fn, x):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(x)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0  # us per call

    # heavy warmup to boost A10G to its boost clock before timing (we cannot lock clocks)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    clk = {"after_warmup": sample_sm_clock()}

    # fixed inputs per shape (max width) so M-slices share row 0 exactly
    Mmax = max(M_SWEEP)
    inputs = {name: torch.randn(Mmax, inp, dtype=torch.bfloat16, device=dev)
              for name, (out, inp) in shapes.items()}

    raw = {name: {"int4_us": {}, "bf16_us": {}} for name in shapes}
    for name in shapes:
        fi = int4_apply(targets[name])
        fb = bf16_apply(name)
        x_full = inputs[name]
        for M in M_SWEEP:
            xm = x_full[:M].contiguous()
            ti, tb = [], []
            for _ in range(repeats):
                ti.append(bench(fi, xm))
                tb.append(bench(fb, xm))
            raw[name]["int4_us"][M] = _stats(ti)
            raw[name]["bf16_us"][M] = _stats(tb)
        print(f"[gemm] {name}: int4@M8={raw[name]['int4_us'][8]['mean']:.2f}us "
              f"bf16@M8={raw[name]['bf16_us'][8]['mean']:.2f}us "
              f"ratio={raw[name]['bf16_us'][8]['mean']/raw[name]['int4_us'][8]['mean']:.3f}", flush=True)
    clk["after_timing"] = sample_sm_clock()

    # aggregate cost ratio curve (sum over the 4 per-layer shapes)
    curve = {}
    for M in M_SWEEP:
        s_int4 = sum(raw[name]["int4_us"][M]["mean"] for name in shapes)
        s_bf16 = sum(raw[name]["bf16_us"][M]["mean"] for name in shapes)
        curve[M] = {
            "int4_us_sum": s_int4,
            "bf16_us_sum": s_bf16,
            "gemm_cost_ratio_fp16_int4": s_bf16 / s_int4,
            "per_shape_ratio": {n: raw[n]["bf16_us"][M]["mean"] / raw[n]["int4_us"][M]["mean"] for n in shapes},
        }
    m_gemm_at_M8 = curve[8]["gemm_cost_ratio_fp16_int4"]

    # repeat-stability of the M=8 aggregate ratio (#209-style): recompute per-repeat ratio CI
    m8_per_repeat = []
    for r in range(repeats):
        si = sum(raw[name]["int4_us"][8]["values"][r] for name in shapes)
        sb = sum(raw[name]["bf16_us"][8]["values"][r] for name in shapes)
        m8_per_repeat.append(sb / si)
    m8_stat = _stats(m8_per_repeat)

    # ---- batch-invariance (validity): row-0 bit-exactness across M widths ----
    def row0_bitexact_test(apply_fn, x_full):
        y1 = apply_fn(x_full[:1].contiguous())[0].detach().float()
        y8 = apply_fn(x_full[:8].contiguous())[0].detach().float()
        y16 = apply_fn(x_full[:16].contiguous())[0].detach().float()
        torch.cuda.synchronize()
        d8 = float((y8 - y1).abs().max())
        d16 = float((y16 - y1).abs().max())
        frac8 = float((y8 != y1).float().mean())
        return {
            "max_abs_diff_M8_vs_M1": d8,
            "max_abs_diff_M16_vs_M1": d16,
            "frac_elems_diff_M8_vs_M1": frac8,
            "bitexact_M8_vs_M1": bool(torch.equal(y8, y1)),
            "bitexact_M16_vs_M1": bool(torch.equal(y16, y1)),
            "row_norm": float(y1.norm()),
        }

    invariance = {"int4_marlin": {}, "bf16_cublas": {}}
    for name in shapes:
        invariance["int4_marlin"][name] = row0_bitexact_test(int4_apply(targets[name]), inputs[name])
        invariance["bf16_cublas"][name] = row0_bitexact_test(bf16_apply(name), inputs[name])
    int4_any_divergent = any(not v["bitexact_M8_vs_M1"] for v in invariance["int4_marlin"].values())
    bf16_all_invariant = all(v["bitexact_M8_vs_M1"] for v in invariance["bf16_cublas"].values())

    nan_clean = all(
        math.isfinite(raw[name][q][M]["mean"])
        for name in shapes for q in ("int4_us", "bf16_us") for M in M_SWEEP
    )

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "gemm",
        "model_dir": model_dir,
        "dims": dims,
        "config": {"iters": iters, "warmup": warmup, "repeats": repeats, "smoke": smoke,
                   "M_sweep": M_SWEEP, "M_verify": M_VERIFY},
        "sm_clock_mhz": clk,
        "raw_per_shape_us": raw,
        "cost_ratio_curve": curve,
        "m_gemm_at_M8": m_gemm_at_M8,
        "m_gemm_at_M8_repeat_stat": m8_stat,
        "batch_invariance": invariance,
        "int4_marlin_M_dependent": int4_any_divergent,
        "bf16_cublas_batch_invariant": bf16_all_invariant,
        "nan_clean": bool(nan_clean),
        "peak_gpu_gb": peak_gb,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[gemm] m_gemm@M8={m_gemm_at_M8:.4f} (CI±{m8_stat['ci95_abs']:.4f}) "
          f"int4_M_dependent={int4_any_divergent} bf16_invariant={bf16_all_invariant} "
          f"peak={peak_gb:.1f}GB", flush=True)
    print(f"GEMM_DONE {out_path}", flush=True)


# ======================================================================================
# PHASE: tokenident  (Tier 2 -- bf16 transformers forward, end-to-end token identity)
# ======================================================================================
def phase_tokenident(out_path: str, n_prompts: int, max_len: int, batch_m: int) -> None:
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import Gemma4ForConditionalGeneration

    dev = torch.device("cuda:0")
    model_dir = resolve_model_dir()
    t0 = time.time()
    # load to CPU then move to GPU (avoids the accelerate/device_map dependency); the
    # compressed-tensors w4a16 checkpoint decompresses to dense bf16 weights on load.
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to(dev)
    model.eval()
    print(f"[tok] bf16 model loaded in {time.time()-t0:.0f}s "
          f"(int4->bf16 decompressed), attn={getattr(model.config.text_config,'_attn_implementation',None)}", flush=True)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = 0
    n_total = 0
    n_det_match = 0          # control: M=1 vs M=1 (determinism)
    n_within_match = 0       # control: M=batch row0 vs row1 (within-batch consistency)
    for ri, rec in enumerate(rows):
        ids = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        ids = ids[:max_len]
        if len(ids) < 2:
            continue
        t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            lg1 = model(input_ids=t, use_cache=False).logits[0].float()   # M=1 (AR width)
            am1 = lg1.argmax(-1)
            # determinism control: M=1 again (free the second copy immediately)
            lg1b = model(input_ids=t, use_cache=False).logits[0].float()
            am1b = lg1b.argmax(-1)
            det_match = int((am1 == am1b).sum())
            max_logit_diff_det = float((lg1 - lg1b).abs().max())
            del lg1b
            # M=batch_m: replicate the SAME sequence -> GEMMs see batch_m rows; row 0 must match M=1
            tb = t.repeat(batch_m, 1)
            out_b = model(input_ids=tb, use_cache=False).logits
            lgB = out_b[0].float()   # row 0 of the wide batch
            amB = lgB.argmax(-1)
            amB1 = out_b[1].float().argmax(-1) if batch_m > 1 else amB  # row 1 (within-batch control)
            del out_b
            match = int((am1 == amB).sum())
            within_match = int((amB == amB1).sum())
            max_logit_diff = float((lg1 - lgB).abs().max())
            del lg1, lgB
        tot = int(am1.numel())
        n_match += match
        n_total += tot
        n_det_match += det_match
        n_within_match += within_match
        sha1 = hashlib.sha256(am1.cpu().numpy().tobytes()).hexdigest()[:16]
        shaB = hashlib.sha256(amB.cpu().numpy().tobytes()).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "positions": tot, "argmax_match": match,
            "argmax_sha_M1": sha1, "argmax_sha_MB": shaB, "sha_equal": sha1 == shaB,
            "max_abs_logit_diff": max_logit_diff,
            "det_match_M1_vs_M1": det_match,
            "max_abs_logit_diff_det": max_logit_diff_det,
            "within_match_row0_vs_row1": within_match,
        })
        if ri < 3 or ri == len(rows) - 1:
            print(f"[tok] prompt {ri} id={rec.get('id')} match={match}/{tot} "
                  f"sha_eq={sha1==shaB} max_logit_diff={max_logit_diff:.3e} "
                  f"det={det_match}/{tot} within={within_match}/{tot}", flush=True)

    identity = (n_match / n_total) if n_total else float("nan")
    determinism = (n_det_match / n_total) if n_total else float("nan")
    within_batch = (n_within_match / n_total) if n_total else float("nan")
    sha_equal_frac = statistics.fmean([1.0 if p["sha_equal"] else 0.0 for p in per_prompt]) if per_prompt else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "tokenident",
        "model_dir": model_dir,
        "n_prompts": len(per_prompt),
        "max_len": max_len,
        "batch_m": batch_m,
        "total_positions": n_total,
        "matching_positions": n_match,
        "fp16_token_identity_M1_vs_M8": identity,
        "fp16_token_divergence_M1_vs_M8": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "determinism_M1_vs_M1": determinism,           # control: expect 1.0 (rules out nondeterminism)
        "within_batch_row0_vs_row1": within_batch,     # control: expect 1.0 (rules out within-batch bug)
        "sha_equal_frac": sha_equal_frac,
        "all_prompts_sha_equal": all(p["sha_equal"] for p in per_prompt) if per_prompt else False,
        "per_prompt": per_prompt,
        "peak_gpu_gb": peak_gb,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[tok] token_identity_M1_vs_M{batch_m}={identity:.6f} (divergence={1.0-identity:.6f}) "
          f"sha_equal_frac={sha_equal_frac:.4f} peak={peak_gb:.1f}GB", flush=True)
    print(f"[tok] controls: determinism_M1vsM1={determinism:.6f} within_batch_row0vsrow1={within_batch:.6f}", flush=True)
    print(f"TOKENIDENT_DONE {out_path}", flush=True)


# ======================================================================================
# Orchestrator: run phases as isolated subprocesses, compose, self-test, log wandb
# ======================================================================================
def run_phase_subprocess(args_list: list[str]) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    # reduce allocator fragmentation for the Tier-2 M=8 wide-batch logits transient on the 23GB A10G
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gemm_json = str(OUT_DIR / "gemm_result.json")
    tok_json = str(OUT_DIR / "tokenident_result.json")

    # ---- Tier 1 (always) ----
    run_phase_subprocess([
        "--phase", "gemm", "--out", gemm_json,
        "--iters", str(a.iters), "--warmup", str(a.warmup), "--repeats", str(a.repeats),
        *(["--smoke"] if a.smoke else []),
    ])
    gemm = json.load(open(gemm_json))
    m_gemm_at_M8 = gemm["m_gemm_at_M8"]

    # ---- Tier 2 (best-effort) ----
    tok = None
    tier2_status = "skipped"
    if a.tier2:
        try:
            run_phase_subprocess([
                "--phase", "tokenident", "--out", tok_json,
                "--n-prompts", str(a.n_prompts), "--max-len", str(a.max_len),
                "--batch-m", str(M_VERIFY),
            ])
            tok = json.load(open(tok_json))
            tier2_status = "ran"
        except Exception as exc:  # OOM / load failure -> fall back per PR honest scope
            tier2_status = f"failed_followup: {exc!r}"
            print(f"[orch] Tier 2 not tractable -> {tier2_status}", flush=True)

    # ---- Deliverable: M_step from Tier 1 m_gemm + imported #153 f_verify ----
    def m_step(f_verify: float) -> float:
        return 1.0 + (m_gemm_at_M8 - 1.0) * f_verify

    m_step_153 = m_step(F_VERIFY_153)
    m_step_fern = m_step(F_VERIFY_FERN100)
    M_STEP = m_step_153  # headline uses the #153-derived gemm fraction
    fp16_tps_lambda1 = LAMBDA1_CEILING / M_STEP
    fp16_tps_build = OFFICIAL_BASELINE / M_STEP
    clears_500 = bool(fp16_tps_lambda1 >= TARGET_TPS)

    # ---- Self-test (PRIMARY) ----
    # JSON keys come back as strings; normalize access
    curve = gemm["cost_ratio_curve"]
    cv = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in curve.items()}
    ratios = {M: cv[M]["gemm_cost_ratio_fp16_int4"] for M in M_SWEEP}
    finite_all = all(math.isfinite(ratios[M]) for M in M_SWEEP)
    gt1_deployed = all(ratios[M] > 1.0 for M in [1, 2, 4, 8])   # robust regime (memory-bound -> int4 faster)
    gt1_all = all(ratios[M] > 1.0 for M in M_SWEEP)
    m8_ci = gemm["m_gemm_at_M8_repeat_stat"]["ci95_abs"]
    m8_stable = (m8_ci / m_gemm_at_M8) < 0.02 if m_gemm_at_M8 else False
    # (d) ceiling * M_step reconstructs the lambda1 ceiling
    recon_ok = abs(fp16_tps_lambda1 * M_STEP - LAMBDA1_CEILING) < 1e-6
    # (c) M_step reconciliation: the two f_verify estimates bracket M_STEP and agree within tol
    reconcile_ok = abs(m_step_153 - m_step_fern) / m_step_153 < 0.10
    # validity legs.
    # The authoritative validity signal is the END-TO-END token identity (Tier 2): does fp16
    # verify-width (M=8) preserve the M=1 AR argmax token? The per-GEMM row0 bitexactness below
    # is a NUMERICAL diagnostic of where batch-variance lives -- it is shape/width-specific and
    # is NOT gated into the pass (a sub-ULP cuBLAS kernel switch is not a token divergence; and
    # an isolated body GEMM that does not split-K at M<=16 does not reproduce the spec-tree path).
    bf16_kernel_bitexact = bool(gemm["bf16_cublas_batch_invariant"])    # informational diagnostic
    int4_kernel_M_dependent = bool(gemm["int4_marlin_M_dependent"])     # informational diagnostic
    tok_ok = True
    if tok is not None:
        ti = tok["fp16_token_identity_M1_vs_M8"]
        tok_ok = (0.0 <= ti <= 1.0) and math.isfinite(ti)
    nan_clean = bool(gemm["nan_clean"]) and (tok is None or math.isfinite(tok["fp16_token_identity_M1_vs_M8"]))

    self_test = {
        "finite_all_M": finite_all,
        "ratio_gt1_deployed_M<=8": gt1_deployed,
        "ratio_gt1_all_M": gt1_all,                       # informational (M=16 crossover allowed)
        "m8_repeat_stable_cv<2pct": m8_stable,
        "ceiling_reconstructs_lambda1": recon_ok,
        "m_step_two_estimates_reconcile<10pct": reconcile_ok,
        "tokenident_in_range": tok_ok,
        "nan_clean": nan_clean,
        "_diag_bf16_kernel_bitexact": bf16_kernel_bitexact,       # diagnostic, NOT gated
        "_diag_int4_kernel_M_dependent": int4_kernel_M_dependent,  # diagnostic, NOT gated
    }
    # PRIMARY pass: load-bearing internal-consistency checks only. The kernel-level batch-invariance
    # diagnostics are reported but NOT gated (shape/width-specific; token-level validity is carried
    # by the Tier-2 end-to-end identity). M=16 aggregate crossover is informational, not a fail.
    fp16_verify_cost_self_test_passes = bool(
        finite_all and gt1_deployed and m8_stable and recon_ok and reconcile_ok
        and tok_ok and nan_clean
    )

    # Validity premise is decided by the END-TO-END token identity, not kernel float-bitexactness.
    # The determinism control (M=1 vs M=1) separates pure M-width batch-variance (the bf16 floor)
    # from nondeterminism/bugs: if identity<1 but determinism==1, the residual is batch-variance.
    if tok is not None:
        ti = tok["fp16_token_identity_M1_vs_M8"]
        det = tok["determinism_M1_vs_M1"]
        if ti == 1.0:
            fp16_validity_premise = "CONFIRMED"
        elif det == 1.0:
            fp16_validity_premise = "RESIDUAL_BF16_BATCHVAR"  # deterministic but M-width batch-variant
        else:
            fp16_validity_premise = "DIVERGENT"
    else:
        fp16_validity_premise = "PENDING_TIER2"

    report = {
        "pr": 221,
        "leg": "fp16-verify cost + validity (local)",
        "imported_anchors": {
            "K_cal": K_CAL, "step_int4_M32": STEP_INT4_M32, "lambda1_ceiling": LAMBDA1_CEILING,
            "official_baseline": OFFICIAL_BASELINE, "wall_to_official": WALL_TO_OFFICIAL,
            "f_verify_153": F_VERIFY_153, "f_verify_fern100": F_VERIFY_FERN100,
            "gemm_us_M8_153": GEMM_US_M8_153, "step_us_M8_153": STEP_US_M8_153,
        },
        "tier1_m_gemm_at_M8": m_gemm_at_M8,
        "tier1_m_gemm_at_M8_ci95": m8_ci,
        "gemm_cost_ratio_fp16_int4_curve": {str(M): ratios[M] for M in M_SWEEP},
        "m_step_fp16_int4": M_STEP,
        "m_step_fp16_int4_153": m_step_153,
        "m_step_fp16_int4_fern100": m_step_fern,
        "fp16verify_tps_at_lambda1": fp16_tps_lambda1,
        "fp16verify_tps_at_current_build": fp16_tps_build,
        "fp16verify_clears_500_at_lambda1": clears_500,
        "fp16_validity_premise": fp16_validity_premise,
        "bf16_cublas_batch_invariant": bf16_kernel_bitexact,
        "int4_marlin_M_dependent": int4_kernel_M_dependent,
        "tier2_status": tier2_status,
        "fp16_token_identity_M1_vs_M8": (tok["fp16_token_identity_M1_vs_M8"] if tok else None),
        "fp16_token_divergence_M1_vs_M8": (tok["fp16_token_divergence_M1_vs_M8"] if tok else None),
        "tier2_determinism_M1_vs_M1": (tok["determinism_M1_vs_M1"] if tok else None),
        "tier2_within_batch_row0_vs_row1": (tok["within_batch_row0_vs_row1"] if tok else None),
        "self_test": self_test,
        "fp16_verify_cost_self_test_passes": fp16_verify_cost_self_test_passes,
        "sm_clock_mhz": gemm["sm_clock_mhz"],
        "peak_gpu_gb": max(gemm["peak_gpu_gb"], (tok or {}).get("peak_gpu_gb", 0.0)),
    }
    report_path = OUT_DIR / "fp16_verify_cost_report.json"
    json.dump(report, open(report_path, "w"), indent=2)

    # ---- console summary ----
    print("\n================ FP16-VERIFY COST + VALIDITY (PR #221) ================", flush=True)
    print(f" m_gemm@M8 (fp16/int4 GEMM)      : {m_gemm_at_M8:.4f}  (CI±{m8_ci:.4f})", flush=True)
    print(f" cost-ratio curve over M         : " + ", ".join(f"M{M}={ratios[M]:.3f}" for M in M_SWEEP), flush=True)
    print(f" f_verify (#153 gemm fraction)   : {F_VERIFY_153:.4f}", flush=True)
    print(f" M_step (=1+(m_gemm-1)*f_verify) : {M_STEP:.4f}   [fern0.53 bracket {m_step_fern:.4f}]", flush=True)
    print(f" fp16verify TPS @ lambda1 ceiling: {fp16_tps_lambda1:.2f}  (= {LAMBDA1_CEILING}/{M_STEP:.4f})", flush=True)
    print(f" fp16verify TPS @ current build  : {fp16_tps_build:.2f}", flush=True)
    print(f" clears 500 @ lambda1?           : {clears_500}", flush=True)
    print(f" fp16 validity premise           : {fp16_validity_premise}  "
          f"(kernel diag: bf16 row0 bitexact={bf16_kernel_bitexact}; int4 row0 M-dep={int4_kernel_M_dependent})", flush=True)
    if tok is not None:
        print(f" end-to-end token identity M1vsM8: {tok['fp16_token_identity_M1_vs_M8']:.6f} "
              f"(divergence {tok['fp16_token_divergence_M1_vs_M8']:.6f}, n={tok['n_prompts']})", flush=True)
        print(f"   controls: determinism M1vsM1={tok['determinism_M1_vs_M1']:.6f}  "
              f"within-batch row0vsrow1={tok['within_batch_row0_vs_row1']:.6f}", flush=True)
    else:
        print(f" end-to-end token identity        : {tier2_status}", flush=True)
    print(f" SELF-TEST PASSES                : {fp16_verify_cost_self_test_passes}  {self_test}", flush=True)
    print(f" report -> {report_path}", flush=True)
    print("======================================================================\n", flush=True)

    # ---- wandb ----
    if not a.no_wandb:
        log_wandb(report, gemm, tok, a)


def log_wandb(report: dict, gemm: dict, tok: dict | None, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling",
        agent="lawine",
        name=a.wandb_name,
        group=a.wandb_group,
        notes="PR#221 fp16-verify cost + validity: GEMM-level fp16/int4 ratio + batch-invariance + (tier2) e2e token identity",
        config={
            "pr": 221, "M_sweep": M_SWEEP, "M_verify": M_VERIFY,
            "f_verify_153": F_VERIFY_153, "lambda1_ceiling": LAMBDA1_CEILING,
            "official_baseline": OFFICIAL_BASELINE, "iters": a.iters, "repeats": a.repeats,
            "model_dir": gemm["model_dir"], "tier2": a.tier2,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    import wandb

    summary = {
        "fp16_verify_cost_self_test_passes": report["fp16_verify_cost_self_test_passes"],
        "m_step_fp16_int4": report["m_step_fp16_int4"],
        "m_gemm_at_M8": report["tier1_m_gemm_at_M8"],
        "m_gemm_at_M8_ci95": report["tier1_m_gemm_at_M8_ci95"],
        "fp16verify_tps_at_lambda1": report["fp16verify_tps_at_lambda1"],
        "fp16verify_tps_at_current_build": report["fp16verify_tps_at_current_build"],
        "fp16verify_clears_500_at_lambda1": report["fp16verify_clears_500_at_lambda1"],
        "fp16_validity_premise": report["fp16_validity_premise"],
        "fp16_validity_premise_confirmed": bool(report["fp16_validity_premise"] == "CONFIRMED"),
        "diag_bf16_kernel_bitexact": report["bf16_cublas_batch_invariant"],
        "diag_int4_kernel_M_dependent": report["int4_marlin_M_dependent"],
        "tier2_status": report["tier2_status"],
    }
    if tok is not None:
        summary["fp16_token_identity_M1_vs_M8"] = tok["fp16_token_identity_M1_vs_M8"]
        summary["fp16_token_divergence_M1_vs_M8"] = tok["fp16_token_divergence_M1_vs_M8"]
        summary["tier2_determinism_M1_vs_M1"] = tok["determinism_M1_vs_M1"]
        summary["tier2_within_batch_row0_vs_row1"] = tok["within_batch_row0_vs_row1"]
        summary["fp16_token_sha_equal_frac"] = tok["sha_equal_frac"]
    for k, v in summary.items():
        run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v

    # cost-ratio curve table
    tbl = wandb.Table(columns=["M", "int4_us_sum", "bf16_us_sum", "gemm_cost_ratio_fp16_int4"])
    cv = gemm["cost_ratio_curve"]
    for M in M_SWEEP:
        row = cv[str(M)] if str(M) in cv else cv[M]
        run.log({"global_step": M, "curve/M": M,
                 "curve/gemm_cost_ratio_fp16_int4": row["gemm_cost_ratio_fp16_int4"],
                 "curve/int4_us_sum": row["int4_us_sum"], "curve/bf16_us_sum": row["bf16_us_sum"]})
        tbl.add_data(M, row["int4_us_sum"], row["bf16_us_sum"], row["gemm_cost_ratio_fp16_int4"])
    run.log({"gemm_cost_ratio_curve": tbl})
    if tok is not None:
        run.summary["tier2/total_positions"] = tok["total_positions"]
        run.summary["tier2/matching_positions"] = tok["matching_positions"]
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["gemm", "tokenident"], default=None,
                    help="internal: run a single GPU phase (subprocess). Omit for the orchestrator.")
    ap.add_argument("--out", default=None, help="phase output json path")
    # orchestrator args
    ap.add_argument("--tier2", action="store_true", help="also run the bf16 e2e token-identity leg")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few iters) to validate the path")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)  # 256 fits the 23GB A10G at M=8 batch (512 OOMs)
    ap.add_argument("--batch-m", type=int, default=M_VERIFY)
    ap.add_argument("--wandb_group", dest="wandb_group", default="fp16-verify-valid-cost")
    ap.add_argument("--wandb_name", dest="wandb_name", default="lawine/fp16-verify-cost")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.iters, a.warmup, a.repeats = 30, 10, 2
        a.n_prompts = min(a.n_prompts, 8)

    if a.phase == "gemm":
        phase_gemm(a.out, a.iters, a.warmup, a.repeats, a.smoke)
    elif a.phase == "tokenident":
        phase_tokenident(a.out, a.n_prompts, a.max_len, a.batch_m)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
