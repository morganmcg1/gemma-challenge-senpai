#!/usr/bin/env python
"""PR #376 -- Real-weight e2e token-identity: does pinning the attention split
restore M=8-verify == M=1-AR greedy identity on the DEPLOYED int4 stack?

This is the real-weight successor to my #365 synthetic strict_attn_e2e harness.
#365 ran a *generic* transformer with synthetic weights and a flash_attn
``num_splits`` knob; it could not load gemma3n (AltUp / per-layer-embeddings /
partial-rotary RoPE / QK-norm). This leg replaces the synthetic body with the
REAL served forward -- the deployed int4-Marlin w4a16 body GEMMs + the real tied
bf16 lm_head -- via vLLM (the SAME faithful path #221/#232 used), and asks the
card's question on real weights:

    On REAL weights, does pinning the attention split drive the
    M=8-verify-vs-M=1-AR token_identity_rate to 1.0, eliminating #362's
    measured 0.52% deployed flip?  If not, WHICH op still diverges?

Two arms, identical except the pin:

  heuristic  -- stock deployed vLLM. The attention split-K reduction order is
                chosen by the kernel heuristic and differs between the M=8
                verify shape and the M=1 AR shape. This reproduces #232's
                clean deployed divergence (identity 0.9927) and is checked
                against #362's deployed flip (0.0052).

  pinned     -- VLLM_BATCH_INVARIANT=1. On the deployed Gemma4 stack flash_attn
                cannot run (heterogeneous head dims 256/512 -> vLLM forces
                TRITON_ATTN), so flash_attn's ``num_splits`` knob does NOT apply.
                The faithful real-stack "pin the attention split" is the
                batch-invariant override, which forces attention num_splits=1
                (single-segment, M-independent) for BOTH M=1 and M=8 AND makes
                the aten matmul / lm_head / norm reductions batch-invariant
                (init_batch_invariance, confirmed active on this exact stack by
                #19 and #122). ``--pin-splits N`` is accepted for CLI parity with
                the #365 skeleton and recorded in config; the realised pin is the
                num_splits=1 single-segment attention path described above.

What this decides (the load-bearing fleet contradiction):
  * #232 (nxwv6pam): clean deployed e2e divergence 0.0073; the int4-Marlin body
    GEMMs are BIT-EXACT at the literal decode width (8 rows vs 1) -> #232 pins
    the residual on the bf16 lm_head + attention.
  * #122 (n5bypf5h) / #19: VLLM_BATCH_INVARIANT cannot reach the int4 Marlin
    GEMM (a custom CUDA op outside the aten dispatcher), and by elimination the
    Marlin GEMM is the M-variant residual.
  Both can be true at DIFFERENT size_m: Marlin's split-K geometry is chosen as a
  function of size_m, so it can be bit-exact at the small decode width (8 rows)
  yet M-variant at the prefill-replication width (8*seq_len rows). This harness
  reports a Marlin size_m sweep that resolves it, and -- with attention + aten +
  lm_head all pinned in the ``pinned`` arm -- isolates by elimination whether any
  residual flip is the Marlin GEMM.

Geometry (honest): the e2e identity number uses prefill-replication (M=8 = 8
identical co-batched prefill replicas; GEMM M-dim = 8*seq_len), the SAME geometry
as #221/#232 so the numbers are apples-to-apples and tractable through vLLM's
high-level API. The literal decode-verify attention width (8 query rows against a
KV cache) is the #362/#365 attention-split mechanism; it is covered here by the
in-process Marlin size_m diagnostic + the aten-mm pin-engaged control + the
documented num_splits=1 override, not by a separate hand-rolled decode kernel.
This caveat is carried in the report's honest band.

LOCAL correctness card on a single A10G. 0 HF Job / 0 submission / 0 served-file
change / 0 official TPS draw / no train.py --launch. The served int4 path is READ,
never modified. The GPU work for each arm runs as an isolated subprocess so vLLM
gets a clean CUDA context (and so the pinned arm's process-wide batch-invariant
override never leaks into the heuristic arm).
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
# Imported fleet anchors (DO NOT re-derive -- import, do not re-measure)
# --------------------------------------------------------------------------------------
INT4_IDENTITY_232 = 0.9927083333333333   # #232 nxwv6pam clean deployed M1-vs-M8 identity
INT4_DIVERGENCE_232 = 0.0072916666666667  # #232 divergence (1 - identity)
INT4_BODY_BITEXACT_DECODE_232 = True      # #232 int4-Marlin body bit-exact at decode width (8 rows)
DEPLOYED_FLIP_362 = 0.0052                # #362 5k3px8p1 deployed M8-verify-vs-M1-AR flip rate
FP16_IDENTITY_221 = 0.98944091796875      # #221 6m40u2bg bf16 floor identity
BATCH_INVARIANT_MARLIN_LOCUS_122 = True   # #122 n5bypf5h: VLLM_BATCH_INVARIANT can't reach Marlin GEMM

OFFICIAL_BASELINE = 481.53                # #52 official TPS (this leg adds 0)
K_SPEC = 7                               # num_speculative_tokens (manifest)
M_VERIFY = K_SPEC + 1                    # = 8, the deployed verify batch width
REPRO_362_TOL = 0.004                    # |heuristic_flip - 0.0052| <= tol  => reproduces #362
IDENTITY_EPS = 1e-12                     # pinned_identity >= 1 - eps treated as "== 1.0" (GREEN)

# Canonical Hub int4 checkpoint (the real-weight proxy the card names): the body
# QKV/MLP GEMMs are the deployed int4-Marlin w4a16 kernel; lm_head is tied to the
# bf16 embeddings (final vocab projection bf16). SAME checkpoint #221/#232 probed.
DEFAULT_PROXY = "google/gemma-4-E4B-it-qat-w4a16-ct"
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/realweight_e2e_token_identity")
ARMS = ("heuristic", "pinned")


# --------------------------------------------------------------------------------------
# Small helpers (resolve_model_dir / read_text_dims reused from #221/#232)
# --------------------------------------------------------------------------------------
def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
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
        "hidden": h, "n_heads": n_heads, "n_kv": n_kv, "head_dim": hd,
        "intermediate": inter, "num_layers": tc.get("num_hidden_layers"),
        "shapes": {
            "qkv_proj": ((n_heads + 2 * n_kv) * hd, h),
            "o_proj": (h, n_heads * hd),
            "gate_up_proj": (2 * inter, h),
            "down_proj": (h, inter),
        },
    }


# ======================================================================================
# PHASE: one arm (vLLM real-weight forward). The pin (VLLM_BATCH_INVARIANT) is set in
# the subprocess ENV by the orchestrator for the ``pinned`` arm; this phase just reads it.
# ======================================================================================
def phase_arm(out_path: str, arm: str, n_prompts: int, max_len: int, batch_m: int,
              gpu_mem_util: float, max_batched_tokens: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    print(f"[arm:{arm}] model={model_dir} layers={dims['num_layers']} hidden={dims['hidden']} "
          f"M_verify={batch_m} VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    # enable_prefix_caching=False so the M identical replicas each do a REAL forward
    # (not served from a shared KV cache); large max_num_batched_tokens so all M
    # replicas land in ONE prefill step (M-dim = M*seq_len); enforce_eager=True so
    # no CUDA-graph batch padding changes M.
    llm = LLM(
        model=model_dir,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=max(512, max_len + 8),
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max(16, batch_m),
        max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"[arm:{arm}] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    # is the batch-invariant override actually installed in THIS process?
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_is_batch_invariant = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_is_batch_invariant = False

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    def argmax_seq(out) -> list[int]:
        pls = out.prompt_logprobs
        am: list[int] = []
        for i in range(len(pls)):
            entry = pls[i]
            if entry is None:
                continue
            best_tok = max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0]
            am.append(int(best_tok))
        return am

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_det_m1 = n_det_m8 = n_within = 0

    for ri, rec in enumerate(rows):
        ids = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        ids = ids[:max_len]
        if len(ids) < 2:
            continue
        prompt = {"prompt_token_ids": ids}

        # M=1 (pure-AR width): one replica -> prefill GEMM M-dim = seq_len
        am1 = argmax_seq(llm.generate([prompt], sp, use_tqdm=False)[0])
        am1b = argmax_seq(llm.generate([prompt], sp, use_tqdm=False)[0])  # det control
        # M=8 (verify width): batch_m identical replicas co-batched -> GEMM M-dim = batch_m*seq_len
        out_m8 = llm.generate([prompt] * batch_m, sp, use_tqdm=False)
        am8_0 = argmax_seq(out_m8[0])
        am8_1 = argmax_seq(out_m8[1]) if len(out_m8) > 1 else am8_0  # within control
        am8b_0 = argmax_seq(llm.generate([prompt] * batch_m, sp, use_tqdm=False)[0])  # det control

        L = min(len(am1), len(am8_0), len(am1b), len(am8_1), len(am8b_0))
        a1, a1b, a80, a81, a8b = am1[:L], am1b[:L], am8_0[:L], am8_1[:L], am8b_0[:L]
        match = sum(1 for x, y in zip(a1, a80) if x == y)     # M1 vs M8 (the signal)
        det_m1 = sum(1 for x, y in zip(a1, a1b) if x == y)    # control: expect L
        det_m8 = sum(1 for x, y in zip(a80, a8b) if x == y)   # control: expect L
        within = sum(1 for x, y in zip(a80, a81) if x == y)   # control: expect L

        n_match += match
        n_total += L
        n_det_m1 += det_m1
        n_det_m8 += det_m8
        n_within += within
        sha1 = hashlib.sha256(bytes(str(a1), "utf8")).hexdigest()[:16]
        sha8 = hashlib.sha256(bytes(str(a80), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "positions": L, "argmax_match_M1_vs_M8": match,
            "sha_equal": sha1 == sha8,
            "det_match_M1_vs_M1": det_m1, "det_match_M8_vs_M8": det_m8,
            "within_match_copy0_vs_copy1": within,
        })
        if ri < 3 or ri == len(rows) - 1:
            print(f"[arm:{arm}] prompt {ri} id={rec.get('id')} match={match}/{L} "
                  f"sha_eq={sha1==sha8} det_m1={det_m1}/{L} det_m8={det_m8}/{L} "
                  f"within={within}/{L}", flush=True)

    identity = (n_match / n_total) if n_total else float("nan")
    det_m1_frac = (n_det_m1 / n_total) if n_total else float("nan")
    det_m8_frac = (n_det_m8 / n_total) if n_total else float("nan")
    within_frac = (n_within / n_total) if n_total else float("nan")
    sha_equal_frac = (statistics.fmean([1.0 if p["sha_equal"] else 0.0 for p in per_prompt])
                      if per_prompt else float("nan"))

    # ---- pin-engaged positive control: aten torch.mm row-0 bit-exactness at M=1 vs M=8 ----
    # Under VLLM_BATCH_INVARIANT the aten matmul is batch-invariant (#19: M1-vs-M7
    # bit-identical), so this MUST be True in the pinned arm -> proves the override is
    # live in THIS process. Under stock it may be False (heuristic batched reduction).
    aten_ctrl = aten_mm_invariance_control(torch, dims["hidden"], batch_m)

    # ---- Marlin size_m sweep: where does the int4 body GEMM stop being bit-exact? ----
    # Resolves #232 (bit-exact at decode width) vs #122/#19 (M-variant residual): Marlin
    # split-K geometry is a function of size_m. VLLM_BATCH_INVARIANT does NOT patch this
    # custom CUDA op, so the result is the same in both arms (reported in both for a check).
    try:
        marlin_diag = marlin_sizem_diag(llm, dims, torch, batch_m, max_len)
    except Exception as exc:
        marlin_diag = {"status": f"failed: {exc!r}", "per_size": {}, "first_divergent_size_m": None,
                       "bitexact_at_decode_width": INT4_BODY_BITEXACT_DECODE_232,
                       "bitexact_at_e2e_width": None}
        print(f"[arm:{arm}] marlin diag unavailable -> {marlin_diag['status']}", flush=True)

    nan_clean = all(math.isfinite(x) for x in (identity, det_m1_frac, det_m8_frac, within_frac))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "arm", "arm": arm, "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "n_prompts": len(per_prompt), "max_len": max_len, "batch_m": batch_m,
        "total_positions": n_total, "matching_positions": n_match,
        "e2e_token_identity_rate": identity,
        "e2e_divergence_rate": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "determinism_M1_vs_M1": det_m1_frac,
        "determinism_M8_vs_M8": det_m8_frac,
        "within_batch_copy0_vs_copy1": within_frac,
        "per_sequence_strict_pass_fraction": sha_equal_frac,
        "aten_mm_control": aten_ctrl,
        "marlin_sizem_diag": marlin_diag,
        "nan_clean": bool(nan_clean), "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[arm:{arm}] identity={identity:.6f} (divergence={1.0-identity:.6f}) "
          f"strict_pass={sha_equal_frac:.4f} peak={peak_gb:.1f}GB", flush=True)
    print(f"[arm:{arm}] controls: det_m1={det_m1_frac:.6f} det_m8={det_m8_frac:.6f} "
          f"within={within_frac:.6f} | pin_engaged(aten_mm bitexact)={aten_ctrl.get('bitexact_M1_vs_M8')} "
          f"attn_batch_invariant={attn_is_batch_invariant}", flush=True)
    print(f"[arm:{arm}] marlin first_divergent_size_m={marlin_diag.get('first_divergent_size_m')} "
          f"bitexact@decode={marlin_diag.get('bitexact_at_decode_width')} "
          f"bitexact@e2e={marlin_diag.get('bitexact_at_e2e_width')}", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


def aten_mm_invariance_control(torch, hidden: int, batch_m: int) -> dict:
    """torch.mm row-0 bit-exactness at M=1 vs M=batch_m -- proves the batch-invariant
    override (pin) is live in this process. Pure aten op (NOT Marlin)."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    n = hidden
    w = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    x = torch.randn(max(batch_m, 16), n, dtype=torch.bfloat16, device=dev)
    y1 = torch.mm(x[:1].contiguous(), w)
    ym = torch.mm(x[:batch_m].contiguous(), w)
    torch.cuda.synchronize()
    bitexact = bool(torch.equal(ym[:1].float(), y1.float()))
    return {
        "bitexact_M1_vs_M8": bitexact,
        "max_abs_diff_M1_vs_M8": float((ym[:1].float() - y1.float()).abs().max()),
        "batch_m": batch_m,
    }


def marlin_sizem_diag(llm, dims: dict, torch, batch_m: int, max_len: int) -> dict:
    """Row-0 bit-exactness of the int4-Marlin body GEMMs across size_m.

    Reuses #232's model navigation. The split-K geometry of the Marlin kernel is
    chosen internally as a function of size_m, so a row-0 comparison vs size_m=1
    localises exactly where the body GEMM stops being M-invariant -- resolving
    #232 (bit-exact at decode width 8) vs #122/#19 (M-variant residual at the
    prefill-replication width 8*seq_len).
    """
    import torch.nn as nn

    dev = torch.device("cuda:0")
    shapes = dims["shapes"]

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

    def find_layers(root):
        chains = [("model", "layers"), ("model", "language_model", "layers"),
                  ("language_model", "model", "layers"), ("language_model", "layers"),
                  ("model", "model", "layers"), ("layers",)]
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

    layers = find_layers(get_model())
    targets = None
    for layer in layers:
        try:
            cand = {
                "qkv_proj": layer.self_attn.qkv_proj,
                "o_proj": layer.self_attn.o_proj,
                "gate_up_proj": layer.mlp.gate_up_proj,
                "down_proj": layer.mlp.down_proj,
            }
        except AttributeError:
            continue
        if all(hasattr(m, "quant_method") and module_out_in(m) == shapes[name]
               for name, m in cand.items()):
            targets = cand
            break
    if targets is None:
        raise RuntimeError("no layer matched canonical body shapes")

    decode_width = batch_m
    e2e_width = batch_m * max_len
    sizes = sorted({1, batch_m, 64, max_len, e2e_width})
    sizes = [s for s in sizes if s <= 4096]  # A10G headroom guard

    torch.manual_seed(0)
    per_size = {}
    first_divergent = None
    for sm in sizes:
        all_bitexact = True
        max_diff = 0.0
        for name, (out, inp) in shapes.items():
            x = torch.randn(max(sm, 1), inp, dtype=torch.bfloat16, device=dev)
            apply_fn = lambda t, _m=targets[name]: _m.quant_method.apply(_m, t, bias=None)
            y1 = apply_fn(x[:1].contiguous())[0].detach().float()
            ym = apply_fn(x[:sm].contiguous())[0].detach().float()
            torch.cuda.synchronize()
            be = bool(torch.equal(ym, y1))
            md = float((ym - y1).abs().max())
            all_bitexact = all_bitexact and be
            max_diff = max(max_diff, md)
        per_size[str(sm)] = {"bitexact_row0_vs_M1": all_bitexact, "max_abs_diff": max_diff}
        if not all_bitexact and first_divergent is None and sm > 1:
            first_divergent = sm

    return {
        "status": "ran",
        "sizes_tested": sizes,
        "per_size": per_size,
        "first_divergent_size_m": first_divergent,
        "bitexact_at_decode_width": per_size.get(str(decode_width), {}).get("bitexact_row0_vs_M1"),
        "bitexact_at_e2e_width": per_size.get(str(e2e_width), {}).get("bitexact_row0_vs_M1"),
        "decode_width": decode_width,
        "e2e_width": e2e_width,
    }


# ======================================================================================
# Orchestrator: two isolated subprocess arms, compose, self-test, wandb
# ======================================================================================
def run_phase_subprocess(args_list: list[str], extra_env: dict | None = None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")  # Gemma4 -> vLLM overrides to TRITON_ATTN
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} "
          f"(VLLM_BATCH_INVARIANT={env.get('VLLM_BATCH_INVARIANT', '0')})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def _run_arm(a: argparse.Namespace, arm: str) -> dict:
    out_json = str(OUT_DIR / f"arm_{arm}_result.json")
    extra_env = {"VLLM_BATCH_INVARIANT": "1"} if arm == "pinned" else {"VLLM_BATCH_INVARIANT": "0"}
    run_phase_subprocess([
        "--phase", "arm", "--arm", arm, "--out", out_json,
        "--n-prompts", str(a.n_prompts), "--max-len", str(a.max_len),
        "--batch-m", str(a.batch_m), "--gpu-mem-util", str(a.gpu_mem_util),
        "--max-batched-tokens", str(a.max_batched_tokens),
    ], extra_env=extra_env)
    return json.load(open(out_json))


def _locus_from_diag(pinned: dict) -> str:
    """If the pinned arm still flips, name the residual locus from the diagnostics.
    With attention (num_splits=1) + aten matmuls + lm_head all batch-invariant in
    the pinned arm, the only forward op the override cannot reach is the int4 Marlin
    GEMM -- so a residual flip is Marlin by elimination, confirmed M-variant by the
    size_m sweep."""
    md = pinned.get("marlin_sizem_diag", {})
    e2e_be = md.get("bitexact_at_e2e_width")
    first_div = md.get("first_divergent_size_m")
    pin_engaged = pinned.get("aten_mm_control", {}).get("bitexact_M1_vs_M8")
    if e2e_be is False:
        return (f"int4 Marlin weight GEMM (size_m-variant split-K; custom CUDA op outside the "
                f"VLLM_BATCH_INVARIANT aten override; first divergent size_m={first_div}, "
                f"M-variant at the e2e width {md.get('e2e_width')}). Confirms #122/#19; corrects "
                f"#232's lm_head attribution at the prefill-replication geometry. "
                f"pin_engaged(aten)={pin_engaged}")
    if e2e_be is True:
        return ("residual NOT the Marlin body (bit-exact across the swept size_m incl. the e2e "
                "width) -> bf16 lm_head / attention accumulation not fully covered by the pin "
                f"(pin_engaged(aten)={pin_engaged}); investigate the tied bf16 lm_head reduction")
    return f"undetermined (marlin diag status={md.get('status')}, pin_engaged(aten)={pin_engaged})"


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arms = {}
    for arm in ARMS:
        arms[arm] = _run_arm(a, arm)

    heuristic = arms["heuristic"]
    pinned = arms["pinned"]

    heuristic_identity = heuristic["e2e_token_identity_rate"]
    pinned_identity = pinned["e2e_token_identity_rate"]
    heuristic_flip = heuristic["e2e_divergence_rate"]
    pinned_flip = pinned["e2e_divergence_rate"]

    # ---- required deliverable fields ----
    pinned_is_one = bool(math.isfinite(pinned_identity) and pinned_identity >= 1.0 - IDENTITY_EPS)
    heuristic_below_one = bool(math.isfinite(heuristic_identity) and heuristic_identity < 1.0 - IDENTITY_EPS)
    pinned_restores_identity_vs_heuristic = bool(pinned_is_one and heuristic_below_one)  # GREEN bar
    heuristic_flip_reproduces_362 = bool(
        math.isfinite(heuristic_flip) and abs(heuristic_flip - DEPLOYED_FLIP_362) <= REPRO_362_TOL)
    residual_divergence_locus = "" if pinned_is_one else _locus_from_diag(pinned)

    verdict = "GREEN" if pinned_restores_identity_vs_heuristic else (
        "RED" if not pinned_is_one else "AMBER")
    # AMBER = pinned reached 1.0 but heuristic ALSO 1.0 (no divergence to eliminate -> harness
    #         couldn't exercise the mechanism at this geometry; report, don't over-claim GREEN).

    # ---- self-test (PRIMARY): HARNESS SANITY / CALIBRATION, not the science verdict ----
    def ctrls_ok(d: dict) -> bool:
        return (d["determinism_M1_vs_M1"] == 1.0 and d["determinism_M8_vs_M8"] == 1.0
                and d["within_batch_copy0_vs_copy1"] == 1.0)
    def arith_ok(d: dict) -> bool:
        ident = d["e2e_token_identity_rate"]
        return (math.isfinite(ident) and 0.0 <= ident <= 1.0
                and abs(d["e2e_divergence_rate"] - (1.0 - ident)) < 1e-9 and bool(d["nan_clean"]))

    heuristic_controls_ok = ctrls_ok(heuristic)
    pinned_controls_ok = ctrls_ok(pinned)
    pin_engaged = bool(pinned["aten_mm_control"].get("bitexact_M1_vs_M8"))  # override is LIVE
    pin_attn_flag = bool(pinned.get("attn_is_batch_invariant"))
    heuristic_arith_ok = arith_ok(heuristic)
    pinned_arith_ok = arith_ok(pinned)

    self_test = {
        "heuristic_controls_eq_1": heuristic_controls_ok,
        "pinned_controls_eq_1": pinned_controls_ok,
        "pin_engaged_aten_mm_bitexact": pin_engaged,
        "pin_attn_is_batch_invariant_flag": pin_attn_flag,
        "heuristic_arith_consistent": heuristic_arith_ok,
        "pinned_arith_consistent": pinned_arith_ok,
    }
    realweight_e2e_identity_self_test_passes = bool(
        heuristic_controls_ok and pinned_controls_ok and pin_engaged and pin_attn_flag
        and heuristic_arith_ok and pinned_arith_ok)

    report = {
        "pr": 376,
        "leg": "real-weight e2e token-identity: pin-the-attention-split vs deployed heuristic (local)",
        "imported_anchors": {
            "int4_identity_232": INT4_IDENTITY_232,
            "int4_divergence_232": INT4_DIVERGENCE_232,
            "int4_body_bitexact_decode_232": INT4_BODY_BITEXACT_DECODE_232,
            "deployed_flip_362": DEPLOYED_FLIP_362,
            "fp16_identity_221": FP16_IDENTITY_221,
            "batch_invariant_marlin_locus_122": BATCH_INVARIANT_MARLIN_LOCUS_122,
            "official_baseline": OFFICIAL_BASELINE, "M_verify": M_VERIFY,
        },
        # ---- REQUIRED deliverable fields ----
        "real_weight_e2e_token_identity_rate": pinned_identity,        # pinned arm (TEST metric)
        "heuristic_e2e_token_identity_rate": heuristic_identity,
        "pinned_restores_identity_vs_heuristic": pinned_restores_identity_vs_heuristic,
        "residual_flip_rate_pinned": pinned_flip,
        "heuristic_flip_reproduces_362": heuristic_flip_reproduces_362,
        "residual_divergence_locus": residual_divergence_locus,
        "realweight_e2e_identity_self_test_passes": realweight_e2e_identity_self_test_passes,  # PRIMARY
        # ---- verdict + supporting ----
        "verdict": verdict,
        "heuristic_flip_rate": heuristic_flip,
        "pin_engaged_aten_mm_bitexact": pin_engaged,
        "pin_attn_is_batch_invariant": pin_attn_flag,
        # cross to fleet anchors
        "pinned_identity_vs_232_delta": pinned_identity - INT4_IDENTITY_232,
        "heuristic_identity_vs_232_delta": heuristic_identity - INT4_IDENTITY_232,
        "heuristic_flip_vs_362_delta": heuristic_flip - DEPLOYED_FLIP_362,
        # per-arm detail
        "arms": {
            arm: {
                "e2e_token_identity_rate": d["e2e_token_identity_rate"],
                "e2e_divergence_rate": d["e2e_divergence_rate"],
                "determinism_M1_vs_M1": d["determinism_M1_vs_M1"],
                "determinism_M8_vs_M8": d["determinism_M8_vs_M8"],
                "within_batch_copy0_vs_copy1": d["within_batch_copy0_vs_copy1"],
                "per_sequence_strict_pass_fraction": d["per_sequence_strict_pass_fraction"],
                "vllm_batch_invariant_env": d["vllm_batch_invariant_env"],
                "attn_is_batch_invariant": d["attn_is_batch_invariant"],
                "aten_mm_control": d["aten_mm_control"],
                "marlin_sizem_diag": d["marlin_sizem_diag"],
                "peak_gpu_gb": d["peak_gpu_gb"],
            } for arm, d in arms.items()
        },
        "self_test": self_test,
        "n_prompts": heuristic["n_prompts"], "max_len": heuristic["max_len"],
        "batch_m": heuristic["batch_m"], "total_positions": heuristic["total_positions"],
        "model_dir": heuristic["model_dir"], "pin_splits_requested": a.pin_splits,
    }
    report_path = OUT_DIR / "realweight_e2e_report.json"
    json.dump(report, open(report_path, "w"), indent=2)

    # ---- console summary ----
    md_h = heuristic["marlin_sizem_diag"]
    print("\n========== REAL-WEIGHT E2E TOKEN-IDENTITY (PR #376) ==========", flush=True)
    print(f" VERDICT                              : {verdict}", flush=True)
    print(f" real_weight_e2e_token_identity_rate (PINNED)     : {pinned_identity:.6f}", flush=True)
    print(f" heuristic_e2e_token_identity_rate                : {heuristic_identity:.6f}", flush=True)
    print(f" residual_flip_rate_pinned                        : {pinned_flip:.6f}", flush=True)
    print(f" heuristic_flip_rate                              : {heuristic_flip:.6f}", flush=True)
    print(f" pinned_restores_identity_vs_heuristic            : {pinned_restores_identity_vs_heuristic}", flush=True)
    print(f" heuristic_flip_reproduces_362 (0.0052)           : {heuristic_flip_reproduces_362}", flush=True)
    if residual_divergence_locus:
        print(f" residual_divergence_locus                        : {residual_divergence_locus}", flush=True)
    print(f"   pin engaged (aten_mm bitexact)   : {pin_engaged}   attn_is_batch_invariant: {pin_attn_flag}", flush=True)
    print(f"   heuristic controls det_m1/det_m8/within: {heuristic['determinism_M1_vs_M1']:.4f}/"
          f"{heuristic['determinism_M8_vs_M8']:.4f}/{heuristic['within_batch_copy0_vs_copy1']:.4f}", flush=True)
    print(f"   pinned    controls det_m1/det_m8/within: {pinned['determinism_M1_vs_M1']:.4f}/"
          f"{pinned['determinism_M8_vs_M8']:.4f}/{pinned['within_batch_copy0_vs_copy1']:.4f}", flush=True)
    print(f"   marlin first_divergent_size_m={md_h.get('first_divergent_size_m')} "
          f"bitexact@decode({md_h.get('decode_width')})={md_h.get('bitexact_at_decode_width')} "
          f"bitexact@e2e({md_h.get('e2e_width')})={md_h.get('bitexact_at_e2e_width')}", flush=True)
    print(f" --- fleet cross ---", flush=True)
    print(f" pinned vs #232 (0.9927) delta        : {report['pinned_identity_vs_232_delta']:+.6f}", flush=True)
    print(f" heuristic vs #232 (0.9927) delta     : {report['heuristic_identity_vs_232_delta']:+.6f}", flush=True)
    print(f" heuristic flip vs #362 (0.0052) delta: {report['heuristic_flip_vs_362_delta']:+.6f}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)           : {realweight_e2e_identity_self_test_passes}  {self_test}", flush=True)
    print(f" report -> {report_path}", flush=True)
    print("==============================================================\n", flush=True)

    if not a.no_wandb:
        log_wandb(report, a)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling",
        agent="stark",
        name=a.wandb_name,
        group=a.wandb_group,
        notes="PR#376 real-weight e2e token-identity: pin-the-attention-split (VLLM_BATCH_INVARIANT) "
              "vs deployed heuristic; does pinned M8-vs-M1 identity reach 1.0?",
        config={
            "pr": 376, "M_verify": report["batch_m"], "n_prompts": report["n_prompts"],
            "max_len": report["max_len"], "model_dir": report["model_dir"],
            "pin_splits_requested": report["pin_splits_requested"],
            "int4_identity_232": INT4_IDENTITY_232, "deployed_flip_362": DEPLOYED_FLIP_362,
            "fp16_identity_221": FP16_IDENTITY_221, "official_baseline": OFFICIAL_BASELINE,
            "repro_362_tol": REPRO_362_TOL,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return

    summary = {
        "realweight_e2e_identity_self_test_passes": report["realweight_e2e_identity_self_test_passes"],
        "real_weight_e2e_token_identity_rate": report["real_weight_e2e_token_identity_rate"],
        "heuristic_e2e_token_identity_rate": report["heuristic_e2e_token_identity_rate"],
        "pinned_restores_identity_vs_heuristic": report["pinned_restores_identity_vs_heuristic"],
        "residual_flip_rate_pinned": report["residual_flip_rate_pinned"],
        "heuristic_flip_reproduces_362": report["heuristic_flip_reproduces_362"],
        "heuristic_flip_rate": report["heuristic_flip_rate"],
        "verdict_green": report["verdict"] == "GREEN",
        "verdict_red": report["verdict"] == "RED",
        "pin_engaged_aten_mm_bitexact": report["pin_engaged_aten_mm_bitexact"],
        "pin_attn_is_batch_invariant": report["pin_attn_is_batch_invariant"],
        "pinned_identity_vs_232_delta": report["pinned_identity_vs_232_delta"],
        "heuristic_identity_vs_232_delta": report["heuristic_identity_vs_232_delta"],
        "heuristic_flip_vs_362_delta": report["heuristic_flip_vs_362_delta"],
        "int4_identity_232": INT4_IDENTITY_232,
        "deployed_flip_362": DEPLOYED_FLIP_362,
    }
    for arm in ARMS:
        d = report["arms"][arm]
        md = d["marlin_sizem_diag"]
        summary[f"{arm}/identity"] = d["e2e_token_identity_rate"]
        summary[f"{arm}/divergence"] = d["e2e_divergence_rate"]
        summary[f"{arm}/det_m1"] = d["determinism_M1_vs_M1"]
        summary[f"{arm}/det_m8"] = d["determinism_M8_vs_M8"]
        summary[f"{arm}/within"] = d["within_batch_copy0_vs_copy1"]
        summary[f"{arm}/strict_pass"] = d["per_sequence_strict_pass_fraction"]
        summary[f"{arm}/aten_mm_bitexact"] = bool(d["aten_mm_control"].get("bitexact_M1_vs_M8"))
        summary[f"{arm}/marlin_first_divergent_size_m"] = md.get("first_divergent_size_m") or 0
        summary[f"{arm}/marlin_bitexact_at_e2e_width"] = bool(md.get("bitexact_at_e2e_width"))
    for k, v in summary.items():
        run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    if report["residual_divergence_locus"]:
        run.summary["residual_divergence_locus"] = report["residual_divergence_locus"]
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["arm"], default=None,
                    help="internal: run one GPU arm (subprocess). Omit for the orchestrator.")
    ap.add_argument("--arm", choices=list(ARMS), default=None, help="internal: which arm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts) to validate the path")
    # card reproduce-skeleton flags (accepted for parity; mapped/recorded)
    ap.add_argument("--gpu", action="store_true", help="(compat) GPU is always used")
    ap.add_argument("--real-lmhead", dest="real_lmhead", action="store_true",
                    help="(compat) vLLM path always uses the real tied bf16 lm_head")
    ap.add_argument("--real-int4-body", dest="real_int4_body", action="store_true",
                    help="(compat) vLLM path always uses the real int4-Marlin body")
    ap.add_argument("--proxy", default=DEFAULT_PROXY, help="(compat) real-weight proxy id; resolved from cache")
    ap.add_argument("--eval-prompts", dest="n_prompts", type=int, default=128,
                    help="number of official eval prompts (alias of the card's --eval-prompts)")
    ap.add_argument("--pin-splits", dest="pin_splits", type=int, default=8,
                    help="recorded for parity with the #365 skeleton; the realised real-stack pin "
                         "is VLLM_BATCH_INVARIANT (attention num_splits=1, single-segment)")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-m", type=int, default=M_VERIFY)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--wandb_group", dest="wandb_group", default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/realweight-e2e-identity")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 6)

    if a.phase == "arm":
        phase_arm(a.out, a.arm, a.n_prompts, a.max_len, a.batch_m,
                  a.gpu_mem_util, a.max_batched_tokens)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
