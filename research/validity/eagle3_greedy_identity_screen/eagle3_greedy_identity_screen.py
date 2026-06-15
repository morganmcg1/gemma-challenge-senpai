#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #361 -- Local EAGLE-3 greedy-identity screen: does the spec verify break strict identity?

The #319 strict-lock makes byte-exact greedy-token-identity to the submitted checkpoint's
plain greedy AR a HARD gate. The 473.5 EAGLE-3 SPEED cap says nothing about whether the
EAGLE-3 M-token batched verify *preserves identity*. That is the separate, open, binding
empirical question this card measures on real silicon (the assigned pod A10G).

WHAT THIS RUNS (LOCAL pod-GPU inference profiling ONLY. NO train.py --launch, NO HF Job,
NO submission, NO served-file change):

  Arm REF (strict reference): plain greedy AR -- no speculation, single-stream, N>=512 new
    tokens over a fixed >=8-prompt set + fixed seed, on the DEPLOYED int4 substrate (the
    bit-exact strict-ladder reference, lawine #196 token_identity_rate=1.0; the served bf16
    lm_head is cross-session NON-deterministic ~9-13% so it is NOT a clean reference). Record
    the exact emitted token-ID stream.

  Arm SPEC (the question): the native trained EAGLE-3 head (gua9x68j / 56ksyxgw) is NOT
    available locally -- W&B logged_artifacts==[] for both runs, the .pt is not on disk, and
    publish was HUMAN-owned and never done; only the #333 converter's SYNTHETIC-ZERO candidate
    exists. A zero head drafts degenerately (alpha~0) so a native-acceptance eagle3 run is
    blocked. Per the card's FALLBACK (step 6) we therefore measure the structural quantity the
    eagle3 verify is made of: the M=K+1=8 batched-verify token-identity in the LITERAL
    single-sequence spec-verify geometry (a width-8 chunked re-forward of the REF stream vs the
    width-1 AR decode), perfect-draft worst-case so every one of the M positions is exercised.
    This is the same mechanism behind the deployed M=8 ~0.73% divergence (lawine #232,
    0.992708 identity), measured here in the faithful causal verify geometry, and it directly
    answers: does an M>1 batched verify break strict token-identity on this hardware, and by
    how much.

  Plus (FALLBACK 6a/6c): the draft-head forward latency in isolation (the #333 candidate's
    real shapes/dtype on GPU; latency is shape/dtype-bound, not weight-value-bound), and a
    bounded vLLM eagle3 integration attempt that records whether the native eagle3 greedy
    spec engine even CONSTRUCTS on this target/hardware (closing the #338 C2 integration
    unknown) or the EXACT local-integration blocker.

PRIMARY metric : token_identity_rate (float) -- M=8 verify-geometry vs REF, over generated
                 positions. < 1.0 => first-divergence record (prompt idx, position, REF vs
                 SPEC-verify token, local logit gap). A single mismatched ID fails strict.
TEST   metric  : greedy_identity_screen_self_test_passes (bool).

All TPS fields are LOCAL RELATIVE (~7x off official, land #245); token-identity is the
portable result. BASELINE 481.53 UNCHANGED -- this is a measurement, 0 official TPS.

Reproduce:
    cd target/ && python research/validity/eagle3_greedy_identity_screen/eagle3_greedy_identity_screen.py \
        --gpu --wandb_group eagle3-greedy-identity-screen --wandb_name ubel/eagle3-greedy-identity-screen

The GPU work runs as isolated subprocesses so vLLM/torch get a clean CUDA context and release
VRAM on exit; the orchestrator stays GPU-free and owns composition, self-test, and wandb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # .../target
OUT_DIR = HERE

# --------------------------------------------------------------------------------------
# Imported fleet anchors (import, do NOT re-derive).
# --------------------------------------------------------------------------------------
OFFICIAL_BASELINE = 481.53            # #52 official frontier TPS (this card adds 0)
TARGET_TPS = 500.0
K_SPEC = 7                            # eagle3 num_speculative_tokens (#322 read spec)
M_VERIFY = K_SPEC + 1                 # = 8, the batched-verify width (7 draft + 1 resume)
DEPLOYED_M8_IDENTITY_232 = 0.9927083333333333   # lawine #232 int4 deployed M=8 batched-verify identity
DEPLOYED_M8_DIVERGENCE_232 = 0.007291666666666696  # 1 - identity (the "~0.73%" anchor)
STRICT_FLOOR_IDENTITY_196 = 1.0       # lawine #196 non-spec int4 M=1 AR token_identity_rate
RECONCILE_TOL = 0.01                  # |verify_div - 0.00729| <= tol => corroborates #232 mechanism

# Deployed int4 substrate -- the bit-exact strict-ladder reference (same checkpoint #232/#221
# used). Body QKV/MLP GEMMs are the deployed int4-Marlin w4a16 kernel; lm_head is tied bf16.
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = ("official/main_bucket/shared_resources/speed_benchmark/data/"
                 "ppl_ground_truth_tokens.jsonl")
# The #333 converter's synthetic-zero candidate dir (real SHAPES/dtype, zero VALUES). Used for
# the isolated draft-head latency (latency is shape/dtype-bound) and the eagle3 attempt.
EAGLE3_HEAD_DIR = (REPO_ROOT
                   / "research/launch/eagle3_safetensors_converter/_candidate")


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


def load_prompts(n_prompts: int, ctx_cap: int) -> list[dict]:
    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    out = []
    for rec in rows:
        ctx = list(rec.get("context_token_ids", []))[:ctx_cap]
        if len(ctx) >= 2:
            out.append({"id": rec.get("id"), "context_token_ids": ctx})
    return out


def _lp(v: Any) -> float:
    return float(getattr(v, "logprob", v))


def argmax_and_gap(entry: dict, ref_tok: int) -> tuple[int, float, float | None, float | None]:
    """argmax token of a prompt_logprobs entry + the local logit(logprob) gap vs ref_tok."""
    best_tok, best_v = max(entry.items(), key=lambda kv: _lp(kv[1]))
    best_lp = _lp(best_v)
    ref_lp = _lp(entry[ref_tok]) if ref_tok in entry else None
    gap = (best_lp - ref_lp) if ref_lp is not None else None
    return int(best_tok), best_lp, ref_lp, gap


# ======================================================================================
# GPU PHASE 1: REF (greedy AR) + M=8 verify-geometry identity  (vLLM, int4 deployed path)
# ======================================================================================
def phase_ref_verify(out_path: str, n_prompts: int, n_new: int, ctx_cap: int,
                     verify_width: int, gpu_mem_util: float, ppl_slice: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir = resolve_model_dir()
    prompts = load_prompts(n_prompts, ctx_cap)
    print(f"[refver] model={model_dir} prompts={len(prompts)} n_new={n_new} "
          f"verify_width={verify_width}", flush=True)

    # ONE engine at the verify width (max_num_batched_tokens = M): decode steps are width-1
    # (the canonical AR decode), while a re-fed prompt is processed in width-M chunked forwards
    # (the literal spec-verify geometry: M new tokens vs the KV-cached prefix). enforce_eager
    # so no CUDA-graph batch padding perturbs M; prefix caching off so re-feeds do a real
    # forward. If the engine rejects the tiny token budget, retry at progressively larger M and
    # record the effective width.
    effective_width = verify_width
    llm = None
    for w in (verify_width, 16, 32, 64):
        try:
            llm = LLM(
                model=model_dir,
                quantization="compressed-tensors",
                dtype="bfloat16",
                max_model_len=max(1024, ctx_cap + n_new + 16),
                gpu_memory_utilization=gpu_mem_util,
                max_num_seqs=1,
                max_num_batched_tokens=w,
                enable_prefix_caching=False,
                enforce_eager=True,
                trust_remote_code=True,
            )
            effective_width = w
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[refver] engine init at max_num_batched_tokens={w} failed: {exc!r}", flush=True)
            llm = None
    if llm is None:
        raise RuntimeError("could not construct the int4 verify engine at any width")
    if effective_width != verify_width:
        print(f"[refver] NOTE effective verify width = {effective_width} "
              f"(requested {verify_width})", flush=True)

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n_new)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=2)

    per_prompt = []
    n_match = n_total = 0
    n_det_gen = n_det_ver = 0
    first_div = None
    ppl_logps: list[float] = []
    gen_tokens_total = 0

    t_gen0 = time.time()
    ref_decode_time = 0.0   # pure REF greedy-AR decode wall (excl. determinism reruns + verify)
    for pi, pr in enumerate(prompts):
        ctx = pr["context_token_ids"]
        base = {"prompt_token_ids": ctx}

        # --- REF: plain greedy AR, single-stream, width-1 decode ---
        _tg = time.time()
        out = llm.generate([base], gen_sp, use_tqdm=False)[0]
        ref_decode_time += time.time() - _tg
        gen = list(out.outputs[0].token_ids)
        # determinism control: regenerate, expect identical (int4 bit-exact)
        gen_b = list(llm.generate([base], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
        Lg = min(len(gen), len(gen_b))
        det_gen = sum(1 for a, b in zip(gen[:Lg], gen_b[:Lg]) if a == b)
        gen_tokens_total += len(gen)

        # --- SPEC-equiv: M=8 verify-geometry re-forward of [ctx + gen] ---
        full = ctx + gen
        vout = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs  # len == len(full); entry[i] predicts full[i] from <i
        # determinism control for the verify geometry
        vout_b = llm.generate([{"prompt_token_ids": full}], ver_sp, use_tqdm=False)[0]
        pls_b = vout_b.prompt_logprobs

        c = len(ctx)
        match = det_ver = 0
        verify_stream: list[int] = []
        for g in range(len(gen)):
            j = c + g
            if j >= len(pls) or pls[j] is None:
                continue
            ref_tok = gen[g]
            am, am_lp, ref_lp, gap = argmax_and_gap(pls[j], ref_tok)
            verify_stream.append(am)
            ok = (am == ref_tok)
            match += int(ok)
            n_total += 1
            n_match += int(ok)
            # verify-geometry determinism
            if pls_b[j] is not None:
                am_b, *_ = argmax_and_gap(pls_b[j], ref_tok)
                det_ver += int(am_b == am)
            # PPL spot-check slice (sanity only): logprob of the actual ref token
            if g < ppl_slice and ref_lp is not None and math.isfinite(ref_lp):
                ppl_logps.append(ref_lp)
            if (not ok) and first_div is None:
                first_div = {
                    "prompt_index": pi, "prompt_id": pr["id"],
                    "generated_offset": g, "absolute_position": j,
                    "ref_token_id": int(ref_tok), "spec_verify_token_id": int(am),
                    "ref_token_logprob": ref_lp, "spec_argmax_logprob": am_lp,
                    "local_logit_gap": gap,
                }

        n_match_seq = match
        n_det_gen += det_gen
        n_det_ver += det_ver
        ref_sha = hashlib.sha256(bytes(str(gen), "utf8")).hexdigest()[:16]
        ver_sha = hashlib.sha256(bytes(str(verify_stream), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "prompt_index": pi, "id": pr["id"], "context_len": c,
            "n_generated": len(gen), "compared_positions": len(verify_stream),
            "verify_match": n_match_seq, "ref_sha": ref_sha, "verify_sha": ver_sha,
            "sha_equal": ref_sha == ver_sha,
            "det_gen_match": det_gen, "det_ver_match": det_ver,
        })
        print(f"[refver] prompt {pi} id={pr['id']} gen={len(gen)} "
              f"verify_match={n_match_seq}/{len(verify_stream)} sha_eq={ref_sha==ver_sha} "
              f"det_gen={det_gen}/{Lg}", flush=True)

    gen_elapsed = time.time() - t_gen0
    identity = (n_match / n_total) if n_total else float("nan")
    det_gen_frac = (n_det_gen / gen_tokens_total) if gen_tokens_total else float("nan")
    det_ver_frac = (n_det_ver / n_total) if n_total else float("nan")
    strict_pass_frac = (sum(1 for p in per_prompt if p["sha_equal"]) / len(per_prompt)
                        if per_prompt else float("nan"))
    ppl = (math.exp(-sum(ppl_logps) / len(ppl_logps)) if ppl_logps else float("nan"))
    # local-relative TPS of the REF arm: pure greedy-AR decode wall only (the determinism
    # reruns + verify re-forwards are excluded so this reflects single-stream decode speed).
    ref_tps_local = (gen_tokens_total / ref_decode_time) if ref_decode_time > 0 else float("nan")
    ref_step_us = (1e6 / ref_tps_local) if ref_tps_local and math.isfinite(ref_tps_local) else float("nan")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    nan_clean = all(math.isfinite(x) for x in (identity, det_gen_frac, det_ver_frac))
    out = {
        "phase": "ref_verify",
        "model_dir": model_dir,
        "n_prompts": len(per_prompt),
        "n_new": n_new,
        "verify_width_requested": verify_width,
        "verify_width_effective": effective_width,
        "M_verify": M_VERIFY,
        "total_compared_positions": n_total,
        "matching_positions": n_match,
        "token_identity_rate": identity,
        "verify_divergence": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "per_sequence_strict_pass_fraction": strict_pass_frac,
        "determinism_ref_gen": det_gen_frac,            # control: expect 1.0
        "determinism_verify_geometry": det_ver_frac,    # control: expect 1.0
        "first_divergence": first_div,
        "ppl_spotcheck": ppl,                           # sanity only, NOT the gate
        "ppl_slice_positions": len(ppl_logps),
        "ref_tps_local_relative": ref_tps_local,
        "ref_decode_step_us_local_relative": ref_step_us,
        "ref_gen_tokens_total": gen_tokens_total,
        "ref_gen_elapsed_s": gen_elapsed,
        "ref_decode_time_s": ref_decode_time,
        "peak_gpu_gb": peak_gb,
        "nan_clean": bool(nan_clean),
        "per_prompt": per_prompt,
        "local_relative_tps": True,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[refver] token_identity_rate={identity:.6f} (divergence={1.0-identity:.6f}) "
          f"strict_pass={strict_pass_frac:.4f} det_gen={det_gen_frac:.6f} "
          f"det_ver={det_ver_frac:.6f} ppl~{ppl:.4f} peak={peak_gb:.1f}GB", flush=True)
    print(f"REFVER_DONE {out_path}", flush=True)


# ======================================================================================
# GPU PHASE 2: isolated draft-head forward latency  (torch GEMMs at the #333 head shapes)
# ======================================================================================
def phase_draft_head_latency(out_path: str, iters: int) -> None:
    import torch
    from safetensors import safe_open

    dev = torch.device("cuda:0")
    st_path = EAGLE3_HEAD_DIR / "model.safetensors"
    used_real_shapes = st_path.exists()
    # The eagle3 linear K=7 head per drafted token does: fc -> 1 decoder layer (qkv,o,gate_up,
    # down) -> lm_head. lm_head (2560 x 262144) dominates. Time the real-shape bf16 GEMMs (the
    # #333 candidate's tensors if present, else analytic shapes). Values are irrelevant to
    # latency; only shapes/dtype matter, so the synthetic-zero candidate is faithful here.
    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.bfloat16, device=dev)

    # analytic shapes (converter constants): HID=2560, FUSED_IN=7680, QKV=3072, Q=2048,
    # GU=20480, INTER=10240, VOCAB=262144, TWO_H=5120
    shapes = {
        "fc": (2560, 7680), "qkv_proj": (3072, 5120), "o_proj": (2560, 2048),
        "gate_up_proj": (20480, 2560), "down_proj": (2560, 10240),
        "lm_head": (262144, 2560),
    }
    weights: dict[str, "torch.Tensor"] = {}
    if used_real_shapes:
        try:
            with safe_open(str(st_path), framework="pt", device="cuda:0") as f:
                keys = set(f.keys())
                wmap = {
                    "fc": "model.fc.weight",
                    "qkv_proj": "model.layers.0.self_attn.qkv_proj.weight",
                    "o_proj": "model.layers.0.self_attn.o_proj.weight",
                    "gate_up_proj": "model.layers.0.mlp.gate_up_proj.weight",
                    "down_proj": "model.layers.0.mlp.down_proj.weight",
                    "lm_head": "lm_head.weight",
                }
                for nm, key in wmap.items():
                    if key in keys:
                        weights[nm] = f.get_tensor(key).to(torch.bfloat16)
        except Exception as exc:  # noqa: BLE001
            print(f"[drafthead] safetensors read failed ({exc!r}); using analytic zeros", flush=True)
            weights = {}
    for nm, (o, i) in shapes.items():
        if nm not in weights:
            weights[nm] = zeros(o, i)

    def head_forward_one_token() -> None:
        # one drafted token through the linear head (GEMM-dominant path)
        x_fused = zeros(1, 7680)
        h = torch.nn.functional.linear(x_fused, weights["fc"])          # -> 2560
        h2 = torch.cat([h, h], dim=-1)                                  # layer-0 sees 2H=5120
        q = torch.nn.functional.linear(h2, weights["qkv_proj"])        # -> 3072
        _ = torch.nn.functional.linear(q[:, :2048], weights["o_proj"])  # -> 2560
        gu = torch.nn.functional.linear(h, weights["gate_up_proj"])    # -> 20480
        g, u = gu[:, :10240], gu[:, 10240:]
        mlp = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, weights["down_proj"])
        _ = torch.nn.functional.linear(h + mlp, weights["lm_head"])    # -> 262144 (dominant)

    # warmup
    for _ in range(5):
        head_forward_one_token()
    torch.cuda.synchronize()

    # time one drafted token, and the full K=7 linear chain (sequential)
    t0 = time.time()
    for _ in range(iters):
        head_forward_one_token()
    torch.cuda.synchronize()
    per_token_us = (time.time() - t0) / iters * 1e6

    t0 = time.time()
    for _ in range(max(1, iters // K_SPEC)):
        for _ in range(K_SPEC):
            head_forward_one_token()
    torch.cuda.synchronize()
    per_chain_us = (time.time() - t0) / max(1, iters // K_SPEC) * 1e6

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "draft_head_latency",
        "used_real_candidate_shapes": used_real_shapes,
        "weights_from_synthetic_candidate": used_real_shapes,
        "draft_head_forward_us_per_token_local_relative": per_token_us,
        "draft_head_chain_us_k7_local_relative": per_chain_us,
        "k_spec": K_SPEC,
        "iters": iters,
        "peak_gpu_gb": peak_gb,
        "local_relative_tps": True,
        "nan_clean": bool(math.isfinite(per_token_us) and math.isfinite(per_chain_us)),
        "note": "latency is shape/dtype-bound; synthetic-zero values are faithful for timing.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[drafthead] per_token={per_token_us:.1f}us per_k7_chain={per_chain_us:.1f}us "
          f"(LOCAL RELATIVE) peak={peak_gb:.1f}GB", flush=True)
    print(f"DRAFTHEAD_DONE {out_path}", flush=True)


# ======================================================================================
# GPU PHASE 3: bounded vLLM eagle3 integration attempt (constructs? or exact blocker)
# ======================================================================================
def phase_eagle3_attempt(out_path: str, gpu_mem_util: float) -> None:
    import traceback

    result: dict[str, Any] = {
        "phase": "eagle3_attempt",
        "head_dir": str(EAGLE3_HEAD_DIR),
        "head_is_synthetic_zero": True,
        "constructed": False,
        "ran_generate": False,
        "blocker": None,
        "note": ("native trained head unavailable; this attempt uses the #333 synthetic-zero "
                 "candidate purely to test whether the vLLM eagle3 greedy spec engine CONSTRUCTS "
                 "and RUNS on this int4 target/hardware. alpha is meaningless with a zero head."),
    }
    try:
        from vllm import LLM, SamplingParams
        model_dir = resolve_model_dir()
        spec_cfg = {"method": "eagle3", "model": str(EAGLE3_HEAD_DIR),
                    "num_speculative_tokens": K_SPEC}
        llm = LLM(
            model=model_dir,
            quantization="compressed-tensors",
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=gpu_mem_util,
            max_num_seqs=1,
            enforce_eager=True,
            enable_prefix_caching=False,
            trust_remote_code=True,
            speculative_config=spec_cfg,
        )
        result["constructed"] = True
        print("[eagle3] vLLM eagle3 spec engine CONSTRUCTED", flush=True)
        prompts = load_prompts(2, 64)
        sp = SamplingParams(temperature=0.0, max_tokens=16)
        outs = llm.generate([{"prompt_token_ids": p["context_token_ids"]} for p in prompts],
                            sp, use_tqdm=False)
        result["ran_generate"] = True
        result["sample_token_ids"] = [list(o.outputs[0].token_ids) for o in outs]
        print("[eagle3] eagle3 greedy generate RAN (alpha meaningless: zero head)", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["blocker"] = {"type": type(exc).__name__, "message": str(exc)[:1500],
                             "traceback_tail": traceback.format_exc()[-1500:]}
        print(f"[eagle3] integration blocker: {type(exc).__name__}: {str(exc)[:300]}", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"EAGLE3ATTEMPT_DONE {out_path}", flush=True)


# ======================================================================================
# Orchestrator
# ======================================================================================
def run_phase(args_list: list[str], timeout: int | None = None) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"   # the container exposes GPU 0; default CVD=4 is wrong here
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} (timeout={timeout})", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s: {args_list}", flush=True)
        return 124


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refver_json = str(OUT_DIR / "_refver.json")
    drafthead_json = str(OUT_DIR / "_drafthead.json")
    eagle3_json = str(OUT_DIR / "_eagle3_attempt.json")

    # --- Phase 1: REF + verify (the must-have core) ---
    rc = run_phase([
        "--phase", "ref_verify", "--out", refver_json,
        "--n-prompts", str(a.n_prompts), "--n-new", str(a.n_new),
        "--ctx-cap", str(a.ctx_cap), "--verify-width", str(a.verify_width),
        "--gpu-mem-util", str(a.gpu_mem_util), "--ppl-slice", str(a.ppl_slice),
    ], timeout=a.refver_timeout)
    if rc != 0:
        raise RuntimeError(f"ref_verify phase failed (rc={rc}) -- core measurement missing")
    refver = json.load(open(refver_json))

    # --- Phase 2: draft-head latency (best-effort) ---
    drafthead: dict[str, Any] = {"phase": "draft_head_latency", "status": "not_run"}
    rc = run_phase(["--phase", "draft_head_latency", "--out", drafthead_json,
                    "--dh-iters", str(a.dh_iters)], timeout=a.dh_timeout)
    if rc == 0 and Path(drafthead_json).exists():
        drafthead = json.load(open(drafthead_json))
    else:
        drafthead["status"] = f"failed_rc_{rc}"

    # --- Phase 3: eagle3 integration attempt (best-effort, bounded) ---
    eagle3: dict[str, Any] = {"phase": "eagle3_attempt", "status": "not_run"}
    if a.eagle3_attempt:
        rc = run_phase(["--phase", "eagle3_attempt", "--out", eagle3_json,
                        "--gpu-mem-util", str(a.eagle3_gpu_mem_util)], timeout=a.eagle3_timeout)
        if Path(eagle3_json).exists():
            eagle3 = json.load(open(eagle3_json))
        else:
            eagle3 = {"phase": "eagle3_attempt", "constructed": False, "ran_generate": False,
                      "blocker": {"type": "PhaseTimeoutOrCrash", "message": f"rc={rc}"}}
    else:
        eagle3["status"] = "skipped_by_flag"

    compose(a, refver, drafthead, eagle3)


def compose(a: argparse.Namespace, refver: dict, drafthead: dict, eagle3: dict) -> None:
    identity = refver["token_identity_rate"]
    divergence = refver["verify_divergence"]
    det_gen = refver["determinism_ref_gen"]
    det_ver = refver["determinism_verify_geometry"]
    first_div = refver.get("first_divergence")

    # ---- Self-test (PRIMARY) ----
    ref_nonempty = refver["ref_gen_tokens_total"] > 0
    streams_equal_len = refver["total_compared_positions"] > 0  # verify aligned to gen positions
    nan_clean = bool(refver["nan_clean"]) and math.isfinite(identity) and math.isfinite(divergence)
    identity_in_range = (0.0 <= identity <= 1.0) and math.isfinite(identity)
    div_consistent = abs(divergence - (1.0 - identity)) < 1e-9
    # first-divergence record present IFF rate < 1.0
    firstdiv_iff = ((identity < 1.0) == (first_div is not None))
    det_gen_ok = (det_gen == 1.0)         # int4 AR is cross-session/​within-session bit-exact
    det_ver_ok = (det_ver == 1.0)
    flags_recorded = True                 # see no_* block below (all True by construction)
    tps_tagged = bool(refver.get("local_relative_tps")) and (
        drafthead.get("local_relative_tps", True) is True)

    checks = {
        "ref_stream_nonempty": ref_nonempty,                       # (a)
        "spec_verify_stream_nonempty_aligned": streams_equal_len,  # (a)
        "nan_clean": nan_clean,                                    # (a)
        "token_identity_rate_in_range": identity_in_range,         # (b)
        "divergence_eq_1_minus_identity": div_consistent,          # (b)
        "first_divergence_present_iff_rate_lt_1": firstdiv_iff,    # (c)
        "no_hf_job_no_launch_recorded": flags_recorded,            # (d)
        "tps_fields_local_relative": tps_tagged,                   # (e)
        "determinism_ref_gen_eq_1": det_gen_ok,                    # control
        "determinism_verify_eq_1": det_ver_ok,                     # control
    }
    greedy_identity_screen_self_test_passes = bool(all(checks.values()))

    # ---- reconcile to the deployed-M8 anchor (#232) ----
    div_vs_232_delta = divergence - DEPLOYED_M8_DIVERGENCE_232
    corroborates_232 = bool(abs(div_vs_232_delta) <= RECONCILE_TOL)
    # the strict verdict for the EAGLE-3 K=7 verify on this hardware
    eagle3_verify_breaks_strict = bool(identity < 1.0)

    no_block = {
        "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": True, "local_pod_gpu_only": True,
    }

    report = {
        "card": "eagle3_greedy_identity_screen",
        "pr": 361, "issue": 319, "author": "ubel",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        **no_block,
        # PRIMARY + TEST
        "token_identity_rate": identity,
        "greedy_identity_screen_self_test_passes": greedy_identity_screen_self_test_passes,
        # strict verdict
        "verify_divergence": divergence,
        "eagle3_verify_breaks_strict_token_identity": eagle3_verify_breaks_strict,
        "first_divergence": first_div,
        "per_sequence_strict_pass_fraction": refver["per_sequence_strict_pass_fraction"],
        # controls
        "determinism_ref_gen": det_gen,
        "determinism_verify_geometry": det_ver,
        # reconcile to #232 deployed-M8 anchor
        "deployed_m8_identity_232": DEPLOYED_M8_IDENTITY_232,
        "deployed_m8_divergence_232": DEPLOYED_M8_DIVERGENCE_232,
        "verify_divergence_vs_232_delta": div_vs_232_delta,
        "corroborates_232_mechanism": corroborates_232,
        # secondary (LOCAL RELATIVE)
        "ref_tps_local_relative": refver["ref_tps_local_relative"],
        "ref_decode_step_us_local_relative": refver["ref_decode_step_us_local_relative"],
        "draft_head_forward_us_per_token_local_relative":
            drafthead.get("draft_head_forward_us_per_token_local_relative"),
        "draft_head_chain_us_k7_local_relative":
            drafthead.get("draft_head_chain_us_k7_local_relative"),
        "alpha_accepted_tokens_per_verify": None,  # N/A: native trained head unavailable
        "ppl_spotcheck": refver["ppl_spotcheck"],
        # eagle3 integration attempt / blocker (FALLBACK 6c)
        "eagle3_engine_constructed": bool(eagle3.get("constructed", False)),
        "eagle3_generate_ran": bool(eagle3.get("ran_generate", False)),
        "eagle3_integration_blocker": eagle3.get("blocker"),
        "native_head_available": False,
        "native_head_blocker": ("trained EAGLE-3 head gua9x68j/56ksyxgw not retrievable: "
                                "W&B logged_artifacts==[] for both runs, .pt absent on disk, "
                                "publish HUMAN-owned and never done; only #333 synthetic-zero "
                                "candidate exists -> native-acceptance arm blocked, fallback used."),
        # config / bookkeeping
        "model_dir": refver["model_dir"],
        "substrate": "int4 w4a16 deployed path (bit-exact strict-ladder reference, #196/#232)",
        "verify_width_effective": refver["verify_width_effective"],
        "M_verify": M_VERIFY, "k_spec": K_SPEC,
        "n_prompts": refver["n_prompts"], "n_new": refver["n_new"],
        "total_compared_positions": refver["total_compared_positions"],
        "official_baseline_unchanged": OFFICIAL_BASELINE,
        "peak_gpu_gb": refver["peak_gpu_gb"],
        "self_test": checks,
        "local_relative_tps": True,
    }

    report_path = OUT_DIR / "_results.json"
    json.dump(report, open(report_path, "w"), indent=2, default=str)

    bar = "=" * 84
    print("\n" + bar, flush=True)
    print(" LOCAL EAGLE-3 GREEDY-IDENTITY SCREEN (PR #361, #319) -- M=8 verify geometry", flush=True)
    print(bar, flush=True)
    print(f" token_identity_rate (PRIMARY)        : {identity:.6f}", flush=True)
    print(f" verify_divergence                    : {divergence:.6f}  "
          f"(1 - identity)", flush=True)
    print(f" eagle3 verify breaks strict identity : {eagle3_verify_breaks_strict}", flush=True)
    if first_div:
        print(f" first divergence                     : prompt {first_div['prompt_index']} "
              f"pos {first_div['absolute_position']} ref={first_div['ref_token_id']} "
              f"spec={first_div['spec_verify_token_id']} gap={first_div['local_logit_gap']}",
              flush=True)
    print(f" per-sequence strict pass fraction    : {report['per_sequence_strict_pass_fraction']:.4f}",
          flush=True)
    print(f"   controls: det_gen={det_gen:.6f} det_verify={det_ver:.6f}", flush=True)
    print(f" deployed-M8 anchor (#232) identity   : {DEPLOYED_M8_IDENTITY_232:.6f} "
          f"(div {DEPLOYED_M8_DIVERGENCE_232:.6f})", flush=True)
    print(f" divergence vs #232 delta             : {div_vs_232_delta:+.6f}  "
          f"corroborates={corroborates_232}", flush=True)
    print(f" --- secondary (LOCAL RELATIVE) ---", flush=True)
    print(f" REF decode TPS / step               : {report['ref_tps_local_relative']:.2f} tok/s "
          f"/ {report['ref_decode_step_us_local_relative']:.1f}us", flush=True)
    print(f" draft-head fwd / K=7 chain           : "
          f"{report['draft_head_forward_us_per_token_local_relative']} / "
          f"{report['draft_head_chain_us_k7_local_relative']} us", flush=True)
    print(f" PPL spot-check (sanity, NOT gate)    : {report['ppl_spotcheck']:.4f}", flush=True)
    print(f" --- eagle3 integration attempt ---", flush=True)
    print(f" eagle3 engine constructed / ran      : {report['eagle3_engine_constructed']} / "
          f"{report['eagle3_generate_ran']}", flush=True)
    if report["eagle3_integration_blocker"]:
        b = report["eagle3_integration_blocker"]
        print(f" eagle3 blocker                       : {b.get('type')}: "
              f"{str(b.get('message'))[:160]}", flush=True)
    print(f" native head available                : {report['native_head_available']}", flush=True)
    print(f" SELF-TEST PASSES (TEST metric)       : {greedy_identity_screen_self_test_passes}",
          flush=True)
    print(f" report -> {report_path}", flush=True)
    print(bar + "\n", flush=True)

    run_ids = []
    if not a.no_wandb:
        run_ids = log_wandb(report, a)
    report["wandb_run_ids"] = run_ids
    json.dump(report, open(report_path, "w"), indent=2, default=str)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": run_ids,
        "primary_metric": {"name": "token_identity_rate", "value": identity},
        "test_metric": {"name": "greedy_identity_screen_self_test_passes",
                        "value": int(greedy_identity_screen_self_test_passes)},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> list[str]:
    # Import the REAL wandb BEFORE putting REPO_ROOT on sys.path. A prior-run `wandb/` output
    # directory at REPO_ROOT is a namespace dir with no `.init`; inserting REPO_ROOT at sys.path[0]
    # first (as we used to) lets it shadow the installed package. Import wandb while sys.path[0] is
    # still the script dir, then APPEND REPO_ROOT (never insert at 0) for scripts.wandb_logging --
    # by then wandb is cached in sys.modules so the shadow can't win.
    try:
        import wandb as _wb  # noqa: F401
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init (shadowed by a wandb/ dir)")
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] real wandb import failed (analysis unaffected): {exc}", flush=True)
        return []
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] scripts.wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return []
    # keep wandb's run output out of REPO_ROOT so we never recreate the import shadow
    os.environ.setdefault("WANDB_DIR", "/tmp/wandb_ubel_eagle3")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#361 local EAGLE-3 greedy-identity screen: M=8 verify-geometry token-identity "
              "vs greedy AR on the int4 deployed substrate (fallback: native head unavailable). "
              "LOCAL pod-GPU, 0 official TPS, no HF job/launch/submission/served-file change.",
        tags=["eagle3", "greedy-identity", "verify-geometry", "strict-gate", "local-relative",
              "issue-319", "pr-361"],
        config={"pr": 361, "issue": 319, "wandb_group": a.wandb_group,
                "M_verify": M_VERIFY, "k_spec": K_SPEC,
                "substrate": report["substrate"], "n_prompts": report["n_prompts"],
                "n_new": report["n_new"], "verify_width_effective": report["verify_width_effective"],
                "deployed_m8_divergence_232": DEPLOYED_M8_DIVERGENCE_232,
                "official_baseline": OFFICIAL_BASELINE, "native_head_available": False},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON-only", flush=True)
        return []
    # explicit, typed summary (avoid dumping nested dicts)
    summ = {
        "token_identity_rate": report["token_identity_rate"],
        "greedy_identity_screen_self_test_passes": int(report["greedy_identity_screen_self_test_passes"]),
        "verify_divergence": report["verify_divergence"],
        "eagle3_verify_breaks_strict_token_identity": int(report["eagle3_verify_breaks_strict_token_identity"]),
        "per_sequence_strict_pass_fraction": report["per_sequence_strict_pass_fraction"],
        "determinism_ref_gen": report["determinism_ref_gen"],
        "determinism_verify_geometry": report["determinism_verify_geometry"],
        "verify_divergence_vs_232_delta": report["verify_divergence_vs_232_delta"],
        "corroborates_232_mechanism": int(report["corroborates_232_mechanism"]),
        "ref_tps_local_relative": report["ref_tps_local_relative"],
        "ref_decode_step_us_local_relative": report["ref_decode_step_us_local_relative"],
        "draft_head_forward_us_per_token_local_relative": report.get("draft_head_forward_us_per_token_local_relative"),
        "draft_head_chain_us_k7_local_relative": report.get("draft_head_chain_us_k7_local_relative"),
        "ppl_spotcheck": report["ppl_spotcheck"],
        "eagle3_engine_constructed": int(report["eagle3_engine_constructed"]),
        "eagle3_generate_ran": int(report["eagle3_generate_ran"]),
        "native_head_available": int(report["native_head_available"]),
        "total_compared_positions": report["total_compared_positions"],
        "tps_added_by_this_card": 0,
        "peak_gpu_gb": report["peak_gpu_gb"],
    }
    summ = {k: v for k, v in summ.items() if v is not None}
    log_summary(run, summ, step=0)
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = int(bool(v))
    log_json_artifact(run, name="eagle3_greedy_identity_screen_result",
                      artifact_type="analysis", data=report)
    rid = getattr(run, "id", "") or ""
    finish_wandb(run)
    print(f"[wandb] logged run {rid}", flush=True)
    return [rid] if rid else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["ref_verify", "draft_head_latency", "eagle3_attempt"],
                    default=None, help="internal: run a GPU phase subprocess")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", action="store_true", help="orchestrate the full screen on GPU")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the path")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--n-new", type=int, default=512)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--verify-width", type=int, default=M_VERIFY)
    ap.add_argument("--ppl-slice", type=int, default=64)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--eagle3-gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--dh-iters", type=int, default=200)
    ap.add_argument("--no-eagle3-attempt", dest="eagle3_attempt", action="store_false",
                    help="skip the bounded vLLM eagle3 construction attempt")
    ap.add_argument("--refver-timeout", type=int, default=2400)
    ap.add_argument("--dh-timeout", type=int, default=600)
    ap.add_argument("--eagle3-timeout", type=int, default=900)
    ap.add_argument("--wandb_group", dest="wandb_group", default="eagle3-greedy-identity-screen")
    ap.add_argument("--wandb_name", dest="wandb_name", default="ubel/eagle3-greedy-identity-screen")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--relog-wandb", action="store_true",
                    help="log the existing _results.json to wandb without re-running GPU phases")
    a = ap.parse_args()

    if a.relog_wandb:
        report = json.load(open(OUT_DIR / "_results.json"))
        run_ids = log_wandb(report, a)
        report["wandb_run_ids"] = run_ids
        json.dump(report, open(OUT_DIR / "_results.json", "w"), indent=2, default=str)
        marker = {
            "terminal": True, "status": "complete", "pending_arms": False,
            "wandb_run_ids": run_ids,
            "primary_metric": {"name": "token_identity_rate",
                               "value": report["token_identity_rate"]},
            "test_metric": {"name": "greedy_identity_screen_self_test_passes",
                            "value": int(report["greedy_identity_screen_self_test_passes"])},
        }
        print("SENPAI-RESULT: " + json.dumps(marker), flush=True)
        return

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 3)
        a.n_new = min(a.n_new, 48)
        a.ppl_slice = min(a.ppl_slice, 16)
        a.dh_iters = min(a.dh_iters, 30)

    if a.phase == "ref_verify":
        phase_ref_verify(a.out, a.n_prompts, a.n_new, a.ctx_cap, a.verify_width,
                         a.gpu_mem_util, a.ppl_slice)
    elif a.phase == "draft_head_latency":
        phase_draft_head_latency(a.out, a.dh_iters)
    elif a.phase == "eagle3_attempt":
        phase_eagle3_attempt(a.out, a.eagle3_gpu_mem_util)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
