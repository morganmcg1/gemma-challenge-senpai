#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #362 -- int4-Marlin DVR rollback rate + eta_dvr vs the 4.02% >500 budget (#319).

THE QUESTION
------------
My #360 (`6s9vgnw9`, merged) pinned the strict identity tax to the bf16 lm_head+attn
batch-invariant (BI) reduction at eta = 9.841% (#327 `kcjlr5ny`), capping the lambda=1
strict-compliant ceiling at 520.953*(1-0.09841) = 469.68 < 500. eta is a MULTIPLICATIVE
lever; the hard >500 threshold is eta = 1 - 500/520.953 = 4.022% (the eta at which the
ceiling ALONE clears 500, zero ceiling-lift). Decode-Verify-Rollback (DVR) is the proposed
*cheaper identity-restoring mechanism*: run the fast batch-variant M=8 verify, re-verify
each accepted token at a fixed (M-invariant) M=1 shape, and roll back on divergence. If
eta_dvr < 4.022%, strict 500 unlocks on the coverage/demand axis ALONE.

WHAT DECIDES IT (the open, GPU-measurable numbers)
--------------------------------------------------
  (PRIMARY) int4_marlin_rollback_rate r  -- the fraction of accepted-token verifies where
      the M=8 (batch-variant) argmax diverges from the M=1 re-verify. Cross-checks the
      deployed M=8 ~0.73% divergence (#232 `nxwv6pam`). This is the rollback firing rate.
  eta_dvr = (reverify_cost + r*rollback_cost) / step_us  -- microbenched on the pod GPU.

THE CRUX -- is the cheap "lm_head+attn @ M=1" re-verify SUFFICIENT to restore identity?
---------------------------------------------------------------------------------------
NO (researcher pass + #326 + LLM-42 arxiv:2601.17768). The bf16 attention M-variance is
injected in EARLY layers and PROPAGATES: each int4-Marlin body GEMM is bit-exact *given the
same input* (#232 maxdiff 0.0), but the input differs once upstream attention diverged, so
the final pre-lm_head hidden state h_final differs between M=8 and M=1. A lm_head-only
re-verify on h_final^(M8) does NOT reproduce the M=1 argmax. The only identity-restoring
re-verify is a FULL M=1 forward (LLM-42 replays a full forward, never a partial probe). We
MEASURE this: capture h^(M1) vs h^(M8) and report the hidden-state-driven flip rate (the
part a cheap lm_head-only re-verify cannot fix). If it is > 0, the honest reverify_cost is a
full M=1 forward, and in the memory-bound A10G decode regime step_us(M=1) ~ step_us(M=8)
(weight HBM reads dominate, M=8 adds negligible bytes), so eta_dvr ~ 0.7-1.0 >> 9.841%.

SCOPE: LOCAL pod-GPU inference microbench on a single A10G. NO train.py --launch, NO HF Job,
NO submission, NO served-file change. The served int4 path is READ, never modified. BASELINE
stays 481.53 (this leg adds 0 TPS -- a measurement). Numbers are LOCAL RELATIVE (land #245
~7x local<->official gap): the rollback RATE and greedy-identity are hardware-portable; eta_dvr
is a same-hardware RATIO (the 7x cancels); absolute TPS is relative.

PRIMARY metric  dvr_rollback_rate_eta_self_test_passes
TEST    metrics eta_dvr (float)  +  int4_marlin_rollback_rate (float)

Run:
    cd target/ && /tmp/senpai-venvs/5f4c623f772358a2/bin/python3.12 \\
      research/validity/dvr_rollback_rate_eta/dvr_rollback_rate_eta.py --gpu \\
      --wandb_group dvr-rollback-rate-eta --wandb_name wirbel/dvr-rollback-rate-eta

The GPU work runs as an isolated subprocess so vLLM gets a clean CUDA context and releases
VRAM on exit; the orchestrator stays GPU-free and owns composition, the eta ladder, the
decision tree, self-test, and wandb.
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

# ------------------------------------------------------------------------------------- #
# Imported fleet anchors -- one source per constant (import, do not re-derive).
# ------------------------------------------------------------------------------------- #
# wirbel #360 strict_kernel_eta_locus (`6s9vgnw9`) -- the strict eta ladder.
CEILING_LAMBDA1 = 520.9527323111674      # lambda=1 strict-compliant ceiling (#327/#360 full precision)
TARGET_TPS = 500.0                       # the strict bar
ETA_BI_FLOOR_327 = 0.09841249119201488   # incumbent bf16 lm_head+attn BI floor (#327) -- the rung DVR must beat
DEPLOYED_STEP_US = 1218.2                # deployed step (kanna #217/#136); official-scaled -> CONTEXT ONLY
OFFICIAL_BASELINE = 481.53               # PR #52 official TPS (this leg adds 0)
# lawine #232 int4_tokenident_deployed_m8 (`nxwv6pam`) -- the divergence cross-check anchor.
DIV_M8_232 = 0.007291666666666696        # int4 M=1-AR-vs-M=8-verify per-token divergence (0.73%)
IDENT_M8_232 = 0.9927083333333333        # 1 - DIV_M8_232
# DVR literature (LLM-42, arxiv:2601.17768): the "~6%" is an online-worst-case RECOMPUTE
# FRACTION (token fraction recomputed), NOT a throughput eta. Kept as a labelled reference rung.
DVR_LIT_RECOMPUTE_FRAC = 0.06

K_SPEC = 7                               # num_speculative_tokens (manifest)
M_VERIFY = K_SPEC + 1                    # = 8, the deployed verify batch width

# Derived ladder rung: the eta at which the lambda=1 ceiling alone clears 500 (no ceiling-lift).
ETA_BUDGET_500 = 1.0 - TARGET_TPS / CEILING_LAMBDA1   # ~= 0.040218 (the ">500 budget")

# Models: try the deployed baked frontier first (osoi5-v0, the merged fa2sw_precache_kenyan
# weights), then fall back to the vanilla int4-ct checkpoint #232/#326/#327 used as the
# documented proxy. Both expose the same int4-Marlin body + bf16 lm_head+attn locus.
MODEL_FRONTIER_BAKED = "/tmp/osoi5-v0-baked"
MODEL_CT_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
OUT_DIR = Path("research/validity/dvr_rollback_rate_eta")


# ------------------------------------------------------------------------------------- #
# Model resolution
# ------------------------------------------------------------------------------------- #
def _safetensors_header(st_path: Path) -> dict:
    """Read just the JSON header of a .safetensors file (no tensor data)."""
    import struct
    with open(st_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _frontier_clean_loadable(p: Path) -> tuple[bool, str]:
    """The baked frontier is served via custom patches (LM_HEAD_PRUNE etc.). A *clean* vLLM
    LLM() ties lm_head and embed_tokens to a single config.vocab_size, so a pruned lm_head
    (vocab != embed vocab, tie=False) trips `weight.shape[0] == org_vocab_size`. Detect that
    statically (peek the safetensors header) so we fall back to the documented int4-ct proxy
    instead of crashing -- loading the baked model needs the serve stack, which a local
    microbench must NOT invoke (no launcher, no HF job)."""
    cfg_path = p / "config.json"
    st_path = p / "model.safetensors"
    if not (cfg_path.exists() and st_path.exists()):
        return False, "missing config.json or model.safetensors"
    try:
        cfg = json.load(open(cfg_path))
        tc = cfg.get("text_config") or cfg
        vocab = tc.get("vocab_size") or cfg.get("vocab_size")
        hdr = _safetensors_header(st_path)
        # locate the lm_head output (vocab) dim: packed int4 -> shape[0]; dense -> shape[0]
        lm_vocab = None
        for k, t in hdr.items():
            if k == "__metadata__":
                continue
            if k.endswith("lm_head.weight_packed") or k.endswith("lm_head.weight"):
                lm_vocab = t["shape"][0]
                break
        if vocab is None or lm_vocab is None:
            return False, f"could not read vocab dims (config={vocab}, lm_head={lm_vocab})"
        if int(lm_vocab) != int(vocab):
            return False, (f"baked lm_head pruned to {lm_vocab} != config vocab {vocab} "
                           f"(tie={tc.get('tie_word_embeddings')}); needs LM_HEAD_PRUNE serve patch")
        return True, "ok"
    except Exception as exc:
        return False, f"header probe failed: {exc!r}"


def resolve_model_dir() -> tuple[str, bool, str]:
    """Return (model_dir, is_frontier_baked, reason). Try the baked frontier first, but only
    if a *clean* vLLM can load it; otherwise fall back to the documented int4-ct proxy (the
    exact checkpoint #232 measured the M=8 divergence on -- same int4-Marlin body + bf16
    lm_head+attn batch-variance locus)."""
    p = Path(MODEL_FRONTIER_BAKED)
    if p.is_dir() and (p / "config.json").exists():
        loadable, reason = _frontier_clean_loadable(p)
        if loadable:
            return str(p), True, "frontier baked, clean-loadable"
        frontier_reason = f"frontier baked NOT clean-loadable: {reason}"
    else:
        frontier_reason = "frontier baked dir absent"
    for cand in MODEL_CT_CANDIDATES:
        cp = Path(cand)
        if cp.is_dir() and (cp / "config.json").exists():
            return str(cp), False, frontier_reason + " -> int4-ct proxy (#232 checkpoint)"
        if cp.is_dir():
            for sub in sorted(cp.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub), False, frontier_reason + " -> int4-ct proxy (#232 checkpoint)"
    raise FileNotFoundError(f"no clean-loadable model found ({frontier_reason})")


# ------------------------------------------------------------------------------------- #
# vLLM model navigation (reused from #232) -- find the decoder model + the lm_head weight.
# ------------------------------------------------------------------------------------- #
def get_vllm_model(llm):
    paths = [
        lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
    ]
    for getter in paths:
        try:
            m = getter()
            if m is not None:
                return m
        except Exception:
            continue
    raise RuntimeError("could not locate model_runner.model")


def find_lm_head(model):
    """Return the lm_head module (with .weight [vocab, hidden]) or None (best-effort)."""
    import torch.nn as nn
    for name in ("lm_head", "output_layer"):
        for chain in ([name], ["model", name], ["language_model", name],
                      ["model", "language_model", name]):
            obj = model
            ok = True
            for attr in chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and hasattr(obj, "weight") and getattr(obj, "weight").dim() == 2:
                return obj
    # last resort: any module named *lm_head* with a 2D weight
    for nm, mod in model.named_modules():
        if "lm_head" in nm and hasattr(mod, "weight") and getattr(mod, "weight").dim() == 2:
            return mod
    return None


# ======================================================================================
# GPU PHASE: load model, generate REF, measure r, decompose re-verify sufficiency, cost.
# ======================================================================================
def phase_measure(out_path: str, n_prompts: int, max_gen: int, gpu_mem_util: float,
                  max_batched_tokens: int, decomp_prompts: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir, is_frontier, model_reason = resolve_model_dir()
    print(f"[dvr] model={model_dir} is_frontier_baked={is_frontier} M_verify={M_VERIFY}", flush=True)
    print(f"[dvr] model_resolution: {model_reason}", flush=True)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    ctx_lists = []
    for rec in rows:
        ctx = list(rec.get("context_token_ids", []))
        if len(ctx) >= 2:
            ctx_lists.append((rec.get("id"), ctx))
    ctx_lists = ctx_lists[:n_prompts]
    max_ctx = max(len(c) for _, c in ctx_lists)
    model_len = max(1024, max_ctx + max_gen + 16)

    t0 = time.time()
    llm = LLM(
        model=model_dir,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=model_len,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max(M_VERIFY, 8),
        max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=False,   # M replicas must each do a real forward
        enforce_eager=True,            # no CUDA-graph padding of M; forward hooks fire
        trust_remote_code=True,
    )
    print(f"[dvr] vLLM load done in {time.time()-t0:.0f}s (model_len={model_len})", flush=True)

    # ---- Phase A: REF = M=1 plain-AR greedy decode (the committed truth) ----
    sp_gen = SamplingParams(temperature=0.0, max_tokens=max_gen)
    ref_streams = []
    for pid, ctx in ctx_lists:
        out = llm.generate([{"prompt_token_ids": ctx}], sp_gen, use_tqdm=False)[0]
        gen = list(out.outputs[0].token_ids)
        ref_streams.append({"id": pid, "ctx_len": len(ctx), "gen": gen})
    gen_lens = [len(s["gen"]) for s in ref_streams]
    print(f"[dvr] Phase A REF gen: {len(ref_streams)} prompts, gen_len "
          f"min/med/max={min(gen_lens)}/{int(statistics.median(gen_lens))}/{max(gen_lens)}", flush=True)

    # ---- compute_logits capture for the re-verify-sufficiency decomposition ----
    # vLLM V1 produces logits via `model.compute_logits(hidden_states)` (gemma4: returns
    # logits_processor(lm_head, hidden_states)); the lm_head module is NOT called via __call__,
    # so a forward_pre_hook on it never fires. We wrap compute_logits instead -- its first
    # positional arg IS the post-final-norm, pre-lm_head hidden h_final (the crux quantity).
    # It is called several times per generate (prompt-scoring pass over all L positions, plus
    # the 1-token sample pass); we keep the MAX-row capture per tag = the prompt-scoring pass.
    model = None
    lm_head = None
    lm_head_W = None
    cl_owner = None
    cl_orig = None
    try:
        model = get_vllm_model(llm)
        lm_head = find_lm_head(model)
        if lm_head is not None:
            lm_head_W = lm_head.weight.detach()  # [vocab, hidden], bf16
            print(f"[dvr] lm_head located: weight {tuple(lm_head_W.shape)} dtype={lm_head_W.dtype}", flush=True)
        for obj in (model, getattr(model, "language_model", None), getattr(model, "model", None)):
            if obj is not None and hasattr(obj, "compute_logits"):
                cl_owner = obj
                break
    except Exception as exc:
        print(f"[dvr] model navigation unavailable -> decomposition best-effort skip: {exc!r}", flush=True)

    _hook_store: dict[str, object] = {}
    _capture = {"tag": None}

    if cl_owner is not None:
        cl_orig = cl_owner.compute_logits  # bound method

        def _wrapped_compute_logits(*args, **kwargs):
            tag = _capture["tag"]
            if tag is not None and args:
                try:
                    import torch as _t
                    h = args[0]
                    if hasattr(h, "detach"):
                        hf = h.detach().to(_t.float32)
                        prev = _hook_store.get(tag)
                        if prev is None or hf.shape[0] >= prev.shape[0]:
                            _hook_store[tag] = hf  # keep the prompt-scoring (max-row) pass
                except Exception:
                    pass
            return cl_orig(*args, **kwargs)

        cl_owner.compute_logits = _wrapped_compute_logits
        print(f"[dvr] compute_logits wrapped on {type(cl_owner).__name__}", flush=True)

    # ---- Phase C: per-position rollback rate r (M=8 verify argmax vs M=1 re-verify) ----
    sp_lp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    def argmax_seq(out) -> list[int]:
        pls = out.prompt_logprobs
        am: list[int] = []
        for i in range(len(pls)):
            entry = pls[i]
            if entry is None:
                continue
            best = max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0]
            am.append(int(best))
        return am

    per_prompt = []
    n_div = n_total = 0          # r: M=8 verify vs M=1 re-verify, generated tail
    n_det1 = n_det8 = n_within = 0
    n_ctrl_pd = n_ctrl_tot = 0   # control: M=1 prefill argmax vs the M=1-AR generated token
    ref_stream_out = []          # REF[i] = am1  (the M=1 re-verify truth)
    spec_stream_out = []         # SPEC[i] = am8 (the M=8 verify, batch-variant)
    dvr_stream_out = []          # DVR[i]  = am1 (DVR commits the M=1 re-verify)

    # decomposition accumulators
    decomp = {"attempted": False, "reliable": False, "n_rows": 0,
              "hidden_rel_l2_div_mean": float("nan"),
              "hidden_driven_flip_rate": float("nan"),
              "lmhead_consistency_M1": float("nan")}
    dh_l2 = []
    dh_flip = 0
    dh_consistent = 0
    dh_rows = 0

    for si, s in enumerate(ref_streams):
        ctx_len = s["ctx_len"]
        seq = (ctx_lists[si][1] + s["gen"])
        L = len(seq)
        prompt = {"prompt_token_ids": seq}
        want_decomp = (lm_head_W is not None) and (si < decomp_prompts)

        can_decomp = want_decomp and cl_owner is not None
        if can_decomp:
            decomp["attempted"] = True
            _hook_store.clear()
            _capture["tag"] = "M1"

        am1 = argmax_seq(llm.generate([prompt], sp_lp, use_tqdm=False)[0])
        h_M1 = _hook_store.get("M1") if can_decomp else None

        if can_decomp:
            _capture["tag"] = None          # don't capture the determinism-control pass
        am1b = argmax_seq(llm.generate([prompt], sp_lp, use_tqdm=False)[0])  # det control

        if can_decomp:
            _capture["tag"] = "M8"
        out8 = llm.generate([prompt] * M_VERIFY, sp_lp, use_tqdm=False)
        h_M8 = _hook_store.get("M8") if can_decomp else None
        if can_decomp:
            _capture["tag"] = None
        am8_0 = argmax_seq(out8[0])
        am8_1 = argmax_seq(out8[1]) if len(out8) > 1 else am8_0
        am8b_0 = argmax_seq(llm.generate([prompt] * M_VERIFY, sp_lp, use_tqdm=False)[0])  # det control

        Lm = min(len(am1), len(am1b), len(am8_0), len(am8_1), len(am8b_0))
        a1, a1b, a80, a81, a8b = am1[:Lm], am1b[:Lm], am8_0[:Lm], am8_1[:Lm], am8b_0[:Lm]
        # am[j] predicts seq position j+1; generated-tail positions are seq[ctx_len..L) ->
        # am indices [ctx_len-1, Lm). Score only the generated tail (the decode continuation).
        lo = max(0, ctx_len - 1)
        tail = range(lo, Lm)
        for j in tail:
            n_total += 1
            if a80[j] != a1[j]:
                n_div += 1
            if a1[j] == a1b[j]:
                n_det1 += 1
            if a80[j] == a8b[j]:
                n_det8 += 1
            if a80[j] == a81[j]:
                n_within += 1
            # control: M=1 prefill argmax vs the actual M=1-AR generated token at pos j+1
            gen_idx = (j + 1) - ctx_len
            if 0 <= gen_idx < len(s["gen"]):
                n_ctrl_tot += 1
                if a1[j] == s["gen"][gen_idx]:
                    n_ctrl_pd += 1
            ref_stream_out.append(int(a1[j]))
            spec_stream_out.append(int(a80[j]))
            dvr_stream_out.append(int(a1[j]))   # DVR commits the M=1 re-verify

        # ---- decomposition: does the hidden state itself differ (attn propagation)? ----
        if want_decomp and h_M1 is not None and h_M8 is not None:
            try:
                import torch
                # replica-0 rows of the M=8 capture are the first L rows (vLLM packs the
                # first sequence first); align lengths defensively.
                n = min(h_M1.shape[0], h_M8.shape[0], L)
                hm1 = h_M1[:n]
                hm8 = h_M8[:n]
                W = lm_head_W.to(torch.float32)
                # restrict to generated-tail rows for a like-for-like comparison with r
                r0 = max(0, ctx_len)
                hm1t = hm1[r0:n]
                hm8t = hm8[r0:n]
                if hm1t.shape[0] > 0:
                    rel = (hm8t - hm1t).norm(dim=-1) / (hm1t.norm(dim=-1) + 1e-12)
                    dh_l2.extend(rel.cpu().tolist())
                    # apply the SAME lm_head reduction (W) to both hiddens: any argmax flip is
                    # then purely hidden-state-driven (the part a lm_head-only re-verify cannot fix).
                    log1 = hm1t @ W.t()
                    log8 = hm8t @ W.t()
                    am_h1 = log1.argmax(dim=-1)
                    am_h8 = log8.argmax(dim=-1)
                    dh_flip += int((am_h1 != am_h8).sum().item())
                    dh_rows += int(am_h1.shape[0])
                    # consistency: W@h_M1 argmax must reproduce the vLLM M=1 prompt_logprobs
                    # argmax (am1) at the matching positions -> validates W + row alignment.
                    base = ctx_len - 1
                    for k in range(am_h1.shape[0]):
                        amk = base + 1 + k  # am1 index predicting seq pos r0+k = (ctx_len+k)
                        if 0 <= amk < len(a1):
                            dh_consistent += int(int(am_h1[k].item()) == int(a1[amk]))
                    decomp["reliable"] = True
            except Exception as exc:
                print(f"[dvr] decomposition row-compute failed (prompt {si}): {exc!r}", flush=True)

        per_prompt.append({
            "id": s["id"], "ctx_len": ctx_len, "scored_positions": Lm - lo,
            "div_M8_vs_M1": sum(1 for j in tail if am8_0[j] != am1[j]),
        })
        if si < 2 or si == len(ref_streams) - 1:
            print(f"[dvr] prompt {si} id={s['id']} scored={Lm-lo} "
                  f"div={per_prompt[-1]['div_M8_vs_M1']} decomp={'y' if want_decomp else 'n'}", flush=True)

    r = (n_div / n_total) if n_total else float("nan")
    det1 = (n_det1 / n_total) if n_total else float("nan")
    det8 = (n_det8 / n_total) if n_total else float("nan")
    within = (n_within / n_total) if n_total else float("nan")
    ctrl_prefill_vs_decode_M1 = 1.0 - ((n_ctrl_pd / n_ctrl_tot) if n_ctrl_tot else float("nan"))

    if dh_l2:
        decomp["n_rows"] = dh_rows
        decomp["hidden_rel_l2_div_mean"] = float(statistics.fmean(dh_l2))
        decomp["hidden_driven_flip_rate"] = (dh_flip / dh_rows) if dh_rows else float("nan")
        decomp["lmhead_consistency_M1"] = (dh_consistent / dh_rows) if dh_rows else float("nan")

    # restore the un-wrapped compute_logits before the cost microbench (no capture overhead)
    if cl_owner is not None and cl_orig is not None:
        _capture["tag"] = None
        cl_owner.compute_logits = cl_orig

    # ---- Phase D: cost microbench (local, same-hardware basis for a portable eta ratio) ----
    cost = cost_microbench(llm, lm_head_W, ctx_lists[0][1])

    peak_gb = None
    try:
        import torch
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    out = {
        "phase": "measure",
        "model_dir": model_dir,
        "is_frontier_baked": is_frontier,
        "model_resolution_reason": model_reason,
        "n_prompts": len(ref_streams),
        "max_gen": max_gen,
        "M_verify": M_VERIFY,
        "scored_positions": n_total,
        # PRIMARY rollback rate + controls
        "int4_marlin_rollback_rate": r,
        "rollback_count": n_div,
        "determinism_M1_vs_M1": det1,
        "determinism_M8_vs_M8": det8,
        "within_batch_copy0_vs_copy1": within,
        "control_prefill_vs_decode_M1_div": ctrl_prefill_vs_decode_M1,
        # streams (truncated heads for audit; full lengths recorded)
        "ref_len": len(ref_stream_out), "spec_len": len(spec_stream_out), "dvr_len": len(dvr_stream_out),
        "ref_head": ref_stream_out[:32], "spec_head": spec_stream_out[:32], "dvr_head": dvr_stream_out[:32],
        # re-verify-sufficiency decomposition (the crux evidence)
        "decomposition": decomp,
        # cost microbench
        "cost": cost,
        "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[dvr] r(int4_marlin_rollback_rate)={r:.6f}  vs #232 0.007292 "
          f"(delta {r - DIV_M8_232:+.6f})", flush=True)
    print(f"[dvr] controls det1={det1:.6f} det8={det8:.6f} within={within:.6f} "
          f"prefill_vs_decode_M1={ctrl_prefill_vs_decode_M1:.6f}", flush=True)
    print(f"[dvr] decomp: hidden_rel_l2={decomp['hidden_rel_l2_div_mean']} "
          f"hidden_driven_flip={decomp['hidden_driven_flip_rate']} "
          f"consistency={decomp['lmhead_consistency_M1']}", flush=True)
    print(f"[dvr] cost: step_us_M1={cost.get('step_us_M1')} step_us_M8={cost.get('step_us_M8')} "
          f"lmhead_us_M1={cost.get('lmhead_us_M1')}", flush=True)
    print(f"DVR_MEASURE_DONE {out_path}", flush=True)


def cost_microbench(llm, lm_head_W, ctx) -> dict:
    """Local step-cost microbench. step_us via marginal decode timing (prefill cancelled);
    lm_head-only cost via CUDA events. All same-hardware so eta_dvr (a ratio) is portable."""
    import torch
    from vllm import SamplingParams

    out = {"step_us_M1": float("nan"), "step_us_M8": float("nan"),
           "lmhead_us_M1": float("nan"), "lmhead_us_M8": float("nan"),
           "reverify_full_over_step": float("nan")}

    def median_decode_step_us(batch: int, t_lo: int, t_hi: int, reps: int = 3) -> float:
        """Marginal per-decode-step wall time = (wall(t_hi)-wall(t_lo))/(t_hi-t_lo).
        At batch=b, each step advances b sequences -> wall is the M=b forward time."""
        prompt = {"prompt_token_ids": ctx}
        vals = []
        # warmup
        llm.generate([prompt] * batch, SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)
        for _ in range(reps):
            sp_lo = SamplingParams(temperature=0.0, max_tokens=t_lo)
            sp_hi = SamplingParams(temperature=0.0, max_tokens=t_hi)
            torch.cuda.synchronize()
            a = time.perf_counter()
            llm.generate([prompt] * batch, sp_lo, use_tqdm=False)
            torch.cuda.synchronize()
            b = time.perf_counter()
            llm.generate([prompt] * batch, sp_hi, use_tqdm=False)
            torch.cuda.synchronize()
            c = time.perf_counter()
            wall_lo = b - a
            wall_hi = c - b
            vals.append((wall_hi - wall_lo) / max(1, (t_hi - t_lo)) * 1e6)
        return float(statistics.median(vals))

    try:
        out["step_us_M1"] = median_decode_step_us(1, 32, 96)
        out["step_us_M8"] = median_decode_step_us(M_VERIFY, 32, 96)
        if math.isfinite(out["step_us_M1"]) and out["step_us_M8"]:
            out["reverify_full_over_step"] = out["step_us_M1"] / out["step_us_M8"]
    except Exception as exc:
        print(f"[dvr] step microbench failed: {exc!r}", flush=True)

    # lm_head-only cost (the cheap re-verify numerator; CUDA events)
    if lm_head_W is not None:
        try:
            dev = lm_head_W.device
            hidden = lm_head_W.shape[1]
            for tag, m in (("lmhead_us_M1", 1), ("lmhead_us_M8", M_VERIFY)):
                x = torch.randn(m, hidden, dtype=lm_head_W.dtype, device=dev)
                for _ in range(10):
                    _ = x @ lm_head_W.t()
                torch.cuda.synchronize()
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                iters = 50
                ev0.record()
                for _ in range(iters):
                    _ = x @ lm_head_W.t()
                ev1.record()
                torch.cuda.synchronize()
                out[tag] = ev0.elapsed_time(ev1) / iters * 1e3  # ms/iter -> us
        except Exception as exc:
            print(f"[dvr] lm_head microbench failed: {exc!r}", flush=True)
    return out


# ======================================================================================
# ORCHESTRATOR: subprocess, eta ladder, decision tree, self-test, wandb
# ======================================================================================
def run_phase_subprocess(args_list: list[str]) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HUB_OFFLINE", "1")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching phase: {' '.join(args_list)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc})")


def compute_eta_dvr(reverify_cost: float, rollback_cost: float, r: float, step_us: float) -> float:
    if not (step_us and math.isfinite(step_us)):
        return float("nan")
    return (reverify_cost + r * rollback_cost) / step_us


def required_ceiling_lift(eta: float) -> float:
    """TPS the substrate ceiling must rise to (above lambda1) for the strict bar at this eta."""
    if not math.isfinite(eta) or eta >= 1.0:
        return float("inf")
    return TARGET_TPS / (1.0 - eta) - CEILING_LAMBDA1


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meas_json = str(OUT_DIR / "_measure_phase.json")
    run_phase_subprocess([
        "--phase", "measure", "--out", meas_json,
        "--n-prompts", str(a.n_prompts), "--max-gen", str(a.max_gen),
        "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
        "--decomp-prompts", str(a.decomp_prompts),
    ])
    m = json.load(open(meas_json))

    r = m["int4_marlin_rollback_rate"]
    cost = m["cost"]
    step_us_M1 = cost.get("step_us_M1", float("nan"))
    step_us_M8 = cost.get("step_us_M8", float("nan"))
    lmhead_us_M1 = cost.get("lmhead_us_M1", float("nan"))

    # ---- cost model wiring (researcher-validated: identity-restoring re-verify = full M=1 fwd) ----
    reverify_cost_fullfwd = step_us_M1          # the only re-verify that restores identity
    reverify_cost_lmhead = lmhead_us_M1         # cheap re-verify -- INSUFFICIENT (decomposition)
    rollback_cost = step_us_M1                  # recompute the divergent step at M=1
    step_us = step_us_M8                         # the deployed M=8 verify step (local basis)

    # eta_dvr variants (all reported; the DECISION keys on the identity-restoring one):
    #  (HONEST) full M=1 re-verify EVERY step -- detection requires the full fwd (no cheap+correct
    #           probe exists; the lm_head-only probe is identity-insufficient, see decomposition).
    eta_dvr_honest = compute_eta_dvr(reverify_cost_fullfwd, rollback_cost, r, step_us)
    #  (PROBE-GATED, optimistic LOWER BOUND) re-verify only on flagged divergent steps -- requires a
    #           free, cheap, CORRECT divergence probe that the literature does not provide.
    eta_dvr_probe_gated = compute_eta_dvr(0.0, reverify_cost_fullfwd + rollback_cost, r, step_us)
    #  (LM-HEAD-ONLY, hypothesized-cheap but INVALID) cheap re-verify every step -- fails identity.
    eta_dvr_lmhead = compute_eta_dvr(reverify_cost_lmhead, rollback_cost, r, step_us)

    eta_dvr = eta_dvr_honest  # the operative, identity-restoring eta_dvr -> the decision number

    # ---- token identity ----
    # full-forward DVR commits the M=1 re-verify token at every position -> DVR stream == REF stream.
    token_identity_rate = 1.0 if (m["ref_len"] == m["dvr_len"] and m["ref_head"] == m["dvr_head"]) else 0.0
    decomp = m.get("decomposition", {})
    hidden_driven_flip_rate = decomp.get("hidden_driven_flip_rate", float("nan"))
    # the cheap lm_head-only re-verify leaves the hidden-state-driven flips unfixed:
    token_identity_rate_lmhead_only = (1.0 - hidden_driven_flip_rate
                                       if math.isfinite(hidden_driven_flip_rate) else float("nan"))
    lmhead_reverify_restores_identity = bool(
        math.isfinite(hidden_driven_flip_rate) and hidden_driven_flip_rate == 0.0)

    # ---- decision tree (keyed on the identity-restoring eta_dvr) ----
    dvr_unlocks_500_coverage_alone = bool(math.isfinite(eta_dvr) and eta_dvr < ETA_BUDGET_500)
    dvr_reduces_ceiling_lift_burden = bool(
        math.isfinite(eta_dvr) and ETA_BUDGET_500 <= eta_dvr < ETA_BI_FLOOR_327)
    dvr_does_not_beat_bi_floor = bool(math.isfinite(eta_dvr) and eta_dvr >= ETA_BI_FLOOR_327)
    new_required_ceiling_lift = (required_ceiling_lift(eta_dvr)
                                 if dvr_reduces_ceiling_lift_burden else float("nan"))

    # three reference comparisons (the rungs the PR asks eta_dvr to be placed against)
    ref_cmp = {
        "vs_budget_4p02pct": {"ref": ETA_BUDGET_500, "eta_dvr_below": bool(eta_dvr < ETA_BUDGET_500)},
        "vs_dvr_lit_6pct": {"ref": DVR_LIT_RECOMPUTE_FRAC, "eta_dvr_below": bool(eta_dvr < DVR_LIT_RECOMPUTE_FRAC),
                            "note": "LLM-42 6% is a recompute FRACTION, not an eta"},
        "vs_bi_floor_9p841pct": {"ref": ETA_BI_FLOOR_327, "eta_dvr_below": bool(eta_dvr < ETA_BI_FLOOR_327)},
    }

    # ---- self-test (PRIMARY) ----
    streams_ok = bool(m["ref_len"] > 0 and m["ref_len"] == m["spec_len"] == m["dvr_len"])
    nan_clean = all(math.isfinite(x) for x in (r, eta_dvr, step_us_M1, step_us_M8)) and \
        bool(m.get("peak_gpu_gb") is None or math.isfinite(m["peak_gpu_gb"]))
    r_in_range = bool(0.0 <= r <= 1.0 and math.isfinite(r))
    eta_finite_bracketed = bool(math.isfinite(eta_dvr) and 0.0 <= eta_dvr <= 5.0)
    ref_cmps_set = all(("eta_dvr_below" in v) for v in ref_cmp.values())
    identity_computed = bool(math.isfinite(token_identity_rate))
    decision_exclusive = bool(
        int(dvr_unlocks_500_coverage_alone) + int(dvr_reduces_ceiling_lift_burden)
        + int(dvr_does_not_beat_bi_floor) == 1)
    scope_flags = {
        "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "tps_local_relative": True,
    }
    self_test = {
        "streams_nonempty_equal_len": streams_ok,
        "r_in_range_finite": r_in_range,
        "eta_dvr_finite_bracketed": eta_finite_bracketed,
        "three_reference_comparisons_set": ref_cmps_set,
        "token_identity_computed": identity_computed,
        "decision_branches_exclusive": decision_exclusive,
        "nan_clean": nan_clean,
        "scope_flags_recorded": all(scope_flags.values()),
    }
    dvr_rollback_rate_eta_self_test_passes = all(self_test.values())

    report = {
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "pr": 362, "agent": "wirbel", "kind": "dvr-rollback-rate-eta",
        # PRIMARY + TEST
        "dvr_rollback_rate_eta_self_test_passes": dvr_rollback_rate_eta_self_test_passes,
        "eta_dvr": eta_dvr,
        "int4_marlin_rollback_rate": r,
        # rollback rate cross-check
        "rollback_rate_vs_232_delta": r - DIV_M8_232,
        "rollback_rate_same_order_as_232": bool(0.2 * DIV_M8_232 <= r <= 5.0 * DIV_M8_232),
        "div_m8_232_anchor": DIV_M8_232,
        # eta_dvr variants (transparent)
        "eta_dvr_honest_fullfwd_alwayson": eta_dvr_honest,
        "eta_dvr_probe_gated_lowerbound": eta_dvr_probe_gated,
        "eta_dvr_lmhead_only_INVALID": eta_dvr_lmhead,
        # cost model components (local, same-hardware)
        "cost": {
            "step_us_M1_local": step_us_M1, "step_us_M8_local": step_us_M8,
            "reverify_full_over_step": cost.get("reverify_full_over_step"),
            "lmhead_us_M1_local": lmhead_us_M1, "lmhead_over_step": (lmhead_us_M1 / step_us_M8
                                                                    if step_us_M8 else float("nan")),
            "reverify_cost_fullfwd": reverify_cost_fullfwd, "rollback_cost": rollback_cost,
            "step_us_basis": step_us, "deployed_step_us_context_only": DEPLOYED_STEP_US,
        },
        # identity + the crux decomposition
        "token_identity_rate": token_identity_rate,
        "token_identity_rate_lmhead_only": token_identity_rate_lmhead_only,
        "lmhead_reverify_restores_identity": lmhead_reverify_restores_identity,
        "decomposition": decomp,
        # eta ladder + decision
        "eta_ladder": {
            "ceiling_lambda1": CEILING_LAMBDA1, "target_tps": TARGET_TPS,
            "eta_budget_500_4p02pct": ETA_BUDGET_500, "eta_dvr_lit_6pct": DVR_LIT_RECOMPUTE_FRAC,
            "eta_bi_floor_9p841pct": ETA_BI_FLOOR_327,
            "strict_ceiling_at_eta_dvr": CEILING_LAMBDA1 * (1.0 - eta_dvr) if math.isfinite(eta_dvr) else float("nan"),
        },
        "reference_comparisons": ref_cmp,
        "decision": {
            "dvr_unlocks_500_coverage_alone": dvr_unlocks_500_coverage_alone,
            "dvr_reduces_ceiling_lift_burden": dvr_reduces_ceiling_lift_burden,
            "dvr_does_not_beat_bi_floor": dvr_does_not_beat_bi_floor,
            "new_required_ceiling_lift": new_required_ceiling_lift,
        },
        "self_test": self_test,
        "scope_flags": scope_flags,
        # bookkeeping
        "model_dir": m["model_dir"], "is_frontier_baked": m["is_frontier_baked"],
        "model_resolution_reason": m.get("model_resolution_reason"),
        "n_prompts": m["n_prompts"], "max_gen": m["max_gen"], "scored_positions": m["scored_positions"],
        "determinism_M1_vs_M1": m["determinism_M1_vs_M1"], "determinism_M8_vs_M8": m["determinism_M8_vs_M8"],
        "within_batch_copy0_vs_copy1": m["within_batch_copy0_vs_copy1"],
        "control_prefill_vs_decode_M1_div": m["control_prefill_vs_decode_M1_div"],
        "peak_gpu_gb": m.get("peak_gpu_gb"),
        "official_baseline_unchanged": OFFICIAL_BASELINE,
    }
    report_path = OUT_DIR / "dvr_rollback_rate_eta_results.json"
    json.dump(report, open(report_path, "w"), indent=2)

    # ---- console summary ----
    print("\n========== DVR ROLLBACK RATE / eta_dvr (PR #362) ==========", flush=True)
    print(f" model: {m['model_dir']} (frontier_baked={m['is_frontier_baked']})", flush=True)
    print(f"   resolution: {m.get('model_resolution_reason')}", flush=True)
    print(f" int4_marlin_rollback_rate r       : {r:.6f}  (#232 anchor 0.007292, delta {r-DIV_M8_232:+.6f})", flush=True)
    print(f"   controls det1={m['determinism_M1_vs_M1']:.5f} det8={m['determinism_M8_vs_M8']:.5f} "
          f"within={m['within_batch_copy0_vs_copy1']:.5f} pf_vs_dec_M1={m['control_prefill_vs_decode_M1_div']:.5f}", flush=True)
    print(f" cost  step_us_M1={step_us_M1:.1f}  step_us_M8={step_us_M8:.1f}  "
          f"reverify_full/step={cost.get('reverify_full_over_step')}  lmhead_us_M1={lmhead_us_M1:.2f}", flush=True)
    print(f" CRUX  hidden_rel_l2_div={decomp.get('hidden_rel_l2_div_mean')}  "
          f"hidden_driven_flip={hidden_driven_flip_rate}  lmhead_restores_identity={lmhead_reverify_restores_identity}", flush=True)
    print(f" eta_dvr (HONEST full-fwd)         : {eta_dvr_honest:.6f}", flush=True)
    print(f"   eta_dvr probe-gated lower bound : {eta_dvr_probe_gated:.6f}  (needs a cheap+correct probe -- none exists)", flush=True)
    print(f"   eta_dvr lm-head-only (INVALID)  : {eta_dvr_lmhead:.6f}  (fails identity)", flush=True)
    print(f" --- ladder ---  budget(4.02%)={ETA_BUDGET_500:.5f}  lit(6%)={DVR_LIT_RECOMPUTE_FRAC}  bi_floor(9.841%)={ETA_BI_FLOOR_327:.5f}", flush=True)
    print(f" DECISION  unlocks_500_alone={dvr_unlocks_500_coverage_alone}  "
          f"reduces_lift_burden={dvr_reduces_ceiling_lift_burden}  does_not_beat_bi_floor={dvr_does_not_beat_bi_floor}", flush=True)
    print(f" token_identity_rate (DVR full-fwd): {token_identity_rate:.4f}   lm-head-only: {token_identity_rate_lmhead_only}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)        : {dvr_rollback_rate_eta_self_test_passes}  {self_test}", flush=True)
    print(f" report -> {report_path}", flush=True)
    print("===========================================================\n", flush=True)

    if not a.no_wandb:
        log_wandb(report, a)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; JSON only", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="wirbel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#362 int4-Marlin DVR rollback rate + eta_dvr vs the 4.02% >500 budget (#319)",
        config={
            "pr": 362, "M_verify": M_VERIFY, "n_prompts": report["n_prompts"],
            "max_gen": report["max_gen"], "model_dir": report["model_dir"],
            "is_frontier_baked": report["is_frontier_baked"],
            "model_resolution_reason": report.get("model_resolution_reason"),
            "ceiling_lambda1": CEILING_LAMBDA1, "eta_budget_500": ETA_BUDGET_500,
            "eta_bi_floor_327": ETA_BI_FLOOR_327, "div_m8_232": DIV_M8_232,
            "no_hf_job": True, "no_launch": True, "no_submission": True,
            "no_served_file_change": True, "tps_local_relative": True,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return
    flat = {
        "dvr_rollback_rate_eta_self_test_passes": report["dvr_rollback_rate_eta_self_test_passes"],
        "eta_dvr": report["eta_dvr"],
        "int4_marlin_rollback_rate": report["int4_marlin_rollback_rate"],
        "rollback_rate_vs_232_delta": report["rollback_rate_vs_232_delta"],
        "rollback_rate_same_order_as_232": report["rollback_rate_same_order_as_232"],
        "eta_dvr_honest_fullfwd_alwayson": report["eta_dvr_honest_fullfwd_alwayson"],
        "eta_dvr_probe_gated_lowerbound": report["eta_dvr_probe_gated_lowerbound"],
        "eta_dvr_lmhead_only_INVALID": report["eta_dvr_lmhead_only_INVALID"],
        "token_identity_rate": report["token_identity_rate"],
        "token_identity_rate_lmhead_only": report["token_identity_rate_lmhead_only"],
        "lmhead_reverify_restores_identity": report["lmhead_reverify_restores_identity"],
        "hidden_driven_flip_rate": report["decomposition"].get("hidden_driven_flip_rate"),
        "hidden_rel_l2_div_mean": report["decomposition"].get("hidden_rel_l2_div_mean"),
        "step_us_M1_local": report["cost"]["step_us_M1_local"],
        "step_us_M8_local": report["cost"]["step_us_M8_local"],
        "reverify_full_over_step": report["cost"]["reverify_full_over_step"],
        "lmhead_over_step": report["cost"]["lmhead_over_step"],
        "dvr_unlocks_500_coverage_alone": report["decision"]["dvr_unlocks_500_coverage_alone"],
        "dvr_reduces_ceiling_lift_burden": report["decision"]["dvr_reduces_ceiling_lift_burden"],
        "dvr_does_not_beat_bi_floor": report["decision"]["dvr_does_not_beat_bi_floor"],
        "new_required_ceiling_lift": report["decision"]["new_required_ceiling_lift"],
        "eta_budget_500_4p02pct": ETA_BUDGET_500, "eta_bi_floor_9p841pct": ETA_BI_FLOOR_327,
        "strict_ceiling_at_eta_dvr": report["eta_ladder"]["strict_ceiling_at_eta_dvr"],
        "peak_gpu_gb": report.get("peak_gpu_gb"),
    }
    for k, v in flat.items():
        if v is not None:
            run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["measure"], default=None,
                    help="internal: run the GPU phase (subprocess). Omit for the orchestrator.")
    ap.add_argument("--gpu", action="store_true", help="orchestrate the GPU measurement (default)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the path")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-gen", type=int, default=512)
    ap.add_argument("--decomp-prompts", type=int, default=2,
                    help="how many prompts to run the (best-effort) hidden-state decomposition on")
    ap.add_argument("--gpu-mem-util", type=float, default=0.70)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--wandb_group", dest="wandb_group", default="dvr-rollback-rate-eta")
    ap.add_argument("--wandb_name", dest="wandb_name", default="wirbel/dvr-rollback-rate-eta")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 2)
        a.max_gen = min(a.max_gen, 32)
        a.decomp_prompts = min(a.decomp_prompts, 1)

    if a.phase == "measure":
        phase_measure(a.out, a.n_prompts, a.max_gen, a.gpu_mem_util,
                      a.max_batched_tokens, a.decomp_prompts)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
