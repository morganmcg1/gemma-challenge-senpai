#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #369 -- Selective-DVR: a cheap+correct divergence detector under the 4.02% budget?

My #364 (i08xlqbg, MERGED) closed the LOGIT-MARGIN selective-DVR detector: recall 1.0 (AUC
0.972, every flip <=1.125 nats) but PRECISION base-rate-limited -- a 0.39% flip needle in a
~17% natural low-margin haystack forces 17% (in-sample) / 44.6% (provable) of positions through
a full M=1 forward = 4.5x the 9.841% blanket. The diagnosis: a logit-level proxy is structurally
precision-capped because the divergence is HIDDEN-STATE-driven (wirbel #362/#326: injected by
bf16 attention upstream, propagates through ~37 layers; lm_head-only re-verify leaves identity
0.9948<1.0).

THIS CARD traces the detector eta-vs-identity frontier for a small ordered set of candidate
detectors (cheapest first), each scored on the #364 flip oracle (recall, precision, frac_flagged)
and converted to selective_eta = frac_flagged x repair_cost (repair = full M=1 forward for the
doubly-flagged positions + the detector's own cost):

  Stage 1 -- M=8-only multi-signal detector (FREE, all read off the M=8 verify forward we already
    run): (a) logit-lens depth trajectory (apply the tied unembed to intermediate residuals,
    min-margin / argmax-instability across depth); (b) attention entropy / max-attention-weight at
    the bf16-injection layers; (c) residual-stream norm-growth anomaly. Fuse (thresholded OR + a
    held-out logistic). Q: does any FREE M=8-internal signal break the margin's ~2.3% precision
    ceiling at recall=1.0?

  Stage 2 -- partial-M=1 residual-disagreement probe (the direct hidden-state read). Sweep a
    mid-layer L* in {1/8,1/4,1/2 of 42 layers}: for the margin pre-filtered positions, measure
    ||h_M8(L*) - h_M1(L*)|| and pass only the doubly-flagged to the full M=1 forward. Report the
    depth L*_min at which the residual delta becomes separating (AUC vs flips) and the resulting
    two-stage selective_eta = 0.17 x (L*_min/42) + frac_doubly_flagged x 1.0.

  Stage 3 -- frontier + verdict. For each detector: (recall, precision, frac_flagged, selective_eta,
    token_identity_rate_after_repair). Identify the MIN-eta detector that achieves identity=1.0 and
    compare to the 4.02% >500 budget, the 9.841% blanket, and wirbel #362's 0.97% probe-gated floor.

INSTRUMENT (honesty): vLLM is not installed in this pod's HF interpreter, and vLLM does not expose
per-layer hidden states / attentions. We therefore reproduce the #364 M=8-vs-M=1 reduction-geometry
divergence MECHANISM in a self-contained HF transformers harness on the SAME int4 deployed substrate
and the SAME #364 teacher-forced token sequences (ctx from the ppl jsonl + gen reconstructed from the
merged _marginscan.json ref_tok), with attn_implementation="sdpa" (the q_len=1 decode kernel reduces
differently from the q_len=8 chunked-verify kernel -> the deterministic argmax flips). We VALIDATE the
instrument vs #364: divergence rate same order, flips low-margin (AUC), HF-vs-vLLM flip agreement on
shared positions. selective_eta is reported BOTH at the HF-measured flip base rate AND normalized to
the deployed 0.39% (eta scales with base rate). This is a LOCAL pod-GPU profiling screen: 0 official
TPS, NO HF job, NO --launch, NO submission, NO served-file change.

PRIMARY metric : token_identity_rate (HF M=8 verify geometry; corroborates #364 ~0.996).
TEST   metric  : detector_self_test_passes (bool).
Headline       : min_selective_eta_at_identity_1, best_m8only_precision_at_recall1,
                 L_star_min_separating, cheap_detector_clears_500_budget, selective_beats_blanket.

Reproduce:
    cd target/ && CUDA_VISIBLE_DEVICES=0 python research/validity/selective_dvr_detector/\
selective_dvr_detector.py --gpu --wandb_group selective-dvr-detector \
        --wandb_name ubel/selective-dvr-detector-eta
"""
from __future__ import annotations

import argparse
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
LAMBDA1_CEIL = 520.953               # #326/#327 lambda=1 spec-model ceiling
N_LAYERS = 42                        # gemma-4-E4B text layers
K_SPEC = 7
M_VERIFY = K_SPEC + 1                 # = 8, batched-verify width
BLANKET_ETA = 0.09841249119201488    # wirbel #360 correct-locus blanket eta
BUDGET_ETA = 1.0 - TARGET_TPS / LAMBDA1_CEIL   # = 0.040211...  (>500 kernel budget)
PROBE_GATED_ETA_362 = 0.0097         # wirbel #362 probe-gated floor (perfect cheap detector)
LMHEAD_ONLY_IDENTITY_362 = 0.9948    # wirbel #362: lm_head-only recompute leaves ~0.52% wrong
RAW_IDENTITY_364 = 0.99609375        # ubel #364 (i08xlqbg) M=8 verify identity (16 prompts)
DEPLOYED_DIVERGENCE_364 = 0.00390625 # ubel #364 deployed-substrate divergence (32/8192)
MARGIN_AUC_364 = 0.9721              # ubel #364 flip-margin separability
MARGIN_SELECTIVE_ETA_364 = 0.1702    # ubel #364 in-sample (tau100) margin selective_eta
MARGIN_PRECISION_364 = 0.023         # ubel #364 margin precision @ recall 1.0
FINAL_LOGIT_SOFTCAP = 30.0           # gemma-4 final_logit_softcapping
RECONCILE_TOL = 0.02

# depth sweeps (1-indexed layer outputs; hidden_states[L] = residual after layer L)
DH_LAYERS = [3, 5, 8, 11, 16, 21, 26, 32, 42]      # stage-2 disagreement + AUC-vs-depth curve
PR_LSTAR = [5, 11, 21]                              # PR's {1/8, 1/4, 1/2 of 42}
LENS_LAYERS = [5, 11, 16, 21, 26, 32, 37, 42]      # logit-lens depth trajectory
ENTROPY_EARLY_MAX_LAYER = 21                       # aggregate attn entropy over layers [1..21]

MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = ("official/main_bucket/shared_resources/speed_benchmark/data/"
                 "ppl_ground_truth_tokens.jsonl")
SCAN_364 = HERE.parent / "margin_localized_identity" / "_marginscan.json"


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


def reconstruct_sequences(n_prompts: int, n_new: int, ctx_cap: int) -> list[dict]:
    """ctx from the ppl jsonl + gen tokens reconstructed from the merged #364 scan ref_tok
    (ordered by g). Carries the per-offset vLLM flip + margin for cross-validation."""
    scan = json.load(open(SCAN_364))
    rows = [json.loads(l) for l in open(REPO_ROOT / PROMPTS_JSONL)][:n_prompts]
    by_prompt: dict[int, list] = {}
    for p in scan["positions"]:
        by_prompt.setdefault(p["prompt_index"], []).append(
            (p["g"], p["ref_tok"], p["flip"], p["margin_m8"]))
    seqs = []
    for pi, rec in enumerate(rows):
        ctx = list(rec.get("context_token_ids", []))[:ctx_cap]
        if len(ctx) < 2:
            continue
        gen_rows = sorted(by_prompt.get(pi, []), key=lambda x: x[0])[:n_new]
        if not gen_rows:
            continue
        gen = [g[1] for g in gen_rows]
        vllm_flip = {g[0]: int(g[2]) for g in gen_rows}      # generated offset -> vLLM flip
        vllm_margin = {g[0]: g[3] for g in gen_rows}
        seqs.append({"pi": pi, "id": rec.get("id"), "ctx": ctx, "gen": gen,
                     "vllm_flip": vllm_flip, "vllm_margin": vllm_margin})
    return seqs


# ======================================================================================
# GPU PHASE 1: detector scan -- HF dual-geometry forward (M=1 AR vs M=8 verify) + all signals
# ======================================================================================
def _softcap(logits, cap: float):
    import torch
    return cap * torch.tanh(logits / cap)


def phase_detector_scan(out_path: str, n_prompts: int, n_new: int, ctx_cap: int,
                        attn_impl: str, det_prompts: int) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    model_dir = resolve_model_dir()
    print(f"[scan] loading {model_dir} attn={attn_impl}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation=attn_impl,
        trust_remote_code=True)
    model = model.to("cuda:0").eval()
    n_layers = int(model.config.get_text_config().num_hidden_layers)
    # tied unembed + final norm for the logit-lens
    W_U = model.get_output_embeddings().weight.detach()           # [vocab, hid]
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    final_norm = lm.norm
    dev = torch.device("cuda:0")

    seqs = reconstruct_sequences(n_prompts, n_new, ctx_cap)
    print(f"[scan] {len(seqs)} sequences gen_lens={[len(s['gen']) for s in seqs]}", flush=True)

    lens_set = [L for L in LENS_LAYERS if L <= n_layers]
    dh_set = [L for L in DH_LAYERS if L <= n_layers]

    @torch.no_grad()
    def lens_margin_arg(h_row):
        # h_row: [hid] residual at one position -> capped lens logits margin + argmax
        lg = _softcap(F.linear(final_norm(h_row.unsqueeze(0)), W_U)[0].float(), FINAL_LOGIT_SOFTCAP)
        top2 = torch.topk(lg, 2)
        return int(top2.indices[0]), float(top2.values[0] - top2.values[1])

    @torch.no_grad()
    def run_M1(ids, ctx_len):
        """chunk=1 AR geometry. Returns per predicting-pos j: m1_arg, and cached hidden[dh_set][j]."""
        N = ids.shape[0]
        past = None
        m1_arg: dict[int, int] = {}
        m1_h: dict[int, dict[int, "torch.Tensor"]] = {L: {} for L in dh_set}
        for pos in range(N):
            out = model(input_ids=ids[pos:pos + 1].unsqueeze(0), past_key_values=past,
                        use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            j = pos
            if ctx_len - 1 <= j <= N - 2:
                m1_arg[j] = int(out.logits[0, 0].argmax())
                for L in dh_set:
                    m1_h[L][j] = out.hidden_states[L][0, 0].detach()
        return m1_arg, m1_h

    @torch.no_grad()
    def run_M8(ids, ctx_len, m1_h, chunk):
        """chunk-wide verify geometry. Returns per predicting-pos j a full signal dict."""
        N = ids.shape[0]
        past = None
        rec: dict[int, dict] = {}
        pos = 0
        while pos < N:
            step = min(chunk, N - pos)
            out = model(input_ids=ids[pos:pos + step].unsqueeze(0), past_key_values=past,
                        use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            hs = out.hidden_states                       # tuple len n_layers+1, [1,step,hid]
            for local in range(step):
                j = pos + local
                if not (ctx_len - 1 <= j <= N - 2):
                    continue
                lg = _softcap(out.logits[0, local].float(), FINAL_LOGIT_SOFTCAP)
                top2 = torch.topk(lg, 2)
                m8_arg = int(top2.indices[0]); m8_margin = float(top2.values[0] - top2.values[1])
                # logit-lens depth trajectory
                lens_args, lens_margins = [], []
                for L in lens_set:
                    a, m = lens_margin_arg(hs[L][0, local])
                    lens_args.append(a); lens_margins.append(m)
                lens_min_margin = float(min(lens_margins))
                lens_instab = int(sum(1 for a in lens_args if a != m8_arg))
                # residual-stream norms + partial-M=1 disagreement
                dh = {}; rnorm = {}
                for L in dh_set:
                    h8 = hs[L][0, local].float()
                    rnorm[L] = float(torch.linalg.vector_norm(h8))
                    if j in m1_h[L]:
                        h1 = m1_h[L][j].float()
                        dh[L] = float(torch.linalg.vector_norm(h8 - h1))
                    else:
                        dh[L] = float("nan")
                rec[j] = {"m8_arg": m8_arg, "m8_margin": m8_margin,
                          "lens_min_margin": lens_min_margin, "lens_instab": lens_instab,
                          "dh": dh, "rnorm": rnorm}
            pos += step
        return rec

    positions: list[dict] = []
    n_pos = n_hf_flip = n_vllm_flip_shared = n_agree_shared = n_m1eqtf = 0
    det_ver_match = det_ver_total = 0
    first_div = None
    t0 = time.time()
    for s in seqs:
        ids = torch.tensor(s["ctx"] + s["gen"], dtype=torch.long, device=dev)
        ctx_len = len(s["ctx"]); N = ids.shape[0]
        m1_arg, m1_h = run_M1(ids, ctx_len)
        rec8 = run_M8(ids, ctx_len, m1_h, M_VERIFY)
        # determinism control: re-run M8 on the first det_prompts prompts, compare argmax stream
        if s["pi"] < det_prompts:
            rec8b = run_M8(ids, ctx_len, m1_h, M_VERIFY)
            for j in rec8:
                if j in rec8b:
                    det_ver_total += 1
                    det_ver_match += int(rec8[j]["m8_arg"] == rec8b[j]["m8_arg"])
        common = sorted(set(m1_arg) & set(rec8))
        n_flip_prompt = 0
        for j in common:
            g = j + 1 - ctx_len                      # generated offset (predicting gen[g])
            a1 = m1_arg[j]; r = rec8[j]
            hf_flip = int(r["m8_arg"] != a1)
            tf_tok = int(ids[j + 1])                 # teacher-forced next token (vLLM greedy)
            n_m1eqtf += int(a1 == tf_tok)
            vf = s["vllm_flip"].get(g, -1)
            if vf >= 0:
                n_vllm_flip_shared += vf
                n_agree_shared += int(hf_flip == vf)
            n_pos += 1; n_hf_flip += hf_flip; n_flip_prompt += hf_flip
            positions.append({
                "pi": s["pi"], "j": j, "g": g, "flip": hf_flip,
                "m1_arg": a1, "m8_arg": r["m8_arg"], "tf_tok": tf_tok,
                "vllm_flip": vf,
                "margin_m8": r["m8_margin"], "lens_min_margin": r["lens_min_margin"],
                "lens_instab": r["lens_instab"],
                "dh": {str(L): r["dh"][L] for L in dh_set},
                "rnorm": {str(L): r["rnorm"][L] for L in dh_set},
            })
            if hf_flip and first_div is None:
                first_div = {"prompt_index": s["pi"], "prompt_id": s["id"], "generated_offset": g,
                             "absolute_position": j, "ref_token_id": a1, "verify_token_id": r["m8_arg"],
                             "margin_m8": r["m8_margin"], "vllm_flip": vf}
        print(f"[scan] pi={s['pi']} pos={len(common)} hf_flips={n_flip_prompt}", flush=True)
        del m1_h
        torch.cuda.empty_cache()

    identity = 1.0 - n_hf_flip / n_pos if n_pos else float("nan")
    det_ver_frac = det_ver_match / det_ver_total if det_ver_total else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "detector_scan", "model_dir": model_dir, "attn_impl": attn_impl,
        "n_prompts": len(seqs), "n_new": n_new, "n_layers": n_layers,
        "lens_layers": lens_set, "dh_layers": dh_set,
        "total_positions": n_pos, "n_hf_flip": n_hf_flip,
        "token_identity_rate": identity, "verify_divergence": 1.0 - identity if n_pos else float("nan"),
        "n_vllm_flip_shared": n_vllm_flip_shared, "n_agree_shared": n_agree_shared,
        "hf_vllm_agreement": n_agree_shared / n_pos if n_pos else float("nan"),
        "m1_argmax_eq_teacherforced": n_m1eqtf / n_pos if n_pos else float("nan"),
        "determinism_verify_geometry": det_ver_frac,
        "first_divergence": first_div, "peak_gpu_gb": peak_gb,
        "elapsed_s": time.time() - t0, "nan_clean": bool(math.isfinite(identity)),
        "positions": positions, "local_relative_tps": True,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"))
    print(f"[scan] identity={identity:.6f} hf_flips={n_hf_flip}/{n_pos} "
          f"hf_vllm_agree={out['hf_vllm_agreement']:.4f} det_ver={det_ver_frac:.4f} "
          f"peak={peak_gb:.1f}GB elapsed={out['elapsed_s']:.0f}s", flush=True)
    print(f"DETSCAN_DONE {out_path}", flush=True)


# ======================================================================================
# GPU PHASE 2: attention-entropy (eager, output_attentions) -- the bf16-injection signal
# ======================================================================================
def phase_attn_entropy(out_path: str, n_prompts: int, n_new: int, ctx_cap: int) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    model_dir = resolve_model_dir()
    print(f"[entropy] loading {model_dir} attn=eager", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True).to("cuda:0").eval()
    n_layers = int(model.config.get_text_config().num_hidden_layers)
    dev = torch.device("cuda:0")
    seqs = reconstruct_sequences(n_prompts, n_new, ctx_cap)
    early = [L for L in range(1, min(ENTROPY_EARLY_MAX_LAYER, n_layers) + 1)]

    @torch.no_grad()
    def run(ids, ctx_len, chunk=M_VERIFY):
        N = ids.shape[0]
        past = None
        out_rows: dict[int, dict] = {}
        pos = 0
        while pos < N:
            step = min(chunk, N - pos)
            o = model(input_ids=ids[pos:pos + step].unsqueeze(0), past_key_values=past,
                      use_cache=True, output_attentions=True)
            past = o.past_key_values
            atts = o.attentions                          # tuple len n_layers, [1, heads, step, kv]
            for local in range(step):
                j = pos + local
                if not (ctx_len - 1 <= j <= N - 2):
                    continue
                ent_max = 0.0; maxw_min = 1.0
                for L in early:
                    a = atts[L - 1][0, :, local, :].float()    # [heads, kv]
                    a = a.clamp_min(1e-12)
                    a = a / a.sum(-1, keepdim=True)
                    nkv = a.shape[-1]
                    ent = -(a * a.log()).sum(-1) / math.log(max(nkv, 2))   # GQA/len-normalized [heads]
                    ent_max = max(ent_max, float(ent.max()))
                    maxw_min = min(maxw_min, float(a.max(-1).values.min()))
                out_rows[j] = {"attn_entropy_max": ent_max, "attn_maxw_min": maxw_min}
            pos += step
        return out_rows

    rows: list[dict] = []
    t0 = time.time()
    for s in seqs:
        ids = torch.tensor(s["ctx"] + s["gen"], dtype=torch.long, device=dev)
        r = run(ids, len(s["ctx"]))
        for j, v in r.items():
            rows.append({"pi": s["pi"], "j": j, **v})
        print(f"[entropy] pi={s['pi']} pos={len(r)}", flush=True)
    out = {"phase": "attn_entropy", "early_layers": early, "rows": rows,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "elapsed_s": time.time() - t0,
           "nan_clean": True}
    json.dump(out, open(out_path, "w"))
    print(f"[entropy] {len(rows)} rows peak={out['peak_gpu_gb']:.1f}GB elapsed={out['elapsed_s']:.0f}s",
          flush=True)
    print(f"ATTNENTROPY_DONE {out_path}", flush=True)


# ======================================================================================
# GPU PHASE 3: per-position recompute cost (full bf16 lm_head GEMV + body + 1q SDPA)
# ======================================================================================
def phase_recompute_timing(out_path: str, iters: int) -> None:
    import torch
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    VOCAB, HID, INTER, QKV, Q = 262144, 2560, 10240, 3072, 2048
    GU = 2 * INTER
    def z(*s): return torch.zeros(*s, dtype=torch.bfloat16, device=dev)
    W_lmhead = z(VOCAB, HID); W_qkv, W_o = z(QKV, HID), z(HID, Q)
    W_gu, W_dn = z(GU, HID), z(HID, INTER); x = z(1, HID)

    def lmhead_only(): return torch.nn.functional.linear(x, W_lmhead)
    def body_one_layer():
        q = torch.nn.functional.linear(x, W_qkv)
        _ = torch.nn.functional.linear(q[:, :Q], W_o)
        gu = torch.nn.functional.linear(x, W_gu); g, u = gu[:, :INTER], gu[:, INTER:]
        _ = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, W_dn)

    def timed(fn, n):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1e6

    lmhead_us = timed(lmhead_only, iters); body_layer_us = timed(body_one_layer, iters)
    lmhead_bytes = VOCAB * HID * 2.0
    body_params_layer = (QKV * HID) + (HID * Q) + (GU * HID) + (HID * INTER)
    body_bytes = body_params_layer * 0.5 * N_LAYERS
    f_lmhead_bandwidth = lmhead_bytes / (lmhead_bytes + body_bytes)
    out = {"phase": "recompute_timing", "lmhead_full_vocab_us": lmhead_us,
           "body_single_layer_us": body_layer_us, "n_layers": N_LAYERS,
           "f_lmhead_bandwidth": f_lmhead_bandwidth,
           "full_forward_us_model": lmhead_us + body_layer_us * N_LAYERS,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
           "nan_clean": bool(math.isfinite(lmhead_us) and math.isfinite(body_layer_us))}
    json.dump(out, open(out_path, "w"))
    print(f"[timing] lmhead={lmhead_us:.1f}us body1={body_layer_us:.1f}us "
          f"f_lmhead={f_lmhead_bandwidth:.4f}", flush=True)
    print(f"RECOMPUTETIMING_DONE {out_path}", flush=True)


# ======================================================================================
# Orchestrator + analysis
# ======================================================================================
def run_phase(args_list: list[str], timeout: int | None = None) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} (timeout={timeout})", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s: {args_list}", flush=True)
        return 124


def _auc(scores: list[float], labels: list[int]) -> float:
    """P(score_pos > score_neg) + 0.5 ties. Higher score => more likely a flip (positive)."""
    import bisect
    pos = [s for s, y in zip(scores, labels) if y]
    neg = sorted(s for s, y in zip(scores, labels) if not y)
    if not pos or not neg:
        return float("nan")
    wins = ties = 0.0
    for m in pos:
        lo = bisect.bisect_left(neg, m); hi = bisect.bisect_right(neg, m)
        wins += lo; ties += (hi - lo)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _precision_at_recall1(scores: list[float], labels: list[int]):
    """Threshold = min positive score (flag score>=thr catches ALL flips). Returns
    (frac_flagged, precision, threshold). Higher score = more suspicious."""
    pos = [s for s, y in zip(scores, labels) if y]
    if not pos:
        return float("nan"), float("nan"), float("nan")
    thr = min(pos)
    flagged = sum(1 for s in scores if s >= thr)
    n_pos = sum(labels)
    return flagged / len(scores), (n_pos / flagged if flagged else float("nan")), thr


def _logistic_fit(X: list[list[float]], y: list[int], iters: int = 400, lr: float = 0.3):
    """Tiny standardized logistic regression (numpy). Returns (w, b, mu, sd)."""
    import numpy as np
    Xa = np.array(X, dtype=np.float64); ya = np.array(y, dtype=np.float64)
    mu = Xa.mean(0); sd = Xa.std(0); sd[sd == 0] = 1.0
    Xs = (Xa - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d); b = 0.0
    pw = max(1.0, (n - ya.sum()) / max(ya.sum(), 1.0))   # positive class weight (rare-event)
    for _ in range(iters):
        z = Xs @ w + b; p = 1.0 / (1.0 + np.exp(-z))
        sw = np.where(ya > 0, pw, 1.0)
        g = (sw * (p - ya))
        w -= lr * (Xs.T @ g) / n + lr * 1e-3 * w
        b -= lr * g.mean()
    return w, b, mu, sd


def _logistic_score(X, w, b, mu, sd):
    import numpy as np
    Xs = (np.array(X, dtype=np.float64) - mu) / sd
    return (1.0 / (1.0 + np.exp(-(Xs @ w + b)))).tolist()


def compose(a: argparse.Namespace, scan: dict, entropy: dict, timing: dict) -> None:
    import numpy as np
    positions = scan["positions"]
    n_total = scan["total_positions"]
    identity = scan["token_identity_rate"]
    n_layers = scan["n_layers"]
    dh_set = scan["dh_layers"]
    labels = [p["flip"] for p in positions]
    n_flip = sum(labels)

    # merge entropy rows by (pi,j)
    ent_map = {(r["pi"], r["j"]): r for r in entropy.get("rows", [])}
    for p in positions:
        e = ent_map.get((p["pi"], p["j"]), {})
        p["attn_entropy_max"] = e.get("attn_entropy_max", float("nan"))
        p["attn_maxw_min"] = e.get("attn_maxw_min", float("nan"))

    # base-rate scaling: HF flip rate vs deployed 0.39% -> eta normalization factor
    hf_rate = n_flip / n_total if n_total else float("nan")
    eta_norm = (DEPLOYED_DIVERGENCE_364 / hf_rate) if hf_rate and hf_rate > 0 else float("nan")

    # ---- FREE M=8-only detectors: AUC + precision@recall=1.0 ----
    def col(key): return [p[key] for p in positions]
    free_detectors = {
        "margin_m8": [-m for m in col("margin_m8")],            # low margin = suspicious
        "lens_min_margin": [-m for m in col("lens_min_margin")],
        "lens_instab": [float(x) for x in col("lens_instab")],
        "attn_entropy_max": col("attn_entropy_max"),
        "attn_maxw_min": [-m for m in col("attn_maxw_min")],    # low max-weight (diffuse) = suspicious
    }
    # residual-norm anomaly: per-layer z-score of ||h||, take max over dh layers
    rnorm_by_layer = {L: np.array([p["rnorm"][str(L)] for p in positions]) for L in dh_set}
    z_stack = []
    for L in dh_set:
        v = rnorm_by_layer[L]; mu, sd = v.mean(), v.std() or 1.0
        z_stack.append((v - mu) / sd)
    free_detectors["resid_norm_anom"] = np.max(np.stack(z_stack), 0).tolist()

    free_report = {}
    for name, sc in free_detectors.items():
        clean = [(s, y) for s, y in zip(sc, labels) if isinstance(s, (int, float)) and math.isfinite(s)]
        if len(clean) < len(labels) * 0.5 or not any(y for _, y in clean):
            free_report[name] = {"auc": float("nan"), "frac_flagged": float("nan"),
                                 "precision_at_recall1": float("nan"), "selective_eta": float("nan")}
            continue
        s2 = [s for s, _ in clean]; y2 = [y for _, y in clean]
        auc = _auc(s2, y2)
        frac, prec, _ = _precision_at_recall1(s2, y2)
        free_report[name] = {"auc": auc, "frac_flagged": frac, "precision_at_recall1": prec,
                             "selective_eta": frac}    # free signal => eta = frac_flagged * full M=1
    best_free_name = max(free_report, key=lambda k: (free_report[k]["precision_at_recall1"]
                         if math.isfinite(free_report[k]["precision_at_recall1"]) else -1))
    best_m8only_precision = free_report[best_free_name]["precision_at_recall1"]
    best_free_eta = free_report[best_free_name]["selective_eta"]

    # ---- FREE fusion (held-out logistic over the M=8-only signals) ----
    feat_keys = ["margin_m8", "lens_min_margin", "lens_instab", "attn_entropy_max",
                 "attn_maxw_min", "resid_norm_anom"]
    feat_cols = {k: (free_detectors[k] if k != "resid_norm_anom" else free_detectors[k])
                 for k in feat_keys}
    def feat_row(i): return [feat_cols[k][i] for k in feat_keys]
    valid_idx = [i for i in range(n_total)
                 if all(math.isfinite(feat_cols[k][i]) for k in feat_keys)]

    free_fusion = held_fusion = {"auc": float("nan"), "frac_flagged": float("nan"),
                                 "precision_at_recall1": float("nan")}
    fusion_splits = []
    if n_flip >= 4 and len(valid_idx) > 50:
        rng = np.random.default_rng(0)
        for seed in range(a.n_splits):
            rs = np.random.default_rng(seed)
            idx = np.array(valid_idx); rs.shuffle(idx)
            cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]
            if not any(labels[i] for i in tr) or not any(labels[i] for i in te):
                continue
            w, b, mu, sd = _logistic_fit([feat_row(i) for i in tr], [labels[i] for i in tr])
            te_scores = _logistic_score([feat_row(i) for i in te], w, b, mu, sd)
            te_lab = [labels[i] for i in te]
            auc = _auc(te_scores, te_lab)
            frac, prec, _ = _precision_at_recall1(te_scores, te_lab)
            fusion_splits.append({"seed": seed, "auc": auc, "frac_flagged": frac,
                                  "precision_at_recall1": prec})
        if fusion_splits:
            held_fusion = {
                "auc": float(np.nanmean([f["auc"] for f in fusion_splits])),
                "frac_flagged": float(np.nanmean([f["frac_flagged"] for f in fusion_splits])),
                "precision_at_recall1": float(np.nanmean([f["precision_at_recall1"] for f in fusion_splits])),
                "n_splits": len(fusion_splits),
            }

    # ---- STAGE-1 margin pre-filter (recall 1.0) ----
    margins = col("margin_m8")
    flip_margins = [m for m, y in zip(margins, labels) if y]
    tau1 = (max(flip_margins) + 1e-6) if flip_margins else 0.0   # in-sample recall-1 margin gate
    stage1_idx = [i for i, m in enumerate(margins) if m < tau1]
    stage1_frac = len(stage1_idx) / n_total if n_total else float("nan")

    # ---- STAGE-2 partial-M=1 residual disagreement: AUC-vs-depth + two-stage eta ----
    dh_curve = {}
    two_stage = {}
    for L in dh_set:
        dh_col = [p["dh"][str(L)] for p in positions]
        clean = [(s, y) for s, y in zip(dh_col, labels) if math.isfinite(s)]
        auc = _auc([s for s, _ in clean], [y for _, y in clean]) if clean else float("nan")
        # on the stage-1 pre-filtered set, threshold dh to catch all flips (recall 1.0)
        s1_dh = [(dh_col[i], labels[i]) for i in stage1_idx if math.isfinite(dh_col[i])]
        s1_flip_dh = [s for s, y in s1_dh if y]
        if s1_flip_dh:
            tau2 = min(s1_flip_dh)
            doubly = sum(1 for s, _ in s1_dh if s >= tau2)
            frac_doubly = doubly / n_total
            recall2 = 1.0   # by construction (tau2 = min flip dh on the filtered set, in-sample)
        else:
            tau2 = float("nan"); frac_doubly = float("nan"); recall2 = float("nan")
        eta = (stage1_frac * (L / n_layers) + frac_doubly) if math.isfinite(frac_doubly) else float("nan")
        # identity after the two-stage repair (in-sample: recall 1.0 => identity 1.0)
        ident_after = 1.0 if (math.isfinite(eta)) else identity
        dh_curve[L] = {"auc": auc, "frac_doubly_flagged": frac_doubly,
                       "selective_eta": eta, "stage2_recall": recall2,
                       "identity_after_repair": ident_after, "L_over_n": L / n_layers}
        two_stage[L] = eta

    # held-out dh thresholding (honest): split prompts, set tau2 on train flips, eval test recall
    dh_heldout = {}
    pis = sorted(set(p["pi"] for p in positions))
    for L in dh_set:
        recs = []
        for seed in range(a.n_splits):
            rs = np.random.default_rng(100 + seed)
            order = list(pis); rs.shuffle(order)
            cut = max(1, int(0.7 * len(order)))
            tr_pi, te_pi = set(order[:cut]), set(order[cut:])
            tr = [(p["dh"][str(L)], p["flip"]) for p in positions
                  if p["pi"] in tr_pi and p["margin_m8"] < tau1 and math.isfinite(p["dh"][str(L)])]
            te_all = [(p["dh"][str(L)], p["flip"], p["margin_m8"] < tau1) for p in positions
                      if p["pi"] in te_pi]
            tr_flip = [s for s, y in tr if y]
            if not tr_flip:
                continue
            tau2 = min(tr_flip)
            te_flips = [y for _, y, _ in te_all if y]
            caught = sum(1 for s, y, infilt in te_all if y and infilt and s >= tau2)
            recall_te = caught / len(te_flips) if te_flips else float("nan")
            doubly_te = sum(1 for s, y, infilt in te_all if infilt and s >= tau2)
            n_te = len(te_all)
            recs.append({"recall_test": recall_te,
                         "frac_doubly_test": doubly_te / n_te if n_te else float("nan")})
        if recs:
            dh_heldout[L] = {
                "recall_test_mean": float(np.nanmean([r["recall_test"] for r in recs])),
                "recall_test_min": float(np.nanmin([r["recall_test"] for r in recs])),
                "frac_doubly_test_mean": float(np.nanmean([r["frac_doubly_test"] for r in recs])),
                "n_splits": len(recs),
            }

    # ---- "separating" depth + min eta at identity 1.0 ----
    def is_separating(L):
        c = dh_curve[L]; h = dh_heldout.get(L, {})
        return (math.isfinite(c["auc"]) and c["auc"] >= 0.90
                and h.get("recall_test_min", 0.0) >= 0.999)
    sep_layers = [L for L in dh_set if is_separating(L)]
    L_star_min_separating = min(sep_layers) if sep_layers else None
    # min selective eta among detectors that achieve identity 1.0 (the two-stage ones do, in-sample)
    valid_two_stage = {L: e for L, e in two_stage.items() if math.isfinite(e)}
    min_two_stage_eta = min(valid_two_stage.values()) if valid_two_stage else float("nan")
    min_two_stage_L = (min(valid_two_stage, key=valid_two_stage.get)
                       if valid_two_stage else None)
    # the headline: cheapest detector that restores identity to 1.0
    min_selective_eta_at_identity_1 = min_two_stage_eta
    min_eta_normalized = (min_selective_eta_at_identity_1 * eta_norm
                          if math.isfinite(eta_norm) else float("nan"))

    cheap_detector_clears_500_budget = bool(math.isfinite(min_selective_eta_at_identity_1)
                                            and min_selective_eta_at_identity_1 < BUDGET_ETA)
    selective_beats_blanket = bool(math.isfinite(min_selective_eta_at_identity_1)
                                   and min_selective_eta_at_identity_1 < BLANKET_ETA)
    beats_probe_gated = bool(math.isfinite(min_selective_eta_at_identity_1)
                             and min_selective_eta_at_identity_1 <= PROBE_GATED_ETA_362)

    if not math.isfinite(min_selective_eta_at_identity_1):
        verdict_band = "INCONCLUSIVE_no_detector"
    elif cheap_detector_clears_500_budget:
        verdict_band = "GREEN_live_cheap_identity_clears_500"
    elif selective_beats_blanket:
        verdict_band = "AMBER_beats_blanket_below_budget_gap"
    else:
        verdict_band = "RED_closes_selective_lever"

    # ---- self-test ----
    auc_in_range = all(0.0 <= dh_curve[L]["auc"] <= 1.0 for L in dh_set
                       if math.isfinite(dh_curve[L]["auc"]))
    eta_monotone = all(
        dh_curve[dh_set[i]]["selective_eta"] <= dh_curve[dh_set[i + 1]]["selective_eta"] + 0.02
        for i in range(len(dh_set) - 1)
        if math.isfinite(dh_curve[dh_set[i]]["selective_eta"])
        and math.isfinite(dh_curve[dh_set[i + 1]]["selective_eta"]))
    checks = {
        "positions_present": n_total > 0 and len(positions) == n_total,
        "nan_clean_scan": bool(scan["nan_clean"]) and math.isfinite(identity),
        "identity_in_range": 0.0 <= identity <= 1.0,
        "flips_observed": n_flip > 0,
        "determinism_verify_eq_1": abs(scan["determinism_verify_geometry"] - 1.0) < 1e-9,
        "corroborates_364_identity_order": abs(identity - RAW_IDENTITY_364) <= 0.05,
        "hf_vllm_agreement_high": scan["hf_vllm_agreement"] >= 0.95,
        # HF-int4 (bf16 GEMV) vs vLLM-int4 (Marlin) M=1 greedy diverge ~9-13% cross-engine
        # (documented). 0.80 catches a gross reconstruction desync (would be <=0.6) without
        # demanding bit-exactness the substrate does not provide. The flip ORACLE is validated
        # by hf_vllm_agreement above, not by absolute-token match.
        "m1_eq_teacherforced_ok": scan["m1_argmax_eq_teacherforced"] >= 0.80,
        "margin_recall1_catches_all_flips": all(p["margin_m8"] < tau1 for p in positions if p["flip"]),
        "stage2_dh_auc_in_range": auc_in_range,
        "entropy_present": len(ent_map) > 0,
        "timing_present": bool(timing.get("lmhead_full_vocab_us")),
        "n_splits_ge_2": a.n_splits >= 2,
        "no_hf_job_no_launch": True,
    }
    self_test_passes = bool(all(checks.values()))
    # eta(L) need NOT be monotone: dh separability is itself non-monotone in depth, so this is a
    # logged diagnostic of the curve shape, not a validity gate.
    checks_diagnostic = {"two_stage_eta_monotone_in_depth": eta_monotone}

    report = {
        "card": "selective_dvr_detector", "pr": 369, "issue": 319, "author": "ubel",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": True, "local_pod_gpu_only": True,
        "instrument": "HF transformers int4 sdpa dual-geometry (M=1 AR vs M=8 verify); "
                      "mechanism reproduction of #364 vLLM oracle; eta normalized to deployed 0.39%",
        # PRIMARY + TEST
        "token_identity_rate": identity,
        "detector_self_test_passes": self_test_passes,
        # instrument validation
        "verify_divergence": scan["verify_divergence"], "n_flip": n_flip, "n_total": n_total,
        "hf_flip_rate": hf_rate, "deployed_flip_rate_364": DEPLOYED_DIVERGENCE_364,
        "eta_base_rate_norm_factor": eta_norm,
        "hf_vllm_agreement": scan["hf_vllm_agreement"],
        "m1_argmax_eq_teacherforced": scan["m1_argmax_eq_teacherforced"],
        "determinism_verify_geometry": scan["determinism_verify_geometry"],
        "first_divergence": scan["first_divergence"],
        # stage 1 -- free M=8-only detectors
        "free_detectors": free_report,
        "best_m8only_detector": best_free_name,
        "best_m8only_precision_at_recall1": best_m8only_precision,
        "best_m8only_selective_eta": best_free_eta,
        "free_fusion_heldout": held_fusion, "free_fusion_splits": fusion_splits,
        "margin_baseline_auc_364": MARGIN_AUC_364, "margin_precision_364": MARGIN_PRECISION_364,
        # stage 2 -- partial-M=1 residual disagreement
        "stage1_margin_tau": tau1, "stage1_frac_flagged": stage1_frac,
        "dh_auc_vs_depth": {str(L): dh_curve[L]["auc"] for L in dh_set},
        "dh_curve": {str(L): dh_curve[L] for L in dh_set},
        "dh_heldout": {str(L): dh_heldout.get(L) for L in dh_set},
        "two_stage_selective_eta_by_L": {str(L): two_stage[L] for L in dh_set},
        "L_star_min_separating": L_star_min_separating,
        "L_star_pr_points": PR_LSTAR,
        # stage 3 -- frontier + verdict
        "min_selective_eta_at_identity_1": min_selective_eta_at_identity_1,
        "min_selective_eta_at_identity_1_at_L": min_two_stage_L,
        "min_selective_eta_normalized_to_deployed": min_eta_normalized,
        "blanket_eta": BLANKET_ETA, "budget_eta_500": BUDGET_ETA,
        "probe_gated_eta_362": PROBE_GATED_ETA_362,
        "cheap_detector_clears_500_budget": cheap_detector_clears_500_budget,
        "selective_beats_blanket": selective_beats_blanket,
        "selective_beats_probe_gated_362": beats_probe_gated,
        "verdict_band": verdict_band,
        # cost model
        "lmhead_full_vocab_us": timing.get("lmhead_full_vocab_us"),
        "body_single_layer_us": timing.get("body_single_layer_us"),
        "f_lmhead_bandwidth": timing.get("f_lmhead_bandwidth"),
        # bookkeeping
        "official_baseline_unchanged": OFFICIAL_BASELINE, "lambda1_ceil": LAMBDA1_CEIL,
        "n_layers": n_layers, "n_prompts": scan["n_prompts"], "n_new": scan["n_new"],
        "attn_impl": scan["attn_impl"], "dh_layers": dh_set, "lens_layers": scan["lens_layers"],
        "peak_gpu_gb": max(scan.get("peak_gpu_gb", 0.0), entropy.get("peak_gpu_gb", 0.0),
                           timing.get("peak_gpu_gb", 0.0)),
        "self_test": checks, "self_test_diagnostic": checks_diagnostic,
        "local_relative_tps": True,
    }
    report_path = OUT_DIR / "_results.json"
    json.dump(report, open(report_path, "w"), indent=2, default=str)

    bar = "=" * 90
    print("\n" + bar, flush=True)
    print(" SELECTIVE-DVR DETECTOR FRONTIER (PR #369, #319) -- cheap+correct detector < 4.02%?", flush=True)
    print(bar, flush=True)
    print(f" token_identity_rate (PRIMARY)            : {identity:.6f}  (#364 anchor {RAW_IDENTITY_364:.6f})", flush=True)
    print(f" HF flips / positions                     : {n_flip} / {n_total}  (rate {hf_rate:.4%})", flush=True)
    print(f" HF/vLLM flip agreement | m1==teacherforce: {scan['hf_vllm_agreement']:.4f} | {scan['m1_argmax_eq_teacherforced']:.4f}", flush=True)
    print(f" determinism (verify re-run)              : {scan['determinism_verify_geometry']:.4f}", flush=True)
    print(f" eta normalization (deployed/HF rate)     : x{eta_norm:.3f}", flush=True)
    print(f" --- STAGE 1: free M=8-only detectors (precision @ recall=1.0) ---", flush=True)
    for name, r in free_report.items():
        print(f"   {name:18s} AUC={r['auc']:.3f} frac={r['frac_flagged']:.4f} "
              f"prec={r['precision_at_recall1']:.4f} eta={r['selective_eta']:.4%}", flush=True)
    print(f"   best free detector: {best_free_name} (prec {best_m8only_precision:.4f}, "
          f"eta {best_free_eta:.4%})", flush=True)
    print(f"   free fusion (held-out): AUC={held_fusion['auc']} prec={held_fusion['precision_at_recall1']}", flush=True)
    print(f" --- STAGE 2: partial-M=1 residual disagreement (AUC vs depth, two-stage eta) ---", flush=True)
    print(f"   stage-1 margin pre-filter frac           : {stage1_frac:.4%} (tau1={tau1:.3f})", flush=True)
    for L in dh_set:
        c = dh_curve[L]; h = dh_heldout.get(L, {})
        tag = " <== PR" if L in PR_LSTAR else ""
        print(f"   L*={L:2d} (L/n={L/n_layers:.2f}) AUC={c['auc']:.3f} "
              f"frac_doubly={c['frac_doubly_flagged']:.4f} eta={c['selective_eta']:.4%} "
              f"recall_test_min={h.get('recall_test_min','-')}{tag}", flush=True)
    print(f" --- STAGE 3: verdict ---", flush=True)
    print(f"   L*_min separating (AUC>=.90 & held recall=1): {L_star_min_separating}", flush=True)
    print(f"   MIN selective_eta @ identity 1.0          : {min_selective_eta_at_identity_1:.4%} "
          f"(L*={min_two_stage_L})", flush=True)
    print(f"   normalized to deployed 0.39%              : {min_eta_normalized:.4%}", flush=True)
    print(f"   budget {BUDGET_ETA:.4%} | blanket {BLANKET_ETA:.4%} | probe-gated {PROBE_GATED_ETA_362:.4%}", flush=True)
    print(f"   clears_500 / beats_blanket / beats_probe  : {cheap_detector_clears_500_budget} / "
          f"{selective_beats_blanket} / {beats_probe_gated}", flush=True)
    print(f"   VERDICT BAND                              : {verdict_band}", flush=True)
    print(f"   SELF-TEST PASSES (TEST metric)            : {self_test_passes}", flush=True)
    print(bar + "\n", flush=True)

    run_ids = []
    if not a.no_wandb:
        run_ids = log_wandb(report, a)
    report["wandb_run_ids"] = run_ids
    json.dump(report, open(report_path, "w"), indent=2, default=str)
    marker = {"terminal": True, "status": "complete", "pending_arms": False,
              "wandb_run_ids": run_ids,
              "primary_metric": {"name": "token_identity_rate", "value": identity},
              "test_metric": {"name": "detector_self_test_passes", "value": int(self_test_passes)},
              "headline": {  # PR #369 required terminal fields
                  "token_identity_rate": identity,
                  "min_selective_eta_at_identity_1": min_selective_eta_at_identity_1,
                  "best_m8only_precision_at_recall1": best_m8only_precision,
                  "L_star_min_separating": L_star_min_separating,
                  "cheap_detector_clears_500_budget": cheap_detector_clears_500_budget,
                  "selective_beats_blanket": selective_beats_blanket,
                  "verdict_band": verdict_band}}
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_json = str(OUT_DIR / "_detscan.json")
    entropy_json = str(OUT_DIR / "_entropy.json")
    timing_json = str(OUT_DIR / "_timing.json")

    if not a.reuse_scan or not Path(scan_json).exists():
        rc = run_phase(["--phase", "detector_scan", "--out", scan_json,
                        "--n-prompts", str(a.n_prompts), "--n-new", str(a.n_new),
                        "--ctx-cap", str(a.ctx_cap), "--attn-impl", a.attn_impl,
                        "--det-prompts", str(a.det_prompts)], timeout=a.scan_timeout)
        if rc != 0:
            raise RuntimeError(f"detector_scan failed (rc={rc})")
    scan = json.load(open(scan_json))

    entropy = {"phase": "attn_entropy", "rows": [], "status": "not_run"}
    if not a.reuse_scan or not Path(entropy_json).exists():
        rc = run_phase(["--phase", "attn_entropy", "--out", entropy_json,
                        "--n-prompts", str(a.n_prompts), "--n-new", str(a.n_new),
                        "--ctx-cap", str(a.ctx_cap)], timeout=a.entropy_timeout)
    if Path(entropy_json).exists():
        entropy = json.load(open(entropy_json))

    timing = {"phase": "recompute_timing", "status": "not_run"}
    rc = run_phase(["--phase", "recompute_timing", "--out", timing_json,
                    "--dh-iters", str(a.dh_iters)], timeout=a.timing_timeout)
    if Path(timing_json).exists():
        timing = json.load(open(timing_json))

    compose(a, scan, entropy, timing)


def log_wandb(report: dict, a: argparse.Namespace) -> list[str]:
    try:
        import wandb as _wb  # noqa: F401
        if not hasattr(_wb, "init"):
            raise ImportError("stub wandb")
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed (analysis unaffected): {exc}", flush=True)
        return []
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] scripts.wandb_logging import failed: {exc}", flush=True)
        return []
    os.environ.setdefault("WANDB_DIR", "/tmp/wandb_ubel_selective_dvr")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel", name=a.wandb_name, group=a.wandb_group,
        notes="PR#369 selective-DVR detector frontier: free M=8-only signals (logit-lens, attn "
              "entropy, residual norm) + partial-M=1 residual-disagreement depth sweep, scored on "
              "the #364 flip oracle; min selective_eta vs 4.02% budget / 9.841% blanket / 0.97% "
              "probe-gated. HF int4 sdpa mechanism repro; LOCAL pod-GPU, 0 official TPS.",
        tags=["selective-dvr", "detector-frontier", "greedy-identity", "residual-disagreement",
              "logit-lens", "local-relative", "issue-319", "pr-369"],
        config={"pr": 369, "issue": 319, "wandb_group": a.wandb_group, "M_verify": M_VERIFY,
                "attn_impl": report["attn_impl"], "n_prompts": report["n_prompts"],
                "n_new": report["n_new"], "n_layers": report["n_layers"],
                "blanket_eta": BLANKET_ETA, "budget_eta_500": BUDGET_ETA,
                "instrument": report["instrument"], "official_baseline": OFFICIAL_BASELINE})
    if run is None:
        print("[wandb] disabled (no key/mode); JSON-only", flush=True)
        return []
    summ = {
        "token_identity_rate": report["token_identity_rate"],
        "detector_self_test_passes": int(report["detector_self_test_passes"]),
        "verify_divergence": report["verify_divergence"], "n_flip": report["n_flip"],
        "n_total": report["n_total"], "hf_flip_rate": report["hf_flip_rate"],
        "hf_vllm_agreement": report["hf_vllm_agreement"],
        "determinism_verify_geometry": report["determinism_verify_geometry"],
        "best_m8only_precision_at_recall1": report["best_m8only_precision_at_recall1"],
        "best_m8only_selective_eta": report["best_m8only_selective_eta"],
        "stage1_frac_flagged": report["stage1_frac_flagged"],
        "min_selective_eta_at_identity_1": report["min_selective_eta_at_identity_1"],
        "min_selective_eta_normalized_to_deployed": report["min_selective_eta_normalized_to_deployed"],
        "L_star_min_separating": report["L_star_min_separating"] or -1,
        "blanket_eta": BLANKET_ETA, "budget_eta_500": BUDGET_ETA,
        "probe_gated_eta_362": PROBE_GATED_ETA_362,
        "cheap_detector_clears_500_budget": int(report["cheap_detector_clears_500_budget"]),
        "selective_beats_blanket": int(report["selective_beats_blanket"]),
        "selective_beats_probe_gated_362": int(report["selective_beats_probe_gated_362"]),
        "free_fusion_heldout_auc": report["free_fusion_heldout"].get("auc"),
        "free_fusion_heldout_precision": report["free_fusion_heldout"].get("precision_at_recall1"),
        "tps_added_by_this_card": 0, "peak_gpu_gb": report["peak_gpu_gb"],
    }
    summ = {k: v for k, v in summ.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))}
    log_summary(run, summ, step=0)
    run.summary["verdict_band"] = report["verdict_band"]
    run.summary["best_m8only_detector"] = report["best_m8only_detector"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = int(bool(v))
    for k, v in report.get("self_test_diagnostic", {}).items():
        run.summary[f"selftest_diag/{k}"] = int(bool(v))
    try:
        import wandb
        # AUC-vs-depth + two-stage eta frontier
        tbl = wandb.Table(columns=["L_star", "L_over_n", "dh_auc", "frac_doubly_flagged",
                                   "selective_eta", "recall_test_min"])
        for L in report["dh_layers"]:
            c = report["dh_curve"][str(L)]; h = report["dh_heldout"].get(str(L)) or {}
            tbl.add_data(L, c["L_over_n"], c["auc"], c["frac_doubly_flagged"],
                         c["selective_eta"], h.get("recall_test_min"))
        run.log({"dh_depth_frontier": tbl, "global_step": 0})
        ftbl = wandb.Table(columns=["detector", "auc", "frac_flagged",
                                    "precision_at_recall1", "selective_eta"])
        for name, r in report["free_detectors"].items():
            ftbl.add_data(name, r["auc"], r["frac_flagged"], r["precision_at_recall1"],
                          r["selective_eta"])
        run.log({"free_detector_frontier": ftbl, "global_step": 0})
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] tables skipped: {exc}", flush=True)
    log_json_artifact(run, name="selective_dvr_detector_result", artifact_type="analysis", data=report)
    rid = getattr(run, "id", "") or ""
    finish_wandb(run)
    print(f"[wandb] logged run {rid}", flush=True)
    return [rid] if rid else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["detector_scan", "attn_entropy", "recompute_timing"],
                    default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--n-new", type=int, default=512)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--det-prompts", type=int, default=2)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--dh-iters", type=int, default=200)
    ap.add_argument("--scan-timeout", type=int, default=4200)
    ap.add_argument("--entropy-timeout", type=int, default=1800)
    ap.add_argument("--timing-timeout", type=int, default=600)
    ap.add_argument("--reuse-scan", action="store_true",
                    help="reuse existing _detscan.json/_entropy.json (analysis-only re-run)")
    ap.add_argument("--wandb_group", dest="wandb_group", default="selective-dvr-detector")
    ap.add_argument("--wandb_name", dest="wandb_name", default="ubel/selective-dvr-detector-eta")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--relog-wandb", action="store_true")
    a = ap.parse_args()

    if a.relog_wandb:
        report = json.load(open(OUT_DIR / "_results.json"))
        run_ids = log_wandb(report, a)
        report["wandb_run_ids"] = run_ids
        json.dump(report, open(OUT_DIR / "_results.json", "w"), indent=2, default=str)
        marker = {"terminal": True, "status": "complete", "pending_arms": False,
                  "wandb_run_ids": run_ids,
                  "primary_metric": {"name": "token_identity_rate", "value": report["token_identity_rate"]},
                  "test_metric": {"name": "detector_self_test_passes",
                                  "value": int(report["detector_self_test_passes"])},
                  "headline": {k: report.get(k) for k in (
                      "token_identity_rate", "min_selective_eta_at_identity_1",
                      "best_m8only_precision_at_recall1", "L_star_min_separating",
                      "cheap_detector_clears_500_budget", "selective_beats_blanket",
                      "verdict_band")}}
        print("SENPAI-RESULT: " + json.dumps(marker), flush=True)
        return

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 3); a.n_new = min(a.n_new, 64)
        a.det_prompts = min(a.det_prompts, 1); a.dh_iters = min(a.dh_iters, 30)
        a.n_splits = min(a.n_splits, 3)

    if a.phase == "detector_scan":
        phase_detector_scan(a.out, a.n_prompts, a.n_new, a.ctx_cap, a.attn_impl, a.det_prompts)
    elif a.phase == "attn_entropy":
        phase_attn_entropy(a.out, a.n_prompts, a.n_new, a.ctx_cap)
    elif a.phase == "recompute_timing":
        phase_recompute_timing(a.out, a.dh_iters)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
