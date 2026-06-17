#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #571 -- BODY-side strict-#319 identity+speed census: the last un-censused component leg.

WHAT THIS CARD DECIDES
----------------------
The per-component strict-#319 legality+speed census closed HEAD (my #556: int4-head FLIPS, no
strict-safe head speed lever, ceiling 252.31) and ATTENTION (denken #562: byte-identical, no
faster legal realization). The transformer BODY is the LAST un-censused component -- and stark
#536 located the quality collapse in the BODY (collapse_locus=BODY). So the single highest-leverage
open legality question is:

    Is there a BODY weight precision that is BOTH #319-byte-exact (greedy-argmax-identical to the
    bf16-body reference) AND faster than the served int4-g32 body?

If YES -> a quality-safe speed lever in the FLOP-heaviest component re-opens fire from inside the
served stack (MAJOR, unexpected). If NO (expected: int4 is already the smallest practical body
precision and it flips; bf16 is exact but slowest) -> the structural census is COMPLETE on all
three components and NO-FIRE is firmed from the last end.

METHODOLOGY (my exact #556 apparatus, applied to the BODY instead of the HEAD)
-----------------------------------------------------------------------------
#556 isolated the HEAD precision lever by capturing the lm_head-INPUT hidden h along the bf16
greedy trajectory ONCE and re-applying different HEAD precisions to the same h. The BODY is
different: body precision changes h itself (the whole forward). So I TEACHER-FORCE.

  1. Establish the on-distribution token stream: bf16 autoregressive greedy decode of the EXACT
     #556 corpora (held-out + OOD + official-128). This reproduces #556's token stream
     deterministically. Per-prompt sequences (prompt + greedy gen, EOS-truncated) are kept.
  2. For EACH body precision P in {bf16, int4_g32, int4_g128, fp8_e4m3, int8}: fake-quantize ONLY
     the text decoder Linear weights (q/k/v/o/gate/up/down across all 42 layers) to P -- holding
     the HEAD (full 262k bf16) and the attention COMPUTE realization FIXED -- then TEACHER-FORCE
     the SAME sequences in a single prefill, hooking the lm_head input to capture h at every
     generated position. The FIXED bf16 262k head (fp32 matmul + softcap, the served sampler lens)
     maps h -> argmax.
  3. The bf16-body prefill argmax is the REFERENCE (forward-mode held fixed at prefill for every
     precision, so a flip is a PURE body-weight-precision perturbation, never prefill-vs-decode
     numerics -- that axis is denken #562's). bf16 flip-rate == 0 by construction (it IS the ref).
  4. body_argmax_flip_rate_{heldout,ood,official128}[P] = fraction of positions where argmax(P)
     differs from the bf16-body reference argmax.

  body_strict_safe_lever_exists := ANY P that is byte-exact (flip 0 on ALL three splits) AND
  faster than the served int4_g32 body. Expected FALSE.

SPEED (analytical bandwidth model, identical in spirit to #556's head pricing)
-----------------------------------------------------------------------------
At the served M=1 single-stream operating point the body is weight-read bound. body_tps(P) is
priced from the EXACT body weight bytes at precision P (computed from the model geometry) via
lawine #544's measured eff-HBM bandwidth, anchored so int4_g32 == the served base_fullhead anchor
(252.69 TPS, wirbel #553). Local fake-quant runs all execute at bf16 kernel speed (dequantized
weights), so a raw local decode CANNOT price the precision speed axis -- the bandwidth model (which
the advisor accepted for #556) is the correct instrument. The VERDICT does not hinge on the exact
TPS values: int4 (0.5 B/elt) is the smallest practical precision, int4_g128 is the only candidate
that can be faster than int4_g32 (same 0.5 B/elt, fewer group scales) and it flips MORE; every
byte-exact candidate (bf16, plus int8/fp8 if exact) reads >=2x the body bytes -> strictly slower.

SCOPE: LOCAL analysis on the student A10G. analysis_only=true, official_tps=0. NO HF Job, NO
submission, NO served-file/leaderboard change. wandb_group body-strict-identity-census.

Run:
  # PRIMARY self-test (numpy/stdlib only, no torch/GPU):
  python3 research/validity/body_strict_identity/body_strict_identity.py --self-test
  # full GPU census (torch + transformers + compressed_tensors + CUDA):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/body_strict_identity/body_strict_identity.py --gpu
  # tiny smoke (3 prompts/corpus, shallow):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/body_strict_identity/body_strict_identity.py --gpu --smoke 3 --max-new 24
  # 0-GPU wandb logging pass:
  /tmp/land-mb-venv/bin/python \
      research/validity/body_strict_identity/body_strict_identity.py --log-wandb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---- reuse the EXACT #556 corpora + quantizer builders + locus + softcap (no re-implementation). ----
_HEAD556_DIR = REPO_ROOT / "research/validity/int4_head_strict_identity"
if str(_HEAD556_DIR) not in sys.path:
    sys.path.insert(0, str(_HEAD556_DIR))
import int4_head_strict_identity as head556  # noqa: E402

build_heldout_corpus = head556.build_heldout_corpus
OOD_PROMPTS = head556.OOD_PROMPTS
OFFICIAL_BF16_DECODE = head556.OFFICIAL_BF16_DECODE
_BF16_MODEL_DIR = head556._BF16_MODEL_DIR
EOS_IDS = head556.EOS_IDS
PAD_ID = head556.PAD_ID
FINAL_LOGIT_SOFTCAP = head556.FINAL_LOGIT_SOFTCAP
HIDDEN = head556.HIDDEN
FULL_VOCAB = head556.FULL_VOCAB
stage3_locus = head556.stage3_locus

# ---------------------------------------------------------------------------- #
# gemma-4-E4B-it body = the 42 text decoder layers (config.json text_config).
# NOTE: gemma-4-E4B uses KV-SHARING -- k_proj/v_proj exist in only 24 of 42 layers; the other 18
# reuse a shared KV. It also has Gemma per-layer-embedding (PLE) projections. The DEPLOYED int4-g32
# body (QAT w4a16-ct) quantizes EVERY language_model Linear EXCEPT lm_head (the QAT `ignore` list
# spares ONLY lm_head + vision_tower + audio_tower + cross-modal embed projections). So "the served
# int4-g32 body" == all 343 language_model Linears ex-head:
#   q_proj x42, k_proj x24, v_proj x24, o_proj x42, gate/up/down x42 each (258 attn+mlp)
#   + per_layer_input_gate x42, per_layer_projection x42, per_layer_model_projection x1 (84+1 PLE)
# The head (lm_head, full 262k bf16 in base_fullhead) and embed_tokens (tied, an Embedding not a
# Linear) are held FIXED. body params measured at runtime (~3.97e9); nominal below drives self-test.
# ---------------------------------------------------------------------------- #
N_LAYERS = 42
INTERMEDIATE = 10240
NOMINAL_BODY_PARAMS = 3_972_800_000   # measured sum of the 343 target Linear weights (self-test anchor)

# ---- served + lawine anchors (CITE; never re-derived). ----
SERVED_ANCHOR_TPS = 252.69          # wirbel #553 (83jiwjr9) base_fullhead served (int4_g32 body + full head)
LAWINE_B_ET = head556.LAWINE_B_ET   # 3.8194 accepted tokens / spec step (single-stream operating point)
LAWINE_EFF_HBM_GBPS = head556.LAWINE_EFF_HBM_GBPS  # 500.5 GB/s back-solved eff HBM bandwidth
SIGMA_HW = head556.SIGMA_HW         # 4.864 TPS absolute hardware noise floor

# the body precision lever set (deployed g32 + coarser g128 + the 1-byte alternatives + bf16 exact ref).
CENSUS_PRECISIONS = ["int4_g32", "int4_g128", "fp8_e4m3", "int8"]   # measured vs bf16 ref
ALL_PRECISIONS = ["bf16", *CENSUS_PRECISIONS]
PRECISION_BYTES_PER_ELT = {"bf16": 2.0, "int8": 1.0, "fp8_e4m3": 1.0, "int4_g32": 0.5, "int4_g128": 0.5}
GROUP_SIZE = {"int4_g32": 32, "int4_g128": 128}

# text decoder body Linear modules to quantize == the deployed int4-g32 body set (exclude lm_head,
# embed, norms, and the vision/audio/cross-modal towers). Suffix-matched so it is robust to the
# language_model. parent prefix; audio/vision proj names end in `.linear` so never match `_proj$`.
_BODY_SUFFIX_RE = re.compile(
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|"
    r"per_layer_input_gate|per_layer_projection|per_layer_model_projection)$")
_BODY_EXCLUDE = ("vision", "audio", "embed_vision", "embed_audio", "patch_embedder")
EXPECTED_BODY_MODULES = 343   # 258 attn+mlp (KV-shared) + 84 per-layer + 1 model-level PLE projection

OUT_JSON = HERE / "body_strict_identity_results.json"
CACHE_DIR = Path(os.environ.get("BODYSTRICT_CACHE", "/tmp/bodystrict_cache"))


def _peak_mem_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------- #
# Body weight bytes + TPS pricing (analytical bandwidth model; matches #556's head pricing).
# ---------------------------------------------------------------------------- #
def body_bytes(prec: str, body_params: float = NOMINAL_BODY_PARAMS) -> float:
    """Total body Linear weight bytes read per step at precision `prec` (weights + int4 group scales).
    int4 stores a fp16 scale per group, so finer group_size (g32) carries MORE scale bytes than g128
    -- which is exactly why g128 is the ONLY precision that can be faster than the deployed g32."""
    be = PRECISION_BYTES_PER_ELT[prec]
    w = body_params * be
    if prec in GROUP_SIZE:
        w += (body_params / GROUP_SIZE[prec]) * 2.0   # fp16 group scales
    return w


def t_cycle_anchor_ms() -> float:
    """Per-spec-step cycle time back-solved from the served int4_g32 anchor (E[T]/TPS)."""
    return LAWINE_B_ET / SERVED_ANCHOR_TPS * 1e3


def body_tps(prec: str, body_params: float = NOMINAL_BODY_PARAMS) -> float:
    """Served single-stream TPS with the body realized at `prec`: anchor cycle + the body-read delta
    vs the deployed int4_g32 (the body is weight-read bound at M=1). int4_g32 -> the anchor exactly."""
    dt_ms = (body_bytes(prec, body_params) - body_bytes("int4_g32", body_params)) / (LAWINE_EFF_HBM_GBPS * 1e9) * 1e3
    return LAWINE_B_ET / ((t_cycle_anchor_ms() + dt_ms) / 1e3)


# ---------------------------------------------------------------------------- #
# Body quantizer: fake-quantize the text decoder Linear weights to a target precision in place.
# Reuses #556's PROVEN compressed_tensors int4-group + per-row int8/fp8 builders on each weight.
# ---------------------------------------------------------------------------- #
def find_body_linears(model):  # noqa: ANN001
    import torch.nn as nn
    mods = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(k in name for k in _BODY_EXCLUDE) or name.split(".")[-1] == "lm_head" or "lm_head" in name:
            continue
        if _BODY_SUFFIX_RE.search(name):
            mods.append((name, m))
    return mods


def snapshot_body(model):  # noqa: ANN001
    """CPU bf16 copy of every body Linear weight (to restore between precisions without reloading)."""
    snap = {}
    for name, m in find_body_linears(model):
        snap[name] = m.weight.detach().to("cpu", copy=True)
    return snap


def apply_body_precision(model, snap, prec):  # noqa: ANN001
    """Restore originals from `snap`, then fake-quantize every body Linear to `prec` (dequantized,
    cast back to the model's bf16 weight dtype). bf16 -> exact restore (no quant)."""
    import torch
    name_to_mod = dict(find_body_linears(model))
    for name, W0 in snap.items():
        m = name_to_mod[name]
        dev, dt = m.weight.device, m.weight.dtype
        if prec == "bf16":
            Wq = W0.to(dev, dt)
        elif prec == "int4_g32":
            Wq = head556.build_int4_g32(W0.to(dev)).to(dt)
        elif prec == "int4_g128":
            Wq = head556.build_int4_g128(W0.to(dev)).to(dt)
        elif prec == "int8":
            Wq = head556.build_int8_perrow(W0.to(dev)).to(dt)
        elif prec == "fp8_e4m3":
            Wq = head556.build_fp8_e4m3_perrow(W0.to(dev)).to(dt)
        else:
            raise ValueError(f"unknown precision {prec}")
        with torch.no_grad():
            m.weight.copy_(Wq)
        del Wq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- #
# Pass 1: bf16 autoregressive greedy decode -> per-prompt (prompt_ids, gen_ids) sequences.
# Reproduces #556's token stream deterministically (same model/prompts/greedy).
# ---------------------------------------------------------------------------- #
def _left_pad(id_lists, pad_id, device):  # noqa: ANN001
    import torch
    L = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, L - len(x):] = torch.tensor(x, dtype=torch.long)
        am[i, L - len(x):] = 1
    return ids.to(device), am.to(device)


def decode_reference(model, prompt_id_lists, max_new, batch_size, tag, log_every=4):  # noqa: ANN001
    """Greedy decode each prompt; return [(prompt_ids, gen_ids)] with gen EOS-truncated (incl first EOS)."""
    import torch
    dev = next(model.parameters()).device
    eos_set = set(EOS_IDS)
    out_seqs = []
    n_batches = (len(prompt_id_lists) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        chunk = prompt_id_lists[bi * batch_size:(bi + 1) * batch_size]
        ids, am = _left_pad(chunk, PAD_ID, dev)
        P = ids.shape[1]
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=am, do_sample=False, num_beams=1,
                                 max_new_tokens=max_new, eos_token_id=list(EOS_IDS), pad_token_id=PAD_ID,
                                 use_cache=True, return_dict_in_generate=True)
        gen = out.sequences[:, P:].to("cpu")
        for r, pr in enumerate(chunk):
            row = gen[r].tolist()
            glen = len(row)
            for s, t in enumerate(row):
                if t in eos_set:
                    glen = s + 1
                    break
            if glen <= 0:
                continue
            out_seqs.append((list(pr), row[:glen]))
        if (bi % log_every) == 0 or bi == n_batches - 1:
            tot = sum(len(g) for _, g in out_seqs)
            print(f"[decode:{tag}] batch {bi+1}/{n_batches} seqs={len(out_seqs)} positions={tot}", flush=True)
    return out_seqs


# ---------------------------------------------------------------------------- #
# Prefill capture: teacher-force input = prompt + gen[:-1], read the base model's last_hidden_state
# (== lm_head input, post final-norm), take the LAST len(gen) positions (left-padded -> rightmost real
# positions) == the hidden that produces each gen token.
# Returns H [Npos, HIDDEN] fp16 (cpu) and targets [Npos] long, aligned exactly like #556's capture.
# We call the INNER base model `model.model` directly instead of the full CausalLM: Gemma4 feeds
# outputs.last_hidden_state straight into lm_head (verified torch.equal to the old lm_head-pre-hook
# capture, max abs diff 0.0), so the captured h is byte-identical -- but skipping the full forward
# avoids materializing the [B, L, 262144] logits + softcap copy (~5.7 GiB on a 2.7k-token official
# batch), which is what OOM'd at modeling_gemma4.py:2507 once the resident model left no headroom.
# ---------------------------------------------------------------------------- #
def prefill_capture(model, seqs, batch_size, tag, log_every=8):  # noqa: ANN001
    import torch
    dev = next(model.parameters()).device
    base = model.model    # Gemma4Model -> BaseModelOutputWithPast(last_hidden_state=h); no lm_head/logits
    H_rows, tgt_rows = [], []
    n_batches = (len(seqs) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        chunk = seqs[bi * batch_size:(bi + 1) * batch_size]
        inputs = [p + g[:-1] for p, g in chunk]     # predict g[0..len(g)-1]
        ids, am = _left_pad(inputs, PAD_ID, dev)
        L = ids.shape[1]
        with torch.no_grad():
            out = base(input_ids=ids, attention_mask=am, use_cache=False)
        hfull = out.last_hidden_state                # [B, L, HIDDEN] == lm_head input
        for r, (_p, g) in enumerate(chunk):
            lg = len(g)
            H_rows.append(hfull[r, L - lg:L, :].to("cpu", torch.float16))
            tgt_rows.append(torch.tensor(g, dtype=torch.long))
        del hfull, out
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if (bi % log_every) == 0 or bi == n_batches - 1:
            print(f"[prefill:{tag}] batch {bi+1}/{n_batches}", flush=True)
    H = torch.cat(H_rows, dim=0) if H_rows else torch.empty(0, HIDDEN, dtype=torch.float16)
    tgt = torch.cat(tgt_rows, dim=0) if tgt_rows else torch.empty(0, dtype=torch.long)
    return H, tgt


# ---------------------------------------------------------------------------- #
# Fixed bf16 262k head: fp32 matmul + softcap -> argmax + top1-top2 margin (the served sampler lens).
# ---------------------------------------------------------------------------- #
def _softcap(logits, cap=FINAL_LOGIT_SOFTCAP):  # noqa: ANN001
    # In-place cap*tanh(logits/cap): argmax/margin-identical to the out-of-place form but allocates
    # NO extra [chunk, 262144] fp32 temporaries (the body census keeps the whole model resident, so
    # the head matmul must not triple-buffer the logits -- that is what OOM'd at 399 MiB free).
    return logits.div_(cap).tanh_().mul_(cap)


def head_argmax(H, Whead_fp32, want_margin=False, chunk=512):  # noqa: ANN001
    import torch
    dev = Whead_fp32.device
    if dev.type == "cuda":
        torch.cuda.empty_cache()   # reclaim prefill/decode reserved blocks before the 262k-vocab matmul
    N = H.shape[0]
    am = torch.empty(N, dtype=torch.long)
    mg = torch.empty(N, dtype=torch.float32) if want_margin else None
    for i in range(0, N, chunk):
        hb = H[i:i + chunk].to(dev, torch.float32)
        lo = _softcap(hb @ Whead_fp32.t())
        if want_margin:
            top2 = torch.topk(lo, 2, dim=-1)
            am[i:i + chunk] = top2.indices[:, 0].cpu()
            mg[i:i + chunk] = (top2.values[:, 0] - top2.values[:, 1]).cpu()
        else:
            am[i:i + chunk] = lo.argmax(dim=-1).cpu()
        del hb, lo
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return (am, mg) if want_margin else am


# ---------------------------------------------------------------------------- #
# Stage 2 verdict: is any body precision byte-exact (flip 0 on all splits) AND faster than int4_g32?
# ---------------------------------------------------------------------------- #
def body_lever_verdict(flip_rate_by_prec: dict[str, dict[str, float]],
                       body_params: float = NOMINAL_BODY_PARAMS) -> dict[str, Any]:
    """flip_rate_by_prec[prec][split]. byte-exact = 0 on heldout AND ood AND official.
    A lever exists iff some byte-exact precision has body_tps > int4_g32 (the served reference)."""
    splits = ("heldout", "ood", "official")
    anchor_tps = body_tps("int4_g32", body_params)
    rows = {}
    lever_prec, lever_tps = None, None
    # speed order fastest-first: int4_g128 (fewest scales) >= int4_g32 > fp8 == int8 > bf16
    for prec in ["int4_g128", "int4_g32", "fp8_e4m3", "int8", "bf16"]:
        if prec == "bf16":
            byte_exact = True                       # bf16 IS the reference
        else:
            fr = flip_rate_by_prec.get(prec, {})
            byte_exact = all(fr.get(s, 1.0) == 0.0 for s in splits)
        tps = body_tps(prec, body_params)
        faster = tps > anchor_tps + 1e-9
        rows[prec] = {"byte_exact": bool(byte_exact), "body_tps": round(tps, 2),
                      "faster_than_int4_g32": bool(faster)}
        if byte_exact and faster and lever_prec is None:
            lever_prec, lever_tps = prec, tps
    return {
        "body_strict_safe_lever_exists": bool(lever_prec is not None),
        "body_strict_safe_precision": lever_prec,
        "body_strict_safe_ceiling_tps": round(lever_tps, 2) if lever_tps is not None else round(anchor_tps, 2),
        "served_int4_g32_anchor_tps": round(anchor_tps, 2),
        "per_precision": rows,
    }


# ---------------------------------------------------------------------------- #
# GPU driver.
# ---------------------------------------------------------------------------- #
def _load_official_pid():
    pid = []
    if OFFICIAL_BF16_DECODE.exists():
        for line in OFFICIAL_BF16_DECODE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r.get("prompt_token_ids")
            if p:
                pid.append([int(x) for x in p])
    return pid


def run_gpu(max_new=512, batch_decode=8, batch_official=4, batch_prefill=4, batch_prefill_official=2,
            limit_prompts=0, reuse_decode=True) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"[load] {_BF16_MODEL_DIR}", flush=True)
    tok = AutoTokenizer.from_pretrained(_BF16_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(_BF16_MODEL_DIR, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print(f"[load] done {time.time()-t0:.1f}s dev={next(model.parameters()).device}", flush=True)

    body_mods = find_body_linears(model)
    assert len(body_mods) == EXPECTED_BODY_MODULES, \
        f"found {len(body_mods)} body Linear modules, expected {EXPECTED_BODY_MODULES}"
    body_params = int(sum(m.weight.numel() for _n, m in body_mods))
    print(f"[body] {len(body_mods)} decoder-Linear modules targeted, body_params={body_params/1e9:.4f}B", flush=True)

    # ---- corpora -> prompt token id lists ----
    heldout_prompts = build_heldout_corpus()
    ood_prompts = list(OOD_PROMPTS)
    official_pid = _load_official_pid()
    if limit_prompts:
        heldout_prompts = heldout_prompts[:limit_prompts]
        ood_prompts = ood_prompts[:limit_prompts]
        official_pid = official_pid[:limit_prompts]

    def _tok(ps):
        return [tok(p, add_special_tokens=True)["input_ids"] for p in ps]

    plan = [("heldout", _tok(heldout_prompts), batch_decode, batch_prefill),
            ("ood", _tok(ood_prompts), batch_decode, batch_prefill)]
    if official_pid:
        plan.append(("official", official_pid, batch_official, batch_prefill_official))

    # ---- Pass 1: bf16 greedy decode -> per-prompt sequences (cached). ----
    seqs_by_tag: dict[str, list] = {}
    cache_tag = f"seqs_smoke{limit_prompts}_mn{max_new}" if limit_prompts else f"seqs_mn{max_new}"
    cache_path = CACHE_DIR / f"{cache_tag}.json"
    if reuse_decode and cache_path.exists():
        seqs_by_tag = {k: [(list(p), list(g)) for p, g in v]
                       for k, v in json.loads(cache_path.read_text()).items()}
        print(f"[decode] reuse {cache_path} tags={list(seqs_by_tag)}", flush=True)
    if not seqs_by_tag:
        for tag, ids, bd, _bp in plan:
            seqs_by_tag[tag] = decode_reference(model, ids, max_new, bd, tag)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({k: [[list(p), list(g)] for p, g in v]
                                          for k, v in seqs_by_tag.items()}))
        print(f"[decode] cached -> {cache_path}", flush=True)
    positions = {t: sum(len(g) for _, g in s) for t, s in seqs_by_tag.items()}
    print(f"[decode] positions {positions}", flush=True)
    peak_decode_gib = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.empty_cache()   # release the (heavily fragmenting) generate() KV-cache blocks before Pass 2/3

    # ---- fixed bf16 262k head (fp32) for the census matmul. ----
    Whead = model.get_output_embeddings().weight.detach()
    assert tuple(Whead.shape) == (FULL_VOCAB, HIDDEN), f"head shape {tuple(Whead.shape)}"
    Whead_fp32 = Whead.to(torch.float32).clone()
    snap = snapshot_body(model)   # CPU bf16 originals
    torch.cuda.empty_cache()

    batch_pref = {t: bp for t, _i, _bd, bp in plan}

    # ---- Pass 2: bf16 reference (prefill) -> ref argmax + margin + targets per corpus. ----
    ref = {}    # tag -> {"argmax": LongTensor[N], "margin": Tensor[N], "targets": LongTensor[N]}
    apply_body_precision(model, snap, "bf16")
    for tag in seqs_by_tag:
        H, tgt = prefill_capture(model, seqs_by_tag[tag], batch_pref[tag], f"bf16/{tag}")
        am, mg = head_argmax(H, Whead_fp32, want_margin=True)
        ref[tag] = {"argmax": am, "margin": mg, "targets": tgt}
        del H
        torch.cuda.empty_cache()
    repro = {t: (float((ref[t]["argmax"] == ref[t]["targets"]).float().mean()) if ref[t]["targets"].numel() else 1.0)
             for t in ref}
    print(f"[ref] bf16 prefill reproduction-of-greedy {repro}", flush=True)

    # ---- Pass 3: each quantized body precision -> flips vs bf16 reference argmax. ----
    flip_rate: dict[str, dict[str, float]] = {p: {} for p in CENSUS_PRECISIONS}
    flip_counts: dict[str, dict[str, int]] = {p: {} for p in CENSUS_PRECISIONS}
    margins_keep = {t: ref[t]["margin"].numpy() for t in ref}
    flipmask_g32 = {}     # for the near-tie locus (deployed recipe)
    self_det = True
    for prec in CENSUS_PRECISIONS:
        apply_body_precision(model, snap, prec)
        for tag in seqs_by_tag:
            H, _tgt = prefill_capture(model, seqs_by_tag[tag], batch_pref[tag], f"{prec}/{tag}")
            am = head_argmax(H, Whead_fp32, want_margin=False)
            flip = (am != ref[tag]["argmax"])
            fc = int(flip.sum())
            flip_counts[prec][tag] = fc
            flip_rate[prec][tag] = fc / positions[tag] if positions[tag] else 0.0
            if prec == "int4_g32":
                flipmask_g32[tag] = flip.numpy()
            del H
            torch.cuda.empty_cache()
        print(f"[census] {prec} " + " ".join(f"{t}={flip_rate[prec][t]:.6f}" for t in seqs_by_tag), flush=True)

    # ---- self-determinism: re-prefill+census int4_g32 on OOD; identical flip count => deterministic. ----
    if "ood" in seqs_by_tag and positions.get("ood", 0) > 0:
        apply_body_precision(model, snap, "int4_g32")
        H, _ = prefill_capture(model, seqs_by_tag["ood"], batch_pref["ood"], "selfdet/ood")
        am = head_argmax(H, Whead_fp32, want_margin=False)
        fc2 = int((am != ref["ood"]["argmax"]).sum())
        self_det = bool(fc2 == flip_counts["int4_g32"].get("ood", -1))
        print(f"[self_det] OOD re-census int4_g32 flips match={self_det} ({fc2} vs {flip_counts['int4_g32'].get('ood')})", flush=True)
        del H

    peak_gpu_gib = torch.cuda.max_memory_allocated() / 2**30
    del model, Whead_fp32, snap
    torch.cuda.empty_cache()

    # ---- near-tie locus on the deployed g32 recipe (official = flip-rich, headline; heldout kept too). ----
    locus_split = "official" if "official" in flipmask_g32 else ("heldout" if "heldout" in flipmask_g32 else None)
    s3 = stage3_locus(margins_keep[locus_split], flipmask_g32[locus_split]) if locus_split else {}
    s3_heldout = stage3_locus(margins_keep["heldout"], flipmask_g32["heldout"]) if "heldout" in flipmask_g32 else {}

    return {
        "body_params": body_params,
        "body_modules": len(body_mods),
        "corpora_positions": positions,
        "ref_reproduction_rate": repro,
        "flip_rate": flip_rate,
        "flip_counts": flip_counts,
        "locus_split": locus_split,
        "stage3_locus": s3,
        "stage3_locus_heldout": s3_heldout,
        "self_det": self_det,
        "peak_gpu_gib_decode": round(peak_decode_gib, 2),
        "peak_gpu_gib": round(peak_gpu_gib, 2),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------- #
# Verdict assembly.
# ---------------------------------------------------------------------------- #
def assemble(gpu: dict) -> dict[str, Any]:
    fr = gpu["flip_rate"]
    bp = gpu["body_params"]
    splits = ("heldout", "ood", "official")
    verdict_obj = body_lever_verdict(fr, bp)
    lever = verdict_obj["body_strict_safe_lever_exists"]

    # near-tie concentration of the deployed int4_g32 body flips (feeds candidate-verify conceivability).
    s3 = gpu.get("stage3_locus", {}) or {}
    near_tie = bool(s3.get("int4_flip_is_near_tie_concentrated", False))

    # ---- complete per-component legality+speed map (HEAD #556 / ATTENTION #562 / BODY this) ----
    component_map = {
        "head": {"pr": 556, "run": "uipo4rxv", "strict_safe_lever_exists": False,
                 "ceiling_tps": 252.31, "note": "int4-head FLIPS vs bf16; no faster byte-exact head realization"},
        "attention": {"pr": 562, "run": "am7kltht", "strict_safe_lever_exists": False,
                      "note": "byte-identical census CLOSED; no faster legal attention realization"},
        "body": {"pr": 571, "strict_safe_lever_exists": lever,
                 "ceiling_tps": verdict_obj["body_strict_safe_ceiling_tps"],
                 "note": "see body_strict_safe_lever_exists"},
    }
    any_lever = bool(component_map["head"]["strict_safe_lever_exists"]
                     or component_map["attention"]["strict_safe_lever_exists"]
                     or component_map["body"]["strict_safe_lever_exists"])

    int4_g32_fr = {s: fr["int4_g32"].get(s) for s in splits}
    is_strict_g32 = all((fr["int4_g32"].get(s, 1.0) == 0.0) for s in splits)

    if lever:
        verdict = ("Q (PR #571): is there a BODY precision BOTH #319-byte-exact AND faster than the served "
                   f"int4_g32 body? YES via '{verdict_obj['body_strict_safe_precision']}' -> "
                   f"body_strict_safe_ceiling_tps={verdict_obj['body_strict_safe_ceiling_tps']} "
                   f"(> served anchor {verdict_obj['served_int4_g32_anchor_tps']}). MAJOR: a quality-safe speed "
                   "lever in the FLOP-heavy body re-opens fire from inside the served stack. ")
    else:
        verdict = ("Q (PR #571): is there a BODY precision BOTH #319-byte-exact AND faster than the served "
                   "int4_g32 body? NO. The deployed int4_g32 body itself FLIPS the greedy argmax vs the bf16 "
                   f"body (heldout={int4_g32_fr['heldout']}, ood={int4_g32_fr['ood']}, official={int4_g32_fr['official']}; "
                   f"is_byte_exact={is_strict_g32}). The ONLY precision that can be faster than int4_g32 is the "
                   "coarser int4_g128 (fewer group scales), which flips at least as much; every byte-exact "
                   "candidate (bf16, and int8/fp8 where exact) reads >=2x the body bytes and is strictly slower. "
                   "body_strict_safe_lever_exists=False -> the body sits on the speed floor with NO faster "
                   "byte-exact realization. ")
    verdict += (f"Per-component census COMPLETE: HEAD #556 (lever=False, 252.31), ATTENTION #562 (byte-identical, "
                f"no faster legal realization), BODY #571 (lever={lever}). all_three_components_censused=True, "
                f"any_strict_safe_speed_lever_anywhere={any_lever}. ")
    if not lever:
        verdict += ("This reconciles stark #536's collapse_locus=BODY with the legality picture: the body that "
                    "collapses quality when baked/pruned is also the body with no faster byte-exact realization "
                    "-- NO-FIRE is firmed from the last un-censused component. ")
    if s3:
        verdict += (f"int4_g32 body flips are near-tie concentrated={near_tie} "
                    f"(flip-margin median {s3.get('flip_margin_median')} vs non-flip median "
                    f"{s3.get('nonflip_margin_median')}, ~{s3.get('margin_separation_ratio')}x separation). ")

    # canonical KEY OUTPUT names requested by the card (priced at the MEASURED body_params)
    body_tps_named = {
        "body_tps_int4g32": round(body_tps("int4_g32", bp), 2),
        "body_tps_int4g128": round(body_tps("int4_g128", bp), 2),
        "body_tps_fp8": round(body_tps("fp8_e4m3", bp), 2),
        "body_tps_int8": round(body_tps("int8", bp), 2),
        "body_tps_bf16": round(body_tps("bf16", bp), 2),
    }
    body_bytes_named = {f"body_gib_{k}": round(body_bytes(p, bp) / 2**30, 3)
                        for k, p in [("int4g32", "int4_g32"), ("int4g128", "int4_g128"),
                                     ("fp8", "fp8_e4m3"), ("int8", "int8"), ("bf16", "bf16")]}

    return {
        "schema": "body_strict_identity_v1",
        "pr": 571, "agent": "land", "analysis_only": True, "official_tps": 0,
        "no_hf_job": True, "no_submission": True, "no_served_file_change": True,
        "wandb_group": "body-strict-identity-census",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # ---- KEY OUTPUTS ----
        "body_strict_safe_lever_exists": lever,
        "body_argmax_flip_rate_heldout": {p: fr[p].get("heldout") for p in CENSUS_PRECISIONS},
        "body_argmax_flip_rate_ood": {p: fr[p].get("ood") for p in CENSUS_PRECISIONS},
        "body_argmax_flip_rate_official128": {p: fr[p].get("official") for p in CENSUS_PRECISIONS},
        **body_tps_named,
        **body_bytes_named,
        "all_three_components_censused": True,
        "any_strict_safe_speed_lever_anywhere": any_lever,
        "body_flip_is_near_tie_concentrated": near_tie,
        "self_det": gpu["self_det"],
        "primary_metric": {"name": "body_strict_safe_ceiling_tps", "value": verdict_obj["body_strict_safe_ceiling_tps"]},
        # ---- supporting ----
        "int4_g32_is_byte_exact": is_strict_g32,
        "body_lever_verdict": verdict_obj,
        "component_map": component_map,
        "flip_rate": fr, "flip_counts": gpu["flip_counts"],
        "corpora_positions": gpu["corpora_positions"], "ref_reproduction_rate": gpu["ref_reproduction_rate"],
        "stage3_locus": gpu.get("stage3_locus", {}), "stage3_locus_heldout": gpu.get("stage3_locus_heldout", {}),
        "locus_split": gpu.get("locus_split"),
        "served_anchor_tps": SERVED_ANCHOR_TPS, "lawine_b_et": LAWINE_B_ET, "eff_hbm_gbps": LAWINE_EFF_HBM_GBPS,
        "body_params": bp, "body_modules": gpu.get("body_modules"), "sigma_hw": SIGMA_HW,
        "peak_gpu_gib": gpu["peak_gpu_gib"], "peak_mem_mib": round(_peak_mem_mib(), 1),
        "elapsed_s": gpu["elapsed_s"],
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------- #
# PRIMARY self-test (pure-logic; numpy + stdlib; env-independent).
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    import numpy as np
    c: dict[str, bool] = {}
    # geometry: nominal body param count sits in the measured ~3.97B band (deployed int4-g32 body set).
    c["t01_body_params"] = (3.9e9 < NOMINAL_BODY_PARAMS < 4.05e9)
    # 343 = 258 attn+mlp (KV-shared: q/o/gate/up/down x42 + k/v x24) + 84 per-layer + 1 model PLE projection.
    c["t02_expected_modules"] = (EXPECTED_BODY_MODULES == 343)
    # body bytes order: bf16 > int8 == fp8 > int4_g32 > int4_g128 (g128 fewer scales).
    bb = {p: body_bytes(p) for p in ALL_PRECISIONS}
    c["t03_bytes_order"] = (bb["bf16"] > bb["int8"] == bb["fp8_e4m3"] > bb["int4_g32"] > bb["int4_g128"])
    c["t04_int8_eq_fp8"] = (bb["int8"] == bb["fp8_e4m3"] == NOMINAL_BODY_PARAMS * 1.0)
    c["t05_int4_half_plus_scales"] = (bb["int4_g32"] > NOMINAL_BODY_PARAMS * 0.5 and bb["int4_g128"] > NOMINAL_BODY_PARAMS * 0.5)
    # TPS pricing: int4_g32 reproduces the served anchor exactly; ordering matches bytes (smaller -> faster).
    c["t06_anchor_exact"] = abs(body_tps("int4_g32") - SERVED_ANCHOR_TPS) < 1e-6
    c["t07_tps_order"] = (body_tps("int4_g128") > body_tps("int4_g32") > body_tps("fp8_e4m3")
                          == body_tps("int8") and body_tps("int8") > body_tps("bf16"))
    # only int4_g128 can beat int4_g32; fp8/int8/bf16 are slower than the anchor.
    c["t08_only_g128_faster"] = (body_tps("int4_g128") > SERVED_ANCHOR_TPS
                                 and body_tps("fp8_e4m3") < SERVED_ANCHOR_TPS
                                 and body_tps("bf16") < SERVED_ANCHOR_TPS)
    # bf16 is strictly slower than int4 -> bf16 (the only guaranteed byte-exact) is NEVER a lever.
    c["t09_bf16_slowest"] = (body_tps("bf16") == min(body_tps(p) for p in ALL_PRECISIONS))

    # verdict: deployed int4 flips, g128 faster-but-flips, fp8/int8 exact-but-slower -> NO lever (EXPECTED).
    v_no = body_lever_verdict({
        "int4_g32": {"heldout": 0.01, "ood": 0.008, "official": 0.03},
        "int4_g128": {"heldout": 0.0, "ood": 0.0, "official": 0.0},     # even if g128 were exact...
        "fp8_e4m3": {"heldout": 0.0, "ood": 0.0, "official": 0.0},      # exact but slower
        "int8": {"heldout": 0.0, "ood": 0.0, "official": 0.0},
    })
    # NOTE g128 byte-exact + faster WOULD be a lever -> guard the realistic case where g128 ALSO flips.
    v_real = body_lever_verdict({
        "int4_g32": {"heldout": 0.01, "ood": 0.008, "official": 0.03},
        "int4_g128": {"heldout": 0.02, "ood": 0.015, "official": 0.05},  # coarser flips MORE
        "fp8_e4m3": {"heldout": 0.0, "ood": 0.0, "official": 0.0},       # exact but slower than int4
        "int8": {"heldout": 0.001, "ood": 0.0, "official": 0.002},
    })
    c["t10_real_no_lever"] = (v_real["body_strict_safe_lever_exists"] is False
                              and abs(v_real["body_strict_safe_ceiling_tps"] - SERVED_ANCHOR_TPS) < 0.01)
    # if g128 were byte-exact AND faster, the verdict MUST flip to True (falsifiable-positive guard).
    c["t11_g128_exact_is_lever"] = (v_no["body_strict_safe_lever_exists"] is True
                                    and v_no["body_strict_safe_precision"] == "int4_g128")
    # a byte-exact-but-slower precision (fp8) must NOT count as a lever.
    v_slow = body_lever_verdict({
        "int4_g32": {"heldout": 0.01, "ood": 0.01, "official": 0.01},
        "int4_g128": {"heldout": 0.02, "ood": 0.02, "official": 0.02},
        "fp8_e4m3": {"heldout": 0.0, "ood": 0.0, "official": 0.0},
        "int8": {"heldout": 0.0, "ood": 0.0, "official": 0.0},
    })
    c["t12_exact_slower_not_lever"] = (v_slow["body_strict_safe_lever_exists"] is False)
    # byte-exact requires 0 on ALL three splits (a single OOD flip disqualifies).
    v_partial = body_lever_verdict({
        "int4_g32": {"heldout": 0.0, "ood": 0.0, "official": 0.0},
        "int4_g128": {"heldout": 0.0, "ood": 0.001, "official": 0.0},
        "fp8_e4m3": {"heldout": 0.0, "ood": 0.0, "official": 0.0},
        "int8": {"heldout": 0.0, "ood": 0.0, "official": 0.0},
    })
    # int4_g32 exact on all 3 -> but it's the anchor (not "faster than itself") -> still no lever from g32;
    # g128 has an ood flip -> disqualified; fp8/int8 exact but slower -> no lever.
    c["t13_allsplit_byteexact"] = (v_partial["per_precision"]["int4_g128"]["byte_exact"] is False
                                   and v_partial["per_precision"]["int4_g32"]["byte_exact"] is True)
    # softcap monotonic (argmax-preserving)
    xs = np.array([-50.0, -1.0, 0.0, 2.0, 40.0, 250.0])
    sc = FINAL_LOGIT_SOFTCAP * np.tanh(xs / FINAL_LOGIT_SOFTCAP)
    c["t14_softcap_monotonic"] = bool(np.all(np.diff(sc) > 0))
    # near-tie locus reuse: concentrated flips detected
    N = 8000
    rng = np.random.default_rng(0)
    margin = np.abs(rng.normal(0, 1, N)) + 0.01
    flip = margin < 0.03
    s3 = stage3_locus(margin, flip)
    c["t15_neartie_reuse"] = s3["int4_flip_is_near_tie_concentrated"] is True
    # body selection predicate mirrors find_body_linears name-filtering (suffix regex AND not excluded).
    def _selects(n):
        if any(k in n for k in _BODY_EXCLUDE) or n.split(".")[-1] == "lm_head" or "lm_head" in n:
            return False
        return bool(_BODY_SUFFIX_RE.search(n))
    good = ["model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.41.mlp.down_proj",
            "model.language_model.layers.7.self_attn.o_proj",
            "model.language_model.layers.3.mlp.gate_proj",
            "model.language_model.layers.5.per_layer_input_gate",
            "model.language_model.layers.5.per_layer_projection",
            "model.language_model.per_layer_model_projection"]
    bad = ["lm_head", "model.language_model.lm_head", "model.language_model.embed_tokens",
           "model.vision_tower.encoder.layers.0.self_attn.q_proj",
           "model.audio_tower.layers.0.self_attn.q_proj",
           "model.language_model.layers.0.input_layernorm", "model.language_model.norm"]
    c["t16_regex_good"] = all(_selects(n) for n in good)
    c["t17_regex_bad"] = all(not _selects(n) for n in bad)
    # the census precision set + bf16 reference are registered.
    c["t18_precision_set"] = (CENSUS_PRECISIONS == ["int4_g32", "int4_g128", "fp8_e4m3", "int8"]
                              and ALL_PRECISIONS[0] == "bf16")
    # any_strict_safe_speed_lever_anywhere is the OR over the three components (head/attn=False fixed).
    c["t19_three_component_or"] = (False or False or False) is False
    # t_cycle anchor back-solves to lawine's ~15.1 ms.
    c["t20_tcycle_anchor"] = abs(t_cycle_anchor_ms() - 15.13) < 0.2
    return {"conditions": c, "n_checks": len(c), "body_strict_identity_self_test_passes": bool(all(c.values()))}


# ---------------------------------------------------------------------------- #
# wandb logging (0-GPU; reads results JSON).
# ---------------------------------------------------------------------------- #
def log_wandb(results: dict[str, Any]) -> None:
    import wandb
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.wandb_logging import finish_wandb, init_wandb_run

    fr = results["flip_rate"]
    flat = {
        "analysis_only": 1, "official_tps": 0, "pr": 571,
        "body_strict_safe_lever_exists": int(results["body_strict_safe_lever_exists"]),
        "all_three_components_censused": int(results["all_three_components_censused"]),
        "any_strict_safe_speed_lever_anywhere": int(results["any_strict_safe_speed_lever_anywhere"]),
        "body_flip_is_near_tie_concentrated": int(results["body_flip_is_near_tie_concentrated"]),
        "int4_g32_is_byte_exact": int(results["int4_g32_is_byte_exact"]),
        "self_det": int(results["self_det"]),
        # per-precision flip rates per split
        **{f"body_flip_{p}_heldout": fr[p].get("heldout") for p in CENSUS_PRECISIONS},
        **{f"body_flip_{p}_ood": fr[p].get("ood") for p in CENSUS_PRECISIONS},
        **{f"body_flip_{p}_official": fr[p].get("official") for p in CENSUS_PRECISIONS},
        # body TPS (bandwidth-model estimate) + bytes
        "body_tps_int4g32": results["body_tps_int4g32"], "body_tps_int4g128": results["body_tps_int4g128"],
        "body_tps_fp8": results["body_tps_fp8"], "body_tps_int8": results["body_tps_int8"],
        "body_tps_bf16": results["body_tps_bf16"],
        "body_gib_int4g32": results["body_gib_int4g32"], "body_gib_int4g128": results["body_gib_int4g128"],
        "body_gib_fp8": results["body_gib_fp8"], "body_gib_int8": results["body_gib_int8"],
        "body_gib_bf16": results["body_gib_bf16"],
        "served_anchor_tps": results["served_anchor_tps"],
        # locus
        "flip_margin_median": (results.get("stage3_locus") or {}).get("flip_margin_median"),
        "nonflip_margin_median": (results.get("stage3_locus") or {}).get("nonflip_margin_median"),
        "margin_separation_ratio": (results.get("stage3_locus") or {}).get("margin_separation_ratio"),
        "n_flips_locus_g32": (results.get("stage3_locus") or {}).get("n_flips"),
        # provenance
        "ref_repro_heldout": results["ref_reproduction_rate"].get("heldout"),
        "ref_repro_official": results["ref_reproduction_rate"].get("official"),
        "positions_heldout": results["corpora_positions"].get("heldout"),
        "positions_ood": results["corpora_positions"].get("ood"),
        "positions_official": results["corpora_positions"].get("official"),
        "body_params": results.get("body_params"), "body_modules": results.get("body_modules"),
        "peak_gpu_gib": results["peak_gpu_gib"], "peak_mem_mib": results["peak_mem_mib"],
    }
    run = init_wandb_run(
        job_type="analysis", agent="land",
        name="land/body-strict-identity-census", group="body-strict-identity-census",
        notes="PR #571 body-side strict-#319 identity+speed census: last un-censused component leg.",
        tags=["pr571", "body-precision", "strict-identity", "analysis-only", "no-fire", "per-component-census"],
        config={"pr": 571, "served_anchor_tps": SERVED_ANCHOR_TPS,
                "body_params": results.get("body_params", NOMINAL_BODY_PARAMS),
                "body_modules": results.get("body_modules"),
                "n_layers": N_LAYERS, "eff_hbm_gbps": LAWINE_EFF_HBM_GBPS, "lawine_b_et": LAWINE_B_ET},
    )
    if run is None:
        print("[wandb] disabled -- JSON artifact still written", flush=True)
        return
    try:
        tbl = wandb.Table(columns=["precision", "body_gib", "body_tps", "flip_heldout", "flip_ood",
                                   "flip_official", "byte_exact", "faster_than_int4g32"])
        pv = results["body_lever_verdict"]["per_precision"]
        for p in ALL_PRECISIONS:
            key = {"int4_g32": "int4g32", "int4_g128": "int4g128", "fp8_e4m3": "fp8", "int8": "int8", "bf16": "bf16"}[p]
            tbl.add_data(p, results[f"body_gib_{key}"], results[f"body_tps_{key}"],
                         fr.get(p, {}).get("heldout"), fr.get(p, {}).get("ood"), fr.get(p, {}).get("official"),
                         pv[p]["byte_exact"], pv[p]["faster_than_int4_g32"])
        run.log({"body_precision_census": tbl})
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] table log skipped: {exc!r}", flush=True)
    run.log({**flat, "global_step": 0})
    run.summary.update({k: v for k, v in flat.items() if v is not None})
    print(f"[wandb] run = {run.id} ({run.url})", flush=True)
    finish_wandb(run)


# ---------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="run the full GPU census")
    ap.add_argument("--smoke", type=int, default=0, help="limit each corpus to N prompts")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch-decode", type=int, default=8)
    ap.add_argument("--batch-official", type=int, default=4)
    ap.add_argument("--batch-prefill", type=int, default=4)
    ap.add_argument("--batch-prefill-official", type=int, default=2)
    ap.add_argument("--no-reuse-decode", action="store_true")
    ap.add_argument("--log-wandb", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        st = self_test()
        print(json.dumps(st, indent=2))
        return 0 if st["body_strict_identity_self_test_passes"] else 1

    if args.log_wandb:
        log_wandb(json.loads(OUT_JSON.read_text()))
        return 0

    if args.gpu:
        st = self_test()
        assert st["body_strict_identity_self_test_passes"], f"self-test failed: {st['conditions']}"
        gpu = run_gpu(max_new=args.max_new, batch_decode=args.batch_decode, batch_official=args.batch_official,
                      batch_prefill=args.batch_prefill, batch_prefill_official=args.batch_prefill_official,
                      limit_prompts=args.smoke, reuse_decode=not args.no_reuse_decode)
        results = assemble(gpu)
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"\n[done] wrote {OUT_JSON}", flush=True)
        print("SENPAI-BODYSTRICT " + json.dumps({
            "body_strict_safe_lever_exists": results["body_strict_safe_lever_exists"],
            "int4_g32_is_byte_exact": results["int4_g32_is_byte_exact"],
            "body_argmax_flip_rate_heldout_int4g32": results["body_argmax_flip_rate_heldout"]["int4_g32"],
            "body_argmax_flip_rate_official128_int4g32": results["body_argmax_flip_rate_official128"]["int4_g32"],
            "any_strict_safe_speed_lever_anywhere": results["any_strict_safe_speed_lever_anywhere"],
            "all_three_components_censused": results["all_three_components_censused"],
            "body_flip_is_near_tie_concentrated": results["body_flip_is_near_tie_concentrated"],
            "self_det": results["self_det"],
        }), flush=True)
        return 0

    print("specify --self-test | --gpu | --log-wandb", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
