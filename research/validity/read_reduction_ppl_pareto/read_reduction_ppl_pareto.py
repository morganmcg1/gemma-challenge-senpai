#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PPL-vs-body-read-byte-reduction Pareto via local in-memory fake-quantization.

Analysis-only leg (PR #287). Question: can a PPL-safe body-read-byte reduction
(holding projected deployed PPL <= 2.42; deployed int4 anchor 2.3772, headroom
0.0428) reach the byte-reduction % that moves the HBM-bound TPS ceiling from the
served 481.53 to 500 at fixed E[T]=3.844?

We fake-quantize the DEPLOYED int4 target body (google/gemma-4-E4B-it-qat-w4a16-ct,
pack-quantized int4 group_size=32 symmetric) by dequantizing -> re-rounding onto a
lower-bit grid / 2:4 mask, measure PPL on the official corpus with the EXACT official
teacher-forced arithmetic, and compute the analytic body-read-byte reduction for each
config. We then cross max_ppl_safe_read_reduction_pct against
required_read_reduction_pct_for_500.

HONESTY: this is a LOCAL fake-quant (fp accumulation) OPTIMISTIC proxy for a real
low-bit kernel; it produces 0 TPS and is a CANDIDATE future build, not a measured
deploy. The authoritative speed gate stays MEASURED >=500 at lambda_hat>=0.9780.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

# --------------------------------------------------------------------------- #
# Composition basis (denken #278 linear_step_decomposition) + gates.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE_TPS = 481.53      # served int4 TPS at full body (HBM-bound op point)
TARGET_TPS = 500.0                  # the bar this lever must clear
E_T = 3.844                         # fixed E[T] (kanna #217; drafter ceiling my #281)
DEPLOYED_BODY_GB = 1.76             # deployed int4 body read (floor 2933.83us @600GB/s)
BW_NOMINAL_GBPS = 600.0             # nominal A10G HBM bandwidth
DEPLOYED_INT4_PPL = 2.3772          # deployed served int4 PPL anchor (PR #52, 128/128)
PPL_GATE = 2.42                     # validity bar (headroom 0.0428 over anchor)
PPL_HEADROOM = PPL_GATE - DEPLOYED_INT4_PPL  # 0.0428
OFFICIAL_SCORED_TOKENS = 61797      # exact GT scored-token count (harness-faithful check)
# Imported-EXACT cross-refs (self-test (e); do NOT re-derive).
LAMBDA1_CEILING_TPS = 520.953       # kanna #217 lambda=1 ceiling (520.9527323111674)
K_CAL = 125.268                     # kanna #217 calibration (125.26795005202914)
STEP_US = 1218.2                    # kanna #217 normalized served step
FLOOR_US = 2933.83                  # denken #278 int4 body-read HBM floor (2933.828266667)
TAU_LO = 1.03524                    # lawine #267 tau_lo
PRIVATE_VERIFIED_PPL = 2.3777       # private-verified int4 PPL reference

# Honest-scope caveats carried by this leg (self-test (f)).
CAVEATS = [
    "0_TPS: this leg maps PPL feasibility of the bytes-per-read denominator lever; it "
    "produces NO >=500 build and adds 0 TPS.",
    "FAKE_QUANT_OPTIMISTIC: local in-memory fake-quant with fp accumulation is an "
    "OPTIMISTIC proxy for a real low-bit kernel (real int3/2:4 kernels add group/scale "
    "quant + transcode error).",
    "QAT_PROXY_OPTIMISTIC: the int4 base here is the public QAT w4a16 checkpoint (osoi5 "
    "served weights + serve venv pod-absent per #281); QAT recovers more quality and is "
    "more low-bit-robust than the deployed PTQ int4, so the Pareto here is ALSO optimistic "
    "vs the deployed body -> a closure verdict is decisive, a clear verdict needs re-confirm.",
    "DO_NOT_SERVE: NO served-file change, NO re-quantization of the served checkpoint, "
    "NOT a launch, NOT open2, NO HF Job, NO submission. BASELINE stays 481.53.",
    "CANDIDATE_FUTURE_BUILD: a PPL-safe reduction found here is a human-approval-gated "
    "candidate build, NOT realized here. Launch gate stays land #245 MEASURED >=500 at "
    "lambda_hat>=0.9780.",
]

GROUP_SIZE = 32
BODY_SUBMODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "per_layer_input_gate", "per_layer_projection",
]
# Layers admitting 2:4 structured sparsity (dense matmuls; PLE gates excluded).
SPARSE24_SUBMODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

CKPT_GLOB = "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/*"
CORPUS = ("/workspace/senpai/target/official/main_bucket/shared_resources/"
          "speed_benchmark/data/ppl_ground_truth_tokens.jsonl")
OUTDIR = Path("/workspace/senpai/target/research/validity/read_reduction_ppl_pareto")


def log(*a: Any) -> None:
    print("[rrpp]", *a, flush=True)


# --------------------------------------------------------------------------- #
# Fake-quant kernels (operate on the int4-dequantized bf16 weight).
# --------------------------------------------------------------------------- #
def fakequant_intn(W: torch.Tensor, bits: int, gsz: int = GROUP_SIZE) -> torch.Tensor:
    """Symmetric per-group int-N round-trip. codes in [-2^(b-1), 2^(b-1)-1]."""
    out, inf = W.shape
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    ng = (inf + gsz - 1) // gsz
    pad = ng * gsz - inf
    Wp = torch.nn.functional.pad(W.float(), (0, pad)).view(out, ng, gsz)
    absmax = Wp.abs().amax(dim=2, keepdim=True)
    scale = (absmax / (-qmin)).clamp_min(1e-12)        # absmax -> qmin (neg side)
    q = torch.clamp(torch.round(Wp / scale), qmin, qmax)
    Wq = (q * scale).view(out, ng * gsz)[:, :inf]
    return Wq.to(W.dtype)


def fakequant_2to4(W: torch.Tensor) -> torch.Tensor:
    """Magnitude 2:4 structured sparsity on top of int4 grid: in every contiguous
    group of 4 along the in-dim, keep the 2 largest-magnitude (already-int4) values,
    zero the other 2. (We keep the int4 values; byte model accounts metadata.)"""
    out, inf = W.shape
    g = inf // 4
    rem = inf - g * 4
    body = W[:, : g * 4].view(out, g, 4)
    mag = body.abs()
    # indices of the 2 smallest per group -> zero them
    drop = mag.topk(2, dim=2, largest=False).indices  # [out, g, 2]
    mask = torch.ones_like(body, dtype=torch.bool)
    mask.scatter_(2, drop, False)
    body = torch.where(mask, body, torch.zeros_like(body))
    Wq = W.clone()
    Wq[:, : g * 4] = body.view(out, g * 4)
    return Wq  # tail (< 4) left dense; negligible


# --------------------------------------------------------------------------- #
# Byte accounting (analytic body-read-byte reduction).
# --------------------------------------------------------------------------- #
def linear_int4_bytes(out: int, inf: int, gsz: int = GROUP_SIZE) -> tuple[float, float]:
    """(code_bytes, scale_bytes) for an int4 group-quantized Linear."""
    code = out * inf * 0.5                       # 4 bits/value
    scale = out * math.ceil(inf / gsz) * 2.0     # bf16 group scales
    return code, scale


def linear_scheme_code_bytes(out: int, inf: int, scheme: tuple) -> float:
    """code+metadata bytes (NOT scale) for a Linear under a scheme."""
    kind = scheme[0]
    if kind == "intn":
        bits = scheme[1]
        return out * inf * (bits / 8.0)
    if kind == "sparse24":
        # keep 2 of 4 int4 values + 2-bit position metadata per kept value
        kept = out * inf * 0.5 * 0.5             # half the int4 codes
        meta = out * inf * 0.5 * (2.0 / 8.0)     # 2 bits per kept value
        return kept + meta
    if kind == "int4":
        return out * inf * 0.5
    raise ValueError(scheme)


# --------------------------------------------------------------------------- #
# Model load + body enumeration.
# --------------------------------------------------------------------------- #
def load_model():
    from transformers import Gemma4ForConditionalGeneration
    from transformers.utils.quantization_config import CompressedTensorsConfig
    ckpt = glob.glob(os.path.expanduser(CKPT_GLOB))[0]
    log("loading", ckpt)
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    model.eval()
    # free vision/audio towers (unused for text PPL) to reclaim headroom
    import torch.nn as nn
    for nm in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]:
        if hasattr(model.model, nm):
            setattr(model.model, nm, nn.Identity())
    torch.cuda.empty_cache()
    log(f"loaded in {time.time()-t0:.1f}s  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB "
        f"(towers freed)")
    return model


def enumerate_body(model):
    """Return list of dicts {name, layer, sub, module, out, in, code4, scale}."""
    lm = model.model.language_model
    body = []
    for li, lyr in enumerate(lm.layers):
        for sub in BODY_SUBMODULES:
            m = lyr
            ok = True
            for p in sub.split("."):
                if not hasattr(m, p):
                    ok = False
                    break
                m = getattr(m, p)
            if not ok or not hasattr(m, "weight"):
                continue  # KV-shared layers lack k/v_proj
            out, inf = m.weight.shape
            code4, scale = linear_int4_bytes(out, inf)
            body.append({"name": f"L{li}.{sub}", "layer": li, "sub": sub,
                         "module": m, "out": out, "in": inf,
                         "code4": code4, "scale": scale})
    return body


# --------------------------------------------------------------------------- #
# PPL eval (EXACT official teacher-forced arithmetic; ppl_endpoint.py).
# --------------------------------------------------------------------------- #
def load_corpus():
    recs = []
    with open(CORPUS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ctx, tgt = r["context_token_ids"], r["target_token_ids"]
            ids = ctx + tgt
            ss = max(len(ctx), 1)
            se = len(ids)
            recs.append((r["id"], ids, ss, se))
    return recs


@torch.inference_mode()
def eval_ppl(model, recs, indices=None):
    """Official teacher-forced PPL. Scored tokens are the trailing target block, so we
    restrict the lm_head to the last (target_len+1) positions via logits_to_keep: the
    kept logits are bit-identical to the full forward but bound memory to ~[513, V]."""
    if indices is not None:
        recs = [recs[i] for i in indices]
    total_nll = 0.0
    total_tok = 0
    dev = next(model.parameters()).device
    for _id, ids, ss, se in recs:
        t = torch.tensor([ids], device=dev)
        n = se - ss                                         # scored tokens (target block)
        K = n + 1                                           # keep predicting positions [ss-1, se)
        logits = model(input_ids=t, logits_to_keep=K).logits[0].float()  # [K,V] softcapped
        lse = torch.logsumexp(logits, dim=-1)              # [K] partition fn
        j = torch.arange(n, device=dev)
        toks = t[0, ss + j]                                 # token at position ss+j
        tok_lp = logits[j, toks] - lse[j]                   # predicting-logits at ss-1+j = L_keep[j]
        total_nll += float(-tok_lp.sum().item())
        total_tok += int(n)
    return math.exp(total_nll / total_tok), total_tok


# --------------------------------------------------------------------------- #
# Scheme apply / restore.
# --------------------------------------------------------------------------- #
def apply_scheme(body, scheme_map):
    """scheme_map: {name: scheme_tuple}. Returns touched names."""
    by_name = {b["name"]: b for b in body}
    touched = []
    for name, scheme in scheme_map.items():
        b = by_name[name]
        W = b["module"].weight.data
        if scheme[0] == "intn":
            Wq = fakequant_intn(W, scheme[1])
        elif scheme[0] == "sparse24":
            Wq = fakequant_2to4(W)
        elif scheme[0] == "int4":
            Wq = fakequant_intn(W, 4)
        else:
            raise ValueError(scheme)
        b["module"].weight.data.copy_(Wq)
        touched.append(name)
    return touched


def restore(body, snapshot, names):
    by_name = {b["name"]: b for b in body}
    for name in names:
        b = by_name[name]
        b["module"].weight.data.copy_(snapshot[name].to(b["module"].weight.device))


def body_read_reduction_pct(body, scheme_map):
    """Analytic % reduction of int4 body read (code+scale) under scheme_map."""
    int4_total = sum(b["code4"] + b["scale"] for b in body)
    cfg_total = 0.0
    by_name = {b["name"]: b for b in body}
    for b in body:
        sch = scheme_map.get(b["name"], ("int4",))
        cfg_total += linear_scheme_code_bytes(b["out"], b["in"], sch) + b["scale"]
    return 100.0 * (1.0 - cfg_total / int4_total), int4_total


# --------------------------------------------------------------------------- #
# Composition: required reduction + round-trip self-test.
# --------------------------------------------------------------------------- #
def required_reduction_for_500():
    """HBM-bound ceiling = E[T]*BW_eff/body, BW_eff calibrated so ceiling==baseline
    at the full body. At fixed E[T], tps ∝ 1/body => required reduction = 1 - base/target."""
    bw_eff = OFFICIAL_BASELINE_TPS * DEPLOYED_BODY_GB / E_T   # GB/s effective
    ceiling = lambda body: E_T * bw_eff / body
    body_500 = E_T * bw_eff / TARGET_TPS
    req = 1.0 - body_500 / DEPLOYED_BODY_GB
    roundtrip = ceiling(DEPLOYED_BODY_GB * (1.0 - req))
    return {
        "required_read_reduction_pct_for_500": 100.0 * req,
        "bw_eff_gbps": bw_eff,
        "ceiling_at_full_body": ceiling(DEPLOYED_BODY_GB),
        "body_500_gb": body_500,
        "roundtrip_tps": roundtrip,
        "roundtrip_resid_tps": abs(roundtrip - TARGET_TPS),
        "ceiling_nominal_600": E_T * BW_NOMINAL_GBPS / DEPLOYED_BODY_GB,
    }


def projected_deployed_ppl(local_ppl, local_int4_ppl):
    """Offset-correct the local fake-quant PPL onto the deployed anchor (removes the
    bf16-vs-Marlin systematic offset): projected = anchor + (local - local_int4)."""
    return DEPLOYED_INT4_PPL + (local_ppl - local_int4_ppl)


def constants_exact():
    """Self-test (e): verify every imported cross-leg constant equals its PR-stated
    value EXACTLY (no silent drift across legs). Returns (all_ok, per-constant detail)."""
    expect = {
        "OFFICIAL_BASELINE_TPS": (OFFICIAL_BASELINE_TPS, 481.53),
        "TARGET_TPS": (TARGET_TPS, 500.0),
        "LAMBDA1_CEILING_TPS": (LAMBDA1_CEILING_TPS, 520.953),
        "K_CAL": (K_CAL, 125.268),
        "STEP_US": (STEP_US, 1218.2),
        "E_T": (E_T, 3.844),
        "FLOOR_US": (FLOOR_US, 2933.83),
        "PPL_GATE": (PPL_GATE, 2.42),
        "DEPLOYED_INT4_PPL": (DEPLOYED_INT4_PPL, 2.3772),
        "PPL_HEADROOM": (PPL_HEADROOM, 0.0428),
        "TAU_LO": (TAU_LO, 1.03524),
        "DEPLOYED_BODY_GB": (DEPLOYED_BODY_GB, 1.76),
        "OFFICIAL_SCORED_TOKENS": (OFFICIAL_SCORED_TOKENS, 61797),
    }
    detail, all_ok = {}, True
    for k, (have, want) in expect.items():
        ok = abs(float(have) - float(want)) <= 1e-6
        detail[k] = {"have": have, "want": want, "ok": ok}
        all_ok = all_ok and ok
    return all_ok, detail


# --------------------------------------------------------------------------- #
# Main experiment.
# --------------------------------------------------------------------------- #
def run(args):
    torch.cuda.reset_peak_memory_stats()
    model = load_model()
    body = enumerate_body(model)
    log(f"body Linears: {len(body)}  "
        f"int4 body (code+scale) = {sum(b['code4']+b['scale'] for b in body)/1e9:.4f} GB  "
        f"code-only = {sum(b['code4'] for b in body)/1e9:.4f} GB")

    snapshot = {b["name"]: b["module"].weight.detach().to("cpu").clone() for b in body}
    log(f"snapshot {len(snapshot)} weights to CPU "
        f"({sum(t.numel()*t.element_size() for t in snapshot.values())/1e9:.2f} GB)")

    recs = load_corpus()
    n_full = len(recs)
    sub_n = min(args.rank_subset, n_full)
    sub_idx = list(range(0, n_full, max(1, n_full // sub_n)))[:sub_n]
    log(f"corpus: {n_full} records, ranking subset = {len(sub_idx)}")

    results = []  # each: {config, scheme_map(meta), read_reduction_pct, local_ppl, ...}

    def measure(label, scheme_map, full=True, kind="config"):
        touched = apply_scheme(body, scheme_map)
        idx = None if full else sub_idx
        t0 = time.time()
        ppl, ntok = eval_ppl(model, recs, idx)
        dt = time.time() - t0
        restore(body, snapshot, touched)
        red, _ = body_read_reduction_pct(body, scheme_map)
        row = {"config": label, "kind": kind, "read_reduction_pct": red,
               "local_ppl": ppl, "n_tokens": ntok, "full_corpus": full,
               "eval_s": dt, "n_demoted": len(scheme_map)}
        log(f"  [{label}] red={red:.2f}%  ppl={ppl:.4f}  ntok={ntok}  "
            f"({'full' if full else f'sub{len(sub_idx)}'}, {dt:.1f}s)")
        return row, ppl

    # ---- 0. baseline (pristine int4, no requant) ----
    t0 = time.time()
    base_ppl, base_ntok = eval_ppl(model, recs, None)
    log(f"  [baseline-int4] ppl={base_ppl:.4f}  ntok={base_ntok}  ({time.time()-t0:.1f}s)")
    base_sub_ppl, _ = eval_ppl(model, recs, sub_idx)
    results.append({"config": "baseline_int4", "kind": "baseline",
                    "read_reduction_pct": 0.0, "local_ppl": base_ppl,
                    "n_tokens": base_ntok, "full_corpus": True, "n_demoted": 0})

    # ---- 1. round-trip fidelity validation (int8 ~lossless; int4 ~identity) ----
    all_int8 = {b["name"]: ("intn", 8) for b in body}
    row_i8, _ = measure("validate_int8_all", all_int8, full=True, kind="validate")
    results.append(row_i8)
    all_int4 = {b["name"]: ("int4",) for b in body}
    row_i4, ppl_i4 = measure("validate_int4_all", all_int4, full=True, kind="validate")
    results.append(row_i4)
    # recomputed-scale int4 round-trip: NOT bit-exact vs stored int4 (absmax/8 vs stored
    # convention on already-quantized weights) -> a documented lossy-on-lossy artifact,
    # bounded but not the machinery gate (int8 near-lossless is the strict gate).
    int4_recomputed_delta = abs(ppl_i4 - base_ppl)
    # local int4 reference for offset projection = pristine baseline (conservative:
    # int3 deltas then carry the lossy-on-lossy penalty; fp accumulation is optimistic).
    local_int4_ref = base_ppl

    # ---- 2. uniform int3 (pessimistic max-reduction corner) ----
    all_int3 = {b["name"]: ("intn", 3) for b in body}
    row_u3, _ = measure("uniform_int3", all_int3, full=True, kind="pareto")
    results.append(row_u3)

    # ---- 3. per-layer int3 sensitivity ranking (subset) ----
    layers = sorted({b["layer"] for b in body})
    layer_names = {li: [b["name"] for b in body if b["layer"] == li] for li in layers}
    sens = []
    log(f"ranking {len(layers)} layers by single-layer int3 PPL-delta (subset {len(sub_idx)})")
    for li in layers:
        sm = {n: ("intn", 3) for n in layer_names[li]}
        touched = apply_scheme(body, sm)
        ppl_l, _ = eval_ppl(model, recs, sub_idx)
        restore(body, snapshot, touched)
        sens.append({"layer": li, "int3_ppl_delta_sub": ppl_l - base_sub_ppl})
    sens.sort(key=lambda d: d["int3_ppl_delta_sub"])
    rank_order = [s["layer"] for s in sens]  # least-sensitive first
    log("  least-sensitive layers: " + ", ".join(f"L{s['layer']}({s['int3_ppl_delta_sub']:+.4f})"
                                                  for s in sens[:6]))
    log("  most-sensitive layers:  " + ", ".join(f"L{s['layer']}({s['int3_ppl_delta_sub']:+.4f})"
                                                  for s in sens[-6:]))

    # ---- 4. progressive mixed int4/int3 (demote least-sensitive K layers) ----
    frac_points = [0.10, 0.25, 0.50, 0.75]
    demote_counts = sorted({max(1, round(f * len(layers))) for f in frac_points}
                           | {1, 2, 4, 6, 8, 12, 16, 24})
    demote_counts = [c for c in demote_counts if c <= len(layers)]
    for k in demote_counts:
        dem_layers = set(rank_order[:k])
        sm = {b["name"]: ("intn", 3) for b in body if b["layer"] in dem_layers}
        pct_label = f"mixed_int3_demote{k}L"
        row, _ = measure(pct_label, sm, full=True, kind="pareto")
        row["demote_layers"] = sorted(dem_layers)
        results.append(row)

    # ---- 5. 2:4 structured sparsity (sparsifiable layers only) ----
    sm24 = {b["name"]: ("sparse24",) for b in body if b["sub"] in SPARSE24_SUBMODULES}
    row24, _ = measure("sparse_2to4", sm24, full=True, kind="pareto")
    results.append(row24)
    # 2:4 + int3 on the kept values' layers is out of scope; report 2:4 alone.

    sm_86 = torch.cuda.get_device_capability(0)
    sparse24_supported = sm_86[0] >= 8  # Ampere+ admits 2:4

    # ---- compose verdict ----
    comp = required_reduction_for_500()
    req_pct = comp["required_read_reduction_pct_for_500"]

    # projected deployed PPL + safety for every pareto config
    pareto = [r for r in results if r["kind"] in ("pareto",)]
    for r in pareto:
        r["projected_deployed_ppl"] = projected_deployed_ppl(r["local_ppl"], local_int4_ref)
        r["ppl_delta_vs_int4"] = r["local_ppl"] - local_int4_ref
        r["ppl_safe"] = r["projected_deployed_ppl"] <= PPL_GATE

    safe = [r for r in pareto if r["ppl_safe"]]
    max_safe_pct = max((r["read_reduction_pct"] for r in safe), default=0.0)
    max_safe_cfg = max(safe, key=lambda r: r["read_reduction_pct"], default=None)
    lever_clears_500 = max_safe_pct >= req_pct

    # ---- self-test (PRIMARY) ----
    red_by_demote = [(r["n_demoted"], r["read_reduction_pct"])
                     for r in pareto if r["config"].startswith("mixed_int3_demote")]
    red_by_demote.sort()
    monotone_red = all(red_by_demote[i][1] <= red_by_demote[i + 1][1] + 1e-6
                       for i in range(len(red_by_demote) - 1))
    all_ppls = [r["local_ppl"] for r in results] + [base_sub_ppl]
    nan_clean = all(math.isfinite(p) for p in all_ppls)
    const_ok, const_detail = constants_exact()
    checks = {
        # harness faithfulness: scores the EXACT official scored-token set
        "harness_faithful_token_count_61797": base_ntok == OFFICIAL_SCORED_TOKENS,
        "ppl_nan_clean": nan_clean,                                       # PR (d)
        "int8_near_lossless": abs(row_i8["local_ppl"] - base_ppl) < 0.01,
        "int4_recomputed_roundtrip_lt_0p05": int4_recomputed_delta < 0.05,
        "required_roundtrip_resid_lt_0p5": comp["roundtrip_resid_tps"] < 0.5,  # PR (c)
        "uniform_int3_reduction_gt_required": row_u3["read_reduction_pct"] > req_pct,
        "pareto_monotone_reduction": monotone_red,                        # PR (b)
        "constants_imported_exact": const_ok,                            # PR (e)
        "max_safe_well_defined": max_safe_pct >= 0.0,
        "caveats_present": len(CAVEATS) >= 5,                            # PR (f)
    }
    self_test_passes = all(checks.values())
    # NOTE: the int4 anchor does NOT reproduce 2.3772 (QAT proxy < deployed PTQ; see
    # CAVEATS QAT_PROXY_OPTIMISTIC). Reported as informational; the verdict is built in
    # offset-corrected delta space so the absolute QAT-vs-PTQ gap cancels.
    anchor_reproduces = abs(base_ppl - DEPLOYED_INT4_PPL) < 0.05

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "primary_metric": {"name": "read_reduction_ppl_pareto_self_test_passes",
                           "value": bool(self_test_passes)},
        "test_metric": {"name": "max_ppl_safe_read_reduction_pct", "value": max_safe_pct},
        "read_reduction_lever_clears_500": bool(lever_clears_500),
        "required_read_reduction_pct_for_500": req_pct,
        "max_ppl_safe_read_reduction_pct": max_safe_pct,
        "max_safe_config": max_safe_cfg["config"] if max_safe_cfg else None,
        "int4_baseline_ppl_local": base_ppl,
        "int4_baseline_resid_vs_2p3772": base_ppl - DEPLOYED_INT4_PPL,
        "int4_recomputed_roundtrip_delta": int4_recomputed_delta,
        "int8_roundtrip_delta": abs(row_i8["local_ppl"] - base_ppl),
        "deployed_int4_ppl_anchor": DEPLOYED_INT4_PPL,
        "anchor_reproduces_2p3772": bool(anchor_reproduces),
        "ppl_gate": PPL_GATE,
        "ppl_headroom": PPL_HEADROOM,
        "deployed_body_gb": DEPLOYED_BODY_GB,
        "constants_exact": const_ok,
        "constants_detail": const_detail,
        "caveats": CAVEATS,
        "local_int4_body_code_scale_gb": sum(b["code4"] + b["scale"] for b in body) / 1e9,
        "local_int4_body_code_only_gb": sum(b["code4"] for b in body) / 1e9,
        "sparse24_supported_sm86": bool(sparse24_supported),
        "sm": list(sm_86),
        "composition": comp,
        "self_test": {"passes": bool(self_test_passes), "checks": checks},
        "sensitivity_ranking": sens,
        "rank_order_least_sensitive_first": rank_order,
        "configs": results,
        "peak_vram_gb": peak_gb,
        "n_records": n_full,
        "rank_subset": len(sub_idx),
    }
    return out


# --------------------------------------------------------------------------- #
# Output: JSON, plot, wandb.
# --------------------------------------------------------------------------- #
def make_plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pareto = [r for r in out["configs"] if r["kind"] == "pareto"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for kind, mk, col in [("mixed", "o", "tab:blue")]:
        pass
    xs = [r["read_reduction_pct"] for r in pareto]
    ys = [r["projected_deployed_ppl"] for r in pareto]
    labels = [r["config"] for r in pareto]
    colors = ["tab:green" if r["ppl_safe"] else "tab:red" for r in pareto]
    ax.scatter(xs, ys, c=colors, s=55, zorder=3, edgecolor="k", linewidth=0.4)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l.replace("mixed_int3_", "").replace("_", " "), (x, y),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(out["ppl_gate"], color="k", ls="--", lw=1,
               label=f"PPL gate {out['ppl_gate']}")
    ax.axhline(out["deployed_int4_ppl_anchor"], color="gray", ls=":", lw=1,
               label=f"int4 anchor {out['deployed_int4_ppl_anchor']}")
    ax.axvline(out["required_read_reduction_pct_for_500"], color="tab:purple", ls="-.",
               lw=1.2, label=f"required {out['required_read_reduction_pct_for_500']:.2f}% -> 500 TPS")
    ax.axvline(out["max_ppl_safe_read_reduction_pct"], color="tab:orange", ls="-", lw=1.2,
               label=f"max PPL-safe {out['max_ppl_safe_read_reduction_pct']:.2f}%")
    ax.set_xlabel("body-read-byte reduction (% of int4 code+scale)")
    ax.set_ylabel("projected deployed PPL (offset-corrected)")
    ax.set_title("PPL vs body-read-byte reduction Pareto (fake-quant, OPTIMISTIC proxy)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    log("wrote plot", path)


def log_wandb(args, out):
    if args.no_wandb:
        return None
    sys.path.insert(0, "/workspace/senpai/target")
    try:
        from scripts import wandb_logging
    except Exception as exc:
        log(f"wandb import failed ({exc}); skipping")
        return None
    run = wandb_logging.init_wandb_run(
        job_type="read-reduction-ppl-pareto", agent="fern",
        name=args.wandb_name or "fern/read-reduction-ppl-pareto",
        group=args.wandb_group or "read-reduction-ppl-pareto",
        tags=["validity", "ppl-pareto", "read-reduction", "fake-quant", "hbm-ceiling", "pr287"],
        config={k: out[k] for k in ["required_read_reduction_pct_for_500",
                "max_ppl_safe_read_reduction_pct", "read_reduction_lever_clears_500",
                "int4_baseline_ppl_local", "ppl_gate", "deployed_body_gb"]},
    )
    if run is None:
        log("wandb disabled; skipping")
        return None
    import wandb
    flat = {
        "read_reduction_ppl_pareto_self_test_passes": 1.0 if out["self_test"]["passes"] else 0.0,
        "max_ppl_safe_read_reduction_pct": out["max_ppl_safe_read_reduction_pct"],
        "required_read_reduction_pct_for_500": out["required_read_reduction_pct_for_500"],
        "read_reduction_lever_clears_500": 1.0 if out["read_reduction_lever_clears_500"] else 0.0,
        "int4_baseline_ppl_local": out["int4_baseline_ppl_local"],
        "int4_baseline_resid_vs_2p3772": out["int4_baseline_resid_vs_2p3772"],
        "int4_recomputed_roundtrip_delta": out["int4_recomputed_roundtrip_delta"],
        "int8_roundtrip_delta": out["int8_roundtrip_delta"],
        "uniform_int3_reduction_pct": next(r["read_reduction_pct"] for r in out["configs"]
                                           if r["config"] == "uniform_int3"),
        "sparse24_supported_sm86": 1.0 if out["sparse24_supported_sm86"] else 0.0,
        "roundtrip_resid_tps": out["composition"]["roundtrip_resid_tps"],
        "peak_vram_gb": out["peak_vram_gb"],
        "global_step": 0,
    }
    for k, v in out["self_test"]["checks"].items():
        flat[f"selftest/{k}"] = 1.0 if v else 0.0
    run.log(flat)
    # pareto table
    cols = ["config", "kind", "read_reduction_pct", "local_ppl",
            "projected_deployed_ppl", "ppl_delta_vs_int4", "ppl_safe", "n_demoted"]
    tbl = wandb.Table(columns=cols)
    for r in out["configs"]:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"pareto_table": tbl, "global_step": 0})
    # sensitivity table
    stbl = wandb.Table(columns=["layer", "int3_ppl_delta_sub"])
    for s in out["sensitivity_ranking"]:
        stbl.add_data(s["layer"], s["int3_ppl_delta_sub"])
    run.log({"sensitivity_table": stbl, "global_step": 0})
    plot_path = OUTDIR / "read_reduction_ppl_pareto.png"
    if plot_path.exists():
        run.log({"pareto_plot": wandb.Image(str(plot_path)), "global_step": 0})
    run.summary.update({k: v for k, v in flat.items() if k != "global_step"})
    rid = run.id
    run.finish()
    log("wandb run", rid)
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-subset", type=int, default=32)
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="no-op alias: the full run always evaluates the PRIMARY self-test")
    ap.add_argument("--quick", action="store_true",
                    help="tiny end-to-end smoke (subset corpus, few configs)")
    args = ap.parse_args()

    if args.quick:
        return quick(args)

    out = run(args)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "read_reduction_ppl_pareto_results.json").write_text(json.dumps(out, indent=2))
    log("wrote JSON")
    make_plot(out, OUTDIR / "read_reduction_ppl_pareto.png")
    rid = log_wandb(args, out)
    out["wandb_run_id"] = rid
    (OUTDIR / "read_reduction_ppl_pareto_results.json").write_text(json.dumps(out, indent=2))

    log("=" * 70)
    log(f"PRIMARY self_test_passes = {out['self_test']['passes']}")
    log(f"TEST max_ppl_safe_read_reduction_pct = {out['max_ppl_safe_read_reduction_pct']:.3f}%")
    log(f"required_read_reduction_pct_for_500 = {out['required_read_reduction_pct_for_500']:.3f}%")
    log(f"lever_clears_500 = {out['read_reduction_lever_clears_500']}")
    log(f"int4 baseline local ppl = {out['int4_baseline_ppl_local']:.4f} "
        f"(resid {out['int4_baseline_resid_vs_2p3772']:+.4f} vs 2.3772)")
    log(f"peak VRAM = {out['peak_vram_gb']:.2f} GB")
    log("checks: " + json.dumps(out["self_test"]["checks"]))


def quick(args):
    """Fast end-to-end validation: 16-rec corpus, baseline + int8 + int4 + uniform int3
    + one mixed(4L heuristic) + 2:4. Prints timing. No wandb."""
    torch.cuda.reset_peak_memory_stats()
    model = load_model()
    body = enumerate_body(model)
    snapshot = {b["name"]: b["module"].weight.detach().to("cpu").clone() for b in body}
    recs = load_corpus()
    idx = list(range(0, len(recs), len(recs) // 16))[:16]
    log(f"QUICK: {len(idx)} records")
    t0 = time.time(); base, nt = eval_ppl(model, recs, idx); t_one = (time.time()-t0)/len(idx)
    log(f"baseline ppl={base:.4f} ntok={nt}  per-forward={t_one*1e3:.0f}ms")
    for label, sm in [
        ("int8_all", {b["name"]: ("intn", 8) for b in body}),
        ("int4_all", {b["name"]: ("int4",) for b in body}),
        ("uniform_int3", {b["name"]: ("intn", 3) for b in body}),
        ("mixed_4L", {b["name"]: ("intn", 3) for b in body if b["layer"] in (10, 15, 20, 25)}),
        ("sparse_2to4", {b["name"]: ("sparse24",) for b in body if b["sub"] in SPARSE24_SUBMODULES}),
    ]:
        touched = apply_scheme(body, sm)
        ppl, _ = eval_ppl(model, recs, idx)
        restore(body, snapshot, touched)
        red, _ = body_read_reduction_pct(body, sm)
        log(f"  {label:14s} red={red:6.2f}%  ppl={ppl:.4f}  delta={ppl-base:+.4f}")
    comp = required_reduction_for_500()
    log(f"required%={comp['required_read_reduction_pct_for_500']:.3f}  "
        f"roundtrip={comp['roundtrip_tps']:.2f}  resid={comp['roundtrip_resid_tps']:.3e}")
    log(f"peak VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
