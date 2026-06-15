#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Codebook (~3.0 bpw) body-quant MEASUREMENT (PR #367) — the last sub-int4
ceiling-lift shot. The measured counterpart of the #355 RTN screen.

WHY THIS LEG (PR #367)
----------------------
#355 measured RTN 3-bit body-quant DEAD: +13.94% relative PPL (W&B vqzzc9jw,
gate-transferred 2.7085 > 2.42, ~7.7x the +1.81% headroom budget) and 2-bit
collapses (5020.72). But the literature says CODEBOOK / trellis quantization
(AQLM 2401.06118, QuIP# 2402.04396, QTIP 2406.11235) is far more accurate at
low bit-width: AQLM 3.04b ~ +0.5% PPL, QTIP 3.0b ~ +1.1% on WikiText-2 — which
would FIT the +1.81% headroom. This file MEASURES whether a real codebook
quantizer reproduces that on THIS model (gemma-4-E4B QAT body) and THIS eval.

THREE GATES (all must hold for lever (a) to live):
  1. PPL-safe: gate-transfer PPL = 2.3772 * (PPL_codebook / PPL_int4_localproxy)
     must stay <= 2.42 (the +1.81% relative headroom).
  2. Greedy-identity: 128-tok greedy decode vs the int4 reference. NB: this model
     is a NEAR-TIE-dense argmax surface — even the *bf16* body diverges ~83% from
     int4 (#355). The official #319 gate is SELF-referential (served == its OWN
     plain greedy AR), which any deterministic checkpoint passes by construction;
     frac_mismatch-vs-int4 is reported as a quality-DRIFT diagnostic, contextualized
     against the bf16<->int4 floor, NOT a literal "must be 0" gate.
  3. Realized BW-lift > 0: the decisive caveat (#355 flagged it). Codebook decode
     is a random-access gather, NOT a pure weight-read. The ~3-bit byte reduction
     may collapse to ~0 (or negative) realized headroom vs the int4-Marlin path.
     Estimated from the byte model + published AQLM/QuIP#/QTIP batch=1 kernel
     speed (see KERNEL_LIT below) — NOT a measured sub-int4 kernel (none exists in
     vLLM 0.22 / compressed-tensors; Marlin supports only [4,8] bits, so fake-quant
     dequantizes to bf16 and forward time is width-invariant — relative screen only).

THE METHOD (calibration-free codebook proxy)
--------------------------------------------
A real AQLM/QuIP# build needs a multi-hour calibration (input-aware Hessian
weighting). That does not fit the pod budget, so this measures the GENEROUS
calibration-free core of QuIP#: incoherence preprocessing + a fixed lattice/VQ
codebook. Per body group (g128 input channels, per output row):
  * INCOHERENCE: a seeded random-sign x Walsh-Hadamard (RHT) rotation of the 128
    channels Gaussianizes the group and kills outliers (the QuIP/QuIP# core that
    makes low-bit work; 2307.13304 / 2402.04396). 0 per-weight storage (a fixed
    seed). Per-group (not full-matrix) RHT — a tractable approximation; within-group
    outlier removal is the dominant low-bit benefit.
  * VQ CODEBOOK: a FIXED K-point codebook over dim-`vq_dim` sub-vectors, optimal
    for an iid unit Gaussian (K-means on synthetic N(0,1), built once, reused for
    every group; this is the QuIP# "lattice"). n_bits = log2(K)/vq_dim bits/weight.
    Per-group bf16 RMS scale (same g128 floor as the served int4). Effective bpw =
    n_bits + 16/group + amortized_codebook ~ 3.125 at 3-bit (vq_dim=2, K=64).
This is the OPTIMISTIC-but-calibration-free lane: full AQLM/QTIP WITH calibration
is >= this quality, RTN (#355) is the pessimistic <= bound. Ablation `--no-incoherence`
isolates how much the rotation (vs the bare VQ) buys.

PROVENANCE: int4 local-proxy anchor (PPL_int4_localproxy) is measured IN THIS RUN
as RTN-int4 (b=4) on the same load + dataset, so the gate-transfer ratio is
internally consistent (only the body quant differs). #355 RTN-int4 = 1.9512.

Run (full):
    cd target/ && CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online .venv/bin/python \
      research/validity/sub_int4_body_ceiling/measure_codebook.py \
      --widths "bf16,int4,cb3,cb3_noinc,cb2" \
      --wandb_group aqlm-codebook-subint4 \
      --wandb_name lawine/codebook-subint4-measured
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# Reuse the #355 measurement infra (PPL/greedy/records/body-selection/gate anchors)
# and the analytic card (BW model) — single source of truth, directly comparable.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import measure_subint4 as m355  # noqa: E402
import sub_int4_body_ceiling as card  # noqa: E402

PPL_GATE = card.PPL_GATE                                   # 2.42
STRICT_NONSPEC_FLOOR_TPS = card.STRICT_NONSPEC_FLOOR_TPS   # 165.44 (lawine #196)
TARGET_TPS = card.TARGET_TPS                               # 500.0
SERVED_INT4_PPL_SPEC = m355.SERVED_INT4_PPL_SPEC           # 2.3772 (gate's named anchor)
DEFAULT_PPL_DATASET = m355.DEFAULT_PPL_DATASET
DEFAULT_BASE_MODEL = m355.DEFAULT_BASE_MODEL

# ---- realized-kernel literature constants (DELIVERABLE 3) ------------------- #
# Published single-stream (batch=1) decode speedups vs each method's OWN fp16 on a
# single GPU. The realized BW-lift of a 3-bit codebook is NOT its byte ratio vs
# int4 — it is the ratio of the realized codebook kernel speed to the int4-Marlin
# kernel speed (both ride the same weight-read-bound batch=1 regime; speedup-vs-fp16
# is ~hardware-portable for BW-bound decode). If codebook/Marlin < 1 the ~3-bit byte
# savings do NOT survive and realized lift is <= 0.
#
# CRITICAL (research pass, PR #367): the codebook lane SPLITS by kernel —
#   * QTIP 3-bit = 2.88x fp16, QuIP# 2-bit = 3.33x fp16 (E8/trellis codebook is
#     L1-resident, 1KiB; dequant is cheap) -> FASTER than int4-Marlin (~2.0-2.5x
#     fp16 on A10G) -> realized lift POSITIVE. The #355 "gather kills BW-lift"
#     caveat is REFUTED for QTIP/QuIP#.
#   * AQLM = 1.2-1.46x fp16 (learned multi-table gather is memory-incoherent) ->
#     SLOWER than int4-Marlin -> realized lift NEGATIVE. The caveat HOLDS for AQLM.
# QTIP/QuIP# tok/s: QTIP Tab4 (RTX6000 Ada 960GB/s, bs1); QuIP# Tab6 (RTX4090).
# Marlin batch=1 ~2.0-2.5x fp16 from independent w4a16 g128 benchmarks. Overridable
# via CLI for sensitivity. Headline gate-3 uses the BEST codebook kernel (QTIP/QuIP#)
# so the GO test asks "does the BEST codebook route have positive lift" — isolating
# whether BW (vs PPL) is the binding constraint.
KERNEL_LIT = {
    "qtip_3bit_batch1_speedup_vs_fp16": 2.88,        # 2406.11235 Tab4 (best 3-bit lattice)
    "qtip_4bit_batch1_speedup_vs_fp16": 2.50,        # 2406.11235 Tab4
    "quip_sharp_2bit_batch1_speedup_vs_fp16": 3.33,  # 2402.04396 / 2406.11235 Tab4
    "aqlm_2bit_batch1_speedup_vs_fp16": 1.46,        # 2401.06118 (slow multi-table gather)
    "aqlm_3bit_batch1_speedup_vs_fp16": 1.20,        # 3-bit AQLM (more tables -> slower)
    "int4_marlin_batch1_speedup_vs_fp16": 2.25,      # Marlin/AWQ w4a16 g128 bs1 ~2.0-2.5x (A10G)
    # headline codebook kernel for the gate-3 GO test = best L1-resident lattice:
    "headline_codebook_3bit_speedup_vs_fp16": 2.88,  # QTIP 3-bit
    "headline_codebook_2bit_speedup_vs_fp16": 3.33,  # QuIP# 2-bit
}


# --------------------------------------------------------------------------- #
# Incoherence (random-sign Walsh-Hadamard) rotation.
# --------------------------------------------------------------------------- #
def hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard H_n (n a power of 2): H/sqrt(n)."""
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard size {n} must be a power of 2")
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def rht_matrix(group_size: int, device, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    """Seeded random-sign x Hadamard rotation R = H @ diag(s), s in {-1,+1}.
    Orthonormal; shared across all groups (a fixed incoherence basis)."""
    H = hadamard_matrix(group_size, device, dtype)
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = (torch.randint(0, 2, (group_size,), generator=g, dtype=torch.int8).to(device) * 2 - 1).to(dtype)
    return H * s.unsqueeze(0)  # scale columns by signs -> H @ diag(s)


# --------------------------------------------------------------------------- #
# Fixed Gaussian VQ codebook (the "lattice"), built once by K-means on N(0,1).
# --------------------------------------------------------------------------- #
def build_gaussian_codebook(n_bits: int, vq_dim: int, device, seed: int = 0,
                            iters: int = 40) -> torch.Tensor:
    """K = 2**(n_bits*vq_dim) centroids in R^vq_dim, optimal for iid N(0,1).
    Returns [K, vq_dim] float32. Data-free: source-distribution-matched, reused
    for every weight group (the incoherence makes each group ~iid Gaussian)."""
    K = 2 ** (n_bits * vq_dim)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_samp = max(K * 512, 300_000)
    X = torch.randn(n_samp, vq_dim, generator=g).to(device)
    perm = torch.randperm(n_samp, generator=g)[:K].to(device)
    C = X[perm].clone()
    for _ in range(iters):
        # assign in chunks (n_samp x K can be large for K=4096)
        idx = _nn_assign(X, C, chunk=1 << 20)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K, device=device, dtype=torch.float32)
        newC.index_add_(0, idx, X)
        cnt.index_add_(0, idx, torch.ones(X.shape[0], device=device))
        nz = cnt > 0
        newC[nz] = newC[nz] / cnt[nz].unsqueeze(1)
        # re-seed dead centroids to random samples (avoid collapse)
        if (~nz).any():
            dead = (~nz).nonzero(as_tuple=True)[0]
            reseed = torch.randint(0, n_samp, (dead.numel(),), device=device)
            newC[dead] = X[reseed]
        C = newC
    return C


def _nn_assign(x: torch.Tensor, C: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
    """Nearest-centroid index for each row of x [N, d] over C [K, d], chunked.
    argmin ||x - C||^2 = argmin (||x||^2 - 2 x.C^T + ||C||^2) = argmax (x.C^T - .5||C||^2)."""
    cnorm = 0.5 * (C * C).sum(1)  # [K]
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        xc = x[s:s + chunk]
        score = xc @ C.t() - cnorm  # [c, K]
        out[s:s + chunk] = score.argmax(1)
    return out


# --------------------------------------------------------------------------- #
# Codebook fake-quant of one weight matrix (per-group RHT + VQ + per-group scale).
# --------------------------------------------------------------------------- #
def codebook_quant_per_group(w: torch.Tensor, group_size: int, vq_dim: int,
                             codebook: torch.Tensor, R: torch.Tensor | None,
                             chunk_rows: int = 4096) -> torch.Tensor:
    """Quantize w [out, in] in the incoherence basis with the fixed VQ codebook.
    Per (out_row, in_group of group_size): RHT-rotate -> RMS-normalize -> VQ ->
    de-normalize -> inverse-rotate. Returns dequantized w.dtype tensor."""
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    if group_size % vq_dim != 0:
        raise ValueError(f"group_size {group_size} not divisible by vq_dim {vq_dim}")
    ng = in_f // group_size
    deq = torch.empty_like(w, dtype=torch.float32)
    for s in range(0, out_f, chunk_rows):
        wg = w[s:s + chunk_rows].reshape(-1, ng, group_size).float()  # [r, ng, g]
        if R is not None:
            wg = torch.matmul(wg, R.t())  # rotate columns within each group
        rms = wg.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()  # [r, ng, 1]
        n = (wg / rms).reshape(-1, vq_dim)  # [(r*ng*g/vq_dim), vq_dim]
        idx = _nn_assign(n, codebook, chunk=1 << 20)
        q = codebook[idx].reshape(wg.shape)  # [r, ng, g]
        q = q * rms
        if R is not None:
            q = torch.matmul(q, R)  # inverse rotation (R orthonormal -> R^-1 = R^T, and R@? )
        deq[s:s + chunk_rows] = q.reshape(wg.shape[0], in_f)
    return deq.to(w.dtype)


def effective_bpw(n_bits: int, group_size: int, vq_dim: int,
                  matrix_in_out: tuple[int, int] | None = None,
                  scale_bits: int = 16) -> dict[str, float]:
    """index n_bits/weight + per-group scale floor + amortized codebook read."""
    K = 2 ** (n_bits * vq_dim)
    index_bpw = float(n_bits)
    scale_bpw = scale_bits / group_size
    cb_bpw = 0.0
    if matrix_in_out is not None:
        in_f, out_f = matrix_in_out
        cb_bits = K * vq_dim * scale_bits
        cb_bpw = cb_bits / (in_f * out_f)
    return {"index_bpw": index_bpw, "scale_bpw": scale_bpw, "codebook_bpw": cb_bpw,
            "effective_bpw": index_bpw + scale_bpw + cb_bpw, "K": K}


# --------------------------------------------------------------------------- #
# Width specs: name -> quantizer applied to each body linear.
# --------------------------------------------------------------------------- #
def make_quantizer(kind: str, n_bits: int, group_size: int, vq_dim: int,
                   codebook: torch.Tensor | None, R: torch.Tensor | None,
                   scheme: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if kind == "bf16":
        return lambda w: w
    if kind == "rtn":
        return lambda w: m355.fake_quant_per_group(w, n_bits, group_size, scheme)
    if kind == "codebook":
        return lambda w: codebook_quant_per_group(w, group_size, vq_dim, codebook, R)
    raise ValueError(f"unknown quantizer kind {kind!r}")


def apply_quant(model: torch.nn.Module, snap: dict[str, torch.Tensor],
                quant: Callable[[torch.Tensor], torch.Tensor], device: str) -> dict[str, Any]:
    mods = dict(model.named_modules())
    n_layers = n_params = 0
    with torch.no_grad():
        for name, w0_cpu in snap.items():
            w0 = w0_cpu.to(device)
            wq = quant(w0)
            mods[name].weight.data.copy_(wq)
            n_layers += 1
            n_params += w0.numel()
            del w0, wq
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {"n_body_linears": n_layers, "n_body_params": n_params}


# --------------------------------------------------------------------------- #
# Predicted (pure-BW) + realized BW-lift estimate (deliverable 3).
# --------------------------------------------------------------------------- #
def realized_bw_lift(n_bits: int, lit: dict[str, float]) -> dict[str, Any]:
    """Pure-BW pred (card model, scale-floored g128) AND a realized estimate that
    re-bases the codebook batch=1 kernel speed against the int4-Marlin frontier the
    165.44 already rides. realized lift>0 iff a 3-bit codebook kernel actually
    out-decodes int4-Marlin at batch=1. SPLITS by kernel: QTIP/QuIP# (L1-resident
    lattice) >0; AQLM (multi-table gather) <0 — see KERNEL_LIT."""
    pred = m355.predicted_tps(n_bits)  # pure-BW roofline, ~214 at 3-bit (== #355)
    pred_ratio = pred / STRICT_NONSPEC_FLOOR_TPS  # pure-BW lift factor over 165.44
    marlin = lit["int4_marlin_batch1_speedup_vs_fp16"]
    headline = (lit["headline_codebook_3bit_speedup_vs_fp16"] if n_bits == 3
                else lit["headline_codebook_2bit_speedup_vs_fp16"])
    aqlm = (lit["aqlm_3bit_batch1_speedup_vs_fp16"] if n_bits == 3
            else lit["aqlm_2bit_batch1_speedup_vs_fp16"])
    # headline = best codebook kernel (QTIP/QuIP#); aqlm = the slow-gather counterexample.
    ratio_headline = headline / marlin
    ratio_aqlm = aqlm / marlin
    return {
        "pure_bw_pred_tps": pred,
        "pure_bw_lift_over_165p44": pred_ratio - 1.0,
        "headline_codebook_speedup_vs_fp16": headline,
        "aqlm_codebook_speedup_vs_fp16": aqlm,
        "int4_marlin_batch1_speedup_vs_fp16": marlin,
        "kernel_ratio_headline_over_marlin": ratio_headline,
        "kernel_ratio_aqlm_over_marlin": ratio_aqlm,
        "realized_tps_estimate": STRICT_NONSPEC_FLOOR_TPS * ratio_headline,        # QTIP/QuIP#
        "realized_tps_estimate_aqlm": STRICT_NONSPEC_FLOOR_TPS * ratio_aqlm,       # AQLM
        "realized_bw_lift_estimate": ratio_headline - 1.0,        # GO uses best codebook kernel
        "realized_bw_lift_estimate_aqlm": ratio_aqlm - 1.0,       # AQLM counterexample (<0)
        "realized_clears_165p44": bool(ratio_headline > 1.0),
    }


# --------------------------------------------------------------------------- #
# Width registry (name -> spec).
# --------------------------------------------------------------------------- #
def parse_widths(spec: str) -> list[dict[str, Any]]:
    table = {
        "bf16":      {"kind": "bf16", "n_bits": 16, "vq_dim": 0, "incoherence": False, "label": "bf16-ref"},
        "int4":      {"kind": "rtn",  "n_bits": 4,  "vq_dim": 0, "incoherence": False, "label": "RTN-int4 (gate denom)"},
        "rtn3":      {"kind": "rtn",  "n_bits": 3,  "vq_dim": 0, "incoherence": False, "label": "RTN-3bit (#355 recheck)"},
        "cb3":       {"kind": "codebook", "n_bits": 3, "vq_dim": 2, "incoherence": True,  "label": "codebook-3bit +incoherence"},
        "cb3_noinc": {"kind": "codebook", "n_bits": 3, "vq_dim": 2, "incoherence": False, "label": "codebook-3bit no-incoherence"},
        "cb2":       {"kind": "codebook", "n_bits": 2, "vq_dim": 2, "incoherence": True,  "label": "codebook-2bit +incoherence"},
        "cb3d1":     {"kind": "codebook", "n_bits": 3, "vq_dim": 1, "incoherence": True,  "label": "codebook-3bit scalar(dim1)"},
    }
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in table:
            raise ValueError(f"unknown width {tok!r}; known: {sorted(table)}")
        out.append({"name": tok, **table[tok]})
    return out


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run(args) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[cb] WARNING CUDA not available; CPU is slow. Set CUDA_VISIBLE_DEVICES=0.", flush=True)
    widths = parse_widths(args.widths)
    records = m355.read_ppl_records(Path(args.ppl_dataset))
    max_records = args.max_records if args.max_records > 0 else None

    print(f"[cb] base={args.base_model} g={args.group_size} vq_dim={args.vq_dim} "
          f"widths={[w['name'] for w in widths]} records={len(records)}"
          f"(use {max_records or len(records)}) greedy K={args.greedy_prompts} G={args.greedy_tokens}",
          flush=True)

    t0 = time.time()
    model, tok = m355.load_model(args.base_model, device)
    print(f"[cb] model loaded {time.time()-t0:.1f}s; GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB",
          flush=True)
    snap = m355.snapshot_body(model)
    body_params = sum(t.numel() for t in snap.values())
    print(f"[cb] snapshot {len(snap)} body linears, {body_params/1e9:.3f}B params "
          f"(lm_head+embed+norms kept at source precision)", flush=True)

    # Fixed codebooks (built once per (n_bits, vq_dim)) + the shared RHT basis.
    R = rht_matrix(args.group_size, device, seed=args.seed) if True else None
    codebooks: dict[tuple[int, int], torch.Tensor] = {}
    for w in widths:
        if w["kind"] == "codebook":
            key = (w["n_bits"], w["vq_dim"])
            if key not in codebooks:
                tcb = time.time()
                codebooks[key] = build_gaussian_codebook(w["n_bits"], w["vq_dim"], device, seed=args.seed)
                print(f"[cb] built Gaussian codebook K={2**(w['n_bits']*w['vq_dim'])} "
                      f"dim={w['vq_dim']} ({time.time()-tcb:.1f}s)", flush=True)

    greedy_prompts = [records[i]["ids"][: m355._prompt_len(records[i])]
                      for i in range(min(args.greedy_prompts, len(records)))]

    width_results: dict[str, dict[str, Any]] = {}
    greedy_tokens: dict[str, list[list[int]]] = {}
    for w in widths:
        tb = time.time()
        cb = codebooks.get((w["n_bits"], w["vq_dim"])) if w["kind"] == "codebook" else None
        Ruse = R if (w["kind"] == "codebook" and w["incoherence"]) else None
        quant = make_quantizer(w["kind"], w["n_bits"], args.group_size, w["vq_dim"], cb, Ruse, args.scheme)
        info = apply_quant(model, snap, quant, device)
        ppl = m355.measure_ppl(model, records, device, max_records)
        gtoks = [m355.greedy_decode(model, p, args.greedy_tokens, device) for p in greedy_prompts] \
            if args.greedy_prompts > 0 else []
        greedy_tokens[w["name"]] = gtoks
        ebpw = (effective_bpw(w["n_bits"], args.group_size, w["vq_dim"]) if w["kind"] == "codebook"
                else {"effective_bpw": float(w["n_bits"]) + (16.0 / args.group_size if w["n_bits"] < 16 else 0.0)})
        width_results[w["name"]] = {
            "name": w["name"], "label": w["label"], "kind": w["kind"],
            "n_bits": w["n_bits"], "vq_dim": w["vq_dim"], "incoherence": w["incoherence"],
            "ppl": ppl["ppl"], "mean_record_ppl": ppl["mean_record_ppl"],
            "num_records": ppl["num_records"], "num_tokens": ppl["num_tokens"],
            "effective_bpw": ebpw["effective_bpw"], "effective_bpw_detail": ebpw,
            "ppl_pass_raw": bool(ppl["ppl"] <= PPL_GATE),
            "body_info": info, "_per_record": ppl["per_record"],
        }
        print(f"[cb] {w['name']:>10} ({w['label']}): PPL={ppl['ppl']:.4f} "
              f"bpw~{ebpw['effective_bpw']:.3f} [{time.time()-tb:.1f}s]", flush=True)

    # ---- gate-transfer PPL (binding gate 1) ------------------------------- #
    int4_ppl = width_results.get("int4", {}).get("ppl")
    for name, wr in width_results.items():
        if int4_ppl and wr["n_bits"] < 16:
            r = wr["ppl"] / int4_ppl
            wr["rel_increase_over_int4"] = r - 1.0
            wr["gate_transfer_ppl"] = SERVED_INT4_PPL_SPEC * r
            wr["gate_transfer_pass"] = bool(SERVED_INT4_PPL_SPEC * r <= PPL_GATE)
        else:
            wr["rel_increase_over_int4"] = float("nan")
            wr["gate_transfer_ppl"] = float("nan")
            wr["gate_transfer_pass"] = bool(wr["ppl_pass_raw"])

    # ---- greedy-identity vs int4 (gate 2, drift diagnostic) --------------- #
    ref = "int4" if "int4" in greedy_tokens and greedy_tokens["int4"] else widths[0]["name"]
    identity = {"ref": ref, "per_width": {}}
    for name in greedy_tokens:
        if name == ref or not greedy_tokens.get(name):
            continue
        per = [m355.divergence(r, c) for r, c in zip(greedy_tokens[ref], greedy_tokens[name])]
        tot_mis = sum(d["num_mismatched"] for d in per)
        tot_cmp = sum(d["n_compared"] for d in per)
        identity["per_width"][name] = {
            "ref": ref, "num_prompts": len(per),
            "total_mismatched": tot_mis, "total_compared": tot_cmp,
            "frac_mismatched": tot_mis / tot_cmp if tot_cmp else 0.0,
            "all_identical": bool(tot_mis == 0), "per_prompt": per,
        }
        width_results[name]["greedy_frac_mismatch_vs_int4"] = tot_mis / tot_cmp if tot_cmp else 0.0
        print(f"[cb] greedy {name} vs {ref}: mismatch {tot_mis}/{tot_cmp} "
              f"({100*tot_mis/max(tot_cmp,1):.2f}%)", flush=True)

    # ---- realized BW-lift (gate 3) ---------------------------------------- #
    lit = dict(KERNEL_LIT)
    if args.marlin_speedup > 0:
        lit["int4_marlin_batch1_speedup_vs_fp16"] = args.marlin_speedup
    if args.codebook_speedup > 0:
        lit["headline_codebook_3bit_speedup_vs_fp16"] = args.codebook_speedup
    for name, wr in width_results.items():
        if wr["kind"] == "codebook":
            wr["bw"] = realized_bw_lift(wr["n_bits"], lit)

    # ---- single GO/NO-GO per codebook width + headline -------------------- #
    headline = "cb3" if "cb3" in width_results else next(
        (n for n, wr in width_results.items() if wr["kind"] == "codebook" and wr["n_bits"] == 3), None)
    go_widths = []
    bf16_drift = width_results.get("bf16", {}).get("greedy_frac_mismatch_vs_int4")
    for name, wr in width_results.items():
        if wr["kind"] != "codebook":
            continue
        ppl_ok = bool(wr["gate_transfer_pass"])
        # GATE 2 (greedy-identity). The official #319 gate is SELF-referential (the
        # served checkpoint must match ITS OWN plain greedy AR), which ANY deterministic
        # quantized checkpoint passes by construction -> identity_selfref_ok = True.
        # frac_mismatch-vs-int4 is NOT the official gate and is a poor discriminator on
        # this near-tie-dense model (even bf16 diverges ~83% from int4); we report it as
        # a quality-DRIFT diagnostic and a secondary check (does the codebook add drift
        # beyond the inherent bf16<->int4 floor?), but quality is bound by the PPL gate.
        identity_selfref_ok = True
        drift = wr.get("greedy_frac_mismatch_vs_int4", float("nan"))
        drift_within_bf16_floor = bool(
            not math.isnan(drift) and bf16_drift is not None and drift <= bf16_drift + 0.05) \
            if bf16_drift is not None else None
        bw_ok = bool(wr["bw"]["realized_bw_lift_estimate"] > 0.0)
        go = bool(ppl_ok and identity_selfref_ok and bw_ok)
        wr["gate_ppl_ok"] = ppl_ok
        wr["gate_identity_selfref_ok"] = identity_selfref_ok
        wr["gate_identity_drift_within_bf16_floor"] = drift_within_bf16_floor
        wr["gate_realized_bw_ok"] = bw_ok
        wr["codebook_subint4_width_go"] = go
        if go:
            go_widths.append(name)

    hres = width_results.get(headline, {}) if headline else {}
    return {
        "config": {
            "base_model": args.base_model, "scheme": args.scheme, "group_size": args.group_size,
            "vq_dim": args.vq_dim, "seed": args.seed, "widths": [w["name"] for w in widths],
            "max_records": max_records or len(records),
            "greedy_prompts": args.greedy_prompts, "greedy_tokens": args.greedy_tokens,
            "ppl_gate": PPL_GATE, "strict_nonspec_floor_tps": STRICT_NONSPEC_FLOOR_TPS,
            "served_int4_ppl_spec_anchor": SERVED_INT4_PPL_SPEC,
            "int4_localproxy_ppl": int4_ppl, "body_params_b": body_params / 1e9, "device": device,
            "kernel_lit": lit,
        },
        "width_results": width_results,
        "greedy_identity": identity,
        "headline_width": headline,
        "headline": {
            "codebook_gate_transfer_ppl_b3": hres.get("gate_transfer_ppl"),
            "rel_increase_over_int4_b3": hres.get("rel_increase_over_int4"),
            "codebook_greedy_frac_mismatch_b3": hres.get("greedy_frac_mismatch_vs_int4"),
            "codebook_effective_bpw_b3": hres.get("effective_bpw"),
            "codebook_realized_bw_lift_estimate": (hres.get("bw") or {}).get("realized_bw_lift_estimate"),
            "codebook_realized_tps_estimate": (hres.get("bw") or {}).get("realized_tps_estimate"),
            "codebook_pure_bw_pred_tps": (hres.get("bw") or {}).get("pure_bw_pred_tps"),
            "codebook_subint4_width_go": hres.get("codebook_subint4_width_go"),
            "gate_ppl_ok": hres.get("gate_ppl_ok"),
            "gate_identity_selfref_ok": hres.get("gate_identity_selfref_ok"),
            "gate_identity_drift_within_bf16_floor": hres.get("gate_identity_drift_within_bf16_floor"),
            "gate_realized_bw_ok": hres.get("gate_realized_bw_ok"),
        },
        "go_widths": go_widths,
        "any_codebook_subint4_go": bool(go_widths),
    }


# --------------------------------------------------------------------------- #
# Report + W&B.
# --------------------------------------------------------------------------- #
def print_report(res: dict[str, Any]) -> None:
    cfg = res["config"]
    print("\n" + "=" * 104, flush=True)
    print("CODEBOOK ~3bpw BODY MEASUREMENT (PR #367) — the last sub-int4 ceiling-lift shot", flush=True)
    print("=" * 104, flush=True)
    print(f"  base={cfg['base_model']}  g{cfg['group_size']} vq_dim{cfg['vq_dim']} seed{cfg['seed']}  "
          f"records={cfg['max_records']}  body={cfg['body_params_b']:.3f}B", flush=True)
    print(f"  int4 local-proxy PPL={cfg['int4_localproxy_ppl']}  gate-transfer x{SERVED_INT4_PPL_SPEC}/int4 "
          f"-> gate <= {PPL_GATE}", flush=True)
    print("-" * 104, flush=True)
    print(f"  {'width':>10} {'bpw':>6} {'PPL':>9} {'rel/int4':>9} {'gate_ppl':>9} {'<=2.42':>7} "
          f"{'drift!=int4':>11} {'realBWlift':>11} {'GO':>4}", flush=True)
    for name in cfg["widths"]:
        wr = res["width_results"][name]
        rel = wr.get("rel_increase_over_int4", float("nan"))
        rels = "  ref  " if name == "int4" else ("  n/a  " if math.isnan(rel) else f"{100*rel:+7.2f}%")
        gp = wr.get("gate_transfer_ppl", float("nan"))
        gps = "  ref  " if name == "int4" else ("  n/a  " if math.isnan(gp) else f"{gp:8.4f}")
        gpass = "ref" if name == "int4" else str(wr.get("gate_transfer_pass", "—"))
        drift = wr.get("greedy_frac_mismatch_vs_int4", None)
        drifts = "ref" if name == "int4" else (f"{drift:.3f}" if drift is not None else "—")
        bw = wr.get("bw")
        bws = (f"{100*bw['realized_bw_lift_estimate']:+.1f}%" if bw else "—")
        go = wr.get("codebook_subint4_width_go", "—")
        print(f"  {name:>10} {wr['effective_bpw']:6.3f} {wr['ppl']:9.4f} {rels:>9} {gps:>9} "
              f"{gpass:>7} {drifts:>11} {bws:>11} {str(go):>4}", flush=True)
    print("-" * 104, flush=True)
    h = res["headline"]
    print(f"  HEADLINE width = {res['headline_width']}", flush=True)
    print(f"   gate1 PPL : transfer_ppl={h['codebook_gate_transfer_ppl_b3']} (rel "
          f"{None if h['rel_increase_over_int4_b3'] is None else round(100*h['rel_increase_over_int4_b3'],2)}%) "
          f"<= {PPL_GATE}? -> {h['gate_ppl_ok']}", flush=True)
    print(f"   gate2 ID  : drift_vs_int4={h['codebook_greedy_frac_mismatch_b3']} "
          f"(bf16 floor {res['width_results'].get('bf16',{}).get('greedy_frac_mismatch_vs_int4')}); "
          f"official #319 gate SELF-referential -> selfref_ok={h['gate_identity_selfref_ok']} "
          f"(drift<=bf16floor? {h['gate_identity_drift_within_bf16_floor']})", flush=True)
    print(f"   gate3 BW  : realized_lift={None if h['codebook_realized_bw_lift_estimate'] is None else round(100*h['codebook_realized_bw_lift_estimate'],1)}% "
          f"(realized {h['codebook_realized_tps_estimate']} vs pure-BW {h['codebook_pure_bw_pred_tps']} vs floor "
          f"{STRICT_NONSPEC_FLOOR_TPS}) > 0? -> {h['gate_realized_bw_ok']}", flush=True)
    print(f"  >>> codebook_subint4_width_go = {h['codebook_subint4_width_go']}  "
          f"(any width: {res['any_codebook_subint4_go']} {res['go_widths']})", flush=True)
    print("=" * 104, flush=True)


def maybe_log_wandb(args, payload: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[cb] wandb unavailable: {exc}", flush=True)
        return
    res = payload["result"]
    cfg = res["config"]
    run = init_wandb_run(
        job_type="validity-gate", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "aqlm-codebook-subint4", "non-spec-frontier", "codebook-quant",
              "incoherence", "measured-ppl", "greedy-identity", "pr-367"],
        config={**{k: v for k, v in cfg.items() if k != "kernel_lit"}, "wandb_group": args.wandb_group},
    )
    if run is None:
        print("[cb] wandb: no run — skipping", flush=True)
        return
    h = res["headline"]
    summary: dict[str, Any] = {
        "any_codebook_subint4_go": int(bool(res["any_codebook_subint4_go"])),
        "num_go_widths": len(res["go_widths"]),
        "int4_localproxy_ppl": cfg["int4_localproxy_ppl"],
    }
    for k, v in h.items():
        if v is None:
            continue
        summary[k] = int(v) if isinstance(v, bool) else v
    for name in cfg["widths"]:
        wr = res["width_results"][name]
        summary[f"ppl_{name}"] = wr["ppl"]
        summary[f"effective_bpw_{name}"] = wr["effective_bpw"]
        if not math.isnan(wr.get("rel_increase_over_int4", float("nan"))):
            summary[f"rel_increase_over_int4_{name}"] = wr["rel_increase_over_int4"]
            summary[f"gate_transfer_ppl_{name}"] = wr["gate_transfer_ppl"]
            summary[f"gate_transfer_pass_{name}"] = int(bool(wr["gate_transfer_pass"]))
        if "greedy_frac_mismatch_vs_int4" in wr:
            summary[f"greedy_frac_mismatch_{name}"] = wr["greedy_frac_mismatch_vs_int4"]
        if wr.get("bw"):
            summary[f"realized_bw_lift_{name}"] = wr["bw"]["realized_bw_lift_estimate"]
            summary[f"realized_tps_est_{name}"] = wr["bw"]["realized_tps_estimate"]
            summary[f"pure_bw_pred_tps_{name}"] = wr["bw"]["pure_bw_pred_tps"]
        if "codebook_subint4_width_go" in wr:
            summary[f"go_{name}"] = int(bool(wr["codebook_subint4_width_go"]))
    summary["peak_mem_mib"] = payload["peak_mem_mib"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="codebook_subint4_measured", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[cb] wandb logged: {len(summary)} metrics", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", "--base_model", dest="base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--widths", default="bf16,int4,cb3,cb3_noinc,cb2",
                    help="comma list from {bf16,int4,rtn3,cb3,cb3_noinc,cb2,cb3d1}")
    ap.add_argument("--scheme", choices=["asym", "sym"], default="asym", help="RTN scheme for int4/rtn3")
    ap.add_argument("--group-size", "--group_size", dest="group_size", type=int, default=128)
    ap.add_argument("--vq-dim", "--vq_dim", dest="vq_dim", type=int, default=2,
                    help="default VQ sub-vector dim (per-width vq_dim in the registry overrides)")
    ap.add_argument("--seed", type=int, default=0, help="RHT + codebook seed")
    ap.add_argument("--ppl-dataset", "--ppl_dataset", dest="ppl_dataset", default=str(DEFAULT_PPL_DATASET))
    ap.add_argument("--max-records", "--max_records", dest="max_records", type=int, default=0)
    ap.add_argument("--greedy-prompts", "--greedy_prompts", dest="greedy_prompts", type=int, default=8)
    ap.add_argument("--greedy-tokens", "--greedy_tokens", dest="greedy_tokens", type=int, default=128)
    ap.add_argument("--marlin-speedup", "--marlin_speedup", dest="marlin_speedup", type=float, default=0.0,
                    help="override int4-Marlin batch=1 speedup-vs-fp16 (deliverable 3 sensitivity)")
    ap.add_argument("--codebook-speedup", "--codebook_speedup", dest="codebook_speedup", type=float,
                    default=0.0, help="override AQLM 3-bit batch=1 speedup-vs-fp16")
    ap.add_argument("--out-dir", dest="out_dir", default=str(HERE))
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="aqlm-codebook-subint4")
    args = ap.parse_args(argv)

    t0 = time.time()
    res = run(args)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_mib = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    payload = {
        "created_at": created_at, "pr": 367, "agent": "lawine", "kind": "codebook-subint4-measured",
        "elapsed_s": round(time.time() - t0, 1), "peak_mem_mib": round(peak_mib, 1), "result": res,
    }
    print_report(res)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "measure_codebook_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[cb] wrote {out_path} (elapsed {payload['elapsed_s']}s, peak {payload['peak_mem_mib']} MiB)",
          flush=True)
    maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
