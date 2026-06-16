#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tighten the cb3 supply x demand rung: a REAL cb3 fake-quant (RHT + dim-2 VQ),
replacing #410's int8-anchored stand-in with a direct cb3-rung perturbation. (PR #422, ubel).

WHY (the #410 follow-up #2)
---------------------------
My #410 (7rzf74q5) proved the cb3-supply x MTP-demand cross-term is approximately_additive
and anchored the faithful input-leg tax on the gate-safe int8 RTN rung (KV L2-rel 0.0071,
1.6% body-argmax flip -> <= 15.74 TPS = 14.9% of the +105.59 demand lift). That int8 anchor was
a principled STAND-IN (int3-RTN is the gate-DEAD #355 regime; crude-int4-RTN-g128 over-states at
20.5% flip), but it was NOT a direct cb3 measurement. #410's own follow-up #2 named the fix: wire a
REAL cb3 fake-quant and replace the int8-anchored estimate with the actual cb3-rung perturbation.

WHAT THIS CARD DOES
-------------------
cb3 = Random-Hadamard-Transform (RHT) incoherence + L1-resident K=64 dim-2 Gaussian vector
quantization (the QTIP/QuIP#-class recipe), 3.2369 effective bpw (byte-ratio 0.785 vs int4,
#289). #410 could only BRACKET it with scalar RTN because there is no shipping cb3 kernel
(#372/cb3_kernel_realized_bw). This card IMPLEMENTS the recipe as a FAKE-QUANT -- a numpy/torch
read-only simulation of the body-weight transform What = invRHT(VQ(RHT(W))) -- applied to the bf16
body on the local A10G, then measures the perturbation at the EXACT drafter-read tensors:
  * shared_kv_states     = pre-RoPE k_proj/v_proj of the 24 KV-owning layers (L2-rel is RoPE-invariant).
  * inputs_embeds_hidden = post-final-norm hidden feeding the drafter's inputs_embeds.
and places the cb3-real rung on #410's monotone curve {fp16, int8, cb3-real, crude-int4-RTN, int3}.

THE RECIPE (PR instruction 2; QuIP#/QTIP, confirmed against Tseng'24 / Chee'23 / QuaRot)
----------------------------------------------------------------------------------------
(a) RHT incoherence (two-sided): Wtilde = U W V^T, U = diag(s_L) H_out / sqrt(out),
    V = diag(s_R) H_in / sqrt(in). H are +-1 Hadamard matrices (Sylvester for 2^a; Paley-I
    Kronecker for the odd factor: in {2560=128x20, 10240=512x20}, out {...,10752=128x84}).
    The random sign diagonals s_L,s_R make it a RANDOMIZED Hadamard (kills H's column structure).
    U,V orthogonal => quant-free round-trip is exact (validated: e_rht_roundtrip_exact).
(b) K=64 dim-2 Gaussian VQ: after RHT the weights are ~iid Gaussian; per group-of-64 RMS scale,
    then quantize adjacent input-dim pairs to the nearest of 64 Lloyd-optimal N(0,I_2) codewords.
    Code rate = log2(64)/2 = 3.0 bpw + fp16 scale/64 = 0.25 bpw => 3.25 bpw (brackets cb3's 3.2369
    within +0.4%; the perturbation is measured directly, so exact bpw is provenance, not load-bearing).
(c) inverse-RHT: What = U^T VQ(Wtilde) V.

THE LDLQ CAVEAT (honest; load-bearing for the verdict)
------------------------------------------------------
QuIP#/QTIP reach NEAR-LOSSLESS PPL at 3-4 bpw largely via LDLQ (Hessian/calibration error-feedback),
NOT incoherence+VQ alone. This fake-quant is DATA-FREE (no LDLQ): it omits the calibration step, so it
is a CONSERVATIVE proxy -- it perturbs >= real (LDLQ-calibrated) cb3. The incoherence benefit that
matters for the PRIMARY metric (random/incoherent weight error AVERAGES OUT in the GEMM -> small
activation perturbation; QuIP Thm, QuaRot) survives data-free; the residual tf-PPL gap over fp16 is the
known data-free penalty. So: the measured cb3-real rung is an UPPER bound on the real cb3 rung; the
PPL cross-check is read as "careful-VQ-class, NOT crude-RTN-class", with real LDLQ-cb3 holding the gate
(2.3812) at-or-below this. The bracket cb3-real <= crude-int4-RTN is the load-bearing faithful signal.

NOT a launch, NOT a submission, NO served-file change, 0 official TPS. GPU used ONLY for the local
perturbation forward. PPL UNCHANGED (deployed greedy target token untouched; cb3 stays 2.3812 < 2.42).
analysis_only = no_hf_job = no_served_file_change = True; official_tps = 0.

REPRODUCE (needs torch + a visible GPU; the repo .venv has no torch -- use /usr/bin/python3):
    cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m \
      research.validity.cb3_real_rung_perturbation.cb3_real_rung_perturbation --self-test
    cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m \
      research.validity.cb3_real_rung_perturbation.cb3_real_rung_perturbation \
        --quant-sweep fp16,int8,cb3-real,int4,int3 --measure shared_kv_states,inputs_embeds_hidden \
        --wandb_group cb3-real-rung-perturbation --wandb_name ubel/cb3-real-rung-perturbation
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # target/

# Reuse the #410 measurement spine EXACTLY (load + capture + perturbation accumulators + RTN proxy).
from research.validity.cb3_acceptance_crossterm.cb3_acceptance_crossterm import (  # noqa: E402
    MODEL_ID, PROMPTS_PATH, FINAL_LOGIT_SOFTCAP,
    INT4_BPW, CB3_BPW_EFF, CB3_MIXED_FRAC, CB3_BYTE_RATIO,
    LADDER_289, DRAFTER_TOP1_289, E_ACCEPTED_289, E_T_289,
    GROSS_TPS_PER_UNIT_COV, CORRECTED_STRICT_BASE, GAP_TO_500, CB3_SUPPLY_LIFT_388,
    DEMAND_COVERAGE_BAND, DEMAND_LIFT_TPS_BAND, NEGLIGIBLE_FRAC,
    BASELINE_TPS, BASELINE_PPL, CB3_PPL, PPL_GATE, GATE_HEADROOM_REL, CB3_REL_PPL_OVER_INT4,
    INT3_RTN_RELPPL_355, INT3_RTN_GATE_PPL_355,
    load_body, quantizable_linears, cache_original_weights, register_kv_hooks,
    build_prompt_ids, forward_capture, fake_quant_grouped, TensorPerturb, expected_accepted,
)

# ===========================================================================
# Section 0 -- banked anchors imported from my #410 results (7rzf74q5) for the tightening compare
# ===========================================================================
INT8_KV_L2REL_410 = 0.0071407895516850195   # #410 gate-safe int8 rung (the stand-in we replace)
INT8_HID_L2REL_410 = 0.06498690782584027
INT8_FLIP_410 = 0.016358306804929557
INT8_DTPS_410 = 15.741107889179563          # #410 int8-anchored faithful upper bound (the band hi we tighten)
INT4_KV_L2REL_410 = 0.07342104081134417     # #410 crude-int4-RTN-g128 (over-states)
INT4_HID_L2REL_410 = 0.6547745743543942
INT4_FLIP_410 = 0.2047467456740316
INT3_KV_L2REL_410 = 0.16359887454873268     # #410 int3 gate-DEAD ceiling
INT3_FLIP_410 = 0.5411163992813692
FP16_TF_PPL_410 = 25.733985341277865        # #410 bf16-head teacher-forced PPL anchor
INT4_TF_PPL_410 = 30.348819770011534        # #410 crude-int4-RTN tf-ppl (+18% over fp16; the crude regime)

# the 5%/15% negligibility bands on the +105.59 demand lift
NEGLIGIBLE_TPS = NEGLIGIBLE_FRAC * DEMAND_LIFT_TPS_BAND          # 5%  = 5.28 TPS
ADDITIVE_NOT_NEGLIGIBLE_HI_TPS = 0.15 * DEMAND_LIFT_TPS_BAND     # 15% = 15.84 TPS

# cb3-real fake-quant numerics
CB3_VQ_DIM = 2
CB3_VQ_K = 64                      # 64 dim-2 codewords -> log2(64)/2 = 3.0 bpw code rate
CB3_VQ_CODE_BPW = math.log2(CB3_VQ_K) / CB3_VQ_DIM
CB3_VQ_SCALE_GROUP = 64            # fp16 scale per 64 weights -> 16/64 = 0.25 bpw
CB3_VQ_SCALE_BPW = 16.0 / CB3_VQ_SCALE_GROUP
CB3_VQ_EFF_BPW = CB3_VQ_CODE_BPW + CB3_VQ_SCALE_BPW             # 3.25 bpw realized
CB3_VQ_SEED = 1337                 # deterministic RHT signs + codebook

# cb3-LDLQ: the calibrated rung. The PR's LITERAL recipe (RHT+VQ, data-free) is `cb3-real`; it OMITS
# the QuIP#/QTIP error-feedback (LDLQ) step that supplies the bulk of the near-lossless quality and that
# the served cb3 (#388, gate-hold 2.3812) actually uses. `cb3-ldlq` adds it: block-pair GPTQ/LDLQ error
# feedback in the RHT domain, calibrated on the SAME 128 deployed prompts. tr(E H E^T) == E||E x||^2, so
# GPTQ directly minimizes the measured drafter-input activation perturbation -> this is the FAITHFUL rung.
CB3_LDLQ_BLOCKSIZE = 128           # GPTQ lazy-batch column block (cols, even; pairs handled inside)
CB3_LDLQ_DAMP_FRAC = 0.01          # Hessian damping: H += damp * mean(diag(H)) * I (GPTQ standard)

# self-test tolerances
ZERO_TOL = 1e-4
MONO_EPS = 1e-6
RHT_ROUNDTRIP_TOL = 5e-3           # quant-free invRHT(RHT(W)) == W rel-Frobenius (bf16 round-trip noise)
# "careful-int4 PPL parity": data-free cb3-real cannot reach strict fp16-parity (no LDLQ); the load-bearing,
# faithful claim is it lands FAR closer to fp16 than crude-int4-RTN (careful-VQ-class, not crude-RTN-class).
CAREFUL_PPL_FRACTION_OF_CRUDE = 0.5   # cb3-real tf-ppl gap over fp16 must be < half crude-int4-RTN's gap


# ===========================================================================
# Section 1 -- scheme parsing (fp16 | int8/int4/int3 crude-RTN | cb3-real RHT+VQ)
# ===========================================================================

def scheme_spec(name: str):
    """(kind, bits): ('ref',None) | ('rtn',b) | ('cb3real',None) | ('cb3ldlq',None)."""
    n = name.strip().lower()
    if n in ("fp16", "bf16", "fp32", "none", "ref"):
        return ("ref", None)
    if n in ("int8", "w8", "8bit"):
        return ("rtn", 8)
    if n in ("int4", "w4", "4bit", "crude-int4"):
        return ("rtn", 4)
    if n in ("int3", "w3", "3bit"):
        return ("rtn", 3)
    if n in ("cb3-real", "cb3real", "cb3", "cb3-rht-vq", "cb3-nocal", "cb3-datafree"):
        return ("cb3real", None)
    if n in ("cb3-ldlq", "cb3ldlq", "cb3-cal", "cb3-real-ldlq"):
        return ("cb3ldlq", None)
    raise ValueError(f"unknown quant scheme: {name!r}")


def scheme_aggr(name: str) -> float:
    """Aggressiveness rank for the monotone-ordering self-test (the HYPOTHESIS ordering).
    The FAITHFUL (calibrated) rung is cb3-ldlq, expected in the bracket:
        fp16(0) < int8(1) < cb3-ldlq(2) < crude-int4-RTN(3) < int3(4),
    with the data-free literal recipe cb3-real(3.5) as the loose UPPER bound above int4."""
    kind, bits = scheme_spec(name)
    if kind == "ref":
        return 0.0
    if kind == "cb3ldlq":
        return 2.0
    if kind == "cb3real":
        return 3.5
    return {8: 1.0, 4: 3.0, 3: 4.0, 2: 5.0}.get(bits, 9.0)


def scheme_role(name: str) -> str:
    kind, bits = scheme_spec(name)
    if kind == "ref":
        return "reference (bf16; Delta=0 sanity)"
    if kind == "cb3ldlq":
        return (f"cb3-LDLQ (FAITHFUL rung): two-sided RHT incoherence + K={CB3_VQ_K} dim-{CB3_VQ_DIM} "
                f"Gaussian VQ + block-pair GPTQ/LDLQ error-feedback (calibrated on the 128 prompts) "
                f"@ {CB3_VQ_EFF_BPW:.3f}bpw -- the supply-rung the band tightens to")
    if kind == "cb3real":
        return (f"cb3-real (LITERAL recipe, data-free): two-sided RHT + K={CB3_VQ_K} dim-{CB3_VQ_DIM} "
                f"Gaussian VQ @ {CB3_VQ_EFF_BPW:.3f}bpw, NO LDLQ -> strict UPPER bound (over-perturbs)")
    if bits == 8:
        return "int8 RTN g128 (gate-safe rung; #410 faithful anchor we replace)"
    if bits == 4:
        return "crude int4 RTN g128 (scalar; OVER-states cb3 -- the upper bracket edge)"
    if bits == 3:
        return "int3 RTN g128 (gate-DEAD scalar regime, #355; strict ceiling)"
    return f"int{bits} RTN g128"


# ===========================================================================
# Section 2 -- Hadamard construction (Sylvester 2^a (x) Paley-I odd factor)
# ===========================================================================

def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _sylvester(a: int):
    """2^a x 2^a +-1 Hadamard via Sylvester recursion (symmetric)."""
    import torch  # noqa: PLC0415
    H = torch.ones((1, 1), dtype=torch.float32)
    for _ in range(a):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def _paley1(order: int):
    """order x order +-1 Hadamard via Paley type-I (order = q+1, q prime, q % 4 == 3).
    H = I + C with C the skew conference matrix from the Jacobsthal (Legendre) core."""
    import torch  # noqa: PLC0415
    q = order - 1
    if not (_is_prime(q) and q % 4 == 3):
        raise ValueError(f"Paley-I needs order-1 prime == 3 mod 4; got order {order}")

    def leg(a):
        a %= q
        if a == 0:
            return 0
        return 1 if pow(a, (q - 1) // 2, q) == 1 else -1

    C = torch.zeros((order, order), dtype=torch.float32)
    for j in range(1, order):
        C[0, j] = 1.0
        C[j, 0] = -1.0
    for i in range(q):
        for j in range(q):
            C[i + 1, j + 1] = float(leg(j - i))   # skew core (leg is odd since q==3 mod 4)
    H = torch.eye(order, dtype=torch.float32) + C
    return H


def _small_hadamard(k: int):
    """k x k +-1 Hadamard for a 'known small order' k (pow2 -> Sylvester; else Paley-I)."""
    if _is_pow2(k):
        a = int(round(math.log2(k)))
        return _sylvester(a)
    return _paley1(k)


def _is_known_order(k: int) -> bool:
    if _is_pow2(k):
        return True
    return k % 4 == 0 and _is_prime(k - 1) and (k - 1) % 4 == 3


_HAD_CACHE: dict = {}


def build_hadamard(n: int, device="cuda"):
    """n x n +-1 Hadamard (unnormalized). Factor n = 2^a * K with K a known small order;
    H_n = H_{2^a} (x) H_K (Kronecker). Cached on device. Returns None if n is not constructible."""
    import torch  # noqa: PLC0415
    key = (n, str(device))
    if key in _HAD_CACHE:
        return _HAD_CACHE[key]
    # find odd-resident known small order K = odd * 2^k that is a known Hadamard order
    odd = n
    a = 0
    while odd % 2 == 0:
        odd //= 2
        a += 1
    H = None
    for k in range(0, a + 1):
        K = odd * (2 ** k)
        if _is_known_order(K) and _is_pow2(n // K):
            two = n // K
            HK = _small_hadamard(K)
            H2 = _sylvester(int(round(math.log2(two)))) if two > 1 else torch.ones((1, 1))
            H = torch.kron(H2, HK).to(device=device, dtype=torch.float32)
            break
    if H is None:
        _HAD_CACHE[key] = None
        return None
    # numerical validation: H H^T == n I (orthogonality up to scale)
    chk = (H @ H.t() - n * torch.eye(n, device=H.device)).abs().max().item()
    if chk > 1e-3:
        _HAD_CACHE[key] = None
        return None
    _HAD_CACHE[key] = H
    return H


# ===========================================================================
# Section 3 -- dim-2 Gaussian VQ codebook (Lloyd / k-means on N(0, I_2))
# ===========================================================================

_CODEBOOK_CACHE: dict = {}


def gaussian_codebook_2d(k=CB3_VQ_K, n_samples=200_000, iters=60, seed=CB3_VQ_SEED, device="cuda"):
    """Lloyd-optimal k-codeword dim-2 codebook for a standard 2D Gaussian source (data-free, fixed).
    Cached. Returns (codebook [k,2] float32, normalized_distortion float)."""
    import torch  # noqa: PLC0415
    key = (k, n_samples, iters, seed, str(device))
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(n_samples, 2, generator=g).to(device)
    # k-means++ style seeding: first center random, then farthest-weighted picks
    idx0 = torch.randint(0, n_samples, (1,), generator=g).item()
    centers = X[idx0:idx0 + 1].clone()
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(k - 1):
        probs = d2 / d2.sum().clamp_min(1e-12)
        nxt = torch.multinomial(probs, 1).item()
        centers = torch.cat([centers, X[nxt:nxt + 1]], 0)
        d2 = torch.minimum(d2, ((X - X[nxt]) ** 2).sum(1))
    # Lloyd iterations
    for _ in range(iters):
        dist = torch.cdist(X, centers)            # [N,k]
        assign = dist.argmin(1)                    # [N]
        for c in range(k):
            m = assign == c
            if m.any():
                centers[c] = X[m].mean(0)
    dist = torch.cdist(X, centers)
    mind2 = (dist.min(1).values ** 2)
    norm_distortion = float(mind2.mean() / 2.0)    # per-dim normalized MSE (source var per dim = 1)
    cb = centers.to(torch.float32)
    _CODEBOOK_CACHE[key] = (cb, norm_distortion)
    return cb, norm_distortion


def _nearest_codeword(P, cb, chunk=2_000_000):
    """P [N,2] -> nearest codebook index [N] (chunked argmin; memory-safe)."""
    import torch  # noqa: PLC0415
    N = P.shape[0]
    out = torch.empty(N, dtype=torch.long, device=P.device)
    cb2 = (cb * cb).sum(1)                          # [k]
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        blk = P[s:e]
        d = (blk * blk).sum(1, keepdim=True) - 2.0 * (blk @ cb.t()) + cb2[None, :]
        out[s:e] = d.argmin(1)
    return out


# ===========================================================================
# Section 4 -- the cb3-real fake-quant: invRHT(VQ(RHT(W)))
# ===========================================================================

def _signs(n, seed, device):
    import torch  # noqa: PLC0415
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g).to(device=device, dtype=torch.float32) * 2.0 - 1.0)
    return s


def cb3_real_fake_quant(W_cpu, codebook, name="", device="cuda"):
    """Apply the cb3 recipe to one [out,in] bf16 weight: two-sided RHT incoherence -> dim-2 K=64
    Gaussian VQ (group-64 RMS scale) -> inverse RHT. Returns (What bf16 on device, info dict).
    Falls back to one-sided (input-only) RHT if an output Hadamard is unconstructible (logged)."""
    import torch  # noqa: PLC0415
    out, inn = W_cpu.shape
    W = W_cpu.to(device=device, dtype=torch.float32)
    # deterministic per-tensor signs (seed mixes the shape so each matrix gets distinct randomization)
    seed_mix = CB3_VQ_SEED ^ (out * 1000003 + inn)
    sL = _signs(out, seed_mix ^ 0x5DEECE66, device)
    sR = _signs(inn, seed_mix ^ 0x1234567, device)
    Hin = build_hadamard(inn, device)
    Hout = build_hadamard(out, device)
    two_sided = Hout is not None and Hin is not None
    info = {"two_sided": bool(two_sided), "hin": Hin is not None, "hout": Hout is not None}
    if Hin is None:
        # input dim must always be constructible for these shapes; guard with a clear error
        raise RuntimeError(f"no input Hadamard for in={inn} ({name})")
    rin = math.sqrt(inn)
    rout = math.sqrt(out)

    # forward RHT: Wtilde = U W V^T, U=diag(sL)Hout/rout, V=diag(sR)Hin/rin
    Wt = W
    if two_sided:
        Wt = (Hout @ (sL[:, None] * Wt)) / rout            # U W
    Wt = ((Wt * sR[None, :]) @ Hin.t()) / rin              # @ V^T

    # dim-2 K=64 VQ with group-64 RMS scaling
    P = Wt.reshape(out, inn // 2, 2)
    ppg = CB3_VQ_SCALE_GROUP // 2                           # pairs per scale group (32)
    ng = (inn // 2) // ppg if (inn // 2) % ppg == 0 else 1
    if (inn // 2) % ppg == 0:
        Pg = P.reshape(out, ng, ppg, 2)
        scale = Pg.pow(2).mean(dim=(2, 3), keepdim=True).clamp_min(1e-12).sqrt()   # [out,ng,1,1]
        Pn = (Pg / scale).reshape(-1, 2)
        idx = _nearest_codeword(Pn, codebook)
        Q = (codebook[idx].reshape(out, ng, ppg, 2) * scale).reshape(out, inn // 2, 2)
    else:                                                   # fallback: single per-row scale
        scale = P.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        Pn = (P / scale).reshape(-1, 2)
        idx = _nearest_codeword(Pn, codebook)
        Q = (codebook[idx].reshape(out, inn // 2, 2) * scale)
    Wq = Q.reshape(out, inn)

    # inverse RHT: What = U^T Wq V = diag(sL)Hout^T (.) /rout @ ... @ Hin diag(sR)/rin
    Wh = ((Wq @ Hin) / rin) * sR[None, :]                  # @ V  (right)
    if two_sided:
        Wh = sL[:, None] * (Hout.t() @ Wh) / rout          # U^T (left)

    del Wt, P, Pn, Q, Wq
    return Wh.to(torch.bfloat16), info


# ===========================================================================
# Section 4b -- the cb3-LDLQ fake-quant: RHT + dim-2 VQ + block-pair GPTQ error feedback
# ===========================================================================

def collect_calib_hessians(text, linears, ids_list, device="cuda"):
    """One forward over the calibration prompts on the PRISTINE bf16 body; accumulate the per-linear
    input second moment H_name = (1/T) sum_t x_t x_t^T (the GPTQ/LDLQ proxy Hessian). Stored on CPU
    (fp32, ~23 GB total here -- trivial in 700 GB RAM). The proxy loss tr(E H E^T) == E||E x||^2 is
    EXACTLY the activation perturbation this card measures, so calibrating against it is what makes the
    cb3-ldlq rung faithful rather than data-free."""
    import torch  # noqa: PLC0415
    H = {name: torch.zeros(m.weight.shape[1], m.weight.shape[1], dtype=torch.float32)
         for name, m in linears}
    counts = {name: 0 for name, _ in linears}

    def mk(name):
        def pre_hook(_m, args):
            x = args[0]
            if x is None:
                return
            xf = x.reshape(-1, x.shape[-1]).to(torch.float32)   # [T, in] on GPU
            H[name].add_((xf.t() @ xf).to("cpu"))
            counts[name] += int(xf.shape[0])
        return pre_hook

    handles = [m.register_forward_pre_hook(mk(name)) for name, m in linears]
    for ids in ids_list:
        text(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    for name, _ in linears:
        if counts[name] > 0:
            H[name] /= counts[name]
    torch.cuda.synchronize()
    return H, counts


def _block_pair_gptq(Wt, H_rot, codebook, device="cuda",
                     blocksize=CB3_LDLQ_BLOCKSIZE, damp_frac=CB3_LDLQ_DAMP_FRAC,
                     scale_group=CB3_VQ_SCALE_GROUP):
    """GPTQ/LDLQ in the RHT domain with a dim-2 VQ quantizer. Quantizes adjacent input-dim PAIRS to the
    nearest Gaussian codeword (group-RMS scale) in column order, propagating each column's residual to the
    not-yet-quantized columns through the inverse-Hessian Cholesky (Frantar'22 lazy-batch). Minimizes
    tr((Wt-Q) H_rot (Wt-Q)^T) = the rotated-domain activation error. Returns Q [out,in] in the RHT domain."""
    import torch  # noqa: PLC0415
    out, inn = Wt.shape
    W = Wt.clone()
    Q = torch.zeros_like(W)
    H = H_rot
    didx = torch.arange(inn, device=device)
    dead = H[didx, didx] == 0
    if bool(dead.any()):
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    damp = damp_frac * H[didx, didx].mean()
    H[didx, didx] += damp
    # upper Cholesky of H^{-1}: U with U^T U = H^{-1}; U[i,i] is the GPTQ per-column denominator
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:  # noqa: BLE001  (extra damping if not PD -- rank-deficient calib)
        H[didx, didx] += 10.0 * damp
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    # fixed per-(row,group) RMS scale from the rotated weights (group is 64-wide; pairs never straddle it)
    ng = inn // scale_group
    sc_g = W.reshape(out, ng, scale_group).pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()  # [out,ng,1]
    scale = sc_g.expand(out, ng, scale_group).reshape(out, inn)                                   # [out,in]

    for i1 in range(0, inn, blocksize):
        i2 = min(i1 + blocksize, inn)
        cols = i2 - i1
        Wb = W[:, i1:i2].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Ub = U[i1:i2, i1:i2]
        for j in range(0, cols - 1, 2):
            sc = scale[:, i1 + j].unsqueeze(1)              # [out,1] (group-constant across the pair)
            pair = Wb[:, j:j + 2]                           # [out,2]
            qn = _nearest_codeword(pair / sc, codebook)
            Qb[:, j:j + 2] = codebook[qn] * sc
            for t in range(2):
                jj = j + t
                err = (Wb[:, jj] - Qb[:, jj]) / Ub[jj, jj]
                if jj + 1 < cols:
                    Wb[:, jj + 1:] -= err.unsqueeze(1) * Ub[jj, jj + 1:].unsqueeze(0)
                Eb[:, jj] = err
        Q[:, i1:i2] = Qb
        if i2 < inn:
            W[:, i2:] -= Eb @ U[i1:i2, i2:]
    return Q


def cb3_ldlq_fake_quant(W_cpu, H_cpu, codebook, name="", device="cuda"):
    """The FAITHFUL cb3 rung: two-sided RHT incoherence -> dim-2 K=64 VQ WITH block-pair GPTQ/LDLQ
    error feedback (calibrated on the body's own activations, H_cpu) -> inverse RHT. This is the QuIP#/QTIP
    recipe with the error-feedback step the literal `cb3-real` omits. Returns (What bf16 on device, info)."""
    import torch  # noqa: PLC0415
    out, inn = W_cpu.shape
    W = W_cpu.to(device=device, dtype=torch.float32)
    seed_mix = CB3_VQ_SEED ^ (out * 1000003 + inn)
    sL = _signs(out, seed_mix ^ 0x5DEECE66, device)
    sR = _signs(inn, seed_mix ^ 0x1234567, device)
    Hin = build_hadamard(inn, device)
    Hout = build_hadamard(out, device)
    two_sided = Hout is not None and Hin is not None
    if Hin is None:
        raise RuntimeError(f"no input Hadamard for in={inn} ({name})")
    rin, rout = math.sqrt(inn), math.sqrt(out)

    # forward RHT on W (same convention as cb3_real_fake_quant)
    Wt = W
    if two_sided:
        Wt = (Hout @ (sL[:, None] * Wt)) / rout
    Wt = ((Wt * sR[None, :]) @ Hin.t()) / rin

    # rotate the input Hessian into the SAME RHT input basis. The rotated input is xt = Hin (sR (.) x)/rin
    # (sign-flip FIRST, then Hadamard -- matching the forward weight transform above), so the rotated-domain
    # proxy Hessian is H_rot = E[xt xt^T] = Hin diag(sR) H diag(sR) Hin^T / inn  (= V H V^T, V=Hin diag(sR)/rin).
    Hh = H_cpu.to(device=device, dtype=torch.float32)
    Hs = sR[:, None] * Hh * sR[None, :]                     # diag(sR) H diag(sR): sign-scale first
    H_rot = (Hin @ Hs @ Hin.t()) / inn                      # then Hadamard-conjugate
    del Hh, Hs

    Wq = _block_pair_gptq(Wt, H_rot, codebook, device=device)
    del H_rot

    # inverse RHT
    Wh = ((Wq @ Hin) / rin) * sR[None, :]
    if two_sided:
        Wh = sL[:, None] * (Hout.t() @ Wh) / rout
    del Wt, Wq
    return Wh.to(torch.bfloat16), {"two_sided": bool(two_sided)}


def apply_scheme(linears, originals, spec, codebook=None, hessians=None, verbose=False):
    """Set each quantizable Linear's weight per the scheme spec, from its cached bf16 original.
    Returns aggregate cb3-real info (two-sided coverage) for the report."""
    import torch  # noqa: PLC0415
    kind, bits = spec
    agg = {"n": 0, "two_sided": 0, "one_sided": 0}
    for name, m in linears:
        src = originals[name]
        if kind == "ref":
            w = src.to(m.weight.device, dtype=m.weight.dtype)
        elif kind == "rtn":
            w = fake_quant_grouped(src, bits).to(m.weight.device, dtype=m.weight.dtype)
        elif kind == "cb3real":
            wq, info = cb3_real_fake_quant(src, codebook, name=name, device=m.weight.device)
            w = wq.to(m.weight.device, dtype=m.weight.dtype)
            agg["n"] += 1
            agg["two_sided"] += int(info["two_sided"])
            agg["one_sided"] += int(not info["two_sided"])
        elif kind == "cb3ldlq":
            if hessians is None or name not in hessians:
                raise RuntimeError(f"cb3-ldlq needs a calibration Hessian for {name}")
            wq, info = cb3_ldlq_fake_quant(src, hessians[name], codebook, name=name, device=m.weight.device)
            w = wq.to(m.weight.device, dtype=m.weight.dtype)
            agg["n"] += 1
            agg["two_sided"] += int(info["two_sided"])
            agg["one_sided"] += int(not info["two_sided"])
        else:
            raise ValueError(kind)
        m.weight.data.copy_(w)
    torch.cuda.synchronize()
    return agg


# ===========================================================================
# Section 5 -- body greedy-argmax + teacher-forced PPL probe (the secondary acceptance proxy)
# ===========================================================================

def body_logits_stats(hidden_cpu, ids, lm_head):
    """Body greedy argmax per position + teacher-forced NLL through Gemma's real head
    (post-final-norm hidden -> lm_head -> final-logit softcap=30). Returns (argmax[T], nll, ntok)."""
    import torch  # noqa: PLC0415
    if lm_head is None:
        return None
    h = hidden_cpu.to("cuda")
    z = lm_head(h)
    z = FINAL_LOGIT_SOFTCAP * torch.tanh(z / FINAL_LOGIT_SOFTCAP)
    am = z.argmax(dim=-1).to("cpu")
    tgt = ids[0, 1:]
    lp = torch.log_softmax(z[:-1].float(), dim=-1)
    nll = float(-lp[torch.arange(tgt.numel(), device=lp.device), tgt].sum())
    del z, lp
    return am, nll, int(tgt.numel())


# ===========================================================================
# Section 6 -- measurement driver
# ===========================================================================

def run_measurement(schemes, measure, max_prompts, max_seq_len, calib_max_prompts=None, verbose=True):
    import torch  # noqa: PLC0415

    prompts_all = json.loads(PROMPTS_PATH.read_text())
    prompt_texts = [p["conversations"][0]["value"] for p in prompts_all][:max_prompts]
    n_prompts = len(prompt_texts)

    t_load = time.time()
    model, tok, text, lm_head, norm = load_body()
    load_s = time.time() - t_load
    linears = quantizable_linears(text)
    originals = cache_original_weights(linears)
    handles, captured, kv_layers = register_kv_hooks(text)
    n_layers = len(text.layers)
    n_kv = len(kv_layers)

    # build the cb3 Gaussian codebook once (deterministic) if any cb3 rung is requested
    codebook = cb_distortion = None
    needs_cb3 = any(scheme_spec(s)[0] in ("cb3real", "cb3ldlq") for s in schemes)
    if needs_cb3:
        codebook, cb_distortion = gaussian_codebook_2d(device="cuda")
    needs_ldlq = any(scheme_spec(s)[0] == "cb3ldlq" for s in schemes)

    ids_list = [build_prompt_ids(tok, t, max_seq_len) for t in prompt_texts]
    seqlens = [int(x.shape[1]) for x in ids_list]

    # reference (bf16) pass -- store drafter-input tensors + body argmax/NLL on CPU
    apply_scheme(linears, originals, ("ref", None))
    ref_hidden, ref_K, ref_V, ref_stats = [], [], [], []
    for ids in ids_list:
        h, K, V = forward_capture(text, captured, n_kv, ids)
        ref_hidden.append(h)
        ref_K.append(K)
        ref_V.append(V)
        ref_stats.append(body_logits_stats(h, ids, lm_head))
    torch.cuda.synchronize()
    ref_peak_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # cb3-LDLQ calibration: collect per-linear input Hessians on the PRISTINE bf16 body (weights are
    # still ref here). One extra forward over the SAME prompts -> faithful GPTQ/LDLQ feedback.
    hessians = calib_counts = None
    calib_s = 0.0
    n_calib = 0
    if needs_ldlq:
        # GPTQ/LDLQ calibration only needs a FULL-RANK activation set (tok >= max_in = 10240), not the whole
        # measurement set. Decoupling lets the perturbation/flip stay measured on ALL n_prompts deployed prompts
        # while the Hessian forward (the expensive per-prompt in^2 CPU offload) runs on a full-rank subset.
        calib_ids = ids_list if not calib_max_prompts else ids_list[:calib_max_prompts]
        n_calib = len(calib_ids)
        t_cal = time.time()
        hessians, calib_counts = collect_calib_hessians(text, linears, calib_ids)
        calib_s = time.time() - t_cal
        if verbose:
            mintok = min(calib_counts.values())
            print(f"  cb3-ldlq calibration: {len(hessians)} Hessians over {n_calib} prompts "
                  f"(min {mintok} tok/linear; full-rank requires tok>=max_in=10240) in {calib_s:.1f}s")

    per_scheme = {}
    cb3_info = cb3ldlq_info = None
    for sname in schemes:
        spec = scheme_spec(sname)
        t0 = time.time()
        info = apply_scheme(linears, originals, spec, codebook=codebook, hessians=hessians)
        apply_s = time.time() - t0
        if spec[0] == "cb3real":
            cb3_info = info
        if spec[0] == "cb3ldlq":
            cb3ldlq_info = info
        acc_hidden = TensorPerturb()
        acc_kv = TensorPerturb()
        argmax_flips = argmax_tokens = 0
        nll_sum = 0.0
        nll_tok = 0
        for pi, ids in enumerate(ids_list):
            h, K, V = forward_capture(text, captured, n_kv, ids)
            if "inputs_embeds_hidden" in measure:
                acc_hidden.update(ref_hidden[pi], h)
            if "shared_kv_states" in measure:
                for li in range(n_kv):
                    acc_kv.update(ref_K[pi][li], K[li])
                    acc_kv.update(ref_V[pi][li], V[li])
            cur = body_logits_stats(h, ids, lm_head)
            if cur is not None and ref_stats[pi] is not None:
                am_cur, nll, ntok = cur
                am_ref = ref_stats[pi][0]
                m = min(am_ref.numel(), am_cur.numel())
                argmax_flips += int((am_ref[:m] != am_cur[:m]).sum())
                argmax_tokens += m
                nll_sum += nll
                nll_tok += ntok
        res = {
            "scheme": sname, "spec": list(spec), "role": scheme_role(sname), "aggr": scheme_aggr(sname),
            "apply_s": round(apply_s, 2),
            "inputs_embeds_hidden": acc_hidden.finalize() if "inputs_embeds_hidden" in measure else None,
            "shared_kv_states": acc_kv.finalize() if "shared_kv_states" in measure else None,
            "body_proxy": {
                "argmax_flip_rate": (argmax_flips / argmax_tokens) if argmax_tokens else None,
                "n_positions": argmax_tokens,
                "teacher_forced_ppl": (math.exp(nll_sum / nll_tok) if nll_tok else None),
            },
        }
        per_scheme[sname] = res
        if verbose:
            kv = res["shared_kv_states"] or {}
            hh = res["inputs_embeds_hidden"] or {}
            bp = res["body_proxy"]
            print(f"  [{sname:>9}] {scheme_spec(sname)} apply={apply_s:5.1f}s "
                  f"kv_L2rel={kv.get('l2_relative', float('nan')):.6f} "
                  f"hid_L2rel={hh.get('l2_relative', float('nan')):.6f} "
                  f"flip={bp['argmax_flip_rate']} tf_ppl={bp['teacher_forced_ppl']}")

    for h in handles:
        h.remove()

    meta = {
        "n_prompts": n_prompts, "max_seq_len": max_seq_len,
        "seqlen_min": min(seqlens), "seqlen_max": max(seqlens),
        "seqlen_mean": sum(seqlens) / len(seqlens),
        "n_layers": n_layers, "n_kv_layers": n_kv, "kv_layers": kv_layers,
        "n_quantized_linears": len(linears),
        "load_s": round(load_s, 2), "ref_peak_vram_mb": ref_peak_mb,
        "lm_head_available": lm_head is not None and norm is not None,
        "cb3_codebook_norm_distortion": cb_distortion,
        "cb3_two_sided_coverage": cb3_info,
        "cb3ldlq_two_sided_coverage": cb3ldlq_info,
        "cb3_ldlq_calibrated": needs_ldlq,
        "cb3_ldlq_calib_prompts": (n_calib if needs_ldlq else 0),
        "cb3_ldlq_calib_min_tokens": (min(calib_counts.values()) if calib_counts else None),
        "cb3_ldlq_calib_s": round(calib_s, 2),
    }
    return per_scheme, meta


# ===========================================================================
# Section 7 -- propagation to demand-TPS + band tightening + verdict
# ===========================================================================

def _propagate_flip(flip):
    """flip = body greedy-argmax flip (per-position proxy for the drafter's verification target moving)
    -> UPPER bound on |Delta-top1|; through the #289 ladder to Delta-E[accepted] and via the #402 secant
    (962.27) to Delta-demand_TPS (Delta-coverage ~ Delta-top1, the largest single-position drop)."""
    if flip is None:
        return None
    ladder_lo = list(LADDER_289)
    ladder_lo[0] = max(0.0, ladder_lo[0] - flip)
    de_acc = expected_accepted(LADDER_289) - expected_accepted(ladder_lo)
    return {"delta_top1_bound": flip, "delta_e_accepted_bound": de_acc,
            "delta_coverage_bound": flip, "delta_demand_tps": flip * GROSS_TPS_PER_UNIT_COV}


def _find(schemes, kind=None, bits=None):
    for s in schemes:
        k, b = scheme_spec(s)
        if kind is not None and k != kind:
            continue
        if bits is not None and b != bits:
            continue
        return s
    return None


def propagate_and_verdict(per_scheme, schemes):
    s_ldlq = _find(schemes, kind="cb3ldlq")          # the FAITHFUL (calibrated) rung -> headline
    s_cb3 = _find(schemes, kind="cb3real")           # the data-free literal recipe -> strict UPPER bound
    s8 = _find(schemes, bits=8)
    s4 = _find(schemes, bits=4)
    s3 = _find(schemes, bits=3)

    def flip(s):
        return per_scheme[s]["body_proxy"]["argmax_flip_rate"] if s and per_scheme.get(s) else None

    ldlq_prop = _propagate_flip(flip(s_ldlq))
    datafree_prop = _propagate_flip(flip(s_cb3))
    int8_prop = _propagate_flip(flip(s8))
    int4_prop = _propagate_flip(flip(s4))

    # the headline cb3-real rung is the FAITHFUL (LDLQ) measurement when present; else the data-free one
    head_prop = ldlq_prop if ldlq_prop is not None else datafree_prop
    cb3_dtps = head_prop["delta_demand_tps"] if head_prop else None
    datafree_dtps = datafree_prop["delta_demand_tps"] if datafree_prop else None
    band_lo = 0.0                                   # faithful per-position read still blocked -> 0 floor
    band_hi = cb3_dtps                              # the new DIRECT (calibrated) cb3 upper bound

    out = {
        "cb3_ldlq_prop": ldlq_prop, "cb3_real_prop": datafree_prop,
        "int8_prop": int8_prop, "int4_prop": int4_prop,
        "headline_rung": ("cb3-ldlq" if ldlq_prop is not None else "cb3-real(data-free)"),
        "cb3_real_rung_delta_demand_tps": cb3_dtps,
        "cb3_datafree_upper_bound_dtps": datafree_dtps,
        "band_lo": band_lo, "band_hi": band_hi,
        "old_band_hi_int8_anchor_410": INT8_DTPS_410,
        "band_width_tightened_from_15p74": band_hi,                       # the new tighter upper bound
        "band_hi_delta_vs_int8_anchor": (band_hi - INT8_DTPS_410) if band_hi is not None else None,
        "tightened_below_int8_anchor": (band_hi is not None and band_hi < INT8_DTPS_410),
        "negligible_threshold_tps": NEGLIGIBLE_TPS,
        "additive_not_negligible_hi_tps": ADDITIVE_NOT_NEGLIGIBLE_HI_TPS,
        "demand_lift_tps_band": DEMAND_LIFT_TPS_BAND,
    }
    if band_hi is None:
        out["cross_term_now_negligible"] = None
        out["verdict_zone"] = "blocked:unmeasured"
        out["cross_term_destructive"] = None
        out["supply_demand_additive"] = None
        return out

    frac = band_hi / DEMAND_LIFT_TPS_BAND
    out["cb3_real_frac_of_demand_lift"] = frac
    # The band is [0, band_hi] -- band_lo is pinned at 0 because the FAITHFUL per-position top-K drafter read
    # is still blocked (#372) and the body-argmax flip only UPPER-bounds |Delta-top1|. So every verdict here is
    # an UPPER-BOUND certification: we can certify "negligible" / "additive" only when the upper bound sits
    # below the threshold. We can NEVER certify "destructive" (the realistic value may be ~0). frac>=0.50 means
    # additivity is NOT certifiable and the destructive zone is NOT excluded -- NOT that the cross-term IS destructive.
    out["cross_term_now_negligible"] = bool(band_hi < NEGLIGIBLE_TPS)     # certified small (upper bound < 5% of lift)
    out["additivity_certified"] = bool(frac < 0.50)                      # even the worst case stays additive
    out["destructive_not_excluded"] = bool(frac >= 0.50)                 # upper bound reaches the destructive zone
    out["supply_demand_additive"] = out["additivity_certified"]          # alias (#410 schema continuity)
    out["cross_term_destructive"] = out["destructive_not_excluded"]      # NOTE: "not excluded", NOT "is destructive"
    if band_hi < NEGLIGIBLE_TPS:
        out["verdict_zone"] = "negligible_certified_below_5pct"
    elif frac < 0.50:
        out["verdict_zone"] = "additive_certified_5_to_50pct"
    else:
        out["verdict_zone"] = "additivity_not_certified_destructive_not_excluded"
    return out


# ===========================================================================
# Section 8 -- self-tests (>= 20)
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def rht_roundtrip_check(codebook, device="cuda"):
    """Quant-free invRHT(RHT(W)) must reconstruct W (Hadamard orthogonality). Test on the real shapes."""
    import torch  # noqa: PLC0415
    worst = 0.0
    g = torch.Generator(device="cpu").manual_seed(7)
    for (out, inn) in [(512, 2560), (10240, 2560), (2560, 10240), (2048, 2560), (10752, 2560)]:
        W = torch.randn(out, inn, generator=g).to(torch.bfloat16)
        Wcpu = W
        # run the transform with an identity "codebook" path by short-circuiting VQ: reuse the math directly
        Wd = Wcpu.to(device=device, dtype=torch.float32)
        seed_mix = CB3_VQ_SEED ^ (out * 1000003 + inn)
        sL = _signs(out, seed_mix ^ 0x5DEECE66, device)
        sR = _signs(inn, seed_mix ^ 0x1234567, device)
        Hin = build_hadamard(inn, device)
        Hout = build_hadamard(out, device)
        two = Hout is not None and Hin is not None
        rin, rout = math.sqrt(inn), math.sqrt(out)
        Wt = Wd
        if two:
            Wt = (Hout @ (sL[:, None] * Wt)) / rout
        Wt = ((Wt * sR[None, :]) @ Hin.t()) / rin
        # inverse (no quant)
        Wh = ((Wt @ Hin) / rin) * sR[None, :]
        if two:
            Wh = sL[:, None] * (Hout.t() @ Wh) / rout
        rel = float((Wh - Wd).norm() / Wd.norm().clamp_min(1e-9))
        worst = max(worst, rel)
    return worst


def run_self_tests(per_scheme, schemes, measure, meta, prop, codebook, cb_distortion) -> dict:
    c = {}
    fp16 = _find(schemes, kind="ref")
    s_ldlq = _find(schemes, kind="cb3ldlq")     # FAITHFUL (calibrated) rung
    s_cb3 = _find(schemes, kind="cb3real")      # data-free literal recipe (loose UB)
    s_head = s_ldlq if s_ldlq is not None else s_cb3
    s8, s4, s3 = _find(schemes, bits=8), _find(schemes, bits=4), _find(schemes, bits=3)

    # a) fp16-vs-fp16 perturbation == 0 (within-process determinism; flip is the perturbation's)
    if fp16 is not None:
        for tn in measure:
            t = per_scheme[fp16][tn]
            c[f"a_fp16_{tn}_l2rel_zero"] = t["l2_relative"] < ZERO_TOL
            c[f"a_fp16_{tn}_cos_one"] = abs(t["cosine"] - 1.0) < ZERO_TOL

    # b) crude-RTN harness monotonicity (independent of the cb3 hypothesis): int8 <= int4 <= int3
    rtn_order = [s for s in (s8, s4, s3) if s]
    for tn in measure:
        seq = [per_scheme[s][tn]["l2_relative"] for s in rtn_order]
        c[f"b_rtn_{tn}_l2rel_monotone"] = all(seq[i] <= seq[i + 1] + MONO_EPS for i in range(len(seq) - 1))

    # c) #355 reproduction: int3 gate-DEAD scalar collapse (tf-ppl + flip >> int8); int8 gate-safe
    def bp(s, key):
        return per_scheme[s]["body_proxy"][key] if s and per_scheme.get(s) else None
    p8, p4, p3 = bp(s8, "teacher_forced_ppl"), bp(s4, "teacher_forced_ppl"), bp(s3, "teacher_forced_ppl")
    f8, f4, f3 = bp(s8, "argmax_flip_rate"), bp(s4, "argmax_flip_rate"), bp(s3, "argmax_flip_rate")
    if None not in (p8, p3):
        c["c_int3_ppl_collapse_reproduces_355"] = p3 > p8 * 1.10
    if None not in (f8, f3):
        c["c_int3_flip_collapse_reproduces_355"] = f3 > max(0.20, f8 * 5)
    if f8 is not None:
        c["c_int8_gate_safe_low_flip"] = f8 < 0.10

    # d) finite + in-range across all schemes
    okf = okr = True
    for s in schemes:
        for tn in measure:
            t = per_scheme[s][tn]
            okf = okf and _finite(t["l2_relative"]) and _finite(t["cosine"])
            okr = okr and (0.0 <= t["l2_relative"] < 2.5) and (-1.001 <= t["cosine"] <= 1.001)
    c["d_all_finite"] = okf
    c["d_all_in_range"] = okr

    # e) provenance round-trips
    c["e_cb3_byte_ratio_0p785"] = round(CB3_BYTE_RATIO, 3) == 0.785
    c["e_ladder_len_7"] = len(LADDER_289) == 7
    c["e_eaccepted_roundtrips_289"] = abs(expected_accepted(LADDER_289) - E_ACCEPTED_289) < 1e-9
    c["e_secant_962"] = abs(GROSS_TPS_PER_UNIT_COV - 962.27) < 1e-6
    c["e_demand_lift_positive"] = DEMAND_LIFT_TPS_BAND > 0
    c["e_vq_code_bpw_3p0"] = abs(CB3_VQ_CODE_BPW - 3.0) < 1e-9
    c["e_vq_eff_bpw_brackets_cb3"] = abs(CB3_VQ_EFF_BPW - CB3_BPW_EFF) < 0.05   # 3.25 vs 3.2369

    # f) measurement coverage
    c["f_prompts_used"] = meta["n_prompts"] >= 1
    c["f_layers_42"] = meta["n_layers"] == 42
    c["f_kv_layers_24"] = meta["n_kv_layers"] == 24

    # g) PPL gate untouched (0-TPS card)
    c["g_baseline_passes_gate"] = BASELINE_PPL <= PPL_GATE
    c["g_cb3_holds_gate"] = CB3_PPL <= PPL_GATE
    c["g_cb3_ppl_parity_with_int4"] = CB3_REL_PPL_OVER_INT4 < GATE_HEADROOM_REL

    # h) cb3-real HARNESS correctness
    if cb_distortion is not None:
        c["h_codebook_distortion_below_bound"] = cb_distortion < 0.05   # optimal 2D VQ @3b ~0.02-0.03
    c["h_rht_roundtrip_exact"] = (meta.get("rht_roundtrip_rel", 1.0) < RHT_ROUNDTRIP_TOL)
    cov = meta.get("cb3_two_sided_coverage") or {}
    if cov:
        c["h_cb3_two_sided_all"] = (cov.get("n", 0) > 0 and cov.get("one_sided", 1) == 0)
    lcov = meta.get("cb3ldlq_two_sided_coverage") or {}
    if lcov:
        c["h_cb3ldlq_two_sided_all"] = (lcov.get("n", 0) > 0 and lcov.get("one_sided", 1) == 0)
    if meta.get("cb3_ldlq_calibrated"):
        c["h_cb3ldlq_calib_full_rank"] = (meta.get("cb3_ldlq_calib_min_tokens") or 0) >= 10240

    # i) cb3-LDLQ HYPOTHESIS checks (the FAITHFUL rung). These are the *claim being tested*, NOT harness
    #    invariants -- a rank-deficient calib (smoke) or a genuine negative (LDLQ doesn't beat crude-int4)
    #    can flip them while the measurement is still perfectly trustworthy. So they are recorded in `c`
    #    (visible + counted) but do NOT gate `passes`; they gate `hypothesis_supported` instead.
    hyp = set()
    if s_ldlq is not None:
        for tn in measure:
            lv = per_scheme[s_ldlq][tn]["l2_relative"]
            v8 = per_scheme[s8][tn]["l2_relative"] if s8 else None
            v4 = per_scheme[s4][tn]["l2_relative"] if s4 else None
            if v8 is not None and v4 is not None:
                # the PR's `cb3_rung_sits_between_int8_and_crude_int4`: calibrated cb3 brackets [int8, int4]
                k = f"i_cb3ldlq_{tn}_brackets_int8_to_int4"
                c[k] = (v8 - MONO_EPS <= lv <= v4 + MONO_EPS); hyp.add(k)
        # full hypothesis ordering fp16<=int8<=cb3-ldlq<=int4<=int3 (KV); cb3-real(data-free) sits above int4
        if "shared_kv_states" in measure:
            ladder = [s for s in (fp16, s8, s_ldlq, s4, s3) if s]
            seq = [per_scheme[s]["shared_kv_states"]["l2_relative"] for s in ladder]
            c["i_full_order_monotone_kv_l2rel"] = all(seq[i] <= seq[i + 1] + MONO_EPS for i in range(len(seq) - 1))
            hyp.add("i_full_order_monotone_kv_l2rel")
        # calibrated cb3 is careful-int4-PPL-class, NOT crude-RTN (PR `cb3_fakequant_ppl_matches_careful_int4`)
        pl, fl = bp(s_ldlq, "teacher_forced_ppl"), bp(s_ldlq, "argmax_flip_rate")
        pf = per_scheme[fp16]["body_proxy"]["teacher_forced_ppl"] if fp16 else None
        if None not in (pl, p4, pf):
            ldlq_gap = pl / pf - 1.0
            crude_gap = p4 / pf - 1.0
            c["i_cb3ldlq_ppl_careful_not_crude"] = (pl < p4) and (ldlq_gap < CAREFUL_PPL_FRACTION_OF_CRUDE * crude_gap)
            c["i_cb3_fakequant_ppl_matches_careful_int4"] = c["i_cb3ldlq_ppl_careful_not_crude"]
            hyp.update({"i_cb3ldlq_ppl_careful_not_crude", "i_cb3_fakequant_ppl_matches_careful_int4"})
        if None not in (fl, f4):
            c["i_cb3ldlq_flip_below_crude_int4"] = fl <= f4 + 1e-9; hyp.add("i_cb3ldlq_flip_below_crude_int4")
        # cross-term-not-destructive is also hypothesis-dependent (destructive iff cb3 turns out int4-class)
        c["i_cross_term_not_destructive"] = (prop.get("cross_term_destructive") is False)
        hyp.add("i_cross_term_not_destructive")
        # band_hi finite+nonneg IS a harness invariant (the propagation must produce a real number)
        bh = prop.get("band_hi")
        c["i_band_hi_finite_positive"] = _finite(bh) and bh is not None and bh >= 0

    # j) data-free cb3-real bounds.
    #    `_is_upper_bound_on_ldlq` is a near-invariant of the no-LDLQ path: GPTQ/LDLQ error-feedback
    #    minimizes tr(E H E^T) == E||E x||^2, the SAME activation-perturbation energy this card measures,
    #    so ldlq <= data-free. It GATES `passes`.
    #    `_above_int4` is a CROSS-METHOD empirical ordering (3.25bpw data-free RHT+VQ vs 4.125bpw scalar
    #    int4-RTN). If RHT incoherence ALONE (no LDLQ) beats crude int4, that is a RESULT, not a harness
    #    bug -- so it is a HYPOTHESIS claim (recorded + counted, gates `hypothesis_supported`), NOT an
    #    invariant. Gating `passes` on it would let a good incoherence outcome masquerade as a broken harness.
    if s_cb3 is not None:
        for tn in measure:
            dv = per_scheme[s_cb3][tn]["l2_relative"]
            v4 = per_scheme[s4][tn]["l2_relative"] if s4 else None
            if v4 is not None:
                k = f"j_cb3datafree_{tn}_above_int4"
                c[k] = dv >= v4 - MONO_EPS
                hyp.add(k)
            if s_ldlq is not None:
                lv = per_scheme[s_ldlq][tn]["l2_relative"]
                c[f"j_cb3datafree_{tn}_is_upper_bound_on_ldlq"] = dv >= lv - MONO_EPS

    # `passes` = HARNESS validity (every non-hypothesis invariant). The hypothesis outcome is reported
    # separately so a genuine negative (or a rank-deficient smoke) never masquerades as a broken harness.
    inv = {k: v for k, v in c.items() if k not in hyp}
    passes = all(inv.values())
    hyp_present = {k: c[k] for k in hyp if k in c}
    hypothesis_supported = bool(hyp_present) and all(hyp_present.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v),
            "passes": passes, "n_invariants": len(inv), "n_invariants_passed": sum(1 for v in inv.values() if v),
            "hypothesis_keys": sorted(hyp), "hypothesis_supported": hypothesis_supported,
            "n_hypothesis_passed": sum(1 for v in hyp_present.values() if v), "n_hypothesis": len(hyp_present)}


# ===========================================================================
# Section 9 -- report assembly + W&B + print + CLI
# ===========================================================================

def build_report(per_scheme, meta, schemes, measure, prop, codebook, cb_distortion) -> dict:
    s_ldlq = _find(schemes, kind="cb3ldlq")
    s_cb3 = _find(schemes, kind="cb3real")
    s_head = s_ldlq if s_ldlq is not None else s_cb3     # faithful rung headline (fallback to data-free)
    selftest = run_self_tests(per_scheme, schemes, measure, meta, prop, codebook, cb_distortion)

    def kvl2(s):
        return per_scheme[s]["shared_kv_states"]["l2_relative"] if (s and "shared_kv_states" in measure) else float("nan")

    def hidl2(s):
        return per_scheme[s]["inputs_embeds_hidden"]["l2_relative"] if (s and "inputs_embeds_hidden" in measure) else float("nan")

    def flipof(s):
        return per_scheme[s]["body_proxy"]["argmax_flip_rate"] if s else None

    def pplof(s):
        return per_scheme[s]["body_proxy"]["teacher_forced_ppl"] if s else None

    fp16 = _find(schemes, kind="ref")
    s4 = _find(schemes, bits=4)
    fp16_ppl = pplof(fp16)
    crude4_ppl = pplof(s4)
    head_ppl = pplof(s_head)
    head_flip = flipof(s_head)
    df_ppl = pplof(s_cb3)

    ppl_match = selftest["conditions"].get("i_cb3_fakequant_ppl_matches_careful_int4")
    bkey = "i_cb3ldlq" if s_ldlq is not None else "i_cb3real"
    brackets = all(selftest["conditions"].get(f"{bkey}_{tn}_brackets_int8_to_int4", True) for tn in measure)

    return {
        "pr": 422, "agent": "ubel", "kind": "cb3-real-rung-perturbation",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used_for_analysis_only": True, "official_tps": 0,
        "baseline_unchanged_tps": BASELINE_TPS, "baseline_unchanged_ppl": BASELINE_PPL,
        "inputs": {
            "model_id": MODEL_ID, "schemes": schemes, "measure": measure,
            "cb3_real_recipe": "two-sided RHT incoherence + K=64 dim-2 Gaussian VQ (data-free, NO LDLQ -> strict UB)",
            "cb3_ldlq_recipe": "two-sided RHT + K=64 dim-2 VQ + block-pair GPTQ/LDLQ error-feedback (calibrated -> FAITHFUL rung)",
            "cb3_vq_k": CB3_VQ_K, "cb3_vq_dim": CB3_VQ_DIM, "cb3_vq_code_bpw": CB3_VQ_CODE_BPW,
            "cb3_vq_scale_group": CB3_VQ_SCALE_GROUP, "cb3_vq_eff_bpw": CB3_VQ_EFF_BPW,
            "cb3_ldlq_blocksize": CB3_LDLQ_BLOCKSIZE, "cb3_ldlq_damp_frac": CB3_LDLQ_DAMP_FRAC,
            "cb3_bpw_eff_banked": CB3_BPW_EFF, "cb3_byte_ratio": CB3_BYTE_RATIO, "cb3_mixed_frac": CB3_MIXED_FRAC,
            "int4_bpw": INT4_BPW, "ladder_289": LADDER_289, "drafter_top1_289": DRAFTER_TOP1_289,
            "e_accepted_289": E_ACCEPTED_289, "gross_tps_per_unit_cov_402": GROSS_TPS_PER_UNIT_COV,
            "demand_coverage_band_401": DEMAND_COVERAGE_BAND, "demand_lift_tps_band": DEMAND_LIFT_TPS_BAND,
            "negligible_frac": NEGLIGIBLE_FRAC, "negligible_tps": NEGLIGIBLE_TPS,
            "baseline_tps": BASELINE_TPS, "baseline_ppl": BASELINE_PPL, "cb3_ppl_388": CB3_PPL, "ppl_gate": PPL_GATE,
            "int8_dtps_410": INT8_DTPS_410, "int8_kv_l2rel_410": INT8_KV_L2REL_410, "int4_kv_l2rel_410": INT4_KV_L2REL_410,
            "fp16_tf_ppl_410": FP16_TF_PPL_410, "int4_tf_ppl_410": INT4_TF_PPL_410,
            "cb3_vq_seed": CB3_VQ_SEED, "source_410_run": "7rzf74q5", "source_289_run": "fi34s269",
            "source_401_run": "i2qsjyp6", "source_402_run": "8pcyhe2r", "source_355_run": "vqzzc9jw",
            "source_388_ref": "cb3_supply_lift_mtp_honest", "source_kernel_ref": "cb3_kernel_realized_bw",
        },
        "meta": meta, "per_scheme": per_scheme, "propagation": prop,
        # ---- headline scalars (PR instruction 6); the cb3-real rung = the FAITHFUL (cb3-ldlq) measurement ----
        "headline_rung": prop.get("headline_rung"),
        "cb3_real_rung_delta_demand_tps": prop.get("cb3_real_rung_delta_demand_tps"),  # PRIMARY (faithful)
        "cb3_real_rung_kv_l2_relative": kvl2(s_head),
        "cb3_real_rung_hidden_l2_relative": hidl2(s_head),
        "cb3_real_rung_body_argmax_flip": head_flip,
        "cb3_real_tf_ppl": head_ppl, "fp16_tf_ppl": fp16_ppl, "crude_int4_tf_ppl": crude4_ppl,
        "cb3_real_rel_ppl_over_fp16": (head_ppl / fp16_ppl - 1.0) if (head_ppl and fp16_ppl) else None,
        "crude_int4_rel_ppl_over_fp16": (crude4_ppl / fp16_ppl - 1.0) if (crude4_ppl and fp16_ppl) else None,
        "cb3_fakequant_ppl_matches_careful_int4": bool(ppl_match) if ppl_match is not None else None,
        "cb3_rung_sits_between_int8_and_crude_int4": bool(brackets),
        # ---- the data-free LITERAL recipe = strict UPPER bound (the honest "recipe-without-LDLQ" rung) ----
        "cb3_datafree_kv_l2_relative": kvl2(s_cb3),
        "cb3_datafree_hidden_l2_relative": hidl2(s_cb3),
        "cb3_datafree_body_argmax_flip": flipof(s_cb3),
        "cb3_datafree_tf_ppl": df_ppl,
        "cb3_datafree_upper_bound_dtps": prop.get("cb3_datafree_upper_bound_dtps"),
        "band_lo": prop.get("band_lo"), "band_hi": prop.get("band_hi"),
        "band_width_tightened_from_15p74": prop.get("band_width_tightened_from_15p74"),
        "tightened_below_int8_anchor": prop.get("tightened_below_int8_anchor"),
        "cross_term_now_negligible": prop.get("cross_term_now_negligible"),
        "verdict_zone": prop.get("verdict_zone"),
        "cb3_real_frac_of_demand_lift": prop.get("cb3_real_frac_of_demand_lift"),
        "additivity_certified": prop.get("additivity_certified"),
        "destructive_not_excluded": prop.get("destructive_not_excluded"),
        "supply_demand_additive": prop.get("supply_demand_additive"),
        "cross_term_destructive": prop.get("cross_term_destructive"),
        "self_test": selftest,
        "cb3_real_rung_self_test_passes": selftest["passes"],            # HARNESS validity (invariants only)
        "cb3_real_rung_hypothesis_supported": selftest["hypothesis_supported"],  # the cb3<=int4 tightening claim
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"

    def blk(x):
        return "blocked:unmeasured" if x is None else x
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        summ = {
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "headline_rung": report.get("headline_rung"),
            "cb3_real_rung_delta_demand_tps": blk(report["cb3_real_rung_delta_demand_tps"]),
            "cb3_real_rung_kv_l2_relative": report["cb3_real_rung_kv_l2_relative"],
            "cb3_real_rung_hidden_l2_relative": report["cb3_real_rung_hidden_l2_relative"],
            "cb3_real_rung_body_argmax_flip": blk(report["cb3_real_rung_body_argmax_flip"]),
            "cb3_real_tf_ppl": blk(report["cb3_real_tf_ppl"]), "fp16_tf_ppl": blk(report["fp16_tf_ppl"]),
            "crude_int4_tf_ppl": blk(report["crude_int4_tf_ppl"]),
            "cb3_real_rel_ppl_over_fp16": blk(report["cb3_real_rel_ppl_over_fp16"]),
            "cb3_fakequant_ppl_matches_careful_int4": blk(report["cb3_fakequant_ppl_matches_careful_int4"]),
            "cb3_rung_sits_between_int8_and_crude_int4": report["cb3_rung_sits_between_int8_and_crude_int4"],
            # data-free literal recipe = strict upper bound
            "cb3_datafree_kv_l2_relative": report.get("cb3_datafree_kv_l2_relative"),
            "cb3_datafree_body_argmax_flip": blk(report.get("cb3_datafree_body_argmax_flip")),
            "cb3_datafree_tf_ppl": blk(report.get("cb3_datafree_tf_ppl")),
            "cb3_datafree_upper_bound_dtps": blk(report.get("cb3_datafree_upper_bound_dtps")),
            "band_lo": report["band_lo"], "band_hi": blk(report["band_hi"]),
            "band_width_tightened_from_15p74": blk(report["band_width_tightened_from_15p74"]),
            "old_band_hi_int8_anchor_410": INT8_DTPS_410,
            "tightened_below_int8_anchor": blk(report["tightened_below_int8_anchor"]),
            "cross_term_now_negligible": blk(report["cross_term_now_negligible"]),
            "verdict_zone": report["verdict_zone"], "supply_demand_additive": blk(report["supply_demand_additive"]),
            "cross_term_destructive": blk(report["cross_term_destructive"]),
            "additivity_certified": blk(report.get("additivity_certified")),
            "destructive_not_excluded": blk(report.get("destructive_not_excluded")),
            "cb3_real_frac_of_demand_lift": blk(report.get("cb3_real_frac_of_demand_lift")),
            "cb3_real_rung_self_test_passes": report["cb3_real_rung_self_test_passes"],
            "cb3_real_rung_hypothesis_supported": report["cb3_real_rung_hypothesis_supported"],
            "n_self_test_checks": report["self_test"]["n_checks"], "n_self_test_passed": report["self_test"]["n_passed"],
            "n_invariants": report["self_test"].get("n_invariants"), "n_invariants_passed": report["self_test"].get("n_invariants_passed"),
            "n_hypothesis": report["self_test"].get("n_hypothesis"), "n_hypothesis_passed": report["self_test"].get("n_hypothesis_passed"),
            "cb3_codebook_norm_distortion": report["meta"].get("cb3_codebook_norm_distortion"),
            "rht_roundtrip_rel": report["meta"].get("rht_roundtrip_rel"),
            "cb3_ldlq_calibrated": report["meta"].get("cb3_ldlq_calibrated"),
            "cb3_ldlq_calib_prompts": report["meta"].get("cb3_ldlq_calib_prompts"),
            "cb3_ldlq_calib_min_tokens": report["meta"].get("cb3_ldlq_calib_min_tokens"),
            "ref_peak_vram_mb": report["meta"]["ref_peak_vram_mb"], "n_prompts": report["meta"]["n_prompts"],
        }
        wandb.summary.update(summ)
        for sname, res in report["per_scheme"].items():
            aggr = res["aggr"]
            for tn in ("shared_kv_states", "inputs_embeds_hidden"):
                t = res.get(tn)
                if t is None:
                    continue
                wandb.log({"scheme/aggr": aggr,
                           f"perturb/{tn}/l2_relative": t["l2_relative"], f"perturb/{tn}/cosine": t["cosine"],
                           f"perturb/{tn}/max_abs_delta": t["max_abs_delta"],
                           f"perturb/{tn}/per_channel_max_abs_delta_max": t["per_channel_max_abs_delta_max"]})
            bpx = res.get("body_proxy") or {}
            wandb.log({"scheme/aggr": aggr,
                       "body_proxy/argmax_flip_rate": bpx.get("argmax_flip_rate") or float("nan"),
                       "body_proxy/teacher_forced_ppl": bpx.get("teacher_forced_ppl") or float("nan")})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    print("\n=== cb3-REAL rung perturbation: RHT+VQ fake-quant (PR #422, ubel) ===")
    m = r["meta"]
    print(f"body={MODEL_ID}  prompts={m['n_prompts']}  seqlen[min/mean/max]="
          f"{m['seqlen_min']}/{m['seqlen_mean']:.0f}/{m['seqlen_max']}  quantized_linears={m['n_quantized_linears']}  "
          f"peak_vram={m['ref_peak_vram_mb']}MB  codebook_distortion={m.get('cb3_codebook_norm_distortion')}")
    cov = m.get("cb3_two_sided_coverage") or {}
    lcov = m.get("cb3ldlq_two_sided_coverage") or {}
    print(f"  cb3 RHT coverage(data-free): {cov}  cb3-ldlq: {lcov}  rht_roundtrip_rel={m.get('rht_roundtrip_rel')}")
    if m.get("cb3_ldlq_calibrated"):
        print(f"  cb3-ldlq calibrated on {m.get('cb3_ldlq_calib_prompts')} prompts "
              f"(min {m.get('cb3_ldlq_calib_min_tokens')} tok/linear; full-rank>=10240) in {m.get('cb3_ldlq_calib_s')}s")
    print("\n-- perturbation at the drafter-input tensors (vs bf16; #410 monotone curve + cb3-ldlq(faithful) + cb3-real(data-free UB)) --")
    print(f"   {'scheme':>9} {'aggr':>4} | {'kv_L2rel':>9} {'kv_cos':>8} | {'hid_L2rel':>9} | {'flip':>7} {'tf_ppl':>8}")
    for s in sorted(r["per_scheme"], key=lambda x: r["per_scheme"][x]["aggr"]):
        res = r["per_scheme"][s]
        kv = res.get("shared_kv_states") or {}
        hh = res.get("inputs_embeds_hidden") or {}
        bp = res.get("body_proxy") or {}
        print(f"   {s:>9} {res['aggr']:>4.1f} | {kv.get('l2_relative', float('nan')):>9.5f} "
              f"{kv.get('cosine', float('nan')):>8.5f} | {hh.get('l2_relative', float('nan')):>9.5f} | "
              f"{(bp.get('argmax_flip_rate') or float('nan')):>7.4f} {(bp.get('teacher_forced_ppl') or float('nan')):>8.3f}")
    p = r["propagation"]
    print(f"\n   headline rung = {r.get('headline_rung')}  (the FAITHFUL cb3 supply-rung the band tightens to)")
    print(f"   cb3-real(faithful) KV L2-rel = {r['cb3_real_rung_kv_l2_relative']:.6f}  (int8 {INT8_KV_L2REL_410:.5f} .. "
          f"crude-int4 {INT4_KV_L2REL_410:.5f});  brackets[int8,int4]={r['cb3_rung_sits_between_int8_and_crude_int4']}")
    print(f"   cb3-real(faithful) flip = {r['cb3_real_rung_body_argmax_flip']}  tf_ppl={r['cb3_real_tf_ppl']} "
          f"(fp16 {r['fp16_tf_ppl']}, crude-int4 {r['crude_int4_tf_ppl']});  ppl_careful_not_crude="
          f"{r['cb3_fakequant_ppl_matches_careful_int4']}")
    print(f"   cb3 data-free UB (literal recipe, no LDLQ): KV L2-rel={r.get('cb3_datafree_kv_l2_relative')} "
          f"flip={r.get('cb3_datafree_body_argmax_flip')} -> UB dtps={r.get('cb3_datafree_upper_bound_dtps')}")
    print("\n-- PRIMARY: propagate to demand-TPS (#402 secant 962.27) + band tightening --")
    print(f"   cb3_real_rung_delta_demand_tps = {r['cb3_real_rung_delta_demand_tps']}  (PRIMARY, faithful rung)")
    print(f"   band [lo, hi] = [{p['band_lo']}, {p['band_hi']}]   old int8-anchor hi = {INT8_DTPS_410:.3f}")
    print(f"   band_width_tightened_from_15p74 = {r['band_width_tightened_from_15p74']}  "
          f"(delta_vs_int8={p.get('band_hi_delta_vs_int8_anchor')}, tightened_below={p.get('tightened_below_int8_anchor')})")
    print(f"   5% threshold={NEGLIGIBLE_TPS:.3f}  15%={ADDITIVE_NOT_NEGLIGIBLE_HI_TPS:.3f}  "
          f"frac_of_lift={p.get('cb3_real_frac_of_demand_lift')}")
    print(f"   verdict_zone = {p.get('verdict_zone')}  cross_term_now_negligible={r['cross_term_now_negligible']}")
    print(f"   band=[0,band_hi] -> UPPER-BOUND certification:  additivity_certified={p.get('additivity_certified')}  "
          f"destructive_not_excluded={p.get('destructive_not_excluded')}  (frac_of_lift={p.get('cb3_real_frac_of_demand_lift')})")
    st = r["self_test"]
    print(f"\nself-test (HARNESS validity): {st.get('n_invariants_passed')}/{st.get('n_invariants')} invariants  "
          f"cb3_real_rung_self_test_passes = {r['cb3_real_rung_self_test_passes']}")
    print(f"hypothesis (cb3<=crude-int4 tightening): {st.get('n_hypothesis_passed')}/{st.get('n_hypothesis')} claims  "
          f"hypothesis_supported = {r['cb3_real_rung_hypothesis_supported']}")
    inv_fails = [k for k, v in st["conditions"].items() if not v and k not in set(st.get("hypothesis_keys", []))]
    hyp_fails = [k for k, v in st["conditions"].items() if not v and k in set(st.get("hypothesis_keys", []))]
    if inv_fails:
        print(f"  INVARIANT FAILURES (harness bug): {inv_fails}")
    if hyp_fails:
        print(f"  hypothesis claims not met (a result, not a bug): {hyp_fails}")


def main() -> int:
    ap = argparse.ArgumentParser(description="cb3-real rung perturbation (PR #422).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quant-sweep", "--quant_sweep", dest="quant_sweep",
                    default="fp16,int8,cb3-ldlq,cb3-real,int4,int3")
    ap.add_argument("--measure", default="shared_kv_states,inputs_embeds_hidden")
    ap.add_argument("--self-test", action="store_true", help="fast reduced GPU gate (few prompts, both tensors)")
    ap.add_argument("--max-prompts", "--max_prompts", dest="max_prompts", type=int, default=128)
    ap.add_argument("--max-seq-len", "--max_seq_len", dest="max_seq_len", type=int, default=512)
    ap.add_argument("--calib-prompts", "--calib_prompts", dest="calib_prompts", type=int, default=None,
                    help="cb3-ldlq Hessian calibration prompt count (default: all measurement prompts). "
                         "A full-rank subset (tok>=10240, i.e. ~>=ceil(10240/seqlen) prompts) keeps the "
                         "calibration forward tractable while perturbation/flip stay measured on all --max-prompts.")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="cb3-real-rung-perturbation")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="ubel/cb3-real-rung-perturbation")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/cb3_real_rung_perturbation/cb3_real_rung_perturbation_results.json")
    args = ap.parse_args()

    schemes = [s for s in (x.strip() for x in args.quant_sweep.split(",")) if s]
    measure = [s for s in (x.strip() for x in args.measure.split(",")) if s]
    for s in schemes:
        scheme_spec(s)

    max_prompts = 4 if args.self_test else args.max_prompts
    max_seq_len = 128 if args.self_test else args.max_seq_len

    # cheap up-front harness check: RHT round-trip exactness (Hadamard orthogonality on real shapes)
    codebook, cb_distortion = gaussian_codebook_2d(device="cuda") if any(
        scheme_spec(s)[0] == "cb3real" for s in schemes) else (None, None)
    rht_rel = rht_roundtrip_check(codebook) if codebook is not None else None

    per_scheme, meta = run_measurement(schemes, measure, max_prompts, max_seq_len,
                                       calib_max_prompts=args.calib_prompts, verbose=True)
    meta["rht_roundtrip_rel"] = rht_rel
    prop = propagate_and_verdict(per_scheme, schemes)
    report = build_report(per_scheme, meta, schemes, measure, prop, codebook, cb_distortion)
    print_report(report)

    if args.self_test:
        out = HERE / "cb3_real_rung_perturbation_selftest.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out}\ncb3_real_rung_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "cb3_real_rung_delta_demand_tps": report["cb3_real_rung_delta_demand_tps"],
        "cb3_real_rung_kv_l2_relative": float(report["cb3_real_rung_kv_l2_relative"]),
        "cb3_real_rung_body_argmax_flip": report["cb3_real_rung_body_argmax_flip"],
        "band_lo": report["band_lo"], "band_hi": report["band_hi"],
        "band_width_tightened_from_15p74": report["band_width_tightened_from_15p74"],
        "tightened_below_int8_anchor": report["tightened_below_int8_anchor"],
        "cross_term_now_negligible": report["cross_term_now_negligible"],
        "cb3_fakequant_ppl_matches_careful_int4": report["cb3_fakequant_ppl_matches_careful_int4"],
        "cb3_rung_sits_between_int8_and_crude_int4": report["cb3_rung_sits_between_int8_and_crude_int4"],
        "headline_rung": report.get("headline_rung"),
        "cb3_datafree_upper_bound_dtps": report.get("cb3_datafree_upper_bound_dtps"),
        "verdict_zone": report["verdict_zone"], "supply_demand_additive": report["supply_demand_additive"],
        "additivity_certified": report.get("additivity_certified"), "destructive_not_excluded": report.get("destructive_not_excluded"),
        "cb3_real_frac_of_demand_lift": report.get("cb3_real_frac_of_demand_lift"),
        "cb3_real_rung_self_test_passes": bool(report["cb3_real_rung_self_test_passes"]),
        "cb3_real_rung_hypothesis_supported": bool(report["cb3_real_rung_hypothesis_supported"]),
        "primary_metric": {"name": "cb3_real_rung_delta_demand_tps", "value": report["cb3_real_rung_delta_demand_tps"]},
        "test_metric": {"name": "cb3_real_rung_self_test_passes", "value": float(report["cb3_real_rung_self_test_passes"])},
    }, default=str))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
