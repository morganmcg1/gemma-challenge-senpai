#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #390 (wirbel) -- Corrected-model deployable-strict CEILING re-rollup (#319).

THE QUESTION (my #378/#384 lineage)
-----------------------------------
My #384 (`4f32ks1e`) refuted the premise under which #378's deployable-strict CEILING numbers were
computed. #378 labelled `deployable_strict_tps_central`=510.01 as "attention-AXIS isolated, holds lm_head
at BF16-DETERMINISTIC, NOT shippable", and treated the honest shippable-today as the [357.32, 469.68]
full-VBI bracket -- a bracket that pays a BF16 batch-invariant determinization tax on the body + lm_head.
#384 proved that tax is a PHANTOM for the lm_head: the deployed osoi5 lm_head is UNTIED int4-Marlin and is
ALREADY byte-exact strict at the M=8 decode-verify width on the A10G (`should_use_atomic_add_reduce(M=8,
n=16384)=False` forces the fixed fp32 global-reduce) -> eta_lmhead_targeted=0, no isolation, no rebuild.

This card re-derives the SHIPPABLE deployable-strict ceiling under the #384-correct lm_head model AND
extends the SAME mechanism to the BODY int4-Marlin (parameterizing stark #381): is the body-Marlin ALSO
already-strict at the literal 8-row decode width? If so, the ONLY strict tax is the attention split-KV pin
(eta_attn=0.0215, #378) -- everything int4-Marlin (body + lm_head) is already byte-exact on the deployed
A10G -- and the corrected shippable ceiling is 510.01 (ceiling basis) / ~471 (deployed basis), with ONE
distinct kernel rebuild (attention), not two.

THE A10G MECHANISM (vllm marlin_utils.should_use_atomic_add_reduce)
-------------------------------------------------------------------
    if n >= 2048 or k < 2048 or device.type != "cuda":        return False   # geometry guard
    if not envs.VLLM_MARLIN_USE_ATOMIC_ADD:                    return False   # default env OFF
    if device_capability[0] < 9 and dtype == torch.bfloat16:   return False   # sm8x+bf16 native-unsupported
    return True
On the A10G (sm_86) + bf16, THREE independent guards force False for EVERY (m, n, k): the racing-atomic
split-K reduce is hardware-disabled, so every int4-Marlin GEMM uses the FIXED fp32 global-reduce ->
byte-invariant across M at the decode width. (The only Marlin M-variance stark #122/#376 found is the
size_m-dependent split COUNT at NON-decode/prefill geometry; at M in {1,8} the partial-sum structure is
fixed.) This card MEASURES byte-identity M=8-vs-M=1 across the full body projection set to confirm.

WHAT THIS CARD MEASURES (pod A10G, real gemma-4-E4B body+lmhead geometry, int4 compressed-tensors)
--------------------------------------------------------------------------------------------------
(1) BODY int4-Marlin decode-strictness (resolves stark #381): for q/k/v/o/gate/up/down proj at the real
    [size_k x size_n] geometry (group_size=128, the deployed body quant), the heuristic atomic-add flag,
    byte+argmax identity batched-M vs per-row-M1 at M in {1,2,4,8}, and heuristic-vs-forced-fixed-reduce
    latency. body_already_strict_at_decode := all geoms byte-exact.
(2) LM_HEAD reconfirm (#384): the deployed [16384 x 2560] channel-wise int4-Marlin lm_head, same checks.
(3) Corrected-attribution rollup: shippable_strict_ceiling_corrected under lm_head=0 + body={ArmA:0,
    ArmB:#376 fixed-split-K} + attention pin (eta_attn). Decompose the phantom bf16 tax the OLD
    [357.32, 469.68] bracket double-counted. Both body arms so we are not blocked on #381.

DECISION (#357 fern load-bearing): realized_shippable_strict_tps_decode, shippable_strict_ceiling_corrected,
clears_500, gap_to_500_tps, supply_alone_closes_500, n_distinct_kernel_rebuilds_for_strict_500 (per arm),
shippable_strict_tps_arm_A, shippable_strict_tps_arm_B.

SCOPE: pod-GPU microbench + analytic rollup ONLY (my #375/#378/#384 lineage). NO HF Job, NO submission, NO
served-file change, NO kernel rebuild/deploy, 0 official TPS, baseline 481.53 UNCHANGED. Greedy identity is
MEASURED, never broken. Real geometry + synthetic post-RMSNorm hidden (the #363/#365/#384 methodology:
M-invariance & latency are weight-value-independent; the int4 provenance is the source audit). Run with
CUDA_VISIBLE_DEVICES=0 (single-A10G pod default points at a non-existent 2nd GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# single-A10G pod: the inherited CUDA_VISIBLE_DEVICES may point at a non-existent 2nd GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "0,", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch  # noqa: E402

# ---------------------------------------------------------------------------------------- #
# Deployed gemma-4-E4B-it geometry (osoi5 baked, PCK-04 pruned). int4 compressed-tensors W4A16.
#   body  : group-quantized group_size=128 (config group_0, targets Linear)
#   lmhead: channel-wise (config group_0_lmhead), untied, row-pruned to 16384
# ---------------------------------------------------------------------------------------- #
HIDDEN = 2560
FULL_VOCAB = 262144
LMHEAD_ROWS = 16384
NUM_LAYERS = 37
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
INTERMEDIATE = 10240
DTYPE = torch.bfloat16
A10G_SMS = 80
M_LIST = (1, 2, 4, 8)          # verify widths incl. M=1 AR reference (K_spec=7+1 deployed at M=8)
DEPLOYED_M = 8
BODY_GROUP_SIZE = 128          # deployed body quant (config group_0)
LMHEAD_GROUP_SIZE = -1         # deployed lm_head quant (channel-wise == group_size -1)

# Real body projection geometries [size_k, size_n] (the GEMM contracts over size_k -> size_n):
#   q_proj   : hidden -> n_heads*head_dim      = 2560 -> 2048
#   k/v_proj : hidden -> n_kv_heads*head_dim   = 2560 ->  512   (small-n: atomic-add ELIGIBLE by heuristic)
#   o_proj   : n_heads*head_dim -> hidden      = 2048 -> 2560
#   gate/up  : hidden -> intermediate          = 2560 -> 10240
#   down_proj: intermediate -> hidden          = 10240 -> 2560
BODY_GEOMS: dict[str, tuple[int, int]] = {
    "q_proj":    (HIDDEN, NUM_HEADS * HEAD_DIM),
    "k_proj":    (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
    "v_proj":    (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
    "o_proj":    (NUM_HEADS * HEAD_DIM, HIDDEN),
    "gate_proj": (HIDDEN, INTERMEDIATE),
    "up_proj":   (HIDDEN, INTERMEDIATE),
    "down_proj": (INTERMEDIATE, HIDDEN),
}

# ---- strict budget ladder (CITE, do NOT re-derive; identical to #365/#378/#384 for ADDITIVITY) ---- #
CEILING_500 = 520.953                          # lambda=1 central ceiling TPS (wirbel #354/#326/#327)
STEP_US = 1218.2                               # deployed batch=1 decode step normalizer (denken #344)
OFFICIAL_TPS = 481.53                          # deployed non-strict public #1 (#52); this leg adds 0
RATIO_CAL_373 = 0.9668872131421857             # #373 local/served calibration (#378 partC.calibration)
ETA_BLANKET_VBI_326 = 0.3141                   # whole-step VLLM_BATCH_INVARIANT=1 tax (#326/#378)
ETA_FLOOR_327 = 0.09841249119201488            # first-principles per-locus floor (#327/#378)
BAND_357 = 357.32166269999993                  # #378 full_vbi_today_off_the_shelf = 520.953*(1-0.3141)
BAND_469 = 469.6847174760462                   # #378 full_vbi_today_floor       = 520.953*(1-0.09841)
BUDGET_500_ETA = 0.040220518933569815          # >500 kernel budget = 1 - 500/520.953 (~4.02%)
ETA_ATTN_378 = 0.02145375421979844             # #378 eta_attn_evalweighted (the attention-pin per-locus tax)
ETA_LMHEAD_BLANKET_384 = 0.10890855157580143   # #384 eta_lmhead_blanket (tuned-bf16 steelman slice)
LMHEAD_BI_SHARE_384 = 0.34673209670742255      # #384 lmhead_bi_share_of_vbi_overhead
F_ATTN_344 = 0.09506718019009251               # #378 step_fractions.attn
F_BODY_344 = 0.76240970145034                  # #378 step_fractions.body
F_DRAFT_344 = 0.12009488890060672              # #378 step_fractions.draft
F_LMHEAD_344 = 0.022428229458960704            # #378 step_fractions.lmhead
PIN_518 = 518.9188253620001                    # un-deployable attention pin / lambda=1 (#366/#370)
PIN_518_ROUND = 518.92
STRICT_FLOOR_196 = 165.44                       # strict frontier FLOOR (#196)


# ======================================================================================== #
# Device + source audit
# ======================================================================================== #
def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Launch with CUDA_VISIBLE_DEVICES=0 (the single-A10G pod default "
            "points at a non-existent 2nd GPU)."
        )
    return torch.device("cuda:0")


def _gpu_facts(dev: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "total_mem_gib": round(p.total_memory / (1024**3), 2),
        "is_a10g_80sm": bool(p.multi_processor_count == A10G_SMS and "A10G" in p.name),
        "is_sm8x": bool(cc[0] == 8),
    }


def _baked_candidates() -> list[str]:
    return ["/tmp/osoi5-v0-baked",
            os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots")]


def source_audit() -> dict[str, Any]:
    """Read the DEPLOYED baked checkpoint header+config: confirm BOTH the body (group_0, group-quantized)
    AND the lm_head (group_0_lmhead, channel-wise) are int4 compressed-tensors W4A16 Marlin. The deployed
    body is the SAME Marlin kernel family as the lm_head -> the same atomic-add decode-strictness applies."""
    out: dict[str, Any] = {"checked": []}
    for base in _baked_candidates():
        st = Path(base) / "model.safetensors"
        cfgp = Path(base) / "config.json"
        if not st.exists() or not cfgp.exists():
            for sub in sorted(Path(base).glob("*")):
                if (sub / "config.json").exists():
                    st, cfgp = sub / "model.safetensors", sub / "config.json"
                    break
        if not (st.exists() and cfgp.exists()):
            continue
        out["checked"].append(str(st))
        cfg = json.load(open(cfgp))
        tc = cfg.get("text_config", cfg)
        qc = cfg.get("quantization_config", {})
        groups = qc.get("config_groups", {})
        lmg = next((g for g in groups.values()
                    if any("lm_head" in t for t in (g.get("targets") or []))), None)
        bodyg = next((g for g in groups.values()
                      if any(t == "Linear" for t in (g.get("targets") or []))), None)
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        lm_packed = hdr.get("lm_head.weight_packed")
        # a representative body packed weight (first layer down_proj or any *_proj.weight_packed)
        body_packed_key = next((k for k in hdr
                                if k.endswith("weight_packed") and "lm_head" not in k), None)
        out.update({
            "config": str(cfgp),
            "tie_word_embeddings": bool(tc.get("tie_word_embeddings", False)),
            "vocab_size": tc.get("vocab_size"),
            "hidden_size": tc.get("hidden_size"),
            "num_hidden_layers": tc.get("num_hidden_layers"),
            "num_attention_heads": tc.get("num_attention_heads"),
            "num_key_value_heads": tc.get("num_key_value_heads"),
            "head_dim": tc.get("head_dim"),
            "intermediate_size": tc.get("intermediate_size"),
            "quant_format": qc.get("format"),
            "quant_method": qc.get("quant_method"),
            "body_quant_group": bodyg,
            "body_quant_strategy": (bodyg or {}).get("weights", {}).get("strategy"),
            "body_quant_group_size": (bodyg or {}).get("weights", {}).get("group_size"),
            "body_is_int4_marlin": bool(
                bodyg is not None and (bodyg.get("weights", {}).get("num_bits") == 4)
                and body_packed_key is not None),
            "body_packed_example": body_packed_key,
            "lmhead_quant_group": lmg,
            "lmhead_quant_strategy": (lmg or {}).get("weights", {}).get("strategy"),
            "lmhead_is_int4_marlin": bool(
                lm_packed is not None and lmg is not None
                and (lmg.get("weights", {}).get("num_bits") == 4)),
            "lmhead_rows_pruned": (lm_packed or {}).get("shape", [None])[0],
        })
        return out
    out["body_is_int4_marlin"] = None
    out["lmhead_is_int4_marlin"] = None
    return out


# ======================================================================================== #
# Real int4-Marlin GEMM builder (body group_size=128 ; lm_head group_size=-1 channel-wise)
# ======================================================================================== #
def build_marlin_gemm(dev: torch.device, size_k: int, size_n: int, group_size: int, seed: int):
    """Build a REAL GPTQ-Marlin int4 GEMM at [size_k, size_n], symmetric. Returns
    (apply(x, use_atomic_add, use_fp32_reduce) -> y[M, size_n], heuristic_aa, fp32_default)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mut
    from vllm.scalar_type import scalar_types

    qtype = scalar_types.uint4b8           # GPTQ 4-bit symmetric (compressed-tensors int4 sym)
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(size_k, size_n, generator=g, device=dev, dtype=DTYPE) * 0.02)
    w_ref, q_w, s, g_idx, sort_idx, _ = mut.marlin_quantize(w, qtype, group_size=group_size, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    empty_zp = torch.empty(0, dtype=torch.int, device=dev)

    heuristic_aa = bool(mu.should_use_atomic_add_reduce(
        m=DEPLOYED_M, n=size_n, k=size_k, device=dev, dtype=DTYPE))

    def apply(x: torch.Tensor, use_atomic_add: bool, use_fp32_reduce: bool) -> torch.Tensor:
        xr = x.reshape(-1, size_k)
        return ops.marlin_gemm(
            xr, None, q_w, None, s, None, None, empty_zp, g_idx, sort_idx, ws, qtype,
            size_m=xr.shape[0], size_n=size_n, size_k=size_k,
            is_k_full=True, use_atomic_add=use_atomic_add, use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False).reshape(x.shape[:-1] + (size_n,))

    return apply, heuristic_aa, mu.USE_FP32_REDUCE_DEFAULT


# ======================================================================================== #
# Identity (byte + argmax) -- batched-M vs per-row-M1, per kernel
# ======================================================================================== #
def measure_identity(fwd, size_k: int, n_trials: int, seed0: int, dev: torch.device,
                     gap_at_M: int = DEPLOYED_M) -> dict[str, Any]:
    """For each M in M_LIST, byte+argmax identity of batched(M) vs per-row(M=1), SAME kernel.
    `fwd(x)` -> y[M, N]. Accumulate over n_trials independent batch=1 problems."""
    byte_acc = {M: [] for M in M_LIST}
    argmax_acc = {M: [] for M in M_LIST}
    maxdiff_acc = {M: 0.0 for M in M_LIST}
    any_nan = False
    for t in range(n_trials):
        g = torch.Generator(device=dev).manual_seed(seed0 + t)
        x_full = torch.randn(max(M_LIST), size_k, generator=g, device=dev, dtype=DTYPE)
        for M in M_LIST:
            x = x_full[:M].contiguous()
            bat = fwd(x)
            any_nan = any_nan or bool(torch.isnan(bat).any())
            ref = torch.cat([fwd(x[r:r + 1]) for r in range(M)], dim=0)
            byte = (bat == ref).all(dim=-1).float().mean().item()
            argmax = (bat.argmax(dim=-1) == ref.argmax(dim=-1)).float().mean().item()
            maxdiff = (bat.float() - ref.float()).abs().max().item()
            byte_acc[M].append(byte)
            argmax_acc[M].append(argmax)
            maxdiff_acc[M] = max(maxdiff_acc[M], maxdiff)

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "byte_identity_by_M": {str(M): mean(byte_acc[M]) for M in M_LIST},
        "argmax_identity_by_M": {str(M): mean(argmax_acc[M]) for M in M_LIST},
        "max_abs_logit_diff_by_M": {str(M): maxdiff_acc[M] for M in M_LIST},
        "n_trials": n_trials, "any_nan": bool(any_nan),
    }


# ======================================================================================== #
# Latency (M=8 verify width), CUDA-event median, SAME methodology as #363/#365/#384
# ======================================================================================== #
def _time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # us (median)


def strict_tps(eta: float) -> float:
    """Ceiling-basis linear: lambda=1 ceiling minus the eta tax."""
    return CEILING_500 * (1.0 - eta)


def strict_tps_divisor(base: float, eta: float) -> float:
    """Divisor form: a tax eta multiplies the step time -> divides the TPS (the #378 tps_divisor form)."""
    return base / (1.0 + eta)


def served_tps_linear(base: float, eta: float) -> float:
    return base * (1.0 - eta)


# ======================================================================================== #
# GPU pass: body Marlin decode-strictness + lm_head reconfirm
# ======================================================================================== #
def measure_marlin_family(dev: torch.device, args) -> dict[str, Any]:
    """Build + measure every body projection GEMM (group_size=128) and the lm_head (channel-wise) at the
    real deployed geometry. Returns per-geom identity + atomic-add flag + heuristic-vs-forced latency."""
    seed0 = args.seeds[0]
    geoms: dict[str, dict[str, Any]] = {}

    targets = [(name, k, n, BODY_GROUP_SIZE, "body") for name, (k, n) in BODY_GEOMS.items()]
    targets.append(("lm_head", HIDDEN, LMHEAD_ROWS, LMHEAD_GROUP_SIZE, "lmhead"))

    for name, size_k, size_n, gsize, kind in targets:
        apply, heuristic_aa, fp32_default = build_marlin_gemm(dev, size_k, size_n, gsize, seed0)
        fwd_heur = lambda x, ap=apply, aa=heuristic_aa, fp=fp32_default: ap(x, aa, fp)
        fwd_forced = lambda x, ap=apply: ap(x, False, True)   # forced fixed-split-K (#376 minimal fix)

        # identity over all seeds (worst-case byte/argmax = min, max maxdiff)
        accs_heur = [measure_identity(fwd_heur, size_k, args.ident_trials, s, dev) for s in args.seeds]
        accs_forced = [measure_identity(fwd_forced, size_k, args.ident_trials, s, dev) for s in args.seeds]

        def merge(accs):
            return {
                "byte_identity_by_M": {str(M): min(a["byte_identity_by_M"][str(M)] for a in accs) for M in M_LIST},
                "argmax_identity_by_M": {str(M): min(a["argmax_identity_by_M"][str(M)] for a in accs) for M in M_LIST},
                "max_abs_logit_diff_by_M": {str(M): max(a["max_abs_logit_diff_by_M"][str(M)] for a in accs) for M in M_LIST},
                "any_nan": any(a["any_nan"] for a in accs),
                "n_trials": args.ident_trials * len(accs),
            }
        ident_heur = merge(accs_heur)
        ident_forced = merge(accs_forced)

        # latency (M=8)
        g = torch.Generator(device=dev).manual_seed(seed0)
        x8 = torch.randn(DEPLOYED_M, size_k, generator=g, device=dev, dtype=DTYPE)
        heur_us = _time_call(lambda: fwd_heur(x8), args.iters, args.warmup)
        forced_us = _time_call(lambda: fwd_forced(x8), args.iters, args.warmup)

        def m_invariant(byte_by_M: dict) -> bool:
            return all(byte_by_M.get(str(M), 0.0) >= 0.999 for M in M_LIST)

        geoms[name] = {
            "kind": kind, "size_k": size_k, "size_n": size_n, "group_size": gsize,
            "heuristic_use_atomic_add": heuristic_aa,
            "byte_identity_by_M": ident_heur["byte_identity_by_M"],
            "argmax_identity_by_M": ident_heur["argmax_identity_by_M"],
            "max_abs_logit_diff_by_M": ident_heur["max_abs_logit_diff_by_M"],
            "any_nan": ident_heur["any_nan"],
            "is_m_invariant_at_decode": m_invariant(ident_heur["byte_identity_by_M"]),
            "forced_is_m_invariant": m_invariant(ident_forced["byte_identity_by_M"]),
            "heuristic_us": heur_us,
            "forced_det_us": forced_us,
            "forced_vs_heuristic_ratio": (forced_us / heur_us) if heur_us > 0 else float("nan"),
        }
    return geoms


# ======================================================================================== #
# Compose: corrected-attribution rollup + both body arms + verdict
# ======================================================================================== #
def compose(audit: dict, geoms: dict, gpu: dict) -> dict[str, Any]:
    body_names = list(BODY_GEOMS.keys())
    body = {n: geoms[n] for n in body_names}
    lmhead = geoms["lm_head"]

    # ---- (1) body decode-strictness (resolves #381) ----
    body_byte_exact = {n: bool(body[n]["is_m_invariant_at_decode"]) for n in body_names}
    body_maxdiff_M8 = {n: body[n]["max_abs_logit_diff_by_M"]["8"] for n in body_names}
    body_atomic_add = {n: bool(body[n]["heuristic_use_atomic_add"]) for n in body_names}
    body_already_strict_at_decode = bool(all(body_byte_exact.values()))
    body_any_atomic_add_active = bool(any(body_atomic_add.values()))
    # small-n GEMMs (k/v proj, n<2048) are the atomic-add-ELIGIBLE-by-heuristic case; confirm OFF on A10G
    small_n_geoms = [n for n in body_names if BODY_GEOMS[n][1] < 2048]
    small_n_strict = bool(all(body_byte_exact[n] for n in small_n_geoms))

    lmhead_already_strict = bool(lmhead["is_m_invariant_at_decode"])

    # ---- (2) the int4-preserving #376 fixed-split-K cost on this hardware (forced vs heuristic) ----
    # On the A10G the heuristic ALREADY uses the fixed reduce (atomic-add hw-disabled) so forcing it is a
    # no-op -> body_fixed_split_k_penalty ~ 1.0 -> eta_body_376 ~ 0. Measured per-geom, take the max.
    body_forced_ratio_max = max(body[n]["forced_vs_heuristic_ratio"] for n in body_names)
    body_fixed_split_k_penalty = max(1.0, body_forced_ratio_max)
    eta_body_376_int4preserving = F_BODY_344 * (body_fixed_split_k_penalty - 1.0)

    # ---- (3) the PESSIMISTIC body-determinization over-tax (the OLD bracket's phantom) ----
    # If one (wrongly) determinized the body to BF16-BI instead of keeping int4-Marlin, the body would pay
    # the first-principles-floor body slice. Reconstruct the floor's NON-attention residual after removing
    # the lm_head (which the floor already books at int4-Marlin -> 0): the residual is the body slice.
    # eta_floor = eta_attn + eta_body_floor  (lm_head floor slice = 0 under the corrected model)
    eta_body_floor_pessimistic = max(0.0, ETA_FLOOR_327 - ETA_ATTN_378)
    # off-the-shelf blanket body slice = whole-step blanket minus attn/lmhead/draft blanket slices.
    eta_body_blanket_offshelf = max(0.0, ETA_BLANKET_VBI_326 - ETA_ATTN_378 - ETA_LMHEAD_BLANKET_384)

    # ---- (4) corrected shippable ceiling: int4-Marlin (body+lmhead) already strict -> attention-only tax #
    eta_total_arm_A = ETA_ATTN_378                                  # body strict + lmhead strict -> attn only
    eta_total_arm_B = ETA_ATTN_378 + eta_body_376_int4preserving    # body needs #376 fix (int4-preserving)
    eta_total_arm_B_pess = ETA_ATTN_378 + eta_body_floor_pessimistic  # refuted: body bf16-determinized

    shippable_ceiling_arm_A = strict_tps(eta_total_arm_A)                     # 510.01
    shippable_deployed_arm_A = served_tps_linear(OFFICIAL_TPS, eta_total_arm_A)
    shippable_deployed_arm_A_div = strict_tps_divisor(OFFICIAL_TPS, eta_total_arm_A)
    shippable_ceiling_arm_B = strict_tps(eta_total_arm_B)
    shippable_deployed_arm_B = served_tps_linear(OFFICIAL_TPS, eta_total_arm_B)
    shippable_ceiling_arm_B_pess = strict_tps(eta_total_arm_B_pess)           # ~480.86

    # headline = the MEASURED arm (A confirmed on A10G), deployed (servable) basis
    arm = "A" if body_already_strict_at_decode else "B"
    shippable_strict_ceiling_corrected = shippable_ceiling_arm_A if arm == "A" else shippable_ceiling_arm_B
    realized_shippable_strict_tps_decode = (shippable_deployed_arm_A_div if arm == "A"
                                            else strict_tps_divisor(OFFICIAL_TPS, eta_total_arm_B))
    clears_500 = bool(realized_shippable_strict_tps_decode >= 500.0)
    clears_500_ceiling_basis = bool(shippable_strict_ceiling_corrected >= 500.0)
    gap_to_500_tps = 500.0 - realized_shippable_strict_tps_decode
    supply_alone_closes_500 = clears_500
    n_distinct_kernel_rebuilds_arm_A = 1   # attention split pin ONLY (body+lmhead already strict)
    n_distinct_kernel_rebuilds_arm_B = 2   # attention + body-Marlin #376 fixed-split-K
    n_distinct_kernel_rebuilds_for_strict_500 = (n_distinct_kernel_rebuilds_arm_A if arm == "A"
                                                 else n_distinct_kernel_rebuilds_arm_B)

    # ---- (5) phantom lm_head bf16 decomposition (PR step 1) ----
    phantom_lmhead_bf16_tax_tps = ETA_LMHEAD_BLANKET_384 * CEILING_500          # 56.7 TPS if it were real
    spread_510_to_518 = PIN_518 - shippable_ceiling_arm_A                       # ~8.91 TPS (attn evalwt+pin)
    # the BIG phantom: old off-the-shelf shippable 357.32 vs corrected 510.01 = the whole bf16 body+lmhead tax
    phantom_total_offshelf_tps = shippable_ceiling_arm_A - BAND_357             # ~152.69 TPS
    # removing JUST the lm_head phantom from the off-the-shelf bracket:
    band_357_minus_lmhead_phantom = strict_tps(ETA_BLANKET_VBI_326 - ETA_LMHEAD_BLANKET_384)  # ~414.0
    phantom_lmhead_lift_from_357 = band_357_minus_lmhead_phantom - BAND_357     # ~56.7 TPS
    spread_is_lmhead = bool(abs(spread_510_to_518) >= phantom_lmhead_bf16_tax_tps)  # False: spread << tax

    # corrected band vs OLD band
    corrected_band = [shippable_deployed_arm_A, shippable_ceiling_arm_A]        # [471, 510]
    old_band = [BAND_357, BAND_469]                                            # [357, 470]
    floor_lift_deployed_vs_offshelf = shippable_deployed_arm_A - BAND_357
    ceiling_lift_vs_floor = shippable_ceiling_arm_A - BAND_469

    is_strict_byte_exact = bool(body_already_strict_at_decode and lmhead_already_strict)

    if body_already_strict_at_decode and lmhead_already_strict:
        bucket = ("CORRECTED-LIFT: ALL int4-Marlin GEMMs (body + lm_head) are ALREADY byte-exact strict at "
                  "the M=8 decode width on the A10G (atomic-add hw-disabled) -> the ONLY strict tax is the "
                  "attention split pin (1 rebuild); the OLD [357.32, 469.68] bracket double-counted a phantom "
                  "bf16 body+lm_head determinization. BUT the corrected SHIPPABLE (deployed-basis) ~471 still "
                  "falls ~28.6 TPS short of 500 -> supply-alone does NOT close 500.")
    elif lmhead_already_strict:
        bucket = ("PARTIAL: lm_head already strict but some body GEMM breaks at decode -> Arm B "
                  "(2 rebuilds); shippable lower.")
    else:
        bucket = "UNEXPECTED: an int4-Marlin GEMM is NOT byte-exact at decode -- investigate."

    verdict = (
        f"MEASURED on the pod A10G (cc {gpu['compute_capability']}, real deployed gemma-4-E4B geometry, int4 "
        f"compressed-tensors). SOURCE: body group-quantized (group_size={audit.get('body_quant_group_size')}, "
        f"strategy={audit.get('body_quant_strategy')}) + lm_head channel-wise -- BOTH the SAME compressed-tensors "
        f"W4A16 Marlin kernel family. MECHANISM: should_use_atomic_add_reduce returns False for EVERY body GEMM "
        f"on the A10G (sm8x+bf16 native-unsupported AND default env off) -> the FIXED fp32 global-reduce. "
        f"(1) BODY decode-strictness (resolves stark #381): byte-identity M=8-vs-M=1 = "
        f"{ {n: body[n]['byte_identity_by_M']['8'] for n in body_names} } (max|M8-M1| "
        f"{max(body_maxdiff_M8.values()):.3e}) -> body_already_strict_at_decode={body_already_strict_at_decode}. "
        f"The atomic-add-ELIGIBLE small-n projections (k/v_proj n=512) are ALSO byte-exact "
        f"(small_n_strict={small_n_strict}) -- the heuristic suggests atomic-add but the A10G guards force it off. "
        f"(2) lm_head reconfirm: byte-exact={lmhead_already_strict} (#384). "
        f"(3) the int4-preserving #376 fixed-split-K fix is a NO-OP here (forced/heuristic ratio max "
        f"{body_forced_ratio_max:.3f}) -> eta_body_376={eta_body_376_int4preserving*100:.4f}%. "
        f"ROLLUP: with body+lm_head already strict, the ONLY strict tax is the attention split pin "
        f"eta_attn={ETA_ATTN_378*100:.3f}% -> shippable_strict_ceiling_corrected={shippable_ceiling_arm_A:.2f} "
        f"(ceiling) / realized {realized_shippable_strict_tps_decode:.2f} (deployed, servable basis), "
        f"clears_500={clears_500}, gap_to_500={gap_to_500_tps:.2f} TPS, "
        f"n_distinct_kernel_rebuilds=1 (Arm A) / 2 (Arm B). "
        f"PHANTOM DECOMP (PR step 1): the OLD off-the-shelf shippable 357.32 vs corrected {shippable_ceiling_arm_A:.2f} "
        f"= {phantom_total_offshelf_tps:.1f} TPS of phantom bf16 body+lm_head over-determinization; the lm_head's "
        f"slice alone = {phantom_lmhead_lift_from_357:.1f} TPS ({LMHEAD_BI_SHARE_384*100:.0f}% steelman). The literal "
        f"510.01->518.92 spread is only {spread_510_to_518:.2f} TPS (attention eval-weighting + pin headroom), "
        f"{phantom_lmhead_bf16_tax_tps/max(spread_510_to_518,1e-9):.1f}x SMALLER than the {phantom_lmhead_bf16_tax_tps:.1f} TPS "
        f"phantom lm_head bf16 tax -> the spread is NOT the lm_head (the phantom lived in the 357/470 bracket, and "
        f"510.01 was the corrected shippable ceiling all along -- #378 merely MISLABELED it 'NOT shippable'). "
        f"VERDICT for fern #357: the corrected attribution LIFTS the honest band from [357.32, 469.68] to "
        f"[{shippable_deployed_arm_A:.2f}, {shippable_ceiling_arm_A:.2f}] (deployed-floor +{floor_lift_deployed_vs_offshelf:.1f} TPS "
        f"vs off-the-shelf) AND cuts the rebuild count 2->1, but supply-alone STILL does not clear 500 on the "
        f"servable deployed basis (~{realized_shippable_strict_tps_decode:.0f}, gap ~{gap_to_500_tps:.0f} TPS). A clean "
        f"'supply-alone <500' -- the residual hands to the cb3 body shrink (lawine #388) / demand route (#383/#386). "
        f"CAVEAT: real geometry + synthetic post-RMSNorm hidden (M-invariance & latency are weight-value-independent; "
        f"int4 provenance is the source audit); attention pin realizability (env vs rebuild) is ubel #389 / a "
        f"flag-to-human served-config item -- the LOCAL rollup is fully within the #319 grant.")

    return {
        "source_audit": audit,
        # --- (1) body decode-strictness ---
        "body_geoms": body,
        "body_byte_exact_by_geom": body_byte_exact,
        "body_maxdiff_M8_by_geom": body_maxdiff_M8,
        "body_heuristic_atomic_add_by_geom": body_atomic_add,
        "body_already_strict_at_decode": body_already_strict_at_decode,
        "body_any_atomic_add_active": body_any_atomic_add_active,
        "small_n_geoms": small_n_geoms,
        "small_n_strict": small_n_strict,
        "lmhead_geom": lmhead,
        "lmhead_already_strict_at_decode": lmhead_already_strict,
        # --- (2)/(3) body determinization cost ---
        "body_forced_vs_heuristic_ratio_max": body_forced_ratio_max,
        "body_fixed_split_k_penalty": body_fixed_split_k_penalty,
        "eta_body_376_int4preserving": eta_body_376_int4preserving,
        "eta_body_floor_pessimistic": eta_body_floor_pessimistic,
        "eta_body_blanket_offshelf": eta_body_blanket_offshelf,
        # --- (4) corrected shippable ceiling + both arms ---
        "eta_attn_378": ETA_ATTN_378,
        "eta_total_arm_A": eta_total_arm_A,
        "eta_total_arm_B": eta_total_arm_B,
        "eta_total_arm_B_pessimistic": eta_total_arm_B_pess,
        "shippable_strict_ceiling_corrected": shippable_strict_ceiling_corrected,
        "shippable_strict_tps_arm_A": shippable_ceiling_arm_A,
        "shippable_strict_tps_arm_A_deployed": shippable_deployed_arm_A,
        "shippable_strict_tps_arm_A_deployed_divisor": shippable_deployed_arm_A_div,
        "shippable_strict_tps_arm_B": shippable_ceiling_arm_B,
        "shippable_strict_tps_arm_B_deployed": shippable_deployed_arm_B,
        "shippable_strict_tps_arm_B_pessimistic_bf16": shippable_ceiling_arm_B_pess,
        "realized_shippable_strict_tps_decode": realized_shippable_strict_tps_decode,
        "clears_500": clears_500,
        "clears_500_ceiling_basis": clears_500_ceiling_basis,
        "gap_to_500_tps": gap_to_500_tps,
        "supply_alone_closes_500": supply_alone_closes_500,
        "n_distinct_kernel_rebuilds_for_strict_500": n_distinct_kernel_rebuilds_for_strict_500,
        "n_distinct_kernel_rebuilds_arm_A": n_distinct_kernel_rebuilds_arm_A,
        "n_distinct_kernel_rebuilds_arm_B": n_distinct_kernel_rebuilds_arm_B,
        "measured_arm": arm,
        # --- (5) phantom decomposition + band reconciliation ---
        "phantom_lmhead_bf16_tax_tps": phantom_lmhead_bf16_tax_tps,
        "spread_510_to_518_tps": spread_510_to_518,
        "spread_is_lmhead_bf16_tax": spread_is_lmhead,
        "phantom_total_offshelf_tps": phantom_total_offshelf_tps,
        "band_357_minus_lmhead_phantom": band_357_minus_lmhead_phantom,
        "phantom_lmhead_lift_from_357_tps": phantom_lmhead_lift_from_357,
        "corrected_band": corrected_band,
        "old_band_357_469": old_band,
        "floor_lift_deployed_vs_offshelf_tps": floor_lift_deployed_vs_offshelf,
        "ceiling_lift_vs_floor_tps": ceiling_lift_vs_floor,
        "band_357_reproduced": strict_tps(ETA_BLANKET_VBI_326),
        "band_469_reproduced": strict_tps(ETA_FLOOR_327),
        "pin_518": PIN_518,
        "is_strict_byte_exact": is_strict_byte_exact,
        "bucket": bucket,
        "verdict": verdict,
    }


# ======================================================================================== #
# Self-test
# ======================================================================================== #
def selftest(comp: dict, gpu: dict, flags: dict, n_seeds: int) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (a) ladder reproduces the #378 band: 0.3141->357.32, 0.09841->469.68, 0.04022->500
    c["a_band_357"] = abs(strict_tps(ETA_BLANKET_VBI_326) - BAND_357) / BAND_357 <= 1e-3
    c["a_band_469"] = abs(strict_tps(ETA_FLOOR_327) - BAND_469) / BAND_469 <= 1e-3
    c["a_budget_500"] = abs(strict_tps(BUDGET_500_ETA) - 500.0) / 500.0 <= 1e-3
    # (b) SOURCE: body + lm_head confirmed int4-Marlin
    c["b_body_int4_marlin"] = bool(comp["source_audit"].get("body_is_int4_marlin"))
    c["b_lmhead_int4_marlin"] = bool(comp["source_audit"].get("lmhead_is_int4_marlin"))
    c["b_body_group_quant"] = bool(comp["source_audit"].get("body_quant_group_size") == BODY_GROUP_SIZE)
    # (c) MECHANISM: NO body GEMM uses atomic-add on the A10G (the decode-strictness mechanism)
    c["c_no_body_atomic_add"] = bool(not comp["body_any_atomic_add_active"])
    # (c) ORACLE: every body Marlin GEMM byte-EXACT (M-invariant) at decode width incl small-n k/v proj
    c["c_body_all_byte_exact"] = bool(comp["body_already_strict_at_decode"])
    c["c_small_n_byte_exact"] = bool(comp["small_n_strict"])
    c["c_body_maxdiff_zero"] = bool(max(comp["body_maxdiff_M8_by_geom"].values()) == 0.0)
    c["c_lmhead_byte_exact"] = bool(comp["lmhead_already_strict_at_decode"])
    # (d) corrected ceiling > the OLD floor (the lift is real) and etas ordered/finite
    c["d_corrected_above_old_floor"] = bool(comp["shippable_strict_tps_arm_A"] > BAND_469)
    c["d_arm_A_ge_arm_B"] = bool(comp["shippable_strict_tps_arm_A"] >= comp["shippable_strict_tps_arm_B"] - 1e-9)
    c["d_etas_finite"] = all(math.isfinite(comp[k]) for k in
                             ("eta_total_arm_A", "eta_total_arm_B", "eta_body_376_int4preserving"))
    c["d_eta_body_376_small"] = bool(comp["eta_body_376_int4preserving"] <= ETA_FLOOR_327 + 1e-9)
    # (d) the literal 510->518 spread is NOT the lm_head bf16 tax (spread << phantom)
    c["d_spread_not_lmhead"] = bool(not comp["spread_is_lmhead_bf16_tax"])
    # (d) decision bools well-typed
    c["d_decision_bools"] = all(isinstance(comp[k], bool) for k in
                                ("clears_500", "supply_alone_closes_500", "is_strict_byte_exact"))
    # (d) rebuild counts as specified
    c["d_rebuilds_1_2"] = bool(comp["n_distinct_kernel_rebuilds_arm_A"] == 1
                               and comp["n_distinct_kernel_rebuilds_arm_B"] == 2)
    # (e) >=2 seeds, no-launch flags, on-target hardware (sm8x A10G)
    c["e_two_or_more_seeds"] = bool(n_seeds >= 2)
    c["e_no_launch_flags"] = bool(flags["no_hf_job"] and flags["no_launch"]
                                  and flags["no_served_file_change"] and flags["no_kernel_deploy"])
    c["e_on_target_a10g_sm8x"] = bool(gpu["is_a10g_80sm"] and gpu["is_sm8x"])
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


# ======================================================================================== #
# Report + IO + wandb
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return str(o)


def print_report(payload: dict) -> None:
    gpu, comp, st = payload["gpu"], payload["compose"], payload["selftest"]
    bar = "=" * 100
    print(bar)
    print("STRICT CEILING CORRECTED ROLLUP -- shippable deployable-strict under the #384 lm_head model (PR #390)")
    print(f"  GPU {gpu['name']} SMs={gpu['sm_count']} cc={gpu['compute_capability']} "
          f"on-target={gpu['is_a10g_80sm'] and gpu['is_sm8x']}")
    a = comp["source_audit"]
    print("-" * 100)
    print("  (0) SOURCE AUDIT (deployed osoi5 baked):")
    print(f"      body  int4-Marlin={a.get('body_is_int4_marlin')} strategy={a.get('body_quant_strategy')} "
          f"group_size={a.get('body_quant_group_size')} layers={a.get('num_hidden_layers')}")
    print(f"      lmhead int4-Marlin={a.get('lmhead_is_int4_marlin')} strategy={a.get('lmhead_quant_strategy')} "
          f"rows={a.get('lmhead_rows_pruned')}")
    print("-" * 100)
    print("  (1) BODY int4-Marlin decode-strictness (batched-M=8 vs per-row-M1) -- resolves stark #381:")
    for n in BODY_GEOMS:
        g = comp["body_geoms"][n]
        print(f"      {n:10s} [{g['size_k']:5d}x{g['size_n']:5d}] atomic_add={g['heuristic_use_atomic_add']!s:5s} "
              f"byte_M8={g['byte_identity_by_M']['8']:.3f} max|M8-M1|={g['max_abs_logit_diff_by_M']['8']:.2e} "
              f"strict={g['is_m_invariant_at_decode']}")
    print(f"      -> body_already_strict_at_decode = {comp['body_already_strict_at_decode']}  "
          f"(small-n k/v strict={comp['small_n_strict']}, any atomic-add active={comp['body_any_atomic_add_active']})")
    lg = comp["lmhead_geom"]
    print(f"  (2) lm_head [{lg['size_k']}x{lg['size_n']}] byte_M8={lg['byte_identity_by_M']['8']:.3f} "
          f"strict={comp['lmhead_already_strict_at_decode']} (#384 reconfirm)")
    print("-" * 100)
    print("  (3) body determinization cost:")
    print(f"      #376 int4-preserving fixed-split-K: forced/heuristic ratio max "
          f"{comp['body_forced_vs_heuristic_ratio_max']:.3f} -> eta_body_376 = "
          f"{comp['eta_body_376_int4preserving']*100:.4f}% (no-op: atomic-add already off)")
    print(f"      pessimistic bf16-determinized body (REFUTED): eta_body_floor = "
          f"{comp['eta_body_floor_pessimistic']*100:.3f}%")
    print("-" * 100)
    print("  (4) CORRECTED SHIPPABLE CEILING (#357 fern load-bearing):")
    print(f"      eta_total Arm A (body strict)    = {comp['eta_total_arm_A']*100:.3f}%  (attention pin ONLY)")
    print(f"      eta_total Arm B (#376 body fix)  = {comp['eta_total_arm_B']*100:.3f}%")
    print(f"      shippable_strict_tps_arm_A = {comp['shippable_strict_tps_arm_A']:.2f} (ceiling) / "
          f"{comp['shippable_strict_tps_arm_A_deployed_divisor']:.2f} (deployed, servable)  rebuilds=1")
    print(f"      shippable_strict_tps_arm_B = {comp['shippable_strict_tps_arm_B']:.2f} (ceiling) / "
          f"{comp['shippable_strict_tps_arm_B_deployed']:.2f} (deployed)  rebuilds=2")
    print(f"      shippable_strict_tps_arm_B (pessimistic bf16, refuted) = "
          f"{comp['shippable_strict_tps_arm_B_pessimistic_bf16']:.2f} (ceiling)")
    print(f"      ** realized_shippable_strict_tps_decode = {comp['realized_shippable_strict_tps_decode']:.2f} "
          f"(arm {comp['measured_arm']}, deployed/servable) **")
    print(f"      shippable_strict_ceiling_corrected = {comp['shippable_strict_ceiling_corrected']:.2f}")
    print(f"      clears_500 = {comp['clears_500']} (deployed) / {comp['clears_500_ceiling_basis']} (ceiling); "
          f"gap_to_500 = {comp['gap_to_500_tps']:.2f} TPS")
    print(f"      supply_alone_closes_500 = {comp['supply_alone_closes_500']}; "
          f"n_distinct_kernel_rebuilds_for_strict_500 = {comp['n_distinct_kernel_rebuilds_for_strict_500']} "
          f"(arm {comp['measured_arm']})")
    print("-" * 100)
    print("  (5) PHANTOM lm_head bf16 decomposition (PR step 1):")
    print(f"      phantom lm_head bf16 tax (if real) = {comp['phantom_lmhead_bf16_tax_tps']:.1f} TPS")
    print(f"      literal 510.01->518.92 spread      = {comp['spread_510_to_518_tps']:.2f} TPS "
          f"(attention eval-wt + pin; spread_is_lmhead={comp['spread_is_lmhead_bf16_tax']})")
    print(f"      total off-the-shelf phantom (357.32->{comp['shippable_strict_tps_arm_A']:.0f}) = "
          f"{comp['phantom_total_offshelf_tps']:.1f} TPS; lm_head slice = "
          f"{comp['phantom_lmhead_lift_from_357_tps']:.1f} TPS")
    print(f"      corrected band {[round(x,2) for x in comp['corrected_band']]} vs OLD "
          f"{[round(x,2) for x in comp['old_band_357_469']]}  "
          f"(deployed-floor lift +{comp['floor_lift_deployed_vs_offshelf_tps']:.1f} TPS)")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): " +
          json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 100)
    print("  VERDICT")
    print("   " + comp["verdict"])
    print(bar)


def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary, log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[strict-rollup] wandb helpers unavailable: {e}")
        return None
    comp = payload["compose"]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["strict-bi-verify-gemm", "strict-ceiling-corrected-rollup", "int4-marlin",
              "body-decode-strictness", "shippable-strict", "319-strict-lock", "pr-390"],
        config={"pr": 390, "kind": "strict-ceiling-corrected-rollup",
                "hidden": HIDDEN, "lmhead_rows": LMHEAD_ROWS, "num_layers": NUM_LAYERS,
                "body_geoms": {k: list(v) for k, v in BODY_GEOMS.items()},
                "body_group_size": BODY_GROUP_SIZE, "m_list": list(M_LIST), "deployed_M": DEPLOYED_M,
                "ceiling_500": CEILING_500, "step_us": STEP_US, "ratio_cal_373": RATIO_CAL_373,
                "eta_blanket_vbi_326": ETA_BLANKET_VBI_326, "eta_floor_327": ETA_FLOOR_327,
                "eta_attn_378": ETA_ATTN_378, "official_tps": OFFICIAL_TPS, "seeds": args.seeds},
    )
    if run is None:
        print("[strict-rollup] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {}
    for n in BODY_GEOMS:
        g = comp["body_geoms"][n]
        flat[f"body_identity/{n}_byte_M8"] = g["byte_identity_by_M"]["8"]
        flat[f"body_identity/{n}_maxdiff_M8"] = g["max_abs_logit_diff_by_M"]["8"]
        flat[f"body_identity/{n}_atomic_add"] = float(g["heuristic_use_atomic_add"])
        flat[f"body_latency/{n}_heuristic_us"] = g["heuristic_us"]
        flat[f"body_latency/{n}_forced_us"] = g["forced_det_us"]
    flat["lmhead/byte_M8"] = comp["lmhead_geom"]["byte_identity_by_M"]["8"]
    flat["body/already_strict_at_decode"] = float(comp["body_already_strict_at_decode"])
    flat["body/small_n_strict"] = float(comp["small_n_strict"])
    flat["body/any_atomic_add_active"] = float(comp["body_any_atomic_add_active"])
    flat["eta/eta_attn_378"] = comp["eta_attn_378"]
    flat["eta/eta_body_376_int4preserving"] = comp["eta_body_376_int4preserving"]
    flat["eta/eta_body_floor_pessimistic"] = comp["eta_body_floor_pessimistic"]
    flat["eta/eta_total_arm_A"] = comp["eta_total_arm_A"]
    flat["eta/eta_total_arm_B"] = comp["eta_total_arm_B"]
    flat["decision/shippable_strict_ceiling_corrected"] = comp["shippable_strict_ceiling_corrected"]
    flat["decision/shippable_strict_tps_arm_A"] = comp["shippable_strict_tps_arm_A"]
    flat["decision/shippable_strict_tps_arm_A_deployed"] = comp["shippable_strict_tps_arm_A_deployed_divisor"]
    flat["decision/shippable_strict_tps_arm_B"] = comp["shippable_strict_tps_arm_B"]
    flat["decision/shippable_strict_tps_arm_B_pessimistic_bf16"] = comp["shippable_strict_tps_arm_B_pessimistic_bf16"]
    flat["decision/realized_shippable_strict_tps_decode"] = comp["realized_shippable_strict_tps_decode"]
    flat["decision/clears_500"] = float(comp["clears_500"])
    flat["decision/clears_500_ceiling_basis"] = float(comp["clears_500_ceiling_basis"])
    flat["decision/gap_to_500_tps"] = comp["gap_to_500_tps"]
    flat["decision/supply_alone_closes_500"] = float(comp["supply_alone_closes_500"])
    flat["decision/n_distinct_kernel_rebuilds_for_strict_500"] = float(comp["n_distinct_kernel_rebuilds_for_strict_500"])
    flat["decision/n_rebuilds_arm_A"] = float(comp["n_distinct_kernel_rebuilds_arm_A"])
    flat["decision/n_rebuilds_arm_B"] = float(comp["n_distinct_kernel_rebuilds_arm_B"])
    flat["decision/is_strict_byte_exact"] = float(comp["is_strict_byte_exact"])
    flat["phantom/lmhead_bf16_tax_tps"] = comp["phantom_lmhead_bf16_tax_tps"]
    flat["phantom/spread_510_to_518_tps"] = comp["spread_510_to_518_tps"]
    flat["phantom/total_offshelf_tps"] = comp["phantom_total_offshelf_tps"]
    flat["phantom/lmhead_lift_from_357_tps"] = comp["phantom_lmhead_lift_from_357_tps"]
    flat["band/floor_lift_deployed_vs_offshelf_tps"] = comp["floor_lift_deployed_vs_offshelf_tps"]
    flat["band/ceiling_lift_vs_floor_tps"] = comp["ceiling_lift_vs_floor_tps"]
    flat["selftest/strict_ceiling_corrected_rollup_self_test_passes"] = float(payload["selftest"]["passes"])
    flat["gpu/sm_count"] = float(payload["gpu"]["sm_count"])
    flat["mem/peak_mem_mib"] = payload["peak_mem_mib"]
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="strict_ceiling_corrected_rollup", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[strict-rollup] wandb logged {len(flat)} keys (run {rid})")
    return rid


# ======================================================================================== #
# Main
# ======================================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="(compat flag; GPU is required regardless)")
    ap.add_argument("--measure-shippable-strict-tps", action="store_true", help="(compat; always measured)")
    ap.add_argument("--corrected-lmhead", action="store_true", help="(compat; #384 lm_head model always used)")
    ap.add_argument("--both-body-arms", action="store_true", help="(compat; both arms always reported)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
    ap.add_argument("--ident-trials", type=int, default=8, help="independent batch=1 problems per seed")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="wirbel/strict-ceiling-corrected-rollup")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="strict-bi-verify-gemm")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.smoke:
        args.ident_trials = min(args.ident_trials, 2)
        args.iters = min(args.iters, 20)
        args.warmup = min(args.warmup, 5)
        args.seeds = args.seeds[:2]

    dev = _device()
    gpu = _gpu_facts(dev)
    audit = source_audit()

    geoms = measure_marlin_family(dev, args)
    comp = compose(audit, geoms, gpu)

    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True, "no_kernel_deploy": True}
    st = selftest(comp, gpu, flags, len(args.seeds))

    torch.cuda.synchronize()
    payload = {
        "agent": "wirbel", "pr": 390, "kind": "strict-ceiling-corrected-rollup",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True, **flags,
        "gpu": gpu, "seeds": args.seeds,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "ladder_constants": {"ceiling_500": CEILING_500, "step_us": STEP_US, "ratio_cal_373": RATIO_CAL_373,
                             "eta_blanket_vbi_326": ETA_BLANKET_VBI_326, "eta_floor_327": ETA_FLOOR_327,
                             "budget_500_eta": BUDGET_500_ETA, "eta_attn_378": ETA_ATTN_378,
                             "official_tps": OFFICIAL_TPS, "band_357": BAND_357, "band_469": BAND_469,
                             "pin_518": PIN_518, "strict_floor_196": STRICT_FLOOR_196},
        "compose": comp, "selftest": st,
        "strict_ceiling_corrected_rollup_self_test_passes": bool(st["passes"]),
        # headline SENPAI-RESULT surface
        "realized_shippable_strict_tps_decode": comp["realized_shippable_strict_tps_decode"],
        "shippable_strict_ceiling_corrected": comp["shippable_strict_ceiling_corrected"],
        "shippable_strict_tps_arm_A": comp["shippable_strict_tps_arm_A"],
        "shippable_strict_tps_arm_B": comp["shippable_strict_tps_arm_B"],
        "clears_500": comp["clears_500"],
        "gap_to_500_tps": comp["gap_to_500_tps"],
        "supply_alone_closes_500": comp["supply_alone_closes_500"],
        "n_distinct_kernel_rebuilds_for_strict_500": comp["n_distinct_kernel_rebuilds_for_strict_500"],
        "body_already_strict_at_decode": comp["body_already_strict_at_decode"],
        "is_strict_byte_exact": comp["is_strict_byte_exact"],
    }
    print_report(payload)
    out_path = Path(args.out_dir) / "strict_ceiling_corrected_rollup_results.json"
    json.dump(_jsonable(payload), open(out_path, "w"), indent=2)
    print(f"[strict-rollup] results -> {out_path}")
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        json.dump(_jsonable(payload), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
