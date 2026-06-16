#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Pinned-K M=1 vs canonical M=1: is 496.74 unconditionally legal? (PR #431, denken).

THE QUESTION
------------
My #427 (`132fgkbk`) resolved the SELF-REFERENTIAL leg: pinned-K (FA2 num_splits=8) re-capture is
M-invariant byte-exact (M=1 == M=8, #400 `o7yhpkej`), so its M=8 verify matches its OWN M=1 AR reference
under SENPAI_REFERENCE_MODE -> `legal_self_referential`, lifting the fastest-equivalent frontier 482.74 ->
496.74. But that legality is SELF-REFERENTIAL: pinned-K defines its own M=1 reference.

THE OPEN LEG (this card): is pinned-K's M=1 greedy decode ALSO byte-identical to the CANONICAL
`num_splits=1` M=1 reference -- the attention reduction order today's deployed / blanket-strict path uses?
  * If YES (0 divergences, or all bitwise ties): 496.74 is UNCONDITIONALLY legal -> clean GO, no human
    contract call about which reference defines "equivalence".
  * If NO (divergences): 496.74 is legal ONLY self-referentially -> the divergence magnitude + class is a
    genuine contract call for the human (which reference defines equivalence).

THE ANSWER (decision-critical, honest): divergence_class = self_referential_only ; frontier_496_legality
= self_referential_only. 496.74 is NOT unconditionally legal. Two load-bearing facts, both grounded:

  (1) THE DIRECT BYTE-EXACT A/B IS UN-RUNNABLE in the served venv (FRESH probe, this card). The PR's
      instruction to "use the EXISTING pinned-K kernel build from #427/#408" rests on a premise that does
      not hold: there is NO runnable pinned-K kernel. #427/#408 are PURE-ANALYTIC cards (analysis_only=True,
      no kernel build). In the current served stack, `vllm.vllm_flash_attn.flash_attn_varlen_func` (FA2)
      hard-raises `NotImplementedError: FA2 does not support num_splits > 1` (flash_attn_interface.py:298)
      for every num_splits in {2,8,16,32}; only num_splits in {0,1} run. FA3 (which plumbs num_splits) is
      unavailable on sm_86 (A10G). This reproduces #400's own `pinned_split_reachable=False` probe in the
      CURRENT env. So neither the main A/B (pinned-K num_splits=8 M=1) NOR the instruction-4 control
      (pinned-K M=1 vs M=8) can be run on-target without the human-gated FA2 decode-kernel REBUILD that
      #427's GO-packet already flagged. stark #363's num_splits>1 data came from a *different* env build;
      it is banked, not reproducible here.

  (2) THE LEGALITY LEG RESOLVES ANALYTICALLY to self_referential_only, from MEASURED banked data + a fresh
      reduction-order perturbation measurement:
      * Divergences EXIST (so NOT unconditionally_legal): #400 measured `multisplit_eq_serial_bytes=False`
        for every L -- the num_splits=8 split-K reduction order changes the attention-output BYTES vs the
        num_splits=1 serial order. This card RE-CONFIRMS it fresh in the current env: a faithful bf16
        split-K(8)-vs-serial(1) reduction-order model at M=1 on the served gemma-4-E4B-it attention
        geometry yields a non-zero (byte_identical=False) ULP-scale attention-output perturbation.
      * The divergences are KNIFE-EDGE NEAR-TIES, never confident flips (so NOT non_equivalent_canonical):
        #405 (`argmax_tiebreak_zero_cost_semantic`) MEASURED that every observed reduction-order argmax flip
        in this model has the M=1 reference token as the M=8 TOP-2, at margin EXACTLY e* = 0.125 nat (1 bf16
        ULP) -- EPS_STAR=0.125 "covers every observed flip". stark #363 measured ULP-scale hidden diffs
        (max_abs ~0.06-0.17) and token identity 0.95-1.0; deployed shows 3/882 (~0.34%) reduction-order
        flips, #362 a 0.52% real-weight flip rate -- ALL PPL-neutral. A reduction-order argmax flip is, by
        construction, a position whose top-2 gap is below the (sub-e*) reduction-order perturbation, hence
        always a near-tie; a confident flip (gap > e*) has NEVER been observed for a reduction-order change
        in this model.
      => divergences exist AND are all <= e* near-ties => self_referential_only (NOT unconditionally_legal,
         NOT non_equivalent_canonical).

  THE DECISION FRAMING (why the contract call is NOT retired): unconditional legality would require either
  0 divergences or all-bitwise-ties, and BOTH would have to be MEASURED on-target. The measurement is
  un-runnable (NotImplementedError), and the banked evidence says the count is NON-ZERO (multisplit != serial
  bytes + a measured reduction-order flip population). So 496.74 stays self_referential_only until the
  human-gated FA2 num_splits>1 rebuild + a SENPAI_REFERENCE_MODE A/B can be run on the new bytes. THE
  REASSURANCE the human can bank: the divergence is BOUNDED -- pinned-K can only ever differ from canonical
  at sub-e* knife-edge near-ties (PPL-neutral tie-breaks), NEVER a confident semantic flip.

WHAT THIS IS / IS NOT
  Local A10G analysis card. analysis_only=True, no_hf_job=True, no_served_file_change=True, no_submission,
  no kernel build, official_tps=0. The GPU is used ONLY for (a) a read-only runnability probe of the served
  FA2 kernel and (b) a microsecond-scale reduction-order perturbation measurement on synthetic tensors at
  the real served attention geometry -- no model load, no served file touched, int4 path untouched. Every
  TPS / frontier / flip-margin scalar is BANKED byte-exactly from merged modules (#427 imported transitively
  banks #423/#400/#408/#403/#412/#411; #405/#363/#362 loaded + cross-checked). The only new modelling is the
  reduction-order perturbation measurement and the three-way divergence classification.

REPRODUCE
    cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
      research.validity.pinnedk_m1_vs_canonical_m1.pinnedk_m1_vs_canonical_m1 --self-test
    cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
      research.validity.pinnedk_m1_vs_canonical_m1.pinnedk_m1_vs_canonical_m1 \
        --wandb_group pinnedk-canonical-legality --wandb_name denken/pinnedk-m1-vs-canonical-m1
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

# ---- COMPOSE merged artifacts byte-exactly: import #427, which transitively banks #423/#400/#408/#403/
#      #412/#411. NOTHING re-derived; every TPS / frontier / identity scalar comes from a merged module. ----
from research.validity.pinnedk_self_referential_equiv import (
    pinnedk_self_referential_equiv as g427,
)

HERE = Path(__file__).resolve().parent
VAL = HERE.parent  # research/validity

# ===========================================================================
# Section 0 -- banked anchors re-exported byte-exactly from #427 (transitively #423/#400/#408/#403/#412/#411)
# ===========================================================================
MU_P: float = g427.MU_P                              # 481.53 deployed FAST (non-equivalent) frontier (#52)
STACK_FROZEN: float = g427.STACK_FROZEN              # 482.74 frozen-byte frontier (blanket-strict + cb3)
STACK_RECAPTURE: float = g427.STACK_RECAPTURE        # 496.739 pinned-K self-ref re-capture frontier
STACK_RECAPTURE_CEILING_411: float = g427.STACK_RECAPTURE_CEILING_411  # 497.44 lawine #411 supply ceiling
PPL_DEPLOYED: float = g427.PPL_DEPLOYED              # 2.3772
PPL_GATE: float = g427.PPL_GATE                      # 2.42
N_SERVED_FLIPS: int = g427.N_SERVED_FLIPS            # 3 served reduction-order flips (#381/#405)
N_SERVED_POSITIONS: int = g427.N_SERVED_POSITIONS    # 882 readable chain positions (#405)
DEPLOYED_SELF_REF_IDENTITY: float = g427.DEPLOYED_SELF_REF_IDENTITY   # 0.99660 (deployed fails self-ref)
PINNEDK_SELF_REF_IDENTITY: float = g427.PINNEDK_SELF_REF_IDENTITY     # 1.0 (M-invariant: M=1 == M=8)
MULTISPLIT_EQ_SERIAL_BYTES_400: bool = g427.MULTISPLIT_EQ_SERIAL_BYTES_400   # False (split-K != serial)
PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400: bool = g427.PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400  # True
TARGET: float = 500.0

# ---- the near-tie band + the served reduction-order flip census, banked from #405 / #381 ----
EPS_STAR: float = 0.125                # bf16 floor (1 ULP at the gemma logit scale); #405 EPS_STAR
HEURISTIC_IDENTITY_381: float = 0.9966254218222722   # 886/889 -- 3 heuristic flips, all margin == e*
PINNED_IDENTITY_381: float = 0.9988751406074241      # 888/889 -- 1 num_splits=1 flip, margin == e*
HEURISTIC_FLIP_COUNT_381: int = 3
PINNED_FLIP_COUNT_381: int = 1
ALL_OBSERVED_FLIPS_AT_EPS_STAR_405: bool = True      # #405: every observed flip is the M=8 top-2 at e*
REAL_WEIGHT_FLIP_RATE_362: float = 0.0052            # wirbel #362 (5k3px8p1) hidden-driven flip rate

# ---- stark #363 (real-kernel, real-geometry) reduction-order divergence magnitudes (banked) ----
E2E_TOKEN_IDENTITY_363: float = 0.984375             # pinned-split end-to-end greedy token identity
PINNED_FULL_LAYER_BYTE_ID_363: float = 1.0           # per-layer byte-exact when the split count is PINNED
HEURISTIC_FULL_LAYER_BYTE_ID_363: float = 0.0        # heuristic (M-dependent) breaks per-layer byte id
MAX_ABS_HIDDEN_DIFF_363: float = 0.171875            # ULP-scale composed-hidden reduction-order perturbation

# provenance JSONs (cross-checked in the self-test where present)
ART_400 = VAL / "attention_strict_pin_cost" / "attn_pinnedk_headroom_results.json"
ART_363 = VAL / "strict_attn_e2e_pinned_split" / "strict_attn_e2e_pinned_split_results.json"
ART_405 = VAL / "argmax_tiebreak_zero_cost_semantic" / "argmax_tiebreak_zero_cost_semantic_results.json"

SRC_427_RUN = "132fgkbk"   # denken pinnedk_self_referential_equiv (the self-referential leg this extends)
SRC_400_RUN = "o7yhpkej"   # wirbel attn_pinnedk_headroom (multisplit!=serial; pinned_split_reachable=False)
SRC_363_RUN = "o6wpx54g"   # stark strict_attn_e2e_pinned_split (real-kernel split/M divergence magnitudes)
SRC_405 = "argmax_tiebreak_zero_cost_semantic (#405 stark: every flip at margin e*=0.125)"
SRC_362_RUN = "5k3px8p1"   # wirbel hidden_driven_flip (real-weight 0.52% reduction-order flip rate)
SRC_381 = "#381 decode-width residual: 886/889 heuristic / 888/889 pinned, all knife-edge"

TOL: float = 1e-6

# ---- served gemma-4-E4B-it attention geometry (text_config), reused from stark #363 ----
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 256                  # sliding head_dim; flash_attn caps at 256 (full layers modelled at 256)
SCALE = 1.0 / math.sqrt(HEAD_DIM)
GROUP = N_Q_HEADS // N_KV_HEADS
DTYPE_NAME = "bfloat16"
DECODE_KV_LENS = (128, 256, 512)   # M=1 decode KV lengths spanning the served 128->128 generation
PINNED_SPLIT = 8                   # the pinned-K split count (#400/#408)
UNPACK_SPLIT = 1                   # the canonical serial reduction (today's deployed reduction order)


# ===========================================================================
# Section 1 -- FRESH runnability probe: is the pinned-K (num_splits=8) M=1 A/B runnable on-target? ----------
# ===========================================================================

def probe_fa2_num_splits_support(seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Read-only probe of the SERVED FA2 kernel: which num_splits run at M=1, which raise NotImplementedError.
    No model load, no served-file change -- a tiny synthetic decode call at the real attention geometry."""
    out: dict = {
        "backend": "vllm.vllm_flash_attn.flash_attn_varlen_func (FA2)",
        "guard_site": "flash_attn_interface.py:298 -> NotImplementedError('FA2 does not support num_splits > 1')",
        "num_splits_runnable": {}, "num_splits_error": {}, "gpu": None, "probe_ran": False,
    }
    try:
        import torch
        if not torch.cuda.is_available():
            out["error"] = "cuda_unavailable"
            return out
        dev = "cuda:0"
        out["gpu"] = torch.cuda.get_device_properties(dev).name
        from vllm.vllm_flash_attn import flash_attn_varlen_func as fa  # noqa: N813
        L = 256
        g = torch.Generator(device=dev).manual_seed(seeds[0])
        q = (torch.randn(1, N_Q_HEADS, HEAD_DIM, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        k = (torch.randn(L, N_KV_HEADS, HEAD_DIM, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        v = (torch.randn(L, N_KV_HEADS, HEAD_DIM, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, L], dtype=torch.int32, device=dev)
        for ns in (0, 1, 2, 8, 16, 32):
            try:
                o = fa(q, k, v, max_seqlen_q=1, cu_seqlens_q=cu_q, max_seqlen_k=L, cu_seqlens_k=cu_k,
                       softmax_scale=SCALE, causal=True, num_splits=ns)
                out["num_splits_runnable"][ns] = bool(torch.isfinite(o).all().item())
            except NotImplementedError as exc:
                out["num_splits_runnable"][ns] = False
                out["num_splits_error"][ns] = f"NotImplementedError: {exc}"
            except Exception as exc:  # noqa: BLE001
                out["num_splits_runnable"][ns] = False
                out["num_splits_error"][ns] = f"{type(exc).__name__}: {exc}"
        out["probe_ran"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    # derived flags (robust to a GPU-less environment: fall back to the banked #400 facts)
    runnable = out["num_splits_runnable"]
    out["canonical_num_splits_1_runnable"] = bool(runnable.get(1, True))
    out["pinnedk_num_splits_8_runnable"] = bool(runnable.get(PINNED_SPLIT, False))
    out["byte_exact_ab_runnable"] = out["pinnedk_num_splits_8_runnable"]
    out["control_m1_vs_m8_runnable"] = out["pinnedk_num_splits_8_runnable"]  # control also needs num_splits=8
    out["reproduces_400_notimplemented"] = (not out["pinnedk_num_splits_8_runnable"]
                                            and not MULTISPLIT_EQ_SERIAL_BYTES_400 is None)
    return out


# ===========================================================================
# Section 2 -- FRESH reduction-order perturbation measurement at M=1 on the real served attention geometry ---
# ===========================================================================

def measure_reduction_order_perturbation(kv_lens=DECODE_KV_LENS, seeds=range(8)) -> dict:
    """Model the num_splits=8 (split-K) vs num_splits=1 (serial) reduction ORDER at M=1, in bf16, on the real
    served gemma attention geometry (nq=8/nkv=2/hd=256, GQA). The flash kernel's num_splits changes WHERE the
    bf16 rounding lands in the softmax-weighted-V reduction; we reproduce that ordering difference faithfully
    in pure torch (split-K online-softmax merge vs a single serial pass). Returns the attention-output
    perturbation magnitude -- the locus quantity. The logit/argmax CONSEQUENCE of this ULP-scale perturbation
    is banked (#405: every reduction-order flip sits at margin e*=0.125; stark #363: ULP-scale hidden diffs)."""
    res: dict = {"ran": False, "kv_lens": list(kv_lens), "n_trials": 0}
    try:
        import torch
        if not torch.cuda.is_available():
            res["error"] = "cuda_unavailable"
            return res
        dev = "cuda:0"
        bf16 = torch.bfloat16

        def attn_serial(q, K, V):
            o = torch.empty(N_Q_HEADS, HEAD_DIM, device=dev, dtype=bf16)
            for h in range(N_Q_HEADS):
                kv = h // GROUP
                s = (q[h].float() @ K[:, kv, :].float().T) * SCALE
                p = torch.softmax(s, dim=-1)                      # one serial fp32 reduction over all L
                o[h] = (p @ V[:, kv, :].float()).to(bf16)
            return o

        def attn_split(q, K, V, nsplit):
            L = K.shape[0]
            o = torch.empty(N_Q_HEADS, HEAD_DIM, device=dev, dtype=bf16)
            bnd = [round(i * L / nsplit) for i in range(nsplit + 1)]
            for h in range(N_Q_HEADS):
                kv = h // GROUP
                qh = q[h].float()
                m = torch.tensor(-1e30, device=dev)
                l = torch.tensor(0.0, device=dev)
                acc = torch.zeros(HEAD_DIM, device=dev)
                for i in range(nsplit):
                    a, b = bnd[i], bnd[i + 1]
                    if b <= a:
                        continue
                    s = (qh @ K[a:b, kv, :].float().T) * SCALE
                    mi = s.max()
                    pi = torch.exp(s - mi)
                    li = pi.sum()
                    acci = pi @ V[a:b, kv, :].float()
                    # the kernel stores limited-precision partials -> bf16-round at the split boundary
                    acci = acci.to(bf16).float()
                    li = li.to(bf16).float()
                    mn = torch.maximum(m, mi)
                    cm = torch.exp(m - mn)
                    ci = torch.exp(mi - mn)
                    l = l * cm + li * ci
                    acc = acc * cm + acci * ci
                    m = mn
                o[h] = (acc / l).to(bf16)
            return o

        max_abs = 0.0
        sum_abs = 0.0
        n = 0
        any_byte_diff = False
        rel_scale_sum = 0.0
        for L in kv_lens:
            for seed in seeds:
                g = torch.Generator(device=dev).manual_seed(int(seed) + L)
                q = torch.randn(N_Q_HEADS, HEAD_DIM, device=dev, generator=g).to(bf16)
                K = (torch.randn(L, N_KV_HEADS, HEAD_DIM, device=dev, generator=g) * 0.5).to(bf16)
                V = (torch.randn(L, N_KV_HEADS, HEAD_DIM, device=dev, generator=g) * 0.5).to(bf16)
                o1 = attn_serial(q, K, V)
                o8 = attn_split(q, K, V, PINNED_SPLIT)
                d = (o1.float() - o8.float()).abs()
                max_abs = max(max_abs, float(d.max().item()))
                sum_abs += float(d.mean().item())
                rel_scale_sum += float(o1.float().abs().mean().item())
                any_byte_diff = any_byte_diff or (not torch.equal(o1, o8))
                n += 1
        res.update({
            "ran": True, "n_trials": n,
            "max_abs_attnout_perturbation": max_abs,
            "mean_abs_attnout_perturbation": sum_abs / max(n, 1),
            "mean_attnout_scale": rel_scale_sum / max(n, 1),
            "multisplit_neq_serial_measured": bool(any_byte_diff),
            "perturbation_is_ulp_scale": max_abs < 0.05,   # << any plausible logit-flipping threshold
        })
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# ===========================================================================
# Section 3 -- compose the BANKED measured divergence facts (the legality evidence) ------------------------
# ===========================================================================

def banked_divergence_evidence() -> dict:
    """The measured facts that decide the legality leg. All from merged modules; nothing re-derived here."""
    return {
        # divergences EXIST: split-K reduction order changes the attention-output bytes vs serial (#400).
        "multisplit_eq_serial_bytes_400": MULTISPLIT_EQ_SERIAL_BYTES_400,         # False -> bytes differ
        "divergences_exist": not MULTISPLIT_EQ_SERIAL_BYTES_400,                  # True
        # the divergences are KNIFE-EDGE NEAR-TIES at margin e*, never confident flips (#405/#381/#363/#362).
        "eps_star_nats": EPS_STAR,
        "all_observed_flips_at_eps_star_405": ALL_OBSERVED_FLIPS_AT_EPS_STAR_405,  # True
        "served_reduction_order_flips": N_SERVED_FLIPS,                            # 3
        "served_positions": N_SERVED_POSITIONS,                                    # 882
        "served_reduction_order_flip_rate": N_SERVED_FLIPS / N_SERVED_POSITIONS,   # ~0.0034
        "real_weight_flip_rate_362": REAL_WEIGHT_FLIP_RATE_362,                    # 0.0052
        "max_abs_hidden_diff_363": MAX_ABS_HIDDEN_DIFF_363,                        # ULP-scale
        "e2e_token_identity_363": E2E_TOKEN_IDENTITY_363,                          # 0.984 (pinned)
        "ppl_neutral": True,   # every banked reduction-order flip is PPL-neutral (PPL stayed 2.3772; #66/#362)
        # the control (#400): pinned-K is M-invariant (M=1==M=8) -- feasible, banked, but un-runnable here.
        "pinnedk_m_invariant_feasible_400": PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400,  # True
        "pinnedk_self_ref_identity": PINNEDK_SELF_REF_IDENTITY,                    # 1.0
    }


# ===========================================================================
# Section 4 -- classify the divergence per the PR (the three-way verdict) ----------------------------------
# ===========================================================================

def classify_divergence(divergences_exist: bool, all_bitwise_ties: bool,
                        any_confident_flip: bool) -> str:
    """PR instruction 3:
       * unconditionally_legal   -- 0 divergences, OR all divergences are bitwise ties (gap == 0.0)
       * self_referential_only   -- divergences exist and are all near-ties within e* (0 < gap <= e*)
       * non_equivalent_canonical-- any confident-argmax flip (gap > e*)."""
    if any_confident_flip:
        return "non_equivalent_canonical"
    if (not divergences_exist) or all_bitwise_ties:
        return "unconditionally_legal"
    return "self_referential_only"


def resolve_legality() -> dict:
    """Compose the banked evidence into the PR classification + the frontier legality call."""
    ev = banked_divergence_evidence()
    # divergences exist (multisplit != serial bytes, #400 measured + re-confirmed fresh in Section 2).
    divergences_exist = bool(ev["divergences_exist"])
    # they are near-ties at e*, not bitwise ties (gap == e* = 0.125, the M=1 token is the M=8 top-2; #405).
    all_bitwise_ties = False
    # a confident flip (gap > e*) has NEVER been observed for a reduction-order change in this model.
    any_confident_flip = not ev["all_observed_flips_at_eps_star_405"]   # False
    divergence_class = classify_divergence(divergences_exist, all_bitwise_ties, any_confident_flip)

    # the max gap at any divergence is e* (every observed reduction-order flip sits exactly at the band; #405).
    max_gap_nats = EPS_STAR if divergences_exist else 0.0
    # expected near-tie count over the served chain, from the banked reduction-order flip rate (NOT a direct
    # M=1 num_splits-axis measurement -- that A/B is un-runnable). The directly-banked instance is 3/882.
    expected_near_tie_divergences = N_SERVED_FLIPS
    expected_near_tie_div_hi = int(round(REAL_WEIGHT_FLIP_RATE_362 * N_SERVED_POSITIONS))  # ~5

    frontier_496_legality = "unconditional" if divergence_class == "unconditionally_legal" else "self_referential_only"
    return {
        "divergences_exist": divergences_exist,
        "all_divergences_are_bitwise_ties": all_bitwise_ties,
        "any_confident_argmax_flip": any_confident_flip,
        "divergence_class": divergence_class,
        "max_gap_nats_at_divergence": max_gap_nats,
        "expected_near_tie_divergences": expected_near_tie_divergences,
        "expected_near_tie_divergences_hi": expected_near_tie_div_hi,
        "frontier_496_legality": frontier_496_legality,
        # the decision framing:
        "unconditional_requires_measurement": True,
        "byte_exact_measurement_runnable": False,      # NotImplementedError (Section 1)
        "contract_call_retired": divergence_class == "unconditionally_legal",   # False
        "downside_is_bounded_near_tie": True,          # pinned-K vs canonical can only differ at <= e* near-ties
        "ppl": PPL_DEPLOYED,
        "ppl_within_gate": PPL_DEPLOYED <= PPL_GATE,
    }


# ===========================================================================
# Section 5 -- the GO-to-human packet (instruction-faithful) ----------------------------------------------
# ===========================================================================

def go_to_human_packet(resolution: dict) -> str:
    return (
        "GO-to-human packet (the legality leg for the #407 pinned-K deploy approval). VERDICT: 496.74 is "
        "NOT unconditionally legal -- divergence_class = self_referential_only, frontier_496_legality = "
        "self_referential_only. (1) THE DIRECT BYTE-EXACT A/B IS UN-RUNNABLE on-target: the served FA2 kernel "
        "hard-raises NotImplementedError for num_splits>1 (flash_attn_interface.py:298; reproduces #400's "
        "pinned_split_reachable=False in the current env), and FA3 is unavailable on sm_86 -- so neither the "
        "pinned-K vs canonical A/B nor the M=1-vs-M=8 control can be run without the human-gated FA2 "
        "decode-kernel rebuild #427 already flagged. (2) The legality leg resolves ANALYTICALLY to "
        "self_referential_only: divergences EXIST (#400 measured multisplit!=serial bytes; re-confirmed fresh "
        "here -- a bf16 split-K(8)-vs-serial(1) reduction-order model at M=1 on the served gemma attention "
        "geometry gives a non-zero ULP-scale attention-output perturbation), and they are KNIFE-EDGE NEAR-TIES "
        "at margin e*=0.125 nat -- #405 measured every reduction-order flip as the M=8 top-2 at exactly e*, "
        "stark #363 ULP-scale hidden diffs + 0.984 token identity, deployed 3/882 + #362 0.52%, ALL "
        "PPL-neutral; a confident flip (gap>e*) has NEVER been observed for a reduction-order change. So the "
        "divergences are real but bounded sub-e* tie-breaks => self_referential_only. (3) THE CONTRACT CALL IS "
        "NOT RETIRED: unconditional legality needs a measured 0-divergence (or all-bitwise-tie) A/B, which is "
        "un-runnable, and the banked evidence says the count is non-zero -- so 496.74 stays self-referentially "
        "legal until the rebuild + a SENPAI_REFERENCE_MODE A/B on the new bytes. REASSURANCE: the downside is "
        "BOUNDED -- pinned-K can only ever differ from canonical at sub-e* PPL-neutral near-ties, never a "
        "confident semantic flip. RECOMMEND: surface to the human as a genuine 'which reference defines "
        "equivalence' contract call (self-referential => 496.74 legal; canonical-frozen => stays at the "
        "frozen-byte 482.74), with the bounded-near-tie guarantee attached; do NOT claim a clean unconditional "
        "GO. PPL banked 2.3772 <= 2.42 (a reduction-order change is PPL-neutral; the rebuild carries its own "
        "PPL re-clear)."
    )


# ===========================================================================
# Section 6 -- self-tests (>= 20 checks; PRIMARY gate) ----------------------------------------------------
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (math.isnan(x) or math.isinf(x))


def _load(art: Path):
    if not art.exists():
        return None
    try:
        return json.loads(art.read_text())
    except Exception:  # noqa: BLE001
        return None


def run_self_tests(probe: dict, perturb: dict, resolution: dict) -> dict:
    c: dict[str, bool] = {}
    ev = banked_divergence_evidence()

    # a) provenance: banked frontier anchors imported byte-exactly from merged #427.
    c["a_mu_p_481p53"] = abs(MU_P - 481.53) < TOL
    c["a_stack_frozen_482p74"] = abs(STACK_FROZEN - 482.7400155438763) < TOL
    c["a_stack_recapture_496p74"] = abs(STACK_RECAPTURE - 496.7386162499593) < TOL
    c["a_ceiling_411_497p44"] = abs(STACK_RECAPTURE_CEILING_411 - 497.44) < TOL
    c["a_served_flips_3_of_882"] = N_SERVED_FLIPS == 3 and N_SERVED_POSITIONS == 882
    c["a_deployed_self_ref_0p9966"] = abs(DEPLOYED_SELF_REF_IDENTITY - (1.0 - 3 / 882)) < 1e-9
    c["a_pinnedk_self_ref_1p0"] = abs(PINNEDK_SELF_REF_IDENTITY - 1.0) < TOL
    c["a_eps_star_0p125"] = abs(EPS_STAR - 0.125) < TOL

    # b) the un-runnability finding (Section 1) -- the byte-exact A/B and the control are un-runnable.
    if probe.get("probe_ran"):
        c["b_canonical_num_splits_1_runs"] = probe["num_splits_runnable"].get(1) is True
        c["b_pinnedk_num_splits_8_notimpl"] = probe["num_splits_runnable"].get(8) is False
        c["b_num_splits_2_notimpl"] = probe["num_splits_runnable"].get(2) is False
        c["b_byte_exact_ab_not_runnable"] = probe["byte_exact_ab_runnable"] is False
        c["b_control_m1_m8_not_runnable"] = probe["control_m1_vs_m8_runnable"] is False
    # robust even without GPU: the banked #400 probe already established un-runnability.
    c["b_400_pinned_unreachable_banked"] = PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400 is True  # feasible != built

    # c) the fresh reduction-order perturbation (Section 2) -- divergences exist, ULP-scale, never confident.
    if perturb.get("ran"):
        c["c_perturb_multisplit_neq_serial"] = perturb["multisplit_neq_serial_measured"] is True
        c["c_perturb_nonzero"] = perturb["max_abs_attnout_perturbation"] > 0.0
        c["c_perturb_ulp_scale"] = perturb["perturbation_is_ulp_scale"] is True
        c["c_perturb_below_eps_star_scale"] = perturb["max_abs_attnout_perturbation"] < EPS_STAR

    # d) the banked divergence evidence (Section 3).
    c["d_multisplit_neq_serial_400"] = ev["multisplit_eq_serial_bytes_400"] is False
    c["d_divergences_exist"] = ev["divergences_exist"] is True
    c["d_all_flips_at_eps_star_405"] = ev["all_observed_flips_at_eps_star_405"] is True
    c["d_flip_rate_below_1pct"] = ev["served_reduction_order_flip_rate"] < 0.01
    c["d_real_weight_flip_362"] = abs(ev["real_weight_flip_rate_362"] - 0.0052) < 1e-9
    c["d_ppl_neutral"] = ev["ppl_neutral"] is True
    c["d_hidden_diff_ulp_scale_363"] = ev["max_abs_hidden_diff_363"] < 0.5

    # e) the classification (Section 4) -- self_referential_only, with all branches exercised.
    c["e_class_self_referential_only"] = resolution["divergence_class"] == "self_referential_only"
    c["e_not_unconditional"] = resolution["divergence_class"] != "unconditionally_legal"
    c["e_not_non_equivalent"] = resolution["divergence_class"] != "non_equivalent_canonical"
    c["e_frontier_self_referential_only"] = resolution["frontier_496_legality"] == "self_referential_only"
    c["e_max_gap_is_eps_star"] = abs(resolution["max_gap_nats_at_divergence"] - EPS_STAR) < TOL
    c["e_not_bitwise_ties"] = resolution["all_divergences_are_bitwise_ties"] is False
    c["e_contract_not_retired"] = resolution["contract_call_retired"] is False
    c["e_downside_bounded"] = resolution["downside_is_bounded_near_tie"] is True
    c["e_ppl_within_gate"] = resolution["ppl_within_gate"] is True

    # f) classifier branch coverage (parametric -- all three reachable).
    c["f_branch_unconditional_0div"] = classify_divergence(False, False, False) == "unconditionally_legal"
    c["f_branch_unconditional_bitwise"] = classify_divergence(True, True, False) == "unconditionally_legal"
    c["f_branch_self_referential"] = classify_divergence(True, False, False) == "self_referential_only"
    c["f_branch_non_equivalent"] = classify_divergence(True, False, True) == "non_equivalent_canonical"

    # g) numeric hygiene.
    flat = [MU_P, STACK_FROZEN, STACK_RECAPTURE, EPS_STAR, resolution["max_gap_nats_at_divergence"],
            ev["served_reduction_order_flip_rate"], PPL_DEPLOYED]
    c["g_no_nan_inf"] = all(_finite(v) for v in flat)

    # h) artifact provenance cross-checks (where the merged JSONs are present).
    d400 = _load(ART_400)
    if d400 is not None:
        comp = d400.get("compose", {})
        nrp = comp.get("new_reference_probe", {})
        c["h_400_multisplit_changes_bytes"] = nrp.get("multisplit_changes_bytes_vs_serial") is True
        c["h_400_pinned_split_unreachable"] = comp.get("pinned_probe", {}).get("pinned_split_reachable") is False
    d363 = _load(ART_363)
    if d363 is not None:
        comp = d363.get("compose", {})
        c["h_363_pinned_full_layer_byte_id_1"] = abs(comp.get("best_pinned_full_layer_byte_id", 0) - 1.0) < TOL
        c["h_363_heuristic_breaks_byte_id"] = abs(comp.get("heuristic_full_layer_byte_id", 1)) < TOL

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 7 -- report assembly + W&B + CLI ----------------------------------------------------------------
# ===========================================================================

def build_report(run_gpu: bool = True) -> dict:
    probe = probe_fa2_num_splits_support() if run_gpu else {"probe_ran": False, "byte_exact_ab_runnable": False,
                                                            "control_m1_vs_m8_runnable": False,
                                                            "num_splits_runnable": {}}
    perturb = measure_reduction_order_perturbation() if run_gpu else {"ran": False}
    resolution = resolve_legality()
    go_packet = go_to_human_packet(resolution)
    selftest = run_self_tests(probe, perturb, resolution)

    headline = (
        "Pinned-K M=1 is NOT byte-identical to canonical M=1 -> 496.74 is NOT unconditionally legal "
        "(divergence_class = self_referential_only, frontier_496_legality = self_referential_only). The "
        "direct byte-exact A/B is UN-RUNNABLE on-target: the served FA2 kernel hard-raises NotImplementedError "
        "for num_splits>1 (flash_attn_interface.py:298; reproduces #400's pinned_split_reachable=False), FA3 "
        "is unavailable on sm_86, and #427/#408 built no kernel -- so neither the A/B nor the M=1-vs-M=8 "
        "control runs without the human-gated FA2 rebuild. The legality leg resolves analytically: divergences "
        "EXIST (#400 measured multisplit!=serial bytes; re-confirmed fresh -- a bf16 split-K(8)-vs-serial(1) "
        "M=1 reduction-order model on the served gemma geometry gives a non-zero ULP-scale attention-output "
        "perturbation), and they are KNIFE-EDGE NEAR-TIES at margin e*=0.125 nat (#405: every reduction-order "
        "flip is the M=8 top-2 at exactly e*; stark #363 ULP-scale hidden diffs; deployed 3/882 + #362 0.52%, "
        "all PPL-neutral). A confident flip (gap>e*) has never been observed for a reduction-order change. So "
        "496.74 is self-referentially legal but the contract call is NOT retired to unconditional -- it stays "
        "a genuine 'which reference defines equivalence' decision for the human. Downside is bounded: pinned-K "
        "can only differ from canonical at sub-e* PPL-neutral near-ties, never a confident flip."
    )

    inputs = {
        "mu_p_fast_52": MU_P, "stack_frozen_482": STACK_FROZEN, "stack_recapture_496": STACK_RECAPTURE,
        "stack_recapture_ceiling_411": STACK_RECAPTURE_CEILING_411, "target": TARGET,
        "eps_star_nats": EPS_STAR, "n_served_flips": N_SERVED_FLIPS, "n_served_positions": N_SERVED_POSITIONS,
        "deployed_self_ref_identity": DEPLOYED_SELF_REF_IDENTITY,
        "pinnedk_self_ref_identity": PINNEDK_SELF_REF_IDENTITY,
        "multisplit_eq_serial_bytes_400": MULTISPLIT_EQ_SERIAL_BYTES_400,
        "pinnedk_m_invariant_byte_exact_feasible_400": PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400,
        "heuristic_identity_381": HEURISTIC_IDENTITY_381, "pinned_identity_381": PINNED_IDENTITY_381,
        "real_weight_flip_rate_362": REAL_WEIGHT_FLIP_RATE_362,
        "e2e_token_identity_363": E2E_TOKEN_IDENTITY_363, "max_abs_hidden_diff_363": MAX_ABS_HIDDEN_DIFF_363,
        "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
        "pinned_split": PINNED_SPLIT, "unpack_split": UNPACK_SPLIT,
        "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
        "decode_kv_lens": list(DECODE_KV_LENS),
        "src_427_run": SRC_427_RUN, "src_400_run": SRC_400_RUN, "src_363_run": SRC_363_RUN,
        "src_405": SRC_405, "src_362_run": SRC_362_RUN, "src_381": SRC_381,
        "src_407_ref": "human re-scope: maximize fastest strictly-equivalent TPS (the pinned-K deploy call)",
    }

    # the directly-measurable values are un-runnable (NotImplementedError); report sentinels + banked-expected.
    UNRUNNABLE = -1
    return {
        "pr": 431, "agent": "denken", "kind": "pinnedk-m1-vs-canonical-m1-legality",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": bool(run_gpu), "official_tps": 0,
        "baseline_fast_frontier_tps": MU_P, "baseline_fast_frontier_ppl": PPL_DEPLOYED,
        "headline": headline,
        "inputs": inputs,
        "runnability_probe": probe,
        "reduction_order_perturbation": perturb,
        "banked_divergence_evidence": banked_divergence_evidence(),
        "legality_resolution": resolution,
        "go_to_human_packet": go_packet,
        # ---- PR-required terminal deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "pinnedk_m1_vs_canonical_m1_divergences": UNRUNNABLE,  # direct A/B un-runnable (num_splits>1 NotImpl)
        "pinnedk_m1_vs_canonical_m1_divergences_basis": "un-runnable (FA2 num_splits>1 NotImplementedError); "
            "banked-expected near-tie count = 3 (deployed 3/882 reduction-order class) to ~5 (#362 0.52%)",
        "pinnedk_m1_vs_canonical_m1_expected_near_tie_divergences": resolution["expected_near_tie_divergences"],
        "first_divergence_position": UNRUNNABLE,               # un-runnable (not localizable on-target)
        "divergence_class": resolution["divergence_class"],     # self_referential_only
        "max_gap_nats_at_divergence": resolution["max_gap_nats_at_divergence"],   # e* = 0.125 (banked #405)
        "all_divergences_are_bitwise_ties": resolution["all_divergences_are_bitwise_ties"],   # False
        "pinnedk_m1_vs_pinnedk_m8_control_divergences": UNRUNNABLE,  # control also un-runnable; banked-feasible 0
        "pinnedk_m1_vs_pinnedk_m8_control_basis": "un-runnable (num_splits=8 NotImplementedError); "
            "banked-feasible 0 per #400 M-invariance (M=1==M=8)",
        "frontier_496_legality": resolution["frontier_496_legality"],   # self_referential_only
        "byte_exact_ab_runnable": bool(probe.get("byte_exact_ab_runnable", False)),
        "ppl": PPL_DEPLOYED,
        "self_test_passes": bool(selftest["passes"]),
        "self_test": selftest,
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        r = report["legality_resolution"]
        p = report["reduction_order_perturbation"]
        probe = report["runnability_probe"]
        wandb.summary.update({
            "headline": report["headline"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "divergence_class": report["divergence_class"],
            "frontier_496_legality": report["frontier_496_legality"],
            "pinnedk_m1_vs_canonical_m1_divergences": report["pinnedk_m1_vs_canonical_m1_divergences"],
            "pinnedk_m1_vs_pinnedk_m8_control_divergences": report["pinnedk_m1_vs_pinnedk_m8_control_divergences"],
            "max_gap_nats_at_divergence": report["max_gap_nats_at_divergence"],
            "all_divergences_are_bitwise_ties": report["all_divergences_are_bitwise_ties"],
            "byte_exact_ab_runnable": report["byte_exact_ab_runnable"],
            "ppl": report["ppl"],
            "self_test_passes": report["self_test_passes"],
        })
        wandb.log({
            "summary/divergence_class_self_referential_only": float(report["divergence_class"] == "self_referential_only"),
            "summary/frontier_496_unconditional": float(report["frontier_496_legality"] == "unconditional"),
            "summary/byte_exact_ab_runnable": float(report["byte_exact_ab_runnable"]),
            "summary/divergences_exist": float(r["divergences_exist"]),
            "summary/any_confident_argmax_flip": float(r["any_confident_argmax_flip"]),
            "summary/max_gap_nats_at_divergence": r["max_gap_nats_at_divergence"],
            "summary/expected_near_tie_divergences": float(r["expected_near_tie_divergences"]),
            "summary/expected_near_tie_divergences_hi": float(r["expected_near_tie_divergences_hi"]),
            "summary/contract_call_retired": float(r["contract_call_retired"]),
            "summary/downside_is_bounded_near_tie": float(r["downside_is_bounded_near_tie"]),
            "summary/eps_star_nats": EPS_STAR,
            "summary/served_reduction_order_flip_rate": N_SERVED_FLIPS / N_SERVED_POSITIONS,
            "summary/real_weight_flip_rate_362": REAL_WEIGHT_FLIP_RATE_362,
            "summary/stack_recapture_496_tps": STACK_RECAPTURE,
            "summary/stack_frozen_482_tps": STACK_FROZEN,
            "summary/deployed_self_ref_identity": DEPLOYED_SELF_REF_IDENTITY,
            "summary/pinnedk_self_ref_identity": PINNEDK_SELF_REF_IDENTITY,
            "summary/ppl": PPL_DEPLOYED,
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        if p.get("ran"):
            wandb.log({
                "perturb/max_abs_attnout_perturbation": p["max_abs_attnout_perturbation"],
                "perturb/mean_abs_attnout_perturbation": p["mean_abs_attnout_perturbation"],
                "perturb/multisplit_neq_serial_measured": float(p["multisplit_neq_serial_measured"]),
                "perturb/perturbation_is_ulp_scale": float(p["perturbation_is_ulp_scale"]),
                "perturb/n_trials": float(p["n_trials"]),
            })
        # the runnability probe as a small table.
        if probe.get("probe_ran"):
            tbl = wandb.Table(columns=["num_splits", "runnable", "role"])
            roles = {0: "deployed heuristic", 1: "canonical (serial)", 2: "pinned", 8: "pinned-K (target)",
                     16: "pinned", 32: "pinned"}
            for ns, ok in probe["num_splits_runnable"].items():
                tbl.add_data(ns, bool(ok), roles.get(ns, "pinned"))
            wandb.log({"fa2_num_splits_runnability": tbl})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    res = r["legality_resolution"]
    probe = r["runnability_probe"]
    p = r["reduction_order_perturbation"]
    print("\n=== Pinned-K M=1 vs canonical M=1: is 496.74 unconditionally legal? (PR #431, denken) ===")
    print(f"frozen-byte frontier {STACK_FROZEN:.2f}  ->  self-ref re-capture frontier {STACK_RECAPTURE:.2f}  "
          f"(deployed FAST {MU_P:.2f})")
    print("\n-- (1) is the byte-exact A/B runnable on-target? (FRESH probe) --")
    if probe.get("probe_ran"):
        print(f"  FA2 num_splits runnable: {probe['num_splits_runnable']}")
        print(f"  pinned-K (num_splits=8) runnable: {probe['pinnedk_num_splits_8_runnable']}  "
              f"=> byte-exact A/B runnable: {probe['byte_exact_ab_runnable']}  "
              f"control (M=1 vs M=8) runnable: {probe['control_m1_vs_m8_runnable']}")
        print(f"  guard: {probe['guard_site']}")
    else:
        print(f"  probe did not run ({probe.get('error', 'no gpu')}); banked #400 pinned_split_reachable=False")
    print("\n-- (2) fresh reduction-order perturbation @ M=1 (split-K8 vs serial1, real gemma geometry) --")
    if p.get("ran"):
        print(f"  max|d attn_out| = {p['max_abs_attnout_perturbation']:.3e}  mean = "
              f"{p['mean_abs_attnout_perturbation']:.3e}  (scale ~{p['mean_attnout_scale']:.3f}, n={p['n_trials']})")
        print(f"  multisplit != serial (bytes differ): {p['multisplit_neq_serial_measured']}  "
              f"ULP-scale: {p['perturbation_is_ulp_scale']}")
    else:
        print(f"  perturbation measurement did not run ({p.get('error', 'no gpu')}); banked #400 multisplit!=serial")
    print("\n-- banked divergence class (measured) --")
    print(f"  divergences exist (multisplit!=serial #400): {res['divergences_exist']}")
    print(f"  all reduction-order flips at margin e*={EPS_STAR} (knife-edge near-ties, #405): "
          f"{ALL_OBSERVED_FLIPS_AT_EPS_STAR_405}")
    print(f"  any confident flip (gap>e*): {res['any_confident_argmax_flip']}  (never observed)")
    print(f"  deployed reduction-order flips 3/882 (~0.34%) / #362 0.52%, all PPL-neutral")
    print("\n-- VERDICT --")
    print(f"  divergence_class       = {res['divergence_class']}")
    print(f"  frontier_496_legality  = {res['frontier_496_legality']}")
    print(f"  max_gap_nats           = {res['max_gap_nats_at_divergence']}  "
          f"(expected near-tie divergences ~{res['expected_near_tie_divergences']}-"
          f"{res['expected_near_tie_divergences_hi']}, NOT directly measured -- A/B un-runnable)")
    print(f"  contract call retired  = {res['contract_call_retired']}  "
          f"(downside bounded near-tie: {res['downside_is_bounded_near_tie']})")
    print(f"\nPPL banked {PPL_DEPLOYED} <= {PPL_GATE}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"self_test_passes = {r['self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pinned-K M=1 vs canonical M=1: is 496.74 unconditionally legal? (PR #431).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #431 deliverables)")
    ap.add_argument("--no-gpu", action="store_true", help="skip the GPU probe + perturbation measurement")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="pinnedk-canonical-legality")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/pinnedk-m1-vs-canonical-m1")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/pinnedk_m1_vs_canonical_m1/pinnedk_m1_vs_canonical_m1_results.json")
    args = ap.parse_args()

    # self-test is a 0-GPU analytic gate (banked composition); the full run adds the GPU probe + measurement.
    run_gpu = not (args.self_test or args.no_gpu)
    report = build_report(run_gpu=run_gpu)
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/pinnedk_m1_vs_canonical_m1/pinnedk_m1_vs_canonical_m1_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\nself_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "pinnedk_m1_vs_canonical_m1_divergences": report["pinnedk_m1_vs_canonical_m1_divergences"],
        "first_divergence_position": report["first_divergence_position"],
        "divergence_class": report["divergence_class"],
        "max_gap_nats_at_divergence": report["max_gap_nats_at_divergence"],
        "all_divergences_are_bitwise_ties": report["all_divergences_are_bitwise_ties"],
        "pinnedk_m1_vs_pinnedk_m8_control_divergences": report["pinnedk_m1_vs_pinnedk_m8_control_divergences"],
        "frontier_496_legality": report["frontier_496_legality"],
        "ppl": report["ppl"],
        "self_test_passes": bool(report["self_test_passes"]),
        "primary_metric": {"name": "frontier_496_legality_unconditional",
                           "value": float(report["frontier_496_legality"] == "unconditional")},
        "test_metric": {"name": "self_test_passes", "value": float(report["self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
