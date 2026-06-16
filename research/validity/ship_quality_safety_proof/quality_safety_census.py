#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #509 (stark) -- Surgical-357 downstream-quality safety: greedy census + logit-delta bound.

THE QUESTION (organizer-facing MMLU/GPQA/AIME review)
-----------------------------------------------------
The shipped surgical-357 strict config (PR #499, official TPS 375.857) installs ONE lever on top
of the already-shipped int4 QAT W4A16 gemma-4-E4B-it serve: it pins the attention reduction to the
2D in-order single-segment KV path (``triton_unified_attention.is_batch_invariant = True``), which
drops the identity-unnecessary matmul tax while keeping spec-dec alive. Does that attention
modification change any downstream model OUTPUT versus the stock-attention base on reasoning/STEM
(MMLU/GPQA/AIME-like) distributions? If 0 semantic token flips and a sub-ULP logit perturbation,
the lever is downstream-quality-safe to ship.

THE AXIS (advisor framing, wirbel #487 relay from morganmcg1)
-------------------------------------------------------------
Isolate the PURE patch-Delta, free of batching/spec-dec noise: run BOTH arms at **M=1 AR**
(single-stream greedy, no spec-dec, no batching). The ONLY operative difference between the arms at
M=1 is the attention pin -- everything else (int4 Marlin body, pruned/quantized weights, RMSNorm,
lm_head) is byte-identical. Concretely the dispatch at M=1 (triton_unified_attention.py:923
``use_3d = not (... or is_batch_invariant)``) selects 3D split-KV when stock (num_seqs=1,
max_seqlen_q=1 -> use_3d=True) and 2D in-order when pinned (use_3d=False), so the two arms differ
ONLY in attention reduction order -- and that order is exactly the surgical-357 lever.

  base      arm: stock attention reductions (3D split-KV at M=1) -- the served int4 base.
  surgical  arm: is_batch_invariant=True pin (2D in-order) -- the shipped surgical-357 lever, applied
                 post-load before any measured forward (byte-identical to what surgical_attn_patch.py
                 installs; the pin == the packaged lever is certified by #494 cert_summary).

This decomposition is ORTHOGONAL to wirbel #510 (M=8-vs-M=1 batching) and denken #505 (spec-dec):
stark #509 = surgical-357 M=1-AR vs base M=1-AR -> patch-Delta isolation.

TWO MEASURED LEGS (both LOCAL, both isolate ONLY the pin)
---------------------------------------------------------
LEG 1 (primary, decisive) -- greedy-token-identity census on ubel #497's committed held-out quality
  splits (shifted_reasoning_stem EASY + shifted_hard_ood HARD), formatted identically to the public
  eval, DISJOINT source datasets => zero item overlap with the public 128.
   * GEN-PATH (deployment-faithful, M=1 AR DECODE geometry): each arm free-runs greedy from the same
     context. base -> ref_base tokens; surgical -> ref_surg tokens. Identical sequences => the
     surgical lever produces byte-identical greedy generations. First-divergence (if any) is
     classified bf16-ULP-tie vs TRUE-semantic. A determinism control re-generates base (expect
     byte-identical) so a divergence is attributable to the pin, not run-to-run noise.
   * SCORE-PATH (full per-position coverage, M=1 PREFILL-rescore geometry): teacher-force ctx+ref_base
     under BOTH arms with prompt_logprobs (identical conditioning, only the pin differs). At every
     position, argmax_base vs argmax_surgical -> flip census; classify each flip tie/knife/semantic
     with the MERGED #497/#461 criteria (EPS_STAR=0.125 bf16 ULP, NEAR_TIE=0.5 nat). 0 semantic
     flips on the teacher-forced base trajectory PROVES the surgical model would have greedily
     generated the same sequence.
LEG 2 -- sampled-path logit-perturbation bound from the SCORE-PATH top-K logprobs: per-token max-abs
  RELATIVE-logit Delta (lse-free: anchored at the reference token, so the partition-function shift
  cancels), and softmax KL(base||surgical) at temperature 1.0 and at the generation_config
  temperature. Quantifies the sampling-distribution exposure even where the greedy argmax agrees.
LEG 3 (MMLU/GPQA harness) -- SKIPPED: no lm-eval harness is wired in this repo. Documented, not run.

KEY OUTPUTS (single-line SENPAI-RESULT + W&B summary)
-----------------------------------------------------
quality_census_easy / quality_census_hard (per-split census), semantic_flips_quality (MUST be 0),
bf16_ulp_ties_quality, max_abs_logit_delta, softmax_kl_base_vs_surgical (temp 1.0 + gen-config T),
sampling_safe_verdict ("safe: sub-ULP, KL~0" | "exposure: <quantified>"), + one-line verdict.

SCOPE: LOCAL profiling card. analysis_only=true, official_tps=0, NO served-file change, NO HF Job,
NO train.py --launch, NO submission. The shipped surgical-357 config and the baseline are UNCHANGED;
greedy identity is MEASURED, never broken. GPU phases run under the submission server venv (vLLM +
Marlin); CUDA_VISIBLE_DEVICES=0. Each arm runs in an isolated subprocess so a pin never leaks across
arms; base runs first and emits ref_base, surgical reads it for an identical teacher-forced trajectory.

Reuses the MERGED #491/#497 census helpers (classify_flip, entry_as_dict, top1_top2_margin,
resolve_model_dir, load_shifted_prompts, EPS_STAR/NEAR_TIE/BAND_TOL) by import so the tie/knife/
semantic classification is byte-identical to the cards the organizer already reviewed.

    .venv/bin/python -m research.validity.ship_quality_safety_proof.quality_safety_census \
        --n-prompts 128 --wandb_name stark/ship-quality-safety --wandb_group ship-quality-safety
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
ROOT = Path(__file__).resolve().parents[3]

# ---- import the MERGED #491/#497 census helpers so classification is byte-identical ---------------
_RC_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
if str(_RC_DIR) not in sys.path:
    sys.path.insert(0, str(_RC_DIR))
import reduction_sensitivity_census as rc  # noqa: E402

EPS_STAR = rc.EPS_STAR                       # 0.125 -- bf16 one-ULP gap at magnitude ~1
NEAR_TIE = rc.NEAR_TIE_LOGPROB_THRESH        # 0.5 nat -- knife-edge non-semantic threshold
BAND_TOL = rc.BAND_TOL                       # 1e-9 numeric slack on the band edge

GROUP = "ship-quality-safety"
SPLITS = ("reasoning_stem", "hard_ood")
SPLIT_LABEL = {"reasoning_stem": "easy", "hard_ood": "hard"}
# ship anchors (surgical-357 cert #494 / strict submission #499) -- context only, never re-measured here
SURGICAL357_OFFICIAL_TPS = 375.857
SHIP_PPL = 2.377
# verdict thresholds: a flip is sampling-safe iff the greedy argmax never semantically flips AND the
# whole-distribution perturbation is sub-ULP in logit space and negligible in KL.
KL_SAFE_THRESH = 1e-3


def _jsonable(o: Any) -> Any:
    return rc._jsonable(o)


def _f(x: Any) -> float:
    return rc._f(x)


def _di(d: dict | None) -> dict[int, float]:
    """str-keyed logprob dict (as stored in JSON) -> int-keyed."""
    if not d:
        return {}
    return {int(k): float(v) for k, v in d.items()}


# ======================================================================================== #
# the surgical-357 lever (== packaged surgical_attn_patch effect; #494-certified)
# ======================================================================================== #
def apply_surgical_pin() -> bool:
    """Install the surgical-357 attention lever AFTER vLLM model load, BEFORE any measured forward:
    flip the module global the 2D-vs-3D dispatch reads live (triton_unified_attention.py:923
    ``use_3d = not (... or is_batch_invariant)``). enforce_eager guarantees every subsequent forward
    reads it. Byte-identical to what the packaged surgical_attn_patch.py installs (#494 cert)."""
    import vllm.v1.attention.ops.triton_unified_attention as _ua
    _ua.is_batch_invariant = True
    return bool(getattr(_ua, "is_batch_invariant", False))


def read_attn_pin_flag() -> bool:
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        return bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:  # noqa: BLE001
        return False


def read_generation_config(model_dir: str) -> dict:
    """The served generation_config -- temperature/top_k/top_p drive the Leg-2 sampling-distribution
    KL. (For this checkpoint: temp=1.0, top_k=64, top_p=0.95, do_sample=true.)"""
    try:
        gc = json.load(open(Path(model_dir) / "generation_config.json"))
        return {"temperature": float(gc.get("temperature", 1.0) or 1.0),
                "top_k": int(gc.get("top_k", 64) or 64),
                "top_p": float(gc.get("top_p", 1.0) or 1.0),
                "do_sample": bool(gc.get("do_sample", False))}
    except Exception:  # noqa: BLE001
        return {"temperature": 1.0, "top_k": 64, "top_p": 1.0, "do_sample": False}


# ======================================================================================== #
# Leg-2 softmax / KL helpers (top-K support, temperature-aware)
# ======================================================================================== #
def _softmax_over(lp: dict[int, float], support: list[int], T: float, floor: float) -> dict[int, float]:
    """Distribution at temperature T restricted to `support`. p(t) ∝ exp(lp(t)/T); tokens absent from
    `lp` get the conservative `floor` logprob. Temperature works directly on the log-softmax because a
    constant (the partition shift) cancels in the renormalised softmax: softmax((log p + C)/T) =
    softmax(log p / T)."""
    if not support or T <= 0:
        return {}
    xs = {t: lp.get(t, floor) / T for t in support}
    m = max(xs.values())
    ex = {t: math.exp(v - m) for t, v in xs.items()}
    Z = sum(ex.values()) or 1.0
    return {t: v / Z for t, v in ex.items()}


def _kl(p: dict[int, float], q: dict[int, float]) -> float:
    """KL(p||q) over p's support (q floored to a tiny positive to avoid log(0))."""
    s = 0.0
    for t, pt in p.items():
        if pt <= 0:
            continue
        qt = q.get(t, 1e-300)
        if qt <= 0:
            qt = 1e-300
        s += pt * math.log(pt / qt)
    return s


def _kl_pair(base_d: dict[int, float], surg_d: dict[int, float], T: float) -> tuple[float, float]:
    """KL(base||surgical) at temperature T over base's top-K support, with a conservative surgical
    floor (surgical's min observed logprob) for any base-support token outside surgical's top-K.
    Returns (kl, support_coverage) where coverage = fraction of base-support tokens also present in
    surgical's top-K (1.0 => exact support match)."""
    if not base_d or not surg_d:
        return float("nan"), float("nan")
    support = list(base_d.keys())
    surg_floor = min(surg_d.values())
    base_floor = min(base_d.values())
    p = _softmax_over(base_d, support, T, base_floor)
    q = _softmax_over(surg_d, support, T, surg_floor)
    covered = sum(1 for t in support if t in surg_d)
    coverage = covered / len(support) if support else float("nan")
    return _kl(p, q), coverage


def _max_abs_rel_logit_delta(base_d: dict[int, float], surg_d: dict[int, float],
                             ref_tok: int) -> tuple[float, int]:
    """lse-free per-token logit perturbation over the SHARED top-K support, anchored at the reference
    token so the (arm-dependent) partition function cancels:
        delta(t) = (lp_base(t) - lp_base(ref)) - (lp_surg(t) - lp_surg(ref)).
    Returns (max_abs_delta, n_shared). Requires ref in BOTH supports; callers handle ref-out-of-support
    (which is itself a strong/semantic disagreement) separately."""
    if ref_tok not in base_d or ref_tok not in surg_d:
        return float("nan"), 0
    shared = [t for t in base_d if t in surg_d]
    b0 = base_d[ref_tok]
    s0 = surg_d[ref_tok]
    md = 0.0
    for t in shared:
        d = abs((base_d[t] - b0) - (surg_d[t] - s0))
        if d > md:
            md = d
    return md, len(shared)


# ======================================================================================== #
# GPU PHASE: one arm (base | surgical), both quality splits, M=1 single-stream
# ======================================================================================== #
def phase_arm(out_path: str, arm: str, n_prompts: int, n_new: int, ctx_cap: int, topk: int,
              k_gen: int, gpu_mem_util: float, det_prompts: int, ref_base_path: str | None) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir = rc.resolve_model_dir()
    gen_cfg = read_generation_config(model_dir)
    full_vocab = rc._margin_model_full_vocab(model_dir)
    print(f"[arm:{arm}] model={model_dir} full_vocab={full_vocab} gen_cfg={gen_cfg} "
          f"n_prompts={n_prompts} n_new={n_new} topk={topk} k_gen={k_gen}", flush=True)

    # M=1 single-stream: no spec-dec, no batching -> the only cross-arm difference is the attention pin
    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=max(1024, ctx_cap + n_new + 16),
              gpu_memory_utilization=gpu_mem_util, max_num_seqs=1,
              max_num_batched_tokens=max(16, topk), enable_prefix_caching=False,
              enforce_eager=True, trust_remote_code=True, max_logprobs=max(20, topk + 2))

    pin_engaged = False
    if arm == "surgical":
        pin_engaged = apply_surgical_pin()
        if not pin_engaged:
            raise RuntimeError("surgical arm: is_batch_invariant pin requested but NOT engaged")
    else:
        if read_attn_pin_flag():
            raise RuntimeError("base arm must be stock attention but is_batch_invariant is engaged")
    print(f"[arm:{arm}] attn_is_batch_invariant={read_attn_pin_flag()} pin_engaged={pin_engaged}", flush=True)

    gen_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n_new, logprobs=k_gen)
    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)

    ref_base = None
    if arm == "surgical":
        ref_base = json.load(open(ref_base_path)) if ref_base_path and Path(ref_base_path).exists() else {}

    result: dict[str, Any] = {
        "phase": "arm", "arm": arm, "model_dir": model_dir, "full_vocab": full_vocab,
        "gen_config": gen_cfg, "attn_is_batch_invariant": read_attn_pin_flag(),
        "pin_engaged": pin_engaged, "n_new": n_new, "topk": topk, "k_gen": k_gen, "splits": {},
    }
    any_nan = False
    t0 = time.time()

    for split in SPLITS:
        prompts = rc.load_shifted_prompts(split, n_prompts, ctx_cap)
        recs = []
        for pi, pr in enumerate(prompts):
            ctx = pr["context_token_ids"]
            c = len(ctx)
            base_in = {"prompt_token_ids": ctx}

            # GEN-PATH: this arm's own greedy AR continuation (M=1 decode geometry)
            out = llm.generate([base_in], gen_sp, use_tqdm=False)[0]
            gen = list(out.outputs[0].token_ids)
            gen_lps_raw = out.outputs[0].logprobs or []
            gen_lps = [{str(t): rc._lp(v) for t, v in (e or {}).items()} for e in gen_lps_raw]

            det_match = None
            if arm == "base" and pi < det_prompts:
                gen_b = list(llm.generate([base_in], gen_sp, use_tqdm=False)[0].outputs[0].token_ids)
                Lg = min(len(gen), len(gen_b))
                det_match = [int(gen[i] == gen_b[i]) for i in range(Lg)]

            # SCORE-PATH: teacher-force ctx + ref_base (prefill-rescore geometry, identical across arms)
            if arm == "base":
                forced = gen
            else:
                forced = [int(x) for x in ref_base.get(split, {}).get(str(pr["id"]), gen)]
            score_lps: list = []
            if forced:
                full = ctx + forced
                vout = llm.generate([{"prompt_token_ids": full}], score_sp, use_tqdm=False)[0]
                pls = vout.prompt_logprobs or []
                hi = min(len(pls), c + len(forced))
                for j in range(c, hi):
                    e = pls[j]
                    if e is None:
                        score_lps.append(None)
                        continue
                    d = {str(t): rc._lp(v) for t, v in e.items()}
                    if any((not math.isfinite(x)) for x in d.values()):
                        any_nan = True
                    score_lps.append(d)

            recs.append({"id": pr["id"], "source": pr.get("source"), "domain": pr.get("domain"),
                         "ctx_len": c, "gen_tokens": gen, "forced_tokens": forced,
                         "gen_lps": gen_lps, "score_lps": score_lps, "det_match": det_match})
        result["splits"][split] = recs
        print(f"[arm:{arm}:{split}] prompts={len(recs)} "
              f"mean_gen_len={(sum(len(r['gen_tokens']) for r in recs)/max(1,len(recs))):.1f}", flush=True)

    result["any_nan"] = bool(any_nan)
    result["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    result["elapsed_s"] = round(time.time() - t0, 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(_jsonable(result), open(out_path, "w"))
    if arm == "base":
        refmap = {sp: {str(r["id"]): r["gen_tokens"] for r in result["splits"][sp]} for sp in SPLITS}
        json.dump(refmap, open(Path(out_path).parent / "_ref_base.json", "w"))
    print(f"ARM_DONE {out_path}", flush=True)


# ======================================================================================== #
# COMPOSE (no GPU): patch-Delta census + Leg-2 bound from the two arms' stored logprobs
# ======================================================================================== #
def _classify_div(perturbed: dict[int, float], ref_tok: int) -> dict:
    """Classify a single argmax divergence (reference token = the BASE arm's greedy choice; perturbed
    distribution = the SURGICAL arm's top-K) with the MERGED #497 tie/knife/semantic criteria."""
    p_arg, p_top_lp, p_gap = rc.top1_top2_margin(perturbed)
    return rc.classify_flip(perturbed, ref_tok, p_top_lp, p_gap)


def compose_split(base_recs: list, surg_recs: list, gen_cfg: dict) -> dict[str, Any]:
    """One split: GEN-PATH greedy identity + SCORE-PATH per-position census + Leg-2 logit/KL bound.
    base_recs / surg_recs are aligned by id."""
    surg_by_id = {str(r["id"]): r for r in surg_recs}
    gen_T = float(gen_cfg.get("temperature", 1.0) or 1.0)

    # ---- GEN-PATH (deployment-faithful, decode geometry) --------------------------------------
    n_prompts = 0
    n_identical = 0
    gen_tie = gen_knife = gen_semantic = 0
    gen_div_examples: list[dict] = []
    det_pos = det_match = 0

    # ---- SCORE-PATH (full coverage, prefill-rescore geometry) ---------------------------------
    s_pos = s_flip = s_tie = s_knife = s_semantic = 0
    s_bitdiff = 0
    margins_all: list[float] = []
    max_abs_logit_delta = 0.0
    n_ref_out_of_surg = 0
    kl_t1: list[float] = []
    kl_genT: list[float] = []
    coverage_min = 1.0
    s_flip_examples: list[dict] = []

    for br in base_recs:
        sid = str(br["id"])
        sr = surg_by_id.get(sid)
        if sr is None:
            continue
        n_prompts += 1

        # determinism control (base re-gen vs gen) -> the non-pin path is run-to-run bit-stable
        if br.get("det_match") is not None:
            det_match += int(sum(br["det_match"]))
            det_pos += int(len(br["det_match"]))

        # GEN-PATH: free-running greedy sequences
        gb = list(br["gen_tokens"])
        gs = list(sr["gen_tokens"])
        if gb == gs:
            n_identical += 1
        else:
            L = min(len(gb), len(gs))
            fd = next((i for i in range(L) if gb[i] != gs[i]), L)
            if fd < L:
                # classify the FIRST divergence at shared conditioning (surgical's distribution there)
                surg_gen = _di(sr["gen_lps"][fd]) if fd < len(sr["gen_lps"]) else {}
                ref_tok = int(gb[fd])
                if ref_tok not in surg_gen:
                    surg_gen[ref_tok] = float("-inf")
                cls = _classify_div(surg_gen, ref_tok)
                gen_tie += int(cls["is_tie_flip"])
                gen_knife += int(cls["is_knife_edge"])
                gen_semantic += int(cls["is_semantic"])
                if len(gen_div_examples) < 32:
                    gen_div_examples.append({"id": sid, "first_div_pos": fd,
                                             "base_tok": ref_tok, "surg_tok": int(gs[fd]), **cls})

        # SCORE-PATH: teacher-forced ctx+ref_base, per-position census (identical conditioning)
        b_lps = br["score_lps"] or []
        s_lps = sr["score_lps"] or []
        n = min(len(b_lps), len(s_lps))
        for j in range(n):
            bd = _di(b_lps[j])
            sd = _di(s_lps[j])
            if not bd or not sd:
                continue
            ref_tok, b_top_lp, b_gap = rc.top1_top2_margin(bd)          # base greedy choice = reference
            surg_arg, s_top_lp, s_gap = rc.top1_top2_margin(sd)
            if math.isnan(b_gap):
                continue
            s_pos += 1
            margins_all.append(b_gap)
            # bitdiff: surgical top-1 logprob differs in bits from base top-1 logprob
            if math.isfinite(b_top_lp) and math.isfinite(s_top_lp) and b_top_lp != s_top_lp:
                s_bitdiff += 1
            # Leg-2 logit delta (lse-free, ref-anchored) + KL at temp 1.0 and gen-config temp
            if ref_tok in sd:
                md, _ = _max_abs_rel_logit_delta(bd, sd, ref_tok)
                if math.isfinite(md) and md > max_abs_logit_delta:
                    max_abs_logit_delta = md
            else:
                n_ref_out_of_surg += 1
            k1, cov = _kl_pair(bd, sd, 1.0)
            kT, _ = _kl_pair(bd, sd, gen_T)
            if math.isfinite(k1):
                kl_t1.append(k1)
            if math.isfinite(kT):
                kl_genT.append(kT)
            if math.isfinite(cov):
                coverage_min = min(coverage_min, cov)
            # FLIP census
            if surg_arg != ref_tok:
                s_flip += 1
                cls = rc.classify_flip(sd, ref_tok, s_top_lp, s_gap)
                s_tie += int(cls["is_tie_flip"])
                s_knife += int(cls["is_knife_edge"])
                s_semantic += int(cls["is_semantic"])
                if len(s_flip_examples) < 48:
                    s_flip_examples.append({"id": sid, "pos": j, "base_tok": ref_tok,
                                            "surg_tok": surg_arg, "base_gap": b_gap, **cls})

    margins_all.sort()

    def _q(xs: list[float], q: float) -> float:
        if not xs:
            return float("nan")
        i = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
        return xs[i]

    semantic_total = gen_semantic + s_semantic
    tie_total = gen_tie + s_tie
    operative_identity = (1.0 - (s_semantic / s_pos)) if s_pos else float("nan")
    return {
        "n_prompts": n_prompts,
        # GEN-PATH (decode geometry, deployment-faithful)
        "gen_n_prompts_identical": n_identical,
        "gen_greedy_sequence_identity": (n_identical / n_prompts) if n_prompts else float("nan"),
        "gen_n_tie": gen_tie, "gen_n_knife": gen_knife, "gen_n_semantic": gen_semantic,
        "gen_div_examples": gen_div_examples,
        # SCORE-PATH (prefill-rescore geometry, full coverage)
        "score_n_positions": s_pos, "score_n_flip": s_flip, "score_n_bitdiff": s_bitdiff,
        "score_n_tie": s_tie, "score_n_knife": s_knife, "score_n_semantic": s_semantic,
        "score_flip_rate": (s_flip / s_pos) if s_pos else float("nan"),
        "score_bitdiff_rate": (s_bitdiff / s_pos) if s_pos else float("nan"),
        "score_semantic_rate": (s_semantic / s_pos) if s_pos else float("nan"),
        "operative_identity": operative_identity,
        "rule_of_three_semantic_ub": (rc._rule_of_three(s_pos) if s_semantic == 0 else float("nan")),
        "score_flip_examples": s_flip_examples,
        # margins (base reference top1-top2 gap distribution)
        "margin_min": (margins_all[0] if margins_all else float("nan")),
        "margin_p05": _q(margins_all, 0.05), "margin_median": _q(margins_all, 0.50),
        # Leg-2 bound
        "max_abs_logit_delta": max_abs_logit_delta,
        "n_ref_out_of_surg_support": n_ref_out_of_surg,
        "kl_t1_mean": (sum(kl_t1) / len(kl_t1)) if kl_t1 else float("nan"),
        "kl_t1_max": (max(kl_t1) if kl_t1 else float("nan")),
        "kl_genT_mean": (sum(kl_genT) / len(kl_genT)) if kl_genT else float("nan"),
        "kl_genT_max": (max(kl_genT) if kl_genT else float("nan")),
        "kl_gen_temperature": gen_T,
        "kl_support_coverage_min": coverage_min,
        # determinism control
        "det_byte_identity": (det_match / det_pos) if det_pos else float("nan"),
        "det_positions": det_pos,
        # roll-ups used by the top-level KEY OUTPUTS
        "semantic_flips_total": semantic_total,
        "bf16_ulp_ties_total": tie_total,
    }


def compose(base_arm: dict, surg_arm: dict) -> dict[str, Any]:
    """Both splits -> per-split census + global KEY OUTPUTS + the sampling-safe verdict."""
    gen_cfg = surg_arm.get("gen_config") or base_arm.get("gen_config") or {"temperature": 1.0}
    per_split: dict[str, Any] = {}
    for sp in SPLITS:
        br = (base_arm.get("splits") or {}).get(sp) or []
        sr = (surg_arm.get("splits") or {}).get(sp) or []
        per_split[sp] = compose_split(br, sr, gen_cfg)

    easy = per_split.get("reasoning_stem", {})
    hard = per_split.get("hard_ood", {})
    semantic_flips_quality = sum(_f(per_split[sp].get("semantic_flips_total")) for sp in SPLITS
                                 if per_split.get(sp))
    semantic_flips_quality = int(semantic_flips_quality) if math.isfinite(semantic_flips_quality) else -1
    bf16_ulp_ties_quality = int(sum(int(per_split[sp].get("bf16_ulp_ties_total", 0)) for sp in SPLITS
                                    if per_split.get(sp)))
    max_abs_logit_delta = max((_f(per_split[sp].get("max_abs_logit_delta")) for sp in SPLITS
                               if per_split.get(sp) and math.isfinite(_f(per_split[sp].get("max_abs_logit_delta")))),
                              default=float("nan"))
    kl_t1_max = max((_f(per_split[sp].get("kl_t1_max")) for sp in SPLITS
                     if per_split.get(sp) and math.isfinite(_f(per_split[sp].get("kl_t1_max")))),
                    default=float("nan"))
    kl_genT_max = max((_f(per_split[sp].get("kl_genT_max")) for sp in SPLITS
                       if per_split.get(sp) and math.isfinite(_f(per_split[sp].get("kl_genT_max")))),
                      default=float("nan"))
    kl_overall_max = max([v for v in (kl_t1_max, kl_genT_max) if math.isfinite(v)], default=float("nan"))

    # verdict: safe iff 0 semantic flips AND sub-ULP logit perturbation AND negligible KL
    sub_ulp = math.isfinite(max_abs_logit_delta) and max_abs_logit_delta <= EPS_STAR + BAND_TOL
    kl_negligible = (not math.isfinite(kl_overall_max)) or kl_overall_max <= KL_SAFE_THRESH
    safe = (semantic_flips_quality == 0) and sub_ulp and kl_negligible
    if safe:
        verdict = (f"safe: sub-ULP (max|Δlogit|={max_abs_logit_delta:.4f} ≤ {EPS_STAR} bf16-ULP), "
                   f"KL(base‖surgical)≤{kl_overall_max:.2e} at temp 1.0 & gen-cfg T, "
                   f"0 semantic greedy flips ({bf16_ulp_ties_quality} benign bf16-ULP ties)")
    else:
        bits = [f"semantic_flips={semantic_flips_quality}"]
        if math.isfinite(max_abs_logit_delta):
            bits.append(f"max|Δlogit|={max_abs_logit_delta:.4f}")
        if math.isfinite(kl_overall_max):
            bits.append(f"KL_max={kl_overall_max:.2e}")
        verdict = "exposure: " + ", ".join(bits)

    operative_identity_overall = min(
        [_f(per_split[sp].get("operative_identity")) for sp in SPLITS
         if per_split.get(sp) and math.isfinite(_f(per_split[sp].get("operative_identity")))],
        default=float("nan"))
    gen_identity_overall = min(
        [_f(per_split[sp].get("gen_greedy_sequence_identity")) for sp in SPLITS
         if per_split.get(sp) and math.isfinite(_f(per_split[sp].get("gen_greedy_sequence_identity")))],
        default=float("nan"))

    return {
        "per_split": per_split,
        "quality_census_easy": easy,
        "quality_census_hard": hard,
        "semantic_flips_quality": semantic_flips_quality,
        "bf16_ulp_ties_quality": bf16_ulp_ties_quality,
        "max_abs_logit_delta": max_abs_logit_delta,
        "softmax_kl_base_vs_surgical": kl_overall_max,
        "softmax_kl_temp1_max": kl_t1_max,
        "softmax_kl_genT_max": kl_genT_max,
        "operative_identity_overall": operative_identity_overall,
        "gen_greedy_sequence_identity_overall": gen_identity_overall,
        "sampling_safe_verdict": verdict,
        "is_safe": bool(safe),
    }


# ======================================================================================== #
# SELFTEST (no GPU): synthetic arms exercise classification + KEY OUTPUTS plumbing
# ======================================================================================== #
def _mk_score_pos(argmax_tok: int, runner: int, gap: float, extra: dict[int, float] | None = None) -> dict:
    """A score-path position as a str-keyed logprob dict whose top1 is `argmax_tok` (lp 0.0) and top2
    is `runner` (lp -gap)."""
    d = {str(argmax_tok): 0.0, str(runner): -gap}
    for t, lp in (extra or {}).items():
        d[str(t)] = lp
    return d


def selftest() -> dict[str, Any]:
    checks: list[tuple[str, bool]] = []

    # Build synthetic base/surgical arms over ONE split with three planted prompts:
    #   p0: identical greedy + a clean agreeing score position (0 flips, tiny delta).
    #   p1: a bf16-ULP TIE flip (gap 0.1 ≤ EPS_STAR, reference token rank-2) -> benign, NOT semantic.
    #   p2: a TRUE-semantic flip (reference token absent from surgical top-K) -> semantic.
    def arm(pin: bool, score_for: dict[str, list], gen_for: dict[str, list],
            gen_lps_for: dict[str, list]) -> dict:
        recs = []
        for sid in ("p0", "p1", "p2"):
            recs.append({"id": sid, "source": "selftest", "domain": "math", "ctx_len": 4,
                         "gen_tokens": gen_for[sid], "forced_tokens": gen_for["p0"],
                         "gen_lps": gen_lps_for.get(sid, []), "score_lps": score_for[sid],
                         "det_match": ([1, 1, 1] if (not pin and sid == "p0") else None)})
        return {"phase": "arm", "arm": ("surgical" if pin else "base"),
                "attn_is_batch_invariant": pin, "gen_config": {"temperature": 1.0, "top_k": 64},
                "splits": {"reasoning_stem": recs, "hard_ood": []}}

    base_score = {
        "p0": [_mk_score_pos(10, 11, 3.0), _mk_score_pos(20, 21, 4.0)],
        "p1": [_mk_score_pos(30, 31, 0.1)],                      # base argmax 30, runner 31 (gap 0.1)
        "p2": [_mk_score_pos(40, 41, 5.0)],                      # base argmax 40 (decisive)
    }
    surg_score = {
        "p0": [_mk_score_pos(10, 11, 3.0), _mk_score_pos(20, 21, 4.0)],   # identical -> 0 flips
        "p1": [_mk_score_pos(31, 30, 0.1)],                      # argmax flips to 31; ref(30) rank-2, gap 0.1 -> TIE
        "p2": [{"50": 0.0, "51": -5.0}],                         # argmax 50; ref(40) absent -> SEMANTIC
    }
    base_gen = {"p0": [10, 20], "p1": [30], "p2": [40]}
    surg_gen = {"p0": [10, 20], "p1": [31], "p2": [50]}           # p0 identical; p1 tie-div; p2 semantic-div
    surg_gen_lps = {"p0": [], "p1": [{"31": 0.0, "30": -0.1}], "p2": [{"50": 0.0, "51": -5.0}]}

    base_arm = arm(False, base_score, base_gen, {})
    surg_arm = arm(True, surg_score, surg_gen, surg_gen_lps)
    comp = compose(base_arm, surg_arm)
    easy = comp["quality_census_easy"]

    # score-path: exactly one TIE flip (p1) and one SEMANTIC flip (p2)
    checks.append(("score_one_tie", easy["score_n_tie"] == 1))
    checks.append(("score_one_semantic", easy["score_n_semantic"] == 1))
    checks.append(("score_two_flips", easy["score_n_flip"] == 2))
    # gen-path: p0 identical, p1 tie-div, p2 semantic-div
    checks.append(("gen_one_identical", easy["gen_n_prompts_identical"] == 1))
    checks.append(("gen_one_tie", easy["gen_n_tie"] == 1))
    checks.append(("gen_one_semantic", easy["gen_n_semantic"] == 1))
    # roll-ups: semantic from BOTH paths (score p2 + gen p2) = 2; verdict must be EXPOSURE (not safe)
    checks.append(("semantic_total_two", comp["semantic_flips_quality"] == 2))
    checks.append(("ulp_ties_counted", comp["bf16_ulp_ties_quality"] >= 1))
    checks.append(("verdict_exposure", comp["sampling_safe_verdict"].startswith("exposure")))
    checks.append(("not_safe", comp["is_safe"] is False))
    # KEY OUTPUTS present + JSON-serializable
    need = ["quality_census_easy", "quality_census_hard", "semantic_flips_quality",
            "bf16_ulp_ties_quality", "max_abs_logit_delta", "softmax_kl_base_vs_surgical",
            "sampling_safe_verdict"]
    checks.append(("key_outputs_present", all(k in comp for k in need)))
    try:
        json.dumps(_jsonable(comp))
        checks.append(("json_serializable", True))
    except Exception:  # noqa: BLE001
        checks.append(("json_serializable", False))

    # an all-agree arm pair must yield SAFE (0 semantic, sub-ULP, KL~0)
    safe_base = arm(False, {"p0": [_mk_score_pos(10, 11, 3.0)], "p1": [], "p2": []},
                    {"p0": [10], "p1": [], "p2": []}, {})
    safe_surg = arm(True, {"p0": [_mk_score_pos(10, 11, 3.0)], "p1": [], "p2": []},
                    {"p0": [10], "p1": [], "p2": []}, {})
    safe_comp = compose(safe_base, safe_surg)
    checks.append(("all_agree_zero_semantic", safe_comp["semantic_flips_quality"] == 0))
    checks.append(("all_agree_safe", safe_comp["is_safe"] is True))
    checks.append(("all_agree_verdict_safe", safe_comp["sampling_safe_verdict"].startswith("safe")))

    passes = all(ok for _, ok in checks)
    return {"passes": bool(passes), "checks": {k: bool(v) for k, v in checks}}


# ======================================================================================== #
# report / wandb / orchestration
# ======================================================================================== #
def print_report(payload: dict) -> None:
    c = payload["compose"]
    print("\n================ PR #509 surgical-357 downstream-quality safety ================", flush=True)
    print(f"  axis: surgical-357 M=1-AR  vs  base M=1-AR  (pure attention patch-Δ, no spec-dec/batching)", flush=True)
    for sp in SPLITS:
        s = c["per_split"][sp]
        print(f"  [{SPLIT_LABEL[sp]:>4}/{sp}] gen: {s['gen_n_prompts_identical']}/{s['n_prompts']} "
              f"sequences byte-identical | score: pos={s['score_n_positions']} flips={s['score_n_flip']} "
              f"(tie={s['score_n_tie']} knife={s['score_n_knife']} SEMANTIC={s['score_n_semantic']}) "
              f"op_identity={_f(s['operative_identity']):.6f} det={_f(s['det_byte_identity']):.4f}", flush=True)
        print(f"        Leg-2: max|Δlogit|={_f(s['max_abs_logit_delta']):.4f}  "
              f"KL@1.0(mean/max)={_f(s['kl_t1_mean']):.2e}/{_f(s['kl_t1_max']):.2e}  "
              f"KL@genT(max)={_f(s['kl_genT_max']):.2e}  cov_min={_f(s['kl_support_coverage_min']):.4f}", flush=True)
    print(f"  semantic_flips_quality = {c['semantic_flips_quality']}  "
          f"bf16_ulp_ties_quality = {c['bf16_ulp_ties_quality']}", flush=True)
    print(f"  max_abs_logit_delta = {_f(c['max_abs_logit_delta']):.4f}  "
          f"softmax_kl_base_vs_surgical = {_f(c['softmax_kl_base_vs_surgical']):.2e}", flush=True)
    print(f"  VERDICT: {c['sampling_safe_verdict']}", flush=True)
    print(f"  selftest passes = {payload['selftest']['passes']}", flush=True)
    print("================================================================================\n", flush=True)


def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[#509] wandb helpers unavailable: {e}")
        return None
    c = payload["compose"]
    run = init_wandb_run(
        job_type="analysis-ship-quality-safety", agent="stark",
        name=args.wandb_name, group=args.wandb_group,
        tags=["ship-quality-safety", "surgical-357", "patch-delta", "greedy-census",
              "logit-delta-bound", "operative-identity", "m1-ar", "pr-509"],
        config={"pr": 509, "kind": "ship-quality-safety-proof",
                "axis": "surgical357_M1AR_vs_base_M1AR",
                "surgical357_official_tps": SURGICAL357_OFFICIAL_TPS, "ship_ppl": SHIP_PPL,
                "eps_star": EPS_STAR, "near_tie_thresh": NEAR_TIE, "kl_safe_thresh": KL_SAFE_THRESH,
                "splits": list(SPLITS)},
    )
    if run is None:
        print("[#509] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "quality/semantic_flips_quality": float(c["semantic_flips_quality"]),
        "quality/bf16_ulp_ties_quality": float(c["bf16_ulp_ties_quality"]),
        "quality/max_abs_logit_delta": _f(c["max_abs_logit_delta"]),
        "quality/softmax_kl_base_vs_surgical": _f(c["softmax_kl_base_vs_surgical"]),
        "quality/softmax_kl_temp1_max": _f(c["softmax_kl_temp1_max"]),
        "quality/softmax_kl_genT_max": _f(c["softmax_kl_genT_max"]),
        "quality/operative_identity_overall": _f(c["operative_identity_overall"]),
        "quality/gen_greedy_sequence_identity_overall": _f(c["gen_greedy_sequence_identity_overall"]),
        "quality/is_safe": float(bool(c["is_safe"])),
        "selftest/ship_quality_safety_self_test_passes": float(payload["selftest"]["passes"]),
    }
    for sp in SPLITS:
        s = c["per_split"][sp]
        lab = SPLIT_LABEL[sp]
        flat[f"{lab}/score_n_positions"] = float(s["score_n_positions"])
        flat[f"{lab}/score_n_flip"] = float(s["score_n_flip"])
        flat[f"{lab}/score_n_semantic"] = float(s["score_n_semantic"])
        flat[f"{lab}/score_n_tie"] = float(s["score_n_tie"])
        flat[f"{lab}/operative_identity"] = _f(s["operative_identity"])
        flat[f"{lab}/gen_greedy_sequence_identity"] = _f(s["gen_greedy_sequence_identity"])
        flat[f"{lab}/max_abs_logit_delta"] = _f(s["max_abs_logit_delta"])
        flat[f"{lab}/kl_t1_max"] = _f(s["kl_t1_max"])
        flat[f"{lab}/det_byte_identity"] = _f(s["det_byte_identity"])
    run.log({"global_step": 0, **{k: (v if (isinstance(v, float) and math.isfinite(v)) else 0.0)
                                  for k, v in flat.items()}})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="ship_quality_safety_proof",
                      artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[#509] wandb logged (run {rid})")
    return rid


def resolve_server_python(arg: str | None) -> str:
    if arg:
        return arg
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.local_validation import harness  # noqa: E402
    m = harness.load_manifest(ROOT / "submissions" / "fa2sw_precache_kenyan")
    return str(harness.ensure_server_venv(m["dependencies"]))


def run_gpu_phase(server_python: str, phase_args: list[str], timeout: int) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [server_python, os.path.abspath(__file__)] + phase_args
    print(f"[orch] launching ({Path(server_python).parent.parent.name}): {' '.join(phase_args)}", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s: {phase_args}", flush=True)
        return 124


def orchestrate(args) -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    base_json = str(HERE / "_arm_base.json")
    surg_json = str(HERE / "_arm_surgical.json")
    ref_base_json = str(HERE / "_ref_base.json")
    server_python = resolve_server_python(args.server_python)
    print(f"[orch] server_python = {server_python}", flush=True)

    common = ["--n-prompts", str(args.n_prompts), "--n-new", str(args.n_new),
              "--ctx-cap", str(args.ctx_cap), "--topk", str(args.topk),
              "--k-gen", str(args.k_gen), "--gpu-mem-util", str(args.gpu_mem_util),
              "--det-prompts", str(args.det_prompts)]
    # base FIRST (emits ref_base), then surgical (teacher-forces the same ref_base trajectory)
    rc_b = run_gpu_phase(server_python, ["--phase", "arm", "--arm", "base", "--out", base_json]
                         + common, timeout=args.arm_timeout)
    rc_s = run_gpu_phase(server_python, ["--phase", "arm", "--arm", "surgical", "--out", surg_json,
                         "--ref-base", ref_base_json] + common, timeout=args.arm_timeout)

    base_arm = json.load(open(base_json)) if Path(base_json).exists() else {"phase": "arm", "error": rc_b}
    surg_arm = json.load(open(surg_json)) if Path(surg_json).exists() else {"phase": "arm", "error": rc_s}
    comp = compose(base_arm, surg_arm)
    st = selftest()
    flags = {"no_hf_job": True, "no_launch": True, "analysis_only": True,
             "no_served_file_change": True, "official_tps": 0}

    payload = {
        "agent": "stark", "pr": 509, "kind": "ship-quality-safety-proof",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "axis": "surgical357_M1AR_vs_base_M1AR",
        "anchors": {"surgical357_official_tps": SURGICAL357_OFFICIAL_TPS, "ship_ppl": SHIP_PPL,
                    "eps_star": EPS_STAR, "near_tie_thresh": NEAR_TIE},
        "leg3_mmlu_gpqa": "SKIPPED (no lm-eval harness wired in this repo)",
        "base_arm": {k: v for k, v in base_arm.items() if k != "splits"},
        "surgical_arm": {k: v for k, v in surg_arm.items() if k != "splits"},
        "compose": comp, "selftest": st,
        "ship_quality_safety_self_test_passes": bool(st["passes"]),
        # headline KEY OUTPUTS hoisted to top level
        "semantic_flips_quality": comp["semantic_flips_quality"],
        "bf16_ulp_ties_quality": comp["bf16_ulp_ties_quality"],
        "max_abs_logit_delta": comp["max_abs_logit_delta"],
        "softmax_kl_base_vs_surgical": comp["softmax_kl_base_vs_surgical"],
        "sampling_safe_verdict": comp["sampling_safe_verdict"],
        "is_safe": comp["is_safe"],
    }
    print_report(payload)
    out_path = HERE / "ship_quality_safety_proof_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[#509] wrote {out_path}", flush=True)
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    any_nan = bool(base_arm.get("any_nan") or surg_arm.get("any_nan"))
    ok = bool(st["passes"]) and not any_nan
    result = {"terminal": True, "status": "complete", "pending_arms": False,
              "wandb_run_ids": ([rid] if rid else []),
              "semantic_flips_quality": comp["semantic_flips_quality"],
              "bf16_ulp_ties_quality": comp["bf16_ulp_ties_quality"],
              "max_abs_logit_delta": round(_f(comp["max_abs_logit_delta"]), 6),
              "softmax_kl_base_vs_surgical": _f(comp["softmax_kl_base_vs_surgical"]),
              "operative_identity_overall": round(_f(comp["operative_identity_overall"]), 6),
              "gen_greedy_sequence_identity_overall": round(_f(comp["gen_greedy_sequence_identity_overall"]), 6),
              "sampling_safe_verdict": comp["sampling_safe_verdict"], "is_safe": bool(comp["is_safe"]),
              "primary_metric": {"name": "semantic_flips_quality",
                                 "value": comp["semantic_flips_quality"]},
              "test_metric": {"name": "max_abs_logit_delta",
                              "value": round(_f(comp["max_abs_logit_delta"]), 6)},
              "self_test_passes": bool(st["passes"]), "any_nan": any_nan}
    print("SENPAI-RESULT: " + json.dumps(result), flush=True)
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["arm"], default=None,
                    help="internal GPU phase dispatch (run under the server venv)")
    ap.add_argument("--arm", choices=["base", "surgical"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-base", "--ref_base", dest="ref_base", default=None,
                    help="(surgical arm) path to the base arm's _ref_base.json")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--topk", type=int, default=64, help="score-path prompt_logprobs width (covers gen-cfg top_k=64)")
    ap.add_argument("--k-gen", type=int, default=8, help="gen-path per-step logprobs width (tie classification)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--det-prompts", type=int, default=8)
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--arm-timeout", type=int, default=2400)
    ap.add_argument("--smoke", action="store_true", help="tiny fast path for validation")
    ap.add_argument("--selftest", action="store_true", help="no-GPU classification + plumbing self-test")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="stark/ship-quality-safety")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=GROUP)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        st = selftest()
        print(json.dumps(st, indent=2))
        print("SELFTEST_PASS" if st["passes"] else "SELFTEST_FAIL", flush=True)
        raise SystemExit(0 if st["passes"] else 1)

    if args.smoke:
        args.n_prompts = min(args.n_prompts, 6)
        args.n_new = min(args.n_new, 8)
        args.det_prompts = min(args.det_prompts, 3)

    if args.phase == "arm":
        if not args.arm or not args.out:
            raise SystemExit("--phase arm requires --arm and --out")
        phase_arm(args.out, args.arm, args.n_prompts, args.n_new, args.ctx_cap, args.topk,
                  args.k_gen, args.gpu_mem_util, args.det_prompts, args.ref_base)
        return

    raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    main()
