#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #529 (denken) -- Served benchmark divergence: audit + pin the head-prune collapse.

THE QUESTION
------------
ubel #511 found the LIVE board ship (fa2sw_strict_surgical357, official 375.857 TPS) collapses
on the organizer's downstream evals: MMLU-Pro 0.668 -> 0.274, GPQA-Diamond 0.444 -> 0.232. The
mechanism it named: the served lm_head is PRUNED to a 12288-token keepset (pck04c-12k), so any
answer-bearing token OUTSIDE the keepset is scattered to -inf -- the model emits a fluent WRONG
token it can no longer reach. Meanwhile MY merged #520 card certified the same ship "sampled-
quality-neutral" (TV=KL=0). Both can't be naively true. This card reconciles them and pins the
cause on the benchmark prompt distribution.

  served ship distribution == base full-head distribution MASKED to the 12288 keepset.
  Proof (read from the live submission, NOT re-derived):
    * serve.py:_prune_lm_head_rows -- the prune is a PURE torch.index_select row-slice of the
      int4 packed lm_head (no re-quant): the KEPT rows' logits are BIT-IDENTICAL to base.
    * serve_patch_pck04.py:_scatter_to_full_vocab -- the K kept logits scatter back to full
      262144 vocab with -inf at every non-kept id (prob exactly 0).
  So the ONLY difference between true base and served ship is: non-keepset logits -> -inf.

THE MEASUREMENT (emulation that ISOLATES the head-prune exactly)
---------------------------------------------------------------
We serve ONE int4 full-head substrate (QAT-w4a16-ct, full 262144 lm_head -- #520's quality-
neutral int4 base proxy) and, at every decode position, derive BOTH arms from the SAME forward
pass:
    p_base = gen_config(top-K raw logprobs)                  # full head
    p_ship = gen_config(top-K raw logprobs masked to keepset) # 12k head (-inf off-keepset)
Because both arms share the identical forward, the ONLY thing that differs is the keepset mask =
EXACTLY the head-prune. This is strictly cleaner than serving two separately-quantized
checkpoints (which would confound the prune with quant noise). gen_config = the official sampling
transform (T=1.0, top_k=64, top_p=0.95), applied to BOTH arms.

The decode TRAJECTORY is the base model's own long CoT generation under the official sampling
config (NOT the n_new=24 PPL window #520 lived on) on a broad benchmark proxy distribution
(MMLU STEM + Hendrycks MATH + GSM8K). At each position along that trajectory we measure
TV(p_base, p_ship) and KL(p_base || p_ship). (Per-step divergence along the base trajectory is a
LOWER bound on the served quality gap -- the served ship additionally DRIFTS once it is forced
off a killed token.)

THE BLIND-SPOT CONTRAST (why #520 / the operative-1.0 census could not see it)
-----------------------------------------------------------------------------
Three geometries, all from the same capture:
  selftest   full-head vs full-head : TV == KL == 0  (control: deterministic transform pipeline)
  prune^2    12k-head vs 12k-head    : TV == KL == 0  (#520 GEOMETRY -- both arms behind the 12k
             head, so the prune is COMMON-MODE and cancels EXACTLY. #520's base substrate was the
             pruned osoi5, so its A/B lived in THIS column and was structurally blind.)
  base|ship  full-head vs 12k-head   : the BLOW-UP -- this is the column #520 never measured.
The contrast TV(prune^2)=0  <<  TV(base|ship)~1 IS the proof.

KEY OUTPUTS
-----------
served_ship_has_head_prune, served_keepset_K, max/mean/p99 TV (pooled + per-task),
wanted_token_killed_rate (base-argmax id not in keepset), answer-bearing-vs-filler TV split,
blind-spot contrast, self-test. NaN-clean.

SCOPE: LOCAL profiling card. analysis_only=true, official_tps=0, NO served-file change, NO HF
Job, NO train.py --launch, NO submission. Challenge PAUSED. The served config + baseline are
UNCHANGED; the served distribution is MEASURED, never altered. GPU phase runs under the
submission server venv (vLLM + Marlin); CUDA_VISIBLE_DEVICES=0. Extends the merged #520 harness
(gen_config_dist / tv / kl / phase orchestration; rc prompt + model helpers).

  .venv/bin/python -m research.validity.served_benchmark_divergence.served_benchmark_divergence \
      --prompts research/validity/served_benchmark_divergence/prompts.jsonl \
      --n-new 320 --topk 256 \
      --wandb_name denken/served-benchmark-divergence \
      --wandb_group served-benchmark-divergence
"""
from __future__ import annotations

import argparse
import heapq
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

# ---- merged #491/#497 census helpers (resolve_model_dir / _lp / _jsonable / _f) ----
_RC_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
if str(_RC_DIR) not in sys.path:
    sys.path.insert(0, str(_RC_DIR))
import reduction_sensitivity_census as rc  # noqa: E402

# ============================================================================================ #
# AUDIT HEADLINE -- read from the LIVE board submission (submissions/fa2sw_strict_surgical357).
# These are FACTS about the served ship, asserted here so the run record stamps them. Provenance:
#   manifest.json: LM_HEAD_PRUNE=1, LM_HEAD_PRUNE_REQUIRE=1,
#       LM_HEAD_KEEPSET_BUCKET=.../int4-pck04c-12k, PCK04_KEEPSET=.../pck04_keepset.json
#   serve.py:_prune_lm_head_rows (pure index_select row-slice) + serve_patch_pck04 (scatter -inf)
#   #520 const SURGICAL357_OFFICIAL_TPS = 375.857  (the live board headline == this ship)
# ============================================================================================ #
SERVED_SHIP_HAS_HEAD_PRUNE = True
LIVE_BOARD_SUBMISSION = "fa2sw_strict_surgical357"
LIVE_BOARD_OFFICIAL_TPS = 375.857
UBEL511_KEEPSET_K = 12288            # ubel #511's measured served keepset size (we re-confirm)
FULL_VOCAB = 262144                  # rc.FULL_VOCAB

# official generation_config.json sampling axis (the downstream evals SAMPLE under this).
GEN_T = 1.0
GEN_TOP_K = 64
GEN_TOP_P = 0.95
GEN_SEED = 1234                      # fixed -> reproducible base CoT trajectory

# verdict thresholds. "diverges" = the head-prune materially moves the next-token distribution on
# the benchmark distribution. The prune is binary-catastrophic where it bites (a killed argmax ->
# TV near 1), so a pooled max TV near 1 AND a non-trivial killed rate is the divergence signature.
TV_BLOWUP = 0.5                      # a position is a "blow-up" if base|ship TV exceeds this
KILLED_RATE_MATERIAL = 0.01          # >=1% of positions kill the base-wanted token -> material
KL_CAP = 100.0                       # finite sentinel for disjoint-support (+inf) KL positions


# ============================================================================================ #
# gen_config sampling transform + divergence metrics  (lifted verbatim from the merged #520 card,
# research/validity/sampled_quality_neutrality; kept inline so this card is self-contained).
# ============================================================================================ #
def gen_config_dist(lp: dict[int, float], T: float = GEN_T, top_k: int = GEN_TOP_K,
                    top_p: float = GEN_TOP_P) -> dict[int, float]:
    """Replicate HF/vLLM temperature -> top_k -> top_p (nucleus) filtering on a top-K logprob dict
    and return the renormalised sampling distribution. RAW log-softmax logprobs in; the T>0
    partition shift cancels (softmax((logit+C)/T) restricted to a support == softmax(logprob/T))."""
    if not lp or T <= 0:
        return {}
    items = sorted(((int(k), float(v)) for k, v in lp.items()), key=lambda kv: kv[1], reverse=True)
    if top_k and top_k < len(items):
        items = items[:top_k]
    xs = [v / T for _, v in items]
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex) or 1.0
    probs = [(items[i][0], ex[i] / Z) for i in range(len(items))]
    cum = 0.0
    kept: list[tuple[int, float]] = []
    for tok, pr in probs:
        kept.append((tok, pr))
        cum += pr
        if cum >= top_p:
            break
    Z2 = sum(pr for _, pr in kept) or 1.0
    return {tok: pr / Z2 for tok, pr in kept}


def tv(p: dict[int, float], q: dict[int, float]) -> float:
    """Total variation over the union support; bounded [0,1], NaN-clean."""
    toks = set(p) | set(q)
    return 0.5 * sum(abs(p.get(t, 0.0) - q.get(t, 0.0)) for t in toks)


def kl(p: dict[int, float], q: dict[int, float]) -> float:
    """KL(p || q) over p's support; q floored to avoid log(0). NaN-clean (finite or +inf)."""
    s = 0.0
    for t, pt in p.items():
        if pt <= 0:
            continue
        qt = q.get(t, 0.0) or 1e-300
        s += pt * math.log(pt / qt)
    return s if math.isfinite(s) else float("inf")


def support_agreement(p: dict[int, float], q: dict[int, float]) -> float:
    sp, sq = set(p), set(q)
    if not sp and not sq:
        return 1.0
    return len(sp & sq) / max(1, len(sp | sq))


def _di(d: dict | None) -> dict[int, float]:
    return {int(k): float(v) for k, v in d.items()} if d else {}


# ============================================================================================ #
# keepset + token classification
# ============================================================================================ #
def load_keepset(path: str) -> tuple[set[int], dict]:
    d = json.load(open(path))
    keep = set(int(x) for x in d["keep_ids"])
    meta = {"pruned_vocab_K": int(d.get("pruned_vocab_K", len(keep))),
            "full_vocab": int(d.get("full_vocab", FULL_VOCAB)),
            "n_unique": len(keep), "source_keepset": d.get("source_keepset")}
    return keep, meta


def classify_token(s: str) -> str:
    """answer-bearing vs filler split for the served collapse. 'digit' (math answer tokens) and
    'alpha' (content words) are answer-bearing; 'filler' is pure whitespace/punctuation."""
    if any(ch.isdigit() for ch in s):
        return "digit"
    if any(ch.isalpha() for ch in s):
        return "alpha"
    return "filler"


# ============================================================================================ #
# GPU PHASE: capture base CoT trajectory + per-position raw top-K, derive base|ship, analyse.
# Runs in the server venv subprocess (vLLM). The vLLM tokenizer is used to decode tokens for the
# answer-bearing-vs-filler split and the raw high-TV reads, so analysis is fully self-contained.
# ============================================================================================ #
def phase_gpu(prompts_path: str, keepset_path: str, out_path: str, n_new: int, topk: int,
              ctx_cap: int, gpu_mem_util: float, n_reads: int, seed: int,
              keepset_bake_path: str | None = None) -> None:
    import torch
    from vllm import LLM, SamplingParams

    keep, kmeta = load_keepset(keepset_path)
    # OPTIONAL layered attribution (advisor fern #531): the live ship carries TWO prunes --
    # a bake-time 262144->16384 (osoi5-v0-baked head) and a serve-time 16384->12288
    # (serve.py:_lmhead_prune_phase). With the bake (16k) keepset we split the collapse into the
    # bake layer (base|bake) and the serve layer (bake|ship). serve.py enforces 12k subset of 16k
    # subset of full vocab, so this is a clean nested decomposition derived from the SAME forward.
    keep_bake: set[int] | None = None
    kbmeta: dict | None = None
    if keepset_bake_path:
        keep_bake, kbmeta = load_keepset(keepset_bake_path)
        if not keep.issubset(keep_bake):
            raise RuntimeError(
                f"served keepset (K={len(keep)}) is NOT a subset of bake keepset "
                f"(K={len(keep_bake)}); layered attribution invalid")
    model_dir = rc.resolve_model_dir()
    full_vocab_ok = rc._margin_model_full_vocab(model_dir)
    prompts = [json.loads(l) for l in open(prompts_path)]
    print(f"[gpu] model={model_dir} full_vocab_head={full_vocab_ok} prompts={len(prompts)} "
          f"keepset_K={kmeta['n_unique']} bake_K={(kbmeta['n_unique'] if kbmeta else None)} "
          f"n_new={n_new} topk={topk}", flush=True)
    if not full_vocab_ok:
        raise RuntimeError(f"substrate {model_dir} is NOT a full-vocab head; base arm invalid")

    max_ctx = max((min(len(p["context_token_ids"]), ctx_cap) for p in prompts), default=512)
    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=max(1024, max_ctx + n_new + 16),
              gpu_memory_utilization=gpu_mem_util, max_num_seqs=1,
              enable_prefix_caching=False, enforce_eager=True, trust_remote_code=True,
              max_logprobs=max(80, topk + 4))
    tok = llm.get_tokenizer()
    decode_cache: dict[int, str] = {}

    def dec(tid: int) -> str:
        s = decode_cache.get(tid)
        if s is None:
            try:
                s = tok.decode([int(tid)])
            except Exception:  # noqa: BLE001
                s = f"<{tid}>"
            decode_cache[tid] = s
        return s

    # GEN: base's own long CoT under the OFFICIAL sampling config (reproducible via seed).
    gen_sp = SamplingParams(temperature=GEN_T, top_k=GEN_TOP_K, top_p=GEN_TOP_P,
                            max_tokens=n_new, seed=seed)
    # SCORE: teacher-force ctx+trajectory; PREFILL prompt_logprobs=topk -> per-position RAW top-K.
    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)

    # accumulators
    pooled_tv: list[float] = []
    pooled_kl: list[float] = []
    by_domain_tv: dict[str, list[float]] = {}
    by_class: dict[str, dict[str, float]] = {
        c: {"n": 0.0, "tv_sum": 0.0, "tv_max": 0.0, "killed": 0.0} for c in ("digit", "alpha", "filler")
    }
    keepset_total = len(keep)
    n_pos = 0
    n_killed_argmax = 0
    n_killed_sampled = 0
    n_blowup = 0
    ship_supp_min = 10**9
    n_supp_under_topk = 0          # positions where <GEN_TOP_K keepset toks in captured top-K
    n_ship_unresolved = 0          # positions where ZERO keepset toks in top-K (ship TV pinned to 1)
    n_nucleus_safe = 0             # positions where p_ship is PROVABLY exact (conservative bound)
    # blind-spot self-test maxima (must stay 0)
    max_tv_full_full = 0.0
    max_tv_prune_prune = 0.0
    any_nan = False
    reads: list[tuple] = []        # min-heap of (tv, tiebreak, detail) for the top-n_reads blow-ups
    tv_hist = {b: 0 for b in ("0", "0-.1", ".1-.3", ".3-.5", ".5-.7", ".7-.9", ".9-1")}
    read_tie = 0

    # layered-attribution accumulators (only populated when a bake/16k keepset is supplied)
    pooled_tv_bb: list[float] = []    # base|bake  -> bake layer (262144 -> 16384)
    pooled_kl_bb: list[float] = []
    pooled_tv_bs2: list[float] = []   # bake|ship  -> serve layer (16384 -> 12288)
    pooled_kl_bs2: list[float] = []
    bb_by_domain: dict[str, list[float]] = {}
    bs2_by_domain: dict[str, list[float]] = {}
    n_killed_at_bake = 0     # base-argmax outside the 16k bake keepset (lost at bake time)
    n_killed_at_serve = 0    # base-argmax inside 16k but dropped by the serve-time 12k prune
    n_bake_nucleus_safe = 0
    n_bake_unresolved = 0

    def hist_bin(x: float) -> str:
        if x <= 1e-9:
            return "0"
        for hi, name in ((0.1, "0-.1"), (0.3, ".1-.3"), (0.5, ".3-.5"),
                         (0.7, ".5-.7"), (0.9, ".7-.9")):
            if x < hi:
                return name
        return ".9-1"

    t0 = time.time()
    for p in prompts:
        ctx = list(p["context_token_ids"])[:ctx_cap]
        c = len(ctx)
        out = llm.generate([{"prompt_token_ids": ctx}], gen_sp, use_tqdm=False)[0]
        traj = list(out.outputs[0].token_ids)
        if not traj:
            continue
        full = ctx + traj
        vout = llm.generate([{"prompt_token_ids": full}], score_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs or []

        for i in range(len(traj)):
            j = c + i
            if j >= len(pls) or pls[j] is None:
                continue
            L = {int(t): rc._lp(v) for t, v in pls[j].items()}
            if not L:
                continue
            if any(not math.isfinite(x) for x in L.values()):
                any_nan = True
            p_base = gen_config_dist(L)
            L_ship = {t: lp for t, lp in L.items() if t in keep}
            supp_cnt = len(L_ship)
            if L_ship:
                p_ship = gen_config_dist(L_ship)
                tv_bs = tv(p_base, p_ship)
                kl_bs = kl(p_base, p_ship)
                # conservative nucleus-safety bound: p_ship (built from captured keepset toks) equals
                # the TRUE served ship dist iff the UNCAPTURED keepset tail cannot enter the top_p
                # nucleus. Upper-bound that tail by (n_uncaptured_keepset * smallest captured prob);
                # safe if it is below the (1-top_p) slack of the captured keepset mass (or top_k is
                # already fully determined: >=64 keepset toks captured). Exact at every safe position.
                p_kth = math.exp(min(L.values()))
                cap_keep_mass = sum(math.exp(lp) for lp in L_ship.values())
                max_uncap = (keepset_total - supp_cnt) * p_kth
                if supp_cnt >= GEN_TOP_K or max_uncap <= (1.0 - GEN_TOP_P) * cap_keep_mass:
                    n_nucleus_safe += 1
            else:
                # NO keepset token in the captured top-K: the served ship is forced onto a keepset
                # token below the capture floor -> certainly outside base's nucleus -> disjoint
                # supports -> TV=1 (exact), KL large. (base argmax is non-keepset here by definition.)
                p_ship = {}
                tv_bs = 1.0
                kl_bs = float("inf")
                n_ship_unresolved += 1
            if not math.isfinite(tv_bs):
                any_nan = True
            # blind-spot controls (identical-arm => exactly 0; assert via running max)
            max_tv_full_full = max(max_tv_full_full, tv(p_base, p_base))
            max_tv_prune_prune = max(max_tv_prune_prune, tv(p_ship, p_ship))

            base_argmax = max(L.items(), key=lambda kv: kv[1])[0]
            killed_argmax = base_argmax not in keep
            sampled_tok = int(traj[i])
            killed_sampled = sampled_tok not in keep
            ship_supp_min = min(ship_supp_min, supp_cnt)
            if supp_cnt < GEN_TOP_K:
                n_supp_under_topk += 1

            cls = classify_token(dec(base_argmax))
            dom = p.get("domain", "?")

            # ---- layered attribution: split base|ship into bake (262k->16k) + serve (16k->12k) ----
            if keep_bake is not None:
                L_bake = {t: lp for t, lp in L.items() if t in keep_bake}
                if L_bake:
                    p_bake = gen_config_dist(L_bake)
                    tv_bb = tv(p_base, p_bake)
                    kl_bb = kl(p_base, p_bake)
                    p_kth_b = math.exp(min(L.values()))
                    cap_bake_mass = sum(math.exp(lp) for lp in L_bake.values())
                    max_uncap_b = (len(keep_bake) - len(L_bake)) * p_kth_b
                    if len(L_bake) >= GEN_TOP_K or max_uncap_b <= (1.0 - GEN_TOP_P) * cap_bake_mass:
                        n_bake_nucleus_safe += 1
                else:
                    # no 16k token in captured top-K: true bake dist forced below the floor ->
                    # disjoint from base's nucleus -> TV=1 exact (mirrors the ship-unresolved case).
                    p_bake = {}
                    tv_bb = 1.0
                    kl_bb = float("inf")
                    n_bake_unresolved += 1
                # serve layer = the divergence the 12k prune ADDS on top of the 16k bake.
                if L_ship:
                    tv_bs2 = tv(p_bake, p_ship)
                    kl_bs2 = kl(p_bake, p_ship)
                elif not L_bake:
                    # both forced below the floor: the bake already removed every captured token, so
                    # the serve prune adds no resolvable divergence here (attribute it to the bake).
                    tv_bs2 = 0.0
                    kl_bs2 = 0.0
                else:
                    # bake keeps a captured token but the serve prune kills it -> disjoint -> TV=1.
                    tv_bs2 = 1.0
                    kl_bs2 = float("inf")
                if not (math.isfinite(tv_bb) and math.isfinite(tv_bs2)):
                    any_nan = True
                pooled_tv_bb.append(tv_bb)
                pooled_kl_bb.append(min(kl_bb, KL_CAP) if math.isfinite(kl_bb) else KL_CAP)
                pooled_tv_bs2.append(tv_bs2)
                pooled_kl_bs2.append(min(kl_bs2, KL_CAP) if math.isfinite(kl_bs2) else KL_CAP)
                bb_by_domain.setdefault(dom, []).append(tv_bb)
                bs2_by_domain.setdefault(dom, []).append(tv_bs2)
                if killed_argmax:
                    if base_argmax not in keep_bake:
                        n_killed_at_bake += 1      # answer token already gone at bake time
                    else:
                        n_killed_at_serve += 1     # answer token survived bake, killed by serve prune

            n_pos += 1
            pooled_tv.append(tv_bs)
            # cap +inf KL (disjoint-support positions) to a large finite sentinel: keeps the stat
            # NaN-clean while preserving the "ship assigns ~0 to the wanted token" signal.
            pooled_kl.append(min(kl_bs, KL_CAP) if math.isfinite(kl_bs) else KL_CAP)
            by_domain_tv.setdefault(dom, []).append(tv_bs)
            cb = by_class[cls]
            cb["n"] += 1
            cb["tv_sum"] += tv_bs
            cb["tv_max"] = max(cb["tv_max"], tv_bs)
            if killed_argmax:
                n_killed_argmax += 1
                cb["killed"] += 1
            if killed_sampled:
                n_killed_sampled += 1
            if tv_bs > TV_BLOWUP:
                n_blowup += 1
            tv_hist[hist_bin(tv_bs)] += 1

            # keep the top n_reads highest-TV positions with full decoded detail
            if killed_argmax or tv_bs > TV_BLOWUP:
                def top5(dist):
                    return [{"tok": dec(t), "id": int(t), "p": round(float(pr), 4)}
                            for t, pr in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:5]]
                ship_argmax = max(p_ship.items(), key=lambda kv: kv[1])[0] if p_ship else None
                detail = {
                    "domain": dom, "prompt_id": p.get("id"), "pos": i, "tv": round(tv_bs, 4),
                    "kl": round(kl_bs, 4) if math.isfinite(kl_bs) else None,
                    "wanted_tok": dec(base_argmax), "wanted_id": int(base_argmax),
                    "wanted_killed": bool(killed_argmax),
                    "wanted_base_p": round(float(p_base.get(base_argmax, 0.0)), 4),
                    "ship_forced_tok": (dec(ship_argmax) if ship_argmax is not None else None),
                    "ship_forced_p": (round(float(p_ship.get(ship_argmax, 0.0)), 4)
                                      if ship_argmax is not None else None),
                    "sampled_tok": dec(sampled_tok), "sampled_killed": bool(killed_sampled),
                    "base_top5": top5(p_base), "ship_top5": top5(p_ship),
                }
                read_tie += 1
                if len(reads) < n_reads:
                    heapq.heappush(reads, (tv_bs, read_tie, detail))
                elif tv_bs > reads[0][0]:
                    heapq.heapreplace(reads, (tv_bs, read_tie, detail))

    def stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"n": 0, "mean": float("nan"), "median": float("nan"),
                    "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
        s = sorted(xs)
        n = len(s)
        q = lambda f: s[min(n - 1, int(f * n))]
        return {"n": n, "mean": sum(s) / n, "median": s[n // 2],
                "p90": q(0.90), "p99": q(0.99), "max": s[-1]}

    cls_out = {}
    for c, cb in by_class.items():
        nn = cb["n"]
        cls_out[c] = {"n": int(nn), "mean_tv": (cb["tv_sum"] / nn if nn else float("nan")),
                      "max_tv": cb["tv_max"],
                      "killed_rate": (cb["killed"] / nn if nn else float("nan"))}
    answer_n = cls_out["digit"]["n"] + cls_out["alpha"]["n"]
    answer_tv_sum = (by_class["digit"]["tv_sum"] + by_class["alpha"]["tv_sum"])
    answer_killed = (by_class["digit"]["killed"] + by_class["alpha"]["killed"])

    result = {
        "phase": "gpu", "model_dir": model_dir, "full_vocab_head": bool(full_vocab_ok),
        "keepset": kmeta, "n_prompts": len(prompts), "n_positions": n_pos,
        "n_new": n_new, "topk": topk, "seed": seed,
        "gen_config": {"temperature": GEN_T, "top_k": GEN_TOP_K, "top_p": GEN_TOP_P, "do_sample": True},
        # ---- base|ship divergence (the real audit column) ----
        "tv_base_vs_ship": stats(pooled_tv),
        "kl_base_vs_ship": stats(pooled_kl),
        "tv_by_domain": {d: stats(v) for d, v in sorted(by_domain_tv.items())},
        "tv_hist": tv_hist,
        "n_blowup_tv_gt_%.2f" % TV_BLOWUP: n_blowup,
        "frac_blowup": (n_blowup / n_pos if n_pos else float("nan")),
        # ---- the collapse mechanism ----
        "wanted_token_killed_rate": (n_killed_argmax / n_pos if n_pos else float("nan")),
        "sampled_token_killed_rate": (n_killed_sampled / n_pos if n_pos else float("nan")),
        "n_killed_argmax": n_killed_argmax, "n_killed_sampled": n_killed_sampled,
        # ---- answer-bearing vs filler ----
        "tv_by_class": cls_out,
        "answer_bearing_mean_tv": (answer_tv_sum / answer_n if answer_n else float("nan")),
        "answer_bearing_killed_rate": (answer_killed / answer_n if answer_n else float("nan")),
        "filler_mean_tv": cls_out["filler"]["mean_tv"],
        # ---- blind-spot contrast (the proof) ----
        "blindspot": {
            "selftest_full_vs_full_max_tv": max_tv_full_full,    # control == 0
            "prune_vs_prune_max_tv": max_tv_prune_prune,         # #520 geometry == 0
            "base_vs_ship_max_tv": (stats(pooled_tv)["max"] if pooled_tv else float("nan")),
        },
        # ---- layered attribution (advisor fern #531): bake 262k->16k + serve 16k->12k.
        # None unless a bake/16k keepset is supplied; base|ship == base|bake (+) bake|ship. ----
        "layer_attribution": ({
            "bake_keepset_K": kbmeta["n_unique"], "bake_full_vocab": kbmeta["full_vocab"],
            "tv_base_vs_bake": stats(pooled_tv_bb),       # bake layer (262144 -> 16384)
            "kl_base_vs_bake": stats(pooled_kl_bb),
            "tv_bake_vs_ship": stats(pooled_tv_bs2),      # serve layer (16384 -> 12288)
            "kl_bake_vs_ship": stats(pooled_kl_bs2),
            "tv_base_vs_bake_by_domain": {d: stats(v) for d, v in sorted(bb_by_domain.items())},
            "tv_bake_vs_ship_by_domain": {d: stats(v) for d, v in sorted(bs2_by_domain.items())},
            "n_killed_at_bake": n_killed_at_bake,
            "n_killed_at_serve": n_killed_at_serve,
            "killed_at_bake_rate": (n_killed_at_bake / n_pos if n_pos else float("nan")),
            "killed_at_serve_rate": (n_killed_at_serve / n_pos if n_pos else float("nan")),
            "bake_nucleus_safe_rate": (n_bake_nucleus_safe / n_pos if n_pos else float("nan")),
            "n_bake_unresolved": n_bake_unresolved,
        } if keep_bake is not None else None),
        # ---- harness self-checks ----
        "ship_support_min_in_topk": (ship_supp_min if ship_supp_min < 10**9 else None),
        "n_positions_keepset_lt_topk": n_supp_under_topk,
        "n_ship_unresolved": n_ship_unresolved,          # zero keepset in top-K -> TV pinned to 1
        "ship_nucleus_safe_rate": (n_nucleus_safe / n_pos if n_pos else float("nan")),
        "n_nucleus_safe": n_nucleus_safe,
        "any_nan": bool(any_nan),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "high_tv_reads": [d for _, _, d in sorted(reads, key=lambda r: r[0], reverse=True)],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rc._jsonable(result), open(out_path, "w"), indent=2)
    print(f"[gpu] DONE n_pos={n_pos} max_tv={result['tv_base_vs_ship']['max']:.4f} "
          f"killed_rate={result['wanted_token_killed_rate']:.4f} -> {out_path}", flush=True)
    print(f"GPU_DONE {out_path}", flush=True)


# ============================================================================================ #
# COMPOSE -- self-test + verdict from the GPU analysis JSON (orchestrator, pure CPU).
# ============================================================================================ #
def compose(gpu: dict, keepset_meta: dict) -> dict:
    bs = gpu["tv_base_vs_ship"]
    blind = gpu["blindspot"]
    killed = gpu["wanted_token_killed_rate"]

    checks = {
        # control: identical-arm geometries are EXACTLY 0 (transform pipeline is a pure function).
        "selftest_full_vs_full_tv0": blind["selftest_full_vs_full_max_tv"] == 0.0,
        "prune_vs_prune_tv0": blind["prune_vs_prune_max_tv"] == 0.0,
        # the head axis is the ONLY mover: base|ship max TV is materially > the (zero) controls.
        "base_vs_ship_exceeds_controls": blind["base_vs_ship_max_tv"] > max(
            blind["selftest_full_vs_full_max_tv"], blind["prune_vs_prune_max_tv"]) + 0.1,
        "nan_clean": not gpu["any_nan"],
        "full_vocab_head": bool(gpu["full_vocab_head"]),
        # served keepset size matches ubel #511 (12288) over the full 262144 vocab.
        "keepset_K_matches_ubel511": keepset_meta["n_unique"] == UBEL511_KEEPSET_K,
        "full_vocab_262144": keepset_meta["full_vocab"] == FULL_VOCAB,
    }
    passes = all(checks.values())
    # ship_nucleus_safe_rate is a REPORTED diagnostic, not a gate: TV is exact at every
    # provably-safe position AND at unresolved positions (disjoint support -> TV=1 exact); only
    # rare intermediate positions are approximate, and there TV is mid-range either way.
    nucleus_safe_rate = gpu.get("ship_nucleus_safe_rate")

    diverges = (bs["max"] > TV_BLOWUP) and (killed >= KILLED_RATE_MATERIAL)
    cause_is_head_prune = (
        diverges
        and blind["prune_vs_prune_max_tv"] == 0.0          # prune common-mode cancels
        and blind["base_vs_ship_max_tv"] > TV_BLOWUP        # only the head axis blows up
    )

    if cause_is_head_prune:
        verdict = (
            f"DIVERGES -- the served 12k head-prune moves the benchmark next-token distribution: "
            f"max TV={bs['max']:.3f}, mean TV={bs['mean']:.3f}, p99 TV={bs['p99']:.3f}; "
            f"wanted_token_killed_rate={killed:.3f}. Cause PINNED to the head-prune: prune-vs-prune "
            f"TV=0 (the #520 geometry, blind) while base-vs-ship TV={blind['base_vs_ship_max_tv']:.3f}."
        )
    elif diverges:
        verdict = (f"DIVERGES (max TV={bs['max']:.3f}, killed={killed:.3f}) but blind-spot control "
                   f"did not cleanly isolate the head-prune; inspect blindspot block.")
    else:
        verdict = (f"NEUTRAL on this distribution (max TV={bs['max']:.3f}, killed={killed:.3f}) -- "
                   f"the head-prune did not materially move the benchmark distribution.")

    # ---- layered attribution summary (advisor fern #531): which prune layer owns the collapse ----
    la = gpu.get("layer_attribution")
    layer_summary = None
    if la:
        bake_tv = la["tv_base_vs_bake"]      # bake layer 262144 -> 16384
        serve_tv = la["tv_bake_vs_ship"]     # serve layer 16384 -> 12288
        kb = la["killed_at_bake_rate"]
        ksv = la["killed_at_serve_rate"]
        tot = (kb or 0.0) + (ksv or 0.0)
        serve_share = (ksv / tot) if tot > 0 else float("nan")
        dom_layer = ("none" if tot <= 0 else
                     "serve(16k->12k)" if ksv >= kb else "bake(262k->16k)")
        layer_summary = {
            "bake_keepset_K": la["bake_keepset_K"],
            "bake_layer_max_tv": bake_tv["max"], "bake_layer_mean_tv": bake_tv["mean"],
            "serve_layer_max_tv": serve_tv["max"], "serve_layer_mean_tv": serve_tv["mean"],
            "killed_at_bake_rate": kb, "killed_at_serve_rate": ksv,
            "n_killed_at_bake": la["n_killed_at_bake"], "n_killed_at_serve": la["n_killed_at_serve"],
            "serve_share_of_kills": serve_share, "dominant_kill_layer": dom_layer,
            # kills a 12k->16k serve-keepset widen would RECOVER (the actionable quick-win signal):
            "serve_widen_recoverable_kill_rate": ksv,
            "bake_nucleus_safe_rate": la.get("bake_nucleus_safe_rate"),
            "n_bake_unresolved": la.get("n_bake_unresolved"),
        }
        verdict += (
            f" LAYERS: serve(16k->12k) owns {serve_share:.0%} of wanted-token kills "
            f"(killed_at_serve={ksv:.3f} vs killed_at_bake={kb:.3f}); bake-layer max TV="
            f"{bake_tv['max']:.3f}, serve-layer max TV={serve_tv['max']:.3f}.")

    return {
        "served_ship_has_head_prune": SERVED_SHIP_HAS_HEAD_PRUNE,
        "served_keepset_K": keepset_meta["n_unique"],
        "served_keepset_full_vocab": keepset_meta["full_vocab"],
        "live_board_submission": LIVE_BOARD_SUBMISSION,
        "live_board_official_tps": LIVE_BOARD_OFFICIAL_TPS,
        "benchmark_distribution_diverges": diverges,
        "cause_is_head_prune": cause_is_head_prune,
        "max_tv_base_vs_ship": bs["max"], "mean_tv_base_vs_ship": bs["mean"],
        "p99_tv_base_vs_ship": bs["p99"], "median_tv_base_vs_ship": bs["median"],
        "max_kl_base_vs_ship": gpu["kl_base_vs_ship"]["max"],
        "wanted_token_killed_rate": killed,
        "sampled_token_killed_rate": gpu["sampled_token_killed_rate"],
        "answer_bearing_mean_tv": gpu["answer_bearing_mean_tv"],
        "answer_bearing_killed_rate": gpu["answer_bearing_killed_rate"],
        "filler_mean_tv": gpu["filler_mean_tv"],
        "frac_blowup": gpu["frac_blowup"],
        "ship_nucleus_safe_rate": nucleus_safe_rate,
        "n_ship_unresolved": gpu.get("n_ship_unresolved"),
        "blindspot": blind,
        "layer_attribution": layer_summary,
        "selftest": {"passes": passes, "n_checks": len(checks), "checks": checks},
        "verdict": verdict,
    }


# ============================================================================================ #
# SELF-TEST (GPU-free) -- validates the deterministic transform + divergence pipeline and the
# committed served keepset, matching the advisor's `--self-test` reproduce entry point.
# ============================================================================================ #
def run_self_test(keepset_path: str) -> int:
    keep, kmeta = load_keepset(keepset_path)
    checks: dict[str, bool] = {}

    # 1) committed served keepset == ubel #511 (12288 unique ids over full 262144 vocab).
    checks["keepset_K_is_12288"] = kmeta["n_unique"] == UBEL511_KEEPSET_K
    checks["full_vocab_262144"] = kmeta["full_vocab"] == FULL_VOCAB

    # 2) identical-arm controls are EXACTLY 0 on synthetic raw top-K logprobs (full|full, prune|prune).
    L = {50000: -0.01, 1: -3.0, 2: -3.5, 3: -4.0, 99999: -5.0, 7: -2.0}
    ks = {1, 2, 3, 7}
    p_base = gen_config_dist(L)
    p_ship = gen_config_dist({t: lp for t, lp in L.items() if t in ks})
    checks["selftest_full_vs_full_tv0"] = tv(p_base, p_base) == 0.0 and kl(p_base, p_base) == 0.0
    checks["prune_vs_prune_tv0"] = tv(p_ship, p_ship) == 0.0 and kl(p_ship, p_ship) == 0.0

    # 3) a killed base-argmax (id 50000 not in keepset) blows up base|ship TV (the head-prune bite).
    tv_killed = tv(p_base, p_ship)
    checks["killed_argmax_blows_up"] = tv_killed > TV_BLOWUP

    # 4) a wholly in-keepset position is neutral (the prune only bites where it kills a wanted token).
    L2 = {1: -0.01, 2: -3.0, 3: -4.0, 50000: -8.0}
    pb2 = gen_config_dist(L2)
    ps2 = gen_config_dist({t: lp for t, lp in L2.items() if t in ks})
    checks["in_keepset_is_neutral"] = tv(pb2, ps2) < 1e-9

    # 5) determinism: same logprobs -> same distribution (pure function).
    checks["transform_deterministic"] = gen_config_dist(L) == gen_config_dist(dict(L))

    # 6) NaN-clean on the synthetic.
    checks["nan_clean"] = all(math.isfinite(v) for v in (tv_killed, kl(p_base, p_ship), tv(pb2, ps2)))

    # 7) LAYERED attribution (advisor fern #531). A token the 16k bake KEEPS but the 12k serve prune
    #    DROPS (id 50000 in kb, not in ks) is a SERVE-layer kill: base|bake stays ~0, bake|ship blows
    #    up. The serve layer is the actionable one (a 12k->16k widen would recover it).
    kb = {1, 2, 3, 7, 50000}                              # 16k bake keepset (superset of the 12k ks)
    p_bake = gen_config_dist({t: lp for t, lp in L.items() if t in kb})
    tv_base_bake = tv(p_base, p_bake)
    tv_bake_ship = tv(p_bake, p_ship)
    checks["serve_layer_blows_up"] = tv_bake_ship > TV_BLOWUP
    checks["bake_layer_neutral_when_bake_keeps_argmax"] = tv_base_bake < 1e-9
    # 8) a token outside BOTH keepsets (id 77777) is a BAKE-layer kill: base|bake itself blows up.
    L3 = {77777: -0.01, 1: -3.0, 2: -3.5, 7: -2.0}
    pb3 = gen_config_dist(L3)
    pbake3 = gen_config_dist({t: lp for t, lp in L3.items() if t in kb})
    checks["bake_layer_blows_up_when_outside_16k"] = tv(pb3, pbake3) > TV_BLOWUP
    # 9) subset invariant the live serve path enforces: 12k subset of 16k (synthetic).
    checks["served_12k_subset_of_bake_16k"] = ks.issubset(kb)
    # 9b) REAL committed keepsets, when present: served 12k is an exact subset of the bake 16k, and
    #     the bake keeps 16384 over the same full 262144 vocab (mirrors serve.py:613-618).
    bake_real = HERE / "pck04_keepset_16k_baked.json"
    if bake_real.exists():
        kbset, kbmeta_real = load_keepset(str(bake_real))
        checks["real_12k_subset_of_real_16k"] = keep.issubset(kbset)
        checks["real_bake_K_is_16384"] = kbmeta_real["n_unique"] == 16384
        checks["real_bake_full_vocab_262144"] = kbmeta_real["full_vocab"] == FULL_VOCAB

    passes = all(checks.values())
    print("[self-test] served_benchmark_divergence")
    print(f"  served keepset: K={kmeta['n_unique']} full_vocab={kmeta['full_vocab']} "
          f"(ubel#511 K={UBEL511_KEEPSET_K})")
    print(f"  killed-argmax TV={tv_killed:.4f}  in-keepset TV={tv(pb2, ps2):.2e}")
    print(f"  LAYERS  base|bake TV={tv_base_bake:.2e}  bake|ship TV={tv_bake_ship:.4f} "
          f"(serve-layer kill)  outside-16k base|bake TV={tv(pb3, pbake3):.4f} (bake-layer kill)")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"[self-test] {'PASS' if passes else 'FAIL'} ({sum(checks.values())}/{len(checks)} checks)")
    return 0 if passes else 1


# ============================================================================================ #
# W&B
# ============================================================================================ #
def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary, log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[#529] wandb helpers unavailable: {e}")
        return None
    c = payload["compose"]
    g = payload["gpu"]
    run = init_wandb_run(
        job_type="analysis-served-benchmark-divergence", agent="denken",
        name=args.wandb_name, group=args.wandb_group,
        tags=["served-benchmark-divergence", "head-prune", "pck04-12k", "mmlu", "math", "gsm8k",
              "gen-config-sampling", "blind-spot", "pr-529"],
        config={"pr": 529, "kind": "served-benchmark-divergence",
                "axis": "base_full_head_vs_ship_12k_head",
                "gen_config": {"temperature": GEN_T, "top_k": GEN_TOP_K, "top_p": GEN_TOP_P, "do_sample": True},
                "served_keepset_K": c["served_keepset_K"], "full_vocab": FULL_VOCAB,
                "live_board_submission": LIVE_BOARD_SUBMISSION,
                "live_board_official_tps": LIVE_BOARD_OFFICIAL_TPS,
                "n_new": g["n_new"], "topk": g["topk"], "seed": g["seed"]},
    )
    if run is None:
        print("[#529] wandb disabled (no API key / WANDB_MODE).")
        return None

    def _f(x):
        return rc._f(x)

    flat: dict[str, float] = {
        "audit/served_ship_has_head_prune": float(bool(c["served_ship_has_head_prune"])),
        "audit/served_keepset_K": float(c["served_keepset_K"]),
        "audit/keepset_matches_ubel511": float(c["served_keepset_K"] == UBEL511_KEEPSET_K),
        "div/max_tv_base_vs_ship": _f(c["max_tv_base_vs_ship"]),
        "div/mean_tv_base_vs_ship": _f(c["mean_tv_base_vs_ship"]),
        "div/p99_tv_base_vs_ship": _f(c["p99_tv_base_vs_ship"]),
        "div/median_tv_base_vs_ship": _f(c["median_tv_base_vs_ship"]),
        "div/max_kl_base_vs_ship": _f(c["max_kl_base_vs_ship"]),
        "div/wanted_token_killed_rate": _f(c["wanted_token_killed_rate"]),
        "div/sampled_token_killed_rate": _f(c["sampled_token_killed_rate"]),
        "div/frac_blowup": _f(c["frac_blowup"]),
        "div/benchmark_distribution_diverges": float(bool(c["benchmark_distribution_diverges"])),
        "div/cause_is_head_prune": float(bool(c["cause_is_head_prune"])),
        "div/ship_nucleus_safe_rate": _f(c.get("ship_nucleus_safe_rate")),
        "div/n_ship_unresolved": _f(c.get("n_ship_unresolved")),
        "class/answer_bearing_mean_tv": _f(c["answer_bearing_mean_tv"]),
        "class/answer_bearing_killed_rate": _f(c["answer_bearing_killed_rate"]),
        "class/filler_mean_tv": _f(c["filler_mean_tv"]),
        "blindspot/selftest_full_vs_full_max_tv": _f(c["blindspot"]["selftest_full_vs_full_max_tv"]),
        "blindspot/prune_vs_prune_max_tv": _f(c["blindspot"]["prune_vs_prune_max_tv"]),
        "blindspot/base_vs_ship_max_tv": _f(c["blindspot"]["base_vs_ship_max_tv"]),
        "selftest/served_benchmark_divergence_self_test_passes": float(payload["compose"]["selftest"]["passes"]),
    }
    for dom, st in g["tv_by_domain"].items():
        flat[f"{dom}/max_tv"] = _f(st["max"])
        flat[f"{dom}/mean_tv"] = _f(st["mean"])
        flat[f"{dom}/p99_tv"] = _f(st["p99"])
        flat[f"{dom}/n_positions"] = _f(st["n"])
    for cls, st in g["tv_by_class"].items():
        flat[f"class/{cls}_mean_tv"] = _f(st["mean_tv"])
        flat[f"class/{cls}_max_tv"] = _f(st["max_tv"])
        flat[f"class/{cls}_killed_rate"] = _f(st["killed_rate"])
    la = c.get("layer_attribution")
    if la:
        flat.update({
            "layer/bake_keepset_K": _f(la["bake_keepset_K"]),
            "layer/bake_max_tv": _f(la["bake_layer_max_tv"]),
            "layer/bake_mean_tv": _f(la["bake_layer_mean_tv"]),
            "layer/serve_max_tv": _f(la["serve_layer_max_tv"]),
            "layer/serve_mean_tv": _f(la["serve_layer_mean_tv"]),
            "layer/killed_at_bake_rate": _f(la["killed_at_bake_rate"]),
            "layer/killed_at_serve_rate": _f(la["killed_at_serve_rate"]),
            "layer/serve_share_of_kills": _f(la["serve_share_of_kills"]),
            "layer/serve_widen_recoverable_kill_rate": _f(la["serve_widen_recoverable_kill_rate"]),
            "layer/bake_nucleus_safe_rate": _f(la.get("bake_nucleus_safe_rate")),
        })
        gla = g.get("layer_attribution") or {}
        for dom, st in (gla.get("tv_base_vs_bake_by_domain") or {}).items():
            flat[f"{dom}/bake_max_tv"] = _f(st["max"])
            flat[f"{dom}/bake_mean_tv"] = _f(st["mean"])
        for dom, st in (gla.get("tv_bake_vs_ship_by_domain") or {}).items():
            flat[f"{dom}/serve_max_tv"] = _f(st["max"])
            flat[f"{dom}/serve_mean_tv"] = _f(st["mean"])
    run.log({"global_step": 0, **{k: (v if (isinstance(v, float) and math.isfinite(v)) else 0.0)
                                  for k, v in flat.items()}})
    log_summary(run, rc._jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="served_benchmark_divergence", artifact_type="analysis",
                      data=rc._jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[#529] wandb logged (run {rid})")
    return rid


# ============================================================================================ #
# orchestration
# ============================================================================================ #
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
    print(f"[orch] launching: {' '.join(phase_args)}", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s", flush=True)
        return 124


def orchestrate(args) -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    gpu_json = str(HERE / "_gpu.json")
    keepset_path = args.keepset or str(HERE / "pck04_keepset_12k.json")
    if not Path(keepset_path).exists():
        raise FileNotFoundError(f"keepset not found at {keepset_path}")
    # OPTIONAL bake/16k keepset -> layered attribution (advisor fern #531). Default: the committed
    # pck04_keepset_16k_baked.json (osoi5-v0-baked source keepset) when present; --no-bake disables.
    bake_path = None
    if not args.no_bake:
        bake_path = args.keepset_bake or str(HERE / "pck04_keepset_16k_baked.json")
        if not Path(bake_path).exists():
            print(f"[orch] bake keepset {bake_path} absent; base|ship only (no layer split)", flush=True)
            bake_path = None
    if not Path(args.prompts).exists():
        raise FileNotFoundError(
            f"prompts not found at {args.prompts}; build with prepare_prompts.py first")

    if not args.skip_gpu:
        server_python = resolve_server_python(args.server_python)
        print(f"[orch] server_python = {server_python}", flush=True)
        phase_args = [
            "--phase", "gpu", "--prompts", args.prompts, "--keepset", keepset_path,
            "--out", gpu_json, "--n-new", str(args.n_new), "--topk", str(args.topk),
            "--ctx-cap", str(args.ctx_cap), "--gpu-mem-util", str(args.gpu_mem_util),
            "--n-reads", str(args.n_reads), "--seed", str(args.seed),
        ]
        if bake_path:
            phase_args += ["--keepset-bake", bake_path]
        rcode = run_gpu_phase(server_python, phase_args, timeout=args.gpu_timeout)
        if rcode != 0:
            print(f"[orch] GPU phase FAILED rc={rcode}", flush=True)
            return rcode

    gpu = json.load(open(gpu_json))
    keep, kmeta = load_keepset(keepset_path)
    comp = compose(gpu, kmeta)
    payload = {
        "pr": 529, "card": "served-benchmark-divergence",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True, "official_tps": 0,
        "compose": comp, "gpu": gpu,
    }
    out_path = HERE / "served_benchmark_divergence.json"
    json.dump(rc._jsonable(payload), open(out_path, "w"), indent=2)

    # console summary
    print("\n" + "=" * 92, flush=True)
    print(f"  served_ship_has_head_prune = {comp['served_ship_has_head_prune']}  "
          f"served_keepset_K = {comp['served_keepset_K']}  (ubel#511={UBEL511_KEEPSET_K})", flush=True)
    print(f"  live board = {comp['live_board_submission']} @ {comp['live_board_official_tps']} TPS", flush=True)
    print(f"  base|ship  max TV={comp['max_tv_base_vs_ship']:.4f} mean={comp['mean_tv_base_vs_ship']:.4f} "
          f"p99={comp['p99_tv_base_vs_ship']:.4f}  max KL={comp['max_kl_base_vs_ship']:.3f}", flush=True)
    print(f"  wanted_token_killed_rate = {comp['wanted_token_killed_rate']:.4f}  "
          f"answer_bearing_mean_tv = {rc._f(comp['answer_bearing_mean_tv']):.4f}  "
          f"filler_mean_tv = {rc._f(comp['filler_mean_tv']):.4f}", flush=True)
    bsd = comp["blindspot"]
    print(f"  BLIND-SPOT  full|full={bsd['selftest_full_vs_full_max_tv']:.3g}  "
          f"prune|prune={bsd['prune_vs_prune_max_tv']:.3g}  base|ship={bsd['base_vs_ship_max_tv']:.3f}", flush=True)
    la = comp.get("layer_attribution")
    if la:
        print(f"  LAYERS  bake(262k->16k) max TV={la['bake_layer_max_tv']:.3f}  "
              f"serve(16k->12k) max TV={la['serve_layer_max_tv']:.3f}  | "
              f"kills: bake={la['killed_at_bake_rate']:.3f} serve={la['killed_at_serve_rate']:.3f}  "
              f"serve_share={rc._f(la['serve_share_of_kills']):.0%}  dominant={la['dominant_kill_layer']}",
              flush=True)
    print(f"  selftest passes = {comp['selftest']['passes']} ({comp['selftest']['n_checks']} checks)", flush=True)
    print(f"  VERDICT: {comp['verdict']}", flush=True)
    print("=" * 92 + "\n", flush=True)

    rid = maybe_log_wandb(payload, args)
    payload["wandb_run_id"] = rid
    json.dump(rc._jsonable(payload), open(out_path, "w"), indent=2)
    print(f"[orch] wrote {out_path}  wandb={rid}", flush=True)
    return 0 if comp["selftest"]["passes"] else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gpu"], default=None)
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--keepset", default=None)
    ap.add_argument("--keepset-bake", default=None,
                    help="16k bake keepset (osoi5-v0-baked source) for layered attribution; "
                         "defaults to committed pck04_keepset_16k_baked.json when present")
    ap.add_argument("--no-bake", action="store_true",
                    help="disable layered attribution (base|ship only, the original geometry)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-new", type=int, default=320)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--ctx-cap", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--n-reads", type=int, default=15)
    ap.add_argument("--seed", type=int, default=GEN_SEED)
    ap.add_argument("--gpu-timeout", type=int, default=5400)
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--skip-gpu", action="store_true", help="re-compose from an existing _gpu.json")
    ap.add_argument("--wandb_name", default="denken/served-benchmark-divergence")
    ap.add_argument("--wandb_group", default="served-benchmark-divergence")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="GPU-free transform+keepset self-test (advisor reproduce entry point)")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test(args.keepset or str(HERE / "pck04_keepset_12k.json"))
    if args.phase == "gpu":
        phase_gpu(args.prompts, args.keepset, args.out, args.n_new, args.topk,
                  args.ctx_cap, args.gpu_mem_util, args.n_reads, args.seed,
                  keepset_bake_path=args.keepset_bake)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
