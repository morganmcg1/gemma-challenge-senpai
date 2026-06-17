#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #540 (denken) -- Fast-kernel numerics tax: does the speed stack cost reasoning?

THE QUESTION
------------
fern #535 saw the served fast-kernel ship drop AIME (0.267 -> 0.167, n=30) below the
>=0.240 validity gate. Is that a REAL distributional tax the fast-kernel speed stack
{surgical 2D attention (fa_sliding) + split-KV + PLE fold + bf16 compute} levies on the
native 262144-vocab int4 Gemma head -- a concentrated per-position divergence that could
flip answer tokens -- or is it n=30 sampling noise (fern's own delta=0.10 is 1.24 se,
within noise)? This card answers the WHY/WHERE mechanism OFFLINE: it measures the
per-position next-token divergence between a PLAIN int4 base (Arm P) and a FAST-kernel int4
base (Arm F) on the same benchmark proxy distribution, under the official sampling config.

WHAT THE FAST STACK REDUCES TO ON THE NATIVE HEAD (read from the live submission)
--------------------------------------------------------------------------------
The four levers, audited against submissions/fa2sw_strict_m1ar_int4:
  * bf16 compute   -- both arms load dtype=bfloat16: COMMON-MODE, cancels exactly.
  * PLE fold       -- folds embed_scale_per_layer = sqrt(256) = 16.0 (a power of two) into
                      bf16 weights; x * 16.0 is EXACT in bf16 (no mantissa loss): BIT-IDENTICAL.
                      Not even in the offline native path (we load the native snapshot direct).
  * split-KV       -- a decode-time log-sum-exp KV reduction; mathematically exact up to fp
                      associativity, and DORMANT under teacher-forced prefill scoring (M=1).
  * surgical 2D attn (fa_sliding) -- the ONLY lever that can move logits: it swaps eligible
                      sliding-window (head_size=256) layers from the model's uniform TRITON_ATTN
                      backend to FlashAttention. vLLM forces uniform TRITON_ATTN on this
                      heterogeneous-head-dim (256/512) model precisely to avoid mixed-backend
                      numerical divergence -- which is exactly what fa_sliding re-introduces.

So on the native head the fast-kernel tax == the fa_sliding backend swap. We measure it.

THE AS-RUN FINDING vs THE AS-DESIGNED COUNTERFACTUAL (the blind-spot contrast)
-----------------------------------------------------------------------------
fa_sliding_patch only swaps when hf_config.model_type == "gemma4". The native int4
checkpoint's text config reports "gemma4_text", so the guard NEVER matches and the swap is
silently inert (fa flips = 0). The as-run fast stack is therefore BIT-IDENTICAL to plain on
the native head. To prove that the TV=0 we measure is a real property and NOT a blind probe,
we add a POSITIVE CONTROL arm (F_forced, research/validity/fast_kernel_divergence/
_forced_fa_patch.py) that relaxes the guard to startswith("gemma4") so the swap engages on
the eligible head-256 sliding layers. Four arms, all scoring the SAME base CoT trajectory:
  P        plain int4, default backend            -- generates + scores the base trajectory
  P'       plain int4, re-score                   -- DETERMINISM/NOISE FLOOR (must be TV=0)
  F_asrun  fern recipe (fa_sliding, FA_SLIDING=1) -- the as-run answer (inert -> TV=0)
  F_forced forced FA on eligible sliding layers   -- LATENT as-designed tax (probe sensitivity)
The contrast P|P'=0 and P|F_asrun=0 while P|F_forced>0 proves: the probe can see a tax, and
the as-run stack has none because the swap never fires -- not because we are blind.

KEY OUTPUTS (from the as-run pair P|F_asrun, the question fern actually ran)
---------------------------------------------------------------------------
mean_tv_plain_vs_fast, max_tv, tv_histogram (diffuse|bimodal), kernel_argmax_flip_rate,
answer_vs_filler_concentration, tax_locus (diffuse|concentrated),
kernel_numerics_tax (real-concentrated|real-diffuse|negligible), corroborates_fern_aime
(yes|no), one-line verdict. Plus F_forced reported as the latent as-designed tax. NaN-clean.

SCOPE: LOCAL profiling card. analysis_only=true, official_tps=0, NO served-file change, NO
HF Job, NO train.py --launch, NO submission. The served config + baseline are UNCHANGED; the
distribution is MEASURED, never altered. GPU phase runs under the submission server venv
(vLLM + Marlin); CUDA_VISIBLE_DEVICES=0. Reuses the merged #529 prompts + gen_config / tv / kl
transform and the rc model helpers.

  .venv/bin/python -m research.validity.fast_kernel_divergence.fast_kernel_divergence \
      --prompts research/validity/served_benchmark_divergence/prompts.jsonl \
      --n-new 320 --topk 256 \
      --wandb_name denken/fast-kernel-divergence \
      --wandb_group fast-kernel-divergence
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

# ---- merged census helpers (resolve_model_dir / _lp / _jsonable / _f / full_vocab) ----
_RC_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
if str(_RC_DIR) not in sys.path:
    sys.path.insert(0, str(_RC_DIR))
import reduction_sensitivity_census as rc  # noqa: E402

# ============================================================================================ #
# AUDIT HEADLINE -- read from the live board submission (submissions/fa2sw_strict_m1ar_int4).
# Facts about the fast stack, stamped into the run record. Provenance:
#   fa_sliding_patch.py: swap gated on hf_config.model_type == "gemma4" (native head reports
#       "gemma4_text" -> guard mismatch -> fa flips = 0, swap inert).
#   manifest.json: FA_SLIDING=1, SPLITKV_VERIFY=1, PLE_FOLD_*=1 (embed_scale=sqrt(256)=16.0).
# ============================================================================================ #
LIVE_BOARD_SUBMISSION = "fa2sw_strict_m1ar_int4"
FERN_AIME_BASE = 0.267        # fern #535 plain AIME (n=30)
FERN_AIME_FAST = 0.167        # fern #535 fast-kernel AIME (n=30)
FERN_AIME_GATE = 0.240        # organizer validity gate
FERN_N = 30
FULL_VOCAB = 262144

# official generation_config.json sampling axis (the downstream evals SAMPLE under this).
GEN_T = 1.0
GEN_TOP_K = 64
GEN_TOP_P = 0.95
GEN_SEED = 1234               # fixed -> reproducible base CoT trajectory

# verdict thresholds.
TV_BLOWUP = 0.5              # a position is a "blow-up" (argmax-flippable scale) if TV > this
TV_MATERIAL = 1e-6          # mean/max TV above this (and above the determinism floor) is "real"
FLIP_MATERIAL = 1e-4        # argmax-flip rate above this is a real greedy-token tax
CONCENTRATION_HI = 2.0      # answer-bearing mean TV >= 2x filler mean TV -> "concentrated on answers"
LOCUS_TOP1PCT_SHARE = 0.5   # top 1% of positions hold >= 50% of TV mass -> concentrated locus
KL_CAP = 100.0              # finite sentinel for disjoint-support (+inf) KL positions

# arm wiring. gen=True arm produces the shared base trajectory; others re-score it.
ARMS = ("plain", "plainB", "fast_asrun", "fast_forced")
ARM_LABEL = {
    "plain": "P (plain int4, default backend)",
    "plainB": "P' (plain re-score: determinism floor)",
    "fast_asrun": "F_asrun (fern fa_sliding FA_SLIDING=1; inert by mt guard)",
    "fast_forced": "F_forced (positive control: forced FA on sliding layers)",
}
# the pairs we compose. (numerator base is always plain == the trajectory owner.)
PAIRS = (("plain", "plainB"), ("plain", "fast_asrun"), ("plain", "fast_forced"))


# ============================================================================================ #
# gen_config sampling transform + divergence metrics (lifted verbatim from the merged #529/#520
# card; kept inline so this card is self-contained and --self-test runs standalone).
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


def classify_token(s: str) -> str:
    """answer-bearing vs filler split. 'digit' (math answer tokens) and 'alpha' (content words)
    are answer-bearing; 'filler' is pure whitespace/punctuation."""
    if any(ch.isdigit() for ch in s):
        return "digit"
    if any(ch.isalpha() for ch in s):
        return "alpha"
    return "filler"


def stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    s = sorted(xs)
    n = len(s)
    q = lambda f: s[min(n - 1, int(f * n))]
    return {"n": n, "mean": sum(s) / n, "median": s[n // 2],
            "p90": q(0.90), "p99": q(0.99), "max": s[-1]}


def hist_bin(x: float) -> str:
    if x <= 1e-9:
        return "0"
    for hi, name in ((0.1, "0-.1"), (0.3, ".1-.3"), (0.5, ".3-.5"),
                     (0.7, ".5-.7"), (0.9, ".7-.9")):
        if x < hi:
            return name
    return ".9-1"


HIST_BINS = ("0", "0-.1", ".1-.3", ".3-.5", ".5-.7", ".7-.9", ".9-1")


def locus_of(tvs: list[float]) -> dict[str, Any]:
    """Characterise WHERE the divergence lives: a concentrated/bimodal tax (a few near-1
    blow-ups, e.g. argmax flips) vs a diffuse low-grade numeric tax (small TV everywhere)."""
    n = len(tvs)
    total = sum(tvs)
    s = sorted(tvs, reverse=True)
    ktop = max(1, int(math.ceil(0.01 * n)))
    top1pct_share = (sum(s[:ktop]) / total) if total > 1e-12 else 0.0
    n_high = sum(1 for x in tvs if x > TV_BLOWUP)
    n_mid = sum(1 for x in tvs if 0.1 <= x <= 0.9)
    n_low = sum(1 for x in tvs if x < 0.05)
    frac_high = n_high / n if n else 0.0
    frac_mid = n_mid / n if n else 0.0
    frac_low = n_low / n if n else 0.0
    # bimodal: mass piles at the extremes (some near 1, most near 0) with an empty middle.
    bimodal = (n_high > 0) and (frac_low > 0.5) and (frac_mid < 0.2)
    concentrated = (top1pct_share >= LOCUS_TOP1PCT_SHARE and n_high > 0) or bimodal
    return {
        "top1pct_share": top1pct_share, "n_high_gt_%.2f" % TV_BLOWUP: n_high,
        "frac_high": frac_high, "frac_mid": frac_mid, "frac_low": frac_low,
        "histogram_shape": ("bimodal" if bimodal else "diffuse"),
        "tax_locus": ("concentrated" if concentrated else "diffuse"),
    }


# ============================================================================================ #
# GPU PHASE: ONE arm. Apply the arm's patch (if any) BEFORE importing vLLM, then capture
# per-position raw top-K logprobs along the shared base CoT trajectory. The "plain" arm also
# GENERATES the trajectory (seed) and classifies each base-argmax token (tokenizer available
# here) so the CPU compose can split answer-bearing vs filler without a tokenizer.
# ============================================================================================ #
def phase_arm(arm: str, prompts_path: str, traj_in_path: str | None, out_path: str,
              n_new: int, topk: int, ctx_cap: int, gpu_mem_util: float, seed: int) -> None:
    fa_flips = None
    fa_eligible_mismatch = None
    patch_mod = None
    if arm == "fast_asrun":
        sub = str(ROOT / "submissions" / "fa2sw_strict_m1ar_int4")
        sys.path.insert(0, sub)
        os.environ["FA_SLIDING"] = "1"
        os.environ.setdefault("FA_SLIDING_DIAG", "90")
        import fa_sliding_patch as patch_mod  # noqa: F401
        print(f"[arm:{arm}] imported fa_sliding_patch from {sub}", flush=True)
    elif arm == "fast_forced":
        sys.path.insert(0, str(HERE))
        os.environ["FORCE_FA_SLIDING"] = "1"
        os.environ.setdefault("FA_SLIDING_DIAG", "90")
        import _forced_fa_patch as patch_mod  # noqa: F401
        print(f"[arm:{arm}] imported _forced_fa_patch (positive control)", flush=True)

    import torch
    from vllm import LLM, SamplingParams

    model_dir = rc.resolve_model_dir()
    full_vocab_ok = rc._margin_model_full_vocab(model_dir)
    if not full_vocab_ok:
        raise RuntimeError(f"substrate {model_dir} is NOT a full-vocab head; base arm invalid")
    prompts = [json.loads(l) for l in open(prompts_path)]
    backend_env = os.environ.get("VLLM_ATTENTION_BACKEND", "<unset/default>")
    print(f"[arm:{arm}] model={model_dir} full_vocab={full_vocab_ok} prompts={len(prompts)} "
          f"backend_env={backend_env} n_new={n_new} topk={topk}", flush=True)

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

    gen_sp = SamplingParams(temperature=GEN_T, top_k=GEN_TOP_K, top_p=GEN_TOP_P,
                            max_tokens=n_new, seed=seed)
    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)

    traj_in = json.load(open(traj_in_path)) if traj_in_path else {}
    is_plain = (arm == "plain")
    rows: list[dict] = []
    any_nan = False
    n_pos = 0
    t0 = time.time()
    for p in prompts:
        ctx = list(p["context_token_ids"])[:ctx_cap]
        c = len(ctx)
        pid = str(p["id"])
        if pid in traj_in:
            traj = list(traj_in[pid])
        else:
            o = llm.generate([{"prompt_token_ids": ctx}], gen_sp, use_tqdm=False)[0]
            traj = list(o.outputs[0].token_ids)
        if not traj:
            continue
        full = ctx + traj
        vout = llm.generate([{"prompt_token_ids": full}], score_sp, use_tqdm=False)[0]
        pls = vout.prompt_logprobs or []
        per_pos: list[dict | None] = []
        argmax_ids: list[int | None] = []
        cls_list: list[str | None] = []
        for i in range(len(traj)):
            j = c + i
            if j >= len(pls) or pls[j] is None:
                per_pos.append(None)
                argmax_ids.append(None)
                cls_list.append(None)
                continue
            L = {int(t): rc._lp(v) for t, v in pls[j].items()}
            if not L:
                per_pos.append(None)
                argmax_ids.append(None)
                cls_list.append(None)
                continue
            if any(not math.isfinite(x) for x in L.values()):
                any_nan = True
            per_pos.append({str(t): v for t, v in L.items()})
            am = max(L.items(), key=lambda kv: kv[1])[0]
            argmax_ids.append(int(am))
            cls_list.append(classify_token(dec(am)) if is_plain else None)
            n_pos += 1
        row = {"id": p["id"], "domain": p.get("domain", "?"), "ctx_len": c,
               "traj": traj, "per_pos": per_pos}
        if is_plain:
            row["argmax_ids"] = argmax_ids
            row["cls"] = cls_list
        rows.append(row)

    if patch_mod is not None:
        st = getattr(patch_mod, "_stats", {})
        fa_flips = int(st.get("fa", 0))
        fa_eligible_mismatch = int(st.get("eligible_mt_mismatch", 0)) if "eligible_mt_mismatch" in st else None

    out = {
        "phase": "arm", "arm": arm, "arm_label": ARM_LABEL[arm], "model_dir": model_dir,
        "full_vocab_head": bool(full_vocab_ok), "backend_env": backend_env,
        "n_prompts": len(prompts), "n_positions": n_pos, "n_new": n_new, "topk": topk,
        "seed": seed, "fa_flips": fa_flips, "fa_eligible_mt_mismatch": fa_eligible_mismatch,
        "any_nan": bool(any_nan),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"))
    print(f"[arm:{arm}] DONE n_pos={n_pos} fa_flips={fa_flips} "
          f"mismatch={fa_eligible_mismatch} -> {out_path}", flush=True)
    print(f"ARM_DONE {arm} {out_path}", flush=True)


# ============================================================================================ #
# COMPOSE (pure CPU): per-pair divergence over the SHARED trajectory + verdict.
# ============================================================================================ #
def _rows_by_id(armobj: dict) -> dict[str, dict]:
    return {str(r["id"]): r for r in armobj["rows"]}


def compose_pair(base: dict, other: dict, classes: dict[str, list], n_reads: int = 12) -> dict:
    """Compose one arm-pair: TV/KL/argmax-flip per position along the trajectory, plus the
    answer-bearing-vs-filler split and the high-TV reads. `base` is always plain (it owns the
    trajectory + the per-position token classes in `classes`)."""
    b_by = _rows_by_id(base)
    o_by = _rows_by_id(other)
    pooled_tv: list[float] = []
    pooled_kl: list[float] = []
    by_domain: dict[str, list[float]] = {}
    by_class: dict[str, dict[str, float]] = {
        c: {"n": 0.0, "tv_sum": 0.0, "tv_max": 0.0, "flip": 0.0} for c in ("digit", "alpha", "filler")
    }
    tv_hist = {b: 0 for b in HIST_BINS}
    n_pos = 0
    n_flip = 0
    any_nan = False
    reads: list[tuple] = []
    read_tie = 0
    for pid, br in b_by.items():
        oro = o_by.get(pid)
        if oro is None:
            continue
        bpp = br["per_pos"]
        opp = oro["per_pos"]
        cls_row = classes.get(pid, [])
        for i in range(min(len(bpp), len(opp))):
            Lb_raw = bpp[i]
            Lo_raw = opp[i]
            if Lb_raw is None or Lo_raw is None:
                continue
            Lb = {int(k): float(v) for k, v in Lb_raw.items()}
            Lo = {int(k): float(v) for k, v in Lo_raw.items()}
            pb = gen_config_dist(Lb)
            po = gen_config_dist(Lo)
            t = tv(pb, po)
            k = kl(pb, po)
            if not math.isfinite(t):
                any_nan = True
            am_b = max(Lb.items(), key=lambda kv: kv[1])[0]
            am_o = max(Lo.items(), key=lambda kv: kv[1])[0]
            flip = am_b != am_o
            cls = cls_row[i] if i < len(cls_row) and cls_row[i] else "filler"
            dom = br.get("domain", "?")
            n_pos += 1
            pooled_tv.append(t)
            pooled_kl.append(min(k, KL_CAP) if math.isfinite(k) else KL_CAP)
            by_domain.setdefault(dom, []).append(t)
            cb = by_class[cls]
            cb["n"] += 1
            cb["tv_sum"] += t
            cb["tv_max"] = max(cb["tv_max"], t)
            if flip:
                n_flip += 1
                cb["flip"] += 1
            tv_hist[hist_bin(t)] += 1
            if flip or t > TV_BLOWUP:
                def top5(L):
                    d = gen_config_dist(L)
                    return [{"tok_id": int(tt), "lp": round(float(L[tt]), 4), "p": round(float(pr), 4)}
                            for tt, pr in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:5]]
                detail = {"prompt_id": br["id"], "domain": dom, "pos": i, "cls": cls,
                          "tv": round(t, 4), "kl": (round(k, 4) if math.isfinite(k) else None),
                          "plain_argmax_id": int(am_b), "arm_argmax_id": int(am_o),
                          "argmax_flip": bool(flip),
                          "plain_top5": top5(Lb), "arm_top5": top5(Lo)}
                read_tie += 1
                if len(reads) < n_reads:
                    heapq.heappush(reads, (t, read_tie, detail))
                elif t > reads[0][0]:
                    heapq.heapreplace(reads, (t, read_tie, detail))

    cls_out = {}
    for c, cb in by_class.items():
        nn = cb["n"]
        cls_out[c] = {"n": int(nn), "mean_tv": (cb["tv_sum"] / nn if nn else float("nan")),
                      "max_tv": cb["tv_max"], "flip_rate": (cb["flip"] / nn if nn else float("nan"))}
    answer_n = cls_out["digit"]["n"] + cls_out["alpha"]["n"]
    answer_tv_sum = by_class["digit"]["tv_sum"] + by_class["alpha"]["tv_sum"]
    answer_flip = by_class["digit"]["flip"] + by_class["alpha"]["flip"]
    answer_mean_tv = (answer_tv_sum / answer_n) if answer_n else float("nan")
    filler_mean_tv = cls_out["filler"]["mean_tv"]
    conc = (answer_mean_tv / filler_mean_tv) if (filler_mean_tv and filler_mean_tv > 1e-12) else (
        float("inf") if (answer_mean_tv and answer_mean_tv > 1e-12) else float("nan"))
    locus = locus_of(pooled_tv) if pooled_tv else {"tax_locus": "diffuse", "histogram_shape": "diffuse"}
    return {
        "n_positions": n_pos,
        "tv": stats(pooled_tv), "kl": stats(pooled_kl),
        "argmax_flip_rate": (n_flip / n_pos if n_pos else float("nan")), "n_argmax_flip": n_flip,
        "tv_by_domain": {d: stats(v) for d, v in sorted(by_domain.items())},
        "tv_hist": tv_hist,
        "tv_by_class": cls_out,
        "answer_bearing_mean_tv": answer_mean_tv, "filler_mean_tv": filler_mean_tv,
        "answer_vs_filler_concentration": conc,
        "answer_bearing_flip_rate": (answer_flip / answer_n if answer_n else float("nan")),
        "locus": locus,
        "any_nan": bool(any_nan),
        "high_tv_reads": [d for _, _, d in sorted(reads, key=lambda r: r[0], reverse=True)],
    }


def compose(arm_objs: dict[str, dict]) -> dict:
    plain = arm_objs["plain"]
    classes = {str(r["id"]): r.get("cls", []) for r in plain["rows"]}
    pairs: dict[str, dict] = {}
    for a, b in PAIRS:
        if a in arm_objs and b in arm_objs:
            pairs[f"{a}|{b}"] = compose_pair(arm_objs[a], arm_objs[b], classes)

    floor = pairs.get("plain|plainB", {}).get("tv", {}).get("max", 0.0)
    floor = floor if (isinstance(floor, float) and math.isfinite(floor)) else 0.0
    asrun = pairs.get("plain|fast_asrun")
    forced = pairs.get("plain|fast_forced")

    # ---- verdict from the AS-RUN pair (the question fern actually ran) ----
    a_meantv = asrun["tv"]["mean"] if asrun else float("nan")
    a_maxtv = asrun["tv"]["max"] if asrun else float("nan")
    a_flip = asrun["argmax_flip_rate"] if asrun else float("nan")
    a_locus = asrun["locus"]["tax_locus"] if asrun else "diffuse"
    a_conc = asrun["answer_vs_filler_concentration"] if asrun else float("nan")
    a_ans = asrun["answer_bearing_mean_tv"] if asrun else float("nan")
    a_fil = asrun["filler_mean_tv"] if asrun else float("nan")

    above_floor = (math.isfinite(a_maxtv) and a_maxtv > floor + TV_MATERIAL)
    real_flip = (math.isfinite(a_flip) and a_flip > FLIP_MATERIAL)
    is_real = above_floor or real_flip
    if not is_real:
        kernel_numerics_tax = "negligible"
    else:
        kernel_numerics_tax = "real-concentrated" if a_locus == "concentrated" else "real-diffuse"
    # answer-concentrated AND real -> could corroborate an AIME-style answer-token collapse.
    answer_concentrated = (math.isfinite(a_conc) and a_conc >= CONCENTRATION_HI) or (
        math.isfinite(a_ans) and math.isfinite(a_fil) and a_ans > a_fil + TV_MATERIAL)
    corroborates_fern_aime = bool(is_real and answer_concentrated)

    asrun_fa_flips = arm_objs.get("fast_asrun", {}).get("fa_flips")
    forced_fa_flips = arm_objs.get("fast_forced", {}).get("fa_flips")
    forced_mismatch = arm_objs.get("fast_forced", {}).get("fa_eligible_mt_mismatch")

    # ---- self-test: controls that prove the measurement is not blind ----
    checks = {
        # plain re-score is the determinism floor: must be EXACTLY 0 (int4 forward is reproducible).
        "determinism_floor_tv0": floor == 0.0,
        "determinism_floor_no_flip": (pairs.get("plain|plainB", {}).get("argmax_flip_rate", 1.0) == 0.0),
        "nan_clean": not any(p.get("any_nan") for p in pairs.values()),
        "full_vocab_head": bool(plain.get("full_vocab_head")),
        # the as-run patch did NOT fire on the native head (the central mechanistic claim).
        "asrun_fa_inert": (asrun_fa_flips == 0) if asrun_fa_flips is not None else True,
    }
    sensitivity = None
    if forced is not None:
        # POSITIVE CONTROL: the forced swap fired AND moved the distribution above the floor ->
        # the probe can see a real kernel tax. This is the blind-spot contrast (P|F_forced > 0).
        f_max = forced["tv"]["max"]
        sensitivity = {
            "forced_fa_flips": forced_fa_flips,
            "forced_eligible_mt_mismatch": forced_mismatch,
            "forced_max_tv": f_max, "forced_mean_tv": forced["tv"]["mean"],
            "forced_argmax_flip_rate": forced["argmax_flip_rate"],
            "probe_sees_tax": bool((forced_fa_flips or 0) > 0 and math.isfinite(f_max) and f_max > floor + TV_MATERIAL),
        }
        checks["forced_swap_fired"] = (forced_fa_flips or 0) > 0
        checks["probe_sees_forced_tax"] = sensitivity["probe_sees_tax"]
    passes = all(checks.values())

    if kernel_numerics_tax == "negligible":
        verdict = (
            f"NEGLIGIBLE on the native int4 head -- the as-run fast-kernel stack is BIT-IDENTICAL "
            f"to plain: P|F_asrun mean TV={a_meantv:.2e}, max TV={a_maxtv:.2e}, argmax-flip "
            f"rate={a_flip:.2e} (== the determinism floor {floor:.2e}). The fa_sliding swap NEVER "
            f"fires (fa_flips={asrun_fa_flips}: native model_type 'gemma4_text' != guard 'gemma4'), "
            f"and PLE-fold/bf16/split-KV are bit-identical/common-mode. There is NO concentrated "
            f"per-position divergence, so the fast stack CANNOT mechanistically cause fern #535's "
            f"AIME drop ({FERN_AIME_BASE}->{FERN_AIME_FAST}, n={FERN_N}); that gap is consistent "
            f"with n=30 sampling noise. corroborates_fern_aime=NO."
        )
    else:
        verdict = (
            f"REAL ({kernel_numerics_tax}) on the native head -- P|F_asrun mean TV={a_meantv:.3f}, "
            f"max TV={a_maxtv:.3f}, argmax-flip rate={a_flip:.3f}, answer/filler "
            f"concentration={rc._f(a_conc):.2f}. corroborates_fern_aime="
            f"{'YES' if corroborates_fern_aime else 'NO'}."
        )
    if sensitivity is not None:
        verdict += (
            f" PROBE SENSITIVITY (positive control F_forced, guard relaxed): forced swap fired on "
            f"{forced_fa_flips} layers -> max TV={sensitivity['forced_max_tv']:.3f}, flip "
            f"rate={sensitivity['forced_argmax_flip_rate']:.3f} ({forced['locus']['tax_locus']} locus) "
            f"-- the as-designed surgical-attn tax, latent because the guard never matches the "
            f"native head: exactly the {forced_fa_flips} eligible sliding layers the as-run guard "
            f"silently skipped (relaxed-guard eligibility-drift={forced_mismatch})."
        )

    return {
        "live_board_submission": LIVE_BOARD_SUBMISSION,
        "fern_aime": {"base": FERN_AIME_BASE, "fast": FERN_AIME_FAST, "gate": FERN_AIME_GATE,
                      "n": FERN_N, "delta": round(FERN_AIME_BASE - FERN_AIME_FAST, 4)},
        "determinism_floor_max_tv": floor,
        # ---- KEY OUTPUTS (as-run P|F_asrun) ----
        "mean_tv_plain_vs_fast": a_meantv,
        "max_tv": a_maxtv,
        "tv_histogram": (asrun["tv_hist"] if asrun else None),
        "tv_histogram_shape": (asrun["locus"]["histogram_shape"] if asrun else None),
        "kernel_argmax_flip_rate": a_flip,
        "answer_vs_filler_concentration": a_conc,
        "answer_bearing_mean_tv": a_ans,
        "filler_mean_tv": a_fil,
        "tax_locus": a_locus,
        "kernel_numerics_tax": kernel_numerics_tax,
        "corroborates_fern_aime": corroborates_fern_aime,
        # ---- mechanism audit ----
        "asrun_fa_flips": asrun_fa_flips,
        "forced_fa_flips": forced_fa_flips,
        "forced_eligible_mt_mismatch": forced_mismatch,
        "probe_sensitivity": sensitivity,
        "pairs": pairs,
        "selftest": {"passes": passes, "n_checks": len(checks), "checks": checks},
        "verdict": verdict,
    }


# ============================================================================================ #
# SELF-TEST (GPU-free): the deterministic transform + divergence + locus pipeline, and the
# probe-sensitivity unit control (a synthetic tax MUST be detected). Advisor reproduce entry.
# ============================================================================================ #
def run_self_test() -> int:
    checks: dict[str, bool] = {}
    # 1) determinism: same logprobs -> same distribution (pure function).
    L = {50000: -0.2, 1: -1.0, 2: -1.6, 3: -2.0, 99999: -5.0, 7: -2.4}
    checks["transform_deterministic"] = gen_config_dist(L) == gen_config_dist(dict(L))
    # 2) tv/kl identity: tv(p,p)=0, kl(p,p)=0 (the determinism-floor control).
    p = gen_config_dist(L)
    checks["tv_identity_zero"] = tv(p, p) == 0.0 and kl(p, p) == 0.0
    # 3) tv symmetry + disjoint-support = 1 (bound sanity).
    a = {1: 1.0}
    b = {2: 1.0}
    checks["tv_disjoint_one"] = abs(tv(a, b) - 1.0) < 1e-12
    checks["tv_symmetric"] = abs(tv(p, gen_config_dist({**L, 2: -0.5})) -
                                 tv(gen_config_dist({**L, 2: -0.5}), p)) < 1e-12
    # 4) PROBE SENSITIVITY (the F_forced analog at unit level): a perturbation that swaps the top
    #    two logits MUST produce TV>0 and a detected argmax flip. Proves the probe is not blind.
    L2 = dict(L)
    L2[50000], L2[1] = L[1], L[50000]      # swap top-1 and a lower token -> argmax moves
    q = gen_config_dist(L2)
    am_p = max(L.items(), key=lambda kv: kv[1])[0]
    am_q = max(L2.items(), key=lambda kv: kv[1])[0]
    tv_tax = tv(p, q)
    # material TV (orders above the 0 determinism floor) AND a detected argmax flip == the probe
    # is not blind. A near-disjoint perturbation must additionally reach the blow-up scale.
    checks["probe_detects_tax_tv"] = tv_tax > TV_MATERIAL
    checks["probe_detects_argmax_flip"] = am_p != am_q
    q_big = gen_config_dist({7: 0.0, 50000: -9.0, 1: -9.5, 2: -10.0})
    checks["probe_detects_blowup_tv"] = tv(p, q_big) > TV_BLOWUP
    # 5) classify_token split.
    checks["classify_digit"] = classify_token("42") == "digit"
    checks["classify_alpha"] = classify_token(" the") == "alpha"
    checks["classify_filler"] = classify_token("  ") == "filler" and classify_token(".") == "filler"
    # 6) locus: a concentrated/bimodal TV vector (one near-1, rest ~0) -> concentrated; a uniformly
    #    tiny TV vector -> diffuse. (Distinguishes an answer-flip collapse from low-grade fp noise.)
    conc_vec = [0.0] * 999 + [1.0]
    diff_vec = [1e-4] * 1000
    checks["locus_concentrated"] = locus_of(conc_vec)["tax_locus"] == "concentrated"
    checks["locus_diffuse"] = locus_of(diff_vec)["tax_locus"] == "diffuse"
    checks["hist_bimodal_flag"] = locus_of(conc_vec)["histogram_shape"] == "bimodal"
    # 7) NaN-clean across the synthetic path.
    checks["nan_clean"] = all(math.isfinite(x) for x in (tv_tax, kl(p, q), tv(a, b)))

    passes = all(checks.values())
    print("[self-test] fast_kernel_divergence")
    print(f"  determinism floor: tv(p,p)={tv(p, p):.2e}  perturbed tax: tv={tv_tax:.4f} "
          f"argmax_flip={am_p != am_q}")
    print(f"  locus concentrated={locus_of(conc_vec)}  diffuse={locus_of(diff_vec)['tax_locus']}")
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
        print(f"[#540] wandb helpers unavailable: {e}")
        return None
    c = payload["compose"]
    run = init_wandb_run(
        job_type="analysis-fast-kernel-divergence", agent="denken",
        name=args.wandb_name, group=args.wandb_group,
        tags=["fast-kernel-divergence", "fa-sliding", "surgical-attn", "int4", "mmlu", "math",
              "gsm8k", "gen-config-sampling", "positive-control", "pr-540"],
        config={"pr": 540, "kind": "fast-kernel-divergence",
                "axis": "plain_int4_vs_fast_kernel_int4",
                "levers": ["fa_sliding(surgical_2d_attn)", "split_kv", "ple_fold", "bf16"],
                "gen_config": {"temperature": GEN_T, "top_k": GEN_TOP_K, "top_p": GEN_TOP_P, "do_sample": True},
                "live_board_submission": LIVE_BOARD_SUBMISSION, "full_vocab": FULL_VOCAB,
                "n_new": payload["gpu_meta"]["n_new"], "topk": payload["gpu_meta"]["topk"],
                "seed": payload["gpu_meta"]["seed"], "n_prompts": payload["gpu_meta"]["n_prompts"],
                "fern_aime": c["fern_aime"]},
    )
    if run is None:
        print("[#540] wandb disabled (no API key / WANDB_MODE).")
        return None

    def _f(x):
        return rc._f(x)

    flat: dict[str, float] = {
        "div/mean_tv_plain_vs_fast": _f(c["mean_tv_plain_vs_fast"]),
        "div/max_tv": _f(c["max_tv"]),
        "div/kernel_argmax_flip_rate": _f(c["kernel_argmax_flip_rate"]),
        "div/answer_vs_filler_concentration": _f(c["answer_vs_filler_concentration"]),
        "div/answer_bearing_mean_tv": _f(c["answer_bearing_mean_tv"]),
        "div/filler_mean_tv": _f(c["filler_mean_tv"]),
        "div/determinism_floor_max_tv": _f(c["determinism_floor_max_tv"]),
        "div/kernel_numerics_tax_real": float(c["kernel_numerics_tax"] != "negligible"),
        "div/corroborates_fern_aime": float(bool(c["corroborates_fern_aime"])),
        "audit/asrun_fa_flips": _f(c["asrun_fa_flips"]),
        "audit/forced_fa_flips": _f(c["forced_fa_flips"]),
        "audit/forced_eligible_mt_mismatch": _f(c["forced_eligible_mt_mismatch"]),
        "selftest/fast_kernel_divergence_self_test_passes": float(c["selftest"]["passes"]),
    }
    ps = c.get("probe_sensitivity")
    if ps:
        flat.update({
            "sensitivity/forced_max_tv": _f(ps["forced_max_tv"]),
            "sensitivity/forced_mean_tv": _f(ps["forced_mean_tv"]),
            "sensitivity/forced_argmax_flip_rate": _f(ps["forced_argmax_flip_rate"]),
            "sensitivity/probe_sees_tax": float(bool(ps["probe_sees_tax"])),
        })
    for pname, pr in c["pairs"].items():
        tag = pname.replace("|", "_vs_")
        flat[f"pair/{tag}/mean_tv"] = _f(pr["tv"]["mean"])
        flat[f"pair/{tag}/max_tv"] = _f(pr["tv"]["max"])
        flat[f"pair/{tag}/p99_tv"] = _f(pr["tv"]["p99"])
        flat[f"pair/{tag}/argmax_flip_rate"] = _f(pr["argmax_flip_rate"])
        flat[f"pair/{tag}/answer_mean_tv"] = _f(pr["answer_bearing_mean_tv"])
        flat[f"pair/{tag}/filler_mean_tv"] = _f(pr["filler_mean_tv"])
    asrun = c["pairs"].get("plain|fast_asrun")
    if asrun:
        for dom, st in asrun["tv_by_domain"].items():
            flat[f"{dom}/asrun_max_tv"] = _f(st["max"])
            flat[f"{dom}/asrun_mean_tv"] = _f(st["mean"])
    run.log({"global_step": 0, **{k: (v if (isinstance(v, float) and math.isfinite(v)) else 0.0)
                                  for k, v in flat.items()}})
    log_summary(run, rc._jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="fast_kernel_divergence", artifact_type="analysis",
                      data=rc._jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[#540] wandb logged (run {rid})")
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
    m = harness.load_manifest(ROOT / "submissions" / "fa2sw_strict_m1ar_int4")
    return str(harness.ensure_server_venv(m["dependencies"]))


def run_arm_phase(server_python: str, arm: str, phase_args: list[str], timeout: int) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # DO NOT force VLLM_ATTENTION_BACKEND: we want the model's own default (uniform TRITON_ATTN);
    # forcing flash here would erase the very surgical-attn tax under measurement. Each arm controls
    # its backend only via its in-process patch (none / fa_sliding / forced-fa).
    env.pop("VLLM_ATTENTION_BACKEND", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [server_python, os.path.abspath(__file__)] + phase_args
    print(f"[orch] launching arm={arm}: {' '.join(phase_args)}", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] arm {arm} TIMED OUT after {timeout}s", flush=True)
        return 124


def orchestrate(args) -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    if not Path(args.prompts).exists():
        raise FileNotFoundError(f"prompts not found at {args.prompts}")
    arms = list(args.arms.split(",")) if args.arms else list(ARMS)
    traj_path = str(HERE / "_traj.json")
    arm_json = {a: str(HERE / f"_arm_{a}.json") for a in ARMS}

    if not args.skip_gpu:
        server_python = resolve_server_python(args.server_python)
        print(f"[orch] server_python = {server_python}", flush=True)
        for arm in arms:
            out = arm_json[arm]
            pa = ["--phase", "arm", "--arm", arm, "--prompts", args.prompts, "--out", out,
                  "--n-new", str(args.n_new), "--topk", str(args.topk),
                  "--ctx-cap", str(args.ctx_cap), "--gpu-mem-util", str(args.gpu_mem_util),
                  "--seed", str(args.seed)]
            # plain GENERATES the trajectory (extracted from its JSON below); everyone re-scores it.
            if arm != "plain":
                if not Path(traj_path).exists():
                    raise FileNotFoundError(f"trajectory {traj_path} missing; run the plain arm first")
                pa += ["--traj-in", traj_path]
            rcode = run_arm_phase(server_python, arm, pa, timeout=args.gpu_timeout)
            if rcode != 0:
                print(f"[orch] arm {arm} FAILED rc={rcode}", flush=True)
                return rcode
            # extract the shared trajectory from the plain arm for the re-score arms.
            if arm == "plain":
                pj = json.load(open(out))
                traj = {str(r["id"]): r["traj"] for r in pj["rows"]}
                json.dump(traj, open(traj_path, "w"))
                print(f"[orch] wrote shared trajectory ({len(traj)} prompts) -> {traj_path}", flush=True)

    arm_objs = {a: json.load(open(arm_json[a])) for a in arms if Path(arm_json[a]).exists()}
    if "plain" not in arm_objs:
        raise RuntimeError("plain arm output missing; cannot compose")
    comp = compose(arm_objs)
    gpu_meta = {"model_dir": arm_objs["plain"]["model_dir"],
                "n_prompts": arm_objs["plain"]["n_prompts"],
                "n_positions": arm_objs["plain"]["n_positions"],
                "n_new": arm_objs["plain"]["n_new"], "topk": arm_objs["plain"]["topk"],
                "seed": arm_objs["plain"]["seed"],
                "peak_mem_mib": max(a.get("peak_mem_mib", 0) for a in arm_objs.values()),
                "arms": {a: {"fa_flips": o.get("fa_flips"), "n_positions": o.get("n_positions"),
                             "elapsed_s": o.get("elapsed_s"), "backend_env": o.get("backend_env"),
                             "any_nan": o.get("any_nan")} for a, o in arm_objs.items()}}
    payload = {
        "pr": 540, "card": "fast-kernel-divergence",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True, "official_tps": 0,
        "compose": comp, "gpu_meta": gpu_meta,
    }
    out_path = HERE / "fast_kernel_divergence.json"
    json.dump(rc._jsonable(payload), open(out_path, "w"), indent=2)

    # console summary
    print("\n" + "=" * 92, flush=True)
    print(f"  live board = {comp['live_board_submission']}   fern AIME "
          f"{comp['fern_aime']['base']}->{comp['fern_aime']['fast']} (n={comp['fern_aime']['n']}, "
          f"gate>={comp['fern_aime']['gate']}, delta={comp['fern_aime']['delta']})", flush=True)
    print(f"  DETERMINISM FLOOR max TV = {comp['determinism_floor_max_tv']:.3g}", flush=True)
    print(f"  P|F_asrun  mean TV={rc._f(comp['mean_tv_plain_vs_fast']):.3g}  max TV={rc._f(comp['max_tv']):.3g}  "
          f"argmax_flip={rc._f(comp['kernel_argmax_flip_rate']):.3g}  "
          f"answer/filler_conc={rc._f(comp['answer_vs_filler_concentration']):.3g}", flush=True)
    print(f"  tax_locus = {comp['tax_locus']}   hist_shape = {comp['tv_histogram_shape']}   "
          f"asrun_fa_flips = {comp['asrun_fa_flips']}", flush=True)
    ps = comp.get("probe_sensitivity")
    if ps:
        print(f"  POSITIVE CONTROL F_forced: fa_flips={ps['forced_fa_flips']} "
              f"(=eligible layers skipped as-run; elig-drift={comp['forced_eligible_mt_mismatch']})  "
              f"max TV={rc._f(ps['forced_max_tv']):.3g}  "
              f"flip={rc._f(ps['forced_argmax_flip_rate']):.3g}  probe_sees_tax={ps['probe_sees_tax']}", flush=True)
    print(f"  kernel_numerics_tax = {comp['kernel_numerics_tax'].upper()}   "
          f"corroborates_fern_aime = {comp['corroborates_fern_aime']}", flush=True)
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
    ap.add_argument("--phase", choices=["arm"], default=None)
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--prompts", default=str(ROOT / "research" / "validity" /
                                             "served_benchmark_divergence" / "prompts.jsonl"))
    ap.add_argument("--arms", default=None, help="comma list subset of arms (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--traj-in", default=None)
    ap.add_argument("--n-new", type=int, default=320)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--ctx-cap", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=GEN_SEED)
    ap.add_argument("--gpu-timeout", type=int, default=5400)
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--skip-gpu", action="store_true", help="re-compose from existing _arm_*.json")
    ap.add_argument("--wandb_name", default="denken/fast-kernel-divergence")
    ap.add_argument("--wandb_group", default="fast-kernel-divergence")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="GPU-free transform+locus+probe-sensitivity self-test (advisor reproduce entry)")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if args.phase == "arm":
        phase_arm(args.arm, args.prompts, args.traj_in, args.out, args.n_new, args.topk,
                  args.ctx_cap, args.gpu_mem_util, args.seed)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
