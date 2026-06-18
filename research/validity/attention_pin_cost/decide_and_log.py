#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #691 (land) -- combine the per-config ctx sweeps into the crossover decision.

Reads runs/baseline.json (3D split) + runs/fixed2d.json (2D pin) [+ runs/bi1.json], computes
the per-ctx pin-cost curve  pin_delta(ctx) = attn_2d(ctx) - attn_3d(ctx)  (negative=free),
locates the 3D->2D crossover ctx, the WORST-CASE pin cost across the swept range, bounds it
against the ctx-independent bi1 -16% blanket floor, and translates to the official-equiv
K=5 spec-dec TPS basis (MODELLED, x0.870).

DECISION MODEL: identical to land #684 (denken #677 anchored). The pin's attention delta in
ABSOLUTE ms transfers onto denken's deployed 8.14 ms AR step (lm_head GEMM cancels in the
3D->2D difference -> head-independent):
    T_iter(K) = (8.14 + delta_pin) + DRAFT_TERM ;  DRAFT_TERM derived so the model reproduces
    denken's PUBLISHED strict-rescue official 137.14 @K5 exactly.  TPS = E/T_iter * 0.870.

analysis_only=1, official_tps=0, no_hf_job=1, fires=0. Run with /usr/bin/python (has wandb).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# ---- anchors (land #684 / denken #677 baseline; do not change) ----
AR_STEP_LOCAL_MS = 8.14          # denken deployed M=1 AR step (official 126.378)
LOCAL_TO_OFFICIAL = 0.870
REF_OFFICIAL_TPS = 126.378
PLUS10_BAR = 136.378
E_ACCEPT = 3.343                 # denken e_accept* at optimal K
K_OPT = 5                        # verify width M=6
RESCUE_BASE_LOCAL_MS = 12.98     # denken strict per-iter base step (AR 8.14 + 4.84 BI+rescue)
RESCUE_TAX_MS = 4.84
RESCUE_OFFICIAL_AT_K5 = 137.14   # denken PUBLISHED strict-rescue official @K5 (+0.76)
# bi1 ctx-INDEPENDENT blanket deployed cost = lawine #675 -16% AR -> +1.55 ms (the floor tier)
BI1_BLANKET_DELTA_MS = 1.55

# deployed benchmark decode-ctx geometry (128 public prompts, ppl_ground_truth_tokens.jsonl;
# decode trajectory = prompt_len .. prompt_len+512). Measured 2026-06-18.
DEPLOYED_PROMPT_MEDIAN = 234
DEPLOYED_PROMPT_P90 = 392
DEPLOYED_PROMPT_MAX = 2431
DEPLOYED_DECODE_CEILING = DEPLOYED_PROMPT_MAX + 512    # 2943, strict worst tail
DEPLOYED_DECODE_P90 = DEPLOYED_PROMPT_P90 + 512        # 904, representative bulk ceiling

_RESCUE_LOCAL = RESCUE_OFFICIAL_AT_K5 / LOCAL_TO_OFFICIAL
_T_ITER_RESCUE = 1000.0 * E_ACCEPT / _RESCUE_LOCAL
DRAFT_TERM_MS = _T_ITER_RESCUE - RESCUE_BASE_LOCAL_MS

GROUP = os.environ.get("WANDB_GROUP", "strict319-attention-pin-cost-land")
NAME = os.environ.get("WANDB_NAME", "land/pin-ctx-crossover")


def _load(cfg):
    p = RUNS / f"{cfg}.json"
    return json.load(open(p)) if p.exists() else None


def _specdec_official(base_step_local_ms, e=E_ACCEPT):
    t_iter = base_step_local_ms + DRAFT_TERM_MS
    tps_local = 1000.0 * e / t_iter
    return tps_local * LOCAL_TO_OFFICIAL, t_iter, tps_local


def _interp(xs, ys, x):
    """Linear interpolation of y at x given sorted xs (clamped at ends)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def _crossover(ctxs, deltas):
    """First ctx where delta crosses 0 from <=0 to >0 (linear interp). Returns
    (crossover_ctx or None, 'free_all'|'costly_all'|'crosses')."""
    if all(d <= 0 for d in deltas):
        return None, "free_all"
    if all(d > 0 for d in deltas):
        return ctxs[0], "costly_all"     # already costly at smallest ctx
    for i in range(1, len(ctxs)):
        if deltas[i - 1] <= 0 < deltas[i]:
            t = (0.0 - deltas[i - 1]) / (deltas[i] - deltas[i - 1])
            return ctxs[i - 1] + t * (ctxs[i] - ctxs[i - 1]), "crosses"
    # delta dips back below 0 after a positive excursion, or other shape -> first +ve point
    for i, d in enumerate(deltas):
        if d > 0:
            return ctxs[i], "crosses"
    return None, "free_all"


def main() -> int:
    base = _load("baseline")
    fix = _load("fixed2d")
    bi1 = _load("bi1")
    if not base or not fix:
        print("[691] missing baseline.json or fixed2d.json -- cannot compute crossover")
        return 1

    b_by_ctx = {r["ctx"]: r["attn_step_ms"] for r in base["per_ctx"]}
    f_by_ctx = {r["ctx"]: r["attn_step_ms"] for r in fix["per_ctx"]}
    bi_by_ctx = {r["ctx"]: r["attn_step_ms"] for r in bi1["per_ctx"]} if bi1 else {}
    ctxs = sorted(set(b_by_ctx) & set(f_by_ctx))

    curve = []
    deltas = []
    for ctx in ctxs:
        t3d = b_by_ctx[ctx]
        t2d = f_by_ctx[ctx]
        pin_delta = t2d - t3d                       # absolute ms (negative = free)
        cost_frac_deployed = pin_delta / (AR_STEP_LOCAL_MS + pin_delta)   # matches #684 -11.6%
        cost_frac_arstep = pin_delta / AR_STEP_LOCAL_MS
        spec_off, t_iter, spec_loc = _specdec_official(AR_STEP_LOCAL_MS + pin_delta)
        margin = spec_off - PLUS10_BAR
        bi_delta = (bi_by_ctx.get(ctx, float("nan")) - t3d) if bi_by_ctx else None
        deltas.append(pin_delta)
        curve.append({
            "ctx": ctx,
            "attn_3d_ms": t3d, "attn_2d_ms": t2d,
            "pin_delta_ms": pin_delta,
            "cost_frac_deployed": cost_frac_deployed,
            "cost_frac_arstep": cost_frac_arstep,
            "specdec_k5_official": spec_off,
            "margin_vs_plus10": margin,
            "clears_plus10": bool(spec_off > PLUS10_BAR),
            "bi1_attn_delta_ms": bi_delta,
        })

    # crossover + worst case across the FULL swept range
    crossover_ctx, shape = _crossover(ctxs, deltas)
    worst_i = max(range(len(deltas)), key=lambda i: deltas[i])
    worst = curve[worst_i]
    worstcase_pin_delta = worst["pin_delta_ms"]
    worstcase_pin_cost_frac = worst["cost_frac_deployed"]
    worstcase_ctx = worst["ctx"]
    worstcase_specdec = worst["specdec_k5_official"]
    worstcase_margin = worst["margin_vs_plus10"]

    # deployed-range checks: interpolate pin_delta at the deployed decode ceilings
    pin_at_deploy_ceiling = _interp(ctxs, deltas, DEPLOYED_DECODE_CEILING)
    pin_at_deploy_p90 = _interp(ctxs, deltas, DEPLOYED_DECODE_P90)
    spec_at_ceiling, _, _ = _specdec_official(AR_STEP_LOCAL_MS + pin_at_deploy_ceiling)
    margin_at_ceiling = spec_at_ceiling - PLUS10_BAR
    # breach iff any swept ctx <= deployed ceiling has fixed2d specdec margin < 0
    deployed_pts = [c for c in curve if c["ctx"] <= DEPLOYED_DECODE_CEILING]
    breach = any(c["margin_vs_plus10"] < 0 for c in deployed_pts) or (margin_at_ceiling < 0)
    free_across_deployed = (crossover_ctx is None) or (crossover_ctx > DEPLOYED_DECODE_CEILING)

    # bi1 floor tier (ctx-independent aten blanket, anchored at deployed ctx)
    bi1_spec_off, _, _ = _specdec_official(AR_STEP_LOCAL_MS + BI1_BLANKET_DELTA_MS)
    bi1_margin = bi1_spec_off - PLUS10_BAR
    bi1_floor_clears = bool(bi1_spec_off > PLUS10_BAR)
    # fixed2d dominates bi1 at every ctx: bi1 = (same 2D attention) + ctx-indep aten blanket.
    # Confirm bi1's measured ATTENTION delta ~= fixed2d's (both 2D), so any ctx-dependence of
    # bi1 is the shared attention; the +1.55ms blanket is the ctx-independent residual.
    bi1_attn_matches_fixed2d = None
    if bi_by_ctx:
        diffs = [abs((bi_by_ctx[c] - b_by_ctx[c]) - (f_by_ctx[c] - b_by_ctx[c]))
                 for c in ctxs if c in bi_by_ctx]
        bi1_attn_matches_fixed2d = bool(diffs and max(diffs) < 0.30)  # within 0.3 ms/step
    fixed2d_beats_bi1_all_ctx = bool(worstcase_pin_delta <= BI1_BLANKET_DELTA_MS)

    # rescue self-consistency (proves the model basis ties out to denken before trusting it)
    rescue_off, _, _ = _specdec_official(RESCUE_BASE_LOCAL_MS)
    rescue_reproduces_denken = bool(abs(rescue_off - RESCUE_OFFICIAL_AT_K5) < 0.05)
    # fixed2d tier at the deployed anchor ctx=512 (the +50.97 number) -- look it up
    # explicitly (robust to a ladder that starts below 512, e.g. 256/384 free-zone points).
    c512 = next((c for c in curve if c["ctx"] == 512), None)
    pin_at_512 = c512["pin_delta_ms"] if c512 else float("nan")
    spec_at_512 = c512["specdec_k5_official"] if c512 else float("nan")
    # #684 measured the pin as FREE (-0.843 ms) because its real-prompt decode KV averaged
    # BELOW the 512 sliding window. Reconcile via the free zone: the most-negative pin_delta
    # and the ctx where it occurs. A negative delta at a sub-512 ctx confirms #684's direction.
    min_i = min(range(len(deltas)), key=lambda i: deltas[i])
    freezone_min_delta_ms = deltas[min_i]
    freezone_min_ctx = ctxs[min_i]
    reproduces_684 = bool(freezone_min_delta_ms < 0.0 and freezone_min_ctx < 512)

    # verdict
    if breach:
        verdict = "PIN_COST_BREACHES"
    elif free_across_deployed:
        verdict = "PIN_FREE_AT_SERVED_CTX"
    else:
        verdict = "PIN_COSTLY_AT_LONG_CTX"

    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "verdict": verdict,
        # primary / test metrics
        "worstcase_pin_cost_frac": worstcase_pin_cost_frac,
        "worstcase_pin_delta_ms": worstcase_pin_delta,
        "worstcase_ctx": worstcase_ctx,
        "worstcase_specdec_k5_official": worstcase_specdec,
        "worstcase_margin_vs_plus10": worstcase_margin,
        "crossover_ctx": (crossover_ctx if crossover_ctx is not None else -1),
        "crossover_shape": shape,
        "crossover_beyond_deployed": bool(free_across_deployed),
        # deployed operating point
        "deployed_decode_ceiling_ctx": DEPLOYED_DECODE_CEILING,
        "deployed_decode_p90_ctx": DEPLOYED_DECODE_P90,
        "pin_delta_at_deploy_ceiling_ms": pin_at_deploy_ceiling,
        "pin_delta_at_deploy_p90_ms": pin_at_deploy_p90,
        "specdec_k5_at_deploy_ceiling": spec_at_ceiling,
        "margin_at_deploy_ceiling": margin_at_ceiling,
        "deployed_breach_plus10": int(breach),
        # fixed2d tier at deployed (ctx 512) -- the +50.97 number
        "fixed2d_pin_delta_at_512_ms": pin_at_512,
        "fixed2d_specdec_k5_at_512": spec_at_512,
        "fixed2d_margin_at_512": (spec_at_512 - PLUS10_BAR) if math.isfinite(spec_at_512) else None,
        "reproduces_684_free_pin": int(reproduces_684),
        "freezone_min_delta_ms": freezone_min_delta_ms,
        "freezone_min_ctx": freezone_min_ctx,
        # bi1 floor tier (ctx-independent, the airtight fallback)
        "bi1_blanket_delta_ms": BI1_BLANKET_DELTA_MS,
        "bi1_specdec_k5_official": bi1_spec_off,
        "bi1_margin_vs_plus10": bi1_margin,
        "bi1_floor_clears_plus10": int(bi1_floor_clears),
        "bi1_attn_matches_fixed2d": (int(bi1_attn_matches_fixed2d)
                                     if bi1_attn_matches_fixed2d is not None else None),
        "fixed2d_beats_bi1_all_ctx": int(fixed2d_beats_bi1_all_ctx),
        # self-consistency
        "rescue_specdec_k5_official": rescue_off,
        "rescue_reproduces_denken_137p14": int(rescue_reproduces_denken),
        # anchors
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
        "local_to_official": LOCAL_TO_OFFICIAL, "e_accept": E_ACCEPT, "k_opt": K_OPT,
        "ar_step_local_ms": AR_STEP_LOCAL_MS, "draft_term_ms_derived": DRAFT_TERM_MS,
        "n_layers": base.get("n_layers"),
        "peak_mem_gib": max(base.get("peak_mem_gib", 0), fix.get("peak_mem_gib", 0)),
    }
    # per-ctx flat scalars
    for c in curve:
        ctx = c["ctx"]
        for k in ("attn_3d_ms", "attn_2d_ms", "pin_delta_ms", "cost_frac_deployed",
                  "cost_frac_arstep", "specdec_k5_official", "margin_vs_plus10",
                  "bi1_attn_delta_ms"):
            summary[f"ctx{ctx}/{k}"] = c[k]

    print("=" * 78)
    print(json.dumps({k: v for k, v in summary.items() if "/" not in k}, indent=2, default=str))
    print("--- per-ctx crossover curve ---")
    print(f"  {'ctx':>7} {'attn_3d':>9} {'attn_2d':>9} {'pin_delta':>10} {'cost%dep':>9} "
          f"{'specK5':>8} {'margin':>8} {'bi1_attn_d':>10}")
    for c in curve:
        bid = c["bi1_attn_delta_ms"]
        bid_s = f"{bid:+.4f}" if isinstance(bid, (int, float)) and math.isfinite(bid) else "   n/a  "
        print(f"  {c['ctx']:>7} {c['attn_3d_ms']:>9.5f} {c['attn_2d_ms']:>9.5f} "
              f"{c['pin_delta_ms']:>+10.5f} {c['cost_frac_deployed']*100:>+8.2f}% "
              f"{c['specdec_k5_official']:>8.2f} {c['margin_vs_plus10']:>+8.2f} {bid_s:>10}")
    print(f"  crossover_ctx={crossover_ctx} ({shape})  worstcase: ctx={worstcase_ctx} "
          f"pin={worstcase_pin_delta:+.4f}ms frac={worstcase_pin_cost_frac*100:+.2f}% "
          f"specK5={worstcase_specdec:.2f} margin={worstcase_margin:+.2f}")
    print(f"  free zone: min pin_delta={freezone_min_delta_ms:+.4f}ms @ctx={freezone_min_ctx} "
          f"(reproduces #684 free pin={reproduces_684})")
    print(f"  deployed ceiling ctx={DEPLOYED_DECODE_CEILING}: pin={pin_at_deploy_ceiling:+.4f}ms "
          f"margin={margin_at_ceiling:+.2f}  breach={breach}")
    print(f"  bi1 floor: specK5={bi1_spec_off:.2f} margin={bi1_margin:+.2f} clears={bi1_floor_clears}"
          f"  fixed2d_beats_bi1_all_ctx={fixed2d_beats_bi1_all_ctx}")
    print(f"  VERDICT={verdict}")
    print("=" * 78)

    json.dump({"summary": summary, "curve": curve},
              open(RUNS / "decision.json", "w"), indent=2, default=str)

    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("[691] wandb disabled via env -- decision.json written")
        return 0
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[691] wandb unavailable ({exc}) -- decision.json written")
        return 0

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=NAME, group=GROUP, job_type="analysis",
        tags=["gemma-challenge", "analysis", "attention-pin", "ctx-crossover",
              "triton-attn", "split-kv", "spec-dec", "issue-319", "pr-691"],
        config={
            "pr": 691, "issue": 319, "analysis_only": True, "no_hf_job": True,
            "wandb_group": GROUP, "vllm_version": "0.22.0", "attn_backend": "TRITON_ATTN",
            "ctx_ladder": ctxs, "model_dir": base.get("model_dir"),
            "margin_model_full_vocab": base.get("margin_model_full_vocab"),
            "n_new": base.get("n_new"), "reps": base.get("reps"),
            "deployed_prompt_median": DEPLOYED_PROMPT_MEDIAN,
            "deployed_prompt_p90": DEPLOYED_PROMPT_P90, "deployed_prompt_max": DEPLOYED_PROMPT_MAX,
            "head_independent_note": "pin cost = M=1 3D->2D attention delta in absolute ms; "
            "lm_head GEMM cancels in the difference -> transfers onto denken's 8.14ms AR step.",
        },
    )
    flat = {k: v for k, v in summary.items()
            if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}
    run.summary.update(flat)

    # crossover curve table + line plots
    cols = ["ctx", "attn_3d_ms", "attn_2d_ms", "pin_delta_ms", "cost_frac_deployed",
            "specdec_k5_official", "margin_vs_plus10", "bi1_attn_delta_ms"]
    tbl = wandb.Table(columns=cols)
    for c in curve:
        bid = c["bi1_attn_delta_ms"]
        tbl.add_data(c["ctx"], c["attn_3d_ms"], c["attn_2d_ms"], c["pin_delta_ms"],
                     c["cost_frac_deployed"], c["specdec_k5_official"], c["margin_vs_plus10"],
                     (bid if isinstance(bid, (int, float)) and math.isfinite(bid) else None))
    run.log({"crossover_curve": tbl})
    try:
        run.log({"pin_delta_vs_ctx": wandb.plot.line(tbl, "ctx", "pin_delta_ms",
                                                     title="pin cost (ms/step) vs ctx"),
                 "specdec_k5_vs_ctx": wandb.plot.line(tbl, "ctx", "specdec_k5_official",
                                                     title="fixed2d K=5 official vs ctx")})
    except Exception:  # noqa: BLE001
        pass

    print(f"[691] verdict={verdict}  crossover_ctx={crossover_ctx}  "
          f"worstcase_frac={worstcase_pin_cost_frac:+.4f}  W&B: {run.url}  id={run.id}")
    run.finish()
    print(f"WANDB_RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
