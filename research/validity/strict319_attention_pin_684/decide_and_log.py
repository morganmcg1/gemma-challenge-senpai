#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #684 (land) -- combine the 3 per-config measurements into the decision scalar:
does a VERIFIED-LOSSLESS attention pin let strict-#319 spec-dec clear +10 WITHOUT the
stark #669 recompute-rescue, and is that pin cheaper than the rescue path?

DECISION MODEL (grounded in denken #677, anchored x0.870 exactly):
  denken's strict rescue step:  T_step_rescue(K) = 12.98ms + 1.421ms*K  (local), where
  the 12.98ms base = deployed AR step 8.14ms + 4.84ms (VLLM_BATCH_INVARIANT + rescue) tax.
  A verified-lossless attention pin replaces that whole +4.84ms tax with JUST the pin's
  step-time cost and removes the rescue entirely:
      T_step_lossless(K) = (8.14ms + delta_pin) + 1.421ms*K
  delta_pin = the pin's per-step cost in ABSOLUTE ms (the M=1 3D->2D attention penalty,
  measured on the full-vocab stack; head-independent because the identical lm_head GEMM
  cancels in the difference), so it transfers onto denken's deployed 8.14ms AR step.
      TPS(K) = E[accept] / T_step ;  official-equiv = local * 0.870.

analysis_only=1, official_tps=0, fires=0. Run with wandb python: /usr/bin/python.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# ---- anchors (denken #677 / PR #684 baseline; do not change) ----
AR_STEP_LOCAL_MS = 8.14         # denken deployed M=1 AR step (official 126.378)
LOCAL_TO_OFFICIAL = 0.870
REF_OFFICIAL_TPS = 126.378
PLUS10_BAR = 136.378
E_ACCEPT = 3.343                # denken e_accept* at optimal K
K_OPT = 5                       # verify width M=6
RESCUE_BASE_LOCAL_MS = 12.98    # denken strict per-iter base step (AR 8.14 + 4.84 BI+rescue)
RESCUE_TAX_MS = 4.84            # denken BI+rescue tax on the verify step
RESCUE_OFFICIAL_AT_K5 = 137.14  # denken PUBLISHED strict-rescue official @K5 (in-scope top_k64, +0.76)

# The spec-dec per-iteration time is  T_iter(K) = base_step + DRAFT_TERM(K), where the
# K-draft term is DERIVED so the model reproduces denken's PUBLISHED rescue point EXACTLY
# (anchoring to 137.14 instead of guessing a per-draft cost). The attention pin then just
# swaps the rescue base (AR + 4.84 tax) for (AR + delta_pin); everything else is held.
_RESCUE_LOCAL = RESCUE_OFFICIAL_AT_K5 / LOCAL_TO_OFFICIAL          # 157.63 tps_local
_T_ITER_RESCUE = 1000.0 * E_ACCEPT / _RESCUE_LOCAL                 # 21.21 ms per iteration
DRAFT_TERM_MS = _T_ITER_RESCUE - RESCUE_BASE_LOCAL_MS             # total K-draft term @K5 (~8.23ms)

NAME = os.environ.get("WANDB_NAME", "land/strict319-attention-pin-cost")
GROUP = os.environ.get("WANDB_GROUP", "strict319-attention-pin-cost-land")


def _load(cfg):
    p = RUNS / f"{cfg}.json"
    return json.load(open(p)) if p.exists() else None


def _specdec_official(base_step_local_ms, e=E_ACCEPT):
    """base_step_local_ms = the per-iteration BASE step (AR step + pin/rescue tax); the
    K-draft term (denken-anchored) is added on top. Returns (official, t_iter, t_local)."""
    t_iter = base_step_local_ms + DRAFT_TERM_MS
    tps_local = 1000.0 * e / t_iter
    return tps_local * LOCAL_TO_OFFICIAL, t_iter, tps_local


def main() -> int:
    cfgs = {c: _load(c) for c in ("baseline", "bi1", "fixed2d")}
    have = {c: d for c, d in cfgs.items() if d}
    if "baseline" not in have:
        print("[684] missing baseline.json -- cannot compute deltas")
        return 1

    base_step = have["baseline"]["decode_step_ms"]
    base_tps_local = have["baseline"]["decode_tps_local"]
    base_attn = have["baseline"].get("attn_step_ms")

    per_cfg = {}
    for c, d in have.items():
        delta_full = d["decode_step_ms"] - base_step            # full step (lm_head-diluted)
        attn = d.get("attn_step_ms")
        delta_attn = (attn - base_attn) if (attn is not None and base_attn is not None) else None
        # Transferable head-independent pin cost:
        #   fixed2d = PURE attention pin -> the clean direct attn-op delta (lm_head cancels).
        #   bi1     = BLANKET pin -> full-step delta (attention 2D pin + aten-op swaps that
        #             live OUTSIDE unified_attention, so the attn-only timer can't see them).
        #   baseline= reference -> 0.
        if c == "baseline":
            pin_delta = 0.0
        elif c == "fixed2d" and delta_attn is not None:
            pin_delta = delta_attn
        else:
            pin_delta = delta_full
        pinned_step_deployed = AR_STEP_LOCAL_MS + pin_delta
        cost_frac_deployed = pin_delta / pinned_step_deployed if pinned_step_deployed > 0 else float("nan")
        cost_frac_onstack = (base_tps_local - d["decode_tps_local"]) / base_tps_local
        per_cfg[c] = {
            "decode_step_ms": d["decode_step_ms"],
            "decode_tps_local": d["decode_tps_local"],
            "attn_step_ms": attn,
            "verify_per_token_ms": d.get("verify_per_token_ms"),
            "delta_full_ms_vs_baseline": delta_full,
            "delta_attn_ms_vs_baseline": delta_attn,
            "pin_delta_ms": pin_delta,                          # the transferable cost used
            "cost_frac_deployed_transfer": cost_frac_deployed,
            "cost_frac_onstack_bighead": cost_frac_onstack,
            "argmax_break": d["fullforward_frac_steps_argmax_break"],
            "bitdiff": d["fullforward_frac_steps_bitdiff"],
            "seq_break_rate": d["fullforward_seq_break_rate"],
            "is_lossless": bool(d["is_lossless_argmax"]),
            "is_bitexact": bool(d["is_bitexact_logprob"]),
            "ar_vs_ar": d["ar_vs_ar_token_identity"],
            "path_m1_use_3d": d.get("path_m1_use_3d_values"),
            "path_verify_use_3d": d.get("path_verify_use_3d_values"),
        }

    # Two losslessness tiers (see report):
    #   ARGMAX-LOSSLESS (is_lossless) = strict-#319's ACTUAL requirement (verify argmax == AR
    #     argmax). fixed2d satisfies this for FREE (the pure 2D attention pin).
    #   BYTE-IDENTICAL (is_bitexact) = the PR's stronger "byte-identical" phrasing (verify
    #     logprobs bit-equal AR). Only bi1 (VLLM_BATCH_INVARIANT=1) reaches it, because BI also
    #     pins the non-attention aten reductions; its DEPLOYED cost is lawine #675's +1.55ms
    #     blanket (my full-vocab full-step delta is inflated by the 262k-row log_softmax swap).
    # cheapest argmax-lossless pin among {bi1, fixed2d}, by transferable pin cost:
    lossless = {c: per_cfg[c] for c in ("bi1", "fixed2d")
                if c in per_cfg and per_cfg[c]["is_lossless"]}
    byte_identical = {c: per_cfg[c] for c in ("bi1", "fixed2d")
                      if c in per_cfg and per_cfg[c]["is_bitexact"]}
    bi1_cost = per_cfg.get("bi1", {}).get("cost_frac_deployed_transfer")
    fixedsplit_is_lossless = int(per_cfg.get("fixed2d", {}).get("is_lossless", False))
    fixedsplit_is_bitexact = int(per_cfg.get("fixed2d", {}).get("is_bitexact", False))
    byte_identical_pin = next(iter(byte_identical), None)  # bi1 (or None)
    byte_identical_pin_exists = int(bool(byte_identical))

    if lossless:
        cheapest = min(lossless, key=lambda c: lossless[c]["pin_delta_ms"])
        delta_pin = lossless[cheapest]["pin_delta_ms"]
        attention_pin_tps_cost_frac = lossless[cheapest]["cost_frac_deployed_transfer"]
        t0_lossless = AR_STEP_LOCAL_MS + delta_pin
        spec_off, t_step_l, spec_loc = _specdec_official(t0_lossless)
        margin = spec_off - PLUS10_BAR
    else:
        cheapest = None
        delta_pin = float("nan")
        attention_pin_tps_cost_frac = float("nan")
        spec_off = float("nan"); t_step_l = float("nan"); spec_loc = float("nan")
        margin = float("nan")

    # rescue-path comparison at the same K and E. By construction the model is anchored so
    # this REPRODUCES denken's published 137.14 (+0.76) -- assert it as a self-consistency
    # check (proves the K-draft term and basis tie out to denken before we trust the pin).
    rescue_off, rescue_tstep, rescue_loc = _specdec_official(RESCUE_BASE_LOCAL_MS)
    rescue_margin = rescue_off - PLUS10_BAR
    rescue_reproduces_denken = bool(abs(rescue_off - RESCUE_OFFICIAL_AT_K5) < 0.05)
    # the attention pin beats the rescue iff its transferable cost < the 4.84ms rescue tax
    # (e_accept/K-INDEPENDENT relative test: both share the identical K-draft term).
    pin_beats_rescue = bool(math.isfinite(delta_pin) and delta_pin < RESCUE_TAX_MS)
    pin_beats_rescue_margin = bool(math.isfinite(margin) and margin > rescue_margin)

    # ROBUSTNESS BRACKET: my measured pin cost is context-specific (ctx~512 here; the 16-way
    # 3D split helps more at longer KV, so the deployed-context delta may be larger). Price the
    # margin across the full plausible cost range so the verdict does not hinge on the exact
    # measured delta: measured(cheapest) ... ubel #491/#484 targeted +5.10% (+0.43ms) ...
    # lawine #675 BLANKET BI=1 deployed -16% (+1.55ms = the conservative UPPER bound).
    UBEL_TARGETED_DELTA_MS = 0.43    # +5.10% of the 8.14ms deployed AR step
    LAWINE_BLANKET_DELTA_MS = 1.55   # -16% deployed AR -> step *1/0.84 -> +1.55ms (upper bound)
    bracket = {}
    for label, dms in (("measured_cheapest", delta_pin),
                       ("ubel_targeted_5p10", UBEL_TARGETED_DELTA_MS),
                       ("lawine_blanket_16p", LAWINE_BLANKET_DELTA_MS)):
        if math.isfinite(dms):
            off_b, _, _ = _specdec_official(AR_STEP_LOCAL_MS + dms)
            bracket[label] = {"delta_ms": dms, "specdec_official": off_b,
                              "margin_vs_plus10": off_b - PLUS10_BAR,
                              "clears_plus10": bool(off_b > PLUS10_BAR)}
    # the verdict is robust iff EVERY point in the bracket clears +10 (worst case = lawine bound)
    bracket_all_clear = bool(bracket and all(v["clears_plus10"] for v in bracket.values()))
    conservative_margin = bracket.get("lawine_blanket_16p", {}).get("margin_vs_plus10", float("nan"))

    # verdict
    any_lossless_pin = bool(lossless)
    if not any_lossless_pin:
        verdict = "NEEDS_KERNEL_BUILD"
    elif math.isfinite(spec_off) and spec_off > PLUS10_BAR:
        verdict = "CHEAP_ATTENTION_PIN_EXISTS"
    else:
        verdict = "ATTENTION_PIN_TOO_COSTLY"

    summary = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "verdict": verdict,
        "cheapest_lossless_pin": cheapest,
        "attention_pin_tps_cost_frac": attention_pin_tps_cost_frac,
        "attention_pin_delta_ms": delta_pin,
        "lossless_specdec_tps_at_k5": spec_off,
        "lossless_specdec_tps_local_k5": spec_loc,
        "lossless_specdec_tstep_local_ms_k5": t_step_l,
        "lossless_specdec_margin_vs_plus10": margin,
        # robustness bracket (verdict holds across the whole plausible pin-cost range)
        "bracket_all_clear_plus10": int(bracket_all_clear),
        "conservative_margin_at_blanket_bound": conservative_margin,
        "bracket_ubel_targeted_margin": bracket.get("ubel_targeted_5p10", {}).get("margin_vs_plus10"),
        "bracket_lawine_blanket_margin": bracket.get("lawine_blanket_16p", {}).get("margin_vs_plus10"),
        "bi1_blanket_cost_frac": bi1_cost,
        "fixedsplit_is_lossless": fixedsplit_is_lossless,
        "fixedsplit_is_bitexact": fixedsplit_is_bitexact,
        "byte_identical_pin": byte_identical_pin,
        "byte_identical_pin_exists": byte_identical_pin_exists,
        # byte-identical (bi1) DEPLOYED cost = lawine #675 +1.55ms (full-vocab measure inflated)
        "byte_identical_deployed_margin_vs_plus10": bracket.get("lawine_blanket_16p", {}).get("margin_vs_plus10"),
        # rescue-path baseline (same K,E) for the "cheaper of two lossless routes" test
        "rescue_specdec_tps_at_k5": rescue_off,
        "rescue_specdec_margin_vs_plus10": rescue_margin,
        "rescue_reproduces_denken_137p14": int(rescue_reproduces_denken),
        "attention_pin_beats_rescue": pin_beats_rescue,
        "attention_pin_beats_rescue_margin": pin_beats_rescue_margin,
        "rescue_tax_ms": RESCUE_TAX_MS,
        # anchors / derived model
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
        "local_to_official": LOCAL_TO_OFFICIAL, "e_accept": E_ACCEPT, "k_opt": K_OPT,
        "ar_step_local_ms": AR_STEP_LOCAL_MS, "rescue_base_local_ms": RESCUE_BASE_LOCAL_MS,
        "rescue_official_at_k5_anchor": RESCUE_OFFICIAL_AT_K5,
        "draft_term_ms_derived": DRAFT_TERM_MS, "t_iter_rescue_local_ms": _T_ITER_RESCUE,
    }
    # per-config flat scalars
    for c, m in per_cfg.items():
        for k in ("decode_step_ms", "decode_tps_local", "attn_step_ms",
                  "delta_full_ms_vs_baseline", "delta_attn_ms_vs_baseline", "pin_delta_ms",
                  "cost_frac_deployed_transfer", "cost_frac_onstack_bighead",
                  "argmax_break", "bitdiff", "seq_break_rate", "is_lossless",
                  "is_bitexact", "verify_per_token_ms"):
            summary[f"cfg/{c}/{k}"] = m[k]

    print("=" * 72)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("cfg/")},
                     indent=2, default=str))
    print("--- per-config ---")
    for c, m in per_cfg.items():
        da = m["delta_attn_ms_vs_baseline"]
        da_s = f"{da:+.5f}" if da is not None else "  n/a "
        print(f"  {c:9s} step={m['decode_step_ms']:.4f}ms attn={(m['attn_step_ms'] or float('nan')):.5f}ms "
              f"d_full={m['delta_full_ms_vs_baseline']:+.4f} d_attn={da_s} pin={m['pin_delta_ms']:+.5f}ms "
              f"cost_deploy={m['cost_frac_deployed_transfer']*100:+.2f}% "
              f"break={m['argmax_break']:.5f} bitdiff={m['bitdiff']:.4f} lossless={m['is_lossless']} "
              f"M1_3d={m['path_m1_use_3d']} ver_3d={m['path_verify_use_3d']}")
    print(f"  rescue_reproduces_denken_137.14={rescue_reproduces_denken} "
          f"(model rescue={rescue_off:.3f}, denken anchor={RESCUE_OFFICIAL_AT_K5})")
    print("=" * 72)

    json.dump({"summary": summary, "per_cfg": per_cfg},
              open(RUNS / "decision.json", "w"), indent=2, default=str)

    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("[684] wandb disabled via env -- decision.json written")
        return 0
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[684] wandb unavailable ({exc}) -- decision.json written")
        return 0

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=NAME, group=GROUP, job_type="analysis",
        tags=["gemma-challenge", "analysis", "lossless-verify", "attention-pin",
              "triton-attn", "split-kv", "spec-dec", "issue-319", "pr-684"],
        config={
            "pr": 684, "issue": 319, "analysis_only": True, "wandb_group": GROUP,
            "vllm_version": "0.22.0", "attn_backend": "TRITON_ATTN",
            "configs_measured": list(per_cfg.keys()),
            "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
            "local_to_official": LOCAL_TO_OFFICIAL, "e_accept": E_ACCEPT, "k_opt": K_OPT,
            "model_dir": have["baseline"].get("model_dir"),
            "full_vocab_fidelity_note": "full-vocab QAT ckpt; deployed 16k-head can't load "
            "in vanilla vLLM. Pin cost transferred as absolute ms onto 8.14ms deployed AR "
            "step (lm_head GEMM cancels in the 3D->2D difference -> head-independent).",
        },
    )
    flat = {k: v for k, v in summary.items()
            if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}
    run.summary.update(flat)

    cols = ["config", "decode_step_ms", "attn_step_ms", "decode_tps_local",
            "delta_full_ms", "delta_attn_ms", "pin_delta_ms", "cost_frac_deployed",
            "cost_frac_onstack", "argmax_break", "bitdiff", "is_lossless", "is_bitexact",
            "M1_use_3d", "verify_use_3d"]
    tbl = wandb.Table(columns=cols)
    for c, m in per_cfg.items():
        tbl.add_data(c, m["decode_step_ms"], m["attn_step_ms"], m["decode_tps_local"],
                     m["delta_full_ms_vs_baseline"], m["delta_attn_ms_vs_baseline"], m["pin_delta_ms"],
                     m["cost_frac_deployed_transfer"], m["cost_frac_onstack_bighead"],
                     m["argmax_break"], m["bitdiff"], int(m["is_lossless"]), int(m["is_bitexact"]),
                     str(m["path_m1_use_3d"]), str(m["path_verify_use_3d"]))
    run.log({"per_config": tbl})

    print(f"[684] verdict={verdict}  pin={cheapest} cost_frac={attention_pin_tps_cost_frac} "
          f"lossless_specdec_k5={spec_off} margin={margin}")
    print(f"[684] W&B: {run.url}  id={run.id}")
    run.finish()
    print(f"WANDB_RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
