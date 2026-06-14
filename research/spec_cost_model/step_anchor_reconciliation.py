"""Step-anchor stack reconciliation (PR #168).

Collapse the FOUR measured/derived step-time anchors for the depth-9 tree decode
step into ONE launch-REALIZED step for descent-only and ONE for both-bugs, with the
roofline<->overlap spread as the residual uncertainty band. Pure-analytic CPU-only
synthesis of committed outputs (#136, #154, #161); it IMPORTS them and does NOT
re-derive any measurement.

The four anchors and their regime:

  1.2127  roofline ideal-overlap (graphed floor, attention launch idle hidden)   [#136]
  1.2182  measured idle-hidden overlap (realistic eager star-attn path)          [#136 / #161]
  1.2047  scatter+LP-avoidance reduced decode-path (-1.108% @ the clear-500 bar)  [#154]
  1.2182  both-bugs-neutral (the depth-1 spine adds 0 per-step cost)             [#161]

Reconciliation logic
--------------------
* roofline (1.2127) and measured-overlap (1.2182) are SUBSTITUTES -- two estimates of
  the SAME physical step. The +0.45% is the REAL exposed star-attn launch idle that
  survives realistic GEMM overlap (#136: 43.3 us/step). Under PRECACHE_BENCH=1 the
  timed window is pure-decode and the served fa2sw stack runs compute_logits / star
  attention EAGERLY (outside the CUDA graph), so it PAYS that idle -> the served
  reality is the measured-overlap 1.2182. roofline 1.2127 is the optimistic floor a
  fully-graphed attention build would recover (blocker #2, NOT shipped). => the launch
  quotes ONE step (overlap 1.2182) and carries roofline<->overlap as the band.
* #154's 1.2047 is the same step with an AVOIDABLE tax REMOVED (decode-path
  [M,262144] scatter + sampling LogitsProcessor). It is a CONDITIONAL substitute: it
  applies ONLY if the argmax-only decode build ships (the compute_logits
  token-selection vs prompt_logprobs branch). That build has NOT shipped, so it does
  NOT lower the launch-realized step; it is reported as a separate not-yet-realized
  lane (+4.3..5.6 TPS, bar 4.862 -> 4.808..4.820).
* #161's 1.2182 confirms bug-1 (depth-1 spine) is an ADDITIVE component of magnitude
  EXACTLY 0 (measured marginal accept-prep device-busy -0.031 us, step-neutral). So
  both-bugs step == descent-only step == the measured-overlap 1.2182.

No HF Job / submission / served-file change. BASELINE stays 481.53 (PPL 2.3777).
Adds 0 TPS -- a step-denominator closeout that replaces the four-anchor 1.1% spread
in fern #155's launch packet with ONE defensible number + a 0.45% residual band.
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
M136_PATH = os.path.join(ROOT, "fp32_star_steptime_measured_anchor.json")
M154_PATH = os.path.join(ROOT, "step_denominator_reduction_audit.json")
M161_PATH = os.path.join(os.path.dirname(ROOT), "both_bugs_step_cost", "both_bugs_step_cost.json")
OUT_PATH = os.path.join(ROOT, "step_anchor_reconciliation.json")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _official(k_cal, e_t, step, tau=1.0):
    """official = K_cal * (E[T]/step) * tau."""
    return k_cal * (e_t / step) * tau


def _bar(k_cal, step, target=500.0, tau=1.0):
    """E[T] s.t. official == target at this step:  target = K_cal*E[T]/step*tau."""
    return target * step / (k_cal * tau)


def _row_at(block, e_t, tol=1e-6):
    for r in block["rows"]:
        if abs(r["E_T"] - e_t) <= tol:
            return r
    raise KeyError(f"no row at E_T={e_t} in {block.get('label')}")


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def reconcile():
    m136 = _load(M136_PATH)
    m154 = _load(M154_PATH)
    m161 = _load(M161_PATH)

    # ---- one source of truth per constant (imported, cross-checked) ----
    k_cal = m161["anchors"]["k_cal"]
    tau = m136["anchors"]["tau_central"]  # 1.0, folded
    target = m154["projection_constants"]["target"]  # 500.0

    # cross-check K_cal agreement across the three imports (no re-derivation)
    k_cal_136 = m136["anchors"]["k_cal"]
    k_cal_154 = m154["projection_constants"]["K_cal"]
    k_cal_agreement = (
        abs(k_cal - k_cal_136) < 1e-9 and abs(k_cal - k_cal_154) < 1e-9
    )

    # ---- the four anchors (full precision) ----
    roofline_step = m136["anchors"]["step_wstar_depth9_roofline"]      # 1.2127
    overlap_step = m136["step1_measured_step"]["step_overlap_central"]  # 1.2182 (measured)
    anchor_136_rounded = m161["anchors"]["measured_step_136"]           # 1.2182
    delta_overlap_vs_roofline_pct = m136["step1_measured_step"]["delta_vs_roofline_pct"]
    idle_overlap_us = m136["partD2_overlap_hidden"]["exposed_idle_overlap_us"]

    # both-bugs neutrality (#161): bug-1 adds 0 -> both-bugs step == descent step
    both_bugs_step_pinned = m161["propagation"]["both_bugs_step_pinned"]      # 1.2182
    both_bugs_step_delta_pct = m161["propagation"]["both_bugs_step_delta_pct"]  # 0.0
    marginal_bug1_busy_us = m161["propagation"]["marginal_busy_us"]            # -0.031 us

    # #154 reduced-step (scatter+LP avoidance) at the clear-500 bar (E[T]=4.862)
    red_real_pct_at_bar = m154["recoverable_step_pct_realistic"]  # 1.1079 %
    red_cons_pct_at_bar = m154["recoverable_step_pct"]            # 0.8573 %
    step_154_realistic = overlap_step * (1.0 - red_real_pct_at_bar / 100.0)   # ~1.2047
    step_154_conservative = overlap_step * (1.0 - red_cons_pct_at_bar / 100.0)  # ~1.2078

    # ---- E[T] anchors (imported from #161) ----
    e_t_descent = m161["anchors"]["e_t_descent_only"]  # 5.0564
    e_t_both = m161["anchors"]["e_t_both_bugs"]         # 5.2070

    # =====================================================================
    # STEP 1 -- enumerate the four anchors with provenance + regime
    # =====================================================================
    anchors = [
        {
            "value": roofline_step,
            "name": "roofline_ideal_overlap",
            "provenance": "#136 step_wstar_depth9_roofline (graphed floor; attn launch idle hidden)",
            "regime": "SUBSTITUTE",
            "role": "optimistic floor of the SAME physical step; recovered only by a fully-graphed attn build (blocker #2, NOT shipped)",
        },
        {
            "value": overlap_step,
            "name": "measured_idle_hidden_overlap",
            "provenance": "#136 step_overlap_central / measured_depth9_step_time; reconfirmed #161",
            "regime": "SUBSTITUTE",
            "role": f"LAUNCH-REALIZED served reality: eager star-attn pays +{delta_overlap_vs_roofline_pct:.3f}% exposed launch idle ({idle_overlap_us:.1f} us/step) that survives GEMM overlap",
        },
        {
            "value": step_154_realistic,
            "name": "scatter_lp_reduced_decode_path",
            "provenance": f"#154 recoverable_step_pct_realistic={red_real_pct_at_bar:.4f}% applied to overlap @ bar E[T]=4.862",
            "regime": "CONDITIONAL_SUBSTITUTE",
            "role": "same step with avoidable scatter+LP tax REMOVED; applies ONLY if the argmax-only decode build ships (NOT in the current served path)",
        },
        {
            "value": both_bugs_step_pinned,
            "name": "both_bugs_neutral",
            "provenance": f"#161 both_bugs_step_pinned (marginal bug-1 accept-prep busy {marginal_bug1_busy_us:.3f} us, step-neutral)",
            "regime": "ADDITIVE_ZERO",
            "role": "bug-1 (depth-1 spine) is an additive step component of magnitude EXACTLY 0 -> both-bugs step == descent step",
        },
    ]

    # =====================================================================
    # STEP 2 -- compose the ONE launch-realized step (descent / both-bugs)
    # =====================================================================
    # Served reality under PRECACHE_BENCH=1 (pure-decode window, eager star-attn):
    #   launch-realized step = measured-overlap 1.2182.
    #   bug-1 adds 0 (#161) -> both-bugs step == descent step.
    #   roofline 1.2127 is the optimistic band edge (idle fully hidden).
    launch_realized_step_descent = overlap_step
    launch_realized_step_both_bugs = overlap_step + 0.0  # + bug1_additive(=0), #161

    band = {
        "lo_step_roofline": roofline_step,   # optimistic (higher TPS)
        "hi_step_overlap": overlap_step,     # realized/conservative (lower TPS)
        "half_width_pct": delta_overlap_vs_roofline_pct / 2.0,
        "full_spread_pct": delta_overlap_vs_roofline_pct,
        "interpretation": "roofline<->overlap substitutes; ONE step quoted (overlap), the spread is the residual step-anchor uncertainty (matches fern #155's 0.5% half-width).",
    }

    # =====================================================================
    # STEP 3 -- propagate via official = K_cal * (E[T]/step) * tau
    # =====================================================================
    def propagate(e_t, label):
        off_overlap = _official(k_cal, e_t, overlap_step, tau)
        off_roofline = _official(k_cal, e_t, roofline_step, tau)
        return {
            "label": label,
            "e_t": e_t,
            "official_launch_realized_overlap": off_overlap,
            "official_optimistic_roofline": off_roofline,
            "tps_band_pm": (off_roofline - off_overlap),  # +TPS upside if roofline recovered
        }

    prop_descent = propagate(e_t_descent, "descent_only")
    prop_both = propagate(e_t_both, "both_bugs")

    # clear-500 bar at the reconciled (overlap) step and at the roofline edge
    bar_overlap = _bar(k_cal, overlap_step, target, tau)        # 4.862 (operative)
    bar_roofline = _bar(k_cal, roofline_step, target, tau)      # 4.841

    # CONDITIONAL #154 lane (separate, not-yet-realized) -- import #154's own table
    real_tree = m154["propagation"]["realistic_M32_tree"]
    cons_tree = m154["propagation"]["conservative_M32_tree"]
    bar_154_realistic = _row_at(real_tree, m154["projection_constants"]["clear_500_bar"])["clear_500_bar_new"]
    bar_154_conservative = _row_at(cons_tree, m154["projection_constants"]["clear_500_bar"])["clear_500_bar_new"]
    # conditional TPS if #154 ships, at the descent / both-bugs E[T] (import #154 rows)
    cond_descent_real = _row_at(real_tree, 5.0564)["official_new_tps"]
    cond_descent_cons = _row_at(cons_tree, 5.0564)["official_new_tps"]
    cond_both_real = _row_at(real_tree, 5.207)["official_new_tps"]
    cond_both_cons = _row_at(cons_tree, 5.207)["official_new_tps"]

    conditional_154 = {
        "applies_in_served_path": False,
        "gating_build": "compute_logits token-selection vs prompt_logprobs branch (argmax-only decode); NOT shipped",
        "reduced_step_realistic": step_154_realistic,
        "reduced_step_conservative": step_154_conservative,
        "bar_realistic": bar_154_realistic,
        "bar_conservative": bar_154_conservative,
        "descent_tps_if_shipped": [cond_descent_cons, cond_descent_real],
        "both_bugs_tps_if_shipped": [cond_both_cons, cond_both_real],
        "dtps_at_bar_band": [
            _row_at(cons_tree, m154["projection_constants"]["clear_500_bar"])["dtps"],
            _row_at(real_tree, m154["projection_constants"]["clear_500_bar"])["dtps"],
        ],
    }

    # =====================================================================
    # STEP 4 -- self-test (PRIMARY)
    # =====================================================================
    # (a) reconciled step reproduces #136's 1.2182 within tolerance
    delta_a_pct = abs(launch_realized_step_both_bugs - anchor_136_rounded) / anchor_136_rounded * 100.0
    check_a = delta_a_pct < 0.10

    # (b) descent ~522 and both-bugs ~535-538 across the roofline<->overlap band
    check_b = (
        519.0 <= prop_descent["official_launch_realized_overlap"] <= 523.0
        and 519.0 <= prop_descent["official_optimistic_roofline"] <= 523.0
        and 535.0 <= prop_both["official_launch_realized_overlap"] <= 538.0
        and 535.0 <= prop_both["official_optimistic_roofline"] <= 538.0
    )

    # (c) applying #154's reduction lowers the clear-500 bar to 4.808-4.820
    check_c = (
        4.805 <= bar_154_realistic <= 4.812
        and 4.815 <= bar_154_conservative <= 4.823
    )

    # (d) roofline/overlap substitutes are NOT double-counted: the reconciled step is
    #     ONE substitute (overlap), not roofline+overlap (which would be ~2.43), and the
    #     band is a SPREAD (|overlap-roofline|/overlap == #136 delta), not an added cost.
    recon_is_single_substitute = abs(launch_realized_step_both_bugs - overlap_step) < 1e-9
    not_summed = launch_realized_step_both_bugs < (roofline_step + overlap_step) - 1e-6
    # spread expressed #136-style (relative to the roofline floor) must reproduce
    # #136's delta_vs_roofline_pct -- i.e. the band is that SPREAD, not an added cost.
    spread_pct_vs_roofline = (overlap_step - roofline_step) / roofline_step * 100.0
    spread_matches_136 = abs(spread_pct_vs_roofline - delta_overlap_vs_roofline_pct) < 1e-6
    check_d = recon_is_single_substitute and not_summed and spread_matches_136

    self_test = {
        "check_a_reproduces_136_1p2182": {
            "passes": bool(check_a),
            "reconciled_step": launch_realized_step_both_bugs,
            "anchor_136": anchor_136_rounded,
            "delta_pct": delta_a_pct,
            "tol_pct": 0.10,
        },
        "check_b_descent_522_both_535_538": {
            "passes": bool(check_b),
            "descent_overlap": prop_descent["official_launch_realized_overlap"],
            "descent_roofline": prop_descent["official_optimistic_roofline"],
            "both_overlap": prop_both["official_launch_realized_overlap"],
            "both_roofline": prop_both["official_optimistic_roofline"],
        },
        "check_c_154_lowers_bar_4p808_4p820": {
            "passes": bool(check_c),
            "bar_realistic": bar_154_realistic,
            "bar_conservative": bar_154_conservative,
        },
        "check_d_substitutes_not_double_counted": {
            "passes": bool(check_d),
            "reconciled_is_single_substitute": bool(recon_is_single_substitute),
            "not_summed": bool(not_summed),
            "spread_matches_136_delta": bool(spread_matches_136),
            "reconciled_step": launch_realized_step_both_bugs,
            "roofline_plus_overlap_NOT_used": roofline_step + overlap_step,
            "spread_pct_vs_roofline": spread_pct_vs_roofline,
            "delta_136_vs_roofline_pct": delta_overlap_vs_roofline_pct,
        },
    }
    all_pass = all(v["passes"] for v in self_test.values()) and k_cal_agreement

    # =====================================================================
    # STEP 5 -- hand-off (the single defensible step the launch quotes)
    # =====================================================================
    handoff = {
        "launch_should_quote_step": launch_realized_step_both_bugs,  # 1.2182
        "rationale": "measured-overlap is the served reality under PRECACHE_BENCH=1 (eager star-attn pays the +0.45% exposed launch idle). bug-1 adds 0 (#161) so descent-only and both-bugs share it.",
        "official_descent_only": prop_descent["official_launch_realized_overlap"],
        "official_both_bugs": prop_both["official_launch_realized_overlap"],
        "tps_uncertainty_band": {
            "source": f"roofline<->overlap substitutes (0.45% step spread = +{prop_both['tps_band_pm']:.2f} TPS at the 537 level)",
            "descent_only": [prop_descent["official_launch_realized_overlap"], prop_descent["official_optimistic_roofline"]],
            "both_bugs": [prop_both["official_launch_realized_overlap"], prop_both["official_optimistic_roofline"]],
        },
        "operative_clear_500_bar": bar_overlap,
        "conditional_154_lane": f"NOT in the launch-realized step; if the argmax-only decode build ships -> bar {bar_154_realistic:.3f}..{bar_154_conservative:.3f}, +4.3..5.6 TPS",
        "replaces": "the four-anchor 1.1% spread in fern #155's launch packet (default step 1.2182) with ONE number + a 0.45% band; confirms the 0.5% step-anchor half-width #155 already carries.",
    }

    out = {
        "pr": 168,
        "scope": "Pure-analytic CPU-only synthesis of #136/#154/#161. No HF Job / submission / served-file change. BASELINE stays 481.53. Adds 0 TPS.",
        "imports": {
            "136": os.path.basename(M136_PATH),
            "154": os.path.basename(M154_PATH),
            "161": os.path.basename(M161_PATH),
        },
        "constants": {
            "k_cal": k_cal,
            "k_cal_agreement_across_imports": bool(k_cal_agreement),
            "tau": tau,
            "target": target,
            "e_t_descent_only": e_t_descent,
            "e_t_both_bugs": e_t_both,
        },
        "step1_anchors": anchors,
        "step2_launch_realized_step": {
            "descent_only": launch_realized_step_descent,
            "both_bugs": launch_realized_step_both_bugs,
            "band": band,
            "both_bugs_minus_descent": launch_realized_step_both_bugs - launch_realized_step_descent,
            "both_bugs_step_delta_pct_161": both_bugs_step_delta_pct,
        },
        "step3_propagation": {
            "descent_only": prop_descent,
            "both_bugs": prop_both,
            "clear_500_bar_overlap": bar_overlap,
            "clear_500_bar_roofline": bar_roofline,
            "conditional_154": conditional_154,
        },
        "step4_self_test": self_test,
        "step5_handoff": handoff,
        "primary_metric": {
            "name": "step_reconciliation_self_test_passes",
            "value": int(all_pass),
        },
        "test_metric": {
            "name": "launch_realized_step_both_bugs",
            "value": launch_realized_step_both_bugs,
        },
    }

    # NaN-clean guard over every numeric leaf
    def _walk(x, path="root"):
        bad = []
        if isinstance(x, dict):
            for k, v in x.items():
                bad += _walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                bad += _walk(v, f"{path}[{i}]")
        elif isinstance(x, float) and not math.isfinite(x):
            bad.append(path)
        return bad

    nan_paths = _walk(out)
    out["metrics_nan_clean"] = int(len(nan_paths) == 0)
    out["nan_offenders"] = nan_paths
    return out


def log_wandb(out, group):
    try:
        import wandb
    except Exception as e:  # never lose the completed synthesis to a logging import error
        print("WANDB_SKIPPED import:", e)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY"),
            group=group,
            name="lawine/launch-step-reconciliation",
            job_type="analysis",
            config={
                "pr": 168,
                "scope": out["scope"],
                "k_cal": out["constants"]["k_cal"],
                "tau": out["constants"]["tau"],
                "target": out["constants"]["target"],
                "e_t_descent_only": out["constants"]["e_t_descent_only"],
                "e_t_both_bugs": out["constants"]["e_t_both_bugs"],
                "no_hf_job": True,
                "no_served_file_change": True,
                "baseline_unchanged_481_53": True,
                "imports": out["imports"],
            },
        )
        st = out["step4_self_test"]
        prop = out["step3_propagation"]
        b = out["step2_launch_realized_step"]
        cond = prop["conditional_154"]
        wandb.summary.update({
            # PRIMARY / TEST as named by the PR
            "step_reconciliation_self_test_passes": out["primary_metric"]["value"],
            "launch_realized_step_both_bugs": out["test_metric"]["value"],
            "launch_realized_step_descent_only": b["descent_only"],
            # the reconciled band
            "step_band_lo_roofline": b["band"]["lo_step_roofline"],
            "step_band_hi_overlap": b["band"]["hi_step_overlap"],
            "step_band_full_spread_pct": b["band"]["full_spread_pct"],
            "both_bugs_minus_descent_step": b["both_bugs_minus_descent"],
            # propagated official TPS (launch-realized overlap + roofline edge)
            "official_descent_only_overlap": prop["descent_only"]["official_launch_realized_overlap"],
            "official_descent_only_roofline": prop["descent_only"]["official_optimistic_roofline"],
            "official_both_bugs_overlap": prop["both_bugs"]["official_launch_realized_overlap"],
            "official_both_bugs_roofline": prop["both_bugs"]["official_optimistic_roofline"],
            "both_bugs_tps_band_pm": prop["both_bugs"]["tps_band_pm"],
            "clear_500_bar_overlap": prop["clear_500_bar_overlap"],
            "clear_500_bar_roofline": prop["clear_500_bar_roofline"],
            # conditional #154 lane (separate, not realized)
            "cond154_applies_in_served_path": int(cond["applies_in_served_path"]),
            "cond154_bar_realistic": cond["bar_realistic"],
            "cond154_bar_conservative": cond["bar_conservative"],
            "cond154_both_bugs_tps_realistic": cond["both_bugs_tps_if_shipped"][1],
            "cond154_both_bugs_tps_conservative": cond["both_bugs_tps_if_shipped"][0],
            # self-test breakdown
            "self_test_a_passes": int(st["check_a_reproduces_136_1p2182"]["passes"]),
            "self_test_a_delta_pct": st["check_a_reproduces_136_1p2182"]["delta_pct"],
            "self_test_b_passes": int(st["check_b_descent_522_both_535_538"]["passes"]),
            "self_test_c_passes": int(st["check_c_154_lowers_bar_4p808_4p820"]["passes"]),
            "self_test_d_passes": int(st["check_d_substitutes_not_double_counted"]["passes"]),
            # health
            "metrics_nan_clean": out["metrics_nan_clean"],
            "k_cal_agreement_across_imports": int(out["constants"]["k_cal_agreement_across_imports"]),
        })

        # anchor table
        a_cols = ["name", "value", "regime", "provenance", "role"]
        a_rows = [[a["name"], a["value"], a["regime"], a["provenance"], a["role"]]
                  for a in out["step1_anchors"]]
        wandb.log({"step_anchors": wandb.Table(columns=a_cols, data=a_rows)})

        # self-test table
        s_cols = ["check", "passes"]
        s_rows = [[k, int(v["passes"])] for k, v in st.items()]
        wandb.log({"self_test_checks": wandb.Table(columns=s_cols, data=s_rows)})

        run_id = run.id
        run_url = run.url
        run.finish()
        print("WANDB_RUN_ID", run_id)
        print("WANDB_RUN_URL", run_url)
        return run_id
    except Exception as e:
        print("WANDB_SKIPPED runtime:", e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default="launch-step-reconciliation")
    args = ap.parse_args()

    out = reconcile()
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # human-readable summary
    b = out["step2_launch_realized_step"]
    prop = out["step3_propagation"]
    st = out["step4_self_test"]
    print("=" * 72)
    print("STEP-ANCHOR RECONCILIATION (PR #168)")
    print("=" * 72)
    print("LAUNCH-REALIZED STEP (descent-only) = %.6f" % b["descent_only"])
    print("LAUNCH-REALIZED STEP (both-bugs)    = %.6f  (band %.4f roofline <-> %.4f overlap)"
          % (b["both_bugs"], b["band"]["lo_step_roofline"], b["band"]["hi_step_overlap"]))
    print("official descent-only: %.2f (overlap) .. %.2f (roofline)"
          % (prop["descent_only"]["official_launch_realized_overlap"],
             prop["descent_only"]["official_optimistic_roofline"]))
    print("official both-bugs   : %.2f (overlap) .. %.2f (roofline)   band +-%.2f TPS"
          % (prop["both_bugs"]["official_launch_realized_overlap"],
             prop["both_bugs"]["official_optimistic_roofline"],
             prop["both_bugs"]["tps_band_pm"]))
    print("clear-500 bar (operative, overlap) = %.4f" % prop["clear_500_bar_overlap"])
    print("conditional #154 (NOT in served path): bar %.4f..%.4f, both-bugs %.2f..%.2f"
          % (prop["conditional_154"]["bar_realistic"],
             prop["conditional_154"]["bar_conservative"],
             prop["conditional_154"]["both_bugs_tps_if_shipped"][0],
             prop["conditional_154"]["both_bugs_tps_if_shipped"][1]))
    for k, v in st.items():
        print("  self-test %-44s %s" % (k, "PASS" if v["passes"] else "FAIL"))
    print("PRIMARY step_reconciliation_self_test_passes =", out["primary_metric"]["value"])
    print("TEST    launch_realized_step_both_bugs       = %.6f" % out["test_metric"]["value"])
    print("metrics_nan_clean =", out["metrics_nan_clean"], "| wrote", OUT_PATH)

    if args.wandb:
        log_wandb(out, args.wandb_group)


if __name__ == "__main__":
    main()
