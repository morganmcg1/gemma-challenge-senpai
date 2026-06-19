#!/usr/bin/env python3
"""PR #700 instr #2-3: activation-weighted output-impact re-rank, localization
stats, and byte-law footprint->TPS mapping.

impact = rel_div x act_norm  (per-weight g128-excess error  x  AIME activation
magnitude that flows through the module).  rel_div is reused verbatim from #695
(ref695/sqnr_probe.json); act_norm from activation_probe.py (act_norms.json).

For the impact distribution we recompute the SAME localization stats #695 logged
on the WEIGHT-space distribution (CV, top-1/8/16 energy share, energy_gini_like,
p90/p99) so the two axes are directly comparable.  Decision: does activation
reweighting CONCENTRATE the output damage (-> a tight activation-critical subset,
selective lever REVIVES) or stay DIFFUSE (~ #695 weight-space -> recipe closed)?

Byte-law (denken #676 / ubel #679 endpoints): TPS(f)=126.378/(1+0.06005*f),
f=body-param fraction put on the finer g32 grid.  Note TPS(f)<=126.378 for all
f>=0 (g32 is finer => MORE scale bytes => slower); a "speed-free" REVIVE therefore
requires the clearing footprint f to be ~0 (cost within anchor noise).

Robustness: the same localization is recomputed for three alternative magnitudes
(rel_div alone = weight-space relative baseline; diff_norm x act_norm = absolute
output-perturbation; diff_norm x act_norm / sqrt(in_dim) = isotropic-physical),
so the diffuse/concentrate verdict is shown robust to the impact definition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
REF695 = json.load(open(HERE / "ref695" / "sqnr_probe.json"))
ACT = json.load(open(HERE / "act_norms.json"))

TPS_FLOOR = 126.378
BYTE_K = 0.06005

# #679 AIME endpoints for the proportional-recovery coverage assumption.
AIME_G128 = 0.350   # f=0  (remove 0% of g128-excess energy)
AIME_G32 = 0.438    # f=1  (uniform g32, remove 100%)
AIME_BAR = 0.420    # #515 clearing bar
AIME_BASE = 0.4667  # vanilla bf16 base
# fraction of g128-excess (output-damage) energy that must be REMOVED to lift
# 0.350 -> 0.420 under "recovery proportional to energy removed" (#695 assumption).
CLEARING_COVERAGE = (AIME_BAR - AIME_G128) / (AIME_G32 - AIME_G128)  # ~0.795


def tps_at(f: float) -> float:
    return TPS_FLOOR / (1.0 + BYTE_K * f)


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def localization_stats(items):
    """items: list of dicts with 'mag' (>=0 magnitude) and 'params'.
    energy = mag**2 (mirrors #695 energy=||diff||^2). Returns the #695 stat set
    plus the greedy energy-per-param Pareto curve (best damage removed per byte)."""
    mags = [it["mag"] for it in items]
    n = len(mags)
    total_params = sum(it["params"] for it in items)
    energies = [m * m for m in mags]
    total_energy = sum(energies) or 1e-30

    mean = sum(mags) / n
    var = sum((m - mean) ** 2 for m in mags) / n
    std = var ** 0.5
    smags = sorted(mags)

    # top-k energy share (energy = mag^2), ranked by energy desc
    by_e = sorted(range(n), key=lambda i: energies[i], reverse=True)
    def topk(k):
        return sum(energies[by_e[i]] for i in range(min(k, n))) / total_energy
    topk_share = {k: topk(k) for k in (1, 4, 8, 16, 32)}

    # greedy by energy-per-param (best damage removed per speed-byte)
    order = sorted(range(n), key=lambda i: energies[i] / items[i]["params"], reverse=True)
    cum_p = cum_e = 0.0
    curve = []
    for i in order:
        cum_p += items[i]["params"]
        cum_e += energies[i]
        curve.append({
            "module": items[i]["module"],
            "f_param": cum_p / total_params,
            "f_energy": cum_e / total_energy,
            "tps_proj": tps_at(cum_p / total_params),
        })
    # f_param at energy-coverage thresholds
    def f_at(thr):
        for c in curve:
            if c["f_energy"] >= thr:
                return c["f_param"]
        return 1.0
    f_thresholds = {p: f_at(p) for p in (0.10, 0.25, 0.50, CLEARING_COVERAGE, 0.80, 0.90, 0.95)}

    # gini-like concentration: 2*(area under f_energy vs f_param - 0.5)
    xs = [0.0] + [c["f_param"] for c in curve]
    ys = [0.0] + [c["f_energy"] for c in curve]
    area = sum(0.5 * (ys[j] + ys[j - 1]) * (xs[j] - xs[j - 1]) for j in range(1, len(xs)))
    gini_like = 2.0 * (area - 0.5)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else float("nan"),
        "median": quantile(smags, 0.50),
        "min": smags[0],
        "max": smags[-1],
        "p90": quantile(smags, 0.90),
        "p99": quantile(smags, 0.99),
        "topk_energy_share": topk_share,
        "energy_gini_like": gini_like,
        "f_at_coverage": f_thresholds,
        "tps_at_coverage": {k: tps_at(v) for k, v in f_thresholds.items()},
        "_curve": curve,
    }


def main():
    ref_rows = {r["module"]: r for r in REF695["rows"]}
    act_rows = {r["module"]: r for r in ACT["rows"]}
    assert set(ref_rows) == set(act_rows), "module set mismatch ref695 vs act"

    rows = []
    for m in ref_rows:
        rr, ar = ref_rows[m], act_rows[m]
        rel_div = rr["rel_div"]
        diff_norm = rr["diff_norm"]
        params = rr["params"]
        in_dim = rr["in_dim"]
        act = ar["act_norm_prefill"]
        act_dec = ar["act_norm_decode"]
        rows.append({
            "module": m, "layer": rr["layer"], "proj": rr["proj"],
            "params": params, "in_dim": in_dim,
            "rel_div": rel_div, "diff_norm": diff_norm,
            "act_norm": act, "act_norm_decode": act_dec,
            "impact": rel_div * act,                         # PR PRIMARY
            "abs_out": diff_norm * act,                      # absolute output-perturb
            "phys_out": diff_norm * act / math.sqrt(in_dim), # isotropic-physical
        })

    def stats_for(key):
        return localization_stats([{"mag": r[key], "params": r["params"], "module": r["module"]} for r in rows])

    S_impact = stats_for("impact")     # PR primary
    S_reldiv = stats_for("rel_div")    # weight-space relative baseline (no activation)
    S_absout = stats_for("abs_out")    # absolute output-perturbation
    S_physout = stats_for("phys_out")  # isotropic-physical

    # #695 weight-space reference (absolute diff_norm energy) for the headline compare
    s695 = REF695["summary"]
    ref695_compare = {
        "rel_div_cv_695": s695["rel_div_std"] / s695["rel_div_mean"],
        "top16_energy_share_695_diffnorm": s695["topk_energy_share"]["16"],
        "energy_gini_like_695_diffnorm": s695["energy_gini_like"],
        "f_param_at_50pct_695": s695["f_param_at_50pct_energy"],
        "tps_at_50pct_695": s695["tps_at_50pct_energy"],
    }

    # ---- byte-law clearing footprint on the activation-weighted impact ranking ----
    clearing_f = S_impact["f_at_coverage"][CLEARING_COVERAGE]
    clearing_tps = tps_at(clearing_f)

    # ---- verdict ----
    impact_top16 = S_impact["topk_energy_share"][16]
    reldiv_top16 = S_reldiv["topk_energy_share"][16]
    # concentration relative to the weight-space relative baseline (isolates the
    # PURE activation reweighting effect on the same rel_div quantity) and to #695.
    concentrates = impact_top16 > 1.5 * reldiv_top16 and S_impact["energy_gini_like"] > 0.45
    # speed-free revival needs clearing footprint cost within anchor noise (~0.5 TPS).
    speed_free = clearing_tps >= (TPS_FLOOR - 0.5)
    if concentrates and speed_free:
        verdict = "ACTIVATION_LOCALIZED_SELECTIVE_REVIVES"
    elif (not concentrates) and S_impact["energy_gini_like"] < 0.40 and clearing_f > 0.30:
        verdict = "ACTIVATION_DIFFUSE_RECIPE_CLOSED"
    else:
        verdict = "ACTIVATION_PARTIAL"

    summary = {
        "verdict": verdict,
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # PRIMARY metric
        "activation_weighted_top16_energy_share": impact_top16,
        # TEST metric
        "activation_critical_tps_at_clearing_footprint": clearing_tps,
        "clearing_coverage_assumption": CLEARING_COVERAGE,
        "clearing_footprint_f_param": clearing_f,
        "tps_cost_vs_anchor": TPS_FLOOR - clearing_tps,
        "speed_free_unlock": int(speed_free),
        "activation_concentrates": int(concentrates),
        # impact distribution localization (the #695 stat set, activation axis)
        "impact_cv": S_impact["cv"],
        "impact_mean": S_impact["mean"],
        "impact_p90": S_impact["p90"],
        "impact_p99": S_impact["p99"],
        "impact_top1_energy_share": S_impact["topk_energy_share"][1],
        "impact_top8_energy_share": S_impact["topk_energy_share"][8],
        "impact_top16_energy_share": S_impact["topk_energy_share"][16],
        "impact_top32_energy_share": S_impact["topk_energy_share"][32],
        "impact_energy_gini_like": S_impact["energy_gini_like"],
        "impact_f_at_50pct_energy": S_impact["f_at_coverage"][0.50],
        "impact_tps_at_50pct_energy": S_impact["tps_at_coverage"][0.50],
        "impact_f_at_clearing": clearing_f,
        # side-by-side: weight-space (rel_div, no act) vs activation-weighted (impact)
        "reldiv_top16_energy_share": reldiv_top16,
        "reldiv_energy_gini_like": S_reldiv["energy_gini_like"],
        "reldiv_cv": S_reldiv["cv"],
        # robustness: absolute & physical output-perturbation localization
        "absout_top16_energy_share": S_absout["topk_energy_share"][16],
        "absout_energy_gini_like": S_absout["energy_gini_like"],
        "absout_f_at_clearing": S_absout["f_at_coverage"][CLEARING_COVERAGE],
        "absout_tps_at_clearing": S_absout["tps_at_coverage"][CLEARING_COVERAGE],
        "physout_top16_energy_share": S_physout["topk_energy_share"][16],
        "physout_energy_gini_like": S_physout["energy_gini_like"],
        "physout_f_at_clearing": S_physout["f_at_coverage"][CLEARING_COVERAGE],
        "physout_tps_at_clearing": S_physout["tps_at_coverage"][CLEARING_COVERAGE],
        # validity of the activation measurement
        "spearman_prefill_decode": ACT["meta"]["spearman_prefill_decode"],
        "n_calib_prompts": ACT["meta"]["n_prompts"],
        **ref695_compare,
    }

    # ---- validity diagnostics: what drives the concentration? ----
    imp2 = lambda r: r["impact"] ** 2
    tot_e = sum(imp2(r) for r in rows) or 1e-30
    ple_share = sum(imp2(r) for r in rows if "per_layer" in r["proj"]) / tot_e
    by_imp = sorted(rows, key=imp2, reverse=True)
    top1_share = imp2(by_imp[0]) / tot_e
    top3_share = sum(imp2(r) for r in by_imp[:3]) / tot_e
    # exclude per-layer-embedding pathway -> does the MAIN transformer concentrate too?
    main_rows = [r for r in rows if "per_layer" not in r["proj"]]
    S_main = localization_stats([{"mag": r["impact"], "params": r["params"], "module": r["module"]} for r in main_rows])
    # clearing subset composition at the headline coverage
    clearing_curve = []
    for c in S_impact["_curve"]:
        clearing_curve.append(c)
        if c["f_energy"] >= CLEARING_COVERAGE:
            break
    clearing_mods = {c["module"] for c in clearing_curve}
    clear_comp = {}
    for r in rows:
        if r["module"] in clearing_mods:
            clear_comp[r["proj"]] = clear_comp.get(r["proj"], 0) + 1
    summary.update({
        "ple_pathway_impact_energy_share": ple_share,
        "top1_module_impact_share": top1_share,
        "top3_module_impact_share": top3_share,
        "top1_module": by_imp[0]["module"],
        "exclude_ple_top16_energy_share": S_main["topk_energy_share"][16],
        "exclude_ple_energy_gini_like": S_main["energy_gini_like"],
        "exclude_ple_clearing_f": S_main["f_at_coverage"][CLEARING_COVERAGE],
        "exclude_ple_clearing_tps": S_main["tps_at_coverage"][CLEARING_COVERAGE],
        "clearing_subset_n_modules": len(clearing_mods),
        "clearing_subset_composition": clear_comp,
        "validity_caveat": (
            "impact = rel_div x INPUT-activation-norm is a first-order LOCAL proxy "
            "for output damage; it does NOT measure the realized effect on AIME "
            "logits/answers (which depends on downstream propagation + output "
            "sensitivity). Concentration is genuine on the proxy and robust "
            "(survives excluding the per-layer-embedding pathway; q/k/v in early/"
            "late layers concentrate too), but the AIME-relevance of the "
            "activation-critical subset is a HYPOTHESIS the proxy generates, not a "
            "verified fact. Disposition: GPU-approval-gated HELD build to "
            "empirically test selective-g32 on the subset -> AIME recovery; NOT a fire."
        ),
    })

    # top modules by impact (the activation-critical candidates)
    top_impact = sorted(rows, key=lambda r: r["impact"], reverse=True)[:16]
    summary["top16_modules_by_impact"] = [
        {"module": r["module"], "impact": r["impact"], "rel_div": r["rel_div"],
         "act_norm": r["act_norm"], "params": r["params"], "proj": r["proj"], "layer": r["layer"]}
        for r in top_impact
    ]

    out = {
        "summary": summary,
        "rows": rows,
        "stats": {"impact": S_impact, "rel_div": S_reldiv, "abs_out": S_absout, "phys_out": S_physout},
    }
    Path(HERE / "impact_localization.json").write_text(json.dumps(out, indent=2))

    # ---- console report ----
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print("=" * 72)
    print(f"clearing coverage assumption (energy to remove for 0.350->0.420): {CLEARING_COVERAGE:.3f}")
    print(f"PRIMARY activation_weighted_top16_energy_share = {impact_top16:.4f}")
    print(f"   (#695 weight-space diff_norm top16 = {ref695_compare['top16_energy_share_695_diffnorm']:.4f};")
    print(f"    rel_div-only [no activation] top16    = {reldiv_top16:.4f})")
    print(f"TEST activation_critical_tps_at_clearing_footprint = {clearing_tps:.3f} "
          f"(f={clearing_f:.3f}, cost {TPS_FLOOR-clearing_tps:.2f} TPS vs 126.378 anchor)")
    print(f"impact CV = {S_impact['cv']:.4f}  gini_like = {S_impact['energy_gini_like']:.4f}")
    print(f"impact top1/8/16/32 energy share = "
          f"{S_impact['topk_energy_share'][1]:.4f}/{S_impact['topk_energy_share'][8]:.4f}/"
          f"{S_impact['topk_energy_share'][16]:.4f}/{S_impact['topk_energy_share'][32]:.4f}")
    print("--- robustness (alt impact magnitudes) ---")
    print(f"abs_out  (diff_norm x act)            top16={S_absout['topk_energy_share'][16]:.4f} "
          f"gini={S_absout['energy_gini_like']:.4f} clearing_tps={S_absout['tps_at_coverage'][CLEARING_COVERAGE]:.3f}")
    print(f"phys_out (diff_norm x act / sqrt dim) top16={S_physout['topk_energy_share'][16]:.4f} "
          f"gini={S_physout['energy_gini_like']:.4f} clearing_tps={S_physout['tps_at_coverage'][CLEARING_COVERAGE]:.3f}")
    print(f"Spearman(prefill,decode) act-norm rank = {summary['spearman_prefill_decode']:.4f}")
    print("--- top-6 modules by impact ---")
    for r in top_impact[:6]:
        print(f"  L{r['layer']:>2} {r['proj']:<22} impact={r['impact']:.3f} "
              f"rel_div={r['rel_div']:.4f} act={r['act_norm']:.1f} params={r['params']}")
    print(f"\nwrote {HERE / 'impact_localization.json'}")


if __name__ == "__main__":
    main()
