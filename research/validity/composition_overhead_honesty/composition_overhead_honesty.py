#!/usr/bin/env python3
"""PR #274 -- Composition honesty: do draft-side 500 claims survive fixed-overhead phi?

CPU-only analytic re-pricing. NO GPU, NO HF Job, NO submission/served-file change.
Re-derivation over existing MERGED measurements only.

Core question
-------------
Every "clears 500" projection credits a draft-pass saving against the FULL E[T]/step
ratio. But the served wall step is not all model-forward: a large fraction is FIXED
serving overhead (CPU/Python scheduler, sampler, detokenize, request plumbing) that does
NOT shrink when draft passes are cut. Decompose

    step(K, M) = fixed_overhead + draft_cost(K) + verify_us(M)

solve fixed_overhead by anchoring the served point, and discount every draft-side gain by
phi = model_forward_fraction < 1. Re-price the portfolio under honest composition.

All anchors are pulled from merged artifacts (denken #257, stark #266, ubel #263,
denken #268, fern #262). No GPU is touched; this is pure arithmetic.
"""
import json
import math
import os
import sys
import time

# ============================================================================
# ANCHORS -- merged artifacts, do not edit (provenance in comments)
# ============================================================================
BASE_TPS = 481.53                     # official served TPS (PR #52)
TARGET_TPS = 500.0
K_CAL = 125.26795005202914            # denken #257 steps/sec calibration
STEP_SERVED_NORM_MS = 1.2182          # NORMALIZED composition step (NOT wall clock)
E_T_REAL = 3.8444537125748504         # stark #266 e_t_base: real accepted tok/step @ K=7
E_T_PARAM = 4.6827608                 # denken #257 E_T_served: param E[T]
LAMBDA1_CEIL = 520.9527323111674      # K_cal * E[T] * tau ceiling

# --- g_d fork (denken #271 measuring; report BOTH edges) ---
G_D_MEASURED = 0.019498025961743392   # denken #257, full-forward measured
G_D_ASSUMED = 0.168                   # fleet anchor (composition basis)

# --- measured WALL phase costs (denken #257, run h1gj2ved), microseconds ---
DRAFT_PASS_US_GRAPHED = 100.6822395324707
VERIFY_US = {                         # full-forward verify wall cost by M
    1: 5119.65170264244, 2: 5118.817284107208, 4: 5141.319851875306,
    8: 5163.714507818222, 16: 5404.861371517183, 32: 5979.945073127746,
}
S_SERVED_ABS_US = 5868.490184545517   # = 7*draft + verify(8): WALL model-forward @ served
SERVED_WALL_CLOCK_EST_US = 9724.754013249434  # denken #257 served_wall_clock_est

# kanna #264 draft-pass internal decomposition (us/pass) -- provenance only
DRAFT_DECOMP_US = {"io": 13.9, "attn": 28.5, "mlp": 50.7, "vocab_head": 4.9}

# --- normalized (assumed-g_d) composition phase costs (stark #266), ms ---
VERIFY_MS_NORM = 0.5598345588235294
COMPUTE_PER_PASS_MS_NORM = 0.09405220588235294   # = g_d_assumed * verify_ms_norm

K_SERVED, M_SERVED = 7, 8

# --- portfolio composition claims to re-price ---
# lever: "draft_pass_cut" lowers E[T] to cut passes; "et_raise" raises accepted tok/step.
PORTFOLIO = {
    "static_k4":  {"comp_tps": 502.12279296537685, "lever": "draft_pass_cut",
                   "E_T": 3.080339640718563, "K": 4, "M": 8, "src": "stark#266"},
    "static_k5":  {"comp_tps": 500.78615614206257, "lever": "draft_pass_cut",
                   "E_T": 3.38082377245509, "K": 5, "M": 8, "src": "stark#266"},
    "tree_width": {"comp_tps": 577.6, "lever": "et_raise",
                   "E_T": None, "K": 7, "M": 32, "src": "denken#268"},
    "private_505": {"comp_tps": 505.4635557048992, "lever": "et_raise",
                    "E_T": 3.6406313872810236, "K": 7, "M": 8, "src": "ubel#263"},
}

EPS = 1e-9


# ============================================================================
# (#6) SMOKE: served anchor reproduces 481.53 at (K=7, M=8)
# ============================================================================
def smoke_reproduce_anchor():
    """Three independent reconstructions of 481.53 at the served point."""
    # (a) composition law: TPS = K_cal * E[T]_param / step_served_norm * tau, tau=1
    tps_comp = K_CAL * E_T_PARAM / STEP_SERVED_NORM_MS
    # (b) clean K_cal form: TPS = K_cal * E[T]_real (wall step = 1/K_cal)
    tps_clean = K_CAL * E_T_REAL
    # (c) static-K composition at K=7 (step_factor = 1 + g_d*K, assumed g_d)
    sf7 = 1.0 + G_D_ASSUMED * K_SERVED
    et_over_sf = E_T_REAL / sf7
    # the static-K table net_tps at K=7 == BASE by construction; verify the ratio anchor
    tps_statick7 = BASE_TPS * (et_over_sf) / (E_T_REAL / sf7)  # identity -> BASE
    recon = {
        "tps_composition_law": tps_comp,
        "tps_clean_kcal": tps_clean,
        "tps_statick7_identity": tps_statick7,
        "rel_err_composition": abs(tps_comp - BASE_TPS) / BASE_TPS,
        "rel_err_clean": abs(tps_clean - BASE_TPS) / BASE_TPS,
    }
    recon["reproduces_481_53"] = (recon["rel_err_composition"] <= 0.01
                                  and recon["rel_err_clean"] <= 0.01)
    return recon


# ============================================================================
# (#2) STEP DECOMPOSITION + fixed_overhead + phi, under BOTH g_d edges
# ============================================================================
def model_forward_us(g_d_edge, K, M):
    """Model-forward (draft+verify) wall cost in us under a g_d basis.

    measured g_d -> REAL wall phase costs (draft 100.68us, verify 5163.7us @ M=8).
    assumed g_d  -> NORMALIZED composition basis (draft 94.05us, verify 559.8us).
    The g_d edge IS the choice of phase-cost basis: g_d = draft_pass / verify.
    """
    if g_d_edge == "measured":
        draft = K * DRAFT_PASS_US_GRAPHED
        verify = VERIFY_US[M]
    elif g_d_edge == "assumed":
        draft = K * (COMPUTE_PER_PASS_MS_NORM * 1000.0)
        verify = VERIFY_MS_NORM * 1000.0
    else:
        raise ValueError(g_d_edge)
    return draft, verify, draft + verify


def decompose_phi():
    """Solve fixed_overhead and phi at the served point under both g_d edges and
    both wall-step conventions. Anchor: step(K=7,M=8) reproduces 481.53."""
    walls = {
        "served_wall_clock_est": SERVED_WALL_CLOCK_EST_US,  # denken #257 (central)
        "kcal_clean": 1e6 / K_CAL,                          # 1/K_cal dimensionally clean
    }
    out = {"wall_step_conventions_us": walls, "edges": {}}
    for g_d_edge in ("measured", "assumed"):
        d, v, mf = model_forward_us(g_d_edge, K_SERVED, M_SERVED)
        edge = {"model_forward_us": mf, "draft_cost_us": d, "verify_us": v,
                "g_d_value": G_D_MEASURED if g_d_edge == "measured" else G_D_ASSUMED,
                "walls": {}}
        for wname, wus in walls.items():
            fixed = wus - mf
            phi = mf / wus
            edge["walls"][wname] = {
                "fixed_overhead_us": fixed,
                "phi": phi,
                "fixed_overhead_ge_0": fixed >= -EPS,
                "phi_in_0_1": (0.0 < phi < 1.0),
            }
        out["edges"][g_d_edge] = edge
    return out


# ============================================================================
# (#3) RE-PRICE PORTFOLIO -- honest_corrected_tps = BASE + (comp-BASE)*phi
# ============================================================================
def reprice(phi_central, decomp, wall_key="served_wall_clock_est"):
    """For each claim: composition_tps vs honest_corrected_tps, realization ratio = phi."""
    rows = {}
    for name, c in PORTFOLIO.items():
        comp = c["comp_tps"]
        comp_gain = comp - BASE_TPS
        row = {"composition_tps": comp, "composition_gain": comp_gain,
               "lever": c["lever"], "src": c["src"], "K": c["K"], "M": c["M"],
               "honest_corrected": {}, "clears_500": {}, "realization_ratio": {}}
        for edge in ("measured", "assumed"):
            phi = decomp["edges"][edge]["walls"][wall_key]["phi"]
            honest = BASE_TPS + comp_gain * phi
            row["honest_corrected"][edge] = honest
            row["clears_500"][edge] = honest >= TARGET_TPS
            # realization ratio = honest_gain / composition_gain == phi (by construction)
            row["realization_ratio"][edge] = (
                (honest - BASE_TPS) / comp_gain if abs(comp_gain) > EPS else float("nan"))
        rows[name] = row
    return rows


# ============================================================================
# RIGOROUS lever-specific wall-step recompute (corroboration, strengthens verdict)
# ============================================================================
def rigorous_wallstep_staticK():
    """The phi-discount-of-net-gain is GENEROUS to draft-pass-cut levers: it discounts
    the whole net gain, but the E[T] PENALTY (3.84->3.08 at K=4) is fully real while only
    the step saving is phi-discounted. Recompute static-K honestly via the wall step:

        TPS_honest(K) / TPS_served = [E[T](K)/E[T](7)] * [wall_step(7)/wall_step(K)]

    Only draft_cost shrinks (7->K passes); fixed_overhead and verify are unchanged.
    """
    out = {}
    for g_d_edge in ("measured", "assumed"):
        _, _, mf7 = model_forward_us(g_d_edge, 7, M_SERVED)
        for wname, wus in (("served_wall_clock_est", SERVED_WALL_CLOCK_EST_US),
                           ("kcal_clean", 1e6 / K_CAL)):
            fixed = wus - mf7
            combo = {}
            for name, c in (("static_k4", PORTFOLIO["static_k4"]),
                            ("static_k5", PORTFOLIO["static_k5"])):
                K = c["K"]
                _, _, mfK = model_forward_us(g_d_edge, K, M_SERVED)
                wall_K = fixed + mfK
                ratio = (c["E_T"] / E_T_REAL) * (wus / wall_K)
                tps = BASE_TPS * ratio
                combo[name] = {"tps": tps, "delta_vs_base": tps - BASE_TPS,
                               "clears_500": tps >= TARGET_TPS,
                               "wall_step_K_us": wall_K, "wall_step_served_us": wus}
            out[f"{g_d_edge}__{wname}"] = combo
    # extract the robust band for static_k4
    k4_vals = [v["static_k4"]["tps"] for v in out.values()]
    out["static_k4_band_tps"] = [min(k4_vals), max(k4_vals)]
    out["static_k4_all_below_500"] = all(v["static_k4"]["tps"] < TARGET_TPS
                                         for v in out.values()
                                         if isinstance(v, dict) and "static_k4" in v)
    out["note"] = (
        "phi-discount-of-net-gain is the PR-prescribed metric and is CONSERVATIVE "
        "(generous to the claim). The rigorous wall-step recompute keeps the full E[T] "
        "penalty and phi-discounts only the step saving -> static-K is a LOSS (~397-401 "
        "TPS), not a gain. Both models agree static-K does not clear 500.")
    return out


# ============================================================================
# (#4) INVERTED BREAK-EVEN
# ============================================================================
def inverted_break_even(decomp, wall_key="served_wall_clock_est"):
    """How much composition gain / E[T] is needed to clear 500 once phi is applied."""
    out = {"phi_discount_space": {}, "honest_et_real_floor": {}, "per_lever_shortfall": {}}
    for edge in ("measured", "assumed"):
        phi = decomp["edges"][edge]["walls"][wall_key]["phi"]
        # honest = BASE + (comp-BASE)*phi >= 500  =>  comp >= BASE + (500-BASE)/phi
        comp_needed = BASE_TPS + (TARGET_TPS - BASE_TPS) / phi
        out["phi_discount_space"][edge] = {
            "phi": phi,
            "composition_tps_needed_to_clear_500": comp_needed,
            "composition_gain_pct_needed": 100.0 * (comp_needed - BASE_TPS) / BASE_TPS,
        }
    # honest E[T]_real floor (lever-agnostic, clean K_cal frame): TPS=K_cal*E_T_real
    et_floor = TARGET_TPS / K_CAL
    out["honest_et_real_floor"] = {
        "E_T_real_needed_to_clear_500": et_floor,
        "E_T_real_served": E_T_REAL,
        "E_T_real_gap": et_floor - E_T_REAL,
        "pct_rise_needed": 100.0 * (et_floor - E_T_REAL) / E_T_REAL,
        "note": ("clean K_cal frame: honest TPS ~= K_cal*E_T_real, so clearing 500 needs "
                 "E_T_real >= 3.991. Draft-pass CUTS lower E[T] (K=4->3.08), moving AWAY; "
                 "only E[T]-RAISING levers can reach it. This is why the deployed path is "
                 "K=7. Analog of fern #262's E[T]>=5.288 (param/grounded-step convention)."),
        "fern262_param_analog": 5.288,
    }
    # per-lever shortfall vs the E[T]_real floor and vs the phi-corrected target
    for name, c in PORTFOLIO.items():
        et = c["E_T"]
        out["per_lever_shortfall"][name] = {
            "lever": c["lever"],
            "E_T": et,
            "E_T_real_floor": et_floor,
            "E_T_shortfall": (et_floor - et) if et is not None else None,
            "moves_toward_floor": (et is not None and et > E_T_REAL),
        }
    return out


# ============================================================================
# TORNADO -- sensitivity of the "clears 500 under honest composition" verdict
# ============================================================================
def tornado(decomp):
    """Which input swings honest_corrected_static_k4_tps and the clears-500 verdict most.
    Central = measured g_d, served_wall_clock_est wall."""
    comp_gain_k4 = PORTFOLIO["static_k4"]["comp_tps"] - BASE_TPS
    central_phi = decomp["edges"]["measured"]["walls"]["served_wall_clock_est"]["phi"]
    central = BASE_TPS + comp_gain_k4 * central_phi

    def hc(phi):
        return BASE_TPS + comp_gain_k4 * phi

    swings = []
    # g_d edge: measured(0.6034) -> assumed(0.1253)
    phi_assumed = decomp["edges"]["assumed"]["walls"]["served_wall_clock_est"]["phi"]
    swings.append({"input": "g_d_edge (0.0195 -> 0.168)",
                   "low_tps": hc(phi_assumed), "high_tps": hc(central_phi),
                   "swing_tps": abs(hc(central_phi) - hc(phi_assumed))})
    # wall-step convention: served_wall_clock_est(9724.75) -> kcal_clean(7982.9), measured g_d
    phi_clean = decomp["edges"]["measured"]["walls"]["kcal_clean"]["phi"]
    swings.append({"input": "wall_step convention (9724.75us -> 7982.9us)",
                   "low_tps": hc(central_phi), "high_tps": hc(phi_clean),
                   "swing_tps": abs(hc(phi_clean) - hc(central_phi))})
    # E[T] band on comp_tps: +-0.5% on the composition projection (proxy uncertainty)
    lo = BASE_TPS + (comp_gain_k4 * 0.995) * central_phi
    hi = BASE_TPS + (comp_gain_k4 * 1.005) * central_phi
    swings.append({"input": "E[T] band on comp_tps (+-0.5%)",
                   "low_tps": lo, "high_tps": hi, "swing_tps": abs(hi - lo)})
    # draft_cost basis: graphed 100.68us -> eager 359.2us shifts model_forward/phi
    mf_eager = 7 * 359.19872283935547 + VERIFY_US[8]
    phi_eager = mf_eager / SERVED_WALL_CLOCK_EST_US
    swings.append({"input": "draft_cost basis (graphed 100.68 -> eager 359.2 us)",
                   "low_tps": hc(central_phi), "high_tps": hc(phi_eager),
                   "swing_tps": abs(hc(phi_eager) - hc(central_phi))})
    swings.sort(key=lambda s: s["swing_tps"], reverse=True)
    # does ANY input push static_k4 >= 500?
    any_clears = any(max(s["low_tps"], s["high_tps"]) >= TARGET_TPS for s in swings)
    return {"central_honest_corrected_static_k4_tps": central,
            "ranked_swings": swings,
            "dominant_input": swings[0]["input"],
            "any_input_pushes_static_k4_to_500": any_clears,
            "verdict_robust_static_k4_below_500": not any_clears}


# ============================================================================
# (#5) CROSS-REFERENCE stark's empirical A/B
# ============================================================================
def cross_ref_stark(decomp):
    """phi predicts the realization ratio stark should measure for K=4 vs K=7."""
    out = {}
    for edge in ("measured", "assumed"):
        phi = decomp["edges"][edge]["walls"]["served_wall_clock_est"]["phi"]
        comp_gain_pct = 100.0 * (PORTFOLIO["static_k4"]["comp_tps"] - BASE_TPS) / BASE_TPS
        # honest wall gain pct (phi-discount model)
        honest_gain_pct = comp_gain_pct * phi
        out[edge] = {
            "phi": phi,
            "composition_gain_k4_pct": comp_gain_pct,
            "predicted_phi_discount_wall_gain_k4_pct": honest_gain_pct,
        }
    # rigorous lever-specific prediction (the physical one stark's wall-clock A/B measures)
    rig = rigorous_wallstep_staticK()
    k4_rig = rig["measured__served_wall_clock_est"]["static_k4"]["tps"]
    out["rigorous_wallstep_prediction"] = {
        "measured_local_wall_tps_k4": k4_rig,
        "measured_local_wall_tps_gain_k4_vs_k7_pct": 100.0 * (k4_rig - BASE_TPS) / BASE_TPS,
        "note": ("stark measures measured_local_wall_tps_gain_k4_vs_k7_pct empirically; "
                 "fern derives it. The phi-discount model predicts a small POSITIVE wall "
                 "gain (+2.6% measured / +0.5% assumed) but the rigorous wall-step model "
                 "predicts a NEGATIVE wall gain (~-17%, static-K LOSES). If stark measures "
                 "a loss, the rigorous model is confirmed; if a small gain, the phi-discount "
                 "bound holds. EITHER outcome keeps static-K below 500."),
    }
    out["discriminating_measurement"] = (
        "stark's K=4-vs-K=7 wall-clock A/B: sign of the gain discriminates phi-discount "
        "(small +) vs rigorous-wallstep (-). Both agree on does-not-clear-500.")
    return out


# ============================================================================
# ASSEMBLE + SELF-TESTS
# ============================================================================
def main():
    t0 = time.time()
    smoke = smoke_reproduce_anchor()
    decomp = decompose_phi()
    # central phi = measured g_d (physically grounded edge), served_wall_clock_est wall
    phi_central = decomp["edges"]["measured"]["walls"]["served_wall_clock_est"]["phi"]
    phi_assumed = decomp["edges"]["assumed"]["walls"]["served_wall_clock_est"]["phi"]
    portfolio = reprice(phi_central, decomp)
    rigorous = rigorous_wallstep_staticK()
    ibe = inverted_break_even(decomp)
    trn = tornado(decomp)
    xref = cross_ref_stark(decomp)

    honest_k4_central = portfolio["static_k4"]["honest_corrected"]["measured"]
    honest_k4_assumed = portfolio["static_k4"]["honest_corrected"]["assumed"]

    # boolean: any draft lever clears 500 under honest composition, per g_d edge
    any_clears = {}
    by_mech = {}
    for edge in ("measured", "assumed"):
        clears = {n: portfolio[n]["clears_500"][edge] for n in PORTFOLIO}
        any_clears[edge] = any(clears.values())
        by_mech[edge] = {
            "draft_pass_cut_any_clears": any(
                clears[n] for n in PORTFOLIO if PORTFOLIO[n]["lever"] == "draft_pass_cut"),
            "et_raise_any_clears": any(
                clears[n] for n in PORTFOLIO if PORTFOLIO[n]["lever"] == "et_raise"),
            "per_claim": clears,
        }

    # ---- SELF-TEST gates (PRIMARY) ----
    all_tps = [honest_k4_central, honest_k4_assumed]
    for n in PORTFOLIO:
        all_tps += list(portfolio[n]["honest_corrected"].values())
    nan_clean = all(math.isfinite(x) for x in all_tps)
    # (a) reproduce 481.53 at served point within +-1%
    st_a = smoke["reproduces_481_53"]
    # (b) 0<phi<1 and fixed_overhead>=0 under BOTH g_d edges (central wall)
    st_b = all(
        decomp["edges"][e]["walls"]["served_wall_clock_est"]["phi_in_0_1"]
        and decomp["edges"][e]["walls"]["served_wall_clock_est"]["fixed_overhead_ge_0"]
        for e in ("measured", "assumed"))
    # (c) NaN-clean, all corrected TPS finite
    st_c = nan_clean
    # (d) realization ratios reported for every claim (both edges, finite)
    st_d = all(math.isfinite(portfolio[n]["realization_ratio"][e])
               for n in PORTFOLIO for e in ("measured", "assumed"))
    # (e) inverted break-even reported
    st_e = ("composition_tps_needed_to_clear_500"
            in ibe["phi_discount_space"]["measured"]
            and math.isfinite(ibe["honest_et_real_floor"]["E_T_real_needed_to_clear_500"]))
    self_tests = {"a_reproduces_481_53_within_1pct": st_a,
                  "b_phi_in_0_1_and_fixed_ge_0_both_gd_edges": st_b,
                  "c_nan_clean_all_corrected_finite": st_c,
                  "d_realization_ratios_for_every_claim": st_d,
                  "e_inverted_break_even_reported": st_e}
    primary = all(self_tests.values())

    report = {
        "pr": 274,
        "agent": "fern",
        "task": "composition_overhead_honesty",
        "kind": "CPU-only analytic re-pricing (no GPU, no HF job, no submission)",
        # ----- PRIMARY -----
        "composition_overhead_honesty_self_test_passes": int(primary),
        "self_tests": self_tests,
        # ----- TEST metric + phi -----
        "honest_corrected_static_k4_tps": honest_k4_central,
        "honest_corrected_static_k4_tps_band": {
            "measured_gd": honest_k4_central, "assumed_gd": honest_k4_assumed},
        "model_forward_fraction_phi": {
            "central_measured_gd": phi_central,
            "assumed_gd": phi_assumed,
            "band": [min(phi_central, phi_assumed), max(phi_central, phi_assumed)],
            "central_basis": "measured g_d (denken #257 flagged assumed g_d ~9x too high), "
                             "served_wall_clock_est wall",
        },
        # ----- decomposition -----
        "smoke_reproduce_anchor": smoke,
        "step_decomposition": decomp,
        "fixed_overhead_us": {
            "measured_gd": decomp["edges"]["measured"]["walls"][
                "served_wall_clock_est"]["fixed_overhead_us"],
            "assumed_gd": decomp["edges"]["assumed"]["walls"][
                "served_wall_clock_est"]["fixed_overhead_us"],
        },
        # ----- re-priced portfolio -----
        "repriced_portfolio": portfolio,
        "rigorous_wallstep_recompute": rigorous,
        # ----- inverted break-even -----
        "inverted_break_even": ibe,
        # ----- booleans -----
        "any_draft_lever_clears_500_under_honest_composition": {
            "measured_gd": any_clears["measured"],
            "assumed_gd": any_clears["assumed"],
            "by_mechanism": by_mech,
            "headline": ("under measured g_d only the E[T]-RAISING tree_width clears "
                         "(539.5); NO draft-pass-CUTTING lever (static-K/K5) clears under "
                         "EITHER g_d edge; under assumed g_d NOTHING clears."),
        },
        # ----- tornado -----
        "tornado": trn,
        # ----- cross-ref stark -----
        "cross_ref_stark": xref,
        "anchors_used": {
            "BASE_TPS": BASE_TPS, "K_CAL": K_CAL, "E_T_REAL": E_T_REAL,
            "E_T_PARAM": E_T_PARAM, "STEP_SERVED_NORM_MS": STEP_SERVED_NORM_MS,
            "G_D_MEASURED": G_D_MEASURED, "G_D_ASSUMED": G_D_ASSUMED,
            "S_SERVED_ABS_US": S_SERVED_ABS_US,
            "SERVED_WALL_CLOCK_EST_US": SERVED_WALL_CLOCK_EST_US,
            "DRAFT_PASS_US_GRAPHED": DRAFT_PASS_US_GRAPHED,
            "VERIFY_US_M8": VERIFY_US[8], "VERIFY_US_M32": VERIFY_US[32],
            "LAMBDA1_CEIL": LAMBDA1_CEIL,
        },
        "runtime_s": time.time() - t0,
        "wandb_run_ids": [],
    }
    return report


if __name__ == "__main__":
    rep = main()
    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, "report.json")

    # ---- optional W&B logging (analytic; never fail the run on W&B issues) ----
    wandb_run_id = ""
    if "--no-wandb" not in sys.argv:
        try:
            import wandb
            run = wandb.init(
                entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                name="fern/composition-overhead-honesty",
                group="composition-honesty",
                job_type="analytic",
                config=rep["anchors_used"],
            )
            wandb_run_id = run.id if run is not None else ""
            flat = {
                "composition_overhead_honesty_self_test_passes":
                    rep["composition_overhead_honesty_self_test_passes"],
                "honest_corrected_static_k4_tps": rep["honest_corrected_static_k4_tps"],
                "honest_corrected_static_k4_tps_assumed_gd":
                    rep["honest_corrected_static_k4_tps_band"]["assumed_gd"],
                "phi_measured_gd": rep["model_forward_fraction_phi"]["central_measured_gd"],
                "phi_assumed_gd": rep["model_forward_fraction_phi"]["assumed_gd"],
                "fixed_overhead_us_measured_gd": rep["fixed_overhead_us"]["measured_gd"],
                "fixed_overhead_us_assumed_gd": rep["fixed_overhead_us"]["assumed_gd"],
                "any_draft_lever_clears_500_measured_gd":
                    int(rep["any_draft_lever_clears_500_under_honest_composition"]["measured_gd"]),
                "any_draft_lever_clears_500_assumed_gd":
                    int(rep["any_draft_lever_clears_500_under_honest_composition"]["assumed_gd"]),
                "rigorous_static_k4_tps_min": rep["rigorous_wallstep_recompute"]["static_k4_band_tps"][0],
                "rigorous_static_k4_tps_max": rep["rigorous_wallstep_recompute"]["static_k4_band_tps"][1],
                "honest_et_real_floor":
                    rep["inverted_break_even"]["honest_et_real_floor"]["E_T_real_needed_to_clear_500"],
            }
            for nm, r in rep["repriced_portfolio"].items():
                flat[f"honest_{nm}_measured_gd"] = r["honest_corrected"]["measured"]
                flat[f"honest_{nm}_assumed_gd"] = r["honest_corrected"]["assumed"]
            wandb.log(flat)
            wandb.summary.update(flat)
            rep["wandb_run_ids"] = [wandb_run_id] if wandb_run_id else []
            wandb.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] skipped: {type(e).__name__}: {e}", file=sys.stderr)

    with open(outpath, "w") as f:
        json.dump(rep, f, indent=2)

    # ---- console summary ----
    print(f"PRIMARY composition_overhead_honesty_self_test_passes = "
          f"{rep['composition_overhead_honesty_self_test_passes']}")
    print(f"phi: measured g_d = {rep['model_forward_fraction_phi']['central_measured_gd']:.4f}, "
          f"assumed g_d = {rep['model_forward_fraction_phi']['assumed_gd']:.4f}")
    print(f"fixed_overhead_us: measured={rep['fixed_overhead_us']['measured_gd']:.1f}, "
          f"assumed={rep['fixed_overhead_us']['assumed_gd']:.1f}")
    print(f"TEST honest_corrected_static_k4_tps (measured central) = "
          f"{rep['honest_corrected_static_k4_tps']:.2f} "
          f"(assumed {rep['honest_corrected_static_k4_tps_band']['assumed_gd']:.2f})")
    print("repriced portfolio (honest_corrected measured / assumed | clears500):")
    for nm, r in rep["repriced_portfolio"].items():
        print(f"  {nm:12s} comp={r['composition_tps']:7.2f} -> "
              f"{r['honest_corrected']['measured']:7.2f} / "
              f"{r['honest_corrected']['assumed']:7.2f}  "
              f"clears[{r['clears_500']['measured']}/{r['clears_500']['assumed']}] "
              f"realiz_ratio={r['realization_ratio']['measured']:.3f} [{r['lever']}]")
    print(f"rigorous wall-step static_k4 band = "
          f"{rep['rigorous_wallstep_recompute']['static_k4_band_tps']}")
    print(f"inverted break-even E_T_real floor = "
          f"{rep['inverted_break_even']['honest_et_real_floor']['E_T_real_needed_to_clear_500']:.4f} "
          f"(served {E_T_REAL:.4f})")
    print(f"any_draft_lever_clears_500: measured={rep['any_draft_lever_clears_500_under_honest_composition']['measured_gd']}, "
          f"assumed={rep['any_draft_lever_clears_500_under_honest_composition']['assumed_gd']}")
    print(f"tornado dominant input = {rep['tornado']['dominant_input']}")
    print(f"wandb_run_ids = {rep['wandb_run_ids']}")
    print(f"wrote {outpath}")
