#!/usr/bin/env python3
"""PR #737 -- quantify the #730 PRIVATE-DQ risk for the un-rescued stock-drafter
K=6 config (`int4_mtp_batchinv`). Acceptance-drift budget delta_max + P(DQ) + a
parametric P(DQ)-vs-delta_stock curve (plugs into lawine #734 with zero rework).
`analysis_only` -- NO HF Job, NO fire. Guard flags live in wandb.summary.

This is the natural successor to my #735 (`3hdyip2b`): #735 settled the SPEED leg
(official-equiv 167.9 public / ~157 private; P(clears the 126.378 BAR after the
4-9% haircut) = 1.00). #737 quantifies the SEPARATE private-repro validity gate.

=============================================================================
THE LOAD-BEARING FINDING (read first): WHICH gate actually DQs?
=============================================================================
The PR's step-1 OPERATIONALIZES "DQ" as "private official-equiv falls under the
126.378 BAR". But the program's *actual* binding validity rule is DIFFERENT, and
it bites MUCH sooner. Per BASELINE.md (lines 36-37, 40-41, 19):

    "The verifier re-runs on a private prompt set; top drafter stacks lose
     4-9% TPS on it. Submissions DIE on the 5% TPS-reproduction gap, not on PPL."
    Flagship #52: organizer re-run 460.85 private vs 481.53 public -> Delta 4.3%
     <= 5% -> VERIFIED VALID. kanna #44: 12.4% chat-proxy -> WOULD-FAIL.

So there are TWO candidate DQ conditions, and they disagree by ~4-5x in budget:

  G1  5% TPS-REPRODUCTION gap  [BINDING -- BASELINE.md, empirically confirmed]
      DQ iff (public_private_TPS_drop) > 5%.  RELATIVE to the submission's OWN
      public number -> the transfer factor and the central CANCEL -> budget is a
      flat 5% regardless of where the central sits.

  G2  BAR-crossing             [PR step-1 literal -- NON-binding here]
      DQ iff private_official_equiv < 126.378.  Central-dependent; needs an
      18-25% drift to trip -> 3.7-4.9x looser than G1.  G2 NEVER binds first.

G1 trips at 5%; G2 trips at 18-25%. => G1 is the operative private-DQ gate; the
PR's bar framing understates the risk by ~4-5x. I compute BOTH, lead with G1, and
flag the discrepancy so the advisor can re-rule if the bar was truly intended.

=============================================================================
THE ACCEPTANCE->TPS MODEL (so "drift in mean acceptance" maps to a TPS haircut)
=============================================================================
`e_accept_exact` is the BLOCK EFFICIENCY = mean tokens emitted per spec step,
INCLUDING the always-correct bonus token. Verified on the #730 K=6 records:
    e_accept_exact 3.6552 == accepted_per_step 2.6552 + 1 bonus   (accept_rate 44.3%)
With per-step wall time ~constant in acceptance (you always draft K=6 then verify
K=6 in one pass), TPS = e_accept_exact / T_step  =>  TPS PROPORTIONAL to e_accept.
So a relative drop delta in E_accept (=e_accept_exact, the PR's exact definition
"E_accept_private = (1-delta)*E_accept_public") gives an EQUAL relative TPS drop:

    haircut h(delta) = delta            [LINEAR -- the PR-faithful primary model]

This matches kanna #44 empirically (acceptance ratio 0.872 ~= TPS ratio 0.887, no
cushion). Because h == delta, the #725 documented haircut band U[4%,9%] IS the
delta_stock prior with no conversion, and P(delta_stock>5%) = 0.80 transfers
directly. (A cushioned alt -- if one instead measures the per-TOKEN accept-rate
alpha drop -- gives h = delta_alpha * (g-1)/g = 0.727*delta_alpha, i.e. a LARGER
budget; linear is both PR-faithful AND the conservative/ tighter-budget choice.)
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/senpai/target")
HERE = Path(__file__).resolve().parent
K6_PAIRED = ROOT / "research/walltps_ab/optionb_bi1_stock_int4/ksweep/k6/paired_ab.json"
RECON_735 = HERE / "results/officialequiv_reconcile_730.json"

BAR_OFFICIAL = 126.378              # int4_g128_lmhead PR#4 pure-AR (private==public; no haircut on the bar)

# --- transfer family (from #735; int4 same-precision anchored) ---
T_INT4_MATCH = 126.378 / 128.13    # 0.9863 central
T_INT4_PESS = 126.378 / 131.60     # 0.9603 floor
T_DEFINITION = 1.000               # ceiling
MC_LO, MC_HI = T_INT4_PESS, T_DEFINITION   # int4 band [0.960, 1.000]

# --- the two centrals the PR asks for ---
# optimistic = #735 upper-bound proxy (fast /tmp/qat-assistant local on the ship base model)
# honest-stock = the literal stock-Hub un-rescued drafter, never locally speed-measured (#735 caveat)
HONEST_STOCK_LO, HONEST_STOCK_HI = 150.0, 160.0
HONEST_STOCK_MID = 155.0

# --- the two gates ---
REPRO_GATE = 0.05                  # G1 BINDING: private TPS must stay within 5% of public
# G2 = BAR_OFFICIAL (non-binding bar-crossing; PR step-1 literal)

# --- #725 drift prior: documented 4-9% private-verify band (BASELINE.md 36) ---
# Under the LINEAR model h==delta, this band is BOTH the TPS-haircut prior and the
# delta_stock (acceptance-drop) prior. P(delta>5%) = (9-5)/(9-4) = 0.80 == #725.
DRIFT_LO, DRIFT_HI = 0.04, 0.09
PRIOR_CENTER = 0.5 * (DRIFT_LO + DRIFT_HI)   # 6.5% -- NOTE: already ABOVE the 5% gate

# --- named drift anchors (sourced) ---
DRIFT_FLAGSHIP = 0.043             # #52 organizer private re-run (wide-trained stack -> PASS)
DRIFT_44_PROXY = 0.124             # kanna #44 pure-chat proxy upper bound (-> would-FAIL)

BREAKEVEN_TRANSFER_735 = 0.8159    # #735: transfer at which the bar-gate fails @ 9% haircut

N_MC = 400_000
SEED = 737


def _load():
    k6 = json.loads(K6_PAIRED.read_text())
    cand = k6["arms"]["candidate"]
    local_k6 = float(k6["verdict"]["candidate_median_wall_tps"])         # 170.21
    e_accept = float(cand["e_accept_exact"]["median"])                   # 3.6585 block-eff
    e_accept_mean = float(cand["e_accept_exact"]["mean"])
    rec0 = cand["records"][0]
    return {
        "local_k6_wall_tps": local_k6,
        "e_accept_exact_median": e_accept,
        "e_accept_exact_mean": e_accept_mean,
        "num_spec": int(cand["records"][0].get("num_speculative_tokens", 6)),
        "accepted_per_step": e_accept - 1.0,                            # block-eff minus bonus
        "accept_rate_per_token": (e_accept - 1.0) / 6.0,
        "central_optimistic_public": round(local_k6 * T_INT4_MATCH, 2), # 167.9
        "rec_total_accepted": rec0.get("total_accepted_tokens"),
        "rec_total_drafted": rec0.get("total_drafted_tokens"),
    }


# ------------------------------------------------------------------ budgets --
def delta_max_repro_gate():
    """G1 budget: a 5% TPS drop. Under h==delta this is a flat 5% acceptance drop,
    INDEPENDENT of the central (the 5% is relative to the submission's own public)."""
    return REPRO_GATE


def delta_max_bar(central_public):
    """G2 budget: drift at which public_central*(1-delta) == 126.378.
    delta_max = 1 - BAR/central. Central-DEPENDENT (PR step-1 literal)."""
    return 1.0 - BAR_OFFICIAL / central_public


def delta_max_bar_cushioned(central_public, e_accept):
    """If delta is instead defined on the per-TOKEN accept-rate (not E_accept),
    the same TPS haircut needs a LARGER acceptance-rate drop: delta_alpha =
    h * g/(g-1). Reported as a sensitivity (looser budget)."""
    h = delta_max_bar(central_public)
    g = e_accept
    return h * g / (g - 1.0)


# --------------------------------------------------------------- P(DQ) prior --
def p_dq_prior_repro():
    """G1 at the #725 prior: delta_stock ~ U[4%,9%]; DQ iff delta_stock>5%.
    Transfer cancels (relative gate) -> P(DQ) = P(delta>5%) = 0.80 exactly."""
    return (DRIFT_HI - REPRO_GATE) / (DRIFT_HI - DRIFT_LO)


def p_dq_prior_bar(local_k6, rng):
    """G2 at the #725 prior: integrate transfer U[0.960,1.000] x drift U[4%,9%];
    DQ iff local*T*(1-h) < BAR. (== 1 - #735's P(clears) = ~0.)"""
    T = rng.uniform(MC_LO, MC_HI, N_MC)
    h = rng.uniform(DRIFT_LO, DRIFT_HI, N_MC)
    priv = local_k6 * T * (1 - h)
    return float((priv < BAR_OFFICIAL).mean())


# ----------------------------------------------------- parametric P(DQ) curve --
def sweep_curve(local_k6, e_accept, rng):
    """P(DQ) vs an ASSUMED point delta_stock, for both gates. lawine #734 pins the
    real delta_stock; the human reads P(DQ) off this curve.
      G1 (repro): transfer cancels -> hard step at 5% (truth). A finite-128-prompt
                  realization-noise sigma smooths it to a sigmoid (illustrative).
      G2 (bar):   transfer band [0.960,1.000] gives a genuine smooth curve."""
    deltas = np.round(np.linspace(0.0, 0.30, 121), 5)
    T = rng.uniform(MC_LO, MC_HI, N_MC)
    rows = []
    for d in deltas:
        # G2 bar: P(local*T*(1-d) < BAR)
        p_bar = float((local_k6 * T * (1 - d) < BAR_OFFICIAL).mean())
        # G1 repro hard step
        p_repro_hard = 1.0 if d > REPRO_GATE else 0.0
        # G1 repro with finite-sample realization noise (illustrative sigma)
        p_repro_s1 = _phi((d - REPRO_GATE) / 0.01)
        p_repro_s2 = _phi((d - REPRO_GATE) / 0.02)
        rows.append({
            "delta_stock": float(d),
            "p_dq_repro_hard": p_repro_hard,
            "p_dq_repro_sigma1pct": p_repro_s1,
            "p_dq_repro_sigma2pct": p_repro_s2,
            "p_dq_bar_optimistic": p_bar,
        })
    # coin-flip (P=0.5) crossings
    coin_repro = REPRO_GATE                                   # step / sigmoid center
    coin_bar_opt = _coin_bar(local_k6, 0.5 * (MC_LO + MC_HI)) # transfer median 0.9801
    coin_bar_honest = delta_max_bar(HONEST_STOCK_MID)         # deterministic honest-stock
    return deltas, rows, {
        "coinflip_repro_gate_delta": coin_repro,
        "coinflip_bar_optimistic_delta": coin_bar_opt,
        "coinflip_bar_honeststock_delta": coin_bar_honest,
    }


def _coin_bar(local, t_mid):
    # delta at which local*t_mid*(1-delta) == BAR
    return 1.0 - BAR_OFFICIAL / (local * t_mid)


def _phi(z):
    # standard normal CDF without scipy
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# ---------------------------------------------------------------- the analysis --
def analyze():
    inp = _load()
    rng = np.random.default_rng(SEED)
    local = inp["local_k6_wall_tps"]
    e_accept = inp["e_accept_exact_median"]
    central_opt = inp["central_optimistic_public"]            # 167.9

    # ---- step 1: acceptance-drift budget delta_max (both gates, both centrals) ----
    budgets = {
        "G1_repro_5pct_BINDING": {
            "delta_max": delta_max_repro_gate(),               # 0.05 -- central-independent
            "central_dependent": False,
            "note": "5% TPS drop relative to own public; transfer+central cancel.",
        },
        "G2_bar_NONbinding": {
            "delta_max_optimistic_167p9": delta_max_bar(central_opt),
            "delta_max_honeststock_mid_155": delta_max_bar(HONEST_STOCK_MID),
            "delta_max_honeststock_lo_150": delta_max_bar(HONEST_STOCK_LO),
            "delta_max_honeststock_hi_160": delta_max_bar(HONEST_STOCK_HI),
            "central_dependent": True,
            "cushioned_alpha_optimistic": delta_max_bar_cushioned(central_opt, e_accept),
            "cushioned_alpha_honeststock_155": delta_max_bar_cushioned(HONEST_STOCK_MID, e_accept),
            "note": "PR step-1 literal (private official-equiv < 126.378). 3.7-4.9x looser than G1.",
        },
    }

    # ---- step 2: prior P(DQ) for both gates ----
    p_dq = {
        "G1_repro_at_prior_BINDING": p_dq_prior_repro(),       # 0.80
        "G2_bar_at_prior_NONbinding": p_dq_prior_bar(local, rng),  # ~0
        "_prior": "delta_stock ~ U[4%,9%] (BASELINE.md documented band; == #725)",
        # context anchors for G1 (the binding gate)
        "G1_repro_if_drift_eq_flagship_4p3pct": 1.0 if DRIFT_FLAGSHIP > REPRO_GATE else 0.0,
        "G1_repro_if_drift_eq_priorcenter_6p5pct": 1.0 if PRIOR_CENTER > REPRO_GATE else 0.0,
        "G1_repro_if_drift_eq_44proxy_12p4pct": 1.0 if DRIFT_44_PROXY > REPRO_GATE else 0.0,
    }

    # ---- step 3: parametric P(DQ) vs delta_stock ----
    deltas, curve, coinflips = sweep_curve(local, e_accept, rng)

    # ---- step 4: sensitivity / dominant term ----
    dmax_central_swing_bar = abs(delta_max_bar(central_opt) - delta_max_bar(HONEST_STOCK_MID))
    sensitivity = {
        "binding_gate_G1": {
            "delta_max_moves_with_central": 0.0,               # flat 5% everywhere
            "delta_max_moves_with_drift_prior": 0.0,           # gate is fixed; the DRIFT moves P(DQ)
            "p_dq_swing_over_central": 0.0,                    # 0.80 at any central
            "p_dq_swing_over_drift": 1.0,                      # delta_stock<5% -> 0 ; >5% -> 1
            "load_bearing_assumption": "THE DRIFT (delta_stock). The central choice -- the whole "
                                       "object #735 reconciled -- does NOT touch the binding "
                                       "private-DQ risk. delta_max is pinned at the 5% gate; only "
                                       "where delta_stock truly falls moves P(DQ).",
        },
        "nonbinding_gate_G2": {
            "delta_max_central_swing_167p9_vs_155": dmax_central_swing_bar,  # ~6.3pp
            "p_dq_swing_over_central": 0.0,                    # ~0 at every central (worst corner clears)
            "load_bearing_assumption": "central choice (but G2 never binds, so moot).",
        },
        "dominant_term": "GATE CHOICE (G1 vs G2) >> drift (within G1) >> central. The decision "
                         "hinges on the 5% repro gate, NOT on the 167.9-vs-155 central #735 settled.",
    }

    # ---- step 5: rank the two residual risks ----
    ranking = {
        "risk_a_identity_gate_roll": {
            "what": "organizer strict greedy-identity handling of benign int4 ties.",
            "status": "LOW / already-favorable. BASELINE.md L49 + kanna #38: the official HF-Jobs "
                      "gate = PPL + completion + modalities, and NEVER compares served tokens to a "
                      "greedy-AR reference -> spec stacks are leaderboard-legal (the whole ~420 "
                      "frontier ships MTP spec). The int4 strict-tie concern is an INTERNAL "
                      "pre-flight, not the official gate. Fire buys little NEW info here.",
            "residual": "small",
        },
        "risk_b_private_dq_gate": {
            "what": "5% private-repro TPS gap for the un-rescued stock-Hub drafter.",
            "status": "HIGH / live / UNMEASURED. Budget is only 5%; the documented drift band's "
                      "CENTER (6.5%) already EXCEEDS the gate; naive P(DQ)=0.80. The one favorable "
                      "anchor (flagship 4.3%) is a WIDE-trained stack; the #730 stock-Hub drafter is "
                      "NOT wide-trained (BASELINE.md L38) so is, if anything, MORE drift-prone. Its "
                      "private E_accept drift has never been measured on-branch.",
            "residual": "dominant",
        },
        "verdict": "RISK (b) PRIVATE-DQ DOMINATES (a) IDENTITY-ROLL. The identity gate is "
                   "known-favorable (#38); the binding private-repro gate is the live, material, "
                   "unmeasured risk at naive P(DQ)=0.80.",
        "fire_now": "NO -- not on the binding gate. The SPEED leg (P(clears bar)=1.00, #735) and "
                    "the IDENTITY leg (#38 no-token-check) are both settled-favorable, but they are "
                    "NOT the binding constraint. The dominant residual risk (5% private-repro) sits "
                    "at naive 0.80 and is UNMEASURED for the stock drafter. Recommend: HOLD the fire "
                    "until lawine #734 pins delta_stock; fire only if delta_stock < ~3.5% (comfortable "
                    "margin under the 5% gate), else the human must knowingly accept ~80% DQ odds.",
    }

    verdict = {
        "primary_metric_name": "p_dq_private_gate_at_prior",
        "primary_metric_value": round(p_dq["G1_repro_at_prior_BINDING"], 4),    # 0.80 (binding)
        "test_metric_name": "accept_drift_budget_delta_max",
        "test_metric_value": round(budgets["G1_repro_5pct_BINDING"]["delta_max"], 4),  # 0.05 (binding budget-before-DQ)
        "binding_gate": "G1 5% TPS-reproduction gap (BASELINE.md 36-37; flagship 4.3% PASS, #44 12.4% FAIL)",
        "delta_max_binding_repro": REPRO_GATE,
        "delta_max_bar_optimistic_167p9": round(delta_max_bar(central_opt), 4),
        "delta_max_bar_honeststock_155": round(delta_max_bar(HONEST_STOCK_MID), 4),
        "p_dq_binding_repro_at_prior": round(p_dq["G1_repro_at_prior_BINDING"], 4),
        "p_dq_bar_at_prior_nonbinding": round(p_dq["G2_bar_at_prior_NONbinding"], 4),
        "coinflip_delta_repro": coinflips["coinflip_repro_gate_delta"],
        "coinflip_delta_bar_optimistic": round(coinflips["coinflip_bar_optimistic_delta"], 4),
        "two_risk_ranking": ranking["verdict"],
        "fire_now": ranking["fire_now"],
        "headline": (
            "PRIVATE-DQ DOMINATES and the fire is NOT clear on the binding gate. The operative "
            "validity rule is the 5%% TPS-REPRODUCTION gap (BASELINE.md), NOT the 126.378 bar the "
            "PR step-1 names -- it trips at a %.0f%% acceptance-drift budget vs the bar's %.1f-%.1f%% "
            "(3.7-4.9x looser, never binds). At the documented 4-9%% drift prior, P(DQ on the 5%% "
            "gate) = %.2f (the band's 6.5%% CENTER already exceeds the 5%% gate); P(DQ on the bar) = "
            "%.2f. delta_max for the binding gate is CENTRAL-INVARIANT, so #735's 167.9-vs-155 "
            "reconciliation does NOT touch this risk -- the load-bearing unknown is the stock "
            "drafter's true private drift, UNMEASURED on-branch. Of the two risks the fire buys "
            "info on, the identity-gate roll is known-favorable (#38: official gate has no "
            "token-identity check) while the private-repro DQ is live at ~0.80. RECOMMEND HOLD "
            "until lawine #734 pins delta_stock < ~3.5%%."
        ) % (
            REPRO_GATE * 100,
            delta_max_bar(HONEST_STOCK_MID) * 100, delta_max_bar(central_opt) * 100,
            p_dq["G1_repro_at_prior_BINDING"], p_dq["G2_bar_at_prior_NONbinding"],
        ),
    }

    return {
        "pr": 737, "student": "kanna",
        "card": "quantify the #730 private-DQ risk: acceptance-drift budget delta_max + P(DQ)",
        "guard_flags": {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0},
        "bar_official": BAR_OFFICIAL,
        "inputs": inp,
        "acceptance_tps_model": {
            "e_accept_is_block_efficiency": True,
            "evidence": f"e_accept_exact {e_accept:.4f} == accepted/step {e_accept-1:.4f} + 1 bonus",
            "model": "h == delta (LINEAR); TPS proportional to e_accept_exact",
            "pr_faithful_and_conservative": True,
        },
        "gate_resolution": {
            "G1_repro_5pct": "BINDING (BASELINE.md 36-37,40; flagship 4.3% PASS / #44 12.4% FAIL)",
            "G2_bar_126378": "PR step-1 literal; NON-binding (3.7-4.9x looser, never trips first)",
            "discrepancy_flag": "PR step-1 equates DQ with below-bar; the contract DQs at the 5% "
                                "repro gap long before. I lead with G1; happy to collapse to the "
                                "bar reading on an advisor ruling.",
        },
        "step1_budgets": budgets,
        "step2_p_dq_prior": p_dq,
        "step3_curve": curve,
        "step3_coinflips": coinflips,
        "step4_sensitivity": sensitivity,
        "step5_two_risk_ranking": ranking,
        "anchors": {
            "drift_flagship_4p3pct_PASS": DRIFT_FLAGSHIP,
            "drift_priorcenter_6p5pct": PRIOR_CENTER,
            "drift_44proxy_12p4pct_FAIL": DRIFT_44_PROXY,
            "breakeven_transfer_735": BREAKEVEN_TRANSFER_735,
        },
        "verdict": verdict,
    }


def _print(out):
    v = out["verdict"]
    print("VERDICT:", v["headline"])
    print(f"\n  primary  p_dq_private_gate_at_prior (G1 5% repro) = {v['primary_metric_value']}")
    print(f"  test     accept_drift_budget_delta_max (G1)        = {v['test_metric_value']}")
    print("\n-- step1 budgets --")
    print(f"  G1 repro (BINDING)   delta_max = {REPRO_GATE:.3f}  (central-INVARIANT)")
    b = out["step1_budgets"]["G2_bar_NONbinding"]
    print(f"  G2 bar  optimistic 167.9       = {b['delta_max_optimistic_167p9']:.4f}")
    print(f"  G2 bar  honest-stock 155       = {b['delta_max_honeststock_mid_155']:.4f}")
    print(f"  G2 bar  honest-stock [150,160] = [{b['delta_max_honeststock_lo_150']:.4f}, {b['delta_max_honeststock_hi_160']:.4f}]")
    print("\n-- step2 P(DQ) at prior U[4%,9%] --")
    p = out["step2_p_dq_prior"]
    print(f"  G1 repro (BINDING)  P(DQ) = {p['G1_repro_at_prior_BINDING']:.4f}")
    print(f"  G2 bar (non-bind)   P(DQ) = {p['G2_bar_at_prior_NONbinding']:.4f}")
    print("\n-- step3 coin-flip (P=0.5) crossings --")
    for k, val in out["step3_coinflips"].items():
        print(f"  {k} = {val:.4f}")
    print("\n-- step5 ranking --")
    print(" ", out["step5_two_risk_ranking"]["verdict"])
    print(" FIRE NOW:", out["step5_two_risk_ranking"]["fire_now"][:90], "...")


def _plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = out["step3_curve"]
    d = [r["delta_stock"] * 100 for r in curve]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(d, [r["p_dq_bar_optimistic"] for r in curve], color="#1f77b4", lw=2,
            label="G2 bar (NON-binding, optimistic 167.9 x transfer band)")
    ax.plot(d, [r["p_dq_repro_sigma1pct"] for r in curve], color="#d62728", lw=2,
            label="G1 5% repro (BINDING, sigma=1% realization noise)")
    ax.plot(d, [r["p_dq_repro_sigma2pct"] for r in curve], color="#d62728", lw=1.2, ls="--",
            label="G1 5% repro (sigma=2%)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.axvline(REPRO_GATE * 100, color="#d62728", lw=0.9, ls=":")
    ax.axvspan(DRIFT_LO * 100, DRIFT_HI * 100, color="orange", alpha=0.12,
               label="documented drift prior U[4%,9%]")
    for x, lab, col in [(DRIFT_FLAGSHIP * 100, "flagship 4.3% (PASS)", "green"),
                        (PRIOR_CENTER * 100, "prior center 6.5%", "black"),
                        (DRIFT_44_PROXY * 100, "#44 proxy 12.4% (FAIL)", "purple")]:
        ax.axvline(x, color=col, lw=0.8, ls="-.")
        ax.text(x + 0.2, 0.06, lab, rotation=90, fontsize=7, color=col, va="bottom")
    ax.set_xlabel("assumed private acceptance drift  delta_stock  (% relative drop in E_accept)")
    ax.set_ylabel("P(DQ)")
    ax.set_title("PR #737: P(DQ) vs delta_stock -- binding 5% repro gate vs non-binding bar\n"
                 "(read lawine #734's measured delta_stock off this curve)")
    ax.set_xlim(0, 30); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default="kanna-730-private-dq-risk")
    ap.add_argument("--name", default="kanna/730-private-dq-risk")
    ap.add_argument("--out", default=str(HERE / "results/private_dq_risk_730.json"))
    ap.add_argument("--plot", default=str(HERE / "results/private_dq_risk_730_curve.png"))
    args = ap.parse_args()

    out = analyze()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    out["_outpath"] = str(outp)
    plot_path = _plot(out, args.plot)
    _print(out)
    print("\nWROTE", outp)
    print("WROTE", plot_path)

    if args.wandb:
        import wandb
        v = out["verdict"]
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=args.name, job_type="analysis",
            config={
                "pr": 737, "student": "kanna",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "bar_official": BAR_OFFICIAL,
                "candidate": "int4_mtp_batchinv un-rescued stock-Hub drafter K=6 (#730)",
                "binding_gate": "5pct_TPS_reproduction_gap",
                "acceptance_tps_model": "linear_h_eq_delta_e_accept_block_efficiency",
                "drift_prior_band": [DRIFT_LO, DRIFT_HI],
                "transfer_band": [MC_LO, MC_HI],
                "local_k6_wall_tps": out["inputs"]["local_k6_wall_tps"],
                "e_accept_exact": out["inputs"]["e_accept_exact_median"],
                "central_optimistic_public": out["inputs"]["central_optimistic_public"],
                "honest_stock_central_band": [HONEST_STOCK_LO, HONEST_STOCK_HI],
            },
            tags=["pr737", "kanna", "analysis_only", "private-dq-risk", "730-fire",
                  "5pct-repro-gate-binding", "bar-nonbinding", "two-risk-ranking"],
        )
        b = out["step1_budgets"]["G2_bar_NONbinding"]
        summary = {
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "primary_metric_name": v["primary_metric_name"],
            "primary_metric_value": v["primary_metric_value"],
            "test_metric_name": v["test_metric_name"],
            "test_metric_value": v["test_metric_value"],
            # the binding gate (decision)
            "p_dq_private_gate_at_prior": v["primary_metric_value"],
            "accept_drift_budget_delta_max": v["test_metric_value"],
            "binding_gate_repro_5pct": 1,
            "delta_max_binding_repro": REPRO_GATE,
            "p_dq_binding_repro_at_prior": v["p_dq_binding_repro_at_prior"],
            # the non-binding bar (PR step-1 literal)
            "delta_max_bar_optimistic_167p9": b["delta_max_optimistic_167p9"],
            "delta_max_bar_honeststock_155": b["delta_max_honeststock_mid_155"],
            "p_dq_bar_at_prior_nonbinding": v["p_dq_bar_at_prior_nonbinding"],
            # coin-flips
            "coinflip_delta_repro": v["coinflip_delta_repro"],
            "coinflip_delta_bar_optimistic": v["coinflip_delta_bar_optimistic"],
            # anchors
            "drift_flagship_4p3pct": DRIFT_FLAGSHIP,
            "drift_prior_center_6p5pct": PRIOR_CENTER,
            "drift_44proxy_12p4pct": DRIFT_44_PROXY,
            "e_accept_exact": out["inputs"]["e_accept_exact_median"],
            "accept_rate_per_token": out["inputs"]["accept_rate_per_token"],
            "local_k6_wall_tps": out["inputs"]["local_k6_wall_tps"],
            "central_optimistic_public": out["inputs"]["central_optimistic_public"],
            "two_risk_ranking_verdict": v["two_risk_ranking"],
            "fire_now_verdict": v["fire_now"],
        }
        run.summary.update(summary)
        wandb.log(summary)
        try:
            wandb.log({"p_dq_vs_delta_stock_curve": wandb.Image(plot_path)})
        except Exception as e:
            print("plot-log skipped:", e)
        print("WANDB_RUN_ID", run.id)
        print("WANDB_RUN_URL", run.url)
        wandb.finish()


if __name__ == "__main__":
    main()
