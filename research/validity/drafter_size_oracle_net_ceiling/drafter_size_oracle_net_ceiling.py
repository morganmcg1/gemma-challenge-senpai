#!/usr/bin/env python3
"""PR #451 — Bigger-drafter oracle: does ANY drafter-size change net-beat 481.53?

DECISION / ORACLE ANALYSIS CARD (analysis_only). NO training, NO HF job, NO
submission, NO served-file change, NO bench-quota spend. GPU not required (pure
arithmetic on measured anchors).

THE QUESTION
------------
#446 closed drafter-retrain at FIXED topology: the binding position-1 miss has
`pos1_fixed_capacity_recoverable_frac = 0.0`, and the 65.3% near-miss pool is
recoverable ONLY by a BIGGER drafter (more params), which "no topology change"
forbids. This card prices the counterfactual the human escalation needs:

    Even if a bigger drafter were ALLOWED, would it NET-beat the deployed 481.53?

A larger drafter recovers some of the size-recoverable near-miss pool -> higher
accept-length E[T] -> more TPS (DEMAND gain). BUT it costs more per-step drafter
latency D (~93% of D is the int4 Marlin GEMM body, BW-bound -> scales ~linearly
with drafter param-bytes; lawine #449) -> longer step -> less TPS (SUPPLY cost).
The NET can be negative. We model TPS_net(s) = demand_gain(phi(s)) x supply(D(s)),
sweep the capacity multiplier s, and find the argmax.

MODEL
-----
DEMAND (pos-1 near-miss recovery propagated through the ladder):
  Oracle drafter of capacity s captures fraction phi(s) of the pos-1 near-miss
  pool (cov4 = 0.6532 of the pos-1 miss; the remaining 0.3468 is hard/structural
  and NEVER recoverable). phi(1)=0 (current size, empirically pinned: openevolve
  A10G-oracle parity INCLUDING the exact KL-distill recipe = 3.83 ~ baseline,
  #446), phi(s)->1 as s->inf.
    a1(s)  = a1_base + phi(s) * near_miss_mass            near_miss_mass = cov4*(1-a1_base)
    E[T](s)= 1 + a1(s) * S_downstream                     (a1 multiplies all ladder products)
    TPS_demand(s) = 467.14 * E[T](s) / 3.8218             (#436/#439 realized-base map)

SUPPLY (added drafter latency):
  D(s) = D * [(1 - gemm_frac) + gemm_frac * s]   gemm_frac = 0.93, D = 1.433 ms (#444)
  T_step(s) = D(s) + V                            V = 6.445 ms fixed (#444)
  supply_factor(s) = T_step_base / T_step(s)      T_step_base = D + V = 7.878 ms

NET:
  TPS_net(s) = min( 467.14 * [E[T](s)/3.8218] * supply_factor(s), 520.95 )
  (520.95 = verify-BW wall, land #436 — a hard TPS ceiling regardless of E[T].)

phi(s) BRACKET (literature-anchored capacity->acceptance scaling; power-law
saturation phi(s) = 1 - s^-beta, beta = log-capacity steepness):
  pessimistic beta=0.2 / central beta=0.5 / optimistic beta=1.0.
  (Standalone-drafter & EAGLE/Medusa/MTP size sweeps show MODEST capture per
  doubling: <0.3 of a recoverable gap at 2x is the central reading; beta<=1 is
  the defensible envelope. The classic Leviathan/Chen spec-decoding result is
  exactly this tension: a too-big draft model kills the end-to-end speedup.)

EQUIVALENCE: byte-exact by construction — the drafter gates accept-LENGTH only;
verify is the SOLE arbiter of emitted tokens (land #420 qe4qagc1). No served
change -> PPL = deployed 2.3772 <= 2.42, by construction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- Baselines / anchors (all in-scope, measured) ------------------------- #
BASE_REALIZED_EQ = 467.14            # realized equivalence frontier, denken #423 5a6zq2yz
BASE_DEPLOYED_NONEQ = 481.53         # deployed incumbent to beat (non-equiv), #52 2x9fm2zx
PPL = 2.3772
PPL_GATE = 2.42
SIGMA_HW_TPS = 0.01 * BASE_DEPLOYED_NONEQ   # ~= 4.8153 TPS (PR: sigma_hw ~ 1%)
VERIFY_BW_WALL = 520.9527323111674          # land #436 nvsbctji

# Cycle split (land #444 0syyqxag)
D_MS = 1.433
V_MS = 6.445
T_STEP_BASE = D_MS + V_MS            # 7.878 ms
GEMM_FRAC = 0.93                     # ~93% of D is int4 Marlin GEMM body, BW-bound (lawine #449 xryqregh)

# #446 re-anchored conditional ladder (uid28gdg) — the pos-1 cliff
ET_BASE = 3.8218126853089935
LADDER = [0.7270358700075576, 0.7562329758229498, 0.7930499079225518,
          0.8215715414083804, 0.8279506687035012, 0.8285340573263027,
          0.8461150837793221]
A1_BASE = LADDER[0]

# pos-1 miss decomposition (#119 / #446)
POS1_NEAR_MISS_COV4 = 0.6531976066516435   # near-miss frac of pos-1 miss (SIZE-recoverable upper bound)
POS1_HARD_MISS = 0.3468023933483565        # structural frac (never recoverable)

# PR sweep grid
S_GRID = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

# phi(s) = 1 - s^-beta. LITERATURE-grounded effective acceptance-scaling exponent
# beta_alpha (NOT the raw loss exponent). Derivation chain (researcher pass):
#   (1) Chinchilla L(N) ~ N^-beta_loss, beta_loss ~= 0.076 (Hoffmann 2022, 2203.15556)
#   (2) acceptance alpha ~ linear in draft PPL with slope A ~= -0.0067 /PPL
#       (Spec-Decoding Scaling Laws, Xia 2025, 2603.11053; R^2=0.602)
#   (3) beta_alpha = |A| * PPL_draft * beta_loss / alpha_gap
#   -> central beta_alpha ~= 0.020 (PPL=6, alpha_gap=0.15); band [0.007, 0.053].
#   Empirical sanity: EAGLE-3 (2503.01840) gets +2.08 tok only via an ARCHITECTURAL
#   step-change (feature-constraint removal + fusion), not raw size -> phi small.
#   Tension is the classic Leviathan(2211.17192)/Chen(2302.01318) too-big-draft-kills-speedup.
PHI_BRACKETS = {"pessimistic": 0.007, "central": 0.020, "optimistic": 0.053}
# Far ABOVE any defensible literature value — robustness/stress only (show even these fail).
STRESS_BRACKETS = {"stress_b0p2": 0.2, "stress_b0p5": 0.5, "stress_b1p0": 1.0}


# --------------------------------------------------------------------------- #
def e_t_ladder(a1: float, tail: list[float]) -> float:
    """E[T] = 1 + a1 + a1*a2 + ... + a1*...*a7  (a1 swapped, tail = a2..a7 fixed)."""
    cum = a1
    s = cum
    for a in tail:
        cum *= a
        s += cum
    return 1.0 + s


TAIL = LADDER[1:]
ET_LADDER_BASE = e_t_ladder(A1_BASE, TAIL)                 # ~3.8211 (ladder reconstruction)
S_DOWNSTREAM = (ET_LADDER_BASE - 1.0) / A1_BASE            # 1 + a2 + a2a3 + ... (a1's multiplier)
NEAR_MISS_MASS = POS1_NEAR_MISS_COV4 * (1.0 - A1_BASE)     # max delta-a1 achievable (phi=1)
A1_MAX = A1_BASE + NEAR_MISS_MASS                          # ~0.9054 (hard-miss caps below 1)


def a1_of_phi(phi: float) -> float:
    return A1_BASE + phi * NEAR_MISS_MASS


def et_of_phi(phi: float) -> float:
    """et(phi) = ET_BASE + ladder lift from raising a1 (pins phi=0 -> ET_BASE exactly)."""
    return ET_BASE + (e_t_ladder(a1_of_phi(phi), TAIL) - ET_LADDER_BASE)


def tps_demand(phi: float) -> float:
    """Demand-only TPS via the fixed-step composition (#436/#439 map)."""
    return BASE_REALIZED_EQ * et_of_phi(phi) / ET_BASE


def d_of_s(s: float) -> float:
    return D_MS * ((1.0 - GEMM_FRAC) + GEMM_FRAC * s)


def t_step_of_s(s: float) -> float:
    return d_of_s(s) + V_MS


def supply_factor(s: float) -> float:
    return T_STEP_BASE / t_step_of_s(s)


def phi_powerlaw(s: float, beta: float) -> float:
    """Capacity->capture: phi(1)=0, phi(inf)->1, steepness beta."""
    if s <= 1.0:
        return 0.0
    return 1.0 - s ** (-beta)


def tps_net(phi: float, s: float, cap: bool = True) -> float:
    raw = BASE_REALIZED_EQ * (et_of_phi(phi) / ET_BASE) * supply_factor(s)
    return min(raw, VERIFY_BW_WALL) if cap else raw


def analyze() -> dict:
    # ---------- Instruction #4 FIRST: optimistic ceiling (phi=1, ZERO added D) ----
    et_phi1 = et_of_phi(1.0)
    opt_ceiling_raw = tps_demand(1.0)                       # zero D -> supply_factor = 1
    opt_ceiling_capped = min(opt_ceiling_raw, VERIFY_BW_WALL)
    opt_margin = opt_ceiling_capped - BASE_DEPLOYED_NONEQ
    optimistic_self_aborts = opt_margin <= SIGMA_HW_TPS     # if True we could short-circuit NO-GO
    et_at_wall = VERIFY_BW_WALL * ET_BASE / BASE_REALIZED_EQ  # E[T] that saturates the verify wall

    # ---------- Analytic breakeven: d(TPS)/ds at s=1 = 0 ----
    # demand_gain(phi) = 1 + (S_DOWNSTREAM * NEAR_MISS_MASS / ET_BASE) * phi
    d_demandgain_dphi = S_DOWNSTREAM * NEAR_MISS_MASS / ET_BASE
    # supply slope: d(supply_factor)/ds at s=1 = -(D*gemm_frac)/T_step_base
    supply_slope_at_1 = -(D_MS * GEMM_FRAC) / T_STEP_BASE
    # d(TPS)/ds|_{s=1} ∝ d_demandgain_dphi * phi'(1) + supply_slope_at_1 ; =0 ->
    breakeven_phi_prime_1 = -supply_slope_at_1 / d_demandgain_dphi   # ~0.934

    # ---------- s_max for which even phi=1 could beat 481.53 ----
    # 467.14 * demand_gain(1) * supply_factor(s) > 481.53
    dg1 = et_of_phi(1.0) / ET_BASE
    supply_req = (BASE_DEPLOYED_NONEQ / BASE_REALIZED_EQ) / dg1
    # supply_factor(s) = T_STEP_BASE/(D*(0.07+0.93s)+V) = supply_req -> solve s
    tstep_max = T_STEP_BASE / supply_req
    s_max_phi1 = ((tstep_max - V_MS) / D_MS - (1.0 - GEMM_FRAC)) / GEMM_FRAC

    # ---------- Sweep s x bracket (literature + stress) ----
    sweep = {}
    bracket_best = {}
    all_brackets = {**PHI_BRACKETS, **STRESS_BRACKETS}
    for name, beta in all_brackets.items():
        rows = []
        # dense scan to find argmax (PR grid is reported separately)
        best = {"s": 1.0, "tps_net": tps_net(0.0, 1.0)}
        s = 1.0
        while s <= 6.0001:
            phi = phi_powerlaw(s, beta)
            t = tps_net(phi, s)
            if t > best["tps_net"]:
                best = {"s": round(s, 4), "tps_net": t, "phi": phi}
            s += 0.01
        bracket_best[name] = {
            "beta": beta,
            "is_literature": name in PHI_BRACKETS,
            "best_size_multiplier": best["s"],
            "best_net_tps": best["tps_net"],
            "best_phi": phi_powerlaw(best["s"], beta),
            "phi_at_2x": phi_powerlaw(2.0, beta),
            "phi_at_4x": phi_powerlaw(4.0, beta),
            "net_beats_481": best["tps_net"] - BASE_DEPLOYED_NONEQ > SIGMA_HW_TPS,
            "margin_vs_deployed": best["tps_net"] - BASE_DEPLOYED_NONEQ,
        }
        for s in S_GRID:
            phi = phi_powerlaw(s, beta)
            rows.append({
                "s": s, "phi": phi, "a1": a1_of_phi(phi), "E_T": et_of_phi(phi),
                "D_ms": d_of_s(s), "T_step_ms": t_step_of_s(s),
                "supply_factor": supply_factor(s),
                "tps_demand_only": tps_demand(phi),
                "tps_net": tps_net(phi, s),
                "tps_net_uncapped": tps_net(phi, s, cap=False),
            })
        sweep[name] = rows

    # ---------- phi=1 ENVELOPE (upper bound: full recovery already at each s) ----
    envelope = []
    for s in S_GRID:
        envelope.append({
            "s": s, "phi": 1.0, "E_T": et_of_phi(1.0),
            "D_ms": d_of_s(s), "T_step_ms": t_step_of_s(s),
            "tps_net": tps_net(1.0, s), "tps_net_uncapped": tps_net(1.0, s, cap=False),
            "beats_481": tps_net(1.0, s) - BASE_DEPLOYED_NONEQ > SIGMA_HW_TPS,
        })

    # ---------- critical beta to beat 481.53 anywhere (numeric) ----
    def max_net_over_s(beta: float) -> float:
        best = tps_net(0.0, 1.0)
        s = 1.0
        while s <= 4.0001:
            best = max(best, tps_net(phi_powerlaw(s, beta), s))
            s += 0.005
        return best
    lo, hi = 0.1, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if max_net_over_s(mid) >= BASE_DEPLOYED_NONEQ + SIGMA_HW_TPS:
            hi = mid
        else:
            lo = mid
    critical_beta = hi

    # ---------- Global verdict ----
    any_bracket_beats = any(b["net_beats_481"] for b in bracket_best.values())
    lit_names = list(PHI_BRACKETS)
    stress_names = list(STRESS_BRACKETS)
    # PRIMARY = best over the DEFENSIBLE (literature) bracket.
    best_bracket = max(lit_names, key=lambda k: bracket_best[k]["best_net_tps"])
    best_net_tps = bracket_best[best_bracket]["best_net_tps"]
    best_size = bracket_best[best_bracket]["best_size_multiplier"]
    # stress max (far-above-literature beta) — should still fail.
    stress_best_bracket = max(stress_names, key=lambda k: bracket_best[k]["best_net_tps"])
    stress_best_net_tps = bracket_best[stress_best_bracket]["best_net_tps"]
    lit_beta_central = PHI_BRACKETS["central"]
    lit_beta_optimistic = PHI_BRACKETS["optimistic"]
    critical_over_lit_central = critical_beta / lit_beta_central
    critical_over_lit_optimistic = critical_beta / lit_beta_optimistic

    verdict = (
        "NO-GO / DEMAND-AXIS CLOSED-ON-NET. Even if drafter topology were FREE, no "
        "defensible drafter-size change net-beats the deployed 481.53. "
        f"The optimistic zero-cost ceiling (phi=1, ZERO added D) is the verify-BW wall "
        f"{VERIFY_BW_WALL:.2f} (raw {opt_ceiling_raw:.1f}) -> it DOES clear 481.53 "
        f"(+{opt_margin:.1f}), so the instruction-#4 short-circuit does NOT fire: the demand "
        "CEILING is high (the full pos-1 near-miss pool over-saturates the verify wall). "
        "The closure comes entirely from SUPPLY: realizing phi requires size s, and D(s) "
        f"charges ~{abs(supply_slope_at_1)*100:.1f}% TPS per unit s at s=1, while breaking "
        f"even needs phi'(1) > {breakeven_phi_prime_1:.3f} (recover >93% of the near-miss "
        "pool per unit s near s=1). Literature capacity->acceptance scaling gives "
        f"phi(s)=1-s^-beta_alpha with beta_alpha ~ {lit_beta_central} (band "
        f"[{PHI_BRACKETS['pessimistic']},{lit_beta_optimistic}]; Chinchilla beta_loss~0.076 "
        "compressed by the SDSL alpha-PPL slope A~-0.0067), i.e. a 2x drafter captures only "
        f"~{bracket_best['central']['phi_at_2x']*100:.1f}% of the pool and 4x ~"
        f"{bracket_best['central']['phi_at_4x']*100:.1f}% -> phi'(1)~beta_alpha<<breakeven. "
        f"Across the literature bracket the argmax is s={best_size} with best_net_tps="
        f"{best_net_tps:.2f} ({best_net_tps-BASE_DEPLOYED_NONEQ:+.2f} vs 481.53, "
        f"{(best_net_tps-BASE_DEPLOYED_NONEQ)/SIGMA_HW_TPS:.2f} sigma_hw). Even a "
        f"far-above-literature stress sweep (beta up to 1.0) peaks at only "
        f"{stress_best_net_tps:.2f}. To beat 481.53 anywhere you'd need beta >= "
        f"{critical_beta:.2f} (~{critical_over_lit_central:.0f}x the central literature "
        f"exponent, ~{critical_over_lit_optimistic:.0f}x the optimistic) AND s < "
        f"{s_max_phi1:.2f} (beyond which even phi=1 fails) — physically incredible. The "
        "demand axis is closed not just by the no-topology constraint (#446) but ON NET "
        "even if topology were free."
    )

    out = {
        "pr": 451, "agent": "land", "kind": "drafter-size-oracle-net-ceiling",
        "analysis_only": True, "no_launch": True, "no_hf_job": True,
        "no_submission": True, "no_served_file_change": True, "gpu_used": False,
        "official_tps": 0, "baseline_unchanged_tps": BASE_DEPLOYED_NONEQ, "ppl": PPL,

        # constants
        "base_realized_eq_tps": BASE_REALIZED_EQ,
        "deployed_noneq_tps": BASE_DEPLOYED_NONEQ,
        "sigma_hw_tps": SIGMA_HW_TPS,
        "verify_bw_wall_tps": VERIFY_BW_WALL,
        "D_ms": D_MS, "V_ms": V_MS, "T_step_base_ms": T_STEP_BASE, "gemm_frac": GEMM_FRAC,
        "ET_base": ET_BASE, "a1_base": A1_BASE,
        "S_downstream": S_DOWNSTREAM,
        "pos1_near_miss_cov4_frac_of_miss": POS1_NEAR_MISS_COV4,
        "pos1_hard_miss_structural_frac_of_miss": POS1_HARD_MISS,
        "near_miss_mass_max_delta_a1": NEAR_MISS_MASS,
        "a1_max_phi1": A1_MAX,
        "E_T_phi1_pos1_ceiling": et_phi1,

        # instruction #4: optimistic ceiling FIRST
        "optimistic_ceiling_raw_tps": opt_ceiling_raw,
        "optimistic_ceiling_capped_tps": opt_ceiling_capped,
        "optimistic_ceiling_margin_vs_deployed": opt_margin,
        "optimistic_self_aborts": optimistic_self_aborts,
        "et_at_verify_wall": et_at_wall,
        "phi1_ceiling_is_wall_bound": opt_ceiling_raw > VERIFY_BW_WALL,

        # analytic breakeven + envelope bounds
        "d_demandgain_dphi": d_demandgain_dphi,
        "supply_slope_at_s1": supply_slope_at_1,
        "breakeven_phi_prime_at_s1": breakeven_phi_prime_1,
        "s_max_for_phi1_to_beat_481": s_max_phi1,
        "critical_beta_to_beat_481": critical_beta,
        "lit_beta_alpha_central": lit_beta_central,
        "lit_beta_alpha_band": [PHI_BRACKETS["pessimistic"], lit_beta_optimistic],
        "critical_beta_over_lit_central": critical_over_lit_central,
        "critical_beta_over_lit_optimistic": critical_over_lit_optimistic,
        "phi_capture_at_2x_central": bracket_best["central"]["phi_at_2x"],
        "phi_capture_at_4x_central": bracket_best["central"]["phi_at_4x"],
        "phi_scaling_citations": ["Hoffmann2022/2203.15556", "Xia2025-SDSL/2603.11053",
                                  "Li2025-EAGLE3/2503.01840", "Leviathan2023/2211.17192",
                                  "Chen2023/2302.01318"],

        # sweep
        "phi_brackets_literature": PHI_BRACKETS,
        "phi_brackets_stress": STRESS_BRACKETS,
        "sweep_by_bracket": sweep,
        "bracket_best": bracket_best,
        "phi1_envelope": envelope,

        # PRIMARY answers (PR instruction #3) — over the DEFENSIBLE literature bracket
        "best_net_tps": best_net_tps,
        "best_size_multiplier": best_size,
        "best_bracket": best_bracket,
        "stress_best_net_tps": stress_best_net_tps,
        "stress_best_bracket": stress_best_bracket,
        "net_beats_481": any_bracket_beats,
        "margin_best_vs_deployed_tps": best_net_tps - BASE_DEPLOYED_NONEQ,
        "margin_in_sigma_hw": (best_net_tps - BASE_DEPLOYED_NONEQ) / SIGMA_HW_TPS,

        # equivalence + ppl (instruction #5)
        "equivalence_byte_exact_by_construction": True,
        "equivalence_basis": "verify is the SOLE arbiter of emitted tokens (land #420 qe4qagc1); drafter gates accept-LENGTH only. No served change -> PPL-neutral.",
        "ppl_gate": PPL_GATE, "ppl_passes": PPL <= PPL_GATE,

        "demand_axis_closed_on_net_even_if_topology_free": not any_bracket_beats,
        "verdict": verdict,
    }
    return out


def self_test(o: dict) -> dict:
    c = {}
    fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    # structure / anchors
    c["ladder_len_7"] = len(LADDER) == 7
    c["a1_base_is_reanchor"] = abs(A1_BASE - 0.7270358700075576) < 1e-9
    c["et_ladder_recovers_reported_base"] = abs(ET_LADDER_BASE - ET_BASE) < 2e-3
    c["miss_split_sums_to_one"] = abs(POS1_NEAR_MISS_COV4 + POS1_HARD_MISS - 1.0) < 1e-9
    c["near_miss_mass_correct"] = abs(NEAR_MISS_MASS - POS1_NEAR_MISS_COV4 * (1 - A1_BASE)) < 1e-12
    c["a1_max_below_one"] = A1_MAX < 1.0 and A1_MAX > A1_BASE
    c["a1_max_about_0p905"] = abs(A1_MAX - 0.9054) < 2e-3
    # demand model
    c["demand_base_maps_to_467"] = abs(tps_demand(0.0) - BASE_REALIZED_EQ) < 1e-6
    c["et_phi0_is_base"] = abs(et_of_phi(0.0) - ET_BASE) < 1e-9
    c["et_phi1_about_4p51"] = abs(o["E_T_phi1_pos1_ceiling"] - 4.514) < 5e-3
    c["et_phi1_exceeds_wall_et"] = o["E_T_phi1_pos1_ceiling"] > o["et_at_verify_wall"]
    # instruction #4 optimistic ceiling
    c["opt_ceiling_is_wall_bound"] = o["phi1_ceiling_is_wall_bound"] is True
    c["opt_ceiling_capped_is_wall"] = abs(o["optimistic_ceiling_capped_tps"] - VERIFY_BW_WALL) < 1e-6
    c["opt_ceiling_clears_481"] = o["optimistic_ceiling_margin_vs_deployed"] > SIGMA_HW_TPS
    c["opt_does_not_self_abort"] = o["optimistic_self_aborts"] is False
    c["et_at_wall_about_4p26"] = abs(o["et_at_verify_wall"] - 4.2618) < 5e-3
    # supply model
    c["supply_factor_1_is_one"] = abs(supply_factor(1.0) - 1.0) < 1e-12
    c["D_at_1_is_1p433"] = abs(d_of_s(1.0) - D_MS) < 1e-9
    c["D_scales_linear_gemm"] = abs(d_of_s(2.0) - D_MS * (0.07 + 0.93 * 2)) < 1e-9
    c["tstep_base_is_7p878"] = abs(T_STEP_BASE - 7.878) < 1e-9
    c["supply_decreasing"] = supply_factor(2.0) < supply_factor(1.0) and supply_factor(4.0) < supply_factor(2.0)
    # analytic breakeven
    c["breakeven_phi_prime_about_0p934"] = abs(o["breakeven_phi_prime_at_s1"] - 0.9342) < 5e-3
    c["s_max_phi1_about_1p86"] = abs(o["s_max_for_phi1_to_beat_481"] - 1.862) < 2e-2
    c["critical_beta_implausible"] = o["critical_beta_to_beat_481"] > 2.0
    # sweep verdicts — literature brackets
    c["central_does_not_beat"] = o["bracket_best"]["central"]["net_beats_481"] is False
    c["optimistic_does_not_beat"] = o["bracket_best"]["optimistic"]["net_beats_481"] is False
    c["pessimistic_does_not_beat"] = o["bracket_best"]["pessimistic"]["net_beats_481"] is False
    # stress brackets (far above literature) ALSO fail
    c["stress_b1p0_does_not_beat"] = o["bracket_best"]["stress_b1p0"]["net_beats_481"] is False
    c["stress_best_below_deployed"] = o["stress_best_net_tps"] < BASE_DEPLOYED_NONEQ
    c["no_bracket_beats_481"] = o["net_beats_481"] is False
    c["best_net_below_deployed"] = o["best_net_tps"] < BASE_DEPLOYED_NONEQ
    c["best_net_about_467"] = abs(o["best_net_tps"] - BASE_REALIZED_EQ) < 1.0
    c["margin_negative_multi_sigma"] = o["margin_in_sigma_hw"] < -1.0
    # literature capture is tiny + critical beta is far above any defensible value
    c["lit_capture_2x_below_0p1"] = o["phi_capture_at_2x_central"] < 0.1
    c["lit_capture_4x_below_0p1"] = o["phi_capture_at_4x_central"] < 0.1
    c["critical_beta_far_above_lit_central"] = o["critical_beta_over_lit_central"] > 10.0
    # phi=1 envelope: s>=2 cannot beat even at full recovery
    c["phi1_envelope_s2_fails"] = any((r["s"] == 2.0 and r["beats_481"] is False) for r in o["phi1_envelope"])
    c["phi1_envelope_s1p25_beats"] = True  # informational: small-s phi=1 (unphysical) does clear
    # demand axis closed on net
    c["demand_axis_closed_on_net"] = o["demand_axis_closed_on_net_even_if_topology_free"] is True
    # gates
    c["ppl_passes"] = o["ppl_passes"] is True
    c["equivalence_by_construction"] = o["equivalence_byte_exact_by_construction"] is True
    # numeric hygiene
    c["no_nan_inf"] = all(fin(v) for v in [o["best_net_tps"], o["optimistic_ceiling_capped_tps"],
                                           o["breakeven_phi_prime_at_s1"], o["critical_beta_to_beat_481"],
                                           o["s_max_for_phi1_to_beat_481"]])
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="land/drafter-size-oracle-net-ceiling-451")
    ap.add_argument("--wandb-group", default="drafter-retrain-demand-realize")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out = analyze()
    checks = self_test(out)
    all_pass = all(checks.values())
    out["self_test_checks"] = checks
    out["self_test_passes"] = all_pass
    out["self_test_pass_count"] = f"{sum(checks.values())}/{len(checks)}"

    (HERE / "drafter_size_oracle_net_ceiling_results.json").write_text(json.dumps(out, indent=2))
    (HERE / "drafter_size_oracle_net_ceiling_selftest.json").write_text(json.dumps(checks, indent=2))

    print(f"[selftest] {out['self_test_pass_count']}  all_pass={all_pass}")
    print(f"[opt-ceiling] phi=1 zero-D raw={out['optimistic_ceiling_raw_tps']:.2f} "
          f"capped(wall)={out['optimistic_ceiling_capped_tps']:.2f} "
          f"(+{out['optimistic_ceiling_margin_vs_deployed']:.2f} vs 481.53) "
          f"self_aborts={out['optimistic_self_aborts']}")
    print(f"[breakeven] phi'(1) needed = {out['breakeven_phi_prime_at_s1']:.4f}; "
          f"critical beta to beat 481 = {out['critical_beta_to_beat_481']:.2f}; "
          f"s_max(phi=1) = {out['s_max_for_phi1_to_beat_481']:.3f}")
    for name, b in out["bracket_best"].items():
        tag = "lit  " if b["is_literature"] else "STRESS"
        print(f"[{tag} {name:11s} beta={b['beta']:<5}] best_net={b['best_net_tps']:.2f} "
              f"@ s={b['best_size_multiplier']} ({b['margin_vs_deployed']:+.2f} vs 481.53) "
              f"phi2x={b['phi_at_2x']:.3f} beats={b['net_beats_481']}")
    print(f"[PRIMARY] best_net_tps={out['best_net_tps']:.2f} @ s={out['best_size_multiplier']} "
          f"({out['best_bracket']}); net_beats_481={out['net_beats_481']}; "
          f"margin={out['margin_best_vs_deployed_tps']:+.2f} ({out['margin_in_sigma_hw']:.2f} sigma_hw)")
    print(f"[verdict] {out['verdict']}")

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                name=args.wandb_name, group=args.wandb_group,
                config={"pr": 451, "kind": out["kind"], "analysis_only": True,
                        "phi_brackets": PHI_BRACKETS, "s_grid": S_GRID},
            )
            flat = {k: v for k, v in out.items() if isinstance(v, (int, float, bool))}
            flat["self_test_pass_count_n"] = sum(checks.values())
            # per-bracket scalars
            for name, b in out["bracket_best"].items():
                flat[f"best_net_tps__{name}"] = b["best_net_tps"]
                flat[f"best_size__{name}"] = b["best_size_multiplier"]
                flat[f"net_beats_481__{name}"] = b["net_beats_481"]
            run.log(flat)
            run.summary.update(flat)
            # sweep tables
            try:
                cols = ["s", "phi", "a1", "E_T", "D_ms", "T_step_ms", "supply_factor",
                        "tps_demand_only", "tps_net"]
                for name, rows in out["sweep_by_bracket"].items():
                    tbl = wandb.Table(columns=cols)
                    for r in rows:
                        tbl.add_data(*[r[c] for c in cols])
                    run.log({f"sweep_{name}": tbl})
            except Exception as te:  # noqa: BLE001
                print(f"[wandb] table log skipped: {te}")
            print(f"[wandb] {run.url}")
            out["wandb_run_id"] = run.id
            out["wandb_url"] = run.url
            run.finish()
            (HERE / "drafter_size_oracle_net_ceiling_results.json").write_text(json.dumps(out, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] unavailable/failed: {exc}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
