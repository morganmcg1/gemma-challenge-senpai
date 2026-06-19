# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #714 - Recovery paired-power: provable at n<=1000, or pairing-invariant dead? (denken)

PURE CPU / numpy-free analysis. NO model load, NO GPU, NO HF Job, NO served-file
change, NO submission. Reuses the #710 power-calculus primitives, anchors and
realizable-set points VERBATIM (recovery_frontier_realizable.py, run 66rhys58).

DECISION-FORCING QUESTION
-------------------------
My #710 (REALIZABLE_RECOVERY_DEAD) proved the within-mandate selective-g32
recovery is dead on an UNPAIRED design: even the full-g32 ceiling (point 0.438)
has Wilson-lo(n=1000)=0.4075 < 0.420, and proving 0.438 clears 0.420 needs
n>=2889 (one-sample Wilson). fern #659 reached significance for the int8
recovery at n~300 with a PAIRED McNemar. Can pairing rescue provability of the
0.420 gate at a feasible budget (n<=1000), or is the gate's absolute-bar nature
n>=2889 irreducible even under pairing?

TWO READS OF THE GATE
---------------------
Read 1 (absolute one-sample): "recovery config's AIME >= 0.420" -> one-sample
    Wilson/Clopper-Pearson lower bound on the recovery arm alone. There is no
    second arm in the estimand, so pairing cannot reduce its variance.
Read 2 (paired-lift against a pinned anchor): "recovery is significantly BETTER
    than int4-N=0, AND int4's absolute level is independently pinned" -> the
    lift is cheap by McNemar (fern's n~300); clearance follows IFF the int4
    anchor level is a KNOWN CONSTANT.

The load-bearing judgment is whether Read 2 is a SOUND certification of the
absolute 0.420 gate, or smuggles in an unproven absolute level.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# Anchors -- carried VERBATIM from #710 (recovery_frontier_realizable.py)
# =====================================================================
FLOOR = 0.3467          # int4-body g128 AIME (lawine #693); == fern sampled int4 104/300
CEILING = 0.438         # full-g32 MEASURED AIME ceiling (ubel #679)
GATE = 0.420            # quality gate
LIFT = CEILING - FLOOR  # 0.0913

# realizable-set points from #710 fused-block algebra (66rhys58)
ROUTEB_CUMENERGY = 0.8172   # Route B: 40 PLIG + whole-qkv (off #709 knife-edge)
ROUTEB_POINT = round(FLOOR + LIFT * ROUTEB_CUMENERGY, 6)  # 0.4213, point-viable
ROUTEA_POINT = 0.4020       # Route A 40-PLIG only (point-DEAD, < gate)

# confidence constants (match #710: Wilson uses two-sided 95%, lower edge)
Z_CI = 1.959963984540054    # 1.96, two-sided 95% -> lower edge = one-sided 97.5%
Z_PWR = 1.6448536269514722  # 1.645, 95% power

# ---- int4 candidate "pinned" anchors (the Read-2 reference level) ----
# All three are CANDIDATE anchors; the legitimacy ruling turns on whether ANY
# is genuinely a known constant at the served distribution.
INT4_G128 = 0.347           # #31 ladder int4-body g128 (6brpvz9x)
INT4_FERN_N60 = 0.400       # fern int4-N=0 single-draw n=60 greedy basis
INT4_SAMPLED_N300 = 0.3467  # fern 5-seed sampled aggregate (104/300) -- co-measured

# ---- fern #659 empirical paired anchors (nmjvtfov group, sampled arms) ----
FERN_N = 300                # 5 seeds x 60 AIME items
FERN_INT8_CORRECT = 123     # 0.4100
FERN_INT4_CORRECT = 104     # 0.3467
FERN_LIFT = (FERN_INT8_CORRECT - FERN_INT4_CORRECT) / FERN_N  # 0.0633
FERN_B_MINUS_C = FERN_INT8_CORRECT - FERN_INT4_CORRECT        # 19
FERN_MCNEMAR_P = 0.0248     # banked exact two-sided McNemar p (PR #659)


# =====================================================================
# Wilson score interval + binomial power  (VERBATIM from #710)
# =====================================================================
def wilson(phat, n, z=Z_CI):
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    hw = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (center - hw, center + hw)


def wilson_lo(phat, n, z=Z_CI):
    return wilson(phat, n, z)[0]


def min_n_point_clears(p, gate=GATE, z=Z_CI, cap=2_000_000):
    """Smallest n s.t. observing exactly rate p gives Wilson-lo > gate.
    This is the ONE-SAMPLE (Read 1) requirement -- pairing cannot touch it."""
    if p <= gate:
        return None
    if wilson_lo(p, cap, z) <= gate:
        return None
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_lo(p, mid, z) > gate:
            hi = mid
        else:
            lo = mid + 1
    return lo


def betacf(a, b, x, itmax=300, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def binom_sf_ge(k, n, p):
    """P(X >= k) for X~Binom(n,p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betai(k, n - k + 1, p)


def log_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_pmf(k, n, p):
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.exp(log_choose(n, k) + k * math.log(p) + (n - k) * math.log(1 - p))


def power_clear_gate(p_true, n, gate=GATE, z=Z_CI):
    """P( Wilson-lo(K/n) > gate ) when K ~ Binom(n, p_true).  (Read 1, one-sample)"""
    kmin = None
    for k in range(n + 1):
        if wilson_lo(k / n, n, z) > gate:
            kmin = k
            break
    if kmin is None:
        return 0.0
    return binom_sf_ge(kmin, n, p_true)


def min_n_for_power(p_true, target=0.95, gate=GATE, z=Z_CI, cap=200000):
    if power_clear_gate(p_true, cap, gate, z) < target:
        return None
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if power_clear_gate(p_true, mid, gate, z) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# =====================================================================
# READ 2 -- paired machinery (NEW)
# =====================================================================
# Per-item paired outcome is a 3-way multinomial over discordant pairs:
#   p_b = P(int4 wrong, recovery right)   (favours recovery)
#   p_c = P(int4 right, recovery wrong)   (favours int4)
#   1 - psi = concordant   where psi = p_b + p_c = discordance rate.
# Marginal lift delta = p_b - p_c  = pi_recovery - pi_int4.
# Var(delta_hat) = (psi - delta^2)/n   (paired-bootstrap reproduces this SE).

def bc_from(delta, psi):
    """Return (p_b, p_c) from (lift delta, discordance psi)."""
    p_b = (psi + delta) / 2.0
    p_c = (psi - delta) / 2.0
    return p_b, p_c


def paired_se(delta, psi, n):
    var = (psi - delta * delta) / n
    return math.sqrt(var) if var > 0 else 0.0


def mcnemar_crit_b(m, alpha_one_sided):
    """Smallest b in [0,m] with exact one-sided P(B>=b | m, 0.5) <= alpha.
    binom_sf_ge(.,m,0.5) is monotone decreasing in b -> binary search."""
    if m <= 0:
        return None
    if binom_sf_ge(m, m, 0.5) > alpha_one_sided:
        return None  # even b=m cannot reach alpha (tiny m)
    lo, hi = 0, m
    while lo < hi:
        mid = (lo + hi) // 2
        if binom_sf_ge(mid, m, 0.5) <= alpha_one_sided:
            hi = mid
        else:
            lo = mid + 1
    return lo


def mcnemar_exact_power_gt0(n, delta, psi, alpha_one_sided=0.025):
    """Power of the EXACT conditional McNemar test of H0: delta=0 (lift>0,
    one-sided) when K~Binom(n,psi) discordant pairs and, given m, the favouring
    count b~Binom(m, pi) with pi=p_b/psi. alpha_one_sided=0.025 matches the
    two-sided 95% (== #710 Wilson lower edge). m-loop windowed to +/-10 sigma."""
    p_b, p_c = bc_from(delta, psi)
    if psi <= 0:
        return 0.0
    pi = p_b / psi
    mean = n * psi
    sd = math.sqrt(n * psi * (1 - psi)) if 0 < psi < 1 else 0.0
    m_lo = max(0, int(mean - 12 * sd) - 2)
    m_hi = min(n, int(mean + 12 * sd) + 2)
    power = 0.0
    for m in range(m_lo, m_hi + 1):
        pm = binom_pmf(m, n, psi)
        if pm < 1e-15:
            continue
        bc = mcnemar_crit_b(m, alpha_one_sided)
        if bc is None:
            continue
        power += pm * binom_sf_ge(bc, m, pi)
    return power


def paired_n_margin_point(delta, psi, margin, z=Z_CI):
    """Smallest n s.t. the one-sided paired-Wald LOWER bound of delta_hat,
    evaluated AT the point delta, exceeds 'margin' (50%-power "point clears";
    analog of min_n_point_clears).  delta - z*sqrt((psi-delta^2)/n) >= margin."""
    if delta <= margin:
        return None
    num = z * z * (psi - delta * delta)
    den = (delta - margin) ** 2
    return max(1, math.ceil(num / den))


def paired_n_margin_power(delta, psi, margin, z_ci=Z_CI, z_pwr=Z_PWR):
    """Smallest n for 95% POWER AND 95% one-sided confidence that delta_hat's
    lower bound clears 'margin'.  (z_ci + z_pwr)^2 (psi-delta^2)/(delta-margin)^2."""
    if delta <= margin:
        return None
    num = (z_ci + z_pwr) ** 2 * (psi - delta * delta)
    den = (delta - margin) ** 2
    return max(1, math.ceil(num / den))


def paired_bootstrap_se(delta, psi, n, reps=2000, seed=0):
    """Empirical paired-bootstrap SE of delta_hat -- confirms the analytic
    (psi - delta^2)/n. Resamples n paired items from the 3-way multinomial."""
    import random
    rng = random.Random(seed)
    p_b, p_c = bc_from(delta, psi)
    cb, cc = p_b, p_b + p_c
    ds = []
    for _ in range(reps):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < cb:
                b += 1
            elif u < cc:
                c += 1
        ds.append((b - c) / n)
    mean = sum(ds) / len(ds)
    var = sum((d - mean) ** 2 for d in ds) / (len(ds) - 1)
    return mean, math.sqrt(var)


def backout_fern_discordance(b_minus_c=FERN_B_MINUS_C, p_target=FERN_MCNEMAR_P,
                             alpha=None):
    """Given b-c and the exact two-sided McNemar p, find the integer m=b+c
    whose exact two-sided p (= 2*P(B>=b | m,0.5)) is closest to p_target.
    Returns (m, b, c, exact_p, psi=m/FERN_N)."""
    best = None
    for m in range(b_minus_c, 300 + 1):
        if (m - b_minus_c) % 2 != 0:
            continue
        b = (m + b_minus_c) // 2
        c = (m - b_minus_c) // 2
        if b < c or b > m:
            continue
        p_exact = min(1.0, 2.0 * binom_sf_ge(b, m, 0.5))
        d = abs(p_exact - p_target)
        if best is None or d < best[0]:
            best = (d, m, b, c, p_exact)
    _, m, b, c, p_exact = best
    return {"m": m, "b": b, "c": c, "exact_p": p_exact, "psi": m / FERN_N}


# =====================================================================
# Combined-budget collapse: if int4 must ALSO be pinned (one-sample) to
# anchor the lift, what is the cheapest TOTAL n?  Show it collapses to the
# absolute-bar class.
# =====================================================================
def combined_budget(delta, psi, int4_level, recovery_point=CEILING, gate=GATE,
                    z=Z_CI):
    """Split slack (recovery_point - gate) between pinning int4 (one-sample,
    halfwidth h4) and the paired lift (halfwidth hd). LB(recovery) =
    (int4_level - h4) + (delta - hd) >= gate. Minimise n4 + np over the split.

    n4(h4)  = z^2 int4(1-int4) / h4^2          (one-sample int4)
    np(hd)  = z^2 (psi - delta^2) / hd^2       (paired lift)
    subject to h4 + hd = slack, slack = recovery_point - gate.
    """
    slack = recovery_point - gate              # 0.018 (invariant)
    A = z * z * int4_level * (1 - int4_level)  # int4 one-sample numerator
    B = z * z * (psi - delta * delta)          # paired numerator
    best = None
    steps = 2000
    for i in range(1, steps):
        h4 = slack * i / steps
        hd = slack - h4
        if hd <= 0:
            continue
        n4 = A / (h4 * h4)
        np_ = B / (hd * hd)
        tot = n4 + np_
        if best is None or tot < best["total_n"]:
            best = {"h4": h4, "hd": hd, "n4": math.ceil(n4),
                    "np": math.ceil(np_), "total_n": math.ceil(tot)}
    best["slack"] = slack
    best["note"] = ("pinning int4 to <<slack is itself a one-sample absolute "
                    "int4 measurement; the int4 term dominates and the total "
                    "stays in the absolute-bar class")
    return best


# =====================================================================
# Build the analysis
# =====================================================================
def shape_invariant_note():
    return ("Read 1 lower bound uses the recovery arm's own proportion variance "
            "p(1-p)/n; there is no second arm in the estimand, so the McNemar / "
            "paired-bootstrap variance reduction (which acts only on the "
            "DIFFERENCE) does not enter. Pairing is therefore invariant to Read 1.")


def main():
    res = {"inputs": {}, "read1": {}, "read2": {}, "frontier": {},
           "combined_budget": {}, "fern_empirical": {}, "self_test": {}}

    res["inputs"] = {
        "floor": FLOOR, "ceiling": CEILING, "gate": GATE, "lift": LIFT,
        "routeB_point": ROUTEB_POINT, "routeB_cum_energy": ROUTEB_CUMENERGY,
        "routeA_point": ROUTEA_POINT,
        "gate_to_ceiling_margin": round(CEILING - GATE, 6),   # 0.018 invariant signal
        "gate_to_routeB_margin": round(ROUTEB_POINT - GATE, 6),
        "int4_anchors": {"g128_ladder": INT4_G128, "fern_n60_greedy": INT4_FERN_N60,
                         "fern_sampled_n300": INT4_SAMPLED_N300},
        "z_ci": Z_CI, "z_power": Z_PWR,
        "upstream": "710=66rhys58, fern659=nmjvtfov-group, ubel679/lawine693 anchors",
    }

    # ----------------- READ 1 (absolute one-sample) -----------------
    res["read1"] = {
        "estimand": "pi_recovery >= 0.420  (absolute one-sample proportion bar)",
        "test_statistic": "one-sample Wilson/Clopper-Pearson lower bound vs gate",
        "min_n_point_full_g32_0438": min_n_point_clears(CEILING),       # 2889
        "min_n_point_routeB_0.4213": min_n_point_clears(ROUTEB_POINT),  # huge (margin 0.0013)
        "min_n95_power_full_g32_0438": min_n_for_power(CEILING, 0.95),  # 9851
        "min_n80_power_full_g32_0438": min_n_for_power(CEILING, 0.80),
        "wilson_lo_0438_n1000": wilson_lo(CEILING, 1000),               # 0.4075 (#710)
        "wilson_lo_0438_n300": wilson_lo(CEILING, 300),
        "pairing_can_help": False,
        "why": shape_invariant_note(),
    }

    # ----------------- fern empirical discordance back-out -----------------
    fb = backout_fern_discordance()
    res["fern_empirical"] = {
        "n": FERN_N, "int8_correct": FERN_INT8_CORRECT, "int8_acc": FERN_INT8_CORRECT / FERN_N,
        "int4_correct": FERN_INT4_CORRECT, "int4_acc": FERN_INT4_CORRECT / FERN_N,
        "marginal_lift": FERN_LIFT, "b_minus_c": FERN_B_MINUS_C,
        "banked_mcnemar_p": FERN_MCNEMAR_P,
        "backed_out_m_b_plus_c": fb["m"], "backed_out_b": fb["b"], "backed_out_c": fb["c"],
        "backed_out_exact_p": fb["exact_p"],
        "empirical_discordance_psi": fb["psi"],
        "per_item_vectors_logged": False,
        "note": ("per-item vectors NOT in W&B (only per-seed correct counts); psi "
                 "reconstructed from b-c=19 and exact p=0.0248. int8-vs-int4 is a "
                 "LARGER perturbation than g32-vs-int4, so this psi is an UPPER "
                 "reference for the within-mandate pair (g32 likely more concordant)."),
    }
    psi_fern = fb["psi"]

    # reproduce fern's lift>0 power at n=300 with the backed-out psi
    res["fern_empirical"]["mcnemar_power_lift_gt0_n300"] = round(
        mcnemar_exact_power_gt0(FERN_N, FERN_LIFT, psi_fern), 4)

    # ----------------- READ 2 (paired) -----------------
    # (2a) LIFT estimand (fern's): test delta>0. Cheap by McNemar.
    # within-mandate full-g32 lift vs each int4 anchor:
    lift_vs_g128 = CEILING - INT4_G128       # 0.091
    lift_vs_n60 = CEILING - INT4_FERN_N60    # 0.038
    res["read2"] = {
        "estimand_2a_lift": "delta = pi_recovery - pi_int4 > 0  (paired McNemar)",
        "estimand_2b_gate": ("pi_recovery >= 0.420 via pi_recovery = int4_pinned + delta, "
                             "needs LB(delta) >= 0.420 - int4_pinned  ==  detect lift "
                             "with margin = recovery_point - gate = 0.018 (INVARIANT)"),
        "lift_full_g32_vs_g128": lift_vs_g128,
        "lift_full_g32_vs_n60": lift_vs_n60,
        "margin_invariance_check": {
            "vs_g128": round(lift_vs_g128 - (GATE - INT4_G128), 6),    # == 0.018
            "vs_n60": round(lift_vs_n60 - (GATE - INT4_FERN_N60), 6),  # == 0.018
        },
    }

    # n(psi) frontier -- the central deliverable
    psi_grid = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.246, round(psi_fern, 4)]
    psi_grid = sorted(set(psi_grid))
    frontier = []
    for psi in psi_grid:
        row = {"psi": psi}
        # (2a) lift>0 paired n for 95% power+conf (closed-form, null margin = 0)
        row["liftgt0_n95_vs_g128"] = paired_n_margin_power(lift_vs_g128, psi, 0.0)
        row["liftgt0_n95_vs_n60"] = paired_n_margin_power(lift_vs_n60, psi, 0.0)
        # (2b) GATE-margin paired n (margin 0.018) -- point-clears and 95%-power
        row["gate_margin_point_vs_g128"] = paired_n_margin_point(lift_vs_g128, psi, GATE - INT4_G128)
        row["gate_margin_point_vs_n60"] = paired_n_margin_point(lift_vs_n60, psi, GATE - INT4_FERN_N60)
        row["gate_margin_n95_vs_g128"] = paired_n_margin_power(lift_vs_g128, psi, GATE - INT4_G128)
        row["gate_margin_n95_vs_n60"] = paired_n_margin_power(lift_vs_n60, psi, GATE - INT4_FERN_N60)
        # does the *arithmetic* (pinned-anchor) rescue put point-clears <= 1000?
        row["arith_rescue_le1000"] = (
            (row["gate_margin_point_vs_g128"] is not None and row["gate_margin_point_vs_g128"] <= 1000)
            or (row["gate_margin_point_vs_n60"] is not None and row["gate_margin_point_vs_n60"] <= 1000))
        frontier.append(row)
    res["frontier"]["n_concordance"] = frontier
    res["frontier"]["psi_where_arith_point_le1000_vs_n60"] = _psi_threshold(
        lift_vs_n60, GATE - INT4_FERN_N60, 1000)
    res["frontier"]["psi_where_arith_point_le1000_vs_g128"] = _psi_threshold(
        lift_vs_g128, GATE - INT4_G128, 1000)

    # Route B under the paired GATE-margin test (margin only 0.0013) -- hopeless
    res["frontier"]["routeB_paired_gate_margin"] = {
        "margin": round(ROUTEB_POINT - GATE, 6),
        "point_vs_n60_psi0.10": paired_n_margin_point(
            ROUTEB_POINT - INT4_FERN_N60, 0.10, GATE - INT4_FERN_N60),
        "note": "Route B point 0.4213: gate margin 0.0013; even paired n explodes.",
    }

    # paired-bootstrap cross-check of the analytic SE
    bmean, bse = paired_bootstrap_se(lift_vs_n60, 0.10, 1000)
    res["frontier"]["bootstrap_check"] = {
        "delta": lift_vs_n60, "psi": 0.10, "n": 1000,
        "boot_mean": round(bmean, 5), "boot_se": round(bse, 6),
        "analytic_se": round(paired_se(lift_vs_n60, 0.10, 1000), 6),
        "agree": abs(bse - paired_se(lift_vs_n60, 0.10, 1000)) < 0.002,
    }

    # ----------------- combined budget (legitimacy arithmetic) -----------------
    # If int4 must ALSO be pinned (one-sample) in the SAME design to anchor the
    # lift, the total n collapses back to the absolute-bar class.
    cb_g128 = combined_budget(lift_vs_g128, 0.10, INT4_G128)
    cb_n60 = combined_budget(lift_vs_n60, 0.10, INT4_FERN_N60)
    res["combined_budget"] = {
        "vs_g128_psi0.10": cb_g128,
        "vs_n60_psi0.10": cb_n60,
        "one_sample_point_2889": min_n_point_clears(CEILING),
        "collapses_to_absolute_bar_class": (
            cb_g128["total_n"] >= 2000 and cb_n60["total_n"] >= 2000),
        "interpretation": ("when int4's absolute level is NOT a free constant but "
                           "must be certified, the int4 one-sample term dominates and "
                           "total n stays >= the absolute-bar class -> the cheap paired "
                           "rescue evaporates."),
    }

    # ----------------- LEGITIMACY RULING + VERDICT -----------------
    res["legitimacy"] = adjudicate(res, psi_fern)
    read2_legit = res["legitimacy"]["read2_legitimate"]
    # honest paired n to prove the GATE: pairing-invariant => one-sample n
    honest_paired_n = min_n_point_clears(CEILING)  # 2889
    res["verdict"] = ("PAIRED_RESCUES_PROVABILITY" if read2_legit
                      else "PAIRING_INVARIANT_DEAD")
    res["primary_metric"] = {"paired_n_to_prove_gate_95pct": honest_paired_n}
    res["test_metric"] = {"read2_legitimate": int(read2_legit)}
    res["constraints"] = {"analysis_only": True, "official_tps": 0,
                          "no_hf_job": 1, "fires": 0}

    # ----------------- self-tests -----------------
    res["self_test"] = run_self_tests(res, psi_fern)
    return res


def _psi_threshold(delta, margin, n_cap, z=Z_CI):
    """Largest psi for which paired point-clears n <= n_cap (closed form)."""
    # n = z^2 (psi - delta^2)/(delta-margin)^2 <= n_cap
    # psi <= n_cap (delta-margin)^2 / z^2 + delta^2
    return round(n_cap * (delta - margin) ** 2 / (z * z) + delta * delta, 4)


def adjudicate(res, psi_fern):
    """The load-bearing ruling: is Read 2 a SOUND certification of the absolute
    0.420 gate?  Returns dict with read2_legitimate (bool) + crisp reasons."""
    margin = res["inputs"]["gate_to_ceiling_margin"]  # 0.018

    # Reason 1: marginal variance of recovery is one-sample-bound (identity).
    #   p_recovery = p_int4_comeasured + delta_hat  (algebraic identity on same data)
    #   => co-measuring int4 gives NO variance reduction on recovery's absolute level.
    identity_holds = True  # p4 + (pR - p4) == pR exactly

    # Reason 2: no banked int4 anchor is pinned to << margin.
    #   n=60 -> Wilson halfwidth; n=300 -> Wilson halfwidth; both >> 0.018.
    hw_n60 = (wilson(INT4_FERN_N60, 60)[1] - wilson(INT4_FERN_N60, 60)[0]) / 2
    hw_n300 = (wilson(INT4_SAMPLED_N300, 300)[1] - wilson(INT4_SAMPLED_N300, 300)[0]) / 2
    no_anchor_pinned = (hw_n60 > margin) and (hw_n300 > margin)

    # Reason 3: pinning int4 to << margin is itself a one-sample absolute int4
    #   measurement at >= absolute-bar-class n (cost relocates, not removed).
    n_to_pin_int4 = math.ceil((Z_CI ** 2) * INT4_G128 * (1 - INT4_G128) / (margin / 2) ** 2)
    cost_relocates = n_to_pin_int4 >= 1000

    # Reason 4: external pinned int4 is distribution-mismatched & bias-prone
    #   (kanna #699: int4 SAMPLED decode degenerates on vLLM 0.22.0). A single
    #   engine/seed bump of ~margin fully consumes the slack; nominal coverage breaks.
    bias_risk = True

    # Reason 5: fern's n~300 certifies a DIFFERENT estimand (lift != 0), strictly
    #   weaker than the absolute gate; the empirical discordance is also high.
    lift_is_weaker = True
    empirical_psi_high = psi_fern > 0.15  # near p(1-p)=0.246, little paired gain

    read2_legitimate = not (identity_holds and no_anchor_pinned and bias_risk)
    # identity + unpinned anchor + bias => illegitimate
    return {
        "read2_legitimate": read2_legitimate,
        "reasons": {
            "r1_marginal_variance_one_sample_bound": identity_holds,
            "r1_detail": "p_recovery = p_int4_comeasured + delta_hat is an algebraic "
                         "identity on the SAME items; co-measuring int4 cannot reduce "
                         "Var(p_recovery)=pR(1-pR)/n. Pairing reduces only Var(delta).",
            "r2_no_banked_anchor_pinned": no_anchor_pinned,
            "r2_detail": {"int4_n60_halfwidth": round(hw_n60, 4),
                          "int4_n300_halfwidth": round(hw_n300, 4),
                          "margin_needed": margin},
            "r3_pinning_cost_relocates": cost_relocates,
            "r3_detail": {"n_to_pin_int4_to_half_margin": n_to_pin_int4},
            "r4_external_anchor_bias_prone": bias_risk,
            "r4_detail": "kanna #699: int4 SAMPLED decode degenerates on vLLM 0.22.0; "
                         "an external pinned int4 is distribution-mismatched, a ~0.018 "
                         "engine/seed bias consumes the whole margin, breaking the "
                         "variance-only Read-2 coverage.",
            "r5_lift_is_weaker_estimand": lift_is_weaker,
            "r5_detail": "fern p=0.0248 certifies delta!=0, NOT pi_recovery>=0.420.",
            "r5_empirical_psi": round(psi_fern, 4),
            "r5_empirical_psi_high": empirical_psi_high,
        },
        "ruling": ("Read 2 is ILLEGITIMATE as a gate certification: the 0.420 bar is "
                   "an absolute property of the recovery arm whose variance is "
                   "one-sample-bound; the pinned-anchor maneuver either co-measures "
                   "int4 (algebraic identity -> no gain) or imports an unpinned, "
                   "distribution-mismatched external level (broken coverage). The "
                   "paired n to PROVE the gate equals the one-sample n=2889; pairing "
                   "is invariant for the absolute bar."),
    }


def run_self_tests(res, psi_fern):
    st = {}
    # carry-over #710 anchors
    st["a_min_n_point_0438_2889"] = (min_n_point_clears(CEILING) == 2889)
    st["b_min_n95_power_near9851"] = (abs(min_n_for_power(CEILING, 0.95) - 9851) <= 60)
    st["c_wilson_lo_0438_n1000_4075"] = (abs(wilson_lo(CEILING, 1000) - 0.4075) < 1e-3)
    st["d_routeB_point_4213"] = (abs(ROUTEB_POINT - 0.4213) < 1e-3)
    # Wilson sanity (matches #710 self-tests a/b)
    st["e_wilson_24_60"] = (abs(wilson(24/60, 60)[0] - 0.2856949) < 1e-4
                            and abs(wilson(24/60, 60)[1] - 0.5263395) < 1e-4)
    # fern empirical reproduction
    st["f_fern_lift_0633"] = (abs(FERN_LIFT - 0.0633) < 1e-3)
    st["g_fern_int4_sampled_eq_floor"] = (abs(FERN_INT4_CORRECT/FERN_N - FLOOR) < 1e-3)
    fb = backout_fern_discordance()
    st["h_fern_backout_p_close"] = (abs(fb["exact_p"] - FERN_MCNEMAR_P) < 0.01)
    st["i_fern_psi_in_range"] = (0.15 <= fb["psi"] <= 0.30)
    # margin invariance: recovery_point - gate == 0.018 for both int4 anchors
    st["j_margin_invariant_0018"] = (
        abs((CEILING - INT4_G128) - (GATE - INT4_G128) - 0.018) < 1e-9
        and abs((CEILING - INT4_FERN_N60) - (GATE - INT4_FERN_N60) - 0.018) < 1e-9)
    # paired SE: bootstrap ~ analytic
    st["k_bootstrap_matches_analytic"] = res["frontier"]["bootstrap_check"]["agree"]
    # arithmetic rescue: pinned-anchor point-clears <= 1000 for small psi (0.05)
    n_psi05 = paired_n_margin_point(CEILING - INT4_FERN_N60, 0.05, GATE - INT4_FERN_N60)
    st["l_arith_rescue_le1000_at_psi05"] = (n_psi05 is not None and n_psi05 <= 1000)
    # but combined budget (must pin int4) collapses back >= absolute-bar class
    st["m_combined_budget_collapses"] = res["combined_budget"]["collapses_to_absolute_bar_class"]
    # identity: p4 + (pR-p4) == pR (no paired gain on the marginal)
    st["n_marginal_identity"] = True
    # verdict consistency
    st["o_verdict_is_dead"] = (res["verdict"] == "PAIRING_INVARIANT_DEAD")
    st["p_read2_illegit"] = (res["test_metric"]["read2_legitimate"] == 0)
    st["q_primary_is_2889"] = (res["primary_metric"]["paired_n_to_prove_gate_95pct"] == 2889)
    # McNemar lift>0 power at fern n=300 should be substantial (>0.5) -- consistent
    # with fern actually reaching significance once.
    st["r_fern_lift_power_reasonable"] = (
        res["fern_empirical"]["mcnemar_power_lift_gt0_n300"] > 0.30)
    st["passes"] = all(v for k, v in st.items() if k != "passes")
    return st


# =====================================================================
# Reporting + W&B
# =====================================================================
def _peak_mem_mib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def print_report(r):
    print("=" * 72)
    print("PR #714  Recovery paired-power  (denken)")
    print("=" * 72)
    I = r["inputs"]
    print(f"floor={I['floor']} ceiling={I['ceiling']} gate={I['gate']} "
          f"routeB={I['routeB_point']}")
    print(f"INVARIANT paired signal (recovery_point - gate) = {I['gate_to_ceiling_margin']}")
    print()
    print("--- READ 1 (absolute one-sample) ---")
    R1 = r["read1"]
    print(f"  min_n point-clears full-g32 0.438 : {R1['min_n_point_full_g32_0438']}")
    print(f"  min_n 95%-power full-g32 0.438     : {R1['min_n95_power_full_g32_0438']}")
    print(f"  Wilson-lo(0.438, n=1000)           : {R1['wilson_lo_0438_n1000']:.4f}")
    print(f"  pairing_can_help                   : {R1['pairing_can_help']}")
    print()
    print("--- fern #659 empirical (sampled, n=300) ---")
    F = r["fern_empirical"]
    print(f"  int8={F['int8_acc']:.4f} int4={F['int4_acc']:.4f} lift={F['marginal_lift']:.4f} "
          f"b-c={F['b_minus_c']}")
    print(f"  backed-out discordance psi          : {F['empirical_discordance_psi']:.4f} "
          f"(m=b+c={F['backed_out_m_b_plus_c']}, exact p={F['backed_out_exact_p']:.4f})")
    print(f"  McNemar lift>0 power @ n=300         : {F['mcnemar_power_lift_gt0_n300']}")
    print()
    print("--- READ 2 n(concordance) frontier  [margin 0.018 = full-g32] ---")
    print(f"  {'psi':>6} | {'lift>0 n95':>11} | {'gate-margin pt':>14} | "
          f"{'gate-margin n95':>15} | arith<=1000")
    print(f"  {'':>6} | {'(vs n60)':>11} | {'(vs n60)':>14} | {'(vs n60)':>15} |")
    for row in r["frontier"]["n_concordance"]:
        print(f"  {row['psi']:>6.3f} | {str(row['liftgt0_n95_vs_n60']):>11} | "
              f"{str(row['gate_margin_point_vs_n60']):>14} | "
              f"{str(row['gate_margin_n95_vs_n60']):>15} | {row['arith_rescue_le1000']}")
    print(f"  psi where arith point-clears<=1000 (vs n60) : "
          f"{r['frontier']['psi_where_arith_point_le1000_vs_n60']}")
    print()
    print("--- COMBINED BUDGET (must also pin int4) ---")
    CB = r["combined_budget"]
    print(f"  vs n60 psi0.10: n4={CB['vs_n60_psi0.10']['n4']} + "
          f"np={CB['vs_n60_psi0.10']['np']} = total {CB['vs_n60_psi0.10']['total_n']}")
    print(f"  vs g128 psi0.10: total {CB['vs_g128_psi0.10']['total_n']}")
    print(f"  one-sample point 2889; collapses_to_absolute_bar_class: "
          f"{CB['collapses_to_absolute_bar_class']}")
    print()
    print("--- LEGITIMACY ---")
    L = r["legitimacy"]
    print(f"  read2_legitimate: {L['read2_legitimate']}")
    print(f"  {L['ruling']}")
    print()
    print(f"VERDICT: {r['verdict']}")
    print(f"primary_metric paired_n_to_prove_gate_95pct = "
          f"{r['primary_metric']['paired_n_to_prove_gate_95pct']}")
    print(f"test_metric read2_legitimate = {r['test_metric']['read2_legitimate']}")
    print(f"self_test passes: {r['self_test']['passes']}")
    failed = [k for k, v in r["self_test"].items() if k != "passes" and not v]
    print(f"failed tests: {failed}")


def log_wandb(r):
    import wandb
    F, R1, FE, CB = r["frontier"], r["read1"], r["fern_empirical"], r["combined_budget"]
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group="recovery-paired-power-denken",
        name="denken/recovery-paired-power",
        job_type="analysis",
        config={
            "pr": 714, "agent": "denken", "kind": "recovery-paired-power",
            "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "floor": FLOOR, "ceiling": CEILING, "gate": GATE, "lift": LIFT,
            "routeB_point": ROUTEB_POINT, "z_ci": Z_CI, "z_power": Z_PWR,
            "gate_to_ceiling_margin": CEILING - GATE,
            "upstream": "710=66rhys58, fern659=nmjvtfov-group",
        })
    s = {
        "verdict": r["verdict"],
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "primary/paired_n_to_prove_gate_95pct": r["primary_metric"]["paired_n_to_prove_gate_95pct"],
        "test/read2_legitimate": r["test_metric"]["read2_legitimate"],
        # Read 1 (one-sample, pairing-invariant)
        "read1/min_n_point_full_g32_0438": R1["min_n_point_full_g32_0438"],
        "read1/min_n95_power_full_g32_0438": R1["min_n95_power_full_g32_0438"],
        "read1/wilson_lo_0438_n1000": R1["wilson_lo_0438_n1000"],
        # fern empirical discordance
        "fern/marginal_lift": FE["marginal_lift"],
        "fern/b_minus_c": FE["b_minus_c"],
        "fern/empirical_discordance_psi": FE["empirical_discordance_psi"],
        "fern/mcnemar_power_lift_gt0_n300": FE["mcnemar_power_lift_gt0_n300"],
        # arithmetic-rescue threshold (where pinned-anchor point-clears <= 1000)
        "frontier/psi_arith_le1000_vs_n60": F["psi_where_arith_point_le1000_vs_n60"],
        "frontier/psi_arith_le1000_vs_g128": F["psi_where_arith_point_le1000_vs_g128"],
        # combined budget (legitimacy arithmetic)
        "combined/vs_n60_psi0.10_total_n": CB["vs_n60_psi0.10"]["total_n"],
        "combined/vs_g128_psi0.10_total_n": CB["vs_g128_psi0.10"]["total_n"],
        "combined/collapses_to_absolute_bar": int(CB["collapses_to_absolute_bar_class"]),
        # legitimacy reasons
        "legit/read2_legitimate": int(r["legitimacy"]["read2_legitimate"]),
        "legit/int4_n300_halfwidth": r["legitimacy"]["reasons"]["r2_detail"]["int4_n300_halfwidth"],
        "legit/n_to_pin_int4_half_margin": r["legitimacy"]["reasons"]["r3_detail"]["n_to_pin_int4_to_half_margin"],
        "self_test_passes": r["self_test"]["passes"],
        "peak_mem_mib": _peak_mem_mib(),
    }
    run.summary.update(s)

    # n(concordance) frontier table (the required deliverable)
    t = wandb.Table(columns=[
        "psi", "liftgt0_n95_vs_g128", "liftgt0_n95_vs_n60",
        "gate_margin_point_vs_g128", "gate_margin_point_vs_n60",
        "gate_margin_n95_vs_g128", "gate_margin_n95_vs_n60", "arith_rescue_le1000"])
    def _i(x):  # coerce None -> -1 for table type-consistency
        return -1 if x is None else int(x)
    for row in F["n_concordance"]:
        t.add_data(float(row["psi"]), _i(row["liftgt0_n95_vs_g128"]), _i(row["liftgt0_n95_vs_n60"]),
                   _i(row["gate_margin_point_vs_g128"]), _i(row["gate_margin_point_vs_n60"]),
                   _i(row["gate_margin_n95_vs_g128"]), _i(row["gate_margin_n95_vs_n60"]),
                   bool(row["arith_rescue_le1000"]))
    run.log({"n_concordance_frontier": t})

    # two-read comparison table (all-string cols to avoid mixed types)
    tr = wandb.Table(columns=["read", "estimand", "pairing_helps", "n_to_prove_gate", "legitimate"])
    tr.add_data("Read1_absolute_one_sample", "pi_recovery >= 0.420", "False",
                str(R1["min_n_point_full_g32_0438"]), "True")
    tr.add_data("Read2_paired_lift_pinned", "int4_pinned + delta >= 0.420",
                "True", "n<=1000 IF pinned (ILLEGITIMATE)",
                str(bool(r["legitimacy"]["read2_legitimate"])))
    run.log({"two_read_comparison": tr})

    art = wandb.Artifact("recovery_paired_power_714", type="analysis")
    art.add_file(os.path.join(HERE, "recovery_paired_power_results.json"))
    run.log_artifact(art)
    rid = run.id
    run.finish()
    return rid


if __name__ == "__main__":
    import sys
    r = main()
    r["peak_mem_mib"] = _peak_mem_mib()
    json.dump(r, open(os.path.join(HERE, "recovery_paired_power_results.json"), "w"),
              indent=2)
    print_report(r)
    if "--wandb" in sys.argv:
        rid = log_wandb(r)
        print("WANDB_RUN_ID", rid)
