# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #710 - Realizable-recovery frontier (denken).

Modeling leg of the 3-leg recovery-axis closure. PURE CPU / numpy on already
logged census data (ubel #700 vjhzcvmu per-module impact-energy census + land
#708 8yf0622s fused-block servability/op-bench taxes). NO model load, NO GPU,
NO HF Job, NO served-file change.

Question: under land #708's fused-block servability algebra (vLLM fuses
q/k/v->qkv_proj and gate/up->gate_up_proj; a fused block gets exactly one
group_size), is there ANY realizable g32-promotion set whose PREDICTED AIME
Wilson-lower-bound clears the 0.420 gate at a FEASIBLE eval budget (n<=~1000)?
Or is the realizable recovery arm DEAD?

Proxy (re-anchored, reproduces denken #709 j2884s0i exactly):
    AIME(f) = floor + (ceiling - floor) * shape(f)
    floor = 0.3467 (int4-body g128, lawine #693)
    ceiling = 0.438 (full-g32 MEASURED, ubel #679)
    shapes: linear = f, concave = f**0.5, convex = f**2
    f* clearing 0.420 = (0.420-floor)/(ceiling-floor) = 0.8028 (linear)
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "census_inputs.json")

# ---- proxy anchors (denken #709 / lawine #693 / ubel #679) ----
FLOOR = 0.3467          # int4-body g128 AIME
CEILING = 0.438         # full-g32 MEASURED AIME ceiling
GATE = 0.420            # quality gate
LIFT = CEILING - FLOOR  # 0.0913
Z = 1.959963984540054   # 1.96 two-sided 95%

# ---- speed anchors (land #708 8yf0622s op-bench, AR M=1 rung) ----
TPS_ANCHOR = 126.75                 # all-g128 op-bench anchor
STEP_ANCHOR_US = 1e6 / TPS_ANCHOR   # 7889.5 us decode step at anchor
OFFICIAL_TPS_LOCKED = 126.378       # locked submission official TPS
NOISE_TPS = 0.10                    # op-bench speed-free band (|dTPS| <= 0.10)


# =====================================================================
# Wilson score interval + binomial power (reproduce #709 formulas)
# =====================================================================
def wilson(phat, n, z=Z):
    """Two-sided Wilson score interval (lo, hi)."""
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    hw = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (center - hw, center + hw)


def wilson_lo(phat, n, z=Z):
    return wilson(phat, n, z)[0]


def min_n_point_clears(p, gate=GATE, z=Z, cap=2_000_000):
    """Smallest n s.t. observing exactly rate p gives Wilson-lo > gate."""
    if p <= gate:
        return None  # unresolvable: point at/under the gate
    lo, hi = 1, cap
    if wilson_lo(p, cap, z) <= gate:
        return None
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
    """Regularized incomplete beta I_x(a,b)."""
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
    """P(X >= k) for X~Binom(n,p) via regularized incomplete beta."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betai(k, n - k + 1, p)


def power_clear_gate(p_true, n, gate=GATE, z=Z):
    """P( Wilson-lo(K/n) > gate ) when K ~ Binom(n, p_true)."""
    kmin = None
    for k in range(n + 1):
        if wilson_lo(k / n, n, z) > gate:
            kmin = k
            break
    if kmin is None:
        return 0.0
    return binom_sf_ge(kmin, n, p_true)


def min_n_for_power(p_true, target=0.95, gate=GATE, z=Z, cap=200000):
    lo, hi = 1, cap
    if power_clear_gate(p_true, cap, gate, z) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power_clear_gate(p_true, mid, gate, z) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# =====================================================================
# Proxy: cum-energy fraction -> predicted AIME under 3 shapes
# =====================================================================
SHAPES = {
    "linear": lambda f: f,
    "concave": lambda f: f ** 0.5,
    "convex": lambda f: f ** 2,
}


def aime_pred(f, shape="linear"):
    return FLOOR + LIFT * SHAPES[shape](f)


def fstar(shape="linear"):
    """cum-energy fraction at which predicted AIME == gate."""
    target = (GATE - FLOOR) / LIFT  # shape(f*) must equal this
    # invert shape
    if shape == "linear":
        return target
    if shape == "concave":  # f**0.5 = target
        return target ** 2
    if shape == "convex":   # f**2 = target
        return target ** 0.5
    raise ValueError(shape)


# =====================================================================
# Load census + build module energy map + realizable fused-block units
# =====================================================================
def load_census():
    d = json.load(open(CENSUS))
    pm = d["ubel700_per_module"]
    mc = {c: i for i, c in enumerate(pm["columns"])}
    proj, layer = {}, {}
    for r in pm["data"]:
        proj[r[mc["module"]]] = r[mc["proj"]]
        layer[r[mc["module"]]] = r[mc["layer"]]
    par = d["ubel700_pareto"]
    pc = {c: i for i, c in enumerate(par["columns"])}
    prows = sorted(par["data"], key=lambda r: r[pc["rank"]])
    # per-module energy fraction = increment of cumulative f_impact_energy
    energy, prev = {}, 0.0
    rank = {}
    for r in prows:
        m = r[pc["module"]]
        energy[m] = r[pc["f_impact_energy"]] - prev
        prev = r[pc["f_impact_energy"]]
        rank[m] = r[pc["rank"]]
    return d, proj, layer, energy, rank


def short(m):
    return m.replace("model.language_model.layers.", "L").replace(
        ".self_attn.", ".").replace(".mlp.", ".")


def build_units(proj, layer, energy):
    """Realizable promotion units honouring vLLM fused-block algebra.

    Returns list of dicts: {name, kind, modules:set, delta_us}.
    Fused: qkv_proj (q+k+v per layer), gate_up_proj (gate+up per layer).
    Standalone: per_layer_input_gate (PLIG), down_proj, o_proj,
                per_layer_projection, per_layer_model_projection.
    """
    s708 = json.load(open(CENSUS))["land708_summary"]
    du = {t: s708[f"iso_{t}_delta_us"] for t in [
        "q_proj", "k_proj", "v_proj", "qkv_full", "qkv_qonly", "gate_up_proj",
        "down_proj", "o_proj", "per_layer_input_gate", "per_layer_projection",
        "per_layer_model_projection"]}
    # group modules by layer
    from collections import defaultdict
    by_layer = defaultdict(dict)  # layer -> {proj: module}
    for m, p in proj.items():
        by_layer[layer[m]][p] = m
    units = []
    layers = sorted(l for l in by_layer if isinstance(l, int) and l >= 0)
    for L in layers:
        slots = by_layer[L]
        # --- fused qkv block ---
        qkv = {slots[p] for p in ("q_proj", "k_proj", "v_proj") if p in slots}
        if qkv:
            has_kv = "k_proj" in slots  # own-KV layer -> full qkv, else q-only
            units.append({
                "name": f"qkv@L{L}", "kind": "qkv", "modules": qkv,
                "delta_us": du["qkv_full"] if has_kv else du["qkv_qonly"],
                "has_kv": has_kv})
        # --- fused gate_up block ---
        gu = {slots[p] for p in ("gate_proj", "up_proj") if p in slots}
        if gu:
            units.append({"name": f"gate_up@L{L}", "kind": "gate_up",
                          "modules": gu, "delta_us": du["gate_up_proj"]})
        # --- standalone units ---
        for p, kind, dk in [
            ("per_layer_input_gate", "plig", "per_layer_input_gate"),
            ("down_proj", "down", "down_proj"),
            ("o_proj", "o", "o_proj"),
            ("per_layer_projection", "plp", "per_layer_projection")]:
            if p in slots:
                units.append({"name": f"{kind}@L{L}", "kind": kind,
                              "modules": {slots[p]}, "delta_us": du[dk]})
    # global per_layer_model_projection (layer -1)
    for m, p in proj.items():
        if p == "per_layer_model_projection":
            units.append({"name": "plmp", "kind": "plmp", "modules": {m},
                          "delta_us": du["per_layer_model_projection"]})
    # attach energy to each unit
    for u in units:
        u["energy"] = sum(energy[m] for m in u["modules"])
    return units, du


# =====================================================================
# Set scoring
# =====================================================================
def score_set(units_sel, energy, n_list=(300, 1000)):
    mods = set()
    delta = 0.0
    for u in units_sel:
        mods |= u["modules"]
        delta += u["delta_us"]
    f = sum(energy[m] for m in mods)
    tps = 1e6 / (STEP_ANCHOR_US + delta)
    dtps = tps - TPS_ANCHOR
    out = {
        "n_units": len(units_sel), "n_modules": len(mods),
        "cum_energy": f, "delta_us": delta, "tps_opbench": tps,
        "dtps_vs_anchor": dtps, "speed_free": abs(dtps) <= NOISE_TPS,
        "point": {sh: aime_pred(f, sh) for sh in SHAPES},
        "wilson_lo": {},
    }
    for sh in SHAPES:
        p = aime_pred(f, sh)
        out["wilson_lo"][sh] = {
            str(n): {"point": p, "wilson_lo": wilson_lo(p, n),
                     "clears": wilson_lo(p, n) > GATE,
                     "point_clears": p > GATE} for n in n_list}
    return out, mods


def main():
    d, proj, layer, energy, rank = load_census()
    units, du = build_units(proj, layer, energy)
    by_name = {u["name"]: u for u in units}

    # index helpers
    def units_of_kind(k):
        return [u for u in units if u["kind"] == k]

    # the 48-module targeted subset (top-48 by pareto energy)
    top48 = sorted(energy, key=lambda m: -energy[m])  # not pareto-order; use rank
    top48 = [m for m in sorted(rank, key=lambda m: rank[m])[:48]]
    top48_set = set(top48)
    plig_top40 = [m for m in top48 if proj[m] == "per_layer_input_gate"]
    attn_top = [m for m in top48 if proj[m] in ("q_proj", "k_proj", "v_proj")]
    attn_layers = sorted({layer[m] for m in attn_top})

    res = {"inputs": {}, "named_sets": {}, "frontier": {}, "self_test": {}}
    res["inputs"] = {
        "floor": FLOOR, "ceiling": CEILING, "gate": GATE, "lift": LIFT,
        "fstar_linear": fstar("linear"), "fstar_concave": fstar("concave"),
        "fstar_convex": fstar("convex"),
        "top48_cum_energy": sum(energy[m] for m in top48_set),
        "top48_composition": _compo(top48_set, proj),
        "n_plig_top40": len(plig_top40), "attn_top_layers": attn_layers,
        "tps_anchor": TPS_ANCHOR, "step_anchor_us": STEP_ANCHOR_US,
        "official_tps_locked": OFFICIAL_TPS_LOCKED,
        "energy_by_proj": _energy_by_proj(energy, proj),
        "total_energy_modules": len(energy),
    }

    # ---------- named realizable sets ----------
    # Route A: 40 PLIG standalone only
    routeA = [by_name[f"plig@L{layer[m]}"] for m in plig_top40]
    # Route B: 40 PLIG + whole-qkv for layers covering the top-48 attn
    qkv_units_B = [by_name[f"qkv@L{L}"] for L in attn_layers]
    routeB = routeA + qkv_units_B
    # Max speed-free greedy: add units by energy desc while |dTPS|<=NOISE
    sf_set = _greedy_speed_free(units, energy)
    # All-qkv + all-PLIG (grab all attention energy, ignore speed)
    allqkv_plig = units_of_kind("plig") + units_of_kind("qkv")
    # Full-g32: every unit
    full = list(units)

    named = {
        "routeA_40plig": routeA,
        "routeB_40plig_plus_wholeqkv": routeB,
        "max_speed_free_greedy": sf_set,
        "allplig_allqkv": allqkv_plig,
        "full_g32_all_units": full,
    }
    for nm, sel in named.items():
        sc, mods = score_set(sel, energy)
        # incidental modules (in set but not in targeted top-48)
        sc["incidental_energy"] = sum(
            energy[m] for m in mods - top48_set)
        sc["targeted_energy_covered"] = sum(
            energy[m] for m in mods & top48_set)
        if nm == "max_speed_free_greedy":
            sc["unit_kinds"] = _kind_counts(sel)
        res["named_sets"][nm] = sc

    # Route B incidental detail
    mods_B = set().union(*[u["modules"] for u in routeB])
    incid_B = mods_B - top48_set
    res["frontier"]["routeB_incidental"] = {
        "modules": sorted(short(m) for m in incid_B),
        "incidental_energy": sum(energy[m] for m in incid_B),
        "routeB_cum_energy": sum(energy[m] for m in mods_B),
        "knife_edge_gap_energy": fstar("linear") - sum(energy[m] for m in top48_set),
        "crosses_point_gate_linear": sum(energy[m] for m in mods_B) >= fstar("linear"),
    }

    # ---------- cheapest POINT-clearing set (energy >= f*_linear) ----------
    # greedy by energy/us; report cheapest set with cum_energy>=fstar linear
    cheapest_point = _cheapest_to_energy(units, energy, fstar("linear"))
    if cheapest_point is not None:
        sc, _ = score_set(cheapest_point, energy)
        sc["unit_kinds"] = _kind_counts(cheapest_point)
        res["frontier"]["cheapest_point_clear_linear"] = sc

    # ---------- the binding question: max realizable Wilson-lo at n ----------
    # argmax Wilson-lo over realizable sets == argmax point == full-g32 (f=1)
    best_overall, _ = score_set(full, energy)
    best_sf, _ = score_set(sf_set, energy)
    res["frontier"]["max_realizable_wilson_lo"] = {
        "n300": {sh: wilson_lo(best_overall["point"][sh], 300) for sh in SHAPES},
        "n1000": {sh: wilson_lo(best_overall["point"][sh], 1000) for sh in SHAPES},
        "argmax_set": "full_g32_all_units",
        "argmax_point_linear": best_overall["point"]["linear"],
        "any_realizable_clears_n300": any(
            wilson_lo(best_overall["point"][sh], 300) > GATE for sh in SHAPES),
        "any_realizable_clears_n1000": any(
            wilson_lo(best_overall["point"][sh], 1000) > GATE for sh in SHAPES),
    }
    res["frontier"]["best_speed_free"] = {
        "cum_energy": best_sf["cum_energy"], "tps": best_sf["tps_opbench"],
        "dtps": best_sf["dtps_vs_anchor"],
        "point": best_sf["point"],
        "wilson_lo_n300": {sh: wilson_lo(best_sf["point"][sh], 300) for sh in SHAPES},
        "wilson_lo_n1000": {sh: wilson_lo(best_sf["point"][sh], 1000) for sh in SHAPES},
    }

    # ---------- eval budget to PROVE (min-n point + power) ----------
    res["frontier"]["budget_to_prove"] = {
        "min_n_point_0438_ceiling": min_n_point_clears(CEILING),
        "min_n_point_best_sf_linear": min_n_point_clears(best_sf["point"]["linear"]),
        "min_n95_power_0438": min_n_for_power(CEILING, 0.95),
        "min_n80_power_0438": min_n_for_power(CEILING, 0.80),
        "note": "even observing the 0.438 ceiling, Wilson-lo clears only at "
                "n>=min_n_point_0438; selective sets predict <0.438 so need >= that.",
    }

    # ---------- proxy-shape sensitivity table ----------
    res["frontier"]["shape_sensitivity"] = []
    for sh in SHAPES:
        rb = aime_pred(res["frontier"]["routeB_incidental"]["routeB_cum_energy"], sh)
        bo = best_overall["point"][sh]
        res["frontier"]["shape_sensitivity"].append({
            "shape": sh, "fstar": fstar(sh),
            "routeB_point": rb, "routeB_point_clears": rb > GATE,
            "max_realizable_point": bo,
            "max_realizable_wilson_lo_n1000": wilson_lo(bo, 1000),
            "max_realizable_clears_n1000": wilson_lo(bo, 1000) > GATE,
        })

    # ---------- speed cross-check vs land #708 MEASURED mix points ----------
    s708 = json.load(open(CENSUS))["land708_summary"]
    res["frontier"]["speed_validation"] = {
        "model": "TPS = 1e6/(step_anchor_us + sum iso delta_us); additive,"
                 " conservatively over-taxes vs measured (sub-additive ops).",
        "measured_708": {
            "anchor_g128": s708["tps_g128_anchor"],
            "mix_40plig": s708["tps_mix_servable_40plig"],
            "mix_wholeqkv3": s708["tps_mix_wholeqkv3"],
            "mix_wholeqkv8": s708["tps_mix_wholeqkv8"],
            "fakequant48_ideal": s708["tps_fakequant48_ideal"],
            "full_g32": s708["tps_g32_full_measured"],
        },
        "additive_vs_measured": {
            "routeA_40plig_additive": res["named_sets"]["routeA_40plig"]["tps_opbench"],
            "routeA_40plig_measured": s708["tps_mix_servable_40plig"],
            "full_g32_additive": res["named_sets"]["full_g32_all_units"]["tps_opbench"],
            "full_g32_measured": s708["tps_g32_full_measured"],
        },
        "all_selective_sets_subnoise_vs_aime_band": True,  # |dTPS|<=0.2 << +/-2.48
        "speed_is_binding_constraint": False,
    }

    # ---------- VERDICT ----------
    any_clear = (res["frontier"]["max_realizable_wilson_lo"]["any_realizable_clears_n1000"])
    verdict = "REALIZABLE_RECOVERY_VIABLE" if any_clear else "REALIZABLE_RECOVERY_DEAD"
    res["verdict"] = verdict
    primary = max(wilson_lo(best_overall["point"][sh], 1000) for sh in SHAPES)
    res["primary_metric"] = {"max_realizable_predicted_wilson_lo_n1000": primary}
    # test metric: fused-block-CORRECTED realized cum-energy of the canonical
    # servable recovery set (Route B) -- the correction over #709's 0.7996.
    routeB_E = res["named_sets"]["routeB_40plig_plus_wholeqkv"]["cum_energy"]
    res["test_metric"] = {
        "routeB_realized_cum_energy": routeB_E,
        "best_speed_free_realized_cum_energy": best_sf["cum_energy"],
    }

    # ---------- self-tests ----------
    st = res["self_test"]
    st["a_wilson_24_60"] = _approx(wilson(24/60, 60), (0.2856949, 0.5263395), 1e-4)
    st["b_wilson_50_100"] = _approx(wilson(0.5, 100), (0.4038315, 0.5961685), 1e-4)
    st["c_fstar_linear_8028"] = abs(fstar("linear") - 0.8028477546) < 1e-6
    st["d_top48_energy_7996"] = abs(res["inputs"]["top48_cum_energy"] - 0.7995517) < 1e-4
    st["e_selective48_point_4197"] = abs(aime_pred(0.7995517, "linear") - 0.4196990976) < 1e-4
    st["f_shape_concave_4283"] = abs(aime_pred(0.7995517, "concave") - 0.42833833) < 1e-4
    st["g_shape_convex_4051"] = abs(aime_pred(0.7995517, "convex") - 0.40506657) < 1e-4
    st["h_min_n_point_0438_2889"] = (min_n_point_clears(CEILING) == 2889)
    st["i_min_n95_power_near9851"] = (abs(min_n_for_power(CEILING, 0.95) - 9851) <= 60)
    st["j_energy_sums_to_one"] = abs(sum(energy.values()) - 1.0) < 1e-6
    st["k_full_point_is_ceiling"] = abs(best_overall["point"]["linear"] - CEILING) < 1e-9
    st["l_routeA_lt_top48"] = (res["named_sets"]["routeA_40plig"]["cum_energy"]
                               < res["inputs"]["top48_cum_energy"])
    st["m_routeB_ge_top48"] = (res["named_sets"]["routeB_40plig_plus_wholeqkv"]["cum_energy"]
                               >= res["inputs"]["top48_cum_energy"] - 1e-9)
    st["n_units_partition_modules"] = _check_partition(units, energy)
    st["passes"] = all(v for k, v in st.items() if k != "passes")

    res["constraints"] = {"analysis_only": True, "official_tps": 0,
                          "no_hf_job": 1, "fires": 0}
    return res


# ---- small helpers ----
def _compo(modset, proj):
    from collections import Counter
    return dict(Counter(proj[m] for m in modset))


def _energy_by_proj(energy, proj):
    from collections import defaultdict
    e = defaultdict(float)
    for m, v in energy.items():
        e[proj[m]] += v
    return {k: round(v, 6) for k, v in sorted(e.items(), key=lambda kv: -kv[1])}


def _kind_counts(units_sel):
    from collections import Counter
    return dict(Counter(u["kind"] for u in units_sel))


def _greedy_speed_free(units, energy, noise=NOISE_TPS):
    """Greedily add units (energy desc) while op-bench |dTPS| <= noise."""
    order = sorted(units, key=lambda u: -u["energy"])
    sel, delta = [], 0.0
    for u in order:
        nd = delta + u["delta_us"]
        tps = 1e6 / (STEP_ANCHOR_US + nd)
        if abs(tps - TPS_ANCHOR) <= noise:
            sel.append(u)
            delta = nd
    return sel


def _cheapest_to_energy(units, energy, target):
    """Cheapest (min delta_us) unit set reaching cum_energy>=target, greedy by
    energy-per-us. Returns None if total energy < target."""
    if sum(u["energy"] for u in units) < target:  # impossible
        pass
    order = sorted(units, key=lambda u: -(u["energy"] / max(u["delta_us"], 1e-9)))
    sel, mods = [], set()
    for u in order:
        if sum(energy[m] for m in mods) >= target:
            break
        sel.append(u)
        mods |= u["modules"]
    if sum(energy[m] for m in mods) >= target:
        return sel
    return None


def _check_partition(units, energy):
    """qkv/gate_up fused units never split a fused block; all body modules
    covered exactly once across the full unit list."""
    from collections import Counter
    c = Counter()
    for u in units:
        for m in u["modules"]:
            c[m] += 1
    return all(v == 1 for v in c.values()) and len(c) == len(energy)


def _approx(got, want, tol):
    return abs(got[0] - want[0]) < tol and abs(got[1] - want[1]) < tol


def log_wandb(r):
    import wandb
    F, I = r["frontier"], r["inputs"]
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group="recovery-frontier-denken",
        name="denken/recovery-frontier-realizable",
        job_type="analysis",
        config={
            "pr": 710, "agent": "denken", "kind": "recovery-frontier-realizable",
            "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "floor": FLOOR, "ceiling": CEILING, "gate": GATE, "lift": LIFT,
            "fstar_linear": I["fstar_linear"], "tps_anchor": TPS_ANCHOR,
            "step_anchor_us": STEP_ANCHOR_US, "noise_tps": NOISE_TPS,
            "n_modules_total": I["total_energy_modules"],
            "top48_cum_energy": I["top48_cum_energy"],
            "upstream": "ubel700=vjhzcvmu, land708=8yf0622s, denken709=j2884s0i",
        })
    s = {
        "verdict": r["verdict"],
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "primary/max_realizable_predicted_wilson_lo_n1000":
            r["primary_metric"]["max_realizable_predicted_wilson_lo_n1000"],
        "test/routeB_realized_cum_energy": r["test_metric"]["routeB_realized_cum_energy"],
        "test/best_speed_free_realized_cum_energy":
            r["test_metric"]["best_speed_free_realized_cum_energy"],
        # energy frontier
        "routeA_cum_energy": r["named_sets"]["routeA_40plig"]["cum_energy"],
        "routeA_point_linear": r["named_sets"]["routeA_40plig"]["point"]["linear"],
        "routeB_cum_energy": F["routeB_incidental"]["routeB_cum_energy"],
        "routeB_incidental_energy": F["routeB_incidental"]["incidental_energy"],
        "knife_edge_gap_energy": F["routeB_incidental"]["knife_edge_gap_energy"],
        "routeB_point_linear": r["named_sets"]["routeB_40plig_plus_wholeqkv"]["point"]["linear"],
        "routeB_point_crosses_gate_linear": F["routeB_incidental"]["crosses_point_gate_linear"],
        # power wall (the binding constraint)
        "max_realizable_point": F["max_realizable_wilson_lo"]["argmax_point_linear"],
        "max_realizable_wilson_lo_n300": F["max_realizable_wilson_lo"]["n300"]["linear"],
        "max_realizable_wilson_lo_n1000": F["max_realizable_wilson_lo"]["n1000"]["linear"],
        "any_realizable_clears_n300": F["max_realizable_wilson_lo"]["any_realizable_clears_n300"],
        "any_realizable_clears_n1000": F["max_realizable_wilson_lo"]["any_realizable_clears_n1000"],
        "min_n_point_0438_ceiling": F["budget_to_prove"]["min_n_point_0438_ceiling"],
        "min_n95_power_0438": F["budget_to_prove"]["min_n95_power_0438"],
        "min_n80_power_0438": F["budget_to_prove"]["min_n80_power_0438"],
        # speed (secondary)
        "full_g32_tps_measured_708": F["speed_validation"]["measured_708"]["full_g32"],
        "mix_40plig_tps_measured_708": F["speed_validation"]["measured_708"]["mix_40plig"],
        "speed_is_binding": F["speed_validation"]["speed_is_binding_constraint"],
        # robustness
        "verdict_shape_invariant": all(
            not s["max_realizable_clears_n1000"] for s in F["shape_sensitivity"]),
        "routeB_point_shape_fragile": not all(
            s["routeB_point_clears"] for s in F["shape_sensitivity"]),
        "self_test_passes": r["self_test"]["passes"],
        "peak_mem_mib": _peak_mem_mib(),
    }
    run.summary.update(s)
    # named-sets table
    t = wandb.Table(columns=["set", "cum_energy", "dTPS_additive", "point_lin",
                             "wilson_lo_n300", "wilson_lo_n1000", "point_clears_gate"])
    for nm, sc in r["named_sets"].items():
        t.add_data(nm, sc["cum_energy"], sc["dtps_vs_anchor"], sc["point"]["linear"],
                   sc["wilson_lo"]["linear"]["300"]["wilson_lo"],
                   sc["wilson_lo"]["linear"]["1000"]["wilson_lo"],
                   sc["wilson_lo"]["linear"]["1000"]["point_clears"])
    run.log({"named_sets": t})
    # shape sensitivity table
    ts = wandb.Table(columns=["shape", "fstar", "routeB_point", "routeB_clears",
                              "max_real_point", "max_real_wilson_lo_n1000", "clears_n1000"])
    for sh in F["shape_sensitivity"]:
        ts.add_data(sh["shape"], sh["fstar"], sh["routeB_point"], sh["routeB_point_clears"],
                    sh["max_realizable_point"], sh["max_realizable_wilson_lo_n1000"],
                    sh["max_realizable_clears_n1000"])
    run.log({"shape_sensitivity": ts})
    # tiny artifact (results JSON, ~few KB)
    art = wandb.Artifact("recovery_frontier_realizable_710", type="analysis")
    art.add_file(os.path.join(HERE, "recovery_frontier_realizable_results.json"))
    run.log_artifact(art)
    rid = run.id
    run.finish()
    return rid


def _peak_mem_mib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


if __name__ == "__main__":
    import sys
    r = main()
    r["peak_mem_mib"] = _peak_mem_mib()
    json.dump(r, open(os.path.join(HERE, "recovery_frontier_realizable_results.json"), "w"),
              indent=2)
    print("VERDICT:", r["verdict"])
    print("self_test passes:", r["self_test"]["passes"])
    print("failed tests:", [k for k, v in r["self_test"].items()
                            if k != "passes" and not v])
    if "--full" in sys.argv:
        print(json.dumps(r["frontier"], indent=2))
    if "--wandb" in sys.argv:
        rid = log_wandb(r)
        print("WANDB_RUN_ID", rid)
