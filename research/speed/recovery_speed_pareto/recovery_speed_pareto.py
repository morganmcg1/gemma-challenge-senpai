#!/usr/bin/env python3
"""Recovery speed-quality Pareto (PR #706, denken).

ANALYSIS-ONLY pure-CPU model. NO HF Job, NO /v1/jobs:run, NO train.py --launch,
NO submission. Served file + locked rung untouched. official_tps=0, fires=0.

Question (decision-grade for the #481 fork): if a selective g128->g32 higher-
precision recovery of the int4-body AIME gap works, does the recovered config
still clear the speed gates? I.e. is quality-recovery speed-FREE (band above the
locked rung 126.378), speed-CONSTRAINED (clears AIME but at/below the rung), or
PARETO-EMPTY (no point clears both 0.420 and the +10 bar 136.378)?

Spine:
  * SPEED axis. Anchor = locked int4_g128_lmhead @ 126.378 official TPS (PR #4 /
    land a09npwda). The g128->g32 tax is EXTRA SCALE READS at the same int4
    Marlin kernel. Group 32 vs 128 = 4x more BF16 scales. Critically, scale
    elements = out*(in/group) = params/group, so the scale-byte delta is EXACTLY
    proportional to a module's parameter count -> "scale by upgraded-param
    fraction" is exact, not an approximation. We anchor the COEFFICIENT on
    land's measured full-body g32 tax (4.699 TPS / 3.681%, group
    g32-tax-empirical-land, run a09npwda) rather than a first-principles guess,
    and carry the op-bench->official conversion as a BAND (my #704 standard,
    run jl8o9x6e, deployed_envelope_halfwidth_pct=1.96%), never a point.
  * QUALITY axis (PROXY, unproven pending ubel #702). ubel #700 (run vjhzcvmu)
    impact-energy ranking: upgrading top-N modules recovers AIME ~in proportion
    to cumulative impact-energy captured. Floor = int4-body 0.3467, ceiling =
    base ~0.46. Two published normalizations (activation-weighted summary keys
    vs param-weighted pareto table) are both carried; the verdict is shown to be
    invariant to the choice.

Verdict in {RECOVERY_SPEED_FREE, RECOVERY_SPEED_CONSTRAINED, RECOVERY_PARETO_EMPTY}.
"""
import argparse
import json
import math
import os

# ----------------------------------------------------------------------------
# Fixed anchors (cross-reads authorized under #666: my #698/#701/#704, ubel
# #700, land #697, land g32-tax-empirical).
# ----------------------------------------------------------------------------
ANCHOR_OFFICIAL_TPS = 126.378     # locked int4_g128_lmhead (PR #4 / land a09npwda)
PLUS10_BAR = 136.378              # +10 bar
AIME_GATE = 0.420                 # quality fire-gate
AIME_FLOOR = 0.3467               # int4-body AIME (lawine #693, 5qjkdp27)
AIME_BASE_CEILING = 0.46          # base (un-quantized body) AIME ceiling

# land g32-tax-empirical (run a09npwda, group g32-tax-empirical-land):
# upgrading ALL body modules g128->g32 costs this much (additively projected;
# cell B g32_int4head was NOT directly measured -> treat as UPPER-BOUND coeff).
G32_FULL_BODY_TAX_TPS = 4.699     # = 3.681% of 126.378
G32_FULL_BODY_TAX_PCT = 0.03681

# my #704 (run jl8o9x6e): op-bench->official conversion envelope. The AR rung
# conversion is near 1:1 (land #697: op-bench 126.75 -> official 126.378,
# factor 0.99706); the dominant carry is the hardware/deploy envelope.
CONV_ENVELOPE_HALFWIDTH = 0.0196  # deployed_envelope_halfwidth_pct from #704
BI_TAX_BASIS_N = 1                # single basis point, carried forward from #704

# ubel #700 (run vjhzcvmu) cross-read scalars.
UBEL_CLEARING_N = 48
UBEL_CLEARING_F_PARAM = 0.013527           # ubel's published clearing footprint
UBEL_CRIT_TPS_AT_CLEARING = 126.275        # ubel's own projected TPS at clearing
UBEL_TOP1_SHARE = 0.34845
# composition of the 48-module clearing subset (NO MLP; all small modules):
UBEL_SUBSET_COMPOSITION = {"per_layer_input_gate": 40, "k_proj": 3, "q_proj": 3, "v_proj": 2}

# impact-energy cumulative curves at breakpoints, TWO normalizations:
#   activation-weighted (ubel summary keys; the PR cites "top-16 -> 0.672"):
IMPACT_ACT = {1: 0.34845, 8: 0.58670, 16: 0.67192, 32: 0.77091}
#   param-weighted (ubel impact_localization_pareto_curve table f_impact_energy):
IMPACT_PARAM = {1: 0.3485, 4: 0.4554, 8: 0.5296, 16: 0.5834, 32: 0.7010, 48: 0.7996}

# ----------------------------------------------------------------------------
# Gemma4 text_config geometry (int4_g128_lmhead/config.json, validated against
# the safetensors header shapes).
# ----------------------------------------------------------------------------
HID = 2560
INTER = 10240
N_HEADS = 8
N_KV = 2
HEAD_DIM = 256
PLI = 256                 # hidden_size_per_layer_input
N_LAYERS = 42
N_KV_SHARED = 18          # layers 24..41 have no k_proj / v_proj
GROUP_G128 = 128
GROUP_G32 = 32
SCALE_BYTES = 2           # BF16 scale
INT4_BYTES_PER_W = 0.5    # int4 packed weight

# weight matrix (out, in) per module type (in language_model body):
MOD_SHAPE = {
    "q_proj":  (N_HEADS * HEAD_DIM, HID),     # 2048 x 2560
    "k_proj":  (N_KV * HEAD_DIM, HID),        # 512  x 2560
    "v_proj":  (N_KV * HEAD_DIM, HID),        # 512  x 2560
    "o_proj":  (HID, N_HEADS * HEAD_DIM),     # 2560 x 2048
    "gate_proj": (INTER, HID),                # 10240 x 2560
    "up_proj":   (INTER, HID),                # 10240 x 2560
    "down_proj": (HID, INTER),                # 2560 x 10240
    "per_layer_input_gate": (PLI, HID),       # 256 x 2560
    "per_layer_projection": (HID, PLI),       # 2560 x 256
}
PLMP_SHAPE = (10752, HID)                     # top-level per_layer_model_projection


def params_of(shape):
    return shape[0] * shape[1]


def build_module_table():
    """Full list of the 343 quantized body modules with param counts."""
    mods = []
    full_attn_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
    gqa_attn_types = ["q_proj", "o_proj"]   # layers 24..41 share k/v
    mlp_types = ["gate_proj", "up_proj", "down_proj"]
    perlayer_types = ["per_layer_input_gate", "per_layer_projection"]
    for L in range(N_LAYERS):
        attn = full_attn_types if L < (N_LAYERS - N_KV_SHARED) else gqa_attn_types
        for t in attn + mlp_types + perlayer_types:
            mods.append((f"layers.{L}.{t}", t, params_of(MOD_SHAPE[t])))
    mods.append(("per_layer_model_projection", "per_layer_model_projection",
                 params_of(PLMP_SHAPE)))
    return mods


def scale_elems(shape, group):
    out, inn = shape
    assert inn % group == 0, (shape, group)
    return out * (inn // group)


def self_test(total_params):
    """Hard invariants the whole card rests on."""
    checks = []
    # (1) scale elements == params/group, EXACTLY, for every module type -> the
    #     g32 scale-byte tax is exactly proportional to param count.
    for t, shape in MOD_SHAPE.items():
        p = params_of(shape)
        s128 = scale_elems(shape, GROUP_G128)
        s32 = scale_elems(shape, GROUP_G32)
        checks.append(("scale_eq_params_div_g128_%s" % t, s128 == p // GROUP_G128))
        checks.append(("scale_eq_params_div_g32_%s" % t, s32 == p // GROUP_G32))
        checks.append(("g32_is_4x_scales_%s" % t, s32 == 4 * s128))
    # (2) per-module scale-byte delta / weight-bytes is a SHAPE-INDEPENDENT
    #     constant 3/32 (= (params*3/64*2... ) ) -> param-proportional tax.
    ratios = []
    for t, shape in MOD_SHAPE.items():
        p = params_of(shape)
        dscale = (scale_elems(shape, GROUP_G32) - scale_elems(shape, GROUP_G128)) * SCALE_BYTES
        wbytes = p * INT4_BYTES_PER_W
        ratios.append(dscale / wbytes)
    checks.append(("delta_scale_over_weight_const", max(ratios) - min(ratios) < 1e-12))
    checks.append(("delta_scale_over_weight_is_3_32", abs(ratios[0] - 3.0 / 32.0) < 1e-9))
    # (3) module count == 343.
    checks.append(("module_count_343", len(build_module_table()) == 343))
    # (4) clearing-subset composition sums to 48.
    checks.append(("subset_sums_48", sum(UBEL_SUBSET_COMPOSITION.values()) == 48))
    # (5) AIME proxy monotone non-decreasing in cumulative energy.
    checks.append(("aime_proxy_monotone",
                   aime_proj(0.0) <= aime_proj(0.5) <= aime_proj(1.0)))
    # (6) total params positive & ~3.9B (sanity).
    checks.append(("total_params_sane", 3.5e9 < total_params < 4.2e9))
    npass = sum(1 for _, ok in checks if ok)
    return npass, len(checks), checks


def aime_proj(cum_energy_frac):
    """Recovery proxy: AIME recovers in proportion to cumulative impact energy."""
    return AIME_FLOOR + (AIME_BASE_CEILING - AIME_FLOOR) * cum_energy_frac


def interp_breakpoints(table, x):
    """Piecewise-linear interpolation of a {n: value} breakpoint dict at x."""
    ks = sorted(table)
    if x <= ks[0]:
        return table[ks[0]] * (x / ks[0]) if ks[0] else table[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            f = (x - a) / (b - a)
            return table[a] + f * (table[b] - table[a])
    return table[ks[-1]]


def min_n_clearing(impact_curve, need_energy):
    """Smallest (interpolated) N whose cumulative energy >= need_energy."""
    ks = sorted(impact_curve)
    if impact_curve[ks[0]] >= need_energy:
        return ks[0]
    for a, b in zip(ks, ks[1:]):
        if impact_curve[a] < need_energy <= impact_curve[b]:
            f = (need_energy - impact_curve[a]) / (impact_curve[b] - impact_curve[a])
            return a + f * (b - a)
    return None  # not reached within logged breakpoints


def param_frac_proportional(n, plig_p, avg_attn_p, total_params):
    """Param fraction of top-N under proportional interleaving of the 8 attn
    modules among the 48 ranks. Exact at N=48 (== full subset composition)."""
    frac_attn = UBEL_SUBSET_COMPOSITION_TOTAL_ATTN / UBEL_CLEARING_N
    n_attn = n * frac_attn
    n_plig = n - n_attn
    return (n_plig * plig_p + n_attn * avg_attn_p) / total_params


def param_frac_all_plig(n, plig_p, total_params):
    """Optimistic edge: top-N are the smallest (all per_layer_input_gate)."""
    return min(n, 40) * plig_p / total_params


UBEL_SUBSET_COMPOSITION_TOTAL_ATTN = (
    UBEL_SUBSET_COMPOSITION["k_proj"]
    + UBEL_SUBSET_COMPOSITION["q_proj"]
    + UBEL_SUBSET_COMPOSITION["v_proj"]
)


def tps_band(n, param_frac_fn, total_params, plig_p, avg_attn_p):
    """Central TPS(N) and the honest band. Central uses land's full-body tax
    scaled by upgraded-param fraction; band carries (a) the op-bench->official
    hardware envelope (#704, +/-1.96%) and (b) tax-coefficient uncertainty
    (land's 4.699 is an additive projection -> allow up to 1.5x super-additive
    on the hi-tax edge, and the all-PLIG composition on the lo-tax edge)."""
    pf_central = param_frac_fn(n)
    pf_lo = param_frac_all_plig(n, plig_p, total_params)        # smallest modules
    tax_central = G32_FULL_BODY_TAX_TPS * pf_central
    tax_lo = G32_FULL_BODY_TAX_TPS * pf_lo                       # least tax
    tax_hi = 1.5 * G32_FULL_BODY_TAX_TPS * pf_central            # super-additive guard
    tps_central = ANCHOR_OFFICIAL_TPS - tax_central
    # band: least-tax * hi-envelope (best case) ... most-tax * lo-envelope (worst)
    band_hi = (ANCHOR_OFFICIAL_TPS - tax_lo) * (1 + CONV_ENVELOPE_HALFWIDTH)
    band_lo = (ANCHOR_OFFICIAL_TPS - tax_hi) * (1 - CONV_ENVELOPE_HALFWIDTH)
    return tps_central, band_lo, band_hi, tax_central


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_name", default="denken/recovery-speed-pareto")
    ap.add_argument("--wandb_group", default="recovery-speed-pareto-denken")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    mods = build_module_table()
    total_params = sum(p for _, _, p in mods)
    plig_p = params_of(MOD_SHAPE["per_layer_input_gate"])
    avg_attn_p = (
        UBEL_SUBSET_COMPOSITION["k_proj"] * params_of(MOD_SHAPE["k_proj"])
        + UBEL_SUBSET_COMPOSITION["q_proj"] * params_of(MOD_SHAPE["q_proj"])
        + UBEL_SUBSET_COMPOSITION["v_proj"] * params_of(MOD_SHAPE["v_proj"])
    ) / UBEL_SUBSET_COMPOSITION_TOTAL_ATTN

    # independent recompute of the 48-subset footprint (cross-check ubel 1.3527%)
    subset_params = (
        UBEL_SUBSET_COMPOSITION["per_layer_input_gate"] * plig_p
        + UBEL_SUBSET_COMPOSITION["k_proj"] * params_of(MOD_SHAPE["k_proj"])
        + UBEL_SUBSET_COMPOSITION["q_proj"] * params_of(MOD_SHAPE["q_proj"])
        + UBEL_SUBSET_COMPOSITION["v_proj"] * params_of(MOD_SHAPE["v_proj"])
    )
    f_param_48_recompute = subset_params / total_params

    # first-principles full-body tax fraction IF decode were 100% body-weight-
    # bound, vs land's empirical -> implied critical-path body-scale fraction.
    weight_bytes_total = total_params * INT4_BYTES_PER_W
    scale_bytes_g128_total = sum(
        scale_elems(MOD_SHAPE[t] if t in MOD_SHAPE else PLMP_SHAPE, GROUP_G128)
        for _, t, _ in mods
    ) * SCALE_BYTES
    dscale_full = sum(
        (scale_elems(MOD_SHAPE[t] if t in MOD_SHAPE else PLMP_SHAPE, GROUP_G32)
         - scale_elems(MOD_SHAPE[t] if t in MOD_SHAPE else PLMP_SHAPE, GROUP_G128))
        for _, t, _ in mods
    ) * SCALE_BYTES
    fp_tax_frac_body_bound = dscale_full / (weight_bytes_total + scale_bytes_g128_total)
    implied_body_scale_critpath = G32_FULL_BODY_TAX_PCT / fp_tax_frac_body_bound

    npass, ntot, checks = self_test(total_params)

    # ---- quality axis: min-N clearing the AIME gate, both normalizations ----
    need_energy = (AIME_GATE - AIME_FLOOR) / (AIME_BASE_CEILING - AIME_FLOOR)
    min_n_act = min_n_clearing(IMPACT_ACT, need_energy)
    min_n_param = min_n_clearing(IMPACT_PARAM, need_energy)
    # PRIMARY = activation-weighted (the basis the PR cites: "top-16 -> 0.672")
    min_n_primary = min_n_act

    pf_central_fn = lambda n: param_frac_proportional(n, plig_p, avg_attn_p, total_params)

    # ---- speed band at the gate (primary min-N) ----
    tps_gate, gate_lo, gate_hi, tax_gate = tps_band(
        min_n_primary, pf_central_fn, total_params, plig_p, avg_attn_p)

    # ---- Pareto sample points ----
    pareto = {}
    for n in [1, 4, 8, 16, 32, 48]:
        tpsc, blo, bhi, tax = tps_band(n, pf_central_fn, total_params, plig_p, avg_attn_p)
        a_act = aime_proj(interp_breakpoints(IMPACT_ACT, n))
        a_par = aime_proj(interp_breakpoints(IMPACT_PARAM, n))
        pareto[n] = dict(tps_central=tpsc, tps_lo=blo, tps_hi=bhi, tax=tax,
                         aime_act=a_act, aime_param=a_par,
                         param_frac=pf_central_fn(n))

    # ---- does ANY N clear both 0.420 AND the +10 bar 136.378? ----
    # TPS is monotone decreasing in N and maxes below the anchor (g32 only
    # subtracts), so the best achievable TPS over all N is at N->0+ == anchor.
    best_possible_tps = ANCHOR_OFFICIAL_TPS  # limit N->0 (no upgrade)
    any_clears_plus10 = (best_possible_tps >= PLUS10_BAR)  # structurally False

    # ---- noise discipline: tax vs conversion envelope ----
    conv_envelope_tps = ANCHOR_OFFICIAL_TPS * CONV_ENVELOPE_HALFWIDTH
    noise_to_tax_ratio = conv_envelope_tps / tax_gate if tax_gate > 0 else float("inf")

    # ---- VERDICT ----
    # Band-position test against the locked rung. g32 is STRICTLY slower than
    # g128 (positive scale-byte tax), so tps_central is structurally <= anchor.
    if tps_gate > ANCHOR_OFFICIAL_TPS + 1e-9:
        verdict = "RECOVERY_SPEED_FREE"
    elif min_n_primary is None:
        verdict = "RECOVERY_PARETO_EMPTY"
    else:
        # min-N clears AIME, but central TPS sits at/below the rung.
        verdict = "RECOVERY_SPEED_CONSTRAINED"
    # PARETO_EMPTY only if no N clears AIME at all within the lever. Here some N
    # clears AIME, so it is not empty; the binding outcome is CONSTRAINED with a
    # SUB-NOISE constraint (reported explicitly), and +10 is structurally
    # unreachable via selective-g32.

    out = dict(
        # guards
        analysis_only=1, official_tps=0, no_hf_job=1, fires=0,
        # primary / test
        recovery_tps_at_aime_gate=round(tps_gate, 4),
        min_n_clearing_aime_gate=int(math.ceil(min_n_primary)) if min_n_primary else None,
        min_n_clearing_aime_gate_interp=round(min_n_primary, 3) if min_n_primary else None,
        min_n_clearing_aime_gate_param_basis=round(min_n_param, 3) if min_n_param else None,
        recovery_tps_at_gate_band_lo=round(gate_lo, 4),
        recovery_tps_at_gate_band_hi=round(gate_hi, 4),
        tax_at_gate_tps=round(tax_gate, 5),
        # tax model
        g32_full_body_tax_tps=G32_FULL_BODY_TAX_TPS,
        g32_per_module_tax_avg_tps=round(G32_FULL_BODY_TAX_TPS / 343, 5),
        g32_per_plig_module_tax_tps=round(G32_FULL_BODY_TAX_TPS * plig_p / total_params, 6),
        fp_tax_frac_if_body_bound=round(fp_tax_frac_body_bound, 5),
        implied_body_scale_critpath_frac=round(implied_body_scale_critpath, 4),
        delta_scale_over_weight=3.0 / 32.0,
        # footprint cross-check
        f_param_48_recompute=round(f_param_48_recompute, 6),
        f_param_48_ubel=UBEL_CLEARING_F_PARAM,
        total_quant_params=total_params,
        subset48_params=subset_params,
        # gates / references
        anchor_official_tps=ANCHOR_OFFICIAL_TPS,
        plus10_bar=PLUS10_BAR,
        aime_gate=AIME_GATE, aime_floor=AIME_FLOOR, aime_base_ceiling=AIME_BASE_CEILING,
        need_cum_energy_to_clear=round(need_energy, 5),
        # +10 reachability
        best_possible_tps=best_possible_tps,
        any_n_clears_plus10=bool(any_clears_plus10),
        # noise discipline (#704)
        conv_envelope_halfwidth=CONV_ENVELOPE_HALFWIDTH,
        conv_envelope_tps=round(conv_envelope_tps, 4),
        noise_to_tax_ratio=round(noise_to_tax_ratio, 1),
        bi_tax_basis_n=BI_TAX_BASIS_N,
        # ubel cross-check
        ubel_crit_tps_at_clearing=UBEL_CRIT_TPS_AT_CLEARING,
        # self test
        self_test_passes=npass, self_test_total=ntot,
        verdict=verdict,
    )
    # Pareto points flattened for W&B scalars
    for n, d in pareto.items():
        out["recovery_tps_n%d" % n] = round(d["tps_central"], 4)
        out["recovery_tps_lo_n%d" % n] = round(d["tps_lo"], 4)
        out["recovery_tps_hi_n%d" % n] = round(d["tps_hi"], 4)
        out["aime_proj_n%d" % n] = round(d["aime_act"], 5)
        out["aime_proj_param_n%d" % n] = round(d["aime_param"], 5)
        out["param_frac_n%d" % n] = round(d["param_frac"], 6)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pareto_result.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print("\nSELF TEST: %d/%d" % (npass, ntot))
    for name, ok in checks:
        if not ok:
            print("  FAIL:", name)

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                         entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                         name=args.wandb_name, group=args.wandb_group,
                         config=dict(card="recovery_speed_pareto", pr=706, student="denken",
                                     analysis_only=True, official_tps=0, no_hf_job=1, fires=False,
                                     anchor_official_tps=ANCHOR_OFFICIAL_TPS,
                                     g32_full_body_tax_tps=G32_FULL_BODY_TAX_TPS,
                                     conv_envelope_halfwidth=CONV_ENVELOPE_HALFWIDTH,
                                     bi_tax_basis_n=BI_TAX_BASIS_N))
        wandb.summary.update(out)
        # pareto table
        tbl = wandb.Table(columns=["n", "tps_central", "tps_lo", "tps_hi",
                                   "aime_act", "aime_param", "param_frac"])
        for n, d in pareto.items():
            tbl.add_data(n, d["tps_central"], d["tps_lo"], d["tps_hi"],
                         d["aime_act"], d["aime_param"], d["param_frac"])
        wandb.log({"pareto_table": tbl})
        print("WANDB_RUN_ID:", run.id)
        run.finish()


if __name__ == "__main__":
    main()
