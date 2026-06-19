#!/usr/bin/env python3
"""PR #702 aggregator: pool the 5-seed AIME runs per arm into the #31 gate-basis
verdict (matches lawine #693's pooled-Wilson statistic).

For each arm in {full_g128, selective, full_g32} we read 5 per-seed JSONs produced
by aime_eval.py (--k 1, sampled, 12288, min_tokens=8) and POOL them: n_correct /
(5*60) Bernoulli trials, then a Wilson 95% score interval on the pool. This
reproduces #693 exactly (int4-AR = 104/300 = 0.3467, Wilson [0.2951, 0.4022]).

Verdict (PR #702):
  * selective Wilson-hi >= 0.420  -> SELECTIVE_G32_REVIVES_CONFIRMED
  * selective > full_g128 but Wilson-hi < 0.420 -> SELECTIVE_G32_PARTIAL
  * selective ~= full_g128 (0.347) -> SELECTIVE_G32_PROXY_REFUTED
selective_recovery_fraction = (sel - g128) / (g32 - g128), anchored on the two
controls actually measured here (falls back to the PR's 0.347/0.438 if a control
is missing).

analysis_only=1, official_tps=0, no_hf_job=1, fires=0. LOCAL.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAR = 0.420                 # 90% of bf16 base 0.460 (#515 gate)
BF16_BASE = 0.460           # lawine #693 bf16 base on the #31 gate basis
G128_REF = 0.347            # #679/#693 full-g128 anchor
G32_REF = 0.438             # #679 uniform-g32 anchor
ARMS = ["full_g128", "selective", "full_g32"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (canonical #567 impl)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def propagate_recovery_ci(arms: dict, rf_thresh: float, n_draws: int = 400_000,
                          seed: int = 702) -> dict:
    """Monte-Carlo propagation of recovery_fraction = (sel-g128)/(g32-g128) and its
    int4-scale projection. The control separation (g32-g128 ~= 0.083) is only ~1.5x the
    per-arm Wilson half-width, so the ratio is NOISY; the point estimate alone overstates
    confidence. We draw each pooled proportion from its Jeffreys posterior
    Beta(k+0.5, n-k+0.5) INDEPENDENTLY (conservative: ignores the paired-seed positive
    correlation that would TIGHTEN the difference CI), form the ratio per draw, and report
    percentile CIs + a probabilistic clearing rate.

    The honest clearing event is `(g32>g128) AND sel >= g128 + rf_thresh*(g32-g128)`:
    a draw with no control separation (g32<=g128) provides no evidence of recovery and is
    counted as NOT clearing (so P(clear) is not inflated by sign-flipped denominators)."""
    need = ("full_g128", "selective", "full_g32")
    if not all(a in arms and arms[a] for a in need):
        return {}
    rng = np.random.default_rng(seed)

    def draw(a):
        k = arms[a]["pooled_n_correct"]
        n = arms[a]["pooled_n"]
        return rng.beta(k + 0.5, n - k + 0.5, size=n_draws)

    g128 = draw("full_g128")
    g32 = draw("full_g32")
    sel = draw("selective")

    sep = g32 - g128
    has_sep = sep > 0
    # ratio: guard the denominator; where there is no separation the rf is undefined and
    # we mask it out of the percentile summary (but still count it against P(clear)).
    safe = np.abs(sep) > 1e-6
    rf = np.where(safe, (sel - g128) / np.where(safe, sep, 1.0), np.nan)
    proj = G128_REF + rf * (G32_REF - G128_REF)
    clear = has_sep & (sel >= g128 + rf_thresh * sep)

    def pct(x, lo=2.5, hi=97.5):
        xv = x[np.isfinite(x)]
        if xv.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        return (float(np.percentile(xv, 50)),
                float(np.percentile(xv, lo)),
                float(np.percentile(xv, hi)))

    rf_med, rf_lo, rf_hi = pct(rf)
    pj_med, pj_lo, pj_hi = pct(proj)
    sep_med, sep_lo, sep_hi = pct(sep)
    return {
        "n_draws": n_draws,
        "method": "Jeffreys-Beta MC, arms independent (conservative re: paired-seed corr)",
        "rf_median": rf_med, "rf_ci_lo": rf_lo, "rf_ci_hi": rf_hi,
        "proj_median": pj_med, "proj_ci_lo": pj_lo, "proj_ci_hi": pj_hi,
        "separation_median": sep_med, "separation_ci_lo": sep_lo, "separation_ci_hi": sep_hi,
        "p_control_separation": float(np.mean(has_sep)),   # P(g32 ceiling > g128 floor)
        "p_projection_clears_gate": float(np.mean(clear)),  # P(int4-scale proj >= BAR)
        "rf_clear_threshold": rf_thresh,
    }


def load_arm(arm: str) -> dict | None:
    files = sorted(glob.glob(str(HERE / "results" / f"{arm}_seed*.json")))
    if not files:
        return None
    seeds, n_correct_pool, n_total_pool, extract_fail = [], 0, 0, 0
    empty = 0
    per_seed_acc = []
    for fp in files:
        d = json.load(open(fp))
        nc = int(d["n_correct_maj"])  # k=1 => maj@1 correct == the single sample correct
        n = int(d["n_problems"])
        per_seed_acc.append(nc / n if n else float("nan"))
        n_correct_pool += nc
        n_total_pool += n
        extract_fail += int(round(d.get("extract_fail_rate", 0.0) * d.get("total_samples", n)))
        for pr in d.get("per_problem", []):
            for t in (pr.get("texts") or []):
                if not str(t).strip():
                    empty += 1
        seeds.append({"file": Path(fp).name, "seed": d.get("sampling", {}).get("seed"),
                      "acc": nc / n if n else None, "n_correct": nc, "n": n,
                      "wall_s": d.get("wall_s")})
    p = n_correct_pool / n_total_pool if n_total_pool else float("nan")
    lo, hi = wilson(n_correct_pool, n_total_pool)
    return {
        "arm": arm,
        "n_seeds": len(files),
        "pooled_accuracy": p,
        "pooled_n_correct": n_correct_pool,
        "pooled_n": n_total_pool,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "per_seed_acc": per_seed_acc,
        "per_seed_acc_mean": sum(per_seed_acc) / len(per_seed_acc) if per_seed_acc else float("nan"),
        "per_seed_acc_min": min(per_seed_acc) if per_seed_acc else float("nan"),
        "per_seed_acc_max": max(per_seed_acc) if per_seed_acc else float("nan"),
        "pooled_empty": empty,
        "seeds": seeds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    arms = {a: load_arm(a) for a in ARMS}
    arms = {a: r for a, r in arms.items() if r}

    g128 = arms.get("full_g128", {}).get("pooled_accuracy", G128_REF)
    g32 = arms.get("full_g32", {}).get("pooled_accuracy", G32_REF)
    sel = arms.get("selective", {})
    sel_acc = sel.get("pooled_accuracy")
    sel_hi = sel.get("wilson95_hi")

    denom = (g32 - g128)
    recovery_fraction = ((sel_acc - g128) / denom) if (sel_acc is not None and abs(denom) > 1e-9) else float("nan")

    # SCALE CORRECTION. The bf16-dense fake-quant serve path sits a uniform ~0.05 BELOW
    # the int4-Marlin anchors (full_g128 measures ~0.30 here vs 0.347 served; full_g32
    # ~0.38 here vs 0.438 served), because the chaotic 12k-token sampled generation is
    # session-path-sensitive (already on record). So the ABSOLUTE 0.420 bar is depressed
    # on THIS instrument: even full_g32 (the ceiling arm) cannot reach it. The scale-FAIR
    # clearing test re-projects the measured recovery_fraction onto the int4-Marlin scale
    # the 0.420 gate actually lives on: proj = G128_REF + rf*(G32_REF - G128_REF). Clearing
    # the gate there requires rf >= (BAR - G128_REF)/(G32_REF - G128_REF) ~= 0.802.
    int4scale_projection = (G128_REF + recovery_fraction * (G32_REF - G128_REF)) \
        if not math.isnan(recovery_fraction) else float("nan")
    rf_clear_threshold = (BAR - G128_REF) / (G32_REF - G128_REF)  # ~0.802
    int4scale_clears = (not math.isnan(int4scale_projection)) and int4scale_projection >= BAR

    # propagated uncertainty (committed to in the 07:24 PR comment): the point projection
    # alone overstates confidence because the control separation is only ~1.5x the per-arm
    # Wilson half-width. MC gives the rf CI + a probabilistic clearing rate.
    mc = propagate_recovery_ci(arms, rf_clear_threshold) if sel_acc is not None else {}
    p_clear = mc.get("p_projection_clears_gate")

    pc_txt = f" P(int4-scale proj >= {BAR})={p_clear:.2f}" if p_clear is not None else ""

    # ABSOLUTE-WILSON-HI SATURATION. The PR card's original REVIVES trigger was
    # `selective Wilson-hi >= 0.420`. That criterion is INVALID on this bf16-fake-quant
    # instrument: the serve path sits ~0.05 below the int4-Marlin anchors, so even the
    # full_g32 CEILING arm measures only ~0.387 yet its Wilson-hi (~0.443) already crosses
    # 0.420. A test the ceiling arm passes cannot discriminate partial from full recovery,
    # so it would over-claim REVIVES on any arm that merely recovers PART-way toward g32.
    # Per the advisor's 06:35 steer ("read the verdict on the int4-scale projection"), the
    # REVIVES trigger is the SCALE-FAIR point projection (int4scale_clears, i.e. rf>=0.802 =
    # the PR's own rf_clear_threshold). The absolute Wilson-hi is reported informationally
    # only, flagged saturated. This correction was committed BEFORE the 5-seed selective
    # pool was known (only seed-0 on disk), so it is not result-fitted.
    g32_hi = arms.get("full_g32", {}).get("wilson95_hi")
    absolute_wilson_hi_saturated = bool(g32_hi is not None and g32_hi >= BAR)
    selective_wilson_hi_absolute_clears = bool(sel_hi is not None and sel_hi >= BAR)
    sat_txt = (f" [absolute Wilson-hi {sel_hi:.4f} crosses {BAR} but is SATURATED/non-discriminating: "
               f"the full_g32 ceiling arm's Wilson-hi {g32_hi:.4f} also crosses it]"
               if (selective_wilson_hi_absolute_clears and absolute_wilson_hi_saturated) else "")

    verdict = "PENDING"
    note = ""
    if sel_acc is not None:
        if int4scale_clears:
            verdict = "SELECTIVE_G32_REVIVES_CONFIRMED"
            note = (f"selective recovers on the scale-fair int4 projection: measured {sel_acc:.4f}, "
                    f"recovery_fraction {recovery_fraction:.3f} >= {rf_clear_threshold:.3f}, int4-scale "
                    f"projection {int4scale_projection:.4f} >= {BAR}.{pc_txt}{sat_txt} The 48-module "
                    f"activation-critical subset IS the AIME-critical set; speed-flat (~126.275) "
                    f"quality-compliant config (faithful-serve follow-up justified).")
        elif sel_acc > g128 + 0.01:
            verdict = "SELECTIVE_G32_PARTIAL"
            note = (f"selective {sel_acc:.4f} materially above full_g128 {g128:.4f}: recovery_fraction "
                    f"{recovery_fraction:.3f} (< {rf_clear_threshold:.3f}), int4-scale projection "
                    f"{int4scale_projection:.4f} < {BAR}.{pc_txt}{sat_txt} Subset carries real AIME "
                    f"signal but does not project a clear; report residual + min wider-subset footprint.")
        else:
            verdict = "SELECTIVE_G32_PROXY_REFUTED"
            note = (f"selective {sel_acc:.4f} ~= full_g128 {g128:.4f} (recovery_fraction "
                    f"{recovery_fraction:.3f}):{pc_txt} INPUT-activation proxy does not predict realized AIME "
                    f"recovery; recipe axis closed on weight (#695) AND activation (#700).")

    # control validity: do the anchors reproduce the references?
    controls = {
        "full_g128_reproduces_ref": (abs(g128 - G128_REF) <= 0.06) if "full_g128" in arms else None,
        "full_g32_reproduces_ref": (abs(g32 - G32_REF) <= 0.06) if "full_g32" in arms else None,
        "selective_between_controls": (
            (g128 - 0.02 <= sel_acc <= g32 + 0.02) if (sel_acc is not None and "full_g128" in arms and "full_g32" in arms) else None
        ),
    }

    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "bar": BAR, "bf16_base": BF16_BASE, "g128_ref": G128_REF, "g32_ref": G32_REF,
        "protocol": "AIME #31 gate basis: years=2024,2025-I,2025-II n=60, k=1 sampled "
                    "(T=1.0 top_p=0.95 top_k=64), max_tokens=12288, min_tokens=8, 5-seed pooled (300), "
                    "Wilson z=1.96 on pool (matches lawine #693).",
        "arms": arms,
        "selective_g32_aime_compliant": sel_acc,          # PRIMARY metric
        "selective_wilson_hi": sel_hi,
        "selective_recovery_fraction": recovery_fraction,  # TEST metric
        "selective_int4scale_projection": int4scale_projection,  # rf re-projected onto int4-Marlin scale
        "selective_int4scale_clears_gate": int4scale_clears,     # scale-fair clearing test (point) = VERDICT driver
        "rf_clear_threshold": rf_clear_threshold,                # rf needed to project a clear (~0.802)
        "recovery_fraction_mc": mc,                              # propagated CI + P(clear) (07:24 commitment)
        "selective_wilson_hi_absolute_clears": selective_wilson_hi_absolute_clears,  # informational only
        "absolute_wilson_hi_saturated": absolute_wilson_hi_saturated,  # full_g32 ceiling Wilson-hi also >= BAR
        "full_g32_wilson_hi": g32_hi,
        "scale_offset_note": ("bf16-dense fake-quant serve sits ~0.05 below int4-Marlin anchors; the absolute "
                              "0.420 bar is depressed on this instrument (full_g32 ceiling POINT ~0.387 < 0.420, "
                              "and its Wilson-hi ~0.443 crosses 0.420 -> the absolute Wilson-hi test is SATURATED "
                              "and non-discriminating). Verdict is driven by the scale-fair int4-scale projection "
                              "(rf >= 0.802), not the depressed absolute bar."),
        "controls": controls,
        "verdict": verdict,
        "verdict_note": note,
    }
    (HERE / "selective_g32_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("=" * 92)
    print(f"PR #702  AIME #31 gate basis  bar={BAR}  bf16-base={BF16_BASE}  (g128_ref {G128_REF} / g32_ref {G32_REF})")
    print("-" * 92)
    print(f"{'arm':12} {'seeds':>5} {'pooled':>7} {'n_corr':>8} {'wilson95':>18} {'perseed[min,mean,max]':>26}")
    for a in ARMS:
        if a not in arms:
            continue
        r = arms[a]
        ci = f"[{r['wilson95_lo']:.4f},{r['wilson95_hi']:.4f}]"
        ps = f"[{r['per_seed_acc_min']:.3f},{r['per_seed_acc_mean']:.3f},{r['per_seed_acc_max']:.3f}]"
        print(f"{a:12} {r['n_seeds']:>5} {r['pooled_accuracy']:>7.4f} "
              f"{r['pooled_n_correct']:>4}/{r['pooled_n']:<3} {ci:>18} {ps:>26}")
    print("-" * 92)
    print(f"selective_g32_aime_compliant (PRIMARY) = {sel_acc}")
    print(f"selective_wilson_hi = {sel_hi}  (>= {BAR} absolute? {sel_hi is not None and sel_hi >= BAR})")
    print(f"selective_recovery_fraction (TEST) = {recovery_fraction:.4f}" if isinstance(recovery_fraction, float) and not math.isnan(recovery_fraction) else f"selective_recovery_fraction = {recovery_fraction}")
    print(f"selective_int4scale_projection = {int4scale_projection:.4f}  (>= {BAR}? {int4scale_clears})  [rf clear thresh ~{rf_clear_threshold:.3f}]")
    if mc:
        print(f"  [MC propagated, {mc['n_draws']} draws] recovery_fraction = {mc['rf_median']:.3f} "
              f"[{mc['rf_ci_lo']:.3f}, {mc['rf_ci_hi']:.3f}]")
        print(f"  [MC] int4-scale projection = {mc['proj_median']:.4f} "
              f"[{mc['proj_ci_lo']:.4f}, {mc['proj_ci_hi']:.4f}]  "
              f"P(proj >= {BAR}) = {mc['p_projection_clears_gate']:.3f}  "
              f"P(g32>g128 separation) = {mc['p_control_separation']:.3f}")
        print(f"  [MC] control separation g32-g128 = {mc['separation_median']:.4f} "
              f"[{mc['separation_ci_lo']:.4f}, {mc['separation_ci_hi']:.4f}]")
    print(f"  scale note: bf16-serve ~0.05 below int4-Marlin anchors -> read the int4-scale projection, not the depressed absolute bar")
    print(f"controls: {controls}")
    print(f"VERDICT: {verdict}")
    print(f"  {note}")
    print(f"[wrote] {HERE / 'selective_g32_summary.json'}")


if __name__ == "__main__":
    main()
