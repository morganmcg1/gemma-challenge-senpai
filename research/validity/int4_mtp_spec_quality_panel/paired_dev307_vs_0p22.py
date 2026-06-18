#!/usr/bin/env python3
"""PR #634 — paired single-variable (engine) A/B on GPQA-Diamond.

A-arm = vLLM 0.22.0 (manifest) GPQA leg banked by #629 (tag 0p22gb6144).
B-arm = vLLM dev307 GPQA leg measured here (tag dev307gb6144).
Everything else identical: same Option-B stack (int4_g128_lmhead + MTP-K7, BI=1,
max_model_len 8192, max_tokens 6144, conc 16, T=1.0/top_p0.95/top_k64/min_tokens8),
SAME 10 dataset seeds, SAME 198 GPQA-D questions per seed (prompt_sha asserted equal).

Reports (the #634 terminal contract):
  gpqa_dev307_10seed_acc, per-seed list, pooled Wilson CI, gpqa_dev307_ci_lo_clears_bar,
  paired_mean_delta_dev307_minus_0p22 + paired_delta_ci_significant (seed-level paired t),
  de_confounded_acc (+n), implied_base_for_0p22_clear, implied_base_for_dev307_clear, VERDICT.
Plus item-level McNemar (the statistically right paired test on the shared question set).

Usage: paired_dev307_vs_0p22.py <dev_tag> <a_tag> <seed> [seed ...]
  e.g. paired_dev307_vs_0p22.py dev307gb6144 0p22gb6144 12345 23456 ... 13579
No scipy dependency (numpy + stdlib only).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

R = Path("research/validity/int4_mtp_spec_quality_panel/results")
OUT = Path("research/validity/int4_mtp_spec_quality_panel")
BAR = 0.471
T_CRIT_DF9_975 = 2.262157  # two-sided 95% Student-t critical value, df=9


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a,b) (Numerical Recipes)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p-value for a Student-t statistic."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar (binomial) p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def load_seed(tag, seed):
    p = R / f"spec_gpqa_{tag}_s{seed}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def main():
    dev_tag, a_tag = sys.argv[1], sys.argv[2]
    seeds = sys.argv[3:]

    per_seed = []
    dev_c = dev_n = a_c = a_n = 0
    dev_err = a_err = 0
    # item-level pairing accumulators
    b_disc = c_disc = both_correct = both_wrong = 0
    sha_mismatch = 0
    paired_items = 0

    for s in seeds:
        dd = load_seed(dev_tag, s)
        da = load_seed(a_tag, s)
        accd, acca = dd["accuracy"], da["accuracy"]
        nd, na = dd["n_scored"], da["n_scored"]
        cd, ca = dd["n_correct"], da["n_correct"]
        ed, ea = dd.get("n_error", 0), da.get("n_error", 0)
        dev_c += cd; dev_n += nd; a_c += ca; a_n += na
        dev_err += ed; a_err += ea
        per_seed.append({
            "seed": s,
            "acc_dev307": accd, "n_correct_dev307": cd, "n_scored_dev307": nd,
            "n_error_dev307": ed,
            "acc_0p22": acca, "n_correct_0p22": ca, "n_scored_0p22": na,
            "delta_dev_minus_0p22": accd - acca,
            "len_stop_rate_dev307": dd.get("length_stop_rate"),
            "ctok_mean_dev307": dd.get("completion_tokens_mean"),
        })
        # item-level pairing on shared ids (prompt_sha must match by construction)
        amap = {str(r["id"]): r for r in da.get("per_sample", [])}
        for rd in dd.get("per_sample", []):
            ra = amap.get(str(rd["id"]))
            if ra is None:
                continue
            if rd.get("prompt_sha") and ra.get("prompt_sha") and rd["prompt_sha"] != ra["prompt_sha"]:
                sha_mismatch += 1
                continue
            paired_items += 1
            cdv, cav = bool(rd.get("correct")), bool(ra.get("correct"))
            if cdv and not cav:
                b_disc += 1
            elif cav and not cdv:
                c_disc += 1
            elif cdv and cav:
                both_correct += 1
            else:
                both_wrong += 1

    # ---- pooled dev307 ----
    dev_pooled = dev_c / dev_n if dev_n else float("nan")
    dev_se = math.sqrt(dev_pooled * (1 - dev_pooled) / dev_n) if dev_n else float("nan")
    dev_lo_w, dev_hi_w = wilson(dev_pooled, dev_n)
    ci_lo_clears_bar = bool(dev_lo_w >= BAR)

    a_pooled = a_c / a_n if a_n else float("nan")
    a_lo_w, a_hi_w = wilson(a_pooled, a_n)

    # ---- de-confounded (exclude request-error/overflow items) ----
    dev_ndeconf = dev_n - dev_err
    dev_acc_deconf = dev_c / dev_ndeconf if dev_ndeconf else float("nan")
    dev_lo_wd, dev_hi_wd = wilson(dev_acc_deconf, dev_ndeconf)

    # ---- seed-level paired t-test (the #634-required test) ----
    deltas = np.array([r["delta_dev_minus_0p22"] for r in per_seed], dtype=float)
    n_pairs = len(deltas)
    mean_delta = float(deltas.mean())
    sd_delta = float(deltas.std(ddof=1)) if n_pairs > 1 else float("nan")
    se_delta = sd_delta / math.sqrt(n_pairs) if n_pairs > 1 else float("nan")
    t_stat = mean_delta / se_delta if se_delta not in (0.0, float("nan")) and se_delta > 0 else float("nan")
    df = n_pairs - 1
    p_paired = t_sf_two_sided(t_stat, df) if not math.isnan(t_stat) else float("nan")
    ci_d_lo = mean_delta - T_CRIT_DF9_975 * se_delta
    ci_d_hi = mean_delta + T_CRIT_DF9_975 * se_delta
    paired_delta_ci_significant = bool((ci_d_lo > 0) or (ci_d_hi < 0))
    paired_delta_sig_positive = bool(ci_d_lo > 0)

    # ---- item-level McNemar (statistically right paired test on shared items) ----
    mcnemar_p = mcnemar_exact_two_sided(b_disc, c_disc)
    item_delta = (b_disc - c_disc) / paired_items if paired_items else float("nan")
    mcnemar_sig = bool(mcnemar_p < 0.05)

    # ---- bar-sensitivity: solve 0.9*base = acc ----
    implied_base_for_0p22_clear = a_pooled / 0.9
    implied_base_for_dev307_clear = dev_pooled / 0.9

    # ---- VERDICT ----
    clears = ci_lo_clears_bar                      # dev307 CI entirely >= bar
    clearly_below = bool(dev_hi_w < BAR)           # dev307 CI entirely < bar
    if clears and paired_delta_sig_positive:
        verdict = "GPQA_ENGINE_SPECIFIC"
    elif clearly_below:
        verdict = "GPQA_MODEL_LIMITED"
    else:
        # CI touches/straddles the bar (point either side) and no significant +engine delta
        verdict = "GPQA_KNIFE_EDGE_BOTH"

    out = {
        "tag_dev307": dev_tag, "tag_0p22": a_tag, "seeds": seeds, "bar": BAR,
        # ---- headline dev307 ----
        "gpqa_dev307_10seed_acc": dev_pooled,
        "dev307_n_correct": dev_c, "dev307_n_scored": dev_n, "dev307_stderr": dev_se,
        "dev307_ci95_wilson": [dev_lo_w, dev_hi_w],
        "gpqa_dev307_ci_lo_clears_bar": ci_lo_clears_bar,
        "dev307_point_clears_bar": bool(dev_pooled >= BAR),
        # ---- de-confounded ----
        "de_confounded_acc": dev_acc_deconf, "de_confounded_n": dev_ndeconf,
        "de_confounded_ci95_wilson": [dev_lo_wd, dev_hi_wd],
        "de_confounded_n_request_error": dev_err,
        "de_confounded_pass": bool(dev_acc_deconf >= BAR),
        # ---- A-arm banked (0.22.0) ----
        "gpqa_0p22_10seed_acc": a_pooled, "p0p22_n_correct": a_c, "p0p22_n_scored": a_n,
        "p0p22_ci95_wilson": [a_lo_w, a_hi_w],
        # ---- seed-level paired t-test ----
        "paired_mean_delta_dev307_minus_0p22": mean_delta,
        "paired_delta_sd": sd_delta, "paired_delta_se": se_delta,
        "paired_delta_t": t_stat, "paired_delta_df": df, "paired_delta_p_two_sided": p_paired,
        "paired_delta_ci95": [ci_d_lo, ci_d_hi],
        "paired_delta_ci_significant": paired_delta_ci_significant,
        "paired_delta_sig_positive": paired_delta_sig_positive,
        # ---- item-level McNemar (supporting) ----
        "mcnemar_paired_items": paired_items,
        "mcnemar_b_dev_correct_0p22_wrong": b_disc,
        "mcnemar_c_0p22_correct_dev_wrong": c_disc,
        "mcnemar_both_correct": both_correct, "mcnemar_both_wrong": both_wrong,
        "mcnemar_item_delta_dev_minus_0p22": item_delta,
        "mcnemar_p_two_sided": mcnemar_p, "mcnemar_significant": mcnemar_sig,
        "prompt_sha_mismatch": sha_mismatch,
        # ---- bar-sensitivity ----
        "implied_base_for_0p22_clear": implied_base_for_0p22_clear,
        "implied_base_for_dev307_clear": implied_base_for_dev307_clear,
        # ---- verdict ----
        "VERDICT": verdict,
        "per_seed": per_seed,
    }
    (OUT / "paired_dev307_vs_0p22.json").write_text(json.dumps(out, indent=2))

    # ---- human-readable summary ----
    print("=" * 78)
    print("PR #634 — GPQA-D single-variable (engine) A/B: dev307 vs 0.22.0 [#629]")
    print("=" * 78)
    print(f"  bar = {BAR}")
    print(f"  prompt_sha mismatches across arms: {sha_mismatch} (0 = clean same-prompt A/B)")
    print()
    print(f"  {'seed':>7} {'dev307':>8} {'0.22.0':>8} {'delta':>8}")
    for r in per_seed:
        print(f"  {r['seed']:>7} {r['acc_dev307']:>8.4f} {r['acc_0p22']:>8.4f} "
              f"{r['delta_dev_minus_0p22']:>+8.4f}")
    print()
    print(f"  dev307 pooled : {dev_pooled:.4f} ({dev_c}/{dev_n})  "
          f"wilson[{dev_lo_w:.4f}, {dev_hi_w:.4f}]  ci_lo_clears_bar={ci_lo_clears_bar}")
    print(f"  0.22.0 pooled : {a_pooled:.4f} ({a_c}/{a_n})  wilson[{a_lo_w:.4f}, {a_hi_w:.4f}]")
    print(f"  de-confounded : {dev_acc_deconf:.4f} ({dev_c}/{dev_ndeconf}) "
          f"wilson[{dev_lo_wd:.4f}, {dev_hi_wd:.4f}] (excl {dev_err} req-err) "
          f"-> {'PASS' if dev_acc_deconf >= BAR else 'FAIL'}")
    print()
    print(f"  paired Δ (dev307−0.22.0), seed-level t-test (n={n_pairs}, df={df}):")
    print(f"    mean Δ = {mean_delta:+.4f}  sd={sd_delta:.4f}  se={se_delta:.4f}")
    print(f"    t = {t_stat:+.3f}  p(two-sided) = {p_paired:.4f}")
    print(f"    95% CI [{ci_d_lo:+.4f}, {ci_d_hi:+.4f}]  significant={paired_delta_ci_significant}")
    print()
    print(f"  item-level McNemar (n_pairs={paired_items}):")
    print(f"    b(dev✓,0p22✗)={b_disc}  c(0p22✓,dev✗)={c_disc}  "
          f"both✓={both_correct}  both✗={both_wrong}")
    print(f"    item Δ = {item_delta:+.4f}  exact p(two-sided) = {mcnemar_p:.4f}  "
          f"significant={mcnemar_sig}")
    print()
    print(f"  bar-sensitivity (solve 0.9*base = acc):")
    print(f"    implied base for 0.22.0 to clear: {implied_base_for_0p22_clear:.4f}")
    print(f"    implied base for dev307 to clear: {implied_base_for_dev307_clear:.4f}")
    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
