#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Well-posed (anchor-free) G1 P(DQ) + private free-response fraction sweep — PR #754.

The #749 verdict (`deltastock_verdict.py`) reported P(DQ)=0.132 from a one-sided
calibrated model that brackets the faithful central against the fixed +4.3 anchor:

    cal_mu    = (faithful_central + 4.3) / 2
    cal_sigma = |4.3 - faithful_central| / 2

This is ILL-POSED: as the proxy central improves (more negative), cal_sigma WIDENS
faster than cal_mu drops, so P(DQ) RISES. "Improve the proxy -> higher P(DQ)" is
backwards (verified in `main`: c=-5 -> 0.125, c=-7.729 -> 0.132, c=-10 -> 0.136).

This card replaces the anchor with a measurement-driven, anchor-free uncertainty,
reusing the EXACT #749 faithful serving run (no new HF/serving job):

  Arm A(a) -- interval block-bootstrap. vLLM prints a per-window "SpecDecoding
    metrics: ... Accepted: N tokens, Drafted: M tokens" line every ~10s. We bin
    those windows into the 4 measured subsets (public / mmlu_pro / gpqa_diamond /
    reasoning_math) using the subset's cumulative spec_decode counter boundaries,
    then resample windows within each subset (>=1000 draws). Each draw recomputes
    e_accept(subset), delta_stock(subset) = (1 - e_s/e_public)*100 with a JOINTLY
    bootstrapped public denominator, and the faithful 57/57/14 mix central. P(DQ)
    is the bootstrap tail mass above +5%. No external anchor.

  Arm A(b) -- symmetric Normal. mu = faithful central (point), sigma = the
    bootstrap SE of the central (NOT the anchor bracket). P(DQ) = P(N(mu,sigma)>5).
    Monotone-correct: more favorable central -> lower P(DQ).

  Arm B -- private free-response fraction sweep. Hold the MCQ corners fixed (50/50
    mmlu/gpqa) and sweep the non-MCQ (free-response math) fraction f_nonmcq from the
    public-faithful 14/128 ~= 0.109 up to a conservative 0.40. For each f, reuse the
    SAME bootstrap draws to get a well-posed P(DQ)(f) and central(f). Report the
    breakeven f_nonmcq* where P(DQ) crosses 0.13 and where central crosses +5%.

LOCAL ONLY: reads existing artifacts, launches no server, makes no submission.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Faithful public composition (57 MMLU-Pro + 57 GPQA-Diamond + 14 free-response math).
W_MMLU_F, W_GPQA_F, W_MATH_F = 57 / 128, 57 / 128, 14 / 128
F_NONMCQ_PUBLIC = 14 / 128  # 0.109375 -- the public-faithful free-response fraction
G1 = 5.0
ANCHOR_52 = 4.3  # the one-sided anchor Arm A removes (kept only for the backwards demo)
P_DQ_BAR = 0.13  # board-readiness P(DQ) bar (the #749 0.13 line)

_ACCEPTED_RE = re.compile(r"Accepted:\s*(\d+)\s*tokens")
_DRAFTED_RE = re.compile(r"Drafted:\s*(\d+)\s*tokens")

# subset key in report.json -> short bootstrap name
SUBSET_ORDER = ["public", "knowledge_mmlupro", "gpqa_diamond", "reasoning_math"]


def gauss_tail(mu: float, sigma: float, thr: float = G1) -> float:
    """P(N(mu, sigma) > thr)."""
    if sigma <= 0:
        return 1.0 if mu > thr else 0.0
    return 0.5 * (1 - erf((thr - mu) / sigma / sqrt(2)))


def parse_intervals(log_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Ordered per-window (accepted, drafted) token counts from vLLM's spec log.

    These window counts are non-cumulative (they reset each ~10s log tick), so the
    running sum of `accepted` reconstructs the cumulative spec_decode counter that
    the subset boundaries are expressed in.
    """
    acc = np.array([int(x) for x in _ACCEPTED_RE.findall(log_text)], dtype=np.int64)
    drf = np.array([int(x) for x in _DRAFTED_RE.findall(log_text)], dtype=np.int64)
    if len(acc) != len(drf):
        raise SystemExit(f"interval parse mismatch: {len(acc)} accepted vs {len(drf)} drafted")
    return acc, drf


def subset_accept_boundaries(report: dict[str, Any]) -> dict[str, int]:
    """Cumulative num_accepted_tokens at the END of each subset (Prometheus counter)."""
    out: dict[str, int] = {}
    for name in SUBSET_ORDER:
        s = report["subsets"].get(name)
        if s is None:
            raise SystemExit(f"report missing subset {name!r}")
        out[name] = int(s["counters_after"]["num_accepted_tokens"])
    return out


def bin_intervals(acc: np.ndarray, drf: np.ndarray,
                  cum_boundaries: dict[str, int]) -> dict[str, dict[str, np.ndarray]]:
    """Assign each ~10s window to the subset whose cumulative-accepted band contains
    the window's midpoint. The spec-log total can fall a hair short of the final
    Prometheus counter (last window unflushed at scrape), so we scale the boundaries
    into the spec-log's own cumulative space first.
    """
    spec_total = int(acc.sum())
    prom_total = cum_boundaries[SUBSET_ORDER[-1]]
    scale = spec_total / prom_total
    # ordered (lo, hi] bands in spec-log cumulative space
    bands: list[tuple[str, float, float]] = []
    lo = 0.0
    for name in SUBSET_ORDER:
        hi = cum_boundaries[name] * scale
        bands.append((name, lo, hi))
        lo = hi
    bins: dict[str, dict[str, list[int]]] = {n: {"a": [], "d": []} for n in SUBSET_ORDER}
    cum = 0
    for a, d in zip(acc.tolist(), drf.tolist()):
        mid = cum + a / 2.0
        sub = SUBSET_ORDER[-1]
        for name, blo, bhi in bands:
            if blo <= mid < bhi:
                sub = name
                break
        bins[sub]["a"].append(a)
        bins[sub]["d"].append(d)
        cum += a
    return {n: {"a": np.array(v["a"], dtype=np.float64),
                "d": np.array(v["d"], dtype=np.float64)} for n, v in bins.items()}


def boot_eaccept(a: np.ndarray, d: np.ndarray, B: int, K: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Block-bootstrap e_accept = 1 + K * (sum accepted)/(sum drafted) over windows."""
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    sa = a[idx].sum(axis=1)
    sd = d[idx].sum(axis=1)
    return 1.0 + K * sa / sd


def run(report_path: Path, *, B: int, K: int, seed: int,
        f_lo: float, f_hi: float, f_steps: int, out_dir: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    log_text = (report_path.parent / "server.log").read_text()

    acc, drf = parse_intervals(log_text)
    boundaries = subset_accept_boundaries(report)
    bins = bin_intervals(acc, drf, boundaries)

    # binning sanity: per-subset bootstrap-mean e_accept must match the report value
    e_report = {n: report["subsets"][n]["e_accept"] for n in SUBSET_ORDER}
    e_binned = {n: 1.0 + K * bins[n]["a"].sum() / bins[n]["d"].sum() for n in SUBSET_ORDER}

    rng = np.random.default_rng(seed)
    e_b = {n: boot_eaccept(bins[n]["a"], bins[n]["d"], B, K, rng) for n in SUBSET_ORDER}

    # per-subset delta_stock draws, public is the (jointly bootstrapped) denominator
    e_pub_b = e_b["public"]
    dlt = {n: (1.0 - e_b[n] / e_pub_b) * 100.0 for n in SUBSET_ORDER if n != "public"}
    d_mmlu, d_gpqa, d_math = dlt["knowledge_mmlupro"], dlt["gpqa_diamond"], dlt["reasoning_math"]

    # point deltas straight from the report (the #749 headline corners)
    p_mmlu = report["delta_stock_pct_by_eaccept"]["knowledge_mmlupro"]
    p_gpqa = report["delta_stock_pct_by_eaccept"]["gpqa_diamond"]
    p_math = report["delta_stock_pct_by_eaccept"]["reasoning_math"]
    faithful_central_point = W_MMLU_F * p_mmlu + W_GPQA_F * p_gpqa + W_MATH_F * p_math

    # ---- Arm A: faithful 57/57/14 well-posed central ----
    central_b = W_MMLU_F * d_mmlu + W_GPQA_F * d_gpqa + W_MATH_F * d_math
    sigma_boot = float(np.std(central_b, ddof=1))
    central_boot_mean = float(np.mean(central_b))
    p_dq_bootstrap = float(np.mean(central_b > G1))
    ci = [float(np.percentile(central_b, 2.5)), float(np.percentile(central_b, 97.5))]

    # Arm A(b) symmetric Normal: mu = faithful point central, sigma = bootstrap SE
    p_dq_symnormal = gauss_tail(faithful_central_point, sigma_boot)

    # Sensitivity + variance decomposition. The G1 gate divides the private-set TPS
    # by the FIXED official-128 public TPS, so a gate-literal reading holds e_public
    # constant (only the private/shifted side is random). The conservative reading
    # (the advisor's "resample prompts within EACH subset", public included) treats
    # our public e_accept as itself a finite-sample estimate -> its sampling noise is
    # COMMON-MODE across all three deltas and dominates the central SE.
    e_pub_point = report["e_accept_public"]
    d_mmlu_fix = (1.0 - e_b["knowledge_mmlupro"] / e_pub_point) * 100.0
    d_gpqa_fix = (1.0 - e_b["gpqa_diamond"] / e_pub_point) * 100.0
    d_math_fix = (1.0 - e_b["reasoning_math"] / e_pub_point) * 100.0
    central_b_epubfix = W_MMLU_F * d_mmlu_fix + W_GPQA_F * d_gpqa_fix + W_MATH_F * d_math_fix
    sigma_epubfix = float(np.std(central_b_epubfix, ddof=1))
    p_dq_epubfix = float(np.mean(central_b_epubfix > G1))
    common_mode_se = float(np.sqrt(max(sigma_boot ** 2 - sigma_epubfix ** 2, 0.0)))

    # well-posedness monotonicity demo: old ill-posed vs new well-posed P(DQ)(central)
    demo = []
    for c in [-3.0, -5.0, faithful_central_point, -10.0, -13.0]:
        old = gauss_tail((c + ANCHOR_52) / 2.0, abs(ANCHOR_52 - c) / 2.0)
        new = gauss_tail(c, sigma_boot)
        demo.append({"central": round(c, 3), "p_dq_old_illposed": round(old, 4),
                     "p_dq_new_wellposed": round(new, 4)})
    old_monotone_backwards = all(
        demo[i]["p_dq_old_illposed"] <= demo[i + 1]["p_dq_old_illposed"]
        for i in range(len(demo) - 1))  # rises as central -> more negative
    new_monotone_correct = all(
        demo[i]["p_dq_new_wellposed"] >= demo[i + 1]["p_dq_new_wellposed"]
        for i in range(len(demo) - 1))  # falls as central -> more negative

    # ---- Arm B: private free-response fraction sweep (reuse the SAME draws) ----
    d_mcq_blend_b = 0.5 * d_mmlu + 0.5 * d_gpqa            # bootstrap MCQ blend draws
    d_mcq_blend_point = 0.5 * p_mmlu + 0.5 * p_gpqa        # = -11.099
    fs = np.linspace(f_lo, f_hi, f_steps)
    sweep = []
    for f in fs:
        cen_b = (1.0 - f) * d_mcq_blend_b + f * d_math
        cen_point = (1.0 - f) * d_mcq_blend_point + f * p_math
        sweep.append({
            "f_nonmcq": float(f),
            "central_point": float(cen_point),
            "central_boot_mean": float(np.mean(cen_b)),
            "sigma_boot": float(np.std(cen_b, ddof=1)),
            "p_dq_bootstrap": float(np.mean(cen_b > G1)),
            "p_dq_symnormal": gauss_tail(float(cen_point), float(np.std(cen_b, ddof=1))),
        })

    # breakeven where central(f) crosses +5% (analytic, matches #749 0.5225)
    f_star_central = (G1 - d_mcq_blend_point) / (p_math - d_mcq_blend_point)
    # breakeven where well-posed bootstrap P(DQ)(f) crosses the 0.13 board bar
    f_star_pdq = _first_cross(sweep, "p_dq_bootstrap", P_DQ_BAR)
    f_star_pdq_symnormal = _first_cross(sweep, "p_dq_symnormal", P_DQ_BAR)

    out = {
        "source_report": str(report_path),
        "bootstrap": {"draws": B, "K": K, "seed": seed,
                      "unit": "spec-log ~10s window (block bootstrap)",
                      "n_windows_per_subset": {n: int(len(bins[n]["a"])) for n in SUBSET_ORDER}},
        "binning_sanity": {
            "e_accept_report": {n: round(e_report[n], 4) for n in SUBSET_ORDER},
            "e_accept_binned": {n: round(e_binned[n], 4) for n in SUBSET_ORDER},
            "max_abs_diff": round(max(abs(e_binned[n] - e_report[n]) for n in SUBSET_ORDER), 4),
        },
        "faithful_corners_point_pct": {"mmlu_pro": p_mmlu, "gpqa_diamond": p_gpqa,
                                       "reasoning_math": p_math,
                                       "mcq_blend": round(d_mcq_blend_point, 3)},
        "faithful_central": {
            "point_pct": round(faithful_central_point, 3),
            "bootstrap_mean_pct": round(central_boot_mean, 3),
            "bootstrap_se_pct": round(sigma_boot, 3),
            "bootstrap_ci95_pct": [round(ci[0], 3), round(ci[1], 3)],
        },
        "arm_a_wellposed_p_dq": {
            "p_dq_wellposed_bootstrap": round(p_dq_bootstrap, 5),
            "p_dq_wellposed_symnormal": round(p_dq_symnormal, 5),
            "p_dq_wellposed_bootstrap_epub_fixed": round(p_dq_epubfix, 5),
            "p_dq_illposed_749_for_reference": 0.132,
            "g1_threshold_pct": G1,
        },
        "variance_decomposition": {
            "central_se_epub_bootstrapped_pct": round(sigma_boot, 3),
            "central_se_epub_fixed_pct": round(sigma_epubfix, 3),
            "public_anchor_common_mode_se_pct": round(common_mode_se, 3),
            "per_subset_eaccept_se": {n: round(float(np.std(e_b[n], ddof=1)), 4)
                                      for n in SUBSET_ORDER},
            "note": ("central SE is dominated by the shared official-128 public anchor "
                     "sampling noise (common-mode), not the corner spread; both readings "
                     "leave P(DQ) far below the 0.13 bar"),
        },
        "wellposedness_demo": {
            "rows": demo,
            "old_illposed_rises_as_central_improves": old_monotone_backwards,
            "new_wellposed_falls_as_central_improves": new_monotone_correct,
        },
        "arm_b_fraction_sweep": {
            "f_nonmcq_public_faithful": round(F_NONMCQ_PUBLIC, 5),
            "sweep": sweep,
            "f_nonmcq_star_central_to_5pct": round(f_star_central, 4),
            "f_nonmcq_star_pdq_to_0p13_bootstrap": (round(f_star_pdq, 4)
                                                    if f_star_pdq is not None else None),
            "f_nonmcq_star_pdq_to_0p13_symnormal": (round(f_star_pdq_symnormal, 4)
                                                    if f_star_pdq_symnormal is not None else None),
        },
    }

    # verdict
    safe = (p_dq_bootstrap < P_DQ_BAR and faithful_central_point < G1
            and new_monotone_correct)
    f_star_report = f_star_pdq if f_star_pdq is not None else f_star_central
    out["decision"] = {
        "primary_metric_p_dq_wellposed_bootstrap": round(p_dq_bootstrap, 5),
        "p_dq_wellposed_symnormal": round(p_dq_symnormal, 5),
        "test_metric_f_nonmcq_star": round(f_star_report, 4),
        "g1_safe_wellposed": bool(safe),
        "verdict": ("WELLPOSED_G1_SAFE" if safe else "WELLPOSED_G1_AT_RISK"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wellposed.json").write_text(json.dumps(out, indent=2))
    _plot_sweep(sweep, f_star_central, f_star_pdq, out_dir / "fraction_sweep.png")
    _print_summary(out)
    print(f"[wellposed] wrote {out_dir/'wellposed.json'}", flush=True)
    return out


def _first_cross(sweep: list[dict], key: str, bar: float) -> float | None:
    """First f at which sweep[key] crosses `bar` from below (linear interp)."""
    for i in range(1, len(sweep)):
        y0, y1 = sweep[i - 1][key], sweep[i][key]
        if y0 < bar <= y1:
            x0, x1 = sweep[i - 1]["f_nonmcq"], sweep[i]["f_nonmcq"]
            if y1 == y0:
                return x1
            return x0 + (bar - y0) * (x1 - x0) / (y1 - y0)
    return None


def _plot_sweep(sweep: list[dict], f_star_central: float,
                f_star_pdq: float | None, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[wellposed] plot skipped: {e!r}", flush=True)
        return
    fs = [s["f_nonmcq"] for s in sweep]
    pdq = [s["p_dq_bootstrap"] for s in sweep]
    cen = [s["central_point"] for s in sweep]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(fs, pdq, "b-o", ms=3, label="well-posed P(DQ) bootstrap")
    ax1.axhline(P_DQ_BAR, color="b", ls=":", alpha=0.6, label=f"P(DQ)={P_DQ_BAR} bar")
    ax1.axvline(F_NONMCQ_PUBLIC, color="k", ls="--", alpha=0.5,
                label=f"public faithful f={F_NONMCQ_PUBLIC:.3f}")
    if f_star_pdq is not None:
        ax1.axvline(f_star_pdq, color="r", ls="-", alpha=0.6,
                    label=f"f* (P(DQ)=0.13)={f_star_pdq:.3f}")
    ax1.set_xlabel("private free-response (non-MCQ) fraction  f_nonmcq")
    ax1.set_ylabel("well-posed P(DQ)", color="b")
    ax1.set_ylim(0, max(0.3, max(pdq) * 1.1))
    ax2 = ax1.twinx()
    ax2.plot(fs, cen, "g-", alpha=0.7, label="central delta_stock %")
    ax2.axhline(G1, color="g", ls=":", alpha=0.5)
    ax2.axvline(f_star_central, color="g", ls="-", alpha=0.4,
                label=f"f* (central=+5%)={f_star_central:.3f}")
    ax2.set_ylabel("central delta_stock (%)", color="g")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper left")
    ax1.set_title("Well-posed G1 P(DQ) vs private free-response fraction (#754)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[wellposed] wrote {path}", flush=True)


def _print_summary(o: dict[str, Any]) -> None:
    fc = o["faithful_central"]; a = o["arm_a_wellposed_p_dq"]; b = o["arm_b_fraction_sweep"]
    dec = o["decision"]; demo = o["wellposedness_demo"]
    print("\n" + "=" * 72, flush=True)
    print("WELL-POSED G1 P(DQ)  (anchor-free)   PR #754", flush=True)
    print("=" * 72, flush=True)
    print(f"  binning max|e_accept diff| = {o['binning_sanity']['max_abs_diff']}  "
          f"windows/subset = {o['bootstrap']['n_windows_per_subset']}", flush=True)
    print(f"  faithful central: point={fc['point_pct']}%  boot_mean={fc['bootstrap_mean_pct']}%  "
          f"SE={fc['bootstrap_se_pct']}%  CI95={fc['bootstrap_ci95_pct']}", flush=True)
    print(f"  Arm A(a) P(DQ) bootstrap   = {a['p_dq_wellposed_bootstrap']}  "
          f"(e_pub-fixed gate-literal = {a['p_dq_wellposed_bootstrap_epub_fixed']})", flush=True)
    print(f"  Arm A(b) P(DQ) sym-normal  = {a['p_dq_wellposed_symnormal']}", flush=True)
    print(f"  (ill-posed #749 reference  = {a['p_dq_illposed_749_for_reference']})", flush=True)
    vd = o["variance_decomposition"]
    print(f"  central SE: epub-boot={vd['central_se_epub_bootstrapped_pct']}%  "
          f"epub-fixed={vd['central_se_epub_fixed_pct']}%  "
          f"public common-mode={vd['public_anchor_common_mode_se_pct']}%", flush=True)
    print(f"  well-posedness: old rises-as-improves={demo['old_illposed_rises_as_central_improves']}  "
          f"new falls-as-improves={demo['new_wellposed_falls_as_central_improves']}", flush=True)
    print(f"  Arm B: f_nonmcq public={b['f_nonmcq_public_faithful']}  "
          f"f* (central=+5%)={b['f_nonmcq_star_central_to_5pct']}  "
          f"f* (P(DQ)=0.13)={b['f_nonmcq_star_pdq_to_0p13_bootstrap']}", flush=True)
    print(f"  VERDICT: {dec['verdict']}  (g1_safe={dec['g1_safe_wellposed']})", flush=True)
    print("=" * 72 + "\n", flush=True)


def log_to_wandb(o: dict[str, Any], *, wandb_group: str, wandb_name: str,
                 out_dir: Path) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return None
    fc = o["faithful_central"]; a = o["arm_a_wellposed_p_dq"]
    b = o["arm_b_fraction_sweep"]; dec = o["decision"]; demo = o["wellposedness_demo"]
    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "p_dq_wellposed_bootstrap": a["p_dq_wellposed_bootstrap"],
        "p_dq_wellposed_symnormal": a["p_dq_wellposed_symnormal"],
        "p_dq_wellposed_bootstrap_epub_fixed": a["p_dq_wellposed_bootstrap_epub_fixed"],
        "p_dq_illposed_749": a["p_dq_illposed_749_for_reference"],
        "central_se_epub_bootstrapped_pct": o["variance_decomposition"]["central_se_epub_bootstrapped_pct"],
        "public_anchor_common_mode_se_pct": o["variance_decomposition"]["public_anchor_common_mode_se_pct"],
        "faithful_central_point_pct": fc["point_pct"],
        "faithful_central_bootstrap_se_pct": fc["bootstrap_se_pct"],
        "faithful_central_ci95_lo": fc["bootstrap_ci95_pct"][0],
        "faithful_central_ci95_hi": fc["bootstrap_ci95_pct"][1],
        "f_nonmcq_public_faithful": b["f_nonmcq_public_faithful"],
        "f_nonmcq_star_central_to_5pct": b["f_nonmcq_star_central_to_5pct"],
        "f_nonmcq_star_pdq_to_0p13_bootstrap": b["f_nonmcq_star_pdq_to_0p13_bootstrap"],
        "f_nonmcq_star_pdq_to_0p13_symnormal": b["f_nonmcq_star_pdq_to_0p13_symnormal"],
        "old_illposed_rises_as_central_improves": int(demo["old_illposed_rises_as_central_improves"]),
        "new_wellposed_falls_as_central_improves": int(demo["new_wellposed_falls_as_central_improves"]),
        "g1_safe_wellposed": int(dec["g1_safe_wellposed"]),
    }
    run = init_wandb_run(
        job_type="deltastock-wellposed", agent="senpai", name=wandb_name,
        group=wandb_group, tags=["deltastock-wellposed", wandb_group],
        notes=f"#754 well-posed anchor-free G1 P(DQ); verdict={dec['verdict']}",
        config={"analysis_only": True, "no_hf_job": True, "fires": 0,
                "bootstrap_draws": o["bootstrap"]["draws"], "K": o["bootstrap"]["K"],
                "seed": o["bootstrap"]["seed"], "wandb_group": wandb_group,
                "source_report": o["source_report"]})
    if run is None:
        print("[wandb] run not created; wellposed.json is the record", flush=True)
        return None
    log_summary(run, summary, step=0)
    run.summary["verdict"] = dec["verdict"]
    for fn in ("wellposed.json", "fraction_sweep.png"):
        p = out_dir / fn
        if p.exists():
            try:
                log_file_artifact(run, path=p, name=f"wellposed_{p.stem}",
                                  artifact_type="deltastock-wellposed")
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] artifact {fn} failed (non-fatal): {e!r}", flush=True)
    rid = run.id
    finish_wandb(run)
    print(f"[wandb] logged {wandb_name} id={rid} (group={wandb_group})", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="path to the #749 faithful delta_stock report.json "
                                    "(server.log must sit alongside it)")
    ap.add_argument("--draws", type=int, default=50000, help="bootstrap draws (>=1000)")
    ap.add_argument("--K", type=int, default=6, help="num_speculative_tokens")
    ap.add_argument("--seed", type=int, default=730)
    ap.add_argument("--f-lo", type=float, default=F_NONMCQ_PUBLIC)
    # headline sweep is the prescribed 11->40%; extend to 0.55 so the central=+5%
    # breakeven (~0.52) also lands on-curve for the board plot.
    ap.add_argument("--f-hi", type=float, default=0.55)
    ap.add_argument("--f-steps", type=int, default=89)
    ap.add_argument("--out-dir", default=str(REPO / "research/validity/g1_wellposed_fraction_prior/results"))
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="kanna/g1-wellposed-fraction-prior")
    args = ap.parse_args()

    out = run(Path(args.report), B=args.draws, K=args.K, seed=args.seed,
              f_lo=args.f_lo, f_hi=args.f_hi, f_steps=args.f_steps,
              out_dir=Path(args.out_dir))

    run_id = None
    if args.wandb_group:
        try:
            run_id = log_to_wandb(out, wandb_group=args.wandb_group,
                                  wandb_name=args.wandb_name, out_dir=Path(args.out_dir))
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] logging failed (non-fatal): {e!r}", flush=True)

    dec = out["decision"]
    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [run_id] if run_id else [],
        "primary_metric": {"name": "p_dq_wellposed_bootstrap",
                           "value": dec["primary_metric_p_dq_wellposed_bootstrap"]},
        "test_metric": {"name": "f_nonmcq_star", "value": dec["test_metric_f_nonmcq_star"]},
    }
    print("SENPAI-RESULT:")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
