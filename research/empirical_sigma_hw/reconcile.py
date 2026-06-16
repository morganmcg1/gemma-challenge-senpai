"""Two-leg sigma_hw reconciliation (PR #467, item 3-4) — the decision-relevant output.

The N=10 driver (``run_sigma_hw.py`` -> ``fresh_n10/sigma_hw.json``, W&B
``jb1a0lab``) measures ONE leg of the served-TPS envelope: the **within-device**
fresh-process run-to-run sigma on the pinned local A10G. This module COMBINES that
measured within-leg with the prior in-launch **between-allocation** leg (cited, not
re-measured) to reconcile the 1% convention and re-state every materiality verdict
on a *basis-matched* footing.

CPU-only, no GPU, no serve — replays the finished JSON so a W&B hiccup never costs
the 43-min benchmark. Run under the repo .venv::

    .venv/bin/python -m research.empirical_sigma_hw.reconcile

Provenance of the cited between-leg (all MERGED on the advisor branch):
  * land #451 (c675zor8): asserts sigma_hw = 4.8153 TPS = 1.00% of deployed 481.53
    (the "1% convention", EXPERIMENTS_LOG line 21).
  * kanna #159: fresh noise floor, n=12 fresh-server restarts on ONE pinned A10G
    -> sigma_within = 0.0111% (0.056 TPS).
  * frantic-penguin same-submission 3-draw across the HF a10g-small POOL (3 independent
    allocations) -> sigma_between = 0.9623% (4.864 TPS).
  * kanna #188 (pp1r5orx): sigma_oneshot = sqrt(within^2 + between^2) = 4.864 == #159
    sigma_hw exactly; ratio between/within ~= 86.6x => CROSS-ALLOCATION DOMINATED.

So the convention is NOT the within-device noise: it is the between-allocation draw a
single official launch faces. My N=10 measures the within-leg independently and finds
it ~13x tighter than the convention -> the convention is VINDICATED for a single
official draw (within-leg does not widen the one-shot) but ~13x TOO LOOSE for
same-device LOCAL A/Bs (the re-anchor work, e.g. #455/#463).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import wandb  # noqa: F401  (import first to win over the ./wandb shadow dir)

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402

# --------------------------------------------------------------------------- #
# Program constants under reconciliation                                       #
# --------------------------------------------------------------------------- #
DEPLOYED_OFFICIAL_TPS = 481.53
CONVENTION_SIGMA_HW = 4.8153          # land #451 (c675zor8): 1.00% x 481.53

# Cited between-allocation leg (NOT measured here; frantic-penguin 3 official draws,
# banked via kanna #159 / #188 pp1r5orx). Two equivalent expressions of the same leg.
SIGMA_BETWEEN_CITED_PCT = 0.9623      # %  (kanna #188)
SIGMA_BETWEEN_CITED_TPS = 4.864       # TPS (kanna #188 == #159 sigma_hw)
SIGMA_WITHIN_159_PCT = 0.0111         # kanna #159 within-device floor (n=12), for context

# Materiality axes (PR #467 baseline section)
STRICT_FRONTIER = 467.14
STRICT_DEPLOYED_GAP = 14.39           # 481.53 - 467.14
UNIFIED_CEILING = 510.87
CEILING_REANCHOR_463 = 510.654        # my #463 re-anchored read-peak ceiling
CEILING_DELTA = abs(UNIFIED_CEILING - CEILING_REANCHOR_463)  # 0.216
MATERIALITY_BAR_TPS = 2.0
# stark #466 "hold at ~467 vs collapse toward 162" separation
HOLD_TPS = 467.14
COLLAPSE_TPS = 162.0


def reconcile(analysis: dict[str, Any]) -> dict[str, Any]:
    # --- measured within-device leg (mine, N=10) ---
    frac = analysis["empirical_sigma_hw_frac"]                  # 0.0007256
    within_pct = analysis["empirical_sigma_hw_frac_pct"]        # 0.07256 %
    within_tps_at_481 = analysis["empirical_sigma_hw_tps"]      # frac x 481.53
    median = analysis["empirical_served_tps_median"]

    # --- cited between-allocation leg ---
    between_tps = SIGMA_BETWEEN_CITED_TPS
    between_pct = SIGMA_BETWEEN_CITED_PCT

    # --- one-shot reconstruction (TPS-space quadrature) ---
    oneshot_tps = math.hypot(within_tps_at_481, between_tps)
    oneshot_pct = 100.0 * oneshot_tps / DEPLOYED_OFFICIAL_TPS
    between_over_within = between_tps / within_tps_at_481
    within_contribution_to_oneshot_pct = 100.0 * (oneshot_tps - between_tps) / between_tps
    # convention checks
    oneshot_vs_convention_pct = 100.0 * (oneshot_tps - CONVENTION_SIGMA_HW) / CONVENTION_SIGMA_HW
    oneshot_reconstructs_convention = abs(oneshot_vs_convention_pct) <= 2.5
    within_negligible_in_oneshot = within_contribution_to_oneshot_pct < 1.0

    # --- basis-matched materiality ---
    def in_sigma(gap, sigma):
        return gap / sigma if sigma else float("nan")

    strict_in_within = in_sigma(STRICT_DEPLOYED_GAP, within_tps_at_481)
    strict_in_between = in_sigma(STRICT_DEPLOYED_GAP, between_tps)
    strict_in_oneshot = in_sigma(STRICT_DEPLOYED_GAP, oneshot_tps)
    # material at the conventional 3-sigma threshold under EVERY basis?
    strict_material_all_bases = min(strict_in_within, strict_in_between, strict_in_oneshot) >= 2.9

    ceil_in_within = in_sigma(CEILING_DELTA, within_tps_at_481)
    ceil_in_oneshot = in_sigma(CEILING_DELTA, oneshot_tps)
    ceiling_holds_all_bases = max(ceil_in_within, ceil_in_oneshot) <= 1.0

    plus2_in_within = in_sigma(MATERIALITY_BAR_TPS, within_tps_at_481)
    plus2_in_between = in_sigma(MATERIALITY_BAR_TPS, between_tps)
    plus2_in_oneshot = in_sigma(MATERIALITY_BAR_TPS, oneshot_tps)

    rec_local_screen_3sigma = 3.0 * within_tps_at_481
    rec_official_shot_3sigma = 3.0 * between_tps

    # --- stark #466 reassurance: hold-vs-collapse is sigma-independent ---
    sep = HOLD_TPS - COLLAPSE_TPS
    sep_in_within = in_sigma(sep, within_tps_at_481)
    sep_in_oneshot = in_sigma(sep, oneshot_tps)
    hold_collapse_sigma_independent = min(sep_in_within, sep_in_oneshot) > 10.0

    return {
        # measured within-leg (mine)
        "sigma_within_measured_pct": within_pct,
        "sigma_within_measured_tps_at_481": within_tps_at_481,
        "sigma_within_local_pstdev_tps": analysis["empirical_sigma_hw_local_tps"],
        "n_served_repeats": analysis["n_served_repeats"],
        "served_tps_median_local": median,
        # cited between-leg
        "sigma_between_cited_pct": between_pct,
        "sigma_between_cited_tps": between_tps,
        "sigma_within_159_cited_pct": SIGMA_WITHIN_159_PCT,
        # reconstruction
        "sigma_oneshot_reconstructed_tps": oneshot_tps,
        "sigma_oneshot_reconstructed_pct": oneshot_pct,
        "between_over_within_ratio": between_over_within,
        "within_contribution_to_oneshot_pct": within_contribution_to_oneshot_pct,
        "convention_sigma_hw": CONVENTION_SIGMA_HW,
        "oneshot_vs_convention_drift_pct": oneshot_vs_convention_pct,
        "oneshot_reconstructs_convention": oneshot_reconstructs_convention,
        "within_negligible_in_oneshot": within_negligible_in_oneshot,
        "convention_over_within_ratio": within_tps_to_ratio(within_tps_at_481),
        # verdicts on the convention
        "convention_vindicated_for_official_draw": oneshot_reconstructs_convention and within_negligible_in_oneshot,
        "convention_too_loose_for_local_AB": (between_tps / within_tps_at_481) > 3.0,
        # basis-matched materiality
        "strict_gap_in_within_sigma": strict_in_within,
        "strict_gap_in_between_sigma": strict_in_between,
        "strict_gap_in_oneshot_sigma": strict_in_oneshot,
        "strict_gap_material_all_bases": strict_material_all_bases,
        "ceiling_delta_in_within_sigma": ceil_in_within,
        "ceiling_delta_in_oneshot_sigma": ceil_in_oneshot,
        "ceiling_holds_all_bases": ceiling_holds_all_bases,
        "plus2_bar_in_within_sigma": plus2_in_within,
        "plus2_bar_in_between_sigma": plus2_in_between,
        "plus2_bar_in_oneshot_sigma": plus2_in_oneshot,
        "recommended_local_screen_3sigma_tps": rec_local_screen_3sigma,
        "recommended_official_shot_3sigma_tps": rec_official_shot_3sigma,
        # stark #466
        "hold_collapse_separation_tps": sep,
        "hold_collapse_in_within_sigma": sep_in_within,
        "hold_collapse_in_oneshot_sigma": sep_in_oneshot,
        "hold_collapse_sigma_independent": hold_collapse_sigma_independent,
    }


def within_tps_to_ratio(within_tps_at_481: float) -> float:
    return CONVENTION_SIGMA_HW / within_tps_at_481 if within_tps_at_481 else float("nan")


def _print(r: dict[str, Any]) -> None:
    print("\n[reconcile] ===== TWO-LEG sigma_hw RECONCILIATION =====", flush=True)
    print(f"  within-device (MEASURED, N={r['n_served_repeats']}): "
          f"{r['sigma_within_measured_pct']:.4f}% = {r['sigma_within_measured_tps_at_481']:.4f} TPS @481.53", flush=True)
    print(f"  between-alloc (CITED #159/#188/frantic-penguin): "
          f"{r['sigma_between_cited_pct']:.4f}% = {r['sigma_between_cited_tps']:.4f} TPS", flush=True)
    print(f"  one-shot = sqrt(within^2+between^2) = {r['sigma_oneshot_reconstructed_tps']:.4f} TPS "
          f"({r['sigma_oneshot_reconstructed_pct']:.4f}%)", flush=True)
    print(f"  convention 4.8153 -> one-shot drift {r['oneshot_vs_convention_drift_pct']:+.2f}%  "
          f"reconstructs={r['oneshot_reconstructs_convention']}  "
          f"within-negligible={r['within_negligible_in_oneshot']}", flush=True)
    print(f"  between/within = {r['between_over_within_ratio']:.1f}x  "
          f"convention/within = {r['convention_over_within_ratio']:.1f}x", flush=True)
    print(f"  VERDICT: vindicated-for-official-draw={r['convention_vindicated_for_official_draw']}  "
          f"too-loose-for-local-AB={r['convention_too_loose_for_local_AB']}", flush=True)
    print("  -- basis-matched materiality --", flush=True)
    print(f"  strict 14.39 gap: within {r['strict_gap_in_within_sigma']:.1f}sig | "
          f"between {r['strict_gap_in_between_sigma']:.2f}sig | one-shot {r['strict_gap_in_oneshot_sigma']:.2f}sig "
          f"-> material_all_bases={r['strict_gap_material_all_bases']}", flush=True)
    print(f"  ceiling 0.216 delta: within {r['ceiling_delta_in_within_sigma']:.2f}sig | "
          f"one-shot {r['ceiling_delta_in_oneshot_sigma']:.3f}sig -> holds_all_bases={r['ceiling_holds_all_bases']}", flush=True)
    print(f"  +2 bar: within {r['plus2_bar_in_within_sigma']:.1f}sig | one-shot {r['plus2_bar_in_oneshot_sigma']:.2f}sig", flush=True)
    print(f"  rec 3-sigma bars: local-screen {r['recommended_local_screen_3sigma_tps']:.2f} TPS | "
          f"official-shot {r['recommended_official_shot_3sigma_tps']:.2f} TPS", flush=True)
    print(f"  #466 hold(467)-vs-collapse(162) sep {r['hold_collapse_separation_tps']:.0f} TPS: "
          f"within {r['hold_collapse_in_within_sigma']:.0f}sig | one-shot {r['hold_collapse_in_oneshot_sigma']:.0f}sig "
          f"-> sigma-independent={r['hold_collapse_sigma_independent']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma-json", type=Path,
                    default=ROOT / "research/empirical_sigma_hw/fresh_n10/sigma_hw.json")
    ap.add_argument("--name", default="lawine/empirical-sigma-hw-reconcile")
    ap.add_argument("--group", default="equivalence-escalation-anchors")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    sigma = json.loads(args.sigma_json.read_text())
    analysis = sigma["analysis"]
    r = reconcile(analysis)
    _print(r)

    out_path = args.sigma_json.parent / "reconciliation.json"
    out_path.write_text(json.dumps(
        {"reconciliation": r, "source_run": analysis.get("wandb_run_id"),
         "source_sigma_json": str(args.sigma_json)}, indent=2))
    print(f"\n[reconcile] artifacts -> {out_path}", flush=True)

    if args.no_wandb:
        return 0
    run = wandb_logging.init_wandb_run(
        job_type="sigma-hw-reconciliation", agent="lawine",
        name=args.name, group=args.group,
        tags=["empirical-sigma-hw", "equivalence-escalation-anchors", "reconciliation",
              "two-leg", "fa2sw_precache_kenyan"],
        config={"deployed_official_tps": DEPLOYED_OFFICIAL_TPS,
                "convention_sigma_hw": CONVENTION_SIGMA_HW,
                "sigma_between_cited_tps": SIGMA_BETWEEN_CITED_TPS,
                "source_within_run": analysis.get("wandb_run_id"),
                "n_served_repeats": analysis["n_served_repeats"]},
    )
    if run is None:
        print("[reconcile] wandb disabled (no API key); skipping", flush=True)
        return 0
    flat = {}
    for k, v in r.items():
        if isinstance(v, bool):
            flat[k] = int(v)
        elif isinstance(v, (int, float)) and math.isfinite(v):
            flat[k] = v
    wandb_logging.log_summary(run, flat, step=0)
    wandb_logging.log_json_artifact(
        run, name="sigma_hw_reconciliation", artifact_type="sigma-hw-reconcile",
        data={"reconciliation": r, "source_run": analysis.get("wandb_run_id")})
    wandb_logging.finish_wandb(run)
    print(f"[reconcile] wandb_run_id={getattr(run, 'id', None)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
