"""Pair two AIME maj@k result JSONs (base + ship) into the A/B deliverable (PR #514).

The card's PRIMARY deliverable is the base-vs-ship AIME delta under *real sampling*:
does the served surgical-357 ship reproduce stock int4 base on AIME maj@k, completing
the MMLU-Pro / GPQA-Diamond / **AIME** quality triad. This script consumes two
``aime_eval.py`` outputs, enforces that they are apples-to-apples (same problem ids,
same k, same sampling params + seed, same years), and computes:

  * ``base_aime`` / ``ship_aime``           -- maj@k accuracy of each row.
  * ``aime_delta_ship_minus_base``          -- the headline identity number (~0 expected).
  * ``mean_pass_rate`` delta (paired)       -- a finer, lower-variance distribution-match
                                               signal than the discrete maj@k bit.
  * a sampling-noise floor for each delta    -- two *independently sampled* runs differ by
                                               sampling noise even when the underlying
                                               distribution is identical; the verdict is
                                               "preserved" only if |delta| sits inside that
                                               floor.
  * ``ship_preserves_aime``                  -- AND of the two deltas-within-noise tests.

Why this is a real measurement and not a foregone ~0: the ship's lm_head is pruned to a
12k keepset and ``compute_logits`` scatters those logits back to the full vocab with
``-inf`` at every non-kept position (serve_patch_pck04.py). Greedy (argmax) is byte-exact
to base because the argmax token is always kept; but maj@k samples at top_k=64 over the
*kept* support. The delta is therefore exactly "do AIME's top-64 tokens ever fall outside
the 12k keepset" -- empirical, and the end-to-end confirmation of denken #505's
kernel-level distribution-preservation proof on the hardest, fully-sampled benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Sampling fields that MUST match for the A/B to be apples-to-apples.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed", "enable_thinking")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _answer_dist(per_problem_entry: dict[str, Any], k: int) -> dict[str, float]:
    """Empirical answer distribution for one problem, including an explicit 'none'
    mass for extraction failures so two rows are compared over the same symbol set."""
    counts = dict(per_problem_entry.get("answer_counts") or {})
    extracted = sum(counts.values())
    none_mass = max(0, k - extracted)
    dist = {str(a): c / k for a, c in counts.items()}
    if none_mass:
        dist["__none__"] = none_mass / k
    return dist


def _tv(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys)


def _binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n > 0 else 0.0


def combine(base: dict[str, Any], ship: dict[str, Any], regime: str = "sampling") -> dict[str, Any]:
    # --- parity: the experiment is only valid if both rows are apples-to-apples ---
    parity_issues: list[str] = []
    if list(base.get("years") or []) != list(ship.get("years") or []):
        parity_issues.append(f"years differ: base={base.get('years')} ship={ship.get('years')}")
    if base.get("k") != ship.get("k"):
        parity_issues.append(f"k differs: base={base.get('k')} ship={ship.get('k')}")
    bsamp, ssamp = base.get("sampling") or {}, ship.get("sampling") or {}
    for key in _SAMPLING_KEYS:
        if bsamp.get(key) != ssamp.get(key):
            parity_issues.append(f"sampling.{key} differs: base={bsamp.get(key)} ship={ssamp.get(key)}")

    base_by_id = {p["id"]: p for p in base.get("per_problem", [])}
    ship_by_id = {p["id"]: p for p in ship.get("per_problem", [])}
    common_ids = [pid for pid in base_by_id if pid in ship_by_id]
    if set(base_by_id) != set(ship_by_id):
        only_b = sorted(set(base_by_id) - set(ship_by_id))
        only_s = sorted(set(ship_by_id) - set(base_by_id))
        parity_issues.append(f"problem-id mismatch: base-only={only_b[:5]} ship-only={only_s[:5]}")

    k = int(base.get("k") or ship.get("k") or 0)
    n = len(common_ids)

    # --- per-problem paired comparison over the common problem set ---
    base_pass = [base_by_id[pid]["pass_rate"] for pid in common_ids]
    ship_pass = [ship_by_id[pid]["pass_rate"] for pid in common_ids]
    paired_d = [s - b for s, b in zip(ship_pass, base_pass)]
    tv_per_problem = [
        _tv(_answer_dist(base_by_id[pid], k), _answer_dist(ship_by_id[pid], k)) for pid in common_ids
    ]
    maj_agree = sum(
        1 for pid in common_ids if base_by_id[pid]["maj_answer"] == ship_by_id[pid]["maj_answer"]
    )

    # maj@k accuracy on the common set (recompute so it tracks common_ids exactly).
    base_maj = sum(int(base_by_id[pid]["maj_correct"]) for pid in common_ids) / n if n else 0.0
    ship_maj = sum(int(ship_by_id[pid]["maj_correct"]) for pid in common_ids) / n if n else 0.0
    base_mpr = sum(base_pass) / n if n else 0.0
    ship_mpr = sum(ship_pass) / n if n else 0.0

    delta_maj = ship_maj - base_maj
    delta_mpr = ship_mpr - base_mpr

    # noise floors. maj@k: two independent binomials. pass-rate: paired SE across problems.
    se_delta_maj = math.sqrt(_binom_se(base_maj, n) ** 2 + _binom_se(ship_maj, n) ** 2)
    se_paired_mpr = (statistics.stdev(paired_d) / math.sqrt(n)) if n > 1 else 0.0

    def _within(delta: float, se: float) -> bool:
        band = 2.0 * se
        return abs(delta) <= band if band > 0 else abs(delta) < 1e-9

    maj_within = _within(delta_maj, se_delta_maj)
    mpr_within = _within(delta_mpr, se_paired_mpr)
    ship_preserves = bool(maj_within and mpr_within and not parity_issues)

    mean_tv = sum(tv_per_problem) / n if n else 0.0

    verdict = (
        f"ship REPRODUCES base AIME under maj@k {regime} decode (delta within noise floor)"
        if ship_preserves
        else f"ship DIVERGES from base AIME under maj@k {regime} decode (delta exceeds noise floor)"
    )
    if parity_issues:
        verdict = "INVALID A/B (rows not apples-to-apples): " + "; ".join(parity_issues)

    return {
        "aime_year(s)": base.get("years"),
        "decode_regime": regime,
        "maj_k": k,
        "n_problems": n,
        "base_aime": base_maj,
        "ship_aime": ship_maj,
        "aime_delta_ship_minus_base": delta_maj,
        "aime_delta_2se_band": 2.0 * se_delta_maj,
        "base_mean_pass_rate": base_mpr,
        "ship_mean_pass_rate": ship_mpr,
        "mean_pass_rate_delta_ship_minus_base": delta_mpr,
        "mean_pass_rate_delta_2se_band": 2.0 * se_paired_mpr,
        "maj_within_noise": maj_within,
        "mean_pass_rate_within_noise": mpr_within,
        "ship_preserves_aime": ship_preserves,
        "maj_answer_agreement_frac": maj_agree / n if n else 0.0,
        "mean_per_problem_tv": mean_tv,
        "base_extract_fail_rate": base.get("extract_fail_rate"),
        "ship_extract_fail_rate": ship.get("extract_fail_rate"),
        "parity_issues": parity_issues,
        "apples_to_apples": not parity_issues,
        "sampling": bsamp,
        "verdict": verdict,
        "per_problem": [
            {
                "id": pid,
                "gold": base_by_id[pid]["gold"],
                "base_maj": base_by_id[pid]["maj_answer"],
                "ship_maj": ship_by_id[pid]["maj_answer"],
                "base_pass": base_by_id[pid]["pass_rate"],
                "ship_pass": ship_by_id[pid]["pass_rate"],
                "tv": tv,
            }
            for pid, tv in zip(common_ids, tv_per_problem)
        ],
    }


def _wandb_log(combined: dict[str, Any], base: dict[str, Any], ship: dict[str, Any], args: argparse.Namespace) -> str | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from scripts import wandb_logging
    except Exception as exc:  # pragma: no cover
        print(f"[aime-combine] wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return None

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "experiment": "downstream-quality-aime",
        "decode_regime": combined.get("decode_regime"),
        "aime_years": combined["aime_year(s)"],
        "maj_k": combined["maj_k"],
        "n_problems": combined["n_problems"],
        "sampling": combined["sampling"],
        "base_submission": base.get("submission"),
        "ship_submission": ship.get("submission"),
        "base_serve_overrides": base.get("serve_overrides"),
        "ship_serve_overrides": ship.get("serve_overrides"),
        "base_model": base.get("model"),
        "ship_model": ship.get("model"),
        "pr": 514,
    }
    run = wandb_logging.init_wandb_run(
        job_type="downstream-quality-aime",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="AIME maj@k base-vs-ship A/B; end-to-end sampled-distribution check of surgical-357 (PR #514).",
        tags=["aime", "downstream-quality", "analysis-only", "pr-514"],
        config=config,
    )
    if run is None:
        print("[aime-combine] wandb disabled/unavailable; skipping log", flush=True)
        return None

    summary = {kk: vv for kk, vv in combined.items() if kk != "per_problem"}
    wandb_logging.log_summary(run, summary, step=0)
    # Per-problem A/B table for later analysis.
    try:
        import wandb

        cols = ["id", "gold", "base_maj", "ship_maj", "base_pass", "ship_pass", "tv"]
        table = wandb.Table(columns=cols)
        for row in combined["per_problem"]:
            table.add_data(*[row[c] for c in cols])
        run.log({"global_step": 0, "aime_ab_table": table})
    except Exception as exc:  # pragma: no cover
        print(f"[aime-combine] table log skipped: {exc}", flush=True)
    wandb_logging.log_json_artifact(run, name="aime_ab_combined", artifact_type="aime-eval", data=combined)
    wandb_logging.log_json_artifact(run, name="aime_base_raw", artifact_type="aime-eval", data=base)
    wandb_logging.log_json_artifact(run, name="aime_ship_raw", artifact_type="aime-eval", data=ship)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="base (int4) aime_eval.py output JSON")
    ap.add_argument("--ship", type=Path, required=True, help="ship (surgical-357) aime_eval.py output JSON")
    ap.add_argument("--out", type=Path, required=True, help="combined A/B JSON path")
    ap.add_argument("--wandb", action="store_true", help="log the A/B to W&B")
    ap.add_argument("--wandb-name", default="fern/aime-base-vs-ship")
    ap.add_argument("--wandb-group", default="downstream-quality-aime")
    ap.add_argument("--regime", default="sampling", choices=["sampling", "greedy"],
                    help="decode regime label for the verdict/artifact (greedy = deployment-faithful)")
    args = ap.parse_args(argv)

    base, ship = _load(args.base), _load(args.ship)
    combined = combine(base, ship, regime=args.regime)
    combined["created_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    run_id = _wandb_log(combined, base, ship, args) if args.wandb else None
    combined["wandb_run_id"] = run_id

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2))
    print(f"[aime-combine] wrote {args.out}", flush=True)

    # The experiment SENPAI-RESULT (key outputs the PR asks for).
    senpai = {
        "analysis_only": True,
        "official_tps": 0,
        "aime_year(s)": combined["aime_year(s)"],
        "decode_regime": combined["decode_regime"],
        "maj_k": combined["maj_k"],
        "n_problems": combined["n_problems"],
        "base_aime": round(combined["base_aime"], 6),
        "ship_aime": round(combined["ship_aime"], 6),
        "aime_delta_ship_minus_base": round(combined["aime_delta_ship_minus_base"], 6),
        "ship_preserves_aime": combined["ship_preserves_aime"],
        "wandb_run_id": run_id,
    }
    print("SENPAI-RESULT " + json.dumps(senpai), flush=True)
    print("[aime-combine] VERDICT: " + combined["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
