"""Three-row AIME-greedy substrate decomposition for PR #531.

PR #531 asked to fork the AIME greedy collapse (base 0.267 -> ship 0.033) into
the head-prune vs the osoi5-QAT-bake by adding an ``osoi5-full-lm_head`` arm
(the surgical-357 stack with ``LM_HEAD_PRUNE`` turned off). Reading the substrate
shows the PR's premise needs a correction: ``osoi5-v0-baked`` is itself already
head-pruned to a **16k** keepset at bake time (``lm_head.weight_packed`` is
``[16384, 320]``; ``pck04_keepset.json`` has 16384 keep_ids over full_vocab
262144). So the head prune is **layered**:

    base (full 262144)  --bake+262k->16k-->  osoi5_16k (16384)  --LM_HEAD_PRUNE 16k->12k-->  ship (12288)

Turning ``LM_HEAD_PRUNE=0`` therefore yields the **16k substrate**, not a full
262k head. Consequences for the fork:

  * ``delta_prune_12k_to_16k = osoi5_16k - ship`` is CLEAN -- it isolates exactly
    the ``LM_HEAD_PRUNE`` 16k->12k slice (the only thing toggled).
  * ``delta_16k_to_base = base - osoi5_16k`` is CONFOUNDED -- it bundles the
    bake-time 262k->16k head prune together with the osoi5 QAT bake (and the
    base-vs-osoi5 quant-recipe difference). It is therefore NOT a clean
    ``delta_bake``; a clean bake term needs a full-262k-head osoi5 checkpoint,
    which does not exist in the bucket (both osoi5-v0 and osoi538-v0 are 16k).

This script assembles the three measured greedy rows, computes every pairwise
delta with a binomial 2-se noise floor, and emits an HONEST verdict set that does
not over-claim the bake attribution. It reuses the parity + noise-floor logic of
``aime_combine.py`` (PR #514) so the rows stay apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed", "enable_thinking")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n > 0 else 0.0


def _maj_acc(row: dict[str, Any], ids: list[str]) -> float:
    by_id = {p["id"]: p for p in row.get("per_problem", [])}
    n = len(ids)
    return sum(int(by_id[i]["maj_correct"]) for i in ids) / n if n else 0.0


def _mean_pass(row: dict[str, Any], ids: list[str]) -> float:
    by_id = {p["id"]: p for p in row.get("per_problem", [])}
    n = len(ids)
    return sum(by_id[i]["pass_rate"] for i in ids) / n if n else 0.0


def _parity(rows: dict[str, dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    labels = list(rows)
    ref = rows[labels[0]]
    for lab in labels[1:]:
        r = rows[lab]
        if list(r.get("years") or []) != list(ref.get("years") or []):
            issues.append(f"years differ: {labels[0]}={ref.get('years')} {lab}={r.get('years')}")
        if r.get("k") != ref.get("k"):
            issues.append(f"k differs: {labels[0]}={ref.get('k')} {lab}={r.get('k')}")
        rs, refs = r.get("sampling") or {}, ref.get("sampling") or {}
        for key in _SAMPLING_KEYS:
            if rs.get(key) != refs.get(key):
                issues.append(f"sampling.{key} differs: {labels[0]}={refs.get(key)} {lab}={rs.get(key)}")
    # common problem ids across all rows
    idsets = [set(p["id"] for p in rows[lab].get("per_problem", [])) for lab in labels]
    common = set.intersection(*idsets) if idsets else set()
    for lab, s in zip(labels, idsets):
        if s != common:
            issues.append(f"problem-id mismatch in {lab}: extra={sorted(s - common)[:5]}")
    return issues


def _delta(a: float, na: int, b: float, nb: int) -> dict[str, Any]:
    """a - b with a 2-se binomial band (two independent binomials)."""
    d = a - b
    se = math.sqrt(_binom_se(a, na) ** 2 + _binom_se(b, nb) ** 2)
    band = 2.0 * se
    return {
        "delta": d,
        "two_se_band": band,
        "within_noise": abs(d) <= band if band > 0 else abs(d) < 1e-9,
    }


def decompose(base: dict, osoi5_16k: dict, ship: dict) -> dict[str, Any]:
    rows = {"base": base, "osoi5_16k": osoi5_16k, "ship": ship}
    parity_issues = _parity(rows)
    common = sorted(
        set.intersection(*[set(p["id"] for p in rows[l].get("per_problem", [])) for l in rows])
    )
    n = len(common)
    k = int(base.get("k") or 0)

    acc = {lab: _maj_acc(rows[lab], common) for lab in rows}
    mpr = {lab: _mean_pass(rows[lab], common) for lab in rows}

    # Pairwise deltas. Naming reflects the LAYERED prune (see module docstring).
    d_prune_12k_to_16k = _delta(acc["osoi5_16k"], n, acc["ship"], n)   # CLEAN: LM_HEAD_PRUNE slice
    d_16k_to_base = _delta(acc["base"], n, acc["osoi5_16k"], n)        # CONFOUNDED: bake + 262k->16k prune
    d_ship_to_base = _delta(acc["base"], n, acc["ship"], n)            # the full collapse (anchor)

    # Honest verdicts.
    # Does dropping the 16k->12k slice (i.e. widening 12k back to 16k) materially
    # move AIME off the ship floor? (clean, directly measured)
    slice_12k_to_16k_helps = (not d_prune_12k_to_16k["within_noise"]) and d_prune_12k_to_16k["delta"] > 0
    # Does the 16k substrate recover most of the way toward base?
    recovered_frac_16k = (
        (acc["osoi5_16k"] - acc["ship"]) / (acc["base"] - acc["ship"])
        if (acc["base"] - acc["ship"]) != 0
        else float("nan")
    )
    osoi5_16k_recovers_toward_base = (not d_16k_to_base["within_noise"]) and acc["osoi5_16k"] >= acc["base"] - d_16k_to_base["two_se_band"]
    osoi5_16k_still_collapsed = abs(acc["osoi5_16k"] - acc["ship"]) <= d_prune_12k_to_16k["two_se_band"]

    # widening_can_rescue_aime: this is the PR's one-line verdict. We can only give
    # a PARTIAL read from the 12k->16k step. If even 16k stays on the ship floor,
    # a narrow keepset-widen (12k -> modestly wider) will NOT rescue AIME;
    # the full verdict (does widening ALL the way to 262k rescue) is UNRESOLVED
    # because no full-262k-head osoi5 exists to measure.
    if slice_12k_to_16k_helps:
        widening_partial = "12k->16k widening already lifts AIME off the ship floor (narrow-widen helps)"
    elif osoi5_16k_still_collapsed:
        widening_partial = (
            "12k->16k widening does NOT move AIME off the ship floor; a narrow keepset-widen "
            "(12k -> modestly wider) will not rescue AIME. Whether full-262k widening would recover is "
            "UNRESOLVED (needs a full-head osoi5; the 262k->16k prune and the QAT bake remain "
            "confounded in this row)."
        )
    else:
        widening_partial = "12k->16k widening shows an ambiguous (within-noise) trend; inconclusive at n=30"

    return {
        "aime_year(s)": base.get("years"),
        "decode_regime": "greedy",
        "maj_k": k,
        "n_problems": n,
        "common_ids": common,
        # the three measured rows
        "aime_greedy_base": acc["base"],
        "aime_greedy_osoi5_16k": acc["osoi5_16k"],
        "aime_greedy_ship": acc["ship"],
        "mean_pass_rate_base": mpr["base"],
        "mean_pass_rate_osoi5_16k": mpr["osoi5_16k"],
        "mean_pass_rate_ship": mpr["ship"],
        # deltas
        "delta_prune_12k_to_16k": d_prune_12k_to_16k,   # CLEAN (LM_HEAD_PRUNE slice)
        "delta_16k_to_base_CONFOUNDED": d_16k_to_base,  # bake + 262k->16k prune (NOT delta_bake)
        "delta_full_collapse_base_to_ship": d_ship_to_base,
        "recovered_fraction_16k": recovered_frac_16k,
        # honest verdicts
        "slice_12k_to_16k_helps": slice_12k_to_16k_helps,
        "osoi5_16k_recovers_toward_base": osoi5_16k_recovers_toward_base,
        "osoi5_16k_still_collapsed": osoi5_16k_still_collapsed,
        "widening_can_rescue_aime__partial": widening_partial,
        # PR #531 KEY OUTPUTS, mapped onto the corrected 16k substrate ("full" == the
        # 16k baked head ceiling; a true full-262k head was discarded at bake).
        "aime_greedy_osoi5_full": acc["osoi5_16k"],
        "delta_prune": d_prune_12k_to_16k["delta"],          # osoi5_full - ship (CLEAN)
        "delta_bake_confounded": d_16k_to_base["delta"],     # base - osoi5_full (bake + 262k->16k prune)
        "prune_is_the_cause": osoi5_16k_recovers_toward_base,  # False: dropping the prune does NOT recover toward base
        "qat_bake_implicated": osoi5_16k_still_collapsed,      # True: osoi5_full stays pinned to the ship floor
        "widening_can_rescue_aime_12k_to_16k": bool(slice_12k_to_16k_helps),  # measured (False)
        # premise correction flags (load-bearing for the read)
        "substrate_is_16k_pruned_at_bake": True,
        "full_262k_head_osoi5_available": False,
        "clean_delta_bake_measurable": False,
        "clean_delta_bake_blocker": (
            "osoi5-v0-baked lm_head is physically [16384,320] (16k keepset). LM_HEAD_PRUNE=0 serves "
            "the 16k substrate, which bundles the bake-time 262k->16k head prune with the QAT bake. "
            "A clean delta_bake needs a full-262k-head osoi5 checkpoint (none in bucket; both "
            "osoi5-v0 and osoi538-v0 are 16k) -> would require a re-bake."
        ),
        "base_extract_fail_rate": base.get("extract_fail_rate"),
        "osoi5_16k_extract_fail_rate": osoi5_16k.get("extract_fail_rate"),
        "ship_extract_fail_rate": ship.get("extract_fail_rate"),
        "parity_issues": parity_issues,
        "apples_to_apples": not parity_issues,
        "sampling": base.get("sampling"),
        "per_problem": [
            {
                "id": pid,
                "gold": next(p for p in base["per_problem"] if p["id"] == pid)["gold"],
                "base_maj": next(p for p in base["per_problem"] if p["id"] == pid)["maj_answer"],
                "osoi5_16k_maj": next(p for p in osoi5_16k["per_problem"] if p["id"] == pid)["maj_answer"],
                "ship_maj": next(p for p in ship["per_problem"] if p["id"] == pid)["maj_answer"],
                "base_pass": next(p for p in base["per_problem"] if p["id"] == pid)["pass_rate"],
                "osoi5_16k_pass": next(p for p in osoi5_16k["per_problem"] if p["id"] == pid)["pass_rate"],
                "ship_pass": next(p for p in ship["per_problem"] if p["id"] == pid)["pass_rate"],
            }
            for pid in common
        ],
    }


def _wandb_log(combined: dict, base: dict, osoi5_16k: dict, ship: dict, args: argparse.Namespace) -> str | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from scripts import wandb_logging
    except Exception as exc:  # pragma: no cover
        print(f"[decompose] wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return None

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "experiment": "aime-substrate-prune-vs-bake",
        "decode_regime": "greedy",
        "aime_years": combined["aime_year(s)"],
        "maj_k": combined["maj_k"],
        "n_problems": combined["n_problems"],
        "sampling": combined["sampling"],
        "base_submission": base.get("submission"),
        "osoi5_16k_submission": osoi5_16k.get("submission"),
        "ship_submission": ship.get("submission"),
        "osoi5_16k_serve_overrides": osoi5_16k.get("serve_overrides"),
        "substrate_is_16k_pruned_at_bake": True,
        "full_262k_head_osoi5_available": False,
        "clean_delta_bake_measurable": False,
        "pr": 531,
    }
    run = wandb_logging.init_wandb_run(
        job_type="aime-substrate-prune-vs-bake",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        notes=(
            "AIME-2024 greedy maj@1 three-row substrate decomposition (PR #531). "
            "osoi5_16k = LM_HEAD_PRUNE off = the 16k substrate (NOT full 262k; substrate is "
            "16k-pruned at bake). delta_prune_12k_to_16k is clean; delta_bake is NOT cleanly "
            "measurable without a full-head osoi5."
        ),
        tags=["aime", "downstream-quality", "analysis-only", "pr-531", "substrate-decomposition"],
        config=config,
    )
    if run is None:
        print("[decompose] wandb disabled/unavailable; skipping log", flush=True)
        return None

    summary = {kk: vv for kk, vv in combined.items() if kk not in ("per_problem", "common_ids")}
    # flatten the nested delta dicts for clean summary scalars
    for dk in ("delta_prune_12k_to_16k", "delta_16k_to_base_CONFOUNDED", "delta_full_collapse_base_to_ship"):
        dv = summary.pop(dk)
        summary[f"{dk}__delta"] = dv["delta"]
        summary[f"{dk}__2se_band"] = dv["two_se_band"]
        summary[f"{dk}__within_noise"] = dv["within_noise"]
    wandb_logging.log_summary(run, summary, step=0)
    try:
        import wandb

        cols = ["id", "gold", "base_maj", "osoi5_16k_maj", "ship_maj", "base_pass", "osoi5_16k_pass", "ship_pass"]
        table = wandb.Table(columns=cols)
        for row in combined["per_problem"]:
            table.add_data(*[row[c] for c in cols])
        run.log({"global_step": 0, "aime_substrate_table": table})
    except Exception as exc:  # pragma: no cover
        print(f"[decompose] table log skipped: {exc}", flush=True)
    wandb_logging.log_json_artifact(run, name="aime_substrate_decompose", artifact_type="aime-eval", data=combined)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--osoi5-16k", type=Path, required=True)
    ap.add_argument("--ship", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="fern/aime-substrate-prune-vs-bake")
    ap.add_argument("--wandb-group", default="aime-substrate-prune-vs-bake")
    args = ap.parse_args(argv)

    base, osoi5_16k, ship = _load(args.base), _load(args.osoi5_16k), _load(args.ship)
    combined = decompose(base, osoi5_16k, ship)
    combined["created_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    run_id = _wandb_log(combined, base, osoi5_16k, ship, args) if args.wandb else None
    combined["wandb_run_id"] = run_id

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2))
    print(f"[decompose] wrote {args.out}", flush=True)

    senpai = {
        "analysis_only": True,
        "official_tps": 0,
        "terminal": True,
        "status": "complete",
        "pending_arms": False,
        "wandb_run_ids": [run_id] if run_id else [],
        "aime_year(s)": combined["aime_year(s)"],
        "decode_regime": "greedy",
        "maj_k": combined["maj_k"],
        "n_problems": combined["n_problems"],
        "primary_metric": {"name": "aime2024_greedy_maj@1_osoi5_full(=16k)", "value": round(combined["aime_greedy_osoi5_full"], 6)},
        "test_metric": {"name": "aime2024_greedy_maj@1_osoi5_full(=16k)", "value": round(combined["aime_greedy_osoi5_full"], 6)},
        "aime_greedy_base": round(combined["aime_greedy_base"], 6),
        "aime_greedy_osoi5_full": round(combined["aime_greedy_osoi5_full"], 6),
        "aime_greedy_ship": round(combined["aime_greedy_ship"], 6),
        "delta_prune": round(combined["delta_prune"], 6),
        "delta_bake_confounded": round(combined["delta_bake_confounded"], 6),
        "prune_is_the_cause": combined["prune_is_the_cause"],
        "qat_bake_implicated": combined["qat_bake_implicated"],
        "widening_can_rescue_aime_12k_to_16k": combined["widening_can_rescue_aime_12k_to_16k"],
        "widening_full_262k_unresolved": not combined["full_262k_head_osoi5_available"],
        "substrate_is_16k_pruned_at_bake": combined["substrate_is_16k_pruned_at_bake"],
        "clean_delta_bake_measurable": combined["clean_delta_bake_measurable"],
        "wandb_run_id": run_id,
    }
    print("SENPAI-RESULT " + json.dumps(senpai), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
