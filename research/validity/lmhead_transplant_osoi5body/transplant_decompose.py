"""HEAD-vs-BODY verdict for the AIME collapse via lm_head transplant (PR #536).

Compares the **osoi5-body + base-262k-head transplant** against the two measured
reference rows from fern #531 (banked on the advisor branch), under the SAME
greedy maj@1 protocol so the rows are apples-to-apples:

    base      (base body + base 262k head)              = 0.267   [fern #531]
    osoi5_16k (osoi5 body + osoi5 16k-int4 pruned head) = 0.033   [fern #531]
    transplant(osoi5 body + base 262k head)             = THIS    [PR #536]

The transplant shares fern's osoi5_16k submission (fa2sw_strict_surgical357) and
body; the ONLY moved variable vs the 0.033 row is the head (16k-int4-pruned ->
full 262k BF16). The 262k head is realised by tying the output head to osoi5's
own embed_tokens, which is byte-identical to base embed_tokens (verified
head/mid/tail) and base is tied (base lm_head == base embed, verified step 1) ->
the tied head IS the base 262k head, exactly, at zero extraction cost.

Verdict logic (PR #536 thresholds + a binomial 2-se noise floor):
  * HEAD  -- transplant recovers toward base (>= ~0.20): the collapse lived in
            the pruned/quantised head; a re-bake-free fast ship exists.
  * BODY  -- transplant stays on the osoi5 floor (<= ~0.10, within noise of
            0.033): the QAT bake degraded the transformer; only base-int4 or a
            re-bake fixes it.
  * MIXED -- partial recovery between the two.
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
HEAD_THRESH = 0.20   # >= this AIME maj@1 => collapse was the head
BODY_THRESH = 0.10   # <= this => collapse is the body (stays on osoi5 floor)


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


def _delta(a: float, na: int, b: float, nb: int) -> dict[str, Any]:
    d = a - b
    se = math.sqrt(_binom_se(a, na) ** 2 + _binom_se(b, nb) ** 2)
    band = 2.0 * se
    return {"delta": d, "two_se_band": band, "within_noise": abs(d) <= band if band > 0 else abs(d) < 1e-9}


def _base_row_from_decompose(dec: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct fern's base row (per-problem maj_correct/pass) from the banked
    decompose.json, so we don't need the raw base JSON to compute pct-of-base."""
    pp = []
    for r in dec["per_problem"]:
        pp.append({
            "id": r["id"], "gold": r["gold"], "maj_answer": r["base_maj"],
            "maj_correct": r["base_maj"] is not None and r["base_maj"] == r["gold"],
            "pass_rate": r["base_pass"],
        })
    return {"per_problem": pp, "sampling": dec.get("sampling"), "k": dec.get("maj_k"),
            "years": dec.get("aime_year(s)"), "label": "base_reconstructed"}


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
    idsets = [set(p["id"] for p in rows[lab].get("per_problem", [])) for lab in labels]
    common = set.intersection(*idsets) if idsets else set()
    for lab, s in zip(labels, idsets):
        if s != common:
            issues.append(f"problem-id mismatch in {lab}: extra={sorted(s - common)[:5]}")
    return issues


def decompose(transplant: dict, osoi5_16k: dict, base: dict) -> dict[str, Any]:
    rows = {"base": base, "osoi5_16k": osoi5_16k, "transplant": transplant}
    parity_issues = _parity(rows)
    common = sorted(set.intersection(*[set(p["id"] for p in rows[l].get("per_problem", [])) for l in rows]))
    n = len(common)

    acc = {lab: _maj_acc(rows[lab], common) for lab in rows}
    mpr = {lab: _mean_pass(rows[lab], common) for lab in rows}

    # The decisive A/B: transplant vs the osoi5 floor (only the head moved).
    d_head_swap = _delta(acc["transplant"], n, acc["osoi5_16k"], n)   # >0 => head swap helped
    d_to_base = _delta(acc["base"], n, acc["transplant"], n)          # ~0 => fully recovered to base
    full_collapse = acc["base"] - acc["osoi5_16k"]
    recovered_frac = ((acc["transplant"] - acc["osoi5_16k"]) / full_collapse) if full_collapse else float("nan")

    t = acc["transplant"]
    recovers_off_floor = (not d_head_swap["within_noise"]) and d_head_swap["delta"] > 0
    on_base = d_to_base["within_noise"] or t >= acc["base"] - d_to_base["two_se_band"]
    if t >= HEAD_THRESH and recovers_off_floor:
        locus = "HEAD"
    elif t <= BODY_THRESH and not recovers_off_floor:
        locus = "BODY"
    else:
        locus = "MIXED"
    cheap_fast_ship_exists = locus == "HEAD"

    return {
        "aime_year(s)": base.get("years"),
        "decode_regime": "greedy",
        "maj_k": int(base.get("k") or 1),
        "n_problems": n,
        "common_ids": common,
        # rows
        "aime_greedy_base": acc["base"],
        "aime_greedy_osoi5_16k": acc["osoi5_16k"],
        "aime_greedy_transplant": acc["transplant"],
        "mean_pass_rate_base": mpr["base"],
        "mean_pass_rate_osoi5_16k": mpr["osoi5_16k"],
        "mean_pass_rate_transplant": mpr["transplant"],
        "aime_pct_of_base": (acc["transplant"] / acc["base"]) if acc["base"] else float("nan"),
        # deltas
        "delta_head_swap_vs_osoi5_16k": d_head_swap,   # the decisive cell
        "delta_transplant_to_base": d_to_base,
        "full_collapse_base_minus_osoi5_16k": full_collapse,
        "recovered_fraction": recovered_frac,
        # verdicts
        "collapse_locus": locus,
        "cheap_fast_ship_exists": cheap_fast_ship_exists,
        "transplant_recovers_off_floor": recovers_off_floor,
        "transplant_on_base": bool(on_base),
        # extract-fail diagnostic (osoi5 floor had 0.367 fail; does the head fix parse-ability?)
        "base_extract_fail_rate": base.get("extract_fail_rate"),
        "osoi5_16k_extract_fail_rate": osoi5_16k.get("extract_fail_rate"),
        "transplant_extract_fail_rate": transplant.get("extract_fail_rate"),
        # validity (proven separately)
        "head_input_dim_compatible": True,
        "tied_head_is_base_head_byte_verified": True,
        "only_moved_variable": "lm_head (osoi5 16k-int4-pruned -> base 262k BF16, via tie to byte-identical embed)",
        "parity_issues": parity_issues,
        "apples_to_apples": not parity_issues,
        "sampling": base.get("sampling"),
        "per_problem": [
            {
                "id": pid,
                "gold": next(p for p in base["per_problem"] if p["id"] == pid)["gold"],
                "base_maj": next(p for p in base["per_problem"] if p["id"] == pid)["maj_answer"],
                "osoi5_16k_maj": next(p for p in osoi5_16k["per_problem"] if p["id"] == pid)["maj_answer"],
                "transplant_maj": next(p for p in transplant["per_problem"] if p["id"] == pid)["maj_answer"],
            }
            for pid in common
        ],
    }


def _wandb_log(combined: dict, transplant: dict, osoi5_16k: dict, args: argparse.Namespace) -> str | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[transplant] wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return None
    config = {
        "analysis_only": True, "official_tps": 0, "experiment": "lm-head-transplant-osoi5body",
        "decode_regime": "greedy", "aime_years": combined["aime_year(s)"], "maj_k": combined["maj_k"],
        "n_problems": combined["n_problems"], "sampling": combined["sampling"],
        "transplant_submission": transplant.get("submission"),
        "transplant_serve_overrides": transplant.get("serve_overrides"),
        "head_input_dim_compatible": True, "tied_head_is_base_head_byte_verified": True, "pr": 536,
    }
    run = wandb_logging.init_wandb_run(
        job_type="lm-head-transplant", agent="stark", name=args.wandb_name, group=args.wandb_group,
        notes=("osoi5-body + base-262k-head transplant AIME-2024 greedy maj@1 (PR #536). "
               "Decisive head-vs-body probe; only the head differs vs fern #531 osoi5_16k=0.033."),
        tags=["aime", "downstream-quality", "analysis-only", "pr-536", "lm-head-transplant"],
        config=config,
    )
    if run is None:
        print("[transplant] wandb disabled/unavailable; skipping log", flush=True)
        return None
    summary = {kk: vv for kk, vv in combined.items() if kk not in ("per_problem", "common_ids")}
    for dk in ("delta_head_swap_vs_osoi5_16k", "delta_transplant_to_base"):
        dv = summary.pop(dk)
        summary[f"{dk}__delta"] = dv["delta"]
        summary[f"{dk}__2se_band"] = dv["two_se_band"]
        summary[f"{dk}__within_noise"] = dv["within_noise"]
    wandb_logging.log_summary(run, summary, step=0)
    try:
        import wandb
        cols = ["id", "gold", "base_maj", "osoi5_16k_maj", "transplant_maj"]
        table = wandb.Table(columns=cols)
        for row in combined["per_problem"]:
            table.add_data(*[row[c] for c in cols])
        run.log({"global_step": 0, "aime_transplant_table": table})
    except Exception as exc:
        print(f"[transplant] table log skipped: {exc}", flush=True)
    wandb_logging.log_json_artifact(run, name="aime_transplant_decompose", artifact_type="aime-eval", data=combined)
    rid = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transplant", type=Path, required=True)
    ap.add_argument("--osoi5-16k", type=Path, required=True)
    ap.add_argument("--decompose", type=Path, required=True, help="fern #531 decompose.json (for the base row)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="stark/lm-head-transplant-osoi5body")
    ap.add_argument("--wandb-group", default="head-transplant-osoi5body")
    args = ap.parse_args(argv)

    transplant, osoi5_16k, dec = _load(args.transplant), _load(args.osoi5_16k), _load(args.decompose)
    base = _base_row_from_decompose(dec)
    combined = decompose(transplant, osoi5_16k, base)
    combined["created_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    run_id = _wandb_log(combined, transplant, osoi5_16k, args) if args.wandb else None
    combined["wandb_run_id"] = run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2))
    print(f"[transplant] wrote {args.out}", flush=True)

    senpai = {
        "analysis_only": True, "official_tps": 0, "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [run_id] if run_id else [],
        "primary_metric": {"name": "aime2024_greedy_maj@1_transplant", "value": round(combined["aime_greedy_transplant"], 6)},
        "test_metric": {"name": "aime2024_greedy_maj@1_transplant", "value": round(combined["aime_greedy_transplant"], 6)},
        "aime_greedy_base": round(combined["aime_greedy_base"], 6),
        "aime_greedy_osoi5_16k": round(combined["aime_greedy_osoi5_16k"], 6),
        "aime_greedy_transplant": round(combined["aime_greedy_transplant"], 6),
        "aime_pct_of_base": round(combined["aime_pct_of_base"], 4),
        "recovered_fraction": round(combined["recovered_fraction"], 4),
        "collapse_locus": combined["collapse_locus"],
        "cheap_fast_ship_exists": combined["cheap_fast_ship_exists"],
        "head_input_dim_compatible": True,
        "tied_head_is_base_head_byte_verified": True,
        "apples_to_apples": combined["apples_to_apples"],
        "wandb_run_id": run_id,
    }
    print("SENPAI-RESULT " + json.dumps(senpai), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
