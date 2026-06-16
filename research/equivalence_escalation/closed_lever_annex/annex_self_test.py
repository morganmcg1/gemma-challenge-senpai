"""Closed-Lever Evidence Annex (PR #456) — 0-GPU self-test + W&B logger.

Validates the machine-readable ledger (`closed_lever_ledger.json`) against the
human-readable annex (`closed_lever_evidence_annex.md`) so the two cannot drift,
then logs a W&B run with the PR-required fields.

Analysis-only: no GPU, no model load, no HF job, no submission, no served-file
change. Greedy/PPL untouched by construction (this is a synthesis card).

Run under the repo .venv (it has wandb):

  cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    research/equivalence_escalation/closed_lever_annex/annex_self_test.py \
    --wandb_group equivalence-escalation-anchors \
    --wandb_name denken/closed-lever-evidence-annex

  # 0-GPU gate only (no network): add --no-wandb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEDGER = HERE / "closed_lever_ledger.json"
ANNEX_MD = HERE / "closed_lever_evidence_annex.md"

ALLOWED_REASON = {"physics", "measurement", "build_no_go"}
ALLOWED_AXES = {"supply", "demand", "fresh_literature"}
RUN_ID_RE = re.compile(r"^[a-z0-9]{8}$")

# The 18 distinct closing/anchor runs confirmed live+finished against the W&B
# public API while authoring this annex (denken, 2026-06-16).
VERIFIED_LIVE_RUNS = {
    "2x9fm2zx", "5a6zq2yz", "nvsbctji", "c5oyb7gv", "0pg4bz25", "hv4xpgf8",
    "5f3e91as", "fn4iz0dz", "xryqregh", "crrq2e1y", "7rb089z3", "qlvakiyu",
    "0syyqxag", "emljqube", "uid28gdg", "c675zor8", "51bdsbpw", "e5n9a2dc",
}


def _collect_run_ids(ledger: dict) -> set[str]:
    ids: set[str] = set()
    for lev in ledger["closed_levers"]:
        rid = lev.get("closing_run_id")
        if rid:
            ids.add(rid)
    for a in ledger["anchors"].values():
        if isinstance(a, dict) and a.get("run_id"):
            ids.add(a["run_id"])
    for rid in ledger["reconciliation"]["anchor_runs"].values():
        if rid:
            ids.add(rid)
    for row in ledger["isolation_collapses"]["rows"]:
        rid = row.get("run_id")
        if rid:
            # isolation rows may carry composite ids like "crrq2e1y"
            for piece in re.split(r"[/, ]+", rid):
                if RUN_ID_RE.match(piece):
                    ids.add(piece)
    return ids


def run_self_test(ledger: dict, md_text: str) -> tuple[dict[str, bool], dict]:
    levers = ledger["closed_levers"]
    count = len(levers)
    recon = ledger["reconciliation"]
    anchors = ledger["anchors"]

    run_ids = _collect_run_ids(ledger)
    md_links_every_run = all(rid in md_text for rid in run_ids)
    all_run_ids_syntactic = all(RUN_ID_RE.match(rid) for rid in run_ids)
    all_run_ids_verified_live = run_ids.issubset(VERIFIED_LIVE_RUNS)

    axes_present = {lev["axis"] for lev in levers}
    reasons_ok = all(lev["reason_class"] in ALLOWED_REASON for lev in levers)
    required_fields = {
        "id", "axis", "lever", "mechanism", "closing_pr",
        "closing_run_id", "closing_number_text", "reason_class", "reason",
    }
    fields_ok = all(required_fields.issubset(lev.keys()) for lev in levers)
    ids_unique = len({lev["id"] for lev in levers}) == count

    dep = anchors["deployed_non_equivalent"]["tps"]
    realized = anchors["realized_blanket_strict_frontier"]["tps"]
    wall = anchors["verify_bw_lambda1_wall"]["tps"]
    roof = anchors["roofline_perfect_retile_ceiling"]["tps"]
    gap = anchors["gap_realized_to_deployed_tps"]

    headroom_pct = recon["headroom_pct_vs_read_peak"]
    headroom_check = abs(
        headroom_pct - (1.0 - recon["int4_gemm_achieved_bw_frac_of_read_peak"]) * 100.0
    ) < 0.6

    iso_rows = ledger["isolation_collapses"]["rows"]

    def _finite(x) -> bool:
        return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")

    numeric_anchor_vals = [dep, realized, wall, roof, gap, headroom_pct]

    checks = {
        "ledger_loads": True,
        "count_is_20": count == 20,
        "count_matches_levers": count == len(levers),
        "md_states_count_20": "closed_lever_count = 20" in md_text,
        "lever_ids_unique": ids_unique,
        "all_required_fields": fields_ok,
        "reason_classes_valid": reasons_ok,
        "all_three_axes_present": axes_present == ALLOWED_AXES,
        "supply_present": "supply" in axes_present,
        "demand_present": "demand" in axes_present,
        "fresh_literature_present": "fresh_literature" in axes_present,
        "run_ids_syntactic": all_run_ids_syntactic,
        "every_run_id_linked_in_md": md_links_every_run,
        "run_ids_verified_live": all_run_ids_verified_live,
        "strict_headroom_is_greedy_unsafe": recon["strict_headroom_is_greedy_unsafe"] is True,
        "recon_447_in_md": recon["anchor_runs"]["447_verify_wall"] in md_text,
        "recon_450_in_md": recon["anchor_runs"]["450_roofline"] in md_text,
        "recon_451_in_md": recon["anchor_runs"]["451_bigger_drafter"] in md_text,
        "anchor_order_realized_lt_deployed": realized < dep,
        "anchor_order_deployed_lt_roofline": dep < roof,
        "anchor_order_roofline_le_wall": roof <= wall,
        "gap_consistent": abs(gap - round(dep - realized, 2)) < 0.011,
        "headroom_pct_consistent": headroom_check,
        "roofline_clears_deployed": roof > dep,
        "realistic_splitk_below_roofline": recon["realistic_splitk_delta_hi_tps"] < (roof - realized) + 1e-6,
        "isolation_has_4_rows": len(iso_rows) == 4,
        "isolation_runs_in_md": all(
            all(p in md_text for p in re.split(r"[/, ]+", r["run_id"]) if RUN_ID_RE.match(p))
            for r in iso_rows
        ),
        "ppl_is_anchor": ledger["ppl"] == 2.3772,
        "ppl_within_gate": ledger["ppl"] <= ledger["ppl_gate"],
        "analysis_only": ledger["analysis_only"] is True,
        "no_served_file_change": ledger["no_served_file_change"] is True,
        "official_tps_zero": ledger["official_tps"] == 0,
        "verdict_frontier_closed": ledger["verdict"]["strict_frontier_closed"] is True,
        "verdict_both_axes": set(ledger["verdict"]["both_axes_closed"]) == {"supply", "demand"},
        "no_nan_inf": all(_finite(v) for v in numeric_anchor_vals),
    }

    derived = {
        "closed_lever_count": count,
        "all_runs_linked": bool(md_links_every_run and all_run_ids_syntactic),
        "strict_headroom_is_greedy_unsafe": bool(recon["strict_headroom_is_greedy_unsafe"]),
        "n_distinct_runs": len(run_ids),
        "n_supply": sum(1 for lev in levers if lev["axis"] == "supply"),
        "n_demand": sum(1 for lev in levers if lev["axis"] == "demand"),
        "n_fresh_literature": sum(1 for lev in levers if lev["axis"] == "fresh_literature"),
        "best_realized_byte_exact_delta_tps": ledger["verdict"]["best_realized_byte_exact_delta_tps"],
        "gap_realized_to_deployed_tps": gap,
    }
    return checks, derived


def maybe_log_wandb(ledger: dict, checks: dict, derived: dict, args) -> str | None:
    if args.no_wandb:
        return None
    sys.path.append(str(ROOT))  # installed wandb beats repo-root ./wandb dir
    try:
        import wandb
        from scripts.wandb_logging import init_wandb_run
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: wandb unavailable ({exc}); skipping log", file=sys.stderr)
        return None

    all_pass = all(checks.values())
    config = {
        "pr": 456,
        "track": "closed-lever-evidence-annex",
        "analysis_only": True,
        "no_hf_job": True,
        "no_submission": True,
        "no_served_file_change": True,
        "local_only": True,
        "for_decision": ledger["for_decision"],
        "closed_lever_count": derived["closed_lever_count"],
        "n_supply": derived["n_supply"],
        "n_demand": derived["n_demand"],
        "n_fresh_literature": derived["n_fresh_literature"],
        "anchors": ledger["anchors"],
        "reconciliation": ledger["reconciliation"],
        "isolation_collapses": ledger["isolation_collapses"],
        "runs_verified_live": sorted(VERIFIED_LIVE_RUNS),
    }
    run = init_wandb_run(
        job_type="evidence-annex",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="PR#456: Closed-lever evidence annex for the relax-strict-equivalence "
              "decision (#407). Every strict TPS lever + the run that closed it + the "
              "physics/measurement. The ~16% int4-GEMM BW headroom is REAL but greedy-unsafe.",
        tags=["pr-456", "evidence-annex", "equivalence-escalation", "closed-levers",
              "analysis-only", "no-served-change", "byte-exact"],
        config=config,
    )
    if run is None:
        print("ERROR: init_wandb_run returned None (WANDB disabled or no API key)", file=sys.stderr)
        return None

    scalars = {
        "primary/closed_lever_count": derived["closed_lever_count"],
        "annex/all_runs_linked": float(derived["all_runs_linked"]),
        "annex/strict_headroom_is_greedy_unsafe": float(derived["strict_headroom_is_greedy_unsafe"]),
        "annex/annex_self_test_passes": float(all_pass),
        "annex/n_distinct_runs": derived["n_distinct_runs"],
        "annex/n_supply": derived["n_supply"],
        "annex/n_demand": derived["n_demand"],
        "annex/n_fresh_literature": derived["n_fresh_literature"],
        "annex/best_realized_byte_exact_delta_tps": derived["best_realized_byte_exact_delta_tps"],
        "annex/gap_realized_to_deployed_tps": derived["gap_realized_to_deployed_tps"],
        "annex/analysis_only": 1.0,
        "annex/no_served_file_change": 1.0,
        "annex/official_tps": 0.0,
        "test/ppl": ledger["ppl"],
        "test/ppl_gate": ledger["ppl_gate"],
        "test/ppl_ok": float(ledger["ppl"] <= ledger["ppl_gate"]),
        "global_step": 0,
    }
    # anchor TPS values
    scalars["anchor/deployed_tps"] = ledger["anchors"]["deployed_non_equivalent"]["tps"]
    scalars["anchor/realized_frontier_tps"] = ledger["anchors"]["realized_blanket_strict_frontier"]["tps"]
    scalars["anchor/verify_bw_wall_tps"] = ledger["anchors"]["verify_bw_lambda1_wall"]["tps"]
    scalars["anchor/roofline_ceiling_tps"] = ledger["anchors"]["roofline_perfect_retile_ceiling"]["tps"]
    # reconciliation numerics
    for k in ("int4_gemm_achieved_bw_gbps", "read_peak_gbps", "headroom_pct_vs_read_peak",
              "perfect_retile_ceiling_tps", "realistic_splitk_delta_lo_tps",
              "realistic_splitk_delta_hi_tps"):
        scalars[f"recon/{k}"] = ledger["reconciliation"][k]
    for k, v in checks.items():
        scalars[f"selftest/{k}"] = float(bool(v))
    run.log(scalars)

    # ---- ledger table ----
    led_tbl = wandb.Table(columns=["axis", "lever", "closing_pr", "closing_run_id",
                                    "closing_number_text", "reason_class"])
    for lev in ledger["closed_levers"]:
        led_tbl.add_data(lev["axis"], lev["lever"], str(lev["closing_pr"]),
                         lev["closing_run_id"] or "", lev["closing_number_text"],
                         lev["reason_class"])
    iso_tbl = wandb.Table(columns=["lever", "pr", "run_id", "modeled", "realized"])
    for r in ledger["isolation_collapses"]["rows"]:
        iso_tbl.add_data(r["lever"], str(r["pr"]), r["run_id"], r["modeled"], r["realized"])
    run.log({"closed_lever_ledger": led_tbl, "isolation_collapses": iso_tbl, "global_step": 0})

    # ---- sticky summary ----
    run.summary["closed_lever_count"] = derived["closed_lever_count"]
    run.summary["all_runs_linked"] = derived["all_runs_linked"]
    run.summary["strict_headroom_is_greedy_unsafe"] = derived["strict_headroom_is_greedy_unsafe"]
    run.summary["annex_self_test_passes"] = all_pass
    run.summary["analysis_only"] = True
    run.summary["no_served_file_change"] = True
    run.summary["official_tps"] = 0
    run.summary["ppl"] = ledger["ppl"]
    run.summary["verdict"] = ledger["verdict"]["escalation_target"]

    art = wandb.Artifact("closed_lever_evidence_annex", type="evidence-annex",
                         metadata={"pr": 456, "closed_lever_count": derived["closed_lever_count"],
                                   "annex_self_test_passes": all_pass})
    art.add_file(str(LEDGER))
    art.add_file(str(ANNEX_MD))
    run.log_artifact(art)

    rid = run.id
    print(json.dumps({"run_id": rid, "url": run.get_url()}, indent=2))
    run.finish()
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="denken/closed-lever-evidence-annex")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    ledger = json.loads(LEDGER.read_text())
    md_text = ANNEX_MD.read_text()
    checks, derived = run_self_test(ledger, md_text)

    all_pass = all(checks.values())
    n_pass = sum(1 for v in checks.values() if v)
    n_total = len(checks)
    print(f"SELF-TEST {n_pass}/{n_total} " + ("PASS" if all_pass else "FAIL"))
    for name, ok in checks.items():
        if not ok:
            print(f"  FAIL: {name}")
    print(json.dumps(derived, indent=2))

    rid = maybe_log_wandb(ledger, checks, derived, args)
    if rid:
        print(f"wandb_run_id={rid}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
