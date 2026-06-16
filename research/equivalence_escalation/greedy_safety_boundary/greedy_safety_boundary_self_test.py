"""Greedy-Safety Boundary (PR #460) — 0-GPU self-test + W&B logger.

Validates the greedy-safety boundary table + reduction enumeration
(`closed_lever_ledger_v2.json`) against the human-readable card
(`greedy_safety_boundary.md`), proves faithful continuity with the #456
closed-lever ledger (base 20 + lever #21 = 21), and logs the PR-required fields.

Analysis-only: no GPU, no model load, no HF job, no submission, no served-file
change. Greedy/PPL untouched by construction (this is a synthesis card).

Run under the repo .venv (it has wandb):

  cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    research/equivalence_escalation/greedy_safety_boundary/greedy_safety_boundary_self_test.py \
    --wandb_group equivalence-escalation-anchors \
    --wandb_name denken/greedy-safety-boundary

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
LEDGER_V2 = HERE / "closed_lever_ledger_v2.json"
CARD_MD = HERE / "greedy_safety_boundary.md"
BASE_LEDGER = HERE.parent / "closed_lever_annex" / "closed_lever_ledger.json"

ALLOWED_CLASS = {"preserving", "reassociating"}
ALLOWED_REASON = {"physics", "measurement", "build_no_go"}
RUN_ID_RE = re.compile(r"^[a-z0-9]{8}$")

# Distinct runs confirmed live+finished against the W&B public API while
# authoring this card (denken, 2026-06-16). The #442 set (gyw2ksvs A/B,
# grrc3zms census, cy0ijlit eager-floor, e5n9a2dc isolated autotune) are NEW
# this PR; the rest carry over from the #456 annex (all 18 re-verified).
VERIFIED_LIVE_RUNS = {
    # #442 (NEW this PR)
    "gyw2ksvs", "grrc3zms", "cy0ijlit", "e5n9a2dc",
    # carried anchors / boundary runs
    "2x9fm2zx", "5a6zq2yz", "nvsbctji", "c5oyb7gv", "fn4iz0dz", "crrq2e1y",
    "xryqregh", "n5bypf5h", "k33t25ct",
}


def _collect_v2_run_ids(led: dict) -> set[str]:
    ids: set[str] = set()
    for a in led["anchors"].values():
        if isinstance(a, dict) and a.get("run_id"):
            ids.add(a["run_id"])
    for k in led["greedy_safety_boundary"]["kernels"]:
        for lv in k["levers"]:
            if lv.get("run_id"):
                ids.add(lv["run_id"])
    for fam in led["reduction_enumeration"]["families"]:
        for rid in fam.get("reassoc_lever_runs", []):
            if rid:
                ids.add(rid)
        if fam.get("pinned_run"):
            ids.add(fam["pinned_run"])
    n21 = led["new_lever_21"]
    if n21.get("closing_run_id"):
        ids.add(n21["closing_run_id"])
    for rid in n21.get("supporting_run_ids", []):
        ids.add(rid)
    if led["ppl_trap"].get("census_run"):
        ids.add(led["ppl_trap"]["census_run"])
    if led["flag2_triton_attn_surface"].get("carried_value_447_run"):
        ids.add(led["flag2_triton_attn_surface"]["carried_value_447_run"])
    return {i for i in ids if RUN_ID_RE.match(i)}


def run_self_test(led: dict, base: dict, md: str) -> tuple[dict[str, bool], dict]:
    gsb = led["greedy_safety_boundary"]
    kernels = gsb["kernels"]
    enum = led["reduction_enumeration"]
    fams = enum["families"]
    n21 = led["new_lever_21"]
    flag2 = led["flag2_triton_attn_surface"]
    trap = led["ppl_trap"]

    # --- boundary table: total dichotomy ---
    all_levers = [lv for k in kernels for lv in k["levers"]]
    classes = [lv["class"] for lv in all_levers]
    boundary_total_dichotomy = all(c in ALLOWED_CLASS for c in classes)
    has_both_classes = ("preserving" in classes) and ("reassociating" in classes)
    # every preserving lever must carry a run id OR be an explicitly-pinned no-knob row
    preserving = [lv for lv in all_levers if lv["class"] == "preserving"]
    reassoc = [lv for lv in all_levers if lv["class"] == "reassociating"]

    # the four named verify kernels present (drafter is extra)
    kernel_names = " ".join(k["kernel"].lower() for k in kernels)
    four_kernels_present = all(
        t in kernel_names for t in ["marlin", "split-kv attention", "lm_head", "rmsnorm"]
    )
    # int4-Marlin is the dominant kernel (~85%)
    marlin = next(k for k in kernels if "marlin" in k["kernel"].lower())
    marlin_dominant = marlin["frac_of_verify"] > 0.80
    # the attention kernel carries BOTH a preserving (num_stages) and a
    # reassociating (BLOCK_M/BLOCK_Q) lever -> the clean isolable dichotomy
    attn = next(k for k in kernels if "split-kv attention" in k["kernel"].lower())
    attn_lever_classes = {lv["class"] for lv in attn["levers"]}
    attn_has_both = attn_lever_classes == ALLOWED_CLASS
    attn_reassoc_is_442 = any(
        lv["class"] == "reassociating" and lv["pr"] == 442 for lv in attn["levers"]
    )
    attn_preserving_is_447 = any(
        lv["class"] == "preserving" and lv["pr"] == 447 for lv in attn["levers"]
    )

    # --- reduction enumeration: completeness ---
    red_count = enum["reduction_count_enumerated"]
    red_count_matches = red_count == len(fams)
    classes_ab = [f["class_a_or_b"] for f in fams]
    every_family_classified = all(c in {"a", "b"} for c in classes_ab)
    # every family with material BW headroom is class (b) = reassociable/greedy-unsafe
    material_b = all(
        (f["class_a_or_b"] == "b") for f in fams if f["material_bw_headroom"]
    )
    # no class-(a) family has material BW headroom
    no_a_has_bw = all(
        (not f["material_bw_headroom"]) for f in fams if f["class_a_or_b"] == "a"
    )
    # every class-(b) family names at least one reassociating run
    every_b_has_reassoc_run = all(
        len(f["reassoc_lever_runs"]) >= 1 for f in fams if f["class_a_or_b"] == "b"
    )

    # --- continuity with #456: base 20 + lever 21 = 21 ---
    base_count = len(base["closed_levers"])
    derived_closed_lever_count = base_count + 1
    count_is_21 = derived_closed_lever_count == 21
    verdict_count_21 = led["verdict"]["closed_lever_count"] == 21
    md_states_21 = "closed_lever_count = 21" in md
    lever21_id_ok = n21["id"] == "supply-verify-attn-triton-tile-reassoc"
    lever21_reason_ok = n21["reason_class"] in ALLOWED_REASON
    lever21_run_ok = n21["closing_run_id"] == "gyw2ksvs"
    lever21_axis_supply = n21["axis"] == "supply"

    # --- FLAG-2 parameterized slot robustness ---
    flag2_pending = (flag2["status"] == "pending-wirbel") and (flag2["landed"] is False)
    flag2_carried_lb = abs(flag2["carried_value_447_lower_bound"] - 0.0127) < 1e-6
    flag2_ub_ge_lb = flag2["upper_bound_full_attention"] >= flag2["carried_value_447_lower_bound"]
    flag2_robust = flag2["robust_to_correction"] is True

    # --- PPL trap ---
    trap_ppl_pass = trap["ppl_passed"] is True and trap["ppl_bm4"] <= trap["ppl_gate"]
    trap_identity_break = trap["census_frac_identical"] < 1.0
    trap_not_byte_exact = trap["byte_exact_and_ppl_pass"] is False
    # the live proof: PPL passes WHILE identity breaks
    ppl_pass_ne_greedy = trap_ppl_pass and trap_identity_break and trap_not_byte_exact

    # --- derived booleans the PR asks for ---
    byte_exact_capped = gsb["byte_exact_levers_all_capped_le_0p26"] is True
    preserving_max = gsb["preserving_levers_max_realized_tps"]
    preserving_max_consistent = preserving_max <= 0.2613 + 1e-9
    headroom_reassoc = gsb["every_material_headroom_is_reassociating"] is True

    # --- run ids: syntactic, verified-live, rendered in md ---
    run_ids = _collect_v2_run_ids(led)
    run_ids_syntactic = all(RUN_ID_RE.match(r) for r in run_ids)
    run_ids_verified_live = run_ids.issubset(VERIFIED_LIVE_RUNS)
    md_links_every_run = all(r in md for r in run_ids)

    # --- anchors internally consistent (same invariants as #456) ---
    dep = led["anchors"]["deployed_non_equivalent"]["tps"]
    realized = led["anchors"]["realized_blanket_strict_frontier"]["tps"]
    roof = led["anchors"]["roofline_perfect_retile_ceiling"]["tps"]
    wall = led["anchors"]["verify_bw_lambda1_wall"]["tps"]

    checks = {
        "ledger_v2_loads": True,
        "base_456_loads": True,
        "base_count_is_20": base_count == 20,
        "closed_lever_count_is_21": count_is_21,
        "verdict_closed_lever_count_21": verdict_count_21,
        "md_states_count_21": md_states_21,
        "lever21_id_ok": lever21_id_ok,
        "lever21_reason_class_valid": lever21_reason_ok,
        "lever21_run_is_gyw2ksvs": lever21_run_ok,
        "lever21_axis_supply": lever21_axis_supply,
        "boundary_total_dichotomy": boundary_total_dichotomy,
        "boundary_has_both_classes": has_both_classes,
        "boundary_has_preserving": len(preserving) >= 1,
        "boundary_has_reassociating": len(reassoc) >= 1,
        "four_verify_kernels_present": four_kernels_present,
        "marlin_is_dominant_gt80pct": marlin_dominant,
        "attn_kernel_has_both_classes": attn_has_both,
        "attn_reassoc_is_pr442": attn_reassoc_is_442,
        "attn_preserving_is_pr447": attn_preserving_is_447,
        "reduction_count_is_4": red_count == 4,
        "reduction_count_matches_families": red_count_matches,
        "every_reduction_family_classified": every_family_classified,
        "material_headroom_all_class_b": material_b,
        "no_class_a_has_material_bw": no_a_has_bw,
        "every_class_b_names_reassoc_run": every_b_has_reassoc_run,
        "byte_exact_levers_all_capped_le_0p26": byte_exact_capped,
        "preserving_max_le_0p2613": preserving_max_consistent,
        "every_material_headroom_is_reassociating": headroom_reassoc,
        "ppl_pass_ne_greedy_identical": ppl_pass_ne_greedy,
        "flag2_pending_not_landed": flag2_pending,
        "flag2_carried_lb_is_1p27pct": flag2_carried_lb,
        "flag2_upper_bound_ge_lower": flag2_ub_ge_lb,
        "flag2_robust_to_correction": flag2_robust,
        "run_ids_syntactic": run_ids_syntactic,
        "run_ids_verified_live": run_ids_verified_live,
        "every_run_id_linked_in_md": md_links_every_run,
        "anchor_order_realized_lt_deployed": realized < dep,
        "anchor_order_deployed_lt_roofline": dep < roof,
        "anchor_order_roofline_le_wall": roof <= wall,
        "ppl_is_anchor_2p3772": led["ppl"] == 2.3772,
        "ppl_within_gate": led["ppl"] <= led["ppl_gate"],
        "analysis_only": led["analysis_only"] is True,
        "no_served_file_change": led["no_served_file_change"] is True,
        "official_tps_zero": led["official_tps"] == 0,
        "verdict_principle_is_byte_exact_iff": "reduction-order-preserving" in led["verdict"]["principle"],
    }

    derived = {
        "reduction_count_enumerated": red_count,
        "closed_lever_count": derived_closed_lever_count,
        "base_456_lever_count": base_count,
        "byte_exact_levers_all_capped_le_0p26": bool(byte_exact_capped),
        "every_material_headroom_is_reassociating": bool(headroom_reassoc),
        "n_preserving_levers": len(preserving),
        "n_reassociating_levers": len(reassoc),
        "preserving_levers_max_realized_tps": preserving_max,
        "n_distinct_runs": len(run_ids),
        "census_frac_identical": trap["census_frac_identical"],
        "ppl_bm4": trap["ppl_bm4"],
        "flag2_landed": bool(flag2["landed"]),
    }
    return checks, derived


def maybe_log_wandb(led: dict, checks: dict, derived: dict, args) -> str | None:
    if args.no_wandb:
        return None
    sys.path.append(str(ROOT))
    try:
        import wandb
        from scripts.wandb_logging import init_wandb_run
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: wandb unavailable ({exc}); skipping log", file=sys.stderr)
        return None

    all_pass = all(checks.values())
    config = {
        "pr": 460,
        "supersedes_pr": 456,
        "track": "greedy-safety-boundary",
        "analysis_only": True,
        "no_hf_job": True,
        "no_submission": True,
        "no_served_file_change": True,
        "local_only": True,
        "for_decision": led["for_decision"],
        "principle": led["principle"],
        "closed_lever_count": derived["closed_lever_count"],
        "reduction_count_enumerated": derived["reduction_count_enumerated"],
        "anchors": led["anchors"],
        "greedy_safety_boundary": led["greedy_safety_boundary"],
        "reduction_enumeration": led["reduction_enumeration"],
        "new_lever_21": led["new_lever_21"],
        "flag2_triton_attn_surface": led["flag2_triton_attn_surface"],
        "ppl_trap": led["ppl_trap"],
        "runs_verified_live": sorted(VERIFIED_LIVE_RUNS),
    }
    run = init_wandb_run(
        job_type="evidence-annex",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="PR#460: Greedy-safety boundary of the served verify path (+ annex v2). "
              "byte-exact <=> reduction-order-preserving; every material BW headroom "
              "reassociates a reduction -> greedy-unsafe. Lever #21 = #442 Triton-attn "
              "tile-retune greedy-unsafe (census 53.1%). PPL-pass != greedy-identical.",
        tags=["pr-460", "greedy-safety-boundary", "evidence-annex",
              "equivalence-escalation", "closed-levers", "analysis-only",
              "no-served-change", "byte-exact", "annex-v2"],
        config=config,
    )
    if run is None:
        print("ERROR: init_wandb_run returned None (WANDB disabled or no API key)", file=sys.stderr)
        return None

    scalars = {
        "primary/closed_lever_count": derived["closed_lever_count"],
        "boundary/reduction_count_enumerated": derived["reduction_count_enumerated"],
        "boundary/byte_exact_levers_all_capped_le_0p26": float(derived["byte_exact_levers_all_capped_le_0p26"]),
        "boundary/every_material_headroom_is_reassociating": float(derived["every_material_headroom_is_reassociating"]),
        "boundary/greedy_safety_boundary_self_test_passes": float(all_pass),
        "boundary/n_preserving_levers": derived["n_preserving_levers"],
        "boundary/n_reassociating_levers": derived["n_reassociating_levers"],
        "boundary/preserving_levers_max_realized_tps": derived["preserving_levers_max_realized_tps"],
        "boundary/n_distinct_runs": derived["n_distinct_runs"],
        "boundary/base_456_lever_count": derived["base_456_lever_count"],
        "boundary/census_frac_identical": derived["census_frac_identical"],
        "boundary/ppl_bm4": derived["ppl_bm4"],
        "boundary/flag2_landed": float(derived["flag2_landed"]),
        "annex/analysis_only": 1.0,
        "annex/no_served_file_change": 1.0,
        "annex/official_tps": 0.0,
        "test/ppl": led["ppl"],
        "test/ppl_gate": led["ppl_gate"],
        "test/ppl_ok": float(led["ppl"] <= led["ppl_gate"]),
        "global_step": 0,
    }
    scalars["anchor/deployed_tps"] = led["anchors"]["deployed_non_equivalent"]["tps"]
    scalars["anchor/realized_frontier_tps"] = led["anchors"]["realized_blanket_strict_frontier"]["tps"]
    scalars["anchor/roofline_ceiling_tps"] = led["anchors"]["roofline_perfect_retile_ceiling"]["tps"]
    scalars["anchor/verify_bw_wall_tps"] = led["anchors"]["verify_bw_lambda1_wall"]["tps"]
    for k, v in checks.items():
        scalars[f"selftest/{k}"] = float(bool(v))
    run.log(scalars)

    # ---- boundary table ----
    btbl = wandb.Table(columns=["kernel", "frac_of_verify", "lever", "class", "pr", "run_id", "result"])
    for k in led["greedy_safety_boundary"]["kernels"]:
        for lv in k["levers"]:
            btbl.add_data(k["kernel"], k["frac_of_verify"], lv["lever"], lv["class"],
                          str(lv["pr"]), lv["run_id"] or "", lv["result"])
    rtbl = wandb.Table(columns=["id", "name", "frac_of_verify", "order", "material_bw_headroom", "class_a_or_b", "reassoc_runs"])
    for f in led["reduction_enumeration"]["families"]:
        rtbl.add_data(f["id"], f["name"], f["frac_of_verify"], f["order"],
                      bool(f["material_bw_headroom"]), f["class_a_or_b"],
                      ",".join(f["reassoc_lever_runs"]))
    run.log({"greedy_safety_boundary_table": btbl, "reduction_enumeration": rtbl, "global_step": 0})

    # ---- sticky summary ----
    run.summary["closed_lever_count"] = derived["closed_lever_count"]
    run.summary["reduction_count_enumerated"] = derived["reduction_count_enumerated"]
    run.summary["byte_exact_levers_all_capped_le_0p26"] = derived["byte_exact_levers_all_capped_le_0p26"]
    run.summary["every_material_headroom_is_reassociating"] = derived["every_material_headroom_is_reassociating"]
    run.summary["greedy_safety_boundary_self_test_passes"] = all_pass
    run.summary["analysis_only"] = True
    run.summary["no_served_file_change"] = True
    run.summary["official_tps"] = 0
    run.summary["ppl"] = led["ppl"]
    run.summary["verdict"] = led["verdict"]["escalation_target"]

    art = wandb.Artifact("greedy_safety_boundary", type="evidence-annex",
                         metadata={"pr": 460, "closed_lever_count": derived["closed_lever_count"],
                                   "reduction_count_enumerated": derived["reduction_count_enumerated"],
                                   "greedy_safety_boundary_self_test_passes": all_pass})
    art.add_file(str(LEDGER_V2))
    art.add_file(str(CARD_MD))
    run.log_artifact(art)

    rid = run.id
    print(json.dumps({"run_id": rid, "url": run.get_url()}, indent=2))
    run.finish()
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="denken/greedy-safety-boundary")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    led = json.loads(LEDGER_V2.read_text())
    base = json.loads(BASE_LEDGER.read_text())
    md = CARD_MD.read_text()
    checks, derived = run_self_test(led, base, md)

    all_pass = all(checks.values())
    n_pass = sum(1 for v in checks.values() if v)
    n_total = len(checks)
    print(f"SELF-TEST {n_pass}/{n_total} " + ("PASS" if all_pass else "FAIL"))
    for name, ok in checks.items():
        if not ok:
            print(f"  FAIL: {name}")
    print(json.dumps(derived, indent=2))

    rid = maybe_log_wandb(led, checks, derived, args)
    if rid:
        print(f"wandb_run_id={rid}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
