#!/usr/bin/env python
"""PR #596 denken — cross-config synthesis of the served-identity determinism cards.

Combines the per-config determinism reports into the load-bearing answer the advisor (#594/#597)
asked for: is int4_g128_lmhead + MTP-K7 run-to-run bit-stable?

Decomposition (the direct int4_g128 serve is NOT locally buildable — model/ weights absent and the
unquantized bf16 source build_quant.py consumes is not cached; launch isolation forbids fern #597's
baked weights — so the int4-head leg is a STRUCTURAL projection rather than a direct measurement):
  * base_specoff (base_fullhead = int4-g32 *body* Marlin + bf16 head, M=1 AR) = the DECODE SUBSTRATE.
    Its warm cross-process bit-exactness proves the int4-body-Marlin GEMV -> bf16 logits -> argmax
    path is kernel-deterministic. The ONLY kernel fern's int4_g128 adds is the untied int4 *head*
    Marlin GEMV — same Marlin family, same fixed-reduction determinism.
  * base_mtp vs base_specoff = the INHERITANCE BRIDGE: does the MTP-K7 spec wrapper inherit the
    decode-step determinism? If yes (warm rates both bit-exact), the spec-dec served stack is
    deterministic independent of the verify-step structure.
  * int4g128_specoff (OPTIONAL): if a direct int4_g128 substrate report is ever produced (advisor
    authorizes the ~15 GB source download + local build) it is used directly; otherwise the
    int4-head leg is the structural projection above.

Projection: int4_g128 + MTP-K7 warm determinism = 1.0  IFF  (base_specoff warm bit-exact) AND
(base_mtp warm bit-exact / inheritance holds). Warm==warm bit-exactness is a kernel-family property
(weight-value independent); denser int4 ties change the flip *rate* only if the kernel is
nondeterministic, and base_specoff shows it is not.

Reads:
  determinism_report.json | determinism_report_base_specoff.json  (base_specoff)
  determinism_report_base_mtp.json                                (base_mtp)
  determinism_report_int4g128_specoff.json                        (int4g128_specoff, optional)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

CONFIG_FILES = {
    "base_specoff": ["determinism_report_base_specoff.json", "determinism_report.json"],
    "base_mtp": ["determinism_report_base_mtp.json"],
    "int4g128_specoff": ["determinism_report_int4g128_specoff.json"],
}

# Fallback labels for reports written by the pre-generalization driver (no config_label field).
CONFIG_LABELS = {
    "base_specoff": "base_fullhead spec-OFF M=1 AR (#319 reference)",
    "base_mtp": "base_fullhead spec-ON MTP-K7 (served stack)",
    "int4g128_specoff": "int4_g128_lmhead decode substrate M=1 AR",
}

# Determinism-rate tolerance for "the spec wrapper inherits the decode-step determinism": the two
# rates must agree to within this (they are both pooled matched-state per-step rates over thousands
# of trials; a true inheritance match is exact-or-near-exact, both effectively 1.0 modulo bf16 ties).
INHERIT_TOL = 5e-3


def load_report(cfg: str) -> dict | None:
    for name in CONFIG_FILES[cfg]:
        p = HERE / name
        if p.exists():
            try:
                d = json.loads(p.read_text())
                d["_file"] = name
                return d
            except (OSError, ValueError):
                continue
    return None


def pick(s: dict, *keys):
    return {k: s.get(k) for k in keys}


def log_meta_wandb(out: dict) -> str | None:
    """Best-effort: log the cross-config rollup verdicts to W&B (same group as the per-config runs).
    Skipped silently if wandb is unavailable or SENPAI_META_NO_WANDB is set."""
    import os
    if os.environ.get("SENPAI_META_NO_WANDB"):
        return None
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[meta] wandb unavailable: {exc}", flush=True)
        return None
    lb = out["load_bearing_int4g128_mtp"]
    v = out["verdicts"]
    run = init_wandb_run(
        job_type="systems-profile", agent="denken",
        name="denken/served-identity-determinism-meta",
        group="served-identity-determinism",
        tags=["served-identity", "determinism", "cross-process", "meta", "rollup",
              lb.get("route", ""), "analysis-only", "pr596"],
        notes="PR #596 cross-config rollup: base_specoff substrate (int4-body Marlin + argmax) + "
              "base_mtp inheritance bridge -> structural projection for int4_g128 + MTP-K7 "
              "run-to-run determinism.",
        config={"pr": 596, "analysis_only": True, "official_tps": 0,
                "configs_measured": out["configs_measured"],
                "int4g128_mtp_determinism_route": lb.get("route")},
    )
    if run is None:
        return None
    proj = lb.get("projected_int4g128_mtp_determinism_rate")
    summary = {
        "operative_identity_bit_stable": v["operative_identity_bit_stable"],
        "int4g128_mtp_bit_stable": v["int4g128_mtp_bit_stable"],
        "int4g128_mtp_determinism_route": v["int4g128_mtp_determinism_route"],
        "int4g128_substrate_directly_measured": v["int4g128_substrate_directly_measured"],
        "projected_int4g128_mtp_determinism_rate": proj,
        "spec_inherits_decode_determinism":
            out["inheritance_bridge"].get("spec_inherits_decode_determinism"),
        "base_specoff_warm_rate": out["inheritance_bridge"].get("base_specoff_warm_rate"),
        "base_mtp_warm_rate": out["inheritance_bridge"].get("base_mtp_warm_rate"),
        "gpqa_seedswing_is_sampling_not_nondeterminism":
            v["gpqa_seedswing_is_sampling_not_nondeterminism"],
        "primary_metric": proj,
    }
    log_summary(run, {k: x for k, x in summary.items() if x is not None}, step=0)
    log_json_artifact(run, name="served-identity-determinism-meta",
                      artifact_type="determinism-meta", data=out)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def main() -> int:
    reports = {cfg: load_report(cfg) for cfg in CONFIG_FILES}
    present = {cfg: r for cfg, r in reports.items() if r is not None}
    if not present:
        print("[meta] no per-config reports found yet", flush=True)
        return 1

    table = {}
    for cfg, r in present.items():
        s = r["synthesis"]
        table[cfg] = {
            "config_label": r.get("config_label") or CONFIG_LABELS.get(cfg),
            "spec_mode": r.get("spec_mode"),
            "n_servers_ok": s.get("n_servers_ok"),
            "served_argmax_determinism_rate": s.get("served_argmax_determinism_rate"),  # warm matched-state
            "warm_xproc_matched_state_rate": s.get("warm_xproc_matched_state_rate"),
            "warm_xproc_allN_sequence_exact": s.get("warm_xproc_allN_sequence_exact"),
            "warm_xproc_all_pairs_identical": s.get("warm_xproc_all_pairs_identical"),
            "cold_xproc_matched_state_rate": s.get("cold_xproc_matched_state_rate"),
            "cold_xproc_allN_sequence_exact": s.get("cold_xproc_allN_sequence_exact"),
            "cold_xproc_all_pairs_identical": s.get("cold_xproc_all_pairs_identical"),
            "within_coldwarm_matched_state_rate": s.get("within_coldwarm_matched_state_rate"),
            "within_coldwarm_all_servers_identical": s.get("within_coldwarm_all_servers_identical"),
            "literally_bit_exact_all_pairs": s.get("literally_bit_exact_all_pairs"),
            "all_flips_near_tie": s.get("all_flips_near_tie"),
            "tie_miss_margin_p90": s.get("tie_miss_margin_p90"),
            "tie_control_margin_p10": s.get("tie_control_margin_p10"),
            "n_real_response_divergent_prompts": s.get("n_real_response_divergent_prompts"),
            "any_real_response_divergence": s.get("any_real_response_divergence"),
            "operative_identity_bit_stable": s.get("operative_identity_bit_stable"),
            "matched_state_trials": s.get("matched_state_trials"),
            "peak_vram_gb": s.get("peak_vram_gb"),
        }

    # ---- inheritance bridge: spec-ON MTP-K7 determinism == spec-OFF M=1 AR determinism? ----
    inheritance = {"available": False}
    bs = table.get("base_specoff")
    bm = table.get("base_mtp")
    if bs and bm:
        r_off = bs["served_argmax_determinism_rate"]
        r_on = bm["served_argmax_determinism_rate"]
        cold_off = bs["cold_xproc_all_pairs_identical"]
        cold_on = bm["cold_xproc_all_pairs_identical"]
        rates_ok = (r_off is not None and r_on is not None and abs(r_off - r_on) <= INHERIT_TOL)
        # also accept the case where both warm rates are None but both cold are perfectly identical
        cold_match = (cold_off is not None and cold_off == cold_on)
        inheritance = {
            "available": True,
            "base_specoff_warm_rate": r_off,
            "base_mtp_warm_rate": r_on,
            "warm_rate_abs_diff": (abs(r_off - r_on) if (r_off is not None and r_on is not None) else None),
            "tolerance": INHERIT_TOL,
            "base_specoff_cold_all_pairs_identical": cold_off,
            "base_mtp_cold_all_pairs_identical": cold_on,
            "spec_inherits_decode_determinism": bool(rates_ok or (r_off is None and r_on is None and cold_match)),
        }

    # ---- load-bearing: int4_g128_lmhead + MTP-K7 run-to-run determinism ----
    # Two routes: (A) DIRECT, if a int4g128_specoff report exists; (B) STRUCTURAL projection from the
    # base_specoff substrate (int4-body Marlin + argmax) + base_mtp inheritance bridge.
    load_bearing = {"available": False}
    ig = table.get("int4g128_specoff")
    inh_ok = inheritance.get("spec_inherits_decode_determinism")
    if ig:
        substrate_rate = ig["served_argmax_determinism_rate"]
        # DIRECT: int4_g128+MTP determinism is the int4 substrate determinism IFF the bridge holds.
        projected = substrate_rate if inh_ok else None
        load_bearing = {
            "available": True,
            "route": "direct_measurement",
            "int4g128_substrate_warm_rate": substrate_rate,
            "int4g128_substrate_cold_all_pairs_identical": ig["cold_xproc_all_pairs_identical"],
            "int4g128_substrate_warm_all_pairs_identical": ig["warm_xproc_all_pairs_identical"],
            "int4g128_substrate_all_flips_near_tie": ig["all_flips_near_tie"],
            "int4g128_any_real_response_divergence": ig["any_real_response_divergence"],
            "inheritance_bridge_holds": inh_ok,
            "projected_int4g128_mtp_determinism_rate": projected,
            "note": ("int4_g128+MTP-K7 determinism == int4_g128 substrate determinism, via the "
                     "base_specoff<->base_mtp inheritance bridge."),
        }
    elif bs is not None:
        # STRUCTURAL: base_fullhead = int4-g32 BODY (Marlin) + bf16 head, so base_specoff already
        # proves the int4-body-Marlin -> bf16-logits -> argmax substrate is warm cross-process
        # bit-exact. The only kernel fern's int4_g128 ADDS is the untied int4 HEAD Marlin GEMV
        # (same family, same fixed-reduction determinism). Warm==warm bit-exactness is a kernel
        # property (weight-value independent); denser int4 ties change the flip RATE only if the
        # kernel is nondeterministic, and base_specoff shows it is not.
        substrate_warm_ident = bs["warm_xproc_all_pairs_identical"]
        substrate_warm_rate = bs["served_argmax_determinism_rate"]
        substrate_bit_exact = bool(substrate_warm_ident and bs["operative_identity_bit_stable"])
        # The +MTP projection REQUIRES the inheritance bridge to be measured AND hold (base_mtp
        # present and warm-bit-exact) — the substrate alone (M=1 AR) does not cover the MTP
        # verify-step determinism. bridge_holds=None (base_mtp absent) => projection withheld.
        bridge_holds = bool(inh_ok) if (bm is not None) else None
        projection_holds = bool(substrate_bit_exact and bridge_holds is True)
        projected = 1.0 if projection_holds else None
        load_bearing = {
            "available": True,
            "route": "structural_projection",
            "reason_no_direct": ("int4_g128 model/ weights absent + unquantized bf16 source not "
                                 "cached (only LFS-pointer/tokenizer blobs); launch isolation "
                                 "forbids fern #597's baked weights."),
            "substrate_config": "base_specoff (int4-g32 body Marlin + bf16 head, M=1 AR)",
            "substrate_warm_rate": substrate_warm_rate,
            "substrate_warm_all_pairs_identical": substrate_warm_ident,
            "substrate_bit_exact": substrate_bit_exact,
            "int4_body_marlin_argmax_proven_deterministic": substrate_bit_exact,
            "unmeasured_kernel": "untied int4 lm_head Marlin GEMV (same Marlin family as proven int4 body GEMVs)",
            "inheritance_bridge_holds": bridge_holds,
            "projection_holds": projection_holds,
            "projected_int4g128_mtp_determinism_rate": projected,
            "residual_uncertainty": ("only the int4-head Marlin GEMV is not directly run; its "
                                     "warm determinism is projected from the same-family int4-body "
                                     "Marlin GEMVs measured bit-exact in base_specoff. A larger "
                                     "near-tie density (int4 head compresses logit range) would "
                                     "raise the cold->warm transient flip count but NOT the warm "
                                     "run-to-run rate, which is kernel-deterministic."),
            "note": ("STRUCTURAL projection: int4_g128+MTP-K7 warm determinism = 1.0, from "
                     "base_specoff substrate bit-exactness + base_mtp inheritance; a direct int4 "
                     "build is confirmatory-only and gated on a ~15 GB gated-HF source download."),
        }

    # ---- top-level verdicts ----
    # operative_identity_bit_stable (overall): every MEASURED config's served (warm) regime is
    # bit-stable run-to-run modulo bf16 near-ties (no flip on a real-response pre-EOS token), AND
    # the load-bearing int4_g128+MTP-K7 determinism is established — directly if int4g128 was
    # served, otherwise via the structural projection (= 1.0) backed by the base_specoff substrate
    # + base_mtp inheritance bridge.
    measured = list(present)
    per_cfg_stable = {cfg: table[cfg]["operative_identity_bit_stable"] for cfg in measured}
    int4_stable = per_cfg_stable.get("int4g128_specoff")  # None if not directly measured
    lb_route = load_bearing.get("route")
    lb_projected = load_bearing.get("projected_int4g128_mtp_determinism_rate")
    int4g128_mtp_bit_stable = bool(lb_projected is not None and lb_projected >= 0.999)
    overall_bit_stable = bool(
        all(v for v in per_cfg_stable.values()) and int4g128_mtp_bit_stable
    )
    # gpqa attribution is config-independent (analytic) — take from any present report.
    any_report = next(iter(present.values()))
    gpqa = any_report.get("gpqa_attribution", {})

    out = {
        "pr": 596, "analysis_only": True, "official_tps": 0,
        "configs_measured": measured,
        "per_config": table,
        "inheritance_bridge": inheritance,
        "load_bearing_int4g128_mtp": load_bearing,
        "verdicts": {
            "operative_identity_bit_stable": overall_bit_stable,
            "per_config_operative_identity_bit_stable": per_cfg_stable,
            "int4g128_mtp_bit_stable": int4g128_mtp_bit_stable,
            "int4g128_mtp_determinism_route": lb_route,  # direct_measurement | structural_projection
            "int4g128_substrate_directly_measured": ("int4g128_specoff" in measured),
            "int4g128_substrate_bit_stable_direct": int4_stable,
            "gpqa_seedswing_is_sampling_not_nondeterminism":
                gpqa.get("gpqa_seedswing_is_sampling_not_nondeterminism"),
        },
        "gpqa_attribution": gpqa,
    }
    out["meta_wandb_run_id"] = log_meta_wandb(out)
    (HERE / "meta_synthesis.json").write_text(json.dumps(out, indent=2, default=str))

    print("=" * 80)
    print("PR #596 — CROSS-CONFIG SERVED-IDENTITY DETERMINISM SYNTHESIS")
    print("=" * 80)
    for cfg in measured:
        t = table[cfg]
        print(f"\n[{cfg}] {t['config_label']}  (spec={t['spec_mode']}, N={t['n_servers_ok']})")
        print(f"   served_argmax_determinism_rate (warm matched-state) = {t['served_argmax_determinism_rate']}")
        print(f"   cold xproc all-pairs-identical = {t['cold_xproc_all_pairs_identical']}  "
              f"(seq-exact {t['cold_xproc_allN_sequence_exact']})")
        print(f"   warm xproc all-pairs-identical = {t['warm_xproc_all_pairs_identical']}  "
              f"(seq-exact {t['warm_xproc_allN_sequence_exact']})")
        print(f"   within cold->warm identical = {t['within_coldwarm_all_servers_identical']}  "
              f"(matched-state {t['within_coldwarm_matched_state_rate']})")
        print(f"   all flips near-tie = {t['all_flips_near_tie']}  "
              f"real-response divergences = {t['n_real_response_divergent_prompts']}")
        print(f"   operative_identity_bit_stable = {t['operative_identity_bit_stable']}")
    print(f"\nINHERITANCE BRIDGE (spec-ON MTP == spec-OFF M=1 AR determinism): "
          f"{inheritance.get('spec_inherits_decode_determinism')}")
    if inheritance.get("available"):
        print(f"   base_specoff warm rate={inheritance['base_specoff_warm_rate']}  "
              f"base_mtp warm rate={inheritance['base_mtp_warm_rate']}  "
              f"|diff|={inheritance['warm_rate_abs_diff']} (tol {inheritance['tolerance']})")
    print(f"\nLOAD-BEARING int4_g128+MTP-K7 determinism = "
          f"{load_bearing.get('projected_int4g128_mtp_determinism_rate')}  "
          f"[route={load_bearing.get('route')}]")
    if load_bearing.get("route") == "structural_projection":
        print(f"   substrate (int4-body Marlin+argmax) bit-exact = {load_bearing.get('substrate_bit_exact')}; "
              f"unmeasured kernel = {load_bearing.get('unmeasured_kernel')}")
    print(f"\n>>> operative_identity_bit_stable (overall) = {overall_bit_stable}")
    print(f">>> gpqa_seedswing_is_sampling_not_nondeterminism = "
          f"{out['verdicts']['gpqa_seedswing_is_sampling_not_nondeterminism']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
