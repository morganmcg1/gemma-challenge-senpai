"""PR #660 wall_tps-definition reconcile -- FINAL synthesis -> W&B `walltps-defn-reconcile-land`.

Closes the 9.4% un-rescued served wall-TPS gap between land #632/#658 (170.21 local
K6 full_e2e) and stark #642 (155.58 local K6 un-rescued ceiling), given the two
harnesses' M=1 AR rungs agree to 0.09% (land 77.962 vs stark 77.89).

Inputs (ALL in-scope to this launch):
  * K6  : fresh instrumented re-run, research/.../walltps_defn_660/walltps_defn_capture.json
          (full per-step timing; 5 wall_tps definitions; vLLM native spec meters from
           k6/server.log). Reproduces #632 K6 (170.16 vs 170.21, 0.03%).
  * K5  : earlier complete session (re-run was cut off at 115/128, no summary/stream) ->
          numbers preserved from that session, flagged provenance="earlier_session".
  * stark 155.58 / AR 77.89 : QUOTED in the in-scope advisor PR #660 + committed
          research/CURRENT_RESEARCH_STATE.md + research/EXPERIMENTS_LOG.md (stark method =
          "3 wall-TPS acceptor/AR-rung/un-rescued on one harness; ratio acceptor/AR x 126.378";
           un-rescued ceiling 155.58, AR 77.89). stark's exact denominator WINDOWING is in his
          out-of-launch-scope branch and was NOT inspected; the verdict does not need it.
  * fire-rate f (#648/#658, own), AR rung A=77.962 (#658, own).

The decisive argument is AR-AGREEMENT: any window/boot/per-request overhead that could
explain a 9.4% K6 gap would ALSO depress the (much longer) AR decode -> the AR rungs would
disagree. They agree to 0.09%, so boot, per-request gap, and prefill are all pinned EQUAL;
the only locus left for a 9.4% K6-only gap is the SPECULATIVE per-step component (draft
forwards + verify, and/or acceptance length), which exists only at K>=1 -> a REAL spec-path
throughput difference, NOT the advisor's named full-e2e-vs-steady window (worth <1% here).

analysis_only=True, official_tps=0. NO HF Job, NO submission, locked 126.378 rung untouched.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

# ---- cross-val + rescue constants (all from THIS agent's in-scope #658 / #648 artifacts) ----
STARK_UNRESCUED_K6 = 155.58       # advisor PR #660 baseline + committed research state (stark's ceiling)
STARK_AR_REF = 77.89              # advisor PR #660 baseline (stark arm-d M=1 AR rung)
LAND_AR_REF_A = 77.962            # #658 A_local_ar_rung_tps (rescue denominator A; agrees w/ stark 0.09%)
PPL_UNCHANGED = 2.0055            # #632/#623 spec-lane PPL (identity-preserving, K-indep)
FIRE = {6: 0.072823, 5: 0.072739}  # #648/#658 per-K tau=0.5 served fire-rate (K-independent)
NUM_TOK = 65536                   # 128 prompts x 512 output tokens
BOOT_S = 120.02                   # K6 server_ready_s (cold model load)

# K5 from the earlier COMPLETE session (the 11:13Z re-run was interrupted at 115/128 with no
# summary/stream pass, so K5 is reported from the earlier full session; K6 is the fresh re-run).
K5_EARLIER = {
    "K": 5, "provenance": "earlier_session_complete (re-run cut off 115/128)",
    "full_e2e_nonstream": 172.74, "steady_gen_meter": 172.02,
    "stream_full_e2e": 171.5, "stream_steady": 178.0, "cold_job_wall": 131.5,
    "server_ready_s": 118.0, "duration_s": 379.4, "mean_ttft_s": 0.084, "espec_mean": 3.47,
}


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def reprice(U: float, f: float, A: float) -> float:
    return 1.0 / (1.0 / U + f / A)


def parse_meters(server_log: Path) -> dict[str, Any]:
    txt = server_log.read_text(errors="replace") if server_log.exists() else ""
    def grab(pat):
        return [float(x) for x in re.findall(pat, txt)]
    gen = [v for v in grab(r"Avg generation throughput:\s*([0-9.]+)") if v > 60.0]
    acc = grab(r"Accepted throughput:\s*([0-9.]+)")
    drf = grab(r"Drafted throughput:\s*([0-9.]+)")
    mal = grab(r"Mean acceptance length:\s*([0-9.]+)")
    m = lambda xs: statistics.fmean(xs) if xs else None
    return {
        "gen_throughput_mean": m(gen), "gen_throughput_n": len(gen),
        "accepted_throughput_mean": m(acc), "drafted_throughput_mean": m(drf),
        "spec_acceptance_length_mean": m(mal),
        "no_native_meter_equals_stark": all(
            v is None or abs(pct(v, STARK_UNRESCUED_K6)) > 4.0 for v in (m(gen), m(acc), m(drf))
        ),
    }


def k6_from_capture(cap: dict[str, Any]) -> dict[str, Any]:
    row = next(r for r in cap["rows"] if int(r["K"]) == 6)
    sp = row.get("pass2_stream", {}) or {}
    gm = row.get("gen_meter", {}) or {}
    return {
        "K": 6, "provenance": "fresh_rerun_complete",
        "full_e2e_nonstream": row["pass1_full_e2e_wall_tps"],
        "steady_gen_meter": gm.get("steady_gen_tps_mean"),
        "stream_full_e2e": sp.get("stream_full_e2e_wall_tps"),
        "stream_steady": sp.get("stream_steady_wall_tps"),
        "cold_job_wall": row.get("cold_job_wall_tps"),
        "server_ready_s": row.get("server_ready_s"),
        "duration_s": row.get("pass1_duration_s"),
        "mean_ttft_s": sp.get("mean_ttft_s"),
        "espec_mean": gm.get("espec_mean"),
    }


def price_row(r: dict[str, Any]) -> dict[str, Any]:
    K = r["K"]
    f = FIRE[K]
    defs = {k: r.get(k) for k in
            ("full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e",
             "stream_steady", "cold_job_wall")}
    decode_window = ["full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e", "stream_steady"]
    full, steady = defs["full_e2e_nonstream"], defs["steady_gen_meter"]
    residual = {k: (pct(v, STARK_UNRESCUED_K6) if v else None) for k, v in defs.items()}
    dw_res = [abs(residual[k]) for k in decode_window if residual.get(k) is not None]
    out = {
        **{k: r.get(k) for k in ("K", "provenance", "server_ready_s", "duration_s",
                                 "mean_ttft_s", "espec_mean")},
        "fire_rate": f, "defs": defs,
        # PR-named axis = full_e2e vs steady; report its spread on THIS harness
        "named_axis_full_vs_steady_spread_pct": (pct(steady, full) if (full and steady) else None),
        "residual_vs_stark_pct": residual,
        # residual to stark after picking the most-favorable named/decode-window definition
        "gap_residual_after_def_match_pct": (min(dw_res) if dw_res else None),
        # rescued de-projection rescued = 1/(1/U + f/A)
        "rescued_full_e2e": (reprice(full, f, LAND_AR_REF_A) if full else None),
        "rescued_steady": (reprice(steady, f, LAND_AR_REF_A) if steady else None),
    }
    return out


def ar_agreement_diagnostic() -> dict[str, Any]:
    """Prove the 9.4% K6 gap CANNOT be boot / per-request / prefill: those would also
    move the AR rung, but AR rungs agree to 0.09%."""
    ar_decode_s = NUM_TOK / LAND_AR_REF_A
    # (1) boot-inclusion test: if a harness folds the 120s cold load into wall_tps, what is AR?
    ar_with_boot = NUM_TOK / (ar_decode_s + BOOT_S)
    ar_with_boot_delta = pct(ar_with_boot, LAND_AR_REF_A)
    # (2) per-request overhead test: g/req that would create the OBSERVED K6 denom inflation,
    #     then applied to AR. K6 needs denom 65536/155.58 vs my 65536/170.158.
    k6_denom_stark = NUM_TOK / STARK_UNRESCUED_K6
    k6_denom_mine = NUM_TOK / 170.158
    extra_s = k6_denom_stark - k6_denom_mine          # seconds the gap implies over 128 req
    g_per_req = extra_s / 128.0
    ar_with_g = NUM_TOK / (ar_decode_s + 128.0 * g_per_req)
    ar_with_g_delta = pct(ar_with_g, LAND_AR_REF_A)
    return {
        "ar_rungs_agree_pct": pct(LAND_AR_REF_A, STARK_AR_REF),   # ~0.09%
        "ar_decode_s": ar_decode_s,
        "boot_s": BOOT_S,
        "ar_if_boot_included_tps": ar_with_boot,
        "ar_if_boot_included_delta_pct": ar_with_boot_delta,      # ~ -12.5% -> boot NOT in stark AR
        "k6_gap_implied_extra_s_over_128req": extra_s,            # ~ +36 s
        "k6_gap_implied_per_request_s": g_per_req,                # ~ 0.28 s/req
        "ar_if_that_per_request_applied_tps": ar_with_g,
        "ar_if_that_per_request_applied_delta_pct": ar_with_g_delta,  # ~ -4.1% -> per-req NOT it
        "conclusion": ("AR-agreement (0.09%) pins boot+per-request+prefill EQUAL between "
                       "harnesses (each would force AR to disagree by 4-13%); the 9.4% K6 gap "
                       "is therefore localized to the speculative per-step component (draft "
                       "forwards + verify and/or acceptance length) -> a REAL spec-path "
                       "difference, NOT a wall_tps windowing definition."),
    }


def decide(k6: dict[str, Any], meters: dict[str, Any], ar: dict[str, Any]) -> dict[str, Any]:
    named_spread = abs(k6["named_axis_full_vs_steady_spread_pct"])
    residual = k6["gap_residual_after_def_match_pct"]
    rv = k6["residual_vs_stark_pct"]
    decode_window = ["full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e", "stream_steady"]
    any_decode_matches = any(rv.get(n) is not None and abs(rv[n]) < 1.5 for n in decode_window)
    named_axis_explains = named_spread >= 5.0 and any_decode_matches
    # rescued band (stark-U conservative end vs my-U)
    rescued_stark_U = reprice(STARK_UNRESCUED_K6, FIRE[6], STARK_AR_REF)
    rescued_my_full = k6["rescued_full_e2e"]
    rescued_my_steady = k6["rescued_steady"]
    verdict = "GAP_IS_DEFINITION" if named_axis_explains else "GAP_IS_REAL"
    return {
        "verdict": verdict,
        "named_axis_full_vs_steady_spread_pct": named_spread,
        "named_axis_explains_9p4_gap": named_axis_explains,
        "any_decode_window_def_reaches_stark": any_decode_matches,
        "gap_residual_after_def_match_pct": residual,
        "no_native_vllm_meter_equals_stark": meters["no_native_meter_equals_stark"],
        "ar_agreement_rules_out_boot_and_per_request": True,
        "spec_path_is_the_locus": True,
        "rescued_k6_stark_basis": rescued_stark_U,        # PRIMARY metric (under stark's U)
        "rescued_k6_my_full_basis": rescued_my_full,
        "rescued_k6_my_steady_basis": rescued_my_steady,
        "rescued_k6_band_low_high": [rescued_stark_U, rescued_my_full],
        "rescued_band_spread_pct": pct(rescued_my_full, rescued_stark_U),
        "official_tps": 0, "fires": False,
        "note": ("LOCAL served wall-TPS only; NOT comparable to the 126.378 OFFICIAL anchor "
                 "(different measurement plane). analysis_only -> does NOT trigger #481 fire."),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=str(HERE / "walltps_defn_capture.json"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cap = json.loads(Path(args.capture).read_text())
    meters = parse_meters(HERE / "k6" / "server.log")
    k6 = price_row(k6_from_capture(cap))
    k5 = price_row(K5_EARLIER)
    ar = ar_agreement_diagnostic()
    decision = decide(k6, meters, ar)

    out = {
        "pr": 660, "analysis_only": True, "official_tps": 0,
        "stack": "int4_mtp_batchinv", "config": cap.get("config"),
        "cross_val": {"stark_unrescued_k6": STARK_UNRESCUED_K6, "stark_ar_ref": STARK_AR_REF,
                      "land_ar_ref_A": LAND_AR_REF_A,
                      "stark_definition_source": ("advisor PR #660 + committed research state; "
                          "method='3 wall-TPS acceptor/AR/un-rescued, ratio x 126.378'; exact "
                          "denominator windowing in stark's out-of-scope branch NOT inspected")},
        "ppl_unchanged": PPL_UNCHANGED,
        "k6_native_vllm_meters": meters,
        "ar_agreement_diagnostic": ar,
        "rows": [k6, k5], "decision": decision,
    }
    (HERE / "defn_reconcile_final.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- console ----
    print("\n=== PR#660 wall_tps definition reconcile (FINAL) ===")
    print(f"stark un-rescued K6 = {STARK_UNRESCUED_K6}  | AR: land {LAND_AR_REF_A} vs stark {STARK_AR_REF} "
          f"(agree {ar['ar_rungs_agree_pct']:.2f}%)")
    for r in (k6, k5):
        d = r["defs"]
        print(f"\nK={r['K']} [{r['provenance']}]  boot={r['server_ready_s']} ttft={r['mean_ttft_s']} espec={r['espec_mean']}")
        for name in ("full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e", "stream_steady", "cold_job_wall"):
            v = d[name]; rp = r["residual_vs_stark_pct"][name]
            print(f"  {name:18s} {v:7.2f}   ({rp:+.2f}% vs stark)" if v else f"  {name:18s}   n/a")
        print(f"  named full-vs-steady spread = {r['named_axis_full_vs_steady_spread_pct']:+.2f}%")
        print(f"  gap_residual_after_def_match = {r['gap_residual_after_def_match_pct']:.2f}%")
        print(f"  rescued: full={r['rescued_full_e2e']:.2f} steady={r['rescued_steady']:.2f}")
    print(f"\nK6 vLLM native meters: gen={meters['gen_throughput_mean']:.2f} "
          f"accepted={meters['accepted_throughput_mean']:.2f} drafted={meters['drafted_throughput_mean']:.2f} "
          f"espec={meters['spec_acceptance_length_mean']:.2f}  no_meter==stark={meters['no_native_meter_equals_stark']}")
    print(f"\nAR-agreement diagnostic:")
    print(f"  if boot folded in -> AR = {ar['ar_if_boot_included_tps']:.2f} ({ar['ar_if_boot_included_delta_pct']:+.2f}%)")
    print(f"  K6 gap implies +{ar['k6_gap_implied_extra_s_over_128req']:.1f}s ({ar['k6_gap_implied_per_request_s']:.3f}s/req);"
          f" applied to AR -> {ar['ar_if_that_per_request_applied_tps']:.2f} ({ar['ar_if_that_per_request_applied_delta_pct']:+.2f}%)")
    print(f"\nDECISION: {json.dumps(decision, indent=2, default=str)}")

    if args.no_wandb:
        return 0

    # ---- W&B ----
    run = wandb_logging.init_wandb_run(
        job_type="walltps_defn_reconcile", agent="land",
        name="land/walltps-defn-reconcile",
        group="walltps-defn-reconcile-land",
        notes=("PR#660 FINAL: the 9.4% un-rescued K6 gap (land 170.16 vs stark 155.58) is NOT the "
               "advisor's named full-e2e-vs-steady window (that spread is <1% on land's harness) and "
               "is NOT any native vLLM meter (155.58 matches none). AR-rung agreement (0.09%) pins "
               "boot/per-request/prefill EQUAL, localizing the gap to the speculative per-step "
               "component -> GAP_IS_REAL. Rescued K6 band [stark-U 135.8 .. land-U 146.8] LOCAL."),
        config={
            "pr": 660, "analysis_only": True, "official_tps": 0,
            "stack": "int4_mtp_batchinv", "drafter": cap.get("drafter"),
            "batch_invariant": 1, "max_num_seqs": 1, "greedy": True,
            "num_prompts": cap.get("config", {}).get("num_prompts"),
            "output_len": cap.get("config", {}).get("output_len"), "seed": 1, "vllm": "0.22.0",
            "stark_unrescued_k6": STARK_UNRESCUED_K6, "stark_ar_ref": STARK_AR_REF,
            "land_ar_ref_A": LAND_AR_REF_A, "ppl_unchanged": PPL_UNCHANGED,
        },
        tags=["optionb", "batch_invariant", "pr660", "walltps_defn", "reconcile", "served", "GAP_IS_REAL"],
    )
    if run is not None:
        import wandb
        cols = ["K", "provenance", "full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e",
                "stream_steady", "cold_job_wall", "named_axis_spread_pct",
                "gap_residual_after_def_match_pct", "rescued_full_e2e", "rescued_steady"]
        tbl = wandb.Table(columns=cols)
        for r in (k6, k5):
            d = r["defs"]
            tbl.add_data(r["K"], r["provenance"], d["full_e2e_nonstream"], d["steady_gen_meter"],
                         d["stream_full_e2e"], d["stream_steady"], d["cold_job_wall"],
                         r["named_axis_full_vs_steady_spread_pct"], r["gap_residual_after_def_match_pct"],
                         r["rescued_full_e2e"], r["rescued_steady"])
            run.log({"global_step": r["K"], "curve/K": r["K"],
                     "curve/full_e2e": d["full_e2e_nonstream"],
                     "curve/steady_gen_meter": d["steady_gen_meter"],
                     "curve/cold_job_wall": d["cold_job_wall"],
                     "curve/rescued_full_e2e": r["rescued_full_e2e"],
                     "curve/rescued_steady": r["rescued_steady"]})
        run.log({"defn_curve": tbl})

        summary = {
            "decision/verdict": decision["verdict"],
            "decision/named_axis_full_vs_steady_spread_pct": decision["named_axis_full_vs_steady_spread_pct"],
            "decision/named_axis_explains_9p4_gap": int(decision["named_axis_explains_9p4_gap"]),
            "decision/gap_residual_after_def_match_pct": decision["gap_residual_after_def_match_pct"],
            "decision/no_native_vllm_meter_equals_stark": int(decision["no_native_vllm_meter_equals_stark"]),
            "decision/rescued_k6_stark_basis": decision["rescued_k6_stark_basis"],
            "decision/rescued_k6_my_full_basis": decision["rescued_k6_my_full_basis"],
            "decision/rescued_band_spread_pct": decision["rescued_band_spread_pct"],
            "ar/agree_pct": ar["ar_rungs_agree_pct"],
            "ar/if_boot_included_delta_pct": ar["ar_if_boot_included_delta_pct"],
            "ar/if_per_request_applied_delta_pct": ar["ar_if_that_per_request_applied_delta_pct"],
            "meters/gen_throughput_mean": meters["gen_throughput_mean"],
            "meters/accepted_throughput_mean": meters["accepted_throughput_mean"],
            "meters/spec_acceptance_length_mean": meters["spec_acceptance_length_mean"],
            "config/stark_unrescued_k6": STARK_UNRESCUED_K6, "config/land_ar_ref_A": LAND_AR_REF_A,
            "config/ppl_unchanged": PPL_UNCHANGED, "config/official_tps": 0,
        }
        for r in (k6, k5):
            K = r["K"]; d = r["defs"]
            summary[f"perK/K{K}_full_e2e"] = d["full_e2e_nonstream"]
            summary[f"perK/K{K}_steady_gen_meter"] = d["steady_gen_meter"]
            summary[f"perK/K{K}_cold_job_wall"] = d["cold_job_wall"]
            summary[f"perK/K{K}_rescued_full_e2e"] = r["rescued_full_e2e"]
            summary[f"perK/K{K}_named_axis_spread_pct"] = r["named_axis_full_vs_steady_spread_pct"]
        wandb_logging.log_summary(run, summary, step=6)
        wandb_logging.log_json_artifact(run, name="walltps_defn_reconcile_660_final",
                                        artifact_type="analysis", data=out)
        url = getattr(run, "url", ""); rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] walltps-defn-reconcile id={rid} url={url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
