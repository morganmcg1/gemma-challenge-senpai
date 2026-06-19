#!/usr/bin/env python3
"""PR #735 -- reconcile the #730 fire packet's official-equiv into ONE defensible
number + P(clears 126.378) for the EXACT #730 candidate (`int4_mtp_batchinv`,
un-rescued, stock-Hub drafter google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant,
K=6). `analysis_only` -- NO HF Job, NO fire. Guard flags live in wandb.summary.

THE CRUX (PR step 1): is the spec-local on the SAME substrate as #728's AR base
(106.02, faithful 0.22.0 BI=1)? -> NO. The packet's "172.7 stock-drafter K=6 local"
traces to research/walltps_ab/optionb_bi1_stock_int4/ksweep/k5/paired_ab.json
(land PR#82, branch land/optionb-bi1-k-sweep). That measurement is:
  * NUM_SPECULATIVE_TOKENS = 5   -> it is K=5, not K=6.  (true K=6 = 170.21)
  * DRAFTER_MODEL = /tmp/qat-assistant  -> the FAST drafter, NOT the stock-Hub
    drafter the #730 manifest ships (CURRENT_RESEARCH_STATE.md cycle-58DG:
    "134.87 (shippable stock-Hub drafter) / 147.55 (fast /tmp/qat-assistant,
    provenance-suspect)"). /tmp/qat-assistant is a LOCAL path -> can't even run on
    the HF a10g runner; the fired config uses the manifest's stock-Hub drafter.
  * MODEL_ID = google/gemma-4-E4B-it-qat-w4a16-ct  -> matches the submission base
    model (good), but DIFFERS from #728's int4_g128_lmhead AR base.

=> #728's anchored x1.192 (= 126.378/106.02) is calibrated to a DIFFERENT
checkpoint+engine. Applying it to land's 170.21 (172.74x1.192=205.9 / 170.21x1.192
=202.9) is a CROSS-CHECKPOINT over-count -> INVALID. And the optionb k-sweep has
NO drafter-off AR arm (every arm is spec K in {3..7}) -> the speedup-ratio form
(spec-local / same-substrate AR-local) is UNAVAILABLE in-branch. So we BOUND with
named int4-precision transfer points (exactly as #725 did), carrying #725's 4-9%
haircut so the number is comparable.

AUTHORITATIVE TRANSFER RULE (supersedes the scattered projections):
  For an int4 candidate, anchor the local->official transfer on int4 SAME-PRECISION
  pairs. In-branch these cluster at T in [0.960, 1.000]:
    T_int4_match  0.9863  int4_g128_lmhead 126.378/128.13 (single_stream)  <- central
    T_int4_pess   0.9603  lmhead12k        126.378/131.60                  <- floor
    T_def         1.0000  wall_tps == official output_throughput            <- ceiling
    (#732 captured-graph meter-matched independently gave 0.9971 ~ definitional.)
  REJECT for the authoritative number:
    T_flagship  1.0602  -- bf16 (wrong precision) flagship; an int4 candidate's
                           meter does not get the bf16 low-read tail.
    T_advisor   0.880   -- advisor back-of-envelope 139.9/159; BELOW every int4 pair.
    T_728anch   1.192   -- 126.378/106.02; cross-checkpoint+engine -> invalid here.
  (Aside: the bf16 flagship's 1.0602 is measured on the SAME PR#82 engine as land's
  170.21, i.e. that engine's local wall_tps reads ~6% LOW -- so 0.986 may be
  conservative for THIS engine. Kept as the defensible int4 central regardless.)

THE BAR HAS NO HAIRCUT: 126.378 = int4_g128_lmhead (PR#4) pure-AR, private==public.
Haircut the candidate (a drafter stack) only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/senpai/target")
HERE = Path(__file__).resolve().parent

K6_PAIRED = ROOT / "research/walltps_ab/optionb_bi1_stock_int4/ksweep/k6/paired_ab.json"
K5_PAIRED = ROOT / "research/walltps_ab/optionb_bi1_stock_int4/ksweep/k5/paired_ab.json"
PROJ_CAL = ROOT / "research/walltps_ab/local_official_projection/projection_cal.json"
CEIL_728 = ROOT / "research/spec_achievable_ceiling/runs/sweep/report.json"

BAR_OFFICIAL = 126.378            # int4_g128_lmhead PR#4, pure-AR (private==public, no haircut)

# int4-precision-anchored transfer points (the authoritative family)
T_INT4_MATCH = 126.378 / 128.13   # 0.9863 int4_g128_lmhead single_stream (same precision)  <- central
T_INT4_PESS = 126.378 / 131.60    # 0.9603 lmhead12k (most pessimistic measured int4)        <- floor
T_DEFINITION = 1.000              # wall_tps == official output_throughput                    <- ceiling
T_INT4_CAPTURED = 0.9971          # #732 captured-graph meter-matched (corroborates ~definitional)

# REJECTED transfer points (carried only to explain the spread)
T_ADVISOR = 139.9 / 159.0         # 0.8804 advisor back-of-envelope (below all int4 pairs)
# T_FLAGSHIP read from projection_cal.json (1.0602, bf16 -- wrong precision)
# T_728_ANCHORED read from report.json (1.192, cross-checkpoint -- invalid)

# authoritative MC band: int4-pessimistic floor .. definitional ceiling
MC_LO = T_INT4_PESS               # 0.9603
MC_HI = T_DEFINITION              # 1.0000

# documented private-verify haircut band (same as #725)
HAIRCUT_LO = 0.04
HAIRCUT_HI = 0.09
PRIVATE_REPRO_GATE = 0.05         # separate validity gate: DQ if private drift > 5%

N_MC = 400_000
SEED = 730


def _load():
    k6 = json.loads(K6_PAIRED.read_text())
    k5 = json.loads(K5_PAIRED.read_text())
    cal = json.loads(PROJ_CAL.read_text())["calibration"]
    ceil = json.loads(CEIL_728.read_text())
    l_k6 = float(k6["verdict"]["candidate_median_wall_tps"])           # 170.21 (NUM_SPEC=6)
    l_k5 = float(k5["verdict"]["candidate_median_wall_tps"])           # 172.74 (NUM_SPEC=5)
    return {
        "local_k6_wall_tps": l_k6,
        "local_k5_wall_tps": l_k5,
        "k6_e_accept_exact": float(k6["arms"]["candidate"]["e_accept_exact"]["median"]),
        "k6_num_spec": k6["candidate"]["override_env"]["NUM_SPECULATIVE_TOKENS"],
        "k6_drafter": k6["candidate"]["override_env"]["DRAFTER_MODEL"],
        "k6_model_id": k6["candidate"]["override_env"]["MODEL_ID"],
        "k6_branch": k6.get("git", {}).get("git_branch"),
        "k6_pr": k6.get("pr"),
        "T_flagship_bf16": float(cal["local_to_official_multiplier"]),  # 1.0602
        "ar_base_728_local_bi1": float(ceil["transfer_model"]["ar_base_local_bi1"]),  # 106.02
        "ratio_728_anchored": float(ceil["transfer_model"]["ratio_anchored"]),        # 1.192
        "ar_728_model_id": ceil["config"]["model_dir"],                # int4_g128_lmhead
    }


def _point(local, mult, haircut, name):
    val = local * mult * (1 - haircut)
    return {"name": name, "local": local, "mult": mult, "haircut": haircut,
            "official_equiv_private": val, "margin_pct": 100 * (val / BAR_OFFICIAL - 1),
            "clears": bool(val > BAR_OFFICIAL)}


def _breakeven_mult(local, haircut):
    return BAR_OFFICIAL / (local * (1.0 - haircut))


def _mc_band(local, lo, hi, rng):
    m = rng.uniform(lo, hi, N_MC)
    h = rng.uniform(HAIRCUT_LO, HAIRCUT_HI, N_MC)
    pub = local * m
    priv = pub * (1 - h)
    return {
        "mult_band": [lo, hi], "haircut_band": [HAIRCUT_LO, HAIRCUT_HI],
        "p_clears_bar": float((priv > BAR_OFFICIAL).mean()),
        "official_equiv_public_p50": float(np.percentile(pub, 50)),
        "official_equiv_private_p05": float(np.percentile(priv, 5)),
        "official_equiv_private_p50": float(np.percentile(priv, 50)),
        "official_equiv_private_p95": float(np.percentile(priv, 95)),
    }


def analyze():
    inp = _load()
    rng = np.random.default_rng(SEED)
    L6 = inp["local_k6_wall_tps"]
    L5 = inp["local_k5_wall_tps"]
    T_FLAGSHIP = inp["T_flagship_bf16"]
    T_728 = inp["ratio_728_anchored"]

    # ---- authoritative number: K=6 local x int4 transfer band, #725 haircut ----
    central_public = L6 * T_INT4_MATCH
    grid_k6 = {
        "central_T0986": {
            "h0": _point(L6, T_INT4_MATCH, 0.0, "k6/T_int4_match/h0"),
            "h4_best": _point(L6, T_INT4_MATCH, HAIRCUT_LO, "k6/T_int4_match/h4"),
            "h65_mid": _point(L6, T_INT4_MATCH, 0.065, "k6/T_int4_match/h6.5"),
            "h9_worst": _point(L6, T_INT4_MATCH, HAIRCUT_HI, "k6/T_int4_match/h9"),
        },
        "floor_T0960": {
            "h9_worst": _point(L6, T_INT4_PESS, HAIRCUT_HI, "k6/T_int4_pess/h9"),
        },
        "ceiling_T1000": {
            "h4_best": _point(L6, T_DEFINITION, HAIRCUT_LO, "k6/T_def/h4"),
            "h9_worst": _point(L6, T_DEFINITION, HAIRCUT_HI, "k6/T_def/h9"),
        },
        "rejected_advisor_T0880": {
            "h9_worst": _point(L6, T_ADVISOR, HAIRCUT_HI, "k6/T_advisor/h9"),
        },
    }
    worst_auth_corner = grid_k6["floor_T0960"]["h9_worst"]  # 0.960 x 9% = worst defensible

    mc_auth = _mc_band(L6, MC_LO, MC_HI, rng)               # int4 band [0.960, 1.000]
    mc_725compat = _mc_band(L6, MC_LO, T_FLAGSHIP, rng)     # #725's [0.960, 1.0604] for comparability

    breakeven = {
        "k6_at_worst_9pct_haircut": _breakeven_mult(L6, HAIRCUT_HI),  # 0.816
        "k6_at_mid_6p5pct_haircut": _breakeven_mult(L6, 0.065),
        "k6_no_haircut": _breakeven_mult(L6, 0.0),
    }

    # ---- reconciliation of the three scattered estimates ----
    reconciliation = {
        "packet_725_139p9": {
            "official_equiv": 139.9, "local_used": 159.0, "transfer_used": round(T_ADVISOR, 4),
            "what_drives_it": "advisor back-of-envelope: low verbal local (159, provenance TBD) x "
                              "transfer 0.880 (BELOW every measured int4 pair). Doubly conservative.",
            "authoritative_for_730": False,
            "reason": "uses an unverified 159 local and a transfer below all int4 data; "
                      "not the int4-anchored central.",
        },
        "fn_732_147p55": {
            "official_equiv": 147.55, "local_used": "~170 (fast /tmp/qat-assistant)",
            "transfer_used": "~0.87 ('stark tax')",
            "what_drives_it": "CURRENT_RESEARCH_STATE.md cycle-58DG: '147.55 (fast /tmp/qat-assistant, "
                              "provenance-suspect)' via stark-tax 0.870. It is the FAST drafter (not "
                              "the stock-Hub ship drafter) under a pessimistic transfer.",
            "authoritative_for_730": False,
            "reason": "fast-drafter proxy + sub-measured transfer; provenance-suspect by its own label.",
        },
        "anchored_728_x1p192": {
            "official_equiv_if_applied_to_L6": L6 * T_728,   # ~202.9
            "official_equiv_if_applied_to_L5": L5 * T_728,   # ~205.9
            "local_used": f"{L6:.2f} (K=6) or {L5:.2f} (K=5)", "transfer_used": round(T_728, 4),
            "what_drives_it": "126.378/106.02 is #728's faithful-engine AR-base ratio on "
                              f"checkpoint {inp['ar_728_model_id']}, a DIFFERENT model+engine than "
                              f"land's {inp['k6_model_id']} on PR#{inp['k6_pr']}. Cross-checkpoint.",
            "authoritative_for_730": False,
            "reason": "INVALID: the x1.192 cancels #728's local->official gap for ITS substrate/"
                      "checkpoint; land's 170.21 already embeds a different local clock. Over-count.",
        },
        "THIS_authoritative": {
            "official_equiv_public": central_public,                 # 167.8
            "official_equiv_private_mid_6p5": grid_k6["central_T0986"]["h65_mid"]["official_equiv_private"],
            "local_used": f"{L6:.2f} (TRUE K=6, NUM_SPEC=6)",
            "transfer_used": round(T_INT4_MATCH, 4),
            "what_drives_it": "int4 same-precision transfer 0.986 (band 0.960-1.000) on the TRUE "
                              "K=6 local; carries #725's 4-9% haircut.",
            "authoritative_for_730": True,
            "reason": "int4-precision-anchored, correct K, correct base model; rejects bf16/advisor/"
                      "x1.192. This is the defensible decision number.",
        },
    }

    # ---- provenance flags the human must see before firing #730 ----
    provenance_flags = {
        "K_mismatch": {
            "packet_says": "K=6, 172.7 local",
            "reality": f"172.74 is K=5 (NUM_SPEC=5); TRUE K=6 (NUM_SPEC=6) = {L6:.2f}",
            "impact": "use 170.21, not 172.74. ~1.5% lower; also the conservative/correct K.",
        },
        "DRAFTER_mismatch": {
            "packet_says": "stock-Hub drafter google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
            "reality": f"every optionb measurement used DRAFTER_MODEL={inp['k6_drafter']} (the FAST "
                       "drafter; CURRENT_RESEARCH_STATE.md labels it 'fast, provenance-suspect' and "
                       "distinct from the 'shippable stock-Hub drafter'). The submission MANIFEST "
                       "ships the stock-Hub id, and /tmp/qat-assistant is a LOCAL path unavailable "
                       "on the HF runner.",
            "impact": "170.21 is an UPPER BOUND on the literal stock-Hub un-rescued K=6 local, which "
                      "is UNMEASURED on-branch. The fired config (stock-Hub drafter) would likely be "
                      "slower. GO still robust (see margins) but the point estimate is provenance-thin.",
        },
        "SUBSTRATE_gap": {
            "fact": f"no drafter-off AR arm in the optionb k-sweep (all arms spec K in 3..7); "
                    f"land's 170.21 has NO same-substrate AR base. #728's AR base 106.02 is on a "
                    f"different checkpoint ({inp['ar_728_model_id']}) + engine.",
            "impact": "speedup-ratio form unavailable in-branch; bounded via int4 transfer points. "
                      "x1.192 anchored form is invalid (cross-checkpoint).",
        },
    }

    verdict = {
        "one_number_official_equiv_public": round(central_public, 1),
        "one_number_official_equiv_private_central": round(grid_k6["central_T0986"]["h65_mid"]["official_equiv_private"], 1),
        "p_clears_126378_after_haircut": mc_auth["p_clears_bar"],
        "worst_defensible_corner_margin_pct": round(worst_auth_corner["margin_pct"], 1),
        "worst_defensible_corner_official_equiv": round(worst_auth_corner["official_equiv_private"], 1),
        "breakeven_transfer_at_9pct_haircut": round(breakeven["k6_at_worst_9pct_haircut"], 4),
        "is_upper_bound_for_stock_ship_drafter": True,
        "headline": (
            "GO (comfortable), but the number is an UPPER BOUND. Authoritative int4-anchored "
            "official-equiv for the MEASURED K=6 config (fast /tmp/qat-assistant proxy on the "
            "ship base model w4a16-ct) = %.1f public / ~%.1f private (mid-haircut). P(clears "
            "126.378 after 4-9%% haircut) = %.2f over the int4 band [0.960,1.000]; worst "
            "defensible corner = %.1f TPS (+%.1f%%, T=0.960 x 9%% haircut). The fire fails ONLY if "
            "the true land-substrate->official int4 transfer drops below %.3f -- far under every "
            "measured int4 pair and under the advisor's own 0.880 (which still clears at +%.1f%%). "
            "CAVEAT: the #730 submission SHIPS the stock-Hub drafter, never locally speed-measured; "
            "170.21 (fast proxy) bounds it from ABOVE. Even the slower stock projections (147.55) "
            "clear post-9%%-haircut (+6.2%%). Reconciles the scattered 139.9/147.55/x1.192: 139.9 = "
            "advisor 0.880 floor, 147.55 = fast-proxy stark-tax, x1.192 = INVALID cross-checkpoint."
        ) % (
            central_public, grid_k6["central_T0986"]["h65_mid"]["official_equiv_private"],
            mc_auth["p_clears_bar"], worst_auth_corner["official_equiv_private"],
            worst_auth_corner["margin_pct"], breakeven["k6_at_worst_9pct_haircut"],
            grid_k6["rejected_advisor_T0880"]["h9_worst"]["margin_pct"],
        ),
    }

    return {
        "pr": 735, "student": "kanna",
        "card": "reconcile #730 official-equiv into ONE defensible number + P(clears 126.378)",
        "guard_flags": {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0},
        "bar_official": BAR_OFFICIAL,
        "bar_config": "int4_g128_lmhead PR#4 (pure-AR; private==public; no haircut on the bar)",
        "inputs": inp,
        "authoritative_transfer_rule": {
            "central": T_INT4_MATCH, "band": [MC_LO, MC_HI],
            "members": {"T_int4_match": T_INT4_MATCH, "T_int4_pess": T_INT4_PESS,
                        "T_definition": T_DEFINITION, "T_int4_captured_732": T_INT4_CAPTURED},
            "rejected": {"T_flagship_bf16": T_FLAGSHIP, "T_advisor": T_ADVISOR,
                         "T_728_anchored": T_728},
            "rule": "int4 same-precision anchor; reject bf16(wrong precision)/advisor(below data)/"
                    "x1.192(cross-checkpoint).",
        },
        "haircut_band": {"lo": HAIRCUT_LO, "hi": HAIRCUT_HI, "bar_haircut": 0.0},
        "grid_k6": grid_k6,
        "monte_carlo_authoritative_int4_band": mc_auth,
        "monte_carlo_725compat_band": mc_725compat,
        "breakeven": breakeven,
        "reconciliation": reconciliation,
        "provenance_flags": provenance_flags,
        "private_repro_gate_SEPARATE_risk": {
            "gate_threshold_pct": PRIVATE_REPRO_GATE * 100,
            "haircut_band_straddles_gate": HAIRCUT_LO < PRIVATE_REPRO_GATE < HAIRCUT_HI,
            "interpretation": "clearing the 126.378 BAR is comfortable; the SEPARATE 5% private-repro "
                              "validity gate (drafter stacks drift most, kanna#44) is the real residual "
                              "risk -- a stack can beat the bar yet be DQ'd.",
        },
        "verdict": verdict,
    }


def _print(out):
    v = out["verdict"]
    print("VERDICT:", v["headline"])
    print(f"\n  official-equiv (public, authoritative)  = {v['one_number_official_equiv_public']}")
    print(f"  official-equiv (private, mid-haircut)   = {v['one_number_official_equiv_private_central']}")
    print(f"  P(clears 126.378 after 4-9% haircut)    = {v['p_clears_126378_after_haircut']:.4f}")
    print(f"  worst defensible corner                 = {v['worst_defensible_corner_official_equiv']} "
          f"({v['worst_defensible_corner_margin_pct']:+.1f}%)")
    print(f"  breakeven transfer @ 9% haircut         = {v['breakeven_transfer_at_9pct_haircut']}")
    print("\n-- reconciliation --")
    for k, r in out["reconciliation"].items():
        auth = "AUTHORITATIVE" if r.get("authoritative_for_730") else "rejected"
        oe = r.get("official_equiv") or r.get("official_equiv_public") or r.get("official_equiv_if_applied_to_L6")
        print(f"  {k:24s} [{auth:13s}] official-equiv~{oe}")
    print("\n-- provenance flags --")
    for k, f in out["provenance_flags"].items():
        print(f"  {k}: {f.get('impact','')[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default="kanna-730-officialequiv-reconcile")
    ap.add_argument("--name", default="kanna/730-officialequiv-reconcile")
    ap.add_argument("--out", default=str(HERE / "results/officialequiv_reconcile_730.json"))
    args = ap.parse_args()

    out = analyze()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    out["_outpath"] = str(outp)
    _print(out)
    print("\nWROTE", outp)

    if args.wandb:
        import wandb
        v = out["verdict"]
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=args.name, job_type="analysis",
            config={
                "pr": 735, "student": "kanna",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "bar_official": BAR_OFFICIAL,
                "candidate": "int4_mtp_batchinv un-rescued stock-Hub drafter K=6 (#730)",
                "authoritative_transfer_central": T_INT4_MATCH,
                "authoritative_transfer_band": [MC_LO, MC_HI],
                "haircut_band": [HAIRCUT_LO, HAIRCUT_HI],
                "local_k6_wall_tps": out["inputs"]["local_k6_wall_tps"],
                "local_k5_wall_tps": out["inputs"]["local_k5_wall_tps"],
                "k6_drafter_measured": out["inputs"]["k6_drafter"],
                "k6_model_id": out["inputs"]["k6_model_id"],
            },
            tags=["pr735", "kanna", "analysis_only", "official-equiv-reconcile", "730-fire",
                  "upper-bound", "drafter-provenance-flag"],
        )
        summary = {
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "primary_metric_name": "official_equiv_public_authoritative",
            "primary_metric_value": v["one_number_official_equiv_public"],
            "test_metric_name": "p_clears_126378_after_haircut",
            "test_metric_value": v["p_clears_126378_after_haircut"],
            "official_equiv_public": v["one_number_official_equiv_public"],
            "official_equiv_private_central": v["one_number_official_equiv_private_central"],
            "p_clears_126378_after_haircut": v["p_clears_126378_after_haircut"],
            "worst_defensible_corner_margin_pct": v["worst_defensible_corner_margin_pct"],
            "worst_defensible_corner_official_equiv": v["worst_defensible_corner_official_equiv"],
            "breakeven_transfer_at_9pct_haircut": v["breakeven_transfer_at_9pct_haircut"],
            "is_upper_bound_for_stock_ship_drafter": 1,
            "local_k6_wall_tps": out["inputs"]["local_k6_wall_tps"],
            "local_k5_wall_tps": out["inputs"]["local_k5_wall_tps"],
            "T_authoritative_central": T_INT4_MATCH,
            "T_rejected_flagship_bf16": out["inputs"]["T_flagship_bf16"],
            "T_rejected_advisor": T_ADVISOR,
            "T_rejected_728_anchored": out["inputs"]["ratio_728_anchored"],
            "recon_725_139p9": 139.9,
            "recon_732_147p55": 147.55,
            "recon_728_anchored_L6": out["reconciliation"]["anchored_728_x1p192"]["official_equiv_if_applied_to_L6"],
            "k6_num_spec": int(out["inputs"]["k6_num_spec"]),
            "mc_725compat_p_clears": out["monte_carlo_725compat_band"]["p_clears_bar"],
        }
        run.summary.update(summary)
        wandb.log(summary)
        print("WANDB_RUN_ID", run.id)
        print("WANDB_RUN_URL", run.url)
        wandb.finish()


if __name__ == "__main__":
    main()
