#!/usr/bin/env python3
"""PR #560 fern — Stage 2: re-project the realized served TPS from the MEASURED
candidate-verify step (Stage 1), replacing #549's read-bound [+28,+44] roofline band
with a measured number. Gated on Stage 1 + Stage 3.

The served decode wall (#549 Pass A) is 14.871 ms/verify-step at warm-median 264.82 TPS;
the bf16 head GEMM contributes 2.8382 ms in-context (M=8 spec-verify). Candidate-verify
replaces that head-path with the MEASURED cv-step (int4 Marlin GEMV + top-8 verify +
re-argmax). head_savings = (head time removed) - (cv-step). We carry a transparent band:
  conservative  : bf16_step_iso  - cv_step_iso        (both isolation, apples-to-apples)
  central        : in-context head GEMM 2.8382 / isolation-calibrated cv-step
  optimistic     : in-context head GEMM 2.8382 - cv_step_iso  (#549's exact basis)
wall_speedup = wall / (wall - savings); realized_tps = anchor * wall_speedup.

OUTPUTS: cv_realized_tps_gain (+CI), cv_realized_quality_safe_tps, cv_projection_confirmed,
cv_realized_vs_projected_delta, cv_served_tps_is_measured (=False; step-microbench based).
Analysis-only."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S1 = json.load(open(HERE / "stage1_cv_microbench.json"))
S3C = json.load(open(HERE / "stage3c_tiebreak.json"))

A = S1["anchors"]
WALL = A["wall_per_step_ms_served"]            # 14.871
TPS_WM = A["tps_warm_median_m1"]               # 264.82 (the wall's own basis)
ANCHOR_252 = A["anchor_base_fullhead_252"]     # 252.31 (wirbel #553 / lawine #544) PRIMARY
ANCHOR_254 = A["anchor_base_fullhead_254"]     # 253.78 (fern #535) -> the "292" basis
HEAD_INCTX = A["head_gemm_ms_incontext_m8"]    # 2.8382 in-context head GEMM (#549)
PROJ_CENTRAL = A["proj_549_gain_central"]      # 40.10 (#549 int4_g128 central, 264.82 basis)
PROJ_BAND = A["proj_549_gain_band"]            # [28.25, 43.68]
SIGMA_HW = 4.864                               # absolute TPS hw noise (#560 baseline)

M8 = S1["per_M"]["8"]                           # served spec-verify operating point
bf16_step = M8["bf16_head_argmax_step_latency_ms"]   # 2.6999
cv_step = M8["cv_step_latency_ms"]                   # 0.7983
bf16_gemv = M8["bf16_gemv_alone_latency_ms"]         # 2.6815
cv_ci = M8["cv_step"]["ci95_halfwidth_ms"]
bf16_ci = M8["bf16_step"]["ci95_halfwidth_ms"]

# in-context / isolation calibration from the bf16 head (served head GEMM vs isolation GEMV)
calib = HEAD_INCTX / bf16_gemv                       # ~1.058
cv_step_inctx = cv_step * calib

savings = {
    "conservative_iso": bf16_step - cv_step,         # ~1.902
    "central_calibrated": HEAD_INCTX - cv_step_inctx,  # ~1.993
    "optimistic_inctx": HEAD_INCTX - cv_step,        # ~2.040
}


def project(save_ms: float, anchor: float) -> dict:
    new_wall = WALL - save_ms
    spd = WALL / new_wall
    tps = anchor * spd
    return {"head_savings_ms": save_ms, "new_wall_ms": new_wall, "wall_speedup": spd,
            "realized_tps": tps, "gain": tps - anchor}


# central projection (calibrated savings) on each basis
central = {b: project(savings["central_calibrated"], a)
           for b, a in (("252", ANCHOR_252), ("254", ANCHOR_254), ("wm265", TPS_WM))}
# gain band across the savings-basis band, on the 252 anchor (primary) and 265 (wall basis)
band_252 = sorted(project(s, ANCHOR_252)["gain"] for s in savings.values())
band_265 = sorted(project(s, TPS_WM)["gain"] for s in savings.values())

# Stage-1 latency CI -> negligible extra gain uncertainty (savings +- (cv_ci+bf16_ci))
lat_ci = cv_ci + bf16_ci
gain_ci_252 = project(savings["central_calibrated"] - lat_ci, ANCHOR_252)["gain"], \
              project(savings["central_calibrated"] + lat_ci, ANCHOR_252)["gain"]

primary_gain_252 = central["252"]["gain"]
primary_tps_252 = central["252"]["realized_tps"]
projection_confirmed = bool(min(band_252) >= PROJ_BAND[0] and max(band_252) <= PROJ_BAND[1] + 1e-6
                            and PROJ_BAND[0] <= primary_gain_252 <= PROJ_BAND[1])
# also confirm the central lands within the band AND near +40
near_40 = abs(primary_gain_252 - PROJ_CENTRAL) <= SIGMA_HW

rep = {
    "pr": 560, "stage": 2, "analysis_only": True, "official_tps": 0,
    "method": "measured_step_reprojection",
    "cv_served_tps_is_measured": False,
    "cv_served_tps_is_measured_note": "Stage-1 step microbench + this re-projection is the "
        "bankable PRIMARY (PR-sanctioned). Not a full served decode-loop measurement; "
        "time-boxed away from a vLLM decode-loop integration per the card.",
    "operating_point_M": 8,
    "measured_step": {
        "int4_gemv_ms": M8["int4_gemv_latency_ms"],
        "cv_step_ms": cv_step, "bf16_step_ms": bf16_step, "bf16_gemv_alone_ms": bf16_gemv,
        "cv_int4_nominator_bw_GBs": M8["cv_int4_nominator_bw_GBs"],
        "cv_int4_nominator_pct_of_peak": M8["cv_int4_nominator_pct_of_peak"],
        "cv_step_speedup": M8["cv_step_speedup"],
        "incontext_isolation_calib": calib,
    },
    "head_savings_ms_band": savings,
    "central_projection_by_basis": central,
    "cv_realized_tps_gain": primary_gain_252,
    "cv_realized_tps_gain_basis": "anchor_252.31_central_calibrated_savings",
    "cv_realized_tps_gain_band_252": [min(band_252), max(band_252)],
    "cv_realized_tps_gain_band_265": [min(band_265), max(band_265)],
    "cv_realized_tps_gain_ci95_252": list(gain_ci_252),
    "cv_realized_quality_safe_tps": primary_tps_252,
    "cv_realized_quality_safe_tps_252": central["252"]["realized_tps"],
    "cv_realized_quality_safe_tps_254": central["254"]["realized_tps"],
    "cv_realized_quality_safe_tps_wm265": central["wm265"]["realized_tps"],
    "proj_549_gain_central": PROJ_CENTRAL,
    "proj_549_gain_band": PROJ_BAND,
    "proj_549_realized_tps_254": 292.2,
    "cv_projection_confirmed": projection_confirmed,
    "cv_realized_vs_projected_delta": primary_gain_252 - PROJ_CENTRAL,
    "cv_realized_vs_projected_delta_265basis": central["wm265"]["gain"] - PROJ_CENTRAL,
    "central_within_sigma_hw_of_proj40": near_40,
    "sigma_hw_abs_tps": SIGMA_HW,
    "argmax_identity_rate": S3C["identity_vs_served_bf16"]["B_vocTB"],
    "identity_hard_gate_pass": bool(S3C["identity_vs_served_bf16"]["B_vocTB"] == 1.0),
    "head_ceiling_holds_at_292": bool(primary_tps_252 >= 288.0),
}
(HERE / "stage2_reproject.json").write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))

print("=" * 12 + " PR #560 STAGE 2 — MEASURED REALIZED-TPS RE-PROJECTION " + "=" * 12)
print(f"  measured int4 GEMV BW (M8)   = {M8['cv_int4_nominator_bw_GBs']:.1f} GB/s "
      f"({100*M8['cv_int4_nominator_pct_of_peak']:.1f}% peak)")
print(f"  measured cv-step / bf16-step = {cv_step:.3f} / {bf16_step:.3f} ms "
      f"({M8['cv_step_speedup']:.2f}x)")
print(f"  head_savings band            = [{savings['conservative_iso']:.3f}, "
      f"{savings['optimistic_inctx']:.3f}] ms (central {savings['central_calibrated']:.3f})")
for b, lbl in (("252", "252.31 (wirbel/lawine, PRIMARY)"), ("254", "253.78 (#535)"), ("wm265", "264.82 (wall basis)")):
    c = central[b]
    print(f"  realized TPS @ {lbl:32s} = {c['realized_tps']:.1f}  (gain +{c['gain']:.1f})")
print(f"  cv_realized_tps_gain (252)   = +{primary_gain_252:.1f}  band {[round(x,1) for x in band_252]}  "
      f"(CI95 {[round(x,1) for x in gain_ci_252]})")
print(f"  #549 projection central/band = +{PROJ_CENTRAL:.1f} / {[round(x,1) for x in PROJ_BAND]}")
print(f"  >>> cv_projection_confirmed  = {projection_confirmed}  "
      f"(delta vs proj +{rep['cv_realized_vs_projected_delta']:.2f} on 252 / "
      f"{rep['cv_realized_vs_projected_delta_265basis']:+.2f} on 265)")
print(f"  argmax_identity_rate (HARD)  = {rep['argmax_identity_rate']}  pass={rep['identity_hard_gate_pass']}")
print(f"  head ceiling holds ~292      = {rep['head_ceiling_holds_at_292']}")
print("=" * 80)
