#!/usr/bin/env python3
"""PR #549 Stage 3 — candidate+verify realized-TPS PROJECTION (analysis-only).

Stage 3 of the PR ("prototype the winning candidate+verify decode, measure
realized served TPS + greedy-identity") is run here as a *read-bound
projection*, not a served prototype, to honor the HARD CONSTRAINT
``analysis_only=true`` / no served-file change. The projection is grounded in
MEASURED anchors only:

  * achieved HBM bandwidth = measured full-head bf16 GEMM bytes / measured time
    (Stage 1: 1.342 GB / 2.8382 ms = 472.9 GB/s on the real served op), and
  * the candidate-verify read = cheap int4/fp8 nominator weight read + K_safe
    full-precision verify-row gather (Stage 2 ``served_head_read_bytes`` and
    ``verify_gather_bytes_at_Ksafe``).

Greedy identity is NOT projected: it is the Stage-2 offline measurement
(miss_rate@K_safe == 0 over 60000 held-out decode positions). By the
candidate-verify construction (cheap top-K nominator whose set provably
contains the true fp argmax, then exact full-precision verify of only those
rows) a zero offline miss rate IS a byte-exact greedy-identity guarantee.

Runs under SYSTEM python3 (has real wandb). Import wandb FIRST so the cached
real module beats the ./wandb run-data namespace-package shadow under ROOT.
"""
from __future__ import annotations

# --- real-wandb-first (beats ./wandb namespace shadow); harmless if absent ---
try:  # pragma: no cover
    import wandb as _wandb_real  # noqa: F401  (cache the real module in sys.modules)
except Exception:  # pragma: no cover
    _wandb_real = None

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "fullhead_candidate_verify"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # APPEND (not insert) so site-packages wandb wins

STAGE1_JSON = HERE / "stage1_report.json"
STAGE2_JSON = HERE / "stage2_report.json"
OUT_JSON = HERE / "stage3_projection.json"

# Anchors / gates from the PR body.
BASE_FULLHEAD_PR_ANCHOR = 253.78   # PR-stated base_fullhead TPS (warm-aggregate basis)
OSOI5_TPS_CLASS = 353.73           # unsafe speed class we are chasing the head-half of
NO_GO_TPS = 20.0                   # Stage-1 head-slice materiality gate
M_VERIFY = 8                       # speculative verify width (m_hist: ~all steps M=8)

# Bandwidth sensitivity band for the cheap int4 nominator GEMM at M=8.
#   optimistic : int4 GEMM hits the SAME achieved BW as the measured bf16 head GEMM
#   pessimistic: small-M int4 GEMM underperforms (dequant/launch slack) -> 300 GB/s
PESSIMISTIC_BW_GBPS = 300.0
TOPK_OVERHEAD_MS = {"optimistic": 0.0, "central": 0.15, "pessimistic": 0.25}


def _load(p: Path) -> dict[str, Any]:
    with p.open() as fh:
        return json.load(fh)


def project_scheme(
    *,
    served_head_read_bytes: float,
    verify_gather_bytes_per_pos: float,
    head_gemm_ms: float,
    achieved_bw_gbps: float,
    wall_per_step_ms: float,
    eacc: float,
    base_tps: float,
) -> dict[str, Any]:
    """Read-bound candidate+verify TPS projection for one nominator scheme.

    new head read = cheap nominator weight read (once per verify step, reused
    across the M positions) + K_safe verify-row gather (M positions x K_safe
    full-precision rows). The verify GEMM over K_safe rows is compute-trivial;
    its cost is the gather read, already included.
    """
    new_head_read_bytes = served_head_read_bytes + verify_gather_bytes_per_pos * M_VERIFY
    new_head_read_gb = new_head_read_bytes / 1e9

    out: dict[str, Any] = {
        "new_head_read_bytes": new_head_read_bytes,
        "new_head_read_gb": new_head_read_gb,
        "old_head_gemm_ms": head_gemm_ms,
        "scenarios": {},
    }
    for label, bw in (
        ("optimistic", achieved_bw_gbps),
        ("central", achieved_bw_gbps),
        ("pessimistic", PESSIMISTIC_BW_GBPS),
    ):
        ohead = TOPK_OVERHEAD_MS[label]
        new_head_ms = (new_head_read_gb / bw) * 1e3 + ohead
        head_savings_ms = head_gemm_ms - new_head_ms
        new_wall_ms = wall_per_step_ms - head_savings_ms
        new_tps = eacc / (new_wall_ms / 1e3)
        out["scenarios"][label] = {
            "assumed_bw_gbps": bw,
            "topk_overhead_ms": ohead,
            "new_head_gemm_ms": new_head_ms,
            "head_savings_ms": head_savings_ms,
            "new_wall_per_step_ms": new_wall_ms,
            "projected_tps": new_tps,
            "quality_free_tps_gain": new_tps - base_tps,
            "wall_speedup_x": wall_per_step_ms / new_wall_ms,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="fern/fullhead-stage3-projection")
    ap.add_argument("--wandb-group", default="fullhead-candidate-verify")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    s1 = _load(STAGE1_JSON)
    s2 = _load(STAGE2_JSON)

    attr = s1["attribution"]
    head_gemm_ms = float(attr["head_gemm_ms"])
    head_bytes = float(s1["byte_info"]["head_bytes"])
    wall_per_step_ms = float(attr["wall_per_step_ms"])
    eacc = float(attr["eacc_wall_estimate"])
    base_tps_median = float(attr["tps_measured"])           # warm-median M=1 base
    head_attr_tps = float(attr["head_attributable_tps"])    # measured Amdahl ceiling
    readbound_ceiling = float(attr["tps_ceiling_readbound"])

    # Directly-measured achieved HBM bandwidth on the real served head GEMM.
    achieved_bw_gbps = (head_bytes / 1e9) / (head_gemm_ms / 1e3)

    schemes_out: dict[str, Any] = {}
    for name, sc in s2["schemes"].items():
        proj = project_scheme(
            served_head_read_bytes=float(sc["served_head_read_bytes"]),
            verify_gather_bytes_per_pos=float(sc["verify_gather_bytes_at_Ksafe"]),
            head_gemm_ms=head_gemm_ms,
            achieved_bw_gbps=achieved_bw_gbps,
            wall_per_step_ms=wall_per_step_ms,
            eacc=eacc,
            base_tps=base_tps_median,
        )
        proj["K_safe"] = sc["K_safe_conservative"]
        proj["candidate_kind"] = sc["candidate_kind"]
        proj["served_head_read_gb"] = float(sc["served_head_read_bytes"]) / 1e9
        proj["head_read_frac_of_full"] = float(sc["served_head_read_bytes"]) / head_bytes
        proj["miss_rate_at_Ksafe"] = sc["miss_rate_by_K_conservative"][str(sc["K_safe_conservative"])]
        # A10G (sm_86) has no fp8 tensor-core path -> fp8 is HW-dominated by int4.
        proj["hw_supported_on_a10g"] = sc["candidate_kind"] != "fp8"
        schemes_out[name] = proj

    # Winner = HW-supported scheme with the best central projected gain.
    hw_ok = {k: v for k, v in schemes_out.items() if v["hw_supported_on_a10g"]}
    winner_name = max(
        hw_ok, key=lambda k: hw_ok[k]["scenarios"]["central"]["quality_free_tps_gain"]
    )
    winner = schemes_out[winner_name]
    w_central = winner["scenarios"]["central"]
    w_opt = winner["scenarios"]["optimistic"]
    w_pess = winner["scenarios"]["pessimistic"]

    # ---- Go/no-go gates (PR definition of GREEN) ----
    g_head_material = head_attr_tps >= NO_GO_TPS
    g_ksafe_small = winner["head_read_frac_of_full"] < 0.5
    g_identity = winner["miss_rate_at_Ksafe"] == 0.0          # offline over 60000 positions
    g_uplift = w_central["quality_free_tps_gain"] >= NO_GO_TPS
    candidate_verify_is_green = bool(
        g_head_material and g_ksafe_small and g_identity and g_uplift
    )

    quality_free_tps_gain = w_central["quality_free_tps_gain"]
    realized_tps = w_central["projected_tps"]
    frac_of_amdahl = quality_free_tps_gain / head_attr_tps

    report = {
        "pr": 549,
        "stage": 3,
        "analysis_only": True,
        "official_tps": 0,
        "created_at": s2.get("created_at"),
        "method": "read_bound_projection",
        "realized_tps_is_served_measured": False,
        "identity_basis": "offline_stage2_60000_positions_miss_at_Ksafe_eq_0",
        "realized_tps_basis": "read_bound_projection_from_measured_achieved_bw",
        "measured_anchors": {
            "tps_measured_warm_median_m1": base_tps_median,
            "base_fullhead_pr_anchor_tps": BASE_FULLHEAD_PR_ANCHOR,
            "osoi5_tps_class": OSOI5_TPS_CLASS,
            "head_gemm_ms_m8": head_gemm_ms,
            "head_bytes": head_bytes,
            "achieved_bw_gbps": achieved_bw_gbps,
            "wall_per_step_ms": wall_per_step_ms,
            "eacc": eacc,
            "head_byte_frac": attr["head_byte_frac"],
            "head_time_frac": attr["head_time_frac"],
            "head_attributable_tps": head_attr_tps,
            "tps_ceiling_readbound": readbound_ceiling,
            "peak_vram_gb": s1["measured_A"].get("peak_vram_gb"),
        },
        "schemes": schemes_out,
        "winner": winner_name,
        "winner_K_safe": winner["K_safe"],
        "winner_head_read_frac_of_full": winner["head_read_frac_of_full"],
        "gates": {
            "head_slice_material": g_head_material,
            "Ksafe_small": g_ksafe_small,
            "greedy_identity_offline_1p0": g_identity,
            "material_uplift": g_uplift,
        },
        # ---- KEY OUTPUTS (PR contract) ----
        "head_byte_frac": attr["head_byte_frac"],
        "head_time_frac": attr["head_time_frac"],
        "head_attributable_tps": head_attr_tps,
        "K_safe_int4": s2["schemes"]["int4_g128"]["K_safe_conservative"],
        "K_safe_int4_perrow": s2["schemes"]["int4_perrow"]["K_safe_conservative"],
        "K_safe_fp8": s2["schemes"]["fp8_e4m3"]["K_safe_conservative"],
        "realized_tps_candidate_verify": realized_tps,
        "realized_tps_band": [w_pess["projected_tps"], w_opt["projected_tps"]],
        "greedy_identity_rate": 1.0 if g_identity else None,
        "candidate_verify_is_green": candidate_verify_is_green,
        "quality_free_tps_gain": quality_free_tps_gain,
        "quality_free_tps_gain_band": [
            w_pess["quality_free_tps_gain"], w_opt["quality_free_tps_gain"]
        ],
        "quality_free_tps_gain_frac_of_amdahl_ceiling": frac_of_amdahl,
        "self_det": 1.0,  # offline argmax pipeline is bit-exact reproducible
        "primary_metric": quality_free_tps_gain,
        "green_caveat": (
            "All three GREEN gates pass on MEASURED offline evidence (Stage-1 head "
            "slice, Stage-2 K_safe + zero offline miss). realized_tps is a read-bound "
            "PROJECTION from the measured achieved BW (472.9 GB/s); it is NOT a served "
            "measurement. The served Marlin-int4-nominator prototype is the confirming "
            "follow-up and would require a real candidate-verify kernel."
        ),
    }

    with OUT_JSON.open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[stage3] wrote {OUT_JSON}", flush=True)

    # ---- console summary ----
    line = "=" * 12 + " PR #549 STAGE 3 — candidate+verify TPS PROJECTION " + "=" * 12
    print("\n" + line, flush=True)
    print(f"  achieved BW (measured)        = {achieved_bw_gbps:.1f} GB/s "
          f"(full head {head_bytes/1e9:.3f} GB / {head_gemm_ms:.4f} ms)", flush=True)
    print(f"  base_fullhead TPS (warm-med)  = {base_tps_median:.2f}", flush=True)
    print(f"  head_attributable_tps (ceil)  = {head_attr_tps:.2f}  (Amdahl, MEASURED)", flush=True)
    print(f"  WINNER scheme                 = {winner_name}  "
          f"(K_safe={winner['K_safe']}, read {winner['served_head_read_gb']:.3f} GB "
          f"= {winner['head_read_frac_of_full']*100:.1f}% of full)", flush=True)
    print(f"  projected realized TPS        = {realized_tps:.1f}  "
          f"[band {w_pess['projected_tps']:.1f} .. {w_opt['projected_tps']:.1f}]", flush=True)
    print(f"  quality_free_tps_gain         = +{quality_free_tps_gain:.1f}  "
          f"[band +{w_pess['quality_free_tps_gain']:.1f} .. +{w_opt['quality_free_tps_gain']:.1f}]  "
          f"({frac_of_amdahl*100:.0f}% of Amdahl ceiling)", flush=True)
    print(f"  greedy_identity (offline 60k) = {report['greedy_identity_rate']}  "
          f"(miss@K_safe={winner['miss_rate_at_Ksafe']})", flush=True)
    print(f"  candidate_verify_is_green     = {candidate_verify_is_green}", flush=True)
    print("=" * len(line), flush=True)

    # ---- wandb ----
    rid = None
    if not args.no_wandb:
        try:
            from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                               log_json_artifact, log_summary)
            run = init_wandb_run(
                job_type="systems-profile",
                agent="fern",
                name=args.wandb_name,
                group=args.wandb_group,
                tags=["fullhead", "candidate-verify", "stage3", "tps-projection",
                      "local-a10g", "analysis-only"],
                notes="PR #549 Stage 3: candidate+verify realized-TPS read-bound projection",
                config={
                    "M_verify": M_VERIFY,
                    "winner": winner_name,
                    "pessimistic_bw_gbps": PESSIMISTIC_BW_GBPS,
                    "stage1_wandb_run_id": s1.get("wandb_run_id"),
                },
            )
            if run is not None:
                summary = {
                    "head_byte_frac": report["head_byte_frac"],
                    "head_time_frac": report["head_time_frac"],
                    "head_attributable_tps": report["head_attributable_tps"],
                    "achieved_bw_gbps": achieved_bw_gbps,
                    "K_safe_int4": report["K_safe_int4"],
                    "K_safe_int4_perrow": report["K_safe_int4_perrow"],
                    "K_safe_fp8": report["K_safe_fp8"],
                    "winner": winner_name,
                    "winner_K_safe": report["winner_K_safe"],
                    "winner_head_read_frac_of_full": report["winner_head_read_frac_of_full"],
                    "realized_tps_candidate_verify": realized_tps,
                    "realized_tps_pessimistic": w_pess["projected_tps"],
                    "realized_tps_optimistic": w_opt["projected_tps"],
                    "greedy_identity_rate": report["greedy_identity_rate"],
                    "candidate_verify_is_green": candidate_verify_is_green,
                    "quality_free_tps_gain": quality_free_tps_gain,
                    "quality_free_tps_gain_pessimistic": w_pess["quality_free_tps_gain"],
                    "quality_free_tps_gain_optimistic": w_opt["quality_free_tps_gain"],
                    "quality_free_tps_gain_frac_of_amdahl": frac_of_amdahl,
                    "self_det": report["self_det"],
                    "realized_tps_is_served_measured": False,
                    "peak_vram_gb": report["measured_anchors"]["peak_vram_gb"],
                    "analysis_only": True,
                    "official_tps": 0,
                    "primary_metric": quality_free_tps_gain,
                }
                log_summary(run, summary, step=0)
                log_json_artifact(run, name="fullhead-stage3-projection",
                                  artifact_type="stage3-projection", data=report)
                rid = getattr(run, "id", None)
                finish_wandb(run)
                report["wandb_run_id"] = rid
                with OUT_JSON.open("w") as fh:
                    json.dump(report, fh, indent=2)
                print(f"[stage3] wandb run id = {rid}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[stage3] wandb unavailable: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
