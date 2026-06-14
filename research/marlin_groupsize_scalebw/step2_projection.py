#!/usr/bin/env python
"""PR #140 Step-2 — scale-byte + TPS projection for the servable int4 body group sizes.

Consumes the Step-1 PPL scan (ppl_scan_results.json) and projects, for each PPL-passing
group size, the scale-byte reduction -> wall_tps delta -> official TPS, using the deployed
osoi5-v0-baked basis and the fleet's measured anchors:

  deployed g=128 core-7 verify-GEMM scales : 53.70 MB = 3.06% of the 1754.7 MB int4 body
                                             (= 26.8M FP16 scales)               [PR #104]
  measured scale-byte -> TPS transfer       : 43.0% scale saving -> +0.3% TPS    [PR #110]
                                             (scales ~80% un-overlapped/BW-critical)
  step model / compose                      : official = K_cal * E[T]/step * tau,
                                             K_cal=125.268, verify-GEMM=0.53 of step,
                                             local->official x1.06019              [fern #100/#99]
  baseline                                  : 481.53 official TPS (fa2sw_precache_kenyan)

The g=128 -> g(target) scale-elem RATIO is taken from the measured full-model scan (architecture
-robust: ratio = 128/in_dim averaged over the 343 modules) and applied to the deployed 53.70 MB.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# --- deployed / fleet anchors ---
OFFICIAL_BASELINE = 481.53          # fa2sw_precache_kenyan, PR #52
DEPLOYED_SCALE_MB_G128 = 53.70      # PR #104, folded osoi5-v0-baked core-7
INT4_BODY_MB = 1754.7               # PR #104
PALETTE_SAVE_FRAC = 0.43            # PR #110 measured scale-byte saving
PALETTE_TPS_PCT = 0.30              # PR #110 measured central TPS gain (%) for the 43% saving
PALETTE_TPS_PCT_UPPER = 0.50        # PR #110 upper bound
VERIFY_SHARE = 0.53                 # fern #100 verify-GEMM step share
SCALE_UNOVERLAP = 0.80              # PR #110 fraction of scale bytes on the BW-critical path
PPL_CAP = 2.42
ANCHOR_OFFLINE_G128 = 2.3812        # wirbel #118 cap-comparable offline g=128 anchor (== deployed body)


def tps_gain_from_scale_saving(save_frac: float, mode: str = "anchor") -> float:
    """Return projected official-TPS gain fraction for eliminating `save_frac` of the g=128 scales."""
    if mode == "anchor":            # linear from PR #110 measured 43% -> 0.30% central
        return (PALETTE_TPS_PCT / 100.0) * (save_frac / PALETTE_SAVE_FRAC)
    if mode == "anchor_upper":      # linear from PR #110 upper 43% -> 0.50%
        return (PALETTE_TPS_PCT_UPPER / 100.0) * (save_frac / PALETTE_SAVE_FRAC)
    if mode == "first_principles":  # fern #100 BW model: verify slice * unoverlap * byte fraction
        weight_mb = INT4_BODY_MB + DEPLOYED_SCALE_MB_G128
        return VERIFY_SHARE * SCALE_UNOVERLAP * (save_frac * DEPLOYED_SCALE_MB_G128 / weight_mb)
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", default="research/marlin_groupsize_scalebw/ppl_scan_results.json")
    ap.add_argument("--out", default="research/marlin_groupsize_scalebw/step2_projection.json")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    scan = json.load(open(args.scan))
    # scale-elem counts straight from the measured scan (full 42L model; ratios are arch-robust)
    elems = {g: scan[g]["scale_elems"] for g in scan}
    ppl = {g: scan[g]["ppl"] for g in scan}
    raw_verdict = {g: scan[g]["verdict"] for g in scan}
    base_g = "128"
    assert base_g in elems, "need g=128 control in the scan"

    # Cap-comparability: our scan base is the official-g32 checkpoint dequantized+requantized,
    # which reads ~0.29 PPL *optimistic* vs the established offline g=128 anchor (qat_unq->g128,
    # wirbel #118 = 2.3812). The g=128->g(target) DELTA is pipeline-invariant (same base, same
    # fake-quant both arms), so we pin our g=128 to the anchor and carry the delta:
    #     cap_comparable_ppl(g) = ppl(g) + (ANCHOR_OFFLINE_G128 - ppl(g128))
    # and gate THAT against the 2.42 cap. Equivalently: anchor + (ppl(g) - ppl(g128)).
    offset = ANCHOR_OFFLINE_G128 - ppl[base_g]

    rows = []
    for g in scan:
        ratio = elems[g] / elems[base_g]                 # scale bytes vs g=128
        deployed_scale_mb = DEPLOYED_SCALE_MB_G128 * ratio
        save_frac = max(0.0, 1.0 - ratio)                # fraction of g=128 scales eliminated
        gain = tps_gain_from_scale_saving(save_frac, "anchor")
        gain_up = tps_gain_from_scale_saving(save_frac, "anchor_upper")
        gain_fp = tps_gain_from_scale_saving(save_frac, "first_principles")
        proj = OFFICIAL_BASELINE * (1 + gain)
        cap_ppl = ppl[g] + offset
        cap_verdict = "PASS" if cap_ppl <= PPL_CAP else "FAIL"
        rows.append({
            "group_size": int(g),
            "ppl_raw_harness": round(ppl[g], 4),
            "ppl_cap_comparable": round(cap_ppl, 4),
            "verdict_raw_harness": raw_verdict[g],
            "verdict": cap_verdict,                       # cap-comparable verdict drives selection
            "scale_mb_deployed_basis": round(deployed_scale_mb, 2),
            "scale_save_vs_g128_pct": round(save_frac * 100, 2),
            "tps_gain_pct_central": round(gain * 100, 3),
            "tps_gain_pct_upper": round(gain_up * 100, 3),
            "tps_gain_pct_first_principles": round(gain_fp * 100, 3),
            "official_tps_proj_if_passing": round(proj, 2),
            "clears_500_if_passing": int(proj >= 500.0),
        })

    # "best" for the scale-byte direction = the PPL-passing g that eliminates the MOST scale bytes,
    # but never a g FINER than the deployed g=128 (g=32 adds scale bytes -> wrong direction). The
    # deployed g=128 is always an eligible no-change candidate; a coarser g must strictly beat it.
    candidates = [r for r in rows if r["verdict"] == "PASS"
                  and (r["group_size"] == 128 or r["scale_save_vs_g128_pct"] > 0)]
    best = max(candidates, key=lambda r: r["scale_save_vs_g128_pct"]) if candidates else None
    best_g = best["group_size"] if best else 128
    primary = best["official_tps_proj_if_passing"] if best else OFFICIAL_BASELINE
    clears = int(primary >= 500.0)

    out = {
        "rows": rows,
        "best_ppl_passing_groupsize": best_g,
        "groupsize_scalebw_official_tps_proj": round(primary, 2),
        "groupsize_clears_500": clears,
        "ppl_cap": PPL_CAP,
        "anchor_offline_g128": ANCHOR_OFFLINE_G128,
        "harness_offset_applied": round(offset, 4),
        "cap_headroom_above_anchor": round(PPL_CAP - ANCHOR_OFFLINE_G128, 4),
        "note": ("g=256 UNSERVABLE (Marlin max group=128); only g=-1 is coarser-than-128. "
                 "Verdicts are CAP-COMPARABLE: raw harness PPL pinned to the #118 offline g=128 "
                 "anchor (2.3812) via the pipeline-invariant g128->g delta. The cap leaves only "
                 f"{PPL_CAP - ANCHOR_OFFLINE_G128:.4f} PPL headroom above the g=128 anchor, and "
                 "coarsening to g=-1 costs far more than that -> g=-1 FAILS the cap. With no "
                 "servable coarser g passing, best stays g=128 -> no change (481.53). g=-1's "
                 "scale-byte slice is anyway dominated by the LOSSLESS palette (#110, bit-exact, "
                 "0.3% TPS) which captures it without the PPL cost."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))

    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                         entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                         name="ubel/marlin-groupsize-scalebw-proj", group="marlin-groupsize-scalebw",
                         config={"baseline_official_tps": OFFICIAL_BASELINE,
                                 "deployed_scale_mb_g128": DEPLOYED_SCALE_MB_G128, "ppl_cap": PPL_CAP})
        tbl = wandb.Table(columns=list(rows[0].keys()))
        for r in rows:
            tbl.add_data(*r.values())
        wandb.log({"groupsize_scalebw_projection": tbl})
        wandb.summary["best_ppl_passing_groupsize"] = best_g
        wandb.summary["groupsize_scalebw_official_tps_proj"] = round(primary, 2)
        wandb.summary["groupsize_clears_500"] = clears
        wandb.summary["anchor_offline_g128"] = ANCHOR_OFFLINE_G128
        wandb.summary["harness_offset_applied"] = round(offset, 4)
        run.finish()
        print(f"[wandb] projection run {run.id}")


if __name__ == "__main__":
    main()
