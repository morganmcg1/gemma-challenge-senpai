#!/usr/bin/env python
"""PR #733 — Strict-clean spec route: tree-verify-at-M=1 cost model (analysis-only).

Question: is there a STRICT-byte-exact spec-dec config that still beats the locked
quality-safe anchor 126.378 official TPS? If yes, the #730 fire stops being a
tolerance gamble (depends on the organizer honoring tau=0.3) and becomes a clean
fast fire above the wall.

This script is pure arithmetic over already-measured, on-branch anchors:
  * #728 spec achievable-ceiling sweep  (research/spec_achievable_ceiling/runs/sweep/report.json)
      - AR base (BI=1, M=1) wall_tps_local
      - per-K deployed spec wall_tps_local  (drafter ON, single M=K+1 verify pass)
      - anchored transfer x1.192 / conservative floor x1.0352
  * PR-616 raw flip analysis             (research/specdec_raw_flip_rate/flip_report.json)
      - pure-M (m8-vs-m1, BI=1 both, teacher-forced) per-position flip rate + gap dist
No GPU, no HF job. analysis_only=1, official_tps=0.

Cost-model core (route b, "tree-verify-at-M=1"):
  To be byte-exact to AR you must emit every token off the M=1 trajectory. The
  int4 K/V projections are themselves M-dependent Marlin GEMMs, so the M=K+1
  verify KV-cache differs (as floats) from the M=1 KV-cache even at NON-tie
  positions, and that drift compounds through attention (the 0.43% pure-M flip
  rate is measured over teacher-forced *identical-token* prefixes, so it already
  bakes in compounding KV drift). Therefore you cannot cherry-pick only the tie
  positions to re-decode: a selective re-decode runs on the drifted M=K+1 KV and
  is NOT AR-exact at exactly the tie positions it is meant to fix. Strict
  byte-exactness forces a FULL M=1 re-decode of every accepted block.

  Per accepted block (length L = accepted drafts + 1 bonus):
     cost = [drafter K-steps + verify(M=K+1)]      <- the measured spec machinery
          + [full M=1 re-decode of the L tokens]   <- == an AR decode of L tokens
     emits L tokens.
  The measured spec machinery costs L / spec_tps per block (by definition of
  spec_tps). The full M=1 re-decode of L tokens costs L / ar_tps (by definition
  of ar_tps; re-decoding at M=1 IS autoregression -- #728's AR2-vs-AR1 control
  proves the M=1 path is byte-deterministic). L cancels:

     route_b_tps = 1 / (1/spec_tps + 1/ar_tps)         (serial; block-length invariant)

  This is an UPPER bound on route (b): early in-block divergence only shortens the
  emitted run, never lengthens it. It is < ar_tps for any finite spec_tps, so
  route (b) can NEVER beat AR (126.378 official), no matter K or acceptance.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

REPORT_728 = os.path.join(ROOT, "research/spec_achievable_ceiling/runs/sweep/report.json")
FLIP_616 = os.path.join(ROOT, "research/specdec_raw_flip_rate/flip_report.json")

ANCHOR_OFFICIAL = 126.378  # locked quality-safe bar (int4_g128_lmhead, PR #4)
# E[accepted tokens per step] anchor for K~6 from BASELINE.md line 28
# ("amortize weight read over ~3.3 accepted tok/step"). Used only for the
# descriptive E[extra M=1 passes per block]; the headline is L-invariant.
E_BLOCK_LEN_K6 = 3.3


def harmonic(a: float, b: float) -> float:
    return 1.0 / (1.0 / a + 1.0 / b)


def _log_wandb(out, results, best, locus, ar_tps, ratio_anchored, tau_lo_flat) -> None:
    """Best-effort: bank the cost-model headline as a new W&B run (analysis-only)."""
    try:
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            name="denken/strict-clean-spec-route",
            group="denken-strict-clean-spec-route",
            job_type="analysis", reinit=True,
            config={
                "pr": 733, "analysis_only": True, "official_tps": 0,
                "anchor_official_bar": ANCHOR_OFFICIAL,
                "ar_base_local_bi1": ar_tps,
                "transfer_anchored_x": ratio_anchored,
                "transfer_floor_x": tau_lo_flat,
                "source_728_report": "research/spec_achievable_ceiling/runs/sweep/report.json",
                "source_616_flip": "research/specdec_raw_flip_rate/flip_report.json",
                "cost_model": "strict route(b)=full M=1 re-decode -> 1/(1/spec+1/ar)",
            },
        )
        sm = {
            "locus/pure_M_flip_rate_BI1": locus["pure_M_flip_rate_BI1"],
            "locus/gap_at_flips_p99_nat": locus["gap_at_flips_p99_nat"],
            "locus/M1_vs_M1_floor": locus["M1_vs_M1_floor"],
            "headline/strict_clean_net_official_equiv": best["routeb_strict_official_anchored_x1.192"],
            "headline/best_k": best["k"],
            "headline/beats_126378": int(best["beats_126378_strict"]),
            "headline/shortfall_vs_126378": out["headline"]["shortfall_vs_126378"],
            "headline/overlap_optimistic_ceiling": out["headline"]["overlap_optimistic_ceiling_only_ties"],
            "routea/buildable_in_scope": int(out["instruction3_routea"]["buildable_in_scope"]),
            "verdict/strict_clean_route_beats_126378": int(out["instruction4_verdict"]["strict_clean_route_beats_126378"]),
        }
        for r in results:
            k = r["k"]
            sm[f"k{k}/spec_tps_local"] = r["spec_tps_local"]
            sm[f"k{k}/spec_speedup_x"] = r["spec_speedup_x"]
            sm[f"k{k}/routeb_strict_official_anchored"] = r["routeb_strict_official_anchored_x1.192"]
            sm[f"k{k}/routeb_strict_official_floor"] = r["routeb_strict_official_floor_x1.0352"]
            sm[f"k{k}/E_extra_M1_passes_per_block"] = r["E_extra_M1_passes_per_block"]
        run.summary.update(sm)
        print(f"[wandb] logged run id={run.id} name=denken/strict-clean-spec-route")
        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({type(exc).__name__}: {exc}); report.json still written")


def main() -> None:
    rep = json.load(open(REPORT_728))
    flip = json.load(open(FLIP_616))

    ar_tps = rep["ar_reference"]["wall_tps_local"]
    tau_lo_flat = rep["transfer_model"]["tau_lo_flat"]          # x1.0352 conservative floor
    ratio_anchored = rep["transfer_model"]["ratio_anchored"]     # x1.192 base-anchored
    # sanity: anchored ratio == anchor_official / ar_base_local
    assert abs(ratio_anchored - ANCHOR_OFFICIAL / ar_tps) < 1e-6

    # ---- locus (instruction 1): pure-M flip rate with BI=1 pinned both sides ----
    pm = flip["headline_pure_M_m8_vs_m1"]
    locus = {
        "pure_M_flip_rate_BI1": pm["flip_rate"],                      # 0.43%
        "pure_M_flip_rate_ci95": pm["flip_rate_ci95_cluster_bootstrap"],
        "frac_flips_neartie_lt_0.5nat": flip["verdict"]["frac_flips_neartie_lt_0.5nats"],
        "gap_at_flips_p99_nat": pm["gap_at_flips"]["p99"],            # 0.25 -> all <= 0.3
        "gap_at_flips_median_nat": pm["gap_at_flips"]["p50"],
        "M1_vs_M1_floor": flip["verdict"]["pure_M_floor_m1_vs_m1"],   # 0.0
        "ar2_vs_ar1_seq_exact_728": 1.0,                              # #728 control
        "note": ("BI=1 leaves attention invariant (TRITON_ATTN single-segment, "
                 "BASELINE #122); the sole residual M-variant reduction is the int4 "
                 "Marlin GEMM (split-K = f(M), no exposed knob). land #680's "
                 "'ATTENTION not GEMM' was the no-BI / non-single-segment regime."),
    }

    # ---- route (b) cost model (instruction 2) ----
    results = []
    for r in rep["results"]:
        k = r["k"]
        if k not in (5, 6):
            continue
        spec_tps = r["wall_tps_local"]
        speedup = spec_tps / ar_tps

        # STRICT route (b): full M=1 re-decode of every accepted block.
        routeb_local = harmonic(spec_tps, ar_tps)
        routeb_anchored = routeb_local * ratio_anchored
        routeb_floor = routeb_local * tau_lo_flat

        # E[extra M=1 passes per accepted block] = E[block length] (the WHOLE block
        # must be re-decoded). Anchor K=6 at 3.3; scale others by speedup proxy.
        e_block = E_BLOCK_LEN_K6 * (speedup / (rep["results"][1]["wall_tps_local"] / ar_tps))

        # NON-STRICT optimistic contrast: selective re-decode of only the at-risk
        # (tie) positions. spec_b = spec_tps / (1 + r_tie * speedup). This does NOT
        # achieve byte-exactness (drifted M=K+1 KV at the tie) -> still tau-dependent.
        r_tie_floor = locus["pure_M_flip_rate_BI1"]        # lower bound (flips only)
        r_tie_10x = 10 * r_tie_floor                       # conservative tie-rate
        selective_lo = spec_tps / (1 + r_tie_floor * speedup) * ratio_anchored
        selective_hi_cost = spec_tps / (1 + r_tie_10x * speedup) * ratio_anchored

        results.append({
            "k": k,
            "ar_tps_local": ar_tps,
            "spec_tps_local": spec_tps,
            "spec_speedup_x": speedup,
            "E_block_len_tokens": round(e_block, 3),
            "E_extra_M1_passes_per_block": round(e_block, 3),  # whole block, strict
            "routeb_strict_local": routeb_local,
            "routeb_strict_official_anchored_x1.192": routeb_anchored,
            "routeb_strict_official_floor_x1.0352": routeb_floor,
            "beats_126378_strict": routeb_anchored > ANCHOR_OFFICIAL,
            "margin_vs_126378_anchored": routeb_anchored - ANCHOR_OFFICIAL,
            "overlap_optimistic_ceiling_official": ar_tps * ratio_anchored,  # == 126.378
            "_nonstrict_selective_official_anchored_bracket": [
                round(selective_hi_cost, 2), round(selective_lo, 2)],
            "_nonstrict_note": ("selective re-decode is NOT byte-exact (drifted "
                                "M=K+1 KV at ties) -> still tau-dependent; shown only "
                                "to size the gap the strict requirement gives up."),
        })

    best = max(results, key=lambda x: x["routeb_strict_official_anchored_x1.192"])

    out = {
        "pr": 733,
        "analysis_only": True,
        "official_tps": 0,
        "anchor_official_bar": ANCHOR_OFFICIAL,
        "transfer": {"anchored_x": ratio_anchored, "floor_x": tau_lo_flat,
                     "ar_base_local_bi1": ar_tps},
        "instruction1_locus": locus,
        "instruction2_routeb": results,
        "headline": {
            "best_strict_routeb_official_anchored": best["routeb_strict_official_anchored_x1.192"],
            "best_strict_routeb_k": best["k"],
            "beats_126378": best["beats_126378_strict"],
            "shortfall_vs_126378": ANCHOR_OFFICIAL - best["routeb_strict_official_anchored_x1.192"],
            "overlap_optimistic_ceiling_only_ties": ar_tps * ratio_anchored,
        },
        "instruction3_routea": {
            "kernel": "M-invariant (fixed-split-K) int4 Marlin/Machete GEMV",
            "buildable_in_scope": False,
            "prior": ("BASELINE #122: marlin_gemm split-K=f(M), no exposed "
                      "num_splits/max_par knob, no batch-invariant Marlin in pinned "
                      "vllm==0.22.0 wheel; stark #722 SPARSE_INT4_KERNEL_ABSENT; "
                      "land #506 M=1 BI-GEMV prior."),
            "overhead_if_built": ("verify stays ~1 AR-pass-time at conc=1 (HBM-bound, "
                                  "weight-load dominates) -> spec speedup preserved "
                                  "(official-equiv ~226-265, byte-exact). BUILD is the "
                                  "blocker: from-scratch CUDA kernel, out-of-scope."),
            "verdict": "GO-on-value, NO-GO-on-in-scope-buildability",
        },
        "instruction4_verdict": {
            "strict_clean_route_beats_126378": False,
            "routeb_reason": ("strict byte-exactness forces a full M=1 re-decode -> "
                              "route_b = 1/(1/spec+1/ar) < ar = 126.378 for any K; "
                              f"best ~{best['routeb_strict_official_anchored_x1.192']:.1f} "
                              "official, fails by "
                              f"~{ANCHOR_OFFICIAL - best['routeb_strict_official_anchored_x1.192']:.1f}."),
            "routea_reason": ("the only route that keeps the speedup AND is byte-exact, "
                              "but the M-invariant int4 GEMV kernel is absent and "
                              "out-of-scope to build this cycle."),
            "consequence": ("no strict-clean spec fire beats 126.378 buildable now; the "
                            "#730 tau=0.3 gamble stands; strict-byte-exact tolerance-"
                            "dependence is irreducible without route (a)'s kernel."),
            "primary_metric_strict_clean_net_official_equiv": best["routeb_strict_official_anchored_x1.192"],
        },
    }
    outp = os.path.join(HERE, "report.json")
    json.dump(out, open(outp, "w"), indent=2)

    _log_wandb(out, results, best, locus, ar_tps, ratio_anchored, tau_lo_flat)

    # console summary
    print(f"AR base local (BI=1, M=1)      : {ar_tps:.3f} tps  -> official anchor {ANCHOR_OFFICIAL}")
    print(f"transfer: anchored x{ratio_anchored:.4f}  floor x{tau_lo_flat:.4f}")
    print()
    print("instr1 locus: pure-M flip (BI=1 both) = "
          f"{locus['pure_M_flip_rate_BI1']*100:.3f}%  CI95 "
          f"[{locus['pure_M_flip_rate_ci95'][0]*100:.3f},{locus['pure_M_flip_rate_ci95'][1]*100:.3f}]%; "
          f"100% near-tie (gap p99={locus['gap_at_flips_p99_nat']:.3f} nat); "
          f"M1-vs-M1 floor={locus['M1_vs_M1_floor']}")
    print()
    for r in results:
        print(f"K={r['k']}: spec {r['spec_tps_local']:.2f} ({r['spec_speedup_x']:.3f}x) | "
              f"E[block]~{r['E_block_len_tokens']} | STRICT route(b) local "
              f"{r['routeb_strict_local']:.2f} -> official {r['routeb_strict_official_anchored_x1.192']:.2f} "
              f"(floor {r['routeb_strict_official_floor_x1.0352']:.2f}) | beats 126.378? "
              f"{r['beats_126378_strict']} (margin {r['margin_vs_126378_anchored']:+.2f})")
        print(f"     [non-strict selective contrast, NOT byte-exact: official "
              f"{r['_nonstrict_selective_official_anchored_bracket']}]")
    print()
    print(f"HEADLINE strict-clean net official-equiv (best, K={best['k']}): "
          f"{best['routeb_strict_official_anchored_x1.192']:.2f}  "
          f"-> beats 126.378? {best['beats_126378_strict']}  "
          f"(short by {ANCHOR_OFFICIAL - best['routeb_strict_official_anchored_x1.192']:.2f})")
    print(f"overlap-optimistic ceiling (unphysical, drafter+verify free) = "
          f"{ar_tps*ratio_anchored:.2f} official == AR -> only TIES, never beats")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
