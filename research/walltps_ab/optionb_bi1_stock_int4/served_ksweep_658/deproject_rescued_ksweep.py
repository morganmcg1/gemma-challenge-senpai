#!/usr/bin/env python3
"""PR #658 -- Served wall-TPS K-sweep on the RESCUED (recompute-acceptor ON) stack.

Question: the shipped Option-B manifest fixes NUM_SPECULATIVE_TOKENS=6 (K=6 / M=7
verify). K is a free serving knob -- does the *rescued* served wall-TPS peak at a
K other than 6, while still holding #319 byte-exact identity?

The "rescued" served wall-TPS is a DE-PROJECTION (stark #636/#642's harness): the
live in-engine M=1-recompute acceptor is not a clean runnable wall-TPS number, so
we price it from independently-measured land components, all drawn from the SAME
int4_mtp_batchinv BI=1 served captures so the de-projection is self-consistent:

    rescued_wall_tps(K) = 1 / ( 1/U(K) + f(K) * t_recompute )

  * U(K)         un-rescued spec wall-TPS, LOCAL  ........ land #632 (this dir's ksweep)
  * f(K)         recompute fire-rate per emitted token ... land #648 (tau=0.5, K-indep 7.27%)
  * t_recompute  one M=1 AR forward, sec/forward ......... 1/A, A = local M=1 AR-rung
                 A_local = 77.96 tps (this dir's ar_ref_bi1, == stark arm-d 77.89)

Two per-fire conventions are reported:
  * rescued_local      t_recompute = 1/A_local (77.96)   -> the clean LOCAL number (HEADLINE)
  * rescued_starkmix   t_recompute = 1/126.378 (official) -> stark #636's mixed formula,
                       reproduced ONLY so K=6 is directly comparable to his de-projection.

Identity per K is REUSED from land #651 (served-rescue census, ar_ref_m1_canonical
oracle #654): on-AR head break-rate + confident off-AR miss (>0.5 nat) rate. A K
that BREAKS identity (a confident miss) is not a valid faster-K candidate.

LOCAL only. analysis_only=true, official_tps=0. NO HF Job, NO submission change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KSWEEP = HERE.parent / "ksweep"
FIRE = HERE.parent / "fire_census"
RESCUE = HERE.parent / "rescue_census"
ROOT = HERE.parents[3]

# reuse #632's analyzer verbatim for U(K) -- do NOT reimplement the curve math
sys.path.insert(0, str(KSWEEP))
import analyze_ksweep as A  # noqa: E402

LOCKED_RUNG = 126.378          # strict-#319 AR official rung (int4_g128_lmhead, #4)
SHIPPED_K = 6                  # the manifest's NUM_SPECULATIVE_TOKENS
SWEEP_KS = (5, 6, 7)           # the card's required explore+anchor arms

# --- stark #642 published numbers (from PR #658 body; his branch not inspected) ---
STARK_UNRESCUED_K6 = 155.58    # stark's un-rescued K=6 Option-B ceiling, LOCAL
STARK_AR_REF = 77.89           # stark's arm-d w4a16-ct M=1 AR reference, LOCAL
STARK_636_RESCUED_K7 = 139.20  # stark #636 projected rescued (K=7 base 152.291, f=7.81%)


# Banked AR-rung constant (65536 / 840.615 s) from #651's ar_ref_bi1 capture --
# the int4_mtp_batchinv spec-OFF M=1 AR reference, == stark #642 arm-d 77.89. Used
# as the fallback when the (untracked) #632 capture summary is absent, so #658 stays
# self-contained and re-runnable from the banked inputs alone.
AR_RUNG_TPS_BANKED = 65536 / 840.6150350570679


def local_ar_rung_tps() -> float:
    """Local M=1 AR-rung wall-TPS on the SAME int4_mtp_batchinv stack (spec OFF).

    = num_completion_tokens / decode duration from #651's ar_ref_bi1 capture.
    This is the per-recompute-forward cost anchor (one fire == one M=1 forward).
    Falls back to the banked constant if the source summary is not present.
    """
    p = KSWEEP / "ar_ref_bi1" / "decode_summary.json"
    if p.exists():
        s = json.loads(p.read_text())
        return s["num_completion_tokens"] / s["duration_s"]
    return AR_RUNG_TPS_BANKED


def fire_rate_by_k() -> dict[int, float]:
    """Recompute fire-rate per emitted token at tau=0.5, from land #648.

    Measured at K=3/5/7 (K-independent, spread 0.00017pp). K=6 (and K=4) are
    interpolated as the mean of the bracketing measured K -- immaterial given
    K-independence, but flagged as interpolated below.
    """
    d = json.loads((FIRE / "fire_census_result.json").read_text())
    pk = {int(k): float(v) for k, v in d["per_k_fire_frac"].items()}  # {3,5,7}
    pk[6] = 0.5 * (pk[5] + pk[7])
    pk[4] = 0.5 * (pk[3] + pk[5])
    return pk


def fire_rate_measured_ks() -> set[int]:
    d = json.loads((FIRE / "fire_census_result.json").read_text())
    return {int(k) for k in d["per_k_fire_frac"]}


def identity_by_k() -> dict[int, dict]:
    """#319 identity per K from land #651 served-rescue census (#654 oracle).

    on_AR_head_break_rate + confident off-AR head miss (>0.5 nat) rate. The
    census ran K=3/5/7; K=6 is interpolated for the break-rate, but the decisive
    field -- confident_off_AR_head_misses -- is 0 at EVERY measured K, so K=6
    inherits 0 (K-independent zero, the byte-exact-at-int4-floor invariant)."""
    d = json.loads((RESCUE / "rescue_census_final.json").read_text())
    pk = d["per_k"]
    out: dict[int, dict] = {}
    for tag, row in pk.items():
        out[int(row["k"])] = {
            "on_AR_head_fires": row["on_AR_head_fires"],
            "on_AR_head_breaks": row["on_AR_head_breaks"],
            "on_AR_head_break_rate": row["on_AR_head_break_rate"],
            "confident_off_AR_head_misses": row["head_confident_off_AR_misses"],
            "measured": True,
        }
    # interpolate K=6 break-rate from K5,K7; confident misses are an exact 0 invariant
    if 6 not in out and 5 in out and 7 in out:
        out[6] = {
            "on_AR_head_fires": None,
            "on_AR_head_breaks": None,
            "on_AR_head_break_rate": 0.5 * (out[5]["on_AR_head_break_rate"]
                                            + out[7]["on_AR_head_break_rate"]),
            "confident_off_AR_head_misses": 0,  # K-independent zero (#651/#654)
            "measured": False,
        }
    return out


def deproject(U: float, f: float, t_recompute: float) -> float:
    """rescued_wall_tps = 1 / (1/U + f * t_recompute)."""
    return 1.0 / (1.0 / U + f * t_recompute)


def main() -> int:
    A_local = local_ar_rung_tps()
    t_local = 1.0 / A_local
    t_official = 1.0 / LOCKED_RUNG

    urows = {r["K"]: r for r in A.summarize(A.collect())}   # un-rescued U(K), e_accept
    fk = fire_rate_by_k()
    measured_fk = fire_rate_measured_ks()
    idk = identity_by_k()

    rows = []
    for K in sorted(urows):
        U = urows[K]["wall_tps_median"]
        f = fk[K]
        rl = deproject(U, f, t_local)
        rs = deproject(U, f, t_official)
        idd = idk.get(K, {})
        rows.append({
            "K": K,
            "unrescued_wall_tps_local": round(U, 3),
            "e_accept_mean": round(urows[K]["e_accept_mean"], 4),
            "fire_rate_tau0p5": round(f, 6),
            "fire_rate_measured": K in measured_fk,
            "rescued_local": round(rl, 3),
            "rescued_starkmix_official_fire": round(rs, 3),
            "on_AR_head_break_rate": (round(idd["on_AR_head_break_rate"], 5)
                                      if idd.get("on_AR_head_break_rate") is not None else None),
            "confident_off_AR_head_misses": idd.get("confident_off_AR_head_misses"),
            "identity_measured": idd.get("measured"),
            # None when identity not measured/interpolated at this K (e.g. K=4 context row)
            "identity_holds": (None if idd.get("confident_off_AR_head_misses") is None
                               else idd.get("confident_off_AR_head_misses") == 0),
        })

    sweep = [r for r in rows if r["K"] in SWEEP_KS]
    kstar_row = max(sweep, key=lambda r: r["rescued_local"])
    k6_row = next(r for r in sweep if r["K"] == SHIPPED_K)
    kstar = kstar_row["K"]
    best_local = kstar_row["rescued_local"]

    # all sweep Ks hold identity (0 confident misses) -> no identity-break candidate
    all_identity_holds = all(r["identity_holds"] for r in sweep)
    beats_k6 = kstar != SHIPPED_K and best_local > k6_row["rescued_local"]

    if not all_identity_holds:
        verdict = "K_SWEEP_IDENTITY_BREAK"
    elif beats_k6:
        verdict = "FASTER_K_EXISTS"
    else:
        verdict = "K6_IS_OPTIMAL"

    # ---- cross-validation vs stark #642 (K=6) ----
    my_unrescued_k6 = urows[SHIPPED_K]["wall_tps_median"]
    my_rescued_k6_starkmix = k6_row["rescued_starkmix_official_fire"]
    xval = {
        "ar_ref_local": {"land": round(A_local, 3), "stark_arm_d": STARK_AR_REF,
                         "abs_pct_gap": round(100 * abs(A_local - STARK_AR_REF) / STARK_AR_REF, 2)},
        "unrescued_k6_local": {"land_632": round(my_unrescued_k6, 3), "stark_642": STARK_UNRESCUED_K6,
                               "abs_pct_gap": round(100 * abs(my_unrescued_k6 - STARK_UNRESCUED_K6)
                                                    / STARK_UNRESCUED_K6, 2)},
        "rescued_k6_starkmix": {"land_deproject": round(my_rescued_k6_starkmix, 3),
                                "note": "stark #642 rescued-K6 headline still PENDING; nearest "
                                        "anchor is his #636 K7 projection 139.20"},
        "note": ("AR references agree (<0.1%); the un-rescued K=6 gap is the de-projection's "
                 "harness sensitivity. land #632 wall_tps runs ~9% hotter than stark's K=6 "
                 "ceiling, so the absolute rescued level carries ~that uncertainty before an "
                 "OFFICIAL HF benchmark resolves it."),
    }

    out = {
        "pr": 658,
        "analysis_only": True,
        "official_tps": 0,
        "deprojection_formula": "rescued_wall_tps(K) = 1/(1/U(K) + f(K)/A)",
        "A_local_ar_rung_tps": round(A_local, 4),
        "locked_rung_official": LOCKED_RUNG,
        "shipped_K": SHIPPED_K,
        "sweep_Ks": list(SWEEP_KS),
        "rows": rows,
        "decision": {
            "verdict": verdict,
            "k_star": kstar,
            "served_walltps_best_K_local": best_local,
            "served_walltps_best_K_which": kstar,
            "k6_rescued_local": k6_row["rescued_local"],
            "kstar_vs_k6_local_tps": round(best_local - k6_row["rescued_local"], 3),
            "kstar_vs_k6_local_pct": round(100 * (best_local - k6_row["rescued_local"])
                                           / k6_row["rescued_local"], 3),
            "all_sweep_Ks_hold_identity": all_identity_holds,
        },
        "crossval_stark_642": xval,
    }
    (HERE / "deproject_rescued_ksweep.json").write_text(json.dumps(out, indent=2))

    # ---- human-readable table ----
    print(f"A_local (M=1 AR-rung) = {A_local:.3f} tps   t_recompute_local = {t_local*1e3:.3f} ms\n")
    hdr = (f"{'K':>3} {'U_unresc':>9} {'e_acc':>6} {'fire%':>6} {'resc_LOCAL':>10} "
           f"{'resc_starkmix':>13} {'onAR_brk%':>9} {'conf_miss':>9} {'idOK':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        star = " *" if (r["K"] in SWEEP_KS and r["K"] == kstar) else ""
        ship = " <ship" if r["K"] == SHIPPED_K else ""
        brk = f"{100*r['on_AR_head_break_rate']:.3f}" if r["on_AR_head_break_rate"] is not None else "  n/a"
        fm = "" if r["fire_rate_measured"] else "i"
        im = "" if r["identity_measured"] else "i"
        print(f"{r['K']:>3} {r['unrescued_wall_tps_local']:>9.2f} {r['e_accept_mean']:>6.3f} "
              f"{100*r['fire_rate_tau0p5']:>5.3f}{fm:<1} {r['rescued_local']:>10.2f} "
              f"{r['rescued_starkmix_official_fire']:>13.2f} {brk:>8}{im:<1} "
              f"{str(r['confident_off_AR_head_misses']):>9} {str(r['identity_holds']):>5}{star}{ship}")
    print("\n(i = interpolated for K not directly measured by #648/#651; K-independent)")
    print(f"\nVERDICT: {verdict}")
    print(f"K* (rescued-local argmax over {SWEEP_KS}) = {kstar}  -> {best_local:.2f} local tps")
    print(f"K* vs shipped K=6: {out['decision']['kstar_vs_k6_local_tps']:+.2f} tps "
          f"({out['decision']['kstar_vs_k6_local_pct']:+.2f}%)")
    print(f"all sweep Ks hold identity (0 confident misses): {all_identity_holds}")
    print("\nCROSS-VAL vs stark #642 (K=6):")
    print(f"  AR-ref local:    land {A_local:.2f}  vs stark {STARK_AR_REF}  "
          f"(gap {xval['ar_ref_local']['abs_pct_gap']}%)")
    print(f"  un-rescued K=6:  land {my_unrescued_k6:.2f}  vs stark {STARK_UNRESCUED_K6}  "
          f"(gap {xval['unrescued_k6_local']['abs_pct_gap']}%)")
    print(f"  rescued K=6 (starkmix): land {my_rescued_k6_starkmix:.2f}  "
          f"(stark headline PENDING; #636 K7 proj {STARK_636_RESCUED_K7})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
