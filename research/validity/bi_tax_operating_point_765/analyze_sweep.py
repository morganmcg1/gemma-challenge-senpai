#!/usr/bin/env python3
"""PR #765 analyzer: turn the BI=0 / BI=1 generation-length sweep + cold-prefill
traces into an operating-point robustness ledger for the 31.72% batch-invariance
tax.

Reuses #759's per-kernel-FAMILY classifier (parse_traces.arm_kernel_ms /
roll_families / matmul_subsplit) verbatim, so the family attribution is identical
to the merged COST card. New here:

  1. DECODE-VS-PREFILL split. Decode windows are prefix-cache-isolated pure decode
     -> total_device_ms / completion_tokens = decode_ms_per_tok(GEN). The prefill
     window is a cold prefill(P) + 1 decode step -> prefill_ms = window_total -
     one_decode_step (one_decode_step := decode_ms_per_tok at the smallest GEN,
     the seqlen closest to just-after-prefill). Reported per arm and as BI1-BI0.

  2. GEN-LENGTH curve. decode_ms_per_tok and the profiled device BI-tax % at each
     GEN in the sweep; flatness = spread of the per-GEN decode tax % vs the #759
     anchor (32.73% device ~ 31.72% official).

  3. PREDICTION BAND. Model per-request profiled total D_a(G) = prefill_ms_a +
     G*decode_ms_per_tok_a(G); TPS_prof_a(G)=G/D_a(G); anchor each arm to its #750
     official TPS at G=512 (BI1=156.949, BI0=229.847) via a single scale factor;
     report implied official TPS_a(G) and the operating-point tax(G)=1-TPS_BI1/TPS_BI0
     across plausible benchmark gen-lengths -> a band around ~157.

  4. VERDICT. bi_tax_operating_point_robust=1 iff the per-GEN decode tax % is
     gen-length-flat within tolerance (default: max deviation from the anchor
     <= TOL_REL*anchor, TOL_REL=0.02 i.e. "+-2% of the 31.72%").
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

# ---- import #759 parse_traces (reuse merged classifier; isolation-clean) ------
_P759 = (Path(__file__).resolve().parent.parent
         / "bi_tax_op_ledger_759" / "parse_traces.py")
_spec = importlib.util.spec_from_file_location("parse_traces_759", _P759)
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

# #750 official-anchored constants (PR #750 RESULTS.json), the numbers this card
# calibrates. Both measured at the #319 protocol: 512 new tokens, tau=0.
TPS_OFFICIAL = {0: 229.847, 1: 156.949}
ANCHOR_GEN = 512
# #759 merged device-level BI tax anchor (corroborates the 31.72% official tax).
BI_TAX_ANCHOR_PCT = 0.3273
TOL_REL = 0.02  # "+-2% of the 31.72%" -> relative tolerance on the per-GEN tax %


def window_total_ms(rec: dict) -> tuple[float, dict, dict]:
    """Total device ms over a window + per-family rollup + raw per-kernel ms."""
    pk, _ = pt.arm_kernel_ms(rec["trace_files"])
    fam = pt.roll_families(pk)
    return sum(fam.values()), fam, pk


def find_window(summary: dict, kind: str) -> dict | None:
    for w in summary["windows"]:
        if w["kind"] == kind and not w.get("discarded"):
            return w
    return None


def decode_profile(summary: dict, gens: list[int]) -> dict:
    """Per-GEN: total device ms, ms/tok, family ms/tok, matmul subsplit."""
    out = {}
    for g in gens:
        w = find_window(summary, f"decode_gen{g}")
        if w is None or not w["trace_files"]:
            continue
        tot, fam, pk = window_total_ms(w)
        ctok = w["completion_tokens"] or 1
        out[g] = {
            "completion_tokens": ctok,
            "total_device_ms": round(tot, 4),
            "decode_ms_per_tok": round(tot / ctok, 6),
            "family_ms_per_tok": {f: round(v / ctok, 6) for f, v in fam.items()},
            "matmul_subsplit": pt.matmul_subsplit(pk),
            "wall_s": w["wall_s"],
            "decode_tps_proxy": w["decode_tps_proxy"],
            "prompt_tokens": w["prompt_tokens"],
        }
    return out


def prefill_profile(summary: dict, one_decode_step_ms: float) -> dict:
    w = find_window(summary, "prefill")
    if w is None or not w["trace_files"]:
        return {}
    tot, fam, pk = window_total_ms(w)
    ptok = w["prompt_tokens"] or 1
    prefill_ms = tot - one_decode_step_ms
    return {
        "prompt_tokens": ptok,
        "window_total_device_ms": round(tot, 4),
        "one_decode_step_ms_subtracted": round(one_decode_step_ms, 4),
        "prefill_ms_total": round(prefill_ms, 4),
        "prefill_ms_per_prompt_tok": round(prefill_ms / ptok, 6),
        "family_ms": {f: round(v, 4) for f, v in fam.items()},
        "matmul_subsplit": pt.matmul_subsplit(pk),
        "wall_s": w["wall_s"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi0-summary", required=True)
    ap.add_argument("--bi1-summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s0 = json.loads(Path(args.bi0_summary).read_text())
    s1 = json.loads(Path(args.bi1_summary).read_text())
    gens = sorted(set(s0["gen_sweep"]) & set(s1["gen_sweep"]))

    dec0 = decode_profile(s0, gens)
    dec1 = decode_profile(s1, gens)
    common_g = sorted(set(dec0) & set(dec1))

    # one-decode-step estimate = decode_ms_per_tok at the smallest measured GEN
    g_small = common_g[0]
    pre0 = prefill_profile(s0, dec0[g_small]["decode_ms_per_tok"])
    pre1 = prefill_profile(s1, dec1[g_small]["decode_ms_per_tok"])

    # ---- per-GEN decode BI tax ------------------------------------------------
    gen_rows = []
    for g in common_g:
        b0 = dec0[g]["decode_ms_per_tok"]
        b1 = dec1[g]["decode_ms_per_tok"]
        added = b1 - b0
        pct = (added / b1) if b1 else None
        gen_rows.append({
            "gen": g,
            "bi0_decode_ms_per_tok": round(b0, 6),
            "bi1_decode_ms_per_tok": round(b1, 6),
            "added_decode_ms_per_tok": round(added, 6),
            "decode_bi_tax_pct": round(pct, 5) if pct is not None else None,
            "bi0_total_device_ms": dec0[g]["total_device_ms"],
            "bi1_total_device_ms": dec1[g]["total_device_ms"],
            "bi0_decode_tps_proxy": dec0[g]["decode_tps_proxy"],
            "bi1_decode_tps_proxy": dec1[g]["decode_tps_proxy"],
        })

    tax_pcts = [r["decode_bi_tax_pct"] for r in gen_rows
                if r["decode_bi_tax_pct"] is not None]
    tax_min, tax_max = min(tax_pcts), max(tax_pcts)
    tax_mean = sum(tax_pcts) / len(tax_pcts)
    tax_spread = tax_max - tax_min
    max_dev_from_anchor = max(abs(p - BI_TAX_ANCHOR_PCT) for p in tax_pcts)
    # verdict: per-GEN decode tax flat within +-TOL_REL of the anchor
    robust = int(max_dev_from_anchor <= TOL_REL * BI_TAX_ANCHOR_PCT)

    # ---- prefill BI tax -------------------------------------------------------
    prefill_added = prefill_pct = None
    if pre0 and pre1:
        prefill_added = pre1["prefill_ms_total"] - pre0["prefill_ms_total"]
        prefill_pct = (prefill_added / pre1["prefill_ms_total"]
                       if pre1["prefill_ms_total"] else None)

    # ---- prediction band: implied official TPS_a(G) ---------------------------
    # The cold-prefill window prompt was nonce-salted (longer than the decode
    # window prompt), so for a SELF-CONSISTENT single-request model we scale the
    # prefill term to the decode-window prompt length P_ref via the measured
    # per-prompt-token prefill cost (prefill ~ linear in P: GEMMs exactly linear,
    # sliding_window=512 caps attention on 35/42 layers). Band is reported for a
    # P_ref-token-prompt, G-output request; prefill_ms_per_prompt_tok is logged so
    # the band can be rescaled to any benchmark prompt length.
    P_ref = (dec1[ANCHOR_GEN].get("prompt_tokens")
             or dec0[ANCHOR_GEN].get("prompt_tokens")) if ANCHOR_GEN in dec1 else None

    def prefill_ref_ms(pre):
        return pre["prefill_ms_per_prompt_tok"] * P_ref if P_ref else pre["prefill_ms_total"]

    def Dms(dec, pre, g):  # per-request profiled total device ms at gen-length g
        return prefill_ref_ms(pre) + g * dec[g]["decode_ms_per_tok"]

    band = {}
    scale = {}
    if pre0 and pre1 and ANCHOR_GEN in common_g:
        for arm, dec, pre in ((0, dec0, pre0), (1, dec1, pre1)):
            tps_prof_anchor = ANCHOR_GEN / Dms(dec, pre, ANCHOR_GEN)
            scale[arm] = TPS_OFFICIAL[arm] / tps_prof_anchor
        for g in common_g:
            row = {}
            for arm, dec, pre in ((0, dec0, pre0), (1, dec1, pre1)):
                tps_prof = g / Dms(dec, pre, g)
                row[f"bi{arm}_implied_official_tps"] = round(tps_prof * scale[arm], 3)
            t1, t0 = row["bi1_implied_official_tps"], row["bi0_implied_official_tps"]
            row["op_point_tax_pct"] = round(1 - t1 / t0, 5) if t0 else None
            band[g] = row
        # decode-asymptote (G -> large): use the largest measured GEN rate
        g_big = common_g[-1]
        asym = {}
        for arm, dec in ((0, dec0), (1, dec1)):
            asym[f"bi{arm}_decode_asymptote_tps"] = round(
                scale[arm] / dec[g_big]["decode_ms_per_tok"], 3)
        band["asymptote_using_gen%d" % g_big] = asym

    bi1_tps_vals = [band[g]["bi1_implied_official_tps"]
                    for g in common_g if g in band]
    band_lo = min(bi1_tps_vals) if bi1_tps_vals else None
    band_hi = max(bi1_tps_vals) if bi1_tps_vals else None

    # ---- anchor (GEN=512) per-family ledger (matches #319 protocol length) ----
    anchor_family_ledger = []
    if ANCHOR_GEN in common_g:
        f0 = dec0[ANCHOR_GEN]["family_ms_per_tok"]
        f1 = dec1[ANCHOR_GEN]["family_ms_per_tok"]
        fams = sorted(set(f0) | set(f1))
        added = {f: f1.get(f, 0.0) - f0.get(f, 0.0) for f in fams}
        tot_added_pos = sum(v for v in added.values() if v > 0)
        for f in fams:
            anchor_family_ledger.append({
                "family": f,
                "bi0_ms_per_tok": round(f0.get(f, 0.0), 6),
                "bi1_ms_per_tok": round(f1.get(f, 0.0), 6),
                "added_ms_per_tok": round(added[f], 6),
                "share_of_total_added": round(added[f] / tot_added_pos, 4)
                if tot_added_pos > 0 else 0.0,
            })
        anchor_family_ledger.sort(key=lambda r: -r["added_ms_per_tok"])

    out = {
        "gens": common_g,
        "anchor_gen": ANCHOR_GEN,
        "bi_tax_anchor_pct_759": BI_TAX_ANCHOR_PCT,
        "tps_official_anchor": TPS_OFFICIAL,
        # primary deliverables
        "bi_tax_decode_ms_per_tok": (
            round(dec1[ANCHOR_GEN]["decode_ms_per_tok"], 6)
            if ANCHOR_GEN in dec1 else None),   # BI1 decode ms/tok at the anchor
        "decode_bi_tax_pct_at_anchor": next(
            (r["decode_bi_tax_pct"] for r in gen_rows if r["gen"] == ANCHOR_GEN),
            None),
        "gen_length_curve": gen_rows,
        "decode_tax_pct_min": round(tax_min, 5),
        "decode_tax_pct_max": round(tax_max, 5),
        "decode_tax_pct_mean": round(tax_mean, 5),
        "decode_tax_pct_spread": round(tax_spread, 5),
        "decode_tax_max_dev_from_anchor": round(max_dev_from_anchor, 5),
        "tolerance_rel": TOL_REL,
        "tolerance_abs_pp": round(TOL_REL * BI_TAX_ANCHOR_PCT, 5),
        "bi_tax_operating_point_robust": robust,
        # prefill split
        "prefill_bi0": pre0,
        "prefill_bi1": pre1,
        "prefill_bi_tax_ms_total": round(prefill_added, 4)
        if prefill_added is not None else None,
        "prefill_bi_tax_pct": round(prefill_pct, 5)
        if prefill_pct is not None else None,
        "band_P_ref_prompt_tokens": P_ref,
        "prefill_ref_ms_bi0": round(prefill_ref_ms(pre0), 4) if pre0 and P_ref else None,
        "prefill_ref_ms_bi1": round(prefill_ref_ms(pre1), 4) if pre1 and P_ref else None,
        # prediction band
        "scale_factor": {str(k): round(v, 5) for k, v in scale.items()},
        "prediction_band": band,
        "bi1_band_lo_tps": band_lo,
        "bi1_band_hi_tps": band_hi,
        # anchor-length family ledger
        "anchor_family_ledger": anchor_family_ledger,
        "decode_profile_bi0": dec0,
        "decode_profile_bi1": dec1,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    # pretty print
    print("=" * 86)
    print("PR #765  BI-TAX OPERATING-POINT ROBUSTNESS  (gen-length sweep + prefill split)")
    print("=" * 86)
    print(f"{'GEN':>6}{'BI0 ms/tok':>13}{'BI1 ms/tok':>13}{'added':>10}"
          f"{'decode tax%':>13}{'BI1 proxy tps':>15}")
    for r in gen_rows:
        print(f"{r['gen']:>6}{r['bi0_decode_ms_per_tok']:>13.5f}"
              f"{r['bi1_decode_ms_per_tok']:>13.5f}{r['added_decode_ms_per_tok']:>10.5f}"
              f"{(r['decode_bi_tax_pct'] or 0)*100:>12.2f}%"
              f"{(r['bi1_decode_tps_proxy'] or 0):>15.2f}")
    print("-" * 86)
    print(f"decode tax%% across GEN: min={tax_min*100:.2f} max={tax_max*100:.2f} "
          f"mean={tax_mean*100:.2f} spread={tax_spread*100:.2f}pp "
          f"maxdev_from_anchor={max_dev_from_anchor*100:.2f}pp "
          f"(tol={TOL_REL*BI_TAX_ANCHOR_PCT*100:.2f}pp)")
    print(f"VERDICT bi_tax_operating_point_robust = {robust}")
    print("-" * 86)
    if pre0 and pre1:
        print(f"PREFILL ms total: BI0={pre0['prefill_ms_total']:.3f} "
              f"BI1={pre1['prefill_ms_total']:.3f} "
              f"added={prefill_added:+.3f} pct={ (prefill_pct or 0)*100:.2f}%  "
              f"(prompt_tok BI0={pre0['prompt_tokens']} BI1={pre1['prompt_tokens']})")
    print("-" * 86)
    print("PREDICTION BAND (implied official TPS, anchored to #750 @ GEN=512):")
    for g in common_g:
        if g in band:
            b = band[g]
            print(f"  GEN={g:>5}: BI1={b['bi1_implied_official_tps']:>8.2f}  "
                  f"BI0={b['bi0_implied_official_tps']:>8.2f}  "
                  f"op_tax={ (b['op_point_tax_pct'] or 0)*100:>6.2f}%")
    if band_lo is not None:
        print(f"  BI1 band over swept GEN: [{band_lo:.2f}, {band_hi:.2f}] TPS  "
              f"(anchor 156.95 @ GEN=512)")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
