#!/usr/bin/env python
"""Roll up the spec-break quality-materiality census (PR #682, wirbel).

Body-matched AR-vs-SPEC. BOTH arms serve the SAME int4_g128_lmhead body on the
SAME engine (dev307, BI=1); the ONLY removed variable is speculation (AR k=0 vs
SPEC k=6, the wirbel #671 ~170-band drafter). So the per-benchmark retention
  specbreak_retention_<leg> = spec_score / ar_score
isolates the int4-Marlin verify-width break (kanna #673) from the int4 body's own
pre-existing quality gap. The PRIMARY metric is
  specbreak_worst_retention = min over {gsm8k, mmlu_pro, gpqa_diamond, aime}.

Decision rule (the card's verdict, EVIDENCE for the human's strict-#319 gate
decision -- NOT a recommendation to change the gate):
  SPEC_BREAK_QUALITY_NEUTRAL    worst_retention >= NEUTRAL_BAR (~0.97) AND break present
  SPEC_BREAK_QUALITY_BORDERLINE MATERIAL_BAR <= worst_retention < NEUTRAL_BAR
  SPEC_BREAK_QUALITY_MATERIAL   worst_retention < MATERIAL_BAR (~0.90)
  SPEC_BREAK_INCONCLUSIVE_NO_BREAK   token break-rate is zero -> nothing to cost.

The token break-rate is the gate that the break is ACTIVE on this config; it is
measured the canonical strict-#319 way (token_break_probe.py: 128 sharegpt x512
greedy, decode_outputs.py) and is directly comparable to census #607 / kanna #673.
The quality panel is run at the realistic batched serving config (conc 16); the
two answer different sub-questions (is there a break? does it cost quality?).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

DIR = Path("research/validity/specbreak_quality_materiality")
LEGS = ["gsm8k", "mmlu_pro", "gpqa_diamond", "aime"]
NEUTRAL_BAR = 0.97
MATERIAL_BAR = 0.90

# Published DEV307 SPEC-arm panel baseline (the int4_g128_lmhead+MTP-K7 greedy
# gb6144 numbers this card's SPEC arm should reproduce). Cross-check only, NOT the
# isolated finding -- the isolated finding is spec/ar measured fresh in THIS card.
DEV307_SPEC_REF = {
    "mmlu_pro": 0.664, "gsm8k": 0.928, "aime": 0.400, "gpqa_diamond": 0.4764,
}

# Vanilla (unquantized) base #515 denominators, authoritative BASELINE.md grounding
# (PR #580/#581). The #515 gate is >=90% of these. Used ONLY for the clearly-labeled
# COMBINED body+break ratio (spec/base) -- NOT this card's isolated finding.
# PROTOCOL CAVEAT: these base numbers come from the #580/#581 measurement protocol,
# which is NOT identical to this card's gb6144 panel (notably AIME base=0.100 used a
# 3072-token cap with ~72% truncation vs our 6144 cap). So the combined ratio is
# indicative for the human's downstream read, not a protocol-matched gate verdict.
BASE515_DENOM = {
    "mmlu_pro": 0.6727, "gpqa_diamond": 0.5236, "gsm8k": 0.8967, "aime": 0.100,
}
BASE515_PROTOCOL_MATCHED = 0  # base denoms are a different protocol (see caveat above)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def katz_ratio_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96
                  ) -> tuple[float, float, float]:
    """Ratio R = p1/p2 of two independent binomials, Katz-log CI.

    p1 = spec proportion, p2 = ar proportion. Returns (R, lo, hi). Guards the
    p=0 endpoints (CI becomes one-sided/degenerate, reported as nan there).
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    if p2 == 0:
        return (float("inf") if p1 > 0 else float("nan"), float("nan"), float("nan"))
    R = p1 / p2
    if p1 == 0:
        return (0.0, 0.0, float("nan"))
    se = math.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
    return (R, R * math.exp(-z * se), R * math.exp(z * se))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (sign test on the b+c discordant pairs).

    b = AR-correct/SPEC-wrong, c = AR-wrong/SPEC-correct. Under H0 (the break
    reshuffles correctness symmetrically -> zero net quality effect) each
    discordant pair is a fair coin. p<0.05 => the break degrades (or improves)
    quality beyond noise on that leg; p>=0.05 => the leg is noise-consistent
    (the operative "within benchmark noise" clause of the NEUTRAL definition).
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def paired_diff_ci(b: int, c: int, n_pairs: int, z: float = 1.96
                   ) -> tuple[float, float, float]:
    """Wald CI for the PAIRED accuracy diff (spec_acc - ar_acc) = (c-b)/N.

    Negative => spec (the broken arm) scores below AR. The paired SE uses only
    the discordant counts, so it is far tighter (and correct) for same-question
    same-body arms than the independent-binomial Katz ratio CI.
    """
    if n_pairs == 0:
        return (float("nan"), float("nan"), float("nan"))
    diff = (c - b) / n_pairs
    var = (b + c - (b - c) ** 2 / n_pairs) / (n_pairs * n_pairs)
    se = math.sqrt(var) if var > 0 else 0.0
    return (diff, diff - z * se, diff + z * se)


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def load_leg_items(arm_dir: Path, leg: str) -> dict[str, bool]:
    """Per-item correctness keyed by stable item id for paired AR-vs-SPEC stats.

    Both arms score the SAME items (same seed, same body), so the keys align and
    the paired confusion (n11/n10/n01/n00) isolates the break's per-question
    churn from its net directional effect.
    """
    items: dict[str, bool] = {}
    if leg == "gsm8k":
        hits = sorted(glob.glob(str(arm_dir / "*_greedy_greedy*.json")))
        if not hits:
            return items
        for r in _load_json(Path(hits[0])).get("per_problem", []):
            items[str(r["id"])] = bool(r["correct"])
    elif leg == "mmlu_pro":
        p = arm_dir / "mmlu_pro.json"
        if p.exists():
            for r in _load_json(p).get("per_sample", []):
                items[str(r["id"])] = bool(r["correct"])
    elif leg == "gpqa_diamond":
        for h in sorted(glob.glob(str(arm_dir / "gpqa_diamond_s*.json"))):
            seed = Path(h).stem.split("_s")[-1]
            for r in _load_json(Path(h)).get("per_sample", []):
                items[f"{seed}:{r['id']}"] = bool(r["correct"])
    elif leg == "aime":
        p = arm_dir / "aime.json"
        if p.exists():
            for r in _load_json(p).get("per_problem", []):
                items[str(r["id"])] = bool(r["maj_correct"])
    return items


def paired_stats(spec_dir: Path, ar_dir: Path, leg: str) -> dict[str, Any]:
    sp, ar = load_leg_items(spec_dir, leg), load_leg_items(ar_dir, leg)
    common = sorted(set(sp) & set(ar))
    n11 = n10 = n01 = n00 = 0
    for i in common:
        a, s = ar[i], sp[i]
        if a and s:
            n11 += 1
        elif a and not s:
            n10 += 1          # b: AR right, SPEC wrong (break hurt)
        elif (not a) and s:
            n01 += 1          # c: AR wrong, SPEC right (break helped)
        else:
            n00 += 1
    b, c, N = n10, n01, len(common)
    mp = mcnemar_exact_p(b, c)
    diff, dlo, dhi = paired_diff_ci(b, c, N)
    return {
        "n_pairs": N, "both_correct": n11, "ar_only_correct": b,
        "spec_only_correct": c, "both_wrong": n00, "discordant": b + c,
        "net_spec_minus_ar": c - b, "mcnemar_exact_p": mp,
        "paired_acc_diff_spec_minus_ar": diff, "paired_acc_diff_ci95": [dlo, dhi],
        "noise_consistent": bool(mp >= 0.05),
    }


def load_leg(arm_dir: Path, leg: str) -> dict[str, Any] | None:
    """Return {score, k, n, raw_path} for one arm/leg, or None if missing."""
    if leg == "gsm8k":
        hits = sorted(glob.glob(str(arm_dir / "*_greedy_greedy*.json")))
        if not hits:
            return None
        o = _load_json(Path(hits[0]))
        return {"score": o["accuracy"], "k": o["n_correct"], "n": o["n_problems"],
                "raw_path": hits[0], "extra": {"strict_rate": o.get("strict_rate")}}
    if leg == "mmlu_pro":
        p = arm_dir / "mmlu_pro.json"
        if not p.exists():
            return None
        o = _load_json(p)
        return {"score": o["accuracy"], "k": o["n_correct"], "n": o["n_scored"],
                "raw_path": str(p),
                "extra": {"len_stop_rate": o.get("len_stop_rate"),
                          "n_error": o.get("n_error")}}
    if leg == "gpqa_diamond":
        hits = sorted(glob.glob(str(arm_dir / "gpqa_diamond_s*.json")))
        if not hits:
            return None
        # pool across seeds (each file is one greedy seed)
        kk = nn = 0
        for h in hits:
            o = _load_json(Path(h))
            kk += int(o["n_correct"])
            nn += int(o["n_scored"])
        return {"score": kk / nn if nn else float("nan"), "k": kk, "n": nn,
                "raw_path": ",".join(hits), "extra": {"n_seeds": len(hits)}}
    if leg == "aime":
        p = arm_dir / "aime.json"
        if not p.exists():
            return None
        o = _load_json(p)
        return {"score": o["maj_k_accuracy"], "k": o["n_correct_maj"],
                "n": o["n_problems"], "raw_path": str(p),
                "extra": {"extract_fail_rate": o.get("extract_fail_rate")}}
    raise ValueError(leg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-dir", default=str(DIR / "results" / "spec"))
    ap.add_argument("--ar-dir", default=str(DIR / "results" / "ar"))
    ap.add_argument("--break-json", default=str(DIR / "results" / "token_break.json"))
    ap.add_argument("--out", default=str(DIR / "results" / "rollup.json"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="specbreak-quality-materiality-wirbel")
    args = ap.parse_args(argv)

    spec_dir, ar_dir = Path(args.spec_dir), Path(args.ar_dir)
    rows: list[dict[str, Any]] = []
    for leg in LEGS:
        s = load_leg(spec_dir, leg)
        a = load_leg(ar_dir, leg)
        if s is None or a is None:
            print(f"[rollup] SKIP {leg}: spec={'ok' if s else 'MISSING'} "
                  f"ar={'ok' if a else 'MISSING'}")
            continue
        R, rlo, rhi = katz_ratio_ci(s["k"], s["n"], a["k"], a["n"])
        s_lo, s_hi = wilson(s["k"], s["n"])
        a_lo, a_hi = wilson(a["k"], a["n"])
        rows.append({
            "leg": leg,
            "spec_score": s["score"], "spec_k": s["k"], "spec_n": s["n"],
            "spec_ci95": [s_lo, s_hi], "spec_extra": s["extra"],
            "ar_score": a["score"], "ar_k": a["k"], "ar_n": a["n"],
            "ar_ci95": [a_lo, a_hi], "ar_extra": a["extra"],
            "specbreak_retention": R, "retention_ci95": [rlo, rhi],
            "abs_delta": s["score"] - a["score"],
            "dev307_spec_ref": DEV307_SPEC_REF.get(leg),
            "paired": paired_stats(spec_dir, ar_dir, leg),
            "base515_denom": BASE515_DENOM.get(leg),
            "combined_bodybreak_ratio": (s["score"] / BASE515_DENOM[leg]
                                         if BASE515_DENOM.get(leg) else None),
            "spec_clears_515gate": int(s["score"] >= 0.9 * BASE515_DENOM[leg])
                                   if BASE515_DENOM.get(leg) else None,
            "ar_clears_515gate": int(a["score"] >= 0.9 * BASE515_DENOM[leg])
                                 if BASE515_DENOM.get(leg) else None,
        })

    # token-break gate
    brk: dict[str, Any] = {}
    bp = Path(args.break_json)
    if bp.exists():
        brk = _load_json(bp)
    break_present = bool(brk.get("break_present", False))
    tok_rate = brk.get("specbreak_token_break_rate")
    seq_div = brk.get("specbreak_seq_divergence_rate")

    # PRIMARY metric + verdict
    if rows:
        worst = min(rows, key=lambda r: r["specbreak_retention"])
        worst_ret = worst["specbreak_retention"]
        worst_ret_lo = worst["retention_ci95"][0]
        worst_leg = worst["leg"]
    else:
        worst_ret = worst_ret_lo = float("nan")
        worst_leg = None

    n_present = len(rows)
    full_panel = n_present == len(LEGS)
    if not full_panel:
        verdict = "SPEC_BREAK_PANEL_INCOMPLETE"
    elif not break_present:
        verdict = "SPEC_BREAK_INCONCLUSIVE_NO_BREAK"
    elif worst_ret < MATERIAL_BAR:
        verdict = "SPEC_BREAK_QUALITY_MATERIAL"
    elif worst_ret >= NEUTRAL_BAR:
        verdict = "SPEC_BREAK_QUALITY_NEUTRAL"
    else:
        verdict = "SPEC_BREAK_QUALITY_BORDERLINE"

    # Statistical adjudication of the card's "within benchmark noise" clause.
    # The mechanical `verdict` keys off the worst POINT estimate vs the 0.97 bar;
    # this keys off whether ANY leg shows a statistically significant directional
    # degradation (paired McNemar p<0.05 AND spec net-worse). For a BORDERLINE
    # point estimate that is fully noise-consistent, these disagree -- and the
    # human needs both to price the gate.
    legs_sig_degraded = [
        r["leg"] for r in rows
        if r["paired"]["mcnemar_exact_p"] < 0.05
        and r["paired"]["net_spec_minus_ar"] < 0
    ]
    max_mcnemar = max((r["paired"]["mcnemar_exact_p"] for r in rows), default=float("nan"))
    min_mcnemar = min((r["paired"]["mcnemar_exact_p"] for r in rows), default=float("nan"))
    statistical_verdict = (
        "NO_SIGNIFICANT_DEGRADATION" if (full_panel and not legs_sig_degraded)
        else ("SIGNIFICANT_DEGRADATION" if legs_sig_degraded else "PANEL_INCOMPLETE")
    )

    # Combined body+break gate-separation: does the break cause any NEW #515 gate
    # failure (spec fails where AR passes)? If the only sub-gate leg is sub-gate in
    # BOTH arms, the gate risk is the int4 BODY's gap, not the break.
    break_new_gate_fail = [
        r["leg"] for r in rows
        if r.get("spec_clears_515gate") == 0 and r.get("ar_clears_515gate") == 1
    ]
    sub_gate_both = [
        r["leg"] for r in rows
        if r.get("spec_clears_515gate") == 0 and r.get("ar_clears_515gate") == 0
    ]

    out = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "card": "specbreak_quality_materiality (PR #682, wirbel)",
        "isolation": "body-matched int4_g128_lmhead, dev307, BI=1; only var = spec (ar k=0 vs spec k=6)",
        "neutral_bar": NEUTRAL_BAR, "material_bar": MATERIAL_BAR,
        "specbreak_worst_retention": worst_ret,
        "specbreak_worst_retention_ci95_lo": worst_ret_lo,
        "specbreak_worst_leg": worst_leg,
        "specbreak_token_break_rate": tok_rate,
        "specbreak_token_break_rate_ci95": brk.get("specbreak_token_break_rate_ci95"),
        "specbreak_seq_divergence_rate": seq_div,
        "median_first_break_position": brk.get("median_first_break_position"),
        "break_present": break_present,
        "verdict": verdict,
        "statistical_verdict": statistical_verdict,
        "legs_significantly_degraded": legs_sig_degraded,
        "min_mcnemar_exact_p": min_mcnemar,
        "max_mcnemar_exact_p": max_mcnemar,
        "break_caused_new_515_gate_failures": break_new_gate_fail,
        "sub_515gate_in_both_arms": sub_gate_both,
        "combined_ratio_base_denom_protocol_matched": BASE515_PROTOCOL_MATCHED,
        "combined_ratio_base_denom_source": "BASELINE.md #580/#581",
        "n_legs_present": n_present, "full_panel": full_panel,
        "per_leg": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # report
    print("=" * 84)
    print("SPEC-BREAK QUALITY-MATERIALITY CENSUS  (PR #682, wirbel)  body-matched AR vs SPEC")
    print("=" * 84)
    print(f"{'leg':14s} {'spec':>8s} {'ar':>8s} {'retention':>10s} "
          f"{'ret_CI95':>17s} {'absΔ':>8s}  dev307_ref")
    for r in rows:
        rlo, rhi = r["retention_ci95"]
        print(f"{r['leg']:14s} {r['spec_score']:8.4f} {r['ar_score']:8.4f} "
              f"{r['specbreak_retention']:10.4f} "
              f"[{rlo:6.4f},{rhi:6.4f}] {r['abs_delta']:+8.4f}  "
              f"{r['dev307_spec_ref']}")
    print("-" * 84)
    print("PAIRED (same-question, same-body) McNemar adjudication of 'within noise':")
    print(f"{'leg':14s} {'n_pair':>6s} {'b(AR>SP)':>9s} {'c(SP>AR)':>9s} "
          f"{'net':>5s} {'pairedΔ_CI95':>20s} {'mcnemar_p':>10s} noise_ok")
    for r in rows:
        p = r["paired"]
        dlo, dhi = p["paired_acc_diff_ci95"]
        print(f"{r['leg']:14s} {p['n_pairs']:6d} {p['ar_only_correct']:9d} "
              f"{p['spec_only_correct']:9d} {p['net_spec_minus_ar']:+5d} "
              f"[{dlo:+7.4f},{dhi:+7.4f}] {p['mcnemar_exact_p']:10.4f} "
              f"{p['noise_consistent']}")
    print(f"  statistical_verdict = {statistical_verdict}  "
          f"(legs_sig_degraded={legs_sig_degraded or 'none'})")
    print("-" * 84)
    print("COMBINED body+break vs vanilla-base #515 (base denoms BASELINE.md #580/#581, "
          f"protocol_matched={BASE515_PROTOCOL_MATCHED}):")
    print(f"{'leg':14s} {'spec':>7s} {'base':>7s} {'spec/base':>9s} "
          f"{'gate.9b':>7s} {'AR>=g':>6s} {'SPEC>=g':>7s}")
    for r in rows:
        b = r["base515_denom"]
        print(f"{r['leg']:14s} {r['spec_score']:7.4f} {b:7.4f} "
              f"{r['combined_bodybreak_ratio']:9.4f} {0.9*b:7.4f} "
              f"{str(bool(r['ar_clears_515gate'])):>6s} {str(bool(r['spec_clears_515gate'])):>7s}")
    print(f"  break_caused_new_515_gate_failures = {break_new_gate_fail or 'NONE'}")
    print(f"  sub_515gate_in_both_arms (BODY gap, not break) = {sub_gate_both or 'none'}")
    print("-" * 84)
    print(f"  token break-rate = "
          f"{tok_rate*100:.4f}%" if isinstance(tok_rate, (int, float)) else
          f"  token break-rate = {tok_rate}")
    print(f"  break_present    = {break_present}")
    print(f"  seq divergence   = "
          f"{seq_div*100:.2f}%" if isinstance(seq_div, (int, float)) else
          f"  seq divergence   = {seq_div}")
    print(f"  PRIMARY specbreak_worst_retention = {worst_ret:.4f}  "
          f"(leg={worst_leg}, CI95-lo={worst_ret_lo:.4f})")
    print(f"  VERDICT = {verdict}")
    print(f"  wrote {args.out}")

    if args.wandb:
        _log_wandb(out, rows, args.wandb_group)
    return 0


def _log_wandb(out: dict, rows: list[dict], group: str) -> None:
    import wandb
    ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    COMMON = {
        "card": "specbreak_quality_materiality",
        "config_name": "int4_g128_lmhead + MTP-K6 spec vs AR (wirbel #682)",
        "vllm_version": "0.22.1rc1.dev307+g3e8afdf78",
        "body": "/workspace/gemma_build/int4_g128_lmhead",
        "drafter": "/tmp/qat-assistant", "num_speculative_tokens_spec": 6,
        "batch_invariant": 1, "max_model_len": 8192, "max_num_seqs": 16,
        "min_tokens": 8, "max_tokens": 6144, "budget": "gb6144",
        "is_319_identical": False, "analysis_only": True, "official_tps": 0,
        "fires": 0, "neutral_bar": NEUTRAL_BAR, "material_bar": MATERIAL_BAR,
    }
    run_ids = []
    for r in rows:
        run = wandb.init(project=PROJECT, entity=ENTITY, group=group,
                         name=f"wirbel/specbreak-{r['leg']}", job_type="quality-eval",
                         reinit=True, config={**COMMON, "leg": r["leg"]})
        pd = r["paired"]
        wandb.log({
            f"spec_score_{r['leg']}": r["spec_score"],
            f"ar_score_{r['leg']}": r["ar_score"],
            f"specbreak_retention_{r['leg']}": r["specbreak_retention"],
            "retention_ci95_lo": r["retention_ci95"][0],
            "retention_ci95_hi": r["retention_ci95"][1],
            "abs_delta": r["abs_delta"],
            "spec_n": r["spec_n"], "spec_k": r["spec_k"],
            "ar_n": r["ar_n"], "ar_k": r["ar_k"],
            "spec_ci95_lo": r["spec_ci95"][0], "spec_ci95_hi": r["spec_ci95"][1],
            "ar_ci95_lo": r["ar_ci95"][0], "ar_ci95_hi": r["ar_ci95"][1],
            "dev307_spec_ref": r["dev307_spec_ref"],
            f"paired_discordant_{r['leg']}": pd["discordant"],
            f"paired_net_spec_minus_ar_{r['leg']}": pd["net_spec_minus_ar"],
            f"paired_mcnemar_p_{r['leg']}": pd["mcnemar_exact_p"],
            f"paired_acc_diff_{r['leg']}": pd["paired_acc_diff_spec_minus_ar"],
            "paired_acc_diff_ci95_lo": pd["paired_acc_diff_ci95"][0],
            "paired_acc_diff_ci95_hi": pd["paired_acc_diff_ci95"][1],
            f"paired_noise_consistent_{r['leg']}": int(pd["noise_consistent"]),
            f"base515_denominator_{r['leg']}": r["base515_denom"],
            f"combined_bodybreak_ratio_{r['leg']}": r["combined_bodybreak_ratio"],
            f"spec_clears_515gate_{r['leg']}": r["spec_clears_515gate"],
            f"ar_clears_515gate_{r['leg']}": r["ar_clears_515gate"],
        })
        run.summary[f"specbreak_retention_{r['leg']}"] = r["specbreak_retention"]
        run.summary[f"spec_score_{r['leg']}"] = r["spec_score"]
        run.summary[f"ar_score_{r['leg']}"] = r["ar_score"]
        run.summary["analysis_only"] = 1
        run.summary["official_tps"] = 0
        run.summary["fires"] = 0
        run_ids.append(run.id)
        run.finish()

    run = wandb.init(project=PROJECT, entity=ENTITY, group=group,
                     name="wirbel/specbreak-VERDICT", job_type="verdict",
                     reinit=True, config={**COMMON, "verdict": out["verdict"]})
    vlog = {
        "specbreak_worst_retention": out["specbreak_worst_retention"],
        "specbreak_worst_retention_ci95_lo": out["specbreak_worst_retention_ci95_lo"],
        "specbreak_token_break_rate": out["specbreak_token_break_rate"],
        "specbreak_seq_divergence_rate": out["specbreak_seq_divergence_rate"],
        "median_first_break_position": out["median_first_break_position"],
        "break_present": int(out["break_present"]),
        "full_panel": int(out["full_panel"]),
    }
    for r in rows:
        vlog[f"specbreak_retention_{r['leg']}"] = r["specbreak_retention"]
        vlog[f"spec_score_{r['leg']}"] = r["spec_score"]
        vlog[f"ar_score_{r['leg']}"] = r["ar_score"]
        vlog[f"paired_mcnemar_p_{r['leg']}"] = r["paired"]["mcnemar_exact_p"]
        vlog[f"paired_net_spec_minus_ar_{r['leg']}"] = r["paired"]["net_spec_minus_ar"]
        vlog[f"paired_discordant_{r['leg']}"] = r["paired"]["discordant"]
        vlog[f"combined_bodybreak_ratio_{r['leg']}"] = r["combined_bodybreak_ratio"]
        vlog[f"base515_denominator_{r['leg']}"] = r["base515_denom"]
        vlog[f"spec_clears_515gate_{r['leg']}"] = r["spec_clears_515gate"]
        vlog[f"ar_clears_515gate_{r['leg']}"] = r["ar_clears_515gate"]
    vlog["statistical_verdict_no_sig_degradation"] = int(
        out["statistical_verdict"] == "NO_SIGNIFICANT_DEGRADATION")
    vlog["min_mcnemar_exact_p"] = out["min_mcnemar_exact_p"]
    vlog["break_caused_new_515_gate_failures_n"] = len(out["break_caused_new_515_gate_failures"])
    wandb.log(vlog)
    run.summary["specbreak_worst_retention"] = out["specbreak_worst_retention"]
    run.summary["specbreak_worst_leg"] = out["specbreak_worst_leg"]
    run.summary["verdict"] = out["verdict"]
    run.summary["statistical_verdict"] = out["statistical_verdict"]
    run.summary["legs_significantly_degraded"] = ",".join(out["legs_significantly_degraded"]) or "none"
    run.summary["min_mcnemar_exact_p"] = out["min_mcnemar_exact_p"]
    run.summary["break_present"] = bool(out["break_present"])
    run.summary["surface_to_human"] = True
    run.summary["analysis_only"] = 1
    run.summary["official_tps"] = 0
    run.summary["fires"] = 0
    run.summary["break_caused_new_515_gate_failures"] = len(out["break_caused_new_515_gate_failures"])
    run.summary["sub_515gate_in_both_arms"] = ",".join(out["sub_515gate_in_both_arms"]) or "none"
    run.summary["combined_ratio_base_denom_protocol_matched"] = BASE515_PROTOCOL_MATCHED
    run.summary["combined_ratio_base_denom_source"] = "BASELINE.md #580/#581"
    run_ids.append(run.id)
    run.finish()
    print(f"[wandb] logged {len(rows)} legs + VERDICT to {ENTITY}/{PROJECT} "
          f"group={group} run_ids={run_ids}")


if __name__ == "__main__":
    raise SystemExit(main())
