#!/usr/bin/env python3
"""PR #656 — pool AR vs SPEC GPQA-Diamond 10-seed sampled and contrast them.

Estimand (PR #656): the 10-seed pooled sampled accuracy for each arm
(int4_g128_lmhead body+head; AR = drafter OFF M=1, SPEC = MTP K=6 shipped
manifest), n = 198 x 10 = 1980 each. dev307+spec is non-deterministic, so the
multi-seed mean is the valid estimand; the per-seed spread shows the seed sigma.

Reports:
  - per-seed acc + pooled acc (Wilson 95% CI) for each arm
  - SPEC - AR delta, unpaired two-proportion z + p (the PR's literal ask)
  - McNemar paired test (AR & SPEC share prompts per (seed,id) -> more powerful)
  - per-seed scatter (min/max/sd); is fern #629's 0.4652 inside the SPEC range?
  - %-of-base (/0.5404) and pass/fail vs the 0.4864 (90%) bar, both arms
  - flip diagnostic: paired AR-vs-SPEC answer flips, concentrated vs spread
"""
from __future__ import annotations

import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path

DIR = Path("research/validity/specdec_gpqa_10seed/results")
BASE = 0.5404          # GPQA-D sampled vanilla bf16 base
BAR = 0.4864           # #515 90%-of-base bar
FERN629 = 0.4652       # thin-seed WITH-SPEC point estimate to confirm/refute
UBEL638 = 0.4990       # AR int4 corner reference


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, c - h, c + h)


def load_arm(arm: str):
    """Return {seed: {id: correct_bool}} and per-seed (acc,c,n) rows."""
    rows = []
    by_seed = {}
    for f in sorted(glob.glob(str(DIR / f"gpqa_{arm}_s*.json"))):
        d = json.load(open(f))
        seed = d["seed"]
        rows.append((seed, d["accuracy"], d["n_correct"], d["n_scored"],
                     d.get("n_error", 0), d.get("n_length", 0)))
        by_seed[seed] = {s["id"]: bool(s["correct"]) for s in d["per_sample"]}
    rows.sort()
    return by_seed, rows


def pooled(rows):
    c = sum(r[2] for r in rows)
    n = sum(r[3] for r in rows)
    return c, n


def fmt_arm(name, rows):
    if not rows:
        print(f"\n{name}: NO DATA")
        return None
    accs = [r[1] for r in rows]
    c, n = pooled(rows)
    p, lo, hi = wilson(c, n)
    sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    print(f"\n=== {name} ===")
    for seed, a, cc, nn, ne, nl in rows:
        print(f"  s{seed}: acc={a:.4f} ({cc}/{nn}) err={ne} trunc={nl}")
    print(f"  n_seeds={len(rows)}  per-seed mean={statistics.mean(accs):.4f} "
          f"sd={sd:.4f} min={min(accs):.4f} max={max(accs):.4f}")
    print(f"  POOLED acc={p:.4f} ({c}/{n})  Wilson95=[{lo:.4f},{hi:.4f}]")
    print(f"  %-of-base (/{BASE})={p/BASE*100:.1f}%   vs bar {BAR}: "
          f"{'PASS' if p >= BAR else 'FAIL'} ({p-BAR:+.4f})")
    return {"rows": rows, "accs": accs, "c": c, "n": n, "p": p,
            "lo": lo, "hi": hi, "sd": sd, "min": min(accs), "max": max(accs)}


def two_prop_z(c1, n1, c2, n2):
    p1, p2 = c1 / n1, c2 / n2
    pbar = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p, se


def mcnemar(ar_by_seed, spec_by_seed):
    """Paired over (seed,id): b = AR-correct & SPEC-wrong (SPEC loss);
    c = AR-wrong & SPEC-correct (SPEC gain). Per-question net for concentration."""
    b = c = both = neither = npair = 0
    per_q_loss = Counter()   # id -> (#AR-correct&SPEC-wrong) - (#AR-wrong&SPEC-correct)
    per_q_disc = Counter()
    for seed, ar in ar_by_seed.items():
        sp = spec_by_seed.get(seed)
        if not sp:
            continue
        for qid, arc in ar.items():
            if qid not in sp:
                continue
            spc = sp[qid]
            npair += 1
            if arc and not spc:
                b += 1; per_q_loss[qid] += 1; per_q_disc[qid] += 1
            elif (not arc) and spc:
                c += 1; per_q_loss[qid] -= 1; per_q_disc[qid] += 1
            elif arc and spc:
                both += 1
            else:
                neither += 1
    nd = b + c
    # exact-ish McNemar with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / nd if nd > 0 else 0.0
    pmc = 1 - 0.5 * (1 + math.erf(math.sqrt(chi2) / math.sqrt(2))) if nd > 0 else 1.0
    pmc *= 2 if False else 1  # chi2 1df already two-sided
    return {"npair": npair, "both": both, "neither": neither,
            "b_AR_correct_SPEC_wrong": b, "c_AR_wrong_SPEC_correct": c,
            "n_discordant": nd, "mcnemar_chi2_cc": chi2, "mcnemar_p": pmc,
            "per_q_loss": per_q_loss, "per_q_disc": per_q_disc}


def main():
    ar_by_seed, ar_rows = load_arm("ar_m1")
    spec_by_seed, spec_rows = load_arm("spec_k6")
    AR = fmt_arm("AR (M=1, drafter OFF)", ar_rows)
    SP = fmt_arm("SPEC (MTP K=6, shipped manifest)", spec_rows)
    out = {"base": BASE, "bar": BAR, "fern629": FERN629, "ubel638": UBEL638}
    def per_seed(rows):
        return [{"seed": r[0], "accuracy": r[1], "n_correct": r[2], "n_scored": r[3],
                 "n_error": r[4], "n_length": r[5]} for r in rows]
    if AR:
        out["ar"] = {k: AR[k] for k in ("c", "n", "p", "lo", "hi", "sd", "min", "max")}
        out["ar"]["per_seed"] = per_seed(ar_rows)
        out["ar"]["mean"] = statistics.mean(AR["accs"])
    if SP:
        out["spec"] = {k: SP[k] for k in ("c", "n", "p", "lo", "hi", "sd", "min", "max")}
        out["spec"]["per_seed"] = per_seed(spec_rows)
        out["spec"]["mean"] = statistics.mean(SP["accs"])

    if AR and SP:
        delta = SP["p"] - AR["p"]
        z, pval, se = two_prop_z(SP["c"], SP["n"], AR["c"], AR["n"])
        print("\n=== CONTRAST  SPEC - AR (pooled) ===")
        print(f"  delta = {delta:+.4f}   unpaired z={z:+.3f}  p={pval:.4g}  (se={se:.4f})")
        mc = mcnemar(ar_by_seed, spec_by_seed)
        print(f"  paired (McNemar over {mc['npair']} matched (seed,id)):")
        print(f"    AR-correct&SPEC-wrong b={mc['b_AR_correct_SPEC_wrong']}  "
              f"AR-wrong&SPEC-correct c={mc['c_AR_wrong_SPEC_correct']}  "
              f"discordant={mc['n_discordant']}")
        print(f"    McNemar chi2(cc)={mc['mcnemar_chi2_cc']:.3f}  p={mc['mcnemar_p']:.4g}")
        # flip concentration: net loss per question (summed over seeds)
        net = mc["per_q_loss"]
        nz = {q: v for q, v in net.items() if v != 0}
        loss_q = sorted(nz.items(), key=lambda kv: -kv[1])
        net_total = sum(net.values())
        print(f"\n  flip concentration (net SPEC-loss per question, summed over seeds):")
        print(f"    net total (b-c) = {net_total}  over {len(nz)} questions with nonzero net")
        print(f"    top SPEC-loss questions: {loss_q[:8]}")
        print(f"    top SPEC-gain questions: {loss_q[-8:]}")
        # Gini-ish: share of total |net| from top-5 questions
        absnet = sorted((abs(v) for v in net.values()), reverse=True)
        tot_abs = sum(absnet)
        top5 = sum(absnet[:5]) / tot_abs if tot_abs else 0.0
        print(f"    top-5 questions carry {top5*100:.1f}% of total |net flip| "
              f"({'CONCENTRATED' if top5 > 0.5 else 'SPREAD'})")
        out["contrast"] = {"delta_spec_minus_ar": delta, "z": z, "p": pval, "se": se,
                           "mcnemar": {k: mc[k] for k in
                                       ("npair", "both", "neither",
                                        "b_AR_correct_SPEC_wrong",
                                        "c_AR_wrong_SPEC_correct", "n_discordant",
                                        "mcnemar_chi2_cc", "mcnemar_p")},
                           "net_total_b_minus_c": net_total,
                           "n_questions_nonzero_net": len(nz),
                           "top5_share_abs_flip": top5}
        # fern #629 / ubel #638 placement
        print(f"\n  fern #629 0.4652 inside SPEC per-seed range "
              f"[{SP['min']:.4f},{SP['max']:.4f}]? "
              f"{SP['min'] <= FERN629 <= SP['max']}")
        print(f"  ubel #638 0.4990 inside AR per-seed range "
              f"[{AR['min']:.4f},{AR['max']:.4f}]? "
              f"{AR['min'] <= UBEL638 <= AR['max']}")
        out["fern629_in_spec_range"] = bool(SP["min"] <= FERN629 <= SP["max"])
        out["ubel638_in_ar_range"] = bool(AR["min"] <= UBEL638 <= AR["max"])
        verdict = "SPEC_GPQA_COST_REAL" if (delta < 0 and pval < 0.05) else "SPEC_GPQA_COST_IS_NOISE"
        out["verdict"] = verdict
        print(f"\n  VERDICT: {verdict}")

    Path("research/validity/specdec_gpqa_10seed/contrast.json").write_text(
        json.dumps(out, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print("\nwrote contrast.json")


if __name__ == "__main__":
    main()
