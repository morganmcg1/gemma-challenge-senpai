#!/usr/bin/env python
"""PR #653 (lawine) -- aggregate the 3-arm AIME g32-recovery panel.

Per arm (shipped_g128 / ours_g32 / official_g32): AIME maj@1 acc, Wilson 95% CI,
%-of-bf16 (vs 0.4667), pass/fail vs the 0.420 (90%) bar, extract_fail + truncation
per cell, per-year. Then the two deltas (paired, same 60 problems):
  * group-size effect  : ours_g32  - shipped_g128   (same untied int4 head; HEADLINE)
  * recipe cross-check : official_g32 - shipped_g128 (Google's full g32 recipe)
McNemar discordant (b,c)+exact p and a Newcombe paired-difference 95% CI for each.
GPQA-vs-AIME contrast vs the +0.0283 GPQA group-size move. Plus the ubel-0.350 vs
denken-0.400 reconciliation: my shipped_g128 vs denken's saved AR completions.

Run under the eval venv (only needs stdlib):
  /tmp/eval-serve-venv/bin/python research/validity/aime_g32_recovery_653/aggregate.py
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

BF16 = 0.4667           # ubel #628 bf16 base AIME, gb6144, BI=1, mintok=8, n=60
BAR = 0.420             # 90% of bf16 (0.9 * 0.4667 = 0.420)
GPQA_GROUPSIZE_DELTA = 0.0283   # ours_g32 0.5152 - ours_g128 0.4869 (#639 GPQA-D sampled)
UBEL_SHIPPED = 0.3500   # ubel #638 u13z29hs (aime_eval.py + bf16_base serve, seqs=16)
DENKEN_SHIPPED_AR = 0.4000  # denken #637 dh0tbwpp (int4_mtp_batchinv serve, spec-off, seqs=1)


def wilson(k: int, n: int, z: float = 1.95996398454):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p (binomial, p=0.5) over discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def newcombe_paired_ci(n: int, e: int, f: int, g: int, h: int, z: float = 1.95996398454):
    """Newcombe method 10 (paired difference of proportions) 95% CI.
    2x2 paired table cells: e=both-correct, f=arm1-only, g=arm2-only, h=both-wrong.
    Difference = p1 - p2 = (f - g)/n  (arm1 - arm2).
    """
    if n == 0:
        return (0.0, 0.0)
    p1 = (e + f) / n
    p2 = (e + g) / n
    diff = p1 - p2
    l1, u1 = wilson(e + f, n, z)
    l2, u2 = wilson(e + g, n, z)
    # phi: 2x2 correlation correction (Newcombe 1998 method 10). The estimate is
    # the standard phi coefficient phi = (e*h - f*g)/sqrt(m1*m0*n1*n0) over the four
    # marginals m1=e+f, m0=g+h, n1=e+g, n0=f+h -- a PRODUCT under one sqrt, NOT a sum
    # of products (the latter under-normalizes phi, blows it past 1, clamps to 1.0,
    # and falsely narrows the CI to exclude 0 even when McNemar p=1.0).
    denom = (e + f) * (g + h) * (e + g) * (f + h)
    if denom <= 0:
        phi = 0.0
    else:
        phi = (e * h - f * g) / math.sqrt(denom)
        phi = max(min(phi, 1.0), -1.0)
    lo = diff - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return (lo, hi)


def load_arm(label: str):
    p = RES / f"{label}_aime_gb6144.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pp = d["per_problem"]
    n = d["n_problems"]
    k = d["n_correct_maj"]
    # truncation: any sample with finish_reason == 'length'
    trunc = sum(1 for r in pp if any(fr == "length" for fr in r["finish_reasons"]))
    ef = sum(1 for r in pp for a in r["answers"] if a is None)
    tot = d["total_samples"]
    # truncation-censored accuracy: maj@1 over only the problems that finished
    # (no 'length' sample). Isolates reasoning quality from the 6144-token cap.
    nontrunc = [r for r in pp if not any(fr == "length" for fr in r["finish_reasons"])]
    n_nt = len(nontrunc)
    k_nt = sum(1 for r in nontrunc if r["maj_correct"])
    censored_acc = (k_nt / n_nt) if n_nt else 0.0
    yr = defaultdict(lambda: [0, 0])
    for r in pp:
        yr[r["year"]][0] += int(r["maj_correct"]); yr[r["year"]][1] += 1
    lo, hi = wilson(k, n)
    return {
        "label": label, "acc": k / n, "n_correct": k, "n": n,
        "wilson": (lo, hi), "extract_fail": ef, "extract_fail_rate": d["extract_fail_rate"],
        "trunc": trunc, "trunc_rate": trunc / n,
        "censored_acc": censored_acc, "n_nontrunc": n_nt, "k_nontrunc": k_nt,
        "censored_pct_of_bf16": (censored_acc / BF16) if BF16 else 0.0,
        "pct_of_bf16": (k / n) / BF16, "clears_bar": (k / n) >= BAR,
        "per_year": {y: (yr[y][0], yr[y][1]) for y in sorted(yr)},
        "wall_min": d["wall_s"] / 60.0,
        "correct_by_id": {_norm_id(r["id"]): bool(r["maj_correct"]) for r in pp},
        "ans_by_id": {_norm_id(r["id"]): (r["maj_answer"], r["gold"]) for r in pp},
    }


def _norm_id(pid: str) -> str:
    """Normalize problem ids across harnesses (denken prefixes year twice, etc.)."""
    s = str(pid)
    # collapse a leading duplicate year like '2024-2024-II-4' -> '2024-II-4'
    m = re.match(r"^(\d{4})-(\d{4}-)", s)
    if m and m.group(1) == m.group(2)[:4]:
        s = s[len(m.group(1)) + 1:]
    return s


def paired(a1, a2):
    ids = sorted(set(a1["correct_by_id"]) & set(a2["correct_by_id"]))
    e = f = g = h = 0
    for i in ids:
        c1, c2 = a1["correct_by_id"][i], a2["correct_by_id"][i]
        if c1 and c2: e += 1
        elif c1 and not c2: f += 1
        elif (not c1) and c2: g += 1
        else: h += 1
    n = len(ids)
    diff = (f - g) / n if n else 0.0
    ci = newcombe_paired_ci(n, e, f, g, h)
    return {"n": n, "e_both": e, "f_a1only": f, "g_a2only": g, "h_neither": h,
            "delta": diff, "mcnemar_b": f, "mcnemar_c": g,
            "mcnemar_p": binom_two_sided_p(f, g), "newcombe95": ci}


def reconcile_vs_denken(shipped):
    """My shipped_g128 (aime_eval.py + bf16_base serve) vs denken's saved AR
    completions (int4_mtp_batchinv serve, spec-off). Same int4 body + shared
    prompt/extractor code, so any per-item answer flip is serve-path divergence."""
    dp = Path("research/validity/optionb_319_answer_materiality/results/ar_aime.jsonl")
    if not dp.exists() or shipped is None:
        return None
    denk = {}
    for line in dp.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        denk[_norm_id(r["id"])] = {"correct": bool(r["correct"]), "answer": r.get("answer"),
                                   "gold": r.get("gold"), "ntok": r.get("num_completion_tokens"),
                                   "finish": r.get("finish_reason")}
    def _as_int(x):
        # denken stores the extracted answer as a str ('204'); my harness stores
        # maj_answer as an int (204). A raw != flags ALL 60 as divergent purely on
        # the int-vs-str type mismatch. Normalize to int so the count reflects true
        # answer-value divergence, not a representation artifact.
        try:
            return int(x)
        except (TypeError, ValueError):
            return x
    ids = sorted(set(shipped["ans_by_id"]) & set(denk))
    ans_diff = corr_diff = 0
    flip_detail = []
    for i in ids:
        my_ans, my_gold = shipped["ans_by_id"][i]
        my_corr = shipped["correct_by_id"][i]
        d = denk[i]
        if _as_int(my_ans) != _as_int(d["answer"]):
            ans_diff += 1
        if my_corr != d["correct"]:
            corr_diff += 1
            flip_detail.append({"id": i, "gold": my_gold, "mine": (my_ans, my_corr),
                                "denken": (d["answer"], d["correct"]),
                                "denken_finish": d["finish"], "denken_ntok": d["ntok"]})
    denk_correct = sum(1 for i in ids if denk[i]["correct"])
    my_correct = sum(1 for i in ids if shipped["correct_by_id"][i])
    return {"n_common": len(ids), "my_acc": my_correct / len(ids) if ids else 0,
            "denken_acc": denk_correct / len(ids) if ids else 0,
            "answer_divergent_items": ans_diff, "correctness_flips": corr_diff,
            "flip_detail": flip_detail}


def fmt_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def main():
    arms = {}
    for label in ("shipped_g128", "ours_g32", "official_g32"):
        a = load_arm(label)
        if a:
            arms[label] = a

    out = {"bf16_anchor": BF16, "bar_90pct": BAR, "arms": {}, "deltas": {},
           "gpqa_groupsize_delta": GPQA_GROUPSIZE_DELTA}
    print("=" * 78)
    print("PR #653  AIME g32-recovery panel  (greedy maj@1, M=1 AR, gb6144, BI=1, seqs=1)")
    print(f"bf16 base anchor = {BF16}  |  90% bar = {BAR}")
    print("=" * 78)
    for label, a in arms.items():
        out["arms"][label] = {kk: vv for kk, vv in a.items()
                              if kk not in ("correct_by_id", "ans_by_id")}
        py = "  ".join(f"{y}:{c}/{t}" for y, (c, t) in a["per_year"].items())
        print(f"\n[{label}]  acc={a['acc']:.4f} ({a['n_correct']}/{a['n']})  "
              f"Wilson95={fmt_ci(a['wilson'])}")
        print(f"    %-of-bf16={a['pct_of_bf16']*100:.1f}%   clears {BAR} bar: "
              f"{'YES' if a['clears_bar'] else 'NO'}")
        print(f"    extract_fail={a['extract_fail']}/{a['n']*1} ({a['extract_fail_rate']*100:.1f}%)  "
              f"truncation={a['trunc']}/{a['n']} ({a['trunc_rate']*100:.1f}%)  "
              f"wall={a['wall_min']:.1f}min")
        print(f"    truncation-censored acc={a['censored_acc']:.4f} "
              f"({a['k_nontrunc']}/{a['n_nontrunc']} finished)  "
              f"censored %-of-bf16={a['censored_pct_of_bf16']*100:.1f}%")
        print(f"    per-year: {py}")

    # ---- deltas (paired) -----------------------------------------------------
    print("\n" + "-" * 78)
    print("DELTAS (paired, same 60 problems)")
    if "ours_g32" in arms and "shipped_g128" in arms:
        d = paired(arms["ours_g32"], arms["shipped_g128"])
        out["deltas"]["groupsize_ours_g32_minus_shipped_g128"] = d
        print(f"\n  HEADLINE group-size  (ours_g32 - shipped_g128):  Δ={d['delta']:+.4f}")
        print(f"    Newcombe95={fmt_ci(d['newcombe95'])}  McNemar b={d['mcnemar_b']} "
              f"c={d['mcnemar_c']} p={d['mcnemar_p']:.4f}")
        print(f"    cells: both={d['e_both']} ours_only={d['f_a1only']} "
              f"shipped_only={d['g_a2only']} neither={d['h_neither']}")
        frac = d["delta"] / GPQA_GROUPSIZE_DELTA if GPQA_GROUPSIZE_DELTA else float("nan")
        print(f"    GPQA-vs-AIME contrast: GPQA group-size moved +{GPQA_GROUPSIZE_DELTA:.4f}; "
              f"AIME moved {d['delta']:+.4f}  ({frac:+.2f}x the GPQA move)")
        out["gpqa_vs_aime_fraction"] = frac
    if "official_g32" in arms and "shipped_g128" in arms:
        d2 = paired(arms["official_g32"], arms["shipped_g128"])
        out["deltas"]["recipe_official_g32_minus_shipped_g128"] = d2
        print(f"\n  recipe cross-check  (official_g32 - shipped_g128):  Δ={d2['delta']:+.4f}")
        print(f"    Newcombe95={fmt_ci(d2['newcombe95'])}  McNemar b={d2['mcnemar_b']} "
              f"c={d2['mcnemar_c']} p={d2['mcnemar_p']:.4f}")

    # ---- reconciliation ------------------------------------------------------
    rec = reconcile_vs_denken(arms.get("shipped_g128"))
    if rec:
        out["reconciliation"] = rec
        print("\n" + "-" * 78)
        print("RECONCILIATION  ubel-0.350 vs denken-0.400  (shared prompt+extractor; "
              "differ only by serve stack)")
        print(f"    ubel #638 (bf16_base serve, seqs16) = {UBEL_SHIPPED:.4f}")
        print(f"    denken #637 (int4_mtp_batchinv serve spec-off, seqs1) = {DENKEN_SHIPPED_AR:.4f}")
        print(f"    MY shipped_g128 (bf16_base serve, seqs1) = {rec['my_acc']:.4f}  "
              f"(reproduces ubel path)")
        print(f"    n_common w/ denken AR = {rec['n_common']}  "
              f"answer-divergent items = {rec['answer_divergent_items']}  "
              f"correctness-flips = {rec['correctness_flips']}")
        print(f"    => the 0.350<->0.400 gap is serve-path greedy divergence on the "
              f"chaotic int4 body ({rec['correctness_flips']} of {rec['n_common']} answers flip "
              f"by serve stack alone).")

    (HERE / "panel_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[aggregate] wrote {HERE/'panel_summary.json'}")


if __name__ == "__main__":
    main()
