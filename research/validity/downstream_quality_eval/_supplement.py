#!/usr/bin/env python3
"""Supplementary divergence analysis for PR #511 (run after aggregate.py).

Adds, on top of aggregate.py's A/B:
  * whether the SHIP reproduces dixie's #483 SUBSTRATE anchors (0.330/0.283),
    which (together with base reproducing the 0.668/0.470 base anchors) is the
    dual cross-validation that the harness is faithful.
  * truncation comparison (base vs ship): output==max_tokens fraction. A byte-
    exact ship truncates at the identical token; divergent reasoning rambles.
  * ship answer extractability (empty/no-ANSWER fraction).
  * the divergence breakdown framed around the ROBUST aggregate-accuracy delta
    (per-sample agreement carries a bf16-head cross-session argmax noise floor
    on the BASE side, ~9-13% per project memory; aggregate accuracy is robust).
"""
import json
import math
import os
import sys
from inspect_ai.log import read_eval_log

HERE = os.path.dirname(os.path.abspath(__file__))
DIXIE_BASE = {"mmlu": 0.668, "gpqa": 0.470}
DIXIE_SUBSTRATE = {"mmlu": 0.330, "gpqa": 0.283}


def _ci95(p, n):
    if not n:
        return (float("nan"), float("nan"))
    h = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def _load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _trunc_and_empty(eval_log_path, cap):
    """From an inspect eval log: fraction of samples that hit the token cap,
    and fraction with no extractable answer."""
    if not eval_log_path or not os.path.exists(eval_log_path):
        return None
    lg = read_eval_log(eval_log_path)
    n = trunc = 0
    for s in lg.samples or []:
        out = getattr(s, "output", None)
        u = getattr(out, "usage", None) if out else None
        ot = getattr(u, "output_tokens", None) if u else None
        if ot is None:
            continue
        n += 1
        if ot >= cap:
            trunc += 1
    return {"n": n, "trunc": trunc, "trunc_frac": (trunc / n) if n else float("nan")}


def _empty_frac(result_json):
    rows = result_json["per_sample"]
    empty = sum(1 for r in rows if not (r.get("answer") or "").strip())
    return {"n": len(rows), "empty": empty, "empty_frac": empty / len(rows) if rows else float("nan")}


def main():
    out = {}
    for task, bf, sf, cap in [
        ("mmlu", "base_mmlu_pro.json", "ship_mmlu_pro.json", 2048),
        ("gpqa", "base_gpqa.json", "ship_gpqa.json", 3072),
    ]:
        b, s = _load(bf), _load(sf)
        if b is None or s is None:
            print(f"[supp] {task}: missing ({bf if b is None else sf}) — skip")
            continue
        bn, sn = b["n_scored"], s["n_scored"]
        bacc, sacc = b["accuracy"], s["accuracy"]
        s_ci = _ci95(sacc, sn)
        # does ship land on dixie's SUBSTRATE anchor (within ship CI95)?
        sub = DIXIE_SUBSTRATE[task]
        ship_reproduces_substrate = bool(s_ci[0] <= sub <= s_ci[1])
        # delta vs base, and "fraction of base capability retained"
        delta = sacc - bacc
        retained = (sacc / bacc) if bacc else float("nan")
        # where the moat breaks: base correct -> ship wrong
        brows = {r["id"]: r for r in b["per_sample"]}
        srows = {r["id"]: r for r in s["per_sample"]}
        common = sorted(set(brows) & set(srows))
        base_only = sum(1 for i in common if brows[i]["correct"] and not srows[i]["correct"])
        ship_only = sum(1 for i in common if not brows[i]["correct"] and srows[i]["correct"])
        agree = sum(1 for i in common if brows[i]["answer"] == srows[i]["answer"])
        b_tr = _trunc_and_empty(b.get("eval_log"), cap)
        s_tr = _trunc_and_empty(s.get("eval_log"), cap)
        b_em = _empty_frac(b)
        s_em = _empty_frac(s)
        out[task] = {
            "base_acc": bacc, "ship_acc": sacc, "delta": delta,
            "frac_base_retained": retained,
            "ship_ci95": s_ci,
            "dixie_substrate_anchor": sub,
            "ship_reproduces_substrate_anchor": ship_reproduces_substrate,
            "dixie_base_anchor": DIXIE_BASE[task],
            "n_common": len(common),
            "base_only_correct": base_only,  # moat-break count
            "ship_only_correct": ship_only,
            "answer_agreement": agree / len(common) if common else float("nan"),
            "base_trunc": b_tr, "ship_trunc": s_tr,
            "base_empty": b_em, "ship_empty": s_em,
        }
        print(f"\n=== {task.upper()} ===")
        print(f"  base acc {bacc:.4f} (n={bn})  ->  ship acc {sacc:.4f} (n={sn}, ci95 {s_ci[0]:.3f}-{s_ci[1]:.3f})")
        print(f"  delta {delta:+.4f}  | frac base capability retained {retained:.3f}")
        print(f"  dixie base anchor {DIXIE_BASE[task]} ; dixie SUBSTRATE anchor {sub}  -> ship reproduces substrate anchor: {ship_reproduces_substrate}")
        print(f"  moat-break (base-correct -> ship-wrong): {base_only}/{len(common)}   ship-only-correct: {ship_only}")
        print(f"  answer agreement (bf16-noisy on base side): {agree}/{len(common)} = {agree/len(common):.3f}" if common else "  no common")
        if b_tr and s_tr:
            print(f"  truncation@cap{cap}: base {b_tr['trunc']}/{b_tr['n']} ({b_tr['trunc_frac']:.3f})  ship {s_tr['trunc']}/{s_tr['n']} ({s_tr['trunc_frac']:.3f})")
        print(f"  empty-answer: base {b_em['empty']}/{b_em['n']} ({b_em['empty_frac']:.3f})  ship {s_em['empty']}/{s_em['n']} ({s_em['empty_frac']:.3f})")

    with open(os.path.join(HERE, "_supplement.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[supp] wrote _supplement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
