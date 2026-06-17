#!/usr/bin/env python3
"""PR #624 — Option-B 4-leg quality-panel roll-up (GREEDY gate reads @ gb6144).

Reads the 3 fresh greedy legs (MMLU-Pro / GSM8K / AIME) from results-greedy/ plus
the #612 GPQA gb6144 pooled number (gb6144_pooled.json, 0.4764), computes a Wilson
95% CI + pass/fail vs the authoritative gate bar + truncation rate for each leg,
and emits the 4-leg table + `optionb_all_four_legs_pass`.

Gate bars + base denominators are the authoritative post-#581/#580 set (BASELINE.md
L103/L110; research/validity/base_quality_denominator + base_fullhead_aime_n60):
  leg          bar     base(sampled-mean)   base(greedy-anchor)
  mmlu_pro     0.605   0.6727               0.678
  gpqa_diamond 0.471   0.5236               0.5253     (spec #612 = 0.4764, SAMPLED)
  gsm8k        0.807   0.8967               0.904
  aime         0.090   0.100 (greedy 6/60)  int4_base_fullhead 0.1167 (7/60)

Decode = GREEDY for all 3 fresh legs (PR #624 directive; AIME base is greedy; the
MMLU/GSM8K greedy-anchors sit above their bars). GPQA is reused from #612 where it
was SAMPLED+pooled(6 seeds) — flagged as a decode-mode caveat in the table.

LOCAL ONLY. analysis_only=True, official_tps=0. Optional W&B: --wandb.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESG = HERE / "results-greedy"

# Authoritative gate table (PR #581 + #580). base_greedy = greedy_anchor from the
# base denominator (verdict_marker.json); AIME base IS greedy (0.100 = 6/60).
GATES = {
    "mmlu_pro":     {"bar": 0.605, "base_sampled": 0.6727, "base_greedy": 0.678,  "order": 1},
    "gpqa_diamond": {"bar": 0.471, "base_sampled": 0.5236, "base_greedy": 0.5253, "order": 2},
    "gsm8k":        {"bar": 0.807, "base_sampled": 0.8967, "base_greedy": 0.904,  "order": 3},
    "aime":         {"bar": 0.090, "base_sampled": 0.100,  "base_greedy": 0.100,  "order": 4,
                     "base_int4_fullhead": 0.1167},
}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def leg(name: str, acc: float, n_correct: int, n: int, trunc_rate: float,
        decode: str, extra: dict | None = None) -> dict:
    g = GATES[name]
    lo, hi = wilson(acc, n)
    se = math.sqrt(acc * (1 - acc) / n) if n else float("nan")
    row = {
        "leg": name, "decode": decode, "accuracy": acc,
        "n_correct": n_correct, "n": n,
        "bar": g["bar"], "pass": bool(acc >= g["bar"]),
        "margin_over_bar": acc - g["bar"],
        "sigma_vs_bar": (acc - g["bar"]) / se if se else None,
        "ci95_wilson": [lo, hi],
        "ci95_wilson_lo_clears_bar": bool(lo >= g["bar"]),
        "truncation_rate": trunc_rate,
        "base_sampled": g["base_sampled"], "base_greedy": g["base_greedy"],
        "pct_of_base_greedy": acc / g["base_greedy"] if g["base_greedy"] else None,
        "order": g["order"],
    }
    if extra:
        row.update(extra)
    return row


def load_mmlu() -> dict | None:
    p = RESG / "spec_mmlu_pro_greedy_gb6144.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return leg("mmlu_pro", d["accuracy"], d["n_correct"], d["n_scored"],
               d.get("length_stop_rate", 0.0), d.get("decode", "greedy"),
               {"empty_rate": d.get("empty_rate"), "n_samples": d.get("n_samples"),
                "completion_tokens_mean": d.get("completion_tokens_mean"),
                "completion_tokens_p95": d.get("completion_tokens_p95"),
                # ctx-ceiling artifact flag (PR #624 instr.3: input+6144>8192 -> 'model_length')
                "n_stop_model_length": d.get("n_stop_model_length"),
                "n_stop_max_tokens": d.get("n_stop_max_tokens"),
                "n_error": d.get("n_error"),
                "max_tokens": d.get("max_tokens"), "seed": d.get("seed")})


def load_gsm8k() -> dict | None:
    p = RESG / "spec_greedy_gb6144_greedy.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return leg("gsm8k", d["accuracy"], d["n_correct"], d["n_problems"],
               d.get("truncation_rate", 0.0), d.get("regime", "greedy"),
               {"strict_rate": d.get("strict_rate"),
                "extract_fail_rate": d.get("extract_fail_rate"),
                "n_shot": d.get("n_shot"), "seed": d.get("seed"),
                "max_tokens": d.get("sampling", {}).get("max_tokens")})


def load_aime() -> dict | None:
    p = RESG / "spec_aime_greedy_gb6144.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    n = d["n_problems"]
    # truncation: any sample hit finish_reason 'length' (k=1 -> one per problem)
    n_trunc = 0
    for pp in d.get("per_problem", []):
        if any(fr == "length" for fr in (pp.get("finish_reasons") or [])):
            n_trunc += 1
    return leg("aime", d["maj_k_accuracy"], d["n_correct_maj"], n,
               n_trunc / n if n else 0.0, "greedy (maj@1, no-thinking)",
               {"mean_pass_rate": d.get("mean_pass_rate"),
                "extract_fail_rate": d.get("extract_fail_rate"),
                "maj_k": d.get("maj_k"), "years": d.get("years"),
                "seed": d.get("sampling", {}).get("seed"),
                "base_int4_fullhead": GATES["aime"].get("base_int4_fullhead"),
                "max_tokens": d.get("sampling", {}).get("max_tokens")})


def load_gpqa_612() -> dict | None:
    p = HERE / "gb6144_pooled.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    acc = d["pooled_accuracy"]
    g = GATES["gpqa_diamond"]
    row = {
        "leg": "gpqa_diamond", "decode": "SAMPLED (T=1.0, 6-seed pool) [from #612]",
        "accuracy": acc, "n_correct": d["n_correct"], "n": d["n_scored"],
        "bar": g["bar"], "pass": bool(d.get("pass", acc >= g["bar"])),
        "margin_over_bar": acc - g["bar"],
        "sigma_vs_bar": d.get("sigma_vs_bar"),
        "ci95_wilson": d.get("ci95_wilson"),
        "ci95_wilson_lo_clears_bar": bool(d.get("ci95_wilson", [0, 0])[0] >= g["bar"]),
        "truncation_rate": d.get("pooled_length_stop_rate", 0.0),
        "base_sampled": g["base_sampled"], "base_greedy": g["base_greedy"],
        "pct_of_base_greedy": acc / g["base_greedy"],
        "order": g["order"], "seeds": d.get("seeds"), "source_pr": 612,
        "note": "decode-mode caveat: GPQA reused from #612 (sampled); the 3 fresh "
                "legs are greedy. base bar derived from sampled base mean.",
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true", help="log panel + per-leg to W&B")
    args = ap.parse_args()

    rows = [r for r in (load_mmlu(), load_gpqa_612(), load_gsm8k(), load_aime()) if r]
    rows.sort(key=lambda r: r["order"])
    present = {r["leg"] for r in rows}
    all_four = present == {"mmlu_pro", "gpqa_diamond", "gsm8k", "aime"}
    all_pass = all_four and all(r["pass"] for r in rows)

    panel = {
        "pr": 624, "analysis_only": True, "official_tps": 0,
        "decode_fresh_legs": "greedy (temp=0, top_p=1, top_k off), min_tokens=8",
        "budget": "gb6144 (max_tokens=6144, max_model_len=8192 ctx)",
        "serve_config": "dev307, int4_g128_lmhead + MTP-K7 (BI=1), max_num_seqs=16",
        "legs": rows,
        "optionb_all_four_legs_pass": bool(all_pass),
        "legs_present": sorted(present),
        "denominator_caveat": "measured on dev307; accuracy-validity vs 0.22.0 under "
                              "audit (lawine #615 / ubel #614 / kanna #610). Pass/fail "
                              "provisional on the dev307-fair-denominator resolution.",
    }
    (HERE / "panel_greedy_gb6144.json").write_text(json.dumps(panel, indent=2))

    print("\n=== Option-B 4-leg quality panel (greedy gate reads @ gb6144) ===")
    print(f"{'leg':14s} {'acc':>7s} {'bar':>6s} {'CI95(Wilson)':>20s} {'trunc':>6s} "
          f"{'%base_g':>7s}  verdict  decode")
    for r in rows:
        ci = r.get("ci95_wilson") or [float('nan'), float('nan')]
        pofb = r.get("pct_of_base_greedy") or float('nan')
        print(f"{r['leg']:14s} {r['accuracy']:7.4f} {r['bar']:6.3f} "
              f"[{ci[0]:6.4f},{ci[1]:6.4f}] {r['truncation_rate']:6.3f} "
              f"{pofb:7.3f}  {'PASS' if r['pass'] else 'FAIL':4s}  {r['decode']}")
    print(f"\noptionb_all_four_legs_pass = {all_pass}  (legs present: {sorted(present)})")

    if args.wandb:
        _log_wandb(panel, rows, all_pass)
    return 0


def _log_wandb(panel: dict, rows: list[dict], all_pass: bool) -> None:
    import wandb
    ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    GROUP = "optionb-quality-panel-gb6144"
    COMMON = {
        "config_name": "int4_g128_lmhead + MTP-K7 spec (fern #597)",
        "vllm_version": "0.22.1rc1.dev307+g3e8afdf78", "max_model_len": 8192,
        "max_num_seqs": 16, "min_tokens": 8, "max_tokens": 6144,
        "is_319_identical": False, "analysis_only": True, "official_tps": 0,
        "budget": "gb6144", "decode_fresh_legs": "greedy",
    }
    for r in rows:
        run = wandb.init(project=PROJECT, entity=ENTITY, group=GROUP,
                         name=f"fern/panel-{r['leg']}-greedy-gb6144",
                         job_type="quality-eval", reinit=True,
                         config={**COMMON, "leg": r["leg"], "decode": r["decode"],
                                 "bar": r["bar"]})
        ci = r.get("ci95_wilson") or [None, None]
        wandb.log({f"{r['leg']}_acc_gb6144": r["accuracy"], "bar": r["bar"],
                   "pass": int(r["pass"]), "margin_over_bar": r["margin_over_bar"],
                   "truncation_rate": r["truncation_rate"], "n": r["n"],
                   "n_correct": r["n_correct"], "ci95_wilson_lo": ci[0],
                   "ci95_wilson_hi": ci[1], "pct_of_base_greedy": r.get("pct_of_base_greedy"),
                   "sigma_vs_bar": r.get("sigma_vs_bar")})
        run.summary[f"{r['leg']}_acc_gb6144"] = r["accuracy"]
        run.summary["pass"] = bool(r["pass"])
        run.summary["truncation_rate"] = r["truncation_rate"]
        run.finish()
    run = wandb.init(project=PROJECT, entity=ENTITY, group=GROUP,
                     name="fern/panel-VERDICT-greedy-gb6144", job_type="verdict",
                     reinit=True, config={**COMMON, "verdict": "4-leg-rollup"})
    by = {r["leg"]: r for r in rows}
    vlog = {"optionb_all_four_legs_pass": int(all_pass)}
    for lg in ("mmlu_pro", "gsm8k", "aime"):
        if lg in by:
            vlog[f"{lg}_acc_gb6144"] = by[lg]["accuracy"]
            vlog[f"{lg}_trunc_rate"] = by[lg]["truncation_rate"]
            vlog[f"{lg}_pass"] = int(by[lg]["pass"])
    if "gpqa_diamond" in by:
        vlog["gpqa_acc_gb6144_from612"] = by["gpqa_diamond"]["accuracy"]
        vlog["gpqa_pass"] = int(by["gpqa_diamond"]["pass"])
    wandb.log(vlog)
    run.summary["optionb_all_four_legs_pass"] = bool(all_pass)
    run.summary["surface_to_human"] = True
    run.finish()
    print(f"[wandb] logged {len(rows)} legs + verdict to {ENTITY}/{PROJECT} group={GROUP}")


if __name__ == "__main__":
    raise SystemExit(main())
