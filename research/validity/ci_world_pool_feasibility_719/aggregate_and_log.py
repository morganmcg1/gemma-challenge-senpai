"""PR #719 -- CI-world pool feasibility: census + comparability -> verdict + W&B.

Combines three inputs into the load-bearing artifact `admissible_iid_greedy_pool_size`
and the binary `ci_world_constructible`:

  1. CENSUS (recomputed here): distinct past-AIME supply from di-zhang-fdu/AIME_1983_2024
     (933 integer-answer problems), per-era distinct counts, dedup vs the canonical
     reference year(s), near-dup pruning result.
  2. COMPARABILITY (band_results.json from pastaime_eval.py): base-bf16 greedy rate per
     year-band, vs the canonical anchor +/- tolerance. A band is admissible iff its base
     rate sits inside the band (so ">=90% of base" means the same thing on it).
  3. #716 n(p) FRONTIER (carried verbatim): p=0.420->inf, 0.446->1385, 0.450->1040, 0.470->375.

Verdict: admissible_iid_greedy_pool_size >= 1040 (int8-locus's required n) -> CONSTRUCTIBLE,
else UNREACHABLE with the binding constraint (count vs comparability) and the shortfall.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# ---- #716 cert-budget frontier (carried verbatim) ------------------------------ #
N_OF_P = {0.420: math.inf, 0.446: 1385, 0.450: 1040, 0.470: 375}
INT8_LOCUS_N = 1040      # int8-locus greedy point 0.450 -> required n
FULL_G32_N = 2889        # full-g32 greedy point 0.438 -> required n
GATE_BAR = 0.420         # 90% of base 0.4667

# ---- Theoretical-maximum AIME universe (dataset-independent count argument) ----- #
# The ENTIRE history of AIME, EVERY problem ever set, is finite and small. AIME began
# 1983; one exam/year (15 problems) through 1999; the alternate AIME II was introduced
# in 2000, so two exams/year (30 problems) from 2000 on. Excluding the 2024+2025
# reference set, the complete non-reference universe (1983-2023) is:
#   1983-1999: 17 yr x 15 = 255   +   2000-2023: 24 yr x 30 = 720   = 975 problems.
# This is the most-generous same-format ceiling possible: it ignores difficulty-
# comparability AND contamination AND dataset incompleteness. 975 < 1040 -> the count
# constraint binds even against a perfectly complete past-AIME corpus.
THEORETICAL_MAX_AIME_1983_2023 = 255 + 720  # = 975 distinct same-format problems


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, center - half, center + half


def era_of(y: int) -> str:
    if y <= 1994:
        return "1983-1994"
    if y <= 2004:
        return "1995-2004"
    if y <= 2014:
        return "2005-2014"
    if y <= 2023:
        return "2015-2023"
    return "2024+"


def read_secondary(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load secondary_eval.py output -> {source: measured-row}. Empty if not provided."""
    if not path:
        return {}
    data = json.loads(path.read_text())
    return {r["source"]: r for r in data.get("sources", [])}


def census(exclude_years: set[int]) -> dict[str, Any]:
    from datasets import load_dataset
    import re

    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")

    def to_int(v: Any) -> int | None:
        s = str(v).strip().replace(",", "")
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None

    rows = []
    for r in ds:
        a = to_int(r["Answer"])
        if a is None or not (0 <= a <= 999):
            continue
        rows.append({"id": str(r["ID"]), "year": int(r["Year"])})
    total = len(rows)
    per_year: dict[int, int] = {}
    per_era: dict[str, int] = {}
    for r in rows:
        per_year[r["year"]] = per_year.get(r["year"], 0) + 1
        per_era[era_of(r["year"])] = per_era.get(era_of(r["year"]), 0) + 1
    n_excluded = sum(1 for r in rows if r["year"] in exclude_years)
    return {
        "source": "di-zhang-fdu/AIME_1983_2024",
        "distinct_total": total,
        "per_year": dict(sorted(per_year.items())),
        "per_era": dict(sorted(per_era.items())),
        "excluded_years": sorted(exclude_years),
        "n_excluded_overlap": n_excluded,
        "distinct_after_dedup": total - n_excluded,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band-results", type=Path, nargs="+", required=True,
                    help="one or more pastaime band_results.json; bands are concatenated")
    ap.add_argument("--secondary-results", type=Path, default=None,
                    help="secondary_eval.py output (AMC / MATH-L5 base greedy)")
    ap.add_argument("--anchor-acc", type=float, required=True, help="measured base greedy on canonical ref (same harness)")
    ap.add_argument("--anchor-n", type=int, required=True)
    ap.add_argument("--anchor-label", default="AIME2024+2025")
    ap.add_argument("--tol", type=float, default=0.05, help="comparability tolerance (absolute) around anchor")
    ap.add_argument("--exclude-years", default="2024")
    ap.add_argument("--near-dup-removed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    exclude = {int(y) for y in args.exclude_years.split(",") if y.strip()}
    cen = census(exclude)
    bands: list[dict[str, Any]] = []
    for bf in args.band_results:
        bands.extend(json.loads(bf.read_text())["bands"])

    band_lo, band_hi = args.anchor_acc - args.tol, args.anchor_acc + args.tol
    sec_measured = read_secondary(args.secondary_results)

    # Map each past-AIME band -> distinct supply in its year range (dedup'd), and
    # comparability. past-AIME is integer-answer -> grader is gate-faithful by
    # construction; admissibility reduces to in-band comparability.
    table: list[dict[str, Any]] = []
    covered_years: set[int] = set()
    for b in bands:
        lo, hi = b["lo"], b["hi"]
        distinct = sum(c for y, c in cen["per_year"].items() if lo <= y <= hi and y not in exclude)
        for y in range(lo, hi + 1):
            if y not in exclude and y in cen["per_year"]:
                covered_years.add(y)
        comparable = band_lo <= b["acc"] <= band_hi
        table.append({
            "source": f"past-AIME {b['band']}",
            "family": "past-AIME",
            "sampled_n": b["n"],
            "base_greedy": round(b["acc"], 4),
            "wilson": [round(b["wilson_lo"], 4), round(b["wilson_hi"], 4)],
            "delta_vs_anchor": round(b["acc"] - args.anchor_acc, 4),
            "grader_faithful": True,
            "in_band": comparable,
            "admissible": comparable,
            "distinct_available": distinct,
            "note": "AIME integer-answer 0-999; same grader as canonical anchor",
        })

    # Secondary sources (AMC / MATH-L5): base greedy MEASURED on the same live bf16
    # server (instruction #2), but EXCLUDED from the admissible count on the card's
    # named INSTRUMENT/GRADER criterion ("multiple-choice / open-form vs AIME
    # integer-answer"), independent of the measured difficulty. The measured rate is
    # shown to CONFIRM the exclusion (AMC: too-easy qualifier tier) and for a complete
    # per-source row; admissible stays False because the source is not the gate's
    # free-form integer-answer AIME instrument.
    #   * AMC-12 (AI-MO/aimo-validation-amc): 77/83 answers normalize to ints 0-999, so
    #     a numeric grader IS bolt-on-able, but the SOURCE instrument is 5-way
    #     multiple-choice and AMC is, by construction, the exam that QUALIFIES entrants
    #     for AIME -> strictly easier difficulty class. Excluded on format+difficulty.
    #   * MATH level-5 (nlile/hendrycks-MATH-benchmark, test, level==5): only 65/134
    #     (48.5%) gold answers are bare integers 0-999; 51.5% are fractions/expressions/
    #     tuples -> gate boxed-int-0-999 grader unfaithful on the MAJORITY; the gradeable
    #     48.5% is a biased non-contest slice (different instrument than AIME).
    def sec_row(source: str, distinct_if_faithful: int, note: str,
                aime_integer_gold_frac: float | None = None) -> dict[str, Any]:
        m = sec_measured.get(source)
        acc = m["acc"] if m else None
        n = m["n_graded"] if m else 0
        wl = m["wilson_lo"] if m else None
        wh = m["wilson_hi"] if m else None
        in_band = bool(acc is not None and band_lo <= acc <= band_hi)
        row: dict[str, Any] = {
            "source": f"secondary:{source}",
            "family": "secondary",
            "sampled_n": n,
            "base_greedy": round(acc, 4) if acc is not None else None,
            "wilson": [round(wl, 4) if wl is not None else None,
                       round(wh, 4) if wh is not None else None],
            "delta_vs_anchor": round(acc - args.anchor_acc, 4) if acc is not None else None,
            "grader_faithful": False,   # not the gate's free-form integer-answer AIME instrument
            "in_band": in_band,
            "admissible": False,        # excluded on instrument/grader mismatch regardless of rate
            "distinct_available": 0,
            "distinct_if_faithful": distinct_if_faithful,
            "note": note,
        }
        if aime_integer_gold_frac is not None:
            row["aime_integer_gold_frac"] = aime_integer_gold_frac
        return row

    table.append(sec_row(
        "amc", 77,
        "AI-MO/aimo-validation-amc: 77/83 integer-gradeable, but source instrument is 5-way "
        "MULTIPLE-CHOICE and AMC is the exam that QUALIFIES for AIME -> strictly easier "
        "difficulty class. Excluded on format+difficulty (measured base rate confirms too-easy)."))
    table.append(sec_row(
        "math_l5", 65,
        "nlile/hendrycks-MATH-benchmark test L5: only 65/134 (48.5%) golds are bare integers "
        "0-999; 51.5% non-integer -> gate boxed-int-0-999 grader unfaithful on the majority; "
        "gradeable subset is a biased non-contest slice (different instrument than AIME).",
        aime_integer_gold_frac=0.485))

    # Admissible past-AIME pool. If the sampled bands cover the whole dedup'd corpus
    # AND are all in-band, the admissible past-AIME count == the full dedup ceiling.
    past_aime_admissible = sum(t["distinct_available"] for t in table
                               if t["family"] == "past-AIME" and t["admissible"])
    secondary_admissible = sum(t["distinct_available"] for t in table
                               if t["family"] == "secondary" and t["admissible"])
    admissible = past_aime_admissible + secondary_admissible - args.near_dup_removed
    # most-generous (ignore comparability) same-format ceiling = full dedup'd past-AIME corpus
    raw_ceiling = cen["distinct_after_dedup"]
    n_years_total = len([y for y in cen["per_year"] if y not in exclude])
    coverage_complete = len(covered_years) == n_years_total

    # Theoretical-max ceiling: the COMPLETE non-reference AIME universe (1983-2023),
    # dataset-independent. This is the most-generous count possible (every problem
    # ever set, ignore comparability + contamination + dataset gaps). If even THIS is
    # < 1040, no past-AIME corpus — however complete — can CI-certify the int8 locus.
    theoretical_max = THEORETICAL_MAX_AIME_1983_2023      # 975
    constructible = admissible >= INT8_LOCUS_N
    # count binds against the MOST GENEROUS same-format ceiling (complete AIME universe)
    past_aime_count_short = theoretical_max < INT8_LOCUS_N    # 975 < 1040 -> True
    secondary_rescue_available = secondary_admissible > 0  # any count-sufficient comparable secondary
    if constructible:
        binding = "none"
    elif past_aime_count_short and not secondary_rescue_available:
        binding = "count(past-AIME-universe)_AND_grader/format(secondary-rescue)"
    elif past_aime_count_short:
        binding = "count(past-AIME-universe)"
    else:
        binding = "comparability"

    verdict = "CI_WORLD_CONSTRUCTIBLE" if constructible else "CI_WORLD_UNREACHABLE"
    shortfall_count = INT8_LOCUS_N - raw_ceiling             # realized corpus shortfall
    shortfall_theoretical = INT8_LOCUS_N - theoretical_max   # complete-universe shortfall (the robust one)
    shortfall_comparable = INT8_LOCUS_N - admissible

    summary = {
        "verdict": verdict,
        "ci_world_constructible": int(constructible),
        "admissible_iid_greedy_pool_size": admissible,
        "past_aime_admissible": past_aime_admissible,
        "secondary_admissible": secondary_admissible,
        "raw_dedup_ceiling": raw_ceiling,
        "theoretical_max_complete_aime_1983_2023": theoretical_max,
        "past_aime_coverage_complete": int(coverage_complete),
        "past_aime_years_covered": len(covered_years),
        "past_aime_years_total": n_years_total,
        "required_n_int8_locus": INT8_LOCUS_N,
        "required_n_full_g32": FULL_G32_N,
        "shortfall_vs_1040_raw_count": shortfall_count,
        "shortfall_vs_1040_theoretical_max": shortfall_theoretical,
        "shortfall_vs_1040_comparable": shortfall_comparable,
        "shortfall_vs_2889_theoretical_max": FULL_G32_N - theoretical_max,
        "binding_constraint": binding,
        "anchor": {"label": args.anchor_label, "base_greedy": args.anchor_acc, "n": args.anchor_n},
        "comparability_band": [round(band_lo, 4), round(band_hi, 4)],
        "tolerance_abs": args.tol,
        "gate_bar": GATE_BAR,
        "n_of_p_frontier": {str(k): (None if v == math.inf else v) for k, v in N_OF_P.items()},
        "census": cen,
        "near_dup_removed": args.near_dup_removed,
        "per_source_table": table,
        "analysis_only": 1,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": 0,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "verdict", "admissible_iid_greedy_pool_size", "raw_dedup_ceiling",
        "theoretical_max_complete_aime_1983_2023", "shortfall_vs_1040_raw_count",
        "shortfall_vs_1040_theoretical_max", "shortfall_vs_1040_comparable", "binding_constraint",
    )}, indent=2))
    for t in table:
        bg = f"{t['base_greedy']:.3f}" if t["base_greedy"] is not None else "  n/a"
        dd = f"{t['delta_vs_anchor']:+.3f}" if t["delta_vs_anchor"] is not None else "  n/a"
        print(f"  {t['source']:>24}: base={bg} wilson={t['wilson']} "
              f"d={dd} grader={int(t['grader_faithful'])} in_band={int(t['in_band'])} "
              f"adm={int(t['admissible'])} distinct={t['distinct_available']}")

    if args.wandb:
        import wandb
        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="gemma-challenge-senpai",
            group="ci-world-pool-feasibility-denken",
            name="denken/ci-world-pool-feasibility",
            job_type="analysis",
            config={
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "anchor_base_greedy": args.anchor_acc, "tolerance_abs": args.tol,
                "comparability_band_lo": band_lo, "comparability_band_hi": band_hi,
                "required_n_int8_locus": INT8_LOCUS_N, "required_n_full_g32": FULL_G32_N,
                "gate_bar": GATE_BAR, "exclude_years": sorted(exclude),
            },
        )
        run.summary.update({
            "verdict": verdict,
            "ci_world_constructible": int(constructible),
            "admissible_iid_greedy_pool_size": admissible,
            "past_aime_admissible": past_aime_admissible,
            "secondary_admissible": secondary_admissible,
            "raw_dedup_ceiling": raw_ceiling,
            "theoretical_max_complete_aime_1983_2023": theoretical_max,
            "past_aime_coverage_complete": int(coverage_complete),
            "shortfall_vs_1040_raw_count": shortfall_count,
            "shortfall_vs_1040_theoretical_max": shortfall_theoretical,
            "shortfall_vs_1040_comparable": shortfall_comparable,
            "binding_constraint": binding,
            "anchor_base_greedy_measured": args.anchor_acc,
            "primary_metric": admissible,
            "test_metric": int(constructible),
        })
        tbl = wandb.Table(columns=["source", "family", "sampled_n", "base_greedy", "wilson_lo", "wilson_hi",
                                   "delta_vs_anchor", "grader_faithful", "in_band", "admissible", "distinct_available"])
        for t in table:
            tbl.add_data(t["source"], t["family"], t["sampled_n"], t["base_greedy"], t["wilson"][0], t["wilson"][1],
                         t["delta_vs_anchor"], t["grader_faithful"], t["in_band"], t["admissible"], t["distinct_available"])
        run.log({"per_source_table": tbl})
        eratbl = wandb.Table(columns=["era", "distinct"])
        for e, c in cen["per_era"].items():
            eratbl.add_data(e, c)
        run.log({"era_distinct_table": eratbl})
        print(f"[wandb] run {run.id} ({run.url})")
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
