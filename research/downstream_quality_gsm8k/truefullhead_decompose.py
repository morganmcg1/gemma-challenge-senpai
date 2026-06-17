"""De-confound the GSM8K base->ship collapse at the ITEM level (PR #541).

PR #533 left an open question: the served ship-12k GSM8K is ~25-29% of base, a
-68pp collapse, but that arm bundles THREE changes at once -- (a) surgical
fast-kernels, (b) the osoi5 int4 re-quant+bake, (c) the 262k->16k->12k head prune.
This leg isolates change (a) ALONE: the ``truefullhead`` cell = stock base-int4
(native 262k head, NO osoi5 bake, NO head prune) + the surgical fast-kernel stack.

This script reads the committed per-arm eval JSONs (``per_problem`` records carry
``id``/``correct``/``finish_reason``/``sample_chars``) and computes, per regime:

  * item-level agreement + confusion of truefullhead vs base and vs ship-12k
    (the PR-required A/B at the item level);
  * the empty-completion rate of truefullhead (sample_chars == 0) -- the
    first-token-EOS artifact the empty_probe.py run pinned -- and how much of the
    base->truefullhead regression set those empties explain;
  * the min_tokens=8 EOS-guard RECOVERY: how many truefullhead misses (and how many
    of its empties specifically) the guard converts to correct.

Pure analysis over existing JSONs: no server, no GPU, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DIR = Path("research/downstream_quality_gsm8k")
REGIMES = ("sampled", "greedy")


def _by_id(label: str, regime: str) -> dict[str, dict[str, Any]] | None:
    p = DIR / f"{label}_{regime}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r["id"]: r for r in d["per_problem"]}


def _acc(rows: dict[str, dict[str, Any]]) -> float:
    return sum(1 for r in rows.values() if r["correct"]) / len(rows)


def _empty(r: dict[str, Any]) -> bool:
    return int(r.get("sample_chars") or 0) == 0


def _confusion(a: dict[str, dict], b: dict[str, dict]) -> dict[str, Any]:
    """A vs B over the shared id set: rows = A correctness, cols = B correctness."""
    ids = sorted(set(a) & set(b))
    both = a_only = b_only = neither = 0
    for i in ids:
        ca, cb = a[i]["correct"], b[i]["correct"]
        both += ca and cb
        a_only += ca and not cb
        b_only += cb and not ca
        neither += not ca and not cb
    n = len(ids)
    return {
        "n_shared": n,
        "both_correct": both,
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "both_wrong": neither,
        "agreement_rate": round((both + neither) / n, 4) if n else None,
    }


def _decompose(tfh: dict[str, dict], base: dict[str, dict], ship: dict[str, dict] | None,
               guard: dict[str, dict] | None) -> dict[str, Any]:
    ids = sorted(tfh)
    n = len(ids)
    n_empty = sum(_empty(tfh[i]) for i in ids)
    # base-correct -> truefullhead-wrong = the regressions introduced by fast kernels.
    regress = [i for i in ids if i in base and base[i]["correct"] and not tfh[i]["correct"]]
    regress_empty = [i for i in regress if _empty(tfh[i])]
    out: dict[str, Any] = {
        "n": n,
        "tfh_acc": round(_acc(tfh), 4),
        "tfh_empty_count": n_empty,
        "tfh_empty_rate": round(n_empty / n, 4),
        "vs_base": _confusion(base, tfh),  # rows=base, so a_correct_b_wrong = base-only = regressions
        "regressions_base_to_tfh": len(regress),
        "regressions_that_are_empty": len(regress_empty),
        "regression_empty_share": round(len(regress_empty) / len(regress), 4) if regress else None,
    }
    if ship is not None:
        out["vs_ship12k"] = _confusion(ship, tfh)
    if guard is not None:
        # Recovery: among truefullhead MISSES, how many does the guard fix; and among
        # truefullhead EMPTIES specifically, how many become non-empty & correct.
        misses = [i for i in ids if not tfh[i]["correct"] and i in guard]
        fixed = [i for i in misses if guard[i]["correct"]]
        empties = [i for i in ids if _empty(tfh[i]) and i in guard]
        empties_fixed = [i for i in empties if guard[i]["correct"] and not _empty(guard[i])]
        out["guard"] = {
            "guard_acc": round(_acc(guard), 4),
            "delta_acc": round(_acc(guard) - _acc(tfh), 4),
            "tfh_misses": len(misses),
            "misses_fixed_by_guard": len(fixed),
            "miss_fix_rate": round(len(fixed) / len(misses), 4) if misses else None,
            "tfh_empties": len(empties),
            "empties_recovered_correct": len(empties_fixed),
            "empty_recovery_rate": round(len(empties_fixed) / len(empties), 4) if empties else None,
            "guard_empty_count": sum(_empty(guard[i]) for i in guard),
        }
    return out


def _flat_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the headline numbers (sampled = PRIMARY) for the wandb summary."""
    flat: dict[str, Any] = {"analysis_only": True, "official_tps": 0}
    for regime in REGIMES:
        d = report["by_regime"].get(regime) or {}
        if d.get("missing"):
            continue
        sfx = "" if regime == "sampled" else "_greedy"
        flat[f"tfh_acc{sfx}"] = d["tfh_acc"]
        flat[f"tfh_empty_rate{sfx}"] = d["tfh_empty_rate"]
        flat[f"tfh_empty_count{sfx}"] = d["tfh_empty_count"]
        flat[f"regressions_base_to_tfh{sfx}"] = d["regressions_base_to_tfh"]
        flat[f"regression_empty_share{sfx}"] = d["regression_empty_share"]
        flat[f"agreement_vs_base{sfx}"] = d["vs_base"]["agreement_rate"]
        if "vs_ship12k" in d:
            flat[f"agreement_vs_ship12k{sfx}"] = d["vs_ship12k"]["agreement_rate"]
            flat[f"tfh_ok_ship_wrong{sfx}"] = d["vs_ship12k"]["b_correct_a_wrong"]
            flat[f"ship_ok_tfh_wrong{sfx}"] = d["vs_ship12k"]["a_correct_b_wrong"]
        if "guard" in d:
            g = d["guard"]
            flat[f"guard_acc{sfx}"] = g["guard_acc"]
            flat[f"guard_delta_acc{sfx}"] = g["delta_acc"]
            flat[f"guard_miss_fix_rate{sfx}"] = g["miss_fix_rate"]
            flat[f"guard_empty_recovery_rate{sfx}"] = g["empty_recovery_rate"]
            flat[f"guard_empties_left{sfx}"] = g["guard_empty_count"]
    return flat


def _wandb_log(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[decompose] wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="downstream-quality-gsm8k",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="PR #541: item-level de-confound of base->ship GSM8K collapse "
              "(truefullhead vs base/ship-12k) + first-token-EOS min_tokens=8 recovery.",
        tags=["gsm8k", "downstream-quality", "analysis-only", "pr-541",
              "de-confound", "first-token-eos"],
        config={"analysis_only": True, "official_tps": 0, "pr": 541, "regimes": list(REGIMES)},
    )
    if run is None:
        print("[decompose] wandb disabled/unavailable; skipping", flush=True)
        return None
    wandb_logging.log_summary(run, _flat_summary(report), step=0)
    wandb_logging.log_json_artifact(
        run, name="truefullhead_decompose", artifact_type="gsm8k-eval", data=report)
    rid = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DIR / "truefullhead_decompose.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="wirbel/gsm8k-truefullhead-decompose")
    ap.add_argument("--wandb-group", default="base-fullhead-gsm8k-layerdrop")
    args = ap.parse_args()

    report: dict[str, Any] = {"analysis_only": True, "pr": 541, "by_regime": {}}
    for regime in REGIMES:
        base = _by_id("base", regime)
        tfh = _by_id("truefullhead", regime)
        ship = _by_id("ship12k", regime)
        guard = _by_id("truefullhead_mintok8", regime)
        if base is None or tfh is None:
            report["by_regime"][regime] = {"missing": True}
            continue
        report["by_regime"][regime] = _decompose(tfh, base, ship, guard)

    if args.wandb:
        report["wandb_run_id"] = _wandb_log(report, args)

    args.out.write_text(json.dumps(report, indent=2))
    print(f"[decompose] wrote {args.out}")
    for regime, d in report["by_regime"].items():
        if d.get("missing"):
            print(f"[{regime}] MISSING base or truefullhead json")
            continue
        print(f"\n[{regime}] tfh_acc={d['tfh_acc']} empties={d['tfh_empty_count']}/{d['n']} "
              f"({d['tfh_empty_rate']:.1%})")
        vb = d["vs_base"]
        print(f"  vs base: agreement={vb['agreement_rate']:.1%}  "
              f"both_ok={vb['both_correct']}  base->tfh REGRESS={vb['a_correct_b_wrong']}  "
              f"tfh-only_gain={vb['b_correct_a_wrong']}  both_wrong={vb['both_wrong']}")
        print(f"    regressions={d['regressions_base_to_tfh']}, of which EMPTY="
              f"{d['regressions_that_are_empty']} (share={d['regression_empty_share']})")
        if "vs_ship12k" in d:
            vs = d["vs_ship12k"]
            print(f"  vs ship-12k: agreement={vs['agreement_rate']:.1%}  "
                  f"ship_ok_tfh_wrong={vs['a_correct_b_wrong']}  "
                  f"tfh_ok_ship_wrong={vs['b_correct_a_wrong']}  both_wrong={vs['both_wrong']}")
        if "guard" in d:
            g = d["guard"]
            print(f"  min_tokens=8 GUARD: acc={g['guard_acc']} (Δ={g['delta_acc']:+.4f})  "
                  f"misses_fixed={g['misses_fixed_by_guard']}/{g['tfh_misses']} "
                  f"({g['miss_fix_rate']})  empties_recovered={g['empties_recovered_correct']}/"
                  f"{g['tfh_empties']} ({g['empty_recovery_rate']})  guard_empties_left={g['guard_empty_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
