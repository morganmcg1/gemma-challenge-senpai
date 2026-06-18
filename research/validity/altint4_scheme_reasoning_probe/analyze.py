#!/usr/bin/env python
"""PR #647 — analyze the ALT int4-g128 (GPTQ) scheme-vs-scheme reasoning probe.

Three points per bar (GPQA-D n=198, AIME n=60):
  * bf16  — CITED from ubel #628 (gb6144, BI=1, mintok=8, vLLM 0.22.0): GPQA-D 0.4899,
    AIME 0.4667. Not recomputed here (disk/budget); the cite is the reference column.
  * QAT-int4 — the live body, M=1 AR, from denken #637 banked AR (ar_gpqa.jsonl n=198,
    ar_aime.jsonl n=60). Self-consistent QAT point (same harness as the ALT).
  * ALT-int4 (GPTQ-g128) — this PR, M=1 AR, from results/alt_ar_{gpqa,aime}.jsonl.

Per bar we report alt accuracy + Wilson CI, pct_of_bf16 (alt/bf16), and the SCHEME-DELTA
(alt − QAT) with a paired item bootstrap CI + McNemar (the ALT and QAT cells share item
ids and identical eval config, so the pairing is exact). truncation_rate + extract_fail
are reported per cell. analysis_only=True, official_tps=0.

Verdict (PR #647):
  LOSS_IS_RECIPE_SPECIFIC   — ALT recovers the binding bar(s) materially closer to bf16
                              than QAT (clears/approaches 90%, positive scheme-delta).
  LOSS_IS_INTRINSIC_TO_INT4 — ALT ≈ QAT (scheme-delta CI ∋ 0) on both bars.
  QAT_IS_NEAR_FRONTIER      — ALT is materially WORSE than QAT (negative delta excl. 0).

Usage:
  analyze.py                 # reads results/, writes results/summary.json (+ W&B if keyed)
  analyze.py --no-wandb
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
QAT_RES = ROOT / "research" / "validity" / "optionb_319_answer_materiality" / "results"

for p in (
    str(ROOT),
    str(ROOT / "research" / "specdec_raw_flip_rate"),
    str(ROOT / "research" / "validity" / "spec_distribution_preservation_matched_arm"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from flip_rate import wilson_ci  # noqa: E402
from analyze_matched_arm import cluster_bootstrap, mcnemar  # noqa: E402

# --- cited bf16 reference (ubel #628, gb6144 BI=1 mintok=8 vLLM 0.22.0) ---
BF16 = {"gpqa": 0.4899, "aime": 0.4667}
BAR90 = {"gpqa": 0.4409, "aime": 0.4200}   # 90%-of-bf16 bars (PR #647)
BF16_RUN = {"gpqa": "g3cig1xo", "aime": "zoszxnb0"}
N_EXPECT = {"gpqa": 198, "aime": 60}


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def _is_extract_fail(r: dict) -> bool:
    """Genuine scorer/parse failure: a COMPLETE generation (finish_reason not 'length'
    or 'error', no request error) that the scorer still could not parse. A truncated
    generation that never reached an answer is counted in truncation_rate, and a request
    error in n_error — neither is an extract failure. This isolates true parse bugs, so
    the PR's `extract_fail=0` clean-read check is not contaminated by model truncations
    (e.g. the QAT GPQA cell's 2 answer=None items are both finish_reason=length)."""
    if r.get("finish_reason") in ("length", "error") or r.get("error"):
        return False
    return r.get("extract_mode") == "error" or r.get("answer") is None


def _trunc(r: dict) -> bool:
    return r.get("finish_reason") == "length"


def cell_stats(rows: dict[str, dict], kind: str) -> dict[str, Any]:
    n = len(rows)
    corr = sum(1 for r in rows.values() if r.get("correct"))
    trunc = sum(1 for r in rows.values() if _trunc(r))
    xfail = sum(1 for r in rows.values() if _is_extract_fail(r))
    err = sum(1 for r in rows.values() if r.get("error"))
    acc = corr / n if n else float("nan")
    lo, hi = wilson_ci(corr, n) if n else (float("nan"), float("nan"))
    return {
        "n": n, "correct": corr, "acc": acc, "acc_wilson95": [lo, hi],
        "truncation_rate": trunc / n if n else float("nan"), "n_truncated": trunc,
        "extract_fail": xfail, "n_error": err,
    }


def scheme_delta(alt: dict[str, dict], qat: dict[str, dict]) -> dict[str, Any]:
    """Paired ALT−QAT on shared item ids: item bootstrap CI + McNemar."""
    ids = sorted(set(alt) & set(qat))
    # pairing integrity: ALT and QAT must score the SAME prompt per id (ALT loads the
    # banked QAT prompt_token_ids, so this should be 0 mismatches — a non-zero count
    # means the pairing is broken and the scheme-delta is invalid).
    n_prompt_mismatch = sum(
        1 for i in ids
        if alt[i].get("prompt_sha256") and qat[i].get("prompt_sha256")
        and alt[i]["prompt_sha256"] != qat[i]["prompt_sha256"]
    )
    a = np.array([1 if alt[i].get("correct") else 0 for i in ids], dtype=float)
    q = np.array([1 if qat[i].get("correct") else 0 for i in ids], dtype=float)
    cl = np.array([hash(i) % (2**31) for i in ids])      # each item = its own cluster
    cb = cluster_bootstrap(cl, a, q)                      # spec=alt, ar=qat -> delta=alt−qat
    mc = mcnemar([(int(a[k]), int(q[k])) for k in range(len(ids))])
    return {
        "n_paired": len(ids), "n_prompt_mismatch": n_prompt_mismatch,
        "n_alt_only": len(set(alt) - set(qat)), "n_qat_only": len(set(qat) - set(alt)),
        "alt_acc": cb["spec_acc"], "qat_acc": cb["ar_acc"],
        "delta_alt_minus_qat": cb["delta"], "delta_ci95": cb["delta_ci95"],
        "delta_se_boot": cb["delta_se_boot"],
        "delta_ci_excludes_0_positive": cb["delta_ci95"][0] > 0.0,
        "delta_ci_excludes_0_negative": cb["delta_ci95"][1] < 0.0,
        "delta_ci_contains_0": cb["delta_ci95"][0] <= 0.0 <= cb["delta_ci95"][1],
        "mcnemar": mc,
    }


def verdict(bars: dict[str, dict]) -> dict[str, Any]:
    """Combine per-bar scheme-deltas + 90%-bar clearance into the PR #647 verdict.

    Binding bars are those where QAT-int4 falls below the 90% bar (the loss the PR is
    chasing). RECIPE_SPECIFIC requires the ALT to materially recover a binding bar
    (positive scheme-delta excluding 0 AND alt clears/approaches the 90% bar).
    QAT_NEAR_FRONTIER if the ALT is materially worse on any bar with none better.
    Else INTRINSIC (ALT ≈ QAT)."""
    any_better = any(b["scheme_delta"]["delta_ci_excludes_0_positive"] for b in bars.values())
    any_worse = any(b["scheme_delta"]["delta_ci_excludes_0_negative"] for b in bars.values())
    binding = {k: b for k, b in bars.items() if b["qat"]["acc"] < BAR90[k]}
    recovered = []
    for k, b in binding.items():
        sd = b["scheme_delta"]
        alt_acc = b["alt"]["acc"]
        # "clears or approaches 90%": clears the bar, or closes >=50% of the QAT->bf16 gap
        gap = BF16[k] - b["qat"]["acc"]
        closed = (alt_acc - b["qat"]["acc"]) / gap if gap > 0 else 0.0
        if sd["delta_ci_excludes_0_positive"] and (alt_acc >= BAR90[k] or closed >= 0.5):
            recovered.append(k)

    if recovered:
        v = "LOSS_IS_RECIPE_SPECIFIC"
    elif any_worse and not any_better:
        v = "QAT_IS_NEAR_FRONTIER"
    else:
        v = "LOSS_IS_INTRINSIC_TO_INT4"
    return {
        "verdict": v, "any_bar_alt_better": any_better, "any_bar_alt_worse": any_worse,
        "binding_bars": sorted(binding), "recovered_bars": recovered,
    }


def build_result(calib_desc: str, scheme: str) -> dict[str, Any]:
    bars: dict[str, dict] = {}
    for kind in ("gpqa", "aime"):
        alt = load_rows(RES / f"alt_ar_{kind}.jsonl")
        qat = load_rows(QAT_RES / f"ar_{kind}.jsonl")
        alt_cell = cell_stats(alt, kind)
        qat_cell = cell_stats(qat, kind)
        sd = scheme_delta(alt, qat)
        pct = alt_cell["acc"] / BF16[kind] if BF16[kind] else float("nan")
        pct_lo = alt_cell["acc_wilson95"][0] / BF16[kind]
        pct_hi = alt_cell["acc_wilson95"][1] / BF16[kind]
        bars[kind] = {
            "bf16_ref": BF16[kind], "bf16_run": BF16_RUN[kind], "bar90": BAR90[kind],
            "alt": alt_cell, "qat": qat_cell, "scheme_delta": sd,
            "pct_of_bf16_alt": pct, "pct_of_bf16_alt_ci95": [pct_lo, pct_hi],
            "pct_of_bf16_qat": qat_cell["acc"] / BF16[kind],
        }
    vd = verdict(bars)
    return {
        "scheme": scheme, "calibration": calib_desc, "bars": bars, **vd,
        "analysis_only": True, "official_tps": 0,
    }


def _fmt(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and x == x else str(x)


def print_report(res: dict[str, Any]) -> None:
    print(f"\n===== PR #647 ALT-int4 ({res['scheme']}) scheme reasoning probe =====")
    print(f"calibration: {res['calibration']}")
    for kind in ("gpqa", "aime"):
        b = res["bars"][kind]
        a, q, sd = b["alt"], b["qat"], b["scheme_delta"]
        print(f"\n--- {kind.upper()} (n_alt={a['n']}, n_qat={q['n']}, expect {N_EXPECT[kind]}) ---")
        print(f"  bf16  (cite {b['bf16_run']}): {_fmt(b['bf16_ref'])}   90%-bar {_fmt(b['bar90'])}")
        print(f"  QAT   : acc {_fmt(q['acc'])} CI{[_fmt(x) for x in q['acc_wilson95']]} "
              f"= {_fmt(b['pct_of_bf16_qat']*100,1)}% bf16  trunc {_fmt(q['truncation_rate']*100,1)}% "
              f"xfail {q['extract_fail']}")
        print(f"  ALT   : acc {_fmt(a['acc'])} CI{[_fmt(x) for x in a['acc_wilson95']]} "
              f"= {_fmt(b['pct_of_bf16_alt']*100,1)}% bf16  trunc {_fmt(a['truncation_rate']*100,1)}% "
              f"xfail {a['extract_fail']}")
        print(f"  scheme-delta (ALT−QAT): {_fmt(sd['delta_alt_minus_qat'])} "
              f"CI95 {[_fmt(x) for x in sd['delta_ci95']]}  "
              f"mcnemar b(alt+)={sd['mcnemar']['b']} c(qat+)={sd['mcnemar']['c']} "
              f"p={_fmt(sd['mcnemar']['p_exact'],3)}")
        print(f"  pairing: n_paired={sd['n_paired']} prompt_mismatch={sd['n_prompt_mismatch']} "
              f"(alt_only={sd['n_alt_only']} qat_only={sd['n_qat_only']})")
    print(f"\n==> VERDICT: {res['verdict']}  "
          f"(binding={res['binding_bars']} recovered={res['recovered_bars']} "
          f"better={res['any_bar_alt_better']} worse={res['any_bar_alt_worse']})\n")


def log_wandb(res: dict[str, Any], name: str, group: str) -> str | None:
    from scripts import wandb_logging as wl
    cfg = {
        "pr": 647, "scheme": res["scheme"], "calibration": res["calibration"],
        "arms": "ar", "drafter": "off", "max_num_seqs": 1, "batch_invariant": 1,
        "bf16_ref": BF16, "bar90": BAR90, "analysis_only": True, "official_tps": 0,
    }
    run = wl.init_wandb_run(
        job_type="altint4-scheme-reasoning-probe", agent="denken",
        name=name, group=group,
        notes="PR647 ZOOM-OUT: is the int4 reasoning loss recipe-specific or intrinsic? "
              "Alt int4-g128 GPTQ vs QAT-int4 on GPQA-D + AIME, M=1 AR greedy.",
        tags=["pr647", "altint4", "gptq", "scheme-probe", "gpqa", "aime", "analysis-only"],
        config=cfg,
    )
    if run is None:
        print("[analyze] wandb not configured — skipping (summary.json still written)", flush=True)
        return None
    metrics: dict[str, Any] = {}
    for kind in ("gpqa", "aime"):
        b = res["bars"][kind]
        metrics.update(wl.flatten_numeric(f"{kind}/alt", b["alt"]))
        metrics.update(wl.flatten_numeric(f"{kind}/qat", b["qat"]))
        metrics.update(wl.flatten_numeric(f"{kind}/scheme_delta", b["scheme_delta"]))
        metrics[f"{kind}/pct_of_bf16_alt"] = b["pct_of_bf16_alt"]
        metrics[f"{kind}/pct_of_bf16_qat"] = b["pct_of_bf16_qat"]
        metrics[f"{kind}/bf16_ref"] = b["bf16_ref"]
    metrics["analysis_only"] = 1
    metrics["official_tps"] = 0
    wl.log_event(run, "altint4_probe_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["verdict"] = res["verdict"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr647_altint4_scheme_probe",
                         artifact_type="scheme-reasoning-probe", data=res)
    rid = run.id
    wl.finish_wandb(run)
    print(f"[analyze] wandb run id={rid}", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="gptq")
    ap.add_argument("--calib", default="GSM8K-train CoT + MMLU-Pro-val CoT (reasoning-rich, non-leaking)")
    ap.add_argument("--wandb_name", default="denken/altint4-scheme-reasoning-probe")
    ap.add_argument("--wandb_group", default="altint4-scheme-reasoning-probe-denken")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    res = build_result(args.calib, args.scheme)
    print_report(res)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(res, indent=2))
    print(f"[analyze] wrote {RES/'summary.json'}", flush=True)
    if not args.no_wandb:
        rid = log_wandb(res, args.wandb_name, args.wandb_group)
        if rid:
            res["wandb_run_id"] = rid
            (RES / "summary.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
