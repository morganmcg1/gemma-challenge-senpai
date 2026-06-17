#!/usr/bin/env python
"""PR #535 matched-concurrency, higher-n AIME verdict (the binding quality arm).

The first probe arm left the quality verdict *inconclusive* because the n=30 AIME
maj@1 metric is concurrency-confounded: the unchanged stock base swung 0.100
(conc=1) -> 0.267 (conc=32). The advisor's fix (Morgan #524 gate-bound): measure
**both** arms at the **same** concurrency (conc=32, how a real ship serves) on an
**n~=90** item set, and decide with **Wilson** intervals whether base_fullhead
clears **0.90 x base** at matched conc.

Both arms serve the *same* underlying weights (stock ``gemma-4-E4B-it-qat-w4a16-ct``,
native 262k head); the ONLY delta is the fast-kernel serving stack (surgical 2D
attn + MTP K=7 + split-KV + onegraph + PLE fold) vs the plain ``--dtype auto``
base. So the comparison isolates exactly one question: *do the fast kernels cost
reasoning on long-chain greedy AIME, once the concurrency confound is removed?*

Inputs: two ``aime_eval.py`` outputs (greedy maj@1, k=1, same years, same conc).
Outputs: Wilson 95% CIs for each arm, the Newcombe 95% CI for the difference, the
ratio, a per-year breakdown, and a three-way verdict (GO / NO-GO / INCONCLUSIVE)
against the 0.90 x base bar. Optionally logs a W&B run in the probe group.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

Z95 = 1.959963984540054  # two-sided 95%


def wilson_ci(x: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Wilson score interval. Returns (p_hat, lower, upper)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = x / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def newcombe_diff_ci(x1: int, n1: int, x2: int, n2: int, z: float = Z95) -> tuple[float, float, float]:
    """Newcombe method-10 CI for p1 - p2 (two independent proportions)."""
    p1, l1, u1 = wilson_ci(x1, n1, z)
    p2, l2, u2 = wilson_ci(x2, n2, z)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lower, upper


def _correct_count(result: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    pp = result.get("per_problem", [])
    n = len(pp)
    x = sum(1 for p in pp if p.get("maj_correct"))
    return x, n, pp


def empty_eos_stats(result: dict[str, Any], short_char_thresh: int = 8) -> dict[str, Any]:
    """Per-sample serving-artifact stats (wirbel #541: first-token-EOS empties).

    An immediate first-token-EOS empty is a completion with no content that
    stopped on EOS (``sample_chars==0`` and ``finish_reason=='stop'``). We also
    count very-short completions (``0 < chars < short_char_thresh``) — those are
    the ones a request-level ``min_tokens`` floor could lengthen. On AIME these
    are distinct from extract failures (malformed-but-present boxed answers).
    """
    samples = 0
    empty = 0           # chars==0 (any finish)
    empty_eos = 0       # chars==0 AND finish=='stop' (the wirbel signature)
    short = 0           # 0 < chars < thresh
    min_chars = None
    for p in result.get("per_problem", []):
        chars = p.get("sample_chars") or []
        frs = p.get("finish_reasons") or []
        for i, c in enumerate(chars):
            samples += 1
            fr = frs[i] if i < len(frs) else None
            min_chars = c if min_chars is None else min(min_chars, c)
            if c == 0:
                empty += 1
                if fr == "stop":
                    empty_eos += 1
            elif c < short_char_thresh:
                short += 1
    return {
        "samples": samples,
        "empty_count": empty,
        "empty_rate": (empty / samples) if samples else 0.0,
        "immediate_eos_empty_count": empty_eos,
        "immediate_eos_empty_rate": (empty_eos / samples) if samples else 0.0,
        "short_lt_thresh_count": short,
        "short_char_thresh": short_char_thresh,
        "min_sample_chars": min_chars,
    }


def _per_year(pp: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    by_year: dict[str, list[int]] = defaultdict(list)
    for p in pp.values():
        by_year[str(p.get("year"))].append(int(bool(p.get("maj_correct"))))
    return {y: (sum(v), len(v)) for y, v in by_year.items()}


def build_verdict(base: dict[str, Any], fh: dict[str, Any], quality_frac: float = 0.90) -> dict[str, Any]:
    base_by_id = {p["id"]: p for p in base.get("per_problem", [])}
    fh_by_id = {p["id"]: p for p in fh.get("per_problem", [])}
    common = [pid for pid in base_by_id if pid in fh_by_id]

    parity: list[str] = []
    if list(base.get("years") or []) != list(fh.get("years") or []):
        parity.append(f"years differ base={base.get('years')} fh={fh.get('years')}")
    if base.get("max_num_seqs") != fh.get("max_num_seqs"):
        parity.append(f"concurrency differs base={base.get('max_num_seqs')} fh={fh.get('max_num_seqs')}")
    bsamp, fsamp = base.get("sampling") or {}, fh.get("sampling") or {}
    for kk in ("temperature", "top_p", "top_k", "max_tokens", "seed", "enable_thinking"):
        if bsamp.get(kk) != fsamp.get(kk):
            parity.append(f"sampling.{kk} base={bsamp.get(kk)} fh={fsamp.get(kk)}")
    if set(base_by_id) != set(fh_by_id):
        parity.append("problem-id set mismatch")

    # Accuracy on the common item set (so both arms score the identical problems).
    bx = sum(1 for pid in common if base_by_id[pid].get("maj_correct"))
    fx = sum(1 for pid in common if fh_by_id[pid].get("maj_correct"))
    n = len(common)

    bp, bl, bu = wilson_ci(bx, n)
    fp, fl, fu = wilson_ci(fx, n)
    d, dl, du = newcombe_diff_ci(fx, n, bx, n)  # fh - base

    bar = quality_frac * bp                      # advisor's stated bar: 0.90 x base point
    bar_conservative = quality_frac * bl         # using base's Wilson lower bound

    # Three-way verdict on base_fullhead's Wilson CI vs the (point) bar.
    if fl >= bar:
        verdict = "GO"
        verdict_detail = "base_fullhead Wilson CI lower bound clears 0.90 x base -> quality-safe at matched conc"
    elif fu < bar:
        verdict = "NO-GO"
        verdict_detail = "base_fullhead Wilson CI upper bound is below 0.90 x base -> demonstrably below bar"
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = "base_fullhead Wilson CI straddles 0.90 x base -> n still underpowered to adjudicate"

    # Independent 'do the kernels move AIME at all' read: does the difference CI cover 0?
    diff_covers_zero = dl <= 0.0 <= du
    ratio = (fp / bp) if bp > 0 else None

    by_year_base = _per_year({pid: base_by_id[pid] for pid in common})
    by_year_fh = _per_year({pid: fh_by_id[pid] for pid in common})
    per_year = {
        y: {
            "base": {"x": by_year_base[y][0], "n": by_year_base[y][1],
                     "acc": by_year_base[y][0] / by_year_base[y][1] if by_year_base[y][1] else None},
            "base_fullhead": {"x": by_year_fh.get(y, (0, 0))[0], "n": by_year_fh.get(y, (0, 0))[1],
                              "acc": (by_year_fh.get(y, (0, 0))[0] / by_year_fh.get(y, (0, 0))[1])
                              if by_year_fh.get(y, (0, 0))[1] else None},
        }
        for y in sorted(set(by_year_base) | set(by_year_fh))
    }

    maj_agree = sum(1 for pid in common
                    if base_by_id[pid].get("maj_answer") == fh_by_id[pid].get("maj_answer"))

    return {
        "n_problems": n,
        "concurrency_max_num_seqs": fh.get("max_num_seqs"),
        "years": base.get("years"),
        "sampling": bsamp,
        "apples_to_apples": not parity,
        "parity_issues": parity,
        # base arm
        "base_acc": bp, "base_x": bx, "base_wilson_lo": bl, "base_wilson_hi": bu,
        "base_extract_fail_rate": base.get("extract_fail_rate"),
        # base_fullhead arm
        "base_fullhead_acc": fp, "base_fullhead_x": fx,
        "base_fullhead_wilson_lo": fl, "base_fullhead_wilson_hi": fu,
        "base_fullhead_extract_fail_rate": fh.get("extract_fail_rate"),
        # comparison
        "ratio_fullhead_over_base": ratio,
        "diff_fullhead_minus_base": d,
        "diff_newcombe_lo": dl, "diff_newcombe_hi": du,
        "diff_ci_covers_zero": diff_covers_zero,
        "quality_bar_frac": quality_frac,
        "bar_0p90_x_base_point": bar,
        "bar_0p90_x_base_wilson_lo": bar_conservative,
        "base_fullhead_clears_bar_point": bool(fp >= bar),
        "base_fullhead_wilson_lo_clears_bar": bool(fl >= bar),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "maj_answer_agreement": maj_agree, "maj_answer_agreement_frac": maj_agree / n if n else None,
        "per_year": per_year,
    }


def _arm_acc_on_common(base: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Accuracy + Wilson CI for ``arm`` scored on the items it shares with ``base``."""
    base_ids = {p["id"] for p in base.get("per_problem", [])}
    arm_by_id = {p["id"]: p for p in arm.get("per_problem", [])}
    common = [pid for pid in base_ids if pid in arm_by_id]
    n = len(common)
    x = sum(1 for pid in common if arm_by_id[pid].get("maj_correct"))
    p, lo, hi = wilson_ci(x, n)
    return {"x": x, "n": n, "acc": p, "wilson_lo": lo, "wilson_hi": hi,
            "extract_fail_rate": arm.get("extract_fail_rate")}


def serving_artifact_block(
    base: dict[str, Any],
    fh: dict[str, Any],
    fh_mintok: dict[str, Any] | None,
    fh_rerun: dict[str, Any] | None,
    quality_frac: float = 0.90,
) -> dict[str, Any]:
    """min_tokens recovery + per-arm empty/EOS stats (advisor in-flight steer).

    wirbel #541 found ~10.4% first-token-EOS empties on GSM8K over the 262k head;
    a request-level ``min_tokens=8`` recovered them. This block measures whether
    the same artifact depresses AIME and whether ``min_tokens`` recovers it, and
    brackets the fast stack's run-to-run accuracy noise with a fresh control arm.
    """
    bp = base.get("per_problem", [])
    bx = sum(1 for p in bp if p.get("maj_correct"))
    bn = len(bp)
    base_p, base_lo, _ = wilson_ci(bx, bn)
    bar = quality_frac * base_p

    fh_common = _arm_acc_on_common(base, fh)  # as-served
    blk: dict[str, Any] = {
        "quality_bar_frac": quality_frac,
        "bar_0p90_x_base_point": bar,
        "empty_stats": {
            "base": empty_eos_stats(base),
            "base_fullhead_asserved": empty_eos_stats(fh),
        },
        "base_fullhead_asserved_acc": fh_common["acc"],
        "base_fullhead_asserved_immediate_eos_empty_rate":
            empty_eos_stats(fh)["immediate_eos_empty_rate"],
    }

    if fh_mintok is not None:
        m = _arm_acc_on_common(base, fh_mintok)
        es = empty_eos_stats(fh_mintok)
        blk["empty_stats"]["base_fullhead_mintok8"] = es
        blk["base_fullhead_mintok8_acc"] = m["acc"]
        blk["base_fullhead_mintok8_x"] = m["x"]
        blk["base_fullhead_mintok8_n"] = m["n"]
        blk["base_fullhead_mintok8_wilson_lo"] = m["wilson_lo"]
        blk["base_fullhead_mintok8_wilson_hi"] = m["wilson_hi"]
        blk["base_fullhead_mintok8_immediate_eos_empty_rate"] = es["immediate_eos_empty_rate"]
        blk["mintok8_recovery_vs_asserved"] = m["acc"] - fh_common["acc"]
        blk["mintok8_ratio_over_base"] = (m["acc"] / base_p) if base_p > 0 else None
        # apples-to-apples gate figure for the AIME leg if the guard moved the score
        blk["mintok8_wilson_lo_clears_bar"] = bool(m["wilson_lo"] >= bar)
        blk["mintok8_clears_bar_point"] = bool(m["acc"] >= bar)

    if fh_rerun is not None:
        r = _arm_acc_on_common(base, fh_rerun)
        es = empty_eos_stats(fh_rerun)
        blk["empty_stats"]["base_fullhead_rerun"] = es
        blk["base_fullhead_rerun_acc"] = r["acc"]
        blk["base_fullhead_rerun_immediate_eos_empty_rate"] = es["immediate_eos_empty_rate"]
        # run-to-run accuracy spread across the two as-served draws (chaos bracket)
        blk["asserved_run_to_run_spread"] = abs(r["acc"] - fh_common["acc"])

    # The artifact does NOT depress AIME here iff every measured fast-stack arm has
    # ~0 immediate-EOS empties; then min_tokens is a no-op and any acc wiggle is chaos.
    eos_rates = [s["immediate_eos_empty_rate"] for s in blk["empty_stats"].values()]
    blk["max_immediate_eos_empty_rate"] = max(eos_rates) if eos_rates else 0.0
    blk["artifact_present_on_aime"] = bool(blk["max_immediate_eos_empty_rate"] > 0.0)
    return blk


def _wandb_log(v: dict[str, Any], base: dict[str, Any], fh: dict[str, Any], args: argparse.Namespace) -> str | None:
    # Append (not insert-at-0): the repo root holds a local ``wandb/`` run-dir that,
    # placed ahead of site-packages, shadows the installed ``wandb`` package as an
    # empty namespace (``wandb.init`` AttributeError). Appending keeps ``scripts``
    # importable while letting the real ``wandb`` win.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    try:
        from scripts import wandb_logging
    except Exception as exc:  # pragma: no cover
        print(f"[verdict] wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return None
    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 535,
        "experiment": "base-fullhead-fast-ship-probe",
        "arm": "matched-conc-n90-aime",
        "substrate": "base_int4_native_262k_head",
        "concurrency_max_num_seqs": v.get("concurrency_max_num_seqs"),
        "aime_years": v.get("years"),
        "sampling": v.get("sampling"),
        "base_submission": base.get("submission"),
        "base_fullhead_submission": fh.get("submission"),
        "base_fullhead_serve_overrides": fh.get("serve_overrides"),
        "fast_stack": "surgical_2d_attn + mtp_k7_spec + splitkv + onegraph + ple_fold",
    }
    run = wandb_logging.init_wandb_run(
        job_type="base-fullhead-fast-ship-probe",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        notes=("PR #535 matched-conc n~=90 AIME: both arms at conc=32 on the identical "
               "item set; Wilson-CI verdict on whether the fast kernels cost reasoning "
               "vs plain base. Analysis-only; official_tps=0."),
        tags=["aime", "downstream-quality", "analysis-only", "pr-535", "base-fullhead", "matched-conc"],
        config=config,
    )
    if run is None:
        print("[verdict] wandb disabled/unavailable; JSON only", flush=True)
        return None
    summary = {k: vv for k, vv in v.items() if k not in ("per_year", "serving_artifact_empty_stats")}
    # flatten per-year for first-class summary keys
    for y, cell in (v.get("per_year") or {}).items():
        summary[f"acc_base_{y}"] = cell["base"]["acc"]
        summary[f"acc_base_fullhead_{y}"] = cell["base_fullhead"]["acc"]
    wandb_logging.log_summary(run, summary, step=0)
    wandb_logging.log_json_artifact(run, name="aime_matched_conc_verdict", artifact_type="probe-535", data=v)
    for nm, raw in (("aime_base_conc32_n90", base), ("aime_base_fullhead_conc32_n90", fh)):
        slim = {k: vv for k, vv in raw.items() if k not in ("per_problem",)}
        slim["per_problem"] = [
            {k: pp.get(k) for k in ("id", "year", "gold", "maj_answer", "maj_correct", "k")}
            for pp in raw.get("per_problem", [])
        ]
        wandb_logging.log_json_artifact(run, name=nm, artifact_type="probe-535", data=slim)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="plain-base aime_eval.py JSON (conc=32)")
    ap.add_argument("--fullhead", type=Path, required=True, help="base_fullhead aime_eval.py JSON (conc=32, as-served)")
    ap.add_argument("--fullhead-mintok", type=Path, default=None,
                    help="base_fullhead min_tokens=8 JSON (the serving-artifact treatment arm)")
    ap.add_argument("--fullhead-rerun", type=Path, default=None,
                    help="base_fullhead as-served fresh JSON (run-to-run chaos control)")
    ap.add_argument("--out", type=Path, required=True, help="verdict JSON path")
    ap.add_argument("--quality-frac", type=float, default=0.90)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="fern/base-fullhead-matched-conc-aime")
    ap.add_argument("--wandb-group", default="base-fullhead-fast-ship-probe")
    args = ap.parse_args(argv)

    base = json.loads(args.base.read_text())
    fh = json.loads(args.fullhead.read_text())
    v = build_verdict(base, fh, quality_frac=args.quality_frac)
    v["created_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    fh_mintok = json.loads(args.fullhead_mintok.read_text()) if args.fullhead_mintok else None
    fh_rerun = json.loads(args.fullhead_rerun.read_text()) if args.fullhead_rerun else None
    sa = serving_artifact_block(base, fh, fh_mintok, fh_rerun, quality_frac=args.quality_frac)
    empty_stats = sa.pop("empty_stats")
    for arm_name, st in empty_stats.items():
        v[f"empty_rate_{arm_name}"] = st["empty_rate"]
        v[f"immediate_eos_empty_rate_{arm_name}"] = st["immediate_eos_empty_rate"]
        v[f"min_sample_chars_{arm_name}"] = st["min_sample_chars"]
    for k, val in sa.items():
        v[k] = val
    v["serving_artifact_empty_stats"] = empty_stats

    run_id = _wandb_log(v, base, fh, args) if args.wandb else None
    v["wandb_run_id"] = run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(v, indent=2, default=str))

    print(f"\n[verdict] n={v['n_problems']} conc={v['concurrency_max_num_seqs']} years={v['years']}")
    print(f"[verdict] base          acc={v['base_acc']:.4f} ({v['base_x']}/{v['n_problems']}) "
          f"Wilson95=[{v['base_wilson_lo']:.4f}, {v['base_wilson_hi']:.4f}]")
    print(f"[verdict] base_fullhead acc={v['base_fullhead_acc']:.4f} ({v['base_fullhead_x']}/{v['n_problems']}) "
          f"Wilson95=[{v['base_fullhead_wilson_lo']:.4f}, {v['base_fullhead_wilson_hi']:.4f}]")
    print(f"[verdict] ratio fh/base={v['ratio_fullhead_over_base']:.4f}  "
          f"diff(fh-base)={v['diff_fullhead_minus_base']:+.4f} "
          f"Newcombe95=[{v['diff_newcombe_lo']:+.4f}, {v['diff_newcombe_hi']:+.4f}] "
          f"covers0={v['diff_ci_covers_zero']}")
    print(f"[verdict] bar=0.90*base={v['bar_0p90_x_base_point']:.4f}  "
          f"VERDICT={v['verdict']} -- {v['verdict_detail']}")
    print(f"[artifact] max immediate-EOS-empty rate across fast arms="
          f"{v.get('max_immediate_eos_empty_rate'):.4f}  artifact_present_on_aime={v.get('artifact_present_on_aime')}")
    for arm in ("base", "base_fullhead_asserved", "base_fullhead_mintok8", "base_fullhead_rerun"):
        r = v.get(f"immediate_eos_empty_rate_{arm}")
        if r is not None:
            print(f"    {arm}: immediate-EOS-empty={r:.4f}  min_sample_chars={v.get(f'min_sample_chars_{arm}')}")
    if "base_fullhead_mintok8_acc" in v:
        print(f"[mintok8] base_fullhead+min_tokens=8 acc={v['base_fullhead_mintok8_acc']:.4f} "
              f"({v['base_fullhead_mintok8_x']}/{v['base_fullhead_mintok8_n']}) "
              f"Wilson95=[{v['base_fullhead_mintok8_wilson_lo']:.4f}, {v['base_fullhead_mintok8_wilson_hi']:.4f}]  "
              f"recovery_vs_asserved={v['mintok8_recovery_vs_asserved']:+.4f}  "
              f"ratio/base={v['mintok8_ratio_over_base']:.4f}  clears_bar_lo={v['mintok8_wilson_lo_clears_bar']}")
    if "asserved_run_to_run_spread" in v:
        print(f"[chaos]   as-served run-to-run |Δacc|={v['asserved_run_to_run_spread']:.4f} "
              f"(rerun acc={v.get('base_fullhead_rerun_acc'):.4f})")
    print("[verdict] per-year:")
    for y, cell in (v.get("per_year") or {}).items():
        print(f"    {y}: base {cell['base']['x']}/{cell['base']['n']}={cell['base']['acc']:.3f}  "
              f"base_fullhead {cell['base_fullhead']['x']}/{cell['base_fullhead']['n']}={cell['base_fullhead']['acc']:.3f}")
    print("SENPAI-RESULT " + json.dumps({
        "analysis_only": True, "official_tps": 0,
        "n_problems": v["n_problems"], "concurrency": v["concurrency_max_num_seqs"],
        "base_acc": round(v["base_acc"], 4), "base_fullhead_acc": round(v["base_fullhead_acc"], 4),
        "ratio_fullhead_over_base": round(v["ratio_fullhead_over_base"], 4) if v["ratio_fullhead_over_base"] else None,
        "diff_newcombe_95": [round(v["diff_newcombe_lo"], 4), round(v["diff_newcombe_hi"], 4)],
        "verdict": v["verdict"], "wandb_run_id": run_id,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
