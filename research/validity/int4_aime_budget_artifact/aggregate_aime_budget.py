"""Aggregate the AIME budget-artifact 2x2 (body x budget) into the PR #699 deliverable.

Consumes four ``aime_eval.py`` output JSONs (int4 x {gate,high}, base x {gate,high})
and produces the decision-forcing decomposition:

  - per-cell sampled accuracy (mean_pass_rate = expected single-draw accuracy, the
    gate analog) + maj@k, and the finish-reason TRUNCATION rate
    (frac of samples with finish_reason == 'length');
  - the gate metric int4/base RATIO at each budget, and Dratio = ratio@high - ratio@gate;
  - the load-bearing MECHANISM number truncation_rate_int4 - truncation_rate_base at
    the gate budget (is the int4 body cap-disadvantaged?);
  - a bootstrapped CI on ratio@high (problem-level resample) so the >=0.90 call is not
    a point estimate (the #610 multi-seed-CI discipline, carried to the ratio);
  - the verdict: AIME_BUDGET_ARTIFACT / AIME_REAL_PRECISION_LOSS / AIME_BUDGET_PARTIAL.

Logs everything to W&B (group ``int4-aime-budget-artifact-kanna``) as an explicit
scalar set AND a per-item finish-reason artifact, with the four mandated guard
scalars (analysis_only=1, official_tps=0, no_hf_job=1, fires=0).

Pure-CPU; safe to re-run (the GPU work is the four aime_eval.py runs upstream).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

GATE_BAR = 0.90  # >=90% of vanilla bf16 base on the #31 sampled basis (#515 gate)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _cell_metrics(d: dict[str, Any]) -> dict[str, Any]:
    """Reduce one aime_eval.py output to the metrics this card needs."""
    per = d.get("per_problem", [])
    # Sampled single-draw accuracy = mean over problems of (correct_samples / k).
    # This is the gate analog (the #515 sampled gate averages per-seed accuracy);
    # maj@k is the higher majority-vote metric, carried alongside.
    pass_rates = [p["pass_rate"] for p in per]
    n = len(per)
    fin: list[str] = [fr for p in per for fr in p.get("finish_reasons", [])]
    n_len = sum(1 for fr in fin if fr == "length")
    n_fin = len(fin)
    return {
        "label": d.get("label"),
        "n_problems": n,
        "k": d.get("maj_k") or (d.get("meta", {}) or {}).get("k"),
        "total_samples": n_fin,
        "acc_sampled": (sum(pass_rates) / n) if n else 0.0,  # mean_pass_rate
        "acc_majk": d.get("maj_k_accuracy", 0.0),
        "truncation_rate": (n_len / n_fin) if n_fin else 0.0,
        "n_truncated_samples": n_len,
        "extract_fail_rate": d.get("extract_fail_rate", 0.0),
        "max_tokens": (d.get("sampling", {}) or {}).get("max_tokens"),
        "enable_thinking": (d.get("sampling", {}) or {}).get("enable_thinking"),
        "temperature": (d.get("sampling", {}) or {}).get("temperature"),
        "_pass_rates": pass_rates,  # for bootstrap (stripped before logging)
        "_maj_correct": [1.0 if p.get("maj_correct") else 0.0 for p in per],
    }


def _bootstrap_ratio_ci(
    num_pass: list[float], den_pass: list[float], iters: int, seed: int
) -> tuple[float, float, float]:
    """Problem-level paired bootstrap of the int4/base ratio of mean_pass_rate.

    Assumes the two cells share the same problem ordering (same dataset/seed), so we
    resample problem INDICES jointly. Returns (mean, lo95, hi95).
    """
    if not num_pass or not den_pass:
        return 0.0, 0.0, 0.0
    n = min(len(num_pass), len(den_pass))
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        num = sum(num_pass[i] for i in idx) / n
        den = sum(den_pass[i] for i in idx) / n
        if den > 0:
            ratios.append(num / den)
    if not ratios:
        return 0.0, 0.0, 0.0
    ratios.sort()
    mean = sum(ratios) / len(ratios)
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[min(len(ratios) - 1, int(0.975 * len(ratios)))]
    return mean, lo, hi


def decide(
    ratio_gate: float,
    ratio_high: float,
    d_ratio: float,
    trunc_delta_gate: float,
    ratio_high_ci_lo: float,
    *,
    flat_eps: float = 0.02,
) -> tuple[str, str]:
    """Assign the PR #699 verdict + a one-line rationale.

    flat_eps: |Dratio| below this is "budget-flat" (ratio does not move with budget).
    The >=0.90 call uses the bootstrap CI LOWER bound on ratio@high (comfortable-clear
    discipline), not just the point estimate.
    """
    crosses = ratio_high_ci_lo >= GATE_BAR
    rises = d_ratio > flat_eps
    if crosses and rises:
        return (
            "AIME_BUDGET_ARTIFACT",
            f"int4/base RISES with budget (Dratio=+{d_ratio:.3f}) and clears 0.90 "
            f"(ratio@12288 CI-lo {ratio_high_ci_lo:.3f}); trunc_delta@gate "
            f"{trunc_delta_gate:+.3f}. Hard leg is (partly) a truncation artifact.",
        )
    if abs(d_ratio) <= flat_eps and not crosses:
        return (
            "AIME_REAL_PRECISION_LOSS",
            f"int4/base is budget-FLAT (Dratio={d_ratio:+.3f}, |.|<= {flat_eps}) and "
            f"stays below 0.90 (ratio@12288 {ratio_high:.3f}); trunc_delta@gate "
            f"{trunc_delta_gate:+.3f}. Budget lever dead -> ubel #695 recipe is load-bearing.",
        )
    moved = "rises with budget" if rises else (
        "is budget-flat" if abs(d_ratio) <= flat_eps else "falls with budget")
    return (
        "AIME_BUDGET_PARTIAL",
        f"int4/base {moved} (Dratio={d_ratio:+.3f}) but does NOT comfortably clear 0.90 "
        f"(ratio@12288 {ratio_high:.3f}, CI-lo {ratio_high_ci_lo:.3f}); trunc_delta@gate "
        f"{trunc_delta_gate:+.3f}. Budget explains PART; report residual for ubel #695.",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--int4-gate", type=Path, required=True)
    ap.add_argument("--int4-high", type=Path, required=True)
    ap.add_argument("--base-gate", type=Path, required=True)
    ap.add_argument("--base-high", type=Path, required=True)
    ap.add_argument("--gate-budget", type=int, default=6144)
    ap.add_argument("--high-budget", type=int, default=12288)
    ap.add_argument(
        "--metric",
        choices=("sampled", "majk"),
        default="sampled",
        help="gate metric for the ratio/verdict: 'sampled' (mean_pass_rate, single-draw "
        "gate analog) or 'majk' (maj@k). Pick whichever reconciles against lawine's "
        "0.3467 #31 anchor; BOTH ratios are always logged.",
    )
    ap.add_argument("--base-denominator", type=float, default=0.4667,
                    help="banked greedy bf16 base @gate for context (the 0.420 bar = 0.90x this)")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=1234)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="int4-aime-budget-artifact-kanna")
    ap.add_argument("--wandb-name", default="kanna/int4-aime-budget-artifact")
    ap.add_argument("--peak-gpu-gb", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cells = {
        ("int4", args.gate_budget): _cell_metrics(_load(args.int4_gate)),
        ("int4", args.high_budget): _cell_metrics(_load(args.int4_high)),
        ("base", args.gate_budget): _cell_metrics(_load(args.base_gate)),
        ("base", args.high_budget): _cell_metrics(_load(args.base_high)),
    }

    i_g = cells[("int4", args.gate_budget)]
    i_h = cells[("int4", args.high_budget)]
    b_g = cells[("base", args.gate_budget)]
    b_h = cells[("base", args.high_budget)]

    acc_key = "acc_sampled" if args.metric == "sampled" else "acc_majk"
    per_key = "_pass_rates" if args.metric == "sampled" else "_maj_correct"

    def _ratio(num: dict[str, Any], den: dict[str, Any], key: str) -> float:
        return num[key] / den[key] if den[key] else 0.0

    # Chosen-metric ratios drive the verdict; both metrics' ratios are logged so the
    # reconciliation against lawine's 0.3467 #31 anchor is transparent.
    ratio_gate = _ratio(i_g, b_g, acc_key)
    ratio_high = _ratio(i_h, b_h, acc_key)
    ratio_gate_sampled = _ratio(i_g, b_g, "acc_sampled")
    ratio_high_sampled = _ratio(i_h, b_h, "acc_sampled")
    ratio_gate_majk = _ratio(i_g, b_g, "acc_majk")
    ratio_high_majk = _ratio(i_h, b_h, "acc_majk")
    d_ratio = ratio_high - ratio_gate
    trunc_delta_gate = i_g["truncation_rate"] - b_g["truncation_rate"]
    trunc_delta_high = i_h["truncation_rate"] - b_h["truncation_rate"]

    rh_mean, rh_lo, rh_hi = _bootstrap_ratio_ci(
        i_h[per_key], b_h[per_key], args.bootstrap_iters, args.bootstrap_seed
    )
    rg_mean, rg_lo, rg_hi = _bootstrap_ratio_ci(
        i_g[per_key], b_g[per_key], args.bootstrap_iters, args.bootstrap_seed + 1
    )

    verdict, rationale = decide(ratio_gate, ratio_high, d_ratio, trunc_delta_gate, rh_lo)

    # Residual real-precision-loss relayed to ubel #695: the shortfall to 0.90 that
    # budget does NOT close (clamped at 0 if budget over-recovers).
    residual_to_bar = max(0.0, GATE_BAR - ratio_high)

    # Decode-provenance scalars under lawine #693's exact key names so this card's
    # number reconciles cleanly against the 0.3467 #31 anchor (advisor relay, #666).
    _samp = (_load(args.int4_gate).get("sampling", {}) or {})
    _temp = _samp.get("temperature")
    provenance = {
        "eval_decode_basis": "sampled_#31" if (_temp or 0) > 0 else "greedy",
        "eval_sampling": f"T{_temp}_topp{_samp.get('top_p')}_topk{_samp.get('top_k')}",
        "eval_min_tokens": _samp.get("min_tokens"),
        "eval_seed": _samp.get("seed"),
        "eval_enable_thinking": _samp.get("enable_thinking"),
    }

    summary = {
        "verdict": verdict,
        "rationale": rationale,
        # primary + test metrics (PR-mandated)
        "aime_int4_pct_of_base_at_12288": ratio_high,
        "aime_truncation_rate_delta_int4_vs_base": trunc_delta_gate,
        # gate metric used for the verdict + BOTH metrics' ratios (lawine-0.3467 reconcile)
        "gate_metric": args.metric,
        "ratio_int4_base_at_gate": ratio_gate,
        "ratio_int4_base_at_high": ratio_high,
        "delta_ratio_high_minus_gate": d_ratio,
        "ratio_sampled_at_gate": ratio_gate_sampled,
        "ratio_sampled_at_high": ratio_high_sampled,
        "ratio_majk_at_gate": ratio_gate_majk,
        "ratio_majk_at_high": ratio_high_majk,
        "ratio_high_boot_mean": rh_mean,
        "ratio_high_boot_ci_lo": rh_lo,
        "ratio_high_boot_ci_hi": rh_hi,
        "ratio_gate_boot_ci_lo": rg_lo,
        "ratio_gate_boot_ci_hi": rg_hi,
        "crosses_0p90_at_high_ci_lo": int(rh_lo >= GATE_BAR),
        "residual_real_precision_loss_to_0p90": residual_to_bar,
        # truncation mechanism
        "truncation_rate_delta_at_gate": trunc_delta_gate,
        "truncation_rate_delta_at_high": trunc_delta_high,
        # config
        "gate_budget": args.gate_budget,
        "high_budget": args.high_budget,
        "base_denominator_banked_greedy": args.base_denominator,
        "gate_bar_pct_of_base": GATE_BAR,
        # decode provenance (lawine #693 key names; reconcile vs 0.3467 #31 anchor)
        **provenance,
        # guards (PR-mandated, explicit)
        "analysis_only": 1,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": 0,
    }
    if args.peak_gpu_gb is not None:
        summary["peak_gpu_gb"] = args.peak_gpu_gb

    # per-cell scalars (the 2x2 table, flattened)
    for (body, budget), m in cells.items():
        pre = f"cell_{body}_{budget}"
        summary[f"{pre}_acc_sampled"] = m["acc_sampled"]
        summary[f"{pre}_acc_majk"] = m["acc_majk"]
        summary[f"{pre}_truncation_rate"] = m["truncation_rate"]
        summary[f"{pre}_n_truncated"] = m["n_truncated_samples"]
        summary[f"{pre}_total_samples"] = m["total_samples"]
        summary[f"{pre}_k"] = m["k"]
        summary[f"{pre}_n_problems"] = m["n_problems"]
        summary[f"{pre}_enable_thinking"] = m["enable_thinking"]

    print(json.dumps(summary, indent=2))
    print(f"\n=== VERDICT: {verdict} ===\n{rationale}")

    # 2x2 human table
    print("\n2x2 (body x budget) sampled accuracy / truncation_rate:")
    print(f"{'':>6} | {'budget=' + str(args.gate_budget):>22} | {'budget=' + str(args.high_budget):>22}")
    for body in ("int4", "base"):
        g = cells[(body, args.gate_budget)]
        h = cells[(body, args.high_budget)]
        print(
            f"{body:>6} | acc={g['acc_sampled']:.4f} trunc={g['truncation_rate']:.3f} | "
            f"acc={h['acc_sampled']:.4f} trunc={h['truncation_rate']:.3f}"
        )
    print(f"ratio  | {ratio_gate:>22.4f} | {ratio_high:>22.4f}")

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\n[agg] wrote {args.out}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project="gemma-challenge-senpai",
            entity="wandb-applied-ai-team",
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "gate_budget": args.gate_budget,
                "high_budget": args.high_budget,
                # honest decode-basis from the loaded cells (greedy for this pivot;
                # the int4 SAMPLED basis is unmeasurable on 0.22.0 -- see jqecrucm).
                "decode_basis": f"{provenance['eval_decode_basis']}_{provenance['eval_sampling']}_min{provenance['eval_min_tokens']}",
                "dataset": "aime_2024+2025-I+2025-II_n60",
                "engine": "vllm-0.22.0_BI1",
            },
        )
        run.summary.update(summary)
        # per-item finish-reason artifact (the mandated per-(body,budget) detail)
        rows = []
        for (body, budget), m in cells.items():
            d = _load(
                {
                    ("int4", args.gate_budget): args.int4_gate,
                    ("int4", args.high_budget): args.int4_high,
                    ("base", args.gate_budget): args.base_gate,
                    ("base", args.high_budget): args.base_high,
                }[(body, budget)]
            )
            for p in d.get("per_problem", []):
                fr = p.get("finish_reasons", [])
                rows.append(
                    {
                        "body": body,
                        "budget": budget,
                        "id": p.get("id"),
                        "year": p.get("year"),
                        "gold": p.get("gold"),
                        "pass_rate": p.get("pass_rate"),
                        "maj_correct": p.get("maj_correct"),
                        "n_truncated": sum(1 for x in fr if x == "length"),
                        "n_samples": len(fr),
                        "finish_reasons": ",".join(str(x) for x in fr),
                        "sample_chars_max": max(p.get("sample_chars", [0]) or [0]),
                    }
                )
        table = wandb.Table(columns=list(rows[0].keys()) if rows else [])
        for r in rows:
            table.add_data(*[r[c] for c in table.columns])
        art = wandb.Artifact("aime_budget_2x2_finish_reasons", type="analysis")
        with art.new_file("per_item_finish_reasons.json", mode="w") as f:
            json.dump(rows, f, indent=2)
        run.log({"per_item_finish_reasons": table})
        run.log_artifact(art)
        print(f"[agg] W&B run: {run.id} (group {args.wandb_group})")
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
