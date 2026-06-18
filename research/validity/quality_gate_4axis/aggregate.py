#!/usr/bin/env python
"""PR #661 -- assemble the authoritative #515 FOUR-AXIS quality gate table on the
shipped int4_g128_lmhead body at the gb6144 M=1-AR (seqs=1, BI=1) panel.

Two axes are measured fresh by this PR (GSM8K + MMLU-Pro, paired int4-AR vs
bf16-base at seqs=1); two are pulled from banked cells the advisor supplied
(AIME, GPQA-D). For each axis: int4-AR acc + Wilson 95% CI, bf16-base acc,
pct_of_base = int4/base, and PASS/FAIL vs the >=90%-of-base bar. Also a paired
bootstrap CI for the int4-base delta on the items both arms share.

Reads the result JSONs in results/ (written by run_panel.sh) and emits:
  * results/gate_table.json   -- machine-readable 4-axis table + verdict
  * stdout markdown table
Optionally logs to W&B (group quality-gate-4axis-denken) with --wandb.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

Z = 1.959963984540054  # 95% two-sided


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    """Wilson score interval. Returns (phat, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def load_mmlu(path: Path) -> dict:
    d = json.loads(path.read_text())
    per = {str(s["id"]): bool(s["correct"]) for s in d["per_sample"]}
    sha = {str(s["id"]): s.get("prompt_sha") for s in d["per_sample"]}
    return {"acc": d["accuracy"], "n": d["n_scored"], "k": d["n_correct"],
            "per": per, "sha": sha, "ctok_mean": d.get("completion_tokens_mean"),
            "seed": d.get("seed"), "max_tokens": d.get("max_tokens"),
            "min_tokens": d.get("min_tokens")}


def load_gsm8k(path: Path) -> dict:
    d = json.loads(path.read_text())
    per = {str(p["id"]): bool(p["correct"]) for p in d["per_problem"]}
    return {"acc": d["accuracy"], "n": d["n_problems"], "k": d["n_correct"],
            "per": per, "seed": d.get("seed"),
            "sampling": d.get("sampling", {})}


def paired_bootstrap_delta(int4_per: dict, base_per: dict, key_map_int4=None,
                           key_map_base=None, iters: int = 20000, seed: int = 0):
    """Bootstrap CI for (int4 - base) accuracy on the items BOTH arms scored.

    Pairing key: for MMLU pass prompt_sha maps so items align by content; for
    GSM8K the ids already match across arms (same --seed). Returns
    (delta_point, lo, hi, n_common)."""
    # Build common-item correctness pairs.
    if key_map_int4 and key_map_base:
        # align by shared sha: id->sha each side, then sha->correct
        sha_to_i = {key_map_int4[i]: c for i, c in int4_per.items() if key_map_int4.get(i)}
        sha_to_b = {key_map_base[i]: c for i, c in base_per.items() if key_map_base.get(i)}
        common = sorted(set(sha_to_i) & set(sha_to_b))
        pairs = [(sha_to_i[s], sha_to_b[s]) for s in common]
    else:
        common = sorted(set(int4_per) & set(base_per))
        pairs = [(int4_per[i], base_per[i]) for i in common]
    n = len(pairs)
    if n == 0:
        return None, None, None, 0
    di = [1.0 if a else 0.0 for a, _ in pairs]
    db = [1.0 if b else 0.0 for _, b in pairs]
    point = (sum(di) - sum(db)) / n
    rng = random.Random(seed)
    deltas = []
    idx = range(n)
    for _ in range(iters):
        s_i = 0.0; s_b = 0.0
        for _ in idx:
            j = rng.randrange(n)
            s_i += di[j]; s_b += db[j]
        deltas.append((s_i - s_b) / n)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters)]
    return point, lo, hi, n


def ratio_ci(k_i, n_i, k_b, n_b):
    """pct_of_base point + a conservative CI via Wilson endpoints
    (lo_int/hi_base, hi_int/lo_base)."""
    pi, ilo, ihi = wilson(k_i, n_i)
    pb, blo, bhi = wilson(k_b, n_b)
    point = pi / pb if pb else float("nan")
    lo = ilo / bhi if bhi else float("nan")
    hi = ihi / blo if blo else float("nan")
    return point, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="research/validity/quality_gate_4axis/results")
    ap.add_argument("--int4-mmlu", default=None)
    ap.add_argument("--base-mmlu", default=None)
    ap.add_argument("--int4-gsm8k", default=None)
    ap.add_argument("--base-gsm8k", default=None)
    # Banked cells supplied by the advisor (point estimates + n for Wilson CI).
    ap.add_argument("--aime-int4", type=float, default=0.4000)
    ap.add_argument("--aime-base", type=float, default=0.4667)
    ap.add_argument("--aime-n", type=int, default=60)
    ap.add_argument("--gpqa-int4", type=float, default=0.4798)
    ap.add_argument("--gpqa-base", type=float, default=0.4899)
    ap.add_argument("--gpqa-n", type=int, default=198)
    ap.add_argument("--bar", type=float, default=0.90)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="quality-gate-4axis-denken")
    ap.add_argument("--out", default="research/validity/quality_gate_4axis/results/gate_table.json")
    args = ap.parse_args()

    R = Path(args.results)
    int4_mmlu = load_mmlu(Path(args.int4_mmlu or R / "int4ar_mmlu_pro_greedy.json"))
    base_mmlu = load_mmlu(Path(args.base_mmlu or R / "bf16_mmlu_pro_greedy.json"))
    int4_gsm = load_gsm8k(Path(args.int4_gsm8k or R / "int4ar_gsm8k_greedy.json"))
    base_gsm = load_gsm8k(Path(args.base_gsm8k or R / "bf16_gsm8k_greedy.json"))

    rows = []

    def add_measured(name, ci_arm, base, test_kind):
        # PAIRED gate read: base is the (possibly watchdog-capped) subset; int4 is
        # the full-panel superset. Restrict BOTH arms to the items they SHARE so the
        # int4-vs-base ratio is apples-to-apples on identical items -- exactly how the
        # banked AIME/GPQA-D cells are full-n paired. Pairing key is the item id (both
        # arms ran the same seeded ids / --ids-file); prompt_sha gives a byte-identical
        # integrity check on top.
        common_ids = sorted(set(base["per"].keys()) & set(ci_arm["per"].keys()))
        n_p = len(common_ids)
        k_i_p = sum(1 for i in common_ids if ci_arm["per"][i])   # int4 on shared items
        k_b_p = sum(1 for i in common_ids if base["per"][i])     # base on shared items
        sha_mismatch = 0
        if test_kind == "mmlu":
            sha_mismatch = sum(
                1 for i in common_ids
                if ci_arm["sha"].get(i) and base["sha"].get(i)
                and ci_arm["sha"][i] != base["sha"][i])
        pi, ilo, ihi = wilson(k_i_p, n_p)                        # paired int4 CI
        pb, blo, bhi = wilson(k_b_p, n_p)                        # paired base CI
        rpoint, rlo, rhi = ratio_ci(k_i_p, n_p, k_b_p, n_p)      # paired pct_of_base
        if test_kind == "mmlu":
            dpt, dlo, dhi, ncommon = paired_bootstrap_delta(
                ci_arm["per"], base["per"], ci_arm["sha"], base["sha"])
        else:
            dpt, dlo, dhi, ncommon = paired_bootstrap_delta(ci_arm["per"], base["per"])
        rows.append({
            "axis": name, "int4_acc": pi, "int4_ci": [ilo, ihi],
            "int4_n": n_p, "int4_k": k_i_p,
            # full-panel int4 (better standalone estimate; gate ratio uses paired above)
            "int4_acc_full": ci_arm["acc"], "int4_n_full": ci_arm["n"], "int4_k_full": ci_arm["k"],
            "base_acc": pb, "base_n": n_p, "base_k": k_b_p,
            "pct_of_base": rpoint, "pct_of_base_ci": [rlo, rhi],
            # marginal ratio (full-panel int4 / capped base) for transparency only
            "pct_of_base_marginal": (ci_arm["acc"] / pb) if pb else float("nan"),
            "paired_delta": dpt, "paired_delta_ci": [dlo, dhi], "n_common": ncommon,
            "sha_mismatch": sha_mismatch,
            "pass": rpoint >= args.bar,
            "pass_lcb": rlo >= args.bar,
            "paired": True,
        })

    add_measured("GSM8K", int4_gsm, base_gsm, "gsm8k")
    add_measured("MMLU-Pro", int4_mmlu, base_mmlu, "mmlu")

    # Banked axes (point + Wilson from supplied n).
    def add_banked(name, acc_i, acc_b, n):
        k_i = round(acc_i * n); k_b = round(acc_b * n)
        pi, ilo, ihi = wilson(k_i, n)
        rpoint, rlo, rhi = ratio_ci(k_i, n, k_b, n)
        rows.append({
            "axis": name, "int4_acc": acc_i, "int4_ci": [ilo, ihi],
            "int4_n": n, "int4_k": k_i,
            "int4_acc_full": acc_i, "int4_n_full": n, "int4_k_full": k_i,
            "base_acc": acc_b, "base_n": n, "base_k": k_b,
            "pct_of_base": rpoint, "pct_of_base_ci": [rlo, rhi],
            "pct_of_base_marginal": rpoint,
            "paired_delta": acc_i - acc_b, "paired_delta_ci": [None, None],
            "n_common": n, "sha_mismatch": 0,
            "pass": rpoint >= args.bar, "pass_lcb": rlo >= args.bar,
            "banked": True, "paired": True,
        })

    add_banked("AIME", args.aime_int4, args.aime_base, args.aime_n)
    add_banked("GPQA-D", args.gpqa_int4, args.gpqa_base, args.gpqa_n)

    worst = min(rows, key=lambda r: r["pct_of_base"])
    fails = [r for r in rows if not r["pass"]]
    measured_fail = [r for r in rows if not r.get("banked") and not r["pass"]]
    if len(fails) == 1 and fails[0]["axis"] == "AIME":
        verdict = "AIME_IS_SOLE_BLOCKER"
    elif measured_fail:
        verdict = "QUALITY_FAIL_IS_BROADER"
    elif len(fails) == 0:
        verdict = "ALL_AXES_CLEAR"  # (not one of the two named outcomes; report explicitly)
    else:
        verdict = "QUALITY_FAIL_IS_BROADER"

    out = {
        "panel": {"engine": "vllm==0.22.0", "max_num_seqs": 1, "batch_invariant": 1,
                  "max_model_len": 8192, "max_tokens": 6144, "min_tokens": 8,
                  "decode": "greedy", "drafter": "off", "M": 1},
        "bar": args.bar, "rows": rows, "worst_axis": worst["axis"],
        "primary_metric_worst_pct_of_base": worst["pct_of_base"],
        "test_metric_gsm8k_int4_acc": rows[0]["int4_acc_full"], "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    # Markdown table to stdout.
    print(f"\n#515 FOUR-AXIS QUALITY GATE -- int4_g128_lmhead @ gb6144 seqs=1 AR BI=1")
    print(f"bar = >=90% of bf16 base; verdict = {verdict}\n")
    hdr = ("| axis | int4-AR acc (95% CI) | base acc | pct_of_base (CI) | "
           "paired delta (CI) | PASS? |")
    print(hdr); print("|" + "---|" * 6)
    for r in rows:
        ci = f"[{r['int4_ci'][0]:.3f},{r['int4_ci'][1]:.3f}]"
        pc = f"{r['pct_of_base']*100:.1f}% [{r['pct_of_base_ci'][0]*100:.1f},{r['pct_of_base_ci'][1]*100:.1f}]"
        if r["paired_delta_ci"][0] is None:
            dd = f"{r['paired_delta']:+.4f} (n={r['n_common']})"
        else:
            dd = f"{r['paired_delta']:+.4f} [{r['paired_delta_ci'][0]:+.3f},{r['paired_delta_ci'][1]:+.3f}]"
        tag = " (banked)" if r.get("banked") else ""
        i4 = f"{r['int4_acc']:.4f} {ci} (n={r['int4_n']})"
        if r.get("paired") and not r.get("banked") and r["int4_n_full"] != r["int4_n"]:
            i4 += f" [full-panel {r['int4_acc_full']:.4f} n={r['int4_n_full']}]"
        print(f"| {r['axis']}{tag} | {i4} | "
              f"{r['base_acc']:.4f} (n={r['base_n']}) | {pc} | {dd} | "
              f"{'PASS' if r['pass'] else 'FAIL'}{'' if r['pass_lcb'] else ' (LCB<bar)'} |")
    # Integrity + pairing notes for the measured axes.
    for r in rows:
        if r.get("banked"):
            continue
        note = (f"  [{r['axis']}] paired on n={r['int4_n']} shared items; "
                f"int4 full-panel {r['int4_acc_full']:.4f} (n={r['int4_n_full']}); "
                f"marginal pct {r['pct_of_base_marginal']*100:.1f}%")
        if "sha_mismatch" in r:
            note += f"; prompt_sha mismatches={r['sha_mismatch']}"
        print(note)
    print(f"\nworst axis = {worst['axis']} @ {worst['pct_of_base']*100:.1f}% of base "
          f"(primary_metric)\nGSM8K int4 acc = {rows[0]['int4_acc_full']:.4f} (test_metric)\n")

    if args.wandb:
        import wandb
        run = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                         group=args.wandb_group, name="denken/quality-gate-4axis",
                         job_type="quality_gate", config=out["panel"])
        for r in rows:
            wandb.log({f"gate/{r['axis']}/int4_acc": r["int4_acc"],
                       f"gate/{r['axis']}/base_acc": r["base_acc"],
                       f"gate/{r['axis']}/pct_of_base": r["pct_of_base"],
                       f"gate/{r['axis']}/pass": int(r["pass"])})
        wandb.summary["verdict"] = verdict
        wandb.summary["primary_metric_worst_pct_of_base"] = worst["pct_of_base"]
        wandb.summary["worst_axis"] = worst["axis"]
        wandb.summary["test_metric_gsm8k_int4_acc"] = rows[0]["int4_acc"]
        tbl = wandb.Table(columns=["axis", "int4_acc", "base_acc", "pct_of_base", "pass", "banked"],
                          data=[[r["axis"], r["int4_acc"], r["base_acc"], r["pct_of_base"],
                                 int(r["pass"]), int(bool(r.get("banked")))] for r in rows])
        wandb.log({"gate_table": tbl})
        print(f"[wandb] logged run {run.id} group={args.wandb_group}")
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
