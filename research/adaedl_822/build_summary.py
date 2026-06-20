#!/usr/bin/env python3
"""Consolidate the #822 AdaEDL early-stop study artifacts into one summary.json.

Sources (all under research/adaedl_822/):
  a_unpatched/passes_summary.json  -> ship anchor TPS + PPL
  b1_inf/passes_summary.json       -> control-inf TPS (life 1) + B1 records
  b1_inf/records.jsonl             -> Step-1 premise + offline counterfactual
  b2_sweep/passes_summary.json     -> tau TPS sweep (life 2, same-life as inf)
  b2_sweep/<pass>/rep*/decode_outputs.jsonl -> greedy-identity (noise-floor)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_jsonl(p: Path):
    out = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def premise(records):
    full = [r for r in records if r.get("H") and len(r["H"]) == r.get("draft_len")]
    K = max(r["K"] for r in records)
    accs = [r["accept_length"] for r in records]
    E = sum(accs) / len(accs)
    per_pos = defaultdict(lambda: [0, 0])  # j -> [n, n_accept]
    H_acc, H_rej = [], []
    for r in full:
        a = r["accept_length"]
        for j, h in enumerate(r["H"], start=1):
            per_pos[j][0] += 1
            if j <= a:
                per_pos[j][1] += 1
                H_acc.append(h)
            elif j == a + 1:
                H_rej.append(h)
    pos_acc = {j: per_pos[j][1] / per_pos[j][0] for j in sorted(per_pos)}
    ma = sum(H_acc) / len(H_acc)
    mr = sum(H_rej) / len(H_rej)
    return {
        "n_records": len(records), "K": K, "E_accept": E,
        "per_position_accept_rate": pos_acc,
        "meanH_accepted": ma, "meanH_reject_point": mr,
        "entropy_separation": mr - ma, "premise_holds": mr > ma,
    }


def counterfactual(records, taus):
    full = [r for r in records if r.get("H") and len(r["H"]) == r.get("draft_len")]
    K = max(r["K"] for r in records)
    E = sum(r["accept_length"] for r in full) / len(full)
    rows = {}
    for tau in taus:
        fwds, racc, stopped = [], [], 0
        for r in full:
            L = K
            for j, h in enumerate(r["H"], start=1):
                if h > tau:
                    L = j
                    break
            if L < K:
                stopped += 1
            fwds.append(L)
            racc.append(min(r["accept_length"], L))
        mf = sum(fwds) / len(fwds)
        mra = sum(racc) / len(racc)
        rows[tau] = {
            "fwd_per_step": mf, "fwd_saved": K - mf, "real_acc": mra,
            "acc_loss": E - mra, "acc_loss_pct": 100 * (E - mra) / E,
            "stop_rate": stopped / len(full),
        }
    return rows


def tps_of(summary_path: Path):
    d = json.loads(summary_path.read_text())
    return {p["name"]: {"thresh": p["thresh"], "tps_mean": p["tps_mean"],
                        "tps_std": p["tps_std"], "tps_list": p["tps_list"]}
            for p in d["passes"]}, d


def positionwise_agree(fa: Path, fb: Path):
    A = {(str(r["id"]), r["dataset_index"]): r["completion_token_ids"] for r in load_jsonl(fa)}
    B = {(str(r["id"]), r["dataset_index"]): r["completion_token_ids"] for r in load_jsonl(fb)}
    keys = sorted(set(A) & set(B))
    eq = tot = exact = 0
    for k in keys:
        a, b = A[k], B[k]
        n = min(len(a), len(b))
        e = sum(1 for i in range(n) if a[i] == b[i])
        eq += e
        tot += n
        if a == b:
            exact += 1
    return {"prompts": len(keys), "exact": exact,
            "positionwise_agree_pct": 100 * eq / tot if tot else 0.0}


def main():
    a_tps, a_full = tps_of(HERE / "a_unpatched/passes_summary.json")
    b1_tps, _ = tps_of(HERE / "b1_inf/passes_summary.json")
    b2_tps, _ = tps_of(HERE / "b2_sweep/passes_summary.json")
    records = load_jsonl(HERE / "b1_inf/records.jsonl")

    ship = a_tps["unpatched"]["tps_mean"]
    prem = premise(records)
    taus = [2.477, 1.449, 0.727, 0.402]
    cf = counterfactual(records, taus)

    # TPS table vs ship
    tps_table = {"unpatched_ship": a_tps["unpatched"],
                 "patched_inf_life1": b1_tps.get("inf_clean"),
                 "patched_inf_life2": b2_tps.get("inf")}
    for name, tau in [("tau2477", 2.477), ("tau1449", 1.449),
                      ("tau0727", 0.727), ("tau0402", 0.402)]:
        e = dict(b2_tps[name])
        e["pct_vs_ship"] = 100 * (e["tps_mean"] - ship) / ship
        e["pct_vs_patched_inf"] = 100 * (e["tps_mean"] - b2_tps["inf"]["tps_mean"]) / b2_tps["inf"]["tps_mean"]
        tps_table[name] = e
    tps_table["patched_inf_life2"]["pct_vs_ship"] = 100 * (b2_tps["inf"]["tps_mean"] - ship) / ship

    # greedy identity: noise floor vs early-stop, within B2 (same eager path)
    b2 = HERE / "b2_sweep"
    gi = {
        "noise_floor_inf_rep0_vs_rep1": positionwise_agree(
            b2 / "inf/rep0/decode_outputs.jsonl", b2 / "inf/rep1/decode_outputs.jsonl"),
    }
    for t in ["tau2477", "tau1449", "tau0727", "tau0402"]:
        gi[f"earlystop_inf_vs_{t}"] = positionwise_agree(
            b2 / "inf/rep0/decode_outputs.jsonl", b2 / f"{t}/rep0/decode_outputs.jsonl")

    best = max(["tau2477", "tau1449", "tau0727", "tau0402"],
               key=lambda n: b2_tps[n]["tps_mean"])
    summary = {
        "experiment": "AdaEDL entropy-gated draft early-stop (#822)",
        "local_exploratory": True,
        "hardware": "A10G (local), single-stream HTTP-serial decode proxy; NOT official a10g-small",
        "decode_config": {"num_prompts": 16, "output_len": 512, "temperature": 0.0,
                          "ignore_eos": True, "reps_per_point": 3},
        "ppl": {"value": a_full.get("ppl"), "n_records": a_full.get("ppl_num_records"),
                "n_tokens": a_full.get("ppl_num_tokens"),
                "note": "drafter-independent (prefill scoring); == shipped int4head 2.0027"},
        "premise_step1": prem,
        "counterfactual_offline": cf,
        "tps_table": tps_table,
        "greedy_identity_noise_floor_relative": gi,
        "best_tau": {"name": best, "thresh": b2_tps[best]["thresh"],
                     "tps_mean": b2_tps[best]["tps_mean"],
                     "pct_vs_ship": 100 * (b2_tps[best]["tps_mean"] - ship) / ship},
        "verdict": "NEGATIVE: lever real (+%.1f%% vs eager-entropy baseline) but entropy machinery "
                   "(de-graphed centroid selection + per-position H<->D sync) costs %.1f%% vs ship; "
                   "best early-stop nets %.1f%% vs ship. No fire." % (
                       tps_table[best]["pct_vs_patched_inf"],
                       tps_table["patched_inf_life2"]["pct_vs_ship"],
                       100 * (b2_tps[best]["tps_mean"] - ship) / ship),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))

    # human-readable
    print("=== AdaEDL #822 consolidated summary ===")
    print(f"PPL (n={summary['ppl']['n_records']}) = {summary['ppl']['value']:.4f}")
    print(f"premise: meanH acc={prem['meanH_accepted']:.4f} reject={prem['meanH_reject_point']:.4f} "
          f"sep=+{prem['entropy_separation']:.4f} -> HOLDS={prem['premise_holds']}")
    print(f"per-position accept: " + " ".join(f"{prem['per_position_accept_rate'][j]:.3f}"
                                              for j in sorted(prem['per_position_accept_rate'])))
    print(f"\n{'config':>20} {'tps_mean':>9} {'tps_std':>8} {'%vs_ship':>9} {'%vs_inf':>8}")
    order = ["unpatched_ship", "patched_inf_life2", "tau2477", "tau1449", "tau0727", "tau0402"]
    for k in order:
        e = tps_table[k]
        vs_ship = e.get("pct_vs_ship", 0.0)
        vs_inf = e.get("pct_vs_patched_inf", float("nan"))
        print(f"{k:>20} {e['tps_mean']:>9.3f} {e['tps_std']:>8.3f} {vs_ship:>+8.2f}% "
              f"{vs_inf:>+7.2f}%" if not math.isnan(vs_inf) else
              f"{k:>20} {e['tps_mean']:>9.3f} {e['tps_std']:>8.3f} {vs_ship:>+8.2f}% {'--':>8}")
    print(f"\ngreedy identity (positionwise agree %, B2 same-path):")
    for k, v in gi.items():
        print(f"  {k:>32}: {v['positionwise_agree_pct']:.2f}%  exact={v['exact']}/{v['prompts']}")
    print(f"\nVERDICT: {summary['verdict']}")
    print(f"\nwrote {HERE/'summary.json'}")


if __name__ == "__main__":
    main()
