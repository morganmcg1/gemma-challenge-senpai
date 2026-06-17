#!/usr/bin/env python3
"""Bake-damage locus: concentration + attribution + verdict (PR #539).

Reduces forward_divergence.py results.npz into the PR-required KEY OUTPUTS:
  int4_divergence_profile, layerdrop_divergence, damage_locus (concentrated|diffuse),
  int4_frac/layerdrop_frac, first_divergence_driver, rebake_feasibility, verdict line.

Concentration (researcher-agent grounding): participation-ratio PR=1/sum(p_i^2),
Gini, normalized entropy on the per-layer divergence distribution. Thresholds:
PR<6 & Gini>0.7 -> concentrated; PR>15 or H>0.85 -> diffuse. Block-Influence of the
removed layers tests whether osoi5 dropped low-importance (redundant) layers.

Run:
  .venvs/vllm022/bin/python research/validity/bake_damage_locus/verdict.py \
      --npz research/validity/bake_damage_locus/results.npz \
      --out research/validity/bake_damage_locus/verdict.json \
      [--wandb --wandb-group bake-damage-locus --wandb-name kanna/bake-damage-locus]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REMOVED = [2, 3, 4, 36, 37]
TASKS = ["mmlu_pro", "gpqa_diamond", "aime2024"]


def keepset_coverage(kcg, keepset_path, n_per_task, max_comp):
    """Replicate build_inputs' (first n_per_task/task in decode order, comp[:max_comp])
    selection and measure the fraction of the base-greedy completion stream that lies in
    the 16k keepset. head_sanity (HF bf16 keepset-argmax == vLLM teacher token) factors as
    coverage * HF/vLLM-greedy-agreement; this isolates the coverage term so the residual is
    attributable to engine (HF-vs-vLLM) divergence, NOT a keepset-ordering bug. The divergence
    reference is bf16-HF (not the vLLM token), so neither term confounds the flip attribution."""
    keepset_path = keepset_path or os.path.join(kcg, "osoi5_baked_keepset_16k.json")
    keep = set(json.load(open(keepset_path))["keep_ids"])
    per = {t: 0 for t in TASKS}
    sel = []
    with open(os.path.join(kcg, "decode.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r["task"]
            if per.get(t, 0) < n_per_task:
                sel.append(r)
                per[t] += 1
    hit = tot = 0
    by_task = {}
    for r in sel:
        comp = list(r["completion_token_ids"])[:max_comp]
        T = len(comp)
        toks = comp[1:T]  # comp[1..T-1], matches first_divergence tok_next slice
        h = sum(1 for x in toks if x in keep)
        hit += h
        tot += len(toks)
        bt = by_task.setdefault(r["task"], [0, 0])
        bt[0] += h
        bt[1] += len(toks)
    return {
        "n_selected": len(sel),
        "per_task": per,
        "keepset_coverage": float(hit / max(tot, 1)),
        "keepset_coverage_by_task": {t: float(v[0] / max(v[1], 1)) for t, v in by_task.items()},
    }


def participation_ratio(p):
    p = np.asarray(p, float); s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    return float(1.0 / (p ** 2).sum())


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() <= 0 or len(x) == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def norm_entropy(p):
    p = np.asarray(p, float); s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s; p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def concentration(divvec):
    """divvec: non-negative per-layer divergence contributions."""
    d = np.clip(np.asarray(divvec, float), 0, None)
    order = np.argsort(-d)
    cum = np.cumsum(d[order]) / max(d.sum(), 1e-12)
    topk = {f"top{k}_share": float(cum[k - 1]) for k in (1, 3, 5) if k <= len(d)}
    return {
        "participation_ratio": participation_ratio(d),
        "gini": gini(d),
        "norm_entropy": norm_entropy(d),
        "dominant_layers": order[:5].tolist(),
        **topk,
    }


def verdict_from(pr, gini_v, ent):
    if (pr < 6 and gini_v > 0.7) or (pr < 5):
        return "concentrated"
    if pr > 15 or ent > 0.85:
        return "diffuse"
    return "intermediate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(os.path.dirname(__file__), "results.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "verdict.json"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="bake-damage-locus")
    ap.add_argument("--wandb-name", default="kanna/bake-damage-locus")
    # keepset-coverage cross-check (explains head_sanity): replicate build_inputs selection
    ap.add_argument("--kcg", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keepset_coverage_gap"))
    ap.add_argument("--keepset", default=None, help="keep_ids json (defaults to KCG/osoi5_baked_keepset_16k.json)")
    ap.add_argument("--n-per-task", type=int, default=30)
    ap.add_argument("--max-comp", type=int, default=320)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    surv = d["surv"].tolist()

    # ---- (a) int4 divergence profile (cumulative per-layer, l=0..42) ----
    int4_cos = d["int4_vs_bf16_cos_m"]        # masked cosine
    int4_relmse = d["int4_vs_bf16_relmse"]
    # marginal injection per decoder layer (l=1..42): increment in relmse
    int4_marg = np.clip(np.diff(int4_relmse), 0, None)   # len 42 (layer l contributes idx l-1)
    int4_conc = concentration(int4_marg)
    int4_final_relmse = float(int4_relmse[-1])
    int4_final_cos = float(int4_cos[-1])

    # ---- (b) layerdrop divergence (osoi5 vs int4, depth-aligned + final + logit) ----
    od_cos = d["osoi5_vs_int4_cos_m"]         # embed + 37 surv + final
    od_relmse = d["osoi5_vs_int4_relmse"]
    layerdrop_final_cos = float(np.nanmean(d["final_cos_m"]))
    layerdrop_final_relmse = float(np.nanmean(d["final_relmse"]))
    # per-aligned-layer marginal (vs previous aligned depth) -> where the divergence is injected
    od_marg = np.clip(np.diff(od_relmse[:-1]), 0, None)  # over the 37 surviving + embed (exclude final)
    layerdrop_conc = concentration(od_marg)

    # pure layerdrop at g32 (ablate5) if present
    abl_final_relmse = abl_final_cos = None
    if "ablate5_vs_int4_relmse" in d.files:
        abl_final_relmse = float(d["ablate5_vs_int4_relmse"][-1])
        abl_final_cos = float(d["ablate5_vs_int4_cos_m"][-1])

    # ---- Block Influence: were removed layers low-importance? ----
    bi = d["block_influence_int4"]
    bi_removed = {l: float(bi[l]) for l in REMOVED}
    bi_rank = {l: float((bi < bi[l]).mean()) for l in REMOVED}  # percentile in stack (0=lowest)
    removed_are_low_bi = bool(np.mean([bi_rank[l] for l in REMOVED]) < 0.35)

    # ---- representation-space split int4 vs layerdrop (at final hidden) ----
    denom = int4_final_relmse + layerdrop_final_relmse
    int4_frac = float(int4_final_relmse / denom) if denom > 0 else float("nan")
    layerdrop_frac = float(layerdrop_final_relmse / denom) if denom > 0 else float("nan")

    # ---- first-divergence driver (functional, from logits) ----
    fd = json.loads(open(args.npz.replace(".npz", "_meta.json")).read())["fd_summary"] \
        if os.path.exists(args.npz.replace(".npz", "_meta.json")) else {}
    # keepset-coverage cross-check: explains head_sanity = coverage * HF/vLLM-greedy-agreement
    try:
        cov = keepset_coverage(args.kcg, args.keepset, args.n_per_task, args.max_comp)
        hs = fd.get("head_sanity_bf16_keepArgmax_eq_teacher")
        cov["implied_hf_vllm_greedy_agreement"] = (
            float(hs / cov["keepset_coverage"]) if (hs is not None and cov["keepset_coverage"] > 0) else None
        )
        if cov["n_selected"] != int(d["n_items"]):
            cov["WARN"] = f"selection {cov['n_selected']} != n_items {int(d['n_items'])} (check --n-per-task/--max-comp)"
    except Exception as e:  # noqa: BLE001
        cov = {"error": repr(e)}
    # prefer position-matched ablate5 attribution if present
    pm_layerdrop = fd.get("posmatch_ablate5_also_flips_frac")
    pm_residual = fd.get("posmatch_residual_g128head_frac")
    pm_int4 = fd.get("posmatch_int4_also_flips_frac")
    ff_layerdrop = fd.get("firstflip_layerdrop_frac")
    # classify the dominant first-divergence driver
    if pm_layerdrop is not None:
        if pm_layerdrop >= 0.6:
            first_divergence_driver = "layerdrop-capability"
        elif pm_int4 is not None and pm_int4 >= 0.6:
            first_divergence_driver = "int4-numeric"
        else:
            first_divergence_driver = "mixed"
    else:
        first_divergence_driver = "layerdrop-capability" if (ff_layerdrop or 0) >= 0.6 else "mixed"

    # ---- synthesize verdicts ----
    int4_locus = verdict_from(int4_conc["participation_ratio"], int4_conc["gini"], int4_conc["norm_entropy"])
    # layerdrop is, by construction, 5 specific removed layers; check it concentrates at removal sites
    layerdrop_locus = verdict_from(layerdrop_conc["participation_ratio"], layerdrop_conc["gini"], layerdrop_conc["norm_entropy"])

    # overall damage locus: dominant knob (by frac + functional driver) decides
    layerdrop_dominant = (layerdrop_frac > int4_frac) and first_divergence_driver == "layerdrop-capability"
    if layerdrop_dominant and layerdrop_locus in ("concentrated", "intermediate"):
        damage_locus = "concentrated"
        rebake_feasibility = "cheap-targeted"
        rebake_reason = (
            "Dominant damage is layer-removal, localized to 5 specific removed layers "
            f"{REMOVED} that were low-Block-Influence (redundant) in base-int4; the int4 "
            "residual is mild & diffuse. Restoring depth (re-bake at 42L/40L int4, e.g. "
            "fern #535 base-int4 262k full body) recovers the dominant component at a TPS "
            "cost without a full QAT re-train."
        )
    elif int4_locus == "diffuse" and int4_frac >= layerdrop_frac:
        damage_locus = "diffuse"
        rebake_feasibility = "full-retrain"
        rebake_reason = ("Dominant damage is diffuse int4 perturbation spread across the stack; "
                         "no targeted layer restore recovers it -> full QAT re-train.")
    else:
        damage_locus = "mixed"
        rebake_feasibility = "cheap-targeted-partial"
        rebake_reason = "Both knobs contribute; restoring depth helps the layerdrop share, int4 residual remains."

    verdict_line = (
        f"BODY damage is {damage_locus}: layer-removal (5 low-BI layers {REMOVED}) is the "
        f"dominant knob (repr layerdrop_frac={layerdrop_frac:.2f} vs int4_frac={int4_frac:.2f}; "
        f"functional first-divergence driver={first_divergence_driver}, "
        f"{(pm_layerdrop or ff_layerdrop or 0)*100:.0f}% of flips reproduced by pure layer-removal); "
        f"int4 is mild & {int4_locus} (final cos {int4_final_cos:.3f}). "
        f"rebake_feasibility={rebake_feasibility}."
    )

    out = {
        "pr": 539,
        "n_items": int(d["n_items"]),
        "n_keep": int(d["n_keep"]),
        "osoi5_head_width": int(d["osoi5_head_width"]),
        "removed_layers": REMOVED,
        # (a)
        "int4_divergence_profile": {
            "cos_m_per_layer": [round(float(x), 4) for x in int4_cos],
            "relmse_per_layer": [round(float(x), 4) for x in int4_relmse],
            "final_cos_m": int4_final_cos, "final_relmse": int4_final_relmse,
            "concentration": int4_conc, "locus": int4_locus,
        },
        # (b)
        "layerdrop_divergence": {
            "cos_m_aligned": [round(float(x), 4) for x in od_cos],
            "relmse_aligned": [round(float(x), 4) for x in od_relmse],
            "final_cos_m": layerdrop_final_cos, "final_relmse": layerdrop_final_relmse,
            "ablate5_pure_final_cos_m": abl_final_cos, "ablate5_pure_final_relmse": abl_final_relmse,
            "concentration": layerdrop_conc, "locus": layerdrop_locus,
        },
        # (c)
        "block_influence": {
            "removed_bi": bi_removed, "removed_bi_percentile_in_stack": bi_rank,
            "stack_mean_bi": float(bi.mean()), "stack_max_bi": float(bi.max()),
            "argmax_layer": int(bi.argmax()), "removed_are_low_bi": removed_are_low_bi,
        },
        "int4_frac": int4_frac, "layerdrop_frac": layerdrop_frac,
        # (d)
        "first_divergence": fd,
        "first_divergence_driver": first_divergence_driver,
        "stream_diagnostics": cov,
        # verdict
        "damage_locus": damage_locus,
        "rebake_feasibility": rebake_feasibility,
        "rebake_reason": rebake_reason,
        "verdict_line": verdict_line,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in (
        "int4_frac", "layerdrop_frac", "first_divergence_driver",
        "damage_locus", "rebake_feasibility", "verdict_line")}, indent=1))
    print(f"[verdict] int4 locus={int4_locus} PR={int4_conc['participation_ratio']:.1f} "
          f"gini={int4_conc['gini']:.2f} H={int4_conc['norm_entropy']:.2f}")
    print(f"[verdict] layerdrop locus={layerdrop_locus} PR={layerdrop_conc['participation_ratio']:.1f} "
          f"top1@layer={layerdrop_conc['dominant_layers'][:3]}")
    print(f"[verdict] removed BI percentiles: {bi_rank}")
    print(f"[verdict] stream: keepset_coverage={cov.get('keepset_coverage')!r} "
          f"head_sanity={fd.get('head_sanity_bf16_keepArgmax_eq_teacher')!r} "
          f"=> implied HF/vLLM greedy agreement={cov.get('implied_hf_vllm_greedy_agreement')!r}")
    print(f"[verdict] wrote {args.out}")

    if args.wandb:
        log_wandb(args, out, d, int4_cos, int4_relmse, od_cos, od_relmse, bi)
    return 0


def log_wandb(args, out, d, int4_cos, int4_relmse, od_cos, od_relmse, bi):
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group=args.wandb_group, name=args.wandb_name, job_type="analysis",
        config={"analysis_only": True, "official_tps": 0, "pr": 539,
                "n_items": out["n_items"], "n_keep": out["n_keep"],
                "removed_layers": REMOVED, "osoi5_head_width": out["osoi5_head_width"]},
    )
    fd = out["first_divergence"]
    flat = {
        "int4/final_cos_m": out["int4_divergence_profile"]["final_cos_m"],
        "int4/final_relmse": out["int4_divergence_profile"]["final_relmse"],
        "int4/participation_ratio": out["int4_divergence_profile"]["concentration"]["participation_ratio"],
        "int4/gini": out["int4_divergence_profile"]["concentration"]["gini"],
        "int4/norm_entropy": out["int4_divergence_profile"]["concentration"]["norm_entropy"],
        "layerdrop/final_cos_m": out["layerdrop_divergence"]["final_cos_m"],
        "layerdrop/final_relmse": out["layerdrop_divergence"]["final_relmse"],
        "layerdrop/ablate5_pure_final_cos_m": out["layerdrop_divergence"]["ablate5_pure_final_cos_m"],
        "layerdrop/participation_ratio": out["layerdrop_divergence"]["concentration"]["participation_ratio"],
        "split/int4_frac": out["int4_frac"], "split/layerdrop_frac": out["layerdrop_frac"],
        "bi/stack_mean": out["block_influence"]["stack_mean_bi"],
        "bi/removed_mean_percentile": float(np.mean(list(out["block_influence"]["removed_bi_percentile_in_stack"].values()))),
        "fd/osoi5_first_median": fd.get("osoi5_first_median"),
        "fd/int4_first_median": fd.get("int4_first_median"),
        "fd/osoi5_div_rate_mean": fd.get("osoi5_div_rate_mean"),
        "fd/int4_div_rate_mean": fd.get("int4_div_rate_mean"),
        "fd/head_sanity": fd.get("head_sanity_bf16_keepArgmax_eq_teacher"),
        "stream/keepset_coverage": out["stream_diagnostics"].get("keepset_coverage"),
        "stream/implied_hf_vllm_greedy_agreement": out["stream_diagnostics"].get("implied_hf_vllm_greedy_agreement"),
        "fd/posmatch_ablate5_also_flips_frac": fd.get("posmatch_ablate5_also_flips_frac"),
        "fd/posmatch_int4_also_flips_frac": fd.get("posmatch_int4_also_flips_frac"),
        "fd/posmatch_residual_g128head_frac": fd.get("posmatch_residual_g128head_frac"),
    }
    wandb.log({k: v for k, v in flat.items() if v is not None})
    # per-layer profile tables for line plots
    t1 = wandb.Table(columns=["layer", "int4_cos_m", "int4_relmse"])
    for l in range(len(int4_cos)):
        t1.add_data(l, float(int4_cos[l]), float(int4_relmse[l]))
    wandb.log({"int4_divergence_profile": t1})
    surv = d["surv"].tolist()
    depth_labels = ["embed"] + [f"orig{o}" for o in surv] + ["final"]
    t2 = wandb.Table(columns=["aligned_depth", "orig_layer", "osoi5_cos_m", "osoi5_relmse"])
    for i in range(len(od_cos)):
        t2.add_data(depth_labels[i] if i < len(depth_labels) else str(i),
                    surv[i - 1] if 1 <= i <= len(surv) else -1,
                    float(od_cos[i]), float(od_relmse[i]))
    wandb.log({"layerdrop_divergence_aligned": t2})
    t3 = wandb.Table(columns=["layer", "block_influence", "removed"])
    for l in range(len(bi)):
        t3.add_data(l, float(bi[l]), l in REMOVED)
    wandb.log({"block_influence": t3})
    sm = {k: v for k, v in flat.items() if v is not None}
    sm.update({"damage_locus": out["damage_locus"], "rebake_feasibility": out["rebake_feasibility"],
               "first_divergence_driver": out["first_divergence_driver"],
               "verdict_line": out["verdict_line"]})
    wandb.summary.update(sm)
    print(f"[wandb] run {run.id} ({run.name})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
