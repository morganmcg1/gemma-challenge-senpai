#!/usr/bin/env python3
"""PR #543: offline minimum-damage drop-set census -- is ANY layer-drop quality-safe?

Generalizes #539's ablate5 forward-hook counterfactual (forward_divergence.py:
forward_hidden(..., skip_layers=...)) from the fixed osoi5 set {2,3,4,36,37} to an
arbitrary --ablate_layers set, run on the CLEAN base-bf16 42L model. Builds the
depth<->quality Pareto:

  For an ablation set S, identity-skip those decoder layers (residual passthrough) and
  measure, over the SAME teacher-forced base-bf16 greedy keepset stream (#528 decode):
    * flip-rate  = fraction of completion-token positions whose 16k-keepset argmax
                   differs from the UN-ablated base-bf16 keepset argmax (macro-avg / item,
                   matching #539's int4_div_rate_mean=8.6% definition);
    * final masked-cos = final post-norm hidden cosine vs clean (Gemma massive dims masked).

Sweep:
  1. k=1 census: every single layer 0..41 -> per-layer removal-damage ranking (the thing
     Block-Influence failed to predict in #539).
  2. selected k=2 / k=5 from the lowest-removal-damage layers (greedy, not exhaustive),
     plus the osoi5 anchor {2,3,4,36,37}.
  3. Block-Influence-selected sets of the same size (k lowest-BI layers) -> quantify how
     badly BI mis-ranks removal damage.

Offline quality-safe SCREEN: a set "passes offline" iff flip-rate <= int4-only floor
(8.6% in #539; recomputed in-frame here). Low raw-ablate damage is NECESSARY-but-not-
sufficient for a quality-safe heal'd prune (raw ablation has no heal; #539 showed heal
recovers little so this is a reasonable proxy, not a guaranteed bound).

LOCAL, analysis-only. No HF Job, no submission. base-bf16 + base-int4(floor) only.

Run (assigned GPU; vllm022 venv has transformers+compressed-tensors):
  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/bake_damage_locus/ablate_pareto.py \
      --n-per-task 30 --max-comp 320 \
      --out research/validity/bake_damage_locus/ablate_pareto.npz \
      [--wandb] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import time

# reuse the EXACT #539 counterfactual machinery (do not re-implement)
import forward_divergence as fd
from forward_divergence import (
    BASE_BF16, BASE_INT4, KCG, N_LAYERS_BASE, REMOVED, TASKS,
    build_inputs, cos_relmse, forward_hidden, load_model, log,
)

INT4_FLOOR_539 = 0.0864236111111111  # #539 int4_div_rate_mean (macro-avg flip rate vs bf16)

# Inference-stage bands (Lad/Gurnee/Tegmark 2406.19384, scaled 32L->42L). Used for narrative
# tagging only: literature says detokenization (early) + residual-sharpening (late) stages are
# structurally non-removable; the mid prediction-ensembling band is the redundant one.
STAGE_BANDS = [("detok", 0, 5), ("feature_eng", 6, 20), ("ensembling", 21, 35), ("sharpening", 36, 41)]
MID_BAND = (6, 35)  # avoid first-6 / last-6 per Gromov 2403.17887 (contiguous mid-block candidate)


def stage_of(l):
    for name, lo, hi in STAGE_BANDS:
        if lo <= l <= hi:
            return name
    return "?"


def best_contig(census, w, lo=MID_BAND[0], hi=MID_BAND[1]):
    """Lowest-summed-k1-flip contiguous window of width w fully inside [lo,hi] (literature-
    optimal quality-safe candidate). Returns None if the band isn't fully censused."""
    best_layers, best_v = None, float("inf")
    for s in range(lo, hi - w + 2):
        layers = list(range(s, s + w))
        if not all(l in census for l in layers):
            continue
        v = sum(census[l]["flip_rate"] for l in layers)
        if v < best_v:
            best_v, best_layers = v, layers
    return best_layers


def block_influence_bf16(hs, prompt_len, acc):
    """Accumulate per-layer BI(l)=1-cos(h_in,h_out) over completion positions (bf16 substrate)."""
    import torch
    for l in range(N_LAYERS_BASE):
        hin = hs[l][0].float()[prompt_len:]
        hout = hs[l + 1][0].float()[prompt_len:]
        if hin.shape[0] == 0:
            continue
        c = torch.nn.functional.cosine_similarity(hin, hout, dim=-1)
        acc["sum"][l] += float((1 - c).sum())
        acc["cnt"][l] += c.numel()


def eval_set(m, items, keep_t, ref_argmax, clean_final, final_mask, skip):
    """Forward base-bf16 with `skip` layers identity-skipped; return macro flip-rate +
    final masked-cos vs the clean (un-ablated) reference, plus per-task breakdown."""
    import torch
    skip = set(skip) if skip else None
    per_item_flip, per_item_cos = [], []
    per_task = {t: [] for t in TASKS}
    micro_flip_hits = micro_flip_tot = 0
    for i, it in enumerate(items):
        hs, logits = forward_hidden(m, it["ids"], skip_layers=skip)
        pl = it["prompt_len"]
        am = logits[0][pl:, :][:, keep_t].argmax(-1)  # [Tc] keepset-column argmax
        ra = ref_argmax[i].to(am.device)
        Tc = min(am.shape[0], ra.shape[0])
        if Tc == 0:
            del hs, logits
            continue
        flip_pos = (am[:Tc] != ra[:Tc])
        flip = float(flip_pos.float().mean())
        per_item_flip.append(flip)
        per_task[it["task"]].append(flip)
        micro_flip_hits += int(flip_pos.sum())
        micro_flip_tot += int(Tc)
        # final masked-cos vs clean final hidden
        af = hs[-1][0][pl:].float()
        cf = clean_final[i].to(af.device).float()
        _, cm, _ = cos_relmse(af[:Tc], cf[:Tc], final_mask.to(af.device))
        per_item_cos.append(float(cm.mean()))
        del hs, logits
    import numpy as np
    return {
        "flip_rate": float(np.mean(per_item_flip)) if per_item_flip else float("nan"),
        "flip_rate_micro": float(micro_flip_hits / max(micro_flip_tot, 1)),
        "final_cos_m": float(np.mean(per_item_cos)) if per_item_cos else float("nan"),
        "per_task_flip": {t: (float(np.mean(v)) if v else float("nan")) for t, v in per_task.items()},
        "n_positions": micro_flip_tot,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=30)
    ap.add_argument("--max-comp", type=int, default=320)
    ap.add_argument("--keepset", default=os.path.join(KCG, "osoi5_baked_keepset_16k.json"))
    ap.add_argument("--out", default=os.path.join(fd.HERE, "ablate_pareto.npz"))
    ap.add_argument("--census-layers", default="", help="comma list to override (smoke); default all 42")
    ap.add_argument("--screen-floor", type=float, default=-1.0, help="override int4 floor; <0 = measure")
    ap.add_argument("--no-int4-floor", action="store_true", help="skip int4 floor load, use #539 0.086")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="bake-damage-locus")
    ap.add_argument("--wandb-name", default="kanna/offline-minimum-damage-dropset")
    args = ap.parse_args()

    import numpy as np
    import torch

    t_start = time.time()
    keep_ids = list(json.load(open(args.keepset))["keep_ids"])
    keep_t = torch.tensor(keep_ids, device="cuda")
    n_keep = len(keep_ids)
    log(f"keepset {n_keep} ids")

    items = build_inputs(args.n_per_task, args.max_comp, args.smoke)

    census_layers = ([int(x) for x in args.census_layers.split(",") if x != ""]
                     if args.census_layers else list(range(N_LAYERS_BASE)))
    log(f"census over {len(census_layers)} single layers: {census_layers}")

    # ---- CLEAN base-bf16 pass: reference keepset argmax + clean final hidden + bf16 BI + final mask
    log("=== CLEAN base-bf16 (un-ablated reference) ===")
    m = load_model(BASE_BF16)
    torch.cuda.reset_peak_memory_stats()
    ref_argmax, clean_final = [], []
    bi_acc = {"sum": np.zeros(N_LAYERS_BASE), "cnt": np.zeros(N_LAYERS_BASE)}
    H = None
    dimabs_final = None
    dimcnt = 0
    for it in items:
        hs, logits = forward_hidden(m, it["ids"])
        pl = it["prompt_len"]
        am = logits[0][pl:, :][:, keep_t].argmax(-1).to("cpu")  # [Tc]
        ref_argmax.append(am)
        hf = hs[-1][0][pl:]                                      # [Tc,H] final post-norm
        clean_final.append(hf.to("cpu", torch.bfloat16))
        if H is None:
            H = hf.shape[-1]
            dimabs_final = torch.zeros(H, device="cuda")
        if hf.shape[0] > 0:
            dimabs_final += hf.float().abs().sum(0)
            dimcnt += hf.shape[0]
        block_influence_bf16(hs, pl, bi_acc)
        del hs, logits
    peak_clean = torch.cuda.max_memory_allocated() / 1e9
    # final-layer massive-dim mask (Gemma huge-norm dims compress cosine -> mask >30x median, cap 64)
    meanabs = (dimabs_final / max(dimcnt, 1)).cpu()
    med = float(meanabs.median().clamp_min(1e-9))
    big = (meanabs > 30.0 * med).nonzero().flatten().tolist()
    big = sorted(big, key=lambda d: -float(meanabs[d]))[:64]
    final_mask = torch.ones(H)
    final_mask[big] = 0.0
    bi_bf16 = bi_acc["sum"] / np.clip(bi_acc["cnt"], 1, None)
    log(f"CLEAN done; peak {peak_clean:.1f}GB; H={H}; final massive dims={len(big)}; "
        f"bf16 BI mean={bi_bf16.mean():.4f} min@L{int(bi_bf16.argmin())}={bi_bf16.min():.4f}")
    log(f"bf16 BI low->high (first 8): {np.argsort(bi_bf16)[:8].tolist()}")

    # ---- k=1 census (all single layers)
    log("=== k=1 census ===")
    census = {}
    for j, l in enumerate(census_layers):
        r = eval_set(m, items, keep_t, ref_argmax, clean_final, final_mask, {l})
        census[l] = r
        log(f"  L{l:2d}: flip={r['flip_rate']*100:6.2f}%  cos={r['final_cos_m']:.4f}  "
            f"(bf16BI={bi_bf16[l]:.4f})  [{j+1}/{len(census_layers)}, {time.time()-t_start:.0f}s]")

    ranked = sorted(census_layers, key=lambda l: census[l]["flip_rate"])
    min_layer = ranked[0]
    log(f"min-damage single layer = L{min_layer} flip={census[min_layer]['flip_rate']*100:.2f}%")

    # ---- selected multi-sets: removal-damage greedy, BI-greedy, osoi5 anchor
    bi_order_low = np.argsort(bi_bf16).tolist()  # low BI first (what BI would drop)
    sel_specs = {
        "rd_k2": sorted(ranked[:2]),
        "rd_k5": sorted(ranked[:5]),
        "bi_k2": sorted(bi_order_low[:2]),
        "bi_k5": sorted(bi_order_low[:5]),
        "osoi5": sorted(REMOVED),
    }
    # literature-optimal contiguous mid-block candidates (Gromov 2403.17887; Lu 2411.15558):
    # most likely set to pass the screen -> if even these fail, the negative is decisive.
    for w in (2, 5):
        c = best_contig(census, w)
        if c is not None:
            sel_specs[f"mid_contig_k{w}"] = c
    # BI-selected single layer (measure fresh only if it isn't already in the census subset)
    if bi_order_low[0] not in census:
        sel_specs["bi_k1"] = [int(bi_order_low[0])]
    log(f"selected sets: {sel_specs}")
    sel_results = {}
    for name, s in sel_specs.items():
        r = eval_set(m, items, keep_t, ref_argmax, clean_final, final_mask, s)
        sel_results[name] = r
        log(f"  {name} {s}: flip={r['flip_rate']*100:6.2f}%  cos={r['final_cos_m']:.4f}")

    del m
    import gc
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- int4-only floor, reproduced IN-FRAME (int4 keepset argmax vs the SAME clean bf16 ref)
    int4_floor = None
    if args.screen_floor >= 0:
        int4_floor = args.screen_floor
        log(f"using override screen floor {int4_floor:.4f}")
    elif args.no_int4_floor:
        int4_floor = INT4_FLOOR_539
        log(f"skipping int4 load; using #539 floor {int4_floor:.4f}")
    else:
        log("=== int4-only floor (base-int4 vs clean bf16 ref) ===")
        m4 = load_model(BASE_INT4)
        per_item = []
        for i, it in enumerate(items):
            hs, logits = forward_hidden(m4, it["ids"])
            pl = it["prompt_len"]
            am = logits[0][pl:, :][:, keep_t].argmax(-1)
            ra = ref_argmax[i].to(am.device)
            Tc = min(am.shape[0], ra.shape[0])
            if Tc:
                per_item.append(float((am[:Tc] != ra[:Tc]).float().mean()))
            del hs, logits
        int4_floor = float(np.mean(per_item))
        del m4
        gc.collect(); torch.cuda.empty_cache()
        log(f"int4 floor (measured, in-frame) = {int4_floor*100:.2f}%  (#539 ref {INT4_FLOOR_539*100:.2f}%)")

    # ---- SCREEN: pass iff macro flip-rate <= int4 floor
    floor = float(int4_floor)
    candidates = []  # (label, k, layers, flip, cos)
    for l in census_layers:
        candidates.append((f"L{l}", 1, [l], census[l]["flip_rate"], census[l]["final_cos_m"]))
    for name, s in sel_specs.items():
        r = sel_results[name]
        candidates.append((name, len(s), list(s), r["flip_rate"], r["final_cos_m"]))
    passing = [c for c in candidates if c[3] <= floor]
    passing.sort(key=lambda c: c[3])
    best = min(candidates, key=lambda c: c[3])
    any_safe = len(passing) >= 1
    log(f"SCREEN floor={floor*100:.2f}%  n_passing={len(passing)}  "
        f"best={best[0]} {best[2]} flip={best[3]*100:.2f}%")

    # ---- Pareto points (removal-selected vs BI-selected at k in {1,2,5} + osoi5)
    def point(sel, k, layers, res):
        return {"selection": sel, "k": k, "layers": list(layers),
                "flip_rate": float(res["flip_rate"]), "flip_rate_micro": float(res["flip_rate_micro"]),
                "final_cos_m": float(res["final_cos_m"]), "per_task_flip": res["per_task_flip"],
                "passes_screen": bool(res["flip_rate"] <= floor)}
    bi1 = int(bi_order_low[0])
    bi1_res = census[bi1] if bi1 in census else sel_results["bi_k1"]
    pareto = [
        point("removal", 1, [min_layer], census[min_layer]),
        point("removal", 2, sel_specs["rd_k2"], sel_results["rd_k2"]),
        point("removal", 5, sel_specs["rd_k5"], sel_results["rd_k5"]),
        point("blockinfluence", 1, [bi1], bi1_res),
        point("blockinfluence", 2, sel_specs["bi_k2"], sel_results["bi_k2"]),
        point("blockinfluence", 5, sel_specs["bi_k5"], sel_results["bi_k5"]),
        point("osoi5_anchor", 5, sel_specs["osoi5"], sel_results["osoi5"]),
    ]
    for w in (2, 5):
        nm = f"mid_contig_k{w}"
        if nm in sel_specs:
            pareto.append(point("mid_contig", w, sel_specs[nm], sel_results[nm]))

    # int4 BI cross-ref from #539 results.npz (if present)
    int4_bi = None
    npz539 = os.path.join(fd.HERE, "results.npz")
    if os.path.exists(npz539):
        try:
            int4_bi = np.load(npz539, allow_pickle=True)["block_influence_int4"].tolist()
        except Exception:  # noqa: BLE001
            int4_bi = None

    k1_table = [{
        "layer": int(l),
        "stage": stage_of(l),
        "flip_rate": float(census[l]["flip_rate"]),
        "flip_rate_micro": float(census[l]["flip_rate_micro"]),
        "final_cos_m": float(census[l]["final_cos_m"]),
        "bf16_bi": float(bi_bf16[l]),
        "int4_bi": (float(int4_bi[l]) if int4_bi is not None else None),
        "passes_screen": bool(census[l]["flip_rate"] <= floor),
        "per_task_flip": census[l]["per_task_flip"],
    } for l in census_layers]
    k1_sorted = sorted(k1_table, key=lambda r: r["flip_rate"])

    verdict = {
        "pr": 543,
        "analysis_only": True,
        "official_tps": 0,
        "n_items": len(items),
        "n_keep": n_keep,
        "hidden_size": int(H),
        "smoke": bool(args.smoke),
        "int4_floor_measured": float(int4_floor),
        "int4_floor_539": INT4_FLOOR_539,
        "screen_floor": floor,
        # k=1 census
        "k1_census": k1_sorted,
        "min_damage_layer": int(min_layer),
        "min_damage_flip_rate": float(census[min_layer]["flip_rate"]),
        "min_damage_final_cos_m": float(census[min_layer]["final_cos_m"]),
        "k1_max_damage_layer": int(ranked[-1]),
        "k1_max_damage_flip_rate": float(census[ranked[-1]]["flip_rate"]),
        # selection-rule comparison
        "bi_order_low_to_high": [int(x) for x in bi_order_low],
        "depth_quality_pareto": pareto,
        # screen
        "n_dropsets_passing_offline_screen": len(passing),
        "passing_dropsets": [{"label": c[0], "k": c[1], "layers": c[2], "flip_rate": float(c[3])}
                             for c in passing],
        "best_dropset_label": best[0],
        "best_dropset_layers": list(best[2]),
        "best_dropset_flip_rate": float(best[3]),
        "best_dropset_final_cos_m": float(best[4]),
        "best_dropset_k": int(best[1]),
        # top-line booleans
        "any_quality_safe_drop_exists": bool(any_safe),
        "depth_prune_lane_open": bool(any_safe),
        "screen_semantics": "NECESSARY-but-not-sufficient: low raw-ablate flip-rate (<= int4 floor) "
                            "is required for a quality-safe heal'd prune but a served heal'd run "
                            "(wirbel #541) stays binding. raw ablation has no heal.",
    }

    # ---- save raw + verdict
    np.savez(args.out,
             census_layers=np.array(census_layers),
             k1_flip=np.array([census[l]["flip_rate"] for l in census_layers]),
             k1_cos=np.array([census[l]["final_cos_m"] for l in census_layers]),
             bi_bf16=bi_bf16,
             int4_bi=(np.array(int4_bi) if int4_bi is not None else np.array([])),
             int4_floor=float(int4_floor),
             ref_lens=np.array([int(r.shape[0]) for r in ref_argmax]))
    vpath = args.out.replace(".npz", "_verdict.json")
    json.dump(verdict, open(vpath, "w"), indent=1)
    log(f"saved {args.out} and {vpath}  (elapsed {time.time()-t_start:.0f}s)")

    # ---- structured terminal marker (machine-readable)
    marker = {
        "pr": 543, "analysis_only": True, "official_tps": 0,
        "n_items": len(items), "n_keep": n_keep,
        "int4_floor": round(float(int4_floor), 5),
        "min_damage_layer": int(min_layer),
        "min_damage_flip_rate": round(float(census[min_layer]["flip_rate"]), 5),
        "n_dropsets_passing_offline_screen": len(passing),
        "best_dropset_layers": list(best[2]),
        "best_dropset_flip_rate": round(float(best[3]), 5),
        "best_dropset_k": int(best[1]),
        "any_quality_safe_drop_exists": bool(any_safe),
        "depth_prune_lane_open": bool(any_safe),
        "osoi5_flip_rate": round(float(sel_results["osoi5"]["flip_rate"]), 5),
        "bi_k5_layers": sel_specs["bi_k5"],
        "bi_k5_flip_rate": round(float(sel_results["bi_k5"]["flip_rate"]), 5),
        "rd_k5_layers": sel_specs["rd_k5"],
        "rd_k5_flip_rate": round(float(sel_results["rd_k5"]["flip_rate"]), 5),
    }
    if "mid_contig_k5" in sel_results:
        marker["mid_contig_k5_layers"] = sel_specs["mid_contig_k5"]
        marker["mid_contig_k5_flip_rate"] = round(float(sel_results["mid_contig_k5"]["flip_rate"]), 5)
    if "mid_contig_k2" in sel_results:
        marker["mid_contig_k2_layers"] = sel_specs["mid_contig_k2"]
        marker["mid_contig_k2_flip_rate"] = round(float(sel_results["mid_contig_k2"]["flip_rate"]), 5)
    print("SENPAI-MARKER " + json.dumps(marker), flush=True)

    if args.wandb and not args.smoke:
        log_wandb(args, verdict, k1_table, bi_bf16, int4_bi, pareto)
    log("DONE")
    return 0


def log_wandb(args, verdict, k1_table, bi_bf16, int4_bi, pareto):
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group=args.wandb_group, name=args.wandb_name, job_type="analysis",
        config={"analysis_only": True, "official_tps": 0, "pr": 543,
                "n_items": verdict["n_items"], "n_keep": verdict["n_keep"],
                "screen_floor": verdict["screen_floor"],
                "int4_floor_measured": verdict["int4_floor_measured"]},
    )
    # k=1 census table
    t1 = wandb.Table(columns=["layer", "flip_rate", "final_cos_m", "bf16_bi", "int4_bi", "passes_screen"])
    for r in k1_table:
        t1.add_data(r["layer"], r["flip_rate"], r["final_cos_m"], r["bf16_bi"],
                    (r["int4_bi"] if r["int4_bi"] is not None else float("nan")), r["passes_screen"])
    wandb.log({"k1_census": t1})
    # pareto table
    t2 = wandb.Table(columns=["selection", "k", "layers", "flip_rate", "final_cos_m", "passes_screen"])
    for p in pareto:
        t2.add_data(p["selection"], p["k"], str(p["layers"]), p["flip_rate"],
                    p["final_cos_m"], p["passes_screen"])
    wandb.log({"depth_quality_pareto": t2})
    scal = {
        "screen/int4_floor_measured": verdict["int4_floor_measured"],
        "screen/n_dropsets_passing": verdict["n_dropsets_passing_offline_screen"],
        "screen/min_damage_flip_rate": verdict["min_damage_flip_rate"],
        "screen/min_damage_layer": verdict["min_damage_layer"],
        "screen/best_dropset_flip_rate": verdict["best_dropset_flip_rate"],
        "screen/best_dropset_k": verdict["best_dropset_k"],
        "screen/k1_max_damage_flip_rate": verdict["k1_max_damage_flip_rate"],
        "pareto/osoi5_flip": next(p["flip_rate"] for p in pareto if p["selection"] == "osoi5_anchor"),
        "pareto/bi_k5_flip": next(p["flip_rate"] for p in pareto if p["selection"] == "blockinfluence" and p["k"] == 5),
        "pareto/rd_k5_flip": next(p["flip_rate"] for p in pareto if p["selection"] == "removal" and p["k"] == 5),
    }
    wandb.log(scal)
    sm = dict(scal)
    sm.update({
        "any_quality_safe_drop_exists": verdict["any_quality_safe_drop_exists"],
        "depth_prune_lane_open": verdict["depth_prune_lane_open"],
        "min_damage_layer": verdict["min_damage_layer"],
        "best_dropset_layers": str(verdict["best_dropset_layers"]),
    })
    wandb.summary.update(sm)
    print(f"[wandb] run {run.id} ({run.name})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
