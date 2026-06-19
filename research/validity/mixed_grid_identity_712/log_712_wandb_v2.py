#!/usr/bin/env python3
"""PR #712 v2: compute the MIXED_GRID_IDENTITY verdict from identity_results_v2.json and log
to W&B group `mixed-grid-identity-land`. Re-uses the measured report verbatim -- no recompute.

v2 schema differs from v1: each result carries a `phase` tag --
  - "a_tf_vs_tf"        : method (a) corrected -- routeB teacher-forced argmax vs ANCHOR
                          teacher-forced argmax (same context path). anchor-vs-anchor MUST be 0,
                          so every residual flip is purely the Route-B weight change.
  - "b_decode_vs_decode": method (b) -- routeB served greedy decode vs anchor served decode,
                          token-by-token with early-stop at first divergence (the served-gate test).

Analysis-only; sets the four served-isolation guards. Prints a paste-ready PR summary."""
from __future__ import annotations
try:
    import wandb as _wandb_real  # noqa: F401  (beat ./wandb namespace shadow)
except Exception:
    _wandb_real = None
import json, sys
from pathlib import Path
from statistics import mean

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "validity" / "mixed_grid_identity_712"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scripts.wandb_logging import (finish_wandb, init_wandb_run,  # noqa: E402
                                   log_json_artifact, log_summary)

RES = HERE / "identity_results_v2.json"


def agg_a(results, cfg, mode, alpha):
    """Aggregate method-(a) phase results over seeds."""
    rs = [r for r in results if r.get("phase") == "a_tf_vs_tf" and r["cfg"] == cfg
          and r["mode"] == mode and abs(r["alpha"] - alpha) < 1e-9]
    if not rs:
        return None
    return {
        "n_seeds": len(rs),
        "flip_rate_mean": mean(r["flip_rate"] for r in rs),
        "flip_rate_max": max(r["flip_rate"] for r in rs),
        "flips_mean": mean(r["n_flips"] for r in rs),
        "flips_max": max(r["n_flips"] for r in rs),
        "confident_flips_max": max(r["confident_flips"] for r in rs),
        "confident_flips_mean": mean(r["confident_flips"] for r in rs),
        "confident_rate_mean": mean(r["confident_rate"] for r in rs),
        "prompts_diverged_max": max(r["prompts_diverged"] for r in rs),
        "prompts_total": rs[0]["prompts_total"],
        "n_cont_positions": rs[0]["n_cont_positions"],
        "first_flip_min": min([r["first_flip_min"] for r in rs if r["first_flip_min"] is not None],
                              default=None),
        "first_flip_median": min([r["first_flip_median"] for r in rs
                                  if r["first_flip_median"] is not None], default=None),
        "flip_margin_p50": mean([r["flip_margin_p50"] for r in rs
                                 if r["flip_margin_p50"] is not None]) if any(
            r["flip_margin_p50"] is not None for r in rs) else None,
        "flip_margin_max": max([r["flip_margin_max"] for r in rs
                                if r["flip_margin_max"] is not None], default=None),
        "rel_frob_mean": mean(r["rel_frob_mean"] for r in rs),
    }


def get_b(results, cfg, mode, alpha):
    rs = [r for r in results if r.get("phase") == "b_decode_vs_decode" and r["cfg"] == cfg
          and r["mode"] == mode and abs(r["alpha"] - alpha) < 1e-9]
    return rs[0] if rs else None


def main() -> int:
    out = json.loads(RES.read_text())
    results, meta = out["results"], out["meta"]

    # --- method (a) aggregates ---
    a_anchor = agg_a(results, "both51", "anchor", 0.0)
    a_req_both = agg_a(results, "both51", "requant_g32", 1.0)
    a_req_plig = agg_a(results, "plig40", "requant_g32", 1.0)
    a_req_qkv = agg_a(results, "qkv11", "requant_g32", 1.0)
    a_uni_both = agg_a(results, "both51", "uniform", 1.0)
    a_uni_plig = agg_a(results, "plig40", "uniform", 1.0)
    a_uni_qkv = agg_a(results, "qkv11", "uniform", 1.0)
    a_uni_half = agg_a(results, "both51", "uniform", 0.5)

    # --- method (b) served decode ---
    b_anchor = get_b(results, "both51", "anchor", 0.0)
    b_req = get_b(results, "both51", "requant_g32", 1.0)
    b_uni = get_b(results, "both51", "uniform", 1.0)

    # --- sanity floors: both must be 0 or the differential is contaminated ---
    sanity_a_ok = (a_anchor is not None and a_anchor["flips_max"] == 0)
    sanity_b_ok = (b_anchor is not None and b_anchor["prompts_diverged"] == 0)

    # --- VERDICT: strict-#319 = exact byte-identity. PRESERVED needs (1) clean floors,
    #     (2) the FAITHFUL-magnitude proxy clean, (3) the CONCRETE realizable proxy clean,
    #     and (4) the served-decode test clean. Any nonzero -> BROKEN. ---
    faithful_clean = (a_uni_both is not None and a_uni_both["flips_max"] == 0)
    concrete_clean = (a_req_both is not None and a_req_both["flips_max"] == 0)
    served_clean = (b_req is not None and b_req["prompts_diverged"] == 0
                    and b_uni is not None and b_uni["prompts_diverged"] == 0)
    floors_ok = sanity_a_ok and sanity_b_ok
    preserved = 1 if (floors_ok and faithful_clean and concrete_clean and served_clean) else 0
    test_metric = a_uni_both["flip_rate_mean"] if a_uni_both else None

    # both-halves ablation (only meaningful if something flips)
    toxic_plig = bool(a_uni_plig and a_uni_plig["flips_max"] > 0)
    toxic_qkv = bool(a_uni_qkv and a_uni_qkv["flips_max"] > 0)
    if toxic_plig and toxic_qkv:
        toxic = "both"
    elif toxic_plig:
        toxic = "plig"
    elif toxic_qkv:
        toxic = "qkv"
    else:
        toxic = "neither(clean)"

    summary = {
        "MIXED_GRID_IDENTITY_PRESERVED": preserved,
        "primary_metric": preserved,
        "test_metric": test_metric,                      # faithful per-position argmax flip rate
        # sanity floors (must be 0)
        "sanity_a_anchor_flips": a_anchor["flips_max"] if a_anchor else None,
        "sanity_b_anchor_prompts_diverged": b_anchor["prompts_diverged"] if b_anchor else None,
        "floors_ok": int(floors_ok),
        # faithful magnitude (uniform a=1) -- the headline
        "faithful_both_flip_rate_mean": a_uni_both["flip_rate_mean"] if a_uni_both else None,
        "faithful_both_flip_rate_max": a_uni_both["flip_rate_max"] if a_uni_both else None,
        "faithful_both_flips_max": a_uni_both["flips_max"] if a_uni_both else None,
        "faithful_both_confident_flips_max": a_uni_both["confident_flips_max"] if a_uni_both else None,
        "faithful_both_confident_rate_mean": a_uni_both["confident_rate_mean"] if a_uni_both else None,
        "faithful_both_prompts_diverged_max": a_uni_both["prompts_diverged_max"] if a_uni_both else None,
        "faithful_both_first_flip_min": a_uni_both["first_flip_min"] if a_uni_both else None,
        "faithful_both_flip_margin_max": a_uni_both["flip_margin_max"] if a_uni_both else None,
        "faithful_both_rel_frob": a_uni_both["rel_frob_mean"] if a_uni_both else None,
        "faithful_both_n_seeds": a_uni_both["n_seeds"] if a_uni_both else None,
        "faithful_clean": int(faithful_clean),
        # half ablation (faithful a=1)
        "faithful_plig_flip_rate_mean": a_uni_plig["flip_rate_mean"] if a_uni_plig else None,
        "faithful_plig_flips_max": a_uni_plig["flips_max"] if a_uni_plig else None,
        "faithful_qkv_flip_rate_mean": a_uni_qkv["flip_rate_mean"] if a_uni_qkv else None,
        "faithful_qkv_flips_max": a_uni_qkv["flips_max"] if a_uni_qkv else None,
        "identity_toxic_half": toxic,
        # lower bracket (uniform a=0.5)
        "halfdose_both_flip_rate_mean": a_uni_half["flip_rate_mean"] if a_uni_half else None,
        "halfdose_both_flips_max": a_uni_half["flips_max"] if a_uni_half else None,
        # concrete realizable (requant_g32) -- understates true delta, still a lower bound
        "concrete_both_flip_rate": a_req_both["flip_rate_mean"] if a_req_both else None,
        "concrete_both_flips_max": a_req_both["flips_max"] if a_req_both else None,
        "concrete_both_confident_flips": a_req_both["confident_flips_max"] if a_req_both else None,
        "concrete_both_rel_frob": a_req_both["rel_frob_mean"] if a_req_both else None,
        "concrete_plig_flips_max": a_req_plig["flips_max"] if a_req_plig else None,
        "concrete_qkv_flips_max": a_req_qkv["flips_max"] if a_req_qkv else None,
        "concrete_clean": int(concrete_clean),
        # method (b) served decode divergence (the launch-faithful test)
        "served_concrete_prompts_diverged": b_req["prompts_diverged"] if b_req else None,
        "served_concrete_prompts_total": b_req["prompts_total"] if b_req else None,
        "served_concrete_first_div_median": b_req["first_div_median"] if b_req else None,
        "served_concrete_first_div_min": b_req["first_div_min"] if b_req else None,
        "served_faithful_prompts_diverged": b_uni["prompts_diverged"] if b_uni else None,
        "served_faithful_prompts_total": b_uni["prompts_total"] if b_uni else None,
        "served_faithful_first_div_median": b_uni["first_div_median"] if b_uni else None,
        "served_faithful_first_div_min": b_uni["first_div_min"] if b_uni else None,
        "served_clean": int(served_clean),
        # meta
        "n_prompts": meta["num_prompts"], "cont_len": meta["cont"],
        "margin_neartie_logit": meta["margin_neartie_logit"],
        "plig_modules": meta["plig_modules"], "qkv_modules": meta["qkv_modules"],
        "both_modules": meta["both_modules"],
        "n_cont_positions_total": a_uni_both["n_cont_positions"] if a_uni_both else None,
        # served-isolation guards
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "no_served_file_change": True,
    }

    run = init_wandb_run(
        job_type="validity-analysis", agent="land",
        name="land/mixed-grid-identity-712", group="mixed-grid-identity-land",
        tags=["mixed-grid", "identity", "319", "route-b", "g32", "analysis-only", "local-a10g", "v2"],
        notes=("PR #712 v2: does the realizable Route-B recovery (40 PLIG-g32 + 11 servable whole-qkv-g32 "
               "attn modules) preserve strict-#319 byte-exact greedy identity vs the locked g128 anchor? "
               "Corrected reference: routeB teacher-forced argmax vs ANCHOR teacher-forced argmax (method a, "
               "anchor floor=0) + routeB served greedy decode vs anchor served decode (method b). Faithful "
               "U(-s/2,s/2) g128-residual injection (alpha=1) + structured requant_g32 concrete lower-bracket; "
               "HF-vs-Marlin kernel cancels in the differential (PR #680 GEMM width-invariance)."),
        config={k: summary[k] for k in (
            "n_prompts", "cont_len", "margin_neartie_logit", "plig_modules", "qkv_modules",
            "both_modules", "analysis_only", "official_tps", "no_hf_job", "fires",
            "no_served_file_change")},
    )
    if run is not None:
        # dose-response curve on both51 (anchor, requant, a=0.5, a=1)
        curve = [
            ("anchor", 0.0, a_anchor), ("requant_g32", 1.0, a_req_both),
            ("uniform", 0.5, a_uni_half), ("uniform", 1.0, a_uni_both),
        ]
        for i, (mode, alpha, g) in enumerate(curve):
            if g:
                run.log({"global_step": i, "dose_mode_alpha": f"{mode}@{alpha}",
                         "both51_flip_rate_mean": g["flip_rate_mean"],
                         "both51_flips_max": g["flips_max"],
                         "both51_confident_flips_max": g["confident_flips_max"],
                         "both51_rel_frob": g["rel_frob_mean"]})
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="mixed-grid-identity-712-v2",
                          artifact_type="identity-report", data=out)
        rid = getattr(run, "id", None)
        finish_wandb(run)
    else:
        rid = None
        print("[log] wandb init returned None (no API key / disabled)", flush=True)

    out["verdict"] = summary
    out["wandb_run_id"] = rid
    RES.write_text(json.dumps(out, indent=2))

    # ---------------- paste-ready PR summary ----------------
    verdict_str = "PRESERVED" if preserved else "BROKEN"
    print("\n================ PR #712 v2 VERDICT ================", flush=True)
    print(f"MIXED_GRID_IDENTITY = {verdict_str}  (PRESERVED={preserved}; 1=preserved, 0=broken)")
    print(f"wandb_run_id = {rid}")
    print(f"[floors] method-a anchor-vs-anchor flips = {summary['sanity_a_anchor_flips']} (must be 0)")
    print(f"[floors] method-b anchor determinism prompts_diverged = "
          f"{summary['sanity_b_anchor_prompts_diverged']} (must be 0)")
    if a_uni_both:
        print(f"\n[FAITHFUL uniform a=1 both51] over {a_uni_both['n_seeds']} seeds:")
        print(f"   flip_rate_mean={a_uni_both['flip_rate_mean']:.3e} flips_max={a_uni_both['flips_max']} "
              f"/{a_uni_both['n_cont_positions']} pos")
        print(f"   confident_flips_max={a_uni_both['confident_flips_max']} "
              f"(margin>{meta['margin_neartie_logit']} logit) "
              f"prompts_diverged_max={a_uni_both['prompts_diverged_max']}/{a_uni_both['prompts_total']}")
        print(f"   first_flip_min={a_uni_both['first_flip_min']} "
              f"flip_margin_max={a_uni_both['flip_margin_max']} rel_frob={a_uni_both['rel_frob_mean']:.4f}")
    if a_uni_plig and a_uni_qkv:
        print(f"   half-ablation: plig40 flips_max={a_uni_plig['flips_max']} "
              f"({a_uni_plig['flip_rate_mean']:.3e})  qkv11 flips_max={a_uni_qkv['flips_max']} "
              f"({a_uni_qkv['flip_rate_mean']:.3e})  toxic_half={toxic}")
    if a_uni_half:
        print(f"   half-dose a=0.5 both51: flips_max={a_uni_half['flips_max']} "
              f"({a_uni_half['flip_rate_mean']:.3e})")
    if a_req_both:
        print(f"\n[CONCRETE requant_g32 both51] (understates true delta): "
              f"flips_max={a_req_both['flips_max']} ({a_req_both['flip_rate_mean']:.3e}) "
              f"confident={a_req_both['confident_flips_max']} rel_frob={a_req_both['rel_frob_mean']:.4f}")
        if a_req_plig and a_req_qkv:
            print(f"   half: plig40={a_req_plig['flips_max']} qkv11={a_req_qkv['flips_max']}")
    if b_req:
        print(f"\n[SERVED method-b decode-vs-decode]:")
        print(f"   requant_g32: prompts_diverged={b_req['prompts_diverged']}/{b_req['prompts_total']} "
              f"first_div_median={b_req['first_div_median']} first_div_min={b_req['first_div_min']}")
    if b_uni:
        print(f"   uniform a=1: prompts_diverged={b_uni['prompts_diverged']}/{b_uni['prompts_total']} "
              f"first_div_median={b_uni['first_div_median']} first_div_min={b_uni['first_div_min']}")
    print("===================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
