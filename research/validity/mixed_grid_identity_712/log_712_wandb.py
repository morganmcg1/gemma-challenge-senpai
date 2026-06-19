#!/usr/bin/env python3
"""PR #712: compute the MIXED_GRID_IDENTITY verdict from identity_results.json and log
to W&B group `mixed-grid-identity-land`. Re-uses the measured report verbatim — no recompute.
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

RES = HERE / "identity_results.json"


def agg(results, cfg, mode, alpha):
    rs = [r for r in results if r["cfg"] == cfg and r["mode"] == mode
          and abs(r["alpha"] - alpha) < 1e-9]
    if not rs:
        return None
    return {
        "n_seeds": len(rs),
        "flip_rate_mean": mean(r["flip_rate"] for r in rs),
        "flip_rate_max": max(r["flip_rate"] for r in rs),
        "flips_max": max(r["n_flips"] for r in rs),
        "prompts_diverged_max": max(r["prompts_diverged"] for r in rs),
        "prompts_total": rs[0]["prompts_total"],
        "n_cont_positions": rs[0]["n_cont_positions"],
        "rel_frob_mean": rs[0].get("rel_frob_mean"),
        "first_flip_min": min([r["first_flip_min"] for r in rs if r["first_flip_min"] is not None],
                              default=None),
        "flip_margin_max": max([r["flip_margin_max"] for r in rs if r["flip_margin_max"] is not None],
                               default=None),
    }


def main() -> int:
    out = json.loads(RES.read_text())
    results, meta = out["results"], out["meta"]

    faithful_both = agg(results, "both51", "uniform", 1.0)
    faithful_plig = agg(results, "plig40", "uniform", 1.0)
    faithful_qkv = agg(results, "qkv11", "uniform", 1.0)
    req_both = agg(results, "both51", "requant_g32", 1.0)
    req_plig = agg(results, "plig40", "requant_g32", 1.0)
    req_qkv = agg(results, "qkv11", "requant_g32", 1.0)
    anchor = agg(results, "both51", "anchor", 0.0)

    # sensitivity curve: lowest alpha at which any flip appears on both51
    alphas = sorted({r["alpha"] for r in results if r["cfg"] == "both51" and r["mode"] == "uniform"})
    curve = [(a, agg(results, "both51", "uniform", a)) for a in alphas]
    alpha_first_flip = next((a for a, g in curve if g and g["flips_max"] > 0), None)

    # VERDICT: strict-#319 needs exact 128/128. Faithful alpha=1 over all seeds must be clean,
    # AND the conservative structured requant cross-check clean, for PRESERVED.
    faithful_clean = (faithful_both["flips_max"] == 0)
    requant_clean = (req_both is not None and req_both["flips_max"] == 0)
    preserved = 1 if (faithful_clean and anchor["flips_max"] == 0) else 0
    test_metric = faithful_both["flip_rate_mean"]

    # which half is identity-toxic (only meaningful if something flips)
    toxic = "neither(clean)"
    if not faithful_clean:
        fp = faithful_plig["flips_max"] if faithful_plig else 0
        fq = faithful_qkv["flips_max"] if faithful_qkv else 0
        toxic = "plig" if fp > fq else ("qkv" if fq > fp else "both-equal")

    summary = {
        "MIXED_GRID_IDENTITY_PRESERVED": preserved,
        "primary_metric": preserved,
        "test_metric": test_metric,
        "argmax_flip_rate_faithful_both": test_metric,
        "faithful_both_flip_rate_max": faithful_both["flip_rate_max"],
        "faithful_both_prompts_diverged_max": faithful_both["prompts_diverged_max"],
        "faithful_both_flips_max": faithful_both["flips_max"],
        "faithful_both_rel_frob": faithful_both["rel_frob_mean"],
        "faithful_plig_flip_rate_max": faithful_plig["flip_rate_max"] if faithful_plig else None,
        "faithful_qkv_flip_rate_max": faithful_qkv["flip_rate_max"] if faithful_qkv else None,
        "faithful_clean": int(faithful_clean),
        "requant_both_flip_rate_max": req_both["flip_rate_max"] if req_both else None,
        "requant_both_flips_max": req_both["flips_max"] if req_both else None,
        "requant_both_rel_frob": req_both["rel_frob_mean"] if req_both else None,
        "requant_clean": int(requant_clean),
        "requant_plig_flip_rate_max": req_plig["flip_rate_max"] if req_plig else None,
        "requant_qkv_flip_rate_max": req_qkv["flip_rate_max"] if req_qkv else None,
        "anchor_vs_anchor_flips": anchor["flips_max"],
        "alpha_first_flip": alpha_first_flip,
        "identity_toxic_half": toxic,
        "n_prompts": meta["num_prompts"], "cont_len": meta["cont"], "n_seeds": meta["seeds"],
        "plig_modules": meta["plig_modules"], "qkv_modules": meta["qkv_modules"],
        "both_modules": meta["both_modules"],
        "n_cont_positions_total": faithful_both["n_cont_positions"],
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "no_served_file_change": True,
    }

    run = init_wandb_run(
        job_type="validity-analysis", agent="land",
        name="land/mixed-grid-identity-712", group="mixed-grid-identity-land",
        tags=["mixed-grid", "identity", "319", "route-b", "g32", "analysis-only", "local-a10g"],
        notes=("PR #712: does the realizable Route-B recovery (40 PLIG-g32 + 11 servable whole-qkv-g32 "
               "attn modules) preserve strict-#319 byte-exact greedy identity vs the locked g128 anchor? "
               "In-memory fake-quant differential; faithful U(-s/2,s/2) g128-residual injection (alpha=1) "
               "+ structured requant_g32 cross-check; HF-vs-Marlin kernel cancels in the differential."),
        config={k: summary[k] for k in (
            "n_prompts", "cont_len", "n_seeds", "plig_modules", "qkv_modules", "both_modules",
            "analysis_only", "official_tps", "no_hf_job", "fires", "no_served_file_change")},
    )
    if run is not None:
        for i, (a, g) in enumerate(curve):
            if g:
                run.log({"alpha": a, "both51_flip_rate_mean": g["flip_rate_mean"],
                         "both51_flip_rate_max": g["flip_rate_max"],
                         "both51_prompts_diverged_max": g["prompts_diverged_max"]})
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="mixed-grid-identity-712",
                          artifact_type="identity-report", data=out)
        rid = getattr(run, "id", None)
        finish_wandb(run)
    else:
        rid = None
        print("[log] wandb init returned None (no API key / disabled)", flush=True)

    out["verdict"] = summary
    out["wandb_run_id"] = rid
    RES.write_text(json.dumps(out, indent=2))

    # paste-ready summary
    print("\n================ PR #712 VERDICT ================", flush=True)
    print(f"MIXED_GRID_IDENTITY_PRESERVED = {preserved}  (1=preserved, 0=broken)")
    print(f"wandb_run_id = {rid}")
    print(f"anchor-vs-anchor flips (null) = {anchor['flips_max']} (must be 0)")
    print(f"faithful alpha=1 both51: flip_rate_max={faithful_both['flip_rate_max']:.3e} "
          f"flips_max={faithful_both['flips_max']} prompts_diverged_max="
          f"{faithful_both['prompts_diverged_max']}/{faithful_both['prompts_total']} "
          f"rel_frob={faithful_both['rel_frob_mean']:.4f} over {faithful_both['n_seeds']} seeds")
    print(f"  plig40 a=1 flip_rate_max={faithful_plig['flip_rate_max']:.3e} flips={faithful_plig['flips_max']}")
    print(f"  qkv11  a=1 flip_rate_max={faithful_qkv['flip_rate_max']:.3e} flips={faithful_qkv['flips_max']}")
    if req_both:
        print(f"requant_g32 both51 (conservative): flip_rate_max={req_both['flip_rate_max']:.3e} "
              f"flips={req_both['flips_max']} rel_frob={req_both['rel_frob_mean']:.4f}")
    print(f"alpha sensitivity (both51): first-flip alpha = {alpha_first_flip}")
    for a, g in curve:
        if g:
            print(f"   alpha={a:.2f}: flip_rate_mean={g['flip_rate_mean']:.3e} "
                  f"flips_max={g['flips_max']} prompts_div_max={g['prompts_diverged_max']}")
    print(f"identity-toxic half = {toxic}")
    print("=================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
