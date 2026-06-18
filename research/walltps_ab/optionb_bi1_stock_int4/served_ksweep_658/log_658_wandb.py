"""PR #658 K-sweep closeout -> W&B group `served-ksweep-walltps-land`.

Logs the rescued served wall-TPS de-projection across K={5,6,7} (anchor K=6,
context K=3,4): the per-K curve (rescued_local headline + un-rescued base +
e_accept + fire-rate + #319 identity), the decision (K*, verdict), and the
K=6 cross-validation vs stark #642.

Reads only the local analysis JSON (deproject_rescued_ksweep.json); no server,
no GPU. analysis_only=true, official_tps=0. Run the analyzer first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

PPL_UNCHANGED = 2.0055  # #632/#623 K-sweep PPL (spec lane; identity-preserving, unchanged by K)


def main() -> int:
    data = json.loads((HERE / "deproject_rescued_ksweep.json").read_text())
    rows = data["rows"]
    dec = data["decision"]
    xval = data["crossval_stark_642"]

    run = wandb_logging.init_wandb_run(
        job_type="served_ksweep_walltps",
        agent="land",
        name="land/served-ksweep-walltps-deproject",
        group="served-ksweep-walltps-land",
        notes=("PR#658: rescued (recompute-acceptor ON) served wall-TPS K-sweep. "
               "De-projects rescued_wall_tps(K)=1/(1/U(K)+f(K)/A) from land #632 un-rescued "
               "U(K), #648 fire-rate f(K), #651/#654 identity. Does a K!=6 beat shipped K=6 "
               "while holding #319? Cross-validates stark #642's K=6 de-projection."),
        config={
            "pr": 658, "analysis_only": True, "official_tps": 0,
            "deprojection_formula": data["deprojection_formula"],
            "A_local_ar_rung_tps": data["A_local_ar_rung_tps"],
            "locked_rung_official": data["locked_rung_official"],
            "shipped_K": data["shipped_K"], "sweep_Ks": data["sweep_Ks"],
            "vllm": "0.22.0", "batch_invariant": 1, "max_num_seqs": 1, "greedy": True,
            "num_prompts": 128, "output_len": 512, "seed": 1,
            "reused_runs": {"unrescued_632": "uo6netrr(k5)/obfvs9ma(k6)/8sfauo3i(k7)",
                            "fire_census_648": "dyseni93", "identity_654": "ah3fe0h1"},
        },
        tags=["optionb", "batch_invariant", "pr658", "k_sweep", "rescued", "deproject", "served"],
    )
    if run is None:
        print("WANDB disabled/unavailable; dumping summary only", flush=True)

    import wandb

    cols = ["K", "unrescued_wall_tps_local", "e_accept_mean", "fire_rate_tau0p5",
            "rescued_local", "rescued_starkmix_official_fire", "on_AR_head_break_rate",
            "confident_off_AR_head_misses", "identity_holds"]
    if run is not None:
        tbl = wandb.Table(columns=cols)
        for r in rows:
            tbl.add_data(*[r.get(c) for c in cols])
            run.log({
                "global_step": r["K"], "curve/K": r["K"],
                "curve/rescued_local": r["rescued_local"],
                "curve/rescued_starkmix": r["rescued_starkmix_official_fire"],
                "curve/unrescued_wall_tps_local": r["unrescued_wall_tps_local"],
                "curve/e_accept_mean": r["e_accept_mean"],
                "curve/fire_rate_tau0p5": r["fire_rate_tau0p5"],
                "curve/on_AR_head_break_rate": r["on_AR_head_break_rate"],
            })
        run.log({
            "plot/rescued_local_vs_K": wandb.plot.line(
                tbl, "K", "rescued_local", title="PR#658 rescued served wall_tps vs K (LOCAL)"),
            "plot/unrescued_vs_K": wandb.plot.line(
                tbl, "K", "unrescued_wall_tps_local", title="un-rescued wall_tps vs K"),
            "plot/e_accept_vs_K": wandb.plot.line(
                tbl, "K", "e_accept_mean", title="mean accepted length vs K"),
            "ksweep_rescued_curve": tbl,
        })

    summary = {
        "decision/verdict": dec["verdict"],
        "decision/k_star": dec["k_star"],
        "decision/served_walltps_best_K_local": dec["served_walltps_best_K_local"],
        "decision/served_walltps_best_K_which": dec["served_walltps_best_K_which"],
        "decision/k6_rescued_local": dec["k6_rescued_local"],
        "decision/kstar_vs_k6_local_tps": dec["kstar_vs_k6_local_tps"],
        "decision/kstar_vs_k6_local_pct": dec["kstar_vs_k6_local_pct"],
        "decision/all_sweep_Ks_hold_identity": int(dec["all_sweep_Ks_hold_identity"]),
        "config/A_local_ar_rung_tps": data["A_local_ar_rung_tps"],
        "config/locked_rung_official": data["locked_rung_official"],
        "config/ppl_unchanged": PPL_UNCHANGED,
        # cross-validation vs stark #642
        "xval/ar_ref_local_land": xval["ar_ref_local"]["land"],
        "xval/ar_ref_local_stark": xval["ar_ref_local"]["stark_arm_d"],
        "xval/ar_ref_abs_pct_gap": xval["ar_ref_local"]["abs_pct_gap"],
        "xval/unrescued_k6_land": xval["unrescued_k6_local"]["land_632"],
        "xval/unrescued_k6_stark": xval["unrescued_k6_local"]["stark_642"],
        "xval/unrescued_k6_abs_pct_gap": xval["unrescued_k6_local"]["abs_pct_gap"],
        "xval/rescued_k6_starkmix_land": xval["rescued_k6_starkmix"]["land_deproject"],
    }
    # per-K flattened for quick reading in the run summary
    for r in rows:
        K = r["K"]
        summary[f"perK/K{K}_rescued_local"] = r["rescued_local"]
        summary[f"perK/K{K}_unrescued"] = r["unrescued_wall_tps_local"]
        summary[f"perK/K{K}_e_accept"] = r["e_accept_mean"]
        summary[f"perK/K{K}_fire_rate"] = r["fire_rate_tau0p5"]
        summary[f"perK/K{K}_on_AR_head_break_rate"] = r["on_AR_head_break_rate"]
        summary[f"perK/K{K}_confident_misses"] = r["confident_off_AR_head_misses"]

    if run is not None:
        wandb_logging.log_summary(run, summary, step=int(dec["k_star"]))
        wandb_logging.log_json_artifact(
            run, name="served_ksweep_658_deproject", artifact_type="analysis", data=data)
        url = getattr(run, "url", "")
        rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] served-ksweep-658 id={rid} url={url}", flush=True)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
