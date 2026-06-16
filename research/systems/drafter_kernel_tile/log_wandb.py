"""Log the drafter-kernel-tile-profile (PR #449) artifacts to W&B.

Run UNDER THE REPO .venv (serve venv has no wandb) and as a FILE in this dir so
sys.path[0] is this (wandb-shadow-free) directory:

  cd <repo_root>
  WANDB_API_KEY=... .venv/bin/python research/systems/drafter_kernel_tile/log_wandb.py

ROOT is appended (not prepended) to sys.path so the installed `wandb` package wins
over the repo-root ./wandb run-output dir. group=kernel-tiling-sweep per PR #449.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.append(str(ROOT))  # END of path: installed wandb pkg beats repo-root ./wandb dir

from scripts.wandb_logging import init_wandb_run  # noqa: E402


def _latest(pat: str):
    fs = sorted(glob.glob(str(HERE / pat)))
    return fs[-1] if fs else None


def main() -> int:
    report = json.loads((HERE / "report.json").read_text())
    pr_f = _latest("precise-*.json")
    mb_f = _latest("microbench-*.json")
    bd_f = report["inputs"]["breakdown"]
    precise = json.loads(open(pr_f).read()) if pr_f else {}
    micro = json.loads(open(mb_f).read()) if mb_f else {}
    breakdown = json.loads(open(bd_f).read()) if bd_f and os.path.exists(bd_f) else {}

    hm = report["honest_mapping"]
    bd = report["breakdown"]
    sw = report["tile_sweep"]
    st = report["self_test"]

    config = {
        "pr": 449,
        "track": "drafter-kernel-tile-profile",
        "local_only": True,
        "served_file_change": False,
        "device": report["device"],
        "onegraph": 1,
        "drafter_specific_triton_kernels": report["drafter_specific_triton_kernels"],
        "dims": report["dims"],
        "served_default_tile": {"BLOCK_SELECTED": 16, "num_warps": 8, "num_stages": None},
        "sweep_grid": sw["grid"],
        "n_configs_benched": sw["n_configs_benched"],
        "anchors": {
            "D_us": bd["D_us"], "V_us": bd["V_us"], "cycle_us": bd["D_us"] + bd["V_us"],
            "local_anchor_tps": 465.14, "official_anchor_tps": 481.53,
            "tau_lo_local_to_official": 1.03524,
            "ppl_anchor": report["test_metric"]["value"], "ppl_gate": report["ppl_gate"],
        },
        "public_evidence": [
            "SlimSpec low-rank lm_head (arXiv:2605.10453) — larger drafter lever, needs retraining",
            "Triton autotune / do_bench (triton.testing) — tile sweep methodology",
        ],
    }

    run = init_wandb_run(
        job_type="kernel-profile",
        agent="lawine",
        name="lawine/drafter-kernel-tile-profile",
        group="kernel-tiling-sweep",
        notes="PR#449: Is the MTP K=7 drafter (D=1.433ms) tile-optimal on sm_86? "
              "Local-only profiling + Triton tile microbench. No served change.",
        tags=["pr-449", "drafter", "kernel-tiling", "speculative-decoding", "local-only",
              "byte-exact", "no-go"],
        config=config,
    )
    if run is None:
        print("ERROR: init_wandb_run returned None (WANDB disabled or no API key)", file=sys.stderr)
        return 1

    # ---- headline scalars (logged at step 0) ----
    scalars = {
        "primary/max_honest_endtoend_tps_delta": report["primary_metric"]["value"],
        "primary/ceiling_if_argmax_free_tps": hm["ceiling_if_argmax_free_tps"],
        "primary/ceiling_source_in_graph": 1.0,
        "test/ppl": report["test_metric"]["value"],
        "test/ppl_gate": report["ppl_gate"],
        "test/ppl_ok": float(st["ppl_ok"]),
        "sweep/best_speedup_vs_default": sw["best_speedup_vs_default"],
        "sweep/gain_kernel_us_per_call": sw["gain_kernel_us_per_call"],
        "sweep/served_default_full_us_precise": sw["served_default_us_per_call_precise"],
        "sweep/served_default_full_us_dobench": sw["served_default_us_per_call_dobench"],
        "sweep/n_configs_benched": sw["n_configs_benched"],
        "sweep/all_configs_byte_correct": float(sw["all_configs_byte_correct"]),
        "breakdown/D_us_measured": bd["drafter_gpu_ms_measured"] * 1000.0,
        "breakdown/sparse_argmax_us_per_decode_step": bd["sparse_argmax_us_per_decode_step"],
        "breakdown/sparse_argmax_us_per_call_in_graph": bd["sparse_argmax_us_per_call"],
        "breakdown/sparse_argmax_pct_of_D": bd["sparse_argmax_pct_of_D"],
        "honest/delta_official_tps": hm["delta_official_tps"],
        "honest/delta_local_tps": hm["delta_local_tps"],
        "honest/frac_of_cycle": hm["frac_of_cycle"],
        "selftest/passes": float(report["self_test_passes"]),
        "global_step": 0,
    }
    for k, v in st.items():
        scalars[f"selftest/{k}"] = float(bool(v))
    for k, v in bd["category_pct"].items():
        scalars[f"category_pct/{k}"] = v
    per_step = breakdown.get("sparse_argmax_per_step_by_name", {})
    for k, v in per_step.items():
        scalars[f"sparse_per_step_us/{k}"] = v
    run.log(scalars)

    # ---- tile-sweep table (precise sub-us batched timer is authoritative) ----
    import wandb
    rows = precise.get("rows", [])
    sweep_tbl = wandb.Table(
        columns=["name", "block_arg", "num_warps", "num_stages",
                 "full_us", "full_min_us", "blocks_us", "reduce_us", "x_vs_default", "correct"])
    for r in rows:
        sweep_tbl.add_data(r.get("name"), r.get("block_arg"), r.get("num_warps"),
                           r.get("num_stages"), r.get("full_us"), r.get("full_min_us"),
                           r.get("blocks_us"), r.get("reduce_us"), r.get("x_vs_default"),
                           r.get("correct"))
    # do_bench microbench (all 45 configs, correctness)
    mb_rows = micro.get("results", [])
    micro_tbl = wandb.Table(columns=["block_arg", "num_warps", "num_stages", "us", "correct"])
    for r in mb_rows:
        micro_tbl.add_data(r.get("block_arg"), r.get("num_warps"), r.get("num_stages"),
                           r.get("us"), r.get("correct"))
    # category breakdown table
    cat_tbl = wandb.Table(columns=["category", "pct_of_window_gpu", "us_window"])
    cat_us = breakdown.get("category_us", {})
    for k, v in bd["category_pct"].items():
        cat_tbl.add_data(k, v, cat_us.get(k))
    run.log({"tile_sweep_precise": sweep_tbl,
             "tile_sweep_dobench45": micro_tbl,
             "kernel_category_breakdown": cat_tbl,
             "global_step": 0})

    # ---- summary headline (sticky) ----
    run.summary["max_honest_endtoend_tps_delta"] = report["primary_metric"]["value"]
    run.summary["ceiling_if_argmax_free_tps"] = hm["ceiling_if_argmax_free_tps"]
    run.summary["ppl"] = report["test_metric"]["value"]
    run.summary["self_test_passes"] = report["self_test_passes"]
    run.summary["sparse_argmax_pct_of_D"] = bd["sparse_argmax_pct_of_D"]
    run.summary["best_config"] = sw["best_config"]["name"]
    run.summary["verdict"] = report["verdict"]
    run.summary["go_no_go"] = "NO-GO"

    # ---- artifact: full report + raw inputs ----
    art = wandb.Artifact("drafter_kernel_tile_profile", type="profiling-report",
                         metadata={"pr": 449,
                                   "primary_metric": report["primary_metric"],
                                   "self_test_passes": report["self_test_passes"]})
    art.add_file(str(HERE / "report.json"))
    if pr_f:
        art.add_file(pr_f)
    if mb_f:
        art.add_file(mb_f)
    if bd_f and os.path.exists(bd_f):
        art.add_file(bd_f)
    run.log_artifact(art)

    rid = run.id
    rpath = f"{run.entity}/{run.project}/{rid}"
    print(json.dumps({"run_id": rid, "run_path": rpath, "url": run.get_url()}, indent=2))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
