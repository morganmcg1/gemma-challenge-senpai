"""Log the PR #675 int4 GEMV kernel-selection audit to W&B as ONE rich run.

Pure replay over research/int4_gemv_kernel_audit/results.json (produced by
analyze.py) — decoupled from the GPU sweep so a W&B hiccup never costs a re-run.

ANALYSIS-ONLY: the no-fire guard is logged as explicit summary scalars
``analysis_only=1`` and ``official_tps=0`` (PR #675). Every arm carries its
wall_tps + break_rate + active kernel; the verdict, the byte-identical anchor,
the best byte-identical kernel wall_tps, and the dev307 break_rate noise floor
are headline summary scalars.

Run under the repo .venv (has wandb); import wandb FIRST so the real package wins
over the repo-root ./wandb run dir that would otherwise shadow the import.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import wandb  # noqa: F401  (import first to win over the ./wandb shadow dir)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="lawine/int4-gemv-kernel-audit")
    ap.add_argument("--group", default="int4-kernel-select-lawine")
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args(argv)

    res = json.loads(Path(args.results).read_text())
    arms = res["arms"]
    anchor = res.get("anchor_wall_tps_median_of_base")
    bwblk = res.get("bandwidth", {})
    bw = bwblk.get("per_candidate", {})
    v0220 = res.get("v0220_ship_control", {})

    run = wandb_logging.init_wandb_run(
        job_type="analysis",
        agent="lawine",
        name=args.name,
        group=args.group,
        notes="PR #675: fastest BYTE-IDENTICAL int4 GEMV kernel at strict M=1 AR greedy "
              "on shipped int4_g128_lmhead. ANALYSIS-ONLY, local. official_tps=0.",
        tags=["pr675", "int4_g128_lmhead", "gemv-kernel-select", "analysis-only"],
        config={
            "submission": "int4_g128_lmhead",
            "baseline_official_tps": res.get("baseline_official_tps"),
            "local_ar_anchor_wirbel665": res.get("local_ar_anchor_wirbel665"),
            "stark_tax_local_to_official": res.get("stark_tax_local_to_official"),
            "analysis_only": True,
            "official_tps": 0,
            "predicate": "M=1 AR greedy, spec-OFF, temp=0, official 128x512 sharegpt; "
                         "wall_tps = num_completion_tokens / duration_s",
            "vllm_version": "0.22.1rc1.dev307 (local serve venv 5f4c623f772358a2)",
            "attn_backend": "TRITON_ATTN (Gemma4 het head dims 256/512 force-pinned; fixed, not swept)",
            "active_kernel_all_layers": "MarlinLinearKernel (qkv/o_proj/gate_up/down_proj + int4 lm_head)",
            "machete_selected": False,  # Hopper sm_90 only; N/A on A10G sm_86
            "loadable_byteident_kernels_sm86": ["Marlin", "Humming(crash)", "Triton(no-M=1)"],
        },
    )
    if run is None:
        print("[wandb] disabled / no API key — printing summary only", flush=True)

    # ---- per-arm scalars + table ----
    cols = ["arm", "status", "active_kernel", "extra_env", "wall_tps",
            "achieved_gbps", "pct_read_peak",
            "partial_wall_tps_upper_bound", "delta_wall_tps_vs_anchor",
            "official_equiv_delta", "identity_verdict", "break_rate",
            "token_break_rate", "first_divergence_index", "reason"]
    tbl = wandb.Table(columns=cols) if run is not None else None
    for i, (name, a) in enumerate(arms.items()):
        idt = a.get("identity") or {}
        env = ",".join(a.get("extra_env") or []) or "(shipped serve)"
        bwa = bw.get(name, {})
        gbps = bwa.get("achieved_gbps") or bwa.get("achieved_gbps_upper_bound")
        row = {
            "arm": name, "status": a.get("status"),
            "active_kernel": a.get("active_kernel"), "extra_env": env,
            "wall_tps": a.get("wall_tps"),
            "achieved_gbps": gbps,
            "pct_read_peak": bwa.get("pct_read_peak") or bwa.get("pct_read_peak_upper_bound"),
            "partial_wall_tps_upper_bound": a.get("partial_wall_tps_upper_bound"),
            "delta_wall_tps_vs_anchor": a.get("delta_wall_tps_vs_anchor"),
            "official_equiv_delta": a.get("official_equiv_delta"),
            "identity_verdict": idt.get("verdict"),
            "break_rate": idt.get("break_rate"),
            "token_break_rate": idt.get("token_break_rate"),
            "first_divergence_index": idt.get("first_divergence_index"),
            "reason": a.get("reason"),
        }
        wt = a.get("wall_tps")
        print(f"[arm] {name:11s} {str(a.get('status')):14s} "
              f"{str(a.get('active_kernel')):22s} "
              f"wall_tps={wt if wt is not None else a.get('partial_wall_tps_upper_bound')} "
              f"break_rate={idt.get('break_rate')}", flush=True)
        if run is not None:
            log = {f"arm/{name}/break_rate": idt.get("break_rate"), "global_step": i}
            if wt is not None:
                log[f"arm/{name}/wall_tps"] = wt
                log[f"arm/{name}/beats_anchor"] = int(anchor is not None and wt > anchor)
            if gbps is not None:
                log[f"arm/{name}/achieved_gbps"] = gbps
            wandb.log(log)
            tbl.add_data(*[row[c] for c in cols])

    # ---- headline summary scalars ----
    if run is not None:
        wandb.log({"arms_table": tbl})
        s = run.summary
        s["analysis_only"] = 1                       # no-fire guard (machine-checkable)
        s["official_tps"] = 0                        # no-fire guard
        s["verdict"] = res.get("verdict")
        s["verdict_detail"] = res.get("verdict_detail")
        s["anchor_wall_tps_median_of_base"] = anchor
        s["n_base_reps"] = res.get("n_base_reps")
        s["best_byteident_kernel"] = res.get("best_byteident_kernel")
        s["best_byteident_kernel_walltps"] = res.get("best_byteident_kernel_walltps")
        s["break_rate_noise_floor"] = res.get("break_rate_noise_floor")
        s["n_disqualified"] = res.get("n_disqualified")
        s["mde_wall_tps"] = res.get("mde_wall_tps")
        s["baseline_official_tps"] = res.get("baseline_official_tps")
        # ---- bandwidth (denken #676 selection-side cross-check) ----
        wf = res.get("weight_footprint", {})
        s["W_GB_per_token"] = wf.get("W_GB_per_token")
        s["marlin_achieved_gbps"] = bwblk.get("marlin_median_gbps")
        s["marlin_pct_read_peak"] = bwblk.get("marlin_pct_read_peak")
        s["marlin_vs_triton_speedup"] = bwblk.get("marlin_vs_triton_speedup")
        s["implied_gemv_wall_share"] = bwblk.get("implied_gemv_wall_share")
        anch = bwblk.get("denken676_anchor", {})
        s["denken676_gemv_isolated_gbps"] = anch.get("gemv_isolated_gbps")
        s["denken676_read_peak_gbps"] = anch.get("read_peak_gbps")
        s["denken676_pct_read_peak"] = anch.get("pct_read_peak")
        # ---- v0220 ship-venv determinism control ----
        s["ship_self_break_rate"] = v0220.get("ship_self_break_rate")
        s["ship_env_deterministic"] = int(bool(v0220.get("ship_env_deterministic")))
        s["v0220_wall_tps_a"] = (v0220.get("wall_tps") or {}).get("v0220_a")
        s["v0220_wall_tps_b"] = (v0220.get("wall_tps") or {}).get("v0220_b")
        s["crossversion_dev307_vs_0220_break_rate"] = v0220.get("crossversion_dev307_vs_0220_break_rate")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] logged run: {run.id}", flush=True)
        print(f"[wandb] verdict={res.get('verdict')} "
              f"anchor={anchor} best_byteident={res.get('best_byteident_kernel_walltps')} "
              f"noise_floor={res.get('break_rate_noise_floor')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
