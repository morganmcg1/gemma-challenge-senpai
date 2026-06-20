"""Log the PR #809 Step-1 capture-scope audit to W&B.

Run UNDER THE REPO .venv (serve venv has a shadowed wandb) from the repo root:

  cd <repo_root>
  WANDB_API_KEY=... .venv/bin/python research/cudagraph_sampling_capture/log_wandb.py

ROOT is appended (not prepended) to sys.path so the installed `wandb` package wins
over the repo-root ./wandb run-output dir. group=cudagraph-sampling-capture per PR #809.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root: research/cudagraph_sampling_capture -> target/
sys.path.append(str(ROOT))  # END of path: installed wandb pkg beats repo-root ./wandb dir

from scripts.wandb_logging import flatten_numeric, init_wandb_run  # noqa: E402


def main() -> int:
    res = json.loads((HERE / "audit_result.json").read_text())
    ta = res.get("trace_analysis", {}) or {}

    # Step-1 verdict: the PR widens the cudagraph to remove a *serial per-token
    # sampling sync*. That sync only exists if async scheduling is OFF. It is ON,
    # so there is no serial sync to remove -> NULL, close the lane (per the PR's
    # own Step-1 stop condition + the #784 fast-negative culture).
    async_on = bool(res.get("async_scheduling"))
    busy_share = res.get("gpu_busy_share_of_wall_pct")
    verdict = "NULL-no-serial-sampling-sync" if async_on else "OPEN-sync-present"
    go_no_go = "NO-GO" if async_on else "INVESTIGATE"

    config = {
        "pr": 809,
        "track": "cudagraph-sampling-capture",
        "local_only": True,
        "served_file_change": False,
        "model": res.get("model"),
        "drafter": res.get("drafter"),
        "num_spec": res.get("num_spec"),
        "proxy_note": res.get("proxy_note"),
        "patches_applied": res.get("patches_applied"),
        "hypothesis": "Widen CUDA-graph capture to include greedy sampling and "
                      "remove a per-token host<->device sampling sync.",
        "finding": "vLLM 0.22.0 keeps the sampler OUTSIDE the model-forward graph "
                   "by design; the per-token sampling sync is hidden by ASYNC "
                   "SCHEDULING (auto-enabled because gemma4_mtp is an EAGLE-type "
                   "method), not by graph-widening. No serial sync to claw back.",
        "public_evidence": [
            "stark #798 decode profile (W&B dpc36210): non-GEMM bucket 3.28ms/12.42ms "
            "(26.4%) is GPU-kernel composition, NOT a serial host sync",
            "lawine #787 cudagraph-M7 audit: model forward incl. M=7 verify is 100% "
            "FULL cudagraph; sampler is a separate sample_tokens() step",
            "#784 fast-negative culture (post the null and close the lane)",
            "vLLM config/vllm.py:947-975 async_scheduling auto-enable for EAGLE/MTP",
        ],
    }

    run = init_wandb_run(
        job_type="capture-scope-audit",
        agent="lawine",
        name="lawine/cudagraph-sampling-capture-audit",
        group="cudagraph-sampling-capture",
        notes="PR#809 Step-1: is the MAIN-model greedy sampling inside the CUDA "
              "graph, or eager with a per-token device sync? Answer: sampler is "
              "outside the graph by design, but the sync is overlapped by async "
              "scheduling (enabled). No serial per-token sync -> NULL.",
        tags=["pr-809", "cudagraph", "sampling", "async-scheduling",
              "speculative-decoding", "local-only", "byte-exact", "null-result"],
        config=config,
    )
    if run is None:
        print("ERROR: init_wandb_run returned None (WANDB disabled or no API key)",
              file=sys.stderr)
        return 1

    scalars = {
        "audit/async_scheduling": float(async_on),
        "audit/spec_tps": res.get("spec_tps"),
        "audit/gpu_busy_share_of_wall_pct": busy_share,
        "audit/gpu_busy_per_token_ms": res.get("gpu_busy_per_token_ms"),
        "audit/sampling_pct_of_gpu_busy": res.get("sampling_pct_of_gpu_busy"),
        "audit/serial_overhead_pct": (100.0 - busy_share) if busy_share is not None else None,
        "audit/engine_load_s": res.get("engine_load_s"),
        "audit/profile_tokens": res.get("profile_tokens"),
        "trace/num_streams_with_kernels": ta.get("num_streams_with_kernels"),
        "trace/compute_stream_busy_share_of_span_pct": ta.get("compute_stream_busy_share_of_span_pct"),
        "trace/dtoh_memcpy_count_total": ta.get("dtoh_memcpy_count_total"),
        "trace/dtoh_us_off_compute_stream": ta.get("dtoh_us_off_compute_stream"),
        "trace/dtoh_us_on_compute_stream": ta.get("dtoh_us_on_compute_stream"),
        "trace/dtoh_overlap_capable": float(bool(ta.get("dtoh_overlap_capable"))),
        "trace/host_synchronize_calls_in_window": ta.get("host_synchronize_calls_in_window"),
        "global_step": 0,
    }
    scalars.update(flatten_numeric("category_pct", res.get("category_pct", {})))
    scalars.update(flatten_numeric("category_ms", res.get("category_ms", {})))
    scalars = {k: v for k, v in scalars.items() if v is not None}
    run.log(scalars)

    import wandb
    # top-kernel table
    kt = wandb.Table(columns=["pct", "ms", "count", "category", "kernel"])
    for r in res.get("top_kernels", []):
        kt.add_data(r.get("pct"), r.get("ms"), r.get("count"), r.get("category"),
                    r.get("kernel"))
    run.log({"top_kernels": kt, "global_step": 0})

    run.summary["async_scheduling"] = async_on
    run.summary["gpu_busy_share_of_wall_pct"] = busy_share
    run.summary["sampling_pct_of_gpu_busy"] = res.get("sampling_pct_of_gpu_busy")
    run.summary["dtoh_overlap_capable"] = bool(ta.get("dtoh_overlap_capable"))
    run.summary["spec_tps"] = res.get("spec_tps")
    run.summary["verdict"] = verdict
    run.summary["go_no_go"] = go_no_go

    art = wandb.Artifact("cudagraph_sampling_capture_audit", type="profiling-report",
                         metadata={"pr": 809, "async_scheduling": async_on,
                                   "verdict": verdict})
    art.add_file(str(HERE / "audit_result.json"))
    log = HERE / "audit.log"
    if log.exists():
        art.add_file(str(log))
    tp = res.get("trace_path")
    if tp and Path(tp).exists():
        art.add_file(tp)
    run.log_artifact(art)

    rid = run.id
    rpath = f"{run.entity}/{run.project}/{rid}"
    print(json.dumps({"run_id": rid, "run_path": rpath, "url": run.get_url(),
                      "verdict": verdict}, indent=2))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
