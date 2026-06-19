"""PR #746 route-b setup/heartbeat run — clears the dark-pod flag and records the
plan/config for the strict-clean-routeb-m1verify K-sweep. Local-only, analysis_only,
official_tps=0 (NO HF job). Subsequent per-K measurement runs log to the same group.
"""
from __future__ import annotations

import sys
from pathlib import Path

import wandb  # import first to win over any ./wandb shadow dir  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402

BASELINE_OFFICIAL_TPS = 126.378  # int4_g128_lmhead AR rung (PR #601)
TAU = 1.03524  # local wall_tps -> official scalar (#267)


def main() -> int:
    run = wandb_logging.init_wandb_run(
        job_type="analysis",
        agent="stark",
        name="stark/routeb-m1verify-setup",
        group="strict-clean-routeb-m1verify",
        notes="PR #746 route-b setup heartbeat: strict byte-exact spec (K-seq M=1 verify) "
              "net-TPS K-sweep vs 126.378. Local A10G, analysis_only, official_tps=0, NO HF job.",
        tags=["pr746", "route-b", "m1verify", "strict-byteexact", "analysis-only", "setup"],
        config={
            "pr": 746,
            "base_submission": "int4_mtp_batchinv",
            "baseline_official_tps": BASELINE_OFFICIAL_TPS,
            "tau_local_to_official": TAU,
            "official_tps": 0,  # projected only; local A10G, no HF job
            "analysis_only": True,
            "k_sweep": [2, 3, 4, 5, 6],
            "predicate": "warm steady greedy, MAX_NUM_SEQS=1, temp=0, single-stream, "
                         "strict byte-exact (zero-tol) vs served M=1 AR ref",
            "hypothesis": "route-b removes batched-verify amortization -> net_TPS ~ AR_target - "
                          "drafter_overhead; risk: dominated by the byte-exact M=1 AR ceiling "
                          "(genuine measured ceiling = int4_g128_lmhead AR ~=126.4 local ~= bar; "
                          "NOT the modeled-official 161.70 of the approximated fa2sw stack).",
        },
    )
    if run is None:
        print("[wandb] disabled / no API key — nothing logged", flush=True)
        return 1
    wandb.log({"global_step": 0, "heartbeat": 1, "phase": 0})
    run.summary["status"] = "setup-pod-alive"
    run.summary["baseline_official_tps"] = BASELINE_OFFICIAL_TPS
    print(f"[wandb] route-b setup run live: {run.id} ({run.url})", flush=True)
    wandb_logging.finish_wandb(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
