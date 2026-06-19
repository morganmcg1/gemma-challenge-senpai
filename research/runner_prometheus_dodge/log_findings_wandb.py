"""Log the runner-prometheus-dodge analysis findings to W&B.

Analysis-only (no HF job, no fire). Reads the fresh official-gate re-validation
evidence.json (guarded staging config) to derive ``official_gate_repass``, and
logs the consolidated dodge verdict to W&B group ``fire-runner-dodge-denken``.

Metrics (advisor card, PR #742):
  primary_metric = prometheus_dodge_found (1/0)
  test_metric    = official_gate_repass   (1/0)
  analysis_only=1, no_hf_job=1, fires=0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.wandb_logging import (  # noqa: E402
    finish_wandb,
    init_wandb_run,
    log_json_artifact,
    log_summary,
)

HERE = Path(__file__).resolve().parent
LOCALRUN = ROOT / "research" / "_localrun"


def latest_staging_evidence() -> dict:
    """Most-recent validate evidence for the guarded staging config."""
    cands = sorted(LOCALRUN.glob("validate-staging_guard_b-*/evidence.json"))
    if not cands:
        return {}
    return json.loads(cands[-1].read_text())


def main() -> int:
    findings = json.loads((HERE / "findings.json").read_text())
    ev = latest_staging_evidence()

    gate = ev.get("official_gate")
    official_gate_repass = 1 if gate == "PASS" else 0

    summary = {
        # advisor card metrics
        "prometheus_dodge_found": 1,
        "official_gate_repass": official_gate_repass,
        "analysis_only": 1,
        "no_hf_job": 1,
        "fires": 0,
        # dodge detail
        "path_a_manifest_pin_viable": 1,
        "path_b_sitecustomize_guard_viable": 1,
        "repro_runner_500_reproduced": 1,
        "guard_unit_all_pass": 1,
        "server_venv_submission_influenceable": 1,
        "fix_is_repo_own_http_compat_pin": 1,
        # re-validated gate components (guarded staging config)
        "revalidate_ppl": ev.get("ppl"),
        "revalidate_completed": ev.get("completed"),
        "revalidate_num_prompts": ev.get("num_prompts"),
        "revalidate_all_modalities_loaded": 1 if ev.get("all_modalities_loaded") else 0,
        "revalidate_official_gate_pass": official_gate_repass,
    }

    run = init_wandb_run(
        job_type="analysis",
        agent="senpai",
        name="denken/runner-prometheus-dodge-findings",
        group="fire-runner-dodge-denken",
        notes=(
            "Analysis-only: runner /v1/models 500 (prometheus _IncludedRouter .path) "
            "is submission-side dodgeable. Path A = manifest pin fastapi>=0.115,<0.116 "
            "(== repo's own harness.ensure_serving_http_compat fix, promoted to manifest "
            "so the runner inherits it). Path B = sitecustomize guard on "
            "prometheus_fastapi_instrumentator.routing._get_route_name. Both output-neutral. "
            "No re-fire (held for advisor go)."
        ),
        tags=["fire-runner-dodge", "analysis-only", "no-hf-job", "prometheus-dodge"],
        config={
            "pr": 742,
            "analysis_only": 1,
            "no_hf_job": 1,
            "fires": 0,
            "errored_job_id": "6a3558d73093dba73ce2a3e1",
            "root_cause": "fastapi>=0.118/starlette>=1 _IncludedRouter (no .path) + prometheus-fastapi-instrumentator route.path -> 500 on every request",
            "runner_prometheus_version": "8.0.0",
            "runner_fastapi_resolved": "0.137.2 (fresh-resolve of vllm 0.22.0 open bounds; reproduces exact error)",
            "local_fastapi_pinned": "0.115.14 (harness ensure_serving_http_compat auto-pin)",
            "path_a_patch": "add 'fastapi>=0.115,<0.116' to submissions/int4_mtp_batchinv/manifest.json dependencies",
            "path_b_patch": "wrap prometheus_fastapi_instrumentator.routing._get_route_name -> return None on AttributeError (sitecustomize.py)",
            "staging_evidence_dir": str(LOCALRUN),
            "revalidate_official_gate": gate,
            "prior_pass_wandb": "ox5qtjfk",
            "errored_launch_wandb": "rl4qh0y6",
        },
    )
    if run is None:
        print("WANDB_DISABLED or no creds — printing summary instead:")
        print(json.dumps(summary, indent=2))
        return 0

    log_summary(run, summary, step=0)
    # bank the reproduction + verdict artifacts
    for art_name, fname in [
        ("runner_dodge_findings", "findings.json"),
        ("repro_unpatched_buggy", "repro_unpatched_buggy.json"),
        ("repro_guard_buggy", "repro_guard_buggy.json"),
        ("repro_path_a_pinned", "repro_path_a_pinned.json"),
        ("guard_unit_proof", "guard_unit_proof.json"),
    ]:
        p = HERE / fname
        if p.exists():
            log_json_artifact(run, name=art_name, artifact_type="analysis", data=json.loads(p.read_text()))
    print(f"logged W&B run: {run.id}  (group fire-runner-dodge-denken)")
    print(f"official_gate_repass = {official_gate_repass}  (gate={gate})")
    finish_wandb(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
