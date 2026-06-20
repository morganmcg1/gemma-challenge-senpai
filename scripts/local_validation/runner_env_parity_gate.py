"""Runner-environment-parity launch gate: catch model-loadability before a fire.

Spawns a submission's ``serve.py`` in an environment that faithfully mirrors the
sandboxed HF-Job runner — **token-less HF auth** plus a **fresh, empty HF
cache** — and checks whether the model actually loads and the server reaches
readiness there. This catches the model-loadability failure class (private-repo
401, missing token, missing-from-Hub file, env gaps) that the official PPL /
completion / modalities gate is structurally blind to, BEFORE an HF-Jobs fire is
ever spent. Local A10G only: no HF job, no leaderboard submission.

Why this exists
---------------
The int4head fire (job ``6a36d5de3093dba73ce2b016``) passed every local quality
gate — PPL 2.0029, 128/128 completion, all four modalities, fire-prep panel
GREEN on every axis — and then ERRORED at server startup on a private-repo 401,
before serving a single prompt. Root cause:

* ``int4_mtp_bi0_int4head``'s manifest overrides ``MODEL_ID`` to the PRIVATE Hub
  repo ``gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head``;
* ``serve.py`` ``os.execvpe``'s ``vllm --model $MODEL_ID`` with ``os.environ``,
  which in the sandboxed runner has **no HF token** → 401
  ``RepositoryNotFoundError``;
* ``validate_submission.enforce_launch_gate`` only checks PPL / completion /
  modalities — it never attempts a token-less model fetch, so it ran GREEN in an
  env that *had* ambient credentials and a warm local cache (exactly the two
  masks this gate removes).

The fix is parity, not more quality checks: re-run ``serve.py`` under the
runner's actual auth/cache posture and require the model to load. A private repo
401s here, a public/ungated repo loads — verified empirically against the three
challenge repos (the two ``google/...`` QAT repos are ``gated=False`` and load
token-less; the ``gemma-challenge/...int4head`` repo returns HTTP 401).

Usage
-----
    .venv/bin/python -m scripts.local_validation.runner_env_parity_gate \\
        --submission submissions/int4_mtp_bi0_int4head      # expect RED 401
    .venv/bin/python -m scripts.local_validation.runner_env_parity_gate \\
        --submission submissions/int4_mtp_bi0_surgattn       # expect GREEN

Pass ``--require-loadable`` to turn ``runner_loadable=false`` into a non-zero
exit (the semantics a pre-fire launch gate wants).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from . import harness, paths

# Ambient HF credentials that could silently authenticate the local process and
# mask the token-less runner. A ``None`` value in ``extra_env`` tells
# [[harness.LocalServer]] to UNSET the key (its extra_env plumbing supports
# deletion, banked from wirbel #807). HF_TOKEN_PATH points the hub at an on-disk
# token file, so it must go too; the fresh HF_HOME below relocates the default
# token-file lookup ($HF_HOME/token) to an empty dir as a second line of defense.
HF_CREDENTIAL_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN_PATH",
)

# The HF-Job runner's server startup budget
# (hf_bucket_single_job.py parse_args: --startup-timeout-s default 900). Using
# the same budget makes a local "did not become ready in time" verdict mean the
# same thing it would on the runner.
DEFAULT_STARTUP_BUDGET_S = 900

# failure_class taxonomy, checked in order so the first matching family wins. A
# private-repo 401 (the int4head fire's actual death) must never be mis-labeled
# as a generic file/timeout error even though its traceback also mentions a token
# hint — so 401_private_repo is checked before the token/file families.
FAILURE_CLASS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "401_private_repo",
        (
            r"RepositoryNotFoundError",
            r"401\s*Client\s*Error",
            r"Repository\s+Not\s+Found",
            r"\b401\b[^\n]*(?:Unauthorized|Not\s+Found)",
        ),
    ),
    (
        "missing_token",
        (
            r"GatedRepoError",
            r"Cannot\s+access\s+gated\s+repo",
            r"gated\s+repo",
            r"403\s*Client\s*Error",
            r"[Aa]ccess\s+to\s+model\s+\S+\s+is\s+restricted",
            r"authentication\s+required",
            r"[Tt]oken\s+is\s+required",
        ),
    ),
    (
        "missing_hub_file",
        (
            r"EntryNotFoundError",
            r"does\s+not\s+appear\s+to\s+have\s+a\s+file\s+named",
            r"Entry\s+Not\s+Found",
            r"404\s*Client\s*Error",
        ),
    ),
)


def build_scrubbed_env(tmp_hf: Path) -> dict[str, str | None]:
    """``extra_env`` that makes serve.py's auth/cache posture match the runner.

    A ``None`` value UNSETS the key via [[harness.LocalServer]]'s extra_env
    plumbing. We strip every ambient HF credential and relocate the HF cache (and
    the on-disk token lookup) to a fresh empty dir, so that neither an ambient
    token nor a previously-downloaded local copy can mask the authenticated fetch
    the sandboxed runner must perform. We deliberately do NOT set
    ``HF_HUB_OFFLINE=1``: the runner is ONLINE and must reach the Hub — the goal
    is to reproduce the *authenticated-fetch* path, not to block fetches. Setting
    it to ``"0"`` defensively neutralizes any inherited offline pin.
    """
    hub = tmp_hf / "hub"
    env: dict[str, str | None] = {var: None for var in HF_CREDENTIAL_ENV_VARS}
    env["HF_HOME"] = str(tmp_hf)
    env["HF_HUB_CACHE"] = str(hub)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub)
    env["TRANSFORMERS_CACHE"] = str(tmp_hf / "transformers")
    env["HF_HUB_OFFLINE"] = "0"
    return env


def classify_failure(log_text: str, enter_error: str) -> str:
    """Map a readiness failure to a failure_class from the server log + error."""
    for failure_class, patterns in FAILURE_CLASS_PATTERNS:
        for pat in patterns:
            if re.search(pat, log_text, re.IGNORECASE):
                return failure_class
    # No deterministic load-error signature in the log. A server that was still
    # alive at the deadline is a timeout; an early exit with no recognized
    # signature is something else (OOM, CUDA, import error, ...).
    if "endpoint not ready" in enter_error and "server exited" not in enter_error:
        return "timeout"
    return "other"


def _all_failure_patterns() -> tuple[str, ...]:
    return tuple(pat for _cls, pats in FAILURE_CLASS_PATTERNS for pat in pats)


def error_snippet(log_text: str, max_lines: int = 40) -> str:
    """Excerpt of the server log centered on the decisive load error.

    Prefers a window spanning the lines that match a failure-class signature (so
    a chained ``401 -> OSError`` cascade shows the originating
    ``RepositoryNotFoundError``, not just the wrapper); else the last
    ``Traceback`` block; else the tail. Capped at ``max_lines`` non-empty lines.
    """
    lines = log_text.splitlines()
    sig_idx = [
        i
        for i, line in enumerate(lines)
        if any(re.search(p, line, re.IGNORECASE) for p in _all_failure_patterns())
    ]
    if sig_idx:
        lo = max(0, sig_idx[0] - 3)
        hi = min(len(lines), sig_idx[-1] + 4)
        chosen = lines[lo:hi]
    else:
        last_tb = None
        for i, line in enumerate(lines):
            if "Traceback (most recent call last)" in line:
                last_tb = i
        chosen = lines[last_tb:] if last_tb is not None else lines
    nonempty = [ln for ln in chosen if ln.strip()]
    return "\n".join(nonempty[-max_lines:])


def verdict_line(result: dict) -> str:
    name = result["submission_name"]
    model_id = result["model_id"]
    if result.get("runner_loadable"):
        return (
            f"RUNNER-ENV-PARITY {name}: runner_loadable=true "
            f"wall_clock_to_ready_s={result.get('wall_clock_to_ready_s')} model_id={model_id}"
        )
    return (
        f"RUNNER-ENV-PARITY {name}: runner_loadable=false "
        f"failure_class={result.get('failure_class')} "
        f"wall_clock_to_fail_s={result.get('wall_clock_to_fail_s')} model_id={model_id}"
    )


def run_gate(
    submission: Path,
    *,
    server_python: Path | None = None,
    port: int = 8000,
    budget_s: int = DEFAULT_STARTUP_BUDGET_S,
    out_dir: Path | None = None,
) -> dict:
    """Serve ``submission`` under a token-less, fresh-cache env; return the verdict."""
    submission = Path(submission)
    manifest = harness.load_manifest(submission)
    model_id = harness.serve_model_id(manifest, submission)
    manifest_env = dict(manifest.get("env") or {})
    server_python = Path(server_python) if server_python else harness.ensure_server_venv(manifest["dependencies"])

    name = submission.name
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(out_dir) if out_dir else (paths.LOCALRUN_ROOT / f"runner-env-parity-{name}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"

    tmp_hf = Path(tempfile.mkdtemp(prefix="runner-parity-hf-"))
    extra_env = build_scrubbed_env(tmp_hf)

    result: dict = {
        "submission": str(submission),
        "submission_name": name,
        "model_id": model_id,
        "manifest_env_keys": sorted(manifest_env),
        "scrubbed_credentials": list(HF_CREDENTIAL_ENV_VARS),
        "fresh_hf_home": str(tmp_hf),
        "startup_budget_s": budget_s,
        "server_python": str(server_python),
        "created_at": stamp,
        "out_dir": str(out_dir),
        "server_log": str(log_path),
    }

    srv = harness.LocalServer(
        submission,
        server_python=server_python,
        port=port,
        startup_timeout_s=budget_s,
        log_path=log_path,
        extra_env=extra_env,
    )
    print(f"[parity] scrubbed HF creds={list(HF_CREDENTIAL_ENV_VARS)}", flush=True)
    print(f"[parity] fresh HF_HOME={tmp_hf} (no warm cache, online)", flush=True)
    print(f"[parity] MODEL_ID={model_id} budget={budget_s}s", flush=True)

    t0 = time.time()
    enter_error: str | None = None
    runner_loadable = False
    try:
        srv.__enter__()
        elapsed = time.time() - t0
        runner_loadable = True
    except Exception as exc:  # readiness failure: early exit OR timeout
        elapsed = time.time() - t0
        enter_error = str(exc)
    finally:
        srv.__exit__(None, None, None)
        shutil.rmtree(tmp_hf, ignore_errors=True)

    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if runner_loadable:
        result.update(
            {
                "runner_loadable": True,
                "failure_class": None,
                "wall_clock_to_ready_s": round(elapsed, 2),
            }
        )
    else:
        failure_class = classify_failure(log_text, enter_error or "")
        result.update(
            {
                "runner_loadable": False,
                "failure_class": failure_class,
                "wall_clock_to_fail_s": round(elapsed, 2),
                "enter_error": enter_error,
                "error_snippet": error_snippet(log_text),
            }
        )
    result["verdict_line"] = verdict_line(result)
    (out_dir / "parity_evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _maybe_log_wandb(args, result: dict) -> None:
    """Best-effort W&B log of the parity verdict; no-op without creds/name."""
    if not getattr(args, "wandb_name", None):
        return
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover - logging must never break the gate
        print(f"[parity] wandb logging unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="runner-env-parity-gate",
        agent="senpai",
        name=args.wandb_name,
        tags=["runner-env-parity-gate", *([args.wandb_group] if args.wandb_group else [])],
        group=args.wandb_group,
        config={
            "submission": result.get("submission"),
            "submission_name": result.get("submission_name"),
            "model_id": result.get("model_id"),
            "startup_budget_s": result.get("startup_budget_s"),
            "scrubbed_credentials": result.get("scrubbed_credentials"),
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        return
    summary = {
        "runner_loadable": int(bool(result.get("runner_loadable"))),
        "failure_class": result.get("failure_class") or "none",
        "wall_clock_to_ready_s": result.get("wall_clock_to_ready_s"),
        "wall_clock_to_fail_s": result.get("wall_clock_to_fail_s"),
        "model_id": result.get("model_id"),
    }
    log_summary(run, {k: v for k, v in summary.items() if v is not None}, step=0)
    finish_wandb(run)


def _print_block(result: dict) -> None:
    name = result.get("submission_name", "?")
    line = "=" * 14 + f" RUNNER-ENV-PARITY GATE — {name} " + "=" * 14
    print("\n" + line, flush=True)
    print(f"submission:   {result.get('submission')}", flush=True)
    print(f"model_id:     {result.get('model_id')}", flush=True)
    print(f"scrubbed:     {', '.join(result.get('scrubbed_credentials', []))}", flush=True)
    print(f"fresh_cache:  {result.get('fresh_hf_home')}", flush=True)
    print(f"budget_s:     {result.get('startup_budget_s')}", flush=True)
    if result.get("runner_loadable"):
        print(f"VERDICT:      GREEN  runner_loadable=true  ready in {result.get('wall_clock_to_ready_s')}s", flush=True)
    else:
        print(
            f"VERDICT:      RED    runner_loadable=false  failure_class={result.get('failure_class')}  "
            f"(failed after {result.get('wall_clock_to_fail_s')}s)",
            flush=True,
        )
        snippet = result.get("error_snippet") or ""
        if snippet:
            print("-- error snippet --", flush=True)
            print(snippet, flush=True)
            print("-- end snippet --", flush=True)
    print(f"evidence:     {Path(result.get('out_dir', '.')) / 'parity_evidence.json'}", flush=True)
    print("=" * len(line) + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--server-python", type=Path, default=None, help="python with vLLM (default: build from manifest deps)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout-s", type=int, default=DEFAULT_STARTUP_BUDGET_S,
                    help="server readiness budget; defaults to the HF-Job runner's 900s")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--require-loadable",
        action="store_true",
        help="exit non-zero unless runner_loadable=true (the pre-fire launch-gate semantics)",
    )
    ap.add_argument("--wandb-name", default=None, help="log the parity verdict to W&B under this run name")
    ap.add_argument("--wandb-group", default=None, help="W&B group (e.g. runner-env-parity-gate)")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[parity] {note}", flush=True)

    result = run_gate(
        args.submission,
        server_python=args.server_python,
        port=args.port,
        budget_s=args.startup_timeout_s,
        out_dir=args.out_dir,
    )
    print(result["verdict_line"], flush=True)
    _print_block(result)
    _maybe_log_wandb(args, result)

    if args.require_loadable and not result.get("runner_loadable"):
        print(
            f"[parity] FAIL runner_loadable=false failure_class={result.get('failure_class')} "
            "— launch ineligible (the runner cannot load this submission)",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
