"""One-command local validation for a submission.

Given a submission dir, this:
  1. serves it locally (manifest deps + env, PPL headroom applied),
  2. captures decode token IDs and runs the greedy-identity gate vs the
     checkpoint's exact-greedy AR reference,
  3. runs local PPL against the ground-truth tokens,
  4. probes exploratory single-stream decode TPS,
and prints the compact ``tps / ppl / completed / greedy_verdict`` evidence block
to paste into an HF-Job approval issue (also written to ``evidence.json``).

The greedy reference must already exist for the checkpoint (generate it first,
since it can't share the GPU with the live server). Use the SERVED spec-off
reference — an offline reference diverges from a served candidate on a stochastic
~20% of bf16 prompts purely from FP non-determinism, so it is not a valid gate:
    /tmp/server-venv/bin/python -m scripts.local_validation.gen_greedy_reference \\
        --mode served --model-id <model> --num-prompts <N>

    python -m scripts.local_validation.validate_submission \\
        --submission submissions/vllm_baseline --server-python /tmp/server-venv/bin/python

It also computes the consolidated *official leaderboard gate* (PPL ≤ 2.42 AND
completion AND all-modalities-loaded; #38). Pass ``--official-gate`` to turn that
verdict into a hard exit code (PASS -> 0, FAIL/INCOMPLETE -> 1). The HF-launch
path consumes the same verdict through ``run_official_gate_preflight`` /
``enforce_launch_gate``: a job is launched ONLY when the gate is PASS, and both
FAIL and INCOMPLETE block (an unmeasured component is exactly when a submission
must not spend HF-Jobs quota).

``enforce_launch_gate`` adds a second hard pre-fire block (#819): a cheap,
always-on token-less Hub probe of the EXACT model the runner will anonymously
fetch. The official quality gate runs in a credentialed local env and is blind to
model LOADABILITY, so a private/gated/missing ``model_id`` 401s at server startup
AFTER the gate passes (the int4head fire). The probe raises on an auth/visibility
failure and only warns on a no-network/transient error (never false-RED a public
launch from an offline box).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import greedy_gate, harness, modalities_probe, paths, same_path_ppl
from .ppl_runner import _headroom_overrides

# Default same-path-vs-prompt_logprobs PPL gap (nats of mean log-likelihood,
# expressed as a PPL delta) above which a submission is treated as a
# timed-vs-scored path split. Justified in research/validity/same_path_ppl.md:
# the honest baseline agrees to < 0.02 (FP noise), so 0.05 is a ~2.5x margin —
# wide enough to never flag honest FP/quantization jitter, tight enough that the
# 2.378 (prompt_logprobs) vs 2.55 (same-path) LF29 lane gap of ~0.17 trips it.
DEFAULT_SAME_PATH_THRESHOLD = 0.05

# --------------------------------------------------------------------------- #
# OFFICIAL-GATE LAUNCH PREFLIGHT (PR #45 verdict -> PR #50 hard launch block)
# --------------------------------------------------------------------------- #
# The official leaderboard gate is three-valued (modalities_probe.official_gate_verdict):
#   * PASS       — every check ran and passed                 -> launch eligible
#   * FAIL       — at least one check ran and failed          -> blocked, fix required
#   * INCOMPLETE — at least one check has not been run yet     -> blocked, run all checks
# An HF job may be launched ONLY when the verdict is PASS. INCOMPLETE blocks just as
# hard as FAIL: an unmeasured component is exactly the state where a submission must
# NOT spend HF-Jobs quota, because the preflight has not been fully executed.
GATE_BLOCK_VALUES = {"FAIL", "INCOMPLETE"}

# A launch authorizes the *official* 128-prompt protocol, so the authorizing
# evidence must itself be a full-protocol validation. An 8-prompt smoke can read
# official_gate=PASS for its own 8 prompts, but it does NOT authorize a real
# launch — the preflight downgrades under-prompted evidence to INCOMPLETE.
OFFICIAL_NUM_PROMPTS = paths.NUM_PROMPTS  # 128


def check_launch_eligibility(gate_verdict: str) -> bool:
    """Return True only if ``gate_verdict == "PASS"``. FAIL and INCOMPLETE block.

    Fail-closed: any value in ``GATE_BLOCK_VALUES`` blocks, and an unrecognized
    verdict is also ineligible (only an explicit PASS authorizes a launch).
    """
    if gate_verdict in GATE_BLOCK_VALUES:
        return False
    return gate_verdict == "PASS"


def _normalize_submission_key(value) -> str:
    """Collapse a submission name / dir / prefix to a comparison key.

    Evidence records the local dir name (``fa2sw_precache_kenyan``); a launch
    prefix uses the hyphenated form (``fa2sw-precache-kenyan``). Normalizing the
    basename and folding ``-``/``_`` lets the preflight match either spelling.
    """
    return Path(str(value)).name.strip().lower().replace("-", "_")


def _iter_evidence(localrun_root: Path):
    """Yield (created_at, path, evidence-dict) for every readable evidence.json."""
    if not localrun_root.is_dir():
        return
    for ev_path in localrun_root.glob("*/evidence.json"):
        try:
            data = json.loads(ev_path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            created = str(data.get("created_at") or "")
            yield created, ev_path, data


def run_official_gate_preflight(
    submission,
    *,
    num_prompts: int = OFFICIAL_NUM_PROMPTS,
    localrun_root: Path | None = None,
) -> dict:
    """Resolve the launch-eligibility gate for ``submission`` from local evidence.

    Reads the most recent ``validate_submission`` ``evidence.json`` for this
    submission (under ``paths.LOCALRUN_ROOT`` by default) and returns the
    consolidated gate verdict, recomputed from its raw PPL / completion /
    modalities components so a stale or absent stored verdict can never authorize
    a launch. The launch is the dangerous, quota-spending action, so the preflight
    is deliberately a cheap read of the evidence the student already produced — it
    does not re-serve on GPU.

    Returns a dict with ``official_gate`` (PASS/FAIL/INCOMPLETE), the components,
    and provenance (``evidence_path``, ``reason``). Two safety downgrades to
    INCOMPLETE (blocking):

      * no evidence found for this submission — validation has not been run;
      * the latest evidence ran fewer than ``num_prompts`` prompts — a smoke does
        not authorize the official 128-prompt protocol.
    """
    root = Path(localrun_root) if localrun_root is not None else paths.LOCALRUN_ROOT
    key = _normalize_submission_key(submission)

    matches = [
        (created, path, data)
        for created, path, data in _iter_evidence(root)
        if _normalize_submission_key(data.get("submission_name") or data.get("submission") or "") == key
    ]
    base = {
        "submission": str(submission),
        "required_num_prompts": num_prompts,
        "official_gate": "INCOMPLETE",
        "official_gate_pass": False,
        "ppl": None,
        "completed": None,
        "num_prompts": None,
        "all_modalities_loaded": None,
        "modalities_method": None,
        "evidence_path": None,
        "eligible": False,
    }
    if not matches:
        base["reason"] = (
            f"no official-gate validation evidence found for submission '{key}' under {root} — "
            f"run validate_submission --submission <dir> --num-prompts {num_prompts} --official-gate first"
        )
        return base

    created, path, ev = max(matches, key=lambda m: (m[0], m[1].stat().st_mtime))
    ev_n = ev.get("num_prompts")
    gate = modalities_probe.official_gate_verdict(
        ppl=ev.get("ppl"),
        completed=ev.get("completed"),
        num_prompts=ev_n if isinstance(ev_n, int) else num_prompts,
        all_modalities_loaded=ev.get("all_modalities_loaded"),
    )
    verdict = gate["official_gate"]
    result = {
        **base,
        "official_gate": verdict,
        "official_gate_pass": verdict == "PASS",
        "official_gate_ppl_ok": gate["official_gate_ppl_ok"],
        "official_gate_completion_ok": gate["official_gate_completion_ok"],
        "official_gate_modalities_ok": gate["official_gate_modalities_ok"],
        "ppl": ev.get("ppl"),
        "completed": ev.get("completed"),
        "num_prompts": ev_n,
        "all_modalities_loaded": ev.get("all_modalities_loaded"),
        "modalities_method": ev.get("modalities_method"),
        "evidence_path": str(path),
        "evidence_created_at": created or None,
        # The served model id captured at validation time and the submission dir
        # the evidence came from — consumed by the token-less runner-fetch probe in
        # enforce_launch_gate (PR #819) to resolve what the runner anonymously pulls.
        "model_id": ev.get("model_id"),
        "evidence_submission": ev.get("submission"),
    }
    # Launch-sufficiency: a sub-128 smoke (even a PASS) cannot authorize the
    # official 128-prompt run — downgrade to INCOMPLETE.
    if not isinstance(ev_n, int) or ev_n < num_prompts:
        result["official_gate"] = "INCOMPLETE"
        result["official_gate_pass"] = False
        result["eligible"] = False
        result["reason"] = (
            f"latest validation evidence ran {ev_n} prompts (< required {num_prompts}); "
            f"a smoke does not authorize the official {num_prompts}-prompt launch — "
            f"re-run full-protocol validation"
        )
        return result

    result["eligible"] = check_launch_eligibility(verdict)
    if not result["eligible"]:
        result["reason"] = f"official_gate = {verdict} (not PASS) in {path}"
    return result


# --------------------------------------------------------------------------- #
# TOKEN-LESS RUNNER-FETCH PRE-FIRE PROBE (PR #819)
# --------------------------------------------------------------------------- #
# The official gate above runs in a CREDENTIALED local env with a warm HF cache,
# so it is structurally blind to model LOADABILITY on the runner. The HF-Job vLLM
# subprocess inherits NO HF token, so a private / gated / missing ``model_id``
# 401s at server startup AFTER PPL / completion / modalities have all passed —
# exactly how the int4head fire (job 6a36d5de) died. ``enforce_launch_gate``
# therefore resolves the EXACT model the runner will anonymously fetch (mirroring
# serve.py, including the ``LMHEAD_QUANT_AT_STARTUP=1`` public-base Plan-A path)
# and requires it to be token-less reachable. An AUTH/visibility failure
# (401/403/404, private/gated/missing) BLOCKS; a no-network / offline / transient
# error is inconclusive and only WARNS + skips — it must never false-RED a launch
# whose model is in fact public, just because the local box is offline.


def resolve_runner_fetch_target(submission_dir, *, manifest: dict | None = None) -> dict:
    """The model the HF runner ANONYMOUSLY fetches from the Hub, mirroring serve.py.

    serve.py computes ``model_id = MODEL_ID env or default`` then
    ``maybe_quant_lmhead_at_startup(model_id)``. Under the token-free Plan-A path
    (``LMHEAD_QUANT_AT_STARTUP=1``) the runner ``snapshot_download``s that SAME base
    ``model_id`` (optionally pinned to ``LMHEAD_QUANT_BASE_REV``) and quant-builds it
    on disk, so the Hub-fetch target is still the base ``model_id`` — only a revision
    pin is added. A ``model_id`` that resolves to a local path (absolute, or a path
    that exists inside the submission) is served from disk: no anonymous Hub fetch,
    so the token-less probe is N/A (``is_local=True``). The Plan-A flag/revision are
    read from the MANIFEST env (what ships in the package and what the runner sees).
    """
    submission_dir = Path(submission_dir)
    manifest = manifest if manifest is not None else harness.load_manifest(submission_dir)
    model_id = harness.serve_model_id(manifest, submission_dir)
    env_block = manifest.get("env") or {}
    plan_a = str(env_block.get("LMHEAD_QUANT_AT_STARTUP", "0")) == "1"
    revision = (env_block.get("LMHEAD_QUANT_BASE_REV") or None) if plan_a else None
    resolved = harness.resolve_model_id(model_id, submission_dir)
    is_local = os.path.isabs(resolved) or resolved != model_id
    return {
        "model_id": model_id,
        "revision": revision,
        "plan_a_startup_quant": plan_a,
        "is_local": is_local,
        "resolved_path": resolved if is_local else None,
    }


def _short(exc, n: int = 160) -> str:
    head = (str(exc).splitlines() or [""])[0].strip()
    return (head or type(exc).__name__)[:n]


def _classify_probe_exception(exc) -> dict:
    """Map a ``model_info(token=False)`` exception to a probe verdict.

    AUTH/visibility failures — the token-less runner WILL hit them — BLOCK; a
    no-network / offline / transient error is inconclusive and must NOT block
    (instruction: distinguish auth failure from no-network, cite which).
    """
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    status = getattr(getattr(exc, "response", None), "status_code", None)
    # Gated (403) is a RepositoryNotFoundError subclass, so check it first.
    if isinstance(exc, GatedRepoError) or status == 403:
        return {
            "reachable": False,
            "blocking": True,
            "status": "blocked_gated_403",
            "detail": f"gated repo — token-less access denied (403): {_short(exc)}",
        }
    if isinstance(exc, RepositoryNotFoundError) or status in (401, 404):
        return {
            "reachable": False,
            "blocking": True,
            "status": f"blocked_unauthorized_{status or 'private'}",
            "detail": f"private/missing repo — token-less {status or '401'}: {_short(exc)}",
        }
    # No network, DNS failure, timeout, offline pin, transient 5xx, or anything
    # else: inconclusive, so WARN + skip rather than false-RED a public launch.
    return {
        "reachable": None,
        "blocking": False,
        "status": "skipped_unreachable",
        "detail": f"Hub unreachable, NOT an auth failure ({type(exc).__name__}): {_short(exc)}",
    }


def probe_token_less_fetch(model_id, *, revision: str | None = None, model_info_fn=None) -> dict:
    """Probe whether ``model_id`` is anonymously fetchable — the runner's posture.

    Calls ``HfApi().model_info(model_id, revision=revision, token=False)``;
    ``token=False`` forces the anonymous fetch the sandboxed runner performs,
    regardless of any ambient local credential or warm cache. ``model_info_fn`` is
    injectable for tests. Returns ``reachable`` (True/False/None), ``blocking``
    (bool), ``status``, ``detail``, and ``elapsed_s``.
    """
    result = {"model_id": model_id, "revision": revision, "token_less": True}
    if model_info_fn is None:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:  # pragma: no cover - hub client missing
            result.update(
                {
                    "reachable": None,
                    "blocking": False,
                    "status": "skipped_no_hub_client",
                    "detail": f"huggingface_hub unavailable ({type(exc).__name__}): {_short(exc)}",
                }
            )
            return result
        model_info_fn = HfApi().model_info
    t0 = time.time()
    try:
        info = model_info_fn(model_id, revision=revision, token=False)
    except Exception as exc:
        result.update(_classify_probe_exception(exc))
    else:
        result.update(
            {
                "reachable": True,
                "blocking": False,
                "status": "reachable_token_less",
                "detail": f"anonymously fetchable (sha={getattr(info, 'sha', None)})",
            }
        )
    result["elapsed_s"] = round(time.time() - t0, 3)
    return result


def _resolve_launch_fetch_target(gate: dict) -> dict:
    """Resolve the runner's anonymous Hub-fetch target for ``enforce_launch_gate``.

    Prefers the submission MANIFEST (mirrors serve.py exactly, incl. the Plan-A
    revision pin); falls back to the served ``model_id`` recorded in the validation
    evidence when the manifest is not loadable in this checkout — the SAME
    pre-transform ``MODEL_ID`` the runner fetches, minus only the Plan-A rev pin.
    """
    sub_path = gate.get("evidence_submission")
    if sub_path:
        d = Path(sub_path)
        if not d.is_absolute():
            d = paths.ROOT / sub_path
        if (d / "manifest.json").exists():
            try:
                target = resolve_runner_fetch_target(d)
                target["source"] = "manifest"
                return target
            except Exception:
                pass
    model_id = gate.get("model_id")
    if not model_id:
        return {"model_id": None, "revision": None, "plan_a_startup_quant": None,
                "is_local": False, "resolved_path": None, "source": "none"}
    is_local = os.path.isabs(model_id)
    return {"model_id": model_id, "revision": None, "plan_a_startup_quant": None,
            "is_local": is_local, "resolved_path": model_id if is_local else None,
            "source": "evidence"}


def runner_fetch_pre_fire_probe(gate: dict, *, model_info_fn=None) -> dict:
    """Resolve the runner's fetch target and probe its token-less reachability."""
    target = _resolve_launch_fetch_target(gate)
    if target.get("model_id") is None:
        return {**target, "reachable": None, "blocking": False, "status": "skipped_no_model_id",
                "detail": "no served model_id in evidence and no loadable manifest"}
    if target.get("is_local"):
        return {**target, "reachable": None, "blocking": False, "status": "skipped_local_path",
                "detail": f"served from local files ({target.get('resolved_path')}); no anonymous Hub fetch"}
    probe = probe_token_less_fetch(
        target["model_id"], revision=target.get("revision"), model_info_fn=model_info_fn
    )
    return {**target, **probe}


def enforce_launch_gate(
    submission,
    *,
    num_prompts: int = OFFICIAL_NUM_PROMPTS,
    localrun_root: Path | None = None,
    model_info_fn=None,
) -> dict:
    """Run the preflight and raise RuntimeError unless the gate is PASS *and* the
    served model is anonymously fetchable.

    The single chokepoint the HF-launch path calls before submitting a job. Two
    hard blocks: (1) the official quality gate — FAIL and INCOMPLETE both raise;
    (2) the token-less runner-fetch probe — a private/gated/missing model_id
    raises (the int4head 401 class), while a no-network/transient probe error only
    warns. PASS on both returns the gate dict (with ``runner_fetch_probe`` attached).
    """
    gate = run_official_gate_preflight(submission, num_prompts=num_prompts, localrun_root=localrun_root)
    if not check_launch_eligibility(gate["official_gate"]):
        raise RuntimeError(
            f"HF launch blocked: official_gate = {gate['official_gate']} for submission "
            f"'{submission}'. ppl={gate.get('ppl')} "
            f"completed={gate.get('completed')}/{gate.get('num_prompts')} "
            f"all_modalities_loaded={gate.get('all_modalities_loaded')} "
            f"evidence={gate.get('evidence_path')}. "
            f"{gate.get('reason') or 'Fix the gate failures and re-run full-protocol local validation before launching.'}"
        )

    probe = runner_fetch_pre_fire_probe(gate, model_info_fn=model_info_fn)
    gate["runner_fetch_probe"] = probe
    if probe.get("blocking"):
        rev = probe.get("revision")
        rev_note = f" @ {rev}" if rev else ""
        raise RuntimeError(
            f"HF launch blocked: the runner serves TOKEN-LESS but model_id "
            f"'{probe.get('model_id')}'{rev_note} is NOT anonymously fetchable "
            f"({probe.get('status')}: {probe.get('detail')}). The official gate passed in a "
            f"credentialed local env, but the sandboxed HF-Job runner gets NO HF token — this "
            f"is the private-repo 401 the int4head fire (job 6a36d5de) hit at server startup. "
            f"Publish the model public/ungated, or serve the public base via the "
            f"LMHEAD_QUANT_AT_STARTUP=1 Plan-A path, before launching."
        )
    if probe.get("reachable") is None:
        # Non-blocking inconclusive (no network / local path / no client): surface
        # loudly so the operator knows the loadability check did NOT actually run.
        print(
            f"[launch-gate] WARN token-less runner-fetch probe inconclusive for model_id "
            f"'{probe.get('model_id')}': {probe.get('status')} ({probe.get('detail')}). "
            f"Loadability NOT confirmed — run runner_env_parity_gate --require-loadable "
            f"as the one-shot pre-fire confirmation.",
            flush=True,
        )
    return gate


def _greedy_summary(report) -> str:
    if report.verdict == "GREEDY_IDENTICAL":
        return f"GREEDY_IDENTICAL ({report.num_identical}/{report.num_prompts_compared} identical)"
    if report.verdict == "DIVERGENT":
        return f"DIVERGENT ({report.num_divergent}/{report.num_prompts_compared} prompts differ)"
    return "INCOMPARABLE (prompt sets differ / integrity failure)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--server-python", type=Path, default=None, help="python with vLLM (default: build from manifest deps)")
    ap.add_argument("--reference", type=Path, default=None, help="reference decode_outputs.jsonl (default: auto by model id)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-tps", action="store_true")
    ap.add_argument("--skip-modalities", action="store_true",
                    help="skip the official-gate modalities-load probe (text/image/audio/video)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="served checkpoint dir for the modalities presence tier (default: resolve from manifest)")
    ap.add_argument("--tps-tokens", type=int, default=256)
    ap.add_argument(
        "--check-same-path",
        action="store_true",
        help="also score PPL through the timed generation path (echo+logprobs) and FAIL "
        "(non-zero exit) if it diverges from the prompt_logprobs PPL by more than the threshold",
    )
    ap.add_argument("--same-path-threshold", type=float, default=DEFAULT_SAME_PATH_THRESHOLD,
                    help="max allowed |same_path_ppl - prompt_logprobs_ppl| before the gate fails")
    ap.add_argument(
        "--official-gate",
        action="store_true",
        help="make the process exit code reflect the official leaderboard gate: 0 only if "
        "official_gate == PASS, else non-zero (FAIL/INCOMPLETE both block). The gate is always "
        "computed and printed; this flag turns it into a hard CI/launch gate.",
    )
    ap.add_argument("--wandb-name", default=None, help="log the validation evidence to W&B under this run name")
    ap.add_argument("--wandb-group", default=None, help="W&B group (e.g. fa2sw-precache-validate-and-lf29-check)")
    args = ap.parse_args(argv)

    # The same-path gate compares against the prompt_logprobs PPL, so it needs
    # the PPL stage to run.
    if args.check_same_path and args.skip_ppl:
        ap.error("--check-same-path requires the PPL stage; do not pass --skip-ppl")

    for note in paths.prepare_local_gpu_env():
        print(f"[validate] {note}", flush=True)

    submission = args.submission
    manifest = harness.load_manifest(submission)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    name = submission.name
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.out_dir or (paths.LOCALRUN_ROOT / f"validate-{name}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = _headroom_overrides(manifest.get("env", {}))
    evidence: dict = {
        "submission": str(submission),
        "submission_name": name,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "created_at": stamp,
        "out_dir": str(out_dir),
        "stages": {},
        "failures": [],
    }

    with harness.LocalServer(
        submission,
        server_python=server_python,
        port=args.port,
        log_path=out_dir / "server.log",
        extra_env=overrides,
    ) as srv:
        evidence["model_id"] = srv.model_id
        evidence["reference_model_id"] = srv.reference_model_id
        evidence["served_model_name"] = srv.served_model_name

        # 1) Decode capture + greedy-identity gate.
        if not args.skip_greedy:
            try:
                decode_summary = harness.capture_decode(
                    server_python,
                    base_url=srv.base_url,
                    model=srv.served_model_name,
                    out_file=out_dir / "decode_outputs.jsonl",
                    summary_file=out_dir / "decode_summary.json",
                    num_prompts=args.num_prompts,
                    output_len=args.output_len,
                )
                evidence["stages"]["decode"] = {"num_records": decode_summary["num_records"]}
                evidence["completed"] = decode_summary["num_records"]

                # Canonical auto-resolution (no manual --reference threading): the
                # reference dir is keyed by the submission's collision-free identity,
                # NOT by prompt count, so the same path holds whatever N was last
                # generated. It must hold >= args.num_prompts prompts or the gate reads
                # INCOMPARABLE — e.g. running --num-prompts 128 needs the 128-prompt
                # reference (regenerate: gen_greedy_reference --mode served --submission
                # <dir> --num-prompts 128 [--ref-env <drafter-off>]).
                reference = args.reference or greedy_gate.reference_for(srv.reference_model_id)
                if not Path(reference).exists():
                    msg = (f"greedy reference missing: {reference} — generate it for THIS submission with "
                           f"gen_greedy_reference --mode served --submission {submission} [--spec-off] "
                           f"--num-prompts {args.num_prompts}")
                    evidence["failures"].append(msg)
                    evidence["greedy_verdict"] = "NO_REFERENCE"
                    print(f"[validate] WARN {msg}", flush=True)
                else:
                    # N-mismatch legibility: the gate compares prompt-for-prompt,
                    # so a reference with fewer records than --num-prompts silently
                    # yields INCOMPARABLE for the unmatched prompts. Surface the
                    # record count and warn loudly with the exact fix.
                    ref_n = greedy_gate.reference_num_records(Path(reference))
                    if ref_n is not None:
                        evidence["reference_num_records"] = ref_n
                        if ref_n < args.num_prompts:
                            evidence["reference_n_mismatch"] = True
                            print(
                                f"[validate] WARNING: resolved reference has {ref_n} records but "
                                f"--num-prompts={args.num_prompts}.\n"
                                f"[validate]          Gate will return INCOMPARABLE for the "
                                f"{args.num_prompts - ref_n} prompts with no reference.\n"
                                f"[validate]          Regenerate the reference with --num-prompts >= "
                                f"{args.num_prompts} to get a complete verdict.",
                                flush=True,
                            )
                    report = greedy_gate.compare(Path(reference), out_dir / "decode_outputs.jsonl")
                    onset = greedy_gate.onset_summary(report)
                    ref_kind = greedy_gate.reference_kind(Path(reference))
                    evidence["greedy_verdict"] = report.verdict
                    evidence["greedy_reference_kind"] = ref_kind
                    evidence["greedy_onset"] = onset
                    evidence["stages"]["greedy"] = {
                        "verdict": report.verdict,
                        "reference": str(reference),
                        "reference_kind": ref_kind,
                        "num_identical": report.num_identical,
                        "num_divergent": report.num_divergent,
                        "num_prompts_compared": report.num_prompts_compared,
                        "total_divergent_tokens": report.total_divergent_tokens,
                        "divergence_onset": onset,
                    }
                    (out_dir / "greedy_report.json").write_text(json.dumps(report.to_dict(), indent=2))
                    print(f"[validate] greedy: {_greedy_summary(report)}", flush=True)
                    print(f"[validate] {greedy_gate.onset_line(onset, args.output_len)}", flush=True)
            except Exception as exc:  # keep going; record the failure
                evidence["failures"].append(f"greedy stage error: {exc}")
                evidence["greedy_verdict"] = "ERROR"
                print(f"[validate] ERROR greedy stage: {exc}", flush=True)

        # 2) Local PPL.
        if not args.skip_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    server_python,
                    base_url=srv.base_url,
                    model=srv.served_model_name,
                    out_file=out_dir / "ppl_results.jsonl",
                    summary_file=out_dir / "ppl_summary.json",
                )
                evidence["ppl"] = ppl_summary["ppl"]
                evidence["stages"]["ppl"] = {
                    "ppl": ppl_summary["ppl"],
                    "num_tokens": ppl_summary["num_tokens"],
                    "num_records": ppl_summary["num_records"],
                }
                print(f"[validate] PPL={ppl_summary['ppl']:.4f}", flush=True)
            except Exception as exc:
                evidence["failures"].append(f"ppl stage error: {exc}")
                print(f"[validate] ERROR ppl stage: {exc}", flush=True)

        # 2b) Same-path PPL gate: score the SAME GT span through the timed
        # generation path (echo+logprobs, no prompt_logprobs) and require it to
        # agree with the prompt_logprobs PPL above. A gap is the signature of a
        # submission whose timed-throughput model differs from the scored model.
        if args.check_same_path:
            try:
                sp_summary = same_path_ppl.score_endpoint(
                    srv.base_url,
                    srv.served_model_name,
                    out_dir=out_dir,
                )
                evidence["same_path_ppl"] = sp_summary["ppl"]
                stage = {
                    "same_path_ppl": sp_summary["ppl"],
                    "num_tokens": sp_summary["num_tokens"],
                    "num_records": sp_summary["num_records"],
                    "threshold": args.same_path_threshold,
                }
                ppl_pl = evidence.get("ppl")
                if isinstance(ppl_pl, (int, float)):
                    gap = abs(sp_summary["ppl"] - ppl_pl)
                    verdict = "SAME_PATH_OK" if gap <= args.same_path_threshold else "PATH_SPLIT"
                    evidence["prompt_logprobs_ppl"] = ppl_pl
                    evidence["same_path_gap"] = gap
                    evidence["same_path_verdict"] = verdict
                    stage.update({"prompt_logprobs_ppl": ppl_pl, "gap": gap, "verdict": verdict})
                    if verdict != "SAME_PATH_OK":
                        evidence["failures"].append(
                            f"same-path PPL gate FAILED: |{sp_summary['ppl']:.4f} - {ppl_pl:.4f}| "
                            f"= {gap:.4f} > {args.same_path_threshold} threshold (timed-vs-scored path split)"
                        )
                    print(
                        f"[validate] same-path PPL={sp_summary['ppl']:.4f} "
                        f"prompt_logprobs PPL={ppl_pl:.4f} gap={gap:.4f} -> {verdict}",
                        flush=True,
                    )
                else:
                    evidence["same_path_verdict"] = "NO_PROMPT_LOGPROBS_PPL"
                    evidence["failures"].append(
                        "same-path gate could not compare: prompt_logprobs PPL stage did not produce a number"
                    )
                    print("[validate] same-path gate: WARN no prompt_logprobs PPL to compare against", flush=True)
                evidence["stages"]["same_path"] = stage
            except Exception as exc:
                evidence["failures"].append(f"same-path stage error: {exc}")
                evidence["same_path_verdict"] = "ERROR"
                print(f"[validate] ERROR same-path stage: {exc}", flush=True)

        # 3) Exploratory TPS probe.
        if not args.skip_tps:
            try:
                tps = harness.probe_tps(srv.base_url, srv.served_model_name, decode_tokens=args.tps_tokens)
                evidence["tps_single_stream_a10g"] = tps["decode_tps_single_stream"]
                evidence["stages"]["tps"] = tps
                print(f"[validate] TPS(local a10g, single-stream)={tps['decode_tps_single_stream']:.2f} tok/s", flush=True)
            except Exception as exc:
                evidence["failures"].append(f"tps stage error: {exc}")
                print(f"[validate] ERROR tps stage: {exc}", flush=True)

        # 4) Modalities-load probe — the official-gate criterion the harness never
        # checks (program.md:29-31; #38). Runs LAST so a stray multimodal request
        # can never destabilize the decode/ppl/tps evidence already captured.
        if not args.skip_modalities:
            try:
                mod = modalities_probe.probe_modalities(
                    base_url=srv.base_url,
                    model=srv.served_model_name,
                    manifest=manifest,
                    submission_dir=submission,
                    model_id=srv.model_id,
                    model_dir=args.model_dir,
                )
                evidence["modalities_loaded"] = mod["modalities_loaded"]
                evidence["all_modalities_loaded"] = mod["all_modalities_loaded"]
                evidence["modalities_method"] = mod["modalities_method"]
                evidence["stages"]["modalities"] = mod
                if mod["all_modalities_loaded"] is False:
                    missing = [m for m, v in mod["modalities_loaded"].items() if v is False]
                    evidence["failures"].append(
                        f"modalities gate: {', '.join(missing)} not loaded/non-zero (program.md:29-31)"
                    )
                print(
                    f"[validate] modalities: {mod['modalities_loaded']} "
                    f"-> all_modalities_loaded={mod['all_modalities_loaded']}",
                    flush=True,
                )
            except Exception as exc:
                evidence["failures"].append(f"modalities stage error: {exc}")
                evidence["all_modalities_loaded"] = None
                print(f"[validate] ERROR modalities stage: {exc}", flush=True)

    # Consolidated official leaderboard gate (#38: PPL + completion + modalities,
    # NOT token-identity). Computed from whatever the stages produced; an unknown
    # input yields INCOMPLETE rather than a false PASS.
    gate = modalities_probe.official_gate_verdict(
        ppl=evidence.get("ppl"),
        completed=evidence.get("completed"),
        num_prompts=args.num_prompts,
        all_modalities_loaded=evidence.get("all_modalities_loaded"),
    )
    evidence.update(gate)

    (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    _print_block(evidence, out_dir)
    _maybe_log_wandb(args, evidence)

    # The same-path gate is the only stage that can fail the whole command: a
    # PATH_SPLIT (or an inability to measure it when requested) must be a loud,
    # non-zero exit so an approval issue cannot attach a green block over it.
    if args.check_same_path and evidence.get("same_path_verdict") != "SAME_PATH_OK":
        print(
            f"[validate] FAIL same-path gate verdict={evidence.get('same_path_verdict')} "
            "(see failures above)",
            flush=True,
        )
        return 1

    # --official-gate turns the official leaderboard verdict into a hard exit
    # gate: PASS -> 0, FAIL/INCOMPLETE -> 1. Mirrors check_launch_eligibility so a
    # CI step or a launch-preflight wrapper can treat a non-zero exit as "do not
    # launch" without re-parsing evidence.json.
    if args.official_gate:
        verdict = evidence.get("official_gate")
        if not check_launch_eligibility(verdict):
            print(
                f"[validate] FAIL official_gate = {verdict} (not PASS) — launch ineligible "
                "(FAIL and INCOMPLETE both block)",
                flush=True,
            )
            return 1
        print("[validate] official_gate = PASS — launch eligible", flush=True)
    return 0


def _maybe_log_wandb(args, evidence: dict) -> None:
    """Best-effort W&B log of the validation evidence; no-op without creds/name."""
    if not getattr(args, "wandb_name", None):
        return
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover - logging must never break the gate
        print(f"[validate] wandb logging unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="validate-submission",
        agent="senpai",
        name=args.wandb_name,
        tags=["same-path-ppl-gate", *([args.wandb_group] if args.wandb_group else [])],
        config={
            "submission": evidence.get("submission"),
            "submission_name": evidence.get("submission_name"),
            "model_id": evidence.get("model_id"),
            "same_path_threshold": args.same_path_threshold,
            "check_same_path": args.check_same_path,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        return
    summary = {
        key: evidence[key]
        for key in (
            "ppl",
            "same_path_ppl",
            "prompt_logprobs_ppl",
            "same_path_gap",
            "tps_single_stream_a10g",
            "completed",
            "same_path_verdict",
            "greedy_verdict",
            # Official leaderboard gate (#38: PPL + completion + modalities).
            "official_gate",
            "official_gate_pass",
            "official_gate_ppl_ok",
            "official_gate_completion_ok",
            "official_gate_modalities_ok",
            "all_modalities_loaded",
        )
        if key in evidence
    }
    # Per-modality status as numeric metrics (1 loaded / 0 missing; unknown omitted).
    for mod, value in (evidence.get("modalities_loaded") or {}).items():
        if value is not None:
            summary[f"modality_{mod}"] = int(bool(value))
    log_summary(run, summary, step=0)
    finish_wandb(run)


def _fmt(v, spec="") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def _ok_mark(value) -> str:
    return {True: "[ok]", False: "[FAIL]", None: "[unknown]"}.get(value, "")


def _modalities_line(ev: dict) -> str:
    loaded = ev.get("modalities_loaded") or {}
    method = ev.get("modalities_method") or {}
    flag = {True: "LOADED", False: "MISSING", None: "UNKNOWN"}
    parts = [f"{m}={flag.get(loaded.get(m), 'UNKNOWN')}({method.get(m, '?')})" for m in modalities_probe.MODALITIES]
    return " ".join(parts)


def _print_block(ev: dict, out_dir: Path) -> None:
    name = ev.get("submission_name", "?")
    line = "=" * 16 + f" LOCAL VALIDATION — {name} " + "=" * 16
    print("\n" + line, flush=True)
    print(f"submission:     {ev.get('submission')}", flush=True)
    print(f"model_id:       {ev.get('model_id')}", flush=True)

    # --- OFFICIAL LEADERBOARD GATE: PPL + completion + modalities (#38) -------
    print("\n-- OFFICIAL LEADERBOARD GATE (PPL + completion + modalities; NOT token-identity, #38) --", flush=True)
    print(f"official_gate:  {ev.get('official_gate', 'n/a')}  (leaderboard verdict)", flush=True)
    ppl = ev.get("ppl")
    cap_note = "<= 2.42 cap" if isinstance(ppl, (int, float)) and ppl <= 2.42 else "OVER 2.42 CAP" if isinstance(ppl, (int, float)) else ""
    print(f"  ppl:          {_fmt(ppl, '.4f')}   {cap_note} {_ok_mark(ev.get('official_gate_ppl_ok'))}", flush=True)
    comp = ev.get("completed")
    comp_str = f"{comp}/{ev.get('num_prompts')}" if comp is not None else "n/a"
    print(f"  completed:    {comp_str} {_ok_mark(ev.get('official_gate_completion_ok'))}", flush=True)
    print(f"  modalities:   {_modalities_line(ev)}  -> all={ev.get('all_modalities_loaded')} "
          f"{_ok_mark(ev.get('official_gate_modalities_ok'))}", flush=True)

    # --- INTERNAL HARDENING SIGNALS (reproducibility, NOT official gates) -----
    print("\n-- INTERNAL HARDENING SIGNALS (reproducibility; NOT official leaderboard gates) --", flush=True)
    rk = ev.get("greedy_reference_kind")
    rk_note = f"  [ref: {rk}]" if rk else ""
    print(f"greedy_verdict: {ev.get('greedy_verdict', 'skipped')}{rk_note}", flush=True)
    onset = ev.get("greedy_onset")
    if onset and onset.get("num_divergent"):
        print(f"                {greedy_gate.onset_line(onset, ev.get('output_len'))}", flush=True)
    print("  note: greedy-identity is NOT an official leaderboard gate (kanna #38); "
          "it is an internal reproducibility signal.", flush=True)
    sp_verdict = ev.get("same_path_verdict")
    if sp_verdict:
        sp_ppl = ev.get("same_path_ppl")
        gap = ev.get("same_path_gap")
        gap_note = f"(same_path={_fmt(sp_ppl, '.4f')} gap={_fmt(gap, '.4f')})" if gap is not None else ""
        print(f"same_path:      {sp_verdict}  {gap_note}", flush=True)

    print("", flush=True)
    tps = ev.get("tps_single_stream_a10g")
    print(f"tps:            {_fmt(tps, '.2f')} tok/s  [LOCAL a10g single-stream — exploratory, NOT official a10g-small]", flush=True)
    if ev.get("failures"):
        print(f"failures:       {len(ev['failures'])} (see evidence.json)", flush=True)
        for f in ev["failures"]:
            print(f"  - {f}", flush=True)
    print(f"evidence:       {out_dir / 'evidence.json'}", flush=True)
    print("=" * len(line) + "\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
