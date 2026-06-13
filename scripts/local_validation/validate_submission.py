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
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import greedy_gate, harness, paths, same_path_ppl
from .ppl_runner import _headroom_overrides

# Default same-path-vs-prompt_logprobs PPL gap (nats of mean log-likelihood,
# expressed as a PPL delta) above which a submission is treated as a
# timed-vs-scored path split. Justified in research/validity/same_path_ppl.md:
# the honest baseline agrees to < 0.02 (FP noise), so 0.05 is a ~2.5x margin —
# wide enough to never flag honest FP/quantization jitter, tight enough that the
# 2.378 (prompt_logprobs) vs 2.55 (same-path) LF29 lane gap of ~0.17 trips it.
DEFAULT_SAME_PATH_THRESHOLD = 0.05


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
    ap.add_argument("--tps-tokens", type=int, default=256)
    ap.add_argument(
        "--check-same-path",
        action="store_true",
        help="also score PPL through the timed generation path (echo+logprobs) and FAIL "
        "(non-zero exit) if it diverges from the prompt_logprobs PPL by more than the threshold",
    )
    ap.add_argument("--same-path-threshold", type=float, default=DEFAULT_SAME_PATH_THRESHOLD,
                    help="max allowed |same_path_ppl - prompt_logprobs_ppl| before the gate fails")
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

                reference = args.reference or greedy_gate.reference_for(srv.reference_model_id)
                if not Path(reference).exists():
                    msg = (f"greedy reference missing: {reference} — generate it for THIS submission with "
                           f"gen_greedy_reference --mode served --submission {submission} [--spec-off] "
                           f"--num-prompts {args.num_prompts}")
                    evidence["failures"].append(msg)
                    evidence["greedy_verdict"] = "NO_REFERENCE"
                    print(f"[validate] WARN {msg}", flush=True)
                else:
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
        )
        if key in evidence
    }
    log_summary(run, summary, step=0)
    finish_wandb(run)


def _fmt(v, spec="") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def _print_block(ev: dict, out_dir: Path) -> None:
    name = ev.get("submission_name", "?")
    line = "=" * 16 + f" LOCAL VALIDATION — {name} " + "=" * 16
    print("\n" + line, flush=True)
    print(f"submission:     {ev.get('submission')}", flush=True)
    print(f"model_id:       {ev.get('model_id')}", flush=True)
    rk = ev.get("greedy_reference_kind")
    rk_note = f"  [ref: {rk}]" if rk else ""
    print(f"greedy_verdict: {ev.get('greedy_verdict', 'skipped')}{rk_note}", flush=True)
    onset = ev.get("greedy_onset")
    if onset and onset.get("num_divergent"):
        print(f"                {greedy_gate.onset_line(onset, ev.get('output_len'))}", flush=True)
    ppl = ev.get("ppl")
    cap_note = "<= 2.42 cap" if isinstance(ppl, (int, float)) and ppl <= 2.42 else "OVER 2.42 CAP" if isinstance(ppl, (int, float)) else ""
    print(f"ppl:            {_fmt(ppl, '.4f')}   {cap_note}", flush=True)
    sp_verdict = ev.get("same_path_verdict")
    if sp_verdict:
        sp_ppl = ev.get("same_path_ppl")
        gap = ev.get("same_path_gap")
        gap_note = f"(same_path={_fmt(sp_ppl, '.4f')} gap={_fmt(gap, '.4f')})" if gap is not None else ""
        print(f"same_path:      {sp_verdict}  {gap_note}", flush=True)
    comp = ev.get("completed")
    print(f"completed:      {comp}/{ev.get('num_prompts')}" if comp is not None else "completed:      n/a", flush=True)
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
