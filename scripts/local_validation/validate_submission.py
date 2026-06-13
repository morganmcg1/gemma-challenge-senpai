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

from . import greedy_gate, harness, modalities_probe, paths, same_path_ppl
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
