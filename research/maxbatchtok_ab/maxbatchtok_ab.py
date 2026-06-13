"""Single-variable served A/B of ``max_num_batched_tokens`` on the deployed
split-KV #1 stack (``submissions/fa2sw_precache_kenyan``), PR #56.

Reuses the *exact* canonical timing methodology that produced the 428.37 local
steady-state baseline (``serve_profile.run_timing_pass`` -> vLLM's own
per-interval "Avg generation throughput" meter, ``parse_spec_log`` ->
``steady_gen_tps_mean``). The ONLY variable swept is the
``MAX_NUM_BATCHED_TOKENS`` env (serve.py -> ``--max-num-batched-tokens``); every
other served arg, the split-KV patch, the drafter, lmhead12k, fa2sw, onegraph,
and precache (locally ungated: PRECACHE_DATASET absent -> identical for all arms)
are held fixed at the deployed manifest values.

Per arm, in ONE server session:
  1. capture_decode (128 prompts x 512 tok, seed 1)  -> completion + decode jsonl
  2. snapshot the server log -> steady_gen_tps_mean   (decode-phase isolated:
     read BEFORE the PPL pass, which emits no generation-throughput intervals)
  3. run_ppl (official ppl_endpoint vs GT tokens)     -> ppl (<= 2.42 gate)
  4. after teardown, scan the full server log for the split-KV guardrail
     (M=8 verify -> 3D split-KV redirect, zero 2D fallback), the
     max_num_batched_tokens startup warning, the launched serve cmdline, and the
     attention backend.

Token-neutrality cross-check: each arm's served decode tokens are greedy-compared
against the CONTROL (512) arm's tokens (a scheduling knob must not move tokens)
AND against the submission's canonical M=1 AR reference (for the record).

Local AWS A10G exploratory probe — NOT the official a10g-small TPS.

    /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
        research/maxbatchtok_ab/maxbatchtok_ab.py --arms 512,2048,4096,8192
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths, serve_profile  # noqa: E402
from scripts.local_validation.ppl_runner import _headroom_overrides  # noqa: E402

SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
SERVER_PYTHON = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")
CONTROL_ARM = "512"  # the deployed manifest value (MAX_NUM_BATCHED_TOKENS=512)

# Server-log signatures for the split-KV guardrail + the swept-knob warning.
_REDIRECT_RE = re.compile(r"\[splitkv-verify\] verify batch M=(\d+) .*-> 3D split-KV")
_WRAPPED_RE = re.compile(r"\[splitkv-verify\] wrapped unified_attention")
_FALLBACK_RE = re.compile(r"\[splitkv-verify\] (redirect skipped|patch error)")
_LAUNCH_RE = re.compile(r"\[serve\] launching: (.+)")
_MAXBATCH_WARN_RE = re.compile(r".*max_num_batched_tokens.*", re.IGNORECASE)
_BACKEND_RE = re.compile(r"Using\s+(\w+)\s+backend|attention backend.*?(\w+_ATTN)", re.IGNORECASE)


def _scan_server_log(log_text: str) -> dict:
    """Extract the split-KV guardrail signals + the max_num_batched_tokens
    warning + the launched serve cmdline from one arm's server log."""
    redirects = _REDIRECT_RE.findall(log_text)
    fallbacks = _FALLBACK_RE.findall(log_text)
    launch = _LAUNCH_RE.search(log_text)
    launch_cmd = launch.group(1) if launch else None
    served_maxbatch = None
    if launch_cmd:
        m = re.search(r"--max-num-batched-tokens\s+(\d+)", launch_cmd)
        served_maxbatch = int(m.group(1)) if m else None
    # The startup warning(s) that mention the swept knob (dedup, drop the bench
    # noise lines that merely echo the env in the launch cmd).
    warn_lines = []
    for ln in log_text.splitlines():
        if "max_num_batched_tokens" in ln.lower() and "[serve] launching" not in ln:
            warn_lines.append(ln.strip())
    # Keep distinct warning texts only.
    seen, distinct_warns = set(), []
    for ln in warn_lines:
        key = re.sub(r"\d+", "N", ln)[-200:]
        if key not in seen:
            seen.add(key)
            distinct_warns.append(ln[:400])
    engine_oom = bool(re.search(r"OutOfMemoryError|CUDA out of memory", log_text))
    engine_dead = "EngineDeadError" in log_text
    return {
        "splitkv_wrapped": bool(_WRAPPED_RE.search(log_text)),
        "splitkv_redirect_count": len(redirects),
        "splitkv_redirect_M_values": sorted({int(x) for x in redirects}),
        "splitkv_fallback_count": len(fallbacks),
        "splitkv_engaged": bool(_WRAPPED_RE.search(log_text)) and len(redirects) > 0 and len(fallbacks) == 0,
        "served_max_num_batched_tokens": served_maxbatch,
        "maxbatchtok_warning_lines": distinct_warns,
        "maxbatchtok_warned": len(distinct_warns) > 0,
        "engine_oom": engine_oom,
        "engine_dead": engine_dead,
        "launch_cmd": launch_cmd,
    }


def _greedy(reference: Path, candidate: Path) -> dict:
    report = greedy_gate.compare(reference, candidate)
    onset = greedy_gate.onset_summary(report)
    return {
        "verdict": report.verdict,
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_divergent_tokens": report.total_divergent_tokens,
        "onset_median": onset.get("onset_median"),
        "identical_frac": (report.num_identical / report.num_prompts_compared
                           if report.num_prompts_compared else None),
    }


def run_arm(arm: str, out_dir: Path, *, num_prompts: int, output_len: int,
            reference: Path, control_decode: Path | None) -> dict:
    """Serve the deployed stack with MAX_NUM_BATCHED_TOKENS=arm; measure steady
    TPS + PPL + completion + split-KV-engaged in ONE session."""
    label = f"mbt{arm}"
    log_path = out_dir / f"server_{label}.log"
    decode_out = out_dir / f"decode_{label}.jsonl"
    decode_sum = out_dir / f"decode_{label}.summary.json"
    ppl_out = out_dir / f"ppl_{label}.jsonl"
    ppl_sum = out_dir / f"ppl_{label}.summary.json"

    expected_steps = max(64, num_prompts * output_len // 2)
    # The canonical timing env (serve_profile.run_timing_pass): re-enable vLLM's
    # stat logger (manifest ships --disable-log-stats) so the per-interval
    # "Avg generation throughput" meter is emitted, plus STEPTIME for verify ms.
    # PPL headroom overrides match validate_submission. extra_env wins over the
    # manifest, so MAX_NUM_BATCHED_TOKENS is the single swept variable.
    env = {
        **serve_profile._steptime_env(expected_steps),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
        **_headroom_overrides(harness.load_manifest(SUBMISSION).get("env", {})),
        "MAX_NUM_BATCHED_TOKENS": str(arm),
    }
    res: dict = {"arm": arm, "label": label}
    t_arm = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PYTHON, port=8000,
        log_path=log_path, extra_env=env, startup_timeout_s=1800,
    ) as srv:
        # 1) decode capture (completion + token IDs)
        t0 = time.time()
        decode_summary = harness.capture_decode(
            SERVER_PYTHON, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_sum,
            num_prompts=num_prompts, output_len=output_len, timeout_s=3600,
        )
        res["decode_wall_s"] = time.time() - t0
        res["completed"] = decode_summary["num_records"]
        res["num_completion_tokens"] = decode_summary["num_completion_tokens"]
        res["decode_duration_s"] = decode_summary["duration_s"]
        res["wallclock_tps"] = (decode_summary["num_completion_tokens"]
                                / decode_summary["duration_s"]
                                if decode_summary.get("duration_s") else None)
        # 2) steady-state TPS from the DECODE-PHASE log snapshot (read before PPL,
        #    which emits no generation-throughput intervals and would dilute it).
        decode_log = log_path.read_text()
        spec_log = serve_profile.parse_spec_log(decode_log)
        res["steady_gen_tps_mean"] = spec_log.get("steady_gen_tps_mean")
        res["steady_gen_tps_n"] = spec_log.get("steady_gen_tps_n")
        res["e_accept_exact"] = spec_log.get("e_accept_exact")
        # 3) PPL (official ppl_endpoint vs GT tokens) — the <= 2.42 gate.
        #    Non-fatal: at MAX_NUM_BATCHED_TOKENS > 512 the prompt_logprobs
        #    float32 log_softmax over the larger prefill chunk OOMs the engine at
        #    GPU_MEMORY_UTILIZATION=0.90 (EngineDeadError). Capture that as data
        #    (an invalid arm) instead of discarding the clean decode-phase TPS
        #    that was already snapshotted above (step 2, before this PPL pass).
        try:
            ppl_summary = harness.run_ppl(
                SERVER_PYTHON, base_url=srv.base_url, model=srv.served_model_name,
                out_file=ppl_out, summary_file=ppl_sum, timeout_s=1800,
            )
            res["ppl"] = ppl_summary["ppl"]
            res["ppl_num_tokens"] = ppl_summary["num_tokens"]
        except Exception as exc:  # noqa: BLE001
            res["ppl"] = None
            res["ppl_error"] = repr(exc)[:300]
            print(f"[mbt-ab] arm {arm}: PPL pass FAILED (non-fatal): {exc!r}", flush=True)

    res["arm_wall_s"] = time.time() - t_arm
    # 4) split-KV guardrail + warning + cmdline from the full server log.
    res.update(_scan_server_log(log_path.read_text()))
    # token-neutrality + reference greedy verdict
    res["greedy_vs_reference"] = _greedy(reference, decode_out)
    if control_decode is not None and control_decode != decode_out:
        res["greedy_vs_control"] = _greedy(control_decode, decode_out)
    res["server_log"] = str(log_path)
    res["decode_jsonl"] = str(decode_out)
    return res


def _log_wandb(arm_res: dict, group: str) -> str | None:
    # Fully non-fatal: a wandb import/version/network failure must never discard
    # an expensive completed arm. The serve venv has no `wandb` (the only thing on
    # sys.path is the local ./wandb run-logs dir, an empty namespace package), so
    # run the orchestrator under a python that ships wandb (repo .venv).
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary

        run = init_wandb_run(
            job_type="maxbatchtok-ab",
            agent="lawine",
            name=f"lawine/maxbatchtok-{arm_res['arm']}",
            tags=[group],
            config={
                "submission": "fa2sw_precache_kenyan",
                "max_num_batched_tokens": int(arm_res["arm"]),
                "is_control": arm_res["arm"] == CONTROL_ARM,
                "wandb_group": group,
                "workload": "128 prompts x 512 tok, conc=1, seed 1, no-precache (local)",
            },
        )
        if run is None:
            return None
        summary = {k: arm_res[k] for k in (
            "steady_gen_tps_mean", "wallclock_tps", "ppl", "completed",
            "num_completion_tokens", "decode_duration_s", "e_accept_exact",
            "splitkv_engaged", "splitkv_redirect_count", "splitkv_fallback_count",
            "served_max_num_batched_tokens", "maxbatchtok_warned", "engine_oom",
            "arm_wall_s",
        ) if k in arm_res}
        summary["splitkv_engaged"] = int(bool(arm_res.get("splitkv_engaged")))
        summary["maxbatchtok_warned"] = int(bool(arm_res.get("maxbatchtok_warned")))
        summary["engine_oom"] = int(bool(arm_res.get("engine_oom")))
        _ppl = arm_res.get("ppl")
        summary["ppl_valid"] = int(_ppl is not None)
        summary["ppl_under_cap"] = int(_ppl is not None and _ppl <= 2.42)
        log_summary(run, summary, step=0)
        rid = run.id
        finish_wandb(run)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] logging failed (non-fatal): {exc!r}", flush=True)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="512,2048,4096,8192",
                    help="comma list of max_num_batched_tokens values; 512 = control")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "research" / "maxbatchtok_ab")
    ap.add_argument("--wandb-group", default="maxbatchtok-served-ab")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[mbt-ab] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    ref_id = harness.reference_identity(harness.serve_model_id(manifest, SUBMISSION), SUBMISSION)
    reference = greedy_gate.reference_for(ref_id)
    print(f"[mbt-ab] reference={reference} exists={Path(reference).exists()}", flush=True)
    print(f"[mbt-ab] manifest MAX_NUM_BATCHED_TOKENS={manifest['env'].get('MAX_NUM_BATCHED_TOKENS')} "
          f"MAX_MODEL_LEN={manifest['env'].get('MAX_MODEL_LEN')}", flush=True)
    print(f"[mbt-ab] arms={arms} workload={args.num_prompts}x{args.output_len} conc=1 seed={paths.SEED}", flush=True)

    result = {
        "submission": str(SUBMISSION),
        "control_arm": CONTROL_ARM,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "reference": str(reference),
        "manifest_max_num_batched_tokens": manifest["env"].get("MAX_NUM_BATCHED_TOKENS"),
        "manifest_max_model_len": manifest["env"].get("MAX_MODEL_LEN"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {},
        "wandb_runs": {},
    }
    control_decode = out_dir / f"decode_mbt{CONTROL_ARM}.jsonl"
    # Run the control first so token-neutrality compares are anchored on it.
    ordered = ([CONTROL_ARM] if CONTROL_ARM in arms else []) + [a for a in arms if a != CONTROL_ARM]
    for arm in ordered:
        print(f"\n===== ARM max_num_batched_tokens={arm} "
              f"({'CONTROL' if arm == CONTROL_ARM else 'variant'}) =====", flush=True)
        try:
            arm_res = run_arm(
                arm, out_dir, num_prompts=args.num_prompts, output_len=args.output_len,
                reference=Path(reference),
                control_decode=control_decode if arm != CONTROL_ARM else None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[mbt-ab] ARM {arm} FAILED: {exc!r}", flush=True)
            result["arms"][arm] = {"arm": arm, "error": repr(exc)}
            (out_dir / "maxbatchtok_ab_result.json").write_text(json.dumps(result, indent=2))
            continue
        if not args.no_wandb:
            arm_res["wandb_run_id"] = _log_wandb(arm_res, args.wandb_group)
            result["wandb_runs"][arm] = arm_res["wandb_run_id"]
        result["arms"][arm] = arm_res
        (out_dir / "maxbatchtok_ab_result.json").write_text(json.dumps(result, indent=2))
        g = arm_res.get("greedy_vs_control") or {}
        print(f"[mbt-ab] arm={arm}: steady_tps={arm_res.get('steady_gen_tps_mean')} "
              f"wallclock_tps={arm_res.get('wallclock_tps')} ppl={arm_res.get('ppl')} "
              f"completed={arm_res.get('completed')} splitkv_engaged={arm_res.get('splitkv_engaged')} "
              f"(redirects={arm_res.get('splitkv_redirect_count')} fallbacks={arm_res.get('splitkv_fallback_count')}) "
              f"warned={arm_res.get('maxbatchtok_warned')} "
              f"vs_control_identical={g.get('num_identical')}/{g.get('num_prompts_compared')}", flush=True)

    # A/B deltas vs control.
    arms_d = result["arms"]
    ctrl = arms_d.get(CONTROL_ARM, {})
    ctrl_tps = ctrl.get("steady_gen_tps_mean")
    if ctrl_tps:
        for arm, r in arms_d.items():
            t = r.get("steady_gen_tps_mean")
            if t:
                r["steady_tps_delta_vs_control"] = t - ctrl_tps
                r["steady_tps_frac_vs_control"] = (t - ctrl_tps) / ctrl_tps
    (out_dir / "maxbatchtok_ab_result.json").write_text(json.dumps(result, indent=2))

    # Summary table.
    print("\n[mbt-ab] ===== A/B SUMMARY (steady_gen_tps_mean; baseline 428.37 = #43 split-KV) =====", flush=True)
    print(f"{'mbt':>6} {'steady_tps':>11} {'Δvs512%':>8} {'wall_tps':>9} {'ppl':>7} "
          f"{'compl':>6} {'splitkv':>7} {'warned':>6}", flush=True)
    for arm in ordered:
        r = arms_d.get(arm, {})
        if "error" in r:
            print(f"{arm:>6} ERROR {r['error']}", flush=True)
            continue
        frac = r.get("steady_tps_frac_vs_control")
        print(f"{arm:>6} {(r.get('steady_gen_tps_mean') or 0):>11.2f} "
              f"{(100*frac if frac is not None else 0):>8.2f} "
              f"{(r.get('wallclock_tps') or 0):>9.2f} {(r.get('ppl') or 0):>7.4f} "
              f"{str(r.get('completed')):>6} {str(r.get('splitkv_engaged')):>7} "
              f"{str(r.get('maxbatchtok_warned')):>6}", flush=True)
    print(f"\n[mbt-ab] artifacts -> {out_dir}/maxbatchtok_ab_result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
