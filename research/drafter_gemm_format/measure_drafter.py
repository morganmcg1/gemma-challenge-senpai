"""PR #786 — measure the bi0 MTP drafter serving cost and the effect of
drafter quantization.

One invocation = one config. Starts the int4_mtp_bi0_surgattn serve stack with
the given extra env (e.g. SPECULATIVE_QUANTIZATION=fp8), drives a short conc=1
decode, and records:

  * wall TPS (decode-capture summary + warm probe),
  * drafter propose vs verify execute_model GPU ms (STEPTIME probe -> the
    drafter/verifier pass-cost ratio the PR asks for),
  * spec-decode acceptance rate + accepted tokens/step (/metrics),
  * the decode token-id file, and the official greedy-identity verdict vs a
    reference decode (the control), if one is given.

LOCAL A10G exploratory probe; not the official a10g-small TPS. No HF Job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths, serve_profile  # noqa: E402

OUT_DIR = ROOT / "research" / "drafter_gemm_format" / "runs"


def _get_text(url: str, timeout_s: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return r.read().decode("utf-8", "replace")


def _log_wandb(label: str, group: str, config: dict, summary: dict) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})", flush=True)
        return None
    try:
        run = wandb.init(
            entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
            group=group, name=f"stark/drafter-{label}", job_type="drafter-gemm-format",
            config=config,
        )
        flat = {}
        for k, v in summary.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                flat[k] = v
        run.summary.update(flat)
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid}", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] failed ({exc})", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--submission", default=str(ROOT / "submissions" / "int4_mtp_bi0_surgattn"))
    ap.add_argument("--server-python", default="/senpai-run/home/student-stark/.venvs/vllm022/bin/python")
    ap.add_argument("--spec-quant", default="", help="SPECULATIVE_QUANTIZATION value ('' = bf16 control)")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--reference", default="", help="control decode_outputs.jsonl for identity compare")
    ap.add_argument("--do-ppl", action="store_true")
    ap.add_argument("--wandb-group", default="bi0-drafter-gemm")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)

    submission = Path(args.submission).resolve()
    server_python = Path(args.server_python)
    out_dir = OUT_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    decode_jsonl = out_dir / "decode_outputs.jsonl"
    decode_summary = out_dir / "decode_summary.json"

    extra_env = {
        "STEPTIME": "1",
        "STEPTIME_WARMUP_SKIP": "32",
        "STEPTIME_RAW_START": "32",
        "STEPTIME_RAW_COUNT": "2000",
        "STEPTIME_REPORT_EVERY": "100000",
        "DISABLE_LOG_STATS": "0",   # re-enable Prometheus spec_decode_* counters
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    if args.spec_quant:
        extra_env["SPECULATIVE_QUANTIZATION"] = args.spec_quant

    result: dict = {
        "label": args.label,
        "submission": str(submission),
        "spec_quant": args.spec_quant or "bf16(control)",
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra_env": extra_env,
    }

    t0 = time.time()
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=8000,
            log_path=log_path, extra_env=extra_env, startup_timeout_s=1200,
        ) as srv:
            result["startup_s"] = time.time() - t0
            result["served_model_name"] = srv.served_model_name
            # 1) warm single-stream TPS probe
            try:
                result["tps_probe"] = harness.probe_tps(srv.base_url, srv.served_model_name)
            except Exception as exc:  # noqa: BLE001
                result["tps_probe"] = {"error": str(exc)}
            # 2) official decode capture (token ids for identity + steady tps)
            dt0 = time.time()
            result["decode_summary"] = harness.capture_decode(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=decode_jsonl, summary_file=decode_summary,
                num_prompts=args.num_prompts, output_len=args.output_len, timeout_s=3600,
            )
            result["decode_wall_s"] = time.time() - dt0
            # 3) spec-decode acceptance from Prometheus
            try:
                result["spec_metrics"] = serve_profile.parse_spec_metrics(
                    _get_text(f"{srv.base_url}/metrics")
                )
            except Exception as exc:  # noqa: BLE001
                result["spec_metrics"] = {"error": str(exc)}
            # 4) optional PPL (target-only path; drafter-invariant — run only to certify)
            if args.do_ppl:
                try:
                    result["ppl_summary"] = harness.run_ppl(
                        server_python, base_url=srv.base_url, model=srv.served_model_name,
                        out_file=out_dir / "ppl_outputs.jsonl",
                        summary_file=out_dir / "ppl_summary.json",
                    )
                except Exception as exc:  # noqa: BLE001
                    result["ppl_summary"] = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result["server_started"] = False
        result["error"] = str(exc)
        tail = ""
        try:
            tail = "\n".join(log_path.read_text().splitlines()[-40:])
        except OSError:
            pass
        result["server_log_tail"] = tail
        print(f"[measure] server failed to start: {exc}\n--- log tail ---\n{tail}", flush=True)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        _log_wandb(args.label, args.wandb_group,
                   {"spec_quant": result["spec_quant"], "server_started": False}, result)
        return 0
    result["server_started"] = True

    # STEPTIME parse -> drafter / verify GPU ms + ratio
    steptime = serve_profile.parse_steptime(log_path.read_text())
    result["steptime"] = steptime
    verify_ms = steptime.get("verify_gpu_ms")
    draft_ms = steptime.get("drafter_gpu_ms")
    result["verify_gpu_ms"] = verify_ms
    result["drafter_gpu_ms"] = draft_ms
    if verify_ms and draft_ms is not None:
        result["drafter_over_verify_ratio"] = draft_ms / verify_ms
        result["drafter_frac_of_gpu_busy"] = draft_ms / (draft_ms + verify_ms)

    # Greedy identity vs the control reference (official verifier)
    if args.reference and Path(args.reference).exists():
        try:
            report = greedy_gate.compare(Path(args.reference), decode_jsonl)
            result["greedy_verdict"] = report.verdict
            result["greedy_num_identical"] = report.num_identical
            result["greedy_num_divergent"] = report.num_divergent
            result["greedy_total_divergent_tokens"] = report.total_divergent_tokens
        except Exception as exc:  # noqa: BLE001
            result["greedy_verdict"] = f"ERROR: {exc}"

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    # ---- console summary ----
    ds = result.get("decode_summary", {})
    sm = result.get("spec_metrics", {})
    print("\n========== DRAFTER MEASURE: %s ==========" % args.label, flush=True)
    print(f"spec_quant            = {result['spec_quant']}", flush=True)
    print(f"startup_s             = {result.get('startup_s'):.0f}", flush=True)
    print(f"decode tps (capture)  = {ds.get('tps') or ds.get('output_tps')}", flush=True)
    print(f"warm probe decode tps = {result.get('tps_probe', {}).get('decode_tps_single_stream')}", flush=True)
    print(f"completed             = {ds.get('completed') or ds.get('num_completed')}", flush=True)
    print(f"verify_gpu_ms (p50)   = {verify_ms}", flush=True)
    print(f"drafter_gpu_ms (p50)  = {draft_ms}", flush=True)
    print(f"drafter/verify ratio  = {result.get('drafter_over_verify_ratio')}", flush=True)
    print(f"draft accept rate     = {sm.get('draft_acceptance_rate')}", flush=True)
    print(f"E_accept (len)        = {sm.get('e_accept_mean_acceptance_length')}", flush=True)
    print(f"greedy verdict        = {result.get('greedy_verdict')}", flush=True)

    cfg = {"spec_quant": result["spec_quant"], "num_prompts": args.num_prompts,
           "output_len": args.output_len, "server_started": True}
    summ = {
        "decode_tps_capture": ds.get("tps") or ds.get("output_tps"),
        "warm_probe_decode_tps": result.get("tps_probe", {}).get("decode_tps_single_stream"),
        "completed": ds.get("completed") or ds.get("num_completed"),
        "verify_gpu_ms": verify_ms,
        "drafter_gpu_ms": draft_ms,
        "drafter_over_verify_ratio": result.get("drafter_over_verify_ratio"),
        "drafter_frac_of_gpu_busy": result.get("drafter_frac_of_gpu_busy"),
        "draft_acceptance_rate": sm.get("draft_acceptance_rate"),
        "e_accept_mean_acceptance_length": sm.get("e_accept_mean_acceptance_length"),
        "num_accepted_tokens": sm.get("num_accepted_tokens"),
        "num_draft_tokens": sm.get("num_draft_tokens"),
        "greedy_verdict": result.get("greedy_verdict"),
        "ppl": (result.get("ppl_summary") or {}).get("ppl"),
    }
    rid = _log_wandb(args.label, args.wandb_group, cfg, summ)
    result["wandb_run_id"] = rid
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[measure] artifacts -> {out_dir}  wandb={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
