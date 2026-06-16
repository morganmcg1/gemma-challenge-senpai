#!/usr/bin/env python3
"""Floor-lock full-serve validation (PR #485) — LOCAL only, analysis_only.

Serve ``submissions/fa2sw_strict_m1ar_int4`` (the strict M=1 AR int4 floor-lock)
ONCE on the local A10G and drive the *official* ``summary.json:tps`` path
end-to-end, exactly as ``hf_bucket_single_job.py`` does inside an HF Job (minus
the bucket plumbing), to answer two questions the modeled 161.70 cannot:

  1. does the floor-lock REALIZE ~161.70 official TPS on a real e2e serve, and
  2. is it genuinely LITERAL-1.0 (token-identical to plain greedy AR)?

Stages (server is served once, all stages share the live endpoint):
  * sglang.bench_serving (vllm-chat, 128x512, conc=1, request-rate inf, warmup 4,
    seed 1, ignore_eos) -> summary.json:tps = output_throughput. PRECACHE is
    enabled locally (PRECACHE_DATASET -> the local eval-prompts mirror) so the
    measured stack matches the precache-ON deployed config the 161.70 models.
  * decode capture (official decode_outputs.py) -> wall_tps = num_completion_tokens
    / decode_duration_s. This is the meter the local->official multiplier
    (x1.0602, #99) is calibrated on (anchor 454.338 local wall_tps -> 481.53
    official sglang tps), so projected_official = wall_tps * 1.0602.
  * greedy gate vs the precache_kenyan served_spec_off M=1 AR reference ->
    token_identity_rate (prompt- and token-level) + verdict.
  * PPL (official ppl_endpoint.py) <= 2.42.
  * modalities probe (text/image/audio/video).

Verdicts (logged to W&B group ``floorlock-fullserve-validate``):
  * floorlock_realizes_16170 : projected_official within / above 161.70 - sigma_hw.
  * floorlock_literal_1p0    : greedy gate GREEDY_IDENTICAL (prompt identity 1.0).

NO --launch, NO draw, NO submission. Nothing here uploads or scores remotely.

Example::
    .venv/bin/python -m research.validity.floorlock_fullserve_validate.run_fullserve \
        --wandb-name stark/floorlock-fullserve-validate
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, modalities_probe, paths  # noqa: E402

OUT_ROOT = ROOT / "research" / "validity" / "floorlock_fullserve_validate"

# --- official measurement protocol (mirrors hf_bucket_single_job.py exactly) ---
TOKENIZER = "google/gemma-4-E4B-it"
MAX_CONCURRENCY = 1
REQUEST_RATE = "inf"
WARMUP_REQUESTS = 4
BENCH_DEPS = [
    "sglang==0.5.2",
    "transformers==5.9.0",
    "jinja2==3.1.6",
    "pybase64==1.4.3",
    "pydantic==2.13.4",
]

# --- local -> official transfer (lawine #438 / #99; local_official_projection.py) ---
OFFICIAL_FLOOR_MODEL = 161.70   # modeled floor-lock official TPS (lawine #438)
LOCAL_OFFICIAL_MULTIPLIER = 1.0602  # 481.53 official sglang / 454.338 local wall_tps (#99)
SIGMA_HW = 4.864               # hardware sigma on the official scale

# Default M=1 AR reference: precache_kenyan served_spec_off (byte-equivalent stack
# to the floor-lock; spec-off == plain greedy AR). 128 records, seed 1, len 512.
DEFAULT_REFERENCE = (
    paths.REFERENCE_ROOT
    / "workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it"
    / "decode_outputs.jsonl"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# GPU preflight (clean serving slot before LocalServer.__enter__)
# --------------------------------------------------------------------------- #
def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def preflight_gpu(mem_threshold_mib: int = 1500, timeout_s: int = 180) -> None:
    patterns = ["vllm.entrypoints.openai.api_server", "VLLM::EngineCore",
                "multiprocessing.resource_tracker"]
    reaped = False
    for pat in patterns:
        r = subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        reaped = reaped or (r.returncode == 0)
    if reaped:
        print("[floorlock] preflight: reaped lingering vLLM process(es)", flush=True)
        time.sleep(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_mem_used_mib()
        if used is None or used < mem_threshold_mib:
            if used is not None:
                print(f"[floorlock] preflight: GPU free ({used} MiB used)", flush=True)
            return
        time.sleep(3)
    print(f"[floorlock] preflight: WARN GPU still busy after {timeout_s}s "
          f"({_gpu_mem_used_mib()} MiB)", flush=True)


# --------------------------------------------------------------------------- #
# Stage 0: cheap liveness self-test (fast guard before the long stages)
# --------------------------------------------------------------------------- #
def self_test(base_url: str, model: str, timeout_s: float = 120.0) -> dict[str, Any]:
    payload = {"model": model, "prompt": "The capital of France is",
               "max_tokens": 8, "temperature": 0.0, "stream": False, "ignore_eos": True}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.loads(r.read().decode())
    text = ((resp.get("choices") or [{}])[0].get("text") or "")
    return {"ok": bool(text.strip()), "wall_s": time.time() - t0, "sample": text.strip()[:60]}


# --------------------------------------------------------------------------- #
# Stage 1: official sglang.bench_serving -> summary.json:tps
# (faithful copy of hf_bucket_single_job.run_benchmark + write_summary)
# --------------------------------------------------------------------------- #
def run_sglang_bench(
    bench_python: Path,
    *,
    base_url: str,
    model: str,
    dataset_path: Path,
    output_file: Path,
    num_prompts: int,
    output_len: int,
    seed: int,
    warmup_requests: int = WARMUP_REQUESTS,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bench_python), "-m", "sglang.bench_serving",
        "--backend", "vllm-chat",
        "--base-url", base_url.rstrip("/"),
        "--model", model,
        "--tokenizer", TOKENIZER,
        "--dataset-name", "sharegpt",
        "--dataset-path", str(dataset_path),
        "--sharegpt-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(MAX_CONCURRENCY),
        "--request-rate", REQUEST_RATE,
        "--warmup-requests", str(warmup_requests),
        "--seed", str(seed),
        "--extra-request-body", json.dumps({"ignore_eos": True}),
        "--output-file", str(output_file),
        "--output-details",
        "--disable-stream",
        "--disable-tqdm",
    ]
    print("[sglang]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, check=False, timeout=timeout_s).returncode
    if rc != 0:
        raise RuntimeError(f"sglang.bench_serving exited {rc}")
    result = json.loads(output_file.read_text().strip().splitlines()[-1])
    return result


def summarize_bench(result: dict[str, Any], *, model: str, num_prompts: int, output_len: int) -> dict[str, Any]:
    """Official summary fields (hf_bucket_single_job.write_summary)."""
    total_tps = (result["total_input_tokens"] + result["total_output_tokens"]) / result["duration"]
    return {
        "tps": result["output_throughput"],
        "output_tps": result["output_throughput"],
        "total_tps": total_tps,
        "completed": result["completed"],
        "duration_s": result["duration"],
        "total_input_tokens": result["total_input_tokens"],
        "total_output_tokens": result["total_output_tokens"],
        "request_throughput_req_s": result["request_throughput"],
        "mean_e2e_latency_ms": result.get("mean_e2e_latency_ms"),
        "p99_e2e_latency_ms": result.get("p99_e2e_latency_ms"),
        "max_concurrency": result.get("max_concurrency"),
        "num_prompts": num_prompts,
        "output_len": output_len,
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Stage 3: greedy gate -> token_identity_rate
# --------------------------------------------------------------------------- #
def greedy_identity(reference: Path, candidate: Path, output_len: int) -> dict[str, Any]:
    report = greedy_gate.compare(reference, candidate)
    n_cmp = report.num_prompts_compared
    prompt_rate = (report.num_identical / n_cmp) if n_cmp else None
    token_rate = (
        1.0 - (report.total_divergent_tokens / report.total_tokens_compared)
        if report.total_tokens_compared else None
    )
    onset = greedy_gate.onset_summary(report)
    return {
        "verdict": report.verdict,
        "num_prompts_compared": n_cmp,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "token_identity_rate_prompt": prompt_rate,
        "token_identity_rate_token": token_rate,
        "missing_in_candidate": report.missing_in_candidate[:10],
        "missing_in_reference": report.missing_in_reference[:10],
        "integrity_failures": report.integrity_failures[:10],
        "onset": onset,
        "onset_line": greedy_gate.onset_line(onset, output_len),
        "reference_kind": greedy_gate.reference_kind(reference),
        "reference_num_records": greedy_gate.reference_num_records(reference),
    }


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def compute_verdicts(*, wall_tps: float | None, sglang_tps: float | None,
                     identity: dict[str, Any] | None, ppl: float | None) -> dict[str, Any]:
    proj_wall = wall_tps * LOCAL_OFFICIAL_MULTIPLIER if wall_tps else None
    proj_sglang = sglang_tps * LOCAL_OFFICIAL_MULTIPLIER if sglang_tps else None
    floor = OFFICIAL_FLOOR_MODEL - SIGMA_HW
    realizes = bool(proj_wall is not None and proj_wall >= floor)
    # literal-1.0: the served decode IS plain greedy AR (no spec path), so an
    # exact GREEDY_IDENTICAL verdict (prompt identity 1.0) confirms it empirically.
    literal = bool(identity is not None and identity.get("verdict") == "GREEDY_IDENTICAL")
    return {
        "projected_official_from_wall_tps": proj_wall,
        "projected_official_from_sglang_tps": proj_sglang,
        "official_floor_model": OFFICIAL_FLOOR_MODEL,
        "sigma_hw": SIGMA_HW,
        "realizes_band_low": floor,
        "realizes_band_high": OFFICIAL_FLOOR_MODEL + SIGMA_HW,
        "delta_from_model_wall": (proj_wall - OFFICIAL_FLOOR_MODEL) if proj_wall else None,
        "delta_from_model_sglang": (proj_sglang - OFFICIAL_FLOOR_MODEL) if proj_sglang else None,
        "floorlock_realizes_16170": realizes,
        "floorlock_literal_1p0": literal,
        "ppl_ok": (None if ppl is None else bool(ppl <= modalities_probe.PPL_CAP)),
    }


# --------------------------------------------------------------------------- #
# W&B
# --------------------------------------------------------------------------- #
def log_wandb(args, evidence: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[floorlock] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="floorlock-fullserve-validate",
        agent="stark",
        name=args.wandb_name or "stark/floorlock-fullserve-validate",
        group=args.wandb_group,
        notes="PR #485 floor-lock full-serve validation (LOCAL, analysis_only)",
        tags=["validity", "floorlock", "fullserve", args.submission_name,
              "smoke" if args.smoke else "full"],
        config={
            "submission": args.submission_name,
            "smoke": args.smoke,
            "precache": not args.no_precache,
            "num_prompts": args.num_prompts,
            "output_len": args.output_len,
            "seed": args.seed,
            "reference": str(args.reference),
            "multiplier": LOCAL_OFFICIAL_MULTIPLIER,
            "official_floor_model": OFFICIAL_FLOOR_MODEL,
            "sigma_hw": SIGMA_HW,
            "analysis_only": True,
            "official_tps": 0,
        },
    )
    if run is None:
        print("[floorlock] wandb disabled (no key / WANDB_DISABLED)", flush=True)
        return None
    try:
        v = evidence.get("verdicts", {})
        bench = evidence.get("bench", {})
        decode = evidence.get("decode", {})
        ident = evidence.get("identity", {})
        flat: dict[str, Any] = {
            "tps/sglang_official_local": bench.get("tps"),
            "tps/sglang_completed": bench.get("completed"),
            "tps/wall_tps_local": decode.get("wall_tps"),
            "tps/projected_official_from_wall": v.get("projected_official_from_wall_tps"),
            "tps/projected_official_from_sglang": v.get("projected_official_from_sglang_tps"),
            "tps/delta_from_model_wall": v.get("delta_from_model_wall"),
            "identity/prompt_rate": ident.get("token_identity_rate_prompt"),
            "identity/token_rate": ident.get("token_identity_rate_token"),
            "identity/num_identical": ident.get("num_identical"),
            "identity/num_divergent": ident.get("num_divergent"),
            "ppl/ppl": evidence.get("ppl", {}).get("ppl"),
            "ppl/num_tokens": evidence.get("ppl", {}).get("num_tokens"),
            "verdict/realizes_16170": int(bool(v.get("floorlock_realizes_16170"))),
            "verdict/literal_1p0": int(bool(v.get("floorlock_literal_1p0"))),
            "verdict/ppl_ok": (None if v.get("ppl_ok") is None else int(bool(v.get("ppl_ok")))),
        }
        flat = {k: val for k, val in flat.items() if isinstance(val, (int, float))}
        wandb_logging.log_event(run, "floorlock_fullserve", step=0, metrics=flat)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="floorlock_fullserve_evidence",
            artifact_type="floorlock-fullserve", data=evidence,
        )
        return getattr(run, "id", None)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default="fa2sw_strict_m1ar_int4")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny workload (2x16), precache-off, skip PPL — pipeline self-test")
    ap.add_argument("--no-precache", action="store_true",
                    help="do NOT point PRECACHE_DATASET at the local eval prompts (precache stays off)")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="floorlock-fullserve-validate")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 2)
        args.output_len = min(args.output_len, 16)
        args.no_precache = True
        args.skip_ppl = True

    args.submission_name = args.submission
    for note in paths.prepare_local_gpu_env():
        print(f"[floorlock] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    bench_python = harness.ensure_server_venv(BENCH_DEPS)
    print(f"[floorlock] submission={submission_dir.name}", flush=True)
    print(f"[floorlock] server_python={server_python}", flush=True)
    print(f"[floorlock] bench_python={bench_python}", flush=True)

    tag = "smoke" if args.smoke else "full"
    out_dir = (args.out_dir or (OUT_ROOT / f"run_{tag}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"

    # Precache: point PRECACHE_DATASET at the local eval-prompts mirror so the
    # served stack precaches exactly the bench prompts (matches the precache-ON
    # official deployed config the 161.70 models). Absent override -> serve_patch
    # _precache skips precache (the /harness path doesn't exist locally).
    extra_env: dict[str, str] = {}
    if not args.no_precache:
        extra_env["PRECACHE_DATASET"] = str(paths.EVAL_PROMPTS)

    evidence: dict[str, Any] = {
        "pr": 485,
        "started_utc": _now(),
        "mode": tag,
        "analysis_only": True,
        "official_tps": 0,
        "submission": args.submission,
        "submission_dir": str(submission_dir),
        "manifest_name": manifest.get("name"),
        "spec_off": manifest.get("env", {}).get("SPECULATIVE_CONFIG") == "",
        "max_num_seqs": manifest.get("env", {}).get("MAX_NUM_SEQS"),
        "precache_enabled": not args.no_precache,
        "precache_dataset": extra_env.get("PRECACHE_DATASET"),
        "reference": str(args.reference),
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed},
        "git": _git_info(),
        "stages": {},
    }

    def save() -> None:
        evidence["updated_utc"] = _now()
        (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))

    if not args.reference.exists():
        print(f"[floorlock] WARN reference missing: {args.reference}", flush=True)
        evidence["stages"]["reference"] = {"ok": False, "error": "reference file not found"}

    save()
    preflight_gpu()

    t_serve0 = time.time()
    try:
        with harness.LocalServer(submission_dir, server_python=server_python,
                                 log_path=server_log, extra_env=extra_env) as server:
            evidence["server_ready_s"] = time.time() - t_serve0
            evidence["base_url"] = server.base_url
            evidence["served_model_name"] = server.served_model_name
            model = server.served_model_name
            save()

            # Stage 0: self-test
            try:
                st = self_test(server.base_url, model)
                evidence["stages"]["self_test"] = {"ok": st["ok"], **st}
                print(f"[floorlock] self-test ok={st['ok']} wall={st['wall_s']:.1f}s "
                      f"sample={st['sample']!r}", flush=True)
                save()
                if not st["ok"]:
                    raise RuntimeError("self-test produced empty completion")
            except Exception as exc:
                evidence["stages"]["self_test"] = {"ok": False, "error": str(exc)}
                save()
                raise

            # Stage 1: official sglang TPS
            try:
                t0 = time.time()
                bench_jsonl = out_dir / "benchmark.jsonl"
                result = run_sglang_bench(
                    bench_python, base_url=server.base_url, model=model,
                    dataset_path=paths.EVAL_PROMPTS, output_file=bench_jsonl,
                    num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
                )
                bench = summarize_bench(result, model=model,
                                        num_prompts=args.num_prompts, output_len=args.output_len)
                bench["bench_wall_s"] = time.time() - t0
                (out_dir / "summary.json").write_text(json.dumps(bench, indent=2, sort_keys=True))
                evidence["bench"] = bench
                evidence["stages"]["sglang"] = {"ok": True, "tps": bench["tps"], "completed": bench["completed"]}
                print(f"[floorlock] sglang tps={bench['tps']:.3f} completed={bench['completed']} "
                      f"dur={bench['duration_s']:.1f}s", flush=True)
                save()
            except Exception as exc:
                evidence["stages"]["sglang"] = {"ok": False, "error": str(exc)}
                save()
                print(f"[floorlock] sglang FAILED: {exc}", flush=True)

            # Stage 2: decode capture -> wall_tps + token IDs
            try:
                t0 = time.time()
                decode_jsonl = out_dir / "decode_outputs.jsonl"
                decode_summary = out_dir / "decode_summary.json"
                dsum = harness.capture_decode(
                    server_python, base_url=server.base_url, model=model,
                    out_file=decode_jsonl, summary_file=decode_summary,
                    num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
                )
                n_tok = int(dsum.get("num_completion_tokens", 0))
                dur = float(dsum.get("duration_s", 0.0))
                wall_tps = n_tok / dur if dur > 0 else None
                evidence["decode"] = {
                    "num_records": dsum.get("num_records"),
                    "num_completion_tokens": n_tok,
                    "duration_s": dur,
                    "wall_tps": wall_tps,
                    "capture_wall_s": time.time() - t0,
                }
                evidence["stages"]["decode"] = {"ok": True, "wall_tps": wall_tps}
                print(f"[floorlock] decode wall_tps={wall_tps} n_tok={n_tok} dur={dur:.1f}s", flush=True)
                save()
            except Exception as exc:
                evidence["stages"]["decode"] = {"ok": False, "error": str(exc)}
                save()
                print(f"[floorlock] decode FAILED: {exc}", flush=True)

            # Stage 3b: PPL
            if not args.skip_ppl:
                try:
                    t0 = time.time()
                    psum = harness.run_ppl(
                        server_python, base_url=server.base_url, model=model,
                        out_file=out_dir / "ppl_results.jsonl",
                        summary_file=out_dir / "ppl_summary.json",
                    )
                    evidence["ppl"] = {
                        "ppl": psum.get("ppl"),
                        "num_tokens": psum.get("num_tokens"),
                        "num_records": psum.get("num_records"),
                        "ppl_wall_s": time.time() - t0,
                    }
                    evidence["stages"]["ppl"] = {"ok": True, "ppl": psum.get("ppl")}
                    print(f"[floorlock] ppl={psum.get('ppl')} num_tokens={psum.get('num_tokens')}", flush=True)
                    save()
                except Exception as exc:
                    evidence["stages"]["ppl"] = {"ok": False, "error": str(exc)}
                    save()
                    print(f"[floorlock] ppl FAILED: {exc}", flush=True)

            # Stage 4: modalities
            try:
                mod = modalities_probe.probe_modalities(
                    base_url=server.base_url, model=model, manifest=manifest,
                    submission_dir=submission_dir,
                    model_id=harness.serve_model_id(manifest, submission_dir),
                )
                evidence["modalities"] = mod
                evidence["stages"]["modalities"] = {"ok": True,
                                                    "all_modalities_loaded": mod.get("all_modalities_loaded")}
                print(f"[floorlock] modalities loaded={mod.get('modalities_loaded')} "
                      f"all={mod.get('all_modalities_loaded')}", flush=True)
                save()
            except Exception as exc:
                evidence["stages"]["modalities"] = {"ok": False, "error": str(exc)}
                save()
                print(f"[floorlock] modalities FAILED: {exc}", flush=True)
    except Exception as exc:
        evidence["serve_error"] = str(exc)
        save()
        print(f"[floorlock] SERVE/PIPELINE ERROR: {exc}", flush=True)

    # Stage 3a: greedy gate (after teardown; reads captured decode vs reference)
    decode_jsonl = out_dir / "decode_outputs.jsonl"
    if args.reference.exists() and decode_jsonl.exists():
        try:
            ident = greedy_identity(args.reference, decode_jsonl, args.output_len)
            evidence["identity"] = ident
            evidence["stages"]["identity"] = {"ok": True, "verdict": ident["verdict"]}
            print(f"[floorlock] identity verdict={ident['verdict']} "
                  f"prompt_rate={ident['token_identity_rate_prompt']} "
                  f"{ident['onset_line']}", flush=True)
        except Exception as exc:
            evidence["stages"]["identity"] = {"ok": False, "error": str(exc)}
            print(f"[floorlock] identity FAILED: {exc}", flush=True)
    save()

    # Verdicts + projection
    wall_tps = (evidence.get("decode") or {}).get("wall_tps")
    sglang_tps = (evidence.get("bench") or {}).get("tps")
    ppl = (evidence.get("ppl") or {}).get("ppl")
    verdicts = compute_verdicts(wall_tps=wall_tps, sglang_tps=sglang_tps,
                                identity=evidence.get("identity"), ppl=ppl)
    evidence["verdicts"] = verdicts
    evidence["finished_utc"] = _now()
    save()

    # W&B
    run_id = log_wandb(args, evidence)
    if run_id:
        evidence["wandb_run_id"] = run_id
        save()

    _print_final(evidence)
    return 0


def _git_info() -> dict[str, Any]:
    try:
        from scripts import wandb_logging
        return wandb_logging.git_info()
    except Exception:
        return {}


def _print_final(ev: dict[str, Any]) -> None:
    v = ev.get("verdicts", {})
    b = ev.get("bench", {})
    d = ev.get("decode", {})
    i = ev.get("identity", {})
    p = ev.get("ppl", {})
    print("\n[floorlock] ============ FLOOR-LOCK FULL-SERVE RESULT ============", flush=True)
    print(f"  submission            : {ev.get('submission')} ({ev.get('manifest_name')})", flush=True)
    print(f"  spec_off / M=1        : {ev.get('spec_off')} / max_num_seqs={ev.get('max_num_seqs')}", flush=True)
    print(f"  precache_enabled      : {ev.get('precache_enabled')}", flush=True)
    print(f"  sglang official tps   : {b.get('tps')}  (completed={b.get('completed')}/{ev.get('workload',{}).get('num_prompts')})", flush=True)
    print(f"  local wall_tps        : {d.get('wall_tps')}", flush=True)
    print(f"  projected official    : {v.get('projected_official_from_wall_tps')} (wall x{LOCAL_OFFICIAL_MULTIPLIER})", flush=True)
    print(f"    vs model 161.70 +- {SIGMA_HW}  delta={v.get('delta_from_model_wall')}", flush=True)
    print(f"  PPL                   : {p.get('ppl')}  (cap {modalities_probe.PPL_CAP})", flush=True)
    print(f"  identity verdict      : {i.get('verdict')}  prompt_rate={i.get('token_identity_rate_prompt')} token_rate={i.get('token_identity_rate_token')}", flush=True)
    print(f"  modalities            : {ev.get('modalities',{}).get('all_modalities_loaded')}", flush=True)
    print(f"  VERDICT realizes_16170: {v.get('floorlock_realizes_16170')}", flush=True)
    print(f"  VERDICT literal_1p0   : {v.get('floorlock_literal_1p0')}", flush=True)
    print("[floorlock] ======================================================", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
