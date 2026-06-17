#!/usr/bin/env python
"""PR #608 fern — MTP draft-depth K-sweep SPEED Pareto on the #597 int4_g128+MTP build.

The strict-#319 AR frame is closed at 126.378 (lawine #601). The only path beyond
is spec-dec, and #597 measured int4_g128_lmhead + MTP-K7 @ 427.7 official-proxy TPS.
This card maps the SPEED Pareto across draft depth K to find the speed-optimum.

K is a free serve-time parameter: submissions/int4_mtp_batchinv/serve.py builds
``--speculative-config {"num_speculative_tokens": K}`` straight from the
NUM_SPECULATIVE_TOKENS env var, and the gemma4_assistant drafter (Gemma4MTPModel)
runs autoregressively for K steps. So no rebuild is needed for any K — we reuse
the on-disk /workspace/gemma_build/int4_g128_lmhead (9.7G) + /tmp/qat-assistant.

Protocol — IDENTICAL to #597 (only K varies):
  serve int4_mtp_batchinv with MODEL_ID=int4_g128_lmhead, DRAFTER=/tmp/qat-assistant,
  VLLM_BATCH_INVARIANT=1, MAX_NUM_SEQS=1, VLLM_USE_FLASHINFER_SAMPLER=0,
  MAX_MODEL_LEN=4096, GPU_MEMORY_UTILIZATION=0.90, MAX_NUM_BATCHED_TOKENS=512.
  TPS  = harness.probe_tps(decode_tokens=512).decode_tps_single_stream  (the exact
         call that gave #597 local 413.26 -> proxy 427.7 at K=7), median of R bursts.
  proxy_tps = local_decode_tps * TAU_LO (1.035, banked #594).  official_tps stays 0.
  acceptance = clean Prometheus /metrics delta bracketed around one dedicated 512-tok
               decode on the same probe prompt (+ whole-run server-log cross-check),
               via scripts.local_validation.serve_profile parsers.
    accepted_draft_tokens_per_step = d_accepted / d_drafts   (PR's "mean accepted draft tokens/step")
    e_accept (mean acceptance length)  = 1 + accepted_per_step
    draft_acceptance_rate              = d_accepted / d_draft_tokens

Deliverable: K vs official_proxy_tps vs acceptance -> the fastest spec config.
LOCAL A10G, analysis_only, official_tps=0, single GPU. NO HF Job / --launch /
submission / served-file change. Identity is wirbel #607's axis, quality stark #605's.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

TAU_LO = 1.035  # banked #594 local->official scalar (same as #597)
PROBE_PROMPT = "Explain step by step how a transformer decodes one token at a time."


def load_eval_prompts(n: int) -> list[str]:
    """First-human-turn of the first n official sharegpt eval prompts.

    The probe-prompt acceptance (single self-similar prompt under ignore_eos)
    runs degenerate-high; this realistic, diverse-prompt set gives the honest
    acceptance the real 128-prompt benchmark would see (cf. #597's own 128x512
    sharegpt capture: e_accept ~4.0-4.5 at K=7, not ~7)."""
    data = json.loads(paths.EVAL_PROMPTS.read_text())
    out: list[str] = []
    for item in data[:n]:
        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                out.append(str(turn.get("value", ""))[:4000])
                break
    return out


def base_env(model_id: str, drafter: str, batch_invariant: int) -> dict[str, str]:
    """The exact #597 base env (run_identity._base_env); K is added per-arm."""
    return {
        "MODEL_ID": model_id,
        "DRAFTER_MODEL": drafter,
        "VLLM_BATCH_INVARIANT": str(batch_invariant),
        "MAX_NUM_SEQS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "512",
    }


def _fetch_metrics(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _vram_sampler(stop: threading.Event, peak: dict) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
            if vals:
                peak["mib"] = max(peak["mib"], max(vals))
        except (OSError, subprocess.SubprocessError):
            pass
        stop.wait(2.0)


def _acceptance_delta(m0: dict, m1: dict, k: int) -> dict:
    """Clean per-arm acceptance from the cumulative-counter delta (m1 - m0)."""
    def d(key: str):
        a, b = m0.get(key), m1.get(key)
        return (b - a) if (a is not None and b is not None) else None

    d_drafts = d("num_drafts")
    d_acc = d("num_accepted_tokens")
    d_draft_tok = d("num_draft_tokens")
    out: dict = {
        "delta_num_drafts": d_drafts,
        "delta_num_accepted_tokens": d_acc,
        "delta_num_draft_tokens": d_draft_tok,
        "served_k": k,
    }
    if d_drafts and d_acc is not None:
        out["accepted_draft_tokens_per_step"] = d_acc / d_drafts
        out["e_accept_mean_acceptance_length"] = 1.0 + d_acc / d_drafts
    if d_draft_tok and d_acc is not None:
        out["draft_acceptance_rate"] = d_acc / d_draft_tok
    out["source"] = "prometheus_bracket_delta"
    return out


def run_k(submission: Path, server_python: Path, *, model_id: str, drafter: str,
          batch_invariant: int, k: int, decode_tokens: int, repeats: int,
          accept_tokens: int, real_prompts: int, real_tokens: int,
          port: int, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"server_k{k}.log"
    extra_env = {**base_env(model_id, drafter, batch_invariant),
                 "NUM_SPECULATIVE_TOKENS": str(k)}

    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_vram_sampler, args=(stop, peak), daemon=True)
    sampler.start()

    result: dict = {"k": k, "model_id": model_id, "drafter": drafter,
                    "batch_invariant": batch_invariant, "decode_tokens": decode_tokens,
                    "repeats": repeats, "accept_tokens": accept_tokens, "tau_lo": TAU_LO}
    t0 = time.time()
    try:
        with harness.LocalServer(submission, server_python=server_python, port=port,
                                 log_path=log_path, extra_env=extra_env,
                                 startup_timeout_s=1800) as srv:
            result["serve_ready_s"] = time.time() - t0
            base_url, model = srv.base_url, srv.served_model_name
            # warmup so CUDA graphs are captured before any timed work
            harness._completion(base_url, model, PROBE_PROMPT, 16)

            # --- TPS: median of R single-stream decode bursts (the #597 probe) ---
            tps_runs = []
            for _ in range(repeats):
                tps_runs.append(harness.probe_tps(base_url, model, decode_tokens=decode_tokens,
                                                   prompt=PROBE_PROMPT))
            local_list = sorted(t["decode_tps_single_stream"] for t in tps_runs)
            median_local = local_list[len(local_list) // 2]
            result["tps_runs"] = tps_runs
            result["local_decode_tps_runs"] = local_list
            result["local_decode_tps_single_stream"] = median_local
            result["local_decode_tps_min"] = local_list[0]
            result["local_decode_tps_max"] = local_list[-1]
            result["official_proxy_tps"] = median_local * TAU_LO
            result["beats_126_378"] = result["official_proxy_tps"] > 126.378

            # --- PROBE-BASIS acceptance: clean Prometheus bracket around one
            #     dedicated decode on the SAME probe prompt that sets the proxy
            #     TPS. This explains the proxy TPS but runs degenerate-high. ---
            m0 = serve_profile.parse_spec_metrics(_fetch_metrics(base_url))
            harness._completion(base_url, model, PROBE_PROMPT, accept_tokens)
            m1 = serve_profile.parse_spec_metrics(_fetch_metrics(base_url))
            result["prom_m0"] = m0
            result["prom_m1"] = m1
            result["acceptance_probe"] = _acceptance_delta(m0, m1, k)

            # --- REALISTIC acceptance: clean Prometheus bracket around a diverse
            #     multi-prompt decode over the official sharegpt eval set. This is
            #     the honest acceptance the real 128-prompt benchmark would see. ---
            real = load_eval_prompts(real_prompts)
            rm0 = serve_profile.parse_spec_metrics(_fetch_metrics(base_url))
            for p in real:
                harness._completion(base_url, model, p, real_tokens)
            rm1 = serve_profile.parse_spec_metrics(_fetch_metrics(base_url))
            result["prom_real_m0"] = rm0
            result["prom_real_m1"] = rm1
            acc_real = _acceptance_delta(rm0, rm1, k)
            acc_real["n_prompts"] = len(real)
            acc_real["tokens_per_prompt"] = real_tokens
            result["acceptance_realistic"] = acc_real
    finally:
        stop.set()
        sampler.join(timeout=5)

    # whole-run server-log cross-check (vLLM's own SpecDecoding lines)
    try:
        result["spec_log_wholerun"] = serve_profile.parse_spec_log(log_path.read_text(errors="ignore"))
    except OSError:
        result["spec_log_wholerun"] = {}
    result["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    result["server_log"] = str(log_path)
    (run_dir / f"result_k{k}.json").write_text(json.dumps(result, indent=2, default=str))

    accp = result.get("acceptance_probe", {})
    accr = result.get("acceptance_realistic", {})
    print("\n" + "=" * 64, flush=True)
    print(f"[K={k}] local_decode_tps(median)={median_local:.2f}  "
          f"official_proxy_tps={result['official_proxy_tps']:.2f}  "
          f"(min/max local {local_list[0]:.1f}/{local_list[-1]:.1f})", flush=True)
    print(f"[K={k}] PROBE  accept/step={accp.get('accepted_draft_tokens_per_step')}  "
          f"e_accept={accp.get('e_accept_mean_acceptance_length')}  "
          f"rate={accp.get('draft_acceptance_rate')}", flush=True)
    print(f"[K={k}] REAL   accept/step={accr.get('accepted_draft_tokens_per_step')}  "
          f"e_accept={accr.get('e_accept_mean_acceptance_length')}  "
          f"rate={accr.get('draft_acceptance_rate')}  peak_vram_gb={result['peak_vram_gb']:.2f}",
          flush=True)
    print("=" * 64, flush=True)
    return result


def log_wandb(report: dict, name: str, group: str) -> str | None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})", flush=True)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
            config={
                "pr": 608, "analysis_only": True, "official_tps": 0,
                "model_id": report["model_id"], "drafter": report["drafter"],
                "batch_invariant": report["batch_invariant"],
                "decode_tokens": report["decode_tokens"], "repeats": report["repeats"],
                "tau_lo": TAU_LO, "ks": report["ks"],
                "protocol": "597-identical probe_tps single-stream; only K varies",
            },
        )
        cols = ["k", "local_decode_tps", "official_proxy_tps",
                "probe_accept_per_step", "probe_e_accept", "probe_acc_rate",
                "real_accept_per_step", "real_e_accept", "real_acc_rate",
                "peak_vram_gb"]
        tbl = wandb.Table(columns=cols)
        for arm in report["arms"]:
            accp = arm.get("acceptance_probe", {})
            accr = arm.get("acceptance_realistic", {})
            tbl.add_data(arm["k"], arm["local_decode_tps_single_stream"],
                         arm["official_proxy_tps"],
                         accp.get("accepted_draft_tokens_per_step"),
                         accp.get("e_accept_mean_acceptance_length"),
                         accp.get("draft_acceptance_rate"),
                         accr.get("accepted_draft_tokens_per_step"),
                         accr.get("e_accept_mean_acceptance_length"),
                         accr.get("draft_acceptance_rate"),
                         arm.get("peak_vram_gb"))
            # also log each K as a step so curves render
            run.log({"k": arm["k"],
                     "local_decode_tps": arm["local_decode_tps_single_stream"],
                     "official_proxy_tps": arm["official_proxy_tps"],
                     "probe_accept_per_step": accp.get("accepted_draft_tokens_per_step"),
                     "probe_acc_rate": accp.get("draft_acceptance_rate"),
                     "real_accept_per_step": accr.get("accepted_draft_tokens_per_step"),
                     "real_e_accept": accr.get("e_accept_mean_acceptance_length"),
                     "real_acc_rate": accr.get("draft_acceptance_rate")})
        run.log({"k_sweep_pareto": tbl})
        run.summary.update({
            "fastest_k": report["fastest_k"],
            "fastest_official_proxy_tps": report["fastest_official_proxy_tps"],
            "k7_official_proxy_tps": report.get("k7_official_proxy_tps"),
            "k7_reproduces_427": report.get("k7_reproduces_427"),
            "official_tps": 0, "analysis_only": True,
            "primary_metric": report["fastest_official_proxy_tps"],
        })
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid}", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submissions" / "int4_mtp_batchinv")
    ap.add_argument("--model-id", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--drafter", default="/tmp/qat-assistant")
    ap.add_argument("--batch-invariant", type=int, default=1, choices=[0, 1])
    ap.add_argument("--ks", default="3,5,7,9,11,13")
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--accept-tokens", type=int, default=512)
    ap.add_argument("--real-prompts", type=int, default=8,
                    help="diverse sharegpt eval prompts for the honest acceptance bracket")
    ap.add_argument("--real-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="mtp-k-sweep-speed-pareto")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="K=3 only, decode_tokens=128, repeats=1 — quick wiring check")
    args = ap.parse_args()

    if args.smoke:
        ks = [3]
        args.decode_tokens, args.repeats, args.accept_tokens = 128, 1, 128
        args.real_prompts, args.real_tokens = 4, 128
    else:
        ks = [int(x) for x in args.ks.split(",") if x.strip()]

    for note in paths.prepare_local_gpu_env():
        print(f"[k_sweep] {note}", flush=True)

    manifest = harness.load_manifest(args.submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[k_sweep] server_python={server_python}", flush=True)
    print(f"[k_sweep] model_id={args.model_id} drafter={args.drafter} "
          f"BI={args.batch_invariant} ks={ks}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for k in ks:
        arm = run_k(args.submission, server_python, model_id=args.model_id,
                    drafter=args.drafter, batch_invariant=args.batch_invariant, k=k,
                    decode_tokens=args.decode_tokens, repeats=args.repeats,
                    accept_tokens=args.accept_tokens, real_prompts=args.real_prompts,
                    real_tokens=args.real_tokens, port=args.port, run_dir=args.out_dir)
        arms.append(arm)

    best = max(arms, key=lambda a: a["official_proxy_tps"])
    k7 = next((a for a in arms if a["k"] == 7), None)
    report = {
        "pr": 608, "model_id": args.model_id, "drafter": args.drafter,
        "batch_invariant": args.batch_invariant, "decode_tokens": args.decode_tokens,
        "repeats": args.repeats, "tau_lo": TAU_LO, "ks": ks, "arms": arms,
        "fastest_k": best["k"], "fastest_official_proxy_tps": best["official_proxy_tps"],
        "fastest_local_decode_tps": best["local_decode_tps_single_stream"],
        "k7_official_proxy_tps": (k7 or {}).get("official_proxy_tps"),
        "k7_reproduces_427": (abs((k7 or {}).get("official_proxy_tps", 0) - 427.7) < 25.0)
        if k7 else None,
        "analysis_only": True, "official_tps": 0,
    }
    (args.out_dir / "pareto.json").write_text(json.dumps(report, indent=2, default=str))

    def _g(d, key):
        v = (d or {}).get(key)
        return v if v is not None else float("nan")

    print("\n" + "#" * 78, flush=True)
    print("# MTP K-SWEEP SPEED PARETO (int4_g128 + MTP, BI=1, #597 protocol)", flush=True)
    print(f"# {'K':>3} | {'local_tps':>9} | {'proxy_tps':>9} | "
          f"{'probe a/s':>9} | {'probe rate':>10} | {'REAL a/s':>8} | {'REAL rate':>9}",
          flush=True)
    for a in arms:
        accp, accr = a.get("acceptance_probe", {}), a.get("acceptance_realistic", {})
        print(f"# {a['k']:>3} | {a['local_decode_tps_single_stream']:>9.2f} | "
              f"{a['official_proxy_tps']:>9.2f} | "
              f"{_g(accp,'accepted_draft_tokens_per_step'):>9.3f} | "
              f"{_g(accp,'draft_acceptance_rate'):>10.3f} | "
              f"{_g(accr,'accepted_draft_tokens_per_step'):>8.3f} | "
              f"{_g(accr,'draft_acceptance_rate'):>9.3f}", flush=True)
    print(f"# FASTEST: K={report['fastest_k']} at "
          f"{report['fastest_official_proxy_tps']:.2f} official-proxy TPS "
          f"(vs #597 K7=427.7)", flush=True)
    print("#" * 78, flush=True)

    if args.wandb_name and not args.no_wandb:
        report["wandb_run_id"] = log_wandb(report, args.wandb_name, args.wandb_group)
    print(f"[k_sweep] artifacts -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
