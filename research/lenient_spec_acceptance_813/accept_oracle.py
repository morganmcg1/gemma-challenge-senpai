#!/usr/bin/env python
"""PR #813 Step-2 — Synthetic-acceptance ceiling oracle (advisor probe).

Imposes a chosen draft-acceptance rate on the int4head stack via vLLM 0.22's
``rejection_sample_method='synthetic'`` (greedy branch, temp=0) and measures the
resulting local conc=1 decode TPS. The emitted tokens are GARBAGE (synthetic
accept ignores argmax-match) so this is NOT a quality run and NEVER ships — but
the decode loop runs the identical drafter+verify+lm_head kernels, so
TPS-vs-imposed-rate is a faithful SPEED ceiling for "what if E_accept were
higher". This bounds the prize of the entire verifier-acceptance axis without a
custom kernel patch (Step-1 proved no servable lenient config knob exists).

Flat per-position rate list [r]*K (K=6): mean accepted DRAFT tokens = K*r, mean
acceptance length = 1 + K*r. r=0.56 ~ current E_accept 3.38/K (sanity anchor;
should reproduce ~256 local TPS). r=1.00 = the ceiling (always accept all K).

LOCAL A10G, EXPLORATORY. No HF job. Serves the REAL int4head model
(submissions/int4_mtp_bi0_int4head, MODEL_ID gemma-challenge/...-int4head) — the
256.74 anchor was previously RECONSTRUCTED from a bf16-head serve, so this is
also the first true int4head serve.

Decision rule (advisor): ceiling@r=1.00 > ~282 TPS (>~10% over 256.74) -> real
headroom, greenlight a separate custom greedy-kernel top-k-match PR; < ~270
(<~5%) -> acceptance axis exhausted, close it.

Run (background):
  CUDA_VISIBLE_DEVICES=0 uv run python research/lenient_spec_acceptance_813/accept_oracle.py \
    --rates 0.56,0.70,0.85,1.00 --wandb-group bi0-int4head-accept-oracle
Smoke (cheap pipeline check):
  ... --rates 1.00 --num-prompts 2 --output-len 32 --tag smoke --wandb-group bi0-int4head-accept-oracle-smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.local_validation.serve_profile import (  # noqa: E402
    parse_spec_log,
    parse_spec_metrics,
    _get_text,
)

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_int4head"
SERVER_PY = ROOT / ".venv" / "bin" / "python"
K_DEFAULT = 6


def run_one_rate(
    rate: float, *, k: int, num_prompts: int, output_len: int, out_dir: Path,
    port: int,
) -> dict[str, Any]:
    """Serve int4head at an imposed synthetic accept rate; measure TPS + E_accept."""
    label = f"r{rate:.2f}"
    rates_list = [round(rate, 6)] * k
    log_path = out_dir / f"server_{label}.log"
    extra_env = {
        # Force the in-container GPU + native sampler (CUDA_VISIBLE_DEVICES=1 is
        # inherited but only index 0 exists; FlashInfer sampler JIT crashes here).
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Re-enable Prometheus stats so the spec_decode_* counters (canonical
        # E_accept) are exposed; the manifest ships --disable-log-stats.
        "DISABLE_LOG_STATS": "0",
        # The acceptance-oracle knobs (consumed by the #813 passthrough in serve.py).
        "REJECTION_SAMPLE_METHOD": "synthetic",
        "SYNTHETIC_ACCEPTANCE_RATES": json.dumps(rates_list),
        "NUM_SPECULATIVE_TOKENS": str(k),
    }
    rec: dict[str, Any] = {
        "rate": rate,
        "rates_list": rates_list,
        "k": k,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "imposed_mean_accepted_draft_tokens": k * rate,
        "imposed_mean_acceptance_length": 1.0 + k * rate,
    }
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=port, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        rec["boot_s"] = time.time() - t0
        rec["model_id"] = srv.model_id
        decode_out = out_dir / f"decode_{label}.jsonl"
        decode_sum = out_dir / f"decode_{label}.summary.json"
        td = time.time()
        summary = harness.capture_decode(
            SERVER_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_sum,
            num_prompts=num_prompts, output_len=output_len, timeout_s=3600,
        )
        rec["decode_wall_s"] = time.time() - td
        rec["decode_summary"] = summary
        try:
            rec["spec_metrics"] = parse_spec_metrics(_get_text(f"{srv.base_url}/metrics"))
        except Exception as exc:  # noqa: BLE001
            rec["spec_metrics"] = {"error": str(exc)}
    spec_log = parse_spec_log(log_path.read_text())
    rec["spec_log"] = spec_log

    # Headline metrics.
    rec["steady_gen_tps"] = spec_log.get("steady_gen_tps_mean")
    rec["steady_gen_tps_n"] = spec_log.get("steady_gen_tps_n")
    rec["e_accept_exact_from_log"] = spec_log.get("e_accept_exact")
    rec["draft_acceptance_rate_measured"] = spec_log.get("draft_acceptance_rate")
    pm = rec.get("spec_metrics") or {}
    rec["e_accept_mean_acceptance_length_prom"] = pm.get("e_accept_mean_acceptance_length")
    dur = (summary or {}).get("duration_s")
    ntok = (summary or {}).get("num_completion_tokens")
    rec["decode_wall_tps"] = (ntok / dur) if (dur and ntok) else None
    rec["num_records"] = (summary or {}).get("num_records")
    return rec


def log_wandb(rec: dict[str, Any], *, group: str, name: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed ({exc}); JSON-only", flush=True)
        return None
    try:
        run = wandb.init(
            entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
            group=group, name=name, reinit=True,
            config={
                "pr": 813,
                "probe": "synthetic-acceptance-ceiling-oracle",
                "analysis_only": True,
                "official_tps": 0,
                "synthetic_garbage_tokens": True,
                "model_id": rec.get("model_id"),
                "submission": "int4_mtp_bi0_int4head",
                "imposed_rate": rec["rate"],
                "rates_list": rec["rates_list"],
                "k": rec["k"],
                "num_prompts": rec["num_prompts"],
                "output_len": rec["output_len"],
                "imposed_mean_accepted_draft_tokens": rec["imposed_mean_accepted_draft_tokens"],
                "imposed_mean_acceptance_length": rec["imposed_mean_acceptance_length"],
            },
        )
        run.summary.update({
            "steady_gen_tps": rec.get("steady_gen_tps"),
            "steady_gen_tps_n": rec.get("steady_gen_tps_n"),
            "decode_wall_tps": rec.get("decode_wall_tps"),
            "e_accept_exact_from_log": rec.get("e_accept_exact_from_log"),
            "e_accept_mean_acceptance_length_prom": rec.get("e_accept_mean_acceptance_length_prom"),
            "draft_acceptance_rate_measured": rec.get("draft_acceptance_rate_measured"),
            "imposed_mean_acceptance_length": rec["imposed_mean_acceptance_length"],
            "boot_s": rec.get("boot_s"),
            "decode_wall_s": rec.get("decode_wall_s"),
            "num_records": rec.get("num_records"),
        })
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid} ({name})", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.56,0.70,0.85,1.00")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--wandb-group", default="bi0-int4head-accept-oracle")
    ap.add_argument("--wandb-prefix", default="stark/accept-oracle")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[gpu-env] {n}", flush=True)

    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    suffix = f"-{args.tag}" if args.tag else ""
    out_dir = HERE / "runs" / (args.tag or "sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[oracle] rates={rates} k={args.k} np={args.num_prompts} ol={args.output_len} "
          f"out={out_dir}", flush=True)

    results: list[dict[str, Any]] = []
    for rate in rates:
        print(f"\n===== imposed rate {rate} (rates_list=[{rate}]*{args.k}) =====", flush=True)
        try:
            rec = run_one_rate(
                rate, k=args.k, num_prompts=args.num_prompts,
                output_len=args.output_len, out_dir=out_dir, port=args.port,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[oracle] rate {rate} FAILED: {exc!r}", flush=True)
            rec = {"rate": rate, "error": repr(exc)}
        if not args.no_wandb and "error" not in rec:
            rec["wandb_run_id"] = log_wandb(
                rec, group=args.wandb_group, name=f"{args.wandb_prefix}-r{rate:.2f}{suffix}",
            )
        # Persist incrementally so a timeout/kill keeps completed rates.
        (out_dir / f"rate_{rate:.2f}.json").write_text(json.dumps(rec, indent=2, default=str))
        results.append(rec)
        st = rec.get("steady_gen_tps")
        eacc = rec.get("e_accept_exact_from_log")
        print(f"[oracle] rate={rate}  steady_gen_tps={st}  e_accept={eacc}  "
              f"wall_tps={rec.get('decode_wall_tps')}", flush=True)

    (out_dir / "sweep_summary.json").write_text(json.dumps(results, indent=2, default=str))

    # Ceiling table + relative gain vs the r=0.56 anchor.
    print("\n========== ACCEPT-ORACLE CEILING ==========", flush=True)
    print(f"{'rate':>6} {'mean_acc_len':>12} {'steady_tps':>11} {'e_accept':>9} {'wall_tps':>9}",
          flush=True)
    anchor = None
    for rec in results:
        if "error" in rec:
            print(f"{rec['rate']:>6} ERROR {rec['error']}", flush=True)
            continue
        if abs(rec["rate"] - 0.56) < 1e-6:
            anchor = rec.get("steady_gen_tps")
        print(f"{rec['rate']:>6.2f} {rec['imposed_mean_acceptance_length']:>12.2f} "
              f"{(rec.get('steady_gen_tps') or float('nan')):>11.2f} "
              f"{(rec.get('e_accept_exact_from_log') or float('nan')):>9.3f} "
              f"{(rec.get('decode_wall_tps') or float('nan')):>9.2f}", flush=True)
    ceil = next((r.get("steady_gen_tps") for r in results
                 if abs(r.get("rate", -1) - 1.00) < 1e-6 and "error" not in r), None)
    if anchor and ceil:
        gain = 100.0 * (ceil - anchor) / anchor
        print(f"\nanchor(r=0.56) steady_tps = {anchor:.2f}", flush=True)
        print(f"ceiling(r=1.00) steady_tps = {ceil:.2f}  (+{gain:.1f}% over anchor)", flush=True)
        verdict = ("GREENLIGHT custom-kernel top-k-match PR" if gain > 10.0
                   else "CLOSE acceptance axis" if gain < 5.0
                   else "MARGINAL (5-10%): lean close, report honestly")
        print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
