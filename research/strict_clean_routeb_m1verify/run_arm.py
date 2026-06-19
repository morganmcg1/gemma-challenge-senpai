#!/usr/bin/env python
"""PR #746 route-b arm driver: serve int4_mtp_batchinv in one config, capture the
official 128x512 single-stream greedy decode, record wall_tps (+ optional PPL),
and save decode_outputs.jsonl for byte-exact identity comparison.

Modes (all single-stream, MAX_NUM_SEQS=1, temp=0 -> the 126.378 predicate):
  arref      SENPAI_REFERENCE_MODE=1  -> spec OFF, plain M=1 AR int4 target.
             This is route-b's HARD CEILING (route-b does >=1 M=1 target forward
             per emitted token, same count as plain AR, plus drafter overhead) and
             the strict byte-exact greedy reference.
  batched K  NUM_SPECULATIVE_TOKENS=K -> the #730 batched M=K+1 verify fire. Gives
             accept_len(K) + the byte-exactness-tax denominator wall_tps_batched(K).

Reuses scripts.local_validation.harness verbatim (same serve+decode+ppl path the
ar_identity_safe_tps / int4_g128_lmhead arms use). Local A10G only, NO HF job.

  .venv/bin/python -m research.strict_clean_routeb_m1verify.run_arm \
      --mode arref --out-dir research/strict_clean_routeb_m1verify/arref \
      --wandb-name stark/routeb-arref
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
TAU = 1.03524  # local wall_tps -> official scalar (#267)
BASELINE_OFFICIAL_TPS = 126.378  # int4_g128_lmhead AR rung (PR #601)


def parse_accept_from_log(log_path: Path) -> dict:
    """Pull vLLM v1 spec-decode acceptance stats from the server log.

    vLLM v1 (the version we serve) prints, every metrics interval, a line:

      SpecDecoding metrics: Mean acceptance length: 2.24, Accepted throughput:
      77.10 tokens/s, Drafted throughput: 124.20 tokens/s, Accepted: 771 tokens,
      Drafted: 1252 tokens, Per-position acceptance rate: 0.717, 0.525, Avg Draft
      acceptance rate: 62.1%

    The 'Accepted: N tokens' / 'Drafted: N tokens' counts are PER-INTERVAL (they
    fluctuate, not monotone), so we SUM them across all intervals to get the
    warm-steady aggregate. The mean accepted DRAFT tokens per step is then
    ``a = K * total_accepted / total_drafted`` (each step drafts K tokens), and
    emitted/step = a + 1 (the +1 bonus token). NOTE vLLM's reported 'Mean
    acceptance length' already equals emitted/step (= a + 1), not a -- we keep it
    only as a cross-check; the aggregate count ratio is the primary, exact source.

    (Older code matched 'mean acceptance length' case-sensitively -- vLLM prints
    'Mean ...' -- and then fell through to an 'acceptance rate' regex that grabbed
    'Avg Draft acceptance rate: 62.1%', i.e. a percentage, as 'a'. Both fixed.)
    """
    out: dict = {}
    if not log_path or not log_path.exists():
        return out
    text = log_path.read_text(errors="ignore")
    acc = [int(x) for x in re.findall(r"Accepted:\s*(\d+)\s*tokens", text)]
    drf = [int(x) for x in re.findall(r"Drafted:\s*(\d+)\s*tokens", text)]
    mal = [float(x) for x in re.findall(r"Mean acceptance length:\s*([0-9.]+)",
                                        text, flags=re.IGNORECASE)]
    rate = [float(x) for x in re.findall(
        r"Avg Draft acceptance rate:\s*([0-9.]+)%", text, flags=re.IGNORECASE)]
    if acc:
        out["num_accepted_tokens"] = float(sum(acc))
    if drf:
        out["num_draft_tokens"] = float(sum(drf))
    if mal:
        out["mean_acceptance_length"] = sum(mal) / len(mal)  # avg over intervals (= a+1)
        out["mean_acceptance_length_last"] = mal[-1]
    if rate:
        out["avg_draft_acceptance_rate_pct"] = sum(rate) / len(rate)
    out["spec_metric_intervals"] = float(len(acc))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["arref", "batched"])
    ap.add_argument("--k", type=int, default=6, help="num_speculative_tokens for batched mode")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--server-python", default=str(ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--ppl", action="store_true", help="also run the PPL guardrail")
    ap.add_argument("--no-batch-invariant", action="store_true",
                    help="override the manifest VLLM_BATCH_INVARIANT=1 to 0. Route-b's "
                         "verify is M=1 (single-query, same shape as decode), so its "
                         "byte-exactness comes from shape identity, NOT batch-invariant "
                         "kernels (batchinv was the route-a fix for M=K+1 batched verify). "
                         "This arm measures the FAST-kernel M=1 AR ceiling = route-b's true "
                         "best-case ceiling, since route-b need not pay the batchinv tax.")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="strict-clean-routeb-m1verify")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_python = Path(args.server_python)

    # VLLM_USE_FLASHINFER_SAMPLER=0: this pod's CUDA toolkit lacks curand.h in
    # /usr/local/cuda/include, so flashinfer's warmup JIT of cached_ops/sampling
    # fails the EngineCore startup. The native PyTorch sampler is byte-identical
    # for greedy (argmax==argmax) and TPS-neutral, so this is a pure local-env
    # build workaround, not a serving-path change.
    extra_env = {"CUDA_VISIBLE_DEVICES": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    bi_suffix = ""
    if args.no_batch_invariant:
        extra_env["VLLM_BATCH_INVARIANT"] = "0"
        bi_suffix = "_fastkern"
    if args.mode == "arref":
        extra_env["SENPAI_REFERENCE_MODE"] = "1"
        label = "arref_m1ar_specoff" + bi_suffix
    else:
        extra_env["NUM_SPECULATIVE_TOKENS"] = str(args.k)
        label = f"batched_k{args.k}" + bi_suffix

    log_path = out_dir / f"serve_{label}.log"
    rec: dict = {
        "pr": 746, "mode": args.mode, "k": (None if args.mode == "arref" else args.k),
        "label": label, "submission": str(SUBMISSION),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra_env": extra_env, "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "batch_invariant": (0 if args.no_batch_invariant else 1),
    }
    print(f"[arm:{label}] serving {SUBMISSION.name} env={extra_env}", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=server_python, port=args.port,
        log_path=log_path, extra_env=extra_env,
    ) as srv:
        rec["ready_s"] = time.time() - t0
        out_file = out_dir / "decode_outputs.jsonl"
        summary_file = out_dir / "decode_summary.json"
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=out_file, summary_file=summary_file,
            num_prompts=args.num_prompts, output_len=args.output_len,
        )
        wall_tps = summary["num_completion_tokens"] / summary["duration_s"]
        rec.update({
            "wall_tps": wall_tps,
            "tau_official_proj": wall_tps * TAU,
            "beats_126378_raw": bool(wall_tps > BASELINE_OFFICIAL_TPS),
            "num_completion_tokens": summary["num_completion_tokens"],
            "duration_s": summary["duration_s"],
            "decode_jsonl": str(out_file),
        })
        print(f"[arm:{label}] wall_tps={wall_tps:.3f} official_proj={wall_tps*TAU:.3f} "
              f"(>126.378? {wall_tps>BASELINE_OFFICIAL_TPS})", flush=True)
        if args.ppl:
            ppl_out = out_dir / "ppl_outputs.jsonl"
            ppl_sum = out_dir / "ppl_summary.json"
            ppl_summary = harness.run_ppl(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=ppl_out, summary_file=ppl_sum,
            )
            rec["ppl"] = ppl_summary.get("ppl")
            rec["ppl_pass"] = bool((rec["ppl"] or 99) <= 2.42)
            print(f"[arm:{label}] ppl={rec['ppl']} (<=2.42? {rec.get('ppl_pass')})", flush=True)

    rec["accept_stats"] = parse_accept_from_log(log_path)
    rec["elapsed_s"] = time.time() - t0
    (out_dir / "arm_result.json").write_text(json.dumps(rec, indent=2))
    print(f"[arm:{label}] DONE in {rec['elapsed_s']:.0f}s -> {out_dir/'arm_result.json'}",
          flush=True)

    if args.wandb_name and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            from scripts import wandb_logging
            run = wandb_logging.init_wandb_run(
                job_type="serve-decode", agent="stark", name=args.wandb_name,
                group=args.wandb_group,
                notes=f"PR #746 route-b arm {label}: served wall_tps anchor. local, official_tps=0.",
                tags=["pr746", "route-b", label, "analysis-only"],
                config={**rec.get("extra_env", {}), "pr": 746, "mode": args.mode,
                        "k": rec["k"], "baseline_official_tps": BASELINE_OFFICIAL_TPS,
                        "tau_local_to_official": TAU, "official_tps": 0,
                        "analysis_only": True, "num_prompts": args.num_prompts,
                        "output_len": args.output_len},
            )
            if run is not None:
                wandb.log({"global_step": 0, "wall_tps": wall_tps,
                           "tau_official_proj": wall_tps * TAU,
                           "beats_126378": int(wall_tps > BASELINE_OFFICIAL_TPS),
                           **{f"accept/{k}": v for k, v in rec["accept_stats"].items()},
                           **({"ppl": rec["ppl"]} if args.ppl else {})})
                run.summary.update({"wall_tps": wall_tps,
                                    "tau_official_proj": wall_tps * TAU,
                                    "beats_126378": int(wall_tps > BASELINE_OFFICIAL_TPS),
                                    "baseline_official_tps": BASELINE_OFFICIAL_TPS})
                rec["wandb_run_id"] = run.id
                (out_dir / "arm_result.json").write_text(json.dumps(rec, indent=2))
                wandb_logging.finish_wandb(run)
                print(f"[arm:{label}] W&B run {run.id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[arm:{label}] W&B logging skipped: {exc!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
