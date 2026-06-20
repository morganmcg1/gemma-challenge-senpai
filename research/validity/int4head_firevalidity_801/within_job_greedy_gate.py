#!/usr/bin/env python
"""WITHIN-JOB spec-off vs spec-on greedy-identity gate for int4head (PR #801).

Serves the int4head submission TWICE inside ONE process:

  1. speculation OFF (M=1 AR, SENPAI_REFERENCE_MODE=1)  -> greedy REFERENCE
  2. speculation ON  (shipping K=6 MTP drafter)         -> greedy CANDIDATE

then compares them with the official ``greedy_identity`` verifier. Both captures
hit the SAME freshly-built checkpoint on the SAME engine/venv/GPU minutes apart,
so the ONLY removed variable is speculation. This is the apples-to-apples gate
the cached ``research/greedy_reference/`` files cannot provide (stale / cross-job
references read ~99/128 DIVERGENT even for a shipped submission against itself).

The spec-OFF server also runs PPL (teacher-forced, drafter-independent) and the
128-prompt decode capture, which together are the Step-0 sanity gate
(PPL ~ 2.0029, 128/128 completions).

LOCAL ONLY. No HF job, no submission.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402


class PeakMem:
    """Background sampler of GPU memory.used (MiB); tracks the peak."""

    def __init__(self, period_s: float = 3.0) -> None:
        self.period_s = period_s
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            return max(int(x) for x in out.stdout.split())
        except Exception:
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._sample())
            self._stop.wait(self.period_s)

    def __enter__(self) -> "PeakMem":
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self.peak = max(self.peak, self._sample())


def capture(
    submission: Path, server_python: Path, out_dir: Path, *, label: str,
    extra_env: dict[str, str], reference_kind: str, num_prompts: int,
    output_len: int, seed: int, run_ppl: bool, peak: PeakMem,
) -> dict[str, Any]:
    """Serve once with ``extra_env`` and capture decode (+ optional PPL)."""
    stage = out_dir / label
    stage.mkdir(parents=True, exist_ok=True)
    out = stage / "decode_outputs.jsonl"
    summary_file = stage / "decode_summary.json"
    log_path = stage / "server.log"
    res: dict[str, Any] = {"label": label, "extra_env": extra_env,
                           "reference_kind": reference_kind}

    print(f"\n[{label}] serving int4head (extra_env={extra_env}) ...", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=args_port,
        log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        res["serve_ready_s"] = round(time.time() - t0, 1)
        res["model_id"] = srv.model_id
        peak.peak = max(peak.peak, peak._sample())
        if run_ppl:
            print(f"[{label}] PPL (teacher-forced sanity) ...", flush=True)
            ppl_summary = harness.run_ppl(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=stage / "ppl.jsonl", summary_file=stage / "ppl.summary.json",
            )
            res["ppl"] = ppl_summary.get("ppl")
            res["ppl_num_tokens"] = ppl_summary.get("num_tokens")
            print(f"[{label}] PPL={res['ppl']} over {res['ppl_num_tokens']} tokens", flush=True)
        print(f"[{label}] greedy decode {num_prompts}x{output_len} (temp=0) ...", flush=True)
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=out, summary_file=summary_file, num_prompts=num_prompts,
            output_len=output_len, seed=seed,
        )
        peak.peak = max(peak.peak, peak._sample())
        res["served_model_name"] = srv.served_model_name
    res["num_records"] = summary["num_records"]
    res["num_completion_tokens"] = summary["num_completion_tokens"]
    res["decode_outputs"] = str(out)

    # meta.json compatible with log_greedy_gate_wandb.py provenance read.
    meta = {
        "model_id": res["model_id"],
        "reference_kind": reference_kind,
        "spec_off": extra_env.get(paths.REFERENCE_MODE_ENV, "") not in ("", "0"),
        "ref_env": extra_env,
        "num_records": summary["num_records"],
        "num_completion_tokens": summary["num_completion_tokens"],
        "output_len": output_len,
        "seed": seed,
        "tokenizer": paths.TOKENIZER,
    }
    (stage / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=REPO / "submissions/int4_mtp_bi0_int4head")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "research/validity/int4head_firevalidity_801/withinjob")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ppl-cap", type=float, default=2.42)
    ap.add_argument("--no-ppl", action="store_true",
                    help="skip the PPL sanity (smoke plumbing checks)")
    args = ap.parse_args()
    global args_port
    args_port = args.port

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    submission = args.submission
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = harness.load_manifest(submission)
    print(f"[setup] submission={submission} model_id(manifest)={manifest.get('model_id')}", flush=True)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[setup] server_python={server_python}", flush=True)

    report: dict[str, Any] = {
        "submission": str(submission),
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "seed": args.seed,
        "ppl_cap": args.ppl_cap,
    }

    with PeakMem() as peak:
        # ---- Stage 1: spec-OFF reference (+ PPL / completion sanity) ----------
        ref = capture(
            submission, server_python, out_dir, label="specoff_ref",
            extra_env={paths.REFERENCE_MODE_ENV: "1"},
            reference_kind="served_spec_off", num_prompts=args.num_prompts,
            output_len=args.output_len, seed=args.seed, run_ppl=not args.no_ppl, peak=peak,
        )
        # ---- Stage 2: spec-ON candidate (shipping K=6 MTP) --------------------
        cand = capture(
            submission, server_python, out_dir, label="specon_cand",
            extra_env={}, reference_kind="served_spec_on",
            num_prompts=args.num_prompts, output_len=args.output_len,
            seed=args.seed, run_ppl=False, peak=peak,
        )
        report["peak_gpu_mem_mib"] = peak.peak

    report["specoff_ref"] = ref
    report["specon_cand"] = cand

    # ---- Stage 3: official greedy-identity compare (within-job ref) ----------
    ref_path = Path(ref["decode_outputs"])
    cand_path = Path(cand["decode_outputs"])
    print(f"\n[compare] reference (within-job spec-off) = {ref_path}", flush=True)
    print(f"[compare] candidate (within-job spec-on)  = {cand_path}", flush=True)
    cmp_report = greedy_gate.compare(ref_path, cand_path)
    greedy_gate._print_human(cmp_report)

    cmp_dict = cmp_report.to_dict()
    (out_dir / "greedy_report.json").write_text(json.dumps(cmp_dict, indent=2))
    onset = greedy_gate.onset_summary(cmp_report)

    divergent = [
        {"key": p.key, "first_divergence_index": p.first_divergence_index}
        for p in cmp_report.per_prompt if not p.identical
    ]
    total_tok = cmp_dict.get("total_tokens_compared", 0) or 0
    total_div = cmp_dict.get("total_divergent_tokens", 0) or 0
    report["greedy"] = {
        "verdict": cmp_report.verdict,
        "num_prompts_compared": cmp_report.num_prompts_compared,
        "num_identical": cmp_report.num_identical,
        "num_divergent": cmp_report.num_divergent,
        "total_tokens_compared": total_tok,
        "total_divergent_tokens": total_div,
        "flip_rate_per_token": (total_div / total_tok) if total_tok else 0.0,
        "onset": onset,
        "divergent_prompts": divergent,
    }

    # Sanity-gate verdict (Step 0)
    ppl = ref.get("ppl")
    report["sanity"] = {
        "ppl": ppl,
        "ppl_pass": (ppl is not None and ppl <= args.ppl_cap),
        "specoff_records": ref.get("num_records"),
        "specon_records": cand.get("num_records"),
        "completions_128": ref.get("num_records") == args.num_prompts
        and cand.get("num_records") == args.num_prompts,
    }

    (out_dir / "result.json").write_text(json.dumps(report, indent=2))

    g = report["greedy"]
    print("\n" + "=" * 70, flush=True)
    print("WITHIN-JOB int4head SPEC-OFF vs SPEC-ON GREEDY GATE (PR #801)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Step-0 PPL (spec-off)        : {ppl}  (cap {args.ppl_cap}, pass={report['sanity']['ppl_pass']})", flush=True)
    print(f"  completions  spec-off/spec-on: {ref.get('num_records')}/{args.num_prompts}, {cand.get('num_records')}/{args.num_prompts}", flush=True)
    print(f"  VERDICT                      : {g['verdict']}", flush=True)
    print(f"  greedy match  N/128          : {g['num_identical']}/{g['num_prompts_compared']}", flush=True)
    print(f"  divergent prompts            : {g['num_divergent']}", flush=True)
    print(f"  token flip rate              : {g['flip_rate_per_token']:.6f} ({total_div}/{total_tok} tok)", flush=True)
    print(f"  {greedy_gate.onset_line(onset, args.output_len)}", flush=True)
    print(f"  peak GPU mem (MiB)           : {report['peak_gpu_mem_mib']}", flush=True)
    print("=" * 70, flush=True)
    print(f"[done] result -> {out_dir / 'result.json'}", flush=True)
    return 0


args_port = 8000

if __name__ == "__main__":
    raise SystemExit(main())
