#!/usr/bin/env python3
"""PR #782 — run ONE drafter cell of the bi0 ngram-vs-MTP A/B and record evidence.

Serves a submission via the local-validation harness ``LocalServer`` (so the
serve path, venv, env, and tokenizer are byte-identical to the committed bi0
greedy reference), then:

  * captures the official greedy decode (decode_outputs.py: temp=0, seed=1,
    return_token_ids) -> decode_outputs.jsonl + decode_summary.json (wall_tps =
    num_completion_tokens / duration_s, the official output_throughput proxy);
  * scrapes the vLLM spec-decode prometheus counters from /metrics ->
    accepted/draft acceptance rate and mean accepted tokens per verify step
    (E[T] = accepted/drafts + 1), the apples-to-apples accepted-tokens/step lever;
  * optionally runs the official PPL gate (ppl_endpoint.py over the 128 GT
    records). PPL is a prefill-only prompt_logprobs forward that never invokes
    the drafter, so it is drafter-invariant; we measure it on the control and one
    ngram cell to prove that empirically.

The drafter is selected purely by ``--extra-env`` (e.g. SPECULATIVE_METHOD=ngram,
NUM_SPECULATIVE_TOKENS, PROMPT_LOOKUP_MAX/MIN), so the SAME submission stack
isolates exactly one variable: the proposer. LOCAL ONLY; no HF Job.

Greedy-identity vs the reference is judged OFFLINE afterwards by
``scripts/local_validation/greedy_gate.compare`` (see analyze.py); this script
only produces the per-cell decode/metrics/ppl artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import harness, paths  # noqa: E402

# vLLM 0.22.0 spec-decode prometheus counters (confirmed in the pinned wheel).
_SPEC_METRICS = {
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "draft": "vllm:spec_decode_num_draft_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
}


def scrape_spec_metrics(base_url: str) -> dict[str, float | None]:
    """Sum each spec-decode counter across label sets from /metrics."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=10.0) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    out: dict[str, float | None] = {}
    for key, name in _SPEC_METRICS.items():
        total = None
        pat = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9eE.+-]+)\s*$", re.M)
        for m in pat.finditer(text):
            try:
                total = (total or 0.0) + float(m.group(1))
            except ValueError:
                pass
        out[key] = total
    acc, draft, drafts = out.get("accepted"), out.get("draft"), out.get("drafts")
    out["acceptance_rate"] = (acc / draft) if (acc is not None and draft) else None
    out["mean_accepted_per_step"] = (acc / drafts) if (acc is not None and drafts) else None
    out["mean_tokens_per_step_ET"] = (
        (acc / drafts + 1.0) if (acc is not None and drafts) else None
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True, help="submission dir (rel to repo or abs)")
    ap.add_argument("--label", required=True, help="cell label, used for the output subdir")
    ap.add_argument("--extra-env", default="{}", help="JSON dict of env overrides for serve.py")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ppl", action="store_true", help="also run the official 128-record PPL gate")
    ap.add_argument("--out-root", default=str(REPO / "research/ngram_spec_782/cells"))
    args = ap.parse_args()

    submission = Path(args.submission)
    if not submission.is_absolute():
        submission = (REPO / submission).resolve()
    extra_env = json.loads(args.extra_env)
    out_dir = Path(args.out_root) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    summary: dict[str, object] = {
        "label": args.label,
        "submission": str(submission),
        "extra_env": extra_env,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "tokenizer": paths.TOKENIZER,
        "seed": paths.SEED,
    }
    log_path = out_dir / "serve.log"
    base_url = f"http://127.0.0.1:{args.port}"
    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=args.port,
        log_path=log_path, extra_env=extra_env,
    ) as server:
        decode_summary = harness.capture_decode(
            server_python,
            base_url=base_url,
            model=server.served_model_name,
            out_file=out_dir / "decode_outputs.jsonl",
            summary_file=out_dir / "decode_summary.json",
            num_prompts=args.num_prompts,
            output_len=args.output_len,
        )
        # Scrape spec-decode counters immediately after decode (before any PPL
        # prefill pass), so the cumulative counters reflect exactly the decode run.
        spec_metrics = scrape_spec_metrics(base_url)
        ppl_summary = None
        if args.ppl:
            ppl_summary = harness.run_ppl(
                server_python,
                base_url=base_url,
                model=server.served_model_name,
                out_file=out_dir / "ppl_results.jsonl",
                summary_file=out_dir / "ppl_summary.json",
            )

    dur = decode_summary.get("duration_s") or 0.0
    toks = decode_summary.get("num_completion_tokens") or 0
    summary["wall_tps"] = (toks / dur) if dur else 0.0
    summary["decode_num_records"] = decode_summary.get("num_records")
    summary["decode_num_completion_tokens"] = toks
    summary["decode_duration_s"] = dur
    summary["spec_metrics"] = spec_metrics
    if ppl_summary is not None:
        summary["ppl"] = ppl_summary.get("ppl")
        summary["ppl_num_records"] = ppl_summary.get("num_records")
        summary["ppl_num_tokens"] = ppl_summary.get("num_tokens")
    summary["serve_plus_eval_wall_s"] = time.time() - t0

    (out_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\n[run_cell] CELL SUMMARY", flush=True)
    print(
        f"CELL label={args.label} wall_tps={summary['wall_tps']:.4f} "
        f"acc_rate={spec_metrics.get('acceptance_rate')} "
        f"ET_tok_per_step={spec_metrics.get('mean_tokens_per_step_ET')} "
        f"ppl={summary.get('ppl')} records={summary.get('decode_num_records')}",
        flush=True,
    )
    print(f"[run_cell] artifacts in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
