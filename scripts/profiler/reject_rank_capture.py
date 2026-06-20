#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Capture reject-rank + verifier-entropy records on the int4head greedy verify path (PR #820).

Serves the shipped ``int4_mtp_bi0_int4head`` submission locally with
``REJECTRANK_ENABLE=1`` (the env-gated ``vllm_rejectrank_patch`` hook in its
``sitecustomize.py``), drives the OFFICIAL 128-prompt greedy decode workload
(conc=1, temp=0, ignore_eos, output_len 512 — the same protocol as the leaderboard
benchmark via ``harness.capture_decode``), and collects the per-draft-position JSONL
the probe writes in the engine-core worker.

The probe only ADDS logging and calls the original ``RejectionSampler.forward``
unchanged, so the served greedy tokens are byte-identical to production. We
cross-check that the probe's own strict acceptance length reproduces vLLM's own
server-log E_accept (parsed via ``serve_profile.parse_spec_log``) — if the probe
were perturbing the path those two would diverge.

This is GPU capture only. The offline CDF + TPS projection (uniform top-k vs
entropy-gated top-k, FLy-inspired) lives in ``reject_rank_project.py`` and reads the
JSONL this script writes — so the analysis can be re-run without re-serving.

    python -m scripts.profiler.reject_rank_capture \
        --num-prompts 128 --output-len 512 \
        --out-dir research/reject_rank_entropy/int4head

Use ``--num-prompts 2 --output-len 32`` for a smoke that proves records flow and the
strict acceptance length matches the baseline before the full run.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.local_validation import serve_profile  # noqa: E402

DEFAULT_SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_int4head"


def _count_records(shards: list[str]) -> dict[str, object]:
    """Cheap pass over the JSONL shards: total records + strict acceptance length.

    Strict acceptance length per block = (index of first non-accept) + 1 bonus,
    i.e. ``fd + 1`` where fd is the first depth with acc==0 (or n if all accepted).
    Its mean over blocks must reproduce the baseline E_accept ~3.379 — the probe's
    self-consistency proof that it sees the real greedy accept path.
    """
    n_rec = 0
    n_draft_pos = 0
    n_reject = 0
    sum_len = 0.0
    for path in shards:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                acc = rec["acc"]
                n = rec["n"]
                n_rec += 1
                n_draft_pos += n
                n_reject += sum(1 for a in acc if a == 0)
                fd = n
                for d in range(n):
                    if acc[d] == 0:
                        fd = d
                        break
                sum_len += fd + 1
    return {
        "n_blocks": n_rec,
        "n_draft_positions": n_draft_pos,
        "n_rejects": n_reject,
        "strict_reject_rate": (n_reject / n_draft_pos) if n_draft_pos else None,
        "strict_E_accept_mean_len": (sum_len / n_rec) if n_rec else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "research" / "reject_rank_entropy" / "int4head")
    ap.add_argument("--server-python", type=Path, default=None,
                    help="python with the submission venv (auto-resolved if absent)")
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    args = ap.parse_args(argv)

    submission = args.submission.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[capture] {note}", flush=True)

    manifest = harness.load_manifest(submission)
    server_python = args.server_python if (args.server_python and args.server_python.exists()) \
        else harness.ensure_server_venv(manifest["dependencies"])
    print(f"[capture] submission={submission}", flush=True)
    print(f"[capture] server_python={server_python}", flush=True)

    records_base = out_dir / "rejectrank_records.jsonl"
    # Clear any stale shards from a previous run so the count is clean.
    for old in glob.glob(str(records_base) + ".*"):
        Path(old).unlink()

    server_log = out_dir / "server.log"
    decode_out = out_dir / "decode.jsonl"
    decode_summary = out_dir / "decode.summary.json"

    extra_env = {
        "REJECTRANK_ENABLE": "1",
        "REJECTRANK_OUTPUT": str(records_base),
        # Surface vLLM's own SpecDecoding counters so we can cross-check the probe's
        # strict E_accept against the engine's (parse_spec_log). The leaderboard serve
        # path ships --disable-log-stats; re-enabling stats is host-side only.
        "DISABLE_LOG_STATS": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }

    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=args.port, log_path=server_log,
        extra_env=extra_env, startup_timeout_s=args.startup_timeout_s,
    ) as srv:
        ready_s = time.time() - t0
        print(f"[capture] server ready in {ready_s:.0f}s; driving "
              f"{args.num_prompts} prompts x output_len {args.output_len}", flush=True)
        t1 = time.time()
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_summary,
            num_prompts=args.num_prompts, output_len=args.output_len, timeout_s=7200,
        )
        decode_wall = time.time() - t1
        print(f"[capture] decode done in {decode_wall:.0f}s "
              f"(completed={summary.get('completed')})", flush=True)

    # Probe shards (PID-suffixed). The engine-core worker is the only one that emits.
    shards = sorted(glob.glob(str(records_base) + ".*"))
    shards = [s for s in shards if not s.endswith(".meta.json")]
    counts = _count_records(shards)

    # vLLM's own whole-run E_accept from the server log — the independent cross-check.
    spec_log = serve_profile.parse_spec_log(server_log.read_text())

    result = {
        "submission": str(submission),
        "server_python": str(server_python),
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "decode_summary": summary,
        "decode_wall_s": decode_wall,
        "server_ready_s": ready_s,
        "record_shards": shards,
        "probe_counts": counts,
        "vllm_server_log_spec": spec_log,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "capture_result.json").write_text(json.dumps(result, indent=2))

    print("\n========== REJECT-RANK CAPTURE ==========", flush=True)
    print(f"shards: {len(shards)}  blocks: {counts['n_blocks']}  "
          f"draft_positions: {counts['n_draft_positions']}", flush=True)
    print(f"probe strict reject-rate     = {counts['strict_reject_rate']}", flush=True)
    print(f"probe strict E_accept (len)  = {counts['strict_E_accept_mean_len']}", flush=True)
    print(f"vLLM server-log E_accept     = {spec_log.get('e_accept_exact')} "
          f"(rate {spec_log.get('draft_acceptance_rate')})", flush=True)
    print(f"decode completed             = {summary.get('completed')}/{args.num_prompts}", flush=True)
    print(f"artifacts -> {out_dir}", flush=True)
    if not shards:
        print("[capture] WARNING: no probe shards written — check REJECTRANK hook", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
