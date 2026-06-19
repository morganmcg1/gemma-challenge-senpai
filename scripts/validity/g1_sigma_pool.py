#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Pool several fresh-serve noise-floor batches into one N-run artifact -- PR #766.

``research/tps_noise_floor/run_noise_floor.py`` has no resume/append mode (it
opens ``records.jsonl`` with ``"w"`` and runs ``--n-runs`` from scratch), so the
supported way to reach the PR's N>=10 fresh-serve sample without discarding an
in-flight batch is to run two batches and POOL their per-run records.

Pooling is statistically exact for the run-to-run CV the G1 gate worries about:
every batch is the SAME fire config (int4_mtp_batchinv, BI=1), SAME 128x512 conc=1
seed-1 workload, fresh server per run -- the only difference between batches is
wall-clock time, which is precisely the run-to-run hardware/scheduling jitter we
are trying to sample. Pooling two batches widens the time window, making the CV
estimate MORE representative of the official scorer's cold-server draws, not less.

The pooled aggregate is recomputed with run_noise_floor.aggregate() BYTE-FOR-BYTE
(same fmean / stdev / cv_pct path the harness writes per batch), so the pooled
``noise_floor_fresh.json`` is indistinguishable in schema from a single N-run
artifact and drops straight into scripts/validity/g1_sigma_measured.py.

LOCAL ONLY, CPU only: reads in-repo records, writes one JSON. No serve, no GPU,
no submission, no HF Job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tps_noise_floor.run_noise_floor import aggregate  # noqa: E402

AGG_KEYS = ("steady_gen_tps_mean", "steady_gen_tps_mean_nonzero", "wall_tps",
            "e_accept_exact")


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Records from a batch: a records.jsonl OR a noise_floor_*.json (records[])."""
    if path.is_dir():
        path = path / "records.jsonl"
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    obj = json.loads(text)
    return list(obj.get("records") or [])


def pool(batch_paths: list[Path], *, submission: str, mode: str) -> dict[str, Any]:
    pooled: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for p in batch_paths:
        recs = _load_records(p)
        if not recs:
            raise SystemExit(f"no records in {p}")
        base = len(pooled)
        for j, r in enumerate(recs):
            r = dict(r)
            r["pool_source"] = str(p)
            r["pool_source_run_idx"] = r.get("run_idx")
            r["run_idx"] = base + j  # re-index 0..N-1 across the pool
            pooled.append(r)
        provenance.append({"path": str(p), "n": len(recs),
                           "wall_tps": [r.get("wall_tps") for r in recs]})
    agg = {k: aggregate(pooled, k) for k in AGG_KEYS}
    wl = pooled[0]
    return {
        "mode": mode,
        "submission": submission,
        "n_runs": len(pooled),
        "pooled": True,
        "pool_provenance": provenance,
        "workload": {"num_prompts": wl.get("num_prompts"),
                     "output_len": wl.get("output_len"), "seed": wl.get("seed")},
        "aggregate": agg,
        "records": pooled,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batches", nargs="+",
                    help="batch dirs or records.jsonl / noise_floor_*.json files to pool")
    ap.add_argument("--submission", default="int4_mtp_batchinv")
    ap.add_argument("--mode", default="fresh", choices=["fresh", "reuse"])
    ap.add_argument("--out", required=True, help="output noise_floor_fresh.json path")
    args = ap.parse_args(argv)

    out = pool([Path(b) for b in args.batches], submission=args.submission, mode=args.mode)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    a = out["aggregate"]["wall_tps"]
    print(f"[pool] pooled {len(args.batches)} batch(es) -> N={out['n_runs']} runs", flush=True)
    for pr in out["pool_provenance"]:
        print(f"[pool]   {pr['n']:2d} runs  wall_tps={[round(v,3) for v in pr['wall_tps']]}  "
              f"<- {pr['path']}", flush=True)
    print(f"[pool] wall_tps: n={a['n']} mean={a['mean']:.4f} std={a['std']:.4f} "
          f"CV={a['cv_pct']:.4f}% range={a['min']:.3f}..{a['max']:.3f} "
          f"({a['range_pct']:.4f}%)", flush=True)
    print(f"[pool] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
