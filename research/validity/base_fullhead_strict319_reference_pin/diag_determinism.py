#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #585 diagnostic -- characterize the self_det=False free-running non-determinism.

The full census found ``self_det=False`` (1/8 free-running re-captures byte-identical).
This isolates WHERE the non-determinism lives so the headline teacher-forced flip rate
(0.0676) can be reported with calibrated confidence:

(1) TEACHER-FORCED reproducibility. Run phase_teacher_forced TWICE on the first
    N_TF prompts and compare per-position argmax. The teacher-forced rate conditions
    every position on the FIXED bf16 reference context (max_tokens=1, no cascade), so
    if it is reproducible run-to-run, 0.0676 is a clean per-position number and the
    free-running self_det failure is purely cascade amplification of a few near-tie
    single-forward flips. ``tf_self_consistency_rate`` -> 1.0 means clean.

(2) FREE-RUNNING self-divergence locus. Run phase_free_running TWICE on the first
    N_FR prompts and, per prompt, find the first index where run-1 and run-2 differ.
    A low median locus that then cascades is the batch-invariance signature: surgical357
    deliberately does NOT set VLLM_BATCH_INVARIANT=1 (keeps the fast Marlin path), so a
    single near-tie ULP flip early in the 512-token stream cascades the whole sequence.

SCOPE: LOCAL A10G, analysis-only, spec-OFF base_fullhead, same recipe as the census.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pin_strict319_reference as pin  # noqa: E402  (same dir)

N_TF = 24   # prompts for the teacher-forced reproducibility leg
N_FR = 8    # prompts for the free-running self-divergence leg
OUT = HERE / "diag_determinism_results.json"


def tf_self_consistency(tf_a, tf_b) -> dict:
    a = {r["id"]: r["argmax_token_ids"] for r in tf_a}
    b = {r["id"]: r["argmax_token_ids"] for r in tf_b}
    common = sorted(set(a) & set(b))
    n_pos = 0
    n_disagree = 0
    disagree_examples = []
    for k in common:
        for i, (x, y) in enumerate(zip(a[k], b[k])):
            if x is None or y is None:
                continue
            n_pos += 1
            if x != y:
                n_disagree += 1
                if len(disagree_examples) < 5:
                    disagree_examples.append({"id": k, "pos": i, "run1": x, "run2": y})
    return {
        "n_positions": n_pos,
        "n_disagree": n_disagree,
        "tf_self_consistency_rate": (1.0 - n_disagree / n_pos) if n_pos else None,
        "tf_self_consistent": bool(n_disagree == 0 and n_pos > 0),
        "disagree_examples": disagree_examples,
    }


def fr_self_divergence(fr_a, fr_b) -> dict:
    a = {r["id"]: r["completion_token_ids"] for r in fr_a}
    b = {r["id"]: r["completion_token_ids"] for r in fr_b}
    common = sorted(set(a) & set(b))
    loci = []
    n_identical = 0
    per_prompt = []
    for k in common:
        x, y = a[k], b[k]
        n = min(len(x), len(y))
        fi = next((i for i in range(n) if x[i] != y[i]), None)
        if fi is None and len(x) == len(y):
            n_identical += 1
            per_prompt.append({"id": k, "first_self_divergence": None})
        else:
            locus = fi if fi is not None else n
            loci.append(locus)
            per_prompt.append({"id": k, "first_self_divergence": locus})
    return {
        "n_prompts": len(common),
        "n_self_identical": n_identical,
        "median_first_self_divergence": pin._median_int(loci) if loci else None,
        "min_first_self_divergence": min(loci) if loci else None,
        "max_first_self_divergence": max(loci) if loci else None,
        "per_prompt": per_prompt,
    }


def main() -> int:
    from scripts.local_validation import harness, paths

    for note in paths.prepare_local_gpu_env():
        print(f"[diag] {note}", flush=True)

    records = pin.load_reference()
    n = max(N_TF, N_FR)
    records = records[:n]
    print(f"[diag] {len(records)} reference records", flush=True)

    manifest = harness.load_manifest(pin.SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    log_path = HERE / "server_diag_determinism.log"

    t0 = time.time()
    with harness.LocalServer(
        pin.SUBMISSION, server_python=server_python, port=8000,
        startup_timeout_s=1800, log_path=log_path,
        extra_env=pin.base_fullhead_refmode_env(),
    ) as srv:
        model = srv.served_model_name
        print(f"[diag] served base_fullhead spec-off at {srv.base_url} in {time.time()-t0:.0f}s", flush=True)

        # (1) teacher-forced reproducibility (no cascade; conditions on fixed bf16 ctx).
        tf_a = pin.phase_teacher_forced(records[:N_TF], srv.base_url, model)
        tf_b = pin.phase_teacher_forced(records[:N_TF], srv.base_url, model)
        tf_diag = tf_self_consistency(tf_a, tf_b)
        print(f"[diag] TF self-consistency rate={tf_diag['tf_self_consistency_rate']} "
              f"(disagree {tf_diag['n_disagree']}/{tf_diag['n_positions']})", flush=True)

        # (2) free-running self-divergence locus (cascade-prone).
        fr_a = pin.phase_free_running(records[:N_FR], srv.base_url, model, output_len=512, tag="FR-a")
        fr_b = pin.phase_free_running(records[:N_FR], srv.base_url, model, output_len=512, tag="FR-b")
        fr_diag = fr_self_divergence(fr_a, fr_b)
        print(f"[diag] FR self-divergence median={fr_diag['median_first_self_divergence']} "
              f"min={fr_diag['min_first_self_divergence']} identical={fr_diag['n_self_identical']}/{fr_diag['n_prompts']}",
              flush=True)

    out = {
        "schema": "base_fullhead_strict319_determinism_diag_v1",
        "pr": 585, "agent": "wirbel", "analysis_only": True,
        "n_tf_prompts": N_TF, "n_fr_prompts": N_FR,
        "teacher_forced_reproducibility": tf_diag,
        "free_running_self_divergence": fr_diag,
        "serve_env": pin.base_fullhead_refmode_env(),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[diag] wrote {OUT}", flush=True)
    print("SENPAI-DIAG " + json.dumps({
        "tf_self_consistency_rate": tf_diag["tf_self_consistency_rate"],
        "tf_self_consistent": tf_diag["tf_self_consistent"],
        "fr_median_first_self_divergence": fr_diag["median_first_self_divergence"],
        "fr_min_first_self_divergence": fr_diag["min_first_self_divergence"],
        "fr_n_self_identical": fr_diag["n_self_identical"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
