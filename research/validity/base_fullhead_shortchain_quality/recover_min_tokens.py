#!/usr/bin/env python3
"""PR #542 — min_tokens=8 EOS-guard recovery on the zeroed-item subset. LOCAL, NO FIRE.

The advisor's in-flight steer: report base_fullhead BOTH as-served AND with a
request-level min_tokens=8 EOS-guard, because the wirbel #541 GSM8K cell found
~10% of completions were immediate first-token-EOS empties that spuriously zero an
item. This recovers them WITHOUT a served-file change.

Strategy (cheap + exactly apples-to-apples):
  1. classify the as-served fullhead cells (failmodes.classify_eval) -> the genuine
     immediate-EOS empty ids per axis. min_tokens=8 can ONLY move those; truncations
     and other extract-fails are unaffected, so re-running them would be wasted GPU.
  2. if an axis has >=1 empty: re-serve base_fullhead with the IDENTICAL recipe
     (imported from run_2x2 so it can't drift) and re-run ONLY those ids with
     --min-tokens 8 on a byte-identical prompt (same seed + --ids-file).
  3. patch the recovered scores into the as-served per_sample -> the full-set
     min_tokens-adjusted score (fullhead_<axis>.mintok.json).
  4. axes with zero empties: adjusted == as-served (no re-serve needed).
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_2x2 as R  # noqa: E402  FULLHEAD_OVERRIDES/STOCK/SUBMISSION/EVAL_PY/QE/PORT/SEED/ROOT
import failmodes  # noqa: E402
from scripts.local_validation import harness  # noqa: E402

AXES = {
    "mmlu_pro": {
        "asserved": "fullhead_mmlu_pro.json", "recov": "_recov_fullhead_mmlu.json",
        "mintok": "fullhead_mmlu_pro.mintok.json", "ids": "_empty_ids_mmlu.json",
        "max_tokens": 2048, "n": 500,
    },
    "gpqa_diamond": {
        "asserved": "fullhead_gpqa.json", "recov": "_recov_fullhead_gpqa.json",
        "mintok": "fullhead_gpqa.mintok.json", "ids": "_empty_ids_gpqa.json",
        "max_tokens": 3072, "n": None,
    },
}


def run_recovery_cell(task: str, ids_file: Path, out: Path, max_tokens: int, n, conc: int) -> dict:
    cmd = [
        str(R.EVAL_PY), str(R.QE / "run_eval.py"), "--task", task, "--arm", "fullhead_mintok",
        "--out", str(out), "--seed", str(R.SEED), "--max-tokens", str(max_tokens),
        "--max-connections", str(conc), "--base-url", f"http://127.0.0.1:{R.PORT}/v1",
        "--model", "gemma-4-e4b-it", "--min-tokens", "8", "--ids-file", str(ids_file),
    ]
    if task == "mmlu_pro":
        cmd += ["--n", str(n)]
    print(f"[recover] {task} ->", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.load(open(out))


def patch_full_set(asserved: dict, recov_by_id: dict, out_path: Path) -> dict:
    d = copy.deepcopy(asserved)
    ps = d["per_sample"]
    flipped_to_correct = 0
    for i, r in enumerate(ps):
        sid = str(r["id"])
        if sid in recov_by_id:
            new = recov_by_id[sid]
            if (not r.get("correct")) and new.get("correct"):
                flipped_to_correct += 1
            ps[i] = new
    nc = sum(1 for r in ps if r.get("correct"))
    nscored = sum(1 for r in ps if r.get("value") in ("C", "I"))
    d["n_correct"] = nc
    d["n_scored"] = nscored
    d["accuracy"] = (nc / nscored) if nscored else float("nan")
    d["arm"] = "fullhead_mintok"
    d["min_tokens"] = 8
    d["recovered_n"] = len(recov_by_id)
    d["recovered_flipped_to_correct"] = flipped_to_correct
    Path(out_path).write_text(json.dumps(d, indent=2))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=32)
    a = ap.parse_args()

    # 1) classify the as-served fullhead cells.
    fm = {}
    for axis, cfg in AXES.items():
        p = HERE / cfg["asserved"]
        if not p.exists():
            raise SystemExit(f"[recover] as-served cell missing: {p} (let run_2x2 finish first)")
        fm[axis] = failmodes.classify_eval(p)
    (HERE / "failmodes_fullhead.json").write_text(json.dumps(fm, indent=2))

    total_empty = sum(len(fm[ax]["empty_eos_ids"]) for ax in AXES)
    print(f"[recover] empty-EOS ids: " + ", ".join(
        f"{ax}={len(fm[ax]['empty_eos_ids'])}" for ax in AXES) +
        f" (truncations: " + ", ".join(f"{ax}={fm[ax]['n_truncation']}" for ax in AXES) + ")",
        flush=True)

    asserved = {ax: json.load(open(HERE / AXES[ax]["asserved"])) for ax in AXES}

    if total_empty == 0:
        # min_tokens=8 is a confirmed no-op on both axes -> adjusted == as-served.
        for ax, cfg in AXES.items():
            d = copy.deepcopy(asserved[ax])
            d["arm"] = "fullhead_mintok"
            d["min_tokens"] = 8
            d["recovered_n"] = 0
            d["recovered_flipped_to_correct"] = 0
            (HERE / cfg["mintok"]).write_text(json.dumps(d, indent=2))
        print("[recover] zero immediate-EOS empties on both axes — min_tokens=8 is a "
              "no-op; adjusted == as-served. No re-serve needed.", flush=True)
        return 0

    # 2) re-serve base_fullhead (identical recipe) and recover the empty ids.
    R.assert_full_head(R.STOCK)
    submission_dir = (R.ROOT / "submissions" / R.SUBMISSION).resolve()
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    log = HERE / "server_fullhead_mintok.log"
    with harness.LocalServer(
        submission_dir, server_python=server_python, port=R.PORT,
        log_path=log, extra_env=R.FULLHEAD_OVERRIDES,
    ) as server:
        print(f"[recover] base_fullhead re-served model_id={server.model_id}", flush=True)
        for ax, cfg in AXES.items():
            empties = fm[ax]["empty_eos_ids"]
            if not empties:
                d = copy.deepcopy(asserved[ax])
                d["arm"] = "fullhead_mintok"; d["min_tokens"] = 8
                d["recovered_n"] = 0; d["recovered_flipped_to_correct"] = 0
                (HERE / cfg["mintok"]).write_text(json.dumps(d, indent=2))
                print(f"[recover] {ax}: 0 empties — adjusted == as-served", flush=True)
                continue
            (HERE / cfg["ids"]).write_text(json.dumps(empties))
            recov = run_recovery_cell(
                ax, HERE / cfg["ids"], HERE / cfg["recov"],
                cfg["max_tokens"], cfg["n"], a.conc,
            )
            recov_by_id = {str(r["id"]): r for r in recov["per_sample"]}
            d = patch_full_set(asserved[ax], recov_by_id, HERE / cfg["mintok"])
            print(f"[recover] {ax}: recovered {len(recov_by_id)} empties, "
                  f"{d['recovered_flipped_to_correct']} flipped to correct; "
                  f"as-served acc {asserved[ax]['accuracy']:.4f} -> mintok acc {d['accuracy']:.4f}",
                  flush=True)

    print("[recover] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
