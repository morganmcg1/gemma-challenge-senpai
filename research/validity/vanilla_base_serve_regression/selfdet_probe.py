#!/usr/bin/env python3
"""PR #557 — self-determinism probe for the recovered (ple_fold) serve. NO FIRE.

Brings up ONE ple_fold server (plain vanilla serve + the vLLM-native PLE embed-scale
fold, the recovered HEALTHY config) and runs the SAME small byte-identical MMLU-Pro
subset through the greedy client TWICE, then compares the full completion text per item
(read from the two inspect .eval logs). self_det=true iff every item's completion is
byte-identical across the two passes.

Per the program's greedy-identity policy, int4+vLLM run-to-run nondeterminism is
NOT a gate blocker (the strict gate compares WITHIN a stack/endpoint); this probe
just records whether the recovered serve is self-consistent. One model load, two
quick passes, then teardown. Writes selfdet.json.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_arm  # noqa: E402

HERE = run_arm.HERE
EVAL_PY = run_arm.EVAL_PY
QE = run_arm.QE
N_PROBE = 24  # small: greedy self-determinism is a per-item identity check, not a power test


def _completions(eval_log: str) -> dict:
    """{id: completion_text} from a .eval log, via the eval venv (has inspect_ai)."""
    code = (
        "import json,sys\n"
        "from inspect_ai.log import read_eval_log\n"
        "log=read_eval_log(sys.argv[1])\n"
        "out={}\n"
        "for s in (log.samples or []):\n"
        "    try:\n"
        "        o=s.output; out[str(s.id)]=(o.completion or '') if o else ''\n"
        "    except Exception:\n"
        "        out[str(s.id)]=None\n"
        "json.dump(out,sys.stdout)\n"
    )
    res = subprocess.run([str(EVAL_PY), "-c", code, eval_log],
                         check=True, text=True, capture_output=True)
    return json.loads(res.stdout)


def main() -> int:
    run_arm.wait_gpu_free()
    log = HERE / "server_selfdet.log"
    proc = run_arm.start_server("ple_fold", log)
    out = {"arm": "ple_fold", "n_probe": N_PROBE,
           "build": "vllm-0.22.1rc1.dev307+g3e8afdf78"}
    try:
        run_arm.wait_ready(proc)
        print("[selfdet] server READY — running two greedy passes", flush=True)
        outs = []
        for tag in ("a", "b"):
            o = HERE / f"selfdet_pass_{tag}.json"
            cmd = [
                str(EVAL_PY), str(QE / "run_eval.py"), "--task", "mmlu_pro",
                "--arm", f"selfdet_{tag}", "--out", str(o), "--seed", str(run_arm.SEED),
                "--n", "500", "--limit", str(N_PROBE), "--max-tokens", "2048",
                "--max-connections", "32", "--base-url", f"http://127.0.0.1:{run_arm.PORT}/v1",
                "--model", "gemma-4-e4b-it",
            ]
            subprocess.run(cmd, check=True)
            outs.append(json.load(open(o)))
        a_log, b_log = outs[0]["eval_log"], outs[1]["eval_log"]
        ca, cb = _completions(a_log), _completions(b_log)
        ids = sorted(set(ca) & set(cb))
        identical = [i for i in ids if ca[i] is not None and ca[i] == cb[i]]
        # also compare scored answers (a coarser, scoring-level identity)
        ans_a = {r["id"]: (r.get("answer"), r.get("value")) for r in outs[0]["per_sample"]}
        ans_b = {r["id"]: (r.get("answer"), r.get("value")) for r in outs[1]["per_sample"]}
        ans_ids = sorted(set(ans_a) & set(ans_b))
        ans_match = [i for i in ans_ids if ans_a[i] == ans_b[i]]
        out.update({
            "n_compared": len(ids),
            "n_identical_completion": len(identical),
            "completion_identity_rate": (len(identical) / len(ids)) if ids else None,
            "n_answer_compared": len(ans_ids),
            "n_answer_match": len(ans_match),
            "answer_identity_rate": (len(ans_match) / len(ans_ids)) if ans_ids else None,
            "self_det": bool(ids and len(identical) == len(ids)),
            "self_det_answer_level": bool(ans_ids and len(ans_match) == len(ans_ids)),
            "pass_a_log": a_log, "pass_b_log": b_log,
        })
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            pass
    (HERE / "selfdet.json").write_text(json.dumps(out, indent=2))
    print(f"[selfdet] self_det={out.get('self_det')} "
          f"completion {out.get('n_identical_completion')}/{out.get('n_compared')} "
          f"answer {out.get('n_answer_match')}/{out.get('n_answer_compared')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
