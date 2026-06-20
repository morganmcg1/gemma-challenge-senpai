"""PR #794 determinism control: is the 2D-vs-3D divergence the 3D effect, or
cross-session int4 nondeterminism?

The e2e 2D-vs-3D greedy-compare (4/128 prompts) compares two SEPARATE serve
sessions that differ in the VLLM_SURGATTN toggle. But the int4 (Marlin) + vLLM
stack is known to be non-bit-reproducible ACROSS serve sessions (separate
process => possibly different kernel autotune / reduction order), so some of
that 4/128 could be session noise rather than the 3D split-KV reassociation.

This runs two MORE independent sessions — a second force-2D (2d_b) and a second
3D (3d_b) — decode only, same protocol (128 prompts, seed=1, output_len=512).
Then every pairwise greedy-compare:

  2d_a vs 2d_b   baseline cross-session noise floor (force-2D, identical config)
  3d_a vs 3d_b   3D cross-session noise floor
  2d_a vs 3d_a   cross-config (the e2e headline)  [reuses existing files]
  2d_a vs 3d_b / 2d_b vs 3d_a   cross-config, cross-session

If 2d_a-vs-2d_b ~ 2d_a-vs-3d_a, the divergence is dominated by session noise and
3D is "no worse than 2D is to itself". If 2d_a-vs-2d_b ~ 0 << 2d_a-vs-3d_a, the
4/128 is cleanly the 3D reassociation.

    CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python determinism_control.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

SERVE_PY = Path("/senpai-run/home/student-stark/.venvs/vllm022/bin/python")
SUBMISSION = REPO / "submissions" / "int4_mtp_bi0_surgattn"
OUTDIR = Path(__file__).resolve().parent / "e2e"

# new sessions to capture (arm-file -> VLLM_SURGATTN). 2d_a / 3d_a already exist.
NEW_SESSIONS = {"2d_b": "1", "3d_b": "0"}


def serve_decode(tag: str, surgattn: str) -> dict:
    decode_jsonl = OUTDIR / f"decode_{tag}.jsonl"
    decode_summ = OUTDIR / f"decode_{tag}_summary.json"
    log_path = OUTDIR / f"server_{tag}.log"
    extra_env = {"VLLM_SURGATTN": surgattn, "VLLM_BATCH_INVARIANT": "0",
                 "VLLM_USE_FLASHINFER_SAMPLER": "0", "CUDA_VISIBLE_DEVICES": "0"}
    t0 = time.time()
    with harness.LocalServer(SUBMISSION, server_python=SERVE_PY, port=8000,
                             log_path=log_path, extra_env=extra_env) as srv:
        dsum = harness.capture_decode(
            SERVE_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_jsonl, summary_file=decode_summ,
            num_prompts=paths.NUM_PROMPTS, output_len=paths.OUTPUT_LEN,
            seed=paths.SEED)
    log_txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
    return {
        "tag": tag, "surgattn": surgattn, "wall_s": round(time.time() - t0, 1),
        "num_records": dsum.get("num_records"),
        "force2d_wrapped": "forcing 2D single-pass" in log_txt,
        "surgattn_disabled": "VLLM_SURGATTN=0: force-2D DISABLED" in log_txt,
    }


def pair(a: str, b: str) -> dict:
    rep = greedy_gate.compare(OUTDIR / f"decode_{a}.jsonl", OUTDIR / f"decode_{b}.jsonl")
    onset = greedy_gate.onset_summary(rep)
    return {
        "pair": f"{a}_vs_{b}", "verdict": str(rep.verdict),
        "num_identical": onset.get("num_identical"),
        "num_divergent": onset.get("num_divergent"),
        "total_divergent_tokens": getattr(rep, "total_divergent_tokens", None),
        "onset_min": onset.get("onset_min"), "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"), "onsets": onset.get("onsets"),
    }


def main() -> None:
    sessions = {}
    for tag, surg in NEW_SESSIONS.items():
        print(f"\n===== SESSION {tag} (VLLM_SURGATTN={surg}) =====", flush=True)
        sessions[tag] = serve_decode(tag, surg)
        print(f"  {sessions[tag]}", flush=True)

    arms = ["2d_a", "2d_b", "3d_a", "3d_b"]
    # map *_a to the existing e2e files
    for a, existing in (("2d_a", "2d"), ("3d_a", "3d")):
        src = OUTDIR / f"decode_{existing}.jsonl"
        dst = OUTDIR / f"decode_{a}.jsonl"
        if not dst.exists() and src.exists():
            dst.symlink_to(src.name)

    pairs = [pair(a, b) for a, b in itertools.combinations(arms, 2)]
    out = {"sessions": sessions, "pairs": pairs,
           "num_prompts": paths.NUM_PROMPTS, "output_len": paths.OUTPUT_LEN}
    (OUTDIR / "determinism_control.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n[control] pairwise greedy-compare (divergent / 128):", flush=True)
    for p in pairs:
        print(f"  {p['pair']:>14s}: {p['num_divergent']:>3d} divergent  "
              f"({p['total_divergent_tokens']} tok)  onsets={p['onsets']}", flush=True)
    print(f"\n[control] wrote {OUTDIR/'determinism_control.json'}", flush=True)


if __name__ == "__main__":
    main()
