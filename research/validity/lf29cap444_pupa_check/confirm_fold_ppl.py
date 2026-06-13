"""Confirmatory probe: force the LF29 affine fold ON for *every* request and
re-score both PPL paths.

Why this exists
---------------
The same-path PPL gate (PR #21) reported gap=0.0 / SAME_PATH_OK on
``pupa-lf29cap444`` — i.e. the echo+logprobs probe and the prompt_logprobs probe
returned *bit-identical* PPL (2.37794). That is NOT because the submission is
honest. vLLM's completion protocol sets ``SamplingParams.prompt_logprobs =
logprobs`` for any ``echo=True`` request (``completion/protocol.py:276-277``), so
the gate's own probe populates the model runner's ``num_prompt_logprobs`` — the
exact attribute the LF29 bypass keys on (``serve.py:411-415``). Both probes are
therefore routed to the *exact* dense FFN; the cheap affine fold (which the timed
decode path actually runs) is never exercised by the gate.

This script disables the exemption (``LFFN_PPL_EXACT=0``) so the fold runs for
ALL requests, then scores both paths. Expectation: both jump to ~2.55 (the fold's
true PPL, matching frantic-penguin's independent 2.5499), proving the 2.37794 the
gate saw was purely the exact-FFN exemption.

Run from the repo root:
    VLLM_USE_FLASHINFER_SAMPLER=0 python research/validity/lf29cap444_pupa_check/confirm_fold_ppl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, ppl_runner, same_path_ppl

SUBMISSION = ROOT / "submissions" / "lf29cap444_pupa_check"
OUT_DIR = ROOT / "research" / "validity" / "lf29cap444_pupa_check" / "fold_forced"
PORT = 8000


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[confirm-fold] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    # Headroom parity with the gate run, plus the one override that matters:
    # turn OFF the prompt_logprobs exact-FFN exemption so the affine fold runs
    # for every request shape (timed decode, echo, AND prompt_logprobs).
    overrides = ppl_runner._headroom_overrides(manifest.get("env", {}))
    overrides["LFFN_PPL_EXACT"] = "0"
    print(f"[confirm-fold] extra_env={overrides}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = paths.ppl_dataset()
    log = OUT_DIR / "server.log"

    with harness.LocalServer(
        SUBMISSION,
        server_python=server_python,
        port=PORT,
        log_path=log,
        extra_env=overrides,
    ) as srv:
        pl = ppl_runner.score_endpoint(
            srv.base_url, srv.served_model_name, out_dir=OUT_DIR, dataset=dataset
        )
        sp = same_path_ppl.score_endpoint(
            srv.base_url, srv.served_model_name, out_dir=OUT_DIR, dataset=dataset
        )

    pl_ppl = pl["ppl"]
    sp_ppl = sp["ppl"]
    print("\n================ FOLD-FORCED PPL (LFFN_PPL_EXACT=0) ================")
    print(f"prompt_logprobs PPL (fold): {pl_ppl:.4f}")
    print(f"echo same-path  PPL (fold): {sp_ppl:.4f}")
    print(f"vs gate run (exemption ON): 2.3779 on both -> gap 0.0 (false PASS)")
    print("===================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
