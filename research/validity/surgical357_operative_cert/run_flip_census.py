"""PR #494 — Census 2: run the #461 logit-margin flip-attribution census fresh on
this pod, redirected to the surgical357 cert dir.

WHY A WRAPPER: ``deployed_flip_attribution.py`` hardcodes its OUT_DIR and writes
``deployed_flip_attribution_report.json`` there. That file is a MERGED artifact
from a prior PR (#461). We must not clobber it. This wrapper rebinds the module
global ``dfa.OUT_DIR`` to the cert dir BEFORE orchestration, so every per-arm
artifact (passed to each phase subprocess via an explicit ``--out`` computed from
OUT_DIR) and the final report land under
``research/validity/surgical357_operative_cert/flip_census/`` instead.

WHAT IT CERTIFIES: the census's ``attn_only`` arm pins the attention module global
``is_batch_invariant=True`` -- byte-identical to what the packaged
``surgical_attn_patch.py`` installs. So the census directly certifies the surgical
lever's operative identity: the residual M8-verify-vs-M1-AR flips under the surgical
pin are bf16-ULP knife-edge near-ties (every divergent margin < 0.5 nat, the
NEAR_TIE_LOGPROB_THRESH; the prior #461 run measured <= 0.25, top-2 gap == 0.125 ==
the min representable bf16 logit step), 0 semantic, and ``attn_only`` divergence ==
``all_pin`` (the 222 global-flag config) divergence -- i.e. the surgical lever is
operatively equivalent to the shipped 222, dropping only the identity-unnecessary
matmul tax.

This is the #461 locus methodology (128 prompts x ctx 224, M=8 verify width via
prompt_logprobs) -- distinct from, and complementary to, the served 128x512 decode
self-determinism census (cert_served_identity.py).

LOCAL ONLY. analysis_only=true, official_tps=0. No HF job, no submission.

    .venv/bin/python -m research.validity.surgical357_operative_cert.run_flip_census \
        --n-prompts 128 --wandb_name stark/surgical357-flip-census \
        --wandb_group surgical357-package
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DFA_DIR = ROOT / "research" / "validity" / "deployed_flip_attribution"
if str(DFA_DIR) not in sys.path:
    sys.path.insert(0, str(DFA_DIR))

import deployed_flip_attribution as dfa  # noqa: E402

CERT_OUT = ROOT / "research" / "validity" / "surgical357_operative_cert" / "flip_census"
CERT_OUT.mkdir(parents=True, exist_ok=True)
# Rebind the module global so orchestrate's report + per-arm --out paths target the
# cert dir, leaving the merged deployed_flip_attribution_report.json untouched.
dfa.OUT_DIR = CERT_OUT
print(f"[flip-census] OUT_DIR redirected -> {dfa.OUT_DIR}", flush=True)


if __name__ == "__main__":
    dfa.main()
