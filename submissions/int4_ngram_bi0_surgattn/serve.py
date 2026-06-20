#!/usr/bin/env python
"""OpenAI-compatible vLLM server: int4 W4A16 Gemma-4-E4B target + swappable drafter.

This is the bi0 (``int4_mtp_bi0_surgattn``) serving stack with ONE functional
change vs the shipped submission: the speculative proposer can be the ngram /
prompt-lookup drafter instead of the MTP draft model. Everything else — the
int4 W4A16 Marlin target, VLLM_BATCH_INVARIANT=0 fast kernels, and the surgical
force-2D verify-attention patch (sitecustomize + vllm_force2d_attn_patch) that
makes the M=K spec-verify forward byte-exact to the M=1 AR decode — is copied
verbatim from bi0, so the ONLY isolated variable is the drafter (PR #782).

Target: google/gemma-4-E4B-it-qat-w4a16-ct (official QAT W4A16 compressed-tensors,
loaded via Marlin).

Drafter is selected by ``SPECULATIVE_METHOD``:

* ``ngram`` (this submission's default): vLLM's prompt-lookup proposer. It drafts
  continuations by matching the current generation suffix (an n-gram of size in
  ``[PROMPT_LOOKUP_MIN, PROMPT_LOOKUP_MAX]``) against ``prompt + generated-so-far``
  and copying the continuation — zero model forward, zero GPU weight read. The
  cache is rebuilt per request from THAT request's own decoded text, so it is
  private-prompt-stable (it never hardcodes the public eval prompts).
* unset / anything else with ``DRAFTER_MODEL`` set: the MTP draft model
  (``gemma4_assistant`` -> Gemma4MTPModel), i.e. the exact bi0 control path.

At temperature=0 vLLM's rejection sampler short-circuits to target-argmax for
EITHER drafter, so decode stays greedy-safe by construction: the verifier always
emits the int4 target's argmax and rejects on the first draft!=target mismatch,
so the accepted token at each position is independent of which drafter proposed
it. The drafter only changes WHICH / HOW-MANY positions a verify forward covers.

Set NUM_SPECULATIVE_TOKENS=0 (or empty), or SENPAI_REFERENCE_MODE=1, to disable
speculation and serve the plain int4 M=1 target. That mode is the exact-greedy
reference for the greedy-identity gate; it is identical for the ngram and MTP
submissions (drafter off -> same plain int4 AR), so both share one anchor.

All modalities (text/image/audio) stay enabled: no --limit-mm-per-prompt, no
text-only shortcut. The MTP draft head is left in its native bf16/centroid path
(never force-quantized): the assistant's masked-embedding centroid logits have no
packed-weight branch, so quantizing it would force the ~11x-slower dense path.
The ngram proposer has no weights at all.
"""
from __future__ import annotations

import json
import os
import sys


# Reference-mode contract env var. Mirrors
# scripts/local_validation/paths.REFERENCE_MODE_ENV; hardcoded here because a
# submission's serve.py runs in its own venv and cannot import the harness.
REFERENCE_MODE_ENV = "SENPAI_REFERENCE_MODE"


def reference_mode_active() -> bool:
    """True when the harness asked for the M=1 AR greedy-reference contract.

    When SENPAI_REFERENCE_MODE is truthy, this speculative submission MUST serve
    plain M=1 autoregressive decode (drafter OFF) so the served capture is the
    canonical greedy reference the challenge gate compares against — generated on
    this submission's OWN engine/kernels/quant, so the only removed variable is
    speculation. ``gen_greedy_reference --spec-off`` sets it to "1"; unset/""/"0"
    leave speculation on, so the leaderboard serving path is untouched.
    """
    return os.environ.get(REFERENCE_MODE_ENV, "") not in ("", "0")


def reference_mode_num_spec(num_spec: int) -> int:
    """Force ``num_speculative_tokens=0`` under the reference-mode contract.

    Returns 0 (speculation off -> plain int4 M=1 AR, the exact-greedy reference
    this file documents) when SENPAI_REFERENCE_MODE is truthy, else ``num_spec``
    unchanged. With 0 returned, the ``--speculative-config`` block below is
    skipped, so vLLM starts with ``speculative_config=None``.
    """
    if not reference_mode_active():
        return num_spec
    if num_spec > 0:
        print(
            "[serve] SENPAI_REFERENCE_MODE active: forcing num_speculative_tokens=0 "
            "(M=1 AR greedy reference, drafter OFF)",
            flush=True,
        )
    return 0


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS", "512")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS", "1")
    # Optional online target quantization. Unset for the shipped int4 submission
    # (its checkpoint is already compressed-tensors W4A16). Set e.g.
    # QUANTIZATION=fp8 only for the local precision-localization diagnostic, to
    # dynamically quantize a bf16 target (Marlin fp8 on Ampere) without a
    # separate checkpoint.
    quantization = os.environ.get("QUANTIZATION") or None

    drafter_model = os.environ.get("DRAFTER_MODEL") or None
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "5") or "0")
    num_spec = reference_mode_num_spec(num_spec)
    spec_method = os.environ.get("SPECULATIVE_METHOD")  # "ngram" | None (-> MTP)

    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-num-seqs",
        max_num_seqs,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", max_num_batched_tokens]
    if quantization:
        args += ["--quantization", quantization]

    # Build the speculative-config. The ONLY variable vs bi0 is which proposer:
    #
    #  * ngram / prompt-lookup: NO draft model. vLLM resolves model="ngram" and
    #    matches a suffix n-gram of size in [prompt_lookup_min, prompt_lookup_max]
    #    against prompt+generated, copying up to num_speculative_tokens
    #    continuation tokens for the target to verify. prompt_lookup_max is
    #    required by vLLM's SpeculativeConfig for the ngram method; prompt_lookup_min
    #    defaults to prompt_lookup_max if unset.
    #  * MTP (bi0 control): the gemma4_assistant draft model. vLLM rewrites
    #    model_type gemma4_assistant -> gemma4_mtp (Gemma4MTPModel) and sets
    #    num_kv_shared_layers=0 so the draft attention layers form their own group
    #    and read the target KV cache via the proposer. Drafter stays unquantized.
    #
    # Either way the int4 TARGET and its surgical force-2D verify-attention are
    # untouched, so the M=K verify forward is byte-exact to M=1 AR for both.
    if num_spec > 0:
        if spec_method == "ngram":
            spec_config: dict[str, object] = {
                "method": "ngram",
                "num_speculative_tokens": num_spec,
                "prompt_lookup_max": int(os.environ.get("PROMPT_LOOKUP_MAX", "3")),
            }
            prompt_lookup_min = os.environ.get("PROMPT_LOOKUP_MIN")
            if prompt_lookup_min:
                spec_config["prompt_lookup_min"] = int(prompt_lookup_min)
            args += ["--speculative-config", json.dumps(spec_config)]
        elif drafter_model:
            spec_config = {
                "model": drafter_model,
                "num_speculative_tokens": num_spec,
            }
            if spec_method:
                spec_config["method"] = spec_method
            args += ["--speculative-config", json.dumps(spec_config)]

    if os.environ.get("ENFORCE_EAGER", "0") == "1":
        args += ["--enforce-eager"]

    # Ship the attention-group num_heads backport (see sitecustomize.py /
    # vllm_attn_group_patch.py) into every server process. We pin vllm==0.22.0
    # to match the official vllm/vllm-openai image, which predates the upstream
    # fix for the {8,4} draft/target attention-group assertion; putting this
    # directory on PYTHONPATH makes Python auto-import our sitecustomize.py at
    # startup in the api_server, EngineCore, and worker processes (works under
    # both fork and spawn). The patch is a no-op when speculation is disabled.
    here = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        here + os.pathsep + existing_pythonpath if existing_pythonpath else here
    )

    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
