#!/usr/bin/env python
"""OpenAI-compatible vLLM server: int4 W4A16 Gemma-4-E4B target + MTP drafter.

Target: google/gemma-4-E4B-it-qat-w4a16-ct (official QAT W4A16 compressed-tensors,
loaded via Marlin). Drafter: the QAT-matched gemma4_assistant
(google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant), a lightweight Q-only
KV-shared decoder that vLLM resolves to Gemma4MTPModel and serves as a
speculative proposer. At temperature=0 vLLM's rejection sampler short-circuits to
target-argmax, so decode stays token-identical to plain greedy AR of the int4
target while amortizing the int4 weight read over the accepted draft tokens.

Set NUM_SPECULATIVE_TOKENS=0 (or empty) to disable speculation and serve the
plain int4 target. That mode is the exact-greedy reference for the
greedy-identity gate: reference and candidate then differ only in the drafter.

All modalities (text/image/audio) stay enabled: no --limit-mm-per-prompt, no
text-only shortcut. The draft head is left in its native bf16/centroid path
(never force-quantized): the assistant's masked-embedding centroid logits have no
packed-weight branch, so quantizing it would force the ~11x-slower dense path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


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


def maybe_quant_lmhead_at_startup(model_id: str) -> str:
    """Token-free serve path: quantize the PUBLIC base's bf16 lm_head to int4 at
    startup instead of pulling a pre-built PRIVATE checkpoint.

    When ``LMHEAD_QUANT_AT_STARTUP=1``, resolve ``MODEL_ID`` (a public Hub base
    such as ``google/gemma-4-E4B-it-qat-w4a16-ct``, or a local snapshot dir) to a
    local dir, run the sibling ``build_lmhead_quant.py --num-bits <bits>
    --head-group-size <gs>`` against it, and return the freshly built checkpoint
    dir for vLLM to serve. The on-disk build is byte-identical to the validated
    int4head checkpoint (same deterministic builder, same source snapshot), so
    PPL / greedy / modality behaviour is unchanged; only the model SOURCE changes:
    a public base + an on-disk startup quant, with NO private-repo auth — which
    is exactly the runner-side 401 the pre-built private path hit.

    Unset / ``0`` -> return ``model_id`` unchanged (the default Hub-pointed path,
    byte-for-byte the shipped behaviour). Idempotent: a completed build is reused
    via a ``.startupq_done`` marker so a server restart never rebuilds.

    Env knobs (all optional): ``LMHEAD_QUANT_BITS`` (default 4),
    ``LMHEAD_QUANT_GROUP_SIZE`` (default 32), ``LMHEAD_QUANT_OUT`` (default
    ``/tmp/int4head_startupq``), ``LMHEAD_QUANT_BASE_REV`` (pin the base Hub
    revision for a deterministic, reproducible build).
    """
    if os.environ.get("LMHEAD_QUANT_AT_STARTUP", "0") != "1":
        return model_id

    bits = os.environ.get("LMHEAD_QUANT_BITS", "4")
    group_size = os.environ.get("LMHEAD_QUANT_GROUP_SIZE", "32")
    out_dir = os.environ.get("LMHEAD_QUANT_OUT", "/tmp/int4head_startupq")
    base_rev = os.environ.get("LMHEAD_QUANT_BASE_REV") or None
    here = os.path.dirname(os.path.abspath(__file__))
    builder = os.path.join(here, "build_lmhead_quant.py")

    done_marker = os.path.join(out_dir, ".startupq_done")
    if os.path.exists(done_marker) and os.path.exists(os.path.join(out_dir, "model.safetensors")):
        print(f"[serve] startup lm_head int4: reusing existing build at {out_dir}", flush=True)
        return out_dir

    # Resolve MODEL_ID to a local checkpoint dir: an existing dir is used as-is;
    # a Hub id is materialized with snapshot_download (no token needed for the
    # public base — the whole point of this path).
    if os.path.isdir(model_id):
        src = model_id
    else:
        from huggingface_hub import snapshot_download

        print(
            f"[serve] startup lm_head int4: resolving base {model_id}"
            f"{(' @ ' + base_rev) if base_rev else ''} ...",
            flush=True,
        )
        src = snapshot_download(model_id, revision=base_rev)

    t0 = time.time()
    print(
        f"[serve] startup lm_head int4 quant: src={src} out={out_dir} "
        f"num_bits={bits} head_group_size={group_size}",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            builder,
            "--src", src,
            "--out", out_dir,
            "--num-bits", str(bits),
            "--head-group-size", str(group_size),
        ],
        check=True,
    )
    Path(done_marker).write_text(
        f"num_bits={bits} head_group_size={group_size} src={src} rev={base_rev}\n"
    )
    print(
        f"[serve] startup lm_head int4 quant complete in {time.time() - t0:.1f}s "
        f"-> serving {out_dir}",
        flush=True,
    )
    return out_dir


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
    model_id = maybe_quant_lmhead_at_startup(model_id)
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

    drafter_model = os.environ.get(
        "DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
    )
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6") or "0")
    num_spec = reference_mode_num_spec(num_spec)
    spec_method = os.environ.get("SPECULATIVE_METHOD")  # optional override

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

    # Speculative decoding with the gemma4_assistant MTP drafter. vLLM's
    # speculative config rewrites model_type gemma4_assistant -> gemma4_mtp
    # (Gemma4MTPModel) and sets num_kv_shared_layers=0 so the draft attention
    # layers form their own group and read the target KV cache via the proposer
    # after construction. Leave the drafter unquantized (no "quantization" key).
    if num_spec > 0 and drafter_model:
        spec_config: dict[str, object] = {
            "model": drafter_model,
            "num_speculative_tokens": num_spec,
        }
        if spec_method:
            spec_config["method"] = spec_method
        # PROFILING-ONLY acceptance-oracle knobs (PR #813). Default OFF: when none
        # of these env vars is set the spec_config is byte-identical to the shipped
        # submission, so the leaderboard serving path is untouched. vLLM's
        # rejection_sample_method='synthetic' IMPOSES a chosen accept rate in the
        # greedy branch (emits garbage tokens) so TPS-vs-accept-rate is a faithful
        # speed ceiling for "what if E_accept were higher". NEVER ship synthetic.
        rejection_method = os.environ.get("REJECTION_SAMPLE_METHOD")
        if rejection_method:
            spec_config["rejection_sample_method"] = rejection_method
        synth_rates = os.environ.get("SYNTHETIC_ACCEPTANCE_RATES")
        if synth_rates:
            spec_config["synthetic_acceptance_rates"] = json.loads(synth_rates)
        synth_len = os.environ.get("SYNTHETIC_ACCEPTANCE_LENGTH")
        if synth_len:
            spec_config["synthetic_acceptance_length"] = float(synth_len)
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
