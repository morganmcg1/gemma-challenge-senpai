#!/usr/bin/env python
"""OpenAI-compatible vLLM server: int4 W4A16 Gemma-4-E4B target + MTP drafter.

NEGATIVE RESULT (PR #777). This submission was built to add an fp8 KV cache
(``--kv-cache-dtype``) on top of ``int4_mtp_bi0_surgattn`` to halve KV read
bandwidth in bandwidth-bound single-stream decode. **No fp8 KV dtype is
serviceable on A10G (Ampere, sm_86) for this int4-compressed-tensors +
force-2D-TRITON_ATTN + torch.compile stack.** The default is therefore ``auto``
(= the exact bi0 surgattn bf16 KV; boots and serves the 218.02-TPS baseline);
the fp8 dtypes below remain selectable only to *reproduce* the dead-end. This
re-confirms the prior fp8-KV dead-end (BASELINE.md "rejected by A10G + Gemma4
attn"; wirbel #141) with a precise three-layer root cause:

  1. vLLM init guard (``model_executor/layers/attention/attention.py``
     ``_init_kv_cache_quant``): ``CompressedTensorsConfig.get_quant_method``
     returns a ``CompressedTensorsKVCacheMethod`` for *every* Attention layer, so
     ``fp8_e5m2`` is hard-rejected with
     ``ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints`` --
     even though this W4A16 checkpoint declares no ``kv_cache_scheme``. RELAXED
     here by ``vllm_fp8kv_e5m2_guard_patch.py`` (only when kv_cache_scheme is
     None; a genuine fix for an over-broad guard), which gets e5m2 past layer 1.
  2. attention forward assert (same file, ~L467): when the fp8 KV path is active
     ``assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}`` -- so e5m2 is
     rejected by the kernel itself (it implements only e4m3/nvfp4 dequant). This
     is the "rejected by Gemma4 attn" wall; not patchable without rewriting the
     attention impl.
  3. Triton/sm_86 compile: ``fp8_e4m3`` (and bare ``fp8`` -> e4m3, and ``auto``
     when it would resolve to fp8) maps to Triton ``fp8e4nv``, which raises
     ``ValueError: type fp8e4nv not supported in this architecture. The supported
     fp8 dtypes are ('fp8e4b15', 'fp8e5')`` during inductor autotuning. This is
     the "rejected by A10G" wall.

The two Ampere-compilable Triton fp8 dtypes (``fp8e4b15``, ``fp8e5``) are exactly
the two the vLLM attention forward does NOT accept; the two it accepts
(``fp8_e4m3``/``nvfp4``) need Ada/Hopper/Blackwell. No fp8 dtype clears all three
layers. ``int8`` is not a valid vLLM ``CacheDType`` (only ``int8_per_token_head``,
a different unsupported path). Full evidence: research/validity/bi0_fp8kv/.

Set KV_CACHE_DTYPE to select the cache dtype (only ``auto`` boots on A10G):
  * ``auto`` (default) -- omit ``--kv-cache-dtype``; serves the default bf16 KV,
    i.e. the byte-for-byte ``int4_mtp_bi0_surgattn`` baseline. The ONLY runnable
    config on this hardware.
  * ``fp8_e5m2`` -- gets past layer 1 (guard patch) but dies at layer 2 (kernel
    assert). Kept to reproduce the wall.
  * ``fp8_e4m3`` / ``fp8`` -- pass layers 1-2 but die at layer 3 (Triton
    ``fp8e4nv`` on sm_86). Kept to reproduce the wall.

Everything else is byte-for-byte the bi0 surgattn config: VLLM_BATCH_INVARIANT=0,
the surgical force-2D TRITON_ATTN patch, the gemma4_assistant MTP drafter at
NUM_SPECULATIVE_TOKENS=6, standard int4 W4A16 weights, all modalities enabled.

Set NUM_SPECULATIVE_TOKENS=0 (or empty) to disable speculation and serve the
plain int4 target. That mode is the exact-greedy reference for the
greedy-identity gate: reference and candidate then differ only in the drafter
(the fp8 KV cache is held fixed in both, so the gate isolates speculation).

All modalities (text/image/audio) stay enabled: no --limit-mm-per-prompt, no
text-only shortcut. The draft head is left in its native bf16/centroid path
(never force-quantized): the assistant's masked-embedding centroid logits have no
packed-weight branch, so quantizing it would force the ~11x-slower dense path.
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
    canonical greedy reference the challenge gate compares against -- generated on
    this submission's OWN engine/kernels/quant (INCLUDING the fp8 KV cache, which
    stays on so the gate isolates speculation, not the cache dtype). ``gen_greedy_reference
    --spec-off`` sets it to "1"; unset/""/"0" leave speculation on, so the
    leaderboard serving path is untouched.
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

    # KV cache dtype. Default ``auto`` => omit ``--kv-cache-dtype`` => default
    # bf16 KV == the exact int4_mtp_bi0_surgattn baseline (the only config that
    # boots on A10G; see the module docstring's three-layer fp8 dead-end). The
    # fp8 dtypes are selectable only to reproduce the wall.
    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE", "auto")

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
    # fp8 (or any non-default) KV cache. ``auto``/empty -> omit the flag entirely
    # so vLLM keeps its default bf16 KV cache (matches int4_mtp_bi0_surgattn).
    if kv_cache_dtype and kv_cache_dtype != "auto":
        args += ["--kv-cache-dtype", kv_cache_dtype]
        print(f"[serve] KV cache dtype: {kv_cache_dtype} (fp8 paged KV)", flush=True)

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
