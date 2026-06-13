"""Single source of truth for official tool + dataset locations.

Everything the local harness needs lives in the read-only official mirror under
``official/main_bucket/shared_resources``. Keeping the paths here means the rest
of the package never hard-codes mirror layout, and the fixed benchmark protocol
constants (128 prompts, output_len 512, seed 1) live in exactly one place.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# scripts/local_validation/paths.py -> repo root is two parents up.
ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = ROOT / "official" / "main_bucket" / "shared_resources"
SPEED_BENCH = OFFICIAL / "speed_benchmark"
GREEDY_VERIFIER_DIR = OFFICIAL / "gemma_greedy_identity_verifier_flowian-powers"
PROFILER_DIR = OFFICIAL / "gemma_decode_profiler_claudecode"

DECODE_SCRIPT = SPEED_BENCH / "scripts" / "decode_outputs.py"
PPL_SCRIPT = SPEED_BENCH / "scripts" / "ppl_endpoint.py"
EVAL_PROMPTS = SPEED_BENCH / "data" / "eval_prompts_sharegpt.json"

PROFILE_GRAPH = PROFILER_DIR / "profile_graph.py"
PROFILE_EAGER = PROFILER_DIR / "profile_eager.py"

# Local artifacts (writable). References + run evidence land here.
REFERENCE_ROOT = ROOT / "research" / "greedy_reference"
LOCALRUN_ROOT = ROOT / "research" / "_localrun"

# Fixed benchmark protocol — must match hf_bucket_single_job.py exactly so a
# local capture lines up prompt-for-prompt with an official run.
TOKENIZER = "google/gemma-4-E4B-it"
NUM_PROMPTS = 128
OUTPUT_LEN = 512
SEED = 1
BF16_MODEL = "google/gemma-4-E4B-it"
INT4_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"
DEFAULT_SERVED_NAME = "gemma-4-e4b-it"

# Reference-mode contract. A submission's serve.py that implements any
# speculation/drafter path MUST, when this env var is truthy, serve plain M=1
# autoregressive decode with every drafter/speculative path disabled. That makes
# the served capture the canonical greedy reference the challenge gate compares
# against — generated on the submission's OWN engine + kernels + quantization, so
# the only removed variable is speculation. The harness sets this via
# ``gen_greedy_reference --mode served --submission <dir> --spec-off``.
REFERENCE_MODE_ENV = "SENPAI_REFERENCE_MODE"


def ppl_dataset() -> Path:
    """Path to the PPL ground-truth tokens.

    Prefers a unified top-level ``data/ppl_ground_truth_tokens.jsonl`` if it
    exists (so this runner shares one dataset with a sibling PPL-resolution PR
    rather than duplicating the file), else falls back to the official mirror.
    """
    top = ROOT / "data" / "ppl_ground_truth_tokens.jsonl"
    if top.exists():
        return top
    return SPEED_BENCH / "data" / "ppl_ground_truth_tokens.jsonl"


def import_greedy_identity():
    """Import the official ``greedy_identity`` comparison library by path.

    The verifier is stdlib-only and lives outside any package, so we add its
    directory to ``sys.path`` and import it directly rather than vendoring a
    copy that could drift from the official rule.
    """
    p = str(GREEDY_VERIFIER_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    import greedy_identity  # noqa: E402

    return greedy_identity


def model_tag(model_id: str) -> str:
    """Filesystem-safe tag for a model id, used to key reference artifacts."""
    return model_id.strip("/").replace("/", "__").replace(":", "_")


def normalize_cuda_visible_devices() -> str | None:
    """Point ``CUDA_VISIBLE_DEVICES`` at the in-container GPU index.

    Launch harnesses sometimes pin ``CUDA_VISIBLE_DEVICES`` to a *host* GPU
    index (e.g. ``"6"``) while the container only exposes the single assigned
    GPU at in-container NVML index 0. CUDA/torch then see zero usable devices and
    vLLM dies during model load with an opaque NVML ``Invalid Argument``. For
    this single-GPU local tooling we normalize to ``"0"`` whenever exactly one
    GPU is visible to NVML; genuine multi-GPU or CPU-only hosts are left alone so
    an operator's explicit pin is respected. Returns a note if it changed
    anything, else ``None``.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    count = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    if count != 1:
        return None
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current == "0":
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return (
        f"normalized CUDA_VISIBLE_DEVICES {current!r} -> '0' "
        "(single in-container GPU)"
    )


def default_native_sampler() -> str | None:
    """Default vLLM to the PyTorch-native sampler in this container.

    vLLM's default top-k/top-p backend is FlashInfer, which JIT-compiles a CUDA
    kernel at engine start. This container ships the cuRAND headers only inside
    the pip ``nvidia-cu13`` package (not in ``/usr/local/cuda/include``), so that
    build fails with ``curand.h: No such file or directory`` and the engine dies
    during memory profiling. The sampler backend does not touch model logits, so
    greedy-identity (argmax) and PPL (teacher-forced log-softmax) are unchanged;
    only the exploratory local TPS is marginally affected. We set it via
    ``setdefault`` so an operator or a submission manifest can override. Returns a
    note if it changed anything, else ``None``.
    """
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        return None
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return (
        "set VLLM_USE_FLASHINFER_SAMPLER=0 (PyTorch-native sampler; avoids cuRAND "
        "JIT in this container — does not affect greedy/PPL)"
    )


def prepare_local_gpu_env() -> list[str]:
    """Apply the local single-GPU container shims vLLM needs to start.

    Combines [[normalize_cuda_visible_devices]] and [[default_native_sampler]];
    each entrypoint calls this once before loading/serving vLLM so the offline
    reference and the served submission share the same environment. Returns the
    list of human-readable notes for whatever it changed (possibly empty).
    """
    notes = []
    for fn in (normalize_cuda_visible_devices, default_native_sampler):
        note = fn()
        if note:
            notes.append(note)
    return notes
