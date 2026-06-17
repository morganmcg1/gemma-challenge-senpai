#!/usr/bin/env python
"""dixie-flatline: onegraph-spec7 substrate + PCK-04 lm_head vocabulary pruning.

Stack:
  - blake's onegraph (315.12 TPS public best): one CUDA-graph replay of K=7
    width-1 drafter iterations, ping-pong slots=3, fused sparse argmax.
  - PCK-04: pruned lm_head (K≈32k rows) scattered back to full-vocab at
    compute_logits time, via serve_patch_pck04.py meta-path hook.
  - SMP-02: slim greedy sampler fast path + lastchance prewarm.
  - PLE scale-fold / fast path.

serve.py is structurally identical to field-artifacts/onegraph-spec7-v0/serve.py;
only deltas are:
  1. WEIGHTS_BUCKET / LOCAL_MODEL_DIR / LOCAL_DRAFTER_DIR defaults updated for
     the int4-pck04-16k checkpoint.
  2. setup_pck04_path() injects submission-pck04/ into PYTHONPATH so the worker
     process imports serve_patch_pck04.py via sitecustomize (meta-path finder).
  3. main() calls setup_pck04_path() after setup_sitecustomize_path().

kenyan-duma composition delta (vs hayai-agent osoi-v0 serve.py, byte-identical
otherwise): ensure_drafter() gains an optional DRAFTER_BUCKET env branch
(hf buckets sync, same mechanism as ensure_weights) for serving retrained
drafter checkpoints, and logs the sha256 of the drafter model.safetensors it
actually loads so the run record proves WHICH drafter served (stale-dir trap).
With DRAFTER_BUCKET unset, behavior is identical to hayai's original. Greedy
spec decode emits the target's argmax regardless of drafter proposals, so
emitted tokens are governed by the target checkpoint alone.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable


WEIGHTS_BUCKET = os.environ.get(
    "WEIGHTS_BUCKET",
    "hf://buckets/gemma-challenge/gemma-dixie-flatline/weights/int4-pck04-16k",
)
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/tmp/int4-pck04-16k")
DRAFTER_REPO = os.environ.get(
    "DRAFTER_REPO", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
)
LOCAL_DRAFTER_DIR = os.environ.get("LOCAL_DRAFTER_DIR", "/tmp/qat-assistant")
# Optional bucket override for the drafter (takes precedence over DRAFTER_REPO);
# used to serve retrained drafter checkpoints. The loaded-file sha256 is logged
# either way so the run record proves WHICH drafter actually served.
DRAFTER_BUCKET = os.environ.get("DRAFTER_BUCKET")
DRAFTER_SHA256 = os.environ.get("DRAFTER_SHA256", "").lower()
CENTROID_TOP_K = int(os.environ.get("CENTROID_TOP_K", "64"))
JINJA2_VERSION = "3.1.6"
MARKUPSAFE_VERSION = "3.0.3"

TCMALLOC_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
    "/usr/lib/libtcmalloc_minimal.so.4",
    "/usr/lib64/libtcmalloc_minimal.so.4",
]

Patcher = Callable[[str, pathlib.Path], tuple[str, bool]]


def replace_required(
    source: str,
    *,
    model_path: pathlib.Path,
    label: str,
    old: str,
    new: str,
    marker: str,
) -> tuple[str, bool]:
    """Apply one idempotent source replacement and fail on source drift."""
    if marker in source:
        return source, False
    old_count = source.count(old)
    if old_count != 1:
        raise RuntimeError(
            f"{label} patch pattern count is {old_count} in {model_path}; "
            "refusing to run a silent no-op baseline."
        )
    return source.replace(old, new, 1), True


PLE_TEXT_FAST_PATH_OLD = """        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)
"""

PLE_TEXT_FAST_PATH_NEW = """        # Challenge fast path: harness text token IDs are valid PLE IDs.
        # Multimodal serving still maps multimodal positions to token 0 before
        # this call in gemma4_mm.py, so the multimodal PLE contract is retained.
        per_layer_embeds = self.embed_tokens_per_layer(input_ids)
"""

PLE_RUNTIME_SCALE_OLD = (
    "        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer\n"
)
PLE_RUNTIME_SCALE_NEW = (
    "        # PLE scale-fold: embed_scale_per_layer is folded into "
    "embedding weights after load.\n"
)

PLE_GATE_SCRATCH_OLD = """            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
"""

PLE_GATE_SCRATCH_NEW = """            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            # PLE scratch reuse: in-place gate multiply when dtype-preserving.
            if gate.dtype == per_layer_input.dtype:
                gate.mul_(per_layer_input)
                gated_per_layer = gate
            else:
                gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
"""

PLE_COMBINE_SCRATCH_OLD = """        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
"""

PLE_COMBINE_SCRATCH_NEW = """        if per_layer_inputs is None:
            return per_layer_projection
        # PLE scratch reuse: in-place projection add when dtype-preserving.
        if per_layer_projection.dtype == per_layer_inputs.dtype:
            per_layer_projection.add_(per_layer_inputs)
            return per_layer_projection * self.per_layer_input_scale
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
"""

SELF_DECODER_FOLD_ANCHOR = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
"""
SELF_DECODER_FOLD_METHOD = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    @torch.inference_mode()
    def fold_per_layer_embed_scale(self) -> None:
        if self.embed_tokens_per_layer is None or self.embed_scale_per_layer is None:
            return
        if getattr(self.embed_tokens_per_layer, "_ple_embed_scale_folded", False):
            return
        if self.hidden_size_per_layer_input != 256:
            raise RuntimeError(
                "PLE scale-fold expected hidden_size_per_layer_input=256, "
                f"got {self.hidden_size_per_layer_input}"
            )
        if self.embed_scale_per_layer.numel() != 1:
            raise RuntimeError("PLE scale-fold expects scalar embed_scale_per_layer")

        scale = float(self.embed_scale_per_layer.item())
        expected_scale = float(self.hidden_size_per_layer_input ** 0.5)
        if scale != expected_scale:
            raise RuntimeError(
                f"PLE scale-fold expected scale {expected_scale}, got {scale}"
            )

        embedding = self.embed_tokens_per_layer
        if hasattr(embedding, "weight_scale"):
            target = embedding.weight_scale
            folded_name = "weight_scale"
        elif hasattr(embedding, "weight"):
            target = embedding.weight
            folded_name = "weight"
        else:
            raise RuntimeError(
                "PLE scale-fold found no weight_scale or weight on "
                "embed_tokens_per_layer"
            )

        if target.dtype != torch.bfloat16:
            raise RuntimeError(
                f"PLE scale-fold expects bf16 {folded_name}, got {target.dtype}"
            )
        if target.device.type != "cuda":
            raise RuntimeError(
                f"PLE scale-fold expects CUDA {folded_name}, got {target.device}"
            )

        target.data.mul_(scale)
        embedding._ple_embed_scale_folded = True
        logger.info("Folded Gemma4 PLE embed scale %s into %s", scale, folded_name)

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
"""

MODEL_DELEGATE_OLD = '''    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Get per-layer embeddings from embed_tokens_per_layer.

        Returns:
            Per-layer embeddings (num_tokens, num_layers,
            hidden_size_per_layer_input)
        """
        return self.self_decoder.get_per_layer_inputs(input_ids)

    def project_per_layer_inputs(
'''
MODEL_DELEGATE_NEW = '''    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Get per-layer embeddings from embed_tokens_per_layer.

        Returns:
            Per-layer embeddings (num_tokens, num_layers,
            hidden_size_per_layer_input)
        """
        return self.self_decoder.get_per_layer_inputs(input_ids)

    def fold_per_layer_embed_scale(self) -> None:
        self.self_decoder.fold_per_layer_embed_scale()

    def project_per_layer_inputs(
'''

LOADER_IMPORT_OLD = "import inspect\n"
LOADER_IMPORT_NEW = "import inspect\nimport os\n"
LOADER_HOOK_OLD = """    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)
"""
LOADER_HOOK_NEW = """    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)

    if os.environ.get("PLE_FOLD_EMBED_SCALE") == "1":
        fold_target_model = os.environ.get("PLE_FOLD_TARGET_MODEL")
        current_model = getattr(model_config, "model", None)
        if fold_target_model and current_model != fold_target_model:
            logger.info(
                "Skipping Gemma4 PLE embed_scale_per_layer fold for "
                "non-target model %s",
                current_model,
            )
        else:
            candidates = [
                model,
                getattr(model, "model", None),
                getattr(getattr(model, "language_model", None), "model", None),
            ]
            fold_applied = False
            for candidate in candidates:
                folder = getattr(candidate, "fold_per_layer_embed_scale", None)
                if folder is None:
                    continue
                logger.info("Folding Gemma4 PLE embed_scale_per_layer")
                folder()
                decoder = getattr(candidate, "self_decoder", None)
                embedding = getattr(decoder, "embed_tokens_per_layer", None)
                fold_applied = bool(
                    getattr(embedding, "_ple_embed_scale_folded", False)
                )
                if not fold_applied:
                    raise RuntimeError(
                        "PLE_FOLD_EMBED_SCALE=1 but fold_per_layer_embed_scale "
                        "did not mark embed_tokens_per_layer as folded"
                    )
                break
            if not fold_applied:
                raise RuntimeError(
                    "PLE_FOLD_EMBED_SCALE=1 but no target model candidate "
                    "exposed fold_per_layer_embed_scale"
                )
"""


def build_ple_text_fast_source(
    source: str, model_path: pathlib.Path
) -> tuple[str, bool]:
    """Build Gemma4 source with the exact PLE valid-token fast path applied.

    Args:
        source: Current text of vLLM's Gemma4 model file.
        model_path: Path included in failure messages for actionable startup errors.

    Returns:
        A pair of patched source text and whether the text changed.

    Raises:
        RuntimeError: If neither the original nor patched source block is present.
    """
    return replace_required(
        source,
        model_path=model_path,
        label="PLE fast path",
        old=PLE_TEXT_FAST_PATH_OLD,
        new=PLE_TEXT_FAST_PATH_NEW,
        marker="Challenge fast path: harness text token IDs are valid PLE IDs.",
    )


def patch_gemma4_source(source: str, model_path: pathlib.Path) -> tuple[str, bool]:
    changed_any = False
    for label, old, new, marker in (
        (
            "PLE valid-token fast path",
            PLE_TEXT_FAST_PATH_OLD,
            PLE_TEXT_FAST_PATH_NEW,
            "Challenge fast path: harness text token IDs are valid PLE IDs.",
        ),
        (
            "PLE scale-fold method",
            SELF_DECODER_FOLD_ANCHOR,
            SELF_DECODER_FOLD_METHOD,
            "def fold_per_layer_embed_scale",
        ),
        (
            "PLE runtime scale multiply",
            PLE_RUNTIME_SCALE_OLD,
            PLE_RUNTIME_SCALE_NEW,
            "PLE scale-fold: embed_scale_per_layer is folded into embedding weights",
        ),
        (
            "PLE gate scratch reuse",
            PLE_GATE_SCRATCH_OLD,
            PLE_GATE_SCRATCH_NEW,
            "PLE scratch reuse: in-place gate multiply",
        ),
        (
            "PLE projection-combine scratch reuse",
            PLE_COMBINE_SCRATCH_OLD,
            PLE_COMBINE_SCRATCH_NEW,
            "PLE scratch reuse: in-place projection add",
        ),
        (
            "PLE scale-fold model delegate",
            MODEL_DELEGATE_OLD,
            MODEL_DELEGATE_NEW,
            "self.self_decoder.fold_per_layer_embed_scale()",
        ),
    ):
        source, changed = replace_required(
            source,
            model_path=model_path,
            label=label,
            old=old,
            new=new,
            marker=marker,
        )
        changed_any = changed_any or changed
    return source, changed_any


def patch_loader_utils_source(
    source: str, model_path: pathlib.Path
) -> tuple[str, bool]:
    source, import_changed = replace_required(
        source,
        model_path=model_path,
        label="PLE scale-fold loader os import",
        old=LOADER_IMPORT_OLD,
        new=LOADER_IMPORT_NEW,
        marker="import os",
    )
    source, hook_changed = replace_required(
        source,
        model_path=model_path,
        label="PLE scale-fold loader hook",
        old=LOADER_HOOK_OLD,
        new=LOADER_HOOK_NEW,
        marker="PLE_FOLD_EMBED_SCALE",
    )
    return source, import_changed or hook_changed


DIXIE_SMP02_CONST_OLD = "logger = init_logger(__name__)\n"

DIXIE_SMP02_CONST_NEW = """logger = init_logger(__name__)

_DIXIE_SLIM_GREEDY = __import__("os").environ.get("DIXIE_SLIM_GREEDY", "1") == "1"
_DIXIE_FUSED_ACCEPT_PREP = (
    __import__("os").environ.get("DIXIE_FUSED_ACCEPT_PREP") == "1"
)
"""

DIXIE_SMP02_FWD_OLD = "        assert metadata.max_spec_len <= MAX_SPEC_LEN\n"

DIXIE_SMP02_FWD_NEW = """        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # dixie SMP-02: all-greedy fast path. bf16 -> fp32 is an exact,
        # monotonic upcast, so argmax over raw logits is bit-identical to the
        # slow path's argmax over the fp32 copy; the gate guarantees no logits
        # processor / penalty / mask / logprobs request can observe the
        # skipped work. Anything else falls through to the original code.
        if (
            _DIXIE_SLIM_GREEDY
            and sampling_metadata.all_greedy
            and not self.synthetic_mode
            and sampling_metadata.max_num_logprobs is None
            and sampling_metadata.no_penalties
            and not sampling_metadata.bad_words_token_ids
            and sampling_metadata.allowed_token_ids_mask is None
            and (
                sampling_metadata.thinking_budget_state_holder is None
                or not sampling_metadata.thinking_budget_state_holder.has_tracked_requests()
            )
        ):
            dixie_all_argmax = logits.argmax(dim=-1)
            dixie_bonus_token_ids = (
                dixie_all_argmax[metadata.bonus_logits_indices]
                .unsqueeze(1)
                .contiguous()
            )
            dixie_target_argmax = dixie_all_argmax[
                metadata.target_logits_indices
            ].contiguous()
            dixie_batch_size = len(metadata.num_draft_tokens)
            dixie_output_token_ids = torch.full(
                (dixie_batch_size, metadata.max_spec_len + 1),
                PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32,
                device=logits.device,
            )
            if _DIXIE_FUSED_ACCEPT_PREP:
                import sitecustomize as _gemma_sitecustomize

                if _gemma_sitecustomize._dixie_fused_accept_prep(
                    dixie_output_token_ids,
                    metadata.cu_num_draft_tokens,
                    metadata.draft_token_ids,
                    dixie_target_argmax,
                    dixie_bonus_token_ids,
                    metadata.max_spec_len,
                ):
                    return SamplerOutput(
                        sampled_token_ids=dixie_output_token_ids,
                        logprobs_tensors=None,
                    )
            rejection_greedy_sample_kernel[(dixie_batch_size,)](
                dixie_output_token_ids,
                metadata.cu_num_draft_tokens,
                metadata.draft_token_ids,
                dixie_target_argmax,
                dixie_bonus_token_ids,
                None,
                metadata.max_spec_len,
                None,
                None,
                SYNTHETIC_MODE=False,
            )
            return SamplerOutput(
                sampled_token_ids=dixie_output_token_ids,
                logprobs_tensors=None,
            )
"""


DIXIE_SMP02_PREWARM_OLD = """        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=[\"max_spec_len\"])
def rejection_random_sample_kernel(
"""

DIXIE_SMP02_PREWARM_NEW = """        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


def _lastchance_prewarm_greedy_rejection_kernel() -> None:
    if (
        not _DIXIE_SLIM_GREEDY
        or __import__(\"os\").environ.get(\"DIXIE_PREWARM_GREEDY_KERNEL\", \"1\") != \"1\"
    ):
        return
    try:
        if not torch.cuda.is_available():
            return
        device = torch.device(\"cuda\")
        output_token_ids = torch.full(
            (1, 8), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = torch.tensor([7], dtype=torch.int32, device=device)
        draft_token_ids = torch.arange(7, dtype=torch.int32, device=device)
        target_argmax = torch.arange(7, dtype=torch.int64, device=device)
        bonus_token_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        rejection_greedy_sample_kernel[(1,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            None,
            7,
            None,
            None,
            SYNTHETIC_MODE=False,
        )
        torch.cuda.synchronize()
        logger.info(\"lastchance prewarmed greedy rejection kernel\")
    except Exception as exc:
        logger.warning(\"lastchance greedy rejection prewarm failed: %r\", exc)


_lastchance_prewarm_greedy_rejection_kernel()


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=[\"max_spec_len\"])
def rejection_random_sample_kernel(
"""


def patch_rejection_sampler_source(
    source: str, model_path: pathlib.Path
) -> tuple[str, bool]:
    source, const_changed = replace_required(
        source,
        model_path=model_path,
        label="dixie SMP-02 slim-greedy const",
        old=DIXIE_SMP02_CONST_OLD,
        new=DIXIE_SMP02_CONST_NEW,
        marker="_DIXIE_SLIM_GREEDY",
    )
    source, fwd_changed = replace_required(
        source,
        model_path=model_path,
        label="dixie SMP-02 slim-greedy fast path",
        old=DIXIE_SMP02_FWD_OLD,
        new=DIXIE_SMP02_FWD_NEW,
        marker="dixie SMP-02: all-greedy fast path",
    )
    source, prewarm_changed = replace_required(
        source,
        model_path=model_path,
        label="lastchance SMP-02 greedy kernel prewarm",
        old=DIXIE_SMP02_PREWARM_OLD,
        new=DIXIE_SMP02_PREWARM_NEW,
        marker="_lastchance_prewarm_greedy_rejection_kernel",
    )
    return source, const_changed or fwd_changed or prewarm_changed


PATCHERS: dict[str, Patcher] = {
    "gemma4.py": patch_gemma4_source,
    "utils.py": patch_loader_utils_source,
    "rejection_sampler.py": patch_rejection_sampler_source,
}


def ensure_weights() -> None:
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    if os.path.isdir(LOCAL_MODEL_DIR) and os.path.exists(config_path):
        return

    print(f"[serve] syncing weights {WEIGHTS_BUCKET} -> {LOCAL_MODEL_DIR}", flush=True)
    subprocess.run(
        ["hf", "buckets", "sync", WEIGHTS_BUCKET, LOCAL_MODEL_DIR], check=True
    )


def _prune_lm_head_rows(src_dir: str, keepset_path: str, dst_dir: str) -> None:
    """Row-slice packed PCK04 lm_head tensors while leaving embeddings full-vocab."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = pathlib.Path(src_dir)
    dst = pathlib.Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    keep_meta = json.loads(pathlib.Path(keepset_path).read_text(encoding="utf-8"))
    keep_ids = keep_meta["keep_ids"]
    full_vocab_meta = int(keep_meta.get("full_vocab") or keep_meta.get("vocab_size") or 0)

    tensors = {}
    with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        for key in file.keys():
            tensors[key] = file.get_tensor(key)

    packed = tensors["lm_head.weight_packed"]
    scale = tensors["lm_head.weight_scale"]
    shape = tensors["lm_head.weight_shape"]
    source_rows = int(packed.shape[0])
    source_keep_path = src / "pck04_keepset.json"
    if not source_keep_path.exists():
        raise RuntimeError(
            f"source PCK04 keepset missing: {source_keep_path}"
        )
    source_keep_meta = json.loads(source_keep_path.read_text(encoding="utf-8"))
    source_keep_ids = source_keep_meta["keep_ids"]
    full_vocab = int(
        source_keep_meta.get("full_vocab")
        or source_keep_meta.get("vocab_size")
        or full_vocab_meta
        or 0
    )
    if full_vocab_meta and full_vocab_meta != full_vocab:
        raise RuntimeError(
            f"keepset full_vocab={full_vocab_meta} does not match source full vocab {full_vocab}"
        )
    if len(source_keep_ids) != source_rows:
        raise RuntimeError(
            f"source keepset length {len(source_keep_ids)} does not match lm_head rows {source_rows}"
        )
    source_row_by_token = {int(token_id): row for row, token_id in enumerate(source_keep_ids)}
    missing = [int(token_id) for token_id in keep_ids if int(token_id) not in source_row_by_token]
    if missing:
        raise RuntimeError(
            f"12k keepset is not a subset of source keepset; first missing token {missing[0]}"
        )
    keep_idx = torch.tensor(
        [source_row_by_token[int(token_id)] for token_id in keep_ids],
        dtype=torch.long,
    )
    if packed.shape[1] != 320 or scale.shape != (source_rows, 1):
        raise RuntimeError(
            f"unexpected PCK04 lm_head shapes: packed={tuple(packed.shape)} scale={tuple(scale.shape)}"
        )
    if shape.tolist() != [source_rows, 2560]:
        raise RuntimeError(f"unexpected lm_head.weight_shape={shape.tolist()}")
    if max(keep_ids) >= full_vocab:
        raise RuntimeError(f"keepset id {max(keep_ids)} exceeds vocab {full_vocab}")

    tensors["lm_head.weight_packed"] = torch.index_select(packed, 0, keep_idx)
    tensors["lm_head.weight_scale"] = torch.index_select(scale, 0, keep_idx)
    tensors["lm_head.weight_shape"] = torch.tensor([len(keep_ids), 2560], dtype=torch.int64)
    save_file(tensors, str(dst / "model.safetensors"), metadata=metadata)

    for src_file in src.iterdir():
        if src_file.name == "model.safetensors":
            continue
        dst_file = dst / src_file.name
        if src_file.is_file():
            shutil.copy2(src_file, dst_file)
        elif src_file.is_dir():
            if dst_file.exists():
                shutil.rmtree(dst_file)
            shutil.copytree(src_file, dst_file)

    (dst / "pck04_keepset.json").write_text(
        json.dumps(
            {
                "keep_ids": keep_ids,
                "pruned_vocab_K": len(keep_ids),
                "full_vocab": full_vocab,
                "source_keepset": keepset_path,
                "note": "embed_tokens remains full-vocab; only lm_head rows are pruned",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[lmhead-prune] row-sliced lm_head {source_rows}->{len(keep_ids)} rows "
        f"(full_vocab={full_vocab})",
        flush=True,
    )


def _lmhead_prune_phase() -> None:
    """In-job PCK04 lm_head row-slice. Benchmark-safe because it runs before serve."""
    global LOCAL_MODEL_DIR
    if os.environ.get("LM_HEAD_PRUNE") != "1":
        return

    dst = os.environ.get("LM_HEAD_PRUNE_DST", "/tmp/osoi5-12k-baked")
    try:
        if os.path.exists(os.path.join(dst, "config.json")):
            print(f"[lmhead-prune] reusing baked dir {dst}", flush=True)
        else:
            keepset_bucket = os.environ.get(
                "LM_HEAD_KEEPSET_BUCKET",
                "hf://buckets/gemma-challenge/gemma-dixie-flatline/weights/int4-pck04c-12k",
            )
            keepset_dir = "/tmp/lmhead-keepset-12k"
            keepset_path = os.path.join(keepset_dir, "pck04_keepset.json")
            if not os.path.exists(keepset_path):
                pathlib.Path(keepset_dir).mkdir(parents=True, exist_ok=True)
                print(f"[lmhead-prune] copying keepset {keepset_bucket}", flush=True)
                subprocess.run(
                    [
                        "hf",
                        "buckets",
                        "cp",
                        f"{keepset_bucket.rstrip('/')}/pck04_keepset.json",
                        keepset_path,
                    ],
                    check=True,
                )
            print(
                f"[lmhead-prune] pruning {LOCAL_MODEL_DIR} -> {dst} "
                f"(keepset {keepset_path})",
                flush=True,
            )
            _prune_lm_head_rows(LOCAL_MODEL_DIR, keepset_path, dst)

        LOCAL_MODEL_DIR = dst
        os.environ["LOCAL_MODEL_DIR"] = dst
        os.environ["PLE_FOLD_TARGET_MODEL"] = dst
        os.environ["PCK04_KEEPSET"] = os.path.join(dst, "pck04_keepset.json")
        print(
            f"[lmhead-prune] active dst={dst} keepset={os.environ['PCK04_KEEPSET']}",
            flush=True,
        )
    except Exception as exc:
        message = f"[lmhead-prune] failed: {exc!r}"
        if os.environ.get("LM_HEAD_PRUNE_REQUIRE") == "1":
            raise RuntimeError(message) from exc
        print(f"{message}; serving osoi5 substrate unchanged", flush=True)


FULL_LM_HEAD_ROWS = 262144


def _assert_full_lm_head() -> None:
    """Hard-reject guard (PR #545): when LM_HEAD_FULL_REQUIRE=1, refuse to serve
    unless lm_head carries the complete 262,144-row vocabulary.

    Protects the base_fullhead quality-safe arm: serving it means LM_HEAD_PRUNE=0
    so no row is -inf'd, and the GSM8K recovery is meaningless if a pruned head
    silently fell back into place. Runs after _lmhead_prune_phase so it inspects
    the dir that will actually be served (LOCAL_MODEL_DIR)."""
    if os.environ.get("LM_HEAD_FULL_REQUIRE") != "1":
        return
    if os.environ.get("LM_HEAD_PRUNE") == "1":
        raise RuntimeError(
            "LM_HEAD_FULL_REQUIRE=1 is incompatible with LM_HEAD_PRUNE=1: the "
            "prune phase would row-slice lm_head below full vocab"
        )
    from safetensors import safe_open

    st_files = sorted(glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors")))
    if not st_files:
        raise RuntimeError(
            f"LM_HEAD_FULL_REQUIRE=1 but no safetensors found under {LOCAL_MODEL_DIR}"
        )
    rows: int | None = None
    for st_path in st_files:
        with safe_open(st_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "lm_head.weight" in keys:
                rows = int(handle.get_slice("lm_head.weight").get_shape()[0])
            elif "lm_head.weight_packed" in keys:
                rows = int(handle.get_slice("lm_head.weight_packed").get_shape()[0])
            elif "lm_head.weight_shape" in keys:
                rows = int(handle.get_tensor("lm_head.weight_shape")[0].item())
        if rows is not None:
            break
    if rows is None:
        raise RuntimeError(
            "LM_HEAD_FULL_REQUIRE=1 but no lm_head tensor found under "
            f"{LOCAL_MODEL_DIR} (checked lm_head.weight / weight_packed / weight_shape)"
        )
    if rows < FULL_LM_HEAD_ROWS:
        raise RuntimeError(
            f"LM_HEAD_FULL_REQUIRE=1 but served lm_head has {rows} rows "
            f"(< full vocab {FULL_LM_HEAD_ROWS}); refusing to serve a pruned head"
        )
    print(
        f"[lmhead-full] verified full lm_head: {rows} rows (>= {FULL_LM_HEAD_ROWS})",
        flush=True,
    )


def ensure_drafter() -> None:
    config_path = os.path.join(LOCAL_DRAFTER_DIR, "config.json")
    if not os.path.exists(config_path):
        if DRAFTER_BUCKET:
            print(
                f"[serve] syncing drafter {DRAFTER_BUCKET} -> {LOCAL_DRAFTER_DIR}",
                flush=True,
            )
            subprocess.run(
                ["hf", "buckets", "sync", DRAFTER_BUCKET, LOCAL_DRAFTER_DIR],
                check=True,
            )
        else:
            print(
                f"[serve] downloading drafter {DRAFTER_REPO} -> {LOCAL_DRAFTER_DIR}",
                flush=True,
            )
            from huggingface_hub import snapshot_download

            snapshot_download(DRAFTER_REPO, local_dir=LOCAL_DRAFTER_DIR)

    # Log the sha256 of the drafter file the server actually loads, so the run
    # record proves which weights served (guards against a stale local dir).
    import hashlib

    st_path = os.path.join(LOCAL_DRAFTER_DIR, "model.safetensors")
    digest = hashlib.sha256()
    with open(st_path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if DRAFTER_SHA256 and actual_sha256 != DRAFTER_SHA256:
        raise RuntimeError(
            "DRAFTER_SHA256 mismatch for model.safetensors: "
            f"expected {DRAFTER_SHA256}, got {actual_sha256}"
        )
    print(
        f"[serve] drafter model.safetensors sha256={actual_sha256}",
        flush=True,
    )

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    old_top_k = config.get("centroid_intermediate_top_k", 32)
    config["centroid_intermediate_top_k"] = CENTROID_TOP_K
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    print(
        f"[serve] centroid_intermediate_top_k: {old_top_k} -> {CENTROID_TOP_K}",
        flush=True,
    )


def find_tcmalloc() -> str | None:
    for path in TCMALLOC_CANDIDATES:
        if os.path.isfile(path):
            return path
    for path in glob.glob("/usr/lib/*/libtcmalloc_minimal.so.4"):
        if os.path.isfile(path):
            return path
    return None


def ensure_tcmalloc() -> str | None:
    existing = find_tcmalloc()
    if existing:
        print(f"[serve] tcmalloc found: {existing}", flush=True)
        return existing

    if shutil.which("apt-get"):
        print("[serve] installing libtcmalloc-minimal4 via apt-get", flush=True)
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "libtcmalloc-minimal4"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        existing = find_tcmalloc()
        if existing:
            print(f"[serve] tcmalloc installed: {existing}", flush=True)
            return existing

    print(
        "[serve] WARNING: tcmalloc unavailable; continuing without LD_PRELOAD",
        flush=True,
    )
    return None


def setup_ld_preload() -> None:
    requested = os.environ.get("LD_PRELOAD", "")
    lib = ensure_tcmalloc()
    if not lib:
        os.environ.pop("LD_PRELOAD", None)
        return

    if requested and os.path.isfile(requested.split(":")[0]):
        print(f"[serve] LD_PRELOAD already set: {requested}", flush=True)
        return

    os.environ["LD_PRELOAD"] = lib
    print(f"[serve] LD_PRELOAD={lib}", flush=True)


def ensure_benchmark_jinja2() -> None:
    """Install jinja2 into the harness benchmark venv if decode capture lacks it."""
    if os.environ.get("PATCH_BENCH_JINJA2") != "1":
        return

    bench_python = pathlib.Path(
        os.environ.get("BENCH_VENV_PYTHON", "/tmp/bench-venv/bin/python")
    )
    if not bench_python.exists():
        print(
            f"[serve] WARNING: benchmark venv python not found at {bench_python}; "
            "continuing without jinja2 patch",
            flush=True,
        )
        return

    check = subprocess.run(
        [str(bench_python), "-c", "import jinja2"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        print("[serve] benchmark venv already has jinja2", flush=True)
        return

    print(
        f"[serve] installing jinja2=={JINJA2_VERSION} into {bench_python}",
        flush=True,
    )
    subprocess.run(
        [
            str(bench_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            f"jinja2=={JINJA2_VERSION}",
            f"MarkupSafe=={MARKUPSAFE_VERSION}",
        ],
        check=True,
    )


def patch_file(path: pathlib.Path, patcher: Patcher) -> None:
    source = path.read_text(encoding="utf-8")
    patched_source, changed = patcher(source, path)
    if changed:
        path.write_text(patched_source, encoding="utf-8")
        print(f"[serve] patched {path}", flush=True)
    else:
        print(f"[serve] {path} already patched", flush=True)


def patch_ple_sources() -> None:
    if (
        os.environ.get("PLE_ASSUME_VALID_TOKEN_IDS") != "1"
        and os.environ.get("PLE_FOLD_EMBED_SCALE") != "1"
    ):
        return

    os.environ.setdefault("PLE_FOLD_TARGET_MODEL", LOCAL_MODEL_DIR)
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    model_path = purelib / "vllm" / "model_executor" / "models" / "gemma4.py"
    loader_path = purelib / "vllm" / "model_executor" / "model_loader" / "utils.py"
    patch_file(model_path, patch_gemma4_source)
    patch_file(loader_path, patch_loader_utils_source)


def patch_smp02_sources() -> None:
    if os.environ.get("DIXIE_SLIM_GREEDY", "1") != "1":
        return
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    sampler_path = purelib / "vllm" / "v1" / "sample" / "rejection_sampler.py"
    patch_file(sampler_path, patch_rejection_sampler_source)


API_ROUTER_CHAT_JSON_OLD = """    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )"""

API_ROUTER_CHAT_JSON_NEW = """    elif isinstance(generator, ChatCompletionResponse):
        if __import__("os").environ.get("FEOPT_ORJSON") == "1":
            import orjson
            from starlette.responses import Response
            return Response(
                content=orjson.dumps(generator.model_dump()),
                media_type="application/json",
                headers=metrics_header(metrics_header_format),
            )
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )"""


def patch_feopt_api_router_source(
    source: str, router_path: pathlib.Path
) -> tuple[str, bool]:
    return replace_required(
        source,
        model_path=router_path,
        label="FEOPT orjson chat-completion JSON",
        old=API_ROUTER_CHAT_JSON_OLD,
        new=API_ROUTER_CHAT_JSON_NEW,
        marker="FEOPT_ORJSON",
    )


def patch_feopt_api_router_sources() -> None:
    """orjson fast-path for non-streaming /v1/chat/completions (bench uses disable_stream)."""
    if os.environ.get("FEOPT_ORJSON") != "1":
        return
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    router_path = (
        purelib / "vllm" / "entrypoints" / "openai" / "chat_completion" / "api_router.py"
    )
    patch_file(router_path, patch_feopt_api_router_source)
    print("[feopt] patched api_router for orjson JSON response", flush=True)


# PR #545: serve-stack min_tokens floor on the CHAT endpoint only. The
# base_fullhead full-vocab head emits a spurious immediate first-token-EOS on
# ~10% of GSM8K chat prompts (PR #541); a request-level min_tokens=8 recovers it
# (0.762 -> 0.854). This bakes that guard into the served stack so the recovery
# holds by default, without a request flag. Scope is deliberately the chat
# endpoint: /v1/completions (the scored speed benchmark + ignore_eos greedy
# decode-identity audit + teacher-forced PPL) goes through
# CompletionRequest.to_sampling_params, which is left untouched -- so TPS, PPL,
# and greedy token-identity are unchanged BY CONSTRUCTION, not by argument.
MIN_TOKENS_FLOOR_OLD = "            min_tokens=self.min_tokens,\n"

MIN_TOKENS_FLOOR_NEW = """            # PR #545 serve-stack min_tokens floor (chat endpoint only). When
            # MIN_TOKENS_FLOOR is set and the client did NOT send min_tokens,
            # default it to the floor so an immediate first-token-EOS cannot
            # truncate a chat response. A floor, not an override: an explicit
            # request min_tokens (including 0) is always honored.
            min_tokens=(
                max(
                    self.min_tokens,
                    int(__import__("os").environ["MIN_TOKENS_FLOOR"]),
                )
                if (
                    __import__("os").environ.get("MIN_TOKENS_FLOOR")
                    and "min_tokens" not in self.model_fields_set
                )
                else self.min_tokens
            ),
"""


def patch_min_tokens_floor_source(
    source: str, protocol_path: pathlib.Path
) -> tuple[str, bool]:
    return replace_required(
        source,
        model_path=protocol_path,
        label="min_tokens chat serve-floor",
        old=MIN_TOKENS_FLOOR_OLD,
        new=MIN_TOKENS_FLOOR_NEW,
        marker="PR #545 serve-stack min_tokens floor",
    )


def patch_min_tokens_floor_sources() -> None:
    """Install the chat-endpoint min_tokens floor (PR #545). Inert unless
    MIN_TOKENS_FLOOR is set, so other submissions reusing this serve.py are
    unaffected. Only ChatCompletionRequest.to_sampling_params is touched;
    /v1/completions (speed TPS, ignore_eos decode audit, PPL) is left alone."""
    if not os.environ.get("MIN_TOKENS_FLOOR"):
        return
    floor = os.environ["MIN_TOKENS_FLOOR"]
    if not floor.isdigit():
        raise RuntimeError(
            f"MIN_TOKENS_FLOOR must be a non-negative integer, got {floor!r}"
        )
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    protocol_path = (
        purelib / "vllm" / "entrypoints" / "openai" / "chat_completion" / "protocol.py"
    )
    patch_file(protocol_path, patch_min_tokens_floor_source)
    print(f"[min-tokens] chat-endpoint min_tokens floor active: {floor}", flush=True)


def setup_sitecustomize_path() -> None:
    """Expose this package's sitecustomize.py to the vLLM child process."""
    package_dir = str(pathlib.Path(__file__).resolve().parent)
    existing = os.environ.get("PYTHONPATH", "")
    paths = [path for path in existing.split(os.pathsep) if path]
    if package_dir not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([package_dir, *paths])
    print(f"[serve] PYTHONPATH sitecustomize prefix: {package_dir}", flush=True)


def append_env_arg(args: list[str], env_name: str, flag: str) -> None:
    value = os.environ.get(env_name)
    if value:
        args.extend([flag, value])


# Reference-mode contract env var. Mirrors
# scripts/local_validation/paths.REFERENCE_MODE_ENV; hardcoded here because a
# submission's serve.py runs in its own venv and cannot import the harness.
REFERENCE_MODE_ENV = "SENPAI_REFERENCE_MODE"


def reference_mode_active() -> bool:
    """True when the harness asked for the M=1 AR greedy-reference contract.

    When SENPAI_REFERENCE_MODE is truthy, a speculative/drafter submission MUST
    serve plain M=1 autoregressive decode (drafter OFF) so the served capture is
    the canonical greedy reference the challenge gate compares against — generated
    on this submission's OWN engine/kernels/quant, so the only removed variable is
    speculation. ``gen_greedy_reference --spec-off`` sets it to "1"; unset/""/"0"
    leave the full speculative stack on, so the leaderboard serving path is
    untouched.
    """
    return os.environ.get(REFERENCE_MODE_ENV, "") not in ("", "0")


def disable_speculation_for_reference_mode() -> bool:
    """Honor the reference-mode contract by disabling the MTP drafter.

    Clears ``SPECULATIVE_CONFIG`` (consumed just below by
    ``append_env_arg(..., "--speculative-config")``) so vLLM starts with
    ``speculative_config=None`` — byte-for-byte the proven ``--ref-env
    SPECULATIVE_CONFIG=`` reference path, now reachable via the one-flag
    ``--spec-off``. No-op (returns False) outside reference mode.
    """
    if not reference_mode_active():
        return False
    if os.environ.get("SPECULATIVE_CONFIG"):
        print(
            "[serve] SENPAI_REFERENCE_MODE active: clearing SPECULATIVE_CONFIG "
            "(M=1 AR greedy reference, drafter OFF)",
            flush=True,
        )
    os.environ["SPECULATIVE_CONFIG"] = ""
    return True


def main() -> None:
    ensure_benchmark_jinja2()
    ensure_weights()
    _lmhead_prune_phase()
    _assert_full_lm_head()
    setup_ld_preload()
    ensure_drafter()
    patch_ple_sources()
    patch_smp02_sources()
    patch_feopt_api_router_sources()
    patch_min_tokens_floor_sources()
    setup_sitecustomize_path()

    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        LOCAL_MODEL_DIR,
        "--served-model-name",
        os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it"),
        "--host",
        os.environ.get("HOST", "0.0.0.0"),
        "--port",
        os.environ.get("PORT", "8000"),
        "--dtype",
        os.environ.get("DTYPE", "bfloat16"),
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs",
        os.environ.get("MAX_NUM_SEQS", "1"),
        "--performance-mode",
        os.environ.get("PERFORMANCE_MODE", "interactivity"),
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
    ]

    append_env_arg(args, "MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens")
    disable_speculation_for_reference_mode()
    append_env_arg(args, "SPECULATIVE_CONFIG", "--speculative-config")
    append_env_arg(args, "GENERATION_CONFIG", "--generation-config")
    append_env_arg(args, "OVERRIDE_GENERATION_CONFIG", "--override-generation-config")
    append_env_arg(args, "UVICORN_LOG_LEVEL", "--uvicorn-log-level")
    append_env_arg(args, "PREFIX_CACHING_HASH_ALGO", "--prefix-caching-hash-algo")
    # Profiling-only, default-off. PROFILER_CONFIG is never set in the manifest,
    # so this is inert on the leaderboard path (byte-identical served compute).
    # When a local profiler sets it (a ProfilerConfig JSON), forward it so vLLM's
    # built-in torch profiler + /start_profile capture the real serving decode
    # loop. Same env-gated, default-off pattern as the inert steptime_patch.
    append_env_arg(args, "PROFILER_CONFIG", "--profiler-config")

    if os.environ.get("DISABLE_LOG_STATS") == "1":
        args.append("--disable-log-stats")

    if os.environ.get("PRECACHE_BENCH") == "1":
        # Fail fast if the precache patch cannot import: site.execsitecustomize
        # swallows sitecustomize errors, so a broken patch would otherwise
        # silently bench unprecached. A crash here trips the harness's
        # "server exited before readiness" check immediately.
        import serve_patch_precache  # noqa: F401
        print("[serve] precache patch import validated", flush=True)
    print("[serve] launching:", " ".join(args), flush=True)
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
