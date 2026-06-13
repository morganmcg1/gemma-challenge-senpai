"""Env-required L29 FFN affine replacement for the osoi5-baked Gemma4 target.

This patch is inert unless LFFN_LINEAR=1. When enabled it requires
LFFN_REQUIRE=1, loads one bf16 W tensor with shape [2561, 2560], and patches
only Gemma4DecoderLayer local layer 26. The layer map is fixed by osoi5 removal
of original layers {2, 3, 4, 36, 37}: original 29 -> local 26.
The target branch intentionally bypasses mlp and post_feedforward_layernorm.
With LFFN_PPL_EXACT=1, prompt_logprobs/PPL requests are marked by the runner and
fall back to the original dense layer forward.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import os
import sys
from typing import Any


TARGET_MODULE = "vllm.model_executor.models.gemma4"
LFFN_LINEAR = os.environ.get("LFFN_LINEAR", "0") == "1"
LFFN_REQUIRE = os.environ.get("LFFN_REQUIRE") == "1"
LFFN_WEIGHTS = os.environ.get("LFFN_WEIGHTS", "/tmp/lffn29/L29_ffn_ridge.pt")
LFFN_ALPHA = 1.0
LFFN_PPL_EXACT = os.environ.get("LFFN_PPL_EXACT", "0") == "1"
LFFN_ORIGINAL_LAYER = 29
LFFN_LOCAL_LAYER = 26
REMOVED_ORIGINAL_LAYERS = (2, 3, 4, 36, 37)
EXPECTED_WEIGHT_SHAPE = (2561, 2560)
EXPECTED_HIDDEN_SIZE = 2560

_WEIGHT_CPU: Any | None = None
_WEIGHT_BY_DEVICE: dict[tuple[str, Any], Any] = {}
_LFFN_PPL_EXACT_ACTIVE = False
_LFFN_PPL_EXACT_CALLS = 0
_LFFN_PPL_LAYER_LOGGED = False


def set_lffn_ppl_exact_active(active: bool) -> None:
    global _LFFN_PPL_EXACT_ACTIVE
    _LFFN_PPL_EXACT_ACTIVE = bool(active)


def _enabled_env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def _enabled_env_float(name: str, default: float) -> float:
    value = os.environ.get(name, str(default))
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{name} must be a finite float, got {value!r}")
    return parsed


def _derive_local_layer(original_layer: int) -> int:
    if original_layer in REMOVED_ORIGINAL_LAYERS:
        raise RuntimeError(
            f"LFFN original layer {original_layer} was removed by osoi5"
        )
    return original_layer - sum(
        removed < original_layer for removed in REMOVED_ORIGINAL_LAYERS
    )


def _validate_layer_map(original_layer: int, local_layer: int) -> None:
    derived = _derive_local_layer(original_layer)
    if (
        original_layer != LFFN_ORIGINAL_LAYER
        or local_layer != LFFN_LOCAL_LAYER
        or local_layer != derived
    ):
        raise RuntimeError(
            f"LFFN supports only original layer {LFFN_ORIGINAL_LAYER} "
            f"-> local layer {LFFN_LOCAL_LAYER} "
            f"(got original={original_layer}, local={local_layer}, derived={derived})"
        )


def _validate_weight(weight: Any, *, label: str, device: Any | None = None) -> Any:
    import torch

    if not torch.is_tensor(weight):
        raise RuntimeError(f"LFFN {label} must be a torch.Tensor")
    if tuple(weight.shape) != EXPECTED_WEIGHT_SHAPE:
        raise RuntimeError(
            f"LFFN {label} shape must be {EXPECTED_WEIGHT_SHAPE}, "
            f"got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(f"LFFN {label} must be bf16, got {weight.dtype}")
    if device is not None and weight.device != device:
        raise RuntimeError(
            f"LFFN {label} device must be {device}, got {weight.device}"
        )
    return weight.contiguous()


def _load_lffn_weight_cpu(path: str = LFFN_WEIGHTS) -> Any:
    if not path:
        raise RuntimeError("LFFN_WEIGHTS must be set when LFFN_LINEAR=1")
    if not os.path.isfile(path):
        raise RuntimeError(f"LFFN_WEIGHTS missing: {path}")

    import torch

    weight = torch.load(path, map_location="cpu", weights_only=True)
    return _validate_weight(weight, label="weight")


def _cuda_stream_is_capturing() -> bool:
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        if cuda is None or not hasattr(cuda, "is_current_stream_capturing"):
            return False
        return bool(cuda.is_current_stream_capturing())
    except Exception:
        return False


def _set_lffn_buffer(layer: Any, weight: Any) -> None:
    buffers = getattr(layer, "_buffers", None)
    if isinstance(buffers, dict) and "_lffn_weight" in buffers:
        buffers["_lffn_weight"] = weight
    else:
        setattr(layer, "_lffn_weight", weight)


def _install_lffn_buffer(layer: Any) -> None:
    global _WEIGHT_CPU

    import torch

    if _WEIGHT_CPU is None:
        _WEIGHT_CPU = _load_lffn_weight_cpu()
    weight = _WEIGHT_CPU
    try:
        if torch.cuda.is_available():
            weight = _WEIGHT_CPU.to(
                device=torch.device("cuda", torch.cuda.current_device()),
                dtype=torch.bfloat16,
            )
    except Exception:
        weight = _WEIGHT_CPU

    weight = _validate_weight(weight, label="layer buffer")
    if hasattr(layer, "register_buffer"):
        layer.register_buffer("_lffn_weight", weight, persistent=False)
    else:
        setattr(layer, "_lffn_weight", weight)


def _get_lffn_weight_for(layer: Any, hidden_states: Any) -> Any:
    import torch

    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(f"LFFN hidden_states must be bf16, got {hidden_states.dtype}")
    if getattr(hidden_states.device, "type", None) != "cuda":
        raise RuntimeError(f"LFFN hidden_states must be on CUDA, got {hidden_states.device}")

    weight = getattr(layer, "_lffn_weight", None)
    if weight is None:
        _install_lffn_buffer(layer)
        weight = getattr(layer, "_lffn_weight", None)
    if weight is None:
        raise RuntimeError("LFFN layer buffer was not installed")

    if weight.device != hidden_states.device or weight.dtype != hidden_states.dtype:
        if _cuda_stream_is_capturing():
            raise RuntimeError(
                "LFFN weight is not on the active CUDA device before graph capture"
            )
        weight = weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        weight = _validate_weight(
            weight, label="device layer buffer", device=hidden_states.device
        )
        _set_lffn_buffer(layer, weight)
    return weight


def _lffn_delta(layer: Any, pre_ffn_norm: Any) -> Any:
    import torch

    if pre_ffn_norm.shape[-1] != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"LFFN pre-FFN hidden size must be {EXPECTED_HIDDEN_SIZE}, "
            f"got {pre_ffn_norm.shape[-1]}"
        )
    flat = pre_ffn_norm.reshape(-1, EXPECTED_HIDDEN_SIZE)
    bias = flat.new_ones((flat.shape[0], 1))
    affine_input = torch.cat((flat, bias), dim=-1)
    delta = affine_input @ _get_lffn_weight_for(layer, pre_ffn_norm)
    return delta.reshape(pre_ffn_norm.shape)


def _apply_decoder_patch_to_class(cls: Any) -> None:
    original_init = getattr(cls, "__init__", None)
    original_forward = cls.forward

    if original_init is not None and not getattr(cls, "_lffn_init_patched", False):

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            if getattr(self, "layer_idx", None) == LFFN_LOCAL_LAYER:
                _install_lffn_buffer(self)

        cls.__init__ = __init__
        cls._lffn_init_patched = True

    def forward(
        self: Any,
        positions: Any,
        hidden_states: Any,
        residual: Any,
        per_layer_input: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        if getattr(self, "layer_idx", None) != LFFN_LOCAL_LAYER:
            return original_forward(
                self, positions, hidden_states, residual, per_layer_input, **kwargs
            )
        if LFFN_PPL_EXACT and _LFFN_PPL_EXACT_ACTIVE:
            global _LFFN_PPL_EXACT_CALLS, _LFFN_PPL_LAYER_LOGGED
            _LFFN_PPL_EXACT_CALLS += 1
            if not _LFFN_PPL_LAYER_LOGGED:
                _LFFN_PPL_LAYER_LOGGED = True
                print(
                    f"[lffn-ppl-layer] path=original_forward "
                    f"layer={LFFN_LOCAL_LAYER} "
                    f"exact_calls={_LFFN_PPL_EXACT_CALLS}",
                    file=sys.stderr,
                    flush=True,
                )
            return original_forward(
                self, positions, hidden_states, residual, per_layer_input, **kwargs
            )

        import torch

        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        pre_ffn_norm = self.pre_feedforward_layernorm(hidden_states)
        if getattr(self, "enable_moe_block", False):
            raise RuntimeError("LFFN replacement expects dense E4B layer, not MoE")
        delta = _lffn_delta(self, pre_ffn_norm)
        if LFFN_ALPHA == 0.0:
            hidden_states = residual
        else:
            hidden_states = residual + (delta * LFFN_ALPHA)

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            hidden_states = hidden_states + per_layer_contribution

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None

    cls.forward = forward


def _apply(module: Any) -> None:
    _apply_decoder_patch_to_class(module.Gemma4DecoderLayer)
    print(
        "[lffn] patched Gemma4DecoderLayer.forward for original layer "
        f"{LFFN_ORIGINAL_LAYER} "
        f"-> local layer {LFFN_LOCAL_LAYER} alpha={LFFN_ALPHA} "
        f"ppl_exact={int(LFFN_PPL_EXACT)} (pid {os.getpid()})",
        file=sys.stderr,
        flush=True,
    )


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _apply(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != TARGET_MODULE or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if LFFN_LINEAR:
    if not LFFN_REQUIRE:
        raise RuntimeError("LFFN_LINEAR=1 requires LFFN_REQUIRE=1")
    LFFN_ALPHA = _enabled_env_float("LFFN_ALPHA", 1.0)
    _validate_layer_map(
        _enabled_env_int("LFFN_ORIGINAL_LAYER", LFFN_ORIGINAL_LAYER),
        _enabled_env_int("LFFN_LOCAL_LAYER", LFFN_LOCAL_LAYER),
    )
    _WEIGHT_CPU = _load_lffn_weight_cpu()
    if TARGET_MODULE in sys.modules:
        _apply(sys.modules[TARGET_MODULE])
    else:
        sys.meta_path.insert(0, _Finder())
