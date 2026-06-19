"""PR #717 -- in-memory RTN fake-quant injector for the int8-locus GPQA-D instrument.

Auto-imported by Python's `site` in EVERY process started with this dir on PYTHONPATH,
including the vLLM v1 EngineCore worker (it inherits PYTHONPATH and re-runs site init).
Same mechanism as research/validity/vanilla_base_serve_regression/serve_inject.

FOUR jobs:
  1. Prometheus route-name compat shim (verbatim from serve_inject; numerics-orthogonal).
  2. fa_sliding_patch        (FA_SLIDING=1)            -- #696 surgical attn recipe.
  3. surgical_attn_patch     (SURGICAL_ATTN_USE_3D_OFF=1) -- #696 surgical attn recipe.
  4. IN-MEMORY RTN fake-quant (THIS card): monkeypatch Gemma4ForConditionalGeneration
     .load_weights to wrap the checkpoint weight stream, applying round-to-nearest
     fake-quant (compressed_tensors symmetric, minmax observer -- the SAME math as
     submissions/int4_g128_lmhead/build_quant.py, which is fern #659's `skeleton`):
        - the 343 language-model body Linear modules (official_quantized_modules.json):
          int4 group-128  -- the int4_g128 skeleton
        - the subset in language_model.layers[INT8_LO..INT8_HI]: int8 (fern's L14-27
          upgrade); group size FAKEQUANT_INT8_GROUP (default 128; int8 error ~10x below
          int4 so per-channel vs g128 is immaterial -- logged either way)
        - lm_head: SYNTHESIZED from bf16 embed_tokens as int4 group-128 (fern's
          `lm_head = int4_g128 (locked)`); embed_tokens itself stays bf16
     Served as bf16 (dequantized) -- this is FAKE-quant: it reproduces the quantization
     ROUNDING ERROR faithfully (fern's nmjvtfov used the same in-memory RTN from the
     qat-unquantized source), running bf16 GEMM rather than int4/int8 Marlin kernels.

If FAKEQUANT_DISABLE=1 the RTN patch is a deliberate no-op (full-bf16 control).
Otherwise a missing/failed patch RAISES -- a silent no-op would serve full-precision
bf16 and silently inflate GPQA, so we fail loud.
"""
import json
import os
import sys

# ----------------------------------------------------------------------------- #
# 1. prometheus route-name compat shim (verbatim from serve_inject sitecustomize)
# ----------------------------------------------------------------------------- #
def _install_prometheus_route_compat():
    try:
        import prometheus_fastapi_instrumentator.routing as _r
        from starlette.routing import Match, Mount
    except Exception:
        return

    def _safe_get_route_name(scope, routes, route_name=None):
        for route in routes:
            try:
                match, child_scope = route.matches(scope)
            except Exception:
                continue
            path = getattr(route, "path", None)
            sub = getattr(route, "routes", None)
            if match == Match.FULL:
                if path is None:
                    if sub:
                        return _safe_get_route_name({**scope, **child_scope}, sub, route_name)
                    return route_name
                route_name = path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child = _safe_get_route_name(child_scope, route.routes, route_name)
                    route_name = None if child is None else route_name + child
                return route_name
            elif match == Match.PARTIAL and route_name is None and path is not None:
                route_name = path
        return None

    _r._get_route_name = _safe_get_route_name
    print(f"[sc-717] prometheus route-name compat shim installed pid={os.getpid()}",
          file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- #
# 4. in-memory RTN fake-quant
# ----------------------------------------------------------------------------- #
EMBED_TOKENS = "model.language_model.embed_tokens.weight"
_LAYER_PREFIX = "model.language_model.layers."


def _load_targets():
    """Return (int4_set, int8_set) of module BASE names (without `.weight`)."""
    mod_list_path = os.environ["FAKEQUANT_MODULE_LIST"]
    mods = set(json.load(open(mod_list_path)))
    assert len(mods) == 343, f"expected 343 quant modules, got {len(mods)}"
    lo, hi = (int(x) for x in os.environ.get("FAKEQUANT_INT8_LAYERS", "14-27").split("-"))
    int8_layers = set(range(lo, hi + 1))
    int8_set, int4_set = set(), set()
    for m in mods:
        if m.startswith(_LAYER_PREFIX):
            li = int(m[len(_LAYER_PREFIX):].split(".")[0])
            (int8_set if li in int8_layers else int4_set).add(m)
        else:
            int4_set.add(m)  # e.g. model.language_model.per_layer_model_projection
    return int4_set, int8_set, lo, hi


def _make_fakequant():
    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
    from compressed_tensors.quantization.utils.helpers import calculate_qparams

    def fakequant(w, num_bits, group_size):
        orig_dtype = w.dtype
        w = w.to(torch.float32)
        out_d, in_d = w.shape
        if group_size == -1 or in_d % group_size != 0:
            qa = QuantizationArgs(num_bits=num_bits, type="int", strategy="channel",
                                  symmetric=True, observer="minmax")
            mn = w.amin(dim=-1, keepdim=True)
            mx = w.amax(dim=-1, keepdim=True)
        else:
            qa = QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                                  group_size=group_size, symmetric=True, observer="minmax")
            ng = in_d // group_size
            wg = w.reshape(out_d, ng, group_size)
            mn = wg.amin(dim=-1)
            mx = wg.amax(dim=-1)
        scale, zp = calculate_qparams(mn, mx, qa)
        q = quantize(w, scale, zp, qa)
        deq = dequantize(q, scale, zp, qa)
        rel = float((w - deq).norm() / w.norm().clamp_min(1e-9))
        return deq.to(orig_dtype), rel

    return fakequant


def _install_rtn_fakequant():
    if os.environ.get("FAKEQUANT_DISABLE") == "1":
        print(f"[sc-717] RTN fake-quant DISABLED (full-bf16 control) pid={os.getpid()}",
              file=sys.stderr, flush=True)
        return

    import torch  # noqa: F401
    from vllm.model_executor.models import gemma4_mm

    int4_set, int8_set, lo, hi = _load_targets()
    int8_group = int(os.environ.get("FAKEQUANT_INT8_GROUP", "128"))
    int4_group = int(os.environ.get("FAKEQUANT_INT4_GROUP", "128"))
    lmhead_bits = int(os.environ.get("FAKEQUANT_LMHEAD_BITS", "4"))
    lmhead_group = int(os.environ.get("FAKEQUANT_LMHEAD_GROUP", "128"))
    fakequant = _make_fakequant()

    cls = gemma4_mm.Gemma4ForConditionalGeneration
    orig_load_weights = cls.load_weights

    def _norm_layer_idx(name, kind):
        """Layer index of `...layers.{i}.self_attn.{kind}_norm.weight`, else None."""
        suf = f".self_attn.{kind}_norm.weight"
        if name.startswith(_LAYER_PREFIX) and name.endswith(suf):
            return int(name[len(_LAYER_PREFIX):].split(".")[0])
        return None

    def _rtn_stream(weights):
        embed_w = None
        # Gemma-4-E4B shares KV for the last `num_kv_shared_layers` (=18) layers: those
        # deep layers carry q_proj/o_proj/q_norm but NO k_proj/v_proj/k_norm (KV comes
        # from the last non-shared layer). vLLM allocates a k_norm RMSNorm for EVERY
        # layer (gemma4.py:429) yet only USES it on non-shared layers (gemma4.py:522),
        # so the shared-layer k_norm params are orphaned and the strict
        # track_weights_loading check fails. We synthesize identity (ones) k_norm for
        # any layer that has q_norm but no k_norm -- structurally derived from the
        # stream, and numerically INERT (those k_norm are never called in forward).
        qnorm_shape, knorm_layers = {}, set()  # per-layer q_norm (shape,dtype); k_norm seen
        stats = {"int4": 0, "int8": 0, "passthrough": 0,
                 "int4_rel_sum": 0.0, "int8_rel_sum": 0.0,
                 "int4_rel_max": 0.0, "int8_rel_max": 0.0}
        for name, tensor in weights:
            qi = _norm_layer_idx(name, "q")
            if qi is not None:
                qnorm_shape[qi] = (tuple(tensor.shape), tensor.dtype)
            ki = _norm_layer_idx(name, "k")
            if ki is not None:
                knorm_layers.add(ki)
            if name == EMBED_TOKENS:
                embed_w = tensor  # capture for synthetic lm_head; pass through bf16
                stats["passthrough"] += 1
                yield name, tensor
                continue
            base = name[:-7] if name.endswith(".weight") else None
            if base is not None and base in int8_set:
                deq, rel = fakequant(tensor, 8, int8_group)
                stats["int8"] += 1
                stats["int8_rel_sum"] += rel
                stats["int8_rel_max"] = max(stats["int8_rel_max"], rel)
                yield name, deq
            elif base is not None and base in int4_set:
                deq, rel = fakequant(tensor, 4, int4_group)
                stats["int4"] += 1
                stats["int4_rel_sum"] += rel
                stats["int4_rel_max"] = max(stats["int4_rel_max"], rel)
                yield name, deq
            else:
                stats["passthrough"] += 1
                yield name, tensor
        # synthesize identity k_norm for the KV-shared (orphaned) layers. Gemma-4-E4B's
        # per-layer head_dim VARIES (q/k_norm are [256] or [512] by the layer's attention
        # type), so each synthetic k_norm copies the SAME layer's q_norm shape: q_norm is
        # present for all 42 layers, and one layer's q_norm and k_norm are both
        # RMSNorm(head_dim) -> identical shape. ones is numerically inert (shared-layer
        # k_norm is never called in forward, gemma4.py:522); only the SHAPE must match.
        missing_knorm = sorted(set(qnorm_shape) - knorm_layers)
        synth_shapes = {}
        for li in missing_knorm:
            if li not in qnorm_shape:
                raise RuntimeError(f"layer {li} needs a synthetic k_norm but has no "
                                   f"q_norm in the stream; cannot infer its head_dim")
            shp, dt = qnorm_shape[li]
            synth_shapes[shp[0]] = synth_shapes.get(shp[0], 0) + 1
            yield (f"{_LAYER_PREFIX}{li}.self_attn.k_norm.weight",
                   torch.ones(shp, dtype=dt))
        # synthesize the untied int4-g128 lm_head from the bf16 embedding
        if embed_w is not None:
            lm_deq, lm_rel = fakequant(embed_w, lmhead_bits, lmhead_group)
            print(f"[sc-717] RTN APPLIED pid={os.getpid()} int4={stats['int4']} "
                  f"int8={stats['int8']} (L{lo}-{hi}) passthrough={stats['passthrough']} "
                  f"kv_shared_knorm_synth={len(missing_knorm)}{dict(sorted(synth_shapes.items()))} "
                  f"| int4_rel mean={stats['int4_rel_sum']/max(stats['int4'],1):.4f} "
                  f"max={stats['int4_rel_max']:.4f} | int8_rel "
                  f"mean={stats['int8_rel_sum']/max(stats['int8'],1):.4f} "
                  f"max={stats['int8_rel_max']:.4f} | lm_head int{lmhead_bits}g{lmhead_group} "
                  f"rel={lm_rel:.4f}", file=sys.stderr, flush=True)
            assert stats["int4"] + stats["int8"] == 343, (
                f"fake-quant covered {stats['int4']+stats['int8']} modules, expected 343")
            yield "lm_head.weight", lm_deq
        else:
            raise RuntimeError("embed_tokens.weight never appeared in the weight stream; "
                               "cannot synthesize the untied lm_head")

    def patched_load_weights(self, weights):
        return orig_load_weights(self, _rtn_stream(weights))

    cls.load_weights = patched_load_weights
    print(f"[sc-717] RTN fake-quant patch installed on "
          f"Gemma4ForConditionalGeneration.load_weights pid={os.getpid()} "
          f"int4={len(int4_set)} int8={len(int8_set)} L{lo}-{hi}",
          file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- #
# install
# ----------------------------------------------------------------------------- #
_install_prometheus_route_compat()

_SUB = os.environ.get("PR557_PATCH_DIR",
                      "/workspace/senpai/target/submissions/fa2sw_strict_surgical357")
if _SUB not in sys.path:
    sys.path.insert(0, _SUB)

if os.environ.get("FA_SLIDING", "0") == "1":
    import fa_sliding_patch  # noqa: F401
    print(f"[sc-717] fa_sliding_patch imported pid={os.getpid()}", file=sys.stderr, flush=True)

if os.environ.get("SURGICAL_ATTN_USE_3D_OFF", "0") == "1":
    import surgical_attn_patch  # noqa: F401
    print(f"[sc-717] surgical_attn_patch imported pid={os.getpid()}", file=sys.stderr, flush=True)

_install_rtn_fakequant()
