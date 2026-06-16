"""Env-gated ``bm4`` launch override for the served Triton unified-attention kernel
(PR #442, wirbel).

LOCAL wall-clock A/B toggle ONLY. When ``WIRBEL_BM4_AB`` is set (``=1``) every launch
of ``vllm.v1.attention.ops.triton_unified_attention.kernel_unified_attention`` that
matches the served Gemma-4-E4B *verify* shape (head_dim in {256,512}, GQA
num_queries_per_kv=4 -> deployed BLOCK_M=16/BLOCK_Q=4, 2<=q_rows<=64) is re-launched
with the autotuned ``bm4`` config:

    BLOCK_M  16 -> 4        (occupancy knob; BLOCK_Q 4 -> 1, grid dim0 recomputed)
    num_stages 3 -> 2       (cp.async pipeline depth)

holding TILE_SIZE (=32) and num_warps (=4) at the deployed Triton defaults. This is
the *exact* ``{block_m:4, tile:32, warps:4, stages:2}`` config my joint autotune
(``triton_attn_joint_autotune.py``) modeled at +15.86 TPS over the 467.14 strict base
(Amdahl -> 483.00). This injector lets the decisive end-to-end served-stack wall A/B
test whether that microbench delta *realizes*.

WHY this is byte-exact (greedy-safe, maxdiff 0.0 -- proven empirically by the paired
census, not asserted): BLOCK_M / BLOCK_Q are the *query-row* (M-dimension) tile +
grid-grouping knob. They control which CTA processes which query row, NOT the per-row
reduction order. The QK^T accumulation over head_dim and the online-softmax reduction
over the KV axis are tiled by HEAD_SIZE_PADDED / TILE_SIZE (unchanged), so every
(query_token, head) output is the same FMA sequence regardless of BLOCK_M. In the 3D
split-KV path the per-segment partials are written to ``segm_*`` buffers indexed by
*global query-token row* (independent of BLOCK_Q), and ``reduce_segments`` merges them
per (token, head) on a grid of ``(q.shape[0], num_query_heads)`` (also independent of
BLOCK_Q); for the single-stream served stack (num_seqs=1) ``reduce_segments``'s only
use of BLOCK_Q -- ``find_seq_idx`` -> seq 0 -- is invariant. So ``reduce_segments`` is
left UNTOUCHED and stays byte-identical. ``num_stages`` 3->2 is a pure cp.async
pipeline-depth change (no MMA/K-reduction reorder -> maxdiff 0.0, the banked #270/#298
SDPA result).

CRUX (the realized-NULL hypothesis this A/B exists to test): the served stack runs
``SPLITKV_VERIFY=1`` -- the M=8 verify batch is redirected to the 3D split-KV path
(``submissions/fa2sw_precache_kenyan/splitkv_verify_patch.py`` overrides
max_seqlen_q=1). My +15.86 microbench measured the *2D* verify path (occupancy-bound,
~6 CTAs), but the served verify is already 3D and occupancy-SATURATED (deployed
BLOCK_Q=4 -> ~96 CTAs > 80 A10G SMs). bm4 expands that to ~288 CTAs -- no occupancy
headroom to recover, only extra launch/scheduling overhead -- so the modeled +15.86
(a 2D-occupancy fix) is predicted to collapse to ~the num_stages-only band (#428
<=+0.94 TPS) on the served wall. The per-launch census below logs IS_3D + the grid
expansion so the A/B can PROVE the served verify is 3D (not the 2D path the microbench
tuned).

Mechanism: the same deferred meta-path-finder idiom as the deployed
``splitkv_verify_patch.py`` / the #298 ``sdpa_num_stages_ab.py`` injector, so the
override lands the moment vLLM imports the kernel module -- BEFORE the ONEGRAPH/
LOOPGRAPH capture, so the bm4 cubin is the one baked into the captured graph. Only the
module global ``kernel_unified_attention`` is wrapped (the int4 GEMMs and every other
triton kernel are untouched). It composes with the deployed splitkv-verify finder
exactly as #298's injector did (each finder's ``_busy`` guard + loader-chaining;
splitkv still wraps the OUTER ``unified_attention``).

This file is loaded ONLY via a temporary env-gated hook appended to
``submissions/fa2sw_precache_kenyan/sitecustomize.py`` during the PR #442 wall A/B by
``served_bm4_wall_ab.py``; the hook is reverted and NEVER submitted. With
``WIRBEL_BM4_AB`` unset the deployed stack is byte-identical (this file is not even
imported). Forced launches are counted and logged to stderr so the A/B can PROVE the
candidate arm actually ran bm4 (guards against a silent no-op that would fake a verdict).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "vllm.v1.attention.ops.triton_unified_attention"
_RAW = os.environ.get("WIRBEL_BM4_AB", "").strip()

# bm4 config (the autotune winner): the ONLY deltas from deployed are BLOCK_M and
# num_stages; TILE_SIZE (=32) and num_warps (=4 triton default) are held. BLOCK_M and
# num_stages are env-selectable so the same injector can isolate the contributions:
#   default            -> bm4  : BLOCK_M=4  (the advisor-named config), num_stages=2
#   WIRBEL_BM4_BLOCK_M=8  -> the per-shape head-512 byte-exact optimum
#   WIRBEL_BM4_BLOCK_M=16 -> num_stages-only isolation (BLOCK_M unchanged, BLOCK_Q=4)
# BLOCK_Q is DERIVED per-launch from the real num_queries_per_kv (= max(1, BLOCK_M//nqpkv)),
# matching unified_attention's own rule (triton_unified_attention.py L842).
_BM4_BLOCK_M = int(os.environ.get("WIRBEL_BM4_BLOCK_M", "4"))
_BM4_NUM_STAGES = int(os.environ.get("WIRBEL_BM4_NUM_STAGES", "2"))

# Deployed verify shape gate (Gemma-4-E4B, GQA 8Q/2KV, head_dim in {256,512}).
_DEP_BLOCK_M = 16
_DEP_BLOCK_Q = 4
_NQPKV = 4
_HEADS = (256, 512)
_Q_ROWS_LO = 2            # >=2 separates verify (M=8) from decode (M=1); never touch decode
_Q_ROWS_HI = 64           # SPLITKV_VERIFY_MAX_Q -- the verify / tiny-prefill regime

# shared mutable state (this module is exec'd into sitecustomize's namespace; the
# closures below capture this dict so the forced-launch counter survives).
_AB_STATE = {"forced": 0, "passed": 0, "wrapped": False, "target_seen": False,
             "census": []}
_CENSUS_LIMIT = 6


def _log(msg: str) -> None:
    print(f"[bm4-ab] {msg}", file=sys.stderr, flush=True)


def _gate(kwargs) -> bool:
    """Pure predicate: is this kernel launch the served verify shape we retune?

    Fail-CLOSED on any missing/odd field (pass the launch through unchanged) so an
    unexpected caller never gets silently mis-tiled."""
    try:
        head = int(kwargs.get("HEAD_SIZE"))
        nqpkv = int(kwargs.get("num_queries_per_kv"))
        block_m = int(kwargs.get("BLOCK_M"))
        block_q = int(kwargs.get("BLOCK_Q"))
        q = kwargs.get("query_ptr")
        num_seqs = int(kwargs.get("num_seqs"))
        q_rows = int(q.shape[0])
    except Exception:  # noqa: BLE001 - fail-open
        return False
    return (
        head in _HEADS
        and nqpkv == _NQPKV
        and block_m == _DEP_BLOCK_M
        and block_q == _DEP_BLOCK_Q
        and _Q_ROWS_LO <= q_rows <= _Q_ROWS_HI
        and num_seqs >= 1
    )


def _bm4_block_q(nqpkv) -> int:
    """unified_attention's own rule (triton_unified_attention.py L842):
    BLOCK_Q = BLOCK_M // num_queries_per_kv (>= 1)."""
    return max(1, _BM4_BLOCK_M // int(nqpkv))


def _recompute_grid(grid, q_rows, num_seqs, block_q):
    """grid dim0 = total_num_q_blocks = q_rows // BLOCK_Q + num_seqs (kernel launch
    convention, triton_unified_attention.py L853). Only dim0 depends on BLOCK_Q;
    dims 1.. (num_kv_heads[, num_par_softmax_segments]) are untouched."""
    new_dim0 = q_rows // int(block_q) + num_seqs
    return (new_dim0,) + tuple(grid[1:])


def _make_wrapper(inner_kernel):
    """Wrap the module-global ``kernel_unified_attention`` JITFunction so
    ``kern[grid](...)`` re-tiles the served verify launch to bm4."""

    class _BM4ForcingKernel:
        def __getitem__(self, grid):
            def _call(*args, **kwargs):
                # The deployed unified_attention launches with ALL keyword args and a
                # concrete tuple grid. Only that exact fast path is retuned; anything
                # else (positional args, callable grid) passes straight through.
                if (not args) and isinstance(grid, tuple) and _gate(kwargs):
                    try:
                        q_rows = int(kwargs["query_ptr"].shape[0])
                        num_seqs = int(kwargs["num_seqs"])
                        new_bq = _bm4_block_q(kwargs["num_queries_per_kv"])
                        new_grid = _recompute_grid(grid, q_rows, num_seqs, new_bq)
                        new_kwargs = dict(kwargs)
                        old_bm = new_kwargs.get("BLOCK_M")
                        old_bq = new_kwargs.get("BLOCK_Q")
                        new_kwargs["BLOCK_M"] = _BM4_BLOCK_M
                        new_kwargs["BLOCK_Q"] = new_bq
                        new_kwargs["num_stages"] = _BM4_NUM_STAGES
                        _AB_STATE["forced"] += 1
                        c = _AB_STATE["forced"]
                        if len(_AB_STATE["census"]) < _CENSUS_LIMIT:
                            rec = {
                                "head": int(kwargs.get("HEAD_SIZE")),
                                "is_3d": bool(kwargs.get("IS_3D")),
                                "q_rows": q_rows, "num_seqs": num_seqs,
                                "old_grid": tuple(int(g) for g in grid),
                                "new_grid": tuple(int(g) for g in new_grid),
                                "old_block_m": old_bm, "old_block_q": old_bq,
                                "new_block_m": _BM4_BLOCK_M, "new_block_q": new_bq,
                                "num_stages": _BM4_NUM_STAGES,
                            }
                            _AB_STATE["census"].append(rec)
                            _log(f"CENSUS forced[{c}] head={rec['head']} IS_3D={rec['is_3d']} "
                                 f"q_rows={q_rows} grid {rec['old_grid']}->{rec['new_grid']} "
                                 f"BLOCK_M {old_bm}->{_BM4_BLOCK_M} BLOCK_Q {old_bq}->{new_bq} "
                                 f"num_stages->{_BM4_NUM_STAGES}")
                        elif c in (10, 20, 50, 100, 500) or c % 1000 == 0:
                            _log(f"forced bm4 (count={c})")
                        return inner_kernel[new_grid](*args, **new_kwargs)
                    except Exception as exc:  # noqa: BLE001 - fail-open to baseline
                        _log(f"WARN bm4 override skipped, baseline kept: {exc!r}")
                        return inner_kernel[grid](*args, **kwargs)
                _AB_STATE["passed"] += 1
                return inner_kernel[grid](*args, **kwargs)

            return _call

        def __getattr__(self, name):  # forward everything else to the real kernel
            return getattr(inner_kernel, name)

    return _BM4ForcingKernel()


def _patch_module(module):
    if _AB_STATE["wrapped"]:
        return
    kern = getattr(module, "kernel_unified_attention", None)
    if kern is None:
        _log("WARN: target module has no kernel_unified_attention; not patched")
        return
    module.kernel_unified_attention = _make_wrapper(kern)
    _AB_STATE["wrapped"] = True
    _AB_STATE["target_seen"] = True
    _log(f"PATCHED {_TARGET}.kernel_unified_attention -> bm4 "
         f"(BLOCK_M {_DEP_BLOCK_M}->{_BM4_BLOCK_M} [BLOCK_Q {_DEP_BLOCK_Q}->{max(1, _BM4_BLOCK_M // _NQPKV)}], "
         f"num_stages 3->{_BM4_NUM_STAGES}; TILE=32/num_warps=4 held)")


def _install():
    # belt+suspenders: if vLLM already imported the kernel module, patch in place.
    if _TARGET in sys.modules:
        _patch_module(sys.modules[_TARGET])
        return

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            _patch_module(module)

    class _TargetFinder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self._busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET or self._busy:
                return None
            self._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PatchingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _TargetFinder())
    _log(f"meta-path finder installed for {_TARGET} (bm4 verify retune)")


if _RAW and _RAW not in ("0", "", "false", "False"):
    _install()
elif _RAW:
    _log(f"no-op: WIRBEL_BM4_AB={_RAW!r} (disabled)")
