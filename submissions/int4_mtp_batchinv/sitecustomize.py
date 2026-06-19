"""Auto-loaded at interpreter startup (Python imports ``sitecustomize`` during
``site`` initialization for every process whose ``sys.path`` contains this file).

``serve.py`` prepends this submission directory to ``PYTHONPATH`` before launching
the vLLM OpenAI server, so this module runs in every process in the server tree:
the ``api_server`` process, the (forked or spawned) ``EngineCore`` process, and the
worker process where ``GPUModelRunner`` actually builds attention groups. That is
the only place the ``num_heads`` attention-group fix has to take effect, and the
``EngineCore`` start method may be ``spawn`` (vLLM forces spawn when CUDA is already
initialized in the parent), so a parent-process monkeypatch would not propagate --
``PYTHONPATH`` + ``sitecustomize`` reaches every process regardless of fork/spawn.

We do NOT import vLLM here: ``sitecustomize`` runs at startup for *every* Python
process that uses this venv (pip, helper scripts, the benchmark client), and a full
vLLM/torch import there would be slow and could fail off-GPU. Instead we install a
one-shot ``sys.meta_path`` finder that applies the patch the moment
``vllm.v1.worker.gpu_model_runner`` is first imported, and is a no-op otherwise.
"""

import os
import sys

_TARGET = "vllm.v1.worker.gpu_model_runner"
_HERE = os.path.dirname(os.path.abspath(__file__))


# --- PR #755 lawine: env-gated LOCAL-profiling hooks ---------------------------
# Strict no-op for the shipped/benchmark submission: nothing below imports unless
# BOTH ``SENPAI_PR755_DIR`` (the research method dir) and a specific PR-755 flag
# are set, which only the PR #755 local A10G harness ever does. The manifest /
# leaderboard serving path never sets them, so the served numerics, cudagraph
# capture, and attention grouping are byte-for-byte unchanged there. The two
# hooks live in research/validity/strict_clean_attn_locus_743/ and each installs
# a one-shot meta_path finder (no vllm/torch import here):
#   SENPAI_NUMSPLITS_PROBE=<json>  -> read-only num_splits / is_batch_invariant probe
#   SENPAI_FORCE_NUMSPLITS1=1      -> force served attention to num_splits=1
_PR755_DIR = os.environ.get("SENPAI_PR755_DIR")
if _PR755_DIR and (
    os.environ.get("SENPAI_NUMSPLITS_PROBE")
    or os.environ.get("SENPAI_FORCE_NUMSPLITS1") == "1"
):
    if _PR755_DIR not in sys.path:
        sys.path.insert(0, _PR755_DIR)
    try:
        if os.environ.get("SENPAI_NUMSPLITS_PROBE"):
            import served_numsplits_probe

            served_numsplits_probe.install()
        if os.environ.get("SENPAI_FORCE_NUMSPLITS1") == "1":
            import served_numsplits_force

            served_numsplits_force.install()
    except Exception:
        import logging

        logging.getLogger("pr755.hooks").exception("PR #755 local hook failed")
# --- end PR #755 hooks ---------------------------------------------------------


def _apply(module) -> None:
    try:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import vllm_attn_group_patch

        vllm_attn_group_patch.apply(module)
    except Exception:
        import logging

        logging.getLogger("int4_mtp_drafter.patch").exception(
            "failed to apply attention-group num_heads patch"
        )


if _TARGET in sys.modules:
    _apply(sys.modules[_TARGET])
else:
    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _SpecDecodePatchFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            # One-shot: drop ourselves so the real loaders resolve the spec and
            # we never recurse through the find_spec() call below.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec_module = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec_module):
                _orig(module)
                _apply(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _SpecDecodePatchFinder())
