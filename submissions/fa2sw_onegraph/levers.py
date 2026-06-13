"""fa2sw + onegraph target-side runtime levers for the int4 Gemma4 endpoint.

Both levers are env-gated so one image can serve base / +fa2sw / +onegraph / both:

  FA2SW=1    Neutralise Gemma4Config's heterogeneous-head-dim TRITON force-pin.
             Gemma4 has 35 sliding (head_dim=256) + 7 global (head_dim=512)
             attention layers; vLLM force-pins TRITON_ATTN model-wide to avoid a
             mixed backend. With the pin neutralised, per-head_size selection
             routes the hd=256 sliding layers to FLASH_ATTN (FA2, which honours
             per_layer_sliding_window=512) while the hd=512 global layers stay on
             TRITON_ATTN (FA caps head_size at 256).

  ONEGRAPH=1 Capture the whole decode step as a single full CUDA graph
             (cudagraph_mode=FULL) instead of the FULL_AND_PIECEWISE default.
             serve.py turns this into a --compilation-config CLI arg.

The fa2sw monkeypatch must run in the same process as the model runner, so the
engine has to be in-process (serve.py sets VLLM_ENABLE_V1_MULTIPROCESSING=0
before importing vllm).
"""
from __future__ import annotations

import json
import os


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def fa2sw_enabled() -> bool:
    return _truthy(os.environ.get("FA2SW", "0"))


def onegraph_enabled() -> bool:
    return _truthy(os.environ.get("ONEGRAPH", "0"))


def apply_fa2sw() -> None:
    """Route the sliding hd=256 layers to FLASH_ATTN while keeping the global
    hd=512 layers on TRITON_ATTN. Two changes are needed:

    1. Neutralise Gemma4Config's heterogeneous-head-dim TRITON force-pin, so the
       backend stays None and per-head_size selection runs.
    2. Drop FLASHINFER from the sm_86 priority. Without it the hd=512 global
       layers (FLASH_ATTN caps at 256) would pick FLASHINFER, whose kernel can't
       dispatch head_dim=512 (`Unsupported max_mma_kv: 0`) and crashes at the
       dummy run. With FLASHINFER gone they fall through to TRITON_ATTN.
    """
    import vllm.model_executor.models.config as cfg
    import vllm.platforms.cuda as cuda_mod

    def _noop(vllm_config):  # noqa: ANN001 - mirrors the staticmethod signature
        return None

    cfg.Gemma4Config.verify_and_update_config = staticmethod(_noop)

    _orig_priorities = cuda_mod._get_backend_priorities

    def _no_flashinfer(*args, **kwargs):
        return [b for b in _orig_priorities(*args, **kwargs) if b.name != "FLASHINFER"]

    cuda_mod._get_backend_priorities = _no_flashinfer


def install_backend_recorder(out_path: str | None = None) -> list[dict]:
    """Record each attention layer's (head_size, backend) as it is built, and
    rewrite ``out_path`` on every layer so the final file is the full map."""
    import vllm.model_executor.layers.attention.attention as am

    records: list[dict] = []
    orig = am.Attention.__init__

    def patched(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            records.append(
                {
                    "layer": getattr(self, "layer_name", ""),
                    "head_size": getattr(self, "head_size", None),
                    "backend": self.attn_backend.get_name(),
                }
            )
            if out_path:
                summary: dict[str, int] = {}
                for r in records:
                    key = f"{r['head_size']}|{r['backend']}"
                    summary[key] = summary.get(key, 0) + 1
                with open(out_path, "w") as fh:
                    json.dump({"summary": summary, "layers": records}, fh, indent=2)
        except Exception:
            pass

    am.Attention.__init__ = patched
    return records


def onegraph_compilation_config() -> str:
    """JSON for --compilation-config that forces a single full decode CUDA graph."""
    return json.dumps({"cudagraph_mode": "FULL"})


def apply_levers(backend_map_out: str | None = None) -> list[str]:
    """Apply the in-process levers (fa2sw) and return the active lever names.
    onegraph is applied by serve.py via a CLI arg, but is reported here too."""
    active: list[str] = []
    install_backend_recorder(backend_map_out)
    if fa2sw_enabled():
        apply_fa2sw()
        active.append("fa2sw")
    if onegraph_enabled():
        active.append("onegraph")
    return active
