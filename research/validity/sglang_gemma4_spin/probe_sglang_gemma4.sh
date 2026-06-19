#!/usr/bin/env bash
# PR #745 — SGLang gemma-4-E4B spin probe (local A10G, sm_86).
#
# Reproduces the empirical blockers that prevent SGLang from serving the
# challenge model `google/gemma-4-E4B-it` (arch `Gemma4ForConditionalGeneration`,
# model_type `gemma4`) on the official a10g-small hardware. No HF Job; no serve
# (the runtime cannot be brought up — that is the result).
#
# Two stacks are probed:
#   (L1) the harness-pinned bench stack: sglang==0.5.2 + transformers==5.9.0
#   (L2) the oldest+newest gemma4-capable sglang: >=0.5.11 (native Gemma4 class)
#
# Run each block in its own throwaway uv venv. CUDA_VISIBLE_DEVICES=0 selects the
# single A10G exposed in this container.
set -uo pipefail
MODEL_DIR="$(dirname "$(find "${HOME}/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots" -name config.json 2>/dev/null | head -1)")"
echo "model config dir: ${MODEL_DIR}"

echo "###############################################################"
echo "# L1: harness-pinned sglang 0.5.2 + transformers 5.9.0"
echo "###############################################################"
echo "# Resolver proves the version matrix is unsatisfiable:"
echo "#   sglang[srt]==0.5.2 hard-pins transformers==4.56.1 (conflicts with 5.9.0)."
uv pip install --python 3.12 --dry-run "sglang[srt]==0.5.2" "transformers==5.9.0" 2>&1 | grep -iE "transformers|unsatisfiable|conclude" | head

uv venv /tmp/probe-l1 --python 3.12 >/dev/null 2>&1
uv pip install --python /tmp/probe-l1/bin/python "sglang[srt]==0.5.2" >/dev/null 2>&1   # pulls transformers 4.56.1
CUDA_VISIBLE_DEVICES=0 /tmp/probe-l1/bin/python - "${MODEL_DIR}" <<'PY'
import sys
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print("transformers:", transformers.__version__)
print("  'gemma4'  in CONFIG_MAPPING:", "gemma4" in CONFIG_MAPPING)   # False
print("  'gemma3n' in CONFIG_MAPPING:", "gemma3n" in CONFIG_MAPPING)  # True
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained(sys.argv[1]); print("  AutoConfig: OK")
except Exception as e:
    print("  AutoConfig RAISED:", type(e).__name__, "|", str(e).splitlines()[0])
PY

echo "###############################################################"
echo "# L2: gemma4-capable sglang (>=0.5.11) on A10G sm_86"
echo "###############################################################"
echo "# Every gemma4-capable sglang hard-deps flash-attn-4>=4.0.0b9 (Blackwell)"
echo "# and pins transformers 5.6.0/5.8.1 (never the baseline's 5.9.0):"
uv pip install --python 3.12 --dry-run "sglang[srt]>=0.5.11" "transformers==5.9.0" 2>&1 | grep -iE "transformers==|flash-attn|unsatisfiable|conclude" | head

uv venv /tmp/probe-l2 --python 3.12 >/dev/null 2>&1
uv pip install --python /tmp/probe-l2/bin/python --prerelease=allow "sglang==0.5.13" >/dev/null 2>&1
echo "# sgl_kernel prebuilt arch variants shipped in the 0.5.13 wheel:"
find /tmp/probe-l2/lib/python3.12/site-packages/sgl_kernel -name "common_ops*" 2>/dev/null \
  | sed 's#.*/site-packages/##'   # -> only sm90/ and sm100/, NO sm86
echo "# A10G compute capability: 86 -> no matching common_ops -> registry empty:"
CUDA_VISIBLE_DEVICES=0 /tmp/probe-l2/bin/python - "${MODEL_DIR}" <<'PY'
import sys, torch, transformers
print("torch:", torch.__version__, "cuda_avail:", torch.cuda.is_available())
print("transformers:", transformers.__version__,
      "| parses gemma4:", __import__("transformers").AutoConfig.from_pretrained(sys.argv[1]).model_type == "gemma4")
from sglang.srt.models.registry import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
print("  sglang supported archs (count):", len(archs))   # 0 -> sgl_kernel failed to load
print("  Gemma4ForConditionalGeneration registered:", "Gemma4ForConditionalGeneration" in archs)
PY
echo "DONE"
