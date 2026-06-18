#!/usr/bin/env bash
# PR #639 Arm 3 -- GENUINE MSE-observer rebuild at the SHIPPED group sizes
# (body g128 + untied int4 head g128). Identical to the live int4_g128_lmhead
# recipe EXCEPT the scale-selection observer: instead of raw amin/amax (minmax),
# each (channel/group) clip is chosen to minimize int4 round-trip MSE. Zero
# on-disk-format / Marlin / TPS cost vs the live g128 submission -- only the
# stored scales differ. NOTE: the PR's literal "flip observer='mse'" is a no-op
# here (compressed_tensors 0.15.0.1 ships no observer impl, and the build derives
# scale=max_abs/7.5 from amin/amax directly), so a real MSE arm requires the
# clip-search added behind build_quant.py's new --observer flag. CPU-ONLY build.
set -euo pipefail
ROOT=/workspace/senpai/target
cd "$ROOT"
PY="${PY:-/tmp/vllm0220-srv/bin/python}"
SRC="${SRC:-/workspace/gemma_build/qat_unq}"
OUT="${OUT:-/workspace/gemma_build/int4_g128_mse_lmhead}"
echo "[build] ours-g128-MSE src=$SRC out=$OUT $(date -u +%FT%TZ)"
"$PY" -c "import compressed_tensors as ct; print('[build] compressed_tensors', ct.__version__)"
"$PY" submissions/int4_g128_lmhead/build_quant.py \
  --src "$SRC" \
  --out "$OUT" \
  --group-size 128 \
  --head-group-size 128 \
  --observer mse
echo "[build] DONE $(date -u +%FT%TZ)"
ls -la "$OUT"
