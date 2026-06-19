#!/usr/bin/env python3
"""Compute PR #750 deliverables from the accumulated RESULTS.json.

Official anchoring (frame-invariant, same-checkpoint):
  int4_qat serves the SAME checkpoint as the fire config
  (google/gemma-4-E4B-it-qat-w4a16-ct, int4 W4A16 g32 Marlin, bf16 lm_head),
  no spec, BI=0 -> official a10g-small tps = 95.463 (BASELINE.md).
  R_int4 = 95.463 / local_int4_qat  is the pod->official ratio on the exact
  Marlin int4 body GEMM that dominates fire's single-stream decode.
  tps_BIx_official_anchored = tps_BIx_local * R_int4.

Cross-check (same-stack spec speedup on the official no-spec rung):
  tps_BI1_official_anchored ?= 95.463 * (tps_BI1_local / local_fire_specOFF_BI1)
  These agree iff local_fire_specOFF_BI1 ~= local_int4_qat (same checkpoint, no
  spec) -> validates that the fire serve.py wrapper / BI flag don't move the
  no-spec body TPS.
"""
from __future__ import annotations

import json
from pathlib import Path

BAR = 126.378           # int4_g128_lmhead official (rung to clear)
BAR_PLUS10 = 136.378    # advisor +10 target
ANCHOR_OFFICIAL = 95.463  # int4_qat official, SAME checkpoint as fire

R = json.loads(Path("runs/RESULTS.json").read_text())

tps_BI1 = R.get("tps_BI1")
tps_BI0 = R.get("tps_BI0")
local_int4_qat = R.get("local_int4_qat_tps")

out = dict(R)

# BI tax (the prize wirbel's recovery kernel targets)
if tps_BI1 is not None and tps_BI0 is not None:
    out["bi_tax_pct"] = (tps_BI0 - tps_BI1) / tps_BI0

# Official anchor
if local_int4_qat:
    R_int4 = ANCHOR_OFFICIAL / local_int4_qat
    out["R_int4_anchor"] = R_int4
    out["anchor_official_tps"] = ANCHOR_OFFICIAL
    out["anchor_submission"] = "int4_qat"
    if tps_BI1 is not None:
        out["tps_BI1_official_anchored"] = tps_BI1 * R_int4
    if tps_BI0 is not None:
        out["tps_BI0_official_anchored"] = tps_BI0 * R_int4

# Cross-check via same-stack spec speedup on the official no-spec rung
sp_off1 = R.get("local_fire_specOFF_BI1_tps")
if sp_off1 and tps_BI1 is not None:
    out["spec_speedup_BI1"] = tps_BI1 / sp_off1
    out["tps_BI1_official_anchored_xcheck"] = ANCHOR_OFFICIAL * (tps_BI1 / sp_off1)
sp_off0 = R.get("local_fire_specOFF_BI0_tps")
if sp_off0 and tps_BI0 is not None:
    out["spec_speedup_BI0"] = tps_BI0 / sp_off0

# Verdict
tpa = out.get("tps_BI1_official_anchored")
if tpa is not None:
    out["clears_126378"] = bool(tpa > BAR)
    out["margin_over_126378"] = tpa - BAR
    out["clears_136378_plus10"] = bool(tpa > BAR_PLUS10)
    out["margin_over_136378"] = tpa - BAR_PLUS10

Path("runs/RESULTS.json").write_text(json.dumps(out, indent=2, sort_keys=True))

# Pretty print the deliverables
def g(k):
    v = out.get(k)
    return f"{v:.4f}" if isinstance(v, float) else str(v)

print("================ PR #750 DELIVERABLES ================")
print(f"tps_BI1 (local)                 = {g('tps_BI1')}")
print(f"tps_BI0 (local)                 = {g('tps_BI0')}")
print(f"bi_tax_pct = (BI0-BI1)/BI0      = {g('bi_tax_pct')}")
print(f"identity_BI1                    = {out.get('identity_BI1','?')}")
print(f"identity_BI0                    = {out.get('identity_BI0','?')}")
print(f"local_int4_qat (anchor)         = {g('local_int4_qat_tps')}")
print(f"R_int4 = 95.463/local_int4_qat  = {g('R_int4_anchor')}")
print(f"tps_BI1_official_anchored       = {g('tps_BI1_official_anchored')}")
print(f"  xcheck (95.463*spec_speedup)  = {g('tps_BI1_official_anchored_xcheck')}")
print(f"tps_BI0_official_anchored       = {g('tps_BI0_official_anchored')}")
print(f"CLEARS 126.378 ?               -> {out.get('clears_126378','?')}  (margin {g('margin_over_126378')})")
print(f"CLEARS 136.378 (+10) ?         -> {out.get('clears_136378_plus10','?')}  (margin {g('margin_over_136378')})")
print("=====================================================")
