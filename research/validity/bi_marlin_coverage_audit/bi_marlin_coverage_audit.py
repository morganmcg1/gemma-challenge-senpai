#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #617 (stark): Does VLLM_BATCH_INVARIANT=1 cover the int4-Marlin verify GEMM on 0.22.0?

THE QUESTION
------------
wirbel #607 (yuvztndu): on a clean vLLM 0.22.0 gate the int4+MTP spec-verify path breaks greedy
identity on 31048/65536 tokens vs a 0/65536 plain-AR floor, and attributes the mechanism to
int4-Marlin M-dependence (M=8 spec-verify picks a different kernel schedule than M=1 AR). The
int4_mtp_batchinv submission sets VLLM_BATCH_INVARIANT=1 + MAX_NUM_SEQS=1 and claims that makes the
M=8 verify forward batch-invariant with the M=1 AR forward. This audit asks: does BI=1 actually
patch the Marlin GEMM, and -- if not -- is Marlin even the cause of the #607 break?

STATIC TRACE (resolved from 0.22.0 source, see PLAN.md / PR body)
----------------------------------------------------------------
enable_batch_invariant_mode() (batch_invariant.py:905) registers torch.library "aten" IMPL overrides
ONLY: aten::{mm,addmm,matmul,linear,bmm,_log_softmax,softmax,_softmax,mean.dim} (+ torch.bmm) and
precision backend flags. The int4 path is:
  CompressedTensorsWNA16.apply_weights (NO BI check)
   -> MarlinLinearKernel.apply_weights (NO BI check)
    -> apply_gptq_marlin_linear (NO BI check)
     -> ops.marlin_gemm  ==  torch.ops._C.marlin_gemm   [custom _C op, NOT aten]
So BI's aten-IMPL override structurally CANNOT intercept marlin_gemm, and nothing in the
compressed_tensors / marlin path reads envs.VLLM_BATCH_INVARIANT. On sm_86+bf16
should_use_atomic_add_reduce() is hard-False for all M (marlin_utils.py:461), so the Python-visible
reduce knobs do not vary with M either -- any M-dependence is INSIDE the precompiled CUDA schedule.

WHAT THIS MEASURES (real A10G sm_86, NO build, NO checkpoint, synthetic int4 weights)
------------------------------------------------------------------------------------
1. dispatch: vLLM's own choose_mp_linear_kernel selects Marlin uniquely for the served int4 shapes.
2. bi_runtime_proof: enable_batch_invariant_mode() CHANGES aten::mm output bits but leaves
   torch.ops._C.marlin_gemm bit-identical (same inputs, before vs after) -> runtime proof of
   non-coverage.
3. m_sweep: for each served fused shape, run marlin_gemm on a FIXED activation matrix's M-row
   prefix for M in {1,2,4,7,8,16,32,64,128} and bit-compare shared row 0 against the M=1 result.
   first_divergent_M and m8_bitexact decide whether Marlin is M-dependent AT the decode-verify width
   (=7/8) -- i.e. whether the #607 break is even attributable to Marlin.

VERDICT: one of BI_ALREADY_COVERS_MARLIN | MARLIN_OUTSIDE_BI__EXISTING_PATH_AVAILABLE |
MARLIN_OUTSIDE_BI__REQUIRES_CUSTOM_KERNEL | IMPOSSIBLE_ON_SM86, plus an honest attribution note.
analysis_only=true, official_tps=0, NO HF Job, single A10G.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0"

HERE = Path(__file__).resolve().parent

# served gemma-4-E4B-it int4 config (matches stark #613 SERVED_SHAPES) -------------------------- #
HIDDEN = 2560
INTERMEDIATE = 10240
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 256
N_LAYERS = 37
GROUP_SIZE = 128

# name -> (K=in, N=out): the fused decode GEMMs vLLM serves through CompressedTensorsWNA16->Marlin
SERVED_SHAPES: list[tuple[str, int, int]] = [
    ("qkv", HIDDEN, (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM),  # 2560 -> 3072
    ("o_proj", N_Q_HEADS * HEAD_DIM, HIDDEN),                   # 2048 -> 2560
    ("gate_up", HIDDEN, 2 * INTERMEDIATE),                      # 2560 -> 20480
    ("down", INTERMEDIATE, HIDDEN),                             # 10240 -> 2560
]

# decode-verify widths to probe. spec submissions use NUM_SPECULATIVE_TOKENS 6 (->7) / K_spec 7 (->8)
M_LIST = [1, 2, 4, 7, 8, 16, 32, 64, 128]
M_MAX = max(M_LIST)

# The aten ops enable_batch_invariant_mode() overrides on the SM80 family (read from source).
BI_PATCHED_ATEN_OPS = [
    "aten::mm", "aten::addmm", "aten::matmul", "aten::linear",  # SM80-family branch only
    "aten::bmm", "aten::_log_softmax", "aten::softmax", "aten::_softmax", "aten::mean.dim",
]


def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (set CUDA_VISIBLE_DEVICES=0)")
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name, "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}", "cc_tuple": list(cc),
        "is_sm86": bool(cc == (8, 6)), "is_sm80_family": bool(cc[0] == 8),
    }


def build_marlin(K: int, N: int, dev, seed: int = 0):
    """Faithful served int4 g128-symmetric Marlin weight + a run(a) closure (served defaults)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new, marlin_make_empty_g_idx)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize

    torch.manual_seed(seed)
    wtype = scalar_types.uint4b8
    gs = GROUP_SIZE if K % GROUP_SIZE == 0 else -1
    w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
    w_ref, q_w, s, g_idx, sort_idx, _perm = marlin_quantize(w, wtype, gs, act_order=False)
    zp = marlin_make_empty_g_idx(dev)
    ws = marlin_make_workspace_new(dev)

    def run(a, fp32_reduce: bool = True, atomic: bool = False):
        # served defaults: use_fp32_reduce=USE_FP32_REDUCE_DEFAULT(True), use_atomic_add hard-False
        return ops.marlin_gemm(
            a, None, q_w, None, s, None, None, zp, g_idx, sort_idx, ws,
            wtype, a.shape[0], N, K, True, atomic, fp32_reduce, False)

    return {"K": K, "N": N, "group_size": gs, "run": run, "wtype": wtype}


def resolve_dispatch(dev) -> dict[str, Any]:
    """vLLM's OWN selector picks Marlin for the served int4 shapes (authoritative)."""
    import torch
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig

    def cfg(K, N):
        return MPLinearLayerConfig(
            full_weight_shape=(K, N), partition_weight_shape=(K, N),
            weight_type=scalar_types.uint4b8, act_type=torch.bfloat16,
            group_size=GROUP_SIZE, zero_points=False, has_g_idx=False)

    selected = {name: choose_mp_linear_kernel(cfg(K, N)).__name__ for name, K, N in SERVED_SHAPES}
    uniq = sorted(set(selected.values()))
    return {"selected_per_shape": selected,
            "marlin_is_unique": bool(uniq == ["MarlinLinearKernel"])}


def bi_runtime_proof(dev) -> dict[str, Any]:
    """Decisive runtime proof of non-coverage: enable_batch_invariant_mode() changes aten::mm bits
    but leaves torch.ops._C.marlin_gemm bit-identical for the same inputs."""
    import torch
    from vllm.model_executor.layers import batch_invariant as bi

    out: dict[str, Any] = {}

    # marlin_gemm is in the _C namespace, NOT aten -> the aten-IMPL override cannot reach it.
    out["marlin_gemm_op_namespace"] = "torch.ops._C.marlin_gemm"
    out["marlin_gemm_in_C"] = bool(hasattr(torch.ops._C, "marlin_gemm"))
    out["bi_patches_only_aten"] = True  # from source: torch.library.Library("aten","IMPL")
    out["bi_patched_ops"] = BI_PATCHED_ATEN_OPS
    out["marlin_gemm_in_bi_patched_ops"] = any("marlin" in o for o in BI_PATCHED_ATEN_OPS)

    # Reference marlin GEMM (served int4) BEFORE enabling BI.
    g = build_marlin(HIDDEN, 2 * INTERMEDIATE, dev, seed=1)
    a = (torch.randn(8, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1)
    marlin_before = g["run"](a).clone()

    # Reference dense bf16 aten::mm BEFORE enabling BI.
    x = (torch.randn(8, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1)
    wdense = (torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02)
    mm_before = torch.mm(x, wdense).clone()

    # Enable BI in-process (idempotent; installs the aten overrides on the SM80 family).
    bi.enable_batch_invariant_mode()
    out["bi_mode_enabled"] = bool(bi._batch_invariant_MODE)

    marlin_after = g["run"](a).clone()
    mm_after = torch.mm(x, wdense).clone()

    marlin_bitexact = bool(torch.equal(marlin_before, marlin_after))
    mm_changed = not bool(torch.equal(mm_before, mm_after))
    out["marlin_gemm_bitexact_before_vs_after_BI"] = marlin_bitexact
    out["marlin_gemm_maxdiff_before_vs_after_BI"] = float(
        (marlin_before.float() - marlin_after.float()).abs().max().item())
    out["aten_mm_changed_by_BI"] = mm_changed
    out["aten_mm_maxdiff_before_vs_after_BI"] = float(
        (mm_before.float() - mm_after.float()).abs().max().item())
    # The proof: BI is active (it changed aten::mm) yet marlin_gemm is untouched (bit-identical).
    out["proof_BI_active_marlin_untouched"] = bool(mm_changed and marlin_bitexact)
    return out


def m_sweep(dev) -> dict[str, Any]:
    """Is marlin_gemm M-dependent at the decode-verify width? Compare shared row 0 of the output
    across M for a FIXED activation matrix. first_divergent_M + m8_bitexact decide attribution."""
    import torch
    results: dict[str, Any] = {"per_shape": {}, "M_list": M_LIST}
    for name, K, N in SERVED_SHAPES:
        g = build_marlin(K, N, dev, seed=0)
        a_full = (torch.randn(M_MAX, K, dtype=torch.bfloat16, device=dev) * 0.1)
        outs = {M: g["run"](a_full[:M]) for M in M_LIST}
        ref0 = outs[1][0]
        per_M = {}
        first_div = None
        for M in M_LIST:
            row0 = outs[M][0]
            md = float((row0.float() - ref0.float()).abs().max().item())
            exact = bool(torch.equal(row0, ref0))
            per_M[M] = {"maxdiff_row0_vs_m1": md, "bitexact_row0_vs_m1": exact}
            if (not exact) and first_div is None and M != 1:
                first_div = M
        results["per_shape"][name] = {
            "K": K, "N": N, "group_size": g["group_size"], "per_M": per_M,
            "first_divergent_M": first_div,
            "m8_bitexact_vs_m1": per_M[8]["bitexact_row0_vs_m1"],
            "m7_bitexact_vs_m1": per_M[7]["bitexact_row0_vs_m1"],
        }
    # aggregate: is ANY served shape M-dependent at the verify width (M in {7,8})?
    any_div_at_verify = any(
        not (s["m7_bitexact_vs_m1"] and s["m8_bitexact_vs_m1"])
        for s in results["per_shape"].values())
    results["any_shape_diverges_at_verify_width"] = any_div_at_verify
    results["min_first_divergent_M"] = min(
        [s["first_divergent_M"] for s in results["per_shape"].values()
         if s["first_divergent_M"] is not None], default=None)
    return results


def decide_verdict(dispatch: dict, proof: dict, sweep: dict) -> dict[str, Any]:
    marlin_outside_bi = (proof["proof_BI_active_marlin_untouched"]
                         and not proof["marlin_gemm_in_bi_patched_ops"])
    # Structural coverage verdict:
    if not marlin_outside_bi:
        coverage = "BI_ALREADY_COVERS_MARLIN"
    else:
        # Is there an EXISTING in-wheel batch-invariant FAST int4 GEMM path for compressed-tensors
        # WNA16 on sm_86? No: the only BI-covered GEMM is the dense aten::mm triton-persistent path
        # (dequant-to-bf16 = the ~48% matmul tax the int4 submission exists to avoid); the
        # compressed_tensors/marlin path has zero BI hooks and Marlin has no fixed-schedule mode.
        coverage = "MARLIN_OUTSIDE_BI__REQUIRES_CUSTOM_KERNEL"
    # Attribution: is the #607 break even caused by Marlin? Only if Marlin diverges at width 7/8.
    marlin_is_cause = sweep["any_shape_diverges_at_verify_width"]
    return {
        "coverage_verdict": coverage,
        "marlin_outside_bi": marlin_outside_bi,
        "marlin_m_dependent_at_verify_width": marlin_is_cause,
        "marlin_is_the_607_cause": marlin_is_cause,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_group", default="bi-marlin-coverage-audit")
    ap.add_argument("--wandb_name", default="stark/bi-marlin-coverage-audit")
    ap.add_argument("--resume_id", default=None, help="W&B run id to resume (liveness run)")
    args = ap.parse_args()

    dev = _device()
    gpu = _gpu_facts(dev)
    print(f"[bi-audit] GPU: {gpu['name']} cc={gpu['compute_capability']} sm86={gpu['is_sm86']}")

    dispatch = resolve_dispatch(dev)
    print(f"[bi-audit] marlin_is_unique={dispatch['marlin_is_unique']} {dispatch['selected_per_shape']}")

    # NOTE: run the M-sweep BEFORE enabling BI so the sweep reflects the served (non-BI) numerics;
    # bi_runtime_proof enables BI in-process and must run last.
    sweep = m_sweep(dev)
    for name, s in sweep["per_shape"].items():
        print(f"[bi-audit] {name:8s} K={s['K']:5d} N={s['N']:5d} "
              f"m8_bitexact={s['m8_bitexact_vs_m1']} first_div_M={s['first_divergent_M']}")
    print(f"[bi-audit] any_shape_diverges_at_verify_width(M in 7,8)="
          f"{sweep['any_shape_diverges_at_verify_width']} min_first_div_M={sweep['min_first_divergent_M']}")

    proof = bi_runtime_proof(dev)
    print(f"[bi-audit] BI active (aten::mm changed)={proof['aten_mm_changed_by_BI']} "
          f"marlin bit-identical before/after BI={proof['marlin_gemm_bitexact_before_vs_after_BI']}")
    print(f"[bi-audit] PROOF BI active & marlin untouched={proof['proof_BI_active_marlin_untouched']}")

    verdict = decide_verdict(dispatch, proof, sweep)
    print(f"[bi-audit] VERDICT coverage={verdict['coverage_verdict']} "
          f"marlin_is_607_cause={verdict['marlin_is_the_607_cause']}")

    payload = {
        "pr": 617, "analysis_only": True, "official_tps": 0, "no_hf_job": True, "no_build": True,
        "vllm_version": "0.22.0", "ts": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu, "dispatch": dispatch, "bi_runtime_proof": proof, "m_sweep": sweep,
        "verdict": verdict,
    }
    out_path = HERE / "bi_marlin_coverage_audit_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[bi-audit] wrote {out_path}")

    if not args.no_wandb:
        try:
            import wandb
            resume_id = args.resume_id
            if resume_id is None and (HERE / "run_id.txt").exists():
                resume_id = (HERE / "run_id.txt").read_text().strip()
            init_kw = dict(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                           entity=os.environ.get("WANDB_ENTITY"),
                           group=args.wandb_group, name=args.wandb_name, job_type="analysis")
            if resume_id:
                init_kw.update(id=resume_id, resume="allow")
            run = wandb.init(**init_kw)
            flat = {
                "audit_phase": 1, "audit_done": 1,
                "is_sm86": gpu["is_sm86"], "is_sm80_family": gpu["is_sm80_family"],
                "marlin_is_unique": dispatch["marlin_is_unique"],
                "bi_active_aten_mm_changed": proof["aten_mm_changed_by_BI"],
                "aten_mm_maxdiff_BI": proof["aten_mm_maxdiff_before_vs_after_BI"],
                "marlin_bitexact_before_vs_after_BI": proof["marlin_gemm_bitexact_before_vs_after_BI"],
                "marlin_maxdiff_before_vs_after_BI": proof["marlin_gemm_maxdiff_before_vs_after_BI"],
                "proof_BI_active_marlin_untouched": proof["proof_BI_active_marlin_untouched"],
                "any_shape_diverges_at_verify_width": sweep["any_shape_diverges_at_verify_width"],
                "min_first_divergent_M": (sweep["min_first_divergent_M"] or -1),
                "marlin_outside_bi": verdict["marlin_outside_bi"],
                "marlin_is_the_607_cause": verdict["marlin_is_the_607_cause"],
            }
            wandb.log(flat)
            run.summary.update({"coverage_verdict": verdict["coverage_verdict"]})
            run.config.update({"coverage_verdict": verdict["coverage_verdict"]}, allow_val_change=True)
            # per-shape M-sweep table
            cols = ["shape", "K", "N"] + [f"maxdiff_M{m}" for m in M_LIST] + ["first_div_M", "m8_exact"]
            tbl = wandb.Table(columns=cols)
            for name, s in sweep["per_shape"].items():
                tbl.add_data(name, s["K"], s["N"],
                             *[s["per_M"][m]["maxdiff_row0_vs_m1"] for m in M_LIST],
                             (s["first_divergent_M"] or -1), s["m8_bitexact_vs_m1"])
            wandb.log({"marlin_m_sweep": tbl})
            payload["wandb_run_id"] = run.id
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"[bi-audit] wandb run {run.id}")
            wandb.finish()
        except Exception as ex:  # noqa: BLE001
            print(f"[bi-audit] wandb failed (non-fatal): {type(ex).__name__}: {str(ex)[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
