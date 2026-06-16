#!/usr/bin/env python
"""PR #459 -- Byte-exact retune microbench for the FULL Triton verify-attention surface.

LOCAL on a single A10G (sm_86, on-target). Byte-exact, NO served-file change, NO HF
submission. Microbench only. This is the per-kernel half of the #459 reconciliation;
the served routing count (how many head-256 sliding layers actually reach Triton at the
M=8 verify) is measured separately by ``verify_surface_census.py``.

WHAT THIS MEASURES, and WHY it differs from #447
-------------------------------------------------
#447 (denken) drew the verify-wall map but timed the head-256 *sliding* layers as FA2
(its premise: FA_SLIDING routes all sliding layers to FlashAttention-2, so only the 7
head-512 *global* layers keep Triton -> "Triton verify surface = 1.27%"). PR #442
(wirbel) then showed by served census that head-256 sliding layers DO reach the Triton
3D split-KV kernel at the M=8 verify (FA2 only takes the M=1 drafter decode). So the true
Triton verify surface is LARGER than #447's head-512-only 1.27%.

This microbench times the Triton-3D split-KV kernel at BOTH served head dims --
  * head-512 global   (sliding_window = 0)   -- #447's only Triton layer; CONTROL: must
    reproduce 10.76us deployed + the num_stages 3->2 byte-exact saving 0.659us/layer.
  * head-256 sliding  (sliding_window = 512)  -- the layers #447 mis-attributed to FA2.
-- and finds, per head dim, the realized BYTE-EXACT retune saving.

BYTE-EXACT = the ONLY change is ``num_stages`` (cp.async pipeline depth). The deployed
``unified_attention`` launches ``kernel_unified_attention[grid](...)`` with NO num_stages
kwarg, so Triton's JIT default (num_stages=3, num_warps=4) is baked into the served cubin
(verified by reading vllm/v1/attention/ops/triton_unified_attention.py L967-1036: no
num_stages/num_warps in the launch). num_stages changes the pipeline depth, NOT the
QK^T accumulation order or the online-softmax KV reduction order -> bit-identical output
(maxdiff == 0.0 exactly), the banked #270/#298 SDPA result. We gate STRICTLY on
maxdiff == 0.0 (not #447's loose 2e-3 tol).

Knobs that are NOT byte-exact (excluded from the ceiling, demonstrated empirically here):
  * BLOCK_M / BLOCK_Q (the bm4 lever): #442 measured frac_identical 0.531 on the served
    M=8 verify -> greedy-UNSAFE. NOT byte-exact.
  * TILE_SIZE: changes the KV-tile granularity -> changes the online-softmax reduction
    blocking -> NOT byte-exact.
  * num_warps: changes the warp-level partial reduction -> NOT byte-exact.

THE CEILING (composed downstream in verify_surface_reconcile.py): with n256 head-256
sliding layers routing Triton (census) + 7 head-512 global layers,
  byte_exact_full_surface_ceiling_tps_delta
     = (n256 * saving_us_256 + 7 * saving_us_512) * TPS_PER_US
where TPS_PER_US = REALIZED_TPS_K7 / CYCLE_US (the realized-frontier Amdahl slope, the
#447 / #433 / #437 discipline: a kernel microbench delta is NOT an end-to-end delta).

Anchors (advisor branch approval-gated-8gpu-20260613, cited in PR #459):
  realized blanket-strict frontier  denken #423  5a6zq2yz   467.14 TPS
  deployed incumbent (non-equiv)    PR #52       2x9fm2zx   481.53 TPS / PPL 2.3772
  #447 head-512 Triton surface 1.27% + byte-exact 0.659us/layer (crrq2e1y)
  #442 routing flag: head-256 sliding reaches Triton at verify (gyw2ksvs)
  materiality bar = +2 TPS ; sigma_hw ~ 4.8 TPS ; PPL gate <= 2.42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse #447's EXACT timing primitives + proxy machinery (same CUDA-graph replay basis,
# same correctness-gated launch interception) so the head-512 control reproduces #447
# bit-for-bit and the head-256 measurement is methodologically identical.
from research.profiling.verify_wall_tile_scan import verify_wall_tile_scan as vwts  # noqa: E402

# ---------------------------------------------------------------------------
# Served geometry (CORRECTED from config.json /tmp/osoi5-v0-baked/config.json):
# 37 transformer layers = 7 global (head-512, full_attention) + 30 sliding (head-256,
# window 512). #447 used 42 (35 sliding + 7 global) -- WRONG; corrected here. The
# microbench is per-kernel so the layer-count fix only affects the surface composition,
# not the per-call timing.
# ---------------------------------------------------------------------------
N_GLOBAL_LAYERS = 7
N_SLIDING_LAYERS = 30
N_TOTAL_LAYERS = 37
HEAD_GLOBAL = 512
HEAD_SLIDING = 256
SLIDING_WINDOW = 512
N_Q_HEADS = 8
N_KV_HEADS = 2
M_VERIFY = 8

# Realized-frontier Amdahl slope (identical to #447; re-derived here so this file is
# self-contained and self-test can prove the arithmetic).
REALIZED_TPS_K7 = 467.14
ET_K7 = 3.851185944363104
CYCLE_US = ET_K7 / REALIZED_TPS_K7 * 1e6      # ~8243.7 us
TPS_PER_US = REALIZED_TPS_K7 / CYCLE_US        # ~0.056665 TPS per us saved off verify

# #447 head-512 byte-exact control values (crrq2e1y) -- cross-check the harness reproduces.
H512_447_DEPLOYED_US = 10.786134
H512_447_BYTEEXACT_US = 10.12736
H512_447_SAVING_US = 0.658774
H512_447_SPEEDUP_PCT = 6.107597

# Byte-exact sweep axis: num_stages only. Deployed = Triton JIT default (3).
SWEEP_NUM_STAGES = [2, 3, 4]
DEPLOYED_NUM_STAGES = 3
BYTE_EXACT_TOL = 0.0     # STRICT: maxdiff must be EXACTLY 0.0 (greedy-safe), not 2e-3.

# Non-byte-exact knobs we demonstrate break bit-identity (excluded from the ceiling).
DEMO_TILE_SIZE = 32      # vs deployed 16 -> reduction reblocking
DEMO_NUM_WARPS = 8       # vs deployed 4 -> warp-reduction reorder

CONFIRM_REPS = 7         # winner's-curse guard: alternating head-to-head re-time


# ---------------------------------------------------------------------------
# Per-head-dim byte-exact retune measurement.
# ---------------------------------------------------------------------------
def _byteexact_sweep_one_head(head_size: int, sliding: int, iters: int, warmup: int,
                              label: str) -> dict[str, Any]:
    """Time the deployed Triton-3D split-KV launch for this head dim, then the num_stages
    byte-exact sweep. Returns the deployed us, the best byte-exact us, the per-layer
    saving, and a winner's-curse-confirmed median delta -- all maxdiff==0.0 gated."""
    import torch

    # Served-default (cfg=None) output is the byte-exact REFERENCE for this shape.
    inp, segm, ref = vwts._attn_reference_out(head_size, sliding)

    base_cfg = {"BLOCK_M": None, "TILE_SIZE": None, "num_warps": None,
                "num_stages": None, "_ref": ref}
    base = vwts.time_attn_config(inp, segm, base_cfg, iters, warmup)
    base_us = base["us"]
    print(f"[vsm:{label}] deployed (cfg=None, JIT default s{DEPLOYED_NUM_STAGES}/w4) "
          f"= {base_us:.3f}us valid={base['valid']} maxerr={base['max_abs_err']:.2e} "
          f"captured={base['captured']}", flush=True)

    # num_stages-only sweep (byte-exact axis). BLOCK_M/TILE_SIZE/num_warps held at deployed
    # (None in cfg -> proxy leaves them) so the ONLY launch delta is num_stages.
    ns_rows = []
    for ns in SWEEP_NUM_STAGES:
        cfg = {"BLOCK_M": None, "TILE_SIZE": None, "num_warps": None,
               "num_stages": ns, "_ref": ref}
        r = vwts.time_attn_config(inp, segm, cfg, iters, warmup)
        byte_exact = bool(r["valid"] and r["max_abs_err"] == BYTE_EXACT_TOL)
        ns_rows.append({"num_stages": ns, "us": r["us"], "max_abs_err": r["max_abs_err"],
                        "valid": r["valid"], "byte_exact": byte_exact,
                        "captured": r["captured"]})
        print(f"[vsm:{label}]   num_stages={ns}: {r['us']:.3f}us  maxerr={r['max_abs_err']:.2e}  "
              f"byte_exact={byte_exact}", flush=True)

    # Best byte-exact config = fastest num_stages with maxdiff EXACTLY 0.0.
    be_rows = [r for r in ns_rows if r["byte_exact"]]
    best_be = min(be_rows, key=lambda r: r["us"]) if be_rows else None

    # Non-byte-exact demonstration: TILE_SIZE and num_warps must show maxdiff > 0.
    demo = {}
    for name, cfg in (
        ("tile32", {"BLOCK_M": None, "TILE_SIZE": DEMO_TILE_SIZE, "num_warps": None,
                    "num_stages": None, "_ref": ref, "q_rows": M_VERIFY, "num_seqs": 1}),
        ("warps8", {"BLOCK_M": None, "TILE_SIZE": None, "num_warps": DEMO_NUM_WARPS,
                    "num_stages": None, "_ref": ref}),
    ):
        try:
            r = vwts.time_attn_config(inp, segm, cfg, max(30, iters // 4), max(10, warmup // 4))
            demo[name] = {"us": r["us"], "max_abs_err": r["max_abs_err"],
                          "valid": r["valid"],
                          "byte_exact": bool(r["valid"] and r["max_abs_err"] == 0.0)}
        except Exception as exc:  # noqa: BLE001
            demo[name] = {"error": repr(exc)[:120]}

    # Winner's-curse guard: re-time deployed vs best-byte-exact alternating, CONFIRM_REPS
    # reps, take the median of each then the difference (the honest per-layer saving).
    confirm = {"confirmed": False, "reps": 0,
               "base_median_us": base_us,
               "best_median_us": best_be["us"] if best_be else base_us,
               "delta_us_per_layer": (base_us - best_be["us"]) if best_be else 0.0,
               "speedup_pct": ((base_us - best_be["us"]) / base_us * 100.0
                               if best_be and base_us > 0 else 0.0),
               "base_samples": [], "best_samples": []}
    if best_be is not None and best_be["num_stages"] != DEPLOYED_NUM_STAGES:
        win_cfg = {"BLOCK_M": None, "TILE_SIZE": None, "num_warps": None,
                   "num_stages": best_be["num_stages"], "_ref": ref}
        bs, ws, maxerrs = [], [], []
        for _ in range(CONFIRM_REPS):
            bs.append(vwts.time_attn_config(inp, segm, base_cfg, iters, warmup)["us"])
            wr = vwts.time_attn_config(inp, segm, win_cfg, iters, warmup)
            ws.append(wr["us"])
            maxerrs.append(wr["max_abs_err"])
        b0 = statistics.median(bs)
        b1 = statistics.median(ws)
        d = b0 - b1
        confirm = {"confirmed": True, "reps": CONFIRM_REPS,
                   "base_median_us": b0, "best_median_us": b1,
                   "delta_us_per_layer": d,
                   "speedup_pct": (d / b0 * 100.0 if b0 > 0 else 0.0),
                   "base_samples": bs, "best_samples": ws,
                   "confirm_maxerr_max": max(maxerrs) if maxerrs else None,
                   "confirm_byte_exact": bool(maxerrs and max(maxerrs) == 0.0)}
        print(f"[vsm:{label}] winner reconfirm ({CONFIRM_REPS} reps): base_med={b0:.3f}us "
              f"best_med={b1:.3f}us (s{best_be['num_stages']}) -> per-layer d={d:+.3f}us "
              f"({confirm['speedup_pct']:+.2f}%)  byte_exact={confirm['confirm_byte_exact']}",
              flush=True)

    # Honest per-layer byte-exact saving = CONFIRMED median delta, clamped >= 0 (a
    # served-equivalent-or-slower winner yields no saving).
    saving_us = max(0.0, confirm["delta_us_per_layer"]) if best_be is not None else 0.0
    return {
        "label": label, "head_size": head_size, "sliding_window": sliding,
        "deployed_us": base_us, "deployed_maxerr": base["max_abs_err"],
        "deployed_captured": base["captured"],
        "num_stages_rows": ns_rows,
        "best_byte_exact": best_be,
        "byte_exact_saving_us_per_layer": saving_us,
        "byte_exact_speedup_pct": confirm["speedup_pct"] if best_be is not None else 0.0,
        "best_num_stages": best_be["num_stages"] if best_be else DEPLOYED_NUM_STAGES,
        "confirm": confirm,
        "nonbyteexact_demo": demo,
    }


def run_gpu(args) -> dict[str, Any]:
    import torch

    dev = vwts._device()
    torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    facts = vwts._gpu_facts(dev)
    iters, warmup = args.iters, args.warmup
    vwts._install_attn_proxies()
    print(f"[vsm] device {facts['name']} cc{facts['compute_capability']} "
          f"torch {torch.__version__} iters={iters} warmup={warmup} "
          f"ctx={args.context_len}", flush=True)
    # #447 hard-codes context_len via _make_triton_inputs default arg = 128; keep it.

    heads = {
        "head512_global": _byteexact_sweep_one_head(HEAD_GLOBAL, 0, iters, warmup,
                                                    "h512"),
        "head256_sliding": _byteexact_sweep_one_head(HEAD_SLIDING, SLIDING_WINDOW, iters,
                                                     warmup, "h256"),
    }

    # head-512 control: cross-check against #447's banked byte-exact numbers.
    h512 = heads["head512_global"]
    control = {
        "h512_deployed_us": h512["deployed_us"],
        "h512_447_deployed_us": H512_447_DEPLOYED_US,
        "h512_saving_us": h512["byte_exact_saving_us_per_layer"],
        "h512_447_saving_us": H512_447_SAVING_US,
        "h512_deployed_within_10pct_of_447": bool(
            H512_447_DEPLOYED_US > 0
            and abs(h512["deployed_us"] - H512_447_DEPLOYED_US) / H512_447_DEPLOYED_US < 0.10),
        "h512_saving_within_0p3us_of_447": bool(
            abs(h512["byte_exact_saving_us_per_layer"] - H512_447_SAVING_US) < 0.30),
    }
    print(f"[vsm] CONTROL head-512: deployed {h512['deployed_us']:.3f}us "
          f"(#447 {H512_447_DEPLOYED_US:.3f})  saving {h512['byte_exact_saving_us_per_layer']:.3f}us "
          f"(#447 {H512_447_SAVING_US:.3f})  reproduces={control['h512_deployed_within_10pct_of_447'] and control['h512_saving_within_0p3us_of_447']}",
          flush=True)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "facts": facts, "iters": iters, "warmup": warmup,
        "context_len": args.context_len,
        "geometry": {"n_global": N_GLOBAL_LAYERS, "n_sliding": N_SLIDING_LAYERS,
                     "n_total": N_TOTAL_LAYERS, "head_global": HEAD_GLOBAL,
                     "head_sliding": HEAD_SLIDING, "sliding_window": SLIDING_WINDOW},
        "amdahl": {"realized_tps_k7": REALIZED_TPS_K7, "et_k7": ET_K7,
                   "cycle_us": CYCLE_US, "tps_per_us": TPS_PER_US},
        "heads": heads,
        "control_vs_447": control,
        "peak_vram_gib": peak_vram_gib,
    }


# ---------------------------------------------------------------------------
# Composition helper (also used by verify_surface_reconcile.py).
# ---------------------------------------------------------------------------
def compose_ceiling(saving_us_256: float, saving_us_512: float,
                    n256: int, n512: int = N_GLOBAL_LAYERS) -> dict[str, Any]:
    """byte_exact_full_surface_ceiling_tps_delta = (n256*s256 + n512*s512) * TPS_PER_US."""
    total_saving_us = n256 * saving_us_256 + n512 * saving_us_512
    tps_delta = total_saving_us * TPS_PER_US
    # #447's head-512-only ceiling, for the reconciliation delta.
    h512_only_us = n512 * saving_us_512
    h512_only_tps = h512_only_us * TPS_PER_US
    return {
        "n256_triton_layers": n256, "n512_triton_layers": n512,
        "saving_us_256": saving_us_256, "saving_us_512": saving_us_512,
        "total_byte_exact_saving_us": total_saving_us,
        "byte_exact_full_surface_ceiling_tps_delta": tps_delta,
        "head512_only_ceiling_us": h512_only_us,
        "head512_only_ceiling_tps_delta": h512_only_tps,
        "full_surface_minus_head512_only_tps": tps_delta - h512_only_tps,
        "tps_per_us": TPS_PER_US,
    }


# ---------------------------------------------------------------------------
# 0-GPU self-test.
# ---------------------------------------------------------------------------
def self_test() -> dict[str, Any]:
    res: dict[str, Any] = {}

    def ck(name, cond):
        res[name] = bool(cond)
        print(f"        {'ok ' if cond else 'XX '} {name}", flush=True)

    # Amdahl arithmetic matches #447.
    cyc = ET_K7 / REALIZED_TPS_K7 * 1e6
    ck("a_cycle_us", abs(cyc - CYCLE_US) < 1e-6)
    ck("b_slope_pos", TPS_PER_US > 0 and abs(TPS_PER_US - REALIZED_TPS_K7 / CYCLE_US) < 1e-12)
    # Geometry correction: 37 = 7 + 30 (not 42).
    ck("c_geometry_37", N_GLOBAL_LAYERS + N_SLIDING_LAYERS == N_TOTAL_LAYERS == 37)
    # Compose math: head-512-only reproduces #447's +0.26 TPS at saving 0.659, n256=0.
    c0 = compose_ceiling(0.0, H512_447_SAVING_US, n256=0)
    ck("d_head512_only_matches_447",
       abs(c0["byte_exact_full_surface_ceiling_tps_delta"] - 0.2613) < 0.01)
    # Larger surface monotone: adding head-256 layers raises the ceiling.
    c14 = compose_ceiling(0.40, H512_447_SAVING_US, n256=14)
    c30 = compose_ceiling(0.40, H512_447_SAVING_US, n256=30)
    ck("e_monotone_in_n256",
       c30["byte_exact_full_surface_ceiling_tps_delta"]
       > c14["byte_exact_full_surface_ceiling_tps_delta"]
       > c0["byte_exact_full_surface_ceiling_tps_delta"])
    # Strict-NULL plausibility: even n256=30 at saving 0.5us caps under +2 TPS.
    cmax = compose_ceiling(0.50, 0.66, n256=30)
    ck("f_n30_under_2tps", cmax["byte_exact_full_surface_ceiling_tps_delta"] < 2.0)
    # A surface big enough to reopen: would need ~+2 TPS = ~35us byte-exact saving.
    us_needed = 2.0 / TPS_PER_US
    ck("g_reopen_needs_35us", 34.0 < us_needed < 36.5)
    # Byte-exact gate is STRICT 0.0.
    ck("h_strict_tol_zero", BYTE_EXACT_TOL == 0.0)
    # NaN-clean.
    ck("i_finite", all(math.isfinite(x) for x in
                       [cyc, TPS_PER_US, c30["byte_exact_full_surface_ceiling_tps_delta"],
                        us_needed]))
    npass = sum(1 for v in res.values() if v is True)
    total = len(res)
    print(f"[vsm] self-test: {'PASS' if npass == total else 'FAIL'} ({npass}/{total})",
          flush=True)
    res["_n_pass"] = npass
    res["_n_total"] = total
    return res


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else round(o, 6)
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fast: 30 iters")
    ap.add_argument("--out-dir",
                    default="research/equivalence_escalation/triton_verify_surface")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.self_test:
        st = self_test()
        with open(os.path.join(args.out_dir, "microbench_selftest.json"), "w") as f:
            json.dump(_jsonable(st), f, indent=2)
        return 0 if st["_n_pass"] == st["_n_total"] else 1
    if args.smoke:
        args.iters, args.warmup = 30, 10

    meas = run_gpu(args)
    print("[vsm] running 0-GPU self-test gate ...", flush=True)
    st = self_test()
    meas["self_test"] = _jsonable(st)
    meas["self_test_passes"] = bool(st["_n_pass"] == st["_n_total"])

    with open(os.path.join(args.out_dir, "microbench_results.json"), "w") as f:
        json.dump(_jsonable(meas), f, indent=2)

    print("\n=== MICROBENCH SUMMARY ===", flush=True)
    for k, h in meas["heads"].items():
        print(f"  {k}: deployed {h['deployed_us']:.3f}us  byte-exact best "
              f"s{h['best_num_stages']} saving {h['byte_exact_saving_us_per_layer']:.3f}us/layer "
              f"({h['byte_exact_speedup_pct']:+.2f}%)", flush=True)
    print(f"  control_vs_447: {json.dumps(meas['control_vs_447'])}", flush=True)
    print(f"  artifacts -> {os.path.join(args.out_dir, 'microbench_results.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
