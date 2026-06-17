#!/usr/bin/env python
"""PR #584 lawine — spec-dec ACHIEVABLE PARETO on base_fullhead.

Converts fern's `any_drafter_at_k_clears_ship` from a model-extrapolation into an
EMPIRICAL verdict by MEASURING intermediate drafter points on the cost-acceptance
plane between the two known corners:
  * ngram  (cheap draft, low acceptance)   — fern #573, acceptance 2.2865
  * MTP K=7 (trained draft head, high A)    — lawine #572, acceptance 3.8443

Decisive question: can a TUNED ngram (prompt-lookup) drafter raise its exact-verify
acceptance above the A_ship break-even while keeping verify cost cheap, i.e. does
ANY achievable drafter occupy the upper-left corner (cheap verify AND acceptance
>= A_ship that clears the 375.857 ship)?

This card REUSES fern #573's proven serve harness + exact-verify ngram simulator +
acceptance->TPS energy model (imported, not duplicated) and ADDS:
  1. ngram sweep over num_speculative_tokens K in {3,5,7,10} x prompt_lookup_max
     n in {2,3,4} (served at n=2 for realized TPS; n=3,4 acceptance exact-offline,
     verify cost is M-determined hence n-independent).
  2. the MTP arm at intermediate K (the "1-position MTP head" instruction-3 asks
     for) — /tmp/qat-assistant is a loadable autoregressive draft head, variable K,
     NO training. so only_ngram_loadable = False.
  3. the cost axis grounded on wirbel #575's directly-measured drafter-independent
     verify-cost curve C(M) (M=K+1), and A_ship(C) overlaid in BOTH the #573 anchor
     frame (PR-stated 2.6806 bar, for comparability) AND the clean realized frame
     (#575 corrected: 252.69 is MTP-K7-served, true no-spec = 1/C(1) = 87 TPS).

LOCAL A10G, analysis_only, NO HF Job / --launch / submission / served-file change.
A clear-the-ship point ANYWHERE is an ESCALATION (approval issue), never an auto-fire.

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/specdec_achievable_pareto/pareto_driver.py --smoke --no-wandb
Full:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/specdec_achievable_pareto/pareto_driver.py \
    --num-prompts 64 --output-len 320 --ngram-ks 3,5,7,10 --mtp-ks 3,5 \
    --wandb_name lawine/specdec-achievable-pareto --wandb_group base-fullhead-specdec-pareto
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
NG573 = ROOT / "research" / "base_fullhead_specdec" / "ngram_acceptance_model.py"
MEASURE_FLOOR = ROOT / "research" / "base_int4_floor_tps" / "measure_floor.py"

# --- import fern #573's proven harness (serve, exact-verify sim, energy model) ---
_spec = importlib.util.spec_from_file_location("ng573", str(NG573))
ng = importlib.util.module_from_spec(_spec)
sys.modules["ng573"] = ng
assert _spec and _spec.loader
_spec.loader.exec_module(ng)
# redirect #573's server logs + decode pass files into OUR dir
ng.OUT_ROOT = HERE

# lawine's OWN stock qat-w4a16-ct snapshot (same hash as fern's; self-contained)
MODEL_DIR = ("/senpai-run/home/student-lawine/.cache/huggingface/hub/"
             "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
             "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
MTP_MODEL = "/tmp/qat-assistant"  # loadable Gemma4AssistantForCausalLM draft head (#572)

# ---- wirbel #575 directly-measured verify-cost curve C(M) ms (drafter-independent, M=K+1) ----
C_MEASURED = {1: 11.4960, 2: 12.9621, 3: 13.0065, 5: 13.1367, 8: 13.1369, 9: 13.3716, 17: 14.2623}
C_FIT_INTERCEPT = 12.704375493740388
C_FIT_SLOPE = 0.08462453266874165
TRUE_C1_MS = C_MEASURED[1]                # 11.496 ms (true no-spec step, full 262k head)
VERIFY_K7_MTP_STEP_MS = 11.740686208407   # #575 graph-captured MTP-K7 step (verify+draft)


def c_of_m(m: int) -> float:
    """#575 verify cost at M positions: measured where available, else the M>=2 fit."""
    if m in C_MEASURED:
        return C_MEASURED[m]
    return C_FIT_INTERCEPT + C_FIT_SLOPE * m


# ---- anchors (cited, not re-derived) ----
TAU_LO = ng.TAU_LO                                  # 1.035236 local->official (#267)
SHIP_TPS = ng.SHIP_TPS                              # 375.857 official ship
CAPSTONE_FLOOR = ng.CAPSTONE_FLOOR                  # 311.25 magically-free floor
GATE_500 = 500.0
ANCHOR_252 = ng.ANCHOR_BASE_FULLHEAD_NOSPEC         # 252.69 (#553)
ANCHOR_252_OFFICIAL = ANCHOR_252 * TAU_LO           # 261.59 (#573 frame anchor)
SHIP_LOCAL = SHIP_TPS / TAU_LO                       # 363.08 ship in local warm-median space
TRUE_NOSPEC_LOCAL = 87.18337136776125               # #575 nospec_warm_median_tps (true no-spec)
TRUE_NOSPEC_OFFICIAL = TRUE_NOSPEC_LOCAL * TAU_LO    # ~90.25 (clean-frame no-spec official)
A_SHIP_573 = 2.6805993589363384                     # PR-stated break-even (fern #573 frame)
MTP_K7_ACCEPT_572 = 3.8443                           # lawine #572 measured MTP-K7 acceptance
MTP_K7_LOCAL_572 = 253.99                            # lawine #572 MTP-K7 served local TPS
MTP_K7_PROJ_572 = 262.94                             # #572 MTP-K7 proj (local*TAU_LO)
NGRAM_K7_PROJ_573 = 285.09                           # #573 ngram-K7 anchor-proj


def mtp_spec_config(k: int) -> str:
    return json.dumps({"method": "mtp", "model": MTP_MODEL, "num_speculative_tokens": k})


def _f(x: Any) -> float:
    return ng._f(x)


# ========================================================================== #
# A_ship break-even curves (acceptance needed to clear a TPS bar at verify cost C)
# ========================================================================== #
def a_ship_clean(c_ms: float, bar_official: float = SHIP_TPS) -> float:
    """CLEAN realized-frame break-even: an always-drafting drafter (coverage~1) runs at
    A / (C_ms/1000) local TPS. To clear an OFFICIAL bar it must reach bar/TAU_LO local,
    so A_ship = (bar/TAU_LO) * C_ms/1000.  (#575-grounded; true-physics)."""
    return (bar_official / TAU_LO) * c_ms / 1000.0


def a_ship_573(c_ms: float, bar_official: float = SHIP_TPS) -> float:
    """#573-frame break-even, recomputed on the #575 verify cost C(M): A_ship = c*(bar/anchor),
    c = C/C(1), anchor = 252.69*TAU_LO. NOTE: #573's published 2.6806 used an energy-model
    t_v (~22.4 ms) ~1.7x the #575 direct C(8)=13.14 ms; reported here for frame continuity."""
    c = c_ms / TRUE_C1_MS
    return c * (bar_official / ANCHOR_252_OFFICIAL)


# ========================================================================== #
# served-pass helpers
# ========================================================================== #
def _warm_median(passobj: dict) -> float:
    return _f((passobj.get("tps") or {}).get("warm_median_tps"))


def _warm_agg(passobj: dict) -> float:
    return _f((passobj.get("tps") or {}).get("warm_aggregate_tps"))


def serve_one(mf, harness, paths, serve_profile, *, server_python, label, spec_config,
              num_prompts, output_len, port, request_timeout_s, gpu_mem_util,
              log_stats=False, n_decodes=1, relax_loopgraph=False) -> dict:
    env = ng.build_env(spec_config=spec_config, model_dir=MODEL_DIR,
                       relax_loopgraph=relax_loopgraph, gpu_mem_util=gpu_mem_util,
                       log_stats=log_stats)
    return ng.run_pass(mf, harness, paths, serve_profile, server_python=server_python,
                       label=label, extra_env=env, num_prompts=num_prompts,
                       output_len=output_len, port=port, request_timeout_s=request_timeout_s,
                       n_decodes=n_decodes)


# ========================================================================== #
# synthesis: place every point on the (verify-cost, acceptance) plane
# ========================================================================== #
def build_pareto(ref: dict, ref_rows: list, ngram_served: dict, mtp_served: dict,
                 ngram_ks: list, mtp_ks: list, n_list: list, warm: int) -> dict:
    ref_local = _warm_median(ref)
    t1_local_ms = (1000.0 / ref_local) if ref_local > 0 else float("nan")

    points: list[dict] = []

    # ----- ngram arm: acceptance exact-offline for ALL (n,K); realized TPS served at n=2 -----
    ngram_grid = {}
    for n in n_list:
        for k in ngram_ks:
            sim = ng.offline_sim_pass(ref_rows, n, ng.NGRAM_LOOKUP_MIN, k, warm=warm)
            ngram_grid[(n, k)] = sim
    ngram_max_acc = max(_f(s["e_accept"]) for s in ngram_grid.values())

    for n in n_list:
        for k in ngram_ks:
            sim = ngram_grid[(n, k)]
            m = k + 1
            c_ms = c_of_m(m)
            e_acc = _f(sim["e_accept"])
            cov = _f(sim["coverage"])
            served = ngram_served.get(k) if n == ng.NGRAM_LOOKUP_MAX else None
            served_local = _warm_median(served) if served else float("nan")
            # realized TPS:
            #  - n=2 served directly (stats-OFF warm-median, comparable)
            #  - n=3,4 reconstructed via energy identity with the SAME verify cost C(M)
            #    (M-determined => n-independent) and the n=2 served t_1/coverage shift.
            if math.isfinite(served_local):
                realized_local = served_local
                realized_src = "served"
            else:
                # energy reconstruction: per-step avg tokens / avg time, coverage from offline sim
                # draft step cost = C(M); no-draft step cost = C(1)=t1_local_ms
                avg_tok = cov * e_acc + (1.0 - cov) * 1.0
                avg_ms = cov * c_ms + (1.0 - cov) * t1_local_ms
                realized_local = 1000.0 * avg_tok / avg_ms if avg_ms > 0 else float("nan")
                realized_src = "energy_reconstructed"
            speedup = (realized_local / ref_local) if ref_local > 0 else float("nan")
            points.append({
                "drafter": "ngram", "n": n, "k": k, "M": m,
                "e_accept": e_acc, "coverage": cov,
                "verify_cost_ms": c_ms,
                "realized_local_tps": realized_local, "realized_src": realized_src,
                "speedup_over_ref": speedup,
                "proj_tps_573frame": ANCHOR_252_OFFICIAL * speedup if math.isfinite(speedup) else float("nan"),
                "proj_tps_clean": realized_local * TAU_LO if math.isfinite(realized_local) else float("nan"),
                "a_ship_clean": a_ship_clean(c_ms),
                "a_ship_573": a_ship_573(c_ms),
                "clears_ship_clean": (realized_local * TAU_LO) > SHIP_TPS if math.isfinite(realized_local) else None,
                "accept_ge_268": e_acc >= A_SHIP_573,
            })

    # ----- MTP arm: acceptance served (stats-ON); realized TPS served stats-OFF where present -----
    for k in mtp_ks:
        rec = mtp_served.get(k, {})
        e_acc = _f(rec.get("e_accept"))
        cov = _f(rec.get("coverage"))
        realized_local = _f(rec.get("realized_local_tps"))
        m = k + 1
        c_ms = c_of_m(m)
        # if realized TPS not separately served, reconstruct from acceptance + MTP step cost.
        if not math.isfinite(realized_local) and math.isfinite(e_acc):
            # MTP drafts every step (coverage~1): TPS ~ e_accept / step_cost. step cost ~ C(M)
            # (eager verify upper bound; #575 graph MTP-K7 step is ~1.4 ms cheaper).
            realized_local = 1000.0 * e_acc / c_ms
        speedup = (realized_local / ref_local) if (ref_local > 0 and math.isfinite(realized_local)) else float("nan")
        points.append({
            "drafter": "mtp", "n": None, "k": k, "M": m,
            "e_accept": e_acc, "coverage": cov,
            "verify_cost_ms": c_ms,
            "realized_local_tps": realized_local, "realized_src": rec.get("realized_src", "reconstructed"),
            "speedup_over_ref": speedup,
            "proj_tps_573frame": ANCHOR_252_OFFICIAL * speedup if math.isfinite(speedup) else float("nan"),
            "proj_tps_clean": realized_local * TAU_LO if math.isfinite(realized_local) else float("nan"),
            "a_ship_clean": a_ship_clean(c_ms),
            "a_ship_573": a_ship_573(c_ms),
            "clears_ship_clean": (realized_local * TAU_LO) > SHIP_TPS if math.isfinite(realized_local) else None,
            "accept_ge_268": e_acc >= A_SHIP_573 if math.isfinite(e_acc) else None,
        })

    # ----- known corner: MTP-K7 (#572), used in the plane + the upper-left test -----
    corner_mtp_k7 = {
        "drafter": "mtp_corner_572", "n": None, "k": 7, "M": 8,
        "e_accept": MTP_K7_ACCEPT_572, "coverage": 1.0,
        "verify_cost_ms": c_of_m(8),
        "realized_local_tps": MTP_K7_LOCAL_572, "realized_src": "pr572",
        "speedup_over_ref": MTP_K7_LOCAL_572 / ref_local if ref_local > 0 else float("nan"),
        "proj_tps_573frame": ANCHOR_252_OFFICIAL * (MTP_K7_LOCAL_572 / ref_local) if ref_local > 0 else float("nan"),
        "proj_tps_clean": MTP_K7_PROJ_572,
        "a_ship_clean": a_ship_clean(c_of_m(8)),
        "a_ship_573": a_ship_573(c_of_m(8)),
        "clears_ship_clean": MTP_K7_PROJ_572 > SHIP_TPS,
        "accept_ge_268": MTP_K7_ACCEPT_572 >= A_SHIP_573,
    }
    points.append(corner_mtp_k7)

    # ----- verdicts -----
    # The HONEST projection is the CLEAN frame (realized_local * TAU_LO). The #573 frame
    # (ANCHOR_252_OFFICIAL * speedup) is ANCHOR-INFLATED: #575 established 252.69 is the
    # MTP-K7-SERVED number, not a no-spec baseline (anchor_252_is_mtp_not_nospec=True), so
    # multiplying it by the no-spec->spec speedup DOUBLE-COUNTS the spec benefit (it puts
    # MTP at 654-767 official, past the 500 gate -- physically impossible at M=1 on a memory
    # -bound per-step weight load). The 573-frame numbers are retained ONLY as a labeled
    # diagnostic for continuity with #573's published projection; they NEVER drive a verdict.
    ngram_points = [p for p in points if p["drafter"] == "ngram"]
    best_ngram = max(ngram_points, key=lambda p: p["proj_tps_clean"] if math.isfinite(p["proj_tps_clean"]) else -1)
    best_ngram_proj_573 = best_ngram["proj_tps_573frame"]
    best_ngram_proj_clean = max((p["proj_tps_clean"] for p in ngram_points if math.isfinite(p["proj_tps_clean"])), default=float("nan"))
    # served-only ngram headline (n=2 directly served; n=3,4 are energy-reconstructed upper bounds)
    ngram_served_pts = [p for p in ngram_points if p.get("realized_src") == "served"]
    best_ngram_proj_clean_served = max((p["proj_tps_clean"] for p in ngram_served_pts
                                        if math.isfinite(p["proj_tps_clean"])), default=float("nan"))

    # "ngram cost" reference = the verify cost of the served ngram configs (M=K+1, flat ~13 ms)
    ngram_cost_ref = max(p["verify_cost_ms"] for p in ngram_points)  # most generous (highest K)

    # literal upper-left (PR wording): any measured point with verify_cost <= ngram-cost AND acceptance >= 2.6806
    literal_corner_pts = [p for p in points
                          if math.isfinite(p["verify_cost_ms"]) and math.isfinite(_f(p["e_accept"]))
                          and p["verify_cost_ms"] <= ngram_cost_ref + 1e-9 and _f(p["e_accept"]) >= A_SHIP_573]
    upper_left_literal = len(literal_corner_pts) > 0

    # operational corner: any measured drafter whose REALIZED projection clears the ship
    clears = [p for p in points if p.get("clears_ship_clean") is True]
    any_clears_clean = len(clears) > 0
    clears_573 = [p for p in points if math.isfinite(p["proj_tps_573frame"]) and p["proj_tps_573frame"] > SHIP_TPS]
    any_clears_573 = len(clears_573) > 0

    return {
        "points": points,
        "ngram_grid_offline": {f"n{n}_k{k}": {"e_accept": _f(ngram_grid[(n, k)]["e_accept"]),
                                              "coverage": _f(ngram_grid[(n, k)]["coverage"])}
                               for n in n_list for k in ngram_ks},
        # --- KEY OUTPUTS ---
        "ngram_max_acceptance": ngram_max_acc,
        "ngram_clears_268": bool(ngram_max_acc >= A_SHIP_573),
        # HEADLINE projection = CLEAN frame (honest). 573-frame retained as labeled diagnostic.
        "best_ngram_projected_tps": best_ngram_proj_clean,
        "best_ngram_projected_tps_clean": best_ngram_proj_clean,
        "best_ngram_projected_tps_clean_served_only": best_ngram_proj_clean_served,
        "best_ngram_projected_tps_573frame_INFLATED": best_ngram_proj_573,
        "best_ngram_config": {"n": best_ngram["n"], "k": best_ngram["k"]},
        # HEADLINE verdict = CLEAN frame only. Do NOT OR in the anchor-inflated 573 frame.
        "any_measured_drafter_clears_ship": bool(any_clears_clean),
        "any_measured_drafter_clears_ship_clean_frame": bool(any_clears_clean),
        "any_measured_drafter_clears_ship_573frame_INFLATED": bool(any_clears_573),
        "upper_left_corner_occupied": bool(any_clears_clean),  # operational: clears the ship
        "upper_left_corner_literal_2_68_bar": bool(upper_left_literal),  # PR literal wording
        "upper_left_literal_points": [{"drafter": p["drafter"], "k": p["k"], "e_accept": _f(p["e_accept"]),
                                       "verify_cost_ms": p["verify_cost_ms"]} for p in literal_corner_pts],
        "only_ngram_loadable": False,  # MTP head /tmp/qat-assistant loads w/o training
        "ngram_cost_ref_ms": ngram_cost_ref,
        # frame reconciliation flags (#575)
        "anchor_252_is_mtp_not_nospec": True,
        "a_ship_clean_at_ngram_cost": a_ship_clean(ngram_cost_ref),
        "a_ship_573_stated_bar": A_SHIP_573,
        "true_nospec_local_tps": TRUE_NOSPEC_LOCAL,
        "ref_no_spec_local_tps": ref_local,
        "ship_tps": SHIP_TPS, "ship_local": SHIP_LOCAL,
        "tau_lo": TAU_LO,
        "analysis_only": True, "official_tps": 0,
    }


def _print_pareto(syn: dict) -> None:
    print("\n" + "=" * 12 + " PR #584 — SPEC-DEC ACHIEVABLE PARETO " + "=" * 12, flush=True)
    print(f"  REF no-spec LOCAL = {syn['ref_no_spec_local_tps']:.2f} TPS  (true no-spec #575 = {TRUE_NOSPEC_LOCAL:.2f})", flush=True)
    print(f"  {'drafter':16s} {'K':>3s} {'M':>3s} {'e_accept':>8s} {'cov':>5s} {'C(M)ms':>7s} "
          f"{'real_loc':>8s} {'proj573':>8s} {'projcln':>8s} {'A_ship*':>7s} {'clr?':>4s}", flush=True)
    for p in syn["points"]:
        print(f"  {p['drafter']:16s} {p['k']!s:>3s} {p['M']!s:>3s} {_f(p['e_accept']):>8.3f} "
              f"{_f(p['coverage']):>5.2f} {p['verify_cost_ms']:>7.2f} {_f(p['realized_local_tps']):>8.1f} "
              f"{_f(p['proj_tps_573frame']):>8.1f} {_f(p['proj_tps_clean']):>8.1f} "
              f"{_f(p['a_ship_clean']):>7.2f} {str(p.get('clears_ship_clean')):>4s}", flush=True)
    print(f"  ---", flush=True)
    print(f"  ngram_max_acceptance      = {syn['ngram_max_acceptance']:.4f}  (clears 2.6806? {syn['ngram_clears_268']})", flush=True)
    print(f"  best_ngram_projected_tps  = {syn['best_ngram_projected_tps']:.2f} (CLEAN headline) "
          f"[573-frame INFLATED {syn['best_ngram_projected_tps_573frame_INFLATED']:.2f}]  ship={SHIP_TPS}", flush=True)
    print(f"  any_drafter_clears_ship   = {syn['any_measured_drafter_clears_ship']} (CLEAN headline) "
          f"[573-frame INFLATED {syn['any_measured_drafter_clears_ship_573frame_INFLATED']}]", flush=True)
    print(f"  upper_left (clears ship)  = {syn['upper_left_corner_occupied']}", flush=True)
    print(f"  upper_left (literal 2.68) = {syn['upper_left_corner_literal_2_68_bar']}  "
          f"pts={syn['upper_left_literal_points']}", flush=True)
    print(f"  A_ship clean@ngram-cost   = {syn['a_ship_clean_at_ngram_cost']:.3f}  (vs #573 stated bar {A_SHIP_573:.4f})", flush=True)
    print(f"  only_ngram_loadable       = {syn['only_ngram_loadable']}", flush=True)
    print("=" * 62 + "\n", flush=True)


# ========================================================================== #
# wandb
# ========================================================================== #
def log_wandb(report: dict, args: argparse.Namespace):
    try:
        import wandb
        if not hasattr(wandb, "init"):
            print("[pareto] wandb namespace shadow; run under target/.venv. Skipping W&B (report saved).", flush=True)
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"[pareto] wandb import failed: {exc}; skipping (report saved).", flush=True)
        return None
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
        run = init_wandb_run(
            job_type="systems-profile", agent="lawine",
            name=args.wandb_name or "lawine/specdec-achievable-pareto",
            group=args.wandb_group or "base-fullhead-specdec-pareto",
            tags=["spec-dec", "ngram", "mtp", "prompt-lookup", "base-fullhead",
                  "achievable-pareto", "acceptance-model", "local-a10g", "analysis-only", "pr584"],
            notes="PR #584: empirical intermediate Pareto points (ngram K-sweep + MTP arm) on the "
                  "cost-acceptance plane; A_ship(C) overlay; does any achievable drafter occupy the "
                  "upper-left corner that clears the 375.857 ship?",
            config={"num_prompts": args.num_prompts, "output_len": args.output_len, "concurrency": 1,
                    "ngram_ks": args.ngram_ks, "mtp_ks": args.mtp_ks, "model_dir": MODEL_DIR,
                    "mtp_model": MTP_MODEL},
        )
        if run is None:
            return None
        syn = report["synthesis"]
        summary = {k: v for k, v in syn.items()
                   if isinstance(v, (int, float, bool)) and (not isinstance(v, float) or math.isfinite(v))}
        summary["primary_metric"] = syn["ngram_max_acceptance"]
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="specdec-achievable-pareto-report", artifact_type="specdec-report", data=report)
        rid = getattr(run, "id", None)
        finish_wandb(run)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[pareto] wandb logging failed: {exc}; skipping (report saved).", flush=True)
        return None


# ========================================================================== #
# main
# ========================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=320)
    ap.add_argument("--ngram-ks", default="3,5,7,10")
    ap.add_argument("--mtp-ks", default="3,5")
    ap.add_argument("--n-list", default="2,3,4", help="prompt_lookup_max depths for the offline acceptance grid")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=900)
    ap.add_argument("--gpu-mem-util", default="0.90")
    ap.add_argument("--warm", type=int, default=None)
    ap.add_argument("--skip-mtp", action="store_true")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    mf_spec = importlib.util.spec_from_file_location("measure_floor", str(MEASURE_FLOOR))
    mf = importlib.util.module_from_spec(mf_spec)
    assert mf_spec and mf_spec.loader
    mf_spec.loader.exec_module(mf)
    from scripts.local_validation import harness, paths, serve_profile

    for note in paths.prepare_local_gpu_env():
        print(f"[pareto] {note}", flush=True)

    manifest = harness.load_manifest(ng.SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    warm = args.warm if args.warm is not None else mf.WARMUP_REQUESTS
    HERE.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    ngram_ks = [int(x) for x in str(args.ngram_ks).split(",") if x.strip()]
    mtp_ks = [int(x) for x in str(args.mtp_ks).split(",") if x.strip()]
    n_list = [int(x) for x in str(args.n_list).split(",") if x.strip()]

    if args.smoke:
        np_, ol_ = 4, 48
        ref = serve_one(mf, harness, paths, serve_profile, server_python=server_python, label="smoke_ref",
                        spec_config="", num_prompts=np_, output_len=ol_, port=args.port,
                        request_timeout_s=args.request_timeout_s, gpu_mem_util=args.gpu_mem_util, n_decodes=1)
        ref_rows = ref["per_request"]
        ngm = serve_one(mf, harness, paths, serve_profile, server_python=server_python, label="smoke_ngram_k5",
                        spec_config=ng.ngram_spec_config(5, lookup_max=2), num_prompts=np_, output_len=ol_,
                        port=args.port, request_timeout_s=args.request_timeout_s, gpu_mem_util=args.gpu_mem_util)
        sim = ng.offline_sim_pass(ref_rows, 2, ng.NGRAM_LOOKUP_MIN, 5, warm=0)
        print(f"[pareto] SMOKE ref={_warm_median(ref):.2f} ngram={_warm_median(ngm):.2f} "
              f"sim_e_accept={sim['e_accept']:.3f} cov={sim['coverage']:.3f}", flush=True)
        mtp = serve_one(mf, harness, paths, serve_profile, server_python=server_python, label="smoke_mtp_k3",
                        spec_config=mtp_spec_config(3), num_prompts=np_, output_len=ol_, port=args.port,
                        request_timeout_s=args.request_timeout_s, gpu_mem_util=args.gpu_mem_util, log_stats=True)
        print(f"[pareto] SMOKE mtp_k3 tps={_warm_median(mtp):.2f} accept={mtp.get('acceptance')}", flush=True)
        ok = _warm_median(ref) > 0 and _warm_median(ngm) > 0 and _warm_median(mtp) > 0
        print(f"[pareto] SMOKE {'PASS' if ok else 'CHECK'} ({time.time()-t_start:.0f}s)", flush=True)
        return 0 if ok else 1

    # ---- REF (no-spec), two decodes: oracle + base self-determinism ----
    ref = serve_one(mf, harness, paths, serve_profile, server_python=server_python, label="ref",
                    spec_config="", num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                    request_timeout_s=args.request_timeout_s, gpu_mem_util=args.gpu_mem_util, n_decodes=2)
    ref_rows = ref["decodes"][0]["per_request"]
    self_det = ng.compare_identity(ref["decodes"][0]["per_request"], ref["decodes"][1]["per_request"])
    print(f"[pareto] REF local={_warm_median(ref):.2f} self_det seq={self_det['sequence_exact_rate']} "
          f"tok={self_det['token_identity_rate']} ({time.time()-t_start:.0f}s)", flush=True)

    report = {
        "pr": 584, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "num_prompts": args.num_prompts, "output_len": args.output_len,
        "ngram_ks": ngram_ks, "mtp_ks": mtp_ks, "n_list": n_list, "warm_discarded": warm,
        "model_dir": MODEL_DIR, "mtp_model": MTP_MODEL,
        "ref_local_tps": _warm_median(ref), "ref_self_determinism": self_det,
        "ngram_served": {}, "mtp_served": {}, "ngram_identity": {},
    }

    def _save():
        (HERE / "pareto_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    # ---- ngram K-sweep @ n=2 (stats-OFF: comparable realized TPS) ----
    ngram_served: dict[int, dict] = {}
    for k in ngram_ks:
        try:
            ngp = serve_one(mf, harness, paths, serve_profile, server_python=server_python,
                            label=f"ngram_k{k}", spec_config=ng.ngram_spec_config(k, lookup_max=ng.NGRAM_LOOKUP_MAX),
                            num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                            request_timeout_s=args.request_timeout_s, gpu_mem_util=args.gpu_mem_util)
            ngp["k"] = k
            ngram_served[k] = ngp
            ident = ng.compare_identity(ref_rows, ngp["per_request"])
            report["ngram_identity"][str(k)] = ident
            report["ngram_served"][str(k)] = {"k": k, "tps": ngp.get("tps"),
                                              "peak_vram_gb": ngp.get("peak_vram_gb"),
                                              "acceptance": ngp.get("acceptance")}
            print(f"[pareto] ngram K={k} local={_warm_median(ngp):.2f} "
                  f"ident_seq={ident['sequence_exact_rate']} tok={ident['token_identity_rate']} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
            _save()
        except Exception as exc:  # noqa: BLE001
            print(f"[pareto] ngram K={k} FAILED: {exc!r}; continuing", flush=True)

    # ---- MTP arm at intermediate K: stats-ON acceptance + stats-OFF realized TPS ----
    mtp_served: dict[int, dict] = {}
    if not args.skip_mtp:
        for k in mtp_ks:
            rec: dict[str, Any] = {"k": k}
            try:
                # stats-OFF: realized TPS (comparable)
                mt_off = serve_one(mf, harness, paths, serve_profile, server_python=server_python,
                                   label=f"mtp_k{k}_tps", spec_config=mtp_spec_config(k),
                                   num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                                   request_timeout_s=args.request_timeout_s, gpu_mem_util=args.gpu_mem_util)
                rec["realized_local_tps"] = _warm_median(mt_off)
                rec["realized_src"] = "served_statsoff"
                rec["peak_vram_gb"] = mt_off.get("peak_vram_gb")
                # stats-ON tiny: acceptance
                mt_on = serve_one(mf, harness, paths, serve_profile, server_python=server_python,
                                  label=f"mtp_k{k}_acc", spec_config=mtp_spec_config(k),
                                  num_prompts=min(24, args.num_prompts), output_len=min(224, args.output_len),
                                  port=args.port, request_timeout_s=args.request_timeout_s,
                                  gpu_mem_util=args.gpu_mem_util, log_stats=True)
                acc = mt_on.get("acceptance") or {}
                rec["e_accept"] = _f(acc.get("e_accept"))
                rec["acceptance_rate"] = _f(acc.get("acceptance_rate"))
                rec["coverage"] = 1.0  # MTP drafts every step
                rec["acceptance_raw"] = acc
                mtp_served[k] = rec
                report["mtp_served"][str(k)] = rec
                print(f"[pareto] MTP K={k} tps_local={_f(rec.get('realized_local_tps')):.2f} "
                      f"e_accept={_f(rec.get('e_accept')):.3f} acc_src={acc.get('source')} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
                _save()
            except Exception as exc:  # noqa: BLE001
                print(f"[pareto] MTP K={k} FAILED: {exc!r}; continuing", flush=True)

    # ---- synthesis ----
    syn = build_pareto(ref, ref_rows, ngram_served, mtp_served, ngram_ks, mtp_ks, n_list, warm)
    syn["ref_self_determinism_seq"] = _f(self_det.get("sequence_exact_rate"))
    syn["ref_self_determinism_tok"] = _f(self_det.get("token_identity_rate"))
    syn["peak_vram_gb"] = max([_f(ref.get("peak_vram_gb"))]
                              + [_f(v.get("peak_vram_gb")) for v in ngram_served.values()]
                              + [_f(v.get("peak_vram_gb")) for v in mtp_served.values()] or [0.0])
    report["synthesis"] = syn
    report["elapsed_s"] = time.time() - t_start
    _save()
    _print_pareto(syn)
    print(f"[pareto] report -> {HERE/'pareto_report.json'} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            _save()
            print(f"[pareto] wandb run id={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
