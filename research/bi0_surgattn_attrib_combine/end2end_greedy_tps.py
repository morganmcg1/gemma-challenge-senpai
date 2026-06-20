"""PR #794 end-to-end: bi0 force-2D (control) vs surgattn-OFF 3D (variant).

Serves the shipped bi0 submission twice on the local A10G via the official
local-validation harness, toggling ONLY the reconstructed VLLM_SURGATTN switch:

  - arm '2d' : VLLM_SURGATTN=1  -> force-2D, byte-identical greedy reference.
  - arm '3d' : VLLM_SURGATTN=0  -> M=1 forwards take vanilla 3D split-KV (the
               surgattn-OFF +6.69% arm wirbel #785 measured).

Per arm: 128-prompt greedy decode (-> decode_outputs.jsonl), PPL, and N
single-stream decode-TPS reps (discard rep0, report median + CV). Then runs the
official greedy-identity verifier comparing 3d-vs-2d and surfaces the
first-divergence ONSET distribution: late/stochastic onset == argmax-tie
reassociation noise; early/most-prompts onset == a lossy bug.

Uses the prebuilt vllm022 serve venv (manifest deps: vllm==0.22.0,
transformers==5.9.0) instead of building a fresh one. LOCAL ONLY.

    CUDA_VISIBLE_DEVICES=0 .venvs/.../python end2end_greedy_tps.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

SERVE_PY = Path("/senpai-run/home/student-stark/.venvs/vllm022/bin/python")
SUBMISSION = REPO / "submissions" / "int4_mtp_bi0_surgattn"
OUTDIR = Path(__file__).resolve().parent / "e2e"
ARMS = {"2d": "1", "3d": "0"}  # arm name -> VLLM_SURGATTN value


def run_arm(arm: str, surgattn: str, *, num_prompts: int, output_len: int,
            tps_reps: int, decode_tokens: int) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTDIR / f"server_{arm}.log"
    decode_jsonl = OUTDIR / f"decode_{arm}.jsonl"
    decode_summ = OUTDIR / f"decode_{arm}_summary.json"
    ppl_out = OUTDIR / f"ppl_{arm}.jsonl"
    ppl_summ = OUTDIR / f"ppl_{arm}_summary.json"

    # VLLM_USE_FLASHINFER_SAMPLER=0: the flashinfer sampling op JIT-compiles at
    # engine init and dies on this pod (curand.h not on the include path). Greedy
    # decode (temperature=0) short-circuits to target-argmax BEFORE any sampler
    # backend runs, so the native PyTorch sampler is token-identical here; both
    # arms share the setting, so the 2D-vs-3D delta is unaffected.
    extra_env = {"VLLM_SURGATTN": surgattn, "VLLM_BATCH_INVARIANT": "0",
                 "VLLM_USE_FLASHINFER_SAMPLER": "0",
                 "CUDA_VISIBLE_DEVICES": "0"}
    res: dict = {"arm": arm, "VLLM_SURGATTN": surgattn}
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVE_PY, port=8000,
        log_path=log_path, extra_env=extra_env,
    ) as srv:
        res["serve_ready_s"] = round(time.time() - t0, 1)
        # decode (greedy, 128 prompts)
        dsum = harness.capture_decode(
            SERVE_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_jsonl, summary_file=decode_summ,
            num_prompts=num_prompts, output_len=output_len, seed=paths.SEED,
        )
        res["decode_summary"] = dsum
        # ppl
        psum = harness.run_ppl(
            SERVE_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=ppl_out, summary_file=ppl_summ,
        )
        res["ppl_summary"] = psum
        # TPS reps (discard rep0)
        reps = []
        for i in range(tps_reps):
            p = harness.probe_tps(srv.base_url, srv.served_model_name,
                                  decode_tokens=decode_tokens)
            reps.append(p["decode_tps_single_stream"])
            print(f"  [tps {arm} rep{i}] {p['decode_tps_single_stream']:.2f} tps",
                  flush=True)
        res["tps_reps_raw"] = reps
        kept = reps[1:] if len(reps) > 1 else reps
        res["tps_reps_kept"] = kept
        if kept:
            med = statistics.median(kept)
            mean = statistics.mean(kept)
            sd = statistics.pstdev(kept) if len(kept) > 1 else 0.0
            res["tps_median"] = med
            res["tps_mean"] = mean
            res["tps_std"] = sd
            res["tps_cv"] = (sd / mean) if mean else float("nan")
    res["arm_wall_s"] = round(time.time() - t0, 1)
    # confirm the toggle actually took effect from the worker log
    log_txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
    res["log_force2d_wrapped"] = "forcing 2D single-pass" in log_txt
    res["log_surgattn_disabled"] = "VLLM_SURGATTN=0: force-2D DISABLED" in log_txt
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to de-risk the serve path (4 prompts, 2 reps)")
    args = ap.parse_args()
    if args.smoke:
        num_prompts, output_len, tps_reps, decode_tokens = 4, 32, 2, 32
    else:
        num_prompts, output_len, tps_reps, decode_tokens = (
            paths.NUM_PROMPTS, paths.OUTPUT_LEN, 6, 256)

    results = {"smoke": args.smoke, "num_prompts": num_prompts,
               "output_len": output_len, "arms": {}}
    for arm, surg in ARMS.items():
        print(f"\n===== ARM {arm} (VLLM_SURGATTN={surg}) =====", flush=True)
        results["arms"][arm] = run_arm(
            arm, surg, num_prompts=num_prompts, output_len=output_len,
            tps_reps=tps_reps, decode_tokens=decode_tokens)

    # greedy-identity comparison: 3d (candidate) vs 2d (reference)
    ref = OUTDIR / "decode_2d.jsonl"
    cand = OUTDIR / "decode_3d.jsonl"
    try:
        report = greedy_gate.compare(ref, cand)
        onset = greedy_gate.onset_summary(report)
        verdict = getattr(report, "verdict", None)
        results["greedy_compare"] = {
            "verdict": str(verdict),
            "num_identical": onset.get("num_identical"),
            "num_divergent": onset.get("num_divergent"),
            "onset_min": onset.get("onset_min"),
            "onset_median": onset.get("onset_median"),
            "onset_max": onset.get("onset_max"),
            "onsets": onset.get("onsets"),
            "output_len": output_len,
        }
    except Exception as e:  # noqa: BLE001
        results["greedy_compare"] = {"error": repr(e)}

    # TPS delta + 2-sigma check
    try:
        a2 = results["arms"]["2d"]; a3 = results["arms"]["3d"]
        med2, med3 = a2.get("tps_median"), a3.get("tps_median")
        sd2, sd3 = a2.get("tps_std", 0.0), a3.get("tps_std", 0.0)
        if med2 and med3:
            pooled = (sd2 ** 2 + sd3 ** 2) ** 0.5
            results["tps_delta"] = {
                "tps_2d_median": med2, "tps_3d_median": med3,
                "delta": med3 - med2,
                "pct": 100.0 * (med3 - med2) / med2,
                "pooled_sigma": pooled,
                "delta_in_sigma": ((med3 - med2) / pooled) if pooled else float("inf"),
            }
    except Exception as e:  # noqa: BLE001
        results["tps_delta"] = {"error": repr(e)}

    outpath = OUTDIR / ("results_smoke.json" if args.smoke else "results.json")
    outpath.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[e2e] wrote {outpath}", flush=True)
    gc = results.get("greedy_compare", {})
    td = results.get("tps_delta", {})
    print(f"[e2e] greedy: {gc}", flush=True)
    print(f"[e2e] tps_delta: {td}", flush=True)


if __name__ == "__main__":
    main()
