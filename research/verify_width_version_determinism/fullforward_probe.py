#!/usr/bin/env python
"""PR #681 lawine -- full-forward (served-spec) BODY leg of verify-width version determinism.

The lm_head arm (width_probe.py -> width_results_*.json) shows the int4-Marlin lm_head GEMM is
width-INVARIANT (bit-identical M=1 vs M=6) on BOTH vLLM versions. So kanna #673's verify-width
strict-#319 break is NOT in the head GEMM -- it is BODY-driven (attention + body GEMMs run at the
verify width). This leg tests whether that BODY break reproduces on the SHIP vLLM 0.22.0
(run-to-run deterministic, floor-clean per #601/#675) at the SAME verify width as the head probe
(K=5 -> M=6).

It reuses wirbel #607's COMMITTED census machinery (research/specdec_verify_identity_census/
census_verify.py: run_arm / verify / chaos_crosstab / synthesize), overriding only:
  * MODEL_DIR  -> the locked submission body (submissions/int4_g128_lmhead/model), the exact
                 weights the head probe served, so head and body legs are weight-identical;
  * K          -> 5 (M = K+1 = 6 verify), matching the head probe's headline width;
  * HERE       -> this dir, so decode/log artifacts never collide with wirbel's K=7 census.

Arms (each its own server boot = cross-start):
  ref   : spec-OFF (SENPAI_REFERENCE_MODE=1)            -> plain M=1 AR greedy, boot #1
  ref2  : spec-OFF, separate boot                        -> ref-vs-ref2 = cross-start FLOOR
  cand  : spec-ON  NUM_SPECULATIVE_TOKENS=5 (M=6 verify) -> int4 + MTP-K5, boot #3

VERDICT (on the deterministic 0.22.0 gate, floor expected clean):
  floor clean (ref==ref2) AND cand diverges on STABLE prompts -> structural BODY break on ship
  -> verify-width sensitivity is VERSION-FUNDAMENTAL, NOT a dev307 numerics artifact.

Writes fullforward_report_<tag>.json with `<tag>_structural_break` / `<tag>_cand_seq_exact`
keys consumed by log_wandb.decide(). LOCAL analysis_only, single A10G, NO HF Job / no --launch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness  # noqa: E402

HERE = Path(__file__).resolve().parent

# import wirbel #607 census_verify by path (committed substrate) and override its config below.
_CV_PATH = ROOT / "research" / "specdec_verify_identity_census" / "census_verify.py"
_spec = importlib.util.spec_from_file_location("census_verify_pr607", _CV_PATH)
cv = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(cv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venv-python", required=True, help="serve venv python (e.g. 0.22.0 ship)")
    ap.add_argument("--tag", default="v0220")
    ap.add_argument("--k", type=int, default=5, help="num_speculative_tokens; M=K+1 verify width")
    ap.add_argument("--num-prompts", type=int, default=24)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--model-dir",
                    default=str(ROOT / "submissions" / "int4_g128_lmhead" / "model"))
    ap.add_argument("--arms", default="ref,ref2,cand")
    ap.add_argument("--base-port", type=int, default=8031)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 prompts x 64 tok wiring check")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts, args.output_len = 4, 64

    # ---- override census globals: locked body + matched width + our artifact dir ----
    cv.MODEL_DIR = args.model_dir
    cv.K = args.k
    # Isolate decode_{arm}_{p}/server_*.log per vLLM version: census_verify does NOT version-tag
    # these files, so a dev307 run in the shared dir would resume-reuse the v0220 captures and
    # silently mislabel them. One subdir per tag keeps the two versions byte-separate.
    decode_dir = HERE / f"_decode_{args.tag}"
    decode_dir.mkdir(parents=True, exist_ok=True)
    cv.HERE = decode_dir

    for note in cv.paths.prepare_local_gpu_env():
        print(f"[ff:{args.tag}] {note}", flush=True)

    server_python = Path(args.venv_python)
    vllm_ver = harness._dist_version(server_python, "vllm")
    print(f"[ff:{args.tag}] server_python={server_python} vllm={vllm_ver} "
          f"K={cv.K} M={cv.K + 1} model={cv.MODEL_DIR}", flush=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ports = {a: args.base_port + i for i, a in enumerate(arms)}
    passes = ["a", "b"]
    scored = "b"
    resume = not args.no_resume

    arm_results: dict[str, Any] = {}
    for arm in arms:
        if arm != arms[0]:
            free = cv.wait_gpu_free()
            print(f"[ff:{args.tag}] gpu free before {arm}: {free} MiB", flush=True)
        arm_results[arm] = cv.run_arm(
            server_python=server_python, arm=arm,
            num_prompts=args.num_prompts, output_len=args.output_len,
            port=ports[arm], passes=passes, probe_tps=(arm == "cand"), resume=resume,
        )

    def sf(a: str) -> str | None:
        return arm_results.get(a, {}).get("files", {}).get(scored)

    floor = cv.verify(sf("ref"), sf("ref2")) if sf("ref") and sf("ref2") else None
    spec = cv.verify(sf("ref"), sf("cand")) if sf("ref") and sf("cand") else None
    crosstab = cv.chaos_crosstab(floor, spec) if (floor and spec) else None
    within: dict[str, Any] = {}
    for arm in arms:
        fa = arm_results[arm]["files"].get("a")
        fb = arm_results[arm]["files"].get("b")
        if fa and fb and fa != fb:
            within[arm] = cv.verify(fa, fb)
    verdict = cv.synthesize(floor, spec, crosstab, within) if (floor and spec) else None

    tag = args.tag
    struct = bool(verdict and verdict.get("structural_break_above_floor"))
    report = {
        f"{tag}_structural_break": struct,
        f"{tag}_cand_seq_exact": spec["freerun_seq_exact"] if spec else None,
        f"{tag}_floor_seq_exact": floor["freerun_seq_exact"] if floor else None,
        f"{tag}_floor_clean": bool(verdict and verdict.get("cross_start_floor_clean")),
        f"{tag}_struct_prompts_on_stable":
            crosstab["cand_n_divergent_on_stable"] if crosstab else None,
        f"{tag}_struct_tokens_on_stable":
            crosstab["structural_divergent_tokens_on_stable"] if crosstab else None,
        f"{tag}_cand_warm_token_identity": spec["freerun_token_identity"] if spec else None,
        f"{tag}_onset_min": spec["onset_min"] if spec else None,
        f"{tag}_onset_median": spec["onset_median"] if spec else None,
        "config": {"k": cv.K, "verify_width": cv.K + 1, "model_dir": cv.MODEL_DIR,
                   "drafter": cv.DRAFTER, "vllm_version": vllm_ver,
                   "num_prompts": args.num_prompts, "output_len": args.output_len,
                   "smoke": args.smoke},
        "verdict": verdict, "floor": floor, "spec": spec,
        "crosstab": crosstab, "within": within,
        "peak_vram_gb": max((r.get("peak_vram_gb") or 0.0) for r in arm_results.values()),
    }
    out = HERE / (f"fullforward_report_{tag}.smoke.json" if args.smoke
                  else f"fullforward_report_{tag}.json")
    out.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 72, flush=True)
    print(f"[PR681 fullforward:{tag}] BODY leg ({'SMOKE' if args.smoke else 'FULL'})  "
          f"vLLM={vllm_ver}  K={cv.K} M={cv.K + 1}  {args.num_prompts}x{args.output_len}", flush=True)
    if floor:
        print(f"  CROSS-START FLOOR ref vs ref2 : {floor['verdict']}  "
              f"seq_exact={floor['freerun_seq_exact']}  "
              f"div_tok={floor['total_divergent_tokens']}/{floor['total_tokens_compared']}", flush=True)
    if spec:
        print(f"  SPEC VERDICT      ref vs cand : {spec['verdict']}  "
              f"seq_exact={spec['freerun_seq_exact']}  "
              f"div_tok={spec['total_divergent_tokens']}/{spec['total_tokens_compared']}  "
              f"onset(min/med)={spec['onset_min']}/{spec['onset_median']}", flush=True)
    if crosstab:
        print(f"  chaos cross-tab: floor stable={crosstab['n_stable_floor']} "
              f"chaotic={crosstab['n_chaotic_floor']}; cand divergent_on_STABLE="
              f"{crosstab['cand_n_divergent_on_stable']} "
              f"(struct tokens={crosstab['structural_divergent_tokens_on_stable']})", flush=True)
    if verdict:
        print(f"  >>> {tag}_structural_break = {struct}  "
              f"(census verdict: {verdict['pr607_verdict']})", flush=True)
    print(f"  report -> {out}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
