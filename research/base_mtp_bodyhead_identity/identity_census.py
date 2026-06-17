#!/usr/bin/env python
"""PR #600 denken — base_mtp greedy-identity census: isolate BODY vs HEAD as the int4+MTP
identity-breaker (LOCAL A10G, analysis-only, NO HF fire, NO served-file change).

THE QUESTION (fern #597's open complement): the int4+MTP spec-dec greedy-identity break — is it
caused by the int4 BODY (W4A16 Marlin GEMM, M-dependent tiling; shared by base_fullhead) or the
int4 HEAD (only int4_g128_lmhead int4-quantizes the untied lm_head)? fern's int4_g128 proxy has
BOTH. This card isolates the BODY by running the canonical operative-#319 greedy-identity census
on base_mtp = int4 W4A16 Marlin BODY + full BF16 262k untied HEAD + Linear MTP-K7. The HEAD is held
BF16, so any spec-ON-vs-spec-OFF divergence is attributable to the int4 BODY alone.

PREDICATE (wirbel #588, ZERO tolerance, byte-exact, official check_greedy_identity.py):
  warm steady-state free-running greedy decode, MAX_NUM_SEQS=1, temp=0, ignore_eos (full free-run
  -> #541 EOS-guard auto-satisfied), 128 x 512 sharegpt.
  Compare base_mtp spec-ON (MTP-K7, M=8 verify) served tokens vs the spec-OFF plain-AR greedy (M=1)
  of the SAME checkpoint.

SUBSTRATE (exactly the #596 base_mtp / base_specoff configs, reused infra from PR #576):
  spec-ON  = submissions/fa2sw_strict_surgical357 + base_fullhead overrides + MTP-K7 (arm_env "mtp")
  spec-OFF = SAME submission + SAME overrides + SENPAI_REFERENCE_MODE=1 -> M=1 AR (arm_env "ref")
  Both: int4 W4A16 Marlin body (compressed-tensors), full BF16 262144-row untied lm_head, prune OFF,
  VLLM_USE_FLASHINFER_SAMPLER=0 (PyTorch-native lowest-index argmax tie-break).

CLASSIFICATION (structural-lossy vs FP-noise): an on-stack flip-margin probe reads, on the spec-OFF
(M=1) server, the logprob margin between the spec-OFF argmax (ref_tok) and the spec-ON-emitted token
(cand_tok) at each first-divergence position. Near-zero margins / exact ties, disjoint BELOW the
control (matched-position) rank0-rank1 margins => bf16 near-tie reorder (FP-noise: the M=8 verify
reduces in a different order than M=1 and flips a near-tie). Large margins overlapping control =>
structural (lossy) divergence.

LOCAL only: analysis_only, official_tps=0, no HF Job, no --launch, no submission, no served-file
change.

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/base_mtp_bodyhead_identity/identity_census.py --smoke --no-wandb
Full:
  CUDA_VISIBLE_DEVICES=0 python research/base_mtp_bodyhead_identity/identity_census.py \
    --num-prompts 128 --output-len 512 \
    --wandb_name denken/base-mtp-bodyhead-identity --wandb_group base-mtp-bodyhead-identity
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CENSUS_DIR = ROOT / "research" / "specdec_identity_census"
for _p in (str(ROOT), str(CENSUS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import census_driver as C  # noqa: E402  reuse the base_fullhead serve + #319-verify infra

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE
SEED = C.SEED
SUBMISSION = C.SUBMISSION
MODEL_DIR = C.MODEL_DIR

# Cited anchors (PR #600 body / BASELINE — NOT re-derived).
OPERATIVE_AR_TPS = 126.378            # int4_g128_lmhead strict-identity AR rung (the bar to beat)
OPERATIVE_PPL = 2.019
ANCHOR_BASE_FULLHEAD_SPEC_TPS = 254.00  # base_fullhead+MTP-K7 spec-TPS (lawine #595 8grrygq0)
TAU_LOCAL_TO_OFFICIAL = 1.03524
# fern #597 int4_g128 (int4 BODY + int4 HEAD) proxy — the BOTH-quantized reference point:
FERN_INT4G128_FREERUN_SEQ_EXACT = 0.1875
# my own #576 base_fullhead cold-ref census (SAME substrate, COLD ref confound): seq 0.1875 cold.
# Near-tie threshold for an FP-noise classification of a flip's M=1 logprob margin. bf16 carries 8
# mantissa bits; at decode logit magnitudes (~5-20) one ULP is ~2^-4..2^-3 (~0.06-0.12), so <=4 ULP
# is ~<=0.25-0.5 in logit/logprob-margin space. We report several thresholds; the load-bearing
# classifier is the disjoint-distribution test (miss margins disjoint BELOW control margins).
NEARTIE_MARGIN_THRESHOLDS = [0.0, 0.0625, 0.125, 0.25, 0.5]


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    C._sample_vram(stop, peak)


def capture_arm(harness: Any, paths: Any, *, server_python: Path, arm: str, num_prompts: int,
                output_len: int, port: int, passes: int,
                probe_closure=None) -> dict[str, Any]:
    """Boot the base_mtp/base_specoff stack for one arm, capture `passes` greedy decode files
    (cold, warm[, warm2]). For the spec-OFF (ref) arm an optional `probe_closure(srv)` runs while
    the server is still live (the on-stack flip-margin probe needs the M=1 server)."""
    extra_env = C.arm_env(arm)
    log_path = OUT_ROOT / f"server_{arm}.log"
    tags = (["cold", "warm", "warm2"][:passes]) if passes <= 3 else \
        ["cold"] + [f"warm{i}" for i in range(passes - 1)]
    result: dict[str, Any] = {"arm": arm, "files": {}, "booted": False}

    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            result["booted"] = True
            result["boot_s"] = round(time.time() - t0, 1)
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            for t in tags:
                out_file = OUT_ROOT / f"decode_{arm}_{t}.jsonl"
                summary_file = OUT_ROOT / f"decode_{arm}_{t}.summary.json"
                print(f"[bmh] [{arm}] capture {t} {num_prompts}x{output_len} conc=1 "
                      f"-> {out_file.name}", flush=True)
                harness.capture_decode(
                    server_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_file, summary_file=summary_file,
                    num_prompts=num_prompts, output_len=output_len, seed=SEED,
                    tokenizer=paths.TOKENIZER, dataset=paths.EVAL_PROMPTS, timeout_s=5400)
                result["files"][t] = str(out_file)
            if probe_closure is not None:
                result["probe_result"] = probe_closure(srv)
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"[bmh] [{arm}] FAILED: {exc!r}", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=5)
    result["peak_vram_gb"] = round((peak["mib"] or 0.0) / 1024.0, 2)
    result["plumbing"] = C.grep_log(str(log_path), C.PLUMBING_NEEDLES + ["lmhead-full",
                                     "verified full lm_head", "PCK04_KEEPSET not set"])
    result["log_path"] = str(log_path)
    return result


def _flip_base() -> dict[str, Any]:
    return {
        "classification": None, "mechanism": None, "is_fp_noise": None, "is_structural": None,
        "n_miss_probed": 0, "n_exact_tie": 0, "frac_exact_tie": None,
        "miss_margin_median": None, "miss_margin_p90": None, "miss_margin_max": None,
        "control_margin_median": None, "control_margin_p10": None,
        "miss_disjoint_below_control": None, "separation_ratio_control_over_miss": None,
    }


def classify_flips(tie: dict | None, identity: dict) -> dict[str, Any]:
    """structural-lossy vs FP-noise from the flip-margin probe rows."""
    base = _flip_base()
    if not tie or not tie.get("rows"):
        # no divergences at all -> vacuously identity-safe
        if identity.get("num_divergent", 0) == 0:
            base.update({"classification": "identical", "mechanism": "none",
                         "is_fp_noise": True, "is_structural": False})
        else:
            base.update({"classification": "unprobed", "mechanism": "unknown"})
        return base
    rows = tie["rows"]
    miss = [r["margin"] for r in rows
            if r.get("kind") == "miss" and isinstance(r.get("margin"), (int, float))]
    ctrl = [r["margin"] for r in rows
            if r.get("kind") == "control" and isinstance(r.get("margin"), (int, float))]
    n_exact_tie = sum(1 for r in rows if r.get("exact_tie"))
    n_miss = len(miss)
    near = {f"frac_miss_abs_margin_le_{thr}":
            (sum(1 for m in miss if abs(m) <= thr) / n_miss if n_miss else None)
            for thr in NEARTIE_MARGIN_THRESHOLDS}
    miss_max = max((abs(m) for m in miss), default=None)
    miss_p90 = (statistics.quantiles([abs(m) for m in miss], n=10)[-1] if len(miss) >= 10 else miss_max)
    ctrl_med = statistics.median(ctrl) if ctrl else None
    ctrl_p10 = (statistics.quantiles(ctrl, n=10)[0] if len(ctrl) >= 10 else (min(ctrl) if ctrl else None))
    miss_med = statistics.median(miss) if miss else None
    # disjoint-distribution test (the #576/#566 robust near-tie signature): worst miss margin still
    # below the typical control rank0-rank1 separation -> misses are near-ties, NOT structural.
    disjoint_below_control = (miss_p90 is not None and ctrl_p10 is not None and miss_p90 <= ctrl_p10)
    frac_exact_tie = (n_exact_tie / n_miss) if n_miss else None
    # FP-noise verdict: misses cluster at/near 0 AND are disjoint below control.
    is_fp_noise = bool(
        n_miss > 0
        and (miss_med is not None and miss_med <= 0.25)
        and (disjoint_below_control or (frac_exact_tie is not None and frac_exact_tie >= 0.25))
    )
    is_structural = bool(n_miss > 0 and not is_fp_noise)
    return {
        "classification": ("fp_noise_near_tie_reorder" if is_fp_noise
                           else ("structural_lossy" if is_structural else "unresolved")),
        "mechanism": ("bf16 near-tie reorder via M-dependent int4 Marlin GEMM (M=8 verify vs M=1 AR)"
                      if is_fp_noise else
                      ("structural int4 logit shift (genuinely different argmax)" if is_structural
                       else "unresolved")),
        "is_fp_noise": is_fp_noise,
        "is_structural": is_structural,
        "n_miss_probed": n_miss,
        "n_exact_tie": n_exact_tie,
        "frac_exact_tie": frac_exact_tie,
        "miss_margin_median": miss_med,
        "miss_margin_p90": miss_p90,
        "miss_margin_max": miss_max,
        "control_margin_median": ctrl_med,
        "control_margin_p10": ctrl_p10,
        "miss_disjoint_below_control": disjoint_below_control,
        "separation_ratio_control_over_miss": (ctrl_med / miss_med if (ctrl_med and miss_med and miss_med > 0) else None),
        **near,
    }


def synthesize(identity: dict, self_det: dict | None, flip: dict, peak_vram: float,
               n_prompts: int, output_len: int) -> dict[str, Any]:
    seq_exact = identity["sequence_exact_rate"]
    per_step = identity["matched_state_per_step_identity_rate"]
    verdict = identity["verdict"]
    passes_319 = bool(seq_exact is not None and seq_exact >= 1.0 and verdict == "GREEDY_IDENTICAL")
    # THE body-vs-head verdict: does the int4 BODY ALONE (BF16 head held constant) break the
    # zero-tolerance #319 spec-vs-AR greedy identity?
    body_breaks_identity = bool(not passes_319)
    self_det_ok = (self_det is None) or (self_det.get("verdict") == "GREEDY_IDENTICAL")
    return {
        "freerun_seq_exact": seq_exact,                       # HEADLINE (#588 byte-exact rate)
        "verdict": verdict,
        "num_prompts_compared": identity["num_prompts_compared"],
        "num_identical": identity["num_identical"],
        "num_divergent": identity["num_divergent"],
        "token_id_match_rate_matched_state": per_step,        # per-step matched-state argmax agreement
        "token_id_match_rate_freerun": identity["freerun_positional_identity_rate"],
        "total_tokens_compared": identity["total_tokens_compared"],
        "total_divergent_tokens": identity["total_divergent_tokens"],
        "onset_min": identity["onset_min"],
        "onset_median": identity["onset_median"],
        "onset_max": identity["onset_max"],
        "matched_state_trials": identity["matched_state_trials"],
        "matched_state_failures": identity["matched_state_failures"],
        # spec-OFF reference warm self-determinism guard (the divergence must be spec-induced, not
        # reference noise). None if not captured (then cite #596 q2hm5rc3 warm-det=1.0).
        "specoff_warm_self_det_ok": self_det_ok,
        "specoff_warm_self_det_verdict": (self_det or {}).get("verdict"),
        "specoff_warm_self_det_seq_exact": (self_det or {}).get("sequence_exact_rate"),
        # classification
        "classification": flip["classification"],
        "mechanism": flip["mechanism"],
        "dissents_are_fp_noise_near_tie": flip["is_fp_noise"],
        "dissents_are_structural_lossy": flip["is_structural"],
        "flip_n_miss_probed": flip["n_miss_probed"],
        "flip_n_exact_tie": flip["n_exact_tie"],
        "flip_frac_exact_tie": flip["frac_exact_tie"],
        "flip_miss_margin_median": flip["miss_margin_median"],
        "flip_miss_margin_p90": flip["miss_margin_p90"],
        "flip_control_margin_median": flip["control_margin_median"],
        "flip_control_margin_p10": flip["control_margin_p10"],
        "flip_miss_disjoint_below_control": flip["miss_disjoint_below_control"],
        # VERDICTS
        "passes_319_byte_exact": passes_319,
        "int4_body_alone_breaks_spec_identity": body_breaks_identity,
        # context
        "fern_int4g128_freerun_seq_exact": FERN_INT4G128_FREERUN_SEQ_EXACT,
        "head_adds_divergence": (FERN_INT4G128_FREERUN_SEQ_EXACT < seq_exact) if seq_exact is not None else None,
        "operative_ar_tps": OPERATIVE_AR_TPS,
        "anchor_base_fullhead_spec_tps": ANCHOR_BASE_FULLHEAD_SPEC_TPS,
        "num_prompts": n_prompts,
        "output_len": output_len,
        "peak_vram_gb": peak_vram,
        "analysis_only": True,
        "official_tps": 0,
    }


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and not math.isfinite(x))


def _print_summary(s: dict[str, Any]) -> None:
    line = "=" * 8 + " PR #600 — base_mtp BODY-vs-HEAD greedy-identity census " + "=" * 8
    print("\n" + line, flush=True)
    print(f"  predicate = warm steady-state, MAX_NUM_SEQS=1, temp=0, {s['num_prompts']}x{s['output_len']}, "
          f"official check_greedy_identity, zero tolerance", flush=True)
    print(f"  >>> freerun_seq_exact         = {s['freerun_seq_exact']}  [{s['verdict']}]  "
          f"({s['num_identical']}/{s['num_prompts_compared']})", flush=True)
    print(f"      matched-state per-step    = {s['token_id_match_rate_matched_state']}  "
          f"(fail {s['matched_state_failures']}/{s['matched_state_trials']})", flush=True)
    print(f"      onset min/median/max      = {s['onset_min']}/{s['onset_median']}/{s['onset_max']}", flush=True)
    print(f"      spec-OFF warm self-det    = {s['specoff_warm_self_det_ok']} "
          f"[{s['specoff_warm_self_det_verdict']}] seq={s['specoff_warm_self_det_seq_exact']}", flush=True)
    print(f"  >>> classification            = {s['classification']}", flush=True)
    print(f"      mechanism                 = {s['mechanism']}", flush=True)
    print(f"      miss margin med/p90       = {s['flip_miss_margin_median']}/{s['flip_miss_margin_p90']}  "
          f"control med/p10 = {s['flip_control_margin_median']}/{s['flip_control_margin_p10']}", flush=True)
    print(f"      exact-ties                = {s['flip_n_exact_tie']}/{s['flip_n_miss_probed']}  "
          f"disjoint-below-control={s['flip_miss_disjoint_below_control']}", flush=True)
    print(f"  >>> passes #319 byte-exact    = {s['passes_319_byte_exact']}", flush=True)
    print(f"  >>> int4 BODY ALONE breaks spec-identity = {s['int4_body_alone_breaks_spec_identity']}", flush=True)
    print(f"      (fern int4_g128 BOTH = {s['fern_int4g128_freerun_seq_exact']}; "
          f"head_adds_divergence={s['head_adds_divergence']})", flush=True)
    print(f"  peak VRAM = {s['peak_vram_gb']:.2f} GB", flush=True)
    print("=" * len(line) + "\n", flush=True)


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[bmh] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile", agent="denken",
        name=args.wandb_name or "denken/base-mtp-bodyhead-identity",
        group=args.wandb_group or "base-mtp-bodyhead-identity",
        tags=["specdec", "identity-census", "319", "served", "base-mtp", "body-vs-head",
              "int4-body", "bf16-head", "local-a10g", "analysis-only", "pr600"],
        notes="PR #600: base_mtp (int4 W4A16 Marlin BODY + full BF16 262k untied HEAD + MTP-K7) "
              "operative-#319 greedy-identity census — does the int4 BODY ALONE break spec-vs-AR "
              "identity (isolating BODY vs HEAD, complementing fern #597's int4_g128 which has both)?",
        config={
            "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
            "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": SEED,
            "concurrency": 1, "gpu_mem_util": C.GPU_MEM_UTIL,
            "spec_on": "mtp_k7_m8_verify", "spec_off": "reference_mode_m1_ar",
            "head": "full_bf16_262144", "body": "int4_w4a16_marlin",
            "operative_ar_tps": OPERATIVE_AR_TPS, "anchor_base_fullhead_spec_tps": ANCHOR_BASE_FULLHEAD_SPEC_TPS,
        },
    )
    if run is None:
        return None
    s = report["synthesis"]
    summary = {k: v for k, v in s.items() if _finite(v) or isinstance(v, (bool, str))}
    summary["primary_metric"] = s["freerun_seq_exact"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="base-mtp-bodyhead-identity-report",
                      artifact_type="identity-census-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check (4x24, no probe)")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--tie-probe-limit", type=int, default=80,
                    help="max divergence positions to probe on the M=1 spec-OFF server")
    ap.add_argument("--no-tie-probe", action="store_true")
    ap.add_argument("--specoff-self-det", action="store_true",
                    help="capture a 2nd spec-OFF warm pass to assert the reference is warm-bit-stable")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 4)
        args.output_len = min(args.output_len, 24)
        args.no_tie_probe = True

    from scripts.local_validation import harness, paths
    for note in paths.prepare_local_gpu_env():
        print(f"[bmh] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- Phase 1: spec-ON (MTP-K7) capture: cold (warmup) + warm (measured) ----
    mtp = capture_arm(harness, paths, server_python=server_python, arm="mtp",
                      num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                      passes=2)
    if "warm" not in mtp.get("files", {}):
        print(f"[bmh] FATAL: spec-ON warm capture missing ({mtp.get('error')})", flush=True)
        return 2
    mtp_warm = mtp["files"]["warm"]
    print(f"[bmh] spec-ON captured peak={mtp['peak_vram_gb']:.2f}GB ({time.time()-t_start:.0f}s)", flush=True)

    # gpu settle before the next fresh boot
    free = _wait_gpu_free()
    print(f"[bmh] gpu settle: used={free if free is not None else 'TIMEOUT'} MiB", flush=True)

    # ---- Phase 2: spec-OFF (M=1 AR) capture: cold + warm[ + warm2], then on-stack flip probe ----
    holder: dict[str, Any] = {}

    def _probe(srv) -> dict[str, Any] | None:
        # build the identity verdict (warm spec-OFF vs warm spec-ON) + divergence jobs, then read
        # the M=1 logprob margins on THIS live spec-OFF server.
        specoff_warm = str(OUT_ROOT / "decode_ref_warm.jsonl")
        identity = C.verify_pair(paths, specoff_warm, mtp_warm)
        holder["identity"] = identity
        # spec-OFF warm self-determinism (optional 2nd warm pass)
        if args.specoff_self_det and (OUT_ROOT / "decode_ref_warm2.jsonl").exists():
            holder["self_det"] = C.verify_pair(paths, specoff_warm, str(OUT_ROOT / "decode_ref_warm2.jsonl"))
        if args.no_tie_probe:
            return None
        ref_recs = C.load_decode(specoff_warm)
        cand_recs = C.load_decode(mtp_warm)
        div_jobs = C.build_divergence_jobs(ref_recs, cand_recs, identity["_per_prompt"],
                                           "mtp", args.tie_probe_limit)
        ctrl_jobs = C.build_control_jobs(ref_recs, identity["_per_prompt"], min(args.tie_probe_limit, 60))
        if not (div_jobs or ctrl_jobs):
            return {"rows": [], "note": "no divergences"}
        print(f"[bmh] flip-margin probe on M=1 spec-OFF server: "
              f"{len(div_jobs)} miss + {len(ctrl_jobs)} control positions", flush=True)
        return C.flip_margin_probe(srv.base_url, srv.served_model_name, div_jobs + ctrl_jobs)

    ref = capture_arm(harness, paths, server_python=server_python, arm="ref",
                      num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                      passes=(3 if args.specoff_self_det else 2), probe_closure=_probe)
    print(f"[bmh] spec-OFF captured + verified peak={ref['peak_vram_gb']:.2f}GB "
          f"({time.time()-t_start:.0f}s)", flush=True)

    identity = holder.get("identity")
    if identity is None:
        print(f"[bmh] FATAL: identity verdict not produced ({ref.get('error')})", flush=True)
        return 2
    tie = ref.get("probe_result")
    self_det = holder.get("self_det")
    flip = classify_flips(tie, identity)
    peak_vram = max(mtp.get("peak_vram_gb", 0.0), ref.get("peak_vram_gb", 0.0))
    synthesis = synthesize(identity, self_det, flip, peak_vram, args.num_prompts, args.output_len)

    report = {
        "pr": 600, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
        "substrate": "int4_w4a16_marlin_body + full_bf16_262144_untied_head + linear_mtp_k7",
        "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": SEED,
        "spec_on_arm": {k: v for k, v in mtp.items()},
        "spec_off_arm": {k: v for k, v in ref.items() if k != "probe_result"},
        "identity": {k: v for k, v in identity.items() if k != "_per_prompt"},
        "self_det": self_det,
        "flip_probe": tie,
        "flip_classification": flip,
        "synthesis": synthesis,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    out_file = OUT_ROOT / ("census_smoke.json" if args.smoke else "census_report.json")
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[bmh] report -> {out_file} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[bmh] wandb run id={rid}", flush=True)
    return 0


def _wait_gpu_free(threshold_mib: float = 3000.0, timeout_s: float = 150.0) -> float | None:
    import subprocess
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
            used = max(vals) if vals else 0.0
        except (OSError, subprocess.SubprocessError):
            used = 0.0
        if used < threshold_mib:
            return used
        time.sleep(3)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
