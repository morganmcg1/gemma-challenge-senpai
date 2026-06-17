#!/usr/bin/env python
"""PR #607 wirbel — CONTROLLED WARM spec-dec verify-path #319 greedy-identity census.

THE LOAD-BEARING QUESTION (the #481 option-B fire candidate, fern #597 follow-up #1):
does `int4_g128_lmhead + MTP-K7` spec-dec REALLY break strict byte-exact #319 greedy
identity (structural, fireable), or does its measured divergence collapse toward identity
once the *cross-start* venv-nondeterminism confound is subtracted?

fern #597 measured freerun_seq_exact=0.3125 (DIVERGENT) and denken #600 found the int4
BODY alone breaks spec-identity. BUT lawine #601 showed the engine can be non-deterministic
*cross-start* (dev307: ref-vs-ref2 ~112/128 div; 0.22.0: 128/128 clean). fern's runner
serves the reference (spec-OFF) and candidate (spec-ON) on SEPARATE server boots, so its
divergence could be partly a venv artifact rather than the pure structural int4-Marlin
M-dependence (M=8 verify forward != M=1 AR forward).

THIS census eliminates that confound on lawine's DETERMINISTIC gate (vLLM 0.22.0, the exact
submission pin):

  arm  ref   : spec-OFF (SENPAI_REFERENCE_MODE=1)  -> plain-AR int4, boot #1
  arm  ref2  : spec-OFF, a SEPARATE boot           -> ref vs ref2 = the CROSS-START FLOOR
  arm  cand  : spec-ON  NUM_SPECULATIVE_TOKENS=7   -> int4 + MTP-K7 (M=8 verify), boot #3

Every arm is WARM-scored exactly as the official harness scores #319: each boot captures
TWO full 128x512 decode passes — pass 'a' is the warmup load (cold), pass 'b' is the scored
WARM pass (the operative serving regime, wirbel #599 Q1). All comparisons use the OFFICIAL
`check_greedy_identity.py` verifier at ZERO tolerance.

Headline metrics (warm 'b' passes):
  * ref_vs_ref2 : the cross-start floor (expect 0/65536 on the 0.22.0 deterministic gate)
  * ref_vs_cand : the spec verdict — freerun_seq_exact + token divergence + onset
  * chaos cross-tab : cand divergence on prompts that are STABLE across ref/ref2 = the
                      STRUCTURAL spec break ABOVE the cross-start floor (the fireable number)

VERDICT:
  - floor clean (ref==ref2) AND cand diverges  -> spec STRUCTURALLY breaks strict #319; the
    M-dependence is real; option B necessarily means relaxing #319.
  - cand divergence collapses toward the floor  -> spec is identity-recoverable (major reopen).

LOCAL ONLY: analysis_only, official_tps=0, single A10G, NO HF Job / no --launch / no submission
change. Reuses the official harness primitives so the capture/gate path is the official one.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"

# fern #597 build, confirmed on disk (DO NOT rebuild — 9.7G int4 W4A16 + 183M MTP drafter).
MODEL_DIR = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER = "/tmp/qat-assistant"
K = 7  # num_speculative_tokens (matches fern #597; M = K+1 = 8 verify granularity)

SEED = paths.SEED          # 1 — fixes the same 128 sharegpt prompts for every capture
TAU_LO = 1.035             # banked local->official TPS scalar (#594), for the cand sanity-probe

# Cited anchors (NOT re-derived here) — from the PR #607 baseline block.
ANCHOR_CAND_SEQ_EXACT = 0.3125     # fern #597 freerun_seq_exact (16 prompts, separate boots)
ANCHOR_CAND_PROXY_TPS = 427.7      # fern #597 official-proxy TPS
ANCHOR_REF_OFFICIAL_TPS = 126.378  # plain-AR int4 reference, 128/128 VALID (#905tbujn)
ANCHOR_06_LAWINE_0220 = "0.22.0 cross-start = 128/128 (lawine #601 ivbje4oz)"
ANCHOR_19_INT4_FLIP = 0.00376      # kanna #19 int4+MTP BI=1 flip/tok (separate-boot, MERGED)


def base_env() -> dict[str, str]:
    """The fern #597 serve recipe on int4_mtp_batchinv: int4 target + MTP drafter, M=1, BI=1,
    cudagraph ON (the OPERATIVE serving regime #319 is scored on — NOT eager)."""
    return {
        "MODEL_ID": MODEL_DIR,
        "DRAFTER_MODEL": DRAFTER,
        "VLLM_BATCH_INVARIANT": "1",          # ship config; fern #597 + kanna #19 used BI=1
        "MAX_NUM_SEQS": "1",                  # conc=1 -> spec verify is the only M>1 source
        "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "512",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",   # PyTorch-native lowest-index argmax tie-break
    }


def arm_env(arm: str) -> dict[str, str]:
    env = base_env()
    if arm in ("ref", "ref2"):
        # canonical #319 reference: clear the drafter -> plain M=1 AR greedy on THIS engine.
        env["SENPAI_REFERENCE_MODE"] = "1"
        env["NUM_SPECULATIVE_TOKENS"] = "0"
    elif arm == "cand":
        env["NUM_SPECULATIVE_TOKENS"] = str(K)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return env


def _gpu_used_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def wait_gpu_free(threshold_mib: float = 3000.0, timeout_s: float = 180.0) -> float | None:
    """Block until the previous LocalServer's async SIGTERM frees GPU memory below threshold."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = _gpu_used_mib()
        if last <= threshold_mib:
            return last
        time.sleep(3.0)
    return last


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        peak["mib"] = max(peak["mib"], _gpu_used_mib())
        stop.wait(2.0)


def _pass_files(arm: str, p: str) -> tuple[Path, Path]:
    return (HERE / f"decode_{arm}_{p}.jsonl", HERE / f"decode_{arm}_{p}.summary.json")


def _decode_complete(out_file: Path, summary_file: Path, n: int, output_len: int) -> bool:
    if not out_file.exists() or not summary_file.exists():
        return False
    try:
        summ = json.loads(summary_file.read_text())
    except (OSError, ValueError):
        return False
    if int(summ.get("num_records", -1)) != n:
        return False
    if int(summ.get("num_completion_tokens", -1)) != n * output_len:
        return False
    cnt = 0
    try:
        with open(out_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "completion_token_ids" not in d or "id" not in d:
                    return False
                cnt += 1
    except (OSError, ValueError):
        return False
    return cnt == n


def run_arm(*, server_python: Path, arm: str, num_prompts: int, output_len: int, port: int,
            passes: list[str], probe_tps: bool, resume: bool) -> dict[str, Any]:
    """Boot one arm (its own server -> cross-start), capture each WARM pass. pass 'a' is the
    warmup load, pass 'b' is the scored warm pass. Resumable: a complete capture is reused."""
    extra_env = arm_env(arm)
    log_path = HERE / f"server_{arm}.log"
    result: dict[str, Any] = {"arm": arm, "files": {}, "summaries": {}, "env": extra_env}

    want = [(p, *_pass_files(arm, p)) for p in passes]
    if resume and all(_decode_complete(o, s, num_prompts, output_len) for _, o, s in want):
        print(f"[census] [{arm}] reusing {len(passes)} complete capture(s) — skip boot", flush=True)
        for p, o, s in want:
            result["files"][p] = str(o)
            result["summaries"][p] = json.loads(s.read_text())
        result["reused"] = True
        result["peak_vram_gb"] = 0.0
        return result

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
            result["serve_ready_s"] = time.time() - t0
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            for p, out_file, summary_file in want:
                kind = "warmup/cold" if p == "a" else "SCORED/warm"
                print(f"[census] [{arm}] capture pass '{p}' ({kind}) "
                      f"{num_prompts}x{output_len} conc=1 -> {out_file.name}", flush=True)
                summary = harness.capture_decode(
                    server_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_file, summary_file=summary_file,
                    num_prompts=num_prompts, output_len=output_len, seed=SEED,
                    tokenizer=paths.TOKENIZER, dataset=paths.EVAL_PROMPTS, timeout_s=5400,
                )
                result["files"][p] = str(out_file)
                result["summaries"][p] = summary
            if probe_tps:
                result["tps_probe"] = harness.probe_tps(
                    srv.base_url, srv.served_model_name, decode_tokens=output_len)
    finally:
        stop.set()
        sampler.join(timeout=5)
    result["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    result["log_path"] = str(log_path)
    result["plumbing"] = _grep_log(log_path, [
        "Speculative", "speculative_config", "SENPAI_REFERENCE_MODE active",
        "Overriding", "MarlinLinearKernel", "TRITON_ATTN", "num_speculative_tokens",
    ])
    print(f"[census] [{arm}] done in {time.time() - t0:.0f}s, peak {result['peak_vram_gb']:.1f} GB", flush=True)
    return result


def _grep_log(log_path: Path, needles: list[str]) -> dict[str, bool]:
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return {n: False for n in needles}
    return {n: (n in text) for n in needles}


def verify(ref_file: str, cand_file: str) -> dict[str, Any]:
    """Official #319 verifier (zero tolerance) -> distilled census metrics for a pair."""
    report = greedy_gate.compare(Path(ref_file), Path(cand_file))
    n = report.num_prompts_compared or 0
    tok = report.total_tokens_compared or 0
    onsets = sorted(p.first_divergence_index for p in report.per_prompt
                    if not p.identical and p.first_divergence_index is not None)
    return {
        "verdict": report.verdict,
        "num_prompts_compared": n,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "freerun_seq_exact": (report.num_identical / n) if n else None,
        "total_tokens_compared": tok,
        "total_divergent_tokens": report.total_divergent_tokens,
        "freerun_token_identity": (1.0 - report.total_divergent_tokens / tok) if tok else None,
        "onset_min": onsets[0] if onsets else None,
        "onset_median": int(statistics.median(onsets)) if onsets else None,
        "onset_max": onsets[-1] if onsets else None,
        "divergent_keys": sorted(p.key for p in report.per_prompt if not p.identical),
        "_per_prompt": [
            {"key": p.key, "identical": p.identical,
             "first_divergence_index": p.first_divergence_index,
             "num_divergent_tokens": p.num_divergent_tokens}
            for p in report.per_prompt
        ],
    }


def chaos_crosstab(floor: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The load-bearing diagnosis. floor = ref vs ref2 (cross-start); spec = ref vs cand.

    CHAOTIC prompts (ref != ref2) are intrinsically non-deterministic cross-start — a spec
    'miss' on one of those adds NOTHING beyond the venv floor. A spec divergence on a STABLE
    prompt (ref == ref2) is a NEWLY-broken identity -> the structural M-dependence ABOVE the
    cross-start floor (the fireable number)."""
    chaotic = {p["key"] for p in floor["_per_prompt"] if not p["identical"]}
    stable = {p["key"] for p in floor["_per_prompt"] if p["identical"]}
    cand_div = {p["key"] for p in spec["_per_prompt"] if not p["identical"]}
    on_stable = sorted(cand_div & stable)
    on_chaotic = sorted(cand_div & chaotic)
    # structural token divergence: divergent tokens of cand ONLY on stable prompts.
    struct_tokens = sum(p["num_divergent_tokens"] for p in spec["_per_prompt"]
                        if not p["identical"] and p["key"] in stable)
    return {
        "n_chaotic_floor": len(chaotic),
        "n_stable_floor": len(stable),
        "chaotic_floor_keys": sorted(chaotic),
        "cand_n_divergent": len(cand_div),
        "cand_n_divergent_on_stable": len(on_stable),
        "cand_n_divergent_on_chaotic": len(on_chaotic),
        "cand_divergent_on_stable_keys": on_stable,
        "cand_divergence_subset_of_floor": (len(on_stable) == 0),
        "structural_divergent_tokens_on_stable": struct_tokens,
    }


def synthesize(floor: dict[str, Any], spec: dict[str, Any], crosstab: dict[str, Any],
               within: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The PR #607 verdict: structural spec break vs identity-recoverable, floor-subtracted."""
    floor_clean = (floor["verdict"] == "GREEDY_IDENTICAL" and floor["total_divergent_tokens"] == 0)
    cand_diverges = (spec["verdict"] == "DIVERGENT")
    # structural break = cand diverges on >=1 prompt that is STABLE across ref/ref2 (i.e. not a
    # cross-start chaos artifact). This is the robust, floor-subtracted signal and takes
    # precedence even when the floor is not perfectly clean.
    structural_break = cand_diverges and not crosstab["cand_divergence_subset_of_floor"]
    within_clean = all(w.get("total_divergent_tokens") == 0 for w in within.values() if w)
    if structural_break:
        verdict = "SPEC_STRUCTURALLY_BREAKS_319"
    elif cand_diverges and not floor_clean:
        # every cand miss falls on a cross-start-chaotic prompt -> cannot attribute to spec
        verdict = "CONFOUNDED_floor_not_clean"
    elif not cand_diverges:
        verdict = "SPEC_IDENTITY_RECOVERED"
    else:
        verdict = "SPEC_DIVERGENCE_WITHIN_FLOOR"
    return {
        "pr607_verdict": verdict,
        "cross_start_floor_clean": floor_clean,
        "cross_start_floor_seq_exact": floor["freerun_seq_exact"],
        "cross_start_floor_divergent_tokens": floor["total_divergent_tokens"],
        "within_instance_clean": within_clean,
        "cand_warm_seq_exact": spec["freerun_seq_exact"],
        "cand_warm_token_identity": spec["freerun_token_identity"],
        "cand_warm_divergent_tokens": spec["total_divergent_tokens"],
        "cand_warm_verdict": spec["verdict"],
        "structural_break_above_floor": structural_break,
        "structural_divergent_prompts_on_stable": crosstab["cand_n_divergent_on_stable"],
        "structural_divergent_tokens_on_stable": crosstab["structural_divergent_tokens_on_stable"],
        "fern597_seq_exact_anchor": ANCHOR_CAND_SEQ_EXACT,
        "kanna19_int4_flip_per_tok_anchor": ANCHOR_19_INT4_FLIP,
        "passes_strict_319": (not cand_diverges) and floor_clean,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--passes", default="a,b", help="comma list; 'a'=warmup, 'b'=scored warm")
    ap.add_argument("--arms", default="ref,ref2,cand")
    ap.add_argument("--ref-port", type=int, default=8021)
    ap.add_argument("--ref2-port", type=int, default=8022)
    ap.add_argument("--cand-port", type=int, default=8023)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="4 prompts x 64 tok wiring check (not a verdict)")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts, args.output_len = 4, 64

    for note in paths.prepare_local_gpu_env():
        print(f"[census] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    tf_ver = harness._dist_version(server_python, "transformers")
    print(f"[census] server_python={server_python} vllm={vllm_ver} transformers={tf_ver}", flush=True)

    passes = [p.strip() for p in args.passes.split(",") if p.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scored = "b" if "b" in passes else passes[-1]  # the warm scored pass
    ports = {"ref": args.ref_port, "ref2": args.ref2_port, "cand": args.cand_port}
    resume = not args.no_resume

    arm_results: dict[str, Any] = {}
    for arm in arms:
        if arm != arms[0]:
            free = wait_gpu_free()
            print(f"[census] gpu free before {arm}: {free} MiB", flush=True)
        arm_results[arm] = run_arm(
            server_python=server_python, arm=arm, num_prompts=args.num_prompts,
            output_len=args.output_len, port=ports[arm], passes=passes,
            probe_tps=(arm == "cand"), resume=resume,
        )

    def scored_file(arm: str) -> str | None:
        return arm_results.get(arm, {}).get("files", {}).get(scored)

    # ---- comparisons (official verifier, zero tolerance), warm scored passes ----
    comparisons: dict[str, Any] = {}
    if scored_file("ref") and scored_file("ref2"):
        comparisons["ref_vs_ref2_CROSS_START_FLOOR"] = verify(scored_file("ref"), scored_file("ref2"))
    if scored_file("ref") and scored_file("cand"):
        comparisons["ref_vs_cand_SPEC_VERDICT"] = verify(scored_file("ref"), scored_file("cand"))

    # within-instance self-determinism (cold 'a' vs warm 'b'), free from each boot.
    within: dict[str, Any] = {}
    for arm in arms:
        fa = arm_results[arm]["files"].get("a")
        fb = arm_results[arm]["files"].get("b")
        if fa and fb and fa != fb:
            within[arm] = verify(fa, fb)

    floor = comparisons.get("ref_vs_ref2_CROSS_START_FLOOR")
    spec = comparisons.get("ref_vs_cand_SPEC_VERDICT")
    crosstab = chaos_crosstab(floor, spec) if (floor and spec) else None
    verdict = synthesize(floor, spec, crosstab, within) if (floor and spec) else None

    cand_tps = arm_results.get("cand", {}).get("tps_probe", {})
    local_tps = cand_tps.get("decode_tps_single_stream")
    proxy_tps = local_tps * TAU_LO if isinstance(local_tps, (int, float)) else None

    report = {
        "pr": 607,
        "smoke": args.smoke,
        "config": {
            "model_dir": MODEL_DIR, "drafter": DRAFTER, "k": K,
            "vllm_version": vllm_ver, "transformers_version": tf_ver,
            "batch_invariant": 1, "max_num_seqs": 1, "seed": SEED,
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "passes": passes, "scored_pass": scored, "cudagraph": "ON(operative)",
            "deterministic_gate": ANCHOR_06_LAWINE_0220,
        },
        "arms": {a: {k: v for k, v in r.items() if k != "summaries"}
                 for a, r in arm_results.items()},
        "comparisons": comparisons,
        "within_instance": within,
        "chaos_crosstab": crosstab,
        "verdict": verdict,
        "cand_local_tps": local_tps,
        "cand_official_proxy_tps": proxy_tps,
        "peak_vram_gb": max((r.get("peak_vram_gb") or 0.0) for r in arm_results.values()),
    }
    out = HERE / ("census_report.smoke.json" if args.smoke else "census_report.json")
    out.write_text(json.dumps(report, indent=2, default=str))

    # ---- console verdict ----
    print("\n" + "=" * 72, flush=True)
    print(f"[PR607] spec-dec verify-path #319 census ({'SMOKE' if args.smoke else 'FULL'})", flush=True)
    print(f"  vLLM={vllm_ver}  K={K}  BI=1  {args.num_prompts}x{args.output_len}  scored=warm '{scored}'", flush=True)
    if floor:
        print(f"  CROSS-START FLOOR  ref vs ref2 : {floor['verdict']}  "
              f"seq_exact={floor['freerun_seq_exact']}  div_tok={floor['total_divergent_tokens']}/{floor['total_tokens_compared']}", flush=True)
    if spec:
        print(f"  SPEC VERDICT       ref vs cand : {spec['verdict']}  "
              f"seq_exact={spec['freerun_seq_exact']}  div_tok={spec['total_divergent_tokens']}/{spec['total_tokens_compared']}", flush=True)
        print(f"    onset (tok idx): min={spec['onset_min']} median={spec['onset_median']} max={spec['onset_max']}", flush=True)
    if crosstab:
        print(f"  chaos cross-tab: floor chaotic={crosstab['n_chaotic_floor']} stable={crosstab['n_stable_floor']}; "
              f"cand divergent_on_STABLE={crosstab['cand_n_divergent_on_stable']} "
              f"(struct tokens={crosstab['structural_divergent_tokens_on_stable']})", flush=True)
    if within:
        for a, w in within.items():
            print(f"  within-instance {a} (cold vs warm): {w['verdict']} div_tok={w['total_divergent_tokens']}", flush=True)
    if verdict:
        print(f"  >>> PR607 VERDICT: {verdict['pr607_verdict']}  (passes_strict_319={verdict['passes_strict_319']})", flush=True)
    if proxy_tps:
        print(f"  cand local_tps={local_tps:.1f} proxy={proxy_tps:.1f} (fern #597 anchor {ANCHOR_CAND_PROXY_TPS})", flush=True)
    print(f"  report -> {out}", flush=True)
    print("=" * 72, flush=True)

    if not args.no_wandb:
        _log_wandb(report, name=args.wandb_name, group=args.wandb_group)
    return 0


def _log_wandb(report: dict[str, Any], *, name: str | None, group: str | None) -> None:
    try:
        from scripts import wandb_logging as wl
    except ImportError:
        return
    cfg = report["config"]
    run = wl.init_wandb_run(
        job_type="specdec-verify-identity-census", agent="wirbel",
        name=name or "wirbel/specdec-verify-identity-census",
        group=group or "specdec-verify-identity-census",
        notes="PR607 controlled warm spec-dec verify-path #319 census: cross-start-floor-subtracted",
        tags=["pr607", "specdec", "greedy-identity", "cross-start-floor", "int4-mtp"],
        config=cfg,
    )
    if run is None:
        print("[census] wandb not configured (no API key/mode) — skipping", flush=True)
        return
    metrics: dict[str, Any] = {}
    for label, comp in report.get("comparisons", {}).items():
        for k in ("freerun_seq_exact", "freerun_token_identity", "total_divergent_tokens",
                  "num_identical", "num_divergent", "onset_min", "onset_median", "onset_max"):
            v = comp.get(k)
            if isinstance(v, (int, float)):
                metrics[f"{label}/{k}"] = v
    for k, v in (report.get("verdict") or {}).items():
        if isinstance(v, (int, float, bool)):
            metrics[f"verdict/{k}"] = int(v) if isinstance(v, bool) else v
    for k, v in (report.get("chaos_crosstab") or {}).items():
        if isinstance(v, (int, float)):
            metrics[f"crosstab/{k}"] = v
    for k in ("cand_local_tps", "cand_official_proxy_tps", "peak_vram_gb"):
        v = report.get(k)
        if isinstance(v, (int, float)):
            metrics[k] = v
    wl.log_event(run, "census_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["pr607_verdict"] = (report.get("verdict") or {}).get("pr607_verdict")
    wl.log_json_artifact(run, name="pr607_census_report", artifact_type="census", data=report)
    wl.finish_wandb(run)
    print("[census] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
