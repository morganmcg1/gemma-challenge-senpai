#!/usr/bin/env python
"""PR #596 denken — is base_fullhead's OPERATIVE greedy identity bit-stable run-to-run?
(LOCAL A10G, analysis-only, NO HF fire, NO served-file change.)

THE QUESTION: the converged NO-FIRE rests on base_fullhead being the operative-identity-safe
anchor — its served greedy argmax sequence IS the #319 operative / #407 reference (wirbel #588 is
formalizing that predicate). But no card has tested whether that served greedy argmax is
DETERMINISTIC across fresh server (re)starts. vLLM can carry non-determinism from GPU
atomic/reduction ordering, per-process kernel autotuning, and CUDA-graph capture. The recipe pins
MAX_NUM_SEQS=1 + greedy temp=0 + VLLM_USE_FLASHINFER_SAMPLER=0 (removes batch-order
non-determinism) — but residual GEMV/reduction non-determinism could still flip a near-tie argmax
on a small fraction of token-steps. If the operative identity is not bit-stable across repeated
served decodes, the wirbel #588 predicate is on sand.

WHAT THIS EXTENDS: my own PR #576 self-determinism measured repeated decode passes against ONE
live server (within-process: cold pass-1 vs warm pass-2 -> chaos floor 0.492 seq; warm pass-2 vs
pass-3 -> GREEDY_IDENTICAL 1.0). That is WITHIN-process. This card measures the genuinely
different CROSS-process question the operative identity actually needs: boot the server FRESH N
times (separate process + model reload each) and ask whether each fresh server's served greedy
argmax is token-identical to the others.

RECIPE (the EXACT base_fullhead operative anchor): submissions/fa2sw_strict_surgical357 +
base_fullhead overrides (full native 262k bf16 head, prune OFF, my int4 QAT snapshot, flashinfer
sampler OFF) + SENPAI_REFERENCE_MODE=1 -> plain M=1 AR greedy. This is THE canonical #319 operative
reference (gen_greedy_reference --spec-off contract) — the unambiguous "served greedy argmax the
contract is measured against", with the spec-path confound removed. (The served spec-ON MTP-K7
path that yields the 252.69 anchor inherits this determinism: it verifies against these same M=1
target logits — PR #576 established spec-ON == spec-OFF modulo bf16 ties.)

DESIGN: N fresh servers, 2 passes each:
  * cold = first decode after boot (the #319-reference regime)
  * warm = second decode (the steady-state operative serving regime; HEADLINE)
Cross-process determinism is measured per regime (warm = headline) AND within-process cold->warm
is recomputed (continuity with #576's chaos floor). The headline served_argmax_determinism_rate is
the warm cross-process MATCHED-STATE per-step rate (walk-off removed — the #576 lesson). Flips are
characterised by the M=1 logprob margin (bf16 near-tie vs genuine) and by whether any flip lands
BEFORE the natural EOS (a real-response divergence that could move a GPQA/AIME answer) vs only in
the ignore_eos free-run tail.

LOCAL only: analysis_only, official_tps=0, no HF Job, no --launch, no submission, no served-file
change. base_fullhead is reached purely by serve-env overrides on fa2sw_strict_surgical357.

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/served_identity_determinism/determinism_driver.py --smoke --no-wandb
Full:
  CUDA_VISIBLE_DEVICES=0 python research/served_identity_determinism/determinism_driver.py \
    --num-servers 5 --num-prompts 128 --output-len 256 \
    --wandb_name denken/served-identity-determinism --wandb_group served-identity-determinism
"""
from __future__ import annotations

import argparse
import json
import math
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
HERE = Path(__file__).resolve().parent
CENSUS_DIR = ROOT / "research" / "specdec_identity_census"
if str(CENSUS_DIR) not in sys.path:
    sys.path.insert(0, str(CENSUS_DIR))

import census_driver as C  # noqa: E402  reuse the base_fullhead serve + #319-verify infra

OUT_ROOT = HERE
SEED = C.SEED
SUBMISSION = C.SUBMISSION
MODEL_DIR = C.MODEL_DIR

# gemma-4-E4B-it natural-stop tokens (generation_config.json eos_token_id). Under the #319 audit's
# ignore_eos=True the model still EMITS these but would normally have stopped — used to split a
# divergence into real-response (pre-EOS, operative) vs free-run audit tail (post-EOS, invisible).
EOS_IDS = {1, 106, 50}

# Cited anchors (provided in the PR #596 body / BASELINE — NOT re-derived).
ANCHOR_SPEC_ON_TPS = 252.69     # base_fullhead served MTP-K7 spec-ON (wirbel #553 83jiwjr9)
ANCHOR_PPL = 2.0057
SPEC_OFF_AR_TPS = 97.0          # base_fullhead spec-OFF M=1 AR (#569)

# fern #587 GPQA-D base_fullhead seed-swing (given in the PR #596 body — NOT re-measured here).
GPQA = {
    "n": 198,
    "default_acc": 0.4798, "default_correct": 95,
    "seed12345_acc": 0.4697, "seed12345_correct": 93,
    "swing_problems": 2, "swing_abs": 0.4798 - 0.4697,
    "gate": 0.471,
}
# The model's own generation_config.json (the int4 QAT snapshot served here) — i.e. the sampler the
# downstream eval harness uses (lewtun #31: downstream evals use generation_config sampling, NOT
# greedy). do_sample=True @ temp 1.0 => the GPQA harness SAMPLES; seed-to-seed variance is EXPECTED.
GEN_CONFIG = {"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95}


# ========================================================================== #
# config registry: which served stack's run-to-run determinism we measure
# ========================================================================== #
# Per the advisor #594 re-target, the LOAD-BEARING config is int4_g128_lmhead + MTP-K7 (fern #597's
# fire candidate). Launch isolation forbids inspecting fern's branch where the +MTP-K7 wiring lives,
# and the advisor asked not to double-instrument. So instead of re-building fern's spec+int4 config
# we decompose the answer with resources on THIS branch only:
#   int4g128_specoff = the int4-head DECODE SUBSTRATE (its own clean serve) — the exact
#                      GEMV->logits->argmax compute the K7 verify runs on -> the load-bearing number.
#   base_mtp vs base_specoff = the INHERITANCE BRIDGE: spec-ON MTP-K7 determinism == spec-OFF M=1 AR
#                      determinism (#576: spec divergences share the no-spec chaos-floor bf16-tie
#                      signature) -> licenses "int4_g128+MTP det == int4_g128 substrate det".
INT4_G128_SUBMISSION = ROOT / "submissions" / "int4_g128_lmhead"


def _int4g128_env() -> dict[str, str]:
    """int4 g128 body + untied int4 lm_head, served by the submission's OWN clean vLLM serve.
    conc=1 is enforced client-side (decode_outputs.py is sequential urllib) so the submission's
    default MAX_NUM_SEQS is moot for batch-order determinism; flashinfer sampler OFF matches the
    anchor tie-break (lowest-index argmax) and avoids the in-container cuRAND JIT."""
    return {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "GPU_MEMORY_UTILIZATION": "0.90",
    }


def build_configs() -> dict[str, dict[str, Any]]:
    return {
        "base_specoff": {
            "submission": SUBMISSION, "extra_env": C.arm_env("ref"), "prefix": "",
            "spec_mode": "OFF_M1_AR", "model_dir": MODEL_DIR,
            "label": "base_fullhead spec-OFF M=1 AR (#319 reference)",
        },
        "base_mtp": {
            "submission": SUBMISSION, "extra_env": C.arm_env("mtp"), "prefix": "base_mtp_",
            "spec_mode": "ON_MTP_K7", "model_dir": MODEL_DIR,
            "label": "base_fullhead spec-ON MTP-K7 (served stack)",
        },
        "int4g128_specoff": {
            "submission": INT4_G128_SUBMISSION, "extra_env": _int4g128_env(),
            "prefix": "int4g128_", "spec_mode": "OFF_M1_AR",
            "model_dir": "submission-bundled int4_g128_lmhead/model",
            "label": "int4_g128_lmhead decode substrate M=1 AR",
        },
    }


# ========================================================================== #
# fresh-server capture
# ========================================================================== #
def _gpu_used_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else 0.0
    except (OSError, subprocess.SubprocessError):
        return 0.0


def wait_gpu_free(threshold_mib: float = 3000.0, timeout_s: float = 150.0) -> float | None:
    """Block until the GPU is (nearly) empty so the next fresh server boots cleanly. Each
    base_fullhead server holds ~21.8 GB; the previous LocalServer's SIGTERM frees it asynchronously.
    Returns the observed free-MiB reading, or None on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_used_mib()
        if used < threshold_mib:
            return used
        time.sleep(3)
    return None


def capture_one_server(harness: Any, paths: Any, server_python: Path, *, submission: Path,
                       extra_env: dict[str, str], prefix: str, port: int, run_idx: int,
                       num_prompts: int, output_len: int, passes: int) -> dict[str, Any]:
    """Boot the served stack ONCE (fresh process + model reload) and capture `passes` greedy decode
    files (cold, warm). Each call is an independent operative (re)start. `prefix`/`extra_env`/
    `submission` select which config (base_specoff / base_mtp / int4g128_specoff)."""
    log_path = OUT_ROOT / f"server_{prefix}run{run_idx}.log"
    tags = ["cold", "warm"][:passes]
    result: dict[str, Any] = {"run_idx": run_idx, "files": {}, "booted": False, "reused": False}

    # resume: a fully-complete capture set is a deterministic greedy decode -> reuse identical.
    def _f(t: str) -> tuple[Path, Path]:
        return (OUT_ROOT / f"decode_{prefix}run{run_idx}_{t}.jsonl",
                OUT_ROOT / f"decode_{prefix}run{run_idx}_{t}.summary.json")
    if all(C._decode_complete(o, s, num_prompts, output_len) for o, s in (_f(t) for t in tags)):
        for t in tags:
            result["files"][t] = str(_f(t)[0])
        result["reused"] = True
        result["peak_vram_gb"] = 0.0
        print(f"[det] {prefix}run{run_idx} reusing {passes} complete capture(s) — skip boot", flush=True)
        return result

    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=C._sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            result["booted"] = True
            result["boot_s"] = round(time.time() - t0, 1)
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            for t in tags:
                out_file, summary_file = _f(t)
                print(f"[det] {prefix}run{run_idx} capture {t} {num_prompts}x{output_len} conc=1 "
                      f"-> {out_file.name}", flush=True)
                harness.capture_decode(
                    server_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_file, summary_file=summary_file,
                    num_prompts=num_prompts, output_len=output_len, seed=SEED,
                    tokenizer=paths.TOKENIZER, dataset=paths.EVAL_PROMPTS, timeout_s=5400)
                result["files"][t] = str(out_file)
    except Exception as exc:  # boot/decode failure on one server must not kill the whole run
        result["error"] = repr(exc)
        print(f"[det] run{run_idx} FAILED: {exc!r}", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=5)
    result["peak_vram_gb"] = round((peak["mib"] or 0.0) / 1024.0, 2)
    result["plumbing"] = C.grep_log(str(log_path), C.PLUMBING_NEEDLES)
    result["log_path"] = str(log_path)
    return result


# ========================================================================== #
# cross-process comparison
# ========================================================================== #
def _allN_agreement(files: list[str]) -> dict[str, Any]:
    """Literal 'fraction of token-steps whose argmax is identical across ALL N runs' (free-run) +
    the all-N sequence byte-exact rate. Free-run: dominated by post-divergence walk-off, so it is
    the operative-facing lower bound, NOT a per-step flip rate (see matched-state below)."""
    recs = [C.load_decode(f) for f in files]
    keys = set(recs[0])
    for r in recs[1:]:
        keys &= set(r)
    keys = sorted(keys)
    total = agree = seq_ident = 0
    for k in keys:
        seqs = [r[k]["completion_token_ids"] for r in recs]
        lens = [len(s) for s in seqs]
        L = min(lens)
        seq_same = all(x == lens[0] for x in lens)
        for t in range(L):
            total += 1
            tok0 = seqs[0][t]
            if all(s[t] == tok0 for s in seqs):
                agree += 1
            else:
                seq_same = False
        seq_ident += 1 if seq_same else 0
    return {
        "freerun_allN_positional_rate": (agree / total) if total else None,
        "allN_sequence_exact_rate": (seq_ident / len(keys)) if keys else None,
        "total_steps": total, "agree_steps": agree, "n_prompts": len(keys),
    }


def cross_process_metrics(paths: Any, files: list[str], label: str) -> dict[str, Any]:
    """run[0] is the reference; pairwise-verify every other fresh run against it. Pool the
    matched-state per-step hazard (the #576-grade headline, walk-off removed) and also report the
    literal all-N free-run agreement + all-N sequence-exact. Keeps full per-pair verdicts so the
    flip-margin probe can be built at the exact cross-process divergence positions."""
    ref = files[0]
    pairs_full: list[dict] = []
    pooled_trials = pooled_fail = 0
    onsets: list[int] = []
    for i, f in enumerate(files[1:], start=1):
        v = C.verify_pair(paths, ref, f)
        v["_cand_run"] = i
        v["_cand_file"] = f
        pairs_full.append(v)
        pooled_trials += int(v.get("matched_state_trials") or 0)
        pooled_fail += int(v.get("matched_state_failures") or 0)
        if v.get("onset_min") is not None:
            onsets.append(v["onset_min"])
    pooled_hazard = (pooled_fail / pooled_trials) if pooled_trials else None
    pooled_rate = (1.0 - pooled_hazard) if pooled_hazard is not None else None
    allN = _allN_agreement(files)
    seq_rates = [p["sequence_exact_rate"] for p in pairs_full if p["sequence_exact_rate"] is not None]
    verdicts = [p["verdict"] for p in pairs_full]
    all_identical = bool(pairs_full) and all(v == "GREEDY_IDENTICAL" for v in verdicts)
    return {
        "label": label,
        "n_runs": len(files),
        "ref_file": ref,
        "all_pairs_greedy_identical": all_identical,
        "pairwise_verdicts": verdicts,
        # headline: matched-state per-step (teacher-forced up to first divergence; walk-off removed)
        "matched_state_per_step_rate_pooled": pooled_rate,
        "matched_state_per_step_hazard_pooled": pooled_hazard,
        "matched_state_trials_pooled": pooled_trials,
        "matched_state_failures_pooled": pooled_fail,
        # literal "identical across all N" (free-run) + sequence-exact
        "freerun_allN_positional_rate": allN["freerun_allN_positional_rate"],
        "allN_sequence_exact_rate": allN["allN_sequence_exact_rate"],
        "allN_total_steps": allN["total_steps"],
        "allN_agree_steps": allN["agree_steps"],
        "allN_n_prompts": allN["n_prompts"],
        # pairwise sequence-exact spread (ref vs each fresh run)
        "pairwise_sequence_exact_min": min(seq_rates) if seq_rates else None,
        "pairwise_sequence_exact_mean": statistics.mean(seq_rates) if seq_rates else None,
        "onset_min": min(onsets) if onsets else None,
        "onset_median": int(statistics.median(onsets)) if onsets else None,
        "pairwise": [{k: p[k] for k in (
            "_cand_run", "verdict", "sequence_exact_rate", "matched_state_per_step_identity_rate",
            "freerun_positional_identity_rate", "num_divergent", "onset_min", "onset_median")}
            for p in pairs_full],
        "_pairs_full": pairs_full,  # popped before serialisation
    }


def within_coldwarm(paths: Any, servers: list[dict]) -> dict[str, Any] | None:
    """Within-process cold->warm per server (continuity with PR #576's chaos floor). Pooled."""
    pooled_trials = pooled_fail = 0
    rows: list[dict] = []
    for s in servers:
        f = s.get("files", {})
        if "cold" not in f or "warm" not in f:
            continue
        v = C.verify_pair(paths, f["cold"], f["warm"])
        pooled_trials += int(v.get("matched_state_trials") or 0)
        pooled_fail += int(v.get("matched_state_failures") or 0)
        rows.append({"run_idx": s["run_idx"], "verdict": v["verdict"],
                     "sequence_exact_rate": v["sequence_exact_rate"],
                     "matched_state_per_step_identity_rate": v["matched_state_per_step_identity_rate"]})
    if not rows:
        return None
    hz = (pooled_fail / pooled_trials) if pooled_trials else None
    return {
        "n_servers": len(rows),
        "matched_state_per_step_rate_pooled": (1.0 - hz) if hz is not None else None,
        "sequence_exact_rate_mean": statistics.mean(
            [r["sequence_exact_rate"] for r in rows if r["sequence_exact_rate"] is not None]) if rows else None,
        "all_servers_cold_eq_warm": all(r["verdict"] == "GREEDY_IDENTICAL" for r in rows),
        "rows": rows,
    }


# ========================================================================== #
# flip characterisation: margins + EOS/answer impact
# ========================================================================== #
def build_xproc_jobs(primary: dict[str, Any], limit: int) -> tuple[list[dict], list[dict]]:
    """Divergence + control probe jobs at the cross-process (ref vs each fresh run) flip positions."""
    ref_recs = C.load_decode(primary["ref_file"])
    div_jobs: list[dict] = []
    ctrl_jobs: list[dict] = []
    for p in primary["_pairs_full"]:
        cand_recs = C.load_decode(p["_cand_file"])
        verdict_rows = p["_per_prompt"]
        remaining = limit - len(div_jobs)
        if remaining > 0:
            div_jobs += C.build_divergence_jobs(ref_recs, cand_recs, verdict_rows,
                                                drafter=f"xproc_run{p['_cand_run']}", limit=remaining)
        if len(ctrl_jobs) < limit:
            ctrl_jobs += C.build_control_jobs(ref_recs, verdict_rows, limit - len(ctrl_jobs))
        if len(div_jobs) >= limit:
            break
    return div_jobs, ctrl_jobs


def run_flip_probe(harness: Any, server_python: Path, *, submission: Path,
                   extra_env: dict[str, str], prefix: str, port: int,
                   jobs: list[dict]) -> dict[str, Any] | None:
    """Boot one more fresh server (this config's stack) and read the M=1 logprob margin at each
    cross-process divergence (near-tie bf16 reorder vs genuine precision flip) + control positions.
    For int4g128 this reads the int4-head margins; for base_mtp the target prefill (=base_fullhead)
    margins via prompt_logprobs."""
    if not jobs:
        return None
    log_path = OUT_ROOT / f"server_{prefix}probe.log"
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            print(f"[det] flip-margin probe: {len(jobs)} positions", flush=True)
            return C.flip_margin_probe(srv.base_url, srv.served_model_name, jobs)
    except Exception as exc:
        print(f"[det] flip probe FAILED: {exc!r}", flush=True)
        return {"error": repr(exc)}


def _natural_eos_pos(ids: list[int]) -> int:
    for i, t in enumerate(ids):
        if t in EOS_IDS:
            return i
    return len(ids)


def eos_answer_impact(primary: dict[str, Any]) -> dict[str, Any]:
    """Does any cross-process divergence land BEFORE the model's natural EOS (a real-response flip
    that could move a GPQA/AIME answer) or only in the ignore_eos free-run audit tail (invisible)?"""
    ref_recs = C.load_decode(primary["ref_file"])
    real_response = 0
    freerun_tail = 0
    div_tok_is_eos = 0
    examples: list[dict] = []
    seen: set[str] = set()
    for p in primary["_pairs_full"]:
        cand_recs = C.load_decode(p["_cand_file"])
        for row in p["_per_prompt"]:
            if row["identical"] or row["first_divergence_index"] is None:
                continue
            key = row["key"]
            d = row["first_divergence_index"]
            ref = ref_recs.get(key)
            cand = cand_recs.get(key)
            if ref is None or cand is None:
                continue
            ref_ids = ref["completion_token_ids"]
            cand_ids = cand["completion_token_ids"]
            eos = _natural_eos_pos(ref_ids)
            is_real = d < eos
            if key not in seen:
                seen.add(key)
                if is_real:
                    real_response += 1
                else:
                    freerun_tail += 1
            if d < len(ref_ids) and d < len(cand_ids) and (ref_ids[d] in EOS_IDS or cand_ids[d] in EOS_IDS):
                div_tok_is_eos += 1
            if is_real and len(examples) < 12:
                examples.append({"key": key, "cand_run": p["_cand_run"], "onset": d,
                                 "natural_eos_pos": eos, "ref_tok": ref_ids[d],
                                 "cand_tok": cand_ids[d] if d < len(cand_ids) else None})
    return {
        "n_divergent_prompts_real_response": real_response,    # pre-EOS: could move an answer
        "n_divergent_prompts_freerun_tail": freerun_tail,      # post-EOS: invisible to the eval
        "n_divergence_tokens_are_eos": div_tok_is_eos,
        "any_real_response_divergence": real_response > 0,
        "examples": examples,
    }


# ========================================================================== #
# GPQA seed-swing attribution (analytic; uses generation_config + measured determinism)
# ========================================================================== #
def gpqa_attribution(determinism_rate: float | None, all_flips_near_tie: bool | None) -> dict[str, Any]:
    """Decompose fern #587's 2-problem GPQA-D base_fullhead swing into (a) eval-harness SAMPLING
    variance vs (b) greedy-decode run-to-run non-determinism. lewtun #31: downstream evals use the
    model's generation_config sampling (do_sample=True, temp 1.0), NOT greedy -> seed changes the
    sampling RNG -> different completions -> ± knife-edge problems is EXPECTED. The greedy-decode
    determinism measured by THIS card is a separate axis: even a perfectly deterministic greedy
    decode shows this swing because the eval samples."""
    n = GPQA["n"]
    swing = GPQA["swing_problems"]
    p = (GPQA["default_correct"] + GPQA["seed12345_correct"]) / (2 * n)
    # Sampling-only seed-to-seed swing scale: two INDEPENDENT sampled runs, per-question Bernoulli.
    # Var(correct_a - correct_b) = 2 * sum p_i(1-p_i) <= 2 * n * 0.25 (worst case); a moment estimate
    # at the pooled accuracy p gives sigma_diff ~ sqrt(2 n p (1-p)).
    sigma_diff_moment = math.sqrt(2 * n * p * (1 - p))
    sigma_diff_worst = math.sqrt(2 * n * 0.25)
    swing_in_sigma = swing / sigma_diff_moment if sigma_diff_moment else None
    eval_samples = bool(GEN_CONFIG["do_sample"]) and float(GEN_CONFIG["temperature"]) > 0.0
    # the greedy-decode (non-determinism) component the swing could carry: bounded by the measured
    # cross-process greedy flip rate. ~1.0 determinism => ~0 contribution.
    decode_nondeterminism_component = (1.0 - determinism_rate) if determinism_rate is not None else None
    is_sampling_not_nondeterminism = bool(
        eval_samples
        and (determinism_rate is None or determinism_rate >= 0.999 or bool(all_flips_near_tie))
        and (swing_in_sigma is None or swing_in_sigma <= 1.0)
    )
    return {
        "swing_problems": swing,
        "swing_abs_accuracy": GPQA["swing_abs"],
        "pooled_accuracy": p,
        "eval_uses_sampling": eval_samples,
        "generation_config": GEN_CONFIG,
        "sampling_sigma_diff_problems_moment": round(sigma_diff_moment, 2),
        "sampling_sigma_diff_problems_worstcase": round(sigma_diff_worst, 2),
        "swing_in_sampling_sigma": round(swing_in_sigma, 3) if swing_in_sigma is not None else None,
        "greedy_decode_determinism_rate": determinism_rate,
        "decode_nondeterminism_component": decode_nondeterminism_component,
        "gpqa_seedswing_is_sampling_not_nondeterminism": is_sampling_not_nondeterminism,
    }


# ========================================================================== #
# synthesis
# ========================================================================== #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and not math.isfinite(x))


def synthesize(primary: dict, cold: dict | None, warm: dict | None, coldwarm: dict | None,
               tie: dict | None, eos: dict, gpqa: dict, peak_vram: float,
               n_servers_ok: int) -> dict[str, Any]:
    det_rate = primary["matched_state_per_step_rate_pooled"]
    freerun_allN = primary["freerun_allN_positional_rate"]
    seq_allN = primary["allN_sequence_exact_rate"]
    literally_bit_exact = bool(primary["all_pairs_greedy_identical"])

    # near-tie characterisation: a flip is a bf16 near-tie if its M=1 margin sits in the miss cluster
    # well below the control rank0-rank1 separation (the #576/#566 disjoint-distribution signature).
    median_miss = (tie or {}).get("median_miss_margin")
    median_ctrl = (tie or {}).get("median_control_margin")
    ctrl_p10 = (tie or {}).get("control_margin_p10")
    miss_p90 = (tie or {}).get("miss_margin_p90")
    n_exact_tie = (tie or {}).get("n_exact_tie")
    # all flips near-tie iff the worst (p90) miss margin is still below the typical control floor
    all_flips_near_tie: bool | None = None
    if tie and miss_p90 is not None and ctrl_p10 is not None:
        all_flips_near_tie = miss_p90 <= ctrl_p10
    elif tie and median_miss is not None and median_ctrl is not None:
        all_flips_near_tie = median_miss < median_ctrl
    if det_rate is not None and det_rate >= 1.0:
        all_flips_near_tie = True  # no flips at all -> vacuously near-tie

    any_real_response = eos["any_real_response_divergence"]

    # operative_identity_bit_stable: "determinism rate >= operative threshold / effectively 1.0
    # modulo a near-tie epsilon". TRUE iff there are no GENUINE (large-margin) flips AND no flip
    # changes a real-response (pre-EOS) answer token. A handful of bf16 near-tie reorders in the
    # ignore_eos free-run tail still counts as bit-stable-modulo-epsilon.
    operative_identity_bit_stable = bool(
        (det_rate is not None and det_rate >= 0.999)
        and (all_flips_near_tie in (True, None))
        and (not any_real_response)
    )

    return {
        "n_servers_ok": n_servers_ok,
        "regime_headline": primary["label"],
        "served_argmax_determinism_rate": det_rate,                  # HEADLINE (warm matched-state)
        "served_argmax_determinism_rate_freerun_allN": freerun_allN,  # literal, walk-off-confounded
        "allN_sequence_exact_rate": seq_allN,
        "literally_bit_exact_all_pairs": literally_bit_exact,
        "matched_state_hazard": primary["matched_state_per_step_hazard_pooled"],
        "matched_state_trials": primary["matched_state_trials_pooled"],
        "matched_state_failures": primary["matched_state_failures_pooled"],
        # regime breakdown
        "warm_xproc_matched_state_rate": (warm or {}).get("matched_state_per_step_rate_pooled"),
        "warm_xproc_allN_sequence_exact": (warm or {}).get("allN_sequence_exact_rate"),
        "warm_xproc_all_pairs_identical": (warm or {}).get("all_pairs_greedy_identical"),
        "cold_xproc_matched_state_rate": (cold or {}).get("matched_state_per_step_rate_pooled"),
        "cold_xproc_allN_sequence_exact": (cold or {}).get("allN_sequence_exact_rate"),
        "cold_xproc_all_pairs_identical": (cold or {}).get("all_pairs_greedy_identical"),
        "within_coldwarm_matched_state_rate": (coldwarm or {}).get("matched_state_per_step_rate_pooled"),
        "within_coldwarm_all_servers_identical": (coldwarm or {}).get("all_servers_cold_eq_warm"),
        # flip characterisation
        "all_flips_near_tie": all_flips_near_tie,
        "n_exact_tie_flips": n_exact_tie,
        "tie_median_miss_margin": median_miss,
        "tie_median_control_margin": median_ctrl,
        "tie_miss_margin_p90": miss_p90,
        "tie_control_margin_p10": ctrl_p10,
        "tie_separation_ratio": (tie or {}).get("separation_ratio"),
        # EOS / answer-token impact
        "n_real_response_divergent_prompts": eos["n_divergent_prompts_real_response"],
        "n_freerun_tail_divergent_prompts": eos["n_divergent_prompts_freerun_tail"],
        "any_real_response_divergence": any_real_response,
        # GPQA attribution
        "gpqa_seedswing_is_sampling_not_nondeterminism":
            gpqa["gpqa_seedswing_is_sampling_not_nondeterminism"],
        "gpqa_swing_in_sampling_sigma": gpqa["swing_in_sampling_sigma"],
        "gpqa_eval_uses_sampling": gpqa["eval_uses_sampling"],
        # VERDICTS
        "operative_identity_bit_stable": operative_identity_bit_stable,
        "peak_vram_gb": peak_vram,
    }


def _print_summary(s: dict[str, Any]) -> None:
    line = "=" * 8 + " PR #596 — base_fullhead OPERATIVE GREEDY CROSS-PROCESS DETERMINISM " + "=" * 8
    print("\n" + line, flush=True)
    print(f"  servers OK = {s['n_servers_ok']}   headline regime = {s['regime_headline']}", flush=True)
    print(f"  >>> served_argmax_determinism_rate (matched-state) = {s['served_argmax_determinism_rate']}",
          flush=True)
    print(f"      free-run all-N positional = {s['served_argmax_determinism_rate_freerun_allN']}   "
          f"all-N sequence-exact = {s['allN_sequence_exact_rate']}", flush=True)
    print(f"      literally bit-exact (all pairs GREEDY_IDENTICAL) = {s['literally_bit_exact_all_pairs']}",
          flush=True)
    print(f"  warm xproc: matched={s['warm_xproc_matched_state_rate']} "
          f"seqAllN={s['warm_xproc_allN_sequence_exact']} allPairsIdent={s['warm_xproc_all_pairs_identical']}",
          flush=True)
    print(f"  cold xproc: matched={s['cold_xproc_matched_state_rate']} "
          f"seqAllN={s['cold_xproc_allN_sequence_exact']} allPairsIdent={s['cold_xproc_all_pairs_identical']}",
          flush=True)
    print(f"  within cold->warm: matched={s['within_coldwarm_matched_state_rate']} "
          f"allIdent={s['within_coldwarm_all_servers_identical']}", flush=True)
    print(f"  flips near-tie = {s['all_flips_near_tie']}  (miss p90 {s['tie_miss_margin_p90']} vs "
          f"control p10 {s['tie_control_margin_p10']}; exact-ties {s['n_exact_tie_flips']})", flush=True)
    print(f"  real-response (pre-EOS) divergent prompts = {s['n_real_response_divergent_prompts']}  "
          f"free-run-tail = {s['n_freerun_tail_divergent_prompts']}", flush=True)
    print(f"  >>> operative_identity_bit_stable = {s['operative_identity_bit_stable']}", flush=True)
    print(f"  >>> gpqa_seedswing_is_sampling_not_nondeterminism = "
          f"{s['gpqa_seedswing_is_sampling_not_nondeterminism']} "
          f"(swing {s['gpqa_swing_in_sampling_sigma']}sigma, eval-samples={s['gpqa_eval_uses_sampling']})",
          flush=True)
    print(f"  peak VRAM = {s['peak_vram_gb']:.2f} GB", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# wandb
# ========================================================================== #
def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[det] wandb unavailable: {exc}", flush=True)
        return None
    cfg_name = report.get("config", "base_specoff")
    run = init_wandb_run(
        job_type="systems-profile", agent="denken",
        name=args.wandb_name or f"denken/served-identity-determinism-{cfg_name}",
        group=args.wandb_group or "served-identity-determinism",
        tags=["served-identity", "determinism", "cross-process", "319", "operative",
              cfg_name, report.get("spec_mode", ""), "local-a10g", "analysis-only", "pr596"],
        notes=f"PR #596 [{cfg_name}: {report.get('config_label', '')}]: is the served greedy "
              "identity bit-stable across fresh server (re)starts? Cross-process determinism + "
              "flip-margin + EOS impact + GPQA seed-swing attribution.",
        config={
            "config": cfg_name, "config_label": report.get("config_label"),
            "submission": report.get("submission"), "model_dir": report.get("model_dir"),
            "num_servers": args.num_servers, "passes_per_server": args.passes_per_server,
            "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": SEED,
            "concurrency": 1, "spec_mode": report.get("spec_mode"), "gpu_mem_util": C.GPU_MEM_UTIL,
            "anchor_spec_on_tps": ANCHOR_SPEC_ON_TPS, "anchor_ppl": ANCHOR_PPL,
        },
    )
    if run is None:
        return None
    s = report["synthesis"]
    summary = {k: v for k, v in s.items() if _finite(v) or isinstance(v, (bool, str))}
    summary["primary_metric"] = s["served_argmax_determinism_rate"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="served-identity-determinism-report",
                      artifact_type="determinism-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


# ========================================================================== #
# main
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check (2 servers, 4x24)")
    ap.add_argument("--config", choices=["base_specoff", "base_mtp", "int4g128_specoff"],
                    default="base_specoff",
                    help="which served stack's run-to-run determinism to measure")
    ap.add_argument("--num-servers", type=int, default=5, help="fresh server boots (>=2; >=3 for a rate)")
    ap.add_argument("--passes-per-server", type=int, default=2, help="1=cold only; 2=cold+warm")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--tie-probe-limit", type=int, default=64)
    ap.add_argument("--no-tie-probe", action="store_true")
    ap.add_argument("--wall-budget-min", type=float, default=78.0,
                    help="stop booting NEW servers past this wall-clock; analyse what completed")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_servers = min(args.num_servers, 2)
        args.num_prompts = min(args.num_prompts, 4)
        args.output_len = min(args.output_len, 24)
        args.no_tie_probe = True

    from scripts.local_validation import harness, paths
    for note in paths.prepare_local_gpu_env():
        print(f"[det] {note}", flush=True)

    cfg = build_configs()[args.config]
    cfg_submission = cfg["submission"]
    cfg_env = cfg["extra_env"]
    cfg_prefix = cfg["prefix"]
    print(f"[det] config={args.config} ({cfg['label']}) submission={cfg_submission.name} "
          f"spec_mode={cfg['spec_mode']} prefix={cfg_prefix!r}", flush=True)
    manifest = harness.load_manifest(cfg_submission)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    budget_s = args.wall_budget_min * 60.0

    # ---- Phase 1: N fresh-server captures (each a separate process + model reload) ----
    servers: list[dict] = []
    for i in range(args.num_servers):
        if i > 0 and (time.time() - t_start) > budget_s:
            print(f"[det] wall budget {args.wall_budget_min}min reached after {i} servers "
                  f"-> analysing what completed", flush=True)
            break
        if i > 0:
            free = wait_gpu_free()
            print(f"[det] gpu settle before run{i}: used={free if free is not None else 'TIMEOUT'} MiB",
                  flush=True)
        s = capture_one_server(harness, paths, server_python, submission=cfg_submission,
                               extra_env=cfg_env, prefix=cfg_prefix, port=args.port, run_idx=i,
                               num_prompts=args.num_prompts, output_len=args.output_len,
                               passes=args.passes_per_server)
        servers.append(s)
        print(f"[det] {cfg_prefix}run{i} done booted={s['booted']} reused={s.get('reused')} "
              f"files={list(s.get('files', {}))} ({time.time() - t_start:.0f}s)", flush=True)

    # ---- Phase 2: cross-process comparison per regime ----
    def regime_files(tag: str) -> list[str]:
        return [s["files"][tag] for s in servers if tag in s.get("files", {})]

    warm_files = regime_files("warm")
    cold_files = regime_files("cold")
    warm = cross_process_metrics(paths, warm_files, "warm") if len(warm_files) >= 2 else None
    cold = cross_process_metrics(paths, cold_files, "cold") if len(cold_files) >= 2 else None
    coldwarm = within_coldwarm(paths, servers) if args.passes_per_server >= 2 else None
    primary = warm or cold
    if primary is None:
        print("[det] FATAL: fewer than 2 comparable fresh captures — cannot compute a rate", flush=True)
        return 2
    n_servers_ok = primary["n_runs"]

    # ---- Phase 3: flip-margin probe at the cross-process divergence positions ----
    tie = None
    if not args.no_tie_probe:
        div_jobs, ctrl_jobs = build_xproc_jobs(primary, args.tie_probe_limit)
        if div_jobs or ctrl_jobs:
            free = wait_gpu_free()
            print(f"[det] gpu settle before probe: used={free if free is not None else 'TIMEOUT'} MiB",
                  flush=True)
            tie = run_flip_probe(harness, server_python, submission=cfg_submission,
                                 extra_env=cfg_env, prefix=cfg_prefix, port=args.port,
                                 jobs=div_jobs + ctrl_jobs)
        else:
            print("[det] no cross-process divergences -> flip probe skipped (perfectly identical)",
                  flush=True)
            tie = {"rows": [], "n_miss_probed": 0, "note": "no divergences"}

    # ---- Phase 4: EOS / answer-token impact ----
    eos = eos_answer_impact(primary)

    # ---- Phase 5: GPQA seed-swing attribution ----
    det_rate = primary["matched_state_per_step_rate_pooled"]
    miss_p90 = (tie or {}).get("miss_margin_p90")
    ctrl_p10 = (tie or {}).get("control_margin_p10")
    flips_near_tie = (miss_p90 <= ctrl_p10) if (miss_p90 is not None and ctrl_p10 is not None) else None
    gpqa = gpqa_attribution(det_rate, flips_near_tie)

    peak_vram = max((s.get("peak_vram_gb") or 0.0) for s in servers) if servers else 0.0
    synthesis = synthesize(primary, cold, warm, coldwarm, tie, eos, gpqa, peak_vram, n_servers_ok)

    # strip heavy/unserialisable internals
    for r in (cold, warm):
        if r is not None:
            r.pop("_pairs_full", None)

    report = {
        "pr": 596, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "config": args.config, "config_label": cfg["label"],
        "submission": str(cfg_submission), "model_dir": cfg["model_dir"], "spec_mode": cfg["spec_mode"],
        "num_servers_requested": args.num_servers, "passes_per_server": args.passes_per_server,
        "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": SEED,
        "anchor_spec_on_tps": ANCHOR_SPEC_ON_TPS, "anchor_ppl": ANCHOR_PPL,
        "spec_off_ar_tps": SPEC_OFF_AR_TPS,
        "servers": [{k: v for k, v in s.items()} for s in servers],
        "warm_xproc": warm, "cold_xproc": cold, "within_coldwarm": coldwarm,
        "tie_probe": tie, "eos_impact": eos, "gpqa_attribution": gpqa,
        "synthesis": synthesis,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    suffix = "smoke" if args.smoke else args.config
    out_file = OUT_ROOT / f"determinism_report_{suffix}.json"
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[det] report -> {out_file} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[det] wandb run id={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
