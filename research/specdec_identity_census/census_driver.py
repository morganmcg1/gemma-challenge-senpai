#!/usr/bin/env python
"""PR #576 denken — SERVED spec-dec #319 greedy-identity census (LOCAL A10G, analysis-only,
NO HF fire). Extends the per-component decode-step identity census (#562 attention, #555
head-BW, #569 overhead) from the decode STEP to the spec-dec WRAPPER — the last un-censused
leg of the #319 contract.

THE QUESTION: spec-dec is assumed to pass the #319 strict byte-exact greedy-identity contract
"BY CONSTRUCTION" (greedy exact-verify emits the target's argmax regardless of drafter). But
the M=8 verify forward runs at multi-position granularity, so its logits are NOT guaranteed
bit-identical to the M=1 no-spec decode logits at bf16 near-ties (the #555/#562/#566 mechanism:
faster <=> reorders the reduction <=> not bit-identical). fern #566 just refuted an analogous
"1.0-by-construction" identity claim (candidate-verify: per-step 0.99406 -> sequence 0.140625
byte-exact -> #319 FAILS served, misses are bf16 ties). This card MEASURES the spec-dec leg.

THREE ARMS on the SAME base_fullhead serve recipe (stock base-int4 + full native 262k bf16 head,
prune OFF; same substrate as the #572/#573/#575 speed companions). Only the SPECULATIVE_CONFIG
varies:

  * REF   (no-spec): SENPAI_REFERENCE_MODE=1 clears SPECULATIVE_CONFIG -> plain M=1 AR greedy on
    THIS submission's own engine/kernels/quant. The canonical #319 reference (gen_greedy_reference
    --spec-off contract). Captured TWICE for the self-determinism assert.
  * MTP   (ship drafter): SPECULATIVE_CONFIG mtp K=7 -> M=8 verify (surgical-357's config).
  * NGRAM (always-loads): SPECULATIVE_CONFIG ngram -> M=8 verify, prompt-lookup proposer.

IDENTITY (headline, via the OFFICIAL greedy_identity verifier): REF vs MTP and REF vs NGRAM
completion token ids, per prompt -> SEQUENCE byte-exact rate + per-step (positional) rate +
first-divergence onset. Misses are diagnosed (bf16-tie reorder vs genuine precision divergence)
via the onset distribution + an on-stack prompt_logprobs flip-margin probe.

LOCAL only: analysis_only, official_tps=0, no HF Job, no --launch, no submission, no served-file
change. base_fullhead is reached purely by serve-env overrides on submissions/fa2sw_strict_surgical357.

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/specdec_identity_census/census_driver.py --smoke --no-wandb
Full:
  CUDA_VISIBLE_DEVICES=0 python research/specdec_identity_census/census_driver.py \
    --num-prompts 128 --output-len 512 \
    --wandb_name denken/specdec-identity-census --wandb_group base-fullhead-specdec-identity-census
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
OUT_ROOT = HERE

# base_fullhead substrate: my OWN stock int4 QAT snapshot (NO baked bucket).
MODEL_DIR = (
    "/senpai-run/home/student-denken/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)

SEED = 1
FULL_LM_HEAD_ROWS = 262144
# Full 262k bf16 head needs more KV headroom than the pruned ship; fern #566 found the
# shipped base_fullhead boots conc=1 only with util raised. Applied to ALL arms (util only
# changes KV capacity, not single-stream identity). Override via CENSUS_GPU_MEM_UTIL.
GPU_MEM_UTIL = os.environ.get("CENSUS_GPU_MEM_UTIL", "0.92")

# Cited anchors (NOT re-derived here) — provided in the PR body.
ANCHOR_NOSPEC_TPS = 252.69          # base_fullhead no-spec served (wirbel #553 83jiwjr9)
ANCHOR_NOSPEC_PPL = 2.0057
SHIP_TPS = 375.857                  # surgical-357 official ship (the speed gate spec must clear to matter)
FREE_FLOOR_TPS = 311.25             # magically-free floor (lawine #554)
BASELINE_TPS = 481.53               # public frontier, untouched
# fern #566 candidate-verify precedent (THIS stack, M=8 verify, the template):
CV566_PER_STEP = 0.9940561590489855
CV566_SEQUENCE_EXACT = 0.140625

# ngram proposer config (vLLM 0.22): same M=8 verify granularity as the ship MTP (K=7),
# standard prompt-lookup window. method=ngram needs no drafter model.
NGRAM_K = 7
NGRAM_LOOKUP_MAX = 4
NGRAM_LOOKUP_MIN = 1
NGRAM_SPEC_CONFIG = json.dumps({
    "method": "ngram",
    "num_speculative_tokens": NGRAM_K,
    "prompt_lookup_max": NGRAM_LOOKUP_MAX,
    "prompt_lookup_min": NGRAM_LOOKUP_MIN,
})


def base_fullhead_env() -> dict[str, str]:
    """The base_fullhead serve recipe: stock base-int4 + full native 262k bf16 head, prune OFF.

    Overrides the surgical-357 manifest's baked-bucket + 12k-pruned-head env back to the
    quality-safe full-head substrate. Everything else in the manifest (Marlin int4 body,
    fa-sliding 2D attention, splitkv-verify->2D surgical, fused-argmax, onegraph) is KEPT so
    the census runs on the ACTUAL served stack."""
    return {
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "LM_HEAD_FULL_REQUIRE": "1",        # hard-assert the served head carries all 262,144 rows
        "PCK04_KEEPSET": "",                # no pruned keepset
        "LOCAL_MODEL_DIR": MODEL_DIR,       # stock int4 snapshot (skips the baked WEIGHTS_BUCKET sync)
        "PLE_FOLD_TARGET_MODEL": MODEL_DIR,
        "PLE_FOLD_EMBED_SCALE": "1",
        "GPU_MEMORY_UTILIZATION": GPU_MEM_UTIL,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # PyTorch-native lowest-index argmax (#560/#566 tie-break)
    }


def arm_env(arm: str) -> dict[str, str]:
    env = base_fullhead_env()
    if arm == "ref":
        # The canonical #319 reference: clear the drafter -> plain M=1 AR greedy on this engine.
        env["SENPAI_REFERENCE_MODE"] = "1"
    elif arm == "mtp":
        # Ship drafter: keep the manifest's mtp K=7 SPECULATIVE_CONFIG + DRAFTER_BUCKET as-is.
        pass
    elif arm == "ngram":
        # Always-loads prompt-lookup proposer; needs no drafter model.
        env["SPECULATIVE_CONFIG"] = NGRAM_SPEC_CONFIG
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return env


# ========================================================================== #
# decode capture (boot server, capture token ids through the official decoder)
# ========================================================================== #
def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
            if vals:
                peak["mib"] = max(peak["mib"], max(vals))
        except (OSError, subprocess.SubprocessError):
            pass
        stop.wait(2.0)


PLUMBING_NEEDLES = ["LM_HEAD_FULL_REQUIRE", "Speculative", "speculative_config",
                    "SENPAI_REFERENCE_MODE active: clearing SPECULATIVE_CONFIG"]


def run_arm(harness: Any, paths: Any, *, server_python: Path, arm: str, num_prompts: int,
            output_len: int, port: int, passes: int, do_tie_probe: bool,
            divergence_jobs: list[dict] | None, reuse: bool = False) -> dict[str, Any]:
    """Boot the base_fullhead stack for one arm, capture `passes` decode files, and (for the
    spec-off ref arm) optionally run the on-stack flip-margin probe at known divergence positions.
    Returns capture file paths + peak vram + (optional) tie-probe rows."""
    log_path = OUT_ROOT / f"server_{arm}.log"
    extra_env = arm_env(arm)
    result: dict[str, Any] = {"arm": arm, "capture_files": [], "summaries": []}

    # resume: if every pass output is already a complete capture, reuse it and skip the
    # boot+capture. The mtp/ngram arms are deterministic greedy decodes -> reuse is identical to
    # recapture; only the ref arm must be re-captured fresh (its self-determinism passes share one
    # server boot), so the ref path never sets reuse.
    if reuse:
        tags = [(arm if passes == 1 else f"{arm}_{chr(ord('a') + p)}") for p in range(passes)]
        files = [(OUT_ROOT / f"decode_{t}.jsonl", OUT_ROOT / f"decode_{t}.summary.json") for t in tags]
        if all(_decode_complete(o, s, num_prompts, output_len) for o, s in files):
            print(f"[census] [{arm}] reusing {passes} complete capture(s) "
                  f"({num_prompts}x{output_len}) — skipping recapture", flush=True)
            result["capture_files"] = [str(o) for o, _ in files]
            result["reused"] = True
            result["peak_vram_gb"] = 0.0          # same config; the fresh ref arm sets the reported peak
            result["log_path"] = str(log_path)
            result["plumbing"] = grep_log(str(log_path), PLUMBING_NEEDLES)
            return result

    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            for p in range(passes):
                tag = f"{arm}" if passes == 1 else f"{arm}_{chr(ord('a') + p)}"
                out_file = OUT_ROOT / f"decode_{tag}.jsonl"
                summary_file = OUT_ROOT / f"decode_{tag}.summary.json"
                print(f"[census] [{arm}] capture pass {p + 1}/{passes} "
                      f"{num_prompts}x{output_len} conc=1 -> {out_file.name}", flush=True)
                summary = harness.capture_decode(
                    server_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_file, summary_file=summary_file,
                    num_prompts=num_prompts, output_len=output_len, seed=SEED,
                    tokenizer=paths.TOKENIZER, dataset=paths.EVAL_PROMPTS, timeout_s=5400,
                )
                result["capture_files"].append(str(out_file))
                result["summaries"].append(summary)
            if do_tie_probe and divergence_jobs:
                result["tie_probe"] = flip_margin_probe(
                    srv.base_url, srv.served_model_name, divergence_jobs)
    finally:
        stop.set()
        sampler.join(timeout=5)
    result["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    result["log_path"] = str(log_path)
    result["plumbing"] = grep_log(str(log_path), PLUMBING_NEEDLES)
    return result


def grep_log(log_path: str, needles: list[str]) -> dict[str, bool]:
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return {n: False for n in needles}
    return {n: (n in text) for n in needles}


# ========================================================================== #
# official #319 verifier
# ========================================================================== #
def load_decode(path: str) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs[d["id"]] = d
    return recs


def _decode_complete(out_file: Path, summary_file: Path, n: int, output_len: int) -> bool:
    """True iff a decode capture is already complete and reusable: the summary records exactly
    n prompts and n*output_len completion tokens, and the jsonl carries exactly n parseable
    records each with `id` + `completion_token_ids`. Guards the --reuse-complete resume path
    against a partial/truncated file (e.g. the disk-full crash that wrote 0 ref_b records)."""
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


def verify_pair(paths: Any, ref_file: str, cand_file: str) -> dict[str, Any]:
    """Run the OFFICIAL greedy_identity verifier and distill the #319 census metrics."""
    gi = paths.import_greedy_identity()
    report = gi.compare_files(ref_file, cand_file)
    n = report.num_prompts_compared
    seq_exact_rate = report.num_identical / n if n else None
    # FREE-RUN positional rate: total_divergent_tokens counts EVERY position where the two
    # free-running greedy sequences differ. Once they diverge at onset they walk off, so this
    # is dominated by the post-cascade tail -> LOW (~0.5) and NOT comparable to fern #566's
    # 0.994. Kept as a secondary diagnostic only.
    freerun_pos_rate = (1.0 - report.total_divergent_tokens / report.total_tokens_compared
                        if report.total_tokens_compared else None)
    onsets = sorted(p.first_divergence_index for p in report.per_prompt
                    if not p.identical and p.first_divergence_index is not None)
    # MATCHED-STATE (teacher-forced) per-step rate = the fern #566-comparable headline. Up to a
    # prompt's first divergence the two sequences share an identical prefix, so positions
    # 0..onset-1 are matched-state ARGMAX AGREEMENTS and `onset` is the first matched-state
    # DISAGREEMENT (a genuine flip given the same state). Positions after onset are NOT
    # matched-state observations (prefixes differ) and are excluded. Survival-MLE of the
    # per-position flip hazard p: trials = sum_div(onset+1) + sum_iden(L); failures = num_div.
    ms_trials = 0
    ms_failures = 0
    for p in report.per_prompt:
        if p.identical:
            ms_trials += p.num_compared
        else:
            d = p.first_divergence_index if p.first_divergence_index is not None else p.num_compared
            ms_trials += d + 1
            ms_failures += 1
    ms_hazard = (ms_failures / ms_trials) if ms_trials else None
    ms_rate = (1.0 - ms_hazard) if ms_hazard is not None else None
    seq_miss = (1.0 - seq_exact_rate) if seq_exact_rate is not None else None
    freerun_step_miss = (1.0 - freerun_pos_rate) if freerun_pos_rate is not None else None
    # headline cascade uses the matched-state hazard (the dramatic ~225x the #566 lesson predicts).
    cascade = (seq_miss / ms_hazard) if (seq_miss is not None and ms_hazard not in (None, 0.0)) else None
    freerun_cascade = (seq_miss / freerun_step_miss) if (seq_miss is not None and freerun_step_miss not in (None, 0.0)) else None
    return {
        "verdict": report.verdict,
        "num_prompts_compared": n,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "sequence_exact_rate": seq_exact_rate,
        "per_step_identity_rate": ms_rate,                  # headline = matched-state (fern #566-comparable)
        "matched_state_per_step_identity_rate": ms_rate,
        "matched_state_per_step_hazard": ms_hazard,
        "matched_state_trials": ms_trials,
        "matched_state_failures": ms_failures,
        "freerun_positional_identity_rate": freerun_pos_rate,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "sequence_miss_rate": seq_miss,
        "per_step_miss_rate": ms_hazard,
        "per_step_to_sequence_cascade_factor": cascade,
        "freerun_cascade_factor": freerun_cascade,
        "onset_min": onsets[0] if onsets else None,
        "onset_median": int(statistics.median(onsets)) if onsets else None,
        "onset_max": onsets[-1] if onsets else None,
        "onset_count": len(onsets),
        "_per_prompt": [
            {"key": p.key, "identical": p.identical,
             "first_divergence_index": p.first_divergence_index,
             "ref_len": p.ref_len, "cand_len": p.cand_len,
             "num_divergent_tokens": p.num_divergent_tokens}
            for p in report.per_prompt
        ],
    }


def build_divergence_jobs(ref_recs: dict[str, dict], cand_recs: dict[str, dict],
                          verdict_rows: list[dict], drafter: str, limit: int) -> list[dict]:
    """For each divergent prompt, the prefix to probe and the (ref_tok, cand_tok) at the first
    divergence — the exact position where the M=8 verify argmax differs from M=1."""
    jobs: list[dict] = []
    for row in verdict_rows:
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
        if d >= len(ref_ids) or d >= len(cand_ids):
            continue
        prefix = list(ref["prompt_token_ids"]) + list(ref_ids[:d + 1])  # ends WITH ref_tok
        jobs.append({
            "drafter": drafter, "key": key, "div_index": d,
            "ref_tok": ref_ids[d], "cand_tok": cand_ids[d], "prefix": prefix,
            "kind": "miss",
        })
        if len(jobs) >= limit:
            break
    return jobs


def build_control_jobs(ref_recs: dict[str, dict], verdict_rows: list[dict], limit: int) -> list[dict]:
    """Control: matched (non-divergent) positions in identical prompts -> the typical
    rank0-vs-rank1 margin (the non-tie baseline)."""
    jobs: list[dict] = []
    identical_keys = [r["key"] for r in verdict_rows if r["identical"]]
    for key in identical_keys:
        ref = ref_recs.get(key)
        if ref is None:
            continue
        ref_ids = ref["completion_token_ids"]
        if len(ref_ids) < 64:
            continue
        pos = len(ref_ids) // 2  # a deterministic mid-sequence matched position
        prefix = list(ref["prompt_token_ids"]) + list(ref_ids[:pos + 1])
        jobs.append({"drafter": "control", "key": key, "div_index": pos,
                     "ref_tok": ref_ids[pos], "cand_tok": None, "prefix": prefix, "kind": "control"})
        if len(jobs) >= limit:
            break
    return jobs


def cross_tab(self_det_pp: list[dict], mtp_pp: list[dict], ngram_pp: list[dict]) -> dict[str, Any]:
    """Prompt-level chaos-floor cross-tabulation — the load-bearing diagnosis.

    self_det partitions prompts into CHAOTIC (ref_a != ref_b: an intrinsic bf16 near-tie that
    the fast stack flips run-to-run even with NO spec change) vs STABLE (ref_a == ref_b). A
    spec arm that diverges ONLY on chaotic prompts adds NOTHING beyond the stack's own
    non-determinism (its #319 'miss' is the #555/#562 reduction-order floor, not the drafter).
    A spec arm that diverges on a STABLE prompt has NEWLY broken identity -> spec-specific."""
    chaotic = {p["key"] for p in self_det_pp if not p["identical"]}
    stable = {p["key"] for p in self_det_pp if p["identical"]}

    def classify(pp: list[dict], label: str) -> dict[str, Any]:
        div = {p["key"] for p in pp if not p["identical"]}
        on_stable = sorted(div & stable)
        on_chaotic = sorted(div & chaotic)
        return {
            f"{label}_n_divergent": len(div),
            f"{label}_divergent_on_stable": on_stable,
            f"{label}_n_divergent_on_stable": len(on_stable),
            f"{label}_n_divergent_on_chaotic": len(on_chaotic),
            # the spec arm introduced NO divergence beyond the chaos floor
            f"{label}_divergence_subset_of_chaos": (len(on_stable) == 0),
        }

    out: dict[str, Any] = {
        "n_chaotic": len(chaotic), "n_stable": len(stable),
        "chaotic_keys": sorted(chaotic),
    }
    out.update(classify(mtp_pp, "mtp"))
    out.update(classify(ngram_pp, "ngram"))
    return out


# ========================================================================== #
# on-stack flip-margin probe (prompt_logprobs at the divergence position)
# ========================================================================== #
def _completions_prompt_logprobs(base_url: str, model: str, token_ids: list[int],
                                 top: int = 20, timeout_s: int = 120) -> Any:
    payload = {
        "model": model, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": top, "add_special_tokens": False, "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def flip_margin_probe(base_url: str, model: str, jobs: list[dict]) -> dict[str, Any]:
    """At each divergence position, read the M=1 (no-spec, spec-off server) logprob margin
    between the no-spec token (ref_tok, the argmax) and the spec-emitted token (cand_tok). A
    near-zero margin == a bf16-tie reorder; a large margin == genuine precision divergence.
    Control positions give the typical (non-tie) rank0-rank1 margin."""
    rows: list[dict] = []
    for job in jobs:
        rec = {"drafter": job["drafter"], "key": job["key"], "div_index": job["div_index"],
               "kind": job["kind"], "ref_tok": job["ref_tok"], "cand_tok": job["cand_tok"]}
        try:
            resp = _completions_prompt_logprobs(base_url, model, job["prefix"])
            pls = resp["choices"][0]["prompt_logprobs"]
            last = pls[-1]  # distribution that PRODUCED the final prefix token (ref_tok)
            # vLLM keys prompt_logprobs by stringified token id.
            def lp(tid: int | None):
                if tid is None:
                    return None
                e = last.get(str(tid))
                return e.get("logprob") if isinstance(e, dict) else None
            ref_lp = lp(job["ref_tok"])
            # rank-0 / rank-1 entries from the returned top-k
            ranked = sorted(
                ((e.get("logprob"), int(t)) for t, e in last.items() if isinstance(e, dict)),
                reverse=True)
            rec["rank0_tok"] = ranked[0][1] if ranked else None
            rec["rank0_lp"] = ranked[0][0] if ranked else None
            rec["rank1_tok"] = ranked[1][1] if len(ranked) > 1 else None
            rec["rank1_lp"] = ranked[1][0] if len(ranked) > 1 else None
            rec["ref_lp"] = ref_lp
            if job["kind"] == "control":
                # typical separation between the two top tokens at a matched position
                rec["margin"] = (ranked[0][0] - ranked[1][0]) if len(ranked) > 1 else None
                rec["ref_is_argmax"] = (rec["rank0_tok"] == job["ref_tok"])
            else:
                cand_lp = lp(job["cand_tok"])
                rec["cand_lp"] = cand_lp
                rec["cand_in_topk"] = cand_lp is not None
                rec["ref_is_argmax"] = (rec["rank0_tok"] == job["ref_tok"])
                # flip-margin = how far the spec token is below the no-spec argmax under M=1
                rec["margin"] = (ref_lp - cand_lp) if (ref_lp is not None and cand_lp is not None) else None
                rec["exact_tie"] = (ref_lp is not None and cand_lp is not None and ref_lp == cand_lp)
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            rec["error"] = repr(exc)
        rows.append(rec)

    def _margins_kind(kind: str) -> list[float]:
        return [r["margin"] for r in rows
                if r.get("kind") == kind and isinstance(r.get("margin"), (int, float))]

    def _margins_drafter(drafter: str) -> list[float]:
        return [r["margin"] for r in rows
                if r.get("drafter") == drafter and isinstance(r.get("margin"), (int, float))]
    # "miss" rows = every spec/self-det divergence position (build_divergence_jobs sets kind=miss);
    # "control" rows = matched non-divergent positions. Break the misses out by drafter so the
    # spec (mtp/ngram) near-tie margin and the chaos-floor (selfdet) near-tie margin are comparable.
    miss_m = _margins_kind("miss")
    ctrl_m = _margins_kind("control")
    miss_in_topk = [r for r in rows if r.get("kind") == "miss" and r.get("cand_in_topk")]
    miss_total = [r for r in rows if r.get("kind") == "miss" and "error" not in r]
    by_drafter = {}
    for drafter in ("mtp", "ngram", "selfdet"):
        dm = _margins_drafter(drafter)
        by_drafter[drafter] = {
            "n": len(dm),
            "median_margin": statistics.median(dm) if dm else None,
            "max_margin": max(dm) if dm else None,
        }
    return {
        "rows": rows,
        "n_miss_probed": len(miss_total),
        "n_miss_with_margin": len(miss_m),
        "n_miss_cand_in_topk": len(miss_in_topk),
        "n_exact_tie": sum(1 for r in rows if r.get("exact_tie")),
        "median_miss_margin": statistics.median(miss_m) if miss_m else None,
        "median_control_margin": statistics.median(ctrl_m) if ctrl_m else None,
        "miss_margin_p90": (statistics.quantiles(miss_m, n=10)[-1] if len(miss_m) >= 10 else None),
        "control_margin_p10": (statistics.quantiles(ctrl_m, n=10)[0] if len(ctrl_m) >= 10 else None),
        "separation_ratio": (statistics.median(ctrl_m) / statistics.median(miss_m)
                             if miss_m and ctrl_m and statistics.median(miss_m) > 0 else None),
        "by_drafter": by_drafter,
    }


# ========================================================================== #
# synthesis + wandb
# ========================================================================== #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and not math.isfinite(x))


def synthesize(self_det: dict, self_det_bc: dict | None, mtp: dict, ngram: dict, tie: dict | None,
               ctab: dict | None, peak_vram: float) -> dict[str, Any]:
    mtp_seq = mtp["sequence_exact_rate"]
    ngram_seq = ngram["sequence_exact_rate"]
    # The ship would use the MTP drafter -> the #319-by-construction claim is judged on MTP.
    passes_319 = (mtp_seq is not None and mtp_seq >= 0.999)
    # chaos floor: is the no-spec stack itself byte-deterministic at conc=1 greedy?
    chaos_seq = self_det["sequence_exact_rate"]
    chaos_step = self_det["per_step_identity_rate"]
    ref_nondeterministic = (self_det["verdict"] != "GREEDY_IDENTICAL")
    steady_nondeterministic = (self_det_bc["verdict"] != "GREEDY_IDENTICAL") if self_det_bc else None
    # cross-tab: does the spec arm diverge BEYOND the chaos floor (on a self-det-STABLE prompt)?
    mtp_subset_chaos = (ctab or {}).get("mtp_divergence_subset_of_chaos")
    mtp_n_div_stable = (ctab or {}).get("mtp_n_divergent_on_stable")
    ngram_subset_chaos = (ctab or {}).get("ngram_divergence_subset_of_chaos")
    # #319-by-construction RELATIVE to the floor: spec added no divergence the no-spec stack
    # would not also produce. This is the honest 'is spec the cause' test.
    no_div_above_floor = (mtp_n_div_stable == 0) if mtp_n_div_stable is not None else None
    # failure attribution (prompt-level heuristic; read alongside the flip-margins, since a
    # 'spec-specific' divergence on a prompt that was merely stable across N no-spec passes may
    # still be a bf16 near-tie that spec happened to flip and no-spec happened not to).
    if passes_319:
        attribution = "none"            # literally byte-identical, nothing to attribute
    elif no_div_above_floor:
        attribution = "stack-chaos"     # spec divergence is a SUBSET of the no-spec chaos floor
    elif ref_nondeterministic:
        attribution = "spec-and-chaos"  # the no-spec stack AND spec both contribute divergence
    elif no_div_above_floor is False:
        attribution = "spec-specific"   # no-spec was deterministic yet spec still diverged
    else:
        attribution = "unresolved"
    # tie diagnosis: misses are bf16-tie reorders if the no-spec flip-margin at misses is
    # tiny relative to the typical matched-position margin (the #556 ~71x near-tie separation).
    miss_is_tie = None
    if tie is not None and _finite(tie.get("median_miss_margin")) and _finite(tie.get("median_control_margin")):
        mm = tie["median_miss_margin"]
        cm = tie["median_control_margin"]
        # near-tie if the spec token sits within a small logprob gap of the no-spec argmax AND
        # that gap is far smaller than the typical matched-position separation.
        miss_is_tie = bool(mm <= 0.5 and (cm >= 5.0 * max(mm, 1e-9)))
    bd = (tie or {}).get("by_drafter") or {}
    return {
        # headline identity census
        "specdec_mtp_sequence_exact_rate": mtp_seq,
        "specdec_ngram_sequence_exact_rate": ngram_seq,
        "specdec_mtp_per_step_identity_rate": mtp["per_step_identity_rate"],
        "specdec_ngram_per_step_identity_rate": ngram["per_step_identity_rate"],
        "specdec_mtp_cascade_factor": mtp["per_step_to_sequence_cascade_factor"],
        "specdec_ngram_cascade_factor": ngram["per_step_to_sequence_cascade_factor"],
        "per_step_to_sequence_cascade_factor": mtp["per_step_to_sequence_cascade_factor"],
        # secondary (clearly labeled): free-run positional rate is dominated by the post-onset
        # walk-off (~0.5), NOT comparable to fern #566's 0.994; matched-state hazard is the headline.
        "specdec_mtp_freerun_positional_identity_rate": mtp["freerun_positional_identity_rate"],
        "specdec_ngram_freerun_positional_identity_rate": ngram["freerun_positional_identity_rate"],
        "specdec_mtp_matched_state_hazard": mtp["matched_state_per_step_hazard"],
        "specdec_ngram_matched_state_hazard": ngram["matched_state_per_step_hazard"],
        "specdec_mtp_freerun_cascade_factor": mtp["freerun_cascade_factor"],
        "specdec_mtp_verdict": mtp["verdict"],
        "specdec_ngram_verdict": ngram["verdict"],
        "specdec_mtp_onset_median": mtp["onset_median"],
        "specdec_ngram_onset_median": ngram["onset_median"],
        "specdec_mtp_onset_min": mtp["onset_min"],
        "specdec_mtp_onset_max": mtp["onset_max"],
        # gate decision
        "specdec_passes_319_byconstruction": passes_319,
        "specdec_passes_319_relative_to_chaos_floor": no_div_above_floor,
        "specdec_319_failure_attribution": attribution,
        "specdec_miss_is_bf16_tie_reorder": miss_is_tie,
        "specdec_identity_safe": passes_319,
        "specdec_is_fire_eligible_on_identity": passes_319,
        "quality_gate_passes_by_construction": True,  # base_fullhead full bf16 head; quality 4/4 not in question
        # chaos floor (no-spec self-determinism) — the load-bearing context for the verdict
        "self_det": self_det["verdict"] == "GREEDY_IDENTICAL",
        "self_det_verdict": self_det["verdict"],
        "self_det_sequence_exact_rate": chaos_seq,
        "chaos_floor_sequence_exact_rate": chaos_seq,
        "chaos_floor_per_step_rate": chaos_step,
        "reference_itself_nondeterministic": ref_nondeterministic,
        "self_det_bc_verdict": (self_det_bc or {}).get("verdict"),
        "steady_state_nondeterministic": steady_nondeterministic,
        # cross-tab (spec divergence vs chaos floor, prompt level)
        "n_chaotic_prompts": (ctab or {}).get("n_chaotic"),
        "n_stable_prompts": (ctab or {}).get("n_stable"),
        "specdec_mtp_n_divergent_on_stable": mtp_n_div_stable,
        "specdec_mtp_divergence_subset_of_chaos": mtp_subset_chaos,
        "specdec_ngram_divergence_subset_of_chaos": ngram_subset_chaos,
        # tie probe
        "tie_median_miss_margin": (tie or {}).get("median_miss_margin"),
        "tie_median_control_margin": (tie or {}).get("median_control_margin"),
        "tie_separation_ratio": (tie or {}).get("separation_ratio"),
        "tie_n_exact_tie": (tie or {}).get("n_exact_tie"),
        "tie_n_miss_cand_in_topk": (tie or {}).get("n_miss_cand_in_topk"),
        "tie_n_miss_probed": (tie or {}).get("n_miss_probed"),
        "tie_mtp_median_margin": (bd.get("mtp") or {}).get("median_margin"),
        "tie_ngram_median_margin": (bd.get("ngram") or {}).get("median_margin"),
        "tie_selfdet_median_margin": (bd.get("selfdet") or {}).get("median_margin"),
        # anchors / context
        "anchor_nospec_tps": ANCHOR_NOSPEC_TPS,
        "ship_tps": SHIP_TPS,
        "free_floor_tps": FREE_FLOOR_TPS,
        "baseline_tps": BASELINE_TPS,
        "cv566_per_step_identity": CV566_PER_STEP,
        "cv566_sequence_exact": CV566_SEQUENCE_EXACT,
        "peak_vram_gb": peak_vram,
        "analysis_only": True,
        "official_tps": 0,
    }


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[census] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile", agent="denken",
        name=args.wandb_name or "denken/specdec-identity-census",
        group=args.wandb_group or "base-fullhead-specdec-identity-census",
        tags=["specdec", "identity-census", "319", "served", "base-fullhead",
              "local-a10g", "analysis-only", "pr576"],
        notes="PR #576: served spec-dec #319 byte-exact greedy-identity census (REF no-spec vs "
              "MTP K=7 + ngram), the last un-censused leg of the #319 contract.",
        config={
            "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
            "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": SEED,
            "concurrency": 1, "ngram_spec_config": NGRAM_SPEC_CONFIG,
            "gpu_mem_util": GPU_MEM_UTIL,
        },
    )
    if run is None:
        return None
    s = report["synthesis"]
    summary = {k: v for k, v in s.items() if _finite(v) or isinstance(v, bool) or isinstance(v, str)}
    summary["primary_metric"] = s["specdec_mtp_sequence_exact_rate"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="specdec-identity-census-report",
                      artifact_type="specdec-identity-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(s: dict[str, Any]) -> None:
    line = "=" * 10 + " PR #576 — SERVED SPEC-DEC #319 IDENTITY CENSUS " + "=" * 10
    print("\n" + line, flush=True)
    print(f"  CHAOS FLOOR (no-spec ref_a vs ref_b) seq={s['chaos_floor_sequence_exact_rate']} "
          f"step={s['chaos_floor_per_step_rate']}  [{s['self_det_verdict']}]", flush=True)
    print(f"    steady-state (ref_b vs ref_c)        nondeterministic={s['steady_state_nondeterministic']} "
          f"[{s['self_det_bc_verdict']}]", flush=True)
    print(f"    reference itself nondeterministic    = {s['reference_itself_nondeterministic']}", flush=True)
    print(f"  MTP   sequence byte-exact rate  = {s['specdec_mtp_sequence_exact_rate']}  "
          f"[{s['specdec_mtp_verdict']}]  per-step {s['specdec_mtp_per_step_identity_rate']}", flush=True)
    print(f"  ngram sequence byte-exact rate  = {s['specdec_ngram_sequence_exact_rate']}  "
          f"[{s['specdec_ngram_verdict']}]  per-step {s['specdec_ngram_per_step_identity_rate']}", flush=True)
    print(f"  cascade factor (seq/step miss)  = {s['per_step_to_sequence_cascade_factor']}", flush=True)
    print(f"  chaotic/stable prompts          = {s['n_chaotic_prompts']}/{s['n_stable_prompts']}", flush=True)
    print(f"  >>> MTP divergent on STABLE     = {s['specdec_mtp_n_divergent_on_stable']}  "
          f"(subset-of-chaos={s['specdec_mtp_divergence_subset_of_chaos']})", flush=True)
    print(f"  >>> passes #319 by-construction = {s['specdec_passes_319_byconstruction']}  "
          f"(rel-to-chaos-floor={s['specdec_passes_319_relative_to_chaos_floor']})", flush=True)
    print(f"  >>> #319 failure attribution    = {s['specdec_319_failure_attribution']}", flush=True)
    print(f"  >>> miss is bf16-tie reorder    = {s['specdec_miss_is_bf16_tie_reorder']}  "
          f"(miss {s['tie_median_miss_margin']} vs control {s['tie_median_control_margin']}; "
          f"mtp/ngram/selfdet med margins {s['tie_mtp_median_margin']}/{s['tie_ngram_median_margin']}/{s['tie_selfdet_median_margin']})", flush=True)
    print(f"  >>> identity_safe / fire-elig   = {s['specdec_identity_safe']} / {s['specdec_is_fire_eligible_on_identity']}", flush=True)
    print(f"  peak VRAM                       = {s['peak_vram_gb']:.2f} GB", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# main
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check (4x24, all 3 arms, no tie probe)")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--tie-probe-limit", type=int, default=48, help="max divergence positions to probe per drafter")
    ap.add_argument("--no-tie-probe", action="store_true")
    ap.add_argument("--ref-passes", type=int, default=3,
                    help="no-spec ref passes: 2=chaos floor (cold->warm); 3=also steady-state (warm->warm)")
    ap.add_argument("--reuse-complete", action="store_true",
                    help="reuse already-complete mtp/ngram decode captures (skip their boot+recapture); "
                         "the ref arm is always captured fresh")
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
        print(f"[census] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1) MTP arm (ship drafter) — capture once.
    mtp_arm = run_arm(harness, paths, server_python=server_python, arm="mtp",
                      num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                      passes=1, do_tie_probe=False, divergence_jobs=None, reuse=args.reuse_complete)
    print(f"[census] MTP captured peak={mtp_arm['peak_vram_gb']:.2f}GB "
          f"({time.time() - t_start:.0f}s)", flush=True)

    # 2) ngram arm — capture once.
    ngram_arm = run_arm(harness, paths, server_python=server_python, arm="ngram",
                        num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                        passes=1, do_tie_probe=False, divergence_jobs=None, reuse=args.reuse_complete)
    print(f"[census] ngram captured peak={ngram_arm['peak_vram_gb']:.2f}GB "
          f"({time.time() - t_start:.0f}s)", flush=True)

    # 3) ref arm (no-spec) — capture `ref_passes` times for self-determinism, then verify +
    #    cross-tab + tie-probe while the server is still live. ref_a is THE #319 reference (cold,
    #    spec-off = gen_greedy_reference contract); ref_b/ref_c characterize the chaos floor. All
    #    in ONE boot via a deferred-verify closure -> the whole census stays at 3 boots.
    ref_arm = capture_ref_then_probe(
        harness, paths, server_python=server_python, port=args.port,
        num_prompts=args.num_prompts, output_len=args.output_len,
        mtp_file=mtp_arm["capture_files"][0], ngram_file=ngram_arm["capture_files"][0],
        do_tie_probe=(not args.no_tie_probe), tie_limit=args.tie_probe_limit,
        ref_passes=(2 if args.smoke else args.ref_passes))
    print(f"[census] ref captured + verified peak={ref_arm['peak_vram_gb']:.2f}GB "
          f"({time.time() - t_start:.0f}s)", flush=True)

    self_det = ref_arm["self_det"]
    self_det_bc = ref_arm.get("self_det_bc")
    mtp_v = ref_arm["mtp_verdict"]
    ngram_v = ref_arm["ngram_verdict"]
    ctab = ref_arm.get("cross_tab")
    tie = ref_arm.get("tie_probe")
    peak_vram = max(mtp_arm["peak_vram_gb"], ngram_arm["peak_vram_gb"], ref_arm["peak_vram_gb"])

    synthesis = synthesize(self_det, self_det_bc, mtp_v, ngram_v, tie, ctab, peak_vram)
    report = {
        "pr": 576, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
        "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": SEED,
        "ngram_spec_config": NGRAM_SPEC_CONFIG,
        "arms": {
            "mtp": {k: v for k, v in mtp_arm.items() if k not in ("summaries",)},
            "ngram": {k: v for k, v in ngram_arm.items() if k not in ("summaries",)},
            "ref": {k: v for k, v in ref_arm.items()
                    if k not in ("mtp_verdict", "ngram_verdict", "self_det", "self_det_bc",
                                 "cross_tab", "tie_probe")},
        },
        "self_det": self_det,
        "self_det_bc": self_det_bc,
        "cross_tab": ctab,
        "mtp_verdict": {k: v for k, v in mtp_v.items() if k != "_per_prompt"},
        "ngram_verdict": {k: v for k, v in ngram_v.items() if k != "_per_prompt"},
        "tie_probe": tie,
        "synthesis": synthesis,
        "elapsed_s": time.time() - t_start,
    }
    out_file = OUT_ROOT / ("census_smoke.json" if args.smoke else "census_report.json")
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[census] report -> {out_file} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[census] wandb run id={rid}", flush=True)
    return 0


def capture_ref_then_probe(harness: Any, paths: Any, *, server_python: Path, port: int,
                           num_prompts: int, output_len: int, mtp_file: str, ngram_file: str,
                           do_tie_probe: bool, tie_limit: int, ref_passes: int) -> dict[str, Any]:
    """Boot the no-spec ref stack ONCE: capture `ref_passes` ref passes (self-determinism),
    verify vs the already-captured mtp/ngram files, build the chaos-floor cross-tab, then (server
    still live) run the flip-margin probe at the divergence positions. Keeps the whole census to
    3 server boots.

    ref_a is THE #319 reference (one cold-start spec-off pass = the gen_greedy_reference contract;
    same cold regime as the MTP/ngram first passes -> the comparison is apples-to-apples).
    self_det_ab (ref_a vs ref_b) is the chaos floor INCLUDING the cold->warm transient; with a 3rd
    pass, self_det_bc (ref_b vs ref_c, both warm) isolates GENUINE steady-state non-determinism
    from any cold-start-only effect."""
    log_path = OUT_ROOT / "server_ref.log"
    extra_env = arm_env("ref")
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    out: dict[str, Any] = {"arm": "ref", "capture_files": []}
    tags = [f"ref_{chr(ord('a') + i)}" for i in range(max(2, ref_passes))]
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            for tag in tags:
                out_file = OUT_ROOT / f"decode_{tag}.jsonl"
                summary_file = OUT_ROOT / f"decode_{tag}.summary.json"
                print(f"[census] [ref] capture {tag} {num_prompts}x{output_len} conc=1", flush=True)
                harness.capture_decode(
                    server_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_file, summary_file=summary_file,
                    num_prompts=num_prompts, output_len=output_len, seed=SEED,
                    tokenizer=paths.TOKENIZER, dataset=paths.EVAL_PROMPTS, timeout_s=5400)
                out["capture_files"].append(str(out_file))
            ref_a = out["capture_files"][0]
            ref_b = out["capture_files"][1]
            ref_c = out["capture_files"][2] if len(out["capture_files"]) >= 3 else None
            out["self_det"] = verify_pair(paths, ref_a, ref_b)          # chaos floor (cold->warm)
            if ref_c is not None:
                out["self_det_bc"] = verify_pair(paths, ref_b, ref_c)   # steady-state (warm->warm)
            out["mtp_verdict"] = verify_pair(paths, ref_a, mtp_file)
            out["ngram_verdict"] = verify_pair(paths, ref_a, ngram_file)
            out["cross_tab"] = cross_tab(out["self_det"]["_per_prompt"],
                                         out["mtp_verdict"]["_per_prompt"],
                                         out["ngram_verdict"]["_per_prompt"])

            if do_tie_probe:
                ref_recs = load_decode(ref_a)
                ref_b_recs = load_decode(ref_b)
                mtp_recs = load_decode(mtp_file)
                ngram_recs = load_decode(ngram_file)
                # spec-vs-ref divergences AND the chaos-floor (self-det) divergences -> show they
                # share the same bf16 near-tie margin signature.
                jobs = build_divergence_jobs(ref_recs, mtp_recs, out["mtp_verdict"]["_per_prompt"],
                                             "mtp", tie_limit)
                jobs += build_divergence_jobs(ref_recs, ngram_recs, out["ngram_verdict"]["_per_prompt"],
                                              "ngram", tie_limit)
                jobs += build_divergence_jobs(ref_recs, ref_b_recs, out["self_det"]["_per_prompt"],
                                              "selfdet", tie_limit)
                jobs += build_control_jobs(ref_recs, out["mtp_verdict"]["_per_prompt"],
                                           min(tie_limit, 40))
                if jobs:
                    print(f"[census] [ref] flip-margin probe: {len(jobs)} positions", flush=True)
                    out["tie_probe"] = flip_margin_probe(srv.base_url, srv.served_model_name, jobs)
    finally:
        stop.set()
        sampler.join(timeout=5)
    out["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    out["log_path"] = str(log_path)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
