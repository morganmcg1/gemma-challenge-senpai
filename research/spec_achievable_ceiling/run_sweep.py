#!/usr/bin/env python
"""PR #728 lawine — Spec-dec achievable ceiling: K-sweep for a comfortable-margin
self-consistent config.

DEPLOYED measurement on the faithful vLLM 0.22.0 engine (NOT dev307). For each
K in {5,6,7} we serve the ACTUAL int4 target + MTP-K drafter spec stack (drafter
ON, VLLM_BATCH_INVARIANT=1), free-run greedy-decode the official 128x512 eval set
through the official decode_outputs.py, and measure:

  * wall_tps         = num_completion_tokens / decode_duration_s  (the official
                       output-throughput metric definition; see project_local_tps_bench)
  * 128/128          = num_records==128 and num_completion_tokens==65536
  * strict self-consistency (UN-RESCUED): greedy_gate(config's OWN served-AR
    reference, config's served-spec) -> verdict / seq_exact / per-prompt onset.
    The reference is the config's OWN plain-AR greedy (drafter OFF,
    SENPAI_REFERENCE_MODE=1, BI=1) -> #319 ruling: own-AR, NOT the locked anchor.
  * tau-rescued self-consistency (the advisor #654/#655 gate): each free-run
    onset divergence is classified by the BASE model's logit gap at that SHARED-
    prefix decision point (gap = base_logprob(base_argmax) - base_logprob(spec_token),
    read by teacher-forcing the AR reference through the AR config's prompt_logprobs).
    confident_genuine_flips(tau) = #prompts whose onset gap > tau. A config is
    SELF-CONSISTENT at tau iff confident_genuine_flips(tau)==0 AND no onset token
    is emitted from outside the base top-k (per #654: 0 confident misses; the
    surviving flips are all <=1 int4-quantum grid ties). tau_gate = 0.3 nat.
  * official-equiv   = wall_tps * TAU_LO   (banked local->official scalar; the
                       int4-spec stack ratio may differ -> reported as a projection,
                       NOT a fired official score).

The onset is the only position where the free-run spec and free-run AR share an
identical prefix, so it is the only clean per-position "did the base confidently
disagree" test; positions after onset are on cascaded trajectories. This is the
DEPLOYED analog of the #616/#621 teacher-forced flip rate (which proxied the verify
forward via MAX_NUM_BATCHED_TOKENS=K+1 reference-mode and reached only the prefill
branch). Here the spec token under test is the real deployed (decode-branch) output.

LOCAL ONLY: analysis_only=1, official_tps=0, single A10G, NO HF Job / no --launch /
no submission change. Group: lawine-spec-achievable-ceiling.
"""
from __future__ import annotations

import argparse
import json
import math
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

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"

# The #607/#616/#319-census int4 build + MTP drafter (the PR #728 anchor checkpoint,
# int4_g128_lmhead @ 126.378 official; W&B 905tbujn). Local paths -> no download.
MODEL_DIR = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER = "/tmp/qat-assistant"

K_GRID = [5, 6, 7]
TAU_GATE = 0.3                       # nats; the self-consistency confident-miss threshold (#654)
TAU_GRID = [0.05, 0.1, 0.125, 0.2, 0.3, 0.5, 1.0]  # 0.125 ~= 1 int4 quantum
PROMPT_LOGPROBS = 20
TAU_LO = 1.0352                      # banked local->official scalar (project_local_official_tps_transfer)

ANCHOR_TPS = 126.378                 # locked int4_g128_lmhead official TPS (the bar to beat)
ANCHOR_PPL = 2.019
PPL_GATE = 2.42
COMFORT_BAR = 150.0                  # target official-equiv for comfortable private-haircut margin


# --------------------------------------------------------------------------- serve env
def base_env(model_id: str, drafter: str, batch_invariant: int) -> dict[str, str]:
    """Served recipe shared by the AR reference and every spec candidate (matches the
    #607/#616 census serve: BI=1, conc=1, FlashInfer sampler off, 4096 ctx, 512 chunk)."""
    return {
        "MODEL_ID": model_id,
        "DRAFTER_MODEL": drafter,
        "VLLM_BATCH_INVARIANT": str(batch_invariant),
        "MAX_NUM_SEQS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "512",
    }


def _gpu_used_mib() -> float:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else 0.0
    except Exception:
        return 0.0


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        peak["mib"] = max(peak["mib"], _gpu_used_mib())
        stop.wait(2.0)


# --------------------------------------------------------------------------- prompt_logprobs (base distribution read)
def request_prompt_logprobs(base_url: str, model: str, token_ids: list[int], timeout_s: int) -> dict[str, Any]:
    """Teacher-force ``token_ids`` and read the per-position base distribution (reuses the
    #616 ppl-endpoint request shape: token-id prompt, prompt_logprobs top-k, no specials)."""
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
        "prompt_logprobs": PROMPT_LOGPROBS,
        "add_special_tokens": False,
        "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc


def _normalize_entry(entry: Any) -> dict[int, tuple[float, int | None]]:
    """prompt_logprobs[i] -> {token_id: (logprob, rank)} (handles int/str/'token_id:' keys)."""
    out: dict[int, tuple[float, int | None]] = {}
    if not isinstance(entry, dict):
        return out
    for k, v in entry.items():
        try:
            tok = int(k)
        except (ValueError, TypeError):
            ks = str(k)
            if ks.startswith("token_id:"):
                try:
                    tok = int(ks.split(":", 1)[1])
                except ValueError:
                    continue
            else:
                continue
        if isinstance(v, dict):
            lp = v.get("logprob")
            rank = v.get("rank")
        else:
            lp, rank = (float(v) if isinstance(v, (int, float)) else None), None
        if lp is None:
            continue
        out[tok] = (float(lp), rank)
    return out


def _argmax_token(norm: dict[int, tuple[float, int | None]]) -> tuple[int, float]:
    """argmax = rank-1 token (vLLM rank is 1-indexed); fall back to max logprob, lowest-id tie."""
    rank1 = [(tok, lp) for tok, (lp, rank) in norm.items() if rank == 1]
    if len(rank1) == 1:
        return rank1[0]
    best_tok, best_lp = None, -math.inf
    for tok, (lp, _rank) in norm.items():
        if lp > best_lp or (lp == best_lp and (best_tok is None or tok < best_tok)):
            best_tok, best_lp = tok, lp
    return best_tok, best_lp  # type: ignore[return-value]


def teacher_force_base_dist(base_url: str, model: str, recs: list[dict[str, Any]],
                            timeout_s: int) -> dict[int, list[dict[str, Any]]]:
    """For each ref rec, teacher-force (prompt+completion) and read the base distribution at
    every GENERATED position: {index -> [ {amax, amax_lp, topk:{tok:lp}} for each gen pos ]}.
    This is the base preference at each greedy decision point given the shared (ref) prefix."""
    out: dict[int, list[dict[str, Any]]] = {}
    for n, rec in enumerate(recs):
        full = list(rec["prompt_token_ids"]) + list(rec["completion_token_ids"])
        s, e = len(rec["prompt_token_ids"]), len(full)
        resp = request_prompt_logprobs(base_url, model, full, timeout_s)
        choices = resp.get("choices") or []
        logprobs = (choices[0].get("prompt_logprobs") if choices else None) or resp.get("prompt_logprobs")
        if not isinstance(logprobs, list) or len(logprobs) < e:
            raise RuntimeError(f"index {rec['index']}: got {len(logprobs) if isinstance(logprobs, list) else 'none'} "
                               f"prompt_logprobs for {e} positions")
        per_pos: list[dict[str, Any]] = []
        for idx in range(s, e):
            norm = _normalize_entry(logprobs[idx])
            if not norm:
                per_pos.append({"amax": -1, "amax_lp": 0.0, "topk": {}})
                continue
            amax_tok, amax_lp = _argmax_token(norm)
            per_pos.append({
                "amax": amax_tok, "amax_lp": amax_lp,
                "topk": {int(t): float(lp) for t, (lp, _r) in norm.items()},
            })
        out[rec["index"]] = per_pos
        if (n + 1) % 16 == 0 or n + 1 == len(recs):
            print(f"[ceil]   teacher-forced base dist {n + 1}/{len(recs)}", flush=True)
    return out


# --------------------------------------------------------------------------- decode jsonl io
def load_decode_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    """decode_outputs.jsonl -> {index -> {prompt_token_ids, completion_token_ids, id}}."""
    rows: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows[int(d["index"])] = {
                "index": int(d["index"]),
                "id": d.get("id"),
                "prompt_token_ids": d["prompt_token_ids"],
                "completion_token_ids": d["completion_token_ids"],
            }
    return rows


def wall_tps(summary: dict[str, Any]) -> float | None:
    dur = summary.get("duration_s")
    ntok = summary.get("num_completion_tokens")
    if isinstance(dur, (int, float)) and dur > 0 and isinstance(ntok, (int, float)):
        return ntok / dur
    return None


# --------------------------------------------------------------------------- serve
def serve_capture(submission: Path, server_python: Path, *, label: str, run_dir: Path,
                  extra_env: dict[str, str], port: int, num_prompts: int, output_len: int,
                  do_ppl: bool, do_logprobs: bool, ref_recs: list[dict[str, Any]] | None,
                  startup_timeout_s: int) -> dict[str, Any]:
    """Serve one config, capture the official decode, optionally PPL + teacher-forced base dist."""
    out_file = run_dir / f"{label}.jsonl"
    summary_file = run_dir / f"{label}.summary.json"
    log_path = run_dir / f"{label}.server.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    info: dict[str, Any] = {"label": label, "extra_env": extra_env}
    t0 = time.time()
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=port,
            startup_timeout_s=startup_timeout_s, log_path=log_path, extra_env=extra_env,
        ) as srv:
            info["serve_ready_s"] = time.time() - t0
            print(f"[ceil] [{label}] ready in {info['serve_ready_s']:.0f}s; decoding "
                  f"{num_prompts}x{output_len}", flush=True)
            summary = harness.capture_decode(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=out_file, summary_file=summary_file,
                num_prompts=num_prompts, output_len=output_len, seed=paths.SEED,
            )
            info["decode_summary"] = summary
            info["wall_tps"] = wall_tps(summary)
            info["num_records"] = summary.get("num_records")
            info["num_completion_tokens"] = summary.get("num_completion_tokens")
            if do_ppl:
                ppl_out = run_dir / f"{label}.ppl.jsonl"
                ppl_sum = run_dir / f"{label}.ppl.summary.json"
                print(f"[ceil] [{label}] PPL ...", flush=True)
                ppl = harness.run_ppl(server_python, base_url=srv.base_url,
                                      model=srv.served_model_name, out_file=ppl_out,
                                      summary_file=ppl_sum)
                info["ppl"] = ppl.get("ppl") or ppl.get("perplexity") or ppl
            if do_logprobs:
                recs = ref_recs
                if recs is None:  # teacher-force the decode we just captured (same live server)
                    rows = load_decode_jsonl(out_file)
                    recs = [rows[i] for i in sorted(rows)]
                print(f"[ceil] [{label}] teacher-forcing base distribution over {len(recs)} ref seqs ...",
                      flush=True)
                bd = teacher_force_base_dist(srv.base_url, srv.served_model_name, recs, timeout_s=900)
                (run_dir / f"{label}.base_dist.json").write_text(json.dumps(bd, default=str))
                info["base_dist_positions"] = sum(len(v) for v in bd.values())
                info["_base_dist"] = bd  # in-memory handoff (not serialized into info.json)
    finally:
        stop.set()
        sampler.join(timeout=5)
    info["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    info["server_log"] = str(log_path)
    info["capture_wall_s"] = time.time() - t0
    persist = {k: v for k, v in info.items() if k != "_base_dist"}
    (run_dir / f"{label}.info.json").write_text(json.dumps(persist, indent=2, default=str))
    return info


# --------------------------------------------------------------------------- rescue classification
def classify_onsets(ref_rows: dict[int, dict[str, Any]], cand_rows: dict[int, dict[str, Any]],
                    base_dist: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """For every prompt whose free-run spec output diverges from the free-run AR reference,
    classify the FIRST (onset) divergence by the base model's logit gap at that shared-prefix
    decision point. Returns per-prompt records + tau-swept confident_genuine_flips counts."""
    per_prompt: list[dict[str, Any]] = []
    for index in sorted(ref_rows):
        ref_c = ref_rows[index]["completion_token_ids"]
        cand = cand_rows.get(index)
        if cand is None:
            continue
        cand_c = cand["completion_token_ids"]
        n = min(len(ref_c), len(cand_c))
        onset = next((i for i in range(n) if ref_c[i] != cand_c[i]), None)
        if onset is None:
            per_prompt.append({"index": index, "divergent": False, "onset": None, "gap": 0.0,
                               "cand_tok": None, "base_amax": None, "outside_topk": False})
            continue
        cand_tok = cand_c[onset]
        bd = base_dist.get(index)
        if bd is None or onset >= len(bd):
            per_prompt.append({"index": index, "divergent": True, "onset": onset, "gap": None,
                               "cand_tok": cand_tok, "base_amax": None, "outside_topk": None,
                               "note": "no base_dist"})
            continue
        cell = bd[onset]
        amax_lp = cell.get("amax_lp", 0.0)
        topk = {int(t): float(lp) for t, lp in cell.get("topk", {}).items()}
        cand_lp = topk.get(cand_tok)
        if cand_tok == cell.get("amax"):
            gap, outside = 0.0, False
        elif cand_lp is not None:
            gap, outside = max(0.0, amax_lp - cand_lp), False
        else:
            # spec token outside base top-k -> gap lower-bounded by amax - (k-th logprob).
            floor = min(topk.values()) if topk else amax_lp
            gap, outside = max(0.0, amax_lp - floor), True
        per_prompt.append({"index": index, "divergent": True, "onset": onset, "gap": gap,
                           "cand_tok": cand_tok, "base_amax": cell.get("amax"),
                           "outside_topk": outside})

    divergent = [p for p in per_prompt if p["divergent"]]
    onset_gaps = [p["gap"] for p in divergent if isinstance(p.get("gap"), (int, float))]
    tau_sweep: dict[str, Any] = {}
    for tau in TAU_GRID:
        # a confident genuine flip = gap>tau with a KNOWN gap, OR emitted from outside the base
        # top-k (rank>k => the base disfavors it; conservatively a confident miss).
        confident = sum(1 for p in divergent
                        if (isinstance(p.get("gap"), (int, float)) and p["gap"] > tau and not p["outside_topk"])
                        or p.get("outside_topk") is True)
        tau_sweep[f"tau_{tau}"] = {
            "confident_genuine_flips": confident,
            "rescued_divergent": len(divergent) - confident,
        }
    outside = sum(1 for p in divergent if p.get("outside_topk") is True)
    return {
        "num_divergent_prompts": len(divergent),
        "num_identical_prompts": len(per_prompt) - len(divergent),
        "onset_gap_min": min(onset_gaps) if onset_gaps else None,
        "onset_gap_median": (sorted(onset_gaps)[len(onset_gaps) // 2] if onset_gaps else None),
        "onset_gap_max": max(onset_gaps) if onset_gaps else None,
        "onset_gap_frac_le_0.125": (sum(g <= 0.125 for g in onset_gaps) / len(onset_gaps)) if onset_gaps else None,
        "onset_gap_frac_le_0.3": (sum(g <= 0.3 for g in onset_gaps) / len(onset_gaps)) if onset_gaps else None,
        "num_onset_outside_topk": outside,
        "tau_sweep": tau_sweep,
        "confident_genuine_flips_at_gate": tau_sweep[f"tau_{TAU_GATE}"]["confident_genuine_flips"],
        "per_prompt": per_prompt,
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default="5,6,7", help="comma list of K (num_speculative_tokens)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--model-id", default=MODEL_DIR)
    ap.add_argument("--drafter", default=DRAFTER)
    ap.add_argument("--ppl-k", type=int, default=6, help="also measure PPL on this spec K (drafter-indep sanity)")
    ap.add_argument("--ref-port", type=int, default=8021)
    ap.add_argument("--cand-port", type=int, default=8022)
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--reuse-ref", action="store_true", help="reuse ref.jsonl + base_dist if present")
    ap.add_argument("--smoke", action="store_true", help="4 prompts, K=6 only, no PPL — wiring check")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="lawine/spec-achievable-ceiling")
    ap.add_argument("--wandb-group", default="lawine-spec-achievable-ceiling")
    args = ap.parse_args()

    ks = [6] if args.smoke else [int(x) for x in args.ks.split(",") if x.strip()]
    num_prompts = 4 if args.smoke else args.num_prompts
    do_ppl = not args.smoke
    run_dir = args.out_dir / (f"{args.label}_smoke" if args.smoke else args.label)
    run_dir.mkdir(parents=True, exist_ok=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[ceil] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    tf_ver = harness._dist_version(server_python, "transformers")
    print(f"[ceil] server_python={server_python} vllm={vllm_ver} transformers={tf_ver}", flush=True)
    if vllm_ver != "0.22.0":
        print(f"[ceil] WARNING: expected faithful vllm 0.22.0, got {vllm_ver}", flush=True)

    base = base_env(args.model_id, args.drafter, batch_invariant=1)
    ref_file = run_dir / "ref.jsonl"
    bd_file = run_dir / "ref.base_dist.json"

    # ---- AR M=1 reference (drafter OFF) : the config's OWN plain-AR greedy + base distribution ----
    if args.reuse_ref and ref_file.exists() and bd_file.exists():
        print(f"[ceil] reusing reference {ref_file} + base_dist", flush=True)
        ref_info = json.loads((run_dir / "ref.info.json").read_text()) if (run_dir / "ref.info.json").exists() else {}
        base_dist = {int(k): v for k, v in json.loads(bd_file.read_text()).items()}
    else:
        ref_env = {**base, "SENPAI_REFERENCE_MODE": "1", "NUM_SPECULATIVE_TOKENS": "0"}
        print("[ceil] === AR M=1 REFERENCE (drafter OFF, BI=1): decode + PPL + base dist (one boot) ===",
              flush=True)
        # one boot: capture the AR decode, then teacher-force it for the base distribution.
        ref_info = serve_capture(
            SUBMISSION, server_python, label="ref", run_dir=run_dir, extra_env=ref_env,
            port=args.ref_port, num_prompts=num_prompts, output_len=args.output_len,
            do_ppl=do_ppl, do_logprobs=True, ref_recs=None, startup_timeout_s=1800,
        )
        base_dist = ref_info.get("_base_dist") or {int(k): v for k, v in json.loads(
            (run_dir / "ref.base_dist.json").read_text()).items()}
        # canonical copy of the base dist next to the ref (str keys for json round-trip safety)
        (bd_file).write_text(json.dumps({str(k): v for k, v in base_dist.items()}, default=str))

    ref_rows = load_decode_jsonl(ref_file)
    # engine-coherence sanity: teacher-forced base argmax should reproduce the free-run AR token
    # at the vast majority of positions (prefill-vs-decode branch agreement ~99%); a low number
    # would flag the cycle-58DE engine-corruption concern.
    coh_match = coh_total = 0
    for index, row in ref_rows.items():
        bd = base_dist.get(index) or base_dist.get(str(index))  # tolerate str keys from disk
        if not bd:
            continue
        comp = row["completion_token_ids"]
        for i in range(min(len(comp), len(bd))):
            coh_total += 1
            if bd[i].get("amax") == comp[i]:
                coh_match += 1
    coherence = (coh_match / coh_total) if coh_total else None
    print(f"[ceil] engine coherence (teacher-forced argmax == free-run AR token): "
          f"{coh_match}/{coh_total} = {coherence:.4f}" if coherence is not None else
          "[ceil] engine coherence: n/a", flush=True)

    ar_tps = ref_info.get("wall_tps")
    ar_ppl = ref_info.get("ppl")
    print(f"[ceil] AR reference: wall_tps={ar_tps} ppl={ar_ppl} "
          f"records={ref_info.get('num_records')}/{num_prompts}", flush=True)

    # ---- spec candidates ----
    results: list[dict[str, Any]] = []
    for K in ks:
        cand_env = {**base, "NUM_SPECULATIVE_TOKENS": str(K)}
        label = f"spec_k{K}"
        print(f"[ceil] === SPEC CANDIDATE K={K} (drafter ON, BI=1) ===", flush=True)
        cand_info = serve_capture(
            SUBMISSION, server_python, label=label, run_dir=run_dir, extra_env=cand_env,
            port=args.cand_port, num_prompts=num_prompts, output_len=args.output_len,
            do_ppl=(do_ppl and K == args.ppl_k), do_logprobs=False, ref_recs=None,
            startup_timeout_s=1800,
        )
        cand_file = run_dir / f"{label}.jsonl"
        cand_rows = load_decode_jsonl(cand_file)

        # strict (un-rescued) official gate: config's served-spec vs its OWN served-AR
        report = greedy_gate.compare(ref_file, cand_file)
        onset = greedy_gate.onset_summary(report)
        n_cmp = report.num_prompts_compared or 1
        seq_exact = report.num_identical / n_cmp
        tok_total = report.total_tokens_compared or 1
        tok_id = 1.0 - report.total_divergent_tokens / tok_total

        # tau-rescued self-consistency: classify the deployed onset divergences by base logit gap
        rescue = classify_onsets(ref_rows, cand_rows, base_dist)

        wt = cand_info.get("wall_tps")
        official_equiv = wt * TAU_LO if isinstance(wt, (int, float)) else None
        records = cand_info.get("num_records")
        comp_tokens = cand_info.get("num_completion_tokens")
        complete = (records == num_prompts) and (comp_tokens == num_prompts * args.output_len)
        cgf = rescue["confident_genuine_flips_at_gate"]
        self_consistent = (cgf == 0)

        res = {
            "k": K,
            "wall_tps_local": wt,
            "official_equiv_tps": official_equiv,
            "beats_anchor_126": (official_equiv > ANCHOR_TPS) if official_equiv else None,
            "clears_comfort_150": (official_equiv > COMFORT_BAR) if official_equiv else None,
            "records": records, "completion_tokens": comp_tokens, "complete_128_128": complete,
            "ppl": cand_info.get("ppl"),
            "strict_verdict": report.verdict,
            "strict_seq_exact": seq_exact,
            "strict_token_identity": tok_id,
            "strict_num_divergent": report.num_divergent,
            "onset": {k: onset.get(k) for k in ("onset_min", "onset_median", "onset_max", "num_divergent")},
            "confident_genuine_flips_at_0.3": cgf,
            "self_consistent_tau03": self_consistent,
            "rescue": {k: v for k, v in rescue.items() if k != "per_prompt"},
            "peak_vram_gb": cand_info.get("peak_vram_gb"),
            "serve_ready_s": cand_info.get("serve_ready_s"),
            "server_log": cand_info.get("server_log"),
        }
        (run_dir / f"{label}.rescue.json").write_text(json.dumps(rescue, indent=2, default=str))
        results.append(res)

        print(f"[ceil] K={K}: wall_tps={wt:.2f} official_equiv={official_equiv:.2f} "
              f"(>126:{res['beats_anchor_126']} >150:{res['clears_comfort_150']}) | "
              f"strict={report.verdict} seq_exact={seq_exact:.4f} | "
              f"confident_genuine_flips@0.3={cgf} self_consistent={self_consistent} | "
              f"ppl={res['ppl']} 128/128={complete}"
              if isinstance(wt, (int, float)) and official_equiv else f"[ceil] K={K}: incomplete", flush=True)

    # ---- pick the fastest SELF-CONSISTENT config ----
    self_consistent = [r for r in results if r.get("self_consistent_tau03") and r.get("complete_128_128")]
    fastest_sc = max(self_consistent, key=lambda r: r["wall_tps_local"]) if self_consistent else None
    fastest_any = max((r for r in results if isinstance(r.get("wall_tps_local"), (int, float))),
                      key=lambda r: r["wall_tps_local"], default=None)

    report_obj = {
        "pr": 728,
        "analysis_only": True,
        "official_tps": 0,
        "smoke": args.smoke,
        "config": {
            "model_dir": args.model_id, "drafter": args.drafter, "ks": ks,
            "vllm_version": vllm_ver, "transformers_version": tf_ver,
            "batch_invariant": 1, "max_num_seqs": 1, "max_num_batched_tokens": 512,
            "num_prompts": num_prompts, "output_len": args.output_len, "seed": paths.SEED,
            "tau_gate_nats": TAU_GATE, "tau_lo": TAU_LO, "prompt_logprobs": PROMPT_LOGPROBS,
            "anchor_tps": ANCHOR_TPS, "anchor_ppl": ANCHOR_PPL, "comfort_bar": COMFORT_BAR,
            "method": "deployed serve (drafter ON, K) free-run greedy vs config's OWN served-AR "
                      "(drafter OFF, SENPAI_REFERENCE_MODE=1, BI=1); wall_tps=tokens/decode_s; "
                      "strict greedy_gate + tau-rescued onset-gap (confident_genuine_flips).",
        },
        "engine_coherence_tf_argmax_vs_freerun": coherence,
        "ar_reference": {"wall_tps_local": ar_tps, "ppl": ar_ppl,
                         "records": ref_info.get("num_records")},
        "results": results,
        "fastest_self_consistent": fastest_sc,
        "fastest_any": fastest_any,
    }
    out = run_dir / ("report.smoke.json" if args.smoke else "report.json")
    out.write_text(json.dumps(report_obj, indent=2, default=str))

    print("\n" + "=" * 78, flush=True)
    print(f"[PR728] spec achievable ceiling ({'SMOKE' if args.smoke else 'FULL'}) "
          f"vllm={vllm_ver} BI=1 conc=1 {num_prompts}x{args.output_len}", flush=True)
    print(f"  AR ref wall_tps={ar_tps} ppl={ar_ppl} engine_coherence={coherence}", flush=True)
    for r in results:
        print(f"  K={r['k']:>1} | wall_tps={r['wall_tps_local']} -> official_equiv={r['official_equiv_tps']} "
              f"| strict={r['strict_verdict']} seq_exact={r['strict_seq_exact']:.3f} "
              f"| cgf@0.3={r['confident_genuine_flips_at_0.3']} self_consistent={r['self_consistent_tau03']} "
              f"| ppl={r['ppl']} 128/128={r['complete_128_128']}", flush=True)
    if fastest_sc:
        print(f"  >>> FASTEST SELF-CONSISTENT: K={fastest_sc['k']} "
              f"official_equiv={fastest_sc['official_equiv_tps']:.2f} "
              f"(>126:{fastest_sc['beats_anchor_126']} >150:{fastest_sc['clears_comfort_150']})", flush=True)
    else:
        print("  >>> NO self-consistent config at tau=0.3 (all have confident genuine flips)", flush=True)
    print(f"  report -> {out}", flush=True)
    print("=" * 78, flush=True)

    if not args.no_wandb and not args.smoke:
        try:
            _log_wandb(report_obj, name=args.wandb_name, group=args.wandb_group)
        except Exception as exc:  # noqa: BLE001
            print(f"[ceil] WARNING: wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"report preserved at {out}.", flush=True)
    return 0


def _log_wandb(report: dict[str, Any], *, name: str, group: str) -> None:
    try:
        from scripts import wandb_logging as wl
    except ImportError:
        print("[ceil] wandb_logging unavailable — skipping", flush=True)
        return
    run = wl.init_wandb_run(
        job_type="spec-achievable-ceiling", agent="lawine", name=name, group=group,
        notes="PR728 deployed spec K-sweep: served wall_tps + tau=0.3-rescued self-consistency",
        tags=["pr728", "specdec", "achievable-ceiling", "self-consistency", "int4-mtp", "k-sweep"],
        config=report["config"],
    )
    if run is None:
        print("[ceil] wandb not configured — skipping", flush=True)
        return
    for r in report["results"]:
        m = {
            f"k{r['k']}/wall_tps_local": r.get("wall_tps_local"),
            f"k{r['k']}/official_equiv_tps": r.get("official_equiv_tps"),
            f"k{r['k']}/strict_seq_exact": r.get("strict_seq_exact"),
            f"k{r['k']}/confident_genuine_flips_0.3": r.get("confident_genuine_flips_at_0.3"),
            f"k{r['k']}/self_consistent": 1 if r.get("self_consistent_tau03") else 0,
            f"k{r['k']}/ppl": r.get("ppl") if isinstance(r.get("ppl"), (int, float)) else None,
            f"k{r['k']}/complete_128": 1 if r.get("complete_128_128") else 0,
        }
        run.summary.update({k: v for k, v in m.items() if v is not None})
    fsc = report.get("fastest_self_consistent")
    run.summary["fastest_self_consistent_k"] = fsc["k"] if fsc else None
    run.summary["fastest_self_consistent_official_equiv"] = fsc["official_equiv_tps"] if fsc else None
    run.summary["engine_coherence"] = report.get("engine_coherence_tf_argmax_vs_freerun")
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr728_ceiling_report", artifact_type="achievable-ceiling", data=report)
    wl.finish_wandb(run)
    print("[ceil] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
