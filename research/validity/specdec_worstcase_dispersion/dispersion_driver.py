#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #577 — spec-dec worst-case net-TPS DISPERSION across workload strata.

Serves the quality-safe ``base_fullhead`` substrate (stock base-int4 + full
native 262k head) for ONE drafter and measures, per workload stratum, the net
single-stream TPS and the realized draft acceptance. The decision-relevant
statistic for a quality-safe ship is NOT the mean speedup but the WORST-CASE
per-stratum net-TPS — acceptance is high on predictable text (EASY) and
collapses on hard step-by-step reasoning (HARD), the SAME workload family the
quality gate is certified on.

This script runs ONE drafter over all strata on a single served process (so the
model loads once); per-stratum acceptance comes from the Prometheus
``/metrics`` spec counters snapshotted before/after each stratum's decode, with
the vLLM server-log SpecDecoding lines as a cross-check fallback (serve_profile
notes the Prometheus counters can come back empty). Net TPS comes from each
stratum's own decode summary (num_completion_tokens / duration_s), the official
single-stream conc=1 protocol. Run once per drafter:

    python dispersion_driver.py --drafter nospec   # SENPAI_REFERENCE_MODE, anchor
    python dispersion_driver.py --drafter mtp       # MTP K=7 (the ship's drafter)
    python dispersion_driver.py --drafter ngram     # prompt-lookup (ubel #503 path)

LOCAL A10G analysis only. analysis_only=true, official_tps=0. NO HF Job / launch.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

OUT = ROOT / "research" / "validity" / "specdec_worstcase_dispersion"
SUB = ROOT / "submissions" / "fa2sw_strict_surgical357"

# Own stock base-int4 snapshot (native 262k head, NO baked bucket) — the
# quality-safe substrate shared by the spec-dec ceiling/identity legs.
STARK_BASE_INT4 = (
    "/senpai-run/home/student-stark/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)

MTP_K = 7
NGRAM_K = 7
DRAFTER_ENV = {
    # nospec anchor: serve.py's canonical M=1 AR greedy reference (drafter OFF),
    # everything else (surgical 2D attn, split-KV, PLE fold, native head) stays on.
    "nospec": {"SENPAI_REFERENCE_MODE": "1"},
    # MTP K=7 — the ship's drafter (surgical-357 SPECULATIVE_CONFIG default).
    "mtp": {
        "SPECULATIVE_CONFIG": json.dumps(
            {"method": "mtp", "model": "/tmp/qat-assistant",
             "num_speculative_tokens": MTP_K}
        )
    },
    # ngram / prompt-lookup (ubel #503 always-loads path), depth matched to K=7.
    # The MTP-specific fused kernels (onegraph loopgraph, fused-sparse-argmax on
    # the Gemma4MTP embedder, dixie fused-accept-prep) are drafter-side and keyed
    # to the MTP Gemma4Proposer; the native NgramProposer path does not use them.
    # Disabling their REQUIRE flags (and ONEGRAPH) yields the clean vLLM
    # prompt-lookup path and avoids a spurious hard-fail in the verify/accept step.
    # The verifier (eager target forward) is identical to the MTP arm, so the
    # MTP-vs-ngram throughput comparison stays apples-to-apples on the target side.
    "ngram": {
        "SPECULATIVE_CONFIG": json.dumps(
            {"method": "ngram", "num_speculative_tokens": NGRAM_K,
             "prompt_lookup_max": 3, "prompt_lookup_min": 2}
        ),
        "ONEGRAPH": "0",
        "LOOPGRAPH_REQUIRE_CAPTURE": "0",
        "DIXIE_FUSED_ACCEPT_PREP_REQUIRE": "0",
        "FUSED_SPARSE_ARGMAX_REQUIRE": "0",
    },
}

STRATA = {
    "easy": OUT / "stratum_easy.json",   # boilerplate code (most drafter-predictable)
    "mix": OUT / "stratum_mix.json",     # official-128 (mmlu_pro+gpqa+aime)
    "hard": OUT / "stratum_hard.json",   # step-by-step math CoT (decision-relevant)
}


def base_fullhead_env() -> dict[str, str]:
    return {
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_aime_no_precache.json",
        "MAX_NUM_SEQS": "1",
        "MAX_NUM_BATCHED_TOKENS": "512",
        "LOCAL_MODEL_DIR": STARK_BASE_INT4,
        "PLE_FOLD_TARGET_MODEL": STARK_BASE_INT4,
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",  # native 262k head -> no scatter-back
        "PLE_FOLD_EMBED_SCALE": "1",
        # PyTorch-native lowest-index argmax (#560/#566 tie-break); spec metrics on.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",  # expose spec_decode counters + SpecDecoding log
    }


_POS_RE = re.compile(
    r'^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{[^}]*\bposition="(\d+)"[^}]*\}\s+([\d.eE+-]+)$',
    re.M,
)


def fetch_metrics(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def snapshot_spec(metrics_text: str) -> dict:
    """Raw cumulative spec counters at one instant (deltas give per-stratum)."""
    base = serve_profile.parse_spec_metrics(metrics_text)
    per_pos: dict[int, float] = {}
    for m in _POS_RE.finditer(metrics_text):
        per_pos[int(m.group(1))] = per_pos.get(int(m.group(1)), 0.0) + float(m.group(2))
    return {
        "num_drafts": base.get("num_drafts"),
        "num_accepted_tokens": base.get("num_accepted_tokens"),
        "num_draft_tokens": base.get("num_draft_tokens"),
        "per_pos": per_pos,
    }


def _sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def stratum_accept(before: dict, after: dict, num_spec: int) -> dict:
    """Per-stratum acceptance from cumulative-counter deltas."""
    d_drafts = _sub(after["num_drafts"], before["num_drafts"])
    d_acc = _sub(after["num_accepted_tokens"], before["num_accepted_tokens"])
    d_draft_tok = _sub(after["num_draft_tokens"], before["num_draft_tokens"])
    out: dict = {
        "delta_num_drafts": d_drafts,
        "delta_num_accepted_tokens": d_acc,
        "delta_num_draft_tokens": d_draft_tok,
        "accept_rate": None,            # accepted / proposed(draft_tokens)
        "mean_accept_len": None,        # E[accept] = 1 + accepted/drafts
        "per_pos_accept": None,         # P(position i accepted) = per_pos[i]/drafts
        "source": "prometheus_delta",
    }
    if d_draft_tok:
        out["accept_rate"] = d_acc / d_draft_tok
    if d_drafts:
        out["mean_accept_len"] = 1.0 + d_acc / d_drafts
        pp = {}
        for i in sorted(after["per_pos"]):
            dv = (after["per_pos"].get(i, 0.0) - before["per_pos"].get(i, 0.0))
            pp[str(i)] = dv / d_drafts if d_drafts else None
        if pp:
            out["per_pos_accept"] = pp
    return out


def log_slice_accept(log_path: Path, off0: int, off1: int) -> dict:
    """Fallback: parse the SpecDecoding lines emitted during this stratum."""
    try:
        with open(log_path, "rb") as f:
            f.seek(off0)
            chunk = f.read(max(0, off1 - off0)).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    parsed = serve_profile.parse_spec_log(chunk)
    return {
        "accept_rate": parsed.get("draft_acceptance_rate"),
        "mean_accept_len": parsed.get("e_accept_exact"),
        "interval_mean_accept_len": parsed.get("e_accept_interval_mean"),
        "total_accepted_tokens": parsed.get("total_accepted_tokens"),
        "total_drafted_tokens": parsed.get("total_drafted_tokens"),
        "intervals": parsed.get("intervals"),
        "source": "server_log_slice",
    }


def decode_tps(summary: dict) -> float:
    dur = summary.get("duration_s") or 0.0
    toks = summary.get("num_completion_tokens") or 0
    return toks / dur if dur > 0 else float("nan")


def sha_map(jsonl: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["index"]] = row["completion_token_sha256"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", choices=list(DRAFTER_ENV), required=True)
    ap.add_argument("--strata", default="easy,mix,hard")
    ap.add_argument("--num-prompts", type=int, default=48)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--selfdet", action="store_true",
                    help="rerun the first stratum once and report byte self-det")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    strata = [s for s in args.strata.split(",") if s]
    num_spec = {"nospec": 0, "mtp": MTP_K, "ngram": NGRAM_K}[args.drafter]
    tag = args.tag or f"{args.drafter}_n{args.num_prompts}_l{args.output_len}"
    OUT.mkdir(parents=True, exist_ok=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[disp] {note}", flush=True)
    # prepare_local_gpu_env setdefaults the native sampler; we force it regardless.
    import os
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    extra_env = dict(base_fullhead_env())
    extra_env.update(DRAFTER_ENV[args.drafter])

    manifest = harness.load_manifest(SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    result: dict = {
        "drafter": args.drafter,
        "num_speculative_tokens": num_spec,
        "substrate": "base_fullhead",
        "local_model_dir": STARK_BASE_INT4,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "strata": strata,
        "serve_overrides": extra_env,
        "serve_ok": False,
        "per_stratum": {},
        "analysis_only": True,
        "official_tps": 0,
    }

    peak = {"mib": 0}
    stop = threading.Event()

    def sample_gpu() -> None:
        while not stop.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                peak["mib"] = max(peak["mib"], int(o.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(2)

    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    gpu_thread.start()

    log = OUT / f"server_{tag}.log"
    try:
        with harness.LocalServer(
            SUB, server_python=server_python, port=args.port,
            startup_timeout_s=900, log_path=log, extra_env=extra_env,
        ) as srv:
            result["serve_ok"] = True
            model = srv.served_model_name
            base_url = srv.base_url
            result["served_model_name"] = model
            print(f"[disp] serve_ok drafter={args.drafter} model={model}", flush=True)

            # Sanity self-test via chat-completions (applies the IT chat template
            # server-side, the way the official decode path does) — base_fullhead
            # should answer 'Paris'. A raw /v1/completions prompt would degenerate
            # because the model is instruction-tuned, so it is NOT a health signal.
            st = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps({"model": model, "max_tokens": 8, "temperature": 0.0,
                                 "messages": [{"role": "user",
                                               "content": "What is the capital of France? Answer in one word."}]}).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            ), timeout=120).read().decode())
            msg = (st.get("choices") or [{}])[0].get("message") or {}
            st_text = msg.get("content", "")
            result["self_test_text"] = st_text
            result["self_test_ok"] = "Paris" in st_text
            print(f"[disp] self_test={st_text!r} ok={result['self_test_ok']}", flush=True)

            # Warmup (onegraph capture / JIT) before timed strata.
            harness.capture_decode(
                server_python, base_url=base_url, model=model,
                out_file=OUT / f"_warmup_{tag}.jsonl",
                summary_file=OUT / f"_warmup_{tag}.summary.json",
                dataset=STRATA[strata[0]], num_prompts=2, output_len=16, timeout_s=600,
            )

            for stratum in strata:
                ds = STRATA[stratum]
                m_before = snapshot_spec(fetch_metrics(base_url)) if num_spec else None
                off0 = log.stat().st_size if log.exists() else 0
                df = OUT / f"decode_{tag}_{stratum}.jsonl"
                sf = OUT / f"decode_{tag}_{stratum}.summary.json"
                s = harness.capture_decode(
                    server_python, base_url=base_url, model=model,
                    out_file=df, summary_file=sf, dataset=ds,
                    num_prompts=args.num_prompts, output_len=args.output_len,
                    timeout_s=3600,
                )
                off1 = log.stat().st_size if log.exists() else 0
                tps = decode_tps(s)
                rec: dict = {
                    "stratum": stratum,
                    "net_tps": tps,
                    "num_completion_tokens": s.get("num_completion_tokens"),
                    "duration_s": s.get("duration_s"),
                    "num_records": s.get("num_records"),
                }
                if num_spec:
                    m_after = snapshot_spec(fetch_metrics(base_url))
                    rec["accept_prom"] = stratum_accept(m_before, m_after, num_spec)
                    rec["accept_log"] = log_slice_accept(log, off0, off1)
                    prom = rec["accept_prom"]
                    rec["accept_rate"] = (prom.get("accept_rate")
                                          if prom.get("accept_rate") not in (None, 0)
                                          else rec["accept_log"].get("accept_rate"))
                    rec["mean_accept_len"] = (prom.get("mean_accept_len")
                                              if prom.get("mean_accept_len") not in (None, 0)
                                              else rec["accept_log"].get("mean_accept_len"))
                    rec["per_pos_accept"] = prom.get("per_pos_accept")
                result["per_stratum"][stratum] = rec
                print(f"[disp] {stratum:5s} net_tps={tps:.3f} "
                      f"toks={s.get('num_completion_tokens')} dur={s.get('duration_s'):.2f}s "
                      f"accept_rate={rec.get('accept_rate')}", flush=True)

            if args.selfdet:
                stratum = strata[0]
                df2 = OUT / f"decode_{tag}_{stratum}_selfdet.jsonl"
                sf2 = OUT / f"decode_{tag}_{stratum}_selfdet.summary.json"
                harness.capture_decode(
                    server_python, base_url=base_url, model=model,
                    out_file=df2, summary_file=sf2, dataset=STRATA[stratum],
                    num_prompts=args.num_prompts, output_len=args.output_len,
                    timeout_s=3600,
                )
                a = sha_map(OUT / f"decode_{tag}_{stratum}.jsonl")
                b = sha_map(df2)
                common = sorted(set(a) & set(b))
                ident = sum(1 for i in common if a[i] == b[i])
                result["self_det"] = {
                    "stratum": stratum,
                    "sequences": len(common),
                    "sequences_identical": ident,
                    "self_det": (ident / len(common)) if common else None,
                }
                print(f"[disp] self_det({stratum})={result['self_det']['self_det']}",
                      flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = "".join(
            traceback.format_exception(type(e), e, e.__traceback__))[-6000:]
        print(f"[disp] EXCEPTION serve_ok={result['serve_ok']}\n{result['error']}",
              flush=True)
    finally:
        stop.set()
        gpu_thread.join(timeout=5)

    result["peak_gpu_mib"] = peak["mib"]
    out_json = OUT / f"results_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"[disp] wrote {out_json}", flush=True)
    print("DISP_RESULT " + json.dumps(
        {"drafter": args.drafter, "serve_ok": result["serve_ok"],
         "per_stratum": {k: {"net_tps": v["net_tps"],
                             "accept_rate": v.get("accept_rate")}
                         for k, v in result["per_stratum"].items()}},
        default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
