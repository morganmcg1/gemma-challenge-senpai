#!/usr/bin/env python
"""PR #535 Phase 1 — base int4 (native 262k head) + fast stack: serve gate, warm-median TPS, self-det, PPL.

The load-bearing question of #535: does the fast kernel stack (surgical 2D
attention armed + MTP K=7 drafter spec-on + split-KV) *serve at all* on the base
int4 substrate (stock ``google/gemma-4-E4B-it-qat-w4a16-ct``, native 262,144-row
``lm_head``, NO osoi5 bake, NO head prune)? And if so, at what single-stream TPS
and PPL?

We compose the fast stack by serving the surgical-357 submission with
substrate-swap overrides:
  * LOCAL_MODEL_DIR / PLE_FOLD_TARGET_MODEL -> the base int4 snapshot (native head)
  * LM_HEAD_PRUNE=0 (no 16k->12k prune; base is already full 262k)
  * PCK04_KEEPSET="" (disable the pruned-head logits scatter; native head needs none)
All attention/spec/graph levers stay on (surgical 2D attn, MTP K=7, split-KV,
onegraph, fused-sparse-argmax-on-drafter) — they patch attention + the drafter,
not the target lm_head, so they are substrate-agnostic by construction.

``--substrate osoi5_ship`` reproduces the shipped surgical-357 osoi5-12k stack
unchanged (manifest defaults) so the base-int4 TPS Δ is measured against an
osoi5 number taken with the IDENTICAL local method on this same pod, not only
against the cited 357.06.

TPS here = single-stream (concurrency 1) output throughput over the official
128×512 decode workload (decode_outputs.py: temperature 0, ignore_eos, 512 toks,
seed 1) = num_completion_tokens / duration_s. Two warm passes; we report the
median (== mean for n=2) plus both. This mirrors the official MAX_CONCURRENCY=1
serving benchmark and the README "warm median of two runs" convention.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from statistics import median

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

BASE_INT4 = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
OSOI5_DIR = "/tmp/osoi5-v0-baked"
SUB = ROOT / "submissions" / "fa2sw_strict_surgical357"
OUT = ROOT / "research" / "base_fullhead_fast_probe"


def substrate_overrides(substrate: str) -> dict[str, str]:
    # Deployed single-stream config for an honest TPS (matches official conc=1).
    common = {
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_aime_no_precache.json",
        "MAX_NUM_SEQS": "1",
        "MAX_NUM_BATCHED_TOKENS": "512",
    }
    if substrate in ("base_fullhead", "base_fullhead_specoff"):
        common.update(
            {
                "LOCAL_MODEL_DIR": BASE_INT4,
                "PLE_FOLD_TARGET_MODEL": BASE_INT4,
                "LM_HEAD_PRUNE": "0",
                "LM_HEAD_PRUNE_REQUIRE": "0",
                "PCK04_KEEPSET": "",  # native 262k head -> no scatter-back
            }
        )
        if substrate == "base_fullhead_specoff":
            # M=1 AR greedy reference: drafter/spec OFF, everything else (surgical
            # 2D attn, split-KV, PLE fold) stays on. Isolates whether the MTP
            # spec-accept path is what breaks greedy identity on the native head.
            common["SENPAI_REFERENCE_MODE"] = "1"
    elif substrate == "osoi5_ship":
        # Manifest defaults reproduce the shipped surgical-357 osoi5-12k stack.
        pass
    else:
        raise ValueError(f"unknown substrate {substrate!r}")
    return common


def _completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 300) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def decode_tps(summary: dict) -> float:
    dur = summary.get("duration_s") or 0.0
    toks = summary.get("num_completion_tokens") or 0
    return toks / dur if dur > 0 else float("nan")


def self_det(decode_r1: Path, decode_r2: Path) -> dict:
    def load(p: Path) -> dict[int, str]:
        out = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["index"]] = row["completion_token_sha256"]
        return out

    a, b = load(decode_r1), load(decode_r2)
    common = sorted(set(a) & set(b))
    seq_match = sum(1 for i in common if a[i] == b[i])
    return {
        "sequences": len(common),
        "sequences_identical": seq_match,
        "self_det": (seq_match / len(common)) if common else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=["base_fullhead", "base_fullhead_specoff", "osoi5_ship"], required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--skip-ppl", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.substrate
    for note in paths.prepare_local_gpu_env():
        print(f"[probe] {note}", flush=True)

    manifest = harness.load_manifest(SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    overrides = substrate_overrides(args.substrate)

    result: dict = {
        "substrate": tag,
        "submission": str(SUB.relative_to(ROOT)),
        "serve_overrides": overrides,
        "serve_ok": False,
        "kernel_incompat_detail": None,
    }

    peak = {"mib": 0}
    stop = threading.Event()

    def sample_gpu() -> None:
        while not stop.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                m = int(o.stdout.strip().splitlines()[0])
                peak["mib"] = max(peak["mib"], m)
            except Exception:
                pass
            time.sleep(2)

    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    gpu_thread.start()

    log = OUT / f"server_{tag}.log"
    try:
        with harness.LocalServer(
            SUB,
            server_python=server_python,
            port=args.port,
            startup_timeout_s=1800,
            log_path=log,
            extra_env=overrides,
        ) as srv:
            result["serve_ok"] = True
            model = srv.served_model_name
            base_url = srv.base_url
            result["served_model_name"] = model
            print(f"[probe] serve_ok=True model={model}", flush=True)

            # Known-answer self-test (tiny greedy) before the full workload.
            st = _completion(base_url, model, "The capital of France is", 6)
            st_text = (st.get("choices") or [{}])[0].get("text", "")
            result["self_test_prompt"] = "The capital of France is"
            result["self_test_text"] = st_text
            result["self_test_ok"] = "Paris" in st_text
            print(f"[probe] self_test text={st_text!r} ok={result['self_test_ok']}", flush=True)

            # warmup (trigger onegraph capture / JIT) before timed passes
            _completion(base_url, model, "Explain how a transformer decodes one token at a time.", 16)

            # Two warm 128×512 decode passes.
            tps_runs = []
            decode_files = []
            for i in (1, 2):
                df = OUT / f"decode_{tag}_r{i}.jsonl"
                sf = OUT / f"decode_{tag}_r{i}.summary.json"
                s = harness.capture_decode(
                    server_python, base_url=base_url, model=model,
                    out_file=df, summary_file=sf, timeout_s=3600,
                )
                tps = decode_tps(s)
                tps_runs.append(tps)
                decode_files.append(df)
                print(f"[probe] decode r{i}: tps={tps:.3f} "
                      f"completion_tokens={s.get('num_completion_tokens')} dur={s.get('duration_s'):.2f}s "
                      f"records={s.get('num_records')}", flush=True)

            result["tps_runs"] = tps_runs
            result["warm_median_tps"] = median(tps_runs)
            result.update(self_det(decode_files[0], decode_files[1]))
            print(f"[probe] warm_median_tps={result['warm_median_tps']:.3f} "
                  f"self_det={result['self_det']}", flush=True)

            if not args.skip_ppl:
                ppl_sum = harness.run_ppl(
                    server_python, base_url=base_url, model=model,
                    out_file=OUT / f"ppl_{tag}.jsonl",
                    summary_file=OUT / f"ppl_{tag}.summary.json",
                    timeout_s=1800,
                )
                result["ppl"] = ppl_sum.get("ppl")
                result["ppl_num_tokens"] = ppl_sum.get("num_tokens")
                print(f"[probe] ppl={result['ppl']} num_tokens={result['ppl_num_tokens']}", flush=True)
    except Exception as e:  # capture the verbatim incompat for the gate
        result["kernel_incompat_detail"] = "".join(
            traceback.format_exception(type(e), e, e.__traceback__)
        )[-6000:]
        print(f"[probe] EXCEPTION serve_ok={result['serve_ok']}\n{result['kernel_incompat_detail']}", flush=True)
    finally:
        stop.set()
        gpu_thread.join(timeout=5)

    result["peak_gpu_mib"] = peak["mib"]
    out_json = OUT / f"phase1_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"[probe] wrote {out_json}", flush=True)
    printable = {k: v for k, v in result.items() if k != "kernel_incompat_detail"}
    print("PHASE1 " + json.dumps(printable, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
