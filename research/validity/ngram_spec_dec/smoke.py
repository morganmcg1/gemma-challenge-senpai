"""Tiny smoke for PR #503: prove the deployed fa2sw stack serves an n-gram
(prompt-lookup) drafter locally, and learn which acceptance source is reliable.

Starts the *deployed* fa2sw_precache_kenyan submission (the MTP-K7 base) but
swaps ONLY SPECULATIVE_CONFIG to an ngram drafter, on a 4-prompt x 64-token
workload. Dumps the server-log SpecDecoding counters AND the Prometheus
/metrics spec counters so we know which one to build the real driver around.

Local A10G probe — NOT the official a10g-small TPS.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")
OUT = ROOT / "research" / "validity" / "ngram_spec_dec" / "_smoke"


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[smoke] {note}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    spec = {"method": "ngram", "num_speculative_tokens": 5,
            "prompt_lookup_max": 3, "prompt_lookup_min": 2}
    extra_env = {
        "SPECULATIVE_CONFIG": json.dumps(spec),
        "DISABLE_LOG_STATS": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    log_path = OUT / "server.log"
    print(f"[smoke] spec_config={spec}", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1200,
    ) as srv:
        print(f"[smoke] server up in {time.time()-t0:.0f}s; running 4x64 decode", flush=True)
        summary = harness.capture_decode(
            SERVER_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=OUT / "decode.jsonl", summary_file=OUT / "decode.summary.json",
            num_prompts=4, output_len=64, timeout_s=600,
        )
        print(f"[smoke] decode summary: {json.dumps(summary)[:300]}", flush=True)
        # Prometheus spec counters (whole-run, exact) before teardown.
        try:
            with urllib.request.urlopen(f"{srv.base_url}/metrics", timeout=30) as r:
                metrics_text = r.read().decode("utf-8", "replace")
            (OUT / "metrics.txt").write_text(metrics_text)
            prom = serve_profile.parse_spec_metrics(metrics_text)
        except Exception as exc:  # noqa: BLE001
            prom = {"error": str(exc)}
        print(f"[smoke] prometheus spec metrics: {json.dumps(prom)}", flush=True)
    log_text = log_path.read_text()
    spec_log = serve_profile.parse_spec_log(log_text)
    print(f"[smoke] server-log spec metrics: {json.dumps(spec_log)}", flush=True)
    # Show whether vLLM accepted the ngram speculative_config at engine init.
    for line in log_text.splitlines():
        if "speculative" in line.lower() or "ngram" in line.lower() or "prompt_lookup" in line.lower():
            print(f"[smoke][log] {line[:200]}", flush=True)
    print("[smoke] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
