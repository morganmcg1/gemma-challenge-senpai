#!/usr/bin/env python
"""PR #443 control — cross-session determinism attribution.

The main probe found graph_on vs graph_off greedy identity = 127/128 prompts
(1 divergent, onset idx 105, 23-token suffix cascade). Late+rare onset is the
floating-point-nondeterminism signature (greedy_gate.onset_summary docstring:
structural bugs diverge EARLY and on MOST prompts). This control captures a
SECOND graph_on serve process (same config, ONEGRAPH=1) and compares it against
the first graph_on capture: if on-vs-on shows the SAME late+rare divergence, the
loop-graph is exonerated — the blemish is inherent cross-session FP noise
(SPLITKV_VERIFY atomics / near-tie argmax), not the capture.

Capture-only (no timing window, no PPL). Analysis-only; changes no served file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402
from scripts.profiler import prefill_denominator_probe as pdp  # noqa: E402

OUT = ROOT / "research" / "validity" / "cudagraph_capture_specloop"
SUB = ROOT / "submissions" / "fa2sw_precache_kenyan"
NUM, OLEN, SEED, PORT = 128, 128, paths.SEED, 8000


def capture_arm(tag: str, arm_env: dict[str, str], server_python: Path) -> Path:
    log_path = OUT / f"server_{tag}.log"
    guard = pdp._install_serve_guard(server_python)
    extra_env = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
        "PREFILL_PROBE_GUARD": "1",
        "PRECACHE_DATASET": str(paths.EVAL_PROMPTS),
        **arm_env,
    }
    cap = OUT / f"decode_{tag}.jsonl"
    try:
        with harness.LocalServer(
            SUB, server_python=server_python, port=PORT,
            log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
        ) as srv:
            time.sleep(1.0)
            harness.capture_decode(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=cap, summary_file=OUT / f"decode_{tag}_summary.json",
                num_prompts=NUM, output_len=OLEN, seed=SEED,
            )
    finally:
        for f in guard:
            f.unlink(missing_ok=True)
    print(f"[{tag}] captured -> {cap}", flush=True)
    return cap


def cmp(ref: Path, cand: Path, label: str) -> dict:
    rep = greedy_gate.compare(ref, cand)
    onset = greedy_gate.onset_summary(rep)
    out = {
        "label": label, "verdict": rep.verdict,
        "num_identical": rep.num_identical, "num_prompts_compared": rep.num_prompts_compared,
        "total_divergent_tokens": rep.total_divergent_tokens,
        "total_tokens_compared": rep.total_tokens_compared,
        "onsets": onset.get("onsets"), "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"), "onset_max": onset.get("onset_max"),
        "byte_exact": bool(rep.verdict == "GREEDY_IDENTICAL"),
    }
    print(f"[cmp:{label}] {out['verdict']} {out['num_identical']}/{out['num_prompts_compared']} "
          f"divtok={out['total_divergent_tokens']}/{out['total_tokens_compared']} onsets={out['onsets']}", flush=True)
    return out


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[gpu-env] {note}", flush=True)
    manifest = harness.load_manifest(SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    on_b = capture_arm("graph_on_b", {}, server_python)  # second graph_on process
    on_a = OUT / "decode_graph_on.jsonl"
    off = OUT / "decode_graph_off.jsonl"

    res = {
        "pr": 443, "analysis_only": True, "control": "cross_session_determinism",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": {
            "on_a_vs_on_b_same_config": cmp(on_a, on_b, "on_vs_on (same config, 2 processes)"),
            "off_vs_on_b": cmp(off, on_b, "off_vs_on_b (graph flip + 2 processes)"),
        },
    }
    (OUT / "control_xsession.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"[write] {OUT / 'control_xsession.json'}", flush=True)

    a = res["comparisons"]["on_a_vs_on_b_same_config"]
    interp = ("EXONERATED: same-config on-vs-on shows the SAME late+rare divergence -> "
              "the 1-prompt graph_on/off blemish is cross-session FP nondeterminism, not the capture"
              if not a["byte_exact"] and (a.get("onset_min") or 0) >= 64 else
              ("on-vs-on is byte-exact -> the graph_on/off divergence is attributable to the graph flip itself"
               if a["byte_exact"] else
               "on-vs-on diverges EARLY -> investigate (not a clean FP-noise story)"))
    res["interpretation"] = interp
    (OUT / "control_xsession.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[INTERPRETATION] {interp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
