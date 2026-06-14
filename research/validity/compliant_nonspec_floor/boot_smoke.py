"""Boot smoke for the non-spec int4 throwaway serve (lawine #196).

Confirms `submissions/fa2sw_nonspec_int4` boots, serves M=1 AR (speculative_config=None,
int4 quant ON, cudagraphs ON), and completes a tiny decode. Fast fail before the full
128-prompt runs. LOCAL only, no HF.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

SUB = ROOT / "submissions" / "fa2sw_nonspec_int4"
OUT = ROOT / "research" / "validity" / "compliant_nonspec_floor" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[smoke] {note}", flush=True)
    manifest = harness.load_manifest(SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    log_path = OUT / "smoke_server.log"
    t0 = time.time()
    result = {"submission": str(SUB), "boots": False}
    with harness.LocalServer(SUB, server_python=server_python, port=8000,
                             log_path=log_path) as srv:
        ready_s = time.time() - t0
        result["server_ready_s"] = ready_s
        result["boots"] = True
        result["served_model_name"] = srv.served_model_name
        result["model_id"] = srv.model_id
        print(f"[smoke] server ready in {ready_s:.0f}s; tiny decode (2 prompts x 16 tok)", flush=True)
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=OUT / "smoke_decode.jsonl", summary_file=OUT / "smoke_decode_summary.json",
            num_prompts=2, output_len=16,
        )
        result["decode_records"] = summary["num_records"]
        result["decode_completion_tokens"] = summary["num_completion_tokens"]

    log_text = log_path.read_text(errors="replace")
    # Confirm the engine started with no speculation and int4 + cudagraphs on.
    m = re.search(r"speculative_config=(\w+)", log_text)
    result["speculative_config"] = m.group(1) if m else "NOT_FOUND"
    result["quantization_compressed_tensors"] = "quantization=compressed-tensors" in log_text
    result["enforce_eager_false"] = "enforce_eager=False" in log_text
    result["precache_skipped_ungated"] = "skipping precache, ungating" in log_text
    # We blanked SPECULATIVE_CONFIG directly, so the reference-mode "clearing" line must NOT appear.
    result["reference_mode_clearing_line"] = "SENPAI_REFERENCE_MODE active: clearing" in log_text

    result["spec_off_ok"] = (
        result["speculative_config"] == "None"
        and result["quantization_compressed_tensors"]
        and result["enforce_eager_false"]
        and result["decode_records"] == 2
    )
    (OUT / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print("[smoke] RESULT:", json.dumps(result, indent=2), flush=True)
    print(f"[smoke] spec_off_ok={result['spec_off_ok']} boots={result['boots']}", flush=True)
    return 0 if result["spec_off_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
