"""PPL validity pass for the non-spec int4 throwaway serve (lawine #196, Task 4b).

Serves `submissions/fa2sw_nonspec_int4` once (spec OFF, int4 ON) and runs the
official ppl_endpoint.py to get `ppl_nonspec`. Compliance threshold: PPL <= 2.42
(baseline served PPL 2.3772). Decode-only TPS is measured separately by
paired_tps_ab; this is the unchanged-stack validity check. LOCAL only, no HF.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

SUB = ROOT / "submissions" / "fa2sw_nonspec_int4"
OUT = ROOT / "research" / "validity" / "compliant_nonspec_floor" / "ppl"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[ppl] {note}", flush=True)
    manifest = harness.load_manifest(SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    log_path = OUT / "ppl_server.log"
    t0 = time.time()
    result: dict = {"submission": str(SUB)}
    with harness.LocalServer(SUB, server_python=server_python, port=8000,
                             log_path=log_path) as srv:
        result["server_ready_s"] = time.time() - t0
        result["served_model_name"] = srv.served_model_name
        print(f"[ppl] server ready in {result['server_ready_s']:.0f}s; running PPL", flush=True)
        ppl = harness.run_ppl(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=OUT / "ppl_results.jsonl", summary_file=OUT / "ppl_summary.json",
        )
    result["ppl_nonspec"] = ppl.get("ppl")
    result["ppl_num_records"] = ppl.get("num_records")
    # Confirm spec really off in this serve too (defense-in-depth).
    log_text = log_path.read_text(errors="replace")
    result["speculative_config_none"] = "speculative_config=None" in log_text
    result["ppl_le_2_42"] = isinstance(result["ppl_nonspec"], (int, float)) and result["ppl_nonspec"] <= 2.42
    (OUT / "ppl_check_result.json").write_text(json.dumps(result, indent=2))
    print("[ppl] RESULT:", json.dumps(result, indent=2), flush=True)
    return 0 if result["ppl_le_2_42"] and result["speculative_config_none"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
