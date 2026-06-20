#!/usr/bin/env python
"""PR #805 Step 5 — publish the validated int4head+PLE-dequant build to a
PRIVATE Hub repo (mirror #802), so the submission is a one-step fire candidate.

LOCAL artifact publish only (model weights upload). This is NOT an HF Job and
NOT a competition submission — no /v1/jobs:run, no train.py --launch.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

BUILD = Path("/workspace/gemma_build/bi0_int4head_pledequant")
REPO_ID = "gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head-pledequant"
OUT = Path("/workspace/senpai/target/research/int4head_ple_dequant/publish_hub.json")


def main() -> int:
    assert BUILD.exists(), f"build dir missing: {BUILD}"
    st = BUILD / "model.safetensors"
    assert st.exists(), f"model.safetensors missing in {BUILD}"
    print(f"[publish] build={BUILD} ({st.stat().st_size/1e9:.2f} GB safetensors)", flush=True)
    api = HfApi()
    print(f"[publish] create_repo {REPO_ID} (private, model)", flush=True)
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    t0 = time.time()
    print(f"[publish] upload_folder {BUILD} -> {REPO_ID} ...", flush=True)
    info = upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(BUILD),
        commit_message="PR #805: int4head + PLE-input-gate de-quant (bf16/cuBLAS) build",
    )
    dt = time.time() - t0
    oid = getattr(info, "oid", None) or getattr(info, "commit_id", None) or str(info)
    print(f"[publish] DONE in {dt:.0f}s  commit={oid}", flush=True)
    # Confirm the model is resolvable + read back the file list.
    mi = api.model_info(REPO_ID)
    files = sorted(s.rfilename for s in mi.siblings)
    rec = {
        "repo_id": REPO_ID,
        "private": getattr(mi, "private", None),
        "sha": mi.sha,
        "commit": oid,
        "upload_s": round(dt, 1),
        "files": files,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(rec, indent=2))
    print(f"[publish] repo sha={mi.sha} private={rec['private']} files={files}", flush=True)
    print(f"[publish] record -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
