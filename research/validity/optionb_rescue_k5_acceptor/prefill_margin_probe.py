#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #669 -- prefill-margin probe for the 2 matched-basis pos-0 spec-vs-AR
divergences.

The matched cascade found exactly 2/128 prompts where the K=5 spec serve's first
emitted token differs from the bit-exact int4-AR reference (AR-vs-AR floor = 0).
Both flip to token 8291 ('Here'), the model's dominant MMLU opening (~56/128). The
acceptor cannot touch these (no draft row exists at the prefill bonus position), so
they are structurally outside the recompute-acceptor's domain -- but the advisor's
strict #319 ruling requires PROVING they are a borderline-tie artifact, not asserting
it. This probe MEASURES the pos-0 top-k logit margin under BOTH configs:

  AR config   (SENPAI_REFERENCE_MODE=1, drafter OFF) -- the reference numerics
  spec config (drafter ON, num_spec=5)               -- the submission numerics

For each of the 2 prompts we read the pos-0 logprobs of the AR-pick token and of
'Here' (8291). If the AR-config gap (lp[AR_pick] - lp[Here]) is within the FP
perturbation scale, the drafter's presence flipping the argmax is a near-tie
artifact (same class as the within-stack numerical sensitivity the greedy gate
tolerates), NOT a semantic divergence. analysis_only, NO HF Job.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

HERE_TOKEN = 8291  # 'Here' -- the common MMLU opening both prompts flip to


def load_prompts(spec_jsonl: Path, ar_jsonl: Path, key_prefixes: set[str]):
    """Return [{key, prompt_token_ids, ar_first, spec_first}] for the named prompts."""
    spec = {}
    ar = {}
    for store, p in ((spec, spec_jsonl), (ar, ar_jsonl)):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                k = r.get("prompt_token_sha256", "")
                if k[:8] in key_prefixes:
                    store[k] = r
    out = []
    for k, sr in spec.items():
        ar_r = ar.get(k, {})
        out.append({
            "key": k,
            "prompt_token_ids": sr.get("prompt_token_ids") or ar_r.get("prompt_token_ids"),
            "ar_first": (ar_r.get("completion_token_ids") or [None])[0],
            "spec_first": (sr.get("completion_token_ids") or [None])[0],
        })
    return out


def query_logprobs(base_url: str, model: str, prompt_token_ids: list[int],
                   n_logprobs: int = 20, timeout_s: int = 120) -> dict:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
        "add_special_tokens": False,
        "ignore_eos": True,
        "return_token_ids": True,
        "logprobs": n_logprobs,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def parse_pos0(resp: dict) -> dict:
    """Pull the pos-0 top-k token->logprob map from a vLLM completions response.

    vLLM returns choices[0].logprobs with `tokens` (list[str]), `token_logprobs`
    (list[float]), and `top_logprobs` (list[dict[str,float]]); with return_token_ids
    it may also expose integer ids. We key on whatever ids/strings are available and
    return {argmax_id, argmax_logprob, top: {tokenstr: lp}, top_ids: {id: lp}}."""
    ch = resp["choices"][0]
    lp = ch.get("logprobs") or {}
    top_list = lp.get("top_logprobs") or []
    top0 = top_list[0] if top_list else {}
    # token id list (return_token_ids) aligned to generated tokens
    gen_ids = ch.get("token_ids") or (lp.get("token_ids") if isinstance(lp.get("token_ids"), list) else None)
    argmax_id = None
    if gen_ids:
        # the single generated token id (skip the prompt echo if present)
        argmax_id = gen_ids[-1] if len(gen_ids) == 1 else gen_ids[len(gen_ids) - 1]
    return {
        "argmax_id": argmax_id,
        "tokens": lp.get("tokens"),
        "token_logprobs": lp.get("token_logprobs"),
        "top0": top0,
        "raw_logprobs_keys": list(lp.keys()),
    }


def run_config(label: str, sub_dir: Path, server_python: Path, extra_env: dict,
               prompts: list, out_dir: Path) -> dict:
    log_path = out_dir / f"server_{label}.log"
    results = []
    with harness.LocalServer(sub_dir, server_python=server_python, port=8000,
                             log_path=log_path, extra_env=extra_env) as srv:
        model = srv.served_model_name
        for pr in prompts:
            resp = query_logprobs(srv.base_url, model, pr["prompt_token_ids"])
            parsed = parse_pos0(resp)
            results.append({"key": pr["key"][:12], "ar_first": pr["ar_first"],
                            "spec_first": pr["spec_first"], "parsed": parsed,
                            "raw_choice": resp["choices"][0]})
            print(f"[{label}] key={pr['key'][:12]} ar_first={pr['ar_first']} "
                  f"spec_first={pr['spec_first']} argmax_id={parsed['argmax_id']} "
                  f"top0_keys={list(parsed['top0'])[:6]}", flush=True)
    return {"label": label, "extra_env": extra_env, "results": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="int4_mtp_batchinv")
    ap.add_argument("--matched-dir", type=Path,
                    default=ROOT / "research/validity/optionb_rescue_k5_acceptor/matched_k5")
    ap.add_argument("--keys", default="37227f6b,74200cad")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "research/validity/optionb_rescue_k5_acceptor/prefill_margin")
    a = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[probe] {note}", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = (ROOT / "submissions" / a.submission).resolve()
    manifest = harness.load_manifest(sub_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    key_prefixes = set(a.keys.split(","))
    prompts = load_prompts(a.matched_dir / "spec/decode/run00.jsonl",
                           a.matched_dir / "ar_a/decode/run00.jsonl", key_prefixes)
    print(f"[probe] loaded {len(prompts)} prompts: {[p['key'][:12] for p in prompts]}", flush=True)

    base_spec = {"NUM_SPECULATIVE_TOKENS": "5", "SENPAI_RECOMPUTE_CUDAGRAPH": "1"}
    configs = [
        ("ar", {**base_spec, "SENPAI_REFERENCE_MODE": "1"}),
        ("spec", dict(base_spec)),
    ]
    out = {"pr": 669, "leg": "prefill_margin", "analysis_only": True,
           "official_tps": 0, "no_hf_job": True, "here_token": HERE_TOKEN,
           "configs": []}
    for label, env in configs:
        out["configs"].append(run_config(label, sub_dir, server_python, env, prompts, a.out_dir))

    (a.out_dir / "prefill_margin.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[probe] wrote {a.out_dir/'prefill_margin.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
