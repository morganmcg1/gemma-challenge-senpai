"""Diagnose base_fullhead empty-completion pathology (PR #541).

The truefullhead GSM8K run (surgical fast kernels on the stock 262k-head base-int4)
returned EMPTY completions (0 chars, finish_reason='stop', extract_mode='none') on
~10% (sampled) / ~15% (greedy) of items; base produces ZERO empties. This probe
stands up the SAME base_fullhead server and replays a handful of known-empty item
ids (plus a few known-good as controls) capturing the FULL raw response -- so we can
tell whether the empties are:

  * immediate-EOS  -- usage.completion_tokens in {0,1}, content '' -> a genuine
    fast-kernel first-token-EOS quality pathology on the full head; or
  * thinking-leak  -- content '' but message.reasoning_content non-empty -> a
    harness/chat-template artifact, the reasoning is actually fine.

Replays each id under greedy (matches the eval's diagnostic regime) and one id
under enable_thinking=True to see if a thinking channel captures the body.

LOCAL only, analysis-only, no served-file change. Same serve overrides as the eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gsm8k_eval as G  # the committed harness (prompt build + chat_completion)


def _full_split_by_id() -> dict[str, dict]:
    full = G._load_split("test", n=None)
    return {f"test-{i}": full[i] for i in range(len(full))}


def _probe_one(base_url, model, prompt, *, greedy: bool, enable_thinking: bool, seed: int):
    if greedy:
        s = dict(temperature=0.0, top_p=1.0, top_k=-1)
    else:
        s = dict(temperature=1.0, top_p=0.95, top_k=64)
    resp = G.chat_completion(
        base_url, model, G.build_messages(prompt),
        max_tokens=512, seed=seed, enable_thinking=enable_thinking,
        timeout_s=600, **s,
    )
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    usage = resp.get("usage", {}) or {}
    return {
        "finish_reason": choice.get("finish_reason"),
        "content_len": len(msg.get("content") or ""),
        "content_head": (msg.get("content") or "")[:160],
        "reasoning_content_len": len(msg.get("reasoning_content") or ""),
        "reasoning_head": (msg.get("reasoning_content") or "")[:160],
        "msg_keys": sorted(msg.keys()),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submissions/fa2sw_strict_surgical357")
    ap.add_argument("--empty-ids", required=True, help="comma list of test-<i> known-empty ids")
    ap.add_argument("--good-ids", default="", help="comma list of control ids")
    ap.add_argument("--serve-env", action="append", default=[])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    by_id = _full_split_by_id()
    exemplars, _ = G.build_fewshot(8, args.seed)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.local_validation import harness, paths

    for note in paths.prepare_local_gpu_env():
        print(f"[probe] {note}", flush=True)
    manifest = harness.load_manifest(Path(args.submission))
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    overrides = {"PRECACHE_BENCH": "0", "PRECACHE_REQUIRE": "0",
                 "PRECACHE_DATASET": "/tmp/senpai_gsm8k_no_precache.json", "MAX_NUM_SEQS": "32"}
    for kv in args.serve_env:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v
    log_path = Path("research/downstream_quality_gsm8k/server_empty_probe.log")
    print(f"[probe] serving {args.submission} overrides={overrides}", flush=True)

    empty_ids = [s for s in args.empty_ids.split(",") if s]
    good_ids = [s for s in args.good_ids.split(",") if s]

    with harness.LocalServer(Path(args.submission), server_python=server_python, port=args.port,
                             startup_timeout_s=1800, log_path=log_path, extra_env=overrides) as srv:
        out = {"empty": {}, "good": {}}
        for tag, ids in (("empty", empty_ids), ("good", good_ids)):
            for tid in ids:
                q = by_id[tid]["question"]
                prompt = G.build_prompt(exemplars, q)
                rec = {"greedy": _probe_one(srv.base_url, srv.served_model_name, prompt,
                                            greedy=True, enable_thinking=False, seed=args.seed)}
                if tag == "empty":
                    rec["sampled"] = _probe_one(srv.base_url, srv.served_model_name, prompt,
                                                greedy=False, enable_thinking=False, seed=args.seed)
                    rec["greedy_thinking"] = _probe_one(srv.base_url, srv.served_model_name, prompt,
                                                        greedy=True, enable_thinking=True, seed=args.seed)
                out[tag][tid] = rec
                g = rec["greedy"]
                print(f"[probe] {tag} {tid}: greedy finish={g['finish_reason']} "
                      f"comp_tok={g['completion_tokens']} content_len={g['content_len']} "
                      f"reasoning_len={g['reasoning_content_len']} msg_keys={g['msg_keys']}", flush=True)
        Path("research/downstream_quality_gsm8k/empty_probe.json").write_text(json.dumps(out, indent=2))
        print("[probe] wrote research/downstream_quality_gsm8k/empty_probe.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
