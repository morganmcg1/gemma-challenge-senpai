"""PR #503 — n-gram prompt-lookup spec-dec probe driver.

Drives the *deployed* fa2sw_precache_kenyan stack (the MTP-K7 base), swapping
ONLY the drafter via SPECULATIVE_CONFIG. The verify-side stack (int4 / fa2sw /
precache / split-KV / lmhead12k / loopgraph) is byte-identical across configs;
only the drafter changes, so any TPS/acceptance difference is attributable to
the drafter alone, and the M=8-vs-M=1 verify taxes are the same as MTP.

Per config it starts a fresh server (drafter is fixed at engine init), decodes
the prompt set, and records:
  - acceptance_rate (accepted/drafted) + e_accept (mean acceptance length)
    from BOTH Prometheus /metrics and vLLM's own server-log SpecDecoding lines
  - steady decode TPS (vLLM's whole-run "Avg generation throughput" meter)
  - per-prompt completion token IDs + sha256 (for the operative-identity census)
  - optional same-endpoint PPL

Checkpoints the results JSON after every config so a 90-min wall never loses
finished work (resume skips configs already present).

Prompt sources:
  - "public": official ShareGPT-128 via decode_outputs.py (canonical capture)
  - <path>.jsonl: token-id prompts (`context_token_ids`), POSTed directly

Local A10G exploratory probe — NOT the official a10g-small TPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")


def _sha_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def _post(base_url: str, payload: dict[str, Any], timeout_s: int = 600) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def _completion_token_ids(resp: dict[str, Any], prompt_token_ids: list[int]) -> list[int]:
    ch = (resp.get("choices") or [{}])[0]
    for v in (ch.get("token_ids"), ch.get("output_token_ids"),
              (ch.get("logprobs") or {}).get("token_ids") if isinstance(ch.get("logprobs"), dict) else None):
        if isinstance(v, list) and all(isinstance(t, int) for t in v):
            if len(v) >= len(prompt_token_ids) and v[:len(prompt_token_ids)] == prompt_token_ids:
                return v[len(prompt_token_ids):]
            return v
    return []


def decode_jsonl_prompts(
    base_url: str, model: str, jsonl_path: Path, *, num_prompts: int, output_len: int,
) -> dict[str, Any]:
    """POST token-id prompts (`context_token_ids`) directly; capture completions."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:num_prompts]
    per_prompt = []
    total_completion = 0
    t0 = time.time()
    for i, rec in enumerate(rows):
        ptoks = rec.get("context_token_ids") or rec.get("prompt_token_ids")
        if not ptoks:
            continue
        payload = {
            "model": model, "prompt": ptoks, "max_tokens": output_len,
            "temperature": 0.0, "stream": False, "add_special_tokens": False,
            "ignore_eos": True, "return_token_ids": True,
        }
        resp = _post(base_url, payload)
        ctoks = _completion_token_ids(resp, ptoks)
        total_completion += len(ctoks)
        per_prompt.append({
            "id": rec.get("id", i), "index": i,
            "num_completion_tokens": len(ctoks),
            "completion_token_sha256": _sha_tokens(ctoks),
            "completion_token_ids": ctoks,
        })
    return {
        "num_records": len(per_prompt),
        "num_completion_tokens": total_completion,
        "duration_s": time.time() - t0,
        "output_len": output_len,
        "per_prompt": per_prompt,
    }


def decode_public_prompts(
    base_url: str, model: str, out_dir: Path, label: str, *, num_prompts: int, output_len: int,
) -> dict[str, Any]:
    """Official decode_outputs.py over ShareGPT-128 (canonical token-id capture)."""
    decode_out = out_dir / f"decode_{label}.jsonl"
    decode_summary = out_dir / f"decode_{label}.summary.json"
    summary = harness.capture_decode(
        SERVER_PY, base_url=base_url, model=model,
        out_file=decode_out, summary_file=decode_summary,
        num_prompts=num_prompts, output_len=output_len, timeout_s=3600,
    )
    per_prompt = []
    with open(decode_out) as f:
        for line in f:
            row = json.loads(line)
            per_prompt.append({
                "id": row["id"], "index": row["index"],
                "num_completion_tokens": row["num_completion_tokens"],
                "completion_token_sha256": row["completion_token_sha256"],
                "completion_token_ids": row["completion_token_ids"],
            })
    summary["per_prompt"] = per_prompt
    return summary


def run_config(
    name: str, spec_config: str, prompt_source: str, out_dir: Path,
    *, num_prompts: int, output_len: int, do_ppl: bool,
) -> dict[str, Any]:
    extra_env = {
        "SPECULATIVE_CONFIG": spec_config,
        "DISABLE_LOG_STATS": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    log_path = out_dir / f"server_{name}.log"
    rec: dict[str, Any] = {
        "name": name, "spec_config": spec_config, "prompt_source": prompt_source,
        "num_prompts": num_prompts, "output_len": output_len,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1500,
    ) as srv:
        rec["startup_s"] = time.time() - t0
        print(f"[{name}] up in {rec['startup_s']:.0f}s; decoding {prompt_source} "
              f"{num_prompts}x{output_len}", flush=True)
        if prompt_source == "public":
            decode = decode_public_prompts(
                srv.base_url, srv.served_model_name, out_dir, name,
                num_prompts=num_prompts, output_len=output_len)
        else:
            decode = decode_jsonl_prompts(
                srv.base_url, srv.served_model_name, Path(prompt_source),
                num_prompts=num_prompts, output_len=output_len)
        rec["decode"] = {k: v for k, v in decode.items() if k != "per_prompt"}
        rec["per_prompt"] = decode["per_prompt"]
        rec["decode_tps_walltime"] = (
            decode["num_completion_tokens"] / decode["duration_s"]
            if decode.get("duration_s") else None)
        try:
            rec["tps_probe"] = harness.probe_tps(srv.base_url, srv.served_model_name)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            rec["tps_probe"] = {"error": str(exc)}
        try:
            with urllib.request.urlopen(f"{srv.base_url}/metrics", timeout=30) as r:
                metrics_text = r.read().decode("utf-8", "replace")
            rec["prom_spec"] = serve_profile.parse_spec_metrics(metrics_text)
        except (urllib.error.URLError, OSError) as exc:
            rec["prom_spec"] = {"error": str(exc)}
        if do_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    SERVER_PY, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_dir / f"ppl_{name}.jsonl",
                    summary_file=out_dir / f"ppl_{name}.summary.json", timeout_s=1800)
                rec["ppl"] = ppl_summary
            except Exception as exc:  # noqa: BLE001
                rec["ppl"] = {"error": str(exc)}
    log_text = log_path.read_text()
    rec["spec_log"] = serve_profile.parse_spec_log(log_text)
    # Unified acceptance: prefer Prometheus exact counters, fall back to server log.
    rec["acceptance"] = _resolve_acceptance(rec)
    rec["wall_s"] = time.time() - t0
    return rec


def _resolve_acceptance(rec: dict[str, Any]) -> dict[str, Any]:
    prom = rec.get("prom_spec") or {}
    slog = rec.get("spec_log") or {}
    out: dict[str, Any] = {}
    acc = prom.get("num_accepted_tokens")
    drf = prom.get("num_draft_tokens")
    if acc is not None and drf:
        out["acceptance_rate"] = acc / drf
        out["e_accept"] = prom.get("e_accept_mean_acceptance_length")
        out["source"] = "prometheus"
        out["accepted_tokens"] = acc
        out["draft_tokens"] = drf
    elif slog.get("draft_acceptance_rate") is not None:
        out["acceptance_rate"] = slog.get("draft_acceptance_rate")
        out["e_accept"] = slog.get("e_accept_exact") or slog.get("e_accept_interval_mean")
        out["source"] = "server_log"
        out["accepted_tokens"] = slog.get("total_accepted_tokens")
        out["draft_tokens"] = slog.get("total_drafted_tokens")
    else:
        out["acceptance_rate"] = None
        out["e_accept"] = slog.get("e_accept_interval_mean")
        out["source"] = "none"
    out["steady_gen_tps"] = slog.get("steady_gen_tps_mean")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs-json", required=True,
                    help="JSON list of {name, spec_config} (spec_config is a JSON string or '')")
    ap.add_argument("--prompt-source", default="public",
                    help="'public' (ShareGPT-128) or a .jsonl path of token-id prompts")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--out", required=True, help="results JSON path (checkpointed per config)")
    ap.add_argument("--ppl", action="store_true", help="also run same-endpoint PPL per config")
    ap.add_argument("--resume", action="store_true", help="skip configs already in --out")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[probe] {note}", flush=True)

    configs = json.loads(Path(args.configs_json).read_text())
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text())
    done = set(results.get("configs", {}).keys())

    results.setdefault("meta", {
        "prompt_source": args.prompt_source, "num_prompts": args.num_prompts,
        "output_len": args.output_len, "submission": str(SUBMISSION),
        "note": "local A10G probe; NOT official a10g-small TPS",
    })
    results.setdefault("configs", {})

    for cfg in configs:
        name = cfg["name"]
        if name in done:
            print(f"[probe] skip {name} (resume)", flush=True)
            continue
        print(f"\n===== config {name} spec={cfg['spec_config'] or '(AR/none)'} =====", flush=True)
        try:
            rec = run_config(
                name, cfg["spec_config"], args.prompt_source, out_dir,
                num_prompts=args.num_prompts, output_len=args.output_len, do_ppl=args.ppl)
        except Exception as exc:  # noqa: BLE001
            import traceback
            rec = {"name": name, "spec_config": cfg["spec_config"],
                   "error": str(exc), "traceback": traceback.format_exc()}
            print(f"[probe] config {name} FAILED: {exc}", flush=True)
        results["configs"][name] = rec
        out_path.write_text(json.dumps(results, indent=2))
        acc = (rec.get("acceptance") or {})
        print(f"[probe] {name}: acc_rate={acc.get('acceptance_rate')} "
              f"e_accept={acc.get('e_accept')} steady_tps={acc.get('steady_gen_tps')} "
              f"src={acc.get('source')} (checkpointed -> {out_path})", flush=True)

    print(f"\n[probe] all done -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
