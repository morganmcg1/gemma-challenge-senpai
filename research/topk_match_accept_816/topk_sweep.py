#!/usr/bin/env python
"""PR #816 — Top-k-match spec-decode accept-branch sweep (REAL mechanism).

#813's synthetic oracle proved the speed ceiling: imposing accept rate r=1.00
lifts int4head from 323.29 wall TPS (r=0.56, real E_accept 4.355) to 506.84 wall
(+57%). That oracle emitted GARBAGE tokens. This driver runs the REAL mechanism:
the ``vllm_topk_accept_patch`` relaxes vLLM's greedy accept test to
``draft in topk(target_logits, K)`` behind ``TOPK_ACCEPT_K``. Larger K accepts
more drafts -> higher realized E_accept -> higher TPS, at a quality cost (accepted
tokens are no longer the greedy argmax). Byte-identity is WAIVED under #784.

Per k we measure, against the SERVED int4head endpoint (LOCAL A10G, conc=1,
128 prompts x 512 output tokens, temp=0, ignore_eos — the official decode
condition): steady_gen_tps + decode_wall_tps, realized E_accept(k), 128/128
completion, PPL, and the served completion token-ids (for greedy-divergence vs the
k=1 control). The downstream quality panel (AIME/MMLU-Pro/GPQA/GSM8K) is run by
``quality_panel.py`` against the same per-k serve.

k=1 is the no-op control: the patch does NOT rebind rejection_sample, so it must
reproduce ~323 wall AND be byte-identical to the shipped submission.

LOCAL A10G only. No HF job. Run (background):
  CUDA_VISIBLE_DEVICES=0 uv run python research/topk_match_accept_816/topk_sweep.py \
    --ks 1,2,4,8 --wandb-group bi0-int4head-topk-accept
Smoke (cheap pipeline + mechanism check, no PPL):
  ... --ks 1,2 --num-prompts 2 --output-len 32 --no-ppl --tag smoke \
      --wandb-group bi0-int4head-topk-accept-smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.local_validation.serve_profile import (  # noqa: E402
    parse_spec_log,
    parse_spec_metrics,
    _get_text,
)

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_int4head"
SERVER_PY = ROOT / ".venv" / "bin" / "python"
NUM_SPEC_DEFAULT = 6
# Official-projection factor: the official a10g-small TPS is a wall-aggregate; the
# 218.02 anchor projects from the local WALL number x this factor (#816 Baseline).
OFFICIAL_WALL_FACTOR = 0.9940


def run_one_k(
    k: int, *, num_prompts: int, output_len: int, num_spec: int, out_dir: Path,
    port: int, run_ppl: bool,
) -> dict[str, Any]:
    """Serve int4head with TOPK_ACCEPT_K=k; measure decode TPS, E_accept, (PPL)."""
    label = f"k{k}"
    log_path = out_dir / f"server_{label}.log"
    extra_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Re-enable Prometheus stats so spec_decode_* counters (canonical E_accept)
        # are exposed; the manifest ships --disable-log-stats.
        "DISABLE_LOG_STATS": "0",
        # The real top-k-match accept knob (consumed by vllm_topk_accept_patch via
        # sitecustomize). K<=1 => no rebind => byte-identical control.
        "TOPK_ACCEPT_K": str(k),
        "NUM_SPECULATIVE_TOKENS": str(num_spec),
    }
    rec: dict[str, Any] = {
        "k": k, "num_prompts": num_prompts, "output_len": output_len,
        "num_spec": num_spec,
    }
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=port, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        rec["boot_s"] = time.time() - t0
        rec["model_id"] = srv.model_id
        decode_out = out_dir / f"decode_{label}.jsonl"
        decode_sum = out_dir / f"decode_{label}.summary.json"
        td = time.time()
        summary = harness.capture_decode(
            SERVER_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_sum,
            num_prompts=num_prompts, output_len=output_len, timeout_s=3600,
        )
        rec["decode_wall_s"] = time.time() - td
        rec["decode_summary"] = summary
        rec["decode_jsonl"] = str(decode_out)
        try:
            rec["spec_metrics"] = parse_spec_metrics(_get_text(f"{srv.base_url}/metrics"))
        except Exception as exc:  # noqa: BLE001
            rec["spec_metrics"] = {"error": str(exc)}
        if run_ppl:
            ppl_out = out_dir / f"ppl_{label}.jsonl"
            ppl_sum = out_dir / f"ppl_{label}.summary.json"
            tp = time.time()
            try:
                rec["ppl_summary"] = harness.run_ppl(
                    SERVER_PY, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=ppl_out, summary_file=ppl_sum, timeout_s=1800,
                )
            except Exception as exc:  # noqa: BLE001
                rec["ppl_summary"] = {"error": str(exc)}
            rec["ppl_wall_s"] = time.time() - tp
    spec_log = parse_spec_log(log_path.read_text())
    rec["spec_log"] = spec_log

    # Headline metrics.
    rec["steady_gen_tps"] = spec_log.get("steady_gen_tps_mean")
    rec["steady_gen_tps_n"] = spec_log.get("steady_gen_tps_n")
    rec["e_accept_exact_from_log"] = spec_log.get("e_accept_exact")
    rec["draft_acceptance_rate_measured"] = spec_log.get("draft_acceptance_rate")
    pm = rec.get("spec_metrics") or {}
    rec["e_accept_mean_acceptance_length_prom"] = pm.get("e_accept_mean_acceptance_length")
    dur = (summary or {}).get("duration_s")
    ntok = (summary or {}).get("num_completion_tokens")
    rec["decode_wall_tps"] = (ntok / dur) if (dur and ntok) else None
    if rec["decode_wall_tps"]:
        rec["projected_official_tps"] = rec["decode_wall_tps"] * OFFICIAL_WALL_FACTOR
    rec["num_records"] = (summary or {}).get("num_records")
    rec["completed_128"] = (rec["num_records"] == num_prompts)
    ppls = rec.get("ppl_summary") or {}
    rec["ppl"] = ppls.get("ppl") or ppls.get("perplexity")
    return rec


def compute_divergence(out_dir: Path, ks: list[int]) -> dict[str, Any]:
    """Greedy-divergence of each k vs the k=1 control, from saved decode jsonl.

    Aligns records by prompt_sha256 and compares completion_token_ids position by
    position over the common length. Reports, per k: fraction of prompts whose
    completion differs at all, mean first-divergence position, and the
    token-level divergence rate (differing positions / compared positions)."""
    def load(k: int) -> dict[str, list[int]] | None:
        p = out_dir / f"decode_k{k}.jsonl"
        if not p.exists():
            return None
        by_prompt = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_prompt[r["prompt_sha256"]] = r["completion_token_ids"]
        return by_prompt

    base = load(1)
    out: dict[str, Any] = {}
    if base is None:
        return {"error": "no k=1 control decode capture for divergence baseline"}
    for k in ks:
        if k == 1:
            out["k1"] = {"prompts_diverged": 0, "token_divergence_rate": 0.0,
                         "mean_first_divergence_pos": None, "self_check": True}
            continue
        cur = load(k)
        if cur is None:
            continue
        n_prompts = 0
        n_diverged = 0
        tot_cmp = 0
        tot_diff = 0
        first_positions = []
        for sha, toks in cur.items():
            if sha not in base:
                continue
            n_prompts += 1
            btoks = base[sha]
            m = min(len(toks), len(btoks))
            diffs = [i for i in range(m) if toks[i] != btoks[i]]
            tot_cmp += m
            tot_diff += len(diffs)
            if diffs or len(toks) != len(btoks):
                n_diverged += 1
            if diffs:
                first_positions.append(diffs[0])
        out[f"k{k}"] = {
            "prompts_compared": n_prompts,
            "prompts_diverged": n_diverged,
            "prompt_divergence_frac": (n_diverged / n_prompts) if n_prompts else None,
            "token_divergence_rate": (tot_diff / tot_cmp) if tot_cmp else None,
            "mean_first_divergence_pos": (sum(first_positions) / len(first_positions))
            if first_positions else None,
        }
    return out


def log_wandb(rec: dict[str, Any], *, group: str, name: str, project_tag: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed ({exc}); JSON-only", flush=True)
        return None
    try:
        run = wandb.init(
            entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
            group=group, name=name, reinit=True,
            config={
                "pr": 816,
                "experiment": "topk-match-accept",
                "local_a10g": True,
                "official_tps": 0,
                "byte_identity_waived_784": rec["k"] > 1,
                "topk_accept_k": rec["k"],
                "num_speculative_tokens": rec["num_spec"],
                "model_id": rec.get("model_id"),
                "submission": "int4_mtp_bi0_int4head",
                "num_prompts": rec["num_prompts"],
                "output_len": rec["output_len"],
                "tag": project_tag,
            },
        )
        summ = {
            "topk_accept_k": rec["k"],
            "steady_gen_tps": rec.get("steady_gen_tps"),
            "steady_gen_tps_n": rec.get("steady_gen_tps_n"),
            "decode_wall_tps": rec.get("decode_wall_tps"),
            "projected_official_tps": rec.get("projected_official_tps"),
            "e_accept_exact_from_log": rec.get("e_accept_exact_from_log"),
            "e_accept_mean_acceptance_length_prom": rec.get("e_accept_mean_acceptance_length_prom"),
            "draft_acceptance_rate_measured": rec.get("draft_acceptance_rate_measured"),
            "ppl": rec.get("ppl"),
            "num_records": rec.get("num_records"),
            "completed_128": rec.get("completed_128"),
            "boot_s": rec.get("boot_s"),
            "decode_wall_s": rec.get("decode_wall_s"),
        }
        dv = rec.get("divergence_vs_k1")
        if dv:
            summ["prompt_divergence_frac"] = dv.get("prompt_divergence_frac")
            summ["token_divergence_rate"] = dv.get("token_divergence_rate")
            summ["mean_first_divergence_pos"] = dv.get("mean_first_divergence_pos")
        run.summary.update(summ)
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid} ({name})", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--num-spec", type=int, default=NUM_SPEC_DEFAULT)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--wandb-group", default="bi0-int4head-topk-accept")
    ap.add_argument("--wandb-prefix", default="stark/topk-accept")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[gpu-env] {n}", flush=True)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    suffix = f"-{args.tag}" if args.tag else ""
    out_dir = HERE / "runs" / (args.tag or "sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[topk] ks={ks} num_spec={args.num_spec} np={args.num_prompts} "
          f"ol={args.output_len} ppl={not args.no_ppl} out={out_dir}", flush=True)

    results: list[dict[str, Any]] = []
    for k in ks:
        print(f"\n===== TOPK_ACCEPT_K={k} =====", flush=True)
        try:
            rec = run_one_k(
                k, num_prompts=args.num_prompts, output_len=args.output_len,
                num_spec=args.num_spec, out_dir=out_dir, port=args.port,
                run_ppl=not args.no_ppl,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[topk] k={k} FAILED: {exc!r}", flush=True)
            rec = {"k": k, "error": repr(exc)}
        (out_dir / f"k_{k}.json").write_text(json.dumps(rec, indent=2, default=str))
        results.append(rec)
        st = rec.get("steady_gen_tps")
        eacc = rec.get("e_accept_exact_from_log")
        print(f"[topk] k={k}  steady_gen_tps={st}  e_accept={eacc}  "
              f"wall_tps={rec.get('decode_wall_tps')}  ppl={rec.get('ppl')}  "
              f"128/128={rec.get('completed_128')}", flush=True)

    # Greedy-divergence vs the k=1 control (from the saved decode captures).
    div = compute_divergence(out_dir, ks)
    (out_dir / "divergence.json").write_text(json.dumps(div, indent=2, default=str))
    for rec in results:
        if "error" in rec:
            continue
        rec["divergence_vs_k1"] = div.get(f"k{rec['k']}")

    if not args.no_wandb:
        for rec in results:
            if "error" in rec:
                continue
            rec["wandb_run_id"] = log_wandb(
                rec, group=args.wandb_group,
                name=f"{args.wandb_prefix}-k{rec['k']}{suffix}",
                project_tag=args.tag or "sweep",
            )

    (out_dir / "sweep_summary.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n========== TOPK-ACCEPT SWEEP ==========", flush=True)
    print(f"{'k':>3} {'steady_tps':>11} {'wall_tps':>9} {'E_accept':>9} {'ppl':>7} "
          f"{'tok_div%':>9} {'128/128':>8}", flush=True)
    for rec in results:
        if "error" in rec:
            print(f"{rec['k']:>3} ERROR {rec['error']}", flush=True)
            continue
        dv = rec.get("divergence_vs_k1") or {}
        tdr = dv.get("token_divergence_rate")
        print(f"{rec['k']:>3} "
              f"{(rec.get('steady_gen_tps') or float('nan')):>11.2f} "
              f"{(rec.get('decode_wall_tps') or float('nan')):>9.2f} "
              f"{(rec.get('e_accept_exact_from_log') or float('nan')):>9.3f} "
              f"{(rec.get('ppl') or float('nan')):>7.4f} "
              f"{(100.0 * tdr if tdr is not None else float('nan')):>9.3f} "
              f"{str(rec.get('completed_128')):>8}", flush=True)


if __name__ == "__main__":
    main()
