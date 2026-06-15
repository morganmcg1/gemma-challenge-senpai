#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Measure rho = rank-2+ drafter coverage on the deployed MTP K=7 chain (PR #79).

WHAT THIS ANSWERS
-----------------
PR #76 projected a +18.7% tree-verify TPS gain whose single remaining unmeasured
input is rho = rank-2+ coverage: the probability the target's greedy argmax is the
drafter's rank-r token GIVEN the drafter's rank 1..r-1 all missed, conditioned on
the true greedy prefix (on-path). #76 borrowed EAGLE-3's rho=0.565 (researcher-agent
confirms that number has NO published provenance for an MTP drafter). This script
measures it directly on the deployed stack and re-prices the trees.

A linear chain only ever reveals the drafter's rank-1 token. To read rank-2/3/4 we
look at the drafter's top-W candidate logits per draft depth, on a SCRATCH copy of
``submissions/fa2sw_precache_kenyan`` (served files byte-identical). See
``rankprobe_patch.py`` for the contract-safety argument; in short we only ADD
logging and force eager base_propose so the per-depth selection runs in Python --
the emitted draft chain is byte identical to production.

ESTIMAND (first-divergence on-path; the correct tree-rescue estimand)
---------------------------------------------------------------------
At each decode step the drafter proposes K=7 tokens; the verifier accepts the
longest prefix matching its greedy argmax. At the FIRST divergence depth ``fd`` the
prefix 0..fd-1 is the true greedy continuation (teacher-forced on the target's own
output), so the drafter's ranking at ``fd`` is computed on the true prefix -- which
is exactly the context a width-W tree would branch from. We read the rank of the
true token (target argmax at ``fd``) in the drafter's top-W there. Pooling over many
steps:

    rho2 = P(rank==2 | rank1 missed)              = #(rank_fd==2) / n_div
    rho3 = P(rank==3 | rank1,2 missed)            = #(rank_fd==3) / (n_div - #2)
    rho4 = P(rank==4 | rank1,2,3 missed)          = #(rank_fd==4) / (n_div - #2 - #3)
    cov_W = P(true within drafter top-W | miss1)  = #(2<=rank_fd<=W) / n_div

where n_div counts every first-divergence event and rank_fd==0 (true token beyond
the drafter top-W) STAYS in each rho denominator -- a beyond-W event has missed
ranks 1..r-1 for every r, so it belongs in every conditional-miss denominator. The
rho loop subtracts only the rank==r events that were *caught*, never the beyond-W
mass, so the denominators are n_div, n_div-#2, n_div-#2-#3, ... as written above.

CROSS-CHECK: per-depth rank-1 acceptance q[d] reconstructed here MUST match #76's
independently-measured conditional acceptance (q[0] ~ 0.729). That validates the
whole drafter-path + alignment + verify-pairing end to end.

LOCAL ONLY. Single assigned GPU. No HF Job, no submission launch, no served-file
change.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
OUT_DIR = ROOT / "research" / "rank_coverage"
ACCEPT76 = ROOT / "research" / "accept_calibration" / "accept_calibration_results.json"
PATCH_SRC = Path(__file__).resolve().parent / "rankprobe_patch.py"

# Huge so the onegraph CUDA graph never captures -> base_propose (eager) every step
# -> Gemma4Proposer._greedy_sample (our override) fires at every draft depth.
WARMUP_FOREVER = "1000000000"


# --------------------------------------------------------------------------- #
# Scratch submission copy (served files stay byte-identical)
# --------------------------------------------------------------------------- #
def build_scratch(submission: Path, scratch: Path) -> Path:
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(
        submission, scratch,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(PATCH_SRC, scratch / "rankprobe_patch.py")
    sc = scratch / "sitecustomize.py"
    text = sc.read_text()
    marker = "# --- rank-coverage probe (PR #79, scratch only) ---"
    if marker not in text:
        text += (
            f"\n\n{marker}\n"
            # Fresh-a10g vLLM 0.22.1rc1 mounts pathless `_IncludedRouter`s, so
            # prometheus_fastapi_instrumentator.routing._get_route_name does
            # `route.path` -> AttributeError on EVERY request -> /v1/models 500s ->
            # the profiler server never reaches readiness (0 records, the dead-end
            # the #86 logits path hit). Swallow that one AttributeError (the
            # validated output-neutral guard ported from
            # submissions/fa2sw_treeverify_kenyan/sitecustomize.py, kanna PR #177
            # W&B bjtwr9jn: token-ids 128/128 identical, PPL byte-identical). HTTP
            # metrics route-name lookup only; never touches greedy / PPL / tokens.
            # Scratch-only: the committed served submission is byte-identical.
            "try:\n"
            "    import prometheus_fastapi_instrumentator.routing as _rp_pr  # noqa: E402\n"
            "    _rp_orig_grn = _rp_pr._get_route_name\n"
            "    def _rp_guarded_grn(scope, routes, _o=_rp_orig_grn):\n"
            "        try:\n"
            "            return _o(scope, routes)\n"
            "        except AttributeError:\n"
            "            return None\n"
            "    _rp_pr._get_route_name = _rp_guarded_grn\n"
            "except Exception:\n"
            "    pass\n"
            "import os as _rp_os  # noqa: E402\n"
            "if _rp_os.environ.get('RANKPROBE_ENABLE') == '1':\n"
            "    try:\n"
            "        import rankprobe_patch  # noqa: E402,F401\n"
            "    except Exception as _rp_exc:  # noqa: BLE001\n"
            "        import sys as _rp_sys\n"
            "        print(f'[rankprobe] import failed: {_rp_exc!r}', file=_rp_sys.stderr, flush=True)\n"
        )
        sc.write_text(text)
    return scratch


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_capture(scratch: Path, *, num_prompts: int, output_len: int, seed: int,
                records_path: Path, log_path: Path, logits: bool = True,
                dataset: Path | None = None) -> dict[str, Any]:
    manifest = harness.load_manifest(scratch)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    extra_env = {
        "RANKPROBE_ENABLE": "1",
        "RANKPROBE_OUTPUT": str(records_path),
        "RANKPROBE_W": "4",
        # PR #86: capture drafter top-4 probs + predictive entropy per step, and the
        # verifier's top-1 prob + entropy per target position. Read-only, scratch-only,
        # token-preserving (align_bad self-checks byte identity).
        "RANKPROBE_LOGITS": "1" if logits else "0",
        "LOOPGRAPH_WARMUP_CALLS": WARMUP_FOREVER,
        # native sampler (cuRAND JIT dodge) + re-enable stat loggers for an
        # independent E[T]/acceptance read in the same log. None change tokens.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
    }
    resolved_dataset = dataset or paths.EVAL_PROMPTS
    report: dict[str, Any] = {
        "submission": str(scratch),
        "num_prompts": num_prompts, "output_len": output_len, "seed": seed, "conc": 1,
        "dataset": str(resolved_dataset),
        "dataset_is_public_default": dataset is None,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.time()
    with harness.LocalServer(
        scratch, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        decode_out = records_path.parent / "decode_rank_coverage.jsonl"
        decode_summary = records_path.parent / "decode_rank_coverage.summary.json"
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_summary,
            num_prompts=num_prompts, output_len=output_len, seed=seed,
            dataset=dataset, timeout_s=4800,
        )
        report["decode_summary"] = summary
    report["decode_wall_s"] = time.time() - t0
    return report


# --------------------------------------------------------------------------- #
# Analyze JSONL -> rho table
# --------------------------------------------------------------------------- #
def _record_files(records_path: Path) -> list[Path]:
    """All per-process record shards for this run.

    The probe writes ``{records_path}.{pid}`` (one file per writing process). We
    also accept the exact ``records_path`` (older single-file form / --analyze-only
    pointing straight at a shard). ``.meta.json`` sidecars are excluded.
    """
    import glob
    shards = [Path(p) for p in glob.glob(f"{records_path}.[0-9]*")
              if not p.endswith(".meta.json")]
    if records_path.exists():
        shards.append(records_path)
    # de-dup, keep only non-empty
    seen: dict[str, Path] = {}
    for p in shards:
        try:
            if p.stat().st_size > 0:
                seen[str(p)] = p
        except OSError:
            pass
    return sorted(seen.values(), key=lambda p: str(p))


def analyze(records_path: Path, W: int = 4, max_depth: int = 7) -> dict[str, Any]:
    n_records = 0
    n_align_bad = 0
    shards = _record_files(records_path)
    shard_names = [p.name for p in shards]
    # per-depth tallies
    reached = [0] * max_depth       # reached depth d on true prefix (d <= fd, d < n)
    accept = [0] * max_depth        # rank-1 hit at d (d < fd)
    div_at = [0] * max_depth        # first-divergence count at depth d
    # rank_fd histogram at each divergence depth: ranks 0(miss),2,3,4,..W
    rank_hist_by_depth: dict[int, dict[int, int]] = {d: {} for d in range(max_depth)}
    rank_hist_pooled: dict[int, int] = {}

    for shard in shards:
      with open(shard) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_records += 1
            if not rec.get("align", True):
                n_align_bad += 1
                continue
            n = int(rec["n"])
            fd = int(rec["fd"])
            # per-depth reach/accept from fd
            for d in range(min(n, max_depth)):
                if fd >= d:           # reached d on true prefix
                    reached[d] += 1
                if fd > d:            # accepted (rank-1 hit) at d
                    accept[d] += 1
            if fd < n:                # an observed first divergence (rank-1 miss)
                rank_fd = int(rec.get("rank_fd", 0))
                if fd < max_depth:
                    div_at[fd] += 1
                    rank_hist_by_depth[fd][rank_fd] = rank_hist_by_depth[fd].get(rank_fd, 0) + 1
                rank_hist_pooled[rank_fd] = rank_hist_pooled.get(rank_fd, 0) + 1

    # per-depth conditional rank-1 acceptance q[d]
    q = [accept[d] / reached[d] if reached[d] else None for d in range(max_depth)]

    # pooled rho: marginal conditional rescue ratios + cumulative coverage
    n_div = sum(rank_hist_pooled.values())
    def _cnt(h: dict[int, int], r: int) -> int:
        return h.get(r, 0)
    n_r = {r: _cnt(rank_hist_pooled, r) for r in range(2, W + 1)}
    n_miss_beyond = _cnt(rank_hist_pooled, 0)  # true token beyond top-W

    # marginal rho_r = #(rank==r) / (events that missed ranks 1..r-1).
    # remaining starts at n_div and we subtract ONLY the caught rank==r events each
    # step, so the beyond-W mass (#0) is never removed -> it stays in every
    # denominator, which is exactly "missed ranks 1..r-1".
    rho = {}
    remaining = n_div
    for r in range(2, W + 1):
        rho[r] = (n_r[r] / remaining) if remaining else None
        remaining -= n_r[r]
    # cumulative coverage cov_W = #(2<=rank<=W) / n_div
    cov = {}
    cum = 0
    for r in range(2, W + 1):
        cum += n_r[r]
        cov[r] = (cum / n_div) if n_div else None

    # per-depth rho2 (where sample allows)
    rho2_by_depth = {}
    for d in range(max_depth):
        tot = div_at[d]
        rho2_by_depth[d] = (rank_hist_by_depth[d].get(2, 0) / tot) if tot else None

    return {
        "n_records": n_records,
        "n_align_bad": n_align_bad,
        "record_shards": shard_names,
        "n_divergences": n_div,
        "W": W,
        "per_depth_reached": reached,
        "per_depth_accept": accept,
        "per_depth_div_count": div_at,
        "conditional_rank1_acceptance_q": q,
        "top1_acceptance": q[0],
        "rank_fd_hist_pooled": {str(k): v for k, v in sorted(rank_hist_pooled.items())},
        "rho_marginal": {str(r): rho[r] for r in rho},
        "cumulative_coverage": {str(r): cov[r] for r in cov},
        "rho2_by_depth": {str(d): rho2_by_depth[d] for d in rho2_by_depth},
        "n_true_beyond_topW": n_miss_beyond,
        "frac_true_beyond_topW": (n_miss_beyond / n_div) if n_div else None,
    }


def cross_check(analysis: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"accept76_available": ACCEPT76.exists()}
    if not ACCEPT76.exists():
        return out
    d76 = json.loads(ACCEPT76.read_text())
    cond76 = (d76.get("headline") or {}).get("conditional_acceptance_p") or []
    q = analysis["conditional_rank1_acceptance_q"]
    out["conditional76"] = cond76
    out["conditional_measured_q"] = q
    if cond76 and q and q[0] is not None:
        out["top1_76"] = cond76[0]
        out["top1_measured"] = q[0]
        out["top1_abs_diff"] = abs(q[0] - cond76[0])
        # per-depth abs diffs where both present
        diffs = [abs(q[d] - cond76[d]) for d in range(min(len(q), len(cond76)))
                 if q[d] is not None and cond76[d] is not None]
        out["max_abs_diff_per_depth"] = max(diffs) if diffs else None
    return out


# --------------------------------------------------------------------------- #
# W&B
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[rank-cov] wandb unavailable ({exc})", flush=True)
        return None
    try:
        a = report["analysis"]
        xc = report.get("cross_check", {})
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="profiling",
            config={
                "submission": report["run"]["submission"],
                "num_prompts": report["run"]["num_prompts"],
                "output_len": report["run"]["output_len"],
                "conc": 1, "seed": report["run"]["seed"], "W": a["W"],
                "num_speculative_tokens": 7,
                "dataset": report["run"].get("dataset"),
                "dataset_is_public_default": report["run"].get("dataset_is_public_default"),
            },
        )
        rho = a["rho_marginal"]
        cov = a["cumulative_coverage"]
        flat: dict[str, Any] = {
            "primary/drafter_rank2_coverage": rho.get("2"),
            # rank-2+ CUMULATIVE coverage cov_W = P(true token at draft rank 2..W |
            # rank-1 diverged) -- the tree-recoverable share (the 0.653 public anchor).
            # Distinct from primary/drafter_rank2_coverage above, which is the
            # MARGINAL rho2 = P(rank==2 | miss1). cov_W + beyond_topW == 1 by partition.
            "primary/rank2plus_coverage": cov.get(str(a["W"])),
            "primary/beyond_topW": a["frac_true_beyond_topW"],
            "rho/rho2": rho.get("2"), "rho/rho3": rho.get("3"), "rho/rho4": rho.get("4"),
            "coverage/cov2": cov.get("2"), "coverage/cov3": cov.get("3"),
            "coverage/cov4": cov.get("4"),
            "rank1/top1_acceptance": a["top1_acceptance"],
            "n_divergences": a["n_divergences"],
            "n_records": a["n_records"],
            "n_align_bad": a["n_align_bad"],
            "frac_true_beyond_topW": a["frac_true_beyond_topW"],
            "xcheck/top1_76": xc.get("top1_76"),
            "xcheck/top1_measured": xc.get("top1_measured"),
            "xcheck/top1_abs_diff": xc.get("top1_abs_diff"),
            "xcheck/max_abs_diff_per_depth": xc.get("max_abs_diff_per_depth"),
        }
        run.summary.update(flat)
        # per-depth tables
        q = a["conditional_rank1_acceptance_q"]
        cond76 = xc.get("conditional76") or []
        tbl = wandb.Table(columns=["depth", "q_measured", "q_76", "div_count", "rho2_depth"])
        rho2d = a["rho2_by_depth"]
        for d in range(a["W"] if False else len(q)):
            tbl.add_data(
                d + 1, q[d], cond76[d] if d < len(cond76) else None,
                a["per_depth_div_count"][d] if d < len(a["per_depth_div_count"]) else None,
                rho2d.get(str(d)),
            )
        run.log({"per_depth": tbl})
        rid = run.id
        print(f"[rank-cov] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[rank-cov] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=48)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="wirbel/rank-coverage")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="rank-coverage")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="ShareGPT-format prompt set to profile over (default: the "
                         "public speed-benchmark eval_prompts_sharegpt.json). Pass "
                         "data/private_proxy_sharegpt.json to measure private-proxy "
                         "rank coverage. ONLY the input prompt set changes; the "
                         "drafter, target, rank-counting, and greedy-exact verify "
                         "are identical to the public path.")
    ap.add_argument("--no-logits", action="store_true",
                    help="disable PR #86 drafter/verifier prob+entropy capture "
                         "(reverts to the #79 rank-only probe)")
    ap.add_argument("--debug", action="store_true",
                    help="tiny 2-prompt/64-token smoke run to validate the harness")
    ap.add_argument("--analyze-only", type=Path, default=None,
                    help="skip serving; analyze an existing records JSONL")
    args = ap.parse_args(argv)

    # RANKPROBE_OUTPUT is consumed by the serve subprocess, which runs with
    # cwd=<scratch submission dir>. A RELATIVE --out-dir therefore resolves the
    # records path against the scratch cwd (doubled path) while analyze() reads it
    # against the repo root -> the probe writes records the analyzer never finds
    # (silent 0-records, indistinguishable from the dead #86 logits path). Resolve
    # to absolute so writer and reader agree regardless of the caller's cwd.
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / ("rankprobe_records_debug.jsonl" if args.debug
                                   else "rankprobe_records.jsonl")

    if args.analyze_only is not None:
        records_path = args.analyze_only

    report: dict[str, Any] = {}
    if args.analyze_only is None:
        for note in paths.prepare_local_gpu_env():
            print(f"[rank-cov] {note}", flush=True)
        num_prompts = 2 if args.debug else args.num_prompts
        output_len = 64 if args.debug else args.output_len
        scratch = args.out_dir / "_scratch_submission"
        build_scratch(args.submission.resolve(), scratch)
        print(f"[rank-cov] scratch submission at {scratch}", flush=True)
        log_path = args.out_dir / ("server_rank_coverage_debug.log" if args.debug
                                   else "server_rank_coverage.log")
        report["run"] = run_capture(
            scratch, num_prompts=num_prompts, output_len=output_len, seed=args.seed,
            records_path=records_path, log_path=log_path, logits=not args.no_logits,
            dataset=args.dataset,
        )
    else:
        report["run"] = {"submission": "(analyze-only)", "num_prompts": None,
                         "output_len": None, "seed": args.seed}

    analysis = analyze(records_path)
    report["analysis"] = analysis
    report["cross_check"] = cross_check(analysis)

    out_json = args.out_dir / ("rank_coverage_results_debug.json" if args.debug
                               else "rank_coverage_results.json")
    out_json.write_text(json.dumps(report, indent=2))

    wid = None
    if not args.no_wandb and not args.debug:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    out_json.write_text(json.dumps(report, indent=2))

    a = analysis
    xc = report["cross_check"]
    print("\n========== RANK-COVERAGE (rho) ==========", flush=True)
    print(f"records             : {a['n_records']}  (align_bad={a['n_align_bad']})", flush=True)
    print(f"divergence events   : {a['n_divergences']}", flush=True)
    print(f"top-1 acceptance q0 : {a['top1_acceptance']}", flush=True)
    if xc.get("top1_76") is not None:
        print(f"  vs #76 top-1      : {xc['top1_76']:.4f}  (|diff|={xc['top1_abs_diff']:.4f}, "
              f"max per-depth |diff|={xc.get('max_abs_diff_per_depth')})", flush=True)
    print(f"rho marginal        : {a['rho_marginal']}", flush=True)
    print(f"cumulative coverage : {a['cumulative_coverage']}", flush=True)
    print(f"rho2 by depth       : {a['rho2_by_depth']}", flush=True)
    print(f"true beyond top-W   : {a['frac_true_beyond_topW']}", flush=True)
    print(f"wandb run           : {wid}", flush=True)
    print(f"artifacts           : {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
