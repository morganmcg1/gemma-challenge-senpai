"""PR #500 deliverable (3) — Census 2: run the #461 logit-margin flip-attribution
census fresh on the BYTEEXACT fixed-order split-KV candidate, redirected to the
byteexact399 cert dir.

WHY A NEW WRAPPER (vs surgical357's run_flip_census.py, which just rebinds
``dfa.OUT_DIR`` and calls ``dfa.main()``): the surgical cert certifies the
*existing* ``attn_only`` arm (which pins the attention module global
``is_batch_invariant=True`` -> 2D single-segment in-order KV reduction). The
byteexact candidate is a DIFFERENT lever: it keeps ``is_batch_invariant=False``
(so the fast 3D split-KV / FlashDecoding parallelism stays on) and instead pins
``tiles_per_segment`` to a fixed literal (4) so every parallel softmax segment
covers a FIXED ABSOLUTE key span [s*64,(s+1)*64) regardless of seq_len. That makes
the 3D split-KV reduction order M-invariant -> the M=8 spec-verify is byte-identical
to the M=1 AR decode of the same token, WITHOUT giving up the split-KV parallelism
the surgical 2D path sacrifices. ``deployed_flip_attribution.py`` has no such arm,
so we monkeypatch ``dfa.apply_arm_pin`` to add a ``byteexact`` arm that installs the
PACKAGED ``byteexact_splitkv_patch.install()`` (the exact same re-jit the served
sitecustomize arms), then call ``dfa.phase_arm`` directly. dfa.py is NOT modified.

WHAT IT CERTIFIES: the same #461 locus methodology surgical357 was held to
(128 prompts x ctx 224, M=8 verify width via ``prompt_logprobs``, TEACHER-FORCED so
there is NO greedy AR cascade). For each arm it records, per (prompt,pos), the M=8
chunk argmax vs the M=1 AR token, and the divergent-token margin. A flip is a
bf16-ULP knife-edge near-tie iff its margin < NEAR_TIE_LOGPROB_THRESH (0.5 nat);
margin >= 0.5 (or m1 not in top-5) == a SEMANTIC flip. Target == surgical357's
standard: <= a handful of ULP-tie flips, 0 semantic, and byteexact divergence <=
deployed divergence (byteexact closes the deployed adaptive-3D attention flips).

ARMS (same GPU session, fresh interpreter each, for churn-controlled comparison):
  deployed   : stock 3D split-KV (adaptive tiles_per_segment) -- the M-DEP baseline
               that MUST show attention flips (proves the locus exercises the 3D path)
  byteexact  : install byteexact_splitkv_patch (fixed tiles_per_segment=4, 64 seg)
               -- the CANDIDATE; expect divergence ~ surgical attn_only, 0 semantic
  attn_only  : is_batch_invariant=True (2D in-order) -- the surgical357 reference rung

LOCAL ONLY. analysis_only=true, official_tps=0. No HF job, no submission.

  .venv/bin/python -m research.validity.byteexact399_operative_cert.run_byteexact_census \
      --n-prompts 128 --arms deployed,byteexact,attn_only \
      --wandb_name lawine/byteexact399-flip-census --wandb_group byteexact-splitkv399-package
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DFA_DIR = ROOT / "research" / "validity" / "deployed_flip_attribution"
SUBMISSION_DIR = ROOT / "submissions" / "fa2sw_strict_byteexact_splitkv399"
CERT_OUT = ROOT / "research" / "validity" / "byteexact399_operative_cert" / "flip_census"

if str(DFA_DIR) not in sys.path:
    sys.path.insert(0, str(DFA_DIR))
import deployed_flip_attribution as dfa  # noqa: E402

NEAR_TIE = dfa.NEAR_TIE_LOGPROB_THRESH  # 0.5 nat
FIXED_TPS_DEFAULT = "4"
NUM_SEGMENTS_DEFAULT = "64"

# ---------------------------------------------------------------------------
# The byteexact arm: monkeypatched into dfa's apply_arm_pin.
# ---------------------------------------------------------------------------
_orig_apply_arm_pin = dfa.apply_arm_pin


def patched_apply_arm_pin(arm: str) -> dict:
    if arm != "byteexact":
        return _orig_apply_arm_pin(arm)
    # The patch module reads BYTEEXACT_FIXED_TPS / BYTEEXACT_NUM_SEGMENTS at IMPORT.
    # The subprocess env already sets them; default here too for safety.
    os.environ.setdefault("BYTEEXACT_FIXED_TPS", FIXED_TPS_DEFAULT)
    os.environ.setdefault("BYTEEXACT_NUM_SEGMENTS", NUM_SEGMENTS_DEFAULT)
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    import byteexact_splitkv_patch as bx  # noqa: E402
    installed = bool(bx.install())
    import vllm.v1.attention.ops.triton_unified_attention as _ua
    import vllm.v1.attention.backends.triton_attn as _ta
    flags = {
        "arm": arm,
        "attn_pin_requested": True,        # it pins the attention reduction order (fixed-order 3D)
        "lmhead_pin_requested": False,     # byteexact does NOT install the matmul/lm_head tax
        "rms_env_set": False,
        "byteexact_installed": installed,
        "byteexact_fixed_tps_marker": getattr(_ua, "_byteexact_fixed_tps", None),
        "backend_num_par_softmax_segments": getattr(_ta, "NUM_PAR_SOFTMAX_SEGMENTS", None),
    }
    print(f"[pin:{arm}] byteexact_splitkv_patch.install()={installed} "
          f"fixed_tps_marker={flags['byteexact_fixed_tps_marker']} "
          f"backend_segments={flags['backend_num_par_softmax_segments']} "
          f"(fixed-order 3D split-KV -> M-invariant; matmul tax NOT installed, "
          f"is_batch_invariant stays False)", flush=True)
    return flags


# ---------------------------------------------------------------------------
# Phase entry (subprocess): patch dfa, run one arm in-process.
# ---------------------------------------------------------------------------
def run_phase(a: argparse.Namespace) -> None:
    dfa.apply_arm_pin = patched_apply_arm_pin  # global lookup in phase_arm resolves here
    # byteexact + attn_only run the diag micro-benches only when explicitly asked; the
    # heavy marlin/gemm/rms diagnostics are only meaningful on the deployed arm.
    do_micro = a.microbench
    dfa.phase_arm(a.out, a.arm, a.n_prompts, a.ctx_len, a.n_verify,
                  a.gpu_mem_util, a.max_batched_tokens, a.verbose_k, do_micro)


# ---------------------------------------------------------------------------
# Orchestrator: spawn one fresh-interpreter subprocess per arm, then classify.
# ---------------------------------------------------------------------------
def _spawn_arm(a: argparse.Namespace, arm: str) -> dict:
    out_json = str(CERT_OUT / f"arm_{arm}_result.json")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # byteexact must NOT install the matmul tax (the whole point is fixed-order attn only).
    env["VLLM_BATCH_INVARIANT"] = "0"
    if arm == "byteexact":
        env["BYTEEXACT_FIXED_TPS"] = str(a.fixed_tps)
        env["BYTEEXACT_NUM_SEGMENTS"] = str(a.num_segments)
    else:
        # ensure the byteexact lever is OFF for the deployed/attn_only control arms
        env.pop("BYTEEXACT_FIXED_TPS", None)
        env.pop("BYTEEXACT_NUM_SEGMENTS", None)
    micro = ["--microbench"] if arm == "deployed" else ["--no-microbench"]
    cmd = [sys.executable, os.path.abspath(__file__),
           "--phase", "arm", "--arm", arm, "--out", out_json,
           "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len),
           "--n-verify", str(a.n_verify), "--gpu-mem-util", str(a.gpu_mem_util),
           "--max-batched-tokens", str(a.max_batched_tokens),
           "--verbose-k", str(a.verbose_k)] + micro
    print(f"[orch] launching arm={arm} (VLLM_BATCH_INVARIANT=0 "
          f"BYTEEXACT_FIXED_TPS={env.get('BYTEEXACT_FIXED_TPS', 'unset')})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"arm subprocess failed (rc={rc}): {arm}")
    return json.load(open(out_json))


def _classify_flips(data: dict) -> dict:
    """Per-arm: list every M8!=M1 flip and split into bf16-ULP near-ties (margin <
    NEAR_TIE) vs SEMANTIC (margin >= NEAR_TIE, or m1 not in top-5 => margin None)."""
    flips = []
    for pp in data["per_prompt"]:
        pid = pp.get("id", pp.get("ri"))
        for pr in pp["pos_records"]:
            if not pr["match"]:
                flips.append({"id": pid, "pos": pr["pos"], "rel_pos": pr["rel_pos"],
                              "m8_tok": pr["m8_tok"], "m1_tok": pr["m1_tok"],
                              "margin_vs_m1": pr["margin_vs_m1"], "gap_top2": pr["gap_top2"]})
    ulp = [f for f in flips if f["margin_vs_m1"] is not None and f["margin_vs_m1"] < NEAR_TIE]
    semantic = [f for f in flips if f["margin_vs_m1"] is None or f["margin_vs_m1"] >= NEAR_TIE]
    margins = [f["margin_vs_m1"] for f in flips if f["margin_vs_m1"] is not None]
    return {
        "identity": data["identity"],
        "divergence": data.get("divergence"),
        "n_flips": len(flips),
        "n_ulp_near_tie": len(ulp),
        "n_semantic": len(semantic),
        "max_margin_nat": (max(margins) if margins else None),
        "min_gap_top2_nat": (min((f["gap_top2"] for f in flips if f["gap_top2"] is not None), default=None)),
        "residual_is_knife_edge_near_tie": data["near_tie"]["residual_is_knife_edge_near_tie"],
        "n_prompts": data["n_prompts"], "C": data["C"], "n_verify": data["n_verify"],
        "determinism_M1_vs_M1": data["determinism_M1_vs_M1"],
        "determinism_M8_vs_M8": data["determinism_M8_vs_M8"],
        "within_batch_copy0_vs_copy1": data["within_batch_copy0_vs_copy1"],
        "chunk_isolated_fraction": data["chunk_isolated_fraction"],
        "median_chunk_width": data["median_chunk_width"],
        "engaged": data.get("engaged"),
        "pin_flags": data.get("pin_flags"),
        "flips": flips,
    }


def orchestrate(a: argparse.Namespace) -> None:
    CERT_OUT.mkdir(parents=True, exist_ok=True)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    print(f"[byteexact-census] arms={arms} n_prompts={a.n_prompts} C(ctx)={a.ctx_len} "
          f"n_verify={a.n_verify} NEAR_TIE={NEAR_TIE} nat -> {CERT_OUT}", flush=True)
    per_arm = {}
    for arm in arms:
        data = _spawn_arm(a, arm)
        per_arm[arm] = _classify_flips(data)
        c = per_arm[arm]
        print(f"[byteexact-census] {arm:10s} identity={c['identity']:.6f} "
              f"flips={c['n_flips']} (ulp={c['n_ulp_near_tie']} semantic={c['n_semantic']}) "
              f"max_margin={c['max_margin_nat']} nat", flush=True)

    be = per_arm.get("byteexact")
    dep = per_arm.get("deployed")
    surg = per_arm.get("attn_only")
    verdict = {"available": be is not None}
    if be is not None:
        be_closes_deployed = (dep is not None and be["divergence"] is not None
                              and dep["divergence"] is not None
                              and be["divergence"] <= dep["divergence"])
        verdict.update({
            "byteexact_identity": be["identity"],
            "byteexact_divergence": be["divergence"],
            "byteexact_n_flips": be["n_flips"],
            "byteexact_n_ulp_near_tie": be["n_ulp_near_tie"],
            "byteexact_n_semantic": be["n_semantic"],
            "byteexact_max_margin_nat": be["max_margin_nat"],
            "byteexact_all_residual_flips_bf16_ulp_near_ties": (be["n_semantic"] == 0 and be["n_flips"] >= 0),
            "byteexact_semantic_flips": be["n_semantic"],
            "deployed_divergence": (dep["divergence"] if dep else None),
            "deployed_n_flips": (dep["n_flips"] if dep else None),
            "byteexact_closes_deployed_attention_flips": be_closes_deployed,
            "surgical_attn_only_identity": (surg["identity"] if surg else None),
            "surgical_attn_only_n_flips": (surg["n_flips"] if surg else None),
            "near_tie_threshold_nat": NEAR_TIE,
            # operative-1.0 standard (== surgical357): 0 semantic flips, every residual a bf16-ULP near-tie,
            # and the lever provably closes the deployed adaptive-3D attention flips.
            "operative_1_0_pass": bool(be["n_semantic"] == 0 and be_closes_deployed),
        })
    report = {
        "pr": 500, "generated_by": "lawine", "analysis_only": True, "official_tps": 0,
        "no_hf_job": True, "staged_path": "submissions/fa2sw_strict_byteexact_splitkv399/",
        "methodology": "#461 logit-margin locus (teacher-forced M=8 prompt_logprobs vs M=1 AR), no greedy cascade",
        "candidate": {"fixed_tps": a.fixed_tps, "num_segments": a.num_segments,
                      "chunk_keys": a.fixed_tps * 16, "coverage_keys": a.fixed_tps * a.num_segments * 16},
        "arms": per_arm, "verdict": verdict,
    }
    out_report = CERT_OUT / "byteexact_flip_census_report.json"
    json.dump(report, open(out_report, "w"), indent=2)
    print(f"[byteexact-census] wrote {out_report}", flush=True)
    print("=" * 88)
    if be is not None:
        print(f"[byteexact-census] VERDICT operative_1_0_pass={verdict['operative_1_0_pass']} "
              f"| byteexact identity={be['identity']:.6f} flips={be['n_flips']} "
              f"semantic={be['n_semantic']} ulp_near_tie={be['n_ulp_near_tie']} "
              f"max_margin={be['max_margin_nat']} nat (thresh {NEAR_TIE}) "
              f"| closes_deployed={verdict.get('byteexact_closes_deployed_attention_flips')}", flush=True)
    print("=" * 88)

    if not a.no_wandb:
        _log_wandb(a, report)


def _log_wandb(a: argparse.Namespace, report: dict) -> None:
    try:
        sys.path.insert(0, os.getcwd())
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; JSON only", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="lawine", name=a.wandb_name, group=a.wandb_group,
        notes="PR#500 byteexact399 operative-1.0 census 2: #461 logit-margin locus on the fixed-order "
              "split-KV candidate. 0 semantic flips + closes deployed adaptive-3D attn flips == surgical357 standard.",
        config={"pr": 500, "n_prompts": a.n_prompts, "C": a.ctx_len, "n_verify": a.n_verify,
                "fixed_tps": a.fixed_tps, "num_segments": a.num_segments,
                "near_tie_thresh_nat": NEAR_TIE, "analysis_only": True, "official_tps": 0},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return
    v = report["verdict"]
    for k, val in v.items():
        if isinstance(val, (int, float, bool)) or val is None:
            run.summary[f"census2/{k}"] = val
    for arm, c in report["arms"].items():
        run.summary[f"{arm}/identity"] = c["identity"]
        run.summary[f"{arm}/n_flips"] = c["n_flips"]
        run.summary[f"{arm}/n_semantic"] = c["n_semantic"]
        run.summary[f"{arm}/max_margin_nat"] = c["max_margin_nat"]
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["arm"], default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="deployed,byteexact,attn_only")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=dfa.M_VERIFY)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--fixed-tps", dest="fixed_tps", type=int, default=4)
    ap.add_argument("--num-segments", dest="num_segments", type=int, default=64)
    ap.add_argument("--microbench", dest="microbench", action="store_true")
    ap.add_argument("--no-microbench", dest="microbench", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb_name", dest="wandb_name", default="lawine/byteexact399-flip-census")
    ap.add_argument("--wandb_group", dest="wandb_group", default="byteexact-splitkv399-package")
    ap.add_argument("--no-wandb", action="store_true")
    ap.set_defaults(microbench=True)
    a = ap.parse_args()

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 6)

    if a.phase == "arm":
        run_phase(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
