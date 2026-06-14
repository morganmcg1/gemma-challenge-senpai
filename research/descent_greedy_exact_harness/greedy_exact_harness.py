#!/usr/bin/env python
"""Greedy-exactness differential harness — a per-token contract instrument.

The official scorer checks only TPS + PPL<=2.42 + 128/128 completion. It does
NOT verify, per token, that the committed token equals the target model's argmax
at that position. The current served stack is greedy-exact *by construction*: the
linear ``_dixie_fused_accept_prep_kernel`` stores ``target_argmax_id`` at every
committed position and breaks on the first draft/argmax mismatch. But land #71's
future descent/tree accept-walk replaces that kernel (same 6-arg signature) and
could pass all three scorer gates while committing a token != target argmax — the
BUG-2 class (over-acceptance). PPL is a *soft, averaged* signal; a handful of
wrong tokens can hide under the 2.42 cap. The greedy-identity gate the team
already runs (sha256 of the completion vs an M=1 autoregressive reference) is
*also* blind to this: documented int4-Marlin batch-variance (M=1 vs M=8 split-K
geometry) makes ~56% of completions legitimately DIVERGENT, so a real BUG-2 break
is indistinguishable from benign batch-variance at the completion level.

This harness fills that gap with a **kernel-isolation per-position differential**:

  * It imports and exercises the REAL production accept kernel directly
    (``sitecustomize._get_fused_accept_prep_kernel``) on a battery of synthetic +
    realistic (draft, target_argmax, bonus) cases. No model load, no serve, so it
    is fast and *batch-variance-invariant* — the target_argmax stream IS the
    ground truth, taken from the same step the kernel sees.
  * The greedy-exactness invariant: for every committed position ``p`` of request
    ``r``, ``committed[p] == in_step_argmax_ref[p]`` where the reference is
    ``target_argmax[p]`` at a draft position and ``bonus`` at the all-accept
    position (the bonus token IS the target's argmax at the bonus slot). The
    linear kernel yields rate == 1.0 by construction.
  * It ALSO cross-checks accept LENGTH against an independent pure-Python oracle
    of correct greedy spec-decode, so a descent kernel that commits the right
    argmax *values* but accepts PAST the first mismatch (token-identity holds yet
    over-accepts) is still flagged. Token-substitution AND length-overrun BUG-2
    are both caught.

Self-validation (PRIMARY deliverable until land #71's kernel exists):
  (a) the known-good linear kernel -> GREEDY_EXACT (rate 1.0, sha256 match),
  (b) an injected non-argmax acceptance (``_bug2_overaccept_kernel``) -> CAUGHT
      (rate < 1.0, violating position named),
  (c) a synthetic length-overrun result -> CAUGHT via the oracle length check
      while per-position rate is 1.0 (the subtle descent form),
  (d) PPL cross-check <= 2.42 still holds (on-disk known-good capture).
  ==> ``greedy_exact_harness_self_test_passes``.

Arming for land #71: ``--audit-kernel-symbol module:func`` runs the identical
differential against any same-signature kernel the instant it lands.

Aggregate disambiguation leg: the official ``greedy_identity.compare_files`` is
run on the on-disk spec-ON capture vs the M=1-AR reference and reported as
EXPECTED batch-variance (NOT a contract violation), to contrast the completion
-level DIVERGENT verdict with the per-step exactness rate of 1.0.

Scope: VALIDITY instrument, NOT a TPS lever. BASELINE stays 481.53. It does NOT
authorize and is unrelated to any HF job / submission.

Usage (LOCAL A10G + CPU only; run under the pinned-vLLM venv for triton/torch):

    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 /tmp/server-venv/bin/python \\
        research/descent_greedy_exact_harness/greedy_exact_harness.py --self-test \\
        --wandb-group descent-greedy-exact-harness --wandb-name denken/greedy-exact-harness

    # the instant land #71's kernel lands (same signature):
    ... greedy_exact_harness.py --audit-kernel-symbol my_descent_mod:descent_accept_kernel
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# research/descent_greedy_exact_harness/greedy_exact_harness.py -> repo root is two parents up.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import paths  # noqa: E402

SUBMISSION_DIR = ROOT / "submissions" / "fa2sw_precache_kenyan"
DEFAULT_PPL_CAP = 2.42
# Production speculative depth: SPECULATIVE_CONFIG num_speculative_tokens=7 ->
# max_spec_len 7 (7 draft positions + 1 bonus per request row, width 8).
DEFAULT_MAX_SPEC_LEN = 7
GPU_INDEX = 0  # single in-container GPU (see project_local_a10g_gpu_env memory)

# On-disk artifacts (reused; no fresh serve needed for the aggregate + PPL legs).
DEFAULT_AR_REF = (
    ROOT
    / "research"
    / "greedy_reference"
    / "workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it"
    / "decode_outputs.jsonl"
)
DEFAULT_SPEC_CAPTURE = (
    ROOT
    / "research"
    / "tree_submission_preflight"
    / "runs"
    / "fa2sw_precache_kenyan-20260614T125928Z"
    / "known_good"
    / "decode_outputs.jsonl"
)
DEFAULT_PPL_SUMMARY = DEFAULT_SPEC_CAPTURE.parent / "ppl_summary.json"


# --------------------------------------------------------------------------- #
# Kernels under test
# --------------------------------------------------------------------------- #
def load_reference_kernel(submission_dir: Path = SUBMISSION_DIR) -> Any:
    """Import the REAL production accept kernel (the linear, known-good drop-in).

    This is the exact triton kernel the live serve path invokes (serve.py ->
    ``sitecustomize._dixie_fused_accept_prep``), so the audit is faithful to the
    BUG-2 surface land #71 will replace. Importing ``sitecustomize`` is side-effect
    free (it only registers lazy meta-path finders; no model load).
    """
    p = str(submission_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    import sitecustomize  # noqa: E402

    return sitecustomize._get_fused_accept_prep_kernel()


def build_bug2_overaccept_kernel() -> Any:
    """Negative control: a BUG-2 (over-acceptance) accept kernel.

    Same 6-arg signature as the production kernel, but it commits the DRAFT token
    at every position and never rejects (appends the bonus unconditionally). On a
    case where ``draft[p] != target_argmax[p]`` this commits a non-argmax token,
    i.e. exactly the contract violation the harness must catch and localize.
    """
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["max_spec_len"])
    def _bug2_overaccept_kernel(
        output_token_ids_ptr,
        next_token_ids_ptr,
        valid_counts_ptr,
        cu_num_draft_tokens_ptr,
        draft_token_ids_ptr,
        target_argmax_ptr,
        bonus_token_ids_ptr,
        max_spec_len,
    ) -> None:
        req_idx = tl.program_id(0)
        start_idx = 0
        if req_idx != 0:
            start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
        num_draft_tokens = end_idx - start_idx

        row_offset = req_idx * (max_spec_len + 1)
        next_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
        # BUG-2: commit the DRAFT id at every position, never reject.
        for pos in range(num_draft_tokens):
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos).to(tl.int32)
            tl.store(output_token_ids_ptr + row_offset + pos, draft_token_id)
            next_token_id = draft_token_id
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
        tl.store(output_token_ids_ptr + row_offset + num_draft_tokens, bonus_token_id)
        next_token_id = bonus_token_id
        tl.store(next_token_ids_ptr + req_idx, next_token_id)
        tl.store(valid_counts_ptr + req_idx, num_draft_tokens + 1)

    return _bug2_overaccept_kernel


def load_kernel_symbol(symbol: str) -> Any:
    """Resolve a ``module:func`` armed-audit pointer for land #71's kernel."""
    mod_name, _, func_name = symbol.partition(":")
    if not mod_name or not func_name:
        raise ValueError(f"--audit-kernel-symbol must be 'module:func', got {symbol!r}")
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


# --------------------------------------------------------------------------- #
# Flat launcher — drives ANY same-signature accept kernel
# --------------------------------------------------------------------------- #
def launch_kernel(kernel: Any, cases: list[dict], max_spec_len: int) -> dict:
    """Launch ``kernel`` over a batch of cases and return its committed streams.

    Builds the flat (draft, target_argmax, bonus, cu_num_draft_tokens) int32
    tensors the kernel expects (grid = one program per request) and slices each
    output row to its reported ``valid_count``. The slice is clamped to the row
    width, but the RAW ``valid_count`` is preserved so a length-overrun kernel is
    still flagged by the oracle length check downstream.
    """
    import torch

    drafts: list[int] = []
    argmaxes: list[int] = []
    bonuses: list[int] = []
    cu: list[int] = []
    running = 0
    for c in cases:
        nd = len(c["draft"])
        if len(c["argmax"]) != nd:
            raise ValueError(f"case {c.get('label')!r}: draft/argmax length mismatch")
        drafts.extend(int(x) for x in c["draft"])
        argmaxes.extend(int(x) for x in c["argmax"])
        bonuses.append(int(c["bonus"]))
        running += nd
        cu.append(running)

    n = len(cases)
    dev = "cuda"
    draft_t = torch.tensor(drafts, dtype=torch.int32, device=dev)
    argmax_t = torch.tensor(argmaxes, dtype=torch.int32, device=dev)
    bonus_t = torch.tensor(bonuses, dtype=torch.int32, device=dev)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
    out = torch.full((n, max_spec_len + 1), -1, dtype=torch.int32, device=dev)
    nxt = torch.empty((n,), dtype=torch.int32, device=dev)
    vc = torch.empty((n,), dtype=torch.int32, device=dev)

    kernel[(n,)](out, nxt, vc, cu_t, draft_t, argmax_t, bonus_t, max_spec_len)
    torch.cuda.synchronize()

    out_l = out.cpu().tolist()
    vc_l = [int(v) for v in vc.cpu().tolist()]
    nxt_l = [int(v) for v in nxt.cpu().tolist()]
    width = max_spec_len + 1
    committed_per_req = [out_l[r][: max(0, min(vc_l[r], width))] for r in range(n)]
    return {
        "committed_per_req": committed_per_req,
        "valid_counts": vc_l,
        "next_token_ids": nxt_l,
    }


# --------------------------------------------------------------------------- #
# Oracle + differential audit (PURE functions — unit-checkable, no GPU)
# --------------------------------------------------------------------------- #
def oracle_accept(draft: list[int], argmax: list[int], bonus: int) -> list[int]:
    """Correct greedy spec-decode committed stream for one request.

    Accept ``draft[p]`` iff it equals ``argmax[p]``; at every position commit the
    in-step argmax (so the first-mismatch position commits the *correction* and
    then stops). If all draft tokens are accepted, append the bonus token.
    """
    committed: list[int] = []
    for p in range(len(draft)):
        committed.append(int(argmax[p]))
        if int(draft[p]) != int(argmax[p]):
            return committed
    committed.append(int(bonus))
    return committed


def _has_reject(case: dict) -> bool:
    return any(int(d) != int(a) for d, a in zip(case["draft"], case["argmax"]))


def audit_exactness(result: dict, cases: list[dict], gi: Any) -> dict:
    """Differential audit: kernel committed stream vs the in-step greedy ground truth.

    For each committed position the reference is the in-step target argmax
    (``argmax[p]`` at a draft slot, ``bonus`` at the all-accept slot). Reports:
      * ``exactness_rate`` — fraction of committed positions equal to the argmax
        reference (the PRIMARY per-position metric; 1.0 for the linear kernel).
      * ``num_length_violations`` — requests whose (committed, valid_count) differ
        from the pure-Python oracle (catches length-overrun even when every token
        value coincides with an argmax).
      * ``all_sha256_match`` — official sha256 of kernel-committed vs oracle stream.
      * ``first_violation`` — the first {req, pos, committed, ref_argmax} caught.
    Verdict GREEDY_EXACT iff zero token violations AND zero length violations AND
    all sha256 match; else VIOLATION.
    """
    committed_per_req = result["committed_per_req"]
    valid_counts = result["valid_counts"]

    total_committed = 0
    total_exact = 0
    first_violation: dict | None = None
    length_violations: list[dict] = []
    sha_mismatches: list[int] = []

    for r, c in enumerate(cases):
        draft, argmax, bonus = c["draft"], c["argmax"], int(c["bonus"])
        num_draft = len(draft)
        committed = [int(x) for x in committed_per_req[r]]
        raw_vc = int(valid_counts[r])

        for p, tok in enumerate(committed):
            if p < num_draft:
                ref: int | None = int(argmax[p])
            elif p == num_draft:
                ref = bonus
            else:  # committed beyond the bonus slot -> no valid greedy reference
                ref = None
            total_committed += 1
            if ref is not None and tok == ref:
                total_exact += 1
            elif first_violation is None:
                first_violation = {
                    "req": r,
                    "pos": p,
                    "committed": tok,
                    "ref_argmax": ref,
                    "label": c.get("label", f"case{r}"),
                }

        oracle = oracle_accept(draft, argmax, bonus)
        if raw_vc != len(oracle) or committed != oracle:
            length_violations.append(
                {
                    "req": r,
                    "label": c.get("label", f"case{r}"),
                    "kernel_valid": raw_vc,
                    "oracle_valid": len(oracle),
                    "kernel_committed": committed[:10],
                    "oracle_committed": oracle[:10],
                }
            )

        if gi.sha256_tokens(committed) != gi.sha256_tokens(oracle):
            sha_mismatches.append(r)

    rate = (total_exact / total_committed) if total_committed else 0.0
    num_violations = total_committed - total_exact
    all_sha_match = not sha_mismatches
    verdict = (
        "GREEDY_EXACT"
        if (num_violations == 0 and not length_violations and all_sha_match)
        else "VIOLATION"
    )
    return {
        "verdict": verdict,
        "exactness_rate": rate,
        "total_committed": total_committed,
        "total_exact": total_exact,
        "num_violations": num_violations,
        "first_violation": first_violation,
        "num_length_violations": len(length_violations),
        "length_violations": length_violations[:8],
        "all_sha256_match": all_sha_match,
        "num_sha256_mismatches": len(sha_mismatches),
        "num_requests": len(cases),
    }


# --------------------------------------------------------------------------- #
# Test battery — synthetic edge cases + realistic real-vocab cases
# --------------------------------------------------------------------------- #
def build_battery(ar_ref_path: Path | None = None, num_realistic: int = 16) -> list[dict]:
    """Construct the (draft, argmax, bonus) battery exercising the accept logic.

    Synthetic edge cases cover the full kernel control-flow (all-accept at every
    K, reject at every position 0..K-1, single-token accept/reject, varying
    num_draft). Realistic cases reuse real completion token ids from the on-disk
    M=1-AR reference so the kernel is exercised on production-magnitude int32 vocab
    ids (up to ~262143), half all-accept and half with a single planted mismatch.
    """
    cases: list[dict] = []
    K = DEFAULT_MAX_SPEC_LEN

    # all-accept blocks for num_draft = 1..K (commits argmax... + bonus)
    for nd in range(1, K + 1):
        base = [1000 + i for i in range(nd)]
        cases.append({"label": f"all_accept_nd{nd}", "draft": list(base), "argmax": list(base), "bonus": 2000 + nd})

    # reject at position j for a full K-length block (draft==argmax except at j)
    for j in range(K):
        draft = [3000 + i for i in range(K)]
        argmax = list(draft)
        argmax[j] = draft[j] + 777  # force mismatch at j -> oracle stops at j (+1)
        cases.append({"label": f"reject_at_{j}", "draft": draft, "argmax": argmax, "bonus": 4096})

    # single-token cases
    cases.append({"label": "single_accept", "draft": [55], "argmax": [55], "bonus": 99})
    cases.append({"label": "single_reject", "draft": [55], "argmax": [66], "bonus": 99})

    # varying num_draft with a mid-block reject
    for nd in (2, 3, 5):
        draft = [5000 + i for i in range(nd)]
        argmax = list(draft)
        argmax[nd // 2] = draft[nd // 2] + 11
        cases.append({"label": f"mid_reject_nd{nd}", "draft": draft, "argmax": argmax, "bonus": 6000})

    # realistic real-vocab cases from the on-disk AR reference
    if ar_ref_path and ar_ref_path.exists() and num_realistic > 0:
        cases.extend(_realistic_cases_from_ar(ar_ref_path, num_realistic, K))

    return cases


def _realistic_cases_from_ar(ar_ref_path: Path, num_realistic: int, K: int) -> list[dict]:
    """Derive real-vocab (draft, argmax, bonus) blocks from AR completions.

    Deterministic: even-indexed windows are all-accept; odd-indexed windows plant
    a single mismatch at a rotating position, using the next real token id as the
    substituted value (still a valid in-vocab id).
    """
    out: list[dict] = []
    with open(ar_ref_path, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    idx = 0
    for rec in records:
        toks = rec.get("completion_token_ids") or []
        if len(toks) < K + 2:
            continue
        off = (idx * 37) % (len(toks) - (K + 1))
        window = [int(t) for t in toks[off : off + K]]
        bonus = int(toks[off + K])
        draft = list(window)
        argmax = list(window)
        if idx % 2 == 1:
            j = idx % K
            sub = int(toks[off + K + 1])
            if sub == argmax[j]:
                sub = (sub + 1) % 262144
            argmax[j] = sub  # draft[j] (old) != argmax[j] (new) -> reject at j
        out.append({"label": f"real_{idx}", "draft": draft, "argmax": argmax, "bonus": bonus})
        idx += 1
        if idx >= num_realistic:
            break
    return out


# --------------------------------------------------------------------------- #
# Aggregate AR leg — completion-level batch-variance disambiguation
# --------------------------------------------------------------------------- #
def run_ar_aggregate_leg(spec_path: Path, ar_path: Path, gi: Any) -> dict:
    """Official completion-level compare (spec-ON capture vs M=1-AR reference).

    Reports the official verdict (expected DIVERGENT) and classifies it as the
    documented int4-Marlin batch-variance (M=1 vs M=8 split-K), explicitly NOT a
    greedy-exactness contract violation — the per-step kernel audit certifies the
    contract at rate 1.0. This makes the completion-level DIVERGENT/exact-rate-1.0
    distinction legible instead of conflating the two.
    """
    if not (spec_path.exists() and ar_path.exists()):
        return {
            "available": False,
            "note": f"missing capture(s): spec={spec_path.exists()} ar={ar_path.exists()}",
        }
    report = gi.compare_files(ar_path, spec_path)
    total = report.total_tokens_compared
    return {
        "available": True,
        "verdict": report.verdict,
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_tokens_compared": total,
        "total_divergent_tokens": report.total_divergent_tokens,
        "divergence_rate": (report.total_divergent_tokens / total) if total else 0.0,
        "classification": "batch_variance_expected_NOT_contract_violation",
        "note": (
            "M=1-AR vs batched-verify int4 split-K divergence is documented "
            "batch-variance (issue #124, RESOLVED), orthogonal to the per-step "
            "greedy-exactness contract the kernel-isolation audit certifies at 1.0."
        ),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY) + armed audit
# --------------------------------------------------------------------------- #
def run_self_test(battery: list[dict], gi: Any, ppl_value: float | None, ppl_cap: float, max_spec_len: int) -> dict:
    """PRIMARY self-validation: linear EXACT, injected faults CAUGHT, PPL holds."""
    # (a) known-good linear kernel must be GREEDY_EXACT (rate 1.0, sha match)
    linear = load_reference_kernel()
    res_lin = launch_kernel(linear, battery, max_spec_len)
    audit_lin = audit_exactness(res_lin, battery, gi)
    cond_a = (
        audit_lin["verdict"] == "GREEDY_EXACT"
        and audit_lin["exactness_rate"] == 1.0
        and audit_lin["all_sha256_match"]
    )

    # (b) injected non-argmax acceptance (over-accept, commits draft) must be CAUGHT
    bug2 = build_bug2_overaccept_kernel()
    res_bug = launch_kernel(bug2, battery, max_spec_len)
    audit_bug = audit_exactness(res_bug, battery, gi)
    cond_b = (
        audit_bug["verdict"] == "VIOLATION"
        and audit_bug["exactness_rate"] < 1.0
        and audit_bug["first_violation"] is not None
    )

    # (c) length-overrun differential: commit correct argmax VALUES but accept past
    # the first mismatch (per-position rate stays 1.0, but the oracle length check
    # flags it). This is the subtle descent BUG-2 the team most fears.
    reject_cases = [c for c in battery if _has_reject(c)][:4] or battery[:1]
    overrun_result = {
        "committed_per_req": [[int(x) for x in c["argmax"]] + [int(c["bonus"])] for c in reject_cases],
        "valid_counts": [len(c["argmax"]) + 1 for c in reject_cases],
        "next_token_ids": [int(c["bonus"]) for c in reject_cases],
    }
    audit_overrun = audit_exactness(overrun_result, reject_cases, gi)
    cond_c = audit_overrun["verdict"] == "VIOLATION" and audit_overrun["num_length_violations"] > 0

    # (d) PPL cross-check from the on-disk known-good capture
    cond_d = ppl_value is not None and ppl_value <= ppl_cap

    passes = bool(cond_a and cond_b and cond_c and cond_d)
    return {
        "greedy_exact_harness_self_test_passes": passes,
        "linear_stack_exactness_rate": audit_lin["exactness_rate"],
        "conditions": {
            "linear_GREEDY_EXACT_rate1p0_sha_match": cond_a,
            "bug2_overaccept_CAUGHT_position_named": cond_b,
            "lengthoverrun_CAUGHT_via_oracle_lengthcheck": cond_c,
            "ppl_cross_check_le_cap": cond_d,
        },
        "linear_audit": audit_lin,
        "bug2_audit": audit_bug,
        "lengthoverrun_audit": audit_overrun,
        "ppl": {"value": ppl_value, "cap": ppl_cap, "passed": cond_d},
        "battery_size": len(battery),
        "max_spec_len": max_spec_len,
    }


def run_armed_audit(symbol: str, battery: list[dict], gi: Any, max_spec_len: int) -> dict:
    """Run the identical differential against a same-signature land-#71 kernel."""
    kernel = load_kernel_symbol(symbol)
    res = launch_kernel(kernel, battery, max_spec_len)
    audit = audit_exactness(res, battery, gi)
    return {"audit_kernel_symbol": symbol, "audit": audit, "battery_size": len(battery), "max_spec_len": max_spec_len}


# --------------------------------------------------------------------------- #
# W&B logging (mirrors research/tree_submission_preflight/preflight.py)
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # logging must never break the instrument
        print(f"[greedy-exact] wandb logging unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="descent-greedy-exact-harness",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["descent-greedy-exact-harness", "validity-gate", "greedy-exactness"],
        config={
            "submission": str(SUBMISSION_DIR),
            "self_test": bool(args.self_test),
            "audit_kernel_symbol": args.audit_kernel_symbol or "",
            "ppl_cap": args.ppl_cap,
            "max_spec_len": args.max_spec_len,
            "num_realistic": args.num_realistic,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[greedy-exact] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {}
    if "self_test" in payload:
        st = payload["self_test"]
        summary["greedy_exact_harness_self_test_passes"] = int(bool(st["greedy_exact_harness_self_test_passes"]))
        summary["linear_stack_exactness_rate"] = st["linear_stack_exactness_rate"]
        for key, value in st["conditions"].items():
            summary[f"selftest_{key}"] = int(bool(value))
        summary["bug2_exactness_rate"] = st["bug2_audit"]["exactness_rate"]
        summary["bug2_num_violations"] = st["bug2_audit"]["num_violations"]
        summary["linear_total_committed"] = st["linear_audit"]["total_committed"]
        summary["lengthoverrun_num_length_violations"] = st["lengthoverrun_audit"]["num_length_violations"]
        if _finite(st["ppl"].get("value")):
            summary["known_good_ppl"] = st["ppl"]["value"]
            summary["known_good_ppl_margin"] = st["ppl"]["cap"] - st["ppl"]["value"]
        summary["battery_size"] = st["battery_size"]
    if "armed_audit" in payload:
        au = payload["armed_audit"]["audit"]
        summary["armed_exactness_rate"] = au["exactness_rate"]
        summary["armed_num_violations"] = au["num_violations"]
        summary["armed_num_length_violations"] = au["num_length_violations"]
        summary["armed_greedy_exact"] = int(au["verdict"] == "GREEDY_EXACT")
    if payload.get("ar_aggregate", {}).get("available"):
        ag = payload["ar_aggregate"]
        summary["ar_aggregate_divergence_rate"] = ag["divergence_rate"]
        summary["ar_aggregate_num_divergent"] = ag["num_divergent"]
        summary["ar_aggregate_total_tokens"] = ag["total_tokens_compared"]

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="greedy_exact_harness_result", artifact_type="harness", data=payload)
    finish_wandb(run)
    print(f"[greedy-exact] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_audit(label: str, audit: dict) -> None:
    line = "=" * 12 + f" {label} " + "=" * 12
    print("\n" + line, flush=True)
    print(
        f"  verdict={audit['verdict']}  rate={audit['exactness_rate']:.6f}  "
        f"committed={audit['total_committed']}  violations={audit['num_violations']}  "
        f"len_violations={audit['num_length_violations']}  sha_match={audit['all_sha256_match']}",
        flush=True,
    )
    if audit.get("first_violation"):
        fv = audit["first_violation"]
        print(
            f"  first_violation: req={fv['req']} pos={fv['pos']} label={fv['label']} "
            f"committed={fv['committed']} ref_argmax={fv['ref_argmax']}",
            flush=True,
        )
    print("=" * len(line), flush=True)


def _load_ppl(ppl_arg: float | None, ppl_summary: Path) -> float | None:
    if ppl_arg is not None:
        return ppl_arg
    if ppl_summary.exists():
        try:
            return float(json.loads(ppl_summary.read_text()).get("ppl"))
        except Exception:
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--audit-kernel-symbol", default=None,
                    help="arm the differential against a same-signature kernel: 'module:func' (land #71)")
    ap.add_argument("--ar-ref", type=Path, default=DEFAULT_AR_REF, help="M=1-AR reference decode_outputs.jsonl")
    ap.add_argument("--spec-capture", type=Path, default=DEFAULT_SPEC_CAPTURE, help="spec-ON capture decode_outputs.jsonl")
    ap.add_argument("--ppl", type=float, default=None, help="override PPL (default: read from on-disk ppl_summary.json)")
    ap.add_argument("--ppl-summary", type=Path, default=DEFAULT_PPL_SUMMARY)
    ap.add_argument("--ppl-cap", type=float, default=DEFAULT_PPL_CAP)
    ap.add_argument("--max-spec-len", type=int, default=DEFAULT_MAX_SPEC_LEN)
    ap.add_argument("--num-realistic", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="descent-greedy-exact-harness")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[greedy-exact] {note}", flush=True)

    gi = paths.import_greedy_identity()
    battery = build_battery(args.ar_ref, args.num_realistic)
    print(f"[greedy-exact] battery: {len(battery)} cases (max_spec_len={args.max_spec_len})", flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "runs" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": stamp,
        "submission": str(SUBMISSION_DIR),
        "ppl_cap": args.ppl_cap,
        "max_spec_len": args.max_spec_len,
        "battery_size": len(battery),
    }

    # Completion-level batch-variance disambiguation (free; on-disk captures).
    payload["ar_aggregate"] = run_ar_aggregate_leg(args.spec_capture, args.ar_ref, gi)

    exit_code = 0
    if args.self_test:
        ppl_value = _load_ppl(args.ppl, args.ppl_summary)
        st = run_self_test(battery, gi, ppl_value, args.ppl_cap, args.max_spec_len)
        payload["self_test"] = st
        _print_audit("LINEAR (known-good, must be GREEDY_EXACT)", st["linear_audit"])
        _print_audit("BUG-2 OVER-ACCEPT (must be CAUGHT)", st["bug2_audit"])
        _print_audit("LENGTH-OVERRUN (must be CAUGHT via oracle)", st["lengthoverrun_audit"])
        print(
            f"\nppl={st['ppl']['value']} (cap {st['ppl']['cap']}) -> {st['ppl']['passed']}",
            flush=True,
        )
        print(f"\nlinear_stack_exactness_rate = {st['linear_stack_exactness_rate']}", flush=True)
        print(f"conditions: {json.dumps(st['conditions'])}", flush=True)
        print(f"greedy_exact_harness_self_test_passes = {st['greedy_exact_harness_self_test_passes']}", flush=True)
        if not st["greedy_exact_harness_self_test_passes"]:
            exit_code = 1

    if args.audit_kernel_symbol:
        armed = run_armed_audit(args.audit_kernel_symbol, battery, gi, args.max_spec_len)
        payload["armed_audit"] = armed
        _print_audit(f"ARMED AUDIT ({args.audit_kernel_symbol})", armed["audit"])
        if armed["audit"]["verdict"] != "GREEDY_EXACT":
            exit_code = 1

    if not args.self_test and not args.audit_kernel_symbol:
        print("[greedy-exact] nothing to do: pass --self-test and/or --audit-kernel-symbol", flush=True)

    ag = payload["ar_aggregate"]
    if ag.get("available"):
        print(
            f"\n[aggregate AR leg] verdict={ag['verdict']} "
            f"divergence_rate={ag['divergence_rate']:.4f} "
            f"({ag['num_divergent']}/{ag['num_prompts_compared']} prompts) "
            f"-> {ag['classification']}",
            flush=True,
        )

    result_file = out_dir / "greedy_exact_harness_result.json"
    result_file.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"\n[greedy-exact] result -> {result_file}", flush=True)
    _maybe_log_wandb(args, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
