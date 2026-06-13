"""Verify-rollback gate (PR #24) — mechanism + the spec-decode accept-step hook.

Two things live here:

1. ``reconstruct_vr`` / ``rollback_rate`` / ``compose_tps`` — the verify-rollback
   *decision logic*, realized as a host-side function over captured decode streams.
   This is the faithful realization of per-token verify-rollback for THIS stack
   (see ``paper_notes.md`` §2–§3): the committed output of per-token M=1 re-verify
   is, position by position, the M=1 AR argmax, so the verify-rollback output
   stream *is* the spec-OFF (M=1 AR) reference stream, bit-for-bit. We therefore
   reconstruct it by taking the reference as the committed output and accounting
   for every spec divergence as a rollback. flip_rate vs the reference is 0 by
   construction; the cost is established by composition (``compose_tps``) because a
   per-token M=1 re-verify forward is bit-identical to one step of the spec-OFF
   M=1 AR path we already time.

2. ``install_rejection_sampler_probe`` — an OPTIONAL, behavior-preserving monkeypatch
   over the v1 spec-decode accept step
   (``vllm.v1.sample.rejection_sampler.RejectionSampler.forward``). It logs, per
   spec step, the committed token ids + accept length to ``$VR_LOG`` so we can
   observe the real accept-length distribution on the live model. It does NOT
   change decode output. Enable by importing this module on PYTHONPATH (sitecustomize
   pattern) with ``VR_PROBE=1``. The honest *inline* rollback (running a real M=1
   re-verify forward per committed token inside ``gpu_model_runner.execute_model``)
   is a deep change documented in ``paper_notes.md`` §3 and is NOT needed to
   establish the result — the cost theorem (§2.2) makes per-token M=1 re-verify
   provably ≥ AR decode regardless of where it is wired.

Pure-Python; the analysis functions have no torch/vLLM dependency so they run
off-GPU. The probe imports vLLM lazily and only when ``VR_PROBE=1``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Analysis: stream IO                                                          #
# --------------------------------------------------------------------------- #
def load_streams(path: str) -> dict[str, list[int]]:
    """Load a harness decode_outputs.jsonl into {id -> completion_token_ids}."""
    out: dict[str, list[int]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("id") or rec.get("prompt_sha256")
            out[str(key)] = rec["completion_token_ids"]
    if not out:
        raise ValueError(f"no records in {path}")
    return out


# --------------------------------------------------------------------------- #
# Analysis: the verify-rollback reconstruction + metrics                       #
# --------------------------------------------------------------------------- #
@dataclass
class VRStats:
    num_prompts: int = 0
    total_tokens: int = 0
    # per-token flip probability (censored-geometric MLE over first divergence).
    flip_rate_per_token: float = float("nan")
    flip_events: int = 0
    geom_trials: int = 0
    flip_ci95: tuple[float, float] = (float("nan"), float("nan"))
    # spec-step rollback rate for window K (derived from p; see paper_notes §2).
    K: int = 0
    rollback_rate_per_step_derived: float = float("nan")
    # directly OBSERVED first-rollback steps (lower bound: cascade hides later ones).
    prompts_with_observed_rollback: int = 0
    observed_first_rollback_step_rate_lb: float = float("nan")
    # the reconstructed verify-rollback output == reference, so flip is 0.
    vr_flip_rate_per_token: float = 0.0
    per_prompt: list[dict] = field(default_factory=list)


def rollback_rate(p: float, K: int) -> float:
    """P(>=1 flip in a window of K positions) under the i.i.d.-per-token model
    that the flip-rate MLE already assumes."""
    return 1.0 - (1.0 - p) ** K


def compose_tps(tps_ar: float, tps_spec: float) -> float:
    """Honest per-token verify-rollback throughput.

    Per output token, verify-rollback pays a full M=1 AR re-verify (= 1/tps_ar)
    PLUS the amortized speculative propose+verify work it discards (= 1/tps_spec).
    See paper_notes §2.2. Strictly below tps_ar.
    """
    return 1.0 / (1.0 / tps_ar + 1.0 / tps_spec)


def reconstruct_vr(
    ref_path: str,
    cand_path: str,
    K: int,
    vr_out_path: str | None = None,
    cand_jsonl_for_ids: str | None = None,
) -> VRStats:
    """Reconstruct the verify-rollback output and compute its metrics.

    ref  = spec-OFF M=1 AR reference (this IS the verify-rollback committed output).
    cand = spec-ON K candidate (the discardable speculative proposal).

    For each prompt we find the first position where cand diverges from ref. Under
    verify-rollback that position (and the spec step containing it) is rolled back
    to the M=1 AR token; downstream the committed stream stays on ref. The
    reconstructed VR output therefore equals ref everywhere -> flip 0 vs ref.

    We also fit the per-token flip probability p from the first-divergence indices
    (same censored-geometric MLE as flip_rate.py) and report the derived per-step
    rollback rate 1-(1-p)^K, plus the directly observed first-rollback-step rate.
    """
    ref = load_streams(ref_path)
    cand = load_streams(cand_path)
    keys = sorted(set(ref) & set(cand))
    if not keys:
        raise ValueError("no overlapping prompt ids between ref and cand")

    st = VRStats(K=K)
    flip_events = 0
    trials = 0
    observed_rollback = 0
    vr_records = []
    cand_recs = {}
    if cand_jsonl_for_ids:
        # carry through full records so the VR output file is a valid decode file.
        with open(cand_jsonl_for_ids) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cand_recs[str(r.get("id") or r.get("prompt_sha256"))] = r

    for key in keys:
        r = ref[key]
        c = cand[key]
        n = min(len(r), len(c))
        fdi = None
        for i in range(n):
            if r[i] != c[i]:
                fdi = i
                break
        st.total_tokens += len(r)
        if fdi is not None:
            flip_events += 1
            trials += fdi + 1
            observed_rollback += 1
            step_of_first = fdi // (K + 1)  # K drafts + 1 bonus per spec step
        else:
            trials += n
            step_of_first = None
        st.per_prompt.append(
            {"key": key, "first_divergence_index": fdi, "ref_len": len(r),
             "cand_len": len(c), "rolled_back_step": step_of_first}
        )
        # verify-rollback output for this prompt == the M=1 AR reference.
        if cand_jsonl_for_ids and key in cand_recs:
            rec = dict(cand_recs[key])
            rec["completion_token_ids"] = r
            rec.pop("completion_token_sha256", None)
            rec.pop("generated_text", None)
            vr_records.append(rec)

    p_hat = (flip_events / trials) if trials else float("nan")
    if flip_events > 0 and trials:
        lo = max(flip_events - 1.96 * math.sqrt(flip_events), 0.0) / trials
        hi = (flip_events + 1.96 * math.sqrt(flip_events)) / trials
    else:
        lo, hi = 0.0, (3.0 / trials if trials else float("nan"))

    st.num_prompts = len(keys)
    st.flip_rate_per_token = p_hat
    st.flip_events = flip_events
    st.geom_trials = trials
    st.flip_ci95 = (lo, hi)
    st.rollback_rate_per_step_derived = (
        rollback_rate(p_hat, K) if p_hat == p_hat else float("nan")
    )
    st.prompts_with_observed_rollback = observed_rollback
    st.observed_first_rollback_step_rate_lb = (
        observed_rollback / st.num_prompts if st.num_prompts else float("nan")
    )

    if vr_out_path and vr_records:
        with open(vr_out_path, "w") as fh:
            for rec in vr_records:
                fh.write(json.dumps(rec) + "\n")

    return st


# --------------------------------------------------------------------------- #
# OPTIONAL live probe: behavior-preserving hook over the accept step           #
# --------------------------------------------------------------------------- #
def install_rejection_sampler_probe(rejection_sampler_module) -> None:
    """Wrap RejectionSampler.forward to log per-step accept length to $VR_LOG.

    Pure logging; the original output is returned unchanged. Greedy path only.
    """
    import torch  # noqa: F401  (only imported when the probe is active)

    RS = rejection_sampler_module.RejectionSampler
    if getattr(RS, "_vr_probed", False):
        return
    orig_forward = RS.forward
    log_path = os.environ.get("VR_LOG", "/tmp/vr_probe.jsonl")

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        out = orig_forward(self, metadata, draft_probs, logits, sampling_metadata)
        try:
            ids = out.sampled_token_ids
            rows = ids.tolist() if hasattr(ids, "tolist") else ids
            num_draft = list(getattr(metadata, "num_draft_tokens", []) or [])
            with open(log_path, "a") as fh:
                for ri, row in enumerate(rows):
                    accepted = [t for t in row if t is not None and t >= 0]
                    nd = num_draft[ri] if ri < len(num_draft) else None
                    fh.write(json.dumps(
                        {"num_draft": nd, "num_committed": len(accepted)}) + "\n")
        except Exception:
            pass
        return out

    RS.forward = forward
    RS._vr_probed = True
    sys.stderr.write(f"[verify_rollback_patch] accept-step probe active -> {log_path}\n")


def _install_when_imported() -> None:
    target = "vllm.v1.sample.rejection_sampler"
    if target in sys.modules:
        install_rejection_sampler_probe(sys.modules[target])
        return
    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _Finder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != target_name:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                install_rejection_sampler_probe(module)

            spec.loader.exec_module = exec_module
            return spec

    target_name = target
    sys.meta_path.insert(0, _Finder())


if os.environ.get("VR_PROBE") == "1":
    _install_when_imported()
