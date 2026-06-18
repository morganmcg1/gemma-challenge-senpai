"""Env-gated recompute-acceptor patch — PR #642 cost probe + PR #663 REAL acceptor.

Both modes fire width-1 ``w4a16-ct`` *target* forwards (``_dummy_run`` with
``slot_mapping=-1`` so the injected forward never writes the live KV cache) into the
served spec-decode loop AFTER each verify step. The emitted token stream is left
UNCHANGED in both modes: greedy identity of the rescued stream is certified
separately, offline, along the served M=1 AR trajectory (``served_identity_scan.py``)
— the live verify numerics are not recoverable from the API — so here we only measure
the *cost* and the *realized firing rate*. ``wall_tps = completion_tokens/decode_s``
therefore reflects ``(un-rescued spec throughput) − (recompute overhead)``.

Modes (no-op unless one env is set; the shipped serve path is byte-identical when
neither is):

  * RATE mode  — ``SENPAI_RECOMPUTE_RATE=r`` (r>0).  #642 cost probe. After each step
    fire ``floor(carry + r*emitted)`` forwards (a fractional carry makes the long-run
    total exactly ``r*total_emitted``). ``r=0`` is the un-rescued ceiling; a small
    sweep fits the additive in-loop marginal cost C.

  * TAU mode   — ``SENPAI_ACCEPTOR_TAU=t`` (t>0).  #663 REAL gap-flag acceptor. Read
    the live verify logit gaps (top1−top2) at the spec *target* positions and fire one
    width-1 recompute per position whose gap ``< t``. The firing rate is therefore
    DATA-DRIVEN — it emerges from the real gap distribution, not from an injected rate
    — which is exactly the real gap-flag M=1-recompute acceptor #663 asks for. The
    realized rate (flagged/positions) is logged so it can be checked against the
    offline scan's ``flag_trigger_rate``.

Cudagraph dispatch is RESOLVED empirically (settles #642's [eager,captured]
bracket by direct observation, not assumption):

  * ``SENPAI_RECOMPUTE_CUDAGRAPH=1`` → ``cudagraph_runtime_mode=None``: the dispatcher
    picks the runtime mode, replaying a captured graph iff a key matches (captured arm).
  * unset → ``CUDAGraphMode.NONE``: forced eager — every injected forward breaks the
    captured decode graph exactly as a real interleaved recompute would (eager floor).

On the first fire we dump ``cudagraph_dispatcher.cudagraph_keys`` and the result of
``dispatch(1, uniform_decode=True)`` — the mode ``_dummy_run(1, uniform_decode=True)``
itself resolves to — so whether a width-1 recompute *replays a captured size-1 decode
graph* in this spec stack is answered by the live dispatcher, not guessed.

A non-firing patch fails loud: it logs the realized rate periodically and (unless
``SENPAI_RECOMPUTE_REQUIRE_FIRE=0``) asserts that real forwards fired, so a silent
no-op can never again be mistaken for "zero overhead".
"""

from __future__ import annotations

import json
import os
import sys

_PATCH_FLAG = "_optionb_recompute_acceptor_patch_applied"

_RATE_ENV = "SENPAI_RECOMPUTE_RATE"
_TAU_ENV = "SENPAI_ACCEPTOR_TAU"
_TARGET_ONLY_ENV = "SENPAI_RECOMPUTE_TARGET_ONLY"
_REQUIRE_FIRE_ENV = "SENPAI_RECOMPUTE_REQUIRE_FIRE"
_CUDAGRAPH_ENV = "SENPAI_RECOMPUTE_CUDAGRAPH"
_STAT_DIR_ENV = "SENPAI_RECOMPUTE_STAT_DIR"

# PR #669 Item 2 -- LIVE-trajectory de-teacher-force identity certificate.
#   SENPAI_LIVECERT_REF_JSONL=<R_served decode jsonl>  (the SENPAI_REFERENCE_MODE=1
#     served M=1-AR trajectory; prompts keyed by sha256(",".join prompt_token_ids)).
#   SENPAI_LIVECERT_STAT_DIR=<dir>  (per-PID cert sidecar JSON; read-only otherwise).
# When set, the wrapper records -- READ ONLY, the emitted stream is UNCHANGED -- the
# per-position verify gap + de-teacher-forced flip (verify-argmax != R_served token)
# along the LIVE K-spec trajectory, partitioned PRE-FORK (on R_served) vs GLOBAL (all
# positions), so the min break-free flag predicate is MEASURED on the live loop, not
# reconstructed offline. flips can only occur at draft-verify rows (the bonus row's
# argmax shares R_served's context under BI=1, so it is never a flip on-trajectory).
_LIVECERT_REF_ENV = "SENPAI_LIVECERT_REF_JSONL"
_LIVECERT_STAT_DIR_ENV = "SENPAI_LIVECERT_STAT_DIR"
_LIVECERT_NPROMPTS_ENV = "SENPAI_LIVECERT_N_PROMPTS"
# When >0, the recorder writes a per-step alignment trace (req 0 only) to stderr for the
# first N decode steps -- used to validate the emitted<->R_served pos alignment. 0 = off.
try:
    _LIVECERT_DBG = int(os.environ.get("SENPAI_LIVECERT_DEBUG_STEPS", "0") or "0")
except (TypeError, ValueError):
    _LIVECERT_DBG = 0

# tau-flag sweep for the live cert: loose -> tight. The min break-free tau gives the
# min_safe_live_flag_rate. Includes the offline scan's points (0.2,0.25,0.3,0.5,...)
# plus finer steps around the offline flip_gap_max (0.25) to localize the knee.
_LIVECERT_TAU_SWEEP = (
    0.1, 0.15, 0.2, 0.22, 0.25, 0.27, 0.28, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0,
)

_LOG_EVERY = int(os.environ.get("SENPAI_RECOMPUTE_LOG_EVERY", "4096"))


def _livecert_ref_path() -> str:
    return os.environ.get(_LIVECERT_REF_ENV, "") or ""


def _rate() -> float:
    try:
        return float(os.environ.get(_RATE_ENV, "0") or "0")
    except (TypeError, ValueError):
        return 0.0


def _tau() -> float:
    try:
        return float(os.environ.get(_TAU_ENV, "0") or "0")
    except (TypeError, ValueError):
        return 0.0


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _write_stat(kind: str, payload: dict) -> None:
    """Atomically dump a sidecar stat JSON to ``$SENPAI_RECOMPUTE_STAT_DIR``.

    No-op when the env var is unset (the shipped serving path never writes)."""
    stat_dir = os.environ.get(_STAT_DIR_ENV, "")
    if not stat_dir:
        return
    try:
        os.makedirs(stat_dir, exist_ok=True)
        pid = os.getpid()
        path = os.path.join(stat_dir, f"recompute_{kind}_{pid}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": pid, **payload}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _write_livecert_stat(kind: str, payload: dict) -> None:
    """Dump the live-cert sidecar JSON to ``$SENPAI_LIVECERT_STAT_DIR`` (no-op when
    unset). Written at every progress boundary AND at exit so the final file holds the
    run totals even though there is no clean per-serve teardown hook."""
    stat_dir = os.environ.get(_LIVECERT_STAT_DIR_ENV, "")
    if not stat_dir:
        return
    try:
        os.makedirs(stat_dir, exist_ok=True)
        pid = os.getpid()
        path = os.path.join(stat_dir, f"livecert_{kind}_{pid}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": pid, **payload}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _hash_tokens(token_ids) -> str:
    """sha256 of the comma-joined token ids -- byte-identical to the harness keying
    (``check_greedy_identity.sha256_tokens`` / the reference jsonl ``prompt_token_sha256``)."""
    import hashlib

    return hashlib.sha256(
        ",".join(str(int(t)) for t in token_ids).encode("ascii")
    ).hexdigest()


def _load_ref_index(ref_jsonl: str, n_prompts: int) -> dict[str, list]:
    """Map ``prompt_token_sha256 -> R_served completion_token_ids`` from the served
    M=1-AR reference decode jsonl. Uses the stored ``prompt_token_sha256`` when present
    (verified equal to ``_hash_tokens(prompt_token_ids)``), else recomputes it."""
    index: dict[str, list] = {}
    if not ref_jsonl or not os.path.exists(ref_jsonl):
        return index
    with open(ref_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            pids = r.get("prompt_token_ids") or []
            comp = r.get("completion_token_ids") or []
            if not pids or not comp:
                continue
            key = r.get("prompt_token_sha256") or _hash_tokens(pids)
            index[key] = list(comp)
            if n_prompts and len(index) >= n_prompts:
                break
    return index


def _gaps_at(logits, idx_tensor, torch) -> list:
    """top1-top2 verify gap for every row selected by ``idx_tensor`` (indices into the
    gathered sampled logits ``ems[1]``), returned as a CPU python list. Used only in
    livecert mode (analysis), so the one ``.tolist()`` sync per step is acceptable."""
    if idx_tensor is None or idx_tensor.numel() == 0:
        return []
    rows = logits.index_select(0, idx_tensor)
    top2 = torch.topk(rows, 2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).tolist()


def _count_emitted(output, torch) -> int:
    """Per-step emitted-token count, robust to both served output shapes.

    * sync ``ModelRunnerOutput.sampled_token_ids`` is ``list[list[int]]`` (one inner
      list per request; spec decode emits 1..M accepted ids).
    * async ``AsyncGPUModelRunnerOutput`` does NOT expose a public
      ``sampled_token_ids`` until ``get_output()`` runs later; the device tensor lives
      on ``_sampled_token_ids`` ``[num_reqs, max_spec+1]`` padded with ``-1`` at
      rejected positions. Read it directly so emitted advances (and so RATE-mode
      firing, which keys off ``rate*emitted``, is not a silent no-op on async outputs).
    Returns 0 for the PP/kv-only early-return outputs (``None``/empty)."""
    stids = getattr(output, "sampled_token_ids", None)
    if stids is None:
        stids = getattr(output, "_sampled_token_ids", None)
    if stids is None:
        return 0
    if isinstance(stids, torch.Tensor):
        if stids.numel() == 0:
            return 0
        return int((stids >= 0).sum().item())
    try:
        return int(sum(len(x) for x in stids))
    except TypeError:
        return 0


def _gap_flag_count(self, ems, tau: float, torch) -> tuple[int, int]:
    """Read the live verify-logit gaps at the spec target positions and count how
    many fall below ``tau``. Returns ``(flagged, positions)``.

    ``ems`` is the unpacked ``execute_model_state`` 10-tuple set just before
    ``sample_tokens`` consumes it: ``ems[1]`` is the target ``logits``
    ``[total_scheduled, vocab]`` and ``ems[2]`` is the ``SpecDecodeMetadata`` whose
    ``target_logits_indices`` select the draft-aligned verify rows. The gap is
    ``top1 − top2`` of each verify row — the same statistic the offline identity scan
    flags on, so the realized rate here is directly comparable to the scan's
    ``flag_trigger_rate``."""
    try:
        logits = ems[1]
        sdm = ems[2]
        if logits is None or sdm is None:
            return 0, 0
        tli = getattr(sdm, "target_logits_indices", None)
        if tli is None or tli.numel() == 0:
            return 0, 0
        verify = logits.index_select(0, tli)
        top2 = torch.topk(verify, 2, dim=-1).values
        gaps = top2[:, 0] - top2[:, 1]
        positions = int(gaps.numel())
        flagged = int((gaps < tau).sum().item())
        return flagged, positions
    except Exception:
        return 0, 0


def _count_positions(ems, torch) -> int:
    """Number of spec verify (draft-aligned) positions this step -- ``.numel()`` on the
    index tensor, a SHAPE read, NO device sync. This drives RATE-mode firing
    (``floor(carry + rate*positions)``) so the recompute count tracks the verify
    positions the acceptor actually guards, with zero per-step sync (PR #669 Item 1)."""
    try:
        sdm = ems[2] if ems is not None else None
        if sdm is None:
            return 0
        tli = getattr(sdm, "target_logits_indices", None)
        if tli is None:
            return 0
        return int(tli.numel())
    except Exception:
        return 0


def _accumulate_emitted(self, output, torch) -> None:
    """Advance the emitted-token accumulators WITHOUT a per-step ``.item()`` sync
    (PR #669 Item 1). Emitted count is now pure logging -- neither RATE (per-position)
    nor TAU (gap-mask) firing keys off it -- so we never block the stream to read it:

      * async tensor output -> enqueue a GPU reduction into ``self._rc_emitted_gpu``
        (a 0-dim device tensor); materialized once at each log boundary / at exit.
      * sync list output -> ``sum(len(x))`` is already host-side, add to a python int.

    A run is consistently one shape, so only one accumulator advances."""
    stids = getattr(output, "sampled_token_ids", None)
    if stids is None:
        stids = getattr(output, "_sampled_token_ids", None)
    if stids is None:
        return
    if isinstance(stids, torch.Tensor):
        if stids.numel() == 0:
            return
        contrib = (stids >= 0).sum()  # GPU scalar; no sync
        prev = getattr(self, "_rc_emitted_gpu", None)
        self._rc_emitted_gpu = contrib if prev is None else prev + contrib
    else:
        try:
            self._rc_emitted_cpu = getattr(self, "_rc_emitted_cpu", 0) + int(
                sum(len(x) for x in stids)
            )
        except TypeError:
            pass


def _materialize_emitted(self) -> int:
    """Realize the emitted total (single ``.item()`` -- only at log boundaries / exit)."""
    total = int(getattr(self, "_rc_emitted_cpu", 0))
    g = getattr(self, "_rc_emitted_gpu", None)
    if g is not None:
        try:
            total += int(g.item())
        except Exception:
            pass
    return total


def _emitted_rows(output, torch) -> list[list[int]]:
    """Per-request emitted token-id lists (input-batch req order), unifying both output
    shapes. async ``_sampled_token_ids`` ``[num_reqs, max_spec+1]`` is ``-1``-padded
    (valid ids are the contiguous non-(-1) prefix); sync ``sampled_token_ids`` is
    already the clean ``list[list[int]]``. Used only in livecert mode."""
    stids = getattr(output, "sampled_token_ids", None)
    if stids is not None and not isinstance(stids, torch.Tensor):
        return [[int(t) for t in row] for row in stids]
    stids = getattr(output, "_sampled_token_ids", None)
    if stids is None or not isinstance(stids, torch.Tensor) or stids.numel() == 0:
        return []
    rows: list[list[int]] = []
    for row in stids.tolist():
        valid: list[int] = []
        for t in row:
            if t == -1:
                break
            valid.append(int(t))
        rows.append(valid)
    return rows


def _livecert_record(self, ems, output, torch) -> None:
    """PR #669 Item 2 -- LIVE de-teacher-force recorder (READ ONLY; emitted stream
    UNCHANGED). For every active request, walk this step's emitted tokens (== the live
    verify argmaxes at the accepted/corrected positions) IN ORDER and compare each to
    the served M=1-AR reference R_served at the same output position:

      flip(pos)       = emitted_tok != R_served[pos]          (the live fork signal)
      flag(pos, tau)  = verify_gap(pos) < tau                 (acceptor would recompute)
      break(pos, tau) = flip AND gap >= tau                   (flag MISSED -> identity lost)

    Positions are partitioned PRE-FORK (recorded only while the request's live stream
    is still byte-identical to R_served; recording stops at the first flip, which is
    itself the last on-trajectory measurement) vs GLOBAL (all draft rows every step,
    forked or not -> reproduces the un-rescued cost-probe per-verify-position flag rate).
    The min tau with zero pre-fork breaks gives ``min_safe_live_flag_rate``."""
    lc = getattr(self, "_lc", None)
    if lc is None:
        return
    try:
        req_ids = list(self.input_batch.req_ids)
    except Exception:
        return
    rows = _emitted_rows(output, torch)
    if not rows:
        return
    taus = _LIVECERT_TAU_SWEEP

    # A pure-prefill step samples the first token but carries NO SpecDecodeMetadata
    # (ems[2] is None). It still EMITS one token per active request, which advances the
    # true output position -- so it MUST be walked to keep ``pos[rid]`` aligned with
    # R_served; skipping it desyncs every later comparison by one (the off-by-one that
    # made all requests fork at position 0). Only the gap-based flag/break accounting is
    # spec-only; the prefill bonus is a no-gap emit position.
    logits = ems[1] if ems is not None else None
    sdm = ems[2] if ems is not None else None
    num_draft = getattr(sdm, "num_draft_tokens", None) if sdm is not None else None
    has_spec = logits is not None and num_draft is not None
    if has_spec:
        gaps_draft = _gaps_at(logits, getattr(sdm, "target_logits_indices", None), torch)
        gaps_bonus = _gaps_at(logits, getattr(sdm, "bonus_logits_indices", None), torch)
        # ---- GLOBAL: every draft-verify row this step (un-partitioned) ----
        lc["global_draft_positions"] += len(gaps_draft)
        gfd = lc["global_flag_draft"]
        for g in gaps_draft:
            for t in taus:
                if g < t:
                    gfd[t] += 1
    else:
        gaps_draft = []
        gaps_bonus = []

    # ---- PRE-FORK: per request, emitted tokens until first divergence ----
    n = min(len(req_ids), len(rows))
    off = 0
    pos = lc["pos"]
    forked = lc["forked"]
    refmap = lc["ref"]
    pfe = lc["prefork_flag_emit"]
    pfd = lc["prefork_flag_draft"]
    pbk = lc["prefork_break"]
    step = lc["step"] = lc.get("step", 0) + 1
    for i in range(n):
        D_i = int(num_draft[i]) if has_spec else 0
        draft_off, off = off, off + D_i
        rid = req_ids[i]
        if rid in forked:
            continue
        if rid not in refmap:
            lc["n_reqs_seen"] += 1
            ref = None
            req = self.requests.get(rid) if hasattr(self.requests, "get") else None
            pids = getattr(req, "prompt_token_ids", None) if req is not None else None
            if pids:
                ref = lc["ref_index"].get(_hash_tokens(pids))
            refmap[rid] = ref
            if ref is not None:
                lc["n_matched"] += 1
            pos[rid] = 0
        ref = refmap[rid]
        if ref is None:
            continue
        emitted_i = rows[i]
        if not emitted_i:
            continue
        draft_gaps_i = gaps_draft[draft_off:draft_off + D_i] if has_spec else []
        bonus_g = gaps_bonus[i] if (has_spec and i < len(gaps_bonus)) else None
        base = pos[rid]
        if _LIVECERT_DBG and i == 0 and step <= _LIVECERT_DBG:
            sys.stderr.write(
                "[SENPAI_DIAG livecert] step=%d rid=%s spec=%s D=%d base=%d "
                "emit[:4]=%s ref[base:base+4]=%s\n"
                % (step, rid, has_spec, D_i, base, emitted_i[:4],
                   [int(x) for x in ref[base:base + 4]])
            )
            sys.stderr.flush()
        forked_here = exhausted = False
        consumed = 0
        for j, tok in enumerate(emitted_i):
            rpos = base + j
            if rpos >= len(ref):
                exhausted = True
                break
            if has_spec and j < D_i:
                g = draft_gaps_i[j]
                is_draft_row = True
            elif has_spec:
                g = bonus_g
                is_draft_row = False
            else:
                g = None  # prefill bonus: no spec metadata, no gap
                is_draft_row = False
            is_flip = tok != int(ref[rpos])
            lc["prefork_emit_positions"] += 1
            if g is not None:
                for t in taus:
                    if g < t:
                        pfe[t] += 1
                if is_draft_row:
                    lc["prefork_draft_positions"] += 1
                    for t in taus:
                        if g < t:
                            pfd[t] += 1
            consumed += 1
            if is_flip:
                # A no-gap (prefill) flip is un-flaggable -> breaks at every tau.
                lc["flip_gaps"].append(float(g) if g is not None else float("inf"))
                if g is None:
                    lc["n_prefill_flips"] += 1
                pbd = lc["prefork_break_draftonly"]
                for t in taus:
                    if g is None or g >= t:
                        pbk[t] += 1
                    # draft-only: a no-gap flip is NOT a draft-position break (the
                    # acceptor cannot recompute a position that has no draft row).
                    if g is not None and g >= t:
                        pbd[t] += 1
                if len(lc["flip_records"]) < 2000:
                    lc["flip_records"].append({
                        "rid": str(rid), "rpos": rpos, "step": step,
                        "gap": (float(g) if g is not None else None),
                        "has_spec": bool(has_spec), "is_draft_row": bool(is_draft_row),
                        "emit_tok": int(tok), "ref_tok": int(ref[rpos]),
                    })
                forked_here = True
                break
        if forked_here or exhausted:
            forked.add(rid)
            if forked_here:
                lc["n_forked"] += 1
        else:
            pos[rid] = base + consumed


def _livecert_summary(self) -> dict:
    """Compose the cert from the run accumulators: per-tau flag/break rates (pre-fork &
    global), the min break-free tau (-> min_safe_live_flag_rate), rule-of-three UB."""
    lc = getattr(self, "_lc", None)
    if lc is None:
        return {}
    taus = _LIVECERT_TAU_SWEEP
    pe = lc["prefork_emit_positions"]
    pd = lc["prefork_draft_positions"]
    gd = lc["global_draft_positions"]
    per_tau = {}
    min_safe = None
    min_safe_draftonly = None
    pbd = lc.get("prefork_break_draftonly", {})
    for t in taus:
        brk = lc["prefork_break"][t]
        brk_do = pbd.get(t, 0)
        per_tau[f"{t:g}"] = {
            "tau": t,
            "prefork_break_count": brk,
            "prefork_break_count_draftonly": brk_do,
            "prefork_break_rate_over_draft": (brk / pd) if pd else None,
            "prefork_break_rate_draftonly_over_draft": (brk_do / pd) if pd else None,
            "prefork_flag_rate_per_emit": (lc["prefork_flag_emit"][t] / pe) if pe else None,
            "prefork_flag_rate_per_draft": (lc["prefork_flag_draft"][t] / pd) if pd else None,
            "global_flag_rate_per_draft": (lc["global_flag_draft"][t] / gd) if gd else None,
        }
        if brk == 0 and min_safe is None:
            min_safe = t
        if brk_do == 0 and min_safe_draftonly is None:
            min_safe_draftonly = t
    flip_gaps = lc["flip_gaps"]
    finite_gaps = [g for g in flip_gaps if g != float("inf")]
    summary = {
        "ref_jsonl": _livecert_ref_path(),
        "n_reqs_seen": lc["n_reqs_seen"],
        "n_matched": lc["n_matched"],
        "n_forked": lc["n_forked"],
        "prefork_emit_positions": pe,
        "prefork_draft_positions": pd,
        "global_draft_positions": gd,
        "n_flips_prefork": len(flip_gaps),
        "n_prefill_flips": lc.get("n_prefill_flips", 0),
        "n_draft_flips": len(flip_gaps) - lc.get("n_prefill_flips", 0),
        "flip_gap_max": max(flip_gaps) if flip_gaps else 0.0,
        "flip_gap_max_finite": max(finite_gaps) if finite_gaps else 0.0,
        "flip_gap_min": min(flip_gaps) if flip_gaps else None,
        "per_tau": per_tau,
        "min_safe_tau": min_safe,
        "min_safe_live_flag_rate_per_draft": (
            (lc["prefork_flag_draft"][min_safe] / pd) if (min_safe and pd) else None
        ),
        "min_safe_live_flag_rate_per_emit": (
            (lc["prefork_flag_emit"][min_safe] / pe) if (min_safe and pe) else None
        ),
        "served_rescued_break_rate_at_min_safe": (
            (lc["prefork_break"][min_safe] / pd) if (min_safe and pd) else None
        ),
        # ---- draft-only partition (excludes structurally-unflaggable prefill flips) ----
        "min_safe_tau_draftonly": min_safe_draftonly,
        "min_safe_live_flag_rate_per_draft_draftonly": (
            (lc["prefork_flag_draft"][min_safe_draftonly] / pd)
            if (min_safe_draftonly and pd) else None
        ),
        "min_safe_live_flag_rate_per_emit_draftonly": (
            (lc["prefork_flag_emit"][min_safe_draftonly] / pe)
            if (min_safe_draftonly and pe) else None
        ),
        "served_rescued_break_rate_at_min_safe_draftonly": (
            (pbd.get(min_safe_draftonly, 0) / pd)
            if (min_safe_draftonly and pd) else None
        ),
        "rule_of_three_ub_over_draft": (3.0 / pd) if pd else None,
        "flip_records": lc.get("flip_records", []),
    }
    return summary


def apply(gmr) -> bool:
    """Wrap ``gmr.GPUModelRunner.sample_tokens`` to fire recompute forwards.

    ``gmr`` is the live ``vllm.v1.worker.gpu_model_runner`` module; every free symbol
    is resolved from it so we never guess import paths against a vLLM version. Returns
    True if applied, False if already present or if neither mode is enabled."""
    GPUModelRunner = gmr.GPUModelRunner
    if getattr(GPUModelRunner, _PATCH_FLAG, False):
        return False

    rate = _rate()
    tau = _tau()
    livecert_ref = _livecert_ref_path()
    livecert = bool(livecert_ref)
    if rate <= 0.0 and tau <= 0.0 and not livecert:
        # No-op: shipped serve path. Do not wrap.
        return False

    torch = gmr.torch
    CUDAGraphMode = gmr.CUDAGraphMode
    logger = gmr.init_logger("int4_mtp_drafter.recompute_acceptor")

    target_only = _flag(_TARGET_ONLY_ENV, True)
    require_fire = _flag(_REQUIRE_FIRE_ENV, True)
    use_cudagraph = _flag(_CUDAGRAPH_ENV, False)
    if livecert:
        mode_name = "LIVECERT(de-teacher-force)"
    elif tau > 0.0:
        mode_name = "TAU(real-gap)"
    else:
        mode_name = "RATE(inject,per-position)"

    # Pre-load the served M=1-AR reference index (prompt_token_sha256 -> R_served) once,
    # in the worker that owns the gap tensors. Stashed on the class so the wrapper can
    # lazily seed a per-runner ``self._lc`` accumulator on first use.
    livecert_ref_index: dict[str, list] = {}
    if livecert:
        try:
            n_prompts = int(os.environ.get(_LIVECERT_NPROMPTS_ENV, "0") or "0")
        except (TypeError, ValueError):
            n_prompts = 0
        livecert_ref_index = _load_ref_index(livecert_ref, n_prompts)
        logger.warning(
            "[recompute-acceptor] LIVECERT ref=%s prompts_indexed=%d taus=%s",
            livecert_ref, len(livecert_ref_index), _LIVECERT_TAU_SWEEP,
        )

    logger.warning(
        "[recompute-acceptor] APPLY wrapping GPUModelRunner.sample_tokens mode=%s "
        "rate=%.6f tau=%.6f livecert=%s target_only=%s require_fire=%s cudagraph=%s "
        "(mode=%s) log_every=%d",
        mode_name, rate, tau, livecert, target_only, require_fire, use_cudagraph,
        "None/dispatcher" if use_cudagraph else "NONE/eager", _LOG_EVERY,
    )

    orig_sample_tokens = GPUModelRunner.sample_tokens
    recompute_cg_mode = None if use_cudagraph else CUDAGraphMode.NONE

    def _resolve_dispatch(self) -> None:
        """Dump the live cudagraph dispatch for a width-1 uniform-decode forward —
        the empirical answer to whether the recompute replays a captured graph."""
        try:
            disp = self.cudagraph_dispatcher
            dmode, bd = disp.dispatch(1, uniform_decode=True)
            keys = {
                str(k): sorted(str(d) for d in v)
                for k, v in disp.cudagraph_keys.items()
            }
            cc = self.compilation_config
            payload = {
                "dispatch_1_uniform_decode_mode": str(dmode),
                "dispatch_1_uniform_decode_batch_desc": str(bd),
                "is_captured_replay": str(dmode) != str(CUDAGraphMode.NONE),
                "use_cudagraph_env": use_cudagraph,
                "cudagraph_keys": keys,
                "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
                "cudagraph_mode": str(cc.cudagraph_mode),
                "max_cudagraph_capture_size": getattr(
                    cc, "max_cudagraph_capture_size", None
                ),
                "uniform_decode_query_len": getattr(
                    self, "uniform_decode_query_len", None
                ),
                "num_spec_tokens": getattr(self, "num_spec_tokens", None),
            }
            logger.warning(
                "[recompute-acceptor] DISPATCH RESOLVE: dispatch(1,uniform_decode="
                "True) -> mode=%s captured_replay=%s | capture_sizes=%s "
                "uniform_decode_query_len=%s num_spec_tokens=%s | FULL_keys=%s",
                dmode, payload["is_captured_replay"],
                payload["cudagraph_capture_sizes"],
                payload["uniform_decode_query_len"], payload["num_spec_tokens"],
                keys.get(str(CUDAGraphMode.FULL), []),
            )
            _write_stat("dispatch_resolve", payload)
        except Exception:
            logger.exception("[recompute-acceptor] dispatch resolve failed")

    # Live-cert runners are tracked so an atexit hook can flush the final summary even
    # for the trailing < _LOG_EVERY positions after the last progress boundary.
    _lc_runners: list = []
    _lc_atexit_registered = {"v": False}

    def _lc_flush_all() -> None:
        for r in _lc_runners:
            try:
                _write_livecert_stat("summary", _livecert_summary(r))
            except Exception:
                pass

    def _livecert_init(self) -> None:
        if getattr(self, "_lc", None) is not None:
            return
        self._lc = {
            "ref_index": livecert_ref_index,
            "pos": {}, "forked": set(), "ref": {},
            "global_draft_positions": 0,
            "global_flag_draft": {t: 0 for t in _LIVECERT_TAU_SWEEP},
            "prefork_emit_positions": 0, "prefork_draft_positions": 0,
            "prefork_flag_emit": {t: 0 for t in _LIVECERT_TAU_SWEEP},
            "prefork_flag_draft": {t: 0 for t in _LIVECERT_TAU_SWEEP},
            "prefork_break": {t: 0 for t in _LIVECERT_TAU_SWEEP},
            # break accounting that EXCLUDES no-gap (prefill, output-pos-0) flips, which
            # are structurally un-flaggable by a gap predicate (no draft row exists for
            # the prefill token) and hence outside the recompute-acceptor's domain. The
            # draft-only min-safe tau is the acceptor's true reducibility; the prefill
            # flips are reported separately (PR #669 Item-2 partition).
            "prefork_break_draftonly": {t: 0 for t in _LIVECERT_TAU_SWEEP},
            "n_prefill_flips": 0,
            "flip_records": [],
            "flip_gaps": [], "n_forked": 0, "n_reqs_seen": 0, "n_matched": 0,
            "step": 0,
        }
        _lc_runners.append(self)
        if not _lc_atexit_registered["v"]:
            import atexit

            atexit.register(_lc_flush_all)
            # LocalServer terminates the serve process group with SIGTERM (not a clean
            # interpreter exit), so atexit alone never fires for a short run that has
            # not crossed a _LOG_EVERY boundary. Chain a SIGTERM handler that flushes
            # the final composed cert, then defer to the previous handler so vLLM's
            # graceful shutdown is preserved. signal.signal only works on the main
            # thread; fall back to atexit-only otherwise.
            try:
                import signal

                _prev_sigterm = signal.getsignal(signal.SIGTERM)

                def _lc_sigterm(signum, frame, _prev=_prev_sigterm):
                    _lc_flush_all()
                    if callable(_prev):
                        _prev(signum, frame)
                    elif _prev == signal.SIG_DFL:
                        signal.signal(signal.SIGTERM, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)

                signal.signal(signal.SIGTERM, _lc_sigterm)
            except (ValueError, OSError, RuntimeError):
                pass
            _lc_atexit_registered["v"] = True

    def _fire(self, n: int) -> int:
        if n <= 0:
            return 0
        fired = 0
        with torch.inference_mode():
            for _ in range(n):
                self._dummy_run(
                    1,
                    cudagraph_runtime_mode=recompute_cg_mode,
                    uniform_decode=True,
                    allow_microbatching=False,
                    skip_eplb=True,
                )
                fired += 1
        return fired

    def sample_tokens(self, grammar_output):
        # Peek the ephemeral verify state BEFORE the original consumes/clears it. Our
        # local ``ems`` keeps the 10-tuple (and its logits tensor) alive across the
        # original call even though it nulls ``self.execute_model_state``.
        ems = self.execute_model_state
        # ``positions`` is a SHAPE read (sync-free) and drives BOTH the progress gate
        # and RATE-mode per-position firing -- no ``.item()`` in the hot path (Item 1).
        positions = _count_positions(ems, torch)
        flagged = 0
        if tau > 0.0 and ems is not None:
            # TAU firing is intrinsically gap-mask-driven, so its ``.item()`` stays;
            # the EMITTED ``.item()`` (pure logging) is what Item 1 removes.
            flagged, _ = _gap_flag_count(self, ems, tau, torch)

        output = orig_sample_tokens(self, grammar_output)

        # Emitted advances SYNC-FREE (materialized only at log boundaries / exit).
        _accumulate_emitted(self, output, torch)

        # Live-cert de-teacher-force (read-only; emitted stream unchanged).
        if livecert:
            _livecert_init(self)
            _livecert_record(self, ems, output, torch)

        # First real step: resolve the cudagraph dispatch + first-call diag (firing
        # modes only -- livecert never fires, so dispatch replay is irrelevant there).
        if not getattr(self, "_recompute_diag_done", False) and positions > 0:
            self._recompute_diag_done = True
            if not livecert:
                _resolve_dispatch(self)
            logger.warning(
                "[recompute-acceptor] FIRST-CALL output_type=%s positions=%d "
                "flagged=%d rate=%.6f tau=%.6f livecert=%s",
                type(output).__name__, positions, flagged, rate, tau, livecert,
            )
            _write_stat(
                "firstcall",
                {
                    "output_type": type(output).__name__,
                    "positions": positions,
                    "flagged": flagged,
                    "rate": rate,
                    "tau": tau,
                    "livecert": livecert,
                },
            )

        # Decide how many recompute forwards to fire this step. RATE mode now fires
        # ``floor(carry + rate*positions)`` -- PER VERIFY-POSITION, the unit the real
        # acceptor guards -- so the recompute count is exact and the firing decision is
        # sync-free (no emitted ``.item()``). r=0 reproduces the un-rescued K-spec
        # ceiling with the patch loaded (the Item-1 measurement, not a projection).
        if tau > 0.0:
            n_fire = flagged
        elif rate > 0.0:
            acc = getattr(self, "_recompute_acc", 0.0) + rate * positions
            n_fire = int(acc)
            self._recompute_acc = acc - n_fire
        else:
            n_fire = 0

        # Accumulators (positions/flagged/fired are cheap ints; emitted is lazy).
        self._recompute_positions_total = (
            getattr(self, "_recompute_positions_total", 0) + positions
        )
        self._recompute_flagged_total = (
            getattr(self, "_recompute_flagged_total", 0) + flagged
        )

        if n_fire > 0:
            self._recompute_fired_total = getattr(
                self, "_recompute_fired_total", 0
            ) + _fire(self, n_fire)

        # Periodic progress log + sidecar stat, gated on positions (sync-free, advances
        # every decode step). Emitted is materialized HERE only (one ``.item()``).
        last = getattr(self, "_recompute_last_log", 0)
        pos_total = self._recompute_positions_total
        if pos_total - last >= _LOG_EVERY:
            self._recompute_last_log = pos_total
            emitted_total = _materialize_emitted(self)
            flagged_total = self._recompute_flagged_total
            fired_total = getattr(self, "_recompute_fired_total", 0)
            realized_fire_rate = (fired_total / emitted_total) if emitted_total else 0.0
            realized_flag_rate = (flagged_total / pos_total) if pos_total else 0.0
            logger.warning(
                "[recompute-acceptor] PROGRESS mode=%s emitted=%d positions=%d "
                "flagged=%d fired=%d realized_fire_rate=%.5f realized_flag_rate=%.5f",
                mode_name, emitted_total, pos_total, flagged_total, fired_total,
                realized_fire_rate, realized_flag_rate,
            )
            _write_stat(
                "progress",
                {
                    "mode": mode_name,
                    "rate": rate,
                    "tau": tau,
                    "use_cudagraph": use_cudagraph,
                    "emitted_total": emitted_total,
                    "positions_total": pos_total,
                    "flagged_total": flagged_total,
                    "fired_total": fired_total,
                    "realized_fire_rate": realized_fire_rate,
                    "realized_flag_rate": realized_flag_rate,
                },
            )
            if livecert:
                _write_livecert_stat("summary", _livecert_summary(self))

        return output

    sample_tokens.__name__ = orig_sample_tokens.__name__
    sample_tokens.__qualname__ = getattr(
        orig_sample_tokens, "__qualname__", "GPUModelRunner.sample_tokens"
    )
    GPUModelRunner.sample_tokens = sample_tokens
    setattr(GPUModelRunner, _PATCH_FLAG, True)

    # Fail-loud guard: a require_fire run that emitted tokens but never fired is a
    # silent no-op; surface it by stashing the predicate on the class so the harness
    # / sidecar stat exposes it. (The assert lives in the harness summary, not here,
    # to avoid crashing a live server mid-stream.)
    GPUModelRunner._recompute_require_fire = require_fire
    return True
