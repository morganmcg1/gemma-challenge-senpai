"""AdaEDL entropy-gated early-stop within the K-step MTP draft loop.

AdaEDL (arXiv:2410.18351): when the drafter's predictive entropy at draft
position j is high, the drafter is unsure and deeper draft tokens are almost
always rejected by the target. Stopping the draft early at the first
high-entropy position skips the wasted drafter forward passes at near-zero
acceptance cost.

Quality-safe BY CONSTRUCTION: greedy speculative decoding always emits the
target's argmax token (the rejection sampler's greedy kernel), so output text,
PPL, and greedy identity are independent of how many draft tokens were
proposed. Early-stop only changes E[accept] (i.e. speed), never the emitted
tokens.

Env-gated. ``apply()`` installs the override ONLY when ``DRAFT_STOP_ENTROPY``
is set to a float (the static raw-entropy threshold, in nats). When unset, the
stock vLLM ``propose`` path runs unchanged and the serving path is
byte-identical to the no-AdaEDL submission. The AdaEDL stopping rule
``1 - sqrt(gamma * H) < lambda`` (gamma=0.2) is equivalent to the static
threshold ``H > (1 - lambda)**2 / gamma``; we sweep ``DRAFT_STOP_ENTROPY``
directly rather than EMA-tuning lambda.

Threshold ``DRAFT_STOP_ENTROPY=inf`` is the CONTROL: same eager-entropy code
path, never stops -- isolates the AdaEDL machinery tax (eager de-graphed
centroid selection + per-position entropy + the stop-check H<->D sync) from the
early-stop gain. The clean decomposition is:
    unpatched (stock graphed) -> control(inf): machinery tax
    control(inf) -> threshold(tau): skip-the-tail benefit
    net AdaEDL vs ship = unpatched -> threshold = benefit - tax.

Sweep without reloading the model: set ``ADAEDL_THRESH_FILE`` to a path holding
one float. The threshold is re-read (mtime-cached) at the top of every
``propose`` call, so one inf-launched serve process can sweep all thresholds by
rewriting that file between decode passes. Falls back to the static
``DRAFT_STOP_ENTROPY`` value when the env/file is absent or unreadable.

Optional ``ADAEDL_OUT`` (a path) turns on DIAGNOSTIC logging: one JSONL record
per engine step, pairing the draft's per-position entropy vector
``H = [H_1..H_draft_len]`` with the accept_length the target returned for it
(in-band via ``num_rejected_tokens_gpu``). This is what Step 1 consumes to build
P(accept | H_j). Logging adds per-position H<->D syncs + file writes, so it is
OFF for the clean TPS passes (ADAEDL_OUT unset) and the only per-position sync
left is the intrinsic stop-check.
"""

import json
import math
import os

import torch
from vllm.forward_context import set_forward_context

_STATE = {
    "fh": None,
    "fh_failed": False,  # latch: stop retrying a bad ADAEDL_OUT path
    "step": 0,
    "prev": None,  # (draft_len, H_seq, stopped) for the previous propose()
    "errors": 0,
}

# mtime-cached parse of ADAEDL_THRESH_FILE (shared across the singleton proposer).
_THRESH_CACHE = {"mtime": None, "val": None}


def _log(msg):
    print(f"[adaedl-earlystop] {msg}", flush=True)


def _logging_active() -> bool:
    """Diagnostic logging is on when ADAEDL_OUT is set. When ADAEDL_LOG_FLAG names
    a path, logging is additionally gated on that flag-file existing -- so a single
    persistent serve can alternate logged (E_accept/draft_len records) and clean
    (zero-overhead TPS) decode passes by touching/removing the flag, no relaunch."""
    if not os.environ.get("ADAEDL_OUT"):
        return False
    flag = os.environ.get("ADAEDL_LOG_FLAG")
    if flag:
        return os.path.exists(flag)
    return True


def _fh():
    """Lazily open the ADAEDL_OUT records file (append, line-buffered). A bad
    path degrades to 'logging disabled' (latched) -- a diagnostic I/O error must
    never crash the serving EngineCore."""
    if _STATE["fh_failed"]:
        return None
    if _STATE["fh"] is None:
        path = os.environ.get("ADAEDL_OUT")
        if not path:
            return None
        try:
            _STATE["fh"] = open(path, "a", buffering=1)  # line-buffered
        except OSError as exc:
            _STATE["fh_failed"] = True
            _log(f"ADAEDL_OUT open failed ({path!r}): {exc!r}; logging disabled")
            return None
        _log(f"writing early-stop records to {path}")
    return _STATE["fh"]


def _adaedl_refresh_thresh(self) -> None:
    """Live-update ``self._adaedl_thresh`` from ``ADAEDL_THRESH_FILE`` (one float),
    mtime-cached. No-op (keeps the static apply() value) when the env is unset or
    the file is missing/unreadable/unparseable -- never wedges a running serve."""
    path = os.environ.get("ADAEDL_THRESH_FILE")
    if not path:
        return
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return
    if mt == _THRESH_CACHE["mtime"]:
        if _THRESH_CACHE["val"] is not None:
            self._adaedl_thresh = _THRESH_CACHE["val"]
        return
    try:
        with open(path) as fh:
            val = float(fh.read().strip())
    except (OSError, ValueError):
        return
    _THRESH_CACHE["mtime"] = mt
    _THRESH_CACHE["val"] = val
    self._adaedl_thresh = val
    _log(f"threshold refreshed from {path}: {val}")


def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Eager replacement for the centroids-graph greedy sample.

    One ``_select_and_score`` yields BOTH the byte-identical draft token
    (replicating ``Gemma4MTPMaskedEmbedder.get_top_tokens`` line-for-line) and
    the per-row predictive entropy used by the early-stop gate -- avoiding a
    doubled centroid selection. The token math is identical to get_top_tokens,
    so the proposed draft IDs are byte-for-byte the stock graphed path's; only
    the (unused-by-the-emit) entropy stash is added.
    """
    masked_emb = getattr(self.model, "masked_embedding", None)
    if masked_emb is None:
        # No centroid masking -> no AdaEDL entropy; disable the gate.
        self._adaedl_H = None
        return super(type(self), self)._greedy_sample(hidden_states)
    lm_head_weight = self.model._get_full_lm_head_weight()
    logits, indices = masked_emb._select_and_score(hidden_states, lm_head_weight)
    token = indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
    probs = logits.float().softmax(dim=-1)
    self._adaedl_H = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return token


def _adaedl_should_stop(self) -> bool:
    """True iff every active row's entropy exceeds the threshold.

    Conservative for batch>1 (never truncate a still-confident sequence); for
    the deployed MAX_NUM_SEQS=1 single-stream regime this is the single row.
    The ``.item()`` is an intrinsic H<->D sync: a Python-driven draft loop must
    learn on the CPU whether to break before launching the next iteration."""
    H = getattr(self, "_adaedl_H", None)
    if H is None:
        return False
    return bool((H > self._adaedl_thresh).all().item())


def _row0_entropy(self) -> float:
    """Materialize the single-stream (row 0) entropy scalar for the records.
    Only called when ADAEDL_OUT logging is active."""
    H = getattr(self, "_adaedl_H", None)
    if H is None:
        return float("nan")
    return float(H.reshape(-1)[0].item())


def _noop_centroids(self) -> None:
    """The eager ``_greedy_sample`` above never reads the centroids graphs, so
    skip capturing them (saves load time + VRAM)."""
    self._centroids_sizes = []


def _emit_prev_record(self, num_rejected_tokens_gpu, K):
    """Pair the previous draft's per-position entropy + realized length with the
    accept_length the target just reported for it. ``num_rejected_tokens_gpu`` at
    this step describes the draft produced at the previous step (vLLM uses it to
    resume that sequence). Single-stream: num_reqs == 1."""
    prev = _STATE["prev"]
    if prev is None or num_rejected_tokens_gpu is None:
        return
    fh = _fh()
    if fh is None:
        return
    try:
        num_rej = int(num_rejected_tokens_gpu.sum().item())
        draft_len, h_seq, stopped = prev
        rec = {
            "step": _STATE["step"],
            "K": K,
            "draft_len": draft_len,
            "accept_length": K - num_rej,
            "stopped": stopped,
            "thresh": self._adaedl_thresh,
            "H": [round(h, 6) for h in h_seq if not math.isnan(h)],
        }
        fh.write(json.dumps(rec) + "\n")
        _STATE["step"] += 1
    except Exception as exc:  # noqa: BLE001
        if _STATE["errors"] < 5:
            _log(f"record emit failed: {exc!r}")
        _STATE["errors"] += 1


def _adaedl_propose(
    self,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    token_indices_to_sample,
    common_attn_metadata,
    sampling_metadata,
    mm_embed_inputs=None,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    slot_mappings=None,
) -> torch.Tensor:
    """Faithful copy of ``SpecDecodeBaseProposer.propose`` (vLLM 0.22.0) for the
    Gemma4 MTP proposer, with an AdaEDL entropy-gated early break + pad-to-K.

    The Gemma4 proposer's ``method`` is never eagle3/dflash, so the base class's
    leading eagle3 ``combine_hidden_states`` branch is dead here and omitted.
    Everything else is verbatim; the only additions are the lines marked
    ``# ADAEDL``.
    """
    self._adaedl_H = None  # ADAEDL: reset per-call entropy stash
    self._last_draft_probs = None
    _adaedl_refresh_thresh(self)  # ADAEDL: live threshold (file override)
    log_on = _logging_active()  # ADAEDL: diagnostic logging?
    h_seq = [] if log_on else None  # ADAEDL: per-position entropy accumulator
    batch_size = common_attn_metadata.batch_size()

    # ADAEDL: pair the previous draft with the accept signal carried in-band.
    if log_on:
        _emit_prev_record(self, num_rejected_tokens_gpu, self.num_speculative_tokens)

    num_tokens, token_indices_to_sample, common_attn_metadata = (
        self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
    )

    per_group_attn_metadata, per_layer_attn_metadata = (
        self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
    )

    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        self._determine_batch_execution_and_padding(num_tokens)
    )

    model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
        num_tokens, num_input_tokens, mm_embed_inputs
    )

    with set_forward_context(
        per_layer_attn_metadata,
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=self._get_slot_mapping(
            slot_mapping_size, common_attn_metadata.slot_mapping
        ),
    ):
        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

    sample_hidden_states = last_hidden_states[token_indices_to_sample]

    # Early exit if there is only one draft token to be generated.
    if self.num_speculative_tokens == 1 or self.parallel_drafting:
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        if draft_probs is not None:
            self._last_draft_probs = draft_probs.view(
                -1, self.num_speculative_tokens, draft_probs.shape[-1]
            ).contiguous()
        return draft_token_ids.view(-1, self.num_speculative_tokens)

    if self.uses_mrope:
        positions = self.mrope_positions[:, token_indices_to_sample]
    else:
        positions = self.positions[token_indices_to_sample]
    hidden_states = hidden_states[token_indices_to_sample]

    if self.constant_draft_positions:
        self.positions[:batch_size] = positions

    draft_token_ids, draft_probs = self._sample_draft_tokens(
        sample_hidden_states, sampling_metadata
    )
    draft_probs_list = None if draft_probs is None else [draft_probs]

    if log_on:  # ADAEDL: record entropy of draft position 1
        h_seq.append(_row0_entropy(self))
    stop_now = _adaedl_should_stop(self)  # ADAEDL

    if self.allowed_attn_types is not None:
        for group_md in per_group_attn_metadata:
            if not isinstance(group_md, self.allowed_attn_types):
                raise ValueError(
                    f"Unsupported attention metadata type for speculative "
                    "decoding with num_speculative_tokens > 1: "
                    f"{type(group_md)}. Supported types are: "
                    f"{self.allowed_attn_types}"
                )

    # Generate the remaining draft tokens.
    draft_token_ids_list = [draft_token_ids]

    cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
        self._determine_batch_execution_and_padding(batch_size)
    )

    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
    common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
        self.token_arange_np[: batch_size + 1]
    ).clone()

    if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None

    block_size = self.block_size
    assert block_size > 0, "block_size has not been initialized."
    for token_index in range(self.num_speculative_tokens - 1):
        if stop_now:  # ADAEDL: skip remaining drafter forwards
            break
        input_ids = draft_token_ids_list[-1].int()

        if not self.constant_draft_positions:
            positions = self._update_positions_dependent_metadata(
                positions,
                common_attn_metadata,
                batch_size,
                input_batch_size,
                block_size,
            )

        if not self.constant_draft_positions or token_index == 0:
            _, per_layer_attn_metadata = (
                self.build_per_group_and_layer_attn_metadata(
                    common_attn_metadata, draft_index=token_index + 1
                )
            )

        self.input_ids[:batch_size] = input_ids
        self.hidden_states[:batch_size] = hidden_states
        if self.supports_mm_inputs:
            self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

            input_ids = None
            inputs_embeds = self.inputs_embeds[:input_batch_size]
        else:
            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(input_batch_size),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=input_batch_size,
            num_tokens_across_dp=batch_size_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(input_batch_size),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        hidden_states = hidden_states[:batch_size]
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            last_hidden_states[:batch_size], sampling_metadata
        )
        if draft_probs is not None:
            assert draft_probs_list is not None
            draft_probs_list.append(draft_probs)
        draft_token_ids_list.append(draft_token_ids)
        if log_on:  # ADAEDL: record entropy of this draft position
            h_seq.append(_row0_entropy(self))
        stop_now = _adaedl_should_stop(self)  # ADAEDL: re-check after each sample

    # ADAEDL: pad to K with the last real token (Strategy C). Greedy verify
    # rejects these padded positions; the fixed-width return contract
    # (draft_token_ids_cpu[:num_reqs] is [max_num_reqs, K]) requires width K.
    draft_len = len(draft_token_ids_list)
    if draft_len < self.num_speculative_tokens:
        pad_tok = draft_token_ids_list[-1]
        pad_prob = draft_probs_list[-1] if draft_probs_list is not None else None
        while len(draft_token_ids_list) < self.num_speculative_tokens:
            draft_token_ids_list.append(pad_tok)
            if draft_probs_list is not None:
                draft_probs_list.append(pad_prob)

    # ADAEDL: stash this draft's realized length + entropy vector for next step's
    # accept pairing (diagnostic only).
    if log_on:
        _STATE["prev"] = (draft_len, h_seq, draft_len < self.num_speculative_tokens)

    # [batch_size, num_speculative_tokens]
    draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
    if draft_probs_list is not None:
        self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
    return draft_token_ids


def apply(module):
    """Install the AdaEDL early-stop on ``Gemma4Proposer`` iff DRAFT_STOP_ENTROPY
    is set. No-op otherwise (stock byte-identical propose path)."""
    thr = os.environ.get("DRAFT_STOP_ENTROPY")
    if thr is None:
        return
    threshold = float(thr)  # may be inf for the control
    cls = module.Gemma4Proposer
    if getattr(cls, "_adaedl_installed", False):
        return
    cls._adaedl_thresh = threshold
    cls._greedy_sample = _greedy_sample
    cls.propose = _adaedl_propose
    cls._setup_centroids_cuda_graphs = _noop_centroids
    cls._adaedl_installed = True
    _log(f"installed AdaEDL early-stop: DRAFT_STOP_ENTROPY={threshold}")
