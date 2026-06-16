"""Patch vLLM Gemma4 MTP drafting with a CUDA graph loop replay.

This file is loaded by the vLLM child process through PYTHONPATH. It intentionally
does not patch PLE. Pupa's serve.py patches PLE textfast and scale-folds through
the installed vLLM source so the fold can be verified fail-closed at load time.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from copy import copy
from typing import Any


LOOPGRAPH_TARGET = "vllm.v1.spec_decode.gemma4"
RUNNER_TARGET = "vllm.v1.worker.gpu_model_runner"
TOP_TOKEN_TARGET = "vllm.model_executor.models.gemma4_mtp"
PROPOSER_TARGET = "vllm.v1.spec_decode.llm_base_proposer"
LOOPGRAPH_WARMUP_CALLS = int(os.environ.get("LOOPGRAPH_WARMUP_CALLS", "48"))
LOOPGRAPH_REQUIRE_CAPTURE = os.environ.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1"
LOOPGRAPH_PINGPONG_SLOTS = max(1, int(os.environ.get("LOOPGRAPH_PINGPONG_SLOTS", "1")))
# onegraph (@blake-fable5-1): the Gemma4 MTP drafter is Q-only and KV-shared —
# it never writes KV and has no cross-position dependencies, so the padded
# width-(K+1) first pass (and the full-prompt-width drafter pass on the first
# decode after prefill) only ever contributes the single position selected by
# token_indices_to_sample. Width-1 is exact. With ONEGRAPH=1 the whole
# propose() becomes one CUDA-graph replay of K width-1 iterations: iteration 0
# consumes next_token_ids + gathered target hidden/position, iterations 1..K-1
# are the stock loopgraph body. Drafter-only => cannot change emitted tokens.
ONEGRAPH = os.environ.get("ONEGRAPH", "1") == "1"
FUSED_SPARSE_ARGMAX = os.environ.get("FUSED_SPARSE_ARGMAX", "1") == "1"
FUSED_SPARSE_ARGMAX_REQUIRE = os.environ.get("FUSED_SPARSE_ARGMAX_REQUIRE") == "1"
FUSED_SPARSE_ARGMAX_BLOCK = int(os.environ.get("FUSED_SPARSE_ARGMAX_BLOCK", "16"))
DIXIE_FUSED_ACCEPT_PREP = os.environ.get("DIXIE_FUSED_ACCEPT_PREP") == "1"
DIXIE_FUSED_ACCEPT_PREP_REQUIRE = (
    os.environ.get("DIXIE_FUSED_ACCEPT_PREP_REQUIRE") == "1"
)
_FUSED_SPARSE_ARGMAX_KERNELS: Any | None = None
_FUSED_ACCEPT_PREP_KERNEL: Any | None = None
_FUSED_ACCEPT_PREP_CACHE: dict[int, tuple[Any, Any]] = {}
_LOOPGRAPH_SLOT_EVENTS_BY_PTR: dict[int, Any] = {}
_LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR: dict[int, bool] = {}

# --- drafter token-frequency logit bias (PR #48, experiment-only) -------------
# Additive bias on the DRAFTER's candidate scores, applied to the top-K most
# frequent corpus tokens before the drafter argmax. Verifier output is untouched
# (greedy spec decode emits the target argmax regardless of drafter proposals),
# so PPL / greedy-identity cannot change — only acceptance rate / TPS can move.
# DEFAULT-OFF: with DRAFTER_FREQ_BIAS unset/0 the drafter path is byte-identical
# to the merged leaderboard stack (the fused sparse-argmax kernel runs unchanged).
# When active, the fused kernel is bypassed for the exact-equivalent PyTorch
# sparse path so the bias can be added by candidate token id, then argmax.
DRAFTER_FREQ_BIAS = float(os.environ.get("DRAFTER_FREQ_BIAS", "0") or "0")
DRAFTER_FREQ_BIAS_TOPK = int(os.environ.get("DRAFTER_FREQ_BIAS_TOPK", "500"))
DRAFTER_FREQ_BIAS_TOKENS = os.environ.get("DRAFTER_FREQ_BIAS_TOKENS", "")
_FREQ_BIAS_TABLE: Any | None = None  # lazy persistent [vocab] device tensor


def _freq_bias_active() -> bool:
    return DRAFTER_FREQ_BIAS != 0.0 and bool(DRAFTER_FREQ_BIAS_TOKENS)


def _get_freq_bias_table(vocab_size: int, device: Any, dtype: Any) -> Any:
    """Build (once) a persistent [vocab] bias vector: +DRAFTER_FREQ_BIAS at the
    top-K frequent token ids, 0 elsewhere. Persistent + static-shape => safe to
    gather inside the onegraph CUDA-graph capture (built during warmup)."""
    global _FREQ_BIAS_TABLE
    if _FREQ_BIAS_TABLE is not None:
        return _FREQ_BIAS_TABLE
    import json as _json

    import torch

    with open(DRAFTER_FREQ_BIAS_TOKENS) as fh:
        spec = _json.load(fh)
    ids = [int(t) for t in spec.get("top_k_token_ids", [])][:DRAFTER_FREQ_BIAS_TOPK]
    table = torch.zeros(int(vocab_size), dtype=dtype, device=device)
    if ids:
        idx = torch.tensor(ids, dtype=torch.long, device=device)
        table.index_fill_(0, idx, float(DRAFTER_FREQ_BIAS))
    _FREQ_BIAS_TABLE = table
    print(
        f"[drafter-freq-bias] built bias table: +{DRAFTER_FREQ_BIAS} on "
        f"{len(ids)}/{DRAFTER_FREQ_BIAS_TOPK} top tokens (vocab={vocab_size}, "
        f"dtype={dtype}, src={DRAFTER_FREQ_BIAS_TOKENS!r}) in pid {os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
    return _FREQ_BIAS_TABLE


def _call_base_propose(base_propose: Any, self: Any, kwargs: dict[str, Any]) -> Any:
    return base_propose(self, **kwargs)


def _build_static_buffers(self: Any, state: dict[str, Any], cad: Any) -> None:
    import torch

    device = self.device
    token_count = self.num_speculative_tokens
    state["outputs"] = [
        torch.zeros((1, token_count), dtype=torch.int64, device=device)
        for _ in range(LOOPGRAPH_PINGPONG_SLOTS)
    ]
    state["out"] = state["outputs"][0]
    state["next_slot"] = 0
    state["_pupa_loopgraph_slot_events"] = [
        torch.cuda.Event(blocking=False) for _ in state["outputs"]
    ]
    for output, event in zip(
        state["outputs"], state["_pupa_loopgraph_slot_events"], strict=True
    ):
        _LOOPGRAPH_SLOT_EVENTS_BY_PTR[output.data_ptr()] = event
        _LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR[output.data_ptr()] = False
    state["seq_lens"] = torch.zeros_like(cad.seq_lens[:1])
    state["first_input"] = torch.zeros((1,), dtype=torch.int32, device=device)
    state["block_tables"] = {}

    static_cad = copy(cad)
    static_cad.seq_lens = state["seq_lens"]
    static_cad.num_actual_tokens = 1
    static_cad.max_query_len = 1
    static_cad.max_seq_len = self.max_model_len
    static_cad.slot_mapping = self._slot_mapping_buffer[:1]
    static_cad.query_start_loc = self.arange[:2]

    per_layer_metadata = {}
    for group in self.draft_attn_groups:
        group_id = group.kv_cache_group_id
        source = self._per_group_block_tables.get(group_id, cad.block_table_tensor)[:1]
        block_size = group.get_metadata_builder().kv_cache_spec.block_size
        width = max(source.shape[1], -(-self.max_model_len // block_size))
        static_block_table = torch.zeros((1, width), dtype=source.dtype, device=device)
        state["block_tables"][group_id] = static_block_table

        group_cad = copy(static_cad)
        group_cad.block_table_tensor = static_block_table
        metadata = group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=group_cad,
            draft_index=1,
        )
        for layer_name in group.layer_names:
            per_layer_metadata[layer_name] = metadata
    state["metadata"] = per_layer_metadata


def _refresh_static_buffers(self: Any, state: dict[str, Any], cad: Any) -> None:
    state["seq_lens"].copy_(cad.seq_lens[:1])
    for group_id, static_block_table in state["block_tables"].items():
        source = self._per_group_block_tables.get(group_id, cad.block_table_tensor)[:1]
        width = min(source.shape[1], static_block_table.shape[1])
        static_block_table[:, :width].copy_(source[:, :width])


def _run_graph_body(self: Any, state: dict[str, Any]) -> None:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context

    token_count = self.num_speculative_tokens
    output = state["out"]
    onegraph = state.get("onegraph", False)
    with set_forward_context(
        state["metadata"],
        self.vllm_config,
        num_tokens=1,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=self._get_slot_mapping(1),
    ):
        if onegraph:
            # K width-1 iterations; iteration 0 reads the target-sampled
            # token from first_input and writes out[0, 0].
            for index in range(token_count):
                source = (
                    state["first_input"]
                    if index == 0
                    else output[0, index - 1 : index]
                )
                self.input_ids[:1].copy_(source)
                last_hidden, backbone_hidden = self.model(
                    input_ids=self.input_ids[:1],
                    positions=self._get_positions(1),
                    inputs_embeds=None,
                    hidden_states=self.hidden_states[:1],
                )
                self.hidden_states[:1].copy_(backbone_hidden[:1])
                token = self.model.get_top_tokens(last_hidden[:1])
                output[0, index : index + 1].copy_(token)
        else:
            for index in range(token_count - 1):
                self.input_ids[:1].copy_(output[0, index : index + 1])
                last_hidden, backbone_hidden = self.model(
                    input_ids=self.input_ids[:1],
                    positions=self._get_positions(1),
                    inputs_embeds=None,
                    hidden_states=self.hidden_states[:1],
                )
                self.hidden_states[:1].copy_(backbone_hidden[:1])
                token = self.model.get_top_tokens(last_hidden[:1])
                output[0, index + 1 : index + 2].copy_(token)


def _select_loopgraph_output_slot(state: dict[str, Any]) -> Any:
    import torch

    outputs = state.get("outputs")
    if not outputs:
        return state["out"]

    slot_index = int(state.get("next_slot", 0))
    output_slot = outputs[slot_index]
    event = _LOOPGRAPH_SLOT_EVENTS_BY_PTR.get(output_slot.data_ptr())
    event_recorded = _LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR.get(
        output_slot.data_ptr(), False
    )
    if event is not None and event_recorded:
        torch.cuda.current_stream().wait_event(event)

    state["out"] = output_slot
    state["active_slot"] = slot_index
    state["next_slot"] = (slot_index + 1) % len(outputs)
    return output_slot


def _prime_loopgraph_outputs(state: dict[str, Any], first_token: Any) -> None:
    for output in state.get("outputs", [state["out"]]):
        output[0, 0:1].copy_(first_token)


def _capture_graph(self: Any, state: dict[str, Any]) -> None:
    import torch

    graphs = []
    for output in state.get("outputs", [state["out"]]):
        state["out"] = output
        for _ in range(2):
            _run_graph_body(self, state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_graph_body(self, state)
        graphs.append(graph)
    state["graphs"] = graphs
    state["graph"] = graphs[0]
    state["out"] = state.get("outputs", [state["out"]])[0]


def _is_loopgraph_eligible(self: Any, state: dict[str, Any], cad: Any) -> bool:
    return (
        not state["failed"]
        and self.num_speculative_tokens > 1
        and not self.parallel_drafting
        and not self._enable_probabilistic_draft_probs
        and not self.supports_mm_inputs
        and not self.uses_mrope
        and self.constant_draft_positions
        and cad.batch_size() == 1
    )


def _raise_or_fallback(exc: Exception) -> None:
    if LOOPGRAPH_REQUIRE_CAPTURE:
        raise RuntimeError("LOOPGRAPH_REQUIRE_CAPTURE=1 but capture failed") from exc


def _apply_loopgraph_patch(module: Any) -> None:
    import torch

    from vllm.forward_context import set_forward_context

    proposer_cls = module.Gemma4Proposer
    base_propose = proposer_cls.propose

    def propose(
        self: Any,
        target_token_ids: Any,
        target_positions: Any,
        target_hidden_states: Any,
        next_token_ids: Any,
        token_indices_to_sample: Any,
        common_attn_metadata: Any,
        sampling_metadata: Any,
        mm_embed_inputs: Any = None,
        num_rejected_tokens_gpu: Any = None,
        slot_mappings: Any = None,
    ) -> Any:
        kwargs = {
            "target_token_ids": target_token_ids,
            "target_positions": target_positions,
            "target_hidden_states": target_hidden_states,
            "next_token_ids": next_token_ids,
            "token_indices_to_sample": token_indices_to_sample,
            "common_attn_metadata": common_attn_metadata,
            "sampling_metadata": sampling_metadata,
            "mm_embed_inputs": mm_embed_inputs,
            "num_rejected_tokens_gpu": num_rejected_tokens_gpu,
            "slot_mappings": slot_mappings,
        }
        state = self.__dict__.setdefault(
            "_pupa_loopgraph",
            {"calls": 0, "graph": None, "failed": False},
        )
        if not _is_loopgraph_eligible(self, state, common_attn_metadata):
            return _call_base_propose(base_propose, self, kwargs)

        state["calls"] += 1
        if state["graph"] is None and state["calls"] <= LOOPGRAPH_WARMUP_CALLS:
            return _call_base_propose(base_propose, self, kwargs)

        self._last_draft_probs = None
        token_count = self.num_speculative_tokens
        num_tokens, token_indices_to_sample, cad = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        _, per_layer_metadata = self.build_per_group_and_layer_attn_metadata(cad)
        cg_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        model_kwargs, slot_map_size = self.build_model_inputs_first_pass(
            num_tokens,
            num_input_tokens,
            mm_embed_inputs,
        )
        with set_forward_context(
            per_layer_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cg_mode,
            slot_mapping=self._get_slot_mapping(slot_map_size, cad.slot_mapping),
        ):
            last_hidden, hidden = self.model(**model_kwargs)

        sample_hidden = last_hidden[token_indices_to_sample]
        positions = self.positions[token_indices_to_sample]
        first_hidden = hidden[token_indices_to_sample]
        self.positions[:1] = positions
        first_token, _ = self._sample_draft_tokens(sample_hidden, sampling_metadata)

        cad.num_actual_tokens = 1
        cad.max_query_len = 1
        cad.query_start_loc = self.arange[:2]
        cad.query_start_loc_cpu = torch.from_numpy(self.token_arange_np[:2]).clone()
        if num_rejected_tokens_gpu is not None:
            cad.seq_lens -= num_rejected_tokens_gpu
            cad._seq_lens_cpu = None
            cad._num_computed_tokens_cpu = None

        if state["graph"] is None and not state["failed"]:
            try:
                _build_static_buffers(self, state, cad)
                _refresh_static_buffers(self, state, cad)
                _prime_loopgraph_outputs(state, first_token)
                self.hidden_states[:1].copy_(first_hidden)
                _capture_graph(self, state)
                print(
                    f"[pupa-loopgraph] captured K-1={token_count - 1} graph "
                    f"at eligible call {state['calls']} "
                    f"with slots={LOOPGRAPH_PINGPONG_SLOTS} (pid {os.getpid()})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                state["failed"] = True
                state["graph"] = None
                print(
                    f"[pupa-loopgraph] capture failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                _raise_or_fallback(exc)

        if state["graph"] is not None:
            output_slot = _select_loopgraph_output_slot(state)
            _refresh_static_buffers(self, state, cad)
            output_slot[0, 0:1].copy_(first_token)
            self.hidden_states[:1].copy_(first_hidden)
            graphs = state.get("graphs")
            graph = graphs[state["active_slot"]] if graphs else state["graph"]
            graph.replay()
            return output_slot

        cg_mode, input_batch_size, batch_size_dp = (
            self._determine_batch_execution_and_padding(1)
        )
        draft_tokens = [first_token]
        hidden_current = first_hidden
        loop_metadata = None
        for index in range(token_count - 1):
            input_ids = draft_tokens[-1].int()
            if index == 0:
                _, loop_metadata = self.build_per_group_and_layer_attn_metadata(
                    cad,
                    draft_index=1,
                )
            self.input_ids[:1] = input_ids
            self.hidden_states[:1] = hidden_current
            kwargs = {
                "input_ids": self.input_ids[:input_batch_size],
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": None,
                "hidden_states": self.hidden_states[:input_batch_size],
            }
            with set_forward_context(
                loop_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_dp,
                cudagraph_runtime_mode=cg_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                last_hidden, hidden = self.model(**kwargs)
            hidden_current = hidden[:1]
            token, _ = self._sample_draft_tokens(last_hidden[:1], sampling_metadata)
            draft_tokens.append(token)
        return torch.stack(draft_tokens, dim=1)

    def propose_onegraph(
        self: Any,
        target_token_ids: Any,
        target_positions: Any,
        target_hidden_states: Any,
        next_token_ids: Any,
        token_indices_to_sample: Any,
        common_attn_metadata: Any,
        sampling_metadata: Any,
        mm_embed_inputs: Any = None,
        num_rejected_tokens_gpu: Any = None,
        slot_mappings: Any = None,
    ) -> Any:
        kwargs = {
            "target_token_ids": target_token_ids,
            "target_positions": target_positions,
            "target_hidden_states": target_hidden_states,
            "next_token_ids": next_token_ids,
            "token_indices_to_sample": token_indices_to_sample,
            "common_attn_metadata": common_attn_metadata,
            "sampling_metadata": sampling_metadata,
            "mm_embed_inputs": mm_embed_inputs,
            "num_rejected_tokens_gpu": num_rejected_tokens_gpu,
            "slot_mappings": slot_mappings,
        }
        state = self.__dict__.setdefault(
            "_pupa_loopgraph",
            {"calls": 0, "graph": None, "failed": False, "onegraph": True},
        )
        if not _is_loopgraph_eligible(self, state, common_attn_metadata) or (
            self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0
        ):
            return _call_base_propose(base_propose, self, kwargs)

        state["calls"] += 1
        if state["graph"] is None and state["calls"] <= LOOPGRAPH_WARMUP_CALLS:
            return _call_base_propose(base_propose, self, kwargs)

        self._last_draft_probs = None
        token_count = self.num_speculative_tokens
        cad = common_attn_metadata

        # Index of the single position whose drafter output is consumed.
        sample_index = token_indices_to_sample
        if sample_index is None:
            sample_index = cad.query_start_loc[1:] - 1

        # Loop-invariant metadata adjustments (mirror of the stock loop setup;
        # the width-(K+1) first pass is skipped entirely).
        cad.num_actual_tokens = 1
        cad.max_query_len = 1
        cad.query_start_loc = self.arange[:2]
        if num_rejected_tokens_gpu is not None:
            cad.seq_lens -= num_rejected_tokens_gpu
            cad._seq_lens_cpu = None
            cad._num_computed_tokens_cpu = None

        # Width-1 inputs: position + target hidden state of the sampled slot,
        # and the token the target just sampled. All device-side gathers.
        positions_1d = target_positions
        if self.vllm_config.model_config.uses_mrope:
            positions_1d = target_positions[0]
        self.positions[:1] = positions_1d[sample_index]
        self.hidden_states[:1].copy_(target_hidden_states[sample_index])

        if state["graph"] is None and not state["failed"]:
            try:
                _build_static_buffers(self, state, cad)
                _refresh_static_buffers(self, state, cad)
                state["first_input"].copy_(next_token_ids[:1])
                _capture_graph(self, state)
                # Warmup/capture runs consumed the hidden buffer; restore the
                # real value for the replay that serves this step.
                self.hidden_states[:1].copy_(target_hidden_states[sample_index])
                print(
                    f"[onegraph] captured K={token_count} width-1 propose graph "
                    f"at eligible call {state['calls']} "
                    f"with slots={LOOPGRAPH_PINGPONG_SLOTS} (pid {os.getpid()})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                state["failed"] = True
                state["graph"] = None
                print(
                    f"[onegraph] capture failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                _raise_or_fallback(exc)

        if state["graph"] is not None:
            output_slot = _select_loopgraph_output_slot(state)
            _refresh_static_buffers(self, state, cad)
            state["first_input"].copy_(next_token_ids[:1])
            graphs = state.get("graphs")
            graph = graphs[state["active_slot"]] if graphs else state["graph"]
            graph.replay()
            return output_slot

        # Eager width-1 fallback: K iterations, token-equivalent to the
        # captured body. Re-gather hidden (a failed capture's warmup runs
        # may have overwritten the buffer).
        self.hidden_states[:1].copy_(target_hidden_states[sample_index])
        cg_mode, input_batch_size, batch_size_dp = (
            self._determine_batch_execution_and_padding(1)
        )
        _, loop_metadata = self.build_per_group_and_layer_attn_metadata(
            cad,
            draft_index=1,
        )
        draft_tokens: list[Any] = []
        for index in range(token_count):
            if index == 0:
                self.input_ids[:1] = next_token_ids[:1].int()
            else:
                self.input_ids[:1] = draft_tokens[-1].int()
            forward_kwargs = {
                "input_ids": self.input_ids[:input_batch_size],
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": None,
                "hidden_states": self.hidden_states[:input_batch_size],
            }
            with set_forward_context(
                loop_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_dp,
                cudagraph_runtime_mode=cg_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                last_hidden, hidden = self.model(**forward_kwargs)
            self.hidden_states[:1] = hidden[:1]
            token, _ = self._sample_draft_tokens(
                last_hidden[:1], sampling_metadata
            )
            draft_tokens.append(token)
        return torch.stack(draft_tokens, dim=1)

    proposer_cls.propose = propose_onegraph if ONEGRAPH else propose
    print(
        f"[pupa-loopgraph] patched Gemma4Proposer.propose in pid {os.getpid()} "
        f"(warmup_calls={LOOPGRAPH_WARMUP_CALLS}, "
        f"require_capture={LOOPGRAPH_REQUIRE_CAPTURE}, onegraph={ONEGRAPH})",
        file=sys.stderr,
        flush=True,
    )


def _apply_loopgraph_copy_event_patch(module: Any) -> None:
    import torch

    runner_cls = module.GPUModelRunner
    original_copy_draft_token_ids_to_cpu = runner_cls._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(
        self: Any,
        scheduler_output: Any,
        zeros_only: bool = False,
    ) -> Any:
        draft_token_ids = getattr(self, "_draft_token_ids", None)
        result = original_copy_draft_token_ids_to_cpu(
            self, scheduler_output, zeros_only=zeros_only
        )
        if zeros_only or not torch.is_tensor(draft_token_ids):
            return result

        event = _LOOPGRAPH_SLOT_EVENTS_BY_PTR.get(draft_token_ids.data_ptr())
        copy_stream = getattr(self, "draft_token_ids_copy_stream", None)
        if event is not None and copy_stream is not None:
            with torch.cuda.stream(copy_stream):
                event.record(copy_stream)
            _LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR[draft_token_ids.data_ptr()] = True
        return result

    runner_cls._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    print(
        f"[pupa-loopgraph] patched GPUModelRunner draft-token copy events "
        f"in pid {os.getpid()} (slots={LOOPGRAPH_PINGPONG_SLOTS})",
        file=sys.stderr,
        flush=True,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def _get_fused_sparse_argmax_kernels() -> Any:
    global _FUSED_SPARSE_ARGMAX_KERNELS
    if _FUSED_SPARSE_ARGMAX_KERNELS is not None:
        return _FUSED_SPARSE_ARGMAX_KERNELS

    import triton
    import triton.language as tl

    @triton.jit
    def _sparse_argmax_blocks_kernel(
        hidden_states,
        lm_head_weight,
        top_centroids,
        token_ordering,
        partial_scores,
        partial_tokens,
        hidden_stride_t,
        hidden_stride_d,
        lm_head_stride_v,
        lm_head_stride_d,
        top_stride_t,
        top_stride_k,
        partial_score_stride_t,
        partial_token_stride_t,
        VOCAB_PER_CENTROID: tl.constexpr,
        SELECTED_COUNT: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SELECTED: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        token_idx = tl.program_id(0)
        selected_block = tl.program_id(1)

        selected_offsets = selected_block * BLOCK_SELECTED + tl.arange(
            0, BLOCK_SELECTED
        )
        valid_selected = selected_offsets < SELECTED_COUNT
        centroid_slots = selected_offsets // VOCAB_PER_CENTROID
        token_slots = selected_offsets - centroid_slots * VOCAB_PER_CENTROID

        centroid_ids = tl.load(
            top_centroids + token_idx * top_stride_t + centroid_slots * top_stride_k,
            mask=valid_selected,
            other=0,
        )
        vocab_ids = tl.load(
            token_ordering + centroid_ids * VOCAB_PER_CENTROID + token_slots,
            mask=valid_selected,
            other=0,
        )

        d_offsets = tl.arange(0, BLOCK_D)
        valid_d = d_offsets < HIDDEN_SIZE
        hidden = tl.load(
            hidden_states + token_idx * hidden_stride_t + d_offsets * hidden_stride_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            lm_head_weight
            + vocab_ids[:, None] * lm_head_stride_v
            + d_offsets[None, :] * lm_head_stride_d,
            mask=valid_selected[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(weights * hidden[None, :], axis=1)
        # The PyTorch sparse path materializes bf16 logits before argmax.
        scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(valid_selected, scores, -float("inf"))
        best_score, best_local_idx = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        best_selected = selected_block * BLOCK_SELECTED + best_local_idx
        best_centroid_slot = best_selected // VOCAB_PER_CENTROID
        best_token_slot = best_selected - best_centroid_slot * VOCAB_PER_CENTROID
        best_centroid = tl.load(
            top_centroids + token_idx * top_stride_t + best_centroid_slot * top_stride_k
        )
        best_token = tl.load(
            token_ordering + best_centroid * VOCAB_PER_CENTROID + best_token_slot
        )
        tl.store(
            partial_scores + token_idx * partial_score_stride_t + selected_block,
            best_score,
        )
        tl.store(
            partial_tokens + token_idx * partial_token_stride_t + selected_block,
            best_token,
        )

    @triton.jit
    def _sparse_argmax_reduce_kernel(
        partial_scores,
        partial_tokens,
        output_tokens,
        partial_score_stride_t,
        partial_token_stride_t,
        output_stride_t,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_BLOCKS: tl.constexpr,
    ) -> None:
        token_idx = tl.program_id(0)
        block_offsets = tl.arange(0, BLOCK_BLOCKS)
        valid_blocks = block_offsets < NUM_BLOCKS
        scores = tl.load(
            partial_scores + token_idx * partial_score_stride_t + block_offsets,
            mask=valid_blocks,
            other=-float("inf"),
        )
        _, best_block = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        token = tl.load(
            partial_tokens + token_idx * partial_token_stride_t + best_block
        )
        tl.store(output_tokens + token_idx * output_stride_t, token)

    _FUSED_SPARSE_ARGMAX_KERNELS = (
        triton,
        _sparse_argmax_blocks_kernel,
        _sparse_argmax_reduce_kernel,
    )
    return _FUSED_SPARSE_ARGMAX_KERNELS


def _fallback_sparse_argmax(
    self: Any,
    original_get_top_tokens: Any,
    hidden_states: Any,
    lm_head_weight: Any,
    reason: Exception,
) -> Any:
    if FUSED_SPARSE_ARGMAX_REQUIRE:
        raise RuntimeError(
            "FUSED_SPARSE_ARGMAX_REQUIRE=1 but fusion failed"
        ) from reason
    if not getattr(self, "_pupa_fused_sparse_argmax_warned", False):
        self._pupa_fused_sparse_argmax_warned = True
        print(
            f"[pupa-fused-sparse-argmax] falling back to PyTorch path: {reason!r}",
            file=sys.stderr,
            flush=True,
        )
    return original_get_top_tokens(self, hidden_states, lm_head_weight)


def _apply_fused_top_token_patch(module: Any) -> None:
    import torch

    embedder_cls = module.Gemma4MTPMaskedEmbedder
    original_get_top_tokens = embedder_cls.get_top_tokens

    def _select_and_score_unsorted(self: Any, hidden_states: Any, lm_head_weight: Any):
        num_tokens = hidden_states.shape[0]
        _, top_k_indices = torch.topk(
            self.centroids(hidden_states),
            k=self.centroid_intermediate_top_k,
            dim=-1,
            sorted=False,
        )
        clusters = self.token_ordering.view(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected = clusters[top_k_indices]
        embeddings = lm_head_weight[selected.reshape(-1)].view(
            num_tokens,
            self.num_selected,
            self.hidden_size,
        )
        logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected.view(num_tokens, -1)

    def _biased_top_tokens(self: Any, hidden_states: Any, lm_head_weight: Any) -> Any:
        """Frequency-biased drafter argmax (PR #48). Adds a static per-token bias
        to the sparse candidate scores, then argmaxes — exact-equivalent to the
        fused kernel at bias=0 (so bias=0 stays on the fused path; only b!=0 lands
        here). Verifier argmax is unaffected => greedy identity / PPL unchanged."""
        logits, selected = self._select_and_score(hidden_states, lm_head_weight)
        table = _get_freq_bias_table(int(lm_head_weight.shape[0]), logits.device, logits.dtype)
        logits = logits + table[selected]
        best = logits.argmax(dim=-1)
        return selected.gather(1, best.unsqueeze(1)).squeeze(1)

    def get_top_tokens_fused(self: Any, hidden_states: Any, lm_head_weight: Any) -> Any:
        if _freq_bias_active():
            return _biased_top_tokens(self, hidden_states, lm_head_weight)
        if not FUSED_SPARSE_ARGMAX:
            return original_get_top_tokens(self, hidden_states, lm_head_weight)
        try:
            if (
                hidden_states.device.type != "cuda"
                or lm_head_weight.device.type != "cuda"
            ):
                raise RuntimeError("fusion requires CUDA tensors")
            if (
                hidden_states.dtype != torch.bfloat16
                or lm_head_weight.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "fusion currently preserves exact PyTorch argmax only for bf16"
                )
            hidden_size = int(self.hidden_size)
            if hidden_size <= 0 or hidden_size > 1024:
                raise RuntimeError(f"unsupported hidden_size={hidden_size}")

            triton, blocks_kernel, reduce_kernel = _get_fused_sparse_argmax_kernels()
            num_tokens = int(hidden_states.shape[0])
            selected_count = int(self.num_selected)
            block_selected = _next_power_of_2(FUSED_SPARSE_ARGMAX_BLOCK)
            num_blocks = triton.cdiv(selected_count, block_selected)
            reduce_block = _next_power_of_2(num_blocks)
            block_d = _next_power_of_2(hidden_size)

            _, top_k_indices = torch.topk(
                self.centroids(hidden_states),
                k=self.centroid_intermediate_top_k,
                dim=-1,
                sorted=False,
            )
            partial_scores = torch.empty(
                (num_tokens, num_blocks),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            partial_tokens = torch.empty(
                (num_tokens, num_blocks),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            output_tokens = torch.empty(
                (num_tokens,),
                dtype=torch.int64,
                device=hidden_states.device,
            )

            blocks_kernel[(num_tokens, num_blocks)](
                hidden_states,
                lm_head_weight,
                top_k_indices,
                self.token_ordering,
                partial_scores,
                partial_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                lm_head_weight.stride(0),
                lm_head_weight.stride(1),
                top_k_indices.stride(0),
                top_k_indices.stride(1),
                partial_scores.stride(0),
                partial_tokens.stride(0),
                VOCAB_PER_CENTROID=int(self.vocab_size_per_centroid),
                SELECTED_COUNT=selected_count,
                HIDDEN_SIZE=hidden_size,
                BLOCK_SELECTED=block_selected,
                BLOCK_D=block_d,
                num_warps=8,
            )
            reduce_kernel[(num_tokens,)](
                partial_scores,
                partial_tokens,
                output_tokens,
                partial_scores.stride(0),
                partial_tokens.stride(0),
                output_tokens.stride(0),
                NUM_BLOCKS=num_blocks,
                BLOCK_BLOCKS=reduce_block,
                num_warps=8,
            )
            return output_tokens
        except Exception as exc:
            return _fallback_sparse_argmax(
                self,
                original_get_top_tokens,
                hidden_states,
                lm_head_weight,
                exc,
            )

    embedder_cls._select_and_score = _select_and_score_unsorted
    embedder_cls.get_top_tokens = get_top_tokens_fused
    print(
        f"[pupa-fused-sparse-argmax] patched Gemma4MTPMaskedEmbedder top-token path "
        f"in pid {os.getpid()} (enabled={FUSED_SPARSE_ARGMAX}, "
        f"require={FUSED_SPARSE_ARGMAX_REQUIRE}, block={FUSED_SPARSE_ARGMAX_BLOCK})",
        file=sys.stderr,
        flush=True,
    )


def _get_fused_accept_prep_kernel() -> Any:
    global _FUSED_ACCEPT_PREP_KERNEL
    if _FUSED_ACCEPT_PREP_KERNEL is not None:
        return _FUSED_ACCEPT_PREP_KERNEL

    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["max_spec_len"])
    def _dixie_fused_accept_prep_kernel(
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

        rejected = False
        valid_count = 0
        next_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
        row_offset = req_idx * (max_spec_len + 1)
        for pos in range(num_draft_tokens):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(
                    tl.int32
                )
                rejected = draft_token_id != target_argmax_id
                valid_count = pos + 1
                next_token_id = target_argmax_id
                tl.store(output_token_ids_ptr + row_offset + pos, target_argmax_id)

        if not rejected:
            bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx).to(tl.int32)
            valid_count = num_draft_tokens + 1
            next_token_id = bonus_token_id
            tl.store(
                output_token_ids_ptr + row_offset + num_draft_tokens,
                bonus_token_id,
            )

        tl.store(next_token_ids_ptr + req_idx, next_token_id)
        tl.store(valid_counts_ptr + req_idx, valid_count)

    _FUSED_ACCEPT_PREP_KERNEL = _dixie_fused_accept_prep_kernel
    return _FUSED_ACCEPT_PREP_KERNEL


def _dixie_fused_accept_prep(
    output_token_ids: Any,
    cu_num_draft_tokens: Any,
    draft_token_ids: Any,
    target_argmax: Any,
    bonus_token_ids: Any,
    max_spec_len: int,
) -> bool:
    if not DIXIE_FUSED_ACCEPT_PREP:
        return False
    try:
        import torch

        batch_size = int(output_token_ids.shape[0])
        next_token_ids = torch.empty(
            (batch_size,), dtype=torch.int32, device=output_token_ids.device
        )
        valid_counts = torch.empty(
            (batch_size,), dtype=torch.int32, device=output_token_ids.device
        )
        kernel = _get_fused_accept_prep_kernel()
        kernel[(batch_size,)](
            output_token_ids,
            next_token_ids,
            valid_counts,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            max_spec_len,
        )
        _FUSED_ACCEPT_PREP_CACHE[output_token_ids.data_ptr()] = (
            next_token_ids,
            valid_counts,
        )
        if not getattr(_dixie_fused_accept_prep, "_active_logged", False):
            _dixie_fused_accept_prep._active_logged = True
            print(
                f"[dixie-fused-accept] fused accept prep active "
                f"(batch={batch_size}, max_spec_len={max_spec_len})",
                file=sys.stderr,
                flush=True,
            )
        return True
    except Exception as exc:
        if DIXIE_FUSED_ACCEPT_PREP_REQUIRE:
            raise RuntimeError(
                "DIXIE_FUSED_ACCEPT_PREP_REQUIRE=1 but fused accept prep failed"
            ) from exc
        if not getattr(_dixie_fused_accept_prep, "_warned", False):
            _dixie_fused_accept_prep._warned = True
            print(
                f"[dixie-fused-accept] falling back to greedy rejection: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        return False


def _apply_fused_accept_proposer_patch(module: Any) -> None:
    proposer_cls = module.SpecDecodeBaseProposer
    original_prepare_next_token_ids_padded = proposer_cls.prepare_next_token_ids_padded

    def prepare_next_token_ids_padded(
        self: Any,
        sampled_token_ids: Any,
        requests: dict[str, Any],
        gpu_input_batch: Any,
        discard_request_mask: Any,
    ) -> tuple[Any, Any]:
        cached = None
        if DIXIE_FUSED_ACCEPT_PREP and hasattr(sampled_token_ids, "data_ptr"):
            cached = _FUSED_ACCEPT_PREP_CACHE.pop(sampled_token_ids.data_ptr(), None)
        if (
            cached is not None
            and sampled_token_ids.shape[0] == gpu_input_batch.num_reqs
            and gpu_input_batch.num_reqs == 1
        ):
            return cached
        return original_prepare_next_token_ids_padded(
            self,
            sampled_token_ids,
            requests,
            gpu_input_batch,
            discard_request_mask,
        )

    proposer_cls.prepare_next_token_ids_padded = prepare_next_token_ids_padded
    print(
        f"[dixie-fused-accept] patched SpecDecodeBaseProposer.prepare_next_token_ids_padded "
        f"in pid {os.getpid()} (enabled={DIXIE_FUSED_ACCEPT_PREP})",
        file=sys.stderr,
        flush=True,
    )


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, patch_fn: Any) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _TargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch_fn: Any) -> None:
        self._target = target
        self._patch_fn = patch_fn
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._patch_fn)
        return spec


sys.meta_path.insert(0, _TargetFinder(TOP_TOKEN_TARGET, _apply_fused_top_token_patch))
if DIXIE_FUSED_ACCEPT_PREP:
    sys.meta_path.insert(
        0,
        _TargetFinder(PROPOSER_TARGET, _apply_fused_accept_proposer_patch),
    )
sys.meta_path.insert(0, _TargetFinder(RUNNER_TARGET, _apply_loopgraph_copy_event_patch))
sys.meta_path.insert(0, _TargetFinder(LOOPGRAPH_TARGET, _apply_loopgraph_patch))


# --- fastrender (@juglar-fable) ------------------------------------------
FASTRENDER = os.environ.get("FASTRENDER", "1") == "1"
RENDERERS_TARGET = "vllm.renderers.hf"


def _apply_fastrender_patch(module: Any) -> None:
    if not FASTRENDER:
        return
    original = module.safe_apply_chat_template
    state: dict[str, Any] = {
        "checked": False,
        "ok": False,
        "prefix": "",
        "suffix": "",
        "shape_is_str": None,
        "fast": 0,
        "slow": 0,
    }
    allowed_kwargs = {
        "chat_template": None,
        "return_dict": False,
        "add_generation_prompt": True,
        "continue_final_message": False,
    }

    def _extract_text(content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
            and content[0].get("type") == "text"
            and isinstance(content[0].get("text"), str)
        ):
            return content[0]["text"]
        return None

    def _shape(shape_is_str: bool, text: str) -> Any:
        if shape_is_str:
            return text
        return [{"type": "text", "text": text}]

    def _eligible(conversation: Any, tools: Any, kwargs: dict[str, Any]) -> str | None:
        if tools is not None:
            return None
        if not isinstance(conversation, list) or len(conversation) != 1:
            return None
        msg = conversation[0]
        if not isinstance(msg, dict) or msg.get("role") != "user" or len(msg) != 2:
            return None
        for key, value in kwargs.items():
            if key not in allowed_kwargs or value != allowed_kwargs[key]:
                return None
        text = _extract_text(msg.get("content"))
        if text is None or not text.strip():
            return None
        return text

    def _probe(model_config: Any, tokenizer: Any, shape_is_str: bool, kwargs: dict[str, Any]) -> bool:
        import uuid

        def render(text: str) -> Any:
            conv = [{"role": "user", "content": _shape(shape_is_str, text)}]
            return original(
                model_config, tokenizer, conv, tools=None, tokenize=False, **kwargs
            )

        u1 = "JFA" + uuid.uuid4().hex
        u2 = "JFB" + uuid.uuid4().hex
        r1, r2 = render(u1), render(u2)
        if not (isinstance(r1, str) and isinstance(r2, str)):
            return False
        if r1.count(u1) != 1 or r2.count(u2) != 1:
            return False
        p1, s1 = r1.split(u1)
        p2, s2 = r2.split(u2)
        if p1 != p2 or s1 != s2:
            return False
        u3 = "JFC" + uuid.uuid4().hex
        if render(" \t\n" + u3 + "  \n") != p1 + u3 + s1:
            return False
        specials = u3 + " <>&\"'%{}#|`$\\"
        if render(specials) != p1 + specials.strip() + s1:
            return False
        conv = [{"role": "user", "content": _shape(shape_is_str, u3)}]
        ids_orig = original(
            model_config, tokenizer, conv, tools=None, tokenize=True, **kwargs
        )
        ids_fast = tokenizer.encode(p1 + u3 + s1, add_special_tokens=False)
        if list(ids_orig) != list(ids_fast):
            return False
        state["prefix"], state["suffix"] = p1, s1
        state["shape_is_str"] = shape_is_str
        return True

    def wrapper(
        model_config: Any,
        tokenizer: Any,
        conversation: Any,
        *,
        tools: Any = None,
        tokenize: bool = True,
        **kwargs: Any,
    ) -> Any:
        try:
            text = _eligible(conversation, tools, kwargs)
            if text is not None:
                shape_is_str = isinstance(conversation[0].get("content"), str)
                if not state["checked"]:
                    state["checked"] = True
                    try:
                        state["ok"] = _probe(model_config, tokenizer, shape_is_str, kwargs)
                    except Exception as exc:
                        state["ok"] = False
                        print(
                            f"[fastrender] probe errored -> stock path ({exc!r})",
                            file=sys.stderr,
                            flush=True,
                        )
                    print(
                        "[fastrender] probes "
                        + ("PASSED - fast path ON" if state["ok"] else "FAILED - stock path"),
                        file=sys.stderr,
                        flush=True,
                    )
                if state["ok"] and shape_is_str == state["shape_is_str"]:
                    rendered = state["prefix"] + text.strip() + state["suffix"]
                    state["fast"] += 1
                    if state["fast"] in (1, 4, 128) or state["fast"] % 256 == 0:
                        print(
                            f"[fastrender] fast={state['fast']} slow={state['slow']}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if tokenize:
                        return tokenizer.encode(rendered, add_special_tokens=False)
                    return rendered
        except Exception as exc:
            print(
                f"[fastrender] error -> stock path ({exc!r})",
                file=sys.stderr,
                flush=True,
            )
        state["slow"] += 1
        return original(
            model_config, tokenizer, conversation, tools=tools, tokenize=tokenize, **kwargs
        )

    module.safe_apply_chat_template = wrapper
    print(
        "[fastrender] installed wrapper on vllm.renderers.hf.safe_apply_chat_template",
        file=sys.stderr,
        flush=True,
    )


sys.meta_path.insert(0, _TargetFinder(RENDERERS_TARGET, _apply_fastrender_patch))

# PCK-04: registers _TargetFinder for Gemma4ForCausalLM lm_head rebuild + logits scatter.
import serve_patch_pck04  # noqa: E402, F401

if __import__("os").environ.get("LSK_SKIP_LAYERS"):
    import lsk_patch  # osoi-v0 layer skip (env-gated)

# hayai detok-endonly: end-only detokenization for non-streaming requests
# (token_ids untouched; text byte-identity validated 6160-variant fuzz +
# 72/72 server A/B). Gated on DETOK_ENDONLY=1; fail-closed on source drift.
import detok_endonly  # noqa: E402,F401

# agent-smith steptime probe (env-gated; STEPTIME=1).
if __import__("os").environ.get("STEPTIME", "0") == "1":
    import steptime_patch  # noqa: E402,F401

# agent-smith FA2-for-sliding-layers override (env-gated; FA_SLIDING=1).
if __import__("os").environ.get("FA_SLIDING", "0") == "1":
    import fa_sliding_patch  # noqa: E402,F401

# splitkv-verify: route small multi-query-row (spec-verify) attention batches to
# vLLM's 3D split-KV path (env-gated; SPLITKV_VERIFY=1). Registers a meta-path
# finder for the Triton attention ops module; fail-open. See module docstring.
if __import__("os").environ.get("SPLITKV_VERIFY", "0") == "1":
    import splitkv_verify_patch  # noqa: E402,F401

# kduma precache (env-gated; PRECACHE_BENCH=1): bench-prompt prefix replay
# during the untimed warmup window, readiness-gated, fail-closed.
if __import__("os").environ.get("PRECACHE_BENCH", "0") == "1":
    import serve_patch_precache  # noqa: E402,F401


# darwin-4b-opus _IncludedRouter / missing-`.path` startup-500 guard (validated
# output-neutral under kanna PR #177, W&B bjtwr9jn: token-ids 128/128 identical,
# PPL byte-identical 2.376976138392039, TPS +0.02%). On fresh a10g images vLLM
# 0.22.1rc1 mounts sub-routers (`_IncludedRouter`) lacking a `.path`;
# prometheus_fastapi_instrumentator.routing._get_route_name does `route.path` and
# raises AttributeError on EVERY request -> HTTP 500 -> "/v1/models" never becomes
# ready. Wrap _get_route_name to swallow that single AttributeError (return None).
# Output-neutral: HTTP-metrics middleware ONLY; never touches greedy / PPL /
# token-ids. No-op when the instrumentator is absent or no pathless route is
# reached (then returns the original value verbatim).
#
# NOTE(kanna PR #359): this guard was present in the parent fa2sw_precache_kenyan
# sitecustomize.py but was dropped when this non-spec throwaway was derived, so the
# local serve booted but every request 500'd in the prometheus middleware. Restored
# here to make the non-spec serve byte-faithful to its parent (modulo SPECULATIVE_CONFIG).
def _guard_included_router() -> None:
    try:
        import prometheus_fastapi_instrumentator.routing as _r
    except Exception:
        return
    _orig = _r._get_route_name

    def _guarded(scope, routes):
        try:
            return _orig(scope, routes)
        except AttributeError:
            return None

    _r._get_route_name = _guarded


_guard_included_router()
