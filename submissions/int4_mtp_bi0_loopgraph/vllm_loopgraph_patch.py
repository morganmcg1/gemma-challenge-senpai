"""Loopgraph (onegraph) CUDA-graph replay for the Gemma4 MTP drafter.

Ported from ``fa2sw_precache_kenyan/sitecustomize.py`` (the loopgraph subsystem
only — the precache / fastrender / pck04 / detok / fa_sliding / splitkv / freq-bias
/ dixie-fused-accept patches are intentionally NOT brought across). This module is
layered on top of bi0's existing attn-group + force-2D patches and runs on STOCK
``vllm==0.22.0`` (bi0's pinned dependency), so the only changed variable vs the bi0
control is the loopgraph drafter dispatch.

Why this is greedy-identity-safe (the whole point of the experiment): the Gemma4
MTP drafter is Q-only and KV-shared — it never writes KV and has no cross-position
dependencies, so the padded width-(K+1) first pass only ever contributes the single
position selected by ``token_indices_to_sample``. Width-1 is exact. With ONEGRAPH=1
the whole ``propose()`` becomes one CUDA-graph replay of K width-1 iterations. The
patch only changes HOW the K-step draft loop is dispatched; it never changes the
DRAFTER's emitted token relative to the eager width-1 path, and crucially the
drafter only proposes — the verifier still emits the target argmax at T=0, so the
served greedy token sequence is identical regardless of draft dispatch.

Two appliers, wired from ``sitecustomize.py`` on first import of each target:

* ``apply_proposer`` on ``vllm.v1.spec_decode.gemma4`` — replaces
  ``Gemma4Proposer.propose`` with the onegraph (or stock-loopgraph) capture path.
* ``apply_copy_event`` on ``vllm.v1.worker.gpu_model_runner`` — records a CUDA
  event after the async draft-token D2H copy so the ping-pong output slot is not
  overwritten by the next replay while its copy is still in flight (a no-op for
  correctness with a single slot; matters when LOOPGRAPH_PINGPONG_SLOTS > 1).
"""

from __future__ import annotations

import os
import sys
from copy import copy
from typing import Any


LOOPGRAPH_WARMUP_CALLS = int(os.environ.get("LOOPGRAPH_WARMUP_CALLS", "48"))
LOOPGRAPH_REQUIRE_CAPTURE = os.environ.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1"
LOOPGRAPH_PINGPONG_SLOTS = max(1, int(os.environ.get("LOOPGRAPH_PINGPONG_SLOTS", "1")))
# onegraph (@blake-fable5-1): iteration 0 consumes next_token_ids + gathered target
# hidden/position; iterations 1..K-1 are the stock loopgraph body. Drafter-only =>
# cannot change emitted tokens.
ONEGRAPH = os.environ.get("ONEGRAPH", "1") == "1"

_LOOPGRAPH_SLOT_EVENTS_BY_PTR: dict[int, Any] = {}
_LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR: dict[int, bool] = {}


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


def apply_proposer(module: Any) -> None:
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
                    f"[bi0-loopgraph] captured K-1={token_count - 1} graph "
                    f"at eligible call {state['calls']} "
                    f"with slots={LOOPGRAPH_PINGPONG_SLOTS} (pid {os.getpid()})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                state["failed"] = True
                state["graph"] = None
                print(
                    f"[bi0-loopgraph] capture failed: {exc!r}",
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
                    f"[bi0-onegraph] captured K={token_count} width-1 propose graph "
                    f"at eligible call {state['calls']} "
                    f"with slots={LOOPGRAPH_PINGPONG_SLOTS} (pid {os.getpid()})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                state["failed"] = True
                state["graph"] = None
                print(
                    f"[bi0-onegraph] capture failed: {exc!r}",
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
        f"[bi0-loopgraph] patched Gemma4Proposer.propose in pid {os.getpid()} "
        f"(warmup_calls={LOOPGRAPH_WARMUP_CALLS}, "
        f"require_capture={LOOPGRAPH_REQUIRE_CAPTURE}, onegraph={ONEGRAPH})",
        file=sys.stderr,
        flush=True,
    )


def apply_copy_event(module: Any) -> None:
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
        f"[bi0-loopgraph] patched GPUModelRunner draft-token copy events "
        f"in pid {os.getpid()} (slots={LOOPGRAPH_PINGPONG_SLOTS})",
        file=sys.stderr,
        flush=True,
    )
