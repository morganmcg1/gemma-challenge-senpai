"""DETOK_ENDONLY: end-only detokenization for non-streaming requests.

Finding: vLLM v1 runs per-step incremental detokenization inside the
output-processor loop (vllm/v1/engine/detokenizer.py) for EVERY decode step,
and this work is not overlapped by async scheduling. The challenge benchmark
client is non-streaming (sglang bench_serving --disable-stream reads only
choices[0].message.content + usage.completion_tokens; the decode-capture pass
reads choices[0].text + choices[0].token_ids). For non-streaming requests
(RequestOutputKind.FINAL_ONLY) the per-step text is consumed exactly once, at
completion, so per-step incremental detok is pure per-step latency.

This patch replaces the per-request IncrementalDetokenizer with an end-only
detokenizer for ELIGIBLE requests only:
  * output_kind == FINAL_ONLY (non-streaming),
  * no stop strings (stop-string matching needs per-step text),
  * detokenize=True, fast (rust-backed) tokenizer, prompt token ids present,
  * skip_special_tokens or spaces_between_special_tokens (the added-token
    spacing branch of FastIncrementalDetokenizer needs per-token emission).
Everything else gets the stock detokenizer, untouched.

The end-only detokenizer buffers token ids during update() (no stream.step,
no string concat, no stop scan) and produces the final text with ONE batched
rust decode at completion:

    text = decode(ctx + output_ids)[len(decode(ctx)):]   ctx = prompt tail

byte-identity contract (vs the stock FastIncrementalDetokenizer path):
  * the fast path is only used when its separability precondition provably
    holds: decode(ctx+out) startswith decode(ctx), no U+FFFD at the ctx
    boundary, and no trailing U+FFFD (the stock stream withholds trailing
    incomplete UTF-8 sequences; batch decode would render replacement chars);
  * otherwise the request falls back to an exact replay through the stock
    FastIncrementalDetokenizer class itself (byte-identical by construction,
    just executed once at completion instead of per step);
  * the stop-terminated trim (exclude last token from text but keep it in
    token_ids unless include_stop_str_in_output) is replicated exactly;
  * token_ids are the buffered engine ids - identical by construction.

Env:
  DETOK_ENDONLY=1         enable (default off; module is a no-op otherwise)
  DETOK_ENDONLY_SHADOW=1  validation mode: serve the STOCK text but also run
                          the end-only path and log byte-compare per request
  DETOK_ENDONLY_CTX=8     prompt-tail context tokens for the batched decode

Fail-closed: when enabled, the on-disk source of the two files whose behavior
this patch assumes (detokenizer.py, output_processor.py) is verified against
exact anchors (each must appear exactly once). Any drift => RuntimeError at
module load => the server refuses to boot. No silent no-op, no silent drift.

Applied via meta-path hook on vllm.v1.engine.detokenizer module load (same
mechanism as lsk_patch.py / the loopgraph patches) so it works regardless of
which process imports it; detok runs in the API frontend (AsyncLLM) process.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import pathlib
import sys
from typing import Any

DETOK_ENDONLY = os.environ.get("DETOK_ENDONLY", "0") == "1"
DETOK_ENDONLY_SHADOW = os.environ.get("DETOK_ENDONLY_SHADOW", "0") == "1"
_CTX_TOKENS = max(1, int(os.environ.get("DETOK_ENDONLY_CTX", "8")))
_TARGET = "vllm.v1.engine.detokenizer"

_STATS = {
    "endonly": 0,
    "stock": 0,
    "fast": 0,
    "replay": 0,
    "shadow_match": 0,
    "shadow_mismatch": 0,
}

# --- fail-closed source anchors (verbatim from the installed build) --------
# vllm 0.22.1rc1.dev307+g3e8afdf78

# detokenizer.py: construction dispatch we override (fast-tokenizer branch).
_ANCHOR_DET_DISPATCH = """\
        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)
"""

# detokenizer.py: stop-terminated trim semantics we replicate.
_ANCHOR_DET_STOP_TRIM = """\
        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None
"""

# detokenizer.py: stream is primed with the full prompt (context-sensitive
# first-token rendering) - the property our ctx-subtraction must reproduce.
_ANCHOR_DET_PRIME = """\
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
"""

# detokenizer.py: effective spaces_between_special_tokens (eligibility gate).
_ANCHOR_DET_SPACES = """\
        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )
"""

# output_processor.py: FINAL_ONLY requests never consume text pre-finish.
_ANCHOR_OP_FINAL_ONLY = """\
        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None
"""

# output_processor.py: the per-step update call we make O(1).
_ANCHOR_OP_UPDATE = """\
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
"""

# output_processor.py: final text/token_ids assembly we feed.
_ANCHOR_OP_ASSEMBLE = """\
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids
"""

_DETOKENIZER_ANCHORS = (
    ("from_new_request fast dispatch", _ANCHOR_DET_DISPATCH),
    ("stop-terminated trim", _ANCHOR_DET_STOP_TRIM),
    ("DecodeStream prompt priming", _ANCHOR_DET_PRIME),
    ("spaces_between_special_tokens", _ANCHOR_DET_SPACES),
)

_OUTPUT_PROCESSOR_ANCHORS = (
    ("FINAL_ONLY early return", _ANCHOR_OP_FINAL_ONLY),
    ("per-step detokenizer.update call", _ANCHOR_OP_UPDATE),
    ("final text assembly", _ANCHOR_OP_ASSEMBLE),
)


def _verify_required(source: str, anchor: str, label: str, path: Any) -> None:
    """Fail on source drift: each behavioral anchor must appear exactly once."""
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"[detok-endonly] anchor '{label}' count is {count} (expected 1) "
            f"in {path}; vLLM source drifted - refusing to run (fail-closed). "
            "Unset DETOK_ENDONLY or re-validate the patch against this build."
        )


def _verify_sources(module: Any) -> None:
    det_path = pathlib.Path(module.__file__)
    op_path = det_path.with_name("output_processor.py")
    det_src = det_path.read_text(encoding="utf-8")
    op_src = op_path.read_text(encoding="utf-8")
    for label, anchor in _DETOKENIZER_ANCHORS:
        _verify_required(det_src, anchor, label, det_path)
    for label, anchor in _OUTPUT_PROCESSOR_ANCHORS:
        _verify_required(op_src, anchor, label, op_path)


def _maybe_log_stats() -> None:
    done = _STATS["fast"] + _STATS["replay"]
    if done in (1, 16, 64) or done % 256 == 0:
        print(
            f"[detok-endonly] requests endonly={_STATS['endonly']} "
            f"stock={_STATS['stock']} final_fast={_STATS['fast']} "
            f"final_replay={_STATS['replay']} (pid {os.getpid()})",
            file=sys.stderr,
            flush=True,
        )


def build_classes(module: Any):
    """Build (eligibility fn, end-only class, shadow class) bound to the
    loaded vllm.v1.engine.detokenizer module. Exposed for offline unit tests."""
    from transformers import PreTrainedTokenizerFast

    from vllm.sampling_params import RequestOutputKind

    base_cls = module.IncrementalDetokenizer
    fast_cls = module.FastIncrementalDetokenizer

    def eligible(tokenizer: Any, request: Any) -> bool:
        params = request.sampling_params
        if params is None or tokenizer is None:
            return False
        if not module.USE_FAST_DETOKENIZER:
            return False
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            return False
        if params.output_kind != RequestOutputKind.FINAL_ONLY:
            return False  # streaming consumes per-step text
        if params.stop:
            return False  # stop strings require per-step text scanning
        if not getattr(params, "detokenize", True):
            return False
        if not request.prompt_token_ids:
            return False  # need prompt context; excludes prompt-embeds reqs
        if not (params.skip_special_tokens or params.spaces_between_special_tokens):
            return False  # added-token spacing branch needs per-token emission
        return True

    class EndOnlyDetokenizer(base_cls):
        """Buffer ids per step; one batched decode at completion.

        Byte-identity vs stock FastIncrementalDetokenizer: fast batched path
        only when provably separable, else exact replay through the stock
        class (same code, same token sequence => same bytes).
        """

        def __init__(self, tokenizer: Any, request: Any):
            super().__init__()
            self._tokenizer = tokenizer
            self._request = request
            self._params = request.sampling_params
            self._stop_terminated = False
            self._final_text: str | None = None

        def update(self, new_token_ids: list[int], stop_terminated: bool):
            # Mirrors BaseIncrementalDetokenizer.update() observable state:
            # token_ids gets ALL ids (including a skipped stop token); empty
            # updates are no-ops (incl. their stop_terminated flag).
            if new_token_ids:
                if stop_terminated:
                    self._stop_terminated = True
                self.token_ids.extend(new_token_ids)
            return None  # no stop strings by eligibility => never a stop match

        def get_next_output_text(self, finished: bool, delta: bool) -> str:
            if not finished:
                # FINAL_ONLY: make_request_output() early-returns before text
                # is consumed for unfinished requests (anchor-verified).
                return ""
            if self._final_text is None:
                text = self._fast_final_text()
                if text is None:
                    _STATS["replay"] += 1
                    text = self._replay_final_text()
                else:
                    _STATS["fast"] += 1
                self._final_text = text
                _maybe_log_stats()
            return self._final_text

        def _text_token_ids(self) -> list[int]:
            # Replicate the stop-terminated trim: the last token of a
            # stop-terminated request is excluded from TEXT (kept in
            # token_ids) unless include_stop_str_in_output.
            if (
                self._stop_terminated
                and not self._params.include_stop_str_in_output
                and self.token_ids
            ):
                return self.token_ids[:-1]
            return self.token_ids

        def _fast_final_text(self) -> str | None:
            try:
                ids = self._text_token_ids()
                if not ids:
                    return ""
                rust = self._tokenizer._tokenizer  # underlying rust Tokenizer
                skip = self._params.skip_special_tokens
                ctx = list(self._request.prompt_token_ids[-_CTX_TOKENS:])
                prefix = rust.decode(ctx, skip_special_tokens=skip)
                full = rust.decode(ctx + list(ids), skip_special_tokens=skip)
                if "�" in prefix or not full.startswith(prefix):
                    # ctx/output byte-fallback fusion: not safely separable.
                    return None
                text = full[len(prefix) :]
                if "�" in text:
                    # Invalid/incomplete UTF-8 anywhere: the stock stream's
                    # incremental prefix-diffing renders these differently
                    # (it withholds trailing incomplete sequences, and its
                    # invalid-prefix recovery can emit different bytes than
                    # a batch decode). Defer to exact replay.
                    return None
                return text
            except Exception:
                return None

        def _replay_final_text(self) -> str:
            # Exact replay through the stock class: DecodeStream primed with
            # the full prompt, per-token stepping, identical trim semantics.
            # Emission depends only on the token sequence (the stock update
            # loops per token), so one-chunk feeding is chunking-equivalent.
            det = fast_cls(self._tokenizer, self._request)
            det.update(list(self.token_ids), self._stop_terminated)
            return det.get_next_output_text(True, False)

    class ShadowDetokenizer(fast_cls):
        """Validation-only: stock path serves the response; the end-only path
        runs alongside and every finished request is byte-compared."""

        def __init__(self, tokenizer: Any, request: Any):
            super().__init__(tokenizer, request)
            self._twin = EndOnlyDetokenizer(tokenizer, request)
            self._compared = False

        def update(self, new_token_ids: list[int], stop_terminated: bool):
            self._twin.update(list(new_token_ids), stop_terminated)
            return super().update(new_token_ids, stop_terminated)

        def get_next_output_text(self, finished: bool, delta: bool) -> str:
            stock_text = super().get_next_output_text(finished, delta)
            if finished and not self._compared:
                self._compared = True
                try:
                    endonly_text = self._twin.get_next_output_text(True, False)
                    ids_match = list(self._twin.output_token_ids) == list(
                        self.output_token_ids
                    )
                    text_match = endonly_text.encode("utf-8") == stock_text.encode(
                        "utf-8"
                    )
                    if text_match and ids_match:
                        _STATS["shadow_match"] += 1
                    else:
                        _STATS["shadow_mismatch"] += 1
                        print(
                            "[detok-endonly][SHADOW MISMATCH] "
                            f"req={getattr(self, 'request_id', '?')} "
                            f"ids_match={ids_match}\n"
                            f"  stock  ={stock_text!r}\n"
                            f"  endonly={endonly_text!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    print(
                        f"[detok-endonly][shadow] match={_STATS['shadow_match']} "
                        f"mismatch={_STATS['shadow_mismatch']} "
                        f"fast={_STATS['fast']} replay={_STATS['replay']}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:  # never break serving in shadow mode
                    print(
                        f"[detok-endonly][shadow] compare errored: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
            return stock_text

    return eligible, EndOnlyDetokenizer, ShadowDetokenizer


def _apply(module: Any) -> None:
    _verify_sources(module)
    eligible, endonly_cls, shadow_cls = build_classes(module)

    base_cls = module.IncrementalDetokenizer
    stock_from_new_request = base_cls.from_new_request.__func__

    def _from_new_request(cls: Any, tokenizer: Any, request: Any):
        try:
            is_eligible = eligible(tokenizer, request)
        except Exception:
            is_eligible = False
        if is_eligible:
            _STATS["endonly"] += 1
            if DETOK_ENDONLY_SHADOW:
                return shadow_cls(tokenizer, request)
            return endonly_cls(tokenizer, request)
        _STATS["stock"] += 1
        return stock_from_new_request(cls, tokenizer, request)

    base_cls.from_new_request = classmethod(_from_new_request)
    print(
        f"[detok-endonly] patched IncrementalDetokenizer.from_new_request "
        f"(shadow={DETOK_ENDONLY_SHADOW}, ctx={_CTX_TOKENS}, "
        f"pid {os.getpid()}); anchors verified fail-closed.",
        file=sys.stderr,
        flush=True,
    )


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _apply(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if DETOK_ENDONLY:
    if _TARGET in sys.modules:
        _apply(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
