#!/usr/bin/env python3
"""PR #799 — Tree/EAGLE multi-candidate drafter feasibility probe.

Static, CPU-only dry-run against the pinned vLLM wheel
(0.22.1rc1.dev307+g3e8afdf78). Answers: can this stack express a
tree / multi-candidate / EAGLE-style drafter for gemma4_mtp, or is the
speculative path single-linear-chain only?

No model download, no GPU, no serve, no benchmark. Pure config-surface
introspection + a rejection probe that shows *where* a tree spec is refused.
Run under the pinned wheel's python:

    /tmp/senpai-venvs/5f4c623f772358a2/bin/python research/tree_eagle_feasibility_799/static_dryrun.py
"""

from __future__ import annotations

import inspect
import json
import typing

RESULT: dict[str, object] = {}


def section(title: str) -> None:
    print(f"\n===== {title} =====")


# 1) Wheel identity ----------------------------------------------------------
import vllm  # noqa: E402

section("wheel identity")
print("vllm.__version__ =", vllm.__version__)
RESULT["vllm_version"] = vllm.__version__

# 2) SpeculativeMethod enumeration ------------------------------------------
from vllm.config.speculative import (  # noqa: E402
    SpeculativeConfig,
    SpeculativeMethod,
    MTPModelTypes,
    EagleModelTypes,
)

section("registered speculative methods")
methods = list(typing.get_args(SpeculativeMethod))
# get_args keeps nested Literals as objects; flatten them.
flat_methods: list[str] = []
for m in methods:
    if typing.get_args(m):
        flat_methods.extend(str(x) for x in typing.get_args(m))
    else:
        flat_methods.append(str(m))
print("SpeculativeMethod members:", sorted(set(flat_methods)))
print("MTPModelTypes:", list(typing.get_args(MTPModelTypes)))
RESULT["speculative_methods"] = sorted(set(flat_methods))
tree_methods = [m for m in flat_methods if "tree" in m.lower()]
print("methods whose name contains 'tree':", tree_methods)
RESULT["tree_named_methods"] = tree_methods

# 3) SpeculativeConfig field surface ----------------------------------------
section("SpeculativeConfig fields (topology surface)")
import dataclasses  # noqa: E402

field_report = {}
if dataclasses.is_dataclass(SpeculativeConfig):
    for f in dataclasses.fields(SpeculativeConfig):
        field_report[f.name] = str(f.type)
else:  # pydantic fallback
    for name, info in SpeculativeConfig.model_fields.items():
        field_report[name] = str(info.annotation)
for name in sorted(field_report):
    print(f"  {name}: {field_report[name]}")
RESULT["num_speculative_tokens_type"] = field_report.get("num_speculative_tokens")

topology_terms = ("tree", "branch", "choices", "candidate", "topology", "width", "fork")
topology_fields = {
    n: t for n, t in field_report.items()
    if any(term in n.lower() for term in topology_terms)
}
print("\nfields matching tree/branch/candidate/topology terms:")
print(" ", topology_fields or "{}  <-- NONE")
RESULT["topology_fields"] = topology_fields
# suffix_decoding_max_tree_depth is a *retrieval* suffix-tree depth cap for the
# ngram/suffix method (a prefix-match data structure), NOT a candidate-tree
# topology for the speculative verify path. Exclude it from the verdict.
candidate_tree_fields = {
    n: t for n, t in topology_fields.items() if not n.startswith("suffix_decoding")
}
print("  candidate-tree (verify-path) topology fields:",
      candidate_tree_fields or "{}  <-- NONE")
RESULT["candidate_tree_topology_fields"] = candidate_tree_fields

# 4) Rejection probe: try to express a tree topology ------------------------
# A tree drafter would need num_speculative_tokens to carry a per-level
# branch structure (a list/nested topology). Show the scalar int field
# refuses it -- this is "where it rejects".
section("rejection probe: num_speculative_tokens = [1, 2, 2] (a tree topology)")
try:
    SpeculativeConfig(num_speculative_tokens=[1, 2, 2], method="mtp")
    print("UNEXPECTED: tree-shaped num_speculative_tokens accepted")
    RESULT["tree_topology_rejected"] = False
except Exception as e:  # noqa: BLE001
    print(f"REJECTED by {type(e).__name__}:")
    msg = str(e).splitlines()
    for line in msg[:8]:
        print("   ", line)
    RESULT["tree_topology_rejected"] = True
    RESULT["tree_topology_reject_exc"] = type(e).__name__

# Also probe an unknown 'tree_choices' kwarg (EAGLE-style tree spec).
section("rejection probe: tree_choices=[[0],[0,0]] kwarg")
try:
    SpeculativeConfig(num_speculative_tokens=6, method="mtp",
                      tree_choices=[[0], [0, 0]])
    print("UNEXPECTED: tree_choices kwarg accepted")
    RESULT["tree_choices_rejected"] = False
except Exception as e:  # noqa: BLE001
    print(f"REJECTED by {type(e).__name__}:")
    for line in str(e).splitlines()[:6]:
        print("   ", line)
    RESULT["tree_choices_rejected"] = True
    RESULT["tree_choices_reject_exc"] = type(e).__name__

# 5) Proposer for gemma4_mtp is the linear base proposer --------------------
section("gemma4_mtp proposer class + chain semantics")
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer  # noqa: E402
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer  # noqa: E402
from vllm.v1.spec_decode.eagle import EagleProposer  # noqa: E402

print("Gemma4Proposer base classes:",
      [c.__name__ for c in Gemma4Proposer.__mro__])
print("EagleProposer base classes:",
      [c.__name__ for c in EagleProposer.__mro__])
RESULT["gemma4_proposer_base"] = SpecDecodeBaseProposer.__name__ in [
    c.__name__ for c in Gemma4Proposer.__mro__
]
RESULT["eagle_proposer_base"] = SpecDecodeBaseProposer.__name__ in [
    c.__name__ for c in EagleProposer.__mro__
]

# The base propose() returns [batch_size, num_speculative_tokens] -- rank 2,
# a single linear chain per request (no candidate-rank dimension).
src = inspect.getsource(SpecDecodeBaseProposer.propose)
returns_2d = "draft_token_ids.view(-1, self.num_speculative_tokens)" in src
stacks_chain = "torch.stack(draft_token_ids_list, dim=1)" in src
print("base propose() returns [-1, num_speculative_tokens] (rank-2 chain):",
      returns_2d)
print("base propose() builds chain via torch.stack(list, dim=1):", stacks_chain)
RESULT["propose_returns_linear_chain"] = bool(returns_2d and stacks_chain)

# 6) Verify kernel contract: 1D flattened chain -----------------------------
section("verify kernel contract (rejection_sample)")
from vllm.v1.sample.rejection_sampler import rejection_sample  # noqa: E402

rsrc = inspect.getsource(rejection_sample)
asserts_1d = "assert draft_token_ids.ndim == 1" in rsrc
print("rejection_sample asserts draft_token_ids.ndim == 1 (flat chain):",
      asserts_1d)
RESULT["verify_asserts_1d_chain"] = asserts_1d

# 7) Verdict ----------------------------------------------------------------
section("VERDICT")
single_chain_only = (
    not RESULT["candidate_tree_topology_fields"]
    and not RESULT["tree_named_methods"]
    and RESULT["num_speculative_tokens_type"] == "<class 'int'>"
    and RESULT["tree_topology_rejected"]
    and RESULT["tree_choices_rejected"]
    and RESULT["propose_returns_linear_chain"]
    and RESULT["verify_asserts_1d_chain"]
)
RESULT["single_chain_only"] = single_chain_only
RESULT["verdict"] = "RED" if single_chain_only else "GREEN/AMBER"
print("single_chain_only =", single_chain_only)
print("verdict =", RESULT["verdict"])

print("\n--- JSON ---")
print(json.dumps(RESULT, indent=2, default=str))
