#!/usr/bin/env python3
"""PR #799 — log the tree/EAGLE drafter feasibility verdict to W&B.

Feasibility probe only: no training, no benchmark, no GPU. Records the
code-inspection verdict + load-bearing citations so the result is in the
W&B record alongside the rest of the lineage.

Run under a wandb-capable python (e.g. /usr/bin/python3, wandb 0.27.0):
    WANDB_DIR=research/tree_eagle_feasibility_799 \
      /usr/bin/python3 research/tree_eagle_feasibility_799/log_wandb.py
"""

import json
import os
import re

import wandb

HERE = os.path.dirname(os.path.abspath(__file__))

# Pull the machine-checked signals from the static dry-run output.
with open(os.path.join(HERE, "static_dryrun.out")) as f:
    text = f.read()
probe = json.loads(text[text.index("--- JSON ---") + len("--- JSON ---"):])

# Load-bearing code citations (pinned wheel 0.22.1rc1.dev307+g3e8afdf78).
citations = {
    "config_scalar_num_spec": "vllm/config/speculative.py:80  num_speculative_tokens: int = Field(gt=0)",
    "config_methods_no_tree": "vllm/config/speculative.py:59-68  SpeculativeMethod Literal (no 'tree')",
    "config_non_tree_phrase": "vllm/config/speculative.py:137  use_local_argmax_reduction docs: 'non-tree speculation'",
    "config_suffix_tree_is_retrieval": "vllm/config/speculative.py:167  suffix_decoding_max_tree_depth (retrieval suffix-tree, not candidate tree)",
    "proposer_factory_gemma4": "vllm/v1/worker/gpu_model_runner.py:589-590  use_gemma4_mtp() -> Gemma4Proposer (single proposer)",
    "gemma4_one_token_per_step": "vllm/v1/spec_decode/gemma4.py:31,47  Gemma4Proposer(SpecDecodeBaseProposer); constant_draft_positions=True; 'producing one token'",
    "propose_linear_chain": "vllm/v1/spec_decode/llm_base_proposer.py:588-659  loop feeds draft_token_ids_list[-1]; torch.stack(...,dim=1) -> [batch, num_spec]",
    "dummy_run_tree_fixme": "vllm/v1/spec_decode/llm_base_proposer.py:1493-1497  FIXME: tree-based specdec NOT implemented (1 fwd-pass per spec token)",
    "eagle_same_linear_base": "vllm/v1/spec_decode/eagle.py:10-22  EagleProposer == SpecDecodeBaseProposer (no tree override); separate-weight head, not shared-KV gemma4_mtp",
    "verify_flat_chain": "vllm/v1/sample/rejection_sampler.py:413  assert draft_token_ids.ndim == 1 (flat per-request chain; greedy kernel grid (batch_size,))",
    "medusa_also_linear": "vllm/v1/spec_decode/medusa.py:52-55  MedusaProposer returns [batch, num_heads] (one token/head, not a candidate tree)",
    "stock_0220_same": "stock vLLM 0.22.0 (official image default): identical ndim==1 verify assert + scalar num_speculative_tokens + zero tree methods",
}

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="gemma-challenge-senpai",
    group="tree-eagle-feasibility",
    name="land/tree-eagle-drafter-feasibility",
    job_type="feasibility-probe",
    tags=["pr-799", "spec-decode", "tree", "eagle", "gemma4_mtp", "feasibility", "no-launch"],
    config={
        "pr": 799,
        "kind": "code-inspection + static-load dry-run (CPU only, no HF job, no retrain)",
        "vllm_version": probe["vllm_version"],
        "num_speculative_tokens_type": probe["num_speculative_tokens_type"],
        "speculative_methods": probe["speculative_methods"],
        "tree_named_methods": probe["tree_named_methods"],
        "candidate_tree_topology_fields": probe["candidate_tree_topology_fields"],
        "deployed_spec_config": {"method": "mtp->gemma4_mtp", "num_speculative_tokens": "scalar (6/7)"},
        "citations": citations,
    },
)

# Boolean signals as 0/1 metrics for easy filtering.
wandb.log({
    "verdict_RED": 1,
    "single_chain_only": int(probe["single_chain_only"]),
    "tree_topology_rejected": int(probe["tree_topology_rejected"]),
    "tree_choices_rejected": int(probe["tree_choices_rejected"]),
    "propose_returns_linear_chain": int(probe["propose_returns_linear_chain"]),
    "verify_asserts_1d_chain": int(probe["verify_asserts_1d_chain"]),
    "num_tree_methods": len(probe["tree_named_methods"]),
    "num_candidate_tree_topology_fields": len(probe["candidate_tree_topology_fields"]),
})

run.summary["verdict"] = "RED"
run.summary["verdict_reason"] = (
    "vLLM 0.22 spec-decode is single-linear-chain only for gemma4_mtp: "
    "scalar num_speculative_tokens, no tree method/topology field, "
    "rejection_sampler asserts draft_token_ids.ndim==1, no tree-attention verify kernel."
)
run.summary["build_cost"] = (
    "tree drafter would need a vLLM fork: new tree proposer + tree-attention "
    "verify kernel (inherits #130/#117/#108 1-wave HBM wall) + config topology surface"
)
print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_NAME", run.name)
print("WANDB_RUN_URL", run.url)
run.finish()
