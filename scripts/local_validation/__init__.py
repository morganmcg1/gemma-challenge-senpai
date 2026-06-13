"""Local validation + profiling harness for Gemma submissions.

Reusable, LOCAL-ONLY tooling that proves a submission's hardware-independent
correctness gates (greedy token-identity, perplexity) and captures exploratory
profiling/throughput evidence on the assigned A10G, so every HF-Job approval
issue can ship with all local checks already green.

Modules:
  paths               - single source of truth for official tools + datasets.
  harness             - serve a submission locally, capture decode, run PPL,
                        probe single-stream TPS, manage server venvs.
  greedy_gate         - thin CLI over the official greedy-identity verifier.
  gen_greedy_reference- offline vLLM plain-greedy AR reference generator.
  ppl_runner          - serve + endpoint PPL against the ground-truth tokens.
  profile_decode      - run the official decode op-profiler locally.
  validate_submission - one-command orchestrator over all of the above.
"""
