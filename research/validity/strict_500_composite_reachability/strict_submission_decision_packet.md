================================================================================
 STRICT-SUBMISSION DECISION PACKET  —  PR #357  (fern, CPU-only synthesis)
================================================================================
 RECOMMENDATION : GO-OPERATIVE-PENDING-471-TIES-CONFIRM
                  -> fire `senpai-strict-eqv-456`  (~456.36 TPS)

 A. THE NUMBER (stark #466, realized e2e — collapse to 161.70 REFUTED)
      realized strict  456.36 TPS  headline (conservative @ L=640)
      band [456.50, 467.14]  ·  cluster-mean ~459  ·  +295 over floor  ·  ~23 under deployed
      * composition was OPTIMISTIC: old composed 467.14 over-counts by +10.78 (> sigma_hw 4.82); true frontier in [456.50, <= 467.14]
      config-reachable via VLLM_BATCH_INVARIANT=1 (no served-source edit, no kernel rebuild)

 B. THE GATE (denken #471 served 128-prompt census — THE resolver, do not duplicate)
      binding: census == operative-1.0 (>= 0.99, every residual a bf16 tie, 0 semantic flips)
      * tension: locus 1.0000 (stark#466) vs served prior 0.9989@p90 (land#429) / 0.9978 all-pin (ubel#461)
      reconciliation: order-preserving 2D reduction removes the 3D split-KV near-tie population (denken#464)

 C. THE FORK
      GO-OPERATIVE  (471: 0 semantic flips, ties OK per human 08:24Z) -> `senpai-strict-eqv-456` (~456.36, honest strict win)
      FLOOR-LOCK    (471: >=1 SEMANTIC non-tie flip)                   -> `senpai-strict-m1ar-161` (161.70, literal-1.0 by construction)
      [BLOCKED retired — human ruled operative-1.0 honest-strict, 08:24Z]

 HONESTY HINGE : submission labeled with its TRUE census (e.g. "operative-1.0: 0.9989, 1 tied flip @ p90, 0 semantic"), NOT claimed literal 1.0
 D. CROSS-CHECK: ubel#470 BI-pin + stark#472 in-graph overlap -> pending_cross_check (n_independent=0, spread 2.64 vs sigma_hw 4.82)
 E. REFERENCE  : deployed 481.53 NON-equiv (id 0.9966, 3 ties) OUTSIDE strict set  ·  relax-prize DEAD (stark#452: -0.94 TPS, id 0.730)  ·  PPL 2.3772 <= 2.42 OK
 OWNERSHIP     : this packet = the human call · denken#471 = oracle · land#473 = trigger  ·  CPU-only, official_tps=0, no served-file change
================================================================================
 self-test: 35/35 invariants  ·  recommendation: GO-OPERATIVE-PENDING-471-TIES-CONFIRM
================================================================================
