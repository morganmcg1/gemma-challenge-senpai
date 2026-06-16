================================================================================
 THE HONEST STRICT-FRONTIER MAP  —  PR #357  (fern, CPU-only synthesis)
================================================================================
 A. THE MAP (ranked by realized TPS; private-safety is the 2nd axis)
      rung                         realized TPS   strict?           private-safe?
      ---------------------------  ------------   ---------------   --------------------
      floor-lock M=1 AR             166.23 proj   literal-1.0*      SAFE  (Δ 0.633%)  <- STICKS
      global-flag BI=1             234.47       operative-1.0     RISKY (Δ 4.295%)  <- #474 live
      surgical / 2D byte-exact     456.98 pred  byte-exact(locus) RISKY (Δ 4.295%)  <- OBE strict
      deployed (reference)         481.53       NON-equiv .9966   — (outside strict set)
      * floor-lock literal-1.0 = BY CONSTRUCTION (M=1 AR, no drafter); served census vs the
        M=1 AR reference is the load-bearing confirm (relay-run logged verdict_literal_1p0=0
        vs the precache ref, 119/128 divergent — a DIFFERENT config; flagged to advisor).

 B. THE PRINCIPLE (denken #489): private-safety = f(Δ drafter-gap), NOT f(TPS)
      floor-lock Δ 0.633% (4.37pp headroom, breach ~0.001%)  vs  spec-alive Δ 4.295% (0.71pp, breach 24.3%)
      SCALE-INVARIANT: the 222, the 457 and the 481 all carry the SAME Δ -> SAME breach, any TPS
      => floor-lock (no drafter) is the ONLY strict private-safe ship.  Fast + byte-exact ≠ safe.

 C. >500 CLOSURE (settled): strict >500 DEAD via all known levers
      realized strict ceiling 467.14 < 500 (gap 32.86); IEEE-754 tax irreducible (denken#423);
      no free fast byte-exact GEMM (#481: tax 22-63%, e2e 51.4%).  Only >500 path =
      greedy-UNSAFE ~16% relax-prize (id 0.730, out of lane, human #407).  ~3.0x over the 166 floor.

 D. FORWARD LEVERS (HELD OPEN; finalize to terminal when #488/#491/#492 land)
      [ ] #491 reduction-sensitivity -> FASTER floor-lock (attacks TPS-tax, Δ stays safe)
      [ ] #492 EAGLE-3 drafter      -> PRIVATE-SAFE fast rung (attacks Δ-gate, keeps drafter)
      [ ] #488 surgical realize     -> is the 457.5 a REAL served rung, or a mirage?
      orthogonal axes; only their CONJUNCTION yields faster AND private-safe.  (0/3 landed)

 E. THE LIVE #474 CALL: floor-lock 166 (sticks) vs 234 (fast-risky) — ruling: PENDING
      binding: does a >5% private re-draw INVALIDATE (->floor-lock) or PENALIZE (->222)?
      advisor rec: FLOOR-LOCK unless a breach is known to be only a penalty
 OWNERSHIP: this packet = the MAP · land#473 = trigger · denken#471 = census oracle  ·  CPU-only, official_tps=0, no served-file change
================================================================================
 self-test: 42/42 invariants  ·  terminal=False  ·  >500 strict: DEAD-via-known-levers
================================================================================
