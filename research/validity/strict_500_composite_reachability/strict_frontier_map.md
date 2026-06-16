================================================================================
 THE HONEST STRICT-FRONTIER MAP  —  PR #357  (fern, CPU-only synthesis)  [TERMINAL]
================================================================================
 A. THE MAP (ranked by realized TPS; private-safety is the 2nd axis)
      rung                         realized TPS   strict?           private-safe?
      ---------------------------  ------------   ---------------   --------------------
      floor-lock M=1 AR             166.23 proj   literal-1.0*      SAFE  (Δ 0.633%)  <- fallback
      global-flag BI=1             234.47       operative-1.0     RISKY (Δ 4.295%)  <- superseded
      surgical / 2D byte-exact     357.60 meas  byte-exact e2e    RISKY (Δ 4.295%)  <- #474 SHIP
      deployed (reference)         481.53       NON-equiv .9966   — (outside strict set)
      surgical 357.6 = lawine #488 (ko01dcyy) MEASURED byte-exact e2e — REFUTES the 456.98
      composed prediction (honesty flag #2 vindicated: composed-vs-realized, +135.7 over 222).
      * floor-lock literal-1.0 = BY CONSTRUCTION (M=1 AR int4 = the strict reference); the denken
        #471 served census (bwyhpkd7) ACCEPTED the floor 1.0/0 flips (rejects deployed
        .9966), but confirmed=False — reference M=1-AR-self-vs-greedy unverifiable under iso (non-gating).

   >>> SHIP STATUS: surgical-357 fire armed (land #473, per #474) <<<

 B. THE PRINCIPLE (denken #489): private-safety = f(Δ drafter-gap), NOT f(TPS)
      floor-lock Δ 0.633% (4.37pp headroom, breach ~0.001%)  vs  spec-alive Δ 4.295% (0.71pp, breach 24.3%)
      SCALE-INVARIANT: the 222, the 357.6 and the 481 all carry the SAME Δ -> SAME breach, any TPS
      => floor-lock (no drafter) is the ONLY strict private-safe ship.  Fast + byte-exact ≠ safe.

 C. >500 CLOSURE (settled): strict >500 DEAD via all known levers
      realized strict ceiling 467.14 < 500 (gap 32.86); IEEE-754 tax irreducible (denken#423);
      no free fast byte-exact GEMM (#481: tax 22-63%, e2e 51.4%).  Only >500 path =
      greedy-UNSAFE ~16% relax-prize (id 0.730, out of lane, human #407).  ~3.0x over the 166 floor.

 D. FORWARD RESEARCH (DECOUPLED -> #481 zoom-out menu; NOT capstone-gating)
      [ ] #491 reduction-sensitivity -> FASTER floor-lock (attacks TPS-tax, Δ stays safe)
      [ ] #492 EAGLE-3 drafter       -> PRIVATE-SAFE fast rung (attacks Δ-gate, keeps drafter)
      orthogonal axes; only their CONJUNCTION yields faster AND private-safe.  Wider menu:
      SGLang #498, TRT-LLM, alt spec-dec (Medusa/EAGLE-3/prompt-lookup), GPTQ/AWQ/SmoothQuant, FA3 / torch.compile.

 E. THE #474 DECISION (RESOLVED): human ruled "357 -- go, finish it" (13:51Z) -> SHIP-357
      chose the fast-risky/PENALIZE lane over floor-lock 166; realized scorer has no
      token-identity gate (stark #493), so operative-1.0 clears.  ruling: SHIP-SURGICAL-357-FAST-RISKY
      fire: surgical-357 (357.6 TPS) armed on land #473 (fires the draw on its next poll).
 OWNERSHIP: this packet = the MAP · land#473 = trigger · denken#471 = census oracle  ·  CPU-only, official_tps=0, no served-file change
================================================================================
 self-test: 47/47 invariants  ·  terminal=True  ·  >500 strict: DEAD-via-known-levers  ·  SHIP: surgical-357
================================================================================
