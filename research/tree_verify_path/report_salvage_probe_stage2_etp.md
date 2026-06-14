# PR #71 — STAGE-2a: E[T] projection from the measured STAGE-1 rates

**Status:** the E[T] (tok/step) projection — derived from STAGE-1's **measured**
real-stack rates fed through the validated closed-form / Monte-Carlo accept model
(`tree_spec.expected_committed_tokens` + `monte_carlo_committed`). This converts
the STAGE-1 branch-hit into the **tok/step gate number** fern #142 consumes.
**STAGE-2b (the live M=16 tree-forward that MEASURES E[T] directly + the
both-halves runtime assert) is the confirmation step and is pending** — see the
caveat at the end for the one assumption this projection makes that only the real
forward can test.

## Inputs (all measured on the real stack, STAGE 1)

- spine accept (rank-1) **top1 = 0.729** (from the divergence distribution:
  full_accept 0.177 / div_at_branch 0.489 / div_no_branch 0.333 matches
  P(first div at depth d)=top1^d·(1−top1)).
- branch hit (rank-2 | rank-1 miss at a width≥2 pos) **= 0.360** (405/1126, SE≈0.014).

The closed form takes a per-rank **unconditional, mutually-exclusive** accept
vector `p` where `p[k-1] = P(verifier argmax == the rank-k child's token)`:
- `p1 = top1 = 0.729`
- `p2 = P(rank-1 misses)·P(rank-2==argmax | miss) = (1−0.729)·0.360 = 0.0976`
- (`p3`, for the M=32 max-branch-3 tree, scaled from the oracle ladder by the
  measured/oracle rank-2 ratio 0.360/0.4165.)

## Projection — read it as a MATCHED-SPINE relative gain, not an absolute vs 3.844

**Critical framing.** The 3.844 "deployed ref" (fern #134) is the deployed chain's
tok/step on the *official* set, which implies an effective spine-accept of
**t1≈0.773** (the chain-7 E[T] that equals 3.844). My STAGE-1 spine-accept is
**t1=0.729** (measured on the *local* set, lower). Comparing the local-spine tree
(3.915) to the official-spine ref (3.844) mixes prompt sets and badly under-sells
the tree (apparent +1.8%). The tree's spine **is** the deployed chain (spine
identity by construction), so the honest metric is **tree vs chain at the SAME
spine rate** — which is exactly wirbel #49's relative +16%.

| spine t1 | chain E[T] | M16 tree | M32 tree |
|---|---|---|---|
| **0.729** (local, STAGE-1 measured) | 3.396 | **3.915 (+15.3%)** | **4.416 (+30.0%)** |
| **0.773** (= the 3.844 deployed ref) | 3.844 | **4.414 (+14.8%)** | **4.912 (+27.8%)** |

(branch-hit 0.360; M32 rank-3 scaled 0.2655·0.360/0.4165. Closed-form == 200k-MC to
≤0.004 every row.) **The relative gain is spine-robust: ~+15% at M16, ~+28% at M32.**

## Three decision-relevant reads

1. **The +16% reproduces on the MEASURED rates, spine-independently.** +15.3% (local
   spine) / +14.8% (deployed-ref spine) at M16 — this is wirbel #49's stranded +16%
   E[T], recovered on the real stack, NOT just the oracle projection.
2. **GO is robust to the branch-hit shortfall.** The 0.360-vs-0.4165 gap (STAGE-1's
   honest caveat) costs only **0.06 E[T]** at M16 and 0.14 at M32. Even if the
   rank-2 fidelity gap never closes, the tree still delivers ~+15%/+28%. The gap is
   a *refinement*, not a GO/NO-GO pivot.
3. **The ~5.0 target lives at M=32, and is reached at the deployed-ref spine.** At
   t1=0.773 the M32 tree projects **4.912 ≈ the ~5.0 target** (reconciles ubel
   #157's "descent E[T]=5.04"); M16 reaches 4.41. M=32 is still ≤ the denken #68
   M≤32 ceiling (the +18.4% verify-cost sweet spot). So the build target that hits
   ~5.0 is **M=32 max-branch-3**; M=16 is the +16% milestone proven first.

## The one assumption only the real forward (STAGE-2b) can test

The closed form applies `p1 = 0.729` to **every** rank-1 edge — including the deep
rank-1 continuations *inside* a salvaged branch sub-tree (PARENT_M16: 2→5→8→10 and
4→7). STAGE 1 measured only the **first** salvage edge (the rank-2 hit at the
divergence, nodes 2 and 4). Whether a salvaged branch then *continues* accepting at
the spine rate 0.729 is unmeasured — the linear verify never places node 2's K/V in
the cache, so it cannot compute node 5's argmax. **STAGE-2b's live M=16 tree
forward supplies `node_argmax` for all 16 nodes (incl. 5,8,10,7), which the
validated `descend_accept` walk turns into a directly-measured E[T]** — confirming
(or correcting) this projection and firing the both-halves runtime assert
(qq_bias DISPATCHED at M=16 parent=m16 AND descend walk RAN on the 16-node layout).

## Repro
```
python3 -c "import sys;from importlib import util as u;from pathlib import Path; \
s=u.spec_from_file_location('ts',Path('scripts/profiler/tree_spec.py')); \
ts=u.module_from_spec(s);sys.modules['ts']=ts;s.loader.exec_module(ts); \
t=ts.TreeSpec(ts.PARENT_M16);print(ts.expected_committed_tokens(t,[0.729,(1-0.729)*0.360]))"
# -> 3.9149   (Monte-Carlo monte_carlo_committed -> 3.9122)
```
