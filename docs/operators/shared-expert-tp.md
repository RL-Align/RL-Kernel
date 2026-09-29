# Shared Expert TP/SP + shared-once (P5-8, WS2)

WS2 parallel contract for the P5-5 shared expert: TP column/row split, SP
reduce-scatter equivalence, and the EP shared-once gate.

## Numeric profile `p5-shared-tp-tree8-v1`

FP32 addition is not associative, so no cross-rank merge can be byte-equal to
a purely serial single-card reduction. Following `det_gemm`'s precedent (a
mid-split tree whose children are the contiguous per-rank ranges), the TP=1
reference itself is anchored on a fixed tree:

- **FC1 column-parallel**: gate rows and up rows are EACH split contiguously
  per rank (never the raw `[2F]` rows), so SwiGLU pairs gate/up locally.
  These columns stay byte-equal to the WS1 strict path.
- **FC2 row-parallel**: the K dimension (F) reduces through a fixed 8-leaf
  mid-split tree (leaf = contiguous F/8 chunk, leaf-internal WS1 strict serial
  GEMM, FP32 merge, single BF16 round at the end). A rank owning `8/tp`
  adjacent leaves computes exactly one subtree, so TP ∈ {1, 2, 4, 8} are all
  byte-equal and the cross-rank merge is the upper part of the same tree.
- **Backward**: `dh`/SwiGLU-bwd are rank-local; `dX` merges per-leaf nodes
  `gate_leaf + up_leaf` through the same tree (FP32 out).
- **SP**: every element's tree is identical on every rank, so reduce-scatter
  (merge + keep your rows) equals all-reduce + slice bitwise.
- Leaf order follows **logical tags**, never physical rank ids — the ordered
  collective facade contract (#7), mocked here by `OrderedTreeReducer`.

The tree spec, placement, tp degree and collective all land in
`provenance()`.

## shared-once (EP)

EP replicates the shared expert. `SharedOnceLedger` + `combine_shared_once`
fail-close the combine when the same shared output would be merged twice; the
injection test drives that path.

## Entry points

```
rl_engine.moe.parallel.TPSimulatedSharedExpertProvider(base="cuda"|"triton", tp=1|2|4|8)
rl_engine.moe.parallel.shard_shared_batch / sp_shard / SharedOnceLedger
```

## Acceptance

```bash
pytest tests/test_shared_expert_tp.py        # TP/SP/shared-once gates (single process)
pytest tests/test_shared_expert_tp_dist.py   # gloo multi-process, shards only, same bytes
```

The single-process provider runs every rank's local math with the WS1 strict
kernels and merges through the same tree the multi-process path uses, so the
two are bitwise interchangeable (verified by the dist test).
