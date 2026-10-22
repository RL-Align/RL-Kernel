# MiniMax-H3 Tensor / Sequence / Context Parallel Rows (WS2)

## Summary

RFC #420 rows `tp_attention_7168`, `tp_ffn_14336`, `cp_full_attention` (Ulysses),
`cp_metadata_coshard`, `cp_arbitrary_length`, `distributed_packed_io`, and closeouts
`ws2_tp2_tp4_tp8`, `ws2_tp4_cp2`, `ws2_tp4_cp2_sp`. Contract `h3-parallel-v1`
(`rl_engine/backends/cuda/model_specific/minimax_h3/parallel.py`); the block that composes them is
`rl_engine/validation/models/h3_tp.py`. Cross-node reductions use
`rl_engine/distributed/algorithms/hierarchical.py` (`hierarchical_ipc_nccl_tree`).

Result (H100, 2 nodes x 4 GPUs, block 0 of the pinned checkpoint, 264 packed rows): every rank's
rows of the forward, the input gradient and every weight-gradient shard are **bitwise equal** to the
single-GPU canonical block for TP2, TP4, TP8 (two nodes), TP4 x CP2 (two nodes) and TP4 x CP2 + SP.
Evidence: `reports/experiments/h3-ws2-parallel-h100/*.json`.

## The canonical reduction

All cross-rank sums are defined on **8 canonical units** (the divisors of 8 are the TP sizes):

| Layer | Sharding | Reduction | Rule |
| --- | --- | --- | --- |
| Q/K/V, gate/up forward | output rows (whole heads / halves) | none | the unsharded GEMM's columns (`h3-projection-gemm-v1`) |
| O (7168 -> 5376), down (14336 -> 5376) forward | reduction axis, `8/tp` K-units per rank | K | each K-unit is one fixed-order FP32 GEMM; a rank adds its units as a balanced tree; the collective adds ranks as the same tree; one BF16 rounding |
| O / down backward (`dX`, `dW`) | - | N / tokens | plain fixed-order GEMM, independent of the shard |
| Q/K/V, gate/up backward `dX` | output rows | N | unit `j` = block `j` of every fused weight, summed in the order q,k,v (hidden,gate); tree over units; collective; one rounding |

The unit of the O projection is 896 K (14 K-tiles) and of down 1792 (28 K-tiles), so a unit never
splits a tile. FP32 partials cross ranks (`DeterministicCollective` fixed tree on a node;
`HierarchicalCollective` across nodes). The row-parallel contract differs from the single-sweep
K reduction of the unsharded O/down GEMM rows (claimed by another contributor): it is the
TP-invariant definition, and TP=1 is that definition, not the sweep.

## Cross-node collective

`DeterministicCollective` is CUDA IPC and therefore single node. `HierarchicalCollective(local_size)`
reduces each group of `local_size` ranks with that tree, moves the group partials with NCCL
`all_gather` (exact copies, no NCCL arithmetic) and adds them in one fixed balanced order. For
power-of-two sizes this is the 8-way tree; a unit test checks it equals the single-node tree of the
same world on 4 GPUs for FP32 and BF16 (`local_size` 1 and 2), and the TP8 run above uses it across
two nodes.

## Context parallelism

- `SeqShard`: contiguous row slices differing by at most one row, any length >= 0 (including fewer
  rows than ranks): `cp_arbitrary_length`.
- `coshard`: hidden, AdaLN row index, token tags, rotary tables are cut with one `SeqShard` so the
  metadata follows its rows: `cp_metadata_coshard`.
- `ulysses_attention`: all-to-all sequence <-> head (ragged split sizes), attention on `H/cp` heads
  of the full sequence, all-to-all back; both moves are exact, so rows equal the unsharded
  attention's if the attention is head independent (the repo's deterministic attention is). Tested at
  lengths 1, 2, 7, 130 and in the block at 261 / 263 rows.
- `scatter_packed` / `gather_packed`: exact packed IO, ragged gather by padding: `distributed_packed_io`.
- `seq_sharded_linear`: weight gradient from gathered rows so it equals the unsharded token sweep.

## Sequence parallelism

With `sp=True` the residual stream is sharded by rows across the TP group: norm and gate-residual are
row-local, the column-parallel input is all-gathered (exact) and the row-parallel output is
reduce-scattered with the same tree (`rows % tp == 0` required, checked).

## Tests and evidence

`tests/models/minimax_h3/test_h3_parallel.py`: 70 tests on 4 H100 (sharding, canonical folding for
TP 2/4/8, accuracy against FP64, row independence, Ulysses ragged, ten block grids with backward,
hierarchical collective). `tools/validation/models/h3_parallel_run.py` runs a grid under `srun` on
any node count and compares against the single-GPU block.

| Grid | GPUs / nodes | Collective | Forward | dX | dW shards |
| --- | --- | --- | --- | --- | --- |
| TP2 | 2 / 1 | IPC tree | bitwise | bitwise | bitwise |
| TP4 | 4 / 1 | IPC tree | bitwise | bitwise | bitwise |
| TP8 | 8 / 2 | hierarchical | bitwise | bitwise | bitwise |
| TP4 x CP2 | 8 / 2 | IPC tree (TP in node), NCCL all-to-all (CP across) | bitwise | bitwise | not compared (token axis sharded) |
| TP4 x CP2 + SP | 8 / 2 | same | bitwise | bitwise | not compared |

Timing of the forward per rank (ms): TP2 80, TP4 102, TP8 110, TP4xCP2 113, TP4xCP2+SP 104
(tiny 264-row case, launch-bound; not a scaling claim).

## Limits

- Attention, SwiGLU and QK norm inside the block are the repo's interim operators; the parallel
  rows only require them to be row/head independent, which the bitwise results confirm.
- Pipeline parallelism is out of scope (`ws2_pipeline_parallel` is not part of this change).
- dW under CP is exact through `seq_sharded_linear` for non-TP linears; the TP x CP block compares
  forward and `dX` (CP shards the token axis of the TP weight gradients).
- `HierarchicalCollective` assumes ranks are packed node-major and node counts of 1/2/4/8.

## Hopper evidence chart

These outputs and figures are not part of this diff; they are on the fork's `pr_assets` branch (links in the PR description).
Per-layout results of the 2 nodes x 4 H100 runs (`reports/experiments/h3-ws2-parallel-h100/*.json`): every layout is bitwise equal to the single-GPU reference
(forward and input gradient); mean per-rank forward time of block 0 is in the figure.
