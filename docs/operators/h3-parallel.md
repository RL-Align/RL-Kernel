# MiniMax-H3 Tensor / Sequence / Context Parallel Rows (WS2)

## Summary

RFC #420 rows `tp_attention_7168`, `tp_ffn_14336`, `cp_full_attention` (Ulysses and ring),
`cp_metadata_coshard`, `cp_arbitrary_length`, `distributed_packed_io`, and closeouts
`ws2_tp2_tp4_tp8`, `ws2_tp4_cp2`, `ws2_tp4_cp2_sp`. Contract `h3-parallel-v1`
(`rl_engine/backends/cuda/model_specific/minimax_h3/parallel.py`); the ring is
`.../minimax_h3/ring_attention.py` (`h3-ring-cp-v1`); the block that composes them is
`rl_engine/validation/models/h3_tp.py`, and the full 50-block model on a grid is
`rl_engine/validation/models/h3_parallel_model.py` (`h3-parallel-model-v1`). Cross-node reductions use
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
  rows than ranks) for the data movement (`scatter_packed` / `gather_packed`, Ulysses all-to-all):
  `cp_arbitrary_length`. The compute paths (`seq_sharded_linear`, the parallel block) need at
  least one row per rank; a shorter sequence is refused on every rank alike, before any collective.
- `coshard`: hidden, AdaLN row index, token tags, rotary tables are cut with one `SeqShard` so the
  metadata follows its rows: `cp_metadata_coshard`.
- `ulysses_attention`: all-to-all sequence <-> head (ragged split sizes), attention on `H/cp` heads
  of the full sequence, all-to-all back; both moves are exact, so rows equal the unsharded
  attention's if the attention is head independent (the repo's deterministic attention is). Tested at
  lengths 1, 2, 7, 130 and in the block at 261 / 263 rows.
- `ring_attention` (`h3-ring-cp-v1`, `cp_mode="ring"`): no head split, so it also runs where
  `heads % cp != 0` (CP3, TP8 x CP2). Every rank keeps its query rows; the key/value blocks travel
  around the ring by point-to-point copies, and the H3 full-attention kernels run on the rank's rows
  over all keys in global order. The `dK` / `dV` FP32 chains are continued rank by rank in
  ascending row order, so each rounding is the unsharded one for any shard sizes. The backward is
  serial over the ranks by construction and each rank holds one full K/V: exactness, not speed or
  memory. It needs `attention="h3_full"` (the #553 kernels); the interim attention is Ulysses only.
- `scatter_packed` / `gather_packed`: exact packed IO, ragged gather by padding: `distributed_packed_io`.
  `scatter_from_root` (only one rank holds the packed input; point-to-point row ranges),
  `gather_selected` (only the selected rows travel, e.g. `video_indices`, with an adjoint that
  hands each rank its own rows) and `check_row_identity` (fails when the ranks' rows are not the
  packed sequence in order) complete it.
- `seq_sharded_linear`: weight gradient from gathered rows so it equals the unsharded token sweep.
  The TP x CP block does the same: under CP every weight gradient (column- and row-parallel) is an
  exact gather of all CP ranks' rows followed by the same fixed-order token GEMM, so it equals the
  unsharded `dW` bit for bit (`run_tp_block_rows` passes `cp_shard` / `cp_group`).

## Sequence parallelism

With `sp=True` the residual stream is sharded by rows across the TP group: norm and gate-residual are
row-local, the column-parallel input is all-gathered (exact) and the row-parallel output is
reduce-scattered with the same tree (`rows % tp == 0` required, checked).

## Full model on a grid

`parallel_model_execution(model, grid, sp=...)` binds all 50 blocks of the real model to
`run_tp_block_rows`; the boundary rows keep their strict single-GPU bindings and run replicated.
Each block takes the packed sequence from the previous one, slices its rank's rows (CP rows, or
their TP share with SP), runs the parallel block and all-gathers the rows back (exact movement with
exact adjoints). The canonical reference is the same model on `Grid(1, 1, 0, 0)`.
`tools/validation/models/h3_parallel_full_model.py` (one process per GPU under `srun`) compares,
bit for bit, the video / audio velocity, the three input gradients, every block's output and output
gradient (`first_drift_*` names the first block, logical row, owning rank and modality) and this
rank's weight-shard gradients of all 350 sharded projections. `--inject BLOCK:ROW:RANK` perturbs one
row on one rank: a negative control that has to be reported at that block, row and owner. Every
rank loads the whole checkpoint and runs the canonical model first, so it is a validation harness,
not a memory-sharded implementation.

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

Mean time per rank of one forward call (ms), from the `h3-ws2-parallel-h100/*.json` reports: TP2
33.1, TP4 32.6, TP8 32.4, TP4xCP2 31.5, TP4xCP2+SP 33.3 (tiny 264-row case, launch-bound; not a
scaling claim). Each call also rebuilds the block context, the operators and the weight shards, so
this is the time of a whole `block_case(..., backward=False)` call, not of the kernels alone.

## Limits

- Attention, SwiGLU and QK norm inside the block are the repo's interim operators; the parallel
  rows only require them to be row/head independent, which the bitwise results confirm.
- Pipeline parallelism is out of scope (`ws2_pipeline_parallel` is not part of this change).
- dW under CP is the unsharded token sweep (exact gather of every CP rank's rows, then the same
  fixed-order GEMM), also in the TP x CP block, whose weight-gradient shards are compared with the
  single-GPU block's. The gather makes every rank hold all rows of the GEMM operands for the
  backward.
- The compute paths (`seq_sharded_linear`, `run_tp_block`, `run_tp_block_rows`, hence the full
  model) need at least one row on every CP rank; the ring and the packed IO themselves accept ranks
  that hold no rows.
- The full-model runner keeps the whole 33B checkpoint on every rank and runs the canonical model
  first; it is not run with sharded weight storage.
- `HierarchicalCollective` assumes ranks are packed node-major and node counts of 1/2/4/8.
  With several TP groups (CP > 1) it is given every TP group's ranks (`group_ranks`, as
  `make_grid` does) and every rank creates all their subgroups in one order, as
  `torch.distributed.new_group` requires; a TP subgroup without that list is refused. The
  hierarchical path with CP > 1 has a CPU test of the creation order only, no GPU run.

## Hopper evidence chart

These outputs and figures are not part of this diff; they are on the fork's `pr_assets` branch (links in the PR description).
Per-layout results of the 2 nodes x 4 H100 runs (`reports/experiments/h3-ws2-parallel-h100/*.json`): every layout is bitwise equal to the single-GPU reference
(forward and input gradient); mean per-rank forward time of block 0 is in the figure.
