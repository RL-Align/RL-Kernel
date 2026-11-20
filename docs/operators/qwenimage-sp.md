# Qwen-Image sequence sharding and arbitrary lengths

Implements the `sp_shard_gather` and `sp_arbitrary_length` work items in
[#386](https://github.com/RL-Align/RL-Kernel/issues/386). Image tokens and their
three-axis RoPE positions are partitioned together across sequence-parallel (SP)
ranks; gather restores the original token order. Text tokens and prompt metadata
remain replicated.

## Usage

```python
from rl_engine.distributed.algorithms.sequence_parallel import (
    gather_sequence,
    shard_image_sequence,
)

local_image, local_positions, layout = shard_image_sequence(
    image, image_positions, group=sp_group
)
restored_image = gather_sequence(local_image, layout, group=sp_group)
restored_positions = gather_sequence(
    local_positions, layout, dim=0, group=sp_group
)
```

For batched positions (`[B, S, 3]`), use `dim=1` when gathering positions.
The full image sequence is needed before CFG combine, scheduler updates, and
`logp` computation.

## Input and partitioning

| Tensor | Shape | Requirements |
| --- | --- | --- |
| Image | `[B, S, C]` | FP32 or BF16, nonempty; replicated across SP ranks |
| Positions | `[S, 3]` or `[B, S, 3]` | Same device as image; no gradients |
| Gather input | Local sequence dimension | Same dtype and non-sequence dimensions across ranks |

Input tensors may be noncontiguous. Returned shards and gathered outputs are
contiguous. Original position values and dtypes are preserved. Integer position
tensors are supported; differentiable gather supports FP32 and BF16.

Each rank `r` owns a contiguous interval of width `ceil(S / SP)`, clipped to
`[0, S)`. For `S=6889`, `SP=4`, local compute lengths are
`[1723, 1723, 1723, 1720]`. Empty shards are supported when `S < SP`.

Only communication buffers are padded to equal lengths. Gather concatenates
ranks in order and removes the padded tail. Computation sees the original
image rows and their global RoPE positions.

The implementation uses `TorchDistributedDeterministicCollective` over
CUDA/NCCL or CPU/Gloo. SP sizes 1, 2, 4, and 8 are supported. Distributed tests
cover SP=1/2/4; SP=8 has layout-level test coverage.

## Autograd and correctness

Both operations support first-order autograd:

- **Shard backward:** all-gather the disjoint local gradients to each input replica.
- **Gather backward:** sum output gradients across SP ranks using the balanced
  rank tree, then reduce-scatter to the owning shards. BF16 contributions are
  accumulated in FP32 and cast back at the output.

For SP=4, the gradient reduction order is `(g0 + g1) + (g2 + g3)`.
All ranks participate in backward. A full loss computed independently on every
rank contributes once per rank; the caller controls loss normalization.

Forward sharding and gathering preserve tensor bytes, including BF16/FP32 values
and position metadata. Tests compare against independent CPU indexing and known
gradient values.

## Validation

Run from the repository root using pytest (without `torchrun`):

```bash
OMP_NUM_THREADS=1 NCCL_DEBUG=WARN python -m pytest \
  tests/distributed/sp/test_sequence_parallel.py \
  tests/distributed/collectives/test_transport_deterministic_collective.py -q
```

Coverage includes Gloo/NCCL on SP=1/2/4, FP32/BF16, `B=1/2`, noncontiguous
inputs, subgroup ranks, empty shards, padded-buffer poisoning, byte-exact
round trips, and forward/backward gradients. CUDA cases cover `C=3072` and
sequence lengths 4096, 6032, and 6889.

Recorded validation (2026-11-15): **65 passed** (26 SP tests, 39 collective
regression tests) on **4× A100-SXM4-80GB**, Python 3.11.10,
PyTorch 2.9.0+cu130, CUDA 13.0, NCCL 2.27.7.

## Benchmark

```bash
OMP_NUM_THREADS=1 NCCL_DEBUG=WARN torchrun --standalone --nproc-per-node=4 \
  -m benchmarks.distributed.benchmark_sequence_parallel \
  --dtype bf16 --batch 1 --output artifacts/qwenimage-sp/sp4-bf16.json
```

For a CPU smoke test:

```bash
OMP_NUM_THREADS=1 python -m benchmarks.distributed.benchmark_sequence_parallel \
  --device cpu --hidden 8 --seq-lengths 7 --warmup 1 --iterations 2
```

Mean wall time (ms) on the A100 setup above, `B=1`, `S=6889`, `C=3072`,
BF16; 5 warmup and 20 measurement iterations. Times include collective setup.

| SP | Gather | Gather forward + backward |
| --- | ---: | ---: |
| 1 | 0.057 | 0.121 |
| 2 | 1.283 | 3.268 |
| 4 | 1.663 | 4.849 |

The benchmark also measures sharding and writes per-shape timings, hardware,
runtime, backend, and source metadata to JSON.
