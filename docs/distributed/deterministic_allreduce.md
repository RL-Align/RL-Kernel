# Deterministic All-Reduce

RL-Kernel exposes a small deterministic all-reduce helper for WS2 distributed
operator work. The helper is intentionally conservative: it defines an explicit
contract, provides a slow ordered-rank reference path, and keeps the NCCL fast
path opt-in.

## Contract

`deterministic_all_reduce(tensor, config)` reduces `tensor` in place and returns
the same tensor object. The result is expected to be stable when all of these
inputs are the same:

- world size;
- global rank order;
- input tensor values;
- dtype;
- reduction op;
- helper mode;
- process-group and backend environment.

For `op="mean"`, RL-Kernel performs a sum and divides by `world_size` at a
fixed point. Integer tensors are not accepted for `mean`.

## Modes

### `ordered_rank_fallback`

This is the slow reference mode. Each rank contributes a tensor through
`torch.distributed.all_gather`. Rank 0 then accumulates the gathered tensors in
ascending global rank order, casts the result back to the original dtype, and
broadcasts it to every rank.

The operation order is:

1. make each rank input contiguous;
2. gather rank inputs in global-rank order;
3. on rank 0, accumulate rank `0, 1, ..., world_size - 1`;
4. use FP32 accumulation for floating-point tensors when configured;
5. divide once for `op="mean"`;
6. broadcast the final tensor from rank 0.

This path is memory-heavy because every rank receives the gathered input list. It
is intended for small validation tensors, unsupported-hardware fallbacks, and
test or debug oracles.

### `nccl_ring`

This mode uses `torch.distributed.all_reduce` and optionally the NCCL environment
helper below. It is a fast path, not a blanket promise of bitwise determinism
across all NCCL versions and hardware. Validate it on the target machine before
claiming support.

Call `configure_deterministic_nccl_env()` before
`torch.distributed.init_process_group` when using NCCL:

```python
from rl_engine.distributed import configure_deterministic_nccl_env

configure_deterministic_nccl_env(overwrite=True)
```

The helper sets these variables when they are unset, or overwrites them when
`overwrite=True`:

```bash
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
```

If the process group is already initialized, the helper warns because NCCL may
have already consumed its collective configuration.

## Fallback Behavior

- If `torch.distributed` is unavailable, RL-Kernel raises a clear runtime error.
- If no process group is initialized and `WORLD_SIZE` is unset or `1`, the helper
  returns the input tensor unchanged.
- If no process group is initialized and `WORLD_SIZE > 1`, the helper raises a
  runtime error.
- If `world_size == 1`, the helper returns the input tensor unchanged.
- `async_op=True` is not implemented for the deterministic helper.

## Smoke Tests

Run the unit checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/distributed/test_deterministic_allreduce.py -q
```

Run the ordered fallback smoke on CPU/Gloo:

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/test_deterministic_allreduce.py \
  --backend gloo --mode ordered_rank_fallback --dtype fp32 --device cpu
```

Run the NCCL ring smoke on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
NCCL_MIN_NCHANNELS=1 \
NCCL_MAX_NCHANNELS=1 \
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/test_deterministic_allreduce.py \
  --backend nccl --mode nccl_ring --dtype fp32 --device cuda
```

The smoke test prints rank-0 JSON with `max_abs_diff`, `max_rel_diff`,
`mismatch_count`, and `bitwise_equal`.

## Current Limitations

- NVLS / NVLink-Sharp is not claimed by this helper. It needs a separate probe
  with NCCL logs that prove NVLS was used.
- Multi-node and RDMA behavior are not validated by this document.
- DeepSpeed gradient synchronization order is not controlled by this helper
  until a tested integration point is added.
