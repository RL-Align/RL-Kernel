# Deterministic All-Reduce

RL-Kernel provides a small all-reduce helper for distributed smoke tests and
future WS2 integration work. It has two modes:

- `torch_all_reduce`: calls `torch.distributed.all_reduce`.
- `ordered_rank_reference`: gathers all rank tensors, accumulates them on process-group rank 0 in process-group rank order, then broadcasts the result.

The helper reduces the input tensor in place and returns it.

## Contract

Results are expected to be stable only when the world size, process-group rank
order, inputs, dtype, operation, backend, and environment are unchanged.

`op="mean"` performs a sum and divides by world size at a fixed point. Integer
tensors are rejected for `mean`.

## Ordered-Rank Reference

`ordered_rank_reference` is a reference path, not a high-performance transport.
It uses `all_gather` and `broadcast`, so the active backend must support those
collectives for the tensor device. The operation order is:

1. make each rank input contiguous;
2. gather tensors in process-group rank order;
3. accumulate on process-group rank 0 in that order;
4. optionally accumulate floating-point inputs in FP32;
5. divide once for `op="mean"`;
6. broadcast from process-group rank 0.

This mode is meant for small tensors in tests, debug runs, and reference
comparisons.

## Torch All-Reduce

`torch_all_reduce` is a thin wrapper around `torch.distributed.all_reduce`. For
NCCL runs, callers may set best-effort ring settings before process-group
initialization:

```python
from rl_engine.distributed import configure_deterministic_nccl_env

configure_deterministic_nccl_env(overwrite=True)
```

The helper writes:

```bash
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
```

These settings do not prove bitwise determinism. Validate on the target machine
before making a hardware-specific claim.

## Behavior

- `world_size == 1`: returns the input tensor unchanged.
- no initialized process group and `WORLD_SIZE <= 1`: returns the input tensor unchanged.
- no initialized process group and `WORLD_SIZE > 1`: raises `RuntimeError`.
- `async_op=True`: raises `NotImplementedError`.

## Smoke Tests

Unit and CPU/Gloo smoke checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/distributed/test_deterministic_allreduce.py -q
```

Manual NCCL all-reduce smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
NCCL_MIN_NCHANNELS=1 \
NCCL_MAX_NCHANNELS=1 \
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/test_deterministic_allreduce.py \
  --backend nccl --mode torch_all_reduce --dtype fp32 --device cuda
```

DP gradient smoke compares a fixed DP=1 full-batch gradient with DP=N local
gradients reduced by this helper:

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/test_dp_gradient_determinism.py \
  --backend gloo --mode ordered_rank_reference --dtype fp32 --device cpu
```

## Limitations

- NVLS / NVLink-Sharp is not implemented or claimed here.
- Multi-node and RDMA behavior are not validated here.
- DeepSpeed gradient synchronization is not controlled by this helper yet.
