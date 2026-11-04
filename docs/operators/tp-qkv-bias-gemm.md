# Qwen-Image TP QKV + bias (CUDA)

## Summary

`qwenimage_tp_qkv_bias_gemm` implements RFC #386's head-sharded QKV projection.
Call it separately for the image and text streams, which have independent weights.
The local projection and all gradients reuse the native Linear kernels merged in
#537. The no-activation entry computes QKV without the AdaLN SiLU operation.

## Entry Point

```python
from rl_engine.backends.cuda.model_specific.qwenimage.tp_qkv_bias_gemm import (
    QwenImageTPQKVBiasGemmCudaOp,
    shard_qkv_parameters,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

# TP1: full weights. For TP>1, pass the existing CUDA DeterministicCollective
# and use shard_qkv_parameters with its world_size and group-local rank.
op = QwenImageTPQKVBiasGemmCudaOp(collective=None)
with canonical_backward_session() as session:
    y = op(x, weight, bias, logical_keys=keys, parameter_id="image.qkv")
    y.backward(upstream)
session.validate_complete()
```

For inference, call under `torch.no_grad()`; keys and a session are unnecessary.
For trainable parameters, use a stable parameter ID and logical keys on every
call. All microbatch forwards for a backward graph must register in one session
before backward starts. Use a different parameter ID for each layer and stream.

## Backends

| Backend | Implementation | Status |
| --- | --- | --- |
| CUDA | #537 native Linear, compensated partials and fold | TP1 and real TP2/4/8 qualification on H20 |
| CPU | Independent PyTorch Linear | Accuracy reference only |
| ROCm / Ascend / MUSA | No candidate registered | Unsupported |

## Tensor Contract

Let `P` be TP size, one of 1/2/4/8. There are 24 heads of 128 channels.
Each rank owns the same contiguous `24/P` heads in Q, K and V.

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[M,3072]` | BF16 or FP32 | M>0, replicated input on TP ranks |
| `weight` | `[9216/P,3072]` | Same as x | Local `[Q_rank;K_rank;V_rank]` rows |
| `bias` | `[9216/P]` or None | Same as x | Owned by the local output shard |
| `logical_keys` | `[M,2]` or `[M,3]` | int64 | Input device; stable logical identity, negative first key marks padding |
| output | `[M,9216/P]` | Same as x | Local Q/K/V head features |

Inputs share one NVIDIA CUDA device. Noncontiguous or misaligned storage is
copied through #537's existing `aligned` helper. Full packed weights cannot be
split into P contiguous pieces once: each rank needs a piece of each Q/K/V section.

The collective must describe the same group-local rank and device as the local
weight shard. Every rank must execute matching calls with replicated rows, the
same logical keys, and the same input-gradient requirement. First-order autograd
is supported. Frozen parameters receive no gradient.

## Dispatch Behavior

The CUDA registry entry constructs TP1 by default. Construct the same class with
an existing `DeterministicCollective` for TP. It reuses that object's byte-exact
`all_gather`, device, world size and capacity. The CUDA entry fails when the native
extension is missing, built with fast math, or lacks the new partial/fold symbols.
The CPU reference is separate and is not a CUDA fallback.

The existing collective custom-op registrations use explicit schemas so PyTorch
2.4 can load the module despite postponed type annotations. Their mutation and
output contracts are unchanged.

## Accuracy

The arithmetic is the original #537 contract:

1. Forward keeps its dtype-specific FP32 accumulation order. Bias is added once
   before the declared output conversion.
2. dW/dbias collect valid rows in `CanonicalBackwardSession` key order, then call
   #537's ascending Neumaier reducers, including dW product residuals.
3. dX uses #537's compensated FP32 partials over fixed 64-output-channel chunks.
   TP gathers the partials without arithmetic, restores global `[Q;K;V]` channel
   order, then runs the original ascending Kahan fold and final conversion.

Adding already-folded rank-local dX tensors would change this rule. The existing
balanced-tree all-reduce is therefore not used for dX. Input rows are independent:
at most 128 rows are processed per call, further bounded by the existing collective
capacity. This bounds workspace without changing a row's reduction order.

Accuracy compares complete outputs and gradients with an independent CPU FP32
reference using the repository's `reduction` forward/gradient thresholds. Raw-byte
checks separately compare topology, row permutation, microbatch, padding and
repeat results with the original native TP1 computation. Cancellation tests reuse
the merged #537 parameter-gradient pattern and check against CPU FP64 diagnostics.

## Performance Notes

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python3.11 \
  benchmarks/operators/gemm/benchmark_tp_qkv_bias_gemm.py \
  --rows 1 129 4096 6032 6889 --dtype bfloat16 \
  --backends pytorch native tp1 --seed 386 --warmup 3 --samples 10 \
  --output /tmp/qkv-benchmark.json
```

`native` measures the original #537 manual VJP with canonical parameter collection.
`tp1` measures the public adapter's autograd with the same collection. `pytorch`
is an ordinary Linear/autograd performance reference. The command checks adapter
bytes against native before timing. JSON includes raw samples, peak additional
allocated memory and rows/second. It contains no model weights or downloads.
These are local single-GPU timings, including Python calls and conversions.
The benchmark disables TF32 and BF16 reduced-precision reduction for the PyTorch
control to retain FP32 accumulation.
They do not measure TP transport or multi-GPU throughput.

For real single-host TP transport and throughput, use one process per GPU:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 python3.11 \
  benchmarks/distributed/tp/benchmark_qwenimage_qkv.py \
  --tp 1 2 4 8 --rows 1 129 4096 6032 6889 --dtype bfloat16 \
  --seed 386 --warmup 3 --samples 10 --output /tmp/qkv-real-tp.json
```

Repeat with `--dtype float32`. This command checks Y/dX/dW/dbias bytes against
native TP1 before timing. Forward and forward+backward include the public adapter;
backward includes CUDA IPC transport and canonical parameter reduction. Each
sample reports the slowest rank's elapsed time. Effective rows/second counts the
replicated input rows once, and memory reports the largest per-rank allocation.
The benchmark retains only local parameter and upstream shards during timing.

On the measured H20 profile (driver 535.183.06, NCCL 2.20.5), NCCL's default NVLS
initialization failed in a standalone PyTorch four-rank `all_gather`, before
loading RL-Kernel. Adding `NCCL_NVLS_ENABLE=0` to the command environment allowed
the standalone four/eight-rank probes and the tests below to execute. This is
NVIDIA's existing [NVLS control](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable);
use it for that runtime profile. The operator does not set it.

## Tests

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 python3.11 -m pytest \
  tests/distributed/tp/test_qwenimage_tp_qkv_bias_gemm.py -q

# Regress the existing #537 callers after exposing the two native stages.
PYTHONPATH=. OMP_NUM_THREADS=4 python3.11 -m pytest \
  tests/models/qwenimage/test_adaln.py \
  tests/models/qwenimage/test_parameter_reduction.py \
  tests/models/qwenimage/test_reference.py -q
```

The QKV suite includes FP32/BF16, two seeded parameter sets, M=1/17/129/4096/6032/6889,
complete gradient checks, row layouts, parameter freezing, noncontiguous upstream,
two independent streams, parameter/head mapping, and simulated TP2/4/8.
The single-rank IPC integration test also constructs the existing CUDA collective
and executes its native `all_gather`; it does not establish cross-GPU transport.
Simulation runs the actual shard kernels on one GPU and concatenates peer partials
in a test transport. It establishes arithmetic and mapping, not real IPC or
multi-process correctness.

The `test_real_tp_matches_native_all_gradients` cases require 2/4/8 physical GPUs
and run separate NCCL ranks with the native CUDA IPC collective. Each TP size
checks all six row counts, both dtypes and seeds, both stream parameter sets,
reverse order, microbatching, padding, repeat execution and inference. Every rank
compares complete Y/dX/dW/dbias bytes with native TP1, including the communicated
global dX. With fewer GPUs, only the unavailable TP sizes are skipped.

`implementation_info()` reports configuration;
backward runtime records only native entries used by that backward call.

## Known Limitations

- TP2/4/8 transport qualification covers a single host with CUDA IPC; inter-host
  transport is outside this operator's tested scope.
- The reused #537 dW reducer is costly for long input sequences. This adapter
  preserves its arithmetic rather than substituting the faster vLLM candidate.
- No SP, CFG-parallel, LoRA-specific integration, or full-model binding is added.
- CUDA Graph capture and higher-order gradients are not qualified.
