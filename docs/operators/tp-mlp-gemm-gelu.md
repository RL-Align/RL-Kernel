# Qwen-Image TP MLP GEMM GELU (CUDA)

## Summary

`qwenimage_tp_mlp_gemm_gelu` applies one stream's 3072 → 12288 → 3072 MLP
with biases and tanh-approximate GELU. Image and text streams use separate
parameters. CUDA supports BF16 on NVIDIA SM80 or newer. The qualification
profile uses H20 and the repository's PyTorch 2.4.1 CUDA CI environment.

## Entry Point

```python
import torch

from rl_engine.backends.cuda.model_specific.qwenimage.tp_mlp_gemm_gelu import (
    QwenImageTPMlpGemmGeluCudaOp,
    shard_mlp_parameters,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

op = QwenImageTPMlpGemmGeluCudaOp()  # TP1
with canonical_backward_session() as session:
    y = op(
        x, up_weight, up_bias, down_weight, down_bias,
        logical_keys=keys, parameter_id="block.0.image.mlp",
    )
    y.backward(upstream)
session.validate_complete()
```

For TP2/4/8, initialize the process group, set the current CUDA device, and
pass an existing `DeterministicCollective` to the operator. Use
`shard_mlp_parameters(..., tp_world=P, tp_rank=rank)` to slice matching up
rows and down columns. The down bias is replicated. Inputs, upstream gradients,
logical rows and call order are replicated across the group.

Use distinct parameter IDs for layers and streams. Reuse a parameter ID across
microbatches sharing the same weights. All such calls must finish their forward
registration before backward begins in the shared session.

## Backends

| Backend | Entry | Role |
| --- | --- | --- |
| CUDA | `QwenImageTPMlpGemmGeluCudaOp` | Strict BF16 TP1/2/4/8 execution |
| CPU | `QwenImageTPMlpGemmGeluReference` | Independent FP32 arithmetic reference |
| ROCm / Ascend / MUSA | None registered | Unsupported |

## Tensor Contract

Let `D=3072`, `I=12288`, and `P` be 1, 2, 4 or 8.

| Argument | Per-rank shape | Meaning |
| --- | --- | --- |
| `x` | `[M,D]` | Replicated input rows |
| `up_weight` | `[I/P,D]` | Local intermediate-channel rows |
| `up_bias` | `[I/P]` | Local intermediate-channel bias |
| `down_weight` | `[D,I/P]` | Matching intermediate-channel columns |
| `down_bias` | `[D]` | Replicated output bias |
| `logical_keys` | `[M,2]` or `[M,3]` | Int64 stable row identity |
| output | `[M,D]` | Replicated BF16 output |

The row limit is `1 <= M <= 4194240`, from the native two-dimensional CUDA grid.

Inputs and parameters must share a CUDA device and BF16 dtype. Logical keys
use the same device. A negative first key marks padding. Trainable parameters
require an active canonical backward session and stable keys. Inference under
`torch.no_grad()` requires neither keys nor a session. Noncontiguous operands
are materialized at the adapter boundary. The operation supports first-order
gradients and individually frozen parameters. Distributed autograd requires a
canonical session even when only `x` is trainable. Within each session, the
adapter adds an autograd dependency between successive uses of the same
collective. This fixes reverse communication order across independent requests,
whose equal shapes alone cannot identify them. The edge contributes no numerical
gradient. Its retained tail is released when backward begins.

## Accuracy

The portable CUDA source and FP32 CPU primitives reuse the fixed-tree
implementation from [#541](https://github.com/RL-Align/RL-Kernel/pull/541),
reviewed at `38a41ec1b9a539d263378d73e31820b9dcc5c859`. The down-projection
arithmetic was checked against [#499](https://github.com/RL-Align/RL-Kernel/pull/499)
at `61cae1ec9118a6ad1ca1ad5a4aac9f8ea9e585a1`.

1. Up projection accumulates ascending 32-channel FP32 FMA leaves and merges
   them through the WS1 midpoint tree. Add the local bias once in FP32.
2. Apply the original pinned GELU sequence to the FP32 pre-activation, then
   cast to BF16. Training saves the FP32 pre-activation for its derivative.
3. Down projection returns FP32 subtree roots through the added
   `mlp_up_gemm_gelu_cuda_tree_fp32` entry. The existing fixed-tree AllReduce
   completes the global tree. Add the down bias once, then cast to BF16.
4. Backward computes local `dH`, casts at the WS1 boundary, and applies the
   original GELU gradient gate. Up-projection `dX` uses the same FP32-root
   entry and global tree completion before its BF16 cast.
5. `CanonicalBackwardSession` collects valid parameter-gradient rows in
   logical order. The original `dw` and `db` entries evaluate the ordered
   per-row FP32 folds for the up and down parameters.

CPU checks use explicit FP32 FMA/tree primitives. GEMM contractions are checked
byte for byte on selected entries. GELU and ordinary FP32 contraction checks
use `atol=0.05` for pre-activations/forward values and `atol=0.1` for gradients,
with `rtol=0.02`. The complete CUDA outputs and all five gradients must be
byte-equal across the qualified logical-row and TP layouts.

The FP32-root entry uses the existing tree kernel with activation disabled
and its BF16 destination omitted. Existing forward and backward entries retain
their signatures and output behavior. Row chunks respect collective capacity
while retaining each row's arithmetic order.

## Dispatch Behavior

The CUDA adapter lives in
`rl_engine/backends/cuda/model_specific/qwenimage/tp_mlp_gemm_gelu.py`.
The CPU reference and geometry live in the corresponding `reference` package.
Native computation uses `csrc/cuda/gemm/mlp_up_gemm_gelu.cu` and its GELU header,
with bindings in `csrc/bindings/ops.cpp` and types in `rl_engine/_C.pyi`.
The registry constructs CUDA TP1 by default and registers the CPU reference
for CPU execution. Explicit distributed callers supply their collective.
Unsupported backends, dtypes, shapes and unavailable native entries fail closed.
`implementation_info()` describes the implementation configuration. Backward
records the native reducers actually invoked through the existing trace API.

## Tests

Install in the image and order defined by `ci/scripts/run_gpu_ci.sh`.
Use PyTorch 2.4.1+cu124, Triton 3.0.0 and eight peer-accessible H20 GPUs for the
complete real TP matrix. NCCL uses its default configuration.

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3.11 -m pytest tests/distributed/tp/test_qwenimage_tp_mlp_gemm_gelu.py -q

python3.11 benchmarks/operators/gemm/benchmark_tp_mlp_gemm_gelu.py \
  --rows 1 129 4096 6032 6889 --seed 386 --warmup 3 --samples 10 \
  --output /tmp/mlp-local.json

for tp in 1 2 4 8; do
  python3.11 benchmarks/distributed/tp/benchmark_qwenimage_mlp.py \
    --tp "$tp" --seed 386 --warmup 3 --samples 10 \
    --output "/tmp/mlp-tp$tp.json"
done
python3.11 benchmarks/distributed/tp/benchmark_qwenimage_mlp.py \
  --tp 1 --backend pytorch --seed 386 --warmup 3 --samples 10 \
  --output /tmp/mlp-pytorch.json
```

Tests cover independent CPU precision, FP32-root equivalence, logical-row BI,
frozen parameters, adversarial rank-local backward priorities and real TP2/4/8 across both streams and the seven request
layouts shared by the QKV and attention-output suites. Shapes include
1024², 1328², 1664×928, varied text lengths, multiple prompts, multiple images
per prompt and batched/sequential CFG.

## Performance Notes

The benchmark commands above cover both streams at seven request scenarios.
Benchmarks verify numerical results before timing. Each distributed sample
reports the slowest rank. JSON includes raw synchronized wall-clock samples,
throughput, source hashes and per-rank allocated memory. Input construction and
output concatenation run outside timing. Ordinary PyTorch is the performance
control. Reports and plots are saved outside the tracked source tree.

## Known Limitations

The runtime accepts BF16 only and supports first-order gradients. The CPU
reference evaluates FP32 arithmetic; it is not a performant training backend.
TP groups must use 1, 2, 4 or 8 peer-accessible NVIDIA GPUs with identical rows,
call order and upstream gradients. Local parameter gradients are owned by their
channel shard; the complete down-bias gradient is replicated on every rank.
Each replicated output represents one logical loss. Do not sum its down-bias
gradient again across TP. Other TP sizes, general MLP shapes and second-order
gradients are outside this contract.
