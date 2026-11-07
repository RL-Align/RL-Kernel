# Qwen-Image TP attention output + bias (CUDA)

## Summary

`qwenimage_tp_attn_out_bias_gemm` implements the row-parallel attention output
projection in [RFC #386](https://github.com/RL-Align/RL-Kernel/issues/386).
Call it separately for image `attn.to_out[0]` and text `attn.to_add_out`; their
weights are independent. Each rank owns contiguous whole-head input channels
and the matching weight columns. The complete output and bias are replicated.

The local GEMM and weight-gradient kernels are implemented here with shared-memory
SIMT tiles. The single-GPU path and TP paths share one explicit FP32 arithmetic
contract. The CPU reference evaluates that contract independently. No unmerged
WS1 implementation is required.

## Entry Point

```python
from rl_engine.backends.cuda.model_specific.qwenimage.tp_attn_out_bias_gemm import (
    QwenImageTPAttnOutBiasGemmCudaOp,
    shard_attn_out_parameters,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

# TP1: x [M,3072], weight [3072,3072], bias [3072].
op = QwenImageTPAttnOutBiasGemmCudaOp()
with canonical_backward_session() as session:
    y = op(x, weight, bias, logical_keys=keys, parameter_id="image.attn_out")
    y.backward(upstream)
session.validate_complete()
```

For TP2/4/8, initialize the process group and set the current CUDA device before
constructing the existing `DeterministicCollective`:

```python
from rl_engine.distributed import create_deterministic_collective

with create_deterministic_collective(group=tp_group, device=x.device) as collective:
    local_weight, local_bias = shard_attn_out_parameters(
        weight, bias, tp_world=collective.world_size, tp_rank=collective.rank
    )
    width = 3072 // collective.world_size
    start = collective.rank * width
    local_x = x[:, start : start + width].detach().requires_grad_()
    local_weight = local_weight.detach().requires_grad_()
    local_bias = local_bias.detach().requires_grad_()
    op = QwenImageTPAttnOutBiasGemmCudaOp(collective)
    with canonical_backward_session() as session:
        y = op(
            local_x, local_weight, local_bias,
            logical_keys=keys, parameter_id="image.attn_out",
        )
        y.backward(upstream)
    session.validate_complete()
```

Every rank receives the same complete `upstream`, logical rows and keys. A
replicated output represents one logical loss, rather than independent losses
to sum across ranks. Input and weight gradients remain local; bias gradients
are complete on each rank and must not be summed again across TP.

Call under `torch.no_grad()` for inference. Trainable parameters require one
session covering all microbatch forwards before backward starts. Use a distinct
parameter ID per layer and stream, shared across that parameter's microbatches.

## Backends

| Backend | Entry | Role |
| --- | --- | --- |
| CUDA | `QwenImageTPAttnOutBiasGemmCudaOp` | Strict TP1/2/4/8 execution |
| CPU | `QwenImageTPAttnOutBiasGemmReference` | Independent same-tree reference |
| ROCm / Ascend / MUSA | None registered | Unsupported |

CUDA requires NVIDIA SM80 or newer. Hardware qualification is specific to the
recorded runtime and GPU; the reproduction commands below target H20 SM90.

## Tensor Contract

Let `P` be 1, 2, 4 or 8. Qwen-Image has 24 heads of 128 channels.

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[M,3072/P]` | BF16 or FP32 | M>0; local contiguous head range |
| `weight` | `[3072,3072/P]` | Same as x | Input-column shard of PyTorch `[out,in]` weight |
| `bias` | `[3072]` or None | Same as x | Complete replicated bias |
| `logical_keys` | `[M,2]` or `[M,3]` | int64 | Input device; stable logical identity; negative first key marks padding |
| output | `[M,3072]` | Same as x | Complete result replicated on every rank |

Inputs share one device. Noncontiguous tensors are materialized at the existing
CUDA entry boundaries. Ranks execute matching calls with identical logical row
layouts. Only first-order autograd is supported. Frozen parameters receive no
gradient.

## Dispatch Behavior

The CUDA registry constructs TP1 by default. Pass an existing
`DeterministicCollective` for real TP. The operator checks extension symbols,
the strict-math build probe, device, geometry and the collective's capacity.
Rows are processed in independent slices when one output payload exceeds that
capacity. Channel reduction order is unchanged by those row slices. A native
GEMM launch supports at most 1,048,560 rows; larger unsliced calls fail closed.

The CPU reference is restricted to CPU tensors. In the PyTorch 2.4.1 CI profile,
CUDA `torch.addcmul` differs from scalar `fmaf` on adversarial inputs. Production
execution therefore uses the native CUDA kernels, and CPU reference tests use
the independently checked CPU primitive. There is no CUDA reference fallback.

## Accuracy

The arithmetic contract is `qwenimage-attn-out-tp-tree`:

1. Forward uses ascending correctly-rounded FP32 FMA chains over 32 channels.
   Their roots combine through a midpoint-split tree. TP1/2/4/8 assign 96/48/24/12
   leaves per rank, so each local result is an exact subtree of the WS1 tree.
2. Existing FP32 fixed-tree AllReduce completes the upper tree. Intermediate
   results stay FP32; each rank adds the complete bias once after reduction,
   then converts to the output dtype once.
3. dX uses the same native tree over all 3072 output channels. Each rank computes
   its own input-channel gradient without communication.
4. `CanonicalBackwardSession` sorts valid parameter contributions by logical
   key. dW uses an ascending-row FMA left fold; dbias uses the existing
   FP32 addition left fold. Each parameter gradient is converted once.

Accuracy and invariance are separate gates. Small cases compare complete
outputs and gradients with the independent CPU tree reference. Real layouts
compare complete outputs and gradients byte for byte against the single-GPU CUDA path, across
TP sizes, row permutations, microbatches, padding and repeated calls. The
fixtures use seeded attention-exit values at model-derived layouts; they do
not execute the full attention block or load model weights.

## Implementation Files

- `rl_engine/backends/cuda/model_specific/qwenimage/tp_attn_out_bias_gemm.py`:
  geometry checks, local subtree calls, fixed-tree reduction and autograd.
- `rl_engine/reference/model_specific/qwenimage/tp_attn_out_bias_gemm.py`:
  weight-column sharding and independent CPU tree/gradient reference.
- `csrc/cuda/model_specific/qwenimage/attn_out.cu`: shared-memory SIMT tiles,
  FP32 register tree for forward/dX, ascending-row FMA fold for dW.
- `rl_engine/distributed/algorithms/collectives.py`: reused FP32 AllReduce.
- `rl_engine/ops/autograd/canonical_backward.py` and `vjp_fp32.py`: reused
  logical-row collection and dbias reduction.
- `rl_engine/validation/fixtures/qwenimage_attn_out.py`: both-stream seeded
  scenarios, reusing QKV token layouts and row transformations.
- `tests/distributed/tp/test_qwenimage_tp_attn_out_bias_gemm.py`: local and
  real multi-GPU validation.

## Tests

Follow `ci/scripts/run_gpu_ci.sh` for the GPU image and installation sequence.
Use PyTorch 2.4.1+cu124 and Triton 3.0.0 for the CI qualification profile.

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=4
python3.11 -m pytest tests/distributed/tp/test_qwenimage_tp_attn_out_bias_gemm.py -q
```

The real TP tests require eight peer-accessible NVIDIA GPUs to cover all
topologies. A single-GPU run skips TP2/4/8; that run alone is not distributed
qualification. NCCL bootstrap uses its default configuration.

## Benchmark

```bash
python3.11 benchmarks/operators/gemm/benchmark_tp_attn_out_bias_gemm.py \
  --rows 1 129 4096 6032 6889 --dtype bfloat16 \
  --seed 386 --warmup 3 --samples 10 --output /tmp/attn-out-local-bf16.json

python3.11 benchmarks/distributed/tp/benchmark_qwenimage_attn_out.py \
  --tp 8 --dtype bfloat16 --seed 386 --warmup 3 --samples 10 \
  --output /tmp/attn-out-tp8-bf16.json
```

Repeat with `--dtype float32`, and TP 1, 2, 4 and 8. The distributed benchmark
checks complete results against the single-GPU path before timing. It covers both streams at
1024², 1328² and 1664×928, varied text lengths and batch sizes, multiple images
per prompt and sequential/batched CFG.

Timing reports eager forward and forward+backward wall-clock latency. The
distributed statistic takes the slowest rank for each sample. JSON includes
raw samples, throughput, source hashes and allocator-visible base, peak delta
and peak total memory per rank. PyTorch is an ordinary performance control;
TP1 is the arithmetic and TP scaling control. Reports and plots are generated
outside the tracked source tree.
