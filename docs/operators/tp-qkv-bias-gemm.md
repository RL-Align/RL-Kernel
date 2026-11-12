# Qwen-Image TP QKV + bias (CUDA)

## Summary

`qwenimage_tp_qkv_bias_gemm` implements the head-sharded QKV projection in
[RFC #386](https://github.com/RL-Align/RL-Kernel/issues/386).
Call it separately for the image and text streams, which have independent weights.
The local projection and all gradients reuse the native Linear kernels merged in
[#537](https://github.com/RL-Align/RL-Kernel/pull/537). The no-activation entry
computes QKV without the AdaLN SiLU operation.

## Entry Point

```python
from rl_engine.backends.cuda.model_specific.qwenimage.tp_qkv_bias_gemm import (
    QwenImageTPQKVBiasGemmCudaOp,
    shard_qkv_parameters,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

# TP1: weight [9216,3072], bias [9216], upstream [M,9216].
op = QwenImageTPQKVBiasGemmCudaOp(collective=None)
with canonical_backward_session() as session:
    y = op(x, weight, bias, logical_keys=keys, parameter_id="image.qkv")
    y.backward(upstream)
session.validate_complete()
```

For TP2/4/8, each process uses an initialized TP process group `tp_group` and
sets its current CUDA device before constructing the collective. The example
below uses replicated `x` and full parameters on that device. `x` is a leaf
with `requires_grad=True`; `keys` identifies the same logical rows on every rank.
The caller supplies `local_upstream` with shape `[M,9216/P]` in local Q/K/V order.

```python
from rl_engine.distributed import create_deterministic_collective

with create_deterministic_collective(group=tp_group, device=x.device) as collective:
    local_weight, local_bias = shard_qkv_parameters(
        weight,
        bias,
        tp_world=collective.world_size,
        tp_rank=collective.rank,
    )
    local_weight = local_weight.detach().requires_grad_()
    if local_bias is not None:
        local_bias = local_bias.detach().requires_grad_()
    op = QwenImageTPQKVBiasGemmCudaOp(collective=collective)
    with canonical_backward_session() as session:
        local_y = op(
            x, local_weight, local_bias, logical_keys=keys, parameter_id="image.qkv"
        )
        local_y.backward(local_upstream)
    session.validate_complete()
```

The caller shards the parameters; the operator receives local weights. Backward
produces the complete `x.grad` on each rank and local parameter gradients in
`local_weight.grad` and `local_bias.grad` when bias is present. The collective
must remain open until all backward calls complete. `shard_qkv_parameters` is a
setup helper for full weights; callers with correctly packed local weights can
pass those weights directly.

For inference, call under `torch.no_grad()`; keys and a session are unnecessary.
For trainable parameters, use a stable parameter ID and logical keys on every
call. All microbatch forwards for a backward graph must register in one session
before backward starts. Use a different parameter ID for each layer and stream.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `QwenImageTPQKVBiasGemmCudaOp` | `qwenimage_det_linear_forward`, `qwenimage_det_linear_input_partials`, `qwenimage_det_linear_fold_input_partials`, `qwenimage_det_linear_backward_weight`, `qwenimage_det_linear_backward_bias` | TP1 and real TP2/4/8 qualification on H20 |
| CPU | `QwenImageTPQKVBiasGemmReference` | PyTorch `F.linear` | Accuracy reference only |
| ROCm / Ascend / MUSA | None registered | None | Unsupported |

The CUDA wrapper is in
`rl_engine/backends/cuda/model_specific/qwenimage/tp_qkv_bias_gemm.py`; its native
symbols are defined in `csrc/cuda/model_specific/qwenimage/det_linear.cu` and
exported through `rl_engine._C`. The CPU reference is in
`rl_engine/reference/model_specific/qwenimage/tp_qkv_bias_gemm.py`.

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

The CUDA implementation requires NVIDIA SM80 or newer; qualification covers H20
(SM90). Inputs share one NVIDIA CUDA device. Noncontiguous or misaligned storage is
copied through [#537]'s existing `aligned` helper. Full packed weights cannot be
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

The arithmetic is the original [#537] contract:

1. Forward keeps its dtype-specific FP32 accumulation order. Bias is added once
   before the declared output conversion.
2. dW/dbias collect valid rows in `CanonicalBackwardSession` key order, then call
   [#537]'s ascending Neumaier reducers, including dW product residuals.
3. dX uses [#537]'s compensated FP32 partials over fixed 64-output-channel chunks.
   TP gathers the partials without arithmetic, restores global `[Q;K;V]` channel
   order, then runs the original ascending Kahan fold and final conversion.

Adding already-folded rank-local dX tensors would change this rule. The existing
balanced-tree all-reduce is therefore not used for dX. Input rows are independent:
at most 128 rows are processed per call, further bounded by the existing collective
capacity. This bounds workspace without changing a row's reduction order.

Accuracy compares complete outputs and gradients with an independent CPU FP32
reference using `judgments.forward_accuracy.by_op_class.reduction` and
`judgments.gradient_accuracy.by_op_class.reduction` from the repository's
[WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/blob/test-qwenimage/rl_engine/contracts/profiles/precision/ws1.json).
The tests load these thresholds through `rl_engine.contracts.numerical.load_contract()`.

| Execution dtype | Forward atol / rtol | Gradient atol / rtol |
| --- | --- | --- |
| FP32 | `1e-4` / `1e-4` | `1e-4` / `1e-4` |
| BF16 | `5e-2` / `2e-2` | `1e-1` / `2e-2` |

Accuracy allows error within these tolerances. BI checks require identical raw
bytes, including dtype and shape: topology, row permutation, microbatch, padding
and repeat results are compared with the original native TP1 computation.
Cancellation tests reuse the merged [#537] parameter-gradient pattern and check
against CPU FP64 diagnostics.

## Performance Notes

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python3.11 \
  benchmarks/operators/gemm/benchmark_tp_qkv_bias_gemm.py \
  --rows 1 129 4096 6032 6889 --dtype bfloat16 \
  --backends pytorch native tp1 --seed 386 --warmup 3 --samples 10 \
  --output /tmp/qkv-benchmark.json
```

`native` measures the original [#537] manual VJP with canonical parameter collection.
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
PYTHONPATH=. OMP_NUM_THREADS=1 python3.11 \
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

The qualification environment follows the repository GPU CI installation steps:
Python 3.11, PyTorch 2.4.1+cu124, Triton 3.0.0 and NCCL 2.20.5. The multi-GPU
commands use the default NCCL configuration on H20.

### Model request layouts

The shared fixture in `rl_engine/validation/fixtures/qwenimage_qkv.py` derives
image-token counts from VAE scale 8 and 2x2 latent packing. It constructs seeded
image input projections and text projections, followed by normalization and
per-branch modulation, to provide QKV-entry tensors. VAE, text encoder and
attention execution remain outside these operator checks and timings.

The layout follows pinned
[Diffusers Qwen-Image](https://github.com/huggingface/diffusers/blob/40f63690da440e8c382ab53e67508835deb1ed2f/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py):
the pipeline defaults to 512 text tokens, accepts up to 1024, repeats prompts for
`num_images_per_prompt`, and executes positive/negative CFG calls separately.
The fixtures compare those separate calls with equivalent batched operator calls.

| Workload | Resolution | Prompt batch / images per prompt | Positive / negative text lengths | CFG branches |
| --- | --- | --- | --- | --- |
| `image-1024-text17` | 1024x1024 | 1 / 1 | 17 / none | 1 |
| `image-1664x928-text129` | 1664x928 | 1 / 1 | 129 / none | 1 |
| `image-1328-text512` | 1328x1328 | 1 / 1 | 512 / none | 1 |
| `batch2-mixed-text` | 1024x1024 | 2 / 1 | 17,512 / none | 1 |
| `two-images-per-prompt` | 1024x1024 | 1 / 2 | 129 / none | 1 |
| `batch2-cfg-text1024` | 1328x1328 | 2 / 1 | 512,1024 / 17,129 | 2 |

The largest batched image input is `[27556,3072]`: 2 CFG branches x 2 samples x
6889 image tokens. The largest padded text input is `[4096,3072]`, with 1682
active rows across the two branches. Logical keys carry sample, CFG branch and
token position; negative sample keys identify padding. Image and text weights
remain separate.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 \
  benchmarks/distributed/tp/benchmark_qwenimage_qkv.py \
  --tp 1 2 4 8 --dtype bfloat16 --seed 386 --warmup 3 --samples 10 \
  --workloads image-1024-text17 image-1664x928-text129 image-1328-text512 \
    batch2-mixed-text two-images-per-prompt batch2-cfg-text1024 \
  --output /tmp/qkv-model-layouts.json
```

Repeat with `--dtype float32`. Both streams are measured. CFG workloads report
batched and sequential-branch timing separately. Input construction stays outside
timing. Throughput counts active logical rows once, including each CFG branch's
own contribution. Memory includes resident local tensors for both streams and
the measured call's outputs/gradients; full parameter controls are released first.

## Tests

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python3.11 -m pytest \
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

`test_model_workloads_accuracy_and_bi` checks both streams at each named request
layout against an independent CPU FP32 reference. The real TP cases additionally
check the same workloads on every rank. Batched, sequential CFG, per-request,
microbatch, reversed, extra-padding and repeated calls preserve active Y/dX rows
and complete dW/dbias bytes in canonical sample/branch/position order.

`implementation_info()` reports configuration;
backward runtime records only native entries used by that backward call.

## Known Limitations

- TP2/4/8 transport qualification covers a single host with CUDA IPC; inter-host
  transport is outside this operator's tested scope.
- The reused [#537] dW reducer is costly for long input sequences. This adapter
  preserves its arithmetic rather than substituting the faster vLLM candidate.
- No SP, CFG-parallel, LoRA-specific integration, or full-model binding is added.
- CUDA Graph capture and higher-order gradients are not qualified.

[#537]: https://github.com/RL-Align/RL-Kernel/pull/537
