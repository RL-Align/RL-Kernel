# Shared Index Select

`index_select` takes rows from a tensor along an explicit axis. This release covers
the unique-index calls in MiniMax-H3 and BAGEL described in [#680](https://github.com/RL-Align/RL-Kernel/issues/680).

## API

```python
from rl_engine.runtime.registry import kernel_registry

select = kernel_registry.get_op("index_select", device=x.device)
y = select(x, index, dim=dim, unique_indices=True)
```

The Triton and PyTorch implementations expose the same keyword arguments on
`__call__` and `forward`. The PyTorch accuracy reference also accepts them on
`forward_fp32`.

| Argument | Contract |
| --- | --- |
| `x` | Rank of at least one, in FP32, FP16, or BF16. Strided and expanded views are supported. |
| `index` | One-dimensional int32 or int64 tensor on `x.device`. Every entry lies in `[0, x.shape[dim])`. Entries may be unsorted. |
| `dim` | Required keyword specifying the axis. Negative dimensions count from the end of `x`. |
| `unique_indices` | Pass `True` when every row number occurs at most once in this call. Omitting it or passing `False` raises `NotImplementedError` before tensor work, including in inference and on empty inputs. |
| Output | Contiguous tensor on `x.device` with `x.dtype`. It has the shape of `x` with `x.shape[dim]` replaced by `index.numel()`. |

The caller guarantees uniqueness. Both implementations trust that promise and
skip scanning for duplicates. Passing repeated indices with `True` gives
undefined gradients. The repeated-index case proposed in #680 is future work.
The same index vector applies to every coordinate outside `dim`. For an empty
index the output is empty and the input gradient is zero. A selected axis of
length zero requires an empty index. On CUDA and ROCm a bounds error can surface
at the next device synchronization.

Triton reads tensor storage directly. Callers are responsible for resolving any
lazy negative views of `x`, `index`, or the upstream gradient `grad_y`, for
example with `tensor.resolve_neg()`.

## Model calls

| Use case | Input | Selection axis | Output |
| --- | --- | --- | --- |
| H3 video/audio output after its projection | `[B, S, D_out]` | `dim=1` or `dim=-2` | `[B, K, D_out]` |
| BAGEL packed hidden states | `[S, D]` | `dim=0` | `[K, D]` |
| BAGEL packed query states | `[S, H, D_head]` | `dim=0` | `[K, H, D_head]` |

```python
# H3 selects sequence positions after projection. For [B, S, D_out], -2 is axis 1.
video = select(video_projected, video_positions, dim=-2, unique_indices=True)

# BAGEL selects packed token rows before routing them to the understanding branch.
understanding = select(packed_hidden, understanding_positions, dim=0, unique_indices=True)
```

In both examples, the indices are positions in the current tensor. Two tokens
with the same vocabulary ID still occupy distinct positions.

H3 constructs its video/audio indices from disjoint ranges of sequence positions.
The output heads consume them after each projection. The example in
`examples/models/minimax_h3/packed_index_select.py` demonstrates both calls.
[H3 index construction](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L359-L364)
and [output selection](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py#L622-L627).

BAGEL assigns each packed token its own sequence position. Concatenating text
and vision positions can change their order while preserving uniqueness. These
positions select the inputs to the understanding and generation branches.
[BAGEL packing](https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/data/dataset_base.py#L324-L369)
and [branch selection](https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/qwen2_navit.py#L419-L448).

## Computation and precision

For `index = [3, 1]`, forward copies rows 3 and 1 into output rows 0 and 1,
preserving their bits. Backward sends `grad_y[0]` to input row 3 and `grad_y[1]`
to input row 1. All other rows receive zero. PyTorch uses its built-in
`Tensor.index_select` and its autograd backward.

The Triton kernel reads strided upstream gradients directly and uses the original
indices to write each selected row of a zero-filled `grad_x` in parallel.

```text
y[j]               = x[index[j]]
grad_x[index[j]]   = cast_to_x_dtype(FP32(+0) + FP32(grad_y[j]))
grad_x[unselected] = +0
```

Row numbers in these formulas refer to the selected axis. The FP32 addition maps
an incoming negative zero to positive zero. Forward and backward support independent
batch slices, and autograd computes first-order gradients for `x`.

## Validation and benchmark

The registry entry selects Triton on CUDA, ROCm, XPU and MUSA with a matching
Triton backend. Pass `device=x.device` for dispatch. On CPU, instantiate
`rl_engine.reference.embedding.index_select.NativeIndexSelectOp` directly or
use the validation CLI below. PyTorch exposes ROCm tensors through the `cuda`
device type.
`forward_fp32` casts the input to FP32 before calling the same PyTorch gather.
This lets native autograd scatter FP32 upstream gradients before rounding the
result to the input dtype.

```bash
export PYTHONPATH=.
python -m pytest -q tests/ops/embedding/test_index_select.py tests/benchmarks/test_index_select_benchmark.py
python tools/validation/operators/check_operator.py --op index_select --candidate pytorch --device cpu --dtype bf16 --select-shape 129 4 128 --select-dim 0 --check-grad --grad-mode random
python benchmarks/operators/embedding/benchmark_index_select.py --shapes 1x129x32 1x4097x96 --dim 1 --dry-run
python benchmarks/operators/embedding/benchmark_index_select.py --shapes 1x129x32 1x38734x96 --dim 1 --dtype float32 --output reports/index-select/h3.json
python benchmarks/operators/embedding/benchmark_index_select.py --shapes 129x3584 4097x28x128 --dim 0 --dtype bfloat16 --output reports/index-select/bagel.json
```

The CUDA/ROCm benchmark uses the same random selection without replacement for
eager PyTorch, deterministic-mode PyTorch, and Triton. Before timing each one,
it checks the forward result against a CPU gather, the gradient against a CPU
FP64 scatter-add, and both against a repeated run. Triton's unique-index
gradient must also match the reference bit for bit after conversion to the
input dtype. A numerical or bitwise mismatch stops the run and is recorded in
the output JSON.

The report includes device-event and synchronized wall time for forward,
backward and combined execution, in milliseconds. Peak allocated memory is
reported in bytes above the starting allocation for a combined call. Input
construction, reference computation, compilation and warmup precede timing.
