# Packed Index Copy

## Summary

`packed_index_copy` constructs H3's packed text/video/audio sequence for
[issue #420](https://github.com/RL-Align/RL-Kernel/issues/420). It initializes the
output to zero, then copies each modality's rows to its explicit sequence indices.
Backward gathers each source's gradient from those same rows.

The dtype and copy order follow the packed-input portion of pinned Diffusers
[`MiniMaxH3Transformer3DModel.forward`](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py).
Text determines the packed dtype; video and audio are cast before copying. H3's
FP32 video/audio projections can therefore feed its BF16 packed residual stream.

## Entry Point

```python
import torch
from rl_engine.runtime.registry import kernel_registry

text = torch.randn(2, 2, 5376, device="cuda", dtype=torch.bfloat16, requires_grad=True)
video = torch.randn(2, 3, 5376, device="cuda", dtype=torch.float32, requires_grad=True)
audio = torch.randn(2, 1, 5376, device="cuda", dtype=torch.float32, requires_grad=True)
text_indices = torch.tensor([3, 0], device="cuda")
video_indices = torch.tensor([1, 5, 2], device="cuda")
audio_indices = torch.tensor([4], device="cuda")
pack = kernel_registry.get_op("packed_index_copy", device="cuda")
packed = pack(text, video, audio, text_indices, video_indices, audio_indices, 8)
packed.sum().backward()
```

Both backends expose `forward` and `forward_fp32` with argument order
`(text, video, audio, text_indices, video_indices, audio_indices, sequence_length)`.
Calling the operator uses `forward`. `forward_fp32` validates the original inputs,
then promotes all sources and returns FP32 without first quantizing video/audio to
the text dtype. Gradients return to each source's original dtype.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `PackedIndexCopyCudaOp` | `_C.packed_index_copy_forward` / `_C.packed_index_copy_backward` | NVIDIA CUDA forward and first-order backward. |
| PyTorch native | `NativePackedIndexCopyOp` | None | CPU reference; callable explicitly on CUDA as the eager baseline. |

The native kernels are part of the shared `rl_engine._C` extension. Use the normal
[source installation](../getting_started/installation.md#from-source).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `text`, `video`, `audio` | `[B, S_modality, D]` | FP16, BF16 or FP32; may differ | Contiguous, same device, batch and hidden width. |
| `*_indices` | `[S_modality]` | int64 | Contiguous, same device as sources; shared across batch. |
| `sequence_length` | Scalar | Python int | Positive packed row count. |
| output | `[B, sequence_length, D]` | Text dtype; FP32 for `forward_fp32` | New allocation; unaddressed rows are positive zero. |

Batch and hidden dimensions must be nonempty. An individual modality may have zero
rows; at least one source row is required overall. Sources are not modified.
Indices may be unsorted, but must be in range, unique within each modality, and
disjoint between modalities. This legal-layout contract is stricter than the
upstream ordered copies, which allow later modalities to overwrite earlier ones.
Duplicate or overlapping layouts are rejected before scatter/gather accesses.

## Dispatch Behavior

CUDA lookup selects `PackedIndexCopyCudaOp` and fails if the shared native symbols
are absent. CPU lookup selects `NativePackedIndexCopyOp`. ROCm, NPU and MUSA have
no registered implementation; CUDA dispatch has no eager fallback.

## Accuracy

Each output value has one writer. After the declared input dtype conversion, the
kernel copies bits without floating-point arithmetic, reductions, atomics or
additional casts. Backward is the inverse row gather and also performs no
floating-point arithmetic. PyTorch autograd carries gradients through the input
casts to the original source dtypes.

The tests distinguish four comparisons:

- **Forward and gradient accuracy:** compare against independent FP32 row
  assignment and gradients using the shared `elementwise` tolerances.
- **Batch and row invariance:** require bitwise agreement across repeated execution,
  batch placement, chunking, source and packed-row permutations, and unrelated rows.
- **Eager parity:** compare outputs and all three input gradients bitwise against
  the eager PyTorch reference with the same source dtypes.
- **Upstream parity:** call the pinned Diffusers transformer forward with identity
  projections and no transformer blocks. Compare the captured packed tensor and all
  three source gradients bitwise, including mixed dtypes and FP32 promotion.

Accuracy and invariance use the comparison roles in the
[WS1 numerical standard](../contracts/ws1-numerical-precision-standard.md).

## Performance Notes

```bash
python -m benchmarks.operators.packing.benchmark_packed_index_copy
```

The benchmark compares public CUDA and eager PyTorch wrappers on identical
GPU-resident inputs after checking inference and training outputs and all three
input gradients bitwise.
Default shapes are `1x257x5376`, `1x1025x5376`, `1x8193x5376` and `2x257x5376`,
with FP16, BF16, FP32 and mixed (BF16 text, FP32 video/audio) source profiles.

The shared `PerformanceProfiler` measures inference forward, backward on a
prebuilt retained graph, and a fresh forward plus backward. Validation,
allocation, input casts and autograd dispatch are timed; input generation and
correctness checks are outside timing.
The JSON results in `artifacts/benchmarks/packed-index-copy/results.json` include
latency, timing variation, throughput and peak allocation. `report.md` summarizes
latency and speedup. Use `--help` for workload and output options.

Index validation uses a CUDA integer bitset to check range, uniqueness and
cross-modality overlap, followed by one host status check in both native forward
and backward. Integer atomics belong only to validation; the copy kernels still
use no atomics or floating-point arithmetic. Each validation call allocates
`4 * (ceil(sequence_length / 32) + 1)` bytes of scratch, cleared before use.
The checks and synchronization are included in the benchmark. Native construction
uses zero initialization followed by one scatter launch; backward gathers all
three modalities in one launch. Wide rows with aligned pointers and row strides
use 16-byte vector copies in row tiles, avoiding per-element division. Small
widths, unaligned contiguous views and widths with tails retain the scalar path.
These are operator measurements and do not establish full H3 performance.

### Reported SM90 results

[@KJLdefeated's H200 retest on 2026-11-09](https://github.com/RL-Align/RL-Kernel/pull/545#issuecomment-6731261186)
used commit `71776fb1cf7fed7322074271b40013d3827d9123` on SM90, with driver
550.127.08, PyTorch 2.9.0+cu128, nvcc 12.4.99, GCC 12.5.0 and Python 3.12.12.
The retest exited successfully, and optimized CUDA was faster than eager PyTorch
in all 48 benchmark cases across the four shapes and dtype profiles above.

| Mode | Eager / optimized CUDA | Optimized cases faster than eager | Optimized CUDA median |
| --- | --- | --- | --- |
| Forward | 2.26–4.26x | 16/16 | 0.057–0.168 ms |
| Backward | 1.17–3.03x | 16/16 | 0.086–0.184 ms |
| Forward + backward | 1.64–3.81x | 16/16 | 0.163–0.319 ms |

These timings measure the public operator API, including validation and
synchronization.

## Tests

Optional upstream parity checks require Diffusers with the MiniMax-H3 transformer.
Install the pinned reference before running the tests for reproducibility:

```bash
python -m pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@f53d552036a0d1bd5570782a39cd40cfabf112bc"
RL_KERNEL_REQUIRE_EXT=1 python -m pytest tests/ops/packing/test_packed_index_copy.py -q
python -m pytest tests/benchmarks/test_benchmark_packed_index_copy.py -q
python -m tools.validation.operators.check_operator --op packed_index_copy --candidate cuda \
  --device cuda --dtype bf16 --batch 2 --seq 7 --normalized-dim 257 --check-grad
```

Coverage includes input validation, empty and sparse layouts, mixed-dtype casts,
forward/backward accuracy, batch and row invariance, registry dispatch, vector
and scalar copy paths, and CUDA stream ordering. CPU-only runs skip native CUDA
cases. Upstream parity tests skip when the Diffusers MiniMax-H3 transformer is
unavailable.

## Known Limitations

CUDA Graph capture is unsupported because index validation synchronizes with the
host. FP64, noncontiguous sources/indices, overlapping layouts and higher-order
CUDA gradients are outside the supported contract. The reported SM90 operator
results do not establish full H3 block/trajectory integration.

## Implementation Files

- `rl_engine/reference/packing/packed_index_copy.py` — PyTorch reference and shared input validation
- `rl_engine/backends/cuda/packing/packed_index_copy.py` — CUDA autograd wrapper
- `csrc/cuda/packing/packed_index_copy.cu` — CUDA validation and forward/backward kernels
- `benchmarks/operators/packing/benchmark_packed_index_copy.py` — public-wrapper benchmark
