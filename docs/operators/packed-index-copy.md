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

Tests separate independent FP32 accuracy under the shared WS1 `elementwise`
tolerances from byte equality with the same-dtype eager path. Repetition, batch
placement, chunking, source/packed-row permutations and unrelated rows must
preserve bytes. The independent accuracy and invariance comparison roles follow
the [WS1 numerical standard](../contracts/ws1-numerical-precision-standard.md).

## Performance Notes

```bash
python -m benchmarks.operators.packing.benchmark_packed_index_copy
```

The benchmark compares public CUDA and eager PyTorch wrappers after checking
output and all three source gradients. The shared `PerformanceProfiler` measures
forward, backward on a prebuilt graph, and forward plus backward. Validation,
allocation, input casts and autograd dispatch are inside timing; input generation
and correctness checks are outside. See `--help` for workloads and output options.
Default shapes are `1x257x5376`, `1x1025x5376`, `1x8193x5376` and `2x257x5376`,
with FP16, BF16, FP32 and mixed (BF16 text, FP32 video/audio) source profiles.
`results.json` and `report.md` are written under
`artifacts/benchmarks/packed-index-copy/`.

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

### Local optimization results

An RTX 2060 (SM75) comparison of the original CUDA implementation at `32fce8bb`
and the optimized implementation at `2f67bc3e` passed all 48 benchmark cases,
using 25 warmups and 100 measured calls. The per-workload original/optimized CUDA
latency ratios had medians of 2.02x forward, 1.87x backward and 1.94x combined.
One small BF16 backward case was approximately tied with eager (1.005x).
See the [experiment report and per-case summaries](https://github.com/RL-Align/RL-Kernel/tree/test-h3/reports/experiments/packed-index-copy)
for source hashes, variation, allocation measurements and timing limitations.
These focused-extension SM75 results do not validate the optimized path on H200.

### Reported SM90 results

[@KJLdefeated reported an H200 run on 2026-10-31](https://github.com/RL-Align/RL-Kernel/pull/545#issuecomment-6542672236):
SM90, driver 550.127.08, PyTorch 2.9.0+cu128, nvcc 12.4.99, GCC 12.5.0 and
Python 3.12.12. The normal shared-extension build/import passed, as did 211
operator/benchmark tests with no skips and BF16/FP16/FP32 output/gradient checks.
All 48 benchmark cases passed bitwise output/gradient checks with no fallback,
using 25 warmups and 100 measured calls across four dtype profiles and four shapes.

| Mode | Eager PyTorch / CUDA speedup | Workloads faster than eager | CUDA median |
| --- | --- | --- | --- |
| Forward | 1.09–1.53x | 16/16 | 0.124–0.458 ms |
| Backward | 0.65–1.00x | 0/16 | 0.160–0.517 ms |
| Forward + backward | 0.89–1.30x | 1/16 | 0.320–0.931 ms |

Speedup below 1 means CUDA is slower. Backward was at best break-even; combined
execution improved only for FP32 `1x8193x5376`. The public-wrapper timings include
index validation and host synchronization, so they do not isolate scatter/gather
throughput or establish a general training speedup.
The run preceded the rebase and bitset-validation optimization and did not record
the tested commit. Raw per-case results are unavailable, and the optimized head
still requires SM90 validation.

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

CPU-only runs skip native CUDA cases. Upstream parity tests skip when the
Diffusers MiniMax-H3 transformer is unavailable.

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
