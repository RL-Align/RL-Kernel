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
Results are written under `artifacts/benchmarks/packed-index-copy/`.

Index validation sorts concatenated integer indices and performs a host check in
both native forward and backward. This synchronization cost is included in the
benchmark. Native construction uses zero initialization followed by one scatter
launch; backward gathers all three modalities in one launch. These are operator
measurements and do not establish full H3 performance.

## Tests

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pytest tests/ops/packing/test_packed_index_copy.py -q
python -m pytest tests/benchmarks/test_benchmark_packed_index_copy.py -q
python -m tools.validation.operators.check_operator --op packed_index_copy --candidate cuda \
  --device cuda --dtype bf16 --batch 2 --seq 7 --normalized-dim 257 --check-grad
```

CPU-only runs skip native CUDA cases. Optional upstream parity checks require
Diffusers with the MiniMax-H3 transformer; install the pinned source for
reproducibility:

```bash
python -m pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@f53d552036a0d1bd5570782a39cd40cfabf112bc"
```

## Known Limitations

CUDA Graph capture is unsupported because index validation synchronizes with the
host. FP64, noncontiguous sources/indices, overlapping layouts and higher-order
CUDA gradients are outside the supported contract. SM90 execution/performance and
full H3 block/trajectory integration require separate hardware evidence.
