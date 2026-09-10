# Token Embedding

The embedding operator maps integer token ids to their hidden-state rows, the first
layer of the Qwen3/Llama stack. It is a **WS1 ground-truth reference** (issue #108):
a pure-PyTorch definition of the "correct answer" that downstream fused CUDA/Triton
kernels are validated against.

- **Embedding** (`NativeEmbeddingOp`): `out = weight[token_ids]`, a plain row gather.

For Qwen3-8B the table is the input embedding `[vocab=151936, hidden=4096]` and is
**independent** from the lm_head weight (`tie_word_embeddings=false`); the two weights
are not shared.

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

embedding = kernel_registry.get_op("embedding")

h = embedding(token_ids, weight)   # [B, S], [vocab, hidden]  ->  [B, S, hidden]
```

The op exposes the WS1 dual-path contract:

- `forward(...)` gathers in the weight's native dtype, casts the gathered rows back to
  the weight dtype (Axis-B accuracy candidate / dtype-behavior path).
- `forward_fp32(...)` uses native-dtype gather, then upcasts the result to fp32 (the
  ground-truth golden path).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeEmbeddingOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA SM90 (H200/Hopper) | `SM90EmbeddingOp` | `_C.embedding_sm90_forward` | Single-card batch-invariant forward backend; deterministic duplicate-id backward in the wrapper. |
| Ascend NPU | `AscendEmbeddingOp` | `_C_npu.embedding_ascend` | Batch-invariant Ascend C forward (pure row copy); reuses the SM90 op's deterministic sorted-segment backward. |
| Triton | `TritonEmbeddingOp` | `_embedding_fwd`, `_embedding_bwd` | CUDA gather with deterministic, atomic-free sorted-segment backward. |
| ROCm | N/A | N/A | Falls back to the PyTorch native reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `token_ids` | `[B, S]` (any shape) | integer | Index dtype; cast to int64 internally. Values in `[0, vocab)`. |
| `weight` | `[vocab, hidden]` | float (fp16/bf16/fp32) | Embedding table (Qwen3-8B `[151936, 4096]`). |
| output | `token_ids.shape + (hidden,)` | `forward`: weight dtype; `forward_fp32`: float32 | Gathered rows. |

Output dtype follows `weight` (the float operand); `token_ids` stay integer. Pure
function: no randomness, no in-place mutation, device/dtype follow the inputs.

## Dispatch Behavior

`kernel_registry.get_op("embedding")` resolves through the `OpBackend` priority map. On
CPU, ROCm, and CUDA devices without the SM90 extension, dispatch uses the PyTorch native op
(`PYTORCH_NATIVE_EMBEDDING`). On H200/Hopper-class builds that expose `_C.embedding_sm90_forward`,
the CUDA SM90 single-card batch-invariant backend is prepended and the native op remains
the fallback.

On `npu` the priority is:

1. `ASCEND_EMBEDDING` — `AscendEmbeddingOp` (batch-invariant Ascend C forward, bf16/fp16/fp32).
2. `PYTORCH_NATIVE_EMBEDDING` — `NativeEmbeddingOp` (fallback).

The Ascend kernel implements the same semantics as the SM90 CUDA kernel: a pure row
gather (`out[t, :] = weight[token_ids[t], :]`). Every token row is copied end-to-end by
exactly one AI-core block with a fixed tile size, so the copy sequence for a row depends
only on `hidden`, never on the token count or block assignment. Because the copy performs
no arithmetic, the Ascend output is **bitwise identical** to the CUDA kernel (and to the
PyTorch reference) for identical inputs at every supported dtype; the fp32-output path
upcasts the gathered rows afterwards, which is exact for bf16/fp16. The backward reuses the
SM90 op's deterministic sorted-segment dweight (stable-sorted ids, fixed addition order),
so duplicate-id gradients match the CUDA op bit for bit.

## Accuracy

Reference semantics (`forward_fp32`):

```python
out = F.embedding(token_ids.long(), weight).to(torch.float32)
```

- **Ground truth**: `forward_fp32` gathers in the native dtype, then upcasts to fp32.
  Because a gather is a lossless row copy, this is bitwise-identical to upcasting the
  whole table first, but it never allocates a multi-GB fp32 copy of the full vocab
  table for a tiny lookup; only the gathered rows are upcast.
- **Dtype path**: `forward` runs the same gather, then casts back to the weight dtype;
  it is bitwise-equal to `forward_fp32(...).to(dtype)`.
- **Lossless gather, no accuracy drift**: a row gather performs no reduction and no
  floating-point accumulation, so the result is **bit-exact** at every dtype. There is no
  Axis-B tolerance to calibrate; the gathered rows equal direct indexing exactly.
- **Axis A batch invariance**: each token's row is independent, so the output is
  bitwise-identical regardless of batch size or padding (`torch.equal`, `atol=0`).

## Performance Notes

The SM90 backend is a simple single-card forward gather for H200/Hopper builds. It is
not a TP/vocab-parallel integration path; downstream fused kernels carry their own
benchmarks and are measured against the PyTorch reference for correctness. Its backward
path is intentionally conservative: token ids are sorted, duplicate ids are reduced in
a fixed order, and only unique rows are written back. That avoids CUDA atomic-add
nondeterminism for repeated token ids at the cost of throughput.

## Tests

```bash
python -m pytest \
  tests/test_embedding.py \
  tests/test_triton_embedding.py \
  tests/test_canonical_embedding.py \
  tests/test_embedding_ascend.py -v
```

Covers: correctness vs direct indexing (bitwise), dtype paths, non-int64 id tolerance,
Axis-A batch invariance (slice + padding), input purity, gradient flow to `weight`
(including sparse-grad: unused rows stay zero), registry dispatch, and a GPU-only smoke
test at the real Qwen3-8B dims (`vocab=151936, hidden=4096`, boundary ids `0` and
`vocab-1`) that skips when CUDA or GPU memory is unavailable. Additional tests cover the
Triton sorted-segment backward and canonical logical-row ordering.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/linear/embedding.py`
- `rl_engine/kernels/ops/triton/linear/embedding.py`
- `rl_engine/kernels/ops/cuda/linear/embedding.py`
- `rl_engine/kernels/ops/canonical_embedding.py`
- `csrc/cuda/embedding_lm_head_sm90.cu`
- `rl_engine/kernels/ops/ascend/linear/embedding.py` — Ascend deterministic op
- `csrc/ascend/embedding_ascend.asc` — Ascend C forward kernel
- `csrc/ascend/npu_module.cpp` — shared pybind entry for `rl_engine._C_npu`
- `rl_engine/kernels/registry.py`
- `tests/test_embedding.py`
- `tests/test_triton_embedding.py`
- `tests/test_canonical_embedding.py`
- `tests/test_embedding_ascend.py`

## Known Limitations

- The CUDA SM90 and Triton paths are single-card coverage, not TP/vocab-parallel
  integration paths.
- Token ids must be in `[0, vocab)`. The Triton fast path reports invalid ids
  asynchronously; use its explicit validator when the error must be raised at the call
  boundary.
- The deterministic SM90 backward is a reference path, not a tuned training kernel.
- The standalone operator has no logical-row metadata, so its deterministic guarantee is
  limited to a fixed flattened order. Cross-layout and cross-chunk invariance requires the
  canonical Qwen training wrapper and its logical keys.
