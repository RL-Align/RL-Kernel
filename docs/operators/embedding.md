# Token Embedding

The embedding operator maps integer token ids to their hidden-state rows — the first
layer of the Qwen3/Llama stack. It is a **WS1 ground-truth reference** (issue #108):
a pure-PyTorch definition of the "correct answer" that downstream fused CUDA/Triton
kernels are validated against.

- **Embedding** (`NativeEmbeddingOp`): `out = weight[token_ids]` — a plain row gather.

For Qwen3-8B the table is the input embedding `[vocab=151936, hidden=4096]` and is
**independent** from the lm_head weight (`tie_word_embeddings=false`) — the two weights
are not shared.

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

embedding = kernel_registry.get_op("embedding")

h = embedding(token_ids, weight)   # [B, S], [vocab, hidden]  ->  [B, S, hidden]
```

The op exposes the WS1 dual-path contract:

- `forward(...)` — gathers in the weight's native dtype, casts the gathered rows back to
  the weight dtype (Axis-B accuracy candidate / dtype-behavior path).
- `forward_fp32(...)` — native-dtype gather, then upcasts the result to fp32 (the
  ground-truth golden path).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeEmbeddingOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA / ROCm / Triton | — | — | Planned: downstream fused kernels validate against this reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `token_ids` | `[B, S]` (any shape) | integer | Index dtype; cast to int64 internally. Values in `[0, vocab)`. |
| `weight` | `[vocab, hidden]` | float (fp16/bf16/fp32) | Embedding table (Qwen3-8B `[151936, 4096]`). |
| output | `token_ids.shape + (hidden,)` | `forward`: weight dtype · `forward_fp32`: float32 | Gathered rows. |

Output dtype follows `weight` (the float operand); `token_ids` stay integer. Pure
function — no randomness, no in-place mutation, device/dtype follow the inputs.

## Dispatch Behavior

`kernel_registry.get_op("embedding")` resolves through the `OpBackend` priority map. On
`cuda` / `rocm` / `cpu` the only registered backend today is the PyTorch native op
(`PYTORCH_NATIVE_EMBEDDING`), so every device dispatches to the fp32 reference. When fused
kernels land, they are prepended to the priority list and the native op becomes the fallback.

## Accuracy

Reference semantics (`forward_fp32`):

```python
out = F.embedding(token_ids.long(), weight).to(torch.float32)
```

- **Ground truth**: `forward_fp32` gathers in the native dtype, then upcasts to fp32.
  Because a gather is a lossless row copy, this is bitwise-identical to upcasting the
  whole table first — but it never allocates a multi-GB fp32 copy of the full vocab
  table for a tiny lookup; only the gathered rows are upcast.
- **Dtype path**: `forward` runs the same gather, then casts back to the weight dtype;
  it is bitwise-equal to `forward_fp32(...).to(dtype)`.
- **Lossless gather — no accuracy drift**: a row gather performs no reduction and no
  floating-point accumulation, so the result is **bit-exact** at every dtype. There is no
  Axis-B tolerance to calibrate; the gathered rows equal direct indexing exactly.
- **Axis A — batch invariance**: each token's row is independent, so the output is
  bitwise-identical regardless of batch size or padding (`torch.equal`, `atol=0`).

## Performance Notes

Reference operator — no fused kernel or benchmark yet. Downstream fused kernels carry their
own benchmarks and are measured against this reference for correctness.

## Tests

```bash
python -m pytest tests/test_embedding.py -v
```

Covers: correctness vs direct indexing (bitwise), dtype paths, non-int64 id tolerance,
Axis-A batch invariance (slice + padding), input purity, gradient flow to `weight`
(including sparse-grad: unused rows stay zero), registry dispatch, and a GPU-only smoke
test at the real Qwen3-8B dims (`vocab=151936, hidden=4096`, boundary ids `0` and
`vocab-1`) that skips when CUDA or GPU memory is unavailable.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/linear/embedding.py`
- `rl_engine/kernels/registry.py`
- `tests/test_embedding.py`

## Known Limitations

- PyTorch fallback only; no fused CUDA/Triton backend yet (downstream work).
- Out-of-range token ids are not validated; callers must keep ids in `[0, vocab)`.
- **GPU backward is bitwise-reproducible only under deterministic algorithms.** The
  forward is a lossless gather (always reproducible), but `∂L/∂weight` is a scatter-add:
  every repeated token id (padding, common tokens) accumulates into the same `weight.grad`
  row. On CUDA that accumulation uses atomic adds, whose ordering is nondeterministic, so
  the weight gradient is not bit-exact across runs when ids collide. PyTorch documents
  `embedding` backward as a nondeterministic CUDA op for this reason. Since `forward_fp32`
  is the backward golden source, callers that need a reproducible GPU gradient must enable
  `torch.use_deterministic_algorithms(True)` (the gradient test does this). CPU backward is
  always deterministic.
