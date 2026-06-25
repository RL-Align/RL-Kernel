# LM Head

The lm_head operator projects hidden states back to vocabulary logits — the final
layer of the Qwen3/Llama stack. It is a **WS1 ground-truth reference** (issue #108):
a pure-PyTorch definition of the "correct answer" that downstream fused CUDA/Triton
kernels are validated against.

- **LM Head** (`NativeLMHeadOp`): `out = hidden @ weight.t() (+ bias)`.

For Qwen3-8B the weight is the output projection `[vocab=151936, hidden=4096]` in the
HF `nn.Linear` `[out, in]` convention, so it is transposed internally. It is
**independent** from the embedding table (`tie_word_embeddings=false`) — the two
weights are not shared — and Qwen3 has **no bias** (`bias=None`).

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

lm_head = kernel_registry.get_op("lm_head")

logits = lm_head(hidden, weight)          # [B, S, hidden], [vocab, hidden] -> [B, S, vocab]
logits = lm_head(hidden, weight, bias=b)  # optional [vocab] bias
```

The op exposes the WS1 dual-path contract:

- `forward(...)` — projects in the input dtype, returns the input dtype (Axis-B accuracy
  candidate / dtype-behavior path).
- `forward_fp32(...)` — upcasts to fp32, accumulates in fp32, returns fp32 (the
  ground-truth golden path). The matmul runs with autocast disabled and CUDA TF32
  turned off, so it stays a true fp32 reference regardless of the caller's ambient
  precision context (the global `allow_tf32` flag is saved and restored around it).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeLMHeadOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA / ROCm / Triton | — | — | Planned: downstream fused kernels validate against this reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `hidden` | `[B, S, hidden]` (any leading dims) | float (fp16/bf16/fp32) | Hidden states (Qwen3-8B `hidden=4096`). |
| `weight` | `[vocab, hidden]` | float | Output projection in HF `[out, in]` layout; transposed internally. Qwen3-8B `[151936, 4096]`. |
| `bias` | `[vocab]` or `None` | float | Optional; Qwen3 has none (`None`). |
| output | `hidden.shape[:-1] + (vocab,)` | `forward`: hidden dtype · `forward_fp32`: float32 | Logits. |

Output dtype follows `hidden`. Pure function — no randomness, no in-place mutation,
device/dtype follow the inputs.

> **Difference from the bare `matmul` op**: lm_head takes the weight in HF `[out, in]`
> layout and transposes it internally (`weight.t()`); the `matmul` op computes a bare
> `a @ b` with no transpose. Do not use them interchangeably.

## Dispatch Behavior

`kernel_registry.get_op("lm_head")` resolves through the `OpBackend` priority map. On
`cuda` / `rocm` / `cpu` the only registered backend today is the PyTorch native op
(`PYTORCH_NATIVE_LM_HEAD`), so every device dispatches to the fp32 reference. When fused
kernels land, they are prepended to the priority list and the native op becomes the fallback.

## Accuracy

Reference semantics (`forward_fp32`):

```python
out = hidden.float() @ weight.float().t()
if bias is not None:
    out = out + bias.float()
```

- **Ground truth**: `forward_fp32` accumulates in and returns fp32, with autocast and
  CUDA TF32 disabled so it is a true fp32 reference even if the caller has TF32 or
  autocast enabled.
- **Dtype path**: `forward` runs the projection in the input dtype. Because this is a
  reduction over `hidden`, low-precision accumulation **drifts** from the fp32 reference
  (unlike the lossless embedding gather). Unlike `forward_fp32`, this path intentionally
  follows the ambient precision context (it is the dtype-behavior path): with an fp32
  input it is bitwise-equal to the ground truth **when ambient TF32/autocast is off**, but
  on a TF32-enabled GPU it tracks real hardware behavior and may drift. bf16/fp16 are
  always checked with a tolerance.
- **Axis-B — accuracy tolerance**: measured as max absolute error relative to the output
  peak magnitude. On a SMALL load point with the real `hidden=4096` reduction length,
  bf16 drifts ~0.3–0.4% of peak and fp16 ~0.05%. Elementwise `rtol` is not used: many
  logits are near zero while the accumulated error tracks the reduction length, not the
  output value.
- **Axis-A — batch invariance**: a row's logits are independent of the rest of the batch,
  so the output is bitwise-identical regardless of batch size or padding (`torch.equal`,
  `atol=0`) — **provided the reduction order is fixed**. Multi-threaded CPU GEMM splits
  the `hidden` reduction across threads by the `M = batch*seq` dimension, which silently
  breaks bitwise batch invariance for large `hidden`; the tests pin a single thread to fix
  the order. On GPU, cuBLAS likewise splits K by `M`, so a bitwise batch-invariant GEMM is
  a downstream kernel concern, not a free property of `torch.matmul`.

## Performance Notes

Reference operator — no fused kernel or benchmark yet. Downstream fused kernels carry their
own benchmarks and are measured against this reference for correctness.

## Tests

```bash
python -m pytest tests/test_lm_head.py -v
```

Covers: fp32 correctness vs naive matmul (bitwise, with ambient TF32 pinned off),
`forward_fp32` precision-context safety (true fp32 under ambient autocast + restores the
global TF32 flag on CPU; numerically beats a TF32 matmul on GPU), bf16/fp16 dtype-path
accuracy (relative-to-peak tolerance, with `bias`), output shape, bias semantics, Axis-A
batch invariance (slice + padding, single-thread reduction, all dtypes), input purity,
gradient flow to `hidden`/`weight` (closed-form check), registry dispatch, and a GPU-only
smoke test at the real Qwen3-8B dims (`vocab=151936, hidden=4096`) that skips when CUDA or
GPU memory is unavailable.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/linear/lm_head.py`
- `rl_engine/kernels/registry.py`
- `tests/test_lm_head.py`

## Known Limitations

- PyTorch fallback only; no fused CUDA/Triton backend yet (downstream work).
- Axis-A bitwise batch invariance holds only with a fixed reduction order (single-thread on
  CPU); a batch-invariant GEMM on GPU is a downstream concern.
