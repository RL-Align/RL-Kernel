# LM Head

The lm_head operator projects hidden states back to vocabulary logits, the final
layer of the Qwen3/Llama stack. It is a WS1 ground-truth reference for issue #108:
a pure-PyTorch definition of the correct answer that downstream fused CUDA/Triton
kernels are validated against.

- **LM Head** (`NativeLMHeadOp`): mathematically `out = hidden @ weight.t() (+ bias)`.
  The native reference implements this as row-wise fixed-K GEMV projections so the
  reference path is batch-invariant.

For Qwen3-8B the weight is the output projection `[vocab=151936, hidden=4096]` in the
HF `nn.Linear` `[out, in]` convention. It is independent from the embedding table
(`tie_word_embeddings=false`), and Qwen3 has no bias (`bias=None`).

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

lm_head = kernel_registry.get_op("lm_head")

logits = lm_head(hidden, weight)          # [B, S, hidden], [vocab, hidden] -> [B, S, vocab]
logits = lm_head(hidden, weight, bias=b)  # optional [vocab] bias
```

The op exposes the WS1 dual-path contract:

- `forward(...)` projects in the input dtype and returns the input dtype.
- `forward_fp32(...)` upcasts to fp32, accumulates in fp32, and returns fp32. The
  fixed-K projection runs with autocast disabled and CUDA TF32 turned off, so it
  stays a true fp32 reference regardless of the caller's ambient precision context.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeLMHeadOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA SM90 (H200/Hopper) | `SM90LMHeadOp` | `_C.lm_head_sm90_forward` | Single-card batch-invariant forward backend; no Split-K; bf16 backward uses deterministic GEMM. |
| Ascend NPU | `AscendLMHeadOp` | `_C_npu.lm_head_ascend` | Single-card batch-invariant forward backend: one output element per block, full K reduction in fp32 over a fixed tile order; fp32-formula VJP backward. |
| ROCm / Triton | N/A | N/A | Falls back to the PyTorch native reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `hidden` | `[B, S, hidden]` or any leading dims | fp16/bf16/fp32 | Hidden states. |
| `weight` | `[vocab, hidden]` | fp16/bf16/fp32 | Output projection in HF `[out, in]` layout. |
| `bias` | `[vocab]` or `None` | fp16/bf16/fp32 | Optional; Qwen3 uses `None`. |
| output | `hidden.shape[:-1] + (vocab,)` | `forward`: hidden dtype; `forward_fp32`: fp32 | Logits. |

Output dtype follows `hidden`. The op is pure: no randomness and no in-place mutation.

## Accuracy

Reference semantics (`forward_fp32`):

```python
flat_hidden = hidden.float().reshape(-1, hidden.size(-1))
if flat_hidden.size(0) == 0:
    flat_out = flat_hidden @ weight.float().t()
else:
    rows = [torch.mv(weight.float(), row) for row in flat_hidden]
    flat_out = torch.stack(rows)
out = flat_out.reshape(*hidden.shape[:-1], weight.size(0))
if bias is not None:
    out = out + bias.float()
```

- **Ground truth**: `forward_fp32` accumulates in and returns fp32, with autocast and
  CUDA TF32 disabled.
- **Dtype path**: `forward` runs the projection in the input dtype. Because this is a
  reduction over `hidden`, low-precision accumulation drifts from the fp32 reference and
  is checked with tolerance.
- **Axis-A batch invariance**: a row's logits are bitwise-identical regardless of batch
  size or padding. The native reference enforces this by flattening leading dimensions
  and projecting each row through the same GEMV-shaped K reduction instead of relying on
  batched GEMM, whose reduction tree can change with `M = batch * seq`.

## Dispatch Behavior

`kernel_registry.get_op("lm_head")` resolves through the `OpBackend` priority map. On
CPU, ROCm, and CUDA devices without the SM90 extension, dispatch uses the PyTorch native op
(`PYTORCH_NATIVE_LM_HEAD`). On H200/Hopper-class builds that expose `_C.lm_head_sm90_forward`,
the CUDA SM90 single-card batch-invariant backend is prepended. Its forward path assigns
one CTA to each output logit and performs the full K reduction without Split-K.

The one-CTA-per-logit design is a deliberate invariance tradeoff: CTAs re-read the
hidden row and vocab weight row independently, so large-vocab projections are expected
to be memory-bandwidth bound compared with a tiled GEMM. This path exists to preserve a
fixed hidden-dimension accumulation order for the WS1/H200 correctness gate.

On `npu` the priority is:

1. `ASCEND_LM_HEAD` — `AscendLMHeadOp` (batch-invariant Ascend C forward; fp32/bf16/fp16).
2. `PYTORCH_NATIVE_LM_HEAD` — `NativeLMHeadOp` (fallback).

The Ascend kernel implements the same structure as the SM90 CUDA kernel: one output
element per block iteration, the full hidden-dimension reduction inside that block over
a fixed tile order (products -> per-tile sum -> sequential scalar accumulation), bias
added in fp32, final cast to the output dtype. There is no Split-K, so a row's logits
depend only on H — never on N or block assignment — and are bitwise identical across
batch sizes, row positions, and block assignments on the NPU. The per-tile sums use the
Ascend vector unit's fixed hardware tree instead of CUDA's warp-shuffle tree, so the
comparison against the PyTorch reference (torch.mv) is tolerance-based per the
reduction contract, not bitwise.

For bf16 H200 training, `SM90LMHeadOp.backward` routes `dhidden` through
`_C.det_gemm_da` and `dweight` through `_C.det_gemm_db` (`hidden.T @ dlogits`, transposed
back to the HF `[vocab, hidden]` layout). The wrapper fails fast if those deterministic
GEMM symbols are missing instead of silently falling back to cuBLAS for bf16 gradients.

## Tests

```bash
python -m pytest tests/test_lm_head.py tests/test_lm_head_ascend.py -v
```

Covers fp32 correctness vs the fixed-K reference, precision-context safety, bf16/fp16
accuracy, output shape, bias semantics, Axis-A batch invariance, input purity, gradient
flow to `hidden` and `weight`, registry dispatch, and a GPU-only smoke test at the real
Qwen3-8B dimensions. The Ascend suite adds: contract-tolerance correctness vs the
reference (fp32/bf16/fp16, with and without bias), fp32-formula VJP backward, bitwise
batch invariance (batch sizes 1 vs {2,4,16,300}, row positions, multi-tile H=10000,
repeated runs), and NPU registry dispatch.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/linear/lm_head.py`
- `rl_engine/kernels/ops/cuda/linear/lm_head.py`
- `csrc/cuda/embedding_lm_head_sm90.cu`
- `rl_engine/kernels/ops/ascend/linear/lm_head.py` — Ascend deterministic op
- `csrc/ascend/lm_head_ascend.asc` — Ascend C forward kernel
- `csrc/ascend/npu_module.cpp` — shared pybind entry for `rl_engine._C_npu`
- `rl_engine/kernels/registry.py`
- `tests/test_lm_head.py`
- `tests/test_lm_head_ascend.py`
