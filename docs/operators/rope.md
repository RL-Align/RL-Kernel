# RoPE

RoPE applies rotary position embeddings to per-head query or key tensors. The
current implementation is a pure PyTorch reference operator for Issue #108
ground-truth validation; it is not a fused CUDA or Triton kernel.

This page documents the PyTorch baseline version.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

rope = kernel_registry.get_op("rope")
output = rope.forward(x, positions, theta=1_000_000.0)
reference = rope.forward_fp32(x, positions, theta=1_000_000.0)
```

The operator can also be imported directly:

```python
from rl_engine.kernels.ops.pytorch.rotary_embedding import NativeRoPEOp

rope = NativeRoPEOp()
```

## Backend

| Backend | Wrapper | Native symbol | Notes |
| --- | --- | --- | --- |
| PyTorch native | `NativeRoPEOp` | None | Reference baseline for Qwen3-style RoPE. |

`kernel_registry.get_op("rope")` dispatches to the PyTorch native backend on CPU,
CUDA, and ROCm. CUDA/Triton fused RoPE kernels should compare against this reference.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[B, H, S, D]` | `float32`, `bfloat16`, or `float16` | Query or key tensor; Qwen3 uses `D=128`. |
| `positions` | `[S]` or `[B, S]` | Integer | Absolute token positions. |
| `theta` | scalar | float | Defaults to `1_000_000.0` for Qwen3. |
| Output | `[B, H, S, D]` | See below | Same shape as `x`. |

`forward(...)` returns the input dtype. `forward_fp32(...)` computes and returns
`float32` and is the gold-standard reference path.

## Reference Semantics

The implementation uses the Hugging Face rotate-half convention, pairing dimensions
`(i, i + D/2)` rather than adjacent dimensions.

```python
half = x.shape[-1] // 2
inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
freqs = positions.float()[..., None] * inv_freq
cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

a, b = x.float()[..., :half], x.float()[..., half:]
rotated = torch.cat([-b, a], dim=-1)
out = x.float() * cos + rotated * sin
```

For Qwen3-8B validation, RoPE is applied after QK-Norm and before attention:

```text
RMSNorm(q), RMSNorm(k) -> RoPE(theta=1e6) -> attention
```

## Accuracy

RoPE is categorized as an `elementwise` operator in the numerical contract.
Expected comparison behavior:

| Path | Expected dtype | Purpose |
| --- | --- | --- |
| `forward` | Same as `x.dtype` | Candidate dtype behavior. |
| `forward_fp32` | `torch.float32` | Deterministic reference output. |

Batch invariance is expected to be bitwise: applying RoPE to a full batch and then
slicing a row must match applying RoPE to that row alone.

## Tests

```bash
python -m pytest tests/test_rope.py -q
```

The test covers shape, dtype behavior, HF rotate-half equivalence, `positions`
as `[S]` and `[B, S]`, batch invariance, and Qwen3 query/key head shapes.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/rotary_embedding/rope.py`
- `rl_engine/kernels/ops/pytorch/rotary_embedding/__init__.py`
- `rl_engine/kernels/registry.py`
- `tests/test_rope.py`
