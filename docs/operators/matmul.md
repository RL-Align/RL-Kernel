# Matmul

Matmul provides the PyTorch reference GEMM operator. It targets dense projection shapes used in Qwen3-style post-training workloads, including Q, K, V, O, gate, up, and down projections.

This page documents the PyTorch baseline version.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

matmul = kernel_registry.get_op("matmul")
output = matmul.forward(a, b)
reference = matmul.forward_fp32(a, b)
```

The operator can also be imported directly:

```python
from rl_engine.kernels.ops.pytorch.linear import NativeMatmulOp

matmul = NativeMatmulOp()
```

## Backend

| Backend | Wrapper | Native symbol | Notes |
| --- | --- | --- | --- |
| PyTorch native | `NativeMatmulOp` | None | Reference baseline; no split-K or manual blocking. |

`kernel_registry.get_op("matmul")` dispatches to the PyTorch native backend on CPU,
CUDA, and ROCm. CUDA/Triton fused GEMM kernels should compare against this reference.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `a` | `[..., K]` | `float32`, `bfloat16`, or `float16` | Left operand. |
| `b` | `[K, N]` | Same floating dtype family | Right operand in `[in, out]` layout. |
| Output | `[..., N]` | See below | Result of `a @ b`. |

`forward(...)` returns `a.dtype`. `forward_fp32(...)` casts both inputs to
`float32`, calls `torch.matmul` once, and returns `float32`.

## Qwen3 Projection Shapes

The reference covers the Qwen3-8B projection dimensions:

| Projection | Shape |
| --- | --- |
| `q_proj`, `o_proj` | `4096 -> 4096` |
| `k_proj`, `v_proj` | `4096 -> 1024` |
| `gate_proj`, `up_proj` | `4096 -> 12288` |
| `down_proj` | `12288 -> 4096` |

LM head uses a separate operator because its weight follows the HF `[out, in]`
layout and is internally transposed.

## Reference Semantics

```python
ref = torch.matmul(a.float(), b.float())
```

The gold path intentionally avoids split-K, tiled reductions, or any manual
accumulation order. Those optimizations belong in downstream fused kernels and
should be validated against this baseline.

## Accuracy

Matmul is categorized as a `reduction` operator in the numerical contract.
Expected comparison behavior:

| Path | Expected dtype | Purpose |
| --- | --- | --- |
| `forward` | `a.dtype` | Candidate dtype behavior. |
| `forward_fp32` | `torch.float32` | Deterministic reference output. |

Batch invariance is expected for the reference path: applying matmul to a full
batch and then slicing a row should match applying matmul to that row alone.

## Tests

```bash
python -m pytest tests/test_matmul.py -q
```

The test covers shape, dtype behavior, fp32 reference equivalence, batch
invariance, Qwen3 projection dimensions, and registry dispatch.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/linear/matmul.py`
- `rl_engine/kernels/ops/pytorch/linear/__init__.py`
- `rl_engine/kernels/registry.py`
- `tests/test_matmul.py`
