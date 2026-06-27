# SiLU / SwiGLU Activation

The activation operators are the element-wise core of the Qwen3/Llama gated MLP. They are
**WS1 ground-truth references** (issue #108): pure-PyTorch, fp32-accumulating definitions of
the "correct answer" that downstream fused CUDA/Triton MLP kernels are validated against.

- **SiLU** (`NativeSiLUOp`): `silu(x) = x * sigmoid(x)` — the `hidden_act="silu"` gate.
- **SwiGLU** (`NativeSwiGLUOp`): `swiglu(gate, up) = silu(gate) * up` — the gated MLP middle
  stage. `gate` / `up` are the `gate_proj` / `up_proj` outputs (already at the intermediate
  width); the following `down_proj` is a plain Matmul and is **not** part of this operator.

```text
hidden --gate_proj--> gate --\
                              swiglu --> down_proj --> hidden
hidden --up_proj----> up ----/
```

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

silu = kernel_registry.get_op("silu")
swiglu = kernel_registry.get_op("swiglu")

# SiLU: single element-wise activation
y = silu(x)                       # [..., N]  ->  [..., N]

# SwiGLU: gated activation (gate and up must share shape)
h = swiglu(gate, up)              # [..., I], [..., I]  ->  [..., I]
```

Both ops expose the WS1 dual-path contract:

- `forward(...)` — computes in fp32, casts back to the input dtype (Axis-B accuracy
  candidate / dtype-behavior path).
- `forward_fp32(...)` — computes and returns fp32 (the ground-truth golden path).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeSiLUOp` / `NativeSwiGLUOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA / ROCm / Triton | — | — | Planned: downstream fused MLP kernels validate against this reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` (SiLU) | `[..., N]` | float (fp16/bf16/fp32) | Any shape; last dim arbitrary (Qwen3-8B `I=12288`). |
| `gate` (SwiGLU) | `[..., I]` | float | `gate_proj` output. |
| `up` (SwiGLU) | `[..., I]` | float | `up_proj` output; **must share `gate`'s shape**. |
| output | same as input | `forward`: input dtype · `forward_fp32`: float32 | Same shape as input. |

Element-wise and shape-agnostic: the Qwen3-8B intermediate dim `I=12288` is just one valid
last-dim size, not a hard requirement. Pure functions — no randomness, no in-place
mutation, device/dtype follow the inputs.

## Dispatch Behavior

`kernel_registry.get_op("silu" | "swiglu")` resolves through the `OpBackend` priority map.
On `cuda` / `rocm` / `cpu` the only registered backend today is the PyTorch native op
(`PYTORCH_NATIVE_SILU` / `PYTORCH_NATIVE_SWIGLU`), so every device dispatches to the
fp32 reference. When fused kernels land, they are prepended to the priority list and the
native op becomes the fallback.

## Accuracy

Reference semantics (`forward_fp32`, fp32 accumulation):

```python
# SiLU
out = x.float() * torch.sigmoid(x.float())

# SwiGLU
gate_f = gate.float()
out = gate_f * torch.sigmoid(gate_f) * up.float()
```

- **Ground truth**: `forward_fp32` always accumulates in and returns fp32.
- **Dtype path**: `forward` runs the same fp32 math, then casts back to the input dtype;
  it is bitwise-equal to `forward_fp32(x).to(dtype)`.
- **Axis A — batch invariance**: element-wise and row-independent, so a row's output is
  bitwise-identical regardless of batch size or padding (`torch.equal`, `atol=0`).
- **Axis B — tolerance**: as `elementwise` ops, low-precision tolerance follows the
  `elementwise` row of the WS1 numerical contract.

## Performance Notes

Reference operators — no fused kernel or benchmark yet. Downstream fused MLP kernels carry
their own benchmarks and are measured against this reference for correctness.

## Tests

```bash
python -m pytest tests/test_swiglu.py -v
```

Covers: correctness vs an independent fp32 formula, dtype paths, Axis-A batch invariance
(slice + padding), input purity, gradient flow, the SwiGLU shape guard, and registry
dispatch.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/activation/swiglu.py`
- `rl_engine/kernels/registry.py`
- `tests/test_swiglu.py`

## Known Limitations

- PyTorch fallback only; no fused CUDA/Triton backend yet (downstream work).
- SwiGLU requires `gate` and `up` to share shape (raises `ValueError` otherwise); no
  broadcasting.
