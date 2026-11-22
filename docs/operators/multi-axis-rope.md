# Qwen-Image Multi-Axis RoPE

Implements the unfused CUDA `multi_axis_rope` row of
[RFC #386](https://github.com/RL-Align/RL-Kernel/issues/386), with a Triton backend
and a PyTorch diagnostic reference. It rotates adjacent channel pairs;
the existing GPT-NeoX half-split RoPE has different semantics.

## API and position semantics

```python
import torch
from rl_engine.ops.rope.multi_axis_rope import (
    build_multi_axis_cos_sin, get_multi_axis_rope_op, qwen_image_positions,
)

positions, text_positions = qwen_image_positions(64, 64, 128)
cos, sin = build_multi_axis_cos_sin(positions, device="cuda")
x = torch.randn(1, 4096, 24, 128, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
op = get_multi_axis_rope_op("cuda", device=x.device)
y = op(x, cos, sin)
y.float().square().mean().backward()
print(op.trace())
```

Semantics follow Diffusers
[`40f63690`](https://github.com/huggingface/diffusers/blob/40f63690da440e8c382ab53e67508835deb1ed2f/src/diffusers/models/transformers/transformer_qwenimage.py):
`QwenEmbedRope(scale_rope=True)` and `apply_rotary_emb_qwen(use_real=False)`.
Axes have widths `[16, 56, 56]`. Image coordinates are `(0, h, w)` in row-major
order, centered in `[-ceil(size/2), floor(size/2))`; text coordinates begin at
`max(height//2, width//2)` on each axis. The caller chooses image/text stream order.
This helper covers single-frame text-to-image at packed latent resolution.

| Argument | Contract |
| --- | --- |
| `x` | `[B, S, H, 128]`, H > 0; FP32, FP16 or BF16 |
| `cos`, `sin` | Matching FP32 `[S, 64]` shared or `[B, S, 64]` per-sample tables |
| Device/layout | Input and tables share a device; independent strides, including packed-QKV views, are read directly |
| Output | Input shape/dtype/device, contiguous; empty B/S supported |
| Gradients | Input gradients and recursive input VJPs; table gradients rejected |
| Position builder | int32/int64 `[S, 3]` or `[B, S, 3]`, values in `[-4096, 4095]` |

Tables use the pinned upstream CPU FP32 `pow`/`polar` operations before copying
to the target device. Retain them before CUDA Graph capture; table construction
is independent of default-device contexts.

## Arithmetic and dispatch

Contract ID: `qwenimage-interleaved-rope-fp32-v1`. For each adjacent pair `(a, b)`,
compute `(a*c - b*s, b*c + a*s)` with separately rounded FP32 multiply/adds and
one output cast. CUDA uses scalar round-to-nearest PTX without FTZ; Triton disables
FP contraction. Transpose VJPs flip the sine sign bit, preserving signed zeros.
Their byte oracle is the explicitly rounded transpose; generic slice autograd
may erase negative-zero signs. NaN payload equality is not guaranteed.

`cuda` selects the native dispatcher; `triton` selects the adapted rotary kernel;
`reference` selects PyTorch FP32. Explicit GPU selections fail when unavailable.
`auto` tries CUDA, Triton, then reference on NVIDIA and uses reference elsewhere.
Use an explicit backend and retain `op.trace()` for acceptance results.

Both GPU backends support fake tensors, autograd, fullgraph forward/backward and
CUDA Graph replay. Use the public API above for gradients: direct grad-enabled
calls to `torch.ops.rl_kernel_native.multi_axis_rope` reject gradient-carrying
inputs. Explicit no-grad/inference calls remain supported. Backward telemetry's
Python execution count does not count CUDA Graph replays.

## Build and validation

Ordinary CUDA builds include the operator in `rl_engine._C`. Use a CUDA toolkit
matching PyTorch; `RL_KERNEL_REQUIRE_EXT=1` makes a requested native build fail
rather than silently falling back. The RoPE scalar arithmetic preserves
subnormals under the extension's shared fast-math flags.

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .
python -m pytest tests/ops/rope/test_multi_axis_rope.py \
  tests/validation/operators/test_operator_inputs.py -q -rs
python tools/validation/operators/check_operator.py --op multi_axis_rope \
  --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 7 --n-heads 3 --check-grad
# Repeat with --candidate triton for the other GPU backend.
python -m benchmarks.operators.rope.benchmark_multi_axis_rope --output /tmp/rope-benchmark.json
mkdocs build --strict -f mkdocs.yaml
```

Use the standard pytest entry points for CPU contracts; GPU cases require a real
NVIDIA GPU and native/Triton dependencies. The operator tests exercise independent FP64
accuracy, exact byte invariance, strides/no-copy, signed zeros/subnormals, empty
inputs, stream/device guards, higher VJPs, opcheck, fullgraph and graph replay.
The WS1 tests reuse standard forward/gradient accuracy and invariance APIs.
OP_SPECS, the ordinary CLI and the gradient adapter share centered grid prefixes
indexed by full logical sample/token identity across packing, chunks and padding.

The benchmark covers the short case and all three reference image grids, with
contiguous and packed-QKV layouts. It checks forward/VJP correctness before
reporting eager CUDA-event timings for CUDA, Triton and the PyTorch reference.
Native/Triton checks compare bytes with the explicit FP32 forward/transpose;
reference autograd is numerically checked and its zero-sign differences recorded.
Tables, input allocation and first-use compilation are outside the timing region.
The JSON records dtype, runtime/GPU identity, backend traces and timing parameters.

## Attribution and limits

The Triton implementation adapts
[FlashAttention@98c1cee](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/ops/triton/rotary.py),
retaining its BSD-3-Clause license in the source tree, wheels and source distributions.
[Ascend PR #411](https://github.com/RL-Align/RL-Kernel/pull/411) addresses the same
roadmap row with different reference semantics; sharing an interface requires
alignment. This contribution qualifies the unfused operator on WSL2/SM120.
Bare-metal Linux, other GPU architectures, multi-GPU execution, video/editing,
QK-norm fusion and full-model trajectories remain unqualified. It establishes
no performance advantage over FlashAttention or model-throughput speedup.
