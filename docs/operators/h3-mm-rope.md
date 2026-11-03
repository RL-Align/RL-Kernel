# MiniMax-H3 Partial Multimodal RoPE

`h3_mm_rope_3axis_partial` applies the H3 position tables to Q or K after QK RMSNorm.
It pairs channel `i` with `i + 48` for `0 <= i < 48` and leaves channels `96:128`
unchanged. The caller generates the FP32 cosine/sine tables from packed `(t, h, w)`
positions: 16 frequencies per axis, concatenated to 48 and duplicated to 96.
The same tables can be reused for Q and K.

## Interface

| Value | Shape | Dtype / requirements |
|---|---|---|
| `x` | `[B, S, H, 128]` | BF16 or FP32; nonempty; strided views accepted |
| `cos`, `sin` | `[S, 96]` | FP32 constants on x's device; no gradients; strided views accepted |
| Output, input gradient | `[B, S, H, 128]` | x's dtype; contiguous; last 32 channels pass through unchanged |

H3 uses 56 heads; head shards can use fewer. Sequence length and tensor strides are
runtime metadata. Triton specializes head count and channel geometry, and launches
with grid `(S, H, B)` so real packed sequences longer than 65,535 tokens do not occupy
CUDA's limited grid-y axis. Batch and head counts must fit the device's grid-y/z limits.

```python
from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

# q: [B, S, H, 128], cos/sin: precomputed FP32 [S, 96] H3 tables on q.device
op = TritonH3MMRoPEOp()
q_rotated = op(q, cos, sin)
```

`NativeH3MMRoPEOp` in `rl_engine.reference.rope.h3_mm_rope` provides the matching
PyTorch implementation on CPU or GPU. The Triton entry accepts CUDA/ROCm tensors,
uses their device/current stream, and has no automatic fallback. These operators
are called explicitly; no registry entry is added yet. FP16 is out of scope for
this version: preserving its eager product rounding failed the independent accuracy
gate. Both supported dtypes must pass that gate.

## Rounding and Gradients

Forward preserves H3's eager arithmetic, including BF16 cast points:

1. Convert the FP32 tables to x's dtype.
2. Compute each rotation product, rounding it to x's dtype.
3. Add/subtract the two products and round the output to x's dtype.

Training, `no_grad`, and inference use the same Triton forward kernel. FP fusion is
disabled so multiplying and adding cannot become a single FMA with different rounding.

Backward uses the **same quantized coefficients** as forward, promotes them and the
upstream gradient to FP32, and computes the transpose rotation:

```text
dx[i]      = dy[i] * cos[i]      + dy[i + 48] * sin[i + 48]
dx[i + 48] = dy[i + 48] * cos[i + 48] - dy[i] * sin[i]
```

The products and sum remain FP32; only the final input gradient is cast to x's dtype.
This follows the usual autograd treatment of floating-point casts, while avoiding
the eager BF16 backward's extra product rounding. The PyTorch reference therefore
has an explicit backward too. The table halves are read separately, and the last
32 gradient channels pass through bit-for-bit. There are no atomics or reductions
across tokens, heads or batches. Only first-order gradients for x are part of the contract.

## Validation

`tests/ops/rope/test_h3_mm_rope.py` separates three checks:

- **Forward path:** bitwise comparison with the pinned H3 provider expression,
  including full-head/sharded geometry and noncontiguous inputs/tables.
- **Backward path:** bitwise comparison with autograd applied to an independent
  FP32 rotation using forward-quantized coefficients, followed by the final cast.
  A rounding regression distinguishes this from both eager BF16 backward and an
  unquantized-coefficient backward.
- **Independent accuracy:** original FP32 coefficients and FP32 arithmetic, for
  both output and gradient, using the project's `elementwise` tolerances from
  `rl_engine.contracts.numerical`. This reference does not copy the candidate's
  coefficient quantization or product rounding.

Both supported dtypes also check repeatability, training/inference equality, batch
and sequence-prefix invariance, strided views, and broadcast upstream gradients.
GPU launch regression covers `S = 65535, 65536, 73426`. Comparisons use the same
logical inputs and position tables; differing positions are not expected to agree.

```bash
python -m pytest tests/ops/rope/test_h3_mm_rope.py -q -rs
```

The [H3 block replay](h3-one-block.md) now calls the Triton candidate for `rope_q`
and `rope_k`. Its original provider and FP32 golden remain independent. With the
pinned block weights and required GPU backends available:

```bash
RL_KERNEL_H3_WEIGHTS=<dir> python -m pytest tests/models/minimax_h3/test_h3_one_block.py -q -rs
```

GPU operator/block regression and performance qualification are pending. Existing
B200 block reports used provider-only RoPE and cannot serve as evidence for this
binding. No standalone MM-RoPE benchmark or performance claim is included yet.
