# Qwen-Image final AdaLN output (CUDA)

> Status: validated on B200 (SM100). Accuracy, batch-invariance, reuse-survey and timing evidence, with reproduction commands, is in [B200 evidence](https://github.com/fusheng-ji/RL-Kernel/blob/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr538/docs/usage/evidence/qwenimage-final-adaln-out-b200/README.md).

RFC #386 `final_adaln_out`, registered as `qwenimage_final_adaln_out`. This covers
`norm_out` only; the subsequent 3072→64 `proj_out` belongs to `latent_proj_gemm`.

## Interface and formula

```python
from rl_engine.runtime.registry import KernelRegistry

op = KernelRegistry().get_op("qwenimage_final_adaln_out", device=x.device)
y = op(x=x, temb=temb, weight=weight, bias=bias)
trace = op.implementation_info(x=x, temb=temb, weight=weight, bias=bias)
```

- `x`: `[B,S,3072]`; `temb`: `[B,3072]`.
- `weight`: `[6144,3072]`; `bias`: `[6144]`.
- Output: `[B,S,3072]`. B and S must be positive.
- All inputs share a CUDA device and BF16/FP32 dtype. Gradients are provided for
  all four inputs. First-order autograd only.

The formula is `LayerNorm(x) * (1 + scale) + shift`, with `scale, shift` from
`Linear(SiLU(temb)).chunk(2)`. LayerNorm uses eps=1e-6 and **no learned affine**;
H3's RMSNorm and `(shift, scale)` order are not interchangeable with this op.

## Rounding and deterministic gradients

Projection uses the [shared conditioning implementation](qwenimage-adaln-mod-projection.md).
The final operator is deliberately unfused: native CUDA LayerNorm computes in
FP32, then stores the normalized values in the execution dtype. Separate
execution-dtype stores occur after `1 + scale`, multiplication, and addition.
Those BF16 stores are part of the forward contract, and the reference models
each one. This is not a claim of bitwise equality to arbitrary Diffusers/cuBLAS
versions; strict training and rollout must both bind this same operator.

Backward uses FP32 intermediates and straight-through derivatives at storage
rounding boundaries. Native LayerNorm backward computes `dx`; scale/shift
contributions reduce over S with recursively balanced 32-token FP32 trees.
The independent CPU reference uses sequential Neumaier summation over tokens
in FP32 (or FP64 for diagnostics). This also retains small contributions under
cancellation, independently of the unchanged CUDA token tree. There are no atomics
or collective reductions. The projection receives FP32
table gradients before the final casts to leaf dtypes. Batch size/position does
not alter a sample's token reduction tree.

Readback identifies the native LayerNorm and elementwise stages, CUDA projection
kernel, recursive token-sum tree, dtype, device, and lack of fallback. No fused
kernel or speedup over the native composition is claimed.

## Supported scope and evidence

Standard Phase A image stream only; no edit `modulate_index`, additional
conditioning, distributed execution, or output projection. NVIDIA SM80+ code,
with measured validation limited to B200/SM100. Missing/fast-math builds and
unsupported inputs fail explicitly.

Tests cover token counts 4096, 6889 and 6032, short cases, tile boundaries,
noncontiguous inputs, forward/backward accuracy, sample batching, repeatability,
parameter contribution ordering, and device isolation. Reports retain existing
tolerances and include FP64 diagnostics of the CPU FP32 reference.

See [B200 evidence and reproduction](https://github.com/fusheng-ji/RL-Kernel/blob/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr538/docs/usage/evidence/qwenimage-final-adaln-out-b200/README.md).
