# Qwen-Image AdaLN modulation projection (CUDA)

> Status: validated on B200 (SM100). Accuracy, batch-invariance, reuse-survey and timing evidence, with reproduction commands, is in [B200 evidence](https://github.com/fusheng-ji/RL-Kernel/blob/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr537/docs/usage/evidence/qwenimage-adaln-mod-projection-b200/README.md).

RFC #386 `adaln_mod_projection`, registered as `qwenimage_adaln_mod_projection`
to keep the H3 and Qwen-Image contracts separate. This is a Phase A single-GPU
operator, not the block's LayerNorm/modulation or gated residual.

## Interface

```python
from rl_engine.runtime.registry import KernelRegistry

op = KernelRegistry().get_op("qwenimage_adaln_mod_projection", device=temb.device)
params = op(temb=temb, weight=weight, bias=bias)
shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, -1)
trace = op.implementation_info(temb=temb, weight=weight, bias=bias)
```

`temb` is `[B,3072]`, `weight` is `[18432,3072]`, `bias` is `[18432]` and
the output is `[B,18432]`. Run this interface separately for `img_mod.1` and
`txt_mod.1`; there is no shared weight assumption. All inputs share a device
and dtype (BF16 or FP32), and B must be positive. Noncontiguous views are copied
explicitly. Backward returns gradients for all three inputs. First-order
autograd only; gradient accumulation into leaves follows PyTorch's usual rules.

The CUDA registry slot has no reference fallback. NVIDIA SM80+ is required for
the implementation; only B200/SM100 is validated here. A missing extension or
fast-math build fails closed. CPU reference classes are separate; no optimized
ROCm, Ascend, or MUSA slot is registered.

## Arithmetic and reuse

Contract `qwenimage-adaln-v2`, backward `qwenimage-parameter-grad-neumaier-v2`:

1. Native CUDA SiLU, with a declared output store in the execution dtype.
2. Fixed-order projection with FP32 accumulation, one bias addition, and an
   execution-dtype output store. FP32 uses one warp's fixed vector/FMA order and
   XOR reduction. BF16 uses zero-initialized `mma.sync` K=16 partials summed in
   a fixed order. Neither depends on B or logical batch position.
3. Backward keeps intermediates FP32. Input gradients use 64-column compensated dot
   partials and an ascending compensated (Kahan) fold. Parameter gradients use
   ascending FP32 Neumaier accumulation, including FMA product residuals for
   `d_weight`; `d_bias` uses ascending Neumaier summation. Only the final
   result is cast. SiLU's analytic VJP is evaluated in
   FP32, and leaf gradients are cast to each input's dtype.

The projection source is adapted from the H3 deterministic GEMV at
`af3311ab3cf22e0d77ddd523cd4283990a9f8377`, retaining the forward arithmetic and
using compensated FP32 summation for the input-gradient dot and final fold. There
is no dependency on H3 operators or their stacked branches.

`adaln_prior_art.py` compares native `SiLU + F.linear`, this reused GEMV and,
optionally, the existing Qwen-Image QKV cuBLASLt extension. The native FP32
projection changes bytes across batch sizes in the measured sweep. The QKV
forward is invariant in that sweep but is slower for these small conditioning
batches. Native BF16 forward passes the measured sweep; this does not establish
its backward or an all-dtype strict contract. The prior-art report makes only
a forward claim for the QKV extension. No process-wide cuBLAS/TF32 settings are
changed by this implementation.

The independent reference uses CPU FP32 SiLU/linear algebra with explicit
storage rounding and straight-through derivatives at cast boundaries. Its
input-gradient dot is blocked into 128-column CPU BLAS operations and pairwise
FP32 addition: a single long CPU BLAS dot can itself miss the repository's
FP32 gradient tolerance. FP64 diagnostics check this reference independently.
This reference does not call the candidate CUDA kernel.

## Validation and reproduction

See the [B200 evidence and review notes](https://github.com/fusheng-ji/RL-Kernel/blob/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr537/docs/usage/evidence/qwenimage-adaln-mod-projection-b200/README.md).
Reports use the existing `reduction` forward/gradient tolerance rows, without
private overrides. Strict replay and batch invariance are checked with
`torch.equal` on contiguous raw-byte views, separately from agreement with the CPU reference.
