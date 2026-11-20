# DSV4 ROCm clamp, SwiGLU and route weighting

This implements [P5 task 46](https://github.com/RL-Align/RL-Kernel/issues/520):
clamp the routed expert's gate/up activations, compute SwiGLU, and multiply the
route weight once before the FC2 activation quantizer. It includes gate/up
backward and a save/recompute option. It is an explicitly selected ROCm candidate,
not an automatically registered replacement for the existing SwiGLU operators.

## Interface and arithmetic

```python
from rl_engine.ops.moe.clamp_swiglu import (
    clamp_swiglu_weighted,
    clamp_swiglu_weighted_packed,
)

h = clamp_swiglu_weighted(gate, up, p_s, recompute=False)
# Equivalent packed layout: gate columns followed by up columns.
h = clamp_swiglu_weighted_packed(gate_up, p_s, recompute=True)
```

The inputs are same-dtype BF16 or FP32 matrices `[M, F]`. Route weights are an
FP32 vector `[M]`. Strided inputs and empty batches are accepted; `F` must be
positive. The output is a contiguous BF16 matrix `[M, F]`.

The routed path evaluates the following operations in FP32, in this order:

1. `g = min(gate, 10)` and `u = clamp(up, -10, 10)`.
2. `sig = sigmoid(g)`, then `silu = g * sig`.
3. `raw = silu * u`, then `weighted = raw * p_s[:, None]`.
4. Round `weighted` to BF16 once when writing `h`.

There is no BF16 rounding between SiLU and route weighting. In particular,
rounding `raw` first is a different computation. Passing `p_s=None` selects the
shared-expert activation: no clamp and no route weighting. Passing a vector of
ones still selects the routed path and retains its clamps.

The low-level `RocmClampSwiGLUWeighted.forward` returns `h` and a
`ClampSwiGLUState` containing contiguous FP32 `sigmoid`, `silu` and `unweighted`
values. The caller must retain the matching original inputs and weights if it
uses low-level backward; these calls do not create an autograd graph.

## Backward and the task 49 boundary

For the routed path, gate/up backward computes:

```text
dy = FP32(dh) * p_s
dsilu = sig * (1 + g * (1 - sig))
dgate = ((dy * u) * dsilu) * (original_gate < 10)
dup = (dy * silu) * (-10 < original_up < 10)
```

The chosen subgradient is zero exactly at the clamp limits. The shared path
omits both the weighting and the clamp masks. Low-level backward returns FP32
gradients; autograd converts them to each input's dtype. The BF16 output cast
uses the usual straight-through derivative. Higher-order gradients are outside
this implementation.

Task 49 owns `dp_s`. Its ordered terms are `(dh * silu) * u`, accumulated across
the feature dimension. Replacing them with `dh * raw` can change rounding;
recovering `raw` by dividing by `p_s` also fails for zero or tiny weights. This
operator retains the separate intermediate values needed for that work.

Until task 49 is integrated, the public autograd entry rejects
`p_s.requires_grad=True` when gradients are enabled. Frozen route weights and
the unweighted path support gate/up training. Rollout under `torch.no_grad()`
can use weights with that flag set. The operator is not yet a complete trainable
router-to-expert module.

Task 43 owns quantizing `h`; task 44 owns expert GEMMs; task 48 owns the complete
expert training module. This change neither quantizes twice nor moves route
weighting after FC2. Combining the activation and quantizer in a later kernel
must preserve this BF16 boundary and the saved backward values.

## ROCm implementation and AITER

The candidate uses one row and a fixed 256-column tile per program, four warps,
64-bit offsets and disabled FP fusion. It has no feature reduction, atomics,
autotuning or batch-dependent tile selection. Its arithmetic profile is
`dsv4-clamp-swiglu-rocm-fp32-v1`; sigmoid uses Triton's FP32 exponential and
round-to-nearest division. PyTorch references check mathematical accuracy,
not bitwise agreement with this exponential implementation.

Eligibility is restricted to MI300X/gfx942 and MI355X/gfx950. Eligibility is not
hardware acceptance: each target needs its own forward/backward, batch-sweep
and timing results. CPU execution is available only through the explicit
diagnostic reference, never as a production fallback. Finite inputs are the
acceptance domain. Ordinary FP32/BF16 overflow is not saturated or hidden.

The following AITER sources were reviewed at commit
`326cffa38c9eda16bef19375ba55334e4d3acd59`:

- [Activation entry points](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/activation.py):
  `silu_and_mul` accepts a clamp limit; `scaled_silu_and_mul` accepts a scale but
  no clamp limit; `silu_and_mul_quant` accepts a clamp limit but no route-weight
  argument.
- [Triton activation backward](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/activation.py):
  `silu_and_mul_backward` exists, but the reviewed implementation is restricted
  to gfx950 and does not include clamp or route-weight derivatives.
- [Native activation implementation](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/csrc/kernels/activation_kernels.cu)
  supplies the corresponding activation and quantization kernels.

Those APIs do not cover this full contract in one call. RLK therefore owns this
initial Triton implementation; no upstream AITER changes are included. A later
AITER provider can replace it once the same input types, rounding points,
saved intermediates, backward rules and target-GPU tests are satisfied.

## Validation

CPU checks cover the independent FP64 formula and finite differences away from
clamp boundaries, explicit boundary subgradients, one-time route weighting,
BF16 rounding, zero/tiny weights, strided layouts, save/recompute and stale
saved-input detection:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/ops/moe/test_clamp_swiglu.py
```

Run hardware acceptance separately on each target with a ROCm PyTorch/Triton
environment. The flag makes an unavailable or unsupported GPU fail instead of
silently skipping:

```bash
RLK_REQUIRE_ROCM=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/backends/rocm/test_dsv4_clamp_swiglu.py
```

The GPU suite checks BF16 and FP32 inputs, feature widths 1/31/33/257/2048/4096,
and batches 1/16/17/64/65 with the same row placed at the beginning, middle and
end. Outputs, saved intermediates and gate/up gradients must retain identical
bytes across these changes and repeated runs. Numerical accuracy is checked
separately against staged math. The suite also checks packed inputs, strides,
recompute and visible input updates.

Passing these checks establishes only the operator configurations exercised.
Full-model training/rollout agreement, Graph replay, distributed execution and
performance acceptance require subsequent integration and hardware runs.
