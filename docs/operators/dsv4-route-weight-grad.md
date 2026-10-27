# DSV4 ROCm route-weight gradient

[Task 49 in #520](https://github.com/RL-Align/RL-Kernel/issues/520) computes one
FP32 gradient per dispatched expert row. It consumes the original up input and
the FP32 SiLU value saved by [task 46, #607](https://github.com/RL-Align/RL-Kernel/pull/607).
The forward route weight stays at the SwiGLU output, before FC2 quantization.

## Interface and arithmetic

```python
from rl_engine.ops.moe.route_weight_grad import route_weight_grad

dp_s = route_weight_grad(dh, saved_silu, original_up)
```

All inputs have shape `[M,F]`, with `F > 0`. `dh` and `up` accept BF16 or FP32;
`saved_silu` must be FP32. Strided inputs, zero strides and empty M are supported.
The result is FP32 `[M]`, in the same row order. This first-order VJP does not
provide higher-order autograd.

For each row, the terms are `(FP32(dh) * saved_silu) * clamp(FP32(up), -10, 10)`.
The two multiplications happen in that order. In particular, this is not
`dh * saved_unweighted`, nor a reconstruction from the rounded BF16 activation.
There is no multiplication or division by `p_s`, so a zero route weight can
still receive a nonzero gradient. Clamp masks belong to the gate/up derivatives;
they must not zero the derivative with respect to the route weight.

The numerical profile is `dsv4-route-weight-grad-rocm-fp32-pair256-v1`:

1. Split each row into consecutive blocks of 256 columns. Pad the last block
   with positive zero.
2. Within each block, add adjacent pairs in FP32: `(0+1), (2+3), ...`, then add
   adjacent pairs of those results until one partial remains.
3. Starting from FP32 positive zero, add partials in ascending block order.

The profile does not depend on M, row position, expert occupancy or strides.
Both kernels use four warps, disable floating-point fusion and use 64-bit
address offsets. They use no atomics, cross-row reductions or autotuning.
Each call allocates its own `[M,ceil(F/256)]` FP32 workspace; there is no global
workspace cache to share between streams or graph captures.

This is an explicit blocked reduction profile. The
[P5 start-kit reference](https://github.com/RL-Align/RL-Kernel/blob/029d95ce30326de6cfae0ff0c05f33ffb9286370/rl_engine/moe/oracle.py)
sums columns serially, so cancellation can produce different bits. The supplied
CPU ordered reference follows the pair256 profile; independent FP64 checks
cover mathematical accuracy. Do not silently substitute either summation
order for the other. FP32 overflow and NaN propagate; there is no saturation,
epsilon or recovery by reassociation. NaN payload equality is not promised.

## Training integration

With [task 48, #613](https://github.com/RL-Align/RL-Kernel/pull/613) available:

```python
from rl_engine.backends.rocm.moe.expert_training import RocmExpertTrainingBackend
from rl_engine.backends.rocm.moe.route_weight_grad import RocmRouteWeightGradient

backend = RocmExpertTrainingBackend(
    base=quantized_expert_backend,  # tasks 43/44
    route_gradient=RocmRouteWeightGradient(),
)
```

Task 48 merges FC2 base/LoRA dH in FP32, rounds it to BF16 and passes it to this
adapter. It also runs task 46's gate/up backward. Saved and recomputed forwards
therefore use the same route-gradient provider. The caller must supply the
matching task 46 saved state; the raw VJP does not track tensor versions.
Task 48's autograd wrapper supplies those mutation checks.

The standalone task 46 autograd entry remains gate/up-only. Use task 48 for
trainable route weights. The task 49 kernel itself can run without the other
P5 PRs; module integration additionally requires #601, #607 and #613. Combined
base/LoRA GPU validation still needs the task 43/44 implementations.

The adapter returns a local gradient for each selected route. P4 must return
those values using the forward dispatch plan; P3 handles selected-score,
normalization and router backward. This PR does not implement an EP transport
or claim gradients across discrete top-k decisions. Local selected-slot tests
check that the module returns gradients to the right FP32 input positions.

The backend is an explicit RLK Triton candidate for MI300X/gfx942 and
MI355X/gfx950. It does not change AITER or register a fallback. Hardware
eligibility and offline compilation are separate from device acceptance.

## Validation

```bash
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q tests/ops/moe/test_route_weight_grad.py

TRITON_INTERPRET=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q tests/backends/rocm/test_route_weight_grad_interpreter.py

RLK_REQUIRE_ROCM=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q tests/backends/rocm/test_dsv4_route_weight_grad.py

# Run after the module and activation/LoRA dependencies are available.
RLK_REQUIRE_P5_DEPENDENCIES=1 RLK_REQUIRE_ROCM=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
tests/models/deepseek_v4/test_route_weight_grad_integration.py
```

CPU tests cover independent FP64 finite differences of the smooth pre-cast
forward, the fixed reduction order, zero/tiny weights, tail columns, overflow
and rejected inputs. Finite differences of the discrete BF16 cast are not an
oracle for its straight-through derivative. Interpreter tests execute the
actual Triton kernel bodies on CPU and do not establish GPU support.

Device tests compare against the ordered CPU reference and sweep M=1/16/17/64/65,
row positions, strides and repeated calls. They also check concurrent streams,
graph replay and input updates. The integration suite checks base/LoRA/combined
CPU wiring and a real LoRA-only ROCm module. Missing required dependencies or
hardware fail when the flags above are set. Run on both target GPUs separately;
no cross-GPU bitwise or performance claim follows from compilation alone.
