# H3 ODE Step

## Summary

Deterministic single step of the MiniMax-H3 rectified-flow Euler sampler for the
WS1 consistency track ([RFC #420](https://github.com/RL-Align/RL-Kernel/issues/420),
work item `h3_ode_step`).

The released H3 sampler uses `eta = 0`, so it defines a deterministic ODE update.
This operator owns the data-ward denoised estimate and the FP32 Euler blend for
both the video and the audio modality:

```
x0     = xt + sigma * v
r      = sigma_next / sigma
x_next = r * xt + (1 - r) * x0
```

`sigma` and `sigma_next` are **per-row semantic inputs**, not metadata: one packed
batch can carry several timestep-table rows, and the video (shift 12.0) and audio
(shift 3.0) schedulers own separate sigma grids that must never share a step index
implicitly.

## Entry Point

```python
from rl_engine.kernels.ops.pytorch.flow.ode_step import NativeH3OdeStepOp
from rl_engine.kernels.ops.triton.flow.ode_step import TritonH3OdeStepOp

x_next, x0 = TritonH3OdeStepOp().forward(xt, v, sigma, sigma_next)
```

`forward` returns results in the input dtype; `forward_fp32` returns FP32 results
and is the reference path consumed by the gtest gold.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `CudaH3OdeStepOp` | `_C.ode_step_forward` / `_C.ode_step_backward` | supported (sm_90 validated) |
| CUDA (portable) | `TritonH3OdeStepOp` | Triton JIT | supported |
| ROCm | `TritonH3OdeStepOp` | Triton JIT | supported |
| PyTorch fallback | `NativeH3OdeStepOp` | n/a | reference |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `xt` | `[R, C]` or `[B, S, C]` | fp32 / bf16 / fp16 | current latent; `R` is the flattened packed-row count |
| `v` | same as `xt` | same as `xt` | model-predicted velocity, data-ward sign |
| `sigma` | scalar, `[R]`, or `[R, 1]` | fp32 | broadcastable; **must be > 0** |
| `sigma_next` | same as `sigma` | fp32 | `0 <= sigma_next <= sigma` |

Outputs (tuple): `x_next`, and the intermediate `x0`, both shaped like `xt`.

All three accepted sigma layouts are normalised to the same per-row semantics: a
1-element tensor becomes a 0-dim scalar, and one value per flattened packed row
becomes `xt.shape[:-1] + (1,)`. A bare `[R]` sigma therefore always means "one
value per row", never "one value per channel" — left unreshaped it would
broadcast against the last axis and silently return wrong values whenever
`R == C`. The Triton and CUDA backends index sigma by flattened row, so all three
backends agree on every documented input form.

## Dispatch Behavior

`forward` resolves to the registered candidate named on the command line: the
native CUDA kernel via `--candidate cuda`, or the portable Triton kernel via
`--candidate triton`. Both select a scalar-sigma or per-row-sigma branch on
shape, never by value. A sigma whose size is neither the scalar nor exactly
one-per-row raises instead of broadcasting, because implicit broadcast is how two
modality grids get silently merged.

There is no silent fallback: the strict path fails closed on unsupported geometry
(RFC #420 section 4, rule 9), and each candidate declares itself in `OP_SPECS`
rather than aliasing another backend.

## Accuracy

* Reference: `NativeH3OdeStepOp.forward_fp32` — the declared expression order in
  FP32 with no reassociation.
* Thresholds come from the shared contract (`tolerance_contract.json`) under
  `op_class="elementwise"`; this page deliberately restates no `atol` / `rtol`.
* Arithmetic declaration: **no reduction** (element-wise); **accumulator FP32**;
  a **single declared epilogue cast** to the storage dtype; **no split policy**
  (Split-K, Stream-K, split-KV and cross-CTA atomics are N/A); TF32 and fast-math
  reassociation disabled.
* `r = sigma_next / sigma` uses a correctly-rounded FP32 divide. In the CUDA
  kernel every step goes through `__fdiv_rn` / `__fmul_rn` / `__fadd_rn`, because
  nvcc contracts `a * b + c` into an FMA by default and that would fuse two
  roundings into one.
* Measured on H20 (sm_90): the CUDA forward and backward are **byte-for-byte
  equal** to the PyTorch FP32 reference; BF16 and FP16 match within the shared
  contract tolerances.
* The blend must not be reassociated to `xt + (sigma - sigma_next) * v`. The two
  forms are algebraically identical but not bitwise identical; ablation probe H13
  exists to catch the substitution. Note that the terminal step (`sigma_next == 0`)
  cannot expose this: both forms collapse to `xt + sigma * v`, so coverage must
  include intermediate steps.

## Performance Notes

The operator is memory-bound: it reads `xt` and `v` and writes `x_next` and `x0`.
Measure with the operator CLI on the pinned hardware and record the actual
backend readback:

```bash
python scripts/check_operator.py --op h3_ode_step --candidate cuda \
  --device cuda --dtype bf16 --batch 4096 --seq 24 --arch-key sm90 --json
```

Report the exact GPU, driver, CUDA, PyTorch and Triton versions with the numbers.
On H20 over 24 configurations (channels 24/32, rows 256-16384, fp32/bf16/fp16) the
CUDA backend measured 1.30x geometric-mean forward and 1.37x backward speedup over
the PyTorch reference path; Triton measured 1.13x and 1.14x.

## Tests

```bash
python -m pytest tests/test_h3_ode_step.py -v
```

Coverage: declared-expression regression (including the H13 non-equality probe),
terminal step (`sigma_next == 0`), monotone grid, dtype coverage, non-power-of-two
channel widths (24 / 32), bitwise Axis-A invariance across batch position, batch
size, repeated execution and unrelated-row mutation, and fail-closed behaviour for
`sigma == 0`, `sigma_next > sigma`, non-finite sigma, shape/device mismatch,
unsupported dtype, bad per-row sigma length, empty input and non-contiguous input.

## Known Limitations

* No backward-into-sigma: `sigma` / `sigma_next` are scheduler scalars and are not
  differentiable inputs in this contract.
* ROCm has the portable Triton path only; no native HIP kernel has been written or
  validated for this operator.
* `sigma` is restricted to scalar or one-value-per-row. Higher-rank per-axis sigma
  layouts are unsupported and fail closed.
* `scripts/check_forward_invariance.py` and `scripts/check_gradient_invariance.py`
  do not accept this operator yet: their `--op` choices are a curated list of the
  existing chain ops. Invariance evidence for `h3_ode_step` therefore comes from
  `tests/test_h3_ode_step.py`. Extending those CLIs to enumerate new operators
  (`rl_engine/kernels/gtest/gradient_adapters.py`) is separate follow-up work.
* The `eta = 0` deterministic ODE gives no transition log-probability. Any
  stochastic sampler logprob is a separate contract (Phase C
  `stochastic_transition_contract`) and must not be derived from this operator.
