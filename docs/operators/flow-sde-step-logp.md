# Qwen-Image Flow SDE step and log-probability

Issue [#386](https://github.com/RL-Align/RL-Kernel/issues/386), WS1
`flow_sde_step_logp`. Implementation status: **draft, small-shape validation passed**.
The native extension builds on RTX 4090 D with PyTorch 2.12.1+cu130 / CUDA 13.0.
The focused suite passed 54 tests (12 full-resolution cases deselected), and
FP32/BF16 gtest forward/backward checks passed at `[2,7,64]`.
Full-resolution qualification and benchmarks have not run. This page defines the
proposed v1 arithmetic contract; it does not qualify the full training/rollout chain.

## Purpose and formula

Advance the gathered, CFG-combined latent and evaluate the mean Gaussian
log-density of the observed transition. The source is the `sde` branch of
[Flow-GRPO's scheduler patch](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py),
revision `879042cf5707f8b90daa98d147d7deac2317c5da`. The Qwen-Image pipeline
imports this function. CPS, ODE, sigma-table generation, CFG, RNG generation,
distributed gather and model integration are outside this operator.

Let `s=sigma`, `sn=sigma_next`, `dt=sn-s`, `l=noise_level` and
`sm=sigma_max` (the scheduler's **second FP32 sigma-table entry**, not the
maximum sigma value). In FP32, in the written order:

```text
std = sqrt(s / (1 - where(s == 1, sm, s))) * l
v = std * std
a = 1 + (v / (2 * s)) * dt
b = (1 + (v * (1 - s)) / (2 * s)) * dt
tau = std * sqrt(-dt)
mean = sample * a + model_output * b
target = mean + tau * noise                  # sampling
target = prev_sample                        # replay
r = detach(target) - mean
density = -(r * r) / (2 * (tau * tau)) - log(tau) - log(sqrt(FP32(2*pi)))
logp = fixed_mean(density over all non-batch dimensions)
```

The density includes Gaussian constants and is a **mean**, not a sum, matching
Flow-GRPO. There is no BF16 output rounding: the caller decides any subsequent
model-input conversion and must record it as part of the trajectory contract.
Source PyTorch reductions are replaced by the declared canonical reduction;
agreement with production mean reductions is an accuracy comparison, not a
byte-equality claim. This v1 proposal still needs integration into the architecture
fingerprint owned by a separate issue row.

## Interface

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("flow_sde_step_logp", device="cuda")
rollout = op(
    sample=x, model_output=velocity, sigma=0.75, sigma_next=0.5,
    sigma_max=0.98, noise_level=0.7, noise=fixed_fp32_noise,
)
training = op(
    sample=x, model_output=recomputed_velocity, sigma=0.75, sigma_next=0.5,
    sigma_max=0.98, noise_level=0.7, prev_sample=rollout.prev_sample.detach(),
)
trace = op.execution_trace(x)
```

`FlowSDEResult` contains `(prev_sample, logp, mean, std_dev)`. Latent outputs
have input shape, while `logp` and `std_dev` are `[B]`. All outputs are FP32.

| Input | Contract |
| --- | --- |
| sample/model_output | Same nonempty `[B,...]` shape/device; FP32 or BF16; finite values |
| noise | Detached FP32 tensor with latent shape/device; sampling only |
| prev_sample | Detached FP32 observed latent; replay only |
| sigma/sigma_next/sigma_max/noise_level | Python scalars or detached CPU FP32 scalar/`[B]` tensors |
| Schedule | `0 <= sigma_next < sigma <= 1`, `sigma > 0`, `0 < sigma_max < 1` |
| Noise scale | Positive; finite representable coefficients and transition variance |
| Geometry | `1 <= B <= 65535`; at most `2^24` latent elements per sample |

Noncontiguous latent inputs are normalized to contiguous logical row order.
Exactly one of noise/prev_sample is required. The op consumes no RNG; the caller
owns noise seed, generator device, dtype, consumption order and sample identity.
Scalar schedule metadata is expanded per sample; per-sample values must travel
with that sample when permuting or batching. GPU schedule tensors are rejected.
NaN/Inf latent values are outside the declared input contract; metadata is
validated on the CPU before launch. CUDA graph capture is not qualified.

## Backends, precision and reduction

| Backend | Implementation | Dispatch |
| --- | --- | --- |
| CPU/PyTorch | Independent FP32 formula, autograd, explicit reduction tree | Explicit `device="cpu"` |
| NVIDIA CUDA | Native precise FP32 forward and VJP, fixed 256-thread tile | CUDA symbol availability required |
| ROCm/Ascend | Unimplemented | Explicit failure |

The CUDA class raises on missing symbols, CPU/ROCm inputs, fast-math builds and
unsupported inputs. The registry does not substitute the reference for CUDA.
`execution_trace` reports requested/actual backend, kernel identity, FP32
arithmetic, reduction order, reference revision, input geometry and no fallback.
Before the first call it is a descriptor with `execution_recorded=false`;
after a call it records that call's shape, mode, dtype and runtime/GPU profile.
CUDA recording means the native stages were submitted, without an implicit
device synchronization. It is not a replacement for checking asynchronous errors.
Using the CPU reference on GPU explicitly is supported for diagnostic accuracy
comparisons but is not a CUDA native execution claim.

Coefficients, step/mean, density, tile reduction and final mean use separate
launches. Arithmetic uses round-to-nearest add/sub/mul/div/sqrt intrinsics, with
no FMA contraction, atomics, TF32, Split-K or cross-sample reduction. Logarithms
use regular `logf`, not `__logf`; fast-math builds are rejected.

Flatten each sample in logical contiguous order, pad the last 256-element tile
with zeros, reduce adjacent pairs at strides 1, 2, ..., 128, then add tile sums
in ascending tile order starting at FP32 zero. Divide by the **unpadded** element
count. The reduction geometry is fixed and cannot be tuned in strict v1.
Batch size/position and launch scheduling do not change this per-sample tree.
Padding here is internal reduction-tail padding, not arbitrary image-token
padding. External latent padding must be removed before calling the operator.

CPU/GPU sqrt/log accuracy is checked under declared tolerances. Within one pinned
CUDA hardware/runtime profile, repeated runs, sampling-versus-replay and batch
layout changes require raw-byte equality. Cross-device/profile equality is not
claimed without qualification. The existing gtest tolerance contract is unchanged.

## Gradients

Sigma and noise are fixed metadata. The sampled/observed target is detached
**inside logp**. Let `D` be the unpadded element count:

```text
dmean = (grad_logp / D) * (target - mean) / tau^2
dmean += grad_mean
dmean += grad_target                 # sampling output only
grad_sample = a * dmean
grad_model_output = b * dmean
```

Replay targets and `std_dev` are nondifferentiable. CUDA backward is a native
elementwise VJP with no cross-sample reduction or fallback. Input BF16 gradients
are cast back by autograd at the input conversion boundary. Higher-order CUDA
gradients are unsupported and explicitly marked once-differentiable.
This local operator enables gradient propagation; full-model LoRA checks belong
to the integration work items.

## Validation commands

Small-shape validation performed on 2026-10-08 using Python 3.12.3, PyTorch
2.12.1+cu130, nvcc 13.0.88, driver 580.76.05 and RTX 4090 D (SM89):

```bash
KERNEL_ALIGN_USE_FAST_MATH=0 KERNEL_ALIGN_DEV_RPATH=1 TORCH_CUDA_ARCH_LIST=8.9 \
  CUDA_HOME=/usr/local/cuda MAX_JOBS=2 \
  python -m pip install -e . --no-deps --no-build-isolation
python -m pytest tests/test_flow_sde_step_logp.py -q -k 'not reference_shapes or 7'
python scripts/check_operator.py --op flow_sde_step_logp --candidate cuda \
  --device cuda --dtype fp32 --batch 2 --seq 7 --check-grad
python scripts/check_operator.py --op flow_sde_step_logp --candidate cuda \
  --device cuda --dtype bf16 --batch 2 --seq 7 --check-grad
```

Result: **54 passed, 12 deselected**; both gtest invocations passed. The gtest
latent/mean outputs and gradients had zero maximum absolute difference, and
`logp` differed by `5.96046448e-08` from the eager reference. This is an accuracy
comparison; replay/batch invariance are separate raw-byte assertions in pytest.
No benchmark was executed. The following full checks remain pending.

From a compatible NVIDIA PyTorch environment with CUDA toolkit:

```bash
KERNEL_ALIGN_USE_FAST_MATH=0 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip install -e . --no-build-isolation
python -m pytest tests/test_flow_sde_step_logp.py -q
python scripts/check_operator.py --op flow_sde_step_logp --candidate cuda \
  --device cuda --dtype fp32 --batch 2 --seq 7 --check-grad
python scripts/check_operator.py --op flow_sde_step_logp --candidate cuda \
  --device cuda --dtype bf16 --batch 2 --seq 6889 --check-grad
python benchmarks/benchmark_flow_sde_step_logp.py --dtype fp32
python benchmarks/benchmark_flow_sde_step_logp.py --dtype bf16 --backward
python -m pytest rl_engine/tests/test_dispatch.py -q
mkdocs build --strict -f mkdocs.yaml
```

The generic gtest adapter compares the three differentiable sampling outputs;
pytest covers the nondifferentiable std output, replay, input failures, per-sample
schedules, strides, independent CPU FP32/FP64 formula accuracy, raw-byte batch
invariance including gradients, and a synthetic full trajectory. Reference shapes
are `[B,4096,64]`, `[B,6889,64]`, `[B,6032,64]` for 1024², 1328², 1664×928;
the short case is `[B,7,64]`. No Qwen-Image checkpoint is needed for these op tests.
The synthetic trajectory is not a full-model/e2e validation.

The benchmark compares production-style PyTorch mean, canonical reference and
native CUDA, and reports forward/optional backward device and wall latency,
incremental allocated memory, environment and trace. CUDA is mandatory; failures
are not relabeled as another backend. Device time includes launch gaps; wall time
also includes CPU metadata validation and copies. Performance has not been measured.

## Implementation files

- `rl_engine/kernels/ops/pytorch/diffusion/flow_sde_step_logp.py`
- `rl_engine/kernels/ops/cuda/diffusion/flow_sde_step_logp.py`
- `csrc/cuda/diffusion/flow_sde_step_logp.cu`
- `tests/test_flow_sde_step_logp.py`
- `benchmarks/benchmark_flow_sde_step_logp.py`
- Build/binding/stubs, registry and gtest registration follow PR #204's pattern.
