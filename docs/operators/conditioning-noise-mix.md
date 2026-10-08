# Conditioning Noise Mix

## Summary

`conditioning_noise_mix` blends a sample with caller-provided noise for H3
flow-matching conditioning ([issue #420](https://github.com/RL-Align/RL-Kernel/issues/420)):

```text
out = timestep * sample + (1 - timestep) * noise
```

The formula and public argument order follow Diffusers
[`MiniMaxH3Scheduler.scale_noise`](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/schedulers/scheduling_minimax_h3.py).

## Entry Point

```python
import torch
from rl_engine.kernels.registry import kernel_registry

sample = torch.randn(2, 257, device="cuda", dtype=torch.bfloat16, requires_grad=True)
noise = torch.randn_like(sample, requires_grad=True)
timestep = torch.tensor([0.25, 0.75], device=sample.device, dtype=sample.dtype)
mix = kernel_registry.get_op("conditioning_noise_mix", device="cuda")
out = mix(sample, timestep, noise)
out.sum().backward()
```

Both backends expose `forward(sample, timestep, noise)` and
`forward_fp32(sample, timestep, noise)`. `forward` returns the input dtype;
`forward_fp32` promotes the inputs and returns FP32. Calling the operator uses
`forward`; keyword arguments use the same names.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `ConditioningNoiseMixCudaOp` | `_C.conditioning_noise_mix_forward` / `_C.conditioning_noise_mix_backward` | NVIDIA CUDA forward and first-order backward. |
| PyTorch native | `NativeConditioningNoiseMixOp` | None | CPU reference; can also run explicitly on CUDA as the eager baseline. |

The CUDA kernels are part of the shared `rl_engine._C` extension. Build it with
the normal [source installation](../getting_started/installation.md#from-source).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `sample` | `[B, ...]` | fp16, bf16 or fp32 | Nonempty and contiguous. |
| `timestep` | Scalar, one value, or `[B]` | Python float or fp16/bf16/fp32 tensor | Fixed metadata; tensor must be contiguous and on the sample device. |
| `noise` | Same as `sample` | Same as `sample` | Contiguous and on the sample device. |
| output | Same as `sample` | `forward`: input dtype; `forward_fp32`: fp32 | New allocation; inputs are not modified. |

A per-sample timestep may have trailing singleton axes up to the input rank.
For `forward`, timestep is converted to the sample dtype before arithmetic.
Callers must supply finite timestep values in `[0, 1]`; value ranges are not
checked at runtime. First-order gradients are supported for `sample` and `noise`.

## Dispatch Behavior

`kernel_registry.get_op("conditioning_noise_mix", device="cuda")` selects
`ConditioningNoiseMixCudaOp`; `device="cpu"` selects `NativeConditioningNoiseMixOp`.
CUDA lookup raises if the native extension or symbols are unavailable; there is
no PyTorch fallback in the CUDA priority list. ROCm, NPU and MUSA have no registered
implementation.

## Accuracy

The FP32 reference evaluates the formula using the same input values promoted to
FP32. CUDA preserves the eager PyTorch operation boundaries: FP16/BF16 results
round to the input dtype after subtraction, each multiplication and addition.
This differs from computing the entire expression in FP32 and casting only once.

The tests distinguish three comparisons:

- **Forward and gradient accuracy:** compare against an independent FP32 formula
  and FP32 input gradients using the shared `elementwise` tolerances.
- **Batch invariance:** require bitwise agreement across repeated execution,
  permutation, chunking, batch position, padding and unrelated rows.
- **Eager parity:** additionally check CUDA outputs and both input gradients
  bitwise against the same-dtype PyTorch reference.

Accuracy and invariance use the comparison roles in the
[WS1 numerical standard](../design/ws1-numerical-precision-standard.md), with
thresholds from `rl_engine/kernels/gtest/tolerance_contract.json`.

## Performance Notes

```bash
python benchmarks/benchmark_conditioning_noise_mix.py
```

The benchmark compares the public CUDA and eager PyTorch wrappers on identical
GPU-resident inputs after checking outputs and both input gradients bitwise.
Default shapes are `3x257`, `1x32x400`, `1x24x1x48x80` and `1x24x17x48x80`,
with FP16, BF16 and FP32 inputs.

The shared `PerformanceProfiler` measures inference forward, backward on a
prebuilt graph, and a fresh forward plus backward. Wrapper validation, allocation
and autograd dispatch are timed; input generation and correctness checks are not.
The JSON results in `reports/conditioning-noise-mix/results.json` include latency,
timing variation, throughput and peak allocation. `report.md` summarizes latency
and speedup.
Use `--help` for workload and output options.

## Tests

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pytest tests/test_conditioning_noise_mix.py -q
python -m pytest tests/test_benchmark_conditioning_noise_mix.py -q
python scripts/check_operator.py --op conditioning_noise_mix --candidate cuda \
  --device cuda --dtype bf16 --batch 2 --seq 7 --normalized-dim 257 --check-grad
```

Coverage includes input validation, scalar/per-sample timesteps, dtype rounding,
forward/backward accuracy, batch invariance, registry dispatch and CUDA stream
ordering. CPU-only runs skip native CUDA tests.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/conditioning_noise_mix.py` — PyTorch reference
  and shared input validation
- `rl_engine/kernels/ops/cuda/conditioning_noise_mix.py` — CUDA autograd wrapper
- `csrc/cuda/conditioning_noise_mix.cu` — CUDA forward/backward kernels
- `benchmarks/benchmark_conditioning_noise_mix.py` — public-wrapper benchmark

## Known Limitations

- No trainable or per-element timesteps, strided inputs, empty tensors or FP64.
- CUDA supports first-order gradients only.
- SM90 is the integration target; runtime correctness and performance on SM90,
  and full H3 integration, remain pending validation.
