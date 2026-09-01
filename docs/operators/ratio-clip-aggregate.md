# Ratio Clip Aggregate

`ratio_clip_aggregate` fuses the policy-ratio clipping, active-token masking,
loss reduction, optional penalty reduction, and clip-fraction metric used by
PPO and GRPO training. It is the downstream companion to `ratio_kl`: the latter
produces per-token ratios and KL terms from logits, while this operator consumes
those compact tensors without materializing broadcast advantages or surrogate
loss tensors.

## Public entry point

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("ratio_clip_aggregate", device=ratio.device)
loss, policy_loss, mean_penalty, clip_fraction = op(
    ratio,
    advantages,
    completion_mask,
    clip_low=0.2,
    clip_high=0.2,
    penalty_terms=kl_terms,
    penalty_coef=0.04,
)
```

## Contract

| Input | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `ratio` | `[B, T]` or `[N]` | FP16, BF16, FP32 | Per-token policy importance ratio |
| `advantages` | `[B, T]`, `[N]`, or `[B]` | floating point | Detached target; `[B]` is supported only with `[B, T]` ratios |
| `mask` | same as `ratio` | bool or integer | Nonzero entries participate in the reduction |
| `penalty_terms` | same as `ratio` | floating point | Optional per-token KL or other additive penalty |
| `clip_low` | scalar | Python float | Lower bound is `1 - clip_low`; must satisfy `0 <= clip_low < 1` |
| `clip_high` | scalar | Python float | Upper bound is `1 + clip_high`; must be non-negative |
| `penalty_coef` | scalar | Python float | Multiplier applied to `mean_penalty` |

All tensor inputs must be on the same device. Inputs may be non-contiguous; each
backend normalizes layout internally. Advantages are RL targets and must not
require gradients.

The four scalar FP32 outputs are:

1. `loss = policy_loss + penalty_coef * mean_penalty`;
2. the masked mean clipped policy loss;
3. the masked mean optional penalty, or zero when no penalty is supplied;
4. the fraction of active ratios outside the clip interval.

An all-false mask returns finite zeros for every output and produces zero input
gradients.

## Semantics

For every active token:

```python
clipped_ratio = ratio.clamp(1 - clip_low, 1 + clip_high)
policy_term = -torch.minimum(
    ratio * advantage,
    clipped_ratio * advantage,
)
```

The Triton backward computes analytical gradients for `ratio` and
`penalty_terms`. `policy_loss` and `mean_penalty` remain independently
differentiable outputs; `clip_fraction` is a metric and is non-differentiable.

## Backends and fallback

| Device | Preferred backend | Fallback |
| --- | --- | --- |
| NVIDIA CUDA | `TritonRatioClipAggregateOp` | `NativeRatioClipAggregateOp` |
| AMD ROCm | `TritonRatioClipAggregateOp` | `NativeRatioClipAggregateOp` |
| CPU | `NativeRatioClipAggregateOp` | N/A |

For up to 65,536 tokens, the Triton implementation uses a tuned single-program
reduction. Larger inputs use fixed 256-token tiles followed by a fixed-order
final reduction. Neither path uses `atomicAdd` or copies the active-token count
to the CPU. Per-sequence advantages are gathered directly in the kernel, so no
`[B, T]` broadcast tensor is allocated.

## Integration

`TritonGRPOLossOp` composes:

```text
logits -> ratio_kl -> ratio_clip_aggregate -> scalar loss
```

Group reward normalization remains separate because it reduces over generation
groups rather than tokens. The ratio/KL online-softmax work from `ratio_kl` is
not duplicated.

## Validation and benchmark

- Correctness and gradients: `tests/test_ratio_clip_aggregate.py`
- GRPO integration: `tests/test_grpo_loss.py`
- Performance and peak memory:
  `python benchmarks/benchmark_ratio_clip_aggregate.py --warmup 100 --iterations 500`

The native backend is the semantic reference. FP16/BF16 comparisons use FP32
accumulation and dtype-appropriate tolerances.

Indicative results on an NVIDIA RTX PRO 5000 Blackwell with PyTorch 2.11,
CUDA 12.9, and Triton 3.6 are below. Inputs are FP32 with 90% active tokens,
per-sequence advantages, and a penalty tensor; timings use 100 warmups and 500
iterations.

| Shape | PyTorch fwd | Triton fwd | Speedup | PyTorch fwd+bwd | Triton fwd+bwd | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 × 256 | 0.144 ms | 0.038 ms | 3.78× | 0.409 ms | 0.192 ms | 2.13× |
| 128 × 1024 | 0.152 ms | 0.051 ms | 2.98× | 0.423 ms | 0.208 ms | 2.03× |
| 256 × 4096 | 0.154 ms | 0.051 ms | 3.00× | 0.425 ms | 0.211 ms | 2.01× |
| 256 × 16384 | 0.318 ms | 0.051 ms | 6.25× | 0.664 ms | 0.211 ms | 3.15× |

Peak forward intermediates at 4.19M tokens fall from 64.0 MiB to 0.3 MiB.
Results are workload- and device-specific; rerun the benchmark for deployment
hardware.

## Limitations

- V1 implements PPO/GRPO clipped-surrogate semantics. It does not claim GSPO,
  DAPO, CISPO, or Dr. GRPO objective support.
- Advantages are intentionally non-differentiable targets.
- The deterministic staged reduction supports at most 16,777,216 tokens per
  invocation; larger inputs must be chunked by the caller.
