# Policy Ratio + KL Penalty

The Ratio/KL operator is the fused front-end of the PPO/GRPO loss: from policy and
reference **logits** it computes, per active token, the importance `policy_ratio` and the
reference `kl_penalty` in a single kernel launch.

It is the upstream stage that the GRPO/PPO loss wrappers sit on top of:

```
logits --[ratio_kl op]--> (ratio, kl) --[clipped surrogate + beta*kl]--> loss
```

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

ratio_kl = kernel_registry.get_op("ratio_kl")

policy_ratio, kl_penalty = ratio_kl(
    policy_logits,    # [B, T, V] current policy logits (differentiable)
    ref_logits,       # [B, T, V] frozen reference logits
    action_ids,       # [B, T] token taken at each position
    attention_mask,   # [B, T] 1 marks active completion tokens
    old_logps,        # [B, T] cached behavior-policy log-probs
)

# downstream clipped surrogate (advantages from GAE for PPO / group-norm for GRPO)
surrogate = -torch.minimum(
    policy_ratio * advantages,
    torch.clamp(policy_ratio, 1 - clip_eps, 1 + clip_eps) * advantages,
)
loss = masked_mean(surrogate, attention_mask) + beta * masked_mean(kl_penalty, attention_mask)
loss.backward()       # gradient flows into policy_logits
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA / ROCm | `TritonRatioKLOp` | Triton JIT kernels | Fused forward + analytic backward. |
| PyTorch fallback | `NativeRatioKLOp` | None | Reference path; CPU and Triton-less GPUs. |

```
grad_policy_logits[v] = c * (1[v == action] - softmax_policy(v))
```

so the backward also avoids materializing any `[B, T, V]` probability tensor (only the
unavoidable `[B, T, V]` gradient output is written).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `policy_logits` | `[B, T, V]` | float (fp16/bf16/fp32) | Differentiable input; contiguous. |
| `ref_logits` | `[B, T, V]` | float | Constant (no grad); contiguous. |
| `action_ids` | `[B, T]` | int | Token id per position (in `[0, V)`). |
| `attention_mask` | `[B, T]` | bool / {0,1} | `True`/1 marks active tokens. |
| `old_logps` | `[B, T]` | float | Constant (no grad). |
| `policy_ratio` (output) | `[B, T]` | float32 | `exp(logp_policy - old_logp)`. |
| `kl_penalty` (output) | `[B, T]` | float32 | `exp(d) - d - 1`, `d = logp_ref - logp_policy`. |

## Accuracy

Reference semantics (`NativeRatioKLOp`, mask-before-exp matching `grpo_loss`):

```python
logp_policy = log_softmax(policy_logits, -1).gather(-1, action_ids)   # selected token logp
with torch.no_grad():
    logp_ref = log_softmax(ref_logits, -1).gather(-1, action_ids)

delta = (logp_policy - old_logps).masked_fill(~mask, 0.0)
diff  = (logp_ref - logp_policy).masked_fill(~mask, 0.0)
policy_ratio = torch.exp(delta)                      # exp(0) = 1 on inactive tokens
kl_penalty   = torch.exp(diff) - diff - 1.0          # k3 estimator
```

The Triton op matches the native reference on `ratio` and `kl` (forward) and on the
`policy_logits` gradient (backward) to `atol=1e-4` (fp32). In fp16 the KL difference is
~`1e-4` from rounding; the ratio difference is ~`1e-9`.

## Performance Notes

```bash
python benchmarks/benchmark_ratio_kl.py
python benchmarks/benchmark_ratio_kl.py --g-sizes 8 --completion-lens 512 --vocab-sizes 32768,131072
```

Indicative forward-only results (fp16, `B=16`, `T=512`):

| vocab | active tokens | forward speedup | peak VRAM (native → Triton) |
| --- | --- | --- | --- |
| 32,768 | 8,192 | 6.8× | 3.0 GB → 1.0 GB |
| 131,072 | 8,192 | 10.0× | 12.0 GB → 4.0 GB |

## Tests

```bash
python -m pytest tests/test_ratio_kl.py -v
```

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/ratio_kl.py`
- `rl_engine/kernels/ops/triton/loss/ratio_kl.py`
- `rl_engine/kernels/registry.py`
- `tests/test_ratio_kl.py`
- `benchmarks/benchmark_ratio_kl.py`
