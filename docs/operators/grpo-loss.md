# GRPO Loss

GRPO Loss computes the Group Relative Policy Optimization objective for RL post-training:
it normalizes raw sequence rewards within each generation group into advantages, then
evaluates the clipped surrogate objective plus a reference-KL penalty over the active
completion tokens. It targets the GRPO training step, where a naive PyTorch implementation
allocates several broadcasted `[batch, completion_len]` intermediates and a per-token
advantage tensor.

The operator consumes **logits** directly and builds on the [Policy Ratio + KL
Penalty](ratio-kl.md) operator: the per-token `policy_ratio` and `kl_penalty` come from the
fused ratio/KL kernel (logits → ratio/KL via online softmax), and the group-normalized advantages + clipped surrogate are applied on top:

```
logits --[ratio_kl op]--> (ratio, kl) --[group adv + clipped surrogate]--> loss
```

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

grpo_loss = kernel_registry.get_op("grpo_loss")

loss, policy_loss, kl = grpo_loss(
    policy_logits,        # [B, T, V] current policy logits (differentiable)
    ref_logits,           # [B, T, V] frozen reference logits
    action_ids,           # [B, T] token taken at each position
    old_logps,            # [B, T] cached behavior-policy log-probs
    rewards,              # [B]
    completion_mask,      # [B, T]
    clip_eps=0.2,
    beta=0.04,
    samples_per_prompt=8,  # uniform groups; or pass group_boundaries=[...]
)

loss.backward()           # gradient flows into policy_logits
```

Note: `B = num_prompts * samples_per_prompt`. `old_logps` is the cached behavior-policy
log-prob from rollout, required for the importance ratio (see [ratio-kl](ratio-kl.md)).

### Group specification

Provide **exactly one** of:
- `samples_per_prompt: int` — uniform groups (every prompt has the same number of samples).
- `group_boundaries` — CSR-style offsets of length `num_groups + 1` (e.g. `[0, 8, 16, 24]`)
  for variable-sized groups.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA / ROCm | `TritonGRPOLossOp` | `ratio_kl` + `_group_norm_kernel` | Fused ratio/KL + analytic backward. |
| PyTorch fallback | `NativeGRPOLossOp` | None | Reference path; CPU and Triton-less GPUs. |

The Triton op composes the [`ratio_kl`](ratio-kl.md) kernel (per-token `ratio`/`kl` from
logits, with the analytic backward into `policy_logits`) with the `_group_norm_kernel`
(per-group reward mean/std in registers). The clipped surrogate + reference-KL reduction is
a thin autograd-friendly PyTorch layer — no bespoke GRPO loss kernel is needed. The native
op mirrors this using `NativeRatioKLOp`.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `policy_logits` | `[B, T, V]` | float | Differentiable input; contiguous. |
| `ref_logits` | `[B, T, V]` | float | Constant (no grad); contiguous. |
| `action_ids` | `[B, T]` | int | Token id per position (in `[0, V)`). |
| `old_logps` | `[B, T]` | float | Constant (no grad). |
| `rewards` | `[B]` | float | One scalar per sequence. |
| `completion_mask` | `[B, T]` | bool / {0,1} | 2-D; `True` marks active tokens. |
| `loss` (output) | scalar | float32 | `policy_loss + beta * kl`. |
| `policy_loss`, `kl` (output) | scalar | float32 | Detached reporting values. |

Gradients flow into `policy_logits` only (`ref_logits` is frozen; `old_logps` is cached).

## Accuracy

Reference semantics (`NativeGRPOLossOp`):

```python
# advantages: group-normalized rewards (population std, unbiased=False)
grouped = rewards.view(-1, samples_per_prompt)
adv = (grouped - grouped.mean(1, keepdim=True)) / grouped.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
adv = adv.reshape(-1)[:, None].expand_as(completion_mask).masked_fill(~completion_mask, 0.0)

# ratio + kl from the ratio_kl op (mask-before-exp; see ratio-kl.md)
ratio, kl = ratio_kl(policy_logits, ref_logits, action_ids, completion_mask, old_logps)
policy = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
loss = masked_mean(policy, completion_mask) + beta * masked_mean(kl, completion_mask)
```

The Triton op matches the native reference (forward and backward) to `atol=1e-4`.

## Performance Notes

The cost is dominated by the [`ratio_kl`](ratio-kl.md) stage (the vocab-dimension work);
reward normalization and the clipped-surrogate reduction operate on `[B, T]` tensors and are
negligible.

```bash
python benchmarks/benchmark_grpo_loss.py
python benchmarks/benchmark_grpo_loss.py --configs "4,8,256,32768;4,8,256,131072"
```

Indicative results (RTX PRO 6000, SM120, fp16, B=32, T=256; native PyTorch vs Triton):

| shape (P×S×L×V) | fwd speedup | fwd+bwd speedup | peak fwd VRAM (native → Triton) |
| --- | --- | --- | --- |
| 4×8×256×32768 | 5.2× | 2.8× | 2048 MB → ~0 MB |
| 4×8×256×50257 | 7.3× | 2.4× | 3141 MB → ~0 MB |
| 4×8×256×131072 | 10.3× | 3.4× | 8192 MB → ~0 MB |

Both speedup and the VRAM advantage grow with vocabulary size: the native path materializes
the `[B, T, V]` log-softmax (forward peak scales with `V`), while the fused op streams it
online — the forward peak is independent of `V`.

## Tests

```bash
python -m pytest tests/test_grpo_loss.py -v
```

Covers the native reference (group advantages + loss from logits), Triton forward/backward
vs native, masked-token invariance, an SGD loss step, and registry dispatch. Triton tests
skip without CUDA + Triton.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py`
- `rl_engine/kernels/ops/triton/loss/grpo_loss.py`
- `rl_engine/kernels/ops/triton/loss/ratio_kl.py`, `rl_engine/kernels/ops/pytorch/loss/ratio_kl.py`
- `rl_engine/kernels/registry.py`
- `tests/test_grpo_loss.py`
- `benchmarks/benchmark_ratio_kl.py`
