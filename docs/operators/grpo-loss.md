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

The backends above consume dense `[B, T, V]` logits and reduce with a plain masked
mean, so they are single-shard only. For vocab-parallel TP, or when the reduction
must be bitwise reproducible across parallel degrees, see [Tensor and Data
Parallel](#tensor-and-data-parallel) below.

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

Both Native and Triton normalize rewards in FP32 after subtracting each group's
first reward. They compute the mean of these differences, center them, and use
their mean squared value as the population variance. The standard deviation is
floored at `eps` (default `1e-6`). Removing the shared offset before reducing avoids
rounding a large absolute mean: FP32 rewards `[100000000., 100000008.]` produce
advantages `[-1., 1.]`. Constant groups, including seven identical FP32 rewards of
`100.1`, produce exactly zero advantages.

The Native path adds group-offset indexing and elementwise work proportional to
the number of rewards; normalization remains FP32 and linear in batch size.

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

## Tensor and Data Parallel

`DistributedGRPOLossOp`
(`rl_engine/kernels/ops/pytorch/loss/distributed_grpo_loss.py`)
**Every TP × DP degree produces bit-identical loss, per-sequence totals, and
gradients.** It is a reference backend on top of the deterministic
[vocab-parallel logprob](batch-invariant-logp.md#tensor-parallel); the backends
above stay the default single-GPU path.

The objective is elementwise — ratio, clipping, reference KL and the group-relative
advantage all act per token or per sequence — so the only place the parallel layout
can change the answer is the final sum over tokens.

1. Selected logprobs come from `VocabParallelLogprobOp`, so ratio and KL inherit
   cross-TP bitwise equality for free. TP needs no further handling here: by the
   time the objective sees a logprob, the vocabulary has already been reduced.
2. A DP rank owns each of its sequences **whole**. `sequence_shard_bounds` is a
   contiguous `[0, num_sequences)` partition in DP-rank order, exactly like
   `ShardingSpec.vocab_shard_bounds` for the vocabulary.
3. Two nested reductions, each with a contract-fixed extent: `padded_seq_len`
   token slots → one sequence total (entirely local, since the rank owns the
   whole sequence); `num_sequences` totals → the scalar numerator.
4. Only per-sequence totals cross a rank boundary. They travel by `all_gather`
   and are concatenated in DP-rank order into a `[num_sequences]` vector. The
   collective moves bytes and placement is an exact copy, so every degree
   performs identical arithmetic on identical inputs. `all_reduce` is excluded on
   purpose: its combine order follows the collective's topology, not the declared
   sequence order.
5. Advantages are replicated, not merged — rewards are one scalar per sequence,
   so every rank normalizes every group over the identical `[num_sequences]`
   tensor and keeps its own slice. This is what lets an advantage group straddle
   DP ranks with no extra machinery.
6. The normalizer divides by a **global** active-token count, gathered as
   integers so it is exact at every degree.

Step 3 is why the determinism argument is short: because no sequence's token sum
is ever split across ranks, there is no partial-sequence state to merge and no
alignment rule to get wrong.

```python
sequence_shard_bounds = ((0, 8),)              # DP=1
sequence_shard_bounds = ((0, 4), (4, 8))       # DP=2
```

### Context parallelism is out of scope

`cp_world_size` must be 1; anything else raises `LossContractError` rather than
silently reducing over a partial batch. CP is an attention-level concern — attention
is the only op with a cross-token dependency — and this operator consumes logits, by
which point CP has already been resolved upstream. Supporting it here would mean
splitting a sequence's token sum across ranks and reducing over the very axis the
logprob contract declares a *non-merge* axis. In a CP job that reduction belongs to
the caller. `cp_rank`/`cp_world_size` are carried for provenance only, mirroring
`ShardingSpec`.

### Normalizer semantics

`TokenNormalizer` makes the GRPO normalizer ambiguity explicit. The modes differ by
more than a scale factor once sequence lengths vary, so the choice is part of the
numerical identity and travels in the contract fingerprint.

| Mode | Denominator | Notes |
| --- | --- | --- |
| `global_active_tokens` (default) | active tokens in the global batch | Matches `NativeGRPOLossOp`'s masked mean at DP=1. Long sequences weigh more. |
| `per_sequence_then_mean` | per-sequence count, then mean over live sequences | Original GRPO form; sequences weigh equally. |
| `fixed_constant` | declared constant | Dr.GRPO form; independent of the mask. |

Usage goes through the contract-aware entry point:

```python
from rl_engine.kernels.registry import kernel_registry

dispatched = kernel_registry.get_loss_op(contract)   # GRPOLossContract from
result = dispatched.op.apply(                        # rl_engine.kernels.loss_contract
    policy_local_logits,   # [n, local_vocab] differentiable
    action_ids,            # [n]
    old_logps,             # [n]
    rewards,               # [local_num_sequences]
    contract=contract,
    ref_local_logits=ref_local_logits,   # required when beta > 0
    tp_group=tp_group,     # vocab-parallel subgroup
    dp_group=dp_group,     # data-parallel subgroup
)
result.loss.backward()     # gradients flow into policy_local_logits only

loss, policy_loss, kl = result   # unpacks like the single-GPU op
```

A preflight `all_gather_object` runs on **both** the DP and TP axes before any other
collective. Neither alone suffices: the loss is replicated across TP, so two TP
siblings disagreeing on `beta` would compute different losses for one sharded model,
and the logprob path's own preflight cannot see that because `beta` is not part of
the logprob contract. Other loud failures, with no silent fallback: sequence bounds
that are non-contiguous or leave a gap; a nested logprob contract whose token count
disagrees with the owned sequences; a determinism scope stronger or weaker than the
logprob path's; and population-std advantages over a singleton group.

### Comparing configurations

Compare `per_sequence_policy` / `per_sequence_kl`, not the scalar loss. Measured on
this operator's test inputs, regrouping the token sum — a real change of the
summation tree — moves the per-sequence vector in 12 of 12 seeds but the scalar loss
in only 5 of 12: averaging `num_sequences` totals into one fp32 number rounds most
reorderings away. A drift report that compares only the scalar will under-report
reduction differences. `GRPOLossResult` exposes both, plus `advantages`,
`per_sequence_active_tokens`, and a `provenance` dict.

Bitwise here means *across parallel degrees on one PyTorch build and GPU model*. It
rests on PyTorch's reduction kernels being deterministic for a fixed shape on a fixed
device; cross-version and cross-architecture equality is neither tested nor claimed.

## Accuracy

Reference semantics (`NativeGRPOLossOp`):

```python
# advantages: group-normalized rewards (population std, unbiased=False)
grouped = rewards.float().view(-1, samples_per_prompt)
grouped = grouped - grouped[:, :1]
adv = (grouped - grouped.mean(1, keepdim=True)) / grouped.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
adv = adv.reshape(-1)[:, None].expand_as(completion_mask).masked_fill(~completion_mask, 0.0)

# ratio + kl from the ratio_kl op (mask-before-exp; see ratio-kl.md)
ratio, kl = ratio_kl(policy_logits, ref_logits, action_ids, completion_mask, old_logps)
policy = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
loss = masked_mean(policy, completion_mask) + beta * masked_mean(kl, completion_mask)
```

The Triton op matches the native reference (forward and backward) to `atol=1e-4`.

For `DistributedGRPOLossOp`, the reference-equals-policy identity is exact rather
than approximate: with `ref_logits is policy_logits` and `old_logps == logp_policy`
the ratio is `exp(0) = 1` bitwise, so the result is invariant to the clip epsilon,
and the KL is exactly `0.0`.

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
python -m pytest tests/test_grpo_loss.py -v            # single-GPU backends
python -m pytest tests/test_grpo_loss_contract.py -v   # TP/DP/CP contract, CPU only
python -m pytest tests/test_distributed_grpo_loss.py -v
```

`test_grpo_loss.py` covers the native reference (group advantages + loss from logits),
Triton forward/backward vs native, masked-token invariance, an SGD loss step, and
registry dispatch. Reward-normalization regressions cover large offsets against
an FP64 reference, unrepresentable absolute means, constant non-power-of-two groups,
both group specifications, and variable-sized groups with distinct offsets and
singletons across Native CPU, Native CUDA, and Triton CUDA. Native empty batches
are also covered. Triton tests skip without CUDA + Triton.

`test_distributed_grpo_loss.py` covers every `(TP, DP)` combination reachable with
four ranks — `tp2`, `tp4`, `dp2`, `dp4`, `tp2xdp2` — each compared bitwise against a
single-rank GPU baseline, plus the KL=0 identity, run-to-run stability, negative
controls, and the DP- and TP-axis preflight guards. Larger degrees run unchanged on a
bigger node. Multi-rank tests need one GPU per rank and skip otherwise; they are
deliberately small (1000-token vocabulary, 8 sequences of 32 slots) and each worker
caps itself with `torch.cuda.set_per_process_memory_fraction`, so the suite can share
a node with a running training job.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py`
- `rl_engine/kernels/ops/triton/loss/grpo_loss.py`
- `rl_engine/kernels/ops/triton/loss/ratio_kl.py`, `rl_engine/kernels/ops/pytorch/loss/ratio_kl.py`
- `rl_engine/kernels/ops/pytorch/loss/distributed_grpo_loss.py`
- `rl_engine/kernels/loss_contract.py`
- `rl_engine/kernels/registry.py` (`register_loss_backend`, `get_loss_op`)
- `tests/test_grpo_loss.py`, `tests/test_grpo_loss_contract.py`, `tests/test_distributed_grpo_loss.py`
- `benchmarks/benchmark_ratio_kl.py`
