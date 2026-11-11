# TP-sharded Reward and Critic models

This page describes how rollout token IDs reach a tensor-parallel (TP) Reward or
Critic model and how scalar rewards return to the rollout ranks (issue #44). The
building blocks are in `rl_engine/integrations/common/weights.py` and
`rl_engine/integrations/common/tp_scalar.py`. Import `TensorPacketBridge` from
`integrations.common.weights` and `TPModelInputs` and `TPScalarExecutor` from
`integrations.common.tp_scalar`, as shown below.

## Data path

1. **Rollout tokens to the TP group.** `TensorPacketBridge` broadcasts an agreed,
   ordered tensor schema (for example `input_ids`, `attention_mask`,
   `position_ids`) with dynamic shapes from one source rank. Use a group that
   contains the rollout leader and every rank of the Reward TP group.
2. **TP forward.** Each TP rank runs `TPScalarExecutor.rewards()` on the same
   tokens. Rewards are pooled at the last valid token **before** any
   communication, so a `local_sum` model needs one reduce of `B` scalars to the
   TP leader. No hidden states or vocabulary logits are gathered.
3. **Rewards back to rollout.** The TP leader broadcasts the `[B]` rewards with
   another `TensorPacketBridge` whose group contains the TP leader and the
   rollout ranks.

```python
from rl_engine.integrations.common.tp_scalar import TPModelInputs, TPScalarExecutor
from rl_engine.integrations.common.weights import TensorPacketBridge

fields = ("input_ids", "attention_mask", "position_ids")
tokens = TensorPacketBridge(group=to_reward, source=rollout_leader, device=device)
packet = tokens.broadcast(fields, batch if rank == rollout_leader else None)

executor = TPScalarExecutor(model, group=tp_group, leader=reward_leader, output_layout="local_sum")
rewards = executor.rewards(TPModelInputs(*(packet[name] for name in fields)))

back = TensorPacketBridge(group=to_rollout, source=reward_leader, device=device)
rewards = back.broadcast(("rewards",), {"rewards": rewards} if rank == reward_leader else None)
```

## Contracts

- **`TensorPacketBridge`:**
  - The source is a **global rank**, and the process group must be passed explicitly.
  - All group members call `broadcast` in the same order with the same field tuple.
  - Only the source supplies tensors, and they must already be on the configured device.
  - Transported tensors are detached snapshots, not distributed autograd edges.
  - Schema errors are raised on every member before any payload is sent.
- **`TPScalarExecutor` model adapters:**
  - The adapter receives `input_ids`, `attention_mask`, `position_ids` and `use_cache=False`.
  - It returns `[B, S]` values, or `[B]` or `[B, S]` rewards.
  - `output_layout="replicated"` means the model has already completed its own head communication.
  - `output_layout="local_sum"` means each rank returns an additive contribution. Include any bias on exactly one rank.
  - The layout must be the same on every rank of a TP group.
- **`rewards()`:**
  - Runs without gradients, in eval mode, and restores the caller's module modes afterwards.
  - Only the leader receives the tensor; other ranks get `None`.
- **`values()`:**
  - Returns differentiable `[B, S]` Critic values on every TP rank.
  - Its `local_sum` reduction (`sum_tp_contributions`) is SUM in forward and identity in backward, because every TP rank evaluates the same loss. It is not a general distributed-autograd SUM.
- **`clip_tp_grad_norm_`:**
  - Clips Critic gradients by the TP-global norm.
  - Takes a layout per parameter: `sharded`, or `replicated` if the gradient is already synchronized.
  - Replicated gradients count once in the norm.
- **Equivalence:** native PyTorch reductions give numerical, not bitwise, equivalence across TP sizes.

`TPScalarExecutor.head_metrics` and `TensorPacketBridge.metrics` count reduce
calls and payload bytes, excluding the small validation collectives.

## Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/integrations/common/test_tp_scalar.py
# Additional NCCL check on two GPUs:
RUN_TP_CUDA=1 OMP_NUM_THREADS=1 python -m pytest -q tests/integrations/common/test_tp_scalar.py -k nccl
```
