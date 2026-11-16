# Shared training log-probabilities

`rl_engine.integrations.common.training_logp` provides `TrainingLogpInputs`,
`compute_training_logp`, and `LinearLogpResult`. Training adapters can call this
interface without importing an orchestrator.

The caller supplies local token rows, global target IDs, vocabulary bounds, and
TP/CP coordinates. TP owns vocabulary shards; CP describes token ownership and
is not a log-probability reduction axis. Results contain `[T, 1]` log-probabilities,
optional `[T]` entropy, backend/contract IDs, and provenance.

```python
from rl_engine.integrations.common.training_logp import (
    TrainingLogpInputs,
    compute_training_logp,
)

inputs = TrainingLogpInputs(
    logits=local_logits,
    target_ids=target_ids,
    real_vocab_size=real_vocab_size,
    padded_vocab_size=padded_vocab_size,
    vocab_start_index=vocab_start_index,
    tp_group=tp_group,
    tp_rank=tp_rank,
    tp_world_size=tp_world_size,
    cp_rank=cp_rank,
    cp_world_size=cp_world_size,
    cp_layout=cp_layout,  # "single", "zigzag", or "allgather"
    num_vocab_tiles=num_vocab_tiles,
)
result = compute_training_logp(inputs)
```

Without a `linear_logp` wrapper, the scorer uses the existing explicit WS2
vocabulary-parallel backend. These logits must already have the intended
scaling; this path does not apply temperature or top-p replay.

For strict projection/reuse, also supply `hidden`, `weight`, vocabulary shard
start, and an initialized `LinearLogpWrapper` via `linear_logp=wrapper`.
`temperature` applies when recomputing the projection. With
`reuse_local_logits=True`, supply `local_logits` and set
`local_logits_temperature` to the remaining scaling, or `None` for already
scaled logits. Dense replay masks and sparse token IDs use the existing wrapper
paths. `with_entropy_grad=False` keeps entropy metric-only.

Vime's provider converts its request before calling the shared scorer or the
active Megatron integration. Vime environment settings, metadata conventions,
and legacy temperature markers remain in that adapter. Existing provider import
paths and result/exception exports remain available. Other orchestrators should
construct `TrainingLogpInputs` from their own request formats.

This interface does not install verl, add NPU operators, or establish hardware
consistency. Those integrations still require device tests.
