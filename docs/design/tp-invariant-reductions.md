# TP-Invariant Reductions

This design note defines the reference semantics for matching FSDP(TP=1)
training paths with TP>1 rollout or scoring paths.

Target identity:

```text
same model + same sequence + same policy state
=> selected logprobs and masked loss reductions are invariant to TP degree
```

## Vocab-Sharded Selected Logprob

For vocab-sharded logits, the denominator must be reduced globally:

```text
global_max = all_reduce_max(local_max(logit_shard))
global_sum = all_reduce_sum(sum(exp(logit_shard - global_max)))
global_lse = global_max + log(global_sum)
selected_logp = selected_target_logit - global_lse
```

The owning rank only provides the selected target logit. It must not compute a
local-only logsumexp. Averaging per-rank logsumexp values is also invalid.

The repository reference is `selected_logprobs_tp_reference(...)` in
`rl_engine.testing`. It accepts simulated vocab shards so tests can validate
TP=1 versus TP=2/4/8 without launching a distributed engine.

`selected_logprobs_distributed_tp_reference(...)` exercises the same semantics
with real `torch.distributed.all_reduce` collectives. Each rank owns one
contiguous vocab shard, contributes local max / exp-sum / selected target logit,
and receives the same selected-logprob tensor.

## Dtype Policy

The semantic reference uses:

- fp16/bf16/fp32 input logits;
- fp32 reduction state by default for max, exp-sum, log, selected-logit compare,
  and masked reductions;
- explicit output dtype only after the fixed reduction result is computed.

Backend kernels may choose lower-level implementation details, but parity tests
should compare against this contract and declare any backend-specific tolerance.

## Masked Loss Reductions

Masked sums and means must reduce global sums and global active-token counts:

```text
global_sum = all_reduce_sum(local_masked_sum)
global_count = all_reduce_sum(local_active_count)
masked_mean = global_sum / max(global_count, eps)
```

Averaging local means is not invariant when shards or micro-batches have
different active-token counts. The reference helpers are
`sharded_masked_sum(...)`, `sharded_active_token_count(...)`, and
`sharded_masked_mean(...)`.

The distributed equivalents are `distributed_masked_sum(...)`,
`distributed_active_token_count(...)`, and `distributed_masked_mean(...)`.
They use real all-reduce collectives and are covered by a Gloo multi-process
smoke test. NCCL multi-GPU coverage should be added in hardware CI when a
multi-GPU runner is available.

## Diagnostics

`summarize_tp_logprob_drift(...)` reports:

- max and mean absolute error;
- max and mean relative error;
- active-token count;
- flat and multi-index of the worst token;
- target token id;
- owning TP rank and vocab range;
- backend, reduction name, dtype, and TP size;
- candidate/reference values and signed error.

That is enough to tell whether a failure is likely from vocab logsumexp,
selected-token ownership, mask denominator semantics, or dtype behavior.

Future end-to-end rollout/training cross-benchmarks should reuse the same
summary fields so failures from vLLM/sglang rollout, FSDP scoring, and native
kernel tests can be compared without changing report schemas.

## Test Entry Points

Focused parity tests:

```bash
pytest tests/test_tp_invariant_reductions.py
```

Reference helper regressions:

```bash
pytest tests/test_reference_ops.py tests/test_tp_invariant_reductions.py
```

CUDA smoke coverage runs automatically when CUDA is available; otherwise it is
skipped without blocking CPU CI.
