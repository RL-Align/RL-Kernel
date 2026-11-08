# Indexer scale and ReLU head sum (items 10, 14)

This is the boundary between raw Indexer correlations and Top-K:

```text
ICV[M,H,N] (FP32), weights[M,H] (FP32) -> scores[M,N] (FP32)
scores[m,n] = sum_h relu(ICV[m,h,n] * softmax_scale) * (weights[m,h] * head_scale)
```

`ICV` is the item 16 output with quantization de-scales already applied.
`weights` are the unscaled head-weight projection outputs. The causal mask,
invalid rows and Top-K are not part of this boundary.

## Scale contract

|Scale|Value|Applied to|
|---|---|---|
|`softmax_scale`|`head_dim ** -0.5`|ICV, **before** ReLU|
|`head_scale`|`num_heads ** -0.5`|head weights|

Both come from `IndexerScoreContract(num_heads, head_dim)`, which has no model
defaults. The vLLM and Megatron adapters read `index_n_heads`/`index_head_dim`
and `dsa_indexer_n_heads`/`dsa_indexer_head_dim`. `head_scale` always uses the
**global** head count, so with `tp_size > 1` the shard outputs sum to the
unsharded scale. Neither framework currently shards Indexer heads.

The scale order changes bytes, not batch invariance; rollout and training must
use the same one. This order follows the RFC and #631's composition (tested
byte-equal). vLLM's fused path folds `softmax_scale * head_scale` into the
weights and applies ReLU first, which is equal in math but not in bytes; using
this operator in rollout means passing unscaled weights instead.

## Arithmetic

Scale and ReLU are PyTorch FP32 elementwise operations. The head sum reuses
aiter `mha_fused_bwd._bwd_preprocess` as a fixed per-row dot product, one
program per `(query, key)` with a fixed launch (`ROW_DOT_LAUNCH`); M, N and
neighbouring rows do not change the reduction. There is no new GPU kernel and
no fallback. Only gfx950 is validated. The implementation fingerprint covers the
contract, launch plan, aiter and adapter source hashes and Torch/HIP/Triton
versions; pass `expected_implementation` to reject a mismatch.

## Backward

`backward(record, grad_scores)` is an explicit VJP for #631, not autograd:

```text
d_weights[m,h] = head_scale * sum_n relu[m,h,n] * grad[m,n]     (same row dot)
d_icv[m,h,n]   = (grad[m,n] * scaled_weights[m,h]) * softmax_scale  if relu > 0, else 0
```

The ReLU derivative at 0 is 0, including values that underflow to 0 after
scaling. The record checks tensor versions, pointers and the producing
implementation before use.

## Intermediate outputs

|Stage|Shape|Source|
|---|---|---|
|Scaled ReLU ICV|`[M,H,N]`|`record.relu`|
|Scaled head weights|`[M,H]`|`record.scaled_weights`|
|Per-head partial scores|`[M,H,N]`|`record.head_partials()`|
|Summed score|`[M,N]`|`record.scores`|

`head_partials()` is computed on request and is batch invariant. The head-sum
kernel fuses each product into its accumulation, so re-summing the partials
matches `scores` only within rounding, not byte for byte.

## Integration

```python
from rl_engine.integrations.engines.rollout.vllm.indexer_score import contract_from_hf_config
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore

op = IndexerScore(contract_from_hf_config(hf_config))
record = op.forward(icv, weights)          # icv from item 16, weights unscaled
topk_input = record.scores                 # FP32 [M, N]; mask/Top-K downstream
```

Training uses `contract_from_transformer_config` from the Megatron adapter.
Inputs must be FP32, finite, on the backend device and outside CUDA Graph
capture. The backward accepts at most 4096 keys per call (`max_backward_keys`).

## Verification

```bash
python -m pytest tests/models/deepseek_v4/test_indexer_score.py \
  tests/models/deepseek_v4/test_indexer_score_rocm.py -q
python benchmarks/benchmark_indexer_score.py --repeats 30 --output benchmark.json
```

GPU tests check FP64 accuracy, byte equality with #631, batch, key-count,
layout, repeat and fresh-process invariance, TP shards and input rejection.
Results are in `reports/indexer-score-mi355x-20261101.md`.

Not covered: gfx942, aiter
[`pa_mqa_logits_fp4`](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py)
(different scale order, not evaluated), causal mask, Top-K, cross-TP merge,
CUDA Graph, multi-GPU and the full rollout/training chain.

References: [#599](https://github.com/RL-Align/RL-Kernel/pull/599),
[#631](https://github.com/RL-Align/RL-Kernel/pull/631),
[vLLM weight fold](https://github.com/vllm-project/vllm/blob/6a03f43c68dec77fb8ba6932a6d57c4ba9ff710c/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py),
[Megatron DSA Indexer](https://github.com/NVIDIA/Megatron-LM/blob/2cff3b007f531a808e78f198e9786eae44f5463d/megatron/core/transformer/experimental_attention_variant/dsa.py).
