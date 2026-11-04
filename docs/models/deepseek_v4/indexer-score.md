# DSV4 items 10/14: Indexer scale and ReLU head sum

This is the boundary between raw Indexer correlations and Top-K:

```text
ICV[M,H,N] (FP32), weights[M,H] (FP32) -> scores[M,N] (FP32)
scores[m,n] = sum_h relu(ICV[m,h,n] * softmax_scale) * (weights[m,h] * head_scale)
```

`ICV` is item 16 output: raw per-head Q·K correlation with quantization
de-scales already applied. `weights` are the unscaled Indexer head-weight
projection outputs. The causal mask, invalid rows and Top-K are not part of
this boundary.

Code is based on `RL-Align/RL-Kernel:dsv4-rocm`, not KLR.

## Scale contract

|Scale|Value|Applied to|
|---|---|---|
|`softmax_scale`|`head_dim ** -0.5`|ICV, **before** ReLU|
|`head_scale`|`num_heads ** -0.5`|head weights|

Both are derived from `IndexerScoreContract(num_heads, head_dim)`. The contract
has no model defaults; adapters read `index_n_heads`/`index_head_dim` (vLLM HF
config) or `dsa_indexer_n_heads`/`dsa_indexer_head_dim` (Megatron
`TransformerConfig`), so both sides derive the same scales from one model config.

`head_scale` always uses the **global** head count. With `tp_size > 1` the
operator accepts `local_heads` but keeps `num_heads ** -0.5`; summing shard
outputs therefore matches the scale of the unsharded result. Neither vLLM
(`ReplicatedLinear`) nor Megatron (`parallel_mode="duplicated"`) currently
shards Indexer heads; the cross-shard merge order is not defined here.

### Why the order is fixed

Floating-point multiplication does not commute with rounding, so where
`softmax_scale` is applied changes result bytes. Rollout and training must use
one order to be bitwise consistent; the order itself does not affect batch
invariance. For DSV4 `head_scale = 64 ** -0.5 = 2 ** -3` is exact, so its
position does not change bytes, while `128 ** -0.5` is not a power of two.

The chosen order matches the RFC text (correlation, scale, ReLU, weight, head
sum) and the composition used by #631's Indexer aux-loss, so #631 can call this
operator without changing its bytes (tested).

vLLM's fused path instead folds `q_scale * softmax_scale * head_scale` into the
weights (`fused_indexer_q_rope_quant`) and applies ReLU before `softmax_scale`.
That is mathematically equal for positive scales but not byte-equal. Using this
operator in rollout means replacing that fold, and passing unscaled weights.

## Fixed arithmetic

- Scale and ReLU are PyTorch elementwise FP32 operations; each output element
  depends only on its own input.
- The head sum reuses aiter `mha_fused_bwd._bwd_preprocess` as a fixed
  per-row dot product (one program per `(query, key)`, one `tl.sum` over a
  power-of-two-padded head axis). Grid and launch parameters are fixed in
  `ROW_DOT_LAUNCH`; M, N and neighbouring rows do not select the reduction.
- No new GPU kernel is added and there is no fallback to another backend.
- Only gfx950 (MI355X) is validated; other architectures are rejected.
- The implementation fingerprint covers the contract, the fixed launch plan,
  the loaded aiter source hash, this adapter's source hash and Torch/HIP/Triton
  versions. Training may pass `expected_implementation` to reject a mismatch.

## Backward

`backward(record, grad_scores)` is an explicit VJP for #631's aux-loss, not
autograd (the whole operator runs under `torch.no_grad`):

```text
d_weights[m,h] = head_scale * sum_n relu[m,h,n] * grad[m,n]     (same row dot)
d_icv[m,h,n]   = (grad[m,n] * scaled_weights[m,h]) * softmax_scale  if relu > 0
                 0                                                  otherwise
```

The ReLU derivative at 0 is 0. The mask uses the stored ReLU output, so an ICV
value that underflows to zero after scaling also gets a zero gradient. The
record keeps `relu` and `scaled_weights`; it checks tensor versions, pointers
and the producing implementation before use.

## Integration

```python
from rl_engine.integrations.engines.rollout.vllm.indexer_score import contract_from_hf_config
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore

op = IndexerScore(contract_from_hf_config(hf_config))
record = op.forward(icv, weights)          # icv from item 16, weights unscaled
topk_input = record.scores                 # FP32 [M, N]; mask/Top-K downstream
```

Training uses
`rl_engine.integrations.engines.train.megatron.indexer_score.contract_from_transformer_config`
and the same `IndexerScore`. Neither adapter patches a running model.

Inputs must be FP32, finite, on the backend device and outside CUDA Graph
capture. The finite check reads back one boolean per input tensor; its cost is measured
separately in the benchmark. The backward head sum accepts at most 4096 keys
per call (`max_backward_keys`); the caller chunks wider key spans.

## Verification and benchmark

```bash
python -m pytest tests/models/deepseek_v4/test_indexer_score.py -q
python -m pytest tests/models/deepseek_v4/test_indexer_score_rocm.py -q
python benchmarks/benchmark_indexer_score.py --repeats 30 --output benchmark.json
```

GPU tests run the actual aiter kernel and check:

1. Forward and backward against an FP64 reference at H=64, D=128.
2. Byte equality with #631's composition.
3. Batch invariance: bytes do not change with batch composition, key count,
   non-contiguous input, repeated calls, or two fresh processes with distinct
   Triton JIT caches.
4. Zero gradient at ReLU zero and underflow, TP shard sums, empty inputs, and
   rejection of CPU tensors, Graph capture and unvalidated architectures.

## Not covered

- aiter `pa_mqa_logits_fp4` (gfx950 fused FP4 score) applies scales in a
  different order and was not evaluated against this contract.
- Per-head partial scores are not returned; only the summed score and the
  backward record.
- Causal mask, invalid query/key rows, Top-K and cross-TP-shard merge.
- CUDA Graph, multi-GPU runs and a full rollout/training chain.

## Pinned references

- [RFC item grouping](https://github.com/RL-Align/RL-Kernel/issues/511)
- [#599 item 16 ICV](https://github.com/RL-Align/RL-Kernel/pull/599)
- [#631 Indexer aux-loss](https://github.com/RL-Align/RL-Kernel/pull/631)
- [vLLM weight fold](https://github.com/vllm-project/vllm/blob/6a03f43c68dec77fb8ba6932a6d57c4ba9ff710c/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py)
- [Megatron DSA Indexer](https://github.com/NVIDIA/Megatron-LM/blob/2cff3b007f531a808e78f198e9786eae44f5463d/megatron/core/transformer/experimental_attention_variant/dsa.py)
- [aiter FP4 fused score](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py)
