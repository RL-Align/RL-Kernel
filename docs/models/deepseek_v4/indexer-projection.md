# DSV4 item 13: contract-bound BI Indexer projections

This is the **Indexer Q and head-weight projection boundary**, not Indexer
scoring, RoPE, Top-K, or a complete DSV4 model implementation.
It is implemented independently of item 12, against the `dsv4-rocm` branch.

## Reuse instead of new kernels

|Work|Existing implementation|
|---|---|
|BF16 GEMM and FP32 output|aiter `gemm_a16w16`|
|Block-FP8 Q GEMM|aiter `gemm_a8w8_blockscale`|
|Per-token group-128 UE8M0 activation quantization|vLLM `per_token_group_quant_fp8`|
|OCP checkpoint to gfx942 FNUZ representation|vLLM `normalize_e4m3fn_to_e4m3fnuz`|
|Canonical parameter-gradient aggregation|RLK `CanonicalBackwardSession`|
|Master-weight layout and VJP cast rules|Megatron mLite `native_linear_vjp` contract|

No Triton, CUDA, HIP, quantization or reduction kernel is added. The new
autograd function is **plumbing**: its forward, input-gradient and weight-gradient
matrix products call the library GEMMs. It does not implement a matrix reduction.

The native Megatron TE/`torch.mm` VJP is an accuracy/semantics reference, not
assumed to be batch invariant. The strict path uses existing aiter arithmetic and
RLK's canonical aggregator instead. The vLLM BI linear is also an independent
reference; a different reduction tree is not declared byte-identical by name.

## Exact boundary

```text
qr [M, q_lora_rank] -> Q projection -> raw Q [M, index_heads, index_head_dim]
hidden [M, hidden_size] -> weights projection -> raw weights [M, index_heads]
```

The Flash-0731 configuration supplies `hidden_size=4096`, `q_lora_rank=1024`,
`index_n_heads=64`, and `index_head_dim=128`. Tests use these geometries with
synthetic inputs/weights, not the full model checkpoint.

This boundary applies **no** RoPE, RMSNorm, Indexer scales, ReLU or head sum.
The caller supplies the already-produced QR input and the correctly aligned
hidden-state rows. Items 10/11/14 own the adjacent operations.

The combined callable is convenient for testing. Separate `bound.q(qr)` and
`bound.weights(hidden)` hooks allow the original framework to retain its native
stream/overlap schedule. RLK neither creates replacement streams nor inserts
hidden device synchronization.

## Frozen arithmetic

`IndexerProjectionContract` fixes dimensions, precision, output casts,
backend/VJP identities and supported parallelism:

- Inputs and trainable masters are BF16.
- Q precision is explicitly `bf16` or `block_fp8_ue8m0`; never inferred by M.
- Q and weights outputs independently choose BF16 or **true FP32 GEMM output**.
  FP32 is not implemented by widening a rounded BF16 result.
- BF16 GEMMs use fixed `BLOCK_K=64`.
- The `4096 -> 64` weights projection uses eight fixed 512-wide K partitions.
  aiter's fixed FP32 partial-reduction tree is reused.
- BF16 Q uses one ascending-K pass. FP8 Q uses group/block K=128 and one pass.
- MFMA instruction shape, kpack, tile geometry, split counts and grouping are
  pinned. The provider performs no actual-M tuning-table lookup.
- BF16 gradients use fixed library plans. Canonical dW sees the same ordered
  logical rows irrespective of how those rows were physically partitioned.
- Initial signoff is gfx942, TP1/CP1, eager operation. Other architectures,
  collectives and Graph capture are explicit gaps.

vLLM #59952 motivates M-independent K grouping and the skinny-N case. This
implementation does **not** copy its CUDA-specific launch/reduction code or
claim that its arithmetic tree equals aiter's. Both RLK framework adapters bind
the *same* provider/contract.

The operator exposes both a semantic contract fingerprint and an implementation
fingerprint. The latter includes actual library source hashes, Torch/HIP/Triton
versions, the adapter and canonical-backward source, and frozen plans. Pass
`expected_implementation` when binding the other framework/node to reject a
different implementation before numerical dispatch.

## BF16 and FP8 usage

```python
from rl_engine.contracts.operators.indexer_projection import IndexerProjectionContract
from rl_engine.backends.rocm.gemm.indexer_projection import AiterIndexerProjectionBackend
from rl_engine.integrations.engines.rollout.vllm.indexer_projection import bind_indexer_projection
from rl_engine.models.deepseek_v4.indexer_projection import PreparedIndexerFP8Q

backend = AiterIndexerProjectionBackend()
contract = IndexerProjectionContract(
    q_precision="block_fp8_ue8m0",
    q_output="bf16",
    weights_output="fp32",
)

# Existing native checkpoint loader supplies unshuffled [N,K] FP8 weights and
# decoded FP32 [N/128,K/128] power-of-two scales. Do not reinterpret packed bytes.
packed = PreparedIndexerFP8Q.from_checkpoint(
    indexer.wq_b.weight, decoded_scales,
    version=weight_and_scale_revision, backend=backend,
)
bound = bind_indexer_projection(
    indexer, contract=contract, layer_id="layer0",
    backend=backend, packed_q=packed,
    expected_implementation=training_implementation_fingerprint,
)

# Keep the framework's existing scheduling; these may be on its separate streams.
raw_weights = bound.weights(hidden_states)
raw_q = bound.q(qr)
```

Serving does not need a duplicate BF16 Q weight. For training, pass the explicit
BF16 master to `from_checkpoint(..., master=master_weight)`; the framework
adapter verifies it is the actual model parameter. Packed data/scales themselves
must be frozen. Weight packing/repacking remains the native loader/weight-sync
owner's responsibility; this adapter does not create a new checkpoint quantizer.

The native normalizer may modify OCP negative-zero bits, so preparation clones
checkpoint storage before invoking it. It also adjusts the scale to preserve
finite dequantized values. Tests compare this behavior against the native
normalizer and check that the caller's original storage is unchanged.

Data, scales, source tensors and the master are stamped. Tracked mutations and
version mismatches reject stale packed weights. Inference tensors without version
counters require the caller's explicit weight/scale revision barrier; no adapter
can infer an undeclared in-place update in such storage.

No BF16 fallback is selected for FP8. Plain-BF16 contracts reject FP8 tensor
dtypes and recognized deployment-scale metadata instead of silently ignoring it.
The supported activation domain is finite BF16 values; this is not a NaN-scrubbing
operator.

## Training and the precise BI claim

```python
from rl_engine.models.deepseek_v4.indexer_projection import IndexerLogicalRows
from rl_engine.integrations.engines.train.megatron.indexer_projection import (
    bind_indexer_projection,
    canonical_backward_session,
)

rows = IndexerLogicalRows.bind(logical_keys)  # int64 [M,2] or [M,3], once per batch
train_bound = bind_indexer_projection(
    indexer, contract=contract, layer_id="layer0", backend=backend, packed_q=packed,
)
with canonical_backward_session() as session:
    result = train_bound(qr, hidden_states, logical_rows=rows)
    loss = downstream_loss(result.q, result.weights)
    loss.backward()
    session.validate_complete()
```

Logical keys identify a request/token (with an optional extra ordering column);
negative first-column IDs are padding. Binding snapshots and validates keys once,
not inside GEMM. Duplicate active identities and reuse of a parameter ID for
another weight are rejected.

For multiple microbatches, register their forward uses in the **same session
before backward starts**, then backpropagate the combined graph. The existing
canonical session collects their rows, removes padding, sorts by logical key,
and performs **one** library dW reduction for that parameter. RLK does not sum
already-rounded per-microbatch dWs in arrival order.

The native mLite BF16-master VJP rule is preserved: cast dY to BF16 before dX/dW
GEMMs, return BF16 master/input gradients. Thus FP32 forward output does not claim
an all-FP32 VJP. FP8 forward uses this declared master-weight surrogate, not a
derivative of the discrete quantizer. Input-only gradients are supported with
frozen weights without requiring a canonical parameter session.

BI acceptance checks:

1. A fixed logical row's raw Q and weights stay byte-identical across M,
   physical row positions, neighbors, padding and permutations.
2. A row's dQR/dHidden stays byte-identical under the same conditions.
3. dW is byte-identical for the **same effective logical token set and upstream
   gradients**, even when its microbatch partition/order changes.

Adding different nonzero contributions naturally changes dW and is not a BI
failure. Separate optimizer steps, independent backward graphs, interleaved
pipeline schedules that register future forwards after backward starts, DP
all-reduces and global gradient accumulation are not claimed by this profile.

## Verification

```bash
python -m pytest tests/models/deepseek_v4/test_indexer_projection.py -q
RLK_MEGATRON_DENSE_SOURCE=/path/to/pinned/primitive/dense.py \
  python -m pytest tests/models/deepseek_v4/test_indexer_projection_rocm.py -q
python benchmarks/benchmark_indexer_projection.py \
  --repeats 50 --output .codex-work/indexer-benchmark.json
```

The source-reference test executes the unmodified `native_linear_vjp` helper on
CPU, avoiding its unvalidated TE GPU branch. It checks native cast/gradient
semantics and numerical accuracy; it is not a full Megatron training run.
Real-library GPU tests separately verify both adapters and the strict BI VJP.

The microbenchmark labels BF16-vs-BF16 native comparisons separately from the
FP32 and block-FP8 paths. Different arithmetic contracts are not advertised as
interchangeable, and primitive timings are not model throughput. On the
validated container the run produced:

|M|BF16 Q aiter (us)|BF16 Q vLLM BI (us)|BF16 weights aiter (us)|BF16 weights vLLM BI (us)|FP32 weights aiter (us)|BF16 chain (us)|block-FP8 chain (us)|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|40.3|32.6|65.7|66.5|66.2|137.1|184.5|
|8|39.6|32.7|66.3|66.2|65.2|137.3|183.8|
|128|40.0|33.1|65.8|66.8|65.0|151.9|190.4|
|512|84.6|37.8|67.5|66.9|67.2|141.5|188.6|

Times are medians of five synchronized wall-time loops with 50 calls per loop.
Python dispatch is included. The fixed plans enforce the numerical contract;
these measurements are not evidence of a speedup. The selected BF16 Q recipe is
slower than this installed vLLM BI reference, especially at M=512. Narrow
weights-projection latency is roughly comparable. Further recipe optimization
requires renewed BI acceptance, rather than silently changing schedules.

Not included: model checkpoint execution, Indexer scoring/Top-K, RoPE, distributed
projection collectives, full attention backward, Graph, and end-to-end logprob
or optimizer-trajectory acceptance.

## References

- [Item grouping / Indexer projection](https://github.com/RL-Align/RL-Kernel/issues/511)
- [vLLM #59952](https://github.com/vllm-project/vllm/pull/59952)
- [Pinned vLLM BI matrix reference](https://github.com/inaniloquentee/vllm/blob/76aacf36da38ec341a4cae843a9ba7a497f837d4/vllm/model_executor/determinism/batch_invariant.py)
- [Pinned mLite dense/VJP reference](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/dense.py)
- [Pinned mLite Indexer call sites](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/attention/module.py)
- [aiter shared GEMM reference](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/gemm/basic/gemm_a16w16.py)
- [Flash-0731 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)

Verified checkpoint-config revision:
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`; dimensions were read from this
revision rather than assumed from a different model variant.
