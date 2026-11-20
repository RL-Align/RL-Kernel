# DSV4 item 16: independent quantized Indexer ICV

This is the raw per-head correlation boundary

```text
Q[M,H,D], K[N,D], quantization scales -> ICV[M,H,N] (FP32)
```

It does not compute Indexer scores, ReLU, head weights, head sums, global
Indexer scaling, causal masks or Top-K. Quantization **de-scales** belong to
this boundary; the distinct model-wide scales in item 10 do not.

Code is based on `RL-Align/RL-Kernel:dsv4-rocm`, not KLR. The RFC is used for
the task contract. Items 12/13 are separate unmerged changes and are not copied
into this implementation.

## Existing implementations and why they are not interchangeable

- The fixed aiter `dsv4_indexer.indexer_fwd` reference first computes a floating
  `einsum("shd,pd->shp")`, then applies ReLU, weights, head sum and a causal mask.
  Tests use this as a separate numerical reference after explicitly composing
  those later operations. It is not a strict BI production backend.
- aiter's `flydsl_pa_mqa_logits_fp4` is a **gfx950** paged FP4 fused-score kernel.
  Its ReLU and weighted head reduction lose independent per-head ICV. This
  interface is not relabelled as ICV or assumed usable on MI300X.
- The pinned mLite `official_indexer_topk` calls `fp8_fp4_mqa_logits`, but its
  Q preprocessing explicitly uses `use_fp4=False`. Furthermore, Q scales are
  fused into head weights. Those two outputs alone cannot reconstruct an
  independent Q-scale-aware ICV. The adapter requires separately retained
  scales, and rejects `None` rather than dividing by potentially zero weights.

The strict MI300X paths therefore reuse:

|Path|Actual existing numerical primitives|
|---|---|
|MXFP4 group-32|aiter `upcast_from_mxfp`, then `gemm_a16w16` with FP32 output|
|FP8 per Q-head/K-row scale|aiter `gemm_a8w8_per_token_scale` with FP32 output|
|OCP FP8 representation on gfx942|vLLM `normalize_e4m3fn_to_e4m3fnuz` on a protected copy|

No numerical GPU kernel, unpacking kernel, quantizer or reduction is added.
RLK provides typed contracts, validation, request/global-row metadata, explicit
representation selection, prepared K reuse, bounded output iteration and thin
vLLM/mLite forward adapters.

## Precision is part of the contract

### Actual MXFP4, not a BF16 replacement model

The default profile accepts **already-quantized packed E2M1 bytes** and E8M0
group-32 scales:

```text
P = ceil(D / 32) * 32
Q bytes:  [M, H, P/2] uint8
Q scales: [M, H, P/32] uint8
K bytes:  [N, P/2] uint8
K scales: [N, P/32] uint8
layout:   e2m1.low_nibble_first.row_major.e8m0_group32
```

Two consecutive values are stored low nibble then high nibble. Scale bytes
encode `2**(byte - 127)`. Extra dimensions beyond logical D are ignored after
library decode, including odd-D nibble and group tails. Swizzled/paged gfx950
layouts are not accepted under a row-major label; the framework must unpack
their addressing/layout first.

On MI300X this profile is explicitly **MXFP4 emulation**. E2M1 values times the
admitted powers of two are exactly representable in BF16, so library upcast is
lossless on those values before FP32-accumulating GEMM. It does not quantize
BF16 master values afresh or substitute an unquantized BF16 model.

For nonzero E2M1 groups, the admitted E8M0 byte range is **[2,252]**. All-zero
groups additionally accept bytes 0/1/253/254 because their decoded signed zeros
remain exact; byte 255 (NaN scale) is always rejected. This permits native
quantizer outputs for zero/padding rows without silently zeroing nonzero values.
The current library decoder's
exponent-zero special case, BF16 subnormal handling and overflow/clamping
outside this range are not claimed. Other encodings fail during binding.
Tests cover all 16 E2M1 codes, both signed zeros and selected extreme admitted
exponents against an independent CPU FP64 oracle.

### Separate FP8 reference/deployment profile

```text
Q:       [M,H,D] torch.float8_e4m3fn OR torch.float8_e4m3fnuz
Q scale: [M,H,1] FP32
K:       [N,D] same explicit FP8 encoding
K scale: [N,1] FP32
layout:  e4m3.row_major.fp32_per_row_scale
```

The scale is a finite positive power of two in `2**[-60,30]`; the range is
explicitly bounded in this profile. This is **per-row/per-head**, not group-32
MXFP4. A successful FP8 test is not recorded as MXFP4 acceptance.
The lower end includes the native quantizer's epsilon-derived scale for zero rows.

OCP input is converted with vLLM's existing bit/scale normalization before
gfx942 execution, preserving finite dequantized values. The original data
is not mutated. The kernel epilogue is the documented order
`dot_fp32 * (q_scale * k_scale)`; the scales are applied once. Callers must
not pass the fused Indexer head weights as Q scales.

## Fixed arithmetic and identities

- Output is independently observable **FP32 per head**, never BF16 then widened.
- gfx942 aiter GEMM recipe is fixed: BM32, BN64, K tiles 64, one K split,
  four warps, two stages, fixed instruction shape/kpack.
- The per-token FP8 wrapper lowers the K tile for short D from D alone.
  M, N, neighboring requests and rank arrival order do not select reduction
  arithmetic.
- Default geometry is H=64, D=128. The tested interface admits H<=128,
  1<=D<=256 with explicit tails.
- Implementations expose a fingerprint covering the semantic contract, fixed
  plan, actual library-source hashes and Torch/HIP/Triton versions. The second
  framework may bind `expected_implementation` to reject a mismatch.
- Backend/device/layout/scale/version errors do not silently select another
  precision or a `torch.einsum` fallback.

This is a quantized forward operator. Trainable input/scale tensors are rejected
instead of detached silently. The separate item 26 owns quantization surrogate
gradients, repeated-K gradient reduction and full Indexer autograd.

## Global K and request isolation

The framework owns native CP communication and paged-cache addressing. This
boundary consumes complete, materialized Q/K tensors:

- Query IDs: int64 `[M,2]`, `(request_id, request_global_token_position)`.
- Key IDs: int64 `[N,2]`, `(request_id, request_global_compressed_row)`.
- Padding ID: exactly `(-1,-1)`, ignored by numerical output.
- Scheduler supplies `request_key_counts`: **global**, not rank-local counts.
- CP>1 materializations additionally supply `key_owners[N]`, with valid ranks.

Binding rejects duplicates, missing global key rows, local-only K presented as
global K, incorrect owners, unknown requests, or mismatched versions. Every
request must contain the unique global rows `0..count-1`. Prepared K is ordered
by logical row, never by physical address or rank arrival.

Each output block belongs to one request; there is no cross-request dense
matrix with ambiguous masked values. Query order is retained and accompanied
by logical positions and original packed-row IDs. Masking by query position
and compressed-row causal visibility remains a downstream responsibility.

CP tests exercise CP1/2/4 **materialized ownership/layouts on one GPU**,
including non-contiguous/reversed rank order and missing-shard rejection.
A separate test launches two actual CPU Gloo ranks, gathers uneven quantized
shards and metadata, then verifies the GPU ICV consumer against unsharded input.
This is not NCCL/RCCL GPU communication, a distributed Megatron run or multi-node
acceptance.

## Integration

```python
from rl_engine.contracts.operators.indexer_icv import IndexerICVContract
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV
from rl_engine.integrations.engines.rollout.vllm.indexer_icv import (
    bind_icv_queries, bind_global_icv_keys,
)

contract = IndexerICVContract(precision="mxfp4", num_heads=64, head_dim=128)
op = IndexerICV(contract)

# Inputs are produced by item 15 / item 22 and the native framework transport.
queries = bind_icv_queries(
    op, q_packed, q_scales, query_ids,
    layout=contract.layout, version=model_preprocessing_and_request_kv_epoch,
)
keys = bind_global_icv_keys(
    op, k_packed, k_scales, global_key_ids,
    layout=contract.layout, version=model_preprocessing_and_request_kv_epoch,
    request_key_counts=global_compressed_row_counts,
    key_owners=native_cp_owners, cp_size=native_cp_size,
)
for tile in op.iter_tiles(queries, keys):
    # tile.values: FP32 [query_rows, H, key_rows]
    # tile.key_start/key_end: request-global compressed-row interval
    # tile.query_positions / query_physical_rows: explicit row identities
    consume_raw_icv(tile)
```

The same functions are exposed from
`rl_engine.integrations.engines.train.megatron.indexer_icv`; this is the
training-**forward** boundary, not an automatic full-model patch. Preserve Q
scales before native fusion if using the FP8 mLite preprocessing path.

Binding checks values and copies metadata on the host; its synchronization cost
is measured separately. It belongs outside the GEMM hot loop but Q binding
is still needed for new inputs. Prepared K can be reused only while its source,
scales and model/preprocessing version are unchanged.
The opaque version string must also identify the request-generation/KV snapshot
epoch: reusing a request ID for a different request must not reuse old prepared K
under the same version. The caller owns this barrier; no cache allocator or
request lifecycle manager is invented inside item 16.

Binding exposes two explicit ownership policies (including both framework
adapters). `ownership="borrowed"` is the zero-copy default: the caller must
prevent **all** writes through **all** aliases until consumers finish. Version
stamps catch ordinary PyTorch mutations, not `.data`, NumPy/from-blob aliases
or fresh tensors sharing storage with independent counters. They are diagnostic
guards, **not proof of immutable storage**.

Use `ownership="snapshot"` when the original storage can be reused or mutated.
It copies data and scales **before** content validation, isolating both from
untracked caller aliases. Inference tensors automatically take this policy.
Producer writes must be ordered before binding/copying; concurrent source writes
during the copy are unsupported. Bound operand tensors, selectors and prepared
workspaces themselves remain read-only by contract, including through aliases.
The snapshots add source-sized allocations and a copy at binding; they do not
change the numerical recipe. Large K buffers are not unconditionally copied in
borrowed mode. Prepared arrays and selectors retain ordinary mutation guards.
The compute/iteration path has no `.cpu()`, `.item()`, `.tolist()` or tensor-to-bool
readback.

Provider provenance fails closed with `IndexerICVError` when a module has no
readable source (`__file__` absent/None, namespace packages, missing files);
package names or versions alone are not substituted for source hashes.

## Bounded output, not a full quadratic allocation by default

Independent ICV can be much larger than fused scores. `iter_tiles` emits bounded
query/key tiles and does not retain earlier tiles. The caller owns tile lifetime.
`materialize` is an explicit small-request/debug convenience and refuses an
output larger than `max_output_bytes` **before** running GEMM.
For an admitted materialization it uses the full key span per query tile; a
request fitting one query tile returns the upstream GEMM output directly, without
extra key-tile launches or output copies. This changes output partitioning only,
not K/D accumulation, and is byte-compared against bounded streaming.

`max_output_bytes` bounds logical output or a single emitted tile, not total
allocator memory including inputs, retained caller tiles and library scratch.
`max_prepared_key_bytes` separately bounds the valid prepared K representation.
Metadata, original quantized sources and temporary layout copies are additional.

## Verification and benchmark

```bash
python -m pytest tests/models/deepseek_v4/test_indexer_icv.py -q
RLK_AITER_INDEXER_REFERENCE=/path/to/pinned/dsv4_indexer.py \
  python -m pytest tests/models/deepseek_v4/test_indexer_icv_rocm.py -q
python benchmarks/benchmark_indexer_icv.py \
  --repeats 30 --output .codex-work/benchmark.json
```

GPU tests call the actual aiter/vLLM functions. They separately check:

1. Bitwise BI across M/N, query chunks, request permutation, padding, CP
   materializations and tile sizes.
2. Numerical raw ICV versus independently decoded FP64 matrix products.
3. Decode bytes/values for E2M1 and E8M0 boundary cases.
4. Explicit test-only postprocessing versus the pinned aiter floating reference.

Additional adversarial/stress coverage includes:

- Ten deterministic random seeds for each of MXFP4/OCP-FP8/FNUZ-FP8, with ragged
  multi-request batches, padding, CP ownership, permutations and tile choices.
- Full H=64 geometry through M=257; head-count/D boundaries; equal-valued keys
  retained as distinct logical columns.
- Wide scale spectra checked against a sum-of-absolute-products FP32 forward
  error bound, separate from exact BI comparisons.
- Native quantized zero rows, NaN payload bits, scale limits, malformed strides,
  dtype/rank/layout errors and missing/duplicate global rows.
- Weight/scale/KV-epoch mismatch and input/prepared-workspace mutation guards.
- Two-stream reuse, 50-call live-allocation stability, and bounded-stream peak
  memory below the full raw-ICV allocation.
- Fresh processes with distinct Triton JIT caches, plus the actual two-rank
  Gloo fixture described above.
- Explicit Graph rejection and fail-closed missing-library/unsupported-device
  paths, rather than claiming these unsupported configurations worked.

Python statement/branch coverage is collected for the new contract, adapters
and backend wrappers. It does not measure aiter/vLLM compiled-kernel instruction
coverage and is not a substitute for the numerical/BI tests.

The benchmark reports binding/preparation, prepared-dot, streaming and
materialization separately. Its predecoded FP32 `torch.einsum` reference has
a different arithmetic/backend contract. Fused MQA logits emit different
outputs and are not a valid latency comparison. No model-throughput or universal
speedup claim is made.

Not validated: full Flash-0731 checkpoint, all E8M0 exponents, gfx950 native
MX instructions, full framework/preprocessing chain, actual multi-GPU CP
transport, full Indexer backward, Graph, and end-to-end raw logprob.

## Pinned references

- [RFC item grouping](https://github.com/RL-Align/RL-Kernel/issues/511)
- [aiter floating Indexer](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/attention/dsv4_indexer.py)
- [gfx950 FP4 fused-score source](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py)
- [mLite actual FP8 preprocessing/runtime](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/attention/runtime.py)
- [Shared aiter GEMM reference](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/gemm/basic/gemm_a16w16.py)
