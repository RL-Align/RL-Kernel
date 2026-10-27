# DSV4 item 16: independent ICV — MI300X validation

Validation date: 2026-10-26 (Asia/Shanghai).

## Baseline

- Repository: RL-Align/RL-Kernel.
- Target baseline: `dsv4-rocm`, `5ba769f681b7fa765ea6b29f9939454a858390ba`; rechecked during export.
- Development branch: `feat/dsv4-indexer-icv-bi`.
- Upstream main currently differs: `d8ef83b220b6c952035e0fa1d935b67412bafea4`. It is not silently substituted
  for the explicitly requested integration branch.
- This change is independent of item 12 / #579 and item 13 / #590.
- Item 16 was re-read from the RFC on 2026-10-26; its contract remains raw
  quantized Q/K correlation, fixed FP32 accumulation/output and global K layout.

## Actual implementation

Raw `ICV[M,H,N]`, with separate request outputs and global compressed-key IDs.
No global Indexer scales, ReLU, head weights, head sum, causal mask or Top-K.

|Profile|Reused implementation|
|---|---|
|MXFP4/group-32|aiter upcast_from_mxfp + gemm_a16w16 (FP32 output)|
|FP8/per-Q-head/per-K-row scale|aiter gemm_a8w8_per_token_scale (FP32 output)|
|OCP FP8 on gfx942|vLLM normalize_e4m3fn_to_e4m3fnuz, original input protected|

No new GPU numerical kernel, decoder, quantizer or reduction. RLK adds typed
contracts, metadata validation, version/mutation guards, global-K ordering,
prepared-key reuse, bounded outputs and shared vLLM/mLite forward adapters.

MXFP4 is explicitly emulated on MI300X using the **actual quantized E2M1 data**.
For nonzero groups E8M0 bytes [2,252] decode exactly into BF16 before FP32 GEMM;
all-zero groups may additionally use other finite scale bytes. Byte 255 is
rejected. This is not conversion of an unquantized model to BF16, nor a claim of
native gfx950 MX instructions. FP8 is a distinct profile and does not substitute
for MXFP4 acceptance.

The pinned gfx950 paged FP4 MQA operator fuses ReLU and weighted head reduction,
so it cannot expose this interface unchanged. The pinned mLite path uses
`use_fp4=False` and folds Q scales into head weights: the adapter requires
separately preserved scales and does not attempt to divide by fused weights.

## Acceptance

|Suite|Result|
|---|---:|
|CPU-safe contract/layout/state/adversarial tests|81 passed|
|Actual-library MI300X and process/transport stress tests|121 passed|
|Existing operator-contract regression|142 passed|
|Total|344 passed, 0 failures/errors/skips|

Ruff check and format checks passed, as did patch whitespace/application checks.
The entire 344-test suite passed twice: once with Python branch coverage
instrumentation and once without it, both with zero skips. All GPU tests used
one visible MI300X VF. Other users' jobs were left untouched.

Tests include:

- Real H=64/D=128 geometry; varied M/N, neighboring requests, permutations,
  padding, query chunking and output tile sizes.
- Odd/even D and group tails (including 1/31/32/33/63/65/127/129/256).
- Independent FP64 decoded-value ICV oracle; true FP32, signed correlations.
- All E2M1 codes and admitted extreme exponents, including signed zeros.
- Native aiter MXFP4 and vLLM FP8 quantizer outputs, including all-zero rows.
- Exactly-once quantization scale application.
- CP1/2/4 **materialized** global K order/ownership and missing-shard rejection.
- Shared vLLM/Megatron adapter outputs, layout/scale/version errors, mutation
  guards, empty inputs, output budgets and no hot-path host readback.
- Test-only later-stage composition against unmodified pinned aiter floating
  Indexer reference, without conflating its fused scores with ICV.
- Ten deterministic random seeds per precision profile, varying requests,
  shapes, CP layouts, physical permutations, padding and tile sizes.
- Long batches through M=257 at H=64, repeated/equal-valued keys retained as
  distinct columns, and wide-scale numerical forward-error bounds.
- Two CUDA streams sharing prepared K, 50-call live-allocation stability,
  explicit Graph rejection and fail-closed missing-library/wrong-device cases.
- Two fresh processes using distinct Triton JIT caches: input, scale and
  ICV SHA256 digests match for MXFP4, OCP FP8 and FNUZ FP8.

### Memory evidence

- Full raw ICV for the memory case: **67,108,864 bytes**.
- Streaming extra peak allocation: **4,718,592 bytes**.
- Emitted tiles: **32**; maximum output/tile:
  **2,097,152 bytes**.

The measured peak includes live temporary tensors tracked by PyTorch, not total
reserved allocator memory. The full output was refused under the small output
budget; streaming was verified without retaining all tiles.

### Python coverage

- Statements: **354/354
  (100.00%)**.
- Branches: **86/86
  (100.00%)**.

Coverage is scoped to the new contracts/adapters/backend wrappers. It does not
cover native aiter/vLLM GPU instructions and does not replace the exact-byte
and independent numerical tests. Detailed missing lines/branches are retained
in `indexer16-coverage.json` and `indexer16-coverage.txt`.

## CP and training scope

Framework-native transport must have materialized complete global K before this
boundary. In addition to simulated CP1/2/4 materializations, an actual **two-rank
CPU Gloo all-gather** test transports uneven quantized shards and metadata, then
compares the single-GPU consumer with complete unsharded input byte for byte.
The rank lengths are [9, 8].
This is **not RCCL/NCCL GPU transport**, distributed Megatron or a multi-node run.

This is quantized **forward**. Trainable data/scales are rejected rather than
silently detached. Quantization surrogate gradients and repeated-K reduction
are the separate item 26; no full Indexer backward is claimed.

## Environment / provenance

- GPU: `AMD Instinct MI300X VF` (gfx942), isolated container with only host GPU 5 visible.
- Torch: `2.12.0+rocm7.14.0a20260608`.
- HIP: `7.14.60850`.
- Triton: `3.7.0`.
- aiter reference commit: `326cffa38c9eda16bef19375ba55334e4d3acd59`.
- Megatron reference commit: `327a238239448de6771beea185dc6902a8de3eaa`.
- Actual loaded numerical-module hashes: `indexer16-benchmark.json`;
  the fixed recipe is recorded in the backend and integration document.
- Read-only source snapshot hashes: `indexer16-provenance.json`.

The implementation fingerprint includes the semantic contract, actual numerical
sources and software versions. Dependency changes require revalidation; a
matching function name is not taken as BI evidence.

## Compute microbenchmarks (microseconds)

Median of 5 synchronized wall-time loops, 30 calls/loop, including Python
dispatch/launch overhead. Input validation/preparation is separated below.

|Precision|M|N|Prepared upstream dot|Bounded stream|Materialized pipeline|Predecoded FP32 einsum reference|
|---|---:|---:|---:|---:|---:|---:|
|mxfp4|1|128|42.0|148.5|151.6|26.7|
|mxfp4|8|512|41.9|201.5|150.7|25.5|
|mxfp4|32|2048|41.5|527.8|159.2|27.3|
|fp8|1|128|47.0|109.5|112.8|25.8|
|fp8|8|512|48.3|174.1|114.1|27.4|
|fp8|32|2048|48.5|551.0|122.7|24.8|

The raw einsum reference uses already-decoded FP32 inputs and a different
arithmetic/backend contract. It is not a precision-/work-equivalent fused-score
comparison. These measurements do not support a universal speedup claim.
Python validation, metadata and streaming have measurable overhead.

For small admitted materializations, the code returns a full-key-span library
GEMM result directly instead of adding unnecessary key-tile launches/copies.
Large requests can stream tiles, limiting live output size rather than forcing
quadratic output allocation.

## Binding/preparation cost (steady-state microseconds)

|Precision|M|N|Query validation/bind|Global-K validation/prepare|
|---|---:|---:|---:|---:|
|mxfp4|1|128|167.5|335.5|
|mxfp4|8|512|169.2|490.7|
|mxfp4|32|2048|188.9|1378.2|
|fp8|1|128|232.8|311.2|
|fp8|8|512|246.7|471.8|
|fp8|32|2048|255.3|1235.0|

These stages include host checks/synchronization and must not be hidden in a
kernel-only latency claim. Reuse prepared keys only while their sources and
version remain valid. Raw JSON also records cold calls separately, which may
include first-use framework/kernel overhead.

## Reproduction

```bash
RLK_AITER_INDEXER_REFERENCE=/path/to/pinned/aiter/ops/triton/attention/dsv4_indexer.py \
  python -m pytest tests/contracts/operators tests/models/deepseek_v4 -q
python benchmarks/benchmark_indexer_icv.py \
  --repeats 30 --output .codex-work/benchmark.json
```

The pinned floating-reference file was supplied in the recorded run; there were
no skips. Full usage and supported tensor layouts are documented in
`docs/models/deepseek_v4/indexer-icv.md`.

## Explicit gaps

- Complete Flash-0731 checkpoint/preprocessing/framework chain and raw logprob.
- Native gfx950 MX hardware path or support for every possible E8M0 exponent.
- Actual multi-GPU CP transport, Graph and full Indexer backward.
- Automatic replacement of fused MQA logits or installation in a running model.

No unsupported configuration is silently labelled validated.
