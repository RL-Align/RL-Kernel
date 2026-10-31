# Item 13: Indexer projection BI — MI300X validation

Validation date: 2026-10-25 (Asia/Shanghai).

## Baseline and delivery scope

- Repository: RL-Align/RL-Kernel, not KLR.
- Target baseline: `dsv4-rocm`, `5ba769f681b7fa765ea6b29f9939454a858390ba`; rechecked during export.
- Development branch: `feat/dsv4-indexer-projection-bi`.
- Independent of item 12 / PR #579; no item-12 commits are included.
- This is an explicit opt-in projection/gradient boundary, not a complete DSV4
  model hook installation. The existing stream schedule can use its separate
  Q and head-weight projection methods.

## Reused arithmetic

|Operation|Reused implementation|
|---|---|
|BF16 Q and head-weight projection, true FP32 output|aiter gemm_a16w16|
|Block-FP8 Q projection|aiter gemm_a8w8_blockscale|
|UE8M0 group-128 activation quantization|vLLM per_token_group_quant_fp8|
|OCP-to-FNUZ checkpoint normalization|vLLM normalize_e4m3fn_to_e4m3fnuz|
|Canonical dW aggregation|RLK CanonicalBackwardSession|
|Master layout and dY cast semantics|Megatron mLite native_linear_vjp reference|

No new GPU numerical kernel, quantizer, allocator or reduction implementation.
The new autograd bridge calls library GEMMs for forward/dX/dW and delegates
cross-microbatch ordering to the existing canonical session.
Native TE/torch.mm is not silently labelled BI: it is a semantics/accuracy
reference, while the strict path binds fixed aiter plans.

## Verified geometry

Flash-0731 config revision:
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

- Q: `[M,1024] x [8192,1024]^T -> [M,64,128]`.
- Weights: `[M,4096] x [64,4096]^T -> [M,64]`.
- Synthetic data/checkpoint-format fixtures were used; full checkpoint execution
  is not part of this report.

Q supports explicitly selected BF16 and real block-FP8 paths. Outputs can be
BF16 or true FP32. FP8 checkpoint and master/scale versions are validated;
there is no BF16 substitution for a requested FP8 deployment path.

## Acceptance

|Suite|Result|
|---|---:|
|New CPU-safe contract/gradient-boundary tests|44 passed|
|Real-library MI300X GPU tests|20 passed|
|Existing operator-contract regression|142 passed|
|Total|206 passed, 0 failures/errors/skips|

Ruff check/format and patch whitespace/applicability validation also passed.

Byte checks cover raw Q, head weights, input gradients, and parameter gradients
for the **same effective logical token set and upstream gradients** under
batch-size changes, reordering, padding and microbatch splitting. Tests also
cover empty inputs, ragged K/N, FP32 cast placement, source-scale/master
mutation, prepared-weight generation, shared adapter behavior, native streams,
and no tensor readback in the forward hot path.

The Megatron comparison executes the unmodified `native_linear_vjp` source
function on CPU to avoid its unvalidated TE GPU branch. Strict GPU forward/VJP
and both adapters are tested separately. This is not full Megatron execution.

## Environment / implementation identity

- GPU: AMD Instinct MI300X VF; one MI300X visible (gfx942).
- Torch: `2.12.0+rocm7.14.0a20260608`.
- HIP: `7.14.60850`.
- Triton: `3.7.0`.
- Pinned Megatron reference: `327a238239448de6771beea185dc6902a8de3eaa`.
- Actual source hashes and frozen launch plans: `indexer13-benchmark.json`.
- Contract and implementation fingerprints can be checked by the second adapter
  before execution. Source paths are not included in those identity hashes.

The package may emit a generic RLK `_C`-unavailable import warning. This
implementation calls the explicit aiter/vLLM functions, not a registry fallback.

## Microbenchmark (microseconds)

Median of five synchronized wall-time loops, 50 calls per loop, including Python
launch overhead. These are kernel-boundary measurements, not model throughput.

|M|BF16 Q aiter|BF16 Q vLLM BI|FP32 weights aiter|BF16 Q + FP32 weights chain|FP8 Q + FP32 weights chain|
|---:|---:|---:|---:|---:|---:|
|1|40.3|32.6|66.2|137.1|184.5|
|8|39.6|32.7|65.2|137.3|183.8|
|128|40.0|33.1|65.0|151.9|190.4|
|512|84.6|37.8|67.2|141.5|188.6|

**No speedup is claimed.** The selected fixed BF16 Q recipe is slower than the
installed vLLM BI reference, notably at M=512. The narrow BF16 weights projection
is approximately comparable to that reference. FP32 and FP8 results are not
presented as precision-equivalent BF16 comparisons. Optimizing recipes remains
possible, but each candidate must retain its fixed arithmetic and rerun BI tests.

## Reproduce

```bash
RLK_MEGATRON_DENSE_SOURCE=/path/to/pinned/primitive/dense.py \
  python -m pytest tests/contracts/operators tests/models/deepseek_v4 -q
python benchmarks/benchmark_indexer_projection.py \
  --repeats 50 --output .codex-work/benchmark.json
```

The full integration contract, tensor ABI and source references are in
`docs/models/deepseek_v4/indexer-projection.md`. The native-source environment
variable is needed for the Megatron reference case; the recorded acceptance run
provided it and had no skips.

## Not claimed

- Complete Flash-0731 train/rollout logprob or optimizer-trajectory equivalence.
- Automatic installation into a running framework's full DSV4 model.
- RoPE, Indexer scoring/scales/Top-K or full attention backward.
- TP/CP > 1, interleaved pipeline backward, cross-node gradient reduction, Graph.
- Gradient equality when adding different nonzero token contributions.

Trainable dW requires a shared canonical session: register the forward uses
before starting backward, then validate completion. Native interleaved
Megatron pipeline scheduling needs separate integration work.
