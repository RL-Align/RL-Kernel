# DSV4 items 10/14: Indexer scale and ReLU head sum — MI355X validation

Validation date: 2026-11-01 (Asia/Shanghai); tests re-run 2026-11-08 after
adding per-head partials.

## Baseline

- Repository: RL-Align/RL-Kernel.
- Target baseline: `dsv4-rocm`, `dcda3909ba22211753d085d2aec964502cab87f7`.
- Development branch: `xh/dsv4-indexer-scale`.
- Inputs are item 16 / #599 raw ICV. #599 itself only admits gfx942, so these
  tests generate FP32 ICV directly instead of running #599 on MI355X.

## Actual implementation

```text
scores[m,n] = sum_h relu(ICV[m,h,n] * head_dim**-0.5) * (weights[m,h] * num_heads**-0.5)
```

|Step|Reused implementation|
|---|---|
|Scale, ReLU, weight scale|PyTorch FP32 elementwise|
|Fixed-order head sum (forward and `d_weights`)|aiter `mha_fused_bwd._bwd_preprocess`|

No new GPU kernel. RLK adds the typed contract, scale derivation from the model
config, the vLLM/Megatron contract adapters, the explicit VJP for #631, the
backward record guard and the implementation fingerprint.

## Acceptance

|Suite|Result|
|---|---:|
|CPU contract/oracle/guard tests (`test_indexer_score.py`)|23 passed|
|Actual-library MI355X tests (`test_indexer_score_rocm.py`)|16 passed|
|Total|39 passed, 0 failures/errors/skips|

Ruff check and format passed. GPU tests used one visible MI355X (GPU 0).
The single warning is PyTorch reporting the intentionally empty Graph in the
Graph-rejection test.

GPU tests include:

- H=64/D=128 forward against FP64 at (M,N) = (1,1), (7,129), (64,2048), (256,4096).
- Byte equality with the #631 aux-loss composition.
- Per-head partials against FP64 and batch invariant. Re-summing them matches
  `scores` within 1.2e-7 absolute, not byte for byte, because the head-sum
  kernel fuses each product into its accumulation.
- Bytes independent of batch composition, key count, non-contiguous ICV and
  repeated calls.
- Backward against FP64 and batch-invariant; zero gradient at ReLU zero.
- TP=2 shards summed against the unsharded result with the global head scale.
- Empty inputs; rejection of CPU tensors, Graph capture and a non-gfx950 arch.
- Two fresh processes with distinct Triton JIT caches: score, `d_icv` and
  `d_weights` SHA256 digests and implementation fingerprints match.

## Environment / provenance

- GPU: `AMD Instinct MI355X` (gfx950), container `vllm/vllm-openai-rocm:v0.21.0`.
- Torch: `2.12.1+rocm7.2`.
- HIP: `7.2.53211`.
- Triton: `3.7.0`.
- Loaded `aiter/ops/triton/_triton_kernels/attention/mha_fused_bwd.py` SHA256:
  `e9183920ca149205978c1aace8f07ff69a271c51104cb6de3d829e29632bc33e`.
- Backend adapter SHA256:
  `69f7e427bae58825707628ade3202f3caa5075777494217a5ab2ada8538396d3`.

## Compute microbenchmarks (microseconds)

Median of 5 synchronized wall-time loops, 30 calls/loop, including Python
dispatch/launch overhead. H=64, D=128.

|M|N|ICV bytes|Forward|Input check|Scale+ReLU+transpose|Head-sum kernel|Backward|PyTorch sum reference|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|512|131,072|144.6|76.4|25.6|18.2|117.7|28.9|
|32|2048|16,777,216|166.1|91.5|34.9|23.9|121.0|32.9|
|128|4096|134,217,728|578.6|196.2|197.1|177.6|270.3|147.6|
|512|4096|536,870,912|2391.5|598.5|890.9|896.5|887.1|652.3|

The PyTorch reference is `(relu(icv*s) * (w*c)[:,:,None]).sum(1)`. It is a
speed reference only; its head-sum order is chosen by the library and is not
batch invariant.

The operator is 3.7x slower than the reference at the largest shape. At small
shapes the finite-value check (one host readback per input) takes about half of
the forward time. At large
shapes the transpose copy and the head-sum kernel (one program per
`(query, key)` reducing only 64 values) each take about as long as the whole
reference. The design favours fixed reduction order over speed; a fused
scale+ReLU+sum kernel with the same order is the obvious next step and is not
part of this change.

## Reproduction

```bash
python -m pytest tests/models/deepseek_v4/test_indexer_score.py \
  tests/models/deepseek_v4/test_indexer_score_rocm.py -q
python benchmarks/benchmark_indexer_score.py --repeats 30 --output benchmark.json
```

Usage is documented in `docs/models/deepseek_v4/indexer-score.md`.

## Explicit gaps

- MI300X/gfx942 is not validated and is rejected.
- aiter gfx950 FP4 fused score (`pa_mqa_logits_fp4`) uses a different scale
  order and was not compared.
- Causal mask, invalid rows, Top-K, cross-TP merge, Graph, multi-GPU and the
  full rollout/training chain.
- Real #599 ICV on MI355X (blocked by #599's gfx942-only gate).
