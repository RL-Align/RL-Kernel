# Items 10/14 MI355X operator report, 2026-11-01

Base `dsv4-rocm`: `dcda3909ba22211753d085d2aec964502cab87f7`; branch
`xh/dsv4-indexer-scale`. Tests re-run 2026-11-08 after adding per-head partials.
GPU: MI355X/gfx950, one visible card, container `vllm/vllm-openai-rocm:v0.21.0`.
Torch `2.12.1+rocm7.2`, HIP `7.2.53211`, Triton `3.7.0`. Numerical provider:
aiter `mha_fused_bwd.py` SHA256 `e9183920ca149205978c1aace8f07ff69a271c51104cb6de3d829e29632bc33e`.
No new GPU kernel.

Validation: 39 tests passed (23 CPU, 16 MI355X), no skips; Ruff passed. The
single warning is PyTorch reporting the intentionally empty Graph in the
Graph-rejection test. Inputs are generated FP32 ICV, not #599 output, because
#599 admits only gfx942. Re-summing `head_partials()` matches `scores` within
1.2e-7 absolute, not byte for byte, because the head-sum kernel fuses each
product into its accumulation.

`benchmark_indexer_score.py --repeats 30`, H=64, D=128, median wall
microseconds including Python launch overhead:

|M|N|Forward|Input check|Scale+ReLU+transpose|Head-sum kernel|Backward|PyTorch sum|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|512|144.6|76.4|25.6|18.2|117.7|28.9|
|32|2048|166.1|91.5|34.9|23.9|121.0|32.9|
|128|4096|578.6|196.2|197.1|177.6|270.3|147.6|
|512|4096|2391.5|598.5|890.9|896.5|887.1|652.3|

"PyTorch sum" is `(relu(icv*s) * (w*c)[:,:,None]).sum(1)`, a speed reference
whose reduction order is not batch invariant. The operator is 3.7x slower at
the largest shape: at small shapes the finite checks take about half the
forward; at large shapes the transpose copy and the head-sum kernel each cost
about as much as the reference. A fused kernel with the same order is future work.

Outstanding: gfx942, aiter `pa_mqa_logits_fp4`, real #599 ICV on MI355X, causal
mask, Top-K, cross-TP merge, Graph, multi-GPU and the full rollout/training chain.
