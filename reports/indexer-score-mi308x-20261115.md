# Items 10/14 MI308X report, 2026-11-15

Base `dsv4-rocm` `b184c3fb`. Same environment as #678. No new GPU kernel.

|Component|Version|
|---|---|
|Image|`vllm/vllm-openai-rocm:v0.26.0` (`sha256:5709fafe…a97a18`)|
|Torch / HIP / Triton|`2.11.0+gitd0c8b1f` / `7.2.53211` / `3.6.0`|
|aiter / vLLM|`0.1.16.post3` / `0.26.0+rocm723`|
|GPU|MI308X `gfx942`, one card, `TRITON_F32_DEFAULT=ieee`|

Tests: 39 passed (23 CPU, 16 GPU), no skips. `tests/contracts/operators
tests/models/deepseek_v4`: 1583 passed, 14 skipped. Re-summed
`head_partials()` differs from `scores` by at most 1.2e-7.

`benchmark_indexer_score.py --repeats 30`, H=64, D=128, median µs:

|M|N|Forward|Check|Scale+ReLU+T|Head sum|Backward|Torch sum|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|512|227.5|120.9|41.1|33.5|179.2|44.4|
|32|2048|345.6|144.6|109.8|83.3|190.7|64.6|
|128|4096|1774.7|336.1|735.4|675.3|575.2|290.0|
|512|4096|7733.3|1097.0|3174.3|3486.9|2138.9|1273.9|

Torch sum is a speed reference only (not BI). At the largest shape the
operator is 6.1x slower; transpose and head sum dominate.

Not covered: gfx950, real #599 ICV, `pa_mqa_logits_fp4`, causal mask, Top-K,
cross-TP merge, Graph, multi-GPU, full chain.
