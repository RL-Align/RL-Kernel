# MiniMax-H3 Projection Rows

## Summary

Four RFC #420 rows share one contract: `audio_input_projection_fp32`,
`audio_output_projection_fp32`, `text_context_projection` and `h3_qkv_gemm`. Each is a linear
projection pinned to checkpoint tensors (`MiniMaxAI/MiniMax-H3@42ed227`, see
the architecture fingerprint (`h3-arch-fingerprint`, #567)):

| Row | in -> out | dtype | bias | tensors |
| --- | --- | --- | --- | --- |
| `audio_input_projection_fp32` | 32 -> 5376 | float32 | yes | `audio_proj_in` |
| `audio_output_projection_fp32` | 5376 -> 32 | float32 | yes | `audio_proj_out` |
| `text_context_projection` | 5120 -> 5376 | bfloat16 | yes | `context_embedder` |
| `h3_qkv_gemm` (per projection) | 5376 -> 7168 | bfloat16 | no | `attn.to_q/to_k/to_v` |

`audio_*` modules are FP32 modules: a lower-precision argument is a contract violation, not
a variant. `h3_qkv_gemm` also covers the fused serving layout (`[Q; K; V]` rows concatenated,
as in vLLM-Omni's `QKVParallelLinear`) and tensor-parallel head shards (any multiple of
128 output columns, so TP2/4/8 over 56 heads).

## Entry Points

```python
from rl_engine.runtime.registry import KernelRegistry

registry = KernelRegistry()
op = registry.get_op("h3_qkv_gemm", device="cuda")
q, k, v = op.forward(x, wq, wk, wv)            # Diffusers layout
fused = op.forward_fused(x, torch.cat([wq, wk, wv]))   # serving layout, == [q | k | v] bitwise
y = registry.get_op("text_context_projection", device="cuda")(x, weight, bias)
```

## Backends

| Backend | Class | Kernel | Status |
| --- | --- | --- | --- |
| CUDA (Triton) | `rl_engine.backends.cuda.model_specific.minimax_h3.projection.H3*CudaOp` | `rl_engine.backends.shared.triton.gemm.h3_projection._projection_kernel` | strict |
| PyTorch reference | `rl_engine.reference.minimax_h3.projection.NativeH3*Op` | n/a | `forward`: provider `F.linear`; `forward_fp32`: FP64 golden |
| ROCm | n/a | n/a | falls back to the PyTorch reference |

## Reduction Contract `h3-projection-gemm-v1`

* one program per output tile; the reduction axis is walked in ascending `BLOCK_K` steps into an
  FP32 accumulator; no split-K, stream-K, atomics, autotuning or M/N-dependent configuration;
* fixed tile configurations (part of the contract): BF16 `128x128x64`, 8 warps, 3 stages;
  FP32 `64x64x32`, 4 warps, 3 stages;
* FP32 rows use `input_precision="ieee"` (no TF32); BF16 rows use the tensor-core dot. The
  bias is added once in FP32 and the result rounded once to the output dtype;
* `d_input = g @ W` reduces the output axis ascending; `d_weight = g^T x` and `d_bias`
  reduce the token axis (the one cross-token reduction); results are cast once. Every op takes
  an optional `logical_keys` (`[T, 2|3]` int64, a negative first component marks padding):
  rows are then reduced in ascending key order with padding dropped, so `d_weight` / `d_bias`
  are bitwise independent of how the caller permutes, pads or splits the rows. Active keys must
  be unique. Without keys the reduction follows the call order and is only invariant for an
  unchanged row order; the backward provenance records which of the two was used. The key
  format follows #562;
* `h3_qkv_gemm` backward is the fused `[Q; K; V]` backward: `N = 7168` is a multiple of
  `BLOCK_K`, so concatenating the three operands moves no block boundary and the separate and
  fused layouts have identical gradient bytes.

Consequences, all tested bitwise: a token's output does not depend on the batch size, the
row position or the other rows; a TP output shard equals the same columns of the full product;
fused equals separate. The accumulation order inside one MMA instruction is the architecture's,
so the contract is bitwise per architecture family (validated on Hopper/H100; Ampere/Ada
use `mma.sync` and Blackwell its own instruction, so bytes may differ across families).

An earlier candidate, the warp-per-column GEMV of `h3-det-linear-v1` (also exact), was measured
at 50-250x cuBLAS time at thousands of tokens and rejected for these rows; it stays the right
kernel for the handful-of-rows conditioning layers.

## Accuracy

Against the FP64 golden (rounded once to the module dtype), measured as max error over max
|golden|: BF16 rows stay within 1.25x of the provider plus a 4e-3 one-ulp floor; FP32 rows
within 1.25x the provider plus 1e-6 (K = 32) or 4e-6 (K = 5376). A sequential FP32 chain over
K = 5376 lands at 2-3e-6 where cuBLAS's blocked sums reach about 5e-7: this is the price of a
fixed single-chain order, and the test names it. Gradients are checked against FP64 `g @ W`,
`g^T x`, `sum g`. Real checkpoint tensors (shard 1: audio, context embedder; block 0: Q/K/V) are
used when `RL_KERNEL_H3_WEIGHTS` points at them.

## Performance (H100 SXM, torch 2.13, forward, provider = cuBLAS with TF32 off)

Evidence: `tools/validation/models/h3_projection_bench.py`.

| Row | T = 64 | T = 1024 | T = 4096 |
| --- | --- | --- | --- |
| `audio_input_projection_fp32` | x2.6 | x2.4 | x2.1 |
| `audio_output_projection_fp32` | x11.9 | x9.4 | x3.5 |
| `text_context_projection` | x2.8 | x2.1 | x1.4 |
| `h3_qkv_gemm` (3 projections) | x1.7 | x1.4 | x1.4 |

Ratios are candidate time over provider time. `audio_output_projection_fp32` is 0.55 ms flat: with
`N = 32` the grid has only `T / 64` programs and one sequential FP32 chain over 5376, so it
cannot fill the GPU; it runs once per step on the audio tokens, and a per-row tile
configuration is the `cuda_perf_closeout` follow-up.

## Fail-Closed Contract

Wrong dtype (no promotion), missing or unexpected bias, wrong `K`, empty or non-2-D input,
widths that are neither full nor a whole-unit shard, mixed devices, CPU tensors for the CUDA
op, mismatched Q/K/V widths and a missing Triton all raise. Nothing falls back to cuBLAS.

## Tests

`tests/models/minimax_h3/test_h3_projection_rows.py`: reference/provider/golden agreement;
dtype/bias/shape rejection; forward and backward against FP64 on synthetic and pinned
weights; batch, position, permutation and padding invariance; TP shards; repeat and backward
determinism; fused vs separate QKV forward and backward bitwise; registry dispatch.
`h3_block` now binds Q/K/V to this row (`status="row"`).

## Hopper evidence (H100)

These outputs and figures are not part of this diff; they are on the fork's `pr_assets` branch (links in the PR description).
`reports/experiments/h3-projection-rows-h100/bench.json` (from `tools/validation/models/h3_projection_bench.py`) and the chart below record
forward time and relative error vs an FP64 `F.linear`, per row and token count, for the RL-Kernel operator and the cuBLAS provider.
The fixed-order kernels are 1.3x to 4.3x slower than cuBLAS on the BF16 rows; `audio_output_projection_fp32` (FP32, K=5376, ieee
precision, 64x64x32 tile) is the slowest (3.5x to 12x) and is the first candidate for a per-row tile sweep. Strict mode forbids
split-K, so this is the cost of the fixed reduction order.
