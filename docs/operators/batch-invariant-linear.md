# Batch-Invariant Linear (ROCm)

## Summary

`y = x @ weightᵀ + bias` whose output row does not depend on the other rows in the
batch, on ROCm through aiter's `gemm_a16w16`. DSV4 RFC task 9
([#510](https://github.com/RL-Align/RL-Kernel/issues/510)): a shared GEMM
with a fixed K accumulation order, bias and output conversion, with fp32 output
for DeepSeek-V4's lm_head.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

linear = kernel_registry.get_op("batch_invariant_linear", device="cuda")
logits = linear(x, weight, out_dtype=torch.float32)  # x [..., K], weight [N, K]
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | — | — | Not implemented |
| ROCm | `AiterBatchInvariantLinearOp` in `rl_engine/backends/rocm/gemm/aiter_gemm.py` | aiter `gemm_a16w16` (Triton) | Forward batch invariant; backward through det_gemm's fixed-order Triton GEMM in fp32 |
| PyTorch fallback | — | — | None: a missing aiter raises |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[..., K]` | bf16 | other dtypes raise |
| `weight` | `[N, K]` | same as `x` | |
| `bias` | `[N]` | `out_dtype` | optional |
| output | `[..., N]` | `out_dtype`: bf16 (default) or fp32 | |

## Dispatch Behavior

The registry hands out `AiterBatchInvariantLinearOp` for `batch_invariant_linear`
on ROCm only, without fallback. CUDA, MUSA, CPU and NPU have no backend, and
`get_op` raises there. Inputs outside the tensor contract raise; an
empty batch returns an empty output without a launch. The semantic catalog
declares it for bf16 on ROCm with `batch_invariant: True`.

aiter chooses gemm_a16w16's config by (M, N, K), and some of its configs split K,
which makes results change with M. The op passes the config through aiter's
public `config` argument, chosen by `rl_engine/backends/rocm/config_utils.py`:

- A shape with a table for the arch of the input's GPU uses the config of M's
  bucket (`M_LEQ_x`, the smallest x ≥ M, else `"any"`). Today that is the
  DeepSeek-V4-Flash lm_head on gfx942,
  `configs/gfx942/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json`.
- Every other shape and GPU uses `FIXED_LINEAR_CONFIG` for every M.
- Checked when a table is loaded, with `ValueError` otherwise: every key is
  `M_LEQ_x` or `"any"`, the `"any"` entry exists, and every entry keeps the
  parameters that change the output bits. Measured on MI325X with Triton 3.7:
  `NUM_KSPLIT` and `kpack` change them; `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`,
  `num_warps`, `num_stages`, `waves_per_eu` and `cache_modifier` do not. Tables
  keep `NUM_KSPLIT=1`, `kpack=1` and `matrix_instr_nonkdim=16`; only whole-K
  configs are accepted.
- `rl_engine.backends.rocm.config_utils.selected_configs()` lists the configs
  that ran, one record per GPU arch, op, table file and bucket, with the smallest
  and largest M that used it (`m_min`, `m_max`); see
  [mHC](mhc.md#dispatch-behavior) for the fields.

## Accuracy

bf16 inputs, fp32 accumulation, one rounding to `out_dtype`. DeepSeek-V4's lm_head
multiplies in fp32, but `head.weight` is BF16 in the checkpoint and its input holds
bf16 values; the product of two bf16 values is exact in fp32, so only the
accumulation order differs from the reference. Output and input gradient match an
fp32 PyTorch linear within the WS1 contract tolerance
(`rl_engine/contracts/profiles/precision/ws1.json`, bfloat16 `reduction` cells of
`forward_accuracy` and `gradient_accuracy`).

## Performance Notes

Device time of the DeepSeek-V4-Flash lm_head GEMM: x [M, 4096] bf16 × weight
[129280, 4096] bf16, fp32 logits; ms per call from GPU events, median over 5
trials, one run. Baseline: aiter's own config (no batch-invariance guarantee). Ours: the
gfx942 table. The output bits of ours equal `FIXED_LINEAR_CONFIG`'s at every M.

| M | baseline: aiter's own config (ms) | ours: tuned table (ms) | ours / baseline (ratio) | reference: `FIXED_LINEAR_CONFIG` (ms) | reference: `torch.matmul`, hipBLASLt, fp32 output, not batch invariant (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2715 | 0.2155 | 0.79 | 0.2725 | 0.2412 |
| 16 | 0.2865 | 0.2242 | 0.78 | 0.2866 | 0.2536 |
| 17 | 0.2881 | 0.2446 | 0.85 | 0.2895 | 0.2578 |
| 64 | 0.2959 | 0.2755 | 0.93 | 0.2987 | 0.2919 |
| 256 | 0.6756 | 0.5810 | 0.86 | 1.1509 | 0.5335 |
| 1024 | 1.7177 | 1.6855 | 0.98 | 4.4109 | 1.5802 |
| 3000 | 5.2874 | 5.2716 | 1.00 | 13.0877 | 4.7584 |
| 4096 | 6.2061 | 6.1617 | 0.99 | 17.8398 | 6.4660 |
| 8192 | 12.2308 | 12.2341 | 1.00 | 35.7319 | 12.7593 |

Measured before the port, on the KLR branch with the same aiter kernels (aiter
`326cffa`, `AITER_TRITON_ONLY=1`), the same launch-config tables and the same
benchmark code, on AMD Instinct MI325X (gfx942), Triton 3.7, torch 2.12, ROCm 7.14.
Rerun the commands below to measure this branch.

```bash
python benchmarks/operators/gemm/benchmark_lm_head_linear.py --config-table \
    rl_engine/backends/rocm/configs/gfx942/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json
```

`benchmarks/tuning/tune_batch_invariant_linear.py` regenerates the table.

## Tests

```bash
python -m pytest tests/backends/rocm/test_rocm_batch_invariant_linear.py \
    tests/backends/rocm/test_rocm_launch_config.py -v
```

Covers the DeepSeek-V4 mHC projection shapes (N = 24 and 4, K = 16384), the WS1
manifest's GEMM shapes (K = 4096, N = 4096 and 12288), a shape whose N and K are
not tile multiples, the lm_head shape, bf16 and fp32 output, and bias: a row alone
and in batches of M = 1 to 16384 (8192 for lm_head, including the WS1 manifest's
8 and 59), chunking, reordering and padding, other input strides, repeated runs,
the fp32 reference, the input gradient, empty batches and unsupported inputs.
Every shape but lm_head runs `FIXED_LINEAR_CONFIG`.

## Known Limitations

- Only the lm_head shape has a tuned table, measured on gfx942 with Triton 3.7;
  other shapes use one fixed config.
- `matrix_instr_nonkdim=32` gave the same bits as 16 in the measurement, but it was
  not checked that Triton emitted a different instruction, so tables keep 16 and
  the table check treats it as a parameter that changes the bits.
- No CUDA kernel.
