# Attn-Out Bias GEMM (Qwen-Image WS1)

Deterministic attention output projection for the Qwen-Image MMDiT:
`y = single_cast(x @ W.T + b)` over the already-dtype-rounded joint-attention
output. Covers the mathematically identical `to_out` (image stream) and
`to_add_out` (text stream) call sites, `[3072, 3072]` with bias.

## Summary

Issue [#386](https://github.com/RL-Align/RL-Kernel/issues/386) kernel-table
row `attn_out_bias_gemm` (forward + backward; WS2 will add the TP-row parallel
variant). The numeric contract version is
`attn-out-bias-gemm-tree-v1`, frozen as follows:

- **K/N-dim reduction tree**: the reduction length splits into 32-wide leaves
  (short tail allowed); each leaf is an ascending-k fp32 chain from `+0.0`;
  leaves merge through a mid-split tree `T(l, r) = T(l, m) + T(m, r)`. The
  tree depends only on the reduction length, so row outputs are
  batch-invariant by construction, and a contiguous half-K split composes
  (WS2 TP-row friendly).
- **Multiply-add discipline**: FMA (fused, one rounding per multiply-add) on
  every device backend, via explicit intrinsics (`libdevice.fma_rn` /
  `__fmaf_rn`). On bf16 inputs FMA and separate mul-add are provably and
  measurably bit-identical (products are exact in fp32).
- **RNE everywhere**; bias is added once in fp32 after the complete tree;
  the single fp32-to-output-dtype cast happens at the final store. No
  split-K, no atomics, no tensor-core mma, no fast-math, no TF32.
- **S-dim reductions** (parameter gradients `dW`/`db`) use ascending-row left
  folds; `dx = dY @ W` reuses the K-dim tree.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("attn_out_bias_gemm")           # dispatch
out = op(x, weight, bias=bias)                              # [*lead, 3072]
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton (leaf chunks + shared host tree combine) | `TritonAttnOutBiasGemmOp` | — | Supported (default on CUDA) |
| CUDA (per-thread full tree) | `CudaAttnOutBiasGemmOp` | `_C.attn_out_bias_gemm_cuda_forward`, `_C.attn_out_tree_gemm_cuda`, `_C.attn_out_dw_left_fold_cuda` | Supported |
| PyTorch same-tree reference | `NativeAttnOutBiasGemmOp` | — | Gold + CPU fallback |

ROCm dispatches Triton then PyTorch (the CUDA source is not ROCm-validated).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[*lead, 3072]` | bf16 / fp32 | already rounded to the input dtype by the caller |
| `weight` | `[3072, 3072]` | same as `x` | HF `[out, in]` convention |
| `bias` | `[3072]` | same as `x` | added once, in fp32, after the tree |
| return | `[*lead, 3072]` | same as `x` | single RNE cast at the output |

## Dispatch Behavior

- CUDA: Triton -> CUDA -> PyTorch (Triton leaf chunks are the product path;
  the CUDA per-thread kernel is the memory-light correctness anchor)
- ROCm: Triton -> PyTorch
- CPU: PyTorch

## Accuracy

The acceptance structure is star-shaped with a single gold: the FP32 CPU
same-tree reference (explicit elementwise ops, no matmul, no implicit
reductions, bit-identical on every device).

- **bf16 inputs (working dtype)**: every backend matches the reference byte
  for byte (products exact in fp32; FMA and separate mul-add coincide).
- **fp32 inputs**: device backends match each other byte for byte (same FMA
  discipline); against the torch reference (which cannot express elementwise
  FMA on Python 3.12) a tight declared tolerance of a few ulps applies.
- **batch invariance**: bitwise (`torch.equal` plus a dtype-bitcast
  assertion on the logical elements) across batch composition, leading
  shapes, padding, and repeat runs, for outputs and `dx` alike.
- Tolerance values for the generic gtest matrix come from
  `rl_engine/kernels/gtest/tolerance_contract.json`; this page intentionally
  does not restate ad-hoc numbers.

## Performance Notes

```bash
python benchmarks/benchmark_attn_out_bias_gemm.py --dtype bf16
python benchmarks/benchmark_attn_out_bias_gemm.py --rows 512,4096,6889 --backward
```

Measured on RTX 5090 (sm_120), bf16 forward, torch 2.13.0+cu130: Triton
leaf-chunk backend 1.9 ms at S=512, 17 ms at S=4096, 33 ms at S=6889 with a
peak of ~3.2 GB; the CUDA per-thread kernel trades speed (~6x slower) for
O(S*N) memory (no partial planes). The Triton path bounds partials to
`(12 + log2(96))` output tiles via chunked depth-first combination.

## Tests

```bash
python -m pytest tests/test_attn_out_bias_gemm.py -q
python scripts/check_operator.py --op attn_out_bias_gemm --candidate pytorch --device cpu --dtype fp32
python scripts/check_operator.py --op attn_out_bias_gemm --candidate triton --device cuda --dtype bf16 --check-grad
python scripts/check_operator.py --op attn_out_bias_gemm --candidate cuda --device cuda --dtype bf16 --check-grad
```

The test file also carries the bit-equality harness (four-step comparison:
shape, dtype, contiguous materialisation, dtype-bitcast of the logical
elements) required by the issue acceptance list.

## Known Limitations

- The Triton forward materialises leaf partial planes per chunk; peak extra
  memory is ~3.2 GB at the largest acceptance tier (S=6889). Streamed or
  in-kernel tree combination is future work.
- Tensor-core (mma) paths are excluded on purpose: PTX does not specify the
  internal accumulation tree of `mma.sync`, so the reduction tree cannot be
  frozen. Any future mma path must re-pass the bit-equality harness and
  revise the contract first.
- WS2 (TP-row) sharding is reserved in the design (contiguous half-K tree
  composition) but not implemented in WS1.
