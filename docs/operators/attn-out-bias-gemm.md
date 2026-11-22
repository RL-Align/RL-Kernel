# Attn-Out Bias GEMM (`attn_out_bias_gemm`)

Deterministic attention output projection for the Qwen-Image MMDiT:
`y = single_cast(x @ W.T + b)` over the already-dtype-rounded joint-attention
output. Covers the mathematically identical `to_out` (image stream) and
`to_add_out` (text stream) call sites, `[3072, 3072]` with bias.

WS1 #386 kernel-table row (forward + backward; WS2 will add the TP-row
parallel variant). The numeric contract version is
`attn-out-bias-gemm-tree-v1`, frozen as follows:

- **K/N-dim reduction tree**: the reduction length splits into 32-wide leaves
  (short tail allowed); each leaf is an ascending-k fp32 chain from `+0.0`;
  leaves merge through a mid-split tree `T(l, r) = T(l, m) + T(m, r)`. The
  tree depends only on the reduction length, so row outputs are
  batch-invariant. Two partial sums reproduce the root addition when the
  partition follows its left and right subtrees. For K=3072 there are 96
  leaves and the root splits them into two groups of 48 at K offset 1536.
- **Multiply-add discipline**. The leaf chains and `dW` fold use FP32 FMA.
  Triton uses `tl.fma`, native CUDA uses `__fmaf_rn`, and the reference
  uses an exact FP64 product and a compensated sum with round-to-odd before narrowing
  on CPU, and `torch.addcmul` on CUDA. Tests compare the primitive with `libm fmaf`.
- **RNE everywhere**; bias is added once in fp32 after the complete tree;
  the single fp32-to-output-dtype cast happens at the final store. No
  split-K, no atomics, no tensor-core mma, no fast-math, no TF32.
- **S-dim reductions** (parameter gradients `dW`/`db`) use ascending-row left
  folds; `dx = dY @ W` reuses the K-dim tree.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("attn_out_bias_gemm")           # dispatch
out = op(x, weight, bias=bias)                              # [*lead, 3072]
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton (leaf chunks + device tensor additions) | `TritonAttnOutBiasGemmOp` | — | Supported (default on CUDA) |
| CUDA (per-thread full tree) | `CudaAttnOutBiasGemmOp` | `_C.attn_out_bias_gemm_cuda_forward`, `_C.attn_out_tree_gemm_cuda`, `_C.attn_out_dw_left_fold_cuda` | Supported |
| PyTorch same-tree reference | `NativeAttnOutBiasGemmOp` | — | Gold + CPU fallback |

The NVIDIA CUDA build includes the native attn-out
CUDA bindings only with strict math flags. For example, `--use_fast_math`
enables flush-to-zero. An FP32 FMA of `2^-126 * 0.5 + 0` then produces
zero, while strict math preserves the exactly representable `2^-127`.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[*lead, K]` | bf16 / fp32 | already rounded to the input dtype by the caller |
| `weight` | `[N, K]` | same as `x` | HF `[out, in]` convention |
| `bias` (optional) | `[N]` | same as `x` | added once in fp32 when supplied |
| return | `[*lead, N]` | same as `x` | single RNE cast at the output |

Qwen-Image uses K=N=3072. Public operators accept positive K and N,
including rectangular weights and empty batches. All input tensors must
share a device. Low-level tree GEMM helpers define an empty reduction as
a zero dot product. Backward computes only the
gradients requested by autograd, including when projection weights are frozen.

## Dispatch Behavior

- CUDA: Triton -> CUDA -> PyTorch (Triton leaf chunks are the product path;
  the CUDA per-thread kernel is the memory-light correctness anchor)
- CPU: PyTorch

## Accuracy

The PyTorch reference evaluates the tree with elementwise operations on the
input device. In the dedicated tests it also serves as an FP32 CPU reference.
CPU forward and backward require gradual underflow to preserve subnormal bits.
If denormal flushing is enabled, the reference raises an error. Disable it with
`torch.set_flush_denormal(False)` before execution and CPU worker initialization.

- **Device comparisons.** At K=N=3072 the shape sweep compares CUDA and
  Triton outputs directly in bf16 and fp32. At S=1, 7 and 300 it also
  compares native CUDA directly with the CPU reference.
- **Batch invariance.** Tests compare output rows across changes in batch
  composition, leading shape, padding and repeated runs. The reference
  gradient test compares `dx` for a row slice with the same upstream rows
  from the full batch.
- **Strict acceptance.** For the forward output and each requested gradient
  (`dx`, `dW`, `db`), compare the candidate and reference at the final
  output dtype. A NaN or either sign of infinity on either side fails that
  comparison, even if the two encodings are identical. Finite results must
  match in shape, dtype and every logical element's bits, including the sign
  of zero and subnormal values. This rule applies during validation.
- The generic gtest matrix enforces the same bar for this operator
  (`bitwise_strict` in `rl_engine/validation/operators/operator_specs.py`): the
  forward pass criterion is the gold cast ONCE to the output dtype, compared
  on the dtype bitcast; the gradient leg differentiates the gold OP itself
  (its ``autograd.Function.backward`` IS the frozen chain) with the same
  upstream bits as the candidate, then compares bit-for-bit. Tolerances are
  never consulted for ``passed``; the printed error statistics (against the
  fp32 gold) are diagnostics only.

## Performance Notes

```bash
python benchmarks/benchmark_attn_out_bias_gemm.py --dtype bf16
python benchmarks/benchmark_attn_out_bias_gemm.py --rows 512,4096,6889 --backward
```

The commands above report average elapsed time and speedup relative to the
PyTorch reference running on the same GPU. Recorded RTX 5090 results
(sm_120, bf16, torch 2.13.0+cu130) are listed below. The memory columns
give the separately reported peak allocated GPU memory.

| Rows | Triton forward | Speedup | Forward + backward | Speedup | Peak forward | Peak forward + backward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.9 ms | 26.5× | 7.1 ms | 16.8× | 0.17 GiB | 0.24 GiB |
| 4096 | 17.5 ms | 21.1× | 72.1 ms | 12.3× | 1.05 GiB | 1.31 GiB |
| 6889 | 41 ms | 16.1× | 152.5 ms | 10.0× | 1.71 GiB | 2.07 GiB |

The native CUDA kernel was about 3–7× slower than Triton across these tiers
and modes. Each CUDA forward thread computes one output element and writes
it directly. Triton computes up to 12 full FP32 leaf planes per chunk and
combines the chunks depth first.

## Tests

The validation CLI defaults to the model width of 3072 for this operator.

```bash
python -m pytest tests/ops/gemm/test_attn_out_bias_gemm.py -q
python scripts/check_operator.py --op attn_out_bias_gemm --candidate pytorch --device cpu --dtype fp32
python scripts/check_operator.py --op attn_out_bias_gemm --candidate triton --device cuda --dtype bf16 --check-grad
python scripts/check_operator.py --op attn_out_bias_gemm --candidate cuda --device cuda --dtype bf16 --check-grad
```

The dedicated harness checks finiteness and compares contiguous integer
views (`int16` for bf16, `int32` for fp32).

## Implementation Files

- `rl_engine/reference/gemm/attn_out_bias_gemm.py` — FP32 same-tree reference (gold)
- `rl_engine/backends/shared/triton/gemm/attn_out_bias_gemm.py` — Triton leaf-chunk backend
- `rl_engine/backends/cuda/gemm/attn_out_bias_gemm.py` — CUDA backend wrapper
- `csrc/cuda/gemm/attn_out_bias_gemm.cu` — CUDA kernels
- `rl_engine/runtime/registry.py`
- `rl_engine/validation/operators/operator_specs.py`
- `tests/ops/gemm/test_attn_out_bias_gemm.py`
- `benchmarks/benchmark_attn_out_bias_gemm.py`

## Known Limitations

- The Triton forward materialises leaf planes in GPU memory. Streaming
  those partials or combining them inside the kernel is future work.
- Tensor-core (mma) paths are excluded on purpose: PTX does not specify the
  internal accumulation tree of `mma.sync`, so the reduction tree cannot be
  frozen. Any future mma path must re-pass the bit-equality harness and
  revise the contract first.
- WS2 TP-row sharding at the root split is planned for a later implementation.
