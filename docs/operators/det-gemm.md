# Batch-Invariant Deterministic GEMM (`det_gemm`)

WS1 #146. A matrix multiply whose output for a given row is **bitwise invariant**
to batch size, chunked-prefill splitting, and padding — the property cuBLAS does
not provide, and the root fix for matmul-driven KL drift between rollout and
training.

## Why

Matmul is the most frequent op in a transformer (QKV, MLP, LM head), so
batch-dependent drift here dominates everything downstream. cuBLAS selects
kernels by problem shape and may use split-K, both of which change the
K-reduction order when batch size or sequence length shifts the chosen kernel.
`det_gemm` pins the accumulation order so a row's result never depends on the
rows around it.

## Guarantees

- Forward `C = A @ B`, backward `dA = dC·Bᵀ`, `dB = Aᵀ·dC`.
- BF16 inputs, FP32 accumulation, no TF32, no split-K, fixed K-loop order.
- Bitwise-identical output for a fixed row across batch=1/N, chunked-prefill
  on/off, and padding layouts.

## Backends

| Backend | Deterministic | Notes |
|---|---|---|
| CUDA (`DetGemmOp`) | yes | Hand-written kernel. First milestone is a naive FP32 implementation (correctness first); a tensor-core (`mma.sync`) pass matching `prefix_shared_attention.cu` follows. NVIDIA SM80+. |
| Triton (`TritonDetGemmOp`) | yes | Autotune disabled, BLOCK pinned, no split-K. Portable / ROCm fallback and cross-backend reference. |
| Ascend (`DetGemmAscendOp`) | yes | Ascend C (CANN) forward + backward. Mirrors the CUDA kernel's mid-split K tree: 32-element FP32 leaves rounded to BF16, merged with BF16 adds in fixed ascending leaf order; every output row-tile is reduced end-to-end by one AI-core block with a `MAX_BLOCKS`-capped strided launch, so no split-K merge exists. |
| PyTorch (`NativeGemmOp`) | **no** | Plain `torch.matmul`. Reference & benchmark target ONLY — cuBLAS is not batch-invariant. Excluded from registry dispatch. |

Registry dispatch for `det_gemm` includes only the deterministic backends
(CUDA → Triton on CUDA/ROCm, Ascend on NPU). The PyTorch op must be called
explicitly.

### Ascend NPU backend

`DetGemmAscendOp` implements the same strict contract on the NPU through
`_C_npu.det_gemm_ascend_*` (Ascend C, built with `KERNEL_ALIGN_FORCE_ASCEND=1`):

- **Entry points mirror the CUDA surface 1:1**: `det_gemm_ascend_fwd`,
  `fwd_rhs_transposed`, `fwd_fp32`, `da`, `db`, `db_transposed`. Backward
  reuses the forward kernel on transposed operands (`dA = dC @ Bᵀ`,
  `dB = Aᵀ @ dC`; `db_transposed` stores the weight gradient born contiguous
  in the canonical `[N,K]` layout).
- **Reduction tree**: 32-element leaves accumulate in FP32 in ascending order
  and round once to BF16 (round-to-nearest-even); leaves merge through the
  CUDA `mid_tree_merge_count` mid-split tree of BF16 adds. A contiguous
  half-K GEMM is one child of the tree, so simulated TP=2 (a+b) matches
  TP=1 bitwise — the same TP contract as CUDA/Triton.
- **Batch-invariance**: each `(row, 128-column tile)` is processed end-to-end
  by one AI-core block in fixed leaf order; tiles are strided across at most
  128 blocks, so per-element numerics depend only on `R`, never on `M` or
  block assignment.
- The Ascend vector unit's fixed 32-lane MAC order inside a leaf differs from
  CUDA's sequential leaf accumulation, so cross-platform bitwise parity with
  the CUDA kernel is not claimed — the guarantee is batch-invariant
  determinism and the contiguous-half-K TP property (the same platform-level
  contract every other Ascend kernel in this repo provides).

Dtypes: BF16 in, FP32 accumulation, BF16 out (`forward_fp32` returns FP32).
The reduction dimension is capped at 32768 (the training contract), matching
the CUDA tree-depth budget.

## Usage

```python
from rl_engine.kernels.registry import kernel_registry
gemm = kernel_registry.get_op("det_gemm")     # CUDA if built, else Triton
c = gemm(a, b)                                 # a:[M,K] bf16, b:[K,N] bf16
```

## Scope

In: single-rank forward + backward, BF16 / FP32-accum, SM80+.
Out: tensor-parallel GEMM (WS2), FP8, ROCm-native kernel (Triton covers ROCm).

## Performance

`det_gemm` trades speed for determinism. The naive CUDA kernel is slow by
design; see `benchmarks/benchmark_det_gemm.py`. Overhead is reported vs cuBLAS
with TF32 disabled (the fair, same-FP32-path baseline), not as a speedup. A
slower deterministic baseline is the accepted first milestone (#146).
