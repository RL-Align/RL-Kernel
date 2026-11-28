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

### Strict fixed-leaf tree

Muse-Glimmer dense projections use the `fixed_leaf_tree` mode for `Y`, `dX`,
and `dW`. Each GEMM partitions its reduction axis into consecutive 32-term
leaves starting at index zero. The last leaf can be shorter. Each leaf starts
from FP32 `+0` and visits its terms in ascending index order, using one scalar
FP32 FMA with a single rounding per term. The leaf result rounds to BF16
round-to-nearest-even (RNE). Only the valid terms of a short leaf participate,
preserving a computed `-0` that a padded `+0` FMA could erase.

Each range of `n` leaves splits after `n // 2` leaves. The parent adds its two
BF16 children in FP32 and rounds to BF16 RNE. Every merge, including the root,
rounds this way, so the output is BF16 even for a single-leaf tree. The weight
gradient uses the token axis as its reduction and returns a contiguous
`[N,K]` tensor.

- **Entry points** accept `fixed_leaf_tree=True` in CUDA
  `_C.det_gemm_fwd`, `_C.det_gemm_fwd_rhs_transposed`, and
  `_C.det_gemm_db_transposed`, and in Triton's `_triton_tree_gemm`. The
  ops-level `rl_engine.ops.gemm.fixed_leaf_linear(a, weight, backend=...)`
  selects this mode for BF16 `[M,K]` activations and native `[N,K]` weights,
  including both gradients through autograd.
- **Projection geometry** has tiled kernels on CUDA and NVIDIA Triton for
  reduction lengths 256, 4096, 6656, and 19968. CUDA output tiling requires
  `N % 64 == 0`, with `N` the number of output columns. For the
  RHS-transposed entry, this tiling starts at `M >= 16`. A vectorized path
  handles thinner batches for reduction lengths 4096, 6656, and 19968 with
  16-byte-aligned input pointers. Triton masks partial tiles and can use a
  dedicated one-row kernel when both operands have unit reduction stride.
  General scalar fixed-leaf kernels cover other reduction lengths and
  geometries that miss the specialized paths.
- **Weight-gradient geometry** uses a CUDA small-token kernel for `0 < M < 32`.
  For `M >= 32`, CUDA tiling requires `K % 64 == 0` and `N % 32 == 0` in
  the native `[N,K]` weight layout, subject to launch and shared-memory limits.
  Both backends have a specialized 96-token schedule and support short final
  leaves for other token counts.

CUDA rejects strict mode in fast-math builds. Build with
`KERNEL_ALIGN_USE_FAST_MATH=0`. The strict CUDA and Triton paths guarantee
bitwise-identical `Y`, `dX`, and `dW` on the qualified NVIDIA shapes. They
exclude tensor-core MMA because PTX leaves the BF16 `mma` accumulation order
and rounding unspecified. In the Muse-Glimmer K/V projection at
`M=96, K=6656, N=256`, the original CUDA MMA path differed from scalar Triton
by one BF16 ULP in 4 of 24,576 output words.
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

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
from rl_engine.runtime.registry import kernel_registry
gemm = kernel_registry.get_op("det_gemm")     # CUDA if built, else Triton
c = gemm(a, b)                                 # a:[M,K] bf16, b:[K,N] bf16
```

## Scope

In: single-rank forward + backward, BF16 / FP32-accum, SM80+.
Out: tensor-parallel GEMM (WS2), FP8, ROCm-native kernel (Triton covers ROCm).

## Performance

`det_gemm` trades speed for determinism. The naive CUDA kernel is slow by
design; see `benchmarks/operators/gemm/benchmark_det_gemm.py`. Overhead is reported vs cuBLAS
with TF32 disabled (the fair, same-FP32-path baseline), not as a speedup. A
slower deterministic baseline is the accepted first milestone (#146).
