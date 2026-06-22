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
| PyTorch (`NativeGemmOp`) | **no** | Plain `torch.matmul`. Reference & benchmark target ONLY — cuBLAS is not batch-invariant. Excluded from registry dispatch. |

Registry dispatch for `det_gemm` includes only the deterministic backends
(CUDA → Triton). The PyTorch op must be called explicitly.

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