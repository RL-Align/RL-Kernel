# MHC H Aggregate

`mhc_pre_h_aggregate` collapses four residual streams into the hidden activation
used by the mHC controller. It provides CUDA, Triton, and PyTorch implementations
with an explicit FP32 backward boundary for the composite controller.

## Interface

```python
from rl_engine.kernels.registry import KernelRegistry

op = KernelRegistry().get_op("mhc_pre_h_aggregate", device=residual.device)
hidden = op(residual, pre)
grad_residual, grad_pre = op.backward_fp32(grad_hidden, residual, pre)
```

| Tensor | Shape | Dtype |
|---|---|---|
| `residual` / `r_old` | `[T, 4, D]` | BF16 |
| `pre` | `[T, 4]` | FP32 |
| `hidden` | `[T, D]` | BF16 |
| `grad_hidden` / `dh` | `[T, D]` | BF16 or FP32 |
| `grad_residual` | `[T, 4, D]` | FP32 |
| `grad_pre` | `[T, 4]` | FP32 |

`D=4096` is the production geometry; `D=128` supports the P1 acceptance fixtures.
Inputs must be contiguous and on the same device. Empty batches are supported.
CUDA and Triton require NVIDIA SM80 or newer; PyTorch also supports CPU.

On CUDA, registry priority is CUDA → Triton → PyTorch. Import/initialization
failures advance to the next registered backend. Unsupported inputs raise at
execution instead of silently selecting a different implementation. Instantiate
`MHCPreHAggregateCudaOp` or `MHCPreHAggregateTritonOp` directly to require a backend.
The registry's existing `get_op` selection logic is unchanged.

CUDA and Triton expose `backward_fp32` rather than standalone autograd. Their
forward rejects grad-enabled inputs requiring autograd. The composite caller
combines the FP32 residual gradients with the controller contribution before any
BF16 conversion. In particular, FP32 `dh` from RMSNorm is consumed without an
intermediate BF16 cast.

## Numerical contract

Forward converts each residual value to FP32, evaluates
`(pre0 * R0 + pre1 * R1) + (pre2 * R2 + pre3 * R3)`, then casts once to BF16.
Multiplication and addition are separate operations, without FMA contraction.

Backward computes `dR = dh * pre`. Each `dPRE` uses one FP32 accumulator initialized
to zero, adding `dh[d] * R[d]` in ascending `d` order. There are no atomics or tree
reductions in this fold. This retains the P1 `fixed_sum` contract; replacing it
with a faster reduction changes the required bytes.

Both implementations must match the P1 oracle byte-for-byte at the forward and
FP32 gradient boundaries. Batch invariance means the same token produces the
same forward and backward bytes when run alone, packed with other tokens, or
surrounded by padding. Tests include the CUDA launch transition at 128 tokens.

## P1 providers and validation

`CudaHAggregateProvider` and `TritonHAggregateProvider` in
`rl_engine.mhc.h_aggregate_provider` implement the oracle signatures:

```python
hidden = provider.h_aggregate_fwd(pre, r_old)
dr, dpre = provider.h_aggregate_bwd(dh, pre, r_old)
```

They replace only `h_aggregate`; other P1 operators use `ReferenceProvider`.
Provenance reports the selected aggregate backend plus `oracle-rest`. Providers
invoke their backend directly and do not fall back.

```bash
python -m pytest tests/test_mhc_pre_h_aggregate.py tests/test_p1_provider.py -q
python scripts/check_p1.py --provider rl_engine.mhc.h_aggregate_provider:CudaHAggregateProvider --device cuda
python scripts/check_p1.py --provider rl_engine.mhc.h_aggregate_provider:TritonHAggregateProvider --device cuda
python scripts/check_operator.py --op mhc_pre_h_aggregate --candidate cuda --device cuda --dtype bf16 --batch 1 --seq 16 --check-grad
python scripts/check_operator.py --op mhc_pre_h_aggregate --candidate triton --device cuda --dtype bf16 --batch 1 --seq 16 --check-grad
```

The unit tests cover oracle numerical correctness, FP32/BF16 upstream gradients,
ascending-sum cancellation, repeated execution, forward/backward batch invariance,
empty batches, invalid inputs, registry order, and actual provider invocation.

## Benchmark

```bash
python benchmarks/benchmark_mhc_pre_h_aggregate.py --tokens 1 16 128 129 1024 --hidden 4096 --grad-dtype fp32 --warmup 3 --repeat 10
```

The script checks forward and backward bytes against the P1 oracle before timing
each backend. It prints a Markdown table of median CUDA-event latency and extra
peak allocated memory, including outputs and temporaries but excluding existing
inputs and oracle tensors. Backward timing includes both gradient outputs. The
PyTorch baseline uses the same ascending FP32 fold, so its many separate kernel
launches are included. These measurements are not a comparison against a tree
reduction with a different numerical contract.

### H100 results

Measured on 2026-10-12 with an NVIDIA H100 80GB HBM3, PyTorch 2.11.0+cu128,
Triton 3.6.0, and CUDA 12.8. `D=4096`, FP32 upstream gradients, 3 warmups,
10 timed repetitions; commands and timing semantics are listed above.

| T | Backend | Forward (us) | Backward (us) | Forward extra (MiB) | Backward extra (MiB) | Oracle bytes |
|---:|:---|---:|---:|---:|---:|:---|
| 1 | PyTorch | 91.42 | 32145.10 | 0.172 | 0.188 | PASS |
| 1 | CUDA | 11.68 | 97.33 | 0.008 | 0.063 | PASS |
| 1 | Triton | 38.22 | 91.97 | 0.008 | 0.063 | PASS |
| 16 | PyTorch | 90.70 | 31972.37 | 2.750 | 3.000 | PASS |
| 16 | CUDA | 10.80 | 98.00 | 0.125 | 1.000 | PASS |
| 16 | Triton | 36.29 | 89.58 | 0.125 | 1.000 | PASS |
| 128 | PyTorch | 88.51 | 32385.26 | 22.000 | 24.000 | PASS |
| 128 | CUDA | 10.96 | 100.77 | 1.000 | 8.002 | PASS |
| 128 | Triton | 36.91 | 89.62 | 1.000 | 8.002 | PASS |
| 129 | PyTorch | 83.44 | 32143.55 | 22.172 | 24.188 | PASS |
| 129 | CUDA | 11.78 | 100.75 | 1.008 | 8.065 | PASS |
| 129 | Triton | 37.62 | 93.12 | 1.008 | 8.065 | PASS |
| 1024 | PyTorch | 196.24 | 32375.39 | 176.000 | 192.000 | PASS |
| 1024 | CUDA | 19.25 | 235.20 | 8.062 | 64.016 | PASS |
| 1024 | Triton | 41.41 | 167.12 | 8.062 | 64.016 | PASS |

Both CUDA and Triton passed all five `check_p1.py` cases with byte-equal boundaries.
Both also passed gtest forward/gradient accuracy and forward/backward invariance.
The focused operator, gtest, and dispatch regression suite passed 122 tests.
The existing `test_golden_manifest_anchor` fails under this environment on both
the unmodified PR head and this revision; the oracle and committed manifest were
not changed. Same-device P1 comparisons above pass independently of that stored
CPU manifest.
