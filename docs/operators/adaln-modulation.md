# AdaLN Modulation

## Summary

AdaLN modulates Qwen-Image attention or MLP inputs with LayerNorm, shift and scale.
It returns a separate gate for downstream use; the gate is not multiplied into the
modulated output.

Image/text streams and attention/MLP calls use separate inputs. The conditioning
projection (`SiLU + Linear`) and gated residual update are separate operators.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

# x: [B, S, H]; modulation: [B, 3H], ordered as shift, scale, gate.
op, trace = kernel_registry.get_adaln_modulation_op(x.device, hidden=x.shape[-1])
modulated, gate = op(x, modulation, eps=1e-6)
```

Each token is normalized over H without learned LayerNorm affine parameters:

```text
modulated = LayerNorm(x, eps) * (1 + scale) + shift
```

## Tensor Contract

| Value | Shape | Dtype / requirements |
| --- | --- | --- |
| `x` | `[B,S,H]` | FP32 or BF16; B, S and H must be positive |
| Shared `modulation` | `[B,3H]` | Same device and dtype as x |
| `modulated` | `[B,S,H]` | Same device and dtype as x |
| Shared `gate` | `[B,1,H]` | Same device and dtype as x; retains gradients to modulation |
| `eps` | scalar | Defaults to `1e-6`; see Numerical Contract |

Shared shift, scale and gate apply to every token in the same sample. GPU wrappers
accept strided inputs and copy them to contiguous storage before kernel execution.

### Indexed modulation

For image editing, pass `modulate_index` to both resolution and execution:

```python
op, trace = kernel_registry.get_adaln_modulation_op(
    x.device, hidden=x.shape[-1], modulate_index=modulate_index
)
modulated, gate = op(x, modulation, modulate_index=modulate_index)
```

| Value | Indexed contract |
| --- | --- |
| `modulation` | `[2B,3H]`; same device and dtype as x |
| `modulate_index` | Boolean or integer `[B,S]` or `[1,S]`; values 0 or 1; same device as x |
| `modulated` | `[B,S,H]`; same device and dtype as x |
| `gate` | `[B,S,H]`; same device and dtype as x |

Index 0 selects the sample's modulation from the first B rows; index 1 selects it
from the second B rows. A `[1,S]` index broadcasts across the batch.

## Backends and Dispatch

| Backend | Implementation | Use |
| --- | --- | --- |
| CUDA | `CudaAdaLNModulationOp` | NVIDIA CUDA extension; H <= 4096 |
| Triton | `TritonAdaLNModulationOp` | NVIDIA CUDA validation target; ROCm is not qualified |
| PyTorch | `NativeAdaLNModulationOp` | CPU fixed-order baseline and framework fallback |

On NVIDIA CUDA, shared modulation tries CUDA, then Triton, then PyTorch.
H > 4096 skips CUDA. Missing extension symbols also permit fallback; fast-math
builds omit the AdaLN CUDA symbols.

CPU, ROCm, MUSA and NPU registry entries select PyTorch. Indexed modulation always
selects the PyTorch select01 path, with `fallback=true` and reason
`modulate_index_requires_select01`.

### Strict mode and trace

Pass `strict=True` when resolving the shared path to require its declared arithmetic
profile. CPU Native and NVIDIA CUDA/Triton are permitted; indexed modulation,
unknown backends and PyTorch execution on GPU are rejected.

Strict mode rechecks the input device and hidden size at execution. It enforces
the declared calculation policy, not hardware qualification. The indexed path
does not claim fixed-order CUDA/Triton backward equivalence.

The returned callable is an `AdaLNDispatch` wrapper with an `implementation`
attribute. `selected_backend` records resolution; `actual_backend`, dtype and
device record the latest successful forward. A failed call clears that observation.

Copy `trace` to retain an observation and use separate resolutions for concurrent
callers. A successful-forward trace is not a profiler measurement or proof of
backward execution. Semantic fingerprints are not compiled-binary hashes.

## Numerical Contract

Shared CUDA/Triton arithmetic and the CPU Native baseline use an FP32 hidden
reduction tree, pairing lower and upper halves after padding to the next power
of two. Shared parameter gradients accumulate in ascending logical token order.

Intermediates and accumulators remain FP32, with one final cast for BF16 results.
CUDA/Triton kernels use no atomic partial sums, Split-K, Stream-K, TF32 or fast math.

For shared modulation, eps must be finite, nonnegative and round to finite
FP32. Positive values that round to zero are rejected; FP32 subnormals, including
2^-149, are accepted.

Indexed modulation checks only finite, nonnegative eps.

With eps=0, zero variance produces NaN. Nonfinite normalization arithmetic may
propagate NaNs; those NaN payloads are outside byte-equality guarantees.
The shared path preserves gate/dgate payload bits, including nonfinite values.

### Fixed-order implementation and independent gold

`NativeAdaLNModulationOp` uses the fixed reduction order for fallback execution and
byte comparisons. Its square root is evaluated in FP64 and rounded to FP32 to match
correctly rounded CUDA sqrt; its reductions remain FP32.

`GtestAdaLNReference` is the independent accuracy gold. It uses CPU FP32
`F.layer_norm` and autograd for normalization, plus an independent affine backward
that follows the specified token order. It calls no production reduction helpers.

The validation harness creates FP32 reference input leaves to retain FP32 gradients.
It checks both outputs and gradients of x and modulation using the shared
`reduction` tolerances for `forward_accuracy` and `gradient_accuracy`.

Accuracy checks and fixed-reference byte comparisons are separate tests.
The CLI records trace snapshots without automatically constructing C1
`BackendProvenance`; its accuracy report alone is not a WS1 system gate.

## Tests and Benchmark

```bash
python -m pytest tests/ops/norm/test_adaln_modulation.py \
  tests/validation/operators/test_adaln_gtest.py tests/runtime/test_adaln_dispatch.py -q -rs

python tools/validation/operators/check_operator.py --op adaln_modulation \
  --candidate pytorch --device cpu --dtype bf16 --batch 2 --seq 3 \
  --normalized-dim 3072 --check-grad --json

python benchmarks/operators/norm/benchmark_adaln_modulation.py --real --dtype bf16 --json
```

Tests cover forward/gradient accuracy, fixed-order byte equality, dtype boundaries,
strided inputs/upstreams, gate-only gradients, nonfinite values and dispatch.
Set `RLK_ADALN_REAL_SHAPES=1` to include the three full-image workloads.

The benchmark reports median forward and forward+backward latency and peak allocated
memory. Use `--dtype fp32` for FP32, `--warmup`/`--repeat` for sample counts and
`--seed` for reproducible inputs (default 386).

Forward+backward timing includes upstream `ones_like` allocations.
The ordinary `F.layer_norm` plus FP32 affine baseline uses the same timing method,
without a fixed-order replay claim.

See the [H100 validation runbook](../validation/adaln-h100.md) for workload shapes,
fresh extension builds, sanitizer commands and evidence records.

## Known Limitations

The CUDA extension requires B <= 65535 and at most 2^31-1 elements in each input.
Python checks these bounds before contiguous copies; C++ checks them before launch.

Exceeding these limits raises an error; resolution only uses H to select fallback.

Registry dispatch uses default launch settings. Direct constructors accept
CUDA `threads={128,256,512}` (default 256), Triton `num_warps={4,8}` (default 4)
and `reduction_tile={64,128,256}` (default 128).

These settings alter scheduling, not the declared reduction order. GPU support
and numerical acceptance must be checked on the target hardware; CPU tests
alone do not establish CUDA/Triton acceptance.
