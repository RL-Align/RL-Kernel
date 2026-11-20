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

CPU, ROCm, MUSA and NPU registry entries select PyTorch. Dedicated indexed CUDA
and NVIDIA Triton accept FP32/BF16, H <= 4096, B <= 65535, both route shapes and
bool/uint8/int8/int16/int32/int64 indices. Inputs and upstreams may be strided or
expanded; internal copies preserve gradients to source leaves. Indexed route storage
is converted to contiguous int32 after validation.

Indexed CUDA/Triton check element counts and accessed element offsets before copies or
kernel output/temporary allocations. The checks include both modulation banks,
route storage, y/gate/dx/dmod and each FP32 partial buffer. The B grid dimension
contains B entries; each reduction thread writes both banks.

Valid unsupported geometry, addressing or eps executes the existing framework
select01 fallback in ordinary dispatch. A resolved indexed CUDA implementation
records `indexed_cuda_domain_unsupported`; resolution that selected framework or a
shared implementation records `modulate_index_requires_select01`. Indexed Triton
records `indexed_triton_domain_unsupported` for the corresponding indexed
resolution. Illegal inputs
raise and clear the previous successful execution observation. Direct CUDA/Triton calls
reject unsupported domains instead of silently falling back.

Historical ticket03 AdaLN-only SM86 evidence recorded an indexed matrix of
361 PASS (including CPU reference/metadata checks), small
memcheck/racecheck 107 PASS each, and related regression 417 PASS / 6 opt-in
workload SKIP. Two oversized-upstream tests require >5 GiB free GPU memory and
remain unexecuted on the 4 GiB target. This evidence applies only to its recorded
snapshot; affected current-source GPU checks need refresh. The dedicated indexed
Triton implementation has a separate local SM86 run on RTX 3050 Ti with
Torch 2.12.1+cu126, Triton 3.7.1 and bundled ptxas 12.8.93: indexed matrix
297 PASS, small sanitizer matrix 129 PASS per tool (memcheck/racecheck/initcheck/
synccheck, no reported errors or hazards), and shared regression 13 PASS.
FP32/BF16 subnormal checks and retained generated-code inspection are included;
this is local evidence, not CUDA/shared evidence inheritance or general target
qualification. Two large-upstream cases and twelve full-image workload nodes
remain unexecuted because the 4 GiB target cannot meet their memory requirements.
Full-extension and H100 qualification remain separate. Indexed strict stays closed.

### Strict mode and trace

Pass `strict=True` to enforce the applicable arithmetic policy. CPU Native and
NVIDIA CUDA/Triton retain the shared profile. Indexed strict additionally requires
an explicit `IndexedAdaLNQualification` from `rl_engine.runtime.adaln`, passed as
`indexed_qualification=scope` to `get_adaln_modulation_op`. Its default is `None`:
**indexed strict delivery qualification remains closed**. Implemented policy and
CPU policy tests do not establish hardware qualification.

The immutable scope names one backend, an explicit CUDA ordinal, GPU model and
compute capability, exact Torch/CUDA runtime versions, and the Triton version for
Triton. It expresses activation after external evidence review; it does not
verify a certificate or infer evidence. CUDA qualification cannot activate Triton.
Resolution checks backend/runtime/hidden eligibility without launching kernels;
execution rechecks actual device, GPU identity, H and the full indexed domain.
Inputs determine shared/indexed/framework-select01 profiles on every call; there
is no caller profile parameter. An indexed resolution may execute shared inputs
and vice versa, but strict never permits framework select01 or unsupported input.
Illegal input remains an input error and every failure clears earlier observations.

Acceptance uses this same explicit scope and public resolver/callable path, in a
separate source package for real Triton selection. After ticket06 accepts evidence,
activation uses the reviewed target values without modifying arithmetic, backend
selection or launch code. Source/binary/compiler identity, profiler and sanitizer
evidence stay in external manifests. The scope does not attest those properties;
a changed core path requires new affected evidence. No scope is installed globally
or enabled by an environment variable.

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

Indexed framework fallback checks only finite, nonnegative eps. Dedicated indexed
CUDA/Triton use the shared FP32 eps domain. Their per-sample/bank/channel dshift,
dscale and dgate start at FP32 +0 and fold selected tokens in ascending order;
each dscale product rounds to FP32 before addition. Gate copies original bits.
This is the first `adaln-indexed-fp32-tree-v1` arithmetic choice, independently
subject to acceptance, not a permanent restriction on future indexed profiles.

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
`IndexedAdaLNFixedReference` is a separate CPU FP32/BF16 callable for indexed byte
tests, with its own H tree and explicit VJP. It is not the framework fallback.
Indexed reference literals check sample/bank routing, absent versus explicit zero
dy, frozen leaves, cancellation order, separate product/add rounding, empty-bank
+0 isolation with NaN/Inf, gate payload copies and subnormal contributions.
The accuracy gold uses an independent selected-token affine VJP while retaining
CPU LayerNorm/autograd normalization; it is not a fixed-order byte oracle.
The same literals exercise the public CUDA callable; BF16 tests add final-output
rounding, ties-to-even and raw16 gate payloads.
All0 tests compare indexed bank0 dshift/dscale with shared bytes, check bank1 +0
separately and retain independent gold comparisons; shared/indexed dgate are not
equated. CUDA and public-dispatch tests require source-specific GPU evidence.
The local Triton run includes the new BF16 minimum-subnormal parameter-gradient
cases for one/three tokens and gate-only/both upstreams. Ticket 03 still has two
unexecuted oversized CUDA upstream cases; indexed strict remains closed.
The CLI records trace snapshots without automatically constructing C1
`BackendProvenance`; its accuracy report alone is not a WS1 system gate.

## Tests and Benchmark

```bash
python -m pytest tests/ops/norm/test_adaln_modulation.py \
  tests/reference/test_adaln_indexed.py tests/ops/norm/test_adaln_indexed.py \
  tests/runtime/test_adaln_indexed.py \
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

Shared CUDA and Triton require at most 2^31-1 elements in each input (`x` and
`modulation`) for int32 element addressing. Python rejects larger inputs before
contiguous copies or output/partial allocation. The CUDA extension additionally
requires B <= 65535 and repeats its bounds checks in C++ before launch.

Direct backend calls reject unsupported domains. Ordinary dispatch rechecks
actual inputs and uses a real Native fallback when shared device/geometry is
unsupported, preserving the shared validation and arithmetic contract. Its trace
records `shared_cuda_domain_unsupported` or `shared_triton_domain_unsupported`.
Strict rejects this fallback. Resolution uses the metadata available at that point.

Registry dispatch uses default launch settings. Direct constructors accept
CUDA `threads={128,256,512}` (default 256), Triton `num_warps={4,8}` (default 4)
and `reduction_tile={64,128,256}` (default 128).

These settings alter scheduling, not the declared reduction order. GPU support
and numerical acceptance must be checked on the target hardware; CPU tests
alone do not establish CUDA/Triton acceptance.

Indexed NVIDIA dispatch tries CUDA symbols, then Triton, then framework. To test
actual Triton selection while CUDA remains the default, use a separate source
package without `_C.so`, disable editable import hooks, and assert the extension
is absent before resolving. Keep the current binary in place. Successful indexed
Triton execution reports `actual_backend=triton`, indexed arithmetic profile,
`fallback=false` and `fallback_reason=None`; failed calls clear old observations.

Indexed Triton uses a separate row kernel and selected-bank FP32 fold. It reuses
the shared H statistics helpers without changing shared arithmetic or validation.
Both kernels set `enable_fp_fusion=False` and `enable_reflect_ftz=False`. Gate
copies use original-width integer pointers. Generated code and target subnormal
behavior require separate checks; flags alone are not evidence. Local SM86
inspection found RN sqrt/div lowering uses internal FTZ/FMA with correction paths,
and single-token BF16 gate addition may become FMA(v,1,+0). The latter is an
equivalent final-result specialization, not multi-token low-precision accumulation.
Do not claim the entire SASS is free of FTZ/FMA. No compiler/runtime
beyond the recorded acceptance environment is qualified. Unsupported geometry,
eps, addressing and non-NVIDIA devices reject directly or explicitly fall back
through ordinary dispatch. Indexed strict remains default-closed; its explicit
activation path is implemented for acceptance and later reviewed delivery.

Indexed B/S/H/eps, broadcast mode, forward/backward, upstream presence and launch
parameters specialize kernels; S is `tl.constexpr`, so new sequence lengths need
separate variants and identical calls may reuse cache entries. No runtime-S
optimization or launch-annotation performance gain is claimed.

Triton currently specializes all three shared kernels on S. A new S can require
a new compiled variant; repeated calls can reuse the cached variant. Removing
`tl.constexpr` can reduce variants, but runtime integer specialization can still
distinguish values such as 1 and alignment classes. Runtime division/remainder
and loop-bound costs need target-GPU measurements before changing this choice.
