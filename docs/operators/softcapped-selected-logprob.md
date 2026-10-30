# Softcapped selected logprob

A shared operator that applies softcap, normalizes over the vocabulary, and
returns one selected-token log-probability per row. Introduced for Gemma
([issue #415](https://github.com/RL-Align/RL-Kernel/issues/415)), it accepts a
per-call softcap so other models can use the same operation.

For logits $x$, selected token IDs $t$, and softcap $c$:

$$
s_{ij}=c\tanh(x_{ij}/c),\qquad
\ell_i=s_{i,t_i}-\log\sum_j e^{s_{ij}}.
$$

With upstream gradient $g_i=\partial L/\partial\ell_i$ and softmax probabilities
$p_{ij}=e^{s_{ij}}/\sum_k e^{s_{ik}}$:

$$
\frac{\partial L}{\partial x_{ij}}
=g_i\bigl(\mathbf{1}_{j=t_i}-p_{ij}\bigr)\bigl(1-\tanh^2(x_{ij}/c)\bigr).
$$

## Interface and precision

| Direction | Value | Shape | Dtype / requirement |
| --- | --- | --- | --- |
| Forward input | `logits` | `[M, V]` | FP16 / BF16 / FP32; finite values; `M >= 0`, `V > 0` |
| Forward input | `token_ids` | `[M]` | int64; same device as logits; values in `[0, V)` |
| Forward argument | `softcap` | Scalar | Required keyword argument; positive real number; no gradient |
| Forward output | Selected logprob | `[M]` | FP32 |
| Backward input | Upstream gradient | `[M]` | FP16 / BF16 / FP32; computed in FP32 |
| Backward output | Logits gradient | `[M, V]` | Same dtype as logits |

All intermediate arithmetic and reductions use FP32. Triton saves `row_max`
and `log_sum_exp_shifted`, each FP32 `[M]`, for backward. Inputs and saved
statistics are not modified. Noncontiguous inputs/upstreams are accepted and
copied to contiguous storage. Autograd supports first-order gradients.
There is no ignore-index or token-masking argument.

`softcap` belongs to each forward call, not the Op's configuration. Autograd
saves that call's value, so another call cannot change an outstanding graph.
Both public Op interfaces and internal functions require an explicit cap.
Omitting `softcap` raises `TypeError`; callers choose the value for their model.

Both implementations retain the same accepted cap/vocabulary domain:

- `c` is finite and at least `torch.finfo(torch.float32).tiny`.
- `exp(-c) >= 2 * float32.tiny`.
- `V * exp(c) <= float32.max / 2`.

Validation evaluates these bounds in log space using scalar/shape metadata.
For example, `30` and `50` are accepted at `V=262144`; `100` is not. These
conservative bounds preserve the original interface. Maximum shifting does
not require all of them; extending the domain is a separate API change.

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("softcapped_selected_logprob", device=logits.device)
selected_logprob = op(logits, token_ids, softcap=30.0)
selected_logprob_cap50 = op(logits, token_ids, softcap=50.0)
```

For direct reference/candidate calls:

```python
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)
from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    SoftcappedLogprobStrategy,
    TritonSoftcappedSelectedLogprobOp,
)

reference = NativeSoftcappedSelectedLogprobOp()
auto = TritonSoftcappedSelectedLogprobOp()
row = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW)
expected = reference(logits, token_ids, softcap=50.0)
actual = auto(logits, token_ids, softcap=50.0)
```

The registry prefers Triton on CUDA/ROCm and can fall back to native PyTorch
when the backend cannot load. Tests and benchmarks instantiate Triton directly
so they cannot silently validate or time a fallback. The Triton wrapper accepts
`cuda`, `hip`, `xpu` and `musa`; accepted device types do not establish numerical
or performance qualification. ROCm and other backends require their own runs.

## Reference and Triton arithmetic

Both PyTorch references live in
`rl_engine/reference/logprob/softcapped_selected_logprob.py`, with separate entry points:

| Entry point | Purpose |
| --- | --- |
| `NativeSoftcappedSelectedLogprobOp(nn.Module)` | Independent accuracy and performance baseline, with native autograd |
| `ordered_forward` / `ordered_backward` | Explicit arithmetic-path checks and inspectable intermediate stages; no public Op or autograd graph |

The independent numerical reference uses native PyTorch and autograd:

1. Convert logits to FP32 and divide by an FP32 tensor containing `softcap` on
   the input device. This avoids the CUDA Python-scalar division path's rounded
   reciprocal multiplication.
2. Compute `softcap * torch.tanh(scaled_logits)` in FP32.
3. Apply `torch.log_softmax(..., dim=-1)` and gather one token per row.

Triton uses `tl.div_rn` and `libdevice.tanh` for softcap, and `libdevice.exp` for
both forward exponentials and backward probabilities. Forward first finds the
maximum FP32 logit, then applies softcap once to obtain the whole-row score
maximum `m`. For a positive cap, `max(softcap(x)) = softcap(max(x))` mathematically.
This removes per-element division/tanh from the maximum pass; the second pass
still loads logits and computes each shifted exponential:

```python
sum_exp_shifted = sum(exp(scores - m))
selected_logprob = (selected_score - m) - log(sum_exp_shifted)
```

The maximum is shared by all tiles in that row. The final two FP32 subtractions
remain separate; the kernel does not reconstruct and round an unshifted
`m + log(sum_exp_shifted)`. Maximum shifting controls the exponentials but does
not remove floating-point rounding error. Both forward log-sum paths explicitly
use `libdevice.log`.

ROW and ROW_PIPELINED scan raw logits for their maximum. PARALLEL writes raw
tile maxima, merges them, and applies the same scalar softcap arithmetic once
per row. All paths still save the **softcapped** maximum for backward and
canonicalize zero to positive zero. Regression tests compare this maximum
against the original per-element-softcap maximum, including tails, ties,
saturation and signed zeros; numerical qualification is still backend-specific.

### Elementary functions and rounding

The numerical-path audit checks Triton 3.7.1 and PyTorch 2.13.0 CUDA sources.
The explicit library choices do not make transcendental functions exact, nor
do they promise identical library results on different GPU backends.

| Operation | Triton path | Independent PyTorch path |
| --- | --- | --- |
| Division by the cap | `tl.div_rn`, with FP32 operands | `torch.div` with a device FP32 tensor divisor; avoids Python-scalar reciprocal multiplication |
| Softcap nonlinearity | `libdevice.tanh` | Native `torch.tanh` |
| Exponentials, including backward probabilities | `libdevice.exp` | Native exponentials within `log_softmax` and its autograd backward |
| Logarithm of the shifted sum | `libdevice.log` | Native logarithm within `log_softmax` |
| Summation | Declared tile tree and ascending tile fold | Native `log_softmax` reduction; ordered PyTorch is a separate path reference |
| Multiply/add rounding | `enable_fp_fusion=False` preserves separate rounding points | Eager operations and native kernel/autograd arithmetic; not a bitwise oracle for Triton |
| Dtype conversion | FP32 inputs/intermediates/statistics; gradient cast at final store | FP32 reference input, output and autograd gradient for accuracy checks |

NVIDIA bindings select `__nv_expf`, `__nv_logf`, and `__nv_tanhf`; AMD bindings
select their OCML counterparts. These are standard device-library functions,
not the `fast_*` variants. The previous `tl.exp` NVIDIA path explicitly uses
an approximate base-two exponential. `tl.log` lowering depends on the compiler
and backend; the explicit library call makes the intended choice clear without
assuming every compiler version lowers `tl.log` to the same approximation.
See the [Triton NVIDIA bindings](https://github.com/triton-lang/triton/blob/v3.7.1/third_party/nvidia/language/cuda/libdevice.py),
[AMD bindings](https://github.com/triton-lang/triton/blob/v3.7.1/third_party/amd/language/hip/libdevice.py),
and [NVIDIA elementwise lowering](https://github.com/triton-lang/triton/blob/v3.7.1/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp).

`_math_launch_options` also preserves subnormals: CUDA uses
`enable_reflect_ftz=False`, while ROCm uses `allow_flush_denorm=False`.
Disabling FP fusion alone does not disable CUDA libdevice's separate FTZ
(flush-to-zero) setting. A probability near `1e-39` can still contribute a
gradient near `1e-3` when the finite upstream is `1e36`; it must not be discarded
before multiplication. Backend-specific options are not passed to other backends.
See the [CUDA compiler options](https://github.com/triton-lang/triton/blob/v3.7.1/third_party/nvidia/backend/compiler.py)
and [ROCm compiler options](https://github.com/triton-lang/triton/blob/v3.7.1/third_party/amd/backend/compiler.py).
The CUDA path requires a compiler exposing `enable_reflect_ftz` (verified in
3.7.1). Older versions such as 3.0/3.4 lack this control; the launcher raises a
clear upgrade error rather than relying on an unsupported flag being honored.

The native reference remains independent. Its tensor divisor avoids the
[scalar-division optimization](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu),
but [native log-softmax](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/SoftMax.cu)
still has its own reduction and rounding. Autograd also retains multiplication
and division by the cap, whereas the analytic Triton derivative cancels them.
FP32 rounding, tanh saturation and eventual underflow remain possible. The
operator contains no matrix multiplication, so TF32 settings do not control
this arithmetic. GPU accuracy, cross-strategy consistency and performance must
be rechecked after changing library functions or compiler options.

Shared input and arithmetic configuration lives in
`rl_engine/contracts/operators/softcapped_selected_logprob.py`:

- `SUPPORTED_DTYPES` supplies both public implementations; the cap is always
  supplied by the caller, not by shared configuration.
- `TRITON_SUPPORTED_DEVICES` records device types accepted by Triton; the
  PyTorch references remain usable on CPU.
- `REDUCTION_CONFIG` supplies the tile width, eight-value sequential groups and
  four-section tree to Triton and the fixed-order PyTorch reference. Group size
  and section count define the arithmetic; they are not per-strategy tuning knobs.
  Construction rejects nonpositive and non-power-of-two tile widths.
  Launchers pass group size and section count explicitly through the forward
  kernels to `sum_tile`; this operator does not rely on the helper's defaults.
- Strategy types, device keys, tuning ranges and cached selection are defined
  at the start of `backends/shared/triton/logprob/softcapped_selected_logprob.py`.
  Input validation, kernels, launchers and the public autograd Op follow in that order.

The contract contains no kernels or tensor computation. The native PyTorch
`log_softmax`/autograd reference remains independent of the ordered reduction;
the accuracy tests do not reuse the fixed-order helper.

The configured arithmetic tile currently has 1024 elements. Masked lanes contribute negative
infinity to maxima and positive zero to sums. Zero maxima are canonicalized to
positive zero. `ordered_tile_reduce` with `COMBINE_FN=reduce_add` declares these addition pairs:

1. Form 128 groups of eight adjacent values. Add each group left to right,
   starting from the first value.
2. Form four sections of 32 group totals. Pair lower and upper halves at
   distances `16, 8, 4, 2, 1`.
3. Merge section totals as `(s0 + s2) + (s1 + s3)`.
4. Starting from positive zero, add tile totals in ascending vocabulary order.

Every addition rounds in FP32. Each tree step explicitly selects two operands
and calls `reduce_add`. The groups describe arithmetic, not a physical warp layout. Forward
uses four warps with FP fusion disabled; `V` and `softcap` are compile-time
constants. Changing a cap can compile a separate specialization.

`ordered_tile_reduce` accepts any positive power-of-two vector width within
Triton's resource limits. The width determines the tree at compile time:
2048 adds a distance-32 stage before `16, 8, 4, 2, 1`; 1024 retains the original
addition pairs. Widths below eight use one shorter sequential group. Widths
8 and 16 yield one or two group totals, returned directly or added once. Width
32 has exactly four groups and only needs the final section merge.

The generic `ordered_tile_reduce` does not impose a dtype or cast its input; the
supplied combiner determines its arithmetic dtype behavior. This operator uses
the `sum_tile` convenience wrapper with explicit FP32 computation, preserving
its intermediate precision and public dtype contract.

The operator imports `sum_tile` and `sum_partials` from
the single `rl_engine.backends.shared.triton.reduction` package entry point. Their
[shared contract and reuse guide](../architecture/ordered-reduction.md) are
independent of logprob:

| Helper | Responsibility |
| --- | --- |
| `_softcapped_exp_tile_sum` (operator-local) | Load one tile, compute FP32 softcap and shifted exp, zero padded lanes, then call `sum_tile` |
| `sum_tile` (shared) | Select addition and this operator's explicit FP32 compute dtype for `ordered_tile_reduce`; it performs no softcap or exp |
| `sum_partials` (shared) | Fold parallel tile sums in ascending order from zero; infer FP32 from their storage |
| `_store_softcapped_selected_logprob` (operator-local) | Compute the selected logprob and save the shifted log-sum, using identical arithmetic in both schedules |

The local tile helper receives `BLOCK_V`, `SOFTCAP`, `MAX_GROUP_SIZE` and
`SECTION_COUNT` explicitly, along with the row pointer, tile index and runtime
row maximum. It passes the resulting vector to `sum_tile` with explicit FP32
precision and the shared arithmetic configuration. ROW and ROW_PIPELINED own
the ascending tile loop and
accumulate from positive zero; PARALLEL stores each partial in its tile slot
and calls `sum_partials`. The helper supplies addition and the same positive-zero seed
once for the whole row, never for each partial. It adds partials left to right rather
than treating them as another tile to be reduced.

The finalization helper receives row-relative pointers, the row maximum and
the completed exponential sum. It computes the shifted log-sum with
`libdevice.log`, loads the selected token's score and stores
`(selected_softcapped - row_max) - log_sum_exp_shifted` with the original
parentheses. ROW saves its maximum separately; PARALLEL reuses the maximum
already saved by its earlier merge kernel. No additional kernel launch is needed.

Softcap/exp and reduction stay in the same kernel, with no full exponentials
buffer or additional launch. The shared reduction package has no map/context
interface. Padded lanes are cleared **after** exp because a zero logit does not
produce a zero exponential.

The parallel sum stage has a producer kernel and a merge kernel; a JIT helper does
not launch or synchronize other programs. This refactor keeps the existing
left-to-right fold between tiles. It does not introduce a balanced tree across
tile totals or claim that a fixed order eliminates rounding error. Changing
that fold requires updating both schedules and the ordered reference together.

The PyTorch-only `ordered_reduce` diagnostic collector in
`rl_engine/reference/reduction/ordered_reduce.py` calls the same-named
`ordered_reduce_partial` and `ordered_reduce_merge` references, then returns named
`partials` and `total` fields. The operator's `tile_size` is passed as `block_size`,
matching Triton's `BLOCK_SIZE`.
`ordered_exp_sum` passes inspectable, already-computed exponentials directly.
The independent native Op continues to use `log_softmax` and autograd. Tests
cover FP32 conversion, tail masking, serial/pipelined/parallel equality and
partial-buffer integrity. Embedded-helper tests also compute elementwise
transforms in the caller before reducing their local tiles.
Generic reduction checks live in `tests/reference/reduction` and
`tests/backends/shared/triton/reduction`; operator-stage, accuracy and
training/inference checks remain under the logprob test paths.

The PyTorch fixed-order helper accepts the same widths; its full-row reference
takes an explicit `tile_size` when testing another configuration. Tests cover
powers of two from 1 through 8192, nonassociative inputs, padded tails, the old
1024 tree and GPU execution with four/eight warps. Cross-strategy consistency
is required **at the same configured width**. Changing `REDUCTION_CONFIG.tile_size` changes both
the row's addition grouping and the policy's tile counts; it requires fresh
numerical and performance qualification, not equality with the old width.
Change this configuration before importing the implementations, then restart
the process for testing; reference tile-width defaults and the backward launcher's
default block width are bound at import time.

| Strategy | Execution | Temporary global storage |
| --- | --- | --- |
| `ROW` | One program per row: scan raw logits for the maximum, softcap once, then scan for shifted sums | No tile scratch |
| `ROW_PIPELINED` | Same scans and arithmetic, with a loop pipeline depth selected by device, dtype and shape range | No tile scratch |
| `PARALLEL` | Four kernels: raw-logit tile maxima, maximum merge + scalar softcap, shifted tile sums, sum merge | FP32 `[M, ceil(V / 1024)]`, reused between maxima and sums |

Parallel kernels launch on the input GPU's same current stream. Dependent
stages execute in order without host synchronization. Strategy changes preserve
the tile tree and ascending tile fold, including when automatic selection
changes after a batch split.

All strategies save two FP32 values per row (`8 * M` bytes) and share one
backward kernel. Each backward program owns a disjoint row/vocabulary tile:
it recomputes scores, evaluates `libdevice.exp((score - row_max) - log_sum_exp_shifted)`,
applies the analytic derivative, and casts only the final gradient store.
Backward contains no reduction. Neither direction saves a full score or
probability matrix. Direct backward callers must supply the statistics and cap
from the same forward invocation.

`ROW_PIPELINE_CONFIGS` in the Triton operator module maps `ForwardConfigKey` ranges to
`RowPipelineConfig(pipeline_stages=...)`. The key uses the input GPU model/backend,
dtype, inclusive row range and vocabulary tile-count range. The pipelined launcher
resolves parameters for each call, so different shapes can select different depths
even when using the same public Op instance. Concrete-device matches precede shared
defaults; overlapping matches at the same scope are rejected.

The pipeline table is deliberately empty pending server benchmarks. Unmatched
inputs, including shapes beyond measured ranges, use `DEFAULT_ROW_PIPELINE_CONFIG`
with the provisional depth of 3. This is a candidate, not a measured optimum. Compare
depths with the benchmark's `--compare-strategies --pipeline-stages` sweep, check
output/statistic/gradient consistency, then record measured ranges in `ROW_PIPELINE_CONFIGS`.
Changes to this table or its fallback require
`select_softcapped_logprob_pipeline_config.cache_clear()`. The selector caches 1024
exact `(device, dtype, M, V)` combinations and ignores gradient mode and input values.
Pipeline tuning does not alter the separate automatic strategy policy or the shared
reduction tree. Tests use synthetic depth-switching ranges to check batch-split
invariance; those ranges are not production tuning recommendations.

## Automatic strategy selection

`rl_engine/backends/shared/triton/logprob/softcapped_selected_logprob.py`
keeps tuning policy and execution together, organized in dependency order:

1. Strategy/device types, shape-range keys, defaults and the two tuning maps
   (`FORWARD_STRATEGY_CONFIGS` and `ROW_PIPELINE_CONFIGS`).
2. Cached strategy/pipeline selectors and device lookup.
3. Input validation and numerical compiler options.
4. Shared arithmetic helpers, ROW/parallel forward kernels, then backward.
5. Launcher interfaces, forward/backward launchers and the strategy-to-launcher map.
6. Autograd state management and the public Op.

Shared arithmetic configuration remains in contracts. CPU-only policy tests
execute the production host definitions without importing Triton kernels.

Importing the operator requires Triton but does not query hardware. Device lookup
happens during automatic strategy selection or ROW_PIPELINED parameter selection.
`_FORWARD_LAUNCHERS`, defined after the
launchers, binds strategies to their implementations. Public operator registration
and the native PyTorch reference do not change.

Each `ForwardConfigKey`
contains a device variant, logits dtype, inclusive row range and inclusive
vocabulary tile-count range. `max_tiles=None` leaves the upper range open.

- `SoftcappedLogprobDefaultDevice()` carries no hardware fields and shares a policy.
- `SoftcappedLogprobDevice(backend=..., gpu_name=...)` identifies a concrete GPU
  model. ROCm and CUDA remain distinct; the ordinal is not part of the model key.

Selection checks concrete-device rules, then shared defaults, then the ROW
fallback. Overlapping matches at the same scope raise an error. Hardware-query
errors propagate. Explicit strategies bypass automatic strategy matching;
ROW_PIPELINED still selects its pipeline parameters from input metadata.

The current shared rules use the same range for all three input dtypes:

| Rows M | Vocabulary tile count `ceil(V / 1024)` | Strategy |
| --- | --- | --- |
| 1–4096 | 32 or more | `PARALLEL` |
| All other valid shapes | Any | `ROW` |

These are scheduling bounds, not input restrictions. The 32-tile boundary
starts at `V=31745`; the input is not rounded to a power of two or padded by
selection. `ROW_PIPELINED` is explicit-only.

The selector caches 1024 exact `(device, dtype, M, V)` combinations. GPU names
are cached by backend and resolved device index. Selection does not inspect
input values, gradient mode or runtime benchmark files. Clear
`select_softcapped_logprob_strategy.cache_clear()` after changing the map or
fallback at runtime.

The initial ranges came from earlier H100 80GB HBM3 measurements at cap 30.
Those measurements preceded the current maximum-shifted kernels and explicit
addition tree. They explain the starting policy, but **do not validate current
speedups or optimal dispatch boundaries**. Re-run the benchmark before updating
performance claims or tuning another device.

## Tests

Regular acceptance separates consistency from independent numerical accuracy.
Cases that share input preparation collect separate pytest items for the two
layers, so an accuracy failure cannot suppress a consistency check.

| File | Responsibility |
| --- | --- |
| `tests/ops/logprob/test_softcapped_selected_logprob.py` | Public interface, autograd, cap boundaries and independent FP32 accuracy; retained precision regressions |
| `tests/ops/logprob/test_softcapped_selected_logprob_invariance.py` | Raw-bit equality across strategies, batch splits, permutations, layouts and inference modes |
| `tests/ops/logprob/test_softcapped_selected_logprob_ordered.py` | Explicit PyTorch tile tree and staged arithmetic conformance |
| `tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_launch.py` | Two-GPU device restoration, scratch/output bounds and 64-bit row offsets |
| `tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_config.py` | Range selection, ambiguity, device lookup and caches |
| Runtime, validation-harness and benchmark tests | Registry binding, shared validation integration, timing boundaries and report generation |

The fixed-order functions in `rl_engine/reference/logprob/softcapped_selected_logprob.py`
provide a pure PyTorch specification
of the fixed addition tree and rounding points. Its backward is explicit.
It is separate from the independent Native FP32/autograd accuracy reference and
is never the benchmark baseline. Elementary functions can differ between
PyTorch and Triton: small test-only GPU probes isolate the log/backward-exp
boundaries for arithmetic-conformance checks. Independent accuracy checks run
the complete public operation without these probes.

FP32 output accuracy uses `atol=1e-5, rtol=0`. Gradient accuracy uses the
output dtype's contract row: FP32 `1e-5`, FP16 `5e-3`, BF16 `5e-2`, with zero
relative tolerance. The reference receives the same quantized inputs promoted
to FP32 and retains FP32 gradients. It is not rounded to the candidate dtype
before comparison. The shared harness retains its contract selection by case
dtype; see the [numerical contract](../contracts/ws1-numerical-precision-standard.md).

The retained regressions include 14 caps from `1e-6` through `70`, supported
endpoints, signed zeros, saturation, masked tails, saved-state reuse, and the
five wide-input failures at `[1024, 262144]` with caps 64/70, and subnormal
probabilities amplified by large finite upstream gradients. Those wide cases
retain GPU generation, seed 415 and the original full draw order.

With the development environment installed, run:

```bash
python -m pytest tests/ops/logprob/test_softcapped_selected_logprob*.py \
  tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_launch.py \
  -m softcap_consistency -q -ra
python -m pytest tests/ops/logprob/test_softcapped_selected_logprob*.py \
  tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_launch.py \
  -m softcap_accuracy -q -ra
python -m pytest \
  tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_config.py \
  tests/runtime/test_softcapped_selected_logprob_registry.py \
  tests/validation/operators/test_softcapped_selected_logprob_harness.py \
  tests/benchmarks/test_softcapped_selected_logprob_benchmark.py -q -ra
```

### Optional FP64 diagnosis

FP64 evaluates the original formula on the same already-quantized inputs.
It remains an independent diagnostic, outside the default `tests/` collection:

```bash
python -m tools.validation.operators.check_softcapped_math -q -ra \
  --junitxml=reports/softcapped-fp64.xml
```

This command retains the previous extra targets: forward absolute error `1e-5`,
and gradient absolute errors `1e-5`, `5e-3`, `5e-2` for FP32/FP16/BF16.
Exceeding them still returns failure; no tolerance was relaxed and no case was
marked `xfail`. These goals do not redefine WS1's FP32-reference contract.
The exact FP32 division check remains in regular acceptance, even though it
uses FP64 to construct the expected rounded quotient.

The saved RTX A5000 report `softcapped-ordered-20261017-084233-NkJgr0` recorded
1376 consistency passes and 1580 FP32 accuracy/API passes, with 48 two-GPU skips
in each layer. FP64 recorded 1414 passes and 14 failures, all forward-output
checks. These are **pre-cleanup GPU results**, not a GPU rerun of this layout.
The optional FP64 command now excludes 30 division checks moved to regular
acceptance. CPU results can differ with the host's native math implementation.

The 14 failures comprise two large-cap inputs repeated over six implementations
(BF16/cap70 and FP32/cap64), one ordered-reference case and one wide CPU Native
case. Saved GPU stages show cumulative softcap and final-subtraction rounding
in the two main cases. Their neighboring FP32 outputs are about `1.526e-5`
apart: improving agreement with FP64 can exceed the `1e-5` comparison against
the current Native FP32 result. Keep this evidence separate from consistency
and FP32 acceptance; passing one layer does not establish another.

## Benchmark and performance evidence

The benchmark compares public `NativeSoftcappedSelectedLogprobOp` with automatic
`TritonSoftcappedSelectedLogprobOp` on the same CUDA/ROCm GPU. It maintains no
second PyTorch baseline and imports no other operator's private helpers.

```bash
python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py --softcap 30 --list-cases
python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --softcap 30 --dtypes bf16 --shapes 3x1025 --warmup 1 --repeat 2 \
  --output-dir reports/softcapped-selected-logprob/smoke
python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --softcap 30 \
  --output-dir reports/softcapped-selected-logprob
```

`--softcap` is required. The other defaults cover five shapes, three dtypes and
forward/backward/forward+backward:
15 inputs and 45 comparisons. `--softcap`, `--shapes`, `--dtypes` and `--modes`
can extend coverage. Accuracy and raw-bit strategy/training-inference checks
run before timing, even when only forward is timed. A failed gate stops the
case; FP64 diagnosis is separate.

Timing uses `PerformanceProfiler` accelerator events, 10 warmups and 50 repeats
by default. No `torch.compile`, CUDA Graph capture or explicit cache eviction
is used. Inputs are reused and the selector cache is warm. Compilation, input
preparation and correctness checks are outside timing; public wrapper allocation
and dispatch are included. Native divisor creation is included in forward.
Forward uses `no_grad`; backward reuses a retained graph; forward+backward builds
a fresh graph per call. Extra peak allocation excludes inputs/prebuilt graphs.
Small cases can be dominated by host dispatch gaps.

`--compare-strategies` times Native, AUTO, ROW, PARALLEL and ROW_PIPELINED at
`--pipeline-stages 1 2 3 4 5 6 8` by default. These are 11 providers, including
seven independently named `row_pipelined_sN` candidates. `--rounds` controls the
rotated timing rounds (default 4); setting it to the provider count places each
provider in every position once. Forward and forward+backward compare all
candidates; backward-only compares Native with AUTO because all strategies
share the same backward kernel. Reports retain each round's timing position,
standard deviation and memory, the median of round medians, round max/min, and
AUTO latency divided by the fastest explicit candidate. Near ties and noisy
rounds require repeated measurements before changing a dispatch range.

The standalone, single-threaded benchmark temporarily replaces the in-memory
pipeline table/fallback for each candidate, then restores both and clears the
selector cache, including on failure. This happens **outside timing**. Timed
calls still use the public Op, its normal selector, allocations and autograd.
Warmup fills the cache. AUTO always uses the original policy. Production source
and arithmetic settings are unchanged; no benchmark result edits the Map.

Every candidate must match AUTO's output, saved maximum, saved shifted log-sum,
and gradient bytes. Training/no_grad/inference_mode outputs are also compared
for each candidate. AUTO is checked against independent Native FP32/autograd;
byte equality then transfers those same accuracy bounds to the other candidates.
No failing candidate is timed. `row_pipeline_stages` records the unmodified
shape-selected default; each timing candidate's actual override is recorded
separately. The plan distinguishes input/mode comparisons (`measurement_count`)
from provider measurements (`provider_measurement_count`); neither counts
warmups or individual repetitions. For range validation:

```bash
python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --softcap 30 --compare-strategies --pipeline-stages 1 2 3 4 5 6 8 \
  --modes forward --rounds 4 --warmup 10 --repeat 50 \
  --shapes 16x31744 16x31745 16x32768 4096x32768 4097x32768 \
    16x65537 1024x202048 4096x202048 4097x202048 16x2097153 \
  --output-dir reports/softcapped-selected-logprob-ranges
```

JSON/Markdown reports include environment, source commit/dirty status, arithmetic
configuration, cap, chosen strategy, median latency,
standard deviation, speedup, extra allocation and accuracy errors. Reports are
saved after each completed case and remain `complete: false` on interruption.
Speedup is Native latency divided by Triton latency. Default reports go to
`reports/softcapped-selected-logprob/`; attach results to the PR rather than
committing generated output.

[PR #463](https://github.com/RL-Align/RL-Kernel/pull/463) contains earlier tuning
and benchmark attachments. Old unshifted/scalar-division timings must not be
presented as current speedups. Current-kernel performance regression and ROCm
qualification remain pending.
