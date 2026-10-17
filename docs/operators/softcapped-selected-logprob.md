# Softcapped selected logprob

A shared logprob operator that applies a configurable softcap and returns one
selected-token log-probability per row. It was introduced for Gemma
([issue #415](https://github.com/RL-Align/RL-Kernel/issues/415)); the interface
has no model-topology dependency. The scalar `softcap` is denoted by $c$ below
and defaults to `30.0`:

$$
s_{ij}=c\tanh(x_{ij}/c),\qquad
\ell_i=s_{i,t_i}-\log\sum_j e^{s_{ij}}.
$$

With upstream gradient $g_i=\partial L/\partial\ell_i$ and
$p_{ij}=\exp(s_{ij}-\log\sum_k e^{s_{ik}})$:

$$
\frac{\partial L}{\partial x_{ij}}
=g_i\bigl(\mathbf{1}_{j=t_i}-p_{ij}\bigr)\bigl(1-\tanh^2(x_{ij}/c)\bigr).
$$

## Interface and precision

| Value | Shape | Dtype |
| --- | --- | --- |
| `logits` | `[M, V]`, `M >= 0`, `V > 0` | FP16 / BF16 / FP32 |
| `token_ids` | `[M]`, values in `[0, V)` | int64, same device as logits |
| `softcap` forward keyword argument | Scalar; default `30.0` | Positive real number; no gradient |
| Output | `[M]` | FP32 |
| Upstream gradient accepted by the backward launcher | `[M]` | FP16 / BF16 / FP32; computed in FP32 |
| Input gradient | `[M, V]` | Same dtype as logits |

Inputs must contain finite logits and valid indices. There is no ignore-index
or token-masking argument. Noncontiguous inputs and upstream gradients are
accepted; Triton copies them to contiguous storage. Inputs are not modified.
Autograd supports first-order gradients. Native PyTorch supplies its own autograd.

`softcap` is passed per call, independently of the strategy/tuning configuration.
The same Op can serve calls with different caps. Each autograd invocation keeps
its own cap for backward; the Op does not store or mutate this value.
It must be finite and at least `torch.finfo(torch.float32).tiny`. Native uses
FP32 `torch.log_softmax`; Triton explicitly subtracts the row maximum before
its exponential sum. Both keep the original accepted softcap/vocabulary domain:
they reject cap/vocabulary combinations unless
`exp(-c) >= 2 * float32.tiny` and `V * exp(c) <= float32.max / 2`.
These conservative bounds came from the original unshifted `exp -> sum -> log`
implementation. They are retained for interface compatibility, not because
the shifted calculation requires those same bounds.
The bounds are checked in log space, using only the scalar and shape metadata.
For example, both `30` and `50` are accepted at `V=262144`; `100` is not.
Expanding this interface range is outside the current experiment.

The PyTorch numerical reference creates an FP32 `softcap` tensor on the logits
device and uses `torch.div(logits.float(), divisor)`. This avoids the CUDA
Python-scalar division path, which multiplies by a precomputed rounded reciprocal
([PyTorch implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)).
The reference remains entirely native PyTorch with FP32 intermediates and native
autograd. After softcap, it computes `torch.log_softmax(softcapped, dim=-1)` and
gathers one token per row. For row maximum `m`, the normalized expression is
`(selected_score - m) - log(sum(exp(scores - m)))`; it does not first reconstruct
and round the unshifted log-sum. This is mathematically the same log-probability,
but its intermediate rounding differs from the earlier reference.
Triton retains `tl.div_rn`; it does not adopt reciprocal multiplication
to reproduce the old reference's rounding. WS1 accuracy compares with this FP32
reference. Independent FP64 mathematical tests have separate, explicitly stated
targets; they do not redefine the shared contract's reference or tolerances.
Native reductions can still differ with batching, so reference accuracy and
Triton's bitwise training/inference and batch invariance are separate checks.

Triton forward uses `libdevice.exp` with FP32 inputs and outputs.
On A6000, the fast `tl.exp` approximation caused a BF16, `softcap=64` case
to differ from Native FP32 by `1.52587890625e-5`, exceeding the `1e-5` forward
limit. Changing only these forward exponentials to `libdevice.exp` removed
that discrepancy in the focused experiment. The current candidate evaluates
`exp(score - row_max)` and finishes entirely in FP32. Division and tile-sum
order are unchanged. The saved logarithm uses `tl.log`; backward computes
`tl.exp((score - row_max) - log_sum_exp_shifted)`. The generic
`triton.language.extra.libdevice` import also supports ROCm, but accuracy and
performance need validation on each target backend.

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("softcapped_selected_logprob", device=logits.device)
selected_logprob = op(logits, token_ids)
selected_logprob_cap20 = op(logits, token_ids, softcap=20.0)
```

Omitting `softcap` uses `30.0`. Both registry-created and directly constructed
Ops accept another cap at call time. The native implementation function requires
an explicit `softcap`; its default is supplied only by the Op's `forward` method:

```python
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)
from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    TritonSoftcappedSelectedLogprobOp,
)

reference = NativeSoftcappedSelectedLogprobOp()
op = TritonSoftcappedSelectedLogprobOp()
expected = reference(logits, token_ids, softcap=50.0)
selected_logprob = op(logits, token_ids, softcap=50.0)
```

The registry prefers Triton on CUDA/ROCm, falling back to native PyTorch if the
backend cannot load. CPU/MUSA/NPU use native PyTorch. The Triton wrapper accepts
`cuda`, `hip`, `xpu` and `musa`, following `final_logit_softcap`; acceptance does
not establish validation on those devices. Performance tuning targets H100;
ROCm qualification remains separate. Tests and benchmarks instantiate Triton
directly so they cannot silently measure a native fallback.

## Triton implementation

Softcap, maxima, shifted exponentials, reductions, logarithms, final subtraction
and backward arithmetic use FP32. For each row, first compute its maximum
`m = max(scores)`, then evaluate:

```python
sum_exp_shifted = sum(exp(scores - m))
selected_logprob = (selected_score - m) - log(sum_exp_shifted)
```

The maximum is shared by every tile in that row. It does not change while
accumulating tile sums. The final expression subtracts the maximum first;
it never reconstructs and rounds `m + log(sum_exp_shifted)` before subtraction.
This removes large exponentials and avoids subtracting two large nearly equal
unshifted values, but does not eliminate FP32 rounding error.

Vocabulary tiles contain 1024 elements and use four warps, with FP fusion disabled. The
vocabulary width and softcap are compile-time constants; partial tiles are masked.
Different caps may compile separate kernels. Masked columns contribute negative
infinity to the maximum and zero to the sum. A zero maximum is canonicalized to
positive zero. `_ordered_tile_sum` declares the same addition pairs for every cap:

1. Split a tile into 128 groups of eight adjacent FP32 values. Accumulate each
   group left to right, starting from its first value.
2. Split those 128 subtotals into four sections of 32. In each section, add
   the lower half to the upper half at distances `16, 8, 4, 2, 1`.
3. Merge the four section totals as `(s0 + s2) + (s1 + s3)`.
4. Starting from FP32 positive zero, add tile totals in ascending vocabulary
   order. Pad the final tile with positive zeros.

Every addition rounds in FP32. The Triton helper spells out these pairs;
its remaining `tl.sum` operations each reduce exactly two operands. These
are logical groups, not assumptions about a backend's physical warp size.
This tree reproduced all 256 captured tile sums from the row-939 diagnostic
when replayed on identical FP32 exponentials. That replay verifies the
reference against one captured case; GPU compilation, strategy invariance
and performance of the newly explicit kernel tree still require regression.

- `ROW`: one program scans the entire row for its maximum, then scans again,
  reducing each shifted tile and accumulating its sum in ascending tile order.
- `ROW_PIPELINED`: the same two scans and arithmetic with a three-stage loop
  pipeline hint.
- `PARALLEL`: four kernels compute tile maxima, merge row maxima, compute
  shifted tile sums, and merge sums in the same order as ROW. The maxima and
  sums reuse one `4 * M * ceil(V / 1024)`-byte temporary buffer. All four launches
  use the same stream, so a stage finishes before the next consumes its output.

All paths save FP32 `row_max` and `log_sum_exp_shifted` per row (`8 * M` bytes) and use the same
backward kernel. Backward processes disjoint row/vocabulary tiles, reusing the
saved statistics without repeating the reduction. It casts gradients only on
store. Neither path materializes a full softcapped or probability tensor.
The cap bounds scores to `[-c, c]`; host validation checks the FP32 range above.
Autograd saves the cap used by each forward call, so later calls with a different
cap do not change that graph's backward. Direct backward-launcher callers must
pass the same cap used to compute both saved statistics.

ROW, ROW_PIPELINED and PARALLEL are designed to preserve the same arithmetic
order when changing strategy with batch size. The shifted candidate still
requires GPU validation of this property. Tests check
outputs, saved statistics and gradients, as well as forward equality with
and without gradient recording. This is operator-level coverage, not full-model
training/rollout, tensor-parallel or cross-device equivalence.

## Automatic strategy selection

The current policy retains earlier unshifted-kernel timing decisions. Maximum
scanning and the two extra PARALLEL launches change the cost; the shifted
implementation needs fresh benchmark comparisons before this policy can be
considered tuned for it.

```python
from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    SoftcappedLogprobStrategy,
    TritonSoftcappedSelectedLogprobOp,
)

auto_op = TritonSoftcappedSelectedLogprobOp()  # forward_impl=None
row_op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW)
parallel_op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.PARALLEL)
```

The static map and selector live alongside the shared Triton implementation in
`rl_engine/backends/shared/triton/logprob/softcapped_selected_logprob.py`.
`ForwardConfigKey` records device identity, dtype, inclusive row-count bounds and
inclusive **vocabulary tile-count bounds**. The selector computes
`n_tiles = (V + 1023) // 1024`; `max_tiles=None` means no upper bound. This describes
how many tiles the existing kernels process. It does **not** round the actual
vocabulary, pad the input, or change the arithmetic and reduction order.

Its `device` field uses two immutable variants, following the shared RMSNorm
configuration design. `SoftcappedLogprobDefaultDevice()` carries no metadata
and denotes the shared default. A concrete device is written as
`SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100 80GB HBM3")`.
It matches a backend and hardware model, not a device ordinal such as `cuda:0`.
ROCm uses `backend="rocm"`, even though PyTorch exposes it through `torch.cuda`.
There is no string sentinel or positional device tuple in the configuration.

All rule constructors name their fields explicitly:

```python
import torch

from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    ForwardConfigKey,
    SoftcappedLogprobDefaultDevice,
)

key = ForwardConfigKey(
    device=SoftcappedLogprobDefaultDevice(),
    dtype=torch.bfloat16,
    min_rows=1,
    max_rows=4096,
    min_tiles=32,
    max_tiles=None,
)
```

The generalized candidate has three entries, one for each supported dtype:

| Rows M | Vocabulary tiles | FP16 / BF16 / FP32 |
| --- | --- | --- |
| 1–4096 | 32 or more | PARALLEL |
| All remaining supported shapes, including M=0 | Any | ROW fallback |

32 tiles begin at `V=31745`: `V=31744` fills exactly 31 tiles. Gemma's
`V=262144` uses 256 tiles; `V=202048` uses 198 tiles and shares the same policy.
A new model does not need an exact vocabulary entry. Larger unseen vocabularies
also use PARALLEL within the configured row range; batches above 4096 use ROW.
These are launch-policy bounds, not input-size restrictions or proven optimal
hardware crossovers.

Lookup checks device-specific entries, then the default-device variant, then
`DEFAULT_FORWARD_STRATEGY` (ROW). Device rules can specify a finite upper tile
bound and override a default PARALLEL choice with ROW. Rules at the same scope
and dtype must not overlap in **both** row and tile ranges. Ambiguous matches
raise an error; device-query failures propagate. Other devices inherit the
H100-derived default until device-specific measurements justify overrides.

Selection is cached for 1024 exact metadata combinations. It does not depend on
softcap, gradient mode, input values or runtime timing. GPU names are cached by
backend and device index. After changing the map or final fallback at runtime,
call `select_softcapped_logprob_strategy.cache_clear()`. No benchmark reports
are loaded at runtime.

The rule was checked offline against existing H100 80GB HBM3 measurements at
`softcap=30.0`, PyTorch 2.13.0+cu130 and Triton 3.7.1. Replaying the choices over
837 eager and 27 CUDA Graph cases agrees with every conclusive ROW/PARALLEL
winner in those reports. Based on their recorded medians, the chosen explicit
implementation is at most 2.73% slower than the faster of ROW/PARALLEL in the
eager sweep. A separate historical policy-check dataset (39 eager / 12 graph
cases) gives maxima of 0% / 1.89%. Near ties can now choose PARALLEL instead of
retaining an exact-width exception.

This is an **offline policy check**, not a new GPU execution or a timing of the
new AUTO wrapper. Earlier correctness results apply to the tested kernels and
shapes. Newly interpolated widths, the 31/32-tile boundary, large-vocabulary
extrapolation and nondefault caps still require GPU regression and performance
checks. The compact comparison below covers those gaps before treating this
candidate policy as validated. Performance on other hardware is unverified.

The explicit `ROW_PIPELINED` experiment requests three loop stages with ROW
arithmetic. It is not selected by the automatic map. `ROW_ACCUMULATE` was removed:
its vector-before-reduction grouping can change saved statistics and gradients.
Every remaining selectable strategy is subject to the same byte-equality tests.

## Tests and benchmark

```bash
uv sync --extra dev
uv run --no-sync python -m pytest \
  tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_config.py \
  tests/runtime/test_softcapped_selected_logprob_registry.py \
  tests/validation/operators/test_softcapped_selected_logprob_harness.py \
  tests/benchmarks/test_softcapped_selected_logprob_benchmark.py -q -rs
uv run --no-sync python tools/validation/operators/check_operator.py --op softcapped_selected_logprob \
  --candidate triton --device cuda --dtype bf16 --batch 2 --seq 16 --vocab 262144

# Run these independently so one layer's failure cannot suppress another.
# Save each command's exit status as well as its log when running on a server.
uv run --no-sync python -m pytest \
  tests/ops/logprob/test_softcapped_selected_logprob*.py -m softcap_consistency -q -ra
uv run --no-sync python -m pytest \
  tests/ops/logprob/test_softcapped_selected_logprob*.py -m softcap_accuracy -q -ra
# Additional mathematical checks; known discrepancies remain ordinary failures.
uv run --no-sync python -m pytest \
  tests/ops/logprob/test_softcapped_selected_logprob*.py -m softcap_math -q -ra

# Inspect the benchmark plan without a GPU.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py --list-cases

# Small GPU smoke run, covering all three timing modes.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --dtypes bf16 --shapes 3x1025 --warmup 1 --repeat 2 \
  --output-dir reports/softcapped-selected-logprob/smoke

# Default comparison; --shapes, --dtypes and --modes can narrow or extend it.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --output-dir reports/softcapped-selected-logprob

# Optional dispatch check: AUTO vs explicit ROW/PARALLEL and eager PyTorch.
# Three dtypes; row/tile boundaries, unseen widths and large-V extrapolation.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --compare-strategies --modes forward --warmup 50 --repeat 200 \
  --shapes 16x31744 16x31745 16x32768 4095x32768 4096x32768 4097x32768 \
    16x65537 1024x202048 4096x202048 4097x202048 16x2097153 \
    4096x262144 8192x65536 \
  --output-dir reports/softcapped-selected-logprob-ranges

# Check a different cap at an unseen vocabulary width on both sides of M=4096.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --softcap 20 --compare-strategies --modes forward --warmup 50 --repeat 200 \
  --shapes 16x202048 1024x202048 4096x202048 4097x202048 \
  --output-dir reports/softcapped-selected-logprob-cap20
```

Validation has three layers:

| Layer / pytest marker | Comparison and requirement | Location |
| --- | --- | --- |
| Fixed-path consistency / `softcap_consistency` | Compare the declared tile tree with an explicit PyTorch reference. Separately require identical output/statistic/gradient bytes across strategies, training, `no_grad`, `inference_mode`, repeated backward and batch/chunk/layout changes. | `tests/ops/logprob/test_softcapped_selected_logprob_ordered.py` and consistency items in `test_softcapped_selected_logprob.py` |
| Independent FP32 accuracy and API checks / `softcap_accuracy` | Compare with Native FP32 `log_softmax` and autograd on the same device. Promote already-quantized inputs/upstreams to FP32 and retain FP32 reference gradients. Keep the WS1 `logprob` tolerances. | Accuracy items in `tests/ops/logprob/test_softcapped_selected_logprob.py`; the shared harness separately retains its checks |
| Independent FP64 mathematics / `softcap_math` | Evaluate stable FP64 `log_softmax` and autograd without calling either implementation. Check Native, public Triton strategies and the ordered PyTorch reference against the unchanged extra mathematical targets. | `tests/ops/logprob/test_softcapped_selected_logprob_math.py` |

Previously mixed accuracy/invariance cases now produce separate pytest items.
An FP32 comparison failure therefore cannot prevent a matching consistency
case from running. The original wide-input failures, caps and tolerances are
retained. Do not interpret success in one layer as success in another.

`tests/helpers/softcapped_ordered_reference.py` is a test-only arithmetic
specification. It uses eager PyTorch additions to reproduce the tile partition,
tree, FP32 accumulation and subtraction/cast points. It does not call the
production validator or kernel and is neither the independent accuracy oracle
nor the timed PyTorch baseline. Its backward is an explicit analytic expression;
the operator's backward contains no reduction, and Native autograd remains
the independent derivative check.

Matching additions alone does not guarantee matching elementary functions:
native `torch.log` and Triton's `tl.log`, or `torch.exp` and backward `tl.exp`,
can round differently. The ordered tests first compare tile sums on identical
FP32 values, independently of `exp` and `tanh`. Full-path checks separately
compare forward tile partials, then use small, explicitly named GPU probes at
the log/backward-exp boundaries before comparing subsequent PyTorch arithmetic
bitwise. Those probes do not establish the intrinsics' mathematical accuracy;
the independent suites exercise the complete production operation without
them. The pure PyTorch reference is also checked directly against FP64.

The direct FP32 tests retain the existing tolerance selection: forward FP32
outputs use `atol=1e-5, rtol=0`; gradients use their output dtype's contract row
(FP32 `1e-5`, FP16 `5e-3`, BF16 `5e-2`, all with `rtol=0`). The FP32 reference
is not rounded to FP16/BF16 before comparison. The shared harness retains its
own contract selection by case dtype. Neither path permits private tolerance
relaxation. See the [FP32 reference definition](../contracts/ws1-numerical-precision-standard.md#3-definition-of-the-fp32-reference).

The extra FP64 tests preserve the earlier numerical targets: forward absolute
error at most `1e-5`; gradient absolute error at most `1e-5`, `5e-3` or `5e-2`
for FP32, FP16 or BF16 output respectively, with zero relative tolerance.
These are test-specific mathematical goals, not a project-wide FP64 contract.
They retain FP64 oracle values during comparison and do not borrow the WS1
resolver. Passing FP32 agreement does not automatically pass these stricter
checks. Running the whole test directory includes all three files; the extra checks
are not skipped or marked `xfail` to obtain a green result.

Random upstream gradients, saved-tensor reuse, strategy boundaries and indexing
guards remain covered. GPU tests skip on CPU hosts; a GPU host that cannot load
Triton fails rather than silently skipping.
Nondefault-cap cases cover reference outputs and gradients, all three strategies
and automatic selection, training/inference equality, batch-split strategy
changes, and two-GPU launches. The configurable-cap consistency sweep uses
`1e-6, 1e-3, 0.125, 0.5, 1, 3.7, 10, 17.3, 20, 30, 39.5, 50, 64, 70`;
it requires bitwise training/`no_grad`/`inference_mode` agreement and repeated
backward agreement. Triton also compares individual rows against the batch,
and AUTO is checked while crossing a ROW/PARALLEL boundary with each cap.
Endpoint tests use `float32.tiny` and a cap `1e-4` below the allowed maximum
at `V=1, 1025, 202048, 262144`, covering saturation and rejecting values outside
the bounds. Thus the largest tested valid caps vary with V (about 75.6–86.6).
Native GPU reference checks cover the same 14-cap sweep at `V=1025, 202048`.
All these checks cover FP16/BF16/FP32 without relaxing the existing tolerances.
`test_cap64_exp_rounding_regression` retains the failing seed `438` at
`[17, 32767]` across all three dtypes, the three retained strategies and AUTO.
Output and gradient accuracy run as separate cases; it also checks
training/inference equality and repeated backward. The focused A6000 experiment
passed for BF16 across the five then-available Triton selections, including bitwise agreement
between whole-batch ROW and split-batch PARALLEL outputs, statistics and gradients.
The subsequent H100 run passed the existing FP32-reference and bitwise suites,
but the expanded benchmark found five further FP32-reference failures at
`[1024, 262144]`: FP16/FP32 with cap `64`, and FP16/BF16/FP32 with cap `70`.
Their maximum output error was `1.52587890625e-5`, above the unchanged `1e-5`
limit. The benchmark did not time these failing cases. These FP32 acceptance
failures are separate from the FP64 target below.

The captured-row diagnostic on RTX 4090 reproduced the original production
partials, sum, log and output bitwise. Using the same FP32 sum but computing
`FP32(FP64(selected_score) - log(FP64(sum_exp)))` removed all nine failing rows
against both the saved H100 Native output and local Native output. Just replacing
FP32 `tl.log` with FP32 `libdevice.log` did not fix them; rounding the FP64 log
back to FP32 before subtraction also retained failures.

That FP64-finishing candidate has now been replaced by the whole-row maximum
shift described above. Forward and backward share the maximum and shifted
log-sum, and production arithmetic is entirely FP32. The nine-row result is
historical evidence for the previous candidate, not validation of the shifted
implementation. Native now uses FP32 `log_softmax`; all accuracy limits remain unchanged. Full-input
regressions, FP64 checks, and cross-strategy output/statistic/gradient tests must
run again on a GPU before claiming accuracy or performance improvements.

Additional tests cover fully negative rows, saturated uniform rows, signed
zeros, a maximum in the final partial tile, and tied maxima across tile
boundaries. They compare all three strategies and AUTO, including saved
statistics and training/inference output bytes.

`test_large_cap_wide_benchmark_regression` retains all five combinations with
independent output-accuracy, gradient-accuracy and training/inference bitwise
checks (15 cases). It uses the benchmark's GPU generator, seed `415`, full
`[1024, 262144]` draw, scale `30`, and original token-ID/upstream draw order.
Do not substitute CPU generation or a smaller initial draw when reproducing
the H100 failures. To debug on another GPU, capture the exact affected input
rows first and verify that their isolated outputs match the original batch.
These regressions do not use `xfail` or wider tolerances. Run them with:

```bash
python -m pytest tests/ops/logprob/test_softcapped_selected_logprob.py \
  -k large_cap_wide_benchmark_regression -q -rs
```

Two-GPU launch coverage retains caps `30` and `50`.
Division-stage tests compare with an independently
rounded FP64 quotient, including `39.5 / 50`. A deterministic wide-row regression
at `softcap=50`, `V=202048` checks both the revised FP32 reference and FP64.
Omitted and explicit `softcap=30` calls must agree bitwise. These GPU checks must
pass on the target hardware before extending validation claims to new caps.

The original unshifted reference had an extra FP64-target failure at `softcap=70`,
`[M, V]=[17, 32767]`, seed `438`, with an error around `1.18e-5`.
A four-way diagnostic on RTX 3090 compared the shifted Triton strategies,
the unshifted Native reference, an FP32 `log_softmax` comparator and independent
FP64 mathematics across all 42 dtype/cap combinations of this fixture. Each
Triton strategy agreed with the FP32 `log_softmax` comparator within `1e-5`;
the maximum difference was `7.62939453125e-6`. All strategy output, saved-statistic
and gradient byte comparisons, and training/inference output comparisons passed.
Native now uses that same FP32 `log_softmax` expression. This focused diagnostic
does not replace the full GPU regression or qualify the large-vocabulary cases.

Both the stable PyTorch comparator and Triton still exceeded the extra FP64
output target for BF16 cap `70` (row 12, `1.01174e-5`) and FP32 cap `64`
(row 13, `1.02073e-5`). The two FP32 subtractions can accumulate rounding error
even after maximum shifting. `test_fp64_math_large_cap_rounding_regression`
retains these cases and the original FP32 cap `70` case, with output and gradient
checks separated. No tolerance is widened and no failure is marked `xfail`.
FP32-reference acceptance and these extra mathematical targets remain distinct.
The broader CPU suite also exposes additional `log_softmax` rounding discrepancies
on wide rows. For example, local PyTorch 2.13.0 at `[3, 202048]`, FP32 cap `0.001`
and seed `463` differs from FP64 by approximately `3.14e-4`. Maximum shifting
does not guarantee that every backend's reduction satisfies the extra `1e-5`
target. These failures remain visible in the mathematical suite, and the focused
GPU results must not be generalized to all widths or backends.
Capture the FP32 cap `70` intermediate values with:

```bash
python -m tools.validation.operators.diagnose_softcapped_precision \
  --output-dir reports/softcapped-precision
```

This diagnostic records Native FP32, shifted Triton and legacy unshifted
arithmetic against FP64, with FP64 continuation from each rounding stage.
The instrumented ROW output and both saved statistics must match production bitwise;
otherwise the trace fails and cannot attribute the production error. Its exit
status checks instrumentation, not accuracy: run the FP32 and FP64 pytest
layers separately to obtain their respective accuracy results.

`NativeSoftcappedSelectedLogprobOp` is both the numerical reference and the timed
PyTorch baseline. The benchmark calls the public Native and automatic Triton
Ops on the same CUDA/ROCm GPU, without maintaining another PyTorch implementation
or importing private input-validation helpers. Triton must pass output and
gradient checks against Native FP32 before timing. The accuracy check retains
FP32 reference gradients, while timing still calls both Ops with the original
FP16/BF16/FP32 inputs. Native's FP32 device-divisor
creation and transfer are included in forward and forward+backward timing;
backward timing reuses the divisor saved in the prebuilt graph. Reports identify
this division path and FP32 `log_softmax`. Earlier scalar-division or unshifted
exp/sum/log measurements used a different baseline, so their speedups must not
be presented as measurements against the current reference. The revised baseline
requires a new performance comparison.
Defaults cover five shapes
(`1x1025`, `1x262144`, `16x262144`, `128x262144`, `1024x262144`), all three input
dtypes and forward/backward/forward+backward: 15 input cases and 45 comparisons.
Inputs are contiguous and reused without explicit cache eviction. No
`torch.compile` or graph capture is used. The default run performs no strategy sweep.

It reuses `PerformanceProfiler` accelerator-event timing and checks output and
random-upstream gradient accuracy before timing, even with `--modes forward`.
The untimed gate also compares AUTO output/gradient bytes with ROW, ROW_PIPELINED
and PARALLEL, and checks AUTO training/inference equality. JSON records these
checks as `strategy_bitwise` and `train_inference_bitwise`; a mismatch stops the
case before timing. It does not time the extra strategies unless requested.
Forward runs under `no_grad`; backward uses a prebuilt retained graph;
forward+backward constructs a fresh graph per call. Compilation and input
preparation are outside timing. Timed wrappers include allocation and dispatch
with a warm selector cache; small inputs can be affected by host launch gaps.

`--compare-strategies` optionally adds explicit ROW and PARALLEL to **forward**
timing. All four providers rotate through each timing position across four
rounds. The report gives the median of round medians, preserves individual
rounds and reference-accuracy checks, and displays AUTO latency divided by the
faster explicit strategy. Values above 1 expose dispatch overhead or a slower
choice. The aggregate `std_ms` is the largest within-round sample deviation,
not a confidence interval. Repeat near ties on the target GPU; the benchmark
does not modify the map. Backward and forward+backward retain the standard
PyTorch-vs-AUTO comparison even when this flag is enabled.

JSON/Markdown reports record the selected strategy, median latency, speedup,
extra peak allocation, softcap and the hardware/software environment. JSON also retains
sample standard deviations and `accuracy` errors against the native baseline.
Extra allocation excludes
inputs and prebuilt graphs. Cases are saved outside timing as they finish;
interrupted reports remain marked `complete: false`. Speedup is native latency
divided by Triton latency, so values below 1 expose a slowdown. When run from the
repository root, reports default to the directory
`reports/softcapped-selected-logprob/`. Attach performance evidence to the PR
rather than committing generated reports.

### Reports and validation evidence

See [PR #463](https://github.com/RL-Align/RL-Kernel/pull/463) for H100
training/inference consistency results, strategy-tuning evidence and performance
comparisons with eager PyTorch. The PR links the Markdown reports and JSON
measurements, and records the tested source commit and hardware/software
environment. ROCm validation remains pending.
