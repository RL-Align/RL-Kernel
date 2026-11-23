# Device-local ordered reductions

These helpers let a fused kernel use a prescribed floating-point reduction tree.
They are building blocks inside an enclosing GPU kernel, not Tensor operators or
collectives. The caller owns allocation, launches, device/stream selection and
producer dependencies. Elementwise transforms and masks also belong to the caller.

Fixed order and numerical accuracy are separate requirements. The same operands,
tree, dtype, callback and rounding settings must be used when checking bits.
Independent accuracy checks remain necessary: a same-tree reference can repeat
an implementation's mistake.

## Ownership and frontends

| Component | Canonical location | Responsibility |
| --- | --- | --- |
| Arithmetic defaults | `rl_engine/contracts/operators/ordered_reduction.py` | Tile width and initial group size; no backend imports |
| Default implementation | `rl_engine/backends/shared/triton/reduction/` | Helpers embedded in `@triton.jit` kernels |
| Gluon implementation | `rl_engine/backends/shared/triton/reduction/gluon.py` | Shared explicit-layout arithmetic embedded in `@gluon.jit` kernels |
| Matching reference | `rl_engine/reference/reduction/` | Eager PyTorch translation of the same arithmetic |
| Tests | `tests/reference/reduction/`, `tests/backends/shared/triton/reduction/` | Reference and both frontend conformance suites |
| Performance comparison | `benchmarks/backends/shared/triton/benchmark_ordered_reduction.py` | Native baselines and enclosing kernels for both frontends |

Gluon is Triton's explicit-layout frontend, not another hardware platform. Both
implementations share the arithmetic contract. The Gluon source parameterizes
warp/wave width instead of assuming every device has NVIDIA's 32-lane warps.
The caller supplies the compiler target's width and can tune batch size,
vectorization and launch warps without copying the reduction algorithm.
GPU correctness has been validated on H100 and A40; the reported performance
measurements are from H100. Support for configuring an AMD wave layout is not a
claim of validated ROCm execution or equal bits across different GPUs.

Both frontends expose the same six device functions. The upper layer chooses an
entire enclosing kernel and its execution settings. Neither frontend automatically
switches frontends. An ordinary Triton kernel cannot transparently call this Gluon
implementation. Importing the default package does not import Gluon.

The implementation was qualified with Triton 3.7.1. Its callback adapter uses
`triton.constexpr_function`, which is absent from older installations such as
3.4.0. Use the qualified compiler version or validate a new version explicitly.

## Public API

```python
from rl_engine.backends.shared.triton import reduction as ordered

# Inside @triton.jit: values is an already-loaded one-dimensional vector.
total = ordered.sum(values)
prefix = ordered.sum(values, length=128)

# input_ptr is a scalar device pointer; length is required for memory input.
whole_vector = ordered.sum(input_ptr, length=2500)
merged = ordered.sum_partials(partials_ptr, n_tiles)
```

| Function | Input | Result |
| --- | --- | --- |
| `sum` | Loaded vector, or scalar pointer plus length | Fixed-order sum |
| `product` | Same two input forms | Fixed-order product |
| `sum_partials` | Pointer to stored tile sums and count | Ascending left fold from zero |
| `product_partials` | Pointer to stored tile products and count | Ascending left fold from one |
| `reduce` | Loaded vector or pointer, plus `COMBINE_FN` and `IDENTITY` | Custom operation using the same tree |
| `reduce_partials` | Stored partials, callback and `initial` | Custom ascending left fold |

All return a scalar device value for further computation or storage by the caller.
They do not allocate outputs/workspaces, store results, launch another kernel or
synchronize programs. There is no public Python Tensor wrapper or launchable
`sum[grid]` kernel.

Input **type** chooses loaded-vector versus pointer handling at compile time.
`length` selects the participating prefix, not whether the helper loads memory:

- A loaded vector has a positive power-of-two width. Optional `length` masks its
  suffix with the identity without changing that width or tree. To select an
  arbitrary subset, mask values before calling the helper.
- A pointer addresses a contiguous vector. Its required nonnegative `length` is
  compile-time metadata; the enclosing kernel may compute the pointer at runtime.
  Empty memory input returns the identity without loading data.
- `COMPUTE_DTYPE=None` retains input precision. Explicit precision converts inputs
  before arithmetic. The callback owns arithmetic casts; there is no implicit
  widening. Store partials in compute precision and merge in that same precision.

Public JIT options accept named arguments. Their definitions do not use Python's
keyword-only `*`, because the qualified Triton version does not bind it correctly.

## Arithmetic contract

Default pointer tiling uses `BLOCK_SIZE=1024` and `MAX_GROUP_SIZE=8`. Loaded input
uses its existing width. Both sizes must be positive powers of two.

For one tile:

1. Let `G = min(width, MAX_GROUP_SIZE)`. Partition adjacent values into groups of G.
2. Within each group, combine columns 0, 1, ..., G-1 sequentially, starting with
   the first value. Do not insert an extra identity seed.
3. Combine the lower and upper halves of the group totals repeatedly until one
   value remains. Each tree operation has exactly two operands.

For eight group totals, the tree is:

```text
a0 = combine(s0, s4)    a1 = combine(s1, s5)
a2 = combine(s2, s6)    a3 = combine(s3, s7)
b0 = combine(a0, a2)   b1 = combine(a1, a3)
total = combine(b0, b1)
```

The pair equations, including operand order, define the tree. This is not an
adjacent-pair tree such as `(s0+s1)+(s2+s3)`.

For a memory vector, pad its final arithmetic tile with the operation's identity,
compute one unseeded result per tile, then fold in ascending tile order:

```text
total = combine(identity, partial[0])
total = combine(total, partial[1])
total = combine(total, partial[2])
...
```

Serial and parallel producers must preserve these boundaries, partial precision
and the final left fold. Producer completion order is irrelevant: store tile k
in slot k. Do not use atomics, a different tree over the partials, or `sum(partials)`
in place of `sum_partials`. Those would change the arithmetic.

The identity for padding and the initial accumulator are distinct concepts.
`reduce_partials` can take a nonidentity seed, applied exactly once. Custom
callbacks must be pure, preserve shape/dtype, and implement a mathematically
associative and commutative operation with a neutral padding identity on the
supported domain. Floating-point rounding can still depend on order. Subtraction
and division do not meet this contract.

## Embedding and backend selection

A default Triton caller can reduce a row without materializing partial storage:

```python
import triton
import triton.language as tl
from rl_engine.backends.shared.triton import reduction as ordered

@triton.jit
def row_sum(x, out, stride, length: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    total = ordered.sum(x + row * stride, length, COMPUTE_DTYPE=tl.float32)
    tl.store(out + row, total)
```

A caller opting into Gluon supplies a separate enclosing kernel:

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from rl_engine.backends.shared.triton.reduction import gluon as ordered_gluon

@gluon.jit
def gluon_row_sum(x, out, stride, length: gl.constexpr, WARP_SIZE: gl.constexpr):
    row = gl.program_id(0).to(gl.int64)
    total = ordered_gluon.sum(
        x + row * stride, length, COMPUTE_DTYPE=gl.float32, WARP_SIZE=WARP_SIZE,
        TILES_PER_BATCH=32, ELEMENTS_PER_THREAD=8,
    )
    gl.store(out + row, total)

# Host code, after choosing the enclosing Gluon kernel:
# torch.cuda is also PyTorch's device API for ROCm.
with torch.cuda.device(x.device):
    target = triton.runtime.driver.active.get_current_target()
    gluon_row_sum[(x.shape[0],)](
        x, out, x.stride(0), length, WARP_SIZE=target.warp_size,
        num_warps=4, enable_fp_fusion=False,
        **({"allow_flush_denorm": False} if torch.version.hip else {}),
    )
```

The host chooses which kernel to launch using the **input device's** metadata and
its own validated policy. A 32-lane or 64-lane target supplies that exact width;
do not tune this value independently of the hardware. The public default remains
32 to preserve existing NVIDIA callers. AMD callers must pass their actual
target width, which is not necessarily the same across all AMD architectures.
An explicitly selected backend's compilation or correctness
failure must surface, rather than silently substituting a result from another
backend. Callers providing only a Triton kernel remain supported.

For many independent tiles but few rows, the caller may launch parallel producers:

```python
# Inside a producer: values holds one masked arithmetic tile, already in FP32.
partial = ordered.sum(values, MAX_GROUP_SIZE=8)
tl.store(partials_ptr + row * n_tiles + tile, partial)

# Inside a subsequent merge kernel on the same stream:
total = ordered.sum_partials(partials_ptr + row * n_tiles, n_tiles)
tl.store(output_ptr + row, total)
```

Use the same frontend throughout each enclosing kernel. The Gluon API supports
the same composition. `benchmarks/backends/shared/triton/ordered_reduction_gluon_kernels.py`
provides complete examples, including target-width propagation to the merge.
A single program processing more tiles does not spread those tiles across SMs;
separate producer programs expose that parallelism. Cross-stream producers need
an explicit dependency before merge. The helpers themselves do not synchronize.

Custom reductions need no registry or operation-name switch:

```python
@triton.jit
def xor_pair(left, right):
    return left ^ right

# Inside another @triton.jit kernel:
flags = ordered.reduce(values, COMBINE_FN=xor_pair, IDENTITY=0)
merged_flags = ordered.reduce_partials(
    flags_ptr, n_tiles, COMBINE_FN=xor_pair, initial=0,
)
```

## Execution settings and preserved order

Arithmetic settings (`BLOCK_SIZE`, `MAX_GROUP_SIZE`, precision, callback and math
flags) must stay consistent across paths being compared. They are not backend
scheduling knobs. Changing them can change rounding.

| Execution setting | Frontend | Effect |
| --- | --- | --- |
| `num_warps` on the enclosing launch | Both | Threads assigned to the program |
| `PIPELINE_STAGES` | Triton pointer path | Software-pipeline hint; default unhinted |
| `TILE_UNROLL` | Triton pointer path | Loop expansion; default bounded automatic policy |
| `MERGE_UNROLL` | Triton partial merge | Expansion of the ordered merge loop; default 1 |
| `TILES_PER_BATCH` | Gluon pointer path | Independent tiles processed together; default 32, power of two up to 64 |
| `ELEMENTS_PER_THREAD` | Gluon pointer path | Initial memory layout; default 8, positive power of two |
| `WARP_SIZE` | All Gluon helpers | Hardware warp/wave width; default 32, must match the compilation target |

To qualify another NVIDIA or AMD GPU, keep the arithmetic settings fixed and
pass its target width. First run the shared conformance suite, including custom
callbacks, same-tree reference checks, inference and Graph replay. Then measure
execution settings on that device. Retain only configurations that pass those
checks; callers own any hardware/shape dispatch table. A future architecture
needing different instructions can add a separate implementation, while reusing
the numerical contract and tests. No unmeasured AMD tuning table is provided.

The final implementation avoids arithmetic changes while reducing overhead:

- **Group extraction:** reshape/split selects the original value bits. The group
  still folds columns left to right. Gluon explicitly keeps adjacent group
  members in thread-local registers; ordinary Triton leaves layout to compilation.
- **Compact tails:** the Triton pointer path loads fewer lanes for eligible short
  tails but retains every original padding combine and rounding point. It does
  not simply resize the tree to the number of valid values.
- **Bounded unrolling:** automatic full expansion is limited to at most 16 tiles
  and 64 KiB of logical compute values; larger loops remain unexpanded. This is
  a code-size heuristic, not an allocation or hardware limit.
- **Partial prefetch:** independent partial loads are batched in bounded windows,
  while the accumulator consumes slots in ascending order. Window boundaries do
  not reset the seed or insert extra neutral additions.

These choices do not guarantee identical compiler layouts or speed on every GPU.
Requalify correctness, compilation time and performance when changing hardware,
compiler, layouts or execution settings. Use `enable_fp_fusion=False` for the
fixed-reference comparisons; qualified ROCm callers also need matching denormal
settings. The current helpers do not contain a performance dispatch map.

## Matching PyTorch reference and invariance

```python
import torch
from rl_engine.reference import reduction as reference

# Reduce the final dimension of an already-loaded, power-of-two-width Tensor.
tile = reference.sum(values, compute_dtype=torch.float32)

# Model pointer tiling, including its one outer identity seed.
whole = reference.sum(values, from_memory=True, compute_dtype=torch.float32)
merged = reference.sum_partials(partials)
```

The reference exports the same six names, uses lowercase option names, and
supports leading batch dimensions and strided tensors. `from_memory=True` is
reference-only: a Torch Tensor does not distinguish a loaded JIT value from a
scalar pointer. Generic stages correspond to Triton's tile/partial/serial/merge
stages. Tests inspect partials and totals through these stage references directly;
there is no separate diagnostic collector API.

Reference operations preserve PyTorch autograd. Forward order does **not** promise
that PyTorch-generated backward reductions use this tree. A fused operator that
needs backward conformance must explicitly define and test its backward order.
Empty serial and partial folds return independent, writable copies of the
broadcast seed, retaining its autograd connection without sharing its storage.

For prefix invariance, use the same effective values and arithmetic configuration
when physical storage, batch membership or masked suffix length changes. Pointer
input's fixed tiles isolate its arithmetic from the row's physical stride. Loaded
input retains its loaded width; changing that width can change its tree. This is
not an incremental sum state: adding a new valid token changes the input and may
change the final result. Runtime per-row ragged lengths, paged KV-cache addressing
and attention's tuple-valued online-softmax state are outside this API.

## Validation

Validation has separate layers:

1. Fixed-tree conformance against the explicit PyTorch reference, including
   cancellation, rounding boundaries, custom callbacks and nonidentity seeds.
2. Same-input bitwise checks across schedules, prefix padding/batch placement,
   inference mode and CUDA Graph replay. Inputs and poisoned suffixes are checked.
3. Independent numerical accuracy against higher-precision or exact-arithmetic
   references, using the repository's precision contract where applicable.

Tests also cover extraction of signed zero, NaN payloads, subnormals and large
integers; tail compression; merge-window boundaries; configuration validation;
the JIT/interpreter callback adapter; and empty-fold storage independence with
scalar, broadcast and shape-matched seeds. Same-tree equality does not replace
independent accuracy, and helper-level tests do not qualify a full model or
training/rollout engine integration. No cross-device/compiler bitwise guarantee
is made.

Each suite owns a different part of that coverage:

| Suite | Responsibility |
| --- | --- |
| `tests/reference/reduction` | Explicit pairs and frozen expected values; public dtype/shape/custom-callback contracts and autograd |
| Triton `test_ordered_reduce.py` | Internal extraction, compact-tail and merge-window boundaries; serial/parallel scheduling equivalence |
| Triton `test_public_api.py` and `test_input_dispatch.py` | Caller options, loaded/pointer dispatch, independent accuracy and batch/padding invariance |
| `test_public_compile.py` | CPU-side JIT compilation and interpreter checks, independent of GPU availability |
| `test_gluon_ordered_reduction.py` | Gluon layouts, batching, same-tree bits and replay/inference equivalence |
| `test_ordered_reduction_benchmark.py` | Planning, reporting, timeout/replay failure gates and real GPU benchmark smoke tests |

Shared dtype and boundary properties are tested at the public interface rather
than repeated through every internal helper. Internal cases target distinct
branches; scheduling tests vary execution settings while keeping arithmetic
fixed. Historical GPU test counts in the H100 report refer to the recorded run,
not the current collection after test consolidation.

Test launchers live beside the backend suite in `_kernels.py` and
`_gluon_kernels.py` and are imported lazily. `tests/helpers/reduction_inputs.py`
contains shared input data only; `tensor_checks.py` provides the byte comparison.
Backend tests do not depend on benchmark launchers or another operator's tests.

```bash
python -m pytest tests/reference/reduction tests/backends/shared/triton/reduction \
  tests/benchmarks/test_ordered_reduction_benchmark.py -q
```

GPU/Gluon tests skip when the required runtime or accelerator is absent. They use
the selected target's width and are not restricted to H100. H100 correctness and
measured performance are described in the [H100 report](../benchmarking/ordered-reduction-h100.md).

## Performance comparison

The maintained benchmark compares **ordinary `torch.sum`**, **native `tl.sum`**,
**ordered Triton**, and optionally **ordered Gluon**. The fixed-order PyTorch
reference checks conformance; it is not the performance baseline.

```bash
# Inspect the exact plan without importing Triton/Gluon or initializing a GPU.
python -m benchmarks.backends.shared.triton.benchmark_ordered_reduction --dry-run

# Ordinary Triton, native Triton and PyTorch, with hard per-case timeouts.
python -m benchmarks.backends.shared.triton.benchmark_ordered_reduction \
  --timings graph --warps 4 --case-timeout 180

# Add Gluon row and parallel callers on the selected CUDA/ROCm device.
python -m benchmarks.backends.shared.triton.benchmark_ordered_reduction \
  --include-gluon --timings graph --warps 1 --merge-warps 2 \
  --shapes 128x8193 16x262144 --dtypes fp16 bf16 fp32 --case-timeout 180
```

Each run uses **one explicit configuration per schedule**, not a parameter grid
or best-of-search selection. `--warps`, `--merge-warps`, `--pipeline-stages`,
`--gluon-elements-per-thread` and `--gluon-tiles-per-batch` select one execution
setting each; the arithmetic contract is fixed at 1024/8. The benchmark reads
warp/wave width from the selected compilation target and records it in JSON.
Backend API execution overrides remain available to callers. Experimental
candidate implementations, compiler dumps and sweep drivers are not part of the
maintained benchmark.

All providers accumulate/output FP32 and reduce the same valid input prefix.
Native row `tl.sum` is limited to 32768 elements to bound compilation; a two-stage
native baseline also covers larger inputs. Reports retain each provider rather
than hiding slower results. The two-stage ordered paths include both launches.

Inputs, outputs and partial buffers are preallocated. Compilation, validation
and Graph capture are excluded. All providers receive common warmup and shuffled
round ordering; reports preserve the executed order for each round, samples,
round medians, spread, environment, source hashes, scratch bytes and extra peak
allocation. Raw samples appear once, in each result's `samples_ms[round][sample]`;
`round_orders` records the provider and timing mode measured in each round.
Inputs are reused without explicit cache eviction. Graph timing
divides batched replay time by its call count; eager event timings may include
host dispatch gaps. Compare timing modes separately. `torch.compile` is not used.

Independent accuracy and same-tree checks run before timing; all rows are checked
across ordered schedules and sampled rows against the CPU fixed reference. Graph
replay must overwrite NaN-poisoned output and preserve the eager output bits.
Each case runs in an isolated process with a hard timeout covering compilation,
and partial reports survive failure. Local outputs default to
`artifacts/benchmarks/`; measured evidence is published as PR attachments rather
than committed to the source tree.
