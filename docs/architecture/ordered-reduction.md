# Device-local ordered reductions

Import the package and call one function inside your Triton kernel:

```python
import triton.language as tl
from rl_engine.backends.shared.triton.reduction import sum_vector

# Inside @triton.jit; ptr addresses a contiguous vector of length V.
total = sum_vector(ptr, V)
```

The function loads the whole vector in tiles, handles its final incomplete tile
and returns a scalar. `V` can exceed the tile width and need not be a power of
two. No combiner, partial buffer or merge call is required from ordinary callers.

| Package export | Input | Result |
| --- | --- | --- |
| `sum_tile(values)` | Already-loaded/computed power-of-two vector; caller masks invalid lanes with zero | Tile sum, with no extra seed |
| `product_tile(values)` | Already-loaded/computed power-of-two vector; caller masks invalid lanes with one | Tile product, with no extra seed |
| `sum_vector(ptr, length, COMPUTE_DTYPE=None)` | Pointer and any nonnegative length | Sum of the whole vector; empty returns zero |
| `product_vector(ptr, length, COMPUTE_DTYPE=None)` | Pointer and any nonnegative length | Product of the whole vector; empty returns one |
| `sum_partials(ptr, n_tiles)` | Pointer to stored tile sums and their count | Left-to-right sum starting from zero; empty returns zero |
| `product_partials(ptr, n_tiles)` | Pointer to stored tile products and their count | Left-to-right product starting from one; empty returns one |

Tile helpers default to `values.dtype`, and vector helpers to the input pointer's
element type, with no automatic widening. Pass `COMPUTE_DTYPE=tl.float32`, for
example, to accumulate BF16 input in FP32. The tile/vector helpers
have default tree parameters; callers need not choose a strategy or supply a combiner.

For already-computed tile results, the partial helpers need only a pointer and
count. They supply the operation and seed, infer the storage dtype, and merge
in ascending tile order. Optional `MERGE_UNROLL=1` controls loop expansion.
`COMPUTE_DTYPE=None` uses storage precision; if supplied, it must match the
partial buffer dtype. Unlike vector helpers, these helpers add no new tile tree.
For example, inside an operator's merge kernel:

```python
from rl_engine.backends.shared.triton.reduction import sum_partials

total = sum_partials(partials_ptr, n_tiles)
```

For values already computed inside a kernel, use a tile helper directly:

```python
from rl_engine.backends.shared.triton.reduction import sum_tile

# Inside @triton.jit, after loading a power-of-two vector:
transformed = values.to(tl.float32) * 2.0 + 1.0
transformed = tl.where(mask, transformed, 0.0)
total = sum_tile(transformed)  # transformed is already FP32.
```

This avoids storing transformed values just to load them again. A tile helper
uses the supplied vector's width; it neither loads another tile nor adds a
whole-vector seed. Use a vector helper when it should handle loading and tiling.

## Kernel usage and parallelism

```python
import triton
import triton.language as tl
from rl_engine.backends.shared.triton.reduction import sum_vector

@triton.jit
def row_sum_kernel(x_ptr, out_ptr, V: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    total = sum_vector(x_ptr + row * V, V, COMPUTE_DTYPE=tl.float32)
    tl.store(out_ptr + row, total)
```

For contiguous `[M, V]` input, the operator launches this kernel with `M`
programs and an `[M]` output. Different rows can execute concurrently. Within
each program, tiles are processed sequentially; each tile's vector operations
are distributed across the program's threads by Triton.

The caller controls the grid and `num_warps`. `sum_vector` does not split one
vector across multiple programs or launch additional kernels. Its `BLOCK_SIZE`
is an arithmetic tile width, **not a limit on the whole vector length**.
Allocation, device/stream selection and output storage belong to the operator.
There is no standalone Tensor launcher in this Triton package.

Both vector helpers also accept `PIPELINE_STAGES=None`. The default leaves
the tile loop unhinted; an explicit value such as `PIPELINE_STAGES=3` is passed
to `tl.range(num_stages=...)`. This is a loop scheduling hint, not the enclosing
kernel's launch `num_stages`, and it does not change the tile tree or partial
fold. Internal helpers receive the parameter explicitly, without a default.
PyTorch needs no counterpart because its reference describes arithmetic rather
than GPU scheduling.

## Matching PyTorch reference

```python
import torch
from rl_engine.reference.reduction import sum_vector

with torch.no_grad():
    expected = sum_vector(x, compute_dtype=torch.float32)
```

The reference accepts CPU or GPU tensors, including strided inputs. It reduces
the last dimension and preserves leading dimensions: `[M, V]` becomes `[M]`,
and `[V]` becomes a scalar. Omitting `compute_dtype`, or passing `None`, uses
`x.dtype`, matching the Triton default. Run eagerly, under `no_grad` for gradient-requiring
inputs. This reference checks forward arithmetic; its autograd does not define
a custom Triton operator's backward order.

The PyTorch package exports the same names. `sum_tile` and `product_tile`
also reduce the last dimension, which must be a positive power of two. They
default to `values.dtype`; `compute_dtype` optionally overrides it. Fill any
invalid lanes before calling them.
`sum_partials(partials)` and `product_partials(partials)` likewise need only
the tensor; its last dimension supplies the partial count. The optional
`compute_dtype` and `merge_unroll` arguments mirror the Triton helpers.

Keep native PyTorch or higher-precision checks separately for numerical
accuracy. A reference with the same tree is not an independent mathematical
oracle or a performance baseline.

## Optional arithmetic configuration

Defaults come from `contracts/operators/ordered_reduction.py`:

| Triton parameter | PyTorch parameter | Default | Meaning |
| --- | --- | ---: | --- |
| `BLOCK_SIZE` | `block_size` | 1024 | Elements in each arithmetic tile |
| `MAX_GROUP_SIZE` | `max_group_size` | 8 | Maximum adjacent values folded sequentially within each tile |
| `SECTION_COUNT` | `section_count` | 4 | Maximum sections used to combine those group totals |

Each size must be a positive power of two. For example, these calls use matching
nondefault configurations:

```python
# Inside a Triton kernel:
total = sum_vector(
    ptr, V, COMPUTE_DTYPE=tl.float32,
    BLOCK_SIZE=2048, MAX_GROUP_SIZE=16, SECTION_COUNT=8,
)
```

```python
# In eager PyTorch:
expected = sum_vector(
    x, compute_dtype=torch.float32,
    block_size=2048, max_group_size=16, section_count=8,
)
```

All three sizes affect the arithmetic tree and can change rounding. Preserve
them, tile boundaries, precision and elementwise operations across paths that
require identical bits. Fixed order does not prevent numerical error or
establish reproducibility across hardware and compiler versions.

`COMPUTE_DTYPE` selects input conversion, arithmetic, seed and result precision.
All public helpers infer it when omitted or `None`; internal stages receive
the resolved dtype explicitly. Tiles use the current values' dtype, and pointer
helpers use the input storage dtype. Low-precision inputs are not promoted to FP32.
There is no utility-specific dtype whitelist; both libraries and the selected
backend must support the requested operations. Use an integer accumulator to
count booleans, and a width sufficient for the result. No automatic accumulator
widening is supplied. FP64 cannot restore bits lost before the input arrived.

## Integration with softcapped selected logprob

The operator computes softcap and shifted exponentials inside its kernels.
Its tile and merge helpers come from one package entry point:

```python
from rl_engine.backends.shared.triton.reduction import (
    sum_partials,
    sum_tile,
)
```

The PyTorch reference package exposes the same names. The convenience functions
encapsulate operation, seed and defaults; callers do not need generic reducer
parameters or imports from the implementation module.

| Helper | Role in logprob |
| --- | --- |
| `sum_tile` | Reduce the already-computed exponentials in one tile, without writing them to global memory |
| `sum_partials` | Merge stored tile sums in ascending tile order from zero, matching ROW's accumulation |

`_softcapped_exp_tile_sum` loads one tile, computes FP32 softcap and shifted exp,
clears padded lanes, and calls `sum_tile`. All three forward strategies use it:

- ROW folds its tile sums from positive zero within one program.
- ROW_PIPELINED uses the same loop and arithmetic with a pipeline hint.
- PARALLEL writes each sum to its tile slot, then calls `sum_partials` in a
  dependent merge kernel. Producer completion order does not affect addition order.

The shared `_store_softcapped_selected_logprob` then finishes the logprob.
Logprob passes its own `REDUCTION_CONFIG` explicitly, independently of the
utility's defaults. All schedules keep the same FP32 arithmetic and cast points.
Its native `log_softmax`/autograd reference remains independent.
See the [operator documentation](../operators/softcapped-selected-logprob.md)
for the full forward/backward contract and validation commands.

The six package exports cover loaded tiles, complete vectors and stored partials.
The generic merge and binary combiners remain in the implementation module.
Ordinary callers need not assemble or configure those internal stages.

<details>
<summary>Internal stages and conformance checks</summary>

### Files and responsibilities

| File | Responsibility |
| --- | --- |
| `contracts/operators/ordered_reduction.py` | Shared arithmetic defaults; no Torch or Triton dependency |
| `backends/shared/triton/reduction/ordered_reduce.py` | JIT helpers embedded in a caller's kernel |
| `reference/reduction/ordered_reduce.py` | Eager PyTorch translation, with matching names, order and arithmetic |
| Each package's `__init__.py` | Six sum/product helpers through one entry point; the only `__all__` list |
| `tests/helpers/ordered_reduce_kernels.py` | Test-only launch adapters; no production operator dependencies |

Existing reductions retain their own contracts: `reduce_rows_fp32` in
`rl_engine/ops/autograd/vjp_fp32.py` folds dimension zero for parameter gradients;
Attention and GEMM have specialized trees; distributed collectives live in
`rl_engine/distributed/algorithms/collectives.py`. Replacing any of these with a
different tree or axis is not a bitwise-equivalent substitution.

### Internal helpers

| Helper | Input and result |
| --- | --- |
| `reduce_add` / `reduce_multiply` | Pure pairwise `+` / `*` combiners |
| `ordered_tile_reduce` | One loaded power-of-two vector → scalar; no seed or dtype cast |
| `ordered_reduce_partial` | Vector pointer and tile index → one unseeded partial, with conversion and neutral tail padding |
| `ordered_reduce_serial` | Full vector → scalar; fold tiles from `initial` within one program |
| `ordered_reduce_merge` | Stored partials → scalar; fold ascending tile slots from `initial` |

Triton's configurable `ordered_*` helpers require all their arithmetic
arguments. The public tile helpers call `ordered_tile_reduce` directly;
the public vector helpers call `ordered_reduce_serial` directly. Both supply
the chosen operation and resolved configuration without another forwarding layer.
The public partial helpers call `ordered_reduce_merge`, supplying addition/zero
or multiplication/one and resolving storage precision. These wrappers define
usable defaults; the generic helpers remain the single arithmetic implementation.
The PyTorch counterparts use lowercase parameter
names and infer lengths from tensors. Leading dimensions batch independent
vectors. PyTorch omits the scheduling-only `PIPELINE_STAGES` option.

The serial stage calls `ordered_reduce_partial` to load, cast and pad each tile,
then `ordered_tile_reduce` determines its combine pairs. Separate producers can
use the same partial stage, and `ordered_reduce_merge` folds their stored results.
Each stage has a distinct role; the small `reduce_add`/`reduce_multiply` functions
are the binary callbacks required by the shared tree and fold.

The PyTorch-only `ordered_reduce` collector composes partial and merge helpers,
returning `OrderedReduceResult.partials` and `.total` for stage diagnostics.
Its last dimension must be nonempty. It does not implement another tree.

### Fixed tile tree and partial fold

1. Split a tile into adjacent groups of `min(BLOCK_SIZE, MAX_GROUP_SIZE)` values.
   Fold each group left to right, starting from its first value.
2. Split group totals into `min(N_GROUPS, SECTION_COUNT)` equal sections.
   Repeatedly pair each section's lower and upper halves until one total remains.
3. Merge section totals using the same halving-distance pairs. Four sections
   produce `(s0 + s2) + (s1 + s3)` for addition.

Each tree step explicitly selects its lower/upper operands and calls the binary
combiner directly. Triton uses `permute`/`split`; PyTorch selects the same slices.
A one-entry stage passes through without adding an identity. Neither delegates
the tree to a native reduction.

Across tiles, serial and parallel schedules both compute
`(((initial + partial[0]) + partial[1]) + ...)` for addition. This is a left
fold, not another tile tree. `MERGE_UNROLL` accepts any positive integer and
expands loop steps without changing the pairs; incomplete tails preserve order.
Zero partials return `initial`. **Do not use `sum_vector` to merge partials**:
that would introduce another tile tree and change the arithmetic.

`IDENTITY` and `initial` have distinct purposes. The identity fills padded lanes
and must be neutral for the combiner; `initial` enters the whole-vector fold
exactly once. For `10 + sum(x)`, use padding zero and initial ten. Partials must
not contain the seed. Seeds may be runtime scalars; stored partials must already
have the requested compute dtype.

Custom combiners must preserve operand shape and dtype and have a valid padding
identity. These helpers target mathematically associative, commutative operations;
floating-point rounding still depends on the tree. Multi-field statistics and
arbitrary pairwise subtraction or division require a different formulation.
The built-in combiners simply apply `+` or `*`; explicit per-operation casts,
NaN rules and signed-zero behavior belong to the chosen combiner.

### Fused and parallel callers

A fused caller uses the public tile helpers shown above. Apply neutral padding
**after** the transformation, then use the returned scalar in further computation.

For explicit cross-program parallelism, the operator allocates partial storage,
launches producers into fixed tile slots, then launches a dependent merge.
Use the same stream or an explicit dependency across streams. JIT helpers cannot
synchronize independent programs. Partial storage and compute dtypes must match
so the parallel path introduces no extra narrowing.

Keep compiler settings consistent across paths. For composed arithmetic,
`enable_fp_fusion=False` preserves separate multiply/add rounding; denormal and
elementary-function settings remain the enclosing operator's responsibility.
A long serial merge may be expensive. Replacing it with another arithmetic tree
requires coordinated updates to every matching strategy and reference.

### Tests

```bash
python -m pytest tests/reference/reduction tests/backends/shared/triton/reduction -q
```

Coverage includes frozen nonassociative examples, the original 1024/8/4 tree,
small and large tiles, alternative group/section sizes, strided reference inputs,
tails, empty vectors, custom seeds, compute precision, and serial/pipelined/parallel
conformance. Integer, boolean and floating cases exercise their respective
supported operations. Embedded tests check package imports, result use in further
computation, input preservation, output bounds and different warp counts.

Independent exact-math cases and native higher-precision comparisons supplement
fixed-path checks. Sharing constants must not silently redefine expected results.
GPU tests skip on CPU hosts; a CPU pass does not qualify Triton compilation,
GPU numerical behavior or performance.

</details>
