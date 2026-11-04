# Nemotron Nano grouped experts and combine

The `moe_experts_combine` row of [RFC #434](https://github.com/RL-Align/RL-Kernel/issues/434):
grouped up-projection GEMM with a relu-square epilogue, grouped down-projection,
a fixed-order weighted combine over the six route slots, the optional shared
expert and the operator's single output cast, forward and backward. The contract
is `nemotron-experts-sm90-v1` and is shared by two providers.

Two implementations satisfy it:

| Provider id | Column | Source |
| --- | --- | --- |
| `rlkernel.nemotron_experts.sm90.v1` | NVIDIA SM90 CUDA | `csrc/cuda/moe/nemotron_experts.cu` |
| `rlkernel.nemotron_experts.triton.v1` | CUDA and ROCm | `rl_engine/backends/shared/triton/moe/nemotron_experts.py` |

Selection is the registry's, never a manual probe: the CUDA column lists the
native provider first and the Triton provider second, so a build without
`KERNEL_ALIGN_DET_GEMM_SM90=1` falls through to Triton instead of failing; the
ROCm column lists only the Triton provider, which is its sole implementation.
`cpu`, `npu`, `musa` and `mps` fail closed. `nemotron_experts_provider_id()`
reports the provider that would run, and the first call prints one
`[RL-Kernel][route] ... provider=<id>` line when route reporting is enabled.

The router row is separate: `nemotron_router_dispatch` produces `packed`,
`permutation` and `offsets`, this operator consumes them.

## Interface

```python
from rl_engine.ops.moe.nemotron_experts import nemotron_experts

out = nemotron_experts(packed, x, weights, permutation, offsets,
                       w_up, w_down, w_sh1=None, w_sh2=None)
```

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `packed` | `[6*T,2688]` | BF16 | Router payload, differentiable |
| `x` | `[T,2688]` | BF16 | Token payload for the shared expert; may be `None` |
| `weights` | `[T,6]` | FP32 | Router weights, slot ascending |
| `permutation` | `[6*T]` | int64 | Packed position to flat route slot |
| `offsets` | `[129]` | int64 | Exclusive expert prefix counts |
| `w_up` | `[128,1856,2688]` | BF16 | Stacked up-projection weights |
| `w_down` | `[128,2688,1856]` | BF16 | Stacked down-projection weights |
| `w_sh1` | `[3712,2688]` | BF16 | Shared up-projection; paired with `w_sh2` |
| `w_sh2` | `[2688,3712]` | BF16 | Shared down-projection |
| `out` | `[T,2688]` | BF16 | Combined expert output |

Every input must be contiguous and on one CUDA/ROCm device; hidden and
intermediate widths must be 64 aligned. `0 <= T <= 65536`, which bounds every
32-bit index product. `w_sh1`/`w_sh2` come as a pair, and `x` is required only
when they are present. `permutation` and `offsets` are the router's bijection and
exclusive prefix sums: the hot path checks metadata only and does not re-validate
their values, so a hand-built dispatch must guarantee both. An empty batch
returns an empty output without launching a kernel. Validation raises
`ValueError`; an unavailable platform or provider raises `RuntimeError`.

## Fixed arithmetic

The grouped GEMMs are the [det_gemm](det-gemm.md) contract, copied on purpose:
`BK = K_TREE_LEAF = 32` is one TMA tile and one FP32 leaf, each leaf is two
`mma.sync.m16n8k16` calls accumulated in FP32 and rounded once to BF16, and
leaves merge with the canonical midpoint tree. No split-K, no atomics, no TF32,
no autotuning. `tests/backends/cuda/test_nemotron_experts_cuda.py` pins the up
and down projections bitwise against `_C.det_gemm_*`, and the same file pins
`dPacked` and both weight gradients.

Weights are stacked as `[E,N,K]`/`[E,K,N]`, so one tensor map covers all experts
and expert `e` owns rows `[e*N, (e+1)*N)`. Two tensor maps are built per GEMM.

| Stage | Output | Cast |
| --- | --- | --- |
| up | `U [6T,1856]` BF16 | leaf and tree rounding |
| relu square | `Z` BF16, never materialized | `bf16(relu(fp32(U))**2)` |
| down | `Y [6T,2688]` FP32 | BF16 tree value widened, one rounding |
| shared | `sh_out [T,2688]` FP32 | same up/down rules, `E=1` |
| combine + exit | `out = bf16(sum_slots w*Y + sh_out)` | the operator's only cast |
| dY | `bf16(fp32(w) * fp32(dOut))` | one rounding |
| dZ, dPacked | BF16 | leaf and tree rounding |
| dW | BF16 | grouped rows ascending |
| dWeights | `[T,6]` FP32 | 32-wide leaf, fixed FP32 tree |

`relu(x)**2` is applied to BF16 fragments. A BF16 product is exact in FP32, so
`bf16(relu(fp32(x))**2)` and `__hmul2` after `__hmax2` produce identical bits.
The relu-square derivative in the backward epilogue is `2 * relu(U)` multiplied
once and rounded once.

The Triton provider folds leaves in ascending order instead of merging them with
the midpoint tree. Its frontend rejects plain-Python helpers inside a kernel, and
`tl.static_range` rejects the list operations the stack algorithm needs, so a
trace-time stack is not expressible; the alternative, encoding the pop pattern as
eight nested static conditionals, buys a per-leaf association difference that the
contract does not claim. Leaf width, the single BF16 rounding per leaf and the
fixed ascending order are all preserved and every kernel folds the same way, so
batch invariance is unaffected; only the association of the leaf sums differs
from the native tree. `dWeights` folds 32-wide FP32 leaves the same way.

## Ordering policy

* **Experts ascend.** A CTA resolves its `(expert, row tile)` by walking the 129
  `offsets` in shared memory, so the host never reads `offsets` and CUDA Graph
  capture stays legal. The grid is the upper bound `cdiv(rows,BM) + E`; surplus
  CTAs return immediately.
* **Combine slots ascend.** For every token row the six route slots are summed
  in ascending slot order into an FP32 accumulator, four columns per thread.
* **`dW` visits grouped rows ascending.** One pass per expert writes the whole
  weight-gradient tile; empty experts write exact zeros, which is why no
  full-tensor memset is needed.

## Qualification boundary

Within one provider, an identical token row and its upstream gradient keep the
same bits across batch sizes and positions, for `out`, `dPacked`, `dX` and
`dWeights`. `dW` repeats for an identical ordered token set; adding
independently computed microbatch `dW` is not a bitwise full-batch equivalence
claim. The two providers do **not** claim bitwise equality: each is qualified
against FP64 on its own selected branch, exactly as the CUDA and Triton
`det_gemm` implementations coexist.

Measured relative L2 against an unrounded FP64 pipeline, 17 tokens, shared expert
on, worst gradient per provider:

| Geometry | native | Triton | native vs Triton |
| --- | --- | --- | --- |
| core (256/128, 8 experts) | 7.1e-3 | 5.4e-3 | 8.3e-3 |
| Nemotron (2688/1856, 128 experts) | 8.8e-3 | 1.0e-2 | 1.3e-2 |

The cross-provider column stays near one BF16 ulp (7.8e-3): both providers round
every 32-wide leaf and cast once at the exit, but they associate the leaf sums
differently, and a BF16 output cannot resolve a smaller difference. Cross-compiler
equivalence, expert/sequence
parallelism and tensor parallelism are outside this operator. The ROCm column
was authored without ROCm hardware; it runs the same assertions as CUDA and
needs hardware re-qualification before it is quoted as verified.

## Tests

```bash
python -m pytest tests/reference/test_nemotron_experts_reference.py -q
python -m pytest tests/runtime/test_kernel_registry.py -q
python -m pytest tests/backends/cuda/test_nemotron_experts_cuda.py -q
python -m pytest tests/backends/rocm/test_nemotron_experts_rocm.py -q   # ROCm host
```

The reference suite is CPU-only: an independent scalar FP64 oracle on toy
geometry, empty experts, the token-count bound, cast points, token permutation
and FP64 `gradcheck`. The provider suites compare against an FP64 emulation of
the same pipeline, assert raw-bit batch and relocation invariance, repeatability,
CUDA Graph replay, autocast invariance and geometry rejection.

## Performance reproduction

The baseline runs the same math expert by expert with `torch.matmul` in FP32 and a
weighted, slot-ascending combine - the obvious eager formulation. Its expert span
list is read once outside the timed region. The candidate rows are the provider
objects themselves, so the public validation wrapper is outside both sides of the
comparison.

```bash
PYTHONPATH=. python benchmarks/operators/moe/benchmark_nemotron_experts_training.py \
  --tokens 1 16 128 513 1024 4096 8192 32768 --dtype bf16 \
  --providers native triton --distributions random concentrated \
  --qualify-max-tokens 128 --output /tmp/experts-bf16.json
```

Forward and forward+backward are measured separately: 20 alternating paired
samples of five executions, median reported, CUDA Graph replay, cotangents
prepared outside timing. `--reverse` repeats the sweep with the reversed initial
provider order; the two orders agree to a median 0.4% difference (worst sampled
cell 2.8%), so the tables below show the normal order only. The script refuses to
overwrite an existing output file. The 32768-token rows are run as a separate
invocation because three providers plus the FP32 baseline peak near 20 GiB.

### Forward, random routing, median ms

| T | baseline | native | Triton | baseline/native | baseline/Triton | native/Triton |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.452 | 0.785 | 0.475 | 0.58x | 0.95x | 1.65x |
| 16 | 4.297 | 3.530 | 1.825 | 1.22x | 2.35x | 1.93x |
| 128 | 10.463 | 5.940 | 3.392 | 1.76x | 3.08x | 1.75x |
| 513 | 12.207 | 6.382 | 3.753 | 1.91x | 3.25x | 1.70x |
| 1024 | 14.845 | 6.693 | 4.184 | 2.22x | 3.55x | 1.60x |
| 4096 | 30.172 | 15.471 | 9.792 | 1.95x | 3.08x | 1.58x |
| 8192 | 49.437 | 27.988 | 17.004 | 1.77x | 2.91x | 1.65x |
| 32768 | 161.059 | 106.635 | 64.256 | 1.51x | 2.51x | 1.66x |

### Forward + backward, random routing, median ms

| T | baseline | native | Triton | baseline/native | baseline/Triton | native/Triton |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.790 | 6.623 | 1.785 | 2.53x | 9.41x | 3.71x |
| 16 | 217.986 | 12.660 | 5.108 | 17.22x | 42.68x | 2.48x |
| 128 | 402.358 | 18.561 | 8.355 | 21.68x | 48.16x | 2.22x |
| 513 | 413.711 | 20.274 | 9.036 | 20.41x | 45.78x | 2.24x |
| 1024 | 425.562 | 23.213 | 10.544 | 18.33x | 40.36x | 2.20x |
| 4096 | 503.439 | 53.893 | 26.169 | 9.34x | 19.24x | 2.06x |
| 8192 | 600.466 | 96.951 | 47.714 | 6.19x | 12.58x | 2.03x |
| 32768 | 1198.635 | 354.791 | 183.996 | 3.38x | 6.51x | 1.93x |

### Forward + backward, concentrated routing, median ms

| T | baseline | native | Triton | baseline/native | baseline/Triton | native/Triton |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.770 | 6.578 | 1.780 | 2.55x | 9.42x | 3.70x |
| 16 | 17.430 | 6.693 | 1.772 | 2.60x | 9.84x | 3.78x |
| 128 | 18.851 | 7.035 | 1.933 | 2.68x | 9.75x | 3.64x |
| 513 | 24.719 | 11.072 | 3.747 | 2.23x | 6.60x | 2.96x |
| 1024 | 31.486 | 15.047 | 5.581 | 2.09x | 5.64x | 2.70x |
| 4096 | 72.829 | 46.002 | 21.156 | 1.58x | 3.44x | 2.17x |
| 8192 | 132.721 | 91.663 | 44.662 | 1.45x | 2.97x | 2.05x |
| 32768 | 490.167 | 354.221 | 182.743 | 1.38x | 2.68x | 1.94x |

Reading the tables:

* forward+backward beats the eager baseline by 2.5-21.7x (native) and 6.5-48x
  (Triton); forward alone by 1.5-2.2x (native) and 2.5-3.6x (Triton) from T=16 on;
* Triton is 1.6-2.5x faster than native everywhere. `tl.dot` lowers to wgmma with
  a software pipeline, while the native path is hand-written `mma.sync` over an
  unswizzled shared-memory layout - the choice that keeps it bitwise identical to
  `det_gemm`;
* forward at T=1 is the one regression: native 0.785 ms against the baseline's
  0.452 ms (Triton is at parity, 0.475 ms). A six-row payload still pays four
  grouped launches and their per-CTA segment resolution; the operator wins from
  T=16 on;
* concentrated routing touches six experts, so both providers read only non-empty
  experts' weights (T=1: 114 MiB of the 2.4 GiB stack) while the baseline loops
  over all 128 - which is why the baseline loses most in that column.

### Roofline

Measured on the same GPU rather than read from the spec sheet: cuBLAS bf16
8192^3 peaks at **513.6 TFLOP/s**, a 2 GiB device copy sustains **3.51 TB/s**, so
the ridge point is **146 FLOP/byte**.

At T=4096 (24 576 packed rows, 256 expert row tiles) every grouped GEMM has an
arithmetic intensity of 159-179 FLOP/byte - only 1.1-1.2x the ridge - so the
roofline for each stage is its *compute* time, about 0.48 ms:

| Stage | GFLOP | GB | AI | roofline | native | of roofline |
| --- | --- | --- | --- | --- | --- | --- |
| up | 245.2 | 1.37 | 179 | 0.48 ms | 3.52 ms | 13.6% |
| down | 245.2 | 1.54 | 159 | 0.48 ms | 3.33 ms | 14.3% |
| dZ | 245.2 | 1.46 | 168 | 0.48 ms | 4.03 ms | 11.8% |
| dPacked | 245.2 | 1.41 | 174 | 0.48 ms | 4.07 ms | 11.7% |
| dW_up | 245.2 | 1.41 | 174 | 0.48 ms | 4.65 ms | 10.3% |
| dW_down | 245.2 | 1.37 | 179 | 0.48 ms | 4.49 ms | 10.6% |
| whole forward+backward | 1962 | 8.47 | 232 | 3.82 ms | 53.9 ms | 7.1% |

Byte counts assume one weight read per expert per stage (a 9.98 MB expert stack
fits the 50 MB L2) plus the activations that stage writes. The Triton provider
lands at 14.6% of the same roofline (26.2 ms).

So the operator is **not bandwidth-bound**: at these sizes it is marginally
compute-bound, and both providers are an order of magnitude short of their own
compute ceiling. Two explanations survived the schedule sweep below:

1. **The instruction path, read off the SASS.** The native kernel disassembles to
   612 `HMMA` (mma.sync), 224 `LDSM` (ldmatrix) and a 1552-byte local stack for the
   leaf tree at 81 registers per thread; the Triton `_grouped_rows_kernel`
   disassembles to `HGMMA.64` (wgmma) with 8 `LDSM`, no stack and 210 registers,
   because wgmma consumes A and B straight from shared memory through descriptors.
   That drops the ldmatrix traffic and the bank-conflict exposure in one step, and
   it is the difference the schedule sweep could not reach.
2. **No swizzle.** The tensor maps are pinned to
   `CU_TENSOR_MAP_SWIZZLE_NONE` because the `ldmatrix` addressing is plain
   row-major, so a `[BK,BM]` or `[BK,BN]` tile strides 256/128 bytes per row and
   the transposed-operand `ldmatrix.trans` hits a full bank-conflict set.
   `det_gemm` makes the same trade, which is what keeps the two bitwise identical.
   Swizzle pins the inner box at 128 B, so it would force a 64-wide K tile and a
   re-derived addressing scheme for all four operand layouts.

At decode sizes the same measurement says the opposite - T=1 moves 120 MB and
does 0.16 GFLOP, so its roofline is ~0.03 ms against 0.785 ms measured (4%):
that column is latency- and padding-bound, which is why fixing the grid bound and
fusing the combine helped there and the GEMM schedule did not.

### Where the native time goes

`torch.profiler` at T=4096, one forward+backward iteration, per kernel:

| Kernel (native) | ms / launch | launches | Shared-memory layout |
| --- | --- | --- | --- |
| `gemm<MODE_WGRAD>` | 4.65 | 2 | transposed A + KN B, `ldmatrix.trans` |
| `gemm<MODE_WGRAD_R2>` | 4.49 | 2 | same, relu-square on B |
| `gemm<MODE_DX, EPI>` (dZ) | 4.03 | 2 | KN B |
| `gemm<MODE_DX>` (dPacked) | 4.07 | 2 | KN B |
| `gemm<MODE_UP>` | 3.52 | 2 | NK B |
| `gemm<MODE_DOWN>` | 3.33 | 2 | NK B, relu-square on A |
| `dweights_leaf` | 0.456 | 1 | FP32 route dot |
| `dy` | 0.201 | 1 | - |
| `combine` | 0.100 | 1 | - |
| `dweights_merge` | 0.030 | 1 | - |
| total in kernels | **50.6** | | wall 53.9 |

The same profile on the Triton provider totals 26.0 ms, with each grouped GEMM
at 1.9-2.0 ms per launch; the native gap is therefore entirely in the six grouped
GEMM launches, worst for the weight gradients. Effective throughput at T=4096,
counting 479 MFLOP per token: native 40 TFLOP/s, Triton 75 TFLOP/s.

Nsight Compute cannot be used here (`ERR_NVGPUCTRPERM`: an unprivileged container
may not open the GPU performance counters), so this section is built from
`torch.profiler` kernel attribution, `cuobjdump -res-usage` and the parameter
sweep below - all reproducible without elevated permissions.

### Schedule sweep

Every knob below leaves the arithmetic untouched - the leaf stays 32-wide and the
tree is unchanged - so each row is a one-constant rebuild, measured at T=4096,
forward+backward, median of 15 wall-clock iterations:

| BM | BN | K per TMA tile | stages | wall ms | TFLOP/s | of 3.82 ms roofline |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 64 | 32 (1 leaf) | 2 | **49.04** | 40.0 | 7.8% |
| 128 | 64 | 32 | 3 | 49.00 | 40.0 | 7.8% |
| 128 | 64 | 32 | 4 | 48.77 | 40.2 | 7.8% |
| 128 | 64 | 64 (2 leaves) | 3 | 54.74 | 35.8 | 7.0% |
| 128 | 64 | 128 (4 leaves) | 2 | 57.87 | 33.9 | 6.6% |
| 128 | 64 | 128 (4 leaves) | 3 | 68.01 | 28.8 | 5.6% |
| 128 | 128 | 32 | 2 | 49.40 | 39.7 | 7.7% |
| 64 | 64 | 32 | 2 | 58.39 | 33.6 | 6.5% |
| 256 | 64 | 32 | 2, 8 warps | 49.86 | 39.4 | 7.7% |
| 128 | 64 | 32 | 2, 8 warps | 62.00 | 31.6 | 6.2% |

* **Barriers were not the bottleneck.** One TMA tile feeding four leaves cuts the
  barrier count 4x and *loses* 15%; pipeline depth 2 -> 4 changes nothing. The
  first-principles "168 barriers per tile" estimate was wrong: the barrier waits
  overlap with the other CTA resident on the SM.
* Wider K tiles lose because the `ldmatrix` bank-conflict degree follows the
  shared-memory row stride - 64 B gives 8-way, 128 B and 256 B give 16-way. Only
  TMA swizzle removes that, and swizzle caps the inner box at 128 B.
* BN=128 halves the CTA count and doubles A-fragment reuse, but it also doubles
  the register accumulator; the two effects cancel.
* A second provider-level contrast, from `cuobjdump -sass`: native is
  `HMMA`+`LDSM` at `REG:81 STACK:1552`, Triton is `HGMMA.64` at `REG:210 STACK:0`.
  The wgmma path trades registers and occupancy for the whole ldmatrix pipeline.
* Shipped configuration: `REG:81 STACK:1552` per thread with 24.6 KB of dynamic
  shared memory. The 1552-byte stack is the leaf tree itself. With the 8-register
  allocation granularity that is 88 registers per thread, so the occupancy
  formula allows 5 CTAs per SM by registers against 9 by shared memory: registers
  bind, and the kernel runs at 20 of the 64 warp slots (31%).
* That register ceiling explains the whole table. Warp count per SM is
  `65536 / (registers * 32)`, independent of the CTA shape, so any change that
  enlarges the accumulator (BN=128, BM=256) loses the occupancy it gains, and any
  change that shrinks the tile (BM=64, 8 warps at BM=128) pays more per-CTA
  prologue and more B-fragment traffic for the same warps.

Closing the remaining 2x therefore means changing the instruction path (wgmma) or
the shared-memory contract (swizzle), both of which would break
`test_matches_det_gemm_bitwise` unless `det_gemm` changes with it. That is a
versioned arithmetic decision, not a tuning fix, so the schedule stays where the
measurements put it.

The per-row FP64 report (`--qualify-max-tokens 128`, both providers) gives 8.7e-3
to 1.3e-2 relative L2 for the forward and the worst gradient. One T=1 sample
reaches 2.6e-2 because a single route's dot is compared against a near-zero FP64
sum; the unit suite, which qualifies at 17 tokens, asserts the same bounds.

## Verification record

Build and measurements on one NVIDIA H100 NVL (SM90, 95 GiB), CUDA 13.0,
PyTorch 2.11.0+cu130, `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1`:

* `pytest tests/backends/cuda/test_nemotron_experts_cuda.py` - 25 passed, no
  skips: both providers, the FP64 qualification, bitwise batch and relocation
  invariance, graph replay, autocast, provider selection and rejection, and the
  four `det_gemm` bitwise parity cases;
* `pytest tests/backends/cuda/test_sm90_linear_wrappers.py
  tests/backends/cuda/test_nemotron_router_cuda.py` - 88 passed (the det_gemm and
  router contracts are unchanged);
* `pytest tests/runtime/test_kernel_registry.py
  tests/reference/test_nemotron_experts_reference.py
  tests/reference/test_nemotron_router_reference.py
  tests/backends/rocm/test_nemotron_experts_rocm.py` - 40 passed, 5 skipped (the
  ROCm cases skip without a HIP device);
* the benchmark above ran to completion twice - once on the first build and again
  after the combine fusion and the grid-bound change - with three sweeps the second
  time (normal order for 1-4096 and 8192-32768, `--reverse` for 1-4096); every row
  carries its provider id, weight coverage, `n_e` distribution and FP64 report. The
  two builds agree to within 2.8% per row; the tables quote the second run;
* `mkdocs build --strict` succeeds; `mypy --ignore-missing-imports` is clean on
  the four new modules (the repository keeps 14 pre-existing errors elsewhere).

The ROCm column has no hardware record: the same Triton source runs on CUDA, but
a gfx9xx host must re-qualify it before those numbers are quoted.
