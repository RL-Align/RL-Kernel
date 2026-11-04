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

The native provider carries two instruction paths behind one contract (see
[Instruction paths](#instruction-paths)): an `mma.sync` path and, by default, a
`wgmma` path. They are bitwise identical, so the provider id and the contract
version do not change with the path.

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
`BK = K_TREE_LEAF = 32` is one FP32 leaf, each leaf is two `k16` MMA steps
accumulated in FP32 and rounded once to BF16, and leaves merge with the
canonical midpoint tree. No split-K, no atomics, no TF32, no autotuning.
`tests/backends/cuda/test_nemotron_experts_cuda.py` pins the up and down
projections bitwise against `_C.det_gemm_*`, and the same file pins `dPacked`
and both weight gradients.

Weights are stacked as `[E,F,H]`/`[E,H,F]`, so one tensor map covers all experts
and expert `e` owns rows `[e*F, (e+1)*F)`. Two tensor maps are built per GEMM.

| Stage | Output | Cast |
| --- | --- | --- |
| up | `U [6T,1856]` BF16 and `Z = bf16(relu(U)**2)` | leaf and tree rounding; `Z` in two packed instructions |
| down | `Y [6T,2688]` FP32 | BF16 tree value widened, one rounding |
| shared | `sh_out [T,2688]` FP32 | same up/down rules, `E=1` |
| combine + exit | `out = bf16(sum_slots w*Y + sh_out)` | the operator's only cast |
| dY | `bf16(fp32(w) * fp32(dOut))` | one rounding |
| dZ, dPacked | BF16 | leaf and tree rounding |
| dW | BF16 | grouped rows ascending |
| dWeights | `[T,6]` FP32 | 32-wide leaf, fixed FP32 tree |

`relu(x)**2` is a BF16 operation on the up projection's already-rounded value. A
BF16 product is exact in FP32, so `bf16(relu(fp32(x))**2)` and `__hmul2` after
`__hmax2` produce identical bits. The up epilogue therefore materializes `Z`
itself - the down projection and `dW_down` consume it as an operand - instead of
each of them transforming every operand tile it loads. The transform is then
paid once per element rather than once per element per 128-wide output tile.
The relu-square derivative in the backward epilogue is `2 * relu(U)` multiplied
once and rounded once.

## Instruction paths

Both paths compute the arithmetic above and are **bitwise identical**; the choice
is an instruction-path decision, not a numerical one.

| | v1 (`sm90_v1`) | v2 (`sm90_v2`, default) |
| --- | --- | --- |
| Tile | `BM=128`, `BN=64`, `BK=32` | `BM=64`, `BN=128`, `KT=64` (two leaves) |
| MMA | two `mma.sync.m16n8k16` per 32-wide leaf, per warp tile | two `wgmma.mma_async.m64n128k16` per 32-wide leaf, per CTA |
| Shared-memory layout | row-major, TMA swizzle none | TMA `SWIZZLE_128B`, wgmma descriptors |
| Operands | `ldmatrix` / `ldmatrix.trans` | read straight from shared memory by wgmma |
| Pipeline | 2 TMA stages, `__syncthreads` in the K loop | 3 stages, producer warp + 4-warp consumer warpgroup, mbarriers only |
| SASS | `HMMA 32`, `LDSM 12`, `REG 66-82`, `STACK 1552` | `HGMMA 6`, `LDSM 0`, `REG 135` (`170` dW), `STACK 1568` |
| Dynamic shared memory | 24.6 KB | 73.7 KB |
| Occupancy ceiling | 5 CTAs/SM by registers (31% of warp slots) | 3 CTAs/SM (`2` for dW) by registers and shared memory (23%, 16%) |

`RL_KERNEL_NEMOTRON_EXPERTS_BACKEND=sm90_v1|sm90_v2|auto` selects the path
(`auto` is v2) and is read per launch, so one process can measure both. v2 is a
warp-specialised wgmma pipeline applied per expert segment, self-contained in
`csrc/cuda/moe/nemotron_experts.cu` (its descriptor encoding, wgmma wrappers,
swizzled tensor map and barrier primitives are that file's own), with three
grouped additions:

* **Operand majors.** A wgmma operand is K-major when K is contiguous and
  MN-major otherwise, and the kernel derives the descriptor start offsets and the
  instruction's `imm-trans` flag from the mode: the up/down projections are
  K-major on both sides, `DX` reads the stacked weights as an MN-major `[K,N]`
  operand, and a weight gradient is MN-major on both sides (`[rows,M]` and
  `[rows,N]`). An MN-major 128-wide B operand occupies two 64-wide blocks one
  8 KB block apart, which is the descriptor's leading byte offset; the stride
  byte offset is 1024 B for both majors, because in both cases a core matrix is
  eight 128-byte rows.
* **Segment resolution is unchanged.** The CTA still resolves its
  `(expert, row tile)` from the 129 prefix sums in shared memory.
* **Weight-gradient tail padding.** A segment length is not a multiple of the
  32-wide leaf, so the last reduction tile zeroes the rows past the segment end in
  shared memory before the wgmma reads it, with a generic-to-async proxy fence
  and a warpgroup barrier. Rows past the segment belong to the next expert and
  must not join the reduction.

The Triton provider folds leaves in ascending order instead of merging them with
the midpoint tree. Its frontend rejects plain-Python helpers inside a kernel, and
`tl.static_range` rejects the list operations the stack algorithm needs, so a
trace-time stack is not expressible; the alternative, encoding the pop pattern as
eight nested static conditionals, buys a per-leaf association difference that the
contract does not claim. Leaf width, the single BF16 rounding per leaf and the
fixed ascending order are all preserved and every kernel folds the same way, so
batch invariance is unaffected; only the association of the leaf sums differs
from the native tree. `dWeights` folds 32-wide FP32 leaves the same way.

* **Weight-gradient reduction depth is bounded.** A weight gradient reconciles one
  32-wide leaf per 32 rows of an expert segment, and the midpoint tree keeps one
  merge slot per level, so `TREE_DEPTH = 14` covers `kd <= 16384` leaves. The
  operator's own bound (`rows <= 6 * MAX_TOKENS = 393216`, i.e. `kd <= 12288`) is
  inside it, and a hand-built call beyond it raises instead of writing past the
  merge stack. The guard uses the row count, never `offsets`, so the host still
  does not read the routing table and CUDA Graph capture stays legal.
* **Gradients are shaped like their inputs.** `dW_sh1`/`dW_sh2` are `[F,H]`/`[H,F]`
  (not `[1,F,H]`), and the shared suite asserts the shape of every gradient
  against its input.
* **A zero-token rank still returns shaped gradients.** An expert-parallel rank
  can hold no tokens; `out` is then `[0,H]`, payload gradients are empty rows, and
  every requested weight gradient is an exact zero of the weight's shape rather
  than an absent tensor. Both providers implement this and the suite asserts it.

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

## Fail-closed gates

Non-SM90 device, a build without `KERNEL_ALIGN_DET_GEMM_SM90=1`, a dtype, rank or
contiguity mismatch, widths not 64-aligned, `tokens > 65536`, an `offsets` table
that is not `experts + 1` long, unpaired shared weights and a reduction depth
beyond the merge tree all raise. `permutation`/`offsets` *values* are trusted as
the router's contract rather than validated on device - a device-side check would
force a host synchronization and break graph capture - so a hand-built dispatch
must supply a bijection and its exclusive prefix sums.

## Qualification boundary

Within one provider, an identical token row and its upstream gradient keep the
same bits across batch sizes and positions, for `out`, `dPacked`, `dX` and
`dWeights`. `dW` repeats for an identical ordered token set; adding
independently computed microbatch `dW` is not a bitwise full-batch equivalence
claim. The two native instruction paths *are* bitwise equal to each other on
every output and gradient - they differ in how the tile is computed, not in how
it is reduced - which
`test_instruction_paths_agree_bitwise` asserts on the core geometry (including
per-expert segment lengths that are not multiples of the leaf) and on the
Nemotron geometry. The two providers do **not** claim bitwise equality: each is
qualified against FP64 on its own selected branch, exactly as the CUDA and
Triton `det_gemm` implementations coexist.

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
`test_instruction_paths_agree_bitwise` runs both native instruction paths in one
process and compares every output and gradient with `torch.equal`.

## Performance reproduction

The baseline runs the same math expert by expert with `torch.matmul` in FP32 and a
weighted, slot-ascending combine - the obvious eager formulation. Its expert span
list is read once outside the timed region. The candidate rows are the provider
objects themselves, so the public validation wrapper is outside both sides of the
comparison.

**The production grouped-GEMM path is measured for the forward, and it is 1.7-2.8x
faster than this operator.** `torch._grouped_mm` - the cuBLAS/CUTLASS grouped GEMM
a serving stack calls - is in the tables below as `grouped_mm`, measured in the
same paired run with the stacked weights pre-transposed outside the timed region
and the same slot-ascending combine. That gap is not an implementation defect and
is not addressed by any schedule change: the grouped GEMM keeps an FP32
accumulator and casts once per output element, while the contract rounds and
tree-merges every 32-wide leaf, which is `K/32` extra SIMT passes per element. See
[Why v2 is the contract's ceiling](#why-v2-is-the-contracts-ceiling).

What is **not** measured: Transformer Engine `GroupedLinear` (the repository pins
`transformer_engine==2.19.0`; it is not installed and does not build against this
container's CUDA 13.0 + PyTorch 2.11.0+cu130), vLLM's fused MoE (not installed),
and a production *training* baseline - `torch._grouped_mm` groups over token rows
only, so the weight gradients (`dW_up[e] = dU_eᵀ·packed_e`, whose reduction length
differs per expert) are not expressible with it, and the forward+backward tables
therefore still compare against the eager formulation.

```bash
# forward: the two native paths, Triton, the eager formulation and the
# production grouped-GEMM path, in one paired run
PYTHONPATH=. python benchmarks/operators/moe/benchmark_nemotron_experts_training.py \
  --tokens 1 16 128 513 1024 4096 8192 --dtype bf16 \
  --providers native triton --baselines eager grouped_mm --phases forward \
  --distributions random concentrated --native-backend sm90_v2 \
  --output /tmp/experts-fwd-v2.json
# same sweep with the mma.sync path:
PYTHONPATH=. python benchmarks/operators/moe/benchmark_nemotron_experts_training.py \
  --tokens 1 16 128 513 1024 4096 8192 --dtype bf16 --providers native triton \
  --baselines eager grouped_mm --phases forward \
  --native-backend sm90_v1 --output /tmp/experts-fwd-v1.json
# forward+backward (the eager formulation only: a grouped weight gradient needs a
# grouped wgrad kernel, which torch._grouped_mm does not expose)
PYTHONPATH=. python benchmarks/operators/moe/benchmark_nemotron_experts_training.py \
  --tokens 1 16 128 513 1024 4096 8192 --dtype bf16 --providers native triton \
  --baselines eager --phases forward_backward --distributions random concentrated \
  --native-backend sm90_v2 --qualify-max-tokens 128 --output /tmp/experts-bwd-v2.json
```

`--baselines grouped_mm` is `torch._grouped_mm`, the cuBLAS/CUTLASS grouped GEMM a
serving stack calls: 2D operand plus a `[E,N,K]` stack and the group ends, with the
stacked weights materialized in the layout each projection needs outside the timed
region. It is the *production* comparison RFC #434 §4.1 asks for, it is
**forward only** (its weight gradients would need a grouped wgrad kernel), and it
does not implement the leaf-rounding contract, so it measures what the contract
costs rather than a better implementation of it. Transformer Engine
`GroupedLinear` (pinned 2.19.0) and vLLM's fused MoE are not installed in this
container and were not run. Forward and
forward+backward are measured separately: 20 alternating paired samples of five
executions, median reported, CUDA Graph replay, cotangents prepared outside
timing. `--reverse` repeats the sweep with the reversed initial provider order;
the two orders agree to a median 0.4% difference (worst sampled cell 2.8%), so the
tables below show the normal order only. The script refuses to overwrite an
existing output file. Every column of every table below comes from one sequential
session on an idle device at its boost clock (`clocks.sm` 1785 MHz of 1785); the
32768-token rows are a separate invocation because four candidates at that size
exceed the memory budget, and their baselines agree with the 1-8192 run to 1%.

### Forward, random routing, median ms

`grouped_mm` is the production path; `grouped_mm/v2` below 1 means the contract
kernel is slower, which it is everywhere in this direction.

| T | baseline | grouped_mm | native v1 | native v2 | Triton | v1/v2 | grouped_mm/v2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.477 | 0.209 | 0.782 | 0.358 | 0.473 | 2.18x | 0.58x |
| 16 | 4.277 | 0.737 | 3.558 | 1.416 | 1.818 | 2.51x | 0.52x |
| 128 | 10.088 | 1.147 | 5.925 | 2.315 | 3.106 | 2.56x | 0.50x |
| 513 | 11.823 | 1.328 | 6.268 | 2.480 | 3.390 | 2.53x | 0.54x |
| 1024 | 14.210 | 1.535 | 6.608 | 2.914 | 3.670 | 2.27x | 0.53x |
| 4096 | 29.604 | 3.587 | 15.200 | 9.850 | 8.967 | 1.54x | 0.36x |
| 8192 | 49.075 | 6.903 | 28.293 | 19.081 | 16.470 | 1.48x | 0.36x |
| 32768 | 164.884 | 27.742 | 107.390 | 72.175 | 65.414 | 1.49x | 0.38x |

Under concentrated routing the same columns give `grouped_mm/v2` = 0.64 / 0.51 /
0.41 / 0.42 / 0.38 / 0.35 / 0.37 / 0.42, i.e. the production path is 1.6-2.9x
faster there too.

### Forward + backward, random routing, median ms

| T | baseline | native v1 | native v2 | Triton | v1/v2 | baseline/v2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.655 | 4.365 | 3.595 | 1.773 | 1.21x | 4.63x |
| 16 | 217.770 | 10.688 | 7.127 | 5.124 | 1.50x | 30.56x |
| 128 | 403.635 | 16.789 | 10.803 | 8.323 | 1.55x | 37.36x |
| 513 | 414.544 | 18.510 | 11.237 | 8.983 | 1.65x | 36.89x |
| 1024 | 425.932 | 21.360 | 12.624 | 10.647 | 1.69x | 33.74x |
| 4096 | 503.735 | 51.599 | 32.914 | 26.341 | 1.57x | 15.30x |
| 8192 | 602.836 | 95.971 | 58.188 | 49.231 | 1.65x | 10.36x |
| 32768 | 1201.012 | 357.091 | 215.282 | 184.260 | 1.66x | 5.58x |

### Forward + backward, concentrated routing, median ms

| T | baseline | native v1 | native v2 | Triton | v1/v2 | baseline/v2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.762 | 4.314 | 3.564 | 1.775 | 1.21x | 4.70x |
| 16 | 17.422 | 4.429 | 3.717 | 1.772 | 1.19x | 4.69x |
| 128 | 18.825 | 4.761 | 4.045 | 1.934 | 1.18x | 4.65x |
| 513 | 24.737 | 8.882 | 6.445 | 3.789 | 1.38x | 3.84x |
| 1024 | 31.542 | 12.987 | 9.314 | 5.679 | 1.39x | 3.39x |
| 4096 | 73.493 | 44.880 | 28.405 | 22.144 | 1.58x | 2.59x |
| 8192 | 135.217 | 89.930 | 55.296 | 45.252 | 1.63x | 2.45x |
| 32768 | 496.592 | 356.957 | 218.603 | 183.355 | 1.63x | 2.27x |

### Peak memory

Peak allocation increment over the shared weight stack, random routing, MiB
(`torch.cuda.max_memory_allocated` around one captured call, per candidate):

| Phase | T | baseline | grouped_mm | native (both paths) | Triton |
| --- | --- | --- | --- | --- | --- |
| forward | 1 | 38 | 0.2 | 0.1 | 0.1 |
| forward | 1024 | 148 | 127 | 138 | 109 |
| forward | 8192 | 913 | 1010 | 1094 | 862 |
| forward+backward | 1 | 3878 | - | 2496 | 2496 |
| forward+backward | 1024 | 8775 | - | 2713 | 2684 |
| forward+backward | 8192 | 10613 | - | 4349 | 4117 |

The forward+backward rows are dominated by the saved FP32 `down` (`[6T,2688]`) and
the BF16 `up` and `Z` (`[6T,1856]` each); `grouped_mm` keeps no saved state because
it is forward only. Both native paths allocate identically - both materialize `Z`,
which is 182 MiB of the 8192 forward+backward row.

Reading the tables:

* the wgmma path is 1.2-2.6x faster than the mma.sync path: 1.4-2.6x for forward,
  1.2-1.7x for forward+backward, with the smallest gain in the decode-sized
  backward where launch and segment-resolution work dominates;
* forward+backward beats the eager baseline by 2.3-37.4x (v2) and 2.7-47x (Triton);
  forward alone by 1.3-4.9x (v2) and 1.0-3.6x (Triton);
* the production grouped GEMM is 1.7-2.8x faster than this operator in the
  forward (1.6-2.9x concentrated) - the price of the leaf-rounding contract, not
  of the schedule. Triton, which also skips the per-leaf rounding, sits between
  the two at 1.2-1.4x of the native path;
* forward at T=1 is not a regression: the `mma.sync` path's 0.782 ms against a
  0.477 ms eager baseline was 1.6x slower, the wgmma path is 0.358 ms, 1.33x
  faster than the baseline and only 1.7x off `grouped_mm` (0.209 ms).
  `BM=64` starts 6 CTAs instead of 3 on a six-route payload, and the narrower tile
  is the only change that matters at that size;
* concentrated routing touches six experts, so every provider reads only non-empty
  experts' weights (T=1: 114 MiB of the 2.4 GiB stack) while the baseline loops
  over all 128 - which is why the baseline loses most in that column.

### Roofline

Measured on the same GPU rather than read from the spec sheet: cuBLAS bf16
8192^3 peaks at **513.6 TFLOP/s**, a 2 GiB device copy sustains **3.51 TB/s**, so
the ridge point is **146 FLOP/byte**.

At T=4096 (24 576 packed rows, 256 expert row tiles) every grouped GEMM has an
arithmetic intensity of 159-179 FLOP/byte - only 1.1-1.2x the ridge - so the
roofline for each stage is its *compute* time, about 0.48 ms:

| Stage | GFLOP | GB | AI | roofline | v1 ms | of roofline | v2 ms | of roofline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up | 245.2 | 1.37 | 179 | 0.48 ms | 5.69 | 8.4% | 3.25 | 14.8% |
| down | 245.2 | 1.54 | 159 | 0.48 ms | 5.23 | 9.2% | 3.20 | 15.0% |
| dZ | 245.2 | 1.46 | 168 | 0.48 ms | 6.39 | 7.5% | 3.48 | 13.8% |
| dPacked | 245.2 | 1.41 | 174 | 0.48 ms | 6.26 | 7.7% | 3.44 | 13.9% |
| dW_up | 245.2 | 1.41 | 174 | 0.48 ms | 6.16 | 7.8% | 4.71 | 10.2% |
| dW_down | 245.2 | 1.37 | 179 | 0.48 ms | 6.01 | 8.0% | 4.60 | 10.4% |
| whole forward+backward | 1962 | 8.47 | 232 | 3.82 ms | 47.6 | 8.0% | 29.5 | 13.0% |

The per-stage columns are one routed launch at T=4096 from the same
`torch.profiler` pass that produces the wall time in the last row; the headline
tables above are CUDA-graph replays over 20 samples, which is why their whole-row
medians are 1.1x higher. Byte counts assume one weight read per expert per stage
(a 9.98 MB expert stack fits the 50 MB L2) plus the activations that stage
writes. The Triton provider lands at 14.6% of the same roofline (26.2 ms).

So the operator is **not bandwidth-bound**: at these sizes it is marginally
compute-bound, and neither provider reaches its own compute ceiling. The
[schedule sweep](#schedule-sweep) below is what ruled out the v1 path's schedule;
two structural causes remained, and both are addressed by the v2 path:

1. **The instruction path, read off the SASS.** The v1 kernel disassembles to
   `HMMA` (mma.sync) + `LDSM` (ldmatrix) with a 1552-byte local stack for the leaf
   tree at 66-82 registers per thread; wgmma consumes A and B straight from
   shared memory through descriptors, so v2 drops the whole ldmatrix stage
   (`HGMMA 6`, `LDSM 0` per tile).
2. **No swizzle.** v1's tensor maps are pinned to `CU_TENSOR_MAP_SWIZZLE_NONE`
   because the `ldmatrix` addressing is plain row-major, so a `[BK,BM]` or
   `[BK,BN]` tile strides 256/128 bytes per row and every `ldmatrix` core hits a
   bank-conflict set; v2 pins `SWIZZLE_128B`, which forces the 64-wide, two-leaf
   K tile and a re-derived descriptor for all four operand layouts.

Both are instruction-path decisions, not arithmetic ones: the grouped kernels
assert that the wgmma path reproduces the mma.sync path bit for bit
(`test_instruction_paths_agree_bitwise`), and the `det_gemm` parity cases still
pin the shared arithmetic.

At decode sizes the same measurement says the opposite - T=1 moves 120 MB and
does 0.16 GFLOP, so its roofline is ~0.03 ms against 0.34 ms measured (9% for v2,
v1 was 4%): that column is latency- and padding-bound, which is why the grid
bound and the fused combine helped there and the GEMM schedule did not. The v2
schedule helps it a different way - `BM=64` gives 6 CTAs instead of 3 on a
six-route payload, and the 2.17x forward gain at T=1 is mostly that.

### Where the native time goes

`torch.profiler` at T=4096, one forward+backward iteration, per routed launch
(the shared expert's six launches are the same six modes at `T` rows instead of
`6T`, and total 10.3 ms in v1 and 6.0 ms in v2):

| Kernel | launches | v1 ms / launch | v2 ms / launch | v1/v2 |
| --- | --- | --- | --- | --- |
| `gemm<MODE_UP>` | 2 | 5.69 | 3.25 | 1.75x |
| `gemm<MODE_DOWN>` | 2 | 5.23 | 3.20 | 1.63x |
| `gemm<MODE_DX, EPI>` (dZ) | 2 | 6.39 | 3.48 | 1.83x |
| `gemm<MODE_DX>` (dPacked) | 2 | 6.26 | 3.44 | 1.82x |
| `gemm<MODE_WGRAD>` (dW_up) | 2 | 6.16 | 4.71 | 1.31x |
| `gemm<MODE_WGRAD>` (dW_down) | 2 | 6.01 | 4.60 | 1.31x |
| `dweights_leaf` | 1 | 0.47 | 0.44 | - |
| `dy` | 1 | 0.20 | 0.20 | - |
| `combine` | 1 | 0.10 | 0.10 | - |
| `dweights_merge` | 1 | 0.03 | 0.03 | - |
| total in kernels | | **47.2** | **29.7** | 1.59x |

The weight gradient is now the slowest stage and the one that gains least: its
`K` is the expert segment length, so it pays the tail padding above and runs
`REG 170` (two CTAs per SM) against 135 for the other modes. Effective throughput
at T=4096, counting 479 MFLOP per token: v1 41 TFLOP/s, v2 66.5 TFLOP/s, Triton
75 TFLOP/s.

### Why v2 is the contract's ceiling

The leaf epilogue is SIMT work the contract mandates: per output element, one
BF16 rounding of the 32-wide leaf and one BF16 tree merge, i.e. one packed
instruction per two elements for each of the two operations. At T=4096 the
forward+backward performs 1962 GFLOP, which is 30.7e9 element-leaves (64 FLOP
each), so one instruction per element-leaf per thread needs 0.96e12
warp-instructions/s. This GPU issues 132 SM x 4 schedulers x 1.785 GHz = 0.94e12,
and the measured 29.5 ms corresponds to 1.04e12 element-leaves/s. **The kernel is
at the SIMT instruction-issue ceiling** for the epilogue the contract demands,
which is the same wall that keeps `det_gemm` at 14% of peak.

That also explains the residual gap to Triton (26.3 ms, 1.2x) and the larger gap to
the production grouped GEMM (3.59 ms against 9.85 ms at T=4096, 2.7x in the
forward): both keep an FP32 accumulator and cast once per output element instead of
once per 32-wide leaf, so they do `K/32` fewer SIMT passes - they are faster
precisely because they compute a different (non-contract) reduction. That measured
2.7x is the contract's price, and it is the number a fixed performance ratio
against native would have to be set against. Going further under this contract
means a wider leaf or a single cast, which is an arithmetic decision, not a
schedule one.

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

No knob in this space moves the arithmetic, so the sweep found v1's optimum - and
that optimum is 8% of the roofline. Everything the sweep could reach is exhausted:
the remaining gap needed the two structural changes above (wgmma and swizzle), and
`RL_KERNEL_NEMOTRON_EXPERTS_BACKEND=sm90_v2` is exactly those two. They do not
break `test_matches_det_gemm_bitwise`: the wgmma path reproduces v1's reduction
bit for bit, which the parity test in
`tests/backends/cuda/test_nemotron_experts_cuda.py` asserts.

The per-row FP64 report (`--qualify-max-tokens 128`, both providers) gives 8.7e-3
to 1.3e-2 relative L2 for the forward and the worst gradient. One T=1 sample
reaches 2.6e-2 because a single route's dot is compared against a near-zero FP64
sum; the unit suite, which qualifies at 17 tokens, asserts the same bounds.

## Verification record

Build and measurements on one NVIDIA H100 NVL (SM90, 95 GiB), CUDA 13.0,
PyTorch 2.11.0+cu130, `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1`:

* `pytest tests/backends/cuda/test_nemotron_experts_cuda.py` - 28 passed, no
  skips: both providers, the FP64 qualification, bitwise batch and relocation
  invariance, graph replay, autocast, provider selection and rejection, the four
  `det_gemm` bitwise parity cases, and the three `mma.sync`-vs-`wgmma` bitwise
  parity cases (core geometry at 17 and 129 tokens, Nemotron geometry at 64);
* `pytest tests/ops/gemm/test_det_gemm.py
  tests/backends/cuda/test_sm90_linear_wrappers.py
  tests/backends/cuda/test_nemotron_router_cuda.py
  tests/runtime/test_kernel_registry.py
  tests/reference/test_nemotron_experts_reference.py
  tests/reference/test_nemotron_router_reference.py
  tests/backends/rocm/test_nemotron_experts_rocm.py` - 150 passed, 54 skipped
  (49 of the skips are `det_gemm`'s "Triton is unavailable" and ROCm `gfx942`
  rows in this build, 5 are the ROCm provider suite without a HIP device). No file
  outside this operator's own sources changed, so those suites report what they
  reported before;
* `torch.profiler` at T=1024 and T=4096 for both paths; the per-kernel table and
  the 1.59x kernel-time ratio above come from that pass;
* `cuobjdump -res-usage` / `-sass` on the built extension for the numbers in
  [Instruction paths](#instruction-paths) and
  [Where the native time goes](#where-the-native-time-goes);
* the benchmark above ran four times (native v2, native v1, native v2 + Triton at
  1-8192, and the same pair at 32768); every row carries its provider id, weight
  coverage, `n_e` distribution and FP64 report. The baselines of the v1 and v2 runs
  agree to 0.1%; the `--reverse` order agrees to a median 0.4%;
* `black --check --line-length 100` is clean on the three python files touched and
  `mypy --ignore-missing-imports` is clean on the wrapper; `mkdocs build --strict`
  succeeds.

Re-validation after the review fixes (`H100 80GB HBM3`, CUDA 13.0, PyTorch
2.12.0+cu130, same build flags): operator + reference suites 45 passed; the
`det_gemm` / router / wrapper / registry / ROCm neighbours 135 passed, 54 skipped;
`black --check`, `mypy --ignore-missing-imports` and `mkdocs --strict` clean. The
instruction-path ratio does not move on that card (T=4096 forward+backward: wgmma
27.5 ms against mma.sync 43.5 ms, 1.58x against the 1.57-1.61x above), so the
wider merge stack (`TREE_DEPTH` 12 -> 14) costs nothing measurable. The absolute
latencies in the tables remain the H100 NVL figures.

The ROCm column has no hardware record: the same Triton source runs on CUDA, but
a gfx9xx host must re-qualify it before those numbers are quoted. The same applies
to the two native instruction paths - both are SM90 code paths, but only SM90 was
measured.
