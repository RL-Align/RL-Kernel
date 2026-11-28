# MiniMax-H3 Partial Multimodal RoPE

`h3_mm_rope_3axis_partial` applies the H3 position tables to Q or K after QK RMSNorm.
It pairs channel `i` with `i + 48` for `0 <= i < 48` and leaves channels `96:128`
unchanged. The caller generates the FP32 cosine/sine tables from packed `(t, h, w)`
positions: 16 frequencies per axis, concatenated to 48 and duplicated to 96.
The same tables can be reused for Q and K.

## Interface

| Value | Shape | Dtype / requirements |
|---|---|---|
| `x` | `[B, S, H, 128]` | BF16 or FP32; nonempty; strided views accepted |
| `cos`, `sin` | `[S, 96]` | FP32 constants on x's device; no gradients; strided views accepted |
| Output, input gradient | `[B, S, H, 128]` | x's dtype; contiguous; last 32 channels pass through unchanged |

H3 uses 56 heads, and head shards can use fewer. Every head has 128 channels.
The kernel processes its 48 rotary pairs in a 64-lane vector and masks the last
16 lanes. The remaining 32 channels pass through directly.

Each direction launches one kernel with grid `(S, H, B)`, putting long sequences
on the first axis. If the input and both tables are contiguous, `layout=None`
lets the kernel reuse the output row offset. For views, the launcher passes the
element strides of the input, cosine table and sine table in one runtime tuple.
Older Triton versions take the same strides as individual arguments. Backward
uses the actual layout of `grad_y`, including broadcast strides. Head and batch
counts must fit the device's grid-y and grid-z limits, respectively.

```python
from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

# q: [B, S, H, 128], cos/sin: precomputed FP32 [S, 96] H3 tables on q.device
op = TritonH3MMRoPEOp()
q_rotated = op(q, cos, sin)
```

`NativeH3MMRoPEOp` in `rl_engine.reference.rope.h3_mm_rope` provides the matching
PyTorch implementation on CPU or GPU. The direct Triton entry accepts tensor
device types `cuda`, `hip`, `xpu` and `musa` with the corresponding PyTorch and
Triton backends installed. Forward and backward launch Triton directly on the
tensor's device and current stream. PyTorch on ROCm normally reports `cuda`.
Hardware validation for H3 in this PR covers NVIDIA CUDA. The supported
activation dtypes are BF16 and FP32.

The legacy registry name is `h3_mm_rope_3axis_partial`: CUDA prefers Triton and
can fall back to Native; the other platform maps expose Native. For strict H3
integration, use semantic backend `rlkernel.h3_mm_rope.triton`. Its descriptor
declares CUDA, BF16/FP32, training/rollout and WS1/TP1/CP1, records the arithmetic
profile, and rejects unsupported requirements without fallback. Head-sharded
operator tests do not qualify a distributed execution profile. Registration is
dispatch metadata, not evidence of full-model acceptance.

## Rounding and Gradients

Forward preserves the pinned [Diffusers H3 `_apply_rotary_emb`](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py#L56)
arithmetic, including BF16 cast points. The original expression is
`x_rotary * cos.to(x.dtype) + rotate_half(x_rotary) * sin.to(x.dtype)`:

1. Convert the FP32 tables to x's dtype.
2. Compute each rotation product, rounding it to x's dtype.
3. Add/subtract the two products and round the output to x's dtype.

Training, `no_grad`, and inference use the same Triton rotation for both layouts. FP fusion is
disabled so multiplying and adding cannot become a single FMA with different rounding.
For BF16, the kernel performs arithmetic in FP32 registers but explicitly casts
each product to BF16 and back before addition. Loading into FP32 alone would not
preserve the model's product rounding. For FP32 inputs these casts are no-ops;
the multiply and add are still separate FP32 operations.

Backward uses the **same quantized coefficients** as forward, promotes them and the
upstream gradient to FP32, and computes the transpose rotation:

```text
dx[i]      = dy[i] * cos[i]      + dy[i + 48] * sin[i + 48]
dx[i + 48] = dy[i + 48] * cos[i + 48] - dy[i] * sin[i]
```

The products and sum remain FP32; only the final input gradient is cast to x's dtype.
This follows the usual autograd treatment of floating-point casts, while avoiding
the eager BF16 backward's extra product rounding. The PyTorch reference therefore
has an explicit backward too. The table halves are read separately, and the last
32 gradient channels pass through bit-for-bit. There are no atomics or reductions
across tokens, heads or batches. Both Native and Triton mark their backward
with `@once_differentiable`. A second backward through RoPE raises an error.
In a sum with another differentiable term, a call to `torch.autograd.grad`
targeting a specific input can follow just the other term and return its
contribution. Both operators support only first-order gradients for `x`.

## Two references and acceptance

The accuracy check compares the operator with an independent PyTorch rotation
that reproduces H3's intermediate rounding. A second comparison measures how
much that rounding changes the result relative to an all-FP32 rotation. Both
comparisons use the project's elementwise tolerances. See the
[H3 RoPE review](https://github.com/RL-Align/RL-Kernel/pull/656#issuecomment-7069638929) and
the earlier H3 references in [#487](https://github.com/RL-Align/RL-Kernel/pull/487) and
[#489](https://github.com/RL-Align/RL-Kernel/pull/489#discussion_r4395799781).

| Reference | Coefficients | Products / sum | Role |
|---|---|---|---|
| `forward_fp32` | Round each coefficient to the activation dtype, then promote to FP32 | Round each FP32 product to the activation dtype, then sum in FP32 | Determines whether operator accuracy passes |
| `forward_unrounded_fp32` | Use the original FP32 tables | Multiply and sum in FP32 | Reports mathematical error |

The accuracy checker runs the reference in PyTorch eager mode. Its `forward` rounds
each product to `storage_dtype` and adds them in FP32. Calling `backward()` or
`torch.autograd.grad` applies the transpose rotation with the saved coefficients in
FP32 and passes the last 32 gradient channels through directly. Tests check the
rotating channels against an independent PyTorch rotation with the same coefficients.
`forward_fp32` takes its storage dtype from `x.dtype` by default. The unified
checker passes BF16 inputs for BF16 cases, so their leaf gradients round once.
The standalone tests and benchmark also check the gradient before that final
cast. They pass an FP32 leaf and specify the original activation dtype via
`storage_dtype` so the forward still rounds at the model's boundaries.

FP32 is used **before** each storage boundary, preserving the model operation
order. Substituting FP64 arithmetic before a BF16 midpoint can select a different
stored value, as demonstrated in H3 [#485](https://github.com/RL-Align/RL-Kernel/pull/485).
No project tolerances, dtype support or other operators' precision contracts change.

### Two BF16 outputs that exceed the unrounded FP32 tolerance

Pytest replays the two BF16 inputs captured at `[1, 73426, 14, 128]`, seed 656.
It checks that both outputs match H3 and that both differences from
`forward_unrounded_fp32` exceed the existing tolerance.

| Token / head / channel | Model = Native = Triton | Unrounded FP32 | Absolute error | Allowed error |
|---|---:|---:|---:|---:|
| 36201 / 13 / 38 | -0.125000 | -0.101508379 | 0.023491621 | 0.021624133 |
| 48538 / 3 / 11 | 0.328125 | 0.302912712 | 0.025212288 | 0.024846602 |

In each pair the two products nearly cancel, making the BF16 rounding of the
coefficients and products visible in their difference. Each token rotates
independently. A longer sequence samples more input pairs. Tests also cover
normal and near-cancelling inputs at scales 1, 2 and 4.

## Validation

Hard assertions cover provider-forward bitwise parity, Native/Triton equality,
the declared backward, and model-boundary accuracy for BF16/FP32. Coverage also
includes repeatability, training/inference, batch/sequence prefixes, packed-token
permutations, head shards, unrelated batch changes, strided inputs/tables/gradients,
broadcast gradients, checkpointing, streams, CUDA Graphs and long sequences
`65535 / 65536 / 73426`. Reference tests independently pin the forward cast points,
FP32 VJP and mathematical reference, including nonduplicated sine-table halves.

```bash
python -m pytest tests/ops/rope/test_h3_mm_rope.py \
  tests/validation/operators/test_operator_inputs.py \
  tests/benchmarks/test_h3_mm_rope_benchmark.py \
  tests/runtime/test_kernel_registry.py \
  tests/models/minimax_h3/test_h3_one_block.py \
  tests/models/minimax_h3/test_h3_review_fixes.py -q -rs
python tools/validation/operators/check_operator.py \
  --op h3_mm_rope_3axis_partial --candidate triton --device cuda \
  --dtype bf16 --batch 2 --seq 264 --n-heads 56 --check-grad --json
```

Use `--candidate pytorch --device cpu` for Native and `--dtype fp32` for FP32.
The checker calls `NativeH3MMRoPEOp.forward_fp32` for both dtypes. Case names use
the standard operator, dtype and shape format. The checker generates synthetic
three-axis positions. The benchmark and block replay use packed FL2VA positions.

The [H3 block replay](h3-one-block.md) binds Triton to `rope_q` and `rope_k`.
It adds an isolated model-boundary accuracy result for those nodes on the provider's
actual inputs. The summary records the first failing node in
`first_rope_acceptance_failure`, or `null` when both pass. Its full-block mathematical
golden includes
unrounded RoPE, so historical block-error comparisons remain meaningful. Existing
block tests still enforce their comparative error bounds; they are distinct from
the operator's elementwise acceptance gate.

Historical A6000 evidence found two complete-block forward failures at the existing
`norm1` provider-bitwise check. Replacing only RoPE with the original provider left
every node's forward hash unchanged and reproduced the RMSNorm discrepancies.
These are separate from RoPE's local acceptance and are not waived. See the PR's
current run report for fresh hardware, test counts, block attribution and remaining
qualification. CI or single-operator acceptance does not establish full-model,
ROCm or distributed acceptance.

## Benchmark and figures

```bash
python -m benchmarks.models.benchmark_h3_mm_rope --dry-run
python -m benchmarks.models.benchmark_h3_mm_rope \
  --warmup 10 --repeat 40 --output-dir reports/h3-mm-rope
python tools/benchmarking/plot_h3_mm_rope.py \
  reports/h3-mm-rope/results.json --out reports/h3-mm-rope/figures
```

The 16-case plan covers `[1, S, H, 128]`, packed `S=264/925/3160/73426`,
`H=14/56`, BF16/FP32, each representing one Q or K tensor. It measures forward,
backward and forward+backward for the original eager H3 provider, Native and
Triton. Original BF16 autograd rounds backward products earlier; Native and Triton
use the declared FP32 intermediate backward. Both baselines are reported explicitly.

Every shape must first pass provider-forward bitwise parity, Native/Triton
forward/backward bitwise parity and independent model-boundary accuracy. A failed
acceptance case is not timed, leaves `complete=false` and yields nonzero exit.
The JSON field `unrounded_math_diagnostic` names `forward_unrounded_fp32` as its
reference and sets `is_acceptance_gate=false`. It records element counts, values
outside tolerance, maximum error, maximum error/limit and up to eight global
outlier coordinates for forward, candidate backward and provider backward.
Timing can proceed after the accuracy and bitwise checks pass, even when this
diagnostic reports outliers.

Timing uses eager public wrappers, 10 warmups and 40 accelerator-event samples,
without `torch.compile` or CUDA Graph timing. Allocation and autograd dispatch are
included; input/table creation, compilation and validation are excluded. Backward
reuses a prebuilt graph; forward+backward builds a fresh graph each call. CPU
intra-op threads are one and provider order alternates. Events include host
execution gaps, so small eager cases can be noisy. JSON retains all samples,
medians, standard deviations and incremental peak allocation (excluding inputs
and prebuilt graphs).

The plotting script renders all measured shapes, separately for BF16/FP32 and
F/B/F+B, with latency in milliseconds, baseline/Triton speedup, and incremental
peak allocation in MiB. All vertical axes are linear. The long layout has its
own latency and memory panels so the shorter layouts stay readable. Latency whiskers are P10/P90
sample dispersion, not confidence intervals. Raw measurements, source hashes,
logs and figures are PR evidence attachments rather than committed data.
