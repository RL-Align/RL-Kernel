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

H3 uses 56 heads; head shards can use fewer. Sequence length and tensor strides are
runtime metadata. Triton specializes head count and channel geometry, and launches
with grid `(S, H, B)` so real packed sequences longer than 65,535 tokens do not occupy
CUDA's limited grid-y axis. Batch and head counts must fit the device's grid-y/z limits.

```python
from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

# q: [B, S, H, 128], cos/sin: precomputed FP32 [S, 96] H3 tables on q.device
op = TritonH3MMRoPEOp()
q_rotated = op(q, cos, sin)
```

`NativeH3MMRoPEOp` in `rl_engine.reference.rope.h3_mm_rope` provides the matching
PyTorch implementation on CPU or GPU. The direct Triton entry accepts PyTorch's
`cuda` device type (also used by ROCm), uses the input device/current stream, and
has no automatic fallback. NVIDIA CUDA is the platform tested here; ROCm remains
unqualified. FP16 is not supported in this version.

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

Training, `no_grad`, and inference use the same Triton forward kernel. FP fusion is
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
across tokens, heads or batches. Only first-order gradients for x are part of the contract.

## Validation

`tests/ops/rope/test_h3_mm_rope.py` separates three checks:

- **Forward path:** bitwise comparison with the pinned H3 provider expression,
  including full-head/sharded geometry and noncontiguous inputs/tables.
- **Backward path:** bitwise comparison with autograd applied to an independent
  FP32 rotation using forward-quantized coefficients, followed by the final cast.
  A rounding regression distinguishes this from both eager BF16 backward and an
  unquantized-coefficient backward.
- **Independent accuracy:** original FP32 coefficients and FP32 arithmetic, for
  both output and gradient, using the project's `elementwise` tolerances from
  `rl_engine.contracts.numerical`. This reference does not copy the candidate's
  coefficient quantization or product rounding.

Both supported dtypes also check repeatability, training/inference equality, batch
and sequence-prefix invariance, position-aligned token permutations, head shards,
unrelated batch changes, strided views, and broadcast upstream gradients. Checkpoint
recomputation, nondefault streams and CUDA Graph replay cover the autograd wrapper.
GPU launch regression covers `S = 65535, 65536, 73426`. Comparisons use the same
logical inputs and position tables; differing positions are not expected to agree.

```bash
python -m pytest tests/ops/rope/test_h3_mm_rope.py \
  tests/validation/operators/test_operator_inputs.py \
  tests/benchmarks/test_h3_mm_rope_benchmark.py \
  tests/runtime/test_kernel_registry.py \
  tests/models/minimax_h3/test_h3_one_block.py \
  tests/models/minimax_h3/test_h3_review_fixes.py -q -rs
```

The unified operator checker also exposes `h3_mm_rope_3axis_partial`:

```bash
python tools/validation/operators/check_operator.py \
  --op h3_mm_rope_3axis_partial --candidate triton --device cuda \
  --dtype bf16 --batch 2 --seq 264 --n-heads 56 --check-grad --json
```

Use `--candidate pytorch --device cpu` for a CPU check, or `--dtype fp32` for
FP32 inputs. This CLI uses seeded synthetic three-axis positions; the benchmark
and block replay use actual packed FL2VA layouts. `NativeH3MMRoPEOp.forward_fp32`
is the independent gold entry: original FP32 coefficients, autocast disabled,
FP32 products/addition and FP32 output. Its autograd path is independent of the
candidate's explicit backward. The operator's accuracy tests additionally use
FP32 input leaves to keep reference gradients in FP32 until comparison.

The [H3 block replay](h3-one-block.md) now calls the Triton candidate for `rope_q`
and `rope_k`. Its original provider and FP32 golden remain independent. With the
pinned block weights and required GPU backends available:

```bash
RL_KERNEL_H3_WEIGHTS=<dir> python -m pytest tests/models/minimax_h3/test_h3_one_block.py -q -rs
```

The 2026-11-04 RTX A6000 run reported **104 passed, 6 skipped** with Python 3.12.3,
PyTorch 2.8.0+cu128 and Triton 3.4.0. Skips were the two-GPU test and five block /
real-weight checks requiring optional native extensions or checkpoint artifacts.
Existing B200 block reports used provider-only RoPE and cannot qualify this binding.
Full block/model and ROCm qualification remain separate work.

The subsequent CPU completion checks also cover the unified checker and saved
accuracy counterexamples. The two-GPU test now covers both BF16 and FP32.
These additions still need the next GPU regression; they do not alter the
candidate's forward/backward arithmetic or requalify the historical benchmark.

### BF16 accuracy finding from the wider benchmark

The wider benchmark found two forward elements outside the unchanged project
`elementwise` tolerance (`atol=0.02`, `rtol=0.016`) at `[1, 73426, 14, 128]`, seed
656. Triton, Native and the original provider have identical forward results;
all backward elements in the diagnostic passed. One example is:

| Quantity | Value |
|---|---:|
| Token / head / channel | 36201 / 13 / 38 |
| Provider = Native = Triton | -0.125 |
| Independent FP32 result | -0.101508379 |
| Absolute error | 0.023491621 |
| Allowed absolute + relative error | 0.021624133 |

The error comes from the original model's BF16 coefficient/product rounding near
cancellation. Matching the provider does not establish independent accuracy.
This version retains the original model's forward rounding and records these
points as an open accuracy acceptance issue, not an accuracy waiver. Both pairs
are retained as exact-value pytest fixtures. Model-forward parity and the
backward contract remain hard assertions. Their independent forward-accuracy
test is a **strict xfail**, with the unresolved contract conflict named in the
reason; an unexpected pass fails the suite until the marker is reconsidered.
The benchmark records the failure and does not time that case or mark the whole
run qualified.
Thresholds and sampled inputs are not changed to hide it.

The acceptance decision remains open: should H3 RoPE's accuracy reference retain
the model's specified coefficient/product storage boundaries, as some other H3
references do, or should its training and rollout arithmetic both change to an
all-FP32 intermediate path? The current implementation does neither implicitly;
it retains model rounding and reports the unrounded FP32 discrepancy separately.

### Remaining GPU qualification

- Run the complete operator/registry/benchmark regression on a two-GPU NVIDIA
  machine, including the noncurrent-device cases for both supported dtypes.
- Build the existing H3 CUDA extension without fast math, fetch the pinned
  conditioning and block-0 weights with `tools/weights/prepare_h3_weights.py
  --out <dir> --block`, and run the block tests above with `RL_KERNEL_H3_WEIGHTS`
  set. Missing extensions/weights must not be counted as successful integration.
- Retain the independent BF16 failure until the arithmetic/reference contract
  is agreed with the maintainer. A successful repeatability test is not an
  accuracy acceptance result.

## Benchmark

```bash
python -m benchmarks.models.benchmark_h3_mm_rope --dry-run
python -m benchmarks.models.benchmark_h3_mm_rope \
  --warmup 10 --repeat 40 --output-dir reports/h3-mm-rope
```

The default plan covers packed lengths 264, 925, 3160 and 73426, heads 14 and 56,
and BF16/FP32. Each input represents one Q or K tensor. Cosine/sine tables use
the pinned H3 layout/position generator; generation is outside timing.

Three providers are measured for forward, backward and forward+backward:

| Provider | Purpose |
|---|---|
| Original H3 eager expression + autograd | Production-expression baseline; BF16 backward has its original extra product rounding |
| `NativeH3MMRoPEOp` | Matching forward and explicit FP32-intermediate backward |
| `TritonH3MMRoPEOp` | Strictly resolved candidate; actual backend and arithmetic fingerprint recorded in JSON |

Before timing, each shape must pass provider-forward bitwise parity, Native
forward/backward bitwise parity and independent FP32 accuracy. Failed shapes are
recorded and skipped; any failure leaves `complete=false` and a nonzero exit code.
Eager public calls include wrapper allocation and autograd dispatch, without
`torch.compile` or CUDA Graph capture. Backward reuses a prebuilt graph;
forward+backward builds a fresh graph per call. CPU intra-op threads are set to one.
Accelerator-event samples include host dispatch gaps and alternate provider order.
JSON retains raw samples, medians, standard deviations and incremental peak
allocation (excluding inputs and prebuilt graphs). Source hashes and complete
validation logs are kept as PR attachments rather than committed benchmark data.

The A6000 run timed 15 of 16 shapes, producing 135 provider/mode measurements;
the failed BF16 14-head long case above was not timed. At `[1, 73426, 56, 128]`:

| Dtype | Mode | Original eager model ms | Native ms | Triton ms | Original / Triton |
|---|---|---:|---:|---:|---:|
| BF16 | Forward | 21.658 | 21.666 | 8.857 | 2.45x |
| BF16 | Backward | 29.300 | 40.310 | 9.158 | 3.20x |
| BF16 | Forward + backward | 50.759 | 61.769 | 17.641 | 2.88x |
| FP32 | Forward | 36.443 | 36.483 | 12.819 | 2.84x |
| FP32 | Backward | 53.883 | 34.746 | 13.243 | 4.07x |
| FP32 | Forward + backward | 90.066 | 70.825 | 25.544 | 3.53x |

All entries in this table passed the shape's checks before timing. Relative sample
standard deviations for these entries were below 1.4%. Smaller eager measurements
were noisy (60 of 135 measurements had standard deviation above 15% of their
median), so they are preliminary and the raw samples must accompany comparisons.
These are single-operator A6000 measurements, not H100 or model throughput claims;
the unresolved BF16 accuracy finding prevents overall qualification.
