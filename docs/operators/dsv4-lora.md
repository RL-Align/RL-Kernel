# DeepSeek V4 shared BF16 LoRA on ROCm

This implements item 45 of [P5-B #520](https://github.com/RL-Align/RL-Kernel/issues/520):
the shared BF16 LoRA delta and its input/adapter gradients. The frozen base
projection is supplied by the caller. Full expert-module integration is item 48.

The composition follows [PR #416](https://github.com/RL-Align/RL-Kernel/pull/416),
which merged into `dsv4-p5-dev`. This implementation uses the current repository
layout and explicitly calls the FP32-output Triton det_gemm path. The ordinary
BF16 det_gemm path now rounds at intermediate reduction-tree nodes and cannot
be substituted without changing this operator's arithmetic.

## Entry points

```python
from rl_engine.ops.moe.lora_delta import shared_grouped_lora_delta
from rl_engine.models.deepseek_v4.lora import lora_projection

# LoRA only; output is FP32. Alpha is the effective multiplier.
delta = shared_grouped_lora_delta(x, a, b, alpha=0.5, recompute=True)

# base_output is FP32 and must retain its input-gradient graph.
output = lora_projection(x, a, b, alpha=0.5, base_output=base_output)

# Base only.
output = lora_projection(x, None, None, alpha=0.5, base_output=base_output)
```

`alpha` is applied once, with no implicit division by rank. A configuration
using `lora_alpha / rank` must pass that already-computed multiplier.
`shared_grouped` means one A/B adapter shared over already-dispatched rows.
The API does not accept a bank of per-expert adapters or perform EP dispatch.

## Tensor contract

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| X | M × K | BF16 | Original, unquantized local input rows |
| A | R × K | BF16 | First shared adapter matrix |
| B | N × R | BF16 | Second shared adapter matrix |
| U | M × R | BF16 | Saved or recomputed intermediate |
| delta | M × N | FP32 | Scaled LoRA result |
| dY | M × N | BF16 or FP32 | Incoming gradient |
| dX, dA, dB | Shapes of X, A, B | FP32 | Explicit backend VJP outputs |

Operands must be strided 2-D tensors on one device. Noncontiguous inputs are
accepted; det_gemm materializes contiguous operands. K, N and R must be positive.
M may be zero; the empty-batch parameter gradients are zero. Alpha must be a
finite scalar representable in FP32. Arithmetic is not clipped to prevent BF16
overflow; callers must keep their activation and gradient ranges appropriate.

## Arithmetic and backward

Every GEMM below uses `TritonDetGemmOp.forward_fp32`: fixed 64 × 64 × 32 tiles,
FP32 accumulation, no split-K, no autotuning and TF32 disabled.

```text
U     = BF16(GEMM_FP32(X, A.T))
delta = GEMM_FP32(U, B.T) * alpha

dV = BF16(FP32(dY) * alpha)
dU = BF16(GEMM_FP32(dV, B))
dB = GEMM_FP32(dV.T, U)
dA = GEMM_FP32(dU.T, X)
dX = GEMM_FP32(dU, A)
```

The custom autograd wrapper uses this explicit VJP, rather than det_gemm's
generic backward. Its BF16 X/A/B input edges convert returned FP32 gradients to
BF16 when PyTorch accumulates `.grad`. Call `RocmLoRADelta.backward` directly
when the enclosing expert implementation needs FP32 partial gradients for its
own reduction. Second-order differentiation is not supported.

`base_output + delta` is added in FP32. No route weight or final FC2 cast is
performed here. The caller freezes the base weights and supplies its dX path;
detaching `base_output` would incorrectly remove that contribution. The adapter
operator itself neither receives base weights nor computes base dW. Combining
BF16 autograd edges is not a substitute for a future explicit FP32 expert dX
merge; that integration remains in item 48 and P6.

## Saved tensors and updates

By default, backward saves X, A, B and U. With `recompute=True`, it saves X/A/B
and recreates only U with the same projection. PyTorch's saved-tensor checks
reject in-place updates to any saved operand between forward and backward.
Adapter values and transposes are not cached: the next forward reads the current
weights after an optimizer step. Updates must use ordinary version-tracked
PyTorch operations; mutating `.data` bypasses these checks and is unsupported.

This is local autograd lifetime handling. Rollout weight synchronization,
distributed version barriers and deployment repacking remain separate work.

## Backend and validation status

The explicit ROCm entry point accepts gfx942 and gfx950 as qualification targets
(MI300X and MI355X). It has not been GPU-qualified by this change. No automatic
dispatcher registration or CPU/CUDA fallback is added. The profile name is
`dsv4-lora-rocm-fp32-v1`; it describes this composition, not numerical equivalence
to the CUDA BF16 reduction tree or to another GPU architecture/compiler version.

`ReferenceLoRADelta` accumulates products serially in FP32. It is an independent
diagnostic implementation, not a byte-level oracle for a tiled GPU reduction.
CPU tests use it to check casting, first-order autograd, base/LoRA composition,
mutation detection and two successive adapter updates. Finite differences check
the smooth FP64 LoRA formula; BF16 rounding is checked separately.

The ROCm suite checks actual kernel execution, FP64 staged accuracy, tails,
empty rows, noncontiguous operands, saved/recomputed gradients and two updates.
It compares the same row's output, U and dX bytes at M=1/16/17/64/65 and different
row positions, plus chunking and zero padding. dA/dB depend on all rows, so their
test repeats an identical complete batch. It does not claim that changing
microbatch partitions, token order or gradient accumulation yields identical
parameter gradients. Accuracy checks allow a 1% relative norm and maximum-error
bound against the staged FP64 calculation because U/dU round to BF16; byte
invariance checks have zero tolerance and are a separate gate.

## Run the tests

CPU checks, without ROCm or Triton:

```bash
python -m pytest tests/models/deepseek_v4/test_lora.py -q
```

On an MI300X or MI355X machine with ROCm PyTorch and Triton installed, from the
repository root:

```bash
RLK_REQUIRE_ROCM=1 python -m pytest tests/backends/rocm/test_dsv4_lora.py -v -s
```

The GPU command prints the GPU, architecture, PyTorch, HIP, Triton and numerical
profile. `RLK_REQUIRE_ROCM=1` fails when the required hardware is unavailable,
instead of reporting an all-skipped run as acceptance. Keep the git commit and
that output with the test result. MI300X and MI355X need separate runs.

No performance result is claimed yet. These tests cover the isolated LoRA path;
quantized base GEMMs, full expert training, EP/TP/PP/CP integration, Graph capture
and verl/vime end-to-end consistency still require their own task-level tests.
