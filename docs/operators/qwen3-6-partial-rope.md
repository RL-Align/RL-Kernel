# Qwen3.6-35B-A3B Partial RoPE

## Summary

This operator applies **text full-attention RoPE for Qwen/Qwen3.6-35B-A3B** to
one query or key tensor. It rotates the first 64 of 256 channels and copies the
other 192 channels unchanged. It is a separate entry point from the existing
full-dimension `rope` operator; existing Qwen3 callers keep their interface.

The [model configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
uses architecture `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`, head
dimension 256, partial rotary factor 0.25, 16 query heads and 2 KV heads.
`qwen3_6_partial_rope` currently names this model-targeted implementation, not a claim that
every Qwen3.6 checkpoint has the same contract.

This supplies an operator for [#642, task 11](https://github.com/RL-Align/RL-Kernel/issues/642).
Engine wiring, MTP/verification position adapters, multimodal position generation,
and end-to-end Megatron/SGLang equivalence remain separate integration work.

## Entry Point

```python
import torch

from rl_engine.runtime.registry import kernel_registry

# Example caller-owned text tables. Real engines must agree on frequency rules,
# position IDs and the actual FP32 table values supplied to both sides.
device = "cuda:0"
num_positions = 1024
frequencies = 1.0 / (10_000_000.0 ** (torch.arange(32, device=device).float() / 32))
angles = torch.arange(num_positions, device=device).float()[:, None] * frequencies
cos, sin = angles.cos(), angles.sin()  # [1024, 32], FP32 constants

# Two packed text segments: positions 10..12 and positions 0..1.
positions = torch.tensor([10, 11, 12, 0, 1], device=device, dtype=torch.int64)
q = torch.randn(5, 16, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(5, 2, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
rope = kernel_registry.get_op("qwen3_6_partial_rope", device=q.device)
q_rotated = rope(q, cos, sin, positions)
k_rotated = rope(k, cos, sin, positions)

# Decode/verification can use a subset of the same rows. Do not renumber positions.
q_one_token = rope(q[2:3], cos, sin, positions[2:3])
torch.testing.assert_close(q_one_token, q_rotated[2:3], rtol=0, atol=0)
(q_rotated.float().sum() + k_rotated.float().sum()).backward()
```

For explicit backend selection, import `NativeQwen3_6PartialRoPEOp` from
`rl_engine.reference.rope.qwen3_6_partial_rope`, or `TritonQwen3_6PartialRoPEOp` from
`rl_engine.backends.shared.triton.rope.qwen3_6_partial_rope`.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Shared Triton | `TritonQwen3_6PartialRoPEOp` | `_qwen3_6_partial_rope_fwd_kernel`, `_qwen3_6_partial_rope_bwd_kernel` | Current review fixes validated on NVIDIA H100 (SM90), BF16/FP32; prior RTX 5080 (SM120) coverage is historical. |
| PyTorch reference/fallback | `NativeQwen3_6PartialRoPEOp` | Eager PyTorch operations | Validated on NVIDIA H100, BF16/FP32, with CPU contract coverage in the cloud suite. |

PyTorch uses the `cuda` tensor API for both NVIDIA CUDA and AMD ROCm. The same
device guard can select either backend; this does **not** establish ROCm numerical
or performance qualification. No MUSA/Ascend-specific kernel is provided here.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[T, H, 256]` | BF16 or FP32 | One Q or K tensor; token/head counts can vary. |
| `cos`, `sin` | `[P, 32]` | FP32 | Same shape/device; `requires_grad=False`. |
| `positions` | `[T]` | Int32 or int64 | Table row IDs, with `0 <= positions[t] < P`. |
| Output | `[T, H, 256]` | Same as `x` | New contiguous storage; no input mutation. |
| Upstream gradient / `dx` | `[T, H, 256]` | Same as `x` | Only first-order input gradients. |

`T` is the number of tokens in this call, possibly packed from several sequences.
`positions[t]` selects the coefficient row for token `t`; it is not a sequence
boundary marker. `P` is the number of table rows, not token length. Nonempty
inputs need nonempty tables, and valid IDs require `P > max(positions)`.

All tensors share a device. Noncontiguous inputs/tables/indices and zero-stride
views are accepted; `grad_y` can have its own strides. Empty token/head axes
return empty outputs and gradients without a Triton launch. Triton's grid is
`(T, H)`, with `H <= 65535` to respect the second launch-axis limit.
Channel and coefficient offsets use 64-bit multiplication before pointer addition,
including views whose element offsets exceed `INT32_MAX`.

Metadata validation never synchronizes to inspect GPU index values. Bounds are
a caller precondition. With `TRITON_DEBUG=1`, `tl.device_assert` diagnoses invalid
IDs. Without debug, masked table loads produce NaNs in the rotated prefix and
leave the tail unchanged; they do not read outside the tables. The Native path
uses PyTorch `index_select` and its normal invalid-index errors. Behavior on
invalid IDs is not a cross-backend equivalence guarantee.

## Dispatch Behavior

`kernel_registry.get_op("qwen3_6_partial_rope", device=x.device)` tries the shared Triton
implementation on CUDA/ROCm and falls back to Native if the backend cannot be
imported. Other platform entries select the Native implementation only. Platform
selection is not validation evidence. The registry does not catch a bad input or
silently retry a failed kernel launch. Tests and benchmarks use explicit classes
so backend fallback cannot be mistaken for Triton execution.

## Accuracy

The arithmetic contract is the same in forward and backward:

```text
x:   [ first 32 | second 32 |                 tail 192                 ]
          a_i        b_i                 copied without arithmetic
           \        /
    c_i = cos[positions[t], i]       s_i = sin[positions[t], i]

y_first  = FP32(a_i) * c_i - FP32(b_i) * s_i
y_second = FP32(b_i) * c_i + FP32(a_i) * s_i
             separate FP32 products and addition/subtraction
                              |
                     one cast to x.dtype

dx_first  = FP32(dy_first) * c_i + FP32(dy_second) * s_i
dx_second = FP32(dy_second) * c_i - FP32(dy_first) * s_i
dx_tail   = dy_tail
```

Both Triton launches disable implicit FMA (`enable_fp_fusion=False`). Coefficients
and products are never quantized to BF16. Only the rotated results are converted
to the input dtype; the tail is copied directly, preserving signed zeros and NaN
payloads. The transpose rotation needs only saved tables/positions, not `x`.

The Native production path uses an explicit first-order VJP with the same
rounding boundaries. `NativeQwen3_6PartialRoPEOp.forward_fp32` computes FP32 output without
final quantization, and PyTorch differentiates that FP32 expression independently.
For an unrounded gradient diagnostic, pass an FP32 leaf; a BF16 leaf receives a
BF16 gradient regardless of output dtype. No relaxed thresholds or expected
precision failures are introduced. Both forward and gradient checks use the
existing `elementwise` rows in the numerical contract.

The Native candidate and FP32 diagnostic share the forward rotation helper, so
that comparison alone is not an independent mathematical oracle. The separate
FP64 2x2 rotation-matrix test validates pairing, signs, positions and the transpose
gradient without using that helper. The production VJP is also checked against
autograd on the FP32 expression.

Unlike H3 MM-RoPE's model-defined intermediate quantization, this operator chooses
FP32 intermediates. It does **not** claim bitwise equivalence with the unfused
Megatron path that first converts coefficients/products to BF16, or with arbitrary
fused kernels using a different FMA policy. Both engines must use this arithmetic
contract to align. Supplied tables, inputs and positions must also match.

For fixed input values and table rows, each token/head pair is independent:
prefill, single-token decode, chunking, padding or a head shard cannot change the
rotation arithmetic. The tests check this operator boundary; they do not prove
full-model train/rollout log-probability agreement.

## Performance Notes

```bash
python benchmarks/operators/rope/benchmark_qwen3_6_partial_rope.py --dry-run
python benchmarks/operators/rope/benchmark_qwen3_6_partial_rope.py \
  --tokens 1 32 256 2048 8192 --heads 2 16 --dtypes bf16 fp32 \
  --output-dir artifacts/qwen3-6-partial-rope
```

The benchmark compares eager Native PyTorch and the public Triton wrapper for
forward, backward and forward+backward. It checks output and input-gradient
accuracy before timing. CUDA/HIP event medians and sample standard deviations are
reported in microseconds, with extra peak allocated memory in MiB. For each mode,
providers alternate order every warmup and measured round (PyTorch/Triton, then
Triton/PyTorch). JSON retains each sample's latency, round and execution position.
No CUDA Graph or `torch.compile` is used. Small cases include host launch gaps, so these are
public-call timings rather than isolated arithmetic instruction timings.

Input/table generation, correctness checks and JIT compilation are outside
timing. Forward runs without autograd; backward reuses a prebuilt graph;
forward+backward builds a new graph each call without accumulating leaf `.grad`.
Memory is measured separately for each provider, after collecting retired autograd
objects; its inputs and prebuilt backward graph are excluded from the baseline.
The probe rejects changing live allocations or a nonpositive peak. JSON records
the GPU/runtime/compiler versions, source hashes (including uncommitted code),
seed, repetitions and numerical-contract fingerprint. Git metadata is optional.
Measured JSON/Markdown belongs in `artifacts/`, not in the PR source diff.

The review-fix regression used one cloud NVIDIA H100 80GB HBM3 (SM90), PyTorch
2.8.0+cu128 and Triton 3.4.0. Both full runs used 30 warmups and 100 samples per
provider/mode; dtype order was reversed in the repeat. All 20 dtype/shape cases
passed output/gradient gates in both runs. At `[8192,16,256]`, the repeat measured:

| Dtype | Mode | Native median us | Triton median us | Native / Triton | Native extra MiB | Triton extra MiB | Triton / Native memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | Forward + backward | 2293.952 | 1755.360 | 1.31x | 210 | 128 | 61.0% |
| FP32 | Forward + backward | 1896.736 | 1604.224 | 1.18x | 290 | 256 | 88.3% |

These are eager public-call medians, not isolated kernel or model-throughput
measurements. Host launch gaps and timing noise are substantial: 74/120 groups
in the first run and 57/120 in the repeat had sample SD above 15% of the median.
Across matched rows, median absolute drift was 2.26%, with extremes -89.24% to
+861.02%. Both complete runs and raw sample order/latencies are retained in the
PR evidence. Every memory row matched between runs. These data are regression
evidence and do not establish a stable hardware-wide speedup.

## Tests

```bash
python -m pytest tests/ops/rope/test_qwen3_6_partial_rope.py \
  tests/benchmarks/test_qwen3_6_partial_rope_benchmark.py -q -rs
python -m pytest tests/runtime/test_kernel_registry.py \
  tests/validation/operators/test_operator_inputs.py -q
python tools/validation/operators/check_operator.py --op qwen3_6_partial_rope \
  --candidate pytorch --device cpu --dtype bf16 --batch 2 --seq 7 --n-heads 2 --check-grad
```

Tests cover an independent FP64 rotation-matrix oracle, FP32 accuracy, both model
head counts, repeated/offset positions, chunk/padding/head-shard invariance,
strides, empty inputs, tail bits, metadata errors, dispatch, and two-device
forward/backward restoration. A BF16 large-channel-stride regression exercises
forward/backward offsets beyond `INT32_MAX`; it needs about 8.5 GiB of free GPU
memory and skips explicitly on smaller devices. GPU cases skip on CPU; those
skips are not GPU passes. Benchmark tests include wrong-forward and wrong-backward negative controls.

## Known Limitations

- Text position IDs only; no multimodal t/h/w assembly or frequency generation.
- Fixed D=256 and R=64, split-half pairing; no FP16, arbitrary rotation range,
  inplace execution, coefficient gradients or higher-order input gradients.
- The current fixes passed a one-H100 cloud regression (227 passed, 9 skipped:
  2 two-device cases and 7 Ascend cases). The large-stride regression executed.
  Two-device restoration passed on two RTX 5080s before these review fixes; it
  was not rerun on two devices for this revision. ROCm remains unqualified;
  MUSA/Ascend acceleration is outside this implementation.
- Megatron/SGLang adapters, full-model correctness, distributed/CP position
  alignment, and every #642 verification mode require separate integration evidence.
