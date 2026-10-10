# AdaLN modulation (Qwen-Image WS1, #386)

`kernel_registry.get_adaln_modulation_op(device, hidden=H)` returns `(op, trace)`. Call
`modulated, gate = op(x, modulation, eps=1e-6)` with `x: [B,S,H]`
and shared `modulation: [B,3H]`. Both inputs must share device and dtype (FP32
or BF16). The chunks are **shift, scale, gate**. Each row uses LayerNorm over
H with no affine parameters, then `norm(x) * (1 + scale) + shift`. The gate is
`[B,1,H]` and remains connected to `modulation` for downstream gradients.
The Qwen-Image image/text streams and attention/MLP calls use separate inputs.
The `SiLU + Linear` projection and `adaln_gate_residual` are separate operators.

The FP32 hidden reductions use a lower-plus-upper pairwise tree padded to the
next power of two. Each sample's `dshift` and `dscale` contributions are folded
in ascending logical token order. The CUDA shared reduction reads its result
into each thread before a block barrier permits buffer reuse. No atomic partial sums, Split-K, Stream-K,
TF32, or fast math are used by these kernels. BF16 outputs are cast once after
FP32 arithmetic. The CPU reference computes square root in FP64 and rounds it
to FP32 because CPU `torch.sqrt(float32)` can differ by one ULP from CUDA's
correctly rounded square root. Hidden and token accumulators remain FP32.
Inputs with S=0 or H=0 and mixed dtype/device are rejected.
Strided model chunks are copied to contiguous storage inside
GPU wrappers. The CUDA kernel supports H<=4096, B<=65535, and at most
2^31-1 elements in each input tensor. Python checks these bounds before copying
strided inputs; C++ checks them again before launch. Oversized B/address ranges
raise an explicit error; the registry only resolves using H and does not silently
fallback for these other limits. Shape-aware dispatch selects Triton for larger H.
CUDA symbols are omitted from builds with
`KERNEL_ALIGN_USE_FAST_MATH=1`, making fallback explicit.

Shared eps must be finite and nonnegative and round to finite FP32. A positive
eps that rounds to zero is rejected; representable subnormals, including 2^-149,
remain supported. eps=0 remains supported; zero variance then produces IEEE NaN.
Nonfinite LayerNorm arithmetic follows the FP32 formula and may propagate NaNs;
its NaN payloads are outside the strict byte-equality qualification. Gate and
dgate use a bit-preserving passthrough, including nonfinite payloads. This shared
eps policy does not change the separate indexed/select01 fallback contract.

For Qwen-Image editing, pass `modulate_index` to both resolution and execution:

```python
op, trace = kernel_registry.get_adaln_modulation_op(
    x.device, hidden=x.shape[-1], modulate_index=modulate_index
)
modulated, gate = op(x, modulation, modulate_index=modulate_index)
```

This path accepts modulation `[2B,3H]` and an index `[B,S]` or `[1,S]` whose
values are 0 or 1. It uses the explicit PyTorch select01 reference and returns
gate `[B,S,H]`. The trace sets `fallback=true` and
`fallback_reason=modulate_index_requires_select01`; the indexed path does not
claim the fixed-order CUDA/Triton backward contract.

The resolution matrix `{1024², 1328², 1664×928}` implies image sequence
lengths `{4096, 6889, 6032}` for VAE scale 8 and 2x2 latent packing. H=3072.
These are derived model shapes, not tensor shapes stated by the issue.

## Validation

Public-interface tests check BF16 cast boundaries, each reduction, signed zero,
subnormal eps, gate/dgate payloads, strided upstreams and H=3072 batch positions
with padding across all declared launch configurations. AdaLN gtest checks
candidate output dtype and promotes gold leaves to FP32 before autograd;
casting inside the gold function alone would return BF16 gradients to BF16 leaves.

AdaLN is registered in `gtest/operator_specs.py` and `operator_inputs.py` as
`op_class="reduction"`, with PyTorch, CUDA and Triton candidates. The independent
gold uses FP32 CPU `torch.nn.functional.layer_norm` and autograd for normalization.
Its independent affine VJP folds token contributions in ascending FP32 order;
it does not call production reduction or backward helpers. It checks both outputs
(y and gate) and gradients of x and modulation. Accuracy thresholds come from
`resolve_tolerance`: `forward_accuracy` for outputs and `gradient_accuracy`
for input gradients; no private AdaLN thresholds or SM90 override are added.
Existing fixed-reference byte comparisons remain separate, unchanged gates.

```bash
.venv/bin/python scripts/check_operator.py --op adaln_modulation \
  --candidate pytorch --device cpu --dtype bf16 --batch 2 --seq 3 \
  --normalized-dim 3072 --check-grad --json
.venv/bin/python -m pytest tests/test_adaln_gtest.py -q -rs
```

Use CUDA/Triton candidates on the target GPU for native accuracy evidence.
The three full-image tests additionally check independent CPU FP32 forward and
backward accuracy for both GPU backends/dtypes. These assertions require GPU
execution; CPU results alone do not validate them.
CPU tests also cover gate-only gradients, distinct image/text modulation
chunks, corrupted-output/VJP rejection and unavailable-backend routing.
The CLI accuracy report alone is not a provenance-checked WS1 system gate.

Run `python -m pytest tests/test_adaln_modulation.py -q` for focused tests.
Set `RLK_ADALN_REAL_SHAPES=1` to include the three full image shapes.
Run `python benchmarks/benchmark_adaln_modulation.py --real` for the three
image shapes, with `--dtype fp32` for the second dtype. The benchmark prints
median forward and forward+backward latency plus peak allocated memory;
`--warmup` and `--repeat` control the sample count. It explicitly skips the
CUDA extension if its symbols are absent. The registry trace records selected
backend, fallback, reduction order, accumulator dtype, disabled algorithm
flags, and a semantic kernel version. The semantic version is not a hash of the
compiled binary. Direct backend constructors accept validated launch settings:
CUDA `threads={128,256,512}` (default 256); Triton `num_warps={4,8}`
(default 4), `reduction_tile={64,128,256}` (default 128). Registry dispatch
uses those defaults. These settings change scheduling, not the reduction order.

Native acceptance requires a fresh extension build, source/binary identity records,
and complete test and sanitizer logs.
