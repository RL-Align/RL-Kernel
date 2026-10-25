# F2 backward VJP v2: avoid premature r-cubed underflow

Fixed the reviewed P1 numerical defect in CUDA, Triton and the explicit strict
reference. The forward operator and its saved FP32 rstd are unchanged. The fixed
seven-stage addition tree is unchanged, but the backward leaf values and FP32
operation grouping are intentionally revised; gradient bits need not match v1.

## Defect and fix

For global K with all 128 elements equal to 2^55 and an upstream gradient equal to
one on channels 0..63 and zero on 64..127, v1 formed FP32 `r^3` as zero. It returned
`[2^-55, 0]` across the two halves instead of `[2^-56, -2^-56]`, giving normalized
L2 error 1.0 against FP64 on both CUDA and Triton. The forward square/sum was finite.

VJP v2 evaluates, with each FP32 operation separately rounded:

```text
u = FP32(x * saved_rstd)           # no intermediate BF16 cast
sum = fixed_sum_128(FP32(g * u))
mean = FP32(sum * FP32(1/128))
correction = FP32(u * mean)
dx = BF16(FP32(saved_rstd * FP32(g - correction)))
```

This avoids generating `r^3`. Q-scale and local RoPE gradient casts are unchanged.
CUDA uses explicit rounded multiply/subtract intrinsics; Triton retains disabled
FP fusion and NVIDIA FTZ. No fast math was enabled. The previous sum pairing
64/32/16/8/4/2/1 is retained for the new leaves `g*u`.

## Validation

- Local reference suite: **17 passed**.
- Rebuilt standard A100 extension, complete targeted suite: **107 passed, no skips**
  in 6.64 seconds, including CUDA/Triton forward and gradient parity, batch/chunk
  invariance, boundary checks and build configuration tests.
- Added analytic/reference tests at x=2^50, 2^55 and 2^60.
- GPU counterexamples match the BF16-rounded independent FP64 gradients bitwise,
  with normalized FP64 error below 1e-6.
- Added independent FP64 gradient checks at 25 scales from 2^-60 through 2^60 for
  local/global, Q/K and both backends, using a per-head normalized error. The existing
  0.03 smoke bound was retained, not loosened. This bound is not an upstream-approved
  final acceptance threshold.
- Existing strict-reference and CUDA/Triton parity checks pass with VJP v2.
- Formatting and diff checks passed. No new memory layout or synchronization was
  introduced; prior sanitizer results are historical, not claimed as rerun here.

Environment: A100 80GB PCIe SM80, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0.
Remote `/workspace/rlk-f2-integration-e3gNYR1a`; interpreter
`/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`.

## Baseline versioning

Reuse input tensors and tables from
`../f2-sm80-integration-2026-10-24/artifacts-jit/inputs.pt` (SHA-256 unchanged:
`30b84ed1893f5ed9ad40b21e56018737cc22c97f83f49048c2932906f5482c98`).

The new outputs are saved separately as `backward-v2-artifacts/outputs.pt`, hash:
`47b5e51fdd8e2b2594dc4aff2a7c63c11df1acb37cb61a13e06d7d0a808ae349`.
The first four tensors in each case are forward Q/K and rstd; the last two are
v2 input gradients. `compare_artifacts.py` asserts zero differences in all forward
and rstd tensors, reports intentionally changed gradient elements, and requires
all six new tensors to match between CUDA and Triton. See `artifact-comparison.json`.

Do not compare v2 gradients against v1 archived outputs as a parity requirement.
Historical reports remain records of v1, including its numerical limitation.
For future cross-GPU parity tests, use this v2 output baseline with the original
serialized input/table file. Update a baseline only after independent correctness
checks; never treat a new matching CUDA/Triton pair alone as mathematical proof.

## Reproduce

```bash
RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 python setup.py build_ext --inplace
RL_KERNEL_F2_TEST_JIT=0 python -m pytest tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py tests/build/test_build_platform_collectives.py -q
```

`probe.py` is from the integration experiment. Generate v2 outputs with the original
inputs and `--out backward-v2-artifacts`, omitting the old `--compare` target.
Then run `compare_artifacts.py OLD_OUTPUTS NEW_OUTPUTS` to verify the intended scope.
Build/test logs and updated timing samples are archived alongside the new outputs.

This fix does not establish a contract for arbitrary nonfinite inputs, FP32 square
sum overflow, extreme upstream-gradient overflow, or all possible cancellation
patterns. Other-platform validation, table identity enforcement, native-model
benchmark comparisons and final acceptance thresholds remain separate tasks.
