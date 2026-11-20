# Packed MX expert input gradients on ROCm

`RocmPackedMXBackend` explicitly supplies tasks 43/44 to
`RocmExpertTrainingBackend(base=RocmPackedMXBackend())`. Existing fused Triton
expert kernels provide packed decoding and dispatch helpers but do not expose
both FP32 base projections and the training input VJP required by task 48.

The new provider retains checkpoint E2M1 bytes and E8M0 scales. It computes
MXFP8 activation quantization and 32-wide block products in ascending order,
then accumulates in FP32 without atomics or split K. The input VJP decodes
packed weight tiles in registers, rounds them to BF16 and performs fixed-order
BF16-input/FP32-accumulator products. No persistent BF16 weight is materialized;
no TE hardware support check is overridden. The activation derivative is the
explicit identity STE. Frozen base weights have no weight gradient; task 45
can provide trainable LoRA deltas through the existing composition.

The numerical profile is `dsv4-mxfp8-mxfp4-rocm-block32-fp32-v1`. It is not a
claim of byte equality with vLLM's native MoE or the fused P5 MLP profile.

```bash
RLK_DSV4_CHECKPOINT=/models/4l python -m pytest -q \
  tests/backends/rocm/test_packed_mx.py
```

MI300X/gfx942: 2 passed, 0 failed, 0 skipped with original checkpoint SHA256
`1bd423ab1d98f43d40b19617fcb30d973d94436cc7b150fe3f744de79c2de086`.
The checkpoint test executes layer 0, expert 0 gate/up/down and a nonzero,
finite input gradient. The other test covers quantization bytes, independent
FP32 accuracy checks, replay bytes, empty experts and invalid inputs. These
are operator tests; the provider does not itself load a native Megatron actor.
