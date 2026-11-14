# DSV4 shared-expert down projection replay on gfx942

A native vLLM capture with the original four-layer checkpoint reproduces a
rollout-only difference at decode layer 0 `shared_experts.down_proj`. The BF16
input is identical. Capturing the actual post-load FNUZ input/weight and FP32
scales further shows that all four GEMM operands are identical.

The installed AITER configuration selects CK kernel
`a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1`
with `splitK=2` at M=1, N=4096, K=2048. Twenty repeated native calls differed
on 1208–1442 BF16 elements. First captured values were 0.0322265625 (`0x3d04`)
and 0.031982421875 (`0x3d03`). Calling the same CK kernel with `splitK=0`
produced zero differences on all twenty repeats.

`bind_shared_down(model, worker)` is an explicit instance-only opt-in for this
DSV4 shape and gfx942 FNUZ/block128 layout. It binds only local shared-expert
down-projection kernel instances, preserving native activation quantization
and original postprocessed weight/scales. Other AITER dispatch is unchanged.
This is a deterministic profile, not a performance optimization. It does not
claim whole-model or actor/rollout parity.

```bash
RLK_DSV4_SHARED_DOWN_OPERANDS=/outputs/decode0-shared-down-operands.pt \
python -m pytest -q tests/backends/rocm/test_shared_down_fp8.py
```

MI300X: 1 passed, 0 failed, 0 skipped, including original native operands,
20 repeats, and the same row at batch sizes 2 and 17, first and last positions.
A full real verl-owned vLLM worker also bound all four local shared projections
before warmup and generated a response. Changing the reduction profile can
change logits/tokens; samples from the two profiles are not a parity test.
