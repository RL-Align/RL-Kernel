# MXFP8 Activation Quantization (`mxfp8_act_quant`)

DSv4 P5-1. Quantizes routed-expert activations to **MXFP8** — OCP
Microscaling E4M3 elements with one E8M0 shared scale per 32 elements — as the
activation side of the MXFP8 × MXFP4 grouped GEMM (P5-4). Backward is a
straight-through estimator.

## Why

The P5 contract requires **train/infer byte equality** on one numeric profile:
the same activation tensor must quantize to the same bytes in the training
engine and in the rollout engine, or the two paths drift. The block amax is
therefore row-local (never crosses a row, so never a rank), the scale recipe is
pinned, and every backend must reproduce the start-kit oracle
(`rl_engine.moe.mx_format.mx_quantize`) **bit for bit** rather than within a
tolerance.

## Frozen recipe

For each 32-element block along the last dimension:

```
amax   = max(|x|)                                     # row-local, order-independent
code   = clamp(floor(log2(max(amax, FLT_MIN))) - 8, -127, 127) + 127
code   = 127 if amax == 0                             # all-zero block -> scale 1.0
scale  = 2**(code - 127)
elem   = cvt.rn.satfinite.e4m3(clamp(x / scale, -448, 448))
```

- The amax is an integer max over `|x|` bit patterns and `floor(log2(·))` is
  read off the FP32 exponent field after the `FLT_MIN` clamp — exact integer
  arithmetic, equivalent to the oracle's `frexp` path and free of libm rounding.
- The scale is assembled from its bit pattern (`code == 0` is the subnormal
  `2**-127`) and the division is an explicit non-`.ftz` `div.rn.f32` /
  `tl.fdiv(..., ieee_rounding=True)`. Under `--use_fast_math`
  (`KERNEL_ALIGN_USE_FAST_MATH=1`) `fmaxf`/`fabsf`/`__fdiv_rn` become their
  `.ftz` forms, which is why none of them sit on the path that can see a
  subnormal; a fast-math build emits the same bytes.
- E4M3 encoding is clamp-then-RNE (decision D1 of the start kit), which is what
  `cvt.rn.satfinite.e4m3` does in hardware. Saturation is not a corner case
  here: `x / scale` reaches up to just under 512 whenever the block amax has a
  mantissa above 1.75.
- Non-finite input is **fail-closed**: the kernels raise instead of emitting
  garbage codes (`ValueError`), matching `_check_finite` in the oracle.
- Backward is the straight-through estimator `dX = dY` (P5 plan, item 5), with
  dtype and shape preserved (the result is contiguous).

## Determinism

`max` is order-independent over a fixed 32-element window and every element is
then transformed independently, so the emitted bytes do not depend on grid
shape, block size, launch configuration, or how many rows are in flight. A
row's bytes are identical whether it is quantized alone or inside a 16k-row
batch (Axis-A batch invariance), which the tests assert directly.

## Backends

| Backend | Entry point | Notes |
|---|---|---|
| CUDA | `rl_engine.kernels.ops.cuda.moe.mxfp8_act_quant_fwd_cuda` | `csrc/cuda/moe/mxfp8_act_quant.cu`. A thread owns a whole MX block for bf16/fp16 (4 × 16-byte loads, no cross-lane reduction) and 8 elements for fp32; 16-byte alignment is all the vector path needs. SM89+ (E4M3 conversion). |
| Triton | `rl_engine.kernels.ops.triton.moe.mxfp8_act_quant_fwd_triton` | `[32, 32]` tile per program; warp count switches on input size. Portable fallback and cross-backend reference. |
| PyTorch | `rl_engine.moe.mx_format.mx_quantize` | The start-kit oracle. Correct but ~35× slower; reference and benchmark baseline only. |

The CUDA path uses the compiled `rl_engine._C` extension when it exports the
symbols, and otherwise JIT-builds that single `.cu` file so tests and
benchmarks run in a source tree without a full extension build. Set
`RL_KERNEL_P5_DISABLE_JIT=1` to require the AOT symbols.

## Inputs and outputs

- `x`: CUDA tensor, `bf16` / `fp16` / `fp32`, any rank, **last dim divisible by
  32**, contiguous (a non-contiguous input is made contiguous by the wrapper).
- Returns an `MXTensor` with `codes` (`uint8`, same shape as `x`), `scales`
  (`uint8`, last dim `K / 32`), and `elem_format == "e4m3"`.
- `check_finite=False` skips the fail-closed read-back (one device sync per
  call, plus the flag memset). It exists for throughput measurement only: issue
  the P5-1 spec makes the raise part of the contract, so the providers always check.
- Backward takes `dy` of any floating dtype (the oracle hands it FP32
  accumulators) and returns a contiguous `dx` of the same dtype and shape.

## Usage

```python
from rl_engine.kernels.ops.cuda.moe import mxfp8_act_quant_fwd_cuda
from rl_engine.moe.backends.mxfp8_act_quant import CudaMXFP8ActQuantProvider

q = mxfp8_act_quant_fwd_cuda(x)          # x: [tokens, hidden] bf16, hidden % 32 == 0
q.codes, q.scales                        # uint8 [tokens, hidden], uint8 [tokens, hidden/32]

provider = CudaMXFP8ActQuantProvider()   # P5 start-kit provider (oracle elsewhere)
```

Acceptance (start-kit command, both backends pass byte-equal):

```bash
python scripts/check_p5.py --device cuda \
    --provider rl_engine.moe.backends.mxfp8_act_quant:CudaMXFP8ActQuantProvider
python scripts/check_p5.py --device cuda \
    --provider rl_engine.moe.backends.mxfp8_act_quant:TritonMXFP8ActQuantProvider
```

## Accuracy

Bit-exact — the acceptance criterion is `torch.equal` on the raw code and scale
bytes, not `allclose`. The kernels reproduce the committed golden manifest
(`tests/fixtures/p5/golden_hashes.json`, anchored on the CPU x86 oracle) and,
independently, `torchao`'s OCP-MX reference cast in FLOOR mode.

## Performance

H100 80GB HBM3, bf16, `benchmarks/benchmark_mxfp8_act_quant.py`, ms/call
(lower is better). Our rows are measured with `check_finite=False`; the
per-call fail-closed read-back adds one device sync (`--check-finite`,
second table).

| implementation | format | 1×7168 | 128×7168 | 4096×7168 | 16384×7168 | 8192×2048 |
|---|---|---|---|---|---|---|
| torch-native (P5 oracle) | MX | 0.339 | 0.386 | 1.361 | 4.492 | 0.906 |
| torchao `to_mx`, eager | MX | 0.155 | 0.156 | 0.681 | 2.565 | 0.413 |
| torchao `to_mx` + `torch.compile` | MX | 0.046 | 0.045 | 0.044 | 0.122 | 0.046 |
| `triton_kernels.downcast_to_mxfp` | MX | 0.068 | 0.068 | 0.068 | 0.129 | 0.068 |
| vLLM `per_token_group_quant_fp8` | fp8, g128 + fp32 scale | 0.016 | 0.016 | 0.063 | 0.244 | 0.037 |
| vLLM `per_token_group_quant_fp8` | fp8, g32 + e8m0 scale | 0.016 | 0.016 | 0.252 | 1.001 | 0.145 |
| **ours (triton)** | MX | 0.029 | 0.029 | **0.033** | **0.119** | 0.029 |
| **ours (cuda)** | MX | **0.011** | **0.011** | 0.034 | 0.119 | **0.021** |
| roofline (plain fp8 cast) | lower bound | 0.007 | 0.007 | 0.032 | 0.117 | 0.019 |

With the per-call fail-closed read-back on (`--check-finite`; one `.item()`
sync plus a flag memset per call, which is what the providers always do):

| implementation | 1×7168 | 128×7168 | 4096×7168 | 16384×7168 | 8192×2048 |
|---|---|---|---|---|---|
| ours (triton), `check_finite=True` | 0.046 | 0.046 | 0.079 | 0.165 | 0.067 |
| ours (cuda), `check_finite=True` | 0.026 | 0.026 | 0.059 | 0.146 | 0.048 |

The sync is the whole difference. Hoisting the check to once per step would
recover the first table, but that is a contract change (P5-1 spec s4), not a
provider option.

Effective bandwidth of the better of our two kernels: 2.66 TB/s at 4096×7168
and **2.99 TB/s at 16384×7168** — the roofline row is `x.to(float8_e4m3fn)`,
which moves the same bytes without the block reduction or the scale stores, so
there is no meaningful headroom left at the large shapes.

Baseline notes:

- **`triton_kernels.downcast_to_mxfp`** (Triton repo, v3.8.0) is the MX
  quantizer vLLM runs for its MXFP4/GPT-OSS path — the closest production
  kernel to this operator, same block-32 E8M0 layout. In `ROUND_DOWN` scale
  mode it is **byte-identical** to this implementation. We are 1.08× faster at
  16384×7168 and 2.1–4.5× faster below that (its cost is flat ~0.068 ms, so it
  is launch/config bound on small and mid shapes).
- **vLLM `per_token_group_quant_fp8`** dispatches to the hand-written CUDA
  kernel `torch.ops._C.per_token_group_fp8_quant`; SGLang runs the same
  algorithm through `sgl-kernel`. It is a *different format* — group 128 with
  one FP32 scale, the DeepSeek-V3 recipe — so it is a cost reference, not a
  drop-in alternative, and it writes 4× fewer scale bytes than MX at group 128.
  Even so it is ~2× slower than these kernels at the large shapes, and at MX's
  block size (`group_size=32`, `use_ue8m0=True`) it is 7–8× slower: that
  configuration is off its tuned path.
- **torchao** `to_mx` in FLOOR mode is likewise byte-identical to the oracle.
  Its own fused MX kernels (`triton_to_mxfp8_dim0`, `mxfp8_quantize_cuda`) are
  gated on `is_sm_at_least_100()` — MXFP8 is Blackwell-native — so on Hopper
  the compiled eager path is the best torchao can do. NVIDIA Transformer Engine
  gates its MXFP8 quantizer the same way.
- Two independent implementations (torchao, triton_kernels) agreeing byte for
  byte with the start-kit oracle is a stronger correctness signal than our own
  tests alone.
- The STE backward is a pure copy and runs at `torch.clone` speed.

## Scope

In: single-rank forward + STE backward, bf16/fp16/fp32 activations, SM89+.
Out: the MXFP4 weight side and the grouped GEMM (P5-4), EP/TP placement
(P5-7 … P5-9), ROCm (which must register its own numeric profile — no fnuz
silent relaxation).

## Tests and benchmarks

- `tests/test_p5_mxfp8_act_quant.py` — bit-wise alignment vs the oracle and the
  golden manifest, batch invariance, determinism, CUDA-vs-Triton equality, STE
  backward, fail-closed behavior, and the full provider pipeline.
- `benchmarks/benchmark_mxfp8_act_quant.py` — torch-native vs Triton vs CUDA vs
  torchao, with the roofline reference.
