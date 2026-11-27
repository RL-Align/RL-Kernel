# fp8_group_quant (DSv4)

> Part of the DSv4 SM90 batch-invariant kernel set. Design, module map, how to run the
> tests and benchmarks: [DSv4 design](../architecture/dsv4-design.md).

Per-token-group FP8 (E4M3) activation quantization that feeds DeepGEMM's SM90
FP8 GEMMs for DSv4 (attention projections and the shared expert; decisions D2 / D3
in the [DSv4 design](../architecture/dsv4-design.md)). The CUDA kernel is byte-equal to
the PyTorch golden, and the golden reproduces vLLM's
`per_token_group_fp8_quant` with the DSv4 rollout settings, so rollout and
training hand DeepGEMM the same FP8 operands.

## Semantics

For every group of 128 consecutive values along the last dimension:

```text
absmax = fmax(1e-10, max |x|)     NaN does not count toward the max
s      = absmax / 448             FP32, correctly rounded
s      = fmax(s, 1e-10)
s      = 2^ceil(log2(s))          UE8M0 (default); computed from the FP32 bits
q      = e4m3(fmin(fmax(x / s, -448), 448))
```

- Matches vLLM with `VLLM_USE_DEEP_GEMM_E8M0=1` (its default), group 128,
  `eps=1e-10`. `scale_ue8m0=False` keeps the unrounded scale.
- vLLM's FP32-scale kernel computes the UE8M0 step as `exp2f(ceilf(log2f(s)))`,
  which rounds down when `s` sits a few ULPs above a power of two. For BF16 and
  FP16 inputs `s` never does (checked over every finite input value), so the
  bit-based ceiling agrees; FP32 inputs are rejected for this reason.
- As in vLLM, a NaN input becomes -448, and in a group containing an infinity
  the finite values become 0 and the infinity becomes -448.
- DeepGEMM's own `per_token_cast_to_fp8` helper clamps the absmax to `1e-4`, so
  it differs from this operator on groups whose absmax is below `1e-4`.

Each group is quantized independently, so the operator is batch-invariant.

## Outputs and layout

- `fp8_group_quant(x [M, K])`: `q` FP8 `[M, K]`; `scale` FP32 `[M, K / 128]` in
  DeepGEMM's SM90 layout (column-major, column stride `M` rounded up to a
  multiple of 4). DeepGEMM uses it without a layout-conversion kernel.
- `fp8_group_quant_tokens(x [T, C])`: the wgrad operand `x.T` quantized in
  128-token groups; `q` FP8 `[C, T_pad]`, `scale` FP32 `[C, T_pad / 128]`, with
  `T_pad` a multiple of 128 and zero padding (zero tokens add nothing to a
  wgrad, and DeepGEMM's wgrad needs a token count that is a multiple of 16).

Inputs are 2-D BF16 or FP16; `K` must be a multiple of 128.

## Backends

| Backend | Entry point | Notes |
|---|---|---|
| PyTorch golden | `rl_engine.kernels.ops.pytorch.quant.fp8_group_quant` | CPU and CUDA; not capturable in a CUDA graph |
| CUDA | `rl_engine.kernels.ops.cuda.quant.fp8_group_quant_cuda` (`Fp8GroupQuantCudaOp`) | `csrc/cuda/quant/fp8_group_quant.cu`; fails closed without the `rl_engine._C` symbols; no ROCm build |

The CUDA kernel keeps every step that could see a subnormal on integer bit
patterns or explicit non-`.ftz` PTX and refuses to build with
`--use_fast_math`.

## Performance

H200, BF16 input, CUDA-graph replay, median per call:

| Shape | CUDA | DeepGEMM helper + layout conversion |
|---|---|---|
| 1 x 4096 | 2.2 us | 31.9 us |
| 4096 x 4096 | 21.1 us (2.4 TB/s) | 260.8 us |
| 16384 x 4096 | 76.1 us (2.7 TB/s) | 910.1 us |
| tokens 4096 x 4096 | 41.0 us (1.2 TB/s) | - |

## Tests

```bash
python -m pip install --no-build-isolation -e .
python -m pytest tests/ops/quant -q
```

`tests/ops/quant/test_fp8_group_quant.py` pins the golden's semantics on hand-
built cases; `tests/ops/quant/test_fp8_group_quant_cuda.py` checks the CUDA
bytes against the golden (edge values, odd shapes, both dtypes, both scale
modes, both directions), batch invariance, and that DeepGEMM consumes the
scales directly (needs `deep_gemm` and `CUDA_HOME` pointing at CUDA 12.8+).
