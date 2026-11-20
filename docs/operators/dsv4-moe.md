# DSv4 MoE (Triton Fused Kernels)

The DSv4 MoE block has two halves, each a fused MLP:

```
shared expert   x -> fc1 -> SwiGLU -> fc3
routed expert   x_q -> fc1 -> clamp-SwiGLU * p_s -> MX quant -> fc3
```

This page covers the Triton kernels that run those two halves on ROCm. They are
written for inference forward, and held to two requirements: byte-level batch
invariance and throughput. Forward only; there is no backward, because the fused
fc1 never writes `z`.

## Entry points

Kernels, in `rl_engine.kernels.ops.triton.moe`:

| # | Function | Math | Launch |
|---|---|---|---|
| 1 | `fused_shared_fc1_swiglu(x, w_fc1)` | BF16 `x @ w_fc1ᵀ` → `BF16(SiLU(gate) * up)` | fused |
| 2 | `shared_fc3(h, w_fc2, tp_size=1)` | BF16 `h @ w_fc2ᵀ` over a fixed 8-leaf K-tree | GEMM |
| 3 | `routed_fc1_swiglu_quant(x_q, w1, expert_offsets, p_s)` | MXFP8 × MXFP4 fc1 → clamp-SwiGLU * `p_s` → MX quant | fused, grouped GEMM |
| 4 | `routed_fc3(h_q, w2, expert_offsets)` | MXFP8 × MXFP4 fc3 → BF16 | grouped GEMM |

`routed_mlp_forward(x_q, w1, w2, expert_offsets, p_s)` runs 3 then 4.

The routed provider is `rl_engine.moe.triton_fused_mlp.TritonFusedMoeMlp`, with
`fc1_swiglu_quant`, `fc3` and `forward`, numeric profile `p5-triton-fused-v1`.
The shared kernels have no provider; call them directly.

## Backends and fallback

Triton only, on ROCm and CUDA. There is no fallback: the provider fails closed
on a wrong numeric profile, on LoRA, and on tensors that are not on the device.
Tiles are pinned per architecture in `_TILES` (`fused_mlp.py`), with `gfx942`
tuned on MI300X and a `default` table for everything else.

## Inputs and outputs

| Tensor | Dtype | Shape |
|---|---|---|
| `x`, `w_fc1`, `h` (shared fc1) | BF16 | `[T, H]`, `[2F, H]` (gate rows then up rows), `[T, F]` |
| `h`, `w_fc2`, `y` (shared fc3) | BF16 | `[T, F]`, `[H, F]`, `[T, H]` |
| `x_q` | `MXTensor` e4m3 | `[M, H]`, rows sorted by expert |
| `w1`, `w2` | `MXTensor` e2m1 | `[E, 2F, H]`, `[E, H, F]` |
| `expert_offsets` | int32 | `[E + 1]`; rows `offsets[e]:offsets[e+1]` belong to expert `e` |
| `p_s` | FP32 | `[M]` |
| `h_q` | `MXTensor` e4m3 | `[M, F]` |
| `y` (routed) | BF16 | `[M, H]` |

Shape rules:

- Shared fc3: F must be a multiple of 8.
- Routed: H and F are multiples of 32. On `gfx942`, H must also be a multiple of
  128 and F of 64 (the pinned `BK`).

Empty experts are allowed. Shared inputs are made contiguous by the kernels;
`MXTensor` operands are used as they are.

## Numerics

**Batch invariance.** Tiles are constants, never chosen from the token count.
There is no split-K and no atomic, and each output tile reduces the full K inside
one program, so `f(x)[t]` and `f(x[t:t+1])[0]` are byte-equal.

**TP invariance (shared).** fc1 is column-parallel, so a rank passing
`[gate_shard; up_shard]` gets the matching column slice of the TP=1 output. fc3
fixes its reduction to the deterministic all-reduce's tree:

```
F = [ s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 ]          8 leaves of F/8
leaf = BF16(FP32 ascending-BK sum over the leaf)
y    = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))   each + is one BF16 add
```

At TP=t a rank holds 8/t adjacent leaves and passes `tp_size=t`, and summing the
partials with the same tree gives the TP=1 bytes, for TP 1, 2, 4 and 8.

**No FP8 by default.** `tl.float8e4nv` is e4m3fn, while CDNA3's FP8 is e4m3fnuz.
The routed kernels decode both MX operands to BF16 and run a BF16 `tl.dot`. The
decode is exact (E4M3 keeps 4 significand bits, E2M1 keeps 2, BF16 has 8). On
gfx94x the activation goes through the hardware fnuz convert; that is byte-equal
to the integer decode.

**FP8 lane.** `fp8=True` (gfx94x only, profile `p5-triton-fused-fp8-v1`) runs
e4m3fnuz MFMA with one dot per 32-block. `prepare_mxfp4_fp8(w)` computes the
per-column reference exponents, and fails closed when a column's scales span more
than 14 binades. Its fc3 rounding differs from the BF16 lane, so its bytes differ.

**Not byte-equal to other lanes.** `tl.sigmoid` does not reproduce nvcc's `expf`,
and `tl.dot` reduces a tile at a time. The kernels declare their own profile
rather than match the CUDA kernels.

### Accuracy (MI300X)

| Quantity | Measured |
|---|---|
| Routed, max relative error vs FP64, H=4096 F=2048 | 3.1e-2 to 4.2e-2 (the MX floor) |
| Routed vs FP32 oracle at H=128 F=64 | byte-equal, including `h_q` codes and scales |
| Shared fc3, max relative error vs FP64 | 4.7e-3 |
| Shared fc1, elements over 2 BF16 ulp from FP64 | about 2e-5, only on tiny values; torch FP32 is similar |

SiLU flushes to zero around |g| ≥ 89, where the true value is about 1e-37.

## Usage

```python
import torch
from rl_engine.kernels.ops.triton.moe import (
    fused_shared_fc1_swiglu,
    routed_mlp_forward,
    shared_fc3,
)
from rl_engine.moe.mx_format import mx_quantize

T, E, H, F = 512, 8, 4096, 2048
bf16 = dict(dtype=torch.bfloat16, device="cuda")

# Shared expert
x = torch.randn(T, H, **bf16)
w_fc1 = torch.randn(2 * F, H, **bf16) / H**0.5
w_fc2 = torch.randn(H, F, **bf16) / F**0.5
y_shared = shared_fc3(fused_shared_fc1_swiglu(x, w_fc1), w_fc2)

# Routed experts: rows sorted by expert
x_q = mx_quantize(x, "e4m3")
w1 = mx_quantize(torch.randn(E, 2 * F, H, **bf16) / H**0.5, "e2m1")
w2 = mx_quantize(torch.randn(E, H, F, **bf16) / F**0.5, "e2m1")
offsets = torch.tensor([T * e // E for e in range(E + 1)], dtype=torch.int32, device="cuda")
p_s = torch.rand(T, device="cuda")
y_routed = routed_mlp_forward(x_q, w1, w2, offsets, p_s)
```

## Performance (MI300X, H=4096, F=2048)

Wall clock with a device sync around each timed run, median of repeats. `gfx942`
tiles, Triton 3.6.0.

Routed, BF16 lane. "torch" is the same math in a per-expert loop with BF16
weights; "dequant" decodes the MXFP4 weights every call, "pre-dequant" holds them
decoded (4× the weight memory):

| E | M | fc1 | fc3 | total | TFLOPS | torch dequant | torch pre-dequant |
|---|---|---|---|---|---|---|---|
| 8 | 512 | 0.34 ms | 0.16 ms | 0.49 ms | 52 | 7.3 ms | 2.4 ms |
| 8 | 2048 | 0.71 ms | 0.34 ms | 1.05 ms | 98 | 7.9 ms | 1.6 ms |
| 8 | 8192 | 2.76 ms | 1.19 ms | 3.95 ms | 104 | 7.8 ms | 3.0 ms |
| 256 | 4096 | 5.75 ms | 2.28 ms | 8.03 ms | 26 | 196 ms | 28.5 ms |
| 256 | 16384 | 6.29 ms | 2.47 ms | 8.76 ms | 94 | 196 ms | 29.9 ms |
| 256 | 65536 | 23.2 ms | 9.47 ms | 32.6 ms | 101 | 198 ms | 33.1 ms |

Shared, per rank, against the unfused torch BF16 chain:

| TP | T | fc1 | fc3 | total | torch | TFLOPS |
|---|---|---|---|---|---|---|
| 1 | 512 | 0.08 ms | 0.06 ms | 0.14 ms | 0.10 ms | 184 |
| 1 | 2048 | 0.20 ms | 0.21 ms | 0.40 ms | 0.29 ms | 256 |
| 1 | 8192 | 0.72 ms | 0.71 ms | 1.42 ms | 0.96 ms | 290 |
| 8 | 512 | 0.06 ms | 0.03 ms | 0.09 ms | 0.10 ms | 36 |
| 8 | 2048 | 0.07 ms | 0.03 ms | 0.10 ms | 0.10 ms | 133 |
| 8 | 8192 | 0.10 ms | 0.10 ms | 0.20 ms | 0.18 ms | 262 |

Tile choice matters for routed and barely for shared. Against the previous pin
(128-row tiles, 8 warps), `BM=64 BN=128 num_warps=4` is 1.3–3× faster with the
same bytes, the gain coming mostly from E=256. A sweep of the shared tiles found
at most about 10%, so they were left alone. The FP8 tiles have not been retuned;
at E=8 M=8192 the FP8 lane takes 4.5 ms and at E=256 M=65536 it takes 38.7 ms.

## Known limitations

- **`mx_quantize` crash at 2³² elements.** `e2m1_encode` calls `torch.searchsorted`
  on the flattened tensor, which raises SIGFPE on the GPU. `w1` at E=256 H=4096
  F=2048 is exactly 2³² elements, so quantize weights per expert at that size.
- **No offset validation in the kernels.** `expert_offsets` must be sorted and in
  range; out-of-range values are accepted without an error.
- **Device selection.** On ROCm use `HIP_VISIBLE_DEVICES`. With
  `CUDA_VISIBLE_DEVICES`, `torch.cuda.device_count()` still reports every GPU.
- **No LoRA and no backward.**

## Tests and benchmarks

```bash
pytest tests/test_triton_fused_moe_mlp.py
python benchmarks/benchmark_triton_fused_moe_mlp.py [--tokens 512,2048,8192]
```

The tests cover byte-level batch invariance, TP 2/4/8 invariance, ragged and empty
experts, the FP8 lane, the FP32 oracle and FP64 references, and fail-closed inputs.
The FP32 oracle (`rl_engine/moe/oracle.py`) is a test reference only.
