# MiniMax-H3 FP32 Timestep MLP

## Summary

`timestep_mlp_fp32` is H3's `TimestepEmbedding(in_channels=256, time_embed_dim=5376,
out_dim=2688)`. It runs on the sinusoidal features of the distinct timesteps
(RFC #420, WS1 step 3):

```text
temb = linear_2(silu(linear_1(features)))     (T, 256) -> (T, 5376) -> (T, 2688)
```

`time_embedder` is a `_keep_in_fp32_modules` module. Its weights, biases,
activations and output are all FP32. `temb` stays FP32 because every AdaLN
projection applies its own SiLU before casting to BF16.

Pinned model: `MiniMaxAI/MiniMax-H3@42ed227`. The tensors are
`time_embedder.linear_{1,2}.{weight,bias}` (see `rl_engine/testing/h3_manifest.json`).

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("timestep_mlp_fp32", device="cuda")
temb = op(features, w1, b1, w2, b2)          # all float32
```

## Backends

| Backend | Wrapper | Native symbols | Status |
| --- | --- | --- | --- |
| CUDA (SM90, SM100) | `rl_engine.kernels.ops.cuda.h3.timestep_mlp.H3TimestepMLPCudaOp` | `rl_engine._C.h3_det_linear_{forward,backward_input,backward_weight}` | Contract `h3-det-linear-v1` |
| PyTorch reference | `rl_engine.kernels.ops.pytorch.h3.timestep_mlp.NativeH3TimestepMLPOp` | n/a | `forward`: provider path (`F.linear`/`F.silu`); `forward_fp32`: FP64 golden |
| ROCm | n/a | n/a | Falls back to the PyTorch reference |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `(T, K)`, `T >= 1` | float32 | H3: `K = 256` |
| `w1`, `b1` | `(H, K)`, `(H,)` | float32 | H3: `H = 5376` |
| `w2`, `b2` | `(D, H)`, `(D,)` | float32 | H3: `D = 2688` |
| output | `(T, D)` | float32 | |

The CUDA kernel needs `K` and `H` to be multiples of 4, for 16-byte vector loads.
These inputs fail closed: BF16 in any argument (RFC probe H7), mismatched shapes,
an empty `x`, mixed devices, and unaligned `K`.

## Numerics: contract `h3-det-linear-v1`

Forward, for each output `y[t, n]`:

- One warp owns a group of output columns. Lane `l` reads the 16-byte K chunks
  `l, l + 32, l + 64, ...`.
- Each lane accumulates its chunks in ascending order, and the elements inside a chunk
  in ascending order, using `fmaf` into an FP32 accumulator that starts at 0.
- The 32 lane sums are combined with an xor butterfly (16, 8, 4, 2, 1).
- The bias is added once, then SiLU runs in FP32 as `v / (1 + expf(-v))`
  (PyTorch's formula). The result is stored as FP32.

How many columns a warp owns, the row tile (1, 2 or 4), and how many chunks are
loaded ahead change only when loads are issued, never the order in which a column
is summed. A timestep's `temb` is therefore bitwise identical however many other
timesteps share the call, and wherever it sits among them.

Backward is deterministic. There is no cuBLAS and there are no atomics:

- `d_hidden = g @ W2` and `dx = d_pre @ W1`: N is split into fixed 64-row chunks.
  Each chunk is an ascending `fmaf` chain, and the chunks are then left-folded in
  ascending order. These gradients are row-local and batch-invariant.
- `d_pre = d_hidden * s * (1 + z * (1 - s))`, elementwise in FP32, using the saved
  pre-activation `z`.
- `dW[n, k] = sum_t g[t, n] * x[t, k]` and `db[n] = sum_t g[t, n]`: an ascending-`t`
  FP32 fold that starts from 0, in logical row order.

Accuracy, measured on a B200 with the pinned checkpoint weights, TF32 disabled and 200
draws of 4 timesteps (each draw includes t = 0 and t = 1):

| Comparison | CUDA | provider (cuBLAS) |
| --- | --- | --- |
| max abs error vs FP64 golden, median over draws | 1.1e-6 | 5.4e-7 |
| gradients vs FP64 autograd (`x`, `w1`, `b1`, `w2`, `b2`) | <= 1.2e-5 | |

Both outputs are about 100× inside the contract (`reduction` / `float32`, atol and rtol
1e-4). Neither is more accurate in general: cuBLAS's tree happens to do slightly better
on these weights. What the CUDA kernel adds is a fixed summation order, which makes each
row batch- and position-invariant and every run repeat-bitwise, and it is 2× faster
for T >= 2. The two outputs are not bitwise equal, so `scripts/h3_chain_replay.py`
reports this stage as the chain's `first_drift`, which is expected for a reduction.

## Performance Notes

```bash
python benchmarks/benchmark_h3_conditioning.py --op timestep_mlp_fp32
```

B200, pinned weights (63.3 MB FP32). End-to-end op times include the Python wrapper:

| T | CUDA op | provider | kernels only (CUDA) |
| --- | --- | --- | --- |
| 1 | 32.8 µs | 36.5 µs | 3.8 + 11.6 µs |
| 2 | 38.8 µs | 79.9 µs | 3.9 + 13.7 µs |
| 4 | 42.7 µs | 83.1 µs | 5.0 + 15.8 µs |

The 5376→2688 layer reads its weights at about 5.0 TB/s at T = 1. For T >= 2 the
provider switches from cuBLAS GEMV to an SGEMM path, which takes about 52 µs of
kernel time.

## Tests

```bash
export RL_KERNEL_H3_WEIGHTS=<dir written by scripts/prepare_h3_weights.py>
python -m pytest tests/h3/test_h3_timestep_mlp.py -v           # operator
python -m pytest tests/h3/test_h3_conditioning_e2e.py -v       # end to end: sinusoid -> MLP
python scripts/check_operator.py --op timestep_mlp_fp32 --candidate cuda --device cuda \
    --dtype fp32 --batch 3 --check-grad
python scripts/h3_evidence.py --op timestep_mlp_fp32 \
    --out docs/usage/evidence/h3-timestep-mlp-b200/report.json
python scripts/plot_h3_evidence.py docs/usage/evidence/h3-timestep-mlp-b200/report.json
```

## Known Limitations

- FP32 only, by contract. The BF16 and FP16 gtest rows do not apply to this op.
- There is no ROCm kernel. ROCm dispatches the PyTorch reference.
- Weight gradients are reductions across timesteps, so they follow logical row
  order. They are deterministic, but not batch-invariant (no cross-row reduction can be).
