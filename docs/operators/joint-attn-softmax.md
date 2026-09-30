# Qwen-Image Joint-Attention Softmax

`joint_attn_softmax` is the standalone softmax stage in
[WS1 issue #386](https://github.com/RL-Align/RL-Kernel/issues/386), between Qwen-Image's
`joint_attn_qk_gemm` and `joint_attn_av_gemm`. It normalizes every joint
text+image score row over the final key dimension and supplies the matching
deterministic backward.

## Interface

```python
from rl_engine.kernels.registry import kernel_registry

softmax = kernel_registry.get_op("joint_attn_softmax", device=scores.device)
probabilities = softmax(scores)
```

- Input: contiguous or strided `scores[..., K]`, `K > 0`.
- Dtypes: BF16 or FP32.
- `forward(scores)`: same shape, device, and dtype as `scores`.
- `forward_fp32(scores)`: same shape and device, with FP32 output.
- Output strides and contiguity are not part of the public contract; callers
  that need contiguous storage should call `.contiguous()`.
- Backward: `dS = P * (dP - fixed_sum(P * dP))`; gradients use the input dtype.
- The operator is non-causal and has no mask argument. A caller must materialize
  any mask as `-inf` in `scores` before this stage. Each row must retain at
  least one finite key. NaN, `+inf`, and fully masked rows are outside the
  byte-equality contract.

## Frozen arithmetic contract

`TILE_K = 256` is frozen for this operator, not autotuned. Cross-backend
raw-byte tests and the three full Qwen-Image lengths validate this choice on
the environment below; a new target device still needs its own validation.
One CUDA block or Triton program owns one complete row. Tiles are visited
left to right; their online `(max, sum)` states are merged in that order.
Each padded tile uses the binary reduction tree
`128, 64, 32, 16, 8, 4, 2, 1`.
An all-`-inf` tile contributes nothing and is skipped; the first tile with a
finite key initializes the online state. This includes rows whose first one
or two complete tiles are masked, while a fully masked row remains unsupported.

The CPU, CUDA, and Triton implementations use the same explicit FP32
range-reduction and degree-seven exponential polynomial. The CUDA and Triton
kernels request individually rounded FP32 operations; the PyTorch reference
uses eager tensor operations. Cross-backend byte equality is verified by the
tests, not assumed from the source expressions alone. Backward uses the same
tile tree and left-to-right tile merge for `sum(P * dP)`.

There is no Split-K, Stream-K, atomic partial accumulation, TF32, fast math, or
batch-dependent launch choice. BF16 conversion happens once, at the final
output or gradient write. The shared fingerprint is
`joint-attn-softmax-v1-tile256-exp7`; it identifies the arithmetic contract,
not a compiled binary hash.

Each direct backend wrapper has static `provenance` with its backend id,
reduction order, accumulator precision, forbidden-feature state, and fingerprint.
Its `fallback=False` means that wrapper does not silently delegate. An instance
returned by `kernel_registry.get_op("joint_attn_softmax", ...)` additionally
records `actual_backend`, `backend_enum`, `platform`, `fallback`, and
`prior_rejections`. There, `fallback=True` means an earlier registry candidate
could not be loaded or instantiated before this backend was selected. The
registry keeps this dispatch trace on the returned instance, not on the shared
backend class metadata.

## Backends and dispatch

On NVIDIA CUDA, registry order is CUDA, Triton, then PyTorch reference. The
CUDA wrapper does not silently call another implementation; if its compiled
symbols are unavailable, construction fails and the registry logs the rejected
backend before trying Triton. Triton uses CUDA PTX for explicitly rounded FP32
operations and therefore rejects ROCm. CPU, ROCm, MUSA, and NPU dispatch to the
portable PyTorch reference.
The ROCm, MUSA, and NPU routes describe registry fallback policy, not
byte-equality validation on those devices.

Acceptance tests compare forward and backward logical tensor bytes (including
signed zero), dtype, and shape across the three implementations. The PyTorch
result is separately checked against mathematical softmax and its derivative
with numerical tolerances. These `torch.testing.assert_close` thresholds are
for comparisons with `torch.softmax`, **not** permission for cross-backend
byte differences:

| Mathematical-reference check | FP32 `rtol` / `atol` | BF16 `rtol` / `atol` |
|---|---:|---:|
| CPU forward | `1e-6` / `1e-7` | `5e-3` / `5e-4` (masked rows) |
| CPU backward | `2e-6` / `1e-7`; masked rows `1e-6` / `1e-7` | `5e-3` / `5e-4` (masked rows) |
| Triton masked-row forward and backward sanity check | `5e-4` / `1e-6` | `2e-2` / `2e-4` |

CUDA and Triton must still match the CPU/CUDA byte reference on their shared
test inputs; those assertions use no `rtol` or `atol`.

Invariance tests compare a row alone and in batches with different companions
and positions, changing the number of row programs. They also compare output
and gradient bytes for the first 256 keys at `K=256` versus `K=257` with an
added `-inf` key, exercising an extra fully masked tile. Tile width, CUDA block
width, and Triton warp count are fixed production settings, not independently
variable launch configurations validated by these tests.

## Qwen-Image acceptance lengths

With 512 text positions, VAE stride 8, and 2x2 latent packing, the issue-pinned
image shapes become:

| Image shape | Image tokens | Joint keys |
|---|---:|---:|
| 1024 x 1024 | 4096 | 4608 |
| 1328 x 1328 | 6889 | 7401 |
| 1664 x 928 | 6032 | 6544 |

Tests cover all three key lengths with CPU, CUDA, and Triton forward and
backward. A GPU-only acceptance test also allocates each complete BF16
`[1, 1, K, K]` score matrix: it compares every CUDA and Triton output and
gradient byte, then compares the first, middle, and last rows with the CPU
reference. The selected rows include one or two leading fully masked tiles.
This test skips if CUDA is unavailable or free GPU memory is insufficient.

## Validation and benchmark

```bash
python -m pytest -p no:cacheprovider \
  tests/test_joint_attn_softmax.py \
  tests/test_joint_attn_softmax_cuda.py \
  tests/test_joint_attn_softmax_triton.py \
  tests/test_joint_attn_softmax_registry.py \
  tests/test_joint_attn_softmax_full_shapes.py \
  tests/test_build_platform_collectives.py -q

python benchmarks/benchmark_joint_attn_softmax.py --backends cuda,triton --dtype bf16
python benchmarks/benchmark_joint_attn_softmax.py --backends cuda,triton --dtype bf16 --backward
# Repeat both commands with --dtype fp32.
```

The CUDA extension must be built before GPU validation. The validation
environment was WSL Ubuntu, Python 3.12.13, PyTorch 2.13.0+cu130, Triton
3.7.1, CUDA runtime 13.0, and NVIDIA GeForce RTX 5060 (compute capability
12.0). On 2026-09-29, the six-file suite above passed **109 tests, with 0
skipped, 0 failures, and 0 errors**. Separate BF16/FP32 ×
forward/backward benchmark runs passed strict CUDA↔Triton byte checks at all
three key lengths. This is evidence for that environment, not a claim about
untested hardware.

The benchmark checks raw bytes against CUDA before timing. Its JSON records
Python, PyTorch, Triton, CUDA runtime, GPU, compute capability, latency, and
kernel fingerprint. `--rows` controls flattened `B * H * Q` (default 24);
`--backward` times **forward plus `torch.autograd.grad`**, not the backward
kernel in isolation.
