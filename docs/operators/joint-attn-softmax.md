# Qwen-Image Joint-Attention Softmax

## Summary

`joint_attn_softmax` is the standalone softmax stage in
[WS1 issue #386](https://github.com/RL-Align/RL-Kernel/issues/386), between Qwen-Image's
`joint_attn_qk_gemm` and `joint_attn_av_gemm`. It normalizes every joint
text+image score row over the final key dimension and supplies the matching
fixed-order backward.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

softmax = kernel_registry.get_op("joint_attn_softmax", device=scores.device)
probabilities = softmax(scores)

# True keeps a key; False marks prompt padding shared by this batch item.
probabilities = softmax(scores, key_padding_mask=key_padding_mask)
```

Direct backend wrappers also expose `forward_fp32(scores)` when callers need
FP32 probabilities independently of the input dtype.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `JointAttnSoftmaxCudaOp` | `joint_attn_softmax_forward*`, `joint_attn_softmax_backward` | Validated on NVIDIA CUDA |
| Triton | `TritonJointAttnSoftmaxOp` | `_joint_attn_softmax_forward_kernel`, `_joint_attn_softmax_backward_kernel` | Validated on NVIDIA CUDA |
| PyTorch | `NativeJointAttnSoftmaxOp` | Fixed eager PyTorch arithmetic | CPU reference and portable fallback |
| ROCm, MUSA, NPU | `NativeJointAttnSoftmaxOp` | Fixed eager PyTorch arithmetic | Dispatch available; byte equality not qualified |

## Tensor Contract

| Argument / result | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `scores` | `[..., K]` | BF16 or FP32 | `K > 0`; contiguous or strided; `-inf` masking is allowed |
| `key_padding_mask` | `[B, K]` | bool or `None` | Optional; same device as `scores`; `True` = valid key, `False` = prompt padding |
| `forward(scores)` | Same as `scores` | Same as `scores` | Normalized over the final dimension; zero for fully masked rows |
| `forward_fp32(scores)` | Same as `scores` | FP32 | Normalized over the final dimension; zero for fully masked rows |
| `scores.grad` | Same as `scores` | Same as `scores` | First-order backward only |

The backward formula is:

```text
dS = P * (dP - fixed_sum(P * dP))
```

The operator is non-causal. Without `key_padding_mask`, callers may still
materialize masks as `-inf` in `scores`. When a mask is provided, valid keys are
assigned a logical order before the fixed reduction and results are written back
to their physical positions. This keeps the arithmetic order unchanged when
prompt padding moves the image keys. Output strides and contiguity are not part
of the public contract; callers that require contiguous storage should call
`.contiguous()`.

## Dispatch Behavior

On NVIDIA CUDA, registry order is CUDA, Triton, then PyTorch. The CUDA wrapper
does not silently delegate: if its compiled symbols are unavailable,
construction fails and the registry records the rejection before trying
Triton. Triton uses CUDA PTX for explicitly rounded FP32 operations and rejects
ROCm. CPU, ROCm, MUSA, and NPU use the portable PyTorch implementation.

Each direct backend wrapper records its backend id, reduction order,
accumulator precision, forbidden-feature state, and kernel fingerprint. An
instance returned by `kernel_registry.get_op(...)` additionally records
`actual_backend`, `backend_enum`, `platform`, `fallback`, and
`prior_rejections`. Here, `fallback=True` means an earlier candidate could not
be loaded or constructed before this backend was selected.

## Accuracy

### Fixed arithmetic contract

`TILE_K = 256` is fixed rather than autotuned. One CUDA block or Triton program
owns one complete row. Tiles are visited from left to right and their online
`(max, sum)` states are merged in that order. Each padded tile uses the binary
reduction tree `128, 64, 32, 16, 8, 4, 2, 1`.

With `key_padding_mask`, tiles are formed from valid keys in their original
relative order instead of their padded physical positions. An all-`-inf` tile
contributes nothing and is skipped; the first tile with a finite key initializes
the online state. If the entire row is masked, forward returns zero probabilities
and backward returns zero gradients. Backward reuses the forward key mapping and
the same tile tree and left-to-right merge for `sum(P * dP)`.

PyTorch, CUDA, and Triton use the same FP32 range reduction and degree-seven
exponential polynomial. CUDA and Triton request individually rounded FP32
operations. There is no Split-K, Stream-K, atomic partial accumulation, TF32,
fast math, or batch-dependent launch choice. BF16 conversion happens once, at
the final output or gradient write.

The shared fingerprint is `joint-attn-softmax-v2-logical-mask-tile256-exp7`. It identifies
the arithmetic contract, not a compiled binary hash.

### Comparison rules

Mathematical comparisons with `torch.softmax` resolve `forward_accuracy` and
`gradient_accuracy` from the shared
`rl_engine/kernels/gtest/tolerance_contract.json` `reduction` rows. Tests do
not define private `atol` or `rtol` values.

Cross-backend comparisons use raw logical tensor bytes, including signed zero;
CUDA and Triton do not receive an accuracy tolerance against the fixed
reference. Invariance tests compare the same row alone and in batches with
different companions and positions. They also cover:

- `K=256` versus `K=257` with an added `-inf` key;
- compact and padded layouts of the same 73 text plus 257 image keys, using an
  explicit `[B, K]` key-padding mask;
- one or two fully masked leading tiles followed by finite keys;
- fully masked rows with zero forward output and zero backward gradients;
- forward and backward at all three Qwen-Image acceptance lengths.

With 512 text positions, VAE stride 8, and 2x2 latent packing, those lengths
are:

| Image shape | Image tokens | Joint keys |
| --- | ---: | ---: |
| 1024 x 1024 | 4096 | 4608 |
| 1328 x 1328 | 6889 | 7401 |
| 1664 x 928 | 6032 | 6544 |

A GPU-only acceptance test allocates each complete BF16 `[1, 1, K, K]` score
matrix. It compares every CUDA and Triton output and gradient byte, then checks
the first, middle, and last rows against the PyTorch reference. The test skips
when CUDA is unavailable or free GPU memory is insufficient.

## Performance Notes

The benchmark checks output and gradient bytes against CUDA before timing:

```bash
python benchmarks/benchmark_joint_attn_softmax.py --backends cuda,triton --dtype bf16 \
  --warmup 10 --iterations 200
python benchmarks/benchmark_joint_attn_softmax.py --backends cuda,triton --dtype bf16 \
  --backward --warmup 10 --iterations 200
python benchmarks/benchmark_joint_attn_softmax.py --backends cuda,triton --dtype bf16 \
  --valid-text-keys 73 --warmup 10 --iterations 200
python benchmarks/benchmark_joint_attn_softmax.py --backends cuda,triton --dtype bf16 \
  --backward --valid-text-keys 73 --warmup 10 --iterations 200
# Repeat the four commands with --dtype fp32.
```

`--rows` controls flattened `B * H * Q` and defaults to 24. `--backward` times
forward plus `torch.autograd.grad`, not the backward kernel alone.
`--valid-text-keys 73` keeps the first 73 of the 512 prompt slots and masks the
remaining 439 while leaving image keys valid. The JSON output records this
layout together with Python, PyTorch, Triton, CUDA runtime, GPU, compute
capability, git commit, warmup and measured iterations, latency, and kernel
fingerprint.

Fixed tile sizes `64/128/256/512` were compared offline on an RTX 5060. The
best size differed by backend, while 256 gave a reasonable shared CUDA/Triton
balance. The production arithmetic contract therefore keeps `TILE_K=256` and
does not autotune it at runtime.

The explicit-mask path builds one `[B, K]` logical-to-physical map and reuses it
for every head/query row and for backward. On the same RTX 5060, BF16
`[1, 24, 1, 7401]` measurements included mapping construction. Each value below
is the median of three separate benchmark invocations, each with 10 warmups and
200 measured iterations:

| Backend | Forward, no mask | Forward, mask | Forward + backward, no mask | Forward + backward, mask |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 0.0296 ms | 0.0540 ms | 0.1904 ms | 0.2627 ms |
| Triton | 0.0702 ms | 0.1025 ms | 0.2696 ms | 0.3930 ms |

## Tests

```bash
python -m pytest -p no:cacheprovider \
  tests/test_joint_attn_softmax.py \
  tests/test_joint_attn_softmax_cuda.py \
  tests/test_joint_attn_softmax_triton.py \
  tests/test_joint_attn_softmax_registry.py \
  tests/test_joint_attn_softmax_full_shapes.py \
  tests/test_build_platform_collectives.py \
  tests/test_operator_inputs.py -q

python scripts/check_operator.py --op joint_attn_softmax --candidate pytorch \
  --device cpu --dtype fp32 --batch 2 --seq 257 --check-grad
python scripts/check_operator.py --op joint_attn_softmax --candidate cuda \
  --device cuda --dtype bf16 --batch 2 --seq 257 --check-grad
python scripts/check_operator.py --op joint_attn_softmax --candidate triton \
  --device cuda --dtype bf16 --batch 2 --seq 257 --check-grad
```

The validation environment was WSL Ubuntu, Python 3.12.13, PyTorch
2.13.0+cu130, Triton 3.7.1, NVIDIA driver 591.86, CUDA toolkit/runtime 13.0,
GCC 13.3.0, and an NVIDIA GeForce RTX 5060 (compute capability 12.0).

On 2026-10-09, the seven-file suite passed **170 tests, with 0 skipped, 0
failures, and 0 errors**. All three `check_operator.py` runs passed with
gradient checks. Registry dispatch selected the CUDA extension with fingerprint
`joint-attn-softmax-v2-logical-mask-tile256-exp7` and no fallback. Separate BF16/FP32
forward/backward benchmark runs passed strict CUDA↔Triton byte checks at all
three key lengths. These results qualify that environment only.

## Known Limitations

- Byte invariance across different prompt-padding layouts requires the explicit
  `key_padding_mask`; materialized `-inf` remains supported but follows physical
  key positions.
- NaN and `+inf` are unsupported. They propagate NaN results instead of being
  treated as masked values.
- Only first-order backward is covered.
- Output layout is not guaranteed to preserve input strides.
- ROCm, MUSA, and NPU use the portable fallback but are not byte-equality qualified.
- Arithmetic tile width, CUDA block width, and Triton warp count are fixed
  production settings; independent launch-geometry and arithmetic-tiling
  changes are not covered by the current invariance tests.
