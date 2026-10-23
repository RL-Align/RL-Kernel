# Latent normalization (Qwen-Image Phase A)

Issue [#386](https://github.com/RL-Align/RL-Kernel/issues/386) assigns the CUDA
`latent_normalize` row to @JLiu4Coding. This standalone operator implements
normalization, denormalization and first-order input gradients. VAE execution,
packing, full-model integration and distributed execution are separate work items.

## Contract: `latent-normalize-fp32-v1`

Inputs are contiguous channel-first `[B,16,H,W]` or `[B,16,1,H,W]` tensors,
with positive dimensions, in BF16 or FP32. `mean` and `std_inv` are contiguous
FP32 `[16]` vectors on the same device, broadcast across batch and spatial axes.
They are fixed model buffers; `requires_grad=True` is rejected. The caller must
validate finite statistics and positive `std_inv` at model load. The hot path
validates metadata without synchronizing the GPU to inspect tensor values.

| Direction | Forward | Input gradient |
| --- | --- | --- |
| Normalize | `(x - mean) * std_inv` | `dy * std_inv` |
| Denormalize | `x / std_inv + mean` | `dy / std_inv` |

Each arithmetic operation rounds separately in FP32, with round-to-nearest CUDA
intrinsics. No FMA contraction, reciprocal substitution, reductions, atomics,
TF32 or fast math are used. Output and gradient each cast once to the input dtype.
The inverse is not guaranteed to recover original bytes after BF16 rounding.
The build rejects execution when compiled with CUDA flush-to-zero enabled.

This is the issue's FP32 arithmetic contract, not a claim of byte equality with
an unmodified Diffusers BF16 pipeline. The runtime adapter must pin the statistics
values and how the inverse standard deviations are constructed. The released
[pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py)
uses division by the inverse standard deviation for denormalization. This operator
accepts the already-constructed FP32 vectors and does not load checkpoint statistics.

Unsupported layouts/dtypes, trainable statistics, empty tensors and multiple
frames fail explicitly. Higher-order gradients, CUDA graph/torch.compile
integration, ROCm, and other accelerators are outside this initial contract.

## Use

```python
from rl_engine.runtime.registry import kernel_registry

normalize = kernel_registry.get_op("latent_normalize", device="cuda")
denormalize = kernel_registry.get_op("latent_denormalize", device="cuda")
y = normalize(x, mean, std_inv)
z = denormalize(y, mean, std_inv)
print(normalize.last_trace)
```

CUDA dispatch contains only the native implementation: a missing extension does
not fall back to PyTorch. CPU dispatch supplies an independent eager reference.
The trace records requested/actual backend, native forward/backward symbols,
arithmetic contract, dtypes, launch size and fallback state. `last_trace` describes
the last successful forward launch; the backward symbol names its bound autograd
implementation, rather than asserting that backward has already run.

## Build and validation

Use the repository development environment with PyTorch and a matching CUDA
toolkit. For an A100:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST=8.0 RL_KERNEL_REQUIRE_EXT=1 \
  python setup.py build_ext --inplace
python -m pytest tests/ops/norm/test_latent_normalize.py -q
python tools/validation/operators/check_operator.py --op latent_normalize \
  --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 257 --check-grad --json
python tools/validation/operators/check_operator.py --op latent_denormalize \
  --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 257 --check-grad --json
PYTHONPATH=. python benchmarks/operators/norm/benchmark_latent_normalize.py
```

Accuracy uses the shared `elementwise` forward/gradient thresholds from
`rl_engine/contracts/profiles/precision/ws1.json`. The focused tests additionally
require exact bytes against the CPU FP32 reference after the declared output
cast, and exact forward/gradient bytes across batch sizes, batch positions,
launch sizes and repeated executions. Nondefault streams, expanded upstream
gradients, FP32 rounding edges, native-boundary validation and missing-extension
dispatch are covered separately.

The reference image sizes map through the base VAE's spatial factor of eight to
latent grids `128x128`, `166x166`, and `208x116`; normalization precedes 2x2 packing.
The test suite also covers an odd `3x7` synthetic grid. No model weights are needed.

The benchmark produces CPU-reference accuracy reports with checked BF16 backend
provenance, raw-byte mismatch counts, timing methodology, source hashes and runtime
versions. Timings compare the public CUDA and eager PyTorch paths; the native-only
forward time is also reported to separate wrapper overhead. Backward timings use
a retained graph and exclude forward. Results establish only the recorded hardware
profile. H100/SM90, full-model trajectories and other platforms require separate
validation; A100 results do not establish those claims.
