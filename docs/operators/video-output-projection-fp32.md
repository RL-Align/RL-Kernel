# Video Output Projection FP32

## Summary

MiniMax-H3 projects each 5376-wide hidden row to 96 video latent velocity values:
`Y = X @ W.T + bias`. The output head runs after final normalization and AdaLN,
with FP32 inputs and parameters. It processes the complete mixed sequence before
the model selects video rows.

This operator owns the projection and its input, weight, and bias gradients.
The model adapter owns final normalization, the FP32 cast, and video selection.
The `packed_index_select` task in [#420](https://github.com/RL-Align/RL-Kernel/issues/420)
provides the separate selection implementation. Composition tests use PyTorch
`index_select` with unique video indices.

The implementation extends the repository's deterministic GEMM infrastructure
from [#180](https://github.com/RL-Align/RL-Kernel/pull/180), using the FP32 Triton
helper present in merged [#325](https://github.com/RL-Align/RL-Kernel/pull/325)
and canonical parameter-gradient collection. The native CUDA layout interfaces
in [#343](https://github.com/RL-Align/RL-Kernel/pull/343) require BF16 inputs.
The FP32 input-projection work in
[#562](https://github.com/RL-Align/RL-Kernel/pull/562) supplies the chunked GEMM,
tile configuration, shared weight/bias row preparation, and validation patterns.
These paths are qualified separately for output shapes.

The output adapter selects the existing full-K FP32 GEMM for reductions of at
most 96 terms and the fixed 128-term chunk helper for longer reductions.
It also reuses canonical row ordering and shared weight/bias gradient preparation
from the input projection.

## Entry Point

```python
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("video_output_projection_fp32", device=x.device)
with canonical_backward_session() as session:
    y = op(x, weight, bias, logical_keys=keys, parameter_id="transformer.proj_out")
    y.backward(grad_y)
    session.validate_complete()
```

Use the same operator under `torch.no_grad()` for inference. Logical keys are
needed when parameters require gradients. Each key identifies the original
logical row, independently of its physical placement. A parameter keeps the
same `parameter_id` across uses within a backward session.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton on CUDA | `TritonVideoOutputProjectionFP32Op` | Shared FP32 deterministic GEMM kernels | Default CUDA implementation; qualification targets SM90 |
| PyTorch reference | `NativeVideoOutputProjectionFP32Op` | CPU FP32 products and sums | Direct independent accuracy reference |
| ROCm | None | None | Separate platform qualification required |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[M,5376]` or `[B,S,5376]` | FP32 | Non-empty, contiguous |
| `weight` | `[96,5376]` | FP32 | Contiguous; same device as `x` |
| `bias` | `[96]` or `None` | FP32 | Contiguous; same device as `x` |
| `logical_keys` | `[M,2]` or `[M,3]` | int64 | Required for trainable parameters; same device as `x` |
| `parameter_id` | Stable module identity | string | Required for trainable parameters |
| Output | `[M,96]` or `[B,S,96]` | FP32 | Leading dimensions preserved |

For a batched input, `M=B*S`. Keys use `(sample_id, token_id[, use_id])`.
Active keys are unique across uses of one parameter. A negative first column
marks padding for parameter-gradient collection. Register every forward use
before starting backward, then use a fresh session for the next graph.

The shared chunked GEMM supports at most 2,097,152 reduction rows with the
current merge tile. For parameter gradients, this counts all active rows across
uses of the parameter in one session. Larger reductions raise `ValueError`
before the chunked GEMM allocates its workspace or launches its kernels.
Available memory can impose a lower limit.

## Dispatch Behavior

The registry exposes this operator for CUDA, where `get_op` selects Triton. The
candidate checks Hopper SM90 eligibility. An unavailable candidate raises an error. Tests
invoke the independent PyTorch reference directly; it is not a CPU registry
fallback. ROCm, MUSA, and NPU calls are outside this contribution's qualified scope.

`OP_SPECS` exposes `triton` and `pytorch` candidates. The shared validation runner
keeps the canonical session active across candidate forward and backward, then
checks that all registered contributions were received.

## Accuracy

For an upstream gradient `G`, the derivatives are:

- `dX = G @ W`, with 96 terms per output element.
- `dW = G_sorted.T @ X_sorted`, reduced over the complete logical row set.
- `dbias = sum_rows(G_sorted)`.

For long weight-gradient reductions, the backend evaluates
`dW = (X_sorted.T @ G_sorted).T`. Both operands are views. Only the final
`[96,5376]` gradient is copied to contiguous storage. This reuses the input
projection's weight-gradient layout and avoids mostly empty row tiles when
the workspace budget splits the product. Forward and transposed dW use
`128 x 32` output tiles with four warps; their 128-term reduction is unchanged.

Forward has 5376 terms per output element. The canonical session collects
parameter contributions and reduces them in logical-key order. It removes
padding and uses stable sorting when needed. Large CUDA inputs already in
canonical order reuse that order after a key check. A single forward use skips
concatenation. For multiple same-dtype CUDA inputs totaling at least 256 MiB,
the session copies each input directly into canonical row positions with
`index_copy_`. Smaller inputs and other platforms retain the original path.
The CUDA path keeps
padding in an unused tail and passes only active rows to the reducer. The
destination has space for all physical rows; this removes the additional
concatenated input buffer while preserving the existing logical order and
reduction. Mixed-dtype inputs retain concatenation and its type promotion.
When weight and bias have matching complete forward-contribution sequences, they
share the sorted upstream gradient. Other parameter uses retain independent
preparation. A layout comparison must preserve the same input
values, logical keys, and upstream gradients. Adding genuine training samples
changes the expected parameter gradient.

The independent CPU reference uses FP32 arithmetic. Its forward and gradient calculations
form FP32 products and sum the complete reduction dimension. Only output rows
and columns are tiled, so the reference does not copy the GPU reduction schedule.
The WS1 `reduction` FP32 policy uses `atol=rtol=1e-4` for forward and gradient
accuracy. BI comparisons require identical bytes.

The output wrapper selects the original full-K GEMM when `K <= 96` and the
fixed-chunk GEMM otherwise. This keeps the short `dX` reduction on the existing
path. Forward, long `dW`, and long `dbias` share the reused chunk helper. Each
128-term chunk forms two 64-term partial sums from ascending 32-term dots,
then combines them in FP32. A second kernel merges chunk partials in a fixed
balanced order. Chunking is part of the declared arithmetic and does not depend
on physical row placement or occupancy.

### Projection and Video Selection

The projection returns every mixed-sequence row. A composition test applies
`index_select` to the projection output and compares with an independent
projection followed by the same unique indices. For the projection VJP, the
selected upstream values are placed back into a zero-filled full-sequence
buffer. This checks the effect of selection without making the projection
interface depend on `video_indices`.

Selection timing and its independent BI qualification belong to the selection
operator. The benchmark below measures projection calls only.

## Performance Notes

Run from the repository root in the CI-aligned single-H20 environment:

```bash
OMP_NUM_THREADS=4 python3.11 -m benchmarks.operators.gemm.benchmark_video_output_projection_fp32 \
  --backends pytorch triton \
  --rows 1 128 1024 4096 13104 38734 74410 104990 \
  --seed 420 --warmup 5 --samples 40 \
  --output /tmp/h3-output-projection.json
```

The benchmark generates inputs, parameters, and upstream gradients from the
seed. It records versions, source hashes, raw timing samples, and implementation
metadata. Use the same seed, row list, and dependency versions to reproduce the
input values. The benchmark preserves the process's CPU thread settings;
`OMP_NUM_THREADS=4` matches the four-core Worker used for qualification.

Measurements include eager forward, backward, combined forward/backward, and
no-grad forward CUDA Graph replay. CUDA events cover complete operator calls,
including layout copies, bias, canonical collection, and launch gaps. The
backward-only setup creates the forward graph before timing; each sample exits
its backward session and releases graph references. Throughput is total mixed
rows per second for eager forward/backward, not videos per second.
Timed calls use logical keys in canonical row order. Reversed rows are checked
for bytewise BI before timing. The ordinary PyTorch baseline is measured with
the same tensors; its accuracy and BI results are reported separately.

A separate warmed F+B call records incremental peak GPU allocation above the
baseline after clearing input and parameter gradients. This includes outputs,
gradients, and temporary allocations. Inputs, persistent caches, and allocator
reservations are outside the incremental measurement.

### Accuracy Coverage in the Benchmark

For `M <= 128`, the independent reference covers every element. Larger layouts
use the following bounded CPU reference:

| Result | Compared elements | Reduction coverage |
| --- | --- | --- |
| `Y` | 32 evenly spaced rows, all 96 channels | All 5376 input features |
| `dX` | Same 32 rows, all 5376 channels | All 96 output features |
| `dW` | 16 evenly spaced input channels, all 96 output channels | All `M` rows |
| `dbias` | All 96 channels | All `M` rows |

The JSON marks sampled accuracy explicitly and stores selected indices and
checked/total element counts. Every candidate element is scanned for NaN/Inf.
Row-permutation BI compares every element of `Y`, `dX`, `dW`, and `dbias`.
Sampling reduces reference CPU memory and time; it does not reduce the candidate
workload or shorten the parameter-gradient reduction.

### H3 Mixed-Sequence Layouts

The output head receives video, audio, and context rows. The geometry follows
[Diffusers v0.40.0](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py#L116).
`F=17n+5` pixel frames correspond to `T=5n+2` video latent frames.
At 24 FPS, the 15-second target limit uses 345 aligned frames.
The default context budget is 1024 rows.

| Layout | Pixel frames | Video rows | Audio rows | Context rows | Total `S` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Landscape, 1344x768 | 124 | 37,296 | 414 | 1,024 | 38,734 |
| Portrait, 768x1344 | 243 | 72,576 | 810 | 1,024 | 74,410 |
| Landscape at target duration limit, 1344x768 | 345 | 102,816 | 1,150 | 1,024 | 104,990 |
| FL2VA context-limit stress profile | 345 | 108,576 | 1,150 | 262,144 | 371,870 |
| Ref2VA context-limit stress profile | 345 | 364,608 | 2,354 | 262,144 | 629,106 |

Spatial dimensions in this table are width by height. The row counts 1, 128,
1024, 4096, and 13,104 are synthetic size cases. The other default sizes match
mixed-sequence video layouts. Values remain seeded random tensors; these tests
do not load a model checkpoint.

The two larger profiles use the 262,144-token limit declared by the pinned
[H3 text encoder](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/text_encoder/config.json)
and
[tokenizer](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/processor/tokenizer_config.json).
Context includes reference vision tokens as well as text. These are checkpoint
context-limit stress profiles. The standard pipeline uses the 1024-row context
budget shown above; the larger profiles do not define its default maximum task
layout or an unconditional bound on custom pipeline inputs.

FL2VA adds two keyframes at the largest rounded video canvas, 1856x576. Ref2VA
adds nine 8192x2048 images and three reference videos. The reference videos have
56, 56, and 248 source frames, totaling 360 frames. Video preprocessing aligns
the last reference down to 243 frames, while its audio uses the full source
duration. The visual references contribute 258,120 rows. Their soundtracks
contribute 1204 audio rows, in addition to the target's 1150 audio rows.
The audio counts follow the upstream
[reference encoder](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L729-L752)
and
[audio VAE](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py#L607-L609).

Run these larger profiles explicitly when the device has sufficient memory:

```bash
OMP_NUM_THREADS=4 python3.11 -m benchmarks.operators.gemm.benchmark_video_output_projection_fp32 \
  --backends pytorch triton --rows 371870 629106 \
  --seed 420 --warmup 5 --samples 40 \
  --output /tmp/h3-output-projection-context-limit.json
```

## Tests

### Packed Requests

The multi-request tests follow the packed layout in merged
[vLLM-Omni #5810](https://github.com/vllm-project/vllm-omni/pull/5810), pinned to
`d150a4fde77d15d466102323a4048b0a8631d74c`.
Its H3 step execution concatenates requests along the sequence dimension.
Each request retains its own 64-row alignment padding; attention boundaries
isolate requests, and projection outputs are split back by request offsets.
See the upstream
[packing implementation](https://github.com/vllm-project/vllm-omni/blob/d150a4fde77d15d466102323a4048b0a8631d74c/vllm_omni/diffusion/models/minimax_h3/batched_packing.py)
and
[independent-versus-batched tests](https://github.com/vllm-project/vllm-omni/blob/d150a4fde77d15d466102323a4048b0a8631d74c/tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py#L155).

The published 672x384, 209-frame profile has 15,624 video rows, 696 audio rows,
and 9 text rows per request: 16,329 active rows, padded to 16,384. Tests use
two and four such requests and a mixed-length pair. They compare a packed call
with separate per-request calls, request reordering, row reordering, and extra
padding. Logical keys retain the original request and row IDs.

The upstream tests establish inference batching behavior. Our training checks
add bytewise `dW` and `dbias` comparisons for the same complete request set in
one canonical session, alongside `Y` and `dX`. A gradient from one request alone
is not compared with the gradient of several requests. Accuracy uses the
independent reference with complete reduction lengths.

Row-preparation tests compare the direct copy with the previous concatenation
and selection byte for byte. They cover independent allocations, shared views,
noncontiguous inputs, empty uses, padding-only uses, mixed dtypes, and CUDA
deterministic algorithms enabled and disabled. Session tests also check that
weight and bias continue to share matching upstream-gradient preparation.

### Commands

Follow the image and installation sequence in `ci/scripts/run_gpu_ci.sh`.
The current CI image is
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; its installation
selects PyTorch 2.4.1+cu124 and Triton 3.0.0. Keep the resolved image digest and
`python3.11 -m pip freeze` with the experiment results.

```bash
export OMP_NUM_THREADS=4
python3.11 -m pytest tests/ops/gemm/test_video_output_projection_fp32.py -v
python3.11 -m pytest tests/runtime/test_kernel_registry.py \
  tests/validation/operators/test_operator_inputs.py \
  tests/validation/operators/test_op_checks.py -k video_output_projection -v
python3.11 -m tools.validation.operators.check_operator \
  --op video_output_projection_fp32 --candidate triton --device cuda --dtype fp32 \
  --batch 1 --seq 129 --check-grad --grad-mode random
```

Keep generated reports, raw JSON, and plots outside tracked source files.

## Known Limitations

GPU qualification targets one H20 with CUDA SM90. Distributed parameter
gradients, higher-order derivatives, and complete training CUDA Graph require
separate integration contracts. The formal `packed_index_select` backend is an
integration dependency; the current selection composition uses a PyTorch test
reference. Operator evidence covers this output head, while full Transformer
and RL trajectory acceptance remain model-level gates.
