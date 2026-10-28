# Qwen-Image Multi-Axis RoPE

This operator implements the `multi_axis_rope` CUDA row of
[RFC #386](https://github.com/RL-Align/RL-Kernel/issues/386). It rotates **adjacent
channel pairs** with three independently indexed axes of widths `[16, 56, 56]`.
The existing GPT-NeoX rotate-half operator has different semantics.

## Reference and scope

Semantics are pinned to Diffusers
[`40f63690da440e8c382ab53e67508835deb1ed2f`](https://github.com/huggingface/diffusers/blob/40f63690da440e8c382ab53e67508835deb1ed2f/src/diffusers/models/transformers/transformer_qwenimage.py):
`QwenEmbedRope(scale_rope=True)` and `apply_rotary_emb_qwen(use_real=False)`.
The position helper covers single-frame text-to-image at packed latent resolution.
For a 1024×1024 image the packed grid is 64×64. Multi-image editing, video,
QwenEmbedLayer3DRope, distributed execution and full-model integration are outside
this operator's qualification.

Image coordinates are `(0, h, w)` in row-major order. The centered coordinate
range is `[-ceil(size/2), floor(size/2))`, so an odd grid has one extra negative
coordinate. Text starts at `max(height//2, width//2)` on all three axes. The helper
returns image and text positions separately: the caller determines stream order.

## API

```python
import torch
from rl_engine.ops.rope.multi_axis_rope import (
    qwen_image_positions, build_multi_axis_cos_sin, get_multi_axis_rope_op,
)

image_pos, text_pos = qwen_image_positions(64, 64, 128)
cos, sin = build_multi_axis_cos_sin(image_pos, device="cuda")
x = torch.randn(1, 4096, 24, 128, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
op = get_multi_axis_rope_op("cuda", device=x.device)
y = op(x, cos, sin)
y.float().square().mean().backward()
print(op.trace())
```

| Argument | Contract |
| --- | --- |
| `x` | `[B, S, H, 128]`, H > 0; FP16, BF16 or FP32 |
| `cos`, `sin` | Matching FP32 `[S, 64]` shared tables or `[B, S, 64]` per-sample tables |
| Devices | Input and tables on the same device |
| Layout | BSHD, adjacent pairs; input and independent table strides read directly |
| Output | Input shape/dtype/device; contiguous |
| Empty input | B=0 and/or S=0 supported, including backward |
| Gradients | Input gradients and higher-order input VJPs; table gradients rejected |
| Position builder | int32/int64 `[S, 3]` or `[B, S, 3]`, within `[-4096, 4095]` |

Table construction follows the upstream CPU FP32 `pow` and `polar` operations,
then copies the results to the target device. Build and retain tables before CUDA
Graph capture. Reusing these tables also prevents per-batch transcendental drift.
The CPU factories select their device explicitly, including inside a CUDA or
meta default-device context. Packed QKV views require no materializing copy.

## Numerical contract

Contract ID: `qwenimage-interleaved-rope-fp32-v1`.

For each adjacent pair `(a, b)`, compute `(a*c - b*s, b*c + a*s)` in FP32 with
separate rounded multiplications and additions, followed by one output cast.
There is no cross-element reduction, split-K, atomic accumulation or TF32.
CUDA uses round-to-nearest arithmetic intrinsics; Triton disables FP contraction.
Backward uses the transpose rotation (negate sine), including recursive VJPs.

CUDA and Triton must agree in raw bytes within a pinned environment. Compare
batch placement, token chunking, padding and graph replay the same way.
Independent FP64 complex arithmetic provides an accuracy oracle. Upstream's GPU
complex multiplication may contract arithmetic, so its comparison uses declared
tolerances, not an unverified bitwise claim. Representative test tolerances are
`8e-7` for FP32, `0.002` for FP16 and `0.02` for BF16 (absolute and relative).
Non-finite propagation follows the explicit arithmetic; NaN payload equality is
not a cross-backend guarantee.

## Backends and dispatch

- `cuda`: production function in `csrc/cuda/rope/multi_axis_rope.cu`, registered
  in `rl_engine._C` by ordinary CUDA builds. Does not require Hopper-specific PTX.
- `triton`: adapts FlashAttention's existing interleaved rotary kernel, with
  explicit shared/per-sample table strides and `enable_fp_fusion=False`.
  The upstream tile geometry is retained; this profile is limited to NVIDIA CUDA.
- `reference`: PyTorch FP32 arithmetic, including CPU.
- `auto`: CUDA → Triton → reference on NVIDIA, reference on other platforms.

Explicit `cuda`/`triton` requests fail when unavailable; they never silently switch
implementations. Strict callers select an explicit backend and retain `op.trace()`.
`auto` is for diagnostic use and is not a strict-route certification. The legacy
registry also exposes `kernel_registry.get_op("multi_axis_rope", device=...)`.
This PR does not add an Ascend backend or certify ROCm. Existing Ascend work in
PR #411 can share semantics once its reference contract is aligned.

Both GPU paths expose a PyTorch custom operator with fake-tensor and autograd
registration. `torch.compile(op, fullgraph=True)` supports forward and backward.
The native implementation is registered through `TORCH_LIBRARY`; Python does
not attach symbols to an unrelated extension at validation time.

## Related work and reuse

The Triton kernel comes from FlashAttention
[`98c1cee`](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/ops/triton/rotary.py),
with its BSD-3-Clause copyright and full license retained beside the adapted
source and included in wheels/source distributions. Its default fused arithmetic
can differ in low bits; disabling contraction gives the declared profile.
The benchmark loads the original kernel and autograd wrapper from checksum-pinned
source files to compare the default and contraction-disabled variants.

[Ascend PR #411](https://github.com/RL-Align/RL-Kernel/pull/411) covers the same
roadmap row on another platform. Its current half-split reference and uncentered
positions differ from the pinned Diffusers contract here. Combining the two
references requires semantic alignment. Existing Qwen-Image implementations in
[SGLang](https://github.com/sgl-project/sglang/tree/a1966e1b5d8b750b494d7e53cca1d2464f960038/python/sglang/multimodal_gen)
and [vLLM-Omni](https://github.com/vllm-project/vllm-omni/tree/c9c288e3f5828f2a2a9b07acb16b3b709906059a)
also motivate stride support and future QK-norm/RoPE fusion. This contribution
qualifies the unfused operator; it makes no new-algorithm or fused-model claim.

## Validation and benchmarks

```bash
python -m pytest tests/ops/rope/test_multi_axis_rope.py -q -rs
python tools/validation/operators/check_operator.py --op multi_axis_rope \
  --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 7 --n-heads 3 --check-grad
python -m benchmarks.operators.rope.benchmark_multi_axis_rope --output rope.json
python tools/validation/operators/validate_multi_axis_rope.py --ws1 cuda --output ws1-cuda.json
python tools/validation/operators/validate_multi_axis_rope.py --ws1 triton --output ws1-triton.json
mkdocs build --strict -f mkdocs.yaml
```

For operator-only native compilation, without building every unrelated extension:

```bash
python tools/validation/operators/validate_multi_axis_rope.py -q -rs
python tools/validation/operators/validate_multi_axis_rope.py --benchmark --output rope.json
```

To audit against the actual pinned upstream symbols without installing Diffusers,
download the source file linked above and run:

```bash
python tools/validation/operators/check_qwenimage_rope_upstream.py \
  --source transformer_qwenimage.py --output upstream.json
```

The audit checks the source checksum before loading the two relevant definitions,
then compares image/text tables and forward/backward across three dtypes.

The focused runner loads the **same production CUDA source and dispatcher
registration**, without a generated binding or extension monkeypatch. It reuses
an installed native library when available. Set a matching `CUDA_HOME`, CUDA
architecture and host compiler before invoking it.

Ordinary CUDA builds include the source in `rl_engine._C`. An explicit operator
wheel profile builds `rl_engine._rope_cuda` with the same dispatcher registration:

```bash
RL_KERNEL_ROPE_ONLY=1 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/rl_kernel-*.whl
# Change to a directory outside the source checkout before this command:
python /path/to/RL-Kernel/tools/validation/operators/check_multi_axis_rope_install.py \
  --package-root /path/to/site-packages --output installed-wheel.json
```

On Windows set `$env:RL_KERNEL_ROPE_ONLY='1'` in PowerShell and enter the matching
MSVC build environment. Select a CUDA toolkit matching `torch.version.cuda`.
The operator profile does not certify the complete collection of unrelated
native operators. CPU-only/ROCm builds reject this explicit CUDA wheel profile.

`multi_axis_rope` has an `OP_SPECS` entry, standard input builder and gradient
adapter. `--ws1` runs the existing forward and gradient accuracy/invariance APIs
with C1's elementwise tolerance profile and C2's logical configuration transforms.
This gate qualifies the Qwen-Image operator and does not certify the Dense Qwen3
chain, distributed execution, or model assembly.

The standard CLI and WS1 forward/gradient adapters share a centered image-grid
prefix from `qwen_image_positions`. The square grid contains the full logical
sample; a nonsquare sequence length uses its row-major prefix. WS1 selects rows
by `(sample_id, token_position)` from that sample's grid, so physical chunking,
reordering and padding cannot change active-token coordinates. Padding uses
zero coordinates. These are operator fixtures, not a model's image/text assembly.

Backward provenance is recorded inside the opaque custom-op implementation,
outside fake/AOT tracing. Its Python `execution_count` is not a CUDA Graph replay
counter: replay executes the captured GPU kernels without re-entering Python.

For a complete native package installation on Linux, run the dedicated gate:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .
ROPE_PYTHON=python bash ci/scripts/run_qwenimage_rope_ws1.sh
```

The gate requires a real NVIDIA GPU, Triton and an ordinary `rl_engine._C`
containing the native RoPE registration. It rejects the operator-only profile
and fast-math mode, checks required tests actually execute, and permits only
the two-device guard to skip on a single-GPU host. Results include the loaded
binary fingerprint, ordinary operator CLI checks and both WS1 backend reports.
Set `ROPE_WS1_OUTPUT` to select the artifact directory.

On WSL2, keep the checkout and Python environment in the Linux filesystem.
GNU linking of the package's CUDA driver calls may need
`LIBRARY_PATH=/usr/lib/wsl/lib`. Record the WSL kernel and physical GPU in the
evidence; a WSL2 profile does not qualify a bare-metal Linux server or another
GPU architecture. The `Qwen-Image RoPE` workflow runs CPU contracts and changed-file
checks for PRs targeting `test-qwenimage`; GPU qualification uses the gate above.

Tests cover independent forward/VJP accuracy, adjacent-pair layout, shared and
batched positions, odd and rectangular grids, batch/padding/chunk invariance,
non-contiguous tensors, packed QKV views without copies, default-device contexts,
empty inputs, boundary rejection, higher derivatives, CUDA/Triton byte equality,
graph replay, the current CUDA stream, `torch.library.opcheck` and full-graph
compilation. A two-device guard test skips when only one GPU is present.

The benchmark covers a short synthetic case and 64×64, 83×83 and 104×58 grids,
with 24 heads and both contiguous/packed-QKV layouts. It checks forward and input
gradients before timing and records CUDA Graph and synchronized eager wall time,
extra peak allocation, source hashes and actual-backend traces. Table generation
and compilation are excluded. These measurements establish no model speedup.

To include the mature FlashAttention baselines, download `flash_attn/ops/triton/rotary.py`
and `flash_attn/layers/rotary.py` at the linked revision, then pass both paths:

```bash
python tools/validation/operators/validate_multi_axis_rope.py --benchmark \
  --flash-source /path/to/ops/rotary.py --flash-layer-source /path/to/layers/rotary.py \
  --output rope-with-flash.json
```

The normalized source checksums are enforced before loading the audited symbols.
The default variant retains upstream arithmetic; `flash_no_fma` changes only the
kernel launch contraction setting and must match the FP32 gold's output bytes.
