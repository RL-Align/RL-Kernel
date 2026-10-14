# Video Input Projection FP32

## Summary

MiniMax-H3 maps each 96-element video patch to the 5376-wide Transformer residual
space. This operator preserves the FP32 projection boundary of the pinned H3
reference. The model adapter owns the subsequent cast into the BF16 packed stream.

The primary implementation reuses RL-Kernel's Triton FP32 GEMM and canonical
parameter-gradient session. A pinned CUTLASS FP32 SIMT GEMM is an explicit CUDA
candidate. Both use the same semantic adapter, autograd implementation, and
existing Triton tree plan and GPU level reductions. CUTLASS substitutes the
batched leaf GEMM; its reported execution identity includes the Triton reduction.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

op = kernel_registry.get_op("video_input_projection_fp32", device=x.device)
with canonical_backward_session() as session:
    y = op(x, weight, bias, logical_keys=keys, parameter_id="transformer.proj_in")
    y.backward(grad_y)
    session.validate_complete()
```

For inference, call the same operator under `torch.no_grad()` without row keys.
For parameter gradients, keys identify original logical rows, not their current
physical placement. The same parameter keeps the same `parameter_id` across uses.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton on CUDA | `TritonVideoInputProjectionFP32Op` | `_det_gemm_fp32_kernel`, existing tree leaf/reduce kernels | Default CUDA path; SM90 qualification |
| CUDA / CUTLASS + Triton reductions | `CUDAVideoInputProjectionFP32Op` | `_C_cutlass.gemm_fp32_tn`, `_C_cutlass.gemm_fp32_tree_leaves_tn`, shared tree reductions | Explicit candidate; optional build; SM90 |
| PyTorch reference | `NativeVideoInputProjectionFP32Op` | CPU `F.linear` | Accuracy reference |

CUTLASS revision: `56ca28adb32ffadd097566545b8f7d83b652e998`. The build verifies
the complete include-tree fingerprint. The BSD-3-Clause dependency notice is
preserved in the installed CUDA Python wrapper.

```bash
git clone https://github.com/NVIDIA/cutlass.git /tmp/h3-cutlass
git -C /tmp/h3-cutlass checkout 56ca28adb32ffadd097566545b8f7d83b652e998
CUTLASS_ROOT=/tmp/h3-cutlass RL_KERNEL_BUILD_CUTLASS=1 \
  MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
```

The separate `_C_cutlass` extension fixes `--fmad=false`, `--ftz=false`, precise
division/square root, and SM90 code generation. Common `_C` compiler options are
independent. `backend_info()` reads back the candidate's revision and schedule.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[M,96]` or `[B,S,96]` | FP32 | Non-empty, contiguous |
| `weight` | `[5376,96]` | FP32 | Contiguous; same device as `x` |
| `bias` | `[5376]` or `None` | FP32 | Contiguous; same device as `x` |
| `logical_keys` | `[M,2]` or `[M,3]` | int64 | Required for trainable parameters; same device |
| `parameter_id` | Stable module identity | string | Required for trainable parameters |
| Output | `[M,5376]` or `[B,S,5376]` | FP32 | Leading dimensions preserved |

Keys use `(sample_id, patch_id[, use_id])`. Active identities are unique across
uses of one parameter; a negative first column marks padding. Register every
forward use before starting backward in the session. Fresh backward graphs use
fresh sessions. Parameter freezing registers only the parameters that need VJPs.

## Dispatch Behavior

`get_op` selects Triton on CUDA. CUTLASS is selected explicitly through the
semantic catalog or `OP_SPECS` candidate `cuda`. The semantic catalog records
single-device, unsharded support and exact implementation identities. An
unavailable explicit candidate raises an error. CPU/ROCm/MUSA/NPU execution is
outside this contribution's strict support scope.

## Accuracy

Forward computes `Y = X @ W.T`, then adds bias in FP32. The upstream gradient `G`
stays FP32, including normalization of non-contiguous autograd inputs:

- `dX = G @ W` is row-local.
- `dW = G_sorted.T @ X_sorted` follows canonical logical-row order.
- `dbias = ones[1,R] @ G_sorted` uses the same fixed FP32 GEMM bottom layer.

The existing canonical session collects all contributions, removes padding, and
sorts keys before the final parameter reduction. It returns each parameter VJP
once. Weight and bias share sorting and the sorted upstream gradient when their
complete contribution sequences match. Calls with different contribution sets
normalize each parameter independently.

Short reductions up to the 96-wide video input use the existing direct FP32
GEMM. Long reductions reuse the existing midpoint plan with at most 32 terms per
leaf. Workspace, merge nodes, and output stay FP32. One-row complete trees fuse each
Triton leaf pair with its first merge. Other complete trees compute leaves in
one grid. Subsequent
merges use the existing two-level kernel and alternating compact workspace
buffers. Ragged trees retain the original per-level schedule. CUTLASS packs
the same leaf ranges into 32-wide buffers,
pads inactive slots with zero, and uses one `device::GemmArray` launch to write
the plan's leaf nodes. Its transpose-view packer reads adjacent rows together
and transposes a 32x32 tile in shared memory. It uses the same paired level
reductions and compact buffers for complete trees.

The tree depends on logical K and retains its topology across physical row
changes. Output-row chunking keeps tree workspace below the existing element
limit (512 MiB for FP32). If one output row exceeds that limit, independent
output columns are chunked with the same K tree. CUTLASS also allocates packed
operands and pointer arrays. Parameter sorting and public validation are included
in complete-call measurements. Existing BF16 tree callers retain their default
dtype and schedule.

Triton uses fixed 64x32x32 leaves and 1x128x32 leaves for one-row tree calls,
including bias VJPs. The short forward GEMM retains 64x64x32 tiles. TF32 and FP multiply-add fusion are disabled. Schedule
fusion preserves every FP32 leaf and tree addition boundary. CUTLASS
fixes 64x64x8 FP32 SIMT tiles and a single output owner per leaf, without parallel
split-K. Both leaf algorithms have separately versioned backend identities.

The shared WS1 `reduction` FP32 accuracy policy uses `atol=rtol=1e-4` for forward
and gradients. Invariance uses byte equality. Parameter invariance compares the
same logical contribution set and upstream gradients across layout changes.
Adding a genuine training sample changes the expected parameter gradient.

## Performance Notes

```bash
PYTHONPATH=. python benchmarks/operators/gemm/benchmark_video_input_projection_fp32.py \
  --backends pytorch triton cuda --rows 1 128 1024 4096 13104 \
  --output /tmp/h3-input-projection.json
```

The benchmark first checks independent CPU FP32 forward/backward accuracy and
row-permutation VJPs. It measures eager forward, backward, F+B, and no-grad
forward CUDA Graph. Timings include layout copies, bias, and the canonical
parameter-gradient machinery. Raw samples, environment, source hashes, and
actual backend metadata are written to the explicitly selected local output.
PyTorch is a production performance baseline with separately reported BI results.
The shared GPU tree avoids Python recursion per leaf and merge node. Its
workspace traffic, output-row chunking, and CUTLASS operand packing still add
backward work; compare complete operator timings before drawing a speedup claim.

## Tests

```bash
python -m pytest tests/ops/gemm/test_video_input_projection_fp32.py -v
python tools/validation/operators/check_operator.py \
  --op video_input_projection_fp32 --candidate triton --device cuda --dtype fp32 \
  --batch 1 --seq 129 --check-grad --grad-mode random
python tools/validation/operators/check_operator.py \
  --op video_input_projection_fp32 --candidate cuda --device cuda --dtype fp32 \
  --batch 1 --seq 129 --check-grad --grad-mode random
```

Set `H3_PROJECTION_WEIGHTS` to a safetensors fixture containing `proj_in.weight`
and `proj_in.bias` from `MiniMaxAI/MiniMax-H3` at revision
`42ed227ee7df40d41602854ae760620d6eb651fe` to run the latent-pack-projection check.
The fixture metadata must include `repository` and `checkpoint_revision` with
these values. The check uses a synthetic `[1,24,13,48,84]` latent with real input
projection weights; it does not load the complete Transformer.
Keep the weights and generated evidence outside tracked source files.

## Known Limitations

Qualification targets a single H20 (SM90). Distributed parameter gradients,
interleaved forward/backward microbatch accumulation, higher-order derivatives,
gradient checkpointing, and full training CUDA Graph require further integration
contracts. Training capture fails explicitly; no-grad forward capture is tested.
This operator's evidence is one part of the H3 WS1 operator gate. Full Transformer
and complete RL trajectory acceptance belong to the corresponding model gates.
