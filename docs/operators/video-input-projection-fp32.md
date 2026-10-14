# Video Input Projection FP32

## Summary

MiniMax-H3 maps each 96-element video patch to the 5376-wide Transformer residual
space. This operator preserves the FP32 projection boundary of the pinned H3
reference. The model adapter owns the subsequent cast into the BF16 packed stream.

The primary implementation reuses RL-Kernel's Triton FP32 GEMM and canonical
parameter-gradient session. A pinned CUTLASS FP32 SIMT GEMM is an explicit CUDA
candidate. Both use the same semantic adapter and autograd implementation.

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
| Triton on CUDA | `TritonVideoInputProjectionFP32Op` | `_det_gemm_fp32_kernel` | Default CUDA path; SM90 qualification |
| CUDA / CUTLASS | `CUDAVideoInputProjectionFP32Op` | `_C_cutlass.gemm_fp32_tn` | Explicit candidate; optional build; SM90 |
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
once. Ordinary chunk-by-chunk accumulation is a different arithmetic contract.

Long K reductions use a fixed midpoint tree with at most 128 terms per leaf.
Each leaf reuses the selected GEMM and every tree node adds in FP32. This bounds
rounding depth for the 5376-term input VJP and long parameter reductions. The
tree depends on logical K and retains its topology across physical row changes.
Existing canonical linear callers keep their original reduction by default.

Triton fixes the 64x64x32 leaf schedule and disables TF32 and FP fusion. CUTLASS
fixes 64x64x8 FP32 SIMT tiles and a single output owner per leaf, without parallel
split-K. Their leaf arithmetic has separate backend identities.

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
The fixed tree adds kernel launches and layout copies to backward; compare full
operator timings before drawing a speedup conclusion.

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
