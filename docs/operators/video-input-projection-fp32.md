# Video Input Projection FP32

## Summary

MiniMax-H3 maps each 96-element video patch to the 5376-wide Transformer residual
space. This operator preserves the FP32 projection boundary of the pinned H3
reference. The model adapter owns the subsequent cast into the BF16 packed stream.

The implementation reuses RL-Kernel's Triton FP32 GEMM and canonical
parameter-gradient session. Input and weight VJPs use fixed 128-term K chunks
and one balanced FP32 merge.
Long bias VJPs retain the one-row Triton tree.
Only this one-row FP32 tree uses compact paired-leaf scheduling. Other FP32 tree
shapes reuse the existing default tile and per-level schedule.

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
| Triton on CUDA | `TritonVideoInputProjectionFP32Op` | `_det_gemm_fp32_kernel`, `_det_gemm_fp32_chunk_merge_kernel`, bias tree kernels | Default CUDA path; SM90 qualification |
| PyTorch reference | `NativeVideoInputProjectionFP32Op` | CPU `F.linear`; FP32 product/sum VJPs | Accuracy reference |

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

`get_op` selects Triton on CUDA. `OP_SPECS` exposes candidate `triton` and the
independent PyTorch accuracy reference. The semantic catalog records
single-device, unsharded support and exact implementation identities. An
unavailable explicit candidate raises an error. CPU/ROCm/MUSA/NPU execution is
outside this contribution's strict support scope.

The wrapper reuses `canonical_linear_fp32` with `family="triton"` and
`matmul_op=projection_gemm_fp32`. The family identifies the backend. The callable
selects the existing forward and VJP GEMM schedules and handles their input
strides. Forward, dX, dW, and dbias use the same callable. Existing canonical
linear callers omit `matmul_op` and retain their original backend dispatch.
The existing `canonical_row_lm_head` uses the same separation: `family` records
the backend, and `matmul_op` supplies the GEMM implementation. Qwen3 passes its
selected `det_gemm.forward_accum_fp32` to that function.
On CUDA, the session internally checks eligible keys before skipping sorting.
Backward provenance records the actual matmul entry point.

## Accuracy

Forward computes `Y = X @ W.T + bias` in FP32. The Triton forward adds bias
in the existing GEMM epilogue, preserving the rounded FP32 GEMM result before
the bias addition. The upstream gradient `G`
stays FP32, including normalization of non-contiguous autograd inputs:

- `dX = G @ W` is row-local.
- `dW = G_sorted.T @ X_sorted` follows canonical logical-row order.
- `dbias = ones[1,R] @ G_sorted` uses the fixed FP32 GEMM for reductions up to 96
  and the one-row Triton tree for longer reductions.

The existing canonical session collects all contributions, removes padding, and
sorts keys before the final parameter reduction. It returns each parameter VJP
once. Weight and bias share sorting and the sorted upstream gradient when their
complete contribution sequences match. Calls with different contribution sets
normalize each parameter independently.

For at least 4096 submitted CUDA rows, the session checks that every row is
active and adjacent keys strictly increase in logical order. This includes
block boundaries. When the check succeeds, the session reuses contiguous
gradients without sorting or gathering the large matrix. Otherwise it runs the
existing stable sort. The check reads the current keys on each normalization.
Callers supply logical keys and do not select this optimization. Smaller inputs,
CPU, and other device paths retain stable sorting.

Short reductions up to the 96-wide video input use the existing direct FP32
GEMM. The Triton input and weight VJPs reuse that kernel with fixed 128-term
chunks. Each chunk computes four ascending 32-term FP32 dots from zero. It adds the
first pair and the second pair independently, then combines both 64-term halves.
This balanced sum reduces cancellation error in the long weight VJP. Chunk
partials remain FP32. A second grid uses a fixed balanced reduction over the
next power-of-two chunk count, with zero padding. The merge uses 64 output
elements per program and four warps. VJP GEMMs use fixed 128-by-32 output tiles,
32-term dot tiles, and four warps. The forward retains its 64-by-64 output tile.
These settings define the v10 Triton implementation identity. The K partition and VJP
tile do not depend on M, row placement, or occupancy.
The VJP kernels consume positive-stride views directly, including `G.T`.
The Triton forward omits K or N load predicates only when the corresponding
32-term or 64-column tiles are complete. It retains M predicates for every row.
VJP kernels use the compiler's default register allocation for every input layout.
The K accumulation order is fixed. No configuration depends on physical row count.

The fixed-chunk design follows the scheduling separation in merged PR #396.
Its AMD BF16 MFMA instructions are not used by this CUDA FP32 implementation.
The FP32 GEMM body is reused from the helper present in merged PR #325.
The short forward path and long bias tree retain their earlier arithmetic.
The merge layout defines its reduction instruction schedule. The implementation
pins 64 output elements and four warps as part of its implementation identity.

The VJP schedule caps its FP32 partial workspace at 512 MiB by chunking
output rows. The retained midpoint helper also supports output-column chunking
when one complete row exceeds its budget. Parameter sorting and public validation are included in complete
operator measurements. Existing BF16 callers retain their dtype and schedule.
TF32 and compiler multiply-add fusion remain disabled.
The reused Triton `tl.dot` lowers its IEEE FP32 inner products to fixed SIMT
`fma.rn.f32` instructions. The compiler fusion flag controls surrounding
expressions; it does not remove the intrinsic dot's FMA instructions. This
arithmetic is included in the Triton contract and runtime metadata.

The independent reference keeps the forward definition as CPU FP32 `F.linear`.
Its analytic VJPs form FP32 products and use CPU PyTorch `sum` over the complete
K dimension. Only output rows and columns are tiled to bound temporary memory.
It does not reuse candidate K partitions or GPU kernels. This avoids long CPU
GEMM accumulation errors observed in the 13104-row unscaled regression.
The reference identity is `cpu-fp32-linear-products-sum-v2`. FP64 is used only
in a small reference-validation test and diagnostic evidence, not by the
reference implementation or candidate kernels.

The shared WS1 `reduction` FP32 accuracy policy uses `atol=rtol=1e-4` for forward
and gradients. Invariance uses byte equality. Parameter invariance compares the
same logical contribution set and upstream gradients across layout changes.
Adding a genuine training sample changes the expected parameter gradient.

## Performance Notes

```bash
PYTHONPATH=. python benchmarks/operators/gemm/benchmark_video_input_projection_fp32.py \
  --backends pytorch triton --rows 1 128 1024 4096 13104 \
  --output /tmp/h3-input-projection.json
```

The benchmark first checks independent CPU FP32 forward/backward accuracy and
row-permutation VJPs. It measures eager forward, backward, F+B, and no-grad
forward CUDA Graph. Timings include layout copies, bias, and the canonical
parameter-gradient machinery. Raw samples, environment, source hashes, and
actual backend metadata are written to the explicitly selected local output.
PyTorch is a production performance baseline with separately reported BI results.
The Triton VJPs use one partial-GEMM launch and one merge launch when the
workspace fits.
Compare complete operator timings, including sorting, before drawing a speedup
claim. The benchmark records the reference identity and its source hash.

## Tests

```bash
python -m pytest tests/ops/gemm/test_video_input_projection_fp32.py -v
python tools/validation/operators/check_operator.py \
  --op video_input_projection_fp32 --candidate triton --device cuda --dtype fp32 \
  --batch 1 --seq 129 --check-grad --grad-mode random
```

The standard-normal regression covers seeds 420 through 423 at 129 and 13104
rows. It checks CPU FP32 accuracy and byte equality across permutations,
microbatches, padding, and repeated calls. The benchmark rejects NaN or Inf in
either the candidate or reference, including gradients.

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
