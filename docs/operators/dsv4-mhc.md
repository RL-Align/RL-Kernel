# DSV4 mHC Pre, Post, Head and Layer End (ROCm, tasks 1–4 and 8)

## Summary

DeepSeek-V4 keeps n = 4 residual streams of hidden size C per token. Before each
sublayer, mHC pre mixes the streams into one input; after it, mHC post mixes the
sublayer output back into the streams; after the last layer, mHC head contracts the
streams for the final norm and lm_head. The layer end runs mHC head, the final norm,
lm_head and the logprob. These are tasks 1–4 and 8 of the DSV4 ROCm plan
([#508](https://github.com/RL-Align/RL-Kernel/issues/508)). The ROCm ops are batch
invariant: a token's output bytes do not depend on the other tokens in the call.

```text
x      = FP32(flatten(residual))                       [K = n * C]
r      = 1 / sqrt(sum(x^2) / K + rms_eps)              mHC pre
r      = rsqrt(sum(x^2) / K + rms_eps)                 mHC head
pre    = sigmoid(r * scale[0] * (x @ fn[:n]^T) + base[:n]) + hc_pre_eps
post   = hc_post_mult_value * sigmoid(r * scale[1] * (x @ fn[n:2n]^T) + base[n:2n])
comb   = sinkhorn(r * scale[2] * (x @ fn[2n:]^T) + base[2n:])
out[j] = BF16(post[j] * y + sum_h comb[h, j] * residual[h])    mHC post
```

## Entry Point

```python
from rl_engine.ops.mhc.training import mhc_head, mhc_post, mhc_pre
from rl_engine.models.deepseek_v4.layer_end import mhc_layer_end

post_mix, comb_mix, layer_input = mhc_pre(residual, fn, scale, base, backend="rocm_triton")
new_residual = mhc_post(y, residual, post_mix, comb_mix, backend="rocm_triton")
hidden = mhc_head(residual, head_fn, head_scale, head_base, backend="rocm_triton")

stages = mhc_layer_end(residual, head_fn, head_scale, head_base,
                       norm_weight, lm_head_weight, target_ids, backend="rocm")
# {"head", "final_norm", "logits", "logprob"}
```

The mHC entry is task 6's (`rl_engine/ops/mhc/training.py`); `backend` is mandatory.
`"rocm_triton"` runs the Triton ops of this page; `"reference"` (CPU) and `"rocm"`
(gfx942 / gfx950) run task 6's PyTorch row arithmetic, the reference. To train `fn`,
`scale` or `base` with `"rocm_triton"`, pass `logical_keys` and `parameter_id` inside
a canonical backward session (`rl_engine/ops/autograd/canonical_backward.py`).
`mhc_layer_end` returns every stage, so a consistency check can report the first stage
whose bytes differ. Per backend it runs:

| Stage | `backend="rocm"` | `backend="reference"` |
| --- | --- | --- |
| `head` | `TritonMHCHeadOp` | task 6's `mhc_head`, `"reference"` on CPU, `"rocm"` on GPU |
| `final_norm` | task 5's `rmsnorm_residual(..., backend="rocm")`, `y` only | `rmsnorm_residual(..., backend="reference")` |
| `logits` | `TritonBatchInvariantLinearOp`, fp32 output | `F.linear` in fp32 |
| `logprob` | `TritonBatchInvariantLogpOp`, `validate=True` | `NativeBatchInvariantLogpOp`, `validate=True` |

## Backends

| Backend | Wrapper | Implementation | Status |
| --- | --- | --- | --- |
| ROCm (gfx942) | `TritonMHCPreOp`, `TritonMHCPostOp`, `TritonMHCHeadOp` | `rl_engine/backends/rocm/mhc/triton_mhc.py` (Triton forward copied from aiter `326cffa`; RL-Kernel's own Triton backward) | MI300X / MI355X acceptance not run |
| Reference | `mhc_pre`, `mhc_post`, `mhc_head` with `"reference"` / `"rocm"` | `rl_engine/ops/mhc/training.py` (task 6), following DeepSeek-V4-Flash-0731 `inference/model.py` | CPU; gfx942 / gfx950 |
| AITER | — | — | Not used; see **AITER integration boundary** |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `residual` | `[T, n, C]` | bf16 | n = 4 |
| `fn` | `[2n + n², n·C]` | fp32 | `hc_attn_fn` / `hc_ffn_fn` |
| `scale`, `base` | `[3]`, `[2n + n²]` | fp32 | |
| `post_mix`, `comb_mix` | `[T, n]`, `[T, n, n]` | fp32 | |
| `layer_input`, `y`, `hidden` | `[T, C]` | bf16 | |
| `head_fn`, `head_scale`, `head_base` | `[n, n·C]`, `[1]`, `[n]` | fp32 | |
| `norm_weight` | `[C]` | fp32 | task 5's contract; `norm_eps` exactly 1e-6 |
| `lm_head_weight` | `[V, C]` | bf16 | as `head.weight` in the checkpoint |
| `target_ids` | `[T]` | int64 | outside the vocabulary and not -100 raises |

`comb_mix` is indexed [source stream, destination stream]. The `TritonMHC*Op` classes
also accept leading dimensions, flattened into M. Non-contiguous inputs are copied to
contiguous ones.

## Dispatch Behavior

The mHC entry and `mhc_layer_end` use the backend the caller names; there is no
default and no fallback, and an unknown or unavailable backend raises. The Triton op
constructors raise unless the GPU is gfx942 and name the architecture. Inputs outside
the tensor contract raise before any kernel launch. `kernel_registry.get_op("mhc_pre")`
and the semantic catalog still return the same ops for other callers. The arithmetic
each backend follows is recorded in `rl_engine/contracts/operators/mhc.py`
(`describe("mhc_pre")` and so on).

## Accuracy

The reference computes in fp32 and casts `layer_input`, the post output and the head
output to bf16, as DeepSeek-V4 does. The Triton ops, forward and gradients, match it
within the WS1 contract tolerance
(`rl_engine/contracts/profiles/precision/ws1.json`, bfloat16 `reduction` cells of
`forward_accuracy` and `gradient_accuracy`). Batch invariance is checked by bytes.

The arithmetic that fixes the bytes:

- mHC pre and head run a split kernel over `NUM_KSPLIT = 4` segments of K, then add the
  segments in segment order, for every M.
- The sum of squares is accumulated element-wise with `tl.fma` over K tiles of
  `BLOCK_K = 128`, then the 128 columns are added pairwise.
- The projection is `tl.dot` with fp32 `fn`, `input_precision="ieee"` and a fixed MFMA
  shape (`matrix_instr_nonkdim=16`, `kpack=1`).
- The stream sums are one `tl.fma` per stream in stream order, rounded to bf16 once;
  mHC post computes `post · y`, then one `tl.fma` per source stream.
- Sinkhorn row and column sums are pairwise.
- Backward: sums over C are accumulated element-wise with `tl.fma` over tiles of 128,
  then added pairwise; the transposed projection is one `tl.fma` per `fn` row in row
  order; Sinkhorn steps are replayed from the saved logits. The kernels run with
  `enable_fp_fusion=False`.
- `fn`, `scale` and `base` gradients go through the canonical backward session
  (`rl_engine/ops/autograd/canonical_backward.py`), which reduces them in logical-key
  order; without a session a trainable parameter raises.

`NUM_KSPLIT`, `BLOCK_K` and the MFMA options change the bytes; the launch configs only
group tokens and hidden positions. M is not specialized, so a new M compiles nothing.
The RFC states the controller RMS as `1 / (sqrt(mean) + eps)`; the ops implement the
DeepSeek-V4 reference's `rsqrt(mean + eps)`, pending the RFC author's ruling.

## AITER integration boundary

At the pinned AITER revision
[326cffa](https://github.com/ROCm/aiter/tree/326cffa38c9eda16bef19375ba55334e4d3acd59),
`aiter/ops/triton/fusions/mhc.py` provides `mhc_pre_dsv4`, `mhc_post_dsv4` and
`mhc_head_dsv4`. They choose the tile shape, the number of K segments and between a
fused and a split kernel by M, sum with `tl.sum`, whose order follows the tile layout,
and round `fn` to bf16; their backward recomputes the formula in PyTorch. Their bytes
therefore change with M. This PR copies the Triton kernels into RL-Kernel with the
changes under **Accuracy**, adds a Triton backward, and does not modify AITER.

## Performance Notes

Not measured on MI300X or MI355X. The benchmark times the forward of mHC pre, post
and head for M = 1 to 65536 (n = 4, C = 4096, 20 Sinkhorn rounds) against AITER's ops
with their default configs, and reports the error of both against the fp32 reference
at M = 1024:

```bash
PYTHONPATH=<aiter checkout> AITER_TRITON_ONLY=1 \
    python benchmarks/operators/mhc/benchmark_dsv4_mhc.py --output mhc.md
```

## Tests

```bash
# CPU: contract, batch-invariance checks, layer end reference, dispatch.
python -m pytest -q tests/contracts/operators/test_mhc_contract.py \
    tests/validation/common/test_batch_invariance.py \
    tests/models/deepseek_v4/test_layer_end.py tests/runtime/test_dispatch.py

# ROCm GPU
python -m pytest -q tests/backends/rocm/test_rocm_mhc_pre.py \
    tests/backends/rocm/test_rocm_mhc_post.py tests/backends/rocm/test_rocm_mhc_head.py \
    tests/backends/rocm/test_rocm_mhc_batch_invariance.py \
    tests/backends/rocm/test_rocm_mhc_backward.py \
    tests/backends/rocm/test_rocm_dynamic_shape.py \
    tests/models/deepseek_v4/test_layer_end.py \
    tests/models/deepseek_v4/test_layer_end_batch_invariance.py
```

The GPU suite checks a row alone and in batches of M = 1 to 16384, chunking,
reordering, padding, input strides and repeated runs, forward and input gradients, at
C = 4096 and 7168; launch configs bit for bit; the projection, RMS, Sinkhorn and stream
aggregation with exact inputs; outputs and gradients against the reference; the
entry's `"rocm_triton"` backend;
parameter gradients through the canonical session under chunking and reordering;
empty batches; and that unseen M values compile nothing. Skipped GPU tests do not
establish correctness.

## Known Limitations

- gfx942 only; n = 4. `NUM_KSPLIT`, `BLOCK_K` and the launch configs are tuned for
  C = 4096; the backward launch config is not tuned.
- One GPU, no tensor parallelism.
- The Sinkhorn backward replays the earlier steps for every step it walks back, trading
  time for memory.
- The canonical session keeps every call's residual rows until the last use of `fn`.
- The layer end's `norm_weight` and lm_head weight gradients are reduced over the
  call's rows, not through the canonical session.
- Non-contiguous inputs cost a copy.
- Graph capture belongs to task 7 ([#510](https://github.com/RL-Align/RL-Kernel/issues/510)).
