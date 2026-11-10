# Fused Post-Norm, Residual Add, Pre-Norm

Muse-Glimmer-30B's decoder layer norms twice per sublayer: a post-norm on the
sublayer output, then a residual add, then a pre-norm for whatever comes next.
This operator fuses that tail. It is the `fused_post_norm_residual_pre_norm` row of
RFC #501 ([F1]) and follows the WS1 dual-path contract: a pure-PyTorch FP32 ground
truth plus CUDA and Triton candidates validated against it.

```text
sublayer_out --post_norm--> (+) residual_in --> residual_out
                              |                     |
                              +------ pre_norm ---> pre_normed --> next sublayer
```

Two outputs, because both are live: `residual_out` continues the residual stream and
`pre_normed` feeds the next sublayer. The model's final norm is a separate, simpler
operator (`NativeFinalNormOp`) because the fingerprint spells it differently.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

fused = kernel_registry.get_op("fused_post_norm_residual_pre_norm")
residual_out, pre_normed = fused(hidden, residual, post_weight, pre_weight)
# eps default to the model's values; pass post_eps= / pre_eps= to override
```

- `forward(...)` computes in FP32 and returns the input dtype.
- `forward_fp32(...)` on the reference computes and returns FP32, and is the gold.

## Contract

Every rule below comes from the `frozen_semantics` block of the #558 architecture
fingerprint, not from the HF config, because the config does not state most of it.

| Item | Value |
| --- | --- |
| Decoder norm form | `x * rstd * (1 + w)`, the `1 +` applied in FP32 inside the operator |
| Reciprocal square root | `rsqrt` for the decoder norms; `pow_neg_half` for the final norm |
| `post_eps` | `post_norm_eps = 1e-8`, for `post_attention_layernorm` and `post_feedforward_layernorm` |
| `pre_eps` | `rms_norm_eps = 1e-5`, for `input_layernorm` and `pre_feedforward_layernorm` |
| Compute dtype | FP32 throughout, `weight_applied_before_cast` |
| Hidden width | 6656; the operator is width-agnostic but the tests qualify this width |

Two narrowing points are part of the contract, not incidental. The post-norm's cast is
its last step, so the value entering the add is already in the model dtype; and the
sum is narrowed again before the pre-norm reads it, because eager HF adds in the model
dtype and the next norm sees that rounded value. `forward_fp32` has neither narrowing,
which is why it is a different computation rather than `forward` widened, and a test
asserts the two disagree in BF16 so nobody collapses them later.

The `1 +` has to reach the kernel as a parameter. Folding it into a BF16 weight
beforehand rounds the offset away.

The backward has one rule of its own, the mirror of the second narrowing. The
residual output is a model-dtype tensor with two gradient consumers, and eager
autograd accumulates both arrivals in that dtype: `round(g_res + round(d_from_pre))`.
Both kernels do the same. Keeping that sum in FP32 is one ULP per element, which the
post-norm weight gradient then sums over rows; measured at bf16 against the
reference's own autograd it lands 2.6 to 2.9 times outside the contract row at 512
and 2048 rows, a single rounding of the FP32 sum does not help, and the eager
accumulation brings it to within 0.125 and then to 0. A reference test asserts both
halves of that so the rounding cannot be removed as a simplification.

## Backends

| Backend | Class | Notes |
| --- | --- | --- |
| PyTorch | `NativeFusedAddNormOp` | gold; `shape_invariant_meansq` reduction |
| Triton | `TritonFusedAddNormOp` | one program per row, explicit fold in the CUDA kernel's order |
| CUDA | `CudaFusedAddNormOp` | one block per row, shared-memory tree |

Both GPU kernels save the two per-row statistics from the forward and reuse them in
the backward, so the backward cannot drift from what the forward used. Weight
gradients come back as per-row FP32 contributions and are folded with
`reduce_rows_fp32`, the same declared reduction order the other norms use.

## Accuracy

- **Independent FP64 reference.** The reference is checked against definitions
  written a second time in FP64 without importing the production helpers, forward
  and closed-form backward, with random non-zero cotangents on both outputs. Guard
  tests assert the FP64 comparison rejects a dropped `(1 + w)` offset and a swapped
  eps pair, so the check cannot pass vacuously.
- **Tolerance class.** This is a reduction, not an elementwise op: `OP_SPECS` sets
  `op_class="reduction"` and both GPU kernels are held to those rows against the
  gold, FP32 and BF16 exactly. FP16 is `optional` in the contract and cannot meet
  the declared 1e-3: the kernels differ from the gold by one FP16 ULP on a handful
  of elements where the reduction order lands on the far side of a tie in the
  residual add, so FP16 is held to one ULP of the reference instead. Measured, that
  costs nothing: the relaxed FP16 bar catches exactly the same injected bugs the
  required BF16 bar catches.
- **Axis A, batch invariance.** Asserted bitwise for both kernels, forward and
  backward: a row's bytes do not change with batch size or position.
- **CUDA vs Triton.** **Bitwise equal** on both outputs and all four gradients, at
  FP32, BF16 and FP16, which is what RFC #501 asks for. A row reduction does not
  get there by writing the same source twice; three things had to be made to agree
  and all three are load-bearing:

  | what differed | why | how it is pinned |
  | --- | --- | --- |
  | reduction order | `tl.sum` picks its own tree | Triton folds in the CUDA kernel's order: lane `t` over `x[t + k*LANES]`, `k` ascending, then a stride-halving tree. `_reduction_shape` and `choose_threads` share the lane counts |
  | divide by the width | Triton's `/` is `div.full.f32`, approximate; CUDA's is `div.rn.f32` | `tl.div_rn` |
  | reciprocal square root | nvcc's `rsqrtf` is libdevice's refined non-FTZ path; `tl.rsqrt` is `rsqrt.approx.ftz.f32` | the CUDA kernel emits the same instruction |

  Sizes, measured before each fix, so the order of attack is on record: with the
  reduction orders matched but the divide still approximate, the statistics were
  one ULP apart on 22% of rows; the rsqrt difference alone accounted for about 12%
  of inputs. FP contraction had to be off first, otherwise the compiler folded the
  square into the first level of the fold.

  `test_cuda_and_triton_are_bitwise_equal` holds all three, over five shapes and
  three dtypes, and compares byte views rather than `==` so that a signed zero or
  a NaN cannot pass. `test_rms_norm.py` still compares its backends with a
  tolerance, "precisely because the reduction order is allowed to differ"; that is
  the older position and this row is evidence the order can be pinned instead, at
  no cost that shows above measurement noise.

## Ground-truth harness

Registered in `OP_SPECS` with both outputs and all four gradients checked:

```bash
python tools/validation/operators/check_operator.py --op fused_post_norm_residual_pre_norm --candidate triton --dtype bf16 --device cuda --check-grad
python tools/validation/operators/check_operator.py --op fused_post_norm_residual_pre_norm --candidate cuda --dtype bf16 --device cuda --check-grad
```

The two eps are fixed to the model's values inside the input generator; the CLI does not
expose them, which is deliberate since swapping them is the mistake the contract guards
against. `--normalized-dim` sets the width (default 4096; the Muse width is 6656).

## Tests

```bash
python -m pytest tests/ops/norm/test_fused_add_norm.py tests/ops/norm/test_fused_add_norm_triton.py tests/ops/norm/test_fused_add_norm_cuda.py -v
```

Reference: FP64 forward and backward, guards against a dropped offset and swapped
eps, the eager-composition narrowing check, fail-closed on shape, dtype and weight
width. Each GPU kernel: forward and backward against the gold at the reduction rows,
batch invariance bitwise, eps wiring, fail-closed. CUDA against Triton at the
reduction rows.

## Implementation Files

- `rl_engine/reference/norm/fused_add_norm.py`: gold, closed-form backward, final norm
- `rl_engine/reference/norm/rms_norm.py`: `shape_invariant_meansq` / `shape_invariant_rowsum`
- `rl_engine/backends/shared/triton/norm/fused_add_norm.py`: Triton kernels and wrapper
- `rl_engine/backends/cuda/norm/fused_add_norm.py`: CUDA wrapper
- `csrc/cuda/norm/fused_add_norm.cu`: CUDA kernels
- `rl_engine/validation/operators/operator_specs.py`, `operator_inputs.py`
- `tests/ops/norm/test_fused_add_norm*.py`

## Known Limitations

- `hidden` and `residual` must share shape and dtype; no broadcasting. Weights are
  1-D at the hidden width.
- A row must fit one Triton program (`BLOCK_H` up to 65536), so the reduction order
  stays independent of batch layout. Wider rows raise rather than silently splitting.
- The model's final norm (`rmsnorm_times_weight`, `pow_neg_half`) is the reference
  `NativeFinalNormOp` only; it has no GPU kernel in this row.
