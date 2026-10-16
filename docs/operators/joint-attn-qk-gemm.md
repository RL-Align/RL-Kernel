# Joint Attention QK GEMM

Qwen-Image WS1 operator `joint_attn_qk_gemm` from issue #386. First of the three
unfused joint-attention operators:

```text
joint_attn_qk_gemm -> joint_attn_softmax -> joint_attn_av_gemm
```

## Purpose and target workload

Qwen-Image MMDiT runs **non-causal** attention over the joint `[text, image]`
sequence: both streams are concatenated before attention, so this operator sees
one sequence and takes no causal mask. It computes the scaled score matrix

```text
scores[b, h, i, j] = ( sum_d Q[b, h, i, d] * K[b, h, j, d] ) * scale
```

with `scale = 1/sqrt(head_dim)` by default. Per-stream Q/K projection,
learned-gain Q/K RMSNorm and three-axis RoPE are all applied **before** the
call, so `q` and `k` arrive already normalised and rotated. Masking belongs to
`joint_attn_softmax`, not here; this operator emits raw scores.

Model geometry: 60 blocks, hidden size 3072, 24 heads of 128, bf16 checkpoint.
At the three reference resolutions (1024², 1328², 1664x928) the image stream
contributes 4096, 6889 and 6032 tokens, to which the padded text prompt is
prepended.

## Public Python entry point

```python
from rl_engine.runtime.registry import KernelRegistry

op = KernelRegistry().get_op("joint_attn_qk_gemm", device="cuda")
scores = op(q, k, scale=None, query_padding_mask=None, key_padding_mask=None)
```

`forward` casts once to the input dtype; `forward_fp32` returns fp32 and is the
gold-comparison path. Both are differentiable in `q` and `k`.

## Frozen numeric contract

Contract version `joint-attn-qk-gemm-tree-v1`.

1. **Forward reduction tree (head dim).** Each score element reduces over the
   head dimension `D` only. `D` splits into 32-wide leaves (the last may be
   short); each leaf is an ascending-`d` fp32 FMA chain started from `+0.0`, and
   leaves combine through a mid-split tree `T(l, r) = T(l, m) + T(m, r)`.
   Because the tree depends only on `D` -- a per-head architectural constant --
   **every forward score element is batch-, position-, padding- and
   launch-geometry-invariant by construction.** At the production `D = 128` the
   tree is exactly four leaves.
2. **Multiply-add discipline.** One correctly-rounded FP32 FMA at every
   multiply-into-add site. The CUDA kernels use explicit `__fmaf_rn`, so results
   do not depend on compiler FP contraction; there is no `acc += a * b` anywhere.
   The reference uses `torch.addcmul`. RNE rounding everywhere.
3. **scale.** Applied once per output element, in fp32, *after* the complete
   tree and before the single output cast. Never folded into `q`, never applied
   inside a leaf chain or a tree node.
4. **Backward reductions (sequence dims).** `dQ` reduces over `Sk` and `dK` over
   `Sq`, under the same 32-wide-leaf mid-split tree. Those lengths depend on
   prompt padding, so padding invariance requires the explicit masks: valid
   positions are compacted into ascending logical order before the tree is
   built, and padded output rows are defined as exactly zero.
5. **Excluded.** No Split-K, Stream-K, atomic accumulation, TF32, fast math or
   autotuning.

The tree discipline deliberately matches `attn-out-bias-gemm-tree-v1` (issue
\#386 row `attn_out_bias_gemm`) so the two GEMM rows share one reduction
definition. Once that lands, `num_leaves` and `tree_from_leaf_evaluator` should
be de-duplicated into a shared module.

### CPU is the bit-defining device

`torch.addcmul` is a correctly-rounded single-rounding FP32 FMA **on CPU** but
lowers to a separate multiply and add **on CUDA**, where it is bit-identical to
`c + a * b`. The same inputs therefore produce different bits on the two
devices. The canonical probe is `a = b = 1 + 2**-12`, `c = 2**-60`: a true FMA
returns `0x3F801001`, a double-rounded multiply-then-add returns `0x3F801000`.

| Expression | CPU | CUDA |
| --- | --- | --- |
| `torch.addcmul` | correctly-rounded FMA | multiply-then-add |
| `c + a * b` | multiply-then-add | multiply-then-add |

Consequently the fp32 gold is pinned to CPU, which is exactly the "independent
FP32 CPU reference" the issue asks for. `forward_fp32` **fails closed** on any
device failing the probe rather than silently returning different bytes, and the
gtest gold adapter always evaluates the reference on CPU.

## Backends and fallback

| Backend | Path | Notes |
| --- | --- | --- |
| PyTorch reference | `rl_engine/reference/attention/joint_attn_qk_gemm.py` | Fixed-order fp32 gold. Explicit elementwise tree; no `matmul`, no SDPA. |
| CUDA | `rl_engine/backends/cuda/attention/joint_attn_qk_gemm.py`, `csrc/cuda/attention/joint_attn_qk_gemm.cu` | Forward closes the head-dim tree in registers. Backward emits one leaf partial per kernel and merges on the host with the reference's own tree. |

Dispatch priority: `cuda` → CUDA then PyTorch; `rocm`, `musa`, `cpu`, `npu` →
PyTorch. ROCm and Ascend kernels are open rows, left to their platform claimers.

**Declared fallback.** The forward kernel covers head dims that are a
power-of-two multiple of the 32-wide leaf, up to 1024, because the in-register
halving combine equals the mid-split tree only for a power-of-two leaf count.
Any other head dim falls back to the reference, which is evaluated **on CPU** so
the fallback's bytes stay gold-equal, and the fallback is recorded on the plan
readback (`fallback`, `fallback_reason`) rather than being silent.

## Shapes, dtypes, contiguity

| Tensor | Shape | Notes |
| --- | --- | --- |
| `q` | `[B, H, Sq, D]` | fp32 or bf16 |
| `k` | `[B, H, Sk, D]` | same dtype, head count and head dim as `q` |
| `query_padding_mask` | `[B, Sq]` bool, optional | `True` = valid |
| `key_padding_mask` | `[B, Sk]` bool, optional | `True` = valid |
| `scores` | `[B, H, Sq, Sk]` | input dtype from `forward`, fp32 from `forward_fp32` |

Head counts must be equal: this is joint self-attention, not GQA. Non-contiguous
inputs are made contiguous inside the kernel. fp16 is rejected: the WS1 contract
lists it as optional and it is not declared for this row.

## Accuracy

Byte equality, not tolerance. Every backend reproduces the fp32 CPU gold **byte
for byte** in both dtypes, forward and backward, because the reduction tree is
frozen and shared rather than left to each backend. The test suite contains no
`allclose`, `atol`, `rtol` or ULP threshold.

## Trace metadata

`JointAttnQkGemmPlan.to_dict()` reports the arithmetic policy actually used:
reduction dims and orders, accumulator precision, multiply-add discipline, scale
placement, cast points, the Split-K/atomics/TF32/fast-math flags, batch
invariance, and the selected backend with any declared fallback. The dataclass
rejects any policy outside the contract at construction.

## Example

```python
import torch
from rl_engine.runtime.registry import KernelRegistry

op = KernelRegistry().get_op("joint_attn_qk_gemm", device="cuda")
q = torch.randn(1, 24, 4160, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(1, 24, 4160, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)

scores = op(q, k)                      # [1, 24, 4160, 4160] bf16
scores.backward(torch.randn_like(scores))
print(op.last_plan.to_dict()["forward_reduction_order"])
```

## Tests and benchmarks

```bash
python -m pytest tests/ops/attention/test_joint_attn_qk_gemm.py -q

python tools/validation/operators/check_operator.py --op joint_attn_qk_gemm \
    --candidate pytorch --device cpu  --dtype fp32 --n-heads 24 --check-grad
python tools/validation/operators/check_operator.py --op joint_attn_qk_gemm \
    --candidate cuda   --device cuda --dtype bf16 --n-heads 24 --check-grad

python benchmarks/operators/attention/benchmark_joint_attn_qk_gemm.py
```

## Known limitations

- CUDA only. ROCm, MUSA and Ascend fall back to the PyTorch reference.
- Head dims outside a power-of-two leaf count use the declared CPU fallback.
- The backward materialises the full `[B, H, Sq, Sk]` score gradient, which is
  inherent to the unfused WS1 decomposition; the `joint_attention_fused` row
  exists to recover that memory and must reproduce these bytes exactly.
