# MiniMax-H3 Per-Head Q/K RMSNorm

## Summary

`h3_qk_rmsnorm_d128` (RFC #420) is the RMSNorm of MiniMax-H3's queries and keys:
`attn.norm_q` and `attn.norm_k`, in the 50 transformer blocks and the 2 token-refiner blocks. Both are
`nn.RMSNorm(128, eps=1e-5)` over the last dimension of the per-head tensors `(B, S, 56, 128)`, each
with one affine weight of 128 entries shared by the 56 heads:

```text
y = x * rsqrt(mean(x ** 2, dim=-1) + eps) * weight
```

The row adds **no native code**. It pins the H3 shape, dtype and `eps` contract on the kernel of the
merged [`h3_rmsnorm`](h3-rmsnorm.md) row, without its AdaLN modulation.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("h3_qk_rmsnorm_d128", device="cuda")
q = op(q, norm_q_weight)   # (..., heads, 128), BF16
```

## Backends

| Backend | Wrapper | Native symbols | Status |
| --- | --- | --- | --- |
| CUDA (SM80+, validated on SM100) | `rl_engine.backends.cuda.model_specific.minimax_h3.qk_rmsnorm.H3QKRMSNormCudaOp` | `rl_engine._C.h3_rmsnorm_{forward,backward}` | Forward bitwise equal to `nn.RMSNorm(128)`; backward not |
| PyTorch reference | `rl_engine.reference.minimax_h3.qk_rmsnorm.NativeH3QKRMSNormOp` | n/a | `forward`: provider path (`F.rms_norm`); `forward_fp32`: FP64 golden, returned in FP32 |
| ROCm | n/a | n/a | Falls back to the PyTorch reference |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `(..., H, 128)` | bf16 / fp16 / fp32 | `H` divides 56: the full heads, or the `56 / tp` heads of one TP rank (`tp` in 1, 2, 4, 8) |
| `weight` | `(128,)` | `x.dtype` | On `x`'s device |
| `eps` | scalar | - | Pinned to `1e-5` |

Any other head dimension, head count or `eps`, a dtype mismatch, and a CPU tensor passed to the CUDA
wrapper all raise.

## Numerics

- **Statistics** follow contract `h3-rmsnorm-v1`, described in [`h3-rmsnorm`](h3-rmsnorm.md#numerics).
  Every head of every position is its own row, so a TP rank's head shard is the same slice of the
  full result.
- **Forward.** Bitwise equal to `F.rms_norm` at 1, 257, 925 and 4097 rows of 56 heads (tables below).
- **Backward.** `dx` is row-local. `dweight` is folded over fixed 256-row tiles in ascending order in
  FP32, with one cast. The bits of `dx` and `dweight` differ from the provider's in a small number of
  elements (below).

## Existing implementations (RFC #420 reuse rule)

B200, BF16, `(S, 56, 128)`, measured with `torch 2.13.0+cu130`. Errors are `max|err| / max|ref|` against the same
computation in FP64. Latency is the median of forward and of forward + backward on an otherwise
idle GPU of a shared node. Batch invariance is bitwise and covers three checks: rows computed alone
against full batches, the full batch against sub-batches that cover every row, and a batch-size sweep.

| Implementation | Batch-invariant | S = 257: fwd err / dx err / dw err / fwd + bwd | S = 925 | S = 4097 |
|---|---|---|---|---|
| diffusers nn.RMSNorm(128) (F.rms_norm) | yes | 2.5e-3 / 2.1e-3 / 1.7e-3 / 525 µs | 2.5e-3 / 2.1e-3 / 2.7e-3 / 516 µs | 2.2e-3 / 1.9e-3 / 4.2e-3 / 706 µs |
| rl-kernel generic RMSNormCudaOp (the Qwen QK-norm operator) | yes | 2.5e-3 / 2.1e-3 / 1.7e-3 / 3345 µs | 2.5e-3 / 2.1e-3 / 2.7e-3 / 10964 µs | 2.2e-3 / 1.9e-3 / 2.4e-3 / 98229 µs |
| rl-kernel generic RMSNormTritonOp | yes | 2.5e-3 / 2.1e-3 / 1.7e-3 / 3370 µs | 2.5e-3 / 2.1e-3 / 2.7e-3 / 10973 µs | 2.2e-3 / 1.9e-3 / 2.4e-3 / 98733 µs |
| TE 2.20.2 RMSNorm | yes | 2.5e-3 / 2.1e-3 / n/a / 830 µs | 2.5e-3 / 2.1e-3 / n/a / 808 µs | 2.2e-3 / 1.9e-3 / n/a / 820 µs |
| Liger 0.8.4 RMSNorm | **no** | 5.2e-3 / 2.1e-3 / 3.0e-3 / 846 µs | 5.1e-3 / 2.1e-3 / 3.2e-3 / 863 µs | 5.0e-3 / 1.9e-3 / 2.7e-3 / 850 µs |
| SGLang 0.5.21 fused_inplace_qknorm (q + k, forward only) | yes | 2.5e-3 / n/a / n/a / 27 µs (fwd) | 2.5e-3 / n/a / n/a / 37 µs (fwd) | 2.2e-3 / n/a / n/a / 113 µs (fwd) |
| rl-kernel H3QKRMSNormCudaOp | yes | 2.5e-3 / 2.1e-3 / 1.7e-3 / 661 µs | 2.5e-3 / 2.1e-3 / 2.7e-3 / 467 µs | 2.2e-3 / 1.9e-3 / 2.4e-3 / 705 µs |

TE's `dw` is not compared, and SGLang's kernel has no backward; `(fwd)` marks a forward-only time.

**The provider's own `F.rms_norm` passes the batch-invariance checks.** Its forward bytes are this
row's, and so are SGLang's fused kernel's. This row does not replace them for batch invariance or
for forward accuracy. It reuses the merged `h3_rmsnorm` kernel, so no native code is added, and the
difference is in the backward:

- `dweight` is about 1.8 times more accurate at 4097 rows (2.4e-3 against 4.2e-3); the two are equal
  at 257 and 925 rows.
- `dx` is the same to the printed precision.

**Speed.** Forward + backward against the provider is a tie at 4097 rows (705 against 706 µs) and
slower at 257 (661 against 525). At 925 it depends on the run (467 against 516 here, 645 against 522
in an earlier run of the same kernel at `7510cf18`): the medians move by tens of percent between
runs at that size, so no difference is claimed. The generic RMSNorm operators of this repository take
5 to 140 times longer for forward + backward. SGLang's fused forward (q and k together, copies
included) takes 113 µs at 4097 rows against 170 µs for one forward of this row; it has no backward.

### Direct bitwise comparison

Equal errors do not show equal bits, so every candidate is compared byte for byte with the provider
and with this kernel, on identical inputs (`tools/validation/models/h3_bitwise_reuse.py`). Differing
elements against diffusers at S = 1 / 257 / 925 / 4097:

| Candidate | output | `dx` | `dw` (128 entries) |
|---|---|---|---|
| rl-kernel generic RMSNormCudaOp (the Qwen QK-norm operator) | 0 / 2 / 21 / 76 | 0 / 46 / 154 / 638 | 0 / 0 / 0 / 57 |
| rl-kernel generic RMSNormTritonOp | 0 / 3 / 18 / 75 | 0 / 34 / 135 / 601 | 0 / 0 / 0 / 57 |
| TE 2.20.2 RMSNorm | 0 / 2 / 16 / 63 | 0 / 30 / 121 / 493 | n/a |
| Liger 0.8.4 RMSNorm | 1911 / 482170 / 1736089 / 7691553 | 0 / 36 / 142 / 616 | 69 / 73 / 58 / 75 |
| SGLang 0.5.21 fused_inplace_qknorm (q + k, forward only) | 0 / 0 / 0 / 0 | n/a | n/a |
| rl-kernel H3QKRMSNormCudaOp | 0 / 0 / 0 / 0 | 0 / 27 / 122 / 560 | 0 / 0 / 0 / 56 |

The output has 7168, 1842176, 6630400 and 29367296 elements at the four sizes. This row's forward is
bitwise the provider's at every size measured (1, 257, 925 and 4097). Its gradients are bitwise the
provider's at S = 1 (7168 elements) only.

```bash
python tools/validation/models/h3_prior_art.py --op qk_norm --out qk_norm.json
python tools/validation/models/h3_bitwise_reuse.py --op qk_norm --out qk_norm_bitwise.json
```

The reports were written from a clean tree; they are not part of this diff (links in the PR
description).

## Tests

```bash
python -m pytest tests/models/minimax_h3/test_h3_qk_rmsnorm.py -v                  # operator
export RL_KERNEL_H3_WEIGHTS=<dir written by tools/weights/prepare_h3_weights.py>
python -m pytest tests/models/minimax_h3/test_h3_one_block.py \
    tests/models/minimax_h3/test_h3_token_refiner.py -v                            # in the block graphs
python tools/validation/operators/check_operator.py --op h3_qk_rmsnorm_d128 --candidate cuda --device cuda \
    --dtype bf16 --batch 2 --seq 257 --check-grad
```

## Known Limitations

- The backward is not bitwise equal to the provider's (table above).
- The norm weights are unsharded. On a TP grid each rank applies them to its own heads and the weight
  gradients of the ranks are not reduced by this row (see the limits of
  [`h3-parallel`](h3-parallel.md)).
- There is no ROCm kernel. ROCm dispatches the PyTorch reference.
- Validated on B200. The Hopper (SM90) run has not been done.
