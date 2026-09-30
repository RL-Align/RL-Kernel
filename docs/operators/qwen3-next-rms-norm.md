# Qwen3-Next RMSNorm (zero-centred)

## Summary

Qwen3-Next's decoder and final norms store a **zero-centred** weight and compute
`x * rstd * (1 + w)` rather than `x * rstd * w`. The `1 +` is applied in fp32,
after the upcast — folding it into a low-precision weight beforehand rounds the
offset away and silently breaks any bitwise claim.

This operator exists for RFC #428 C1 (Embedding / RMSNorm / residual / final norm
exactness) on the Qwen3-Next rollout-vs-replay path. The Gated DeltaNet block uses
a *different* weight convention and a different cast order; it is a separate
operator.

Upstream references: `transformers` `Qwen3NextRMSNorm`, and vLLM's `GemmaRMSNorm`,
which `vllm/model_executor/models/qwen3_next.py` aliases as `Qwen3NextRMSNorm`.

## Entry Point

```python
from rl_engine.kernels.ops.pytorch.norm.qwen3_next_rms_norm import Qwen3NextRMSNormOp
from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp, rmsnorm_cuda

y = Qwen3NextRMSNormOp().forward(x, weight, eps=1e-6)       # reference
y = Qwen3NextRMSNormCudaOp().forward(x, weight, eps=1e-6)   # CUDA

# The offset is a kernel parameter, not a pre-pass:
y = rmsnorm_cuda(x, weight, eps=1e-6, weight_offset=1.0)
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `Qwen3NextRMSNormCudaOp` | `rl_engine._C.rmsnorm_forward` (`weight_offset=1.0`) | Supported |
| ROCm | — | — | Not implemented |
| PyTorch fallback | `Qwen3NextRMSNormOp` | — | Supported (WS1 gold) |

`Qwen3NextRMSNormOp` subclasses `NativeRMSNormOp` and overrides only
`weight_offset`; the base applies it in fp32.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[..., H]` | fp32 / bf16 / fp16 | CUDA path requires contiguous |
| `weight` | `[H]` | matches `x` | zero-centred (upstream inits to zeros) |
| `eps` | scalar | float | inside the sqrt; `1e-6` for Qwen3-Next |

## Dispatch Behavior

Registered as the `qwen3_next_rms_norm` gtest operator. On CUDA the registry
prefers `Qwen3NextRMSNormCudaOp`; every other platform resolves to the PyTorch
reference. The CUDA op validates the compiled symbols in `__init__`, so on a build
without the extension construction raises and the registry falls through to the
reference rather than handing out an op that fails at call time.

## Accuracy

Claim level: **L0 repeatable, L1 batch-invariant**. L2 is not claimed.

The reduction is the repo's fixed 32-wide chunked sum
(`shape_invariant_rstd`), which is what makes a row's result independent of the
batch layout. It deliberately differs from upstream's `mean(-1)`: measured on B200,
over 20 seeds at `H=2048` in bf16, a plain `mean(-1)` broke slice invariance on 1
of 20 while the chunked reduction broke on 0 of 20.

The cost is that the decoder norm is **not** bitwise equal to stock vLLM — 7
elements of 1048576 differ, `max|diff| = 1.56e-2` in bf16 at `H=2048`. That gap is
inherent: matching stock vLLM bitwise would mean adopting a reduction that is not
itself batch-invariant, i.e. trading L1 for L2.

The in-kernel offset is exact, not an approximation: `weight_offset=1.0` is bitwise
equal to passing an explicit fp32 `1 + w` weight, and differs from a bf16-folded
`1 + w`, both asserted.

Tolerances come from `tolerance_contract.json` (`reduction` × dtype); no private
thresholds.

## Performance Notes

The CUDA path reuses the existing `rmsnorm_fwd_kernel` reduction
(`block_reduce_sum` over `choose_threads(H)`), so the offset costs one fp32 add per
element and no extra memory traffic.

```bash
python scripts/check_operator.py --op qwen3_next_rms_norm --candidate cuda \
    --device cuda --dtype bf16 --check-grad
```

## Tests

```bash
python -m pytest tests/test_qwen3_next_norm.py -v
```

## Known Limitations

- CUDA only; no ROCm, Ascend or Triton backend.
- Not bitwise against stock vLLM (see Accuracy); an L2 claim needs the strict
  provider on both sides, per RFC #428 §1 item 1.
- Measured on sm_100 (B200). Per RFC #428 §2.2 no claim carries across
  H100/H200/B100/B200.
