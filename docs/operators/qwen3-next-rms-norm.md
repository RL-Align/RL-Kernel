# Qwen3-Next RMSNorm (zero-centred)

## Summary

Qwen3-Next's decoder and final norms store a **zero-centred** weight and compute
`x * rstd * (1 + w)` rather than `x * rstd * w`. The `1 +` is applied in fp32,
after the upcast — folding it into a low-precision weight beforehand rounds the
offset away and silently breaks any bitwise claim.

This operator exists for RFC #428 C1 (Embedding / RMSNorm / residual / final norm
exactness) on the Qwen3-Next rollout-vs-replay path. The Gated DeltaNet block uses
a *different* weight convention and a different cast order; see
[Gated RMSNorm](qwen3-next-rms-norm-gated.md).

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

Claim levels:

- **L1** (prefix-slice and concurrency) for `Qwen3NextRMSNormCudaOp`; padding,
  packing and order are not tested for the CUDA op.
- **L1** (slice, concurrency and padding) for `Qwen3NextRMSNormOp`.
- **L0** for `Qwen3NextRMSNormOp` only (CPU, fp32); no test repeats the CUDA op.
- L2 is not claimed.

The PyTorch reference uses the repo's fixed 32-wide chunked sum
(`shape_invariant_rstd`, introduced in `8ed1693` as a device-agnostic reference).
On B200, over 20 seeds at `H=2048` in bf16, a plain `mean(-1)` broke slice
invariance on 1 of 20 seeds (slices `x[3:5]` and `x[:1]` of 64 rows; which one
failed was not recorded), and the chunked reduction broke on none. That is a recorded
observation, not an assertion.

The CUDA kernel has its own fixed-order reduction (per-thread strided partial sums,
then `block_reduce_sum`). It is not bitwise equal to the PyTorch reference.

Neither is bitwise equal to vLLM. In one-off probes (not committed checks), the
reference differed from every vLLM path tried (eager, inductor-compiled, and eager
under `VLLM_BATCH_INVARIANT=1`) on 36–39 of 40 seeds, `max|diff| <= 1.56e-2`
(bf16, `H=2048`, 512 rows). The difference is attributed to reduction order, but that
has not been isolated. Only the no-residual call was compared; vLLM's
`fused_add_rms_norm` path (every decoder norm except layer 0's input norm) has no
reference here.

The in-kernel offset is exact, not an approximation: `weight_offset=1.0` is bitwise
equal to passing an explicit fp32 `1 + w` weight, and differs from a bf16-folded
`1 + w`, both asserted.

Accuracy tests resolve their tolerances from `tolerance_contract.json`
(`forward_accuracy`, `reduction` × dtype). The bounds in
`tests/check_qwen3_next_norm_providers.py` are provider-gap bounds, not contract
thresholds.

## Performance Notes

The CUDA path reuses the existing `rmsnorm_fwd_kernel` reduction
(`block_reduce_sum` over `choose_threads(H)`), so the offset costs one fp32 add per
element and no extra memory traffic.

## Evidence

![zero-centred RMSNorm vs existing implementations on B200](../usage/evidence/qwen3-next-rms-norm-b200/figure.png)

[`report.json`](../usage/evidence/qwen3-next-rms-norm-b200/report.json) was written by
`scripts/qwen3_next_norm_evidence.py` from a clean tree at `3d0bae7`, on an otherwise idle
B200 (torch 2.13.0+cu130, transformers 5.17.0, vLLM 0.30.0, FlashInfer 0.6.18). Hidden 2048,
BF16.

| | rl-kernel CUDA | PyTorch reference | transformers | vLLM `GemmaRMSNorm`* | FlashInfer `gemma_rmsnorm`* |
|---|---|---|---|---|---|
| rows differing alone vs in a batch, forward / `dx` (of 768) | **0 / 0** | 0 / 12 | 2 / 11 | 2 / n/a | 0 / n/a |
| forward, 65536 rows | 425 µs | 2148 µs | 1449 µs | 1449 µs | **90 µs** |
| backward, 65536 rows | 30.9 ms | **3.0 ms** | 3.1 ms | n/a | n/a |

\* forward only, no backward.

- **Accuracy is the same for every implementation:** forward max error 1.56e-2 against the
  FP64 golden (BF16 output rounding; 99.999% of elements correctly rounded), `dx` and
  `dweight` within 2.8e-3 and 2.1e-3 of their maximum.
- **Row invariance:** only this op and FlashInfer give every row the same bits alone and in a
  batch; FlashInfer has no backward. transformers and vLLM differ on 2 forward rows in 768.
- **The backward is about 10× slower than transformers at 65536 rows.** `dweight` is folded
  over rows in ascending order by `reduce_rows_fp32_left_fold`, one thread per column, so
  that the batch `dweight` equals the in-order sum of single-row contributions bitwise (the
  gradient-invariance contract's singleton-aggregate check). That serial fold over rows is
  the cost; a faster tree fold would not pass that check.

## Tests

```bash
python -m pytest tests/test_qwen3_next_norm.py -v
python scripts/check_operator.py --op qwen3_next_rms_norm --candidate cuda \
    --device cuda --dtype bf16 --check-grad
```

## Known Limitations

- CUDA only; no ROCm, Ascend or Triton backend.
- Not bitwise against vLLM (see Accuracy). An L2 claim needs a single source of
  truth for the forward on both sides, per RFC #428 §0 item 1.
- The gated pair (`Qwen3NextRMSNormGatedOp`, `Qwen3NextRMSNormGatedHFOp`) is
  documented with its CUDA kernel on [Gated RMSNorm](qwen3-next-rms-norm-gated.md).
- Measured on sm_100 (B200). Per RFC #428 §2.2 no claim carries across
  H100/H200/B100/B200.
