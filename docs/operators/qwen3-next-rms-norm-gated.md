# Qwen3-Next Gated RMSNorm

## Summary

The norm inside Qwen3-Next's Gated DeltaNet block:

```
out = x * rstd * weight * silu(gate)
```

It is a different operator from the decoder norm, not a variant of it: the weight is
plain rather than zero-centred (upstream initializes it to ones), it takes a second
input tensor, and — critically — `transformers` and vLLM disagree about where the
weight multiply happens. See [Qwen3-Next RMSNorm](qwen3-next-rms-norm.md) for the
decoder-norm convention.

Added for RFC #428 C1 on the Qwen3-Next rollout-vs-replay path.

## Entry Point

```python
from rl_engine.kernels.ops.pytorch.norm.qwen3_next_rms_norm import (
    Qwen3NextRMSNormGatedOp,      # strict / vLLM convention
    Qwen3NextRMSNormGatedHFOp,    # transformers convention, kept as a witness
)
from rl_engine.kernels.ops.cuda.norm.rmsnorm import (
    Qwen3NextRMSNormGatedCudaOp,
    rmsnorm_gated_cuda,
)

y = Qwen3NextRMSNormGatedCudaOp().forward(x, weight, gate, eps=1e-6)
y = rmsnorm_gated_cuda(x, weight, gate, eps=1e-6, activation="sigmoid")
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `Qwen3NextRMSNormGatedCudaOp` | `rl_engine._C.rmsnorm_gated_forward` | Supported |
| ROCm | — | — | Not implemented |
| PyTorch fallback | `Qwen3NextRMSNormGatedOp` | — | Supported (WS1 gold) |

The CUDA op is deliberately **not** a subclass of `RMSNormCudaOp`: it takes an extra
required tensor, so it cannot stand in for one.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[..., H]` | fp32 / bf16 / fp16 | contiguous on the CUDA path |
| `weight` | `[H]` | matches `x` | plain, NOT zero-centred |
| `gate` | same as `x` | same as `x` | shape and dtype are enforced |
| `eps` | scalar | float | `1e-6` for Qwen3-Next |
| `activation` | — | `"silu"`/`"swish"`/`"sigmoid"` | anything else is rejected |

Only `norm_before_gate=True` and `group_size=None` are implemented — the single
configuration vLLM's GDN block constructs. Other configurations fail closed rather
than being approximated (RFC #428 §6 item 7).

## Dispatch Behavior

Registered as `rms_norm_gated`. CUDA prefers the kernel; every other platform
resolves to the PyTorch reference. `__init__` validates the compiled symbols, so on a
build without the extension the registry falls through instead of returning an op
that raises at call time.

## Accuracy

Claim level: **L0 repeatable, L1 batch-invariant**. L2 is not claimed.

`transformers` casts the normalized value back to the input dtype *before* the weight
multiply; vLLM keeps it in fp32. On B200 / bf16 / `head_v_dim=128` the two differ in
35% of elements with `max|diff| = 6.25e-2`, and isolating the cast order alone
reproduces the gap — so the cast order dominates, not the reduction order. Because
RFC #428 measures L2 against vLLM rollout, the fp32 multiply is the strict default;
`Qwen3NextRMSNormGatedHFOp` keeps the other convention as a tested witness.

"Bitwise equal to vLLM" is undefined until a provider is named: vLLM's own
`forward_native` and `forward_cuda` disagreed on 21 of 40 seeds (worst 1.56e-2 in
bf16; ~36% of elements at fp32 ULP in fp32). What this reproduces is the convention;
the residual is the reduction tree.

`rstd` is bitwise identical to the ungated kernel's for the same `x` — asserted, so a
gate leaking into the statistic would fail.

Backward is assembled from deterministic pieces: `dx` from a row-local kernel,
`dweight` from fp32 row contributions reduced by the ascending-row left fold, and
`dgate` elementwise in fp32 with no reduction.

## Performance Notes

Reuses the existing `block_reduce_sum` / `choose_threads(H)` reduction, so the gate
costs one extra load and one fp32 multiply per element.

```bash
python scripts/check_operator.py --op rms_norm_gated --candidate cuda \
    --device cuda --dtype bf16 --check-grad
```

## Tests

```bash
python -m pytest tests/test_qwen3_next_norm.py -v
# imports real vLLM, so it is not collected by default:
python -m pytest tests/check_qwen3_next_norm_providers.py -v
```

## Known Limitations

- CUDA only; no ROCm, Ascend or Triton backend.
- `norm_before_gate=False` and grouped norms are not implemented.
- Not bitwise against either vLLM path (see Accuracy).
- Measured on sm_100 (B200); RFC #428 §2.2 forbids carrying the claim across
  H100/H200/B100/B200.
