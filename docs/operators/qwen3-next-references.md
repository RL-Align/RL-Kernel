# Qwen3-Next reference primitives

These are PyTorch references for sigmoid gating, causal depthwise convolution
and normalized-input Gated DeltaNet. They provide an experimental baseline for
future Ascend adaptation. No vendor kernel or complete model is qualified here.

## Entry points

```python
import torch
from rl_engine.runtime.registry import kernel_registry

gate = kernel_registry.get_op("sigmoid_gate", device="cpu")
x = torch.ones(1, 2, 8)
y = gate(x, torch.zeros_like(x))

conv = kernel_registry.get_op("causal_conv1d", device="cpu")
y, cache = conv(x, torch.ones(8, 4), return_cache=True)

gdn = kernel_registry.get_op("gated_deltanet", device="cpu")
q = torch.ones(1, 1, 2, 4)
k = torch.ones_like(q)
v = torch.ones(1, 2, 2, 3)
beta_logits = torch.zeros(1, 2, 2)
log_decay = -torch.ones_like(beta_logits)
out, state = gdn(q, k, v, beta_logits, log_decay, return_final_state=True)
```

`forward` returns the input dtype; `forward_fp32` returns FP32. Supported input
dtypes are FP16, BF16 and FP32. Recurrent GDN state must be FP32.

## Tensor and numerical boundary

| Primitive | Inputs | Output and continuation state |
| --- | --- | --- |
| Sigmoid gate | x and gate: identical shape, dtype and device | x * sigmoid(gate), same shape |
| Causal convolution | x `[B,T,C]`, weight `[C,K]`, optional bias `[C]`, cache `[B,K-1,C]` | y `[B,T,C]`, compact cache `[B,K-1,C]` |
| GDN | q/k `[B,Hk,T,Dk]`, v `[B,Hv,T,Dv]`, beta logits and log decay `[B,Hv,T]` | out `[B,Hv,T,Dv]`, FP32 state `[B,Hv,Dk,Dv]` |

Convolution uses oldest-to-newest cross-correlation taps and has no fused
activation. GDN requires Hv divisible by Hk, optionally L2-normalizes q/k with
epsilon 1e-6, applies exp(log_decay), and reads the state after its delta write.
Default query scale is Dk**-0.5. `decode_step` accepts exactly one token.

The GDN input `g` is already log decay, whereas beta is a raw sigmoid logit.
This differs from the raw a/b/A_log/dt_bias inputs of the upstream
[GDN contract](../contracts/gdn.md). Its state layout also differs from the
upstream V-major paged state. Explicit adapters are required. Agreement tests
use tolerances; these APIs do not promise provider-level bitwise equivalence.

## Validation and limitations

Tests cover independent FP64 recurrence and gradients, comparison with the
upstream recurrent reference, batch slice/padding, prefill/decode continuation,
extreme gate gradients, compact cache storage and legacy imports. The main CPU
CI includes the new tests. The extreme-gate test also runs on CUDA when available.

The GDN implementation keeps a sequential token loop. Hardware support,
cross-device equality, complete model assembly, MoE and TP/CP qualification
require further work. See the [adaptation design](../architecture/qwen3-next-ascend.md)
for the benchmark command, execution profile and remaining milestones.
