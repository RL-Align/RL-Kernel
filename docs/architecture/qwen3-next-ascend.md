# Qwen3-Next Ascend adaptation

These reference primitives prepare a normalized-input interface for future
Ascend work. They build on the upstream GDN and convolution references extracted
in [PR #659](https://github.com/RL-Align/RL-Kernel/pull/659) and the shared state
contract from [PR #660](https://github.com/RL-Align/RL-Kernel/pull/660).

The CUDA full-model work is tracked separately by
[RFC #428](https://github.com/RL-Align/RL-Kernel/issues/428) and its
`test-qwennext` PR stack. This change does not replace that work or establish
Ascend model support. New registrations live in `rl_engine/runtime/registry.py`;
the upstream legacy registry shim is preserved.

## Current scope

Three normalized-input PyTorch references are available through `get_op`:

| Operator | Canonical implementation | Input / state |
| --- | --- | --- |
| `sigmoid_gate` | `rl_engine/reference/activation/sigmoid_gate.py` | `x * sigmoid(gate)` |
| `causal_conv1d` | `rl_engine/reference/linear_attn/causal_conv1d_sequence.py` | `[B,T,C]`, cache `[B,K-1,C]` |
| `gated_deltanet` | `rl_engine/reference/linear_attn/gated_deltanet.py` | q/k `[B,Hk,T,Dk]`, v `[B,Hv,T,Dv]`, state `[B,Hv,Dk,Dv]` |

Legacy `rl_engine.kernels.ops.pytorch` imports remain compatibility aliases.
All platforms register the reference candidates; this does not qualify a vendor
kernel or prove device-level consistency. No new AscendC kernel, complete
Qwen3-Next model, MoE implementation, TP/CP path or NPU performance qualification
is delivered by this reference change.

## Numerical boundary

The normalized GDN API takes beta **logits**, but g is already **log decay**:
`g = -exp(A_log) * softplus(a + dt_bias)`. Each step performs:

1. Normalize q/k with `x * rsqrt(sum(x*x) + 1e-6)` when enabled.
2. Decay the FP32 state with `exp(g)`; g=None means no decay.
3. Compute the delta write using sigmoid(beta) and the decayed state.
4. Read the updated state with q scaled by `Dk**-0.5` unless explicitly overridden.

Value head h maps to key head `h // (Hv/Hk)`. Heads and dimensions must be
positive and Hv must be divisible by Hk. Caller state is never updated in place.
Empty sequences return empty output and preserve the recurrent state.

The newly merged upstream reference and `GDNContract` define a separate
`vllm_decode_v1` profile: raw a/b/A_log/dt_bias, V-major paged state, fixed-chunk
reductions, and explicit per-step state rounding. See [the GDN contract](../contracts/gdn.md).
The normalized reference does not claim that profile's bitwise arithmetic.
Adapters must convert gate inputs and state layouts explicitly. Regression tests
compare output and final state against that upstream reference within tolerance,
using unequal key/value dimensions so an axis swap cannot be hidden.

The shared sigmoid preserves the composed exp/add/div forward while using an
analytical backward, avoiding overflow-induced NaN gradients for finite extreme
negative gates. Batch slice/padding checks describe the tested host; they do not
establish bitwise parity across ISAs, devices, PyTorch builds or provider profiles.

## Optimization and validation

Q/K normalization, FP32 conversions and grouped-head expansion run once per
prefill rather than once per token. The recurrence remains sequential and
allocates expanded Q/K, so long-sequence production prefill still needs a
qualified device implementation. Convolution copies only the trailing K-1 tokens
into a compact cache instead of keeping the entire prefill storage alive.

A local CPU microbenchmark (one thread, B=2, Hk=2, Hv=4, T=64, Dk=32, Dv=24,
12 samples after 3 warmups) measured the original reference at median 12.18 ms
and the revised implementation at 4.74 ms, approximately 2.57x. This is a local
small-shape observation with corrected numerical semantics, not an NPU result
or a general performance guarantee.

The final targeted run passed 393 tests, including registry, import compatibility,
Attention/LogP and tolerance contracts. Changed Python files pass Black, isort,
Ruff and Flake8. A small CPU prefill/decode benchmark smoke run also completed.
No NPU kernel or complete model run was performed.

Run focused reference checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/ops/activation/test_sigmoid_gate.py \
  tests/reference/linear_attn/ tests/ops/linear_attn/ \
  tests/runtime/test_kernel_registry.py
```

The main CI includes these reference tests. The benchmark accepts semicolon
separated prefill configurations and uses q|k|v convolution channels only; the
z gate bypasses convolution:

```bash
python -m benchmarks.benchmark_qwen3_next_ops --device cpu \
  --configs "1,2" --batches 1 --warmup 0 --iters 1
```

## Next milestones

First qualify sigmoid gate and short-convolution kernels on Ascend. Then add a
GDN decode kernel with an explicit numerical profile, followed by prefill and
backward. Full model assembly must cover gate projections, partial RoPE,
post-GDN gated normalization, MoE and weight mapping before model-level claims.
Reuse the shared numerical contracts and existing acceptance gates.
