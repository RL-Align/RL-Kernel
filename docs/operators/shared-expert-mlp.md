# Shared Expert MLP (P5-5, issue #64)

Shared expert for the DSv4 MoE block: every valid token runs
`fc1 -> one-round SwiGLU -> fc2` once. BF16 frozen weights, backward returns
only `dX` (FP32 accumulator dtype); the shared output stays independent of the
routed path (the combine belongs to P6).

## Fixed math (`oracle-fp32-serial-v1`)

```
z    = x @ w_fc1.T            # BF16 operands, FP32 serial ascending-k, mul-then-add
h    = BF16(SiLU(gate) * up)  # FP32 math, single round; no clamp, no p_s
y    = BF16(h @ w_fc2.T)      # FP32 accumulate, one round
dX   = FP32(dz @ w_fc1)       # dh, dz round BF16 at operator edges
```

Strict kernels reproduce the oracle byte-for-byte on the same device: one lane
owns one output element and reduces serially in ascending k with
`__fmul_rn`/`__fadd_rn` (CUDA) or uncontracted IEEE fp32 arith (Triton), so
there is no cross-lane floating-point reduction and results are
batch/padding invariant.

## Backends

| backend | entry point |
| --- | --- |
| CUDA | `rl_engine.moe.backends.shared_expert:CudaSharedExpertProvider` (`csrc/cuda/moe/shared_expert_mlp.cu`) |
| Triton | `rl_engine.moe.backends.shared_expert:TritonSharedExpertProvider` (`rl_engine/kernels/ops/triton/moe/shared_expert.py`) |

The one-round SwiGLU core runs in shared mode (`p_s = None`, no clamp) and is
the reuse point for P5-2 (#63), which extends the same core with clamp,
route weight, and `dp_s`.

## Acceptance

```bash
python scripts/check_p5.py --provider rl_engine.moe.backends.shared_expert:CudaSharedExpertProvider --device cuda
python scripts/check_p5.py --provider rl_engine.moe.backends.shared_expert:TritonSharedExpertProvider --device cuda
pytest tests/test_shared_expert_mlp.py
python benchmarks/benchmark_shared_expert_mlp.py
```

Fail-closed: non-CUDA input, a missing extension/triton install, or a foreign
numeric profile raises instead of falling back to the oracle.
